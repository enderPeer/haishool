"""Run unselected cosmic seeds through fixed prebiotic chemical experiments."""
from __future__ import annotations

import argparse
from dataclasses import asdict, fields, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import time

import torch

from . import __version__
from .assemblies import Assemblies, AssemblyConfig, folded_contacts
from .chain import FORMAT_VERSION, SOURCE_PATHS, ChainConfig, build_chain, stream_seed
from .chemistry import ChemicalConfig, ChemicalEngine, ChemicalState, ELEMENTS, SPECIES
from .compartments import CompartmentConfig, Compartments
from .protocol import digest, experiment_identity, require_source, source_identity, write_json


PROFILES = ("normal", "dark", "no_template", "no_catalysis", "no_compartments")


class SparsePhase:
    """Run sparse molecular graphs on CPU around bulk chemistry on its device.

    The graph laws, RNGs and chemical ordering are unchanged. A packed float64
    state crosses each device boundary once per step, instead of reading and
    mutating individual GPU scalars for every graph event. The CPU mirror at
    the end of a step supplies the next step's catalysts and permeability.
    Callers must create a new phase after externally replacing state or graphs.
    GPU/CPU floating-point rounding can produce different eventual trajectories.
    """
    def __init__(self, engine, *, force_bridge=False):
        self.engine = engine
        self.cpu_engine = ChemicalEngine(engine.config, "cpu")
        self.bridge = engine.device.type != "cpu" or force_bridge
        self.cpu_state = None
        self.layout = None

    def _download(self, state):
        tensors = []
        layout = []
        offset = 0
        for field in fields(ChemicalState):
            value = getattr(state, field.name)
            if isinstance(value, torch.Tensor):
                if value.dtype != torch.float64:
                    raise TypeError("Sparse bridge requires float64 ChemicalState tensors")
                n = value.numel()
                tensors.append(value.reshape(-1))
                layout.append((field.name, value.shape, offset, n))
                offset += n
        self.layout = layout
        packed = torch.cat(tensors).detach().to("cpu")
        data = {name: packed[start:start+n].reshape(shape)
                for name, shape, start, n in layout}
        for field in fields(ChemicalState):
            if field.name not in data:
                data[field.name] = getattr(state, field.name)
        self.cpu_state = ChemicalState(**data)

    def _upload(self, state):
        packed = torch.cat([getattr(self.cpu_state, name).reshape(-1)
                            for name, _, _, _ in self.layout]).to(self.engine.device)
        for name, shape, start, n in self.layout:
            setattr(state, name, packed[start:start+n].reshape(shape))
        for field in fields(ChemicalState):
            if not isinstance(getattr(self.cpu_state, field.name), torch.Tensor):
                setattr(state, field.name, getattr(self.cpu_state, field.name))

    @torch.no_grad()
    def step(self, state, pool, compartments, *, illumination=1.0, temperature_k=None):
        if self.bridge:
            if self.cpu_state is None:
                self._download(state)
            cpu = self.cpu_state
        else:
            cpu = self.cpu_state = state
        mask = pool.compartment_mask(cpu)
        permeability = torch.where(mask, .1, 1.0).to(torch.float64) * compartments.permeability(cpu)
        catalyst = torch.ones_like(cpu.heat_j)
        if pool.profile != "no_catalysis":
            for row in pool.chains:
                b, (x, y) = row["batch"], row["cell"]
                catalyst[b, x, y] += pool.config.catalytic_gain * folded_contacts(row["sequence"])
        # Upload the two grid fields together. Seeded sparse event RNGs stay
        # inside their existing batch-wide graph objects, including per-planet RNGs.
        if self.bridge:
            factors = torch.stack((permeability, catalyst)).to(self.engine.device)
            permeability, catalyst = factors[0], factors[1]
        self.engine.step(state, illumination=illumination, catalyst_multiplier=catalyst,
                         permeability=permeability, temperature_k=temperature_k)
        if self.bridge:
            self._download(state)
            cpu = self.cpu_state
        pool.step(cpu)
        compartments.step(cpu, self.cpu_engine)
        el = (self.cpu_engine.elemental_totals(cpu) + pool.elemental_totals("cpu")
              + compartments.elemental_totals("cpu"))
        energy = (self.cpu_engine.energy_total(cpu) + pool.energy_total("cpu")
                  + compartments.energy_total("cpu"))
        if self.bridge:
            self._upload(state)
            # One small upload for both ledgers; never one CUDA operation per chain.
            ledgers = torch.cat((el, energy[:, None]), dim=1).to(self.engine.device)
            el, energy = ledgers[:, :-1], ledgers[:, -1]
        return mask, el, energy


def utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def initialize_batch(planets, cfg, device):
    """Independent per-planet initialization; changing a batch does not redraw a planet."""
    engine = ChemicalEngine(cfg, device)
    rows = []
    for row in planets:
        if not math.isfinite(row["temperature_k"]) or row["temperature_k"] <= 0:
            raise ValueError("Planet temperature must be finite and positive; no repair is applied")
        if not math.isfinite(row["flux_earth"]) or row["flux_earth"] < 0:
            raise ValueError("Planet flux must be finite and nonnegative")
        # atom_inventory counts numerical quanta, not individual literal atoms.
        quantum_umol = row.get("atom_quantum_mol", 1e-6) * 1e6
        budgets = {e: row["atom_inventory"].get(e, 0) * quantum_umol for e in ELEMENTS}
        single = ChemicalEngine(replace(cfg, batch_size=1), device)
        rows.append(single.initialize_from_elements(row["chemistry_seed"], budgets,
                                                    temperature_k=row["temperature_k"]))
    data = {}
    for field in fields(ChemicalState):
        values = [getattr(s, field.name) for s in rows]
        data[field.name] = torch.cat(values, dim=0) if isinstance(values[0], torch.Tensor) else values[0]
    return engine, ChemicalState(**data)


def validate_cached_chains(cached, requested_seeds, config, frozen):
    """Validate immutable upstream inputs without choosing outcomes or redrawing a planet.

    The cache's repeat is metadata for its original parcel seed. The current
    run derives chemical microstate seeds again from its declared repeat.
    """
    if isinstance(cached, dict):
        cached = cached.get("chains", [cached])
    if not isinstance(cached, list) or not all(isinstance(row, dict) for row in cached):
        raise ValueError("Cached chains must be a list of chain records")
    if len(set(requested_seeds)) != len(requested_seeds):
        raise ValueError("Requested cosmic seeds must be unique")
    by_seed = {}
    expected_laws = {path: frozen["files"][path] for path in SOURCE_PATHS}
    for row in cached:
        seed = row.get("cosmic_seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("Cached cosmic seed must be a nonnegative integer")
        if seed in by_seed:
            raise ValueError(f"Duplicate cached cosmic seed {seed}")
        if row.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"Cached format mismatch for seed {seed}")
        if row.get("config") != asdict(config):
            raise ValueError(f"Cached chain configuration mismatch for seed {seed}")
        if row.get("source_hashes") != expected_laws:
            raise ValueError(f"Cached upstream laws mismatch for seed {seed}")
        if not isinstance(row.get("planets"), list) or not isinstance(row.get("errors"), list):
            raise ValueError(f"Cached chain fields invalid for seed {seed}")
        by_seed[seed] = row
    missing = set(requested_seeds) - set(by_seed)
    if missing:
        raise ValueError(f"Cached inputs missing requested seeds {sorted(missing)}")
    return [by_seed[seed] for seed in requested_seeds]


def run(args):
    out = Path(args.out).resolve()
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        raise ValueError("An experiment already exists here; use a fresh output folder")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    frozen = source_identity()
    started = time.monotonic()
    cosmic_cfg = ChainConfig(sample_atoms=args.samples)
    cfg_values = {"chain": asdict(cosmic_cfg), "grid": args.grid, "steps": args.steps,
                  "repeat": args.repeat, "profile": args.profile,
                  "parcel_rule": "elemental sample quanta; no useful-species floor or organic input",
                  "photon_input_j_per_cell_s_at_earth_flux": .05,
                  "bath": "fixed upstream temperature, signed exchanged energy booked",
                  "assembly": asdict(AssemblyConfig()), "compartments": asdict(CompartmentConfig())}
    manifest = {"version": __version__, "schema": "life10-run-v1", "status": "building",
                "source": frozen, "config": cfg_values, "seeds": args.seeds,
                "device": args.device, "python": platform.python_version(), "torch": torch.__version__,
                "started_utc": utc(), "seed_selection": "all requested seeds, all formed planets",
                "interpretation": "Phenomenological precursor/peptide model. No abiogenesis or life claim follows from assembly alone."}
    write_json(manifest_path, manifest)
    if args.chains:
        cached = json.loads(Path(args.chains).read_text(encoding="utf-8-sig"))
        chains = validate_cached_chains(cached, args.seeds, cosmic_cfg, frozen)
    else:
        if len(set(args.seeds)) != len(args.seeds):
            raise ValueError("Requested cosmic seeds must be unique")
        chains = [build_chain(s, args.repeat, cosmic_cfg) for s in args.seeds]
    require_source(frozen)
    # Effective input state is part of the identity, whether cached or freshly built.
    manifest["chain_inputs_sha256"] = digest(chains)
    manifest["experiment_sha256"] = experiment_identity(
        cfg_values, args.seeds, frozen, [args.profile],
        chain_inputs_sha256=manifest["chain_inputs_sha256"])
    write_json(out / "chains.json", chains)
    planets = []
    for chain in chains:
        for planet in chain["planets"]:
            row = dict(planet)
            row["cosmic_seed"] = chain["cosmic_seed"]
            row["chemistry_seed"] = stream_seed(chain["cosmic_seed"], args.repeat, row["id"])
            planets.append(row)
    manifest.update(status="running", systems=len(chains), planets=len(planets),
                    chain_outcomes={str(r["cosmic_seed"]): r["outcome"] for r in chains},
                    chain_errors=[{"seed": r["cosmic_seed"], "errors": r["errors"]} for r in chains if r["errors"]])
    write_json(manifest_path, manifest)
    if not planets:
        require_source(frozen)
        manifest.update(status="complete", completed_utc=utc(), executed_steps=0,
                        stop_reason="no_formed_planets", wall_seconds=time.monotonic() - started)
        write_json(out / "summary.json", {"systems": len(chains), "planets": 0,
                                           "outcomes": manifest["chain_outcomes"], "life_verdict": "not_established"})
        write_json(manifest_path, manifest)
        print(json.dumps({"status": "complete", "planets": 0, "outcomes": manifest["chain_outcomes"]}), flush=True)
        return
    chem_cfg = ChemicalConfig(grid_size=args.grid, batch_size=len(planets), photon_input_j=.05,
                              thermostat=True)
    engine, state = initialize_batch(planets, chem_cfg, args.device)
    planet_seeds = [row["chemistry_seed"] for row in planets]
    pool = Assemblies(len(planets), args.grid, planet_seeds,
                      profile=args.profile)
    compartments = Compartments(len(planets), args.grid, planet_seeds,
                                profile=args.profile)
    initial_elements = engine.elemental_totals(state).clone()
    initial_energy = engine.energy_total(state).clone()
    light = torch.tensor([p["flux_earth"] for p in planets], dtype=torch.float64, device=args.device)
    temperatures = torch.tensor([p["temperature_k"] for p in planets],
                                dtype=torch.float64, device=args.device)
    if args.profile == "dark":
        light.zero_()
    worst_elements = worst_energy = 0.0
    summary = {}
    sparse_phase = SparsePhase(engine)
    with (out / "summary.jsonl").open("w", encoding="utf-8") as log:
        for i in range(args.steps):
            mask, el, energy = sparse_phase.step(state, pool, compartments,
                                                illumination=light, temperature_k=temperatures)
            el_error = float(((el - initial_elements).abs() / initial_elements.abs().clamp_min(1)).max())
            en_error = float(((energy - initial_energy - state.cumulative_input_j).abs()
                              / (initial_energy.abs() + state.cumulative_input_j.abs()).clamp_min(1)).max())
            worst_elements, worst_energy = max(worst_elements, el_error), max(worst_energy, en_error)
            if not torch.isfinite(el).all() or not torch.isfinite(energy).all() or max(el_error, en_error) > 1e-8:
                raise ArithmeticError(f"combined ledger failed at step {state.steps}: {el_error}, {en_error}")
            if i == 0 or (i + 1) % args.every == 0 or i + 1 == args.steps:
                require_source(frozen)
                totals = state.amount_umol.sum((1, 2)).detach().cpu()
                by_planet = []
                for b, p in enumerate(planets):
                    live = [r for r in pool.chains if r["batch"] == b]
                    by_planet.append({"seed": p["cosmic_seed"], "planet": p["id"], "type": p["type"],
                                      "temperature_k": p["temperature_k"], "chains": len(live),
                                      "longest": max((len(r["sequence"]) for r in live), default=0),
                                      "compartment_cells": int(mask[b].sum()),
                                      "species_umol": {s.name: float(totals[b, j]) for j, s in enumerate(SPECIES)}})
                summary = {"step": state.steps, "assemblies": pool.summary(),
                           "compartments": compartments.summary(), "planets": by_planet,
                           "ledger_worst_relative": {"elements": worst_elements, "energy": worst_energy},
                           "profile": args.profile, "life_verdict": "not_established"}
                log.write(json.dumps(summary, allow_nan=False, separators=(",", ":")) + "\n")
                log.flush()
                write_json(out / "summary.json", summary)
                print(json.dumps({"step": state.steps, "chains": len(pool.chains),
                                  "ledger_elements": el_error, "ledger_energy": en_error}), flush=True)
    require_source(frozen)
    checkpoint = {"schema": "life10-state-v1", "source": frozen, "manifest": manifest,
                  "chemistry": engine.checkpoint(state), "assemblies": pool.state_dict(),
                  "compartments": compartments.state_dict(),
                  "planets": planets, "initial_elements": initial_elements.cpu(),
                  "initial_energy": initial_energy.cpu()}
    temp = out / "state.pt.tmp"
    torch.save(checkpoint, temp)
    temp.replace(out / "state.pt")
    # Verify serialization now; full step-for-step resume is covered by tests.
    restored = torch.load(out / "state.pt", weights_only=False, map_location="cpu")
    if not torch.equal(restored["chemistry"]["state"]["amount_umol"], state.amount_umol.cpu()):
        raise RuntimeError("checkpoint state roundtrip failed")
    manifest.update(status="complete", completed_utc=utc(), executed_steps=state.steps,
                    wall_seconds=time.monotonic() - started, stop_reason="step_budget",
                    ledger_worst_relative=summary["ledger_worst_relative"],
                    censored=pool.summary()["censored"] or compartments.summary()["censored"],
                    checkpoint_roundtrip_verified=True)
    write_json(manifest_path, manifest)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run")
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--repeat", type=int, default=0)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--grid", type=int, default=4)
    p.add_argument("--samples", type=int, default=1_000_000)
    p.add_argument("--every", type=int, default=25)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--device", default="cpu")
    p.add_argument("--profile", choices=PROFILES, default="normal")
    p.add_argument("--chains", type=Path)
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if min(args.steps, args.grid, args.samples, args.every, args.threads) <= 0:
        parser.error("steps, grid, samples, every and threads must be positive")
    pth = args.out / "manifest.json"
    existed_before = pth.exists()
    try:
        run(args)
    except Exception as error:
        if not existed_before and pth.exists():
            data = json.loads(pth.read_text())
            # Do not modify an already-completed archive on an overwrite refusal.
            if data.get("status") != "complete":
                data.update(status="failed", error=f"{type(error).__name__}: {error}", stopped_utc=utc())
                write_json(pth, data)
        raise


if __name__ == "__main__":
    main()
