"""Conserved coarse fatty-acid compartments, assembled from produced lipid.

Concentration-dependent nucleation, lipid uptake, leakage and surface-threshold
fission are declared phenomenological laws. No compartment is initially supplied
or labelled alive. Membrane molecules and cargo remain in explicit inventories;
surface energy is taken from local heat and returned as heat on dissolution.
This declared thermal-payment rule permits spontaneous lipid assembly in the
dark; no photon-powered membrane construction mechanism is imposed.

This does not resolve bilayer curvature, osmotic pressure or molecular dynamics.
With an engine supplied, trapped species undergo the same fixed reaction catalog
using the surrounding cell's heat and photon reservoirs: instantaneous thermal/
light exchange is an explicit approximation. Sparse peptide graphs currently
remain outside this cargo system; no sequence inheritance is claimed at fission.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
import random

import torch

from .streams import planet_streams, tuples

from .chemistry import ChemicalEngine, ChemicalState, ELEMENTS, REACTIONS, SPECIES, SPECIES_INDEX


LIPID = "hexanoic_acid"


@dataclass(frozen=True)
class CompartmentConfig:
    critical_concentration_umol: float = 0.25  # per external grid cell
    nucleation_membrane_umol: float = 0.10
    nucleation_probability: float = 0.02  # per step at one unit supersaturation
    growth_fraction: float = 0.05
    exchange_fraction: float = 0.01
    dissolution_fraction: float = 0.01
    minimum_membrane_umol: float = 0.01
    volume_fraction_per_membrane_umol: float = 0.1
    maximum_volume_fraction: float = 0.20
    fission_threshold_umol: float = 0.40
    fission_probability: float = 0.03
    surface_energy_j_per_umol: float = 0.01
    max_compartments: int = 512

    def __post_init__(self):
        if self.max_compartments < 1:
            raise ValueError("Compartment capacity must be positive")
        for key, value in asdict(self).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid compartment parameter {key}")
        for key in ("nucleation_probability", "growth_fraction", "exchange_fraction",
                    "dissolution_fraction", "maximum_volume_fraction", "fission_probability"):
            if getattr(self, key) > 1:
                raise ValueError(f"{key} must lie in [0, 1]")
        if self.critical_concentration_umol <= 0 or self.minimum_membrane_umol <= 0:
            raise ValueError("Concentration and minimum membrane must be positive")
        if self.nucleation_membrane_umol < self.minimum_membrane_umol:
            raise ValueError("Nucleated membrane must exceed the dissolution limit")
        if self.fission_threshold_umol < 2 * self.minimum_membrane_umol:
            raise ValueError("Both daughters must retain a viable membrane inventory")
        if self.maximum_volume_fraction >= 1:
            raise ValueError("An external chemical reservoir must remain")


class Compartments:
    def __init__(self, batch, grid, seed, config=None, profile="normal"):
        if batch < 1 or grid < 1:
            raise ValueError("Batch and grid must be positive")
        self.batch, self.grid, self.profile = batch, grid, profile
        self.config = config or CompartmentConfig()
        self.stream_seeds, self.rngs = planet_streams(batch, seed, 'compartments')
        self.compartments = []
        self.next_ids = [1] * batch
        self.counts = [0] * batch
        self.events = {"nucleated": 0, "growth_events": 0, "fissions": 0,
                       "dissolutions": 0, "capacity_hits": 0, "exchange_events": 0}
        self.events_by_planet = [{key: 0 for key in self.events} for _ in range(batch)]
        self._local_engine = None

    @property
    def rng(self):
        if self.batch != 1:
            raise ValueError("Use a planet-specific rngs entry")
        return self.rngs[0]

    @rng.setter
    def rng(self, value):
        if self.batch != 1:
            raise ValueError("Use a planet-specific rngs entry")
        self.rngs[0] = value

    def _event(self, batch, name, amount=1):
        self.events[name] += amount
        self.events_by_planet[batch][name] += amount

    def _new(self, b, cell, membrane, surface, cargo, born, parent=None):
        row = {"id": self.next_ids[b], "batch": b, "cell": list(cell),
               "membrane_umol": membrane, "surface_energy_j": surface,
               "cargo_umol": cargo, "born": born, "parent": parent}
        self.next_ids[b] += 1
        self.counts[b] += 1
        self.compartments.append(row)
        return row

    def _uptake(self, row, state, requested):
        b, (x, y) = row["batch"], row["cell"]
        energy = self.config.surface_energy_j_per_umol
        amount = min(requested, float(state.amount_umol[b, x, y, SPECIES_INDEX[LIPID]]))
        if energy:
            amount = min(amount, float(state.heat_j[b, x, y]) / energy)
        amount = max(0, amount) * (1 - 1e-14)
        state.amount_umol[b, x, y, SPECIES_INDEX[LIPID]] -= amount
        state.heat_j[b, x, y] -= amount * energy
        row["membrane_umol"] += amount
        row["surface_energy_j"] += amount * energy
        return amount

    def _release(self, row, state, fraction):
        b, (x, y) = row["batch"], row["cell"]
        cargo = row["cargo_umol"] * fraction
        state.amount_umol[b, x, y] += cargo
        row["cargo_umol"] -= cargo
        lipid, heat = row["membrane_umol"] * fraction, row["surface_energy_j"] * fraction
        state.amount_umol[b, x, y, SPECIES_INDEX[LIPID]] += lipid
        state.heat_j[b, x, y] += heat
        row["membrane_umol"] -= lipid
        row["surface_energy_j"] -= heat

    def _internal_reactions(self, row, state, engine):
        if self._local_engine is None or self._local_engine.device != state.amount_umol.device:
            self._local_engine = ChemicalEngine(replace(engine.config, batch_size=1, grid_size=1),
                                                state.amount_umol.device)
        b, (x, y) = row["batch"], row["cell"]
        zeros = torch.zeros(1, dtype=torch.float64, device=state.amount_umol.device)
        local = ChemicalState(row["cargo_umol"].reshape(1, 1, 1, -1),
                              state.heat_j[b:b+1, x:x+1, y:y+1],
                              state.photon_j[b:b+1, x:x+1, y:y+1],
                              zeros.clone(), zeros.clone(), zeros.clone(), zeros.clone(),
                              torch.zeros(1, len(ELEMENTS), dtype=torch.float64, device=zeros.device),
                              torch.zeros(1, len(REACTIONS), 2, dtype=torch.float64, device=zeros.device),
                              state.steps)
        for i, reaction in enumerate(REACTIONS):
            self._local_engine._reaction(local, reaction, i, False, 1.0)
            self._local_engine._reaction(local, reaction, i, True, 1.0)
        state.reaction_extent_umol[b] += local.reaction_extent_umol[0]

    @torch.no_grad()
    def step(self, state, engine=None):
        if self.profile == "no_compartments":
            return
        cfg, lipid_i = self.config, SPECIES_INDEX[LIPID]
        occupied = {(r["batch"], *r["cell"]) for r in self.compartments}
        candidates = torch.nonzero(state.amount_umol[..., lipid_i]
                                    >= cfg.critical_concentration_umol + cfg.nucleation_membrane_umol)
        for b, x, y in candidates.detach().cpu().tolist():
            rng = self.rngs[b]
            if (b, x, y) in occupied:
                continue
            concentration = float(state.amount_umol[b, x, y, lipid_i])
            excess = concentration / cfg.critical_concentration_umol - 1
            probability = min(1.0, cfg.nucleation_probability * excess)
            if rng.random() >= probability:
                continue
            if self.counts[b] >= cfg.max_compartments:
                self._event(b, "capacity_hits")
                continue
            cost = cfg.nucleation_membrane_umol * cfg.surface_energy_j_per_umol
            if float(state.heat_j[b, x, y]) < cost:
                continue
            row = self._new(b, (x, y), 0.0, 0.0, torch.zeros(len(SPECIES), dtype=torch.float64,
                                                            device=state.amount_umol.device), state.steps)
            self._uptake(row, state, cfg.nucleation_membrane_umol)
            self._event(b, "nucleated")
        for row in list(self.compartments):
            row["cargo_umol"] = row["cargo_umol"].to(state.amount_umol.device)
            b, (x, y) = row["batch"], row["cell"]
            rng = self.rngs[b]
            concentration = float(state.amount_umol[b, x, y, lipid_i])
            excess = concentration - cfg.critical_concentration_umol
            if excess > 0:
                if self._uptake(row, state, excess * cfg.growth_fraction) > 0:
                    self._event(b, "growth_events")
            elif cfg.dissolution_fraction:
                self._release(row, state, cfg.dissolution_fraction)
            if row["membrane_umol"] < cfg.minimum_membrane_umol:
                self._release(row, state, 1.0)
                self.compartments.remove(row)
                self.counts[b] -= 1
                self._event(b, "dissolutions")
                continue
            volume = min(cfg.maximum_volume_fraction,
                         row["membrane_umol"] * cfg.volume_fraction_per_membrane_umol)
            external = state.amount_umol[b, x, y]
            target = external * volume / (1 - volume)
            transfer = cfg.exchange_fraction * (target - row["cargo_umol"])
            transfer[lipid_i] = 0  # membrane uptake is accounted separately
            external -= transfer
            row["cargo_umol"] += transfer
            self._event(b, "exchange_events", int(bool(transfer.count_nonzero())))
            if engine is not None:
                self._internal_reactions(row, state, engine)
            if row["membrane_umol"] >= cfg.fission_threshold_umol \
                    and rng.random() < cfg.fission_probability:
                if self.counts[b] >= cfg.max_compartments:
                    self._event(b, "capacity_hits")
                    continue
                # Membrane surface area is partitioned, not created; this coarse
                # rule conserves area/energy but does not model daughter shape.
                row["membrane_umol"] *= 0.5
                row["surface_energy_j"] *= 0.5
                row["cargo_umol"] *= 0.5
                self._new(b, (x, y), row["membrane_umol"], row["surface_energy_j"],
                          row["cargo_umol"].clone(), state.steps, row["id"])
                self._event(b, "fissions")

    def elemental_totals(self, device="cpu"):
        total = torch.zeros(self.batch, len(ELEMENTS), dtype=torch.float64, device=device)
        formulas = torch.tensor([s.elements for s in SPECIES], dtype=torch.float64, device=device)
        for row in self.compartments:
            total[row["batch"]] += row["cargo_umol"].to(device) @ formulas
            total[row["batch"]] += formulas[SPECIES_INDEX[LIPID]] * row["membrane_umol"]
        return total

    def energy_total(self, device="cpu"):
        total = torch.zeros(self.batch, dtype=torch.float64, device=device)
        energies = torch.tensor([s.chemical_j_per_umol for s in SPECIES], dtype=torch.float64, device=device)
        for row in self.compartments:
            total[row["batch"]] += (row["cargo_umol"].to(device) @ energies
                                    + row["membrane_umol"] * energies[SPECIES_INDEX[LIPID]]
                                    + row["surface_energy_j"])
        return total

    def permeability(self, state):
        result = torch.ones_like(state.heat_j)
        for row in self.compartments:
            b, (x, y) = row["batch"], row["cell"]
            result[b, x, y] *= 1 / (1 + row["membrane_umol"])
        return result

    def summary(self):
        return {"compartments": len(self.compartments),
                "membrane_umol": sum(r["membrane_umol"] for r in self.compartments),
                "cargo_umol": sum(float(r["cargo_umol"].sum()) for r in self.compartments),
                "events": dict(self.events), "censored": bool(self.events["capacity_hits"]),
                "per_planet": [{"batch": b, "compartments": self.counts[b],
                                "events": dict(self.events_by_planet[b]),
                                "censored": bool(self.events_by_planet[b]["capacity_hits"])}
                               for b in range(self.batch)], "capacity_scope": "per_planet",
                "life_verdict": "not_established", "sequence_inheritance": "not_implemented",
                "surface_energy_rule": "thermal_payment"}

    def state_dict(self):
        rows = [{**r, "cell": list(r["cell"]), "cargo_umol": r["cargo_umol"].detach().cpu().clone()}
                for r in self.compartments]
        return {"config": asdict(self.config), "batch": self.batch, "grid": self.grid,
                "schema": "life10-compartments-per-planet-v1",
                "profile": self.profile, "compartments": rows, "next_ids": list(self.next_ids),
                "events": dict(self.events), "events_by_planet": [dict(e) for e in self.events_by_planet],
                "stream_seeds": list(self.stream_seeds), "rngs": [rng.getstate() for rng in self.rngs]}

    @classmethod
    def from_dict(cls, data, device="cpu"):
        if "rngs" not in data and data["batch"] != 1:
            raise ValueError("Legacy shared-RNG batch checkpoints require their frozen original implementation")
        obj = cls(data["batch"], data["grid"], [0] * data["batch"],
                  CompartmentConfig(**data["config"]), data["profile"])
        obj.compartments = [{**r, "cell": list(r["cell"]),
                            "cargo_umol": torch.as_tensor(r["cargo_umol"], dtype=torch.float64,
                                                          device=device).clone()}
                           for r in data["compartments"]]
        obj.next_ids = list(data.get("next_ids", [data.get("next_id", 1)]))
        obj.counts = [sum(r["batch"] == b for r in obj.compartments) for b in range(obj.batch)]
        obj.events = dict(data["events"])
        obj.events_by_planet = [dict(e) for e in data.get("events_by_planet", [obj.events])]
        obj.stream_seeds = list(data.get("stream_seeds", obj.stream_seeds))
        for rng, state in zip(obj.rngs, data.get("rngs", [data.get("rng")])):
            rng.setstate(tuples(state))
        return obj
