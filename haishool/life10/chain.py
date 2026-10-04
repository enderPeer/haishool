"""Unselected prebiotic input chain for life10.

One cosmic seed names one inherited, coarse stellar-system calculation, not an
atom-resolved universe. No biology module, habitability selection, retries or
replacement founders run here. Failed and barren histories are returned as data.

The legacy levels do not carry atom identities between nuclear, stellar and
gravitational models. This module makes that seam explicit: it retains their raw
states, then allocates the enriched cloud's elemental masses homogeneously among
the star, retained disc and remaining cloud. Planet masses debit the disc. A
microscopic parcel is a declared numerical sample of each planet's inventory;
it is not a second planet-sized inventory or proof of realistic differentiation.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

FORMAT_VERSION = "life10-prebiotic-chain-1"
ROOT = Path(__file__).resolve().parents[2]
ATOMIC_WEIGHTS = ROOT / "data" / "truth-v5" / "atomic_weights.json"
M_EARTH_KG = 5.972168e24
SOLAR_FLUX_W_M2 = 1361.0
ELEMENT_SYMBOLS = {
    "hydrogen": "H", "helium": "He", "carbon": "C", "nitrogen": "N",
    "oxygen": "O", "neon": "Ne", "magnesium": "Mg", "silicon": "Si",
    "iron": "Fe", "phosphorus": "P", "other": "Other",
}
SOURCE_PATHS = (
    "haishool/life10/chain.py", "haishool/cosmos/__init__.py",
    "haishool/cosmos/nucleo.py", "haishool/cosmos/gravity.py",
    "haishool/cosmos/planets.py", "haishool/evo/__init__.py",
    "haishool/evo/stars.py", "haishool/evo/world7.py",
    "haishool/truth/__init__.py", "haishool/truth/formula.py",
    "haishool/truth/substances.py", "haishool/truth/elements.py",
    "haishool/truth/reactions.py", "data/truth-v5/atomic_weights.json",
)


@dataclass(frozen=True)
class ChainConfig:
    """Declared numerical resolution; changing it creates another experiment."""

    collapse_particles: int = 64
    sample_atoms: int = 10000
    atom_quantum_mol: float = 1e-6
    background_temperature_k: float = 2.725

    def check(self) -> None:
        for name, minimum in (("collapse_particles", 32), ("sample_atoms", 0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not math.isfinite(self.background_temperature_k) or self.background_temperature_k < 0:
            raise ValueError("background_temperature_k must be finite and nonnegative")
        if not math.isfinite(self.atom_quantum_mol) or self.atom_quantum_mol <= 0:
            raise ValueError("atom_quantum_mol must be finite and positive")


def _modules():
    # Lazy imports let audits and small tests avoid initializing all legacy levels.
    from haishool.cosmos import gravity, nucleo, planets
    from haishool.evo import stars, world7
    return nucleo, stars, gravity, planets, world7


def _plain(value: Any) -> Any:
    """JSON values with nonfinite numbers identified, never nonstandard NaN JSON."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return {"nonfinite": "nan" if math.isnan(value) else "+inf" if value > 0 else "-inf"}
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if hasattr(value, "tolist"):
        return _plain(value.tolist())
    if hasattr(value, "item"):
        return _plain(value.item())
    raise TypeError(f"unsupported chain state {type(value).__name__}")


def canonical_json(value: dict) -> str:
    """Stable encoding used for experiment identity, without wall-clock metadata."""
    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def source_hashes() -> dict[str, str]:
    """Hashes of the borrowed laws/data and this adapter, independent of Git status."""
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCE_PATHS}


def stream_seed(cosmic_seed: int, repeat_seed: int, planet_id: Any, stream: str = "chemistry") -> int:
    """An independent stream per planet; batching/order cannot change its draws."""
    message = canonical_json({"version": FORMAT_VERSION, "cosmic_seed": cosmic_seed,
                              "repeat_seed": repeat_seed, "planet_id": planet_id, "stream": stream})
    return int.from_bytes(hashlib.sha256(message.encode()).digest()[:8], "little") & ((1 << 63) - 1)


def _rollout(r: Any) -> dict:
    out = {"seed": int(r.seed), "params": _plain(r.params), "steps": _plain(r.steps),
           "summary": _plain(r.summary)}
    for name in ("ledger", "bodies", "events", "system"):
        if hasattr(r, name):
            out[name] = _plain(getattr(r, name))
    return out


def _raw_collapse(gravity, seed: int, spin: float, cooling: float, n: int) -> dict:
    """Use full measurements: gravity.run rounds its output to three digits."""
    ledger: dict = {}
    steps, radius0 = [], None
    for i, (x, v, m) in enumerate(gravity.evolve(seed, spin, cooling, n, gravity.MASS,
                                               gravity.EPS, ledger, rules=7)):
        full = gravity.measure7(x, v, m, gravity.EPS, radius0,
                                i * gravity.OUT_DT, ledger["radiated"])
        radius0 = full["radius"] if radius0 is None else radius0
        steps.append(_plain(full))
    if not steps:
        raise ArithmeticError("collapse returned no states")
    last = steps[-1]
    return {"seed": seed, "params": {"rules": 7, "spin": spin, "cooling": cooling,
                                       "n": n, "mass": gravity.MASS, "eps": gravity.EPS},
            "steps": steps, "summary": {"star": last["largest_clump"],
                                          "disc_mass": last["disc_mass"],
                                          "stage": last["stage"], "clumps": last["clumps"]},
            "ledger": _plain(ledger), "measurement": "unrounded measure7",
            "final_particle_state": {"positions": _plain(x), "velocities": _plain(v),
                                     "masses": _plain(m),
                                     "units": "legacy dimensionless gravity units"}}


def _finite_nonnegative(value: Any, name: str) -> float:
    x = float(value)
    if not math.isfinite(x) or x < 0:
        raise ArithmeticError(f"{name} must be finite and nonnegative, got {value!r}")
    return x


def _fractions(cloud: dict) -> dict[str, float]:
    values = {ELEMENT_SYMBOLS[name]: _finite_nonnegative(cloud.get(name, 0.0), name)
              for name in ELEMENT_SYMBOLS}
    total = math.fsum(values.values())
    if not math.isclose(total, 1.0, abs_tol=1e-9, rel_tol=1e-9):
        raise ArithmeticError(f"source cloud element mass fractions sum to {total}, not one")
    # The upstream star model aggregates the remaining elements in 'other'. Do
    # not convert its unknown share to phosphorus, salts or any useful nutrient.
    return values


def parcel_inventory(element_mass_kg: dict[str, float], sample_atoms: int,
                     atom_quantum_mol: float = 1e-6) -> dict:
    """Integer parcel quanta apportioned from known atomic abundances.

    Unknown 'Other' is not converted into a chemical element. Hamilton rounding
    affects this finite parcel only; the full elemental mass ledger is unchanged.
    Each integer is ``atom_quantum_mol`` moles of an element, not one real atom.
    There is no minimum count of a useful atom and no seed-specific enrichment.
    """
    with ATOMIC_WEIGHTS.open(encoding="utf-8") as handle:
        weights = {k: float(v) for k, v in json.load(handle).items()}
    known = {s: m / (weights[s] / 1000.0) for s, m in element_mass_kg.items()
             if s in weights and m > 0}
    total = math.fsum(known.values())
    count = {s: 0 for s in element_mass_kg if s in weights}
    count.setdefault("P", 0)
    if total and sample_atoms:
        exact = {s: sample_atoms * m / total for s, m in known.items()}
        count.update({s: int(math.floor(v)) for s, v in exact.items()})
        left = sample_atoms - sum(count.values())
        order = sorted(exact, key=lambda s: (-(exact[s] - count[s]), s))
        for s in order[:left]:
            count[s] += 1
    sample_mass = {s: n * weights[s] / 1000.0 * atom_quantum_mol for s, n in count.items()}
    if any(sample_mass[s] > element_mass_kg.get(s, 0.0) for s in sample_mass):
        raise ArithmeticError("parcel atom sample exceeds its planet's element inventory")
    return {"atom_inventory": count, "sample_atoms": sum(count.values()),
            "atom_inventory_unit": "integer atom-equivalent quanta",
            "atom_quantum_mol": atom_quantum_mol,
            "known_element_number_fractions": {s: m / total for s, m in known.items()} if total else {},
            "sample_element_mass_kg": sample_mass,
            "bulk_residual_element_mass_kg": {s: m - sample_mass.get(s, 0.0)
                                               for s, m in element_mass_kg.items()},
            "unresolved_other_mass_kg": element_mass_kg.get("Other", 0.0),
            "method": "finite parcel; largest remainder from known element number fractions"}


def _material_ledger(cloud_earth: float, fractions: dict, star_earth: float,
                     disc_earth: float, planets: list[dict]) -> dict:
    allocated = math.fsum(p["mass_earth"] for p in planets)
    tolerance = max(1e-10, 1e-9 * disc_earth)
    if star_earth + disc_earth > cloud_earth + max(1e-10, 1e-9 * cloud_earth):
        raise ArithmeticError("star and retained disc exceed the source cloud")
    if allocated > disc_earth + tolerance:
        raise ArithmeticError("formed planets exceed the retained disc; refusing resource creation")
    reservoirs = {"star": star_earth, "remaining_cloud": cloud_earth - star_earth - disc_earth,
                  "disc_outside_final_planets": disc_earth - allocated}
    # These residuals include companions, losses and unretained gas, not a rescue
    # reservoir available to chemistry. Nothing is replenished from them.
    if any(m < 0 for m in reservoirs.values()):
        raise ArithmeticError("negative reservoir in elemental allocation")
    source = {s: cloud_earth * M_EARTH_KG * f for s, f in fractions.items()}
    groups = {name: {s: mass * M_EARTH_KG * f for s, f in fractions.items()}
              for name, mass in reservoirs.items()}
    for p in planets:
        groups[f"planet:{p['id']}"] = p["element_mass_kg"]
    errors = {s: math.fsum(g[s] for g in groups.values()) - source[s] for s in fractions}
    return {"source_element_mass_kg": source, "reservoir_element_mass_kg": groups,
            "error_kg": errors,
            "relative_max_error": max((abs(errors[s]) / source[s] for s in source if source[s]), default=0.0),
            "cloud_mass_earth": cloud_earth, "retained_disc_mass_earth": disc_earth,
            "final_planet_mass_earth": allocated,
            "allocation_rule": "homogeneous cloud mass fractions; planet masses debit retained disc",
            "scope": "element allocation from enriched cloud onward; no atom identities across legacy levels"}


def _validate_seed(seed: int, name: str, limit: int = 10 ** 9) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= limit:
        raise ValueError(f"{name} must be an integer from 0 to {limit}")
    return seed


def build_chain(cosmic_seed: int, repeat_seed: int = 0, config: ChainConfig | None = None) -> dict:
    """Return every final planet, or a retained empty/error outcome; never scan for success.

    ``cosmic_seed`` fixes the inherited initial draw and cosmic history.
    ``repeat_seed`` varies only the later parcel microstate, not the planet chain.
    Invalid user configuration raises; failures inside a modeled history are data.
    """
    cosmic_seed = _validate_seed(cosmic_seed, "cosmic_seed")
    repeat_seed = _validate_seed(repeat_seed, "repeat_seed", (1 << 63) - 1)
    cfg = config or ChainConfig()
    cfg.check()
    out = {"format_version": FORMAT_VERSION, "cosmic_seed": cosmic_seed,
           "repeat_seed": repeat_seed, "config": asdict(cfg), "source_hashes": source_hashes(),
           "outcome": "not_started", "stages": {}, "planets": [], "system": [],
           "errors": [], "assumptions": [
               "One seed is one coarse cloud/stellar-system history, not a whole universe.",
               "Borrowed world7 laws are toy models; elemental identities and energy do not span every legacy stage.",
               "Primordial H/He isotopes are combined for enrichment; lithium is excluded and this projection is recorded.",
               "Cloud-to-planet element allocation is homogeneous, not modeled condensation, atmospheric escape or differentiation.",
               "Other elements remain unresolved; phosphorus is zero if absent upstream, never inferred from Other.",
               "Each planet receives a finite parcel of integer atom-equivalent quanta, each worth atom_quantum_mol; its mass is part of the planet's inventory, not an additional inventory.",
               "The inherited greenhouse thermostat is an assumed thermal boundary, not prebiotic atmospheric chemistry.",
               "No plants, bodies, copying motifs, monomer feed, fallback, habitat selection or successful-seed retry.",
           ]}
    stage = "parameters"
    try:
        nucleo, stars, gravity, planets, w7 = _modules()
        p = w7._params(cosmic_seed, {"particles": cfg.collapse_particles})
        out["initial_parameters"] = _plain(p)
        stage = "nucleosynthesis"
        nr = nucleo.run(cosmic_seed, rules=7)
        out["stages"][stage] = _rollout(nr)
        # Preserve the full isotope summary while declaring the inherited
        # enrichment projection; avoid the rounded stars.handoff_in function.
        n = nr.summary
        h = float(n["hydrogen"]) * (1 + 2 * float(n.get("deuterium", 0.0)))
        he = float(n["helium"]) + 3 * float(n.get("helium3", 0.0)) * float(n["hydrogen"])
        gas = {"hydrogen": h / (h + he), "helium": he / (h + he)}
        out["primordial_projection"] = {**gas, "omitted_lithium_number_ratio": n.get("lithium", 0.0),
                                         "normalization": h + he}
        stage = "enrichment"
        sr = stars.run(cosmic_seed, efficiency=p["efficiency"],
                       t_end=w7.enrichment_years(p["formation_time"]) / w7.GYR, **gas)
        out["stages"][stage] = _rollout(sr)
        cloud = stars.cloud(sr)
        out["cloud"] = _plain(cloud)
        fractions = _fractions(cloud)
        z = _finite_nonnegative(cloud["metallicity"], "metallicity")
        cloud_msun = float(p["cloud_mass"]) * (w7.JEANS_FACTOR if z < w7.Z_CRIT else 1.0)
        cooling = w7.COOLING_PRIMORDIAL if z < w7.Z_CRIT else float(p["cooling"])
        stage = "collapse"
        cr = _raw_collapse(gravity, cosmic_seed, float(p["spin"]), cooling, cfg.collapse_particles)
        out["stages"][stage] = cr
        star_share = _finite_nonnegative(cr["summary"]["star"], "star fraction")
        disc_share = _finite_nonnegative(cr["summary"]["disc_mass"], "disc fraction")
        star_msun = cloud_msun * star_share
        cloud_earth = cloud_msun * w7.EARTHS_PER_SUN
        star_earth = cloud_earth * star_share
        retained_earth = cloud_earth * disc_share * w7.DISC_RETAINED
        solids_earth, gas_earth = retained_earth * z, retained_earth * (1 - z)
        fusion = star_msun >= w7.MIN_STAR
        luminosity = planets.luminosity7(star_msun) if fusion else 0.0
        out["star"] = {"mass_msun": star_msun, "luminosity_lsun": luminosity,
                       "fusion": fusion, "cloud_mass_msun": cloud_msun,
                       "disc_solids_earth": solids_earth, "disc_gas_earth": gas_earth,
                       "disc_retained_share": w7.DISC_RETAINED,
                       "earth_masses_per_sun": w7.EARTHS_PER_SUN}
        if not fusion:
            out["outcome"] = "no_star"
        elif solids_earth <= 0:
            out["outcome"] = "no_disc_solids"
        else:
            stage = "planet_formation"
            pr = planets.run(cosmic_seed, rules=7, star_mass=star_msun,
                             disc_mass=solids_earth, gas_mass=gas_earth, t_gas=p["t_gas"])
            out["stages"][stage] = _rollout(pr)
            out["system"] = _plain(pr.system)
            for b in pr.system:
                mass = _finite_nonnegative(b["mass"], "planet mass")
                orbit = _finite_nonnegative(b["orbit"], "planet orbit")
                if not orbit:
                    raise ArithmeticError("planet orbit is zero")
                flux = _finite_nonnegative(planets.flux(luminosity, orbit), "planet flux")
                t_eq = planets.equilibrium_temperature(flux)
                background_t = (t_eq ** 4 + cfg.background_temperature_k ** 4) ** 0.25
                warming = 0.0 if b["type"] == "gas" else planets.greenhouse(flux)
                elements = {s: mass * M_EARTH_KG * f for s, f in fractions.items()}
                row = {"id": b["id"], "type": b["type"], "mass_earth": mass,
                       "orbit_au": orbit, "solid_earth": float(b.get("solid", mass - b.get("gas", 0))),
                       "gas_earth": float(b.get("gas", 0)), "flux_earth": flux,
                       "stellar_flux_w_m2": SOLAR_FLUX_W_M2 * flux,
                       "t_eq_k": t_eq, "temperature_k": background_t + warming,
                       "element_mass_kg": elements,
                       "chemistry_seed": stream_seed(cosmic_seed, repeat_seed, b["id"]),
                       "upstream_diagnostics": _plain(b),
                       "thermal_rule": "unrounded legacy equilibrium + legacy greenhouse + declared radiation background"}
                row.update(parcel_inventory(elements, cfg.sample_atoms, cfg.atom_quantum_mol))
                out["planets"].append(row)
            out["outcome"] = "planets_formed" if out["planets"] else "no_planets"
        stage = "allocation_audit"
        out["material_ledger"] = _material_ledger(cloud_earth, fractions, star_earth,
                                                  retained_earth, out["planets"])
        if out["material_ledger"]["relative_max_error"] > 1e-9:
            raise ArithmeticError("elemental allocation ledger fails its relative tolerance")
    except Exception as error:
        out["outcome"] = "numeric_or_model_error"
        out["errors"].append({"stage": stage, "type": type(error).__name__, "message": str(error)})
    return _plain(out)


def initial_molecules(atom_inventory: dict[str, int]) -> dict[str, int]:
    """Optional explicit initial gas/water assembly; exactly preserves atom counts.

    This is a declared initial-condition convention, not chemistry that evolved.
    Chemistry engines may instead begin directly with the element atoms. No
    nucleotide, activated substrate, polymer, catalyst or compartment is supplied.
    """
    atoms = {s: int(n) for s, n in atom_inventory.items()}
    if any(n < 0 or n != atom_inventory[s] for s, n in atoms.items()):
        raise ValueError("initial atom counts must be nonnegative whole numbers")
    molecules: dict[str, int] = {}
    for name, required in (("H2O", {"H": 2, "O": 1}), ("CO2", {"C": 1, "O": 2}),
                           ("N2", {"N": 2}), ("H2", {"H": 2})):
        count = min(atoms.get(s, 0) // n for s, n in required.items())
        if count:
            molecules[name] = count
            for s, n in required.items():
                atoms[s] -= count * n
    for s, n in atoms.items():
        if n:
            molecules[s] = molecules.get(s, 0) + n
    return molecules
