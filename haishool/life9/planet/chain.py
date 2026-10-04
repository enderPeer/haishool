"""The world7 chain as the starting conditions of a life9 planet (PLANET-SPEC section 2.2).

:func:`chain_inputs` runs world7's chain for one seed up to each planet's ``senses`` era (the
prefix :func:`haishool.life8.bridge.scan_world` runs; the same level runs and seeds as
``world7.run``) and keeps the level rollouts behind the eras, because the eras drop numbers the
planet engine needs:

* ``stars.cloud`` gives all ten cloud mass fractions (the ``stars`` era prints only six);
* ``r.levels['planets']`` gives the planet's unrounded orbit, mass and flux, the star's unrounded
  luminosity (``params['luminosity']``, ``planets.luminosity7``) and the sibling planets;
* the surface era keeps its own flux (``era_flux_earth``: 3-digit luminosity over the 3-digit orbit
  squared, rounded to 3 digits), from which world7 computes the temperature. ``planet.t_eq_k`` is
  the equilibrium temperature of that era flux, so ``t_eq_k + greenhouse_k`` is exactly the value
  world7 rounds to the whole-K ``t_surface_k``; ``flux_earth`` stays the unrounded level flux (the
  insolation);
* ``r.levels['chemistry_k'].summary`` gives the molecule counts.

The planet is the one the life8 bridge uses (``bridge.planet_of``: most links up to ``groups``);
without any climb, the first temperate terran planet with liquid water. A seed with neither
raises :class:`NoHabitablePlanet`.

The result is JSON-ready and cached under ``runs/life9/chain-cache/`` (git-ignored), so world7
runs once per seed. :func:`synthetic_inputs` draws a planet of the same structure from documented
ranges (``"source": "synthetic"``), for volume and robustness tests; it is not a world7 outcome.

Every value has an entry in ``provenance`` (dotted key -> era or level and file:line).
"""
from __future__ import annotations

import inspect
import json
import math
import os
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CACHE_DIR = REPO / "runs" / "life9" / "chain-cache"
CHAIN_VERSION = "life9-chain-v2"          # v2: t_eq_k from the era flux, era_flux_earth, sky t_eq_k
SYNTHETIC_VERSION = "life9-synthetic-v2"


class NoHabitablePlanet(ValueError):
    """The seed's chain has no temperate terran planet with liquid water."""


def _modules():
    from haishool.cosmos import planets
    from haishool.evo import stars, world7
    from haishool.life8 import bridge
    return world7, stars, planets, bridge


def _where(obj) -> str:
    """``path:line`` of a function or class, relative to the repo."""
    path = Path(inspect.getsourcefile(obj)).resolve()
    line = inspect.getsourcelines(obj)[1]
    try:
        path = path.relative_to(REPO)
    except ValueError:
        pass
    return f"{path.as_posix()}:{line}"


def _file(module) -> str:
    """Repo-relative path of a module."""
    path = Path(inspect.getsourcefile(module)).resolve()
    try:
        return path.relative_to(REPO).as_posix()
    except ValueError:
        return path.name


def _plain(x):
    """JSON-ready copy: tuples to lists, numpy scalars to Python numbers."""
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, bool) or x is None or isinstance(x, str):
        return x
    if isinstance(x, int):
        return int(x)
    if isinstance(x, float):
        return float(x)
    if hasattr(x, "item"):
        return x.item()
    return x


# ---------------------------------------------------------------------------------------------
# world7


def _scan(seed: int):
    """world7's chain for ``seed`` cut after each planet's ``senses`` era, as bridge.scan_world
    runs it, but with the level rollouts kept: ``(steps, source, params)``."""
    w7, _, _, bridge = _modules()
    p = w7._params(seed, {})
    source = w7._Live()
    steps = [w7.nucleo_era(source.get("nucleo", 0, seed, {"rules": 7}))]
    gas = w7.first_gas(steps[0]["hydrogen"], steps[0]["helium"])
    span = w7.enrichment_years(p["formation_time"])
    steps.append(w7.stars_era(source.get("stars", 0, seed, {"efficiency": p["efficiency"], "t_end": span / w7.GYR,
                                                            **gas}), p["formation_time"]))
    enriched = steps[-1]
    cloud = w7.cloud_era(enriched, p)
    steps.append(cloud)
    collapse = w7.collapse_era(source.get("collapse", 0, seed, {"spin": cloud["spin"], "cooling": cloud["cooling"],
                                                                "n": p["particles"], "rules": 7}), cloud)
    steps.append(collapse)
    star = w7.star_era(enriched, cloud, collapse)
    steps.append(star)
    if star["fusion"] == "yes" and star["disc_solids"] > 0 \
            and w7.in_time(star["age"], w7.SPAN["planets"], star["star_end"]) == "yes":
        system = source.get("planets", 0, seed, {"star_mass": star["star_mass"], "disc_mass": star["disc_solids"],
                                                 "t_gas": p["t_gas"], "gas_mass": star["disc_gas"], "rules": 7})
        surfaces = w7.surface_eras(system, star, w7.era_end(star["age"], w7.SPAN["planets"]))
        steps.append(w7.planets_era(system, star, p["t_gas"], len(surfaces)))
        for k, surface in enumerate(surfaces, start=1):
            steps.append(surface)
            steps += bridge._climb_to_senses(w7, seed, k, surface, star, p, source)
    return steps, source, p


def _era_base(name: str) -> tuple[str, int]:
    head, _, tail = name.rpartition("_")
    return (head, int(tail)) if tail.isdigit() and head else (name, 0)


def _surface_body(system, star: dict, k: int) -> dict:
    """The level-3 body behind era ``surface_k`` (world7.surface_eras' own selection)."""
    w7, _, _, _ = _modules()
    rocky = sorted((b for b in system.system if b["type"] == "rocky"), key=lambda b: (b["orbit"], b["id"]))
    temperate = [b for b in rocky if w7.zone_of(w7.flux_of(star["luminosity"], w7.sig(b["orbit"]))) == "temperate"]
    return temperate[k - 1]


def run_chain(seed: int) -> dict:
    """The chain inputs of world7 seed ``seed`` (no cache; about 3-4 s)."""
    w7, stars, planets, bridge = _modules()
    steps, source, p = _scan(seed)
    eras = {s["era"]: s for s in steps}
    star = eras["star"]
    k, links, planet_eras = bridge.planet_of({"seed": seed, "steps": steps})
    if not k:
        oceans = [s for s in steps if _era_base(s["era"])[0] == "surface" and w7.ocean(s)]
        if not oceans:
            raise NoHabitablePlanet(f"world7 seed {seed} has no temperate terran planet with liquid water")
        k = _era_base(oceans[0]["era"])[1]
        planet_eras = {_era_base(s["era"])[0]: s for s in steps if _era_base(s["era"])[1] == k}
    surface = planet_eras["surface"]
    system = source.levels["planets"]
    body = _surface_body(system, star, k)
    stars_level = source.levels["stars"]
    cloud = stars.cloud(stars_level)
    timed = [s for s in planet_eras.values() if "duration" in s]
    now = max((w7.era_end(s["age"], s["duration"]) for s in timed), default=surface["age"])

    w_star, w_cloud = _where(w7.star_era), _where(stars.cloud)
    w_planets, w_lum = _where(type(system)), _where(planets.luminosity7)
    prov: dict[str, str] = {}
    out = {"seed": seed, "source": "world7", "chain_version": CHAIN_VERSION,
           "reached": w7.rung_of(steps), "truncated_after": "senses", "planet_index": k,
           "links": list(links), "params": {key: p[key] for key in w7.PARAM_KEYS}}
    prov["reached"] = (f"world7.rung_of over the eras cut after senses ({_where(w7.rung_of)}; "
                       f"bridge.scan_world prefix {_where(bridge.scan_world)})")
    prov["params"] = f"world7._params(seed) ({_where(w7._params)})"
    prov["planet_index"] = f"bridge.planet_of: most links up to groups ({_where(bridge.planet_of)})"

    out["star"] = {"mass_msun": star["star_mass"], "luminosity_lsun": float(system.params["luminosity"]),
                   "age_yr": star["age"], "lifetime_yr": star["lifetime"], "end_yr": star["star_end"],
                   "now_yr": now, "elapsed_yr": now - star["age"], "frost_line_au": star["frost_line"],
                   "hz_inner_au": star["hz_inner"], "hz_outer_au": star["hz_outer"],
                   "disc_solids_earth": star["disc_solids"], "disc_gas_earth": star["disc_gas"]}
    prov.update({
        "star.mass_msun": f"era star.star_mass = cloud_mass * star_share ({w_star})",
        "star.luminosity_lsun": f"level planets params.luminosity = planets.luminosity7(star_mass) unrounded "
                                f"({w_lum}); era star prints {star['luminosity']}",
        "star.age_yr": f"era star.age: years after the Big Bang at which the star is born ({w_star})",
        "star.lifetime_yr": f"era star.lifetime = 1e10 yr * M / L ({_where(w7.lifetime_of)})",
        "star.end_yr": f"era star.star_end ({w_star})",
        "star.now_yr": f"end of the last era of planet {k} (age + duration; {_where(w7.era_end)}): the year the habitat starts",
        "star.elapsed_yr": "star.now_yr - star.age_yr: the star's (and planet's) age when the habitat starts",
        "star.frost_line_au": f"era star.frost_line ({w_star})",
        "star.hz_inner_au": f"era star.hz_inner ({w_star})",
        "star.hz_outer_au": f"era star.hz_outer ({w_star})",
        "star.disc_solids_earth": f"era star.disc_solids ({_where(w7.disc_solids_of)})",
        "star.disc_gas_earth": f"era star.disc_gas ({_where(w7.disc_gas_of)})"})

    out["planet"] = {"id": body["id"], "mass_earth": float(body["mass"]), "orbit_au": float(body["orbit"]),
                     "flux_earth": float(body["flux"]), "era_flux_earth": float(surface["flux"]),
                     "t_eq_k": planets.equilibrium_temperature(surface["flux"]),
                     "greenhouse_k": surface["greenhouse"], "t_surface_k": surface["temperature"],
                     "water_state": surface["water"], "surface": surface["surface"], "gas_earth": float(body["gas"]),
                     "type": body["type"]}
    prov.update({
        "planet.id": f"level planets system id of era {surface['era']} ({_where(w7.surface_eras)})",
        "planet.mass_earth": f"level planets system mass, unrounded ({w_planets}); era prints {surface['mass']}",
        "planet.orbit_au": f"level planets system orbit, unrounded ({w_planets}); era prints {surface['orbit']}",
        "planet.flux_earth": f"level planets system flux = L / a^2 unrounded ({_where(planets.flux)}); "
                             f"era prints {surface['flux']}",
        "planet.era_flux_earth": f"era {surface['era']}.flux = sig(L_sig / a_sig^2), world7.flux_of "
                                 f"({_where(w7.flux_of)}): the flux world7's temperature rule uses",
        "planet.t_eq_k": f"planets.equilibrium_temperature(era flux) = 254.6 K era_flux^0.25 "
                         f"({_where(planets.equilibrium_temperature)}), unrounded: t_eq_k + greenhouse_k is what "
                         f"world7.temperature_of rounds ({_where(w7.temperature_of)}); era prints {surface['t_eq']}",
        "planet.greenhouse_k": f"era {surface['era']}.greenhouse ({_where(w7.greenhouse_of)})",
        "planet.t_surface_k": f"era {surface['era']}.temperature, whole K ({_where(w7.temperature_of)})",
        "planet.water_state": f"era {surface['era']}.water ({_where(w7.water_of)})",
        "planet.surface": f"era {surface['era']}.surface ({_where(w7.surface_of)})",
        "planet.gas_earth": f"level planets system gas ({w_planets})",
        "planet.type": f"level planets system type ({w_planets})"})

    out["cloud"] = {key: float(cloud[key]) for key in (*stars.ELEMENTS, "metallicity")}
    out["cloud"]["time_gyr"] = float(cloud["time"])
    for key in out["cloud"]:
        prov[f"cloud.{key}"] = f"stars.cloud(level stars) mass fraction of the gas the star forms from ({w_cloud})"

    bio, sen = planet_eras.get("bodies"), planet_eras.get("senses")
    skip = {"era", "age", "duration", "tries", "kept"}
    if bio is not None:
        biosphere = {"oxygen_pal": bio["air"], **{key: v for key, v in bio.items() if key not in skip | {"air"}}}
        for key in biosphere:
            name = "air" if key == "oxygen_pal" else key
            prov[f"biosphere.{key}"] = f"era {bio['era']}.{name} (level bodies, {_file(w7.bodies)}; keys world7.ERAS)"
    else:
        biosphere = {"oxygen_pal": 0.0, "light": w7.light_of(surface["flux"]), "trophic_levels": 0, "result": "none"}
        prov["biosphere.oxygen_pal"] = "new_rule: no bodies era, so no oxygenic life: 0 PAL"
        prov["biosphere.light"] = f"world7.light_of(flux) ({_where(w7.light_of)})"
        prov["biosphere.trophic_levels"] = "no bodies era"
        prov["biosphere.result"] = "no bodies era"
    if sen is not None:
        biosphere["senses"] = {key: v for key, v in sen.items() if key not in skip}
        prov["biosphere.senses"] = f"era {sen['era']} (level senses, {_file(w7.senses)}; keys world7.ERAS)"
    else:
        biosphere["senses"] = None
        prov["biosphere.senses"] = "no senses era"
    out["biosphere"] = biosphere

    chem_era = planet_eras.get("chemistry")
    if chem_era is not None:
        level = source.levels[chem_era["era"]]
        out["chemistry"] = {**{key: v for key, v in chem_era.items() if key not in skip},
                            "summary": dict(level.summary)}
        prov["chemistry"] = (f"era {chem_era['era']} and r.levels['{chem_era['era']}'].summary "
                             f"(level chemistry, {_where(w7.chemistry_era)})")
    else:
        out["chemistry"] = None
        prov["chemistry"] = "no chemistry era"

    out["sky"] = [{"id": b["id"], "orbit_au": float(b["orbit"]), "mass_earth": float(b["mass"]),
                   "t_k": b["temperature"], "t_eq_k": planets.equilibrium_temperature(b["flux"]), "type": b["type"]}
                  for b in system.system if b["id"] != body["id"]]
    prov["sky"] = (f"level planets system: the sibling planets ({w_planets}); t_k = level 3's whole-K "
                   f"temperature with its greenhouse (500 K in a runaway; cloud tops for a gas giant; "
                   f"{_where(planets._temperature7)}), t_eq_k = planets.equilibrium_temperature(flux) without greenhouse")
    out["provenance"] = prov
    return _plain(out)


def _cache_path(seed: int, cache_dir: Path | str | None) -> Path:
    return Path(cache_dir if cache_dir is not None else CACHE_DIR) / f"world7-{seed}.json"


def chain_inputs(seed: int, cache_dir: Path | str | None = None, refresh: bool = False) -> dict:
    """The chain inputs of world7 seed ``seed``: read from the JSON cache, or run and cached."""
    path = _cache_path(seed, cache_dir)
    if not refresh and path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if data.get("chain_version") == CHAIN_VERSION and data.get("seed") == seed:
                return data
        except (OSError, ValueError):
            pass
    data = run_chain(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, sort_keys=True)
    os.replace(tmp, path)
    return data


# ---------------------------------------------------------------------------------------------
# synthetic planets

#: documented draw ranges of :func:`synthetic_inputs` (new_rule: they cover world7's temperate
#: terran worlds, not their frequencies)
SYNTHETIC_RANGES = {
    "star_mass_msun": (0.45, 1.2),        # log-uniform; luminosity, lifetime by world7's own laws
    "formation_quarters": (4, 40),        # cloud formation at 1..10 Gyr (world7 draws 0.25..12)
    "elapsed_gyr": (2.0, 9.0),            # the star's age when the habitat starts, at most 0.9 lifetime
    "min_metallicity_zsun": 0.1,          # redraw clouds below 0.1 Z_sun (no rocky planets there)
    "siblings": (1, 5),                   # sibling planets in the sky
    "sibling_mass_earth": (0.1, 20.0),    # log-uniform
    "sibling_orbit_factor": (0.2, 20.0),  # log-uniform multiple of the planet's orbit, kept 1.3x away
}


def synthetic_inputs(seed: int) -> dict:
    """A drawn planet in the structure of :func:`chain_inputs` (``"source": "synthetic"``).

    The planet, its chemistry, bodies and senses eras are the life8 bridge's documented
    synthetic draws (``bridge.synthetic_world``: mass 0.3..10 earth masses, temperate flux, liquid
    water by world7's own temperature rule, air 0.05..0.32 PAL, ...). On top of them, from
    ``random.Random("life9-synthetic:<seed>")``: a star (mass log-uniform in
    ``SYNTHETIC_RANGES``, luminosity ``planets.luminosity7``, lifetime ``world7.lifetime_of``), the
    cloud from the chain's own stars level for a drawn formation time and efficiency, the orbit
    that gives the drawn flux (``sqrt(L / flux)``), the star's age at the habitat's start, and
    sibling planets (level 3's temperature rule ``t_k`` and the bare ``t_eq_k``, as in world7)."""
    w7, stars, planets, bridge = _modules()
    world = bridge.synthetic_world(seed)
    eras = {_era_base(s["era"])[0]: s for s in world["steps"]}
    surface = eras["surface"]
    rng = random.Random(f"life9-synthetic:{seed}")
    lo, hi = SYNTHETIC_RANGES["star_mass_msun"]
    while True:
        mass = w7.sig(math.exp(rng.uniform(math.log(lo), math.log(hi))))
        formation_time = 0.25 * rng.randint(*SYNTHETIC_RANGES["formation_quarters"])
        efficiency = rng.choice(w7.EFFICIENCIES)
        gas = w7.first_gas(0.753, 0.247)
        level = stars.run(seed, efficiency=efficiency, t_end=w7.enrichment_years(formation_time) / w7.GYR, **gas)
        cloud = stars.cloud(level)
        if cloud["metallicity"] >= SYNTHETIC_RANGES["min_metallicity_zsun"] * stars.Z_SUN:
            break
    luminosity = planets.luminosity7(mass)
    lifetime = w7.lifetime_of(mass, w7.sig(luminosity))
    born = w7.formation_year(formation_time)
    e_lo, e_hi = SYNTHETIC_RANGES["elapsed_gyr"]
    elapsed = int(rng.uniform(e_lo, min(e_hi, 0.9 * lifetime / w7.GYR)) * w7.GYR)
    flux = float(surface["flux"])
    orbit = math.sqrt(luminosity / flux)
    sky = []
    for i in range(rng.randint(*SYNTHETIC_RANGES["siblings"])):
        f_lo, f_hi = SYNTHETIC_RANGES["sibling_orbit_factor"]
        while True:
            a = orbit * math.exp(rng.uniform(math.log(f_lo), math.log(f_hi)))
            if max(a / orbit, orbit / a) >= 1.3 and all(max(a / s["orbit_au"], s["orbit_au"] / a) >= 1.3 for s in sky):
                break
        m_lo, m_hi = SYNTHETIC_RANGES["sibling_mass_earth"]
        m = math.exp(rng.uniform(math.log(m_lo), math.log(m_hi)))
        # level 3's labels (planets.planet_type): inside the frost line super_earth from 10 earth
        # masses, else rocky; beyond it ice_giant from 5, else icy
        inside = a < planets.FROST_AU * math.sqrt(luminosity)
        kind = ("super_earth" if m >= 10 else "rocky") if inside else ("ice_giant" if m >= 5 else "icy")
        f_sib = planets.flux(luminosity, a)
        # level 3's whole-K temperature (planets._temperature7: these kinds all have a surface)
        sky.append({"orbit_au": a, "mass_earth": m, "t_k": planets.surface_temperature(f_sib),
                    "t_eq_k": planets.equilibrium_temperature(f_sib), "type": kind})
    sky.sort(key=lambda s: s["orbit_au"])
    sky = [{"id": i + 1, **s} for i, s in enumerate(sky)]
    bio, sen, chem_era = eras.get("bodies"), eras.get("senses"), eras.get("chemistry")
    skip = {"era", "age", "duration", "tries", "kept"}
    out = {"seed": seed, "source": "synthetic", "chain_version": SYNTHETIC_VERSION,
           "reached": "senses" if sen is not None else "bodies", "truncated_after": "senses", "planet_index": 1,
           "links": ["replicators", "cells", "complex_cells", "bodies"] + (["senses"] if sen is not None else []),
           "params": {"formation_time": formation_time, "efficiency": efficiency},
           "star": {"mass_msun": mass, "luminosity_lsun": luminosity, "age_yr": born, "lifetime_yr": lifetime,
                    "end_yr": born + lifetime, "now_yr": born + elapsed, "elapsed_yr": elapsed,
                    "frost_line_au": w7.frost_line_of(luminosity), "hz_inner_au": w7.hz_inner_of(luminosity),
                    "hz_outer_au": w7.hz_outer_of(luminosity), "disc_solids_earth": None, "disc_gas_earth": None},
           "planet": {"id": 0, "mass_earth": float(surface["mass"]), "orbit_au": orbit, "flux_earth": flux,
                      "era_flux_earth": flux, "t_eq_k": planets.equilibrium_temperature(flux),
                      "greenhouse_k": surface["greenhouse"],
                      "t_surface_k": surface["temperature"], "water_state": surface["water"],
                      "surface": surface["surface"], "gas_earth": 0.0, "type": "rocky"},
           "cloud": {**{key: float(cloud[key]) for key in (*stars.ELEMENTS, "metallicity")},
                     "time_gyr": float(cloud["time"])},
           "biosphere": {"oxygen_pal": bio["air"], **{key: v for key, v in bio.items() if key not in skip | {"air"}},
                         "senses": {key: v for key, v in sen.items() if key not in skip} if sen is not None else None},
           "chemistry": {key: v for key, v in chem_era.items() if key not in skip},
           "sky": sky}
    drawn = f"synthetic: random.Random('life9-synthetic:{seed}') draw, ranges SYNTHETIC_RANGES ({_where(synthetic_inputs)})"
    from_bridge = f"synthetic: bridge.synthetic_world({seed}) draw ({_where(bridge.synthetic_world)})"
    prov = {"reached": "synthetic: the bridge's synthetic eras", "params": drawn, "planet_index": "synthetic: one planet",
            "sky": drawn + f"; t_k = planets.surface_temperature(flux) (level 3's whole-K rule with greenhouse, "
                           f"{_where(planets.surface_temperature)}), t_eq_k = planets.equilibrium_temperature(flux)",
            "chemistry": from_bridge}
    for key in out["star"]:
        prov[f"star.{key}"] = drawn
    prov["star.luminosity_lsun"] = f"planets.luminosity7(mass) ({_where(planets.luminosity7)})"
    prov["star.lifetime_yr"] = f"world7.lifetime_of ({_where(w7.lifetime_of)})"
    prov["star.age_yr"] = f"world7.formation_year(drawn formation_time) ({_where(w7.formation_year)})"
    for key in ("frost_line_au", "hz_inner_au", "hz_outer_au"):
        prov[f"star.{key}"] = f"world7 {key[:-3]}_of(L) ({_where(w7.frost_line_of)})"
    prov["star.disc_solids_earth"] = prov["star.disc_gas_earth"] = "synthetic: no disc is drawn"
    for key in out["planet"]:
        prov[f"planet.{key}"] = from_bridge
    prov["planet.orbit_au"] = "derived: sqrt(L / flux) au (planets.flux inverted)"
    prov["planet.era_flux_earth"] = "the bridge's surface era flux (the synthetic orbit reproduces it exactly)"
    prov["planet.t_eq_k"] = f"planets.equilibrium_temperature(flux) ({_where(planets.equilibrium_temperature)})"
    for key in out["cloud"]:
        prov[f"cloud.{key}"] = f"stars.cloud(stars.run(drawn efficiency, formation time)) ({_where(stars.cloud)})"
    for key in out["biosphere"]:
        prov[f"biosphere.{key}"] = from_bridge
    out["provenance"] = prov
    return _plain(out)


def earth_inputs(cache_dir: Path | str | None = None, refresh: bool = False) -> dict:
    """:func:`earth_inputs_computed`, read from the JSON cache (``earth-reference.json`` next to the world7
    caches) or computed and cached, so that hosts without scipy (world7's cosmos levels need it) can build the
    Earth reference from the shared cache, as they do the world7 seeds."""
    path = Path(cache_dir if cache_dir is not None else CACHE_DIR) / "earth-reference.json"
    if not refresh and path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if data.get("chain_version") == CHAIN_VERSION and data.get("source") == "earth_reference":
                return data
        except (OSError, ValueError):
            pass
    data = earth_inputs_computed()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, sort_keys=True)
    os.replace(tmp, path)
    return data


def earth_inputs_computed() -> dict:
    """Earth and the sun in the structure of :func:`chain_inputs` (``"source": "earth_reference"``):
    the calibration case of the formation rules. Reference values: 1 au, 1 earth mass, the solar
    cloud of Asplund et al. 2009 (X 0.7381, Y 0.2485, Z 0.0134, element shares from the
    photospheric log eps and the atomic weights), the sun's age 4.567 Gyr and world7's own Earth
    temperature (254.6 K + 33 K, haishool/cosmos/planets.py), oxygen 1 PAL."""
    from . import constants as C
    _, _, planets, _ = _modules()
    x, z = 0.7381, 0.0134
    named = {"carbon": "C", "nitrogen": "N", "oxygen": "O", "neon": "Ne", "magnesium": "Mg", "silicon": "Si",
             "iron": "Fe"}
    w = C.atomic_weights()
    cloud = {name: x * 10 ** (C.SOLAR_LOG_EPS[sym] - 12.0) * w[sym] / w["H"] for name, sym in named.items()}
    cloud = {"hydrogen": x, "helium": 1.0 - x - z, **cloud, "other": z - sum(cloud.values()), "metallicity": z,
             "time_gyr": None}
    flux = 1.0
    born = 13.8e9 - C.EARTH_AGE_YR
    out = {"seed": 0, "source": "earth_reference", "chain_version": CHAIN_VERSION, "reached": "states",
           "truncated_after": None, "planet_index": 1, "links": [], "params": {},
           "star": {"mass_msun": 1.0, "luminosity_lsun": 1.0, "age_yr": born, "lifetime_yr": 1e10,
                    "end_yr": born + 1e10, "now_yr": 13.8e9, "elapsed_yr": C.EARTH_AGE_YR,
                    "frost_line_au": planets.FROST_AU, "hz_inner_au": planets.hz_inner(1.0),
                    "hz_outer_au": planets.hz_outer(1.0), "disc_solids_earth": None, "disc_gas_earth": None},
           "planet": {"id": 0, "mass_earth": 1.0, "orbit_au": 1.0, "flux_earth": flux, "era_flux_earth": flux,
                      "t_eq_k": planets.equilibrium_temperature(flux), "greenhouse_k": planets.greenhouse(flux),
                      "t_surface_k": planets.surface_temperature(flux), "water_state": "liquid",
                      "surface": "terran", "gas_earth": 0.0, "type": "rocky"},
           "cloud": cloud, "biosphere": {"oxygen_pal": 1.0, "light": 200.0, "trophic_levels": 4, "senses": None},
           "chemistry": None, "sky": []}
    ref = "reference: Earth and the sun (IAU 2015 nominal values; Asplund et al. 2009 solar composition)"
    prov = {key: ref for key in ("reached", "params", "planet_index", "sky", "chemistry")}
    for group in ("star", "planet", "cloud", "biosphere"):
        for key in out[group]:
            prov[f"{group}.{key}"] = ref
    prov["planet.t_eq_k"] = prov["planet.greenhouse_k"] = prov["planet.t_surface_k"] = \
        "world7's own Earth: planets.equilibrium_temperature, greenhouse, surface_temperature at flux 1"
    out["provenance"] = prov
    return _plain(out)


def inputs_for(seed: int, source: str = "world7", **kw) -> dict:
    """``chain_inputs(seed)`` for ``source="world7"``, ``synthetic_inputs(seed)`` for ``"synthetic"``."""
    if source == "world7":
        return chain_inputs(seed, **kw)
    if source == "synthetic":
        return synthetic_inputs(seed)
    raise ValueError(f"source must be world7 or synthetic, not {source!r}")
