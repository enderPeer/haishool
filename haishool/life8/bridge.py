"""Bridge from round 7 (``haishool.evo.world7``) to round 8 (life8): a world7 planet that reached
the rung ``bodies`` becomes the starting state of a living world.

Every number the bridge sets is a documented formula of a world7 era value, and
:func:`world_config` records, for every output, which era (level) it came from. All formulas are
**toy hand-offs**: they translate one toy's units into another's, they are not physics.

Which planet. world7 rollouts list eras per planet (``surface_k``, ``chemistry_k``,
``replicators_k``, ``cells_k``, ``bodies_k``, ``senses_k``, ...). The bridge takes the planet
with the most links among those up to ``groups`` (``replicators`` .. ``groups``; the inner one of
equals), and requires ``bodies`` among them. Capping the links at ``groups`` makes the choice the
same for a full rollout and for the faster truncated scan of :func:`candidate_worlds`.

Environment (level 3 hand-off ``surface_k``: mass M in earth masses, flux S in earth units,
temperature T in K)::

    radius           R = M ** 0.27                        rocky mass-radius scaling, from memory
    gravity          g = M / R**2 = M ** 0.46              earth = 1
    brightness       b = min(1, S)                          world7's own level-10 rule
    visibility       = 0.5 + 0.5 * b                        Config.visibility, multiplies sight
    action_cost      = 0.08 * g ** 0.5                       every action works against weight
    base_metabolism  = 0.045 * g ** 0.25 * (1 + 0.01 * |T - 288|)
                                                            posture plus thermoregulation;
                                                            288 K = the earth's mean
    food_regrowth    = 0.10 * clip(sqrt(light / 200), 0.5, 1.5)
                                                            light = level 9's producers' light
                                                            (200 * S), earth = 200

0.08, 0.045 and 0.10 are life8's defaults, so an earth twin (M = 1, S = 1, T = 288,
light = 200) keeps them exactly.

Founders (level 9 ``bodies_k`` and level 10 ``senses_k``). Every gene is drawn uniformly from
``center * (0.75, 1.25)``, the width of Matrix's own founders (``World.create`` draws 0.75-1.25)::

    body cells       n = consumer_size (the common consumers' body; level 9's hand-off to level 10)
    body_mass        center = clip(1 + 0.1 * (log2(n) - 7), 0.5, 2.0)    128 cells -> 1.0
    metabolism       center = clip((n / 128) ** -0.25, 0.5, 2.0)          Kleiber per cell, as level 9
    manipulation     center = clip(0.6 + 0.08 * cell_types, 0.5, 2.0)     5 cell types -> 1.0
    speed            center = 1.0                                         no level sets it (default)
    sensing          center = clip(0.6 + 0.16 * max(eyes, ears), 0.4, 1.8) level 10 mean sensor
                                                                          levels (0-5); 2.5 -> 1.0
    voice share      = talkers                                            level 10: share with a voice
    hearing share    = min(1, ears)                                       level 10 mean ear level;
                                                                          an upper bound on the share
                                                                          with ears of level >= 1

Without a ``senses`` era, sensing keeps center 1.0 and every founder has voice and hearing
(life8's defaults), and the provenance says so.

Objects (level 4 ``chemistry_k``: ``mix``, ``h2o``, ``organic``). The ``initial_objects`` of the
config are shared out over materials by weight (largest remainder, ties in the order below)::

    stone   1.0                                   rock of a terran planet, always
    wood    0.5 * clip(organic / 50, 0.2, 2.0)    plant-like matter from prebiotic carbon (toy)
    fiber   0.5 * clip(organic / 50, 0.2, 2.0)
    clay    0.5 * min(1, h2o / 5000)              weathered rock in water
    metal   0.25 if mix == "reducing" else 0      native metal in a reducing crust (toy)

Ecology hints. Predators, injury and an oxygen limit on body mass are not yet life8 mechanisms.
The bridge maps them into ``founders["ecology_hints"]`` (marked ``"connected": False``) for the
ecology stage to wire to its own config fields::

    predator_density   world7.predators_of(predator_share, trophic_levels)    level 9 via level 10's
                       rule: 3 * share, at most 0.3; 0 below 3 trophic levels
    predator_pressure  senses_k.predator_pressure (None without a senses era)
    attack_injury      0.25 * (trophic_levels - 2), clip 0 .. 0.5: a longer food chain has larger
                       top predators (toy, invented)
    oxygen             bodies_k.air (present-day units)
    oxygen_max_cells   bodies.oxygen_max_size(air) (level 9's diffusion limit)
    max_body_mass      the body_mass formula above at n = oxygen_max_cells
    gravity            g (for a movement-cost factor), ambient_temperature_c = T - 273.15
                       (life8's objects cool toward 20 C today)

Candidate worlds. :func:`candidate_worlds` scans world7 seeds in order and keeps the first ``n``
that reach bodies. Its fast path runs world7's own chain (same level runs, same seeds) but stops
each planet's climb after ``senses``: signals and society, the costly upper levels, are never
needed here. The eras it returns equal the prefix of ``world7.run(seed).steps`` (tested).
:func:`synthetic_worlds` draws plausible planets directly when volume is needed; those are
labelled ``"source": "synthetic"`` and are not world7 outcomes.
"""
from __future__ import annotations

import copy
import math
import random
from dataclasses import replace

from . import materials
from .config import Config

BRIDGE_VERSION = "life8-bridge-v1"
#: salt of the founders' own RNG: the bridge never draws from the world's stream
BRIDGE_SALT = 0x6272_6964_6765  # "bridge"
#: links of a planet's climb that the bridge looks at (world7.LINKS from replicators to groups)
CLIMB_LINKS = ("replicators", "cells", "complex_cells", "bodies", "senses", "groups")
#: world7 planet levels the fast path runs (it stops after senses)
FAST_LEVELS = ("replicators", "cells", "bodies", "senses")
MATERIAL_ORDER = ("stone", "wood", "fiber", "clay", "metal")
GENES = ("body_mass", "speed", "sensing", "manipulation", "metabolism")
SPREAD = (.75, 1.25)


def _clip(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def _w7():
    from haishool.evo import world7  # imported lazily: life8's engine core stays standard library
    return world7


# ---------------------------------------------------------------------------------------------
# reading a world7 world


def _steps_of(world) -> tuple[int | None, list[dict], str]:
    if isinstance(world, dict):
        return world.get("seed"), list(world["steps"]), world.get("source", "world7")
    return world.seed, list(world.steps), "world7"


def _base(era: str) -> str:
    return era.rsplit("_", 1)[0] if era.rsplit("_", 1)[-1].isdigit() else era


def _number(era: str) -> int:
    tail = era.rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else 0


def _world7_planet(world) -> tuple[int, list[str], dict[str, dict]]:
    w7 = _w7()
    _, steps, _ = _steps_of(world)
    best, best_links = 0, []
    for s in steps:
        if _base(s["era"]) == "chemistry" and s["precursors"] > 0:
            k = _number(s["era"])
            links = [x for x in w7.planet_links(steps, k) if x in CLIMB_LINKS]
            if len(links) > len(best_links):
                best, best_links = k, links
    eras = {_base(s["era"]): s for s in steps if _number(s["era"]) == best and best} if best else {}
    return best, best_links, eras


def reaches_bodies(world) -> bool:
    return "bodies" in planet_of(world)[1]


# ---------------------------------------------------------------------------------------------
# the formulas (each one a small named function, so tests can check them one by one)


def gravity_of(mass: float) -> float:
    """Surface gravity in earth units from the mass in earth masses: R = M^0.27, g = M^0.46."""
    return mass ** 0.46


def visibility_of(flux: float) -> float:
    return 0.5 + 0.5 * min(1.0, flux)


def action_cost_of(g: float, base: float = .08) -> float:
    return base * g ** 0.5


def metabolism_of(g: float, kelvin: float, base: float = .045) -> float:
    return base * g ** 0.25 * (1 + 0.01 * abs(kelvin - 288))


def food_regrowth_of(light: float, base: float = .10) -> float:
    return base * _clip(math.sqrt(light / 200), 0.5, 1.5)


def body_mass_of(cells: float) -> float:
    return _clip(1 + 0.1 * (math.log2(max(1, cells)) - 7), 0.5, 2.0)


def metabolism_gene_of(cells: float) -> float:
    return _clip((max(1, cells) / 128) ** -0.25, 0.5, 2.0)


def manipulation_of(cell_types: int) -> float:
    return _clip(0.6 + 0.08 * cell_types, 0.5, 2.0)


def sensing_of(eyes: float, ears: float) -> float:
    return _clip(0.6 + 0.16 * max(eyes, ears), 0.4, 1.8)


def material_weights(chemistry: dict) -> dict[str, float]:
    carbon = 0.5 * _clip(chemistry["organic"] / 50, 0.2, 2.0)
    return {"stone": 1.0, "wood": carbon, "fiber": carbon, "clay": 0.5 * min(1.0, chemistry["h2o"] / 5000),
            "metal": 0.25 if chemistry["mix"] == "reducing" else 0.0}


def apportion(total: int, weights: dict[str, float]) -> dict[str, int]:
    """Largest-remainder shares of ``total`` objects; ties go to the earlier material."""
    mass = sum(weights.values())
    if total == 0 or mass <= 0:
        return {m: 0 for m in weights}
    exact = {m: total * w / mass for m, w in weights.items()}
    out = {m: int(math.floor(v)) for m, v in exact.items()}
    left = total - sum(out.values())
    order = sorted(weights, key=lambda m: (-(exact[m] - out[m]), MATERIAL_ORDER.index(m)))
    for m in order[:left]:
        out[m] += 1
    return out


# ---------------------------------------------------------------------------------------------
# the hand-off


def world_config(world, base: Config | None = None) -> tuple[Config, dict]:
    """(Config, founders) for a world7 world (a ``WorldRollout`` or a dict with ``steps``, as
    :func:`candidate_worlds` and :func:`synthetic_worlds` return) that reached ``bodies``.

    ``base`` gives every config field the bridge does not set (default ``Config()``).
    ``founders`` is a JSON-ready dict: ``anatomy`` (gene -> [low, high]), ``organs`` (voice and
    hearing shares), ``objects`` (material -> count), ``environment``, ``ecology_hints`` (not yet
    connected) and ``sources`` (output -> the era and value it came from)."""
    w7 = _w7()
    from haishool.evo import bodies as bodies_level
    seed, _, source = _steps_of(world)
    k, links, eras = planet_of(world)
    if "bodies" not in links:
        raise ValueError(f"world {seed!r} did not reach bodies on any planet (links {links})")
    base = base or Config()
    surface, chem, body = eras["surface"], eras["chemistry"], eras["bodies"]
    senses = eras.get("senses")
    sources: dict[str, str] = {}

    g = gravity_of(surface["mass"])
    kelvin = surface["temperature"]
    cfg = replace(base, visibility=visibility_of(surface["flux"]),
                  action_cost=action_cost_of(g, base.action_cost),
                  base_metabolism=metabolism_of(g, kelvin, base.base_metabolism),
                  food_regrowth=food_regrowth_of(body["light"], base.food_regrowth))
    sources["visibility"] = f"surface_{k}.flux={surface['flux']}"
    sources["action_cost"] = f"surface_{k}.mass={surface['mass']} (gravity)"
    sources["base_metabolism"] = f"surface_{k}.mass={surface['mass']}, surface_{k}.temperature={kelvin}"
    sources["food_regrowth"] = f"bodies_{k}.light={body['light']}"

    cells = body["consumer_size"] if body["consumer_size"] > 0 else body["max_size"]
    types = body["consumer_types"] if body["consumer_size"] > 0 else body["cell_types"]
    centers = {"body_mass": body_mass_of(cells), "metabolism": metabolism_gene_of(cells),
               "manipulation": manipulation_of(types), "speed": 1.0, "sensing": 1.0}
    sources["body_mass"] = sources["metabolism"] = f"bodies_{k}.consumer_size={cells}"
    sources["manipulation"] = f"bodies_{k}.consumer_types={types}"
    sources["speed"] = "life8 default (no level sets it)"
    organs = {"voice": 1.0, "hearing": 1.0}
    if senses is not None:
        centers["sensing"] = sensing_of(senses["eyes"], senses["ears"])
        organs = {"voice": float(senses["talkers"]), "hearing": float(min(1.0, senses["ears"]))}
        sources["sensing"] = f"senses_{k}.eyes={senses['eyes']}, ears={senses['ears']}"
        sources["voice"] = f"senses_{k}.talkers={senses['talkers']}"
        sources["hearing"] = f"senses_{k}.ears={senses['ears']}"
    else:
        sources["sensing"] = sources["voice"] = sources["hearing"] = "life8 default (no senses era)"
    anatomy = {gene: [centers[gene] * SPREAD[0], centers[gene] * SPREAD[1]] for gene in GENES}

    weights = material_weights(chem)
    objects = apportion(cfg.initial_objects, weights)
    sources["objects"] = f"chemistry_{k}.mix={chem['mix']}, organic={chem['organic']}, h2o={chem['h2o']}"

    predators = w7.predators_of(body["predator_share"], body["trophic_levels"])
    o2_cells = bodies_level.oxygen_max_size(body["air"])
    hints = {"connected": False,
             "note": "for the ecology stage: not yet life8 mechanisms; no config field reads these",
             "predator_density": predators, "predator_share": body["predator_share"],
             "trophic_levels": body["trophic_levels"],
             "predator_pressure": senses["predator_pressure"] if senses is not None else None,
             "attack_injury": _clip(0.25 * (body["trophic_levels"] - 2), 0.0, 0.5),
             "oxygen": body["air"], "oxygen_max_cells": o2_cells, "max_body_mass": body_mass_of(o2_cells),
             "gravity": g, "ambient_temperature_c": kelvin - 273.15,
             "sources": {"predator_density": f"bodies_{k}.predator_share, trophic_levels",
                         "predator_pressure": f"senses_{k}.predator_pressure",
                         "attack_injury": f"bodies_{k}.trophic_levels",
                         "oxygen": f"bodies_{k}.air", "oxygen_max_cells": f"bodies_{k}.air",
                         "gravity": f"surface_{k}.mass", "ambient_temperature_c": f"surface_{k}.temperature"}}
    founders = {"version": BRIDGE_VERSION, "seed": seed, "source": source, "planet": k, "links": links,
                "anatomy": anatomy, "organs": organs, "objects": objects, "material_weights": weights,
                "environment": {"mass": surface["mass"], "gravity": g, "flux": surface["flux"],
                                "brightness": min(1.0, surface["flux"]), "temperature_k": kelvin,
                                "light": body["light"], "body_cells": cells, "cell_types": types,
                                "senses_result": senses["result"] if senses is not None else None},
                "ecology_hints": hints, "sources": sources}
    return cfg, founders


def create_world(seed: int, config: Config, founders: dict):
    """A life8 World whose founders and objects follow ``founders``.

    World.create makes the world as usual; the bridge then redraws each founder's genes (agents
    in id order, from its own RNG seeded with ``seed ^ BRIDGE_SALT``, so the world's stream is
    untouched) and gives the objects (in id order) the apportioned materials, interleaved
    round-robin. The material ledger's starting heat is recomputed and the world validated. The
    result is an ordinary state: ``to_dict``/``from_dict`` continue it exactly."""
    from .world import World
    world = World.create(seed, config)
    rng = random.Random(seed ^ BRIDGE_SALT)
    ranges, organs = founders["anatomy"], founders["organs"]
    for identity in sorted(world.agents):
        anatomy = world.agents[identity].anatomy
        for gene in GENES:
            setattr(anatomy, gene, rng.uniform(*ranges[gene]))
        anatomy.voice = 1 if rng.random() < organs["voice"] else 0
        anatomy.hearing = 1 if rng.random() < organs["hearing"] else 0
    left = dict(founders["objects"])
    sequence = []
    while any(left.values()):
        for m in MATERIAL_ORDER:
            if left.get(m, 0) > 0:
                sequence.append(m)
                left[m] -= 1
    free = sorted(world.objects)
    if len(sequence) != len(free):
        raise ValueError(f"founders give {len(sequence)} objects, the config makes {len(free)}")
    for identity, material in zip(free, sequence):
        obj = world.objects[identity]
        obj.material = material
        obj.sharpness = .2 if material in ("stone", "metal") else .03
        obj.properties = materials.material_properties(material, obj.mass)
    world.ledger["initial_material_heat"] = sum(materials.thermal_energy(o) for o in world.objects.values())
    world.validate()
    return world


# ---------------------------------------------------------------------------------------------
# finding worlds


def _climb_to_senses(w7, seed: int, k: int, surface: dict, star: dict, p: dict, source) -> list[dict]:
    """world7._climb with the levels cut after senses (same runs, seeds and era records)."""
    out: list[dict] = []
    age, end = surface["age"], star["star_end"]
    if not w7.ocean(surface) or w7.in_time(age, w7.SPAN["chemistry"], end) != "yes":
        return out
    name = f"chemistry_{k}"
    out.append(w7.chemistry_era(source.get(name, 0, w7.planet_seed(seed, k),
                                           w7.chem_params(seed, k, surface["temperature"], p["atoms"])), k, age))
    eras = {"surface": surface, "chemistry": out[-1]}
    if out[-1]["precursors"] <= 0:
        return out
    age = w7.era_end(age, w7.SPAN["chemistry"])
    for level in w7.LEVELS:
        if level.name not in FAST_LEVELS:
            break
        name, span = f"{level.name}_{k}", w7.SPAN[level.name]
        handed = level.handed(eras)
        tried: list[dict] = []
        for j in range(1, p["max_tries"] + 1):
            if w7.in_time(w7.era_end(age, w7.duration_of(j - 1, span)), span, end) != "yes":
                break
            ts = w7.try_seed(seed, k, level.number, j)
            tried.append(level.fields(source.get(name, j, ts, {**level.params(ts, handed), **level.extra})))
            if level.upper(tried[-1]):
                break
        if not tried:
            break
        kept = w7.kept_try(level, tried)
        source.keep(name, kept)
        era = {"era": name, "age": age, "duration": w7.duration_of(len(tried), span), "tries": len(tried),
               "kept": kept, **{key: handed[param] for key, param in level.shown}, **tried[kept - 1]}
        out.append(era)
        eras[level.name] = era
        if not level.upper(era):
            break
        age = w7.era_end(age, era["duration"])
    return out


def scan_world(seed: int) -> list[dict]:
    """The eras of world7 seed ``seed`` up to each planet's ``senses`` era (world7._chain with the
    climb cut there). Equal to the matching prefix of ``world7.run(seed).steps``."""
    w7 = _w7()
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
            steps += _climb_to_senses(w7, seed, k, surface, star, p, source)
    return steps


def _scan_record(seed: int) -> dict | None:
    steps = scan_world(seed)
    record = {"seed": seed, "source": "world7", "truncated_after": "senses", "steps": steps}
    k, links, _ = planet_of(record)
    if "bodies" not in links:
        return None
    record.update(planet=k, links=links)
    return record


def candidate_worlds(n: int, start_seed: int = 0, workers: int = 1, max_seeds: int = 100_000) -> list[dict]:
    """The first ``n`` world7 seeds from ``start_seed`` on that reach bodies, in seed order, as
    dicts ``{seed, source, truncated_after, steps, planet, links}`` that :func:`world_config`
    accepts. ``workers`` > 1 scans in processes (same results, same order). Stops after
    ``max_seeds`` seeds were looked at."""
    if n <= 0:
        return []
    found: list[dict] = []
    seed = start_seed
    stop = start_seed + max_seeds
    if workers <= 1:
        while len(found) < n and seed < stop:
            record = _scan_record(seed)
            if record is not None:
                found.append(record)
            seed += 1
        return found
    from concurrent.futures import ProcessPoolExecutor
    batch = 4 * workers
    with ProcessPoolExecutor(max_workers=workers) as pool:
        while len(found) < n and seed < stop:
            seeds = list(range(seed, min(seed + batch, stop)))
            for record in pool.map(_scan_record, seeds):
                if record is not None and len(found) < n:
                    found.append(record)
            seed = seeds[-1] + 1
    return found


# ---------------------------------------------------------------------------------------------
# synthetic planets


def synthetic_world(seed: int, senses: bool | None = None) -> dict:
    """One plausible planet drawn directly (``random.Random(seed)``), as era records of the same
    shape as world7's. Labelled ``"source": "synthetic"``: it is not a world7 outcome.

    Draws (ranges chosen to cover the qualifying world7 worlds of seeds 0-299, see the test):
    mass log-uniform 0.3 .. 10 earth masses (world7's terran range), flux uniform over the
    temperate zone 0.356 .. 1.107; the temperature follows from world7's own rule
    (``temperature_of(flux, greenhouse_of(flux, mass, 0))``), redrawn until the water is liquid.
    Chemistry: mix ``reducing`` with chance 1/5 (level 4's rule), organic log-uniform 20 .. 400,
    h2o 4600 .. 5400. Bodies: light = world7's ``light_of(flux)``, consumer_size 2^(6..9) capped
    by the oxygen limit, consumer_types 3 .. 8, trophic_levels 3 or 4, air 0.05 .. 0.32,
    predator_share log-uniform 0.005 .. 0.17. Senses (with chance 0.98, as 48 of the 49
    qualifying seeds 0-299 have a senses era, unless ``senses`` is given): eyes 0 .. 4, ears
    0 .. 4.5, talkers 0 .. 1, predator_pressure 0.05 .. 1.

    The draws are independent, uniform over ranges: they cover what world7 makes but not its
    correlations or its frequencies. Use them for volume and robustness, and world7 seeds
    (:func:`candidate_worlds`) for claims about the round-7 chain."""
    w7 = _w7()
    from haishool.evo import bodies as bodies_level
    rng = random.Random(f"life8-synthetic:{seed}")
    while True:
        mass = sig3(math.exp(rng.uniform(math.log(0.3), math.log(10.0))))
        flux = sig3(rng.uniform(0.356, 1.107))
        greenhouse = w7.greenhouse_of(flux, mass, 0.0)
        kelvin = w7.temperature_of(flux, greenhouse)
        if w7.water_of(kelvin, mass, 0.0) == "liquid" and w7.surface_of(mass, 0.0) == "terran":
            break
    air = sig3(rng.uniform(0.05, 0.32))
    size = min(1 << rng.randint(6, 9), bodies_level.oxygen_max_size(air))
    types = rng.randint(3, 8)
    levels = rng.choice((3, 4))
    steps = [{"era": "surface_1", "orbit": None, "mass": mass, "gas": 0.0, "flux": flux, "surface": "terran",
              "greenhouse": greenhouse, "temperature": kelvin, "water": "liquid"},
             {"era": "chemistry_1", "mix": "reducing" if rng.random() < 0.2 else "ocean",
              "h2o": rng.randint(4600, 5400), "organic": round(math.exp(rng.uniform(math.log(20), math.log(400)))), "precursors": 1},
             {"era": "bodies_1", "light": w7.light_of(flux), "species": rng.randint(15, 40),
              "max_size": max(size, 1 << rng.randint(6, 9)), "cell_types": types + rng.randint(0, 2),
              "consumer_size": size, "consumer_types": types, "trophic_levels": levels, "air": air,
              "predator_share": sig3(math.exp(rng.uniform(math.log(0.005), math.log(0.17)))),
              "result": "animals_like"}]
    if senses if senses is not None else rng.random() < 0.98:
        steps.append({"era": "senses_1", "eyes": sig3(rng.uniform(0, 4)), "ears": sig3(rng.uniform(0, 4.5)),
                      "talkers": sig3(rng.uniform(0, 1)), "predator_pressure": sig3(rng.uniform(0.05, 1)),
                      "brightness": min(1.0, flux), "result": "sensing"})
    return {"seed": seed, "source": "synthetic", "steps": steps, "planet": 1}


def synthetic_worlds(n: int, start_seed: int = 0) -> list[dict]:
    return [synthetic_world(s) for s in range(start_seed, start_seed + n)]


def sig3(x: float) -> float:
    return float(f"{x:.3g}")


def _synthetic_planet(world: dict) -> tuple[int, list[str], dict[str, dict]]:
    eras = {_base(s["era"]): s for s in world["steps"]}
    links = ["replicators", "cells", "complex_cells", "bodies"] + (["senses"] if "senses" in eras else [])
    return 1, links, eras


def planet_of(world) -> tuple[int, list[str], dict[str, dict]]:
    """(k, links up to ``groups``, eras of planet k by base name) of the planet the bridge uses.
    Synthetic worlds hold one planet with its links up to bodies (and senses when drawn)."""
    if isinstance(world, dict) and world.get("source") == "synthetic":
        return _synthetic_planet(world)
    return _world7_planet(world)


def describe(founders: dict) -> dict:
    """A short JSON summary of a hand-off (for logs and sweeps)."""
    return {"seed": founders["seed"], "source": founders["source"], "planet": founders["planet"],
            "links": founders["links"], "environment": copy.deepcopy(founders["environment"]),
            "objects": dict(founders["objects"])}


# ---------------------------------------------------------------- ecology hints -> living-world fields

PREDATORS_PER_DENSITY = 10.  # world7 predator_density 0-0.3 -> 0-3 predators in a 32 x 24 world


def connect_ecology(cfg: Config, founders: dict) -> tuple[Config, dict]:
    """Map founders["ecology_hints"] onto the living-world Config fields (round-8 ecology).

    predators          = round(10 x predator_density)              (bodies_k.predator_share, trophic_levels)
    predator_damage    = 0.1 + 0.6 x attack_injury                  (bodies_k.trophic_levels; attack_injury is a toy)
    predator_visibility= clip(1 - 0.5 x predator_pressure, 0.4, 1)  (senses_k.predator_pressure; unchanged without senses)
    ambient_temperature= ambient_temperature_c                      (surface_k.temperature)
    max_body_mass and oxygen stay hints: life8 has no oxygen limit on body mass yet.
    Use with a living base, e.g. world_config(rec, base=config.living_config(log="compact")).
    Returns a new (Config, founders); the hints are marked connected with the fields they set."""
    hints = dict(founders["ecology_hints"])
    fields = {"predators": int(round(PREDATORS_PER_DENSITY * hints["predator_density"])),
              "predator_damage": round(0.1 + 0.6 * hints["attack_injury"], 6),
              "ambient_temperature": round(_clip(hints["ambient_temperature_c"], -80., 200.), 6)}
    if hints.get("predator_pressure") is not None:
        fields["predator_visibility"] = round(_clip(1 - 0.5 * hints["predator_pressure"], 0.4, 1.0), 6)
    hints.update(connected=True, note="connected by bridge.connect_ecology; max_body_mass and oxygen are not",
                 connected_fields=dict(fields))
    out = copy.deepcopy(founders)
    out["ecology_hints"] = hints
    for name in fields:
        out["sources"][name] = "ecology_hints: " + {"predators": "predator_density",
                                                    "predator_damage": "attack_injury",
                                                    "ambient_temperature": "ambient_temperature_c",
                                                    "predator_visibility": "predator_pressure"}[name]
    return replace(cfg, **fields), out
