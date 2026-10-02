"""Read-only, deterministic biological specimen export for saved world7 runs.

The simulator evolves numbers, not meshes. ``phenotype-display-v1`` is an
explicit display encoding of those numbers, never a claim of evolved anatomy.
Senses agents have snapshot-local IDs; bodies records are real lineages, not
individual organisms. Society is aggregate context, never fabricated citizens.
No run/replay, random draw, simulator mutation, or simulation cache is used here.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import OrderedDict
from numbers import Integral, Real

SCHEMA_VERSION = "inhabitants-v1"
MAPPING_VERSION = "phenotype-display-v1"
GENE_NAMES = ("smell", "eyes", "touch", "ears", "neuron_gene", "speed", "voice")
LIMITATIONS = [
    "Appearance is a versioned display encoding, not physically evolved geometry or anatomy.",
    "Sensor levels are model traits, not numbers of eyes, ears or other organs; each drawn mark is an emblem.",
    "Body size means cells per body. Display coordinates and radii are schematic units, never metres.",
    "Senses IDs identify array entries in one snapshot only; parents, identities across snapshots and positions were not recorded.",
    "Bodies records are aggregate lineages with source IDs and parents, not counts of individual organisms.",
    "Senses body traits are handed population parameters; no mapping to a particular upstream body lineage is known.",
    "Society has aggregate group/population data. Rendered specimens are upstream biology, not identified society citizens.",
]
_PLANET = re.compile(r"^(?:surface|chemistry|life|cells|bodies|senses|signals|society)_(\d+)$")


def _copy(value):
    """Detach all exported data and reject NaN/Infinity instead of writing bad JSON."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("inhabitant source contains a nonfinite number")
        return number
    if isinstance(value, dict):
        if any(not isinstance(k, str) for k in value):
            raise ValueError("inhabitant source object keys must be strings")
        return {key: _copy(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_copy(item) for item in value]
    raise ValueError(f"inhabitant source contains unsupported {type(value).__name__}")


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()


def _whole(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or int(value) != value or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return int(value)


def _selection(steps):
    if steps is None:
        return None
    if isinstance(steps, (str, bytes, Integral)):
        raise ValueError("steps must be None or a nonempty iterable of integer snapshot indices")
    try:
        chosen = list(steps)
    except TypeError as exc:
        raise ValueError("steps must be iterable") from exc
    if not chosen or any(isinstance(x, bool) or not isinstance(x, Integral) or x < 0 for x in chosen):
        raise ValueError("steps must contain nonnegative integer snapshot indices")
    return sorted(set(int(x) for x in chosen))


def _indices(count, selected, stage):
    if selected is not None and any(index >= count for index in selected):
        raise ValueError(f"selected step outside {stage} snapshots (available 0 to {count - 1})")
    return range(count) if selected is None else selected


def _provenance(world, name, level, era):
    tries = getattr(world, "attempts", {}).get(name, [])
    matches = [i + 1 for i, attempt in enumerate(tries) if attempt is level]
    declared = era.get("kept") if era else None
    if declared is not None:
        declared = _whole(declared, "kept attempt", 1)
    if tries:
        if len(matches) != 1 or (declared is not None and declared != matches[0]):
            raise ValueError(f"{name}: selected attempt does not match the stored rollout")
        selected = matches[0]
    else:
        selected = _whole(declared, "kept attempt", 1) if declared is not None else None
    return {"era": name, "module": "haishool.evo." + level.sim,
            "seed": _whole(level.seed, "level seed"), "params": _copy(level.params),
            "summary": _copy(level.summary), "selected_attempt": selected,
            "selection_basis": "stored object identity and era" if tries else "era metadata only" if declared is not None else "direct stored level",
            "attempts": [{"attempt": i + 1, "seed": _whole(attempt.seed, "attempt seed"),
                          "selected": attempt is level} for i, attempt in enumerate(tries)]}


def _geometry(traits, phenotype_hash, kind):
    """Integer/hash-based layout, independent of random state and source order.

    Radius encodes bit-length of cell count; speed only flattens the display
    ellipse. Hues identify source trait bundles. Emblem sizes encode levels.
    One emblem is deliberately not an organ or neuron count.
    """
    radius = 16 + min(16, traits["body_size"].bit_length())
    speed = traits.get("speed") or 0
    rx, ry = radius, round(radius * (1 - .04 * speed), 4)
    hue = int(phenotype_hash[:8], 16) % 360
    shapes = [{"tag": "ellipse", "attrs": {"cx": 0, "cy": 0, "rx": rx, "ry": ry,
                "fill": f"hsl({hue},45%,58%)", "stroke": "#24303d", "stroke-width": 1.5}, "feature": "body"}]
    def add(tag, feature, **attrs):
        shapes.append({"tag": tag, "attrs": attrs, "feature": feature})
    if kind == "senses_agent":
        sensors = traits["sensors"]
        if sensors["eyes"]["present"]:
            add("circle", "eyes", cx=round(rx * .5, 4), cy=round(-ry * .2, 4),
                r=round(2 + sensors["eyes"]["level"] / 3, 4), fill="#f8fafc", stroke="#1e293b")
        if sensors["ears"]["present"]:
            add("path", "hearing", d=f"M -8 {-ry - 2} Q 0 {-ry - 8 - sensors['ears']['level']} 8 {-ry - 2}",
                fill="none", stroke="#4338ca", **{"stroke-width": 2})
        if sensors["smell"]["present"]:
            add("line", "smell", x1=rx, y1=0, x2=rx + 3 + sensors["smell"]["level"], y2=0,
                stroke="#166534", **{"stroke-width": 2})
        if sensors["touch"]["present"]:
            add("ellipse", "touch", cx=0, cy=0, rx=rx + 3, ry=ry + 3,
                fill="none", stroke="#a16207", **{"stroke-width": 1, "stroke-dasharray": "2 3"})
        if traits["voice"]["present"]:
            add("path", "calling", d=f"M {rx + 7} -8 Q {rx + 17} 0 {rx + 7} 8",
                fill="none", stroke="#9d174d", **{"stroke-width": 2})
        if traits["neuron_count"]:
            add("circle", "neurons", cx=-4, cy=0, r=round(2 + traits["genes"]["neuron_gene"] / 3, 4),
                fill="#7e22ce", opacity=.65)
        if speed:
            add("line", "movement", x1=-rx - 3, y1=4, x2=-rx - 3 - 2 * speed, y2=4,
                stroke="#64748b", **{"stroke-width": 2})
    return {"version": MAPPING_VERSION, "encoding_only": True, "units": "schematic display units",
            "viewBox": [-64, -64, 128, 128], "shapes": shapes}


def _phenotype(kind, raw, phenotypes, genotypes):
    from haishool.evo import senses
    genotype_id = None
    if kind == "senses_agent":
        genome = raw["genome"]
        if len(genome) != len(GENE_NAMES):
            raise ValueError("senses genome must contain seven genes")
        genome = [_whole(x, "gene") for x in genome]
        if any(x > senses.MAX_LEVEL for x in genome[:4]) or genome[4] > senses.MAX_NEURON_GENE \
                or not senses.SPEEDS[0] <= genome[5] <= senses.SPEEDS[1] or genome[6] not in (0, 1):
            raise ValueError("senses genome outside model gene ranges")
        genotype_hash = _hash({"model": "senses", "gene_names": GENE_NAMES, "genes": genome})
        genotype_id = "g-" + genotype_hash
        if genotype_id not in genotypes:
            genotypes[genotype_id] = {"id": genotype_id, "sha256": genotype_hash,
                                       "gene_names": list(GENE_NAMES), "genes": genome}
        genes = dict(zip(GENE_NAMES, genome))
        traits = {"genes": genes, "body_size": raw["body_size"], "body_size_unit": "cells",
                  "cell_types": raw["cell_types"], "neuron_count": senses.neuron_count(genome[4]),
                  "sensors": {key: {"level": genes[key], "present": genes[key] > 0,
                                    "meaning": "model sensor level, not organ count"} for key in senses.SENSORS},
                  "voice": {"level": genes["voice"], "present": genes["voice"] > 0}, "speed": genes["speed"]}
    else:
        traits = {"genes": None, "body_size": raw["size"], "body_size_unit": "cells",
                  "cell_types": raw["cell_types"], "neuron_count": None, "sensors": None, "voice": None, "speed": None,
                  "guild": raw.get("guild"), "adhesion": raw.get("adhesion"), "trophic_level": raw.get("level")}
    traits["body_size"] = _whole(traits["body_size"], "body size in cells", 1)
    traits["cell_types"] = _whole(traits["cell_types"], "cell types", 1)
    phenotype_hash = _hash({"mapping": MAPPING_VERSION, "kind": kind, "genotype_id": genotype_id, "traits": traits})
    phenotype_id = "p-" + phenotype_hash
    if phenotype_id not in phenotypes:
        phenotypes[phenotype_id] = {"id": phenotype_id, "sha256": phenotype_hash, "kind": kind,
            "genotype_id": genotype_id, "traits": traits,
            "trait_sources": ({"genes": "selected senses.genomes[snapshot][source_index]",
                               "body_size": "selected senses.params.body_size, cells per body",
                               "cell_types": "selected senses.params.cell_types",
                               "neuron_count": "haishool.evo.senses.neuron_count(neuron_gene)"}
                              if kind == "senses_agent" else {"body_traits": "selected bodies.lineages[snapshot][source_index]",
                                                             "unrecorded_traits": "genes, sensors, neurons, voice and speed are unknown at this level"}),
            "geometry": _geometry(traits, phenotype_hash, kind)}
    return phenotype_id, genotype_id


def _groups(records, unit):
    grouped = OrderedDict()
    for index, record in enumerate(records):
        group = grouped.setdefault(record["phenotype_id"], {"phenotype_id": record["phenotype_id"],
            "genotype_id": record["genotype_id"], "count": 0, "indices": [], "count_unit": unit})
        group["count"] += 1
        group["indices"].append(index)
    return list(grouped.values())


def export_inhabitants(world, steps=None) -> dict:
    """Export saved snapshots without running a simulation.

    ``steps=None`` selects all snapshots. Otherwise pass a nonempty iterable of
    nonnegative indices (duplicates coalesce); every selected index must exist
    in each stored biological level. A senses snapshot t is generation 5*t's
    input population, not the children produced by that generation. No elapsed
    years are inferred. Bodies snapshots use their source model's tick count.
    """
    from haishool.evo import senses, bodies
    selected = _selection(steps)
    levels = getattr(world, "levels", None)
    if not isinstance(levels, dict) or not isinstance(getattr(world, "steps", None), list):
        raise ValueError("export requires a stored WorldRollout with levels and era steps")
    world_source = {"sim": str(world.sim), "world_seed": _whole(world.seed, "world seed"),
                    "params": _copy(world.params), "summary": _copy(world.summary), "eras": _copy(world.steps),
                    "module": "haishool.evo.world7", "selected_steps": selected,
                    "snapshot_spacing": {"senses_generations_per_step": senses.GENS_PER_STEP,
                                         "bodies_ticks_per_step": bodies.TICKS}}
    world_hash = _hash({key: value for key, value in world_source.items() if key != "selected_steps"})
    world_source["id"] = "w-" + world_hash
    phenotypes, genotypes, planets = {}, {}, []
    eras = {row.get("era"): row for row in world.steps}
    numbers = sorted({int(match[1]) for name in list(levels) + list(eras) if isinstance(name, str)
                      and (match := _PLANET.fullmatch(name))})
    for number in numbers:
        planet = {"id": number, "label": f"Planet {number}", "representation": "upstream_biological_specimens",
                  "source": {"planet_number_meaning": "world7 surface ordinal, not an invented celestial identifier",
                             "eras": _copy([row for row in world.steps if str(row.get("era", "")).endswith(f"_{number}")])},
                  "society_context": None, "snapshots": [], "body_lineages": [], "default_source": None, "reason": None}
        for stage in ("bodies", "senses"):
            name = f"{stage}_{number}"
            if name not in levels:
                continue
            level = levels[name]
            provenance = _provenance(world, name, level, eras.get(name))
            planet["source"][stage] = provenance
            populations = getattr(level, "genomes" if stage == "senses" else "lineages", None)
            if populations is None or len(populations) != len(level.steps):
                raise ValueError(f"{name}: saved populations and metric snapshots must align")
            scope = f"{world_source['id']}/planet-{number}/{stage}/seed-{level.seed}/attempt-{provenance['selected_attempt'] or 0}"
            snapshots = []
            for index in _indices(len(populations), selected, name):
                metrics = _copy(level.steps[index])
                snapshot = {"id": scope + f"/snapshot-{index}", "step": index, "stage": stage,
                            "kind": "agents" if stage == "senses" else "lineages", "metrics": metrics,
                            "generation": index * senses.GENS_PER_STEP if stage == "senses" else None,
                            "tick": index * bodies.TICKS if stage == "bodies" else None,
                            "timing": "input population before reproduction" if stage == "senses" else "recorded bodies tick state",
                            "agents": [], "lineages": [], "groups": [], "reason": None}
                if stage == "senses":
                    if len(populations[index]) > senses.CAP:
                        raise ValueError(f"{name}: population exceeds source agent cap")
                    if "population" in metrics and metrics["population"] != len(populations[index]):
                        raise ValueError(f"{name}: source metrics and genome population disagree")
                    for source_index, genome in enumerate(populations[index]):
                        pid, gid = _phenotype("senses_agent", {"genome": list(genome),
                            "body_size": level.params["body_size"], "cell_types": level.params["cell_types"]}, phenotypes, genotypes)
                        snapshot["agents"].append({"id": snapshot["id"] + f"/agent-{source_index}",
                            "source_index": source_index, "phenotype_id": pid, "genotype_id": gid})
                    snapshot["groups"] = _groups(snapshot["agents"], "agents")
                    if not snapshot["agents"]:
                        snapshot["reason"] = "No living sensing agents were recorded in this snapshot."
                else:
                    seen = set()
                    for source_index, row in enumerate(populations[index]):
                        raw = _copy(row)
                        source_id = _whole(raw["id"], "lineage id")
                        if source_id in seen:
                            raise ValueError(f"{name}: duplicate lineage id in one snapshot")
                        seen.add(source_id)
                        pid, gid = _phenotype("body_lineage", raw, phenotypes, genotypes)
                        snapshot["lineages"].append({"id": scope + f"/lineage-{source_id}", "source_index": source_index,
                            "source_lineage_id": source_id, "parent_lineage_id": raw.get("parent"),
                            "phenotype_id": pid, "genotype_id": gid, "biomass": raw.get("biomass"), "raw": raw})
                    snapshot["groups"] = _groups(snapshot["lineages"], "lineages")
                    if not snapshot["lineages"]:
                        snapshot["reason"] = "No living body lineages were recorded in this snapshot."
                snapshots.append(snapshot)
            if stage == "bodies":
                planet["body_lineages"] = snapshots
                planet["snapshots"] = snapshots
                planet["default_source"] = "bodies"
            else:
                planet["snapshots"] = snapshots
                planet["default_source"] = "senses"
        society_name = f"society_{number}"
        if society_name in levels:
            level = levels[society_name]
            planet["society_context"] = {"kind": "aggregate_society", "individuals_available": False,
                "rendered_specimens_are_society_citizens": False,
                "source": _provenance(world, society_name, level, eras.get(society_name))}
        if planet["default_source"] is None:
            planet["reason"] = "This stored planet has no bodies or senses populations; no inhabitants have been invented."
        planets.append(planet)
    result = {"schema_version": SCHEMA_VERSION, "mapping_version": MAPPING_VERSION,
              "source": world_source, "limitations": list(LIMITATIONS), "phenotypes": phenotypes,
              "genotypes": genotypes, "planets": planets,
              "empty_reason": None if phenotypes else "No living biological specimens were present in the selected stored snapshots."}
    # A last detached copy also guarantees strict JSON-finite geometry and prevents
    # mutation through aliases between fallback snapshots/body_lineages arrays.
    return _copy(result)
