"""A new, explicit cell morphogenesis model for an exported biological phenotype.

This model grows and relaxes cells; it does not decorate the earlier specimen
glyph. The earlier evolution model did not store anatomy, tissue identities or
neural connections. Those are NEW assumptions recorded here, not recovered facts.
Distances use a mature cell radius, never metres. Cells are soft spheres with a
small, measured permitted overlap; this is neither atomistic nor a fluid solver.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from numbers import Integral, Real

import numpy as np
from scipy.spatial import cKDTree

SCHEMA_VERSION = "cell-anatomy-v1"
ALGORITHM_VERSION = "division-relaxation-v1"
MAX_CELLS = 4096
UNIT_VOLUME = 4 * math.pi / 3


@dataclass(frozen=True)
class MorphogenesisConfig:
    """All geometric/developmental assumptions, saved verbatim in the result."""
    max_cells: int = MAX_CELLS
    relaxation_steps: int = 140
    final_relaxation_steps: int = 500
    repulsion: float = .3
    adhesion: float = .008
    adhesion_range: float = 2.5
    confinement: float = .002
    displacement_cap: float = .2
    daughter_offset: float = .8
    growth_axes: tuple[float, float, float] = (1.6, 1., .85)
    field_curvature: float = .18
    contact_gap: float = .12
    max_overlap_fraction: float = .08
    neural_neighbors: int = 3


def _integer(value, name, low=0, high=None):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < low or (high is not None and value > high):
        raise ValueError(f"{name} must be an integer in {low}..{high if high is not None else 'unbounded'}")
    return int(value)


def _finite(value, name, low=0., high=float("inf")):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be finite and in {low}..{high}")
    return float(value)


def _config(value):
    if value is None:
        cfg = MorphogenesisConfig()
    elif isinstance(value, MorphogenesisConfig):
        cfg = value
    elif isinstance(value, dict):
        try:
            cfg = MorphogenesisConfig(**value)
        except TypeError as exc:
            raise ValueError("unknown morphogenesis configuration field") from exc
    else:
        raise ValueError("config must be a MorphogenesisConfig or dictionary")
    _integer(cfg.max_cells, "max_cells", 1, MAX_CELLS)
    _integer(cfg.relaxation_steps, "relaxation_steps", 1, 2000)
    _integer(cfg.final_relaxation_steps, "final_relaxation_steps", 1, 5000)
    _integer(cfg.neural_neighbors, "neural_neighbors", 1, 8)
    for name, low, high in (("repulsion", .01, .5), ("adhesion", .001, .1),
                             ("adhesion_range", 2.01, 3.), ("confinement", .00001, .01),
                             ("displacement_cap", .01, .3), ("daughter_offset", .5, .95),
                             ("field_curvature", 0, .5), ("contact_gap", .01, .2),
                             ("max_overlap_fraction", .001, .1)):
        _finite(getattr(cfg, name), name, low, high)
    if not isinstance(cfg.growth_axes, (tuple, list)) or len(cfg.growth_axes) != 3:
        raise ValueError("growth_axes must contain three positive directional weights")
    for axis in cfg.growth_axes:
        _finite(axis, "growth axis", .5, 3.)
    return cfg


def _source(phenotype, cfg):
    if not isinstance(phenotype, dict) or not isinstance(phenotype.get("traits"), dict):
        raise ValueError("phenotype must contain saved traits")
    if phenotype.get("kind") not in ("senses_agent", "body_lineage"):
        raise ValueError("only saved sensing-agent or body-lineage phenotypes can develop")
    traits = copy.deepcopy(phenotype["traits"])
    if traits.get("body_size_unit", "cells") != "cells":
        raise ValueError("source body_size must be a cell count, not a physical length")
    count = _integer(traits.get("body_size"), "body_size", 1)
    if count > cfg.max_cells:
        raise ValueError(f"body_size {count} exceeds the explicit {cfg.max_cells}-cell budget; no downsampling performed")
    types = _integer(traits.get("cell_types"), "cell_types", 1, count)
    neurons = traits.get("neuron_count")
    if neurons is not None:
        _integer(neurons, "neuron_count", 0, count)
    genes = traits.get("genes") or {}
    sensors = traits.get("sensors") or {}
    sensory_levels = {}
    for key in ("eyes", "ears", "smell", "touch"):
        recorded = sensors.get(key)
        level = genes.get(key, recorded.get("level") if recorded else None)
        if level is not None:
            _integer(level, key + " level", 0, 5)
        if recorded and level != recorded.get("level"):
            raise ValueError(f"inconsistent source {key} gene and sensor level")
        if recorded and "present" in recorded and recorded["present"] != (level is not None and level > 0):
            raise ValueError(f"inconsistent source {key} presence")
        sensory_levels[key] = level
    voice = traits.get("voice")
    voice_level = genes.get("voice", voice.get("level") if voice else None)
    if voice_level is not None:
        _integer(voice_level, "voice", 0, 1)
    if voice and voice_level != voice.get("level"):
        raise ValueError("inconsistent source voice gene and trait")
    if voice and "present" in voice and voice["present"] != (voice_level is not None and voice_level > 0):
        raise ValueError("inconsistent source voice presence")
    source = {"phenotype_id": phenotype.get("id"), "kind": phenotype["kind"],
              "genotype_id": phenotype.get("genotype_id"), "traits": traits,
              "body_size": count, "cell_types": types, "neuron_count": neurons,
              "sensor_levels": sensory_levels, "voice_level": voice_level}
    for key in ("world_seed", "planet_id", "snapshot_step", "source_agent_id", "provenance"):
        if key in phenotype:
            source[key] = copy.deepcopy(phenotype[key])
    try:
        source["sha256"] = hashlib.sha256(json.dumps(source, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    except (ValueError, TypeError) as exc:
        raise ValueError("source phenotype must contain finite JSON data") from exc
    return source


def _pairs(positions, radius):
    if len(positions) < 2:
        return np.empty((0, 2), dtype=np.int64)
    pairs = cKDTree(positions).query_pairs(radius, output_type="ndarray")
    if len(pairs):
        pairs = pairs[np.lexsort((pairs[:, 1], pairs[:, 0]))]
    return pairs


def _relax(positions, cfg, iterations):
    """Overdamped soft-sphere repulsion, short-range adhesion and anisotropic field.

    The field acts as a force during growth. No final affine deformation or
    cosmetic appended geometry is applied. Pair forces are equal and opposite.
    """
    positions = positions.copy()
    axes = np.asarray(cfg.growth_axes, dtype=float)
    # Keep field pressure comparable as the tissue grows: for a compact body
    # squared radius scales as N^(2/3). Without this normalization larger bodies
    # would be compressed harder merely because the cell budget was larger.
    field_scale = max(1., len(positions) ** (2 / 3) / 4)
    max_move = 0.
    for _ in range(iterations):
        pairs = _pairs(positions, cfg.adhesion_range)
        forces = -cfg.confinement * positions / (axes * axes * field_scale)
        if len(pairs):
            delta = positions[pairs[:, 1]] - positions[pairs[:, 0]]
            distance = np.linalg.norm(delta, axis=1)
            direction = delta / np.maximum(distance[:, None], 1e-12)
            direction[distance < 1e-12] = (1., 0., 0.)
            compression = distance - 2.
            magnitude = np.where(compression < 0, cfg.repulsion * compression, cfg.adhesion * compression)
            force = direction * magnitude[:, None]
            np.add.at(forces, pairs[:, 0], force)
            np.add.at(forces, pairs[:, 1], -force)
        lengths = np.linalg.norm(forces, axis=1)
        forces *= np.minimum(1., cfg.displacement_cap / np.maximum(lengths, 1e-12))[:, None]
        positions += forces
        positions -= positions.mean(axis=0)
        max_move = float(np.linalg.norm(forces, axis=1).max())
        if not np.isfinite(positions).all():
            raise FloatingPointError("nonfinite morphogenesis relaxation")
    return positions, max_move


def _growth(target, seed, cfg):
    rng = np.random.Generator(np.random.PCG64(seed))
    ids = [0]
    positions = np.zeros((1, 3), dtype=float)
    history = [{"id": 0, "parent_id": None, "generation": 0, "born_round": 0,
                "birth_position": [0., 0., 0.], "divided_round": None}]
    divisions, ledger = [], []
    snapshots = [{"round": 0, "count": 1, "cell_ids": [0], "positions": [[0., 0., 0.]], "radii": [1.]}]
    round_ = 0
    while len(ids) < target:
        round_ += 1
        to_divide = min(len(ids), target - len(ids))
        # A small anterior nutrient bias ranks the eligible cells in partial
        # rounds. Full rounds divide every living cell. This is an assumption.
        score = rng.random(len(ids)) + .1 * positions[:, 0] / max(1., float(np.ptp(positions[:, 0])))
        selected = set(np.argsort(score, kind="stable")[-to_divide:].tolist())
        next_ids, next_positions = [], []
        for index, cell_id in enumerate(ids):
            center = positions[index]
            if index not in selected:
                next_ids.append(cell_id)
                next_positions.append(center.copy())
                continue
            direction = rng.normal(size=3) * np.asarray(cfg.growth_axes)
            direction += cfg.field_curvature * np.array([0., math.sin(float(center[0]) / 4.), math.cos(float(center[0]) / 4.)])
            direction /= np.linalg.norm(direction)
            history[cell_id]["divided_round"] = round_
            history[cell_id]["division_position"] = center.tolist()
            daughters = []
            initial = []
            for sign in (-1., 1.):
                new_id = len(history)
                birth_position = center + sign * cfg.daughter_offset * direction
                daughters.append(new_id)
                initial.append(birth_position.tolist())
                history.append({"id": new_id, "parent_id": cell_id,
                    "generation": history[cell_id]["generation"] + 1, "born_round": round_,
                    "birth_position": birth_position.tolist(), "divided_round": None})
                next_ids.append(new_id)
                next_positions.append(birth_position)
            divisions.append({"round": round_, "parent_id": cell_id, "daughter_ids": daughters,
                              "parent_position": center.tolist(), "daughter_birth_positions": initial,
                              "parent_radius_before_growth": 1., "parent_radius_at_division": 2 ** (1 / 3),
                              "daughter_radii": [1., 1.], "supplied_cell_volume": UNIT_VOLUME})
        before = len(ids)
        ids = next_ids
        positions, residual = _relax(np.asarray(next_positions), cfg, cfg.relaxation_steps)
        ledger.append({"round": round_, "before": before, "divisions": to_divide, "after": len(ids),
                       "supplied_cell_volume": to_divide * UNIT_VOLUME,
                       "living_cell_volume": len(ids) * UNIT_VOLUME,
                       "relaxation_iterations": cfg.relaxation_steps, "last_displacement": residual})
        snapshots.append({"round": round_, "count": len(ids), "cell_ids": list(ids),
                          "positions": positions.tolist(), "radii": [1.] * len(ids)})
    positions, residual = _relax(positions, cfg, cfg.final_relaxation_steps)
    snapshots.append({"round": round_, "phase": "final_relaxation", "count": len(ids), "cell_ids": list(ids),
                      "positions": positions.tolist(), "radii": [1.] * len(ids),
                      "relaxation_iterations": cfg.final_relaxation_steps, "last_displacement": residual})
    return ids, positions, history, divisions, snapshots, ledger


def _assign_tissues(ids, positions, history, source):
    """New spatial differentiation rules, not tissue identities from the old run."""
    count = len(ids)
    roles = [set() for _ in ids]
    neurons = source["neuron_count"] or 0
    # Neural cells condense around an anterior internal organizer; sensory
    # patches are selected on exterior regions using centers actually grown.
    span = np.ptp(positions, axis=0)
    neural_center = np.array([span[0] * .12, 0., 0.])
    neural = np.argsort(np.linalg.norm(positions - neural_center, axis=1), kind="stable")[:neurons]
    for index in neural:
        roles[index].add("neural")
    nearest = cKDTree(positions)
    # Low contact degree is an explicit approximation to exterior exposure.
    degree = np.array([len(nearest.query_ball_point(p, 2.2)) - 1 for p in positions])
    surface_penalty = degree * .15
    high, low = positions.max(axis=0), positions.min(axis=0)
    targets = {"eyes": np.array([high[0], high[1] * .7, 0.]),
               "ears": np.array([high[0] * .6, low[1], 0.]),
               "smell": np.array([high[0], 0., high[2] * .3]),
               "touch": np.array([0., 0., low[2]])}
    for sense, level in source["sensor_levels"].items():
        if not level:
            continue
        allocation = min(count, max(1, int(math.ceil(count * .01 * level))))
        rank = np.argsort(np.linalg.norm(positions - targets[sense], axis=1) + surface_penalty, kind="stable")[:allocation]
        for index in rank:
            roles[index].add(sense)
    if source["voice_level"]:
        target = np.array([high[0] * .6, 0., low[2] * .5])
        rank = np.argsort(np.linalg.norm(positions - target, axis=1), kind="stable")[:max(1, count // 40)]
        for index in rank:
            roles[index].add("voice")
    if (source["traits"].get("speed") or 0) > 0:
        rank = np.argsort(positions[:, 2], kind="stable")[:max(1, count // 5)]
        for index in rank:
            roles[index].add("contractile")
    sensory_names = {"eyes", "ears", "smell", "touch", "voice"}
    names = ["structural"]
    if neurons:
        names.append("neural")
    if any(r & sensory_names for r in roles):
        names.append("sensory")
    if any("contractile" in r for r in roles):
        names.append("contractile")
    names = names[:source["cell_types"]]
    cells = []
    for index, cell_id in enumerate(ids):
        role = "neural" if "neural" in roles[index] else "sensory" if roles[index] & sensory_names else "contractile" if "contractile" in roles[index] else "structural"
        # When types are scarce, retain functional roles on shared structural
        # tissue rather than silently inventing extra source cell types.
        if role not in names:
            role = "structural"
        cells.append({"id": cell_id, "parent_id": history[cell_id]["parent_id"],
                      "generation": history[cell_id]["generation"], "position": positions[index].tolist(),
                      "radius": 1., "tissue_id": names.index(role), "tissue_role": role,
                      "roles": sorted(roles[index]) or ["structural"]})
    # Preserve the exact recorded type count. Macro functional roles need not
    # exhaust the source's distinct cell types: split the largest compartment
    # spatially into unnamed subtypes, without inventing new organ functions.
    names = [name for name in names if any(c["tissue_role"] == name for c in cells)]
    for cell in cells:
        cell["tissue_id"] = names.index(cell["tissue_role"])
    members = [[i for i, cell in enumerate(cells) if cell["tissue_id"] == tissue_id]
               for tissue_id in range(len(names))]
    while len(names) < source["cell_types"]:
        donor = max(range(len(names)), key=lambda i: (len(members[i]), -i))
        indices = members[donor]
        axis = int(np.argmax(np.ptp(positions[indices], axis=0)))
        ordered = sorted(indices, key=lambda i: (positions[i, axis], cells[i]["id"]))
        cut = len(ordered) // 2
        for index in ordered[cut:]:
            cells[index]["tissue_id"] = len(names)
        members[donor] = ordered[:cut]
        members.append(ordered[cut:])
        names.append(names[donor])
    tissues = [{"id": i, "role": name, "name": f"{name}_subtype_{i}",
                "assumed_identity": True, "source_cell_type_identity": None,
                "cell_ids": [cells[index]["id"] for index in sorted(members[i])]} for i, name in enumerate(names)]
    return cells, tissues


def _contacts(cells, gap):
    positions = np.array([cell["position"] for cell in cells], dtype=float)
    contacts = []
    for a, b in _pairs(positions, 2. + gap):
        distance = float(np.linalg.norm(positions[b] - positions[a]))
        contacts.append({"a": cells[int(a)]["id"], "b": cells[int(b)]["id"],
                         "distance": distance, "overlap": max(0., 2. - distance)})
    return contacts


def _neural_network(cells, neighbors):
    neurons = [cell for cell in cells if "neural" in cell["roles"]]
    edges = {}
    if len(neurons) > 1:
        positions = np.array([c["position"] for c in neurons])
        for i in range(len(neurons)):
            distance = np.linalg.norm(positions - positions[i], axis=1)
            order = np.argsort(distance, kind="stable")
            for j in [int(j) for j in order if j != i][:neighbors]:
                a, b = sorted((neurons[i]["id"], neurons[j]["id"]))
                edges[(a, b)] = {"a": a, "b": b, "distance": float(distance[j])}
    return {"assumed": True, "trained": False, "functional_activity_simulated": False,
            "rule": f"undirected union of {neighbors} nearest neural-cell neighbors; no learned synapses",
            "neuron_cell_ids": [c["id"] for c in neurons], "edges": [edges[key] for key in sorted(edges)]}


def _components(ids, contacts):
    adjacency = {cell_id: [] for cell_id in ids}
    for contact in contacts:
        adjacency[contact["a"]].append(contact["b"])
        adjacency[contact["b"]].append(contact["a"])
    unseen, components = set(ids), []
    while unseen:
        stack = [min(unseen)]
        unseen.remove(stack[0])
        size = 0
        while stack:
            current = stack.pop()
            size += 1
            for neighbor in adjacency[current]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        components.append(size)
    return sorted(components, reverse=True)


def validate_anatomy(anatomy) -> dict:
    """Check stored counts, division history and actual spatial distances.

    Returns ``{ok, errors, statistics}``; it never repairs geometry or trusts the
    saved validation report. Connectivity means within the declared adhesion
    contact gap, not that all soft spheres have mathematical surface tangency.
    """
    errors = []
    stats = {}
    try:
        json.dumps(anatomy, allow_nan=False)
        if anatomy.get("schema_version") != SCHEMA_VERSION or anatomy.get("algorithm_version") != ALGORITHM_VERSION:
            errors.append("unknown anatomy schema or algorithm version")
        cfg = _config(anatomy["config"])
        source = anatomy["source"]
        cells, history, divisions = anatomy["cells"], anatomy["cell_history"], anatomy["divisions"]
        target = _integer(source["body_size"], "source body size", 1, cfg.max_cells)
        if len(cells) != target:
            errors.append("living cell count differs from exact source body_size")
        ids = [c["id"] for c in cells]
        if len(set(ids)) != len(ids):
            errors.append("duplicate living cell IDs")
        all_ids = [c["id"] for c in history]
        if all_ids != list(range(2 * target - 1)):
            errors.append("binary division cell-history IDs are incomplete or out of order")
        if len(divisions) != target - 1:
            errors.append("binary division count does not reach target")
        for record in history:
            parent = record["parent_id"]
            if record["id"] == 0:
                if parent is not None or record["generation"] != 0 or record["born_round"] != 0:
                    errors.append("invalid founder history")
            elif not isinstance(parent, int) or parent < 0 or parent >= record["id"]:
                errors.append("non-acyclic or missing historical parent")
            elif record["generation"] != history[parent]["generation"] + 1:
                errors.append("historical generation differs from parent")
        live = {0}
        for division in divisions:
            parent = division["parent_id"]
            daughters = division["daughter_ids"]
            if parent not in live or len(daughters) != 2 or len(set(daughters)) != 2:
                errors.append("invalid active-parent binary division")
                continue
            live.remove(parent)
            if history[parent]["divided_round"] != division["round"]:
                errors.append("parent retirement differs from division round")
            for child in daughters:
                if child <= parent or child >= len(history) or history[child]["parent_id"] != parent or child in live:
                    errors.append("invalid division ancestry")
                elif history[child]["born_round"] != division["round"]:
                    errors.append("daughter birth differs from division round")
                live.add(child)
            if not math.isclose(division["parent_radius_at_division"] ** 3, sum(r ** 3 for r in division["daughter_radii"]), rel_tol=1e-12):
                errors.append("division volume is not conserved after supplied growth")
        if live != set(ids):
            errors.append("living cell IDs differ from division ledger")
        if any(history[cell_id]["divided_round"] is not None for cell_id in ids if 0 <= cell_id < len(history)):
            errors.append("retired cell appears in final living population")
        for cell in cells:
            if cell["id"] >= len(history) or cell["parent_id"] != history[cell["id"]]["parent_id"]:
                errors.append("living cell ancestry differs from history")
            if not math.isclose(_finite(cell["radius"], "cell radius", .01), 1., abs_tol=1e-12):
                errors.append("this model requires mature radius one")
        positions = np.asarray([c["position"] for c in cells], dtype=float)
        if positions.shape != (len(cells), 3) or not np.isfinite(positions).all():
            errors.append("invalid 3D cell positions")
            return {"ok": False, "errors": errors, "statistics": stats}
        actual = _contacts(cells, cfg.contact_gap)
        actual_edges = {(e["a"], e["b"]): e for e in actual}
        supplied = anatomy["contacts"]
        supplied_edges = {(e["a"], e["b"]): e for e in supplied}
        if len(supplied_edges) != len(supplied) or set(actual_edges) != set(supplied_edges):
            errors.append("contact graph differs from measured 3D neighbors")
        for key in actual_edges.keys() & supplied_edges.keys():
            if any(not math.isclose(actual_edges[key][name], supplied_edges[key][name], rel_tol=1e-9, abs_tol=1e-9)
                   for name in ("distance", "overlap")):
                errors.append("contact distance or overlap is incorrect")
                break
        components = _components(ids, actual)
        intersection_components = _components(ids, [edge for edge in actual if edge["distance"] <= 2.])
        overlap = max((e["overlap"] / 2 for e in actual), default=0.)
        extent = np.ptp(positions, axis=0)
        stats.update(cell_count=len(cells), contact_count=len(actual), components=components,
                     sphere_intersection_components=intersection_components,
                     max_overlap_fraction=overlap, center_extent=extent.tolist())
        if len(components) != 1:
            errors.append("grown tissue is disconnected within declared contact gap")
        if overlap > cfg.max_overlap_fraction + 1e-9:
            errors.append("excessive soft-cell interpenetration")
        if len(cells) > 1 and float(np.linalg.norm(extent)) <= 1e-6:
            errors.append("zero spatial extent")
        neural = {c["id"] for c in cells if "neural" in c["roles"]}
        if source["neuron_count"] is not None and len(neural) != source["neuron_count"]:
            errors.append("neural-cell count differs from source")
        if len({c["tissue_id"] for c in cells}) != source["cell_types"]:
            errors.append("tissue type count differs from exact source cell_types")
        tissues = {t["id"]: t for t in anatomy["tissues"]}
        if len(tissues) != len(anatomy["tissues"]):
            errors.append("duplicate tissue identities")
        for tissue_id, tissue in tissues.items():
            assigned = [c["id"] for c in cells if c["tissue_id"] == tissue_id]
            if tissue["cell_ids"] != assigned:
                errors.append("tissue membership differs from cell assignments")
        for cell in cells:
            if cell["tissue_id"] not in tissues or tissues[cell["tissue_id"]]["role"] != cell["tissue_role"]:
                errors.append("cell tissue assignment differs from tissue table")
                break
        for sense, level in source["sensor_levels"].items():
            if not level and any(sense in c["roles"] for c in cells):
                errors.append(f"unavailable {sense} sensory tissue was invented")
        if not source["voice_level"] and any("voice" in c["roles"] for c in cells):
            errors.append("unavailable voice tissue was invented")
        if set(anatomy["neural_network"]["neuron_cell_ids"]) != neural:
            errors.append("neural network nodes differ from neural cells")
        by_id = {c["id"]: c for c in cells}
        for edge in anatomy["neural_network"]["edges"]:
            if edge["a"] not in neural or edge["b"] not in neural or edge["a"] == edge["b"]:
                errors.append("invalid neural connection")
                break
            measured = math.dist(by_id[edge["a"]]["position"], by_id[edge["b"]]["position"])
            if not math.isclose(measured, edge["distance"], rel_tol=1e-9, abs_tol=1e-9):
                errors.append("neural connection distance is incorrect")
                break
        supplied_volume = sum(row["supplied_cell_volume"] for row in anatomy["growth_ledger"])
        if not math.isclose(UNIT_VOLUME + supplied_volume, target * UNIT_VOLUME, rel_tol=1e-12):
            errors.append("external growth volume ledger does not close")
        count_before = 1
        for row in anatomy["growth_ledger"]:
            if row["before"] != count_before or row["after"] != row["before"] + row["divisions"] \
                    or not math.isclose(row["supplied_cell_volume"], row["divisions"] * UNIT_VOLUME, rel_tol=1e-12):
                errors.append("per-round growth ledger is inconsistent")
            count_before = row["after"]
        if count_before != target:
            errors.append("growth ledger does not end at target cell count")
        for snapshot in anatomy["growth_snapshots"]:
            if snapshot["count"] != len(snapshot["cell_ids"]) or len(snapshot["positions"]) != snapshot["count"] \
                    or len(snapshot["radii"]) != snapshot["count"] or len(set(snapshot["cell_ids"])) != snapshot["count"]:
                errors.append("growth snapshot count or identities are inconsistent")
            if np.asarray(snapshot["positions"]).shape != (snapshot["count"], 3):
                errors.append("growth snapshot is not three dimensional")
        final = anatomy["growth_snapshots"][-1]
        if final["cell_ids"] != ids or not np.array_equal(np.asarray(final["positions"]), positions):
            errors.append("final stored growth snapshot differs from final cells")
        stats["neural_cell_count"] = len(neural)
        stats["tissue_type_count"] = len({c["tissue_id"] for c in cells})
    except (KeyError, TypeError, ValueError, IndexError, OverflowError) as exc:
        errors.append(f"malformed anatomy: {exc}")
    return {"ok": not errors, "errors": errors, "statistics": stats}


def build_anatomy(phenotype, *, seed=0, config=None) -> dict:
    """Grow an exact saved body-cell count, or fail clearly if it cannot validate."""
    cfg = _config(config)
    seed = _integer(seed, "seed", 0, 2 ** 64 - 1)
    source = _source(phenotype, cfg)
    ids, positions, history, divisions, snapshots, ledger = _growth(source["body_size"], seed, cfg)
    cells, tissues = _assign_tissues(ids, positions, history, source)
    result = {"schema_version": SCHEMA_VERSION, "algorithm_version": ALGORITHM_VERSION,
        "seed": seed, "units": "mature cell radius", "source": source, "config": asdict(cfg),
        "assumptions": [
            "This is newly simulated developmental anatomy, not geometry stored by the earlier evolution model.",
            "A founder grows through binary division; each parent retires and creates two new daughter IDs.",
            "Before dividing a cell receives one mature cell volume of material, doubles its volume, then divides into two radius-one daughters. This is an external volume supply, not a mass/energy conservation claim.",
            "Pair repulsion, short-range adhesion and an anisotropic confining field govern centers; cells are soft spheres with a measured overlap bound.",
            "Spatial tissue compartments and sensory patches are new differentiation assumptions. Sensor levels set allocated cell fractions, never organ counts.",
            "A sensory role can coexist with another functional role. The exact source cell-type count is preserved by unnamed spatial subtypes where functional compartments are fewer; original subtype identities were not recorded.",
            "Nearest-neighbor neural edges are assumed connectivity, not trained synapses or simulated neural activity.",
            "These IDs describe this development model only; they are not ancestry or society citizens from the original simulation.",
        ], "cells": cells, "cell_history": history, "divisions": divisions,
        "growth_snapshots": snapshots, "growth_ledger": ledger,
        "tissues": tissues, "contacts": _contacts(cells, cfg.contact_gap),
        "contact_definition": "center distance <= sum of mature radii + contact_gap; adhesion-scale contact, not exact surface tangency",
        "neural_network": _neural_network(cells, cfg.neural_neighbors)}
    result["validation"] = validate_anatomy(result)
    if not result["validation"]["ok"]:
        raise ValueError("morphogenesis did not validate: " + "; ".join(result["validation"]["errors"]))
    result["content_sha256"] = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return result
