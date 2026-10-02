"""Exact source budgets and measurable geometry for the new cell model."""
import copy
import json
import math
import os
import random
import subprocess
import sys

import numpy as np
import pytest

from haishool.evo.anatomy import (ALGORITHM_VERSION, MAX_CELLS, SCHEMA_VERSION, UNIT_VOLUME,
                                 build_anatomy, validate_anatomy)


def phenotype(count=256, types=7, neurons=64):
    genes = {"smell": 1, "eyes": 0, "touch": 0, "ears": 2, "neuron_gene": 6, "speed": 4, "voice": 1}
    return {"id": "saved-world85-phenotype", "kind": "senses_agent", "genotype_id": "saved-genotype",
            "world_seed": 85, "planet_id": 1, "snapshot_step": 59,
            "traits": {"genes": genes, "body_size": count, "body_size_unit": "cells", "cell_types": types,
                       "neuron_count": neurons, "speed": 4,
                       "sensors": {key: {"level": genes[key], "present": genes[key] > 0}
                                   for key in ("smell", "eyes", "touch", "ears")},
                       "voice": {"level": 1, "present": True}},
            "geometry": {"forbidden_old_glyph": [1000, 2000, 3000]}}


@pytest.fixture(scope="module")
def sample():
    return build_anatomy(phenotype(), seed=85)


def test_exact_sample_source_counts_and_finite_connected_anatomy(sample):
    assert sample["schema_version"] == SCHEMA_VERSION
    assert sample["algorithm_version"] == ALGORITHM_VERSION
    assert sample["source"]["world_seed"] == 85
    assert sample["units"] == "mature cell radius"
    assert len(sample["cells"]) == 256
    assert sum("neural" in c["roles"] for c in sample["cells"]) == 64
    assert len(sample["tissues"]) == len({c["tissue_id"] for c in sample["cells"]}) == 7
    assert all(c["radius"] == 1 and len(c["position"]) == 3 for c in sample["cells"])
    report = validate_anatomy(sample)
    assert report["ok"], report
    assert report["statistics"]["components"] == [256]
    assert report["statistics"]["sphere_intersection_components"] == [256]
    assert report["statistics"]["max_overlap_fraction"] < .04
    assert min(report["statistics"]["center_extent"]) > 5
    json.dumps(sample, allow_nan=False)


def test_binary_divisions_retire_parents_and_keep_acyclic_cell_history(sample):
    assert len(sample["divisions"]) == 255
    assert len(sample["cell_history"]) == 511
    live = {0}
    for division in sample["divisions"]:
        parent = division["parent_id"]
        daughters = division["daughter_ids"]
        assert parent in live and len(daughters) == 2 and len(set(daughters)) == 2
        live.remove(parent)
        for child in daughters:
            assert child > parent
            history = sample["cell_history"][child]
            assert history["parent_id"] == parent
            assert history["generation"] == sample["cell_history"][parent]["generation"] + 1
        live.update(daughters)
    assert live == {c["id"] for c in sample["cells"]}
    assert all(sample["cell_history"][cell_id]["divided_round"] is None for cell_id in live)
    assert [s["count"] for s in sample["growth_snapshots"]] == [1, 2, 4, 8, 16, 32, 64, 128, 256, 256]


def test_volume_supply_is_explicit_and_closes_without_energy_claim(sample):
    assert sum(row["divisions"] for row in sample["growth_ledger"]) == 255
    supplied = sum(row["supplied_cell_volume"] for row in sample["growth_ledger"])
    assert UNIT_VOLUME + supplied == pytest.approx(256 * UNIT_VOLUME)
    for event in sample["divisions"]:
        before = event["parent_radius_before_growth"] ** 3 * UNIT_VOLUME
        after_growth = event["parent_radius_at_division"] ** 3 * UNIT_VOLUME
        daughters = sum(r ** 3 * UNIT_VOLUME for r in event["daughter_radii"])
        assert before + event["supplied_cell_volume"] == pytest.approx(after_growth)
        assert daughters == pytest.approx(after_growth)
    assert any("not a mass/energy conservation claim" in a for a in sample["assumptions"])


def test_eye_zero_has_no_ocular_tissue_or_invented_organ(sample):
    assert sample["source"]["sensor_levels"]["eyes"] == 0
    assert all("eyes" not in c["roles"] for c in sample["cells"])
    assert all("touch" not in c["roles"] for c in sample["cells"])
    assert sum("smell" in c["roles"] for c in sample["cells"]) > 0
    assert sum("ears" in c["roles"] for c in sample["cells"]) > 0
    assert "geometry" not in sample and "geometry" not in sample["source"]
    assert all(t["assumed_identity"] and t["source_cell_type_identity"] is None for t in sample["tissues"])


def test_no_global_randomness_source_mutation_or_use_of_old_glyph(sample):
    source = phenotype()
    original = copy.deepcopy(source)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    result = build_anatomy(source, seed=85)
    assert source == original
    assert python_state == random.getstate()
    after = np.random.get_state()
    assert numpy_state[0] == after[0]
    np.testing.assert_array_equal(numpy_state[1], after[1])
    assert numpy_state[2:] == after[2:]
    assert result == sample
    source["geometry"] = {"completely_different_glyph": True}
    assert build_anatomy(source, seed=85) == sample


def test_actual_contacts_and_assumed_neural_neighbors(sample):
    by_id = {c["id"]: c for c in sample["cells"]}
    for edge in sample["contacts"]:
        measured = math.dist(by_id[edge["a"]]["position"], by_id[edge["b"]]["position"])
        assert edge["distance"] == pytest.approx(measured)
        assert measured <= 2 + sample["config"]["contact_gap"]
        assert edge["overlap"] == pytest.approx(max(0, 2 - measured))
    network = sample["neural_network"]
    assert network["assumed"] is True and network["trained"] is False
    assert network["functional_activity_simulated"] is False
    assert len(network["neuron_cell_ids"]) == 64
    for edge in network["edges"]:
        assert edge["a"] in network["neuron_cell_ids"] and edge["b"] in network["neuron_cell_ids"]
        assert edge["distance"] == pytest.approx(math.dist(by_id[edge["a"]]["position"], by_id[edge["b"]]["position"]))


@pytest.mark.parametrize("count,types,neurons", [(1, 1, 0), (7, 3, 2), (23, 7, 6), (32, 32, 8)])
def test_partial_division_rounds_and_exact_cell_type_budget(count, types, neurons):
    result = build_anatomy(phenotype(count, types, neurons), seed=4)
    assert len(result["cells"]) == count
    assert len(result["tissues"]) == types
    assert sum("neural" in c["roles"] for c in result["cells"]) == neurons
    assert validate_anatomy(result)["ok"]


def test_zero_voice_and_unknown_body_lineage_traits():
    source = {"id": "lineage", "kind": "body_lineage", "traits": {
        "body_size": 16, "cell_types": 2, "neuron_count": None, "genes": None,
        "sensors": None, "voice": None, "speed": None}}
    result = build_anatomy(source, seed=2)
    assert result["source"]["neuron_count"] is None
    assert result["source"]["sensor_levels"]["eyes"] is None
    assert all("voice" not in c["roles"] and "eyes" not in c["roles"] for c in result["cells"])
    assert not result["neural_network"]["neuron_cell_ids"]


@pytest.mark.parametrize("change", [
    {"body_size": MAX_CELLS + 1}, {"body_size": 0}, {"body_size": True},
    {"cell_types": 257}, {"cell_types": 0}, {"neuron_count": 257}, {"neuron_count": -1},
    {"body_size_unit": "metres"}, {"body_size": float("nan")},
])
def test_invalid_source_budget_rejected_without_downsampling(change):
    source = phenotype()
    source["traits"].update(change)
    with pytest.raises(ValueError):
        build_anatomy(source)


@pytest.mark.parametrize("config", [{"max_cells": 128}, {"max_cells": 5000},
                                    {"growth_axes": [0, 1, 1]}, {"repulsion": float("inf")},
                                    {"unknown_option": True}, {"relaxation_steps": 0}])
def test_invalid_configuration_fails_clearly(config):
    with pytest.raises(ValueError):
        build_anatomy(phenotype(), config=config)


@pytest.mark.parametrize("corruption", ["positions", "contact", "neurons", "eyes", "tissue", "parent", "ledger", "snapshot"])
def test_validator_recomputes_geometry_and_rejects_corruption(sample, corruption):
    bad = copy.deepcopy(sample)
    if corruption == "positions":
        bad["cells"][0]["position"] = [float("nan"), 0, 0]
    elif corruption == "contact":
        bad["contacts"][0]["distance"] += 1
    elif corruption == "neurons":
        next(c for c in bad["cells"] if "neural" in c["roles"])["roles"].remove("neural")
    elif corruption == "eyes":
        bad["cells"][0]["roles"].append("eyes")
    elif corruption == "tissue":
        bad["tissues"][0]["cell_ids"].pop()
    elif corruption == "parent":
        bad["cell_history"][1]["parent_id"] = 1
    elif corruption == "ledger":
        bad["growth_ledger"][0]["supplied_cell_volume"] = 0
    else:
        bad["growth_snapshots"][-1]["positions"][0][0] += 1
    assert not validate_anatomy(bad)["ok"]


def test_mechanical_growth_is_seeded_and_has_a_nonzero_three_dimensional_shape():
    a = build_anatomy(phenotype(32, 4, 8), seed=1)
    b = build_anatomy(phenotype(32, 4, 8), seed=2)
    assert a["cells"] != b["cells"]
    assert all(x > 1 for x in a["validation"]["statistics"]["center_extent"])
    assert a["validation"]["statistics"]["max_overlap_fraction"] < .08


def test_under_relaxed_growth_fails_instead_of_returning_overlapping_fallback():
    with pytest.raises(ValueError, match="interpenetration"):
        build_anatomy(phenotype(32, 4, 8), config={"relaxation_steps": 1, "final_relaxation_steps": 1})


def test_cross_process_hash_is_stable():
    code = '''from haishool.evo.anatomy import build_anatomy
p={"id":"example","kind":"body_lineage","traits":{"body_size":23,"cell_types":3,"neuron_count":None}}
print(build_anatomy(p,seed=13)["content_sha256"])'''
    hashes = [subprocess.check_output([sys.executable, "-c", code], text=True,
                 env={**os.environ, "PYTHONHASHSEED": str(seed)}).strip() for seed in (1, 89)]
    assert hashes[0] == hashes[1]
