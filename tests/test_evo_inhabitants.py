"""Observer contracts, using saved synthetic snapshots rather than long worlds."""
import copy
import hashlib
import json
import os
import random
import subprocess
import sys

import pytest

from haishool.cosmos import Rollout
from haishool.evo import bodies, senses, world7
from haishool.evo.inhabitants import GENE_NAMES, MAPPING_VERSION, SCHEMA_VERSION, export_inhabitants


BLIND = (0, 0, 0, 0, 0, 1, 0)
SENSING = (2, 3, 1, 2, 3, 4, 1)


def fixture_world():
    source = senses.SensesRollout("senses", 101, {"body_size": 256, "cell_types": 5, "light": .7},
        [{"population": 3, "births": 4}, {"population": 2, "births": 1}], {"population": 2},
        [[BLIND, BLIND, SENSING], [BLIND, SENSING]])
    rejected = senses.SensesRollout("senses", 100, {"body_size": 16, "cell_types": 1},
                                   [{"population": 0}], {"population": 0}, [[]])
    lineage = {"id": 7, "parent": 2, "guild": "grazer", "size": 256,
               "adhesion": .875, "cell_types": 5, "level": 2, "biomass": 12.5}
    body = bodies.BodiesRollout("bodies", 99, {"light": 100.}, [{"lineages": 1}, {"lineages": 1}],
                               {"max_size": 256}, [[lineage], [{**lineage, "biomass": 15.}]])
    society = Rollout("society", 102, {"groups": 2}, [{"population": 1000}], {"population": 1000, "groups": 2})
    return world7.WorldRollout("world7", 85, {"formation_time": 4., "spin": .1},
        [{"era": "surface_1", "orbit": 1.1}, {"era": "bodies_1", "kept": 1},
         {"era": "senses_1", "kept": 2}, {"era": "society_1"}], {"outcome": "states"},
        {"bodies_1": body, "senses_1": source, "society_1": society},
        {"bodies_1": [body], "senses_1": [rejected, source]})


def test_schema_source_attempts_and_no_mutation_or_random_draws(monkeypatch):
    world = fixture_world()
    before = copy.deepcopy(world)
    state = random.getstate()
    def forbidden(*args, **kwargs):
        raise AssertionError("observer must not draw random numbers or run/replay")
    monkeypatch.setattr(random, "Random", forbidden)
    monkeypatch.setattr(random, "random", forbidden)
    monkeypatch.setattr(world7, "run", forbidden)
    result = export_inhabitants(world)
    assert result == export_inhabitants(world)
    assert world == before
    assert random.getstate() == state
    assert result["schema_version"] == SCHEMA_VERSION
    assert result["mapping_version"] == MAPPING_VERSION
    assert result["source"]["world_seed"] == 85
    assert result["source"]["params"] == world.params
    chosen = result["planets"][0]["source"]["senses"]
    assert chosen["selected_attempt"] == 2
    assert chosen["seed"] == 101
    assert [a["selected"] for a in chosen["attempts"]] == [False, True]
    json.dumps(result, allow_nan=False)
    result["source"]["params"]["spin"] = 99
    result["phenotypes"][next(iter(result["phenotypes"]))]["traits"]["body_size"] = 9
    assert world == before


def test_individuals_groups_genes_and_snapshot_alignment():
    result = export_inhabitants(fixture_world())
    first, second = result["planets"][0]["snapshots"]
    assert first["kind"] == "agents" and first["stage"] == "senses"
    assert first["generation"] == 0 and second["generation"] == 5
    assert first["timing"] == "input population before reproduction"
    assert len(first["agents"]) == first["metrics"]["population"] == 3
    assert sum(g["count"] for g in first["groups"]) == 3
    assert first["groups"][0]["count"] == 2
    assert first["groups"][0]["indices"] == [0, 1]
    assert first["agents"][0]["phenotype_id"] == first["agents"][1]["phenotype_id"]
    assert first["agents"][0]["id"] != first["agents"][1]["id"]
    assert first["agents"][0]["id"] != second["agents"][0]["id"]
    assert "/attempt-2/snapshot-0/agent-0" in first["agents"][0]["id"]
    assert "parent" not in first["agents"][0]
    genotype = result["genotypes"][first["agents"][2]["genotype_id"]]
    assert genotype["gene_names"] == list(GENE_NAMES)
    assert genotype["genes"] == list(SENSING)
    assert len(genotype["sha256"]) == 64
    assert len(result["genotypes"]) == 2


def test_absent_sensors_voice_and_correct_neuron_count():
    result = export_inhabitants(fixture_world())
    agents = result["planets"][0]["snapshots"][0]["agents"]
    blind = result["phenotypes"][agents[0]["phenotype_id"]]
    seeing = result["phenotypes"][agents[2]["phenotype_id"]]
    assert blind["traits"]["neuron_count"] == 0
    assert blind["traits"]["sensors"]["eyes"] == {"level": 0, "present": False, "meaning": "model sensor level, not organ count"}
    assert blind["traits"]["voice"]["present"] is False
    assert not {"eyes", "hearing", "smell", "touch", "calling", "neurons"}.intersection(
        shape["feature"] for shape in blind["geometry"]["shapes"])
    assert seeing["traits"]["neuron_count"] == senses.neuron_count(3) == 8
    assert seeing["traits"]["sensors"]["eyes"]["level"] == 3
    assert sum(s["feature"] == "eyes" for s in seeing["geometry"]["shapes"]) == 1  # emblem, never three physical eyes
    assert seeing["traits"]["body_size"] == 256
    assert seeing["traits"]["body_size_unit"] == "cells"
    assert seeing["geometry"]["encoding_only"] is True
    assert seeing["geometry"]["units"] == "schematic display units"


def test_body_lineage_is_not_an_agent_and_keeps_real_identity():
    world = fixture_world()
    world.levels.pop("senses_1")
    result = export_inhabitants(world)
    planet = result["planets"][0]
    assert planet["default_source"] == "bodies"
    first, second = planet["snapshots"]
    assert first["kind"] == "lineages" and first["agents"] == []
    assert first["generation"] is None and second["tick"] == bodies.TICKS
    one, two = first["lineages"][0], second["lineages"][0]
    assert one["id"] == two["id"]
    assert one["source_lineage_id"] == 7 and one["parent_lineage_id"] == 2
    assert one["raw"]["biomass"] == 12.5
    assert first["groups"][0]["count"] == 1 and first["groups"][0]["count_unit"] == "lineages"
    assert one["genotype_id"] is None
    assert result["phenotypes"][one["phenotype_id"]]["traits"]["neuron_count"] is None
    assert result["phenotypes"][one["phenotype_id"]]["traits"]["sensors"] is None
    assert planet["snapshots"] == planet["body_lineages"]
    planet["snapshots"][0]["lineages"][0]["raw"]["size"] = 1
    assert planet["body_lineages"][0]["lineages"][0]["raw"]["size"] == 256  # no output alias either


def test_society_context_does_not_invent_people():
    result = export_inhabitants(fixture_world())
    planet = result["planets"][0]
    society = planet["society_context"]
    assert society["source"]["summary"]["population"] == 1000
    assert society["individuals_available"] is False
    assert society["rendered_specimens_are_society_citizens"] is False
    assert len(planet["snapshots"][0]["agents"]) == 3
    assert planet["representation"] == "upstream_biological_specimens"


def test_step_subset_preserves_original_ids_and_validates_bounds():
    world = fixture_world()
    all_steps = export_inhabitants(world)
    subset = export_inhabitants(world, steps=[1, 1])
    assert subset["planets"][0]["snapshots"][0] == all_steps["planets"][0]["snapshots"][1]
    assert subset["source"]["selected_steps"] == [1]
    with pytest.raises(ValueError, match="outside"):
        export_inhabitants(world, steps=[2])


@pytest.mark.parametrize("steps", [1, True, "1", [], [True], [-1], [1.5], [None]])
def test_invalid_steps(steps):
    with pytest.raises(ValueError):
        export_inhabitants(fixture_world(), steps=steps)


def test_empty_and_extinct_worlds_are_explicit():
    world = world7.WorldRollout("world7", 1, {}, [{"era": "surface_1"}], {"outcome": "lifeless"})
    empty = export_inhabitants(world)
    assert empty["empty_reason"]
    assert empty["planets"][0]["reason"]
    assert empty["planets"][0]["snapshots"] == []
    world = fixture_world()
    world.levels.pop("bodies_1")
    for index in range(2):
        world.levels["senses_1"].genomes[index] = []
        world.levels["senses_1"].steps[index]["population"] = 0
    extinct = export_inhabitants(world)
    assert extinct["empty_reason"] and not extinct["phenotypes"]
    assert all(s["reason"] and not s["agents"] for s in extinct["planets"][0]["snapshots"])


def test_nonfinite_corrupt_and_mismatched_source_rejected():
    world = fixture_world()
    world.params["spin"] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        export_inhabitants(world)
    world = fixture_world()
    world.levels["senses_1"].steps[0]["population"] = 4
    with pytest.raises(ValueError, match="disagree"):
        export_inhabitants(world)
    world = fixture_world()
    world.steps[2]["kept"] = 1
    with pytest.raises(ValueError, match="selected attempt"):
        export_inhabitants(world)


def test_same_genotype_different_body_traits_has_separate_phenotype():
    world = fixture_world()
    second = copy.deepcopy(world.levels["senses_1"])
    second.params["body_size"] = 512
    world.levels["senses_2"] = second
    world.steps.append({"era": "senses_2"})
    result = export_inhabitants(world)
    a = result["planets"][0]["snapshots"][0]["agents"][0]
    b = result["planets"][1]["snapshots"][0]["agents"][0]
    assert a["genotype_id"] == b["genotype_id"]
    assert a["phenotype_id"] != b["phenotype_id"]
    assert a["id"] != b["id"]


def test_export_hash_stable_across_process_hash_seeds():
    code = '''import hashlib,json
from types import SimpleNamespace as N
from haishool.evo.inhabitants import export_inhabitants
s=N(sim="senses",seed=3,params={"body_size":256,"cell_types":4},summary={},steps=[{"population":2}],genomes=[[(0,0,0,0,0,1,0),(2,3,1,2,3,4,1)]])
w=N(sim="world7",seed=5,params={},summary={},steps=[{"era":"senses_1"}],levels={"senses_1":s},attempts={})
print(hashlib.sha256(json.dumps(export_inhabitants(w),sort_keys=True,separators=(",",":")).encode()).hexdigest())'''
    values = [subprocess.check_output([sys.executable, "-c", code], env={**os.environ, "PYTHONHASHSEED": str(seed)}, text=True).strip()
              for seed in (1, 99)]
    assert values[0] == values[1]


def test_population_scale_deduplicates_traits_instead_of_fabricating_agents():
    world = fixture_world()
    source = world.levels["senses_1"]
    source.genomes = [[BLIND] * 200 for _ in range(61)]
    source.steps = [{"population": 200} for _ in range(61)]
    result = export_inhabitants(world)
    snapshots = result["planets"][0]["snapshots"]
    assert sum(len(s["agents"]) for s in snapshots) == 12200
    assert len(result["genotypes"]) == 1
    assert all(len(s["groups"]) == 1 and s["groups"][0]["count"] == 200 for s in snapshots)
