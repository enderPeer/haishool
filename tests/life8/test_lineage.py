"""Lineage table, neutral markers, convention agreement and mapping persistence."""
import json
import random
from types import SimpleNamespace

import pytest

from haishool.life8 import lineage
from haishool.life8.config import Config
from haishool.life8.world import World

GENES = {"body_mass": 1., "speed": 1., "sensing": 1., "manipulation": 1., "metabolism": 1., "voice": 1, "hearing": 1}


@pytest.fixture(scope="module")
def regulated_run():
    """Seed 1, food_patches=30, 700 ticks: about 170 individuals, founders all dead by ~600."""
    return lineage.run_with_lineage(1, {"food_patches": 30}, 700)


def test_table_keeps_the_dead_and_every_parent_chain_reaches_a_founder(regulated_run):
    events, world, table = regulated_run
    births = [e for e in events if e["type"] == "birth"]
    deaths = [e for e in events if e["type"] == "death"]
    assert len(table.rows) == 24 + len(births) and len(deaths) > 50
    assert sorted(table.alive()) == sorted(world.agents)
    for identity, row in table.rows.items():
        chain = table.chain(identity)
        assert chain[-1] == row["founder"] and len(chain) == row["depth"] + 1
        assert table.rows[chain[-1]]["parent"] is None
    # The engine forgot the dead; the table still knows the parents of the living.
    for identity in world.agents:
        assert all(ancestor in table.rows for ancestor in table.chain(identity))
    dead_parents = {table.rows[i]["parent"] for i in world.agents} - set(world.agents) - {None}
    assert dead_parents and all(table.rows[p]["death_tick"] is not None for p in dead_parents)
    summary = table.summary()
    assert summary["max_depth"] >= 3 and summary["founders"] == 24


def test_table_survives_json_and_checkpoint_resume():
    config = Config(log="compact", food_patches=30)
    direct = World.create(seed=4, config=config)
    direct_table = lineage.LineageTable.from_world(direct)
    for _ in range(160):
        direct_table.record(direct, direct.step())
    split = World.create(seed=4, config=config)
    table = lineage.LineageTable.from_world(split)
    for _ in range(70):
        table.record(split, split.step())
    resumed = World.from_dict(json.loads(json.dumps(split.to_dict())))
    table = lineage.LineageTable.from_dict(json.loads(json.dumps(table.to_dict())))
    for _ in range(90):
        table.record(resumed, resumed.step())
    assert table.to_dict() == direct_table.to_dict()
    with pytest.raises(ValueError):
        lineage.LineageTable.from_dict({"schema": "other"})


def test_markers_are_inherited_with_mutation_and_independent_of_the_world(regulated_run):
    _, _, table = regulated_run
    copied = mutated = 0
    for row in table.rows.values():
        if row["parent"] is None:
            assert all(.75 <= m <= 1.25 for m in row["markers"])
            continue
        parent = table.rows[row["parent"]]["markers"]
        for mine, theirs in zip(row["markers"], parent):
            if mine == theirs:
                copied += 1
            else:
                mutated += 1
                assert abs(mine / theirs - 1) < .13  # exp(+-0.12)
    rate = mutated / (copied + mutated)
    assert .05 < rate < .11  # mutation_rate 0.08
    rebuilt = lineage.run_with_lineage(1, {"food_patches": 30}, 200)[2]
    assert all(rebuilt.rows[i]["markers"] == table.rows[i]["markers"] for i in rebuilt.rows)


def synthetic_table(founders=4, children=6):
    table = lineage.LineageTable(seed=0)
    for f in range(1, founders + 1):
        table._add_root(SimpleNamespace(id=f, parent_id=None, born_tick=0, anatomy=SimpleNamespace(to_dict=lambda: dict(GENES))))
    identity = founders
    for f in range(1, founders + 1):
        for _ in range(children):
            identity += 1
            table._add_child(identity, table.rows[f], 10, dict(GENES))
    return table


def test_trait_selection_is_judged_against_marker_drift():
    table = synthetic_table()
    for row in table.rows.values():
        if row["depth"]:
            row["anatomy"] = {**GENES, "body_mass": .6}
    for f in range(1, 5):
        table.rows[f]["death_tick"] = 50
    result = lineage.trait_selection(table, "body_mass")
    assert result["n_later"] == 24 and result["beyond_all_markers"]
    assert result["z"] < -5
    assert lineage.trait_selection(table, "speed")["log_change"] == 0


def calls(table, convention, ticks=range(0, 40), rng=None):
    rng = rng or random.Random(0)
    events = []
    for tick in ticks:
        for identity, row in table.rows.items():
            seen = rng.random() < .5
            symbol = convention(row["founder"], seen, rng)
            events.append({"tick": tick, "type": "signal", "agent": identity, "symbol": symbol,
                           "sender": {"food_direction": "e" if seen else "none", "energy_band": "middle"}})
    return events


def test_convention_agreement_separates_lineage_conventions_from_noise():
    table = synthetic_table()
    lineage_specific = calls(table, lambda f, seen, rng: (f + seen) % 4)
    result = lineage.convention_agreement(lineage_specific, table, reps=100)
    assert result["within"] > .95 and result["across"] < .5 and result["p_value"] < .05
    noise = calls(table, lambda f, seen, rng: rng.randrange(4))
    result = lineage.convention_agreement(noise, table, reps=100)
    assert abs(result["within"] - .25) < .05 and abs(result["difference"]) < .05 and result["p_value"] > .01


def test_inherited_habits_without_a_situation_mapping_are_not_a_convention():
    """Critic F2: lineages that share a symbol habit in every situation (inherited bias rows)
    agree far more within than across, but not about situations."""
    table = synthetic_table()
    habits = calls(table, lambda f, seen, rng: f % 4 if rng.random() < .7 else rng.randrange(4), rng=random.Random(3))
    result = lineage.convention_agreement(habits, table, reps=100)
    assert result["difference_raw"] > .2 and result["within"] > result["across"] + .2
    assert abs(result["difference"]) < .05 and result["p_value"] > .01
    assert result["within_blind"] == pytest.approx(result["within"], abs=.05)


def test_mapping_persistence_after_the_founders_died():
    table = synthetic_table(children=4)
    for f in range(1, 5):
        table.rows[f]["death_tick"] = 20
    keep = lambda f, seen, rng: (f + seen) % 4
    stable = calls(table, keep, ticks=range(40))
    result = lineage.mapping_persistence(stable, table, reps=100)
    assert result["lineages_measured"] == 4 and result["lineages_persisting"] == 4
    assert all(r["same_mode"] == 1 and r["cross_same_mode"] < 1 for r in result["lineages"])
    rng = random.Random(1)
    drift = calls(table, keep, ticks=range(20), rng=rng) + calls(table, lambda f, seen, rng: rng.randrange(4),
                                                                 ticks=range(20, 40), rng=rng)
    assert lineage.mapping_persistence(drift, table, reps=100)["lineages_persisting"] == 0


def test_real_world_lineages_share_no_convention(regulated_run):
    """Measured 2026-10-02: seed 1 within 0.250, across 0.252 (window 400-700), p 0.61."""
    events, _, table = regulated_run
    result = lineage.convention_agreement(events, table, start=400, reps=100, seed=1)
    assert result["senders"] > 30 and result["lineages"] >= 3
    assert abs(result["within"] - .25) < .05 and abs(result["difference"]) < .05 and result["p_value"] > .01
    persistence = lineage.mapping_persistence(events, table, reps=100, seed=1)
    assert persistence["lineages_measured"] >= 1 and persistence["lineages_persisting"] == 0


@pytest.mark.slow
def test_six_regulated_worlds_show_no_lineage_convention():
    """Measured 2026-10-02 (docs/life8/results/lineage-real-worlds.json, seeds 1-6, 1500 ticks,
    food_patches=30): within 0.246-0.255 vs across 0.249-0.254, p 0.30-0.83; 0/45 lineage
    mappings persisted after their founder died."""
    report = lineage.measure(range(1, 7), {"food_patches": 30}, 1500, reps=200, jobs=6)
    for row in report["runs"]:
        convention = row["convention_late_window"]
        assert abs(convention["difference"]) < .03 and convention["p_value"] > .01
        assert row["persistence"]["lineages_persisting"] == 0
