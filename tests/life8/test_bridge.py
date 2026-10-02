"""Round 7 -> round 8 bridge: world7 planets that reached bodies become living worlds."""
import json
import math
import os
from dataclasses import replace

import pytest

from haishool.life8 import bridge
from haishool.life8.checkpoint import digest
from haishool.life8.config import Config
from haishool.life8.world import World

#: world7 seeds that reach bodies (seed 6, 7: senses "sensing"; seed 65: senses "seeing")
SEEDS = (6, 7, 65)
SCAN = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "life8", "results", "bridge-scan-0-299.json")


@pytest.fixture(scope="module")
def records():
    return {seed: bridge._scan_record(seed) for seed in SEEDS}


def _vary(record, era, **values):
    """A copy of a scanned world with some values of one era of its planet changed."""
    k = record["planet"]
    steps = [dict(s, **values) if s["era"] == f"{era}_{k}" else dict(s) for s in record["steps"]]
    return {**record, "steps": steps}


def test_three_world7_seeds_give_valid_configs_and_run_200_ticks(records):
    for seed in SEEDS:
        record = records[seed]
        assert record is not None and "bodies" in record["links"]
        cfg, founders = bridge.world_config(record, Config(log="compact"))
        assert Config.from_dict(json.loads(json.dumps(cfg.to_dict()))) == cfg
        json.dumps(founders)  # JSON-ready
        assert sum(founders["objects"].values()) == cfg.initial_objects
        assert founders["ecology_hints"]["connected"] is False
        for gene, (low, high) in founders["anatomy"].items():
            assert 0 < low < high, gene
        for key in ("visibility", "action_cost", "base_metabolism", "food_regrowth", "body_mass", "manipulation",
                    "sensing", "voice", "hearing", "objects"):
            assert founders["sources"][key]
        world = bridge.create_world(seed, cfg, founders)
        for agent in world.agents.values():
            for gene, (low, high) in founders["anatomy"].items():
                assert low <= getattr(agent.anatomy, gene) <= high
        materials_made = sorted(o.material for o in world.objects.values())
        assert materials_made == sorted(m for m, c in founders["objects"].items() for _ in range(c))
        for _ in range(200):
            world.step(None)
            assert world.stop_reason is None, (seed, world.tick, world.stop_reason)
        assert world.tick == 200 and world.agents
        world.validate()


def test_bridged_world_continues_exactly_from_a_checkpoint(records):
    cfg, founders = bridge.world_config(records[65])
    direct = bridge.create_world(65, replace(cfg, log="compact"), founders)
    resumed = bridge.create_world(65, replace(cfg, log="compact"), founders)
    for _ in range(40):
        direct.step(None)
        resumed.step(None)
    resumed = World.from_dict(json.loads(json.dumps(resumed.to_dict())))
    for _ in range(40):
        direct.step(None)
        resumed.step(None)
    assert digest(direct.to_dict()) == digest(resumed.to_dict())


def test_bridge_draws_founders_from_its_own_rng(records):
    cfg, founders = bridge.world_config(records[6])
    assert bridge.create_world(6, cfg, founders).rng.getstate() == World.create(6, cfg).rng.getstate()


def test_fast_scan_is_the_prefix_of_the_world7_run(records):
    """world7 seed 6 reaches signals; the scan stops each climb after senses."""
    from haishool.evo import world7
    full = world7.run(6)
    fast = records[6]["steps"]  # bridge.scan_world(6)
    assert fast == [s for s in full.steps if s["era"].rsplit("_", 1)[0] not in ("signals", "society")]
    assert len(fast) < len(full.steps)  # signals ran in the full world and not in the scan
    # and the planet the bridge picks is the same on the full rollout
    assert bridge.planet_of(full)[:2] == bridge.planet_of({"seed": 6, "steps": fast})[:2]


def test_world_without_bodies_is_refused(records):
    record = records[65]
    below = [s for s in record["steps"] if not s["era"].startswith(("bodies", "senses"))]
    with pytest.raises(ValueError, match="did not reach bodies"):
        bridge.world_config({"seed": 65, "steps": below})
    multicellular = _vary(record, "bodies", result="multicellular")  # bodies' lower rung still holds
    assert "bodies" in bridge.planet_of(multicellular)[1]
    colonies = _vary(record, "bodies", result="colonies")
    with pytest.raises(ValueError, match="did not reach bodies"):
        bridge.world_config(colonies)


def test_more_light_gives_more_visibility_and_more_food(records):
    record = records[7]
    fluxes = (0.36, 0.5, 0.7, 0.9, 1.0)
    vis = [bridge.world_config(_vary(record, "surface", flux=f))[0].visibility for f in fluxes]
    assert all(a < b for a, b in zip(vis, vis[1:]))
    # above the earth's flux the eye is saturated (world7's brightness = min(1, flux)): never less
    assert bridge.world_config(_vary(record, "surface", flux=1.1))[0].visibility == vis[-1]
    regrowth = [bridge.world_config(_vary(record, "bodies", light=x))[0].food_regrowth for x in (80, 120, 160, 200)]
    assert all(a < b for a, b in zip(regrowth, regrowth[1:]))


def test_visibility_changes_what_founders_see():
    seen = {}
    for v in (0.5, 1.0):
        world = World.create(3, Config(visibility=v))
        seen[v] = sum(len(world.observe(a)["food"]) for a in world.agents.values())
    assert seen[0.5] < seen[1.0]


def test_mapping_is_monotone_where_it_should_be(records):
    record = records[7]
    def cfg(era, **v):
        return bridge.world_config(_vary(record, era, **v))
    costs = [cfg("surface", mass=m)[0].action_cost for m in (0.4, 1.0, 3.0, 8.0)]
    assert all(a < b for a, b in zip(costs, costs[1:]))  # stronger gravity, dearer actions
    heat = [cfg("surface", temperature=t)[0].base_metabolism for t in (288, 300, 320, 350)]
    assert all(a < b for a, b in zip(heat, heat[1:]))  # further from 288 K, dearer upkeep
    sizes = (32, 64, 128, 256, 512)
    mass = [cfg("bodies", consumer_size=n)[1]["anatomy"]["body_mass"][0] for n in sizes]
    upkeep = [cfg("bodies", consumer_size=n)[1]["anatomy"]["metabolism"][0] for n in sizes]
    assert all(a < b for a, b in zip(mass, mass[1:])) and all(a > b for a, b in zip(upkeep, upkeep[1:]))
    hands = [cfg("bodies", consumer_types=t)[1]["anatomy"]["manipulation"][0] for t in (3, 5, 7, 9)]
    assert all(a < b for a, b in zip(hands, hands[1:]))
    eyes = [cfg("senses", eyes=e)[1]["anatomy"]["sensing"][0] for e in (0.0, 1.0, 2.0, 3.0)]
    assert all(a <= b for a, b in zip(eyes, eyes[1:])) and eyes[-1] > eyes[0]
    talk = [cfg("senses", talkers=t)[1]["organs"]["voice"] for t in (0.0, 0.2, 0.6)]
    assert talk == [0.0, 0.2, 0.6]
    preds = [cfg("bodies", predator_share=s)[1]["ecology_hints"]["predator_density"] for s in (0.01, 0.05, 0.09)]
    assert all(a < b for a, b in zip(preds, preds[1:]))
    oxygen = [cfg("bodies", air=a)[1]["ecology_hints"]["max_body_mass"] for a in (0.05, 0.1, 0.3, 1.0)]
    assert all(a <= b for a, b in zip(oxygen, oxygen[1:])) and oxygen[-1] > oxygen[0]
    metal = cfg("chemistry", mix="reducing")[1]["objects"]["metal"]
    assert metal > 0 and cfg("chemistry", mix="ocean")[1]["objects"]["metal"] == 0
    carbon = [cfg("chemistry", organic=o)[1]["objects"]["wood"] for o in (10, 50, 100)]
    assert carbon[0] < carbon[1] < carbon[2]


def test_an_earth_twin_keeps_lifes_defaults(records):
    twin = _vary(_vary(records[7], "surface", mass=1.0, flux=1.0, temperature=288), "bodies", light=200.0)
    cfg, _ = bridge.world_config(twin)
    base = Config()
    assert (cfg.visibility, cfg.action_cost, cfg.base_metabolism, cfg.food_regrowth) == \
        (1.0, base.action_cost, base.base_metabolism, base.food_regrowth)


def test_without_senses_the_defaults_are_used_and_said(records):
    record = records[7]
    no_senses = {**record, "steps": [s for s in record["steps"] if not s["era"].startswith("senses")]}
    _, founders = bridge.world_config(no_senses)
    assert founders["organs"] == {"voice": 1.0, "hearing": 1.0}
    assert "default" in founders["sources"]["sensing"] and founders["ecology_hints"]["predator_pressure"] is None


def test_apportion_is_exact_and_ordered():
    assert bridge.apportion(36, {"stone": 1.0, "wood": 0.5, "fiber": 0.5, "clay": 0.5, "metal": 0.0}) == \
        {"stone": 15, "wood": 7, "fiber": 7, "clay": 7, "metal": 0}
    out = bridge.apportion(7, {"stone": 1.0, "wood": 1.0, "fiber": 1.0, "clay": 0.0, "metal": 0.0})
    assert out == {"stone": 3, "wood": 2, "fiber": 2, "clay": 0, "metal": 0}
    assert sum(bridge.apportion(36, {"stone": 1.0, "wood": 0.37, "fiber": 0.37, "clay": 0.51, "metal": 0.25})
               .values()) == 36


def test_synthetic_worlds_are_valid_labelled_and_deterministic():
    worlds = bridge.synthetic_worlds(6, start_seed=10)
    assert worlds == bridge.synthetic_worlds(6, start_seed=10)
    for w in worlds:
        assert w["source"] == "synthetic"
        cfg, founders = bridge.world_config(w)
        assert founders["source"] == "synthetic"
        assert 0.5 < cfg.visibility <= 1.0
    cfg, founders = bridge.world_config(worlds[0])
    world = bridge.create_world(10, cfg, founders)
    for _ in range(30):
        world.step(None)
    world.validate()


def test_synthetic_ranges_cover_the_scanned_world7_worlds():
    """The recorded scan of world7 seeds 0-299 (docs/life8/results) lies inside the synthetic draw ranges."""
    if not os.path.exists(SCAN):
        pytest.skip("scan result not recorded")
    scan = json.load(open(SCAN, encoding="utf8"))
    for eras in scan["eras"].values():
        s, c, b = eras["surface"], eras["chemistry"], eras["bodies"]
        assert 0.3 <= s["mass"] <= 10 and 0.356 <= s["flux"] <= 1.107
        assert 20 <= c["organic"] <= 400 and 4600 <= c["h2o"] <= 5400
        assert 64 <= b["consumer_size"] <= 512 and 3 <= b["consumer_types"] <= 8
        assert b["trophic_levels"] in (3, 4) and 0.05 <= b["air"] <= 0.32
        assert 0.005 <= b["predator_share"] <= 0.17
        if "senses" in eras:
            e = eras["senses"]
            assert e["eyes"] <= 4 and e["ears"] <= 4.5 and e["talkers"] <= 1 and 0.05 <= e["predator_pressure"] <= 1
    assert scan["seeds_scanned"] == 300 and scan["qualifying"] == len(scan["seeds"]) == 49


@pytest.mark.slow
def test_candidate_worlds_parallel_equals_serial():
    serial = bridge.candidate_worlds(2, start_seed=60)
    assert [r["seed"] for r in serial] == [r["seed"] for r in bridge.candidate_worlds(2, start_seed=60, workers=4)]
    assert all("bodies" in r["links"] for r in serial)


def test_candidate_worlds_skips_seeds_that_do_not_reach_bodies(records):
    found = bridge.candidate_worlds(1, start_seed=64)  # 64 stops below bodies, 65 reaches them
    assert [r["seed"] for r in found] == [65]
    assert found[0]["steps"] == records[65]["steps"] and found[0]["links"] == records[65]["links"]
