import copy
import json
import math
import random

import pytest

from haishool.life8.config import Config
from haishool.life8.entities import Anatomy
from haishool.life8.world import World
from haishool.life8 import materials


def controlled(**options):
    cfg = Config(**{**dict(population=2, food_patches=1, initial_objects=3, learning=False,
                          automatic_reproduction=False, base_metabolism=0., food_regrowth=0.), **options})
    world = World.create(11, cfg)
    for agent in world.agents.values():
        agent.position = [5., 5.]
        agent.anatomy = Anatomy()
    for item in (*world.objects.values(), *world.foods.values()):
        item.position = [5., 5.]
    return world


def act(world, first, second="rest"):
    return world.step({1: first, 2: second})


def test_create_and_snapshot_use_individuals_not_milestone_labels():
    world = World.create(85)
    assert world.tick == 0 and len(world.agents) == world.config.population
    assert len({a.id for a in world.agents.values()}) == world.config.population
    view = world.snapshot()
    assert "controller" not in view["agents"][0] and "memory" not in view["agents"][0]
    assert {"id", "x", "y", "anatomy", "age", "energy", "health", "inventory"} <= view["agents"][0].keys()
    assert not {"civilization", "technology_level", "grammar_stage"}.intersection(world.summary())
    assert world.validate()["ok"]
    json.dumps(world.to_dict(), allow_nan=False)


def test_local_observation_cannot_see_distant_objects_or_agents():
    world = controlled()
    world.agents[2].position = [20., 15.]
    world.objects[1].position = [20., 15.]
    world.foods[1].position = [20., 15.]
    seen = world.observe(1)
    assert seen["neighbors"] == [] and seen["food"] == []
    assert 1 not in {o["id"] for o in seen["objects"]}
    before = copy.deepcopy(seen)
    world.agents[2].last_outcome = 100
    assert world.observe(1) == before


def test_toroidal_local_geometry_and_movement_are_actual_costly_actions():
    world = controlled()
    world.agents[1].position = [.1, 5.]
    world.agents[2].position = [world.config.width - .2, 5.]
    assert world.observe(1)["neighbors"][0]["distance"] == pytest.approx(.3)
    initial = world.agents[1].energy
    act(world, "move_w")
    assert world.agents[1].position[0] > world.config.width - 2
    assert world.agents[1].energy < initial
    assert world.validate()["ok"]


def test_anatomy_changes_sensor_range_stride_work_and_cost():
    world = controlled()
    a, b = world.agents.values()
    a.anatomy.speed, b.anatomy.speed = .5, 2.
    a.anatomy.sensing, b.anatomy.sensing = .3, 2.5
    world.foods[1].position = [12., 5.]
    assert world.observe(a)["food"] == [] and world.observe(b)["food"]
    start = list(a.position)
    act(world, "move_e", "move_e")
    assert world.distance(start, b.position) > world.distance(start, a.position)
    assert b.energy < a.energy
    assert b.anatomy.metabolic_factor > a.anatomy.metabolic_factor


def test_food_is_consumed_and_regrowth_is_external_and_recorded():
    world = controlled(food_regrowth=.1)
    amount = world.foods[1].amount
    energy = world.agents[1].energy
    events = act(world, "eat")
    assert world.foods[1].amount == pytest.approx(amount - 1 + .1)
    assert world.agents[1].energy == pytest.approx(energy + 4 - world.config.action_cost)
    assert world.ledger["food_eaten"] == 1 and world.ledger["food_regrown"] == pytest.approx(.1)
    transition = next(e for e in events if e["type"] == "transition" and e["agent"] == 1)
    assert transition["reward"] == pytest.approx(transition["next_observation"]["self"]["energy"] - transition["observation"]["self"]["energy"])
    assert world.validate()["ok"]


def test_materials_persist_through_combine_dismantle_and_maker_death():
    world = controlled()
    act(world, {"type": "collect", "target": 1})
    act(world, {"type": "collect", "target": 2})
    act(world, "combine")
    artifact = world.objects[world.agents[1].inventory[0]]
    assert artifact.mass == 2 and artifact.maker_id == 1 and artifact.components == [1, 2]
    assert 1 not in world.objects and 2 not in world.objects
    world.agents[1].lifespan = world.agents[1].age + 1
    events = act(world, "rest")
    assert 1 not in world.agents and artifact.id in world.objects
    assert artifact.holder is None
    assert any(e["type"] == "death" and artifact.id in e["released_objects"] for e in events)
    world.step({2: {"type": "collect", "target": artifact.id}})
    world.step({2: "dismantle"})
    assert sum(o.mass for o in world.objects.values()) == 3
    assert world.validate()["ok"]


def test_strike_can_autonomously_address_agents_or_objects_without_action_bonus():
    world = controlled()
    health = world.agents[2].health
    events = act(world, "strike_agent")
    assert world.agents[2].health < health
    assert next(e for e in events if e["type"] == "action" and e["agent"] == 1)["reward"] < 0
    durability = world.objects[1].properties["durability"]
    act(world, {"type": "strike_object", "target": 1})
    assert world.objects[1].properties["durability"] < durability
    assert world.objects[1].temperature > 20
    assert world.validate()["ok"]


def test_hot_materials_transfer_bounded_heat_and_cause_real_health_cost():
    world = controlled()
    obj = world.objects[1]
    obj.temperature = 100.
    world.ledger["initial_material_heat"] = sum(materials.thermal_energy(o) for o in world.objects.values())
    energy = materials.thermal_energy(obj)
    events = act(world, {"type": "collect", "target": obj.id})
    assert world.agents[1].health < 1
    assert materials.thermal_energy(obj) < energy
    assert world.ledger["heat_to_body"] > 0
    assert any(e["type"] == "thermal_contact" for e in events)
    assert world.validate()["ok"]


def test_colocated_noiseless_signal_costs_energy_and_carries_no_assigned_meaning():
    # life8: renamed from Matrix's "..._cost_range_noise_..." which tested neither range nor
    # noise (both agents share one spot, noise is off). Range and noise are tested in
    # test_fixes.py::test_call_reach_is_a_hard_range_around_the_sender and
    # test_fixes.py::test_noise_replaces_symbols_at_the_configured_rate.
    world = controlled(signal_noise=0.)
    before = world.agents[1].energy
    events = act(world, "signal_2")
    assert world.agents[1].energy < before - world.config.action_cost
    transition = next(e for e in events if e["type"] == "transition" and e["agent"] == 2)
    assert transition["observation"]["messages"] == []
    assert transition["next_observation"]["messages"][0]["symbol"] == 2
    message = world.observe(2)["messages"][0]
    assert message["symbol"] == 2 and message["age"] == 0
    assert "meaning" not in message
    assert world.validate()["ok"]


def test_full_tick_reward_includes_received_social_energy_and_external_injury():
    world = controlled()
    events = act(world, "rest", {"type": "share", "target": 1})
    transition = next(e for e in events if e["type"] == "transition" and e["agent"] == 1)
    assert transition["reward"] == pytest.approx(2. - world.config.action_cost * .25)
    events = act(world, "rest", {"type": "strike_agent", "target": 1})
    transition = next(e for e in events if e["type"] == "transition" and e["agent"] == 1)
    assert transition["reward_components"]["health_delta"] < 0
    assert transition["reward"] < -world.config.action_cost * .25
    assert transition["next_observation"] == world.observe(1)


def test_culture_is_local_costly_demo_not_copying_global_weights():
    world = controlled(learning=True)
    act(world, "eat")
    assert world.agents[1].memory["last_demo"]["action"] == "eat"
    before = world.agents[2].energy
    events = act(world, "rest", {"type": "imitate", "target": 1})
    assert world.agents[2].energy < before
    assert world.agents[2].controller["demonstrations"] > 0
    assert world.observe(2)["neighbor_familiar"]
    assert world.agents[2].controller is not world.agents[1].controller
    assert any(e["type"] == "action" and e.get("teacher") == 1 for e in events)
    assert world.validate()["ok"]


def test_reproduction_is_asynchronous_inherited_and_mutating_not_generation_reset():
    cfg = Config(population=1, food_patches=0, initial_objects=0, initial_energy=40,
                 maturity_age=0, mutation_rate=1., mutation_scale=.2, learning=False)
    world = World.create(9, cfg)
    parent = world.agents[1]
    old_anatomy = parent.anatomy.to_dict()
    events = world.step({1: "rest"})
    assert world.agents[1] is parent and len(world.agents) == 2
    child = world.agents[2]
    assert child.parent_id == 1 and child.age == 0 and parent.age == 1
    assert child.anatomy.to_dict() != old_anatomy
    assert child.controller["updates"] == 0
    assert any(e["type"] == "birth" and e["agent"] == 2 for e in events)
    assert world.validate()["ok"]


def test_population_limit_pauses_without_silent_culling():
    cfg = Config(population=1, population_limit=1, food_patches=0, initial_objects=0,
                 initial_energy=40, maturity_age=0, learning=False)
    world = World.create(5, cfg)
    events = world.step({1: "rest"})
    assert world.stop_reason == "population_capacity" and list(world.agents) == [1]
    assert any(e["type"] == "capacity_pause" for e in events)
    previous = world.to_dict()
    assert world.step() == [] and world.to_dict() == previous


def test_exact_json_resume_includes_controllers_messages_and_materials():
    world = controlled(learning=True)
    act(world, {"type": "collect", "target": 1}, "signal_1")
    act(world, "eat")
    restored = World.from_dict(json.loads(json.dumps(world.to_dict(), sort_keys=True, allow_nan=False)))
    for _ in range(30):
        assert world.step() == restored.step()
        assert world.to_dict() == restored.to_dict()
    assert world.validate()["ok"]


def test_local_rng_does_not_change_global_random_state():
    state = random.getstate()
    world = World.create(1, Config(population=2))
    world.step()
    assert random.getstate() == state


@pytest.mark.parametrize("override", [{1: "invent_fire"}, {1: {"type": "strike", "effort": float("nan")}},
                                       {1: {"type": "eat", "target": True}}, {1: {"type": "eat", "extra": 5}}])
def test_invalid_override_rejected_before_any_tick(override):
    world = controlled()
    old = world.to_dict()
    with pytest.raises(ValueError):
        world.step(override)
    assert old == world.to_dict()


@pytest.mark.parametrize("options", [{"population": True}, {"width": True}, {"max_age": 0},
                                      {"signal_noise": 2}, {"population": 200, "population_limit": 100},
                                      {"reproduction_threshold": 1}])
def test_configuration_invalid_parameters_fail(options):
    with pytest.raises(ValueError):
        Config(**options)
