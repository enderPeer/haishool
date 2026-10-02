"""Independent checks of state, randomness and individual continuity."""
import copy
import json

import pytest

from haishool.life8.config import Config
from haishool.life8.world import World


def json_state(world):
    # Exercise the actual portable representation, including sorted string keys.
    return json.loads(json.dumps(world.to_dict(), sort_keys=True, allow_nan=False))


def interaction_world(seed=85):
    world = World.create(seed=seed, config=Config(population=4, food_patches=6, initial_objects=6,
                        automatic_reproduction=False, max_age=250, initial_energy=38.))
    for index, agent in enumerate(world.agents.values()):
        agent.position = [5. + .2 * index, 5.]
        agent.anatomy.voice = agent.anatomy.hearing = 1
    for index, obj in enumerate(world.objects.values()):
        obj.position = [5. + .1 * index, 5.]
    for index, food in enumerate(world.foods.values()):
        food.position = [5. + .1 * index, 5.]
    return world


def interventions(world, tick):
    actions = ("collect", "signal_2", "teach", "imitate", "strike", "heat", "combine", "dismantle", "eat", "rest")
    return {agent_id: actions[(tick + index) % len(actions)] for index, agent_id in enumerate(sorted(world.agents))}


def test_exact_continuation_through_learning_messages_and_material_work():
    original = interaction_world()
    for index in range(18):
        original.step(interventions(original, index))
    saved = json_state(original)
    resumed = World.from_dict(saved)
    assert json_state(resumed) == saved
    assert original.objects and any(agent.controller["updates"] > 0 for agent in original.agents.values())
    assert any(agent.controller["memory"] for agent in original.agents.values())
    assert original.counters.get("signals_sent", 0) > 0
    for index in range(18, 55):
        actions = interventions(original, index)
        assert original.step(actions) == resumed.step(copy.deepcopy(actions))
        assert json_state(original) == json_state(resumed)
        original.validate()
        resumed.validate()


def test_saved_state_and_loaded_world_do_not_alias_live_memory():
    world = interaction_world(4)
    world.step(interventions(world, 0))
    saved = world.to_dict()
    untouched = copy.deepcopy(saved)
    world.step(interventions(world, 1))
    assert saved == untouched, "to_dict returned references to mutable live state"
    restored = World.from_dict(saved)
    restored.step(interventions(restored, 2))
    assert saved == untouched, "from_dict retained references to caller-owned checkpoint data"


def test_resume_preserves_spontaneous_action_choices():
    world = interaction_world(19)
    for _ in range(12):
        world.step()
    resumed = World.from_dict(json_state(world))
    for _ in range(30):
        assert world.step() == resumed.step()
        assert json_state(world) == json_state(resumed)


def test_next_observation_is_exactly_the_next_decision_context():
    world = interaction_world(29)
    previous = {}
    for tick in range(20):
        events = world.step(interventions(world, tick))
        current = {event["agent"]: event for event in events if event["type"] == "transition"}
        for identity in previous.keys() & current.keys():
            if not previous[identity]["done"]:
                assert previous[identity]["next_observation"] == current[identity]["observation"]
        previous = current


def test_corrupt_or_incompatible_state_is_refused():
    with pytest.raises((ValueError, KeyError, TypeError)):
        World.from_dict({})
    saved = json_state(interaction_world())
    saved["model_version"] = "a-different-engine"
    with pytest.raises(ValueError, match="incompatible"):
        World.from_dict(saved)


def test_incompatible_nested_controller_is_refused_at_restore():
    saved = json_state(interaction_world())
    saved["agents"][0]["controller"]["schema_version"] = "unknown-controller"
    with pytest.raises(ValueError, match="controller"):
        World.from_dict(saved)


@pytest.mark.parametrize("case", ["fractional_age", "future_birth", "negative_parent", "message_counter",
                                  "stop_reason", "memory_shape", "object_id", "future_message"])
def test_same_version_checkpoint_still_requires_semantically_valid_state(case):
    world = interaction_world()
    world.step({identity: "signal_0" if identity == 1 else "rest" for identity in world.agents})
    saved = json_state(world)
    if case == "fractional_age": saved["agents"][0]["age"] = .5
    elif case == "future_birth": saved["agents"][0]["born_tick"] = world.tick + 10
    elif case == "negative_parent": saved["agents"][0]["parent_id"] = -1
    elif case == "message_counter": saved["next_message_id"] = 0
    elif case == "stop_reason": saved["stop_reason"] = "finished_language"
    elif case == "memory_shape": saved["agents"][0]["memory"]["messages"] = {}
    elif case == "object_id": saved["objects"][0]["id"] = 0
    elif case == "future_message":
        saved["agents"][0]["memory"]["messages"] = [{"sender": 2, "symbol": 0,
            "tick": world.tick + 10, "distance": 1., "message_id": 1}]
    with pytest.raises(ValueError):
        World.from_dict(saved)


def test_same_seed_reproduces_initial_individuals_and_world():
    config = Config(population=4, food_patches=5, initial_objects=4)
    first = World.create(seed=123, config=config)
    second = World.create(seed=123, config=Config.from_dict(config.to_dict()))
    other = World.create(seed=124, config=config)
    assert json_state(first) == json_state(second)
    assert json_state(first) != json_state(other)


def test_birth_death_and_maker_artifact_survive_exact_continuation():
    world = World.create(seed=37, config=Config(population=1, food_patches=0, initial_objects=2,
                    automatic_reproduction=False, initial_energy=40., maturity_age=0, max_age=10,
                    mutation_rate=1., mutation_scale=.2))
    parent = next(iter(world.agents.values()))
    parent.position = [5., 5.]
    for obj in world.objects.values(): obj.position = [5., 5.]
    for obj_id in sorted(world.objects):
        world.step({parent.id: {"type": "collect", "target": obj_id}})
    world.step({parent.id: "combine"})
    artifact = next(iter(world.objects.values()))
    assert artifact.maker_id == parent.id and artifact.components
    events = world.step({parent.id: "reproduce"})
    births = [event for event in events if event["type"] == "birth"]
    assert len(births) == 1
    child = world.agents[births[0]["agent"]]
    assert child.parent_id == parent.id and child.id > parent.id
    assert child.controller["q"] == {} and child.controller["updates"] == 0
    assert parent.controller["updates"] > 0
    resumed = World.from_dict(json_state(world))
    deaths = []
    for _ in range(25):
        actions = {identity: "rest" for identity in world.agents}
        observed = world.step(actions)
        assert observed == resumed.step(actions)
        assert json_state(world) == json_state(resumed)
        deaths += [event for event in observed if event["type"] == "death"]
        world.validate()
    assert {event["agent"] for event in deaths} == {parent.id, child.id}
    assert world.stop_reason == "extinction"
    assert artifact.id in world.objects
    assert world.objects[artifact.id].holder is None
    assert world.objects[artifact.id].maker_id == parent.id
