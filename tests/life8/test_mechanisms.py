"""Causal interventions: capability is distinct from autonomous discovery."""
import copy
import random

import pytest

from haishool.life8.entities import Artifact, Anatomy
from haishool.life8.config import Config
from haishool.life8.world import World
from haishool.life8 import materials, learning


def tool(material="stone", mass=1.):
    return Artifact(id=1, material=material, position=[0., 0.], mass=mass,
                    sharpness=.7, properties=materials.material_properties(material, mass))


def test_tool_performance_depends_on_properties_not_invention_labels():
    ordinary = tool().to_dict()
    renamed = copy.deepcopy(ordinary)
    renamed.update(material="legendary_hammer", name="mathematics", technology="civilization")
    assert materials.strike(ordinary, 2., 1., 1.) == materials.strike(renamed, 2., 1., 1.)
    blunt = copy.deepcopy(ordinary)
    blunt["sharpness"] = 0.
    assert materials.strike(ordinary, 2., 1., 1.)["damage"] > materials.strike(blunt, 2., 1., 1.)["damage"]


def test_manipulation_changes_effectiveness_without_creating_work():
    low = materials.strike(tool(), 2., 1., .25)
    high = materials.strike(tool(), 2., 1., 2.)
    assert high["damage"] > low["damage"]
    for result in (low, high):
        assert result["work"] == 1.
        assert result["delivered_work"] + result["tool_heat"] + result["untracked_dissipation"] == pytest.approx(1.)
        assert result["untracked_dissipation"] >= 0


def test_zero_work_cannot_produce_damage_or_join_material():
    assert materials.strike(tool(), 1., 0., 1.)["damage"] == 0.
    other = tool("wood")
    other.id = 2
    with pytest.raises(ValueError):
        materials.combine_items(tool(), other, 0.)


def quiet_world(**overrides):
    values = dict(population=3, food_patches=1, initial_objects=2, automatic_reproduction=False,
                  learning=False, signal_noise=0., food_regrowth=0., max_age=200)
    values.update(overrides)
    world = World.create(seed=9, config=Config(**values))
    for index, agent in enumerate(world.agents.values()):
        agent.position = [5. + index * 10., 5.]
        agent.anatomy = Anatomy()
    for food in world.foods.values(): food.position = [20., 15.]
    for obj in world.objects.values(): obj.position = [20., 15.]
    return world


def only(world, identity, action):
    return {key: action if key == identity else "rest" for key in world.agents}


def test_hidden_entities_cannot_change_local_observation():
    world = quiet_world()
    observer = world.agents[1]
    before = world.observe(observer)
    assert before["food"] == before["objects"] == before["neighbors"] == []
    world.foods[1].hardness += 10
    world.objects[1].sharpness = .99
    world.agents[2].anatomy.manipulation = 2.5
    assert world.observe(observer) == before
    world.foods[1].position = [5.5, 5.]
    assert [food["id"] for food in world.observe(observer)["food"]] == [1]
    assert learning.state_key(world.observe(observer)) != learning.state_key(before)


def test_toroidal_visibility_does_not_omit_nearest_edge_neighbor():
    world = quiet_world(population=1)
    world.agents[1].position = [.1, 5.]
    world.foods[1].position = [world.config.width - .1, 5.]
    food = world.observe(1)["food"][0]
    assert food["distance"] == pytest.approx(.2)
    assert food["dx"] == pytest.approx(-.2)


@pytest.mark.parametrize("action,kind", [("eat", "food"), ("collect", "object"), ("share", "agent")])
def test_guessed_remote_id_cannot_bypass_physical_reach(action, kind):
    world = quiet_world()
    actor = world.agents[1]
    target = 2 if kind == "agent" else 1
    before_energy = actor.energy
    events = world.step(only(world, actor.id, {"type": action, "target": target}))
    event = next(e for e in events if e["type"] == "action" and e["agent"] == actor.id)
    assert event["success"] is False
    assert actor.energy < before_energy, "an unsuccessful physical action still costs energy"
    assert not actor.inventory
    world.validate()


@pytest.mark.parametrize("action", [{"type": "strike", "target_kind": "not-a-physical-target"},
                                    {"type": "eat", "target": []}, {"type": "drop", "object": True}])
def test_malformed_interventions_are_rejected_before_tick_mutation(action):
    world = quiet_world()
    before = world.to_dict()
    with pytest.raises(ValueError):
        world.step({1: action})
    assert world.to_dict() == before


def test_signals_have_cost_range_arbitrary_tokens_and_expiry():
    world = quiet_world()
    world.agents[2].position = [7., 5.]
    before = world.agents[1].energy
    world.step(only(world, 1, "signal_3"))
    assert before - world.agents[1].energy >= world.config.signal_cost
    world.step({identity: "rest" for identity in world.agents})
    heard = world.observe(2)["messages"]
    assert len(heard) == 1 and heard[0]["symbol"] == 3
    assert world.observe(3)["messages"] == []
    assert not ({"meaning", "food", "target", "instruction"} & set(heard[0]))
    assert all(agent.controller["q"] == {} for agent in world.agents.values())
    for _ in range(world.config.message_ttl + 2):
        world.step({identity: "rest" for identity in world.agents})
    assert world.observe(2)["messages"] == []


def test_most_recent_heard_symbol_reaches_controller():
    world = quiet_world(population=2)
    world.agents[2].position = [6., 5.]
    world.step(only(world, 1, "signal_0"))
    world.step(only(world, 1, "signal_2"))
    world.step({identity: "rest" for identity in world.agents})
    observation = world.observe(2)
    assert [message["symbol"] for message in observation["messages"]] == [0, 2]
    assert learning.observation_features(observation)["heard_symbol"] == 2


def test_deaf_receiver_and_muted_sender_do_not_exchange_messages():
    world = quiet_world(population=2)
    world.agents[2].position = [6., 5.]
    world.agents[2].anatomy.hearing = 0
    world.step(only(world, 1, "signal_1"))
    world.step({identity: "rest" for identity in world.agents})
    assert world.observe(2)["messages"] == []
    world.agents[1].anatomy.voice = 0
    sent = world.counters.get("signals_sent", 0)
    world.step(only(world, 1, "signal_2"))
    assert world.counters.get("signals_sent", 0) == sent


def test_local_culture_transfers_one_experience_not_private_q_table():
    world = quiet_world(learning=True)
    teacher, pupil = world.agents[1], world.agents[2]
    pupil.position = [6., 5.]
    world.foods[1].position = [5., 5.]
    world.step(only(world, teacher.id, "eat"))
    private = {"private_teacher_experience": 123}
    learning.learn(teacher.controller, private, "heat", 50., private)
    private_key = learning.state_key(private)
    before = pupil.energy
    world.step({1: "signal_0", 2: {"type": "imitate", "target": 1},
                3: {"type": "imitate", "target": 1}})
    assert pupil.controller["demonstrations"] == 1
    assert world.agents[3].controller["demonstrations"] == 0
    assert private_key not in pupil.controller["q"]
    assert pupil.energy < before - world.config.culture_cost
    assert any(row["source"] == "local_demonstration" for row in pupil.controller["memory"])


def test_real_food_outcomes_transfer_to_unseen_irrelevant_object_context():
    world = quiet_world(population=1, learning=True, initial_energy=20., initial_objects=1)
    world.foods[1].position = list(world.agents[1].position)
    agent = world.agents[1]
    for _ in range(6):
        events = world.step({1: "eat"})
        transition = next(event for event in events if event["type"] == "transition")
        assert transition["reward"] > 0
    agent.controller["epsilon"] = 0.
    world.objects[1].position = [5.5, 5.]
    observation = world.observe(1)
    assert observation["object_near"] and observation["food_near"]
    assert learning.state_key(observation) not in agent.controller["q"]
    values = learning.action_values(agent.controller, observation, actions=("eat", "rest"))
    assert values["eat"] > values["rest"]
    choices = {learning.choose_action(copy.deepcopy(agent.controller), observation, random.Random(seed),
                                     actions=("eat", "rest")) for seed in range(8)}
    assert choices == {"eat"}


def test_learning_update_size_is_not_a_bonus_for_more_feature_fields():
    sparse = {"food_near": True}
    rich = {"food_near": True, "object_near": False, "neighbor_near": False,
            "energy_band": "middle", "tool_available": False, "health_band": "well"}
    predictions = []
    for observation in (sparse, rich):
        controller = learning.new_controller(alpha=.25, gamma=0., epsilon=0.)
        learning.learn(controller, observation, "eat", 4., observation, terminal=True)
        predictions.append(learning.action_values(controller, observation, actions=("eat",))["eat"])
    assert predictions == pytest.approx([1., 1.])


def test_remembered_local_partner_changes_controller_state():
    world = quiet_world(population=2)
    world.agents[2].position = [6., 5.]
    before = world.observe(2)
    assert before["neighbor_familiar"] is False
    world.step({1: {"type": "share", "target": 2}, 2: "rest"})
    after = world.observe(2)
    assert after["neighbor_familiar"] is True
    assert after["neighbor_last_exchange"] == "energy_shared"
    assert learning.state_key(before) != learning.state_key(after)


def test_faster_morphology_moves_further_but_pays_more_energy():
    slow = quiet_world(population=1)
    fast = World.from_dict(slow.to_dict())
    slow.agents[1].anatomy.speed = .5
    fast.agents[1].anatomy.speed = 2.
    start = list(slow.agents[1].position)
    slow.step({1: "move_e"})
    fast.step({1: "move_e"})
    assert fast.distance(start, fast.agents[1].position) > slow.distance(start, slow.agents[1].position)
    assert fast.agents[1].energy < slow.agents[1].energy
    slow.validate()
    fast.validate()


def test_material_actions_preserve_mass_and_account_for_actual_heat_work():
    world = quiet_world(population=1, food_patches=0)
    world.agents[1].anatomy.manipulation = 2.
    for obj in world.objects.values(): obj.position = [5., 5.]
    for obj_id in list(world.objects): world.step({1: {"type": "collect", "target": obj_id}})
    mass = sum(obj.mass for obj in world.objects.values())
    def composition():
        totals = {}
        for obj in world.objects.values():
            for material, amount in obj.properties["composition"].items():
                totals[material] = totals.get(material, 0.) + amount
        return totals
    initial_composition = composition()
    for action in ("combine", "heat", "dismantle"):
        before_heat = sum(materials.thermal_energy(obj) for obj in world.objects.values())
        before_environment = world.ledger["heat_to_environment"]
        before_supplied = world.ledger["heat_to_objects"]
        before_spent = world.ledger["action_energy"]
        world.step({1: action})
        supplied = world.ledger["heat_to_objects"] - before_supplied
        cooled = world.ledger["heat_to_environment"] - before_environment
        after_heat = sum(materials.thermal_energy(obj) for obj in world.objects.values())
        assert sum(obj.mass for obj in world.objects.values()) == pytest.approx(mass)
        assert composition() == pytest.approx(initial_composition)
        assert after_heat + cooled == pytest.approx(before_heat + supplied)
        assert supplied <= world.ledger["action_energy"] - before_spent + 1e-12
        world.validate()


def test_agent_without_joining_energy_dies_without_minting_or_losing_material():
    world = quiet_world(population=1, food_patches=0, initial_energy=.20, base_metabolism=0.)
    for obj in world.objects.values(): obj.position = [5., 5.]
    initial_ids = set(world.objects)
    for identity in sorted(world.objects): world.step({1: {"type": "collect", "target": identity}})
    assert world.agents[1].energy == pytest.approx(.04)
    events = world.step({1: "combine"})
    event = next(e for e in events if e["type"] == "action")
    assert event["success"] is False
    assert world.stop_reason == "extinction" and not world.agents
    assert set(world.objects) == initial_ids
    assert all(obj.holder is None for obj in world.objects.values())
    world.validate()


def test_capacity_pause_does_not_delete_existing_people():
    world = quiet_world(population=1, population_limit=1, food_patches=0, initial_objects=0,
                        initial_energy=40., maturity_age=0)
    events = world.step({1: "reproduce"})
    assert world.stop_reason == "population_capacity"
    assert list(world.agents) == [1]
    assert any(event["type"] == "capacity_pause" for event in events)
    state = world.to_dict()
    assert world.step() == [] and world.to_dict() == state


def test_reported_reward_is_only_measured_energy_and_health_change():
    world = quiet_world()
    world.agents[2].position = [6., 5.]
    for action in ("signal_0", "signal_3", "share", "rest"):
        events = world.step(only(world, 1, action))
        for event in events:
            if event["type"] == "action":
                measured = event["energy_after"] - event["energy_before"] + 8 * (event["health_after"] - event["health_before"])
                assert event["reward"] == pytest.approx(measured)


def test_received_share_after_own_action_is_learned_as_real_outcome(monkeypatch):
    world = quiet_world(population=2, learning=True)
    world.agents[2].position = [6., 5.]
    # The receiver acts first; its reward must still include the later transfer.
    monkeypatch.setattr(world.rng, "shuffle", lambda order: order.sort())
    before = world.agents[1].energy
    events = world.step({1: "rest", 2: {"type": "share", "target": 1}})
    transition = next(e for e in events if e["type"] == "transition" and e["agent"] == 1)
    expected = world.agents[1].energy - before
    assert expected > 1
    assert transition["reward"] == pytest.approx(expected)
    assert world.agents[1].controller["memory"][-1]["reward"] == pytest.approx(expected)
    assert transition["next_observation"]["self"]["energy"] == world.agents[1].energy


@pytest.mark.parametrize("lethal", [False, True])
def test_later_external_injury_reaches_victim_transition(monkeypatch, lethal):
    world = quiet_world(population=2, learning=True)
    world.agents[2].position = [6., 5.]
    victim = world.agents[1]
    if lethal: victim.health = .001
    before_energy, before_health = victim.energy, victim.health
    monkeypatch.setattr(world.rng, "shuffle", lambda order: order.sort())
    events = world.step({1: "rest", 2: {"type": "strike", "target_kind": "agent", "target": 1}})
    transition = next(e for e in events if e["type"] == "transition" and e["agent"] == 1)
    expected = victim.energy - before_energy + 8 * (victim.health - before_health)
    assert victim.health < before_health
    assert transition["reward"] == pytest.approx(expected)
    assert victim.controller["memory"][-1]["reward"] == pytest.approx(expected)
    assert transition["done"] is lethal
    assert (victim.id not in world.agents) is lethal
    world.validate()
