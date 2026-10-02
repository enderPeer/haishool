"""Regression tests for the Matrix 2cf045a bugs fixed in life8.

Numbers refer to section 7 of the independent review (matrix-review/assessment.md).
Each test fails on the vendored Matrix code (checked by running this file
against an untouched copy of matrix_sim at 2cf045a) and passes on life8.
"""
import hashlib
import json
from pathlib import Path

import pytest

from haishool.life8 import learning
from haishool.life8.checkpoint import SCHEMA as CHECKPOINT_SCHEMA, load, save
from haishool.life8.config import Config
from haishool.life8.entities import Anatomy
from haishool.life8.world import DEMO_FEATURES, World

LIFE8 = Path(__file__).resolve().parents[2] / "haishool" / "life8"


def quiet_world(seed=9, **overrides):
    values = dict(population=3, food_patches=1, initial_objects=2, automatic_reproduction=False,
                  learning=False, signal_noise=0., food_regrowth=0., max_age=200)
    values.update(overrides)
    world = World.create(seed=seed, config=Config(**values))
    for index, agent in enumerate(world.agents.values()):
        agent.position = [5. + index * 10., 5.]
        agent.anatomy = Anatomy()
    for food in world.foods.values():
        food.position = [20., 15.]
    for obj in world.objects.values():
        obj.position = [20., 15.]
    return world


def rest_all(world):
    return world.step({identity: "rest" for identity in world.agents})


def transition(events, agent):
    return next(e for e in events if e["type"] == "transition" and e["agent"] == agent)


# 1. Birth cost credited to the parent's unrelated action --------------------------------------

def birth_world(**overrides):
    cfg = Config(**{**dict(population=1, food_patches=0, initial_objects=0, initial_energy=40., maturity_age=0,
                           base_metabolism=0., learning=True), **overrides})
    return World.create(9, cfg)


def test_birth_transfer_is_not_learned_as_the_outcome_of_the_parents_action():
    born = birth_world()
    blocked = World.from_dict(born.to_dict())
    blocked.agents[1].cooldown = 2  # the same parent, but no birth on this tick
    with_birth, without_birth = born.step({1: "rest"}), blocked.step({1: "rest"})
    assert any(e["type"] == "birth" for e in with_birth) and not any(e["type"] == "birth" for e in without_birth)
    # The world still pays: 4 birth cost to the ledger and 12 energy to the child.
    assert born.ledger["birth_cost"] == pytest.approx(4.)
    assert born.agents[1].energy == pytest.approx(blocked.agents[1].energy - 16.)
    assert born.validate()["ok"]
    # The learner does not: resting is worth the same with or without the birth.
    reward = transition(with_birth, 1)["reward"]
    assert reward == pytest.approx(transition(without_birth, 1)["reward"])
    assert reward > -1, "old Matrix booked about -16 against 'rest'"
    assert transition(with_birth, 1)["reward_components"]["birth_transfer"] == pytest.approx(16.)
    def rest_weights(world):
        return {feature: row["rest"] for feature, row in world.agents[1].controller["weights"].items() if "rest" in row}
    assert rest_weights(born) == pytest.approx(rest_weights(blocked))


def test_birth_transfer_in_reward_switch_restores_the_old_booking_for_interventions():
    old = birth_world(birth_transfer_in_reward=True)
    events = old.step({1: "rest"})
    assert transition(events, 1)["reward"] == pytest.approx(-16. - old.config.action_cost * .25)


# 2. Stale messages presented as fresh ----------------------------------------------------------

def test_late_receiver_hears_the_true_age_of_a_call_and_loses_it_at_expiry():
    world = quiet_world(population=2)
    world.agents[2].position = [15., 5.]  # 10 units away; a call reaches 5
    world.step({1: "signal_0", 2: "rest"})
    assert world.tick == 1 and world.observe(2)["messages"] == []
    for _ in range(4):
        rest_all(world)
    world.agents[2].position = [6., 5.]
    rest_all(world)  # heard at the end of tick 6
    heard = world.observe(2)["messages"]
    assert [m["symbol"] for m in heard] == [0]
    assert heard[0]["age"] == 5, "old Matrix stamped the delivery tick: age 0"
    assert heard[0]["heard_tick"] == 6
    while world.tick < 1 + world.config.message_ttl:
        rest_all(world)
    assert world.observe(2)["messages"], "still audible at age == message_ttl"
    rest_all(world)
    assert world.observe(2)["messages"] == [], "old Matrix kept it audible until tick 6 + ttl"


# 3. A share that moves no energy is still recorded ---------------------------------------------

def test_share_that_moves_no_energy_is_not_a_social_exchange():
    world = quiet_world(population=2)
    world.agents[2].position = [6., 5.]
    world.agents[1].energy = 1.5
    events = world.step({1: {"type": "share", "target": 2}, 2: "rest"})
    action = next(e for e in events if e["type"] == "action" and e["agent"] == 1)
    assert action["success"] is False and action["energy"] == 0.
    assert world.social_edges == {}
    assert "1" not in world.agents[2].memory["social"] and "2" not in world.agents[1].memory["social"]
    assert world.observe(2)["neighbor_last_exchange"] == "none"
    assert world.summary()["social_edge_count"] == 0


# 4. tool_damage_gain counts strikes on soft food -----------------------------------------------

def tool_world():
    world = quiet_world(population=1, food_patches=2, initial_objects=2)
    agent = world.agents[1]
    soft, hard = world.foods[1], world.foods[2]
    assert soft.hardness == 0. and hard.hardness > .5
    soft.position, hard.position = [5., 5.], [5.5, 5.]
    stone = next(o for o in world.objects.values() if o.material == "stone")
    stone.position = [5., 5.]
    world.step({1: {"type": "collect", "target": stone.id}})
    assert agent.inventory == [stone.id]
    return world


def test_tool_strike_on_soft_food_adds_no_tool_damage_gain():
    world = tool_world()
    events = world.step({1: {"type": "strike", "target": 1, "target_kind": "food"}})
    strike = next(e for e in events if e["type"] == "action")
    assert strike["success"] and strike["damage"] > 0 and world.foods[1].opening == 0.
    assert world.counters["tool_strikes"] == 1
    assert world.counters["tool_damage_gain"] == 0., "old Matrix counted about 0.18 of useless damage"
    assert strike["effective_damage"] == 0.


def test_tool_strike_on_hard_food_counts_only_the_opening_it_adds_over_a_bare_hand():
    world = tool_world()
    hard = world.foods[2]
    before = hard.opening
    world.step({1: {"type": "strike", "target": 2, "target_kind": "food"}})
    added = hard.opening / .995 - before  # regrowth step decays the opening by 0.5% after the strike
    assert added > 0
    assert 0 < world.counters["tool_damage_gain"] < added


# 5. The validated effort parameter is ignored ----------------------------------------------------

def holding_world(objects=2):
    world = quiet_world(population=1, food_patches=0, initial_objects=objects)
    world.agents[1].anatomy.manipulation = 2.
    for obj in world.objects.values():
        obj.position = [5., 5.]
    for identity in sorted(world.objects):
        world.step({1: {"type": "collect", "target": identity}})
    assert len(world.agents[1].inventory) == objects
    return world


@pytest.mark.parametrize("action", ["combine", "heat", "dismantle"])
def test_requested_work_is_spent_and_changes_the_result(action):
    worlds = {}
    for effort in (.05, 3.):
        world = holding_world()
        if action == "dismantle":
            world.step({1: "combine"})
        spent = world.ledger["action_energy"]
        world.step({1: {"type": action, "effort": effort}})
        worlds[effort] = (world, world.ledger["action_energy"] - spent)
        world.validate()
    (low, low_spent), (high, high_spent) = worlds[.05], worlds[3.]
    assert high_spent - low_spent == pytest.approx(2.95)
    held_low, held_high = (w.objects[w.agents[1].inventory[0]] for w in (low, high))
    if action == "combine":
        assert held_high.bond_strength > held_low.bond_strength
    elif action == "heat":
        assert held_high.temperature > held_low.temperature
    else:  # 0.05 work cannot break the joint made with 0.8 work; 3.0 can
        assert held_low.components and len(high.agents[1].inventory) == 2


def test_effort_is_bounded_per_action():
    world = holding_world()
    spent = world.ledger["action_energy"]
    world.step({1: {"type": "heat", "effort": 50.}})
    assert world.ledger["action_energy"] - spent == pytest.approx(5. + world.config.action_cost)


# 6. A new composite is not the active tool -------------------------------------------------------

def test_new_composite_is_the_tool_used_by_the_next_strike():
    world = holding_world(objects=3)
    third = world.agents[1].inventory[2]
    world.step({1: "combine"})
    composite = world.agents[1].inventory[0]
    assert world.objects[composite].components and composite != third
    # Put the remaining raw item down and strike it: the held composite must be the tool.
    world.objects[third].holder = None
    world.agents[1].inventory.remove(third)
    world.objects[third].position = [5., 5.]
    events = world.step({1: {"type": "strike_object", "target": third}})
    strike = next(e for e in events if e["type"] == "action")
    assert strike["success"] and strike["tool"] == composite, "old Matrix struck with the third raw item"


# 7. Demonstrations leak private state and are scored without bootstrapping -----------------------

def culture_world():
    world = quiet_world(population=2, learning=True)
    teacher, learner = world.agents[1], world.agents[2]
    learner.position = [6., 5.]
    world.foods[1].position = [5., 5.]
    world.ledger["initial_agent_energy"] -= teacher.energy - 6.
    teacher.energy = 6.  # internal state: energy band "low"
    return world, teacher, learner


def test_demonstration_carries_only_what_a_neighbour_could_observe():
    world, teacher, learner = culture_world()
    world.step({1: "rest", 2: "signal_1"})
    assert learning.observation_features(world.observe(1))["heard_symbol"] == 1
    world.step({1: "eat", 2: "rest"})  # decided while low on energy and hearing symbol 1
    demo = teacher.memory["last_demo"]
    private = {"energy_band", "health_band", "neighbor_familiar", "neighbor_last_exchange", "messages"}
    assert demo["action"] == "eat" and not private & set(demo["observation"])
    assert set(demo["observation"]) <= set(DEMO_FEATURES) and set(demo["next_observation"]) <= set(DEMO_FEATURES)
    world.step({1: "rest", 2: {"type": "imitate", "target": 1}})
    sample = next(row for row in reversed(learner.controller["memory"]) if row["source"] == "local_demonstration")
    assert "energy_band" not in sample["observation"] and sample["observation"].get("heard_symbol") is None


def test_demonstrated_approach_is_valued_through_what_it_leads_to():
    controller = learning.new_controller(epsilon=0., gamma=.9)
    near = {"food_direction": "e", "food_near": True}
    for _ in range(20):
        learning.learn(controller, near, "eat", 4., near, terminal=True)
    far = {"food_direction": "e", "food_near": False}
    learning.observe_demo(controller, far, "move_e", -.12, next_observation=near)
    values = learning.action_values(controller, far, ("move_e", "rest"))
    assert values["move_e"] > values["rest"], "old Matrix scored the paid step alone: Q(move_e) < Q(rest)"
    assert controller["memory"][-1]["next_observation"]["food_near"] is True


# 8. Imitation records the passive demonstrator as the source -------------------------------------

def test_imitation_is_remembered_by_the_imitator_only():
    world, teacher, learner = culture_world()
    world.step({1: "eat", 2: "rest"})
    world.step({1: "rest", 2: {"type": "imitate", "target": 1}})
    assert learner.controller["demonstrations"] == 1
    assert str(learner.id) not in teacher.memory["social"], "old Matrix wrote into the passive teacher's memory"
    assert learner.memory["social"][str(teacher.id)]["last_kind"] == "imitations"
    assert set(world.social_edges) == {f"{learner.id}:{teacher.id}"}
    world.validate()


def test_teaching_is_remembered_by_both():
    world, teacher, learner = culture_world()
    world.step({1: "eat", 2: "rest"})
    world.step({1: {"type": "teach", "target": 2}, 2: "rest"})
    assert teacher.memory["social"][str(learner.id)]["last_kind"] == "demonstrations"
    assert learner.memory["social"][str(teacher.id)]["last_direction"] == "incoming"
    assert set(world.social_edges) == {f"{teacher.id}:{learner.id}"}


# 10. Binary organs flip at the full mutation rate ------------------------------------------------

@pytest.mark.parametrize("flip", [0., 1.])
def test_voice_and_hearing_use_their_own_flip_rate(flip):
    world = birth_world(mutation_rate=1., organ_flip_rate=flip, learning=False)
    parent = world.agents[1].anatomy.to_dict()
    world.step({1: "rest"})
    child = world.agents[2].anatomy
    expected = parent["voice"] if flip == 0 else 1 - parent["voice"]
    assert child.voice == expected and child.hearing == (parent["hearing"] if flip == 0 else 1 - parent["hearing"])
    assert child.body_mass != parent["body_mass"]  # continuous genes still mutate at mutation_rate


def test_default_organ_flip_rate_is_small():
    assert Config().organ_flip_rate == .005 < Config().mutation_rate


# 11. Action concentration after extinction -------------------------------------------------------

def test_action_concentration_is_missing_not_zero_after_extinction():
    world = quiet_world(population=1, initial_energy=.05)
    world.step({1: "eat"})
    assert world.stop_reason == "extinction"
    summary = world.summary()
    assert summary["mean_action_concentration"] is None and summary["dominant_actions"] == {}
    living = quiet_world(population=2)
    for _ in range(3):
        living.step({1: "eat", 2: "rest"})
    assert living.summary()["dominant_actions"] == {"eat": 1, "rest": 1}
    assert living.summary()["mean_action_concentration"] == 1.


# 13. Signal noise was untested --------------------------------------------------------------------

def chorus(noise, receivers=24, seed=3):
    world = quiet_world(seed=seed, population=receivers + 1, signal_noise=noise, max_age=400)
    for index, agent in enumerate(sorted(world.agents.values(), key=lambda a: a.id)[1:]):
        agent.position = [5. + 1.5 * (index % 3), 6. + (index // 3) * .3]  # within reach 5 of the sender
    sent, heard = [], []
    for tick in range(20):
        symbol = tick % 4
        events = world.step({identity: f"signal_{symbol}" if identity == 1 else "rest" for identity in world.agents})
        sent.append(symbol)
        heard += [(symbol, e["symbol"], e["noisy"]) for e in events if e["type"] == "message_received"]
    return world, heard


def test_noise_replaces_symbols_at_the_configured_rate():
    _, clean = chorus(0.)
    assert len(clean) == 24 * 20 and all(s == h and not noisy for s, h, noisy in clean)
    _, garbled = chorus(1.)
    assert all(s != h and noisy for s, h, noisy in garbled)
    substitutes = {}
    for s, h, _ in garbled:
        substitutes[(h - s) % 4] = substitutes.get((h - s) % 4, 0) + 1
    assert set(substitutes) == {1, 2, 3} and min(substitutes.values()) > len(garbled) / 3 * .7
    _, mixed = chorus(.25)
    rate = sum(noisy for _, _, noisy in mixed) / len(mixed)
    assert .18 < rate < .32  # 480 deliveries: 0.25 +/- 3.5 standard errors
    assert all((s != h) == noisy for s, h, noisy in mixed)


def test_call_reach_is_a_hard_range_around_the_sender():
    world = quiet_world(population=3)
    reach = 2. + 3. * world.agents[1].anatomy.sensing
    world.agents[2].position = [5. + reach - .05, 5.]
    world.agents[3].position = [5. + reach + .05, 5.]
    world.step({1: "signal_2", 2: "rest", 3: "rest"})
    assert [m["symbol"] for m in world.observe(2)["messages"]] == [2]
    assert world.observe(3)["messages"] == []


# Checkpoint schema -----------------------------------------------------------------------------------

def test_matrix_and_old_life8_states_are_refused(tmp_path):
    world = quiet_world()
    state = json.loads(json.dumps(world.to_dict()))
    assert state["schema_version"] != "matrix-state-v1"
    for field, old in (("schema_version", "matrix-state-v1"), ("model_version", "persistent-ecology-v1")):
        with pytest.raises(ValueError, match="incompatible"):
            World.from_dict({**state, field: old})
    controller_state = json.loads(json.dumps(state))
    controller_state["agents"][0]["controller"]["schema_version"] = "local-factored-q-controller-v2"
    with pytest.raises(ValueError, match="controller"):
        World.from_dict(controller_state)
    path = tmp_path / "checkpoint.json"
    save(world, path)
    envelope = json.loads(path.read_text())
    assert envelope["schema"] == CHECKPOINT_SCHEMA != "matrix-checkpoint-v1"
    envelope["schema"] = "matrix-checkpoint-v1"
    path.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match="schema"):
        load(path)


def test_demo_memory_with_private_fields_is_refused():
    world, teacher, _ = culture_world()
    world.step({1: "eat", 2: "rest"})
    state = json.loads(json.dumps(world.to_dict()))
    state["agents"][0]["memory"]["last_demo"]["observation"]["energy_band"] = "low"
    with pytest.raises(ValueError, match="private"):
        World.from_dict(state)


# Provenance ------------------------------------------------------------------------------------------

def test_origin_records_the_source_commit_and_licence():
    origin = json.loads((LIFE8 / "ORIGIN.json").read_text(encoding="utf-8"))
    assert origin["source_commit"].startswith("2cf045a")
    assert {"matrix_sim/world.py", "matrix_sim/learning.py", "tests/test_world.py", "LICENSE"} <= set(origin["files"])
    assert all(len(row["sha256"]) == 64 for row in origin["files"].values())
    licence = (LIFE8 / "LICENSE-matrix.txt").read_bytes()
    assert hashlib.sha256(licence).hexdigest() == origin["files"]["LICENSE"]["sha256"]
    assert b"MIT License" in licence
