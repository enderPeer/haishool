import copy
import json
import random

import pytest

from haishool.life8.learning import (ACTIONS, action_values, choose_action, learn, new_controller, observation_features,
                                 observe_demo, state_key, validate_controller)


def test_newborn_controller_has_no_policy_memories_or_symbol_dictionary():
    adult = new_controller()
    observe_demo(adult, {"cue": "food"}, "eat", 8)
    newborn = new_controller()
    assert newborn["q"] == {} and newborn["memory"] == []
    assert newborn["weights"] == {} and newborn["feature_order"] == []
    assert newborn["updates"] == newborn["demonstrations"] == 0
    assert all(name not in newborn for name in ("dictionary", "technology", "recipes"))
    assert {"signal_0", "signal_1", "signal_2", "signal_3", "reproduce", "strike_agent", "strike_object"} <= set(ACTIONS)


def test_local_projection_excludes_identifiers_but_retains_raw_heard_symbols():
    observation = {"self": {"id": 1, "inventory_count": 1}, "energy_band": "low", "food_near": True,
                   "food_direction": "e", "food": [{"id": 32, "amount": 5}],
                   "messages": [{"sender": 2, "symbol": 0, "age": 2}, {"sender": 3, "symbol": 2, "age": 0}]}
    other = copy.deepcopy(observation)
    other["self"]["id"] = 999
    other["food"][0]["id"] = 77
    other["messages"][1]["sender"] = 90
    assert state_key(observation) == state_key(other)
    features = observation_features(observation)
    assert features["heard_symbol"] == 2
    assert not any(name in features for name in ("sender", "meaning", "agent_id"))
    other["messages"][1]["symbol"] = 1
    assert state_key(observation) != state_key(other)


def test_learned_action_transfers_across_unseen_irrelevant_local_context():
    controller = new_controller(epsilon=0, gamma=0)
    experienced = {"food_near": True, "energy_band": "low", "neighbor_familiar": False,
                   "messages": [{"symbol": 0, "age": 0}]}
    for _ in range(8):
        learn(controller, experienced, "eat", 4, experienced, terminal=True)
        learn(controller, experienced, "rest", -.1, experienced, terminal=True)
    novel = {**experienced, "neighbor_familiar": True, "messages": [{"symbol": 3, "age": 0}]}
    assert state_key(novel) not in controller["q"]
    assert action_values(controller, novel, ("eat", "rest"))["eat"] > 0
    for seed in range(20):
        assert choose_action(controller, novel, random.Random(seed), actions=("eat", "rest")) == "eat"


def test_shared_feature_weights_still_learn_context_dependent_actions():
    controller = new_controller(alpha=.35, gamma=0, epsilon=0)
    for _ in range(80):
        for near in (False, True):
            state = {"food_near": near, "energy_band": "low", "neighbor_familiar": False}
            for action in ("eat", "move_e"):
                useful = action == ("eat" if near else "move_e")
                learn(controller, state, action, 3 if useful else -1, state, terminal=True)
    for near in (False, True):
        novel = {"food_near": near, "energy_band": "low", "neighbor_familiar": True,
                 "messages": [{"symbol": 2, "age": 0}]}
        assert choose_action(controller, novel, random.Random(1), actions=("eat", "move_e")) == ("eat" if near else "move_e")


def test_joint_q_table_is_diagnostic_and_does_not_override_factored_policy():
    controller = new_controller(epsilon=0, gamma=0)
    state = {"cue": 1}
    learn(controller, state, "eat", 2, state, terminal=True)
    controller["q"][state_key(state)]["rest"] = 100
    assert choose_action(controller, state, random.Random(1), actions=("eat", "rest")) == "eat"


def test_untrained_actions_have_no_hidden_preferred_tool_or_word():
    controller, rng = new_controller(epsilon=0), random.Random(41)
    observed = {choose_action(controller, {"cue": "unseen"}, rng) for _ in range(800)}
    assert observed == set(ACTIONS)
    assert all(not values for values in controller["q"].values())


@pytest.mark.parametrize("feature", ["target_hardness_band", "target_open_band", "held_mass_band", "held_hardness_band",
                                     "held_friction_band", "held_durability_band", "held_sharpness_band",
                                     "held_temperature_band", "held_bond_band", "neighbor_familiar", "neighbor_last_exchange"])
def test_local_physical_and_experience_bins_are_available_to_the_controller(feature):
    a = {"food_near": True, feature: 0}
    b = {"food_near": True, feature: 1}
    assert state_key(a) != state_key(b)


def test_controller_learns_opposite_actions_from_observed_outcomes():
    controller, rng = new_controller(alpha=.3, gamma=0, epsilon=.2), random.Random(53)
    actions = ("eat", "rest")
    totals = []
    for episode in range(700):
        cue = episode % 2
        observation = {"cue": cue}
        action = choose_action(controller, observation, rng, actions=actions)
        reward = 2 if action == ("eat" if cue == 0 else "rest") else -2
        totals.append(reward)
        learn(controller, observation, action, reward, observation, terminal=True, actions=actions)
    controller["epsilon"] = 0
    assert {choose_action(controller, {"cue": 0}, rng, actions=actions) for _ in range(30)} == {"eat"}
    assert {choose_action(controller, {"cue": 1}, rng, actions=actions) for _ in range(30)} == {"rest"}
    assert sum(totals[-200:]) > 200  # above the uninformed two-action expected reward of zero
    validate_controller(controller)


def test_identical_symbols_acquire_only_different_experienced_values():
    left, right = new_controller(gamma=0, epsilon=0), new_controller(gamma=0, epsilon=0)
    observation = {"food_near": True, "messages": []}
    for _ in range(30):
        for action in ("signal_0", "signal_1"):
            learn(left, observation, action, 2 if action == "signal_0" else -2, observation, terminal=True)
            learn(right, observation, action, 2 if action == "signal_1" else -2, observation, terminal=True)
    assert choose_action(left, observation, random.Random(1), actions=("signal_0", "signal_1")) == "signal_0"
    assert choose_action(right, observation, random.Random(1), actions=("signal_0", "signal_1")) == "signal_1"


def test_disabled_intervention_changes_neither_policy_nor_memories():
    controller, rng = new_controller(), random.Random(1)
    original = copy.deepcopy(controller)
    for _ in range(50):
        action = choose_action(controller, {"cue": 1}, rng, enabled=False)
        learn(controller, {"cue": 1}, action, 10, {"cue": 1}, enabled=False)
        assert not observe_demo(controller, {"cue": 1}, "eat", 20, enabled=False)
    assert controller == original


def test_local_demonstration_is_a_sample_not_a_policy_copy():
    controller = new_controller(epsilon=0)
    original_observation = {"food_direction": "e", "food_near": False}
    assert observe_demo(controller, original_observation, "move_e", {"energy_delta": 3, "health_delta": -.1})
    assert controller["demonstrations"] == 1 and controller["updates"] == 0
    assert controller["memory"][0]["source"] == "local_demonstration"
    assert controller["memory"][0]["reward"] == pytest.approx(2.2)
    assert choose_action(controller, original_observation, random.Random(1)) == "move_e"
    original_observation["food_direction"] = "w"
    assert controller["memory"][0]["observation"]["food_direction"] == "e"
    assert all(set(values) <= {"move_e"} for values in controller["q"].values())


def advance(controller, rng, steps, start=0):
    history = []
    for i in range(start, start + steps):
        observation = {"food_near": i % 3 == 0, "energy_band": i % 4,
                       "messages": [{"sender": 100, "symbol": i % 4, "age": 0}]}
        action = choose_action(controller, observation, rng)
        reward = 3 if action == "eat" and observation["food_near"] else -.2
        learn(controller, observation, action, reward, {**observation, "energy_band": (i + 1) % 4})
        if i % 13 == 0:
            observe_demo(controller, observation, "rest", -.1)
        history.append((action, reward))
    return history


def test_json_and_rng_continuation_is_exact_including_eviction_and_memories():
    controller = new_controller(max_states=5, max_memories=4)
    rng = random.Random(902)
    advance(controller, rng, 75)
    restored = json.loads(json.dumps(controller, sort_keys=True, allow_nan=False))
    state = json.loads(json.dumps(rng.getstate()))
    resumed_rng = random.Random()
    resumed_rng.setstate((state[0], tuple(state[1]), state[2]))
    assert advance(controller, rng, 120, 75) == advance(restored, resumed_rng, 120, 75)
    assert controller == restored
    assert rng.getstate() == resumed_rng.getstate()
    assert len(controller["q"]) <= 5 and len(controller["memory"]) == 4
    validate_controller(controller)


def test_controllers_use_supplied_rng_without_global_random_state():
    before = random.getstate()
    a, b = new_controller(), new_controller()
    assert advance(a, random.Random(2), 100) == advance(b, random.Random(2), 100)
    assert a == b
    assert random.getstate() == before


@pytest.mark.parametrize("reward", [float("nan"), float("inf"), "technology", True])
def test_non_numeric_and_nonfinite_rewards_are_rejected(reward):
    with pytest.raises(ValueError):
        learn(new_controller(), {"cue": 0}, "eat", reward, {"cue": 0})


def test_finite_controller_rejects_corrupt_checkpoint_and_bounds_large_rewards():
    controller = new_controller(reward_clip=10, gamma=.9, alpha=1)
    for _ in range(100):
        learn(controller, {"cue": 1}, "eat", 1e300, {"cue": 1})
    assert controller["q"][state_key({"cue": 1})]["eat"] <= 100 + 1e-9
    validate_controller(controller)
    corrupt = copy.deepcopy(controller)
    corrupt["q"][state_key({"cue": 1})]["eat"] = float("nan")
    with pytest.raises(ValueError):
        validate_controller(corrupt)
    corrupt = copy.deepcopy(controller)
    corrupt["schema_version"] = "future-controller"
    with pytest.raises(ValueError, match="schema"):
        validate_controller(corrupt)


def test_feature_vocabulary_and_weights_remain_bounded_through_eviction_and_resume():
    controller = new_controller(max_features=5, max_states=3, max_memories=2, reward_clip=3, gamma=.5)
    for cue in range(100):
        learn(controller, {"cue": cue}, "eat", 1e20, {"cue": cue + 1})
    assert len(controller["weights"]) == 5
    assert len(controller["feature_order"]) == 5
    assert all(abs(weight) <= 6 for values in controller["weights"].values() for weight in values.values())
    validate_controller(controller)
    restored = json.loads(json.dumps(controller, sort_keys=True))
    learn(controller, {"cue": 101}, "rest", -1e20, {"cue": 102})
    learn(restored, {"cue": 101}, "rest", -1e20, {"cue": 102})
    assert restored == controller
    corrupt = copy.deepcopy(controller)
    corrupt["weights"][corrupt["feature_order"][0]]["eat"] = 7
    with pytest.raises(ValueError, match="feature weight"):
        validate_controller(corrupt)


def test_normalized_feature_update_does_not_scale_with_irrelevant_feature_count():
    small, large = new_controller(alpha=.25, gamma=0), new_controller(alpha=.25, gamma=0)
    a, b = {"cue": 1}, {"cue": 1, **{f"extra_{i}": 0 for i in range(20)}}
    learn(small, a, "eat", 4, a, terminal=True)
    learn(large, b, "eat", 4, b, terminal=True)
    assert action_values(small, a, ("eat",))["eat"] == pytest.approx(1)
    assert action_values(large, b, ("eat",))["eat"] == pytest.approx(1)


def test_unknown_actions_and_nonfinite_observations_are_rejected():
    with pytest.raises(ValueError, match="canonical"):
        choose_action(new_controller(), {}, random.Random(1), actions=("invent_fire",))
    with pytest.raises(ValueError, match="unknown action"):
        learn(new_controller(), {}, "invent_fire", 10, {})
    with pytest.raises(ValueError, match="finite JSON"):
        state_key({"cue": float("nan")})
