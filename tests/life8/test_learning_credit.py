"""Multi-step credit (controller v4): traces, replay, conjunctions, egocentric frame, decaying exploration.

Measured numbers are in docs/life8/README.md ("Multi-step credit") and in the
haishool.life8.analysis.credit reports.
"""
import json
import random

import pytest

from haishool.life8 import learning
from haishool.life8.analysis import credit
from haishool.life8.checkpoint import digest
from haishool.life8.config import Config
from haishool.life8.world import World


def far(direction="e", distance=3.0, **extra):
    return {"food_direction": direction, "food_near": False, "food": [{"distance": distance}],
            "energy_band": "middle", **extra}


def near(direction="e"):
    return {"food_direction": direction, "food_near": True, "food": [{"distance": 1.0}], "energy_band": "middle"}


def test_corridor_reward_only_at_food_is_learned_in_ten_episodes_and_not_by_the_one_step_rule():
    """Reward 1 only for eating within reach, 0 for every other step; 10 actions (chance 0.1).

    Measured over 40 seeds after 10 episodes: multistep 0.90 (35/40 seeds above 0.5),
    one_step 0.16 (6/40). After 40 episodes both reach 1.0: the gain is speed, not
    something the one-step rule can never learn in this clean corridor.
    """
    seeds = range(1, 41)
    multi = [credit.corridor("multistep", seed, 10)["p_toward_greedy"] for seed in seeds]
    old = [credit.corridor("one_step", seed, 10)["p_toward_greedy"] for seed in seeds]
    assert sum(multi) / len(multi) > .75 and sum(p > .5 for p in multi) >= 30
    assert sum(old) / len(old) < .3
    assert sum(multi) / len(multi) > sum(old) / len(old) + .5


def test_trace_credits_an_earlier_step_and_lambda_zero_does_not():
    available = ("rest", "move_e", "move_w", "eat")
    results = {}
    for lam in (0., .8):
        controller = learning.new_controller(lam=lam, replay=0, egocentric=False, conjunctions=False, max_memories=0)
        learning.learn(controller, far(distance=4.), "move_e", 0., far(distance=3.), actions=available)
        learning.learn(controller, {"food_direction": "s", "food_near": False}, "rest", 0.,
                       {"food_direction": "s", "food_near": False}, actions=available)
        learning.learn(controller, near(), "eat", 1., near(), terminal=True, actions=available)
        results[lam] = controller["weights"].get(learning.feature_name("food_direction", "e"), {}).get("move_e", 0.)
        assert controller["trace"] == []  # emptied at the terminal outcome
    assert results[0.] == 0. and results[.8] > 0.


def test_demonstrations_do_not_enter_the_trace():
    controller = learning.new_controller()
    learning.observe_demo(controller, far(), "move_e", 0., next_observation=near())
    assert controller["trace"] == [] and controller["demonstrations"] == 1


def test_egocentric_frame_transfers_an_approach_learned_in_one_direction_to_another():
    available = learning.ACTIONS
    values = {}
    for ego in (False, True):
        controller = learning.new_controller(egocentric=ego, replay=0, lam=0., epsilon=0.)
        for _ in range(20):
            learning.learn(controller, near(), "eat", 1., near(), terminal=True, actions=available)
            learning.learn(controller, far("e", 2.), "move_e", 0., near("e"), actions=available)
        q = learning.action_values(controller, far("s", 2.), available)
        values[ego] = q["move_s"] - max(value for action, value in q.items() if action.startswith("move_") and action != "move_s")
    assert values[False] <= 1e-12 and values[True] > .1
    # Without visible food there is no frame: the action keys are the world's.
    controller = learning.new_controller(egocentric=True)
    assert set(learning.action_values(controller, {"food_direction": "none", "food_near": False})) == set(available)


def test_egocentric_choice_is_a_world_action_and_follows_the_food_bearing():
    controller = learning.new_controller(egocentric=True, epsilon=0., replay=0)
    for _ in range(20):
        learning.learn(controller, far("e", 2.), "move_e", 1., near("e"), terminal=True)
    rng = random.Random(3)
    for direction in ("n", "ne", "se", "w", "nw"):
        assert learning.choose_action(controller, far(direction, 2.), rng) == "move_" + direction


def test_conjunction_features_and_the_heard_bearing_hook():
    controller = learning.new_controller()
    features = learning._features(controller, far("e", 2.5, target_hardness_band="low"))
    assert '["food_direction*food_distance_band","\\"n\\"|\\"close\\""]' in features  # egocentric: food reads "n"
    assert any(feature.startswith('["food_near*energy_band"') for feature in features)
    assert any(feature.startswith('["food_direction*food_distance_band*target_hardness_band*energy_band"')
               for feature in features)
    heard = far(messages=[{"symbol": 2, "age": 0}])
    assert not any("heard_bearing" in feature for feature in learning._features(controller, heard))
    heard = far(messages=[{"symbol": 2, "age": 0, "bearing": "w"}])
    names = learning._features(controller, heard)
    # The bearing is rotated with the frame (food east is "n", so a call from the west reads "s").
    assert '["heard_bearing","s"]' in names
    assert '["heard_symbol*heard_bearing","2|\\"s\\""]' in names
    plain = learning.new_controller(profile="one_step")
    assert not any("*" in feature or "distance" in feature for feature in learning._features(plain, heard))


def test_exploration_decays_with_decisions_and_epsilon_zero_is_greedy():
    controller = learning.new_controller(epsilon=.3, epsilon_floor=.1, epsilon_halflife=200)
    assert learning.current_epsilon(controller) == pytest.approx(.3)
    controller["decisions"] = 200
    assert learning.current_epsilon(controller) == pytest.approx(.3 * (.1 + .9 * .5))
    controller["decisions"] = 10 ** 6
    assert learning.current_epsilon(controller) == pytest.approx(.03, rel=1e-2)
    assert learning.current_epsilon(learning.new_controller(epsilon=0.)) == 0.
    assert learning.current_epsilon(learning.new_controller(profile="one_step")) == .15


def test_replay_is_deterministic_and_uses_no_random_numbers():
    controllers = [learning.new_controller(replay=4), learning.new_controller(replay=4)]
    rng = random.Random(9)
    state = rng.getstate()
    for controller in controllers:
        for step in range(40):
            learning.learn(controller, far(distance=3. + step % 3), "move_e" if step % 2 else "rest",
                           -.1 + (step % 5 == 0), far(distance=2. + step % 3))
    assert rng.getstate() == state
    assert controllers[0] == controllers[1]
    assert controllers[0]["memory"][-1]["terminal"] is False
    none = learning.new_controller(replay=4, max_memories=0)
    learning.learn(none, far(), "rest", 0., far())  # no memory, nothing to replay
    assert none["updates"] == 1


def test_reward_treats_action_names_alike():
    """Relabelling two non-move actions in the experience relabels the learned weights, nothing more."""
    swap = {"eat": "strike", "strike": "eat", "signal_0": "teach", "teach": "signal_0"}
    stream = [(far(), "eat", -.1), (near(), "strike", .4), (far(distance=4.), "signal_0", -.3),
              (near(), "teach", .2), (far(), "move_e", -.1), (near(), "eat", 1.)] * 6
    a, b = learning.new_controller(), learning.new_controller()
    for observation, action, reward in stream:
        learning.learn(a, observation, action, reward, near())
        learning.learn(b, observation, swap.get(action, action), reward, near())
    for feature, row in a["weights"].items():
        assert {swap.get(action, action): value for action, value in row.items()} == b["weights"][feature]


def test_controller_from_biases_seeds_weights_and_otherwise_matches_new_controller():
    assert learning.controller_from_biases(None) == learning.new_controller()
    assert learning.controller_from_biases({}, alpha=.5, epsilon=.05) == learning.new_controller(alpha=.5, epsilon=.05)
    seeded = learning.controller_from_biases({"eat": .5, "signal_2": -.2}, alpha=.1, epsilon=0.)
    assert seeded["alpha"] == .1 and seeded["epsilon"] == 0. and seeded["memory"] == [] and seeded["trace"] == []
    assert seeded["weights"] == {learning.BIAS_FEATURE: {"eat": .5, "signal_2": -.2}}
    learning.validate_controller(seeded)
    assert learning.choose_action(seeded, {"food_direction": "none", "food_near": False}, random.Random(1)) == "eat"
    near_key = learning.feature_name("food_near", True)
    by_feature = learning.controller_from_biases({near_key: {"eat": 1.}, learning.BIAS_FEATURE: {"rest": .1}})
    assert by_feature["feature_order"] == [learning.BIAS_FEATURE, near_key]
    assert learning.action_values(by_feature, near(), ("eat", "rest")) == {"eat": 1., "rest": .1}
    learning.validate_controller(by_feature)
    for bad in ({"fly": 1.}, {"eat": float("nan")}, {"eat": 1e9}, [1, 2]):
        with pytest.raises(ValueError):
            learning.controller_from_biases(bad)


def test_one_step_profile_has_no_trace_replay_frame_or_conjunctions():
    controller = learning.new_controller(profile="one_step")
    assert (controller["lam"], controller["trace_steps"], controller["replay"], controller["egocentric"],
            controller["conjunctions"], learning.current_epsilon(controller)) == (0., 0, 0, False, False, .15)
    with learning.use_profile("one_step", lam=.8):
        assert learning.new_controller()["trace_steps"] == learning.trace_length(.9, .8)
    assert learning.new_controller()["lam"] == learning.PROFILES[learning.DEFAULT_PROFILE]["lam"]
    with pytest.raises(ValueError):
        learning.new_controller(replay=learning.MAX_REPLAY + 1)


def test_controller_state_survives_json_mid_trace_and_continues_identically():
    rng_a, rng_b = random.Random(5), random.Random(5)
    a = learning.new_controller()
    for step in range(12):
        observation = far(distance=2. + step % 4)
        action = learning.choose_action(a, observation, rng_a)
        learning.learn(a, observation, action, -.1, far(distance=1.5 + step % 3))
    assert a["trace"] and a["memory"]
    b = json.loads(json.dumps(a))
    learning.validate_controller(b)
    rng_b.setstate(rng_a.getstate())
    for step in range(12):
        observation = far(distance=2. + step % 4)
        for controller, rng in ((a, rng_a), (b, rng_b)):
            action = learning.choose_action(controller, observation, rng)
            learning.learn(controller, observation, action, .2 if action == "move_e" else -.1, near())
    assert a == b


def test_world_checkpoint_continuation_with_the_multistep_learner():
    config = Config(population=10, food_patches=20, log="compact")
    direct = World.create(seed=4, config=config)
    for _ in range(30):
        direct.step()
    resumed = World.from_dict(json.loads(json.dumps(direct.to_dict())))
    for world in (direct, resumed):
        for _ in range(30):
            world.step()
    assert digest(direct.to_dict()) == digest(resumed.to_dict())
    controller = next(iter(direct.agents.values())).controller
    assert controller["schema_version"] == learning.SCHEMA_VERSION and controller["replay"] == 4


def test_measurement_recording_does_not_change_the_world():
    config = Config(population=10, food_patches=20, log="compact")
    plain = World.create(seed=2, config=config)
    for _ in range(25):
        plain.step()
    rows = []
    with credit._recording(rows):
        recorded = World.create(seed=2, config=config)
        for _ in range(25):
            recorded.step()
    assert rows and digest(plain.to_dict()) == digest(recorded.to_dict())


def test_world_probe_smoke():
    row = credit.world_probe("multistep", 1, 20, {**credit.SPARSE, "log": "compact"})
    assert row["founders"] == 16 and row["ticks_run"] == 20
    assert credit.world_probe("heuristic", 1, 20, {**credit.SPARSE, "log": "compact"}, heuristic=True)["births"] == 0


@pytest.mark.slow
def test_close_soft_food_approach_rises_against_the_one_step_rule_in_the_sparse_world():
    """Sparse world, seeds 1-6, 1500 ticks: founders' P(toward | food visible, out of reach).

    Recorded numbers: docs/life8/README.md, "Multi-step credit".
    """
    report = credit.measure("sparse", ["multistep", "one_step"], range(1, 7), 1500, jobs=6)
    new, old = (report["totals"][name]["founder_rates"] for name in ("multistep", "one_step"))
    pooled = lambda rates: sum(c["p_toward"] * c["n_far"] for b, c in rates.items() if b != "0-50" and c["p_toward"] is not None) \
        / sum(c["n_far"] for b, c in rates.items() if b != "0-50" and c["p_toward"] is not None)
    assert pooled(new) > pooled(old) + .02
