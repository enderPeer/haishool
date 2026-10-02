"""Discovery harness: channel switch, information measures, controls and the known null.

Controls (two-player signalling game with the life8 learner, measured 2026-10-02,
docs/life8/results/discovery-signalling-game-controls.json, seeds 1-6, frame-free cue):
shared payoff: sender information 0.76-0.80 bits (default learner) and 0.59-0.66
(one-step learner, the review's rule); sender pays alone: 0.03-0.10 and 0.01-0.20.
The review's own framing (cue = food east/west, one-step learner) gives 0.60-0.66 vs
0.005-0.098, the review's 0.55-0.66 vs 0.005-0.098.
"""
import random
import statistics

import pytest

from haishool.life8 import discovery, learning
from haishool.life8.checkpoint import digest
from haishool.life8.config import Config
from haishool.life8.entities import Anatomy
from haishool.life8.world import World


def quiet_world(seed=3, **overrides):
    values = dict(population=3, food_patches=1, initial_objects=2, automatic_reproduction=False,
                  learning=False, signal_noise=0., food_regrowth=0., max_age=400)
    values.update(overrides)
    world = World.create(seed=seed, config=Config(**values))
    for index, agent in enumerate(sorted(world.agents.values(), key=lambda a: a.id)):
        agent.anatomy = Anatomy()
        agent.position = [5. + 1.5 * (index % 3), 5. + (index // 3) * .3]
    for item in list(world.foods.values()) + list(world.objects.values()):
        item.position = [20., 15.]
    return world


def chorus(channel, receivers=12, ticks=40, noise=0.):
    world = quiet_world(population=receivers + 1, channel=channel, signal_noise=noise)
    heard = []
    for tick in range(ticks):
        events = world.step({i: f"signal_{tick % 4}" if i == 1 else "rest" for i in world.agents})
        heard += [(tick % 4, e["symbol"]) for e in events if e["type"] == "message_received"]
    return world, heard


# ---------------------------------------------------------------- channel switch

def test_config_accepts_only_known_channels():
    assert Config().channel == "normal"
    for channel in ("normal", "scrambled", "masked"):
        assert Config(channel=channel).channel == channel
    with pytest.raises(ValueError):
        Config(channel="silent")


def test_scrambled_channel_destroys_content_without_touching_the_world_rng():
    normal, clean = chorus("normal")
    scrambled, garbled = chorus("scrambled")
    assert len(clean) == len(garbled) == 12 * 40
    assert all(sent == got for sent, got in clean)
    # Uniform symbols, independent of what was sent.
    counts = [sum(got == s for _, got in garbled) for s in range(4)]
    assert min(counts) > len(garbled) / 4 * .75
    assert discovery.mutual_information([s for s, _ in garbled], [g for _, g in garbled]) < .05
    # The world RNG stream and every cost are unchanged; only the scramble RNG moved.
    assert normal.rng.getstate() == scrambled.rng.getstate()
    assert [round(a.energy, 12) for a in normal.agents.values()] == [round(a.energy, 12) for a in scrambled.agents.values()]
    assert normal.channel_rng.getstate() != scrambled.channel_rng.getstate()


def test_masked_channel_delivers_the_call_but_hides_its_symbol_from_the_controller():
    normal, _ = chorus("normal", receivers=2, ticks=1)
    masked, _ = chorus("masked", receivers=2, ticks=1)
    seen, hidden = normal.observe(2)["messages"], masked.observe(2)["messages"]
    assert [m["symbol"] for m in seen] == [0]
    assert len(hidden) == 1 and "symbol" not in hidden[0] and hidden[0]["sender"] == 1
    assert not learning.observation_features(masked.observe(2)).get("heard_symbol")
    assert learning.observation_features(normal.observe(2)).get("heard_symbol") not in (None, "none")
    assert normal.agents[1].energy == masked.agents[1].energy


@pytest.mark.parametrize("channel", ["scrambled", "masked"])
def test_channel_runs_continue_exactly_from_a_checkpoint(channel):
    config = Config(channel=channel, log="compact", population=16)
    direct = World.create(seed=5, config=config)
    for _ in range(60):
        direct.step()
    split = World.create(seed=5, config=config)
    for _ in range(25):
        split.step()
    resumed = World.from_dict(split.to_dict())
    for _ in range(35):
        resumed.step()
    assert digest(resumed.to_dict()) == digest(direct.to_dict())
    assert resumed.to_dict()["config"]["channel"] == channel


# ---------------------------------------------------------------- measures

def test_mutual_information_and_permutation_test():
    rng = random.Random(1)
    xs = [rng.randrange(4) for _ in range(2000)]
    assert discovery.mutual_information(xs, xs) == pytest.approx(2., abs=.01)
    independent = discovery.permutation_test(xs, [rng.randrange(4) for _ in xs], reps=50)
    assert abs(independent["excess_bits"]) < .01 and independent["p_value"] > .01
    linked = discovery.permutation_test(xs, [x if rng.random() < .5 else rng.randrange(4) for x in xs], reps=50)
    assert linked["excess_bits"] > .3 and linked["p_value"] < .05
    # Within-group shuffles keep each group's symbol counts: a pure group effect is not information.
    groups = [i % 2 for i in range(2000)]
    ys = [g * 2 + rng.randrange(2) for g in groups]
    pooled = discovery.permutation_test(groups, ys, reps=50)
    within = discovery.permutation_test(groups, ys, reps=50, groups=groups)
    assert pooled["excess_bits"] > .9 and abs(within["excess_bits"]) < 1e-9


def test_verdict_needs_all_three_with_explicit_thresholds():
    sender = {"joint": {"excess_bits": .3, "p_value": .005}, "within_sender": {"excess_bits": .25, "p_value": .005}}
    receiver = {"normal": {"pooled": {"excess_bits": .2, "p_value": .005}}, "excess_over_scrambled_bits": .15}
    effect = lambda n, mean, p: {"effects": {"normal_minus_scrambled": {"food_energy_per_agent_tick":
                                                                        {"n": n, "mean": mean, "sign_p": p}}}}
    assert discovery.verdict(sender, receiver, effect(8, .01, .008))["verdict"] == "all_three"
    assert discovery.verdict(sender, receiver, effect(4, .01, .008))["verdict"] == "partial"  # too few branches
    assert discovery.verdict(sender, receiver, None)["verdict"] == "partial"
    # the sender criterion is the within-sender test: a pooled-only excess (habits) does not pass
    habits = {"joint": {"excess_bits": .3, "p_value": .005}, "within_sender": {"excess_bits": .01, "p_value": .4}}
    assert not discovery.verdict(habits, receiver, None)["criteria"]["sender_information"]["passed"]
    # with a control run the within-sender excess must beat the control's by 0.02 bits
    control = lambda excess: {"within_sender": {"excess_bits": excess, "p_value": .005}}
    assert discovery.verdict(sender, receiver, None, sender_control=control(.2))["criteria"]["sender_information"]["passed"]
    assert not discovery.verdict(sender, receiver, None, sender_control=control(.24))["criteria"]["sender_information"]["passed"]
    weak = {"joint": {"excess_bits": .01, "p_value": .3}, "within_sender": {"excess_bits": .0, "p_value": .5}}
    none = {"normal": {"pooled": {"excess_bits": 0., "p_value": .5}}, "excess_over_scrambled_bits": 0.}
    result = discovery.verdict(weak, none, effect(8, -.001, .7))
    assert result["verdict"] == "null" and "not evidence of language" in result["interpretation"]
    assert discovery.sign_test([1] * 8) == pytest.approx(2 / 256)


def test_paired_branches_are_deterministic_and_differ_only_in_the_channel():
    _, _, state = discovery.run_events(2, {"population": 16}, 40, keep_state_at=30)
    first = discovery.branch(state, "normal", 0, 15)
    assert first == discovery.branch(state, "normal", 0, 15)
    report = discovery.branch_fitness(state, ticks=15, reps=2)
    assert report["branch_tick"] == 30 and len(report["runs"]) == 4
    assert set(report["effects"]["normal_minus_scrambled"]) == {"population", "food_energy_per_agent_tick",
                                                                  "survival", "births"}


# ---------------------------------------------------------------- controls

def game(payoff, seeds, profile=None, cue="object_near"):
    rows = []
    for seed in seeds:
        normal = discovery.signalling_game(seed, payoff, profile=profile, cue=cue)
        scrambled = discovery.signalling_game(seed, payoff, "scrambled", profile=profile, cue=cue)
        sender = discovery.sender_information(normal["events"], state=("side",), bands=(), reps=30, seed=seed)
        receiver = discovery.receiver_contrast(normal["events"], scrambled["events"], reps=30, seed=seed)
        rows.append((sender["joint"]["mi_bits"], receiver["excess_over_scrambled_bits"], normal["success_late"]))
    return rows


@pytest.mark.parametrize("profile", [None, "one_step"])
def test_positive_control_shared_payoff_forms_an_informative_convention(profile):
    if profile and profile not in getattr(learning, "PROFILES", {}):
        pytest.skip("learner profile not available")
    rows = game("shared", range(1, 4), profile)
    assert all(sender >= .5 for sender, _, _ in rows), rows
    assert all(receiver >= .4 and success >= .8 for _, receiver, success in rows), rows


@pytest.mark.parametrize("profile", [None, "one_step"])
def test_negative_control_sender_paying_alone_stays_low(profile):
    if profile and profile not in getattr(learning, "PROFILES", {}):
        pytest.skip("learner profile not available")
    individual = game("individual", range(1, 4), profile)
    shared = game("shared", range(1, 4), profile)
    limit = .1 if profile == "one_step" else .25
    assert statistics.mean(s for s, _, _ in individual) <= limit, individual
    assert statistics.mean(s for s, _, _ in individual) < statistics.mean(s for s, _, _ in shared) / 3


def test_review_framing_reproduces_with_the_one_step_learner():
    """The review's probe (sender sees food east/west, Matrix's one-step rule): 0.55-0.66 vs 0.005-0.098."""
    if "one_step" not in getattr(learning, "PROFILES", {}):
        pytest.skip("learner profile not available")
    shared = game("shared", (1, 2), "one_step", cue="food_direction")
    individual = game("individual", (1, 2), "one_step", cue="food_direction")
    assert all(.55 <= s <= .7 for s, _, _ in shared), shared
    assert all(s <= .1 for s, _, _ in individual), individual


# ---------------------------------------------------------------- the known null on real worlds

def test_real_world_reports_the_known_null():
    """Seed 1, food-regulated world, 450 ticks, window 150-450 (no fitness branches here)."""
    config = {"food_patches": 30}
    normal, world, _ = discovery.run_events(1, config, 450)
    scrambled, _, _ = discovery.run_events(1, config, 450, channel="scrambled")
    sender = discovery.sender_information(normal, reps=100, seed=1, start=150)
    receiver = discovery.receiver_contrast(normal, scrambled, reps=100, seed=1, start=150)
    assert world.tick == 450 and sender["signals"] > 200 and receiver["normal"]["deliveries"] > 500
    result = discovery.verdict(sender, receiver, None)
    assert result["verdict"] == "null", result["criteria"]
    assert sender["joint"]["excess_bits"] < discovery.THRESHOLDS["sender_excess_bits"]
    assert sender["symbol_entropy_bits"] > 1.9


@pytest.mark.slow
def test_six_real_worlds_report_the_null_with_fitness_branches():
    """Measured 2026-10-02 (docs/life8/results/discovery-real-worlds.json): 6/6 null."""
    report = discovery.measure(range(1, 7), {"food_patches": 30}, 1500, branch_at=1000, branch_ticks=200,
                               reps=8, perm_reps=200, jobs=6)
    assert report["verdicts"] == {"null": 6}
    assert report["seeds_passing"]["sender_information"] == 0
