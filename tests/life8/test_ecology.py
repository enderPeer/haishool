"""Living-world ecology (round 8): blooms, predators, injuries, call power, kin credit,
receiver and reciprocity features, the LIVING preset and the bridge hint connection."""
import json
import os

import pytest

from haishool.life8 import bridge, learning
from haishool.life8.checkpoint import digest
from haishool.life8.config import LIVING, Config, living_config
from haishool.life8.entities import Anatomy
from haishool.life8.world import World


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
    if world.eco is not None:
        for n, predator in enumerate(world.eco["predators"]):
            predator["position"] = [30., 20. - n]
    return world


def rest(world, **forced):
    actions = {identity: "rest" for identity in world.agents}
    actions.update(forced)
    return world.step(actions)


# ---------------------------------------------------------------- defaults and RNG separation

def test_default_config_has_no_ecology_record_or_bins():
    world = World.create(3, Config(population=4))
    assert world.eco is None and "ecology" not in world.to_dict() and not Config().ecology_on()
    obs = world.observe(next(iter(world.agents)))
    assert not set(learning.ECOLOGY_FEATURES) & set(obs)


def test_ecology_draws_only_from_its_own_rng():
    """Same forced actions: the world RNG is untouched by predators and blooms."""
    plain = World.create(5, Config(learning=False))
    living = World.create(5, Config(learning=False, predators=2, predator_cooldown=20, bloom_interval=5,
                                     permanent_injury=.25))
    for _ in range(30):
        rest(plain), rest(living)
    assert plain.rng.getstate() == living.rng.getstate()
    assert living.counters.get("blooms") == 6
    living.validate()


# ---------------------------------------------------------------- predators and injuries

@pytest.mark.parametrize("mass", [.5, 2.])
def test_predator_wound_scales_with_mass_and_scars_for_good(mass):
    world = quiet_world(predators=1, permanent_injury=.5, predator_damage=.25, predator_mass=1.5)
    victim = world.agents[1]
    victim.anatomy = Anatomy(body_mass=mass)
    world.eco["predators"][0]["position"] = [5.5, 5.]
    events = rest(world)
    wound = .25 * 2 * 1.5 / (1.5 + mass)
    hits = [e for e in events if e["type"] == "predation"]
    assert len(hits) == 1 and hits[0]["agent"] == 1 and hits[0]["wound"] == pytest.approx(wound)
    assert world.eco["scar"]["1"] == pytest.approx(.5 * wound)
    world.eco["predators"][0]["cooldown"] = 10 ** 6  # sated for the rest of the test: it only wanders
    for _ in range(120):  # quiet_world lifespans are 160-240 ticks
        rest(world)
    assert victim.health == pytest.approx(1 - .5 * wound)  # rest heals only up to the ceiling
    world.validate()


def test_a_sated_predator_wanders_and_lethal_wounds_are_predation_deaths():
    world = quiet_world(predators=1, predator_damage=.6, predator_cooldown=2)
    world.eco["predators"][0]["position"] = [5.5, 5.]
    reasons = []
    for _ in range(12):
        reasons += [e["reason"] for e in rest(world) if e["type"] == "death"]
        if 1 not in world.agents:
            break
    assert reasons == ["predation"]
    world.validate()


def test_health_scales_eating_and_sight():
    def bite(health, on):
        world = quiet_world(health_ability=on)
        world.foods[1].position, world.foods[1].hardness = [5., 5.], .8
        world.agents[1].health = health
        before = world.agents[1].energy
        radius = world.agents[1].anatomy.sensor_range * (.5 + .5 * health if on else 1.)
        world.step({1: "eat", 2: "rest", 3: "rest"})
        return world.agents[1].energy - before, radius
    weak, weak_sight = bite(.2, True)
    strong, strong_sight = bite(1., True)
    off_weak, _ = bite(.2, False)
    assert weak < strong and weak < off_weak and weak_sight < strong_sight


def test_strike_wounds_scale_with_attacker_over_target_mass():
    def wound(attacker_mass):
        world = quiet_world(strike_injury=.2, injury_mass_scaling=True, tools=False, log="full")
        world.agents[1].anatomy = Anatomy(body_mass=attacker_mass)
        world.agents[2].position = [5.5, 5.]
        events = world.step({1: {"type": "strike_agent", "target": 2}, 2: "eat", 3: "rest"})  # 2's eat fails: no food
        damage = [e["damage"] for e in events if e["type"] == "action" and e["agent"] == 1][0]
        assert 1 - world.agents[2].health == pytest.approx(.2 * attacker_mass * damage)
        return 1 - world.agents[2].health
    heavy, light = wound(2.), wound(.5)
    assert heavy / light == pytest.approx(4.)


# ---------------------------------------------------------------- blooms

def test_blooms_appear_spoil_and_keep_the_food_ledger_closed():
    world = quiet_world(bloom_interval=5, bloom_ttl=10, bloom_amount=16.)
    for _ in range(5):
        rest(world)
    blooms = dict(world.eco["blooms"])
    assert len(blooms) == 1 and world.foods[int(next(iter(blooms)))].amount == 16.
    for _ in range(11):
        rest(world)
    assert next(iter(blooms)) not in world.eco["blooms"]
    assert world.ledger["food_spoiled"] == pytest.approx(16.) and world.ledger["food_bloomed"] == pytest.approx(48.)
    world.validate()


def test_bloom_is_rich_seen_as_a_bloom_and_can_need_a_partner():
    world = quiet_world(bloom_interval=1, bloom_ttl=50, bloom_richness=2., bloom_partners=2)
    rest(world)
    food = world.foods[int(next(iter(world.eco["blooms"])))]
    food.position = [5., 5.]
    assert world.observe(1)["bloom_seen"] is True and world.observe(2)["bloom_seen"] is False
    before = world.agents[1].energy
    world.step({1: {"type": "eat", "target": food.id}, 2: "rest", 3: "rest"})
    assert world.agents[1].energy < before  # alone: no bite
    world.agents[2].position = [5.5, 5.]
    before, amount = world.agents[1].energy, food.amount
    world.step({1: {"type": "eat", "target": food.id}, 2: "rest", 3: "rest"})
    eaten = amount - food.amount
    assert eaten == pytest.approx(1.) and world.agents[1].energy - before > 7.  # 1 unit x 4 x 2, minus costs
    world.validate()


# ---------------------------------------------------------------- calls, kin credit, receiver features

def test_call_power_is_its_own_gene_and_carries_beyond_sight():
    def call(sensing, power):
        world = quiet_world(call_power=True, call_reach=8.)
        world.agents[1].anatomy = Anatomy(sensing=sensing)
        world.eco["call_power"]["1"] = power
        before = world.agents[1].energy
        events = world.step({1: "signal_0", 2: "rest", 3: "rest"})
        return world.messages[-1]["range"], before - world.agents[1].energy, events
    reach, cost, events = call(1., 1.5)
    assert reach == pytest.approx(12.) and reach > Anatomy().sensor_range
    assert call(2., 1.5)[0] == pytest.approx(reach)  # sensing no longer sets the reach
    assert call(1., .5)[0] == pytest.approx(4.) and call(1., .5)[1] < cost
    assert any(e["type"] == "message_received" and e["agent"] == 2 for e in events)  # 10 units away: heard


def kin_world(related=True, **overrides):
    world = quiet_world(**{"population": 2, "kin_credit": 1., "kin_window": 8, "receiver_features": True, "log": "full",
                           **overrides})
    world.agents[2].position = [7., 5.]
    world.foods[1].position = [7., 5.]
    if related:
        world.eco["pedigree"]["2"] = [1]
    return world


def kin_credits(events):
    return [e for e in events if e["type"] == "kin_credit"]


@pytest.mark.parametrize("symbol", ["signal_0", "signal_3"])
def test_kin_credit_pays_the_call_relatedness_times_the_hearers_advantage(symbol):
    world = kin_world()
    first = world.step({1: symbol, 2: "rest"})
    events = world.step({1: "rest", 2: "eat"})
    before = {e["agent"]: e for e in first if e["type"] == "action"}
    action = {e["agent"]: e for e in events if e["type"] == "action"}
    assert action[2]["reward"] > 3
    # advantage = the hearer's reward minus its running mean (its tick-1 reward so far)
    advantage = action[2]["reward"] - before[2]["reward"]
    gamma = world.agents[1].controller["gamma"]
    [credit] = kin_credits(events)
    assert credit["agent"] == 1 and credit["action"] == symbol and credit["message_id"] == 1 and credit["call_tick"] == 1
    assert credit["credit"] == pytest.approx(.5 * advantage * gamma)  # discounted by one tick
    assert "kin_credit" not in action[1]  # never booked on the sender's later action ("rest")
    assert action[1]["reward"] < 0  # the organism's own outcome (last_outcome) is unchanged
    assert world.agents[1].last_outcome == pytest.approx(action[1]["reward"])
    assert not kin_credits(first)  # the call earns nothing in its own tick


def test_kin_credit_is_zero_for_strangers_and_when_switched_off():
    for world in (kin_world(related=False), kin_world(kin_credit=0.)):
        world.step({1: "signal_1", 2: "rest"})
        events = world.step({1: "rest", 2: "eat"})
        assert kin_credits(events) == []
        for _ in range(world.config.message_ttl):  # a call no relative heard is forgotten when it expires
            assert kin_credits(world.step({1: "rest", 2: "rest"})) == []
        assert world.eco["calls"] == {} and world.eco["kin_credit"] == {}


def test_kin_credit_does_not_depend_on_the_symbol():
    rewards = []
    for symbol in ("signal_0", "signal_1", "signal_2", "signal_3"):
        world = kin_world()
        world.step({1: symbol, 2: "rest"})
        events = world.step({1: "rest", 2: "eat"})
        rewards.append([e["credit"] for e in kin_credits(events)][0])
    assert len(set(rewards)) == 1 and rewards[0] > 0


def test_kin_credit_lands_on_the_signal_transition_only(monkeypatch):
    """Fix F1 (round-8 critic): the credit for a relative's outcome after hearing a call
    reaches the weights of the call decision (its features, its signal action) and nothing
    the sender does later; every learn() gets the organism's own reward only."""
    world = kin_world(learning=True)
    learned, credited = [], []
    original_learn, original_credit = learning.learn, learning.credit_decision

    def spy_learn(controller, observation, action, reward, *args, **kwargs):
        learned.append((id(controller), action, reward))
        return original_learn(controller, observation, action, reward, *args, **kwargs)

    def spy_credit(controller, features, action, amount, **kwargs):
        credited.append((id(controller), list(features), action, amount))
        return original_credit(controller, features, action, amount, **kwargs)

    monkeypatch.setattr(learning, "learn", spy_learn)
    monkeypatch.setattr(learning, "credit_decision", spy_credit)
    sender = world.agents[1].controller
    call_features = learning.decision_features(sender, world.observe(1))
    world.step({1: "signal_2", 2: "rest"})
    call_reward = sender["memory"][-1]["reward"]
    events = world.step({1: "rest", 2: "eat"}) + world.step({1: "move_n", 2: "rest"})
    rewards = {(e["agent"], e["tick"]): e["reward"] for e in events if e["type"] == "action"}
    # learn() saw only own outcomes: the sender's later rest and move carry no kin credit
    assert [r for c, a, r in learned if c == id(sender) and a in ("rest", "move_n")] == \
        [pytest.approx(rewards[1, 2]), pytest.approx(rewards[1, 3])]
    # the credit (ticks 2 and 3 of the kin window) went to the call decision: signal_2 on its features
    assert len(credited) == 2 and all(c == id(sender) and a == "signal_2" and f == call_features
                                      for c, f, a, _ in credited)
    assert [e["credit"] for e in events if e["type"] == "kin_credit"] == [pytest.approx(c[3]) for c in credited]
    assert credited[0][3] > 0  # the hearer ate after the call
    # the remembered call transition carries the credit, so replay keeps it
    assert sender["memory"][-3]["action"] == "signal_2"
    assert sender["memory"][-3]["reward"] == pytest.approx(call_reward + credited[0][3] + credited[1][3])


def test_credit_decision_moves_only_that_decisions_weights():
    controller = learning.new_controller(replay=0, lam=0.)
    obs = {"food_direction": "e", "food_near": False, "energy_band": "low"}
    features = learning.decision_features(controller, obs)
    learning.learn(controller, obs, "signal_1", -.2, obs)
    learning.learn(controller, obs, "rest", -.1, obs)  # a later decision of the same organism
    before = json.loads(json.dumps(controller["weights"]))
    mark = learning.remembered(controller) - 1  # the call was the second-to-last remembered transition
    assert learning.credit_decision(controller, features, "signal_1", 2., remembered_at=mark)
    step = controller["alpha"] * 2. / len(features)
    for feature, row in controller["weights"].items():
        for action, value in row.items():
            expected = before[feature].get(action, 0.) + (step if feature in features and action == "signal_1" else 0.)
            assert value == pytest.approx(expected)
    assert controller["memory"][-2]["reward"] == pytest.approx(-.2 + 2.)  # replay keeps the credit
    assert controller["memory"][-1]["reward"] == pytest.approx(-.1)
    assert not learning.credit_decision(controller, features, "signal_1", 2., enabled=False)


def test_kin_credit_in_a_living_world_reaches_only_calls(monkeypatch):
    """LIVING preset, 220 ticks: every credited decision is a call (the critic measured 89%
    of the credit on non-signal actions before the fix)."""
    credited = []
    original = learning.credit_decision
    monkeypatch.setattr(learning, "credit_decision",
                        lambda c, f, a, amount, **kw: credited.append(a) or original(c, f, a, amount, **kw))
    world = World.create(3, living_config(log="compact"))
    while world.tick < 220 and not world.stop_reason:
        world.step()
    assert len(credited) > 20 and all(a.startswith("signal_") for a in credited)
    world.validate()


def test_receiver_sees_bearing_age_kinship_and_the_sender_its_own_live_call():
    world = kin_world()
    world.step({1: "signal_2", 2: "rest"})
    heard, sender = world.observe(2), world.observe(1)
    assert heard["messages"][0]["bearing"] == "w" and heard["heard_age_band"] == "fresh"
    assert heard["heard_kin_band"] == "close" and sender["own_call"] == 2 and heard["own_call"] is None
    features = learning.observation_features(heard)
    assert features["heard_bearing"] == "w" and features["heard_kin_band"] == "close"
    for _ in range(9):
        world.step({1: "rest", 2: "rest"})
    assert world.observe(1)["own_call"] is None  # expired


def test_relatedness_follows_the_pedigree():
    world = quiet_world(population=3, kin_credit=1.)
    world.eco["pedigree"].update({"2": [1], "3": [1]})
    assert world.relatedness(1, 2) == .5 and world.relatedness(2, 3) == .25 and world.relatedness(1, 1) == 1.
    other = quiet_world(population=2, kin_credit=1.)
    assert other.relatedness(1, 2) == 0.


def test_reciprocity_tallies_are_directed_and_become_a_balance_band():
    world = quiet_world(reciprocity_features=True)
    world.agents[2].position = [5.5, 5.]
    world.step({1: {"type": "share", "target": 2}, 2: "rest", 3: "rest"})
    assert world.agents[1].memory["social"]["2"]["given"] == pytest.approx(2.)
    assert world.agents[2].memory["social"]["1"]["received"] == pytest.approx(2.)
    assert world.observe(2)["neighbor_balance"] == "i_owe" and world.observe(1)["neighbor_balance"] == "they_owe"


def test_learner_sees_danger_and_rotates_its_direction_in_the_egocentric_frame():
    world = quiet_world(predators=1)
    world.eco["predators"][0]["position"] = [7., 5.]
    obs = world.observe(1)
    assert obs["danger_seen"] is True and obs["danger_direction"] == "e"
    assert learning.observation_features(obs)["danger_direction"] == "e"
    assert "danger_direction" in learning.DIRECTIONAL_FEATURES


# ---------------------------------------------------------------- continuation, preset, bridge

@pytest.mark.parametrize("channel", ["normal", "scrambled"])
def test_living_world_continues_exactly_from_a_checkpoint(channel):
    cfg = living_config(log="compact", channel=channel)
    direct = World.create(11, cfg)
    for _ in range(60):
        direct.step()
    first = World.create(11, cfg)
    for _ in range(30):
        first.step()
    resumed = World.from_dict(json.loads(json.dumps(first.to_dict())))
    for _ in range(30):
        resumed.step()
    assert digest(resumed.to_dict()) == digest(direct.to_dict())
    assert resumed.eco["pedigree"] == direct.eco["pedigree"]


def test_living_state_without_its_ecology_record_is_refused():
    state = World.create(2, living_config()).to_dict()
    del state["ecology"]
    with pytest.raises(ValueError):
        World.from_dict(state)


def test_living_preset_is_food_regulated_below_the_cap():
    world = World.create(1, living_config(log="compact"))
    populations = []
    while world.tick < 400 and not world.stop_reason:
        world.step()
        populations.append(len(world.agents))
    assert world.stop_reason is None and max(populations) < world.config.population_limit // 2
    assert world.counters.get("predator_strikes", 0) > 0 and world.counters.get("bloom_meals", 0) > 0
    world.validate()


def test_config_checks_the_ecology_fields():
    for bad in ({"permanent_injury": 1.5}, {"predators": 65}, {"ambient_temperature": 500.}, {"bloom_ttl": 0},
                {"health_ability": 1}):
        with pytest.raises(ValueError):
            Config(**bad)
    assert living_config().ecology_on() and set(LIVING) <= set(Config().to_dict())


def test_bridge_connects_the_ecology_hints_monotonically():
    rec = bridge.synthetic_world(7)
    cfg, founders = bridge.world_config(rec, base=living_config(log="compact"))
    connected, out = bridge.connect_ecology(cfg, founders)
    hints = out["ecology_hints"]
    assert hints["connected"] is True and set(hints["connected_fields"]) >= {"predators", "predator_damage",
                                                                             "ambient_temperature"}
    assert connected.predators == round(10 * hints["predator_density"])
    assert connected.ambient_temperature == pytest.approx(hints["ambient_temperature_c"])
    low = dict(founders, ecology_hints=dict(founders["ecology_hints"], predator_density=.05, attack_injury=0.))
    high = dict(founders, ecology_hints=dict(founders["ecology_hints"], predator_density=.3, attack_injury=.5))
    a, _ = bridge.connect_ecology(cfg, low)
    b, _ = bridge.connect_ecology(cfg, high)
    assert a.predators < b.predators and a.predator_damage < b.predator_damage
    world = bridge.create_world(7, connected, out)
    for _ in range(20):
        world.step()
    world.validate()


@pytest.mark.slow
def test_living_ecology_measurement_runs_on_two_seeds():
    from haishool.life8.analysis import ecology
    result = ecology.measure([1, 2], ticks=600, jobs=2, reps=4, perm_reps=50)
    assert len(result["per_seed"]) == 2
    for row in result["per_seed"]:
        assert row["runs"]["normal"]["stop_reason"] == "step_budget"
        assert row["verdict"]["verdict"] in ("all_three", "partial", "null")
