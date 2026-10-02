"""Visitors (haishool/life8/visitor.py) and the paired arms (visitor_arms.py), with fake policies only."""
import json

import pytest

from haishool.life8 import experience, learning, visitor as V, visitor_arms as A
from haishool.life8.config import Config, living_config
from haishool.life8.world import World


def _twins(seed, config, ids, ticks, policy=None, valid="engine"):
    plain, driven = World.create(seed, config), World.create(seed, config)
    visitors = V.Visitors(policy or V.FakePolicy("mirror"), ids=ids, valid=valid)
    for _ in range(ticks):
        plain.step()
        visitors.step(driven)
    return plain, driven, visitors


@pytest.mark.parametrize("config", [Config(log="compact"), living_config(log="compact"), Config()],
                         ids=["default-compact", "living", "default-full"])
def test_twins_are_identical_when_the_policy_reproduces_the_learner(config):
    plain, driven, visitors = _twins(5, config, [1, 4, 9], 60)
    assert plain.to_dict() == driven.to_dict()
    assert visitors.decisions > 0 and visitors.fallbacks == 0
    assert visitors.matches_native == visitors.decisions


def test_any_other_visitor_action_makes_the_twin_differ():
    plain, driven, visitors = _twins(5, Config(log="compact"), [3], 30, V.FakePolicy("rest"), valid="body")
    assert plain.to_dict() != driven.to_dict()
    assert visitors.actions == {"rest": visitors.decisions}


def test_visitor_sees_only_its_own_observation_and_acts_only_through_its_own_body():
    world = World.create(7, living_config(log="compact"))
    for _ in range(20):
        world.step()
    ids = sorted(world.agents)[:3]
    policy = V.FakePolicy("random", seed=1)
    visitors = V.Visitors(policy, ids=ids)
    expected = {i: experience.prompt_for(experience.features_of(world.observe(world.agents[i]))) for i in ids}
    allowed = {i: V.allowed_actions(world, world.agents[i]) for i in ids}
    overrides = visitors.decide(world)
    # one prompt per visitor, built from that visitor's own local features only
    assert policy.calls == [[expected[i] for i in ids]]
    for prompt in policy.calls[0]:
        assert prompt.startswith("q life8 see ") and prompt.endswith(" want thrive. a")
    # every other organism does what its own controller drew; a visitor's action is one engine word
    for identity, action in overrides.items():
        if identity in ids:
            assert action in allowed[identity] and action in experience.ACTION_WORDS
        else:
            assert action == visitors.native_choices[identity]
    events = world.step(overrides)
    for identity in ids:
        if identity in world.agents:
            assert world.agents[identity].last_action == overrides[identity]
    world.validate()  # the ledgers stay closed
    assert isinstance(events, list)


def test_costs_are_the_engines():
    config = Config(log="compact", automatic_reproduction=False, culture=False)
    world = World.create(11, config)
    visitor = 6
    agent = world.agents[visitor]
    energy, health = agent.energy, agent.health
    upkeep = config.base_metabolism * agent.anatomy.metabolic_factor
    visitors = V.Visitors(V.FakePolicy("rest"), ids=[visitor])
    visitors.step(world)
    assert health == 1.0
    assert world.agents[visitor].energy == pytest.approx(energy - config.action_cost * .25 - upkeep, abs=1e-12)
    # a call costs what it costs a native: action cost plus the sensing-scaled signal cost
    agent = world.agents[visitor]
    assert agent.anatomy.voice
    energy, sent = agent.energy, world.counters.get("signals_sent", 0)
    visitors.policy = V.FakePolicy(lambda prompt, ok: "signal_2")
    visitors.step(world)
    cost = config.action_cost + config.signal_cost * (1 + .15 * agent.anatomy.sensing) + upkeep
    assert world.agents[visitor].energy == pytest.approx(energy - cost, abs=1e-12)
    assert world.counters["signals_sent"] >= sent + 1
    world.validate()


def test_body_mode_refuses_what_the_body_cannot_do_and_falls_back_to_rest():
    world = World.create(2, Config(log="compact"))
    agent = world.agents[1]
    agent.anatomy.voice = 0
    allowed = V.allowed_actions(world, agent)
    assert not agent.inventory
    assert not {"drop", "dismantle", "heat", "combine"} & set(allowed)
    assert not any(a.startswith("signal_") for a in allowed)
    assert "rest" in allowed and "eat" in allowed
    assert V.allowed_actions(world, agent, "engine") == world._enabled_actions()
    visitors = V.Visitors(V.FakePolicy(lambda prompt, ok: "drop"), ids=[1])
    overrides = visitors.decide(world)
    assert overrides[1] == "rest" and visitors.fallbacks == 1


def test_offspring_of_visitors_are_ordinary_inhabitants():
    config = Config(log="compact")
    world = World.create(3, config)
    visitors = V.Visitors(V.FakePolicy("rest"), ids=[1, 2])
    children = []
    while world.tick < 120 and not children:
        events = visitors.step(world)
        children = [e for e in events if e["type"] == "birth" and e["parent"] in visitors.ids]
    assert children, "a resting founder reproduces once mature"
    child = world.agents[children[0]["agent"]]
    assert child.id not in visitors.ids
    # a newborn starts from a blank controller (frozen genome): nothing learned is inherited
    assert child.controller == learning.new_controller(max_memories=config.memory_limit)
    overrides = visitors.decide(world)
    assert overrides[child.id] == visitors.native_choices[child.id]
    assert child.id not in visitors.last_prompts
    snap = visitors.snapshot(world)
    flags = {a["id"]: a.get("visitor", False) for a in snap["agents"]}
    assert flags[child.id] is False


def test_snapshots_and_archives_flag_visitors(tmp_path):
    world = World.create(4, living_config(log="compact"))
    visitors = V.Visitors(V.FakePolicy("random"), ids=[2, 7], label="traveller")
    snap = visitors.snapshot(world)
    flagged = sorted(a["id"] for a in snap["agents"] if a.get("visitor"))
    assert flagged == [2, 7]
    assert all(a["visitor_label"] == "traveller" and a["visitor_since"] == 0 for a in snap["agents"] if a.get("visitor"))
    run = V.write_archive(world, visitors, tmp_path / "arch", 20, sample_every=10)
    manifest = json.loads((run / "manifest.json").read_text())
    assert manifest["visitor_world"] is True and manifest["visitors"]["ids"] == [2, 7]
    rows = [json.loads(line) for line in (run / "snapshots.jsonl").read_text().splitlines()]
    assert len(rows) == 3
    from haishool.life8 import view3d
    data = view3d.collect(run, replay="off", log=lambda *_: None)
    assert sorted(i for i, r in data["scan"].ind.items() if r["visitor"]) == [2, 7]


def test_pick_takes_the_best_allowed_word():
    scores = {"drop": 0., "eat": -1., "rest": -2.}
    assert V.pick(scores, ["eat", "rest"]) == "eat"
    assert V.pick(scores, None) == "drop"
    assert V.pick({}, ["eat"]) == "rest"
    assert V.pick({"eat": -1., "move_n": -1.}, ["move_n", "eat"]) == "move_n"  # tie: ACTION_WORDS order


def test_founder_order_is_nested_and_planet_determined():
    entry = {"id": "plain-3", "source": "plain", "seed": 3, "config": Config().to_dict()}
    world = World.create(3, Config())
    order = A.founder_order(entry, world)
    assert sorted(order) == sorted(world.agents) and order == A.founder_order(entry, World.create(3, Config()))
    one, tenth, every = (A.arm_ids(arm, order) for arm in ("one", "tenth", "all"))
    assert len(one) == 1 and len(tenth) == 2 and every == order
    assert one == tenth[:1] and tenth == every[:2]


def _entry(seed=3):
    return {"id": f"plain-{seed}", "source": "plain", "seed": seed, "config": living_config(log="compact").to_dict()}


def test_arm_runs_are_deterministic():
    first = A.run_arm(_entry(), "tenth", ticks=40, policy="fake:random")
    second = A.run_arm(_entry(), "tenth", ticks=40, policy="fake:random")
    first.pop("seconds"), second.pop("seconds")
    assert first == second
    assert first["visitor_world"] is True and first["visitors"] == A.arm_ids("tenth", first["founder_order"])


def test_mirror_arms_pair_to_exact_zeros(tmp_path):
    report = A.run(tmp_path / "mirror", [_entry(3)], ["baseline", "one", "all", "elder"], ticks=60,
                   policy="fake:mirror", elder_policy="fake:mirror", elder_at=30, log=lambda *_: None)
    assert not report["errors"] and set(report["arms"]) == {"one", "all", "elder"}
    for arm, body in report["arms"].items():
        for name, stats in body["paired"].items():
            if stats["n"]:
                assert stats["mean"] == 0 and stats["zero"] == stats["n"], (arm, name)
        assert body["fallbacks"] == 0 and body["same_as_native"] == body["decisions"]
    elder = next(p for p in report["pairs"] if p["arm"] == "elder")
    assert elder["window"] == [30, 60] and len(elder["visitors"]) == 1
    # resuming skips finished runs and gives the same report
    again = A.run(tmp_path / "mirror", [_entry(3)], ["baseline", "one", "all", "elder"], ticks=60,
                  policy="fake:mirror", elder_policy="fake:mirror", elder_at=30, log=lambda *_: None)
    assert again["arms"] == report["arms"]
    compared = A.compare(tmp_path / "mirror", tmp_path / "mirror", target=tmp_path / "self.json")
    assert all(stats["mean"] in (0, None) for body in compared["arms"].values() for stats in body["paired"].values())


def test_world_measures_and_adoption_on_a_planted_track():
    base = {"start": 0, "population": [2, 2, 2, 2], "food": [0., 1., 2., 3.], "agents": {}, "heard": [],
            "demos": [], "signals": [[1, 5, 0, ["x"]], [2, 5, 1, ["y"]], [3, 5, 2, ["x"]]]}
    arm = {"start": 0, "population": [2, 3, 3, 3], "food": [0., 1., 3., 6.],
           "agents": {"9": {"born": 0, "parent": None, "died": 2, "reason": "starvation", "births": [1]}},
           "heard": [[1, 5, 9]], "demos": [[1, "teach", 9, 5, "eat", True]],
           "signals": [[1, 9, 1, ["x"]], [1, 9, 0, ["y"]], [2, 5, 1, ["x"]], [3, 5, 1, ["x"]], [3, 5, 0, ["y"]]]}
    measures = A.world_measures(arm, 0, 3)
    assert measures["population_mean"] == 3 and measures["agent_ticks"] == 8
    assert measures["food_energy_per_agent_tick"] == pytest.approx(6 / 8)
    mine = A.individual(arm, 9, 0, 3)
    assert mine["survival_ticks"] == 2 and mine["offspring"] == 1 and mine["teach"] == 1 and mine["calls"] == 2
    adoption = A.call_adoption(arm, base, [9], 0, 3, reps=20)
    assert adoption["visitor_calls"] == 2 and adoption["last_visitor_death"] == 2
    # while the visitor lived (ticks 1-2) the native called 1 in situation x: the visitor's symbol
    assert adoption["contact"]["arm"]["raw"] == 1.0 and adoption["contact"]["baseline"]["raw"] == 0.0
    # after its death (tick 3) the native kept symbol 1 in x and used 0 in y
    assert adoption["after"]["arm"]["raw"] == 1.0
    assert A.paired_stats([1., 2., -1., None])["n"] == 3


def test_torch_policy_answers_a_valid_action(tmp_path):
    torch = pytest.importorskip("torch")
    from haishool import student
    from haishool.model import GPT, GPTConfig
    words = ["life8", "see", "want", "thrive", "food_near", "yes", "no", *experience.ACTION_WORDS]
    vocab = student.Vocab(words)
    config = GPTConfig(vocab_size=len(vocab.itos), ctx=64, n_layer=1, d_model=32, n_head=2)
    model = GPT(config)
    torch.save({"model_state": model.state_dict(), "gpt_config": config.to_json(), "itos": vocab.itos},
               tmp_path / "student.pt")
    policy = V.TorchPolicy(tmp_path / "student.pt", device="cpu")
    prompts = ["q life8 see food_near yes want thrive. a", "q life8 see want thrive. a"]
    answers = policy.act(prompts, [["eat", "rest"], ["move_n"]])
    assert answers[0] in ("eat", "rest") and answers[1] == "move_n"
