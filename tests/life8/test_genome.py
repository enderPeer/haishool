"""Heritable behaviour, sexual reproduction and the age curve (haishool/life8/genome.py)."""
import json
import random

import pytest

from haishool.life8 import genome, learning
from haishool.life8.checkpoint import digest
from haishool.life8.config import Config
from haishool.life8.entities import Anatomy
from haishool.life8.world import World

EAT_NEAR = genome.feature_key("food_near", True)


def breeder(seed=9, **overrides):
    """One mature founder that gives birth on its first tick."""
    values = dict(population=1, food_patches=0, initial_objects=0, initial_energy=40., maturity_age=0,
                  learning=False, genome="evolving")
    values.update(overrides)
    return World.create(seed, Config(**values))


def pair(seed=4, **overrides):
    values = dict(population=2, food_patches=0, initial_objects=0, initial_energy=40., maturity_age=0,
                  learning=False, reproduction="sexual", genome="evolving", signal_noise=0.)
    values.update(overrides)
    world = World.create(seed, Config(**values))
    for agent in world.agents.values():
        agent.position = [10., 10.]
        agent.anatomy = Anatomy()
    return world


def test_frozen_default_keeps_newborns_blank_and_the_world_rng_untouched():
    assert Config().genome == "frozen" and Config().reproduction == "asexual" and not Config().age_effects
    world = breeder(genome="frozen")
    world.step({1: "rest"})
    child = world.agents[2]
    assert child.genome == {} and child.controller["weights"] == {} and child.controller["feature_order"] == []
    # With random actions the controller is never consulted: an evolving genome draws
    # only from its own RNG, so the ecological trajectory is identical to the frozen one.
    runs = {}
    for mode in ("frozen", "evolving"):
        w = World.create(3, Config(learning=False, genome=mode, food_patches=30))
        for _ in range(200):
            w.step()
        runs[mode] = [(a.id, a.parent_id, a.position, a.energy, a.anatomy.to_dict(), a.lifespan)
                      for a in w.agents.values()]
        assert w.counters["births"] >= 3
    assert runs["frozen"] == runs["evolving"]


def test_innate_biases_seed_the_newborn_controller_and_drive_its_first_choice():
    g = {"bias": {EAT_NEAR: {"eat": .8}, genome.feature_key("heard_symbol", 2): {"signal_1": .5}},
         "alpha": .4, "epsilon": 0.}
    controller = genome.seed_controller(g, max_memories=8)
    learning.validate_controller(controller)
    assert controller["alpha"] == .4 and controller["epsilon"] == 0.
    assert controller["weights"][EAT_NEAR] == {"eat": .8} and learning.BIAS_FEATURE in controller["weights"]
    assert controller["q"] == {} and controller["memory"] == [] and controller["updates"] == 0
    near = {"food_near": True, "food_direction": "e", "energy_band": "middle", "messages": []}
    assert learning.choose_action(controller, near, random.Random(1)) == "eat"
    heard = {"food_near": False, "food_direction": "none", "messages": [{"symbol": 2, "age": 0}]}
    assert learning.choose_action(controller, heard, random.Random(1)) == "signal_1"
    assert genome.innate_choice_probability(g, Config()) == 1.
    assert genome.innate_choice_probability({}, Config()) == pytest.approx(1 / len(learning.ACTIONS))


def test_seeding_uses_the_learners_constructor_and_matches_the_adapter():
    g = genome.inherit({}, Config(genome="evolving", bias_mutation_rate=.2), random.Random(5))
    assert g["bias"]
    seeded = genome.seed_controller(g, max_memories=16)
    if hasattr(learning, "controller_from_biases"):
        assert seeded == learning.controller_from_biases(g["bias"], alpha=g["alpha"], epsilon=g["epsilon"], max_memories=16)
    assert seeded == genome.adapter_controller(g, max_memories=16)
    assert genome.seed_controller({}, max_memories=16) == learning.new_controller(max_memories=16)


def test_learned_weights_are_never_inherited():
    children = []
    for trained in (False, True):
        world = breeder(seed=11)
        parent = world.agents[1]
        parent.genome = {"bias": {EAT_NEAR: {"eat": .5}}}
        if trained:  # a lifetime of learning that the child must not receive
            for _ in range(50):
                learning.learn(parent.controller, {"food_near": True}, "share", 3., {"food_near": True})
            assert parent.controller["weights"]
        world.step({1: "rest"})
        children.append(world.agents[2])
    assert children[0].genome == children[1].genome
    assert children[0].controller == children[1].controller
    assert all("share" not in row for row in children[1].controller["weights"].values())


def test_mutation_rates_and_bounds():
    cfg = Config(genome="evolving", bias_mutation_rate=0., mutation_rate=0.)
    parent = {"bias": {EAT_NEAR: {"eat": .5}}, "alpha": .3, "epsilon": .2}
    child = genome.inherit(parent, cfg, random.Random(1))
    assert child["bias"] == parent["bias"] and child["alpha"] == .3 and child["epsilon"] == .2
    cfg = Config(genome="evolving", bias_mutation_rate=1., bias_mutation_scale=1., mutation_rate=1., mutation_scale=1.)
    g = parent
    for step in range(30):
        g = genome.inherit(g, cfg, random.Random(step))
        genome.validate_genome(g, cfg)
    assert g["bias"] != parent["bias"] and set(g["bias"]) <= set(genome.INNATE_KEYS)
    assert all(abs(v) <= genome.BIAS_BOUND for row in g["bias"].values() for v in row.values())
    assert genome.ALPHA_BOUNDS[0] <= g["alpha"] <= genome.ALPHA_BOUNDS[1]
    # Life-history genes only exist with age effects on.
    assert "lifespan" not in g and "lifespan" in genome.inherit(parent, Config(age_effects=True), random.Random(1))
    with pytest.raises(ValueError):
        genome.validate_genome({"bias": {'["wealth",true]': {"eat": 1.}}}, cfg)
    with pytest.raises(ValueError):
        Config(genome="lamarckian")


def test_bias_mutation_rate_sets_how_many_entries_change():
    cfg = Config(genome="evolving", bias_mutation_rate=.05, bias_mutation_scale=.25)
    entries = len(genome.INNATE_KEYS) * len(learning.ACTIONS)
    changed = [sum(len(row) for row in genome.inherit({}, cfg, random.Random(s))["bias"].values()) for s in range(200)]
    assert abs(sum(changed) / len(changed) / entries - .05) < .01


def test_sexual_birth_needs_a_willing_partner_and_both_pay():
    world = pair()
    world.agents[2].position = [25., 20.]  # out of reach: no mate, no birth
    world.step({1: "rest", 2: "rest"})
    assert len(world.agents) == 2 and world.counters.get("births", 0) == 0
    world = pair()
    before = {i: a.energy for i, a in world.agents.items()}
    events = world.step({1: "rest", 2: "rest"})
    births = [e for e in events if e["type"] == "birth"]
    assert len(births) == 1 and births[0]["partner"] in (1, 2) and births[0]["parent"] != births[0]["partner"]
    child = world.agents[3]
    assert child.energy == pytest.approx(Config().offspring_energy, abs=.2)
    for identity in (1, 2):
        spent = before[identity] - world.agents[identity].energy
        assert spent > Config().reproduction_cost + Config().offspring_energy / 2 - 1e-9
        assert world.agents[identity].cooldown == Config().birth_cooldown
    assert world.validate()["ok"]


def test_sexual_child_recombines_its_parents_genes():
    left = {name: 1. for name in ("body_mass", "speed", "sensing", "manipulation", "metabolism")}
    right = {name: 2. for name in left}
    picks = [genome.recombine_anatomy(left, right, random.Random(s)) for s in range(50)]
    assert all(set(p.values()) <= {1., 2.} for p in picks)
    assert any(len(set(p.values())) == 2 for p in picks)
    a = {"bias": {EAT_NEAR: {"eat": 1.}}, "alpha": .1, "epsilon": .1}
    b = {"bias": {genome.feature_key("food_near", False): {"rest": 1.}}, "alpha": .9, "epsilon": .9}
    cfg = Config(bias_mutation_rate=0., mutation_rate=0.)
    kids = [genome.inherit(a, cfg, random.Random(s), partner=b) for s in range(40)]
    assert {k["alpha"] for k in kids} == {.1, .9}
    assert any(EAT_NEAR in k["bias"] for k in kids) and any(EAT_NEAR not in k["bias"] for k in kids)


def test_age_curve_penalizes_juveniles_and_the_old():
    cfg = Config(age_effects=True, max_age=600, maturity_age=35)
    world = World.create(1, Config(population=1, food_patches=0, initial_objects=0, age_effects=True))
    agent = world.agents[1]
    agent.lifespan = 600
    values = {}
    for age in (0, 17, 35, 300, 420, 510, 599):
        agent.age = age
        values[age] = genome.ability(agent, cfg)
    assert values[0] == pytest.approx(.5) and values[0] < values[17] < values[35] == 1. == values[300] == values[420]
    assert values[420] > values[510] > values[599] >= .5
    agent.genome = {"maturity": 70.}
    agent.age = 300
    assert genome.ability(agent, cfg) > 1.  # later maturity, stronger adult
    assert genome.ability(agent, Config()) == 1.  # off by default
    agent.genome = {"lifespan": 1200.}
    assert genome.upkeep(agent, cfg) == pytest.approx(2 ** .5) and genome.upkeep(agent, Config()) == 1.


def test_age_effects_change_what_a_newborn_can_eat():
    eaten = {}
    for effects in (False, True):
        world = World.create(2, Config(population=1, food_patches=2, initial_objects=0, learning=False,
                                       automatic_reproduction=False, age_effects=effects, food_regrowth=0.))
        agent, food = world.agents[1], world.foods[2]  # patch 2 is hard
        food.position, food.hardness = list(agent.position), .4
        events = world.step({1: {"type": "eat", "target": 2}})
        eaten[effects] = next(e for e in events if e["type"] == "action")["quantity"]
    assert eaten[True] < eaten[False]


def test_heritable_maturity_sets_when_an_organism_can_reproduce():
    world = breeder(maturity_age=35)
    agent = world.agents[1]
    agent.age, agent.born_tick = 20, 0
    world.tick = 20
    assert not world._fertile(agent)
    agent.genome = {"maturity": 15.}
    assert world._fertile(agent)


def test_evolving_sexual_aging_world_continues_exactly_from_a_checkpoint():
    cfg = Config(genome="evolving", reproduction="sexual", age_effects=True, width=14., height=10., food_patches=24,
                 log="compact")
    direct = World.create(5, cfg)
    for _ in range(240):
        direct.step()
    resumed = World.create(5, cfg)
    for _ in range(120):
        resumed.step()
    resumed = World.from_dict(json.loads(json.dumps(resumed.to_dict(), sort_keys=True)))
    for _ in range(120):
        resumed.step()
    assert direct.counters["births"] >= 3 and any(a.genome.get("bias") for a in direct.agents.values())
    assert any(a.parent_id is not None for a in direct.agents.values())
    assert digest(direct.to_dict()) == digest(resumed.to_dict())
    assert direct.validate()["ok"]


def test_states_without_a_genome_field_still_load():
    world = World.create(1, Config(population=2, food_patches=1, initial_objects=0))
    state = world.to_dict()
    for row in state["agents"]:
        del row["genome"]
    assert World.from_dict(state).agents[1].genome == {}


@pytest.mark.slow
def test_evolving_genome_lowers_newborn_mortality_against_frozen_and_neutral_controls():
    from haishool.life8.analysis import heritable
    # A smaller copy of docs/life8/results/heritable-genome.json (8 seeds x 4000 ticks).
    report = heritable.measure(seeds=range(1, 5), ticks=1200, jobs=12)
    totals = report["totals"]
    assert totals["evolving"]["mortality_before_50"] < totals["frozen"]["mortality_before_50"]
    assert totals["evolving"]["mortality_before_50"] < totals["neutral"]["mortality_before_50"]
    assert report["paired_seeds_evolving_lower_mortality_than_frozen"] >= 3
    evolved = totals["evolving"]["innate_p_eat_near_by_generation_bin"]
    drifted = totals["neutral"]["innate_p_eat_near_by_generation_bin"]
    assert evolved["4-7"][1] > evolved["1-3"][1] and evolved["4-7"][1] > drifted["4-7"][1]
