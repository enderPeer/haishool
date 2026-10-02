import ast
import copy
import doctest
import hashlib
import importlib
import math
import os
import random
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import pytest

import haishool.evo
import haishool.evo.bodies as B
from haishool.cosmos import Simulation
from haishool.evo import LessonGate
from haishool.evo.bodies import (COUNT_KEYS, GONE, GUILD_KEYS, GUILDS, KEYS, LEDGER_KEYS, LESSONS, MUTATIONS, OUTCOMES,
                                 PARAM_KEYS, RULES, SIM, STAGES, STATE_KEYS, STEPS, SUMMARY_KEYS, TICKS, WORD_KEYS,
                                 Bodies, _World, dense_value, describe, handoff_in, handoff_out, lesson_gate, lines,
                                 max_cell_types, needed_cell_types, outcome_of, oxygen_max_size, param_value,
                                 params_line, random_params, simulation, stage_of, summarise, supported_levels)
from haishool.truth import Gate, is_dense, num, parse_num, split_line

SEEDS = list(range(1, 41))
MANY = list(range(1, 201))
#: a rich world with grazers and predators: the reference for the controlled comparisons
BASE = dict(light=300.0, efficiency=0.12, predation=1.2, mutation=2.0, start_oxygen=0.05, complex_cells="yes")
#: the function of the module that answers each lesson
FUNCTIONS = {name: getattr(B, {"kleiber": "kleiber_rate", "upkeep": "upkeep_rate"}.get(name, name)) for name in RULES}


def n_tokens(text: str) -> int:
    """As ``haishool.student.tokens`` counts them (that module needs torch, so not imported)."""
    return len(text.replace(".", " . ").split())


def digits(n: int) -> str:
    return " ".join(str(n))


@pytest.fixture(scope="module")
def sim():
    return Bodies()


@pytest.fixture(scope="module")
def many(sim):
    """The canonical rollouts of seeds 1 to 200."""
    return {seed: sim.rollout(seed) for seed in MANY}


@pytest.fixture(scope="module")
def rollouts(many):
    return {seed: many[seed] for seed in SEEDS}


@pytest.fixture(scope="module")
def all_lines(rollouts):
    return [ln for r in rollouts.values() for ln in lines(r)]


@pytest.fixture(scope="module")
def lessons():
    return LESSONS.generate(random.Random(1), 400)


@pytest.fixture(scope="module")
def controlled(sim):
    """Twenty worlds each for the reference and for one parameter changed."""
    cache: dict[tuple, list] = {}

    def group(**changed):
        key = tuple(sorted(changed.items()))
        if key not in cache:
            cache[key] = [sim.run(seed, **dict(BASE, **changed)) for seed in range(1, 21)]
        return cache[key]

    return group


# ---------------------------------------------------------------------------------------------
# the level as the contract asks for it
# ---------------------------------------------------------------------------------------------

def test_the_examples_in_the_docstrings_hold():
    failed, tried = doctest.testmod(B)
    assert failed == 0 and tried >= 30


def test_protocols():
    s = simulation()
    assert isinstance(s, Simulation) and isinstance(s, Gate)
    assert s.sim == s.topic == SIM == "bodies" and s.records() == [] and simulation() is s
    assert "bodies" in haishool.evo.LEVELS
    for key in PARAM_KEYS + STATE_KEYS + LEDGER_KEYS + SUMMARY_KEYS:
        assert key in KEYS, key
    assert COUNT_KEYS | WORD_KEYS <= set(KEYS)
    assert isinstance(LESSONS, LessonGate) and lesson_gate() is LESSONS and isinstance(LESSONS, Gate)
    assert LESSONS.sim == "bodies" and LESSONS.topic == "predict_bodies" and len(RULES) >= 8
    assert set(LESSONS.KEYS) == set(RULES)
    assert STEPS == 80 and len(STAGES) == 6 and set(OUTCOMES) >= {"single_cells", "colonies", "multicellular",
                                                                 "animals_like"}
    for name in ("run", "rollout", "conserved", "check", "owns", "lines", "random_params", "handoff_in"):
        assert callable(getattr(B, name)), name


def test_module_level_functions_are_the_simulation(sim, rollouts):
    r = B.run(5, **BASE)
    assert r.steps == sim.run(5, **BASE).steps and B.conserved(r).ok
    assert B.rollout(3).steps == rollouts[3].steps
    assert B.owns("bodies seed 3 final outcome") and B.check("bodies seed 3 final outcome", "animals_like").ok
    assert B.owns("bodies predict kleiber mass 1 6") and B.check("bodies predict kleiber mass 1 6", "8").ok
    assert not B.check("bodies predict kleiber mass 1 6", "9").ok
    assert not B.owns("cells seed 3 final outcome") and not B.check("cells seed 3 final outcome", "cells").ok


def test_same_seed_same_lines():
    a, b = Bodies(), Bodies()
    for seed in (1, 7, 4321):
        ra, rb = a.rollout(seed), b.rollout(seed)
        assert ra.steps == rb.steps and ra.summary == rb.summary and ra.params == rb.params
        assert ra.lineages == rb.lineages and ra.events == rb.events
        assert [ln.text for ln in lines(ra)] == [ln.text for ln in lines(rb)]
    assert [ln.text for ln in lines(a.rollout(1))] != [ln.text for ln in lines(a.rollout(2))]
    assert a.run(5, **BASE).steps == b.run(5, **BASE).steps
    assert a.run(5, **BASE).steps != a.run(6, **BASE).steps
    assert a.run(5, **BASE).steps != a.run(5, **dict(BASE, light=301.0)).steps


def test_forty_seeds_replay_exactly(rollouts):
    """Two runs of a seed give the same step dicts, lineages and events, to the last bit."""
    fresh = Bodies()
    for seed, r in rollouts.items():
        again = fresh.run(seed, **random_params(random.Random(seed)))
        assert again.steps == r.steps and again.lineages == r.lineages and again.events == r.events
        assert again.summary == r.summary and again.params == r.params
        assert [repr(v) for s in again.steps for v in s.values()] == [repr(v) for s in r.steps for v in s.values()]


def test_generate_is_deterministic(sim):
    one = [ln.text for ln in sim.generate(random.Random(11), 2)]
    two = [ln.text for ln in Bodies().generate(random.Random(11), 2)]
    assert one == two and len(one) == 2 * 1131
    assert one != [ln.text for ln in sim.generate(random.Random(12), 2)]


def test_values_are_plain_python_numbers_and_words(rollouts):
    for r in rollouts.values():
        assert len(r.steps) == len(r.lineages) == STEPS
        for row in r.steps + [r.summary, r.params]:
            for key, value in row.items():
                assert type(value) in (float, int, str), (key, type(value))
                assert (type(value) is int) == (key in COUNT_KEYS), key
                assert (type(value) is str) == (key in WORD_KEYS), key
                if type(value) is float:
                    assert math.isfinite(value) and (value >= 0 or key.startswith("moved_")), (key, value)
        for step in r.steps:
            assert list(step) == list(STATE_KEYS) + list(LEDGER_KEYS)
            assert step["stage"] in STAGES
        assert list(r.summary) == list(SUMMARY_KEYS) and list(r.params) == list(PARAM_KEYS)
        for alive in r.lineages:
            for ln in alive:
                assert [type(ln[k]) for k in ("id", "parent", "guild", "size", "adhesion", "cell_types", "level",
                                              "biomass")] == [int, int, str, int, float, int, int, float]


def test_pinned_lines_of_seed_3(sim):
    """The lines the module docstring shows, word for word."""
    r = sim.rollout(3)
    texts = [ln.text for ln in lines(r)]
    assert texts[0] == ("bodies seed 3 params. light 5 1. efficiency 0 point 1 3. predation 1 point 2 9. "
                        "mutation 1 point 9 5. start_oxygen 0 point 0 0 1 4. complex_cells yes.")
    assert texts[1] == ("bodies seed 3 step 0. species 1. producers 2 point 5 5. grazers 0. predators 0. max_size 1. "
                        "mean_size 1. cell_types 1. consumer_size 0. consumer_types 0. trophic_levels 1. "
                        "oxygen 0 point 0 0 1 4. extinctions 0. stage single_cells.")
    assert ("bodies seed 3 step 4 0. species 3 1. producers 3 0 point 1. grazers 1 0 point 2. "
            "predators 0 point 2 9 7. max_size 3 2. mean_size 2 1 point 1. cell_types 3. consumer_size 3 2. "
            "consumer_types 3. trophic_levels 2. oxygen 0 point 0 4 9 2. extinctions 3 7. stage colonies.") in texts
    for text in ("q bodies seed 3 step 4 0 max_size. a 3 2.", "q bodies seed 3 step 7 9 stage. a food_web.",
                 "q bodies seed 3 final outcome. a animals_like.", "q bodies seed 3 final consumer_size. a 1 2 8.",
                 "q bodies seed 3 final predator_share. a 0 point 0 1 5 8.", "q bodies seed 3 final species. a 3 1.",
                 "q bodies seed 3 final oxygen. a 0 point 0 4 8 5."):
        assert text in texts, text
    assert r.summary == {"stage": "food_web", "species": 31, "max_size": 128, "cell_types": 6, "consumer_size": 128,
                         "consumer_types": 5, "trophic_levels": 3, "oxygen": 0.0485, "predator_share": 0.0158,
                         "outcome": "animals_like"}
    assert r.steps[40]["raw_producers"] == pytest.approx(30.109198828100126, rel=1e-12)
    assert r.steps[-1]["extinctions"] == 85 == sum(e["kind"] == "extinct" for e in r.events) and len(r.events) == 200
    assert r.events[0] == {"step": 1, "kind": "guild", "id": 1, "parent": 0, "guild": "grazer", "size": 1,
                           "cell_types": 1, "oxygen": pytest.approx(0.0016823239479535751, rel=1e-12)}


def test_random_params_stay_in_range():
    seen_no_predation = seen_simple = 0
    for seed in range(300):
        p = random_params(random.Random(seed))
        assert p == random_params(random.Random(seed)) and list(p) == list(PARAM_KEYS)
        assert 20 <= p["light"] <= 1000 and p["light"] == round(p["light"])
        assert 0.05 <= p["efficiency"] <= 0.2 and p["efficiency"] == round(p["efficiency"], 2)
        assert p["predation"] == 0 or 0.2 <= p["predation"] <= 2
        assert 0.2 <= p["mutation"] <= 3 and 0.001 <= p["start_oxygen"] <= 0.2
        assert p["complex_cells"] in ("yes", "no")
        assert parse_num(param_value("start_oxygen", p["start_oxygen"])) == p["start_oxygen"]
        seen_no_predation += p["predation"] == 0
        seen_simple += p["complex_cells"] == "no"
    assert 20 <= seen_no_predation <= 70 and 60 <= seen_simple <= 120
    assert random_params(random.Random(3)) == {"light": 51.0, "efficiency": 0.13, "predation": 1.29,
                                               "mutation": 1.95, "start_oxygen": 0.0014, "complex_cells": "yes"}
    assert random_params(random.Random(3), rules=7) == random_params(random.Random(3))
    with pytest.raises(ValueError):
        random_params(random.Random(3), rules=6)


def test_run_rejects_impossible_parameters(sim):
    for bad in ({"light": 0.5}, {"light": float("nan")}, {"efficiency": 0.0}, {"efficiency": 0.6},
                {"predation": -1.0}, {"mutation": -0.1}, {"mutation": TICKS + 1}, {"start_oxygen": -0.01},
                {"start_oxygen": float("inf")}, {"complex_cells": "maybe"}, {"rules": 6}):
        with pytest.raises(ValueError):
            sim.run(1, **bad)
    still = sim.run(1, mutation=0.0)
    assert sim.conserved(still).ok and still.summary["species"] == 1 and still.summary["outcome"] == "single_cells"
    assert not still.events and all(s["max_size"] == 1 for s in still.steps)
    tiny = sim.run(1, light=1.0)
    assert sim.conserved(tiny).ok and tiny.steps[0]["raw_producers"] == 0.05
    assert sim.run(1, complex_cells=True).steps == sim.run(1, complex_cells="yes").steps
    assert sim.run(1, complex_cells=False).params["complex_cells"] == "no"


def test_a_run_is_fast(sim):
    worst = 0.0
    for seed in range(100, 120):
        start = time.perf_counter()
        sim.run(seed, **random_params(random.Random(seed)))
        worst = max(worst, time.perf_counter() - start)
    start = time.perf_counter()
    crowded = sim.run(1, **dict(BASE, light=1000.0, efficiency=0.2, mutation=20.0))
    assert max(s["species"] for s in crowded.steps) == 60, "the busiest world: 60 lineages"
    assert max(worst, time.perf_counter() - start) < 2.0


# ---------------------------------------------------------------------------------------------
# lines
# ---------------------------------------------------------------------------------------------

def test_every_line_is_dense_and_short(all_lines):
    assert len(all_lines) == 40 * 1131
    for ln in all_lines:
        assert is_dense(ln.text), ln.text
        assert n_tokens(ln.text) <= 128, ln.text
        assert ln.topic == "bodies"
        if ln.kind != "record":
            assert "." not in ln.prompt and "." not in ln.answer
            assert split_line(ln.text) == (ln.prompt, ln.answer)
            assert ln.kind in ("fact", "calc")
    assert max(n_tokens(ln.text) for ln in all_lines) <= 80


def test_lines_of_a_long_seed_stay_short(sim):
    for seed in (9998, 123456789):
        for ln in lines(sim.rollout(seed)):
            assert is_dense(ln.text) and n_tokens(ln.text) <= 128, ln.text
            if ln.kind != "record":
                assert sim.check(ln.prompt, ln.answer).ok, ln.text


def test_line_kinds(rollouts):
    r = rollouts[3]
    ls = lines(r)
    records = [ln for ln in ls if ln.kind == "record"]
    assert records[0].text == params_line(r).text and records[0].text.startswith("bodies seed 3 params. light ")
    assert len(records) == 1 + STEPS
    prompts = {ln.prompt for ln in ls if ln.kind != "record"}
    assert len(prompts) == STEPS * len(STATE_KEYS) + len(SUMMARY_KEYS)
    assert len(ls) == 1131 == len({ln.text for ln in ls})
    assert "bodies seed 3 step 1 4 oxygen" in prompts and "bodies seed 3 final outcome" in prompts
    assert not any(" next " in p for p in prompts)
    assert not any(k in ln.text for ln in ls for k in LEDGER_KEYS), "the ledger is not trained"
    assert len(lines(r, every=10)) == 1 + 8 * 14 + 10


def test_numbers_use_three_significant_digits(all_lines):
    assert dense_value(1052.53) == "1 0 5 0" and dense_value(0.03234) == "0 point 0 3 2 3"
    assert dense_value(7) == "7" and dense_value(0.0) == "0" and dense_value("colonies") == "colonies"
    assert dense_value(0.000412) == "4 point 1 2 e minus 4" and dense_value(999.7) == "1 0 0 0"
    assert param_value("light", 250.0) == "2 5 0" and param_value("complex_cells", "no") == "no"
    for ln in all_lines:
        if ln.kind == "record" or ln.prompt.split()[-1] in WORD_KEYS:
            continue
        value = parse_num(ln.answer)
        assert value is not None and value >= 0, ln.text
        shown = "".join(w for w in ln.answer.split(" e ")[0].split() if w.isdigit())
        if ln.prompt.split()[-1] not in COUNT_KEYS:
            assert len(shown.strip("0")) <= 3, ln.text
        assert len(ln.answer.split()) <= 8, ln.text


# ---------------------------------------------------------------------------------------------
# check and owns
# ---------------------------------------------------------------------------------------------

def test_gate_agrees_with_every_line(sim, all_lines):
    for ln in all_lines:
        if ln.kind == "record":
            continue
        assert sim.owns(ln.prompt), ln.prompt
        v = sim.check(ln.prompt, ln.answer)
        assert v.ok and v.expected == ln.answer, (ln.text, v)


def test_gate_agrees_with_every_record_field(sim, all_lines):
    for ln in all_lines:
        if ln.kind != "record":
            continue
        head, *fields = [f.strip() for f in ln.text.rstrip(".").split(".")]
        for f in fields:
            key, _, value = f.partition(" ")
            v = sim.check(f"{head} {key}", value)
            assert v.ok and v.expected == value, (ln.text, key)


def test_gate_rejects_wrong_answers(sim, all_lines):
    rng = random.Random(5)
    questions = [ln for ln in all_lines if ln.kind != "record"]
    for ln in rng.sample(questions, 4000):
        key = ln.prompt.split()[-1]
        if key in WORD_KEYS:
            wrong = next(s for s in STAGES + OUTCOMES if s != ln.answer)
        else:
            wrong = "9 " + ln.answer  # a changed leading digit: far outside the 5 % tolerance
        v = sim.check(ln.prompt, wrong)
        assert not v.ok and v.expected == ln.answer, (ln.text, wrong, v)
    assert not sim.check("bodies seed 3 final species", "many").ok
    assert sim.check("bodies seed 3 final species", "many").expected == "3 1"
    assert not sim.check("bodies seed 3 step 7 9 stage", "food").ok
    assert not sim.check("bodies seed 3 params complex_cells", "no").ok
    assert sim.check("bodies seed 3 params complex_cells", "yes").ok


def test_gate_rejects_a_changed_digit(sim, all_lines):
    """One digit of the answer changed: a count is then wrong; a measured number is wrong when
    the change is more than 5 %."""
    rng = random.Random(6)
    numeric = [ln for ln in all_lines if ln.kind != "record" and ln.prompt.split()[-1] not in WORD_KEYS]
    rejected = 0
    for ln in rng.sample(numeric, 4000):
        words = ln.answer.split()
        place = rng.choice([i for i, w in enumerate(words) if w.isdigit()])
        words[place] = str((int(words[place]) + rng.randint(1, 9)) % 10)
        wrong = " ".join(words)
        truth, got = parse_num(ln.answer), parse_num(wrong)
        assert got != truth
        v = sim.check(ln.prompt, wrong)
        assert v.expected == ln.answer
        off = abs(got - truth) / max(abs(got), abs(truth))
        if ln.prompt.split()[-1] in COUNT_KEYS or off > 0.06:
            assert not v.ok, (ln.text, wrong, v)
        elif off < 0.04:
            assert v.ok, (ln.text, wrong, v)
        rejected += not v.ok
    assert rejected > 3000


def test_gate_rejects_answers_that_are_not_finite_numbers(sim):
    for prompt in ("bodies seed 3 final oxygen", "bodies seed 3 step 4 0 producers", "bodies seed 3 final species",
                   "bodies seed 3 params light"):
        for answer in ("1 e 9 9 9", "minus 1 e 9 9 9", "", "point", "e", "yes", "3 0 point 1 units", "3 0 . 1",
                       "minus 3 0 point 1", "1 e 3 0 0"):
            v = sim.check(prompt, answer)
            assert not v.ok and v.expected is not None, (prompt, answer, v)
    assert sim.check("bodies seed 3 final oxygen", "1 e 9 9 9").reason == "not a number"
    assert sim.check("bodies seed 3 step 0 grazers", "0").ok and not sim.check("bodies seed 3 step 0 grazers", "1").ok


def test_tolerance_is_five_percent_for_numbers_and_exact_for_counts_and_parameters(sim, rollouts):
    r = rollouts[3]
    value = r.steps[40]["producers"]
    assert value == 30.1
    for factor, ok in ((1.03, True), (0.97, True), (1.1, False), (0.9, False)):
        assert sim.check("bodies seed 3 step 4 0 producers", dense_value(value * factor)).ok is ok
    assert sim.check("bodies seed 3 final species", "3 1").ok
    assert not sim.check("bodies seed 3 final species", "3 2").ok and not sim.check("bodies seed 3 final species", "3 0").ok
    assert not sim.check("bodies seed 3 final max_size", "1 2 9").ok
    assert sim.check("bodies seed 3 params light", "5 1").ok and not sim.check("bodies seed 3 params light", "5 2").ok
    assert sim.check("bodies seed 3 params efficiency", "0 point 1 3").ok
    assert not sim.check("bodies seed 3 params efficiency", "0 point 1 3 1").ok
    assert sim.check("bodies seed 3 params efficiency", "0 point 1 4").expected == "0 point 1 3"


@pytest.mark.parametrize("prompt", [
    "turkey capital", "q spoon color", "carbon protons", "water molar_mass", "calc 1 2 plus 7",
    "gravity seed 7 step 2 0 clumps", "nucleo seed 3 final helium_fraction", "planets seed 7 final gas",
    "chem seed 7 final water", "life seed 7 final stage", "world seed 3 era planets",
    "stars seed 7 final metallicity", "cells seed 7 step 3 cells", "cells seed 7 final outcome",
    "senses seed 7 final outcome", "senses seed 7 step 3 eyes", "signals seed 7 step 1 2 success",
    "society seed 7 final outcome", "world7 seed 7 final outcome",
    "bodies", "bodies seed 7", "bodies seed 7 step 3", "bodies seed 7 step 3 clumps", "bodies seed 7 step 8 0 species",
    "bodies seed 7 step 3 next species", "bodies seed 7 final extinctions", "bodies seed 7 final mean_size",
    "bodies seed x final stage", "bodies seed 7 final", "bodies seed 7 params stage", "bodies seed 7 params oxygen",
    "bodies seed 12 step 3 species", "bodies seed 7 step 3 species stage", "", "bodies seed 0 7 final stage",
    "bodies seed 7 step 0 3 species", "bodies seed 7 step 0 0 species", "bodies seed minus 7 final stage",
    "bodies seed 7 point 5 final stage", "q bodies seed 7 final stage", "bodies seed 7 final stage extra",
    "bodies  seed 7 final stage", "bodies seed 7 final stage ", "bodies seed 7 step 3 raw_producers",
    "bodies seed 7 step 3 income_producers", "bodies seed 7 step 3 max_take", "bodies seed 7 era bodies",
    "bodies predict kleiber mass 1 6", "bodies seed 7 predict kleiber", "body seed 7 final stage",
])
def test_the_simulation_does_not_own_other_prompts(sim, prompt):
    assert not sim.owns(prompt)
    v = sim.check(prompt, "1")
    assert v.ok is False and v.expected is None and v.reason == "not my question"


@pytest.mark.parametrize("prompt", [
    "bodies seed 7 step 0 species", "bodies seed 7 step 7 9 stage", "bodies seed 1 2 3 4 step 1 2 mean_size",
    "bodies seed 7 final predator_share", "bodies seed 7 final outcome", "bodies seed 7 params start_oxygen",
    "bodies seed 7 params complex_cells", "bodies seed 0 final stage", "bodies seed 7 step 1 0 consumer_size",
])
def test_owns_its_prompts(sim, prompt):
    assert sim.owns(prompt) and B.owns(prompt)
    assert sim.check(prompt, "nonsense words").expected is not None


def test_module_level_owns_rejects_other_levels():
    for prompt in ("cells seed 7 final outcome", "senses seed 7 final outcome", "stars seed 7 final metallicity",
                   "planets seed 7 final gas", "cells predict copy_success fidelity 0 point 9 9 length 1 0 0",
                   "senses predict kin_gain relatedness 0 point 5", "bodies predict nothing mass 1",
                   "bodies predict kleiber cells 1 6", "bodies predict kleiber mass 0", "bodies predict kleiber mass 1 point 5",
                   "bodies predict supply cells 2 cell_types 9", "bodies predict kleiber mass 1 6 extra 1"):
        assert not B.owns(prompt), prompt
        assert not B.check(prompt, "1").ok


# ---------------------------------------------------------------------------------------------
# the gate: energy, biomass, feeding, oxygen
# ---------------------------------------------------------------------------------------------

def test_conserved_passes_on_forty_seeds_and_on_hand_set_worlds(sim, rollouts):
    for seed, r in rollouts.items():
        v = sim.conserved(r)
        assert v.ok, (seed, v)
    for seed in range(41, 61):  # parameters of another seed: any pairing must hold
        r = sim.run(seed, **random_params(random.Random(seed + 1000)))
        assert sim.conserved(r).ok, seed
    for changed in ({"predation": 0.0}, {"complex_cells": "no"}, {"light": 1.0}, {"light": 5000.0, "mutation": 6.0},
                    {"efficiency": 0.5, "predation": 5.0}, {"start_oxygen": 0.0}, {"start_oxygen": 3.0},
                    {"mutation": 20.0, "light": 1000.0, "efficiency": 0.2}, {"efficiency": 0.01}):
        r = sim.run(9, **dict(BASE, **changed))
        v = sim.conserved(r)
        assert v.ok, (changed, v)


def test_energy_is_accounted_for_exactly(many):
    """For every guild and step: income = respired + growth + taken + lost + moved. For the
    producers the income is the production: production = respiration + growth + transferred +
    lost to decomposers (+ the founders that became grazers)."""
    worst = 0.0
    for r in many.values():
        prev = None
        for s in r.steps:
            for g in GUILD_KEYS:
                growth = s["raw_" + g] - prev["raw_" + g] if prev else 0.0
                spent = s["respired_" + g] + growth + s["taken_" + g] + s["lost_" + g] + s["moved_" + g]
                worst = max(worst, abs(s["income_" + g] - spent))
            growth = sum(s["raw_" + g] - prev["raw_" + g] for g in GUILD_KEYS) if prev else 0.0
            whole = sum(s["respired_" + g] + s["lost_" + g] for g in GUILD_KEYS) + growth
            assert s["income_producers"] == pytest.approx(whole, abs=1e-9 * r.params["light"])
            assert s["taken_producers"] == pytest.approx(s["income_grazers"], abs=1e-9 * r.params["light"])
            assert s["taken_grazers"] + s["taken_predators"] == pytest.approx(s["income_predators"], abs=1e-9)
            assert sum(s["moved_" + g] for g in GUILD_KEYS) == pytest.approx(0.0, abs=1e-9)
            assert s["moved_producers"] >= 0 and s["moved_predators"] <= 0, "founders only move up"
            prev = s
        assert r.steps[0]["raw_producers"] == pytest.approx(0.05 * r.params["light"])
        assert all(r.steps[0][k] == 0 for k in LEDGER_KEYS if not k.startswith("raw_"))
    assert worst < 1e-10


def test_biomass_is_never_negative_and_no_consumer_takes_more_than_there_is(many):
    for r in many.values():
        eff = r.params["efficiency"]
        for s, alive in zip(r.steps, r.lineages):
            assert 0.0 <= s["max_take"] < 1.0
            assert all(ln["biomass"] >= B.EXTINCT for ln in alive)
            for g in GUILD_KEYS:
                assert min(s["raw_" + g], s["income_" + g], s["respired_" + g], s["taken_" + g], s["lost_" + g]) >= 0
            for g, guild in zip(GUILD_KEYS, GUILDS):
                assert s["raw_" + g] == pytest.approx(sum(ln["biomass"] for ln in alive if ln["guild"] == guild))
            for g in GUILD_KEYS[1:]:  # consumers keep at most the efficiency of their food
                assert s["lost_" + g] >= (1 - eff) * s["income_" + g] - 1e-9
        # the energy pyramid in flows: a guild passes on no more than it kept of its food and founders brought
        for g in GUILD_KEYS[1:]:
            passed = sum(s["taken_" + g] for s in r.steps)
            reached = sum(eff * s["income_" + g] - s["moved_" + g] for s in r.steps)
            assert passed <= reached + 1e-9
        made = sum(s["income_producers"] for s in r.steps) + r.steps[0]["raw_producers"]
        assert sum(s["income_grazers"] for s in r.steps) <= made + 1e-9


def test_oxygen_change_is_production_minus_consumption(many):
    for r in many.values():
        assert r.steps[0]["raw_oxygen"] == r.params["start_oxygen"]
        for prev, s in zip(r.steps, r.steps[1:]):
            assert s["raw_oxygen"] - prev["raw_oxygen"] == pytest.approx(s["o2_produced"] - s["o2_consumed"], abs=1e-12)
            assert s["o2_produced"] == pytest.approx(B.O2_YIELD * s["income_producers"], rel=1e-9, abs=1e-15)
            assert s["raw_oxygen"] >= 0 and s["o2_consumed"] >= 0
            assert s["oxygen"] == haishool.evo.sig(s["raw_oxygen"])


def test_the_metrics_follow_from_the_lineages(many):
    for r in many.values():
        for s, alive in zip(r.steps, r.lineages):
            told = describe(alive, r.params["light"], s["raw_oxygen"], s["extinctions"])
            assert {k: s[k] for k in told} == told
            assert s["species"] == len(alive) <= 60
            common = [ln for ln in alive if ln["biomass"] >= 0.5]
            assert s["max_size"] in {ln["size"] for ln in alive} and s["cell_types"] in {ln["cell_types"] for ln in alive}
            assert s["consumer_size"] <= s["max_size"] and s["consumer_types"] <= s["cell_types"]
            assert (s["consumer_size"] == 0) == (s["consumer_types"] == 0)
            if s["consumer_size"]:
                assert any(ln["guild"] != "producer" and ln["size"] == s["consumer_size"] for ln in common)
            assert 1 <= s["mean_size"] <= max(ln["size"] for ln in alive)
            assert s["stage"] == stage_of(s["raw_producers"], r.params["light"], s["max_size"], s["cell_types"],
                                          s["trophic_levels"])
            assert len({(ln["guild"], ln["size"], ln["cell_types"]) for ln in alive}) == len(alive)
            assert [ln["id"] for ln in alive] == sorted(ln["id"] for ln in alive)
            for ln in alive:
                assert ln["adhesion"] == 1 - 1 / ln["size"] and ln["size"] & (ln["size"] - 1) == 0
                assert needed_cell_types(ln["size"]) <= ln["cell_types"] <= max_cell_types(ln["size"])
        assert [s["extinctions"] for s in r.steps] == sorted(s["extinctions"] for s in r.steps)
        assert r.steps[-1]["extinctions"] == sum(e["kind"] in GONE for e in r.events)
        assert r.summary == summarise(r.steps)
        last = r.steps[-1]
        assert all(r.summary[k] == last[k] for k in SUMMARY_KEYS if k in last)
        assert r.summary["outcome"] == outcome_of(last["max_size"], last["cell_types"], last["trophic_levels"],
                                                  last["consumer_size"], last["consumer_types"])
        assert 0 <= r.summary["predator_share"] < 1


def test_the_event_log_fits_the_lineages(rollouts):
    for r in rollouts.values():
        born = {0}
        for e in r.events:
            assert 1 <= e["step"] < STEPS
            if e["kind"] in GONE:
                assert set(e) == {"step", "kind", "id"} and e["id"] in born
                assert all(ln["id"] != e["id"] for alive in r.lineages[e["step"]:] for ln in alive)
            else:
                assert e["kind"] in MUTATIONS and e["parent"] in born and e["id"] not in born
                assert set(e) == {"step", "kind", "id", "parent", "guild", "size", "cell_types", "oxygen"}
                born.add(e["id"])
        assert [e["step"] for e in r.events] == sorted(e["step"] for e in r.events)
        assert {ln["id"] for alive in r.lineages for ln in alive} <= born


def test_conserved_catches_tampering(sim, rollouts):
    base = rollouts[3]

    def broken(change) -> bool:
        r = copy.deepcopy(base)
        change(r)
        return not sim.conserved(r).ok

    assert not broken(lambda r: None)
    step = base.steps[40]
    assert broken(lambda r: r.steps[40].update(income_producers=step["income_producers"] + 1e-5))
    assert broken(lambda r: r.steps[40].update(respired_grazers=step["respired_grazers"] * 0.99))
    assert broken(lambda r: r.steps[40].update(lost_grazers=step["lost_grazers"] * 0.5))
    assert broken(lambda r: r.steps[40].update(taken_producers=step["taken_producers"] + 0.001))
    assert broken(lambda r: r.steps[40].update(raw_producers=step["raw_producers"] + 0.001))
    assert broken(lambda r: r.steps[40].update(moved_producers=step["moved_producers"] + 0.01))
    assert broken(lambda r: r.steps[40].update(max_take=1.5))
    assert broken(lambda r: r.steps[40].update(o2_produced=step["o2_produced"] * 2))
    assert broken(lambda r: r.steps[40].update(raw_oxygen=step["raw_oxygen"] + 1e-6))
    assert broken(lambda r: r.steps[40].update(o2_consumed=step["o2_consumed"] + 1e-6))
    assert broken(lambda r: r.steps[40].update(producers=step["producers"] + 0.1))
    assert broken(lambda r: r.steps[40].update(species=step["species"] + 1))
    assert broken(lambda r: r.steps[40].update(max_size=64))
    assert broken(lambda r: r.steps[40].update(trophic_levels=4))
    assert broken(lambda r: r.steps[40].update(stage="food_web"))
    assert broken(lambda r: r.steps[40].update(extinctions=step["extinctions"] + 1))
    assert broken(lambda r: r.steps[79].update(oxygen=0.05))
    assert broken(lambda r: r.summary.update(outcome="multicellular"))
    assert broken(lambda r: r.summary.update(predator_share=0.5))
    assert broken(lambda r: r.lineages.pop())
    assert broken(lambda r: r.lineages[40].pop())
    assert broken(lambda r: r.lineages[40][0].update(biomass=r.lineages[40][0]["biomass"] * 0.9))
    assert broken(lambda r: r.lineages[40][0].update(biomass=-1.0))
    assert broken(lambda r: r.lineages[40][0].update(size=3))
    assert broken(lambda r: r.lineages[40][0].update(cell_types=40))
    assert broken(lambda r: r.lineages[40][0].update(level=4))
    assert broken(lambda r: r.events.clear())
    assert broken(lambda r: r.params.update(complex_cells="no")), "bodies of seed 3 need complex cells"
    assert broken(lambda r: r.params.update(efficiency=0.05)), "the grazers kept 13 percent of their food, not 5"
    founded = next(n for n, e in enumerate(base.events) if e["kind"] in MUTATIONS and e["guild"] != "producer")
    assert broken(lambda r: r.events[founded].update(size=1 << 20)), "a consumer too big for the oxygen"
    predator = next(n for n, e in enumerate(base.events) if e["kind"] == "guild" and e["guild"] == "predator")
    assert broken(lambda r: r.events[predator].update(oxygen=0.001)), "a predator founded without oxygen"
    assert broken(lambda r: r.events[0].update(kind="jump"))
    assert broken(lambda r: r.params.update(predation=0.0)), "guild switches need predation"


def test_the_dead_world_is_named(sim):
    """Nothing alive cannot happen by the rules; the words for it are defined all the same."""
    told = describe([], 100.0, 0.01, 7)
    assert told["species"] == told["max_size"] == told["cell_types"] == told["trophic_levels"] == 0
    assert told["stage"] == "collapse" and told["mean_size"] == 0.0
    assert outcome_of(0, 0, 0) == "collapse" and "collapse" in OUTCOMES
    assert handoff_out({"max_size": 0, "cell_types": 0, "consumer_size": 0, "consumer_types": 0, "trophic_levels": 0,
                        "predator_share": 0.0, "oxygen": 0.01, "species": 0})["predators"] == "no"


# ---------------------------------------------------------------------------------------------
# the rules are the lessons
# ---------------------------------------------------------------------------------------------

def test_exact_arithmetic_without_the_c_library():
    rng = random.Random(2)
    for _ in range(20000):
        x = math.exp(rng.uniform(math.log(1e-12), math.log(700)))
        assert B.fraction_eaten(x) == pytest.approx(-math.expm1(-x), rel=1e-14)
    assert B.fraction_eaten(0.0) == 0.0 == B.fraction_eaten(-1.0) and B.fraction_eaten(1e6) == 1.0
    for n in list(range(1, 2000)) + [1 << k for k in range(21)] + [rng.randint(1, 1 << 20) for _ in range(5000)]:
        assert B._cbrt(n) == pytest.approx(math.cbrt(n), rel=1e-12)
        assert B.kleiber_rate(n) == pytest.approx(n ** 0.75, rel=1e-14)
    assert [B._cbrt(k ** 3) for k in (1, 2, 3, 10, 64)] == [1.0, 2.0, 3.0, 10.0, 64.0]
    assert B.energy_per_cell(16) == 0.5 and B.kleiber_rate(16) == 8.0 and B.surface_to_volume(8) == 1.5


def test_rule_values():
    assert [max_cell_types(n) for n in (1, 2, 3, 4, 8, 64, 1024, 1 << 20)] == [1, 2, 3, 4, 5, 10, 16, 30]
    for n in range(1, 5000):
        assert max_cell_types(n) == min(30, 1 + math.floor(1.5 * math.log2(n) + 1e-12))
    assert [needed_cell_types(n) for n in (1, 32, 33, 64, 1 << 20)] == [1, 1, 2, 2, 2]
    assert [oxygen_max_size(x) for x in (0.0, 0.001, 0.01, 0.02, 0.1, 0.25, 1.0, 4.0)] == [1, 1, 16, 32, 512, 2048,
                                                                                         16384, 131072]
    assert all(oxygen_max_size(a / 1000) <= oxygen_max_size((a + 1) / 1000) for a in range(0, 3000))
    assert B.supply(1, 1) == 1.0 and B.supply(8, 1) == 0.5 and B.supply(8, 2) == 1.0 and B.supply(1000, 5) == 0.5
    assert B.upkeep_rate(1, 1) == 0.05 and B.growth_rate(1, 1) == 0.6 and B.attack_rate(1.0, 1, 1) == 0.05
    assert B.handling_time(1, 1) == 0.25 and B.handling_time(4096, 16) == 2.0
    assert B.vulnerability(1, 1) == 0.5 and B.vulnerability(1, 3) == 0.75 and B.vulnerability(3, 1) == 0.25
    assert B.production(10.0, 0.5, 50.0, 100.0) == 2.5 and B.production(10.0, 0.5, 120.0, 100.0) == 0.0
    assert B.logistic_step(50.0, 0.5, 100.0) == 62.5 and B.logistic_step(150.0, 0.5, 100.0) == 150.0
    assert B.reachable(0.25) == 0.0 and B.reachable(1.0) == 0.5 and B.reachable(0.0) == 0.0
    assert B.predation_pressure(0.05, 10.0, 0.25, 80.0) == 0.25
    assert B.pyramid_flow(1000.0, 0.5, 3) == 250.0 and B.pyramid_flow(80.0, 0.1, 1) == 80.0
    assert B.predator_activity(0.005) == 0.25 and B.predator_activity(0.3) == 1.0
    assert B.adhesion(1) == 0.0 and B.adhesion(64) == 0.984375
    # a larger body needs less energy per cell, and no body is cheaper to run than a single cell is fast
    assert all(B.energy_per_cell(1 << k) > B.energy_per_cell(1 << (k + 1)) for k in range(20))
    for k in range(21):
        for types in range(needed_cell_types(1 << k), max_cell_types(1 << k) + 1):
            assert B.growth_rate(1 << k, types) / B.upkeep_rate(1 << k, types) <= 12.0 + 1e-9
    # the pyramid: levels by light and efficiency
    assert [supported_levels(light, eff) for light, eff in ((20, 0.05), (100, 0.05), (100, 0.1), (200, 0.1),
                                                           (1000, 0.05), (25, 0.2), (1, 0.05))] == [2, 3, 3, 4, 3, 4, 1]
    for light in (20, 50, 200, 1000):
        levels = [supported_levels(light, e / 100) for e in range(5, 21)]
        assert levels == sorted(levels)
    for eff in (0.05, 0.1, 0.2):
        levels = [supported_levels(light, eff) for light in range(20, 1001, 20)]
        assert levels == sorted(levels)


def test_lessons_are_accepted_and_equal_the_simulations_functions(lessons):
    assert len(lessons) == 400 and len({ln.prompt for ln in lessons}) >= 370
    seen = Counter()
    for ln in lessons:
        assert ln.topic == "predict_bodies" and ln.kind == "calc"
        assert is_dense(ln.text) and n_tokens(ln.text) <= 128 and "." not in ln.prompt and "." not in ln.answer
        v = LESSONS.check(ln.prompt, ln.answer)
        assert v.ok and v.expected == ln.answer, (ln.text, v)
        assert B.check(ln.prompt, ln.answer).ok and B.owns(ln.prompt) and not simulation().owns(ln.prompt)
        name, values = LESSONS.parse(ln.prompt)
        truth = FUNCTIONS[name](*values)
        assert ln.answer == (truth if isinstance(truth, str) else num(truth)), ln.text
        assert ln.meta["split_key"] == LESSONS.split_key(ln.prompt)
        seen[name] += 1
    assert set(seen) == set(RULES), "400 lessons show every rule"
    assert max(n_tokens(ln.text) for ln in lessons) <= 60


def test_the_lesson_rules_are_the_functions_the_simulation_calls(sim, monkeypatch):
    assert set(FUNCTIONS) == set(RULES) and len(RULES) == 23
    for name, rule in RULES.items():
        assert rule.compute is FUNCTIONS[name], name
    calls = Counter()

    def counted(name, fn):
        def wrapper(*args):
            calls[name] += 1
            return fn(*args)
        return wrapper

    indirect = {"logistic_step", "prey_step", "grazer_step"}  # one tick of the simulation, see the next tests
    for name, fn in FUNCTIONS.items():
        monkeypatch.setattr(B, fn.__name__, counted(name, fn))
    r = sim.run(3, **random_params(random.Random(3)))
    assert sim.conserved(r).ok
    for name in RULES:
        assert (calls[name] > 0) != (name in indirect), name
    assert calls["fraction_eaten"] > 10000 and calls["production"] > 10000


def reference_tick(community, light, efficiency, predation, oxygen, levels):
    """One tick of upkeep, feeding and growth, written out with the rule functions alone."""
    n = len(community)
    b = [c["biomass"] * (1 - B.upkeep_rate(c["size"], c["types"])) for c in community]
    totals = [sum(x for x, c in zip(b, community) if c["guild"] == g) for g in range(3)]
    within = [x * B.reachable(totals[c["guild"]]) for x, c in zip(b, community)]
    q, pressure, links = [0.0] * n, [0.0] * n, {}
    for j, hunter in enumerate(community):
        if hunter["guild"] == 0:
            continue
        links[j] = [(i, B.vulnerability(prey["size"], hunter["size"])) for i, prey in enumerate(community)
                    if prey["guild"] == hunter["guild"] - 1
                    or (hunter["guild"] == 2 == prey["guild"] and levels >= 4 and prey["size"] * 8 <= hunter["size"])]
        if not links[j]:
            continue
        food = sum(v * within[i] for i, v in links[j])
        attack = B.attack_rate(predation, hunter["size"], hunter["types"])
        if hunter["guild"] == 2:
            attack *= B.predator_activity(oxygen)
        q[j] = B.predation_pressure(attack, b[j], B.handling_time(hunter["size"], hunter["types"]), food)
        for i, v in links[j]:
            pressure[i] += q[j] * v
    eaten = [B.fraction_eaten(p) * w for p, w in zip(pressure, within)]
    gain = [sum(eaten[i] * q[j] * v / pressure[i] for i, v in links.get(j, []) if pressure[i] > 0) for j in range(n)]
    b = [x - e + efficiency * g for x, e, g in zip(b, eaten, gain)]
    total = sum(x for x, c in zip(b, community) if c["guild"] == 0)
    return [x + (B.production(x, B.growth_rate(c["size"], c["types"]), total, light) if c["guild"] == 0 else 0.0)
            for x, c in zip(b, community)]


def test_a_tick_of_the_simulation_is_the_rules_and_nothing_else():
    rng = random.Random(4)
    for trial in range(200):
        light, efficiency = rng.uniform(20, 1000), rng.uniform(0.05, 0.2)
        predation, oxygen = rng.uniform(0, 2), rng.choice([0.001, 0.01, 0.05, 0.5])
        world = _World(light, efficiency, predation, oxygen, True)
        community = []
        for _ in range(rng.randint(1, 12)):
            k = rng.randint(0, 12)
            types = rng.randint(needed_cell_types(1 << k), max_cell_types(1 << k))
            guild = rng.choice([0, 0, 1, 1, 2])
            if any((c["guild"], c["size"], c["types"]) == (guild, 1 << k, types) for c in community):
                continue
            community.append({"guild": guild, "size": 1 << k, "types": types, "biomass": rng.uniform(0.02, 0.3 * light)})
            world.add(guild, k, types, community[-1]["biomass"])
        before = sum(world.b)
        world.ecology()
        expected = reference_tick(community, light, efficiency, predation, oxygen, world.levels)
        assert world.b == pytest.approx(expected, rel=1e-12, abs=1e-15), trial
        assert all(x >= 0 for x in world.b)
        led = world.ledger
        made, respired = led["income_producers"], sum(led["respired_" + g] for g in GUILD_KEYS)
        lost = sum(led["lost_" + g] for g in GUILD_KEYS)
        assert made == pytest.approx(respired + (sum(world.b) - before) + lost, rel=1e-12, abs=1e-12)
        assert led["max_take"] < 1.0


def test_the_two_species_lessons_are_one_tick_of_the_simulation():
    rng = random.Random(8)
    for _ in range(300):
        p, g = round(rng.uniform(1, 1000), 1), round(rng.uniform(0.1, 200), 1)
        light, eff, pred = rng.randint(20, 1000), round(rng.uniform(0.05, 0.2), 2), round(rng.uniform(0, 2), 2)
        world = _World(float(light), eff, pred, 0.5, True)
        world.add(0, 0, 1, p)
        world.add(1, 0, 1, g)
        world.ecology()
        assert [B.prey_step(p, g, light, eff, pred), B.grazer_step(p, g, light, eff, pred)] == world.b
        # and written out as the lesson's own description says
        p1, g1 = p * 0.95, g * 0.95
        reach = max(0.0, p1 - 0.5)
        attack = 0.05 * pred
        eaten = reach * (1 - math.exp(-0.5 * attack * g1 / (1 + attack * 0.125 * reach)))
        rest = p1 - eaten
        assert B.prey_step(p, g, light, eff, pred) == pytest.approx(rest + 0.6 * rest * max(0.0, 1 - rest / light), rel=1e-9)
        assert B.grazer_step(p, g, light, eff, pred) == pytest.approx(g1 + eff * eaten, rel=1e-9)
        # a lone producer lineage: upkeep, then the logistic step
        alone = _World(float(light), eff, pred, 0.5, True)
        alone.add(0, 0, 1, p)
        alone.ecology()
        assert alone.b == [B.logistic_step(p - B.upkeep_rate(1, 1) * p, B.growth_rate(1, 1), float(light))]
    assert B.prey_step(100.0, 10.0, 200, 0.1, 0.0) == pytest.approx(95.0 + 0.6 * 95.0 * (1 - 95.0 / 200))
    assert B.grazer_step(100.0, 10.0, 200, 0.1, 0.0) == pytest.approx(9.5), "no predation: grazers only starve"


def test_lesson_gate_judges_and_draws(lessons):
    assert LESSONS.check("bodies predict kleiber mass 1 6", "8").ok
    assert LESSONS.check("bodies predict max_cell_types cells 6 4", "1 0").ok
    assert not LESSONS.check("bodies predict max_cell_types cells 6 4", "1 1").ok
    assert LESSONS.check("bodies predict oxygen_max_size oxygen 0 point 1", "5 1 2").ok
    assert LESSONS.check("bodies predict supported_levels light 1 0 0 efficiency 0 point 1", "3").ok
    assert LESSONS.check("bodies predict vulnerability prey 4 0 9 6 hunter 1 3 1 0 7 2", "0 point 9 6 9 7").ok
    assert LESSONS.check("bodies predict needed_cell_types cells 6 4", "2").ok
    assert LESSONS.check("bodies predict fraction_eaten pressure 0", "0").ok
    v = LESSONS.check("bodies predict kleiber mass 1 6", "8 point 1")
    assert not v.ok and v.expected == "8"
    assert LESSONS.check("bodies predict kleiber mass 0", "0").reason == "not my question"
    rng = random.Random(9)
    for ln in rng.sample(lessons, 200):  # an altered answer is rejected
        words = ln.answer.split()
        place = rng.choice([i for i, w in enumerate(words) if w.isdigit()])
        words[place] = str((int(words[place]) + rng.randint(1, 9)) % 10)
        wrong = " ".join(words)
        if parse_num(wrong) != parse_num(ln.answer):
            assert not LESSONS.check(ln.prompt, wrong).ok, (ln.text, wrong)
    assert [ln.text for ln in LESSONS.generate(random.Random(1), 50)] == [ln.text for ln in lessons[:50]]
    assert [ln.text for ln in LESSONS.generate(random.Random(2), 50)] != [ln.text for ln in lessons[:50]]
    # body sizes are drawn over the doublings: small bodies and powers of two are common
    sizes = [values[0] for name, values in (LESSONS.parse(ln.prompt) for ln in LESSONS.generate(random.Random(3), 3000))
             if name in ("energy_per_cell", "surface_to_volume", "max_cell_types")]
    assert len(sizes) > 250
    assert sum(s <= 64 for s in sizes) > 0.2 * len(sizes) and sum(s >= 4096 for s in sizes) > 0.2 * len(sizes)
    assert sum(s & (s - 1) == 0 for s in sizes) > 0.4 * len(sizes)
    assert 1 <= min(sizes) and max(sizes) <= 1 << 20
    records = LESSONS.records()
    assert len(records) == len(RULES)
    for rec in records:
        assert is_dense(rec.text) and n_tokens(rec.text) <= 128 and rec.kind == "record", rec.text
        assert rec.text.startswith("bodies predict ")
    told = {rec.text.split(".")[0].split()[-1]: rec.text for rec in records}
    for name in ("adhesion", "max_cell_types", "needed_cell_types", "supply", "reachable", "vulnerability",
                 "oxygen_max_size", "predator_activity", "supported_levels"):
        assert f"bodies predict {name}. toy rule. " in told[name], "the invented rules say so"
    for name in ("upkeep", "growth_rate", "attack_rate", "handling_time", "prey_step", "grazer_step"):
        assert f"bodies predict {name}. toy numbers. " in told[name], "and so do the rules with invented constants"
    for name in ("kleiber", "energy_per_cell", "surface_to_volume", "production", "logistic_step",
                 "predation_pressure", "fraction_eaten", "pyramid_flow"):
        assert "toy" not in told[name]


# ---------------------------------------------------------------------------------------------
# plausibility
# ---------------------------------------------------------------------------------------------

def test_every_stage_and_outcome_occurs(many):
    finals = Counter(r.summary["outcome"] for r in many.values())
    assert set(finals) == {"single_cells", "colonies", "multicellular", "animals_like"}
    assert min(finals.values()) >= 20
    stages = Counter(s["stage"] for r in many.values() for s in r.steps)
    assert set(stages) == set(STAGES) and min(stages.values()) >= 200
    for r in many.values():
        assert r.steps[0]["stage"] == "single_cells" and r.steps[0]["species"] == 1
        assert r.steps[0]["trophic_levels"] == 1 and r.steps[0]["max_size"] == 1


def test_without_predation_bodies_stay_small(many, controlled):
    alone = [r for r in many.values() if r.params["predation"] == 0]
    assert len(alone) >= 20
    for r in alone + controlled(predation=0.0):
        assert r.summary["max_size"] == 1 and r.summary["outcome"] == "single_cells"
        assert max(s["max_size"] for s in r.steps) <= 2 and max(s["mean_size"] for s in r.steps) < 1.2
        assert all(s["raw_grazers"] == 0 and s["raw_predators"] == 0 and s["trophic_levels"] == 1 for s in r.steps)
        assert all(s["cell_types"] == 1 and s["consumer_size"] == 0 for s in r.steps)
        assert all(ln["guild"] == "producer" for alive in r.lineages for ln in alive)
        # the single cells hold the light; the rest are mutants on their way out
        last = r.lineages[-1]
        assert sum(ln["biomass"] for ln in last if ln["size"] == 1) > 0.85 * sum(ln["biomass"] for ln in last)


def test_with_predation_size_and_cell_types_rise(many, controlled):
    hunted = controlled()
    assert all(r.summary["max_size"] >= 64 and r.summary["cell_types"] >= 4 for r in hunted)
    assert statistics.median(r.summary["max_size"] for r in hunted) >= 256
    assert all(r.summary["outcome"] == "animals_like" for r in hunted)
    # the only difference to the worlds of single cells is the predation
    assert all(r.summary["max_size"] == 1 for r in controlled(predation=0.0))
    mild = controlled(predation=0.2)
    assert all(r.summary["max_size"] >= 32 for r in mild)
    # over the canonical seeds
    eaten = [r for r in many.values() if r.params["predation"] >= 0.5 and r.params["complex_cells"] == "yes"]
    assert len(eaten) >= 80
    assert statistics.median(r.summary["max_size"] for r in eaten) >= 64
    assert statistics.median(r.summary["cell_types"] for r in eaten) >= 4
    assert sum(r.summary["cell_types"] >= 3 for r in eaten) >= 0.75 * len(eaten)
    # bodies grow in the course of a run
    for r in hunted:
        sizes = [s["max_size"] for s in r.steps]
        assert max(sizes[:10]) <= 8 and max(sizes[40:]) >= 64


def test_more_mutations_more_change(controlled):
    slow, fast = controlled(mutation=0.2), controlled(mutation=3.0)
    assert statistics.median(r.summary["max_size"] for r in slow) <= 16
    assert statistics.median(r.summary["max_size"] for r in fast) >= 256
    assert statistics.median(r.summary["cell_types"] for r in fast) > statistics.median(
        r.summary["cell_types"] for r in slow)


def test_oxygen_caps_the_consumers_and_gates_the_predators(many, controlled):
    steps = over = 0
    for r in many.values():
        for s, alive in zip(r.steps, r.lineages):
            if s["consumer_size"]:
                steps += 1
                over += s["consumer_size"] > oxygen_max_size(s["raw_oxygen"])
        for e in r.events:  # no consumer is founded bigger than the oxygen allows, no predator below 0.02
            if e["kind"] in MUTATIONS and e["guild"] != "producer":
                assert e["size"] <= oxygen_max_size(e["oxygen"]), (r.seed, e)
            if e["kind"] == "guild" and e["guild"] == "predator":
                assert e["oxygen"] >= B.O2_PREDATOR
    assert steps > 5000 and over <= 0.001 * steps, "the common consumers stay under the cap"
    thin = [r for r in many.values() if max(s["raw_oxygen"] for s in r.steps) < B.O2_PREDATOR]
    assert len(thin) >= 8
    assert not any(ln["guild"] == "predator" for r in thin for alive in r.lineages for ln in alive)
    assert all(s["trophic_levels"] <= 2 for r in thin for s in r.steps)
    capped = [r for r in many.values() if r.summary["consumer_size"]
              and r.summary["consumer_size"] == oxygen_max_size(r.steps[-1]["raw_oxygen"])]
    assert len(capped) >= 15, "in these worlds the oxygen is what stops the consumers"
    # the same worlds with little and with much oxygen at the start
    little = controlled(light=60.0, efficiency=0.15, mutation=2.5, start_oxygen=0.001)
    much = controlled(light=60.0, efficiency=0.15, mutation=2.5, start_oxygen=0.2)
    first = [[next((t for t, s in enumerate(r.steps) if s["raw_predators"] > 0), STEPS) for r in g] for g in (little, much)]
    assert min(first[0]) >= 8 and statistics.median(first[1]) <= 5, "predators have to wait for the oxygen"
    assert all(b.summary["consumer_size"] > a.summary["consumer_size"] for a, b in zip(little, much))
    assert all(b.summary["oxygen"] > a.summary["oxygen"] for a, b in zip(little, much))
    for r in little + much:
        assert r.summary["consumer_size"] == oxygen_max_size(r.steps[-1]["raw_oxygen"])
    # oxygen is made by the producers: it rises in a world that starts without
    for r in controlled(start_oxygen=0.001):
        assert r.steps[-1]["raw_oxygen"] > 0.05 > r.steps[1]["raw_oxygen"] > r.steps[0]["raw_oxygen"]


def test_trophic_levels_never_exceed_the_energy_pyramid(many, controlled):
    for r in many.values():
        supported = supported_levels(r.params["light"], r.params["efficiency"])
        assert max(s["trophic_levels"] for s in r.steps) <= supported
        assert max(ln["level"] for alive in r.lineages for ln in alive) <= supported
    # too little light and too little efficiency: grazers at most
    poor = controlled(light=20.0, efficiency=0.05)
    assert supported_levels(20.0, 0.05) == 2
    assert all(s["trophic_levels"] <= 2 and s["raw_predators"] == 0 for r in poor for s in r.steps)
    # here the rule allows a third level, and the energy does not carry it: predators are founded
    # in every one of these worlds and are established in none
    lean = controlled(light=80.0, efficiency=0.05)
    assert supported_levels(80.0, 0.05) == 3
    assert all(any(ln["guild"] == "predator" for alive in r.lineages for ln in alive) for r in lean)
    assert all(s["trophic_levels"] <= 2 and s["raw_predators"] < 0.5 for r in lean for s in r.steps)
    # more efficiency, more levels
    assert all(r.summary["trophic_levels"] <= 3 for r in controlled(efficiency=0.05))
    assert all(r.summary["trophic_levels"] == 4 for r in controlled(efficiency=0.2))
    assert all(r.summary["trophic_levels"] >= 3 for r in controlled())
    # more light, more levels
    assert all(r.summary["trophic_levels"] == 2 for r in controlled(light=20.0))
    assert sum(r.summary["trophic_levels"] == 4 for r in controlled(light=1000.0)) >= 10
    # over the canonical seeds
    hunted = [r for r in many.values() if r.params["predation"] >= 0.5]
    low = [r for r in hunted if r.params["efficiency"] < 0.1]
    high = [r for r in hunted if r.params["efficiency"] >= 0.15]
    assert len(low) >= 30 and len(high) >= 30
    share = lambda group, n: sum(r.summary["trophic_levels"] >= n for r in group) / len(group)  # noqa: E731
    assert share(low, 3) < 0.5 < 0.75 < share(high, 3)
    assert share(low, 4) == 0 and share(high, 4) > 0.1
    dark = [r for r in hunted if r.params["light"] < 60]
    bright = [r for r in hunted if r.params["light"] >= 300]
    assert share(dark, 3) < 0.4 < 0.75 < share(bright, 3)


def test_without_complex_cells_no_differentiated_bodies(many, controlled):
    simple = [r for r in many.values() if r.params["complex_cells"] == "no"]
    assert len(simple) >= 40
    rich = controlled(complex_cells="no", light=1000.0, efficiency=0.2, mutation=3.0)
    for r in simple + controlled(complex_cells="no") + rich:
        assert all(ln["size"] <= 64 and ln["cell_types"] <= 2 for alive in r.lineages for ln in alive)
        assert all(s["stage"] not in ("differentiated", "food_web") for s in r.steps)
        assert r.summary["outcome"] != "animals_like"
    # the same worlds with complex cells do get there
    assert all(r.summary["outcome"] == "animals_like" for r in controlled())
    assert {r.summary["outcome"] for r in simple} == {"single_cells", "colonies", "multicellular"}
    assert any(s["trophic_levels"] >= 3 for r in rich for s in r.steps), "simple cells can still be predators"


def test_rich_worlds_crash(many):
    """The paradox of enrichment: the stage collapse is a matter of rich worlds."""
    crashed = [r for r in many.values() if any(s["stage"] == "collapse" for s in r.steps)]
    assert len(crashed) >= 30 and min(r.params["light"] for r in crashed) >= 100
    for r in crashed:
        for s in r.steps:
            if s["stage"] == "collapse":
                assert s["raw_producers"] < 0.01 * r.params["light"] and s["raw_grazers"] > 0
    bright = [r for r in many.values() if r.params["light"] >= 300 and r.params["predation"] >= 0.5]
    assert sum(s["stage"] == "collapse" for r in bright for s in r.steps) > 0.05 * STEPS * len(bright)
    # the producers come back: they are never gone
    assert all(s["raw_producers"] > 0.4 for r in many.values() for s in r.steps)


def test_species_and_extinctions(many):
    for r in many.values():
        assert max(s["species"] for s in r.steps) <= 60
    assert max(s["species"] for r in many.values() for s in r.steps) == 60
    assert statistics.median(r.summary["species"] for r in many.values()) >= 10
    assert max(r.steps[-1]["extinctions"] for r in many.values()) > 100
    kinds = Counter(e["kind"] for r in many.values() for e in r.events)
    assert set(kinds) == set(MUTATIONS) | set(GONE) and kinds["extinct"] > 20 * kinds["displaced"] > 0


def test_a_full_vessel_displaces_its_rarest_lineage(sim, many):
    """At 60 lineages a founder pushes out the rarest lineage if that is smaller than the founder."""
    full = [r for r in many.values() if any(e["kind"] == "displaced" for e in r.events)]
    assert 3 <= len(full) <= 40
    crowded = sim.run(1, **dict(BASE, light=1000.0, efficiency=0.2, mutation=20.0))
    assert sim.conserved(crowded).ok and sum(e["kind"] == "displaced" for e in crowded.events) > 100
    assert max(s["species"] for s in crowded.steps) == 60
    for r in full + [crowded]:
        gone = {e["id"]: e["step"] for e in r.events if e["kind"] in GONE}
        assert len(gone) == sum(e["kind"] in GONE for e in r.events) == r.steps[-1]["extinctions"]
        for e in r.events:
            if e["kind"] == "displaced":  # the vessel was full within that step, and the lineage is gone for good
                assert max(r.steps[e["step"] - 1]["species"], r.steps[e["step"]]["species"]) >= 55
                assert all(ln["id"] != e["id"] for alive in r.lineages[e["step"]:] for ln in alive)
    # a world far from full displaces nobody
    assert not any(e["kind"] == "displaced" for r in many.values() if max(s["species"] for s in r.steps) < 55
                   for e in r.events)


# ---------------------------------------------------------------------------------------------
# hand-off
# ---------------------------------------------------------------------------------------------

def test_handoff_in():
    assert handoff_in({"outcome": "complex_cells", "complex": 0.93, "energy_per_cell": 152.0}) == {
        "complex_cells": "yes", "efficiency": 0.2}
    assert handoff_in({"outcome": "cells", "complex": 0.0, "energy_per_cell": 10.0}) == {
        "complex_cells": "no", "efficiency": 0.05}
    assert handoff_in({"outcome": "protocells", "energy_per_cell": 10.0})["complex_cells"] == "no"
    assert handoff_in({"outcome": "complex_cells", "energy_per_cell": 81.9})["efficiency"] == 0.16
    assert handoff_in({"outcome": "complex_cells", "energy_per_cell": 20.0})["efficiency"] == 0.09
    # four times the energy is half way on the logarithm: 0.125, on the edge of two hundredths, goes up
    assert handoff_in({"outcome": "complex_cells", "energy_per_cell": 40.0})["efficiency"] == 0.13
    assert handoff_in({"outcome": "cells"})["efficiency"] == 0.05, "no energy told: that of a cell without a partner"
    assert handoff_in({"complex": 0.7})["complex_cells"] == "yes" and handoff_in({"complex": 0.2})["complex_cells"] == "no"
    assert handoff_in({"outcome": "cells", "energy_per_cell": 1.0})["efficiency"] == 0.05
    assert handoff_in({"outcome": "cells", "energy_per_cell": 5000.0})["efficiency"] == 0.2
    values = [handoff_in({"outcome": "cells", "energy_per_cell": float(e)})["efficiency"] for e in range(10, 161)]
    assert values == sorted(values) and values[0] == 0.05 and values[-1] == 0.2
    assert set(values) == {round(0.05 + k / 100, 2) for k in range(16)}, "every hundredth from 0.05 to 0.2 occurs"
    assert all(0.05 <= handoff_in({"outcome": "cells", "energy_per_cell": e / 10})["efficiency"] <= 0.2
               for e in range(1, 5000, 7))
    for bad in ({"outcome": "collapse"}, {"outcome": "cells", "cells": 0}, {"outcome": "cells", "energy_per_cell": 0.0},
                {"outcome": "cells", "energy_per_cell": float("nan")}):
        with pytest.raises(ValueError):
            handoff_in(bad)


def test_handoff_runs_a_world_and_gives_upward(sim, rollouts):
    params = dict(BASE, **handoff_in({"outcome": "complex_cells", "energy_per_cell": 81.9}))
    assert params["efficiency"] == 0.16 and params["complex_cells"] == "yes"
    r = sim.run(11, **params)
    assert sim.conserved(r).ok
    for r in list(rollouts.values()) + [r]:
        up = handoff_out(r.summary)
        assert set(up) == {"body_size", "cell_types", "predators", "predator_share", "oxygen", "species"}
        assert up["predators"] == ("yes" if r.summary["trophic_levels"] >= 3 else "no")
        assert up["oxygen"] == r.summary["oxygen"] and up["species"] == r.summary["species"]
        if r.summary["consumer_size"]:
            assert (up["body_size"], up["cell_types"]) == (r.summary["consumer_size"], r.summary["consumer_types"])
        else:
            assert (up["body_size"], up["cell_types"]) == (r.summary["max_size"], r.summary["cell_types"])
        assert up["body_size"] >= 1 and 1 <= up["cell_types"] <= max_cell_types(up["body_size"])
    assert handoff_out(rollouts[3].summary) == {"body_size": 128, "cell_types": 5, "predators": "yes",
                                                "predator_share": 0.0158, "oxygen": 0.0485, "species": 31}


# ---------------------------------------------------------------------------------------------
# the review of round 7: what the level must still do after hardening
# ---------------------------------------------------------------------------------------------

def test_animals_like_are_differentiated_consumers(sim, many):
    """``animals_like`` asks for common consumers of more than 32 cells with three or more cell
    types among established predators; differentiated producers alone do not make a world of it."""
    animals = 0
    for r in many.values():
        s = r.summary
        eats = s["consumer_size"] > 32 and s["consumer_types"] >= 3 and s["trophic_levels"] >= 3
        assert (s["outcome"] == "animals_like") == eats, (r.seed, s)
        animals += eats
    assert animals >= 40
    r = sim.rollout(301)  # producers of 64 cells and 4 cell types, consumers of 32 cells, three levels
    assert [r.summary[k] for k in ("max_size", "cell_types", "consumer_size", "consumer_types", "trophic_levels")] == [
        64, 4, 32, 4, 3]
    assert r.summary["stage"] == "food_web" and r.summary["outcome"] == "multicellular"
    assert outcome_of(128, 5, 3, 16, 3) == outcome_of(128, 5, 3, 64, 2) == outcome_of(128, 5, 2, 128, 5) == "multicellular"
    assert outcome_of(128, 5, 3, 64, 3) == "animals_like" == outcome_of(128, 5, 3), "without consumers told: as before"
    assert outcome_of(16, 5, 4, 16, 5) == "colonies" and outcome_of(1, 1, 3, 1, 1) == "single_cells"
    assert outcome_of(0, 0, 3, 0, 0) == "collapse"


def test_a_full_vessel_never_pushes_out_the_last_producers(sim, monkeypatch):
    """A hand-set world a thousand times as rich as any drawn one: the vessel is full, the founders
    of the predators are large and the producers sit in their refuge. As the rule was, the
    founders pushed every producer lineage out and the world starved; now the last one stays."""
    world = _World(100.0, 0.1, 1.0, 0.1, True)
    assert world.last_producer() == -1
    world.add(1, 0, 1, 1.0)
    world.add(0, 0, 1, 5.0)
    assert world.last_producer() == 1
    world.add(0, 1, 1, 5.0)
    assert world.last_producer() == -1
    rich = dict(light=1e6, mutation=20.0, efficiency=0.5)
    took = []
    for _ in range(2):
        start = time.perf_counter()
        r = sim.run(3, **rich)
        took.append(time.perf_counter() - start)
    assert min(took) < 2.0
    assert sim.conserved(r).ok
    assert all(any(ln["guild"] == "producer" for ln in alive) for alive in r.lineages)
    assert min(s["raw_producers"] for s in r.steps) > 0.1 and r.summary["outcome"] == "animals_like"
    assert sum(e["kind"] == "displaced" for e in r.events) > 1000
    for seed in (1, 2):
        other = sim.run(seed, light=1e5, mutation=6.0, efficiency=0.5, predation=5.0)
        assert sim.conserved(other).ok and min(s["raw_producers"] for s in other.steps) > 0
    # the rule as it was, and what the gate says to it
    monkeypatch.setattr(_World, "last_producer", lambda self: -1)
    dead = sim.run(3, **rich)
    assert min(s["raw_producers"] for s in dead.steps) == 0 and dead.summary["outcome"] == "collapse"
    assert dead.summary["species"] > 0, "consumers were left, starving"
    v = sim.conserved(dead)
    assert not v.ok and "the last producer lineage was pushed out" in v.reason


def pair_by_hand(producers, grazers, light, efficiency, predation):
    p1, g1 = 0.95 * producers, 0.95 * grazers
    reach = max(0.0, p1 - 0.5)
    attack = 0.05 * predation
    eaten = reach * (1 - math.exp(-0.5 * attack * g1 / (1 + attack * 0.125 * reach)))
    rest = p1 - eaten
    return rest + 0.6 * rest * max(0.0, 1 - rest / light), g1 + efficiency * eaten


def cell_types_by_hand(cells):
    most = 0
    while 4 ** (most + 1) <= cells ** 3:  # 2^m <= cells^1.5
        most += 1
    return min(30, 1 + most)


def levels_by_hand(light, efficiency):
    n = 0
    for level in (1, 2, 3, 4):
        if 0.15 * light * efficiency ** (level - 1) < 0.025:
            break
        n = level
    return max(1, n)


def supply_by_hand(cells, cell_types):
    return min(1.0, cell_types * cells ** (-1 / 3))


#: every lesson rule once more, written from the module docstring with ``**`` and the C library
BY_HAND = {
    "kleiber": lambda mass: mass ** 0.75,
    "energy_per_cell": lambda cells: cells ** -0.25,
    "surface_to_volume": lambda cells: 3 / cells ** (1 / 3),
    "adhesion": lambda cells: 1 - 1 / cells,
    "max_cell_types": cell_types_by_hand,
    "needed_cell_types": lambda cells: 1 if cells <= 32 else 2,
    "supply": supply_by_hand,
    "upkeep": lambda cells, types: 0.05 * cells ** -0.25 * (1 + 0.03 * (types - 1)),
    "growth_rate": lambda cells, types: 0.6 * cells ** -0.25 * supply_by_hand(cells, types),
    "attack_rate": lambda predation, cells, types: 0.05 * predation * cells ** -0.25 * supply_by_hand(cells, types),
    "handling_time": lambda cells, types: 0.25 / (cells ** -0.25 * supply_by_hand(cells, types)),
    "vulnerability": lambda prey, hunter: hunter / (hunter + prey),
    "reachable": lambda total: max(0.0, 1 - 0.5 / total),
    "predation_pressure": lambda attack, hunters, handling, food: attack * hunters / (1 + attack * handling * food),
    "fraction_eaten": lambda pressure: 1 - math.exp(-pressure),
    "production": lambda biomass, rate, total, capacity: rate * biomass * max(0.0, 1 - total / capacity),
    "logistic_step": lambda biomass, rate, capacity: biomass + rate * biomass * max(0.0, 1 - biomass / capacity),
    "pyramid_flow": lambda made, efficiency, level: made * efficiency ** (level - 1),
    "supported_levels": levels_by_hand,
    "oxygen_max_size": lambda oxygen: max(1, 2 ** math.floor(math.log2(max(1.0, 16384 * oxygen ** 1.5)) + 1e-12)),
    "predator_activity": lambda oxygen: min(1.0, oxygen / 0.02),
    "prey_step": lambda *inputs: pair_by_hand(*inputs)[0],
    "grazer_step": lambda *inputs: pair_by_hand(*inputs)[1],
}


def test_lessons_of_four_seeds_recomputed_by_hand_and_not_one_answer():
    assert set(BY_HAND) == set(RULES)
    answers = defaultdict(Counter)
    for seed in (1, 2, 3, 4):
        lessons = LESSONS.generate(random.Random(seed), 500)
        assert len(lessons) == 500
        assert [ln.text for ln in lessons] == [ln.text for ln in LESSONS.generate(random.Random(seed), 500)]
        for ln in lessons:
            assert is_dense(ln.text) and n_tokens(ln.text) <= 128 and "." not in ln.prompt and "." not in ln.answer
            v = LESSONS.check(ln.prompt, ln.answer)
            assert v.ok and v.expected == ln.answer, ln.text
            name, values = LESSONS.parse(ln.prompt)
            mine, told = BY_HAND[name](*values), parse_num(ln.answer)
            if isinstance(RULES[name].compute(*values), int):
                assert mine == told and isinstance(told, int), (ln.text, mine)
            else:  # the answer has 4 significant digits
                assert told == pytest.approx(mine, rel=6e-4, abs=1e-12), (ln.text, mine)
            answers[name][ln.answer] += 1
    assert set(answers) == set(RULES)
    top = {name: seen.most_common(1)[0][1] / sum(seen.values()) for name, seen in answers.items()}
    assert set(answers["needed_cell_types"]) == {"1", "2"} and top["needed_cell_types"] < 0.6
    assert set(answers["supported_levels"]) == {"1", "2", "3", "4"}
    for name, share in top.items():
        if name != "needed_cell_types":
            assert share < 0.45, (name, share, answers[name].most_common(3))
    assert sum(share < 0.2 for share in top.values()) >= 17


def test_lessons_are_drawn_where_the_rollouts_are():
    by = defaultdict(list)
    for ln in LESSONS.generate(random.Random(5), 4000):
        name, values = LESSONS.parse(ln.prompt)
        by[name].append((values, ln.answer))
    share = lambda name, test: sum(test(v, a) for v, a in by[name]) / len(by[name])  # noqa: E731
    # producers as a share of their capacity: rarely above it, where nothing grows
    assert all(b <= t <= 1.15 * k + 0.05 for (b, _, t, k), _ in by["production"])
    assert 0.05 < share("production", lambda v, a: a == "0") < 0.2
    assert all(b <= 1.15 * k + 0.05 for (b, _, k), _ in by["logistic_step"])
    assert 0.05 < share("logistic_step", lambda v, a: parse_num(a) == v[0]) < 0.2
    for name in ("prey_step", "grazer_step"):
        assert all(1 <= p <= max(1.0, 1.15 * light + 0.05) for (p, _, light, _, _), _ in by[name])
        assert share(name, lambda v, a: v[0] > v[2]) < 0.2
    # the rules with few answers
    assert all(cells <= 1024 for (cells,), _ in by["needed_cell_types"])
    assert 0.4 < share("needed_cell_types", lambda v, a: a == "1") < 0.6
    assert all(oxygen <= 0.03 for (oxygen,), _ in by["predator_activity"])
    assert 0.25 < share("predator_activity", lambda v, a: a == "1") < 0.42
    levels = Counter(a for _, a in by["supported_levels"])
    assert set(levels) == {"1", "2", "3", "4"} and min(levels[n] for n in "234") > 0.2 * len(by["supported_levels"])
    assert 0.15 < share("reachable", lambda v, a: a == "0") < 0.4
    assert len({a for _, a in by["oxygen_max_size"]}) == 15, "every doubling from 1 to 16384 cells"
    # the gate judges the whole range all the same
    assert LESSONS.check("bodies predict needed_cell_types cells 1 0 4 8 5 7 6", "2").ok
    assert LESSONS.check("bodies predict predator_activity oxygen 0 point 0 4 5", "1").ok
    assert LESSONS.check("bodies predict supported_levels light 2 efficiency 0 point 0 5", "1").ok
    assert LESSONS.check("bodies predict supported_levels light 1 0 0 0 efficiency 0 point 2", "4").ok
    assert LESSONS.check("bodies predict production biomass 9 0 0 rate 0 point 5 total 1 1 0 0 capacity 1 0 0 0", "0").ok
    assert LESSONS.check("bodies predict logistic_step biomass 1 1 0 0 rate 0 point 5 capacity 1 0 0 0", "1 1 0 0").ok
    assert not LESSONS.owns("bodies predict production biomass 9 0 0 rate 0 point 5 total 8 0 0 capacity 1 0 0 0")


def test_owns_nothing_of_the_other_levels_or_of_round_5(sim):
    """The questions the other levels and the round-5 gates really write, not hand-made ones."""
    levels = 0
    for name in ("haishool.evo.stars", "haishool.evo.cells", "haishool.evo.senses", "haishool.evo.signals",
                 "haishool.evo.society", "haishool.cosmos.planets", "haishool.cosmos.life"):
        try:  # another level may be in the middle of a change
            theirs = importlib.import_module(name).LESSONS.generate(random.Random(1), 150)
        except Exception:
            continue
        levels += 1
        for ln in theirs:
            assert not B.owns(ln.prompt) and not B.check(ln.prompt, ln.answer).ok, ln.text
            assert B.check(ln.prompt, ln.answer).reason == "not my question"
    assert levels >= 2
    predict = importlib.import_module("haishool.cosmos.predict")
    gates = [importlib.import_module("haishool.truth." + topic).gate() for topic in ("maths", "elements")]
    gates += [predict.gate(topic) for topic in predict.TOPICS]
    asked = 0
    for gate in gates:  # round 5, and the lessons of round 6
        for ln in gate.records()[:50] + gate.generate(random.Random(1), 100):
            assert not B.owns(ln.prompt) and not sim.owns(ln.prompt) and not LESSONS.owns(ln.prompt), ln.text
            asked += 1
    assert asked >= 800
    for prompt in ("solve 3 x plus 4 equals 1 9", "check 1 2 plus 7 equals 2 0", "calc 1 2 plus 7", "carbon protons",
                   "water molar_mass", "predict gravity collapse_time mass 1", "bodies predict", "predict bodies kleiber mass 1 6",
                   "senses predict kleiber mass 1 6", "bodies7 predict kleiber mass 1 6", "bodies seed 3 say predator near"):
        assert not B.owns(prompt), prompt


def test_the_hand_off_fits_the_levels_around_it(rollouts):
    """Level 8 gives 10 units of energy to a cell without a working partner and 16 times that
    to one with; level 10 reads what this level hands upward."""
    try:
        cells = importlib.import_module("haishool.evo.cells")
        senses = importlib.import_module("haishool.evo.senses")
    except Exception:
        pytest.skip("the levels around are being worked on")
    assert B.CELL_ENERGY == cells.ENERGY == 10.0 and B.CELL_GAIN == cells.ENDO_GAIN == 16.0
    assert handoff_in({"outcome": "cells", "energy_per_cell": cells.ENERGY})["efficiency"] == 0.05
    assert handoff_in({"outcome": "complex_cells", "energy_per_cell": cells.ENERGY * cells.ENDO_GAIN})["efficiency"] == 0.2
    assert set(handoff_in({"outcome": "cells"})) <= set(PARAM_KEYS)
    for name in ("outcome", "complex", "energy_per_cell", "cells"):
        assert name in cells.HANDOFF_OUT, name
    for r in rollouts.values():
        up = senses.handoff_in(handoff_out(r.summary))
        assert up["body_size"] == (r.summary["consumer_size"] or r.summary["max_size"]) and up["cell_types"] >= 1
        assert (up["predators"] > 0) == (r.summary["trophic_levels"] >= 3 and r.summary["predator_share"] > 0)
        assert up["oxygen"] == min(1.0, r.summary["oxygen"])


def test_the_drawn_parameters_ask_no_c_library():
    rng = random.Random(3)
    for _ in range(20000):
        x = rng.uniform(-10.0, 10.0)
        assert B._exp(x) == pytest.approx(math.exp(x), rel=1e-13)
    assert B._exp(0.0) == 1.0 and B._exp(B.LN_1000) == pytest.approx(1000.0, rel=1e-14)
    assert [B.LN_20, B.LN_1000, B.LN_0001, B.LN_02, B.LN_1200, B.LN_16] == pytest.approx(
        [math.log(x) for x in (20.0, 1000.0, 0.001, 0.2, 1200.0, 16.0)], rel=1e-15)
    for seed in range(3000):  # the parameters are those the C library's exp and log give
        rng = random.Random(seed)
        light = float(round(math.exp(rng.uniform(math.log(20.0), math.log(1000.0)))))
        efficiency = round(rng.uniform(0.05, 0.2), 2)
        none, strength = rng.random() < 0.15, round(rng.uniform(0.2, 2.0), 2)
        mutation = round(rng.uniform(0.2, 3.0), 2)
        oxygen = float(f"{math.exp(rng.uniform(math.log(0.001), math.log(0.2))):.2g}")
        assert random_params(random.Random(seed)) == {
            "light": light, "efficiency": efficiency, "predation": 0.0 if none else strength, "mutation": mutation,
            "start_oxygen": oxygen, "complex_cells": "yes" if rng.random() < 0.7 else "no"}, seed
    # the module: no numpy, and of the C library's functions the square root alone in the simulation
    tree = ast.parse(Path(B.__file__).read_text(encoding="utf-8"))
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names} | {
        n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert imported == {"__future__", "math", "random", "dataclasses", "haishool.cosmos", "haishool.evo", "haishool.truth"}
    used = Counter(n.attr for n in ast.walk(tree)
                   if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "math")
    assert set(used) == {"sqrt", "isfinite", "fsum", "log"} and used["log"] == 1, "the one log is the hand-off's"
    powers = [n for n in ast.walk(tree) if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Pow)]
    assert len(powers) == 1, "one power, of whole numbers, in max_cell_types"


#: sha256 over the rollouts and lines of five seeds and 400 lessons: the bits every machine must give
PINNED = "4bc0688dd5870344718d832e3a7330c437ad1a046ee19314567f30e6e57e6fa7"
DIGEST = """
import hashlib, json, random
import haishool.evo.bodies as B
out = []
for seed in (1, 2, 3, 7, 11):
    r = B.Bodies().rollout(seed)
    out.append([r.params, r.steps, r.lineages, r.events, r.summary, [ln.text for ln in B.lines(r)]])
out.append([ln.text for ln in B.LESSONS.generate(random.Random(1), 400)])
print(hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest())
"""


def test_same_bits_in_another_process_whatever_the_hash_seed():
    """Nothing depends on the order of a set or on the hash of a string: two fresh interpreters
    with different hash seeds give the pinned digest."""
    root = str(Path(B.__file__).resolve().parents[2])
    for hash_seed in ("0", "4242"):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed, PYTHONPATH=root)
        done = subprocess.run([sys.executable, "-c", DIGEST], env=env, capture_output=True, text=True, cwd=root)
        assert done.returncode == 0, done.stderr
        assert done.stdout.strip() == PINNED


def test_what_the_docstring_says_of_light_and_of_the_starting_oxygen(controlled):
    doublings = lambda group: statistics.mean(r.summary["max_size"].bit_length() - 1 for r in group)  # noqa: E731
    dim, middling, bright = controlled(light=20.0), controlled(light=150.0), controlled(light=1000.0)
    # the bodies are largest in middling light: the rich worlds crash, the poor carry no predators
    assert doublings(dim) + 3 < doublings(middling) and doublings(bright) + 0.5 < doublings(middling)
    assert not any(s["stage"] == "collapse" for r in dim + middling for s in r.steps)
    assert sum(s["stage"] == "collapse" for r in bright for s in r.steps) > 0.1 * STEPS * len(bright)
    # the starting oxygen is still seen at the end of a dim world, hardly in a bright one
    ratio = {}
    for light in (30.0, 1000.0):
        little, much = controlled(light=light, start_oxygen=0.001), controlled(light=light, start_oxygen=0.2)
        ratio[light] = [b.steps[-1]["raw_oxygen"] / a.steps[-1]["raw_oxygen"] for a, b in zip(little, much)]
    assert min(ratio[30.0]) > 1.8 and statistics.median(ratio[30.0]) > 2.2
    assert statistics.median(ratio[1000.0]) < 1.2
    assert 0.999 ** (TICKS * (STEPS - 1)) == pytest.approx(0.206, abs=0.001), "what the rocks leave of the start"


def test_one_parameter_forced_as_the_docstring_measures_it(sim):
    """The protocol of the docstring's forced-regime paragraph: the canonical parameters of a seed,
    predation set to 1.2, then one parameter forced (here on seeds 1 to 30, with margins; the
    docstring gives seeds 1 to 80)."""
    cache: dict[tuple, list] = {}

    def forced(**changed):
        key = tuple(sorted(changed.items()))
        if key not in cache:
            cache[key] = []
            for seed in range(1, 31):
                p = dict(random_params(random.Random(seed)), predation=1.2)
                p.update(changed)
                cache[key].append(sim.run(seed, **p))
        return cache[key]

    three = lambda group: sum(r.summary["trophic_levels"] >= 3 for r in group)  # noqa: E731
    collapsed = lambda group: sum(s["stage"] == "collapse" for r in group for s in r.steps)  # noqa: E731
    median = lambda group: statistics.median(r.summary["max_size"] for r in group)  # noqa: E731
    # more efficiency, more worlds with three or more levels (measured 5, 18, 29 of 30)
    low, mid, high = (three(forced(efficiency=e)) for e in (0.05, 0.12, 0.2))
    assert low <= 10 < mid < 25 <= high
    # more light, more levels and more crashes (measured 0, 23, 28 of 30; collapse 0, 21, 484 steps)
    dim, middling, bright = forced(light=20.0), forced(light=200.0), forced(light=1000.0)
    assert three(dim) == 0 and three(middling) >= 18 and three(bright) >= 25
    assert collapsed(dim) == 0 and collapsed(bright) > 0.1 * STEPS * len(bright) > collapsed(middling)
    # the bodies are larger in middling light than in dim and in bright worlds (measured 8, 96, 48)
    assert median(dim) < median(middling) and median(bright) < median(middling)
    # predation 0, 0.2, 1.2 with complex cells: the largest common body rises (measured 1, 12, 96)
    alone, mild, hunted = (median(forced(predation=v, complex_cells="yes")) for v in (0.0, 0.2, 1.2))
    assert alone == 1 < mild < hunted and hunted >= 64
