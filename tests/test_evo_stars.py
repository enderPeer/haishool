"""Level 7, stars: closed-box chemical evolution. The gates, the lines, the lessons, the targets."""
import copy
import doctest
import inspect
import math
import random
import re
import time

import pytest

from haishool.cosmos import Simulation, close, nucleo
from haishool.evo import LEVELS, MAX_TOKENS, LessonGate
from haishool.evo import stars as S
from haishool.evo.stars import (BINS, DT, ELEMENTS, EXTRA_KEYS, INPUT_KEYS, KEYS, LESSONS, METALS, PARAM_KEYS, RULES,
                                STAGES, STATE_KEYS, SUMMARY_KEYS, WORD_KEYS, Stars, dense_value, lines, param_value,
                                params_line, random_params, simulation)
from haishool.truth import Gate, is_dense, num, parse_num, split_line

SEEDS = list(range(1, 41))
DEFAULTS = {"efficiency": 0.3, "t_end": 13.5, "hydrogen": 0.752, "helium": 0.248, "ia_fraction": 0.004}

#: lesson rule -> the module function that is its truth and that the simulation calls
FUNCTIONS = {"luminosity": "luminosity", "lifetime": "lifetime", "death_time": "death_time",
             "turnoff": "turnoff_mass", "shines": "shines",
             "fate": "fate", "remnant": "remnant_mass", "returned": "returned_mass", "gas_to_stars": "gas_to_stars",
             "stars_above": "stars_above", "ia_iron": "ia_iron", "closed_box": "closed_box",
             "alpha_over_iron": "alpha_over_iron", "stage": "stage"}


def n_tokens(text: str) -> int:
    """As ``haishool.student.tokens`` counts them (that module needs torch, so not imported)."""
    return len(text.replace(".", " . ").split())


def top_metal(step: dict) -> str:
    return max(METALS, key=lambda e: step[e])


@pytest.fixture(scope="module")
def sim():
    return Stars()


@pytest.fixture(scope="module")
def rollouts(sim):
    return {seed: sim.run(seed, **random_params(random.Random(seed))) for seed in SEEDS}


@pytest.fixture(scope="module")
def all_lines(rollouts):
    return [ln for r in rollouts.values() for ln in lines(r)]


@pytest.fixture(scope="module")
def default_run(sim):
    return sim.run(1, **DEFAULTS)


@pytest.fixture(scope="module")
def lessons():
    return LESSONS.generate(random.Random(1), 400)


# --------------------------------------------------------------------------- the contract

def test_protocols():
    s = simulation()
    assert isinstance(s, Simulation) and isinstance(s, Gate)
    assert S.SIM == s.sim == s.topic == "stars" and "stars" in LEVELS and s.records() == []
    assert simulation() is s
    for key in PARAM_KEYS + STATE_KEYS + EXTRA_KEYS + SUMMARY_KEYS:
        assert key in KEYS, key
    assert WORD_KEYS <= set(KEYS) and INPUT_KEYS <= set(PARAM_KEYS) and set(ELEMENTS) <= set(KEYS)
    assert isinstance(LESSONS, LessonGate) and S.lesson_gate() is LESSONS
    assert LESSONS.sim == "stars" and LESSONS.topic == "predict_stars" and len(RULES) >= 8
    for name in ("run", "conserved", "lines", "check", "owns", "random_params", "handoff_in", "rollout", "cloud"):
        assert callable(getattr(S, name)), name
    assert set(STATE_KEYS) >= {"time", "gas", "stars", "remnants", "metallicity", "helium", "oxygen", "carbon", "iron",
                               "nitrogen", "supernovae_cc", "supernovae_ia", "alpha_over_iron", "stage"}
    assert set(SUMMARY_KEYS) >= {"metallicity", "oxygen", "carbon", "iron", "nitrogen", "neon", "magnesium", "silicon",
                                 "helium", "hydrogen", "gas_fraction", "stars_formed", "supernovae", "time"}


def test_the_docstring_names_its_models_and_its_examples_hold():
    doc = S.__doc__
    for name in ("Salpeter (1955)", "Schmidt (1959)", "Kalirai et al. (2008)", "Woosley and Weaver (1995)",
                 "Matteucci and Greggio (1986)", "Tinsley 1980", "Hand-off", "What is real and what is toy",
                 "Kumar 1963", "Heger et al. (2003)", "Kroupa 2001", "Eddington 1924", "Searle and Sargent 1972",
                 "Asplund, Grevesse, Sauval and Scott 2009"):
        assert name in doc, name
    # what is invented or tuned is called so, and the tuned agreement with the sun is not sold as a result
    for phrase in ("simpler and invented", "put in, not predicted", "Not a law of the model", "toy value",
                   "not measurements of the Milky Way", "labels of this toy", "is stretched here", "from memory",
                   "what the toy tables were tuned to, not findings", "G-dwarf problem"):
        assert phrase in " ".join(doc.split()), phrase
    for description in LESSONS.KEYS.values():
        assert "." not in description.replace(". ", " ") and description == description.lower()
    assert "toy" in RULES["remnant"].description and "toy" in RULES["stage"].description and "toy" in RULES["fate"].description
    assert doctest.testmod(S).failed == 0


def test_module_functions_are_the_simulation(sim):
    p = random_params(random.Random(9))
    a, b = S.run(9, **p), sim.run(9, **p)
    assert a.steps == b.steps and a.summary == b.summary and a.params == b.params and a.ledger == b.ledger
    assert S.rollout(9).steps == a.steps and S.conserved(a).ok
    assert S.owns("stars seed 9 final metallicity") and not S.owns("stars predict lifetime mass 2")
    assert S.check("stars seed 9 final metallicity", dense_value(a.summary["metallicity"])).ok


# --------------------------------------------------------------------------- determinism

def test_same_seed_same_lines():
    a, b = Stars(), Stars()
    for seed in (1, 7, 4321):
        ra, rb = a.rollout(seed), b.rollout(seed)
        assert ra.steps == rb.steps and ra.summary == rb.summary and ra.params == rb.params and ra.ledger == rb.ledger
        assert [ln.text for ln in lines(ra)] == [ln.text for ln in lines(rb)]
    assert [ln.text for ln in lines(a.rollout(1))] != [ln.text for ln in lines(a.rollout(2))]
    assert a.run(5, efficiency=0.6).steps == b.run(5, efficiency=0.6).steps
    assert a.run(5, efficiency=0.6).steps != a.run(5, efficiency=0.61).steps


def test_the_seed_only_names_the_run(sim):
    """No chance in the simulation: the same parameters give the same steps under any seed."""
    one, two = sim.run(5, **DEFAULTS), sim.run(6, **DEFAULTS)
    assert one.steps == two.steps and one.summary == two.summary and (one.seed, two.seed) == (5, 6)
    assert sim.rollout(5).params != sim.rollout(6).params, "the seed picks the parameters"


def test_forty_seeds_replay_exactly(rollouts):
    fresh = Stars()
    for seed, r in rollouts.items():
        again = fresh.run(seed, **random_params(random.Random(seed)))
        assert again.steps == r.steps and again.ledger == r.ledger
        assert again.summary == r.summary and again.params == r.params
        assert [repr(v) for s in again.steps for v in s.values()] == [repr(v) for s in r.steps for v in s.values()]
        assert fresh.rollout(seed).steps == r.steps, "rollout(seed) is the run with the seed's random parameters"


def test_values_are_plain_python_numbers_and_words(rollouts):
    for r in rollouts.values():
        assert type(r.seed) is int
        for row in r.steps + [r.summary, r.params]:
            for key, value in row.items():
                assert (type(value) is str) == (key in WORD_KEYS), key
                assert type(value) in (float, str), (key, type(value))
                if type(value) is float:
                    assert math.isfinite(value) and value >= 0, (key, value)
        for led in r.ledger:
            assert all(type(v) is float for part in ("gas", "astrated", "returned") for v in led[part].values())


def test_generate_is_deterministic(sim):
    one = [ln.text for ln in sim.generate(random.Random(11), 3)]
    two = [ln.text for ln in Stars().generate(random.Random(11), 3)]
    assert one == two and len(one) > 400
    assert one != [ln.text for ln in sim.generate(random.Random(12), 3)]
    assert [ln.text for ln in LESSONS.generate(random.Random(4), 50)] == \
           [ln.text for ln in LESSONS.generate(random.Random(4), 50)]


def test_pinned_lines_of_seed_3(sim):
    """The lines the module docstring shows, word for word."""
    r = sim.rollout(3)
    texts = [ln.text for ln in lines(r)]
    assert texts[0] == ("stars seed 3 params. efficiency 0 point 5 9. t_end 6 point 7 5. hydrogen 0 point 7 6. "
                        "helium 0 point 2 4. ia_fraction 0 point 0 0 7 4. metal_yield 0 point 0 0 8 8 3. "
                        "return_fraction 0 point 2 9 1.")
    assert texts[1] == ("stars seed 3 step 0. time 0. gas 1. stars 0. remnants 0. metallicity 0. helium 0 point 2 4. "
                        "oxygen 0. carbon 0. iron 0. nitrogen 0. supernovae_cc 0. supernovae_ia 0. alpha_over_iron 0. "
                        "stage first_stars.")
    assert ("stars seed 3 step 8. time 2. gas 0 point 3 9 8. stars 0 point 5 5 3. remnants 0 point 0 4 9 1. "
            "metallicity 0 point 0 0 5 6 2. helium 0 point 2 4 9. oxygen 0 point 0 0 2 7 8. "
            "carbon 8 point 2 e minus 4. iron 3 point 6 e minus 4. nitrogen 2 point 4 1 e minus 4. "
            "supernovae_cc 4 5 2 0. supernovae_ia 7 1 point 4. alpha_over_iron 1 point 7 5. stage enriching.") in texts
    for text in ("q stars seed 3 step 8 oxygen. a 0 point 0 0 2 7 8.",
                 "q stars seed 3 step 8 next alpha_over_iron. a 1 point 6 1.",
                 "q stars seed 3 step 8 next stage. a enriching.",
                 "q stars seed 3 final metallicity. a 0 point 0 2 1 1.",
                 "q stars seed 3 final supernovae. a 8 0 9 0.", "q stars seed 3 final time. a 6 point 7 5.",
                 "q stars seed 3 final stage. a mature."):
        assert text in texts, text
    assert r.steps[8]["oxygen"] == pytest.approx(0.00277747317705134, rel=1e-9)
    assert r.summary["metallicity"] == pytest.approx(0.021066118730105513, rel=1e-9)
    assert r.summary["supernovae"] == pytest.approx(8089.030574122535, rel=1e-9)
    assert r.params["metal_yield"] == pytest.approx(0.00882528922068645, rel=1e-9)
    assert len(r.steps) == 28 and len(texts) == 819


def test_a_change_in_the_twelfth_digit_moves_no_line(sim, rollouts):
    """Another machine's exp or power may differ in the sixteenth digit. A change ten thousand
    times larger must not move a single line: no decision sits on a knife's edge."""
    for seed, r in rollouts.items():
        base = [ln.text for ln in lines(r)]
        p = random_params(random.Random(seed))
        for f in (1 + 1e-12, 1 - 1e-12):
            helium = p["helium"] * f
            nudged = sim.run(seed, efficiency=p["efficiency"] * f, t_end=p["t_end"], hydrogen=1.0 - helium,
                             helium=helium, ia_fraction=p["ia_fraction"] / f)
            assert nudged.steps != r.steps
            assert [ln.text for ln in lines(nudged)] == base, (seed, f)


def test_no_mass_bin_dies_on_a_step():
    """A lifetime that was a whole number of steps to the last digit would make the step of death
    a matter of rounding. The bins are cut at exactly those masses, so the mean mass of a bin
    stays clear of them: the nearest miss is 0.024 of a step (the sliver from 2.5 to 2.512)."""
    slow = [b for b in BINS if DT < b.lifetime < 2 * S.T_MAX]
    nearest = min(abs(b.lifetime / DT - round(b.lifetime / DT)) for b in slow)
    assert nearest > 0.02 > 1e6 * S.T_EPS / DT
    for b in BINS:
        for born in (1, 7, 30):
            step = S._death_step(born, b)
            assert step >= born and (step - 1) * DT < max(S.death_time((born - 1) * DT, b.mass), born * DT) <= step * DT + 1e-9
    # every star of a bin dies in the same step: the lifetimes at its two edges lie within one step
    for b in slow:
        if b.lifetime <= S.T_MAX:
            n = S._death_step(1, b)
            assert (n - 1) * DT - 1e-9 <= S.lifetime(b.hi) < b.lifetime < S.lifetime(b.lo) <= n * DT + 1e-9, b
    assert all(S._death_step(5, b) == 5 for b in BINS if b.lifetime < DT), "massive stars die in the step that formed them"


# --------------------------------------------------------------------------- parameters

def test_random_params_stay_in_range():
    seen = set()
    for seed in range(300):
        p = random_params(random.Random(seed))
        assert p == random_params(random.Random(seed)) == random_params(random.Random(seed), rules=7)
        assert list(p) == ["efficiency", "t_end", "hydrogen", "helium", "ia_fraction"]
        assert 0.1 <= p["efficiency"] <= 1.0 and p["efficiency"] == round(p["efficiency"], 2)
        assert 1.0 <= p["t_end"] <= 13.5 and (p["t_end"] / DT).is_integer()
        assert 0.23 <= p["helium"] <= 0.27 and abs(p["hydrogen"] + p["helium"] - 1) < 1e-12
        assert 0.001 <= p["ia_fraction"] <= 0.008
        seen.add(p["t_end"])
    assert min(seen) == 1.0 and max(seen) == 13.5
    assert random_params(random.Random(3)) == {"efficiency": 0.59, "t_end": 6.75, "hydrogen": 0.76, "helium": 0.24,
                                               "ia_fraction": 0.0074}
    with pytest.raises(ValueError):
        random_params(random.Random(3), rules=6)


def test_run_rejects_impossible_parameters(sim):
    for bad in ({"efficiency": 0.0}, {"efficiency": -1.0}, {"efficiency": float("nan")}, {"efficiency": 5.0},
                {"t_end": -0.25}, {"t_end": 13.75}, {"t_end": 1.1}, {"t_end": float("inf")},
                {"hydrogen": 0.8}, {"helium": 0.3}, {"hydrogen": 1.2, "helium": -0.2}, {"hydrogen": 0.0, "helium": 1.0},
                {"ia_fraction": -0.001}, {"ia_fraction": 0.2}, {"rules": 6}):
        with pytest.raises(ValueError):
            sim.run(1, **bad)
    assert sim.run(1, rules=7).steps == sim.run(1).steps
    assert sim.run(1).params == {**DEFAULTS, "metal_yield": pytest.approx(0.0078662, rel=1e-4),
                                 "return_fraction": pytest.approx(0.28987, rel=1e-4)}
    extremes = [{"efficiency": 4.0}, {"efficiency": 0.01}, {"hydrogen": 1.0, "helium": 0.0},
                {"hydrogen": 0.51, "helium": 0.49}, {"ia_fraction": 0.1}, {"ia_fraction": 0.0}, {"t_end": 0.25}]
    for params in extremes:
        r = sim.run(1, **params)
        assert sim.conserved(r).ok, (params, sim.conserved(r))


def test_parameters_are_written_as_given(sim):
    assert param_value("t_end", 10.25) == "1 0 point 2 5" and param_value("efficiency", 0.59) == "0 point 5 9"
    assert param_value("ia_fraction", 0.0074) == "0 point 0 0 7 4" and param_value("metal_yield", 0.0088253) == "0 point 0 0 8 8 3"
    for seed in range(1, 200):
        p = random_params(random.Random(seed))
        r = sim.rollout(seed)
        for key in INPUT_KEYS:
            assert parse_num(param_value(key, r.params[key])) == p[key], (seed, key)
            v = sim.check(f"stars seed {' '.join(str(seed))} params {key}", param_value(key, p[key]))
            assert v.ok and v.expected == param_value(key, p[key])
    assert sim.check("stars seed 3 params t_end", "6 point 7 5").ok
    assert not sim.check("stars seed 3 params t_end", "9 point 7 5").ok
    assert sim.check("stars seed 3 params t_end", "9 point 7 5").expected == "6 point 7 5"


# --------------------------------------------------------------------------- lines

def test_every_line_is_dense_and_short(all_lines):
    assert len(all_lines) > 30000
    for ln in all_lines:
        assert is_dense(ln.text), ln.text
        assert n_tokens(ln.text) <= MAX_TOKENS == 128, ln.text
        assert ln.topic == "stars"
        if ln.kind != "record":
            assert "." not in ln.prompt and "." not in ln.answer
            assert split_line(ln.text) == (ln.prompt, ln.answer)
            assert ln.kind in ("fact", "calc")
    assert max(n_tokens(ln.text) for ln in all_lines) <= 115


def test_lines_of_a_long_seed_stay_short(sim):
    for seed in (9998, 123456789):
        for ln in lines(sim.rollout(seed)):
            assert is_dense(ln.text) and n_tokens(ln.text) <= 128, ln.text
            if ln.kind != "record":
                assert sim.check(ln.prompt, ln.answer).ok, ln.text
    # the corners of the parameter space under a nine-digit seed: the longest line measured has 120 tokens
    longest = 0
    for eff in (0.01, 0.1, 1.0, 4.0):
        for ia in (0.0, 0.001, 0.1):
            for helium in (0.0, 0.248, 0.49):
                r = sim.run(123456789, efficiency=eff, ia_fraction=ia, hydrogen=1 - helium, helium=helium)
                assert sim.conserved(r).ok, (eff, ia, helium)
                for ln in lines(r):
                    assert is_dense(ln.text), ln.text
                    longest = max(longest, n_tokens(ln.text))
    assert longest <= 128
    for seed in range(41, 400):
        assert max(n_tokens(ln.text) for ln in lines(sim.rollout(seed))) <= 128, seed


def test_line_kinds(rollouts):
    r = rollouts[3]
    ls = lines(r)
    n = len(r.steps)
    records = [ln for ln in ls if ln.kind == "record"]
    assert records[0].text == params_line(r).text and records[0].text.startswith("stars seed 3 params. efficiency ")
    assert len(records) == 1 + n
    prompts = {ln.prompt for ln in ls if ln.kind != "record"}
    assert len(prompts) == n * len(STATE_KEYS) + (n - 1) * len(STATE_KEYS) + len(SUMMARY_KEYS)
    assert len(ls) == len({ln.text for ln in ls})
    assert "stars seed 3 step 8 oxygen" in prompts and "stars seed 3 step 8 next stage" in prompts
    assert "stars seed 3 final gas_fraction" in prompts and "stars seed 3 step 2 7 next stage" not in prompts
    for ln in ls:
        if " step " in ln.text:
            assert not any(f" {k}" in ln.text for k in ("neon", "magnesium", "silicon", "other", "hydrogen")), ln.text
    assert len(lines(r, every=5)) < len(ls) / 3


def test_gate_agrees_with_every_line(sim, all_lines):
    for ln in all_lines:
        if ln.kind == "record":
            continue
        assert sim.owns(ln.prompt), ln.prompt
        v = sim.check(ln.prompt, ln.answer)
        assert v.ok and v.expected == ln.answer, (ln.text, v)
        assert not LESSONS.owns(ln.prompt)


def test_gate_agrees_with_every_record_field(sim, all_lines):
    for ln in all_lines:
        if ln.kind != "record":
            continue
        head, *fields = [f.strip() for f in ln.text.rstrip(".").split(".")]
        assert fields
        for f in fields:
            key, _, value = f.partition(" ")
            v = sim.check(f"{head} {key}", value)
            assert v.ok and v.expected == value, (ln.text, key)


def test_gate_rejects_wrong_answers(sim, all_lines):
    rng = random.Random(5)
    questions = [ln for ln in all_lines if ln.kind != "record"]
    for ln in rng.sample(questions, 3000):
        key = ln.prompt.split()[-1]
        if key in WORD_KEYS:
            wrong = next(s for s in STAGES if s != ln.answer)
        else:
            wrong = "9 " + ln.answer  # a changed leading digit: far outside the 5 % tolerance
        v = sim.check(ln.prompt, wrong)
        assert not v.ok and v.expected == ln.answer, (ln.text, wrong, v)
    assert not sim.check("stars seed 3 final stage", "matured").ok
    assert sim.check("stars seed 3 final stage", "many").expected == "mature"
    assert not sim.check("stars seed 3 final supernovae", "many").ok


def test_gate_rejects_a_changed_digit(sim, all_lines):
    """One digit of the answer changed: wrong when the change is more than 5 % (the gate compares
    with the unrounded value, hence 4 and 6)."""
    rng = random.Random(6)
    numeric = [ln for ln in all_lines if ln.kind != "record" and ln.prompt.split()[-1] not in WORD_KEYS]
    rejected = 0
    for ln in rng.sample(numeric, 3000):
        words = ln.answer.split()
        place = rng.choice([i for i, w in enumerate(words) if w.isdigit()])
        words[place] = str((int(words[place]) + rng.randint(1, 9)) % 10)
        wrong = " ".join(words)
        truth, got = parse_num(ln.answer), parse_num(wrong)
        assert got != truth
        v = sim.check(ln.prompt, wrong)
        assert v.expected == ln.answer
        off = abs(got - truth) / max(abs(got), abs(truth))
        if off > 0.06:
            assert not v.ok, (ln.text, wrong, v)
        elif off < 0.04:
            assert v.ok, (ln.text, wrong, v)
        rejected += not v.ok
    assert rejected > 1500


def test_gate_rejects_answers_that_are_not_finite_numbers(sim):
    for prompt in ("stars seed 3 final metallicity", "stars seed 3 step 1 4 oxygen", "stars seed 3 final supernovae",
                   "stars seed 3 params efficiency"):
        for answer in ("1 e 9 9 9", "minus 1 e 9 9 9", "", "point", "e", "yes", "0 point 0 2 solar", "0 . 0 2",
                       "minus 0 point 0 2", "1 e 3 0 0"):
            v = sim.check(prompt, answer)
            assert not v.ok and v.expected is not None, (prompt, answer, v)
    assert sim.check("stars seed 3 final metallicity", "1 e 9 9 9").reason == "not a number"
    assert sim.check("stars seed 3 step 0 metallicity", "0").ok
    assert not sim.check("stars seed 3 step 0 metallicity", "1 e minus 9").ok
    assert not sim.check("stars seed 3 step 0 metallicity", "1").ok


def test_tolerance_is_five_percent_for_numbers_and_exact_for_words(sim, rollouts):
    z = rollouts[3].summary["metallicity"]
    assert sim.check("stars seed 3 final metallicity", dense_value(z * 1.03)).ok
    assert sim.check("stars seed 3 final metallicity", dense_value(z * 0.97)).ok
    assert not sim.check("stars seed 3 final metallicity", dense_value(z * 1.1)).ok
    assert not sim.check("stars seed 3 final metallicity", dense_value(z * 0.9)).ok
    assert sim.check("stars seed 3 step 8 stage", "enriching").ok
    assert not sim.check("stars seed 3 step 8 stage", "mature").ok
    for t in (5, 14, 27):
        for key in ("gas", "oxygen", "supernovae_cc", "neon", "other"):
            truth = rollouts[3].steps[t][key]
            assert close(parse_num(dense_value(truth)), truth, rel=0.05)
            assert sim.check(f"stars seed 3 step {' '.join(str(t))} {key}", dense_value(truth)).ok


@pytest.mark.parametrize("prompt", [
    "turkey capital", "q spoon color", "carbon protons", "water molar_mass", "calc 1 2 plus 7",
    "gravity seed 7 step 2 0 clumps", "gravity seed 7 final clumps", "nucleo seed 3 final helium",
    "nucleo seed 3 step 4 stage", "world seed 3 era planets", "chem seed 7 final water",
    "life seed 7 step 1 2 next population", "planets seed 7 final gas", "planets seed 7 step 3 gas",
    "cells seed 7 step 3 population", "cells seed 7 final stage", "bodies seed 7 step 3 size",
    "bodies seed 7 final stage", "senses seed 7 step 3 sensors", "signals seed 7 step 1 2 success",
    "signals seed 7 say predator near", "signals seed 7 meaning ki ta", "society seed 7 final stage",
    "society predict hamilton relatedness 0 point 5 benefit 4 cost 1", "world7 seed 3 era stars",
    "world7 seed 3 final stage", "signals predict expected_success signals 4 states 4 shared 3",
    "stars predict lifetime mass 2", "stars predict fate mass 1 2 point 5",
    "stars", "stars seed 7", "stars seed 7 step 3", "stars seed 7 step 3 clumps", "stars seed 7 step 4 6 gas",
    "stars seed 7 step 4 5 next gas", "stars seed 7 step 5 5 gas", "stars seed 7 final planets",
    "stars seed x final gas_fraction", "stars seed 7 final", "stars seed 7 params time", "stars seed 12 step 3 gas",
    "stars seed 7 step 3 next", "stars seed 7 step 3 gas stars", "", "stars seed 0 7 final metallicity",
    "stars seed 7 step 0 3 gas", "stars seed 7 step 0 0 gas", "stars seed minus 7 final metallicity",
    "stars seed 7 point 5 final metallicity", "q stars seed 7 final metallicity", "stars seed 7 final metallicity extra",
    "stars seed 7 step 3 next next gas", "stars seed 7 era stars", "stars  seed 7 final metallicity",
    "stars seed 7 final metallicity ", " stars seed 7 final metallicity", "stars seed 7 final gas",
    "stars seed 7 step 3 gas_fraction", "stars seed 7 params metallicity", "stars seed 7 step 3 closed_box",
])
def test_does_not_own_other_prompts(sim, prompt):
    assert not sim.owns(prompt)
    assert sim.check(prompt, "1").ok is False
    assert sim.check(prompt, "1").expected is None and sim.check(prompt, "1").reason == "not my question"


@pytest.mark.parametrize("prompt", [
    "stars seed 7 step 0 gas", "stars seed 7 step 4 5 stage", "stars seed 7 step 4 4 next alpha_over_iron",
    "stars seed 1 2 3 4 step 3 next oxygen", "stars seed 7 final supernovae_ia", "stars seed 7 final closed_box",
    "stars seed 7 params metal_yield", "stars seed 7 step 4 neon", "stars seed 7 step 4 next other",
    "stars seed 0 final metallicity", "stars seed 7 step 1 0 supernovae_cc", "stars seed 7 final other",
])
def test_owns_its_prompts(sim, prompt):
    assert sim.owns(prompt)
    assert sim.check(prompt, "nonsense words").expected is not None
    assert not LESSONS.owns(prompt)


def test_numbers_use_three_significant_digits(all_lines):
    assert dense_value(1052.53) == "1 0 5 0" and dense_value(0.03234) == "0 point 0 3 2 3"
    assert dense_value(0.0) == "0" and dense_value("mature") == "mature" and dense_value(7) == "7"
    assert dense_value(0.000412) == "4 point 1 2 e minus 4" and dense_value(40.84) == "4 0 point 8"
    assert dense_value(999.7) == "1 0 0 0" and dense_value(0.9996) == "1" and dense_value(7421.3) == "7 4 2 0"
    for ln in all_lines:
        if ln.kind == "record" or ln.prompt.split()[-1] in WORD_KEYS:
            continue
        value = parse_num(ln.answer)
        assert value is not None and value >= 0, ln.text
        digits = "".join(w for w in ln.answer.split(" e ")[0].split() if w.isdigit())
        assert len(digits.strip("0")) <= 3, ln.text
        assert len(ln.answer.split()) <= 8, ln.text


# --------------------------------------------------------------------------- the gate of conservation

def test_conserved_on_forty_seeds_with_random_params(sim, rollouts):
    assert len(rollouts) == 40
    stages = set()
    for seed, r in rollouts.items():
        assert r.params == {**random_params(random.Random(seed)), "metal_yield": r.params["metal_yield"],
                            "return_fraction": r.params["return_fraction"]}
        v = sim.conserved(r)
        assert v.ok, (seed, v)
        stages |= {s["stage"] for s in r.steps}
    assert stages == set(STAGES)


def test_mass_and_every_element_are_accounted_for(rollouts):
    """The same sums as the gate, written out once more, with a tighter bound."""
    worst = 0.0
    for r in rollouts.values():
        start = r.ledger[0]["gas"]
        assert start["hydrogen"] == pytest.approx(r.params["hydrogen"]) and start["helium"] == pytest.approx(r.params["helium"])
        assert all(start[e] == 0 for e in METALS)
        for prev, s, led in zip([None] + r.steps[:-1], r.steps, r.ledger):
            worst = max(worst, abs(s["gas"] + s["stars"] + s["remnants"] - 1.0))
            assert sum(led["gas"].values()) == pytest.approx(s["gas"], abs=1e-12)
            for e in ELEMENTS:
                assert led["gas"][e] >= 0 and led["astrated"][e] >= 0 and led["returned"][e] >= 0
                assert start[e] - led["astrated"][e] + led["returned"][e] == pytest.approx(led["gas"][e], abs=1e-12)
                assert s[e] == pytest.approx(led["gas"][e] / s["gas"], rel=1e-12)
            assert sum(s[e] for e in ELEMENTS) == pytest.approx(1.0, abs=1e-12), "the mass fractions sum to one"
            assert s["metallicity"] == pytest.approx(sum(s[e] for e in METALS), rel=1e-12)
            assert s["stars"] == pytest.approx(led["formed"] - led["died"], abs=1e-12)
            assert s["remnants"] == pytest.approx(led["died"] - sum(led["returned"].values()), abs=1e-12)
            assert min(s["gas"], s["stars"], s["remnants"]) >= 0
            assert led["returned"]["hydrogen"] <= led["astrated"]["hydrogen"], "no star makes hydrogen"
            if prev is not None:
                assert s["time"] == prev["time"] + DT
                assert s["supernovae_cc"] >= prev["supernovae_cc"] and s["supernovae_ia"] >= prev["supernovae_ia"]
                assert s["hydrogen"] < prev["hydrogen"], "hydrogen only goes down"
                assert s["metallicity"] > prev["metallicity"] and s["gas"] < prev["gas"] and s["remnants"] > prev["remnants"]
                assert s["helium"] > prev["helium"] or s["gas"] < 0.012, "type Ia metals crowd helium out of the last gas"
        assert [s["time"] for s in r.steps] == [k * DT for k in range(len(r.steps))]
        assert r.steps[-1]["time"] == r.params["t_end"]
    assert worst < 1e-12


def test_conserved_catches_tampering(sim, rollouts):
    base = rollouts[3]

    def broken(change) -> bool:
        r = copy.deepcopy(base)
        change(r)
        return not sim.conserved(r).ok

    assert not broken(lambda r: None)
    assert broken(lambda r: r.steps[10].update(gas=r.steps[10]["gas"] + 1e-6))
    assert broken(lambda r: r.steps[10].update(stars=r.steps[10]["stars"] * 0.99))
    assert broken(lambda r: r.steps[10].update(remnants=r.steps[10]["remnants"] + 1e-6))
    assert broken(lambda r: r.steps[27].update(oxygen=r.steps[27]["oxygen"] * 1.01))
    assert broken(lambda r: r.steps[27].update(metallicity=r.steps[27]["metallicity"] * 1.01))
    assert broken(lambda r: r.steps[27].update(helium=r.steps[27]["helium"] + 1e-6))
    assert broken(lambda r: r.steps[12].update(time=r.steps[11]["time"]))
    assert broken(lambda r: r.steps[12].update(supernovae_cc=r.steps[12]["supernovae_cc"] + 1))
    assert broken(lambda r: r.steps[12].update(supernovae_ia=r.steps[12]["supernovae_ia"] + 1))
    assert broken(lambda r: r.steps[12].update(alpha_over_iron=1.0))
    assert broken(lambda r: r.steps[5].update(stage="mature"))
    assert broken(lambda r: r.steps[5].update(stage="finished"))
    assert broken(lambda r: r.ledger[20]["gas"].update(iron=r.ledger[20]["gas"]["iron"] + 1e-6))
    assert broken(lambda r: r.ledger[20]["returned"].update(oxygen=r.ledger[20]["returned"]["oxygen"] + 1e-6))
    assert broken(lambda r: r.ledger[20]["astrated"].update(hydrogen=r.ledger[20]["astrated"]["hydrogen"] - 1e-6))
    assert broken(lambda r: r.ledger[20].update(formed=r.ledger[20]["formed"] + 1e-6))
    assert broken(lambda r: r.ledger[20].update(died=r.ledger[20]["died"] + 1e-6))
    assert broken(lambda r: r.ledger[20]["gas"].pop("other"))
    assert broken(lambda r: r.ledger.pop())
    assert broken(lambda r: setattr(r, "ledger", []))
    assert broken(lambda r: r.steps[0].update(remnants=1e-12, gas=1 - 1e-12))

    # an element turned into another in the gas: the total still adds up, the element does not
    def transmute(r):
        r.ledger[20]["gas"]["oxygen"] -= 1e-5
        r.ledger[20]["gas"]["iron"] += 1e-5
    assert broken(transmute)

    # hydrogen made in a star: more comes back than went in
    def make_hydrogen(r):
        for led in r.ledger[1:]:
            extra = led["astrated"]["hydrogen"] - led["returned"]["hydrogen"] + 1e-6
            led["returned"]["hydrogen"] += extra
            led["gas"]["hydrogen"] += extra
    assert broken(make_hydrogen)
    plain = copy.deepcopy(base)
    del plain.ledger
    assert not sim.conserved(plain).ok


def test_summary_matches_the_last_step(rollouts):
    for r in rollouts.values():
        last, led = r.steps[-1], r.ledger[-1]
        assert list(r.summary) == list(SUMMARY_KEYS) and list(r.params) == list(PARAM_KEYS)
        assert set(last) == set(STATE_KEYS) | set(EXTRA_KEYS)
        for k in ("metallicity", "alpha_over_iron", "stage", "time") + ELEMENTS:
            assert r.summary[k] == last[k], k
        assert r.summary["gas_fraction"] == last["gas"] and r.summary["stars_formed"] == led["formed"]
        assert r.summary["supernovae"] == last["supernovae_cc"] + last["supernovae_ia"]
        assert r.summary["supernovae_cc"] == last["supernovae_cc"] and r.summary["supernovae_ia"] == last["supernovae_ia"]
        assert r.summary["closed_box"] == S.closed_box(r.params["metal_yield"], r.summary["gas_fraction"])
        assert r.summary["time"] == r.params["t_end"]
        gen = S.generation(r.params["ia_fraction"])
        assert r.params["metal_yield"] == gen["metal_yield"] and r.params["return_fraction"] == gen["return_fraction"]


# --------------------------------------------------------------------------- plausibility targets

def test_solar_metallicity_between_eight_and_ten_gyr(sim, default_run):
    """Efficiency 0.3 per Gyr: the gas holds about the sun's share of metals when the sun formed."""
    for step in default_run.steps:
        if 8.0 <= step["time"] <= 10.0:
            assert 0.010 <= step["metallicity"] <= 0.020, step
    at = {s["time"]: s for s in default_run.steps}
    assert at[9.0]["metallicity"] == pytest.approx(0.01355, rel=0.01)
    assert at[8.0]["metallicity"] == pytest.approx(0.01198, rel=0.01) and at[10.0]["metallicity"] == pytest.approx(0.01513, rel=0.01)
    assert at[9.0]["gas"] == pytest.approx(0.1498, rel=0.01)
    assert [s["metallicity"] for s in default_run.steps] == sorted(s["metallicity"] for s in default_run.steps)
    for eff in (0.25, 0.3, 0.35):
        r = sim.run(1, efficiency=eff, t_end=9.0)
        assert 0.010 <= r.summary["metallicity"] <= 0.020, eff
    # the mix is about the sun's (Asplund et al. 2009, from memory), each element within 15 percent
    sun = {"oxygen": 5.73e-3, "carbon": 2.36e-3, "iron": 1.29e-3, "neon": 1.26e-3, "nitrogen": 6.9e-4,
           "magnesium": 7.1e-4, "silicon": 6.7e-4}
    for element, share in sun.items():
        assert at[9.0][element] == pytest.approx(share, rel=0.15), element
    assert 0.268 <= at[9.0]["helium"] <= 0.276 and at[0.0]["helium"] == 0.248, "the sun was born with about 0.27"
    assert at[9.0]["alpha_over_iron"] == pytest.approx(1.0, abs=0.05)


def test_oxygen_is_the_most_abundant_metal(sim, default_run, rollouts):
    assert all(top_metal(s) == "oxygen" for s in default_run.steps[1:])
    for r in rollouts.values():
        for s in r.steps[1:]:
            assert top_metal(s) == "oxygen" or s["stage"] == "exhausted", (r.seed, s)
    exceptions = 0
    for seed in range(41, 400):
        r = sim.rollout(seed)
        for s in r.steps[1:]:
            if top_metal(s) != "oxygen":
                exceptions += 1
                assert s["stage"] == "exhausted" and s["gas"] < 0.014 and top_metal(s) == "iron", (seed, s)
                assert sim.rollout(seed).params["ia_fraction"] > 0.004 and sim.rollout(seed).params["efficiency"] > 0.7
    assert 0 < exceptions < 60


def test_oxygen_over_iron_falls_with_time(sim, default_run, rollouts):
    """Type Ia iron arrives late: a plateau of 2.5 times solar, then a steady fall."""
    a = [s["alpha_over_iron"] for s in default_run.steps]
    assert a[0] == 0 and all(x == pytest.approx(2.5, rel=1e-12) for x in a[1:5])
    assert all(later < earlier for earlier, later in zip(a[4:], a[5:])), "falls at every step from 1.25 Gyr on"
    assert a[36] == pytest.approx(1.001, rel=0.01) and a[54] == pytest.approx(0.702, rel=0.01)
    assert default_run.steps[4]["supernovae_ia"] == 0 < default_run.steps[5]["supernovae_ia"]
    for r in list(rollouts.values()) + [sim.rollout(seed) for seed in range(41, 400)]:
        for prev, s in zip(r.steps[1:], r.steps[2:]):
            assert s["alpha_over_iron"] <= prev["alpha_over_iron"] * (1 + 1e-9), (r.seed, s)
        if r.params["t_end"] >= 1.25:
            assert r.steps[-1]["alpha_over_iron"] < 0.999 * r.steps[1]["alpha_over_iron"], r.seed
    # without type Ia supernovae it stays where the massive stars put it; more of them, more iron
    none = sim.run(1, ia_fraction=0.0)
    assert none.summary["supernovae_ia"] == 0 and none.summary["alpha_over_iron"] == pytest.approx(2.5, rel=1e-9)
    few, many = sim.run(1, ia_fraction=0.002), sim.run(1, ia_fraction=0.008)
    assert none.summary["iron"] < few.summary["iron"] < many.summary["iron"]
    assert few.summary["alpha_over_iron"] > many.summary["alpha_over_iron"]
    assert few.summary["oxygen"] == pytest.approx(many.summary["oxygen"], rel=0.1), "oxygen hardly notices"


def test_zero_end_time_gives_zero_metals(sim):
    r = sim.run(1, t_end=0.0)
    assert len(r.steps) == 1 and sim.conserved(r).ok
    assert r.summary["metallicity"] == 0 and all(r.summary[e] == 0 for e in METALS)
    assert r.summary["hydrogen"] == 0.752 and r.summary["helium"] == 0.248 and r.summary["gas_fraction"] == 1
    assert r.summary["supernovae"] == 0 and r.summary["stars_formed"] == 0 and r.summary["time"] == 0
    assert r.summary["stage"] == "first_stars" and r.summary["closed_box"] == 0 and r.summary["alpha_over_iron"] == 0
    texts = [ln.text for ln in lines(r)]
    assert "q stars seed 1 final metallicity. a 0." in texts and len(texts) == 1 + 1 + len(STATE_KEYS) + len(SUMMARY_KEYS)
    for r in (sim.rollout(seed) for seed in SEEDS):
        assert r.steps[0]["metallicity"] == 0 and r.steps[0]["gas"] == 1 and r.steps[0]["stage"] == "first_stars"
        assert r.steps[1]["metallicity"] > 0, "one step is enough: massive stars die within it"


def test_the_stages_follow_each_other(sim, default_run):
    changes = [(s["time"], s["stage"]) for prev, s in zip(default_run.steps, default_run.steps[1:])
               if prev["stage"] != s["stage"]]
    assert changes == [(1.25, "enriching"), (9.0, "mature")]
    fast = sim.run(1, efficiency=1.0)
    assert [s["stage"] for s in fast.steps][-1] == "exhausted" and fast.summary["gas_fraction"] < 0.05
    slow = sim.run(1, efficiency=0.1)
    assert slow.summary["stage"] == "enriching" and slow.summary["metallicity"] < S.Z_SUN
    order = {"first_stars": 0, "enriching": 1, "mature": 2}
    for eff in (0.1, 0.3, 0.6, 1.0):
        seen = [order[s["stage"]] for s in sim.run(1, efficiency=eff).steps if s["stage"] != "exhausted"]
        assert seen == sorted(seen)


def test_the_faster_the_stars_form_the_sooner_the_gas_is_gone(sim):
    runs = [sim.run(1, efficiency=eff, t_end=6.0) for eff in (0.1, 0.2, 0.4, 0.8)]
    gas = [r.summary["gas_fraction"] for r in runs]
    assert gas == sorted(gas, reverse=True) and gas[0] > 0.5 and gas[-1] < 0.05
    metals = [r.summary["metallicity"] for r in runs]
    assert metals == sorted(metals), "less gas left, more metals in it"
    for r in runs:
        gas_steps = [s["gas"] for s in r.steps]
        assert all(b < a for a, b in zip(gas_steps, gas_steps[1:]))
        assert all(b["remnants"] > a["remnants"] for a, b in zip(r.steps, r.steps[1:]))
        assert r.summary["helium"] > 0.248 and r.summary["hydrogen"] < 0.752


def test_supernova_counts(sim, default_run):
    """6.03 core-collapse supernovae per 1000 solar masses formed (stars of 8 to 25), and they
    all go off in the step that formed them; the first type Ia 1.25 Gyr after the start."""
    for s, led in zip(default_run.steps, default_run.ledger):
        assert s["supernovae_cc"] == pytest.approx(led["formed"] * S.BOX_MASS * 6.027e-3, rel=1e-3)
    gen = S.generation(0.004)
    assert gen["supernovae_cc"] * 1000 == pytest.approx(6.027, rel=1e-3)
    assert gen["supernovae_ia"] * 1000 == pytest.approx(0.567, rel=1e-2)
    assert gen["return_fraction"] == pytest.approx(0.290, rel=1e-2) and gen["metal_yield"] == pytest.approx(0.00787, rel=1e-2)
    at = {s["time"]: s for s in default_run.steps}
    assert at[9.0]["supernovae_ia"] / at[9.0]["supernovae_cc"] == pytest.approx(0.05, abs=0.01)
    assert at[9.0]["supernovae_cc"] == pytest.approx(6885, rel=1e-3)


def test_the_simulation_lags_the_closed_box(sim, default_run):
    """With instant returns Z = y ln(1 / gas). The simulation's returns take time, so it stays
    below the formula and closes in on it."""
    y = default_run.params["metal_yield"]
    ratio = {s["time"]: S.closed_box(y, s["gas"]) / s["metallicity"] for s in default_run.steps[1:]}
    assert ratio[1.0] == pytest.approx(1.44, abs=0.02) and ratio[9.0] == pytest.approx(1.10, abs=0.02)
    assert ratio[13.5] == pytest.approx(1.02, abs=0.02)
    assert all(1.0 < ratio[t] for t in ratio)
    series = [ratio[k * DT] for k in range(1, 55)]
    assert all(b < a for a, b in zip(series, series[1:])), "and closes in on it step by step"
    assert default_run.summary["closed_box"] == pytest.approx(0.02088, rel=0.01)


# --------------------------------------------------------------------------- the rules

def test_the_rules_by_hand():
    assert S.luminosity(1.0) == 1.0 and S.luminosity(2.0) == pytest.approx(11.3137, rel=1e-5)
    assert S.lifetime(1.0) == 10.0 and S.lifetime(2.0) == pytest.approx(10 * 2 ** -2.5)
    assert S.lifetime(4.0) == pytest.approx(0.3125) and S.lifetime(0.5) == pytest.approx(56.57, rel=1e-3)
    assert S.death_time(3.0, 1.0) == 13.0 and S.death_time(0.25, 4.0) == pytest.approx(0.5625)
    assert S.turnoff_mass(10.0) == 1.0 and S.turnoff_mass(0.25) == pytest.approx(4.373, rel=1e-3)
    assert S.turnoff_mass(13.5) == pytest.approx(0.887, rel=1e-3)
    assert all(S.lifetime(S.turnoff_mass(age)) == pytest.approx(age, rel=1e-12) for age in (0.25, 1.0, 4.5, 13.5))
    assert [S.shines(m) for m in (0.01, 0.0799, 0.08, 0.3)] == ["no", "no", "yes", "yes"]
    assert [S.fate(m) for m in (0.05, 0.08, 1.0, 7.99, 8.0, 25.0, 25.01, 100.0)] == [
        "brown_dwarf", "white_dwarf", "white_dwarf", "white_dwarf", "neutron_star", "neutron_star", "black_hole",
        "black_hole"]
    assert S.remnant_mass(0.05) == 0.05 and S.remnant_mass(0.3) == 0.3, "a light star keeps all of itself"
    assert S.remnant_mass(1.0) == pytest.approx(0.5) and S.remnant_mass(7.0) == pytest.approx(1.16)
    assert S.remnant_mass(8.0) == S.remnant_mass(25.0) == 1.4 and S.remnant_mass(30.0) == 15.0
    assert S.returned_mass(1.0) == pytest.approx(0.5) and S.returned_mass(20.0) == pytest.approx(18.6)
    assert S.returned_mass(0.05) == 0 and S.returned_mass(40.0) == 20.0
    assert S.gas_to_stars(1.0, 0.3, 0.25) == pytest.approx(1 - math.exp(-0.075))
    assert S.gas_to_stars(0.0, 0.3, 0.25) == 0 and S.gas_to_stars(2.0, 0.3, 0.25) == 2 * S.gas_to_stars(1.0, 0.3, 0.25)
    assert S.stars_above(1000, 8) == pytest.approx(7.42, rel=1e-3)
    assert S.stars_above(1000, 8) - S.stars_above(1000, 25) == pytest.approx(6.03, rel=1e-3)
    assert S.stars_above(1000, 100) == 0 and S.stars_above(1, 0.1) == pytest.approx(S.imf_number(0.1, 100))
    assert S.ia_iron(0) == 0 and S.ia_iron(3) == pytest.approx(2.1) and S.ia_iron(1) == 0.7
    assert S.closed_box(0.01, 1.0) == 0 and S.closed_box(0.01, math.exp(-1)) == pytest.approx(0.01)
    assert S.closed_box(0.01, 0.2) == pytest.approx(0.016094, rel=1e-4)
    assert S.alpha_over_iron(0.0044, 0.001) == pytest.approx(1.0) and S.alpha_over_iron(0.005, 0.0) == 0
    assert [S.stage(g, z) for g, z in ((1, 0), (0.9, 0.00133), (0.9, 0.00134), (0.5, 0.0133), (0.5, 0.0134),
                                       (0.05, 0.02), (0.049, 0.02), (0.01, 0.0))] == [
        "first_stars", "first_stars", "enriching", "enriching", "mature", "mature", "exhausted", "exhausted"]


def test_the_mass_function_and_its_bins():
    assert S.imf_mass(0.1, 100) == pytest.approx(1.0, abs=1e-12), "one solar mass per solar mass formed"
    assert sum(b.share for b in BINS) == pytest.approx(1.0, abs=1e-12)
    assert sum(b.number for b in BINS) == pytest.approx(S.imf_number(0.1, 100), rel=1e-12)
    assert S.imf_mass(8, 100) == pytest.approx(0.139, rel=1e-2), "14 percent of the mass in stars above 8"
    assert S.imf_number(0.05, 200) == S.imf_number(0.1, 100), "nothing outside 0.1 to 100"
    assert len(S.EDGES) == 31 and 8.0 in S.EDGES and 25.0 in S.EDGES
    edges = S.bin_edges()
    assert len(BINS) == 83 == len(edges) - 1 and BINS[0].lo == 0.1 and BINS[-1].hi == 100
    assert list(edges) == sorted(set(edges)) and set(S.EDGES) <= set(edges)
    cuts = sorted(set(edges) - set(S.EDGES))
    assert len(cuts) == 53, "one cut per step of lifetime, 13.5 Gyr down to 0.25; the cut for 10 Gyr is the edge 1.0"
    assert cuts[0] == S.turnoff_mass(13.5) and cuts[-1] == S.turnoff_mass(0.25) and S.turnoff_mass(10.0) in S.EDGES
    assert all(abs(S.lifetime(m) / DT - round(S.lifetime(m) / DT)) < 1e-9 for m in cuts)
    # the Salpeter slope: ten times the mass, 10^-1.35 times the stars
    assert S.imf_number(10, 12.5) / S.imf_number(1, 1.25) == pytest.approx(10 ** -1.35, rel=1e-9)
    for b, nxt in zip(BINS, BINS[1:] + (None,)):
        assert b.lo < b.mass < b.hi and (nxt is None or b.hi == nxt.lo)
        assert b.mass == pytest.approx(b.share / b.number)
        # every bin is what the lesson functions say of its mean mass
        assert b.fate == S.fate(b.mass) and b.lifetime == S.lifetime(b.mass)
        assert b.returned == S.returned_mass(b.mass) / b.mass and 0 <= b.returned < 1
        assert b.number == S.stars_above(1, b.lo) - S.stars_above(1, b.hi) or b.number == pytest.approx(
            S.stars_above(1, b.lo) - S.stars_above(1, b.hi), rel=1e-12)
    assert [b.fate for b in BINS] == ["white_dwarf"] * 72 + ["neutron_star"] * 5 + ["black_hole"] * 6
    assert all(b.lifetime < DT for b in BINS if b.fate != "white_dwarf"), "massive stars die within one step"
    assert [b.lo for b in BINS if b.lifetime <= S.T_MAX][0] == S.turnoff_mass(13.5), "lighter stars outlive the run"
    assert sum(b.lifetime <= S.T_MAX for b in BINS) == 73


def test_the_yield_tables():
    for fate, table in S.YIELDS.items():
        assert fate in S.FATES and set(table) <= set(ELEMENTS) - {"hydrogen"}
        assert all(v > 0 for v in table.values()) and sum(table.values()) < 0.15
    assert S.YIELDS["brown_dwarf"] == {}
    assert set(S.YIELDS["white_dwarf"]) == {"helium", "carbon", "nitrogen"}
    assert {"oxygen", "neon", "magnesium", "silicon", "iron"} <= set(S.YIELDS["neutron_star"])
    assert not {"oxygen", "iron"} & set(S.YIELDS["black_hole"])
    assert max(S.YIELDS["neutron_star"], key=lambda e: S.YIELDS["neutron_star"][e] * (e != "helium")) == "oxygen"
    assert sum(S.IA_YIELD.values()) == pytest.approx(S.IA_MASS) and S.IA_YIELD["iron"] == S.IA_IRON == 0.7
    assert "hydrogen" not in S.IA_YIELD and "helium" not in S.IA_YIELD


def test_a_star_cannot_make_more_than_its_hydrogen_allows(sim):
    """Gas that is almost all helium: the yields are scaled down, hydrogen never goes negative."""
    r = sim.run(1, hydrogen=0.02, helium=0.98, efficiency=1.0)
    assert sim.conserved(r).ok
    assert all(led["gas"]["hydrogen"] >= 0 for led in r.ledger) and r.summary["hydrogen"] < 0.02
    assert r.summary["metallicity"] < sim.run(1, efficiency=1.0).summary["metallicity"]


# --------------------------------------------------------------------------- lessons

def test_every_lesson_is_accepted_and_is_what_the_simulation_computes(sim, lessons):
    assert len(lessons) == 400 and set(FUNCTIONS) == set(RULES)
    assert {ln.prompt.split()[2] for ln in lessons} == set(RULES), "every rule is drawn"
    for ln in lessons:
        assert is_dense(ln.text) and n_tokens(ln.text) <= MAX_TOKENS, ln.text
        assert "." not in ln.prompt and "." not in ln.answer and ln.topic == "predict_stars" and ln.kind == "calc"
        assert LESSONS.owns(ln.prompt) and not sim.owns(ln.prompt)
        v = LESSONS.check(ln.prompt, ln.answer)
        assert v.ok and v.expected == ln.answer, (ln.text, v)
        name, values = LESSONS.parse(ln.prompt)
        truth = getattr(S, FUNCTIONS[name])(*values)  # the function the simulation itself calls
        assert ln.answer == (truth if isinstance(truth, str) else num(truth)), ln.text
        assert ln.meta["split_key"] == LESSONS.split_key(ln.prompt)
    assert max(n_tokens(ln.text) for ln in lessons) <= 40


def test_lessons_reject_altered_answers(lessons):
    words = {"shines": ("yes", "no"), "fate": S.FATES, "stage": STAGES}
    for ln in lessons:
        name = ln.prompt.split()[2]
        if name in words:
            wrong = next(w for w in words[name] if w != ln.answer)
        else:
            wrong = "9 " + ln.answer
        v = LESSONS.check(ln.prompt, wrong)
        assert not v.ok and v.expected == ln.answer, (ln.text, wrong)
        assert not LESSONS.check(ln.prompt, "1 e 9 9 9").ok and not LESSONS.check(ln.prompt, "").ok
    assert not LESSONS.check("stars seed 3 final metallicity", "0 point 0 2 1 5").ok
    assert LESSONS.check("cells predict lifetime mass 2", "1 point 7 6 8").reason == "not my question"
    assert not LESSONS.owns("stars predict lifetime mass 2 0 0"), "outside the range of the rule"
    assert not LESSONS.owns("stars predict lifetime mass") and not LESSONS.owns("stars predict fate")


def test_the_lessons_of_the_docstring():
    for prompt, answer in (("stars predict lifetime mass 2", "1 point 7 6 8"),
                           ("stars predict fate mass 1 2 point 5", "neutron_star"),
                           ("stars predict closed_box metal_yield 0 point 0 1 gas_fraction 0 point 2", "0 point 0 1 6 0 9"),
                           ("stars predict luminosity mass 2", "1 1 point 3 1"),
                           ("stars predict death_time born 3 mass 1", "1 3"),
                           ("stars predict turnoff age 1 0", "1"),
                           ("stars predict turnoff age 0 point 2 5", "4 point 3 7 3"),
                           ("stars predict shines mass 0 point 0 7 9", "no"),
                           ("stars predict shines mass 0 point 0 8", "yes"),
                           ("stars predict fate mass 0 point 0 5", "brown_dwarf"),
                           ("stars predict fate mass 8", "neutron_star"),
                           ("stars predict fate mass 2 5 point 0 1", "black_hole"),
                           ("stars predict remnant mass 1", "0 point 5"),
                           ("stars predict remnant mass 3 0", "1 5"),
                           ("stars predict returned mass 2 0", "1 8 point 6"),
                           ("stars predict gas_to_stars gas 1 efficiency 0 point 3 dt 0 point 2 5", "0 point 0 7 2 2 6"),
                           ("stars predict stars_above formed 1 0 0 0 mass 8", "7 point 4 2 2"),
                           ("stars predict ia_iron events 3", "2 point 1"),
                           ("stars predict alpha_over_iron oxygen 0 point 0 0 4 4 iron 0 point 0 0 1", "1"),
                           ("stars predict alpha_over_iron oxygen 0 point 0 0 4 4 iron 0", "0"),
                           ("stars predict stage gas 0 point 0 4 metallicity 0 point 0 2", "exhausted"),
                           ("stars predict stage gas 0 point 5 metallicity 0 point 0 2", "mature")):
        v = LESSONS.check(prompt, answer)
        assert v.ok and v.expected == answer, (prompt, v)


def test_lesson_records_are_dense():
    records = LESSONS.records()
    assert len(records) == len(RULES) == 14
    for ln in records:
        assert ln.kind == "record" and is_dense(ln.text) and n_tokens(ln.text) <= MAX_TOKENS, ln.text
        assert ln.text.startswith("stars predict ") and ln.topic == "predict_stars"


def test_the_lesson_functions_are_the_ones_the_simulation_calls(sim, monkeypatch):
    """Each rule's truth is the module's own function, and replacing that function changes the
    simulation: the lesson and the rollout cannot drift apart."""
    for name, function in FUNCTIONS.items():
        assert RULES[name].compute is getattr(S, function), name
    base = sim.run(1, **DEFAULTS)

    def changed(function: str, replacement) -> bool:
        with monkeypatch.context() as m:
            m.setattr(S, function, replacement)
            r = sim.run(1, **DEFAULTS)
        return r.steps != base.steps or r.summary != base.summary

    originals = {f: getattr(S, f) for f in FUNCTIONS.values()}
    assert changed("gas_to_stars", lambda gas, efficiency, dt: 0.5 * originals["gas_to_stars"](gas, efficiency, dt))
    assert changed("death_time", lambda born, mass: originals["death_time"](born, mass) + 1.0)
    assert changed("stars_above", lambda formed, mass: 2 * originals["stars_above"](formed, mass))
    assert changed("ia_iron", lambda events: 0.5 * originals["ia_iron"](events))
    assert changed("alpha_over_iron", lambda oxygen, iron: 2 * originals["alpha_over_iron"](oxygen, iron))
    assert changed("stage", lambda gas, metallicity: "mature")
    assert changed("closed_box", lambda metal_yield, gas_fraction: 0.0)
    assert changed("lifetime", lambda mass: 2 * originals["lifetime"](mass)), "death_time calls lifetime"
    assert changed("luminosity", lambda mass: 2 * originals["luminosity"](mass)), "lifetime calls luminosity"
    assert sim.run(1, **DEFAULTS).steps == base.steps, "and back"
    # the mass bins are built once, from the same functions
    with monkeypatch.context() as m:
        m.setattr(S, "shines", lambda mass: "no")
        assert {b.fate for b in S._bins()} == {"brown_dwarf"} and all(b.returned == 0 for b in S._bins())
    with monkeypatch.context() as m:
        m.setattr(S, "remnant_mass", lambda mass: mass / 4)
        assert all(b.returned == pytest.approx(0.75) for b in S._bins())
    with monkeypatch.context() as m:
        m.setattr(S, "returned_mass", lambda mass: 0.0)
        assert all(b.returned == 0 for b in S._bins())
    with monkeypatch.context() as m:
        m.setattr(S, "fate", lambda mass: "white_dwarf")
        assert {b.fate for b in S._bins()} == {"white_dwarf"}
    with monkeypatch.context() as m:
        m.setattr(S, "turnoff_mass", lambda age: 45.0)
        assert S.bin_edges() == tuple(sorted(S.EDGES + (45.0,))) and len(S._bins()) == 31
    assert S._bins() == BINS


def test_the_lessons_are_true_of_the_rollouts(rollouts):
    """What a lesson says, step by step, with the rollout's own numbers as inputs."""
    for r in rollouts.values():
        eff = r.params["efficiency"]
        for prev, s, prev_led, led in zip(r.steps, r.steps[1:], r.ledger, r.ledger[1:]):
            assert led["formed"] - prev_led["formed"] == pytest.approx(S.gas_to_stars(prev["gas"], eff, DT), abs=1e-12)
        for s, led in zip(r.steps, r.ledger):
            massive = S.stars_above(led["formed"] * S.BOX_MASS, 8.0) - S.stars_above(led["formed"] * S.BOX_MASS, 25.0)
            assert s["supernovae_cc"] == pytest.approx(massive, rel=1e-9, abs=1e-9)
            assert s["stage"] == S.stage(s["gas"], s["metallicity"])
            assert s["alpha_over_iron"] == S.alpha_over_iron(s["oxygen"], s["iron"])
            assert led["ia_mass"] * S.BOX_MASS == pytest.approx(S.IA_MASS * s["supernovae_ia"], abs=1e-6)
        assert r.summary["closed_box"] == S.closed_box(r.params["metal_yield"], r.summary["gas_fraction"])
    # the iron of the type Ia supernovae: no other source changes when ia_fraction does
    sim = Stars()
    none, some = sim.run(1, efficiency=0.3, t_end=1.25, ia_fraction=0.0), sim.run(1, efficiency=0.3, t_end=1.25)
    events = some.summary["supernovae_ia"]
    assert events > 0 and none.summary["supernovae_ia"] == 0
    extra = (some.ledger[-1]["returned"]["iron"] - none.ledger[-1]["returned"]["iron"]) * S.BOX_MASS
    assert extra == pytest.approx(S.ia_iron(events), rel=1e-9)
    # stars of 1.1 solar masses formed in the first step return their gas at their death_time
    b = next(b for b in BINS if b.lo <= 1.1 < b.hi)
    assert S.death_time(0.0, b.mass) == pytest.approx(7.87, abs=0.01) and S._death_step(1, b) == 32
    # 8 Gyr on (step 32), the dead of the first step are its stars above the turnoff mass of 8 Gyr
    assert b.lo == S.turnoff_mass(8.0)
    dead = sum(x.share for x in BINS if S._death_step(1, x) <= 32)
    assert dead == pytest.approx(S.imf_mass(S.turnoff_mass(8.0), 100), rel=1e-12)


# --------------------------------------------------------------------------- hand-off

def test_handoff_in_takes_the_gas_of_the_first_hour(sim):
    assert S.handoff_in({"hydrogen": 0.752, "helium": 0.248}) == {"hydrogen": 0.752, "helium": 0.248}
    for seed in range(1, 60):
        below = nucleo.simulation().run(seed, **nucleo.random_params(random.Random(seed))).summary
        p = S.handoff_in(below)
        assert set(p) == {"hydrogen", "helium"} and abs(p["hydrogen"] + p["helium"] - 1) < 1e-12
        assert p["helium"] == round(p["helium"], 4) and 0 < p["helium"] < 1
        assert p["helium"] == pytest.approx(below["helium"], abs=2e-3), "helium-3 and deuterium are traces"
        assert p["helium"] >= round(below["helium"], 4) - 1e-4
        r = sim.run(seed, t_end=9.0, **p)
        assert sim.conserved(r).ok and r.steps[0]["helium"] == pytest.approx(p["helium"])
        assert r.summary["helium"] > p["helium"] and r.summary["metallicity"] > 0
    with pytest.raises((ValueError, KeyError)):
        S.handoff_in({"helium": 0.25})
    with pytest.raises(ValueError):
        S.handoff_in({"hydrogen": 0.0, "helium": 1.0})
    traces = S.handoff_in({"hydrogen": 0.75, "helium": 0.2497, "deuterium": 1e-4, "helium3": 1e-4})
    assert traces["helium"] == round((0.2497 + 3e-4 * 0.75) / (0.75 + 2e-4 * 0.75 + 0.2497 + 3e-4 * 0.75), 4)


def test_handoff_out_is_the_cloud_a_later_star_forms_from(sim):
    r = sim.run(1, **{**DEFAULTS, "t_end": 9.25})
    end = S.cloud(r)
    assert set(end) == set(ELEMENTS) | {"metallicity", "time"} and end["time"] == 9.25 == r.summary["time"]
    for key in ELEMENTS + ("metallicity",):
        assert end[key] == r.summary[key]
    assert sum(end[e] for e in ELEMENTS) == pytest.approx(1.0, abs=1e-12)
    assert end["metallicity"] == pytest.approx(sum(end[e] for e in METALS))
    early = S.cloud(r, 2.0)
    assert early["time"] == 2.0 and early["oxygen"] == r.steps[8]["oxygen"] and early["metallicity"] < end["metallicity"]
    assert S.cloud(r, 0.0)["metallicity"] == 0
    for bad in (9.5, 2.1, -0.25):
        with pytest.raises(ValueError):
            S.cloud(r, bad)
    # every summary a later level reads is a number or a word that fits a dense line
    for key, value in r.summary.items():
        assert is_dense(f"{key} {dense_value(value)}.")


# --------------------------------------------------------------------------- review: every regime

def test_the_gates_hold_in_every_regime(sim):
    """The parameters forced to every corner ``run`` accepts, far outside what ``random_params``
    draws. What holds there is a law of the model: the gates, a gas that only shrinks, remnants
    that only grow, oxygen over iron that starts at the ratio of the supernova table and never
    rises. Hydrogen, helium and the metals are another matter (the next test)."""
    for eff in (0.01, 0.1, 0.3, 1.0, 2.0, 4.0):
        for ia in (0.0, 0.001, 0.004, 0.008, 0.1):
            for helium in (0.0, 0.248, 0.49):
                r = sim.run(1, efficiency=eff, ia_fraction=ia, hydrogen=1 - helium, helium=helium)
                v = sim.conserved(r)
                assert v.ok, (eff, ia, helium, v)
                for prev, s in zip(r.steps, r.steps[1:]):
                    assert s["gas"] < prev["gas"] and s["remnants"] > prev["remnants"], (eff, ia, s["time"])
                    assert s["supernovae_cc"] > prev["supernovae_cc"]
                    assert all(0 <= s[e] <= 1 for e in ELEMENTS)
                a = [s["alpha_over_iron"] for s in r.steps[1:]]
                assert a[0] == pytest.approx(2.5, rel=1e-12), "0.033 / 0.003 / 4.4, the supernova table"
                assert all(later <= earlier * (1 + 1e-12) for earlier, later in zip(a, a[1:])), (eff, ia)
                if ia == 0:
                    assert a[-1] == pytest.approx(2.5, rel=1e-9) and r.summary["supernovae_ia"] == 0
                else:
                    assert a[-1] < a[0] and r.steps[4]["supernovae_ia"] == 0 < r.steps[5]["supernovae_ia"]
    # the make-up at the start does not change the metals (the yields are shares of the returned gas)
    assert sim.run(1, hydrogen=1.0, helium=0.0).summary["metallicity"] == pytest.approx(
        sim.run(1, hydrogen=0.51, helium=0.49).summary["metallicity"], rel=1e-9)


def test_an_emptied_box_refills_from_its_old_stars(sim):
    """That the metals rise in every step is what the canonical runs do, not a law of the model.
    When the gas is nearly gone, what old stars return (the metal-poor gas they were born from,
    with a little new carbon and nitrogen) is most of the gas, and the metallicity of the gas
    falls unless type Ia supernovae make up for it."""
    r = sim.run(1, efficiency=1.0, ia_fraction=0.0)
    z = [s["metallicity"] for s in r.steps]
    peak = max(range(len(z)), key=z.__getitem__)
    assert r.steps[peak]["time"] == 7.25 and r.steps[peak]["gas"] < 0.02
    assert z[peak] == pytest.approx(0.02022, rel=1e-3) and z[-1] == pytest.approx(0.01904, rel=1e-3)
    assert all(later < earlier for earlier, later in zip(z[peak:], z[peak + 1:])), "falls from the peak to the end"
    assert all(later > earlier for earlier, later in zip(z[:peak], z[1:peak + 1])), "and rose up to it"
    assert sim.conserved(r).ok
    # the corner of the canonical range: the fewest type Ia supernovae, the fastest star formation
    low, high = 0.001, 1.0
    assert all(random_params(random.Random(seed))["ia_fraction"] >= low
               and random_params(random.Random(seed))["efficiency"] <= high for seed in range(300))
    corner = sim.run(1, efficiency=high, ia_fraction=low)
    falls = [s["time"] for prev, s in zip(corner.steps, corner.steps[1:]) if s["metallicity"] <= prev["metallicity"]]
    assert falls == [9.5, 9.75, 10.0, 10.25] and sim.conserved(corner).ok
    # with the default share of type Ia supernovae it rises all the way, however fast the stars form
    for eff in (0.3, 0.6, 1.0):
        steady = [s["metallicity"] for s in sim.run(1, efficiency=eff).steps]
        assert all(later > earlier for earlier, later in zip(steady, steady[1:])), eff
    # a box emptied within two Gyr: the share of hydrogen in its last gas rises for a while,
    # though no star makes hydrogen
    fast = sim.run(1, efficiency=4.0, ia_fraction=0.0)
    rises = [s for prev, s in zip(fast.steps, fast.steps[1:]) if s["hydrogen"] >= prev["hydrogen"]]
    assert [s["time"] for s in rises] == [2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75] and all(s["gas"] < 0.02 for s in rises)
    assert sim.conserved(fast).ok
    assert all(led["returned"]["hydrogen"] < led["astrated"]["hydrogen"] for led in fast.ledger[1:])


def test_the_simulation_passes_the_closed_box_when_the_gas_is_nearly_gone(sim):
    """The closed-box formula is no upper bound. It lets every return come at once; the simulation
    lets the returns of old stars and the type Ia supernovae fall into the little gas that is
    left, and then stands above the formula."""
    r = sim.run(1, ia_fraction=0.008)
    y = r.params["metal_yield"]
    ratio = [S.closed_box(y, s["gas"]) / s["metallicity"] for s in r.steps[1:]]
    above = [s["time"] for s, x in zip(r.steps[1:], ratio) if x < 1]
    assert above == [13.0, 13.25, 13.5] and all(s["gas"] < 0.08 for s in r.steps if s["time"] >= 13.0)
    assert ratio[-1] == pytest.approx(0.982, abs=0.003) and r.summary["closed_box"] < r.summary["metallicity"]
    assert all(x > 1 for s, x in zip(r.steps[1:], ratio) if s["gas"] > 0.1), "with gas to spare the simulation lags"
    assert all(b < a for a, b in zip(ratio, ratio[1:]))


def test_the_targets_over_two_thousand_seeds(sim):
    """The numbers of the table in the module docstring, measured again."""
    ends = {stage: 0 for stage in STAGES}
    finals = []
    not_oxygen = above_formula = helium_falls = 0
    worst, lowest = 0.0, 9.0
    for seed in range(1, 2001):
        r = sim.rollout(seed)
        ends[r.summary["stage"]] += 1
        finals.append(r.summary["metallicity"])
        y = r.params["metal_yield"]
        odd = over = False
        for prev, s in zip(r.steps, r.steps[1:]):
            worst = max(worst, abs(s["gas"] + s["stars"] + s["remnants"] - 1.0))
            assert s["hydrogen"] < prev["hydrogen"] and s["metallicity"] > prev["metallicity"], (seed, s["time"])
            assert s["gas"] < prev["gas"] and s["remnants"] > prev["remnants"], (seed, s["time"])
            assert prev["time"] == 0 or s["alpha_over_iron"] <= prev["alpha_over_iron"] * (1 + 1e-12), (seed, s["time"])
            if s["helium"] <= prev["helium"]:
                helium_falls += 1
                assert s["gas"] < 0.012 and prev["helium"] - s["helium"] < 1e-4, (seed, s["time"])
            if top_metal(s) != "oxygen":
                odd = True
                assert top_metal(s) == "iron" and s["gas"] < 0.014 and s["stage"] == "exhausted", (seed, s["time"])
            x = S.closed_box(y, s["gas"]) / s["metallicity"]
            lowest = min(lowest, x)
            if x < 1:
                over = True
                assert s["gas"] < 0.064 and r.params["efficiency"] >= 0.32 and r.params["ia_fraction"] >= 0.0048, seed
        not_oxygen += odd
        above_formula += over
        assert over == (r.summary["closed_box"] < r.summary["metallicity"]), "once above the formula, above it to the end"
    assert ends == {"enriching": 844, "exhausted": 683, "mature": 447, "first_stars": 26}
    finals.sort()
    assert finals[0] == pytest.approx(0.00072, rel=0.01) and finals[-1] == pytest.approx(0.0424, rel=0.01)
    assert (finals[999] + finals[1000]) / 2 == pytest.approx(0.0158, rel=0.01)
    assert not_oxygen == 24 and above_formula == 56 and lowest == pytest.approx(0.946, abs=0.002)
    assert helium_falls == 177 and worst < 5e-15


# --------------------------------------------------------------------------- review: the type Ia gate

def test_type_ia_supernovae_are_a_share_of_the_white_dwarfs_formed_a_gyr_before(sim, rollouts, monkeypatch):
    assert S.white_dwarfs_formed(1000.0, 0.0) == 0 and S.white_dwarfs_formed(1000.0, -1.0) == 0
    # after 13.5 Gyr: the stars from the turn-off mass 0.887 up to 8 solar masses
    assert S.white_dwarfs_formed(1000.0, 13.5) == pytest.approx(1000 * S.imf_number(S.turnoff_mass(13.5), 8.0), rel=1e-12)
    assert S.white_dwarfs_formed(1000.0, 13.5) == pytest.approx(141.8, rel=1e-3)
    assert S.white_dwarfs_formed(1000.0, 0.25) == pytest.approx(1000 * S.imf_number(4.373, 8.0), rel=1e-3)
    assert S.white_dwarfs_formed(1000.0, 1e-6) == 0, "nothing below 8 solar masses dies that young"
    assert S.white_dwarfs_formed(1000.0, 1e9) == pytest.approx(1000 * S.imf_number(0.1, 8.0)), "in the end all of them"
    for r in rollouts.values():
        born = [0.0] + [(b["formed"] - a["formed"]) * S.BOX_MASS for a, b in zip(r.ledger, r.ledger[1:])]
        for t, s in enumerate(r.steps):
            # the stars of step j were born at (j - 1) * DT; at step t - 4 they are (t - 4 - j + 1) * DT old
            due = r.params["ia_fraction"] * sum(S.white_dwarfs_formed(born[j], (t - 4 - j + 1) * DT)
                                                for j in range(1, t - 3))
            assert s["supernovae_ia"] == pytest.approx(due, rel=1e-9, abs=1e-9), (r.seed, t)
            if t < 5:
                assert s["supernovae_ia"] == 0
    # a box whose white dwarfs explode after half a Gyr obeys every other gate, and fails this one
    with monkeypatch.context() as m:
        m.setattr(S, "IA_DELAY", 0.5)
        early = sim.run(1, **DEFAULTS)
    assert early.steps[3]["supernovae_ia"] > 0
    v = sim.conserved(early)
    assert not v.ok and "white dwarfs formed" in v.reason and v.reason.startswith("step 3:"), v
    assert sim.conserved(sim.run(1, **DEFAULTS)).ok
    # and one that claims another share than it ran with
    liar = sim.run(1, **DEFAULTS)
    liar.params["ia_fraction"] = 0.002
    assert not sim.conserved(liar).ok and "white dwarfs formed" in sim.conserved(liar).reason


# --------------------------------------------------------------------------- review: numbers that do not move

def test_sums_are_added_from_left_to_right_and_not_by_the_builtin():
    """From Python 3.12 on the built-in ``sum`` adds floats with a correction term, so its last
    digit depends on the Python version. The module adds with its own loop."""
    code = inspect.getsource(S).split('"""\n\nfrom __future__', 1)[1]
    assert not re.search(r"(?<![\w.])sum\(", code), "the built-in sum in the module"
    assert "fsum" not in code and "numpy" not in code and "import random" in code
    assert S._sum([]) == 0.0 and type(S._sum([])) is float and S._sum(x for x in (0.5, 0.25)) == 0.75
    assert S._sum([1e16, 1.0, -1e16]) == 0.0, "plain addition drops the 1; a corrected sum would keep it"
    values = [0.1 * k for k in range(1, 40)]
    total = 0.0
    for v in values:
        total += v
    assert S._sum(values) == total
    # pinned to twelve digits: a last digit of exp or pow may differ between machines, no more
    r = Stars().run(1, **DEFAULTS)
    assert r.steps[36]["metallicity"] == pytest.approx(0.013554916325850689, rel=1e-12)
    assert r.ledger[54]["gas"]["iron"] == pytest.approx(0.00018432125428774054, rel=1e-12)
    assert Stars().rollout(3).steps[8]["oxygen"] == pytest.approx(0.00277747317705134, rel=1e-12)
    assert Stars().rollout(3).summary["metallicity"] == pytest.approx(0.021066118730105513, rel=1e-12)


def test_a_cut_that_misses_an_edge_by_a_rounding_error_leaves_no_sliver(monkeypatch):
    """The cut for a lifetime of 10 Gyr is the fixed edge 1.0. A power function that gave
    0.9999999999999998 instead must not make a bin of that width."""
    edges, original = S.bin_edges(), S.turnoff_mass
    assert original(10.0) == 1.0
    for factor in (1 - 2 ** -52, 1 + 2 ** -52, 1 - 1e-12):
        with monkeypatch.context() as m:
            m.setattr(S, "turnoff_mass", lambda age, f=factor: original(age) * f if age == 10.0 else original(age))
            assert S.turnoff_mass(10.0) != 1.0
            assert S.bin_edges() == edges and len(S._bins()) == 83
    gaps = [(hi - lo) / lo for lo, hi in zip(edges, edges[1:])]
    assert min(gaps) > 0.0017, "the narrowest bin is the sliver from 1.2478 to 1.25, a million times the guard"
    assert S.bin_edges() == edges and S._bins() == BINS


def test_canonical_rollouts_are_the_ones_the_gate_knows(sim):
    """A what-if run under a seed has the seed's prompts and other answers. ``is_canonical``
    tells them apart, and the gate does not accept the lines of the what-if run."""
    assert S.is_canonical(sim.rollout(3)) and S.is_canonical(S.rollout(3)) and S.is_canonical(Stars().rollout(123456789))
    assert S.is_canonical(sim.run(3, **random_params(random.Random(3))))
    what_if = sim.run(3, **DEFAULTS)
    assert not S.is_canonical(what_if) and sim.conserved(what_if).ok
    assert not S.is_canonical(sim.run(3, **{**random_params(random.Random(3)), "ia_fraction": 0.0075}))
    assert not S.is_canonical(sim.run(-3, **random_params(random.Random(-3)))), "no prompt can name a negative seed"
    other = copy.deepcopy(sim.rollout(3))
    other.sim = "cells"
    assert not S.is_canonical(other)
    questions = [ln for ln in lines(what_if) if ln.kind != "record"]
    refused = [ln for ln in questions if not sim.check(ln.prompt, ln.answer).ok]
    assert len(refused) > len(questions) / 2
    assert all(S.is_canonical(sim.rollout(seed)) for seed in SEEDS)


# --------------------------------------------------------------------------- review: the other gates

def test_does_not_own_the_prompts_of_rounds_five_and_six(sim, all_lines):
    """Real prompts of the round-5 gates and of the round-6 levels and lessons, not made-up ones."""
    from haishool.cosmos import life, planets, predict
    from haishool.truth import loop

    gates, missing = loop.load_gates(loop.TOPICS)
    assert not missing and len(gates) == 5
    theirs = [ln.prompt for g in gates for ln in g.generate(random.Random(1), 300) if ln.kind != "record"]
    theirs += [ln.prompt for ln in predict.generate(random.Random(1), 500)]
    theirs += [ln.prompt for ln in planets.simulation().generate(random.Random(1), 2) if ln.kind != "record"]
    theirs += [ln.prompt for ln in nucleo.lines(nucleo.rollout(3)) if ln.kind != "record"]
    theirs += [ln.prompt for ln in life.lines(life.simulation().rollout(3)) if ln.kind != "record"]
    assert len(theirs) > 3000
    for prompt in theirs:
        assert not sim.owns(prompt) and not LESSONS.owns(prompt), prompt
        assert sim.check(prompt, "1").reason == LESSONS.check(prompt, "1").reason == "not my question"
    # and none of them takes a question of this level
    mine = [ln.prompt for ln in all_lines[:3000] if ln.kind != "record"]
    mine += [ln.prompt for ln in LESSONS.generate(random.Random(2), 300)]
    for g in gates + [predict, planets.simulation(), nucleo.simulation(), life.simulation()]:
        assert not any(g.owns(prompt) for prompt in mine), g
    # the two gates of this level share no prompt either
    assert not any(sim.owns(p) for p in mine[-300:]) and not any(LESSONS.owns(p) for p in mine[:-300])


# --------------------------------------------------------------------------- review: lessons

def _fate_by_hand(mass: float) -> str:
    if mass < 0.08:
        return "brown_dwarf"
    if mass < 8:
        return "white_dwarf"
    return "neutron_star" if mass <= 25 else "black_hole"


def _remnant_by_hand(mass: float) -> float:
    return {"brown_dwarf": mass, "white_dwarf": min(mass, 0.11 * mass + 0.39), "neutron_star": 1.4,
            "black_hole": mass / 2}[_fate_by_hand(mass)]


def _stage_by_hand(gas: float, metallicity: float) -> str:
    if gas < 0.05:
        return "exhausted"
    if metallicity < 0.00134:
        return "first_stars"
    return "enriching" if metallicity < 0.0134 else "mature"


#: the Salpeter mass function from 0.1 to 100 solar masses, normalised to one solar mass
_SALPETER = 0.35 / (0.1 ** -0.35 - 100 ** -0.35)

#: every rule once more, written from the module docstring alone, with no function of the module
BY_HAND = {
    "luminosity": lambda mass: mass ** 3.5,
    "lifetime": lambda mass: 10 * mass ** -2.5,
    "death_time": lambda born, mass: born + 10 * mass ** -2.5,
    "turnoff": lambda age: (10 / age) ** 0.4,
    "shines": lambda mass: "yes" if mass >= 0.08 else "no",
    "fate": _fate_by_hand,
    "remnant": _remnant_by_hand,
    "returned": lambda mass: mass - _remnant_by_hand(mass),
    "gas_to_stars": lambda gas, efficiency, dt: gas * (1 - math.exp(-efficiency * dt)),
    "stars_above": lambda formed, mass: formed * _SALPETER * (mass ** -1.35 - 100 ** -1.35) / 1.35,
    "ia_iron": lambda events: 0.7 * events,
    "closed_box": lambda metal_yield, gas_fraction: metal_yield * math.log(1 / gas_fraction),
    "alpha_over_iron": lambda oxygen, iron: 0.0 if iron <= 0 else oxygen / iron / 4.4,
    "stage": _stage_by_hand,
}


@pytest.fixture(scope="module")
def many_lessons():
    return {seed: LESSONS.generate(random.Random(seed), 500) for seed in (1, 2, 3, 4)}


def test_lessons_of_four_seeds_are_accepted_and_agree_with_formulas_written_by_hand(sim, many_lessons):
    assert set(BY_HAND) == set(RULES)
    seen = {name: 0 for name in RULES}
    texts = set()
    for seed, lessons in many_lessons.items():
        assert len(lessons) == 500
        assert [ln.text for ln in lessons] == [ln.text for ln in LESSONS.generate(random.Random(seed), 500)]
        for ln in lessons:
            texts.add(ln.text)
            assert is_dense(ln.text) and n_tokens(ln.text) <= 40 < MAX_TOKENS, ln.text
            assert "." not in ln.prompt and "." not in ln.answer
            v = LESSONS.check(ln.prompt, ln.answer)
            assert v.ok and v.expected == ln.answer, (ln.text, v)
            assert not sim.owns(ln.prompt)
            name, values = LESSONS.parse(ln.prompt)
            seen[name] += 1
            mine = BY_HAND[name](*values)
            if isinstance(mine, str):
                assert ln.answer == mine, ln.text
            else:
                # the answer has four significant digits
                assert parse_num(ln.answer) == pytest.approx(mine, rel=6e-4, abs=1e-12), (ln.text, mine)
                assert LESSONS.check(ln.prompt, num(mine)).ok, (ln.text, mine)
    assert min(seen.values()) >= 100, "every rule often enough to judge"
    assert len(texts) > 1900, "hardly a lesson twice"


def test_no_answer_dominates_a_lesson_rule():
    """A rule whose answer is nearly always the same word teaches that word, not the rule. The
    input ranges are set so that the words are balanced as far as an even draw allows."""
    answers = {name: {} for name in RULES}
    for seed in (10, 11, 12, 13):
        for ln in LESSONS.generate(random.Random(seed), 5000):
            counts = answers[ln.prompt.split()[2]]
            counts[ln.answer] = counts.get(ln.answer, 0) + 1
    share = {name: {a: n / sum(counts.values()) for a, n in counts.items()} for name, counts in answers.items()}
    top = {name: max(shares.values()) for name, shares in share.items()}
    assert set(share["shines"]) == {"yes", "no"} and 0.46 <= share["shines"]["yes"] <= 0.54, share["shines"]
    assert set(share["fate"]) == set(S.FATES) and top["fate"] < 0.45, share["fate"]
    assert share["fate"]["white_dwarf"] > 0.15 and abs(share["fate"]["neutron_star"] - share["fate"]["black_hole"]) < 0.06
    assert set(share["stage"]) == set(STAGES) and top["stage"] < 0.52, share["stage"]
    assert min(share["stage"].values()) > 0.03 and abs(share["stage"]["mature"] - share["stage"]["enriching"]) < 0.12
    assert 0.2 < top["remnant"] < 0.32 and max(share["remnant"], key=share["remnant"].get) == "1 point 4"
    for name in set(RULES) - {"shines", "fate", "stage", "remnant"}:
        assert top[name] < 0.02 and len(share[name]) > 300, (name, top[name])
    # what an even draw gives, from the ranges alone
    low, high = RULES["shines"].inputs[0].low, RULES["shines"].inputs[0].high
    assert (high - S.M_SHINE) / (high - low) == pytest.approx(0.5)
    high = RULES["fate"].inputs[0].high
    assert high - S.M_BLACK_HOLE == S.M_BLACK_HOLE - S.M_SUPERNOVA, "as many black holes as neutron stars"


def test_the_lesson_ranges_cover_what_the_simulation_does(rollouts):
    spec = {name: {i.name: i for i in rule.inputs} for name, rule in RULES.items()}
    eff = spec["gas_to_stars"]["efficiency"]
    assert (eff.low, eff.high) == (0.1, 1.0), "the range random_params draws from"
    assert spec["gas_to_stars"]["dt"].low <= DT <= spec["gas_to_stars"]["dt"].high
    assert (spec["gas_to_stars"]["gas"].low, spec["gas_to_stars"]["gas"].high) == (0, 1) == (
        spec["stage"]["gas"].low, spec["stage"]["gas"].high)
    assert (spec["stars_above"]["mass"].low, spec["stars_above"]["mass"].high) == (S.M_MIN, S.M_MAX)
    assert spec["turnoff"]["age"].high == S.T_MAX == spec["death_time"]["born"].high
    assert spec["shines"]["mass"].low < S.M_SHINE < spec["shines"]["mass"].high
    for name in ("fate", "remnant", "returned"):
        assert spec[name]["mass"].low < S.M_SHINE and spec[name]["mass"].high > S.M_BLACK_HOLE
    box = spec["closed_box"]["metal_yield"]
    assert box.low < S.generation(0.001)["metal_yield"] < S.generation(0.008)["metal_yield"] < box.high
    # the numbers of the rollouts, as printed, are questions the lessons can be asked
    asked = 0
    for r in rollouts.values():
        for prev, s, prev_led, led in zip(r.steps, r.steps[1:], r.ledger, r.ledger[1:]):
            gas, z = parse_num(dense_value(s["gas"])), parse_num(dense_value(s["metallicity"]))
            if z <= spec["stage"]["metallicity"].high:
                prompt = f"stars predict stage gas {num(gas, sig=6)} metallicity {num(z, sig=6)}"
                assert LESSONS.check(prompt, S.stage(gas, z)).ok, prompt
                asked += 1
            before = parse_num(dense_value(prev["gas"]))
            ln = LESSONS.line("gas_to_stars", [before, r.params["efficiency"], DT])
            assert ln is not None and parse_num(ln.answer) == pytest.approx(led["formed"] - prev_led["formed"], rel=0.01)
            asked += 1
            if 0 < s["iron"] <= 0.005 and s["oxygen"] <= 0.02:
                o, fe = parse_num(dense_value(s["oxygen"])), parse_num(dense_value(s["iron"]))
                ln = LESSONS.line("alpha_over_iron", [o, fe])
                assert ln is not None and parse_num(ln.answer) == pytest.approx(s["alpha_over_iron"], rel=0.02)
                asked += 1
        if r.summary["gas_fraction"] >= spec["closed_box"]["gas_fraction"].low:
            ln = LESSONS.line("closed_box", [parse_num(dense_value(r.params["metal_yield"])),
                                             parse_num(dense_value(r.summary["gas_fraction"]))])
            assert ln is not None and parse_num(ln.answer) == pytest.approx(r.summary["closed_box"], rel=0.02)
    assert asked > 3000


# --------------------------------------------------------------------------- speed

def test_a_run_is_fast(sim):
    worst = 0.0
    for seed in range(100, 120):
        start = time.perf_counter()
        sim.run(seed, **random_params(random.Random(seed)))
        worst = max(worst, time.perf_counter() - start)
    start = time.perf_counter()
    sim.run(1, **DEFAULTS)
    assert max(worst, time.perf_counter() - start) < 0.5


# --------------------------------------------------------------------------- review, round 7 hardening

def test_the_training_lines_are_pinned():
    """Every line of the canonical seeds 1 to 200, 2000 lessons and the lesson records, as one
    digest. The lines carry three or four significant digits, so a last-digit difference of
    ``exp`` or a power on another machine does not move them; the full-precision steps and
    ledgers were also measured identical on CPython 3.11.9 and 3.12.14 (seeds 1 to 300). Levels
    above (world7) build on these numbers: a change here must be deliberate."""
    import hashlib
    h = hashlib.sha256()
    for seed in range(1, 201):
        for ln in lines(S.rollout(seed)):
            h.update((ln.text + "\n").encode())
    for ln in LESSONS.generate(random.Random(5), 2000):
        h.update((ln.text + "\n").encode())
    for ln in LESSONS.records():
        h.update((ln.text + "\n").encode())
    assert h.hexdigest() == "224ab2bc836fb01e9e4cbaad14b8fe22342f84d9b1c7b5488356357c0984a446"


def test_the_docstring_says_what_is_put_in_and_what_is_rough():
    doc = " ".join(S.__doc__.split())
    for phrase in ("0.0142", "settled inward", "rough average over the main sequence", "nearer 4 around one solar mass",
                   "1.5 to 2 times shorter", "it does not predict it", "Chandrasekhar mass of the classic picture",
                   "many may explode below it", "1.4 solar masses of a typical neutron star"):
        assert phrase in doc, phrase
    assert "holds near one solar mass" not in doc and "of an exploding white dwarf;" not in doc
    for name in ("luminosity", "lifetime"):
        assert "rough average" in RULES[name].description
    # the two laws are exact at one solar mass only, and they are the textbook pair
    assert S.luminosity(1.0) == 1.0 and S.lifetime(1.0) == S.T_SUN
    assert S.lifetime(3.0) == pytest.approx(S.T_SUN * 3.0 / 3.0 ** 3.5)


def test_the_lesson_shares_of_the_docstring_follow_from_the_ranges():
    """The shares the docstring gives, from the input ranges alone and measured on 40000 lessons."""
    def span(rule):
        i = RULES[rule].inputs[0]
        return i.low, i.high
    lo, hi = span("fate")
    expected = {"brown_dwarf": (S.M_SHINE - lo) / (hi - lo), "white_dwarf": (S.M_SUPERNOVA - S.M_SHINE) / (hi - lo),
                "neutron_star": (S.M_BLACK_HOLE - S.M_SUPERNOVA) / (hi - lo), "black_hole": (hi - S.M_BLACK_HOLE) / (hi - lo)}
    assert [round(expected[f], 2) for f in ("black_hole", "neutron_star", "white_dwarf")] == [0.40, 0.40, 0.19]
    lo, hi = span("remnant")
    assert round((S.M_BLACK_HOLE - S.M_SUPERNOVA) / (hi - lo), 2) == 0.28
    counts = {"fate": {}, "remnant": {}, "stage": {}}
    for seed in range(10, 20):
        for ln in LESSONS.generate(random.Random(seed), 4000):
            name = ln.prompt.split()[2]
            if name in counts:
                counts[name][ln.answer] = counts[name].get(ln.answer, 0) + 1
    share = {n: {a: k / sum(c.values()) for a, k in c.items()} for n, c in counts.items()}
    for f, p in expected.items():
        assert share["fate"].get(f, 0) == pytest.approx(p, abs=0.02), f
    assert share["remnant"]["1 point 4"] == pytest.approx(0.28, abs=0.02)
    assert sorted(share["remnant"].values())[-2] < 0.03, "no other numeric answer is common"
    stage_expected = {"exhausted": 0.05, "first_stars": 0.95 * 0.00134 / 0.027,
                      "enriching": 0.95 * (0.0134 - 0.00134) / 0.027, "mature": 0.95 * (0.027 - 0.0134) / 0.027}
    for st, p in stage_expected.items():
        assert share["stage"][st] == pytest.approx(p, abs=0.02), st


def test_type_ia_supernovae_never_destroy_more_white_dwarf_than_formed():
    """Each event takes 1.4 solar masses out of the remnants, more than one white dwarf of this
    model weighs (0.49 to 1.27). At the largest ``ia_fraction`` ``run`` accepts, 0.1, what the
    events of a generation destroy is still under a quarter of the white-dwarf mass it leaves."""
    wd = [b for b in BINS if b.fate == "white_dwarf"]
    masses = [S.remnant_mass(b.mass) for b in wd if b.lifetime <= S.T_MAX]
    assert 0.48 < min(masses) and max(masses) < 1.28 < S.IA_MASS
    locked = 0.0
    for b in wd:
        if b.lifetime <= S.T_MAX:
            locked += b.share * (1 - b.returned)
    destroyed = S.generation(0.1)["supernovae_ia"] * S.IA_MASS
    assert destroyed < 0.25 * locked, (destroyed, locked)
    sim = Stars()
    for eff in (0.1, 1.0, 4.0):
        r = sim.run(1, efficiency=eff, ia_fraction=0.1)
        assert sim.conserved(r).ok and min(s["remnants"] for s in r.steps[1:]) > 0
