import copy
import doctest
import hashlib
import importlib
import math
import os
import random
import re
import statistics
import subprocess
import sys
import time
from fractions import Fraction

import numpy as np
import pytest

from haishool.cosmos import Simulation
from haishool.evo import LessonGate, sig
from haishool.evo import signals as S
from haishool.evo.signals import (COUNT_KEYS, INPUT_KEYS, KEYS, LESSONS, LEXICON_MIN, MAX_SYLLABLES, ORDERS,
                                  OUTCOMES, PARAM_KEYS, PROPERTIES, RULES, SNAPSHOTS, STAGES, STATE_KEYS, STEPS,
                                  SUMMARY_KEYS, VOCABULARY, WORD_KEYS, Signals, dense_value, handoff_in, lines,
                                  param_value, random_params, simulation, text_of, view)
from haishool.truth import Gate, is_dense, num, parse_num, split_line

SEEDS = list(range(1, 41))
#: seeds of the experiments with chosen parameters
TRIALS = list(range(1, 17))
FLOAT_KEYS = ("success", "alarm", "word_length", "compositional", "dialects")


def n_tokens(text: str) -> int:
    """As ``haishool.student.tokens`` counts them (that module needs torch, so not imported)."""
    return len(text.replace(".", " . ").split())


def same(a, b) -> bool:
    """Equality that also reaches into the weight tables kept in the states."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.shape == b.shape and bool((a == b).all())
    if isinstance(a, dict):
        return isinstance(b, dict) and list(a) == list(b) and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


@pytest.fixture(scope="module")
def sim():
    return Signals()


@pytest.fixture(scope="module")
def rollouts(sim):
    return {seed: sim.rollout(seed) for seed in SEEDS}


@pytest.fixture(scope="module")
def all_lines(rollouts):
    return [ln for r in rollouts.values() for ln in lines(r)]


@pytest.fixture(scope="module")
def trials(sim):
    """Runs with chosen parameters, kept for the plausibility tests: ``trials(**params)[seed]``."""
    cache: dict = {}

    def get(**params):
        key = tuple(sorted(params.items()))
        if key not in cache:
            cache[key] = {seed: sim.run(seed, **params) for seed in TRIALS}
        return cache[key]
    return get


# ---------------------------------------------------------------------------------------------
# the level as the contract asks for it


def test_protocols():
    s = simulation()
    assert isinstance(s, Simulation) and isinstance(s, Gate)
    assert S.SIM == s.sim == s.topic == "signals" and s.records() == [] and simulation() is s
    for key in PARAM_KEYS + STATE_KEYS + SUMMARY_KEYS:
        assert key in KEYS, key
    assert COUNT_KEYS | WORD_KEYS <= set(KEYS) and set(INPUT_KEYS) <= set(PARAM_KEYS)
    assert isinstance(LESSONS, LessonGate) and S.lesson_gate() is LESSONS and LESSONS.topic == "predict_signals"
    assert len(RULES) >= 8 and LESSONS.rules is RULES
    for name in ("run", "conserved", "lines", "check", "owns", "random_params", "handoff_in", "simulation"):
        assert callable(getattr(S, name)), name
    r = S.run(5, population=6, meanings=4)
    assert r.sim == "signals" and S.conserved(r).ok and len(r.steps) == STEPS + 1 == 61
    assert set(STAGES) == {"silent", "calls", "words", "lexicon", "grammar", "language"}
    assert set(OUTCOMES) == {"silent", "calls", "lexicon", "language"}


def test_docstring_examples_hold():
    assert doctest.testmod(S).failed == 0


def test_same_seed_same_output():
    a, b = Signals(), Signals()
    for seed in (1, 7, 4321):
        ra, rb = a.rollout(seed), b.rollout(seed)
        assert ra.steps == rb.steps and ra.summary == rb.summary and ra.params == rb.params
        assert same(ra.states, rb.states) and same(ra.languages, rb.languages) and same(ra.world, rb.world)
        assert [ln.text for ln in lines(ra)] == [ln.text for ln in lines(rb)]
    assert [ln.text for ln in lines(a.rollout(1))] != [ln.text for ln in lines(a.rollout(2))]
    assert a.run(5, population=30).steps == b.run(5, population=30).steps
    assert a.run(5, population=30).steps != a.run(6, population=30).steps
    assert a.run(5, population=30).steps != a.run(5, population=31).steps


def test_forty_seeds_replay_exactly(rollouts):
    """Two runs of a seed give the same steps, agents and language, to the last bit."""
    fresh = Signals()
    for seed, r in rollouts.items():
        again = fresh.run(seed, **random_params(random.Random(seed)))
        assert again.steps == r.steps and again.summary == r.summary and again.params == r.params
        assert same(again.states, r.states) and same(again.languages, r.languages)
        assert [repr(v) for s in again.steps for v in s.values()] == [repr(v) for s in r.steps for v in s.values()]


def test_output_does_not_depend_on_the_hash_seed():
    """Sets of words are only asked, never walked: another PYTHONHASHSEED gives the same lines."""
    code = ("import hashlib; from haishool.evo import signals as S; "
            "print(hashlib.sha256('|'.join(l.text for s in (87, 91) for l in S.lines(S.simulation().rollout(s)))"
            ".encode()).hexdigest())")
    here = hashlib.sha256("|".join(ln.text for s in (87, 91) for ln in lines(Signals().rollout(s))).encode()).hexdigest()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for hash_seed in ("1", "2"):
        env = dict(os.environ, PYTHONHASHSEED=hash_seed, PYTHONPATH=root)
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=root)
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == here


def test_values_are_plain_python_numbers_and_words(rollouts):
    for r in rollouts.values():
        for row in r.steps + [r.summary, r.params]:
            for key, value in row.items():
                assert type(value) in (float, int, str), (key, type(value))
                assert (type(value) is int) == (key in COUNT_KEYS), key
                assert (type(value) is str) == (key in WORD_KEYS), key
                if type(value) is float:
                    assert math.isfinite(value) and (value >= 0 or key == "topographic"), (key, value)
        assert list(r.summary) == list(SUMMARY_KEYS) and set(PARAM_KEYS) == set(r.params)
        assert all(list(s) == list(STATE_KEYS) for s in r.steps)


def test_random_numbers_are_the_raw_pcg64_stream():
    """Stage 1 draws from the PCG64 bit stream of the seed, turned into fractions here and not by
    ``Generator``; pinned, and equal to what ``default_rng`` gives on the numpy this was built with."""
    p = {**random_params(random.Random(3)), "capacity": 16}
    world = S._world(3, p)
    group = S._Group(world, 4, 3, 0)
    assert group._uniform(3).tolist() == np.random.default_rng(8 * 3 + 5).random(3).tolist()
    for seed, index in ((0, 0), (3, 1), (9998, 0)):
        ours, numpys = S._Group(world, 4, seed, index), np.random.default_rng(8 * seed + 5 + index)
        assert ours._uniform(40).tolist() == numpys.random(40).tolist()
        assert ours._uniform(7).dtype == np.float64 and 0.0 <= ours._uniform(1)[0] < 1.0
    assert S._Group(world, 4, 3, 0)._uniform(2).tolist() == [0.05004697035406891, 0.5063222985714494]


def test_random_params_stay_in_range():
    seen = {"silent": 0, "all": 0, "none": 0, "split": 0}
    for seed in range(400):
        p = random_params(random.Random(seed))
        assert p == random_params(random.Random(seed)) == random_params(random.Random(seed), rules=7)
        assert list(p) == list(INPUT_KEYS)
        assert 4 <= p["meanings"] <= 12 and 2 <= p["consonants"] <= 8 and 2 <= p["vowels"] <= 5
        assert 8 <= p["population"] <= 60 and 0 <= p["predator_pressure"] <= 1 and 0 <= p["ears_voice"] <= 1
        assert p["neurons"] in (32, 64, 128, 256, 512, 1024) and (p["bottleneck"] == 0 or 6 <= p["bottleneck"] <= 192)
        assert p["split"] in (0, 1) and 4 <= p["things"] <= 6 and 3 <= p["properties"] <= 4
        seen["silent"] += p["ears_voice"] == 0
        seen["all"] += p["ears_voice"] == 1
        seen["none"] += p["bottleneck"] == 0
        seen["split"] += p["split"]
    assert 20 <= seen["silent"] <= 70 and 200 <= seen["all"] <= 280 and 35 <= seen["none"] <= 90
    assert 90 <= seen["split"] <= 150


def test_run_rejects_impossible_parameters(sim):
    for bad in ({"meanings": 1}, {"meanings": 13}, {"consonants": 0}, {"consonants": 9}, {"vowels": 6},
                {"population": 61}, {"population": -1}, {"predator_pressure": 1.5}, {"predator_pressure": -0.1},
                {"ears_voice": 2.0}, {"ears_voice": float("nan")}, {"neurons": -8}, {"bottleneck": -1},
                {"split": 2}, {"things": 1}, {"things": 7}, {"properties": 5}, {"properties": 1},
                {"population": 12.5}, {"meanings": True}, {"rules": 6}, {"bottleneck": 1001},
                {"bottleneck": 10000}):
        with pytest.raises(ValueError):
            sim.run(1, **bad)
    for bad_seed in (-1, 1.5, "7", S.MAX_SEED + 1):
        with pytest.raises(ValueError):
            sim.run(bad_seed)
    with pytest.raises(TypeError):
        sim.run(1, disc_mass=3.0)
    assert sim.run(1, population=10, rules=7).steps == sim.run(1, population=10).steps


# ---------------------------------------------------------------------------------------------
# lines


def test_pinned_lines_of_seed_87(sim):
    """The lines the module docstring shows, word for word: another numpy or another machine
    must not change a rollout unnoticed."""
    r = sim.rollout(87)
    texts = [ln.text for ln in lines(r)]
    assert texts[0] == ("signals seed 8 7 params. meanings 7. consonants 8. vowels 2. population 2 5. "
                        "predator_pressure 0 point 7 5. ears_voice 1. neurons 1 0 2 4. bottleneck 1 9. split 0. "
                        "things 4. properties 4. syllables 1 6. possible_words 4 3 6 8. capacity 1 2 8. "
                        "talkers 2 5. bits 2 point 8 1. pairs 1 6. coverage 0 point 7 5 5.")
    assert texts[1] == "signals seed 8 7 inventory. consonants m p t n s k l b. vowels i e."
    assert texts[2] == ("signals seed 8 7 meanings. named predator water shelter rival storm fire mate. "
                        "things predator water shelter rival. properties far small big near. "
                        "unseen predator small shelter far.")
    for text in (
            "signals seed 8 7 step 0. success 0 point 1 4 3. alarm 0 point 1 4 3. words 0. synonyms 0. homonyms 0. "
            "word_length 0. meanings 0. compositional 0. order none. dialects 0. stage silent.",
            "signals seed 8 7 step 1 2. success 0 point 4 3 9. alarm 0 point 9 9 2. words 4. synonyms 0. "
            "homonyms 0. word_length 1. meanings 3. compositional 0. order none. dialects 0. stage calls.",
            "signals seed 8 7 step 2 4. success 0 point 9 7 3. alarm 1. words 1 0. synonyms 3. homonyms 0. "
            "word_length 2 point 2. meanings 7. compositional 0. order none. dialects 0. stage lexicon.",
            "signals seed 8 7 step 4 5. success 0 point 7 8 8. alarm 1. words 1 7. synonyms 4. homonyms 0. "
            "word_length 1 point 8 8. meanings 2 2. compositional 0 point 5. order thing_first. dialects 0. "
            "stage grammar.",
            "signals seed 8 7 step 6 0. success 0 point 9 8 1. alarm 1. words 1 1. synonyms 0. homonyms 0. "
            "word_length 2. meanings 2 3. compositional 1. order thing_first. dialects 0. stage language.",
            "q signals seed 8 7 step 1 2 success. a 0 point 4 3 9.",
            "q signals seed 8 7 step 2 0 next stage. a words.",
            "q signals seed 8 7 final outcome. a language.",
            "q signals seed 8 7 final topographic. a 0 point 7 5 7.",
            "signals seed 8 7 step 2 0 lexicon. water ne. shelter ti. predator le. rival se.",
            "q signals seed 8 7 step 2 0 word predator. a le.",
            "q signals seed 8 7 step 2 0 word storm. a none.",
            "signals seed 8 7 step 4 0 lexicon. water ne. shelter ti. predator le. rival se. storm be se be. "
            "fire ti ni. mate si si me.",
            "q signals seed 8 7 step 4 0 meaning be se be. a storm.",
            "signals seed 8 7 lexicon. water ne. shelter ti. predator le. rival se. storm be se be. fire ti ni. "
            "mate si si me. far be me ti.",
            "signals seed 8 7 lexicon. small le li. big se le ki. near ti ne.",
            "signals seed 8 7 grammar. order thing_first. compositional 1.",
            "q signals seed 8 7 word near. a ti ne.",
            "q signals seed 8 7 say predator near. a le ti ne.",
            "q signals seed 8 7 meaning le ti ne. a predator near.",
            "q signals seed 8 7 say predator small. a le le li.",
            "q signals seed 8 7 say shelter far. a ti be me ti.",
            "q signals seed 8 7 order. a thing_first."):
        assert text in texts, text
    assert "signals seed 9 1 phrases. sky_danger far me." in [ln.text for ln in lines(sim.rollout(91))]


def test_every_line_is_dense_and_short(all_lines):
    assert len(all_lines) > 50000
    for ln in all_lines:
        assert is_dense(ln.text), ln.text
        assert n_tokens(ln.text) <= 128, ln.text
        assert ln.topic == "signals"
        if ln.kind != "record":
            assert "." not in ln.prompt and "." not in ln.answer
            assert split_line(ln.text) == (ln.prompt, ln.answer)
            assert ln.kind in ("fact", "calc")
    assert max(n_tokens(ln.text) for ln in all_lines) <= 100


def test_lines_of_extreme_runs_stay_dense_and_short(sim):
    for seed, params in ((9998, dict(meanings=12, consonants=8, vowels=5, population=60, things=6, properties=4,
                                     neurons=1024, bottleneck=0)),
                         (123456789, dict(meanings=12, consonants=2, vowels=2, population=60, things=6,
                                          properties=4, neurons=1024, bottleneck=10, split=1)),
                         (77, dict(meanings=12, consonants=8, vowels=5, population=59, things=6, properties=4,
                                   neurons=1024, bottleneck=60, split=1))):
        r = sim.run(seed, **params)
        assert sim.conserved(r).ok
        for ln in lines(r):
            assert is_dense(ln.text) and n_tokens(ln.text) <= 128, ln.text


def test_line_kinds(sim):
    r = sim.rollout(87)
    ls = lines(r)
    head = "signals seed 8 7"
    records = [ln for ln in ls if ln.kind == "record"]
    assert [ln.text.split(". ")[0] for ln in records[:3]] == [head + " params", head + " inventory", head + " meanings"]
    states = [ln for ln in records if re.fullmatch(head + r" step( \d)+", ln.text.split(". ")[0])]
    assert len(states) == STEPS + 1 and all(len(ln.text.split(". ")) == 1 + len(STATE_KEYS) for ln in states)
    prompts = [ln.prompt for ln in ls if ln.kind != "record"]
    assert len(prompts) == len(set(prompts)), "one question, one line"
    step_q = [q for q in prompts if " step " in q and " word " not in q and " meaning " not in q]
    assert len(step_q) == (STEPS + 1) * len(STATE_KEYS) + STEPS * len(STATE_KEYS)
    assert sum(q.startswith(head + " final ") for q in prompts) == len(SUMMARY_KEYS)
    language = r.languages[STEPS]
    pairs = len(language["things"]) * r.params["properties"]
    assert pairs == r.params["pairs"] == 16
    assert sum(q.startswith(head + " say ") for q in prompts) == pairs
    assert head + " order" in prompts
    for step in (20, 40):
        asked = [q for q in prompts if q.startswith(f"{head} step {num(step)} word ")]
        assert len(asked) == r.params["meanings"], "every meaning of the run is asked for, settled or not"
    assert sum(q.startswith(head + " word ") for q in prompts) == r.params["meanings"] + r.params["properties"]
    by_kind = {kind: [ln for ln in records if ln.text.split(". ")[0].endswith(" " + kind)]
               for kind in ("lexicon", "grammar", "phrases")}
    assert len(by_kind["lexicon"]) == 1 + 1 + 2, "steps 20 and 40, and two lines for the eleven words at the end"
    assert len(by_kind["grammar"]) == 1 and not by_kind["phrases"], "a regular language has no phrases to list"
    assert len(ls) == len({ln.text for ln in ls}) and 1400 < len(ls) < 1600
    assert len(lines(r, every=10)) < len(ls) / 4
    # an unseen pair is marked, and is an answer of the rule
    unseen = [ln for ln in ls if ln.meta.get("unseen")]
    assert len(unseen) == len(language["held"]) == 2 and all(ln.kind == "calc" for ln in unseen)
    assert {ln.prompt for ln in unseen} == {head + " say predator small", head + " say shelter far"}
    # a language with irregular phrases lists them, and a silent one has nothing to list
    assert [ln.text for ln in lines(sim.rollout(91)) if " phrases. " in ln.text] == [
        "signals seed 9 1 phrases. sky_danger far me."]


def test_parameters_are_written_as_given(sim, rollouts):
    assert param_value("predator_pressure", 0.57) == "0 point 5 7" and param_value("neurons", 256) == "2 5 6"
    assert param_value("coverage", 0.64412) == "0 point 6 4 4" and param_value("ears_voice", 1.0) == "1"
    for seed, r in rollouts.items():
        p = random_params(random.Random(seed))
        for key in INPUT_KEYS:
            assert r.params[key] == p[key]
            assert parse_num(param_value(key, r.params[key])) == p[key], (seed, key)
        for key in PARAM_KEYS:
            v = sim.check(f"signals seed {num(seed)} params {key}", param_value(key, r.params[key]))
            assert v.ok and v.expected == param_value(key, r.params[key]), (seed, key)
    assert not sim.check("signals seed 8 7 params population", "2 6").ok
    assert sim.check("signals seed 8 7 params population", "2 6").expected == "2 5"


def test_numbers_use_three_significant_digits(all_lines):
    assert dense_value(0.98512) == "0 point 9 8 5" and dense_value(16) == "1 6" and dense_value("calls") == "calls"
    assert dense_value(0.0) == "0" and dense_value(1.0) == "1" and dense_value(2.2449) == "2 point 2 4"
    for ln in all_lines:
        words = ln.prompt.split()
        if ln.kind == "record" or words[-1] in WORD_KEYS or {"word", "meaning", "say", "order"} & set(words[-7:]):
            continue
        if words[-2] == "params":
            continue
        value = parse_num(ln.answer)
        assert value is not None, ln.text
        digits = "".join(w for w in ln.answer.split(" e ")[0].split() if w.isdigit())
        assert len(digits.strip("0")) <= 3, ln.text


# ---------------------------------------------------------------------------------------------
# the gate of the questions


def test_gate_agrees_with_every_line(sim, all_lines):
    for ln in all_lines:
        if ln.kind == "record":
            continue
        assert sim.owns(ln.prompt) and S.owns(ln.prompt), ln.prompt
        v = sim.check(ln.prompt, ln.answer)
        assert v.ok and v.expected == ln.answer, (ln.text, v)
        assert S.check(ln.prompt, ln.answer).ok


def test_gate_agrees_with_every_state_record(sim, all_lines):
    for ln in all_lines:
        head, *fields = [f.strip() for f in ln.text.rstrip(".").split(".")]
        if ln.kind != "record" or head.split()[-1] in ("inventory", "meanings", "lexicon", "grammar", "phrases"):
            continue
        for f in fields:
            key, _, value = f.partition(" ")
            v = sim.check(f"{head} {key}", value)
            assert v.ok and v.expected == value, (ln.text, key)


def test_the_language_records_say_what_the_questions_say(sim, rollouts):
    for seed, r in rollouts.items():
        for ln in lines(r):
            head, *fields = [f.strip() for f in ln.text.rstrip(".").split(".")]
            if ln.kind != "record":
                continue
            kind = head.split()[-1]
            base = head.rsplit(" ", 1)[0]
            if kind == "lexicon":
                for f in fields:
                    name, _, form = f.partition(" ")
                    assert sim.check(f"{base} word {name}", form).ok, ln.text
                    assert sim.check(f"{base} meaning {form}", name).ok, ln.text
            elif kind == "phrases":
                for f in fields:
                    thing, prop, *form = f.split()
                    assert sim.check(f"{base} say {thing} {prop}", " ".join(form)).ok, ln.text
            elif kind == "grammar":
                assert sim.check(f"{base} order", fields[0].split()[1]).ok
                assert sim.check(f"{base} final compositional", fields[1].partition(" ")[2]).ok


def wrong_answer(sim, ln) -> str:
    words = ln.prompt.split()
    key = words[-1]
    accepted = sim.answers(ln.prompt)
    if accepted is not None:
        kind = next(w for w in ("word", "meaning", "say", "order") if w in words)
        if kind == "order":
            return next(o for o in ORDERS + ("none",) if o not in accepted)
        if kind == "meaning":
            return next(m for m in VOCABULARY + ("none",) if m not in accepted)
        return next(u for u in ("none", "ka", "ka ka ka ka ka ka ka") if u not in accepted)
    if key in WORD_KEYS:
        return next(w for w in STAGES + OUTCOMES + ORDERS + ("none",) if w != ln.answer and (
            w in STAGES if key == "stage" else w in OUTCOMES if key == "outcome" else w in ORDERS + ("none",)))
    return "9 " + ln.answer if not ln.answer.startswith("minus") else ln.answer.replace("minus", "minus 9", 1)


def test_gate_rejects_altered_answers(sim, all_lines):
    rng = random.Random(5)
    questions = [ln for ln in all_lines if ln.kind != "record"]
    for ln in rng.sample(questions, 4000):
        wrong = wrong_answer(sim, ln)
        v = sim.check(ln.prompt, wrong)
        assert not v.ok and v.expected == ln.answer, (ln.text, wrong, v)
    spoken = [ln for ln in questions if sim.answers(ln.prompt) is not None]
    assert len(spoken) > 2000
    for ln in spoken:
        v = sim.check(ln.prompt, wrong_answer(sim, ln))
        assert not v.ok and v.expected == ln.answer, ln.text
        assert not sim.check(ln.prompt, ln.answer + " ka").ok and not sim.check(ln.prompt, "").ok
        assert not sim.check(ln.prompt, " " + ln.answer).ok


def test_gate_rejects_a_changed_digit(sim, all_lines):
    """One digit changed: a count is then wrong; another number is wrong when it is more than
    5 percent off."""
    rng = random.Random(6)
    numeric = [ln for ln in all_lines if ln.kind != "record" and sim.answers(ln.prompt) is None
               and ln.prompt.split()[-1] not in WORD_KEYS]
    rejected = 0
    for ln in rng.sample(numeric, 3000):
        words = ln.answer.split()
        place = rng.choice([i for i, w in enumerate(words) if w.isdigit()])
        words[place] = str((int(words[place]) + rng.randint(1, 9)) % 10)
        wrong = " ".join(words)
        truth, got = parse_num(ln.answer), parse_num(wrong)
        v = sim.check(ln.prompt, wrong)
        assert v.expected == ln.answer
        off = abs(got - truth) / max(abs(got), abs(truth))
        if ln.prompt.split()[-1] in COUNT_KEYS or off > 0.06:
            assert not v.ok, (ln.text, wrong, v)
        elif off < 0.04:
            assert v.ok, (ln.text, wrong, v)
        rejected += not v.ok
    assert rejected > 2500


def test_gate_rejects_answers_that_are_not_finite_numbers(sim):
    for prompt in ("signals seed 8 7 final success", "signals seed 8 7 step 3 0 word_length",
                   "signals seed 8 7 final words", "signals seed 8 7 params coverage"):
        for answer in ("1 e 9 9 9", "minus 1 e 9 9 9", "", "point", "e", "yes", "0 point 9 8 1 or so", "0 . 9"):
            v = sim.check(prompt, answer)
            assert not v.ok and v.expected is not None, (prompt, answer, v)
    assert sim.check("signals seed 8 7 final success", "1 e 9 9 9").reason == "not a number"
    assert sim.check("signals seed 8 7 final success", "0 point 9 8 1").ok
    assert sim.check("signals seed 8 7 final success", "0 point 9 6").ok, "within 5 percent"
    assert not sim.check("signals seed 8 7 final success", "0 point 9").ok
    assert sim.check("signals seed 8 7 final words", "1 1").ok and not sim.check("signals seed 8 7 final words", "1 2").ok
    assert not sim.check("signals seed 8 7 step 0 words", "1").ok and sim.check("signals seed 8 7 step 0 words", "0").ok


def test_say_accepts_every_utterance_of_the_language_and_meaning_every_meaning(sim, rollouts):
    two_ways = homonyms = 0
    for seed, r in rollouts.items():
        language = r.languages[STEPS]
        head = f"signals seed {num(seed)}"
        for (thing, prop), utterance in language["say"].items():
            prompt = f"{head} say {thing} {prop}"
            accepted = sim.answers(prompt)
            assert accepted[0] == text_of(utterance) and 1 <= len(accepted) <= 2
            rule = language["rule"].get((thing, prop))
            if rule is not None and rule != utterance:
                two_ways += 1
                assert accepted == [text_of(utterance), text_of(rule)]
            for answer in accepted:
                assert sim.check(prompt, answer).ok and sim.check(prompt, answer).expected == accepted[0]
                # every utterance leads back to its meaning
                assert sim.check(f"{head} meaning {answer}", f"{thing} {prop}").ok, (seed, thing, prop, answer)
        forms = set(language["lexicon"].values()) | set(language["properties"].values()) | set(language["say"].values())
        for form in forms:
            found = S.meanings_of(language, form)
            assert found == sim.answers(f"{head} meaning {text_of(form)}")
            homonyms += len(found) > 1
            for meaning in found:
                assert sim.check(f"{head} meaning {text_of(form)}", meaning).ok
    assert two_ways >= 5, "irregular phrases beside the rule occur in these seeds"
    assert homonyms >= 1, "and so do utterances with two meanings"
    assert sim.answers("signals seed 8 7 meaning ka ka ka ka ka ka") == ["none"]
    assert sim.check("signals seed 8 7 meaning ka ka ka ka ka ka", "none").ok
    assert sim.answers("signals seed 8 7 say night near") == ["none"], "night is no meaning of seed 87"
    assert sim.answers("signals seed 8 7 word night") == ["none"] and sim.answers("signals seed 8 7 final words") is None


def test_an_unseen_pair_is_said_by_the_rule(sim, rollouts):
    """No agent ever heard or remembered the pairs that are never taught; a compositional language
    says them all the same, a holistic one cannot."""
    by_rule = cannot = 0
    for seed, r in rollouts.items():
        language = r.languages[STEPS]
        if not language["things"]:
            continue
        things, props = language["things"], r.world["properties"]
        pairs = [(t, p) for t in things for p in props]
        assert len(language["held"]) == (1 if len(pairs) < 16 else 2)
        held_index = {pairs.index(key) for key in language["held"]}
        for states in r.states[41:]:
            for memory, _, _ in states[0]["agents"]:
                assert not held_index & set(memory), "a pair that is never taught is never remembered"
        for thing, prop in language["held"]:
            prompt = f"signals seed {num(seed)} say {thing} {prop}"
            answer = sim.answers(prompt)
            if language["order"] != "none" and prop in language["properties"]:
                word = sim.answers(f"signals seed {num(seed)} word {thing}")[0]
                morph = sim.answers(f"signals seed {num(seed)} word {prop}")[0]
                assert answer == [f"{word} {morph}" if language["order"] == "thing_first" else f"{morph} {word}"]
                by_rule += 1
            else:
                assert answer == ["none"]
                cannot += 1
    assert by_rule >= 10 and cannot >= 5


@pytest.mark.parametrize("prompt", [
    "turkey capital", "q spoon color", "carbon protons", "calc 1 2 plus 7", "check 1 2 plus 7 equals 2 0",
    "gravity seed 7 step 2 0 clumps", "nucleo seed 3 final helium_fraction", "world seed 3 era planets",
    "planets seed 7 final gas", "chem seed 7 final water", "life seed 7 final stage", "life seed 7 step 1 2 next population",
    "senses seed 3 step 4 0 eyed", "senses seed 3 final outcome", "senses predict escape head_start 3 speed 2 predator_speed 6",
    "stars seed 3 final stage", "cells seed 3 final stage", "bodies seed 3 final stage", "society seed 3 final stage",
    "society predict hamilton relatedness 0 point 5 benefit 4 cost 1", "world7 seed 3 final stage",
    "signals", "signals seed 7", "signals seed 7 step 3", "signals seed 7 step 3 clumps", "signals seed 7 step 6 1 success",
    "signals seed 7 step 6 0 next success", "signals seed 7 final alarm", "signals seed x final success",
    "signals seed 7 final", "signals seed 7 params success", "signals seed 12 step 3 success",
    "signals seed 7 step 3 next", "signals seed 7 step 3 success words", "", "signals seed 0 7 final success",
    "signals seed 7 step 0 3 success", "signals seed minus 7 final success", "signals seed 7 point 5 final success",
    "q signals seed 7 final success", "signals seed 7 final success extra", "signals  seed 7 final success",
    "signals seed 7 final success ", "signals seed 7 step 3 next next success", "signals seed 7 word",
    "signals seed 7 word ka", "signals seed 7 word food water", "signals seed 7 meaning", "signals seed 7 meaning food",
    "signals seed 7 meaning ka ka ka ka ka ka ka", "signals seed 7 meaning xa", "signals seed 7 say food",
    "signals seed 7 say near food", "signals seed 7 say food near far", "signals seed 7 order thing_first",
    "signals seed 7 step 2 0 say food near", "signals seed 7 step 2 0 next order x", "signals seed 7 step 6 0 word food",
    "signals seed 7 step 1 0 word food", "signals seed 7 step 2 0 final success", "signals seed 7 step 2 0 params meanings",
    "signals seed 7 lexicon", "signals seed 7 grammar", "signals seed 7 phrases", "signals predict nonsense signals 4",
    "signals predict expected_success signals 4 states 4 shared 5", "signals predict syllables consonants 4",
    "signals predict zipf_frequency rank 5 meanings 4", "signals predict signal_bits signals 0",
    "signals predict coverage meanings 1 2 bottleneck 0", "signals predict coverage meanings 1 bottleneck 5",
    "signals predict coverage meanings 1 2 bottleneck 2 0 1", "signals predict coverage meanings 1 2",
    "signals seed 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 final success",
    "signals seed 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 final success",
    "signals seed 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 say predator near",
])
def test_does_not_own_other_prompts(sim, prompt):
    assert not S.owns(prompt) and not sim.owns(prompt)
    for check in (S.check, sim.check):
        v = check(prompt, "1")
        assert v.ok is False and v.expected is None and v.reason == "not my question"


@pytest.mark.parametrize("prompt", [
    "signals seed 7 step 0 success", "signals seed 7 step 6 0 stage", "signals seed 7 step 5 9 next order",
    "signals seed 1 2 3 4 step 1 2 next word_length", "signals seed 7 final topographic", "signals seed 7 final outcome",
    "signals seed 7 params coverage", "signals seed 0 final success", "signals seed 7 word food",
    "signals seed 7 word near", "signals seed 7 step 2 0 word predator", "signals seed 7 step 4 0 meaning ka ti",
    "signals seed 7 meaning ka", "signals seed 7 say predator near", "signals seed 7 order",
    "signals seed 7 step 1 0 words", "signals seed 7 step 1 0 meanings",
])
def test_owns_its_prompts(sim, prompt):
    assert sim.owns(prompt) and S.owns(prompt)
    assert sim.check(prompt, "nonsense words").expected is not None


def test_module_check_answers_seed_questions_and_lessons(sim):
    assert S.check("signals seed 8 7 say predator near", "le ti ne").ok
    assert S.check("signals predict expected_success signals 4 states 4 shared 3", "0 point 7 6 5 6").ok
    assert not S.check("signals predict expected_success signals 4 states 4 shared 3", "0 point 7 5").ok
    assert S.owns("signals predict naming_max_words agents 6 0") and not sim.owns("signals predict naming_max_words agents 6 0")
    assert S.check("signals predict naming_max_words agents 6 0", "3 0").ok


# ---------------------------------------------------------------------------------------------
# the gate of the simulation


def test_conserved_passes_on_forty_seeds_and_on_extreme_worlds(sim, rollouts):
    for r in rollouts.values():
        v = sim.conserved(r)
        assert v.ok and v.reason.startswith("weights, bounds"), (r.seed, v)
    for params in (dict(population=0), dict(population=1), dict(population=2), dict(population=3, split=1),
                   dict(meanings=2), dict(consonants=1, vowels=1), dict(consonants=1, vowels=2), dict(neurons=0),
                   dict(neurons=8), dict(neurons=16), dict(neurons=40), dict(neurons=64), dict(neurons=96),
                   dict(bottleneck=1), dict(bottleneck=0), dict(things=2, properties=2), dict(ears_voice=0.05),
                   dict(consonants=2, vowels=1, meanings=12), dict(population=5, split=1, ears_voice=0.6)):
        for seed in (1, 2):
            r = sim.run(seed, **params)
            assert sim.conserved(r).ok, (params, seed, sim.conserved(r))


def test_conserved_catches_tampering(sim, rollouts):
    base = rollouts[21]  # one group, calls, then words, then a grammar with irregular phrases
    assert [s[0]["mode"] for s in base.states][::20] == ["calls", "calls", "naming", "grammar"]
    assert base.steps[30]["synonyms"] > 0 and base.steps[50]["order"] == "thing_first"
    assert 0 < base.steps[50]["compositional"] < 1 and base.steps[12]["stage"] == "calls"

    def broken(change) -> bool:
        r = copy.deepcopy(base)
        change(r)
        return not sim.conserved(r).ok

    assert not broken(lambda r: None)
    # the metrics are those of the agents
    assert broken(lambda r: r.steps[10].update(success=r.steps[10]["success"] + 0.01))
    assert broken(lambda r: r.steps[30].update(words=r.steps[30]["words"] + 1))
    assert broken(lambda r: r.steps[30].update(synonyms=0))
    assert broken(lambda r: r.steps[50].update(compositional=1.0))
    assert broken(lambda r: r.steps[50].update(order="property_first"))
    assert broken(lambda r: r.steps[12].update(stage="language"))
    assert broken(lambda r: r.steps[0].update(success=0.5))
    assert broken(lambda r: r.steps.pop())
    # association weights are never negative, and the success is the one they give
    def negative(r):
        r.states[10][0]["A"][0, 0, 0] = -1.0
    assert broken(negative)

    def heavier(r):
        r.states[10][0]["A"][:, 0, 0] += 50.0
    assert broken(heavier)
    # a talker may not use one form for two meanings; no more words than half the talkers
    def two_meanings(r):
        inv = r.states[30][0]["inventories"][0]
        a, b = list(inv)[:2]
        inv[b] = inv[a]
    assert broken(two_meanings)

    def crowd(r):
        inv = r.states[30][0]["inventories"]
        syl = r.world["syllables"]
        m = next(iter(inv[0]))
        for i, talker in enumerate(inv):
            talker[m] = talker.get(m, ()) + ((syl[i % len(syl)], syl[(i // len(syl)) % len(syl)], syl[0]),)
    assert broken(crowd)
    assert broken(lambda r: r.states[30][0]["inventories"][0].update({0: (("xx",),)}))
    # the lexicon is the word of the majority; lexicon, grammar and phrases fit each other
    def other_word(r):
        r.languages[40]["lexicon"][next(iter(r.languages[40]["lexicon"]))] = ("zz",)
    assert broken(other_word)

    def other_order(r):
        r.languages[60]["order"] = "property_first"
    assert broken(other_order)

    def other_phrase(r):
        key = next(iter(r.languages[60]["say"]))
        r.languages[60]["say"][key] = r.languages[60]["say"][key] + r.languages[60]["say"][key]
    assert broken(other_phrase)
    assert broken(lambda r: r.languages.pop(40))

    def forget(r):
        for memory, _, _ in r.states[60][0]["agents"]:
            memory.clear()
    assert broken(forget)

    def remember_too_much(r):
        memory = r.states[60][0]["agents"][0][0]
        for x in range(200):
            memory[x] = (r.world["syllables"][0],)
    assert broken(remember_too_much)
    # parameters, world and summary
    assert broken(lambda r: r.params.update(capacity=r.params["capacity"] + 1))
    assert broken(lambda r: r.params.update(possible_words=10))
    assert broken(lambda r: r.params.update(syllables=3))
    assert broken(lambda r: r.params.update(talkers=2))
    assert broken(lambda r: r.params.update(pairs=24))
    assert broken(lambda r: r.params.update(coverage=0.5))
    assert broken(lambda r: r.params.update(bits=1.0))
    assert broken(lambda r: r.summary.update(outcome="language"))
    assert broken(lambda r: r.summary.update(topographic=0.99))
    assert broken(lambda r: r.summary.update(words=r.summary["words"] + 1))
    # a rollout without its agents cannot be judged
    bare = S.Rollout("signals", 21, dict(base.params), copy.deepcopy(base.steps), dict(base.summary))
    assert not sim.conserved(bare).ok
    # a silent population may not have signals
    silent = sim.run(3, ears_voice=0.0)
    assert sim.conserved(silent).ok
    loud = copy.deepcopy(silent)
    loud.steps[5].update(words=1)
    assert not sim.conserved(loud).ok


def test_the_language_gate_counts_the_majority_again(rollouts):
    """The lexicon in the records is the word of the majority, and lexicon, grammar and phrases fit
    each other: checked by a second count, talker by talker."""
    r = rollouts[21]
    assert set(r.languages) == set(SNAPSHOTS) == {20, 40, 60}
    for seed in SEEDS:
        for t, states in enumerate(rollouts[seed].states):
            language = S._language(view(states[0], rollouts[seed].world, full=False))
            assert S._language_fault(language, states[0], rollouts[seed].world) is None, (seed, t)
    for step in (20, 40):  # calls, then words
        language = copy.deepcopy(r.languages[step])
        name = next(iter(language["lexicon"]))
        language["lexicon"][name] = (r.world["syllables"][-1], r.world["syllables"][-1], r.world["syllables"][-1])
        assert S._language_fault(language, r.states[step][0], r.world) == \
            f"the word for {name} is not the word of the majority"
    language = copy.deepcopy(r.languages[40])
    name = next(iter(language["lexicon"]))
    del language["lexicon"][name]
    assert S._language_fault(language, r.states[40][0], r.world) == \
        f"the majority has a word for {name} that the lexicon lacks"
    final, state = r.languages[60], r.states[60][0]
    assert final["order"] == "thing_first" and final["properties"] and 0 < final["compositional"] < 1
    language = copy.deepcopy(final)
    language["compositional"] = 1.0
    assert "compositional share" in S._language_fault(language, state, r.world)
    language = copy.deepcopy(final)
    language["order"] = "none"
    assert S._language_fault(language, state, r.world) == "a rule without an order"
    language = copy.deepcopy(final)
    language["order"] = "property_first"
    assert S._language_fault(language, state, r.world).startswith("the rule is not the word of the thing")
    language = copy.deepcopy(final)
    name = next(iter(language["lexicon"]))
    language["lexicon"][name] = language["lexicon"][name] + language["lexicon"][name]
    assert S._language_fault(language, state, r.world) == "the lexicon is not the one the learners hold"
    language = copy.deepcopy(final)
    prop = next(iter(language["properties"]))
    other = (r.world["syllables"][0],) * 3
    for key in language["rule"]:
        if key[1] == prop:  # the rule and what is said follow the changed word, the talkers do not
            language["say"][key] = language["rule"][key] = language["lexicon"][key[0]] + other
    language["properties"][prop] = other
    fault = S._language_fault(language, state, r.world)
    assert fault in (f"the word for {prop} is not the word of the majority",
                     "the compositional share is not the share of the taught pairs that follow the rule")


def test_weights_never_fall_below_their_start(rollouts):
    seen = 0
    for r in rollouts.values():
        for states in r.states:
            for state in states:
                if state["mode"] == "calls":
                    seen += 1
                    A = state["A"]
                    assert A.dtype == np.float64 and A.min() >= 1.0 and bool((A == np.round(A)).all())
        assert all(float(s[0]["A"].max()) == 1.0 for s in r.states[:1] if s[0]["mode"] == "calls")
    assert seen > 500


def brute_single(state: dict, world: dict) -> float:
    """Expected accuracy among the talkers of a group, pair by pair and meaning by meaning."""
    n, named, guess = len(world["meanings"]), world["named"], 1.0 / len(world["meanings"])
    total = 0.0
    if state["mode"] == "calls":
        A = state["A"]
        T = A.shape[0]
        send = A / A.sum(axis=2, keepdims=True)
        recv = A / A.sum(axis=1, keepdims=True)
        for i in range(T):
            for j in range(T):
                if i != j:
                    for m in range(n):
                        total += float((send[i, m] * recv[j, m]).sum()) if m in named else guess
        return total / (T * (T - 1) * n)
    inventories = state["inventories"]
    T = len(inventories)
    owner = [{form: m for m, words in inv.items() for form in words} for inv in inventories]
    for i in range(T):
        for j in range(T):
            if i == j:
                continue
            for m in range(n):
                words = inventories[i].get(m)
                if not words:
                    total += guess
                    continue
                least = min(len(w) for w in words)
                short = [w for w in words if len(w) == least]
                for form in short:
                    heard = owner[j].get(form)
                    total += (guess if heard is None else float(heard == m)) / len(short)
    return total / (T * (T - 1) * n)


def brute_pairs(state: dict, world: dict) -> float:
    lexicon, agents, things = state["lexicon"], state["agents"], state["things"]
    pairs = [(t, p) for t in things for p in range(len(world["properties"]))]
    m, T = len(pairs), len(agents)

    def say(agent, x):
        memory, morph, order = agent
        if x in memory:
            return memory[x]
        t, p = pairs[x]
        if p not in morph:
            return None
        return lexicon[t] + morph[p] if order == "thing_first" else morph[p] + lexicon[t]

    def readings(agent, utterance):
        memory, morph, order = agent
        found = {x for x, u in memory.items() if u == utterance}
        for x, (t, p) in enumerate(pairs):
            if p in morph and (lexicon[t] + morph[p] if order == "thing_first" else morph[p] + lexicon[t]) == utterance:
                found.add(x)
        return found

    total = 0.0
    for i in range(T):
        for j in range(T):
            if i == j:
                continue
            for x in range(m):
                utterance = say(agents[i], x)
                found = readings(agents[j], utterance) if utterance is not None else set()
                total += 1.0 / m if not found else (1.0 / len(found) if x in found else 0.0)
    return total / (T * (T - 1) * m)


def test_success_is_computed_exactly_from_the_agents(sim):
    """The success in a step is the expected accuracy over every speaker, every other hearer and
    every meaning, recomputed here pair by pair from the weight tables, inventories and learners."""
    modes = set()
    for seed, params in ((1, dict(population=14)), (2, dict(population=9, meanings=5, consonants=2, vowels=2)),
                         (3, dict(population=12, bottleneck=30)), (4, dict(population=10, neurons=32)),
                         (5, dict(population=16, split=1, bottleneck=0)), (6, dict(population=11, ears_voice=0.6))):
        r = sim.run(seed, **params)
        world = r.world
        agents, talkers = S._sizes(r.params["population"], r.params["split"], r.params["ears_voice"])[0]
        can = talkers * (talkers - 1) / (agents * (agents - 1))
        for t in (0, 3, 11, 20, 21, 24, 33, 40, 41, 42, 47, 60):
            state = r.states[t][0]
            modes.add(state["mode"])
            if state["mode"] == "grammar":
                among, guess = brute_pairs(state, world), 1.0 / (len(state["things"]) * len(world["properties"]))
            else:
                among, guess = brute_single(state, world), 1.0 / len(world["meanings"])
            assert view(state, world)["success"] == pytest.approx(among, abs=1e-12), (seed, t)
            assert r.steps[t]["success"] == sig(can * view(state, world)["success"] + (1 - can) * guess)
            assert r.steps[t]["success"] == pytest.approx(can * among + (1 - can) * guess, rel=6e-3)
    assert modes == {"calls", "naming", "grammar"}


def test_stage_follows_from_the_numbers(rollouts):
    assert S.stage_of("silent", 0, 8, 0.0) == "silent" and S.stage_of("calls", 0, 8, 0.0) == "silent"
    assert S.stage_of("calls", 2, 8, 0.0) == "calls" and S.stage_of("naming", 3, 8, 0.0) == "words"
    assert S.stage_of("naming", 8, 8, 0.0) == "lexicon" and S.stage_of("grammar", 8, 8, 0.0) == "lexicon"
    assert S.stage_of("grammar", 8, 8, 0.3) == "grammar" and S.stage_of("grammar", 8, 8, 0.9) == "language"
    assert [S.outcome_of(s) for s in STAGES] == ["silent", "calls", "lexicon", "lexicon", "lexicon", "language"]
    seen = set()
    for r in rollouts.values():
        nameable = len(r.world["named"])
        for step, states in zip(r.steps, r.states):
            v = view(states[0], r.world, full=False)
            assert step["stage"] == S.stage_of(states[0]["mode"], v["settled"], nameable, v["compositional"])
            assert step["order"] in ORDERS + ("none",) and step["stage"] in STAGES
            assert (step["order"] == "none") == (step["compositional"] == 0) or step["compositional"] == 0
            seen.add(step["stage"])
        assert r.steps[0]["stage"] == "silent" and r.summary["stage"] == r.steps[-1]["stage"]
        assert r.summary["outcome"] == S.outcome_of(r.summary["stage"]) and r.summary["outcome"] in OUTCOMES
        modes = [s[0]["mode"] for s in r.states]
        order = ["silent", "calls", "naming", "grammar"]
        assert all(order.index(b) >= order.index(a) for a, b in zip(modes, modes[1:])), "the games follow each other"
        assert all(m in ("calls", "silent") for m in modes[:21]), "stage 1 lasts to step 20"
        assert "grammar" not in modes[:41], "stage 3 begins at step 41"
    assert seen == set(STAGES)
    assert {r.summary["outcome"] for r in rollouts.values()} == set(OUTCOMES)


# ---------------------------------------------------------------------------------------------
# lessons


def lesson_inputs(prompt: str) -> tuple[str, list[int]]:
    words = prompt.split()
    rule = RULES[words[2]]
    names = [s.name for s in rule.inputs]
    values, pos = [], 3
    for name in names:
        assert words[pos] == name
        end = pos + 1
        while end < len(words) and words[end] not in names:
            end += 1
        values.append(parse_num(words[pos + 1:end]))
        pos = end
    return words[2], values


def test_lessons_are_accepted_and_equal_the_functions_of_the_simulation():
    functions = {"expected_success": S.expected_success, "signal_bits": S.signal_bits, "syllables": S.syllables,
                 "possible_words": S.possible_words, "holistic_words": S.holistic_words,
                 "compositional_words": S.compositional_words, "coverage": S.coverage,
                 "zipf_frequency": S.zipf_frequency, "naming_max_words": S.naming_max_words,
                 "lexicon_capacity": S.lexicon_capacity}
    assert set(functions) == set(RULES) and all(RULES[k].compute is f for k, f in functions.items())
    lessons = LESSONS.generate(random.Random(1), 400)
    assert len(lessons) == 400 and [ln.text for ln in lessons] == [ln.text for ln in LESSONS.generate(random.Random(1), 400)]
    assert [ln.text for ln in lessons] != [ln.text for ln in LESSONS.generate(random.Random(2), 400)]
    seen = set()
    for ln in lessons:
        assert is_dense(ln.text) and n_tokens(ln.text) <= 128 and ln.topic == "predict_signals" and ln.kind == "calc"
        v = LESSONS.check(ln.prompt, ln.answer)
        assert v.ok and v.expected == ln.answer, ln.text
        assert S.owns(ln.prompt) and S.check(ln.prompt, ln.answer).ok
        name, values = lesson_inputs(ln.prompt)
        seen.add(name)
        assert ln.answer == num(functions[name](*values)), ln.text
        assert ln.meta["split_key"] == LESSONS.split_key(ln.prompt)
        truth = parse_num(ln.answer)
        wrong = num(truth + 1) if isinstance(truth, int) else num(truth * 1.5 + 0.25)
        assert not LESSONS.check(ln.prompt, wrong).ok and not LESSONS.check(ln.prompt, "many").ok, ln.text
    assert seen == set(RULES), "400 lessons reach every rule"
    records = LESSONS.records()
    assert len(records) == len(RULES)
    for ln in records:
        assert is_dense(ln.text) and n_tokens(ln.text) <= 128 and ln.kind == "record", ln.text


def test_lesson_values():
    assert S.expected_success(4, 4, 4) == 1.0 and S.expected_success(4, 4, 0) == 0.25
    assert S.expected_success(4, 4, 3) == 0.75 + 0.25 * 0.25 * 0.25 == 0.765625
    assert S.expected_success(2, 8, 2) == 0.25 and S.expected_success(1, 12, 0) == pytest.approx(1 / 12)
    assert S.signal_bits(1) == 0 and S.signal_bits(2) == 1 and S.signal_bits(8) == 3
    assert S.signal_bits(12) == pytest.approx(3.585, abs=1e-3)
    assert S.syllables(8, 5) == 40 and S.syllables(1, 1) == 1
    assert S.possible_words(4, 3) == 4 + 16 + 64 and S.possible_words(40, 3) == 65640 and S.possible_words(1, 3) == 3
    assert S.holistic_words(6, 4) == 24 and S.compositional_words(6, 4) == 10
    assert S.coverage(12, 0) == 0 and S.coverage(1, 5) == 1 and S.coverage(2, 2) == 0.75
    assert S.coverage(11, 12) == pytest.approx(1 - (10 / 11) ** 12)
    assert S.zipf_frequency(1, 1) == 1 and S.zipf_frequency(2, 2) == pytest.approx(1 / 3)
    assert sum(S.zipf_frequency(r, 12) for r in range(1, 13)) == pytest.approx(1.0)
    assert S.naming_max_words(60) == 30 and S.naming_max_words(3) == 1
    assert S.lexicon_capacity(1024) == 128 and S.lexicon_capacity(39) == 4 and S.lexicon_capacity(7) == 0
    assert LESSONS.check("signals predict signal_bits signals 8", "3").ok
    assert LESSONS.check("signals predict possible_words syllables 4 0 length 3", "6 5 6 4 0").ok
    assert LESSONS.check("signals predict coverage meanings 1 1 bottleneck 1 2", "0 point 6 8 1 4").ok
    assert LESSONS.check("signals predict zipf_frequency rank 1 meanings 2", "0 point 6 6 6 7").ok


def test_expected_success_is_what_settled_weight_tables_give():
    """The lesson is the limit of the simulation's own measure: tables in which ``shared``
    meanings have a call with a heavy weight and every other weight is still 1."""
    for n, calls, shared in ((4, 4, 3), (8, 8, 2), (8, 5, 5), (6, 6, 0), (12, 9, 4), (5, 5, 5)):
        p = {"meanings": n, "consonants": 4, "vowels": 3, "properties": 3, "predator_pressure": 0.0, "capacity": 99,
             "bottleneck": 0, "things": 4}
        world = dict(S._world(1, p), calls=calls)
        A = np.ones((7, n, calls))
        for k in range(shared):
            A[:, k, k] = 1e12
        got = view({"mode": "calls", "A": A}, world)["success"]
        assert got == pytest.approx(S.expected_success(calls, n, shared), abs=1e-9), (n, calls, shared)
        if shared == 0:
            assert got == pytest.approx(1 / n, abs=1e-15)


def test_the_simulation_uses_the_lesson_rules(sim, rollouts, trials):
    for seed, r in rollouts.items():
        p, w = r.params, r.world
        # syllables, possible words, capacity, bits
        assert p["syllables"] == S.syllables(p["consonants"], p["vowels"]) == len(set(w["syllables"]))
        assert all(len(s) == 2 and s[0] in w["consonants"] and s[1] in w["vowels"] for s in w["syllables"])
        assert p["possible_words"] == S.possible_words(p["syllables"], MAX_SYLLABLES)
        assert p["capacity"] == S.lexicon_capacity(p["neurons"]) and len(w["named"]) == min(p["meanings"], p["capacity"])
        assert p["bits"] == S.signal_bits(len(w["named"])) and w["guess"] == S.expected_success(1, p["meanings"], 0)
        assert max(s["words"] for s in r.steps) <= p["possible_words"]
        # step 0: nobody has learnt anything, every signal is a guess
        assert r.steps[0]["success"] == sig(S.expected_success(max(1, w["calls"]), p["meanings"], 0)) == sig(1 / p["meanings"])
        # the talk follows Zipf, and predators come up more often under pressure
        raw = [S.zipf_frequency(rank, p["meanings"]) * (1 + S.PRESSURE_GAIN * p["predator_pressure"] * (m == "predator"))
               for rank, m in enumerate(w["meanings"], start=1)]
        assert list(w["weights"]) == pytest.approx([x / sum(raw) for x in raw], rel=1e-12)
        # never more words for a meaning in circulation than half the talkers
        for states in r.states:
            for state in states:
                if state["mode"] == "naming":
                    bound = S.naming_max_words(len(state["inventories"]))
                    for m in w["named"]:
                        assert len({f for inv in state["inventories"] for f in inv.get(m, ())}) <= bound
        # pairs and coverage
        things = r.languages[STEPS]["things"]
        if things:
            assert p["pairs"] == S.holistic_words(len(things), p["properties"])
            taught = p["pairs"] - len(r.languages[STEPS]["held"])
            assert p["coverage"] == (S.coverage(taught, p["bottleneck"]) if p["bottleneck"] else 1.0)
            state = r.states[STEPS][0]
            assert state["room"] == p["capacity"] - (len(state["lexicon"]) - len(things)) \
                - S.compositional_words(len(things), p["properties"])
        else:
            assert p["pairs"] == 0 and p["coverage"] == 0
    quiet = sim.run(1, predator_pressure=0.0)
    assert list(quiet.world["weights"]) == pytest.approx([S.zipf_frequency(r, 8) for r in range(1, 9)], rel=1e-12)
    # two talkers: one word per meaning at any time
    pair = sim.run(2, population=2)
    assert S.naming_max_words(2) == 1 and max(s["synonyms"] for s in pair.steps[21:41]) == 0
    # a holistic language needs a word per pair, a compositional one a word per property more
    for r in trials(bottleneck=0).values():
        last, language = r.steps[-1], r.languages[STEPS]
        assert last["compositional"] == 0 and last["synonyms"] == 0
        assert last["words"] == len(language["lexicon"]) + S.holistic_words(4, 3) - len(language["held"])
    exact = 0
    for r in trials(bottleneck=12).values():
        last, language = r.steps[-1], r.languages[STEPS]
        if last["compositional"] == 1 and last["synonyms"] == 0 and len(language["properties"]) == 3:
            exact += 1
            assert last["words"] == len(language["lexicon"]) - 4 + S.compositional_words(4, 3)
    assert exact >= 8


def test_learners_hear_the_share_the_coverage_lesson_gives(sim):
    """Each learner hears ``bottleneck`` pairs drawn evenly: the mean share of the taught pairs a
    generation hears is 1 - (1 - 1/m)^b."""
    for bottleneck in (5, 11, 30):
        shares = []
        for seed in range(1, 9):
            r = sim.run(seed, bottleneck=bottleneck, population=60, neurons=1024)
            assert r.params["pairs"] == 12 and r.params["coverage"] == S.coverage(11, bottleneck)
            # a teacher that cannot say a pair (it never heard it and has no word for the property
            # yet) makes the share a little smaller, so only generations taught by the rule or the stock
            shares += [states[0]["seen"] for states in r.states[42:]]
        assert r.states[41][0]["seen"] == 1.0
        assert statistics.mean(shares) == pytest.approx(S.coverage(11, bottleneck), abs=0.02), bottleneck
    r = sim.run(1, bottleneck=0)
    assert r.params["coverage"] == 1.0 and all(s[0]["seen"] == 1.0 for s in r.states[41:])


# ---------------------------------------------------------------------------------------------
# plausibility


def test_without_ears_or_voice_there_are_no_signals(sim):
    for params in (dict(ears_voice=0.0), dict(ears_voice=0.02), dict(population=1), dict(population=0),
                   dict(neurons=8), dict(consonants=1, vowels=1)):
        r = sim.run(3, **params)
        assert sim.conserved(r).ok
        assert r.summary["outcome"] == "silent" and r.summary["words"] == 0 and r.summary["meanings"] == 0
        for s in r.steps:
            assert s["stage"] == "silent" and s["words"] == 0 and s["success"] == sig(1 / 8) and s["order"] == "none"
        assert all(not language["lexicon"] and not language["say"] for language in r.languages.values())
        texts = [ln.text for ln in lines(r)]
        assert "q signals seed 3 word predator. a none." in texts and "q signals seed 3 order. a none." in texts
        assert not any(" lexicon. " in t or " say " in t for t in texts)
    # half the agents deaf or mute: a quarter of the couples can talk
    half = sim.run(3, ears_voice=0.5)
    assert half.params["talkers"] == 20 and half.summary["stage"] == "language"
    assert half.summary["success"] == pytest.approx(20 * 19 / (40 * 39) * 1.0 + (1 - 20 * 19 / (40 * 39)) / 12, abs=0.03)


def test_success_rises_toward_one_in_a_mixed_population(trials):
    runs = trials()
    first = [r.steps[0]["success"] for r in runs.values()]
    calls = [r.steps[20]["success"] for r in runs.values()]
    words = [r.steps[40]["success"] for r in runs.values()]
    final = [r.steps[60]["success"] for r in runs.values()]
    assert set(first) == {0.125}, "eight meanings: a guess is right one time in eight"
    assert 0.2 < statistics.mean(calls) < 0.7 and min(calls) > 0.125, "calls carry a few meanings"
    assert statistics.mean(words) > 0.97 and min(words) > 0.9, "the naming game carries them all"
    assert statistics.mean(final) > 0.95 and min(final) > 0.9
    assert all(r.summary["outcome"] == "language" for r in runs.values())
    for r in runs.values():
        best = 0.0
        for s in r.steps[:41]:  # by and large upwards through the first two stages
            assert s["success"] > best - 0.12
            best = max(best, s["success"])


def test_alarm_calls_come_first_under_predator_pressure(trials):
    calm, pressed, hunted = (trials(predator_pressure=x) for x in (0.0, 0.5, 1.0))
    for step in (5, 10, 20):
        assert statistics.mean(r.steps[step]["alarm"] for r in hunted.values()) > 0.9
        assert statistics.mean(r.steps[step]["alarm"] for r in calm.values()) < 0.5
    first = {}
    for name, runs in (("calm", calm), ("pressed", pressed), ("hunted", hunted)):
        first[name] = 0
        for r in runs.values():
            for states in r.states[1:21]:  # the first call that is settled: is it the warning?
                lexicon = view(states[0], r.world, full=False)["lexicon"]
                if lexicon:
                    first[name] += "predator" in lexicon
                    break
    assert first["calm"] <= 5 and first["pressed"] >= 11 and first["hunted"] >= 8
    for r in hunted.values():
        # the warning is understood before the other meanings are
        others = [(8 * s["success"] - s["alarm"]) / 7 for s in r.steps[:21]]
        assert r.steps[5]["alarm"] > 0.8 and max(others[:6]) < 0.4
        assert r.steps[20]["alarm"] > r.steps[20]["success"]
    assert sum("predator" in r.languages[20]["lexicon"] for r in pressed.values()) >= 12
    assert sum("predator" in r.languages[20]["lexicon"] for r in calm.values()) <= 8


def test_the_naming_game_ends_with_about_one_word_per_meaning(trials):
    runs = trials(meanings=10, population=30, neurons=1024)
    for r in runs.values():
        peak = max(s["words"] for s in r.steps[21:41])
        end = r.steps[40]
        assert peak >= 20 > end["words"], "many words are coined, few survive"
        assert end["meanings"] >= 9 and end["synonyms"] <= 4 and end["homonyms"] == 0
        assert end["words"] <= 10 + end["synonyms"]
        assert max(s["synonyms"] for s in r.steps[21:41]) > 3 * max(1, end["synonyms"])
    assert statistics.mean(r.steps[40]["synonyms"] for r in runs.values()) < 1.5
    assert sum(r.steps[40]["meanings"] == 10 for r in runs.values()) >= 14


def test_frequent_meanings_end_with_shorter_words(trials):
    """Zipf's law of abbreviation, here by wear: a word understood may lose its last syllable."""
    top, bottom = [], []
    for r in trials(meanings=10, population=30, neurons=1024).values():
        lexicon, w = r.languages[40]["lexicon"], r.world
        lengths = [len(lexicon[w["meanings"][m]]) for m in w["named"] if w["meanings"][m] in lexicon]
        assert set(lengths) <= {1, 2, 3}
        top.append(statistics.mean(lengths[:3]))
        bottom.append(statistics.mean(lengths[-3:]))
    assert statistics.mean(top) < 1.6 and statistics.mean(bottom) > 2.2
    assert sum(a < b for a, b in zip(top, bottom)) >= 14


def test_a_tight_bottleneck_gives_a_compositional_language_and_none_a_holistic_one(trials):
    """Kirby's result. Twelve pairs, eleven of them taught."""
    share = {b: [r.summary["compositional"] for r in trials(bottleneck=b).values()] for b in (0, 12, 36, 96)}
    assert min(share[12]) >= 0.9 and all(r.summary["outcome"] == "language" for r in trials(bottleneck=12).values())
    assert max(share[0]) == 0 and max(share[96]) <= 0.2
    assert statistics.mean(share[12]) > statistics.mean(share[36]) > statistics.mean(share[96]) >= statistics.mean(share[0])
    for r in trials(bottleneck=0).values():
        assert r.summary["order"] == "none" and r.summary["stage"] == "lexicon" and r.summary["outcome"] == "lexicon"
        assert all(s["compositional"] == 0 for s in r.steps)
        assert abs(r.summary["topographic"]) < 0.35, "a holistic language has no structure to find"
        # eleven wholes and no way to say the twelfth pair
        assert r.summary["success"] == sig((11 + 1 / 12) / 12)
        assert sum(ln.answer == "none" for ln in lines(r) if " say " in ln.prompt) == 1
    for r in trials(bottleneck=12).values():
        assert r.summary["order"] in ORDERS and r.summary["topographic"] > 0.5
        assert r.steps[41]["compositional"] == 0 and r.steps[41]["order"] == "none", "generation 0 is holistic"
        assert r.summary["meanings"] == r.steps[40]["meanings"] + 12, "the rule says every pair, the unseen one too"
        assert r.summary["words"] < r.steps[41]["words"], "fewer words say more"
    orders = {r.summary["order"] for r in trials(bottleneck=12).values()}
    assert orders == set(ORDERS), "the order is an accident of the seed"


def test_a_small_brain_cannot_hold_a_holistic_language(sim, trials):
    """Toy rule: 8 neurons a word. Calls below 5 words; no pairs without room for the words of a
    rule; and where the wholes do not fit, the rule is all there is, even without a bottleneck."""
    for seed in TRIALS[:6]:
        calls = sim.run(seed, neurons=8 * (LEXICON_MIN - 1))  # 4 words
        assert calls.summary["outcome"] == "calls" and {s[0]["mode"] for s in calls.states} == {"calls"}
        assert 1 <= calls.summary["meanings"] <= 4 and calls.summary["word_length"] == 1
        assert max(s["meanings"] for s in calls.steps) <= 4
        words = sim.run(seed, neurons=64)  # 8 words: the lexicon fills the brain
        assert words.summary["outcome"] == "lexicon" and words.states[60][0]["mode"] == "naming"
        assert words.summary["order"] == "none" and words.params["pairs"] == 0
        tight = sim.run(seed, neurons=96, bottleneck=0)  # 12 words: 8 single ones, 3 properties, 1 whole
        assert tight.states[60][0]["room"] == 1 and tight.summary["compositional"] >= 0.9
        assert tight.summary["outcome"] == "language"
    assert all(r.summary["compositional"] == 0 for r in trials(bottleneck=0).values()), "512 neurons hold the wholes"


def test_split_groups_drift_into_dialects(trials):
    together, apart = trials(), trials(split=1)
    assert all(s["dialects"] == 0 for r in together.values() for s in r.steps)
    final = [r.summary["dialects"] for r in apart.values()]
    assert statistics.mean(final) > 0.9 and min(final) > 0.75
    assert statistics.mean(r.steps[40]["dialects"] for r in apart.values()) > 0.85
    for r in apart.values():
        assert r.params["talkers"] == 40 and len(r.states[0]) == 2 and r.steps[0]["dialects"] == 0
        a, b = (view(state, r.world, full=False) for state in r.states[60])
        differ = sum(a["lexicon"].get(k) != b["lexicon"].get(k) for k in set(a["lexicon"]) | set(b["lexicon"]))
        assert differ >= 5, "the words differ, not only the phrases"


def test_words_are_bounded_by_the_phoneme_inventory(sim):
    tiny = sim.run(4, consonants=1, vowels=2, meanings=12, population=30)
    assert tiny.params["syllables"] == 2 and tiny.params["possible_words"] == 2 + 4 + 8 == 14
    assert max(s["words"] for s in tiny.steps) <= 14 and sim.conserved(tiny).ok
    assert max(s["meanings"] for s in tiny.steps[:41]) <= 14
    for states in tiny.states:
        v = view(states[0], tiny.world, full=False)
        assert all(1 <= len(form) <= MAX_SYLLABLES and set(form) <= set(tiny.world["syllables"]) for form in v["forms"])
    rich = sim.run(4, consonants=8, vowels=5, meanings=12, population=30)
    assert rich.params["possible_words"] == 65640 and rich.steps[40]["meanings"] == 12
    assert tiny.steps[40]["homonyms"] == 0 and tiny.steps[40]["meanings"] <= rich.steps[40]["meanings"]


def test_utterances_are_syllables_of_the_inventory(rollouts):
    for r in rollouts.values():
        legal = set(r.world["syllables"])
        assert set(r.world["consonants"]) <= set(S.CONSONANTS) and set(r.world["vowels"]) <= set(S.VOWELS)
        assert set(r.world["meanings"]) <= set(VOCABULARY) and "predator" in r.world["meanings"]
        assert set(r.world["properties"]) <= set(PROPERTIES)
        for language in r.languages.values():
            for form in list(language["lexicon"].values()) + list(language["properties"].values()):
                assert 1 <= len(form) <= MAX_SYLLABLES and set(form) <= legal
            for form in language["say"].values():
                assert 2 <= len(form) <= 2 * MAX_SYLLABLES or form not in language["rule"].values()
                assert set(form) <= legal
        assert not legal & (set(VOCABULARY) | set(PROPERTIES) | set(STATE_KEYS) | {"none", "seed", "step", "say", "a", "q"})


def test_topographic_similarity():
    language = {"say": {("food", "near"): ("ka", "ti"), ("food", "far"): ("ka", "mu"),
                        ("water", "near"): ("po", "ti"), ("water", "far"): ("po", "mu")}, "held": ()}
    assert S.topographic(language) == pytest.approx(1.0)
    language["say"] = {key: ("ka",) for key in language["say"]}
    assert S.topographic(language) == 0.0
    assert S.topographic({"say": {}, "held": ()}) == 0.0
    assert S._distance(("ka", "ti"), ("ka", "ti")) == 0 and S._distance(("ka", "ti"), ("ti",)) == 1
    assert S._distance((), ("ka", "ti")) == 2 and S._distance(("ka", "ti", "mu"), ("mu", "ti", "ka")) == 2


def test_induce_finds_the_order_and_the_property_words():
    lexicon = {0: ("ka",), 1: ("po", "lo")}
    pairs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    seen = {0: ("ka", "ti"), 1: ("ka", "mu"), 2: ("po", "lo", "ti")}
    assert S.induce(seen, lexicon, pairs, "property_first") == ("thing_first", {0: ("ti",), 1: ("mu",)})
    seen = {0: ("ti", "ka"), 3: ("mu", "po", "lo")}
    assert S.induce(seen, lexicon, pairs, "thing_first") == ("property_first", {0: ("ti",), 1: ("mu",)})
    assert S.induce({0: ("su",), 1: ("ne", "ne")}, lexicon, pairs, "thing_first") == ("thing_first", {})
    assert S.induce({}, lexicon, pairs, "property_first") == ("property_first", {})
    # a whole that begins like the word of its thing is taken apart; the commoner remainder wins
    seen = {0: ("ka", "ti"), 2: ("po", "lo", "su"), 1: ("ka", "mu")}
    assert S.induce(seen, lexicon, pairs, "thing_first") == ("thing_first", {0: ("ti",), 1: ("mu",)})


# ---------------------------------------------------------------------------------------------
# the hand-off


def test_handoff_in_maps_the_senses_to_parameters(sim):
    below = {"stage": "brained", "population": 200, "eyes": 2.9, "ears": 1.4, "neurons": 181.3, "voice": 0.95,
             "eared": 0.93, "talkers": 0.9, "survival": 0.8, "predator_pressure": 0.45, "kin_gain": 0.11,
             "group": "yes", "outcome": "brained"}
    p = handoff_in(below)
    assert p == {"ears_voice": 0.9, "neurons": 181, "predator_pressure": 0.45, "population": 60, "split": 1}
    assert handoff_in({}) == {} and handoff_in({"group": "no"}) == {"population": 6, "split": 0}
    assert handoff_in({"population": 100, "group": "yes"}) == {"population": 60, "split": 0}
    assert handoff_in({"population": 40, "group": "no"}) == {"population": 6, "split": 0}
    assert handoff_in({"population": 3, "group": "no"}) == {"population": 3, "split": 0}
    assert handoff_in({"eared": 0.8, "voice": 0.3}) == {"ears_voice": 0.3}
    assert handoff_in({"talkers": 1.7, "predator_pressure": -0.2, "neurons": float("nan")}) == {
        "ears_voice": 1.0, "predator_pressure": 0.0}
    assert handoff_in({"talkers": True, "neurons": "many"}) == {}
    r = sim.run(11, **{**random_params(random.Random(11)), **p})
    assert sim.conserved(r).ok and r.params["population"] == 60 and r.params["capacity"] == 22
    blind = handoff_in({"talkers": 0.0, "neurons": 0.0, "predator_pressure": 0.2, "population": 150, "group": "no"})
    r = sim.run(11, **{**random_params(random.Random(11)), **blind})
    assert r.summary["outcome"] == "silent"
    # what goes up: the success, the size of the lexicon and whether the language is compositional
    assert {"success", "words", "meanings", "compositional", "order", "outcome"} <= set(SUMMARY_KEYS)


def test_a_run_is_fast(sim):
    worst = 0.0
    for seed in range(100, 116):
        start = time.perf_counter()
        sim.run(seed, **random_params(random.Random(seed)))
        worst = max(worst, time.perf_counter() - start)
    start = time.perf_counter()
    sim.run(3, meanings=12, population=60, neurons=32, split=1)  # the slowest kind: calls for 60 steps, twice
    worst = max(worst, time.perf_counter() - start)
    start = time.perf_counter()  # and the widest bottleneck a run takes, for 60 learners and 22 taught pairs
    r = sim.run(1, meanings=12, population=60, neurons=1024, things=6, properties=4, bottleneck=S.MAX_BOTTLENECK)
    worst = max(worst, time.perf_counter() - start)
    assert r.params["pairs"] == 24 and r.summary["compositional"] == 0
    assert worst < 2.0


# ---------------------------------------------------------------------------------------------
# hardening (round 7 review)


def test_a_seed_no_run_takes_is_nobodys_question(sim):
    """The gate replays a seed, so a seed that ``run`` refuses cannot be asked for; it used to
    raise ValueError out of ``check``."""
    assert S.MAX_SEED == 10 ** 15
    top = " ".join(str(S.MAX_SEED))
    assert sim.owns(f"signals seed {top} final success") and S.owns(f"signals seed {top} order")
    for digits in (" ".join(str(S.MAX_SEED + 1)), " ".join("9" * 16), " ".join("9" * 400)):
        for tail in ("final success", "step 3 words", "step 3 next words", "params coverage", "word predator",
                     "step 2 0 word predator", "meaning ka", "say predator near", "order"):
            prompt = f"signals seed {digits} {tail}"
            assert not sim.owns(prompt) and not S.owns(prompt)
            for check in (sim.check, S.check):
                v = check(prompt, "1")
                assert v.ok is False and v.expected is None and v.reason == "not my question"
            assert sim.answers(prompt) is None and sim.truth(prompt) is None


def test_no_prompt_is_asked_twice(rollouts):
    for seed, r in rollouts.items():
        prompts = [ln.prompt for ln in lines(r) if ln.kind != "record"]
        assert len(prompts) == len(set(prompts)), seed
        texts = [ln.text for ln in lines(r)]
        assert len(texts) == len(set(texts)), seed


def rollout_digest(r) -> str:
    text = "|".join(ln.text for ln in lines(r)) + "|" + repr(r.steps) + repr(r.summary) + repr(sorted(r.params.items()))
    return hashlib.sha256(text.encode()).hexdigest()


def test_pinned_digests_of_whole_rollouts(sim):
    """Every line, step, summary and parameter of five runs, through all three games, two groups,
    deaf agents and a run that stays with calls: written down on numpy 2.4, python 3.11. Another
    numpy, python or machine that changed a single draw, tie or sum would change a digest."""
    assert rollout_digest(sim.rollout(21)) == "fc070c97458f46bb0140aa1184f481766d821eb45b2a1dd685dcff87461cf592"
    assert rollout_digest(sim.rollout(87)) == "94683ca14e8ede5a7ec9c8ee6b2d047e75545961165b5888f36e535af8e21a58"
    assert rollout_digest(sim.rollout(91)) == "d7ab9bcb643500147672a92074e0097501ee0e51a094d216e17ab1292f69d086"
    two = sim.run(5, population=31, split=1, ears_voice=0.6, bottleneck=9, meanings=11, consonants=3, vowels=2)
    assert rollout_digest(two) == "85b065586ed31c7511a1616ff79861d0b9fa24403a54323b9de91fa2b801837a"
    calls = sim.run(6, neurons=32, meanings=12, predator_pressure=1.0)
    assert rollout_digest(calls) == "1aed1b216c87887f44b2c07b6a3e4dbb8bf63f3b1df5c3bbc3fea49230c1be3f"
    assert len(two.states[0]) == 2 and two.params["talkers"] == 19 and calls.summary["outcome"] == "calls"


def test_the_weights_stay_whole_numbers_far_below_two_to_the_53(sim):
    """Why stage 1 cannot depend on how numpy sums: every weight is a whole number, and no table
    comes near the size from which float64 skips whole numbers. Run to step 60 under the largest
    reward."""
    r = sim.run(6, neurons=32, meanings=12, predator_pressure=1.0, population=60)
    assert {s[0]["mode"] for s in r.states} == {"calls"}
    A = r.states[60][0]["A"]
    assert bool((A == np.round(A)).all()) and A.min() >= 1.0
    assert float(A.sum()) < 2.0 ** 40, "exact sums: far from 2 ** 53"
    # the most a weight can have gained: every round a success with the largest reward
    assert A.max() <= 1 + STEPS * S.ROUNDS * S.REWARD * (1 + S.STAKE)


def independent(name: str, v: list[int]) -> float | int:
    """The lesson rules once more, written from the module docstring and in exact fractions."""
    if name == "expected_success":
        signals, states, shared = v
        return float(Fraction(shared, states) + Fraction(states - shared, states) * Fraction(signals - shared, signals)
                     * Fraction(1, states))
    if name == "signal_bits":
        return math.log(v[0]) / math.log(2.0)
    if name == "syllables":
        return sum(v[1] for _ in range(v[0]))
    if name == "possible_words":
        syllables, length = v
        total, power = 0, 1
        for _ in range(length):
            power *= syllables
            total += power
        return total
    if name == "holistic_words":
        return len([(t, p) for t in range(v[0]) for p in range(v[1])])
    if name == "compositional_words":
        return len(range(v[0])) + len(range(v[1]))
    if name == "coverage":
        meanings, bottleneck = v
        return float(1 - Fraction(meanings - 1, meanings) ** bottleneck)
    if name == "zipf_frequency":
        rank, meanings = v
        return float(Fraction(1, rank) / sum(Fraction(1, k) for k in range(1, meanings + 1)))
    if name == "naming_max_words":
        return (v[0] - v[0] % 2) // 2
    if name == "lexicon_capacity":
        return (v[0] - v[0] % 8) // 8
    raise KeyError(name)


def test_lessons_of_four_seeds_are_recomputed_and_no_answer_dominates():
    answers: dict[str, dict[str, int]] = {name: {} for name in RULES}
    for seed in (1, 2, 3, 4):
        lessons = LESSONS.generate(random.Random(seed), 500)
        assert len(lessons) == 500
        for ln in lessons:
            assert is_dense(ln.text) and n_tokens(ln.text) <= 128 and "." not in ln.prompt and "." not in ln.answer
            v = LESSONS.check(ln.prompt, ln.answer)
            assert v.ok and v.expected == ln.answer, ln.text
            name, values = lesson_inputs(ln.prompt)
            assert all(isinstance(x, int) for x in values), "every input of this level is a count"
            assert ln.answer == num(independent(name, values)), ln.text
            answers[name][ln.answer] = answers[name].get(ln.answer, 0) + 1
    for name, counts in answers.items():
        total = sum(counts.values())
        assert total >= 80, (name, total)
        # measured: coverage 0.14 (the answer 1, a wide bottleneck), expected_success 0.09 (the answer 1)
        assert max(counts.values()) / total < 0.2, (name, max(counts, key=counts.get), max(counts.values()) / total)
        assert len(counts) >= 30, name
    assert max(n_tokens(ln.text) for ln in LESSONS.generate(random.Random(9), 500)) <= 40


def test_a_lesson_built_from_a_run_gives_what_the_run_has(sim, rollouts):
    """Every lesson rule, asked with the numbers of a canonical run, answers with what stands in
    that run's parameters, steps or agents."""
    asked = {name: 0 for name in RULES}

    def lesson(name: str, values: list[int], truth: float | int) -> None:
        ln = LESSONS.line(name, values)
        assert ln is not None, (name, values)
        assert LESSONS.check(ln.prompt, num(truth)).ok, (ln.text, truth)
        assert ln.answer == num(truth) and RULES[name].compute(*values) == truth
        asked[name] += 1

    for seed, r in rollouts.items():
        p, w = r.params, r.world
        lesson("syllables", [p["consonants"], p["vowels"]], p["syllables"])
        lesson("possible_words", [p["syllables"], MAX_SYLLABLES], p["possible_words"])
        lesson("lexicon_capacity", [p["neurons"]], p["capacity"])
        lesson("signal_bits", [len(w["named"])], p["bits"])
        # step 0: nothing is shared yet, whatever the number of calls
        assert parse_num(LESSONS.line("expected_success", [max(1, w["calls"]), p["meanings"], 0]).answer) == \
            pytest.approx(r.steps[0]["success"], rel=3e-3)
        lesson("expected_success", [max(1, w["calls"]), p["meanings"], 0], w["guess"])
        if p["predator_pressure"] == 0:
            for rank, share in enumerate(w["weights"], start=1):
                assert LESSONS.check(f"signals predict zipf_frequency rank {num(rank)} meanings {num(p['meanings'])}",
                                     num(share)).ok
        talkers = S._sizes(p["population"], p["split"], p["ears_voice"])[0][1]
        naming = [s[0] for s in r.states if s[0]["mode"] == "naming"]
        if naming:
            most = max(len({f for inv in state["inventories"] for f in inv.get(m, ())})
                       for state in naming for m in w["named"])
            assert most <= S.naming_max_words(talkers)
            lesson("naming_max_words", [talkers], talkers // 2)
        language = r.languages[STEPS]
        if language["things"]:
            taught = p["pairs"] - len(language["held"])
            lesson("holistic_words", [len(language["things"]), p["properties"]], p["pairs"])
            state = r.states[STEPS][0]
            lesson("compositional_words", [len(language["things"]), p["properties"]],
                   p["capacity"] - (len(state["lexicon"]) - len(language["things"])) - state["room"])
            if p["bottleneck"]:
                lesson("coverage", [taught, p["bottleneck"]], p["coverage"])
            else:  # no bottleneck is not "no utterance": the lesson gate does not take the question
                assert p["coverage"] == 1.0
                assert not LESSONS.owns(f"signals predict coverage meanings {num(taught)} bottleneck 0")
    quiet = sim.run(1, predator_pressure=0.0)
    for rank, share in enumerate(quiet.world["weights"], start=1):
        lesson("zipf_frequency", [rank, 8], S.zipf_frequency(rank, 8))
        assert share == pytest.approx(S.zipf_frequency(rank, 8), rel=1e-12)
    assert all(asked[name] >= 8 for name in RULES), asked


def test_the_rule_functions_are_called_by_the_simulation(sim, monkeypatch):
    """Not only equal in value: the simulation goes through the very functions the lessons use.
    Each is replaced by a counting wrapper in the module, and a run must call every one."""
    called = {name: 0 for name in RULES}
    for name in RULES:
        original = getattr(S, name)
        assert RULES[name].compute is original

        def counting(*args, _name=name, _original=original):
            called[_name] += 1
            return _original(*args)
        monkeypatch.setattr(S, name, counting)
    r = sim.run(3)
    assert all(count > 0 for count in called.values()), called
    assert called["zipf_frequency"] == 8 and called["lexicon_capacity"] == 1 and called["naming_max_words"] == 1
    assert sim.conserved(r).ok and r.summary["outcome"] == "language"


def test_other_gates_prompts_are_not_owned(sim):
    """Questions of round 5, of the round-6 lessons and of the other round-7 levels, as their own
    gates write them."""
    seen = 0
    for name in ("maths", "elements", "substances", "reactions", "forces"):
        gate = importlib.import_module("haishool.truth." + name).gate()
        for ln in gate.generate(random.Random(3), 150):
            if ln.kind != "record":
                seen += 1
                assert not S.owns(ln.prompt) and not sim.owns(ln.prompt) and not LESSONS.owns(ln.prompt), ln.prompt
                assert S.check(ln.prompt, ln.answer).reason == "not my question"
    from haishool.cosmos import predict
    for ln in predict.generate(random.Random(3), 300):
        seen += 1
        assert not S.owns(ln.prompt), ln.prompt
    assert seen > 800
    for name in ("stars", "cells", "bodies", "senses", "society"):
        try:  # the other levels of round 7 are being written at the same time
            other = importlib.import_module("haishool.evo." + name)
            theirs = other.lesson_gate().generate(random.Random(3), 100)
        except Exception:  # noqa: BLE001
            continue
        for ln in theirs:
            assert not S.owns(ln.prompt), ln.prompt
            assert ln.prompt.split()[0] == name


def test_bottleneck_zero_means_none_in_a_run_and_is_no_lesson(sim):
    """``bottleneck 0`` is the parameter for no bottleneck (coverage 1); the function gives 0 for
    no utterance. So that no line contradicts another, the lesson starts at 1 utterance."""
    assert S.coverage(12, 0) == 0.0
    spec = {i.name: i for i in RULES["coverage"].inputs}
    assert spec["bottleneck"].low == 1 and spec["meanings"].low == 2
    r = sim.run(2, bottleneck=0)
    assert r.params["bottleneck"] == 0 and r.params["coverage"] == 1.0 and r.params["pairs"] == 12
    assert all(s[0]["seen"] == 1.0 for s in r.states[41:])
    for seed in range(1, 5):
        for ln in LESSONS.generate(random.Random(seed), 500):
            if ln.prompt.startswith("signals predict coverage"):
                assert not ln.prompt.endswith("bottleneck 0") and parse_num(ln.answer) > 0
    assert sim.run(2, bottleneck=S.MAX_BOTTLENECK).summary["compositional"] == 0


def coined_lengths(r) -> list[int]:
    """Syllables of the step-40 words of the meanings that had no call at step 20 (their word was
    coined in stage 2, with 3 syllables), most talked-about meaning first."""
    w, calls, lexicon = r.world, r.languages[20]["lexicon"], r.languages[40]["lexicon"]
    names = [w["meanings"][m] for m in w["named"]]
    return [len(lexicon[name]) for name in names if name in lexicon and name not in calls]


def test_abbreviation_has_two_causes_and_wear_is_one_of_them(sim, trials, monkeypatch):
    """Frequent meanings end with shorter words because they keep their one-syllable calls and
    because use wears words down. The second alone: among the words coined in stage 2, the more
    frequent half is shorter; and without wear every coined word keeps its 3 syllables."""
    frequent, rare = [], []
    for r in trials(meanings=10, population=30, neurons=1024).values():
        lengths = coined_lengths(r)
        half = len(lengths) // 2
        assert half >= 2
        frequent.append(statistics.mean(lengths[:half]))
        rare.append(statistics.mean(lengths[-half:]))
    assert statistics.mean(frequent) < 2.1 and statistics.mean(rare) > 2.3
    assert statistics.mean(rare) - statistics.mean(frequent) > 0.4
    assert sum(a < b for a, b in zip(frequent, rare)) >= 13 and sum(a > b for a, b in zip(frequent, rare)) <= 2
    monkeypatch.setattr(S, "WEAR", 0.0)
    top, bottom = [], []
    for seed in TRIALS[:6]:
        r = sim.run(seed, meanings=10, population=30, neurons=1024)
        assert sim.conserved(r).ok
        assert set(coined_lengths(r)) == {MAX_SYLLABLES}, "no wear: a coined word stays as long as it was coined"
        lexicon, w = r.languages[40]["lexicon"], r.world
        lengths = [len(lexicon[w["meanings"][m]]) for m in w["named"] if w["meanings"][m] in lexicon]
        top.append(statistics.mean(lengths[:3]))
        bottom.append(statistics.mean(lengths[-3:]))
        assert all(len(form) == 1 for form in r.languages[20]["lexicon"].values())
    # the calls alone still make the frequent meanings shorter: the other cause
    assert statistics.mean(bottom) == 3 and 1.3 < statistics.mean(top) < 2.6


def test_punishing_failures_is_what_settles_the_calls(sim, monkeypatch):
    """The docstring's reason for the punishment, measured: calls only (4 nameable meanings, no
    predator pressure), 60 steps. With it nearly all 4 calls settle; plain reinforcement stays in
    partial pooling far more often."""
    def settled():
        runs = [sim.run(seed, predator_pressure=0.0, neurons=32) for seed in TRIALS[:8]]
        assert all(r.summary["outcome"] == "calls" and len(r.world["named"]) == 4 for r in runs)
        return [r.steps[60]["meanings"] for r in runs], [r.steps[60]["success"] for r in runs]
    with_punishment = settled()
    monkeypatch.setattr(S, "PUNISH", 0)
    without = settled()
    assert statistics.mean(with_punishment[0]) >= 3.5 and statistics.mean(without[0]) <= 3.2
    assert sum(a > b for a, b in zip(with_punishment[1], without[1])) >= 7
    best = (4 + 4 / 8) / 8  # four meanings told apart, four guessed
    assert max(with_punishment[1]) <= sig(best) and statistics.mean(with_punishment[1]) > 0.9 * best
    assert statistics.mean(without[1]) < 0.8 * best


def test_too_tight_a_bottleneck_has_the_rule_but_loses_the_property_words(trials):
    """Kirby's bottleneck has two ends: the rule appears below about 12 utterances for 11 taught
    pairs, but with 3 utterances a learner often never hears a property, so the group talks worse."""
    tight, medium, right = (trials(bottleneck=b) for b in (3, 6, 12))
    mean = {name: statistics.mean(r.summary["success"] for r in runs.values())
            for name, runs in (("tight", tight), ("medium", medium), ("right", right))}
    assert mean["tight"] < 0.65 < mean["medium"] < 0.92 < mean["right"]
    assert statistics.mean(r.summary["compositional"] for r in medium.values()) > 0.95
    for r in tight.values():
        assert r.params["coverage"] == S.coverage(11, 3) and r.summary["order"] in ORDERS + ("none",)
        # nothing but the rule is left: no learner remembers more than the 3 wholes it can have heard
        assert all(len(memory) <= 3 for memory, _, _ in r.states[60][0]["agents"])


def test_under_high_pressure_the_warning_is_understood_even_where_no_call_is_settled(trials):
    """The open point of the docstring, held: with synonyms for predator no call is the
    majority's, the lexicon of step 20 lists none, and yet the warning gets through."""
    hunted = trials(predator_pressure=1.0)
    without = [r for r in hunted.values() if "predator" not in r.languages[20]["lexicon"]]
    assert 1 <= len(without) <= 6, "a minority of the runs"
    for r in hunted.values():
        assert r.steps[20]["alarm"] > 0.95
    for r in without:
        A = r.states[20][0]["A"]
        predator = r.world["predator"]
        sending = A[:, predator, :] / A[:, predator, :].sum(axis=1, keepdims=True)
        strong = (sending.mean(axis=0) > 0.2).sum()
        assert strong >= 2, "two or more calls are in use for predator"


def warning_rates(state: dict, world: dict) -> tuple[float, float]:
    """How often a hearer acts on ``predator`` when the speaker saw a predator (hits), and when it
    saw another nameable thing (false alarms): over every speaker and every other hearer."""
    A = state["A"]
    T, predator = A.shape[0], world["predator"]
    send = A / A.sum(axis=2, keepdims=True)
    taken = (A / A.sum(axis=1, keepdims=True))[:, predator, :]  # hearer x call: read as predator
    hit = false = 0.0
    others = [m for m in world["named"] if m != predator]
    for i in range(T):
        for j in range(T):
            if i != j:
                hit += float(send[i, predator] @ taken[j])
                false += sum(float(send[i, m] @ taken[j]) for m in others) / len(others)
    return hit / (T * (T - 1)), false / (T * (T - 1))


def test_alarm_is_a_hit_rate_and_high_pressure_brings_false_alarms(trials):
    """``alarm`` counts warnings understood. It does not count the calls about something else that
    are taken for a warning, and under predator pressure those are many: predators are much of the
    talk, and an unsettled call is read as the likeliest meaning. The docstring says so; held here."""
    false = {}
    for name, pressure in (("calm", 0.0), ("hunted", 1.0)):
        rates = []
        for seed, r in list(trials(predator_pressure=pressure).items())[:8]:
            assert r.params["talkers"] == r.params["population"] == 40
            for step in (10, 20):
                hit, wrong = warning_rates(r.states[step][0], r.world)
                assert hit == pytest.approx(view(r.states[step][0], r.world)["alarm"], abs=1e-9)
                assert r.steps[step]["alarm"] == pytest.approx(hit, rel=6e-3)
                rates.append((step, hit, wrong))
        false[name] = {step: statistics.mean(w for s, _, w in rates if s == step) for step in (10, 20)}
        if name == "hunted":
            assert min(h for _, h, _ in rates) > 0.9
    assert false["calm"][10] < 0.2 and false["calm"][20] < 0.2
    assert 0.3 < false["hunted"][10] < 0.55 and 0.25 < false["hunted"][20] < false["hunted"][10]
    assert false["hunted"][20] > 2 * false["calm"][20]


# ---------------------------------------------------------------------------------------------
# second review: other seeds and another parameter stream than the ones the claims were made on


def test_forty_worlds_of_another_parameter_stream(sim):
    """Not the canonical parameters of a seed: 40 worlds from another stream, each run twice.
    The gate passes, the two runs agree to the last bit, every line is dense and short, a run
    takes well under 2 seconds. ``check`` replays the canonical world of a seed, so it is pointed
    at these runs through a gate whose cache holds them: it accepts every line and rejects every
    altered answer."""
    outcomes = set()
    gate = Signals()
    for k in range(1, 41):
        params = random_params(random.Random(10_000 + 7 * k))
        start = time.perf_counter()
        r = sim.run(500 + k, **params)
        assert time.perf_counter() - start < 2.0
        again = sim.run(500 + k, **params)
        assert r.steps == again.steps and r.summary == again.summary and r.params == again.params
        assert same(r.states, again.states) and same(r.languages, again.languages)
        v = sim.conserved(r)
        assert v.ok, (k, v)
        gate._cache[r.seed] = r
        for ln in lines(r):
            assert is_dense(ln.text) and n_tokens(ln.text) <= 128, ln.text
            if ln.kind != "record":
                assert "." not in ln.prompt and "." not in ln.answer
                assert gate.owns(ln.prompt) and S.owns(ln.prompt) and not LESSONS.owns(ln.prompt)
                assert gate.check(ln.prompt, ln.answer).ok, ln.text
                wrong = wrong_answer(gate, ln)
                assert not gate.check(ln.prompt, wrong).ok, (ln.text, wrong)
        outcomes.add(r.summary["outcome"])
    assert outcomes == set(OUTCOMES)


def test_the_plausibility_table_holds_on_seeds_it_was_not_measured_on(sim):
    """The docstring's table was measured on seeds 1 to 30; seeds 201 to 216 tell the same story."""
    fresh = range(201, 217)
    mean = statistics.mean
    base = [sim.run(s) for s in fresh]  # bottleneck 12, pressure 0.5
    assert {r.steps[0]["success"] for r in base} == {0.125}
    assert 0.25 < mean(r.steps[20]["success"] for r in base) < 0.55
    assert mean(r.steps[40]["success"] for r in base) > 0.97 and mean(r.steps[60]["success"] for r in base) > 0.95
    assert min(r.summary["compositional"] for r in base) >= 0.9
    assert all(r.summary["outcome"] == "language" for r in base)
    loose = [sim.run(s, bottleneck=96) for s in fresh]
    assert max(r.summary["compositional"] for r in loose) <= 0.2
    assert mean(r.summary["topographic"] for r in base) > 0.6 > 0.2 > mean(r.summary["topographic"] for r in loose)
    hunted = [sim.run(s, predator_pressure=1.0) for s in fresh]
    calm = [sim.run(s, predator_pressure=0.0) for s in fresh]
    assert mean(r.steps[10]["alarm"] for r in hunted) > 0.9 > 0.5 > mean(r.steps[10]["alarm"] for r in calm)
    assert sum("predator" in r.languages[20]["lexicon"] for r in hunted) > \
        sum("predator" in r.languages[20]["lexicon"] for r in calm)


def test_lessons_of_four_other_seeds_and_the_share_of_the_commonest_answer():
    """As the lesson test above, with the generator seeds 11 to 14: recomputed independently,
    and no rule dominated by one answer (measured: coverage 0.17 and expected_success 0.10, both
    the answer 1; every other rule 0.07 or less)."""
    answers: dict[str, dict[str, int]] = {name: {} for name in RULES}
    for seed in (11, 12, 13, 14):
        for ln in LESSONS.generate(random.Random(seed), 500):
            assert LESSONS.check(ln.prompt, ln.answer).ok
            name, values = lesson_inputs(ln.prompt)
            assert ln.answer == num(independent(name, values)) == num(RULES[name].compute(*values))
            assert RULES[name].compute is getattr(S, name)
            answers[name][ln.answer] = answers[name].get(ln.answer, 0) + 1
    for name, counts in answers.items():
        share = max(counts.values()) / sum(counts.values())
        assert share < (0.2 if name in ("coverage", "expected_success") else 0.08), (name, share)
