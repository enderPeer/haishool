"""Level 4 under rules 7 (``haishool.cosmos.chem`` with ``rules=7``): the corrected chemistry.

Round 6 stays as it is (tests/test_cosmos_chem.py and tests/test_round6_frozen.py are the guard);
this file tests the option: the enthalpy table, the pool with exchange and sparks, the lines, the
gate, the lessons and the plausibility targets measured on seeds 1-40.
"""
import copy
import hashlib
import json
import math
import random
import time

import numpy as np
import pytest

from haishool.cosmos import Simulation
from haishool.cosmos import chem
from haishool.cosmos.chem import (BOND_KJ, CENSUS_KEYS7, CONDENSE_K, COUNT_KEYS7, ELEMENTS, FORMATION_KJ, HOT_K, KEYS7,
                                  LESSONS, MIXES7, MOLECULES7, PARAM_KEYS7, QUERY_KEYS7, STAGES7, STATE_KEYS7,
                                  VALENCE, WATER_STATES, Pool7, bond_enthalpy, budget7, element_counts,
                                  molecule_counts, random_params, simulation, stability7, stable_below_k,
                                  temperature_at, unit_enthalpy)
from haishool.truth import is_dense, num, parse_num, split_line
from haishool.truth.formula import parse, parse_dense, same

FORTY = range(1, 41)
FIXED = {"t_start": 4600, "t_end": 290, "steps": 30, "atoms": 20000}
MAX_TOKENS = 128


def n_tokens(text: str) -> int:
    return len(text.replace(".", " . ").split())  # haishool.student.tokens


@pytest.fixture(scope="module")
def sim():
    return simulation()


@pytest.fixture(scope="module")
def forty(sim):
    """Seeds 1-40 under rules 7, each with its own parameters (the runs check() replays)."""
    return {seed: sim.rollout(seed, rules=7) for seed in FORTY}


@pytest.fixture(scope="module")
def generated():
    return chem.generate(random.Random(5), 3, rules=7)


def pool_of(counts: dict[str, int], seed: int = 1, **options) -> Pool7:
    return Pool7(dict(counts), np.random.Generator(np.random.PCG64(seed)), **options)


def totals_of(counts: dict[str, int]) -> np.ndarray:
    return np.array([counts.get(e, 0) for e in ELEMENTS])


def final_pool(r) -> Pool7:
    pool = None
    for _, _, pool in chem.cool(r.seed, r.params):
        pass
    return pool


def count_of(counts: dict[str, int], formula: str) -> int:
    return next((c for f, c in counts.items() if same(parse_dense(f), parse(formula))), 0)


def wrong(answer: str) -> str:
    """An answer no tolerance forgives: another word, or the leading digit moved by five."""
    words = answer.split()
    for pool in (STAGES7, WATER_STATES, tuple(MIXES7), ("yes", "no"), ("solid", "vapor")):
        if answer in pool:
            return pool[(pool.index(answer) + 1) % len(pool)]
    if not any(w.isdigit() for w in words):  # a name, a list of words, never
        return "many" if len(words) == 1 else " ".join(words[1:])
    i = next(i for i, w in enumerate(words) if w.isdigit())
    words[i] = str((int(words[i]) + 5) % 10)
    return " ".join(words)


# ---------------------------------------------------------------------------------------------
# the option itself: round 6 is the default and stays what it was
# ---------------------------------------------------------------------------------------------

def test_default_is_still_round_6(sim):
    r = sim.run(1)
    assert "rules" not in r.params and not chem.is_rules7(r)
    assert r.params == random_params(random.Random(1)) == random_params(random.Random(1), rules=6)
    again = sim.run(1, rules=6)
    assert again.params == r.params and again.steps == r.steps and again.summary == r.summary
    assert r.summary == {"mix": "rocky", "h2": 0, "h2o": 22, "ch4": 0, "nh3": 0, "nacl": 17, "sio2": 2798, "organic": 0,
                         "molecules": 6828, "biggest": 17, "water_fraction": 0.00322, "first_water_step": 11}
    texts = [ln.text for ln in chem.lines(r)]
    assert texts[0].startswith("chem seed 1 step 0. mix rocky. temperature 4 8 0 0. stage plasma.")
    assert not any(" rules 7" in t for t in texts)
    assert [ln.text for ln in sim.records()] == [ln.text for ln in sim.records(rules=6)]
    assert [ln.text for ln in sim.table_lines()] == [ln.text for ln in sim.table_lines(rules=6)]
    assert not any("rules 7" in ln.text for ln in sim.records() + sim.table_lines())
    assert chem.MIXES.keys() == {"cosmic", "rocky", "ocean", "carbon"} and chem.STAGES[0] == "plasma"


def test_rules_6_is_the_default_on_twenty_seeds(sim):
    """``run(seed)`` and ``run(seed, rules=6)`` are one rollout, and nothing of it says ``rules``."""
    for seed in [*range(1, 15), 77, 123, 999, 4242, 9001, 9998]:
        a, b = sim.run(seed), sim.run(seed, rules=6)
        assert a.params == b.params == random_params(random.Random(seed)) and a.summary == b.summary
        assert [[(k, type(v), v) for k, v in x.items()] for x in a.steps] == [[(k, type(v), v) for k, v in x.items()] for x in b.steps]
        assert list(a.params) == ["mix", "t_start", "t_end", "steps", "atoms"]  # no rules, no sparks
        texts = [ln.text for ln in chem.lines(a)]
        assert texts == [ln.text for ln in chem.lines(b, rules=6)] and not any("rules" in t for t in texts)
    assert not any("rules" in ln.text for ln in chem.generate(random.Random(3), 2, every=7))


def test_rules_is_6_or_7(sim):
    for bad in (5, 8, True, "7"):
        with pytest.raises(ValueError):
            sim.run(1, rules=bad)
        with pytest.raises(ValueError):
            random_params(random.Random(1), rules=bad)
    r6, r7 = sim.run(2), sim.run(2, rules=7)
    assert chem.is_rules7(r7) and r7.params["rules"] == 7 and r7.sim == "chem"
    assert [ln.text for ln in chem.lines(r7)] == [ln.text for ln in chem.lines(r7, rules=7)] \
        == [ln.text for ln in chem.lines7(r7)]
    with pytest.raises(ValueError):
        chem.lines(r6, rules=7)  # one rollout has one set of rules
    with pytest.raises(ValueError):
        chem.lines7(r6)
    assert chem.run(2, rules=7).steps == r7.steps and chem.run(2).steps == r6.steps
    assert sim.rollout(2, rules=7).steps == r7.steps and sim.rollout(2).steps == r6.steps


def test_is_a_simulation_under_rules_7(sim):
    assert isinstance(sim, Simulation) and sim.sim == "chem"
    r = sim.run(1, rules=7)
    assert set(r.steps[0]) | set(r.summary) <= set(KEYS7)
    assert list(r.steps[0]) == QUERY_KEYS7 == STATE_KEYS7 + CENSUS_KEYS7
    assert r.params == {**random_params(random.Random(1), rules=7), "rules": 7}
    assert tuple(k for k in r.params if k != "rules") == PARAM_KEYS7
    assert len(r.steps) == r.params["steps"] + 1 + r.params["sparks"]
    for bad in ({"mix": "lava"}, {"sparks": -1}, {"sparks": 2.5}, {"sparks": 201}, {"steps": 0}, {"atoms": True},
                {"pressure": 3}):
        with pytest.raises(ValueError):
            sim.run(1, rules=7, **bad)
    assert sim.run(1, rules=7, mix="reducing", sparks=0, **FIXED).params["mix"] == "reducing"
    with pytest.raises(ValueError):
        sim.run(1, mix="reducing")  # round 6 has no such mix


def test_random_params_of_rules_7():
    mixes, water = set(), set()
    for seed in range(400):
        p6, p7 = random_params(random.Random(seed)), random_params(random.Random(seed), rules=7)
        assert p7 == random_params(random.Random(seed), rules=7) and list(p7) == list(PARAM_KEYS7)
        # the five draws of round 6 come first and are the same
        assert all(p7[k] == p6[k] for k in ("t_start", "steps", "atoms"))
        assert p7["mix"] in (p6["mix"], "reducing")
        assert 0 <= p7["sparks"] <= chem.MAX_SPARKS == 20
        if p7["mix"] in chem.WATERY_MIXES:
            assert 200 <= p7["t_end"] <= 500 and p7["t_end"] % 10 == 0
            water.add(chem.water_state(p7["t_end"]))
        else:
            assert p7["t_end"] == p6["t_end"] and 100 <= p7["t_end"] <= 1500
        mixes.add(p7["mix"])
    assert mixes == set(MIXES7) == {"cosmic", "rocky", "ocean", "carbon", "reducing"}
    assert water == set(WATER_STATES)  # the water of a seed's own ocean is ice, liquid or vapor


def test_same_seed_same_rollout(sim, forty):
    for seed in (1, 2, 5, 9):
        a, b = sim.run(seed, rules=7), forty[seed]  # a fresh run and the cached one
        assert a.params == b.params and a.summary == b.summary and len(a.steps) == len(b.steps)
        for x, y in zip(a.steps, b.steps):  # same keys in the same order, same types, same values
            assert [(k, type(v), v) for k, v in x.items()] == [(k, type(v), v) for k, v in y.items()]
        assert [ln.text for ln in chem.lines(a)] == [ln.text for ln in chem.lines(b)]
    assert sim.run(1, rules=7, mix="ocean", sparks=2, **FIXED).steps != sim.run(2, rules=7, mix="ocean", sparks=2, **FIXED).steps
    one = [ln.text for ln in chem.generate(random.Random(9), 2, every=6, rules=7)]
    assert one == [ln.text for ln in chem.generate(random.Random(9), 2, every=6, rules=7)]
    assert one != [ln.text for ln in chem.generate(random.Random(10), 2, every=6, rules=7)]


def test_golden_rollouts():
    """Pinned on numpy 2.4.6, scipy 1.17.1, python 3.11: a seed must stay its rollout on any machine."""
    assert random_params(random.Random(1), rules=7) == {"mix": "rocky", "t_start": 4800, "t_end": 1450, "steps": 26,
                                                        "atoms": 20000, "sparks": 3}
    for seed, mix, digest in GOLDEN:
        r = chem._rollout7(seed)
        assert r.params["mix"] == mix
        assert hashlib.sha256(json.dumps([r.steps, r.summary], sort_keys=True).encode()).hexdigest()[:16] == digest, seed
    own = chem.lines(chem._rollout7(42))
    assert own[25].text == (
        "chem seed 4 2 rules 7 step 1 2. mix cosmic. temperature 2 0 8 8. stage molecular. water vapor. solids none. "
        "spark 0. hits 0. free_atoms 0 point 0 9 1 4. molecules 8 5 9 8. h2 8 5 1 5. h2o 6 6. ch4 0. nh3 0. co 0. "
        "co2 0. o2 0. n2 1. nacl 0. h2s 0. h2co 0. other 1 6.")
    assert own[26].text == ("chem seed 4 2 rules 7 step 1 2 census. organic 0. precursors 0. chains 0. biggest 3. "
                            "rock 0. grains 0. si_o 0. mg_o 1. fe_o 1. fe_metal 0.")


GOLDEN = [(1, "rocky", "c76ebe72f35b8cc4"), (2, "cosmic", "be3a1cd447c1697a"), (5, "ocean", "a84ce6923032a1da"),
          (9, "reducing", "43f3ebbb3113a597"), (11, "carbon", "d1b7a0ed57302784")]


# ---------------------------------------------------------------------------------------------
# lines
# ---------------------------------------------------------------------------------------------

def test_every_line_is_dense_and_short(sim, generated):
    every = generated + sim.records(rules=7) + sim.table_lines(rules=7) + LESSONS.records() \
        + LESSONS.generate(random.Random(3), 600)
    widest = 0
    for mix in MIXES7:  # the widest numbers each mix can give
        every += chem.lines(sim.run(9998, rules=7, mix=mix, t_start=6000, t_end=100, steps=35, atoms=22000, sparks=20))
    assert len(every) > 20000
    for ln in every:
        assert is_dense(ln.text), ln.text
        widest = max(widest, n_tokens(ln.text))
        assert n_tokens(ln.text) <= MAX_TOKENS, ln.text
        if ln.kind != "record":
            assert "." not in ln.prompt and "." not in ln.answer and ln.answer
            assert split_line(ln.text) == (ln.prompt, ln.answer)
    assert 95 <= widest <= 115  # measured: 107


def test_line_kinds(sim, generated):
    texts = [ln.text for ln in generated]
    assert len(texts) == len(set(texts))
    r = sim.rollout(5, rules=7)
    own = chem.lines(r)
    assert own[0].text == ("chem seed 5 rules 7 params. mix ocean. t_start 5 3 0 0. t_end 4 4 0. steps 3 5. "
                           "atoms 2 2 0 0 0. sparks 0.")
    records = [ln for ln in own if ln.kind == "record"]
    assert len(records) == 1 + 2 * len(r.steps)
    assert records[1].text.startswith("chem seed 5 rules 7 step 0. mix ocean. temperature 5 3 0 0. stage hot. water vapor. "
                                      "solids none. spark 0. hits 0. free_atoms 1. molecules 0. h2 0.")
    assert records[2].text == ("chem seed 5 rules 7 step 0 census. organic 0. precursors 0. chains 0. biggest 1. rock 0. "
                               "grains 0. si_o 0. mg_o 0. fe_o 0. fe_metal 0.")
    assert all(ln.prompt.startswith("chem seed 5 rules 7 ") for ln in own)
    now = [ln for ln in own if " step " in ln.prompt and " next " not in ln.prompt and ln.kind != "record"]
    nxt = [ln for ln in own if " next " in ln.prompt]
    final = [ln for ln in own if " final " in ln.prompt]
    params = [ln for ln in own if " param " in ln.prompt]
    assert len(now) == len(r.steps) * len(QUERY_KEYS7) and len(nxt) == (len(r.steps) - 1) * len(QUERY_KEYS7)
    assert len(final) == len(r.summary) and len(params) == len(PARAM_KEYS7)
    assert len(chem.lines(r, every=5)) < len(own)
    own_texts = {ln.text for ln in own}
    for text in ["q chem seed 5 rules 7 step 3 0 h2o. a 5 9 5 2.", "q chem seed 5 rules 7 step 3 0 next o2. a 2 9.",
                 "q chem seed 5 rules 7 final water_share. a 0 point 8 6 5.", "q chem seed 5 rules 7 param atoms. a 2 2 0 0 0.",
                 "q chem seed 5 rules 7 param mix. a ocean.", "q chem seed 5 rules 7 final first_water_step. a 1 3.",
                 "q chem seed 5 rules 7 step 3 0 solids. a corundum silicate iron iron_sulfide.",
                 "q chem seed 5 rules 7 step 3 0 water. a vapor."]:
        assert text in own_texts, text
    # every key of a state is in one of its two records, and the first record closes: other is what no name counts
    for s in r.steps:
        assert s["other"] == s["molecules"] - sum(s[k] for k in MOLECULES7)
    # the summary carries every count of the last step
    assert set(r.summary) == {"mix", "water", *(COUNT_KEYS7 - {"hits"}), "water_share", "water_atoms", "first_water_step"}
    tables = {ln.text for ln in sim.table_lines(rules=7)}
    for text in ["q chem rules 7 bonds fe. a 2.", "q chem rules 7 element fe name. a iron.",
                 "q chem rules 7 element s extra_to_oxygen. a 4.", "q chem rules 7 atom o enthalpy_kj. a 2 4 9 point 2.",
                 "q chem rules 7 compound h2o unit_kj. a 4 6 4.", "q chem rules 7 compound al2o3 formation_kj. a minus 1 6 7 5 point 7.",
                 "q chem rules 7 bond h o order 1 enthalpy_kj. a 4 6 4.", "q chem rules 7 bond h o order 1 stable_below_k. a 2 2 2 7.",
                 "q chem rules 7 bond c o order 2 enthalpy_kj. a 8 0 4.", "q chem rules 7 bond c o highest_order. a 2.",
                 "q chem rules 7 bond h o order 1 source. a h2o.", "q chem rules 7 bond default stable_below_k. a 9 6 0.",
                 "q chem rules 7 bond h he order 1 stable_below_k. a never.", "q chem rules 7 molecule h2o name. a water.",
                 "q chem rules 7 molecule h2co formula. a h 2 c 1 o 1.", "q chem rules 7 mix ocean share o. a 0 point 3 3.",
                 "q chem rules 7 mix cosmic metal_boost. a 8.", "q chem rules 7 mix ocean salt_boost. a 7.",
                 "q chem rules 7 condense water_ice below_k. a 1 8 0.", "q chem rules 7 constant hot_k. a 4 5 3 6."]:
        assert text in tables, text
    assert tables <= set(texts)
    records = [ln.text for ln in sim.records(rules=7)]
    assert records[0] == ("chem rules 7 bonds. h 1. he 0. o 2. c 4. n 3. ne 0. mg 2. si 4. s 2. fe 2. al 3. ca 2. na 1. "
                          "k 1. cl 1.")
    for text in ["chem rules 7 element fe. name iron. bonds 2.", "chem rules 7 element s. name sulfur. bonds 2. extra_to_oxygen 4.",
                 "chem rules 7 compound h2o. formula h 2 o 1. atoms_kj 6 8 5 point 2. formation_kj minus 2 4 1 point 8. "
                 "bond h o. order 1. units 2. unit_kj 4 6 4.",
                 "chem rules 7 bond h o order 1. enthalpy_kj 4 6 4. stable_below_k 2 2 2 7. source h2o.",
                 "chem rules 7 bond default. enthalpy_kj 2 0 0. stable_below_k 9 6 0. highest_order 1.",
                 "chem rules 7 molecule h2o. name water. formula h 2 o 1. atoms 3.",
                 "chem rules 7 mix ocean. h 0 point 5 9. o 0 point 3 3. na 0 point 0 2. cl 0 point 0 2. c 0 point 0 1 2. "
                 "s 0 point 0 1. n 0 point 0 1. mg 0 point 0 0 8. salt_boost 7.",
                 "chem rules 7 condense water_ice. below_k 1 8 0."]:
        assert text in records, text
    # no pair is taught a filler value as if it were a fact about that pair
    assert not any(" source default" in t for t in records) and not any("a default." in t for t in tables)


# ---------------------------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------------------------

def test_gate_agrees_with_its_own_lines(sim, generated):
    questions = [ln for ln in generated if ln.kind != "record"]
    assert len(questions) > 5000
    for ln in questions:
        assert sim.owns(ln.prompt) and chem.owns(ln.prompt), ln.prompt
        v = sim.check(ln.prompt, ln.answer)
        assert v.ok and v.expected == ln.answer, (ln.text, v)
        assert chem.check(ln.prompt, ln.answer) == v


def test_gate_rejects_wrong_answers(sim, generated):
    questions = [ln for ln in generated if ln.kind != "record"]
    for ln in questions:
        v = sim.check(ln.prompt, wrong(ln.answer))
        assert not v.ok and v.expected == ln.answer, (ln.text, wrong(ln.answer), v)
        v = sim.check(ln.prompt, "many")  # a word where a number or another word belongs
        assert not v.ok and v.expected == ln.answer, (ln.text, v)
    for prompt, good, bad in [("chem rules 7 bonds c", "4", "3"), ("chem rules 7 bond o h order 1 enthalpy_kj", "4 6 4", "3 0 0 0"),
                              ("chem rules 7 bond ne o order 1 stable_below_k", "never", "9 6 0"),
                              ("chem rules 7 bond ne o highest_order", "0", "1"),
                              ("chem rules 7 bond fe fe order 1 enthalpy_kj", "4 1 6", "2 0 0"),
                              ("chem rules 7 bond fe fe order 1 source", "metal", "default"),
                              ("chem rules 7 bond si n order 1 enthalpy_kj", "2 0 0", "3 3 2"),
                              ("chem rules 7 bond si n order 1 source", "default", "mean"),
                              ("chem rules 7 bond n n order 3 stable_below_k", "4 5 3 6", "4 0 0 0"),
                              ("chem rules 7 bond c n highest_order", "3", "2"),
                              ("chem rules 7 element he bonds", "0", "2"), ("chem rules 7 element o extra_to_oxygen", "0", "4"),
                              ("chem rules 7 molecule nh3 formula", "n 1 h 3", "n 1 h 4"),
                              ("chem rules 7 molecule co2 name", "carbon_dioxide", "carbon_monoxide"),
                              ("chem rules 7 mix reducing main", "h", "o"),
                              ("chem rules 7 mix rocky elements", " ".join(MIXES7["rocky"]), "o si mg"),
                              ("chem rules 7 mix rocky share mg", "0 point 1 8", "0 point 1"),
                              ("chem rules 7 compound nacl bond", "cl na", "na cl"),
                              ("chem rules 7 constant rounds", "3", "4")]:
        assert sim.check(prompt, good) == chem.Verdict(True, good)
        assert sim.check(prompt, bad) == chem.Verdict(False, good)
    for prompt in ["chem rules 7 bonds xx", "chem rules 7 element zz name", "chem rules 7 atom he enthalpy_kj",
                   "chem rules 7 compound h2so4 unit_kj", "chem rules 7 bond h o order 2 enthalpy_kj",
                   "chem rules 7 bond xx o highest_order", "chem rules 7 molecule sio2 name", "chem rules 7 mix lava main",
                   "chem rules 7 mix ocean share si", "chem rules 7 mix rocky metal_boost", "chem rules 7 condense tin below_k",
                   "chem rules 7 constant speed"]:
        assert sim.owns(prompt) and sim.check(prompt, "1") == chem.Verdict(False, None, "unknown name"), prompt


def test_gate_tolerance(sim):
    r = sim.rollout(2, rules=7)  # check() replays the seed's own run
    t = next(t for t, s in enumerate(r.steps) if s["molecules"] > 1000)
    s, base = r.steps[t], f"chem seed 2 rules 7 step {num(t)}"
    assert sim.check(f"{base} molecules", num(s["molecules"] + 1)).ok  # a count may be off by one
    assert sim.check(f"{base} molecules", num(int(s["molecules"] * 1.04))).ok  # or by under 5 %
    assert not sim.check(f"{base} molecules", num(int(s["molecules"] * 1.2))).ok
    assert sim.check(f"{base} temperature", num(s["temperature"] + 20)).ok
    assert not sim.check(f"{base} temperature", num(s["temperature"] * 2)).ok
    assert not sim.check(f"{base} stage", "hot" if s["stage"] != "hot" else "atomic").ok
    assert not sim.check(f"{base} stage", "frozen").ok and not sim.check(f"{base} stage", "plasma").ok
    assert sim.check(f"chem seed 2 rules 7 step {num(t - 1)} next molecules", num(s["molecules"])).ok
    assert sim.check(f"{base} molecules", "many") == chem.Verdict(False, num(s["molecules"]), "not a number")
    # ordinals and parameters are exact
    last = len(r.steps) - 1
    assert r.params["sparks"] == 9 and r.steps[last]["spark"] == 9
    assert sim.check(f"chem seed 2 rules 7 step {num(last)} spark", "9").ok
    assert not sim.check(f"chem seed 2 rules 7 step {num(last)} spark", "8").ok
    first = r.summary["first_water_step"]
    assert sim.check("chem seed 2 rules 7 final first_water_step", num(first)).ok
    assert not sim.check("chem seed 2 rules 7 final first_water_step", num(first + 1)).ok
    assert not sim.check("chem seed 2 rules 7 final first_water_step", "never").ok
    assert sim.check("chem seed 2 rules 7 param atoms", "1 9 0 0 0").ok
    assert not sim.check("chem seed 2 rules 7 param atoms", "1 9 0 0 1").ok
    assert sim.check("chem seed 2 rules 7 param sparks", "9").ok and not sim.check("chem seed 2 rules 7 param sparks", "1 0").ok
    # only finite numbers
    for prompt in [f"{base} temperature", f"{base} h2o", f"{base} free_atoms", f"{base} next molecules",
                   "chem seed 2 rules 7 final water_share", "chem seed 2 rules 7 final h2", "chem seed 2 rules 7 param t_end"]:
        expected = sim.check(prompt, "0").expected
        assert expected is not None and sim.check(prompt, expected).ok
        for answer in ["1 e 9 9 9", "minus 1 e 9 9 9", "9 e 4 0 0", "", "point", "minus", "e 5", "1 point 2 point 3"]:
            assert sim.check(prompt, answer) == chem.Verdict(False, expected, "not a number"), (prompt, answer)
    # what the rollout does not have
    assert sim.check(f"chem seed 2 rules 7 step {num(last)} next h2o", "1").expected is None
    assert sim.check(f"chem seed 2 rules 7 step {num(last + 5)} h2o", "1").expected is None
    assert sim.check("chem seed 2 rules 7 step 3 clumps", "1").expected is None
    assert sim.check("chem seed 2 rules 7 step 3 sio2", "1").expected is None  # a round-6 key
    assert sim.check("chem seed 2 rules 7 final water_fraction", "1").expected is None
    assert sim.check("chem seed 2 rules 7 param rules", "7").expected is None


def test_one_seed_two_stories(sim):
    """``chem seed 7 step 1 h2o`` keeps meaning round 6; ``rules 7`` in the prompt names the other."""
    r6, r7 = sim.rollout(7), sim.rollout(7, rules=7)
    t = len(r6.steps) - 1
    assert r6.steps[t]["h2o"] != r7.steps[t]["h2o"]
    assert sim.check(f"chem seed 7 step {num(t)} h2o", "0").expected == num(r6.steps[t]["h2o"])
    assert sim.check(f"chem seed 7 rules 7 step {num(t)} h2o", "0").expected == num(r7.steps[t]["h2o"])
    assert sim.check(f"chem seed 7 step {num(t)} stage", "x").expected == r6.steps[t]["stage"]
    assert sim.check(f"chem seed 7 rules 7 step {num(t)} stage", "x").expected == r7.steps[t]["stage"]
    assert sim.check("chem seed 7 final water_fraction", "1").expected is not None
    assert sim.check("chem seed 7 final water_share", "1").expected is None
    # the tables too: a filler value of round 6 against the table of rules 7
    assert sim.check("chem bond c n stable_below_k", "2 0 0 0").ok
    assert sim.check("chem rules 7 bond c n order 3 stable_below_k", num(stable_below_k(855))).ok
    assert sim.check("chem valence fe", "2").ok and sim.check("chem rules 7 bonds fe", "2").ok
    assert not sim.owns("chem rules 7 valence fe")


def test_owns_only_its_own_prompts(sim):
    for prompt in ["chem seed 7 rules 7 step 2 0 h2o", "chem seed 1 2 3 4 rules 7 step 0 next stage",
                   "chem seed 7 rules 7 final water_share", "chem seed 7 rules 7 param sparks", "chem rules 7 bonds si",
                   "chem rules 7 bond cl na order 1 enthalpy_kj", "chem rules 7 bond default enthalpy_kj",
                   "chem rules 7 molecule h2o formula", "chem rules 7 mix rocky elements", "chem rules 7 mix ocean share h",
                   "chem rules 7 condense iron below_k", "chem rules 7 constant spark_ppm", "chem seed 7 step 2 0 h2o"]:
        assert sim.owns(prompt) and chem.owns(prompt), prompt
    for prompt in ["chem7 seed 7 step 1 h2o", "chem7 rules 7 bonds si", "chem seed 7 rules 6 step 1 h2o",
                   "chem seed 7 rules 8 step 1 h2o", "chem seed 7 rules 7", "chem seed 7 rules 7 step 1",
                   "chem seed 7 rules 7 step h2o", "chem seed 0 7 rules 7 step 1 h2o", "chem seed 7 rules 7 step 0 1 h2o",
                   "chem seed 7 rules 7 step 1 h2o ", "chem seed 7 rules 7 step 1 next h2o extra", "chem seed 7 rules 7 final",
                   "chem seed 7 rules 7 params", "chem seed minus 7 rules 7 final h2o", "q chem seed 7 rules 7 final h2o",
                   "chem rules 7", "chem rules 7 bond h o", "chem rules 7 bond h o order 4 enthalpy_kj",
                   "chem rules 7 bond h o stable_below_k", "chem rules 7 valence c", "chem rules 7 mix cosmic",
                   "chem rules 6 bonds c", "rules 7 bonds c", "", "turkey capital", "spoon color",
                   "life seed 7 rules 7 step 3 replicators", "life seed 3 rules 7 param mu", "life seed 3 param mutation",
                   "gravity seed 7 rules 7 final clumps", "gravity seed 7 step 2 0 clumps", "nucleo seed 3 final helium_fraction",
                   "planets seed 2 2 0 2 step 0 planets", "world seed 3 era planets", "world7 seed 3 era chem",
                   "stars seed 7 step 1 metals", "cells seed 7 step 1 cells", "signals seed 7 say predator near",
                   "society predict hamilton relatedness 0 point 5 benefit 4 cost 1", "carbon protons", "water molar_mass",
                   "calc 1 2 plus 7", "check 1 2 plus 7 equals 2 0", "hydrogen_sulfide elements", "water formula"]:
        assert not sim.owns(prompt) and not chem.owns(prompt), prompt
        assert not sim.check(prompt, "1").ok and not chem.check(prompt, "1").ok
    assert sim.check("turkey capital", "ankara") == chem.Verdict(False, None, "not my question")
    # a lesson belongs to the level (module functions), not to the replay gate of the simulation
    lesson = "chem predict stable_below_k enthalpy_kj 4 6 4"
    assert chem.owns(lesson) and LESSONS.owns(lesson) and not sim.owns(lesson)
    assert chem.check(lesson, "2 2 2 7").ok and not chem.check(lesson, "2 2 2 8").ok
    assert not LESSONS.owns("chem seed 7 rules 7 step 1 h2o") and not LESSONS.owns("life predict stage replicators 0")


# ---------------------------------------------------------------------------------------------
# the tables: enthalpies by Hess's law, one rule for the temperature, bond orders, mixes
# ---------------------------------------------------------------------------------------------

def test_bond_table_is_computed_from_enthalpies_of_formation():
    # Hess's law: free atoms minus formation enthalpy, per valence unit bonded
    assert unit_enthalpy(2 * 218.0 + 249.2, -241.8, 2) == 464  # 463.5 rounds up: tenths, not floats, decide
    assert unit_enthalpy(436.0, 0.0, 1) == 436 and unit_enthalpy(716.7 + 2 * 249.2, -393.5, 4) == 402
    for name, (formula, formation, (a, b), order, units) in FORMATION_KJ.items():
        assert a <= b and name == formula.lower()
        atoms = chem.atoms_enthalpy(formula)
        assert atoms == pytest.approx(sum(chem.ATOM_KJ[e.lower()] * n for e, n in parse(formula).items()))
        assert bond_enthalpy(a, b, order) == order * unit_enthalpy(atoms, formation, units), name
        assert chem.BOND_SOURCE[(a, b, order)] == name
        # the valence units of the formula: each takes one valence from either side (sulfur opens two for so2)
        assert 2 * units == sum(VALENCE[e.lower()] * n for e, n in parse(formula).items()) + (2 if name == "so2" else 0)
    for pair, kj in {"h h": 436, "h o": 464, "c h": 416, "h n": 391, "cl h": 432, "h s": 367, "cl na": 640, "cl k": 647,
                     "mg o": 499, "fe o": 469, "o si": 465, "ca o": 531, "al o": 514, "na o": 439, "si si": 225}.items():
        a, b = pair.split()
        assert bond_enthalpy(a, b) == bond_enthalpy(b, a) == kj, pair
    assert BOND_KJ[("c", "o")] == (358, 804) and BOND_KJ[("o", "o")] == (146, 498) and BOND_KJ[("n", "n")] == (163, 418, 945)
    assert BOND_KJ[("c", "c")] == (346, 614, 839) and BOND_KJ[("c", "n")] == (305, 615, 855) and BOND_KJ[("o", "s")] == (265, 536)
    # two metals: the atomisation enthalpy of each per valence, added
    assert bond_enthalpy("fe", "fe") == 416 and bond_enthalpy("na", "na") == 215 and bond_enthalpy("fe", "na") == 315
    assert chem.BOND_SOURCE[("fe", "fe", 1)] == "metal"
    # the filler is one declared number, below every bond that matters, and is not a fact about a pair
    defaults = [pair for pair in BOND_KJ if chem.BOND_SOURCE[(*pair, 1)] == "default"]
    assert len(BOND_KJ) == 91 and len(defaults) == 24 and all(BOND_KJ[p] == (chem.DEFAULT_KJ,) for p in defaults)
    assert chem.DEFAULT_KJ == 200 < min(bond_enthalpy(a, b) for a, b in [("cl", "na"), ("h", "s"), ("h", "na"), ("fe", "s")])
    assert chem.DEFAULT_KJ < min(kj // m for _, _, m, kj, source in chem._bond_rows() if source not in ("mean", "metal"))
    # which pairs they are: silicides, carbides and nitrides of the metals, Si-N, Si-S, N-S and chlorine with O, N, S
    metals = sorted(chem.METALS)
    assert set(defaults) == {*(tuple(sorted((x, m))) for x in ("c", "n", "si") for m in metals), ("n", "si"), ("s", "si"),
                             ("n", "s"), ("cl", "o"), ("cl", "n"), ("cl", "s")}
    assert not {("c", "n"), ("h", "si"), ("fe", "s"), ("cl", "fe"), ("cl", "na")} & set(defaults)
    for a in ELEMENTS:
        assert chem.highest_order("he", a) == chem.highest_order(a, "ne") == 0 == bond_enthalpy("he", a) == stability7(a, "ne")
        for b in ELEMENTS:
            assert bond_enthalpy(a, b) == bond_enthalpy(b, a) and chem.highest_order(a, b) == chem.highest_order(b, a)
            top = chem.highest_order(a, b)
            assert bond_enthalpy(a, b, top + 1) == 0
            kjs = [bond_enthalpy(a, b, m) for m in range(1, top + 1)]
            assert kjs == sorted(kjs) and len(set(kjs)) == len(kjs)  # a higher order is a stronger bond
    assert set(chem.NOTES7) >= {"atom_kj", "formation_kj", "mean_kj", "metal_metal", "default", "kelvin_per_kj",
                                "highest_order", "extra_valence", "condense_k", "mixes", "sparks"}
    assert "from memory" in chem.NOTES7["formation_kj"] and "from memory" in chem.NOTES7["condense_k"]


def test_one_rule_for_the_temperature():
    assert stable_below_k(436) == 2092 and stable_below_k(464) == 2227 and stable_below_k(945) == 4536
    # 4.8 K per kJ/mol is "the enthalpy is 25 R T"
    assert 1000 / (25 * 8.314) == pytest.approx(chem.KELVIN_TENTHS_PER_KJ / 10, abs=0.02)
    # and what the docstring says of H2: half dissociated at 2092 K near 1e-5 bar, at 1e-4 bar near 2300 K (22 R T).
    # log10 of the equilibrium constant of formation of H atoms, JANAF, from memory: -2.790 at 2000 K, -1.601 at 2500 K
    def k_p(kelvin):  # H2 = 2 H in bar, interpolated in 1/T
        along = (1 / 2000 - 1 / kelvin) / (1 / 2000 - 1 / 2500)
        return 10 ** (2 * (-2.790 + along * (2.790 - 1.601)))
    half = 4 * 0.5 ** 2 / (1 - 0.5 ** 2)  # K_p over the pressure when half the molecules are split
    assert 3e-6 < k_p(stable_below_k(436)) / half < 2e-5
    kelvin = next(k for k in range(2000, 2600) if k_p(k) >= half * 1e-4)
    assert 2250 < kelvin < 2400 and 436000 / (8.314 * kelvin) == pytest.approx(22.5, abs=1)
    assert stable_below_k(200) == 960 and stable_below_k(640) == 3072
    for (a, b), kjs in BOND_KJ.items():
        for m, kj in enumerate(kjs, 1):
            assert stability7(a, b, m) == stability7(b, a, m) == kj * 48 // 10
    assert HOT_K == 4536 == max(stability7(a, b, m) for (a, b), kjs in BOND_KJ.items() for m in range(1, len(kjs) + 1))
    # real order of strength, which round 6 had backwards: the N-N triple bond, C=O of carbon dioxide, then Si-O;
    # O=O is stronger than H-H, H-Cl holds above nothing it should not
    assert stability7("n", "n", 3) > stability7("c", "o", 2) > stability7("o", "si") > stability7("h", "h")
    assert stability7("o", "o", 2) > stability7("h", "h") and bond_enthalpy("cl", "h") == 432
    assert chem.STAGES7 == ("hot", "atomic", "forming", "molecular")
    assert chem.stage7(HOT_K, 1) == "hot" and chem.stage7(HOT_K - 1, 1) == "atomic"
    assert chem.stage7(2000, 0.9) == "atomic" and chem.stage7(2000, 0.899) == "forming"
    assert chem.stage7(2000, 0.5) == "forming" and chem.stage7(2000, 0.499) == "molecular"
    assert chem.stage7(650, 0) == "molecular" and chem.stage7(100, 0) == "molecular"  # nothing is called frozen


def test_bond_orders_follow_the_double_bond_rule():
    multiple = {pair: len(kjs) for pair, kjs in BOND_KJ.items() if len(kjs) > 1}
    assert multiple == chem.HIGHEST_ORDER == {("c", "c"): 3, ("c", "n"): 3, ("n", "n"): 3, ("c", "o"): 2, ("o", "o"): 2,
                                              ("n", "o"): 2, ("o", "s"): 2, ("c", "s"): 2, ("s", "s"): 2}
    for counts, formula, order in [({"n": 200}, "N2", 3), ({"o": 200}, "O2", 2), ({"h": 200}, "H2", 1)]:
        pool = pool_of(counts)
        for _ in range(6):
            pool.step(500)
        found = molecule_counts(pool)
        assert count_of(found, formula) >= 98 and len(found) == 1 and set(pool.bond_m.tolist()) == {order}
    # silica is a network of single bonds, each oxygen between two silicon atoms: one grain, no SiO2 units
    pool = pool_of({"si": 300, "o": 600})
    for t in range(1, 21):
        pool.step(temperature_at(4000, 300, t, 20))
    comp, size = pool.census()
    silicon = (pool.el[pool.bond_a] == ELEMENTS.index("si")) | (pool.el[pool.bond_b] == ELEMENTS.index("si"))
    assert set(pool.bond_m[silicon].tolist()) == {1} and size.max() > 800  # measured: 880 of 900 atoms
    assert count_of(molecule_counts(pool), "SiO2") < 5 and pool.audit(totals_of({"si": 300, "o": 600})) == []
    for mix in ("rocky", "carbon"):  # no Si=O, Mg=O, Fe=O, Si-Si triple bonds anywhere
        pool = final_pool(simulation().run(3, rules=7, mix=mix, sparks=0, **{**FIXED, "atoms": 6000}))
        ea, eb = pool.el[pool.bond_a], pool.el[pool.bond_b]
        for metal in ("si", "mg", "fe", "al", "ca", "na", "k", "cl", "h"):
            touched = (ea == ELEMENTS.index(metal)) | (eb == ELEMENTS.index(metal))
            assert (pool.bond_m[touched] == 1).all(), (mix, metal)


def test_mixes_say_what_they_are():
    assert MIXES7["ocean"] == chem.MIXES["ocean"] and MIXES7["carbon"] == chem.MIXES["carbon"]
    assert MIXES7["ocean"] is not chem.MIXES["ocean"]
    for mix, fractions in MIXES7.items():
        assert abs(sum(fractions.values()) - 1) < 1e-9 and set(fractions) <= set(ELEMENTS)  # a record's shares add up
        assert list(fractions.values()) == sorted(fractions.values(), reverse=True)
        for atoms in (18000, 19000, 20000, 21000, 22000):
            counts = element_counts(mix, atoms, rules=7)
            assert sum(counts.values()) == atoms and min(counts.values()) > 0 and list(counts) == list(fractions)
    assert element_counts("cosmic", 20000, rules=7) != element_counts("cosmic", 20000)
    assert element_counts("ocean", 20000, rules=7) == element_counts("ocean", 20000)
    cosmic, rocky = MIXES7["cosmic"], MIXES7["rocky"]
    # the sun's ratios (from memory) times the declared boost
    for e, sun in {"o": 4.5e-4, "c": 2.5e-4, "ne": 7.8e-5, "n": 6.2e-5, "mg": 3.7e-5, "si": 3.0e-5, "fe": 2.9e-5, "s": 1.2e-5}.items():
        assert cosmic[e] == pytest.approx(chem.METAL_BOOST * sun, rel=0.2), e
    mass = {"h": 1, "he": 4, "o": 16, "c": 12, "ne": 20, "n": 14, "mg": 24.3, "si": 28.1, "fe": 55.8, "s": 32.1}
    assert cosmic["he"] * 4 / sum(cosmic[e] * mass[e] for e in cosmic) == pytest.approx(0.25, abs=0.01)  # helium by mass
    assert rocky["mg"] / rocky["si"] == pytest.approx(1.25, abs=0.05)  # the silicate earth, not 0.71
    hydrogen, oxygen = MIXES7["reducing"]["h"], MIXES7["reducing"]["o"]
    assert hydrogen / oxygen > 2 > MIXES7["ocean"]["h"] / MIXES7["ocean"]["o"]  # reducing against oxidising
    assert chem.MIX_BOOSTS7 == {"cosmic": ("metal_boost", 8), "ocean": ("salt_boost", 7)}
    assert chem.MOLECULE_NAMES.keys() == MOLECULES7.keys() and chem.ELEMENT_NAMES.keys() == VALENCE.keys()
    for name, formula in MOLECULES7.items():
        assert name == formula.lower() and name in KEYS7
    assert "sio2" not in MOLECULES7 and "mgo" not in MOLECULES7 and "feo" not in MOLECULES7


def test_water_and_solids_follow_from_the_temperature():
    from haishool.cosmos import planets, world
    assert (chem.WATER_MIN_K, chem.WATER_MAX_K) == (world.WATER_MIN_K, world.WATER_MAX_K) == (273, 373)
    assert [chem.water_state(t) for t in (100, 272, 273, 290, 373, 374, 650, 5000)] \
        == ["ice", "ice", "liquid", "liquid", "liquid", "vapor", "vapor", "vapor"]
    assert list(CONDENSE_K.values()) == sorted(CONDENSE_K.values(), reverse=True)
    assert chem.condensed(180, 179) == "solid" and chem.condensed(180, 180) == "vapor"
    assert chem.solids_at(2000) == "none" and chem.solids_at(1500) == "corundum"
    assert chem.solids_at(1000) == "corundum silicate iron"
    assert chem.solids_at(150) == "corundum silicate iron iron_sulfide water_ice"
    assert chem.solids_at(50) == " ".join(CONDENSE_K)
    # level 3's frost line: a black body at 2.7 au is 169 K, where the table has water ice (below 180 K)
    frost = 278.3 / math.sqrt(planets.FROST_AU)
    assert chem.condensed(CONDENSE_K["water_ice"], round(frost)) == "solid" and 160 < frost < CONDENSE_K["water_ice"]


# ---------------------------------------------------------------------------------------------
# the pool: meeting, exchange, sulfur, sparks, and its laws
# ---------------------------------------------------------------------------------------------

def test_bonds_form_when_cool_and_break_when_hot():
    counts = {"h": 600, "o": 100}
    pool = pool_of(counts, 3)
    pool.step(2500)  # hotter than O-H (2227), H-H (2092) and O=O (2390)
    assert len(pool.bonds) == 0 and pool.formed == 0
    pool.step(2300)  # only O=O holds
    kinds = {tuple(sorted((ELEMENTS[a], ELEMENTS[b]))) for a, b in zip(pool.el[pool.bond_a], pool.el[pool.bond_b])}
    assert kinds == {("o", "o")} and set(pool.bond_m.tolist()) == {2}
    pool.step(2150)  # O-H holds too: hydrogen atoms take the oxygen molecules apart
    for _ in range(3):
        pool.step(1500)  # and H-H
    found = molecule_counts(pool)
    assert found["h 2 o 1"] == 100 and found["h 2"] == 200 and len(found) == 2  # all oxygen is water
    assert pool.audit(totals_of(counts)) == []
    before = len(pool.bonds)
    pool.step(2150)  # H-H breaks, O-H stays
    assert 0 < pool.broken < before and "h 2" not in molecule_counts(pool) and molecule_counts(pool)["h 2 o 1"] == 100
    pool.step(5000)
    assert len(pool.bonds) == 0 and (pool.free == pool.valence).all() and molecule_counts(pool) == {}
    assert pool.audit(totals_of(counts)) == []


def test_exchange_is_a_displacement_that_gives_off_heat():
    # HCl + NaOH -> NaCl + H2O, built by hand: atoms 0 H, 1 Cl, 2 Na, 3 O, 4 H
    pool = Pool7({"h": 2, "cl": 1, "na": 1, "o": 1}, np.random.Generator(np.random.PCG64(1)))
    h1, h2, cl, na, o = 0, 1, 2, 3, 4
    assert [ELEMENTS[e] for e in pool.el] == ["h", "h", "cl", "na", "o"]
    pool._set(h1, cl, 1)
    pool._set(na, o, 1)
    pool._set(o, h2, 1)
    assert molecule_counts(pool) == {"h 1 cl 1": 1, "h 1 o 1 na 1": 1}
    start = pool.enthalpy()
    assert start == 432 + 439 + 464
    assert pool.judge(h1, h2, 300) is None  # H-H 436 for H-Cl 432 and O-H 464: no heat, no move
    assert pool.judge(na, cl, 300) is not None  # Na-Cl 640 and O-H 464 for Na-O 439 and H-Cl 432
    assert molecule_counts(pool) == {"h 2 o 1": 1, "na 1 cl 1": 1} and pool.moves["double"] == 1
    assert pool.enthalpy() - start == pool.heat_kj == chem.reaction_heat(640 + 464, 439 + 432) == 233
    assert pool.judge(na, cl, 300) is None and pool.judge(h1, o, 300) is None  # nothing left to gain
    assert pool.audit(totals_of({"h": 2, "cl": 1, "na": 1, "o": 1})) == []
    # a single displacement: a free H atom takes an O2 molecule apart; the O-O single bond that is left
    # cannot hold at 2000 K and breaks at once
    pool = Pool7({"h": 1, "o": 2}, np.random.Generator(np.random.PCG64(1)))
    pool._set(1, 2, 2)
    assert pool.judge(0, 1, 2000) is not None and molecule_counts(pool) == {"h 1 o 1": 1}
    assert pool.heat_kj == 464 - (498 - 146) and pool.thermal_kj == 146 and pool.moves["displace"] == 1
    # at 500 K the single bond holds: H-O-O
    pool = Pool7({"h": 1, "o": 2}, np.random.Generator(np.random.PCG64(1)))
    pool._set(1, 2, 2)
    assert pool.judge(0, 1, 500) is not None and molecule_counts(pool) == {"h 1 o 2": 1} and pool.thermal_kj == 0
    # two H2 molecules have nothing to gain from each other; neither do two water molecules
    pool = Pool7({"h": 4}, np.random.Generator(np.random.PCG64(1)))
    pool._set(0, 1, 1)
    pool._set(2, 3, 1)
    assert pool.judge(0, 2, 300) is None and pool.exchange(300) == 0 and molecule_counts(pool) == {"h 2": 2}


def test_exchange_makes_the_end_state_not_a_record_of_who_met_whom():
    counts = element_counts("ocean", 6000, rules=7)
    water, salt, knallgas = {}, {}, {}
    for exchange in (True, False):
        for seed in (1, 2, 3):
            pool = pool_of(counts, seed, exchange=exchange)
            for t in range(1, 31):
                pool.step(temperature_at(4600, 290, t, 30))
            assert pool.audit(totals_of(counts)) == []
            found = molecule_counts(pool)
            water.setdefault(exchange, []).append(found["h 2 o 1"])
            salt.setdefault(exchange, []).append(found["na 1 cl 1"])
            knallgas.setdefault(exchange, []).append(found.get("h 2", 0) + found.get("o 2", 0))
    # measured: water 1631-1648 against 1105-1130, salt 83-89 against 28-41, H2 + O2 4-8 against 818-863
    assert min(water[True]) > 1500 > 1300 > max(water[False])
    assert min(salt[True]) > 70 > 50 > max(salt[False])
    assert max(knallgas[True]) < 30 and min(knallgas[False]) > 600


def test_the_prefilter_of_the_exchange_round_is_exact():
    """Judging only the pairs the whole-array bound lets through gives the same pool as judging every pair."""
    for mix in MIXES7:
        counts = element_counts(mix, 3000, rules=7)
        for seed in (1, 2):
            pools = [pool_of(counts, seed, prefilter=flag) for flag in (True, False)]
            for pool in pools:
                for t in range(1, 17):
                    pool.step(temperature_at(4700, 250, t, 16))
                for _ in range(4):
                    pool.spark(250)
            fast, slow = pools
            assert fast.bonds == slow.bonds and fast.moves == slow.moves and fast.heat_kj == slow.heat_kj
            assert fast.valence.tolist() == slow.valence.tolist() and fast.hits == slow.hits
            assert fast.judged < slow.judged / 3  # and it is the reason a run is fast


def test_every_exchange_round_keeps_the_enthalpy_ledger():
    counts = element_counts("ocean", 4000, rules=7)
    pool = pool_of(counts, 2, ledger=True)
    moves = 0
    for t in range(1, 26):
        temperature = temperature_at(4600, 300, t, 25)
        pool.break_bonds(temperature)
        for _ in range(chem.ROUNDS):
            pool.meet(temperature)
            before, heat, thermal = pool.enthalpy(), pool.heat_kj, pool.thermal_kj
            moves += pool.exchange(temperature)
            assert pool.heat_kj >= heat and pool.thermal_kj >= thermal  # every move gives off heat
            assert pool.enthalpy() == before + (pool.heat_kj - heat) - (pool.thermal_kj - thermal)
        pool.temperature = temperature
        assert pool.audit(totals_of(counts)) == []
    assert moves > 500 and pool.heat_kj > 0 and pool.thermal_kj > 0 and sum(pool.moves.values()) == moves
    # a move that books heat it did not give off is seen
    forged = pool_of(counts, 2, ledger=True)
    forged.step(2000)
    real = Pool7.judge

    def cheat(self, a, b, temperature):
        touched = real(self, a, b, temperature)
        if touched:
            self.heat_kj += 1
        return touched
    forged.judge = cheat.__get__(forged)
    for _ in range(3):
        forged.step(1500)
    assert "the bond enthalpy of an exchange round is not what its moves gave off" in forged.audit(totals_of(counts))


def test_sulfur_opens_valences_toward_oxygen_only():
    counts = {"s": 50, "o": 400}
    pool = pool_of(counts)
    for _ in range(8):
        pool.step(2500)  # S=O holds (2572 K), O=O does not (2390 K)
    assert molecule_counts(pool) == {"o 3 s 1": 50} and pool.moves["open"] == 100  # sulfur trioxide
    sulfur = pool.el == ELEMENTS.index("s")
    assert set(pool.valence[sulfur].tolist()) == {6} and (pool.extra[sulfur] == 0).all()
    assert pool.audit(totals_of(counts)) == []
    pool.step(3000)  # too hot for S=O: the atoms are free and sulfur is back at 2
    assert len(pool.bonds) == 0
    pool.step(3000)
    assert set(pool.valence[sulfur].tolist()) == {2} and (pool.extra[sulfur] == 4).all()
    assert pool.audit(totals_of(counts)) == []
    # toward hydrogen sulfur stays 2: H2S and no more
    counts = {"s": 50, "h": 400}
    pool = pool_of(counts)
    for t in range(1, 21):
        pool.step(temperature_at(3000, 300, t, 20))
    assert molecule_counts(pool)["h 2 s 1"] == 50 and set(pool.valence[pool.el == ELEMENTS.index("s")].tolist()) == {2}
    # no other element opens anything, in any mix
    for mix in MIXES7:
        r = simulation().run(4, rules=7, mix=mix, sparks=3, **{**FIXED, "atoms": 5000})
        pool = final_pool(r)
        other = pool.el != ELEMENTS.index("s")
        assert (pool.valence[other] == chem._VAL[pool.el[other]]).all()
        assert set(pool.valence[~other].tolist()) <= {2, 4, 6}
        ea, eb, m = pool.el[pool.bond_a], pool.el[pool.bond_b], pool.bond_m
        held_by_others = np.zeros(pool.n, dtype=np.int64)
        for mine, theirs, atoms in ((ea, eb, pool.bond_a), (eb, ea, pool.bond_b)):
            mask = (mine == ELEMENTS.index("s")) & (theirs != ELEMENTS.index("o"))
            np.add.at(held_by_others, atoms[mask], m[mask])
        # the opened valences hold only oxygen: with anything else sulfur shares at most 2
        assert (held_by_others <= 2).all(), mix
        assert pool.audit(totals_of(element_counts(mix, 5000, rules=7))) == []
    # the audit sees a forged SH4: four bonds, none of them to oxygen
    forged = Pool7({"s": 1, "h": 4}, np.random.Generator(np.random.PCG64(1)))
    forged.valence[0] += 2
    forged.extra[0] -= 2
    forged._extra_left[0] -= 2
    forged._free[0] += 2
    for h in (1, 2, 3, 4):
        forged._set(0, h, 1)
    assert forged.audit(totals_of({"s": 1, "h": 4})) == ["an opened valence holds something other than oxygen"]


def test_sparks(sim, forty):
    r = sim.run(3, rules=7, mix="ocean", sparks=6, **FIXED)
    quench = sim.run(3, rules=7, mix="ocean", sparks=0, **FIXED)  # "quench only"
    n = FIXED["steps"]
    assert len(r.steps) == n + 7 and len(quench.steps) == n + 1 and r.steps[: n + 1] == quench.steps
    assert [s["spark"] for s in r.steps] == [0] * (n + 1) + [1, 2, 3, 4, 5, 6]
    assert all(s["hits"] == 0 for s in r.steps[: n + 1]) and all(s["hits"] > 60 for s in r.steps[n + 1:])
    assert {s["temperature"] for s in r.steps[n:]} == {FIXED["t_end"]}
    assert chem.expected_hits(5000) == 100 and chem.SPARK_PPM == 20000 and chem.SPARK_MAX_ATOMS == 64
    # over all sparks of the forty seeds the hits are what the rule expects: 2 percent of the small groups
    hits = expected = sparks = 0
    for seed, run in forty.items():
        first = run.params["steps"] + 1
        for t in range(first, len(run.steps)):
            sparks += 1
            hits += run.steps[t]["hits"]
            expected += chem.expected_hits(run.steps[t - 1]["molecules"])  # all but a grain or two are small
            assert run.steps[t]["hits"] <= run.steps[t - 1]["molecules"]
    assert sparks == sum(run.params["sparks"] for run in forty.values()) > 250
    assert abs(hits - expected) < 4 * math.sqrt(expected) and hits > 20000  # measured: within 1 %
    # a grain is not atomised: after 20 sparks the rock is still one grain
    rocky = sim.run(3, rules=7, mix="rocky", sparks=20, **FIXED).steps
    assert all(s["grains"] == 1 and s["rock"] > 19500 for s in rocky[n:])


def test_pool_laws_hold_at_every_step():
    for mix in MIXES7:
        params = {"mix": mix, "t_start": 5000, "t_end": 250, "steps": 25, "atoms": 6000, "sparks": 4, "rules": 7}
        counts = element_counts(mix, 6000, rules=7)
        totals = totals_of(counts)
        noble = np.array([VALENCE[e] == 0 for e in ELEMENTS])
        for t, temperature, pool in chem.cool7(3, params, ledger=True):
            assert pool.audit(totals) == [], (mix, t)
            n = pool.n
            held = np.bincount(pool.bond_a, pool.bond_m, minlength=n) + np.bincount(pool.bond_b, pool.bond_m, minlength=n)
            assert (held <= pool.valence).all() and (pool.free >= 0).all() and not held[noble[pool.el]].any()
            comp, size = pool.census()
            assert (comp.sum(0) == totals).all() and not comp[size >= 2][:, noble].any()
            pairs = pool.pair(pool.bond_a, pool.bond_b)
            assert len(set(pairs.tolist())) == len(pairs)  # no two atoms are bonded twice
            # every bond could hold at this temperature
            assert (chem._K3[pool.el[pool.bond_a], pool.el[pool.bond_b], pool.bond_m] >= temperature).all()
            if temperature >= HOT_K:
                assert len(pool.bonds) == 0


def test_audit_sees_a_broken_law():
    counts = {"h": 40, "he": 10, "o": 20, "s": 4}
    totals = totals_of(counts)
    pool = pool_of(counts)
    pool.step(1500)
    assert pool.audit(totals) == [] and len(pool.bonds) > 0
    assert pool.audit(totals + 1) == ["an element's atom count changed"]
    he = int(np.flatnonzero(pool.el == ELEMENTS.index("he"))[0])
    forged = copy.deepcopy(pool)
    forged._set(he, 0, 1)
    problems = forged.audit(totals)
    assert "a noble gas is bonded" in problems and "an atom holds more bonds than its valence" in problems
    forged = copy.deepcopy(pool)
    a, b = int(pool.bond_a[0]), int(pool.bond_b[0])
    forged.nb[a][b] = forged.nb[b][a] = forged.bonds[min(a, b) * forged.n + max(a, b)] = 3  # no H-O or H-H triple bond
    forged._arrays = None
    assert "a bond of impossible order or an atom bonded to itself" in forged.audit(totals)
    forged = copy.deepcopy(pool)
    forged.valence[0] += 2  # hydrogen cannot open valences
    forged.free[0] += 2
    forged._free[0] += 2
    assert "an atom opened valences it does not have" in forged.audit(totals)
    forged = copy.deepcopy(pool)
    forged.temperature = 4000  # hotter than any bond in it
    assert "a bond that cannot hold at this temperature" in forged.audit(totals)


# ---------------------------------------------------------------------------------------------
# rollouts: metrics, summary, conservation
# ---------------------------------------------------------------------------------------------

def test_metrics_are_the_census(sim):
    for mix in MIXES7:
        r = sim.run(4, rules=7, mix=mix, sparks=3, **FIXED)
        pool = final_pool(r)
        counts, last = molecule_counts(pool), r.steps[-1]
        assert list(counts.values()) == sorted(counts.values(), reverse=True)
        assert last["molecules"] == sum(counts.values())
        for name, formula in MOLECULES7.items():
            assert last[name] == count_of(counts, formula), (mix, name)
        sizes = {f: sum(parse_dense(f).values()) for f in counts}
        assert last["other"] == sum(c for f, c in counts.items() if not any(same(parse_dense(f), parse(x)) for x in MOLECULES7.values()))
        assert last["chains"] == sum(c for f, c in counts.items() if parse_dense(f).get("C", 0) > 2)
        assert last["biggest"] == max(sizes.values())
        assert last["rock"] == sum(sizes[f] * c for f, c in counts.items() if sizes[f] >= chem.ROCK_ATOMS)
        assert last["grains"] == sum(c for f, c in counts.items() if sizes[f] >= chem.ROCK_ATOMS)
        assert last["organic"] == chem.ch_molecules(pool) and last["precursors"] == chem.precursor_molecules(pool)
        # organic by hand: groups with a C-H bond; precursors: those with a C-C, C-N or C-O bond as well
        _, _, labels = pool.groups()
        c, h, n, o = (ELEMENTS.index(e) for e in "chno")
        with_ch, with_cx = set(), set()
        for a, b in zip(pool.bond_a.tolist(), pool.bond_b.tolist()):
            pair = {int(pool.el[a]), int(pool.el[b])}
            if pair == {c, h}:
                with_ch.add(int(labels[a]))
            if c in pair and pair <= {c, n, o}:
                with_cx.add(int(labels[a]))
        assert last["organic"] == len(with_ch) and last["precursors"] == len(with_ch & with_cx)
        assert last["ch4"] + last["h2co"] <= last["organic"]  # methane and formaldehyde are organic
        ea, eb = pool.el[pool.bond_a], pool.el[pool.bond_b]
        for key, metal in (("si_o", "si"), ("mg_o", "mg"), ("fe_o", "fe")):
            x = ELEMENTS.index(metal)
            assert last[key] == int((((ea == x) & (eb == o)) | ((ea == o) & (eb == x))).sum())
        iron = ELEMENTS.index("fe")
        metal_atoms = sum(1 for a in np.flatnonzero(pool.el == iron).tolist()
                          if pool.nb[a] and all(pool.el[x] == iron for x in pool.nb[a]))
        assert last["fe_metal"] == metal_atoms
        for s in r.steps:
            assert 0 <= s["free_atoms"] <= 1 and all(s[k] >= 0 for k in COUNT_KEYS7)
            assert parse_num(num(s["free_atoms"])) == s["free_atoms"]  # what the line shows is the value


def test_organic_means_a_carbon_hydrogen_bond(sim):
    """The word has one meaning: methane is organic, a bare carbon cluster is not; round 6's count has its own name."""
    pool = Pool7({"c": 4, "h": 4, "o": 2}, np.random.Generator(np.random.PCG64(1)))
    c1, c2, c3, c4, h1, h2, h3, h4, o1, o2 = range(10)
    for h in (h1, h2, h3, h4):
        pool._set(c1, h, 1)  # methane
    assert chem.ch_molecules(pool) == 1 and chem.precursor_molecules(pool) == 0
    pool._set(c2, c3, 3)
    pool._set(c3, c4, 1)  # a bare C3 cluster: a chain, not organic
    pool._set(c4, o1, 2)
    m = chem.metrics7(pool, 300)
    assert (m["organic"], m["precursors"], m["chains"], m["ch4"]) == (1, 0, 1, 1)
    pool._set(c1, h4, 0)
    pool._set(c1, o2, 1)
    pool._set(o2, h4, 1)  # methanol: a C-H bond and a C-O bond
    m = chem.metrics7(pool, 300)
    assert (m["organic"], m["precursors"], m["chains"], m["ch4"]) == (1, 1, 1, 0)
    # the function world used to keep for itself gives the same count on a round-6 pool
    from haishool.cosmos import world
    r6 = sim.run(3, mix="ocean", **FIXED)
    old = None
    for _, _, old in chem.cool(r6.seed, r6.params):
        pass
    assert chem.ch_molecules(old) == world.organic_molecules(old) > r6.steps[-1]["organic"] == 0


def test_summary_follows_from_the_steps(forty):
    for r in forty.values():
        last, summary = r.steps[-1], r.summary
        for k in COUNT_KEYS7 - {"hits"}:
            assert summary[k] == last[k]
        assert summary["mix"] == r.params["mix"] and summary["water"] == last["water"] == chem.water_state(r.params["t_end"])
        assert summary["water_share"] == chem.water_share(last["h2o"], last["molecules"])
        assert summary["water_share"] == pytest.approx(last["h2o"] / last["molecules"], rel=0.005)
        assert summary["water_atoms"] == chem.water_atoms(last["h2o"], r.params["atoms"])
        assert isinstance(summary["water_share"], float) and isinstance(summary["water_atoms"], float)
        first = summary["first_water_step"]
        if first == "never":
            assert all(s["h2o"] == 0 for s in r.steps)
        else:
            assert r.steps[first]["h2o"] > 0 and all(s["h2o"] == 0 for s in r.steps[:first])
    assert chem.water_share(0, 0) == 0.0 and isinstance(chem.water_share(0, 0), float)


def test_forty_seeds_conserve(sim, forty):
    mixes = set()
    for seed, r in forty.items():
        assert r.params == {**random_params(random.Random(seed), rules=7), "rules": 7}
        assert sim.conserved(r) == chem.Verdict(True, None, ""), seed
        assert chem.conserved(r).ok
        mixes.add(r.params["mix"])
    assert mixes == set(MIXES7)


def test_conserved_is_a_gate(sim):
    r = sim.run(3, rules=7, mix="ocean", sparks=3, **FIXED)
    assert sim.conserved(r).ok
    for change, reason in [(lambda f: f.steps[12].update(h2o=f.steps[12]["h2o"] + 40), "other is not"),
                           (lambda f: f.steps[5].update(o2=-1), "negative"),
                           (lambda f: f.steps[20].update(molecules=10), "other is not"),
                           (lambda f: f.steps[20].update(precursors=f.steps[20]["organic"] + 1), "organic, precursors"),
                           (lambda f: f.steps[20].update(stage="frozen"), "do not follow from the temperature"),
                           (lambda f: f.steps[20].update(water="ice"), "do not follow from the temperature"),
                           (lambda f: f.steps[-1].update(spark=1), "sparks are not counted in order"),
                           (lambda f: f.steps[3].update(hits=7), "sparks are not counted in order"),
                           (lambda f: f.steps[-1].update(rock=5), "rock, grains and biggest"),
                           (lambda f: f.summary.update(water_share=0.5), "not what its seed and parameters give"),
                           (lambda f: f.steps[-1].pop("fe_metal"), "a metric is missing")]:
        forged = copy.deepcopy(r)
        change(forged)
        verdict = sim.conserved(forged)
        assert not verdict.ok and reason in verdict.reason, (reason, verdict)
    forged = copy.deepcopy(r)
    forged.steps[-1]["h2o"] += 1  # a number that fits every budget but is not the replay's
    forged.steps[-1]["other"] -= 1
    assert sim.conserved(forged).reason == "the rollout is not what its seed and parameters give"
    forged = copy.deepcopy(r)
    forged.sim = "gravity"
    assert not sim.conserved(forged).ok
    for bad in [{"steps": 0}, {"sparks": -1}, {"mix": "lava"}, {"rules": 8}, {"atoms": True}, {"extra": 1}]:
        forged = copy.deepcopy(r)
        forged.params.update(bad)
        assert sim.conserved(forged) == chem.Verdict(False, None, "not a chem rollout")
    # a forged rollout is judged, never raised on: a word where a number belongs, a number where a word belongs
    for t, key, value in [(20, "h2o", "many"), (20, "molecules", None), (20, "free_atoms", "half"), (20, "stage", 3),
                          (20, "temperature", "hot"), (20, "rock", 1.5), (20, "h2", True), (20, "free_atoms", float("nan")),
                          (len(r.steps) - 2, "molecules", "many")]:
        forged = copy.deepcopy(r)
        forged.steps[t][key] = value
        verdict = sim.conserved(forged)
        assert not verdict.ok and "not of its metric's kind" in verdict.reason, (key, value, verdict)
    for seed in ("3", 3.0, -3, True, None):
        forged = copy.deepcopy(r)
        forged.seed = seed
        assert sim.conserved(forged) == chem.Verdict(False, None, "not a chem rollout")
    forged = copy.deepcopy(r)
    forged.steps = []
    assert not sim.conserved(forged).ok
    # a rules-7 rollout is not judged by round 6's laws, nor the other way round
    forged = copy.deepcopy(sim.run(3))
    forged.params["rules"] = 7
    assert not sim.conserved(forged).ok
    forged = copy.deepcopy(r)
    del forged.params["rules"]
    assert not sim.conserved(forged).ok


def test_odd_parameters_conserve_too(sim):
    """Pools of a few atoms, one step, heating, a start below every bond: the laws hold and every line is dense."""
    rng = random.Random(77)
    runs = 0
    for _ in range(60):
        params = {"mix": rng.choice(list(MIXES7)), "t_start": rng.choice([rng.randint(1, 8000), rng.randint(3000, 6000)]),
                  "t_end": rng.choice([rng.randint(1, 8000), rng.randint(100, 1500)]), "steps": rng.randint(1, 12),
                  "atoms": rng.choice([1, 2, 5, 17, 60, 150, 400, 1000]), "sparks": rng.choice([0, 0, 1, 3, 12])}
        seed = rng.randint(1, 10 ** 6)
        if chem.bad_params7({**params, "rules": 7}):
            continue  # too few atoms for the mix
        r = sim.run(seed, rules=7, **params)
        runs += 1
        assert sim.conserved(r) == chem.Verdict(True, None, ""), (seed, params)
        for ln in chem.lines(r, every=5):
            assert is_dense(ln.text) and n_tokens(ln.text) <= MAX_TOKENS, ln.text
    assert runs > 50


def test_counted_molecules_fit_into_the_atoms(sim, forty):
    for r in forty.values():
        counts = element_counts(r.params["mix"], r.params["atoms"], rules=7)
        for s in r.steps:
            assert budget7(s, counts) == []
    r = sim.run(3, rules=7, mix="ocean", sparks=0, **FIXED)
    counts = element_counts("ocean", FIXED["atoms"], rules=7)
    last = dict(r.steps[-1])

    def with_(**changes):  # keep the line closed, so only the budget speaks
        step = {**last, **changes}
        step["other"] = step["molecules"] - sum(step[k] for k in MOLECULES7)
        return step
    assert budget7(with_(h2o=counts["o"] + 1, molecules=last["molecules"] + counts["o"] + 1 - last["h2o"]), counts)[:2] \
        == ["more h in the counted molecules than the mix holds", "more o in the counted molecules than the mix holds"]
    assert "more c in the counted molecules than the mix holds" in budget7(with_(organic=counts["c"] + 1), counts)
    assert "more c in the counted molecules than the mix holds" in budget7(with_(chains=counts["c"] // 3 + 1), counts)
    assert "more atoms in molecules than are bonded" in budget7(with_(molecules=FIXED["atoms"]), counts)
    assert budget7(with_(free_atoms=0.5), counts) == ["more atoms in molecules than are bonded"]
    assert budget7(with_(nacl=counts["na"] + 1, molecules=last["molecules"] + 200), counts)[:2] \
        == ["more na in the counted molecules than the mix holds", "more cl in the counted molecules than the mix holds"]
    assert budget7(with_(mg_o=2 * counts["mg"] + 1), counts) == ["more bonds to oxygen than the valences allow"]
    assert budget7(with_(fe_metal=1), counts) == ["more iron in metal and oxide than the mix holds"]
    assert budget7(with_(grains=last["grains"] + 1), counts) == ["rock, grains and biggest do not fit each other"]
    assert budget7({**last, "other": last["other"] + 1}, counts) == ["other is not the molecules without a name"]


def test_heating_breaks_the_molecules_again(sim):
    r = sim.run(3, rules=7, mix="ocean", t_start=500, t_end=5000, steps=30, atoms=20000, sparks=0)
    assert sim.conserved(r).ok
    assert max(s["h2o"] for s in r.steps) > 5000  # at 500 K everything that holds forms at once
    assert r.steps[-1]["molecules"] == 0 and r.steps[-1]["free_atoms"] == 1 and r.steps[-1]["stage"] == "hot"
    small = sim.run(3, rules=7, mix="rocky", t_start=300, t_end=5000, steps=10, atoms=5000, sparks=2)
    assert sim.conserved(small).ok and max(s["rock"] for s in small.steps) > 4500 and small.steps[-1]["rock"] == 0
    assert [s["temperature"] for s in small.steps][-3:] == [5000, 5000, 5000] and small.steps[-1]["spark"] == 2


def test_a_run_takes_under_two_seconds(sim):
    best = 9.0
    for seed in (1, 2):
        for mix in ("ocean", "carbon"):  # the slowest mixes, with the most of everything
            t0 = time.perf_counter()
            sim.run(seed, rules=7, mix=mix, t_start=6000, t_end=100, steps=35, atoms=22000, sparks=20)
            best = min(best, time.perf_counter() - t0)
    assert best < 2.0  # measured: 0.9-1.4 s, depending on what else runs on the machine


# ---------------------------------------------------------------------------------------------
# plausibility targets, measured on seeds 1-40 with their own parameters
# ---------------------------------------------------------------------------------------------

def test_hot_gas_is_atomic_and_the_stages_come_in_order(forty):
    hot = 0
    for r in forty.values():
        cooling = r.steps[: r.params["steps"] + 1]
        temps = [s["temperature"] for s in cooling]
        assert temps[0] == r.params["t_start"] and temps[-1] == r.params["t_end"] and temps == sorted(temps, reverse=True)
        order = [STAGES7.index(s["stage"]) for s in r.steps]
        assert order == sorted(order)
        for t, s in enumerate(r.steps):
            assert s["stage"] == chem.stage7(s["temperature"], s["free_atoms"])
            assert s["water"] == chem.water_state(s["temperature"]) and s["solids"] == chem.solids_at(s["temperature"])
            assert (s["stage"] == "hot") == (s["temperature"] >= HOT_K)
            assert s["temperature"] == (temperature_at(r.params["t_start"], r.params["t_end"], t, r.params["steps"])
                                        if t <= r.params["steps"] else r.params["t_end"])
            if s["temperature"] >= HOT_K:
                hot += 1
                assert s["free_atoms"] == 1 and s["molecules"] == 0 and s["biggest"] == 1
        assert r.steps[0]["free_atoms"] == 1 and r.steps[0]["molecules"] == 0
        assert r.steps[-1]["stage"] == "molecular"
    assert hot > 50


def test_plausibility_targets_on_forty_seeds(forty):
    seen = {mix: 0 for mix in MIXES7}
    water_states = set()
    quench, sparked = [], []
    for seed, r in forty.items():
        mix, last, atoms = r.params["mix"], r.steps[-1], r.params["atoms"]
        have = element_counts(mix, atoms, rules=7)
        ranked = list(molecule_counts(final_pool(r)))
        seen[mix] += 1
        assert last["co"] <= 1  # the toy still ends without carbon monoxide
        if mix == "cosmic":  # hydrogen gas with water, methane and ammonia; helium and neon stay free
            assert ranked[:3] == ["h 2", "h 2 o 1", "h 4 c 1"] and last["h2"] > 0.98 * last["molecules"]
            assert last["h2o"] > 0.78 * have["o"] and last["ch4"] > 0.8 * have["c"]  # measured: 81-94 %, 86-98 %
            if r.params["sparks"]:
                assert last["nh3"] > 0.65 * have["n"]  # measured: 73-100 %
            assert 0.0885 <= last["free_atoms"] <= 0.0889 and last["rock"] == 0 and last["precursors"] == 0
            assert last["o2"] == 0 and last["co2"] == 0 and last["organic"] >= last["ch4"]
        elif mix == "rocky":  # rock condenses by itself into one network of single bonds
            assert last["grains"] == 1 and last["rock"] == last["biggest"] > 0.97 * atoms  # measured: 97.7-99.1 %
            assert last["si_o"] > 0.95 * 4 * have["si"] and last["mg_o"] > 0.99 * 2 * have["mg"]
            assert last["fe_o"] > 0.93 * 2 * have["fe"] and last["fe_metal"] <= 1  # the toy has no core
            assert last["molecules"] < 250 and last["o2"] <= 40 and last["h2"] <= 20 and last["h2o"] <= 10
        elif mix == "ocean":  # water and salt; carbon as carbon dioxide; few organics unless there are sparks
            assert ranked[:2] == ["h 2 o 1", "na 1 cl 1"]
            assert 0.77 < r.summary["water_share"] < 0.9 and 0.73 < r.summary["water_atoms"] < 0.85
            assert last["nacl"] > 0.55 * have["na"] and last["co2"] > 10 and last["ch4"] <= 2 and last["n2"] > 10
            assert last["o2"] < 0.03 * last["molecules"] and last["precursors"] == pytest.approx(last["organic"], abs=2)
            (sparked if r.params["sparks"] else quench).append(last["precursors"])
            water_states.add(last["water"])
        elif mix == "reducing":  # the gas of Miller's experiment: hydrogen left over, methane, ammonia, formaldehyde
            assert ranked[:2] == ["h 2 o 1", "h 2"] and last["h2"] > 500 and last["nh3"] > 150 and last["ch4"] > 50
            assert last["h2co"] > 50 and last["precursors"] > 70 and last["o2"] < 20 and last["rock"] == 0
            water_states.add(last["water"])
        else:  # carbon: more carbon than oxygen, so methane, chains and soot
            assert ranked[0] == "h 4 c 1" and ranked[1] in ("h 3 n 1", "h 2 o 1") and last["ch4"] > 1300
            assert last["organic"] > 1600 and last["chains"] > 100 and last["rock"] > 1000 and last["h2s"] > 100
            assert last["co2"] == 0 and last["o2"] == 0 and last["precursors"] > 250
    assert min(seen.values()) >= 7
    assert water_states == set(WATER_STATES)  # a seed's own watery run ends as ice, liquid or vapor
    # an ocean without an energy source makes few precursors; sparks raise them (measured: 9-17 against 37-76)
    assert len(quench) >= 2 and len(sparked) >= 5 and max(quench) < 25 < min(sparked)
    # the Miller-Urey contrast: every reducing run beats every quenched ocean (measured: 91-202 against 9-17)
    reducing = [r.steps[-1]["precursors"] for r in forty.values() if r.params["mix"] == "reducing"]
    assert min(reducing) > 3 * max(quench)


def test_what_the_docstring_admits(forty):
    """Two things the toy gets wrong and says so: its rock forms hotter than real rock condenses, and
    its sulfur hardly ever ends as a free oxide."""
    first_rock = {"rocky": [], "carbon": []}
    oxides = {"SO2": 0, "SO3": 0, "H2SO4": 0}
    for r in forty.values():
        mix = r.params["mix"]
        if mix in first_rock:
            s = next(s for s in r.steps if s["rock"])
            first_rock[mix].append(s["temperature"])
            if mix == "rocky":  # the table has no solid yet where the pool already has a network
                assert s["solids"] == "none" and s["temperature"] > CONDENSE_K["corundum"] > CONDENSE_K["silicate"]
        if mix == "ocean":
            counts = molecule_counts(final_pool(r))
            for formula in oxides:
                assert count_of(counts, formula) <= 3
                oxides[formula] += count_of(counts, formula)
    # measured: 2180-2225 K (Si-O holds below 2232 K, Mg-O below 2395 K) and 1563-1652 K (C-C holds below 1660 K)
    assert len(first_rock["rocky"]) == 7 and all(2100 < k < stability7("mg", "o") for k in first_rock["rocky"])
    assert len(first_rock["carbon"]) == 7 and all(1500 < k < stability7("c", "c") for k in first_rock["carbon"])
    assert oxides["SO3"] == 0 and oxides["SO2"] <= 8 and oxides["H2SO4"] <= 3  # measured: 0, 5 and 1 in 8 runs


def test_sparks_raise_the_precursors_and_a_reducing_gas_makes_more(sim):
    """Fixed parameters, seeds 1-4, so that only the mix and the sparks differ. Measured (seeds 1-5):
    ocean 14-21 without sparks and 70-90 after 10; reducing 117-144 and 165-181."""
    out = {}
    for mix in ("ocean", "reducing"):
        for sparks in (0, 10):
            out[mix, sparks] = [sim.run(seed, rules=7, mix=mix, sparks=sparks, **FIXED).steps[-1] for seed in (1, 2, 3, 4)]
    pre = {key: [s["precursors"] for s in rows] for key, rows in out.items()}
    assert max(pre["ocean", 0]) < 30 and min(pre["ocean", 10]) > 50
    assert min(pre["ocean", 10]) > 2.5 * max(pre["ocean", 0])
    assert min(pre["reducing", 0]) > 100 and min(pre["reducing", 10]) > max(pre["reducing", 0])
    assert min(pre["reducing", 0]) > 4 * max(pre["ocean", 0]) and min(pre["reducing", 10]) > 1.5 * max(pre["ocean", 10])
    # the docstring's "about 7 times" without sparks and "about 2 times" after ten (measured on seeds 1-4: 7.4 and 2.2)
    assert 6 < sum(pre["reducing", 0]) / sum(pre["ocean", 0]) < 9
    assert 1.7 < sum(pre["reducing", 10]) / sum(pre["ocean", 10]) < 2.7
    # sparks also destroy: the ocean loses water (5475-5492 -> 5053-5096) and gains H2, O2 and H2O2
    assert min(s["h2o"] for s in out["ocean", 0]) > max(s["h2o"] for s in out["ocean", 10]) + 300
    assert max(s["h2"] for s in out["ocean", 0]) <= 2 and min(s["h2"] for s in out["ocean", 10]) > 80
    # the oxidising ocean ends without hydrogen, the reducing gas with hundreds of H2 and no O2
    assert all(s["o2"] == 0 and s["h2"] > 400 for s in out["reducing", 0])
    assert all(10 <= s["o2"] <= 40 for s in out["ocean", 0])  # the leftover the docstring admits


# ---------------------------------------------------------------------------------------------
# lessons
# ---------------------------------------------------------------------------------------------

FUNCTIONS = {"schedule_temperature": "temperature_at", "unit_enthalpy": "unit_enthalpy", "stable_below_k": "stable_below_k",
             "bond_order": "bond_order", "free_valence": "free_valence", "reaction_heat": "reaction_heat",
             "move_allowed": "move_allowed", "stage": "stage7", "water_state": "water_state", "condensed": "condensed",
             "expected_hits": "expected_hits", "water_share": "water_share", "water_atoms": "water_atoms"}


def test_lessons_are_the_functions_of_the_simulation():
    assert chem.lesson_gate() is LESSONS and LESSONS.sim == "chem" and LESSONS.topic == "predict_chem"
    assert set(LESSONS.rules) == set(FUNCTIONS) and len(LESSONS.rules) >= 8
    for name, function in FUNCTIONS.items():
        assert LESSONS.rules[name].compute is getattr(chem, function), name  # the very function, not a copy
    lessons = LESSONS.generate(random.Random(1), 400)
    assert len(lessons) == 400 and {ln.prompt.split()[2] for ln in lessons} == set(FUNCTIONS)
    assert [ln.text for ln in lessons] == [ln.text for ln in LESSONS.generate(random.Random(1), 400)]
    answers = set()
    for ln in lessons:
        assert ln.topic == "predict_chem" and ln.kind == "calc" and is_dense(ln.text) and n_tokens(ln.text) <= MAX_TOKENS
        assert LESSONS.owns(ln.prompt) and chem.owns(ln.prompt)
        v = LESSONS.check(ln.prompt, ln.answer)
        assert v.ok and v.expected == ln.answer, (ln.text, v)
        name, values = LESSONS.parse(ln.prompt)
        truth = getattr(chem, FUNCTIONS[name])(*values)  # what the simulation's own function returns
        assert ln.answer == (truth if isinstance(truth, str) else num(truth)), ln.text
        assert not LESSONS.check(ln.prompt, wrong(ln.answer)).ok, ln.text
        assert ln.meta["split_key"] == LESSONS.split_key(ln.prompt)
        answers.add(ln.answer)
    # every word a lesson can answer with occurs
    assert {"hot", "atomic", "forming", "molecular", "ice", "liquid", "vapor", "solid", "yes", "no"} <= answers
    for ln in LESSONS.records():
        assert ln.kind == "record" and is_dense(ln.text) and ln.text.startswith("chem predict ")
    for prompt, answer in [("chem predict stable_below_k enthalpy_kj 4 6 4", "2 2 2 7"),
                           ("chem predict unit_enthalpy atoms_kj 6 8 5 point 2 formation_kj minus 2 4 1 point 8 units 2", "4 6 4"),
                           ("chem predict bond_order free_a 4 free_b 2 highest 3", "2"),
                           ("chem predict free_valence valence 4 held 3", "1"),
                           ("chem predict reaction_heat made_kj 1 1 0 4 broken_kj 8 7 1", "2 3 3"),
                           ("chem predict reaction_heat made_kj 4 3 6 broken_kj 8 9 6", "minus 4 6 0"),
                           ("chem predict move_allowed made_kj 1 1 0 4 broken_kj 8 7 1", "yes"),
                           ("chem predict move_allowed made_kj 8 7 2 broken_kj 8 7 2", "no"),
                           ("chem predict stage temperature 2 0 0 0 free_atoms 0 point 4", "molecular"),
                           ("chem predict stage temperature 4 5 3 6 free_atoms 1", "hot"),
                           ("chem predict water_state temperature 2 9 0", "liquid"),
                           ("chem predict condensed below_k 1 8 0 temperature 1 5 0", "solid"),
                           ("chem predict expected_hits molecules 5 0 0 0", "1 0 0"),
                           ("chem predict water_share h2o 5 9 9 4 molecules 6 9 2 9", "0 point 8 6 5"),
                           ("chem predict water_atoms h2o 5 9 9 4 atoms 2 2 0 0 0", "0 point 8 1 7"),
                           ("chem predict schedule_temperature t_start 4 0 0 0 t_end 1 0 0 0 step 1 5 steps 3 0", "2 0 0 0")]:
        assert LESSONS.check(prompt, answer) == chem.Verdict(True, answer, "computed from the inputs in the question"), prompt
    for prompt in ["chem predict stable_below_k enthalpy_kj 4 6 4 point 5", "chem predict free_valence valence 2 held 3",
                   "chem predict water_share h2o 9 molecules 3", "chem predict schedule_temperature t_start 1 0 0 t_end 2 0 0 step 1 steps 3",
                   "chem predict bond_order free_a 0 free_b 2 highest 3", "chem predict stage temperature 2 0 0 0",
                   "chem predict unit_enthalpy atoms_kj 2 0 0 formation_kj 2 4 0 units 2"]:
        assert not LESSONS.owns(prompt), prompt


def test_lessons_are_true_of_the_rollouts(sim, forty):
    """What a lesson teaches is what the simulation does: its own steps follow the same functions."""
    from haishool.cosmos import predict
    for r in list(forty.values())[:12]:
        p = r.params
        for t, s in enumerate(r.steps):
            if t <= p["steps"]:
                prompt = (f"chem predict schedule_temperature t_start {num(p['t_start'])} t_end {num(p['t_end'])} "
                          f"step {num(t)} steps {num(p['steps'])}")
                assert LESSONS.check(prompt, num(s["temperature"])).ok
                assert predict.check(prompt, num(s["temperature"])).ok  # round 6's lesson gate agrees
            assert LESSONS.check(f"chem predict stage temperature {num(s['temperature'])} free_atoms {num(s['free_atoms'])}",
                                 s["stage"]).ok
            assert LESSONS.check(f"chem predict water_state temperature {num(s['temperature'])}", s["water"]).ok
            for substance, below in CONDENSE_K.items():
                verdict = LESSONS.check(f"chem predict condensed below_k {num(below)} temperature {num(s['temperature'])}", "solid")
                assert verdict.ok == (substance in s["solids"].split())
        last = r.steps[-1]
        assert LESSONS.check(f"chem predict water_share h2o {num(last['h2o'])} molecules {num(last['molecules'])}",
                             num(r.summary["water_share"])).ok
        assert LESSONS.check(f"chem predict water_atoms h2o {num(last['h2o'])} atoms {num(p['atoms'])}",
                             num(r.summary["water_atoms"])).ok
    # the table the pool reads is the lessons' arithmetic
    for a, b, m, kj, source in chem._bond_rows():
        assert chem._K3[ELEMENTS.index(a), ELEMENTS.index(b), m] == stable_below_k(kj) == stability7(a, b, m)
        assert LESSONS.check(f"chem predict stable_below_k enthalpy_kj {num(kj)}", num(stable_below_k(kj))).ok
    for name, (formula, formation, (a, b), order, units) in FORMATION_KJ.items():
        prompt = (f"chem predict unit_enthalpy atoms_kj {num(chem.atoms_enthalpy(formula), sig=6)} "
                  f"formation_kj {num(formation, sig=6)} units {num(units)}")
        assert LESSONS.check(prompt, num(bond_enthalpy(a, b, order) // order)).ok, name


def test_other_levels_still_read_this_module():
    from haishool.cosmos import predict, world
    assert predict.chem is chem and world.chem is chem
    assert predict.RULES["schedule_temperature"].compute is chem.temperature_at is LESSONS.rules["schedule_temperature"].compute
    gate = predict.gate("chem")
    prompt = "chem predict schedule_temperature t_start 4 0 0 0 t_end 1 0 0 0 step 1 5 steps 3 0"
    assert gate.check(prompt, "2 0 0 0").ok and LESSONS.check(prompt, "2 0 0 0").ok
    assert world.mix_of(290) == "ocean" and world.CHEM_START_K == 4000  # round 6's world is untouched
