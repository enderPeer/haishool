"""Rules 7 of the first hour: the corrections to ``haishool.cosmos.nucleo`` behind ``rules=7``.

Round 6 stays as it was (``tests/test_cosmos_nucleo.py`` and ``tests/test_round6_frozen.py`` are
its guard); this file tests the corrected behaviour and the seam between the two rule sets.
"""
import copy
import itertools
import math
import random
import time

import numpy as np
import pytest

from haishool.cosmos import Rollout, Simulation, close
from haishool.cosmos import nucleo as N
from haishool.evo import MAX_TOKENS, LessonGate, sig
from haishool.truth import Gate, Verdict, is_dense, num, parse_num

SEEDS = [1, 2, 3, 42, 777, 4242, 9998]
FORTY = range(1, 41)
#: every seed :func:`nucleo.generate` can draw
ALL_SEEDS = range(1, 9999)
LINES_PER_ROLLOUT = (1 + len(N.RANGES7) + 30 + 30 * len(N.KEYS7) + 29 * len(N.KEYS7) + len(N.FINAL_KEYS7))
Q_MEV, K_MEV = 1.29333236, 8.617333262e-11


def ntokens(text: str) -> int:
    return len(text.replace(".", " . ").split())


def run7(seed: int = 1, **params) -> Rollout:
    return N.run(seed, rules=7, **params)


@pytest.fixture(scope="module")
def sim():
    return N.simulation(rules=7)


@pytest.fixture(scope="module")
def all_lines():
    return {s: N.lines(N.rollout(s, rules=7)) for s in SEEDS}


@pytest.fixture(scope="module")
def every_rollout():
    return [N.rollout(s, rules=7) for s in ALL_SEEDS]


# ---------------------------------------------------------------------------------------------
# the seam: round 6 is the default and is untouched
# ---------------------------------------------------------------------------------------------

def test_without_rules_everything_is_round_6():
    a, b = N.run(7, **N.random_params(random.Random(7))), N.run(7, rules=6, **N.random_params(random.Random(7), rules=6))
    assert a.params == b.params and a.steps == b.steps and a.summary == b.summary
    assert "rules" not in a.params and "rules" not in N.run(1).params and N.run(1).params == N.DEFAULTS
    assert type(N.rollout(5)) is N.NucleoRollout and type(N.rollout(5, rules=6)) is N.NucleoRollout
    assert [ln.text for ln in N.lines(N.rollout(7))] == [ln.text for ln in N.lines(N.rollout(7), rules=6)]
    texts = {ln.text for ln in N.lines(N.rollout(1))}
    assert "q nucleo seed 1 final helium. a 0 point 2 0 2." in texts  # the line version 5 was trained on
    assert "nucleo seed 1 params. eta_factor 0 point 4 0 9. tau_n 9 3 6. t_freeze 1 point 4 4. expansion_factor 0 point 8 3 5." in texts
    assert not any(" rules " in t for t in texts)
    old = N.simulation()
    assert old.rules == 6 and old.KEYS is N.KEYS and old.DEFAULTS is N.DEFAULTS and old.records() == []
    assert old.run(3).steps == N.run(3).steps and old.random_params(random.Random(3)) == N.random_params(random.Random(3))
    assert N.records() == [] and N.records(rules=6) == []
    assert N.DEFAULTS["tau_n"] == 880.0 and N.DEFAULTS["t_d"] == 8e8 and "plasma" in N.STAGES


def test_twenty_seeds_are_round_6_with_and_without_the_word():
    """``rules=6`` is the default spelled out: the same bits, and no ``rules`` anywhere in a
    round-6 rollout or line."""
    for s in range(1, 21):
        for params in ({}, N.random_params(random.Random(s))):
            a, b = N.run(s, **params), N.run(s, rules=6, **params)
            assert type(a) is type(b) is N.NucleoRollout and list(a.params) == list(b.params) == list(N.DEFAULTS)
            assert [repr(v) for v in a.params.values()] == [repr(v) for v in b.params.values()]
            assert [repr(v) for st in a.steps for v in st.values()] == [repr(v) for st in b.steps for v in st.values()]
            assert [repr(v) for v in a.summary.values()] == [repr(v) for v in b.summary.values()]
            assert list(a.summary) == list(b.summary) == list(N.FINAL_KEYS)
        assert N.random_params(random.Random(s)) == N.random_params(random.Random(s), rules=6)
        assert list(N.random_params(random.Random(s))) == list(N.RANGES)
        texts = [ln.text for ln in N.lines(N.rollout(s))]
        assert texts == [ln.text for ln in N.lines(N.rollout(s, rules=6), rules=6)]
        assert not any("rules" in text for text in texts)
        assert all(ln.meta == {} for ln in N.lines(N.rollout(s)))
    assert "rules" not in N.DEFAULTS and "rules" not in N.RANGES and "rules" not in N.PARAMS


def test_only_the_rule_sets_6_and_7_exist():
    for bad in (5, 8, 0, True, "7", None, 6.5):
        with pytest.raises(ValueError):
            N.run(1, rules=bad)
        with pytest.raises(ValueError):
            N.random_params(random.Random(1), rules=bad)
        with pytest.raises(ValueError):
            N.rollout(1, rules=bad)
        with pytest.raises(ValueError):
            N.simulation(rules=bad)


def test_a_rules7_rollout_says_so_and_only_it():
    r = run7(5)
    assert r.sim == "nucleo" and r.params["rules"] == 7 and type(r) is N.NucleoRollout7
    assert {k: v for k, v in r.params.items() if k != "rules"} == N.DEFAULTS7
    assert list(r.params)[-1] == "rules" and type(r.params["rules"]) is int and type(r.params["neutrinos"]) is int
    assert run7(5).steps == run7(99).steps  # the seed is only a label for run()
    seeded = N.rollout(5, rules=7)
    assert seeded.params["rules"] == 7 and type(seeded) is N.NucleoRollout7
    assert set(r.summary) == set(N.FINAL_KEYS7) and "peak_n_p_ratio" not in r.summary
    assert all(set(s) == set(N.KEYS7) for s in r.steps) and len(r.steps) == 30
    assert [s["time"] for s in r.steps] == list(N.TIMES)  # a step number means the same time in both rule sets


def test_is_a_simulation_and_a_gate(sim):
    assert isinstance(sim, Simulation) and isinstance(sim, Gate)
    assert sim.sim == sim.topic == "nucleo" and sim.rules == 7
    assert sim.KEYS is N.KEYS7 and sim.FINAL_KEYS is N.FINAL_KEYS7 and sim.DEFAULTS is N.DEFAULTS7
    assert sim.run(5).params["rules"] == 7 and sim.run(5).steps == run7(5).steps
    assert sim.run(5, rules=6).steps == N.run(5).steps
    assert sim.random_params(random.Random(4)) == N.random_params(random.Random(4), rules=7)
    assert sim.rollout(4).steps == N.rollout(4, rules=7).steps
    assert [ln.text for ln in sim.lines(sim.rollout(4))] == [ln.text for ln in N.lines(N.rollout(4, rules=7))]
    assert [ln.text for ln in sim.records()] == [ln.text for ln in N.records(rules=7)]
    assert sim.lesson_gate() is N.LESSONS is N.lesson_gate() and isinstance(N.LESSONS, LessonGate)
    assert N.simulation().rules == 6  # a rules-7 object does not change the class


# ---------------------------------------------------------------------------------------------
# determinism and parameters
# ---------------------------------------------------------------------------------------------

def test_forty_seeds_conserved_and_bit_identical(sim):
    for s in FORTY:
        p = N.random_params(random.Random(s), rules=7)
        assert p == N.random_params(random.Random(s), rules=7)
        a, b = run7(s, **p), run7(s, **p)
        assert a.steps == b.steps and a.summary == b.summary and a.params == b.params
        assert [repr(v) for st in a.steps for v in st.values()] == [repr(v) for st in b.steps for v in st.values()]
        assert [repr(v) for v in a.summary.values()] == [repr(v) for v in b.summary.values()]
        assert a.steps == N.rollout(s, rules=7).steps and a.summary == N.rollout(s, rules=7).summary
        v = sim.conserved(a)
        assert v.ok, (s, v)
        assert max(abs(math.fsum(N._fractions(st)) - 1) for st in a.steps) < 1e-12
    assert [ln.text for ln in N.lines(N.rollout(7, rules=7))] == [ln.text for ln in N.lines(N.rollout(7, rules=7))]
    assert N.rollout(7, rules=7).steps != N.rollout(8, rules=7).steps


def test_random_params_are_a_universe_near_ours():
    seen = set()
    for s in range(400):
        rng = random.Random(s)
        p = N.random_params(rng, rules=7)
        assert list(p) == ["eta_factor", "tau_n", "neutrinos"] == list(N.RANGES7)
        assert 0.7 <= p["eta_factor"] <= 1.3 and p["eta_factor"] == float(f"{p['eta_factor']:.3g}")
        assert 870 <= p["tau_n"] <= 890 and p["tau_n"] == round(p["tau_n"])
        assert p["neutrinos"] in (2, 3, 4) and type(p["neutrinos"]) is int
        seen.add(p["neutrinos"])
        again = random.Random(s)  # the draw order: eta_factor, tau_n, neutrinos
        eta = float(f"{math.exp(again.uniform(math.log(0.7), math.log(1.3))):.3g}")
        tau = float(f"{again.uniform(870.0, 890.0):.3g}")
        assert p == {"eta_factor": eta, "tau_n": tau, "neutrinos": again.choice((2, 3, 3, 3, 4))}
    assert seen == {2, 3, 4}
    assert "t_freeze" not in N.RANGES7 and "expansion_factor" not in N.RANGES7 and "t_freeze" not in N.DEFAULTS7
    assert N.DEFAULTS7 == {"eta_factor": 1.0, "tau_n": 878.4, "neutrinos": 3, "kt_freeze": 0.787, "t1": 1e10,
                           "t_d": 9e8, "tau_bind": 30.0}
    assert set(N.PARAMS7) == set(N.DEFAULTS7)


def test_run_refuses_parameters_outside_the_rules():
    for unknown in ({"gravity": 3}, {"t_freeze": 1.2}, {"expansion_factor": 1.1}):
        with pytest.raises(TypeError):
            run7(1, **unknown)
    for bad in [{"eta_factor": 0.3}, {"eta_factor": 3.0}, {"eta_factor": 0.49}, {"eta_factor": 2.01}, {"tau_n": 700},
                {"tau_n": 1000}, {"neutrinos": 0}, {"neutrinos": 7}, {"tau_n": math.nan}, {"t1": math.inf},
                {"tau_bind": -1}, {"kt_freeze": 0}, {"kt_freeze": 0.02}]:
        with pytest.raises(ValueError):
            run7(1, **bad)
    assert run7(1, eta_factor=0.5).summary["helium"] < run7(1, eta_factor=2.0).summary["helium"]
    assert run7(1, neutrinos=3.044).params["neutrinos"] == 3.044  # the standard-model effective number


def test_a_run_that_comes_back_has_passed_its_gate():
    """Far-off fixed parameters (the binding time, the two calibrated temperatures, the clock) give
    a first hour the thirty steps cannot show. ``run`` refuses those instead of handing back a
    rollout its own gate rejects."""
    for bad, why in [({"tau_bind": 2000.0}, "end done"), ({"tau_bind": 300.0}, "thirty steps"),
                     ({"kt_freeze": 0.1}, "so few neutrons"), ({"t_d": 3e7}, "so few neutrons"),
                     ({"t1": 1e11}, "so few neutrons"), ({"t_d": 1e11}, "freeze out before")]:
        with pytest.raises(ValueError, match=why):
            run7(1, **bad)
    rng, came_back, refused = random.Random(11), 0, 0
    for i in range(3000):
        wide = i % 2 == 0  # half the draws far outside anything sensible, half near the defaults
        p = {"eta_factor": rng.uniform(0.5, 2.0), "tau_n": rng.uniform(800.0, 960.0), "neutrinos": rng.uniform(1.0, 6.0),
             "kt_freeze": math.exp(rng.uniform(math.log(0.05), math.log(20.0))) if wide else rng.uniform(0.7, 0.85),
             "t1": math.exp(rng.uniform(math.log(1e8), math.log(1e12))) if wide else rng.uniform(0.95e10, 1.05e10),
             "t_d": math.exp(rng.uniform(math.log(1e7), math.log(1e11))) if wide else rng.uniform(8e8, 1e9),
             "tau_bind": math.exp(rng.uniform(math.log(0.01), math.log(1e4))) if wide else rng.uniform(5.0, 100.0)}
        try:
            r = run7(i, **p)
        except ValueError:
            assert wide, p  # near the defaults every run comes back
            refused += 1
            continue
        assert N.conserved(r).ok, p
        came_back += 1
    assert came_back > 1600 and refused > 1000


# ---------------------------------------------------------------------------------------------
# lines
# ---------------------------------------------------------------------------------------------

def test_lines_are_dense_and_short(all_lines):
    kinds = set()
    for s, lines in all_lines.items():
        assert len(lines) == LINES_PER_ROLLOUT == 638
        assert len({ln.text for ln in lines}) == len(lines)
        for ln in lines:
            assert is_dense(ln.text), ln.text
            assert ntokens(ln.text) <= 90 <= MAX_TOKENS, ln.text
            assert ln.topic == "nucleo" and ln.meta == {"rules": 7}
            assert ln.prompt.startswith(f"nucleo seed {num(s)} rules 7 ")
            words = ln.prompt.split()
            if ln.kind != "record":
                assert "." not in ln.prompt and "." not in ln.answer
                assert ln.text == f"q {ln.prompt}. a {ln.answer}."
            kinds.add((ln.kind, next((w for w in ("next", "final", "param", "params.") if w in words), "step")))
    assert kinds == {("record", "params."), ("record", "step"), ("fact", "param"), ("fact", "step"),
                     ("calc", "next"), ("calc", "final")}


def test_forty_seeds_of_lines_are_dense_short_and_judged(sim):
    """Every line of 40 seeds: dense, at most 128 tokens, accepted by the gate as written and
    rejected with its first digit or its word changed."""
    longest = judged = 0
    for s in FORTY:
        lines = N.lines(N.rollout(s, rules=7))
        assert [ln.text for ln in lines] == [ln.text for ln in N.lines(N.run(s, rules=7, **N.random_params(random.Random(s), rules=7)))]
        for ln in lines:
            assert is_dense(ln.text) and ntokens(ln.text) <= MAX_TOKENS == 128, ln.text
            longest = max(longest, ntokens(ln.text))
            if ln.kind == "record":
                continue
            assert "." not in ln.prompt and "." not in ln.answer
            v = sim.check(ln.prompt, ln.answer)
            assert v.ok and v.expected == ln.answer, (ln.text, v)
            assert not sim.check(ln.prompt, _wrong(ln.answer)).ok, ln.text
            judged += 1
    assert longest == 84 and judged == 40 * (LINES_PER_ROLLOUT - 31)


def test_a_seven_digit_seed_still_fits():
    lines = N.lines(N.rollout(1234567, rules=7))
    assert max(ntokens(ln.text) for ln in lines) <= 89
    assert all(is_dense(ln.text) for ln in lines)
    assert all(N.check(ln.prompt, ln.answer).ok for ln in lines if ln.kind != "record")


def test_every_key_is_in_the_state_line_with_three_digits(all_lines):
    rec = [ln for ln in all_lines[1] if ln.kind == "record" and " step " in ln.text]
    assert len(rec) == 30 and N.STATE_KEYS7 == list(N.KEYS7)
    assert rec[0].text.startswith("nucleo seed 1 rules 7 step 0. time 0 point 1. temperature ")
    assert all(f" {k} " in ln.text for ln in rec for k in ("helium3", "lithium", "deuterium", "stage"))
    assert N._value7(0.24734113) == "0 point 2 4 7" and N._value7(1234.5) == "1 2 3 0" and N._value7(999.96) == "1 0 0 0"
    assert N._value7(3000.0) == "3 0 0 0" and N._value7(3) == "3" and N._value7("done") == "done"
    for ln in all_lines[1]:
        if ln.kind != "record" and ln.answer[0].isdigit():
            digits = "".join(w for w in ln.answer.split(" e ")[0].split() if w.isdigit()).strip("0")
            assert len(digits) <= 3, ln.text


def test_known_lines(all_lines):
    texts = {ln.text for ln in all_lines[1]}
    for want in [
        "nucleo seed 1 rules 7 params. eta_factor 0 point 7 6 1. tau_n 8 8 7. neutrinos 2.",
        "q nucleo seed 1 rules 7 param eta_factor. a 0 point 7 6 1.",
        "q nucleo seed 1 rules 7 param tau_n. a 8 8 7.",
        "q nucleo seed 1 rules 7 param neutrinos. a 2.",
        "nucleo seed 1 rules 7 step 0. time 0 point 1. temperature 3 point 3 1 e 1 0. n_p_ratio 0 point 6 3 5. "
        "free_neutrons 0 point 3 8 8. hydrogen 0 point 6 1 2. helium 0. deuterium 0. helium3 0. lithium 0. "
        "stage equilibrium.",
        "nucleo seed 1 rules 7 step 2 2. time 2 4 9. temperature 8 point 6 4 e 8. n_p_ratio 0 point 1 3 5. "
        "free_neutrons 0 point 0 6 8 7. hydrogen 0 point 8 3 2. helium 0 point 0 9 5 8. deuterium 0 point 0 0 2 3 7. "
        "helium3 4 point 4 1 e minus 6. lithium 1 point 2 9 e minus 1 0. stage bottleneck_opens.",
        "q nucleo seed 1 rules 7 step 8 stage. a freeze_out.",
        "q nucleo seed 1 rules 7 step 2 2 helium. a 0 point 0 9 5 8.",
        "q nucleo seed 1 rules 7 step 2 2 deuterium. a 0 point 0 0 2 3 7.",
        "q nucleo seed 1 rules 7 step 2 2 next helium. a 0 point 2 2 9.",
        "q nucleo seed 1 rules 7 step 2 2 next stage. a helium_forming.",
        "q nucleo seed 1 rules 7 final helium. a 0 point 2 3 3.",
        "q nucleo seed 1 rules 7 final hydrogen. a 0 point 7 6 7.",
        "q nucleo seed 1 rules 7 final traces. a 7 point 6 5 e minus 5.",
        "q nucleo seed 1 rules 7 final freeze_time. a 1 point 3 9.",
        "q nucleo seed 1 rules 7 final freeze_temperature. a 8 point 9 e 9.",
        "q nucleo seed 1 rules 7 final freeze_n_p_ratio. a 0 point 1 8 5.",
        "q nucleo seed 1 rules 7 final bottleneck_time. a 2 3 3.",
        "q nucleo seed 1 rules 7 final bottleneck_temperature. a 8 point 9 2 e 8.",
        "q nucleo seed 1 rules 7 final bottleneck_n_p_ratio. a 0 point 1 3 7.",
        "q nucleo seed 1 rules 7 final expansion_factor. a 0 point 9 1 5.",
        "q nucleo seed 1 rules 7 final end_time. a 3 0 0 0.",
    ]:
        assert want in texts, want


def test_no_prompt_has_two_truths(all_lines):
    """The seam of the audit: a rules-7 line never shares its prompt with a round-6 line, and
    each prompt is judged by its own rule set."""
    for s, lines7 in all_lines.items():
        lines6 = N.lines(N.rollout(s))
        prompts6 = {ln.prompt for ln in lines6}
        prompts7 = {ln.prompt for ln in lines7}
        assert not prompts6 & prompts7
        assert all(" rules 7 " in p for p in prompts7) and not any(" rules " in p for p in prompts6)
    r6, r7 = N.rollout(1), N.rollout(1, rules=7)
    assert N.check("nucleo seed 1 final helium", "0 point 2 0 2").ok
    assert N.check("nucleo seed 1 rules 7 final helium", "0 point 2 3 3").ok
    assert not N.check("nucleo seed 1 final helium", "0 point 2 3 3").ok
    assert not N.check("nucleo seed 1 rules 7 final helium", "0 point 2 0 2").ok
    assert N.check("nucleo seed 1 final helium", "").expected == r6.value(r6.summary["helium"])
    assert N.check("nucleo seed 1 rules 7 final helium", "").expected == r7.value(r7.summary["helium"])


def test_lines_round_trip_the_numbers():
    r = N.rollout(3, rules=7)
    for ln in N.lines(r):
        if ln.kind == "record":
            continue
        words = ln.prompt.split()
        key = words[-1]
        if "final" in words:
            v = r.summary[key]
        elif "param" in words:
            v = r.params[key]
        else:
            end = words.index("next") if "next" in words else len(words) - 1
            step = parse_num(words[words.index("step") + 1:end])
            v = r.steps[step + ("next" in words)][key]
        if isinstance(v, str):
            assert ln.answer == v
        else:
            assert parse_num(ln.answer) == sig(v) and close(parse_num(ln.answer), v, rel=0.006, abs_=0.0)


def test_lines_want_the_seed_rollout():
    with pytest.raises(ValueError):
        N.lines(run7(7))  # the default universe is not seed 7's
    with pytest.raises(ValueError):
        N.lines(run7(7, **N.random_params(random.Random(8), rules=7)))
    with pytest.raises(ValueError):
        N.lines(N.rollout(7, rules=7), horizon=2)
    with pytest.raises(ValueError):
        N.lines(N.rollout(7, rules=7), rules=6)
    with pytest.raises(ValueError):
        N.lines(N.rollout(7), rules=7)
    with pytest.raises(ValueError):
        N.lines(run7(-3, **N.random_params(random.Random(-3), rules=7)))
    changed = N.rollout(7, rules=7)
    changed.steps[25]["helium"] = 0.9
    with pytest.raises(ValueError):
        N.lines(changed)
    assert len(N.lines(N.rollout(7, rules=7), rules=7)) == LINES_PER_ROLLOUT
    assert len(N.lines(N.rollout(7, rules=7), keys=["helium", "stage"])) == 1 + 3 + 30 + 30 * 2 + 29 * 2 + len(N.FINAL_KEYS7)


def test_records_say_the_unit_of_every_key():
    rec = N.records(rules=7)
    texts = [ln.text for ln in rec]
    assert len(rec) == len(N.KEYS7) + len(N.FINAL_KEYS7) + len(N.RANGES7) + 1 == len(set(texts))
    for ln in rec:
        assert ln.kind == "record" and ln.topic == "nucleo" and is_dense(ln.text) and ntokens(ln.text) <= 30
    for want in ["nucleo rules 7 key deuterium. unit nuclei_per_hydrogen.",
                 "nucleo rules 7 key helium. unit mass_fraction.",
                 "nucleo rules 7 key temperature. unit kelvin.",
                 "nucleo rules 7 final key traces. unit mass_fraction.",
                 "nucleo rules 7 final key bottleneck_time. unit seconds.",
                 "nucleo rules 7 param key neutrinos. unit families.",
                 "nucleo rules 7 stages. equilibrium. freeze_out. decay. bottleneck_opens. helium_forming. done."]:
        assert want in texts, want
    assert set(N.UNITS7) == set(N.KEYS7) | set(N.FINAL_KEYS7) | set(N.RANGES7)


# ---------------------------------------------------------------------------------------------
# the gate: check() and owns()
# ---------------------------------------------------------------------------------------------

def test_gate_agrees_with_every_line(sim, all_lines):
    for lines in all_lines.values():
        for ln in lines:
            if ln.kind == "record":
                continue
            assert sim.owns(ln.prompt) and N.owns(ln.prompt) and N.simulation().owns(ln.prompt), ln.text
            v = sim.check(ln.prompt, ln.answer)
            assert v.ok, (ln.text, v)
            assert v.expected == ln.answer
            assert N.check("q " + ln.prompt + ".", ln.answer).ok


def _wrong(answer: str) -> str:
    """Change the first digit, or the word."""
    words = answer.split()
    if words[0].isdigit():
        words[0] = "8" if words[0] == "9" else str(int(words[0]) + 1)
        return " ".join(words)
    return "equilibrium" if answer != "equilibrium" else "done"


def _nudged(answer: str) -> str | None:
    """The third digit one off (a change of under one percent that round 6 would let pass), or
    ``None`` when the answer has no three digits to nudge."""
    mantissa, _, exponent = answer.partition(" e ")
    words = mantissa.split()
    digits = [i for i, w in enumerate(words) if w.isdigit()]
    lead = next((i for i in digits if words[i] != "0"), None)
    significant = [i for i in digits if lead is not None and i >= lead]
    if len(significant) < 3:
        return None
    last = significant[2]
    words[last] = "8" if words[last] == "9" else str(int(words[last]) + 1)
    return " ".join(words) + (" e " + exponent if exponent else "")


def test_gate_rejects_a_changed_digit_or_word(sim, all_lines):
    nudged = 0
    for lines in all_lines.values():
        for ln in lines:
            if ln.kind == "record":
                continue
            v = sim.check(ln.prompt, _wrong(ln.answer))
            assert not v.ok, (ln.text, _wrong(ln.answer), v)
            assert v.expected == ln.answer
            if " e " in ln.answer:  # a wrong exponent, or its sign lost
                mantissa, exp = ln.answer.split(" e ")
                flipped = exp.removeprefix("minus ") if exp.startswith("minus ") else "minus " + exp
                assert not sim.check(ln.prompt, f"{mantissa} e {flipped}").ok, ln.text
            if ln.answer not in N.STAGES7 and ln.answer != "0":
                assert not sim.check(ln.prompt, "minus " + ln.answer).ok, ln.text
                assert not sim.check(ln.prompt, "0").ok, ln.text
            if ln.answer not in N.STAGES7 and _nudged(ln.answer) is not None:
                assert not sim.check(ln.prompt, _nudged(ln.answer)).ok, (ln.text, _nudged(ln.answer))
                nudged += 1
    assert nudged > 2000
    assert not sim.check("nucleo seed 1 rules 7 step 3 helium", "abc").ok
    assert sim.check("nucleo seed 1 rules 7 step 3 helium", "abc").expected == "0"
    assert not sim.check("nucleo seed 1 rules 7 step 3 stage", "0").ok
    assert sim.check("nucleo seed 1 rules 7 step 3 stage", "0").expected == "equilibrium"
    assert not sim.check("nucleo seed 1 rules 7 step 3 stage", "plasma").ok  # the round-6 word


def test_numbers_are_judged_at_the_printed_precision():
    """Round 6 lets 5 percent pass. Under rules 7 the universes are so alike that one constant
    would then pass for most seeds, so a number must round to the three digits the run prints."""
    v = N.rollout(1, rules=7).summary["lithium"]
    assert N.check("nucleo seed 1 final lithium", num(N.rollout(1).summary["lithium"] * 1.04, sig=4)).ok  # round 6
    assert not N.check("nucleo seed 1 rules 7 final lithium", num(v * 1.04, sig=4)).ok
    assert not N.check("nucleo seed 1 rules 7 final lithium", num(v * 1.01, sig=4)).ok
    assert N.check("nucleo seed 1 rules 7 final lithium", num(v, sig=3)).ok
    y = N.rollout(1, rules=7).summary["helium"]  # 0.23264: more digits pass when they round to 0.233
    assert N.check("nucleo seed 1 rules 7 final helium", num(y, sig=4)).ok
    assert N.check("nucleo seed 1 rules 7 final helium", "0 point 2 3 3 0").ok
    assert not N.check("nucleo seed 1 rules 7 final helium", "0 point 2 3 4").ok
    assert not N.check("nucleo seed 1 rules 7 final helium", "0 point 2 3 2").ok
    assert not N.check("nucleo seed 1 rules 7 final lithium", "0").ok
    assert N.check("nucleo seed 1 rules 7 step 0 helium", "0 point 0").ok
    assert not N.check("nucleo seed 1 rules 7 step 0 helium", "1 e minus 1 2").ok
    assert N.check("nucleo seed 1 rules 7 final end_time", "3 0 0 0").ok


def test_the_answer_of_one_seed_is_rejected_for_the_next_unless_it_prints_the_same():
    """For 40 seeds: the answer of seed s to a question passes for seed s + 1 exactly when both
    seeds print the same answer. A constant cannot pass where the truth differs."""
    same = differ = 0
    for s in FORTY:
        mine = {ln.prompt.split(" rules 7 ", 1)[1]: ln.answer for ln in N.lines(N.rollout(s, rules=7)) if ln.kind != "record"}
        for ln in N.lines(N.rollout(s + 1, rules=7)):
            if ln.kind == "record":
                continue
            answer = mine[ln.prompt.split(" rules 7 ", 1)[1]]
            assert N.check(ln.prompt, answer).ok == (answer == ln.answer), (ln.text, answer)
            same += answer == ln.answer
            differ += answer != ln.answer
    assert differ > 5000 and same > 5000  # stages, zeros and grid times are the same; the physics is not
    finals = {s: N.rollout(s, rules=7).value(N.rollout(s, rules=7).summary["helium"]) for s in range(1, 42)}
    assert len(set(finals.values())) >= 12  # no constant answers the helium question


def test_an_answer_that_is_not_a_finite_number_is_wrong():
    for prompt in ["nucleo seed 1 rules 7 final helium", "nucleo seed 1 rules 7 step 0 helium",
                   "nucleo seed 1 rules 7 step 5 next temperature", "nucleo seed 1 rules 7 param tau_n"]:
        for answer in ["1 e 9 9 9", "minus 1 e 9 9 9", " ".join("9" * 400), "", "e 5", "1 point", "point", "minus",
                       "1 point 2 point 3", "nan", "inf", "yes"]:
            v = N.check(prompt, answer)
            assert not v.ok and v.expected is not None, (prompt, answer, v)


def test_parameters_are_judged_exactly():
    r = N.rollout(1, rules=7)
    for k in N.RANGES7:
        assert N.check(f"nucleo seed 1 rules 7 param {k}", r.value(r.params[k])).ok
        assert parse_num(r.value(r.params[k])) == r.params[k]
        off = N.check(f"nucleo seed 1 rules 7 param {k}", num(r.params[k] * 1.01, sig=4))
        assert not off.ok and off.expected == r.value(r.params[k]) and off.reason == "wrong parameter"
    assert N.check("nucleo seed 1 rules 7 param tau_n", "8 8 7 point 0").ok
    for hidden in ("t1", "kt_freeze", "tau_bind", "rules", "t_freeze", "expansion_factor"):
        assert not N.owns(f"nucleo seed 1 rules 7 param {hidden}")


def test_owns(sim):
    for mine in ["nucleo seed 7 rules 7 step 1 5 helium", "nucleo seed 7 rules 7 step 1 5 next stage",
                 "nucleo seed 1 2 3 4 rules 7 final bottleneck_time", "nucleo seed 0 rules 7 param eta_factor",
                 "q nucleo seed 7 rules 7 step 0 time.", "nucleo seed 7 rules 7 final traces",
                 "nucleo seed 7 rules 7 param neutrinos", "nucleo seed 7 rules 7 step 3 lithium",
                 "nucleo seed 7 step 1 5 helium", "nucleo seed 7 final peak_n_p_ratio"]:  # the last two: round 6
        assert sim.owns(mine) and N.owns(mine), mine
    for other in ["turkey capital", "q spoon color", "gravity seed 7 step 2 0 clumps", "carbon protons",
                  "calc 1 2 plus 7", "water molar_mass", "", "nucleo", "nucleo seed", "nucleo seed 7 rules 7",
                  "nucleo seed 7 rules", "nucleo seed 7 rules 7 final", "nucleo seed 7 rules 7 step helium",
                  "nucleo seed 7 rules 7 step 3 next", "nucleo seed 7 rules 7 step 3 colour",
                  "nucleo seed 7 rules 6 step 3 helium", "nucleo seed 7 rules 8 step 3 helium",
                  "nucleo seed 7 rules 7 7 step 3 helium", "nucleo seed 7 rules 0 7 step 3 helium",
                  "nucleo rules 7 seed 7 step 3 helium", "nucleo seed 7 step 3 rules 7 helium",
                  "nucleo seed 7 step 3 helium rules 7", "nucleo7 seed 7 step 3 helium",
                  "nucleo7 seed 7 rules 7 step 3 helium", "nucleo seed x rules 7 step 3 helium",
                  "nucleo seed minus 7 rules 7 step 3 helium", "nucleo seed 0 7 rules 7 step 3 helium",
                  "nucleo seed 7 rules 7 step 0 3 helium", "nucleo seed 7 rules 7 step 3 helium extra",
                  "nucleo seed 7 rules 7 step 3 next next helium", "nucleo seed 7 rules 7 final stage",
                  "nucleo seed 7 rules 7 final time", "nucleo seed 7 rules 7 final peak_n_p_ratio",
                  "nucleo seed 7 rules 7 step 3 bottleneck_time", "nucleo seed 7 rules 7 param helium",
                  "nucleo seed 7 rules 7 param t_freeze", "nucleo seed 7 rules 7 param expansion_factor",
                  "nucleo seed 7 final traces", "nucleo seed 7 final end_time", "nucleo seed 7 param neutrinos",
                  "nucleo seed 7 final freeze_temperature", "nucleo rules 7 key helium",
                  "gravity seed 7 rules 7 step 2 0 clumps", "gravity seed 7 final clumps",
                  "planets seed 7 rules 7 step 3 helium", "chem seed 4 2 step 1 2 ch4",
                  "chem seed 4 2 rules 7 final water_fraction", "life seed 7 step 2 6 replicators",
                  "life seed 7 rules 7 param mu", "world seed 3 era planets", "world7 seed 3 era nucleo",
                  "stars seed 7 step 3 helium", "stars seed 7 final helium", "cells seed 7 step 3 genomes",
                  "senses seed 7 final eyes", "signals seed 7 say predator near", "society seed 7 final groups",
                  "nucleo7 predict helium_ceiling n_p_ratio 0 point 2", "nucleo predict helium_ceiling n_p_ratio 0 point 2",
                  "nucleo predict cooling_temperature temperature 1 0 time 1 next_time 2",
                  "check nucleo seed 7 rules 7 step 3 helium", "judge nucleo seed 7 rules 7 step 3 helium answer 0"]:
        assert not sim.owns(other) and not N.owns(other), other
        assert sim.check(other, "1") == Verdict(False, None, "not my question")


def test_check_handles_missing_steps(sim):
    assert not sim.check("nucleo seed 7 rules 7 step 9 9 helium", "0").ok
    assert not sim.check("nucleo seed 7 rules 7 step 3 0 helium", "0").ok
    assert not sim.check("nucleo seed 7 rules 7 step 2 9 next helium", "0").ok
    assert sim.check("nucleo seed 7 rules 7 step 2 9 next helium", "0").expected is None
    r = N.rollout(7, rules=7)
    assert sim.check("nucleo seed 7 rules 7 step 2 8 next helium", r.value(r.steps[29]["helium"])).ok


def test_the_gate_keeps_its_own_rollout():
    r = N.rollout(5, rules=7)
    truth = r.value(r.steps[25]["helium"])
    r.steps[25]["helium"] = 0.9
    r.summary["helium"] = 0.9
    r.params["tau_n"] = 1.0
    assert N.rollout(5, rules=7).steps[25]["helium"] != 0.9 and N.rollout(5, rules=7).params["tau_n"] != 1.0
    assert not N.check("nucleo seed 5 rules 7 step 2 5 helium", "0 point 9").ok
    assert N.check("nucleo seed 5 rules 7 step 2 5 helium", truth).ok
    assert N.rollout(5, rules=7) is not N.rollout(5, rules=7)
    for seed in (-1, 1.5, "7", True, None):
        with pytest.raises(ValueError):
            N.rollout(seed, rules=7)


def test_generate_is_seeded_questions_the_gate_accepts(sim):
    a = sim.generate(random.Random(3), 150)
    b = N.generate(random.Random(3), 150, rules=7)
    assert [ln.text for ln in a] == [ln.text for ln in b] and len(a) == 150
    assert [ln.text for ln in a] != [ln.text for ln in sim.generate(random.Random(4), 150)]
    assert len({ln.prompt for ln in a}) == 150
    for ln in a:
        assert ln.kind != "record" and ln.topic == "nucleo" and " rules 7 " in ln.prompt
        assert is_dense(ln.text) and ntokens(ln.text) <= 40
        v = sim.check(ln.prompt, ln.answer)
        assert v.ok and v.expected == ln.answer
        assert not sim.check(ln.prompt, _wrong(ln.answer)).ok
    assert sim.generate(random.Random(3), 0) == [] and len(sim.generate(random.Random(3), 1)) == 1
    assert all(" rules " not in ln.prompt for ln in N.generate(random.Random(3), 40))  # the default is round 6
    assert all(" rules " not in ln.prompt for ln in sim.generate(random.Random(3), 40, rules=6))


# ---------------------------------------------------------------------------------------------
# conservation
# ---------------------------------------------------------------------------------------------

def test_conserved_for_defaults_and_every_seed(sim, every_rollout):
    assert sim.conserved(run7(1)).ok and N.conserved(run7(1)).ok and N.simulation().conserved(run7(1)).ok
    for r in every_rollout:
        v = sim.conserved(r)
        assert v.ok, (r.seed, v)


def test_conserved_over_the_whole_range_the_rules_allow(sim):
    for eta, tau, families in itertools.product((0.5, 0.7, 1.0, 1.3, 2.0), (800.0, 870.0, 890.0, 960.0),
                                                (1, 2, 3, 3.044, 4, 6)):
        r = run7(0, eta_factor=eta, tau_n=tau, neutrinos=families)
        assert sim.conserved(r).ok, (eta, tau, families)
        assert 0.19 <= r.summary["helium"] <= 0.31 and 160 <= r.summary["bottleneck_time"] <= 270
        stages = [s["stage"] for s in r.steps]
        assert stages.count("equilibrium") >= 6 and stages.count("helium_forming") >= 1 and stages.count("done") >= 5
    for tau_bind in (5.0, 30.0, 100.0):
        assert sim.conserved(run7(1, tau_bind=tau_bind)).ok, tau_bind
    assert run7(1, tau_bind=100.0).summary["helium"] < run7(1, tau_bind=5.0).summary["helium"]
    for other in ({"t_d": 8e8}, {"t_d": 1e9}, {"kt_freeze": 0.737}, {"t1": 0.997e10}):
        assert sim.conserved(run7(1, **other)).ok, other


def test_conserved_rejects_tampering(sim):
    r = run7(1)

    def reason(change) -> str:
        bad = copy.deepcopy(r)
        change(bad)
        v = sim.conserved(bad)
        assert not v.ok
        return v.reason

    def at(step: int, **values):
        return lambda bad: bad.steps[step].update(values)

    def summary(**values):
        return lambda bad: bad.summary.update(values)

    assert "nucleons" in reason(at(29, helium=r.steps[29]["helium"] + 1e-6))
    assert "temperature did not fall" in reason(at(5, temperature=r.steps[4]["temperature"]))
    assert "cooling law" in reason(at(5, temperature=r.steps[5]["temperature"] * 1.001))
    assert "cooling law" in reason(at(21, temperature=1e10 / math.sqrt(175.0)))  # the round-6 law, 27 percent cold
    assert "protons decreased" in reason(at(29, hydrogen=r.steps[29]["hydrogen"] - 1e-6, free_neutrons=1e-6))
    assert "stage" in reason(at(3, stage="done"))
    assert "stage" in reason(at(3, stage="plasma"))  # a round-6 word is no rules-7 stage
    assert "before the bottleneck" in reason(at(10, deuterium=1e-12))
    assert "summary does not repeat" in reason(summary(helium=0.5))
    assert "summary lacks" in reason(lambda bad: bad.summary.pop("freeze_time"))
    assert "summary lacks" in reason(lambda bad: bad.summary.pop("traces"))
    assert "traces" in reason(summary(traces=0.0))  # hydrogen 0.753 + helium 0.247 alone is not the mix
    assert "end_time" in reason(summary(end_time=2999.0))
    assert "expansion_factor" in reason(summary(expansion_factor=1.078))
    assert "after the freeze-out" in reason(summary(freeze_n_p_ratio=0.1))
    assert "after the freeze-out" in reason(summary(freeze_time=300.0))
    assert "cooling law" in reason(summary(bottleneck_time=156.25))  # where round 6 opened it
    assert "equilibrium ratio" in reason(summary(freeze_n_p_ratio=0.25))
    assert "decayed" in reason(summary(bottleneck_n_p_ratio=0.16))
    assert "end done" in reason(lambda bad: bad.steps.__delitem__(slice(24, None)))
    assert "lacks" in reason(lambda bad: bad.steps[4].pop("lithium"))
    assert reason(at(4, helium=math.nan))
    assert "names tau_n" in reason(lambda bad: bad.params.pop("neutrinos"))
    # the neutrons in nuclei are bound_share of those at the bottleneck: a longer binding time is another run
    assert "bound_share" in reason(lambda bad: bad.params.update(tau_bind=60.0))
    # a default universe whose helium is off: the round-6 default mix is no rules-7 run
    old = N.run(1)
    assert sim.conserved(old).ok  # judged as round 6, which it is
    fake = N.NucleoRollout7("nucleo", 1, {**N.DEFAULTS7, "rules": 7}, old.steps, old.summary)
    assert not sim.conserved(fake).ok
    assert not sim.conserved(Rollout("gravity", 1, {"rules": 7}, [{"clumps": 1}])).ok
    assert not sim.conserved(Rollout("nucleo", 1, {"rules": 7}, [])).ok


def test_the_default_helium_is_gated():
    r = run7(1)
    r2 = copy.deepcopy(r)
    assert N.conserved(r2).ok
    low = run7(1, kt_freeze=0.737)  # a colder switch (n/p 0.173 instead of 0.193): helium 0.225
    assert N.conserved(low).ok and low.summary["helium"] < 0.23
    low.params["kt_freeze"] = 0.787  # ... passed off as the default universe
    v = N.conserved(low)
    assert not v.ok


# ---------------------------------------------------------------------------------------------
# plausibility: what the corrected run must look like
# ---------------------------------------------------------------------------------------------

def test_plausibility_targets_of_the_default_run():
    r = run7(1)
    y, x = r.summary["helium"], r.summary["hydrogen"]
    assert 0.245 <= y <= 0.250 and close(y, 0.2473, rel=0.001)  # real: 0.245 +- 0.003 observed, 0.247 predicted
    assert r.value(y) == "0 point 2 4 7" and r.value(x) == "0 point 7 5 3"
    assert close(x, 0.7526, rel=0.0005)
    assert abs(x + y + r.summary["traces"] - 1) < 1e-12 and close(r.summary["traces"], 6.02e-5, rel=0.002)
    assert 1 / 6 <= r.summary["freeze_n_p_ratio"] <= 1 / 5
    assert r.summary["freeze_n_p_ratio"] == math.exp(-Q_MEV / 0.787)
    assert close(r.summary["freeze_temperature"] * K_MEV, 0.787, rel=1e-12) and close(r.summary["freeze_time"], 1.209, rel=0.001)
    assert 1 / 7.5 <= r.summary["bottleneck_n_p_ratio"] <= 1 / 6.5  # textbook: 1/7
    assert 180 <= r.summary["bottleneck_time"] <= 250 and close(r.summary["bottleneck_time"], 208.7, rel=0.0005)
    assert r.summary["bottleneck_temperature"] == 9e8
    assert r.summary["deuterium"] == 2.5e-5 and r.summary["helium3"] == 1e-5 and r.summary["lithium"] == 5e-10
    assert r.summary["expansion_factor"] == 1.0 and r.summary["end_time"] == 3000.0
    assert [s["stage"] for s in r.steps] == (["equilibrium"] * 8 + ["freeze_out"] + ["decay"] * 13 + ["bottleneck_opens"]
                                             + ["helium_forming"] * 2 + ["done"] * 5)
    # the round-6 default sat above the real value on both counts
    old = N.run(1)
    assert old.summary["helium"] > 0.26 and old.summary["bottleneck_time"] < 160


def test_the_helium_answers_the_neutron_lifetime_and_the_neutrinos_as_the_real_one_does():
    y0 = run7(1).summary["helium"]
    for tau in (870.0, 890.0):
        slope = math.log(run7(1, tau_n=tau).summary["helium"] / y0) / math.log(tau / 878.4)
        assert 0.65 <= slope <= 0.80, slope  # real: 0.73; round 6: 0.21
    old0 = N.run(1).summary["helium"]
    old = math.log(N.run(1, tau_n=890.0).summary["helium"] / old0) / math.log(890.0 / 880.0)
    assert old < 0.25
    more, fewer = run7(1, neutrinos=4).summary["helium"] - y0, run7(1, neutrinos=2).summary["helium"] - y0
    assert 0.011 <= more <= 0.015 and -0.017 <= fewer <= -0.013  # real: about 0.013 a family
    per_efold = (run7(1, eta_factor=1.3).summary["helium"] - run7(1, eta_factor=0.7).summary["helium"]) / math.log(1.3 / 0.7)
    assert 0.003 <= per_efold <= 0.006  # the right direction; the real 0.0096 is not reached (docstring)


def test_parameters_pull_the_right_way():
    base = run7(1).summary
    four, two = run7(1, neutrinos=4).summary, run7(1, neutrinos=2).summary
    assert two["expansion_factor"] < 1 < four["expansion_factor"]
    assert close(four["expansion_factor"], 1.078, rel=0.001) and close(two["expansion_factor"], 0.915, rel=0.001)
    assert two["helium"] < base["helium"] < four["helium"]
    assert four["bottleneck_time"] < base["bottleneck_time"] < two["bottleneck_time"]
    assert four["freeze_time"] < base["freeze_time"] < two["freeze_time"]
    assert four["freeze_temperature"] > base["freeze_temperature"] > two["freeze_temperature"]
    assert four["deuterium"] > base["deuterium"] > two["deuterium"]  # a faster expansion leaves more deuterium
    assert four["lithium"] < base["lithium"] < two["lithium"]
    dense, thin = run7(1, eta_factor=1.3).summary, run7(1, eta_factor=0.7).summary
    assert dense["deuterium"] < base["deuterium"] < thin["deuterium"]
    assert dense["helium3"] < base["helium3"] < thin["helium3"]
    assert dense["lithium"] > base["lithium"] > thin["lithium"]
    assert dense["bottleneck_time"] < base["bottleneck_time"] < thin["bottleneck_time"]
    assert dense["bottleneck_temperature"] > 9e8 > thin["bottleneck_temperature"]
    assert thin["helium"] < base["helium"] < dense["helium"] < base["helium"] + 0.01
    assert dense["freeze_time"] == base["freeze_time"] == thin["freeze_time"]  # the baryons do not set the freeze-out
    long, short = run7(1, tau_n=890).summary, run7(1, tau_n=870).summary
    assert long["helium"] > base["helium"] > short["helium"]
    assert long["freeze_temperature"] > base["freeze_temperature"] > short["freeze_temperature"]  # the coupling
    assert long["freeze_n_p_ratio"] > base["freeze_n_p_ratio"] > short["freeze_n_p_ratio"]
    assert long["bottleneck_time"] == base["bottleneck_time"] == short["bottleneck_time"]


def test_plausibility_targets_of_forty_seeds():
    """The targets of the audit, measured over 40 seeds drawn with random_params(rules=7)."""
    helium, three = [], []
    for s in FORTY:
        r = N.rollout(s, rules=7)
        stages = [st["stage"] for st in r.steps]
        assert [N.STAGES7.index(x) for x in stages] == sorted(N.STAGES7.index(x) for x in stages), s
        assert stages.count("equilibrium") >= 7 and stages.count("freeze_out") == 1 and stages.count("decay") >= 12
        assert stages.count("bottleneck_opens") == 1 and stages.count("helium_forming") >= 1 and stages.count("done") >= 5
        y = r.summary["helium"]
        helium.append(y)
        assert 0.225 <= y <= 0.270, (s, y)  # round 6: 0.155 to 0.469 over seeds 1 to 30
        if r.params["neutrinos"] == 3:
            three.append(y)
            assert 0.243 <= y <= 0.252, (s, y)
        assert abs(y + r.summary["hydrogen"] + r.summary["traces"] - 1) < 1e-12
        assert 185 <= r.summary["bottleneck_time"] <= 240 and 1.0 <= r.summary["freeze_time"] <= 1.5
        assert 0.18 <= r.summary["freeze_n_p_ratio"] <= 0.205 and 0.13 <= r.summary["bottleneck_n_p_ratio"] <= 0.16
        if r.params["neutrinos"] == 3:  # the textbook numbers hold for the three families we have
            assert 1 / 6 <= r.summary["freeze_n_p_ratio"] <= 1 / 5
            assert 1 / 7.5 <= r.summary["bottleneck_n_p_ratio"] <= 1 / 6.5
        assert 0.60 <= r.steps[0]["n_p_ratio"] <= 0.65 and r.steps[-1]["free_neutrons"] == 0
        assert r.summary["freeze_n_p_ratio"] > r.summary["bottleneck_n_p_ratio"]
    assert len(three) >= 15 and min(helium) < 0.24 and max(helium) > 0.255  # the neutrino families still show
    assert abs(sum(helium) / len(helium) - 0.247) < 0.005


def test_plausibility_targets_of_every_seed(every_rollout):
    low, high, three_low, three_high, total, three = 1.0, 0.0, 1.0, 0.0, 0.0, 0
    for r in every_rollout:
        stages = [s["stage"] for s in r.steps]
        assert stages.count("equilibrium") >= 7 and stages.count("freeze_out") == 1 and stages.count("decay") >= 1
        assert stages.count("bottleneck_opens") == 1 and stages.count("helium_forming") >= 1 and stages.count("done") >= 5
        y = r.summary["helium"]
        low, high, total = min(low, y), max(high, y), total + y
        if r.params["neutrinos"] == 3:
            three_low, three_high, three = min(three_low, y), max(three_high, y), three + 1
        assert 185 <= r.summary["bottleneck_time"] <= 240 and 1.0 <= r.summary["freeze_time"] <= 1.5
        assert 0.7 <= r.params["eta_factor"] <= 1.3  # inside the window where the trace fits hold (0.66 to 1.31)
        assert 0.9995 <= y + r.summary["hydrogen"] < 1
    assert 0.225 < low < 0.232 and 0.260 < high < 0.270  # measured 0.2288 and 0.2638
    assert 0.243 < three_low < 0.246 and 0.249 < three_high < 0.252  # measured 0.2440 and 0.2508
    assert abs(total / len(every_rollout) - 0.247) < 0.001 and 0.57 < three / len(every_rollout) < 0.63


def test_deuterium_is_the_stepping_stone(every_rollout):
    """Deuterium peaks while the helium forms and ends at its fitted ratio; helium never falls."""
    for r in every_rollout[:40] + [run7(1)]:
        d = [s["deuterium"] for s in r.steps]
        top = d.index(max(d))
        assert r.steps[top]["stage"] in ("bottleneck_opens", "helium_forming")
        assert 1e-3 < d[top] < 3e-3 and d[top] > 20 * r.summary["deuterium"]
        assert all(a <= b for a, b in zip(d[:top], d[1:top + 1])) and all(a >= b for a, b in zip(d[top:], d[top + 1:]))
        assert d[-1] == r.summary["deuterium"] == N.deuterium_ratio(r.params["eta_factor"], r.params["neutrinos"], r.params["tau_n"])
        he = [s["helium"] for s in r.steps]
        assert all(a <= b for a, b in zip(he, he[1:]))
        before = [s for s in r.steps if s["time"] < r.summary["bottleneck_time"]]
        assert all(s["helium"] == 0 and s["deuterium"] == 0 and s["helium3"] == 0 and s["lithium"] == 0 for s in before)
        assert all(s["free_neutrons"] + s["hydrogen"] == 1 for s in before)
    for r in every_rollout:  # on this grid the peak shows on one step, bottleneck_opens
        d = [s["deuterium"] for s in r.steps]
        at = next(i for i, s in enumerate(r.steps) if s["stage"] == "bottleneck_opens")
        assert d[at] == max(d) and 1.2e-3 < d[at] < 2.5e-3  # measured 1.21e-3 to 2.45e-3
        assert 30 < d[at] / d[-1] < 175 and 1.9 < d[at + 1] / d[-1] < 9.0  # 32 to 172 times the final trace, then 2 to 9
        assert d[at + 2] / d[-1] < 1.2
    assert max(N.deuterium_in_transit(p / 100) for p in range(101)) == N.deuterium_in_transit(0.5) == N.D_PEAK == 0.002
    assert N.deuterium_in_transit(0.0) == 0 == N.deuterium_in_transit(1.0)


def test_at_bottleneck_opens_part_of_the_helium_is_already_bound():
    r = run7(1)
    step = next(s for s in r.steps if s["stage"] == "bottleneck_opens")
    assert step["time"] == 249.0
    share = N.bound_progress(249.0 - r.summary["bottleneck_time"], 878.4, 30.0)
    assert close(share, 0.751, rel=0.002)
    assert 0.70 * r.summary["helium"] < step["helium"] < 0.76 * r.summary["helium"]


def test_physics_invariants():
    for s in SEEDS:
        r = N.rollout(s, rules=7)
        f, tau = r.summary["expansion_factor"], r.params["tau_n"]
        assert f == math.sqrt(1 + 7 * (r.params["neutrinos"] - 3) / 43)
        for st in r.steps:
            assert st["temperature"] == N.temperature(st["time"], f)
        for st in (x for x in r.steps if x["stage"] == "equilibrium"):  # weak equilibrium: n/p = exp(-Q / kT)
            assert close(st["n_p_ratio"], math.exp(-Q_MEV / (K_MEV * st["temperature"])), rel=1e-12)
            assert st["time"] < r.summary["freeze_time"] and st["temperature"] > r.summary["freeze_temperature"]
        decay = [st for st in r.steps if st["stage"] in ("freeze_out", "decay")]
        for a, b in zip(decay, decay[1:]):  # free decay with the neutron lifetime
            assert close(b["free_neutrons"] / a["free_neutrons"], math.exp(-(b["time"] - a["time"]) / tau), rel=1e-9)
        # the freeze-out follows from the neutron lifetime and the expansion rate
        kt = 0.787 * (f * tau / 878.4) ** (1 / 3)
        assert close(r.summary["freeze_temperature"] * K_MEV, kt, rel=1e-12)
        assert close(r.summary["freeze_n_p_ratio"], math.exp(-Q_MEV / kt), rel=1e-12)
        assert close(N.temperature(r.summary["freeze_time"], f), r.summary["freeze_temperature"], rel=1e-12)
        assert close(N.temperature(r.summary["bottleneck_time"], f), r.summary["bottleneck_temperature"], rel=1e-12)
        after = [st for st in r.steps if st["time"] >= r.summary["bottleneck_time"]]
        assert after[0]["stage"] == "bottleneck_opens" and all(st["helium"] > 0 and st["deuterium"] > 0 for st in after)
        assert all(st["temperature"] <= r.summary["bottleneck_temperature"] for st in after)
        y, ratio = r.summary["helium"], r.summary["bottleneck_n_p_ratio"]
        assert y < 2 * ratio / (1 + ratio) and close(y, 2 * ratio / (1 + ratio) * tau / (tau + 30), rel=1e-3)
        eta, families = r.params["eta_factor"], r.params["neutrinos"]
        assert close(r.summary["deuterium"], 2.5e-5 * eta ** -1.6 * (families / 3) ** 0.395 * (tau / 878.4) ** 0.41, rel=1e-9)
        assert close(r.summary["helium3"], 1e-5 * eta ** -0.6 * (families / 3) ** 0.14 * (tau / 878.4) ** 0.15, rel=1e-9)
        assert close(r.summary["lithium"], 5e-10 * eta ** 2 * (families / 3) ** -0.284 * (tau / 878.4) ** 0.43, rel=1e-9)
        h = r.summary["hydrogen"]
        assert r.summary["traces"] == h * (2 * r.summary["deuterium"] + 3 * r.summary["helium3"] + 7 * r.summary["lithium"])


# ---------------------------------------------------------------------------------------------
# the cooling law
# ---------------------------------------------------------------------------------------------

def _exact_cooling() -> tuple[np.ndarray, np.ndarray]:
    """Time in s and photon temperature in K of the standard early universe, integrated: photons
    and electron-positron pairs keep their entropy while the pairs annihilate, three neutrino
    families have decoupled and cool as 1 / a, the Friedmann equation gives the clock. Natural
    units in MeV; constants from memory (CODATA)."""
    me, mpl, hbar = 0.51099895, 1.220890e22, 6.582119569e-22  # MeV, MeV, MeV s
    ln_kt = np.linspace(math.log(100.0), math.log(0.003), 800)
    kt = np.exp(ln_kt)
    u = np.linspace(0.0, 80.0, 2001)  # momentum over temperature
    e = np.sqrt(u[None, :] ** 2 + (me / kt)[:, None] ** 2)
    occ = 1.0 / (np.exp(e) + 1.0)
    pre = 4 / (2 * np.pi ** 2) * kt ** 4  # electrons and positrons, two spins each
    rho_e = pre * np.trapezoid(u ** 2 * e * occ, u, axis=1)
    p_e = pre * np.trapezoid(u ** 4 / (3 * e) * occ, u, axis=1)
    rho_g = np.pi ** 2 / 15 * kt ** 4
    s = (rho_g * 4 / 3 + rho_e + p_e) / kt  # entropy density of photons and pairs
    kt_nu = kt * (s / (2 * np.pi ** 2 / 45 * 5.5 * kt ** 3)) ** (1 / 3)
    rho = rho_g + rho_e + 3 * 7 / 8 * np.pi ** 2 / 15 * kt_nu ** 4
    hubble = np.sqrt(8 * np.pi / 3 * rho) / mpl
    integrand = np.gradient(np.log(s), ln_kt) / 3 / hubble  # dt = -(1/3) dln(s) / H
    t = 1 / (2 * hubble[0]) + np.concatenate([[0.0], np.cumsum(-(integrand[1:] + integrand[:-1]) / 2 * np.diff(ln_kt))])
    return t * hbar, kt / K_MEV


def test_the_cooling_law_follows_the_integrated_relation():
    """Correction 2: the fit with the electron-positron heating stays within 1.5 percent of the
    exact relation from 0.05 s to 5000 s; the round-6 power law is 25 percent too cold."""
    seconds, kelvin = _exact_cooling()
    grid = np.exp(np.linspace(math.log(0.05), math.log(5000.0), 400))
    exact = np.exp(np.interp(np.log(grid), np.log(seconds), np.log(kelvin)))
    fit = np.array([N.temperature(float(t), 1.0) for t in grid])
    assert 0.985 < (fit / exact).min() and (fit / exact).max() < 1.015
    assert 0.988 < (fit / exact).min() < 0.989 and 1.013 < (fit / exact).max() < 1.014  # measured: within 1.4 percent
    old = np.array([1e10 / math.sqrt(t) for t in grid])
    assert (old / exact).min() < 0.76 and (old / exact).max() < 1.01
    forming = (old / exact)[grid >= 150.0]  # where the nuclei form: 22 to 25 percent too cold
    assert 0.745 < forming.min() < 0.755 and 0.770 < forming.max() < 0.780
    assert close(N.temperature(150.0, 1.0) * math.sqrt(150.0) / 1e10, 1.284, rel=0.001)  # 28 to 34 percent hotter
    assert close(N.temperature(3000.0, 1.0) * math.sqrt(3000.0) / 1e10, 1.336, rel=0.001)
    reached = float(np.exp(np.interp(-math.log(8e8), -np.log(kelvin), np.log(seconds))))
    assert 272 < reached < 276  # 8e8 K comes at 274 s by the integrated relation, not at round 6's 156 s
    assert close(N.time_at(8e8, 1.0), 267.8, rel=0.001) and close(N.time_at(8e8, 1.0), reached, rel=0.03)  # the fit: 268 s
    assert close(run7(1, t_d=8e8).summary["helium"], 0.2312, rel=0.001)  # ... which would leave too little helium
    opened = float(np.exp(np.interp(-math.log(9e8), -np.log(kelvin), np.log(seconds))))
    assert 211 < opened < 214 and close(N.time_at(9e8, 1.0), 208.7, rel=0.001)  # 9e8 K: 213 s integrated, 208.7 s here
    early, late = kelvin[10] * math.sqrt(seconds[10]), kelvin[-1] * math.sqrt(seconds[-1])
    assert close(early, 0.997e10, rel=0.002) and close(late, 1.333e10, rel=0.002)
    assert close(late / early, N.HEATING, rel=0.002)


def test_the_cooling_law_is_the_two_power_laws_joined():
    assert N.HEATING == math.sqrt(math.sqrt(10.75 / 3.3626)) and close(N.HEATING, 1.33716, rel=1e-5)
    assert close(2 + 7 / 8 * 6 * (4 / 11) ** (4 / 3), 3.3626, rel=1e-4)  # the degrees of freedom afterwards
    assert close(N.temperature(0.01, 1.0) * math.sqrt(0.01), 1e10, rel=1e-3)
    assert close(N.temperature(1e5, 1.0) * math.sqrt(1e5), 1.33716e10, rel=1e-3)
    assert N.photon_heating(1e12) < 1.000001 and close(N.photon_heating(1e6), N.HEATING, rel=1e-6)
    assert N.temperature(1.2, 1.0) == N.radiation_temperature(1.2, 1.0) * N.photon_heating(N.radiation_temperature(1.2, 1.0))
    assert N.radiation_temperature(4.0, 1.0) == 5e9 and N.radiation_temperature(1.0, 4.0) == 5e9
    times = [0.01 * 1.05 ** i for i in range(400)]
    for f in (0.9, 1.0, 1.1):
        temps = [N.temperature(t, f) for t in times]
        slopes = [math.log(b / a) / math.log(1.05) for a, b in zip(temps, temps[1:])]
        assert all(-0.5 - 1e-9 <= x <= -0.40 for x in slopes)  # strictly falling, never slower than t^-0.4
        assert N.temperature(1.0, f) < N.temperature(1.0, 0.8 * f)  # a faster expansion is colder at a given time


def test_time_at_inverts_the_cooling_law():
    for f in (0.915, 1.0, 1.078):
        for kelvin in (3e10, 9.13e9, 2e9, 9e8, 8e8, 2.5e8):
            t = N.time_at(kelvin, f)
            assert close(N.temperature(t, f), kelvin, rel=1e-12)
            assert N.temperature(t, f) <= kelvin < N.temperature(t * (1 - 1e-12), f)  # the first moment at or below
        assert N.time_at(9e8, f) == N.time_at(9e8, f)
    assert N.time_at(9e8, 1.078) < N.time_at(9e8, 1.0) < N.time_at(9e8, 0.915)
    for kelvin in (1e13, 1e5, 0.0, -1.0, math.nan):
        with pytest.raises(ValueError):
            N.time_at(kelvin, 1.0)


# ---------------------------------------------------------------------------------------------
# lessons
# ---------------------------------------------------------------------------------------------

def test_lessons_are_the_functions_of_the_simulation(monkeypatch):
    """Every lesson rule is a module function under its own name, and a rules-7 run with its
    gate calls each of them."""
    assert len(N.RULES) == 18 >= 8 and N.LESSONS.sim == "nucleo7" == N.SIM7 and N.LESSONS.topic == "predict_nucleo7"
    calls = {name: 0 for name in N.RULES}
    for name, rule in N.RULES.items():
        function = getattr(N, name)
        assert rule.compute is function, name

        def counted(*args, _name=name, _function=function, **kwargs):
            calls[_name] += 1
            return _function(*args, **kwargs)

        monkeypatch.setattr(N, name, counted)
    r = N.run(1, rules=7)
    by_run = dict(calls)
    assert N.conserved(r).ok
    assert all(n > 0 for n in by_run.values()), by_run  # the run alone calls every one of them
    assert all(calls[name] > by_run[name] for name in ("helium_ceiling", "temperature", "bound_share")), calls  # the gate too
    monkeypatch.undo()
    assert r.steps == N.run(1, rules=7).steps  # counting changed nothing


def test_every_generated_lesson_is_true_of_the_simulation():
    lessons = N.LESSONS.generate(random.Random(1), 400)
    assert [ln.text for ln in lessons] == [ln.text for ln in N.LESSONS.generate(random.Random(1), 400)]
    assert [ln.text for ln in lessons] != [ln.text for ln in N.LESSONS.generate(random.Random(2), 400)]
    assert len(lessons) == 400 and {ln.prompt.split()[2] for ln in lessons} == set(N.RULES)
    for ln in lessons:
        assert is_dense(ln.text) and ntokens(ln.text) <= 50 <= MAX_TOKENS, ln.text
        assert ln.topic == "predict_nucleo7" and ln.kind == "calc" and ln.prompt.startswith("nucleo7 predict ")
        assert "." not in ln.prompt and "." not in ln.answer
        assert N.LESSONS.owns(ln.prompt) and not N.owns(ln.prompt)
        v = N.LESSONS.check(ln.prompt, ln.answer)
        assert v.ok and v.expected == ln.answer, ln.text
        name, values = N.LESSONS.parse(ln.prompt)
        assert ln.answer == num(getattr(N, name)(*values)), ln.text  # the simulation's own function
        assert not N.LESSONS.check(ln.prompt, _wrong(ln.answer)).ok, ln.text
        assert ln.meta["split_key"] == N.LESSONS.split_key(ln.prompt)
    assert N.lesson_gate().generate(random.Random(5), 3)[0].prompt.startswith("nucleo7 predict ")


def test_known_lessons():
    for prompt, answer in [
        ("nucleo7 predict expansion_factor neutrinos 3", "1"),
        ("nucleo7 predict expansion_factor neutrinos 4", "1 point 0 7 8"),
        ("nucleo7 predict expansion_factor neutrinos 2", "0 point 9 1 5"),
        ("nucleo7 predict expansion_factor neutrinos 3 point 4", "1 point 0 3 2"),
        ("nucleo7 predict photon_heating temperature 1 point 6 e 9", "1 point 1 6 9"),
        ("nucleo7 predict bound_progress seconds 4 0 point 3 tau_n 8 7 8 point 4 tau_bind 3 0", "0 point 7 5 0 7"),
        ("nucleo7 predict trace_mass hydrogen 0 point 7 5 3 deuterium 2 point 5 e minus 5 helium3 1 e minus 5 "
         "lithium 5 e minus 1 0", "6 point 0 2 4 e minus 5"),
        ("nucleo7 predict helium3_ratio eta_factor 1 neutrinos 4 tau_n 8 7 8 point 4", "1 point 0 4 1 e minus 5"),
        ("nucleo7 predict radiation_temperature time 4 expansion_factor 1", "5 e 9"),
        ("nucleo7 predict temperature time 2 0 8 point 7 expansion_factor 1", "9 e 8"),
        ("nucleo7 predict thermal_energy temperature 9 e 8", "0 point 0 7 7 5 6"),
        ("nucleo7 predict equilibrium_ratio kt_mev 0 point 7 8 7", "0 point 1 9 3 3"),
        ("nucleo7 predict freeze_kt expansion_factor 1 tau_n 8 7 8 point 4", "0 point 7 8 7"),
        ("nucleo7 predict freeze_kt expansion_factor 1 point 0 7 8 tau_n 8 8 1", "0 point 8 0 7 7"),
        ("nucleo7 predict bottleneck_temperature eta_factor 1", "9 e 8"),
        ("nucleo7 predict helium_ceiling n_p_ratio 0 point 1 4 7", "0 point 2 5 6 3"),
        ("nucleo7 predict neutrons_per_nucleon n_p_ratio 0 point 2 5", "0 point 2"),
        ("nucleo7 predict bound_share tau_n 8 7 8 point 4 tau_bind 3 0", "0 point 9 6 7"),
        ("nucleo7 predict neutron_decay free_neutrons 0 point 1 6 2 seconds 2 0 7 point 5 tau_n 8 7 8 point 4", "0 point 1 2 7 9"),
        ("nucleo7 predict deuterium_ratio eta_factor 1 neutrinos 3 tau_n 8 7 8 point 4", "2 point 5 e minus 5"),
        ("nucleo7 predict lithium_ratio eta_factor 1 point 2 neutrinos 3 tau_n 8 7 8 point 4", "7 point 2 e minus 1 0"),
        ("nucleo7 predict deuterium_in_transit progress 0 point 5", "0 point 0 0 2"),
    ]:
        v = N.LESSONS.check(prompt, answer)
        assert v.ok, (prompt, v)
    # ranges: a lesson does not answer for inputs the rules do not cover
    for outside in ["nucleo7 predict deuterium_ratio eta_factor 0 point 3 neutrinos 3 tau_n 8 8 0",
                    "nucleo7 predict lithium_ratio eta_factor 3 neutrinos 3 tau_n 8 8 0",
                    "nucleo7 predict temperature time 1 e 6 expansion_factor 1",
                    "nucleo7 predict expansion_factor neutrinos 9",
                    "nucleo predict freeze_kt expansion_factor 1 tau_n 8 8 0",
                    "nucleo7 predict cooling_temperature temperature 1 0 time 1 next_time 2",
                    "nucleo seed 1 rules 7 final helium", "stars predict lifetime mass 1"]:
        assert not N.LESSONS.owns(outside), outside
        assert N.LESSONS.check(outside, "1") == Verdict(False, None, "not my question")


def test_lesson_records_are_dense_and_say_the_rule():
    rec = N.LESSONS.records()
    assert len(rec) == len(N.RULES) and set(N.LESSONS.KEYS) == set(N.RULES)
    for ln in rec:
        assert ln.kind == "record" and ln.topic == "predict_nucleo7"
        assert is_dense(ln.text) and ntokens(ln.text) <= 70 <= MAX_TOKENS, ln.text
    texts = " ".join(ln.text for ln in rec)
    assert "a fit to the integrated relation" in texts and "a toy number" in texts and "is calibrated" in texts


def test_the_lessons_reproduce_a_story_from_its_printed_numbers():
    """A lesson is true of the simulation: chained from the numbers a story prints (three
    digits), the lesson functions give the story's own numbers back to within a percent."""
    for s in FORTY:
        r = N.rollout(s, rules=7)
        printed = {k: sig(v) for k, v in r.summary.items()}
        eta, tau, families = (r.params[k] for k in ("eta_factor", "tau_n", "neutrinos"))
        f = N.expansion_factor(families)
        assert sig(f) == printed["expansion_factor"]
        kt = N.freeze_kt(printed["expansion_factor"], tau)
        assert close(kt, N.thermal_energy(printed["freeze_temperature"]), rel=0.01)
        assert close(N.equilibrium_ratio(kt), printed["freeze_n_p_ratio"], rel=0.01)
        assert close(N.temperature(printed["freeze_time"], printed["expansion_factor"]), printed["freeze_temperature"], rel=0.01)
        assert close(N.bottleneck_temperature(eta), printed["bottleneck_temperature"], rel=0.01)
        assert close(N.temperature(printed["bottleneck_time"], printed["expansion_factor"]), printed["bottleneck_temperature"], rel=0.01)
        left = N.neutron_decay(N.neutrons_per_nucleon(printed["freeze_n_p_ratio"]),
                               printed["bottleneck_time"] - printed["freeze_time"], tau)
        assert close(left, N.neutrons_per_nucleon(printed["bottleneck_n_p_ratio"]), rel=0.01)
        assert close(N.helium_ceiling(printed["bottleneck_n_p_ratio"]) * N.bound_share(tau, 30.0), printed["helium"], rel=0.01)
        assert close(N.deuterium_ratio(eta, families, tau), printed["deuterium"], rel=0.006)
        assert close(N.helium3_ratio(eta, families, tau), printed["helium3"], rel=0.006)
        assert close(N.lithium_ratio(eta, families, tau), printed["lithium"], rel=0.006)
        assert close(N.trace_mass(printed["hydrogen"], printed["deuterium"], printed["helium3"], printed["lithium"]),
                     printed["traces"], rel=0.01)
        opened = next(st for st in r.steps if st["stage"] == "bottleneck_opens")
        progress = N.bound_progress(opened["time"] - printed["bottleneck_time"], tau, 30.0)
        assert close(progress * r.summary["helium"] - 2 * N.deuterium_in_transit(progress), opened["helium"], rel=0.05)


# ---------------------------------------------------------------------------------------------
# the hand-off, reproducibility, housekeeping
# ---------------------------------------------------------------------------------------------

def test_the_handoff_is_the_default_universe_in_printed_numbers():
    h = N.handoff()
    assert h == N.handoff(N.run(123, rules=7)) == N.handoff(N.run(0, rules=7))
    assert h["helium"] == 0.247 and h["traces"] == 6.02e-5 and h["end_time"] == 3000.0
    assert h["hydrogen"] == 1 - 0.247 - 6.02e-5 and sig(h["hydrogen"]) == 0.753
    assert abs(h["hydrogen"] + h["helium"] + h["traces"] - 1) < 1e-15 and h["traces"] > 0
    assert h["end_time_years"] == 9.51e-5 and close(h["end_time_years"], 3000 / (365.25 * 86400), rel=0.001)
    other = N.handoff(N.rollout(1, rules=7))
    assert other["helium"] == 0.233 and other["traces"] == 7.65e-5
    with pytest.raises(ValueError):
        N.handoff(N.run(1))
    with pytest.raises(ValueError):
        N.handoff(N.rollout(1))


def test_no_decision_hangs_on_the_last_bit(every_rollout):
    """What may differ between machines is the last bit of exp, log and pow. Over every seed that
    can be drawn no grid time is near an event, no neutron threshold near a flip and no printed
    third digit closer than 1e-10 to a rounding edge."""
    grid = np.array(N.TIMES)
    values: list[float] = []
    for r in every_rollout:
        for edge in (r.summary["freeze_time"], r.summary["bottleneck_time"]):
            assert np.abs(grid / edge - 1).min() > 1e-4, r.seed
        n_d = r.summary["bottleneck_n_p_ratio"] / (1 + r.summary["bottleneck_n_p_ratio"])
        for st in r.steps:
            if st["time"] >= r.summary["bottleneck_time"]:
                n = n_d * (1 - N.bound_progress(st["time"] - r.summary["bottleneck_time"], r.params["tau_n"], 30.0))
                assert abs(n / 1e-6 - 1) > 1e-3 and abs(n / 1e-10 - 1) > 1e-3, r.seed
                assert (st["free_neutrons"] == 0) == (n < 1e-10)
            values += [v for k, v in st.items() if k not in ("stage", "time")]
        values += list(r.summary.values())
        rng = random.Random(r.seed)  # the raw draws behind the three-digit parameters
        values += [math.exp(rng.uniform(math.log(0.7), math.log(1.3))), rng.uniform(870.0, 890.0)]
    x = np.array(values, dtype=np.float64)
    x = x[x > 0]
    mantissa = x / 10.0 ** (np.floor(np.log10(x)) - 2)  # 100 .. 1000: the third digit is the last printed
    edge = np.abs(mantissa - np.floor(mantissa) - 0.5) / mantissa
    assert len(x) > 1_000_000 and edge.min() > 1e-10


def test_the_rules7_run_needs_no_numpy_and_is_fast():
    assert not hasattr(N, "np") and not hasattr(N, "numpy")
    r = run7(1)
    assert all(type(v) in (float, str) for st in r.steps for v in st.values())
    assert all(type(v) is float for v in r.summary.values())
    t0 = time.perf_counter()
    for s in range(50):
        N.run(s, rules=7, **N.random_params(random.Random(s), rules=7))
    assert (time.perf_counter() - t0) / 50 < 0.01  # the contract allows 2 s


def test_constants_say_where_they_come_from():
    for name, row in N.CONSTANTS7.items():
        assert any(mark in row["notes"] for mark in ("from memory", "toy number", "a fit of this module")), name
    assert N.TAU_N == 878.4 and N.KT_FREEZE == 0.787 and N.T_BOTTLENECK == 9e8 and N.T1_K == 1e10
    assert N.NEUTRINO_DRAW == (2, 3, 3, 3, 4) and N.STAGES7[0] == "equilibrium" and "plasma" not in N.STAGES7
    assert set(N.LIMITS7) == set(N.RANGES7)
    for key, (lo, hi, _) in N.RANGES7.items():  # the draw stays inside what run() accepts
        assert N.LIMITS7[key][0] <= lo < hi <= N.LIMITS7[key][1]
    assert "Rules 7" in N.__doc__ and "from memory" in N.__doc__ and "calibrated" in N.__doc__
    assert N.KEYS7["temperature"] == "photon temperature in K, temperature(time, expansion_factor)"
