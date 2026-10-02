"""Level 5 under rules 7: the corrected vessel of ``haishool.cosmos.life``.

Round 6 (``tests/test_cosmos_life.py``, ``tests/test_round6_frozen.py``) is untouched; these tests
hold what ``run(seed, rules=7)`` adds: the spent pool, hydrolysis per bond, copying that takes
time and gives the complement, lineages, the length limit, the stricter gate and the lessons.
"""
import copy
import hashlib
import json
import random
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from haishool.cosmos import Rollout, Simulation, life
from haishool.evo import LessonGate
from haishool.truth import is_dense, num, parse_num

SIM = life.simulation()
LESSONS = life.LESSONS
MAX_TOKENS = 128
SEEDS = (1, 2, 3, 7, 11)
FORTY = range(1, 41)
OTHER_FORTY = range(100, 140)
#: sha256 (first 16 hex digits) of whole rollouts: rules 7 as written here, round 6 as it always was
DIGESTS7 = {1: "3b2879950ee5b432", 7: "4823c097bfade73d", 1234: "ede334a03551dae7"}
DIGESTS6 = {1: "adb5acdd45499bb1", 7: "936210babc0720fd", 1234: "307c5389a705ed37"}
OTHER_PROMPTS = (
    "turkey capital", "q spoon color", "carbon protons", "water molar_mass", "calc 1 2 plus 7",
    "check 1 2 plus 7 equals 2 0", "solve 3 x plus 4 equals 1 9", "chem valence c",
    "gravity seed 7 step 2 0 clumps", "gravity seed 7 final clumps", "chem seed 4 2 step 1 2 ch4",
    "chem seed 3 final molecules", "nucleo seed 1 step 1 5 helium", "nucleo seed 1 final helium",
    "planets seed 3 step 1 4 gas", "planets seed 3 final habitable", "world seed 3 era planets",
    "world seed 3 final replicators", "stars seed 3 step 4 gas", "stars seed 3 final metallicity",
    "cells seed 3 step 4 cells", "bodies seed 3 final outcome", "senses seed 3 step 2 sensors",
    "signals seed 7 step 1 2 success", "society seed 7 final outcome", "world7 seed 3 final outcome",
    "stars predict lifetime mass 2", "society predict hamilton relatedness 0 point 5 benefit 4 cost 1",
)
FUNCTIONS = {"intact_probability": life.intact_probability, "copy_probability": life.copy_probability,
             "motif_fidelity": life.motif_fidelity, "growth_factor": life.growth_factor,
             "max_length": life.max_length, "polymer_letters": life.polymer_letters,
             "inflow_per_kind": life.inflow_per_kind, "next_total": life.next_total,
             "stage": life.stage, "outcome": life.outcome}


def n_tokens(text: str) -> int:
    return len(text.replace(".", " . ").split())


def digest(r: Rollout) -> str:
    return hashlib.sha256(json.dumps([r.params, r.steps, r.summary], sort_keys=True).encode()).hexdigest()[:16]


def strings(seqs: np.ndarray) -> list[str]:
    return ["".join(life.LETTERS[x] for x in row if x != life.PAD) for row in seqs]


def lengths(seqs: np.ndarray) -> np.ndarray:
    return (seqs != life.PAD).sum(1)


@pytest.fixture(scope="module")
def rollouts():
    return {seed: SIM.rollout(seed, rules=7) for seed in SEEDS}


@pytest.fixture(scope="module")
def forty():
    """Seeds 1 to 40 with their seeded parameters, given explicitly as ``random_params`` draws them."""
    return {seed: life.run(seed, rules=7, **life.random_params(random.Random(seed), rules=7)) for seed in FORTY}


@pytest.fixture(scope="module")
def generated():
    return SIM.generate(random.Random(1), 600, rules=7)


@pytest.fixture(scope="module")
def all_lines(rollouts, generated):
    return [ln for r in rollouts.values() for ln in life.lines(r)] + generated


@pytest.fixture(scope="module")
def lessons():
    return LESSONS.generate(random.Random(1), 400)


# ---------------------------------------------------------------------------------------------
# round 6 is the default and is not touched
# ---------------------------------------------------------------------------------------------

def test_without_rules_everything_is_round_6():
    assert {seed: digest(life.run(seed)) for seed in DIGESTS6} == DIGESTS6
    assert {seed: digest(life.run(seed, rules=6)) for seed in DIGESTS6} == DIGESTS6
    assert {seed: digest(SIM.run(seed, **life.random_params(random.Random(seed)))) for seed in DIGESTS6} == DIGESTS6
    r = life.run(7)
    assert "rules" not in r.params and list(r.params) == list(life.PARAM_KEYS) and len(r.steps) == life.STEPS + 1 == 41
    assert not life.is_rules7(r) and list(r.steps[0]) == list(life.KEYS)
    texts = [ln.text for ln in life.lines(r)]
    assert texts == [ln.text for ln in life.lines(r, rules=6)]
    assert texts[0] == ("life seed 7 params. monomers 1 0 3 0 5. inflow 2 4 0. poly_rate 0 point 0 0 2. "
                        "decay 0 point 1. mu 0 point 0 2. gap 0.")
    assert not any(" rules " in t for t in texts)
    assert life.random_params(random.Random(7)) == life.random_params(random.Random(7), rules=6) == {
        "monomers": 10305, "inflow": 240, "poly_rate": 0.002, "decay": 0.1, "mu": 0.02, "gap": 0}
    assert SIM.rollout(7) is SIM.rollout(7, rules=6) and SIM.rollout(7).steps == r.steps
    assert [ln.text for ln in SIM.generate(random.Random(3), 200)] == [ln.text for ln in SIM.generate(random.Random(3), 200, rules=6)]
    assert SIM.conserved(r).ok and life.conserved(r).ok
    with pytest.raises(ValueError):
        life.lines(r, rules=7)  # a round-6 rollout has no rules-7 lines
    with pytest.raises(ValueError):
        life.run(1, hydrolysis=0.01)  # round 6 has no hydrolysis
    with pytest.raises(ValueError):
        life.run(1, rules=7, decay=0.1)  # and rules 7 no decay
    for bad in (5, 8, True, "7", None, 7.5):
        for call in (lambda: life.run(1, rules=bad), lambda: life.random_params(random.Random(1), rules=bad),
                     lambda: SIM.rollout(1, rules=bad), lambda: SIM.generate(random.Random(1), 10, rules=bad)):
            with pytest.raises(ValueError):
                call()


def test_one_gate_judges_both_rule_sets_and_keeps_them_apart():
    six, seven = SIM.rollout(7), SIM.rollout(7, rules=7)
    assert six.summary["first_replicator_step"] == 23 and seven.summary["first_replicator_step"] == 20
    assert SIM.check("life seed 7 final first_replicator_step", "2 3").ok
    assert not SIM.check("life seed 7 final first_replicator_step", "2 0").ok
    assert SIM.check("life seed 7 rules 7 final first_replicator_step", "2 0").ok
    assert not SIM.check("life seed 7 rules 7 final first_replicator_step", "2 3").ok
    # the keys of one rule set are no questions of the other
    for prompt in ("life seed 7 step 3 spent", "life seed 7 param hydrolysis", "life seed 7 step 3 copies",
                   "life seed 7 final max_length", "life seed 7 step 3 lineages"):
        assert not SIM.owns(prompt) and SIM.owns(prompt.replace("seed 7 ", "seed 7 rules 7 ")), prompt
    for prompt in ("life seed 7 rules 7 step 3 generations", "life seed 7 rules 7 param decay"):
        assert not SIM.owns(prompt) and SIM.owns(prompt.replace(" rules 7", "")), prompt
    assert SIM.check("life seed 7 step 4 1 stage", "growth").reason == "no such step"  # round 6 ends at step 40
    assert SIM.check("life seed 7 rules 7 step 4 1 stage", seven.steps[41]["stage"]).ok


# ---------------------------------------------------------------------------------------------
# how rules 7 is called, and that it repeats exactly
# ---------------------------------------------------------------------------------------------

def test_a_rules7_rollout_and_its_keys():
    assert isinstance(SIM, Simulation) and life.SIM == SIM.sim == "life"
    r = life.run(5, rules=7)
    assert isinstance(r, Rollout) and r.sim == "life" and life.is_rules7(r)
    assert list(r.params) == [*life.PARAM_KEYS7, "rules"] and r.params["rules"] == 7
    assert list(life.PARAM_KEYS7) == ["monomers", "inflow", "poly_rate", "hydrolysis", "mu", "gap"]
    assert len(r.steps) == life.STEPS7 + 1 == 81
    assert list(r.steps[0]) == list(life.KEYS7) and list(r.summary) == list(life.SUMMARY_KEYS7)
    assert life.METRICS7 == [k for k in life.KEYS7 if k != "step"] and len(life.METRICS7) == 16
    assert r.steps[0] == {"step": 0, "free_monomers": r.params["monomers"], "spent": 0, "bound": 0, "polymers": 0,
                          "replicators": 0, "longest": 0, "mean_length": 0.0, "replicator_length": 0.0, "diversity": 0,
                          "lineages": 0, "dominant_share": 0.0, "effective_lineages": 0.0, "copies": 0, "failed": 0,
                          "origins": 0, "stage": "soup"}
    for s in r.steps:  # plain python numbers, so equal means identical
        assert all(type(v) in (int, float, str) for v in s.values())
    again = life.run(5, **r.params)  # the params of a rollout replay it, rules and all
    assert (again.params, again.steps, again.summary) == (r.params, r.steps, r.summary)
    assert SIM.run(5, rules=7).steps == SIM.run(5, 7).steps == r.steps
    assert SIM.rollout(5, rules=7).steps == r.steps and SIM.rollout(5, rules=7) is SIM.rollout(5, rules=7)
    # the names: copies and hydrolysis, replicators and never the word life
    assert "generations" not in life.KEYS7 and "decay" not in life.PARAM_KEYS7
    assert {"spent", "copies", "failed", "origins", "lineages", "effective_lineages", "replicator_length"} <= set(life.KEYS7)
    for text in (*life.KEYS7.values(), *life.SUMMARY_KEYS7.values(), *life.PARAM_KEYS7.values()):
        assert "life" not in text and "alive" not in text and "living" not in text, text
    assert r.summary["outcome"] in life.OUTCOMES7 and "dominated" not in life.STAGES7


def test_random_params_share_their_draws_with_round_6():
    rng = random.Random(9)
    seen = set()
    for seed in range(1, 301):
        six, seven = life.random_params(random.Random(seed)), life.random_params(random.Random(seed), rules=7)
        assert list(seven) == list(life.PARAM_KEYS7)
        assert {k: v for k, v in six.items() if k != "decay"} == {k: v for k, v in seven.items() if k != "hydrolysis"}
        assert seven["hydrolysis"] in life.HYDROLYSIS
        seen.add(seven["hydrolysis"])
        p = life.random_params(rng, rules=7)
        assert 5000 <= p["monomers"] <= 20000 and (p["inflow"] == 0 or (40 <= p["inflow"] <= 400 and p["inflow"] % 4 == 0))
        assert p["poly_rate"] in (0.002, 0.005, 0.01, 0.02) and 0.001 <= p["mu"] <= 0.05 and p["gap"] in (0, 1, 2, 3)
    assert seen == set(life.HYDROLYSIS) == {0.002, 0.005, 0.01, 0.02, 0.05, 0.1}
    assert life.random_params(random.Random(7), rules=7) == {"monomers": 10305, "inflow": 240, "poly_rate": 0.002,
                                                            "hydrolysis": 0.01, "mu": 0.02, "gap": 0}
    assert life.run(7, rules=7).params == {**life.random_params(random.Random(7), rules=7), "rules": 7}


def test_same_seed_same_rollout():
    a, b = life.run(5, rules=7), life.run(5, rules=7)
    assert a.params == b.params and a.steps == b.steps and a.summary == b.summary
    assert [ln.text for ln in life.lines(a)] == [ln.text for ln in life.lines(b)] == [ln.text for ln in life.lines(a, rules=7)]
    assert [ln.text for ln in life.lines(a)] == [ln.text for ln in life.lines7(a)]
    one, two, other = (SIM.generate(random.Random(s), 300, rules=7) for s in (3, 3, 4))
    assert [ln.text for ln in one] == [ln.text for ln in two] != [ln.text for ln in other]
    assert life.run(1, rules=7).steps != life.run(2, rules=7).steps
    assert life.run(1, rules=7).steps[:41] != life.run(1).steps


def test_forty_seeded_runs_repeat_exactly_and_conserve(forty):
    for seed, a in forty.items():
        b = life.run(seed, rules=7)
        assert a.params == b.params == {**life.random_params(random.Random(seed), rules=7), "rules": 7}
        assert a.steps == b.steps and a.summary == b.summary, seed
        assert len(a.steps) == life.STEPS7 + 1
        v = SIM.conserved(a)
        assert v.ok, (seed, v)
        assert life.conserved7(a).ok and life.conserved(a).ok
        for t, s in enumerate(a.steps):
            assert s["free_monomers"] + s["spent"] + s["bound"] == a.params["monomers"] + a.params["inflow"] * t
            assert 4 * s["copies"] <= a.params["monomers"] + a.params["inflow"] * t  # no letter is used twice


def test_three_rollouts_are_pinned():
    assert {seed: digest(life.run(seed, rules=7)) for seed in DIGESTS7} == DIGESTS7


def test_a_fresh_interpreter_gives_the_same_rollout():
    code = ("import hashlib, json; from haishool.cosmos import life; r = life.run(7, rules=7); "
            "print(hashlib.sha256(json.dumps([r.params, r.steps, r.summary], sort_keys=True).encode()).hexdigest()[:16])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         cwd=Path(__file__).resolve().parents[1], env={**__import__("os").environ, "PYTHONHASHSEED": "123"})
    assert out.stdout.strip() == DIGESTS7[7]


def test_bad_parameters_are_refused_and_odd_ones_conserve():
    for bad in ({"inflow": 10}, {"inflow": -4}, {"monomers": -1}, {"monomers": 10.5}, {"mu": 2}, {"mu": -0.1},
                {"poly_rate": 1.5}, {"hydrolysis": -0.1}, {"hydrolysis": 1.5}, {"gap": -1}, {"gap": 13},
                {"decay": 0.1}, {"mutation": 0.1}):
        with pytest.raises(ValueError):
            life.run(1, rules=7, **bad)
    for params in ({"monomers": 0, "inflow": 0}, {"monomers": 0, "inflow": 400},
                   {"monomers": 3, "inflow": 4, "poly_rate": 1, "hydrolysis": 0, "mu": 1, "gap": 12},
                   {"monomers": 20000, "inflow": 0, "poly_rate": 1.0, "hydrolysis": 0.5, "mu": 0.01, "gap": 3},
                   {"hydrolysis": 1.0}, {"hydrolysis": 0.0, "mu": 0.0}, {"mu": 1.0}):
        r = life.run(3, rules=7, **params)
        v = SIM.conserved(r)
        assert v.ok, (params, v)
        last = r.steps[-1]
        assert last["free_monomers"] + last["spent"] + last["bound"] == r.params["monomers"] + r.params["inflow"] * life.STEPS7
    assert life.run(3, rules=7, monomers=0, inflow=0).summary["outcome"] == "none"
    assert life.run(3, rules=7, hydrolysis=0.0).steps[-1]["spent"] == 0  # nothing breaks, nothing is spent


# ---------------------------------------------------------------------------------------------
# lines
# ---------------------------------------------------------------------------------------------

def test_lines_are_dense_and_short(all_lines, forty):
    assert len(all_lines) > 10000
    longest = 0
    for ln in all_lines + [ln for r in forty.values() for ln in life.lines(r)[:1 + life.STEPS7 + 1]]:
        assert is_dense(ln.text), ln.text
        assert n_tokens(ln.text) <= MAX_TOKENS, ln.text
        longest = max(longest, n_tokens(ln.text))
        assert ln.topic == "life"
        assert ln.text.startswith(("life seed ", "q life seed ")) and " rules 7 " in ln.text
        if ln.kind != "record":
            assert "." not in ln.prompt and "." not in ln.answer
            assert ln.text == f"q {ln.prompt}. a {ln.answer}."
    assert {ln.kind for ln in all_lines} == {"record", "fact", "calc"}
    assert 80 <= longest <= 117  # measured 97 on these seeds; 117 is the most a record can have (next test)


def test_the_longest_possible_record_fits():
    # five-digit counts everywhere (20000 + 400 * 80 letters at most, 4 of them per replicator, lineage or origin at
    # least), a share in e notation, the longest stage name: more than any run can have at once, and it still fits
    worst = {"step": 80, "free_monomers": 52000, "spent": 52000, "bound": 52000, "polymers": 26000, "replicators": 13000,
             "longest": 16, "mean_length": 3.33, "replicator_length": 4.44, "diversity": 13000, "lineages": 13000,
             "dominant_share": 0.0000769, "effective_lineages": 12300.0, "copies": 13000, "failed": 13000, "origins": 13000,
             "stage": "first_replicator"}
    assert list(worst) == list(life.KEYS7)
    r = Rollout("life", 9999, {"rules": 7}, [worst] * (life.STEPS7 + 1))
    line = life.state_line7(r, life.STEPS7)
    assert is_dense(line.text) and n_tokens(line.text) == 117 <= MAX_TOKENS
    params = life.params_line7(Rollout("life", 9999, {"monomers": 20000, "inflow": 400, "poly_rate": 0.002,
                                                      "hydrolysis": 0.002, "mu": 0.001, "gap": 3, "rules": 7}, []))
    assert n_tokens(params.text) == 46


def test_every_line_kind_is_there(rollouts):
    texts = [ln.text for ln in life.lines(rollouts[7])]
    assert texts[0] == ("life seed 7 rules 7 params. monomers 1 0 3 0 5. inflow 2 4 0. poly_rate 0 point 0 0 2. "
                        "hydrolysis 0 point 0 1. mu 0 point 0 2. gap 0.")
    assert ("life seed 7 rules 7 step 3 4. free_monomers 1 5 4 6 7. spent 9 7. bound 2 9 0 1. polymers 5 9 6. "
            "replicators 2 1 5. longest 1 2. mean_length 4 point 8 7. replicator_length 6 point 9 8. diversity 2 4. "
            "lineages 4. dominant_share 0 point 6 8 8. effective_lineages 1 point 9. copies 2 4 8. failed 0. origins 4. "
            "stage growth.") in texts
    for want in ("q life seed 7 rules 7 step 3 4 replicators. a 2 1 5.", "q life seed 7 rules 7 step 3 4 next replicators. a 3 2 4.",
                 "q life seed 7 rules 7 step 4 0 stage. a competition.", "q life seed 7 rules 7 step 4 0 failed. a 1 5 1 5.",
                 "q life seed 7 rules 7 final first_replicator_step. a 2 0.", "q life seed 7 rules 7 final outcome. a dominated.",
                 "q life seed 7 rules 7 final max_length. a 1 6.", "q life seed 7 rules 7 final replicator_length. a 4 point 5 2.",
                 "q life seed 7 rules 7 param hydrolysis. a 0 point 0 1.", "q life seed 7 rules 7 param mu. a 0 point 0 2."):
        assert want in texts, want
    assert sum(t.startswith("life seed 7 rules 7 step ") for t in texts) == life.STEPS7 + 1
    assert not any(" step. " in t or " rules. " in t or " param rules" in t for t in texts)  # both are in the prompt
    n = len(life.METRICS7)
    assert len(texts) == 1 + 81 + len(life.SUMMARY_KEYS7) + len(life.PARAM_KEYS7) + 81 * n + 80 * n == 2675
    assert len(set(texts)) == len(texts)
    assert not set(texts) & {ln.text for ln in life.lines(SIM.rollout(7))}  # no line of round 6 among them


def test_never_is_a_word_not_minus_one():
    seed = next(s for s in range(150, 400) if SIM.rollout(s, rules=7).summary["outcome"] == "none")
    r = SIM.rollout(seed, rules=7)
    assert seed == 183 and r.summary["first_replicator_step"] == "never" and r.summary["peak_replicators"] == 0
    assert "q life seed 1 8 3 rules 7 final first_replicator_step. a never." in [ln.text for ln in life.lines(r)]
    prompt = "life seed 1 8 3 rules 7 final first_replicator_step"
    assert SIM.check(prompt, "never").ok
    for wrong in ("minus 1", "0", "none", "8 0", ""):
        assert not SIM.check(prompt, wrong).ok
    assert all(s["stage"] == "soup" and s["origins"] == 0 for s in r.steps)
    assert not any(ln.answer.startswith("minus") for s in SEEDS for ln in life.lines(SIM.rollout(s, rules=7)))


def test_generate_gives_n_lines_over_several_seeds(generated):
    assert len(generated) == 600
    seeds = {ln.text.split(" rules 7 ")[0].split("seed ")[1] for ln in generated}
    assert len(seeds) >= 3
    assert any(ln.kind == "record" and " params. " in ln.text for ln in generated)
    for word in (" final ", " next ", " param "):
        assert any(word in ln.prompt for ln in generated if ln.kind != "record")
    assert len(SIM.generate(random.Random(2), 7, rules=7)) == 7 and SIM.generate(random.Random(2), 0, rules=7) == []


# ---------------------------------------------------------------------------------------------
# the gate
# ---------------------------------------------------------------------------------------------

def test_gate_agrees_with_its_own_lines(all_lines):
    for ln in all_lines:
        if ln.kind == "record":
            continue
        assert SIM.owns(ln.prompt) and life.owns7(ln.prompt) and life.owns(ln.prompt), ln.prompt
        v = SIM.check(ln.prompt, ln.answer)
        assert v.ok and v.expected == ln.answer and v.reason == "", (ln.text, v)
        assert life.check7(ln.prompt, ln.answer) == v == life.check(ln.prompt, ln.answer)


def test_gate_rejects_wrong_answers(all_lines):
    words = life.STAGES7 + life.OUTCOMES7 + ("never", "dominated")
    for ln in all_lines:
        if ln.kind == "record":
            continue
        x = parse_num(ln.answer)
        if x is None:  # a word: every other word is wrong
            wrongs = [w for w in words if w != ln.answer] + ["wrong", "yes", "minus 1"]
        else:  # a number: far off, and with its first digit changed
            first = next(i for i, w in enumerate(ln.answer.split()) if w.isdigit())
            changed = ln.answer.split()
            changed[first] = str((int(changed[first]) + 2) % 10)
            wrongs = [num(x * 2 + 7 if isinstance(x, int) else x * 2 + 1), " ".join(changed), "growth", "never"]
        for wrong in wrongs + [""]:
            v = SIM.check(ln.prompt, wrong)
            assert not v.ok and v.expected == ln.answer and v.reason, (ln.text, wrong, v)


def test_small_whole_numbers_are_exact_larger_counts_within_one_or_5_percent(rollouts):
    r = rollouts[7]
    head = "life seed 7 rules 7 step"
    # identities and table values: exact
    for prompt, value in ((f"{head} 9 step", 9), (f"{head} 9 next step", 10), ("life seed 7 rules 7 final first_replicator_step", 20),
                          ("life seed 7 rules 7 final max_length", 16), ("life seed 7 rules 7 param gap", 0),
                          ("life seed 7 rules 7 param monomers", 10305), ("life seed 7 rules 7 param inflow", 240)):
        assert SIM.check(prompt, num(value)).ok, prompt
        assert not SIM.check(prompt, num(value + 1)).ok and not SIM.check(prompt, num(value - 1)).ok, prompt
    assert SIM.check("life seed 7 rules 7 param hydrolysis", "0 point 0 1").ok
    assert not SIM.check("life seed 7 rules 7 param hydrolysis", "0 point 0 1 0 3").ok
    # longest is exact whatever its size: 17 letters cannot exist
    seed, t = next((seed, t) for seed, x in rollouts.items() for t, s in enumerate(x.steps) if s["longest"] == 16)
    longest = f"life seed {num(seed)} rules 7 step {num(t)} longest"
    assert SIM.check(longest, "1 6").ok and not SIM.check(longest, "1 7").ok and not SIM.check(longest, "1 5").ok
    # diversity, lineages, origins and failed below 20: 2 for 1 is wrong
    for key in ("diversity", "lineages", "origins", "failed"):
        seed, t, n = next((seed, t, s[key]) for seed, x in rollouts.items() for t, s in enumerate(x.steps)
                          if 1 <= s[key] < life.SMALL7)
        small = f"life seed {num(seed)} rules 7 step {num(t)} {key}"
        assert SIM.check(small, num(n)).ok, key
        for off in (n + 1, n - 1, n + 0.5):
            assert not SIM.check(small, num(off)).ok, (key, n, off)
    assert r.steps[20]["diversity"] == 1 and not SIM.check(f"{head} 2 0 diversity", "2").ok
    assert SIM.check("life seed 7 step 2 3 diversity", "2").ok  # round 6 still takes 2 for 1, as it always did
    # large counts: within 1 or 5 percent, and whole
    for key in ("replicators", "failed", "diversity", "copies", "spent"):
        seed, t, n = next((seed, t, s[key]) for seed, x in rollouts.items() for t, s in enumerate(x.steps)
                          if 40 <= s[key] < 900)
        p = f"life seed {num(seed)} rules 7 step {num(t)} {key}"
        assert SIM.check(p, num(n + 1)).ok and SIM.check(p, num(n - 1)).ok and SIM.check(p, num(int(n * 1.04))).ok, key
        assert not SIM.check(p, num(int(n * 1.2) + 2)).ok and not SIM.check(p, num(n + 0.5)).ok, key
    # a zero is exact both ways; replicators keep the round-6 tolerance of one
    assert r.steps[3]["replicators"] == 0
    assert SIM.check(f"{head} 3 replicators", "0").ok and not SIM.check(f"{head} 3 replicators", "1").ok
    assert r.steps[20]["replicators"] == 1
    assert SIM.check(f"{head} 2 0 replicators", "2").ok and not SIM.check(f"{head} 2 0 replicators", "0").ok
    # measured numbers: within 5 percent
    for key in ("mean_length", "replicator_length", "effective_lineages", "dominant_share"):
        x = r.steps[60][key]
        p = f"{head} 6 0 {key}"
        assert SIM.check(p, num(x * 1.03)).ok and not SIM.check(p, num(x * 1.2)).ok, key
    assert life.agree7("longest", 16, "1 6") and not life.agree7("longest", 16, "1 7")
    assert life.agree7("lineages", 19, "1 9") and not life.agree7("lineages", 19, "2 0") and life.agree7("lineages", 20, "2 1")
    assert life.agree7("first_replicator_step", "never", "never") and not life.agree7("first_replicator_step", "never", "minus 1")
    assert not life.agree7("replicators", 100, "1 e 9 9 9") and not life.agree7("mean_length", 4.5, "1 e 9 9 9")


def test_owns_only_its_own_prompts(all_lines, lessons):
    for other in OTHER_PROMPTS + (
            "life seed 7 rules 7 step 3 clumps", "life seed 7 rules 7 final nope", "life seed 7 rules 7 param clumps",
            "life seed 7 rules 7 final mu", "life seed 7 rules 7 param outcome", "life seed 7 rules 7 param rules",
            "life seed x rules 7 step 3 replicators", "life seed 7 rules 7 step 3", "life seed 7 rules 7",
            "life seed 0 7 rules 7 step 3 replicators", "life seed 7 rules 7 step 0 3 replicators",
            "life seed minus 7 rules 7 step 3 replicators", "life seed 7 rules 7 step 3 replicators ",
            "life  seed 7 rules 7 step 3 replicators", "q life seed 7 rules 7 step 3 replicators",
            "life seed 7 rules 7 step 3 next next replicators", "life seed 7 rules 7 final next replicators",
            "life seed 7 rules 7 step 3 final outcome", "life seed 7 rules 6 step 3 replicators",
            "life seed 7 rules 8 step 3 replicators", "life seed 7 rules 7 7 step 3 replicators",
            "life seed 7 rules step 3 replicators", "life rules 7 seed 7 step 3 replicators",
            "life seed 7 step 3 rules 7 replicators", "life7 seed 7 step 3 replicators", "life7 seed 7 final outcome",
            "life7 seed 7 rules 7 final outcome", "life predict survival_probability length 4 decay 1",
            "life predict expected_mutations length 8 mu 0 point 0 2", "life predict next_total total 1 0 0 inflow 8",
            "life predict copy_probability length 4", ""):
        assert not SIM.owns(other) and not life.owns7(other) and not life.owns(other) and not LESSONS.owns(other), other
        v = SIM.check(other, "4")
        assert not v.ok and v.expected is None and v.reason == "not my question", other
        assert life.check7(other, "4") == v
    assert SIM.owns("life seed 0 rules 7 step 0 stage") and SIM.check("life seed 0 rules 7 step 0 stage", "soup").ok
    assert SIM.check("life seed 7 rules 7 step 9 9 replicators", "4").reason == "no such step"
    assert SIM.check(f"life seed 7 rules 7 step {num(life.STEPS7)} next replicators", "4").reason == "no such step"
    assert SIM.check(f"life seed 7 rules 7 step {num(life.STEPS7)} stage", SIM.rollout(7, rules=7).steps[-1]["stage"]).ok
    # round-6 questions are not rules-7 questions, and a story is no lesson
    for prompt in ("life seed 7 step 3 replicators", "life seed 7 final outcome", "life seed 7 param mu"):
        assert SIM.owns(prompt) and not life.owns7(prompt) and not LESSONS.owns(prompt) and life.owns(prompt)
    for ln in all_lines[:400]:
        if ln.kind != "record":
            assert not LESSONS.owns(ln.prompt)
    for ln in lessons:
        assert not SIM.owns(ln.prompt) and life.owns(ln.prompt)


def test_conserved_catches_a_broken_rollout(rollouts):
    def broken(change) -> bool:
        r = copy.deepcopy(rollouts[7])
        change(r)
        v = SIM.conserved(r)
        return not v.ok and bool(v.reason)

    def setter(t, key, value):
        return lambda r: r.steps[t].__setitem__(key, value(r.steps[t][key]) if callable(value) else value)

    def move(t, source, target, n):  # the ledger still adds up
        def change(r):
            r.steps[t][source] -= n
            r.steps[t][target] += n
        return change

    good = rollouts[7]
    assert SIM.conserved(good).ok and good.steps[34]["stage"] == "growth" and good.steps[60]["stage"] == "competition"
    assert good.steps[34]["lineages"] == 4 and good.steps[34]["failed"] == 0
    assert broken(setter(5, "free_monomers", lambda x: x + 1))
    assert broken(setter(50, "spent", lambda x: x - 1))
    assert broken(setter(50, "bound", lambda x: x - 1))
    assert broken(move(50, "spent", "free_monomers", good.steps[50]["spent"] - good.steps[49]["spent"] + 1))  # un-spending
    assert broken(setter(34, "replicators", lambda x: 10 ** 6))
    assert broken(setter(34, "replicators", 0))
    assert broken(setter(34, "dominant_share", 1.5))
    assert broken(setter(34, "dominant_share", 0.001))  # less than one in lineages
    assert broken(setter(34, "effective_lineages", 0.5))
    assert broken(setter(34, "effective_lineages", 3.9))  # does not fit a largest share of 0.688
    assert broken(setter(34, "effective_lineages", 1.2))
    assert broken(setter(34, "lineages", 0))
    assert broken(setter(34, "lineages", 5))  # more lineages than origins
    assert broken(setter(34, "origins", 0))
    assert broken(setter(34, "origins", lambda x: x + 1))  # origins never fall, so step 35 gives it away
    assert broken(setter(34, "diversity", 0))
    assert broken(setter(34, "diversity", 10 ** 6))
    assert broken(setter(34, "copies", -1))
    assert broken(setter(34, "copies", lambda x: x + 10 ** 6))  # more copies than replicators, and than letters
    assert broken(setter(34, "failed", 1))  # then the stage would be competition
    assert broken(setter(60, "failed", 0))
    assert broken(setter(60, "failed", 10 ** 6))
    assert broken(setter(34, "mean_length", lambda x: x + 1))
    assert broken(setter(34, "longest", 1))
    assert broken(setter(34, "longest", 17))
    assert broken(setter(34, "replicator_length", 2.0))
    assert broken(setter(34, "replicator_length", 15.9))
    for word in ("soup", "competition", "decline", "first_replicator", "extinct", "dominated", "boiling"):
        assert broken(setter(34, "stage", word)), word
    assert broken(setter(34, "step", 35))
    for key, value in (("outcome", "extinct"), ("outcome", "coexist"), ("peak_replicators", 1), ("first_replicator_step", 0),
                       ("first_replicator_step", "never"), ("max_length", 13), ("origins", 5), ("lineages", 3),
                       ("replicator_length", 9.9)):
        assert broken(lambda r: r.summary.__setitem__(key, value)), key
    assert broken(lambda r: r.params.__setitem__("monomers", 10306))
    assert broken(lambda r: r.params.__setitem__("hydrolysis", 0.05))  # another max_length


# ---------------------------------------------------------------------------------------------
# the vessel
# ---------------------------------------------------------------------------------------------

def test_letters_are_conserved_kind_by_kind_and_lineages_follow_the_motif():
    for seed, params in ((1, {}), (3, {}), (11, {}), (20, {}), (107, {"inflow": 0}), (126, {"hydrolysis": 0.1}),
                         (5, {"monomers": 20000, "inflow": 400, "poly_rate": 0.02, "hydrolysis": 0.05, "mu": 0.05, "gap": 3})):
        p = life._params7(seed, params)
        r = life.run(seed, rules=7, **params)
        start = before = None
        for t, v in enumerate(life.simulate7(seed, p)):
            start = v.free.copy() if start is None else start
            assert v.free.dtype == v.spent.dtype == v.lineage.dtype == np.int64 and v.seqs.dtype == np.uint8
            assert (v.free >= 0).all() and (v.spent >= 0).all()
            assert (v.free + v.spent + life.letter_counts(v.seqs).sum(0) == start + p["inflow"] // 4 * t).all(), (seed, t)
            lens = lengths(v.seqs)
            assert (lens >= 2).all() and ((v.seqs != life.PAD) == (np.arange(life.MAX_LEN)[None, :] < lens[:, None])).all()
            # a lineage number exactly where the motif is, and only numbers of lineages that began
            assert ((v.lineage >= 0) == life.has_motif(v.seqs, p["gap"])).all(), (seed, t)
            assert len(v.lineage) == len(v.seqs) and (v.lineage >= -1).all() and (v.lineage < max(v.origins, 1)).all()
            s = r.steps[t]
            assert (s["free_monomers"], s["spent"], s["polymers"], s["copies"], s["failed"], s["origins"]) == \
                (int(v.free.sum()), int(v.spent.sum()), len(v.seqs), v.copies, v.failed, v.origins)
            assert s["replicators"] == int((v.lineage >= 0).sum()) and s["bound"] == int(lens.sum())
            assert s["lineages"] == len(set(v.lineage[v.lineage >= 0].tolist()))
            if before is not None:
                assert (v.spent >= before.spent).all() and v.origins >= before.origins and v.copies >= before.copies
                # a copy needs letters, so a step that made any left the pool smaller than the inflow made it
                if v.copies > before.copies and v.spent.sum() == before.spent.sum():
                    assert v.free.sum() < before.free.sum() + p["inflow"]
            if v.failed and len(v.seqs) < life.MAX_POLYMERS:  # no copy fails while every letter is plentiful
                assert v.free.min() < life.MAX_LEN, (seed, t)
            before = v
        assert start.sum() == p["monomers"] and start.max() - start.min() <= 1


def test_the_rule_functions():
    assert life.intact_probability(2, 0.1) == 0.9 and life.intact_probability(16, 0.0) == 1.0
    assert life.intact_probability(4, 0.5) == 0.125 and life.intact_probability(5, 1.0) == 0.0
    for h in life.HYDROLYSIS:
        for n in range(2, life.MAX_LEN + 1):
            assert life.intact_probability(n, h) == pytest.approx((1 - h) ** (n - 1), rel=1e-12)
            assert life.intact_probability(n + 1, h) < life.intact_probability(n, h)  # long chains break sooner
    assert [life.copy_probability(n) for n in (2, 4, 5, 8, 16)] == [1.0, 1.0, 0.8, 0.5, 0.25]
    assert life.motif_fidelity(0.0) == 1.0 and life.motif_fidelity(0.5) == 0.0625 and life.motif_fidelity(1.0) == 0.0
    assert life.motif_fidelity(0.05) == pytest.approx(0.95 ** 4, rel=1e-12)
    assert life.growth_factor(4, 0.0, 0.0) == 2.0 and life.growth_factor(8, 0.0, 0.0) == 1.5
    assert life.growth_factor(13, 0.02, 0.001) > 1 > life.growth_factor(14, 0.02, 0.001)
    # the length limit: the values of the review
    assert [life.max_length(h, 0.001) for h in life.HYDROLYSIS] == [16, 16, 16, 13, 8, 5]
    assert [life.max_length(h, 0.05) for h in (0.02, 0.05, 0.1)] == [12, 7, 5]
    assert life.max_length(0.3, 0.05) == 0 and life.max_length(0.0, 0.0) == 16 and life.max_length(0.0, 1.0) == 0
    for h in (0.0, 0.01, 0.05, 0.1, 0.2):
        for mu in (0.0, 0.01, 0.05, 0.2):
            top = life.max_length(h, mu)
            for n in range(4, life.MAX_LEN + 1):  # every length up to the limit multiplies, none above it
                assert (life.growth_factor(n, h, mu) > 1) == (n <= top), (h, mu, n)
    # the error threshold of the motif alone is far outside the mu of the level
    assert life.motif_fidelity(0.05) > 0.8 and 2 * life.motif_fidelity(0.3) < 1 < 2 * life.motif_fidelity(0.15)
    assert life.polymer_letters(10000, 0.01) == 100 and life.polymer_letters(5000, 0.01) == 25
    assert life.polymer_letters(20000, 0.01) == 400 and life.polymer_letters(0, 0.02) == 0
    assert life.polymer_letters(50000, 0.5) == 50000 and type(life.polymer_letters(7, 0.5)) is int
    # exactly the rule, poly_rate * free * free / 10000 rounded down, whatever floating point makes of it:
    # int(0.043 * 20000 * (20000 / 10000)) is 1719, int(0.01 * 41000 * 4.1) is 1680
    assert life.polymer_letters(20000, 0.043) == 1720 and life.polymer_letters(41000, 0.01) == 1681
    assert life.polymer_letters(41000, 0.02) == 3362 and life.polymer_letters(np.int64(41000), np.float64(0.02)) == 3362
    rng = random.Random(8)
    for _ in range(20000):
        free, thousandths = rng.randint(0, 60000), rng.randint(0, 50)
        assert life.polymer_letters(free, thousandths / 1000) == min(free, thousandths * free * free // 10 ** 7)
    for rate in (0.002, 0.005, 0.01, 0.02):  # the values of the level, at round numbers of monomers
        for free in range(0, 60001, 500):
            assert life.polymer_letters(free, rate) == min(free, round(rate * 1000) * free * free // 10 ** 7), (free, rate)
    assert life.inflow_per_kind(240) == 60 and life.inflow_per_kind(0) == 0 and life.next_total(10305, 240) == 10545
    assert [life.stage(*a) for a in ((0, 0, 0, 0), (1, 0, 0, 0), (3, 0, 1, 0), (0, 3, 0, 1), (9, 5, 2, 1), (4, 5, 2, 1),
                                     (4, 5, 0, 1), (5, 5, 0, 1), (6, 5, 0, 1))] == \
        ["soup", "first_replicator", "first_replicator", "extinct", "competition", "competition", "decline", "growth", "growth"]
    assert [life.outcome(*a) for a in ((0, 0, 0.0), (9, 0, 0.0), (10, 0, 0.0), (50, 25, 3.0), (50, 26, 1.99),
                                       (50, 26, 2.0), (1, 1, 1.0))] == \
        ["none", "fizzled", "extinct", "declining", "dominated", "coexist", "dominated"]
    assert [life.hydrolysis_at(t) for t in (200, 246, 259.9, 260, 274, 275, 289, 290, 304, 305, 319, 320, 500)] == \
        [0.002, 0.002, 0.002, 0.005, 0.005, 0.01, 0.01, 0.02, 0.02, 0.05, 0.05, 0.1, 0.1]


def test_a_copy_is_the_reverse_complement():
    rng = np.random.default_rng(3)
    n = 4000
    lens = rng.integers(2, life.MAX_LEN + 1, size=n)
    seqs = np.full((n, life.MAX_LEN), life.PAD, dtype=np.uint8)
    mask = np.arange(life.MAX_LEN)[None, :] < lens[:, None]
    seqs[mask] = rng.integers(0, 4, size=int(mask.sum()), dtype=np.uint8)
    rc = life.reverse_complement(seqs)
    pair = str.maketrans("acgu", "ugca")
    assert rc.dtype == np.uint8 and strings(rc) == [s[::-1].translate(pair) for s in strings(seqs)]
    assert (lengths(rc) == lens).all() and (life.reverse_complement(rc) == seqs).all()
    for gap in range(4):  # the motif is its own reverse complement: both strands carry it or neither
        assert (life.has_motif(rc, gap) == life.has_motif(seqs, gap)).all()
    assert strings(life.reverse_complement(np.array([[0, 2, 1, 3] + [life.PAD] * 12], dtype=np.uint8))) == ["agcu"]
    # a strand and its opposite strand are one sequence, and no two others are
    codes = life.canonical(seqs)
    assert codes.dtype == np.int64 and (codes == life.canonical(rc)).all()
    classes = {min(s, s[::-1].translate(pair)) for s in strings(seqs)}
    assert len(np.unique(codes)) == len(classes)
    # in a vessel without errors and without breaks every lineage is its founder and the opposite strand
    both = 0
    for seed in range(100, 110):
        p = life._params7(seed, {"mu": 0.0, "hydrolysis": 0.0})
        *_, v = life.simulate7(seed, p)
        r = life.run(seed, rules=7, mu=0.0, hydrolysis=0.0)
        rep = v.lineage >= 0
        for k in set(v.lineage[rep].tolist()):
            mine = set(strings(v.seqs[v.lineage == k]))
            assert len(mine) <= 2 and len({min(s, s[::-1].translate(pair)) for s in mine}) == 1, (seed, k, mine)
            both += len(mine) == 2
        assert r.summary["diversity"] <= r.summary["lineages"] and r.steps[-1]["spent"] == 0
    assert both >= 10  # plus and minus strands are both there


def test_hydrolysis_cuts_chains_and_spends_single_letters():
    # no polymerisation and no copying after the soup is made: only hydrolysis acts
    seed = 4
    p = life._params7(seed, {"hydrolysis": 0.1, "inflow": 0, "poly_rate": 0.02, "gap": 0})
    vs = list(life.simulate7(seed, p))
    r = life.run(seed, rules=7, hydrolysis=0.1, inflow=0, poly_rate=0.02, gap=0)
    assert all(b["spent"] >= a["spent"] for a, b in zip(r.steps, r.steps[1:])) and r.steps[-1]["spent"] > 1000
    assert r.steps[1]["spent"] == 0  # nothing to break in the first step
    longest = [s["longest"] for s in r.steps if s["polymers"]]
    means = [s["mean_length"] for s in r.steps if s["polymers"]]
    assert longest[0] >= 12 and means[-1] < means[0] and min(lengths(vs[-1].seqs)) >= 2
    # long chains break sooner: the share of chains of 8 or more letters falls
    share = [float((lengths(v.seqs) >= 8).mean()) for v in vs[1:] if len(v.seqs)]
    assert share[-1] < 0.3 * share[0]
    # without hydrolysis nothing is spent and nothing breaks
    calm = life.run(seed, rules=7, hydrolysis=0.0, inflow=0, poly_rate=0.02, gap=0)
    assert all(s["spent"] == 0 for s in calm.steps)


def test_the_simulation_calls_the_lesson_functions(monkeypatch):
    base = life.run(9, rules=7)
    assert base.steps[-1]["copies"] > 0 and base.steps[-1]["spent"] > 0 and base.steps[1]["polymers"] > 0

    def with_patch(name, replacement) -> Rollout:
        with monkeypatch.context() as m:
            m.setattr(life, name, replacement)
            return life.run(9, rules=7)

    assert all(s["spent"] == 0 for s in with_patch("intact_probability", lambda length, hydrolysis: 1.0).steps)
    assert with_patch("copy_probability", lambda length: 0.0).steps[-1]["copies"] == 0
    assert all(s["polymers"] == 0 for s in with_patch("polymer_letters", lambda free, poly_rate: 0).steps)
    assert with_patch("max_length", lambda hydrolysis, mu: 5).summary["max_length"] == 5
    assert with_patch("motif_fidelity", lambda mu: 0.0).summary["max_length"] == 0  # growth_factor <= 1 then
    assert with_patch("growth_factor", lambda length, hydrolysis, mu: 0.5).summary["max_length"] == 0
    assert {s["stage"] for s in with_patch("stage", lambda *a: "soup").steps} == {"soup"}
    assert with_patch("outcome", lambda *a: "none").summary["outcome"] == "none"
    starved = with_patch("inflow_per_kind", lambda inflow: 0)
    assert base.params["inflow"] > 0 and starved.steps[-1]["free_monomers"] + starved.steps[-1]["spent"] + \
        starved.steps[-1]["bound"] == base.params["monomers"]
    assert not life.conserved7(starved).ok  # the gate adds the inflow with next_total
    with monkeypatch.context() as m:
        m.setattr(life, "next_total", lambda total, inflow: total)
        assert not life.conserved7(base).ok
    assert life.conserved7(base).ok and life.run(9, rules=7).steps == base.steps


# ---------------------------------------------------------------------------------------------
# lessons
# ---------------------------------------------------------------------------------------------

def test_lessons_are_the_functions_of_the_simulation(lessons):
    assert isinstance(LESSONS, LessonGate) and life.lesson_gate() is LESSONS
    # a word and a topic of their own: "life predict" and "predict_life" are the round-6 lessons of cosmos.predict
    assert life.SIM7 == "life7" != life.SIM
    assert LESSONS.sim == "life7" and LESSONS.topic == "predict_life7" and LESSONS.rules is life.RULES
    assert set(life.RULES) == set(FUNCTIONS) and len(life.RULES) == 10 >= 8
    for name, rule in life.RULES.items():
        assert rule.compute is FUNCTIONS[name], name
        assert is_dense(f"life7 predict {name}. {rule.description}."), name
    assert len(lessons) == 400 and {ln.prompt.split()[2] for ln in lessons} == set(life.RULES)
    for ln in lessons:
        assert is_dense(ln.text) and n_tokens(ln.text) <= MAX_TOKENS and ln.topic == "predict_life7" and ln.kind == "calc"
        assert "." not in ln.prompt and "." not in ln.answer
        assert LESSONS.owns(ln.prompt) and ln.prompt.startswith("life7 predict ")
        v = LESSONS.check(ln.prompt, ln.answer)
        assert v.ok and v.expected == ln.answer, (ln.text, v)
        assert life.check(ln.prompt, ln.answer).ok
        name, values = LESSONS.parse(ln.prompt)
        truth = FUNCTIONS[name](*values)  # what the simulation's own function returns
        assert ln.answer == (truth if isinstance(truth, str) else num(truth)), ln.text
        assert ln.meta["split_key"] == LESSONS.split_key(ln.prompt)
    assert [ln.text for ln in LESSONS.generate(random.Random(4), 50)] == [ln.text for ln in LESSONS.generate(random.Random(4), 50)]
    assert [ln.text for ln in LESSONS.generate(random.Random(4), 50)] != [ln.text for ln in LESSONS.generate(random.Random(5), 50)]
    assert LESSONS.generate(random.Random(1), 0) == []
    # every stage and outcome word is taught
    big = LESSONS.generate(random.Random(2), 3000)
    answers = {name: {ln.answer for ln in big if ln.prompt.split()[2] == name} for name in ("stage", "outcome")}
    assert answers == {"stage": set(life.STAGES7), "outcome": set(life.OUTCOMES7)}
    assert {ln.answer for ln in big if ln.prompt.split()[2] == "max_length"} >= {"0", "4", "8", "1 6"}
    records = LESSONS.records()
    assert len(records) == 10 and all(is_dense(x.text) and n_tokens(x.text) <= MAX_TOKENS and x.kind == "record" for x in records)
    assert all(x.text.startswith("life7 predict ") and x.topic == "predict_life7" for x in records)
    assert ("life7 predict copy_probability. probability that a template of length letters finishes a copy in one step. "
            "4 over length and never more than 1.") in [x.text for x in records]
    # the plain gate of the contract, with uniform draws, gives valid lessons of the same rules too
    for ln in LessonGate(life.SIM7, life.RULES).generate(random.Random(3), 200):
        assert LESSONS.check(ln.prompt, ln.answer).ok, ln.text


def test_lessons_share_no_prompt_and_no_topic_with_the_round_6_lessons(lessons):
    """``life predict ...`` (topic ``predict_life``) are the lessons of ``haishool.cosmos.predict`` and teach the round-6
    rules; a registry keyed by topic, or a prompt asked without its rule set, must never mix the two."""
    from haishool.cosmos import predict
    old = predict.gate("life")
    assert old.topic == "predict_life" != LESSONS.topic and old.sim == "life" != LESSONS.sim
    assert set(old.KEYS) & set(life.RULES) == {"next_total"}  # one name in both: it must not be one prompt
    old_lines = old.generate(random.Random(1), 200)
    for ln in old_lines:
        assert ln.prompt.startswith("life predict ") and not LESSONS.owns(ln.prompt) and not life.owns(ln.prompt), ln.text
        assert LESSONS.check(ln.prompt, ln.answer).reason == "not my question"
    for ln in lessons:
        assert not old.owns(ln.prompt) and not predict.owns(ln.prompt), ln.text
        assert not old.check(ln.prompt, ln.answer).ok
    assert not {ln.meta["split_key"] for ln in lessons} & {ln.meta["split_key"] for ln in old_lines}
    assert not {x.text for x in LESSONS.records()} & {x.text for x in old.records()}
    assert LESSONS.check("life7 predict next_total total 1 0 0 inflow 8", "1 0 8").ok
    assert not LESSONS.owns("life predict next_total total 1 0 0 inflow 8") and old.owns("life predict next_total total 1 0 0 inflow 8")


def test_lessons_reject_wrong_answers_and_foreign_prompts(lessons):
    words = set(life.STAGES7 + life.OUTCOMES7)
    for ln in lessons:
        x = parse_num(ln.answer)
        if x is None:
            wrongs = sorted(words - {ln.answer})[:3] + ["yes"]
        else:
            wrongs = [num(x * 2 + 7 if isinstance(x, int) else x * 2 + 1), num(x + 1), "growth"]
        for wrong in wrongs + ["", "1 e 9 9 9"]:
            v = LESSONS.check(ln.prompt, wrong)
            assert not v.ok and v.expected == ln.answer, (ln.text, wrong)
    for other in OTHER_PROMPTS + ("life seed 7 rules 7 final outcome", "life7 predict nope length 4", "life7 predict copy_probability",
                                  "life7 predict copy_probability length 1 7", "life7 predict copy_probability length 4 point 5",
                                  "life7 predict inflow_per_kind inflow 1 0", "life7 predict next_total total 1 0 0 inflow 6",
                                  "life7 predict stage replicators 3 previous 2 failed 4 seen_before 1",
                                  "life7 predict stage replicators 3 previous 2 failed 0 seen_before 0",
                                  "life7 predict outcome peak 3 final 4 effective_lineages 1",
                                  "life7 predict outcome peak 9 final 0 effective_lineages 2",
                                  "life7 predict outcome peak 9 final 3 effective_lineages 4",
                                  "life7 predict copy_probability length 4 length 4", "life7 predict copy_probability length",
                                  "life7 predict intact_probability hydrolysis 0 point 1 length 4",
                                  "life7 predict intact_probability length 4 hydrolysis 0 point 3",
                                  "life7 predict intact_probability length 4 hydrolysis 1 e 9 9 9",
                                  "life7 predict survival_probability length 4 decay 1",
                                  "life predict copy_probability length 4", "life predict survival_probability length 4 decay 1",
                                  "life predict expected_mutations length 8 mu 0 point 0 2",
                                  "life predict next_total total 1 0 0 inflow 8",
                                  "life rules 7 predict copy_probability length 4", "cells predict copy_probability length 4"):
        assert not LESSONS.owns(other), other
        assert life.owns(other) == (other == "life seed 7 rules 7 final outcome"), other  # a story, no lesson
        assert LESSONS.check(other, "1").reason == "not my question"
    assert LESSONS.check("life7 predict copy_probability length 8", "0 point 5").ok
    assert LESSONS.check("life7 predict max_length hydrolysis 0 point 0 2 mu 0 point 0 0 1", "1 3").ok
    assert not LESSONS.check("life7 predict max_length hydrolysis 0 point 0 2 mu 0 point 0 0 1", "1 4").ok
    assert LESSONS.check("life7 predict outcome peak 0 final 0 effective_lineages 0", "none").ok
    assert LESSONS.check("life7 predict stage replicators 0 previous 0 failed 0 seen_before 0", "soup").ok
    assert LESSONS.check("life7 predict intact_probability length 6 hydrolysis 0 point 0 5 7", "0 point 7 4 5 7").ok  # the docstring's
    # the rule as written, not as floating point cuts it: 0.043 * 20000 * 20000 / 10000 is 1720
    assert LESSONS.check("life7 predict polymer_letters free 2 0 0 0 0 poly_rate 0 point 0 4 3", "1 7 2 0").ok
    assert not LESSONS.check("life7 predict polymer_letters free 2 0 0 0 0 poly_rate 0 point 0 4 3", "1 7 1 9").ok


def test_lessons_are_true_of_the_rollouts(forty):
    """What a lesson says about a record of a run is what the run recorded."""
    words = Counter()
    for seed, r in forty.items():
        p = r.params
        seen = 0
        for t, s in enumerate(r.steps):
            prev = r.steps[t - 1]["replicators"] if t else 0
            ln = LESSONS.line("stage", [s["replicators"], prev, s["failed"], seen])
            assert ln is not None and ln.answer == s["stage"], (seed, t)
            words[s["stage"]] += 1
            seen = seen or int(s["replicators"] > 0)
            if t:
                total = LESSONS.line("next_total", [r.steps[t - 1]["free_monomers"] + r.steps[t - 1]["spent"] + r.steps[t - 1]["bound"], p["inflow"]])
                assert parse_num(total.answer) == s["free_monomers"] + s["spent"] + s["bound"]
        out = LESSONS.line("outcome", [r.summary["peak_replicators"], r.summary["replicators"], r.summary["effective_lineages"]])
        assert out is not None and out.answer == r.summary["outcome"], seed
        top = LESSONS.line("max_length", [p["hydrolysis"], p["mu"]])
        assert parse_num(top.answer) == r.summary["max_length"]
        # step 1: nothing has broken yet, so every bound letter was polymerised or copied; the strings take
        # at most polymer_letters of the pool after the inflow, and all of it but less than one string
        first = r.steps[1]
        want = life.polymer_letters(p["monomers"] + 4 * life.inflow_per_kind(p["inflow"]), p["poly_rate"])
        if first["copies"] == 0:
            assert want - life.MAX_LEN < first["bound"] <= want, seed
        assert first["spent"] == 0
    assert set(words) == set(life.STAGES7)


# ---------------------------------------------------------------------------------------------
# plausibility: what the review asked the corrected vessel to show, on 40 seeds each
# ---------------------------------------------------------------------------------------------

def test_stages_and_summary_follow_the_steps(forty):
    seen_stages, outcomes = set(), Counter()
    for r in forty.values():
        reps = [s["replicators"] for s in r.steps]
        first = next((t for t, n in enumerate(reps) if n), -1)
        assert r.summary["first_replicator_step"] == (first if first >= 0 else "never")
        assert r.summary["peak_replicators"] == max(reps)
        last = r.steps[-1]
        for key in ("replicators", "diversity", "lineages", "dominant_share", "effective_lineages", "replicator_length", "origins"):
            assert r.summary[key] == last[key], key
        assert r.summary["max_length"] == life.max_length(r.params["hydrolysis"], r.params["mu"])
        outcomes[r.summary["outcome"]] += 1
        peak, final = max(reps), reps[-1]
        assert (r.summary["outcome"] == "none") == (first < 0)
        assert (r.summary["outcome"] == "fizzled") == (first >= 0 and final == 0 and peak < life.FIZZLE_PEAK)
        assert (r.summary["outcome"] == "extinct") == (final == 0 and peak >= life.FIZZLE_PEAK)
        assert (r.summary["outcome"] == "declining") == (final > 0 and 2 * final <= peak)
        assert (r.summary["outcome"] == "dominated") == (2 * final > peak and last["effective_lineages"] < 2)
        assert (r.summary["outcome"] == "coexist") == (2 * final > peak and last["effective_lineages"] >= 2)
        for t, s in enumerate(r.steps):
            seen_stages.add(s["stage"])
            assert s["step"] == t and s["stage"] in life.STAGES7
            assert s["diversity"] <= s["replicators"] <= s["polymers"] and s["lineages"] <= min(s["replicators"], s["origins"])
            assert (s["stage"] == "soup") == (t < first or first < 0)
            assert (s["stage"] == "first_replicator") == (t == first)
            assert (s["stage"] == "extinct") == (first >= 0 and t > first and s["replicators"] == 0)
            # the stage can be read off the record: failed copies are written down
            if first >= 0 and t > first and s["replicators"]:
                assert (s["stage"] == "competition") == (s["failed"] > 0)
                assert (s["stage"] == "decline") == (s["failed"] == 0 and s["replicators"] < r.steps[t - 1]["replicators"])
            if s["replicators"]:
                assert 4 <= s["replicator_length"] <= s["longest"] <= life.MAX_LEN
                assert 1 <= s["effective_lineages"] <= s["lineages"] and 1 / s["lineages"] <= s["dominant_share"] * 1.01 <= 1.01
                assert (s["lineages"] == 1) == (s["dominant_share"] == 1.0)
            else:
                assert s["replicator_length"] == s["dominant_share"] == s["effective_lineages"] == s["lineages"] == 0
            if t:
                made = s["copies"] - r.steps[t - 1]["copies"]
                assert 0 <= made and made + s["failed"] <= s["replicators"]
    assert seen_stages == set(life.STAGES7)
    # measured on seeds 1 to 40: dominated 19, declining 16, coexist 2, extinct 2, fizzled 1
    assert set(outcomes) >= {"dominated", "declining", "coexist", "extinct"} and set(outcomes) <= set(life.OUTCOMES7)
    assert outcomes["dominated"] >= 10 and outcomes["declining"] >= 8 and outcomes["extinct"] + outcomes["fizzled"] >= 1


def test_a_first_replicator_is_a_chance_event_and_origins_repeat(forty):
    firsts = [r.summary["first_replicator_step"] for r in forty.values()]
    had = [t for t in firsts if t != "never"]
    assert len(had) >= 34 and len(set(had)) >= 8  # measured 40 of 40, at steps 1 to 44
    assert 1 <= min(had) and 2 <= statistics.median(had) <= 12  # measured median 4.5
    origins = [r.summary["origins"] for r in forty.values()]
    assert sum(n > 1 for n in origins) >= 30 and 2 <= statistics.median(origins) <= 10  # measured 38 of 40, median 4
    # founders are long, as the strings of the soup that hold a motif are: measured median 8.2 letters
    founders = [r.steps[r.summary["first_replicator_step"]]["replicator_length"] for r in forty.values()
                if r.summary["first_replicator_step"] != "never"]
    assert 6.5 <= statistics.median(founders) <= 10


def test_the_odds_that_a_string_of_the_soup_carries_the_motif():
    """The docstring's odds, summed exactly over the lengths (letters equally common): an automaton that remembers the
    last ``3 + gap`` letters, checked against ``has_motif`` on every string of 4 to 7 letters."""
    def p_hit(length: int, gap: int) -> float:
        a, c, g, u = (life.LETTERS.index(ch) for ch in "acgu")
        states = {(): 1.0}  # the letters so far (the last 3 + gap of them) of strings without the motif -> probability
        for _ in range(length):
            nxt: dict[tuple, float] = {}
            for h, pr in states.items():
                for x in range(4):
                    if x == u and len(h) >= 3 and h[-1] == c and any(
                            len(h) >= 3 + k and h[-2 - k] == g and h[-3 - k] == a for k in range(gap + 1)):
                        continue
                    key = (h + (x,))[-(3 + gap):]
                    nxt[key] = nxt.get(key, 0.0) + pr / 4
            states = nxt
        return 1.0 - sum(states.values())

    for n in (4, 5, 6, 7):  # the automaton against the level's own test, on every string
        codes = np.arange(4 ** n)
        rows = np.full((4 ** n, life.MAX_LEN), life.PAD, dtype=np.uint8)
        for i in range(n):
            rows[:, i] = (codes >> (2 * i)) & 3
        for gap in range(4):
            assert life.has_motif(rows, gap).mean() == pytest.approx(p_hit(n, gap), rel=1e-12), (n, gap)
    assert p_hit(4, 0) == pytest.approx(1 / 256) and p_hit(4, 3) == pytest.approx(1 / 256)
    # lengths as polymerisation draws them: 2 with probability 0.3, then 0.7 times as likely per letter, the rest at 16
    share, weights = 1.0, {}
    for n in range(2, life.MAX_LEN):
        weights[n], share = 0.3 * share, 0.7 * share
    weights[life.MAX_LEN] = share
    assert sum(weights.values()) == pytest.approx(1.0) and len(weights) == 15
    assert weights[2] == pytest.approx(life._LENGTH_CDF[0]) and 1 - weights[16] == pytest.approx(life._LENGTH_CDF[-1])
    one_in, mean_length = [], []
    for gap in range(4):
        hit = {n: w * p_hit(n, gap) for n, w in weights.items()}
        one_in.append(round(1 / sum(hit.values()), 1))
        mean_length.append(round(sum(n * x for n, x in hit.items()) / sum(hit.values()), 2))
    assert one_in == [158.7, 93.7, 75.0, 67.0] and mean_length == [8.4, 8.77, 9.05, 9.26]
    # and the vessel agrees: origins per string made in step 1 over 300 seeds with gap 0 (expected one in 158.7)
    strings = origins = 0
    for seed in range(1, 301):
        p = life._params7(seed, {"gap": 0, "hydrolysis": 0.0, "inflow": 0, "monomers": 20000, "poly_rate": 0.02})
        vessels = life.simulate7(seed, p)
        next(vessels)
        v = next(vessels)
        made = v.copies  # copies of step 1 are in the vessel too; nothing has broken yet
        strings, origins = strings + len(v.seqs) - made, origins + v.origins
    assert strings > 40000 and 120 <= strings / origins <= 210, (strings, origins)


def test_no_perpetual_motion_a_closed_vessel_runs_out():
    params = {"inflow": 0, "monomers": 14361, "poly_rate": 0.01, "gap": 1}
    mild = [life.run(seed, rules=7, hydrolysis=0.01, **params) for seed in OTHER_FORTY]
    harsh = [life.run(seed, rules=7, hydrolysis=0.05, **params) for seed in OTHER_FORTY]
    # measured: no copy at all in the last ten steps (71 to 80) in 73 of the 80 runs
    last_ten = [r.steps[-1]["copies"] - r.steps[-11]["copies"] for r in mild + harsh]
    assert statistics.median(last_ten) == 0 and sum(n == 0 for n in last_ten) >= 65
    for r in mild + harsh:
        last = r.steps[-1]
        assert 4 * last["copies"] <= 14361  # every copy took at least 4 activated letters, and none came back
        assert last["free_monomers"] + last["spent"] + last["bound"] == 14361
        assert last["copies"] - r.steps[-11]["copies"] <= 10 < 1000 < last["copies"], r.seed  # measured at most 3 of 1338 or more
        assert r.summary["peak_replicators"] >= 500  # there was a boom (measured median 1680)
    # hydrolysis 0.01: measured median 1547.5, 817.5, 441.5, 233.5 replicators at steps 20, 40, 60, 80; all 40 declining
    medians = [statistics.median(r.steps[t]["replicators"] for r in mild) for t in (20, 40, 60, 80)]
    assert medians[0] > 1000 and medians[0] > 1.5 * medians[1] > 2 * medians[2] > 3 * medians[3] > 0
    assert sum(r.summary["outcome"] == "declining" for r in mild) >= 36
    # hydrolysis 0.05: measured 1129, 59, 2, 0; extinct 36 of 40, the other 4 declining
    medians = [statistics.median(r.steps[t]["replicators"] for r in harsh) for t in (20, 40, 60, 80)]
    assert medians[0] > 700 and medians[1] < 0.2 * medians[0] and medians[2] <= 10 and medians[3] == 0
    assert sum(r.summary["outcome"] == "extinct" for r in harsh) >= 28
    assert all(20 * r.summary["replicators"] <= r.summary["peak_replicators"] for r in harsh)
    # the same vessel under round 6 recycles for nothing and copies for ever: measured median 1696.5 copies in
    # the last ten steps, and more copies than the letters of the vessel could ever pay for in 30 of 40 runs
    old = [life.run(seed, decay=1.6, **params) for seed in OTHER_FORTY]
    assert statistics.median(r.steps[-1]["generations"] - r.steps[-11]["generations"] for r in old) > 500
    assert sum(4 * r.steps[-1]["generations"] > 14361 for r in old) >= 20


def test_replicators_get_shorter_and_none_outlasts_the_length_limit(forty):
    alive = [r for r in forty.values() if r.summary["replicators"]]
    shorter = [r for r in alive if r.summary["replicator_length"] < r.steps[r.summary["first_replicator_step"]]["replicator_length"]]
    assert len(alive) >= 30 and len(shorter) >= 0.7 * len(alive)  # measured 32 of 37
    # the faster the bonds break, the shorter the replicators that are left (inflow 200, 40 seeds each)
    slow = [life.run(seed, rules=7, inflow=200, hydrolysis=0.002) for seed in OTHER_FORTY]
    fast = [life.run(seed, rules=7, inflow=200, hydrolysis=0.05) for seed in OTHER_FORTY]
    slow_len = statistics.median(r.summary["replicator_length"] for r in slow if r.summary["replicators"])
    fast_len = statistics.median(r.summary["replicator_length"] for r in fast if r.summary["replicators"])
    assert 5 <= slow_len <= 7 and 4 <= fast_len <= 4.3 and slow_len > fast_len + 1  # measured 5.5 and 4.0
    for r in fast:  # max_length is 7 or 8 at hydrolysis 0.05
        assert r.summary["max_length"] in (7, 8)
        if r.summary["replicators"]:
            assert r.summary["replicator_length"] <= r.summary["max_length"]
    # the share of the final replicators longer than max_length: measured 0 in every one of these runs
    over = []
    for seed in OTHER_FORTY:
        p = life._params7(seed, {"inflow": 200, "hydrolysis": 0.1})
        *_, v = life.simulate7(seed, p)
        rep = v.lineage >= 0
        if rep.any():
            over.append(float((lengths(v.seqs[rep]) > life.max_length(0.1, p["mu"])).mean()))
    assert len(over) >= 30 and statistics.median(over) == 0 and max(over) <= 0.1  # measured median 0, largest 0.026


def test_growth_factor_predicts_the_growth_while_letters_are_plentiful():
    ratios = []
    for seed in FORTY:
        p = life._params7(seed, {})
        vessels = list(life.simulate7(seed, p))
        for a, b in zip(vessels, vessels[1:]):
            rep = a.lineage >= 0
            if rep.sum() >= 10 and b.failed == 0:
                predicted = statistics.fmean(life.growth_factor(int(n), p["hydrolysis"], p["mu"]) for n in lengths(a.seqs[rep]))
                ratios.append(int((b.lineage >= 0).sum()) / int(rep.sum()) / predicted)
    # measured: 454 steps, median 1.033, quartiles 1.004 and 1.081 (pieces that keep the motif and new origins are the surplus)
    assert len(ratios) >= 200 and 0.98 <= statistics.median(ratios) <= 1.1
    low, _, high = statistics.quantiles(ratios, n=4)
    assert 0.9 <= low and high <= 1.2


def test_dominated_no_longer_measures_the_mutation_rate():
    def count(word, runs):
        return sum(r.summary["outcome"] == word for r in runs)

    low7 = [life.run(seed, rules=7, inflow=200, hydrolysis=0.005, mu=0.001) for seed in OTHER_FORTY]
    high7 = [life.run(seed, rules=7, inflow=200, hydrolysis=0.005, mu=0.05) for seed in OTHER_FORTY]
    low6 = [life.run(seed, inflow=200, mu=0.001) for seed in OTHER_FORTY]
    high6 = [life.run(seed, inflow=200, mu=0.05) for seed in OTHER_FORTY]
    assert count("dominated", low6) - count("dominated", high6) >= 20  # round 6, measured 33 against 1
    assert abs(count("dominated", low7) - count("dominated", high7)) <= 8  # rules 7, measured 26 against 26
    assert count("dominated", low7) >= 15 and count("coexist", low7) >= 5  # measured 26 and 14
    # mu still does what it should: more distinct sequences (measured median 19 against 57)
    assert statistics.median(r.summary["diversity"] for r in high7) > 1.5 * statistics.median(r.summary["diversity"] for r in low7)


def test_a_run_is_fast_and_the_cap_stops_new_strings():
    t0 = time.perf_counter()
    for seed in range(1, 11):
        life.run(seed, rules=7)
    assert (time.perf_counter() - t0) / 10 < 1.0  # measured 0.1 s per rollout
    for params in ({"monomers": 20000, "inflow": 400, "poly_rate": 0.02, "hydrolysis": 0.002, "mu": 0.05, "gap": 3},
                   {"monomers": 200000, "inflow": 0, "poly_rate": 0.02, "hydrolysis": 0.002, "mu": 0.01, "gap": 2},
                   {"monomers": 200000, "inflow": 400, "poly_rate": 1.0, "hydrolysis": 0.1, "mu": 0.01, "gap": 2}):
        t0 = time.perf_counter()
        r = life.run(5, rules=7, **params)
        assert time.perf_counter() - t0 < 2, params  # measured 0.6 s at most
        assert SIM.conserved(r).ok
    # 200000 letters would make 46000 strings; the cap stops new strings and copies, only pieces go beyond it
    assert life.MAX_POLYMERS <= max(s["polymers"] for s in r.steps) < 1.5 * life.MAX_POLYMERS
    assert max(s["polymers"] for r in (life.run(seed, rules=7) for seed in range(1, 11)) for s in r.steps) < life.MAX_POLYMERS
