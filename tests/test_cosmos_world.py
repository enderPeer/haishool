import copy
import decimal
import math
import random
import time
from fractions import Fraction

import numpy as np
import pytest

from haishool.cosmos import Rollout, Simulation, chem, close, gravity, life, nucleo, planets
from haishool.cosmos.world import (BLACK_BODY, CONDITIONS, CONSTANTS, DEFAULTS, ERA_KEYS, ERAS, EXACT_KEYS,
                                   FIXED_ERAS, HANDOFFS, KEYS, MEASURED_KEYS, MISSING, NOTES, OUTCOMES, PARAM_KEYS,
                                   SUMMARY_KEYS, YIELDS, World, WorldRollout, canonical, check, conditions,
                                   conserved, cooling_of, dense_value, disc_mass_of, enrich, era_line, fusion_of,
                                   frost_line_of, generate, habitable_bodies, handoff_line, handoff_lines, lines,
                                   luminosity_of, metallicity_of, metallicity_printed, mix_of, monomers_of,
                                   organic_molecules, outcome_of, owns, params_line, parse, practice_lines,
                                   random_params, records, rollout, run,
                                   simulation, star_mass_of, summarise, table_lines, table_value, temperature_of,
                                   timeline_line, truth)
from haishool.truth import Gate, is_dense, num, parse_num, split_line

SEEDS = list(range(1, 41))


def n_tokens(text: str) -> int:
    """As ``haishool.student.tokens`` counts them (that module needs torch, so not imported)."""
    return len(text.replace(".", " . ").split())


def sig3(x: float) -> float:
    return float(f"{x:.3g}")


def half_up(x: Fraction, sig: int = 3) -> float:
    """``x`` to ``sig`` significant digits, a 5 rounding up, in whole-number arithmetic."""
    if x == 0:
        return 0.0
    exp = 0
    while x >= Fraction(10) ** (exp + 1):
        exp += 1
    while x < Fraction(10) ** exp:
        exp -= 1
    unit = Fraction(10) ** (exp - sig + 1)
    return float(math.floor(x / unit + Fraction(1, 2)) * unit)


def exact(x: float) -> Fraction:
    """A printed number as the fraction its digits spell."""
    return Fraction(repr(x))


def wrong(answer: str) -> str:
    """An answer that is clearly not ``answer``: another word, or a number far off."""
    value = parse_num(answer)
    if value is None:
        return {"yes": "no", "no": "yes"}.get(answer, " ".join([*answer.split()[:-1], "wrong"]))
    return num(abs(value) * 3 + 7)


def bump(answer: str) -> str:
    """A number with one digit changed: its leading digit (the first that is not 0, else the
    first) moved by 2, which is a fifth off at least and never within 1 of a count."""
    words = answer.split()
    at = next((i for i, w in enumerate(words) if w in "123456789"), None)
    if at is None:
        at = next(i for i, w in enumerate(words) if w == "0")
    digit = int(words[at])
    words[at] = str(digit + 2 if digit < 8 else digit - 2)
    return " ".join(words)


@pytest.fixture(scope="module")
def rollouts():
    return {seed: rollout(seed) for seed in SEEDS}


@pytest.fixture(scope="module")
def all_lines(rollouts):
    return [ln for r in rollouts.values() for ln in lines(r)]


@pytest.fixture(scope="module")
def alive(rollouts):
    """A world that reaches life: it has every kind of era."""
    return next(r for r in rollouts.values() if r.summary["planets_with_life"])


def test_protocols():
    s = simulation()
    assert isinstance(s, Simulation) and isinstance(s, Gate) and isinstance(s, World)
    assert s.sim == s.topic == "world" and simulation() is s
    for key in {*PARAM_KEYS, *SUMMARY_KEYS, *HANDOFFS, *(k for keys in ERA_KEYS.values() for k in keys)}:
        assert key in KEYS, key
    assert EXACT_KEYS | MEASURED_KEYS <= set(KEYS)
    assert set(ERAS) == set(ERA_KEYS) and all(set(ERAS[e]) <= set(ERA_KEYS[e]) for e in ERAS)
    assert set(random_params(random.Random(1))) == set(PARAM_KEYS)
    assert all(row["notes"] for row in CONSTANTS.values()) and NOTES["yields"]


def test_random_params_stay_in_their_ranges():
    for seed in range(300):
        p = random_params(random.Random(seed))
        assert p["generations"] in (0, 1, 2, 3) and 0.05 <= p["spin"] <= 0.4 and 1.0 <= p["t_gas"] <= 10.0
        assert p == random_params(random.Random(seed))
    assert {random_params(random.Random(s))["generations"] for s in range(40)} == {0, 1, 2, 3}


def test_same_seed_same_world(rollouts):
    for seed in (3, 6, 23, 39):
        fresh = run(seed)  # not from the cache
        assert fresh.steps == rollouts[seed].steps and fresh.summary == rollouts[seed].summary
        assert fresh.params == rollouts[seed].params and canonical(fresh)
        assert fresh.primordial == rollouts[seed].primordial and fresh.cloud == rollouts[seed].cloud
        assert list(fresh.levels) == list(rollouts[seed].levels)
        for name, level in fresh.levels.items():  # every state of every level, number by number
            cached = rollouts[seed].levels[name]
            assert level.steps == cached.steps, (seed, name)
            assert level.summary == cached.summary and level.params == cached.params, (seed, name)
        assert [ln.text for ln in lines(fresh)] == [ln.text for ln in lines(rollouts[seed])]
    assert rollouts[3].steps != rollouts[4].steps


def test_known_parameters():
    """A canary for ``random.Random``: the three draws of a seed, as they were when the lines were written."""
    assert random_params(random.Random(1)) == {"generations": 1, "spin": 0.25, "t_gas": 8.22}
    assert random_params(random.Random(2)) == {"generations": 0, "spin": 0.08, "t_gas": 4.25}
    assert random_params(random.Random(3)) == {"generations": 1, "spin": 0.26, "t_gas": 2.17}
    assert random_params(random.Random(9998)) == {"generations": 0, "spin": 0.35, "t_gas": 8.25}


def test_values_are_plain_python(rollouts):
    """Counts are ``int``, measured numbers ``float``, words ``str``. A numpy scalar would be
    written by another rule: an ``int64`` count is no ``int`` and would be cut to 3 digits."""
    for r in rollouts.values():
        for holder in (r.params, r.summary, r.primordial, r.cloud, *r.steps):
            for key, value in holder.items():
                assert type(value) in (int, float, str), (r.seed, key, type(value))
        for s in [*r.steps, r.summary]:
            for key in (EXACT_KEYS | MEASURED_KEYS | {"temperature"}) & set(s):
                assert type(s[key]) is int, (r.seed, key, s[key])


def test_known_world(rollouts):
    """A canary: the chain of seed 3 as this code gave it when the lines were written. Level 2 is
    chaotic, so a machine or a library version that computes anything else would judge the lines
    wrongly; if this fails on the training machine, the lines must be rebuilt there."""
    r = rollouts[3]
    assert r.params == {"generations": 1, "spin": 0.26, "t_gas": 2.17, "particles": 256, "atoms": 20000,
                        "cloud_mass": 1.5}
    assert r.summary == {"hydrogen": 0.643, "helium": 0.35, "metallicity": 0.00663, "clumps": 2, "planets": 7,
                         "habitable": 1, "planets_with_water": 1, "planets_with_life": 1, "eras": 8,
                         "outcome": "life"}
    assert r.era("disc") == {"era": "disc", "flattening": 0.163, "spiral": "no", "clumps": 2, "largest_clump": 0.59,
                             "collapse_time": 1.3}
    assert r.era("chemistry_1") == {"era": "chemistry_1", "orbit": 0.883, "mass": 1.36, "temperature": 299,
                                    "mix": "ocean", "water": 5536, "organic": 20}
    assert r.era("life_1") == {"era": "life_1", "monomers": 14000, "first_replicator_step": 7,
                               "peak_replicators": 2708, "replicators": 2708, "diversity": 17, "outcome": "coexist"}
    texts = [ln.text for ln in lines(r)]
    for text in (
        "world seed 3 params. generations 1. spin 0 point 2 6. t_gas 2 point 1 7.",
        "world seed 3 timeline. eras nucleo enrichment cloud disc star planets chemistry_1 life_1. outcome life.",
        "world seed 3 era nucleo. hydrogen 0 point 6 6 3. helium 0 point 3 3 6. traces 1 point 2 4 e minus 4.",
        "world seed 3 era enrichment. generations 1. hydrogen 0 point 6 4 3. helium 0 point 3 5. "
        "metallicity 0 point 0 0 6 6 3. carbon 0 point 0 0 1 1 9. oxygen 0 point 0 0 2 9 2. iron 6 point 6 3 e minus 4.",
        "world seed 3 era cloud. spin 0 point 2 6. cooling 0 point 2 3.",
        "world seed 3 era star. fusion yes. star_mass 0 point 8 8 5. luminosity 0 point 6 5 2. frost_line 2 point 1 8. "
        "hz_inner 0 point 7 6 7. hz_outer 1 point 3 7. disc_mass 6 6 point 3. t_gas 2 point 1 7.",
        "world seed 3 era planets. planets 7. rocky 3. ice 4. gas 0. habitable 1. largest_mass 2 1 point 3.",
        "q world seed 3 era planets habitable. a 1.",
        "q world seed 3 era enrichment silicon. a 3 point 9 8 e minus 4.",
        "q world seed 3 final planets_with_life. a 1.",
        "world seed 3 era life_1. monomers 1 4 0 0 0. first_replicator_step 7. peak_replicators 2 7 0 8. "
        "replicators 2 7 0 8. diversity 1 7. outcome coexist.",
        "q world seed 3 why life. a metals star planets habitable water organic replicators survivors.",
        "q world handoff temperature star_mass 0 point 8 8 5 orbit 0 point 8 8 3. a 2 9 9.",
        "q world handoff monomers organic 2 0. a 1 4 0 0 0.",
    ):
        assert text in texts, text
    assert len(texts) == 82 and max(n_tokens(t) for t in texts) == 58


def test_known_frozen_world(rollouts):
    """A second canary, down the other branch: seed 1 has its one planet at the cold edge of the
    zone, so level 4 runs the rocky mix and there is no life era. Seed 23 has a star on a rounding
    tie (1.5 * 0.613 = 0.9195) and a dominated vessel."""
    r = rollouts[1]
    assert r.params == {"generations": 1, "spin": 0.25, "t_gas": 8.22, "particles": 256, "atoms": 20000,
                        "cloud_mass": 1.5}
    assert r.steps == [
        {"era": "nucleo", "hydrogen": 0.798, "helium": 0.202, "traces": 0.000208},
        {"era": "enrichment", "generations": 1, "hydrogen": 0.774, "helium": 0.218, "metallicity": 0.00798,
         "oxygen": 0.00351, "carbon": 0.00144, "neon": 0.000798, "iron": 0.000798, "nitrogen": 0.000479,
         "magnesium": 0.000479, "silicon": 0.000479},
        {"era": "cloud", "spin": 0.25, "cooling": 0.26},
        {"era": "disc", "flattening": 0.14, "spiral": "yes", "clumps": 4, "largest_clump": 0.59, "collapse_time": 1.3},
        {"era": "star", "fusion": "yes", "star_mass": 0.885, "luminosity": 0.652, "frost_line": 2.18, "hz_inner": 0.767,
         "hz_outer": 1.37, "disc_mass": 79.8, "t_gas": 8.22},
        {"era": "planets", "planets": 7, "rocky": 3, "ice": 4, "gas": 0, "habitable": 1, "largest_mass": 34.2},
        {"era": "chemistry_1", "orbit": 1.36, "mass": 3.26, "temperature": 247, "mix": "rocky", "water": 45,
         "organic": 0},
    ]
    assert r.summary["outcome"] == "no_water" and len(lines(r)) == 74
    r = rollouts[23]
    assert r.era("disc") == {"era": "disc", "flattening": 0.0345, "spiral": "yes", "clumps": 4, "largest_clump": 0.613,
                             "collapse_time": 1.8}
    assert r.era("star") == {"era": "star", "fusion": "yes", "star_mass": 0.92, "luminosity": 0.747, "frost_line": 2.33,
                             "hz_inner": 0.821, "hz_outer": 1.47, "disc_mass": 165.0, "t_gas": 1.75}
    assert r.era("planets") == {"era": "planets", "planets": 7, "rocky": 3, "ice": 4, "gas": 0, "habitable": 1,
                                "largest_mass": 53.2}
    assert r.era("chemistry_1") == {"era": "chemistry_1", "orbit": 1.13, "mass": 6.36, "temperature": 276,
                                    "mix": "ocean", "water": 5533, "organic": 17}
    assert r.era("life_1") == {"era": "life_1", "monomers": 11900, "first_replicator_step": 14,
                               "peak_replicators": 2146, "replicators": 2146, "diversity": 20, "outcome": "dominated"}


def test_rollout_hands_out_copies(rollouts):
    mine = rollout(3)
    mine.steps[0]["hydrogen"] = 9.0
    mine.levels["gravity"].steps[0]["mass"] = 9.0
    assert rollout(3).steps == rollouts[3].steps and conserved(rollout(3)).ok
    assert not conserved(mine).ok


def test_a_world_runs_in_under_six_seconds():
    start = time.perf_counter()
    r = run(SEEDS[-1] + 1)
    assert time.perf_counter() - start < 6.0
    assert conserved(r).ok


def test_every_line_is_dense_and_short(all_lines):
    everything = all_lines + records() + table_lines() + practice_lines(random.Random(2), 400)
    assert len(all_lines) > 60 * len(SEEDS)
    for ln in everything:
        assert is_dense(ln.text), ln.text
        assert n_tokens(ln.text) <= 80, ln.text
        assert ln.topic == "world"
        if ln.kind == "record":
            assert split_line(ln.text) is None and ln.answer == ""
        else:
            assert "." not in ln.prompt and "." not in ln.answer
            assert split_line(ln.text) == (ln.prompt, ln.answer)
    kinds = {ln.kind for ln in everything}
    assert kinds == {"record", "fact", "calc"}


def test_records_fit_with_the_largest_seed(rollouts):
    """A four-digit seed and two-digit planet numbers still leave room in the context."""
    for r in rollouts.values():
        big = copy.deepcopy(r)
        big.seed = 9998
        for ln in [params_line(big), timeline_line(big), *(era_line(big, t) for t in range(len(big.steps)))]:
            assert n_tokens(ln.text) <= 70, ln.text


def test_lines_of_one_world(rollouts, alive):
    ls = lines(alive)
    texts = [ln.text for ln in ls]
    seed = num(alive.seed)
    assert texts[0].startswith(f"world seed {seed} params. generations ")
    assert texts[1].startswith(f"world seed {seed} timeline. eras nucleo enrichment cloud disc star planets chemistry_1")
    names = [s["era"] for s in alive.steps]
    assert [t.split(".")[0] for t in texts[2:2 + len(names)]] == [f"world seed {seed} era {n}" for n in names]
    assert sum(ln.kind == "record" for ln in ls) == 2 + len(names)
    prompts = [ln.prompt for ln in ls if ln.kind != "record"]
    assert len(prompts) == len(set(prompts))
    for name in names:
        for key in ERA_KEYS[name.split("_")[0] if name[-1].isdigit() else name]:
            assert f"world seed {seed} era {name} {key}" in prompts
    for key in SUMMARY_KEYS:
        assert f"world seed {seed} final {key}" in prompts
    for key in PARAM_KEYS:
        assert f"world seed {seed} param {key}" in prompts
    assert f"world seed {seed} timeline" in prompts
    assert f"world seed {seed} why {alive.summary['outcome']}" in prompts
    assert sum(p.startswith("world handoff ") for p in prompts) == len(handoff_lines(alive))
    # the era record shows what the era questions answer
    record = next(t for t in texts if t.startswith(f"world seed {seed} era planets."))
    for key in ERAS["planets"]:
        assert f" {key} {dense_value(alive.era('planets')[key])}." in record


def test_gate_agrees_with_every_line(all_lines):
    asked = 0
    for ln in all_lines + table_lines() + practice_lines(random.Random(3), 400):
        if ln.kind == "record":
            continue
        asked += 1
        verdict = check(ln.prompt, ln.answer)
        assert owns(ln.prompt), ln.prompt
        assert verdict.ok and verdict.expected == ln.answer, (ln.text, verdict)
        assert truth(ln.prompt) is not None
    assert asked > 1500


def test_gate_rejects_wrong_answers(all_lines):
    for ln in all_lines + table_lines() + practice_lines(random.Random(4), 200):
        if ln.kind == "record":
            continue
        bad = wrong(ln.answer)
        assert bad != ln.answer
        verdict = check(ln.prompt, bad)
        assert not verdict.ok and verdict.expected == ln.answer, (ln.text, bad, verdict)
        for junk in ("", "banana", "1 e 9 9 9", "point", " ".join("9" * 400)):
            assert not check(ln.prompt, junk).ok, (ln.text, junk)


def test_gate_rejects_a_changed_digit_or_word(all_lines):
    """One digit of a number changed, a word swapped for another the lines use, a word dropped
    from a list or the list turned round: none passes."""
    assert bump("0 point 0 0 6 6 3") == "0 point 0 0 8 6 3" and bump("9 6 0 0") == "7 6 0 0" and bump("0") == "2"
    assert bump("minus 1") == "minus 3" and bump("6 point 6 3 e minus 4") == "8 point 6 3 e minus 4"
    words = sorted({ln.answer for ln in all_lines if ln.kind != "record" and parse_num(ln.answer) is None
                    and " " not in ln.answer})
    assert {"yes", "no", "ocean", "rocky", "no_metals", "no_water", "coexist", "dominated"} <= set(words)
    words.append("never")  # the collapse time of a cloud that does not collapse: 2 worlds in 300, none here
    numbers = lists = single = 0
    for ln in all_lines + table_lines() + practice_lines(random.Random(9), 300):
        if ln.kind == "record":
            continue
        if parse_num(ln.answer) is not None:
            changed = [bump(ln.answer), "never", "yes"]
            numbers += 1
        elif " " in ln.answer:
            changed = [ln.answer.rsplit(" ", 1)[0], ln.answer.split(" ", 1)[1], " ".join(reversed(ln.answer.split()))]
            lists += 1
        else:
            changed = [w for w in words if w != ln.answer] + ["1", "0"]
            single += 1
        for bad in changed:
            verdict = check(ln.prompt, bad)
            assert bad != ln.answer and not verdict.ok and verdict.expected == ln.answer, (ln.text, bad, verdict)
    assert numbers > 2000 and lists > 40 and single > 250


def test_tolerances(rollouts, alive):
    seed = num(alive.seed)
    star, system = alive.era("star"), alive.era("planets")
    surface = next(s for s in alive.numbered("chemistry") if s["mix"] == "ocean")
    wet = surface["era"]
    # measured numbers: 5 percent
    prompt = f"world seed {seed} era star star_mass"
    assert check(prompt, num(star["star_mass"] * 1.04)).ok and check(prompt, num(star["star_mass"] * 0.97)).ok
    assert not check(prompt, num(star["star_mass"] * 1.1)).ok
    # counts of the chain: exact, and written as whole numbers
    prompt = f"world seed {seed} era planets planets"
    assert check(prompt, num(system["planets"])).ok
    assert not check(prompt, num(system["planets"] + 1)).ok and not check(prompt, num(system["planets"] - 1)).ok
    assert not check(prompt, num(system["planets"]) + " point 0").ok
    assert not check(f"world seed {seed} final planets_with_life", "0").ok
    # counts measured in a level: within 1 or 5 percent, whole numbers
    prompt = f"world seed {seed} era {wet} water"
    assert check(prompt, num(surface["water"] + 1)).ok and check(prompt, num(int(surface["water"] * 1.04))).ok
    assert not check(prompt, num(int(surface["water"] * 1.2))).ok
    assert not check(prompt, num(surface["water"]) + " point 5").ok
    prompt = f"world seed {seed} era {wet} organic"
    assert check(prompt, num(surface["organic"] - 1)).ok and not check(prompt, num(surface["organic"] + 5)).ok
    # a count of zero is exact: 0 against 1 is no molecule against one
    cold = next(r for r in rollouts.values() if any(s["organic"] == 0 for s in r.numbered("chemistry")))
    k = next(s["era"] for s in cold.numbered("chemistry") if s["organic"] == 0)
    assert check(f"world seed {num(cold.seed)} era {k} organic", "0").ok
    assert not check(f"world seed {num(cold.seed)} era {k} organic", "1").ok
    # words and parameters: exact
    assert check(f"world seed {seed} era {wet} mix", "ocean").ok
    assert not check(f"world seed {seed} era {wet} mix", "rocky").ok
    assert check(f"world seed {seed} param spin", num(alive.params["spin"])).ok
    assert not check(f"world seed {seed} param spin", num(alive.params["spin"] + 0.01)).ok
    assert not check("world constant cloud_mass", "1 point 5 1").ok
    assert check("world constant black_body_k", "2 7 8 point 3").ok
    verdict = check("world constant black_body_k", "2 7 8")  # a table value is not rounded to 3 digits
    assert not verdict.ok and verdict.expected == "2 7 8 point 3"
    # answers are compared after stripping, never trusted to be numbers
    assert check(f"world seed {seed} final outcome", f" {alive.summary['outcome']} ").ok


def test_owns():
    mine = ["world seed 3 era planets habitable", "world seed 3 final planets_with_life", "world seed 3 why life",
            "world seed 1 2 era chemistry_2 water", "world seed 0 era life_1 outcome", "world seed 3 param spin",
            "world seed 3 timeline", "world seed 9 9 9 8 final outcome", "world constant cloud_mass",
            "world yield oxygen", "world handoff cooling metallicity 0 point 0 1 4 8",
            "world handoff temperature star_mass 1 orbit 1", "world handoff monomers organic 1 6",
            "world handoff metallicity hydrogen 0 point 7 5 generations 2",
            "world seed 3 era enrichment silicon", "world seed 3 era disc collapse_time"]
    for prompt in mine:
        assert owns(prompt) and parse(prompt) is not None, prompt
    foreign = ["turkey capital", "q spoon color", "spoon color", "carbon protons", "calc 1 2 plus 7", "", "world",
               "gravity seed 5 step 0 clumps", "gravity seed 5 final clumps", "nucleo seed 1 final helium",
               "planets seed 3 final habitable", "planets seed 3 step 1 4 gas", "chem valence c",
               "chem seed 4 2 step 1 2 ch4", "life seed 7 final outcome", "life seed 7 param mu",
               "q world seed 3 final planets", "world seed 3", "world seed 3 final", "world seed 0 3 final planets",
               "world seed 3 final banana", "world seed 3 final planets.", "world  seed 3 final planets",
               "world seed 3 era planets water", "world seed 3 era chemistry water", "world seed 3 era planets_1 rocky",
               "world seed 3 era chemistry_0 water", "world seed 3 era banana rocky", "world seed 3 why banana",
               "world seed 3 step 2 planets", "world seed 3 step 2 next planets", "world seed minus 3 final planets",
               "world seed 3 param particles", "world seed 3 timeline eras", "world handoff cooling metallicity",
               "world handoff cooling metallicity 2", "world handoff cooling spin 0 point 2",
               "world handoff banana metallicity 0", "world handoff monomers organic 1 point 5",
               "world handoff cooling metallicity 0 point 0 1 0", "world handoff cooling metallicity 0 0 point 1",
               "world handoff cooling metallicity 1 e 9 9 9", "world handoff temperature star_mass 1",
               "world handoff temperature orbit 1 star_mass 1",
               "world handoff metallicity hydrogen 0 point 7 5 generations 4",
               "world constant banana", "world yield gold", "world yields", "world handoff cooling"]
    for prompt in foreign:
        assert not owns(prompt), prompt
        verdict = check(prompt, "1")
        assert not verdict.ok and verdict.expected is None and verdict.reason == "not my question", prompt
        assert truth(prompt) is None
    # the levels do not claim the chain's prompts, nor the chain theirs
    others = [nucleo.simulation(), gravity.simulation(), planets.simulation(), chem.simulation(), life.simulation()]
    for prompt in mine:
        assert not any(o.owns(prompt) for o in others), prompt


def test_does_not_own_the_levels_questions(rollouts, alive):
    """The prompts the five levels write about their own runs (state, next state, summary,
    parameters), for the very runs this chain is made of, are not the chain's."""
    theirs = []
    for r in (alive, rollouts[1]):
        for level in r.levels.values():
            head = f"{level.sim} seed {num(level.seed)}"
            for t in (0, len(level.steps) - 1):
                theirs += [f"{head} step {num(t)} {key}" for key in level.steps[t]]
                theirs += [f"{head} step {num(t)} next {key}" for key in level.steps[t]]
            theirs += [f"{head} final {key}" for key in level.summary] + [f"{head} param {key}" for key in level.params]
    assert len(theirs) > 300 and {p.split()[0] for p in theirs} == {"nucleo", "gravity", "planets", "chem", "life"}
    for prompt in theirs + ["water molar_mass", "solve 3 x plus 4 equals 1 9", "check 1 2 plus 7 equals 2 0",
                            "balance h 2 plus o 2 gives h 2 o 1", "force gravity range", "carbon neutrons"]:
        assert not owns(prompt) and truth(prompt) is None, prompt
        assert check(prompt, "1") == check(prompt, "yes"), prompt
        assert check(prompt, "1").reason == "not my question" and check(prompt, "1").expected is None


def test_questions_without_an_answer(rollouts):
    """A well-formed prompt about an era the world lacks, or a why of another outcome, is owned
    but cannot be judged: no expected value."""
    for r in rollouts.values():
        seed = num(r.seed)
        missing = f"life_{len(r.numbered('chemistry')) + 1}"
        prompt = f"world seed {seed} era {missing} outcome"
        verdict = check(prompt, "none")
        assert owns(prompt) and not verdict.ok and verdict.expected is None and missing in verdict.reason
        other = next(o for o in OUTCOMES if o != r.summary["outcome"])
        verdict = check(f"world seed {seed} why {other}", "metals star")
        assert not verdict.ok and verdict.expected is None and other in verdict.reason


def test_levels_pass_their_own_gates(rollouts):
    gates = {"nucleo": nucleo.conserved, "gravity": gravity.simulation().conserved,
             "planets": planets.simulation().conserved, "chem": chem.simulation().conserved,
             "life": life.simulation().conserved}
    for r in rollouts.values():
        assert conserved(r).ok, (r.seed, conserved(r))
        assert {"nucleo", "gravity"} <= set(r.levels)
        for name, level in r.levels.items():
            assert gates[level.sim](level).ok, (r.seed, name)


def test_handoff_nucleo_to_cloud(rollouts):
    for seed, r in rollouts.items():
        first, era, enriched = nucleo.rollout(seed), r.era("nucleo"), r.era("enrichment")
        assert r.levels["nucleo"].steps == first.steps
        assert r.primordial["hydrogen"] == first.summary["hydrogen"] and r.primordial["helium"] == first.summary["helium"]
        assert era["hydrogen"] == sig3(first.summary["hydrogen"]) and era["helium"] == sig3(first.summary["helium"])
        # mass is still all there after the enrichment
        assert abs(math.fsum(r.primordial.values()) - 1) < 1e-12 and abs(math.fsum(r.cloud.values()) - 1) < 1e-12
        assert min(r.cloud.values()) >= 0
        g = r.params["generations"]
        z = math.fsum(r.cloud[el] for el in YIELDS)
        assert (z == 0) == (g == 0) and enriched["generations"] == g and enriched["metallicity"] == sig3(z)
        assert r.cloud["hydrogen"] == pytest.approx(r.primordial["hydrogen"] * 0.97 ** g, abs=1e-12)
        assert r.cloud["helium"] - r.primordial["helium"] == pytest.approx(2 * z, abs=1e-12)
        for el, share in YIELDS.items():
            assert r.cloud[el] == pytest.approx(share * z, abs=1e-12) and enriched[el] == sig3(r.cloud[el])
        # the numbers the lines print add up too, within their rounding
        printed = enriched["hydrogen"] + enriched["helium"] + enriched["metallicity"] + era["traces"]
        assert abs(printed - 1) < 2e-3
        assert r.summary["hydrogen"] == enriched["hydrogen"] and r.summary["metallicity"] == enriched["metallicity"]


def test_handoff_cloud_to_star(rollouts):
    for seed, r in rollouts.items():
        cloud, disc, star, level = r.era("cloud"), r.era("disc"), r.era("star"), r.levels["gravity"]
        z = r.era("enrichment")["metallicity"]
        assert cloud["spin"] == r.params["spin"]
        cooling = min(Fraction(6, 10), Fraction(1, 10) + 20 * exact(z))
        assert cloud["cooling"] == float(Fraction(math.floor(cooling * 100 + Fraction(1, 2)), 100))
        assert 0.1 <= cloud["cooling"] <= 0.6
        assert level.seed == seed and level.params["spin"] == cloud["spin"] and level.params["cooling"] == cloud["cooling"]
        assert level.params["n"] == DEFAULTS["particles"] == gravity.N
        assert disc["clumps"] == level.steps[-1]["clumps"] and disc["largest_clump"] == level.steps[-1]["largest_clump"]
        assert disc["spiral"] == ("yes" if any(s["spiral"] == "yes" for s in level.steps) else "no")
        assert star["star_mass"] == half_up(Fraction(3, 2) * exact(disc["largest_clump"]))
        assert star["fusion"] == ("yes" if star["star_mass"] >= 0.08 else "no")
        assert star["luminosity"] == sig3(star["star_mass"] ** 3.5)
        assert star["frost_line"] == sig3(2.7 * math.sqrt(star["star_mass"] ** 3.5))
        assert star["fusion"] == "no" or star["hz_inner"] < star["hz_outer"] < star["frost_line"]
        assert star["disc_mass"] == half_up(10000 * exact(z)) and star["t_gas"] == r.params["t_gas"]


def test_handoff_star_to_planets(rollouts):
    for seed, r in rollouts.items():
        star, era, level = r.era("star"), r.era("planets"), r.levels.get("planets")
        assert (level is not None) == (star["fusion"] == "yes" and star["disc_mass"] > 0)
        if level is None:
            assert all(era[k] == 0 for k in ERAS["planets"]) and not r.numbered("chemistry")
            continue
        assert level.seed == seed and level.params["m_star"] == star["star_mass"]
        assert level.params["disc_mass"] == star["disc_mass"] and level.params["t_gas"] == star["t_gas"]
        assert close(level.params["r_frost"], frost_line_of(star["star_mass"]), rel=1e-9)
        assert era["planets"] == era["rocky"] + era["ice"] + era["gas"] == level.summary["planets"]
        assert era["habitable"] == level.summary["habitable"] <= era["rocky"]
        # beyond the frost line sit the ice and gas planets; a merger may carry a gas giant inwards
        merged = {e["kept"] for e in level.events}
        final = [b for b in level.bodies[-1] if b["solid"] + b["gas"] >= planets.PLANET_MIN]
        assert len(final) == era["planets"]
        for b in final:
            if b["type"] == "ice" or (b["type"] == "gas" and b["id"] not in merged):
                assert b["a"] > level.params["r_frost"], (seed, b)
        assert sum(b["type"] in ("ice", "gas") for b in final) == era["ice"] + era["gas"]
        lo, hi = (edge * math.sqrt(level.params["luminosity"]) for edge in planets.HABITABLE)
        worlds = habitable_bodies(level)
        assert len(worlds) == era["habitable"] and [b["a"] for b in worlds] == sorted(b["a"] for b in worlds)
        assert all(b["type"] == "rocky" and lo <= b["a"] <= hi < level.params["r_frost"] for b in worlds)


def test_handoff_planets_to_chemistry_to_life(rollouts):
    seen_mix = set()
    for seed, r in rollouts.items():
        star, surfaces = r.era("star"), r.numbered("chemistry")
        worlds = habitable_bodies(r.levels["planets"]) if "planets" in r.levels else []
        assert [s["era"] for s in surfaces] == [f"chemistry_{k}" for k in range(1, len(worlds) + 1)]
        for k, (s, body) in enumerate(zip(surfaces, worlds), start=1):
            level = r.levels[f"chemistry_{k}"]
            assert s["orbit"] == sig3(body["a"]) and star["hz_inner"] * 0.99 <= s["orbit"] <= star["hz_outer"] * 1.01
            t = 278.3 * star["star_mass"] ** (3.5 / 4) / math.sqrt(s["orbit"]) + 33
            assert abs(s["temperature"] - t) <= 0.5 + 1e-6 and isinstance(s["temperature"], int)
            assert s["mix"] == ("ocean" if 273 <= s["temperature"] <= 373 else "rocky") == level.summary["mix"]
            assert level.seed == seed + k and level.params["t_end"] == s["temperature"]
            assert level.params["t_start"] == 4000 and level.params["atoms"] == DEFAULTS["atoms"]
            assert level.steps[-1]["temperature"] == s["temperature"] and s["water"] == level.steps[-1]["h2o"]
            assert 0 <= s["organic"] <= level.steps[-1]["molecules"]
            seen_mix.add(s["mix"])
            # life only where there is liquid water and an organic molecule, and always there
            vessel = r.era(f"life_{k}")
            ready = s["mix"] == "ocean" and s["water"] > 0 and s["organic"] > 0
            assert (vessel is not None) == ready == (f"life_{k}" in r.levels)
            if vessel is None:
                continue
            run5 = r.levels[f"life_{k}"]
            assert run5.seed == seed + k and vessel["monomers"] == 700 * s["organic"] == run5.params["monomers"]
            assert run5.steps[0]["free_monomers"] == vessel["monomers"]
            assert vessel["replicators"] == run5.steps[-1]["replicators"] and vessel["outcome"] == run5.summary["outcome"]
            assert vessel["outcome"] in life.OUTCOMES
        assert len(r.numbered("life")) <= len(surfaces)
    assert seen_mix == {"ocean", "rocky"}


def test_summary_matches_the_eras(rollouts):
    for r in rollouts.values():
        s, surfaces, vessels = r.summary, r.numbered("chemistry"), r.numbered("life")
        assert list(s) == list(SUMMARY_KEYS) and s == summarise(r.steps)
        assert tuple(x["era"] for x in r.steps[:6]) == FIXED_ERAS
        assert s["eras"] == len(r.steps) == 6 + len(surfaces) + len(vessels)
        assert s["clumps"] == r.era("disc")["clumps"] and s["planets"] == r.era("planets")["planets"]
        assert s["habitable"] == r.era("planets")["habitable"] == len(surfaces)
        assert s["planets_with_water"] == sum(x["mix"] == "ocean" and x["water"] > 0 for x in surfaces)
        assert s["planets_with_life"] == sum(x["replicators"] > 0 for x in vessels)
        assert s["planets_with_life"] <= s["planets_with_water"] <= s["habitable"] <= s["planets"]
        assert s["outcome"] in OUTCOMES and s["outcome"] == outcome_of(r.steps)
        assert (s["outcome"] in ("life", "dominated")) == (s["planets_with_life"] > 0)


def test_why_names_the_links_that_held(rollouts):
    for r in rollouts.values():
        held = conditions(r.steps)
        assert held == [c for c in CONDITIONS if c in held]
        missing = [c for c in CONDITIONS if c not in held]
        outcome = r.summary["outcome"]
        assert outcome == (MISSING[missing[0]] if missing else outcome) and (missing or outcome in ("life", "dominated"))
        assert ("metals" in held) == (r.params["generations"] > 0)
        assert ("water" in held) == (r.summary["planets_with_water"] > 0)
        assert ("survivors" in held) == (r.summary["planets_with_life"] > 0)
        answer = " ".join(held) or "nothing"
        verdict = check(f"world seed {num(r.seed)} why {outcome}", answer)
        assert verdict.ok and verdict.expected == answer
        if outcome in ("life", "dominated"):
            assert held == list(CONDITIONS)
            best = max(r.numbered("life"), key=lambda s: s["replicators"])
            assert outcome == ("dominated" if best["outcome"] == "dominated" else "life")


def test_no_metals_no_rock(rollouts):
    """Generations 0: no metals, so no solids, no planets, and no rocky planet with water."""
    primordial = [r for r in rollouts.values() if r.params["generations"] == 0]
    assert primordial
    forced = [run(seed, generations=0, particles=128) for seed in (3, 5)]  # worlds that have life with their own metals
    for r in primordial + forced:
        assert r.era("enrichment")["metallicity"] == 0 and all(r.era("enrichment")[el] == 0 for el in YIELDS)
        assert r.cloud == r.primordial and r.era("cloud")["cooling"] == 0.1
        assert r.era("star")["disc_mass"] == 0 and "planets" not in r.levels
        assert all(r.era("planets")[k] == 0 for k in ERAS["planets"])
        assert not r.numbered("chemistry") and not r.numbered("life")
        assert r.summary["planets"] == r.summary["planets_with_water"] == r.summary["planets_with_life"] == 0
        assert r.summary["outcome"] == "no_metals" and "metals" not in conditions(r.steps)
        assert conserved(r).ok
    assert all(rollouts[seed].summary["planets_with_life"] for seed in (3, 5))


def test_one_seed_in_twenty_reaches_life(rollouts):
    for block in (SEEDS[:20], SEEDS[20:]):
        living = [seed for seed in block if rollouts[seed].summary["planets_with_life"] > 0]
        assert len(living) >= 1
        assert not any(rollouts[seed].params["generations"] == 0 for seed in living)
    assert {r.summary["outcome"] for r in rollouts.values()} >= {"no_metals", "no_water", "no_habitable", "life",
                                                                 "dominated"}
    # every world with metals has a star and planets; the more generations, the more solids
    for r in rollouts.values():
        if r.params["generations"]:
            assert r.era("star")["fusion"] == "yes" and r.summary["planets"] >= 2
    by_generation = {g: [r.era("star")["disc_mass"] for r in rollouts.values() if r.params["generations"] == g]
                     for g in (1, 2, 3)}
    assert max(by_generation[1]) < min(by_generation[3])


def test_plausibility_targets(rollouts):
    """The targets of the module docstring, each on seeds 1 to 40."""
    solar = 0.014
    gas = {g: [] for g in range(4)}
    surfaces, vessels = [], []
    for r in rollouts.values():
        g, cloud, star, system = r.params["generations"], r.era("cloud"), r.era("star"), r.era("planets")
        z = r.era("enrichment")["metallicity"]
        assert conserved(r).ok
        # the first hour leaves hydrogen and helium and next to nothing else
        assert r.era("nucleo")["hydrogen"] > r.era("nucleo")["helium"] > 0.05 and r.era("nucleo")["traces"] < 0.001
        # metals: none without stars, about the sun's after two generations, more with every one
        assert (z == 0) == (g == 0)
        if g:
            assert z == pytest.approx(r.era("nucleo")["hydrogen"] * (1 - 0.97 ** g) / 3, rel=0.01)
        if g == 2:
            assert 0.6 * solar <= z <= 1.4 * solar
        # every level gets parameters inside the range it was built for
        assert cloud["cooling"] == 0.1 if g == 0 else 0.2 <= cloud["cooling"] <= 0.6
        assert 0.05 <= cloud["spin"] <= 0.4 and 1.0 <= star["t_gas"] <= 10.0
        assert star["fusion"] == "yes" and 0.3 <= star["star_mass"] <= 1.5
        gas[g].append(system["gas"])
        if g == 0:
            assert "planets" not in r.levels and system["planets"] == system["rocky"] == 0
            continue
        assert 0.5 <= star["star_mass"] <= 1.5 and 10 <= star["disc_mass"] <= 300
        assert 2 <= system["planets"] <= 10 and system["habitable"] <= 2
        surfaces += r.numbered("chemistry")
        vessels += r.numbered("life")
        for s in r.numbered("chemistry"):
            assert star["hz_inner"] * 0.99 <= s["orbit"] <= star["hz_outer"] * 1.01 < star["frost_line"]
    # gas giants need solids: none after one generation, most after three
    assert sum(gas[0]) == sum(gas[1]) == 0 < sum(gas[2]) / len(gas[2]) < sum(gas[3]) / len(gas[3])
    # surfaces: between the edges of the zone; an ocean makes water and organics, a frozen rock next to none
    assert len(surfaces) >= 20 and {s["mix"] for s in surfaces} == {"ocean", "rocky"}
    for s in surfaces:
        assert 246 <= s["temperature"] <= 319
        if s["mix"] == "ocean":
            assert 273 <= s["temperature"] and 5000 <= s["water"] <= 6000 and 5 <= s["organic"] <= 30
        else:
            assert s["temperature"] < 273 and s["water"] <= 60 and s["organic"] <= 2
    # life: every ocean gets a vessel inside level 5's range of monomers, and most vessels come alive
    assert len(vessels) == sum(s["mix"] == "ocean" for s in surfaces) >= 10
    assert all(5000 <= s["monomers"] <= 20000 for s in vessels)
    assert sum(s["replicators"] > 0 for s in vessels) > len(vessels) / 2


def test_conserved_catches_a_broken_chain(rollouts, alive):
    def broken(change, r=alive):
        bad = copy.deepcopy(r)
        change(bad)
        verdict = conserved(bad)
        assert not verdict.ok and verdict.reason
        return verdict.reason

    def put(era, key, value):
        """Set one metric of one era; ``value`` may be a function of the old value."""
        def change(r):
            step = r.era(era) if isinstance(era, str) else era(r)
            step[key] = value(step[key]) if callable(value) else value
        return change

    def level_step(name, t, key, value):
        def change(r):
            r.levels[name(r) if callable(name) else name].steps[t][key] = value
        return change

    def swap(r):
        r.steps[2], r.steps[3] = r.steps[3], r.steps[2]

    def drop_life(r):
        name = r.numbered("life")[0]["era"]
        r.steps = [s for s in r.steps if s["era"] != name]
        del r.levels[name]
        r.summary = summarise(r.steps)

    def more_oxygen(r):
        r.cloud["oxygen"] += 1e-6
        r.cloud["carbon"] -= 1e-6

    def first_life(r):
        return r.numbered("life")[0]

    wet = next(s["era"] for s in alive.numbered("chemistry") if s["mix"] == "ocean")
    assert conserved(alive).ok
    # the mass ledger
    assert "sum to" in broken(lambda r: r.cloud.update(oxygen=r.cloud["oxygen"] + 0.01))
    assert "yields" in broken(more_oxygen)
    assert "nucleo" in broken(lambda r: r.levels.update(nucleo=nucleo.rollout(r.seed + 1)))
    assert "enrichment" in broken(put("enrichment", "metallicity", 0.5))
    # the hand-offs
    assert "cooling" in broken(put("cloud", "cooling", lambda c: 0.1 if c == 0.6 else 0.6))
    assert "disc" in broken(put("disc", "clumps", lambda n: n + 1))
    assert "star" in broken(put("star", "star_mass", 2.0))
    assert "star" in broken(put("star", "disc_mass", 500.0))
    assert "planets" in broken(put("planets", "habitable", lambda n: n + 1))
    assert wet in broken(put(wet, "mix", "rocky"))
    assert wet in broken(put(wet, "water", lambda n: n + 5))
    assert wet in broken(put(wet, "organic", lambda n: n + 1))
    assert wet in broken(put(wet, "temperature", lambda t: t + 1))
    assert wet in broken(put(wet, "orbit", 5.0))
    assert "life" in broken(drop_life)
    assert "life" in broken(put(first_life, "replicators", lambda n: n + 1))
    assert "life" in broken(put(first_life, "monomers", 600))
    # the summary and the timeline
    assert "summary" in broken(lambda r: r.summary.update(planets_with_life=r.summary["planets_with_life"] + 1))
    assert "summary" in broken(lambda r: r.summary.update(outcome="sterile"))
    assert "order" in broken(swap)
    assert "metrics" in broken(lambda r: r.era("disc").pop("clumps"))
    assert "levels" in broken(lambda r: r.levels.pop("gravity"))
    # a level that breaks its own law
    assert "level gravity" in broken(level_step("gravity", 5, "mass", 2.0))
    assert "level planets" in broken(level_step("planets", 9, "zone_solids", 1.0))
    assert f"level {wet}" in broken(level_step(wet, -1, "h2o", 3))
    assert "level life" in broken(level_step(lambda r: first_life(r)["era"], 7, "bound", 1))
    # a world without metals may not grow planets
    barren = next(r for r in rollouts.values() if r.params["generations"] == 0)
    assert "planets" in broken(put("planets", "planets", 3), barren)
    # only a world rollout is judged
    assert not conserved(Rollout("world", 1, {}, [])).ok
    assert not conserved(alive.levels["gravity"]).ok
    assert not conserved(WorldRollout("world", 1, {}, [{"era": "nucleo"}])).ok


def test_levels_are_the_levels_own_runs(alive):
    """Each level of the chain is what the level's own module gives for those parameters."""
    seed, star = alive.seed, alive.era("star")
    again = gravity.simulation().run(seed, spin=alive.params["spin"], cooling=alive.era("cloud")["cooling"])
    assert again.steps == alive.levels["gravity"].steps and again.summary == alive.levels["gravity"].summary
    again = planets.simulation().run(seed, m_star=star["star_mass"], disc_mass=star["disc_mass"], t_gas=star["t_gas"])
    assert again.steps == alive.levels["planets"].steps and again.summary == alive.levels["planets"].summary
    for k, surface in enumerate(alive.numbered("chemistry"), start=1):
        again = chem.simulation().run(seed + k, mix=surface["mix"], t_start=4000, t_end=surface["temperature"],
                                      steps=30, atoms=20000)
        assert again.steps == alive.levels[f"chemistry_{k}"].steps
        if alive.era(f"life_{k}"):
            again = life.run(seed + k, monomers=700 * surface["organic"])
            assert again.steps == alive.levels[f"life_{k}"].steps and again.summary == alive.levels[f"life_{k}"].summary


def test_organic_molecules_counts_carbon_hydrogen_bonds():
    def pool(counts, bonds):
        p = chem.Pool(counts, np.random.default_rng(0))
        p.bond_a = np.array([a for a, _ in bonds], dtype=np.int64)
        p.bond_b = np.array([b for _, b in bonds], dtype=np.int64)
        p.bond_m = np.ones(len(bonds), dtype=np.int64)
        return p

    assert organic_molecules(pool({"c": 1, "h": 4}, [])) == 0
    assert organic_molecules(pool({"c": 1, "h": 4}, [(0, 1), (0, 2), (0, 3), (0, 4)])) == 1  # methane, counted once
    assert organic_molecules(pool({"c": 2, "h": 8}, [(0, 2), (0, 3), (1, 6), (7, 1)])) == 2
    assert organic_molecules(pool({"h": 2, "o": 1}, [(0, 2), (1, 2)])) == 0  # water
    assert organic_molecules(pool({"c": 1, "o": 2}, [(0, 1), (0, 2)])) == 0  # carbon dioxide
    # carbonic acid holds carbon and hydrogen but no bond between them: not organic
    assert organic_molecules(pool({"c": 1, "o": 3, "h": 2}, [(0, 1), (0, 2), (0, 3), (2, 4), (3, 5)])) == 0
    # the ocean mix makes some, the rocky mix next to none
    for mix, least, most in (("ocean", 5, 40), ("rocky", 0, 3)):
        level = chem.simulation().run(9, mix=mix, t_start=4000, t_end=290, steps=30, atoms=20000)
        last = None
        for _, _, last in chem.cool(level.seed, level.params):
            pass
        assert least <= organic_molecules(last) <= most
        if mix == "ocean":
            assert level.summary["organic"] == 0  # level 4's own key: chains of more than two carbons


def test_handoff_rules():
    assert sum(YIELDS.values()) == pytest.approx(1.0, abs=1e-12) and list(YIELDS)[:2] == ["oxygen", "carbon"]
    assert BLACK_BODY == 278.3 == CONSTANTS["black_body_k"]["value"]
    rng = random.Random(8)
    for _ in range(200):
        x = rng.uniform(0.4, 0.9)
        y = rng.uniform(0.0, 1 - x)
        for g in range(4):
            cloud = enrich(x, y, g)
            assert abs(math.fsum(cloud.values()) - 1) < 1e-12 and min(cloud.values()) >= -1e-15
            z = math.fsum(cloud[el] for el in YIELDS)
            assert z == pytest.approx(metallicity_of(x, g), abs=1e-12)
            assert z == pytest.approx(x * (1 - 0.97 ** g) / 3, abs=1e-12)
    assert metallicity_of(0.75, 0) == 0 and metallicity_of(0.75, 1) == pytest.approx(0.0075)
    assert metallicity_of(0.75, 2) == pytest.approx(0.014775)
    assert cooling_of(0) == 0.1 and cooling_of(0.015) == 0.4 and cooling_of(0.05) == 0.6 and cooling_of(0.00798) == 0.26
    assert star_mass_of(0.5) == 0.75 and star_mass_of(0.672) == 1.01 and star_mass_of(0) == 0
    assert star_mass_of(0.5, cloud_mass=3.0) == 1.5
    assert fusion_of(0.08) == "yes" and fusion_of(0.0799) == "no"
    assert luminosity_of(1.0) == 1.0 and luminosity_of(0.05) == 0 and luminosity_of(1.21) == pytest.approx(1.21 ** 3.5)
    assert frost_line_of(1.0) == 2.7 and frost_line_of(0.05) == 0
    assert disc_mass_of(0.0148) == 148.0 and disc_mass_of(0) == 0
    assert temperature_of(1.0, 1.0) == 311 and temperature_of(1.0, 4.0) == 172 and temperature_of(0.05, 1.0) == 33
    assert [mix_of(t) for t in (272, 273, 300, 373, 374)] == ["rocky", "ocean", "ocean", "ocean", "rocky"]
    assert monomers_of(16) == 11200 and monomers_of(0) == 0
    # the further out, the colder; the heavier the star, the warmer
    assert temperature_of(1.0, 1.2) < temperature_of(1.0, 1.0) < temperature_of(1.2, 1.0)


def test_products_are_exact_and_a_five_rounds_up():
    """The hand-offs that are products or sums of printed numbers, against whole-number arithmetic
    for every input the lines can print; binary floats would settle the ties by their noise."""
    for k in range(1000):  # every largest clump of 3 decimals
        assert star_mass_of(k / 1000) == half_up(Fraction(3, 2) * Fraction(k, 1000)), k
        assert star_mass_of(k / 1000, cloud_mass=0.7) == half_up(Fraction(7, 10) * Fraction(k, 1000)), k
    for k in range(3001):  # every metallicity up to 0.03 in steps of 1e-5
        z = k / 100000
        cooling = min(Fraction(6, 10), Fraction(1, 10) + 20 * Fraction(k, 100000))
        assert cooling_of(z) == float(Fraction(math.floor(cooling * 100 + Fraction(1, 2)), 100)), z
        assert disc_mass_of(z) == half_up(10000 * Fraction(k, 100000)), z
    for k in range(500, 901):  # every printed hydrogen from 0.5 to 0.9
        for g in range(4):
            z = Fraction(k, 1000) * (1 - Fraction(97, 100) ** g) / 3
            assert metallicity_printed(k / 1000, g) == half_up(z), (k, g)
            assert metallicity_printed(k / 1000, g) == pytest.approx(metallicity_of(k / 1000, g), rel=0.006)
    # the ties: 1.5 * 0.453 = 0.6795 (a float product says 0.67949...), 0.1 + 20 * 0.00725 = 0.245
    assert sig3(1.5 * 0.453) == 0.679 and star_mass_of(0.453) == 0.68
    assert star_mass_of(0.391) == 0.587 and star_mass_of(0.673) == 1.01 and star_mass_of(0.613) == 0.92
    assert cooling_of(0.00725) == 0.25 and cooling_of(0.00775) == 0.26 and cooling_of(0.00025) == 0.11
    assert metallicity_printed(0.75, 2) == 0.0148 and metallicity_printed(0.663, 1) == 0.00663
    # whatever decimal context the caller has set
    with decimal.localcontext() as ctx:
        ctx.prec, ctx.rounding = 2, decimal.ROUND_DOWN
        assert star_mass_of(0.453) == 0.68 and star_mass_of(0.673) == 1.01 and cooling_of(0.00725) == 0.25
        assert disc_mass_of(0.00663) == 66.3 and metallicity_printed(0.75, 3) == 0.0218
        assert star_mass_of(0.5234567890123456, cloud_mass=1.2345678901234567) == 0.646
    # no corner case: of the stars from 0.1 to 1 solar mass every second one is a tie
    assert sum(15 * k % 10 == 5 for k in range(67, 667)) == 300
    # answers come out as the lines print them
    assert handoff_line("star_mass", 0.453).text == "q world handoff star_mass largest_clump 0 point 4 5 3. a 0 point 6 8."
    assert handoff_line("cooling", 0.00725).text == "q world handoff cooling metallicity 0 point 0 0 7 2 5. a 0 point 2 5."


def test_handoff_questions(rollouts):
    assert handoff_line("cooling", 0.0148).text == "q world handoff cooling metallicity 0 point 0 1 4 8. a 0 point 4."
    assert handoff_line("temperature", 1.0, 1.0).text == "q world handoff temperature star_mass 1 orbit 1. a 3 1 1."
    assert handoff_line("mix", 300).text == "q world handoff mix temperature 3 0 0. a ocean."
    assert handoff_line("monomers", 16).text == "q world handoff monomers organic 1 6. a 1 1 2 0 0."
    assert handoff_line("fusion", 0.05).text == "q world handoff fusion star_mass 0 point 0 5. a no."
    assert handoff_line("metallicity", 0.75, 2).text == \
        "q world handoff metallicity hydrogen 0 point 7 5 generations 2. a 0 point 0 1 4 8."
    verdict = check("world handoff cooling metallicity 0 point 0 1 4 8", "0 point 5")
    assert not verdict.ok and verdict.expected == "0 point 4"
    assert check("world handoff disc_mass metallicity 0 point 0 1 4 8", "1 4 8").ok
    assert not check("world handoff mix temperature 2 6 0", "ocean").ok
    a, b = practice_lines(random.Random(5), 300), practice_lines(random.Random(5), 300)
    assert [ln.text for ln in a] == [ln.text for ln in b] != [ln.text for ln in practice_lines(random.Random(6), 300)]
    assert {ln.prompt.split()[2] for ln in a} == set(HANDOFFS)
    # the hand-offs of a world, asked with the numbers its eras print, give the numbers the next era prints
    for r in rollouts.values():
        era = {s["era"]: s for s in r.steps}
        answers = {ln.prompt.split()[2]: ln.answer for ln in handoff_lines(r)}
        assert answers["cooling"] == dense_value(era["cloud"]["cooling"])
        assert answers["star_mass"] == dense_value(era["star"]["star_mass"])
        assert answers["fusion"] == era["star"]["fusion"]
        assert answers["luminosity"] == dense_value(era["star"]["luminosity"])
        assert answers["frost_line"] == dense_value(era["star"]["frost_line"])
        assert answers["disc_mass"] == dense_value(era["star"]["disc_mass"])
        # from the printed hydrogen the metallicity may differ in its last digit
        assert close(parse_num(answers["metallicity"]), era["enrichment"]["metallicity"], rel=0.005)
        for s in r.numbered("chemistry"):
            assert check(f"world handoff temperature star_mass {num(era['star']['star_mass'], sig=6)} "
                         f"orbit {num(s['orbit'], sig=6)}", num(s["temperature"])).expected == num(s["temperature"])
            assert truth(f"world handoff mix temperature {num(s['temperature'])}") == s["mix"]
        for s in r.numbered("life"):
            organic = era[s["era"].replace("life", "chemistry")]["organic"]
            assert truth(f"world handoff monomers organic {num(organic)}") == s["monomers"]


def test_tables():
    recs, table = records(), table_lines()
    assert len(recs) == len(HANDOFFS) + 1 and all(ln.kind == "record" for ln in recs)
    assert len(table) == len(CONSTANTS) + len(YIELDS) and len({ln.prompt for ln in table}) == len(table)
    assert "world handoff cooling. from metallicity. cooling_primordial 0 point 1. cooling_per_metal 2 0. " \
           "cooling_max 0 point 6." in [ln.text for ln in recs]
    assert recs[-1].text.startswith("world yields. oxygen 0 point 4 4. carbon 0 point 1 8.")
    assert "q world constant cloud_mass. a 1 point 5." in [ln.text for ln in table]
    assert "q world constant black_body_k. a 2 7 8 point 3." in [ln.text for ln in table]
    assert all(ln.answer == table_value(truth(ln.prompt)) for ln in table)
    assert "q world yield oxygen. a 0 point 4 4." in [ln.text for ln in table]
    assert simulation().records() == recs and simulation().table_lines() == table
    used = {c for hand in HANDOFFS.values() for c in hand.constants}
    assert used <= set(CONSTANTS)
    # the constants the chain shares with level 3 are level 3's
    assert CONSTANTS["frost_au"]["value"] == planets.FROST_AU
    assert (CONSTANTS["hz_inner_au"]["value"], CONSTANTS["hz_outer_au"]["value"]) == planets.HABITABLE
    assert CONSTANTS["chem_start_k"]["value"] == chem.PLASMA_K


def test_generate():
    a, b = generate(random.Random(11), 100), generate(random.Random(11), 100)
    assert len(a) == 100 and [ln.text for ln in a] == [ln.text for ln in b]
    assert a[:len(table_lines())] == table_lines()
    assert [ln.text for ln in generate(random.Random(12), 100)] != [ln.text for ln in a]
    assert generate(random.Random(1), 0) == [] and len(generate(random.Random(1), 5)) == 5
    for ln in a:
        assert is_dense(ln.text) and n_tokens(ln.text) <= 80
        if ln.kind != "record":
            assert check(ln.prompt, ln.answer).ok, ln.text
    assert any(ln.kind == "record" and " era " in ln.text for ln in a)
    assert simulation().generate(random.Random(11), 20) == a[:20]


def test_other_sizes_and_parameters():
    small = run(3, particles=128, atoms=8000)
    assert small.params["particles"] == 128 and small.levels["gravity"].params["n"] == 128
    assert all(level.params["atoms"] == 8000 for name, level in small.levels.items() if name.startswith("chemistry"))
    assert conserved(small).ok and not canonical(small)
    with pytest.raises(ValueError):
        lines(small)  # check() replays the world a seed names, not this one
    # a cloud too light for a star: the chain ends there
    dark = run(3, cloud_mass=0.05, particles=128)
    assert dark.era("star")["fusion"] == "no" and dark.era("star")["luminosity"] == 0
    assert dark.era("star")["frost_line"] == 0 and "planets" not in dark.levels
    assert dark.summary["outcome"] == "no_star" and dark.summary["planets"] == 0 and conserved(dark).ok
    assert conditions(dark.steps) == ["metals"]
    # more generations: more metals, a heavier disc
    discs = [run(3, generations=g, particles=128).era("star")["disc_mass"] for g in (1, 2, 3)]
    assert discs == sorted(discs) and discs[0] > 0
    for bad in ({"generations": 4}, {"generations": -1}, {"generations": 1.5}, {"spin": -0.1}, {"cloud_mass": 0},
                {"atoms": 10}, {"t_gas": float("nan")}):
        with pytest.raises(ValueError):
            run(3, **bad)
    with pytest.raises(TypeError):
        run(3, cooling=0.3)  # the cooling is a hand-off, not a parameter
    for seed in (-1, 1.5, True, "3"):
        with pytest.raises(ValueError):
            run(seed)
        with pytest.raises(ValueError):
            rollout(seed)
