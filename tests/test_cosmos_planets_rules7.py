"""Level 3 under rules 7: the corrected planets level (``haishool.cosmos.planets``, section "Rules 7").

Round 6 itself is guarded by ``tests/test_cosmos_planets.py`` and ``tests/test_round6_frozen.py``;
here: that the default is still round 6, and what the corrections do.
"""
import copy
import math
import random
import statistics
import time

import pytest

from haishool.cosmos import Simulation
from haishool.cosmos import planets as P
from haishool.evo import LessonGate
from haishool.truth import Gate, is_dense, parse_num, split_line

SEEDS = list(range(1, 41))


def n_tokens(text: str) -> int:
    return len(text.replace(".", " . ").split())


def spell(n: int) -> str:
    return " ".join(str(n))


@pytest.fixture(scope="module")
def sim():
    return P.Planets(rules=7)


@pytest.fixture(scope="module")
def rollouts(sim):
    return {seed: sim.rollout(seed) for seed in SEEDS}


@pytest.fixture(scope="module")
def all_lines(rollouts):
    return [ln for r in rollouts.values() for ln in P.lines(r)]


# ---------------------------------------------------------------------------------------------
# the default is still round 6


def test_the_default_is_round_6():
    s = P.simulation()
    assert s is P.simulation(rules=6) and s.rules == 6 and P.simulation(rules=7).rules == 7
    assert P.simulation(rules=7) is P.simulation(rules=7) and P.simulation(rules=7) is not s
    p6 = P.random_params(random.Random(3))
    assert p6 == P.random_params(random.Random(3), rules=6) == {"m_star": 0.74, "disc_mass": 167.8, "t_gas": 4.33,
                                                                "core_threshold": 10.0}
    r = s.run(3, **p6)
    assert "rules" not in r.params and "m_star" in r.params and r.system == []
    same = P.run(3, rules=6, **p6)
    assert same.params == r.params and same.steps == r.steps and same.summary == r.summary
    assert s.run(3, star_mass=0.74, disc_mass=167.8, t_gas=4.33).steps == r.steps, "star_mass is another name of m_star"
    texts = [ln.text for ln in P.lines(r)]
    assert texts == [ln.text for ln in P.lines(r, rules=6)] == [ln.text for ln in s.lines(r)]
    assert texts[0] == ("planets seed 3 params. m_star 0 point 7 4. disc_mass 1 6 7 point 8. t_gas 4 point 3 3. "
                        "core_threshold 1 0. r_frost 1 point 5 9. luminosity 0 point 3 4 9.")
    assert "q planets seed 3 final habitable. a 1." in texts and not any(" rules " in t for t in texts)
    assert P.check("planets seed 3 final gas", "1").ok and P.owns("planets seed 3 step 1 4 gas")
    assert P.frost_line(0.74) == P.FROST_AU * math.sqrt(0.74 ** 3.5)
    assert P.frost_line(0.74, rules=7) == P.frost_line7(0.74) != P.frost_line(0.74)


def test_only_rules_6_and_7_exist(sim):
    for bad in (5, 8, "7", 7.5, True, None):
        with pytest.raises(ValueError):
            P.run(1, rules=bad)
        with pytest.raises(ValueError):
            P.random_params(random.Random(1), rules=bad)
        with pytest.raises(ValueError):
            P.simulation(rules=bad)
    r6, r7 = P.rollout(3), P.rollout(3, rules=7)
    with pytest.raises(ValueError):
        P.lines(r7, rules=6)
    with pytest.raises(ValueError):
        P.lines(r6, rules=7)
    with pytest.raises(ValueError):
        P.simulation().run(1, gas_mass=5000.0)  # round 6 ties the gas to the solids
    with pytest.raises(ValueError):
        P.run(1, rules=7, m_star=0.8, star_mass=0.9)
    assert P.run(1, rules=7, m_star=0.8).params == P.run(1, rules=7, star_mass=0.8).params


def test_protocols(sim):
    assert isinstance(sim, Simulation) and isinstance(sim, Gate)
    assert sim.sim == sim.topic == P.SIM == "planets" and sim.KEYS is P.KEYS7 and P.simulation().KEYS is P.KEYS
    for key in P.PARAM_KEYS7 + P.STATE_KEYS7 + P.LEDGER_KEYS7 + P.SUMMARY_KEYS7 + P.PLANET_KEYS7:
        assert key in P.KEYS7, key
    assert P.INPUT_KEYS7 <= set(P.PARAM_KEYS7)
    assert isinstance(P.LESSONS, LessonGate) and P.lesson_gate() is P.LESSONS is sim.lesson_gate()
    assert P.LESSONS.sim == "planets7" and P.LESSONS.topic == "predict_planets7" and len(P.RULES) >= 8
    for name in ("simulation", "random_params", "run", "rollout", "conserved", "lines", "check", "owns"):
        assert callable(getattr(P, name)), name


# ---------------------------------------------------------------------------------------------
# determinism


def test_same_seed_same_rollout_and_lines():
    a, b = P.Planets(rules=7), P.Planets()
    for seed in (1, 7, 23, 4321):
        ra, rb = a.rollout(seed), b.rollout(seed, rules=7)
        assert ra.params == rb.params and ra.steps == rb.steps and ra.summary == rb.summary
        assert ra.bodies == rb.bodies and ra.events == rb.events and ra.system == rb.system
        assert [repr(v) for s in ra.steps for v in s.values()] == [repr(v) for s in rb.steps for v in s.values()]
        assert [ln.text for ln in P.lines(ra)] == [ln.text for ln in a.lines(rb)]
        again = P.run(seed, rules=7, **P.random_params(random.Random(seed), rules=7))
        assert again.steps == ra.steps and again.params == ra.params and again.system == ra.system
    assert a.rollout(1).steps != a.rollout(2).steps
    assert a.run(5, disc_mass=200.0).steps == P.run(5, rules=7, disc_mass=200.0).steps
    assert a.run(5, disc_mass=200.0).steps != a.run(6, disc_mass=200.0).steps
    assert a.run(5, disc_mass=200.0).params["rules"] == 7 and a.run(5, disc_mass=200.0, rules=6).params.get("rules") is None
    one = [ln.text for ln in a.generate(random.Random(11), 3)]
    assert one == [ln.text for ln in b.generate(random.Random(11), 3, rules=7)] and len(one) > 2000
    assert all(" rules 7 " in t for t in one) and one != [ln.text for ln in a.generate(random.Random(12), 3)]


def test_pinned_numbers_and_lines_of_seed_3(sim):
    """What the module docstring shows, word for word; the numbers pin the run across numpy versions
    (numpy only supplies the raw PCG64 bits, which the round-6 tests pin)."""
    r = sim.rollout(3)
    assert r.params["rules"] == 7 and list(r.params) == [*P.PARAM_KEYS7, "gas_initial", "rules"]
    assert {k: r.params[k] for k in P.INPUT_KEYS7} == {"star_mass": 0.74, "disc_mass": 124.2, "t_gas": 1.92,
                                                       "core_threshold": 10.0}
    assert r.steps[14]["largest_mass"] == pytest.approx(7.016884506379852, rel=1e-9)
    assert r.summary["total_planet_mass"] == pytest.approx(34.97961336355045, rel=1e-9)
    assert len(r.events) == 19 and r.events[0]["ids"] == [4, 5] and r.events[0]["step"] == 12
    assert sum(bool(e["ejected"]) for e in r.events) == 10 == r.summary["ejected"]
    texts = [ln.text for ln in P.lines(r)]
    assert texts[0] == ("planets seed 3 rules 7 params. star_mass 0 point 7 4. disc_mass 1 2 4 point 2. "
                        "t_gas 1 point 9 2. core_threshold 1 0. luminosity 0 point 3. frost_line 1 point 4 8. "
                        "hz_inner 0 point 5 2. hz_outer 0 point 9 1 8.")
    assert texts[1] == ("planets seed 3 rules 7 step 0. time 0. planets 0. rocky 0. super_earths 0. icy 0. "
                        "ice_giants 0. gas_giants 0. ejected 0. largest_mass 0. innermost_au 0. outermost_au 0. "
                        "debris 0. gas_present yes. stage dust.")
    for text in (
            "planets seed 3 rules 7 step 1 4. time 4. planets 1 3. rocky 7. super_earths 0. icy 2. ice_giants 4. "
            "gas_giants 0. ejected 7. largest_mass 7 point 0 2. innermost_au 0 point 1 8 1. outermost_au 1 4 point 9. "
            "debris 0 point 5. gas_present no. stage clearing.",
            "q planets seed 3 rules 7 step 1 4 ejected. a 7.",
            "q planets seed 3 rules 7 step 1 4 next stage. a clearing.",
            "planets seed 3 rules 7 planet 2. orbit 0 point 4 9 9. mass 4 point 3 8. type rocky. flux 1 point 2 1. "
            "temperature 7 6 7. habitable no.",
            "q planets seed 3 rules 7 planet 2 temperature. a 7 6 7.",
            "q planets seed 3 rules 7 final habitable. a 0.",
            "q planets seed 3 rules 7 final planets. a 5.",
            "q planets seed 3 rules 7 final total_planet_mass. a 3 5."):
        assert text in texts, text


def test_the_random_numbers_are_round_6s_stream():
    """The zone efficiencies are the first 24 numbers of the seed's PCG64 stream, as in round 6."""
    for seed in (0, 3, 1874):
        disc = P._Disc7(P._Rng(seed), 1.0, 100.0, 3.0)
        assert [b.eps for b in disc.rows] == P._Rng(seed).uniform(*P.EPS_RANGE, size=24).tolist()
        assert [b.eps for b in disc.rows] == P._Disc(P._Rng(seed), 1.0, 100.0, 3.0).eps.tolist()
        assert [b.id for b in disc.rows] == list(range(24))


def test_values_are_plain_python_numbers_and_words(rollouts):
    for r in rollouts.values():
        for row in r.steps + [r.summary, r.params] + r.system:
            for key, value in row.items():
                assert type(value) in (float, int, str), (key, type(value))
                if type(value) is float:
                    assert math.isfinite(value) and value >= 0, (key, value)
        for bodies in r.bodies:
            for b in bodies:
                assert [type(b[k]) for k in ("id", "a", "solid", "gas", "type")] == [int, float, float, float, str]
                assert b["type"] in P.TYPES7
        for step in r.steps:
            assert list(step) == [*P.STATE_KEYS7, *P.LEDGER_KEYS7]
        assert list(r.summary) == list(P.SUMMARY_KEYS7)


def test_a_change_in_the_twelfth_digit_moves_no_line(sim, rollouts):
    """Another machine's exp or cbrt may differ in the sixteenth digit; a change ten thousand times
    larger must not move a single number of a line."""

    def printed(r):
        out = [P.dense_value(s[k]) for s in r.steps for k in P.STATE_KEYS7]
        out += [P.dense_value(v) for v in r.summary.values()]
        return out + [P.dense_value(p[k]) for p in r.system for k in P.PLANET_KEYS7]

    for seed, r in rollouts.items():
        p = P.random_params(random.Random(seed), rules=7)
        for f in (1 + 1e-12, 1 - 1e-12):
            nudged = sim.run(seed, **dict(p, star_mass=p["star_mass"] * f, disc_mass=p["disc_mass"] / f))
            assert nudged.steps != r.steps
            assert printed(nudged) == printed(r), (seed, f)


def test_a_run_is_fast(sim):
    worst = 0.0
    for seed in range(100, 140):
        start = time.perf_counter()
        sim.run(seed, **P.random_params(random.Random(seed), rules=7))
        worst = max(worst, time.perf_counter() - start)
    assert worst < 1.0


# ---------------------------------------------------------------------------------------------
# parameters (correction 4)


def test_random_params_under_rules_7():
    draws = []
    for seed in range(1, 2001):
        p = P.random_params(random.Random(seed), rules=7)
        assert p == P.random_params(random.Random(seed), rules=7) == P.simulation(rules=7).random_params(random.Random(seed))
        assert list(p) == ["star_mass", "disc_mass", "t_gas", "core_threshold"] and p["core_threshold"] == 10.0
        assert p["star_mass"] == P.random_params(random.Random(seed))["m_star"], "the same first draw as round 6"
        assert 0.5 <= p["star_mass"] <= 1.5 and 1.0 <= p["t_gas"] <= 10.0
        assert 10 * p["star_mass"] - 0.06 <= p["disc_mass"] <= 300 * p["star_mass"] + 0.06, "solids scale with the star"
        assert p["disc_mass"] == round(p["disc_mass"], 1) and p["t_gas"] == round(p["t_gas"], 2)
        draws.append(p)
    assert P.random_params(random.Random(3), rules=7) == {"star_mass": 0.74, "disc_mass": 124.2, "t_gas": 1.92,
                                                         "core_threshold": 10.0}
    t = [p["t_gas"] for p in draws]
    # discs fade roughly as exp(-t / 2.5 Myr): about a third left at 3 Myr, under a tenth at 6
    assert 0.30 < sum(x > 3 for x in t) / len(t) < 0.43
    assert 0.04 < sum(x > 6 for x in t) / len(t) < 0.12
    assert 2.6 < statistics.mean(t) < 3.3 and statistics.median(t) < 2.7
    old = [P.random_params(random.Random(seed))["t_gas"] for seed in range(1, 2001)]
    assert sum(x > 6 for x in old) / len(old) > 0.4, "round 6: uniform, 44 % of discs older than 6 Myr"
    assert P.gas_lifetime(0.0) == 1.0 and P.gas_lifetime(0.999999) == 10.0
    assert P.gas_lifetime(1 - math.exp(-1)) == 3.0 and P.gas_lifetime(0.5) == round(1 + 2 * math.log(2), 2)


def test_run_rejects_impossible_parameters(sim):
    for bad in ({"star_mass": 0.0}, {"star_mass": -1.0}, {"disc_mass": 0.0}, {"disc_mass": float("nan")},
                {"t_gas": -1.0}, {"t_gas": float("inf")}, {"core_threshold": 0.0}, {"gas_mass": -1.0},
                {"gas_mass": float("inf")}):
        with pytest.raises(ValueError):
            sim.run(1, **bad)
    no_gas = sim.run(1, disc_mass=300.0, t_gas=0.0)
    assert sim.conserved(no_gas).ok and no_gas.summary["gas_giants"] == 0
    assert all(s["gas_captured"] == 0 and s["gas_left"] == 0 and s["gas_present"] == "no" for s in no_gas.steps)
    crumbs = sim.run(1, disc_mass=0.5)
    assert sim.conserved(crumbs).ok and crumbs.summary["largest_mass"] < 1
    assert sim.conserved(sim.run(1, disc_mass=200.0, gas_mass=0.0)).ok


# ---------------------------------------------------------------------------------------------
# lines and the gate (correction 3)


def test_every_line_is_dense_and_short(all_lines, rollouts):
    assert len(all_lines) > 30000
    for ln in all_lines:
        assert is_dense(ln.text), ln.text
        assert n_tokens(ln.text) <= 128, ln.text
        assert ln.topic == "planets" and ln.meta == {"rules": 7}
        head = ln.text.split(".")[0].split()
        head = head[1:] if head[0] == "q" else head
        assert head[:2] == ["planets", "seed"] and "rules 7" in " ".join(head), ln.text
        if ln.kind != "record":
            assert "." not in ln.prompt and "." not in ln.answer
            assert split_line(ln.text) == (ln.prompt, ln.answer)
            assert ln.kind in ("fact", "calc")
    assert max(n_tokens(ln.text) for ln in all_lines) <= 80
    for seed in (9998, 123456789):
        for ln in P.lines(P.rollout(seed, rules=7)):
            assert is_dense(ln.text) and n_tokens(ln.text) <= 128, ln.text


def test_line_kinds(rollouts):
    r = rollouts[3]
    ls = P.lines(r)
    records = [ln for ln in ls if ln.kind == "record"]
    assert records[0].text == P.params_line7(r).text
    assert len(records) == 1 + len(P.TIMES) + len(r.system) and len(r.system) == r.summary["planets"] == 5
    prompts = [ln.prompt for ln in ls if ln.kind != "record"]
    n_state, n_times = len(P.STATE_KEYS7), len(P.TIMES)
    assert len(prompts) == len(set(prompts)) == (n_times * n_state + (n_times - 1) * n_state
                                                 + len(r.system) * len(P.PLANET_KEYS7) + len(P.SUMMARY_KEYS7))
    assert len(ls) == 903 == len({ln.text for ln in ls})
    assert not any(f" {k}" in ln.text for ln in ls for k in P.LEDGER_KEYS7), "the ledger is not trained"
    assert len(P.lines(r, every=5)) < len(ls) / 3


def test_no_prompt_has_two_answers(rollouts):
    """A rules-7 story never uses a round-6 prompt: seed 3 under round 6 and under rules 7 are two
    different systems."""
    for seed in (3, 7, 24):
        new = {ln.prompt for ln in P.lines(rollouts[seed]) if ln.kind != "record"}
        old = {ln.prompt for ln in P.lines(P.rollout(seed)) if ln.kind != "record"}
        assert new and old and not new & old
        assert {ln.text.split(".")[0] for ln in P.lines(rollouts[seed]) if ln.kind == "record"}.isdisjoint(
            {ln.text.split(".")[0] for ln in P.lines(P.rollout(seed)) if ln.kind == "record"})
        assert all(P.simulation().parse(p) is None for p in new), "round 6's parser does not read a rules-7 prompt"
        assert all(P.parse7(p) is None for p in old)
    assert P.rollout(3).steps != rollouts[3].steps


def test_only_the_canonical_rollout_of_a_seed_has_lines(sim):
    with pytest.raises(ValueError):
        P.lines(sim.run(3, disc_mass=200.0))
    p = P.random_params(random.Random(3), rules=7)
    with pytest.raises(ValueError):
        P.lines(sim.run(3, **p, gas_mass=5000.0))
    assert P.lines(sim.run(3, **p))[0].text == P.lines(sim.rollout(3))[0].text


def test_gate_agrees_with_every_line(all_lines):
    for ln in all_lines:
        if ln.kind == "record":
            continue
        assert P.owns(ln.prompt), ln.prompt
        v = P.check(ln.prompt, ln.answer)
        assert v.ok and v.expected == ln.answer, (ln.text, v)
    assert P.simulation().check(all_lines[5].prompt, all_lines[5].answer).ok, "a round-6 object judges both kinds"


def test_gate_agrees_with_every_record_field(all_lines):
    for ln in all_lines:
        if ln.kind != "record":
            continue
        head, *fields = [f.strip() for f in ln.text.rstrip(".").split(".")]
        for f in fields:
            key, _, value = f.partition(" ")
            v = P.check(f"{head} {key}", value)
            assert v.ok and v.expected == value, (ln.text, key)


def test_gate_rejects_altered_answers(all_lines, rollouts):
    rng = random.Random(5)
    questions = [ln for ln in all_lines if ln.kind != "record"]
    words = {"gas_present": ("yes", "no"), "stage": P.STAGES, "type": P.TYPES7, "habitable": ("yes", "no")}
    for ln in rng.sample(questions, 4000):
        if parse_num(ln.answer) is None:
            wrong = next(w for w in words[ln.prompt.split()[-1]] if w != ln.answer)
        else:
            wrong = "9 " + ln.answer  # a changed leading digit: far outside the 5 % tolerance
        v = P.check(ln.prompt, wrong)
        assert not v.ok and v.expected == ln.answer, (ln.text, wrong, v)
    # one digit changed: a count or a kelvin is then wrong, a measured number when it is more than 5 % off
    numeric = [ln for ln in questions if parse_num(ln.answer) is not None]
    rejected = 0
    for ln in rng.sample(numeric, 3000):
        digits = ln.answer.split()
        place = rng.choice([i for i, w in enumerate(digits) if w.isdigit()])
        digits[place] = str((int(digits[place]) + rng.randint(1, 9)) % 10)
        wrong = " ".join(digits)
        truth, got = parse_num(ln.answer), parse_num(wrong)
        v = P.check(ln.prompt, wrong)
        off = abs(got - truth) / max(abs(got), abs(truth))
        exact = isinstance(P.simulation().truth(ln.prompt), int)
        if exact or off > 0.06:
            assert not v.ok, (ln.text, wrong, v)
        elif off < 0.04:
            assert v.ok, (ln.text, wrong, v)
        rejected += not v.ok
    assert rejected > 2000
    for answer in ("1 e 9 9 9", "minus 1 e 9 9 9", "", "point", "yes", "7 . 5", "many"):
        for prompt in ("planets seed 3 rules 7 final largest_mass", "planets seed 3 rules 7 final planets",
                       "planets seed 3 rules 7 params star_mass", "planets seed 3 rules 7 planet 1 temperature"):
            v = P.check(prompt, answer)
            assert not v.ok and v.expected is not None, (prompt, answer)
    # tolerance: 5 % for what is measured; a parameter that was put in exactly; counts exactly
    mass = rollouts[3].summary["total_planet_mass"]
    assert P.check("planets seed 3 rules 7 final total_planet_mass", P.dense_value(mass * 1.03)).ok
    assert not P.check("planets seed 3 rules 7 final total_planet_mass", P.dense_value(mass * 1.1)).ok
    assert P.check("planets seed 3 rules 7 params star_mass", "0 point 7 4").ok
    assert not P.check("planets seed 3 rules 7 params star_mass", "0 point 7 5").ok
    assert P.check("planets seed 3 rules 7 params star_mass", "0 point 7 5").expected == "0 point 7 4"
    assert P.check("planets seed 3 rules 7 params frost_line", "1 point 5").ok
    assert not P.check("planets seed 3 rules 7 final planets", "6").ok
    assert not P.check("planets seed 3 rules 7 planet 2 temperature", "7 6 6").ok


@pytest.mark.parametrize("prompt", [
    "", "planets", "turkey capital", "carbon protons", "calc 1 2 plus 7", "gravity seed 7 step 2 0 clumps",
    "gravity seed 7 rules 7 step 2 0 clumps", "nucleo seed 3 final helium", "nucleo seed 3 rules 7 final helium",
    "chem seed 7 final water", "life seed 7 final stage", "world seed 3 era planets", "stars seed 7 final metallicity",
    "cells seed 7 step 3 cells", "signals seed 7 say predator near", "society seed 7 final stage",
    "planets7 seed 3 rules 7 final planets", "planets7 predict flux luminosity 1 orbit 1",
    "planets predict solids_remaining remaining 1 efficiency 1 orbit 1 star_mass 1 dt 1",
    "planets seed 3 rules 7", "planets seed 3 rules 7 final", "planets seed 3 rules 7 final gas",
    "planets seed 3 rules 7 final ice", "planets seed 3 rules 7 step 1 4 gas", "planets seed 3 rules 7 step 1 4 ice",
    "planets seed 3 rules 7 params m_star", "planets seed 3 rules 7 params r_frost",
    "planets seed 3 rules 7 params gas_initial", "planets seed 3 rules 7 params rules",
    "planets seed 3 rules 6 final planets", "planets seed 3 rules 8 final planets", "planets seed 3 rules final planets",
    "planets seed 3 rules 7 step 3 0 planets", "planets seed 3 rules 7 step 2 9 next planets",
    "planets seed 3 rules 7 step 0 3 planets", "planets seed 0 3 rules 7 final planets",
    "planets seed minus 3 rules 7 final planets", "planets seed 3 rules 7 final planets extra",
    "planets seed 3 rules 7 step 3 next", "planets seed 3 rules 7 step 3 next next planets",
    "planets seed 3 rules 7 final stage", "planets seed 3 rules 7 planet 0 type", "planets seed 3 rules 7 planet 2 5 type",
    "planets seed 3 rules 7 planet 2", "planets seed 3 rules 7 planet 2 stage", "planets seed 3 rules 7 planet 2 next type",
    "planets seed 3 rules 7 era planets", "q planets seed 3 rules 7 final planets", "planets  seed 3 rules 7 final planets",
    "planets seed 3 rules 7 final planets ", "planets seed 3 final gas_giants", "planets seed 3 step 1 4 ejected",
    "planets seed 3 final inner_giants", "planets seed 3 params star_mass", "planets seed 3 planet 2 type",
])
def test_does_not_own_other_prompts(prompt):
    assert not P.owns(prompt) and not P.simulation(rules=7).owns(prompt)
    v = P.check(prompt, "1")
    assert v.ok is False and v.expected is None and v.reason == "not my question"


@pytest.mark.parametrize("prompt", [
    "planets seed 7 rules 7 step 0 planets", "planets seed 7 rules 7 step 2 9 stage",
    "planets seed 7 rules 7 step 2 8 next gas_present", "planets seed 1 2 3 4 rules 7 step 1 2 next largest_mass",
    "planets seed 7 rules 7 final total_planet_mass", "planets seed 7 rules 7 final habitable",
    "planets seed 7 rules 7 final inner_giants", "planets seed 7 rules 7 params frost_line",
    "planets seed 7 rules 7 params hz_outer", "planets seed 7 rules 7 step 4 zone_solids",
    "planets seed 7 rules 7 step 4 gas_ejected", "planets seed 0 rules 7 final gas_giants",
    "planets seed 7 rules 7 step 1 0 ejected", "planets seed 7 rules 7 planet 1 flux",
])
def test_owns_its_prompts(prompt):
    assert P.owns(prompt) and P.simulation(rules=7).owns(prompt) and P.simulation().owns(prompt)
    assert P.check(prompt, "nonsense words").expected is not None
    assert not P.LESSONS.owns(prompt), "the lesson gate has its own word"


def test_a_planet_the_system_does_not_have(rollouts):
    r = rollouts[3]
    n = len(r.system)
    assert P.check(f"planets seed 3 rules 7 planet {n} type", r.system[-1]["type"]).ok
    v = P.check(f"planets seed 3 rules 7 planet {n + 1} type", "rocky")
    assert not v.ok and v.expected is None and "no planet" in v.reason
    assert P.simulation().truth(f"planets seed 3 rules 7 planet {n + 1} type") is None


# ---------------------------------------------------------------------------------------------
# the gate of the run: conserved()


def test_conserved_on_forty_seeds_with_random_params(sim, rollouts):
    for seed in SEEDS:
        r = sim.run(seed, **sim.random_params(random.Random(seed)))
        v = sim.conserved(r)
        assert v.ok, (seed, v)
        assert P.conserved(r).ok and P.simulation().conserved(r).ok, "any object judges a rollout by the rules it carries"
        assert r.steps == rollouts[seed].steps
        for s in r.steps:
            assert abs(s["solids_in_bodies"] + s["zone_solids"] + s["debris_mass"] - r.params["disc_mass"]) <= 1e-9
            assert s["debris"] == pytest.approx(s["debris_mass"] / r.params["disc_mass"], rel=1e-12)
            assert s["gas_in_bodies"] == pytest.approx(s["gas_captured"] - s["gas_ejected"], abs=1e-9)
            assert s["gas_captured"] + s["gas_left"] <= r.params["gas_initial"] + 1e-9
            assert s["planets"] == sum(s[k] for k in P.TYPE_COUNTS7.values())
        assert r.steps[0]["zone_solids"] == pytest.approx(r.params["disc_mass"], abs=1e-9)
        assert math.fsum(e["debris"] for e in r.events) == pytest.approx(r.steps[-1]["debris_mass"], abs=1e-9)
        assert math.fsum(e["gas_ejected"] for e in r.events) == pytest.approx(r.steps[-1]["gas_ejected"], abs=1e-9)
        assert sum(len(e["ejected"]) for e in r.events) == r.summary["ejected"]
    # other stars, other discs, the optional gas budget
    for seed, kw in enumerate(({"star_mass": 0.08, "disc_mass": 11.0}, {"star_mass": 0.2, "disc_mass": 30.0},
                               {"star_mass": 3.0, "disc_mass": 400.0}, {"star_mass": 10.0, "disc_mass": 1400.0},
                               {"disc_mass": 450.0, "t_gas": 10.0}, {"disc_mass": 300.0, "t_gas": 10.0, "gas_mass": 9700.0},
                               {"disc_mass": 300.0, "t_gas": 6.0, "core_threshold": 5.0}), start=50):
        assert sim.conserved(sim.run(seed, **kw)).ok, kw


def test_conserved_catches_tampering(sim):
    base = sim.rollout(23)  # a run with mergers, ejections, a migrating gas giant and ejected gas
    assert sim.conserved(base).ok and base.steps[-1]["gas_ejected"] > 0 and base.summary["inner_giants"] == 1

    def broken(change) -> bool:
        r = copy.deepcopy(base)
        change(r)
        return not sim.conserved(r).ok

    assert not broken(lambda r: None)
    assert broken(lambda r: r.steps[10].update(zone_solids=r.steps[10]["zone_solids"] + 1e-6))
    assert broken(lambda r: r.steps[10].update(solids_in_bodies=r.steps[10]["solids_in_bodies"] * 0.99))
    assert broken(lambda r: r.steps[29].update(debris=r.steps[29]["debris"] + 0.01))
    assert broken(lambda r: r.steps[29].update(debris_mass=r.steps[29]["debris_mass"] + 0.5))
    assert broken(lambda r: r.steps[12].update(planets=r.steps[12]["planets"] + 1))
    assert broken(lambda r: r.steps[12].update(rocky=r.steps[12]["rocky"] + 1, icy=r.steps[12]["icy"] - 1))
    assert broken(lambda r: r.steps[29].update(ejected=r.steps[29]["ejected"] + 1))
    assert broken(lambda r: r.steps[12].update(time=r.steps[11]["time"]))
    assert broken(lambda r: r.steps[14].update(gas_captured=r.params["gas_initial"] * 2))
    assert broken(lambda r: r.steps[29].update(gas_in_bodies=r.steps[29]["gas_in_bodies"] + 1.0))
    assert broken(lambda r: r.steps[29].update(gas_ejected=0.0))
    assert broken(lambda r: r.steps[5].update(stage="finished"))
    assert broken(lambda r: r.bodies[29][0].update(solid=r.bodies[29][0]["solid"] * 0.5))
    assert broken(lambda r: r.bodies[29][0].update(type="icy"))
    assert broken(lambda r: r.bodies[29][0].update(a=r.bodies[29][0]["a"] * 1.01)), "a rocky body does not move"
    assert broken(lambda r: r.bodies[29].pop())
    assert broken(lambda r: r.bodies.pop())
    assert broken(lambda r: r.events.clear())
    assert broken(lambda r: r.events.pop(0))
    assert broken(lambda r: r.events[0].update(step=0))
    assert broken(lambda r: r.events[0].update(kept=99))
    throw = next(n for n, e in enumerate(base.events) if e["ejected"])
    join = next(n for n, e in enumerate(base.events) if not e["ejected"])
    assert broken(lambda r: r.events[throw].update(debris=r.events[throw]["debris"] + 0.001))
    assert broken(lambda r: r.events[throw].update(ejected=[]))
    assert broken(lambda r: r.events[throw].update(orbit=r.events[throw]["orbit"] * 1.01))
    assert broken(lambda r: r.events[throw].update(theta=0.5))
    assert broken(lambda r: r.events[join].update(orbit=r.events[join]["orbit"] * 1.001)), "angular momentum"
    assert broken(lambda r: r.events[join].update(mass_after=r.events[join]["mass_after"] * 0.9))
    assert broken(lambda r: r.events[join].update(debris=0.1))
    assert broken(lambda r: r.events[join]["members"][0].update(solid=r.events[join]["members"][0]["solid"] + 1.0))
    assert broken(lambda r: r.system.pop())
    assert broken(lambda r: r.system[0].update(temperature=r.system[0]["temperature"] + 1))
    assert broken(lambda r: r.system[0].update(habitable="yes"))
    assert broken(lambda r: r.summary.update(habitable=r.summary["habitable"] + 1))
    assert broken(lambda r: r.summary.update(planets=r.summary["planets"] + 1))
    assert broken(lambda r: r.params.pop("frost_line"))

    def shift(r):  # the totals still add up, the bodies do not
        r.bodies[29][0]["solid"] -= 0.5
        r.bodies[29][1]["solid"] += 0.5
    assert broken(shift)


# ---------------------------------------------------------------------------------------------
# lessons


def test_lessons_are_judged_and_true_of_the_simulation():
    lessons = P.LESSONS.generate(random.Random(1), 400)
    assert len(lessons) == 400 and {ln.prompt.split()[2] for ln in lessons} == set(P.RULES)
    assert [ln.text for ln in lessons] == [ln.text for ln in P.LESSONS.generate(random.Random(1), 400)]
    for ln in lessons:
        assert is_dense(ln.text) and n_tokens(ln.text) <= 128, ln.text
        assert ln.topic == "predict_planets7" and ln.prompt.startswith("planets7 predict ")
        assert P.LESSONS.owns(ln.prompt) and not P.owns(ln.prompt)
        v = P.LESSONS.check(ln.prompt, ln.answer)
        assert v.ok and v.expected == ln.answer, (ln.text, v)
        name, values = P.LESSONS.parse(ln.prompt)
        own = getattr(P, P.RULES[name].compute.__name__)(*values)  # the simulation's own function
        assert ln.answer == (own if isinstance(own, str) else P.num(own)), ln.text
        wrong = ("no" if ln.answer == "yes" else "yes") if parse_num(ln.answer) is None else "9 " + ln.answer
        assert not P.LESSONS.check(ln.prompt, wrong).ok, ln.text
    for ln in P.LESSONS.records():
        assert is_dense(ln.text) and n_tokens(ln.text) <= 128 and ln.kind == "record", ln.text
    assert len(P.LESSONS.records()) == len(P.RULES) == 24
    words = {ln.answer for ln in P.LESSONS.generate(random.Random(2), 3000) if parse_num(ln.answer) is None}
    assert words == {"yes", "no", *P.TYPES7, *P.CLIMATES7}, "every word answer occurs"


def test_every_lesson_function_is_called_by_the_run(monkeypatch):
    """A lesson is true of the simulation because its ``compute`` is the function the run calls:
    count the calls of every one of them during one run with mergers, ejections and migration."""
    calls = dict.fromkeys(P.RULES, 0)

    def counted(name, fn):
        def wrapper(*args, **kwargs):
            calls[name] += 1
            return fn(*args, **kwargs)
        return wrapper

    for name, rule in P.RULES.items():
        fn = rule.compute
        assert getattr(P, fn.__name__) is fn, name
        monkeypatch.setattr(P, fn.__name__, counted(name, fn))
    params = P.random_params(random.Random(23), rules=7)
    r = P.Planets(rules=7).run(23, **params)
    assert all(n > 0 for n in calls.values()), {k: n for k, n in calls.items() if n == 0}
    monkeypatch.undo()
    assert r.steps == P.rollout(23, rules=7).steps


def test_a_substep_is_the_lessons(sim):
    """One substep of the disc, recomputed with the lesson functions alone."""
    disc = P._Disc7(P._Rng(5), 0.9, 200.0, 3.0)
    before = [(b.s_rem, b.eps, b.a, b.solid) for b in disc.rows]
    assert [b.s_rem for b in disc.rows] == [P.zone_solids(200.0, i) for i in range(24)]
    assert math.fsum(b.s_rem for b in disc.rows) == pytest.approx(200.0, abs=1e-12)
    disc.advance(P._Rng(6), 0.0, 0.005, 10.0)
    assert len(disc.rows) == 24
    for b, (s_rem, eps, a, solid) in zip(disc.rows, before):
        assert b.s_rem == P.solids_remaining(s_rem, eps, a, 0.9, 0.005) and b.solid == solid + (s_rem - b.s_rem)
    r = sim.rollout(23)
    lum, frost = r.params["luminosity"], r.params["frost_line"]
    assert lum == P.luminosity7(r.params["star_mass"]) and frost == P.frost_line7(r.params["star_mass"])
    assert (r.params["hz_inner"], r.params["hz_outer"]) == (P.hz_inner(lum), P.hz_outer(lum))
    for bodies in r.bodies:
        for b in bodies:
            assert b["type"] == P.planet_type(b["solid"], b["gas"], b["a"], frost)
    assert {p["type"] for p in r.system} >= {"gas", "rocky"}
    for p in r.system:
        assert p["flux"] == P.flux(lum, p["orbit"])
        if p["type"] == "gas":  # no surface: the equilibrium temperature of the cloud tops
            assert p["temperature"] == math.floor(P.equilibrium_temperature(p["flux"]) + 0.5)
        else:
            assert p["temperature"] == P.surface_temperature(p["flux"])
        bare_rock = p["type"] == "rocky" and p["gas"] == 0
        assert p["habitable"] == (P.habitable(lum, p["orbit"], p["mass"]) if bare_rock else "no")
    for e in r.events:
        kept = next(m for m in e["members"] if m["id"] == e["kept"])
        assert e["theta"] == P.safronov(kept["solid"] + kept["gas"], kept["a"], r.params["star_mass"])
        assert bool(e["ejected"]) == (P.scatters(kept["solid"] + kept["gas"], kept["a"], r.params["star_mass"]) == "yes")
        if not e["ejected"] and len(e["members"]) == 2:
            x, y = e["members"]
            assert e["orbit"] == P.merged_orbit(x["solid"] + x["gas"], x["a"], y["solid"] + y["gas"], y["a"])


# ---------------------------------------------------------------------------------------------
# the corrected rules, one by one


def test_luminosity_is_the_piecewise_law_and_the_grid_follows_it(sim):
    """Correction 5: 0.23 M^2.3, M^4, 1.4 M^3.5, 32000 M; frost line and zone on the same zones for every star."""
    assert P.luminosity7(1.0) == 1.0 and P.luminosity7(0.5) == 0.0625 and P.luminosity7(1.5) == 5.0625
    assert P.luminosity7(0.1) == pytest.approx(0.23 * 0.1 ** 2.3) and P.luminosity7(3.0) == pytest.approx(1.4 * 3 ** 3.5)
    assert P.luminosity7(60.0) == 32000 * 60.0
    for edge in (0.43, 2.0, 55.0):  # the pieces meet within 4 percent
        assert P.luminosity7(edge - 1e-9) == pytest.approx(P.luminosity7(edge), rel=0.04)
    assert all(P.luminosity7(a / 100) < P.luminosity7((a + 1) / 100) for a in range(5, 6000))
    assert 0.5 ** 3.5 / P.luminosity7(0.5) > 1.4, "round 6 made a half-solar star 41 % too bright"
    assert P.frost_line7(1.0) == P.FROST_AU == P.frost_line(1.0)
    shares = None
    for m in (0.08, 0.2, 0.5, 1.0, 1.5, 3.0):
        disc = P._Disc7(P._Rng(1), m, 100.0, 3.0)
        root = math.sqrt(P.luminosity7(m))
        assert disc.rows[0].r_in == pytest.approx(P.R_IN * root) and disc.rows[-1].r_out == pytest.approx(P.R_OUT * root)
        assert [b.a > disc.r_frost for b in disc.rows] == [i >= P.FIRST_ICY_ZONE7 for i in range(24)]
        inside = [P.climate(P.flux(disc.luminosity, b.a)) == "temperate" for b in disc.rows]
        assert inside == [i in (6, 7, 8) for i in range(24)], "the same three zones start in the habitable zone"
        assert shares in (None, [b.s_rem for b in disc.rows])
        shares = [b.s_rem for b in disc.rows]
    assert P.FIRST_ICY_ZONE7 == 11 and math.fsum(P.ZONE_SOLIDS7) == pytest.approx(1) == pytest.approx(math.fsum(P.ZONE_GAS7))
    # no zone centre near the frost line or a zone edge: the nearest is 0.9 percent off
    centres = [b.a for b in P._Disc7(P._Rng(1), 1.0, 100.0, 3.0).rows]
    for line in (P.FROST_AU, P.hz_inner(1.0), P.hz_outer(1.0)):
        assert min(abs(c - line) / line for c in centres) > 0.005


def test_small_stars_can_have_habitable_planets(sim):
    """Round 6's grid began at 0.3 au, so no star below 0.39 solar masses could have a planet in
    its zone; most stars are that small."""
    for m in (0.2, 0.3):
        runs = [sim.run(seed, star_mass=m, disc_mass=140.0 * m, t_gas=3.0) for seed in SEEDS]
        assert all(sim.conserved(r).ok for r in runs)
        assert sum(r.summary["habitable"] >= 1 for r in runs) >= 28, m
        old = [P.simulation().run(seed, m_star=m, disc_mass=140.0, t_gas=3.0).summary["habitable"] for seed in SEEDS]
        assert sum(old) == 0


def test_the_zone_is_a_flux_and_its_water_is_liquid(rollouts):
    """Correction 1: one rule for the zone and the temperature that goes with it."""
    assert (P.S_MAX_GREENHOUSE, P.S_RUNAWAY) == (0.356, 1.107)
    assert P.hz_inner(1.0) == pytest.approx(0.9504, abs=1e-4) and P.hz_outer(1.0) == pytest.approx(1.676, abs=1e-3)
    assert P.hz_inner(4.0) == 2 * P.hz_inner(1.0) and P.flux(4.0, 2.0) == 1.0
    assert [P.climate(s) for s in (1.2, 1.107, 1.0, 0.356, 0.3559)] == ["runaway", "temperate", "temperate",
                                                                      "temperate", "frozen"]
    assert P.T_EQ_EARTH == pytest.approx(278.3 * 0.7 ** 0.25, abs=0.05)
    assert P.surface_temperature(1.0) == 288, "the earth"
    assert P.surface_temperature(P.S_MAX_GREENHOUSE) == 273 and P.surface_temperature(P.S_RUNAWAY) == 294
    assert P.equilibrium_temperature(0.356) + P.greenhouse(0.356) == pytest.approx(273.0, abs=0.01)
    assert P.surface_temperature(1.1071) == 761 and P.surface_temperature(1 / 0.723 ** 2) == 799, "Venus' orbit"
    assert P.surface_temperature(1 / 1.524 ** 2) == 278, "Mars' orbit lies inside the zone; Mars is too small"
    # beyond the outer edge no greenhouse holds 273 K: the bare equilibrium temperature
    assert P.greenhouse(0.3559) == 0 == P.greenhouse(0.001) and P.greenhouse(0.356) == pytest.approx(76.34, abs=0.01)
    assert [P.greenhouse(s) for s in (1.2, 1.107, 1.0)] == [500, 33, 33]
    assert P.surface_temperature(0.3559) == 197 == math.floor(P.equilibrium_temperature(0.3559) + 0.5)
    assert P.surface_temperature(1 / 30.07 ** 2) == 46, "Neptune's orbit: 46 K, not the 147 K of a line that runs on"
    for thousandths in range(1, 3001):  # liquid water (273 to 373 K) exactly where the flux is temperate
        s = thousandths / 1000
        liquid = P.WATER_K[0] <= P.surface_temperature(s) <= P.WATER_K[1]
        assert liquid == (P.climate(s) == "temperate"), s
    assert all(P.surface_temperature(s / 1000) <= P.surface_temperature((s + 1) / 1000) for s in range(1, 3000))
    # a gas giant has no surface and no runaway greenhouse: the equilibrium temperature of its cloud tops
    assert P._temperature7(1 / 5.203 ** 2, "gas") == 112, "Jupiter's orbit"
    assert P._temperature7(400.0, "gas") == 1139 and P._temperature7(400.0, "rocky") == 1639 == P.surface_temperature(400.0)
    assert P._temperature7(0.8, "gas") == 241 and P._temperature7(0.8, "super_earth") == P.surface_temperature(0.8) == 287
    for ty in P.TYPES7:
        assert P._temperature7(0.01, ty) == P.surface_temperature(0.01) == 81
    assert P.habitable(1.0, 1.524, 0.107) == "no" and P.habitable(1.0, 1.524, 0.5) == "yes"
    assert P.habitable(1.0, 1.0, 1.0) == "yes" and P.habitable(1.0, 0.723, 0.815) == "no"
    assert P.habitable(1.0, 1.0, 0.29) == "no" and P.habitable(1.0, 1.0, 10.5) == "no"
    for hundredths in range(36, 111):  # everywhere in the zone water is liquid
        s = hundredths / 100
        if P.climate(s) == "temperate":
            assert P.WATER_K[0] <= P.surface_temperature(s) <= P.WATER_K[1], s
    found = 0
    for r in rollouts.values():
        lum = r.params["luminosity"]
        for p in r.system:
            in_zone = P.S_MAX_GREENHOUSE <= lum / p["orbit"] ** 2 <= P.S_RUNAWAY
            bare_rock = p["type"] == "rocky" and p["gas"] == 0
            assert (p["habitable"] == "yes") == (bare_rock and in_zone and 0.3 <= p["mass"] <= 10.0)
            liquid = 273 <= p["temperature"] <= 373
            if p["type"] != "gas":
                assert liquid == in_zone, "liquid water in the zone and nowhere else"
            if p["habitable"] == "yes":
                found += 1
                assert 273 <= p["temperature"] <= 294 and r.params["hz_inner"] <= p["orbit"] <= r.params["hz_outer"]
            elif in_zone:
                assert not bare_rock or not 0.3 <= p["mass"] <= 10.0
            else:
                assert p["temperature"] <= 197 or p["temperature"] >= 761 or p["type"] == "gas"
            if p["orbit"] > r.params["frost_line"]:
                assert p["temperature"] == math.floor(P.equilibrium_temperature(p["flux"]) + 0.5) <= 155
        assert r.summary["habitable"] == sum(p["habitable"] == "yes" for p in r.system)
    assert found >= 28
    # round 6 on the same seeds: the next level's temperature rule freezes many of its habitable planets
    frozen = total = 0
    for seed in SEEDS:
        r = P.rollout(seed)
        lo, hi = (h * math.sqrt(r.params["luminosity"]) for h in P.HABITABLE)
        for b in r.bodies[-1]:
            if b["solid"] + b["gas"] >= P.PLANET_MIN and b["type"] == "rocky" and lo <= b["a"] <= hi:
                total += 1
                frozen += 278.3 * r.params["luminosity"] ** 0.25 / math.sqrt(b["a"]) + 33 < 272.5
    assert total >= 30 and frozen >= 10


def test_a_planet_that_kept_disc_gas_is_not_habitable(sim):
    """A core of 5 earth masses or more that met the disc gas keeps a hydrogen envelope; it may
    still be ``rocky`` by type (the envelope is lighter than the core), but it has no earth-like sky."""
    lum = 1.0
    bodies = [{"id": 6, "a": 1.0, "solid": 6.0, "gas": 0.0, "type": "rocky"},
              {"id": 7, "a": 1.2, "solid": 6.0, "gas": 0.5, "type": "rocky"},
              {"id": 8, "a": 1.4, "solid": 12.0, "gas": 0.0, "type": "super_earth"},
              {"id": 9, "a": 1.6, "solid": 12.0, "gas": 40.0, "type": "gas"},
              {"id": 12, "a": 5.2, "solid": 0.04, "gas": 0.0, "type": "icy"}]
    system = P._system7(bodies, lum)
    assert [p["id"] for p in system] == [6, 7, 8, 9], "a body below 0.1 earth masses is no planet"
    assert [p["habitable"] for p in system] == ["yes", "no", "no", "no"]
    assert all(P.climate(p["flux"]) == "temperate" for p in system)
    assert [p["temperature"] for p in system] == [288, 286, 281, 201], "the giant: equilibrium temperature alone"
    # the canonical seeds have no such planet in the zone; light cores that run away in a gas-poor disc give some
    seen = 0
    for seed in range(1, 41):
        r = sim.run(seed, star_mass=1.0, disc_mass=200.0, t_gas=3.0, core_threshold=1.0, gas_mass=100.0)
        assert sim.conserved(r).ok
        for p in r.system:
            if p["type"] == "rocky" and p["gas"] > 0 and P.climate(p["flux"]) == "temperate" and 0.3 <= p["mass"] <= 10:
                seen += 1
                assert p["habitable"] == "no" and 273 <= p["temperature"] <= 294
            if p["habitable"] == "yes":
                assert p["gas"] == 0 and p["type"] == "rocky"
    assert seen >= 10
    for seed in range(1, 41):
        assert not any(p["gas"] > 0 and p["habitable"] == "yes" for p in sim.rollout(seed).system)
    # a hot jupiter has no runaway greenhouse, and the gate knows
    heavy = (sim.run(seed, star_mass=1.0, disc_mass=300.0, t_gas=10.0) for seed in range(1, 41))
    r = copy.deepcopy(next(x for x in heavy if any(p["type"] == "gas" and p["flux"] > P.S_RUNAWAY for p in x.system)))
    hot = next(p for p in r.system if p["type"] == "gas" and p["flux"] > P.S_RUNAWAY)
    assert sim.conserved(r).ok and hot["habitable"] == "no"
    assert hot["temperature"] == math.floor(P.equilibrium_temperature(hot["flux"]) + 0.5) == P.surface_temperature(hot["flux"]) - 500
    hot.update(temperature=P.surface_temperature(hot["flux"]))
    assert not sim.conserved(r).ok, "a giant with a greenhouse"


def test_parked_giants_merge_at_the_parking_orbit(sim):
    """Two giants at 0.05 au merge at 0.05 au: the merged orbit never leaves the two orbits by a
    rounding error (it did, and the next giant then parked a last digit beside the first)."""
    rng = random.Random(4)
    for _ in range(2000):
        m1, m2 = rng.uniform(0.01, 400), rng.uniform(0.01, 400)
        a1, a2 = rng.uniform(0.05, 30), rng.uniform(0.05, 30)
        assert P.merged_orbit(m1, P.A_PARK7, m2, P.A_PARK7) == P.A_PARK7
        assert P.merged_orbit(m1, a1, m2, a1) == a1
        merged = P.merged_orbit(m1, a1, m2, a2)
        assert min(a1, a2) <= merged <= max(a1, a2)
        assert (m1 + m2) * math.sqrt(merged) == pytest.approx(m1 * math.sqrt(a1) + m2 * math.sqrt(a2), rel=1e-12)
    assert P.merged_orbit(2.0, 3.0, 0.0, 9.0) == 3.0 and P.merged_orbit(1.0, 1.0, 1.0, 1.21) == pytest.approx(1.1025)
    # found by fuzzing: light cores run away, a chain of giants reaches the cavity one after the other
    r = sim.run(328496, star_mass=0.655, disc_mass=153.15, t_gas=3.0, core_threshold=0.3, gas_mass=10456.4)
    v = sim.conserved(r)
    assert v.ok, v
    assert sum(e["orbit"] == P.A_PARK7 and not e["ejected"] for e in r.events) >= 3
    for bodies in r.bodies:
        orbits = [b["a"] for b in bodies]
        assert orbits == sorted(set(orbits)) and sum(a == P.A_PARK7 for a in orbits) <= 1
        assert all(a >= P.A_PARK7 for a in orbits)


def test_the_types(rollouts):
    """Correction 7: five types, and rocky is no longer the rest class."""
    assert [P.planet_type(*x) for x in ((1, 0, 1, 2.7), (9.9, 2, 1, 2.7), (10, 2, 1, 2.7), (4.9, 0, 5, 2.7),
                                         (5, 4.9, 5, 2.7), (5, 5, 5, 2.7), (12, 12, 1, 2.7), (3, 0, 2.7, 2.7))] == \
        ["rocky", "rocky", "super_earth", "icy", "ice_giant", "gas", "gas", "rocky"]
    seen = set()
    for r in rollouts.values():
        frost = r.params["frost_line"]
        for s, bodies in zip(r.steps, r.bodies):
            planets = [b for b in bodies if b["solid"] + b["gas"] >= P.PLANET_MIN]
            assert s["planets"] == len(planets) == sum(s[k] for k in P.TYPE_COUNTS7.values())
            for ty, key in P.TYPE_COUNTS7.items():
                assert s[key] == sum(b["type"] == ty for b in planets)
            assert s["largest_mass"] == max(b["solid"] + b["gas"] for b in bodies)
        for p in r.system:
            seen.add(p["type"])
            if p["type"] == "rocky":
                assert p["orbit"] <= frost and p["solid"] < 10 and p["gas"] < p["solid"]
            elif p["type"] == "super_earth":
                assert p["orbit"] <= frost and p["solid"] >= 10 and p["gas"] < p["solid"]
            elif p["type"] == "icy":
                assert p["orbit"] > frost and p["solid"] < 5
            elif p["type"] == "ice_giant":
                assert p["orbit"] > frost and p["solid"] >= 5 and p["gas"] < p["solid"]
            else:
                assert p["gas"] >= p["solid"] >= 8
        assert [p["orbit"] for p in r.system] == sorted(p["orbit"] for p in r.system)
        assert r.summary["total_planet_mass"] == pytest.approx(sum(p["mass"] for p in r.system))
    assert seen == set(P.TYPES7)


def test_planets_eject_each_other_and_mergers_keep_angular_momentum(rollouts):
    """Corrections 2 and 9: a meeting is decided by the Safronov number of the heaviest body."""
    assert P.safronov(1.0, 1.0, 1.0) == pytest.approx(0.141, abs=5e-4), "the earth"
    assert P.safronov(317.8, 5.2, 1.0) > 20 and P.safronov(0.0, 5.0, 1.0) == 0.0
    assert P.safronov(8.0, 2.0, 1.0) == pytest.approx(4 * 2 * P.safronov(1.0, 1.0, 1.0))
    assert P.scatters(0.6, 10.0, 1.0) == "yes" and P.scatters(0.5, 10.0, 1.0) == "no" == P.scatters(1.0, 1.0, 1.0)
    assert P.merged_orbit(1.0, 1.0, 1.0, 1.21) == pytest.approx(1.1025) and P.merged_orbit(2.0, 3.0, 0.0, 9.0) == pytest.approx(3.0)
    mergers = ejections = 0
    for r in rollouts.values():
        assert r.events and [e["t"] for e in r.events] == sorted(e["t"] for e in r.events)
        for e in r.events:
            assert set(e) == {"t", "step", "ids", "kept", "mass_before", "theta", "members", "debris", "gas_ejected",
                              "ejected", "orbit", "mass_after"}
            t = e["step"]
            assert r.steps[t - 1]["time"] < e["t"] <= r.steps[t]["time"] + 1e-12
            assert len(e["ids"]) >= 2 and e["kept"] in e["ids"] and e["ids"] == sorted(set(e["ids"]))
            assert set(e["ids"]) <= {b["id"] for b in r.bodies[t - 1]}
            masses = {m["id"]: m["solid"] + m["gas"] for m in e["members"]}
            orbits = {m["id"]: m["a"] for m in e["members"]}
            assert masses[e["kept"]] == max(masses.values())
            if e["ejected"]:
                ejections += 1
                assert e["theta"] >= 1 and e["orbit"] == orbits[e["kept"]] and e["mass_after"] == masses[e["kept"]]
                assert e["ejected"] == sorted(set(e["ids"]) - {e["kept"]}) and e["debris"] > 0
                assert e["debris"] == pytest.approx(sum(m["solid"] + m["zone"] for m in e["members"] if m["id"] != e["kept"]))
            else:
                mergers += 1
                assert e["theta"] < 1 and e["debris"] == 0 == e["gas_ejected"]
                assert e["mass_after"] == pytest.approx(sum(masses.values()), rel=1e-12)
                assert e["mass_after"] * math.sqrt(e["orbit"]) == pytest.approx(
                    sum(masses[i] * math.sqrt(orbits[i]) for i in masses), rel=1e-12)
                assert min(orbits.values()) <= e["orbit"] <= max(orbits.values())
        gone = 24 - len(r.bodies[-1])
        assert gone == sum(len(e["ids"]) - 1 for e in r.events) >= r.summary["ejected"] > 0
    assert mergers > 200 and ejections > 200


def test_gas_giants_migrate(sim):
    """Correction 6: a gas giant drifts inward while the disc has gas, and parks at 0.05 au."""
    assert P.migrated_orbit(5.0, 1.0, 5.0, 0.1, 3.0) == 5.0, "no gas, no drift"
    assert P.migrated_orbit(0.04, 1.0, 0.0, 0.1, 3.0) == 0.04 and P.migrated_orbit(0.06, 1.0, 0.0, 1.0, 3.0) == P.A_PARK7
    a = P.migrated_orbit(5.0, 1.0, 1.0, 0.1, 3.0)
    assert 4.9 < a < 5.0 and P.migrated_orbit(5.0, 1.0, 2.5, 0.1, 3.0) > a, "the drift fades with the gas"
    assert P.migrated_orbit(a, 1.0, 1.1, 0.1, 3.0) == pytest.approx(P.migrated_orbit(5.0, 1.0, 1.0, 0.2, 3.0), rel=1e-12)
    assert 5.0 - P.migrated_orbit(5.0, 1.5, 1.0, 0.1, 3.0) > 5.0 - a
    # at t = 0 and 1 au the drift time is TAU_MIG7: da = -a dt / tau
    assert 1.0 - P.migrated_orbit(1.0, 1.0, 0.0, 1e-4, 1e9) == pytest.approx(1e-4 / P.TAU_MIG7, rel=1e-3)
    heavy = [sim.run(seed, star_mass=1.0, disc_mass=300.0, t_gas=10.0) for seed in SEEDS]
    assert all(sim.conserved(r).ok for r in heavy)
    assert sum(r.summary["inner_giants"] >= 1 for r in heavy) >= 30
    hot = [p for r in heavy for p in r.system if p["type"] == "gas" and p["orbit"] <= 0.1]
    assert len(hot) >= 5 and min(p["orbit"] for p in hot) == P.A_PARK7, "hot jupiters, parked at the cavity"
    assert all(b["a"] >= P.A_PARK7 for r in heavy for bodies in r.bodies for b in bodies)
    for r in heavy[:10]:  # between events only a body with gas moves, inward, while the disc has gas
        for t in range(1, len(r.bodies)):
            met = {i for e in r.events if e["step"] == t for i in e["ids"]}
            before = {b["id"]: b for b in r.bodies[t - 1]}
            for b in r.bodies[t]:
                if b["id"] not in met and b["a"] != before[b["id"]]["a"]:
                    assert b["a"] < before[b["id"]]["a"] and b["gas"] > 0 and r.steps[t - 1]["gas_present"] == "yes"
    short = [sim.run(seed, star_mass=1.0, disc_mass=300.0, t_gas=1.0) for seed in SEEDS[:10]]
    assert all(r.summary["gas_giants"] == 0 for r in short), "without time no giant, and nothing drifts"
    # round 6 with the same star, disc and gas: a merger may carry a giant a step inside the frost line, no further
    old = [P.simulation().run(seed, m_star=1.0, disc_mass=300.0, t_gas=10.0) for seed in SEEDS]
    assert min(b["a"] / r.params["r_frost"] for r in old for b in r.bodies[-1] if b["type"] == "gas") > 0.9


def test_heavier_discs_and_longer_lived_gas_grow_more_giants(sim):
    def mean_gas(disc_mass: float, t_gas: float = 5.0) -> float:
        return statistics.mean(sim.run(seed, star_mass=1.0, disc_mass=disc_mass, t_gas=t_gas).summary["gas_giants"]
                               for seed in range(1, 21))

    light, middle, heavy, heaviest = mean_gas(30.0), mean_gas(150.0), mean_gas(225.0), mean_gas(300.0)
    assert light == 0 and light <= middle < heavy < heaviest and heaviest >= 1.5
    assert mean_gas(300.0, 1.0) == 0 < mean_gas(300.0, 3.0) < mean_gas(300.0, 10.0)


def test_the_gas_budget_may_be_given(sim):
    """Correction 8: with ``gas_mass`` the gas no longer follows the solids."""
    a = sim.run(1, disc_mass=140.0, gas_mass=9860.0)
    b = sim.run(1, disc_mass=280.0, gas_mass=9860.0)
    assert a.params["gas_mass"] == 9860.0 and "gas_mass" not in sim.run(1, disc_mass=140.0).params
    assert a.params["gas_initial"] == pytest.approx(P.GAS_REACH * 9860.0) == pytest.approx(b.params["gas_initial"])
    assert a.steps[0]["gas_left"] == pytest.approx(a.params["gas_initial"])
    assert P.zone_gas(140.0, 5, 9860.0) == P.GAS_REACH * 9860.0 * P.ZONE_GAS7[5]
    old, older = sim.run(1, disc_mass=140.0), sim.run(1, disc_mass=280.0)
    assert older.params["gas_initial"] == pytest.approx(2 * old.params["gas_initial"]), "round 6's budget: 100 times the rock"
    assert P.zone_gas(140.0, 5) == pytest.approx(P.GAS_REACH * P.GAS_TO_ROCK * P.zone_solids(140.0, 5))
    assert P.zone_gas(140.0, 15) == pytest.approx(P.GAS_REACH * P.GAS_TO_ROCK * P.zone_solids(140.0, 15) / P.ICE_FACTOR)
    masses = []
    for seed in SEEDS:
        r = sim.run(seed, star_mass=1.0, disc_mass=300.0, t_gas=6.0, gas_mass=9700.0)
        assert sim.conserved(r).ok
        masses += [p["mass"] for p in r.system if p["type"] == "gas"]
    assert len(masses) >= 40 and 20 < min(masses) and max(masses) < 1500 and 60 < statistics.median(masses) < 600


# ---------------------------------------------------------------------------------------------
# what the systems look like (the review's targets, on the canonical seeds 1 to 40)


def test_typical_systems(rollouts):
    finals = [r.summary for r in rollouts.values()]
    counts = [f["planets"] for f in finals]
    assert all(2 <= c <= 10 for c in counts) and 5 <= statistics.median(counts) <= 7
    assert sum(f["rocky"] >= 1 for f in finals) >= 36 and sum(f["ice_giants"] >= 1 for f in finals) >= 30
    assert sum(f["habitable"] >= 1 for f in finals) >= 25 and all(0 <= f["habitable"] <= 2 for f in finals)
    assert {s["stage"] for r in rollouts.values() for s in r.steps} == set(P.STAGES)
    for r in rollouts.values():
        assert [s["time"] for s in r.steps] == [float(t) for t in P.TIMES]
        assert r.steps[0]["stage"] == "dust" and r.steps[0]["planets"] == 0
        assert next(s["stage"] for s in r.steps if s["stage"] != "dust") == "embryos"
        assert r.steps[-1]["stage"] in ("clearing", "done")
        order = [P.STAGES.index(s["stage"]) for s in r.steps]
        assert all(b >= a or (a, b) == (4, 3) for a, b in zip(order, order[1:]))
        assert max(s["planets"] for s in r.steps) <= 24
        for s in r.steps:
            assert s["gas_present"] == ("yes" if s["time"] < r.params["t_gas"] else "no")
            assert 0 <= s["innermost_au"] <= s["outermost_au"] <= 30 * math.sqrt(r.params["luminosity"])


def test_no_body_piles_up_solids(rollouts):
    """Round 6 ended nine of its first 30 runs with a solid body above 100 earth masses."""
    solids = [p["solid"] for r in rollouts.values() for p in r.system]
    ice = [p["solid"] for r in rollouts.values() for p in r.system if p["type"] == "ice_giant"]
    cores = [p["solid"] for r in rollouts.values() for p in r.system if p["type"] == "gas"]
    assert max(solids) < 40 and len(ice) >= 60 and 7 < statistics.median(ice) < 15
    assert cores and 10 * 0.99 <= min(cores) and max(cores) < 35, "giant cores of 10 to 35 earth masses"
    old = [b["solid"] for seed in SEEDS for b in P.rollout(seed).bodies[-1]]
    assert max(old) > 100
    debris = [r.steps[-1]["debris"] for r in rollouts.values()]
    assert 0.1 < min(debris) and max(debris) < 0.9 and 0.5 < statistics.median(debris) < 0.8, "the open cost"
    assert all(5 <= r.summary["ejected"] <= 23 for r in rollouts.values())


def test_gas_giants_are_rarer_and_weigh_about_as_much_as_real_ones(rollouts):
    with_giant = [r for r in rollouts.values() if r.summary["gas_giants"] >= 1]
    old = sum(P.rollout(seed).summary["gas"] >= 1 for seed in SEEDS)
    assert 1 <= len(with_giant) <= 12 and old >= 15, "about one run in ten, not one in two"
    masses = [p["mass"] for r in with_giant for p in r.system if p["type"] == "gas"]
    assert 20 < min(masses) and max(masses) < 600 and 60 < statistics.median(masses) < 320, "Saturn 95, Jupiter 318"
    assert all(r.params["disc_mass"] >= 100 and r.params["t_gas"] >= 1.3 for r in with_giant)
    light = [r for r in rollouts.values() if r.params["disc_mass"] < 100]
    assert len(light) >= 8 and all(s["gas_giants"] == 0 for r in light for s in r.steps)
    assert sum(r.summary["inner_giants"] for r in rollouts.values()) <= 3


# ---------------------------------------------------------------------------------------------
# the real-world numbers the "Rules 7" section states, recomputed from the code's own rules


def test_the_stated_reference_values():
    # the habitable zone of the sun and the climates of the solar system's orbits
    assert round(P.hz_inner(1.0), 3) == 0.950 and round(P.hz_outer(1.0), 3) == 1.676
    assert P.surface_temperature(1.0) == 288 and P.surface_temperature(1 / 1.524 ** 2) == 278, "earth, Mars' orbit"
    assert P.surface_temperature(P.S_MAX_GREENHOUSE) == 273 and P.surface_temperature(P.S_RUNAWAY) == 294
    assert P.surface_temperature(0.3559) == 197, "just beyond the outer edge"
    assert round(P.equilibrium_temperature(1.0)) == 255, "the earth's effective temperature"
    assert round(P.equilibrium_temperature(1 / 5.203 ** 2)) == 112 and round(P.equilibrium_temperature(1 / 30.07 ** 2)) == 46
    # Safronov numbers: the earth, Jupiter with the earth's density and with its real radius, 1 at 10 au
    assert round(P.safronov(1.0, 1.0, 1.0), 3) == 0.141
    assert round(P.safronov(317.8, 5.203, 1.0)) == 34
    assert round(P.safronov(317.8, 5.203, 1.0) * 6371.0 * math.cbrt(317.8) / 69911.0) == 21
    assert round(((1.0 / 0.14105 / 10.0) ** 1.5), 1) == 0.6
    # disc lifetimes: share of discs left at 3 and 6 Myr, against exp(-t / 2.5 Myr)
    assert round(math.exp(-(3 - P.T_GAS_SHIFT7) / P.T_GAS_MEAN7), 2) == 0.37
    assert round(math.exp(-(6 - P.T_GAS_SHIFT7) / P.T_GAS_MEAN7), 2) == 0.08
    assert round(math.exp(-3 / 2.5), 2) == 0.30 and round(math.exp(-6 / 2.5), 2) == 0.09
    # a thousandth of an earth mass of hydrogen on a planet of the earth's size and gravity:
    # the earth's air, 5.15e18 kg of 5.972e24 kg, makes 1.013 bar
    bars = 1e-3 / (5.15e18 / 5.972e24) * 1.013
    assert 900 < bars < 1300, "about a thousand bars"
    # the earth per sun ratio round 6 uses, against the measured one (GM_sun / GM_earth)
    assert abs(P.EARTH_PER_SUN / (1.32712440018e20 / 3.986004418e14) - 1 - 2.5e-5) < 1e-6
