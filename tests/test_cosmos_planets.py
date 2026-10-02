import copy
import math
import random
import statistics
import time

import numpy as np
import pytest

from haishool.cosmos import Simulation, close
from haishool.cosmos.planets import (COUNT_KEYS, HABITABLE, INPUT_KEYS, KEYS, LEDGER_KEYS, PARAM_KEYS, PLANET_MIN,
                                     STAGES, STATE_KEYS, SUBSTEPS, SUMMARY_KEYS, T_EPS, TIMES, WORD_KEYS, Planets,
                                     _Disc, _Rng, dense_value, frost_line, lines, param_value, params_line,
                                     random_params, simulation)
from haishool.truth import Gate, is_dense, parse_num, split_line

SEEDS = list(range(1, 41))


def n_tokens(text: str) -> int:
    """As ``haishool.student.tokens`` counts them (that module needs torch, so not imported)."""
    return len(text.replace(".", " . ").split())


def merged_ids(r, t: int) -> set[int]:
    """The bodies that took part in a merger between output steps ``t - 1`` and ``t``."""
    return {i for e in r.events if e["step"] == t for i in e["ids"]}


@pytest.fixture(scope="module")
def sim():
    return Planets()


@pytest.fixture(scope="module")
def rollouts(sim):
    return {seed: sim.rollout(seed) for seed in SEEDS}


@pytest.fixture(scope="module")
def all_lines(rollouts):
    return [ln for r in rollouts.values() for ln in lines(r)]


def test_protocols():
    s = simulation()
    assert isinstance(s, Simulation) and isinstance(s, Gate)
    assert s.sim == s.topic == "planets" and s.records() == []
    assert simulation() is s
    for key in PARAM_KEYS + STATE_KEYS + LEDGER_KEYS + SUMMARY_KEYS:
        assert key in KEYS, key
    assert COUNT_KEYS | WORD_KEYS <= set(KEYS) and INPUT_KEYS <= set(PARAM_KEYS)


def test_same_seed_same_lines():
    a, b = Planets(), Planets()
    for seed in (1, 7, 4321):
        ra, rb = a.rollout(seed), b.rollout(seed)
        assert ra.steps == rb.steps and ra.summary == rb.summary and ra.params == rb.params
        assert ra.bodies == rb.bodies and ra.events == rb.events
        assert [ln.text for ln in lines(ra)] == [ln.text for ln in lines(rb)]
    assert [ln.text for ln in lines(a.rollout(1))] != [ln.text for ln in lines(a.rollout(2))]
    assert a.run(5, disc_mass=200.0).steps == b.run(5, disc_mass=200.0).steps
    assert a.run(5, disc_mass=200.0).steps != a.run(6, disc_mass=200.0).steps


def test_forty_seeds_replay_exactly(rollouts):
    """Two runs of a seed give the same step dicts, bodies and mergers, to the last bit."""
    fresh = Planets()
    for seed, r in rollouts.items():
        again = fresh.run(seed, **random_params(random.Random(seed)))
        assert again.steps == r.steps and again.bodies == r.bodies and again.events == r.events
        assert again.summary == r.summary and again.params == r.params
        assert [repr(v) for s in again.steps for v in s.values()] == [repr(v) for s in r.steps for v in s.values()]


def test_values_are_plain_python_numbers_and_words(rollouts):
    for r in rollouts.values():
        for row in r.steps + [r.summary, r.params]:
            for key, value in row.items():
                assert type(value) in (float, int, str), (key, type(value))
                assert (type(value) is int) == (key in COUNT_KEYS), key
                assert (type(value) is str) == (key in WORD_KEYS), key
                if type(value) is float:
                    assert math.isfinite(value) and value >= 0, (key, value)
        for bodies in r.bodies:
            for b in bodies:
                assert [type(b[k]) for k in ("id", "a", "solid", "gas", "type")] == [int, float, float, float, str]


def test_generate_is_deterministic(sim):
    one = [ln.text for ln in sim.generate(random.Random(11), 3)]
    two = [ln.text for ln in Planets().generate(random.Random(11), 3)]
    assert one == two and len(one) > 1000
    assert one != [ln.text for ln in sim.generate(random.Random(12), 3)]


def test_random_numbers_are_the_raw_pcg64_stream():
    """Pinned, so that another numpy cannot change the rollouts unnoticed; and equal to what
    ``default_rng`` gives on the numpy this was built with."""
    rng = _Rng(3)
    assert [rng.random(), rng.random()] == [0.08564916714362436, 0.2368105065960997]
    assert rng.uniform(0.6, 1.4, size=3).tolist() == [1.2410195721651176, 1.0657296288514941, 0.6753029137923193]
    for seed in (0, 3, 1874, 123456789):
        ours, numpys = _Rng(seed), np.random.default_rng(seed)
        assert ours.uniform(0.6, 1.4, size=24).tolist() == numpys.uniform(0.6, 1.4, size=24).tolist()
        assert [ours.random() for _ in range(50)] == [float(numpys.random()) for _ in range(50)]
        assert ours.uniform(0.02, 0.2) == float(numpys.uniform(0.02, 0.2))
        assert 0.0 <= ours.random() < 1.0 and ours.uniform(0.6, 1.4, size=24).dtype == np.float64


def test_pinned_lines_of_seed_3(sim):
    """The lines the module docstring shows, word for word."""
    r = sim.rollout(3)
    texts = [ln.text for ln in lines(r)]
    assert texts[0] == ("planets seed 3 params. m_star 0 point 7 4. disc_mass 1 6 7 point 8. t_gas 4 point 3 3. "
                        "core_threshold 1 0. r_frost 1 point 5 9. luminosity 0 point 3 4 9.")
    assert ("planets seed 3 step 1 4. time 4. planets 2 2. rocky 1 5. ice 5. gas 2. largest_mass 4 0 point 8. "
            "innermost_au 0 point 3 3. outermost_au 2 7 point 3. debris 0. gas_present yes. "
            "stage giants_forming.") in texts
    for text in ("q planets seed 3 step 1 4 gas. a 2.", "q planets seed 3 step 1 4 next stage. a clearing.",
                 "q planets seed 3 step 1 4 next planets. a 2 0.", "q planets seed 3 final habitable. a 1.",
                 "q planets seed 3 final planets. a 7.", "q planets seed 3 final largest_mass. a 7 0 point 5.",
                 "q planets seed 3 final total_planet_mass. a 2 0 2."):
        assert text in texts, text
    assert r.steps[14]["largest_mass"] == pytest.approx(40.757372198830154, rel=1e-9)
    assert r.summary["total_planet_mass"] == pytest.approx(201.7464665430249, rel=1e-9)
    assert len(r.events) == 17 and r.events[0]["ids"] == [14, 15] and r.events[0]["step"] == 14


def test_a_change_in_the_twelfth_digit_moves_no_line(sim, rollouts):
    """Another machine's exp or cbrt may differ in the sixteenth digit. A change ten thousand
    times larger must not move a single line: no decision sits on a knife's edge."""
    for seed, r in rollouts.items():
        base = [ln.text for ln in lines(r)]
        p = random_params(random.Random(seed))
        for f in (1 + 1e-12, 1 - 1e-12):
            nudged = sim.run(seed, **dict(p, m_star=p["m_star"] * f, disc_mass=p["disc_mass"] / f))
            assert nudged.steps != r.steps
            assert [ln.text for ln in lines(nudged)] == base, (seed, f)


def test_gas_ends_by_rule_when_a_mid_step_time_meets_t_gas():
    """Mid-step times are multiples of 0.0025 Myr, ``t_gas`` of 0.01: they meet. In floating
    point 3.0 + 3 * 0.1 + 0.05 falls just below 3.35 and 8.0 + 3 * 0.1 + 0.05 just above 8.35;
    the rule says the same for both: at ``t_gas`` the gas is gone."""
    assert 3.0 + 3 * 0.1 + 0.05 < 3.35 and 8.0 + 3 * 0.1 + 0.05 > 8.35
    assert 0 < T_EPS < 0.0025 / 2
    mids = [t0 + i * ((t1 - t0) / SUBSTEPS) + (t1 - t0) / SUBSTEPS / 2
            for t0, t1 in zip(TIMES, TIMES[1:]) for i in range(SUBSTEPS)]
    met = 0
    for hundredths in range(100, 1001):
        disc = _Disc(_Rng(1), 1.0, 100.0, hundredths / 100)
        for t in mids + TIMES:
            exact = round(t * 400) < hundredths * 4  # in units of 0.0025 Myr
            assert disc.gas_at(t) == exact, (t, hundredths)
            met += round(t * 400) == hundredths * 4
    assert met >= 80, "the times do meet"
    disc = _Disc(_Rng(1), 1.0, 100.0, 3.35)
    assert not disc.gas_at(3.0 + 3 * 0.1 + 0.05) and disc.gas_at(3.25) and disc.gas_available(3.35).sum() == 0


def test_random_params_stay_in_range():
    for seed in range(200):
        p = random_params(random.Random(seed))
        assert p == random_params(random.Random(seed))
        assert 0.5 <= p["m_star"] <= 1.5 and 10 <= p["disc_mass"] <= 300 and 1 <= p["t_gas"] <= 10
        assert p["core_threshold"] == 10.0
        assert p["m_star"] == round(p["m_star"], 2) and p["disc_mass"] == round(p["disc_mass"], 1)
    assert random_params(random.Random(3)) == {"m_star": 0.74, "disc_mass": 167.8, "t_gas": 4.33,
                                               "core_threshold": 10.0}


def test_run_rejects_impossible_parameters(sim):
    for bad in ({"m_star": 0.0}, {"m_star": -1.0}, {"disc_mass": 0.0}, {"disc_mass": float("nan")},
                {"t_gas": -1.0}, {"t_gas": float("inf")}, {"core_threshold": 0.0}):
        with pytest.raises(ValueError):
            sim.run(1, **bad)
    no_gas = sim.run(1, disc_mass=300.0, t_gas=0.0)
    assert sim.conserved(no_gas).ok and no_gas.summary["gas"] == 0
    assert all(s["gas_captured"] == 0 and s["gas_left"] == 0 and s["gas_present"] == "no" for s in no_gas.steps)
    assert no_gas.steps[0]["stage"] == "dust"
    crumbs = sim.run(1, disc_mass=0.5)
    assert sim.conserved(crumbs).ok and crumbs.summary["largest_mass"] < 1


def test_every_line_is_dense_and_short(all_lines):
    assert len(all_lines) > 20000
    for ln in all_lines:
        assert is_dense(ln.text), ln.text
        assert n_tokens(ln.text) <= 80, ln.text
        assert ln.topic == "planets"
        if ln.kind != "record":
            assert "." not in ln.prompt and "." not in ln.answer
            assert split_line(ln.text) == (ln.prompt, ln.answer)
            assert ln.kind in ("fact", "calc")
    assert max(n_tokens(ln.text) for ln in all_lines) <= 70


def test_lines_of_a_long_seed_stay_short(sim):
    for seed in (9998, 123456789):
        for ln in lines(sim.rollout(seed)):
            assert is_dense(ln.text) and n_tokens(ln.text) <= 80, ln.text
            if ln.kind != "record":
                assert sim.check(ln.prompt, ln.answer).ok, ln.text


def test_line_kinds(rollouts):
    r = rollouts[3]
    ls = lines(r)
    records = [ln for ln in ls if ln.kind == "record"]
    assert records[0].text == params_line(r).text and records[0].text.startswith("planets seed 3 params. m_star ")
    assert len(records) == 1 + len(TIMES)
    assert records[1].text == ("planets seed 3 step 0. time 0. planets 0. rocky 0. ice 0. gas 0. largest_mass 0. "
                               "innermost_au 0. outermost_au 0. debris 0. gas_present yes. stage dust.")
    prompts = {ln.prompt for ln in ls if ln.kind != "record"}
    assert len(prompts) == len(TIMES) * len(STATE_KEYS) + (len(TIMES) - 1) * len(STATE_KEYS) + len(SUMMARY_KEYS)
    assert len(ls) == 687 == len({ln.text for ln in ls})
    assert "planets seed 3 step 1 4 gas" in prompts and "planets seed 3 step 1 4 next stage" in prompts
    assert "planets seed 3 final habitable" in prompts
    assert not any(k in ln.text for ln in ls for k in LEDGER_KEYS), "the ledger is not trained"
    assert len(lines(r, every=5)) < len(ls) / 4


def test_parameters_are_written_as_given(sim):
    assert param_value("disc_mass", 253.5) == "2 5 3 point 5" and param_value("m_star", 0.74) == "0 point 7 4"
    assert param_value("t_gas", 4.0) == "4" and param_value("r_frost", 1.5936) == "1 point 5 9"
    text = params_line(sim.rollout(24)).text
    assert text == ("planets seed 2 4 params. m_star 1 point 2 1. disc_mass 2 5 3 point 5. t_gas 2 point 6 4. "
                    "core_threshold 1 0. r_frost 3 point 7 7. luminosity 1 point 9 5.")
    for seed in range(1, 200):
        p = random_params(random.Random(seed))
        r = sim.rollout(seed)
        for key in INPUT_KEYS:
            assert parse_num(param_value(key, r.params[key])) == p[key], (seed, key)
            v = sim.check(f"planets seed {' '.join(str(seed))} params {key}", param_value(key, p[key]))
            assert v.ok and v.expected == param_value(key, p[key])
    assert sim.check("planets seed 2 4 params disc_mass", "2 5 3 point 5").ok
    assert not sim.check("planets seed 2 4 params disc_mass", "3 5 3 point 5").ok
    assert sim.check("planets seed 2 4 params disc_mass", "3 5 3 point 5").expected == "2 5 3 point 5"


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
    for ln in rng.sample(questions, 3000):
        key = ln.prompt.split()[-1]
        if key in WORD_KEYS:
            wrong = "no" if ln.answer == "yes" else "yes" if key == "gas_present" else \
                next(s for s in STAGES if s != ln.answer)
        else:
            wrong = "9 " + ln.answer  # a changed leading digit: far outside the 5 % tolerance
        v = sim.check(ln.prompt, wrong)
        assert not v.ok and v.expected == ln.answer, (ln.text, wrong, v)
    assert sim.check("planets seed 3 final gas", "many") == sim.check("planets seed 3 final gas", "many")
    assert not sim.check("planets seed 3 final gas", "many").ok
    assert sim.check("planets seed 3 final gas", "many").expected == "1"
    assert not sim.check("planets seed 3 step 1 4 stage", "giants").ok


def test_gate_rejects_a_changed_digit(sim, all_lines):
    """One digit of the answer changed: a count is then wrong; a measured number is wrong when
    the change is more than 5 % (the gate compares with the unrounded value, hence 4 and 6)."""
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
        if ln.prompt.split()[-1] in COUNT_KEYS or off > 0.06:
            assert not v.ok, (ln.text, wrong, v)
        elif off < 0.04:
            assert v.ok, (ln.text, wrong, v)
        rejected += not v.ok
    assert rejected > 2000


def test_gate_rejects_answers_that_are_not_finite_numbers(sim):
    for prompt in ("planets seed 3 final largest_mass", "planets seed 3 step 1 4 outermost_au",
                   "planets seed 3 final planets", "planets seed 3 params m_star"):
        for answer in ("1 e 9 9 9", "minus 1 e 9 9 9", "", "point", "e", "yes", "7 0 point 5 earth", "7 0 . 5",
                       "minus 7 0 point 5", "1 e 3 0 0"):
            v = sim.check(prompt, answer)
            assert not v.ok and v.expected is not None, (prompt, answer, v)
    assert sim.check("planets seed 3 final largest_mass", "1 e 9 9 9").reason == "not a number"
    assert sim.check("planets seed 3 step 0 largest_mass", "0").ok
    assert not sim.check("planets seed 3 step 0 largest_mass", "1").ok
    assert not sim.check("planets seed 3 step 0 planets", "1 e 9 9 9").ok


def test_tolerance_is_five_percent_for_numbers_and_exact_for_counts(sim, rollouts):
    r = rollouts[3]
    mass = r.summary["total_planet_mass"]
    assert sim.check("planets seed 3 final total_planet_mass", sim.rollout(3).value(mass * 1.03)).ok
    assert sim.check("planets seed 3 final total_planet_mass", sim.rollout(3).value(mass * 0.97)).ok
    assert not sim.check("planets seed 3 final total_planet_mass", sim.rollout(3).value(mass * 1.1)).ok
    assert not sim.check("planets seed 3 final total_planet_mass", sim.rollout(3).value(mass * 0.9)).ok
    n = r.summary["planets"]
    assert sim.check("planets seed 3 final planets", str(n)).ok
    assert not sim.check("planets seed 3 final planets", str(n + 1)).ok
    assert not sim.check("planets seed 3 final planets", str(n - 1)).ok


@pytest.mark.parametrize("prompt", [
    "turkey capital", "q spoon color", "spoon color", "carbon protons", "water molar_mass", "calc 1 2 plus 7",
    "solve 3 x plus 4 equals 1 9", "check 1 2 plus 7 equals 2 0", "balance h 2 plus o 2",
    "gravity seed 7 step 2 0 clumps", "gravity seed 7 final clumps", "gravity seed 7 step 2 0 next clumps",
    "nucleo seed 3 final helium_fraction", "nucleo seed 3 step 4 stage", "world seed 3 era planets",
    "chem seed 7 step 1 molecules", "chem seed 7 final water", "life seed 7 step 1 2 next population",
    "life seed 7 final stage", "planets", "planets seed 7", "planets seed 7 step 3",
    "planets seed 7 step 3 clumps", "planets seed 7 step 3 0 planets", "planets seed 7 step 2 9 next planets",
    "planets seed 7 final stage", "planets seed x final gas", "planets seed 7 final", "planets seed 7 params time",
    "planets seed 12 step 3 planets", "planets seed 7 step 3 next", "planets seed 7 step 3 planets rocky", "",
    "planets seed 0 7 final gas", "planets seed 7 step 0 3 planets", "planets seed 7 step 0 0 planets",
    "planets seed minus 7 final gas", "planets seed 7 point 5 final gas", "q planets seed 7 final gas",
    "planets seed 7 final gas extra", "planets seed 7 step 3 next next planets", "planets seed 7 era planets",
    "planets  seed 7 final clumps", "planets seed 7 final helium_fraction", "planets seed 7 params gas_initial",
])
def test_does_not_own_other_prompts(sim, prompt):
    assert not sim.owns(prompt)
    assert sim.check(prompt, "1").ok is False
    assert sim.check(prompt, "1").expected is None and sim.check(prompt, "1").reason == "not my question"


@pytest.mark.parametrize("prompt", [
    "planets seed 7 step 0 planets", "planets seed 7 step 2 9 stage", "planets seed 7 step 2 8 next gas_present",
    "planets seed 1 2 3 4 step 1 2 next largest_mass", "planets seed 7 final total_planet_mass",
    "planets seed 7 final habitable", "planets seed 7 params r_frost", "planets seed 7 step 4 zone_solids",
    "planets seed 0 final gas", "planets seed 7 step 1 0 debris",
])
def test_owns_its_prompts(sim, prompt):
    assert sim.owns(prompt)
    assert sim.check(prompt, "nonsense words").expected is not None


def test_solids_are_accounted_for_exactly(rollouts):
    for r in rollouts.values():
        for s in r.steps:
            assert abs(s["solids_in_bodies"] + s["zone_solids"] + s["debris_mass"] - r.params["disc_mass"]) <= 1e-9
            assert s["debris"] == pytest.approx(s["debris_mass"] / r.params["disc_mass"], rel=1e-12)
        assert r.steps[0]["zone_solids"] == pytest.approx(r.params["disc_mass"], abs=1e-9)
        assert sum(e["debris"] for e in r.events) == pytest.approx(r.steps[-1]["debris_mass"], abs=1e-9)


def test_gas_counts_masses_and_time(rollouts):
    for r in rollouts.values():
        assert [s["time"] for s in r.steps] == [float(t) for t in TIMES]
        assert len(r.steps) == 30 and r.steps[-1]["time"] == 50
        for prev, s, bodies in zip([None] + r.steps[:-1], r.steps, r.bodies):
            assert s["planets"] == s["rocky"] + s["ice"] + s["gas"]
            assert s["planets"] == sum(b["solid"] + b["gas"] >= PLANET_MIN for b in bodies)
            assert s["gas_captured"] + s["gas_left"] <= r.params["gas_initial"] + 1e-9
            assert s["gas_captured"] == pytest.approx(sum(b["gas"] for b in bodies), abs=1e-9)
            assert s["gas_present"] == ("yes" if s["time"] < r.params["t_gas"] else "no")
            assert s["stage"] in STAGES
            assert [b["a"] for b in bodies] == sorted(b["a"] for b in bodies), "bodies stay in order of orbit"
            assert 0 <= s["innermost_au"] <= s["outermost_au"] <= 30
            if s["gas_present"] == "no":
                assert s["gas_left"] == 0 and s["stage"] in ("clearing", "done")
            if prev is not None:
                # sums over merged zones may differ in the last place, hence the 1e-9
                assert s["time"] > prev["time"] and s["gas_captured"] >= prev["gas_captured"] - 1e-9
                assert s["debris"] >= prev["debris"] and s["zone_solids"] <= prev["zone_solids"] + 1e-9
                if prev["gas_present"] == "no":
                    assert s["gas_captured"] == pytest.approx(prev["gas_captured"], abs=1e-9), "the gas is gone"
        for t in range(1, len(r.bodies)):
            merged = merged_ids(r, t)
            before = {b["id"]: b["solid"] + b["gas"] for b in r.bodies[t - 1]}
            for b in r.bodies[t]:
                if b["id"] not in merged:
                    assert b["solid"] + b["gas"] >= before[b["id"]] - 1e-9
            assert set(before) - {b["id"] for b in r.bodies[t]} <= merged


def test_the_merger_log_fits_the_steps(rollouts):
    for r in rollouts.values():
        assert r.events, "every run has mergers"
        assert [e["t"] for e in r.events] == sorted(e["t"] for e in r.events)
        for e in r.events:
            assert set(e) == {"t", "step", "ids", "kept", "mass_before", "debris"}
            t = e["step"]
            assert r.steps[t - 1]["time"] < e["t"] <= r.steps[t]["time"] + 1e-12
            assert len(e["ids"]) >= 2 and e["kept"] in e["ids"] and e["ids"] == sorted(set(e["ids"]))
            assert 0 <= e["debris"] <= 0.2 * e["mass_before"]
            assert set(e["ids"]) <= {b["id"] for b in r.bodies[t - 1]}
            gone = set(e["ids"]) - {b["id"] for b in r.bodies[t]}
            assert len(gone) >= len(e["ids"]) - 1
        ids_left = {b["id"] for b in r.bodies[-1]}
        assert len(ids_left) == 24 - sum(len(e["ids"]) - 1 for e in r.events)


def test_conserved_passes_and_catches_tampering(sim, rollouts):
    for r in rollouts.values():
        v = sim.conserved(r)
        assert v.ok, v
    base = rollouts[3]

    def broken(change) -> bool:
        r = copy.deepcopy(base)
        change(r)
        return not sim.conserved(r).ok

    assert not broken(lambda r: None)
    assert broken(lambda r: r.steps[10].update(zone_solids=r.steps[10]["zone_solids"] + 1e-6))
    assert broken(lambda r: r.steps[10].update(solids_in_bodies=r.steps[10]["solids_in_bodies"] * 0.99))
    assert broken(lambda r: r.steps[29].update(debris=r.steps[29]["debris"] + 0.01))
    assert broken(lambda r: r.steps[12].update(planets=r.steps[12]["planets"] + 1))
    assert broken(lambda r: r.steps[12].update(time=r.steps[11]["time"]))
    assert broken(lambda r: r.steps[14].update(gas_captured=r.params["gas_initial"] * 2))
    assert broken(lambda r: r.steps[20].update(gas_captured=0.0))
    assert broken(lambda r: r.steps[5].update(stage="finished"))
    assert broken(lambda r: r.bodies[29][0].update(solid=r.bodies[29][0]["solid"] * 0.5))
    assert broken(lambda r: r.bodies[29].pop())
    assert broken(lambda r: r.events.clear())
    # the merger log itself is checked against the steps and the bodies
    with_debris = next(n for n, e in enumerate(base.events) if e["debris"] > 0)
    assert broken(lambda r: r.events[with_debris].update(debris=r.events[with_debris]["debris"] + 0.001))
    assert broken(lambda r: r.events[with_debris].update(debris=0.0))
    assert broken(lambda r: r.events[-1].update(mass_before=r.events[-1]["mass_before"] * 1.5))
    assert broken(lambda r: r.events[0].update(step=0))
    assert broken(lambda r: r.events[0].update(kept=99))
    assert broken(lambda r: r.events.pop(0))
    # moved solids from one body to another: the totals still add up, the bodies do not
    def shift(r):
        r.bodies[29][0]["solid"] -= 0.5
        r.bodies[29][1]["solid"] += 0.5
    assert broken(shift)


def test_summary_matches_the_last_step(rollouts):
    for r in rollouts.values():
        last, bodies = r.steps[-1], r.bodies[-1]
        assert list(r.summary) == list(SUMMARY_KEYS)
        for k in ("planets", "rocky", "ice", "gas", "largest_mass"):
            assert r.summary[k] == last[k]
        planets = [b for b in bodies if b["solid"] + b["gas"] >= PLANET_MIN]
        root_l = math.sqrt(r.params["m_star"] ** 3.5)
        lo, hi = HABITABLE[0] * root_l, HABITABLE[1] * root_l
        assert r.summary["habitable"] == sum(b["type"] == "rocky" and lo <= b["a"] <= hi for b in planets)
        assert r.summary["total_planet_mass"] == pytest.approx(sum(b["solid"] + b["gas"] for b in planets))
        assert r.summary["largest_mass"] == pytest.approx(max(b["solid"] + b["gas"] for b in bodies))
        assert r.params["r_frost"] == pytest.approx(2.7 * root_l) == pytest.approx(frost_line(r.params["m_star"]))
        assert r.params["luminosity"] == pytest.approx(r.params["m_star"] ** 3.5)


def test_no_zone_centre_sits_on_the_frost_line_or_a_habitable_edge():
    """A zone centre that met the frost line of some star mass to the last digit would make its
    side of the line a matter of rounding. The nearest miss is 1.6e-4 of the distance."""
    centres = _Disc(_Rng(1), 1.0, 100.0, 3.0).a
    assert len(centres) == 24 and centres.dtype == np.float64
    nearest = 1.0
    for hundredths in range(50, 151):
        root_l = math.sqrt((hundredths / 100) ** 3.5)
        for line in (frost_line(hundredths / 100), HABITABLE[0] * root_l, HABITABLE[1] * root_l):
            nearest = min(nearest, float(np.abs(centres - line).min() / line))
    assert nearest > 1e-4


def test_giants_form_only_beyond_the_frost_line(sim, rollouts):
    seen = 0
    for r in rollouts.values():
        frost = r.params["r_frost"]
        for t, bodies in enumerate(r.bodies):
            merged = merged_ids(r, t)
            before = {b["id"]: b for b in r.bodies[t - 1]} if t else {}
            for b in bodies:
                if b["type"] in ("gas", "ice"):
                    seen += 1
                    assert b["a"] > frost, (r.seed, t, b)
                if b["id"] in before and b["id"] not in merged and b["gas"] > before[b["id"]]["gas"] + 1e-9:
                    assert b["a"] > frost, "gas is only captured beyond the frost line"
                if b["a"] <= frost and b["gas"] > 0:
                    # an inner body only holds gas it brought along in a merger
                    assert any(b["id"] in e["ids"] for e in r.events), (r.seed, t, b)
    assert seen > 1000


def test_a_merger_can_carry_a_giant_just_inside_the_frost_line(sim):
    """Seed 1874, the one case in seeds 1-2000: the giant formed outside, merged with a rocky
    neighbour and came to rest 0.04 au inside. It formed outside, and the gate still holds."""
    r = sim.rollout(1874)
    frost = r.params["r_frost"]
    assert sim.conserved(r).ok
    inside = [(t, b) for t, bodies in enumerate(r.bodies) for b in bodies if b["type"] == "gas" and b["a"] <= frost]
    assert len(inside) == 1
    t, b = inside[0]
    assert frost - 0.1 < b["a"] <= frost
    events = [e for e in r.events if b["id"] in e["ids"] and e["step"] == t]
    assert events, "only a merger moves a body"
    was = next(x for x in r.bodies[t - 1] if x["id"] == b["id"])
    assert was["a"] > frost and was["gas"] > 0, "it was already capturing gas beyond the frost line"
    assert not any(x["type"] in ("gas", "ice") and x["a"] <= frost for x in r.bodies[-1])


def test_heavier_discs_grow_more_giants(sim):
    def mean_gas(disc_mass: float) -> float:
        return statistics.mean(sim.run(seed, m_star=1.0, disc_mass=disc_mass, t_gas=5.0).summary["gas"]
                               for seed in range(1, 21))

    light, middle, heavy, heaviest = mean_gas(30.0), mean_gas(150.0), mean_gas(225.0), mean_gas(300.0)
    assert light == 0
    assert light <= middle < heavy <= heaviest and heavy >= 1
    # without gas no giant can form, whatever the disc
    quick = [sim.run(seed, m_star=1.0, disc_mass=300.0, t_gas=1.0).summary["gas"] for seed in range(1, 11)]
    slow = [sim.run(seed, m_star=1.0, disc_mass=300.0, t_gas=10.0).summary["gas"] for seed in range(1, 11)]
    assert sum(quick) < sum(slow) and statistics.mean(slow) >= 2


def test_light_discs_grow_no_gas_giant(sim, rollouts):
    """No gas giant below 100 earth masses of solids, whatever the star and the gas lifetime."""
    light = [r for r in rollouts.values() if r.params["disc_mass"] < 100]
    assert len(light) >= 8 and all(s["gas"] == 0 for r in light for s in r.steps)
    for seed in SEEDS:
        p = random_params(random.Random(seed))
        r = sim.run(seed, m_star=p["m_star"], disc_mass=99.0, t_gas=10.0)
        assert r.summary["gas"] == 0 and max(s["gas"] for s in r.steps) == 0, seed
    heavy = [r for r in rollouts.values() if r.params["disc_mass"] >= 200]
    assert sum(r.summary["gas"] >= 1 for r in heavy) >= 0.6 * len(heavy)


def test_typical_systems(rollouts):
    finals = [r.summary for r in rollouts.values()]
    counts = [f["planets"] for f in finals]
    assert 4 <= statistics.median(counts) <= 8
    assert all(2 <= c <= 10 for c in counts)
    assert sum(f["rocky"] >= 1 for f in finals) >= 0.9 * len(finals)
    assert any(f["gas"] >= 1 for f in finals) and any(f["gas"] == 0 for f in finals)
    assert any(f["ice"] >= 1 for f in finals)
    assert all(0 <= f["habitable"] <= 3 for f in finals) and sum(f["habitable"] >= 1 for f in finals) >= 30
    stages = {s["stage"] for r in rollouts.values() for s in r.steps}
    assert stages == set(STAGES)
    for r in rollouts.values():
        assert r.steps[0]["stage"] == "dust" and r.steps[0]["planets"] == 0
        assert next(s["stage"] for s in r.steps if s["stage"] != "dust") == "embryos"
        assert r.steps[-1]["stage"] in ("clearing", "done")
        assert 0 <= r.steps[-1]["debris"] < 0.5
        assert max(s["planets"] for s in r.steps) <= 24
        order = [STAGES.index(s["stage"]) for s in r.steps]
        # only a finished system may go back, to clearing
        assert all(b >= a or (a, b) == (4, 3) for a, b in zip(order, order[1:]))


def test_where_the_planets_are_and_what_they_weigh(rollouts):
    """Rock near the star, gas and ice far out; gas giants about as heavy as real ones."""
    rocky_in = rocky_out = giants = 0
    gas_masses, gas_orbits = [], []
    for r in rollouts.values():
        frost = r.params["r_frost"]
        for b in r.bodies[-1]:
            mass = b["solid"] + b["gas"]
            if mass < PLANET_MIN:
                continue
            if b["type"] == "rocky":
                rocky_in += b["a"] <= frost
                rocky_out += b["a"] > frost
                assert b["a"] <= frost or b["solid"] < 5, "a rocky body beyond the frost line is a small one"
            else:
                giants += 1
                assert b["a"] > frost, (r.seed, b)
            if b["type"] == "gas":
                gas_masses.append(mass)
                gas_orbits.append(b["a"] / frost)
                assert b["gas"] >= b["solid"] >= 10 * 0.8, "a gas giant has a core of about 10 earth masses or more"
            if b["type"] == "ice":
                assert b["solid"] >= 5 and b["gas"] < b["solid"]
    assert rocky_in >= 0.85 * (rocky_in + rocky_out) and giants > 80
    assert len(gas_masses) >= 20 and 20 < min(gas_masses) and max(gas_masses) < 3000
    assert 95 < statistics.median(gas_masses) < 1000, "between Saturn and three Jupiters"
    assert 1 < min(gas_orbits) and statistics.median(gas_orbits) < 6


def test_numbers_use_three_significant_digits(all_lines):
    assert dense_value(1052.53) == "1 0 5 0" and dense_value(0.03234) == "0 point 0 3 2 3"
    assert dense_value(7) == "7" and dense_value(0.0) == "0" and dense_value("done") == "done"
    assert dense_value(0.000412) == "4 point 1 2 e minus 4" and dense_value(40.84) == "4 0 point 8"
    assert dense_value(999.7) == "1 0 0 0" and dense_value(0.9996) == "1"
    for ln in all_lines:
        if ln.kind == "record" or ln.prompt.split()[-1] in WORD_KEYS:
            continue
        value = parse_num(ln.answer)
        assert value is not None and value >= 0, ln.text
        digits = "".join(w for w in ln.answer.split(" e ")[0].split() if w.isdigit())
        assert len(digits.strip("0")) <= 3, ln.text
        assert len(ln.answer.split()) <= 8, ln.text


def test_check_matches_close(sim, rollouts):
    r = rollouts[7]
    for t in (5, 14, 29):
        for key in ("largest_mass", "outermost_au", "debris"):
            truth = r.steps[t][key]
            answer = r.value(truth)
            assert close(parse_num(answer), truth, rel=0.05)
            assert sim.check(f"planets seed 7 step {' '.join(str(t))} {key}", answer).ok


def test_a_run_is_fast(sim):
    worst = 0.0
    for seed in range(100, 120):
        start = time.perf_counter()
        sim.run(seed, **random_params(random.Random(seed)))
        worst = max(worst, time.perf_counter() - start)
    assert worst < 0.5
