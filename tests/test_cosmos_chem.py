import copy
import hashlib
import json
import random
import time

import numpy as np
import pytest

from haishool.cosmos import Simulation
from haishool.cosmos import chem
from haishool.cosmos.chem import (BONDS, COUNT_KEYS, DEFAULT_K, ELEMENTS, KEYS, MIXES, MOLECULES, QUERY_KEYS,
                                  STAGES, STATE_KEYS, VALENCE, Pool, bad_params, budget, cool, element_counts,
                                  molecule_counts, random_params, simulate, simulation, stability, temperature_at)
from haishool.truth import is_dense, num, parse_num, split_line
from haishool.truth.formula import dense, parse, parse_dense, same

SEEDS = range(1, 9)
FORTY = range(1, 41)
FIXED = {"t_start": 5000, "t_end": 300, "steps": 30, "atoms": 20000}


def n_tokens(text: str) -> int:
    return len(text.replace(".", " . ").split())  # haishool.student.tokens


@pytest.fixture(scope="module")
def sim():
    return simulation()


@pytest.fixture(scope="module")
def generated():
    return chem.generate(random.Random(5), 5)


@pytest.fixture(scope="module")
def forty():
    """Seeds 1-40, each with its own parameters (the runs check() replays)."""
    return {seed: chem._rollout(seed) for seed in FORTY}


def final_pool(seed: int, params: dict) -> Pool:
    pool = None
    for _, _, pool in cool(seed, params):
        pass
    return pool


def count_of(counts: dict[str, int], formula: str) -> int:
    return next((c for f, c in counts.items() if same(parse_dense(f), parse(formula))), 0)


def wrong(answer: str) -> str:
    """An answer no tolerance forgives: another word, or the leading digit moved by five."""
    words = answer.split()
    for pool in (STAGES, tuple(MIXES)):
        if answer in pool:
            return pool[(pool.index(answer) + 1) % len(pool)]
    if answer == "never":
        return "3"
    if not any(w.isdigit() for w in words):  # an element or a list of elements
        return " ".join(words[1:]) or ("o" if answer != "o" else "h")
    i = next(i for i, w in enumerate(words) if w.isdigit())
    words[i] = str((int(words[i]) + 5) % 10)
    return " ".join(words)


def test_is_a_simulation(sim):
    assert isinstance(sim, Simulation) and sim.sim == "chem"
    r = sim.run(1)
    assert set(r.steps[0]) | set(r.summary) <= set(KEYS)
    assert set(STATE_KEYS) <= set(r.steps[0]) and set(QUERY_KEYS) <= set(r.steps[0])
    assert r.params == random_params(random.Random(1))
    assert len(r.steps) == r.params["steps"] + 1
    with pytest.raises(ValueError):
        sim.run(1, mix="lava")


def test_same_seed_same_rollout(sim):
    for seed in (1, 2, 3):
        a, b = sim.run(seed), sim.run(seed)
        assert a.params == b.params and a.steps == b.steps and a.summary == b.summary
        assert [ln.text for ln in chem.lines(a)] == [ln.text for ln in chem.lines(b)]
    assert sim.run(1, **FIXED, mix="ocean").steps != sim.run(2, **FIXED, mix="ocean").steps
    one = [ln.text for ln in chem.generate(random.Random(9), 2, every=5)]
    assert one == [ln.text for ln in chem.generate(random.Random(9), 2, every=5)]
    assert one != [ln.text for ln in chem.generate(random.Random(10), 2, every=5)]


def test_random_params_stay_in_range():
    mixes = set()
    for seed in range(300):
        p = random_params(random.Random(seed))
        assert p == random_params(random.Random(seed))
        assert 3000 <= p["t_start"] <= 6000 and 100 <= p["t_end"] <= 1500
        assert 25 <= p["steps"] <= 35 and 18000 <= p["atoms"] <= 22000
        mixes.add(p["mix"])
    assert mixes == set(MIXES)


def test_every_line_is_dense_and_short(sim, generated):
    every = generated + sim.records() + sim.table_lines()
    for mix in MIXES:  # the widest numbers each mix can give
        every += chem.lines(sim.run(9998, mix=mix, t_start=6000, t_end=100, steps=35, atoms=22000))
    assert len(every) > 5000
    for ln in every:
        assert is_dense(ln.text), ln.text
        assert n_tokens(ln.text) <= 80, ln.text
        if ln.kind != "record":
            assert "." not in ln.prompt and "." not in ln.answer and ln.answer
            assert split_line(ln.text) == (ln.prompt, ln.answer)


def test_line_kinds(sim, generated):
    texts = [ln.text for ln in generated]
    assert len(texts) == len(set(texts))
    r = sim.run(1)
    own = chem.lines(r)
    state = [ln for ln in own if ln.kind == "record"]
    assert len(state) == len(r.steps) and all(ln.text.startswith("chem seed 1 step ") for ln in state)
    assert state[0].text.startswith(f"chem seed 1 step 0. mix {r.params['mix']}. temperature ")
    now = [ln for ln in own if " step " in ln.prompt and " next " not in ln.prompt and ln.kind != "record"]
    nxt = [ln for ln in own if " next " in ln.prompt]
    final = [ln for ln in own if " final " in ln.prompt]
    assert len(now) == len(r.steps) * len(QUERY_KEYS) and len(nxt) == (len(r.steps) - 1) * len(QUERY_KEYS)
    assert len(final) == len(r.summary)
    assert len(chem.lines(r, every=5)) < len(own)
    assert "q chem valence c. a 4." in texts and "q chem bond h o stable_below_k. a 3 0 0 0." in texts
    assert "q chem bond h he stable_below_k. a never." in texts
    assert "q chem molecule h2o formula. a h 2 o 1." in texts and "q chem mix cosmic main. a h." in texts
    records = [ln.text for ln in sim.records()]
    assert "chem bond h o. stable_below_k 3 0 0 0." in records
    assert "chem molecule sio2. formula si 1 o 2. atoms 3." in records
    assert records[0].startswith("chem valence. h 1. he 0. o 2. c 4. n 3. ne 0.")


def test_gate_agrees_with_its_own_lines(sim, generated):
    questions = [ln for ln in generated if ln.kind != "record"]
    assert len(questions) > 4000
    for ln in questions:
        assert sim.owns(ln.prompt), ln.prompt
        v = sim.check(ln.prompt, ln.answer)
        assert v.ok and v.expected == ln.answer, (ln.text, v)


def test_gate_rejects_wrong_answers(sim, generated):
    questions = [ln for ln in generated if ln.kind != "record"]
    assert len(questions) > 5000
    for ln in questions:
        v = sim.check(ln.prompt, wrong(ln.answer))
        assert not v.ok and v.expected == ln.answer, (ln.text, wrong(ln.answer), v)
        v = sim.check(ln.prompt, "many")  # a word where a number or another word belongs
        assert not v.ok and v.expected == ln.answer, (ln.text, v)
    for prompt, good, bad in [("chem valence c", "4", "3"), ("chem bond h o stable_below_k", "3 0 0 0", "2 0 0 0"),
                              ("chem bond o h stable_below_k", "3 0 0 0", "never"),
                              ("chem bond ne o stable_below_k", "never", "2 0 0 0"),
                              ("chem bond fe mg stable_below_k", "1 0 0 0", "2 0 0 0"),
                              ("chem bond c n stable_below_k", "2 0 0 0", "2 5 0 0"),
                              ("chem bond default stable_below_k", "2 0 0 0", "1 0 0 0"),
                              ("chem molecule nh3 formula", "n 1 h 3", "n 1 h 4"),
                              ("chem molecule co2 atoms", "3", "2"), ("chem mix ocean main", "h", "o"),
                              ("chem mix rocky elements", " ".join(MIXES["rocky"]), "o si mg")]:
        assert sim.check(prompt, good) == chem.Verdict(True, good)
        assert sim.check(prompt, bad) == chem.Verdict(False, good)


def test_gate_tolerance(sim):
    r = sim.run(2)  # check() replays the seed's own run
    t = next(t for t, s in enumerate(r.steps) if s["molecules"] > 1000)
    s, base = r.steps[t], f"chem seed 2 step {num(t)}"
    assert sim.check(f"{base} molecules", num(s["molecules"] + 1)).ok  # a count may be off by one
    assert sim.check(f"{base} molecules", num(int(s["molecules"] * 1.04))).ok  # or by under 5 %
    assert not sim.check(f"{base} molecules", num(int(s["molecules"] * 1.2))).ok
    assert sim.check(f"{base} temperature", num(s["temperature"] + 20)).ok
    assert not sim.check(f"{base} temperature", num(s["temperature"] * 2)).ok
    assert not sim.check(f"{base} stage", "plasma" if s["stage"] != "plasma" else "frozen").ok
    assert sim.check(f"chem seed 2 step {num(t - 1)} next molecules", num(s["molecules"])).ok
    assert sim.check(f"{base} molecules", "many") == chem.Verdict(False, num(s["molecules"]), "not a number")
    last = len(r.steps) - 1
    assert sim.check(f"chem seed 2 step {num(last)} next h2o", "1").expected is None
    assert sim.check(f"chem seed 2 step {num(last + 5)} h2o", "1").expected is None
    assert sim.check("chem seed 2 step 3 clumps", "1").expected is None
    assert sim.check("chem seed 2 final clumps", "1").expected is None
    assert sim.check("chem valence xx", "1").expected is None
    assert sim.check("turkey capital", "ankara") == chem.Verdict(False, None, "not my question")


def test_owns_only_its_own_prompts(sim, generated):
    for prompt in ["chem seed 7 step 2 0 h2o", "chem seed 1 2 3 4 step 0 next stage", "chem seed 7 final water_fraction",
                   "chem valence si", "chem bond na cl stable_below_k", "chem bond default stable_below_k",
                   "chem molecule h2o formula", "chem mix rocky elements"]:
        assert sim.owns(prompt), prompt
    for prompt in ["turkey capital", "q spoon color", "spoon color", "gravity seed 7 step 2 0 clumps",
                   "nucleo seed 3 final helium_fraction", "world seed 3 era planets", "carbon protons",
                   "water molar_mass", "calc 1 2 plus 7", "check 1 2 plus 7 equals 2 0", "chem", "chem seed",
                   "chem seed x step 1 h2o", "chem seed 7 step h2o", "q chem seed 7 final h2o", "",
                   "nucleo seed 2 2 0 2 step 2 8 next lithium", "planets seed 2 2 0 2 step 0 planets",
                   "life seed 2 2 0 2 final replicators", "life seed 3 param mutation", "gravity seed 7 final clumps",
                   "meitnerium econf", "check arsenic electrons 3 3", "electrostatic unit",
                   "phosphorus_combustion equation", "hydrogen_sulfide elements", "water formula",
                   "chem seed 0 7 step 1 h2o", "chem seed 7 step 0 1 h2o", "chem seed 7 step 1 h2o ",
                   "chem seed 7 step 1 next h2o extra", "chem seed 7 final", "chem seed minus 7 final h2o",
                   "chem valence carbon", "chem bond h o", "chem mix cosmic"]:
        assert not sim.owns(prompt), prompt
        assert not sim.check(prompt, "1").ok


def test_tables():
    given = {"h h": 2000, "o h": 3000, "c h": 2500, "c o": 3500, "n n": 4000, "si o": 3800, "na cl": 1500,
             "mg o": 3000, "fe o": 2500, "c c": 2800, "n h": 2200, "s h": 1800, "s o": 2600, "o o": 1200}
    for pair, kelvin in given.items():
        a, b = pair.split()
        assert stability(a, b) == stability(b, a) == kelvin
    assert len(chem.GIVEN_BONDS) == len(given) and not set(chem.GIVEN_BONDS) & set(chem.ADDED_BONDS)
    assert set(chem.NOTES) == {"valence", "given_bonds", "added_bonds", "metal_metal", "mixes"}
    assert max(chem.ADDED_BONDS.values()) < chem.PLASMA_K == 4000 and chem.FROZEN_K == 1000
    assert VALENCE == {"h": 1, "he": 0, "o": 2, "c": 4, "n": 3, "ne": 0, "mg": 2, "si": 4, "s": 2, "fe": 2,
                       "al": 3, "ca": 2, "na": 1, "k": 1, "cl": 1}
    assert stability("c", "n") == stability("si", "si") == DEFAULT_K == 2000
    for a in ELEMENTS:
        assert stability("he", a) == stability(a, "ne") == 0
        for b in ELEMENTS:
            assert stability(a, b) == stability(b, a)
    assert all(a <= b for a, b in BONDS)
    for name, formula in MOLECULES.items():
        assert name == formula.lower() and dense(formula) in KEYS[name]
    for mix, fractions in MIXES.items():
        assert abs(sum(fractions.values()) - 1) < 0.011 and set(fractions) <= set(ELEMENTS)
        assert list(fractions.values()) == sorted(fractions.values(), reverse=True)
        for atoms in (18000, 20000, 21000, 22000):
            counts = element_counts(mix, atoms)
            assert sum(counts.values()) == atoms and min(counts.values()) > 0


@pytest.mark.parametrize("mix", list(MIXES))
def test_atoms_valence_and_nobles_hold_at_every_step(mix):
    for seed in (1, 2):
        params = {**random_params(random.Random(seed)), "mix": mix}
        counts = element_counts(mix, params["atoms"])
        totals = np.array([counts.get(e, 0) for e in ELEMENTS])
        noble = np.array([VALENCE[e] == 0 for e in ELEMENTS])
        for _, _, pool in cool(seed, params):
            assert pool.audit(totals) == []
            n = len(pool.el)
            held = np.bincount(pool.bond_a, pool.bond_m, minlength=n) + np.bincount(pool.bond_b, pool.bond_m, minlength=n)
            assert (held <= pool.valence).all() and (pool.free >= 0).all()
            assert not held[noble[pool.el]].any()
            comp, size = pool.census()
            assert (comp.sum(0) == totals).all() and (comp >= 0).all()
            assert not comp[size >= 2][:, noble].any()  # no molecule holds a noble atom


def test_conserved_is_a_gate(sim):
    for seed in SEEDS:
        r = sim.run(seed)
        assert sim.conserved(r) == chem.Verdict(True, None, "")
    r = sim.run(3, mix="ocean", **FIXED)
    assert sim.conserved(r).ok
    forged = copy.deepcopy(r)
    forged.steps[12]["h2o"] += 40
    assert not sim.conserved(forged).ok
    forged = copy.deepcopy(r)
    forged.steps[5]["o2"] = -1
    assert "negative" in sim.conserved(forged).reason
    forged = copy.deepcopy(r)
    forged.steps[20]["molecules"] = 10
    assert "more named molecules" in sim.conserved(forged).reason
    forged = copy.deepcopy(r)
    forged.summary["water_fraction"] = 0.5
    assert not sim.conserved(forged).ok
    forged = copy.deepcopy(r)
    forged.sim = "gravity"
    assert not sim.conserved(forged).ok


def test_audit_sees_a_broken_law():
    counts = {"h": 40, "he": 10, "o": 20}
    totals = np.array([counts.get(e, 0) for e in ELEMENTS])
    pool = Pool(counts, np.random.default_rng(1))
    pool.step(1500)
    assert pool.audit(totals) == [] and len(pool.bond_a) > 0
    he = int(np.flatnonzero(pool.el == ELEMENTS.index("he"))[0])
    pool.bond_a, pool.bond_b = np.append(pool.bond_a, he), np.append(pool.bond_b, 0)
    pool.bond_m = np.append(pool.bond_m, 1)
    problems = pool.audit(totals)
    assert "a noble gas is bonded" in problems and "an atom holds more bonds than its valence" in problems
    assert pool.audit(totals + 1)[-1] == "an element's atom count changed"


def test_bonds_form_when_cool_and_break_when_hot():
    pool = Pool({"h": 600, "o": 100}, np.random.default_rng(3))
    pool.step(3500)  # hotter than O-H, H-H and O-O
    assert len(pool.bond_a) == 0 and pool.formed == 0
    pool.step(2500)  # only O-H holds
    kinds = {tuple(sorted((ELEMENTS[a], ELEMENTS[b]))) for a, b in zip(pool.el[pool.bond_a], pool.el[pool.bond_b])}
    assert kinds == {("h", "o")}
    for _ in range(4):
        pool.step(1500)  # H-H holds too, O-O not yet
    counts = molecule_counts(pool)
    assert counts["h 2 o 1"] > 90 and counts["h 2"] > 90 and "o 2" not in counts
    before = len(pool.bond_a)
    pool.step(2500)  # H-H breaks, O-H stays
    assert 0 < pool.broken < before
    counts = molecule_counts(pool)
    assert "h 2" not in counts and counts["h 2 o 1"] > 90
    pool.step(5000)
    assert len(pool.bond_a) == 0 and (pool.free == pool.valence).all() and molecule_counts(pool) == {}


def test_multiple_bonds_fill_the_valence():
    for counts, formula in [({"n": 200}, "N2"), ({"o": 200}, "O2"), ({"h": 200}, "H2")]:
        pool = Pool(counts, np.random.default_rng(1))
        for _ in range(6):
            pool.step(500)
        found = molecule_counts(pool)
        assert count_of(found, formula) >= 98 and len(found) == 1
        assert set(pool.bond_m) == {VALENCE[next(iter(counts))]}
    pool = Pool({"si": 100, "o": 200}, np.random.default_rng(1))
    for _ in range(12):
        pool.step(3700)  # only Si-O holds
    assert count_of(molecule_counts(pool), "SiO2") > 60


def test_metrics_are_the_census(sim):
    for mix in MIXES:
        params = {**FIXED, "mix": mix}
        r = sim.run(4, **params)
        counts = molecule_counts(final_pool(4, params))
        last = r.steps[-1]
        assert list(counts.values()) == sorted(counts.values(), reverse=True)
        assert last["molecules"] == sum(counts.values())
        for name, formula in MOLECULES.items():
            assert last[name] == count_of(counts, formula), (mix, name)
        assert last["organic"] == sum(c for f, c in counts.items() if parse_dense(f).get("C", 0) > 2)
        assert last["biggest"] == max(sum(parse_dense(f).values()) for f in counts)
        for s in r.steps:
            assert 0 <= s["free_atoms"] <= 1 and s["molecules"] >= sum(s[k] for k in (*MOLECULES, "organic"))
            assert all(s[k] >= 0 for k in COUNT_KEYS if k in s)


def test_summary_follows_from_the_steps(sim):
    for seed in SEEDS:
        r = sim.run(seed)
        last, summary = r.steps[-1], r.summary
        for k in ("h2o", "ch4", "nh3", "organic", "molecules"):
            assert summary[k] == last[k]
        assert summary["mix"] == r.params["mix"]
        assert summary["water_fraction"] == pytest.approx(last["h2o"] / last["molecules"], rel=0.005)
        first = summary["first_water_step"]
        assert r.steps[first]["h2o"] > 0 and all(s["h2o"] == 0 for s in r.steps[:first])
        assert parse_num(num(summary["water_fraction"])) == summary["water_fraction"]


def test_temperature_falls_and_stages_follow(forty):
    for r in forty.values():
        temps = [s["temperature"] for s in r.steps]
        assert temps[0] == r.params["t_start"] and temps[-1] == r.params["t_end"]
        assert temps == sorted(temps, reverse=True)
        order = [STAGES.index(s["stage"]) for s in r.steps]
        assert order == sorted(order)
        for s in r.steps:
            assert (s["stage"] == "plasma") == (s["temperature"] >= 4000)
            assert (s["stage"] == "frozen") == (s["temperature"] < 1000)
        assert r.steps[0]["free_atoms"] == 1 and r.steps[0]["molecules"] == 0 and r.steps[0]["biggest"] == 1


def test_hot_gas_is_atomic(forty):
    hot = 0
    for r in forty.values():
        for s in r.steps:
            if s["temperature"] >= 4000:
                hot += 1
                assert s["free_atoms"] == 1 and s["molecules"] == 0
        if r.params["t_start"] > 4000:
            assert r.steps[0]["stage"] == "plasma"
    assert hot > 20


@pytest.mark.parametrize("seed", SEEDS)
def test_cosmic_mix_ends_as_hydrogen_gas_with_water(sim, seed):
    params = {**random_params(random.Random(seed)), "mix": "cosmic"}
    last = sim.run(seed, **params).steps[-1]
    counts = molecule_counts(final_pool(seed, params))
    assert next(iter(counts)) == "h 2" and last["h2"] > 0.9 * last["molecules"]
    with_oxygen = [f for f in counts if "O" in parse_dense(f)]
    assert with_oxygen[0] == "h 2 o 1"
    assert last["h2o"] > 0.8 * element_counts("cosmic", params["atoms"])["o"]
    assert 0.09 < last["free_atoms"] < 0.1  # helium and neon stay free


@pytest.mark.parametrize("seed", SEEDS)
def test_rocky_mix_ends_as_oxides(sim, seed):
    params = {**random_params(random.Random(seed)), "mix": "rocky"}
    last = sim.run(seed, **params).steps[-1]
    counts = molecule_counts(final_pool(seed, params))
    assert same(parse_dense(next(iter(counts))), parse("SiO2"))
    assert same(parse_dense(list(counts)[1]), parse("MgO"))
    assert last["sio2"] > 1500 and last["mgo"] > 1000 and last["feo"] > 300
    assert last["h2"] < 20 and last["stage"] in ("molecular", "frozen")


@pytest.mark.parametrize("seed", SEEDS)
def test_ocean_mix_ends_as_water_and_salt(sim, seed):
    params = {**random_params(random.Random(seed)), "mix": "ocean"}
    r = sim.run(seed, **params)
    assert r.summary["water_fraction"] > 0.75
    cold = {**params, "t_end": min(params["t_end"], 1200)}  # salt needs the gas to pass 1500 K
    last = sim.run(seed, **cold).steps[-1]
    counts = molecule_counts(final_pool(seed, cold))
    assert next(iter(counts)) == "h 2 o 1" and list(counts)[1] == "na 1 cl 1"
    assert last["nacl"] > 100
    assert sim.run(seed, **{**params, "t_end": 1500}).steps[-1]["nacl"] == 0


@pytest.mark.parametrize("seed", SEEDS)
def test_carbon_mix_makes_chains(sim, seed):
    params = {**random_params(random.Random(seed)), "mix": "carbon"}
    last = sim.run(seed, **params).steps[-1]
    assert next(iter(molecule_counts(final_pool(seed, params)))) == "h 2"
    assert last["organic"] > 100 and last["ch4"] > 100 and last["biggest"] > 8
    for mix in ("cosmic", "rocky", "ocean"):
        assert sim.run(seed, **{**params, "mix": mix}).steps[-1]["organic"] < 10


def test_heating_breaks_the_molecules_again(sim):
    r = sim.run(3, mix="ocean", t_start=500, t_end=5000, steps=30, atoms=20000)
    assert sim.conserved(r).ok
    assert max(s["h2o"] for s in r.steps) > 1000
    assert r.steps[-1]["molecules"] == 0 and r.steps[-1]["free_atoms"] == 1 and r.steps[-1]["stage"] == "plasma"


def test_a_run_takes_under_a_second(sim):
    best = 9.0
    for seed in (1, 2, 3):
        t0 = time.perf_counter()
        sim.run(seed, mix="rocky", t_start=6000, t_end=100, steps=35, atoms=22000)
        best = min(best, time.perf_counter() - t0)
    assert best < 1.0


def test_forty_seeds_conserve_and_repeat(sim, forty):
    mixes = set()
    for seed, r in forty.items():
        again = sim.run(seed)  # a second, uncached run
        assert again.params == r.params == random_params(random.Random(seed))
        assert again.summary == r.summary and len(again.steps) == len(r.steps)
        for a, b in zip(r.steps, again.steps):  # exactly: same keys in the same order, same types, same values
            assert [(k, type(v), v) for k, v in a.items()] == [(k, type(v), v) for k, v in b.items()]
        assert sim.conserved(r) == chem.Verdict(True, None, ""), seed
        mixes.add(r.params["mix"])
    assert mixes == set(MIXES)


def test_plausibility_targets_on_forty_seeds(forty):
    seen = {mix: 0 for mix in MIXES}
    for seed, r in forty.items():
        mix, last = r.params["mix"], r.steps[-1]
        have = element_counts(mix, r.params["atoms"])
        ranked = list(molecule_counts(final_pool(seed, r.params)))
        seen[mix] += 1
        for s in r.steps:
            if s["temperature"] >= 4000:
                assert s["free_atoms"] == 1 and s["molecules"] == 0 and s["stage"] == "plasma"
            assert s["o2"] <= 5  # free oxygen hardly ever forms
        assert last["co"] <= 5 and last["molecules"] > 5000
        if mix == "cosmic":
            assert ranked[0] == "h 2" and last["h2"] > 0.97 * last["molecules"]
            assert next(f for f in ranked if "O" in parse_dense(f)) == "h 2 o 1"
            assert last["h2o"] > 0.9 * have["o"] and last["ch4"] > 0.8 * have["c"] and last["nh3"] > 0.8 * have["n"]
            assert 0.09 < last["free_atoms"] < 0.093 and last["organic"] == 0
        elif mix == "rocky":
            assert ranked[:3] == ["o 2 si 1", "o 1 mg 1", "o 1 fe 1"]
            assert last["sio2"] > 0.8 * have["si"] and last["mgo"] > 0.8 * have["mg"] and last["feo"] > 0.5 * have["fe"]
            assert last["h2"] == 0 and last["organic"] <= 2 and last["free_atoms"] < 0.02
        elif mix == "ocean":
            assert ranked[0] == "h 2 o 1" and 0.8 < r.summary["water_fraction"] < 0.9
            assert last["h2o"] > 0.8 * have["o"] and last["organic"] <= 2
            if r.params["t_end"] < 1500:
                assert ranked[1] == "na 1 cl 1" and last["nacl"] > 0.5 * have["na"]
            else:
                assert last["nacl"] == 0
        else:
            assert ranked[0] == "h 2" and ranked[1] in ("h 4 c 1", "h 3 n 1", "n 2")
            assert last["ch4"] > 250 and last["organic"] > 200 and last["biggest"] > 15
    assert min(seen.values()) >= 8


def test_stage_follows_the_numbers_in_the_line(forty):
    for r in forty.values():
        for s in r.steps:
            assert s["stage"] == chem.stage(s["temperature"], s["free_atoms"])
            assert parse_num(num(s["free_atoms"])) == s["free_atoms"]  # what the line shows is the value
    assert chem.stage(2000, 0.9) == "atomic" and chem.stage(2000, 0.899) == "forming"
    assert chem.stage(2000, 0.5) == "forming" and chem.stage(2000, 0.499) == "molecular"
    assert chem.stage(4000, 1) == "plasma" and chem.stage(3999, 1) == "atomic"
    assert chem.stage(1000, 0) == "molecular" and chem.stage(999, 0) == "frozen"


def test_no_two_atoms_are_bonded_twice():
    pool = Pool({"c": 2}, np.random.default_rng(1))
    for _ in range(4):  # the two carbons meet in every round; they bond once, with three valences
        pool.step(500)
    assert pool.bond_m.tolist() == [3] and pool.free.tolist() == [1, 1] and molecule_counts(pool) == {"c 2": 1}
    assert pool.audit(np.array([2 if e == "c" else 0 for e in ELEMENTS])) == []
    for seed in (4, 14, 45):  # carbon runs in which a triple-bonded pair meets again
        params = {**random_params(random.Random(seed)), "mix": "carbon"}
        counts = element_counts("carbon", params["atoms"])
        totals = np.array([counts.get(e, 0) for e in ELEMENTS])
        for _, _, pool in cool(seed, params):
            pairs = pool.pair(pool.bond_a, pool.bond_b)
            assert len(set(pairs.tolist())) == len(pairs) and (pool.bond_m <= 3).all() and (pool.bond_m >= 1).all()
            assert pool.audit(totals) == []
    pool.bond_a, pool.bond_b = np.append(pool.bond_a, pool.bond_a[0]), np.append(pool.bond_b, pool.bond_b[0])
    pool.bond_m = np.append(pool.bond_m, 1)
    assert "two atoms are bonded twice" in pool.audit(totals)


def test_temperature_is_rounded_with_whole_numbers():
    for t_start in range(3000, 6001, 100):
        for t_end in range(100, 1501, 50):
            for n in (25, 28, 35):
                assert temperature_at(t_start, t_end, 0, n) == t_start and temperature_at(t_start, t_end, n, n) == t_end
                for t in range(1, n):
                    k = temperature_at(t_start, t_end, t, n)
                    twice_x_to_n = 2 ** n * t_start ** (n - t) * t_end ** t
                    assert (2 * k - 1) ** n < twice_x_to_n < (2 * k + 1) ** n  # nearest, and never an exact half
                    assert k == int(round(t_start * (t_end / t_start) ** (t / n)))
    assert temperature_at(4000, 1000, 15, 30) == 2000 and temperature_at(3800, 1400, 26, 28) == 1504
    assert temperature_at(5000, 300, 500, 1000) == 1225 and temperature_at(500, 5000, 1, 4) == 889
    assert temperature_at(3, 1, 1, 2) == 2 and temperature_at(1, 1, 3, 7) == 1  # sqrt(3) = 1.73


def test_gate_takes_only_finite_numbers(sim):
    r = sim.run(2)
    for prompt in ["chem seed 2 step 3 temperature", "chem seed 2 step 3 h2o", "chem seed 2 step 3 free_atoms",
                   "chem seed 2 step 3 next molecules", "chem seed 2 final water_fraction", "chem seed 2 final h2"]:
        expected = sim.check(prompt, "0").expected
        assert expected is not None
        for answer in ["1 e 9 9 9", "minus 1 e 9 9 9", "9 e 4 0 0", "", "point", "minus", "e 5", "1 point 2 point 3"]:
            assert sim.check(prompt, answer) == chem.Verdict(False, expected, "not a number"), (prompt, answer)
        assert sim.check(prompt, expected).ok
    assert sim.check("chem seed 2 final first_water_step", num(r.summary["first_water_step"])).ok
    assert not sim.check("chem seed 2 final first_water_step", "never").ok
    assert not sim.check("chem seed 2 final mix", "1 e 9 9 9").ok


def test_run_refuses_parameters_it_cannot_simulate(sim):
    for bad in [{"steps": 0}, {"steps": 2.5}, {"t_start": 0}, {"t_end": -5}, {"atoms": 0}, {"atoms": True}, {"mix": "lava"}]:
        with pytest.raises(ValueError):
            sim.run(1, **bad)
        r = sim.run(1)
        r.params.update(bad)
        assert sim.conserved(r) == chem.Verdict(False, None, "not a chem rollout")
    good = random_params(random.Random(1))
    assert bad_params(good) == "" and "t_end" in bad_params({k: v for k, v in good.items() if k != "t_end"})
    r = sim.run(3, mix="ocean", t_start=500, t_end=5000, steps=4, atoms=300)  # small, short and heating up
    assert [s["temperature"] for s in r.steps] == [500, 889, 1581, 2812, 5000] and sim.conserved(r).ok


def test_counted_molecules_fit_into_the_atoms(sim, forty):
    for r in forty.values():
        counts = element_counts(r.params["mix"], r.params["atoms"])
        for s in r.steps:
            assert budget(s, counts) == []
    r = sim.run(3, mix="ocean", **FIXED)
    counts = element_counts("ocean", FIXED["atoms"])
    last = dict(r.steps[-1])
    assert budget({**last, "h2o": counts["o"] + 1, "molecules": counts["o"] + 500}, counts) \
        == ["more h in the counted molecules than the mix holds", "more o in the counted molecules than the mix holds",
            "more atoms in molecules than are bonded"]
    assert budget({**last, "organic": counts["c"] // 3 + 1}, counts)[0] == "more c in the counted molecules than the mix holds"
    assert budget({**last, "molecules": FIXED["atoms"]}, counts) == ["more atoms in molecules than are bonded"]
    assert budget({**last, "free_atoms": 0.5}, counts) == ["more atoms in molecules than are bonded"]
    forged = copy.deepcopy(r)
    forged.steps[-1]["nacl"] = counts["na"] + 1
    reason = sim.conserved(forged).reason
    assert "more na in the counted molecules" in reason and "more cl in the counted molecules" in reason


def test_golden_rollouts():
    """Pinned on numpy 2.4.6, scipy 1.17.1, python 3.11: a seed must stay its rollout on any machine."""
    assert [int(x) for x in np.random.PCG64(1).random_raw(3)] \
        == [9441442522235856127, 17532960557476522086, 2659275481604167885]
    assert random_params(random.Random(1)) == {"mix": "rocky", "t_start": 4800, "t_end": 1450, "steps": 26, "atoms": 20000}
    r = chem._rollout(1)
    assert r.steps[10] == {"mix": "rocky", "temperature": 3029, "stage": "molecular", "free_atoms": 0.463,
                           "molecules": 3593, "h2": 0, "h2o": 0, "ch4": 0, "nh3": 0, "co": 3, "co2": 397, "o2": 0,
                           "n2": 30, "nacl": 0, "sio2": 2797, "mgo": 0, "feo": 0, "organic": 0, "biggest": 5}
    assert r.summary == {"mix": "rocky", "h2": 0, "h2o": 22, "ch4": 0, "nh3": 0, "nacl": 17, "sio2": 2798, "organic": 0,
                         "molecules": 6828, "biggest": 17, "water_fraction": 0.00322, "first_water_step": 11}
    for seed, mix, digest in [(1, "rocky", "765bd11f89b7c7d8"), (2, "cosmic", "e8aea5e74b5c111e"),
                              (5, "ocean", "3cea803a59226a9d"), (9, "carbon", "3bde93222da89d53")]:
        r = chem._rollout(seed)
        assert r.params["mix"] == mix
        assert hashlib.sha256(json.dumps([r.steps, r.summary], sort_keys=True).encode()).hexdigest()[:16] == digest, seed
    assert chem.lines(chem._rollout(42))[12].text == (
        "chem seed 4 2 step 1 2. mix cosmic. temperature 2 0 8 8. stage atomic. free_atoms 0 point 9 7 5. "
        "molecules 1 2 6. h2 0. h2o 6 6. ch4 4 2. nh3 1 7. nacl 0. sio2 0. organic 0. biggest 5.")
