import collections
import copy
import hashlib
import importlib
import itertools
import json
import math
import random
import statistics
import time

import numpy as np
import pytest

from haishool.cosmos import Rollout, Simulation, close
from haishool.evo import LessonGate, sig
from haishool.evo import cells
from haishool.evo.cells import (CAP, CENSUS_KEYS, COUNT_KEYS, DEATH, DERIVED_KEYS, ENERGY, GENS_PER_STEP, GMAX,
                                HANDOFF_OUT, INFLOW_SCALE, KEYS, LED_KEYS, LEDGER_KEYS, LESSONS, OUTCOMES,
                                PARAM_KEYS, RULES, SIM, STAGES, STATE_KEYS, STEPS, SUMMARY_KEYS, SURVIVAL_ODDS,
                                WORD_KEYS, Cells, CellRollout, _Rng, _Vessel, balance, census, copy_success,
                                dense_value, derived, derived_exact, derived_line, doubling_time, efficiency,
                                energy_per_gene,
                                error_threshold, expected_waste, expression, full_set, generations_to_fix,
                                handoff_in, lesson_gate, lines, nutrient_share, params_line, partner_gain,
                                random_params, run, simulation, split, stage_of, start_rate, summarise,
                                superiority, tables, under_threshold)
from haishool.truth import Gate, is_dense, num, parse_num, split_line

SEEDS = list(range(1, 41))


def n_tokens(text: str) -> int:
    """As ``haishool.student.tokens`` counts them (that module needs torch, so not imported)."""
    return len(text.replace(".", " . ").split())


def digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def canonical(seed: int) -> CellRollout:
    return run(seed, **random_params(random.Random(seed)))


def over_threshold(r: CellRollout) -> bool:
    """Is the gene of this run longer than the error threshold of its protocells?"""
    return r.params["gene_length"] > r.derived["l_max"]


@pytest.fixture(scope="module")
def sim():
    return Cells()


@pytest.fixture(scope="module")
def rollouts():
    """The canonical runs of seeds 1 to 40, with their cells."""
    return {seed: canonical(seed) for seed in SEEDS}


@pytest.fixture(scope="module")
def all_lines(rollouts):
    return [ln for r in rollouts.values() for ln in lines(r)]


@pytest.fixture(scope="module")
def lessons():
    return LESSONS.generate(random.Random(1), 400)


# ---------------------------------------------------------------- the level as a whole

def test_protocols_and_module_interface():
    s = simulation()
    assert isinstance(s, Simulation) and isinstance(s, Gate)
    assert s.sim == s.topic == SIM == "cells" and s.records() == [] and simulation() is s
    assert cells.KEYS is KEYS and s.KEYS is KEYS
    for key in STATE_KEYS + LEDGER_KEYS:
        assert key in KEYS, key
    assert set(SUMMARY_KEYS) - {"first_chromosome", "outcome"} <= set(STATE_KEYS)
    assert set(HANDOFF_OUT) == set(SUMMARY_KEYS) and COUNT_KEYS | WORD_KEYS <= set(STATE_KEYS) | set(SUMMARY_KEYS)
    assert set(CENSUS_KEYS) | set(LED_KEYS) | {"bases"} == set(LEDGER_KEYS) | {"cells"}
    assert lesson_gate() is LESSONS and isinstance(LESSONS, LessonGate) and LESSONS.sim == "cells"
    assert LESSONS.topic == "predict_cells" and len(RULES) >= 8
    for name in ("conserved", "check", "owns", "lines", "run", "random_params", "handoff_in"):
        assert callable(getattr(cells, name)), name
    assert STAGES == ("soup", "protocells", "cells", "chromosomes", "complex_cells", "collapse")
    assert OUTCOMES == ("collapse", "protocells", "cells", "complex_cells")
    assert STEPS == 60 and GENS_PER_STEP == 5 and CAP == 400 and SURVIVAL_ODDS == pytest.approx((1 - DEATH) / DEATH)


def test_same_seed_same_output():
    for seed in (1, 7, 4321):
        a, b = canonical(seed), canonical(seed)
        assert a.steps == b.steps and a.summary == b.summary and a.params == b.params
        assert a.derived == b.derived and a.snaps == b.snaps
        assert [repr(v) for s in a.steps for v in s.values()] == [repr(v) for s in b.steps for v in s.values()]
        assert [ln.text for ln in lines(a)] == [ln.text for ln in lines(b)]
    assert [ln.text for ln in lines(canonical(2))] != [ln.text for ln in lines(canonical(3))]
    assert run(5, gene_types=3, threshold=24).steps == run(5, gene_types=3, threshold=24).steps
    assert run(5, gene_types=3, threshold=24).steps != run(6, gene_types=3, threshold=24).steps
    assert run(5).params == cells.DEFAULTS and run(5, rules=7).steps == run(5).steps


def test_forty_seeds_replay_exactly(rollouts, sim):
    """A second run gives the same steps, cells and lines; the cached rollout of the gate too."""
    for seed, r in rollouts.items():
        again = canonical(seed)
        assert again.steps == r.steps and again.snaps == r.snaps and again.summary == r.summary
        assert [repr(v) for s in again.steps for v in s.values()] == [repr(v) for s in r.steps for v in s.values()]
        assert again.params == r.params and again.derived == r.derived
        cached = sim.rollout(seed)
        assert cached.steps == r.steps and cached.summary == r.summary and cached.derived == r.derived
        assert cached.snaps == [] and sim.rollout(seed) is cached


def test_pinned_digests():
    """Pinned, so that another numpy or machine cannot change the rollouts unnoticed."""
    pins = {
        7: ("1ef0818b3e281f82b7b17d195b8f844254596fb0c34a16dc540a1fac06e635b6",
            "5407fb280ad18658d82af72d0ee8cedf100011e3f3719e3aa05191c4bd0804dc",
            "bf81ef16c522a4681fcd0970acea854ba0f2c79c7c886f8a37a190d8048bd300", 1658),
        16: ("092c49b725310fc5d6de78f34ec05ddbde38a703ba5789aa610ffdf6e159bacf",
             "92bc30aa23786d65802bedd37858b71775e56cea7affd0935848f9dbb11db20c",
             "0b0469f4e0f1281700a2b352eac113bda0797e451b963dd16c14e70cb790e1d3", 726),
        33: ("4ceb6ed8781ebbbb8d21babbc81a56d2c8dc656b037e4abbd6e7f41a00743bf3",
             "2db92361dac07c0f3aebb8797df76bf1a41f24262fb79758482afa81992feb12",
             "fb6040e5f83c616ba0b39644bcee13fbec0480627bb2b01e0a91c6d0e856abaa", 1658),
    }
    for seed, (rollout, text, snaps, n) in pins.items():
        r = canonical(seed)
        assert digest([r.params, r.steps, r.summary, r.derived]) == rollout, seed
        assert digest([ln.text for ln in lines(r)]) == text and len(lines(r)) == n, seed
        assert digest(r.snaps) == snaps, seed


def test_random_numbers_are_the_raw_pcg64_stream():
    rng = _Rng(3)
    assert rng.uniform(2).tolist() == [0.08564916714362436, 0.2368105065960997]
    assert rng.uniform(3).tolist() == [0.8012744652063969, 0.5821620360643678, 0.09412864224039919]
    assert rng.uniform(0).tolist() == [] and rng.uniform(0).dtype == np.float64
    for seed in (0, 3, 123456789):
        ours = _Rng(seed).uniform(50)
        assert ours.tolist() == np.random.default_rng(seed).random(50).tolist()
        assert ours.dtype == np.float64 and 0.0 <= ours.min() and ours.max() < 1.0


def test_values_are_plain_python_numbers_and_words(rollouts):
    for r in rollouts.values():
        assert type(r) is CellRollout and isinstance(r, Rollout) and r.sim == "cells"
        for row in r.steps:
            assert list(row) == list(STATE_KEYS) + list(LEDGER_KEYS)
            for key, value in row.items():
                want = str if key == "stage" else float if key == "best_growth" else \
                    int if key in LEDGER_KEYS or key == "cells" else float
                assert type(value) is want, (key, value)
                assert value in STAGES if key == "stage" else math.isfinite(value) and value >= 0
        assert list(r.params) == list(PARAM_KEYS) and list(r.derived) == list(DERIVED_KEYS)
        assert list(r.summary) == list(SUMMARY_KEYS) and r.summary["outcome"] in OUTCOMES
        assert type(r.summary["cells"]) is int and type(r.summary["first_chromosome"]) in (int, str)
        assert all(type(x) is int for snap in r.snaps for cell in snap for x in cell)


def test_random_params_stay_in_their_lists():
    for seed in range(300):
        p = random_params(random.Random(seed))
        assert p == random_params(random.Random(seed)) == random_params(random.Random(seed), rules=7)
        assert list(p) == list(PARAM_KEYS)
        assert 2 <= p["gene_types"] <= 6 and p["gene_length"] in (20, 40, 80, 160, 320, 640)
        assert p["fidelity"] in (0.95, 0.98, 0.99, 0.995, 0.998, 0.999) and p["selfish"] in (0.02, 0.05, 0.1, 0.2)
        assert p["inflow"] in (200000, 500000, 1000000, 2000000, 5000000)
        assert p["threshold"] // p["gene_types"] in (6, 8, 10, 12, 16) and p["threshold"] % p["gene_types"] == 0
        assert p["innovation"] in (0.0, 0.002, 0.005, 0.01, 0.02) and p["replicators"] in (100, 300, 1000, 3000)
    assert random_params(random.Random(33)) == {"gene_types": 6, "gene_length": 40, "fidelity": 0.999,
                                                "selfish": 0.05, "inflow": 1000000, "threshold": 72,
                                                "innovation": 0.02, "replicators": 300}
    with pytest.raises(ValueError):
        random_params(random.Random(1), rules=6)


def test_run_rejects_impossible_parameters(sim):
    for bad in ({"gene_types": 0}, {"gene_types": 17}, {"gene_types": 2.5}, {"gene_length": 0}, {"fidelity": 1.0},
                {"fidelity": 0.0}, {"fidelity": float("nan")}, {"fidelity": 0.5, "gene_length": 5000},
                {"selfish": -0.1}, {"innovation": 1.5}, {"inflow": -1}, {"replicators": -5},
                {"threshold": 1}, {"gene_types": 5, "threshold": 4}, {"inflow": True}, {"speed": 3}):
        with pytest.raises(ValueError):
            run(1, **bad)
    with pytest.raises(ValueError):
        run(1, rules=6)
    nothing = run(1, replicators=0)
    assert sim.conserved(nothing).ok and nothing.summary["outcome"] == "collapse"
    assert nothing.steps[0]["stage"] == "collapse" and nothing.steps[-1]["unused"] == nothing.steps[-1]["inflow_total"]
    hungry = run(1, inflow=0)
    assert sim.conserved(hungry).ok and hungry.summary["outcome"] == "collapse" and hungry.steps[-1]["built"] == 0
    assert hungry.steps[0]["stage"] == "soup" and hungry.steps[0]["cells"] > 0
    no_fast_gene = run(1, selfish=0.0)
    assert sim.conserved(no_fast_gene).ok and no_fast_gene.derived["takeover"] == "never"
    assert derived_line(no_fast_gene).text.endswith("takeover never.")
    one_gene = run(1, gene_types=1, threshold=8)
    assert sim.conserved(one_gene).ok and all(s["viable"] in (0.0, 1.0) for s in one_gene.steps)


def test_a_run_is_fast():
    worst = 0.0
    for seed in range(100, 130):
        start = time.perf_counter()
        canonical(seed)
        worst = max(worst, time.perf_counter() - start)
    assert worst < 2.0


# ---------------------------------------------------------------- the lines

def test_every_line_is_dense_and_short(all_lines):
    assert len(all_lines) > 30000
    for ln in all_lines:
        assert is_dense(ln.text), ln.text
        assert n_tokens(ln.text) <= 128, ln.text
        assert ln.topic == "cells"
        if ln.kind != "record":
            assert "." not in ln.prompt and "." not in ln.answer
            assert split_line(ln.text) == (ln.prompt, ln.answer)
            assert ln.kind in ("fact", "calc")
    assert max(n_tokens(ln.text) for ln in all_lines) <= 100
    assert max(n_tokens(ln.text) for ln in all_lines if ln.kind != "record") <= 24


def test_lines_of_a_long_seed_stay_short(sim):
    for seed in (9998, 123456789):
        ls = lines(sim.rollout(seed))
        assert ls[0].text.startswith(f"cells seed {' '.join(str(seed))} params. ")
        for ln in ls:
            assert is_dense(ln.text) and n_tokens(ln.text) <= 128, ln.text
            if ln.kind != "record":
                assert sim.check(ln.prompt, ln.answer).ok, ln.text


def test_pinned_lines_of_seed_33(rollouts):
    """The lines the module text shows, word for word."""
    r = rollouts[33]
    texts = [ln.text for ln in lines(r)]
    assert texts[0] == params_line(r).text == (
        "cells seed 3 3 params. gene_types 6. gene_length 4 0. fidelity 0 point 9 9 9. selfish 0 point 0 5. "
        "inflow 1 0 0 0 0 0 0. threshold 7 2. innovation 0 point 0 2. replicators 3 0 0.")
    assert texts[1] == derived_line(r).text == (
        "cells seed 3 3 derived. copy_success 0 point 9 6 1. sigma 5 point 6 2. l_max 1 7 3 0. "
        "full_set 0 point 9 9 9. copy_waste 1 point 5 7. doubling 1 point 4 7. takeover 8 5 point 5.")
    assert ("cells seed 3 3 step 0. cells 8. genes_per_cell 3 6. gene_types 6. genome_length 2 4 0. "
            "fidelity 0 point 9 9 9. error_load 0. viable 1. energy_per_cell 1 0. linked 0. dna 0. complex 0. "
            "selfish 0 point 1 5 3. stage soup.") in texts
    assert ("cells seed 3 3 step 3 1. cells 4 0 0. genes_per_cell 4 8 point 3. gene_types 7 point 2 8. "
            "genome_length 2 9 1. fidelity 0 point 9 9 9 9 8 2 6. error_load 0 point 0 0 6 0 2. viable 1. "
            "energy_per_cell 8 1 point 9. linked 1. dna 0 point 9 9 3. complex 0 point 5 3. "
            "selfish 0 point 1 4. stage complex_cells.") in texts
    for text in ("q cells seed 3 3 step 3 1 energy_per_cell. a 8 1 point 9.",
                 "q cells seed 3 3 step 3 0 next stage. a complex_cells.",
                 "q cells seed 3 3 final first_chromosome. a 2 5.", "q cells seed 3 3 final outcome. a complex_cells.",
                 "q cells seed 3 3 final energy_per_cell. a 1 4 6.", "q cells seed 3 3 final genome_length. a 3 7 5.",
                 "q cells seed 3 3 param threshold. a 7 2.", "q cells seed 3 3 derived l_max. a 1 7 3 0."):
        assert text in texts, text
    assert r.steps[31]["built"] == 24427080 and r.steps[60]["lost"] == 61108520 and r.steps[60]["unused"] == 237207172
    assert r.summary == {"stage": "complex_cells", "cells": 400, "genome_length": 375.0, "complex": 1.0,
                         "energy_per_cell": 146.0, "first_chromosome": 25, "outcome": "complex_cells"}


def test_line_kinds(rollouts):
    r = rollouts[33]
    ls = lines(r)
    records = [ln for ln in ls if ln.kind == "record"]
    assert len(records) == 2 + (STEPS + 1) and records[2].text.startswith("cells seed 3 3 step 0. cells 8. ")
    prompts = [ln.prompt for ln in ls if ln.kind != "record"]
    n_state = len(STATE_KEYS)
    assert len(prompts) == len(set(prompts)) == (len(PARAM_KEYS) + len(DERIVED_KEYS) + (STEPS + 1) * n_state
                                                 + STEPS * n_state + len(SUMMARY_KEYS))
    assert len(ls) == 1658 == len({ln.text for ln in ls})
    assert not any(f" {k} " in f" {ln.text} " for ln in ls for k in LEDGER_KEYS), "the ledger is not trained"
    assert len(lines(r, every=5)) < len(ls) / 4
    for where, kind in ((" param ", "fact"), (" derived ", "calc"), (" final ", "calc")):
        assert {ln.kind for ln in ls if where in ln.prompt} == {kind}
    assert {ln.kind for ln in ls if " next " in ln.prompt} == {"calc"}
    assert {ln.kind for ln in ls if " step " in ln.prompt and " next " not in ln.prompt and ln.kind != "record"} == {"fact"}


def test_an_empty_vessel_is_told_once(rollouts, sim):
    """After the first ``collapse`` step the steps are left out of the lines; the gate still knows them."""
    seen = 0
    for seed, r in rollouts.items():
        if r.summary["outcome"] != "collapse":
            continue
        seen += 1
        first = next(t for t, s in enumerate(r.steps) if s["stage"] == "collapse")
        ls = lines(r)
        records = [ln.text for ln in ls if ln.kind == "record" and " step " in ln.text]
        assert len(records) == first + 1 and records[-1].endswith("stage collapse.")
        assert f"q cells seed {dense_value(seed)} final outcome. a collapse." in [ln.text for ln in ls]
        assert all(s["cells"] == 0 and s["stage"] == "collapse" for s in r.steps[first:])
        assert first < STEPS
        for key in STATE_KEYS:
            if key != "stage":
                assert sim.check(f"cells seed {dense_value(seed)} step 6 0 {key}", "0").ok
                assert not sim.check(f"cells seed {dense_value(seed)} step 6 0 {key}", "1").ok
    assert seen >= 10


def test_numbers_use_three_significant_digits(all_lines):
    assert dense_value(0.99999) == "0 point 9 9 9 9 9" and dense_value(1230.0) == "1 2 3 0"
    assert dense_value(0.000123) == "1 point 2 3 e minus 4" and dense_value(7) == "7" and dense_value("soup") == "soup"
    assert dense_value(0.0) == "0" and dense_value(5000000) == "5 0 0 0 0 0 0" and dense_value(146.0) == "1 4 6"
    for ln in all_lines:
        if ln.kind == "record":
            continue
        key = ln.prompt.split()[-1]
        if ln.answer in STAGES + OUTCOMES + ("never",) or " param " in ln.prompt:
            continue
        value = parse_num(ln.answer)
        assert value is not None and value >= 0, ln.text
        if key in COUNT_KEYS:
            assert isinstance(value, int), ln.text
            continue
        shown = 1.0 - value if key == "fidelity" and value else value
        assert shown == pytest.approx(sig(shown), rel=1e-9, abs=1e-15), ln.text
        assert len(ln.answer.split()) <= 9, ln.text


# ---------------------------------------------------------------- the gate of the lines

def test_gate_agrees_with_every_line(sim, all_lines):
    for ln in all_lines:
        if ln.kind == "record":
            continue
        assert sim.owns(ln.prompt), ln.prompt
        v = sim.check(ln.prompt, ln.answer)
        assert v.ok and v.expected == ln.answer, (ln.text, v)
    assert cells.owns(all_lines[5].prompt) and cells.check(all_lines[5].prompt, all_lines[5].answer).ok


def test_gate_agrees_with_every_record_field(sim, all_lines):
    for ln in all_lines:
        if ln.kind != "record":
            continue
        head, *fields = [f.strip() for f in ln.text.rstrip(".").split(".")]
        head = head.replace(" params", " param")
        for f in fields:
            key, _, value = f.partition(" ")
            v = sim.check(f"{head} {key}", value)
            assert v.ok and v.expected == value, (ln.text, key)


def test_gate_rejects_altered_answers(sim, all_lines):
    rng = random.Random(5)
    questions = [ln for ln in all_lines if ln.kind != "record"]
    for ln in rng.sample(questions, 4000):
        truth = parse_num(ln.answer)
        if truth is None:  # a word: another word of its kind, or a number
            others = [w for w in STAGES + OUTCOMES + ("never", "1 5") if w != ln.answer]
            wrong = rng.choice(others)
        elif ln.prompt.split()[-1] == "fidelity" and " param " not in ln.prompt:
            wrong = dense_value(1.0 - 2.0 * (1.0 - truth)) if truth < 1 else "0 point 5"  # twice the error rate
        elif truth == 0:
            wrong = rng.choice(["1", "0 point 0 1", "never"])
        else:
            wrong = "9 " + ln.answer  # a changed leading digit: far outside 5 percent
        v = sim.check(ln.prompt, wrong)
        assert not v.ok and v.expected == ln.answer, (ln.text, wrong, v)
    assert not sim.check("cells seed 3 3 final outcome", "cells").ok
    assert sim.check("cells seed 3 3 final outcome", "cells").expected == "complex_cells"
    assert not sim.check("cells seed 3 3 final first_chromosome", "never").ok
    assert not sim.check("cells seed 3 3 final first_chromosome", "3 0").ok
    assert sim.check("cells seed 7 final first_chromosome", "never").ok
    assert not sim.check("cells seed 7 final first_chromosome", "2 5").ok


def test_tolerance_is_five_percent_for_numbers_and_exact_for_counts_and_parameters(sim, rollouts):
    r = rollouts[33]
    energy = r.summary["energy_per_cell"]
    for factor, ok in ((1.03, True), (0.97, True), (1.1, False), (0.9, False)):
        assert sim.check("cells seed 3 3 final energy_per_cell", dense_value(sig(energy * factor))).ok is ok
    assert sim.check("cells seed 3 3 final cells", "4 0 0").ok
    assert not sim.check("cells seed 3 3 final cells", "3 9 9").ok and not sim.check("cells seed 3 3 final cells", "4 0 1").ok
    assert sim.check("cells seed 3 3 param fidelity", "0 point 9 9 9").ok
    assert not sim.check("cells seed 3 3 param fidelity", "0 point 9 9 8").ok
    assert not sim.check("cells seed 3 3 param inflow", "1 0 0 0 0 0 1").ok
    assert sim.check("cells seed 3 3 derived l_max", "1 7 0 0").ok and not sim.check("cells seed 3 3 derived l_max", "1 5 0 0").ok
    # the fidelity is judged by its error rate: 0.9999826 is an error of 1.74e-5
    assert r.steps[31]["fidelity"] == pytest.approx(0.9999826, abs=1e-12)
    assert sim.check("cells seed 3 3 step 3 1 fidelity", "0 point 9 9 9 9 8 2 6").ok
    assert sim.check("cells seed 3 3 step 3 1 fidelity", "0 point 9 9 9 9 8 3").ok  # error 1.7e-5
    assert not sim.check("cells seed 3 3 step 3 1 fidelity", "0 point 9 9 9 9 8").ok  # error 2e-5
    assert not sim.check("cells seed 3 3 step 3 1 fidelity", "1").ok and not sim.check("cells seed 3 3 step 3 1 fidelity", "0 point 9 9 9").ok
    assert sim.check("cells seed 3 3 step 0 linked", "0").ok and not sim.check("cells seed 3 3 step 0 linked", "0 point 0 0 1").ok


def test_gate_rejects_answers_that_are_not_finite_numbers(sim):
    for prompt in ("cells seed 3 3 final energy_per_cell", "cells seed 3 3 step 3 1 genes_per_cell",
                   "cells seed 3 3 final cells", "cells seed 3 3 param fidelity", "cells seed 3 3 derived copy_waste",
                   "cells seed 3 3 step 3 1 fidelity"):
        for answer in ("1 e 9 9 9", "minus 1 e 9 9 9", "", "point", "e", "yes", "1 4 6 units", "1 4 . 6",
                       "minus 1 4 6", "never"):
            v = sim.check(prompt, answer)
            assert not v.ok and v.expected is not None, (prompt, answer, v)
    assert sim.check("cells seed 3 3 final energy_per_cell", "1 e 9 9 9").reason == "not a number"
    assert not sim.check("cells seed 3 3 final stage", "4 0 0").ok and not sim.check("cells seed 3 3 final stage", "").ok


@pytest.mark.parametrize("prompt", [
    "turkey capital", "q spoon color", "carbon protons", "calc 1 2 plus 7", "balance h 2 plus o 2", "",
    "planets seed 7 step 3 planets", "planets seed 7 final gas", "life seed 7 final outcome",
    "life seed 7 step 1 2 next replicators", "life seed 7 param mu", "gravity seed 7 final clumps",
    "nucleo seed 3 final helium_fraction", "chem seed 7 final water", "world seed 3 era planets",
    "stars seed 7 step 3 stage", "bodies seed 7 step 3 cells", "senses seed 7 final outcome",
    "signals seed 7 say predator near", "society seed 7 final cells", "world7 seed 7 era cells",
    "cells predict copy_success fidelity 0 point 9 9 length 1 0 0", "cells predict nutrient_share inflow 1 0 cells 2",
    "cells", "cells seed 7", "cells seed 7 step 3", "cells seed 7 step 3 planets", "cells seed 7 step 6 1 cells",
    "cells seed 7 step 6 0 next cells", "cells seed 7 final gene_types", "cells seed 7 final viable",
    "cells seed x final cells", "cells seed 7 final", "cells seed 7 param outcome", "cells seed 7 params gene_types",
    "cells seed 7 derived fidelity", "cells seed 7 param l_max", "cells seed 12 step 3 cells",
    "cells seed 7 step 3 next", "cells seed 7 step 3 cells viable", "cells seed 0 7 final cells",
    "cells seed 7 step 0 3 cells", "cells seed 7 step 0 0 cells", "cells seed minus 7 final cells",
    "cells seed 7 point 5 final cells", "q cells seed 7 final cells", "cells seed 7 final cells extra",
    "cells seed 7 step 3 next next cells", "cells  seed 7 final cells", "cells seed 7 final cells ",
    " cells seed 7 final cells", "cells seed 7 step 3 built", "cells seed 7 step 3 n_viable",
    "cells seed 7 step 3 generation", "cells seed 7 step 3 next unused", "cells seed 7 era cells",
])
def test_does_not_own_other_prompts(sim, prompt):
    assert not sim.owns(prompt) and not cells.owns(prompt)
    v = sim.check(prompt, "1")
    assert v.ok is False and v.expected is None and v.reason == "not my question"


@pytest.mark.parametrize("prompt", [
    "cells seed 7 step 0 cells", "cells seed 7 step 6 0 stage", "cells seed 7 step 5 9 next selfish",
    "cells seed 1 2 3 4 step 1 2 next energy_per_cell", "cells seed 7 final first_chromosome",
    "cells seed 7 final outcome", "cells seed 7 param replicators", "cells seed 7 derived takeover",
    "cells seed 0 final cells", "cells seed 7 step 1 0 fidelity", "cells seed 7 step 1 0 gene_types",
])
def test_owns_its_prompts(sim, prompt):
    assert sim.owns(prompt)
    assert sim.check(prompt, "nonsense words").expected is not None
    assert not LESSONS.owns(prompt)


def test_generate_is_deterministic(sim):
    one = [ln.text for ln in sim.generate(random.Random(11), 3)]
    two = [ln.text for ln in Cells().generate(random.Random(11), 3)]
    assert one == two and len(one) > 200
    assert one != [ln.text for ln in sim.generate(random.Random(12), 3)]


# ---------------------------------------------------------------- the gate of the simulation

def test_conserved_passes_on_forty_seeds_and_on_other_parameters(sim, rollouts):
    for seed, r in rollouts.items():
        v = sim.conserved(r)
        assert v.ok, (seed, v)
        assert cells.conserved(r).ok
        light = copy.copy(r)
        light.snaps = []
        assert sim.conserved(light).ok, "the steps alone are judged too"
    for seed in range(200, 240):  # parameters that are not the seed's own
        r = run(seed, **random_params(random.Random(seed + 1000)))
        v = sim.conserved(r)
        assert v.ok, (seed, v)


def test_the_nutrient_ledger_is_exact(rollouts):
    """Every base that came in was built into a gene, wasted on a failed copy, flowed out unused
    or is still stored in a living cell; cumulative, in whole bases."""
    for r in rollouts.values():
        p = r.params
        for t, (s, snap) in enumerate(zip(r.steps, r.snaps)):
            assert s["generation"] == t * GENS_PER_STEP and s["inflow_total"] == p["inflow"] * s["generation"]
            assert s["built"] + s["waste"] + s["unused"] + s["stored"] == s["inflow_total"]
            assert s["bases"] == r.steps[0]["bases"] + s["built"] - s["lost"]
            assert s["bases"] == s["genes"] * p["gene_length"] == sum(sum(c[5:]) for c in snap) * p["gene_length"]
            assert s["stored"] == sum(c[4] for c in snap) and s["failures"] <= s["attempts"]
            if t:
                before = r.steps[t - 1]
                assert all(s[k] >= before[k] for k in ("built", "waste", "unused", "lost", "attempts", "failures"))
        first = r.steps[0]
        assert first["built"] == first["waste"] == first["unused"] == first["lost"] == first["stored"] == 0
        assert first["cells"] == min(CAP, p["replicators"] // (p["threshold"] // 2))
        assert first["genes"] == first["cells"] * (p["threshold"] // 2)
    # while every gene is copied on its own, a failed copy wastes one gene and a good one builds one
    loose = [r for r in rollouts.values() if r.params["innovation"] == 0]
    assert len(loose) >= 5
    for r in loose:
        last, length = r.steps[-1], r.params["gene_length"]
        assert last["waste"] == last["failures"] * length
        assert last["built"] == (last["attempts"] - last["failures"]) * length
        assert all(s["linked"] == s["dna"] == s["complex"] == 0 for s in r.steps)


def test_gene_counts_are_never_negative_and_the_shares_fit_the_cells(rollouts):
    for r in rollouts.values():
        length = r.params["gene_length"]
        for s, snap in zip(r.steps, r.snaps):
            assert len(snap) == s["cells"] <= CAP
            viable = 0
            for cell in snap:
                types, linked, dna, partner, stored = cell[:5]
                genes = cell[5:]
                assert len(genes) == types and 1 <= types <= GMAX and min(genes) >= 0 and sum(genes) > 0
                assert {linked, dna, partner} <= {0, 1} and 0 <= stored < length * (types if linked else 1)
                viable += min(genes) > 0
                if linked:
                    assert min(genes) == max(genes) >= 1, "a chromosome carries one gene of each type"
            assert viable == s["n_viable"] and s["n_linked"] <= s["n_viable"] <= s["cells"]
            assert s["n_working"] <= min(s["n_dna"], s["n_complex"])
            if s["cells"]:
                assert s["viable"] == sig(viable / s["cells"]) and 0 <= s["viable"] <= 1
                assert s["linked"] == sig(s["n_linked"] / s["cells"]) and s["dna"] == sig(s["n_dna"] / s["cells"])
                assert s["complex"] == sig(s["n_complex"] / s["cells"])
                assert s["genes_per_cell"] == sig(s["genes"] / s["cells"])
                assert s["gene_types"] == sig(s["types_sum"] / s["cells"])
                assert s["genome_length"] == sig(s["types_sum"] * length / s["cells"])
                assert s["selfish"] == sig(s["selfish_genes"] / s["genes"])
                assert r.params["fidelity"] - 1e-12 <= s["fidelity"] < 1 and ENERGY <= s["energy_per_cell"] <= 16 * ENERGY
            else:
                assert all(s[k] == 0 for k in STATE_KEYS if k not in ("stage", "error_load"))
                assert s["attempts"] > 0 or s["error_load"] == 0, "the last copies may have failed"
            assert census(snap) == {k: s[k] for k in CENSUS_KEYS}


def test_stages_and_summary_follow_from_the_counts(rollouts):
    assert stage_of(0, 8, 8, 0, 0, 0) == "soup" and stage_of(5, 8, 3, 0, 0, 0) == "soup"
    assert stage_of(5, 0, 0, 0, 0, 0) == "collapse" == stage_of(0, 0, 0, 0, 0, 0)
    assert stage_of(5, 8, 8, 0, 0, 0) == "protocells" and stage_of(5, 8, 8, 3, 4, 3) == "cells"
    assert stage_of(5, 8, 8, 4, 3, 3) == "chromosomes" and stage_of(5, 8, 8, 8, 8, 4) == "complex_cells"
    for r in rollouts.values():
        for s in r.steps:
            assert s["stage"] == stage_of(s["generation"], s["cells"], s["n_viable"], s["n_linked"], s["n_dna"],
                                          s["n_working"])
        last = r.steps[-1]
        assert r.summary == summarise(r.steps)
        for key in ("stage", "cells", "genome_length", "complex", "energy_per_cell"):
            assert r.summary[key] == last[key]
        linked = [s["generation"] for s in r.steps if s["cells"] and 2 * s["n_linked"] >= s["cells"]]
        assert r.summary["first_chromosome"] == (linked[0] if linked else "never")
        assert r.summary["outcome"] == {"collapse": "collapse", "soup": "protocells", "protocells": "protocells",
                                        "cells": "cells", "chromosomes": "cells",
                                        "complex_cells": "complex_cells"}[last["stage"]]
        assert r.steps[0]["stage"] in ("soup", "collapse")
        assert all(s["cells"] == 0 for s in r.steps[next((t for t, s in enumerate(r.steps) if s["cells"] == 0), 61):])
        assert r.derived == derived(r.params)


def test_conserved_catches_tampering(sim, rollouts):
    base = rollouts[33]
    length = base.params["gene_length"]

    def broken(change) -> bool:
        r = copy.deepcopy(base)
        change(r)
        return not sim.conserved(r).ok

    assert not broken(lambda r: None)
    assert broken(lambda r: r.steps[10].update(unused=r.steps[10]["unused"] + 1))
    assert broken(lambda r: r.steps[10].update(built=r.steps[10]["built"] + length))
    assert broken(lambda r: r.steps[10].update(waste=r.steps[10]["waste"] + 1, unused=r.steps[10]["unused"] - 2))
    assert broken(lambda r: r.steps[10].update(waste=0, unused=r.steps[10]["unused"] + r.steps[10]["waste"]))
    assert broken(lambda r: r.steps[10].update(stored=r.steps[10]["stored"] - 1, unused=r.steps[10]["unused"] + 1))
    assert broken(lambda r: r.steps[20].update(lost=r.steps[20]["lost"] - length))
    assert broken(lambda r: r.steps[12].update(cells=399))
    assert broken(lambda r: r.steps[12].update(viable=0.5))
    assert broken(lambda r: r.steps[12].update(n_viable=r.steps[12]["n_viable"] - 1))
    assert broken(lambda r: r.steps[12].update(error_load=r.steps[12]["error_load"] * 2))
    assert broken(lambda r: r.steps[40].update(energy_per_cell=10.0))
    assert broken(lambda r: r.steps[40].update(fidelity=0.999))
    assert broken(lambda r: r.steps[5].update(stage="cells"))
    assert broken(lambda r: r.steps[0].update(stage="protocells"))
    assert broken(lambda r: r.steps[3].update(generation=14))
    assert broken(lambda r: r.steps[30].update(n_working=r.steps[30]["n_dna"] + 1))
    assert broken(lambda r: r.steps[30].update(attempts=r.steps[29]["attempts"] - 1))
    assert broken(lambda r: r.steps[30].update(selfish=-0.1))
    assert broken(lambda r: r.steps[30].update(built=float("nan")))
    assert broken(lambda r: r.steps.pop())
    assert broken(lambda r: r.steps[7].pop("selfish"))
    assert broken(lambda r: r.summary.update(outcome="cells"))
    assert broken(lambda r: r.summary.update(first_chromosome=30))
    assert broken(lambda r: r.derived.update(l_max=2000.0))
    assert broken(lambda r: r.params.update(inflow=r.params["inflow"] + 1))
    assert broken(lambda r: r.params.update(gene_length=41))
    assert broken(lambda r: r.params.update(extra=1))
    assert broken(lambda r: r.params.update(fidelity=1.5))
    # the cells themselves
    assert broken(lambda r: r.snaps.pop())
    assert broken(lambda r: r.snaps[10].pop())
    assert broken(lambda r: r.snaps[10][0].__setitem__(5, -1))
    assert broken(lambda r: r.snaps[10][0].__setitem__(5, r.snaps[10][0][5] + 1))
    assert broken(lambda r: r.snaps[10][0].__setitem__(2, 1 - r.snaps[10][0][2]))
    assert broken(lambda r: r.snaps[10][0].__setitem__(4, 10 ** 6))
    assert broken(lambda r: r.snaps[10][0].__setitem__(0, 40))
    assert broken(lambda r: r.snaps[10][0].__setitem__(1, 2))

    def unequal_chromosome(r):  # the totals still add up, the linked cell does not
        cell = next(c for c in r.snaps[60] if c[1])
        cell[5] += 1
        cell[6] -= 1
    assert broken(unequal_chromosome)

    def shifted(r):  # a gene moved from one cell to another of the same kind: counts the same, cells not
        r.snaps[0][0][5] += 3
        r.snaps[0][1][5] -= 3
        r.snaps[0][1][5] = max(r.snaps[0][1][5], -1)
    assert broken(shifted) or base.snaps[0][1][5] >= 3


def test_error_catastrophe_is_gated(sim):
    """Push the fidelity down: beyond the error threshold no run keeps its genes, and the gate
    fails one that claims to."""
    kw = dict(gene_types=3, threshold=24, gene_length=160, selfish=0.05, inflow=2000000, innovation=0.0,
              replicators=1000)
    for fidelity in (0.98, 0.95):
        for seed in (1, 2, 3):
            r = run(seed, fidelity=fidelity, **kw)
            assert r.params["gene_length"] > r.derived["l_max"]
            v = sim.conserved(r)
            assert v.ok and v.reason.endswith("beyond the error threshold")
            assert r.steps[-1]["keeper_gens"] == 0 and r.steps[-1]["best_growth"] < 1 / SURVIVAL_ODDS
            assert r.summary["outcome"] == "collapse" and r.steps[-1]["bases"] == 0
    healthy = run(1, fidelity=0.999, **kw)
    assert sim.conserved(healthy).ok and healthy.summary["cells"] == CAP and healthy.steps[-1]["keeper_gens"] > 0
    assert not sim.conserved(healthy).reason.endswith("beyond the error threshold")
    # the same steps, but the ledger now says that no cell was ever under its threshold
    claim = copy.deepcopy(healthy)
    for s in claim.steps:
        s.update(keeper_gens=0, best_growth=min(s["best_growth"], 0.01))
    v = sim.conserved(claim)
    assert not v.ok and v.reason == "a cell under its error threshold, but none in the ledger"
    claim.snaps = []
    v = sim.conserved(claim)
    assert not v.ok and v.reason.startswith("error catastrophe") and v.expected == "0"
    for s in claim.steps:
        s.update(best_growth=min(healthy.steps[-1]["best_growth"], s["best_growth"] * 60))
    assert not sim.conserved(claim).ok, "no cell under its threshold can grow faster than cells die"
    # the threshold in the tables is the lesson's, with sigma = 9 * rate of the kind of cell
    t = tables(0.98, 160)
    rate = expression(energy_per_gene(ENERGY, 3, 0, 1.0))
    assert t.rate[0, 3] == rate == start_rate(healthy.params)
    assert t.l_max[0, 3] == error_threshold(superiority(rate), 0.98) < 160
    assert tables(0.999, 160).l_max[0, 3] == error_threshold(superiority(rate), 0.999) > 160
    assert t.copy[0, 1] == copy_success(0.98, 160) and t.copy[0, 1] * superiority(rate) < 1


# ---------------------------------------------------------------- the lessons

def test_lessons_are_accepted_and_are_the_simulations_own_functions(lessons):
    assert len(lessons) == 400 and len({ln.prompt for ln in lessons}) > 380
    own = {"copy_success": copy_success, "error_threshold": error_threshold, "under_threshold": under_threshold,
           "full_set": full_set,
           "expected_waste": expected_waste, "nutrient_share": nutrient_share, "energy_per_gene": energy_per_gene,
           "generations_to_fix": generations_to_fix, "doubling_time": doubling_time, "balance": balance,
           "efficiency": efficiency, "expression": expression, "partner_gain": partner_gain}
    assert set(own) == set(RULES) and len(own) >= 8
    for name, fn in own.items():
        assert RULES[name].compute is fn is getattr(cells, name), name
    seen = set()
    for ln in lessons:
        assert is_dense(ln.text) and n_tokens(ln.text) <= 128 and ln.topic == "predict_cells" and ln.kind == "calc"
        assert "." not in ln.prompt and "." not in ln.answer
        v = LESSONS.check(ln.prompt, ln.answer)
        assert v.ok and v.expected == ln.answer, (ln.text, v)
        name, values = LESSONS.parse(ln.prompt)
        truth = own[name](*values)
        assert ln.answer == (truth if isinstance(truth, str) else num(truth)), (ln.text, truth)
        assert LESSONS.owns(ln.prompt) and not simulation().owns(ln.prompt)
        assert ln.meta["split_key"] == LESSONS.split_key(ln.prompt)
        seen.add(name)
    assert seen == set(RULES), "400 lessons reach every rule"
    assert max(n_tokens(ln.text) for ln in lessons) <= 40
    assert [ln.text for ln in LESSONS.generate(random.Random(1), 400)] == [ln.text for ln in lessons]
    assert [ln.text for ln in LESSONS.generate(random.Random(2), 400)] != [ln.text for ln in lessons]


def test_lessons_reject_altered_answers_and_foreign_prompts(lessons):
    rng = random.Random(3)
    for ln in lessons:
        truth = parse_num(ln.answer)
        if ln.answer in ("yes", "no"):  # a word is judged exactly
            for wrong in ("no" if ln.answer == "yes" else "yes", "1", "0", "", "yes no", "never"):
                v = LESSONS.check(ln.prompt, wrong)
                assert not v.ok and v.expected == ln.answer, (ln.text, wrong)
            continue
        wrong = "1" if truth == 0 else "9 " + ln.answer
        assert not LESSONS.check(ln.prompt, wrong).ok, (ln.text, wrong)
        assert not LESSONS.check(ln.prompt, "yes").ok and not LESSONS.check(ln.prompt, "").ok
        words = ln.answer.split()
        place = rng.choice([i for i, w in enumerate(words) if w.isdigit()])
        words[place] = str((int(words[place]) + rng.randint(1, 9)) % 10)
        changed = " ".join(words)
        if parse_num(changed) != truth:
            assert not LESSONS.check(ln.prompt, changed).ok, (ln.text, changed)
        assert not LESSONS.check(ln.prompt, "never").ok and not LESSONS.check(ln.prompt, "1 e 9 9 9").ok
    for prompt in ("cells seed 7 final cells", "life predict survival_probability length 4 decay 1",
                   "cells predict copy_success fidelity 0 point 9 9", "cells predict copy_success length 5 fidelity 0 point 9 9",
                   "cells predict copy_success fidelity 0 point 9 9 length 0", "cells predict copy_success fidelity 2 length 5",
                   "cells predict nutrient_share inflow 1 0 0 cells 0", "cells predict full_set copies 1 point 5 types 2",
                   "cells predict nothing rate 1", "bodies predict copy_success fidelity 0 point 9 9 length 5",
                   "cells predict copy_success fidelity 0 point 9 9 length 6 4 1",
                   "cells predict under_threshold length 0 sigma 5 fidelity 0 point 9 9",
                   "cells predict under_threshold length 5 0 sigma 1 fidelity 0 point 9 9",
                   "cells predict under_threshold sigma 5 length 5 0 fidelity 0 point 9 9",
                   "cells predict full_set copies 2 types 3 3", "cells predict partner_gain fidelity 0 point 9 9 9"):
        assert not LESSONS.owns(prompt), prompt
        assert LESSONS.check(prompt, "1").reason == "not my question"


def test_lessons_shown_in_the_module_text(lessons):
    """The lesson lines of the module text, word for word: they are among the 400 of seed 1."""
    shown = [
        "q cells predict copy_success fidelity 0 point 9 9 4 8 6 length 5 1 3. a 0 point 0 7 1 1.",
        "q cells predict error_threshold sigma 2 4 point 2 fidelity 0 point 9 2 1 9. a 4 0 point 8.",
        "q cells predict under_threshold length 1 0 3 sigma 5 point 9 fidelity 0 point 9 9 7 1. a yes.",
        "q cells predict full_set copies 6 types 2 3. a 0 point 6 9 6 1.",
        "q cells predict expected_waste fidelity 0 point 9 9 0 1 5 length 1 9 5. a 1 6 6 point 7.",
        "q cells predict nutrient_share inflow 3 6 3 3 9 3 4 cells 2 1 7. a 1 6 7 4 6.",
        "q cells predict energy_per_gene energy 2 9 point 1 genes 6 4 partner 1 gain 5 9 point 3. a 2 6 point 9 6.",
        "q cells predict generations_to_fix population 1 3 4 0 1 advantage 1 point 8. a 5 point 2 7 9.",
        "q cells predict doubling_time rate 0 point 5 7 4. a 1 point 5 2 8.",
    ]
    texts = {ln.text for ln in lessons}
    for text in shown:
        assert text in texts and text in cells.__doc__, text
        prompt, answer = split_line(text)
        assert LESSONS.check(prompt, answer).ok
    # each by hand, from the formula in the module text
    assert 0.99486 ** 513 == pytest.approx(0.0711, rel=1e-3) and (63 / 64) ** 23 == pytest.approx(0.6961, rel=1e-3)
    assert 195 * (1 - 0.99015 ** 195) == pytest.approx(166.7, rel=1e-3)
    assert 29.1 * 59.3 / 64 == pytest.approx(26.96, rel=1e-3)
    assert 103 < math.log(5.9) / (1 - 0.9971) and math.log(24.2) / (1 - 0.9219) == pytest.approx(40.8, rel=1e-3)
    assert 3633934 // 217 == 16746 and math.log(13401) / 1.8 == pytest.approx(5.279, abs=1e-3)
    assert math.log(2) / math.log(1.574) == pytest.approx(1.528, abs=1e-3)


def test_lesson_records_are_dense():
    records = LESSONS.records()
    assert len(records) == len(RULES) == 13
    for ln in records:
        assert ln.kind == "record" and is_dense(ln.text) and n_tokens(ln.text) <= 128, ln.text
        assert ln.text.startswith("cells predict ")
    text = {ln.text.split(".")[0]: ln.text for ln in records}
    for name in ("generations_to_fix", "balance", "efficiency", "expression", "partner_gain"):
        assert text["cells predict " + name].startswith(f"cells predict {name}. toy rule"), name
    assert "lane and martin" in text["cells predict energy_per_gene"]
    assert "after eigen" in text["cells predict error_threshold"]
    assert "in this toy sigma" in text["cells predict error_threshold"]


def test_lesson_rules_by_hand():
    assert copy_success(0.99, 100) == pytest.approx(0.99 ** 100, rel=1e-12) and copy_success(0.5, 3) == 0.125
    assert copy_success(0.999, 0) == 1.0 and copy_success(1.0, 640) == 1.0
    assert error_threshold(9.0, 0.99) == pytest.approx(math.log(9) / 0.01)
    assert full_set(1, 1) == 0.5 and full_set(2, 2) == 0.5625 and full_set(6, 6) == pytest.approx((63 / 64) ** 6)
    assert expected_waste(0.99, 100) == pytest.approx(100 * (1 - 0.99 ** 100))
    assert nutrient_share(1000000, 7) == 142857 and nutrient_share(5, 7) == 0 and type(nutrient_share(10, 2)) is int
    assert energy_per_gene(10.0, 4, 0, 16.0) == 2.5 and energy_per_gene(10.0, 4, 1, 16.0) == 40.0
    assert generations_to_fix(400, 0.2) == pytest.approx(math.log(400) / 0.2)
    assert doubling_time(1.0) == pytest.approx(1.0) and doubling_time(0.5) == pytest.approx(math.log(2) / math.log(1.5))
    assert balance(4, 4, 4) == 1.0 and balance(0, 5, 5) == 0.0 == balance(0, 0, 0)
    assert balance(9, 6, 5) == pytest.approx(9 / (20 * (1 / 9 + 1 / 6 + 1 / 5)))
    assert balance(1, 1, 10) < balance(2, 4, 6) < balance(3, 4, 5) < 1
    assert efficiency(8) == 0.5 and efficiency(24) == 0.75 and expression(1.0) == 0.5 and expression(9.0) == 0.9
    assert superiority(1.0) == 9.0 and superiority(0.5) == 4.5
    # sigma as a count: rate attempts in each of the 1 / DEATH generations of a cell's life, each copy
    # still there after the generation it was made in with chance 1 - DEATH
    assert superiority(0.6) == pytest.approx(0.6 * (1 / DEATH) * (1 - DEATH))
    assert under_threshold(160, 6.92, 0.995) == "yes" and under_threshold(160, 6.92, 0.98) == "no"
    assert under_threshold(219, 9.0, 0.99) == "yes" and under_threshold(220, 9.0, 0.99) == "no"  # ln 9 / 0.01 = 219.7
    assert under_threshold(1, 1.0, 0.999) == "no" == under_threshold(1, 0.5, 0.999)
    assert LESSONS.check("cells predict under_threshold length 2 1 9 sigma 9 fidelity 0 point 9 9", "yes").ok
    assert LESSONS.check("cells predict under_threshold length 2 2 0 sigma 9 fidelity 0 point 9 9", "no").ok
    assert not LESSONS.check("cells predict under_threshold length 2 2 0 sigma 9 fidelity 0 point 9 9", "yes").ok
    assert partner_gain(0.999) == pytest.approx(1.0, abs=1e-3) and 14.5 < partner_gain(0.99999) < 14.6
    assert LESSONS.check("cells predict nutrient_share inflow 1 0 0 0 0 0 0 cells 7", "1 4 2 8 5 7").ok
    assert not LESSONS.check("cells predict nutrient_share inflow 1 0 0 0 0 0 0 cells 7", "1 4 2 8 5 8").ok
    assert LESSONS.check("cells predict energy_per_gene energy 1 0 genes 4 partner 1 gain 1 6", "4 0").ok
    assert LESSONS.check("cells predict balance first 4 second 4 third 4", "1").ok
    assert LESSONS.check("cells predict full_set copies 2 types 2", "0 point 5 6 2 5").ok


def test_the_tables_of_a_run_are_built_from_the_lesson_rules():
    t = tables(0.999, 40)
    assert t.fidelity == (0.999, 1.0 - (1.0 - 0.999) / 100) and t.gain == (partner_gain(0.999), partner_gain(t.fidelity[1]))
    for genes in (1, 2, 6, 17, GMAX):
        assert t.eff[genes] == efficiency(genes)
        assert t.copy[0, genes] == copy_success(0.999, 40 * genes) and t.copy[1, genes] == copy_success(t.fidelity[1], 40 * genes)
        for kind in range(8):
            linked, dna, partner = kind >> 2, (kind >> 1) & 1, kind & 1
            pace = (0.7 if linked else 1.0) * (0.9 if dna else 1.0) * (0.8 if partner else 1.0)
            rate = pace * expression(energy_per_gene(ENERGY, genes, partner, t.gain[dna]))
            assert t.rate[kind, genes] == rate < 1
            assert t.l_max[kind, genes] == error_threshold(superiority(rate), t.fidelity[dna])
            piece = 40 * (genes if linked else 1)
            assert t.keeper[kind, genes] == (under_threshold(piece, superiority(rate), t.fidelity[dna]) == "yes")
            assert t.keeper[kind, genes] == (piece < t.l_max[kind, genes])
    # a partner without dna gives next to nothing and costs a fifth of the pace
    assert t.rate[1, 6] < 0.81 * t.rate[0, 6] and t.rate[3, 6] > 1.2 * t.rate[2, 6]


def test_derived_record_follows_from_the_lessons(rollouts):
    for r in rollouts.values():
        p, d = r.params, r.derived
        q = copy_success(p["fidelity"], p["gene_length"])
        rate = expression(energy_per_gene(ENERGY, p["gene_types"], 0, 1.0))
        assert d == {"copy_success": sig(q), "sigma": sig(superiority(rate)),
                     "l_max": sig(error_threshold(superiority(rate), p["fidelity"])),
                     "full_set": sig(full_set(p["threshold"] // p["gene_types"], p["gene_types"])),
                     "copy_waste": sig(expected_waste(p["fidelity"], p["gene_length"])),
                     "doubling": sig(doubling_time(rate * q)),
                     "takeover": sig(generations_to_fix(p["threshold"], p["selfish"]))}
        assert d == {k: sig(v) for k, v in derived_exact(p).items()}


def test_full_set_is_the_chance_of_the_simulations_coin_tosses():
    """``split`` is what a dividing cell does; the lesson is its exact expectation."""
    rng = _Rng(1)
    for copies, types in ((1, 2), (2, 3), (3, 6), (4, 4), (6, 6)):
        mothers = np.full((20000, types), copies, dtype=np.int64)
        first = split(rng, mothers)
        assert first.shape == mothers.shape and first.min() >= 0 and (first <= mothers).all()
        got = float(((first > 0).all(axis=1)).mean())
        other = float((((mothers - first) > 0).all(axis=1)).mean())
        expect = full_set(copies, types)
        tolerance = 4 * math.sqrt(expect * (1 - expect) / 20000) + 1e-9
        assert abs(got - expect) < tolerance and abs(other - expect) < tolerance, (copies, types, got, expect)
        assert abs(float(first.mean()) - copies / 2) < 0.02


def test_copy_success_waste_and_doubling_are_what_a_generation_does():
    """A vessel of fed, balanced protocells: one generation fails 1 - copy_success of its
    attempts, wastes ``expected_waste`` bases per attempt and grows the genes by rate * copy_success."""
    for fidelity, length, types in ((0.995, 80, 3), (0.999, 320, 2), (0.98, 40, 4)):
        p = cells._params(dict(gene_types=types, gene_length=length, fidelity=fidelity, selfish=0.0, inflow=10 ** 11,
                               threshold=1000, innovation=0.0, replicators=CAP * 500))
        vessel = _Vessel(_Rng(11), p)
        start = int(vessel.genes.sum())
        assert len(vessel.types) == CAP and start == CAP * 500
        vessel.advance()
        led = vessel.led
        q, rate = copy_success(fidelity, length), start_rate(p)
        attempts = led["attempts"]
        assert attempts > 50000 and led["waste"] == led["failures"] * length
        assert led["failures"] / attempts == pytest.approx(1 - q, abs=4 * math.sqrt(q * (1 - q) / attempts))
        assert led["waste"] / attempts == pytest.approx(expected_waste(fidelity, length), rel=0.05)
        assert attempts / start == pytest.approx(rate, rel=0.02), "balance is 1 within a percent with 500 genes"
        growth = led["built"] / (start * length)
        assert growth == pytest.approx(rate * q, rel=0.03)
        assert doubling_time(growth) == pytest.approx(derived(p)["doubling"], rel=0.05)
        # the share offered to a cell is the lesson's, and what is not taken flows out
        share = nutrient_share(p["inflow"], CAP)
        assert led["unused"] + led["built"] + led["waste"] + int(vessel.store.sum()) + 0 >= share * CAP
        assert led["inflow_total"] == p["inflow"] == led["built"] + led["waste"] + led["unused"] + int(vessel.store.sum())


def test_a_hungry_cell_takes_balance_times_efficiency_of_its_share():
    """Little food and nothing else limiting: what the cells spend is share * balance * efficiency."""
    p = cells._params(dict(gene_types=3, gene_length=1, fidelity=0.999, selfish=0.0, inflow=CAP * 200,
                           threshold=1000, innovation=0.0, replicators=CAP * 500))
    vessel = _Vessel(_Rng(5), p)
    m = cells._balance(vessel.genes[:, :3], vessel.types)
    assert m.min() > 0.97 and m.max() <= 1.0
    expect = np.floor(nutrient_share(p["inflow"], CAP) * m * efficiency(3)).astype(np.int64)
    rate = start_rate(p)
    vessel.advance()
    spent = vessel.led["built"] + vessel.led["waste"]
    assert nutrient_share(p["inflow"], CAP) == 200 and int(expect.sum()) == pytest.approx(CAP * 200 * 3 / 11, rel=0.02)
    assert spent == pytest.approx(rate * int(expect.sum()), rel=0.03)
    assert vessel.led["unused"] + spent + int(vessel.store.sum()) == p["inflow"]


# ---------------------------------------------------------------- plausibility targets

def test_outcomes_of_forty_seeds(rollouts):
    outcomes = [r.summary["outcome"] for r in rollouts.values()]
    assert set(outcomes) == set(OUTCOMES), "every outcome occurs"
    assert 12 <= outcomes.count("collapse") <= 26 and outcomes.count("cells") >= 8
    assert 1 <= outcomes.count("complex_cells") <= 8, "complex cells are a minority"
    stages = {s["stage"] for r in rollouts.values() for s in r.steps}
    assert stages == set(STAGES)
    for r in rollouts.values():
        assert r.steps[0]["stage"] == "soup" and r.steps[0]["cells"] >= 1
        assert max(s["cells"] for s in r.steps) <= CAP
        if r.params["innovation"] == 0:
            assert r.summary["outcome"] in ("collapse", "protocells") and r.summary["first_chromosome"] == "never"
            assert all(s["gene_types"] <= r.params["gene_types"] for s in r.steps)


def test_low_fidelity_collapses_or_stays_at_protocells(sim, rollouts):
    """Beyond the error threshold a population does not keep its genes."""
    beyond = [r for r in rollouts.values() if over_threshold(r)]
    assert len(beyond) >= 8
    assert all(r.summary["outcome"] in ("collapse", "protocells") for r in beyond)
    assert sum(r.summary["outcome"] == "collapse" for r in beyond) >= 0.9 * len(beyond)
    more = [sim.rollout(seed) for seed in range(41, 201)]
    beyond = [r for r in more if over_threshold(r)]
    assert len(beyond) >= 30 and sum(r.summary["outcome"] == "collapse" for r in beyond) >= 0.95 * len(beyond)
    assert all(r.summary["outcome"] == "collapse" for r in beyond if r.params["innovation"] == 0)
    below = [r for r in more if not over_threshold(r)]
    assert sum(r.summary["outcome"] != "collapse" for r in below) >= 0.6 * len(below)
    # the same cells with one parameter changed: the fidelity
    kw = dict(gene_types=3, threshold=24, gene_length=160, selfish=0.05, inflow=2000000, innovation=0.0,
              replicators=1000)
    for fidelity, lives in ((0.999, True), (0.995, True), (0.99, False), (0.98, False), (0.95, False)):
        runs = [run(seed, fidelity=fidelity, **kw) for seed in range(1, 6)]
        # at 0.99 the gene (160 bases) is still below L_max (193) and is lost all the same: a ceiling, not the edge
        assert (runs[0].derived["l_max"] > 160) is (lives or fidelity == 0.99)
        assert all(sim.conserved(r).ok for r in runs)
        assert all((r.summary["outcome"] == "protocells") is lives for r in runs), fidelity
        assert all((r.summary["outcome"] == "collapse") is not lives for r in runs), fidelity
        assert all(r.summary["cells"] >= 200 for r in runs) if lives else all(r.steps[30]["cells"] == 0 for r in runs)
    # error load: the share of failed copies is 1 - copy_success while every gene is copied on its own
    r = run(1, fidelity=0.995, **kw)
    assert r.steps[40]["error_load"] == pytest.approx(1 - copy_success(0.995, 160), abs=0.03)


def test_chromosomes_spread_when_the_assortment_load_is_high():
    base = dict(gene_length=40, fidelity=0.999, selfish=0.05, inflow=2000000, innovation=0.01, replicators=1000)
    high = [run(seed, gene_types=6, threshold=36, **base) for seed in range(1, 11)]
    low = [run(seed, gene_types=2, threshold=32, **base) for seed in range(1, 11)]
    assert high[0].derived["full_set"] < 0.95 < 0.999 < low[0].derived["full_set"]
    alive = [r for r in high if r.summary["outcome"] != "collapse"]
    assert len(alive) >= 3
    for r in alive:
        assert r.summary["first_chromosome"] != "never" and r.summary["first_chromosome"] <= 60
        assert r.steps[-1]["linked"] >= 0.95 and r.steps[-1]["viable"] == 1.0
        loose = [s["viable"] for s in r.steps[1:] if s["linked"] < 0.1]
        assert loose and min(loose) < 0.85, "before chromosomes many daughters lack a gene"
    for r in high:
        if r.summary["outcome"] == "collapse":
            assert r.summary["first_chromosome"] == "never", "they died before a chromosome arose"
    for r in low:
        assert r.summary["first_chromosome"] == "never" and max(s["linked"] for s in r.steps) < 0.3
        assert min(s["viable"] for s in r.steps[1:]) > 0.95 and r.summary["cells"] == CAP
    # in between: 4 gene types with 8 copies each
    middle = [run(seed, gene_types=4, threshold=32, **base) for seed in range(1, 5)]
    assert all(r.steps[-1]["linked"] >= 0.95 for r in middle)


def test_complex_cells_come_only_after_dna_and_in_a_minority(sim):
    runs = [sim.rollout(seed) for seed in range(1, 201)]
    reached = 0
    for r in runs:
        dna_first = partner_first = None
        for s in r.steps:
            if s["cells"] >= 20 and dna_first is None and 2 * s["n_dna"] >= s["cells"]:
                dna_first = s["generation"]
            if s["cells"] >= 20 and partner_first is None and 2 * s["n_complex"] >= s["cells"]:
                partner_first = s["generation"]
            if s["stage"] == "complex_cells":
                assert 2 * s["n_dna"] >= s["cells"] and 2 * s["n_working"] >= s["cells"]
                assert s["energy_per_cell"] > 2 * ENERGY, "the partners work"
        if partner_first is not None:
            assert dna_first is not None and dna_first <= partner_first, r.seed
        if any(s["stage"] == "complex_cells" for s in r.steps):
            reached += 1
            assert r.params["innovation"] > 0 and r.params["fidelity"] >= 0.98
            first = next(s for s in r.steps if s["stage"] == "complex_cells")
            assert first["generation"] >= 50 and first["dna"] >= 0.5
    assert 10 <= reached <= 50, "a minority of the seeds"
    final = [r for r in runs if r.summary["outcome"] == "complex_cells"]
    linked = [r for r in runs if r.summary["stage"] == "chromosomes"]
    gain = statistics.mean(r.steps[-1]["gene_types"] - r.params["gene_types"] for r in final)
    assert gain > 1.5 > 1.0 > statistics.mean(r.steps[-1]["gene_types"] - r.params["gene_types"] for r in linked)
    assert all(r.summary["energy_per_cell"] > 2 * ENERGY for r in final)
    assert all(r.summary["energy_per_cell"] == ENERGY for r in runs if r.summary["cells"] and r.summary["complex"] == 0)
    # without dna a partner is no use: at the fidelity of rna it never works
    assert partner_gain(0.999) < 1.001 and partner_gain(1 - 0.001 / 100) > 14


def test_the_selfish_gene_is_held_in_check_by_selection_among_cells(sim):
    """In populations of protocells the fast gene's share stays near the fair share; without
    selection among cells it does not, and no cell keeps a full set."""
    excess = []
    for seed in range(1, 201):
        r = sim.rollout(seed)
        fair = 1 / r.params["gene_types"]
        shares = [s["selfish"] - fair for s in r.steps[1:]
                  if s["cells"] >= 20 and s["viable"] >= 0.5 and s["linked"] < 0.5]
        if shares:
            excess.append(max(shares))
    assert len(excess) >= 100 and max(excess) < 0.35 and statistics.median(excess) < 0.1
    kw = dict(gene_types=3, threshold=24, gene_length=40, fidelity=0.999, selfish=0.2, inflow=2000000,
              innovation=0.0, replicators=1000)
    assert derived(cells._params(kw))["takeover"] == sig(math.log(24) / 0.2)
    for seed in range(1, 5):
        kept = run(seed, **kw)
        free = run(seed, corrector=False, **kw)
        assert max(s["selfish"] for s in kept.steps) < 1 / 3 + 0.2
        assert kept.steps[-1]["viable"] > 0.9 and kept.summary["cells"] == CAP
        assert free.steps[20]["selfish"] > 1 / 3 + 0.2, "100 generations, six times the takeover time"
        assert free.steps[-1]["viable"] == 0.0 and free.steps[-1]["stage"] == "soup"
        assert simulation().conserved(free).ok, "the ledger holds in the control too"


def test_a_twelfth_digit_moves_no_cell(rollouts):
    """No decision of the dynamics sits on a knife's edge: with the fidelity changed in the
    thirteenth digit every cell is where it was (a printed mean may move in its last digit)."""
    for seed in (2, 13, 16, 33):
        r = rollouts[seed]
        for factor in (1 - 1e-13, 1 + 1e-13):
            nudged = run(seed, **dict(r.params, fidelity=r.params["fidelity"] * factor))
            assert nudged.snaps == r.snaps and nudged.steps != r.steps
            whole = [k for k in LEDGER_KEYS if k != "best_growth"] + ["cells", "stage"]
            assert [[s[k] for k in whole] for s in nudged.steps] == [[s[k] for k in whole] for s in r.steps]


# ---------------------------------------------------------------- hand-off

def test_handoff_in_maps_the_summary_of_life():
    below = {"replicators": 1158, "diversity": 12, "outcome": "dominated", "mu": 0.02, "inflow": 240, "monomers": 9000}
    assert handoff_in(below) == {"replicators": 1158, "fidelity": 0.98, "inflow": 240 * INFLOW_SCALE}
    assert INFLOW_SCALE == 12500 and handoff_in({}) == {}
    assert handoff_in({"replicators": 0, "mu": 0.001, "inflow": 0}) == {"replicators": 0, "fidelity": 0.999, "inflow": 0}
    assert handoff_in({"fidelity": 0.995, "mu": 0.5}) == {"fidelity": 0.995}
    assert handoff_in({"mu": 0.9}) == {"fidelity": 0.5} and handoff_in({"mu": 0.0}) == {"fidelity": 0.999999}
    assert handoff_in({"replicators": -3}) == {"replicators": 0}
    for mu in (0.001, 0.002, 0.005, 0.01, 0.02, 0.05):  # the values of level 5 land on this level's list
        assert handoff_in({"mu": mu})["fidelity"] in (0.999, 0.998, 0.995, 0.99, 0.98, 0.95)
    for inflow in (40, 400):
        assert 200000 <= handoff_in({"inflow": inflow})["inflow"] <= 5000000
    # the hand-off feeds a run, and the run hands the summary on
    p = dict(random_params(random.Random(9)), **handoff_in(below))
    r = run(9, **p)
    assert r.params["replicators"] == 1158 and r.params["fidelity"] == 0.98 and r.params["inflow"] == 3000000
    assert simulation().conserved(r).ok and set(HANDOFF_OUT) == set(r.summary)
    assert run(9, **dict(p, **handoff_in({"replicators": 0}))).summary["outcome"] == "collapse"
    assert run(9, **dict(p, **handoff_in({"inflow": 0}))).summary["outcome"] == "collapse"


def test_check_matches_close(sim, rollouts):
    r = rollouts[13]
    for t in (5, 31, 60):
        for key in ("genes_per_cell", "genome_length", "selfish", "viable"):
            truth = r.steps[t][key]
            answer = dense_value(truth)
            assert close(parse_num(answer), truth, rel=0.05)
            assert sim.check(f"cells seed 1 3 step {' '.join(str(t))} {key}", answer).ok


# ---------------------------------------------------------------- review of round 7

#: every lesson rule written again from the module text, without the module's functions
BY_HAND = {
    "copy_success": lambda fidelity, length: fidelity ** length,
    "error_threshold": lambda sigma, fidelity: math.log(sigma) / (1 - fidelity),
    "under_threshold": lambda length, sigma, fidelity: "yes" if length < math.log(sigma) / (1 - fidelity) else "no",
    "full_set": lambda copies, types: (1 - 0.5 ** copies) ** types,
    "expected_waste": lambda fidelity, length: length * (1 - fidelity ** length),
    "nutrient_share": lambda inflow, n: inflow // n,
    "energy_per_gene": lambda energy, genes, partner, gain: energy * (gain if partner == 1 else 1) / genes,
    "generations_to_fix": lambda population, advantage: math.log(population) / advantage,
    "doubling_time": lambda rate: math.log(2) / math.log(1 + rate),
    "balance": lambda a, b, c: 0.0 if min(a, b, c) == 0 else 9 / ((a + b + c) * (1 / a + 1 / b + 1 / c)),
    "efficiency": lambda genes: genes / (genes + 8),
    "expression": lambda energy: energy / (energy + 1),
    "partner_gain": lambda fidelity: 1 + 15 * fidelity ** 10000,
}


def test_lessons_of_four_seeds_recomputed_by_hand_and_not_dominated_by_one_answer():
    assert set(BY_HAND) == set(RULES)
    by_rule = collections.defaultdict(list)
    for seed in (1, 2, 3, 4):
        batch = LESSONS.generate(random.Random(seed), 500)
        assert len(batch) == 500
        for ln in batch:
            assert is_dense(ln.text) and n_tokens(ln.text) <= 128 and "." not in ln.prompt and "." not in ln.answer
            v = LESSONS.check(ln.prompt, ln.answer)
            assert v.ok and v.expected == ln.answer, ln.text
            name, values = LESSONS.parse(ln.prompt)
            mine = BY_HAND[name](*values)
            assert ln.answer == (mine if isinstance(mine, str) else num(mine)), (ln.text, mine)
            by_rule[name].append((values, ln.answer))
    assert set(by_rule) == set(RULES) and min(len(v) for v in by_rule.values()) >= 100
    for name, items in by_rule.items():
        answers = collections.Counter(answer for _, answer in items)
        top = answers.most_common(1)[0][1] / len(items)
        if name == "under_threshold":
            assert set(answers) == {"yes", "no"} and 0.4 <= answers["yes"] / len(items) <= 0.6, answers
        else:
            assert top <= 0.2, (name, answers.most_common(2))
            assert len(answers) >= 40, (name, len(answers))
    # an answer that only repeats an input teaches nothing: a copy that practically always fails
    waste = by_rule["expected_waste"]
    assert sum(answer == num(values[1]) for values, answer in waste) / len(waste) <= 0.12
    tiny = sum(parse_num(answer) < 1e-6 for _, answer in by_rule["copy_success"]) / len(by_rule["copy_success"])
    assert tiny <= 0.02, "copy_success is asked where the threshold lies, not where nothing is ever copied"
    # the inputs cover what the simulation itself computes
    spec = {name: {i.name: (i.low, i.high) for i in rule.inputs} for name, rule in RULES.items()}
    assert spec["copy_success"]["length"] == spec["expected_waste"]["length"] == (1, 640)
    assert spec["full_set"]["types"] == (1, GMAX) and spec["nutrient_share"]["cells"] == (1, CAP)
    for fidelity in (0.95, 0.98, 0.99, 0.995, 0.998, 0.999):  # every dna fidelity of random_params is a lesson input
        low, high = spec["partner_gain"]["fidelity"]
        assert low <= tables(fidelity, 40).fidelity[1] <= high


def test_the_balance_lesson_is_the_simulations_balance_of_a_cell_with_three_gene_types():
    """The simulation calls ``_balance`` on all cells at once; the lesson is its three-type case,
    bit for bit, on the cells of real runs."""
    seen = 0
    for seed in (1, 2, 3):
        r = run(seed, gene_types=3, threshold=24, innovation=0.0)
        for snap in r.snaps[::10]:
            if not snap:
                continue
            genes = np.array([c[5:] for c in snap], dtype=np.int64)
            kernel = cells._balance(genes, np.full(len(snap), 3, dtype=np.int64))
            for row, value in zip(genes.tolist(), kernel.tolist()):
                assert balance(*row) == value == pytest.approx(BY_HAND["balance"](*row), rel=1e-12, abs=1e-15)
                seen += 1
    assert seen > 1000
    # and with more gene types the kernel is the same formula: types^2 / (sum * sum of inverses)
    wide = np.array([[3, 1, 4, 1, 5, 9, 2, 6], [2, 2, 2, 2, 0, 0, 0, 0], [5, 0, 1, 1, 0, 0, 0, 0]], dtype=np.int64)
    got = cells._balance(wide, np.array([8, 4, 4], dtype=np.int64)).tolist()
    assert got[0] == pytest.approx(64 / (31 * sum(1 / x for x in wide[0].tolist()))) and got[1:] == [1.0, 0.0]


def test_no_logarithm_sits_on_a_rounding_edge():
    """``math.log`` may differ in its last bit between machines. For every parameter set
    ``random_params`` can draw, no derived number that takes a logarithm is within a millionth of
    the edge between two printed values, and no piece is within a millionth of its threshold."""
    types_, lengths, fidelities = range(2, 7), (20, 40, 80, 160, 320, 640), (0.95, 0.98, 0.99, 0.995, 0.998, 0.999)
    n = 0
    for types, length, fidelity, selfish, copies in itertools.product(types_, lengths, fidelities,
                                                                      (0.02, 0.05, 0.1, 0.2), (6, 8, 10, 12, 16)):
        p = cells._params(dict(gene_types=types, gene_length=length, fidelity=fidelity, selfish=selfish,
                               threshold=types * copies))
        exact = derived_exact(p)
        assert derived(p) == {k: sig(v) for k, v in exact.items()}
        for key in ("l_max", "doubling", "takeover"):
            x = exact[key]
            assert sig(x * (1 - 1e-6)) == sig(x) == sig(x * (1 + 1e-6)), (p, key, x)
            n += 1
    assert n == 3 * 5 * 6 * 6 * 4 * 5
    for length, fidelity in itertools.product(lengths, fidelities):
        t = tables(fidelity, length)
        for kind in range(8):
            for genes in range(1, GMAX + 1):
                piece = length * (genes if kind >> 2 else 1)
                assert t.l_max[kind, genes] > 0 and abs(piece / t.l_max[kind, genes] - 1) > 1e-6


def test_does_not_own_the_generated_prompts_of_rounds_5_and_6(sim):
    """Real prompts of the frozen levels and gates, not only hand-written ones."""
    prompts = []
    for name in ("nucleo", "planets", "chem", "life"):
        level = importlib.import_module(f"haishool.cosmos.{name}")
        r = level.simulation().run(3, **level.random_params(random.Random(3)))
        prompts += [ln.prompt for ln in level.lines(r) if ln.kind != "record"]
    predict = importlib.import_module("haishool.cosmos.predict")
    prompts += [ln.prompt for ln in predict.generate(random.Random(1), 200)]
    for name in ("maths", "elements"):
        gate = importlib.import_module(f"haishool.truth.{name}").gate()
        prompts += [ln.prompt for ln in gate.generate(random.Random(1), 200) if ln.kind != "record"]
    assert len(prompts) > 3000
    for prompt in prompts:
        assert not sim.owns(prompt) and not LESSONS.owns(prompt), prompt
        assert sim.check(prompt, "1").reason == "not my question" == LESSONS.check(prompt, "1").reason
    # and the two gates of this level do not own each other's prompts
    for ln in LESSONS.generate(random.Random(2), 200):
        assert LESSONS.owns(ln.prompt) and not sim.owns(ln.prompt)
    for ln in lines(sim.rollout(7)):
        if ln.kind != "record":
            assert sim.owns(ln.prompt) and not LESSONS.owns(ln.prompt)


def test_conserved_and_lines_hold_far_outside_the_canonical_parameters(sim):
    """The hand-off may bring any fidelity, inflow and number of replicators: the ledger stays
    exact and the lines dense and short."""
    rng = random.Random(77)
    ran = 0
    for seed in range(60):
        types = rng.randint(1, 16)
        p = dict(gene_types=types, gene_length=rng.choice([1, 3, 20, 100, 700, 2000]),
                 fidelity=rng.choice([0.5, 0.9, 0.97, 0.999, 0.9999, 0.999999]), selfish=rng.choice([0, 0.3, 2, 10]),
                 inflow=rng.choice([0, 1, 399, 12500, 10 ** 6, 10 ** 8, 10 ** 10]),
                 threshold=rng.choice([max(2, types), 2 * types, 5 * types, 40 * types, 1000]),
                 innovation=rng.choice([0, 0.001, 0.05, 0.3, 1.0]), replicators=rng.choice([0, 1, 7, 100, 5000, 10 ** 6]))
        try:
            r = run(seed, **p)
        except ValueError:  # a fidelity that allows no copy at all
            continue
        ran += 1
        v = sim.conserved(r)
        assert v.ok, (p, v)
        assert r.steps == run(seed, **p).steps
        for ln in lines(r):
            assert is_dense(ln.text) and n_tokens(ln.text) <= 128, ln.text
        assert all(c[0] <= GMAX for snap in r.snaps for c in snap)
    assert ran >= 40



# ---------------------------------------------------------------- the numbers the module text names

def test_forty_seeds_of_random_parameters_from_another_stream(sim):
    """40 seeds whose parameters are not the canonical ones: conserved, replayed exactly (steps,
    cells and summary), each run under 2 seconds."""
    rng = random.Random(2026)
    for _ in range(40):
        seed = rng.randint(1, 10 ** 6)
        p = random_params(random.Random(7 * seed + 1))
        start = time.perf_counter()
        r = run(seed, **p)
        assert time.perf_counter() - start < 2.0
        again = run(seed, **p)
        assert r.steps == again.steps and r.snaps == again.snaps and r.summary == again.summary
        assert sim.conserved(r).ok, (seed, p)


def test_the_regimes_named_in_the_module_text():
    """The counts the module text gives for its forced regimes, with the text's seeds."""
    kw = dict(gene_types=3, threshold=24, gene_length=160, selfish=0.05, inflow=2000000, innovation=0.0,
              replicators=1000)
    for fidelity, outcome in ((0.999, "protocells"), (0.995, "protocells"), (0.99, "collapse"),
                              (0.98, "collapse"), (0.95, "collapse")):
        assert [run(seed, fidelity=fidelity, **kw).summary["outcome"] for seed in range(1, 9)] == [outcome] * 8
    base = dict(gene_length=40, fidelity=0.999, selfish=0.05, inflow=2000000, innovation=0.01, replicators=1000)
    high = [run(seed, gene_types=6, threshold=36, **base) for seed in range(1, 11)]
    alive = [r for r in high if r.summary["outcome"] != "collapse"]
    assert 3 <= len(alive) <= 7
    assert all(10 <= r.summary["first_chromosome"] <= 40 for r in alive)
    viable = [min(s["viable"] for s in r.steps[1:] if s["linked"] < 0.1) for r in alive]
    assert 0.5 <= min(viable) and max(viable) <= 0.8, viable
    low = [run(seed, gene_types=2, threshold=32, **base) for seed in range(1, 11)]
    assert max(max(s["linked"] for s in r.steps) for r in low) < 0.25
    control = dict(gene_types=3, threshold=24, gene_length=40, fidelity=0.999, selfish=0.2, inflow=2000000,
                   innovation=0.0, replicators=1000)
    for seed in range(1, 7):
        free = run(seed, corrector=False, **control)
        assert free.steps[20]["selfish"] > 0.55 and free.steps[-1]["viable"] == 0.0


def test_the_canonical_numbers_named_in_the_module_text(sim):
    """Seeds 1 to 200 (the module text gives 1 to 400): complex cells reach their stage between
    generation 100 and 300 and never before dna; complex genomes gain more gene types than
    linked ones; the selfish gene stays within 0.29 of its fair share in protocells."""
    firsts, gain, excess = [], collections.defaultdict(list), []
    for seed in range(1, 201):
        r = sim.rollout(seed)
        firsts += [s["generation"] for s in r.steps if s["stage"] == "complex_cells"][:1]
        gain[r.summary["stage"]].append(r.steps[-1]["gene_types"] - r.params["gene_types"])
        fair = 1 / r.params["gene_types"]
        shares = [s["selfish"] - fair for s in r.steps[1:]
                  if s["cells"] >= 20 and s["viable"] >= 0.5 and s["linked"] < 0.5]
        excess += [max(shares)] if shares else []
    assert 100 <= min(firsts) and max(firsts) <= 300 and 150 <= statistics.median(firsts) <= 230
    assert statistics.mean(gain["complex_cells"]) > 2.0 and statistics.mean(gain["chromosomes"]) < 1.2
    assert max(excess) <= 0.29 and statistics.median(excess) <= 0.08
