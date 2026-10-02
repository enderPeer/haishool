import collections
import copy
import doctest
import importlib
import math
import random
import statistics
import time
import types

import pytest

from haishool.cosmos import Simulation, close
from haishool.evo import LessonGate, sig
from haishool.evo import society
from haishool.evo.society import (COUNT_KEYS, EDGE_RULES, FARMING, GAME_SIZE, GENERATIONS, HIDDEN_PARAM_KEYS,
                                  INPUT_KEYS, KEYS, LADDERS, LEDGER_KEYS, LESSONS, MIN_GROUP, PARAM_KEYS,
                                  ROUNDED_KEYS, RULES, SIM, STAGES,
                                  STATE_KEYS, STEP_YEARS, STEPS, SUMMARY_KEYS, TECH_NAMES, TECHS, TERRITORIES,
                                  WORD_KEYS, WRITING, Society, _World, capacity, critical_population, dense_value,
                                  doubling_years, effective_alpha, gini, hamilton, handoff_in, hierarchy_levels,
                                  kin_relatedness, lesson_gate, lines, multiplier_of, param_value, params_line,
                                  payoff_cooperator, payoff_defector, random_params, replicator_step, role_count,
                                  simulation, skill_change, stage_of, stake_of, teaching_fidelity, technologies,
                                  tremble, wealth_fifths)
from haishool.truth import Gate, is_dense, num, parse_num, split_line

SEEDS = list(range(1, 41))


def n_tokens(text: str) -> int:
    """As ``haishool.student.tokens`` counts them (that module needs torch, so not imported)."""
    return len(text.replace(".", " . ").split())


@pytest.fixture(scope="module")
def sim():
    return Society()


@pytest.fixture(scope="module")
def rollouts(sim):
    return {seed: sim.rollout(seed) for seed in SEEDS}


@pytest.fixture(scope="module")
def all_lines(rollouts):
    return [ln for seed in SEEDS[:12] for ln in lines(rollouts[seed])]


# ---------------------------------------------------------------------------------------------
# the contract
# ---------------------------------------------------------------------------------------------

def test_protocols():
    s = simulation()
    assert isinstance(s, Simulation) and isinstance(s, Gate)
    assert SIM == s.sim == s.topic == "society" and s.records() == [] and simulation() is s
    for key in PARAM_KEYS + HIDDEN_PARAM_KEYS + STATE_KEYS + LEDGER_KEYS + SUMMARY_KEYS:
        assert key in KEYS, key
    assert COUNT_KEYS | WORD_KEYS | ROUNDED_KEYS <= set(KEYS) and INPUT_KEYS <= set(PARAM_KEYS)
    assert STATE_KEYS == ("year", "population", "groups", "largest_group", "cooperation", "skill", "technologies",
                          "farming", "writing", "hierarchy", "roles", "gini", "trade", "conflicts", "stage")
    assert STAGES == ("bands", "tribes", "villages", "chiefdoms", "states", "collapse")
    assert isinstance(LESSONS, LessonGate) and lesson_gate() is LESSONS and LESSONS.topic == "predict_society"
    assert len(RULES) >= 10 and LESSONS.rules is RULES
    for name in ("run", "rollout", "conserved", "check", "owns", "lines", "random_params", "handoff_in"):
        assert callable(getattr(society, name)), name
    assert society.run(5).steps == s.run(5).steps and society.conserved(society.rollout(5)).ok


def test_docstring_examples_are_true():
    assert doctest.testmod(society).failed == 0
    assert doctest.testmod(society).attempted >= 20


def test_same_seed_same_lines():
    a, b = Society(), Society()
    for seed in (1, 7, 4321):
        ra, rb = a.rollout(seed), b.rollout(seed)
        assert ra.steps == rb.steps and ra.summary == rb.summary and ra.params == rb.params
        assert ra.groups == rb.groups and ra.events == rb.events
        assert [repr(v) for s in ra.steps for v in s.values()] == [repr(v) for s in rb.steps for v in s.values()]
        assert [ln.text for ln in lines(ra)] == [ln.text for ln in lines(rb)]
    assert [ln.text for ln in lines(a.rollout(1))] != [ln.text for ln in lines(a.rollout(2))]
    assert a.run(5, land=2000.0).steps == b.run(5, land=2000.0).steps
    assert a.run(5, land=2000.0).steps != a.run(6, land=2000.0).steps
    assert a.run(5, land=2000.0).steps != a.run(5, land=2010.0).steps


def test_forty_seeds_replay_exactly(rollouts):
    fresh = Society()
    for seed, r in rollouts.items():
        again = fresh.run(seed, **random_params(random.Random(seed)))
        assert again.steps == r.steps and again.groups == r.groups and again.events == r.events
        assert again.summary == r.summary and again.params == r.params


def test_values_are_plain_python_numbers_and_words(rollouts):
    for r in rollouts.values():
        assert list(r.params) == list(PARAM_KEYS + HIDDEN_PARAM_KEYS) and list(r.summary) == list(SUMMARY_KEYS)
        assert len(r.steps) == STEPS + 1 == 81 == len(r.groups)
        for row in r.steps + [r.summary, r.params]:
            for key, value in row.items():
                assert type(value) in (float, int, str), (key, type(value))
                if type(value) is float:
                    assert math.isfinite(value), (key, value)
                if key in WORD_KEYS:
                    assert type(value) is str, key
                if key in ROUNDED_KEYS:
                    assert type(value) is int, key
        for s in r.steps:
            assert list(s) == list(STATE_KEYS + LEDGER_KEYS)
            assert all(type(s[k]) is int for k in COUNT_KEYS if k in s)
        for groups in r.groups:
            for g in groups:
                assert all(type(v) in (float, int, str) for v in g.values())


def test_random_params_stay_in_range():
    seen = collections.Counter()
    for seed in range(400):
        p = random_params(random.Random(seed))
        assert p == random_params(random.Random(seed)) == random_params(random.Random(seed), rules=7)
        assert set(p) == INPUT_KEYS
        assert 1000 <= p["land"] <= 5000 and p["land"] % 10 == 0
        assert 0.05 <= p["relatedness"] <= 0.5 and 1.5 <= p["benefit"] <= 6 and p["cost"] == 1.0
        assert p["punishment"] == 0 or 0.1 <= p["punishment"] <= 1.5
        assert 2.4 <= p["alpha"] <= 3.6 and 0.9 <= p["beta"] <= 1.1 and 0.1 <= p["connectedness"] <= 1
        assert 0 <= p["success"] <= 0.2 or 0.5 <= p["success"] <= 1
        assert type(p["words"]) is int and 2 <= p["words"] <= 40 and 0 <= p["compositional"] <= 1
        assert type(p["isolation"]) is int and (p["isolation"] == 0 or 7500 <= p["isolation"] <= 15000)
        assert p["isolation"] % STEP_YEARS == 0 and 0.15 <= p["growth"] <= 0.35
        seen[("punished", p["punishment"] > 0)] += 1
        seen[("language", p["success"] >= 0.5)] += 1
        seen[("rule", "none" if p["compositional"] == 0 else "all" if p["compositional"] == 1 else "some")] += 1
        seen[("isolated", p["isolation"] > 0)] += 1
    assert 80 < seen[("punished", False)] < 160 and 15 < seen[("language", False)] < 70
    assert 100 < seen[("rule", "none")] < 180 and 140 < seen[("rule", "all")] < 220
    assert 30 < seen[("isolated", True)] < 100
    assert 50 < seen[("rule", "some")] < 110
    assert random_params(random.Random(3)) == {
        "land": 1950.0, "relatedness": 0.29, "benefit": 3.2, "cost": 1.0, "punishment": 0.98, "alpha": 2.48,
        "beta": 0.9, "connectedness": 0.85, "success": 0.62, "words": 40, "compositional": 1.0, "isolation": 0,
        "growth": 0.25}
    with pytest.raises(ValueError):
        random_params(random.Random(1), rules=6)


def test_run_rejects_impossible_parameters(sim):
    for bad in ({"land": 0.0}, {"land": 999.0}, {"land": float("nan")}, {"relatedness": -0.1}, {"relatedness": 1.5},
                {"benefit": 0.0}, {"cost": 0.0}, {"cost": -1.0}, {"punishment": -0.1}, {"alpha": 0.0},
                {"beta": 0.0}, {"beta": float("inf")}, {"connectedness": 1.2}, {"success": -0.2}, {"success": 1.01},
                {"words": -1}, {"words": 2.5}, {"compositional": "yes"}, {"compositional": 1.5}, {"isolation": -250},
                {"compositional": -0.1}, {"success": "high"}, {"land": True},
                {"isolation": 2.5}, {"growth": 0.0}, {"growth": 1.5}, {"rules": 6}):
        with pytest.raises(ValueError):
            sim.run(1, **bad)
    with pytest.raises(TypeError):
        sim.run(1, clumps=3)
    assert society.LAND_MIN == TERRITORIES * 2 * MIN_GROUP / society.HARVEST_FLOOR == 1000
    edge = sim.run(1, land=1000.0, relatedness=0.0, punishment=0.0, connectedness=0.0, success=0.0, words=0,
                   compositional=0.0, growth=1.0)
    assert sim.conserved(edge).ok and edge.summary["technologies"] == 0 and edge.summary["technology_list"] == "none"


# ---------------------------------------------------------------------------------------------
# the lines
# ---------------------------------------------------------------------------------------------

def test_every_line_is_dense_and_short(all_lines):
    assert len(all_lines) == 12 * 2512
    for ln in all_lines:
        assert is_dense(ln.text), ln.text
        assert n_tokens(ln.text) <= 128, ln.text
        assert ln.topic == "society"
        if ln.kind != "record":
            assert "." not in ln.prompt and "." not in ln.answer
            assert split_line(ln.text) == (ln.prompt, ln.answer)
            assert ln.kind in ("fact", "calc")
    assert max(n_tokens(ln.text) for ln in all_lines) <= 105


def test_lines_of_a_long_seed_stay_short(sim):
    for seed in (9998, 123456789):
        r = sim.rollout(seed)
        assert sim.conserved(r).ok
        for ln in lines(r):
            assert is_dense(ln.text) and n_tokens(ln.text) <= 128, ln.text
            if ln.kind != "record":
                assert sim.check(ln.prompt, ln.answer).ok, ln.text
    # the longest numbers a run can show: a large land, a late isolation
    big = sim.run(123456789, land=200000.0, isolation=19750, alpha=2.41, beta=0.97, relatedness=0.33, benefit=5.5,
                  punishment=1.25, connectedness=0.95, success=0.95, words=40, growth=0.33)
    assert sim.conserved(big).ok and big.summary["population"] > 1000000
    assert max(n_tokens(ln.text) for ln in lines(big)) <= 128
    assert all(is_dense(ln.text) for ln in lines(big))


def test_pinned_lines_of_seed_3(sim):
    """The lines the module docstring shows, word for word."""
    r = sim.rollout(3)
    texts = [ln.text for ln in lines(r)]
    assert texts[0] == (
        "society seed 3 params. land 1 9 5 0. relatedness 0 point 2 9. benefit 3 point 2. cost 1. "
        "punishment 0 point 9 8. alpha 2 point 4 8. beta 0 point 9. connectedness 0 point 8 5. success 0 point 6 2. "
        "words 4 0. compositional 1. isolation 0. growth 0 point 2 5. hamilton no. fidelity 0 point 5 7 2. "
        "critical_population 6 9 point 4. doubling_years 7 7 point 7.")
    assert texts[1] == (
        "society seed 3 step 0. year 0. population 7 2. groups 4. largest_group 2 3. cooperation 0 point 5 4 7. "
        "skill 0. technologies 0. farming no. writing no. hierarchy 0. roles 1. gini 0. trade 0. conflicts 0. "
        "stage bands.")
    assert ("society seed 3 step 4 0. year 1 0 0 0 0. population 5 2 0 0 0. groups 3 1. largest_group 3 8 9 0. "
            "cooperation 0 point 9 8 2. skill 5 6 3. technologies 6. farming yes. writing no. hierarchy 2. "
            "roles 1 0. gini 0 point 3 1 2. trade 1. conflicts 3 6. stage villages.") in texts
    for text in ("q society seed 3 step 4 0 largest_group. a 3 8 9 0.",
                 "q society seed 3 step 4 0 next skill. a 5 8 0.",
                 "q society seed 3 step 4 0 next groups. a 3 0.",
                 "q society seed 3 final technology_list. a fire stone_tools clothing boats pottery farming metal.",
                 "q society seed 3 final stage. a chiefdoms.", "q society seed 3 final outcome. a chiefdoms.",
                 "q society seed 3 final population. a 6 5 7 0 0.", "q society seed 3 final first_farming. a 8 5 0 0.",
                 "q society seed 3 final first_writing. a never.", "q society seed 3 final gini. a 0 point 3 8 9."):
        assert text in texts, text
    assert texts.index("q society seed 3 final stage. a chiefdoms.") == len(texts) - len(SUMMARY_KEYS)


def test_line_kinds(rollouts):
    r = rollouts[3]
    ls = lines(r)
    records = [ln for ln in ls if ln.kind == "record"]
    assert records[0].text == params_line(r).text and len(records) == 1 + 81
    prompts = [ln.prompt for ln in ls if ln.kind != "record"]
    assert len(prompts) == len(set(prompts)) == 81 * 15 + 80 * 15 + len(SUMMARY_KEYS)
    assert len(ls) == 2512 == len({ln.text for ln in ls})
    assert "society seed 3 step 1 4 stage" in prompts and "society seed 3 step 7 9 next stage" in prompts
    assert "society seed 3 step 8 0 stage" in prompts and "society seed 3 step 8 0 next stage" not in prompts
    for ln in ls:  # the ledger is not trained
        assert not any(f" {k} " in f" {ln.prompt} " for k in LEDGER_KEYS), ln.text
    assert len(lines(r, every=5)) < len(ls) / 4
    kinds = collections.Counter(ln.kind for ln in ls)
    assert kinds == {"record": 82, "fact": 81 * 15, "calc": 80 * 15 + 15}


def test_numbers_use_three_significant_digits(all_lines):
    assert dense_value("population", 12345) == "1 2 3 0 0" and dense_value("largest_group", 149) == "1 4 9"
    assert dense_value("population", 1234567) == "1 point 2 3 e 6" and dense_value("groups", 40) == "4 0"
    assert dense_value("cooperation", 0.98765) == "0 point 9 8 8" and dense_value("skill", 0.0) == "0"
    assert dense_value("gini", 0.000412) == "4 point 1 2 e minus 4" and dense_value("stage", "bands") == "bands"
    assert dense_value("year", 20000) == "2 0 0 0 0" and dense_value("first_farming", "never") == "never"
    assert param_value("land", 2350.0) == "2 3 5 0" and param_value("words", 12) == "1 2"
    assert param_value("isolation", 12750) == "1 2 7 5 0" and param_value("critical_population", 69.44) == "6 9 point 4"
    assert param_value("compositional", 0.95) == "0 point 9 5" and param_value("alpha", 2.48) == "2 point 4 8"
    assert param_value("compositional", 1.0) == "1" and param_value("hamilton", "yes") == "yes"
    for ln in all_lines:
        if ln.kind == "record":
            continue
        key = ln.prompt.split()[-1]
        if key in WORD_KEYS or ln.answer == "never":
            continue
        value = parse_num(ln.answer)
        assert value is not None and value >= 0, ln.text
        if key not in COUNT_KEYS:
            digits = "".join(w for w in ln.answer.split(" e ")[0].split() if w.isdigit())
            assert len(digits.strip("0")) <= 3, ln.text
        assert len(ln.answer.split()) <= 8, ln.text


# ---------------------------------------------------------------------------------------------
# the gate of the rollouts
# ---------------------------------------------------------------------------------------------

def test_conserved_passes_on_forty_seeds_with_random_params(sim, rollouts):
    for seed, r in rollouts.items():
        assert r.params["land"] == random_params(random.Random(seed))["land"]
        v = sim.conserved(r)
        assert v.ok, (seed, v)
    for seed in range(1000, 1010):  # other parameters than the seed's own
        r = sim.run(seed, **random_params(random.Random(seed + 7)))
        assert sim.conserved(r).ok, seed


def test_people_are_accounted_for_exactly(rollouts):
    for r in rollouts.values():
        for t, (s, groups) in enumerate(zip(r.steps, r.groups)):
            alive = [g for g in groups if g["gone"] == "no"]
            assert s["population"] == sum(g["n"] for g in alive) and s["groups"] == len(alive) <= TERRITORIES
            assert s["largest_group"] == max(g["n"] for g in alive)
            assert all(g["n"] >= MIN_GROUP for g in alive)
            for g in groups:
                assert g["n"] == g["n_prev"] + g["births"] - g["deaths"] + g["moved_in"] - g["moved_out"]
            assert sum(g["moved_in"] for g in groups) == sum(g["moved_out"] for g in groups) == s["moved"]
            if t:
                assert s["population"] == r.steps[t - 1]["population"] + s["births"] - s["deaths"]
                assert s["births"] > 0 and s["deaths"] > 0
            assert sum(g["territories"] for g in alive) + s["free_land"] == TERRITORIES
        assert r.steps[0]["births"] == r.steps[0]["deaths"] == r.steps[0]["moved"] == 0
        assert 3 <= r.steps[0]["groups"] <= 8


def test_food_is_accounted_for(rollouts):
    for r in rollouts.values():
        store = 0.0
        for t, s in enumerate(r.steps):
            assert s["produced"] == pytest.approx(s["eaten"] + s["stored"] + s["lost"], rel=1e-9, abs=1e-6)
            assert s["produced"] >= 0 and s["eaten"] >= 0 and s["lost"] >= -1e-6 and s["store"] >= 0
            if t:
                assert s["store"] == pytest.approx(store + s["stored"], rel=1e-9, abs=1e-6)
                # nobody eats more than one ration per generation
                assert s["eaten"] <= GENERATIONS * max(r.steps[t - 1]["population"], s["population"]) * 3
                assert s["eaten"] == int(s["eaten"])
            store = s["store"]
            assert s["store"] == pytest.approx(sum(g["store"] for g in r.groups[t] if g["gone"] == "no"))
            assert s["climate"] == 1.0 or 0.6 <= s["climate"] <= 0.9
        dry = sum(s["climate"] < 1 for s in r.steps)
        assert dry <= 25
    assert sum(s["climate"] < 1 for r in rollouts.values() for s in r.steps) > 150, "about one step in ten is dry"


def test_payoffs_are_those_of_the_game(rollouts):
    """Recomputed from the group's own record; and without punishment the difference is Hamilton's."""
    for r in rollouts.values():
        p = r.params
        assert p["stake"] == stake_of(p["benefit"], p["cost"], GAME_SIZE)
        assert p["multiplier"] == multiplier_of(p["benefit"], p["cost"], GAME_SIZE)
        for groups in r.groups[1::8]:
            for g in groups:
                if g["gone"] == "yes":
                    continue
                kin, x = kin_relatedness(p["relatedness"], g["n_before"]), g["share_before"]
                assert kin == g["kin"] and 0 <= kin <= p["relatedness"]
                pay_c = payoff_cooperator(GAME_SIZE, 1 + 4 * (kin + (1 - kin) * x), p["multiplier"], p["stake"],
                                          p["punishment"])
                pay_d = payoff_defector(GAME_SIZE, 4 * (1 - kin) * x, p["multiplier"], p["stake"], p["punishment"])
                assert pay_c == g["payoff_c"] and pay_d == g["payoff_d"]
                assert g["cooperation"] == tremble(replicator_step(x, pay_c, pay_d, 0.2))
                closed = (kin * p["benefit"] - p["cost"]
                          + 4 * p["punishment"] * (1 - kin) * (x - (1 - x) / 3))
                assert pay_c - pay_d == pytest.approx(closed, abs=1e-9)
                if p["punishment"] == 0:
                    assert (pay_c > pay_d) == (hamilton(kin, p["benefit"], p["cost"]) == "yes") \
                        or abs(pay_c - pay_d) < 1e-9
                assert 0.01 - 1e-12 <= g["cooperation"] <= 0.99 + 1e-12


def test_technologies_keep_their_order_and_hierarchy_fits_the_size(rollouts):
    assert TECH_NAMES[:8] == ("fire", "stone_tools", "clothing", "boats", "pottery", "farming", "metal", "writing")
    assert [t for _, t, _ in TECHS] == sorted(t for _, t, _ in TECHS) and len(set(TECH_NAMES)) == len(TECHS) == 10
    for r in rollouts.values():
        for s, groups in zip(r.steps, r.groups):
            alive = [g for g in groups if g["gone"] == "no"]
            for g in alive:
                assert g["technologies"] == technologies(g["skill"], hierarchy_levels(g["n"]))
                if g["technologies"] >= WRITING:
                    assert hierarchy_levels(g["n"]) >= 3 and g["n"] >= 5400
            assert s["technologies"] == max(g["technologies"] for g in alive)
            assert s["farming"] == ("yes" if s["technologies"] >= FARMING else "no")
            assert s["writing"] == ("yes" if s["technologies"] >= WRITING else "no")
            assert s["hierarchy"] == hierarchy_levels(s["largest_group"])
            assert s["stage"] in ("collapse", stage_of(s["largest_group"]))
            top = max(alive, key=lambda g: g["n"])
            assert s["roles"] == role_count(int(top["store"]))
            assert s["gini"] == gini(*wealth_fifths(top["n"], top["store"], s["hierarchy"]))
            assert 0 <= s["gini"] <= 0.8 and 0 <= s["trade"] <= 1 and 0 <= s["cooperation"] <= 1
            if s["hierarchy"] == 0:
                assert s["gini"] == 0, "a band shares equally"
        names = r.summary["technology_list"].split() if r.summary["technology_list"] != "none" else []
        assert tuple(names) == TECH_NAMES[:r.summary["technologies"]]


def test_summary_matches_the_run(rollouts):
    for r in rollouts.values():
        last = r.steps[-1]
        for k in ("stage", "population", "groups", "cooperation", "technologies", "farming", "writing", "hierarchy",
                  "roles", "gini", "conflicts"):
            assert r.summary[k] == last[k], k
        peak = max(s["population"] for s in r.steps)
        low = last["stage"] == "collapse" or last["population"] < 0.6 * peak
        assert r.summary["outcome"] == ("collapse" if low else last["stage"]) and r.summary["outcome"] in STAGES
        for key, flag in (("first_farming", "farming"), ("first_writing", "writing")):
            years = [s["year"] for s in r.steps if s[flag] == "yes"]
            assert r.summary[key] == (years[0] if years else "never")
        assert last["year"] == 20000 and [s["year"] for s in r.steps] == [250 * t for t in range(81)]
        p = r.params
        fid = teaching_fidelity(p["success"], p["words"], p["compositional"], 0)
        assert 0.2 <= fid <= 0.8
        assert p["fidelity"] == fid and p["hamilton"] == hamilton(p["relatedness"], p["benefit"], p["cost"])
        assert p["critical_population"] == critical_population(effective_alpha(p["alpha"], 0, fid), p["beta"])
        assert p["doubling_years"] == pytest.approx(doubling_years(math.log(1 + p["growth"]) / 25), rel=1e-12)
        assert (1 + p["growth"]) ** (p["doubling_years"] / 25) == pytest.approx(2)


def test_the_log_fits_the_steps(rollouts):
    kinds = collections.Counter()
    for r in rollouts.values():
        assert [e["step"] for e in r.events] == sorted(e["step"] for e in r.events)
        for e in r.events:
            assert set(e) == {"step", "kind", "a", "b", "moved", "killed_a", "killed_b"}
            assert e["kind"] in ("bud", "raid", "conquest", "merge", "split", "extinct") and 1 <= e["step"] <= STEPS
            kinds[e["kind"]] += 1
            ids = {g["id"] for g in r.groups[e["step"]]}
            assert e["a"] in ids and (e["b"] in ids or e["kind"] == "extinct")
            if e["kind"] in ("conquest", "merge"):
                loser = next(g for g in r.groups[e["step"]] if g["id"] == e["b"])
                assert loser["gone"] == "yes" and loser["moved_out"] >= e["moved"]
        for t, s in enumerate(r.steps):
            assert s["conflicts"] == sum(e["kind"] in ("raid", "conquest") and e["step"] <= t for e in r.events)
    assert all(kinds[k] > 0 for k in ("bud", "raid", "conquest", "merge", "split"))


def test_conserved_catches_tampering(sim, rollouts):
    base = rollouts[3]
    assert base.summary["conflicts"] > 0 and base.summary["farming"] == "yes"

    def broken(change) -> bool:
        r = copy.deepcopy(base)
        change(r)
        return not sim.conserved(r).ok

    def alive(r, t, i=0):
        return [g for g in r.groups[t] if g["gone"] == "no"][i]

    assert not broken(lambda r: None)
    # people
    assert broken(lambda r: r.steps[10].update(population=r.steps[10]["population"] + 1))
    assert broken(lambda r: r.steps[10].update(births=r.steps[10]["births"] + 1))
    assert broken(lambda r: r.steps[10].update(deaths=r.steps[10]["deaths"] - 1))
    assert broken(lambda r: r.steps[10].update(moved=r.steps[10]["moved"] + 1))
    assert broken(lambda r: r.steps[10].update(groups=r.steps[10]["groups"] + 1))
    assert broken(lambda r: r.steps[10].update(largest_group=r.steps[10]["largest_group"] + 1))
    assert broken(lambda r: alive(r, 10).update(n=alive(r, 10)["n"] + 1))
    assert broken(lambda r: alive(r, 10).update(births=alive(r, 10)["births"] + 1, deaths=alive(r, 10)["deaths"] + 1))
    assert broken(lambda r: alive(r, 10).update(moved_in=alive(r, 10)["moved_in"] + 5,
                                                births=alive(r, 10)["births"] - 5))
    assert broken(lambda r: alive(r, 10).update(n_prev=alive(r, 10)["n_prev"] + 1, births=alive(r, 10)["births"] - 1))
    assert broken(lambda r: r.groups[10].pop(0))

    def shift(r):  # one group richer in people, another poorer: the totals still add up
        a, b = alive(r, 10, 0), alive(r, 10, 1)
        a["n"] += 1
        b["n"] -= 1
    assert broken(shift)
    # food and land
    assert broken(lambda r: r.steps[10].update(produced=r.steps[10]["produced"] * 1.001))
    assert broken(lambda r: r.steps[10].update(eaten=r.steps[10]["eaten"] - 1))
    assert broken(lambda r: r.steps[10].update(lost=-5.0, stored=r.steps[10]["stored"] + r.steps[10]["lost"] + 5.0))
    assert broken(lambda r: r.steps[10].update(store=r.steps[10]["store"] + 1.0))
    assert broken(lambda r: alive(r, 10).update(store=alive(r, 10)["store"] + 1.0))
    assert broken(lambda r: r.steps[10].update(free_land=r.steps[10]["free_land"] + 1))
    assert broken(lambda r: alive(r, 10).update(territories=alive(r, 10)["territories"] + 1))
    assert broken(lambda r: r.steps[10].update(climate=0.3))
    # payoffs and shares
    assert broken(lambda r: alive(r, 10).update(payoff_c=alive(r, 10)["payoff_c"] + 0.01))
    assert broken(lambda r: alive(r, 10).update(payoff_d=alive(r, 10)["payoff_d"] - 0.01))
    assert broken(lambda r: alive(r, 10).update(kin=alive(r, 10)["kin"] + 0.01))
    assert broken(lambda r: alive(r, 10).update(cooperation=alive(r, 10)["cooperation"] * 0.9))
    assert broken(lambda r: alive(r, 10).update(share_before=alive(r, 10)["share_before"] * 0.9))
    assert broken(lambda r: r.steps[10].update(cooperation=1.2))
    assert broken(lambda r: r.steps[10].update(trade=-0.1))
    assert broken(lambda r: r.steps[10].update(gini=r.steps[10]["gini"] + 0.01))
    assert broken(lambda r: r.params.update(punishment=r.params["punishment"] + 0.1))
    assert broken(lambda r: r.params.update(relatedness=r.params["relatedness"] + 0.1))
    # technologies, hierarchy, stage, conflicts, year
    assert broken(lambda r: alive(r, 40).update(technologies=alive(r, 40)["technologies"] + 1))
    assert broken(lambda r: alive(r, 40).update(skill=10.0))
    assert broken(lambda r: r.steps[40].update(technologies=r.steps[40]["technologies"] + 1))
    assert broken(lambda r: r.steps[40].update(farming="no"))
    assert broken(lambda r: r.steps[40].update(writing="yes"))
    assert broken(lambda r: r.steps[40].update(hierarchy=r.steps[40]["hierarchy"] + 1))
    assert broken(lambda r: r.steps[40].update(roles=r.steps[40]["roles"] + 1))
    assert broken(lambda r: r.steps[40].update(stage="states"))
    assert broken(lambda r: r.steps[40].update(stage="finished"))
    assert broken(lambda r: r.steps[40].update(conflicts=r.steps[40]["conflicts"] + 1))
    assert broken(lambda r: r.events.clear())
    assert broken(lambda r: r.steps[40].update(year=10001))
    assert broken(lambda r: r.summary.update(technology_list="fire boats"))
    assert broken(lambda r: r.summary.update(technology_list="stone_tools fire clothing boats pottery farming metal"))
    assert broken(lambda r: r.groups.pop())
    assert broken(lambda r: setattr(r, "groups", []))


def test_gate_agrees_with_every_line(sim, all_lines):
    for ln in all_lines:
        if ln.kind == "record":
            continue
        assert sim.owns(ln.prompt), ln.prompt
        v = sim.check(ln.prompt, ln.answer)
        assert v.ok and v.expected == ln.answer, (ln.text, v)
        assert not LESSONS.owns(ln.prompt)


def test_gate_agrees_with_every_record_field(sim, all_lines):
    n = 0
    for ln in all_lines:
        if ln.kind != "record":
            continue
        head, *fields = [f.strip() for f in ln.text.rstrip(".").split(".")]
        for f in fields:
            key, _, value = f.partition(" ")
            v = sim.check(f"{head} {key}", value)
            assert v.ok and v.expected == value, (ln.text, key)
            n += 1
    assert n == 12 * (81 * 15 + len(PARAM_KEYS))


def test_gate_rejects_altered_answers(sim, all_lines):
    rng = random.Random(5)
    questions = [ln for ln in all_lines if ln.kind != "record"]
    words = {"farming": ("yes", "no"), "writing": ("yes", "no"), "stage": STAGES, "outcome": STAGES}
    for ln in rng.sample(questions, 4000):
        key = ln.prompt.split()[-1]
        if key in words:
            wrong = next(w for w in words[key] if w != ln.answer)
        elif key == "technology_list":
            wrong = "fire" if ln.answer != "fire" else "none"
        elif ln.answer == "never":
            wrong = "5 0 0 0"
        elif ln.answer == "0":
            wrong = "1"
        else:
            wrong = "9 " + ln.answer  # a changed leading digit: far outside the 5 % tolerance
        v = sim.check(ln.prompt, wrong)
        assert not v.ok and v.expected == ln.answer, (ln.text, wrong, v)
    assert not sim.check("society seed 3 final first_farming", "never").ok
    assert sim.check("society seed 3 final first_farming", "never").expected == "8 5 0 0"
    assert not sim.check("society seed 3 final first_writing", "8 5 0 0").ok
    assert not sim.check("society seed 3 final stage", "chiefdoms ").ok
    assert not sim.check("society seed 3 final technology_list", "fire stone_tools").ok


def test_gate_rejects_a_changed_digit(sim, all_lines):
    """One digit of the answer changed: a count is then wrong; a measured number is wrong when the
    change is more than 5 % (the gate compares with the unrounded value, hence 4 and 6)."""
    rng = random.Random(6)
    numeric = [ln for ln in all_lines if ln.kind != "record" and ln.prompt.split()[-1] not in WORD_KEYS
               and ln.answer != "never"]
    rejected = 0
    for ln in rng.sample(numeric, 4000):
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
    assert rejected > 2600


def test_gate_rejects_answers_that_are_not_finite_numbers(sim):
    for prompt in ("society seed 3 final population", "society seed 3 step 4 0 skill", "society seed 3 final gini",
                   "society seed 3 step 4 0 groups", "society seed 3 params land",
                   "society seed 3 final first_farming"):
        for answer in ("1 e 9 9 9", "minus 1 e 9 9 9", "", "point", "e", "yes", "5 2 0 0 0 people", "5 2 . 0",
                       "1 e 3 0 0", "never"):
            v = sim.check(prompt, answer)
            assert not v.ok and v.expected is not None, (prompt, answer, v)
    assert sim.check("society seed 3 final population", "1 e 9 9 9").reason == "not a number"
    assert sim.check("society seed 3 step 0 skill", "0").ok and not sim.check("society seed 3 step 0 skill", "1").ok


def test_tolerance_is_five_percent_for_numbers_and_exact_for_counts(sim, rollouts):
    r = rollouts[3]
    people = r.summary["population"]
    assert sim.check("society seed 3 final population", dense_value("population", round(people * 1.03))).ok
    assert sim.check("society seed 3 final population", dense_value("population", round(people * 0.97))).ok
    assert not sim.check("society seed 3 final population", dense_value("population", round(people * 1.1))).ok
    assert not sim.check("society seed 3 final population", dense_value("population", round(people * 0.9))).ok
    assert close(parse_num(dense_value("population", people)), people, rel=0.05)
    assert sim.check("society seed 3 final cooperation", num(sig(r.summary["cooperation"] * 0.97))).ok
    assert not sim.check("society seed 3 final cooperation", num(sig(r.summary["cooperation"] * 0.9))).ok
    n = r.summary["groups"]
    assert sim.check("society seed 3 final groups", num(n)).ok
    assert not sim.check("society seed 3 final groups", num(n + 1)).ok
    assert not sim.check("society seed 3 final groups", num(n - 1)).ok
    assert sim.check("society seed 3 final first_farming", "8 5 0 0").ok
    assert not sim.check("society seed 3 final first_farming", "8 7 5 0").ok
    assert sim.check("society seed 3 params words", "4 0").ok and not sim.check("society seed 3 params words", "4 1").ok
    assert sim.check("society seed 3 params compositional", "1").ok
    assert not sim.check("society seed 3 params compositional", "0").ok
    assert not sim.check("society seed 3 params compositional", "yes").ok
    assert sim.check("society seed 3 params hamilton", "no").ok
    assert not sim.check("society seed 3 params hamilton", "yes").ok
    assert sim.check("society seed 3 params land", "1 9 5 0").ok
    assert not sim.check("society seed 3 params land", "2 9 5 0").ok


@pytest.mark.parametrize("prompt", [
    "turkey capital", "q spoon color", "carbon protons", "calc 1 2 plus 7", "check 1 2 plus 7 equals 2 0",
    "gravity seed 7 step 2 0 clumps", "gravity seed 7 final clumps", "nucleo seed 3 final helium_fraction",
    "planets seed 7 final gas", "planets seed 7 step 3 planets", "chem seed 7 final water",
    "life seed 7 step 1 2 next population", "life seed 7 final stage", "world seed 3 era planets",
    "stars seed 3 final metallicity", "stars seed 3 step 4 stage", "cells seed 3 final stage",
    "bodies seed 3 step 4 population", "senses seed 3 final stage", "signals seed 7 step 1 2 success",
    "signals seed 7 say predator near", "signals seed 7 final words", "world7 seed 3 era society",
    "world7 seed 3 final stage", "society predict hamilton relatedness 0 point 5 benefit 4 cost 1",
    "society predict hierarchy population 3 4 5 6 3", "signals predict expected_success signals 4 states 4 shared 3",
    "society", "society seed 7", "society seed 7 step 3", "society seed 7 step 3 clumps",
    "society seed 7 step 8 1 population", "society seed 7 step 8 0 next population", "society seed 7 final skill",
    "society seed 7 final trade", "society seed x final stage", "society seed 7 final", "society seed 7 params year",
    "society seed 7 params stake", "society seed 7 params multiplier", "society seed 12 step 3 population",
    "society seed 7 step 3 next", "society seed 7 step 3 population groups", "", "society seed 0 7 final stage",
    "society seed 7 step 0 3 population", "society seed 7 step 0 0 population", "society seed minus 7 final stage",
    "society seed 7 point 5 final stage", "q society seed 7 final stage", "society seed 7 final stage extra",
    "society seed 7 step 3 next next population", "society seed 7 era society", "society  seed 7 final stage",
    "society seed 7 final stage ", " society seed 7 final stage", "society seed 7 step 3 final stage",
    "society seed 7 final technology_list fire", "society seed 7 step 3 n_prev", "society seed 7 final peak",
])
def test_does_not_own_other_prompts(sim, prompt):
    assert not sim.owns(prompt) and not society.owns(prompt)
    v = sim.check(prompt, "1")
    assert v.ok is False and v.expected is None and v.reason == "not my question"


def test_does_not_own_the_lines_of_round_5_and_round_6(sim):
    """Real prompts of the frozen levels and of a round-5 gate: none is this level's."""
    prompts = []
    for name in ("nucleo", "gravity", "planets", "chem", "life"):
        m = importlib.import_module(f"haishool.cosmos.{name}")
        r = m.simulation().run(3, **m.random_params(random.Random(3)))
        prompts += [ln.prompt for ln in m.lines(r) if ln.kind != "record"][::7]
    predict = importlib.import_module("haishool.cosmos.predict")
    prompts += [ln.prompt for ln in predict.generate(random.Random(1), 200)]
    maths = importlib.import_module("haishool.truth.maths")
    prompts += [ln.prompt for ln in maths.gate().generate(random.Random(1), 200) if ln.kind != "record"]
    assert len(prompts) > 800
    for prompt in prompts:
        assert not sim.owns(prompt) and not LESSONS.owns(prompt), prompt
        assert sim.check(prompt, "1").reason == LESSONS.check(prompt, "1").reason == "not my question"
    for prompt in ("society seed 3 final stage", "society predict hierarchy population 1 5 0"):
        assert not maths.gate().owns(prompt)


def test_does_not_own_the_lines_and_lessons_of_the_other_round_7_levels():
    """Real prompts of the levels beside this one, their story and their lessons: none is this
    level's. A level that does not import or run here is not this file's to test."""
    checked = 0
    for name in ("stars", "cells", "bodies", "senses", "signals", "world7"):
        try:
            m = importlib.import_module(f"haishool.evo.{name}")
            r = m.simulation().run(3, **m.random_params(random.Random(3)))
            prompts = [ln.prompt for ln in m.lines(r) if ln.answer][::3]
            prompts += [ln.prompt for ln in m.LESSONS.generate(random.Random(1), 200)]
        except Exception:  # noqa: BLE001 - another level, another file's tests
            continue
        for prompt in prompts:
            assert not society.owns(prompt) and not LESSONS.owns(prompt), (name, prompt)
            assert society.check(prompt, "1").reason == LESSONS.check(prompt, "1").reason == "not my question"
        checked += 1
    assert checked >= 4, "most of the other levels can be asked"


@pytest.mark.parametrize("prompt", [
    "society seed 7 step 0 population", "society seed 7 step 8 0 stage", "society seed 7 step 7 9 next stage",
    "society seed 1 2 3 4 step 1 2 next cooperation", "society seed 7 final technology_list",
    "society seed 7 final first_writing", "society seed 7 final outcome", "society seed 7 params critical_population",
    "society seed 7 params compositional", "society seed 7 step 4 births", "society seed 7 step 4 next store",
    "society seed 0 final stage", "society seed 7 step 1 0 gini",
])
def test_owns_its_prompts(sim, prompt):
    assert sim.owns(prompt) and society.owns(prompt) and not LESSONS.owns(prompt)
    assert sim.check(prompt, "nonsense words").expected is not None
    assert society.check(prompt, "nonsense words").ok is False


# ---------------------------------------------------------------------------------------------
# the lessons
# ---------------------------------------------------------------------------------------------

def test_lessons_are_accepted_and_equal_the_simulations_own_functions():
    made = LESSONS.generate(random.Random(1), 400)
    assert len(made) == 400 and [ln.text for ln in made] == [ln.text for ln in LESSONS.generate(random.Random(1), 400)]
    assert [ln.text for ln in made] != [ln.text for ln in LESSONS.generate(random.Random(2), 400)]
    seen = collections.Counter()
    for ln in made:
        assert is_dense(ln.text) and n_tokens(ln.text) <= 128, ln.text
        assert "." not in ln.prompt and "." not in ln.answer
        assert ln.topic == "predict_society" and ln.kind == "calc"
        assert LESSONS.owns(ln.prompt) and not society.owns(ln.prompt)
        v = LESSONS.check(ln.prompt, ln.answer)
        assert v.ok and v.expected == ln.answer, (ln.text, v)
        name, values = LESSONS.parse(ln.prompt)
        truth = getattr(society, RULES[name].compute.__name__)(*values)  # the function the simulation calls
        assert RULES[name].compute is getattr(society, RULES[name].compute.__name__)
        assert ln.answer == (truth if isinstance(truth, str) else num(truth)), ln.text
        assert ln.meta["split_key"] == LESSONS.split_key(ln.prompt) and ln.meta["conditioning"] == "explicit_inputs"
        assert len(ln.answer.split()) <= 9, ln.text
        seen[name] += 1
    assert set(seen) == set(RULES) and len(RULES) == 22, "400 lessons cover every rule"
    for name in ("hamilton", "payoff_cooperator", "payoff_defector", "replicator", "skill_change",
                 "critical_population", "logistic", "capacity", "hierarchy", "gini", "doubling_years"):
        assert name in RULES, name


def test_lessons_reject_altered_answers():
    for ln in LESSONS.generate(random.Random(3), 400):
        if ln.answer in ("yes", "no"):
            wrong = "no" if ln.answer == "yes" else "yes"
        elif ln.answer in STAGES:
            wrong = next(s for s in STAGES if s != ln.answer)
        else:
            wrong = "9 " + ln.answer if not ln.answer.startswith("minus") else ln.answer.replace("minus ", "", 1) + " 1"
        v = LESSONS.check(ln.prompt, wrong)
        assert not v.ok and v.expected == ln.answer, (ln.text, wrong)
        for nonsense in ("", "many", "1 e 9 9 9", "point"):
            assert not LESSONS.check(ln.prompt, nonsense).ok
    assert LESSONS.check("society seed 3 final stage", "chiefdoms") .reason == "not my question"
    assert not LESSONS.owns("society predict hamilton relatedness 0 point 5 benefit 4")
    assert not LESSONS.owns("society predict hamilton relatedness 2 benefit 4 cost 1")
    assert not LESSONS.owns("signals predict hamilton relatedness 0 point 5 benefit 4 cost 1")
    assert not LESSONS.owns("society predict payoff_cooperator group 4 cooperators 5 multiplier 2 stake 1 punishment 0")
    assert not LESSONS.owns("society predict payoff_defector group 4 cooperators 4 multiplier 2 stake 1 punishment 0")
    assert not LESSONS.owns("society predict hierarchy population 3 point 5")
    assert not LESSONS.owns("society predict gini poorest 0 poor 0 middle 0 rich 0 richest 0")


def test_pinned_lessons():
    for prompt, answer in (
            # the three lessons the module docstring shows
            ("society predict hamilton relatedness 0 point 5 9 benefit 1 point 5 cost 1", "no"),
            ("society predict skill_change alpha 1 0 point 6 5 beta 1 point 6 6 learners 9 6 7 6 0", "9 point 3 6 5"),
            ("society predict hierarchy population 3 4 5 6 3", "4"),
            ("society predict hamilton relatedness 0 point 5 benefit 4 cost 1", "yes"),
            ("society predict hamilton relatedness 0 point 1 2 5 benefit 4 cost 1", "no"),
            ("society predict stake benefit 4 cost 1 group 5", "2"),
            ("society predict multiplier benefit 4 cost 1 group 5", "2 point 5"),
            ("society predict payoff_cooperator group 5 cooperators 3 multiplier 2 point 5 stake 2 "
             "punishment 0 point 6", "0 point 6"),
            ("society predict payoff_defector group 5 cooperators 3 multiplier 2 point 5 stake 2 punishment 0 point 5",
             "1 point 5"),
            ("society predict kin relatedness 0 point 5 population 1 0 0", "0 point 1 2 5"),
            ("society predict replicator share 0 point 5 cooperator 2 defector 1 rate 0 point 2", "0 point 5 5"),
            ("society predict replicator share 0 point 5 cooperator 1 0 defector minus 5 rate 0 point 5", "1"),
            ("society predict fidelity success 1 words 1 0 compositional 0 writing 0", "0 point 5"),
            ("society predict fidelity success 1 words 1 0 compositional 1 writing 1", "1"),
            ("society predict fidelity success 1 words 1 0 compositional 0 point 5 writing 0", "0 point 6 5"),
            ("society predict effective_alpha alpha 3 skill 6 0 0 fidelity 0 point 5", "1 2"),
            ("society predict learners own 1 0 0 others 1 0 0 0 connectedness 0 point 5", "6 0 0"),
            ("society predict skill_change alpha 3 beta 1 learners 1 0 0", "2 point 1 8 2"),
            ("society predict critical_population alpha 3 beta 1", "1 1 point 2 8"),
            ("society predict technologies skill 1 0 0 0 hierarchy 2", "7"),
            ("society predict technologies skill 1 0 0 0 hierarchy 3", "8"),
            ("society predict capacity land 1 0 0 technologies 6 cooperation 1", "2 6 9 6"),
            ("society predict capacity land 1 0 0 technologies 0 cooperation 0", "4 0"),
            ("society predict food population 1 0 0 capacity 2 0 0 technologies 6", "1 2 5"),
            ("society predict logistic population 1 0 0 rate 0 point 2 5 capacity 2 0 0", "1 1 2 point 5"),
            ("society predict doubling_years rate 0 point 0 1", "6 9 point 3 1"),
            ("society predict hierarchy population 1 4 9", "0"),
            ("society predict hierarchy population 1 5 0", "1"),
            ("society predict hierarchy population 3 2 4 0 0", "4"),
            ("society predict stage population 5 3 9 9", "villages"),
            ("society predict stage population 5 4 0 0", "chiefdoms"),
            ("society predict roles specialists 2 0 0", "8"),
            ("society predict gini poorest 1 poor 2 middle 3 rich 4 richest 5", "0 point 2 6 6 7"),
            ("society predict gini poorest 0 poor 0 middle 0 rich 0 richest 5", "0 point 8"),
            ("society predict win_probability size 1 0 0 cooperation 1 other_size 1 0 0 other_cooperation 0",
             "0 point 7 1 4 3")):
        v = LESSONS.check(prompt, answer)
        assert v.ok, (prompt, v)


def test_lesson_records_say_what_the_rules_compute():
    records = LESSONS.records()
    assert len(records) == len(RULES)
    for ln in records:
        assert ln.kind == "record" and is_dense(ln.text) and n_tokens(ln.text) <= 128, ln.text
    text = {ln.prompt.split(".")[0]: ln.text for ln in records}
    assert "fire 3 0. stone_tools 9 0. clothing 1 5 0. boats 2 2 0. pottery 3 2 0. farming 4 5 0. metal 7 2 0. " \
           "writing 9 0 0. mathematics 1 2 6 0. printing 1 6 2 0." in text["society predict technologies"]
    assert "farming 1 0. metal 2. writing 1." in text["society predict capacity"]
    assert "1 from 1 5 0. 2 from 9 0 0. 3 from 5 4 0 0. 4 from 3 2 4 0 0." in text["society predict hierarchy"]
    assert "0 point 2 plus 0 point 6 times success" in text["society predict fidelity"]
    assert "skill over 6 0 0" in text["society predict effective_alpha"]
    assert "surplus rate 0 point 0 5. from pottery 0 point 1. from farming 0 point 2 5. from metal 0 point 3." \
        in text["society predict food"]


def test_the_simulation_calls_the_lesson_functions(monkeypatch):
    """Every lesson's function is called by a run, with the module's own name: a lesson is true of
    the simulation because the simulation computes with it."""
    calls = collections.Counter()

    def counted(name, f):
        def wrapper(*args):
            calls[name] += 1
            return f(*args)
        return wrapper

    for name, rule in RULES.items():
        monkeypatch.setattr(society, rule.compute.__name__, counted(name, rule.compute))
    r = Society().rollout(3)
    assert r.summary["conflicts"] > 0
    assert set(calls) == set(RULES), set(RULES) - set(calls)
    for name in ("payoff_cooperator", "payoff_defector", "replicator", "skill_change", "logistic", "capacity",
                 "kin", "fidelity", "effective_alpha", "learners", "food", "technologies"):
        assert calls[name] > 10000, name
    assert calls["hamilton"] == calls["critical_population"] == calls["doubling_years"] == 1
    assert calls["gini"] == calls["roles"] == 81 and calls["win_probability"] == r.summary["conflicts"]


def test_the_rules_themselves():
    assert hamilton(0.5, 2.1, 1) == "yes" and hamilton(0.5, 1.9, 1) == "no" and hamilton(0.5, 2, 1) == "no"
    # the game with benefit b to the others and net cost c: Hamilton's rule falls out of the payoffs
    for r, b, c, x in ((0.5, 3.0, 1.0, 0.3), (0.1, 3.0, 1.0, 0.7), (0.25, 4.0, 1.0, 0.5), (0.0, 6.0, 2.0, 0.9)):
        s, m = stake_of(b, c, 5), multiplier_of(b, c, 5)
        diff = payoff_cooperator(5, 1 + 4 * (r + (1 - r) * x), m, s, 0) - payoff_defector(5, 4 * (1 - r) * x, m, s, 0)
        assert diff == pytest.approx(r * b - c)
        assert 1 < m < 5, "a dilemma: the pot grows, but not enough to pay for oneself"
    assert payoff_cooperator(4, 4, 1.6, 20, 0) == pytest.approx(12) and payoff_defector(4, 0, 1.6, 20, 0) == 0
    assert payoff_defector(4, 3, 1.6, 20, 3) == pytest.approx(24 - 9)
    assert payoff_cooperator(4, 3, 1.6, 20, 3) == pytest.approx(24 - 20 - 1)
    assert replicator_step(0.0, 5, 1, 0.2) == 0 and replicator_step(1.0, 1, 5, 0.2) == 1
    assert replicator_step(0.9, 9, 0, 0.5) == 1 and replicator_step(0.1, 0, 9, 0.5) == 0
    assert tremble(0) == 0.01 and tremble(1) == pytest.approx(0.99) and tremble(0.5) == pytest.approx(0.5)
    assert skill_change(3, 1, critical_population(3, 1)) == pytest.approx(0, abs=1e-12)
    assert skill_change(3, 1, 10) < 0 < skill_change(3, 1, 12)
    assert teaching_fidelity(0, 40, 1, 0) == 0.2 and teaching_fidelity(1, 0, 0, 0) == 0.2
    assert teaching_fidelity(1, 40, 1, 0) == pytest.approx(0.8) and teaching_fidelity(1, 40, 1, 1) == 1
    assert teaching_fidelity(0.9, 40, 0, 0) < teaching_fidelity(0.9, 40, 0.5, 0) < teaching_fidelity(0.9, 40, 1, 0)
    assert teaching_fidelity(0.9, 5, 0, 0) < teaching_fidelity(0.9, 40, 0, 0), "more words say more"
    assert teaching_fidelity(0.9, 5, 1, 0) == teaching_fidelity(0.9, 40, 1, 0), "a rule needs no long list"
    assert [technologies(t, 9) for _, t, _ in TECHS] == list(range(1, 11))
    assert [technologies(t - 0.5, 9) for _, t, _ in TECHS] == list(range(10))
    assert technologies(5000, 2) == 7 and technologies(5000, 3) == 10 and technologies(0, 9) == 0
    assert capacity(100, 6, 1) / capacity(100, 5, 1) == pytest.approx(10), "farming feeds ten times as many"
    assert capacity(100, 3, 0) / capacity(100, 3, 1) == pytest.approx(0.4)
    assert [hierarchy_levels(n) for n in (1, 149, 150, 899, 900, 5399, 5400, 32399, 32400, 194400, 1166400)] == \
        [0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 6]
    assert [stage_of(n) for n in (30, 150, 900, 5400, 32400, 10 ** 7)] == list(STAGES[:5]) + ["states"]
    assert [role_count(s) for s in (0, 1, 2, 3, 4, 7, 8, 1023, 1024)] == [1, 2, 2, 3, 3, 4, 4, 11, 11]
    assert gini(3, 3, 3, 3, 3) == 0 and gini(0, 0, 0, 0, 0) == 0 and gini(0, 0, 0, 0, 7) == pytest.approx(0.8)
    assert gini(1, 2, 3, 4, 5) == gini(5, 3, 1, 2, 4) == pytest.approx(4 / 15)
    assert wealth_fifths(100, 50.0, 0) == pytest.approx((0.6,) * 5)
    assert sum(wealth_fifths(100, 50.0, 3)) * 20 == pytest.approx(0.1 * 100 + 50.0), "the store is shared out whole"
    assert list(wealth_fifths(100, 50.0, 3)) == sorted(wealth_fifths(100, 50.0, 3))
    assert doubling_years(math.log(2)) == pytest.approx(1)


# --- the review of round 7: formulas written a second time, from the docstring alone ---------------

_GAMMA = 0.5772156649015329
_THRESHOLDS = (30, 90, 150, 220, 320, 450, 720, 900, 1260, 1620)
_FACTORS = (1.2, 1.3, 1.2, 1.2, 1.2, 10, 2, 1, 1.2, 1.2)


def _ref_technologies(skill, hierarchy):
    count = 0
    for i, threshold in enumerate(_THRESHOLDS):
        if skill < threshold or (i == 7 and hierarchy < 3):  # writing is the eighth
            break
        count += 1
    return count


def _ref_hierarchy(people):
    return sum(people >= size for size in (150, 900, 5400, 32400, 194400, 1166400))


def _ref_capacity(land, count, cooperation):
    return land * math.prod(_FACTORS[:count]) * (0.4 + 0.6 * cooperation)


def _ref_gini(*wealth):
    pairs = sum(abs(wealth[i] - wealth[j]) for i in range(5) for j in range(i + 1, 5))
    return pairs / (5 * sum(wealth))


def _ref_strength(size, cooperation):
    return size * (0.4 + 0.6 * cooperation)


REFERENCE = {
    "hamilton": lambda r, b, c: "yes" if r * b > c else "no",
    "stake": lambda b, c, g: c + b / (g - 1),
    "multiplier": lambda b, c, g: b * g / ((g - 1) * c + b),
    "payoff_cooperator": lambda g, k, m, s, p: m * s * k / g - s - p * (g - k) / 3,
    "payoff_defector": lambda g, k, m, s, p: m * s * k / g - p * k,
    "kin": lambda r, n: r if n <= 25 else r * 25 / n,
    "replicator": lambda x, c, d, rate: min(1, max(0, x + rate * x * (1 - x) * (c - d))),
    "fidelity": lambda s, w, comp, wr: min(1, 0.2 + 0.6 * s * (comp + (1 - comp) * w / (w + 10)) + 0.2 * wr),
    "effective_alpha": lambda a, z, f: a * (1 + z / 600) / f,
    "learners": lambda own, others, c: own + c * others,
    "skill_change": lambda a, b, n: -a + b * (_GAMMA + math.log(n)),
    "critical_population": lambda a, b: math.exp(a / b - _GAMMA),
    "technologies": _ref_technologies,
    "capacity": _ref_capacity,
    "food": lambda n, k, t: (1 + (0.3 if t >= 7 else 0.25 if t >= 6 else 0.1 if t >= 5 else 0.05)) * min(n, k),
    "logistic": lambda n, r, k: n + r * n * (1 - n / k),
    "doubling_years": lambda r: math.log(2) / r,
    "hierarchy": _ref_hierarchy,
    "stage": lambda n: ("bands", "tribes", "villages", "chiefdoms", "states", "states", "states")[_ref_hierarchy(n)],
    "roles": lambda s: 1 + int(math.floor(math.log2(1 + s))),
    "gini": _ref_gini,
    "win_probability": lambda n, x, m, y: _ref_strength(n, x) / (_ref_strength(n, x) + _ref_strength(m, y)),
}


def test_lessons_of_four_seeds_agree_with_formulas_written_a_second_time():
    assert set(REFERENCE) == set(RULES)
    seen = collections.Counter()
    for seed in (1, 2, 3, 4):
        made = LESSONS.generate(random.Random(seed), 500)
        assert len(made) == 500 and [ln.text for ln in made] == \
            [ln.text for ln in LESSONS.generate(random.Random(seed), 500)]
        for ln in made:
            assert is_dense(ln.text) and n_tokens(ln.text) <= 60, ln.text
            assert "." not in ln.prompt and "." not in ln.answer
            v = LESSONS.check(ln.prompt, ln.answer)
            assert v.ok and v.expected == ln.answer, (ln.text, v)
            assert not society.owns(ln.prompt)
            name, values = LESSONS.parse(ln.prompt)
            mine = REFERENCE[name](*values)
            if isinstance(mine, str):
                assert ln.answer == mine, ln.text
            elif isinstance(mine, int):
                assert parse_num(ln.answer) == mine, (ln.text, mine)
            else:  # the answer has 4 significant digits
                assert parse_num(ln.answer) == pytest.approx(mine, rel=6e-4, abs=1e-12), (ln.text, mine)
            seen[name] += 1
    assert set(seen) == set(RULES) and min(seen.values()) >= 40


def test_lesson_answers_are_not_dominated_by_one_value():
    made = LESSONS.generate(random.Random(11), 11000)
    answers = collections.defaultdict(collections.Counter)
    for ln in made:
        answers[ln.prompt.split()[2]][ln.answer] += 1
    top = {name: c.most_common(1)[0][1] / sum(c.values()) for name, c in answers.items()}
    assert set(top) == set(RULES)
    assert 0.45 <= answers["hamilton"]["yes"] / sum(answers["hamilton"].values()) <= 0.55, "a yes/no rule in balance"
    assert set(answers["hierarchy"]) == set("012345") and top["hierarchy"] < 0.25
    assert set(answers["stage"]) == set(STAGES[:5]) and top["stage"] < 0.36, "states spans two levels"
    assert set(answers["roles"]) == {num(k) for k in range(1, 18)} and top["roles"] < 0.12
    assert set(answers["technologies"]) == {num(k) for k in range(11)} and top["technologies"] < 0.25,         "skill is drawn bracket first: one number of technologies in eleven, 7 also for writers without office"
    assert top["payoff_defector"] < 0.22 and top["replicator"] < 0.16
    for name in set(RULES) - {"hamilton", "hierarchy", "stage", "roles", "technologies", "payoff_defector",
                              "replicator"}:
        assert top[name] < 0.03, (name, top[name])
    # a quarter of the kin lessons are inside a band, where the full relatedness holds
    kin = [LESSONS.parse(ln.prompt)[1][1] for ln in made if ln.prompt.split()[2] == "kin"]
    assert 0.15 < sum(n <= 25 for n in kin) / len(kin) < 0.35
    # the change of skill is asked on both sides of the critical population
    change = [ln.answer for ln in made if ln.prompt.split()[2] == "skill_change"]
    assert 0.25 < sum(a.startswith("minus") for a in change) / len(change) < 0.6
    assert len({ln.prompt for ln in made}) > 0.9 * len(made)


def test_lesson_ladders_fit_the_ranges_and_ask_at_the_thresholds():
    assert isinstance(LESSONS, society.SocietyLessons) and isinstance(LESSONS, LessonGate)
    assert EDGE_RULES <= {rule for rule, _ in LADDERS}
    for (rule, name), edges in LADDERS.items():
        spec = next(s for s in RULES[rule].inputs if s.name == name)
        assert spec.integer and edges[0] == spec.low and edges[-1] == spec.high + 1, (rule, name)
        assert list(edges) == sorted(set(edges)) and all(type(e) is int for e in edges)
    assert LADDERS[("hierarchy", "population")][1:6] == (150, 900, 5400, 32400, 194400)
    assert [hierarchy_levels(e) for e in LADDERS[("hierarchy", "population")][:6]] == [0, 1, 2, 3, 4, 5]
    assert [role_count(e) for e in LADDERS[("roles", "specialists")][:17]] == list(range(1, 18))
    assert [technologies(e, 9) for e in LADDERS[("technologies", "skill")][:11]] == list(range(11))
    made = LESSONS.generate(random.Random(21), 6000)
    skills = [LESSONS.parse(ln.prompt)[1][0] for ln in made if ln.prompt.split()[2] == "technologies"]
    for _, edge, _ in TECHS:
        assert edge in skills and edge - 1 in skills, edge
    assert all(type(x) is int for x in skills) and 0 <= min(skills) < 30 and 1620 < max(skills) <= 2000
    people = [LESSONS.parse(ln.prompt)[1][0] for ln in made if ln.prompt.split()[2] in ("hierarchy", "stage")]
    for edge in (150, 900, 5400, 32400, 194400):
        assert edge in people and edge - 1 in people, edge
    assert 1 <= min(people) < 150 and 194400 < max(people) <= 400000
    assert LESSONS.check("society predict hierarchy population 1 9 4 4 0 0", "5").ok
    assert LESSONS.check("society predict hierarchy population 1 9 4 3 9 9", "4").ok
    assert LESSONS.check("society predict stage population 4 0 0 0 0 0", "states").ok
    assert not LESSONS.owns("society predict hierarchy population 4 0 0 0 0 1")
    assert not LESSONS.owns("society predict hamilton relatedness 0 point 5 benefit 4 cost 4 point 6")
    # whatever the draw, the base class judges: a lesson made by hand is judged the same way
    assert LESSONS.line("hierarchy", [150]).text == "q society predict hierarchy population 1 5 0. a 1."
    assert LessonGate(SIM, RULES).check("society predict hierarchy population 1 5 0", "1").ok


def test_toy_rules_are_called_toy_rules_in_the_records():
    text = {ln.prompt.split(".")[0].split()[-1]: ln.text for ln in LESSONS.records()}
    for name in ("kin", "fidelity", "effective_alpha", "learners", "technologies", "capacity", "food", "hierarchy",
                 "stage", "roles", "win_probability"):
        assert "toy" in text[name].split(), name
    for name in ("hamilton", "stake", "multiplier", "payoff_cooperator", "payoff_defector", "replicator",
                 "skill_change", "critical_population", "logistic", "doubling_years", "gini"):
        assert "toy" not in text[name].split(), name
    assert "30 to 1620" in KEYS["skill"] and (TECHS[0][1], TECHS[-1][1]) == (30, 1620)


def test_punishment_gives_two_stable_states_with_the_threshold_of_the_docstring():
    """cooperator - defector = r b - c + 4 p (1 - r) (x - (1 - x) / 3), which is zero at
    x = 1 / 4 + 3 (c - r b) / (16 p (1 - r))."""
    for r, b, c, p in ((0.0, 3.0, 1.0, 1.0), (0.1, 3.0, 1.0, 0.5), (0.02, 4.5, 1.0, 1.5), (0.0, 2.0, 1.0, 0.8)):
        s, m = stake_of(b, c, GAME_SIZE), multiplier_of(b, c, GAME_SIZE)
        edge = 0.25 + 3 * (c - r * b) / (16 * p * (1 - r))
        assert 0.25 < edge < 1

        def difference(x):
            return (payoff_cooperator(GAME_SIZE, 1 + 4 * (r + (1 - r) * x), m, s, p)
                    - payoff_defector(GAME_SIZE, 4 * (1 - r) * x, m, s, p))

        assert difference(edge) == pytest.approx(0, abs=1e-12)
        assert difference(edge - 0.05) < 0 < difference(edge + 0.05)
        assert replicator_step(edge - 0.05, difference(edge - 0.05), 0.0, 0.2) < edge - 0.05
        assert replicator_step(edge + 0.05, difference(edge + 0.05), 0.0, 0.2) > edge + 0.05
    # a small fine makes the start harder even where kin would do: r b > c, yet a defector gains among few cooperators
    s, m = stake_of(2.1, 1.0, GAME_SIZE), multiplier_of(2.1, 1.0, GAME_SIZE)
    assert hamilton(0.48, 2.1, 1.0) == "yes"
    assert payoff_cooperator(GAME_SIZE, 1 + 4 * 0.48, m, s, 0.22) < payoff_defector(GAME_SIZE, 0.0, m, s, 0.22)
    assert payoff_cooperator(GAME_SIZE, 1 + 4 * 0.48, m, s, 0.0) > payoff_defector(GAME_SIZE, 0.0, m, s, 0.0)


# ---------------------------------------------------------------------------------------------
# plausibility
# ---------------------------------------------------------------------------------------------

def test_without_language_skill_does_not_accumulate_and_groups_stay_bands(sim):
    for seed in SEEDS:
        p = random_params(random.Random(seed))
        r = sim.run(seed, **dict(p, success=0.02))
        assert sim.conserved(r).ok
        assert max(s["technologies"] for s in r.steps) == 0 and max(s["skill"] for s in r.steps) < 1, seed
        assert {s["stage"] for s in r.steps} <= {"bands", "collapse"}, seed
        assert r.summary["first_farming"] == r.summary["first_writing"] == "never"
        assert r.summary["hierarchy"] == 0 and r.summary["gini"] == 0 and r.summary["technology_list"] == "none"
        assert r.params["critical_population"] > 20000, "more learners than the land can hold"


def test_with_a_compositional_language_large_connected_populations_reach_farming_and_writing(sim):
    farming = writing = states = 0
    for seed in SEEDS:
        p = random_params(random.Random(seed))
        p.update(success=0.9, compositional=1.0, connectedness=max(0.6, p["connectedness"]),
                 land=max(3000.0, p["land"]), isolation=0)
        r = sim.run(seed, **p)
        farming += r.summary["first_farming"] != "never"
        writing += r.summary["first_writing"] != "never"
        states += r.summary["outcome"] == "states"
        if r.summary["first_writing"] != "never":
            assert r.summary["first_farming"] != "never" and r.summary["first_writing"] > r.summary["first_farming"]
            assert max(s["hierarchy"] for s in r.steps) >= 3, "writing needs an administration"
    assert farming >= 30, "most of them farm"
    assert 4 <= writing <= 30, "a minority writes: only where punishment holds large groups together"
    assert 4 <= states <= 30
    # the same language in small bands without any link: no farming, and mostly nothing at all
    nothing = 0
    for seed in SEEDS[:20]:
        p = random_params(random.Random(seed))
        p.update(success=0.9, compositional=1.0, connectedness=0.0, land=1000.0, isolation=0)
        r = sim.run(seed, **p)
        assert r.summary["first_farming"] == "never" and r.summary["technologies"] <= 4, seed
        assert all(s["trade"] == 0 for s in r.steps)
        nothing += r.summary["technologies"] == 0
    assert nothing >= 12


def test_without_punishment_a_full_language_farms_but_never_writes(sim):
    """The docstring: writing appears only where punishment holds large groups together. Kin thin
    out above a band, so without a fine cooperation fails in a large group and none reaches the
    5400 people of three levels; farming does not need that."""
    farming = 0
    for seed in SEEDS:
        p = random_params(random.Random(seed))
        p.update(success=0.9, compositional=1.0, connectedness=max(0.6, p["connectedness"]),
                 land=max(3000.0, p["land"]), isolation=0, punishment=0.0)
        r = sim.run(seed, **p)
        assert r.summary["first_writing"] == "never" and max(s["hierarchy"] for s in r.steps) < 3, seed
        assert r.summary["cooperation"] < 0.2, seed
        farming += r.summary["first_farming"] != "never"
    assert farming >= 30


def test_canonical_runs_are_varied(sim, rollouts):
    """Seeds 1 to 120 with their own parameters: every outcome occurs, a third farms, an eighth writes."""
    runs = [rollouts[seed] if seed in rollouts else sim.rollout(seed) for seed in range(1, 121)]
    outcomes = collections.Counter(r.summary["outcome"] for r in runs)
    assert set(outcomes) == set(STAGES)
    assert 50 <= outcomes["bands"] <= 90 and 8 <= outcomes["states"] <= 30 and outcomes["villages"] >= 6
    assert outcomes["tribes"] >= 4 and outcomes["collapse"] >= 3
    farming = sum(r.summary["first_farming"] != "never" for r in runs)
    writing = sum(r.summary["first_writing"] != "never" for r in runs)
    assert 24 <= farming <= 48 and 8 <= writing <= 24 and writing < farming
    assert sum(r.summary["technologies"] == 0 for r in runs) <= 50
    assert max(r.summary["technologies"] for r in runs) == 10
    stages = {s["stage"] for r in rollouts.values() for s in r.steps}
    assert stages == set(STAGES)
    assert all(r.steps[0]["stage"] == "bands" and r.steps[0]["technologies"] == 0 for r in runs)
    for r in runs:
        if r.summary["first_writing"] != "never":
            assert r.summary["first_farming"] != "never" and r.summary["first_writing"] > r.summary["first_farming"]


def test_cooperation_collapses_without_kin_and_without_punishment(sim):
    for seed in SEEDS[:20]:
        p = random_params(random.Random(seed))
        p.update(relatedness=0.1, benefit=3.0, punishment=0.0, isolation=0)
        r = sim.run(seed, **p)
        assert r.params["hamilton"] == "no"
        assert max(s["cooperation"] for s in r.steps[8:]) < 0.1, seed
        assert max(s["hierarchy"] for s in r.steps) <= 2, "no chiefdom and no state without cooperation"
        assert r.summary["first_writing"] == "never"
        # the same groups with punishment: cooperation holds where it started high enough
        r = sim.run(seed, **dict(p, punishment=1.0))
        assert r.summary["cooperation"] > 0.4, seed
        # kin alone carry cooperation in small bands
        r = sim.run(seed, **dict(p, relatedness=0.5, benefit=6.0, land=1000.0, success=0.02))
        assert r.params["hamilton"] == "yes" and r.summary["cooperation"] > 0.9, seed
        assert r.steps[-1]["largest_group"] <= 25 * 3


def test_more_cooperation_more_food(sim):
    """Two runs that differ only in punishment: where cooperation holds, the land feeds more."""
    for seed in SEEDS[:20]:
        base = dict(land=2000.0, relatedness=0.1, benefit=3.0, success=0.02, connectedness=0.5)
        low, high = sim.run(seed, punishment=0.0, **base), sim.run(seed, punishment=1.5, **base)
        assert low.summary["cooperation"] < 0.1 and high.summary["cooperation"] > 0.6, seed
        assert high.summary["population"] > 1.4 * low.summary["population"], seed
        assert high.steps[-1]["produced"] > 1.4 * low.steps[-1]["produced"], seed
        assert low.summary["technologies"] == high.summary["technologies"] == 0, "nothing else differs"


def test_farming_raises_the_population_by_an_order_of_magnitude(rollouts, sim):
    ratios = []
    for seed in range(1, 121):
        r = rollouts[seed] if seed in rollouts else sim.rollout(seed)
        if r.summary["first_farming"] == "never":
            continue
        k = r.summary["first_farming"] // STEP_YEARS
        before = r.steps[k - 1]["population"]
        after = max(s["population"] for s in r.steps[k:k + 9])  # within 2000 years
        ratios.append(after / before)
    assert len(ratios) >= 25
    assert min(ratios) > 4 and 8 <= statistics.median(ratios) <= 15


def test_more_learners_more_skill(sim):
    """Henrich's formula and the links of Powell, Shennan and Thomas: with everything else the same,
    a larger land and links between the groups let more skill accumulate."""
    def median_skill(**params):
        return statistics.median(sim.run(seed, **params).steps[-1]["skill"] for seed in SEEDS[:20])

    lands = (1000.0, 2000.0, 3000.0, 5000.0)
    alone = [median_skill(land=land, connectedness=0.0) for land in lands]
    linked = [median_skill(land=land, connectedness=0.3) for land in lands]
    assert alone == sorted(alone) and linked == sorted(linked) and alone[0] < 1 < alone[1] < alone[3] < 400
    assert all(b > a + 500 for a, b in zip(alone, linked))


def test_isolating_small_groups_makes_them_lose_technologies(sim):
    base = dict(land=1500.0, relatedness=0.4, benefit=4.0, punishment=0.5, alpha=3.3, beta=1.0, connectedness=0.8,
                success=0.8, words=20, compositional=1.0, growth=0.25)
    for seed in SEEDS[:20]:
        linked = sim.run(seed, **base)
        cut = sim.run(seed, isolation=10000, **base)
        assert cut.steps[:41] == linked.steps[:41], "the same until the links are cut"
        assert cut.steps[40]["technologies"] == 5 and cut.steps[40]["farming"] == "no"
        assert cut.steps[40]["largest_group"] < 150, "small groups"
        assert all(s["trade"] == 0 for s in cut.steps[41:]) and cut.steps[40]["trade"] > 0
        assert cut.summary["technologies"] == 0 < 5 <= linked.summary["technologies"], seed
        assert cut.summary["population"] < 0.6 * cut.steps[40]["population"] and cut.summary["outcome"] == "collapse"
        lost = [s["technologies"] for s in cut.steps[40:]]
        assert lost == sorted(lost, reverse=True), "lost one after the other, the last learned first"
    # large farming groups keep what they know when the links are cut
    big = dict(base, land=4000.0, alpha=2.8, success=0.95)
    for seed in SEEDS[:10]:
        cut = sim.run(seed, isolation=10000, **big)
        assert cut.steps[40]["farming"] == "yes" and cut.summary["farming"] == "yes", seed


def test_inequality_and_roles_grow_with_hierarchy(rollouts):
    by_level = collections.defaultdict(list)
    roles = collections.defaultdict(list)
    for r in rollouts.values():
        for s in r.steps:
            by_level[s["hierarchy"]].append(s["gini"])
            roles[s["stage"]].append(s["roles"])
    assert set(by_level) >= {0, 1, 2, 3, 4}
    medians = [statistics.median(by_level[k]) for k in sorted(by_level)]
    assert medians == sorted(medians) and medians[0] == 0 and 0.3 < medians[4] < 0.6
    assert statistics.median(roles["bands"]) <= 2 < statistics.median(roles["villages"]) \
        < statistics.median(roles["states"])


def test_a_change_in_the_twelfth_digit_moves_no_line(sim, rollouts):
    """Another machine's log may differ in the sixteenth digit. A change ten thousand times larger
    in alpha and beta must not move a single line: no decision sits on a knife's edge."""
    for seed in SEEDS[:20]:
        r = rollouts[seed]
        base = [ln.text for ln in lines(r)]
        p = random_params(random.Random(seed))
        for f in (1 + 1e-12, 1 - 1e-12):
            nudged = sim.run(seed, **dict(p, alpha=p["alpha"] * f, beta=p["beta"] / f))
            assert [ln.text for ln in lines(nudged)] == base, (seed, f)


def test_a_last_digit_of_log_and_exp_moves_no_line(rollouts, monkeypatch):
    """What may differ between machines is the last digit of ``log``, ``exp`` and ``log1p``. Every
    result of theirs is moved here by one and by a thousand last digits, up, down and at random:
    not one line of 20 seeds may change."""
    def shaken(mode: str, ulps: int):
        rng = random.Random(5)
        fake = types.SimpleNamespace(**{k: getattr(math, k) for k in dir(math) if not k.startswith("__")})

        def shake(f):
            def g(x):
                value = f(x)
                way = {"up": 1, "down": -1}.get(mode) or rng.choice((-1, 0, 1))
                for _ in range(ulps if way else 0):
                    value = math.nextafter(value, way * math.inf)
                return value
            return g

        for name in ("log", "exp", "log1p"):
            setattr(fake, name, shake(getattr(math, name)))
        return fake

    base = {seed: [ln.text for ln in lines(rollouts[seed])] for seed in SEEDS[:20]}
    plain = skill_change(3, 1, 100)
    for mode, ulps in (("up", 1), ("down", 1), ("random", 1), ("random", 1000)):
        monkeypatch.setattr(society, "math", shaken(mode, ulps))
        if mode != "random":
            assert skill_change(3, 1, 100) != plain, "the change reaches the rules"
        fresh = Society()
        for seed in SEEDS[:20]:
            assert [ln.text for ln in lines(fresh.rollout(seed))] == base[seed], (seed, mode, ulps)
    monkeypatch.undo()
    assert society.math is math and skill_change(3, 1, 100) == plain


def test_plain_python_numbers_and_one_seeded_generator():
    """Nothing here can depend on a numpy version, on the clock or on the global generator."""
    source = open(society.__file__, encoding="utf-8").read()
    for word in ("import numpy", "from numpy", "import time", "random.seed(", "random.random(", "random.choice(",
                 "random.uniform(", "random.randint(", "random.shuffle(", "random.sample(", "hash("):
        assert word not in source, word


def test_an_empty_land_stays_empty():
    p = Society().run(1).params
    w = _World(random.Random(1), p)
    w.groups.clear()
    w.begin(1)
    w.generation(1)
    state, groups = w.snapshot(1, [100])
    assert state["population"] == 0 and state["groups"] == 0 and state["stage"] == "collapse" and groups == []
    assert state["free_land"] == TERRITORIES and state["technologies"] == 0 and state["farming"] == "no"


def test_groups_can_die_out(sim):
    """Slow growth and raids: bands fall below five people and die; the gate still holds."""
    died = 0
    for seed in range(1, 7):
        r = sim.run(seed, land=1000.0, relatedness=0.0, benefit=1.5, punishment=0.0, success=0.0, growth=0.01,
                    connectedness=0.0)
        assert sim.conserved(r).ok
        died += sum(e["kind"] == "extinct" for e in r.events)
        for e in r.events:
            if e["kind"] == "extinct":
                gone = next(g for g in r.groups[e["step"]] if g["id"] == e["a"])
                assert gone["gone"] == "yes" and gone["n"] == 0 and gone["store"] == 0
        assert r.summary["population"] >= MIN_GROUP
    assert died >= 10


# ---------------------------------------------------------------------------------------------
# the hand-off
# ---------------------------------------------------------------------------------------------

def test_handoff_in_maps_the_language_to_the_parameters(sim):
    assert handoff_in({"stage": "language", "success": 0.914, "words": 12, "compositional": 0.95}) == \
        {"success": 0.914, "words": 12, "compositional": 0.95}
    assert handoff_in({"stage": "calls", "success": 0.4, "words": 2}) == \
        {"success": 0.4, "words": 2, "compositional": 0.0}
    assert handoff_in({"success": 1, "words": 30.0, "compositional": "yes"}) == \
        {"success": 1.0, "words": 30, "compositional": 1.0}
    assert handoff_in({"success": 0, "compositional": "no"}) == {"success": 0.0, "words": 0, "compositional": 0.0}
    for bad in ({}, {"words": 3}, {"success": "high"}, {"success": 1.2}, {"success": -0.1}, {"success": True},
                {"success": float("nan")}, {"success": 0.5, "words": -1}, {"success": 0.5, "words": "many"},
                {"success": 0.5, "words": 2.5}, {"success": 0.5, "compositional": "maybe"},
                {"success": 0.5, "compositional": 1.5}, {"success": 0.5, "compositional": True}):
        with pytest.raises(ValueError):
            handoff_in(bad)
    spoken = sim.run(4, **handoff_in({"success": 0.93, "words": 14, "compositional": 0.96}))
    mute = sim.run(4, **handoff_in({"success": 0.03, "words": 2, "compositional": 0.0}))
    assert sim.conserved(spoken).ok and sim.conserved(mute).ok
    assert spoken.params["success"] == 0.93 and spoken.params["words"] == 14 and spoken.params["compositional"] == 0.96
    assert spoken.summary["technologies"] > mute.summary["technologies"] == 0
    assert set(SUMMARY_KEYS) == set(spoken.summary)
    assert "compositional 0 point 9 6." in params_line(spoken).text


def test_handoff_in_reads_the_summary_of_the_level_below():
    """The level below, if it is there, gives what this level takes: success, words, compositional."""
    signals = pytest.importorskip("haishool.evo.signals")
    for seed in (1, 2, 3):
        try:
            below = signals.simulation().run(seed, **signals.random_params(random.Random(seed))).summary
        except Exception as error:  # the level below is not this file's to test
            pytest.skip(f"signals does not run here: {error!r}")
        got = handoff_in(below)
        assert set(got) == {"success", "words", "compositional"}
        assert got["success"] == below["success"] and got["words"] == below["words"]
        assert got["compositional"] == below["compositional"]
        assert Society().run(seed, **got).params["words"] == below["words"]


def test_a_run_is_fast(sim):
    worst = 0.0
    for seed in range(100, 110):
        start = time.perf_counter()
        sim.run(seed, **random_params(random.Random(seed)))
        worst = max(worst, time.perf_counter() - start)
    assert worst < 2.0
