"""Level 13, the corrected chain on one clock (``haishool/evo/world7.py``).

The module-wide fixture runs the worlds of seeds 1 to 40 once (about two minutes); every test
that needs a world takes it from there. Seed 85, the world of the module docstring's examples,
climbs the whole ladder and has its own fixture.
"""

from __future__ import annotations

import collections
import copy
import hashlib
import os
import random
import subprocess
import sys
import time

import pytest

from haishool.cosmos import Simulation, chem, gravity, life, nucleo, planets
from haishool.cosmos import world as world6
from haishool.evo import LessonGate, bodies, cells, senses, signals, society, stars
from haishool.evo import world7 as w
from haishool.truth import Line, is_dense, num, parse_num

SEEDS = tuple(range(1, 41))
TIMES: dict[int, float] = {}

#: four whole worlds, pinned: a star that dies before its cells get a second try, a world that
#: speaks and stays in bands, a metal-free cloud, and a red dwarf's world that climbs to states.
#: A change of any level or hand-off shows here.
CANARIES = {
    3: {
        "eras": "nucleo stars cloud collapse star planets surface_1 chemistry_1 replicators_1 cells_1",
        "summary": {"hydrogen": 0.723, "helium": 0.266, "metallicity": 0.0108, "formation_time": 4.0,
                    "star_mass": 1.36, "star_lifetime": 3980000000, "planets": 5, "temperate": 1, "oceans": 1,
                    "planet": 1, "rung": "replicators", "outcome": "out_of_time", "by_now": "replicators",
                    "age_at_end": 6712568000, "eras": 10, "technologies": 0, "words": 0, "sentence": "none",
                    "meaning": "none"},
        "lines": 176,
        "sha256": "7f053abb3ce68b5e91164dd43fdf39cdf2648ab050611cd8f66a6cf8ab3a1219",
    },
    29: {
        "eras": "nucleo stars cloud collapse star planets surface_1 chemistry_1 replicators_1 cells_1 bodies_1 "
                "senses_1 signals_1 society_1",
        "summary": {"hydrogen": 0.735, "helium": 0.258, "metallicity": 0.00659, "formation_time": 9.0,
                    "star_mass": 0.339, "star_lifetime": 177000000000, "planets": 5, "temperate": 1, "oceans": 1,
                    "planet": 1, "rung": "language", "outcome": "language", "by_now": "complex_cells",
                    "age_at_end": 14712688000, "eras": 14, "technologies": 0, "words": 7,
                    "sentence": "pa ta ta sa", "meaning": "predator far"},
        "lines": 311,
        "sha256": "6214943e18179f96467f6b9dceeb5d103126b2de4272924510b22797afcda94f",
    },
    31: {
        "eras": "nucleo stars cloud collapse star",
        "summary": {"hydrogen": 0.753, "helium": 0.247, "metallicity": 0.0, "formation_time": 0.25,
                    "star_mass": 5.59, "star_lifetime": 96700000, "planets": 0, "temperate": 0, "oceans": 0,
                    "planet": 0, "rung": "no_metals", "outcome": "no_metals", "by_now": "no_metals",
                    "age_at_end": 252568000, "eras": 5, "technologies": 0, "words": 0, "sentence": "none",
                    "meaning": "none"},
        "lines": 90,
        "sha256": "29a59f024670dbd7a94363eedb5fdb21eb506c37623cc177743f17eeb993e76a",
    },
    85: {
        "eras": "nucleo stars cloud collapse star planets surface_1 chemistry_1 replicators_1 cells_1 bodies_1 "
                "senses_1 signals_1 society_1",
        "summary": {"hydrogen": 0.73, "helium": 0.262, "metallicity": 0.00857, "formation_time": 3.25,
                    "star_mass": 0.0827, "star_lifetime": 1110000000000, "planets": 7, "temperate": 1, "oceans": 1,
                    "planet": 1, "rung": "states", "outcome": "states", "by_now": "states",
                    "age_at_end": 12562688000, "eras": 14, "technologies": 10, "words": 8,
                    "sentence": "me pa pa ma", "meaning": "predator small"},
        "lines": 321,
        "sha256": "80fc8b24d7a57a41812978ef5b895df299d27aad8ac37bbb9d4f3e9f9f67e96a",
    },
}

#: how many of the worlds of seeds 1 to 40 end at each rung
RUNGS_OF_FORTY = {"no_metals": 1, "no_star": 5, "no_temperate": 13, "no_water": 2, "sterile": 2, "replicators": 5,
                  "cells": 5, "bodies": 1, "groups": 1, "signals": 4, "language": 1}


@pytest.fixture(scope="module")
def worlds() -> dict[int, w.WorldRollout]:
    out = {}
    for seed in SEEDS:
        start = time.perf_counter()
        out[seed] = w.run(seed)
        TIMES[seed] = time.perf_counter() - start
    return out


@pytest.fixture(scope="module")
def texts(worlds) -> dict[int, list[Line]]:
    return {seed: w.lines(r) for seed, r in worlds.items()}


@pytest.fixture(scope="module")
def showcase() -> tuple[w.WorldRollout, list[Line]]:
    """Seed 85: the world of the docstring's examples, and its lines."""
    r = w.run(85)
    return r, w.lines(r)


def digest(lines: list[Line]) -> str:
    return hashlib.sha256("\n".join(ln.text for ln in lines).encode()).hexdigest()


def altered(answer: str) -> str:
    """An answer that is certainly another one: another word, or a number more than twice as large
    (the next whole number would still round to the three digits a measured number prints)."""
    n = parse_num(answer)
    return "wrong_word" if n is None else num(n * 2 + 1)


def rung_index(r: w.WorldRollout) -> int:
    return w.LADDER.index(r.summary["rung"])


# ---------------------------------------------------------------------------------------------
# the contract


def test_it_is_a_level_of_the_ladder():
    sim = w.simulation()
    assert isinstance(sim, Simulation) and sim.sim == w.SIM == "world7"
    assert isinstance(w.LESSONS, LessonGate) and w.lesson_gate() is w.LESSONS
    assert w.LESSONS.sim == "world7" and w.LESSONS.topic == "predict_world7"
    assert len(w.RULES) >= 12
    for keys in (*w.ERAS.values(), w.SUMMARY_KEYS, w.PARAM_KEYS):
        assert all(key in w.KEYS for key in keys), [key for key in keys if key not in w.KEYS]


def test_the_ladder_is_defined_once():
    assert len(w.LADDER) == len(w.LINKS) + 1 == len(set(w.LADDER))
    assert w.LADDER[w.LOWER + 1:] == w.LINKS[w.LOWER:]
    assert w.OUTCOMES == (*w.LADDER, "out_of_time")
    assert [link for level in w.LEVELS for link in level.links] == list(w.LINKS[w.LOWER:])
    assert [level.name for level in w.LEVELS] == list(w.PLANET_ERAS[2:])
    assert set(w.SPAN) == {"collapse", "planets", "chemistry", *(level.name for level in w.LEVELS)}
    # the spans from the star's birth: the earth's age within a percent (toy calibration)
    assert sum(w.SPAN.values()) == 4_512_588_000


def test_known_parameters():
    assert w.random_params(random.Random(3)) == {"formation_time": 4.0, "efficiency": 0.6, "cloud_mass": 2.0,
                                                 "spin": 0.06, "cooling": 0.57, "t_gas": 6.0}
    assert w.random_params(random.Random(29)) == {"formation_time": 9.0, "efficiency": 0.15, "cloud_mass": 0.7,
                                                  "spin": 0.19, "cooling": 0.44, "t_gas": 1.5}
    for seed in range(200):
        p = w.random_params(random.Random(seed))
        assert tuple(p) == w.PARAM_KEYS
        assert 0.25 <= p["formation_time"] <= 12 and p["formation_time"] * 4 == int(p["formation_time"] * 4)
        assert p["efficiency"] in w.EFFICIENCIES and p["cloud_mass"] in w.CLOUD_MASSES and p["t_gas"] in w.T_GAS
        assert 0.02 <= p["spin"] <= 0.3 and 0.2 <= p["cooling"] <= 0.6
    with pytest.raises(ValueError):
        w.random_params(random.Random(1), rules=6)


def test_bad_parameters_are_refused():
    for bad in ({"formation_time": 0.3}, {"formation_time": 0.0}, {"formation_time": 14.0}, {"spin": 0.5},
                {"cloud_mass": 0.0}, {"efficiency": 0.0}, {"max_tries": 0}, {"max_tries": 10}, {"atoms": 10},
                {"particles": 2.5}, {"cooling": "fast"}, {"rules": 6}):
        with pytest.raises(ValueError):
            w.run(1, **bad)
    with pytest.raises(TypeError):
        w.run(1, generations=2)
    for seed in (-1, 1.5, True, "3", 10 ** 10):
        with pytest.raises(ValueError):
            w.run(seed)


def test_canary_worlds(worlds, texts, showcase):
    for seed, pinned in CANARIES.items():
        r, lines = (worlds[seed], texts[seed]) if seed in worlds else showcase
        assert r.seed == seed and " ".join(s["era"] for s in r.steps) == pinned["eras"]
        assert r.summary == pinned["summary"]
        assert tuple(r.summary) == w.SUMMARY_KEYS
        assert len(lines) == pinned["lines"]
        assert digest(lines) == pinned["sha256"]
        assert w.conserved(r).ok


def test_a_rerun_gives_the_same_world(worlds, texts):
    for seed in (3, 29):
        again = w.run(seed)
        assert again.steps == worlds[seed].steps and again.summary == worlds[seed].summary
        assert again.params == worlds[seed].params
        assert [ln.text for ln in w.lines(again)] == [ln.text for ln in texts[seed]]
        assert sorted(again.levels) == sorted(worlds[seed].levels)
        for name, level in again.levels.items():
            assert level.steps == worlds[seed].levels[name].steps and level.summary == worlds[seed].levels[name].summary


def test_the_same_bits_in_another_process_whatever_the_hash_seed():
    code = ("import hashlib; from haishool.evo import world7 as w; "
            "print(hashlib.sha256('\\n'.join(ln.text for ln in w.lines(w.run(3))).encode()).hexdigest())")
    env = {**os.environ, "PYTHONHASHSEED": "7"}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True,
                         cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert out.stdout.strip() == CANARIES[3]["sha256"]


def test_every_line_is_dense_and_short(texts):
    total = 0
    for seed, lines in texts.items():
        assert len({ln.text for ln in lines}) == len(lines), f"seed {seed}: a line twice"
        for ln in lines:
            total += 1
            assert is_dense(ln.text), ln.text
            assert w.tokens(ln.text) <= 128, (w.tokens(ln.text), ln.text)
            assert ln.topic in ("world7", "predict_world7")
            if ln.kind != "record":
                assert "." not in ln.prompt and "." not in ln.answer and ln.answer
    assert total > 40 * 85


def test_the_records_print_the_eras(worlds, texts):
    for seed, r in worlds.items():
        head = f"world7 seed {num(seed)} era "
        printed = {}
        for ln in texts[seed]:
            if ln.kind == "record" and ln.prompt.startswith(head) and " drawn." not in ln.prompt:
                name, *fields = [f.strip() for f in ln.prompt[len(head):].rstrip(".").split(".")]
                printed[name] = dict(f.split(" ", 1) for f in fields)
        assert list(printed) == [s["era"] for s in r.steps]
        for step in r.steps:
            assert printed[step["era"]] == {k: w.dense_value(v) for k, v in step.items() if k != "era"}
            assert tuple(printed[step["era"]]) == w.ERAS[w._base(step["era"])]


def test_the_gate_holds_for_forty_seeds(worlds):
    for seed, r in worlds.items():
        verdict = w.conserved(r)
        assert verdict.ok, (seed, verdict.reason)
        assert w.canonical(r)
        assert r.params == {**w.random_params(random.Random(seed)), **w.DEFAULTS}


def test_check_accepts_every_line_and_rejects_another_answer(texts):
    asked = 0
    for seed, lines in texts.items():
        for ln in lines:
            if ln.kind == "record":
                continue
            asked += 1
            assert w.owns(ln.prompt), ln.prompt
            verdict = w.check(ln.prompt, ln.answer)
            assert verdict.ok, (ln.text, verdict)
            assert not w.check(ln.prompt, altered(ln.answer)).ok, ln.text
            assert not w.check(ln.prompt, "").ok
    assert asked > 3000
    # a measured number passes exactly when it rounds to the three digits printed
    assert w.check("world7 seed 3 era star luminosity", "3 point 4 2").ok
    assert w.check("world7 seed 3 era star luminosity", "3 point 4 2 4").ok
    assert not w.check("world7 seed 3 era star luminosity", "3 point 4 3").ok
    assert not w.check("world7 seed 3 era star luminosity", "3 point 5").ok
    # years, counts and parameters exactly
    assert w.check("world7 seed 3 era star star_end", "7 9 8 2 5 6 8 0 0 0").ok
    assert not w.check("world7 seed 3 era star star_end", "7 9 8 2 5 6 8 0 0 1").ok
    assert not w.check("world7 seed 3 era star star_end", "7 point 9 8 e 9").ok
    assert not w.check("world7 seed 3 param cooling", "0 point 5 6 9").ok
    # what a world does not have
    assert w.check("world7 seed 3 era bodies_1 result", "animals_like") == w.Verdict(False, None,
                                                                                    "world7 seed 3 has no era bodies_1")
    assert not w.check("world7 seed 3 why states", "metals").ok
    assert not w.check("world7 seed 3 say predator near", "none").ok
    assert w.truth("world7 seed 3 final rung") == "replicators" and w.truth("signals seed 3 final outcome") is None


def test_owns_only_its_own_prompts(texts):
    foreign = [
        "world seed 3 era planets habitable", "world handoff cooling metallicity 0 point 0 0 6 6 3",
        "world predict handoff cooling metallicity 0 point 0 0 6 6 3", "planets seed 3 rules 7 final habitable",
        "planets7 predict flux luminosity 4 point 7 8 6 orbit 1 point 1 4 8", "chem seed 9 rules 7 final precursors",
        "life seed 7 rules 7 final outcome", "gravity seed 5 rules 7 final star_time",
        "nucleo seed 1 rules 7 final helium", "stars seed 3 final metallicity", "cells seed 3 3 final outcome",
        "bodies seed 3 final outcome", "senses seed 3 5 final outcome", "signals seed 8 7 say predator near",
        "society seed 3 final outcome", "society predict hamilton relatedness 0 point 5 benefit 4 cost 1",
        "world7", "world7 seed 3", "world7 seed 0 3 final rung", "world7 seed 3 final nothing",
        "world7 seed 3 era planets_1 planets", "world7 seed 3 era cells result", "world7 seed 3 era cells_0 result",
        "world7 seed 3 era cells_1 hydrogen", "world7 seed 3 why out_of_time", "world7 seed 3 param particles",
        "world7 seed 3 say predator", "world7 seed 3 word banana", "world7 seed 3 meaning xx",
        "world7 constant nothing", "world7 span nucleo", "world7 ladder after out_of_time",
        "world7 predict star_mass cloud_mass 2", "world7 predict star_mass star_share 0 point 5 cloud_mass 2",
        "world7 predict star_mass cloud_mass 2 star_share 2", "world7 predict gene_length drawn 3 0 l_max 9 0",
        "world7 predict formation_year formation_time 0 point 3", "World7 seed 3 final rung", "q world7 seed 3 final rung",
        " world7 seed 3 final rung", "world7 seed 3 final rung ", "world7  seed 3 final rung",
    ]
    for gate in (planets.LESSONS, signals.LESSONS, society.LESSONS, cells.LESSONS, senses.LESSONS):
        foreign += [ln.prompt for ln in gate.generate(random.Random(2), 20)]
    for prompt in foreign:
        assert not w.owns(prompt), prompt
        assert w.check(prompt, "1").ok is False
    mine = [ln.prompt for ln in texts[29] if ln.kind != "record"]
    assert len(mine) > 250
    others = (world6, nucleo, gravity, planets, chem, life, stars, cells, bodies, senses, signals, society)
    for prompt in mine[::7] + [ln.prompt for ln in w.table_lines()]:
        assert w.owns(prompt)
        for other in others:
            assert not other.owns(prompt), (other.__name__, prompt)


def test_the_tables():
    table = w.table_lines()
    assert len(table) == len(w.CONSTANTS) + len(w.SPAN) + len(w.LADDER)
    for ln in table:
        assert is_dense(ln.text) and w.check(ln.prompt, ln.answer).ok
        assert not w.check(ln.prompt, altered(ln.answer)).ok
    assert w.check("world7 constant max_tries", "6").ok and w.check("world7 span cells", "2 1 6 0 0 0 0 0 0 0").ok
    assert w.check("world7 ladder after cells", "complex_cells").ok and w.check("world7 ladder after states", "none").ok
    assert w.check("world7 constant z_crit", "1 e minus 5").ok
    for ln in w.records():
        assert ln.kind == "record" and is_dense(ln.text) and w.tokens(ln.text) <= 128
    assert len(w.records()) == 2 + len(w.RULES)
    assert all(row["notes"] for row in w.CONSTANTS.values())
    # the real mass function, from the formula: not what the chain samples
    shares = [w.kroupa_share(a, b) for a, b in ((0.08, 0.5), (0.5, 1.5), (1.5, 8), (8, 100))]
    assert abs(sum(shares) - 1) < 1e-12
    assert [round(s, 2) for s in shares[:3]] == [0.76, 0.18, 0.05] and 0.005 < shares[3] < 0.007
    assert w.CONSTANTS["stars_below_half"]["value"] == round(shares[0], 3)


def test_the_examples_of_the_docstring_are_real(showcase):
    """Every example line of the module docstring that is not cut short is a line of seed 85, a table
    question or a lesson, word for word."""
    examples, current = [], None
    for raw in w.__doc__.splitlines():
        if raw.startswith("    world7 ") or raw.startswith("    q world7 "):
            current = [raw.strip()]
            examples.append(current)
        elif current is not None and raw.startswith("        ") and raw.strip():
            current.append(raw.strip())
        else:
            current = None
    whole = [" ".join(parts) for parts in examples if "..." not in " ".join(parts)]
    assert len(whole) >= 20
    real = {ln.text for ln in showcase[1]} | {ln.text for ln in w.table_lines()}
    for text in whole:
        assert text in real, text
    for parts in examples:  # a cut example still starts like a real line
        text = " ".join(parts)
        if "..." in text:
            start = text.split(" ...")[0]
            assert any(line.startswith(start) for line in real), text


def test_generate_gives_the_number_asked_for():
    lines = w.generate(random.Random(5), 60)
    assert len(lines) == 60 and all(is_dense(ln.text) for ln in lines)
    assert [ln.text for ln in lines[:len(w.table_lines())]] == [ln.text for ln in w.table_lines()]
    assert w.generate(random.Random(5), 0) == []


# ---------------------------------------------------------------------------------------------
# lessons


def test_four_hundred_lessons_are_what_the_chain_computes():
    lessons = w.LESSONS.generate(random.Random(1), 400)
    assert [ln.text for ln in lessons] == [ln.text for ln in w.LESSONS.generate(random.Random(1), 400)]
    assert len(lessons) == 400
    seen = collections.Counter()
    for ln in lessons:
        assert is_dense(ln.text) and w.tokens(ln.text) <= 128 and "." not in ln.prompt and "." not in ln.answer
        assert ln.topic == "predict_world7" and ln.kind == "calc"
        assert w.LESSONS.owns(ln.prompt) and w.owns(ln.prompt)
        assert w.LESSONS.check(ln.prompt, ln.answer).ok and w.check(ln.prompt, ln.answer).ok
        assert not w.LESSONS.check(ln.prompt, altered(ln.answer)).ok
        name, values = w.LESSONS.parse(ln.prompt)
        seen[name] += 1
        truth = w.RULES[name].compute(*values)
        assert ln.answer == (truth if isinstance(truth, str) else num(truth))
    assert set(seen) == set(w.RULES)
    for name, rule in w.RULES.items():
        assert getattr(w, rule.compute.__name__) is rule.compute, name
    assert len(w.LESSONS.records()) == len(w.RULES)


def test_lesson_answers_are_not_one_sided():
    lessons = w.LESSONS.generate(random.Random(7), 3000)
    by_rule = collections.defaultdict(collections.Counter)
    for ln in lessons:
        by_rule[ln.prompt.split()[2]][ln.answer] += 1
    assert set(by_rule) == set(w.RULES)
    for name, answers in by_rule.items():
        assert len(answers) >= 2 and max(answers.values()) <= 0.7 * sum(answers.values()), (name, answers)
    for name, words in (("fusion", 2), ("in_time", 2), ("zone", 3), ("surface", 3), ("water", 4)):
        assert len(by_rule[name]) == words and min(by_rule[name].values()) >= 0.1 * sum(by_rule[name].values())


def test_a_world_calls_every_lesson_function(worlds, monkeypatch):
    calls = collections.Counter()
    for name, rule in w.RULES.items():
        def counted(*args, _name=name, _fn=rule.compute):
            calls[_name] += 1
            return _fn(*args)
        monkeypatch.setattr(w, rule.compute.__name__, counted)
    again = w.run(29)
    assert again.steps == worlds[29].steps
    assert sorted(name for name in w.RULES if not calls[name]) == []


def test_a_changed_hand_off_changes_the_world(worlds, monkeypatch):
    monkeypatch.setattr(w, "star_mass_of", lambda cloud_mass, star_share: 1.0)
    changed = w.run(31, formation_time=9.0)
    assert changed.era("star")["star_mass"] == 1.0 and changed.era("star")["lifetime"] == 10_000_000_000


def test_every_hand_off_follows_from_the_lines(worlds, texts):
    counted = collections.Counter()
    for seed, r in worlds.items():
        printed = {ln.text for ln in texts[seed]}
        for name, inputs, expected in w.handoffs(r):
            lesson = w.LESSONS.line(name, list(inputs))
            assert lesson is not None, (seed, name, inputs)
            assert lesson.answer == w.dense_value(expected), (seed, lesson.text, expected)
            assert lesson.text in printed
            counted[name] += 1
    assert set(counted) == set(w.RULES)
    # the inputs are numbers the eras print: rebuilt from the text of seed 29's lines
    r = worlds[29]
    star = {k: w.dense_value(v) for k, v in r.era("star").items()}
    line = f"q world7 predict lifetime star_mass {star['star_mass']} luminosity {star['luminosity']}. a {star['lifetime']}."
    assert line in {ln.text for ln in texts[29]}
    assert line == ("q world7 predict lifetime star_mass 0 point 3 3 9 luminosity 0 point 0 1 9 1. "
                    "a 1 7 7 0 0 0 0 0 0 0 0 0.")


# ---------------------------------------------------------------------------------------------
# the gate notices


def test_the_gate_notices_tampering(worlds):
    def broken(change, seed=29):
        r = copy.deepcopy(worlds[seed])
        change(r)
        verdict = w.conserved(r)
        assert not verdict.ok, change
        return verdict.reason

    assert w.conserved(copy.deepcopy(worlds[29])).ok

    def set_era(name, key, value):
        return lambda r: r.era(name).__setitem__(key, value)

    assert "star" in broken(set_era("star", "star_mass", 0.34))
    assert "cloud" in broken(set_era("cloud", "cooling", 0.3))
    assert "surface_1" in broken(set_era("surface_1", "temperature", 281))
    assert "cells_1" in broken(set_era("cells_1", "result", "cells"))
    assert "signals_1" in broken(set_era("signals_1", "words", 8))
    broken(set_era("stars", "metallicity", 0.0))
    broken(set_era("star", "lifetime", 178000000000))
    broken(set_era("bodies_1", "age", 11712568001))
    broken(set_era("bodies_1", "tries", 1))
    broken(set_era("society_1", "kept", 1))
    broken(set_era("replicators_1", "monomers", 8000))
    broken(lambda r: r.era("nucleo").pop("traces"))
    broken(lambda r: r.steps.pop())
    broken(lambda r: r.steps.append(dict(r.steps[-1])))
    broken(lambda r: r.steps.insert(5, r.steps.pop(6)))
    # the summary
    broken(lambda r: r.summary.__setitem__("rung", "society"))
    broken(lambda r: r.summary.__setitem__("technologies", 3))
    broken(lambda r: r.summary.__setitem__("outcome", "out_of_time"))
    broken(lambda r: r.summary.__setitem__("sentence", "ka ka"))
    broken(lambda r: r.summary.__setitem__("by_now", "states"))
    # the levels
    broken(lambda r: r.levels["collapse"].summary.__setitem__("star", 0.5))
    assert "chemistry_1" in broken(lambda r: setattr(r.levels["chemistry_1"], "seed", 292))
    assert "planets" in broken(lambda r: r.levels["planets"].params.__setitem__("t_gas", 2.0))
    assert "stars" in broken(lambda r: r.levels["stars"].params.__setitem__("efficiency", 0.3))
    assert "nucleo" in broken(lambda r: r.levels["nucleo"].summary.__setitem__("helium", 0.25))
    broken(lambda r: r.levels.pop("planets"))
    broken(lambda r: r.attempts["bodies_1"].pop(0))
    broken(lambda r: r.attempts["bodies_1"].append(r.attempts["bodies_1"][-1]))
    broken(lambda r: r.attempts.pop("society_1"))
    broken(lambda r: r.levels.__setitem__("cells_2", r.levels["cells_1"]))
    broken(lambda r: r.levels.__setitem__("bodies_1", r.attempts["bodies_1"][0]))
    broken(lambda r: r.attempts["senses_1"][-1].summary.__setitem__("group", "no"))
    broken(lambda r: r.attempts["cells_1"][0].params.__setitem__("gene_length", 320))
    # the language
    broken(lambda r: setattr(r, "language", None))
    broken(lambda r: r.language.languages[signals.STEPS].__setitem__("order", "property_first"))
    # the parameters
    broken(lambda r: r.params.__setitem__("cooling", 0.3))
    broken(lambda r: r.params.__setitem__("max_tries", 3))
    broken(lambda r: r.params.pop("atoms"))
    # a world that was stopped by the clock, with the clock moved
    assert worlds[3].summary["outcome"] == "out_of_time"
    broken(lambda r: r.era("star").__setitem__("star_end", 99982568000), seed=3)
    broken(lambda r: r.era("cells_1").__setitem__("duration", 4320000000), seed=3)
    # things that are no world
    assert not w.conserved(worlds[29].levels["planets"]).ok
    assert not w.conserved(w.WorldRollout("world7", 29, {}, [])).ok
    assert not w.conserved(w.WorldRollout("world", 29, dict(worlds[29].params), list(worlds[29].steps))).ok


def test_lines_only_for_the_world_a_seed_names(worlds):
    short = w.run(29, max_tries=1)
    assert w.conserved(short).ok and not w.canonical(short)
    assert all(s.get("tries", 1) == 1 for s in short.steps)
    assert rung_index(short) <= rung_index(worlds[29])
    with pytest.raises(ValueError):
        w.lines(short)
    # the gate's own copy is the seed's world, not the last one run
    assert w.check("world7 seed 2 9 final rung", "language").ok


# ---------------------------------------------------------------------------------------------
# what a world must look like


def test_one_universe(worlds):
    first = worlds[1].era("nucleo")
    assert first == {"era": "nucleo", "age": 0, "seconds": 3000, "hydrogen": 0.753, "helium": 0.247, "traces": 6.02e-05}
    for r in worlds.values():
        assert r.era("nucleo") == first
        assert r.levels["nucleo"].params["rules"] == 7 and r.levels["stars"].params["helium"] == 0.247


def test_the_clock(worlds):
    for seed, r in worlds.items():
        era = {s["era"]: s for s in r.steps}
        star = era["star"]
        assert era["stars"]["age"] == 250_000_000
        assert era["cloud"]["age"] == era["collapse"]["age"] == int(r.params["formation_time"] * 4) * 250_000_000
        assert star["age"] == era["cloud"]["age"] + 2_568_000
        assert star["star_end"] == star["age"] + star["lifetime"]
        ages = [s["age"] for s in r.steps if w._number(s["era"]) == 0]
        assert ages == sorted(ages)
        k = 0
        while f"surface_{k + 1}" in era:
            k += 1
            mine = r.planet(k)
            assert mine[0]["age"] == star["age"] + 50_000_000
            for before, after in zip(mine, mine[1:]):
                assert after["age"] == before["age"] + before.get("duration", 0)
            assert [s["age"] for s in mine[1:]] == sorted({s["age"] for s in mine[1:]}), "a planet's clock moves on"
            for s in mine[1:]:
                assert s["age"] + s["duration"] <= star["star_end"], (seed, s["era"])
                if "tries" in s:
                    assert 1 <= s["kept"] <= s["tries"] <= 6 and s["duration"] == s["tries"] * w.SPAN[w._base(s["era"])]
        assert r.summary["age_at_end"] <= max(star["star_end"], star["age"] + 50_000_000)
        if r.summary["outcome"] == "out_of_time":
            assert star["fusion"] == "yes" and r.summary["rung"] != "states"


def test_the_seeds_of_the_levels(worlds):
    for seed, r in worlds.items():
        for name in ("nucleo", "stars", "collapse", "planets"):
            assert name not in r.levels or r.levels[name].seed == seed
        for name, level in r.levels.items():
            k = w._number(name)
            if w._base(name) == "chemistry":
                assert level.seed == seed * 10 + k
        for name, tries in r.attempts.items():
            k = w._number(name)
            number = w.LEVEL_OF[w._base(name)].number
            assert [t.seed for t in tries] == [(seed * 10 + k) * 1000 + number * 10 + j
                                               for j in range(1, len(tries) + 1)]
            assert r.levels[name] is tries[r.era(name)["kept"] - 1]
        assert set(r.attempts) == {s["era"] for s in r.steps if "tries" in s}
    assert [level.number for level in w.LEVELS] == [5, 8, 9, 10, 11, 12]
    every = [t.seed for r in worlds.values() for tries in r.attempts.values() for t in tries]
    assert len(set(every)) == len(every), "no two tries of any level, planet or world share a seed"


def test_the_ladder_is_monotone(worlds):
    for seed, r in worlds.items():
        held = w.links_held(r.steps)
        assert tuple(held) == w.LINKS[:len(held)]
        assert r.summary["rung"] == w.LADDER[len(held)]
        assert r.summary["outcome"] in (r.summary["rung"], "out_of_time")
        assert w.LADDER.index(r.summary["by_now"]) <= rung_index(r)
        fixed = [s for s in r.steps if w._number(s["era"]) == 0]
        k = 0
        while r.era(f"surface_{k + 1}") is not None:
            k += 1
            mine = r.planet(k)
            rungs = [w.LADDER.index(w.rung_of(fixed + mine[:i])) for i in range(1, len(mine) + 1)]
            assert rungs == sorted(rungs), (seed, k, rungs)
            # a level runs only on the upper rung of the level below
            for below, level in zip(w.LEVELS, w.LEVELS[1:]):
                if r.era(f"{level.name}_{k}") is not None:
                    assert below.upper(r.era(f"{below.name}_{k}"))
        assert w.why_line(r).answer == (" ".join(held) or "nothing")


def test_the_census(worlds):
    assert set(w.CENSUS["rung"]) == set(w.LADDER), "every rung occurs in seeds 1 to 300, or is no rung"
    assert set(w.CENSUS["outcome"]) <= set(w.OUTCOMES) and w.CENSUS["outcome"]["out_of_time"] > 0
    assert set(w.CENSUS["by_now"]) <= set(w.LADDER)
    for counts in w.CENSUS.values():
        assert sum(counts.values()) == 300 and min(counts.values()) > 0
    for key in ("rung", "outcome", "by_now"):
        mine = collections.Counter(r.summary[key] for r in worlds.values())
        for word, n in mine.items():
            assert n <= w.CENSUS[key].get(word, 0), (key, word, n)
    # the rungs of seeds 1 to 40, pinned
    rungs = collections.Counter(r.summary["rung"] for r in worlds.values())
    assert dict(rungs) == RUNGS_OF_FORTY


def test_metals_only_after_the_first_stars(worlds):
    for seed, r in worlds.items():
        z = r.summary["metallicity"]
        assert (z == 0) == (r.params["formation_time"] == 0.25)
        assert 0 <= z < 0.035
        level = r.levels["stars"]
        assert abs(level.summary["hydrogen"] + level.summary["helium"] + level.summary["metallicity"] - 1) < 1e-9
    # the sun: efficiency 0.3, 9.25 Gyr after the Big Bang (9 Gyr after the first stars)
    sun = stars.run(1, efficiency=0.3, t_end=w.enrichment_years(9.25) / w.GYR, **w.first_gas(0.753, 0.247))
    assert abs(sun.summary["metallicity"] - 0.0135) < 0.00135
    ratio = (sun.summary["helium"] - 0.247) / sun.summary["metallicity"]
    assert 1.5 < ratio < 2.5
    assert stars.run(1, efficiency=0.6, t_end=0.0, **w.first_gas(0.753, 0.247)).summary["metallicity"] == 0


def test_a_cloud_before_the_first_metals_has_no_rocky_planets(worlds):
    assert worlds[31].params["formation_time"] == 0.25
    for r in (worlds[31], w.run(29, formation_time=0.25), w.run(3, formation_time=0.25)):
        drawn = r.params["cloud_mass"]
        assert r.summary["metallicity"] == 0 and r.summary["rung"] == r.summary["outcome"] == "no_metals"
        assert [s["era"] for s in r.steps] == list(w.FIXED_ERAS)
        assert "planets" not in r.levels and r.summary["planets"] == r.summary["temperate"] == 0
        assert abs(r.era("cloud")["cloud_mass"] - 90 * drawn) < 1e-9 and r.era("cloud")["cooling"] == 0.1
        assert r.era("star")["disc_solids"] == 0 and r.era("star")["disc_gas"] > 0
        assert r.era("star")["star_mass"] > 20 * drawn and r.era("star")["lifetime"] < 2_000_000_000
        assert w.conserved(r).ok
    # the same seeds with their own formation times have metals and planets
    assert worlds[29].summary["planets"] > 0 and worlds[3].summary["planets"] > 0


def test_a_massive_star_is_out_of_time_before_bodies(worlds):
    bodies_rung = w.LADDER.index("bodies")
    heavy = [r for r in worlds.values() if r.summary["star_mass"] >= 1.6 and r.summary["oceans"] > 0]
    assert len(heavy) >= 2
    for r in heavy:
        assert r.summary["outcome"] == "out_of_time" and rung_index(r) < bodies_rung, r.seed
        assert not any(s["era"].startswith(("bodies", "cells")) for s in r.steps)
    for r in worlds.values():
        if rung_index(r) >= bodies_rung:
            assert r.summary["star_mass"] < 1.4
        assert (r.era("star")["lifetime"] == 0) == (r.era("star")["fusion"] == "no")
    # seed 3: 1.36 solar masses, 3.98e9 years: one try at cells, which fails, and no time for a second
    r = worlds[3]
    cells_era, star = r.era("cells_1"), r.era("star")
    assert cells_era["tries"] == 1 and cells_era["result"] == "collapse" and r.summary["outcome"] == "out_of_time"
    assert r.summary["rung"] == "replicators"
    assert cells_era["age"] + 2 * w.SPAN["cells"] > star["star_end"] >= cells_era["age"] + w.SPAN["cells"]
    assert w.in_time(cells_era["age"] + w.SPAN["cells"], w.SPAN["cells"], star["star_end"]) == "no"
    # the textbook lifetimes
    assert w.lifetime_of(1.0, w.luminosity_of(1.0)) == 10_000_000_000
    assert w.lifetime_of(1.5, w.luminosity_of(1.5)) == 2_960_000_000
    assert w.lifetime_of(2.0, w.luminosity_of(2.0)) == 1_270_000_000
    assert w.lifetime_of(0.5, w.luminosity_of(0.5)) == 80_000_000_000


def test_a_world_in_forty_speaks(worlds, texts):
    signals_rung = w.LADDER.index("signals")
    speaking = [seed for seed, r in worlds.items() if rung_index(r) >= signals_rung]
    assert len(speaking) >= 1 and 29 in speaking
    for seed, r in worlds.items():
        lines = texts[seed]
        lexicon = [ln for ln in lines if ln.kind == "record" and ln.prompt.startswith(f"world7 seed {num(seed)} lexicon.")]
        assert bool(lexicon) == (seed in speaking) == (r.language is not None)
        if seed not in speaking:
            assert r.summary["sentence"] == r.summary["meaning"] == "none"
            assert not w.check(f"world7 seed {num(seed)} order", "none").ok
            continue
        k = r.summary["planet"]
        assert r.summary["words"] == r.era(f"signals_{k}")["words"] > 0
        said, meant = r.summary["sentence"], r.summary["meaning"]
        assert w.check(f"world7 seed {num(seed)} meaning {said}", meant).ok
        # what is said leads back to what is meant
        for ln in lines:
            head = f"world7 seed {num(seed)} say "
            if ln.prompt.startswith(head) and ln.answer != "none":
                assert w.check(f"world7 seed {num(seed)} meaning {ln.answer}", ln.prompt[len(head):]).ok
    assert worlds[29].summary["rung"] == "language" and worlds[29].era("society_1")["result"] == "bands"


def test_the_world_that_climbs_the_whole_ladder(showcase):
    r, lines = showcase
    text = {ln.text for ln in lines}
    assert r.summary["rung"] == r.summary["outcome"] == r.summary["by_now"] == "states"
    assert w.links_held(r.steps) == list(w.LINKS)
    assert [s["tries"] for s in r.steps if "tries" in s] == [1, 3, 1, 1, 2, 6]
    assert r.era("star")["star_mass"] == 0.0827 and r.era("star")["lifetime"] == 1_110_000_000_000
    assert r.summary["age_at_end"] == 12_562_688_000 < w.NOW
    assert "world7 seed 8 5 lexicon. fire na. sky_danger le pe. predator ma. young bu la. small me pa pa. " \
           "far lu pe. near ma su. big me ma." in text
    assert "world7 seed 8 5 grammar. order property_first. compositional 1." in text
    assert w.check("world7 seed 8 5 say predator small", "me pa pa ma").ok
    assert w.check("world7 seed 8 5 meaning me pa pa ma", "predator small").ok
    assert w.check("world7 seed 8 5 word predator", "ma").ok and w.check("world7 seed 8 5 order", "property_first").ok
    assert not w.check("world7 seed 8 5 say predator small", "ma me pa pa").ok
    assert w.check("world7 seed 8 5 word water", "none").ok and w.check("world7 seed 8 5 meaning ka", "none").ok
    society_era = r.era("society_1")
    assert society_era["technology_list"] == ("fire stone_tools clothing boats pottery farming metal writing "
                                              "mathematics printing")
    assert r.summary["technologies"] == len(society_era["technology_list"].split()) == 10
    assert society_era["first_farming"] == 2500 < society_era["first_writing"] == 4750
    for ln in lines:
        if ln.kind != "record":
            assert w.check(ln.prompt, ln.answer).ok and not w.check(ln.prompt, altered(ln.answer)).ok, ln.text
        assert is_dense(ln.text) and w.tokens(ln.text) <= 128


def test_stars_discs_and_planets(worlds):
    shining = [r for r in worlds.values() if r.era("star")["fusion"] == "yes"]
    assert 30 <= len(shining) < 40, "some clumps are brown dwarfs"
    for r in worlds.values():
        cloud, collapse, star = r.era("cloud"), r.era("collapse"), r.era("star")
        assert 0.25 <= collapse["star_share"] <= 0.8 and 0.05 <= collapse["disc_share"] <= 0.6
        assert star["star_mass"] == w.star_mass_of(cloud["cloud_mass"], collapse["star_share"])
        if star["fusion"] == "no":
            assert star["star_mass"] < 0.08 and r.summary["rung"] in ("no_star", "no_metals")
            assert star["luminosity"] == star["frost_line"] == star["hz_outer"] == 0 and "planets" not in r.levels
            continue
        assert star["hz_inner"] < star["hz_outer"] < star["frost_line"]
        # the disc holds 1 to 12 percent of the star's mass
        share = (star["disc_solids"] + star["disc_gas"]) / 333000 / star["star_mass"]
        assert 0.008 < share < 0.13, (r.seed, share)
        if "planets" in r.levels:
            level = r.levels["planets"]
            assert level.params["star_mass"] == star["star_mass"] and level.params["disc_mass"] == star["disc_solids"]
            assert level.params["gas_mass"] == star["disc_gas"] and level.params["rules"] == 7
            assert 0 <= r.era("planets")["planets"] <= 10
    masses = sorted(r.summary["star_mass"] for r in shining)
    assert masses[0] < 0.2 and masses[-1] > 3
    assert w.luminosity_of(1.0) == 1.0 and w.luminosity_of(0.079) == 0.0 and w.fusion_of(0.08) == "yes"


def test_surfaces(worlds):
    kinds = collections.Counter()
    for r in worlds.values():
        star = r.era("star")
        surfaces = [s for s in r.steps if s["era"].startswith("surface_")]
        assert len(surfaces) == r.summary["temperate"] == (r.era("planets") or {"temperate": 0})["temperate"]
        orbits = [s["orbit"] for s in surfaces]
        assert orbits == sorted(orbits)
        for s in surfaces:
            kinds[s["surface"]] += 1
            assert 0.356 <= s["flux"] <= 1.107 and s["flux"] == w.flux_of(star["luminosity"], s["orbit"])
            if s["surface"] == "terran":
                assert 273 <= s["temperature"] <= 294 and s["water"] == "liquid" and 33 <= s["greenhouse"] <= 76.4
                assert 0.3 <= s["mass"] <= 10 and s["gas"] == 0
            elif s["surface"] == "thin":
                assert s["mass"] < 0.3 and s["greenhouse"] == 5 and s["temperature"] < 273 and s["water"] == "ice"
            else:
                assert s["water"] == "none" and s["greenhouse"] == 0
            k = int(s["era"].split("_")[1])
            assert (r.era(f"chemistry_{k}") is not None) == (w.ocean(s) and
                                                             s["age"] + w.SPAN["chemistry"] <= star["star_end"])
    assert kinds["terran"] >= 15 and kinds["thin"] >= 1
    # the earth and Mars by the chain's rules
    assert w.temperature_of(1.0, w.greenhouse_of(1.0, 1.0, 0.0)) == 288 and w.t_eq_of(1.0) == 255.0
    mars = w.flux_of(1.0, 1.524)
    assert w.surface_of(0.107, 0.0) == "thin" and w.temperature_of(mars, w.greenhouse_of(mars, 0.107, 0.0)) == 211
    assert w.water_of(211, 0.107, 0.0) == "ice" and w.water_of(288, 1.0, 0.0) == "liquid"
    assert w.water_of(288, 4.0, 0.5) == "none" and w.water_of(400, 1.0, 0.0) == "vapor"
    # the edges of the zone are the edges of liquid water for a terran planet
    assert w.temperature_of(0.356, w.greenhouse_of(0.356, 1.0, 0.0)) == 273
    assert w.temperature_of(1.107, w.greenhouse_of(1.107, 1.0, 0.0)) == 294
    assert w.zone_of(0.355) == "frozen" and w.zone_of(1.108) == "runaway" and w.zone_of(1.0) == "temperate"


def test_the_climb(worlds):
    results = collections.defaultdict(collections.Counter)
    for r in worlds.values():
        for s in r.steps:
            base = w._base(s["era"])
            if "tries" in s:
                results[base][s["result"]] += 1
            if base == "chemistry":
                k = int(s["era"].split("_")[1])
                level = r.levels[s["era"]]
                assert level.params["t_end"] == r.era(f"surface_{k}")["temperature"] and level.params["t_start"] == 4600
                assert level.params["mix"] in ("ocean", "reducing") and level.summary["water"] == "liquid"
                assert s["h2o"] > 4000 and 0 < s["precursors"] <= s["organic"]
            if base == "replicators":
                assert s["monomers"] == 100 * r.era(s["era"].replace("replicators", "chemistry"))["precursors"]
                assert s["inflow"] == 20 * r.era(s["era"].replace("replicators", "chemistry"))["sparks"]
            if base == "cells":
                assert s["fidelity"] == round(1 - r.era(s["era"].replace("cells", "replicators"))["mu"], 6)
                assert s["gene_length"] in w.GENE_LENGTHS and (s["gene_length"] == 20 or 2 * s["gene_length"] <= s["l_max"])
            if base == "senses":
                assert s["body_size"] >= 2 and 0 <= s["brightness"] <= 1 and 0 <= s["air"] <= 1
            if base == "signals":
                below = r.era(s["era"].replace("signals", "senses"))
                assert below["group"] == "yes" and s["band"] == min(60, below["agents"])
                assert s["capacity"] == s["brain"] // 8 and s["brain"] == round(below["neurons"])
    assert results["replicators"] and results["cells"]["complex_cells"] >= 3 and results["bodies"]["animals_like"] >= 3
    assert sum(results["senses"].values()) >= 3 and sum(results["signals"].values()) >= 1


def test_a_world_runs_in_seconds(worlds):
    slowest = max(TIMES, key=TIMES.get)
    if TIMES[slowest] >= 12:  # a busy machine: the better of two runs counts
        start = time.perf_counter()
        w.run(slowest)
        TIMES[slowest] = min(TIMES[slowest], time.perf_counter() - start)
    assert max(TIMES.values()) < 12, TIMES
    # alone: median 1.5 s, mean 2.0 s, 0.95 to 5.9 s (the module docstring); a busy machine gets room
    assert sorted(TIMES.values())[len(TIMES) // 2] < 4


# ---------------------------------------------------------------------------------------------
# the review: formulas written again from the docstring, foreign prompts, rare rungs, the clock


def _decimal(x):
    from decimal import Decimal
    return Decimal(x) if isinstance(x, int) else Decimal(repr(float(x)))


def _half_up(d, digits=3):
    """An exact decimal to 3 significant digits, a 5 rounding up."""
    from decimal import ROUND_HALF_UP, Decimal
    return 0.0 if d == 0 else float(d.quantize(Decimal((0, (1,), d.adjusted() - digits + 1)), rounding=ROUND_HALF_UP))


def _times(*xs):
    from decimal import Decimal
    out = Decimal(1)
    for x in xs:
        out *= _decimal(x)
    return out


def _s3(x):
    return 0.0 if x == 0 else float(f"{x:.3g}")


def _kind(m, g):
    return "thin" if m < 0.3 else ("enveloped" if g > 0 or m > 10 else "terran")


def _warming(f, m, g):
    k = _kind(m, g)
    if k != "terran":
        return 5.0 if k == "thin" else 0.0
    return 500.0 if f > 1.107 else 33.0 if f >= 1 else _s3(33 + 67.3 * (1 - f)) if f >= 0.356 else 0.0


def _light(m):
    if m < 0.08:
        return 0.0
    return _s3(0.23 * m ** 2.3 if m < 0.43 else m ** 4 if m < 2 else 1.4 * m ** 3.5 if m < 55 else 32000 * m)


def _hydrolysis(t):
    return next((v for edge, v in ((320, 0.1), (305, 0.05), (290, 0.02), (275, 0.01), (260, 0.005)) if t >= edge),
                0.002)


def _genes(d, l_max):
    while d > 20 and 2 * d > l_max:
        d //= 2
    return d


def _transfer(e):
    import math
    from decimal import ROUND_HALF_UP, Decimal
    v = min(0.2, max(0.05, 0.05 + 0.15 * math.log(e / 10) / math.log(16)))
    return float(Decimal(repr(v)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


#: every hand-off rule written again from the module docstring and the rule descriptions, not
#: from the module's code
FORMULAS = {
    "helium_share": lambda h, he: round(he / (h + he), 4),
    "formation_year": lambda t: int(t * 4) * 250_000_000,
    "enrichment_years": lambda t: int(t * 4) * 250_000_000 - 250_000_000,
    "cooling": lambda z, d: 0.1 if z < 1e-5 else d,
    "cloud_mass": lambda d, z: float(_times(90, d)) if z < 1e-5 else d,
    "star_mass": lambda c, s: _half_up(_times(c, s)),
    "fusion": lambda m: "yes" if m >= 0.08 else "no",
    "luminosity": _light,
    "lifetime": lambda m, lum: int(_half_up(_times(10 ** 10, m) / _decimal(lum))),
    "frost_line": lambda lum: _s3(2.7 * lum ** 0.5),
    "hz_inner": lambda lum: _s3((lum / 1.107) ** 0.5),
    "hz_outer": lambda lum: _s3((lum / 0.356) ** 0.5),
    "disc_solids": lambda ds, c, z: _half_up(_times(26640, ds, c, z)),
    "disc_gas": lambda ds, c, z: _half_up(_times(26640, ds, c) * (1 - _decimal(z))),
    "flux": lambda lum, a: _s3(lum / a / a),
    "zone": lambda f: "runaway" if f > 1.107 else ("frozen" if f < 0.356 else "temperate"),
    "surface": _kind,
    "t_eq": lambda f: _s3(254.6 * f ** 0.25),
    "greenhouse": _warming,
    "temperature": lambda f, g: int(254.6 * f ** 0.25 + g + 0.5),
    "water": lambda t, m, g: ("none" if _kind(m, g) == "enveloped" else "ice" if t < 273 else "liquid" if t <= 373
                              else "vapor"),
    "monomers": lambda p: 100 * p,
    "inflow": lambda s: 20 * s,
    "hydrolysis": _hydrolysis,
    "fidelity": lambda mu: max(0.5, 1 - mu),
    "nutrient": lambda i: 12500 * i,
    "gene_length": _genes,
    "efficiency": _transfer,
    "light": lambda f: float(int(_times(200, f) + _decimal(0.5))),
    "brightness": lambda f: min(1.0, f),
    "body_size": lambda c, m: c if c > 0 else max(1, m),
    "predators": lambda p, t: 0.0 if t < 3 else _s3(min(0.3, 3 * p)),
    "neurons": lambda n: int(round(n)),
    "band": lambda a: min(60, a),
    "duration": lambda t, s: t * s,
    "era_end": lambda a, d: a + d,
    "in_time": lambda a, s, e: "yes" if a + s <= e else "no",
}


def _agrees(prompt, value):
    return w.LESSONS.check(prompt, value if isinstance(value, str) else num(value)).ok


def test_lessons_agree_with_formulas_written_from_the_docstring():
    assert set(FORMULAS) == set(w.RULES)
    seen = collections.Counter()
    for seed in (1, 2, 3, 4):
        for ln in w.LESSONS.generate(random.Random(seed), 500):
            assert w.LESSONS.check(ln.prompt, ln.answer).ok
            name, values = w.LESSONS.parse(ln.prompt)
            seen[name] += 1
            assert _agrees(ln.prompt, FORMULAS[name](*values)), (ln.text, FORMULAS[name](*values))
    assert set(seen) == set(w.RULES) and min(seen.values()) >= 25
    # a lesson is the chain's own function, which the chain calls through the module by its name
    source = open(w.__file__, encoding="utf-8").read()
    for name, rule in w.RULES.items():
        fn = rule.compute.__name__
        assert getattr(w, fn) is rule.compute
        assert source.count(f"{fn}(") - source.count(f"def {fn}(") >= 1, (name, fn)


def test_yes_no_lessons_are_near_balance():
    by_rule = collections.defaultdict(collections.Counter)
    for ln in w.LESSONS.generate(random.Random(11), 8000):
        by_rule[ln.prompt.split()[2]][ln.answer] += 1
    for name in ("fusion", "in_time"):
        share = by_rule[name]["yes"] / sum(by_rule[name].values())
        assert 0.4 <= share <= 0.65, (name, share)
    for name, answers in by_rule.items():
        assert max(answers.values()) <= 0.6 * sum(answers.values()), (name, answers.most_common(2))


def test_the_hand_offs_of_forty_worlds_by_the_written_formulas(worlds):
    counted = 0
    for seed, r in worlds.items():
        for name, inputs, printed in w.handoffs(r):
            prompt = w.LESSONS.line(name, list(inputs)).prompt
            assert _agrees(prompt, FORMULAS[name](*inputs)), (seed, name, inputs)
            assert _agrees(prompt, printed), (seed, name, inputs, printed)
            counted += 1
    assert counted > 600


def test_owns_no_round_5_or_other_level_prompt():
    from haishool.cosmos import predict
    from haishool.truth.loop import all_gates
    foreign = []
    for gate in all_gates():
        foreign += [ln.prompt for ln in gate.generate(random.Random(3), 30)]
    for topic in predict.TOPICS:
        foreign += [ln.prompt for ln in predict.gate(topic).generate(random.Random(3), 30)]
    for module in (nucleo, gravity, chem, life, stars, bodies):
        foreign += [ln.prompt for ln in module.LESSONS.generate(random.Random(3), 30)]
    foreign += [ln.prompt for ln in stars.lines(stars.run(3)) if ln.kind != "record"][:200]
    assert len(foreign) > 500
    for prompt in foreign:
        assert not w.owns(prompt), prompt
        assert not w.check(prompt, "1").ok


#: a world for every rung that seeds 1 to 40 and 85 do not reach: no rung is a word no world earns
RARE_RUNGS = {"no_planets": 176, "no_organics": 205, "complex_cells": 219, "senses": 65, "society": 207}


def test_every_rung_is_reached(worlds, showcase):
    reached = {r.summary["rung"] for r in worlds.values()} | {showcase[0].summary["rung"]}
    assert set(w.LADDER) - reached == set(RARE_RUNGS)
    for rung, seed in RARE_RUNGS.items():
        r = w.run(seed)
        assert r.summary["rung"] == rung and w.conserved(r).ok, (rung, seed, r.summary["rung"])
        assert w.CENSUS["rung"][rung] >= 1


def test_no_water_is_the_planets_not_two_zones(worlds):
    """``no_water`` is not two rules disagreeing about the zone: level 3's own habitable count is
    the number of terran surfaces, and a terran planet in the zone always has liquid water."""
    systems = 0
    for seed, r in worlds.items():
        if "planets" not in r.levels:
            continue
        systems += 1
        surfaces = [s for s in r.steps if w._base(s["era"]) == "surface"]
        star = r.era("star")
        assert r.levels["planets"].summary["habitable"] == sum(s["surface"] == "terran" for s in surfaces), seed
        for s in surfaces:
            assert star["hz_inner"] * 0.99 <= s["orbit"] <= star["hz_outer"] * 1.01
            assert (s["water"] == "liquid") == (s["surface"] == "terran")
        if r.summary["rung"] == "no_water":
            assert surfaces and all(s["surface"] in ("thin", "enveloped") for s in surfaces)
    assert systems >= 20
    for flux in (0.356, 0.5, 0.75, 1.0, 1.107):
        kelvin = w.temperature_of(flux, w.greenhouse_of(flux, 1.0, 0.0))
        assert w.water_of(kelvin, 1.0, 0.0) == "liquid" and w.zone_of(flux) == "temperate"


def test_the_clock_has_the_real_orders_of_magnitude(worlds):
    """The physical eras within a few times of the real durations; the biological spans are toy
    rates calibrated on the earth and add up to its age."""
    assert 3 * 60 <= worlds[1].era("nucleo")["seconds"] <= 3600  # the light elements: the first minutes
    assert 10 ** 8 <= w.FIRST_STARS <= 3 * 10 ** 8  # the first stars: some 10^8 years
    assert 10 ** 5 <= w.SPAN["collapse"] <= 10 ** 7  # a core collapses in 10^5 to 10^6 years
    assert 10 ** 6 <= w.SPAN["planets"] <= 10 ** 8  # planets form in 10^6 to 10^8 years
    assert abs(sum(w.SPAN.values()) - 4.54e9) < 0.01 * 4.54e9  # the earth's age: toy calibration
    for text in ("**toy rates**", "nobody knows", "no one knows"):
        assert text in w.__doc__
    # the textbook lifetimes, within a factor of two of real stars (from memory)
    for mass, real in ((1.0, 1e10), (1.35, 3e9), (2.0, 1.2e9), (5.0, 1e8)):
        assert real / 2 <= w.lifetime_of(mass, w.luminosity_of(mass)) <= real * 2, mass


def test_worlds_with_drawn_parameters_are_deterministic():
    """Worlds that are not a seed's own (parameters from another stream, two tries a level):
    the same parameters give the same eras and levels, and the gate holds."""
    for seed in (2, 6, 9, 13):
        p = w.random_params(random.Random(10_000 + seed))
        a, b = w.run(seed, max_tries=2, **p), w.run(seed, max_tries=2, **p)
        assert a.steps == b.steps and a.summary == b.summary and a.params == b.params
        assert sorted(a.levels) == sorted(b.levels)
        for name in a.levels:
            assert a.levels[name].steps == b.levels[name].steps and a.levels[name].summary == b.levels[name].summary
        assert w.conserved(a).ok and not w.canonical(a)
