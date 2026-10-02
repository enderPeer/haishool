import collections
import copy
import doctest
import importlib
import math
import random
import statistics
import time
from functools import lru_cache

import pytest

from haishool.cosmos import Rollout, Simulation, close
from haishool.evo import MAX_TOKENS, LessonGate, sig
from haishool.evo import senses
from haishool.evo.senses import (CAP, COUNT_KEYS, GENS_PER_STEP, INPUT_KEYS, KEYS, LEDGER_KEYS, LESSONS, OUTCOMES,
                                 PARAM_KEYS, RULES, SENSORS, STAGES, START, STATE_KEYS, STEPS, SUMMARY_KEYS,
                                 WORD_KEYS, Senses, SensesLessons, dense_value, handoff_in, lesson_gate, lines,
                                 param_value, params_line, random_params, simulation, traits)
from haishool.truth import Gate, is_dense, num, parse_num, split_line

SEEDS = list(range(1, 41))
ENERGY = ("e_sensors", "e_neurons", "e_movement", "e_reproduction", "e_stored", "e_eaten")
#: rule name -> the function of the simulation that answers it
FUNCTIONS = {
    "detection_area": senses.detection_area, "pooling_gain": senses.pooling_gain,
    "sensor_range": senses.sensor_range, "encounter": senses.encounter_rate, "intake": senses.food_intake,
    "energy_yield": senses.energy_yield, "sensor_cost": senses.sensor_cost,
    "movement_cost": senses.movement_cost, "break_even": senses.break_even_level,
    "escape": senses.escape_probability, "attack": senses.attack_probability,
    "survival": senses.survival_probability, "energy_budget": senses.net_energy,
    "offspring": senses.expected_offspring, "births": senses.births_from,
    "neurons_needed": senses.neurons_needed, "sensor_use": senses.sensor_use, "max_neurons": senses.max_neurons,
    "max_level": senses.max_sensor_level, "predator_step": senses.predator_step,
    "group_detection": senses.group_detection, "mate_found": senses.mate_found, "kin_gain": senses.kin_gain,
    "stage": senses.stage_of,
}


def _hand_neurons_needed(associations):
    need = 2 * (associations - 1)  # one association is free, every further one needs two neurons
    return 0 if need <= 0 else max(2, 1 << (need - 1).bit_length())


def _hand_sensor_use(neurons, associations):
    return min(1.0, (1 + neurons // 2) / associations)


def _hand_encounter(speed, density, patchiness, cells):
    return speed * density * (1 - patchiness) * cells * min(1 / (1 - patchiness), cells)


def _hand_max_neurons(body_size, cell_types):
    quarter = body_size // 4
    if cell_types < 3 or quarter < 2:
        return 0
    return min(1024, 1 << (quarter.bit_length() - 1))


def _hand_break_even(speed, density, patchiness, light, cost, dims):
    best, best_net = 0, None
    for level in range(6):
        extra = (2 * level + 1) ** dims - 1  # an eye of this level sees level cells far
        neurons = min(1024, _hand_neurons_needed(extra))
        cells = 1 + (_hand_sensor_use(neurons, extra) if extra else 1.0) * light * extra
        rate = _hand_encounter(speed, density, patchiness, cells)
        eye = cost * 2 * (level * level + 3) if level else 0.0
        net = 100 * rate / (1 + 0.5 * rate) - eye - cost * 0.05 * neurons
        if best_net is None or net > best_net + 1e-9:
            best, best_net = level, net
    return best


def _hand_predator_step(sense, escape, prey_share):
    def meal(level):
        return (2 * level + 1) * (1.0 - escape) * prey_share

    if meal(sense) < 1.5:
        return min(5, sense + 1)
    if sense > 0 and meal(sense - 1) >= 1.5:
        return sense - 1
    return sense


def _hand_stage(population, predator_sense, neurons, eyed, sensing):
    if population == 0:
        return "starved"
    if predator_sense >= 3:
        return "arms_race"
    if neurons >= 32:
        return "brained"
    if eyed >= 0.5:
        return "seeing"
    return "smelling" if sensing >= 0.5 else "blind"


#: rule name -> the rule as the module docstring states it, written out here a second time
BY_HAND = {
    "detection_area": lambda range_, dims: (2 * range_ + 1) ** dims,
    "pooling_gain": lambda receptors: receptors ** 0.5,
    "sensor_range": lambda level, reach: reach * level,
    "encounter": _hand_encounter,
    "intake": lambda rate, yield_: yield_ * rate / (1 + rate / 2),
    "energy_yield": lambda oxygen: min(100.0, 6.25 + 93.75 * oxygen / 0.05),
    "sensor_cost": lambda level, unit: unit * (level ** 2 + 3) if level else 0.0,
    "movement_cost": lambda speed: speed ** 3 / 2,
    "break_even": _hand_break_even,
    "escape": lambda head_start, speed, predator_speed: min(1.0, head_start * speed / (4 * (predator_speed - speed))),
    "attack": lambda predators, predator_sense: 1 - math.exp(-predators * (2 * predator_sense + 1)),
    "survival": lambda attack, escape: 1 - attack * (1 - escape),
    "energy_budget": lambda intake, sensors, neurons, movement: max(0.0, intake - sensors - neurons - movement),
    "offspring": lambda fitness, total_fitness, births: births * fitness / total_fitness,
    "births": lambda surplus: min(200, math.floor(surplus * 2)),
    "neurons_needed": _hand_neurons_needed,
    "sensor_use": _hand_sensor_use,
    "max_neurons": _hand_max_neurons,
    "max_level": lambda cell_types: 5 if cell_types >= 2 else 0,
    "predator_step": _hand_predator_step,
    "group_detection": lambda p, n: 1 - (1 - p) ** n,
    "mate_found": lambda voice, hearing, voice_share, mean_hearing: (voice * mean_hearing + hearing * voice_share) / 2,
    "kin_gain": lambda attack, escape, hearing, callers: attack * (1 - escape) * hearing * (1 - (1 - callers) ** 5),
    "stage": _hand_stage,
}
#: a world in which every sense can pay: clumped food, today's air, a body with cells and cell types enough
WORLD = dict(patchiness=0.6, density=0.2, cost=1.0, mutation=0.02, oxygen=1.0, body_size=5000, cell_types=6)


def n_tokens(text: str) -> int:
    """As ``haishool.student.tokens`` counts them (that module needs torch, so not imported)."""
    return len(text.replace(".", " . ").split())


@pytest.fixture(scope="module")
def sim():
    return Senses()


@pytest.fixture(scope="module")
def rollouts(sim):
    return {seed: sim.rollout(seed) for seed in SEEDS}


@pytest.fixture(scope="module")
def all_lines(rollouts):
    return [ln for r in rollouts.values() for ln in lines(r)]


@pytest.fixture(scope="module")
def lessons():
    return LESSONS.generate(random.Random(1), 400)


@lru_cache(maxsize=None)
def _final(seed: int, items: tuple) -> tuple:
    r = simulation().run(seed, **dict(items))
    return tuple(sorted({**r.steps[-1], **{"final_" + k: v for k, v in r.summary.items()}}.items()))


def finals(n: int = 8, **params) -> list[dict]:
    """The last step (and the summary, keys ``final_...``) of seeds 1..n in a hand-set world."""
    return [dict(_final(seed, tuple(sorted({**WORLD, **params}.items())))) for seed in range(1, n + 1)]


def mean(rows: list[dict], key: str) -> float:
    return statistics.mean(row[key] for row in rows)


# ---------------------------------------------------------------------------------------------
# the level as a simulation: protocol, determinism, lines


def test_protocols_and_module_api():
    s = simulation()
    assert isinstance(s, Simulation) and isinstance(s, Gate)
    assert s.sim == s.topic == senses.SIM == "senses" and s.records() == [] and simulation() is s
    assert lesson_gate() is LESSONS and isinstance(LESSONS, LessonGate) and LESSONS.topic == "predict_senses"
    assert type(LESSONS) is SensesLessons and LESSONS.rules is RULES and isinstance(LESSONS, Gate)
    for name in ("SIM", "KEYS", "simulation", "random_params", "run", "conserved", "lines", "check", "owns",
                 "LESSONS", "lesson_gate", "handoff_in"):
        assert hasattr(senses, name), name
    for key in PARAM_KEYS + STATE_KEYS + LEDGER_KEYS + SUMMARY_KEYS:
        assert key in KEYS, key
    assert COUNT_KEYS | WORD_KEYS <= set(KEYS) and set(INPUT_KEYS) <= set(PARAM_KEYS)
    assert STATE_KEYS == ("population", "smell", "eyes", "touch", "ears", "eyed", "neurons", "voice", "speed",
                          "intake", "survival", "predator_sense", "stage")
    assert STAGES == ("blind", "smelling", "seeing", "brained", "arms_race", "starved")
    assert {"stage", "eyes", "ears", "neurons", "voice", "survival", "outcome"} <= set(SUMMARY_KEYS)
    assert len(RULES) >= 8 and set(RULES) == set(FUNCTIONS) == set(BY_HAND)
    assert doctest.testmod(senses).failed == 0


def test_same_seed_same_lines():
    a, b = Senses(), Senses()
    for seed in (1, 7, 4321):
        ra, rb = a.rollout(seed), b.rollout(seed)
        assert ra.steps == rb.steps and ra.summary == rb.summary and ra.params == rb.params
        assert ra.genomes == rb.genomes and ra.generations == rb.generations
        assert [ln.text for ln in lines(ra)] == [ln.text for ln in lines(rb)]
    assert [ln.text for ln in lines(a.rollout(1))] != [ln.text for ln in lines(a.rollout(7))]
    assert a.run(5, light=0.9).steps == b.run(5, light=0.9).steps == senses.run(5, light=0.9, rules=7).steps
    assert a.run(5, light=0.9).steps != a.run(6, light=0.9).steps


def test_forty_seeds_replay_exactly(rollouts):
    """Two runs of a seed give the same records, genomes and summary, to the last bit."""
    fresh = Senses()
    for seed, r in rollouts.items():
        again = fresh.run(seed, **random_params(random.Random(seed)))
        assert again.steps == r.steps and again.generations == r.generations and again.genomes == r.genomes
        assert again.summary == r.summary and again.params == r.params
        assert [repr(v) for s in again.generations for v in s.values()] == \
               [repr(v) for s in r.generations for v in s.values()]


def test_values_are_plain_python_numbers_and_words(rollouts):
    for r in rollouts.values():
        for row in r.generations + [r.summary, r.params]:
            for key, value in row.items():
                assert type(value) in (float, int, str), (key, type(value))
                assert (type(value) is int) == (key in COUNT_KEYS), key
                assert (type(value) is str) == (key in WORD_KEYS), key
                if type(value) is float:
                    assert math.isfinite(value) and value >= 0, (key, value)
        assert len(r.steps) == STEPS == 60 and len(r.generations) == STEPS * GENS_PER_STEP == 300
        assert all(r.steps[t] is r.generations[GENS_PER_STEP * t] for t in range(STEPS))
        assert all(list(s) == list(STATE_KEYS + LEDGER_KEYS) for s in r.generations)
        assert list(r.summary) == list(SUMMARY_KEYS) and list(r.params) == list(PARAM_KEYS)
        for pop in r.genomes:
            assert all(type(g) is tuple and len(g) == 7 and all(type(x) is int for x in g) for g in pop)


def test_generate_is_deterministic(sim):
    one = [ln.text for ln in sim.generate(random.Random(11), 2)]
    two = [ln.text for ln in Senses().generate(random.Random(11), 2)]
    assert one == two and len(one) == 2 * 1621
    assert one != [ln.text for ln in sim.generate(random.Random(12), 2)]


def test_pinned_lines_of_seed_35(sim):
    """The lines the module docstring shows, word for word."""
    r = sim.rollout(35)
    texts = [ln.text for ln in lines(r)]
    assert texts[0] == ("senses seed 3 5 params. light 0 point 7 6. patchiness 0 point 2 3. density 0 point 4 9. "
                        "predators 0 point 2 6. cost 1 point 3 5. mutation 0 point 0 2 4. dims 2. "
                        "oxygen 0 point 2 0 2. body_size 6 5 9 1. cell_types 1 0. max_level 5. "
                        "max_neurons 1 0 2 4. eye_pays 2.")
    assert texts[1] == ("senses seed 3 5 step 0. population 1 0 0. smell 0. eyes 0. touch 0. ears 0. eyed 0. "
                        "neurons 0. voice 0. speed 1. intake 3 1 point 7. survival 0 point 7 7 1. "
                        "predator_sense 0. stage blind.")
    assert ("senses seed 3 5 step 4 0. population 2 0 0. smell 0 point 1. eyes 1 point 1 3. touch 0 point 0 9. "
            "ears 1 point 9 6. eyed 0 point 9 7. neurons 8 5 point 6. voice 0 point 5. speed 4. intake 1 7 4. "
            "survival 0 point 9 3 3. predator_sense 5. stage arms_race.") in texts
    for text in ("q senses seed 3 5 step 4 0 eyed. a 0 point 9 7.",
                 "q senses seed 3 5 step 4 0 next neurons. a 8 5 point 3.",
                 "q senses seed 3 5 step 4 0 next stage. a arms_race.",
                 "q senses seed 3 5 final stage. a arms_race.", "q senses seed 3 5 final population. a 2 0 0.",
                 "q senses seed 3 5 final eyes. a 1 point 0 8.", "q senses seed 3 5 final ears. a 2 point 0 6.",
                 "q senses seed 3 5 final neurons. a 7 4 point 9.", "q senses seed 3 5 final voice. a 0 point 5 3.",
                 "q senses seed 3 5 final eared. a 1.", "q senses seed 3 5 final talkers. a 0 point 5 3.",
                 "q senses seed 3 5 final survival. a 0 point 9 5.",
                 "q senses seed 3 5 final predator_pressure. a 0 point 9 4 3.",
                 "q senses seed 3 5 final kin_gain. a 0 point 0 1 9 1.", "q senses seed 3 5 final group. a yes.",
                 "q senses seed 3 5 final outcome. a brained."):
        assert text in texts, text
    changes = [(t, s["stage"]) for t, s in enumerate(r.steps) if t == 0 or s["stage"] != r.steps[t - 1]["stage"]]
    assert changes == [(0, "blind"), (2, "seeing"), (17, "brained"), (23, "arms_race")]
    assert LESSONS.check("senses predict escape head_start 3 speed 2 predator_speed 6", "0 point 3 7 5").ok
    assert random_params(random.Random(35)) == {
        "light": 0.76, "patchiness": 0.23, "density": 0.49, "predators": 0.26, "cost": 1.35, "mutation": 0.024,
        "dims": 2, "oxygen": 0.202, "body_size": 6591, "cell_types": 10}
    assert max(n_tokens(text) for text in texts) == 84


def test_random_params_stay_in_range():
    dark = hunted = rings = 0
    for seed in range(400):
        p = random_params(random.Random(seed))
        assert p == random_params(random.Random(seed)) == random_params(random.Random(seed), rules=7)
        assert list(p) == list(INPUT_KEYS)
        assert p["light"] == 0 or 0.05 <= p["light"] <= 1
        assert 0 <= p["patchiness"] <= 0.8 and 0.1 <= p["density"] <= 0.5
        assert p["predators"] == 0 or 0.02 <= p["predators"] <= 0.3
        assert 0.5 <= p["cost"] <= 2 and 0.005 <= p["mutation"] <= 0.03 and p["dims"] in (1, 2)
        assert 0.01 <= p["oxygen"] <= 1 and 10 <= p["body_size"] <= 31623 and 1 <= p["cell_types"] <= 12
        assert type(p["dims"]) is type(p["body_size"]) is type(p["cell_types"]) is int
        for key in ("light", "patchiness", "density", "predators", "cost"):
            assert p[key] == round(p[key], 2)
        assert p["mutation"] == round(p["mutation"], 3) and p["oxygen"] == round(p["oxygen"], 3)
        dark += p["light"] == 0
        hunted += p["predators"] > 0
        rings += p["dims"] == 1
    assert 30 <= dark <= 90 and 280 <= hunted <= 350 and 160 <= rings <= 240


def test_run_rejects_impossible_parameters(sim):
    for bad in ({"light": 1.5}, {"light": -0.1}, {"patchiness": 1.0}, {"density": -1.0}, {"predators": -0.1},
                {"cost": float("nan")}, {"mutation": 2.0}, {"oxygen": 1.2}, {"oxygen": float("inf")}, {"dims": 3},
                {"dims": 0}, {"body_size": 0}, {"body_size": 10.5}, {"cell_types": 0}, {"cell_types": True},
                {"rules": 6}):
        with pytest.raises(ValueError):
            sim.run(1, **bad)
    with pytest.raises(TypeError):
        sim.run(1, clumps=3)
    still = sim.run(1, mutation=0.0)
    assert sim.conserved(still).ok and still.summary["outcome"] == "blind"
    assert all(g == (0, 0, 0, 0, 0, 1, 0) for pop in still.genomes for g in pop), "without mutation nothing changes"


# ---------------------------------------------------------------------------------------------
# lines and the gate of the rollouts


def test_every_line_is_dense_and_short(all_lines):
    assert len(all_lines) == 40 * 1621
    longest = 0
    for ln in all_lines:
        assert is_dense(ln.text), ln.text
        assert n_tokens(ln.text) <= MAX_TOKENS == 128, ln.text
        longest = max(longest, n_tokens(ln.text))
        assert ln.topic == "senses"
        if ln.kind != "record":
            assert "." not in ln.prompt and "." not in ln.answer
            assert split_line(ln.text) == (ln.prompt, ln.answer)
            assert ln.kind in ("fact", "calc")
    assert longest <= 100


def test_lines_of_a_long_seed_stay_short(sim):
    for seed in (9998, 123456789):
        for ln in lines(sim.rollout(seed)):
            assert is_dense(ln.text) and n_tokens(ln.text) <= 110, ln.text
            if ln.kind != "record":
                assert sim.check(ln.prompt, ln.answer).ok, ln.text


def test_line_kinds(rollouts):
    r = rollouts[3]
    ls = lines(r)
    records = [ln for ln in ls if ln.kind == "record"]
    assert records[0].text == params_line(r).text and records[0].text.startswith("senses seed 3 params. light ")
    assert len(records) == 1 + STEPS
    assert records[1].text.startswith("senses seed 3 step 0. population 1 0 0. smell 0. eyes 0. touch 0. ears 0. "
                                      "eyed 0. neurons 0. voice 0. speed 1. intake ")
    assert records[1].text.endswith(" predator_sense 0. stage blind.")
    prompts = {ln.prompt for ln in ls if ln.kind != "record"}
    assert len(prompts) == STEPS * len(STATE_KEYS) + (STEPS - 1) * len(STATE_KEYS) + len(SUMMARY_KEYS)
    assert len(ls) == 1621 == len({ln.text for ln in ls})
    assert "senses seed 3 step 4 0 eyed" in prompts and "senses seed 3 step 4 0 next stage" in prompts
    assert "senses seed 3 final outcome" in prompts and "senses seed 3 step 5 9 next stage" not in prompts
    assert not any(f" {k}" in ln.text for ln in ls if ln.kind == "record" and " step " in ln.text
                   for k in LEDGER_KEYS[:11]), "the ledger is not trained"
    assert len(lines(r, every=5)) < len(ls) / 4


def test_parameters_are_written_as_given(sim):
    assert param_value("light", 0.57) == "0 point 5 7" and param_value("mutation", 0.026) == "0 point 0 2 6"
    assert param_value("body_size", 14228) == "1 4 2 2 8" and param_value("oxygen", 0.011) == "0 point 0 1 1"
    for seed in range(1, 120):
        p = random_params(random.Random(seed))
        r = sim.rollout(seed)
        for key in INPUT_KEYS:
            assert r.params[key] == p[key] and parse_num(param_value(key, r.params[key])) == p[key], (seed, key)
            v = sim.check(f"senses seed {num(seed)} params {key}", param_value(key, p[key]))
            assert v.ok and v.expected == param_value(key, p[key])
        assert r.params["max_neurons"] == senses.max_neurons(p["body_size"], p["cell_types"])
        assert r.params["max_level"] == senses.max_sensor_level(p["cell_types"])
        assert 0 <= r.params["eye_pays"] <= r.params["max_level"]
    assert not sim.check("senses seed 3 params light", "0 point 5 8").ok, "a parameter is exact"


def test_gate_agrees_with_every_line(sim, all_lines):
    for ln in all_lines:
        if ln.kind == "record":
            continue
        assert sim.owns(ln.prompt) and senses.owns(ln.prompt), ln.prompt
        v = sim.check(ln.prompt, ln.answer)
        assert v.ok and v.expected == ln.answer, (ln.text, v)
        assert senses.check(ln.prompt, ln.answer) == v


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
        if key == "group":
            wrong = "no" if ln.answer == "yes" else "yes"
        elif key in WORD_KEYS:
            wrong = next(s for s in STAGES + OUTCOMES if s != ln.answer)
        else:
            wrong = "9 " + ln.answer  # a changed leading digit: far outside the 5 % tolerance
        v = sim.check(ln.prompt, wrong)
        assert not v.ok and v.expected == ln.answer, (ln.text, wrong, v)
        assert not senses.check(ln.prompt, wrong).ok
    assert not sim.check("senses seed 3 final outcome", "many").ok
    assert not sim.check("senses seed 3 step 1 4 stage", "see").ok


def test_gate_rejects_a_changed_digit(sim, all_lines):
    """One digit of the answer changed: a count is then wrong; a measured number is wrong when
    the change is more than 5 %."""
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
    for prompt in ("senses seed 3 final neurons", "senses seed 3 step 1 4 intake", "senses seed 3 final population",
                   "senses seed 3 params light", "senses seed 3 step 2 0 next survival"):
        for answer in ("1 e 9 9 9", "minus 1 e 9 9 9", "", "point", "e", "yes", "7 0 point 5 units", "7 0 . 5",
                       "1 e 3 0 0"):
            v = sim.check(prompt, answer)
            assert not v.ok and v.expected is not None, (prompt, answer, v)
    assert sim.check("senses seed 3 final neurons", "1 e 9 9 9").reason == "not a number"
    assert sim.check("senses seed 3 step 0 eyes", "0").ok and not sim.check("senses seed 3 step 0 eyes", "1").ok


def test_tolerance_is_five_percent_for_numbers_and_exact_for_counts(sim, rollouts):
    r = rollouts[3]
    intake = r.steps[30]["intake"]
    assert sim.check("senses seed 3 step 3 0 intake", dense_value(intake * 1.03)).ok
    assert sim.check("senses seed 3 step 3 0 intake", dense_value(intake * 0.97)).ok
    assert not sim.check("senses seed 3 step 3 0 intake", dense_value(intake * 1.1)).ok
    assert not sim.check("senses seed 3 step 3 0 intake", dense_value(intake * 0.9)).ok
    n = r.steps[30]["population"]
    assert sim.check("senses seed 3 step 3 0 population", num(n)).ok
    assert not sim.check("senses seed 3 step 3 0 population", num(n + 1)).ok
    assert not sim.check("senses seed 3 step 3 0 population", num(n - 1)).ok
    for t in (5, 30, 59):
        for key in ("intake", "survival", "speed"):
            truth = r.steps[t][key]
            assert close(parse_num(dense_value(truth)), truth, rel=0.05)


@pytest.mark.parametrize("prompt", [
    "turkey capital", "q spoon color", "carbon protons", "calc 1 2 plus 7", "check 1 2 plus 7 equals 2 0",
    "gravity seed 7 step 2 0 clumps", "nucleo seed 3 final helium_fraction", "world seed 3 era planets",
    "planets seed 7 final gas", "chem seed 7 final water", "life seed 7 step 1 2 next population",
    "stars seed 7 step 3 population", "cells seed 7 final outcome", "bodies seed 7 step 3 predators",
    "bodies seed 7 final cell_types", "signals seed 7 step 3 success", "signals seed 7 final outcome",
    "signals seed 7 say predator near", "society seed 7 final outcome", "world7 seed 7 final outcome",
    "bodies predict max_neurons body_size 1 0 0 cell_types 4", "signals predict escape head_start 3 speed 2 predator_speed 6",
    "society predict hamilton relatedness 0 point 5 benefit 4 cost 1", "cosmos predict stage population 0",
    "senses", "senses seed 7", "senses seed 7 step 3", "senses seed 7 step 3 clumps", "senses seed 7 step 6 0 eyes",
    "senses seed 7 step 5 9 next eyes", "senses seed 7 final eyed", "senses seed x final eyes",
    "senses seed 7 final", "senses seed 7 params stage", "senses seed 12 step 3 eyes", "senses seed 7 step 3 next",
    "senses seed 7 step 3 eyes ears", "", "senses seed 0 7 final eyes", "senses seed 7 step 0 3 eyes",
    "senses seed minus 7 final eyes", "senses seed 7 point 5 final eyes", "q senses seed 7 final eyes",
    "senses seed 7 final eyes extra", "senses seed 7 step 3 next next eyes", "senses seed 7 era eyes",
    "senses  seed 7 final eyes", "senses seed 7 final helium_fraction", "senses seed 7 params kin_gain",
    "senses predict", "senses predict escape", "senses predict escape head_start 3 speed 2",
    "senses predict escape head_start 9 speed 2 predator_speed 6", "senses predict escape speed 2 head_start 3 predator_speed 6",
    "senses predict nothing level 3", "senses predict stage population 1 0 predator_sense 0 neurons 5 eyed 0 point 9 sensing 0 point 2",
    "senses predict detection_area range 2 dims 3", "senses predict max_level cell_types 2 point 5",
])
def test_does_not_own_other_prompts(sim, prompt):
    assert not sim.owns(prompt) and not senses.owns(prompt) and not LESSONS.owns(prompt)
    for gate in (sim, senses, LESSONS):
        v = gate.check(prompt, "1")
        assert v.ok is False and v.expected is None and v.reason == "not my question"


@pytest.mark.parametrize("prompt", [
    "senses seed 7 step 0 population", "senses seed 7 step 5 9 stage", "senses seed 7 step 5 8 next eyed",
    "senses seed 1 2 3 4 step 1 2 next neurons", "senses seed 7 final kin_gain", "senses seed 7 final group",
    "senses seed 7 params eye_pays", "senses seed 7 step 4 e_intake", "senses seed 0 final eyes",
    "senses seed 7 step 1 0 births",
])
def test_owns_its_prompts(sim, prompt):
    assert sim.owns(prompt) and senses.owns(prompt) and not LESSONS.owns(prompt)
    assert sim.check(prompt, "nonsense words").expected is not None


# ---------------------------------------------------------------------------------------------
# the gate of the simulation


def test_conserved_passes_on_forty_seeds_with_random_params(sim, rollouts):
    for r in rollouts.values():
        v = sim.conserved(r)
        assert v.ok, (r.seed, v)
        assert senses.conserved(r) == v
    rng = random.Random(99)
    for seed in range(1000, 1010):  # and with parameters that do not belong to the seed
        assert sim.conserved(sim.run(seed, **random_params(rng))).ok


def test_energy_is_accounted_for_in_every_generation(rollouts):
    worst = 0.0
    for r in rollouts.values():
        for g in r.generations:
            spent = math.fsum(g[k] for k in ENERGY)
            assert abs(g["e_intake"] - spent) <= 1e-9 * max(1.0, g["e_intake"]), (r.seed, g)
            worst = max(worst, abs(g["e_intake"] - spent) / max(1.0, g["e_intake"]))
            assert all(g[k] >= 0 for k in ENERGY + ("e_intake",))
            assert g["e_reproduction"] == 0.5 * g["births"]
            if g["population"]:
                assert g["intake"] == pytest.approx(g["e_intake"] / g["population"], rel=0.006)
            if g["attack"] == 0:
                assert g["e_eaten"] == 0 and g["caught"] == 0
    assert worst < 1e-12


def test_heads_are_accounted_for_in_every_generation(rollouts):
    for r in rollouts.values():
        assert r.generations[0]["population"] == START == 100
        for g, nxt in zip(r.generations, r.generations[1:] + [None]):
            assert g["starved"] + g["caught"] + g["old"] == g["population"] <= CAP == 200
            assert min(g["starved"], g["caught"], g["old"], g["births"]) >= 0
            assert g["births"] <= CAP and (g["births"] == 0 or g["old"] > 0)
            assert g["births"] == senses.births_from(g["e_reproduction"] + g["e_stored"])
            if nxt is not None:
                # births minus deaths: every adult dies, the offspring are the next generation
                assert nxt["population"] == g["population"] + g["births"] - (g["starved"] + g["caught"] + g["old"])
        for pop, s in zip(r.genomes, r.steps):
            assert len(pop) == s["population"]


def test_probabilities_and_shares_are_consistent(rollouts):
    for r in rollouts.values():
        p = r.params
        for g in r.generations:
            n = g["population"]
            for key in ("attack", "escape", "survival", "eyed", "voice", "eared", "talkers", "sensing"):
                assert 0.0 <= g[key] <= 1.0, (r.seed, key, g[key])
            want = senses.attack_probability(p["predators"], g["predator_sense"]) if p["predators"] and n else 0.0
            assert g["attack"] == want
            if n:
                assert g["survival"] == pytest.approx(senses.survival_probability(g["attack"], g["escape"]), abs=0.006)
                assert g["caught"] <= n - g["starved"]
            assert g["eyed"] <= g["eyes"] * 1.006 + 1e-12 and g["eyes"] <= 5 * g["eyed"] * 1.006 + 1e-12
            assert g["eared"] <= g["ears"] * 1.006 + 1e-12 and g["ears"] <= 5 * g["eared"] * 1.006 + 1e-12
            assert g["talkers"] <= min(g["eared"], g["voice"]) * 1.006
            assert g["sensing"] * 1.006 >= max(g["eyed"], g["eared"])
            assert all(0 <= g[k] <= p["max_level"] for k in SENSORS) and g["neurons"] <= p["max_neurons"] * 1.006
            assert g["stage"] == senses.stage_of(n, g["predator_sense"], g["neurons"], g["eyed"], g["sensing"])
        for pop, s in zip(r.genomes, r.steps):
            n = len(pop)
            if n:
                assert s["eyed"] == sig(sum(g[1] > 0 for g in pop) / n)
                assert s["eyes"] == sig(sum(g[1] for g in pop) / n)
                assert s["voice"] == sig(sum(g[6] for g in pop) / n)
                assert s["neurons"] == sig(sum(senses.neuron_count(g[4]) for g in pop) / n)
                assert s["speed"] == sig(sum(g[5] for g in pop) / n)


def test_predators_follow_their_rule(rollouts):
    moved = 0
    for r in rollouts.values():
        assert r.steps[0]["predator_sense"] == 0
        for before, s in zip(r.steps, r.steps[1:]):
            if r.params["predators"] == 0:
                assert s["predator_sense"] == 0 and s["attack"] == 0 and s["survival"] in (0.0, 1.0)
                continue
            want = senses.predator_step(before["predator_sense"], before["escape"], before["population"] / CAP)
            assert s["predator_sense"] == want and abs(want - before["predator_sense"]) <= 1
            moved += want != before["predator_sense"]
        for t in range(STEPS):  # the predators hold their level for the five generations of a step
            assert len({g["predator_sense"] for g in r.generations[5 * t:5 * t + 5]}) == 1
    assert moved > 40


def test_conserved_catches_tampering(sim, rollouts):
    base = rollouts[35]
    assert base.params["predators"] > 0 and base.steps[40]["eyes"] > 0.5 and base.steps[40]["caught"] > 0

    def broken(change) -> bool:
        r = copy.deepcopy(base)
        change(r)
        return not sim.conserved(r).ok

    assert not broken(lambda r: None)
    assert broken(lambda r: r.steps[10].update(e_intake=r.steps[10]["e_intake"] + 1e-3))
    assert broken(lambda r: r.generations[7].update(e_stored=r.generations[7]["e_stored"] + 0.01))
    assert broken(lambda r: r.steps[10].update(e_sensors=r.steps[10]["e_sensors"] * 0.5))
    assert broken(lambda r: r.steps[10].update(e_eaten=-1.0))
    assert broken(lambda r: r.steps[12].update(births=r.steps[12]["births"] - 1))
    assert broken(lambda r: r.steps[12].update(population=r.steps[12]["population"] - 1))
    assert broken(lambda r: r.generations[13].update(caught=r.generations[13]["caught"] + 1))
    assert broken(lambda r: r.steps[12].update(survival=1.5))
    assert broken(lambda r: r.steps[12].update(survival=r.steps[12]["survival"] * 0.9))
    assert broken(lambda r: r.steps[12].update(attack=r.steps[12]["attack"] * 0.5))
    assert broken(lambda r: r.steps[40].update(eyed=0.0))
    assert broken(lambda r: r.steps[40].update(eyes=5.5))
    assert broken(lambda r: r.steps[40].update(neurons=4096.0))
    assert broken(lambda r: r.steps[40].update(voice=r.steps[40]["voice"] + 0.3))
    assert broken(lambda r: r.steps[40].update(intake=r.steps[40]["intake"] * 1.2))
    assert broken(lambda r: r.steps[5].update(stage="starved"))
    assert broken(lambda r: r.steps[5].update(stage="finished"))
    assert broken(lambda r: r.steps[20].update(predator_sense=(r.steps[20]["predator_sense"] + 2) % 6))
    assert broken(lambda r: r.generations[21].update(predator_sense=r.generations[21]["predator_sense"] + 1))
    assert broken(lambda r: r.genomes[40].pop())
    assert broken(lambda r: r.genomes[40].__setitem__(0, (5, 5, 5, 5, 10, 5, 1)))
    assert broken(lambda r: r.genomes[40].__setitem__(0, (0, 9, 0, 0, 0, 1, 0)))
    assert broken(lambda r: r.generations.pop())
    assert broken(lambda r: r.steps.pop())
    assert broken(lambda r: r.steps.__setitem__(3, dict(r.steps[4])))
    assert broken(lambda r: r.summary.update(outcome="blind" if r.summary["outcome"] != "blind" else "seeing"))
    assert broken(lambda r: r.summary.update(kin_gain=0.5))
    assert broken(lambda r: r.summary.update(group="yes" if r.summary["group"] == "no" else "no"))
    assert broken(lambda r: r.summary.update(neurons=r.summary["neurons"] + 1.0))
    assert broken(lambda r: r.params.update(max_neurons=2048))
    assert broken(lambda r: r.params.update(max_level=0))
    assert broken(lambda r: r.params.update(predators=0.0))
    assert broken(lambda r: r.params.update(light=0.0)), "another world would give other intakes"
    plain = Rollout("senses", base.seed, base.params, base.steps, base.summary)
    assert not sim.conserved(plain).ok, "without the generations and the genomes there is nothing to check"


def test_summary_and_what_it_hands_upward(rollouts):
    groups = set()
    for r in rollouts.values():
        last, s = r.steps[-1], r.summary
        for key in ("stage", "population", "eyes", "ears", "neurons", "voice", "eared", "talkers", "survival"):
            assert s[key] == last[key]
        assert s["predator_pressure"] == sig(last["attack"]) and 0 <= s["kin_gain"] <= 1
        assert s["group"] == ("yes" if s["kin_gain"] >= 0.01 else "no")
        assert s["outcome"] in OUTCOMES and s["stage"] in STAGES
        assert s["outcome"] == senses.outcome_of(last["population"], last["neurons"], last["eyed"], last["sensing"])
        assert s["talkers"] <= min(s["eared"], s["voice"]) * 1.006
        if r.params["predators"] == 0 or s["voice"] == 0 or s["eared"] == 0:
            assert s["kin_gain"] == 0 and s["group"] == "no"
        pop = r.genomes[-1]
        if pop:
            tr = [traits(g, r.params) for g in pop]
            callers = sum(1 for g, x in zip(pop, tr) if g[6] and x.notices) / len(pop)
            hearing = math.fsum(x.hearing for x in tr) / len(pop)
            assert s["kin_gain"] == sig(senses.kin_gain(last["attack"], last["escape"], hearing, callers))
        groups.add(s["group"])
        up = senses.handoff_out(s)
        assert set(up) == {"eared", "voice", "talkers", "neurons", "group", "kin_gain", "predator_pressure",
                           "population", "outcome"} and all(up[k] == s[k] for k in up)
    assert groups == {"yes", "no"}


def test_handoff_in_maps_the_bodies():
    summary = {"stage": "food_web", "species": 43, "max_size": 512, "cell_types": 7, "trophic_levels": 3,
               "oxygen": 0.0755, "predator_share": 0.239, "outcome": "animals_like"}
    assert handoff_in(summary) == {"body_size": 512, "cell_types": 7, "predators": 0.3, "oxygen": 0.0755}
    out = {"body_size": 128, "cell_types": 6, "predators": "yes", "predator_share": 0.0155, "oxygen": 0.0363,
           "species": 35}
    assert handoff_in(out) == {"body_size": 128, "cell_types": 6, "predators": 0.0465, "oxygen": 0.0363}
    assert handoff_in({"max_size": 1, "cell_types": 1, "trophic_levels": 1, "oxygen": 0.16,
                       "predator_share": 0.0}) == {"body_size": 1, "cell_types": 1, "predators": 0.0, "oxygen": 0.16}
    assert handoff_in({"predators": "no", "predator_share": 0.2})["predators"] == 0.0
    assert handoff_in({"oxygen": 2.5}) == {"oxygen": 1.0} and handoff_in({}) == {}
    assert handoff_in({"oxygen": float("nan"), "cell_types": "many"}) == {}
    for below in (summary, out):
        r = senses.run(3, **{**random_params(random.Random(3)), **handoff_in(below)})
        assert senses.conserved(r).ok and r.params["body_size"] == below.get("max_size", below.get("body_size"))
    # single cells below: no organs, so the level stays blind and hands a mute, deaf population upward
    r = senses.run(3, **{**random_params(random.Random(3)), **handoff_in({"max_size": 1, "cell_types": 1})})
    assert r.summary["outcome"] in ("blind", "starved") and r.summary["talkers"] == r.summary["neurons"] == 0


# ---------------------------------------------------------------------------------------------
# lessons


def test_lesson_rules_are_the_functions_of_the_simulation():
    for name, rule in RULES.items():
        assert rule.compute is FUNCTIONS[name], name
        assert rule.compute.__module__ == "haishool.evo.senses"
    assert [ln.text for ln in LESSONS.generate(random.Random(1), 50)] == \
           [ln.text for ln in lesson_gate().generate(random.Random(1), 50)]


def test_every_lesson_is_accepted_and_true_of_the_simulation(lessons):
    assert len(lessons) == 400 and {ln.prompt.split()[2] for ln in lessons} == set(RULES)
    for ln in lessons:
        assert is_dense(ln.text) and n_tokens(ln.text) <= MAX_TOKENS, ln.text
        assert ln.topic == "predict_senses" and ln.kind == "calc" and "." not in ln.prompt + ln.answer
        v = LESSONS.check(ln.prompt, ln.answer)
        assert v.ok and v.expected == ln.answer, (ln.text, v)
        assert senses.check(ln.prompt, ln.answer).ok and senses.owns(ln.prompt), ln.text
        name, values = LESSONS.parse(ln.prompt)
        own = FUNCTIONS[name](*values)
        assert ln.answer == (own if isinstance(own, str) else num(own)), (ln.text, own)
        assert ln.meta["split_key"] == LESSONS.split_key(ln.prompt)
    assert max(n_tokens(ln.text) for ln in lessons) <= 60


def test_lessons_reject_altered_answers(lessons):
    for ln in lessons:
        if ln.prompt.split()[2] == "stage":
            wrong = next(s for s in STAGES if s != ln.answer)
        else:
            wrong = "9 " + ln.answer
        for gate in (LESSONS, senses):
            v = gate.check(ln.prompt, wrong)
            assert not v.ok and v.expected == ln.answer, (ln.text, wrong)
        assert not LESSONS.check(ln.prompt, "1 e 9 9 9").ok and not LESSONS.check(ln.prompt, "").ok


def test_lesson_records_say_what_each_rule_computes():
    records = LESSONS.records()
    assert len(records) == len(RULES) == 24
    for ln in records:
        assert ln.kind == "record" and is_dense(ln.text) and n_tokens(ln.text) <= MAX_TOKENS, ln.text
        assert ln.text.startswith("senses predict ")


def test_lessons_by_hand():
    for prompt, answer in (
            ("senses predict detection_area range 2 dims 1", "5"),
            ("senses predict detection_area range 2 dims 2", "2 5"),
            ("senses predict detection_area range 0 point 5 dims 2", "4"),
            ("senses predict pooling_gain receptors 1 6", "4"),
            ("senses predict pooling_gain receptors 2", "1 point 4 1 4"),
            ("senses predict sensor_range level 3 reach 1", "3"),
            ("senses predict sensor_range level 5 reach 0 point 1", "0 point 5"),
            ("senses predict encounter speed 2 density 0 point 2 patchiness 0 point 5 cells 1", "0 point 2"),
            ("senses predict encounter speed 2 density 0 point 2 patchiness 0 point 5 cells 3", "1 point 2"),
            ("senses predict intake rate 2 yield 1 0 0", "1 0 0"),
            ("senses predict energy_yield oxygen 0", "6 point 2 5"),
            ("senses predict energy_yield oxygen 0 point 0 2 5", "5 3 point 1 2"),
            ("senses predict energy_yield oxygen 0 point 2", "1 0 0"),
            ("senses predict sensor_cost level 3 unit 2", "2 4"),
            ("senses predict sensor_cost level 0 unit 2", "0"),
            ("senses predict movement_cost speed 3", "1 3 point 5"),
            ("senses predict escape head_start 3 speed 2 predator_speed 6", "0 point 3 7 5"),
            ("senses predict escape head_start 0 speed 5 predator_speed 6", "0"),
            ("senses predict escape head_start 5 speed 4 predator_speed 6", "1"),
            ("senses predict attack predators 0 point 1 predator_sense 2", "0 point 3 9 3 5"),
            ("senses predict survival attack 0 point 5 escape 0 point 6", "0 point 8"),
            ("senses predict energy_budget intake 6 0 sensors 1 4 neurons 1 point 6 movement 4", "4 0 point 4"),
            ("senses predict energy_budget intake 6 sensors 1 4 neurons 1 point 6 movement 4", "0"),
            ("senses predict offspring fitness 3 0 total_fitness 6 0 0 0 births 2 0 0", "1"),
            ("senses predict births surplus 7 point 3", "1 4"),
            ("senses predict births surplus 1 5 0", "2 0 0"),
            ("senses predict neurons_needed associations 1", "0"),
            ("senses predict neurons_needed associations 2", "2"),
            ("senses predict neurons_needed associations 2 4", "6 4"),
            ("senses predict sensor_use neurons 1 6 associations 2 4", "0 point 3 7 5"),
            ("senses predict sensor_use neurons 0 associations 2", "0 point 5"),
            ("senses predict sensor_use neurons 6 4 associations 2 4", "1"),
            ("senses predict max_neurons body_size 5 1 2 cell_types 7", "1 2 8"),
            ("senses predict max_neurons body_size 5 1 2 cell_types 2", "0"),
            ("senses predict max_level cell_types 1", "0"),
            ("senses predict max_level cell_types 2", "5"),
            ("senses predict predator_step predator_sense 1 escape 0 point 6 prey_share 1", "2"),
            ("senses predict predator_step predator_sense 1 escape 0 prey_share 1", "1"),
            ("senses predict predator_step predator_sense 3 escape 0 prey_share 1", "2"),
            ("senses predict group_detection p 0 point 5 n 3", "0 point 8 7 5"),
            ("senses predict mate_found voice 1 hearing 0 point 4 voice_share 0 point 5 mean_hearing 0 point 6",
             "0 point 4"),
            ("senses predict kin_gain attack 0 point 5 escape 0 point 5 hearing 0 point 4 callers 1", "0 point 1"),
            ("senses predict stage population 0 predator_sense 4 neurons 5 0 eyed 1 sensing 1", "starved"),
            ("senses predict stage population 2 0 0 predator_sense 3 neurons 5 0 eyed 1 sensing 1", "arms_race"),
            ("senses predict stage population 2 0 0 predator_sense 2 neurons 3 2 eyed 0 sensing 1", "brained"),
            ("senses predict stage population 2 0 0 predator_sense 2 neurons 3 1 point 9 eyed 0 point 5 sensing 1",
             "seeing"),
            ("senses predict stage population 2 0 0 predator_sense 0 neurons 0 eyed 0 point 4 sensing 0 point 5",
             "smelling"),
            ("senses predict stage population 2 0 0 predator_sense 0 neurons 0 eyed 0 point 1 sensing 0 point 4",
             "blind")):
        v = LESSONS.check(prompt, answer)
        assert v.ok and v.expected == answer, (prompt, v)
    # the break-even eye: none in the dark, more with light, less when tissue costs more
    def pays(light: float, cost: float, dims: int) -> int:
        return int(LESSONS.expected("break_even", (2, 0.2, 0.6, light, cost, dims)))
    assert pays(0.0, 1.0, 1) == pays(0.0, 0.5, 2) == 0
    assert 0 < pays(0.3, 1.0, 1) <= pays(1.0, 1.0, 1) and pays(1.0, 2.0, 1) <= pays(1.0, 0.5, 1)
    assert pays(1.0, 2.0, 2) <= pays(1.0, 0.5, 2) and pays(1.0, 0.5, 1) >= 4


def test_an_agent_is_the_sum_of_the_lessons():
    """One agent worked out with the lessons alone gives what the simulation computes for it."""
    p = {"light": 0.8, "patchiness": 0.5, "density": 0.2, "predators": 0.1, "cost": 1.5, "dims": 2, "oxygen": 0.03}
    g = (2, 3, 0, 4, 5, 3, 1)  # smell 2, eyes 3, no touch, ears 4, 32 neurons, speed 3, a voice

    def ask(name: str, *values) -> float:
        return parse_num(LESSONS.expected(name, values))

    ranges = {"smell": ask("sensor_range", 2, 1), "eyes": ask("sensor_range", 3, 1), "ears": ask("sensor_range", 4, 1)}
    extra = {s: ask("detection_area", r, 2) - 1 for s, r in ranges.items()}
    assert extra == {"smell": 24, "eyes": 48, "ears": 80}
    use = ask("sensor_use", 32, 152)
    cells = 1 + use * max(0.4 * extra["smell"], 0.8 * extra["eyes"])
    intake = ask("intake", ask("encounter", 3, 0.2, 0.5, cells), ask("energy_yield", 0.03))
    x = traits(g, p)
    assert x.neurons == 32 and x.intake == pytest.approx(intake, rel=2e-3)
    assert x.sensors == pytest.approx(ask("sensor_cost", 2, 1.5) + ask("sensor_cost", 3, 3) + ask("sensor_cost", 4, 1.5))
    assert x.move == ask("movement_cost", 3) and x.neurons_cost == pytest.approx(1.5 * 0.05 * 32)
    head_start = use * max(0.5 * 0.8 * ranges["eyes"], ranges["ears"])
    assert x.escape == pytest.approx(ask("escape", head_start, 3, 6), rel=1e-3) and x.notices
    assert 0 < x.escape < 1
    assert x.hearing == pytest.approx(use * 4 / 5, rel=1e-3)
    blind = traits((0, 0, 0, 0, 0, 1, 0), p)
    assert blind.intake == pytest.approx(ask("intake", ask("encounter", 1, 0.2, 0.5, 1), ask("energy_yield", 0.03)),
                                         rel=1e-3)
    assert blind.sensors == blind.neurons_cost == blind.escape == blind.hearing == 0 and not blind.notices


def test_lessons_hold_in_the_rollouts(rollouts):
    """What a lesson says about a step is what the rollout shows."""
    stages = steps = 0
    for r in rollouts.values():
        p = r.params
        first = r.steps[0]
        rate = LESSONS.expected("encounter", (1, p["density"], p["patchiness"], 1))
        founders = senses.food_intake(parse_num(rate), senses.energy_yield(p["oxygen"]))
        assert first["intake"] == pytest.approx(founders, rel=0.01), "the founders eat by random encounter"
        assert LESSONS.check(f"senses predict max_level cell_types {num(p['cell_types'])}", num(p["max_level"])).ok
        assert LESSONS.check(f"senses predict max_neurons body_size {num(p['body_size'])} cell_types "
                             f"{num(p['cell_types'])}", num(p["max_neurons"])).ok
        for before, s in zip(r.steps, r.steps[1:]):
            if p["predators"] > 0:
                assert LESSONS.check(f"senses predict attack predators {num(p['predators'])} predator_sense "
                                     f"{num(s['predator_sense'])}", num(s["attack"])).ok or s["population"] == 0
                prompt = (f"senses predict predator_step predator_sense {num(before['predator_sense'])} escape "
                          f"{num(before['escape'], sig=6)} prey_share {num(before['population'] / CAP, sig=6)}")
                assert LESSONS.check(prompt, num(s["predator_sense"])).ok, prompt
                steps += 1
            prompt = (f"senses predict stage population {num(s['population'])} predator_sense "
                      f"{num(s['predator_sense'])} neurons {num(s['neurons'])} eyed {num(s['eyed'])} sensing "
                      f"{num(s['sensing'])}")
            assert LESSONS.check(prompt, s["stage"]).ok, prompt
            stages += 1
    assert stages == 40 * (STEPS - 1) and steps > 1000
    # the eye that pays, in a world of today's air and a body that allows every neuron
    for light, cost, dims in ((0.0, 1.0, 1), (0.4, 1.0, 1), (1.0, 0.5, 2), (1.0, 2.0, 2)):
        r = senses.run(1, light=light, cost=cost, dims=dims, **{k: v for k, v in WORLD.items() if k != "cost"})
        assert LESSONS.check(f"senses predict break_even speed 2 density 0 point 2 patchiness 0 point 6 light "
                             f"{num(light)} cost {num(cost)} dims {num(dims)}", num(r.params["eye_pays"])).ok


# ---------------------------------------------------------------------------------------------
# plausibility: what the worlds must look like


def test_no_eyes_in_the_dark():
    """Where there is no light an eye only costs, and selection keeps it away (the energy-saving
    explanation of the lost eyes of cave fish, which is one of several and is not tested here)."""
    for dims in (1, 2):
        dark = finals(light=0.0, predators=0.0, dims=dims)
        assert mean(dark, "eyes") < 0.3 and mean(dark, "eyed") < 0.25, dims
        assert all(row["eyes"] < 1 and row["stage"] != "seeing" for row in dark)
        assert mean(dark, "smell") + mean(dark, "touch") > 2, "they find their food by smell or touch instead"
        assert all(row["final_outcome"] in ("sensing", "brained") for row in dark)


def test_eyes_spread_with_light():
    for dims in (1, 2):
        dark = finals(light=0.0, predators=0.0, dims=dims)
        lit = finals(light=1.0, predators=0.0, dims=dims)
        assert mean(lit, "eyed") > 0.95 and mean(lit, "eyes") > 2, dims
        assert all(row["eyed"] > 0.9 and row["final_outcome"] in ("seeing", "brained") for row in lit)
        assert mean(lit, "eyes") > mean(dark, "eyes") + 2
        assert mean(lit, "smell") < 0.6 < mean(dark, "smell"), "with eyes the nose is given up"
        assert mean(lit, "intake") > 50


def test_canonical_worlds_see_where_an_eye_pays(rollouts):
    dark = [r for r in rollouts.values() if r.params["light"] == 0]
    assert len(dark) >= 4
    assert all(r.params["eye_pays"] == 0 and r.summary["eyes"] < 0.3 for r in dark)
    assert all(s["eyed"] < 0.5 for r in dark for s in r.steps)
    for r in rollouts.values():
        if r.params["eye_pays"] == 0:
            assert r.summary["eyes"] < 0.5, r.seed
    bright = [r for r in rollouts.values() if r.params["light"] >= 0.5 and r.params["eye_pays"] >= 1]
    seeing = [r for r in bright if r.steps[-1]["eyed"] >= 0.5]
    assert len(bright) >= 15 and len(seeing) >= 0.6 * len(bright)
    assert statistics.mean(r.summary["eyes"] for r in bright) > 0.8
    pays = [r.summary["eyes"] for r in rollouts.values() if r.params["eye_pays"] >= 3]
    assert len(pays) >= 4 and statistics.mean(pays) > 1


def test_neurons_rise_with_the_useful_sensors_and_fall_with_their_cost():
    def in_use(row: dict) -> int:
        return sum(row[key] >= 0.5 for key in SENSORS)

    for dims in (1, 2):
        one = finals(light=0.0, predators=0.0, dims=dims)  # a nose (or touch) is all that pays
        two = finals(light=0.0, predators=0.2, dims=dims)  # ears pay as well
        assert mean(two, "ears") > mean(one, "ears") + 1
        few = [row["neurons"] for row in one + two if in_use(row) <= 1]
        many = [row["neurons"] for row in one + two if in_use(row) >= 2]
        assert len(few) >= 4 and len(many) >= 4
        assert statistics.mean(many) > 1.5 * statistics.mean(few), dims
        cheap = finals(light=1.0, predators=0.2, dims=dims, cost=0.5)
        dear = finals(light=1.0, predators=0.2, dims=dims, cost=2.0)
        assert mean(cheap, "neurons") > 1.4 * mean(dear, "neurons"), dims
        assert mean(cheap, "eyes") > mean(dear, "eyes")
    ring_one, ring_two = finals(light=0.0, predators=0.0, dims=1), finals(light=0.0, predators=0.2, dims=1)
    assert mean(ring_two, "neurons") > 1.3 * mean(ring_one, "neurons")
    ring, grid = finals(light=1.0, predators=0.0, dims=1), finals(light=1.0, predators=0.0, dims=2)
    assert mean(grid, "neurons") > 2 * mean(ring, "neurons"), "a grid has more places to tell apart than a ring"


def test_two_ways_out_of_the_dark_on_a_grid():
    """A nose with a brain, or touch with hardly any neurons: which hill a population climbs is
    chance, and from the low one no single step leads to the high one."""
    dark = finals(light=0.0, predators=0.0, dims=2)
    noses = [row for row in dark if row["smell"] > 3 and row["neurons"] > 100]
    touchers = [row for row in dark if row["touch"] > 4 and row["smell"] < 0.5 and row["neurons"] < 16]
    assert len(noses) >= 4 and len(touchers) >= 1 and len(noses) + len(touchers) == len(dark)
    assert min(row["intake"] for row in noses) > max(row["intake"] for row in touchers)
    world = {**WORLD, "light": 0.0, "predators": 0.0, "dims": 2}
    toucher = traits((0, 0, 5, 0, 2, 3, 0), world)
    step = traits((1, 0, 5, 0, 2, 3, 0), world)  # a first nose on top: more to wire, the same few neurons
    nose = traits((3, 0, 0, 0, 7, 3, 0), world)

    def net(x) -> float:
        return x.intake - x.sensors - x.neurons_cost - x.move

    assert net(step) < net(toucher) < net(nose)


def test_no_neurons_without_sensors(rollouts):
    """Neurons only pay for what they connect: a population without sensors has next to none."""
    for r in rollouts.values():
        for s in r.steps:
            if s["sensing"] < 0.1 and s["population"]:
                assert s["neurons"] < 4, (r.seed, s)


def test_predators_bring_ears_speed_and_a_voice():
    for dims in (1, 2):
        safe = finals(light=0.0, predators=0.0, dims=dims)
        hunted = finals(light=0.0, predators=0.2, dims=dims)
        assert mean(safe, "ears") < 0.5 and mean(hunted, "ears") > 1, dims
        assert mean(hunted, "speed") > mean(safe, "speed") + 0.5, dims
        assert mean(safe, "voice") < 0.2 < 0.3 < mean(hunted, "voice"), dims
        assert mean(hunted, "final_talkers") > 0.2 > 0.1 > mean(safe, "final_talkers")
        assert mean(hunted, "predator_sense") >= 2.5 and mean(hunted, "survival") > 0.7
        assert all(row["predator_sense"] == 0 and row["survival"] == 1 for row in safe)
        assert sum(row["stage"] == "arms_race" for row in hunted) >= 3, "prey that escape, and sharper predators"
        for row in safe + hunted:  # in every run the voice comes with the ears
            assert row["voice"] < 0.3 or row["ears"] > 1, row


def test_a_voice_pays_only_where_there_are_ears(rollouts):
    spare = 50.0
    mute = senses.social_cost(0, senses.mate_found(0, 0.0, 0.0, 0.0), spare)
    deaf_world = senses.social_cost(1, senses.mate_found(1, 0.0, 0.0, 0.0), spare)
    hearing_world = senses.social_cost(1, senses.mate_found(1, 0.0, 0.0, 0.6), spare)
    assert deaf_world > mute > hearing_world, "calling costs; it is repaid only by those who hear it"
    assert senses.social_cost(0, senses.mate_found(0, 0.6, 1.0, 0.0), spare) < mute, "and ears find the callers"
    with_ears = [r.summary["voice"] for r in rollouts.values() if r.summary["eared"] >= 0.5]
    without = [r.summary["voice"] for r in rollouts.values() if r.summary["eared"] < 0.5 and r.summary["population"]]
    assert len(with_ears) >= 2 and statistics.mean(with_ears) > 3 * statistics.mean(without)
    assert statistics.mean(without) < 0.15 and max(without) < 0.5


def test_the_body_limits_the_senses():
    world = {k: v for k, v in WORLD.items() if k not in ("body_size", "cell_types")}
    one_kind = senses.run(2, light=1.0, predators=0.1, dims=1, body_size=5000, cell_types=1, **world)
    assert one_kind.params["max_level"] == 0 and one_kind.summary["outcome"] == "blind"
    assert all(s["sensing"] == 0 and s["voice"] == 0 and s["neurons"] == 0 for s in one_kind.steps)
    two_kinds = senses.run(2, light=1.0, predators=0.1, dims=1, body_size=5000, cell_types=2, **world)
    assert two_kinds.params["max_level"] == 5 and two_kinds.params["max_neurons"] == 0
    assert two_kinds.summary["outcome"] == "seeing" and all(s["neurons"] == 0 for s in two_kinds.steps)
    assert two_kinds.summary["eyes"] < 1.5, "one free association: a level-1 eye is all it can use"
    small = senses.run(2, light=1.0, predators=0.1, dims=2, body_size=40, cell_types=6, **world)
    large = senses.run(2, light=1.0, predators=0.1, dims=2, body_size=5000, cell_types=6, **world)
    assert small.params["max_neurons"] == 8 and large.params["max_neurons"] == 1024
    assert max(s["neurons"] for s in small.steps) <= 8 < large.summary["neurons"]
    assert large.summary["eyes"] > small.summary["eyes"] and large.summary["outcome"] == "brained"


def test_a_world_too_poor_starves(sim):
    r = sim.run(4, light=0.5, patchiness=0.8, density=0.1, predators=0.2, oxygen=0.01, mutation=0.01)
    assert sim.conserved(r).ok and r.summary["outcome"] == r.summary["stage"] == "starved"
    assert r.summary["population"] == 0 and r.steps[0]["population"] == 100
    dead = next(t for t, s in enumerate(r.steps) if s["population"] == 0)
    for s in r.steps[dead:]:
        assert s["stage"] == "starved" and all(s[k] == 0 for k in STATE_KEYS[:-2] + LEDGER_KEYS[:-5])
    for ln in lines(r)[-13:]:
        assert is_dense(ln.text)


def test_stages_and_outcomes(rollouts):
    stages = {s["stage"] for r in rollouts.values() for s in r.steps}
    assert stages == set(STAGES)
    outcomes = {r.summary["outcome"] for r in rollouts.values()}
    assert {"blind", "sensing", "seeing", "brained"} <= outcomes <= set(OUTCOMES)
    for r in rollouts.values():
        assert r.steps[0]["stage"] == "blind" and r.steps[0]["population"] == 100
        assert r.steps[0]["speed"] == 1 and r.steps[0]["sensing"] == 0 and r.steps[0]["neurons"] == 0
        if r.params["max_level"] == 0:
            assert r.summary["outcome"] in ("blind", "starved")
        if r.summary["stage"] == "arms_race":
            assert r.params["predators"] > 0 and r.steps[-1]["predator_sense"] >= 3
    alive = [r for r in rollouts.values() if r.summary["population"]]
    assert len(alive) >= 36 and all(r.summary["population"] == 200 for r in alive)


def test_a_change_in_the_twelfth_digit_moves_no_line(sim, rollouts):
    """Another machine's exp may differ in the last digit. A change ten thousand times larger
    must not move a single line: no draw sits on a knife's edge."""
    for seed in SEEDS[:8]:
        r = rollouts[seed]
        base = [ln.text for ln in lines(r)]
        p = random_params(random.Random(seed))
        nudged = sim.run(seed, **dict(p, density=p["density"] * (1 + 1e-12), cost=p["cost"] * (1 - 1e-12)))
        assert nudged.generations != r.generations
        got = [ln.text for ln in lines(nudged)]
        assert got[1:] == base[1:], seed


def test_numbers_use_three_significant_digits(all_lines):
    assert dense_value(1052.53) == "1 0 5 0" and dense_value(0.03234) == "0 point 0 3 2 3"
    assert dense_value(7) == "7" and dense_value(0.0) == "0" and dense_value("seeing") == "seeing"
    assert dense_value(0.000412) == "4 point 1 2 e minus 4" and dense_value(0.9996) == "1"
    for ln in all_lines:
        if ln.kind == "record" or ln.prompt.split()[-1] in WORD_KEYS:
            continue
        value = parse_num(ln.answer)
        assert value is not None and value >= 0, ln.text
        digits = "".join(w for w in ln.answer.split(" e ")[0].split() if w.isdigit())
        assert len(digits.strip("0")) <= 3, ln.text
        assert len(ln.answer.split()) <= 8, ln.text


def test_a_run_is_fast(sim):
    worst = 0.0
    for seed in range(100, 112):
        start = time.perf_counter()
        sim.run(seed, **random_params(random.Random(seed)))
        worst = max(worst, time.perf_counter() - start)
    assert worst < 2.0


# ---------------------------------------------------------------------------------------------
# the review of the level: lessons checked a second way, balance, other gates' prompts, arithmetic


@pytest.fixture(scope="module")
def many_lessons():
    """500 lessons for each of 4 seeds."""
    return [ln for seed in (1, 2, 3, 4) for ln in LESSONS.generate(random.Random(seed), 500)]


def test_two_thousand_lessons_are_accepted_and_agree_with_the_rules_written_out_again(many_lessons):
    assert len(many_lessons) == 2000
    seen = collections.Counter()
    for ln in many_lessons:
        assert is_dense(ln.text) and n_tokens(ln.text) <= MAX_TOKENS and "." not in ln.prompt + ln.answer, ln.text
        v = LESSONS.check(ln.prompt, ln.answer)
        assert v.ok and v.expected == ln.answer, (ln.text, v)
        name, values = LESSONS.parse(ln.prompt)
        own = FUNCTIONS[name](*values)
        assert ln.answer == (own if isinstance(own, str) else num(own)), (ln.text, own)
        hand = BY_HAND[name](*values)
        if isinstance(own, str) or isinstance(own, int):
            assert hand == own and type(hand) is type(own), (ln.text, hand)
        else:  # the same number, and the answer is that number at four digits
            assert hand == pytest.approx(own, rel=1e-12, abs=1e-12), (ln.text, hand, own)
            assert parse_num(ln.answer) == pytest.approx(hand, rel=5.1e-4, abs=1e-12), (ln.text, hand)
        seen[name] += 1
    assert set(seen) == set(RULES) and min(seen.values()) >= 55, "every rule is drawn about as often"
    assert max(n_tokens(ln.text) for ln in many_lessons) <= 60


def test_no_answer_dominates_a_rule(many_lessons):
    """Drawn evenly from the ranges, max_level would answer 5 in 97 of 100 lessons and
    energy_yield 100 in 74 (measured before :class:`SensesLessons`)."""
    answers = collections.defaultdict(collections.Counter)
    prompts = collections.defaultdict(set)
    for ln in many_lessons:
        answers[ln.prompt.split()[2]][ln.answer] += 1
        prompts[ln.prompt.split()[2]].add(ln.prompt)
    share = {name: c.most_common(1)[0][1] / sum(c.values()) for name, c in answers.items()}
    assert max(share.values()) <= 0.6, share
    assert 0.4 <= share["max_level"] <= 0.6 and set(answers["max_level"]) == {"0", "5"}, "the two-answer rule"
    assert share["energy_yield"] < 0.5 and share["neurons_needed"] < 0.35 and share["max_neurons"] < 0.35
    assert sum(v <= 0.36 for v in share.values()) >= 20, share
    assert set(answers["stage"]) == set(STAGES), "every stage is asked for, starved and blind too"
    assert min(answers["stage"].values()) >= 2
    assert set(answers["break_even"]) >= {"0", "2", "3", "4"} and set(answers["predator_step"]) >= set("12345")
    assert {"0", "2", "1 0 2 4"} <= set(answers["max_neurons"]) and {"0", "2", "1 0 2 4"} <= set(answers["neurons_needed"])
    assert {"6 point 2 5", "1 0 0"} & set(answers["energy_yield"]) and len(answers["energy_yield"]) >= 20
    for name in set(RULES) - {"max_level"}:
        assert len(prompts[name]) >= 30, (name, len(prompts[name]))
    assert len(prompts["movement_cost"]) >= 30, "speeds between the whole ones too"


def test_lesson_inputs_are_drawn_within_their_ranges():
    rng = random.Random(8)
    dark = nobody = 0
    for _ in range(3000):
        name = rng.choice(sorted(RULES))
        values = []
        for spec in RULES[name].inputs:
            value = LESSONS.draw(rng, name, spec, values)
            assert spec.low <= value <= spec.high, (name, spec.name, value)
            assert not spec.integer or type(value) is int, (name, spec.name, value)
            assert spec.integer or value == round(value, spec.places), (name, spec.name, value)
            values.append(value)
        if name == "stage":
            assert values[3] <= values[4], "an agent with an eye is an agent with a sensor"
            nobody += values[0] == 0
        if name == "max_neurons":
            assert values[0] < 2 ** 15
        dark += name == "break_even" and values[3] == 0
    assert dark >= 5 and nobody >= 5
    assert [ln.text for ln in LESSONS.generate(random.Random(3), 80)] == \
           [ln.text for ln in SensesLessons(senses.SIM, RULES).generate(random.Random(3), 80)]
    assert [ln.text for ln in LESSONS.generate(random.Random(3), 80)] != \
           [ln.text for ln in LESSONS.generate(random.Random(4), 80)]


def test_the_lessons_judge_every_body_level_9_can_hand_over():
    for prompt, answer in (("senses predict max_neurons body_size 1 0 4 8 5 7 6 cell_types 3 0", "1 0 2 4"),
                           ("senses predict max_neurons body_size 3 1 6 2 3 cell_types 1 2", "1 0 2 4"),
                           ("senses predict max_neurons body_size 1 0 4 8 5 7 6 cell_types 2", "0"),
                           ("senses predict max_neurons body_size 7 cell_types 3", "0"),
                           ("senses predict max_neurons body_size 8 cell_types 3", "2"),
                           ("senses predict max_level cell_types 3 0", "5"),
                           ("senses predict energy_yield oxygen 1", "1 0 0"),
                           ("senses predict energy_yield oxygen 0 point 0 5", "1 0 0"),
                           ("senses predict energy_yield oxygen 0 point 0 4", "8 1 point 2 5"),
                           ("senses predict movement_cost speed 2 point 5", "7 point 8 1 2"),
                           ("senses predict group_detection p 0 point 2 n 5", "0 point 6 7 2 3"),
                           ("senses predict stage population 2 0 0 predator_sense 0 neurons 1 0 2 4 eyed 0 sensing 1",
                            "brained")):
        v = LESSONS.check(prompt, answer)
        assert v.ok and v.expected == answer, (prompt, v)
    for prompt in ("senses predict max_neurons body_size 1 0 4 8 5 7 7 cell_types 3",
                   "senses predict max_level cell_types 3 1", "senses predict energy_yield oxygen 1 point 0 1",
                   "senses predict group_detection p 0 point 2 n 1 1", "senses predict movement_cost speed 5 point 1"):
        assert not LESSONS.owns(prompt), prompt


def test_the_eye_that_pays_is_the_best_of_all_six_levels():
    """Not the first level at which a better eye stops paying: the two differ now and then."""
    rng = random.Random(3)
    differ = 0
    for _ in range(3000):
        args = (rng.randint(1, 5), round(rng.uniform(0.05, 0.5), 2), round(rng.uniform(0, 0.8), 2),
                round(rng.uniform(0, 1), 2), round(rng.uniform(0.5, 2), 2), rng.randint(1, 2))
        nets = [senses.eye_net(level, *args) for level in range(6)]
        best = senses.break_even_level(*args)
        assert best == _hand_break_even(*args)
        assert nets[best] >= max(nets) - 1e-9 and all(nets[level] < nets[best] for level in range(best))
        first = 0
        while first < 5 and nets[first + 1] > nets[first] + 1e-9:
            first += 1
        differ += first != best
    assert 0 < differ < 150


SOURCES = ("truth.maths", "truth.elements", "truth.substances", "truth.reactions", "truth.forces", "cosmos.predict",
           "cosmos.nucleo", "cosmos.gravity", "cosmos.planets", "cosmos.chem", "cosmos.life", "evo.stars",
           "evo.cells", "evo.bodies", "evo.signals", "evo.society")


def _prompts_of(source: str) -> list[str]:
    """Real prompts of a round-5 gate, of the round-6 lessons, or of another level (its rollout of
    seed 3 and, where it has them, its lessons)."""
    rng = random.Random(7)
    module = importlib.import_module("haishool." + source)
    if source.startswith("truth."):
        return [ln.prompt for ln in module.gate().generate(rng, 80) if ln.kind != "record"]
    if source == "cosmos.predict":
        return [ln.prompt for ln in module.generate(rng, 80)]
    r = module.simulation().run(3, **module.random_params(random.Random(3)))
    out = [ln.prompt for ln in module.lines(r) if ln.kind != "record"]
    if hasattr(module, "LESSONS"):
        out += [ln.prompt for ln in module.LESSONS.generate(rng, 80)]
    return out


@pytest.mark.parametrize("source", SOURCES)
def test_does_not_own_the_prompts_of_other_gates(sim, source):
    try:
        prompts = _prompts_of(source)
    except Exception as exc:  # another level under repair: its prompts cannot be asked for now
        pytest.skip(f"{source}: {exc!r}")
    assert len(prompts) >= 40
    for prompt in prompts:
        assert not sim.owns(prompt) and not LESSONS.owns(prompt) and not senses.owns(prompt), prompt
        assert senses.check(prompt, "1") == senses.check(prompt, "yes"), prompt
        assert senses.check(prompt, "1").expected is None and senses.check(prompt, "1").reason == "not my question"


def test_powers_are_multiplied_out():
    """A product is rounded the same way on every machine; ``pow`` need not be."""
    for tenth in range(0, 51):
        r = tenth / 10
        side = 2.0 * r + 1.0
        assert senses.detection_area(r, 1) == side and senses.detection_area(r, 2) == side * side
        v = 1 + tenth * 0.08
        assert senses.movement_cost(v) == 0.5 * v * v * v
    for speed in range(1, 6):
        assert senses.movement_cost(speed) == 0.5 * speed ** 3 and type(senses.movement_cost(speed)) is float
    for hundredth in range(0, 101):
        p = hundredth / 100
        for n in range(1, 11):
            none = 1.0
            for _ in range(n):
                none *= 1.0 - p
            assert senses.group_detection(p, n) == 1.0 - none
            assert senses.group_detection(p, n) == pytest.approx(1 - (1 - p) ** n, abs=1e-15)
    assert senses.group_detection(0.3, 0) == 0.0 and senses.group_detection(1.0, 3) == 1.0
    for level in range(6):  # the square root of a square is exact
        assert senses.sensor_range(level, 1.0) == level and senses.pooling_gain(level * level) == level


def test_no_attack_answer_sits_on_a_rounding_edge():
    """``expm1`` is the one library function left whose last digit a machine may round differently.
    No answer of an attack lesson, and no ``predator_pressure``, changes when the value moves by
    a part in 10^12."""
    for hundredth in range(0, 31):
        for sense in range(6):
            value = senses.attack_probability(hundredth / 100, sense)
            assert value == pytest.approx(1 - math.exp(-(hundredth / 100) * (2 * sense + 1)), abs=1e-15)
            for factor in (1 - 1e-12, 1 + 1e-12):
                assert num(value * factor) == num(value), (hundredth, sense)
                assert dense_value(sig(value * factor)) == dense_value(sig(value)), (hundredth, sense)


def test_a_dim_light_is_for_noses():
    """Below a light of 0.4 a smelled cell counts more than a seen one, at half the cost per
    receptor: the population senses, and does not see."""
    for dims in (1, 2):
        dim = finals(light=0.3, predators=0.0, dims=dims)
        lit = finals(light=1.0, predators=0.0, dims=dims)
        assert mean(dim, "eyes") < 0.3 and mean(dim, "eyed") < 0.25, dims
        assert mean(dim, "smell") + mean(dim, "touch") > 3 and mean(lit, "eyes") > mean(dim, "eyes") + 2
        assert all(row["final_outcome"] in ("sensing", "brained") and row["stage"] != "seeing" for row in dim)
        assert simulation().run(1, light=0.3, predators=0.0, dims=dims, **WORLD).params["eye_pays"] >= 1, \
            "an eye alone would pay: it is the nose that keeps it away"


def test_clumped_food_speeds_the_first_eye_and_changes_nothing_else():
    """What the docstring says of clumped food: the blind are poorer, so the first eye spreads
    sooner; the level the eyes reach, and the level that pays, are the same."""
    def spread(patchiness: float, dims: int) -> tuple[float, float, set]:
        first, level, pays = [], [], set()
        for seed in range(1, 9):
            r = simulation().run(seed, light=1.0, predators=0.0, dims=dims, **{**WORLD, "patchiness": patchiness})
            first.append(next(t for t, s in enumerate(r.steps) if s["eyed"] >= 0.5))
            level.append(r.steps[-1]["eyes"])
            pays.add(r.params["eye_pays"])
        return statistics.mean(first), statistics.mean(level), pays

    for dims in (1, 2):
        even, clumped = spread(0.0, dims), spread(0.8, dims)
        assert clumped[0] < even[0] - 1, (dims, even, clumped)
        assert abs(clumped[1] - even[1]) < 0.3 and even[1] > 2, (dims, even, clumped)
        assert even[2] == clumped[2] == {3}
    blind = (0, 0, 0, 0, 0, 1, 0)
    world = {**WORLD, "light": 1.0, "dims": 1}
    assert traits(blind, {**world, "patchiness": 0.0}).intake > 4 * traits(blind, {**world, "patchiness": 0.8}).intake
    eyed = (0, 3, 0, 0, 3, 2, 0)  # makes use of 6 cells: more than the 5 items of a clump at patchiness 0.8
    assert traits(eyed, {**world, "patchiness": 0.0}).intake == pytest.approx(
        traits(eyed, {**world, "patchiness": 0.8}).intake, rel=1e-12)


def test_forty_seeds_with_parameters_drawn_at_random_pass_the_gate_twice(sim):
    """Parameters that do not belong to the seed: conserved, and the same on a second run."""
    rng = random.Random(2024)
    seen = set()
    for seed in range(2001, 2041):
        p = random_params(rng)
        start = time.perf_counter()
        a = sim.run(seed, **p)
        took = time.perf_counter() - start
        b = Senses().run(seed, **p)
        assert sim.conserved(a).ok, (seed, p, sim.conserved(a))
        assert a.steps == b.steps and a.generations == b.generations and a.genomes == b.genomes
        assert a.summary == b.summary and a.params == b.params
        assert [repr(v) for s in a.steps for v in s.values()] == [repr(v) for s in b.steps for v in s.values()]
        assert took < 2.0
        for ln in lines(a, every=6):
            assert is_dense(ln.text) and n_tokens(ln.text) <= MAX_TOKENS
        seen.add(a.summary["outcome"])
    assert len(seen) >= 3


def test_stage_lessons_are_not_mostly_arms_race():
    """Drawn evenly, the predators' level is 3 or more half the time, and that alone decides the
    stage: 0.48 of the lessons were arms_race. Over many draws no stage is now above 0.35."""
    rng = random.Random(11)
    counts = collections.Counter()
    for _ in range(20000):
        values = []
        for spec in RULES["stage"].inputs:
            values.append(LESSONS.draw(rng, "stage", spec, values))
        counts[senses.stage_of(*values)] += 1
    shares = {stage: counts[stage] / 20000 for stage in STAGES}
    assert set(counts) == set(STAGES) and max(shares.values()) < 0.35, shares
    assert 0.15 < shares["arms_race"] < 0.25 and shares["blind"] > 0.03, shares


def test_random_params_sit_on_no_rounding_edge():
    """``10 ** x`` comes from the C library, which may round its last digit differently on another
    machine. For every seed the chain can draw (1 to 9999), the oxygen and the body size that
    random_params rounds stay the same when the power moves by a part in 10^12."""
    class Recording(random.Random):
        def __init__(self, seed):
            super().__init__(seed)
            self.drawn = []

        def uniform(self, a, b):
            value = super().uniform(a, b)
            self.drawn.append((a, b, value))
            return value

    for seed in range(1, 10000):
        rng = Recording(seed)
        p = random_params(rng)
        powers = {(a, b): 10 ** value for a, b, value in rng.drawn if (a, b) in ((-2.0, 0.0), (1.0, 4.5))}
        assert set(powers) == {(-2.0, 0.0), (1.0, 4.5)}, seed
        assert p == random_params(random.Random(seed)), seed
        for factor in (1 - 1e-12, 1 + 1e-12):
            assert round(powers[(-2.0, 0.0)] * factor, 3) == p["oxygen"], seed
            assert int(round(powers[(1.0, 4.5)] * factor)) == p["body_size"], seed
