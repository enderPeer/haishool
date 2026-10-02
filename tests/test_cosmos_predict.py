"""Operator lessons must match simulator phases, not just their own checker."""

import random

import numpy as np
import pytest

from haishool.cosmos import chem, gravity, life, nucleo, planets, predict, world
from haishool.student import tokens
from haishool.truth import Gate, is_dense, num, parse_num


@pytest.mark.parametrize("topic", predict.TOPICS)
def test_fresh_bounded_gate_examples(topic):
    gate = predict.gate(topic)
    assert isinstance(gate, Gate)
    first = gate.generate(random.Random(123), 200)
    assert first == gate.generate(random.Random(123), 200)
    assert first != gate.generate(random.Random(456), 200)
    assert len(first) == 200
    for row in first:
        assert row.topic == "predict_" + topic
        assert "seed" not in row.prompt
        assert is_dense(row.text)
        assert len(tokens(row.text)) <= predict.MAX_TOKENS
        assert gate.owns(row.prompt)
        assert gate.check(row.prompt, row.answer).ok
        assert not gate.check(row.prompt, "1 e 9 9 9").ok
        assert not gate.check(row.prompt, "minus 9 9 9 9 9 9").ok
        assert row.meta["split_key"] == predict.split_key(row.prompt)
    for row in gate.records():
        assert is_dense(row.text)
        assert len(tokens(row.text)) <= predict.MAX_TOKENS


def test_nucleo_temperature_matches_rollout():
    rollout = nucleo.run(9, expansion_factor=1.2)
    rule = predict.RULES["cooling_temperature"].compute
    for before, after in zip(rollout.steps, rollout.steps[1:]):
        assert rule(before["temperature"], before["time"], after["time"]) == pytest.approx(after["temperature"])


def test_gravity_cooling_matches_actual_substep(monkeypatch):
    monkeypatch.setattr(gravity, "N_OUT", 2)
    monkeypatch.setattr(gravity, "SUB_STEPS", 1)
    seed, n, spin, cooling = 7, 12, .2, .3
    x, v, masses = gravity.initial_state(seed, n, gravity.MASS, spin, gravity.EPS)
    # Supply the phase the lesson promises: velocities after leapfrog, before drag.
    v += .5 * gravity.DT * gravity._accel(x, masses, gravity.EPS ** 2)
    x += gravity.DT * v
    v += .5 * gravity.DT * gravity._accel(x, masses, gravity.EPS ** 2)
    mean = float(np.dot(masses, v[:, 2]) / gravity.MASS)
    expected = np.array([predict.RULES["cooling_vertical"].compute(
        float(vz), mean, cooling, gravity.DT / gravity.T_FF) for vz in v[:, 2]])
    stream = gravity.evolve(seed, n=n, spin=spin, cooling=cooling)
    next(stream)
    _, actual, _ = next(stream)
    np.testing.assert_allclose(expected, actual[:, 2], atol=1e-14)


def test_planets_sweep_matches_actual_advance(monkeypatch):
    rng = planets._Rng(5)
    disc = planets._Disc(rng, m_star=.9, disc_mass=20, t_gas=3)
    # Remove downstream collision phase: these questions explicitly predict sweep only.
    monkeypatch.setattr(disc, "collide", lambda *args: None)
    dt = .013
    expected = np.array([predict.RULES["solids_remaining"].compute(
        float(rem), float(eps), float(orbit), disc.m_star, dt)
        for rem, eps, orbit in zip(disc.s_rem, disc.eps, disc.a)])
    disc.advance(rng, 0, dt, 10)
    np.testing.assert_allclose(expected, disc.s_rem, atol=1e-14)


def test_chem_schedule_matches_actual_rollout():
    rollout = chem.simulation().run(7, t_start=4000, t_end=300, steps=30, atoms=100)
    for index, state in enumerate(rollout.steps):
        row = predict._line("schedule_temperature", [4000, 300, index, 30])
        assert parse_num(row.answer) == state["temperature"]


def test_life_letter_budget_matches_full_run():
    rollout = life.run(8, monomers=1000, inflow=44, poly_rate=.02, decay=1, mu=.02)
    for before, after in zip(rollout.steps, rollout.steps[1:]):
        total = before["free_monomers"] + before["bound"]
        row = predict._line("next_total", [total, 44])
        assert parse_num(row.answer) == after["free_monomers"] + after["bound"]


def test_life_expectations_are_probabilities_not_random_outcomes():
    survival = predict.RULES["survival_probability"].compute
    mutations = predict.RULES["expected_mutations"].compute
    assert survival(2, 4) == 0
    assert survival(16, 0) == 1
    assert survival(8, 2) == .75
    assert mutations(8, .25) == 2
    assert mutations(16, 1) == 16
    assert mutations(16, 0) == 0


def test_answers_use_printed_inputs_and_integer_counts_are_exact():
    row = predict._line("cooling_temperature", [1234567.89, 1.23456789, 9.87654321])
    name, values = predict._parse(row.prompt)
    assert row.answer == num(predict.RULES[name].compute(*values))
    count = predict._line("next_total", [100, 8])
    assert predict.check(count.prompt, "1 0 8").ok
    assert not predict.check(count.prompt, "1 0 9").ok
    # evolve skips both drag and momentum correction entirely without cooling.
    assert predict.RULES["cooling_vertical"].compute(2, 1, 0, .01) == 2


@pytest.mark.parametrize("prompt", [
    "gravity seed 7 final clumps",
    "life predict expected_mutations length 1 7 mu 0 point 2",
    "life predict expected_mutations length 8 mu 1 e 9 9 9",
    "life predict next_total total 8 inflow 3",
    "chem predict schedule_temperature t_start 1 t_end 2 step 1 steps 1",
    "chem predict schedule_temperature t_start 9 t_end 2 step 2 steps 1",
    "nucleo predict cooling_temperature temperature 5 time 2 next_time 1",
    "life predict expected_mutations length 8 mu 0 point 2 junk 1",
])
def test_rejects_invalid_or_insufficient_inputs(prompt):
    assert not predict.owns(prompt)
    assert not predict.check(prompt, "0").ok


def test_canonical_split_identity_and_world_ownership():
    a = "life predict expected_mutations length 8 mu 0 point 2"
    b = "life predict expected_mutations length 8 point 0 mu 2 e minus 1"
    assert predict.split_key(a) == predict.split_key(b)
    source = world.handoff_line("temperature", 1, 1)
    translated = source.prompt.replace("world handoff", "world predict handoff")
    assert predict.split_key(source.prompt) == predict.split_key(translated)
    assert not world.owns(translated)
    assert predict.owns(translated)
    assert predict.check(translated, source.answer).ok
    assert not predict.gate("life").check(translated, source.answer).ok


def test_sealed_input_exclusion_with_independent_rng_streams():
    sealed = predict.generate(random.Random(202), 200)
    keys = {predict.split_key(row.prompt) for row in sealed}
    training = [row for row in predict.generate(random.Random(101), 1000)
                if predict.split_key(row.prompt) not in keys]
    assert training
    assert not keys.intersection(predict.split_key(row.prompt) for row in training)
