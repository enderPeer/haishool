"""Semantic leakage regressions shared by builds, training and feedback."""

import random

import pytest

from haishool.student import _held_key, split_extra
from haishool.truth import Line, loop, maths
from haishool.truth.splits import canonical_prompt


@pytest.mark.parametrize("prompt,expected", [
    ("q calc 1 2 plus 7. a 1 9.", "calc 1 2 plus 7"),
    ("calc 1 2 plus 7 steps", "calc 1 2 plus 7"),
    ("check 1 2 plus 7 equals 1 9", "calc 1 2 plus 7"),
    ("check 1 2 plus 7 equals 2 0", "calc 1 2 plus 7"),
    ("check 3 x plus 4 equals 1 9 x 5", "solve 3 x plus 4 equals 1 9"),
    ("check iron protons 2 6", "iron protons"),
    ("check water formula h 2 o 2", "water formula"),
    ("check mass_percent water oxygen 8 8 point 8 1", "mass_percent water oxygen"),
    ("check weight m 5 equals 4 9 point 0 5", "weight m 5"),
    ("weight m 5 steps", "weight m 5"),
    ("molar_mass water steps", "molar_mass water"),
    ("judge check 1 2 plus 7 equals 2 0 answer no", "calc 1 2 plus 7"),
    ("judge calc 1 2 plus 7 steps answer bad_work", "calc 1 2 plus 7"),
    ("judge calc 1 2 plus 7 answer", "calc 1 2 plus 7"),
    ("check iron heavier_than carbon", "check iron heavier_than carbon"),
    ("unknown question", "unknown question"),
    ("", ""),
])
def test_question_families_share_identity(prompt, expected):
    assert canonical_prompt(prompt) == expected


def test_student_keeps_worked_check_and_judge_answers_with_calculation():
    lines = [
        "q calc 1 2 plus 7. a 1 9.",
        "q calc 1 2 plus 7 steps. a then result 1 9.",
        "q check 1 2 plus 7 equals 1 9. a yes.",
        "q check 1 2 plus 7 equals 2 0. a no.",
        "q judge calc 1 2 plus 7 steps answer then result 1 9. a wrong.",
    ]
    assert {_held_key(line) for line in lines} == {"q calc 1 2 plus 7"}
    for seed in range(10):
        train, held = split_extra(lines, 0.5, seed)
        assert (train == lines and held == []) or (held == lines and train == [])


@pytest.mark.parametrize("excluded,candidate", [
    ("calc 1 2 plus 7", "check 1 2 plus 7 equals 1 9"),
    ("check 1 2 plus 7 equals 2 0", "calc 1 2 plus 7"),
    ("calc 1 2 plus 7 steps", "calc 1 2 plus 7"),
    ("judge calc 1 2 plus 7 answer 1 9", "calc 1 2 plus 7 steps"),
])
def test_feedback_excludes_equivalent_questions(monkeypatch, excluded, candidate):
    gate = maths.gate()
    answer = gate.check(candidate, "").expected
    monkeypatch.setattr(gate, "generate", lambda rng, n: [Line(candidate, answer, "maths", "calc")])
    asked = []
    result = loop.run(asked.append, [gate], random.Random(1), 1, {excluded}, None)
    assert asked == []
    assert result["questions"]["maths"]["asked"] == 0
    assert result["questions"]["maths"]["excluded"] == loop.MAX_ROUNDS
