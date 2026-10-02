"""The calculation curriculum must validate the work, not just its final answer."""

import random
from collections import Counter
from fractions import Fraction

import pytest

from haishool.truth import is_dense
from haishool.truth.forces import CALC, WORKED_MAX_TOKENS as FORCE_LIMIT, gate as forces, n_tokens
from haishool.truth.substances import WORKED_MAX_TOKENS as SUBSTANCE_LIMIT, gate as substances
from haishool.truth.worked import decimal_tokens


@pytest.fixture(scope="module", params=[forces, substances])
def curriculum(request):
    return request.param()


def test_worked_generation_is_deterministic_bounded_and_checks_every_step(curriculum):
    lines = curriculum.generate_worked(random.Random(814), 300)
    assert lines == curriculum.generate_worked(random.Random(814), 300)
    assert len({line.prompt for line in lines}) == 300
    limit = FORCE_LIMIT if curriculum.topic == "forces" else SUBSTANCE_LIMIT
    for line in lines:
        assert is_dense(line.text) and n_tokens(line.text) <= limit
        assert curriculum.owns(line.prompt)
        assert curriculum.check(line.prompt, line.answer).ok
        base = line.meta["base_prompt"]
        result = line.answer.split(" then result ")[-1]
        assert curriculum.check(base, result).ok
        assert not curriculum.check(line.prompt, result).ok
        steps = line.answer.split(" then ")
        for i in range(len(steps)):
            corrupted = steps.copy()
            corrupted[i] += " wrong"
            verdict = curriculum.check(line.prompt, " then ".join(corrupted))
            assert not verdict.ok
            assert f"step {i + 1}" in verdict.reason
        assert not curriculum.check(line.prompt, " then ".join(steps[:-1])).ok
        assert not curriculum.check(line.prompt, line.answer + " then result " + result).ok
    heads = Counter(line.prompt.split()[0] for line in lines)
    assert set(heads) == (set(CALC) if curriculum.topic == "forces" else {"molar_mass", "mass_percent"})


def test_short_context_is_an_explicit_subset(curriculum):
    lines = curriculum.generate_worked(random.Random(33), 100, max_tokens=96)
    assert len(lines) == 100
    assert all(n_tokens(line.text) <= 96 for line in lines)
    assert curriculum.generate_worked(random.Random(1), 0) == []
    with pytest.raises(ValueError, match="within 1 tokens"):
        curriculum.generate_worked(random.Random(1), 1, max_tokens=1)


@pytest.mark.parametrize(("prompt", "answer"), [
    ("weight m 5 on moon steps", "formula m times g then substitute 5 times 1 point 6 2 then result 8 point 1"),
    ("centripetal m 2 v 3 r 1 point 5 steps", "formula m times v squared over r then substitute 2 times 3 squared over 1 point 5 then result 1 2"),
    ("net_force f 1 0 f 2 0 opposite steps", "formula abs open f_1 minus f_2 close then substitute abs open 1 0 minus 2 0 close then result 1 0"),
    ("escape_speed m 0 r 1 steps", "formula sqrt open 2 times big_g times m over r close then substitute sqrt open 2 times 6 point 6 7 4 e minus 1 1 times 0 over 1 close then result 0"),
])
def test_force_explicit_formula_and_substitution(prompt, answer):
    assert forces().check(prompt, answer).ok


def test_signed_charge_and_constants_are_retained():
    gate = forces()
    prompt = "coulomb_force charge minus 1 e minus 6 charge 2 e minus 6 r 0 point 1 steps"
    answer = gate.expected(prompt)[0]
    assert "substitute 8 point 9 8 8 e 9 times minus 1 e minus 6 times 2 e minus 6" in answer
    assert answer.endswith("result minus 1 point 7 9 8")
    assert not gate.check(prompt, answer.replace("minus 1 e minus 6", "1 e minus 6")).ok


@pytest.mark.parametrize("prompt", [
    "pressure f 1 area 0 steps", "weight m 5 on pluto steps", "gravity_force m 1 m 2 r 0 steps",
    "orbital_speed m 1 e 9 9 r 1 steps", "acceleration f 1 m minus 1 steps",
    "tidal m 1 m 2 r 3 d 1 e 9 9 9 steps", "spring_force k 1 x 2 bogus steps",
])
def test_force_domain_failures_are_not_training_truth(prompt):
    verdict = forces().check(prompt, "result 0")
    assert not verdict.ok and verdict.expected is None


def test_water_work_is_exact_before_final_rounding():
    gate = substances()
    expected = (
        "formula sum count times atomic_weight then h 2 times 1 point 0 0 8 is 2 point 0 1 6 "
        "then o 1 times 1 5 point 9 9 9 is 1 5 point 9 9 9 "
        "then sum 2 point 0 1 6 plus 1 5 point 9 9 9 is 1 8 point 0 1 5 then result 1 8 point 0 1 5"
    )
    assert gate.check("molar_mass water steps", expected).ok
    assert gate.check("molar_mass o 1 h 2 steps", expected).ok
    assert not gate.check("molar_mass water steps", expected.replace("is 2 point 0 1 6", "is 2 point 0 1 5")).ok
    percentage = gate.worked_answer("mass_percent water oxygen".split())
    assert "substitute 1 0 0 times 1 5 point 9 9 9 over 1 8 point 0 1 5" in percentage
    assert percentage.endswith("result 8 8 point 8 1")
    assert gate.check("mass_percent water oxygen steps", percentage).ok


def test_absent_element_has_zero_percentage_and_valid_work():
    gate = substances()
    answer = gate.worked_answer("mass_percent water carbon".split())
    assert "substitute 1 0 0 times 0 over 1 8 point 0 1 5" in answer
    assert answer.endswith("result 0")
    assert gate.check("mass_percent water carbon steps", answer).ok


@pytest.mark.parametrize("prompt", ["molar_mass air steps", "molar_mass xx 2 steps", "molar_mass h 0 steps",
                                         "mass_percent water nobody steps", "mass_percent water oxygen extra steps"])
def test_unknown_or_mixture_formula_is_not_judged(prompt):
    verdict = substances().check(prompt, "result 0")
    assert not verdict.ok and verdict.expected is None


def test_decimal_intermediates_are_exact_and_ignore_decimal_context():
    from decimal import localcontext
    with localcontext() as context:
        context.prec = 2
        assert decimal_tokens(Fraction("18.998403163")) == "1 8 point 9 9 8 4 0 3 1 6 3"
        assert decimal_tokens(Fraction("2.016")) == "2 point 0 1 6"
        assert decimal_tokens(Fraction(0)) == "0"
    with pytest.raises(ValueError, match="terminate"):
        decimal_tokens(Fraction(1, 3))
