"""Shared canonical trace checks and exact decimal rendering for worked lessons.

Like maths' column-method lessons, every segment is required and checked, including
the substitution and intermediate results. A correct final value alone cannot pass.
"""

from decimal import Decimal, localcontext
from fractions import Fraction

from haishool.truth import Verdict


def check_steps(expected: str, answer: str) -> Verdict:
    """Judge each canonical segment and identify the first incorrect/missing step."""
    want = expected.split(" then ")
    got = " ".join(answer.split()).split(" then ")
    for i, (actual, gold) in enumerate(zip(got, want), 1):
        if actual != gold:
            return Verdict(False, expected, f"wrong step {i}: {gold}")
    if len(got) != len(want):
        return Verdict(False, expected, f"expected {len(want)} steps, got {len(got)}")
    return Verdict(True, expected)


def decimal_tokens(value: Fraction) -> str:
    """An exact terminating decimal, without intermediate rounding or float noise."""
    denominator = value.denominator
    for prime in (2, 5):
        while denominator % prime == 0:
            denominator //= prime
    if denominator != 1:
        raise ValueError("worked decimal must terminate")
    with localcontext() as context:
        context.prec = len(str(abs(value.numerator))) + len(str(value.denominator)) + 2
        plain = format(Decimal(value.numerator) / Decimal(value.denominator), "f")
    if "." in plain:
        plain = plain.rstrip("0").rstrip(".")
    return " ".join({"-": "minus", ".": "point"}.get(c, c) for c in plain)
