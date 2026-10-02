import random
import re
from collections import Counter
from fractions import Fraction

import pytest
import sympy

from haishool.truth import DIGITS, Gate, Verdict, is_dense, num, parse_num
from haishool.truth.maths import MIX, Maths, gate

X = sympy.Symbol("x")
SEED_LINES = 3000


def ntokens(text: str) -> int:
    return len(text.replace(".", " . ").split())


def as_rational(dense: str) -> sympy.Rational:
    """``minus 1 2 point 5`` -> Rational(-25, 2), read independently of the gate."""
    return sympy.Rational("".join({"point": ".", "minus": "-"}.get(w, w) for w in dense.split()))


def to_sympy(tokens: list[str]) -> sympy.Expr:
    """An independent reading of a generated dense expression (numbers as exact rationals)."""
    out, i, close_after_number = [], 0, 0
    words = {"plus": "+", "minus": "-", "times": "*", "over": "/", "power": "**", "open": "(", "close": ")",
             "x": "x", "percent": "/100*", "of": ""}
    while i < len(tokens):
        t = tokens[i]
        if t in DIGITS:
            j = i
            while j < len(tokens) and tokens[j] in DIGITS:
                j += 1
            text = "".join(tokens[i:j])
            if j < len(tokens) and tokens[j] == "point":
                k = j + 1
                while k < len(tokens) and tokens[k] in DIGITS:
                    k += 1
                text += "." + "".join(tokens[j + 1:k])
                j = k
            out.append(f"Rational('{text}')" + ")" * close_after_number)
            close_after_number = 0
            i = j
            if i < len(tokens) and tokens[i] == "x":
                out.append("*x")
                i += 1
            continue
        if t == "root":
            out.append("sqrt(")
            close_after_number += 1
        else:
            out.append(words[t])
        i += 1
    return sympy.sympify(" ".join(out), locals={"x": X})


@pytest.fixture(scope="module")
def g() -> Maths:
    return gate()


@pytest.fixture(scope="module")
def lines(g):
    return g.generate(random.Random(20261001), SEED_LINES)


def test_protocol_and_records(g):
    assert isinstance(g, Gate)
    assert g.topic == "maths" and g.records() == []
    assert set(g.KEYS) >= {"calc", "solve", "compare", "check"}


def test_same_seed_same_lines(g, lines):
    again = g.generate(random.Random(20261001), SEED_LINES)
    assert [ln.text for ln in again] == [ln.text for ln in lines]
    other = g.generate(random.Random(7), SEED_LINES)
    assert [ln.text for ln in other] != [ln.text for ln in lines]


def test_lines_are_dense_short_and_distinct(g, lines):
    assert len(lines) == SEED_LINES
    assert len({ln.prompt for ln in lines}) == SEED_LINES
    for ln in lines:
        assert is_dense(ln.text), ln.text
        assert ntokens(ln.text) <= 80, ln.text
        assert ln.topic == "maths" and ln.kind in ("calc", "yesno")
        assert ln.kind == ("yesno" if ln.prompt.startswith("check ") else "calc")
        assert "." not in ln.prompt and "." not in ln.answer


def test_mix_of_forms(g):
    many = g.generate(random.Random(3), 6000)
    forms = Counter(ln.meta["form"] for ln in many)
    for form, share in MIX:
        assert abs(forms[form] / len(many) - share) < 0.03, (form, forms[form])
    checks = [ln for ln in many if ln.kind == "yesno"]
    yes = sum(ln.answer == "yes" for ln in checks)
    assert 0.4 < yes / len(checks) < 0.6
    assert any(" x " in ln.prompt for ln in checks) and any(" x " not in ln.prompt for ln in checks)
    compares = Counter(ln.answer for ln in many if ln.meta["form"] == "compare")
    assert set(compares) == {"greater", "less", "equal"}


def test_many_distinct_lines_over_seeds(g):
    prompts = set()
    for seed in range(1, 7):
        prompts |= {ln.prompt for ln in g.generate(random.Random(seed), 5000)}
    assert len(prompts) >= 20000


def calc_family(body: list[str]) -> str:
    """The family of a generated calc prompt, read from its words alone."""
    if "point" in body:
        return "decimal"
    if "percent" in body:
        return "percent"
    ops = [t for i, t in enumerate(body) if t in ("plus", "times", "over", "power")
           or (t == "minus" and i > 0 and (body[i - 1] in DIGITS or body[i - 1] == "close"))]
    if "open" in body or len(ops) >= 2:
        return "two"
    return "root" if body[0] == "root" else "plus_minus" if ops[0] in ("plus", "minus") else ops[0]


def test_distribution_over_ten_seeds(g):
    """The documented shares inside each form (within 5 points), and no flood of trivial answers."""
    lines = [ln for seed in range(1, 11) for ln in g.generate(random.Random(seed), 3000)]
    assert len({ln.text for ln in lines}) >= 20000
    by_form: dict[str, list] = {form: [] for form, _ in MIX}
    for ln in lines:
        assert is_dense(ln.text) and ntokens(ln.text) <= 80 and "." not in ln.prompt and "." not in ln.answer, ln.text
        by_form[ln.meta["form"]].append(ln)
    for form, share in MIX:
        assert abs(len(by_form[form]) / len(lines) - share) < 0.05, form

    def shares(counter: Counter) -> dict[str, float]:
        return {k: v / sum(counter.values()) for k, v in counter.items()}

    def close(got: dict[str, float], want: dict[str, float]) -> None:
        assert set(got) == set(want), got
        for k, share in want.items():
            assert abs(got[k] - share) < 0.05, (k, got[k], share)

    close(shares(Counter(calc_family(ln.prompt.split()[1:]) for ln in by_form["calc"])),
          {"plus_minus": 0.30, "times": 0.15, "over": 0.10, "power": 0.07, "root": 0.06, "percent": 0.08, "two": 0.14,
           "decimal": 0.10})
    steps = Counter()
    for ln in by_form["steps"]:
        body = ln.prompt.split()[1:-1]
        op = "plus" if "plus" in body else "times"
        steps[op if op == "plus" else "times one" if len(body) - body.index(op) == 2 else "times two"] += 1
    close(shares(steps), {"plus": 0.60, "times one": 0.28, "times two": 0.12})
    compare = by_form["compare"]
    close(shares(Counter("decimal" if "point" in ln.prompt else "integer" for ln in compare)), {"decimal": 0.5, "integer": 0.5})
    kinds = Counter()
    for ln in compare:
        a, b = (as_rational(side) for side in ln.prompt[len("compare "):].split(" with "))
        kinds["equal" if a == b else "near" if abs(a - b) in [sympy.Rational(10) ** k for k in range(-3, 3)] else "far"] += 1
        assert ln.answer == ("equal" if a == b else "greater" if a > b else "less")
    got = shares(kinds)
    assert abs(got["equal"] - 0.15) < 0.05 and got["near"] > 0.3 and got["far"] > 0.3, got
    answers = shares(Counter(ln.answer for ln in compare))
    assert abs(answers["greater"] - answers["less"]) < 0.08, answers
    checks = by_form["check"]
    close(shares(Counter("equation" if " x " in ln.prompt else "expression" for ln in checks)), {"expression": 0.6, "equation": 0.4})
    for part in ("equation", "expression"):
        some = [ln for ln in checks if (" x " in ln.prompt) == (part == "equation")]
        assert abs(sum(ln.answer == "yes" for ln in some) / len(some) - 0.5) < 0.05, part
    # answers are not dominated by trivial cases
    for form, most_single, most_common in (("calc", 0.10, 0.02), ("solve", 0.50, 0.05)):
        got = [ln.answer for ln in by_form[form]]
        single = sum(len([w for w in a.split() if w in DIGITS]) == 1 for a in got) / len(got)
        assert single < most_single, (form, single)
        assert Counter(got).most_common(1)[0][1] / len(got) < most_common, (form, Counter(got).most_common(3))
    assert len({ln.answer for ln in by_form["calc"]}) > 4000
    assert {parse_num(ln.answer[2:]) for ln in by_form["solve"]} >= set(range(-20, 21))


def test_same_lines_in_another_process():
    """No dependence on the hash seed of the process: sets are only used for membership."""
    import hashlib
    import os
    import subprocess
    import sys
    code = ("import random, hashlib; from haishool.truth.maths import gate; "
            "print(hashlib.sha256(''.join(ln.text for ln in gate().generate(random.Random(4), 1500)).encode()).hexdigest())")
    here = hashlib.sha256("".join(ln.text for ln in gate().generate(random.Random(4), 1500)).encode()).hexdigest()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for hash_seed in ("1", "987"):
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=root,
                             env={**os.environ, "PYTHONHASHSEED": hash_seed})
        assert out.stdout.strip() == here, out.stderr


def test_gate_agrees_with_its_own_answers(g, lines):
    for ln in lines:
        assert g.owns(ln.prompt), ln.prompt
        v = g.check(ln.prompt, ln.answer)
        assert v.ok and v.expected == ln.answer, (ln.text, v)


def mutate(answer: str, rng: random.Random) -> str:
    """One changed digit or word: never the same value."""
    if answer in ("yes", "no"):
        return "no" if answer == "yes" else "yes"
    if answer in ("greater", "less", "equal"):
        return rng.choice([w for w in ("greater", "less", "equal") if w != answer])
    words = answer.split()
    i = rng.choice([i for i, w in enumerate(words) if w in DIGITS])
    words[i] = str((int(words[i]) + rng.randint(1, 9)) % 10)
    return " ".join(words)


def test_gate_rejects_wrong_answers(g, lines):
    rng = random.Random(5)
    for ln in lines:
        wrong = mutate(ln.answer, rng)
        v = g.check(ln.prompt, wrong)
        assert not v.ok and v.expected == ln.answer and v.reason, (ln.text, wrong, v)
    # a wrong sign, a missing x, words instead of a number
    assert not g.check("calc 1 2 plus 7", "minus 1 9").ok
    assert not g.check("solve 3 x plus 4 equals 1 9", "5").ok
    assert not g.check("calc 1 2 plus 7", "nineteen").ok
    assert not g.check("compare 3 with 4", "yes").ok


@pytest.mark.parametrize("prompt,answer", [
    ("calc 1 2 plus 7", "1 9"),
    ("calc 1 5 percent of 2 0 0", "3 0"),
    ("calc 2 plus 3 times 4", "1 4"),
    ("calc open 2 plus 3 close times 4", "2 0"),
    ("calc 2 times open 3 plus 4 close", "1 4"),
    ("calc 1 0 0 over 4", "2 5"),
    ("calc 5 power 3", "1 2 5"),
    ("calc minus 3 power 2", "9"),
    ("calc 2 minus 3 power 2", "minus 7"),
    ("calc root 1 4 4", "1 2"),
    ("calc root 8 1 plus 2", "1 1"),
    ("calc 5 minus minus 3", "8"),
    ("calc minus 1 2 3 4 plus 5 6", "minus 1 1 7 8"),
    ("calc 1 2 point 5 minus 3 point 7 5", "8 point 7 5"),
    ("calc 0 point 1 plus 0 point 2", "0 point 3"),
    ("calc 3 point 5 minus 3 point 5", "0"),
    ("calc 9 9 9 times 9 9", "9 8 9 0 1"),
    ("solve 3 x plus 4 equals 1 9", "x 5"),
    ("solve 5 x minus 3 equals 2 x plus 9", "x 4"),
    ("solve x over 2 equals 7", "x 1 4"),
    ("solve minus x plus 3 equals 1", "x 2"),
    ("solve 2 x equals minus 8", "x minus 4"),
    ("solve open x plus 3 close times 2 equals 1 0", "x 2"),
    ("compare 3 point 5 with minus 2", "greater"),
    ("compare minus 3 with minus 5", "greater"),
    ("compare 0 point 5 with 0 point 4 5", "greater"),
    ("compare 0 point 0 9 with 0 point 1", "less"),
    ("compare 7 with 7", "equal"),
    ("check 1 2 plus 7 equals 2 0", "no"),
    ("check 1 2 plus 7 equals 1 9", "yes"),
    ("check 3 x plus 4 equals 1 9 x 5", "yes"),
    ("check 3 x plus 4 equals 1 9 x 6", "no"),
    ("check 2 x plus 1 equals minus 5 x minus 3", "yes"),
    ("calc 4 7 plus 3 8 steps",
     "7 plus 8 is 1 5 write 5 carry 1 then 4 plus 3 plus 1 is 8 write 8 then result 8 5"),
    ("calc 9 5 plus 8 steps",
     "5 plus 8 is 1 3 write 3 carry 1 then 9 plus 0 plus 1 is 1 0 write 0 carry 1 then result 1 0 3"),
    ("calc 4 7 times 6 steps",
     "7 times 6 is 4 2 write 2 carry 4 then 4 times 6 is 2 4 plus 4 is 2 8 write 8 carry 2 then result 2 8 2"),
    ("calc 4 7 times 3 6 steps",
     "4 7 times 3 0 is 1 4 1 0 then 4 7 times 6 is 2 8 2 then 1 4 1 0 plus 2 8 2 is 1 6 9 2 then result 1 6 9 2"),
])
def test_known_answers(g, prompt, answer):
    assert g.owns(prompt)
    v = g.check(prompt, answer)
    assert v.ok and v.expected == answer, v


def test_steps_are_checked_step_by_step(g):
    prompt = "calc 4 7 plus 3 8 steps"
    right = "7 plus 8 is 1 5 write 5 carry 1 then 4 plus 3 plus 1 is 8 write 8 then result 8 5"
    lost_carry = "7 plus 8 is 1 5 write 5 then 4 plus 3 is 7 write 7 then result 7 5"
    wrong_step_right_result = "7 plus 8 is 1 4 write 5 carry 1 then 4 plus 3 plus 1 is 8 write 8 then result 8 5"
    right_steps_wrong_result = "7 plus 8 is 1 5 write 5 carry 1 then 4 plus 3 plus 1 is 8 write 8 then result 8 6"
    assert g.check(prompt, right).ok
    v = g.check(prompt, lost_carry)
    assert not v.ok and v.reason.startswith("wrong step 1") and v.expected == right
    v = g.check(prompt, wrong_step_right_result)
    assert not v.ok and v.reason.startswith("wrong step 1")
    v = g.check(prompt, right_steps_wrong_result)
    assert not v.ok and v.reason.startswith("wrong result")
    assert not g.check(prompt, "8 5").ok
    assert not g.check(prompt, right + " then result 8 5").ok
    # outside the supported sizes the gate does not own the prompt
    assert not g.owns("calc 1 2 3 4 5 plus 1 steps")
    assert not g.owns("calc 1 2 3 4 times 1 2 steps")
    assert not g.owns("calc 1 2 times 1 2 3 steps")
    assert not g.owns("calc 1 2 point 5 plus 1 steps")
    assert not g.owns("calc minus 1 2 plus 1 steps")


def test_exact_values_have_no_tolerance(g):
    # maths is exact: a neighbouring value is wrong however close
    assert not g.check("calc 1 0 0 0 plus 1", "1 0 0 2").ok
    assert not g.check("calc 1 2 point 5 plus 0 point 2 5", "1 2 point 7 4").ok
    # e notation is no excuse: the stated value must be the value
    assert g.check("calc 1 0 0 0 times 1 0 0 0", "1 e 6").ok
    assert g.check("calc 1 2 3 4 5 6 times 1 0", "1 point 2 3 4 5 6 e 6").ok
    assert g.check("calc 1 over 8", "1 point 2 5 e minus 1").ok
    for prompt, close in [("calc 1 2 3 4 5 6 times 1 0", "1 point 2 3 4 6 e 6"), ("calc 1 2 plus 7", "1 point 9 0 5 e 1"),
                          ("calc 1 0 0 0 times 1 0 0 0", "1 point 0 0 4 e 6"), ("calc 0 times 5", "1 e minus 9"),
                          ("solve 2 x equals 4 0 0 0", "x 2 point 0 0 1 e 3")]:
        v = g.check(prompt, close)
        assert not v.ok and v.expected and v.reason, (prompt, close)
    # another spelling of the same value is still that value
    for same in ("3", "3 point 0", "3 e 0", "3 0 e minus 1", "0 point 3 e 1"):
        assert g.check("calc 1 plus 2", same).ok, same
    assert g.check("calc 1 over 2", "0 point 5 0").ok and g.check("calc 3 minus 3", "minus 0").ok


def test_a_value_that_does_not_terminate_takes_its_correct_rounding(g):
    v = g.check("calc 7 over 3", "2 point 3 3 3")
    assert v.ok and v.expected == "2 point 3 3 3 3 3 3 3 3 3"
    assert g.check("calc 7 over 3", v.expected).ok and g.check("calc 7 over 3", "2 point 3 3").ok
    for wrong in ("2 point 3", "2", "2 point 3 3 4", "2 point 3 3 3 4", "2 point 3 4", "2 point 3 3 3 3 3 3 3 3 4",
                  "minus 2 point 3 3 3"):
        assert not g.check("calc 7 over 3", wrong).ok, wrong
    # rounded, not cut off: two thirds is 0.667
    assert g.check("calc 2 over 3", "0 point 6 6 7").ok and g.check("calc 2 over 3", "6 point 6 6 6 7 e minus 1").ok
    assert not g.check("calc 2 over 3", "0 point 6 6 6").ok and not g.check("calc 2 over 3", "0 point 6 7").ok
    assert g.check("solve 3 x equals minus 2", "x minus 0 point 6 6 6 7").ok
    assert not g.check("solve 3 x equals minus 2", "x 0 point 6 6 6 7").ok
    # the notation of num(): e below 0.001 and from a million on
    assert g.check("calc 1 over 3 0 0 0 0", "0").expected == "3 point 3 3 3 3 3 3 3 3 3 e minus 5"
    assert g.check("calc 2 0 0 0 0 0 0 0 over 3", "0").expected == "6 point 6 6 6 6 6 6 6 6 7 e 6"
    assert g.check("calc 2 0 0 0 0 0 0 over 3", "0").expected == "6 6 6 6 6 6 point 6 6 6 7"
    assert g.check("calc 2 0 0 0 0 0 0 over 3", "6 6 6 6 6 7").ok and not g.check("calc 2 0 0 0 0 0 0 over 3", "6 6 6 6 6 6").ok


def test_rounded_answers_match_num_and_are_accepted(g):
    """The canonical 10 digits are rounded exactly, agree with num() and pass the gate's own check."""
    rng = random.Random(8)
    n = 0
    while n < 400:
        p, q, shift = rng.randint(1, 10 ** rng.randint(1, 9)), rng.choice([3, 7, 9, 11, 13, 97, 331]), rng.randint(0, 9)
        if p % q == 0:
            continue
        value = Fraction(p, q * 10 ** shift) * rng.choice([1, -1])
        prompt = f"calc {'minus ' if value < 0 else ''}{num(p)} over {num(q * 10 ** shift)}"
        expected = g.check(prompt, "0").expected
        assert expected == num(float(value), sig=10), prompt
        assert g.check(prompt, expected).ok, prompt
        assert abs(Fraction(str(as_rational(expected))) - value) <= abs(value) / 10 ** 9, prompt
        words = expected.split()
        last = max(i for i, w in enumerate(words) if w in DIGITS and (" e " not in expected or i < words.index("e")))
        words[last] = str(int(words[last]) + 1) if words[last] != "9" else "8"  # one unit in the last place
        assert not g.check(prompt, " ".join(words)).ok, (prompt, words)
        n += 1
    # far outside the range of a float the answer is still exact arithmetic
    big = "open " * 4 + "1 0" + " power 3 close" * 4 + " power 3"  # 10 ** 243
    tiny = f"calc 1 over 3 over {big} over {big}"
    assert g.check(tiny, "0").expected == "3 point 3 3 3 3 3 3 3 3 3 e minus 4 8 7"
    assert g.check(tiny, "3 point 3 3 e minus 4 8 7").ok and not g.check(tiny, "0").ok
    huge = f"calc {big} times {big} over 3"
    assert g.check(huge, "0").expected == "3 point 3 3 3 3 3 3 3 3 3 e 4 8 5"
    assert g.check(huge, "3 point 3 3 3 e 4 8 5").ok and not g.check(huge, "3 point 3 3 3 e 4 8 4").ok


@pytest.mark.parametrize("answer", [
    "0 3", "0 0 3", "3 point", "point 3", "3 point point 0", "minus minus 3", "3 minus", "minus", "e", "3 e", "e 3",
    "3 e e 0", "3 e 0 0", "3 e 1 point 5", "3 e minus", "3 x", "x 3", "three", "3 3", "", " ", "3 plus 0", "3 . 0",
    "\u0663", "\uff13", "3\u00a0point 0 1", "3 e 9 9 9 9", "3 e 9 9 9 9 9 9 9 9 9 9 9 9", "3 e minus 9 9 9 9 9 9 9 9 9 9",
])
def test_an_answer_must_be_written_as_a_number(g, answer):
    """Leading zeros, a bare point, a huge exponent: clean verdicts, quickly (no 10 ** 999999999)."""
    v = g.check("calc 1 plus 2", answer)
    assert v == Verdict(False, "3", "not a number") or v == Verdict(False, "3", "expected 3"), v
    assert not g.check("solve x minus 1 equals 2", "x " + answer).ok


def test_long_answers_and_stacked_powers_end_quickly(g):
    assert g.check("calc 1 plus 2", " ".join("3" + "0" * 6000)) == Verdict(False, "3", "not a number")
    assert not g.check("calc 1 plus 2", "3 point " + "0 " * 6000 + "1").ok
    assert g.check("calc 1 plus 2", "3 point " + "0 " * 3000).ok
    # nine cubed nineteen times over would have billions of digits: outside the grammar
    for k in (5, 6, 12, 19):
        prompt = "calc " + "open " * k + "9" + " power 3 close" * k + " power 3"
        assert len(prompt.split()) <= 80
        assert g.owns(prompt) is False and g.check(prompt, "1") == Verdict(False, None, "not my question")
    # up to about 300 digits a power is still computed, exactly
    prompt = "calc " + "open " * 4 + "9" + " power 3 close" * 4 + " power 3"
    v = g.check(prompt, "1")
    assert v.expected == num(9 ** 243) and g.check(prompt, v.expected).ok
    assert g.check("calc " + "9 " * 70 + "power 3", num((10 ** 70 - 1) ** 3)).ok


@pytest.mark.parametrize("prompt", [
    "turkey capital", "q spoon color", "carbon protons", "water molar_mass", "gravity seed 7 step 2 0 clumps",
    "q calc 1 2 plus 7", "", "calc", "calc 1 plus", "calc plus 1", "calc 5 over 0", "calc 5 over x",
    "calc root 2", "calc root minus 4", "calc 2 power 4", "calc 2 power minus 1", "calc x plus 1",
    "calc 0 5 plus 1", "calc 1 2 plus 7 equals 1 9", "calc open 1 plus 2", "calc 1 plus 2 close",
    "solve 3 plus 4 equals 7", "solve 3 x plus 4", "solve x times x equals 4", "solve 2 x equals 4 equals 4",
    "compare 1 2 3", "compare 1 with", "compare with 1", "check 1 plus 1", "check 1 plus 1 equals 2 x",
    "check 1 plus 1 equals 2 x y", "calc 1 2 plus 7 steps extra", "calc 1 plus 2 plus 3 steps",
    "calc 1 point", "calc point 5", "calc 1 2 . a 3",
    # other topics, other alphabets, other cases
    "h 2 o 1 molar_mass", "balance h 2 plus o 2", "force mass 2 times 3", "judge calc 1 plus 2 given 3",
    "calc \u0661 plus \u0662", "calc \uff11 plus \uff12", "CALC 1 plus 2", "Calc 1 Plus 2", "calc 1 plus 2.", "calc 1 + 2",
    "calc 1 plus two", "check 1 plus 1 equals two", "check foo", "check 3 x plus 4 equals 1 9 x five", "calc 1 e 5",
    # double operators, division by zero in disguise, powers of powers, x where it cannot be
    "calc 1 plus plus 1", "calc 1 times times 2", "calc 1 times over 2", "calc 0 over 0", "calc 1 over open 2 minus 2 close",
    "calc 1 over 0 x", "calc 2 power 3 power 2", "calc 2 power 0 3", "calc 2 power 1 0", "calc 2 power 2 point 0",
    "calc x power 0", "calc 3 x", "calc open close", "calc 1 point 5 point 5", "calc 3 percent", "calc percent of 3",
    "calc 1 0 percent of 5 0 percent of 2 0 0", "calc x percent of 5", "calc root x", "calc 1 open 2 close",
    "solve x equals x", "solve x plus 1 equals x plus 2", "solve 0 x equals 0", "solve equals 1", "solve x equals",
    "solve x over 0 equals 1", "solve 1 over x equals 1", "compare 1 with 1 with 1", "compare x with 1", "compare 1 with x",
    "compare 1 equals 1", "check 1 with 1", "check equals", "check 1 equals", "check 1 equals 1 equals 1",
    # a check with a value for x needs an equation in x, and exactly one value
    "check 1 plus 1 equals 2 x 5", "check 2 x equals 2 x", "check 2 x equals 4 x 2 x 2", "check 2 x equals 4 x 2 plus 0",
    "check 2 x equals 4 x minus", "check 2 x equals 4 x 0 2", "check 2 x equals 4 x 2 point",
    # steps outside the documented sizes and shapes
    "calc 5 times 0 steps", "calc 1 2 times 1 0 0 steps", "calc 1 2 minus 3 steps", "calc 1 2 over 3 steps", "calc steps",
    "calc 1 2 steps", "calc plus steps", "calc 1 plus steps", "calc steps plus 1", "calc 1 2 plus 3 steps steps",
    "calc 0 1 plus 1 steps", "calc 1 plus x steps", "calc open 1 plus 2 close steps", "solve x plus 1 equals 2 steps",
])
def test_owns_only_its_own_prompts(g, prompt):
    assert g.owns(prompt) is False
    assert g.check(prompt, "1") == Verdict(False, None, "not my question")


def test_check_never_raises(g):
    rng = random.Random(11)
    vocab = sorted(DIGITS) + ["plus", "minus", "times", "over", "power", "root", "percent", "of", "open", "close",
                              "point", "x", "equals", "with", "steps", "calc", "solve", "compare", "check", "then", "foo"]
    for _ in range(3000):
        prompt = " ".join(rng.choice(vocab) for _ in range(rng.randint(0, 12)))
        answer = " ".join(rng.choice(vocab) for _ in range(rng.randint(0, 6)))
        assert isinstance(g.owns(prompt), bool)
        assert isinstance(g.check(prompt, answer), Verdict)
    assert isinstance(g.check("calc " + "open " * 200 + "1" + " close" * 200, "1"), Verdict)
    assert isinstance(g.check("calc 9 " * 60, "1"), Verdict)


def random_expr(rng: random.Random, depth: int) -> tuple[list[str], str]:
    """A random expression twice, with the same brackets: dense tokens and python text over Fraction."""
    def atom(d: int) -> tuple[list[str], str]:
        r = rng.random()
        if d > 0 and r < 0.3:
            dense, py = random_expr(rng, d - 1)
            dense, py = ["open", *dense, "close"], f"({py})"
        elif r < 0.45:
            n = rng.randint(0, 40)
            return ["minus", *num(n).split()], f"-Fraction({n})"
        else:
            n = rng.randint(0, 40)
            dense, py = num(n).split(), f"Fraction({n})"
        if rng.random() < 0.15:
            e = rng.randint(0, 3)
            dense, py = [*dense, "power", str(e)], f"{py}**{e}"
        return dense, py

    dense, py = atom(depth)
    for _ in range(rng.randint(0, 3)):
        op = rng.choice(["plus", "minus", "times", "over"])
        more, more_py = atom(depth)
        dense, py = [*dense, op, *more], py + {"plus": " + ", "minus": " - ", "times": " * ", "over": " / "}[op] + more_py
    return dense, py


def test_parser_agrees_with_python_arithmetic(g):
    """Precedence, left-to-right minus and over, signs and brackets on random expression trees."""
    rng = random.Random(42)
    judged = refused = 0
    for _ in range(1500):
        dense, py = random_expr(rng, 3)
        if len(dense) > 70:
            continue
        prompt = "calc " + " ".join(dense)
        try:
            value = eval(py, {"Fraction": Fraction})
        except ZeroDivisionError:
            assert not g.owns(prompt), prompt
            refused += 1
            continue
        expected = g.check(prompt, "0").expected
        assert expected is not None, prompt
        got = Fraction(str(as_rational(expected)))
        if value.denominator == 1 or set(sympy.factorint(value.denominator)) <= {2, 5}:
            assert got == value, (prompt, py)
        else:
            assert abs(got - value) <= abs(value) * Fraction(1, 10 ** 8), (prompt, py)
        assert g.check(prompt, expected).ok
        judged += 1
    assert judged > 1000 and refused > 10
    assert g.check("calc 8 minus 3 minus 2", "3").ok and g.check("calc 1 0 0 over 5 over 2", "1 0").ok


def test_trailing_period_and_spaces_in_an_answer(g):
    assert g.check("calc 1 2 plus 7", " 1 9. ").ok
    assert not g.check("calc 1 2 plus 7", "1.9").ok


def test_sympy_agrees(lines):
    """Every generated line, read independently with sympy."""
    for ln in lines:
        words = ln.prompt.split()
        head, body = words[0], words[1:]
        if head == "calc" and body[-1] != "steps":
            assert to_sympy(body) == as_rational(ln.answer), ln.text
        elif head == "solve":
            i = body.index("equals")
            sol = sympy.solve(sympy.Eq(to_sympy(body[:i]), to_sympy(body[i + 1:])), X)
            assert sol == [as_rational(ln.answer[2:])], ln.text
        elif head == "compare":
            i = body.index("with")
            a, b = to_sympy(body[:i]), to_sympy(body[i + 1:])
            assert ln.answer == ("greater" if a > b else "less" if a < b else "equal"), ln.text
        elif head == "check":
            if "x" in body:
                cut = len(body) - 1 - body[::-1].index("x")
                value = as_rational(" ".join(body[cut + 1:]))
                i = body.index("equals")
                lhs, rhs = to_sympy(body[:i]).subs(X, value), to_sympy(body[i + 1:cut]).subs(X, value)
            else:
                i = body.index("equals")
                lhs, rhs = to_sympy(body[:i]), to_sympy(body[i + 1:])
            assert ln.answer == ("yes" if sympy.simplify(lhs - rhs) == 0 else "no"), ln.text


STEP_PLUS = re.compile(r"^(\d) plus (\d)(?: plus (\d))? is ((?:\d ?)+) write (\d)(?: carry (\d))?$")
STEP_TIMES = re.compile(r"^(\d) times (\d) is ((?:\d ?)+)(?: plus (\d) is ((?:\d ?)+))? write (\d)(?: carry (\d))?$")


def test_steps_lines_add_up(lines):
    """The worked methods, checked column by column without the gate's own code."""
    seen = Counter()
    for ln in lines:
        if ln.meta["form"] != "steps":
            continue
        body = ln.prompt.split()[1:-1]
        op = "plus" if "plus" in body else "times"
        i = body.index(op)
        a, b = int("".join(body[:i])), int("".join(body[i + 1:]))
        steps = ln.answer.split(" then ")
        assert steps[-1] == "result " + " ".join(str(a + b if op == "plus" else a * b))
        if op == "plus":
            seen["plus"] += 1
            assert len(steps) - 1 == len(str(max(a, b)))
            carry = 0
            for step in steps[:-1]:
                m = STEP_PLUS.match(step)
                assert m, step
                x, y, c, s, w, c2 = m.groups()
                assert int(c or 0) == carry
                assert int(s.replace(" ", "")) == int(x) + int(y) + carry
                assert int(w) == int(s.replace(" ", "")) % 10
                carry = int(c2 or 0)
                assert carry == int(s.replace(" ", "")) // 10
        elif b < 10:
            seen["times one digit"] += 1
            assert len(steps) - 1 == len(str(a))
            carry = 0
            for step in steps[:-1]:
                m = STEP_TIMES.match(step)
                assert m, step
                x, y, p, c, s, w, c2 = m.groups()
                assert int(y) == b and int(p.replace(" ", "")) == int(x) * b
                assert int(c or 0) == carry
                total = int((s or p).replace(" ", ""))
                assert total == int(x) * b + carry and int(w) == total % 10
                carry = int(c2 or 0)
                assert carry == total // 10
        else:
            seen["times two digits"] += 1
            p1, p2, s = (int(parse_num(st.split(" is ")[1])) for st in steps[:3])
            assert (p1, p2, s) == (a * (b // 10 * 10), a * (b % 10), a * b)
            assert steps[0].startswith(" ".join(str(a)) + " times " + " ".join(str(b // 10 * 10)) + " is ")
    assert set(seen) == {"plus", "times one digit", "times two digits"}


def test_steps_end_in_the_calc_answer(g, lines):
    """The result of a worked method is the answer to the same prompt without ``steps``; the written
    digits, read from the last column back, spell it; every supported size fits in 80 tokens."""
    edge = [0, 1, 9, 10, 99, 100, 999, 1000, 5005, 9999]
    prompts = [ln.prompt for ln in lines if ln.meta["form"] == "steps"]
    prompts += [f"calc {num(a)} plus {num(b)} steps" for a in edge for b in edge]
    prompts += [f"calc {num(a)} times {num(b)} steps" for a in edge for b in (1, 2, 9)]
    prompts += [f"calc {num(a)} times {num(b)} steps" for a in edge if a < 1000 for b in (10, 11, 20, 99)]
    rng = random.Random(9)
    prompts += [f"calc {num(rng.randint(0, 9999))} plus {num(rng.randint(0, 9999))} steps" for _ in range(500)]
    prompts += [f"calc {num(rng.randint(0, 9999))} times {num(rng.randint(1, 9))} steps" for _ in range(500)]
    prompts += [f"calc {num(rng.randint(0, 999))} times {num(rng.randint(10, 99))} steps" for _ in range(500)]
    for prompt in prompts:
        assert g.owns(prompt), prompt
        steps = g.check(prompt, "").expected
        assert steps is not None and g.check(prompt, steps).ok, prompt
        assert ntokens(f"q {prompt}. a {steps}.") <= 80 and is_dense(f"q {prompt}. a {steps}."), prompt
        result = steps.rsplit("then result ", 1)[1]
        assert g.check(prompt.removesuffix(" steps"), result) == Verdict(True, result), prompt
        written = re.findall(r"write (\d)", steps)
        if written:
            carry = re.search(r"carry (\d) then result", steps)
            assert (carry.group(1) if carry else "") + "".join(reversed(written)) == result.replace(" ", ""), prompt
        # a step left out, or one too many, is wrong even with the right result
        parts = steps.split(" then ")
        assert not g.check(prompt, " then ".join(parts[1:])).ok
        assert not g.check(prompt, " then ".join(parts[:-1] + parts[-2:])).ok


def test_wrong_checks_are_plausible(lines):
    """A ``no`` check names a value in the range of the truth and its operands, not a random number."""
    for ln in lines:
        if ln.kind != "yesno" or ln.answer != "no" or " x " in ln.prompt:
            continue
        body = ln.prompt.split()[1:]
        i = body.index("equals")
        truth, shown = to_sympy(body[:i]), as_rational(" ".join(body[i + 1:]))
        operands = [abs(as_rational(run)) for run in re.findall(r"\d(?: \d)*(?: point \d(?: \d)*)?", " ".join(body[:i]))]
        assert shown != truth
        assert abs(shown - truth) <= 100 * max(abs(truth), *operands) + 100, ln.text
    for ln in lines:
        if ln.kind == "yesno" and ln.answer == "no" and " x " in ln.prompt:
            body = ln.prompt.split()[1:]
            shown = as_rational(" ".join(body[len(body) - body[::-1].index("x"):]))
            assert abs(shown) <= 400, ln.text


def test_solutions_are_integers_and_coefficients_small(lines):
    for ln in lines:
        if ln.meta["form"] != "solve":
            continue
        x = parse_num(ln.answer[2:])
        assert isinstance(x, int) and abs(x) <= 200, ln.text
        for coef in re.findall(r"((?:\d )+)x", ln.prompt):
            assert len(coef.split()) <= 2, ln.text
