"""Round 5, topic ``maths``: arithmetic that one small parser computes exactly, so that every
answer is checkable by rule and no line comes from memory.

Prompts and answers are dense tokens: digits one by one (:func:`haishool.truth.num`), ``point``
for the decimal point, ``minus`` both for a negative number and for subtraction. Five forms,
with the share of each in :meth:`Maths.generate` (the mix in :data:`MIX`):

    q calc 1 2 plus 7. a 1 9.                                           calc     45 %
    q calc 1 5 percent of 2 0 0. a 3 0.
    q calc open 2 plus 3 close times 4. a 2 0.
    q calc 4 7 plus 3 8 steps. a 7 plus 8 is 1 5 write 5 carry 1
        then 4 plus 3 plus 1 is 8 write 8 then result 8 5.             steps    10 %
    q solve 3 x plus 4 equals 1 9. a x 5.                               solve    20 %
    q compare 3 point 5 with minus 2. a greater.                        compare  10 %
    q check 1 2 plus 7 equals 2 0. a no.                                check    15 %
    q check 3 x plus 4 equals 1 9 x 5. a yes.

``calc``, ``steps``, ``solve`` and ``compare`` lines have kind ``calc``; ``check`` lines have
kind ``yesno``. The topic has no table, so :meth:`Maths.records` is empty.

Inside each form, what :meth:`Maths.generate` draws:

* ``calc``: plus/minus of integers up to 4 digits, negatives included (30 %); times, up to
  3 digits by 2 (15 %); exact over (10 %); power 2 or 3 (7 %); root of a perfect square (6 %);
  percent of, with a whole result (8 %); two operators with precedence or brackets (14 %);
  plus/minus of decimals with one or two places (10 %).
* ``steps``: plus (60 %), times by one digit (28 %), times by two digits (12 %).
* ``solve``: eight shapes in equal parts, among them ``a x plus b equals c``, ``a x plus b
  equals c x plus d``, ``x over k equals c``, ``c equals a x plus b`` and ``open x plus b
  close times a equals c``; x from -20 to 20 (to 180 with ``over``).
* ``compare``: integers (50 %) and decimals with up to three places (50 %), about 15 % equal
  and a good third only one digit apart.
* ``check``: an expression against a value (60 %) or a value for x in an equation (40 %);
  half ``yes``, half ``no``.

These are the shares of the draw. One call never repeats a prompt: a duplicate is drawn again
within its form. ``power`` (117 prompts: squares to 99, cubes to 20) and ``root`` (149 prompts:
squares of 2 to 150) are small closed sets, so a large ``n`` uses them up and fills the rest of
the ``calc`` share with the other families.

**Expressions** (``calc``, both sides of ``solve``/``check``, both numbers of ``compare``)::

    pexpr  := expr ('percent' 'of' expr)?          p percent of n = p times n over 100
    expr   := term (('plus' | 'minus') term)*
    term   := factor (('times' | 'over') factor)*
    factor := signed ('power' digits)?              exponent 0 to 3, base a number
    signed := 'minus' signed | 'root' signed | atom  prefixes bind to the atom: minus 3 power 2 is 9
    atom   := 'open' pexpr 'close' | 'x' | number 'x'?    3 x is 3 times x
    number := digits ('point' digits)?               no leading zeros (0 point 5 is fine)

Precedence as in school maths: ``2 plus 3 times 4`` is 14, brackets first. Everything is
evaluated as a linear form ``a x + b`` over :class:`fractions.Fraction`, so ``x times x``,
``over`` by zero or by ``x``, ``root`` of anything but a perfect square, and ``power`` of
``x`` are outside the grammar: such a prompt is not owned and gets ``not my question``.
``compare`` needs the word ``with`` between its two numbers because two digit runs would
otherwise melt into one (``compare 1 2 3`` could be 12 with 3 or 1 with 23).

**Steps** (``calc <a> <plus|times> <b> steps``, non-negative integers): the column method,
one step per column from the units up, joined by ``then``, closed by ``then result <n>``.

* ``plus`` (a, b up to 4 digits; a missing digit counts as 0)::

      <da> plus <db> [plus <carry in>] is <sum> write <sum mod 10> [carry <sum div 10>]

  ``calc 9 5 plus 8 steps`` -> ``5 plus 8 is 1 3 write 3 carry 1 then 9 plus 0 plus 1 is
  1 0 write 0 carry 1 then result 1 0 3`` (a carry left at the end leads the result).
* ``times`` by one digit (a up to 4 digits, b 1 to 9)::

      <da> times <b> is <product> [plus <carry in> is <sum>] write <sum mod 10> [carry <sum div 10>]

  ``calc 4 7 times 6 steps`` -> ``7 times 6 is 4 2 write 2 carry 4 then 4 times 6 is 2 4
  plus 4 is 2 8 write 8 carry 2 then result 2 8 2``.
* ``times`` by two digits (a up to 3 digits, b 10 to 99): the tens and the units of ``b``
  as two partial products, then their sum::

      <a> times <tens> is <p1> then <a> times <units> is <p2> then <p1> plus <p2> is <n> then result <n>

:meth:`Maths.check` recomputes every step from the operands and names the first step that
differs, so a right result with a wrong carry is still wrong.

**Solve**: a linear equation in ``x`` with one ``equals``; the answer is ``x <number>``.
Generated equations have integer solutions and coefficients of at most two digits.

**Check**: ``check <expr> equals <expr>`` is ``yes`` when both sides are equal, and
``check <equation> x <number>`` is ``yes`` when that number solves the equation (the last
``x`` of the prompt separates the equation from the value, and the equation itself must
contain ``x``). About half of the generated checks are ``no``, with a plausible slip: a
changed digit, a lost carry, a wrong sign, swapped operands, a wrong precedence or a shifted
decimal point.

**Judging**: maths is exact, so this gate has no tolerance. A numeric answer must be written
as a number (an optional ``minus``, digits without leading zeros, an optional ``point`` part,
an optional ``e`` exponent of up to three digits) and must equal the expected value: ``3``,
``3 point 0`` and ``3 e 0`` all answer ``calc 1 plus 2``, but ``0 3``, ``3 point`` and
``3 point 0 1 e 0`` do not. Every generated answer is an integer or a terminating decimal.
Only a value that does not terminate (``calc 7 over 3`` from outside) cannot be written in
full: its canonical answer has 10 significant digits (``2 point 3 3 3 3 3 3 3 3 3``; ``e``
notation below 0.001 and from a million on, as :func:`haishool.truth.num` writes it), and any
answer that is that value correctly rounded to the digits it states, at least three, is right
(``2 point 3 3 3``, not ``2 point 3`` and not ``2 point 3 3 4``). Word answers (``yes``,
``no``, ``greater``, ``less``, ``equal``) must match exactly. One trailing ``.`` on an answer
is ignored.

**Limits**: a prompt of more than 80 tokens is not owned, and neither is a ``power`` whose
result would pass about 300 digits (brackets can stack powers: nine cubed nineteen times over
fits in 80 tokens). So no value has more than about 1000 digits or 3300 decimal places, an
answer of more than 4000 tokens is not a number, and judging always ends quickly. Words are
split on any whitespace, as in the other gates.

The same parser serves :meth:`generate`, :meth:`check` and :meth:`owns`: a generated prompt
gets its answer by being parsed, never from the generator's own arithmetic.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from fractions import Fraction

from haishool.truth import DIGITS, Line, Verdict, num

TOPIC = "maths"
#: the limit of ``haishool.student.tokens`` on a whole line (``q ... . a ... .``)
MAX_TOKENS = 80
#: the largest numerator or denominator a ``power`` may produce, in bits (about 300 digits)
MAX_BITS = 1000
#: the longest answer that is read as a number, in tokens; no value of an owned prompt needs more
MAX_ANSWER = 4000
#: significant digits of the canonical answer for a value that does not terminate
SIG = 10

KEYS: dict[str, str] = {
    "calc": "the value of an arithmetic expression, exact: an integer or a decimal",
    "steps": "calc a plus b steps, calc a times b steps: the worked column method, step by step, then the result",
    "solve": "the value of x in a linear equation, as x followed by a number",
    "compare": "greater, less or equal: the first number against the second (compare a with b)",
    "check": "yes or no: whether an expression equals a value, or a value solves an equation in x",
}
#: the words of the prompt and answer grammar
WORDS: dict[str, str] = {
    "plus": "addition", "minus": "subtraction, or the sign of a negative number",
    "times": "multiplication", "over": "division", "power": "exponent, 0 to 3",
    "root": "square root of a perfect square", "percent of": "p percent of n is p times n over 100",
    "open": "opening bracket", "close": "closing bracket", "point": "decimal point",
    "x": "the unknown; 3 x is 3 times x", "equals": "the equals sign",
    "with": "separates the two numbers of a compare", "steps": "asks for the worked column method",
    "then": "separates the steps of a worked method", "is": "the value of one step",
    "write": "the digit written in the column", "carry": "the carry into the next column",
    "result": "the final result of a worked method",
}
#: form -> share of generate(); the order is the order of the draw
MIX: list[tuple[str, float]] = [("calc", 0.45), ("steps", 0.10), ("solve", 0.20), ("compare", 0.10), ("check", 0.15)]

_HEADS = ("calc", "solve", "compare", "check")
_VOCAB = DIGITS | set(_HEADS) | {"plus", "minus", "times", "over", "power", "root", "percent", "of",
                                 "open", "close", "point", "x", "equals", "with", "steps"}
_PERCENTS = [1, 2, 5, 10, 12, 15, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 100]

#: a linear form ``a x + b``
Lin = tuple[Fraction, Fraction]
_ZERO = Fraction(0)


class _Bad(Exception):
    """A prompt or an answer outside the grammar."""


# ---------------------------------------------------------------------------------------------
# the one evaluator


def _const(v: Lin) -> Fraction:
    if v[0] != 0:
        raise _Bad("x where a number is needed")
    return v[1]


def _mul(v: Lin, w: Lin) -> Lin:
    if v[0] == 0:
        return v[1] * w[0], v[1] * w[1]
    if w[0] == 0:
        return v[0] * w[1], v[1] * w[1]
    raise _Bad("x times x is not linear")


def _div(v: Lin, w: Lin) -> Lin:
    d = _const(w)
    if d == 0:
        raise _Bad("division by zero")
    return v[0] / d, v[1] / d


def _root(c: Fraction) -> Fraction:
    if c < 0:
        raise _Bad("root of a negative number")
    p, q = math.isqrt(c.numerator), math.isqrt(c.denominator)
    if p * p != c.numerator or q * q != c.denominator:
        raise _Bad("root of a non-square")
    return Fraction(p, q)


class _Parser:
    """Recursive descent over dense tokens, evaluating as it goes (grammar in the module docstring)."""

    def __init__(self, tokens: list[str]) -> None:
        self.t, self.i = tokens, 0

    def peek(self) -> str | None:
        return self.t[self.i] if self.i < len(self.t) else None

    def take(self, word: str | None = None) -> str:
        w = self.peek()
        if w is None or (word is not None and w != word):
            raise _Bad(f"expected {word or 'more'} at token {self.i}")
        self.i += 1
        return w

    def end(self) -> None:
        if self.i != len(self.t):
            raise _Bad(f"unexpected {self.t[self.i]!r}")

    def digits(self, zeros: bool = False) -> str:
        start = self.i
        while self.peek() in DIGITS:
            self.i += 1
        text = "".join(self.t[start:self.i])
        if not text or (not zeros and len(text) > 1 and text[0] == "0"):
            raise _Bad("not a number")
        return text

    def number(self) -> Fraction:
        whole = self.digits()
        if self.peek() == "point":
            self.take()
            return Fraction(f"{whole}.{self.digits(zeros=True)}")
        return Fraction(int(whole))

    def signed_number(self) -> Fraction:
        """One number with an optional ``minus``, nothing else."""
        negative = self.peek() == "minus"
        if negative:
            self.take()
        v = self.number()
        self.end()
        return -v if negative else v

    def pexpr(self) -> Lin:
        v = self.expr()
        if self.peek() == "percent":
            self.take()
            self.take("of")
            v = _mul((_ZERO, _const(v) / 100), self.expr())
        return v

    def expr(self) -> Lin:
        v = self.term()
        while self.peek() in ("plus", "minus"):
            sign = 1 if self.take() == "plus" else -1
            w = self.term()
            v = v[0] + sign * w[0], v[1] + sign * w[1]
        return v

    def term(self) -> Lin:
        v = self.factor()
        while self.peek() in ("times", "over"):
            op = self.take()
            w = self.factor()
            v = _mul(v, w) if op == "times" else _div(v, w)
        return v

    def factor(self) -> Lin:
        v = self.signed()
        if self.peek() == "power":
            self.take()
            e = int(self.digits())
            if e > 3:
                raise _Bad("power above 3")
            base = _const(v)
            if e * max(base.numerator.bit_length(), base.denominator.bit_length()) > MAX_BITS:
                raise _Bad("number too large")
            v = _ZERO, base ** e
        return v

    def signed(self) -> Lin:
        w = self.peek()
        if w == "minus":
            self.take()
            v = self.signed()
            return -v[0], -v[1]
        if w == "root":
            self.take()
            return _ZERO, _root(_const(self.signed()))
        return self.atom()

    def atom(self) -> Lin:
        w = self.peek()
        if w == "open":
            self.take()
            v = self.pexpr()
            self.take("close")
            return v
        if w == "x":
            self.take()
            return Fraction(1), _ZERO
        n = self.number()
        if self.peek() == "x":
            self.take()
            return n, _ZERO
        return _ZERO, n


def _evaluate(tokens: list[str]) -> Lin:
    p = _Parser(tokens)
    v = p.pexpr()
    p.end()
    return v


def _value(tokens: list[str]) -> Fraction:
    return _const(_evaluate(tokens))


def _signed_number(tokens: list[str]) -> Fraction:
    return _Parser(tokens).signed_number()


def _split_once(tokens: list[str], word: str) -> tuple[list[str], list[str]]:
    if tokens.count(word) != 1:
        raise _Bad(f"need exactly one {word}")
    i = tokens.index(word)
    if i == 0 or i == len(tokens) - 1:
        raise _Bad(f"nothing on one side of {word}")
    return tokens[:i], tokens[i + 1:]


def _places(v: Fraction) -> int | None:
    """Decimal places of a terminating fraction, ``None`` when it does not terminate."""
    d, twos, fives = v.denominator, 0, 0
    while d % 2 == 0:
        d, twos = d // 2, twos + 1
    while d % 5 == 0:
        d, fives = d // 5, fives + 1
    return max(twos, fives) if d == 1 else None


def _round_sig(v: Fraction, sig: int) -> tuple[int, int]:
    """``|v|`` rounded half up to ``sig`` significant digits, as ``(m, e)`` with the value
    ``m * 10**e`` and ``m`` of exactly ``sig`` digits. Exact: no float is involved."""
    a = abs(v)
    lead = len(str(a.numerator)) - len(str(a.denominator))  # floor(log10(a)), or one more
    if a < Fraction(10) ** lead:
        lead -= 1
    e = lead - sig + 1
    m = math.floor(a / Fraction(10) ** e + Fraction(1, 2))
    if m == 10 ** sig:
        m, e = m // 10, e + 1
    return m, e


def _sig_digits(v: Fraction) -> int:
    """Significant digits of a terminating decimal: ``0 point 0 3 5 0`` has two."""
    n = abs(v.numerator) * 10 ** (_places(v) or 0) // v.denominator
    while n and n % 10 == 0:
        n //= 10
    return len(str(n))


def _fmt(v: Fraction) -> str:
    """A fraction as digit tokens: exact for integers and terminating decimals, else rounded to
    :data:`SIG` significant digits in the notation of :func:`haishool.truth.num`."""
    if v.denominator == 1:
        return num(int(v))
    sign = "minus " if v < 0 else ""
    k = _places(v)
    if k is None:
        m, e = _round_sig(v, SIG)
        if Fraction(1, 1000) <= abs(v) < 10 ** 6:
            return sign + _fmt(m * Fraction(10) ** e)
        mantissa = f"{str(m)[0]}.{str(m)[1:]}".rstrip("0").rstrip(".")
        return sign + " ".join("point" if c == "." else c for c in mantissa) + " e " + num(e + SIG - 1)
    n = abs(v.numerator) * 10 ** k // v.denominator
    text = f"{n // 10 ** k}.{n % 10 ** k:0{k}d}"
    return sign + " ".join("point" if c == "." else c for c in text)


def _ntokens(text: str) -> int:
    """The same count as ``haishool.student.tokens`` (which needs torch to import)."""
    return len(text.replace(".", " . ").split())


# ---------------------------------------------------------------------------------------------
# worked column methods


def _steps_plus(a: int, b: int) -> list[str]:
    da, db = [int(c) for c in str(a)[::-1]], [int(c) for c in str(b)[::-1]]
    n = max(len(da), len(db))
    da, db = da + [0] * (n - len(da)), db + [0] * (n - len(db))
    out, carry = [], 0
    for x, y in zip(da, db):
        s = x + y + carry
        step = f"{x} plus {y}" + (f" plus {carry}" if carry else "") + f" is {num(s)} write {s % 10}"
        carry = s // 10
        out.append(step + (f" carry {carry}" if carry else ""))
    return out + [f"result {num(a + b)}"]


def _steps_times(a: int, b: int) -> list[str]:
    if b < 10:
        out, carry = [], 0
        for x in (int(c) for c in str(a)[::-1]):
            p = x * b
            s = p + carry
            step = f"{x} times {b} is {num(p)}" + (f" plus {carry} is {num(s)}" if carry else "") + f" write {s % 10}"
            carry = s // 10
            out.append(step + (f" carry {carry}" if carry else ""))
        return out + [f"result {num(a * b)}"]
    tens, units = b // 10 * 10, b % 10
    p1, p2 = a * tens, a * units
    return [f"{num(a)} times {num(tens)} is {num(p1)}", f"{num(a)} times {num(units)} is {num(p2)}",
            f"{num(p1)} plus {num(p2)} is {num(p1 + p2)}", f"result {num(a * b)}"]


def _steps(tokens: list[str]) -> list[str]:
    """``4 7 plus 3 8`` -> the canonical steps, or ``_Bad`` outside the supported sizes."""
    ops = [i for i, t in enumerate(tokens) if t in ("plus", "times")]
    if len(ops) != 1:
        raise _Bad("steps need one plus or times")
    i = ops[0]
    left, right = tokens[:i], tokens[i + 1:]
    if any(t not in DIGITS for t in left + right):
        raise _Bad("steps take whole numbers")
    a, b = int(_Parser(left).digits()), int(_Parser(right).digits())
    if tokens[i] == "plus":
        if a > 9999 or b > 9999:
            raise _Bad("steps add numbers of up to 4 digits")
        return _steps_plus(a, b)
    if not (1 <= b <= 9 and a <= 9999) and not (10 <= b <= 99 and a <= 999):
        raise _Bad("steps multiply up to 4 digits by 1 digit, or 3 digits by 2")
    return _steps_times(a, b)


def _check_steps(steps: list[str], answer: str) -> Verdict:
    expected = " then ".join(steps)
    got, group = [], []
    for w in answer.split() + ["then"]:
        if w == "then":
            got.append(" ".join(group))
            group = []
        else:
            group.append(w)
    for i, (g, s) in enumerate(zip(got, steps)):
        if g != s:
            where = "result" if i == len(steps) - 1 else f"step {i + 1}"
            return Verdict(False, expected, f"wrong {where}: {s}")
    if len(got) != len(steps):
        return Verdict(False, expected, f"{len(steps) - 1} steps and a result expected")
    return Verdict(True, expected)


# ---------------------------------------------------------------------------------------------
# prompts


@dataclass(frozen=True)
class _Parsed:
    form: str                 # calc, steps, solve, compare or check
    kind: str                 # calc or yesno
    answer: str               # the canonical answer
    value: Fraction | None = None
    steps: tuple[str, ...] = ()


def _parse(prompt: str) -> _Parsed:
    tokens = prompt.split()
    if len(tokens) < 2 or len(tokens) > MAX_TOKENS or tokens[0] not in _HEADS or any(t not in _VOCAB for t in tokens):
        raise _Bad("not a maths prompt")
    head, body = tokens[0], tokens[1:]
    if head == "calc":
        if body[-1] == "steps":
            steps = _steps(body[:-1])
            return _Parsed("steps", "calc", " then ".join(steps), steps=tuple(steps))
        v = _value(body)
        return _Parsed("calc", "calc", _fmt(v), v)
    if head == "solve":
        lhs, rhs = (_evaluate(side) for side in _split_once(body, "equals"))
        a, b = lhs[0] - rhs[0], rhs[1] - lhs[1]
        if a == 0:
            raise _Bad("not an equation in x")
        return _Parsed("solve", "calc", "x " + _fmt(b / a), b / a)
    if head == "compare":
        a, b = (_value(side) for side in _split_once(body, "with"))
        return _Parsed("compare", "calc", "greater" if a > b else "less" if a < b else "equal")
    if "x" in body:
        cut = len(body) - 1 - body[::-1].index("x")
        if "x" not in body[:cut]:
            raise _Bad("no x in the equation")
        x = _signed_number(body[cut + 1:])
        lhs, rhs = (_evaluate(side) for side in _split_once(body[:cut], "equals"))
        ok = lhs[0] * x + lhs[1] == rhs[0] * x + rhs[1]
    else:
        lhs, rhs = (_value(side) for side in _split_once(body, "equals"))
        ok = lhs == rhs
    return _Parsed("check", "yesno", "yes" if ok else "no")


def _parse_or_none(prompt: str) -> _Parsed | None:
    try:
        return _parse(prompt)
    except (_Bad, ValueError, ZeroDivisionError, OverflowError, RecursionError):
        return None


def _answer_value(text: str) -> Fraction | None:
    """The number an answer states, ``None`` when it is not written as one (module docstring)."""
    p = _Parser(text.split())
    if len(p.t) > MAX_ANSWER:
        return None
    try:
        negative = p.peek() == "minus"
        if negative:
            p.take()
        v = p.number()
        if p.peek() == "e":
            p.take()
            down = p.peek() == "minus"
            if down:
                p.take()
            exponent = p.digits()
            if len(exponent) > 3:  # 1 e 9 9 9 9 9 9 9 9 9 would take minutes to build
                raise _Bad("exponent too large")
            v *= Fraction(10) ** (-int(exponent) if down else int(exponent))
        p.end()
    except (_Bad, ValueError):
        return None
    return -v if negative else v


def _same(expected: Fraction, got: Fraction) -> bool:
    """Equal, or (only for a value that does not terminate) the value correctly rounded to the
    digits the answer states, at least three."""
    if got == expected:
        return True
    if _places(expected) is not None or got == 0 or (got < 0) != (expected < 0):
        return False
    m, e = _round_sig(expected, max(3, _sig_digits(got)))
    return abs(got) == m * Fraction(10) ** e


# ---------------------------------------------------------------------------------------------
# generation


def _int(rng: random.Random, digits: int) -> int:
    """A non-negative integer of exactly ``digits`` digits (a one-digit number may be 0)."""
    return rng.randint(0 if digits == 1 else 10 ** (digits - 1), 10 ** digits - 1)


def _decimal(rng: random.Random) -> Fraction:
    """A positive decimal with one or two places and up to three digits before the point."""
    k = rng.randint(1, 2)
    return Fraction(rng.randint(1, 999 * 10 ** k), 10 ** k)


def _two_operators(rng: random.Random) -> tuple[str, Fraction]:
    """An expression with two operators, and the value a wrong precedence would give."""
    a, b, c = (rng.randint(2, 12) if rng.random() < 0.6 else rng.randint(2, 99) for _ in range(3))
    q = rng.randint(2, 99)
    forms = [
        (f"{num(a)} plus {num(b)} times {num(c)}", (a + b) * c),
        (f"{num(a)} minus {num(b)} times {num(c)}", (a - b) * c),
        (f"{num(a)} times {num(b)} plus {num(c)}", a * (b + c)),
        (f"{num(a)} times {num(b)} minus {num(c)}", a * (b - c)),
        (f"open {num(a)} plus {num(b)} close times {num(c)}", a + b * c),
        (f"open {num(a)} minus {num(b)} close times {num(c)}", a - b * c),
        (f"{num(a)} times open {num(b)} plus {num(c)} close", a * b + c),
        (f"{num(a)} plus {num(b)} minus {num(c)}", a - b + c),
        (f"{num(q * c)} over {num(c)} plus {num(b)}", q - b),
        (f"{num(a)} plus {num(q * c)} over {num(c)}", (a + q * c) // c),
        (f"{num(a)} times {num(q * c)} over {num(c)}", a * q + c),
        (f"{num(a)} power 2 plus {num(b)}", (a + b) ** 2),
    ]
    text, alt = forms[rng.randrange(len(forms))]
    return text, Fraction(alt)


def _expression(rng: random.Random) -> tuple[str, str, list[Fraction]]:
    """An expression without ``x``: its dense text, its family and the operands a slip needs."""
    r = rng.random()
    if r < 0.30:
        a, b = _int(rng, rng.randint(1, 4)), _int(rng, rng.randint(1, 4))
        a = -a if rng.random() < 0.25 else a
        b = -b if rng.random() < 0.15 else b
        op = "plus" if rng.random() < 0.5 else "minus"
        return f"{num(a)} {op} {num(b)}", "add" if op == "plus" else "sub", [Fraction(a), Fraction(b)]
    if r < 0.45:
        if rng.random() < 0.3:
            a, b = rng.randint(2, 12), rng.randint(2, 12)
        else:
            a, b = _int(rng, rng.randint(2, 3)), _int(rng, rng.randint(1, 2))
        return f"{num(a)} times {num(b)}", "mul", [Fraction(a), Fraction(b)]
    if r < 0.55:
        d = rng.randint(2, 12) if rng.random() < 0.6 else rng.randint(2, 99)
        q = rng.randint(2, 999 if d <= 12 else 99)
        return f"{num(q * d)} over {num(d)}", "div", [Fraction(q * d), Fraction(d)]
    if r < 0.62:
        e = 3 if rng.random() < 0.35 else 2
        base = rng.randint(2, 20 if e == 3 else 99)
        return f"{num(base)} power {num(e)}", "power", [Fraction(base), Fraction(e)]
    if r < 0.68:
        k = rng.randint(2, 150)
        return f"root {num(k * k)}", "root", [Fraction(k * k), Fraction(k)]
    if r < 0.76:
        p = rng.choice(_PERCENTS) if rng.random() < 0.7 else rng.randint(1, 99)
        step = 100 // math.gcd(p, 100)
        n = step * rng.randint(1, max(1, 5000 // step))
        return f"{num(p)} percent of {num(n)}", "percent", [Fraction(p), Fraction(n)]
    if r < 0.90:
        text, alt = _two_operators(rng)
        return text, "two", [alt]
    a, b = _decimal(rng), _decimal(rng)
    a = -a if rng.random() < 0.15 else a
    op = "plus" if rng.random() < 0.5 else "minus"
    return f"{_fmt(a)} {op} {_fmt(b)}", "decimal", [a, b]


def _wrong(rng: random.Random, v: Fraction, family: str, ops: list[Fraction]) -> Fraction:
    """A plausible wrong value for a ``no`` check: a digit slip, a lost carry, a sign, a swap."""
    k = rng.randrange(max(1, len(str(abs(int(v))))))
    cands = [v + 1, v - 1, -v, v + 10 ** k, v - 10 ** k]
    if family in ("add", "sub"):
        a, b = ops
        cands += [v + 10, v - 10, a - b, b - a, a + b]
    elif family == "mul":
        a, b = ops
        cands += [a * (b + 1), a * (b - 1), (a + 1) * b, v * 10, a + b, v + a, v - a]
    elif family == "div":
        a, d = ops
        cands += [v * 10, v / 10, v + 2, v - 2, v * 2, d, a - d]
    elif family == "power":
        base, e = ops
        cands += [base * e, v + base, v - base, base ** int(5 - e)]
    elif family == "root":
        n = ops[0]
        cands += [n / 2, n / 4, v * 2, v + 10]
    elif family == "percent":
        p, n = ops
        cands += [p * n / 10, p * n / 1000, n - p, n + p, p]
    elif family == "two":
        cands += [ops[0], v + 10, v - 10]
    elif family == "decimal":
        a, b = ops
        cands += [v + Fraction(1, 10), v - Fraction(1, 10), v * 10, v / 10, a - b, b - a, a + b,
                  v + Fraction(1, 100), v - Fraction(1, 100)]
    cands = [c for c in cands if c != v]
    return cands[rng.randrange(len(cands))]


def _ax(a: int) -> str:
    """The term ``a x``: ``x``, ``minus x``, ``3 x``, ``minus 3 x``."""
    sign = "minus " if a < 0 else ""
    return sign + ("x" if abs(a) == 1 else f"{num(abs(a))} x")


def _pc(b: int) -> str:
    """``plus b`` / ``minus b`` as a tail, empty for 0."""
    return "" if b == 0 else f" plus {num(b)}" if b > 0 else f" minus {num(-b)}"


def _equation(rng: random.Random) -> tuple[str, int]:
    """A linear equation in x with an integer solution, and that solution."""
    x = rng.randint(-20, 20)
    a = rng.randint(1, 12) if rng.random() < 0.7 else rng.randint(1, 99)
    a = -a if rng.random() < 0.25 else a
    b = rng.randint(-99, 99)
    if a == 1 and b == 0:
        b = 1  # no bare ``x equals 5``
    form = rng.randrange(8)
    if form == 0:
        return f"{_ax(a)}{_pc(b)} equals {num(a * x + b)}", x
    if form == 1:
        a = 2 * a if abs(a) == 1 else a
        return f"{_ax(a)} equals {num(a * x)}", x
    if form == 2:
        b = b or 1
        return f"x{_pc(b)} equals {num(x + b)}", x
    if form == 3:
        c = rng.randint(1, 12) * (1 if rng.random() < 0.7 else -1)
        c = -c if c == a else c
        return f"{_ax(a)}{_pc(b)} equals {_ax(c)}{_pc(a * x + b - c * x)}", x
    k = rng.randint(2, 12)
    if form == 4:
        x = k * rng.randint(-15, 15)
        return f"x over {num(k)} equals {num(x // k)}", x
    if form == 5:
        x = k * rng.randint(-15, 15)
        return f"x over {num(k)}{_pc(b)} equals {num(x // k + b)}", x
    if form == 6:
        return f"{num(a * x + b)} equals {_ax(a)}{_pc(b)}", x
    b = b or 1
    return f"open x{_pc(b)} close times {num(abs(a))} equals {num(abs(a) * (x + b))}", x


def _calc_prompt(rng: random.Random) -> str:
    return "calc " + _expression(rng)[0]


def _steps_prompt(rng: random.Random) -> str:
    r = rng.random()
    if r < 0.6:
        return f"calc {num(_int(rng, rng.randint(2, 4)))} plus {num(_int(rng, rng.randint(1, 4)))} steps"
    if r < 0.88:
        return f"calc {num(_int(rng, rng.randint(2, 4)))} times {num(rng.randint(2, 9))} steps"
    return f"calc {num(_int(rng, rng.randint(2, 3)))} times {num(10 * rng.randint(1, 9) + rng.randint(1, 9))} steps"


def _solve_prompt(rng: random.Random) -> str:
    return "solve " + _equation(rng)[0]


def _compare_prompt(rng: random.Random) -> str:
    if rng.random() < 0.5:
        a = rng.randint(-9999, 9999) if rng.random() < 0.5 else _int(rng, rng.randint(1, 4)) * (-1 if rng.random() < 0.3 else 1)
        r = rng.random()
        b = a if r < 0.15 else a + [-1, 1, -10, 10, -100, 100][rng.randrange(6)] if r < 0.5 else rng.randint(-9999, 9999)
        return f"compare {num(a)} with {num(b)}"
    pa, pb = rng.randint(1, 3), rng.randint(1, 3)
    a = Fraction(rng.randint(-99 * 10 ** pa, 99 * 10 ** pa), 10 ** pa)
    r = rng.random()
    if r < 0.15:
        b = a
    elif r < 0.55:
        b = a + Fraction(1 if rng.random() < 0.5 else -1, 10 ** rng.randint(1, 3))
    else:
        b = Fraction(rng.randint(-99 * 10 ** pb, 99 * 10 ** pb), 10 ** pb)
    return f"compare {_fmt(a)} with {_fmt(b)}"


def _check_prompt(rng: random.Random) -> str:
    if rng.random() < 0.6:
        text, family, ops = _expression(rng)
        v = _value(text.split())
        shown = v if rng.random() < 0.5 else _wrong(rng, v, family, ops)
        return f"check {text} equals {_fmt(shown)}"
    eq, x = _equation(rng)
    if rng.random() >= 0.5:
        slips = [c for c in (x + 1, x - 1, x + 2, x - 2, -x, 2 * x, x + 10, x - 10) if c != x]
        x = slips[rng.randrange(len(slips))]
    return f"check {eq} x {num(x)}"


_PROMPTS = {"calc": _calc_prompt, "steps": _steps_prompt, "solve": _solve_prompt,
            "compare": _compare_prompt, "check": _check_prompt}


# ---------------------------------------------------------------------------------------------
# the gate


class Maths:
    """The maths gate (module docstring): purely computed, no table."""

    topic = TOPIC
    KEYS = KEYS

    def records(self) -> list[Line]:
        return []

    def generate(self, rng: random.Random, n: int) -> list[Line]:
        """``n`` lines with distinct prompts in the shares of :data:`MIX`; same seed, same lines."""
        out: list[Line] = []
        seen: set[str] = set()
        tries = 0
        while len(out) < n and tries < 20 * n + 100:
            tries += 1
            r, form = rng.random(), MIX[-1][0]
            for name, share in MIX:
                if r < share:
                    form = name
                    break
                r -= share
            # a duplicate is redrawn in the same form, so the small sub-spaces (roots, powers)
            # do not shrink the share of their form
            prompt = next((p for p in (_PROMPTS[form](rng) for _ in range(30)) if p not in seen), None)
            if prompt is None:
                continue
            p = _parse(prompt)
            line = Line(prompt, p.answer, TOPIC, p.kind, {"form": form})
            if _ntokens(line.text) > MAX_TOKENS:
                continue
            seen.add(prompt)
            out.append(line)
        return out

    def check(self, prompt: str, answer: str) -> Verdict:
        p = _parse_or_none(prompt)
        if p is None:
            return Verdict(False, None, "not my question")
        got = " ".join(answer.strip().removesuffix(".").split())
        if p.form == "steps":
            return _check_steps(list(p.steps), got)
        if p.form in ("compare", "check"):
            return Verdict(got == p.answer, p.answer, "" if got == p.answer else f"expected {p.answer}")
        if p.form == "solve":
            if not got.startswith("x "):
                return Verdict(False, p.answer, "an answer to solve starts with x")
            got = got[2:]
        value = _answer_value(got)
        if value is None:
            return Verdict(False, p.answer, "not a number")
        assert p.value is not None
        if _same(p.value, value):
            return Verdict(True, p.answer)
        return Verdict(False, p.answer, f"expected {p.answer}")

    def owns(self, prompt: str) -> bool:
        return _parse_or_none(prompt) is not None


def gate() -> Maths:
    return Maths()
