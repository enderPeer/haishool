"""Round 5: closed topics with truth gates.

Rounds 1-4 taught facts and views: things people observed or agree on, which a model can only
take in. Round 5 adds truths that follow from rules and can be *checked by calculation*: maths,
the elements, what substances are made of, balanced reactions, the forces. Each topic is a
closed world with a **gate**: code that generates questions, knows the right answer and judges
any answer the model gives. Nothing in a gate comes from the teacher model.

Every topic module implements :class:`Gate`. The build (``haishool.round5``) calls the gates
to write the training lines, a sealed test set the model never sees, and later the feedback
loop (:mod:`haishool.truth.loop`) asks the trained model fresh questions, lets the gates judge
them and feeds the confirmed and the corrected answers back into training.

Dense form (the same alphabet as rounds 1-4: ``[a-z0-9_ .]``; ``.`` separates fields, so a
number never contains one):

    carbon element. number 6. symbol c. protons 6. neutrons 6. mass 1 2 point 0 1 1. ...
    q carbon protons. a 6.
    q water molar_mass. a 1 8 point 0 1 5.
    q calc 1 2 plus 7. a 1 9.
    q solve 3 x plus 4 equals 1 9. a x 5.
    q check 1 2 plus 7 equals 2 0. a no.

Numbers are written digit by digit (:func:`num`), so the model can learn to calculate:
``1838`` -> ``1 8 3 8``, ``-3.5`` -> ``minus 3 point 5``, ``6.674e-11`` ->
``6 point 6 7 4 e minus 1 1``. Keys name their unit where there is one (``melting_k``,
``density_g_cm3``); each topic documents its keys in a ``KEYS`` dict like ``schema.TYPES``.
"""

from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

ALPHABET = re.compile(r"^[a-z0-9_]+( [a-z0-9_]+)*$")
DIGITS = set("0123456789")
NUMBER_WORDS = {"point", "minus", "e"}


def num(x: float | int | str, sig: int = 4) -> str:
    """A number as digit tokens.

    Integers are exact; floats keep ``sig`` significant digits (trailing zeros dropped) and
    switch to ``e`` notation below 1e-3 or from 1e6 on, so an answer stays short enough for a
    small model to learn digit by digit.

    >>> num(1838), num(-3.5), num(18.015), num(6.674e-11), num(0.25)
    ('1 8 3 8', 'minus 3 point 5', '1 8 point 0 2', '6 point 6 7 4 e minus 1 1', '0 point 2 5')
    """
    if isinstance(x, str):
        x = float(x) if any(c in x for c in ".eE") else int(x)
    if isinstance(x, bool):
        raise TypeError("num() takes numbers, not bools")
    if isinstance(x, int) or (isinstance(x, float) and x.is_integer() and abs(x) < 1e6):
        n = int(x)
        return ("minus " if n < 0 else "") + " ".join(str(abs(n)))
    if not math.isfinite(x):
        raise ValueError(f"cannot write {x!r} as digits")
    sign = "minus " if x < 0 else ""
    x = abs(x)
    if x != 0 and (x < 1e-3 or x >= 1e6):
        mantissa, exp = f"{x:.{sig - 1}e}".split("e")
        mantissa = mantissa.rstrip("0").rstrip(".")
        return sign + _digits(mantissa) + " e " + num(int(exp))
    text = f"{x:.{sig}g}"
    if "e" in text:  # python falls back to e-notation for large sig; expand it
        text = f"{x:f}".rstrip("0").rstrip(".")
    return sign + _digits(text)


def _digits(text: str) -> str:
    return " ".join("point" if c == "." else c for c in text)


def parse_num(tokens: str | list[str]) -> float | int | None:
    """The number in a digit-token string, or ``None`` if it is not one.

    >>> parse_num("1 8 point 0 1 5"), parse_num("minus 4"), parse_num("6 point 6 7 4 e minus 1 1")
    (18.015, -4, 6.674e-11)
    """
    words = tokens.split() if isinstance(tokens, str) else list(tokens)
    if not words:
        return None
    text = ""
    for w in words:
        if w in DIGITS:
            text += w
        elif w == "point":
            text += "."
        elif w == "minus":
            text += "-"
        elif w == "e":
            text += "e"
        else:
            return None
    try:
        value = float(text)
    except ValueError:
        return None
    if "." not in text and "e" not in text:
        return int(text)
    return value


def is_finite(x: float | int | None) -> bool:
    """True when ``x`` is a number a float can hold. :func:`parse_num` reads ``1 e 9 9 9`` as
    infinity and a run of 400 digits as an integer too large for a float; neither is an answer,
    and ``math.isfinite`` raises on the second.

    >>> is_finite(parse_num("1 8 point 0 1 5")), is_finite(parse_num("1 e 9 9 9")), is_finite(10 ** 400), is_finite(None)
    (True, False, False, False)
    """
    try:
        return x is not None and math.isfinite(x)
    except OverflowError:
        return False


def is_dense(line: str) -> bool:
    """True when a whole training line is in the dense alphabet: fields of lowercase words,
    digits and underscores, single spaces, each field ending in ``.``."""
    if not line.endswith(".") or "  " in line or line != line.strip():
        return False
    fields = [f.strip() for f in line[:-1].split(".")]
    return all(f and ALPHABET.match(f) for f in fields)


@dataclass(frozen=True)
class Line:
    """One training sample: a question and its answer, or a record (``prompt`` only).

    ``kind`` is one of ``fact`` (looked up in a table), ``calc`` (computed by a rule),
    ``yesno`` (a gate check: ``q check ... a yes/no``) or ``record`` (a thing's description).
    """
    prompt: str
    answer: str = ""
    topic: str = ""
    kind: str = "fact"
    meta: dict = field(default_factory=dict, compare=False, hash=False)

    @property
    def text(self) -> str:
        if self.kind == "record":
            return self.prompt if self.prompt.endswith(".") else self.prompt + "."
        return f"q {self.prompt}. a {self.answer}."

    def __post_init__(self) -> None:
        if self.kind != "record" and ("." in self.prompt or "." in self.answer):
            raise ValueError(f"a question or answer may not contain '.': {self.prompt!r} / {self.answer!r}")
        if not is_dense(self.text):
            raise ValueError(f"not dense: {self.text!r}")


def split_line(text: str) -> tuple[str, str] | None:
    """``q carbon protons. a 6.`` -> ``("carbon protons", "6")``; ``None`` for a record line."""
    if not text.startswith("q ") or ". a " not in text:
        return None
    prompt, _, answer = text[2:].rstrip(".").partition(". a ")
    return prompt, answer


@dataclass(frozen=True)
class Verdict:
    ok: bool
    expected: str | None = None
    reason: str = ""


@runtime_checkable
class Gate(Protocol):
    """A closed topic: it writes its own questions and judges any answer."""

    #: short topic name, used in reports and file names (``maths``, ``elements``)
    topic: str
    #: key -> what it means, with its unit, like ``schema.TYPES``
    KEYS: dict[str, str]

    def records(self) -> list[Line]:
        """The topic's table as record lines (empty for a purely computed topic)."""

    def generate(self, rng: random.Random, n: int) -> list[Line]:
        """``n`` question lines; different seeds give different questions."""

    def check(self, prompt: str, answer: str) -> Verdict:
        """Judge an answer to one of this topic's prompts by rule or table, never by guessing.
        A prompt the gate cannot judge gets ``Verdict(False, None, "not my question")``."""

    def owns(self, prompt: str) -> bool:
        """True when ``prompt`` is a question of this topic."""


def dedupe(lines: list[Line]) -> list[Line]:
    seen: set[str] = set()
    out = []
    for ln in lines:
        if ln.text not in seen:
            seen.add(ln.text)
            out.append(ln)
    return out


def sealed_split(lines: list[Line], sealed_prompts: set[str]) -> list[Line]:
    """Training lines minus anything whose prompt is in the sealed test set."""
    return [ln for ln in lines if ln.kind == "record" or ln.prompt not in sealed_prompts]
