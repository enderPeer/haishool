"""Chemical formulas as atom counts, shared by substances, reactions and the chemistry toy.

A formula is written in its usual form (``H2O``, ``Ca(OH)2``, ``C6H12O6``, ``Fe2O3``) in the
tables, and in the dense form (``h 2 o``, ``ca 1 o 2 h 2``) in the training
lines: lowercase element symbols and digit tokens, brackets expanded, so that every token is
either an element or a digit.

    >>> parse("Ca(OH)2")
    {'Ca': 1, 'O': 2, 'H': 2}
    >>> dense("Ca(OH)2")
    'ca 1 o 2 h 2'
    >>> parse_dense("ca 1 o 2 h 2")
    {'Ca': 1, 'O': 2, 'H': 2}
"""

from __future__ import annotations

import re

_TOKEN = re.compile(r"([A-Z][a-z]?|\(|\)|\d+)")


def parse(formula: str) -> dict[str, int]:
    """Atom counts of a formula with optional brackets. Raises ``ValueError`` on bad input."""
    tokens = _TOKEN.findall(formula)
    if "".join(tokens) != formula:
        raise ValueError(f"bad formula {formula!r}")
    counts: dict[str, int] = {}
    stack: list[dict[str, int]] = [counts]
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t == "(":
            stack.append({})
        elif t == ")":
            group = stack.pop()
            mult = 1
            if i + 1 < len(tokens) and tokens[i + 1].isdigit():
                mult = int(tokens[i + 1])
                i += 1
            for el, n in group.items():
                stack[-1][el] = stack[-1].get(el, 0) + n * mult
        elif t.isdigit():
            raise ValueError(f"bad formula {formula!r}")
        else:
            n = 1
            if i + 1 < len(tokens) and tokens[i + 1].isdigit():
                n = int(tokens[i + 1])
                i += 1
            stack[-1][t] = stack[-1].get(t, 0) + n
        i += 1
    if len(stack) != 1 or not counts:
        raise ValueError(f"bad formula {formula!r}")
    return counts


def dense(formula: str) -> str:
    """``Ca(OH)2`` -> ``ca 1 o 2 h 2``: every element followed by its count, as digit tokens."""
    return " ".join(f"{el.lower()} {' '.join(str(n))}" for el, n in parse(formula).items())


def parse_dense(text: str) -> dict[str, int] | None:
    """The reverse of :func:`dense` (a missing count means 1); ``None`` if the text is not a dense formula."""
    words = text.split()
    counts: dict[str, int] = {}
    i = 0
    while i < len(words):
        el = words[i]
        if not el.isalpha() or len(el) > 2:
            return None
        digits = ""
        i += 1
        while i < len(words) and words[i].isdigit():
            digits += words[i]
            i += 1
        counts[el.capitalize()] = counts.get(el.capitalize(), 0) + (int(digits) if digits else 1)
    return counts or None


def same(a: dict[str, int], b: dict[str, int]) -> bool:
    return {k: v for k, v in a.items() if v} == {k: v for k, v in b.items() if v}
