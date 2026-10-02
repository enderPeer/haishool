"""Shared question identities for training splits and feedback exclusions.

A calculation, its worked answer, a check of that answer and a judge of any of
those must stay on the same side of a split. These keys are conservative: even
a false check is grouped with the calculation whose answer it constrains.
"""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from typing import Iterable


@lru_cache(maxsize=1)
def _gates() -> tuple:
    # Imported only for check wrappers; ordinary prompts need no gate imports.
    return tuple(import_module(f"haishool.truth.{name}").gate()
                 for name in ("maths", "elements", "substances", "reactions", "forces"))


def canonical_prompt(text: str, gates: Iterable | None = None) -> str:
    """The bare underlying question of a prompt or dense q/a line.

    Unknown question forms retain their own identity. Optional gates let custom
    generators use the same grouping without depending on the built-in topics.
    """
    prompt = " ".join(text.split()).split(". a ", 1)[0].rstrip(".")
    if prompt.startswith("q "):
        prompt = prompt[2:]
    words = prompt.split()
    while len(words) > 3 and words[0] == "judge" and "answer" in words[2:]:
        words = words[1:words.index("answer", 2)]
    if words and words[-1] == "steps":
        words = words[:-1]
    prompt = " ".join(words)
    if (len(words) > 1 and words[1] == "predict") or prompt.startswith("world handoff "):
        from haishool.cosmos import predict
        try:
            return predict.split_key(prompt)
        except ValueError:
            return prompt
    if not words or words[0] != "check" or len(words) < 3:
        return prompt
    rest = words[1:]
    candidates = []
    if "x" in rest:
        candidates.append("solve " + " ".join(rest[:len(rest) - 1 - rest[::-1].index("x")]))
    if "equals" in rest:
        inner = " ".join(rest[:rest.index("equals")])
        candidates.extend(("calc " + inner, inner))
    candidates.extend(" ".join(rest[:cut]) for cut in range(2, len(rest)))
    available = tuple(_gates() if gates is None else gates)
    for candidate in candidates:
        for gate in available:
            try:
                if gate.owns(candidate) and gate.check(candidate, "").expected is not None:
                    return canonical_prompt(candidate, available)
            except Exception:
                continue
    return prompt
