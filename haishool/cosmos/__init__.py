"""Round 6: a toy universe, one level on top of the other.

No computer can simulate the real universe from the first second down to living cells. What
can be done is a chain of small simulations with real rules, each simplified, each feeding the
next, and each one its own truth gate: the model learns to predict what the simulation does,
and the simulation judges the prediction.

    0  tables     the elements and the forces (round 5)
    1  nucleo     an expanding, cooling universe: protons and neutrons become hydrogen, helium
    2  gravity    a rotating gas cloud collapses into a disc with spiral arms and clumps
    3  planets    the disc's clumps grow into planets; rock near the star, gas and ice far out
    4  chem       atoms bond by valence rules into molecules (water, methane, ammonia, ...)
    5  life       molecules that copy themselves, with mutation and selection
    6  world      the chain: nucleo -> gravity -> planets -> chem -> life, one timeline

Every simulation is deterministic given a seed and parameters, and produces a :class:`Rollout`:
a list of steps, each a dict of named numbers. Conservation laws (mass, charge, baryon number,
energy or angular momentum within a tolerance) are its gate. The dense lines (same alphabet as
round 5, numbers as digits) are:

    gravity seed 7 step 2 0. radius 3 point 2. flattening 0 point 8. clumps 4. spiral yes.
    q gravity seed 7 step 2 0 clumps. a 4.
    q gravity seed 7 step 2 0 next clumps. a 5.             (what happens next)
    q gravity seed 7 final clumps. a 6.                      (how it ends)
    world seed 3 era planets. rocky 2. gas 1. ice 1.

Held-out seeds test whether the model has learned the rules or only the rollouts it saw.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from haishool.truth import Line, Verdict, is_finite, num


@dataclass
class Rollout:
    sim: str
    seed: int
    params: dict[str, float | int | str]
    steps: list[dict[str, float | int | str]]
    #: numbers summarising the whole run (``final_clumps``, ``helium_fraction``)
    summary: dict[str, float | int | str] = field(default_factory=dict)

    def value(self, text: float | int | str) -> str:
        return text if isinstance(text, str) else num(text)


def state_line(r: Rollout, t: int, keys: list[str] | None = None) -> Line:
    """``gravity seed 7 step 2 0. radius 3 point 2. clumps 4.``"""
    step = r.steps[t]
    keys = keys or list(step)
    fields = " ".join(f"{k} {r.value(step[k])}." for k in keys)
    return Line(f"{r.sim} seed {num(r.seed)} step {num(t)}. {fields}", topic=r.sim, kind="record")


def query_lines(r: Rollout, t: int, keys: list[str] | None = None, horizon: int = 1) -> list[Line]:
    """One question per metric at step ``t``, and what it is ``horizon`` steps later."""
    step = r.steps[t]
    keys = keys or list(step)
    out = [Line(f"{r.sim} seed {num(r.seed)} step {num(t)} {k}", r.value(step[k]), r.sim, "fact") for k in keys]
    if t + horizon < len(r.steps):
        nxt = r.steps[t + horizon]
        out += [Line(f"{r.sim} seed {num(r.seed)} step {num(t)} next {k}", r.value(nxt[k]), r.sim, "calc")
                for k in keys if k in nxt]
    return out


def summary_lines(r: Rollout) -> list[Line]:
    return [Line(f"{r.sim} seed {num(r.seed)} final {k}", r.value(v), r.sim, "calc") for k, v in r.summary.items()]


@runtime_checkable
class Simulation(Protocol):
    """One level of the toy universe."""

    sim: str
    #: metric -> meaning and unit (the model only sees the metric names)
    KEYS: dict[str, str]

    def run(self, seed: int, **params) -> Rollout:
        """Deterministic for a given seed and parameters."""

    def conserved(self, r: Rollout) -> Verdict:
        """The simulation's own truth gate: the conservation laws it must obey."""

    def check(self, prompt: str, answer: str) -> Verdict:
        """Replay the rollout a prompt names and compare, with this simulation's tolerance."""

    def owns(self, prompt: str) -> bool:
        """True when ``prompt`` is a question about this simulation."""


def close(a: float, b: float, rel: float = 0.05, abs_: float = 1e-9) -> bool:
    """True when two numbers agree within ``rel`` of the larger (or within ``abs_``). A number no
    float can hold is close to nothing: infinity would otherwise be within 5 percent of itself,
    so ``1 e 9 9 9`` would pass as any answer.

    >>> close(100, 104), close(100, 106), close(float("inf"), 5.0), close(10 ** 400, 5.0)
    (True, False, False, False)
    """
    if not (is_finite(a) and is_finite(b)):
        return False
    return abs(a - b) <= max(abs_, rel * max(abs(a), abs(b)))


def seeds(rng: random.Random, n: int, lo: int = 1, hi: int = 9999) -> list[int]:
    return rng.sample(range(lo, hi), n)
