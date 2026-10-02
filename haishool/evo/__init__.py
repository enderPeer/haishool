"""Round 7: from replicators to societies, and a corrected chain.

Round 6 stops at molecules that copy themselves. Round 7 continues the ladder with the major
transitions of evolution, each as a small simulation with real, named rules, and repairs the
hand-offs and toy rules of round 6 that a review found wrong or arbitrary.

     7  stars     generations of stars: lifetimes, supernovae, the elements they return
     8  cells     replicators inside compartments: genomes, metabolism, an energy budget
     9  bodies    cells that stay together: size, cell types, who eats whom
    10  senses    bodies with sensors and nerves: what sensing costs and when it pays
    11  signals   senders and receivers settle on shared signals: a lexicon, then grammar
    12  society   groups that cooperate, teach and accumulate knowledge
    13  world7    the corrected chain, from the first second to a society, on one clock

These are toy models of evolution, not a reconstruction of the history of life: nobody knows how
life began, how nervous systems arose in detail, or how language started. What each level does
is implement a published, simple model whose every number can be recomputed (the module
docstring names the model and says what is taken from it and what is invented): the error
threshold and the stochastic corrector, Lotka-Volterra dynamics, Hamilton's rule, the Lewis
signalling game, the naming game, iterated learning, public-goods games, cumulative culture.

**Round 6 stays as it is.** The version-5 models were trained on round-6 rollouts, and those are
committed reproducible byte for byte. A correction to a round-6 level is therefore an option,
never a change of the default: ``run(seed, rules=7, ...)`` and ``random_params(rng, rules=7)``
apply it, and without ``rules`` (or with ``rules=6``) every output is bit-identical to before.
The round-6 tests are the guard and are not edited.

A level is a :class:`haishool.cosmos.Simulation` (``run``, ``conserved``, ``check``, ``owns``,
module functions ``simulation()``, ``random_params(rng)``, ``lines(rollout)``) and, like round 6,
gives two kinds of training lines:

* the story of one seed, to be recalled: ``signals seed 7 step 1 2. success 0 point 8 1. words 9.``
* **lessons** that carry their inputs, so the answer follows from the question alone and a model
  can learn the rule (:class:`LessonGate`, same form as ``haishool.cosmos.predict``):

      q society predict hamilton relatedness 0 point 5 benefit 4 cost 1. a yes.
      q signals predict expected_success signals 4 states 4 shared 3. a 0 point 7 5.

  A lesson is a specified step or an exact expectation of the model, never a guess about a
  chaotic outcome. The simulation itself must use the same functions, so a lesson is true of it.

Language levels also *speak*: an utterance is a string of syllables from the level's own
phoneme inventory (``ka``, ``mi tu``), and its meaning is checkable against the lexicon and
grammar the population has settled on:

      q signals seed 7 say predator near. a ki ta.
      q signals seed 7 meaning ki ta. a predator near.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable

from haishool.truth import Line, Verdict, is_finite, num, parse_num

LEVELS = ("stars", "cells", "bodies", "senses", "signals", "society", "world7")
#: a lesson may be longer than a round-5 line: the final trainer uses a 256-token context
MAX_TOKENS = 128


@dataclass(frozen=True)
class Input:
    """One named input of a lesson and the range it is drawn from."""
    name: str
    low: float
    high: float
    integer: bool = False
    #: decimals kept when a value is drawn, so prompts stay short and exactly reproducible
    places: int = 3


@dataclass(frozen=True)
class Rule:
    """An input-conditioned lesson: ``compute(*inputs)`` is the truth.

    ``compute`` returns a number (judged at 4 significant digits; integers exactly) or a word
    (judged exactly). ``valid`` rejects input combinations the rule does not cover.
    """
    inputs: tuple[Input, ...]
    compute: Callable[..., float | int | str]
    description: str
    valid: Callable[..., bool] | None = None


class LessonGate:
    """The truth gate of one level's lessons: topic ``predict_<sim>``.

    Prompts are ``<sim> predict <rule> <input> <value> ...`` with every input named, in the
    rule's order. Implements the round-5 Gate protocol plus ``split_key`` (the identity that keeps
    one question on one side of a train/dev/sealed split whatever its number spelling).
    """

    def __init__(self, sim: str, rules: dict[str, Rule]) -> None:
        self.sim = sim
        self.topic = "predict_" + sim
        self.rules = rules
        self.KEYS = {name: rule.description for name, rule in rules.items()}

    def parse(self, prompt: str) -> tuple[str, tuple[float | int, ...]] | None:
        words = prompt.split()
        if len(words) < 3 or words[0] != self.sim or words[1] != "predict" or words[2] not in self.rules:
            return None
        rule = self.rules[words[2]]
        names = {s.name for s in rule.inputs}
        pos, values = 3, []
        for spec in rule.inputs:
            if pos >= len(words) or words[pos] != spec.name:
                return None
            start = pos = pos + 1
            while pos < len(words) and words[pos] not in names:
                pos += 1
            value = parse_num(words[start:pos])
            if not is_finite(value) or not spec.low <= value <= spec.high:
                return None
            if spec.integer and int(value) != value:
                return None
            values.append(int(value) if spec.integer else value)
        if pos != len(words) or (rule.valid and not rule.valid(*values)):
            return None
        return words[2], tuple(values)

    def owns(self, prompt: str) -> bool:
        return self.parse(prompt) is not None

    def expected(self, name: str, values: tuple[float | int, ...]) -> str:
        value = self.rules[name].compute(*values)
        return value if isinstance(value, str) else num(value)

    def check(self, prompt: str, answer: str) -> Verdict:
        parsed = self.parse(prompt)
        if parsed is None:
            return Verdict(False, None, "not my question")
        name, values = parsed
        truth = self.rules[name].compute(*values)
        rendered = truth if isinstance(truth, str) else num(truth)
        if isinstance(truth, str):
            ok = answer.strip() == truth
        else:
            got = parse_num(answer)
            ok = is_finite(got) and (got == truth if isinstance(truth, int) and not isinstance(truth, bool)
                                     else num(float(got)) == rendered)
        return Verdict(bool(ok), rendered, "computed from the inputs in the question")

    def split_key(self, prompt: str) -> str:
        parsed = self.parse(prompt)
        if parsed is None:
            raise ValueError("not a lesson of this level")
        name, values = parsed
        return f"{self.sim} predict {name} " + " ".join(
            f"{spec.name} {num(value, sig=12)}" for spec, value in zip(self.rules[name].inputs, values))

    def line(self, name: str, values: list[float | int]) -> Line | None:
        prompt = f"{self.sim} predict {name} " + " ".join(
            f"{spec.name} {num(value, sig=6)}" for spec, value in zip(self.rules[name].inputs, values))
        parsed = self.parse(prompt)  # the answer is computed from the numbers as printed
        if parsed is None:
            return None
        return Line(prompt, self.expected(*parsed), self.topic, "calc",
                    {"conditioning": "explicit_inputs", "split_key": self.split_key(prompt)})

    def generate(self, rng: random.Random, n: int) -> list[Line]:
        """Exactly ``n`` lessons, rules drawn uniformly, inputs uniformly in their ranges."""
        names = sorted(self.rules)
        out: list[Line] = []
        tries = 0
        while len(out) < n:
            tries += 1
            if tries > 50 * n + 1000:
                raise RuntimeError(f"{self.sim}: the lesson rules reject almost every drawn input")
            name = rng.choice(names)
            values = [rng.randint(int(s.low), int(s.high)) if s.integer
                      else round(rng.uniform(s.low, s.high), s.places) for s in self.rules[name].inputs]
            ln = self.line(name, values)
            if ln is not None:
                out.append(ln)
        return out

    def records(self) -> list[Line]:
        """One record per rule saying what it computes, in dense words."""
        return [Line(f"{self.sim} predict {name}. {rule.description}.", topic=self.topic, kind="record")
                for name, rule in sorted(self.rules.items())]


def sig(x: float, digits: int = 3) -> float:
    """Round to significant digits, the form in which a metric is printed and handed on."""
    if x == 0 or not math.isfinite(x):
        return 0.0 if x == 0 else x
    return round(x, digits - 1 - int(math.floor(math.log10(abs(x)))))
