"""Input-conditioned lessons for the toy simulators, independent of rollout seeds.

These are *specified substeps* and conditional expectations, not claims that a
compressed cloud summary determines its next chaotic state. Gravity conditions
on velocities after the leapfrog step; planets asks only the solids sweep before
gas capture/collisions. Life probabilities integrate over the random draw rather
than smuggling its seed into a question. World lessons reuse its handoff rules.

Every answer is recalculated from the numbers actually printed in the prompt.
``gate(topic)`` implements the truth Gate API with topic ``predict_<topic>``.
``generate(rng, n, topic)`` returns exactly n questions. Independent RNG seeds do
not by themselves guarantee disjoint prompts: split builders MUST exclude the
canonical ``split_key`` of sealed questions from training and feedback.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable

from haishool.cosmos import chem, gravity, life, planets, world
from haishool.truth import Line, Verdict, is_finite, num, parse_num

TOPICS = ("nucleo", "gravity", "planets", "chem", "life", "world")
MAX_TOKENS = 128


@dataclass(frozen=True)
class Input:
    name: str
    low: float
    high: float
    integer: bool = False


@dataclass(frozen=True)
class Rule:
    topic: str
    inputs: tuple[Input, ...]
    compute: Callable[..., float | int | str]
    description: str


RULES = {
    "cooling_temperature": Rule("nucleo", (
        Input("temperature", 1, 1e10), Input("time", .01, 1000),
        Input("next_time", .01, 10000)),
        lambda temperature, time, next_time: temperature * math.sqrt(time / next_time),
        "temperature times sqrt time over next_time. fixed expansion parameters. kelvin and seconds"),
    "cooling_vertical": Rule("gravity", (
        Input("vertical_velocity", -5, 5), Input("mean_vertical_velocity", -5, 5),
        Input("cooling", 0, .6), Input("dt_tff", .0001, .1)),
        lambda v, mean, cooling, dt: v if cooling == 0 else (v - mean) / (1 + gravity.COOL_RATE * cooling * dt),
        "after leapfrog before cooling. zero cooling leaves velocity unchanged. otherwise vertical_velocity minus mass weighted mean_vertical_velocity divided by one plus four times cooling times dt_tff. cooling and momentum correction only"),
    "solids_remaining": Rule("planets", (
        Input("remaining", 0, 100), Input("efficiency", .6, 1.4),
        Input("orbit", .3, 30), Input("star_mass", .03, 2), Input("dt", 0, 2)),
        lambda remaining, efficiency, orbit, mass, dt:
            remaining * math.exp(-efficiency * dt / (planets.TAU_SWEEP * math.sqrt(orbit ** 3 / mass))),
        "solids sweep only before collisions. remaining times exp minus efficiency times dt over tau_sweep times sqrt orbit cubed over star_mass. tau_sweep 0 point 2. masses earth and solar units. orbit au. dt myr"),
    "schedule_temperature": Rule("chem", (
        Input("t_start", 1, 10000, True), Input("t_end", 1, 10000, True),
        Input("step", 0, 60, True), Input("steps", 1, 60, True)),
        chem.temperature_at,
        "temperature at step in cooling schedule. t_start times t_end over t_start raised to step over steps. rounded kelvin"),
    "survival_probability": Rule("life", (
        Input("length", 2, life.MAX_LEN, True), Input("decay", 0, 4)),
        lambda length, decay: max(0.0, 1 - decay / length),
        "one chain survives decay phase with probability max zero one minus decay over length. before polymerisation and replication"),
    "expected_mutations": Rule("life", (
        Input("length", 2, life.MAX_LEN, True), Input("mu", 0, 1)),
        lambda length, mu: length * mu,
        "expected changed letters in one attempted copy equals length times mu. conditional on attempting that copy before resource checks"),
    "next_total": Rule("life", (
        Input("total", 0, 1000000, True), Input("inflow", 0, 10000, True)),
        lambda total, inflow: total + inflow,
        "next total free and bound letters equals total plus inflow. inflow is a multiple of four. all other phases conserve letters"),
}


def _parse(prompt: str) -> tuple[str, tuple[float | int, ...]] | None:
    words = prompt.split()
    if len(words) < 4 or words[1] != "predict" or words[2] not in RULES:
        return None
    name = words[2]
    rule = RULES[name]
    if words[0] != rule.topic:
        return None
    pos, values = 3, []
    for spec in rule.inputs:
        if pos >= len(words) or words[pos] != spec.name:
            return None
        start = pos = pos + 1
        while pos < len(words) and words[pos] not in {s.name for s in rule.inputs}:
            pos += 1
        value = parse_num(words[start:pos])
        if not is_finite(value) or not spec.low <= value <= spec.high:
            return None
        if spec.integer and int(value) != value:
            return None
        values.append(int(value) if spec.integer else value)
    if pos != len(words):
        return None
    if name == "cooling_temperature" and values[2] <= values[1]:
        return None
    if name == "schedule_temperature" and (values[2] > values[3] or values[1] > values[0]):
        return None
    if name == "next_total" and values[1] % 4:
        return None
    return name, tuple(values)


def owns(prompt: str) -> bool:
    return _parse(prompt) is not None or (prompt.startswith("world predict handoff ")
                                         and world.owns(prompt.replace(" predict", "", 1)))


def check(prompt: str, answer: str) -> Verdict:
    if prompt.startswith("world predict handoff "):
        return world.check(prompt.replace(" predict", "", 1), answer)
    parsed = _parse(prompt)
    if parsed is None:
        return Verdict(False, None, "not a valid input conditioned question")
    name, values = parsed
    expected = RULES[name].compute(*values)
    rendered = num(expected)
    got = parse_num(answer)
    # Compare at the target's declared four-significant-digit precision.
    # Integer targets (counts and the chemistry schedule) are exact.
    if isinstance(expected, int):
        ok = is_finite(got) and got == expected
    else:
        ok = is_finite(got) and num(float(got)) == rendered
    return Verdict(bool(ok), rendered, "computed from supplied inputs")


def split_key(prompt: str) -> str:
    """Canonical input identity: variants of numeric spelling share a split.

    Call before any optional explanation/trace is appended to an answer; no answer
    or solution trace participates in the identity.
    """
    parsed = _parse(prompt)
    if parsed is not None:
        name, values = parsed
        rule = RULES[name]
        return f"{rule.topic} predict {name} " + " ".join(
            f"{spec.name} {num(value, sig=12)}" for spec, value in zip(rule.inputs, values))
    if prompt.startswith(("world handoff ", "world predict handoff ")):
        parsed_world = world.parse(prompt.replace(" predict", "", 1))
        if parsed_world and parsed_world[0] == "handoff":
            _, name, values = parsed_world
            return "world handoff " + name + " " + " ".join(
                f"{key} {num(value, sig=12)}" for key, value in zip(world.HANDOFFS[name].inputs, values))
    raise ValueError("not a valid prediction prompt")


def _line(name: str, values: list[float | int]) -> Line:
    rule = RULES[name]
    prompt = f"{rule.topic} predict {name} " + " ".join(
        f"{spec.name} {num(value)}" for spec, value in zip(rule.inputs, values))
    verdict = check(prompt, "")
    if verdict.expected is None:
        raise ValueError(f"invalid generated prompt: {prompt}")
    return Line(prompt, verdict.expected, "predict_" + rule.topic, "calc",
                {"conditioning": "explicit_inputs", "split_key": split_key(prompt)})


def generate(rng: random.Random, n: int, topic: str | None = None) -> list[Line]:
    """Fresh bounded lessons; callers must remove cross-split input collisions."""
    if topic is not None and topic not in TOPICS:
        raise ValueError(f"unknown topic {topic}")
    if n < 0:
        raise ValueError("n must be nonnegative")
    out = []
    for _ in range(n):
        selected = topic or rng.choice(TOPICS)
        if selected == "world":
            source = world.practice_lines(rng, 1)[0]
            # Recompute from the serialized inputs, just as the gate does.
            answer = world.check(source.prompt, "").expected
            prompt = source.prompt.replace("world handoff", "world predict handoff", 1)
            out.append(Line(prompt, answer, "predict_world", "calc",
                            {"conditioning": "explicit_inputs", "split_key": split_key(prompt)}))
            continue
        name = rng.choice([key for key, rule in RULES.items() if rule.topic == selected])
        values = [rng.randint(int(s.low), int(s.high)) if s.integer else rng.uniform(s.low, s.high)
                  for s in RULES[name].inputs]
        if name == "cooling_temperature":
            values[2] = min(10000, values[1] + rng.uniform(1, 8000))
        if name == "schedule_temperature":
            values[0], values[1] = max(values[:2]), min(values[:2])
            values[2] = rng.randint(0, values[3])
        if name == "next_total":
            values[1] -= values[1] % 4
        out.append(_line(name, values))
    return out


def records(topic: str | None = None) -> list[Line]:
    if topic is not None and topic not in TOPICS:
        raise ValueError(f"unknown topic {topic}")
    out = [Line(f"{rule.topic} predict {name}. {rule.description}.",
                topic="predict_" + rule.topic, kind="record")
           for name, rule in RULES.items() if topic is None or rule.topic == topic]
    if topic is None or topic == "world":
        out.extend(Line(row.prompt.replace("world handoff", "world predict handoff", 1),
                        topic="predict_world", kind="record") for row in world.records())
    return out


class PredictionGate:
    def __init__(self, topic: str):
        if topic not in TOPICS:
            raise ValueError(f"unknown topic {topic}")
        self.sim = topic
        self.topic = "predict_" + topic
        self.KEYS = {name: rule.description for name, rule in RULES.items() if rule.topic == topic}
        if topic == "world":
            self.KEYS = {name: "explicit handoff inputs" for name in world.HANDOFFS}

    def records(self) -> list[Line]:
        return records(self.sim)

    def generate(self, rng: random.Random, n: int) -> list[Line]:
        return generate(rng, n, self.sim)

    def owns(self, prompt: str) -> bool:
        return prompt.startswith(self.sim + " ") and owns(prompt)

    def check(self, prompt: str, answer: str) -> Verdict:
        return check(prompt, answer) if self.owns(prompt) else Verdict(False, None, "not my question")


def gate(topic: str) -> PredictionGate:
    return PredictionGate(topic)
