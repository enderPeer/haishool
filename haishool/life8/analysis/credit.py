"""Multi-step credit measurements (review gap 4, assessment section 6 step 3).

Three probes, each run with the same seeds under learner profiles that switch the new
mechanisms off (learning.PROFILES / learning.use_profile):

* corridor   A one-dimensional corridor driven through learning.py directly. Food sits
             east of the learner; reward is 1 for eating it within reach and 0 for
             every other step (no step cost, no shaping). Measured: the greedy
             probability of move_e in the visible-but-out-of-reach states after N
             episodes (ties split evenly; 10 actions, so chance is 0.1).
* sparse     The review's sparse-food world (exp_learning3.py "sparse"): food_patches 15,
             population 16, automatic_reproduction off, tools/communication/culture off,
             max_age 2000, food_capacity 60, 1500 ticks. Measured: founders that starved,
             and P(move toward | food visible but out of reach) by founder age. A
             three-line heuristic (eat if in reach, else step toward visible food, else a
             random step) runs on the same seeds.
* default    Default Config worlds (population_limit raised so runs are not stopped by
             the capacity pause). Measured: P(move toward | food visible, out of reach)
             and P(eat | food in reach) for all organisms by age in decisions.

Measurement only: decisions are recorded by wrapping learning.learn inside the worker
process, which passes every argument through unchanged (the wrapped and unwrapped runs
have identical state hashes; see tests/life8/test_learning_credit.py).

usage: python -m haishool.life8.analysis.credit corridor|sparse|default [--seeds 1-6]
       [--ticks N] [--episodes N] [--profiles multistep,one_step,...] [--jobs N] [--out FILE]
"""
from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path

from .. import learning
from ..config import Config
from ..world import DIRECTIONS, World

# Profiles compared: the new default, the old rule, and single-mechanism variants.
VARIANTS = {
    "multistep": ("multistep", {}),
    "one_step": ("one_step", {}),
    "traces_only": ("one_step", {"lam": .8}),
    "no_traces": ("multistep", {"lam": 0.}),
    "no_conjunctions": ("multistep", {"conjunctions": False}),
    "no_egocentric": ("multistep", {"egocentric": False}),
    "no_replay": ("multistep", {"replay": 0}),
    "replay_8": ("multistep", {"replay": 8}),
    "replay_only": ("one_step", {"replay": 4}),
    "gain5": ("multistep", {"conjunction_gain": 5.}),
    "alpha_0.1": ("multistep", {"alpha": .1}),
    "alpha_0.1_replay8": ("multistep", {"alpha": .1, "replay": 8}),
    "gain5_replay0": ("multistep", {"conjunction_gain": 5., "replay": 0}),
    "gain5_replay2": ("multistep", {"conjunction_gain": 5., "replay": 2}),
    "gain10_replay2": ("multistep", {"conjunction_gain": 10., "replay": 2}),
    "egocentric_only": ("one_step", {"egocentric": True}),
    "constant_epsilon": ("multistep", {"epsilon": .15, "epsilon_floor": 1., "epsilon_halflife": 0}),
}
for _lam in (.5, .7, .9):
    VARIANTS[f"lam_{_lam}"] = ("multistep", {"lam": _lam})

CORRIDOR_ACTIONS = ("rest", "move_n", "move_ne", "move_e", "move_se", "move_s", "move_sw", "move_w", "move_nw", "eat")
SPARSE = {"food_patches": 15, "population": 16, "automatic_reproduction": False, "tools": False,
          "communication": False, "culture": False, "max_age": 2000, "food_capacity": 60.}


@contextmanager
def _variant(name):
    profile, overrides = VARIANTS[name]
    with learning.use_profile(profile, **overrides):
        yield


def _corridor_observation(distance, sight):
    visible = distance <= sight
    return {"food_direction": "e" if visible else "none", "food_near": distance <= 1,
            "food": [{"distance": float(distance)}] if visible else [], "energy_band": "mid"}


def corridor(variant, seed, episodes=60, *, length=10, sight=5, start=(2, 5), max_steps=30):
    """Episodes in a corridor; returns greedy P(move_e) in the visible, out-of-reach states."""
    rng = random.Random(seed)
    with _variant(variant):
        controller = learning.new_controller()
    lengths = []
    for _ in range(episodes):
        distance = rng.randint(*start)
        for step in range(max_steps):
            observation = _corridor_observation(distance, sight)
            action = learning.choose_action(controller, observation, rng, actions=CORRIDOR_ACTIONS)
            reward, terminal = 0., False
            if action == "eat" and distance <= 1:
                reward, terminal = 1., True
            elif action == "move_e":
                distance = max(1, distance - 1)
            elif action == "move_w":
                distance = min(length, distance + 1)
            following = _corridor_observation(distance, sight)
            last = step == max_steps - 1
            learning.learn(controller, observation, action, reward, following, terminal=terminal,
                           actions=CORRIDOR_ACTIONS)
            if terminal or last:
                lengths.append(step + 1)
                break
    toward = []
    for distance in range(start[0], start[1] + 1):
        values = learning.action_values(controller, _corridor_observation(distance, sight), CORRIDOR_ACTIONS)
        best = max(values.values())
        ties = [action for action, value in values.items() if value == best]
        toward.append(("move_e" in ties) / len(ties))
    return {"variant": variant, "seed": seed, "episodes": episodes, "p_toward_greedy": sum(toward) / len(toward),
            "mean_steps_last_10": sum(lengths[-10:]) / len(lengths[-10:])}


@contextmanager
def _recording(rows):
    """Record (controller id, decisions, food_direction, food_near, action) per learn call."""
    original = learning.learn

    def learn(controller, observation, action, reward, next_observation, enabled=True, **options):
        rows.append((id(controller), controller["decisions"], observation.get("food_direction"),
                     observation.get("food_near"), action))
        return original(controller, observation, action, reward, next_observation, enabled, **options)

    learning.learn = learn
    try:
        yield
    finally:
        learning.learn = original


AGE_BINS = ((0, 50), (50, 100), (100, 200), (200, 400), (400, 800), (800, 1600))


def _rates(rows, keep):
    out = {}
    for low, high in AGE_BINS:
        far = toward = near = eat = 0
        for ident, age, direction, food_near, action in rows:
            if not keep(ident) or not low < age <= high:
                continue
            if food_near:
                near += 1
                eat += action == "eat"
            elif direction not in (None, "none"):
                far += 1
                toward += action == "move_" + direction
        if far or near:
            out[f"{low}-{high}"] = {"p_toward_given_visible_out_of_reach": toward / far if far else None, "n_far": far,
                                    "p_eat_given_in_reach": eat / near if near else None, "n_near": near}
    return out


def world_probe(variant, seed, ticks, config, heuristic=False):
    rows = []
    started = time.perf_counter()
    with _variant(variant if variant != "heuristic" else "multistep"), _recording(rows):
        world = World.create(seed=seed, config=Config(**config))
        founders = {id(agent.controller): identity for identity, agent in world.agents.items()}
        rng = random.Random(seed + 7)
        starved = died = 0
        death_ages = []
        while world.tick < ticks and not world.stop_reason:
            actions = None
            if heuristic:
                actions = {}
                for identity, agent in world.agents.items():
                    seen = world.observe(agent)
                    if seen["food_near"]:
                        actions[identity] = "eat"
                    elif seen["food_direction"] != "none":
                        actions[identity] = "move_" + seen["food_direction"]
                    else:
                        actions[identity] = "move_" + rng.choice(sorted(DIRECTIONS))
            for event in world.step(actions):
                if event["type"] == "death" and event["agent"] in founders.values():
                    died += 1
                    starved += event.get("reason") == "starvation"
                    death_ages.append(event["tick"])
    founder_ids = set(founders)
    return {"variant": variant, "seed": seed, "ticks_run": world.tick,
            "stop_reason": world.stop_reason or "step_budget", "founders": len(founders),
            "founders_starved": starved, "founders_died": died,
            "median_founder_death_tick": sorted(death_ages)[len(death_ages) // 2] if death_ages else None,
            "final_population": len(world.agents), "births": world.counters.get("births", 0),
            "founder_rates": _rates(rows, founder_ids.__contains__),
            "all_rates": _rates(rows, lambda _: True), "seconds": round(time.perf_counter() - started, 1)}


def _job(args):
    probe, variant, seed, size, config = args
    if probe == "corridor":
        return corridor(variant, seed, size)
    return world_probe(variant, seed, size, config, heuristic=variant == "heuristic")


def measure(probe, variants, seeds, size, jobs=6, config=None):
    if probe == "sparse":
        config = {**SPARSE, "log": "compact", **(config or {})}
    elif probe == "default":
        config = {"log": "compact", "population_limit": 256, **(config or {})}
    tasks = [(probe, variant, seed, size, config) for variant in variants for seed in seeds]
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            rows = list(pool.map(_job, tasks))
    else:
        rows = [_job(task) for task in tasks]
    report = {"probe": probe, "seeds": list(seeds), "size": size, "config": config, "runs": rows, "totals": {}}
    for variant in variants:
        mine = [row for row in rows if row["variant"] == variant]
        if probe == "corridor":
            report["totals"][variant] = {
                "mean_p_toward_greedy": sum(row["p_toward_greedy"] for row in mine) / len(mine),
                "seeds_above_half": sum(row["p_toward_greedy"] > .5 for row in mine)}
            continue
        pooled = {}
        for key in ("founder_rates", "all_rates"):
            bins = {}
            for row in mine:
                for name, cell in row[key].items():
                    slot = bins.setdefault(name, [0., 0, 0., 0])
                    if cell["p_toward_given_visible_out_of_reach"] is not None:
                        slot[0] += cell["p_toward_given_visible_out_of_reach"] * cell["n_far"]
                        slot[1] += cell["n_far"]
                    if cell["p_eat_given_in_reach"] is not None:
                        slot[2] += cell["p_eat_given_in_reach"] * cell["n_near"]
                        slot[3] += cell["n_near"]
            pooled[key] = {name: {"p_toward": round(s[0] / s[1], 4) if s[1] else None, "n_far": s[1],
                                  "p_eat_near": round(s[2] / s[3], 4) if s[3] else None, "n_near": s[3]}
                           for name, s in bins.items()}
        report["totals"][variant] = {
            "founders_starved": sum(row["founders_starved"] for row in mine),
            "founders": sum(row["founders"] for row in mine), "births": sum(row["births"] for row in mine),
            "final_population": sum(row["final_population"] for row in mine), **pooled}
    return report


def _seeds(text):
    if "-" in text:
        low, high = text.split("-")
        return range(int(low), int(high) + 1)
    return [int(part) for part in text.split(",")]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("probe", choices=("corridor", "sparse", "default"))
    parser.add_argument("--seeds", default="1-6")
    parser.add_argument("--size", type=int, default=None, help="episodes (corridor) or ticks (worlds)")
    parser.add_argument("--profiles", default="multistep,one_step")
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    size = args.size or {"corridor": 60, "sparse": 1500, "default": 600}[args.probe]
    report = measure(args.probe, args.profiles.split(","), _seeds(args.seeds), size, args.jobs)
    text = json.dumps(report, indent=1, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(json.dumps(report["totals"], indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
