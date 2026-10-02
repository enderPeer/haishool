"""Heritable behaviour: newborn mortality and the innate eat bias, against controls.

Conditions on the same seeds and config:
  frozen    Config(genome="frozen"): newborns start with blank controllers (Matrix rule)
  evolving  Config(genome="evolving"): innate biases, alpha and epsilon are inherited
            with mutation and seed the newborn controller
  neutral   the same genome evolves with the same mutation draws, but is NOT expressed
            (newborn controllers stay blank), so its genes drift without selection on
            behaviour: the control for "the innate eat bias rises by selection"

Measured per run (full in-memory event stream, nothing written):
  newborn mortality before age 50: non-founders born by tick T-50 that died before age 50
  innate P(eat | food in reach): from each newborn's genome alone (genome.innate_choice_
            probability, epsilon-greedy over its innate values, 24 food-in-reach contexts),
            by generation (founders are generation 0; a child is its parent's + 1)
  early P(eat | food in reach): newborns' actual decisions at ages 0-19, by tick window

usage: python -m haishool.life8.analysis.heritable [--seeds 1-6] [--ticks 2000] [--jobs 18] [--out FILE]
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .. import genome, learning
from ..config import Config
from ..world import World

CONDITIONS = ("frozen", "evolving", "neutral")
BASE = {"food_patches": 30, "population_limit": 256}
EARLY_AGE = 20
WINDOW = 500


def run(condition, seed, ticks=2000, config=None):
    overrides = {**BASE, **(config or {}), "genome": "frozen" if condition == "frozen" else "evolving"}
    cfg = Config(**overrides)
    original = genome.seed_controller
    if condition == "neutral":  # genes inherited and mutated, never expressed
        genome.seed_controller = lambda g, max_memories=32: learning.new_controller(max_memories=max_memories)
    try:
        world = World.create(seed=seed, config=cfg)
        generation = {identity: 0 for identity in world.agents}
        born = {identity: 0 for identity in world.agents}
        died = {}
        innate = {}
        early = {}  # window -> [near decisions, eat decisions]
        started = time.perf_counter()
        while world.tick < ticks and not world.stop_reason:
            for event in world.step():
                kind = event["type"]
                if kind == "birth":
                    child = event["agent"]
                    generation[child] = generation.get(event["parent"], 0) + 1
                    born[child] = event["tick"]
                    innate[child] = genome.innate_choice_probability(world.agents[child].genome, cfg)
                elif kind == "death":
                    died[event["agent"]] = event["tick"]
                elif kind == "transition" and generation.get(event["agent"], 0) > 0:
                    age = event["tick"] - 1 - born[event["agent"]]
                    if age < EARLY_AGE and event["observation"].get("food_near"):
                        row = early.setdefault(str((event["tick"] - 1) // WINDOW * WINDOW), [0, 0])
                        row[0] += 1
                        row[1] += event["action"]["type"] == "eat"
    finally:
        genome.seed_controller = original
    newborns = [a for a, g in generation.items() if g > 0 and born[a] <= world.tick - 50]
    dead_young = sum(1 for a in newborns if a in died and died[a] - born[a] < 50)
    by_window = {}
    for a in newborns:
        row = by_window.setdefault(str(born[a] // 1000 * 1000), [0, 0])
        row[0] += 1
        row[1] += a in died and died[a] - born[a] < 50
    by_generation = {}
    for child, p in innate.items():
        by_generation.setdefault(generation[child], []).append(p)
    return {"condition": condition, "seed": seed, "ticks_run": world.tick, "stop_reason": world.stop_reason or "step_budget",
            "births": world.counters.get("births", 0), "final_population": len(world.agents),
            "newborns_followed": len(newborns), "newborns_dead_before_50": dead_young,
            "newborn_mortality_before_50": dead_young / len(newborns) if newborns else None,
            "mortality_before_50_by_birth_window": {w: [n, d, d / n] for w, (n, d) in sorted(by_window.items(), key=lambda x: int(x[0]))},
            "max_generation": max(generation.values()),
            "innate_p_eat_near_by_generation": {str(g): [len(v), sum(v) / len(v)] for g, v in sorted(by_generation.items())},
            "early_p_eat_near_by_window": {w: [n, e / n if n else None] for w, (n, e) in sorted(early.items(), key=lambda x: int(x[0]))},
            "seconds": round(time.perf_counter() - started, 1)}


def _job(args):
    return run(*args)


def _bins(rows, generation_bins=((1, 3), (4, 7), (8, 11), (12, 15), (16, 19), (20, 10 ** 6))):
    out = {}
    for low, high in generation_bins:
        n = total = 0.
        for row in rows:
            for g, (count, mean) in row["innate_p_eat_near_by_generation"].items():
                if low <= int(g) <= high:
                    n += count
                    total += count * mean
        out[f"{low}-{high if high < 10 ** 6 else ''}"] = [int(n), total / n if n else None]
    return out


def measure(seeds=range(1, 7), ticks=2000, jobs=18, config=None):
    tasks = [(condition, seed, ticks, config) for condition in CONDITIONS for seed in seeds]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        rows = list(pool.map(_job, tasks))
    report = {"measurement": "newborn mortality before age 50 and innate P(eat | food in reach) by generation",
              "seeds": list(seeds), "ticks": ticks, "config": {**BASE, **(config or {})}, "runs": rows, "totals": {}}
    for condition in CONDITIONS:
        mine = [row for row in rows if row["condition"] == condition]
        followed = sum(row["newborns_followed"] for row in mine)
        dead = sum(row["newborns_dead_before_50"] for row in mine)
        windows = {}
        for row in mine:
            for w, (n, d, _) in row["mortality_before_50_by_birth_window"].items():
                windows.setdefault(w, [0, 0])
                windows[w][0] += n
                windows[w][1] += d
        report["totals"][condition] = {"newborns_followed": followed, "dead_before_50": dead,
                                       "mortality_before_50_by_birth_window": {w: [n, d, d / n] for w, (n, d) in
                                                                               sorted(windows.items(), key=lambda x: int(x[0]))},
                                       "mortality_before_50": dead / followed if followed else None,
                                       "innate_p_eat_near_by_generation_bin": _bins(mine)}
    by = {(row["condition"], row["seed"]): row for row in rows}
    report["paired_seeds_evolving_lower_mortality_than_frozen"] = sum(
        (by[("evolving", s)]["newborn_mortality_before_50"] or 1) < (by[("frozen", s)]["newborn_mortality_before_50"] or 1)
        for s in seeds)
    report["paired_seeds_evolving_lower_mortality_than_neutral"] = sum(
        (by[("evolving", s)]["newborn_mortality_before_50"] or 1) < (by[("neutral", s)]["newborn_mortality_before_50"] or 1)
        for s in seeds)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", default="1-6")
    parser.add_argument("--ticks", type=int, default=2000)
    parser.add_argument("--jobs", type=int, default=18)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    low, _, high = args.seeds.partition("-")
    report = measure(range(int(low), int(high or low) + 1), args.ticks, args.jobs)
    text = json.dumps(report, indent=1)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(json.dumps(report["totals"], indent=1))
    print("paired lower mortality vs frozen:", report["paired_seeds_evolving_lower_mortality_than_frozen"],
          "vs neutral:", report["paired_seeds_evolving_lower_mortality_than_neutral"], "of", len(report["seeds"]))


if __name__ == "__main__":
    main()
