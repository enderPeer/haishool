"""Founder survival and early eating, with the birth transfer kept out of / put back into the reward.

This re-runs the review's key learning measurement (matrix-review assessment, section 7
bug 1): default Config, seeds 1-6, 600 ticks. "Founders alive at age 300" counts the
24 founders of each world that had not died before age 300 (founders are born at tick 0,
so this is the population of founders at tick 300). P(eat | food in reach) pools the
founders' decisions while aged 50-99.

Conditions on the same seeds:
  fixed            life8 default: the birth transfer stays in the ledger, not in the reward
  birth_in_reward  intervention: Config(birth_transfer_in_reward=True), Matrix's old booking

With the default population_limit (128) a world whose founders survive well can reach
the capacity pause before tick 300; --population-limit 256 lets every run reach tick 300
(the trajectory up to the default cap is unchanged, the cap only stops a run).

usage: python -m haishool.life8.analysis.founders [--seeds 1-6] [--ticks 600] [--jobs 6]
                                                  [--population-limit N] [--out FILE]
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from ..config import Config
from ..world import World

CONDITIONS = {"fixed": {}, "birth_in_reward": {"birth_transfer_in_reward": True}}


def run(condition, seed, ticks=600, config=None):
    world = World.create(seed=seed, config=Config(**{**CONDITIONS[condition], **(config or {})}))
    founders = set(world.agents)
    died_at = {}
    near = eat_near = reproduce = decisions = 0
    started = time.perf_counter()
    while world.tick < ticks and not world.stop_reason:
        for event in world.step():
            kind = event["type"]
            if kind == "death" and event["agent"] in founders:
                died_at[event["agent"]] = event["tick"]
            elif kind == "transition" and event["agent"] in founders:
                age = event["tick"] - 1
                action = event["action"]["type"]
                if 50 <= age < 100:
                    decisions += 1
                    reproduce += action == "reproduce"
                    if event["observation"]["food_near"]:
                        near += 1
                        eat_near += action == "eat"
    reached = world.tick >= 300
    alive_300 = sum(1 for agent in founders if died_at.get(agent, 10 ** 9) >= 300) if reached else None
    return {"condition": condition, "seed": seed, "ticks_run": world.tick, "stop_reason": world.stop_reason or "step_budget",
            "founders": len(founders), "founders_alive_at_age_300": alive_300,
            "founder_eat_when_food_in_reach_age_50_99": eat_near / near if near else None,
            "founder_decisions_age_50_99": decisions, "founder_reproduce_share_age_50_99": reproduce / decisions if decisions else None,
            "births": world.counters.get("births", 0), "deaths": world.counters.get("deaths", 0),
            "final_population": len(world.agents), "seconds": round(time.perf_counter() - started, 1)}


def _job(args):
    return run(*args)


def measure(seeds=range(1, 7), ticks=600, jobs=6, config=None):
    tasks = [(condition, seed, ticks, config) for condition in CONDITIONS for seed in seeds]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        rows = list(pool.map(_job, tasks))
    report = {"measurement": "founders alive at age 300 and founder P(eat | food in reach) at age 50-99",
              "seeds": list(seeds), "ticks": ticks, "config_overrides": config or {}, "runs": rows, "totals": {}}
    for condition in CONDITIONS:
        mine = [row for row in rows if row["condition"] == condition]
        report["totals"][condition] = {
            "founders_alive_at_age_300": sum(row["founders_alive_at_age_300"] or 0 for row in mine),
            "founders": sum(row["founders"] for row in mine),
            "runs_stopped_before_300": sum(row["founders_alive_at_age_300"] is None for row in mine),
            "mean_eat_when_food_in_reach": sum(row["founder_eat_when_food_in_reach_age_50_99"] or 0 for row in mine) / len(mine)}
    paired = [(a["founders_alive_at_age_300"], b["founders_alive_at_age_300"]) for a in rows if a["condition"] == "fixed"
              for b in rows if b["condition"] == "birth_in_reward" and b["seed"] == a["seed"]]
    report["paired_seeds_fixed_higher"] = sum(a > b for a, b in paired if a is not None and b is not None)
    report["paired_seeds"] = len(paired)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", default="1-6")
    parser.add_argument("--ticks", type=int, default=600)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--population-limit", type=int,
                        help="raise the capacity pause (dynamics are identical until the default cap is reached)")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    low, _, high = args.seeds.partition("-")
    config = {"population_limit": args.population_limit} if args.population_limit else None
    report = measure(range(int(low), int(high or low) + 1), args.ticks, args.jobs, config)
    text = json.dumps(report, indent=1)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(json.dumps(report["totals"], indent=1), report["paired_seeds_fixed_higher"], "of", report["paired_seeds"])


if __name__ == "__main__":
    main()
