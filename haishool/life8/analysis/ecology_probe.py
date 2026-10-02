"""Diagnostic probe for the living-world null: is the receiver side or the sender side the bottleneck?

"Oracle senders" (a supplied mapping, used ONLY as a diagnostic intervention and never
as an engine mechanism): any individual with a voice that sees a predator calls
signal_0, and one that sees a bloom calls signal_1, unless its own call is still live.
Everyone else acts on its own controller. The same seeds run with the normal and the
scrambled channel, so the only difference is whether the heard symbol carries the
oracle's content. If receivers learn to use oracle calls, receiver information rises
above the scrambled run and predation or bloom intake improves; if they do not, the
receiver side (or the ecology) is the bottleneck, not the sender credit route.

  python -m haishool.life8.analysis.ecology_probe --seeds 1-6 --ticks 1200 --jobs 12 --out FILE
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

from .. import discovery
from ..config import LIVING, Config
from ..world import World

KEEP = ("signal", "heard", "birth", "death", "predation")


def run(seed, ticks, channel, oracle=True, overrides=None):
    world = World.create(seed, Config(**{**LIVING, **(overrides or {}), "log": "compact", "channel": channel}))
    events, agent_ticks, window = [], 0, ticks // 2
    strikes = blooms = window_ticks = 0
    while world.tick < ticks and not world.stop_reason:
        forced = {}
        if oracle:
            for identity, agent in world.agents.items():
                if not agent.anatomy.voice:
                    continue
                obs = world.observe(agent)
                if obs.get("own_call") is not None:
                    continue
                if obs.get("danger_seen"):
                    forced[identity] = "signal_0"
                elif obs.get("bloom_seen"):
                    forced[identity] = "signal_1"
        before_strikes = world.counters.get("predator_strikes", 0)
        before_blooms = world.counters.get("bloom_meals", 0)
        population = len(world.agents)
        step = world.step(forced or None)
        events.extend(e for e in step if e["type"] in KEEP)
        if world.tick > window:
            window_ticks += population
            strikes += world.counters.get("predator_strikes", 0) - before_strikes
            blooms += world.counters.get("bloom_meals", 0) - before_blooms
        agent_ticks += population
    return events, {"seed": seed, "channel": channel, "oracle": oracle, "ticks_run": world.tick,
                    "stop_reason": world.stop_reason or "step_budget", "population": len(world.agents),
                    "births": world.counters.get("births", 0),
                    "predator_strikes_per_1000_agent_ticks": round(1000 * strikes / max(1, window_ticks), 4),
                    "bloom_meals_per_1000_agent_ticks": round(1000 * blooms / max(1, window_ticks), 4),
                    "deaths_by_predation": sum(1 for e in events if e["type"] == "death" and e["reason"] == "predation")}


def seed_probe(args):
    seed, ticks, overrides = args
    normal_events, normal = run(seed, ticks, "normal", overrides=overrides)
    scrambled_events, scrambled = run(seed, ticks, "scrambled", overrides=overrides)
    contrast = discovery.receiver_contrast(normal_events, scrambled_events, reps=200, seed=seed, start=ticks // 2)
    return {"seed": seed, "normal": normal, "scrambled": scrambled,
            "receiver_normal_excess": contrast["normal"]["pooled"]["excess_bits"],
            "receiver_scrambled_excess": contrast["scrambled"]["pooled"]["excess_bits"],
            "receiver_over_scrambled": contrast["excess_over_scrambled_bits"],
            "receiver_p": contrast["normal"]["pooled"]["p_value"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", default="1-6")
    parser.add_argument("--ticks", type=int, default=1200)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--config", default="{}")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    low, _, high = args.seeds.partition("-")
    seeds = list(range(int(low), int(high or low) + 1))
    tasks = [(seed, args.ticks, json.loads(args.config)) for seed in seeds]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        rows = list(pool.map(seed_probe, tasks))
    keys = ("predator_strikes_per_1000_agent_ticks", "bloom_meals_per_1000_agent_ticks", "population")
    summary = {"receiver_over_scrambled": [r["receiver_over_scrambled"] for r in rows],
               "receiver_p": [r["receiver_p"] for r in rows],
               **{f"{k}_normal_minus_scrambled": [round(r["normal"][k] - r["scrambled"][k], 4) for r in rows] for k in keys}}
    result = {"note": "diagnostic only: oracle senders use a supplied mapping", "preset": LIVING, "ticks": args.ticks,
              "per_seed": rows, "summary": summary}
    if args.out:
        with open(args.out, "w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
