"""The listener gate: can hearers profit from calls at all in this world?

life8's oracle probe (docs/life8/README.md) showed that even perfectly informative calls
did not help hearers, because a predator struck the nearest individual (usually the caller)
and a bloom was gone before a hearer arrived. Until hearers can profit, no sender route can
select informative calls. life9 must pass this gate before any emergence run is read.

Arms on the same seed (same terrains and founders, world by world):

* ``oracle``: every voiced individual that sees a predator calls one fixed vector (a
  supplied meaning, diagnostic only), and hearers evolve and learn as usual;
* ``oracle_deaf``: the same calls are made and paid, but nobody hears them.

Per world: predation deaths per 1000 individual-ticks, and, among individuals a call
reached while they could not see the predator, P(their move took them away from it). The
call reaches the same kind of individual in both arms; only the oracle arm hears it, so
the deaf arm is the matched control (hearers vs non-hearers within one arm is confounded:
callers stand near predators, so hearers do too). The gate passes when, in a significant
majority of worlds (two-sided sign test p <= 0.05), predation is lower with hearing and
reached individuals move away more often with hearing. The worlds are paired replicates.
"""
from __future__ import annotations

import json
import math
import time

from .config import Config9
from .world import World9


def sign_test(wins, losses):
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def rates(world):
    s = world.stats
    ticks = s["alive_ticks"].clamp(min=1)
    deaths = (s["deaths_predation"] + s["deaths_starvation"] + s["deaths_age"]).clamp(min=1)
    return {"predation_per_1000": (1000 * s["deaths_predation"] / ticks).tolist(),
            "predation_share_of_deaths": (s["deaths_predation"] / deaths).tolist(),
            "mean_hidden": world.summary()["mean_hidden"], "mean_voice": world.summary()["mean_voice"],
            "alive": world.alive.sum(1).tolist(),
            "hearer_flee": (s["heard_flee"] / s["heard_moves"].clamp(min=1)).tolist(),
            "quiet_flee": (s["quiet_flee"] / s["quiet_moves"].clamp(min=1)).tolist(),
            "hearer_moves": s["heard_moves"].tolist()}


def run(cfg: Config9, seed, ticks, device="cpu", log=print):
    arms = {"oracle": cfg.but(oracle=True, channel="normal"), "oracle_deaf": cfg.but(oracle=True, channel="deaf")}
    results = {}
    for name, arm in arms.items():
        world, start = World9(arm, seed, device), time.time()
        for t in range(ticks):
            world.step()
        results[name] = rates(world)
        log(f"{name}: {ticks} ticks x {arm.worlds} worlds in {time.time() - start:.1f}s")
    oracle, deaf = results["oracle"]["predation_per_1000"], results["oracle_deaf"]["predation_per_1000"]
    wins = sum(o < d for o, d in zip(oracle, deaf))
    losses = sum(o > d for o, d in zip(oracle, deaf))
    heard, unheard = results["oracle"], results["oracle_deaf"]
    pairs = [(h, d) for h, d, m, n in zip(heard["hearer_flee"], unheard["hearer_flee"], heard["hearer_moves"],
                                          unheard["hearer_moves"]) if m and n]
    flee_wins = sum(h > d for h, d in pairs)
    flee_losses = sum(h < d for h, d in pairs)
    p = sign_test(wins, losses)
    p_flee = sign_test(flee_wins, flee_losses)
    verdict = {"predation_lower_with_hearing": [wins, losses], "sign_p": p,
               "reached_flee_more_with_hearing": [flee_wins, flee_losses], "flee_sign_p": p_flee,
               "gate": bool(wins > losses and p <= .05 and flee_wins > flee_losses and p_flee <= .05)}
    return {"schema": "life9-listener-gate-v1", "seed": seed, "ticks": ticks, "config": cfg.to_dict(),
            "arms": results, "verdict": verdict}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ticks", type=int, default=2000)
    parser.add_argument("--worlds", type=int, default=32)
    parser.add_argument("--capacity", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out")
    args = parser.parse_args()
    report = run(Config9(worlds=args.worlds, capacity=args.capacity), args.seed, args.ticks, args.device)
    print(json.dumps(report["verdict"], indent=1))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=1)
