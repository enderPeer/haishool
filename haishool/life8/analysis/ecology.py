"""Living-world ecology: does meaning emerge when signalling can pay?

For each seed (same seed in every arm) this runs, with the compact log:
  normal     the LIVING preset (config.LIVING)
  scrambled  the same with Config.channel="scrambled" (content destroyed, costs kept)
  no_kin     the same with kin_credit=0 (the sender credit route switched off)
and measures, over the window [window_start, ticks):
  * sender information I(symbol; state) against a permutation null, for discovery's
    default joint state and for the ecology state (danger_seen, bloom_seen);
  * receiver information I(heard symbol; next action), normal minus scrambled;
  * fitness under scrambling: paired branches from the normal run (discovery.branch_fitness);
  * lineage conventions (lineage.convention_agreement) on both situations;
  * diagnostics of why: call rates by situation, information asymmetry (the sender sees
    danger or a bloom, the hearer does not), the share of deliveries to kin, and the
    receivers' moves relative to the call's bearing.

Toy-world measurement only; no claim of language. Usage:
  python -m haishool.life8.analysis.ecology --seeds 1-12 --ticks 2000 --jobs 12 --out FILE
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

from .. import discovery, lineage
from ..config import LIVING, Config
from ..world import World

ECOLOGY_STATE = ("danger_seen", "bloom_seen")
KEEP = ("signal", "heard", "birth", "death", "capacity_pause", "predation")


def _run(seed, overrides, ticks, *, channel="normal", keep_state_at=None, with_table=False):
    world = World.create(seed=seed, config=Config(**{**overrides, "log": "compact", "channel": channel}))
    table = lineage.LineageTable.from_world(world) if with_table else None
    kept, state, deaths = [], None, Counter()
    while world.tick < ticks and not world.stop_reason:
        if keep_state_at is not None and world.tick == keep_state_at:
            state = world.to_dict()
        events = world.step()
        if table is not None:
            table.record(world, events)
        for event in events:
            if event["type"] == "death":
                deaths[event["reason"]] += 1
        kept.extend(e for e in events if e["type"] in KEEP)
    summary = {"ticks_run": world.tick, "stop_reason": world.stop_reason or "step_budget",
               "population": len(world.agents), "deaths": dict(deaths),
               **{k: round(world.counters.get(k, 0), 3) for k in ("births", "signals_sent", "signals_received",
                                                                   "predator_strikes", "bloom_meals", "meals")}}
    return kept, world, state, table, summary


def _rate(rows):
    return round(sum(rows) / len(rows), 4) if rows else None


def diagnostics(events, *, start=0):
    """Why a signal could or could not mean something here (no permutation tests)."""
    signals = {e["message_id"]: e for e in events if e["type"] == "signal" and e["tick"] >= start}
    heard = [e for e in events if e["type"] == "heard" and e["tick"] >= start and e.get("receiver")]
    by_state = Counter((e["sender"].get("danger_seen"), e["sender"].get("bloom_seen")) for e in signals.values())
    asym = {"danger": [], "bloom": []}
    kin = Counter()
    for e in heard:
        kin[e["receiver"].get("heard_kin_band")] += 1
        sent = signals.get(e["message_id"])
        if sent is None:
            continue
        if sent["sender"].get("danger_seen"):
            asym["danger"].append(not e["receiver"].get("danger_seen"))
        if sent["sender"].get("bloom_seen"):
            asym["bloom"].append(not e["receiver"].get("bloom_seen"))
    # symbol -> sender state table (does any symbol mark danger or blooms?)
    table = {}
    for e in signals.values():
        key = f"danger={e['sender'].get('danger_seen')},bloom={e['sender'].get('bloom_seen')}"
        table.setdefault(key, Counter())[e["symbol"]] += 1
    return {"signals": len(signals), "deliveries": len(heard),
            "signals_by_sender_state": {f"danger={k[0]},bloom={k[1]}": v for k, v in sorted(by_state.items(), key=repr)},
            "symbol_counts_by_sender_state": {k: dict(sorted(v.items())) for k, v in sorted(table.items())},
            "hearer_lacks_what_sender_sees": {"danger": _rate(asym["danger"]), "danger_n": len(asym["danger"]),
                                              "bloom": _rate(asym["bloom"]), "bloom_n": len(asym["bloom"])},
            "deliveries_by_kin_band": dict(sorted(kin.items(), key=repr))}


def seed_report(seed, ticks=2000, *, window_start=None, branch_at=None, branch_ticks=200, reps=8, perm_reps=200,
                overrides=None):
    started = time.perf_counter()
    overrides = {**LIVING, **(overrides or {})}
    window_start = ticks // 2 if window_start is None else window_start
    branch_at = ticks * 3 // 4 if branch_at is None else branch_at
    normal, world, state, table, n_summary = _run(seed, overrides, ticks, keep_state_at=branch_at, with_table=True)
    scrambled, _, _, _, s_summary = _run(seed, overrides, ticks, channel="scrambled")
    no_kin, _, _, _, k_summary = _run(seed, {**overrides, "kin_credit": 0.}, ticks)
    out = {"seed": seed, "ticks": ticks, "window_start": window_start, "branch_at": branch_at,
           "runs": {"normal": n_summary, "scrambled": s_summary, "no_kin": k_summary}}
    for name, events in (("normal", normal), ("scrambled", scrambled), ("no_kin", no_kin)):
        out.setdefault("sender", {})[name] = {
            "default_joint": discovery.sender_information(events, reps=perm_reps, seed=seed, start=window_start)["joint"],
            "ecology_joint": discovery.sender_information(events, state=ECOLOGY_STATE, bands=ECOLOGY_STATE,
                                                          reps=perm_reps, seed=seed, start=window_start)["joint"]}
    out["receiver"] = discovery.receiver_contrast(normal, scrambled, reps=perm_reps, seed=seed, start=window_start)
    out["receiver_no_kin_minus_scrambled"] = discovery.receiver_contrast(no_kin, scrambled, reps=perm_reps, seed=seed,
                                                                         start=window_start)["excess_over_scrambled_bits"]
    out["fitness"] = discovery.branch_fitness(state, ticks=branch_ticks, reps=reps) if state and reps else None
    # Sender criteria use the within-sender test and the scrambled run as control (critic B2).
    out["verdict"] = discovery.verdict(
        discovery.sender_information(normal, reps=perm_reps, seed=seed, start=window_start), out["receiver"],
        out["fitness"], sender_control=discovery.sender_information(scrambled, reps=perm_reps, seed=seed,
                                                                    start=window_start))
    eco_normal, eco_scrambled = (discovery.sender_information(events, state=ECOLOGY_STATE, bands=(), reps=perm_reps,
                                                              seed=seed, start=window_start)["within_sender"]
                                 for events in (normal, scrambled))
    t = discovery.THRESHOLDS
    out["ecology_sender_within"] = {"normal": eco_normal, "scrambled": eco_scrambled}
    out["ecology_sender_pass"] = (eco_normal["excess_bits"] or 0) >= t["sender_excess_bits"] \
        and (eco_normal["p_value"] or 1) <= t["sender_p"] \
        and (eco_normal["excess_bits"] or 0) - (eco_scrambled["excess_bits"] or 0) >= t["sender_over_control_bits"]
    out["lineage"] = {
        "default_situation": lineage.convention_agreement(normal, table, start=window_start, reps=perm_reps, seed=seed),
        "ecology_situation": lineage.convention_agreement(normal, table, situation=ECOLOGY_STATE, start=window_start,
                                                          reps=perm_reps, seed=seed)}
    out["diagnostics"] = {"normal": diagnostics(normal, start=window_start),
                          "no_kin": diagnostics(no_kin, start=window_start)}
    out["seconds"] = round(time.perf_counter() - started, 1)
    return out


def _job(args):
    seed, kwargs = args
    return seed_report(seed, **kwargs)


def measure(seeds, ticks=2000, *, jobs=12, **kwargs):
    tasks = [(seed, {"ticks": ticks, **kwargs}) for seed in seeds]
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            rows = list(pool.map(_job, tasks))
    else:
        rows = [_job(task) for task in tasks]
    return {"preset": LIVING, "ticks": ticks, "seeds": list(seeds), "per_seed": rows, "summary": summarize(rows)}


def summarize(rows):
    def col(path):
        values = []
        for row in rows:
            value = row
            for key in path:
                value = value.get(key) if isinstance(value, dict) else None
            values.append(value)
        return values
    fit = [row["fitness"]["effects"]["normal_minus_scrambled"] if row["fitness"] else None for row in rows]
    return {
        "verdicts": dict(Counter(row["verdict"]["verdict"] for row in rows)),
        "sender_default_excess": col(("sender", "normal", "default_joint", "excess_bits")),
        "sender_default_p": col(("sender", "normal", "default_joint", "p_value")),
        "sender_ecology_excess": col(("sender", "normal", "ecology_joint", "excess_bits")),
        "sender_ecology_p": col(("sender", "normal", "ecology_joint", "p_value")),
        "sender_ecology_excess_scrambled_run": col(("sender", "scrambled", "ecology_joint", "excess_bits")),
        "sender_ecology_excess_no_kin": col(("sender", "no_kin", "ecology_joint", "excess_bits")),
        "ecology_sender_pass": col(("ecology_sender_pass",)),
        "receiver_over_scrambled": col(("receiver", "excess_over_scrambled_bits")),
        "receiver_no_kin_over_scrambled": col(("receiver_no_kin_minus_scrambled",)),
        "fitness_food_energy_normal_minus_scrambled": [f["food_energy_per_agent_tick"]["mean"] if f else None for f in fit],
        "fitness_survival_normal_minus_scrambled": [f["survival"]["mean"] if f else None for f in fit],
        "fitness_population_normal_minus_scrambled": [f["population"]["mean"] if f else None for f in fit],
        "lineage_within": col(("lineage", "ecology_situation", "within")),
        "lineage_across": col(("lineage", "ecology_situation", "across")),
        "lineage_p": col(("lineage", "ecology_situation", "p_value")),
        "population_final": col(("runs", "normal", "population")),
        "stop_reasons": col(("runs", "normal", "stop_reason")),
        "hearer_lacks_danger": col(("diagnostics", "normal", "hearer_lacks_what_sender_sees", "danger")),
        "hearer_lacks_bloom": col(("diagnostics", "normal", "hearer_lacks_what_sender_sees", "bloom")),
        "seconds": col(("seconds",)),
    }


def _seeds(text):
    if "-" in text:
        low, high = text.split("-")
        return list(range(int(low), int(high) + 1))
    return [int(part) for part in text.split(",")]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", default="1-12")
    parser.add_argument("--ticks", type=int, default=2000)
    parser.add_argument("--reps", type=int, default=8)
    parser.add_argument("--perm-reps", type=int, default=200)
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--config", default="{}", help="JSON overrides on top of LIVING")
    parser.add_argument("--out")
    args = parser.parse_args(argv)
    result = measure(_seeds(args.seeds), args.ticks, jobs=args.jobs, reps=args.reps, perm_reps=args.perm_reps,
                     overrides=json.loads(args.config))
    text = json.dumps(result, indent=1, default=str)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as stream:
            stream.write(text)
    print(json.dumps(result["summary"], indent=1, default=str))


if __name__ == "__main__":
    main()
