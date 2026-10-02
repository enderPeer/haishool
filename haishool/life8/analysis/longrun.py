"""Run one search world much longer and measure, again and again, whether its calls become language.

The search (``haishool.life8.search``) stops every world at tick 7,500. This driver takes one
kept world's bundle (``out/checkpoints/<id>.r3.json.gz``) and continues it in segments of
``--every`` ticks up to ``--horizon`` (or extinction / a capacity pause). After each segment it
measures exactly what the search measures, with the search's own functions:

* the segment's own measures (``search.run_segment``: sender information within senders against
  a permutation null, receiver information, conventions, reciprocity) over its last half;
* paired branches from the segment's end state (``search.run_branch``) in three arms -- normal,
  masked (symbols hidden: content cannot matter) and scrambled (random symbols) -- with
  ``--reps`` replicates of ``--branch-ticks`` ticks, every replicate reseeded identically in
  every arm, summarised by ``search.paired_measures``;
* ``search.criteria`` and ``search.score`` on the combined row (the screen). ``language_verdict``
  is ``all_three`` only when the sender, receiver and fitness-cost-of-scrambling tests all pass.
  ``tool_benefit`` is None (no tools_off arm), so long-run scores are not comparable to search scores.

Calibrated tests (``row["calibrated"]``, recorded next to the screen, not replacing it). The
screen's sender and receiver flags rest on fixed 0.02-bit differences without a significance test
(review 2026-10-02). The calibrated block measures, per replicate, the within-sender excess in the
normal minus the masked arm and the receiver excess in the normal minus the scrambled arm, and
sign-tests those 24 differences; fitness needs at least 20 non-tied replicates. Each passes at
sign_p <= 0.01 with a positive mean (and the screen's pooled threshold).

Pre-declared rule: language EMERGED in a future when the calibrated verdict is all_three at two
adjacent measurement points AND the baseline (the world at tick 7,500 measured with the same
protocol, ``baseline`` command) is not all_three. A firing is a screen result; it is claimed only
after the confirmation protocol below.

Confirmation protocol (pre-registered, runs only if the rule fires; m = number of futures that
fired): re-measure both end states of the streak with fresh branch seeds (replicate offset
100000), 48 replicates x 300 ticks, arms normal, masked, scrambled and -- once the engine has it --
a frequency-keeping shuffle arm (scrambling with uniform symbols also changes symbol frequencies,
which alone can change behaviour). Tests one-sided at alpha 0.01 / m: fitness sign test (normal
minus scrambled, and minus shuffle), per-replicate receiver and sender sign tests with the pooled
0.02-bit thresholds. Language is claimed only if all three pass at both points.

Futures: ``--future 0`` continues the world with its own random state (the exact continuation).
``--future k`` (k >= 1) reseeds the world's random generators and its seed (the seed keys every
newborn's genome mutation stream, genome.rng_for) at the start, so the same world (same
individuals, genomes, learned weights, food, predators) lives through independent chance events
from then on. Nothing else changes: no meaning, symbol or behaviour is supplied.

Branch measurements run in a process pool while the world simulates its next segment. Every
segment's end state is kept (``seg-NNN.json.gz``) so any point can be re-measured or replayed.
The run is resumable (restart the same command) and refuses a second driver in the same folder.

    python -m haishool.life8.analysis.longrun run --bundle B --out DIR --future K \\
        [--horizon 102500] [--every 5000] [--reps 24] [--branch-ticks 300] [--workers 4]
    python -m haishool.life8.analysis.longrun baseline --bundle B --row R3.json --out FILE [--workers 8]
    python -m haishool.life8.analysis.longrun report [--baseline FILE] DIR [DIR ...]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .. import search
from ..config import Config

ARMS = ("normal", "masked", "scrambled")
FUTURE_SALT = 0x10C6_0781
SCHEMA = "life8-longrun-v1"
DRIVER = 2  # 2: seed reseeded for futures, calibrated block, atomic row-before-state, adjacency check
CAL_ALPHA = .01
CAL_MIN_NONZERO = 20
CAL_PERM_REPS = 20


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def entry_of(bundle):
    identity = bundle["id"]
    source, _, number = identity.partition("-")
    return {"id": identity, "source": source, "seed": int(number) if number.isdigit() else 0}


def reseed(bundle, future):
    """The same world state with fresh random generators and seed (future >= 1); future 0 is unchanged."""
    if future == 0:
        return bundle
    world, tracker = search._restore(bundle)
    base = FUTURE_SALT * 1009 + future
    world.rng = random.Random(base)
    world.channel_rng = random.Random(base * 7 + 1)
    world.eco_rng = random.Random(base * 13 + 2)
    world.seed = (base * 31 + 3) % 2 ** 64  # keys genome.rng_for: newborns' mutations differ per future
    return search._bundle(entry_of(bundle), world, tracker, bundle.get("round"))


def _segment_path(out, index):
    return Path(out) / f"seg-{index:03d}.json.gz"


def _row_path(out, index):
    return Path(out) / f"seg-{index:03d}.row.json"


def _write_json(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _read_rows(out):
    path = Path(out) / "rows.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.open(encoding="utf-8"):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:  # a torn last line from a kill; the segment is re-measured
                break
    seen, unique = set(), []
    for row in rows:
        if row["segment"] not in seen:
            seen.add(row["segment"])
            unique.append(row)
    return unique


def _append(out, row):
    with (Path(out) / "rows.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _status(out, **data):
    _write_json(Path(out) / "status.json", {**data, "utc": utc()})


def _lock(out):
    """One driver per folder (Linux hosts); a no-op where fcntl is missing."""
    try:
        import fcntl
    except ImportError:
        return None
    handle = open(Path(out) / ".lock", "w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit(f"another driver is running in {out}")
    return handle


def _strip(branch):
    """A branch result without its bulky event lists."""
    return {k: v for k, v in branch.items() if k not in ("heard", "signals")}


def calibrated(branches, *, state, seed, screen):
    """Per-replicate paired differences and sign tests (see the module docstring)."""
    by = {(b["arm"], b["rep"]): b for b in branches}
    reps = sorted({b["rep"] for b in branches})

    def within(arm, rep, offset):
        info = search.sender_information(by[arm, rep].get("signals", []), reps=CAL_PERM_REPS,
                                         seed=seed + offset + rep, state=state)
        return (info.get("within_sender") or {}).get("excess_bits")

    def receiver(arm, rep, offset):
        return search.receiver_information(by[arm, rep].get("heard", []), reps=CAL_PERM_REPS,
                                           seed=seed + offset + rep).get("excess_bits")

    sender_d, receiver_d, fitness_d = [], [], []
    for rep in reps:
        if ("normal", rep) in by and ("masked", rep) in by:
            a, b = within("normal", rep, 1000), within("masked", rep, 1000)
            if a is not None and b is not None:
                sender_d.append(a - b)
        if ("normal", rep) in by and ("scrambled", rep) in by:
            a, b = receiver("normal", rep, 2000), receiver("scrambled", rep, 2000)
            if a is not None and b is not None:
                receiver_d.append(a - b)
            fitness_d.append(by["normal", rep]["food_energy_per_agent_tick"]
                             - by["scrambled", rep]["food_energy_per_agent_tick"])
    out = {"sender_normal_minus_masked": search.paired(sender_d),
           "receiver_normal_minus_scrambled": search.paired(receiver_d),
           "fitness_normal_minus_scrambled": search.paired(fitness_d)}

    def passes(p, extra=True):
        return bool(extra and p["mean"] is not None and p["mean"] > 0 and p["sign_p"] <= CAL_ALPHA)

    fit = out["fitness_normal_minus_scrambled"]
    flags = {"sender": passes(out["sender_normal_minus_masked"], screen.get("sender_information") is True),
             "receiver": passes(out["receiver_normal_minus_scrambled"], screen.get("receiver_information") is True),
             "fitness": passes(fit, fit["positive"] + fit["negative"] >= CAL_MIN_NONZERO)}
    out["flags"] = flags
    out["verdict"] = "all_three" if all(flags.values()) else "partial" if any(flags.values()) else "null"
    return out


def finish_row(row, branches, *, perm_reps, seed, state, reps, branch_ticks):
    good = [b for b in branches if "error" not in b]
    row["paired"] = search.paired_measures(good, perm_reps=perm_reps, seed=seed, state=state) if good else None
    row["paired_errors"] = [b["error"] for b in branches if "error" in b]
    row["branch_reps"], row["branch_ticks"], row["branch_arms"] = reps, branch_ticks, list(ARMS)
    row["criteria"] = search.criteria(row)
    row["criteria"]["tool_benefit"] = None  # not measured: no tools_off arm
    row["score"], row["components"] = search.score(row)
    row["calibrated"] = calibrated(good, state=state, seed=seed, screen=row["criteria"]) if good else None
    row["branches"] = [_strip(b) for b in good]
    row["driver"] = DRIVER
    return row


def finish_stopped(row):
    """A stopped state (extinction, capacity pause) cannot be branched: the row stays unpaired."""
    row["paired"], row["calibrated"], row["driver"] = None, None, DRIVER
    row["criteria"] = search.criteria(row)
    row["score"], row["components"] = search.score(row)
    return row


def run(bundle_path, out, *, future=0, horizon=102500, every=5000, reps=24, branch_ticks=300, workers=4,
        perm_reps=100, log=print):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    lock = _lock(out)
    meta_path = out / "meta.json"
    args = {"bundle": str(bundle_path), "future": future, "horizon": horizon, "every": every, "reps": reps,
            "branch_ticks": branch_ticks, "perm_reps": perm_reps, "arms": list(ARMS)}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["args"] != args:
            raise SystemExit(f"{out} was started with other arguments: {meta['args']}")
    else:
        meta = {"schema": SCHEMA, "args": args, "started": utc(), "driver": DRIVER,
                "host": os.uname().nodename if hasattr(os, "uname") else ""}
        _write_json(meta_path, meta)

    rows = _read_rows(out)
    measured = {r["segment"] for r in rows}
    # a state counts only with its row file (rows are written before states since driver 2;
    # a state without one is re-simulated from the previous state, which is deterministic)
    segments = sorted(i for i in (int(p.name[4:7]) for p in out.glob("seg-*.json.gz"))
                      if i == 0 or _row_path(out, i).exists())
    while segments and segments[-1] > 0 and segments[-1] - 1 not in segments:
        segments.pop()
    if segments:
        last = segments[-1]
        bundle = search._read_gz(_segment_path(out, last))
    else:
        last = 0
        bundle = reseed(search._read_gz(Path(bundle_path)), future)
        search._write_gz(_segment_path(out, 0), bundle)
    entry = entry_of(bundle)
    state = search.sender_state(Config.from_dict(bundle["world"]["config"]))
    stop = bundle["world"].get("stop_reason")

    pool = ProcessPoolExecutor(max_workers=max(1, workers))
    pending = {}  # segment index -> (row, [futures])

    def branch_seed(index):
        return entry["seed"] + 104729 * (index + 10)

    def submit(index, row):
        path = str(_segment_path(out, index))
        jobs = [pool.submit(search._branch_job, (path, arm, rep, branch_ticks)) for rep in range(reps) for arm in ARMS]
        pending[index] = (row, jobs)

    def harvest(block=False):
        for index in sorted(pending):
            row, jobs = pending[index]
            if not block and not all(j.done() for j in jobs):
                continue
            branches = [j.result() for j in jobs]
            finish_row(row, branches, perm_reps=perm_reps, seed=branch_seed(index), state=state, reps=reps,
                       branch_ticks=branch_ticks)
            _append(out, row)
            c, cal = row["criteria"], row["calibrated"] or {}
            log(json.dumps({"segment": index, "tick": row["end_tick"], "population": row["population"],
                            "verdict": c.get("language_verdict"), "calibrated": cal.get("verdict"),
                            "passed": [k for k, v in c.items() if v is True], "utc": utc()}))
            del pending[index]

    for index in segments:  # states whose measurement was lost in a restart
        if index >= 1 and index not in measured:
            row = json.loads(_row_path(out, index).read_text(encoding="utf-8"))
            if row.get("stop_reason"):
                _append(out, finish_stopped(row))
            else:
                submit(index, row)

    index = last
    try:
        while not stop and bundle["tick"] < horizon:
            ticks = min(every, horizon - bundle["tick"])
            row, bundle = search.run_segment(entry, ticks, start=bundle, perm_reps=perm_reps,
                                             seed=entry["seed"] + 7919 * (index + 10))
            index += 1
            row.update({"kind": "longrun", "future": future, "segment": index})
            _write_json(_row_path(out, index), row)
            search._write_gz(_segment_path(out, index), bundle)
            stop = row["stop_reason"]
            _status(out, status="running", tick=bundle["tick"], segment=index, population=row["population"],
                    stop_reason=stop, measuring=sorted(pending) + ([] if stop else [index]))
            if stop:
                _append(out, finish_stopped(row))
                break
            submit(index, row)
            harvest()
        harvest(block=True)
    finally:
        pool.shutdown(wait=True, cancel_futures=False)
    _status(out, status="done", tick=bundle["tick"], segment=index, population=len(bundle["world"]["agents"]),
            stop_reason=stop)
    if lock:
        lock.close()
    return _read_rows(out)


def baseline(bundle_path, search_row, out, *, reps=24, branch_ticks=300, workers=8, perm_reps=100):
    """The world at the fork (tick 7,500) measured with the long-run protocol. The segment measures
    come from the search's own row for that world (its last round's window); the branches are new."""
    bundle = search._read_gz(Path(bundle_path))
    state = search.sender_state(Config.from_dict(bundle["world"]["config"]))
    entry = entry_of(bundle)
    row = {k: v for k, v in search_row.items() if k not in ("paired", "criteria", "score", "components",
                                                              "paired_errors", "host")}
    row.update({"kind": "longrun_baseline", "future": None, "segment": 0, "search_round": search_row.get("round")})
    tasks = [(str(bundle_path), arm, rep, branch_ticks) for rep in range(reps) for arm in ARMS]
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        branches = list(pool.map(search._branch_job, tasks))
    finish_row(row, branches, perm_reps=perm_reps, seed=entry["seed"] + 104729 * 10, state=state, reps=reps,
               branch_ticks=branch_ticks)
    _write_json(out, row)
    return row


def _ci(p):
    if p.get("mean") is None or p.get("sd") is None or not p.get("n"):
        return None
    half = 1.96 * p["sd"] / math.sqrt(p["n"])
    return [round(p["mean"] - half, 6), round(p["mean"] + half, 6)]


def summarise(rows, base=None):
    """One line per measured segment, and the pre-declared emergence check."""
    lines, emerged = [], []
    base_verdict = ((base or {}).get("calibrated") or {}).get("verdict")
    by_future = {}
    for r in sorted(rows, key=lambda r: (r["future"], r["segment"])):
        by_future.setdefault(r["future"], []).append(r)
    for future, rs in by_future.items():
        streak, previous = 0, None
        for r in rs:
            c, p, m, cal = r.get("criteria") or {}, r.get("paired") or {}, r["measures"], r.get("calibrated") or {}
            fit = (p.get("normal_minus_scrambled") or {}).get("food_energy_per_agent_tick") or {}
            verdict = cal.get("verdict")
            streak = (streak + 1 if previous == r["segment"] - 1 else 1) if verdict == "all_three" else 0
            previous = r["segment"]
            if streak == 2 and base_verdict != "all_three":
                emerged.append({"future": future, "tick": r["end_tick"]})
            stop = r.get("stop_reason")
            lines.append({
                "future": future, "segment": r["segment"], "tick": r["end_tick"], "population": r["population"],
                "lineages": r.get("lineages_alive"), "screen": c.get("language_verdict"), "calibrated": verdict,
                "passed": "".join(ch for name, ch in (("sender_information", "S"), ("receiver_information", "R"),
                                                       ("fitness_cost_of_scrambling", "F"), ("convention", "C"),
                                                       ("cooperation", "Q")) if c.get(name) is True) or "-",
                "sender_within_bits": (m["sender"].get("within_sender") or {}).get("excess_bits"),
                "sender_over_masked_bits": (p.get("sender") or {}).get("excess_over_control_bits"),
                "receiver_over_scrambled_bits": (p.get("receiver") or {}).get("excess_over_scrambled_bits"),
                "fitness_mean": fit.get("mean"), "fitness_ci95": _ci(fit) if fit else None,
                "fitness_nonzero": (fit.get("positive", 0) + fit.get("negative", 0)) if fit else None,
                "fitness_pos_neg": f"{fit.get('positive')}/{fit.get('negative')}" if fit else None,
                "stop": None if not stop else "extinction" if stop == "extinction" else f"censored ({stop})",
                "driver": r.get("driver", 1)})
    return {"baseline": None if base is None else {"screen": (base.get("criteria") or {}).get("language_verdict"),
                                                   "calibrated": base_verdict},
            "points": lines, "emerged": emerged,
            "rule": "language emerged = calibrated all_three at two adjacent measurement points of one future, "
                    "with a baseline that is not all_three; claimed only after the confirmation protocol",
            "detectable": "the fitness test detects about 0.010 food energy per agent-tick (80% power per point)"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--bundle", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--future", type=int, default=0)
    r.add_argument("--horizon", type=int, default=102500)
    r.add_argument("--every", type=int, default=5000)
    r.add_argument("--reps", type=int, default=24)
    r.add_argument("--branch-ticks", type=int, default=300)
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--perm-reps", type=int, default=100)
    b = sub.add_parser("baseline")
    b.add_argument("--bundle", required=True)
    b.add_argument("--row", required=True, help="the search's results row (JSON) for this world's last round")
    b.add_argument("--out", required=True)
    b.add_argument("--workers", type=int, default=8)
    p = sub.add_parser("report")
    p.add_argument("--baseline")
    p.add_argument("dirs", nargs="+")
    args = ap.parse_args(argv)
    if args.cmd == "run":
        log = lambda text: print(text, flush=True)  # noqa: E731
        run(args.bundle, args.out, future=args.future, horizon=args.horizon, every=args.every, reps=args.reps,
            branch_ticks=args.branch_ticks, workers=args.workers, perm_reps=args.perm_reps, log=log)
    elif args.cmd == "baseline":
        row = json.loads(Path(args.row).read_text(encoding="utf-8"))
        result = baseline(args.bundle, row, args.out, workers=args.workers)
        print(json.dumps({"screen": result["criteria"].get("language_verdict"),
                          "calibrated": result["calibrated"]["verdict"], "flags": result["calibrated"]["flags"]}))
    else:
        rows = [row for d in args.dirs for row in _read_rows(d)]
        base = json.loads(Path(args.baseline).read_text(encoding="utf-8")) if args.baseline else None
        json.dump(summarise(rows, base), sys.stdout, indent=1)
        print()


if __name__ == "__main__":
    main()
