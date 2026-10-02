"""Visitor arms: paired twins of life8 planets with and without visitors (round 8).

A visitor world is a separate arm, never part of the search's emergence results: every arm runs
on a copy of a planet from the r8a plan (same seed, same config, same founders) and is compared
with the untouched twin of the same planet, run here as ``baseline``. Outputs go to their own
folder (default ``runs/visitors/<name>``) with schema ``life8-visitor-arms-v1``; nothing is
written next to the search.

Arms (``visitor.Visitors`` drives them; with a policy that copies the native choice every arm
equals the baseline bit for bit, so differences come from the visitors' actions alone):

| arm | visitors | from tick | policy |
|---|---|---|---|
| ``baseline`` | none (the untouched twin) | - | - |
| ``one`` | the first founder of the planet's founder order | 0 | ``--policy`` (traveller) |
| ``tenth`` | the first max(1, round(n/10)) founders (24 founders: 2) | 0 | ``--policy`` |
| ``all`` | every founder | 0 | ``--policy`` |
| ``elder`` | one living individual at tick ``--elder-at`` (1000) | 1000 | ``--elder-policy`` |

The founder order is a shuffle of the founder ids by ``Random("life8-visitors:<id>:<seed>")``, so
the arms are nested (one in tenth in all) and the choice depends on the planet alone. The elder
individual is drawn uniformly from the living at ``--elder-at`` by
``Random("life8-visitor-elder:<id>:<seed>:<tick>")``; the elder arm's world up to that tick is the
baseline's world (the same run).

Measures per run (window: from the visitors' placement tick to the end), each paired with the
baseline twin over the same window:

* visitor survival (ticks from placement to death, censored at the end) and offspring, against the
  native that had the same id (the same founder slot) in the baseline twin; also how each died;
* population (mean over the window, and at the end) and food energy per agent-tick;
* demonstrations with the visitor as demonstrator: its successful ``teach`` actions plus other
  individuals' ``imitate`` actions that copied it (compact ``act`` events), against the native slot;
* call adoption: the visitors' situation -> symbol mapping (modes of their calls per situation,
  ``lineage._modes``, with I(symbol; situation) against a permutation null, ``lineage._info``) and how
  often natives' calls in the same situations use the visitors' symbol, minus what their
  situation-blind symbol habits give (as ``lineage.convention_agreement``, critic F2); in the arm
  and in the baseline twin over the same window. ``moved`` = arm minus baseline while a visitor
  lives (contact window), ``persists`` = arm minus baseline after the last visitor died. Natives who
  heard a visitor are also measured alone (after their first hearing). Situation: food_seen,
  energy_band (+ danger_seen, bloom_seen with the ecology on).

``report`` pairs every arm with its baseline per planet and gives, over planets, the mean +- SE of
the differences and a two-sided sign test (``discovery.sign_test``).

Commands:

    python -m haishool.life8.visitor_arms run --out runs/visitors/NAME --group traveller --first 8 \\
        --arms baseline,one,tenth,all --policy model/visitor.pt --ticks 2000 --workers 8
    python -m haishool.life8.visitor_arms run --out runs/visitors/NAME --group elder --arms baseline,elder \\
        --elder-policy "model/elder/{id}.pt" --ticks 2000
    python -m haishool.life8.visitor_arms report --out runs/visitors/NAME

``--policy fake:mirror`` (the twin-identity check: every difference must be exactly 0),
``fake:random`` and ``fake:rest`` run without torch. When a checkpoint has an export manifest
(``<checkpoint>.json`` with ``planets``, from ``experience export``), a traveller arm refuses a planet
the model was trained or evaluated on, and an elder arm refuses a planet outside the elder set.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from . import visitor as V

SCHEMA = "life8-visitor-arms-v1"
ARMS = ("baseline", "one", "tenth", "all", "elder")
FOUNDER_ARMS = ("one", "tenth", "all")
DEFAULT_PLAN = Path("runs/life8-search/r8a/stage/plan.json")
SITUATION = ("food_seen", "energy_band")
ECOLOGY_SITUATION = ("danger_seen", "bloom_seen")
NOTE = ("Visitor arm: some individuals were controlled by a model trained on life8 experience. A separate "
        "intervention, paired with untouched twins; never mixed into the r8a search's emergence results. "
        "Toy-world measurements: not language, culture or real probabilities.")


# ---------------------------------------------------------------- choices

def founder_order(entry, world) -> list[int]:
    ids = sorted(a.id for a in world.agents.values() if a.parent_id is None)
    random.Random(f"life8-visitors:{entry['id']}:{entry['seed']}").shuffle(ids)
    return ids


def arm_ids(arm, order) -> list[int]:
    if arm == "one":
        return order[:1]
    if arm == "tenth":
        return order[:max(1, round(len(order) / 10))]
    if arm == "all":
        return list(order)
    raise ValueError(f"not a founder arm: {arm}")


def elder_choice(entry, world) -> int | None:
    living = sorted(world.agents)
    if not living:
        return None
    return random.Random(f"life8-visitor-elder:{entry['id']}:{entry['seed']}:{world.tick}").choice(living)


def situation_of(config) -> tuple:
    return SITUATION + (ECOLOGY_SITUATION if config.ecology_on() else ())


# ---------------------------------------------------------------- tracking one run

class Tracker:
    """Per-tick population and food, every individual's fate, calls, hearings of watched senders
    and demonstrations, from the compact events of one run, starting at the current tick."""

    def __init__(self, world, watch=()):
        self.start = world.tick
        self.watch = set(watch)
        self.population = [len(world.agents)]          # at start, then after every step
        self.food = [world.ledger["energy_from_food"]]  # cumulative, same ticks
        self.agents = {a.id: {"born": a.born_tick, "parent": a.parent_id, "died": None, "reason": None, "births": []}
                       for a in world.agents.values()}
        self.signals = []   # [tick, sender, symbol, bands]
        self.heard = []     # [tick, receiver, sender] for watched senders
        self.demos = []     # [tick, kind, teacher, learner, demonstrated action, success]
        self.situation = situation_of(world.config)

    def watch_more(self, ids):
        self.watch.update(ids)

    def after_step(self, world, events):
        for event in events:
            kind = event.get("type")
            if kind == "birth":
                parent = self.agents.get(event["parent"])
                if parent is not None:
                    parent["births"].append(event["tick"])
                self.agents[event["agent"]] = {"born": event["tick"], "parent": event["parent"], "died": None,
                                               "reason": None, "births": []}
            elif kind == "death":
                row = self.agents.get(event["agent"])
                if row is not None:
                    row["died"], row["reason"] = event["tick"], event.get("reason")
            elif kind == "signal":
                sender = event["sender"]
                state = dict(sender)
                state["food_seen"] = sender.get("food_direction") not in (None, "none")
                self.signals.append([event["tick"], event["agent"], event["symbol"],
                                     [state.get(name) for name in self.situation]])
            elif kind == "heard":
                if event.get("sender") in self.watch:
                    self.heard.append([event["tick"], event["agent"], event["sender"]])
            elif kind == "act" and event.get("action") in ("teach", "imitate") and "teacher" in event:
                self.demos.append([event["tick"], event["action"], event["teacher"], event["learner"],
                                   event.get("demonstrated_action"), bool(event.get("success"))])
        self.population.append(len(world.agents))
        self.food.append(world.ledger["energy_from_food"])

    def to_dict(self):
        return {"start": self.start, "situation": list(self.situation), "population": self.population,
                "food": [round(f, 6) for f in self.food],
                "agents": {str(k): v for k, v in sorted(self.agents.items())},
                "signals": self.signals, "heard": self.heard, "demos": self.demos}


def _advance(world, visitors, tracker, ticks):
    while world.tick < ticks and not world.stop_reason:
        events = visitors.step(world) if visitors is not None else world.step()
        if tracker is not None:
            tracker.after_step(world, events)


# ---------------------------------------------------------------- one run (a pool job)

def _check_manifest(spec, entry, arm):
    """Refuse a planet the model has seen (traveller) or a planet outside the elder set (elder)."""
    if spec.startswith("fake:"):
        return
    path = Path(spec.replace("{id}", entry["id"])).with_suffix(".json")
    if not path.is_file():
        return
    planets = json.loads(path.read_text(encoding="utf-8")).get("planets") or {}
    if arm == "elder":
        if planets.get("elder") is not None and entry["id"] not in planets["elder"]:
            raise ValueError(f"elder model {spec} was not trained on the early ticks of {entry['id']}")
        return
    seen = {p for key in ("train", "dev", "sealed", "elder") for p in planets.get(key, [])}
    if entry["id"] in seen:
        raise ValueError(f"traveller model {spec} has seen planet {entry['id']}; a traveller must visit a new planet")


def run_arm(entry, arm, *, ticks, policy="fake:rest", elder_policy=None, elder_at=1000, valid="body",
            device=None, bulk_dir=None):
    """One run of one arm on one planet; returns its row (and writes the bulk record if asked)."""
    from .search import build_world
    started = time.monotonic()
    world = build_world(entry)
    order = founder_order(entry, world)
    visitors, ids, placed, spec = None, [], 0, None
    if arm in FOUNDER_ARMS:
        spec = policy
        _check_manifest(spec, entry, arm)
        ids = arm_ids(arm, order)
        visitors = V.Visitors(V.load_policy(spec, device=device, seed=entry["seed"]), ids, label="traveller",
                              valid="engine" if spec == "fake:mirror" else valid, placed_tick=0)
    elif arm == "elder":
        spec = elder_policy.replace("{id}", entry["id"]) if elder_policy else None
        if spec is None:
            raise ValueError("the elder arm needs --elder-policy")
        _check_manifest(spec, entry, arm)
        _advance(world, None, None, elder_at)
        placed = world.tick
        chosen = elder_choice(entry, world) if not world.stop_reason else None
        ids = [chosen] if chosen is not None else []
        visitors = V.Visitors(V.load_policy(spec, device=device, seed=entry["seed"]), label="elder",
                              valid="engine" if spec == "fake:mirror" else valid)
        visitors.place(world, ids)
    elif arm != "baseline":
        raise ValueError(f"unknown arm {arm}")
    if arm == "baseline":
        # the untouched twin records who hears every founder and, from elder_at on, the elder slot,
        # so the arms' contacted natives have a like-for-like control
        tracker = Tracker(world, watch=order)
        if elder_at < ticks:
            _advance(world, None, tracker, elder_at)
            if not world.stop_reason:
                tracker.watch_more([elder_choice(entry, world)])
    else:
        tracker = Tracker(world, watch=ids)
    _advance(world, visitors, tracker, ticks)
    row = {"schema": SCHEMA, "planet": entry["id"], "source": entry.get("source"), "seed": entry["seed"],
           "arm": arm, "visitor_world": arm != "baseline", "visitors": ids, "placed_tick": placed,
           "founder_order": order, "end_tick": world.tick, "requested_ticks": ticks,
           "stop_reason": world.stop_reason, "policy": spec, "population_end": len(world.agents),
           "seconds": round(time.monotonic() - started, 2),
           "visitor_summary": visitors.summary() if visitors is not None else None,
           "summary": {k: v for k, v in world.summary().items() if k in ("population", "counters", "food", "mean_energy")}}
    if bulk_dir is not None:
        path = Path(bulk_dir) / f"{entry['id']}.{arm}.json.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with gzip.open(tmp, "wt", encoding="utf-8") as handle:
            json.dump(tracker.to_dict(), handle, separators=(",", ":"))
        tmp.replace(path)
        row["bulk"] = path.name
    else:
        row["tracker"] = tracker.to_dict()
    return row


def _job(task):
    entry, arm, options = task
    try:
        return run_arm(entry, arm, **options)
    except Exception as exc:  # noqa: BLE001 - a failed run keeps its row
        return {"schema": SCHEMA, "planet": entry["id"], "arm": arm, "error": f"{type(exc).__name__}: {exc}"}


# ---------------------------------------------------------------- measures (pairing)

def _series_at(track, tick, key):
    """Value of a per-tick series at an absolute tick; past the run's end: 0 population after
    extinction, the last value otherwise (food is cumulative)."""
    series = track[key]
    index = tick - track["start"]
    if index < 0:
        raise ValueError("tick before the tracked window")
    return series[index] if index < len(series) else series[-1]


def world_measures(track, start, end, stop_reason=None):
    """Mean population over [start, end] (after each step), end population, food energy per agent-tick."""
    series = track["population"]
    pops = []
    for tick in range(start + 1, end + 1):
        index = tick - track["start"]
        if index < len(series):
            pops.append(series[index])
        else:
            pops.append(0 if stop_reason == "extinction" else series[-1])
    # agent-ticks: the population at the start of every step taken inside the window
    agent_ticks = sum(series[t - track["start"]] for t in range(start, end) if t - track["start"] + 1 < len(series))
    food = _series_at(track, end, "food") - _series_at(track, start, "food")
    return {"population_mean": sum(pops) / len(pops) if pops else None,
            "population_end": pops[-1] if pops else None, "agent_ticks": agent_ticks,
            "food_energy_per_agent_tick": food / agent_ticks if agent_ticks else None}


def individual(track, identity, start, end):
    row = track["agents"].get(str(identity))
    if row is None:
        return None
    died = row["died"]
    inside = lambda tick: start < tick <= end
    teach = sum(1 for d in track["demos"] if d[1] == "teach" and d[2] == identity and d[5] and inside(d[0]))
    imitated = sum(1 for d in track["demos"] if d[1] == "imitate" and d[2] == identity and d[5] and inside(d[0]))
    return {"survival_ticks": (died if died is not None else end) - start, "censored": died is None,
            "death_reason": row["reason"], "offspring": sum(1 for t in row["births"] if inside(t)),
            "teach": teach, "imitated": imitated, "demonstrations": teach + imitated,
            "calls": sum(1 for s in track["signals"] if s[1] == identity and inside(s[0]))}


def _mapping(rows):
    """Pooled {situation: Counter(symbol)} of signal rows."""
    pooled = defaultdict(Counter)
    for _, _, symbol, situation in rows:
        pooled[tuple(situation)][symbol] += 1
    return pooled


def _agreement(rows, modes):
    """raw: share of rows whose symbol is the visitors' mode in that situation; blind: the same
    expected from the rows' situation-blind symbol distribution; specific = raw - blind."""
    rows = [r for r in rows if tuple(r[3]) in modes]
    if not rows:
        return {"calls": 0, "raw": None, "blind": None, "specific": None}
    marginal = Counter(r[2] for r in rows)
    n = len(rows)
    raw = sum(r[2] == modes[tuple(r[3])] for r in rows) / n
    blind = sum(marginal[modes[tuple(r[3])]] / n for r in rows) / n
    return {"calls": n, "raw": round(raw, 6), "blind": round(blind, 6), "specific": round(raw - blind, 6)}


def call_adoption(arm_track, base_track, visitors, start, end, *, reps=200, seed=0):
    """Do natives' calls move toward the visitors' situation -> symbol mapping, and does it last?"""
    from .lineage import _info, _modes
    visitors = set(visitors)
    own = [r for r in arm_track["signals"] if r[1] in visitors and start < r[0] <= end]
    pooled = _mapping(own)
    modes = _modes(pooled)
    deaths = [arm_track["agents"].get(str(v), {}).get("died") for v in visitors]
    last_death = max(deaths) if deaths and all(d is not None for d in deaths) else None
    result = {"visitor_calls": len(own), "situations": len(modes),
              "visitor_mapping": {"|".join(map(str, k)): v for k, v in sorted(modes.items(), key=repr)},
              "visitor_info": _info(pooled, reps, seed) if own else None, "last_visitor_death": last_death}
    if not modes:
        result.update(contact=None, after=None, contacted=None)
        return result

    def natives(track, lo, hi):  # calls made in the steps that end at ticks lo+1 .. hi
        return [r for r in track["signals"] if r[1] not in visitors and lo < r[0] <= hi]
    contact_end = min(last_death, end) if last_death is not None else end
    contact = {"arm": _agreement(natives(arm_track, start, contact_end), modes),
               "baseline": _agreement(natives(base_track, start, contact_end), modes)}
    contact["moved"] = _diff(contact["arm"]["specific"], contact["baseline"]["specific"])
    result["contact"] = contact
    if last_death is not None and last_death < end:
        after = {"arm": _agreement(natives(arm_track, last_death, end), modes),
                 "baseline": _agreement(natives(base_track, last_death, end), modes)}
        after["persists"] = _diff(after["arm"]["specific"], after["baseline"]["specific"])
        result["after"] = after
    else:
        result["after"] = None
    def contacted(track):  # natives' calls after they first heard a visitor (baseline: the same slots)
        first = {}
        for tick, receiver, sender in track["heard"]:
            if sender in visitors and receiver not in visitors and start <= tick <= end:
                first[receiver] = min(first.get(receiver, tick), tick)
        rows = [r for r in track["signals"] if r[1] in first and r[0] > first[r[1]] and start < r[0] <= end]
        return {"natives": len(first), **_agreement(rows, modes)}
    arm_c, base_c = contacted(arm_track), contacted(base_track)
    result["contacted"] = {"arm": arm_c, "baseline": base_c, "moved": _diff(arm_c["specific"], base_c["specific"])}
    return result


def _diff(a, b):
    return None if a is None or b is None else round(a - b, 6)


def _mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def pair(arm_row, arm_track, base_row, base_track, *, reps=200):
    """Measures of one arm run against its baseline twin over the arm's window."""
    start, end = arm_row["placed_tick"], arm_row["requested_ticks"]
    out = {"planet": arm_row["planet"], "arm": arm_row["arm"], "visitors": arm_row["visitors"], "window": [start, end],
           "stop_reason": arm_row["stop_reason"], "baseline_stop_reason": base_row["stop_reason"]}
    a = world_measures(arm_track, start, end, arm_row["stop_reason"])
    b = world_measures(base_track, start, end, base_row["stop_reason"])
    out["world"] = {"arm": a, "baseline": b, "diff": {k: _diff(a[k], b[k]) for k in a}}
    slots = []
    for identity in arm_row["visitors"]:
        mine, native = individual(arm_track, identity, start, end), individual(base_track, identity, start, end)
        if mine is None or native is None:
            continue
        slots.append({"id": identity, "visitor": mine, "native": native,
                      "diff": {k: mine[k] - native[k] for k in ("survival_ticks", "offspring", "demonstrations", "calls")}})
    out["slots"] = slots
    out["slot_mean_diff"] = {k: _mean([s["diff"][k] for s in slots])
                             for k in ("survival_ticks", "offspring", "demonstrations", "calls")} if slots else None
    out["slot_mean"] = {side: {k: _mean([s[side][k] for s in slots]) for k in
                               ("survival_ticks", "offspring", "demonstrations", "calls")} for side in ("visitor", "native")} \
        if slots else None
    out["deaths"] = {side: dict(Counter(str(s[side]["death_reason"]) for s in slots)) for side in ("visitor", "native")}
    out["adoption"] = call_adoption(arm_track, base_track, arm_row["visitors"], start, end, reps=reps,
                                    seed=arm_row["seed"]) if arm_row["visitors"] else None
    out["visitor_summary"] = {k: arm_row["visitor_summary"][k] for k in ("decisions", "fallbacks", "same_as_native",
                                                                          "actions")} if arm_row.get("visitor_summary") else None
    return out


METRICS = {
    "population_mean": lambda p: p["world"]["diff"]["population_mean"],
    "population_end": lambda p: p["world"]["diff"]["population_end"],
    "food_energy_per_agent_tick": lambda p: p["world"]["diff"]["food_energy_per_agent_tick"],
    "visitor_minus_native_survival_ticks": lambda p: (p["slot_mean_diff"] or {}).get("survival_ticks"),
    "visitor_minus_native_offspring": lambda p: (p["slot_mean_diff"] or {}).get("offspring"),
    "visitor_minus_native_demonstrations": lambda p: (p["slot_mean_diff"] or {}).get("demonstrations"),
    "visitor_minus_native_calls": lambda p: (p["slot_mean_diff"] or {}).get("calls"),
    "adoption_moved_specific": lambda p: ((p["adoption"] or {}).get("contact") or {}).get("moved"),
    "adoption_persists_specific": lambda p: ((p["adoption"] or {}).get("after") or {}).get("persists"),
    "adoption_contacted_moved_specific": lambda p: ((p["adoption"] or {}).get("contacted") or {}).get("moved"),
}


def paired_stats(values):
    """Mean +- SE and a two-sided sign test over per-planet differences (None dropped)."""
    from .discovery import sign_test
    values = [v for v in values if v is not None]
    n = len(values)
    mean = sum(values) / n if n else None
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1)) if n > 1 else None
    return {"n": n, "mean": None if mean is None else round(mean, 6), "sd": None if sd is None else round(sd, 6),
            "se": None if sd is None else round(sd / math.sqrt(n), 6),
            "positive": sum(v > 0 for v in values), "negative": sum(v < 0 for v in values),
            "zero": sum(v == 0 for v in values), "sign_p": round(sign_test(values), 6)}


# ---------------------------------------------------------------- folders

def _read_rows(out):
    path = Path(out) / "rows.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _track(out, row):
    if "tracker" in row:
        return row["tracker"]
    with gzip.open(Path(out) / "bulk" / row["bulk"], "rt", encoding="utf-8") as handle:
        return json.load(handle)


def select(plan_path, *, group=None, ids=None, first=None):
    """Plan entries by ids, or a group of experience.select_planets with the sources alternating
    (world7, synthetic, world7, ...; each in id order), so --first N mixes both sources."""
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))["plan"]
    by_id = {e["id"]: e for e in plan}
    if ids:
        chosen = list(ids)
    elif group:
        from . import experience
        members = experience.select_planets(plan)[group]
        lanes = [[i for i in members if by_id[i]["source"] == source] for source in ("world7", "synthetic")]
        lanes.append([i for i in members if by_id[i]["source"] not in ("world7", "synthetic")])
        chosen = []
        for k in range(max(len(lane) for lane in lanes)):
            chosen += [lane[k] for lane in lanes if k < len(lane)]
    else:
        chosen = [e["id"] for e in plan]
    missing = [i for i in chosen if i not in by_id]
    if missing:
        raise ValueError(f"planets not in the plan: {missing[:5]}")
    chosen = chosen[:first] if first else chosen
    return [by_id[i] for i in chosen]


def run(out, entries, arms, *, ticks, policy, elder_policy=None, elder_at=1000, valid="body", device=None,
        workers=1, log=None):
    log = log or (lambda text: print(text, flush=True))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.json"
    meta = {"schema": SCHEMA, "note": NOTE, "arms": list(arms), "ticks": ticks, "policy": policy,
            "elder_policy": elder_policy, "elder_at": elder_at, "valid": valid,
            "planets": [e["id"] for e in entries], "python": sys.version.split()[0]}
    if meta_path.is_file():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        for key in ("ticks", "policy", "elder_policy", "elder_at", "valid"):
            if old.get(key) != meta[key]:
                raise ValueError(f"{out} was run with {key}={old.get(key)!r}; use a new folder")
        meta["planets"] = sorted(set(old.get("planets", [])) | set(meta["planets"]))
        meta["arms"] = sorted(set(old.get("arms", [])) | set(arms), key=ARMS.index)
    meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    done = {(r["planet"], r["arm"]) for r in _read_rows(out) if "error" not in r}
    options = {"ticks": ticks, "policy": policy, "elder_policy": elder_policy, "elder_at": elder_at,
               "valid": valid, "device": device, "bulk_dir": str(out / "bulk")}
    tasks = [(e, arm, options) for e in entries for arm in arms if (e["id"], arm) not in done]
    started = time.monotonic()
    log(json.dumps({"tasks": len(tasks), "skipped": len(entries) * len(arms) - len(tasks), "workers": workers}))
    with (out / "rows.jsonl").open("a", encoding="utf-8") as handle:
        def write(row):
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            handle.flush()
            log(json.dumps({"planet": row["planet"], "arm": row["arm"], "end_tick": row.get("end_tick"),
                            "stop": row.get("stop_reason"), "error": row.get("error"), "seconds": row.get("seconds"),
                            "elapsed": round(time.monotonic() - started, 1)}))
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for row in pool.map(_job, tasks, chunksize=1):
                    write(row)
        else:
            for task in tasks:
                write(_job(task))
    return report(out)


def report(out, *, reps=200):
    out = Path(out)
    rows = [r for r in _read_rows(out) if "error" not in r]
    errors = [r for r in _read_rows(out) if "error" in r]
    base = {r["planet"]: r for r in rows if r["arm"] == "baseline"}
    pairs = []
    for row in sorted((r for r in rows if r["arm"] != "baseline"), key=lambda r: (r["planet"], ARMS.index(r["arm"]))):
        twin = base.get(row["planet"])
        if twin is None:
            continue
        pairs.append(pair(row, _track(out, row), twin, _track(out, twin), reps=reps))
    arms = {}
    for arm in ARMS[1:]:
        mine = [p for p in pairs if p["arm"] == arm]
        if not mine:
            continue
        arms[arm] = {"planets": len(mine), "visitors": sum(len(p["visitors"]) for p in mine),
                     "visitor_deaths": sum(not s["visitor"]["censored"] for p in mine for s in p["slots"]),
                     "native_deaths": sum(not s["native"]["censored"] for p in mine for s in p["slots"]),
                     "fallbacks": sum((p["visitor_summary"] or {}).get("fallbacks", 0) for p in mine),
                     "decisions": sum((p["visitor_summary"] or {}).get("decisions", 0) for p in mine),
                     "same_as_native": sum((p["visitor_summary"] or {}).get("same_as_native", 0) for p in mine),
                     "visitor_actions": dict(sum((Counter((p["visitor_summary"] or {}).get("actions", {}))
                                                  for p in mine), Counter()).most_common()),
                     "slot_means": {side: {k: _mean([(p["slot_mean"] or {}).get(side, {}).get(k) for p in mine])
                                           for k in ("survival_ticks", "offspring", "demonstrations", "calls")}
                                    for side in ("visitor", "native")},
                     "paired": {name: paired_stats([get(p) for p in mine]) for name, get in METRICS.items()}}
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8")) if (out / "meta.json").is_file() else {}
    result = {"schema": SCHEMA, "note": NOTE, "meta": meta, "runs": len(rows), "errors": errors,
              "baselines": len(base), "arms": arms, "pairs": pairs,
              "interpretation": "Differences are arm minus the untouched twin of the same planet over the visitors' "
                                "window; sign_p is a two-sided sign test over planets. A null is a null."}
    (out / "report.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


def compare(out, control, *, target=None):
    """Arm effects of ``out`` minus those of ``control`` (e.g. a model against fake:random visitors on the
    same planets, arms, ticks): per planet (arm - baseline) - (control arm - control baseline). Both
    baselines are the same deterministic untouched twin, so this is the model arm minus the control arm."""
    mine = json.loads((Path(out) / "report.json").read_text(encoding="utf-8"))
    other = json.loads((Path(control) / "report.json").read_text(encoding="utf-8"))
    theirs = {(p["planet"], p["arm"]): p for p in other["pairs"]}
    arms = {}
    for arm in ARMS[1:]:
        matched = [(p, theirs[(p["planet"], arm)]) for p in mine["pairs"] if p["arm"] == arm and (p["planet"], arm) in theirs]
        if matched:
            arms[arm] = {"planets": len(matched), "paired": {
                name: paired_stats([None if get(a) is None or get(b) is None else get(a) - get(b) for a, b in matched])
                for name, get in METRICS.items()}}
    result = {"schema": SCHEMA, "kind": "compare", "out": str(out), "control": str(control),
              "policy": mine.get("meta", {}).get("policy"), "control_policy": other.get("meta", {}).get("policy"),
              "elder_policy": mine.get("meta", {}).get("elder_policy"),
              "control_elder_policy": other.get("meta", {}).get("elder_policy"), "arms": arms}
    path = Path(target) if target else Path(out) / f"compare-{Path(control).name}.json"
    path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="run paired twins (resumable per planet and arm)")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    r.add_argument("--group", choices=("traveller", "elder", "train", "dev", "sealed"),
                   help="planets of experience.select_planets (reserved traveller/elder sets)")
    r.add_argument("--ids", help="comma-separated planet ids from the plan")
    r.add_argument("--first", type=int, help="only the first N selected planets")
    r.add_argument("--arms", help=f"comma-separated, from {ARMS} (default: founder arms, or baseline,elder for --group elder)")
    r.add_argument("--ticks", type=int, default=2000)
    r.add_argument("--policy", default="fake:rest", help="student checkpoint, or fake:rest|random|first|mirror")
    r.add_argument("--elder-policy", help="checkpoint (may contain {id}) or fake:<mode>")
    r.add_argument("--elder-at", type=int, default=1000)
    r.add_argument("--valid", choices=V.VALID_MODES, default="body")
    r.add_argument("--device")
    r.add_argument("--workers", type=int, default=1)
    p = sub.add_parser("report", help="pair the arms with their baselines")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--reps", type=int, default=200)
    c = sub.add_parser("compare", help="arm effects of one folder minus another's (e.g. model minus fake:random)")
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--control", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.command == "run":
        entries = select(args.plan, group=args.group, ids=args.ids.split(",") if args.ids else None, first=args.first)
        arms = args.arms.split(",") if args.arms else (["baseline", "elder"] if args.group == "elder"
                                                       else ["baseline", *FOUNDER_ARMS])
        for arm in arms:
            if arm not in ARMS:
                raise SystemExit(f"unknown arm {arm}")
        if "elder" in arms and args.ticks <= args.elder_at:
            raise SystemExit("--ticks must exceed --elder-at for the elder arm")
        result = run(args.out, entries, arms, ticks=args.ticks, policy=args.policy, elder_policy=args.elder_policy,
                     elder_at=args.elder_at, valid=args.valid, device=args.device, workers=args.workers)
    elif args.command == "compare":
        result = compare(args.out, args.control)
    else:
        result = report(args.out, reps=args.reps)
    print(json.dumps({arm: {k: v for k, v in body.items() if k == "paired"} for arm, body in result["arms"].items()},
                     indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
