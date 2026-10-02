"""Many-world search over life8 worlds by successive halving (standard library only).

A search runs a population of living worlds with compact logging in a process pool,
measures each one, keeps the best fraction by a documented composite score, continues
the kept worlds from their checkpoints for longer, and so on.

Candidates (``--source``): ``plain`` (World.create(seed) with the pass-through world
options), ``synthetic`` (bridge.synthetic_world planets), ``world7`` (round-7 worlds
that reach bodies, bridge.candidate_worlds; slow to scan), ``mixed`` (plain and
synthetic alternately) or ``bridged`` (world7 and synthetic alternately). World
options are a pass-through dict of Config fields (``--option key=value``, value parsed
as JSON when possible), so fields added by other stages work unchanged; for bridged
worlds they form the base Config and the bridge sets its own fields on top.
``--preset living`` puts config.LIVING under the options: the worlds where signalling
can pay (blooms, predators, calls beyond sight, kin credit, receiver features, evolving
genome). With the ecology on, bridged worlds also get their planet's predators and
temperature (bridge.connect_ecology), and the sender state and convention situation
gain danger_seen, danger_direction and bloom_seen (ECOLOGY_BANDS, as analysis/ecology.py).

Rounds. Round r runs every surviving world for ``ticks * growth**r`` ticks (round 0
from tick 0, later rounds from the checkpoint saved when the world was kept).

* Round 0 is a screen: only measurements from the single normal run.
* Rounds >= 1 add paired branches from the end-of-round state, for the kept worlds
  only: ``reps`` replicates of ``branch_ticks`` ticks in four arms (normal,
  ``channel=masked``, ``channel=scrambled``, ``tools=False``), with the world RNG
  reseeded identically per replicate in every arm (discovery.BRANCH_SALT).

Measurements (window = the last half of the round's ticks):

* sender: I(symbol; joint sender state) of compact ``signal`` events minus a
  permutation null (pooled), and the same against shuffles within each sender's own
  calls (within-sender: removes individual symbol habits that happen to correlate with
  state). The joint state is food_near, food_seen, energy_band, neighbor_near, plus
  ECOLOGY_BANDS with the ecology on. The criterion uses the within-sender test; in
  paired rounds it also needs the normal arm's within-sender excess to beat the masked
  arm's (where symbols cannot matter; scrambled if masked is missing) by
  sender_over_control_bits (round-8 critic B2). Screen score: the smaller excess and
  the larger p of pooled and within; paired score: the excess over the control.
* receiver: I(heard symbol; next action) minus a permutation null. Screen: within the
  normal run. Paired: normal-branch excess minus scrambled-branch excess.
* scramble_fitness (paired only): normal minus scrambled food energy per agent-tick
  over the replicates, exact sign test (normal minus masked is reported too).
* convention: situation-specific symbol agreement within founder lineages minus across
  lineages (pair agreement minus the agreement of the pair's situation-blind symbol
  distributions, so shared or inherited habits alone give 0; critic F2), label-shuffle p
  (as lineage.convention_agreement). Situation: food_seen, energy_band (+ ECOLOGY_BANDS).
* tools: paired: normal minus tools-off food energy per agent-tick, sign test.
  Screen: the engine's tool_damage_gain counter per agent-tick (a within-tick
  counterfactual of damage, not a fitness benefit; it gets the unpaired factor).
* cooperation: share reciprocity against a proximity-matched null. For every share
  A->B, "gave back" means B shared with A in the previous ``RECIPROCITY_WINDOW`` ticks.
  Observed = share of recipients who gave back; null = the same for a recipient drawn
  uniformly from all individuals within interaction range of A at that tick (the
  actual recipient included); p from ``perm_reps`` such draws.

Composite score (ranking only; the same definition for every world in one round):
    score = sum over components of WEIGHTS[c] * min(2, max(0, x_c) / SCALE[c]) * f_c
with f_c = 1 if the component's p <= 0.05 and 0.25 otherwise (and 0.25 for the
unpaired tool screen). A component that could not be measured (no calls, no shares)
contributes 0 and is reported as None. Worlds that stopped (extinction, capacity) are
scored and kept in results.jsonl but are not eligible to continue. Ties break on
world id. Pass/fail flags use stricter cut-offs (CRITERIA; the language ones equal
discovery.THRESHOLDS). Toy-world numbers: a pass means a measured dependency or
effect in this simulation, not language, technology or institutions.

Files in ``--out``: search.json (plan, arguments, status, stop reason), results.jsonl
(one ``world`` row per world per round, including failures, plus one ``selection``
row per round listing kept and dropped ids and the cut-off), checkpoints/<id>.r<round>.json.gz
(only for kept worlds, the latest round only: world state, lineage founders, recent shares), pending/ (a
round's end states until selection; then promoted or deleted). Re-running ``run``
on the same folder resumes: finished (round, world) rows are skipped, and a kept
world whose end state was lost is re-simulated deterministically.

usage: python -m haishool.life8.search run --out DIR --worlds N --rounds R --ticks T --keep F --workers W
           [--source plain|synthetic|world7|mixed|bridged] [--preset living] [--start-seed S] [--growth G]
           [--reps K] [--branch-ticks B] [--perm-reps P] [--max-output-mb M] [--option key=value ...]
       python -m haishool.life8.search plan --out PLAN.json --worlds N --source S --start-seed S
           [--preset living] [--option ...]
       python -m haishool.life8.search split --plan PLAN.json --out-dir DIR NAME=COUNT ...
       python -m haishool.life8.search run --out DIR --plan PLAN.json ...   (no numpy needed on that host)
       python -m haishool.life8.search report --out DIR [--top N]
       python -m haishool.life8.search collect --out MERGED DIR [DIR ...]
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .config import LIVING, Config
from .world import World

SCHEMA = "life8-search-v1"
BUNDLE_SCHEMA = "life8-search-bundle-v1"
DEFAULT_OPTIONS = {"food_patches": 30, "population_limit": 256}
# --preset NAME merges these Config fields under the --option values ("living" = config.LIVING:
# blooms, predators, calls beyond sight, kin credit, receiver and reciprocity features, evolving genome).
PRESETS = {"living": LIVING}
SENDER_STATE = ("food_near", "food_seen", "energy_band", "neighbor_near")
SITUATION = ("food_seen", "energy_band")
# With the ecology on, these living-world bins are what calls could be about (as analysis/ecology.py).
ECOLOGY_BANDS = ("danger_seen", "danger_direction", "bloom_seen")
BRANCH_SALT = 0x51A7E5  # same reseeding rule as discovery.branch
# masked: calls are delivered and paid for but their symbol is hidden from every controller,
# so symbol content cannot matter; it is the control arm of the sender criterion.
ARMS = {"normal": {}, "masked": {"channel": "masked"}, "scrambled": {"channel": "scrambled"},
        "tools_off": {"tools": False}}
SENDER_CONTROLS = ("masked", "scrambled")  # the first one measured is the sender control
RECIPROCITY_WINDOW = 100
WEIGHTS = {"sender": 1., "receiver": 1., "scramble_fitness": 1., "convention": 1., "tools": 1., "cooperation": 1.}
SCALE = {"sender": .05, "sender_over_control": .02, "receiver": .02, "scramble_fitness": .004, "convention": .05,
         "tools": .004, "tools_screen": .01, "cooperation": .05}
ALPHA = .05
CRITERIA = {"sender_excess_bits": .05, "sender_p": .01, "sender_over_control_bits": .02,
            "receiver_excess_bits": .02, "receiver_p": .01,
            "receiver_over_scrambled_bits": .02, "fitness_min_reps": 6, "fitness_sign_p": .05,
            "convention_difference": .05, "convention_p": .01, "tool_min_reps": 6, "tool_sign_p": .05,
            "cooperation_excess": .05, "cooperation_p": .01, "cooperation_min_shares": 20}
INTERPRETATION = ("Toy-world search. Scores rank worlds for further simulation; a passed criterion is a measured "
                  "dependency or paired effect in this simulation, not language, technology or institutions.")


# ---------------------------------------------------------------- pure-python statistics

def _mi(xs, ys):
    n = len(xs)
    if not n:
        return 0.
    joint, cx, cy = Counter(zip(xs, ys)), Counter(xs), Counter(ys)
    return sum(c / n * math.log2(c * n / (cx[x] * cy[y])) for (x, y), c in joint.items())


def permutation_mi(xs, ys, *, reps=100, seed=0, groups=None):
    """I(X;Y) in bits against ``reps`` shuffles of Y (only within each group when
    ``groups`` is given): excess = observed - null mean, p = (1 + #null >= observed) / (reps + 1)."""
    n = len(xs)
    if n == 0:
        return {"n": 0, "mi_bits": None, "excess_bits": None, "p_value": None}
    observed = _mi(xs, ys)
    rng, shuffled, null = random.Random(seed), list(ys), []
    blocks = []
    if groups is not None:
        index = defaultdict(list)
        for position, group in enumerate(groups):
            index[group].append(position)
        blocks = [index[g] for g in sorted(index, key=repr)]
    for _ in range(reps):
        if groups is None:
            rng.shuffle(shuffled)
        else:
            for block in blocks:
                values = [ys[i] for i in block]
                rng.shuffle(values)
                for i, v in zip(block, values):
                    shuffled[i] = v
        null.append(_mi(xs, shuffled))
    mean = sum(null) / reps if reps else 0.
    return {"n": n, "mi_bits": round(observed, 6), "null_mean": round(mean, 6),
            "excess_bits": round(observed - mean, 6),
            "p_value": round((1 + sum(v >= observed - 1e-12 for v in null)) / (reps + 1), 6)}


def sign_test(differences):
    """Two-sided exact binomial sign test over nonzero paired differences."""
    wins, losses = sum(d > 0 for d in differences), sum(d < 0 for d in differences)
    n = wins + losses
    if not n:
        return 1.
    return min(1., 2 * sum(math.comb(n, i) for i in range(max(wins, losses), n + 1)) / 2 ** n)


def paired(values):
    n = len(values)
    mean = sum(values) / n if n else None
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1)) if n > 1 else None
    return {"n": n, "mean": None if mean is None else round(mean, 6), "sd": None if sd is None else round(sd, 6),
            "positive": sum(v > 0 for v in values), "negative": sum(v < 0 for v in values),
            "sign_p": round(sign_test(values), 6)}


def _food_seen(state):
    return state.get("food_direction") not in (None, "none")


def _bands(state, names):
    return tuple(_food_seen(state) if k == "food_seen" else state.get(k) for k in names)


def sender_state(config):
    """The joint sender state of a world: SENDER_STATE, plus ECOLOGY_BANDS when the ecology is on."""
    return SENDER_STATE + (ECOLOGY_BANDS if config.ecology_on() else ())


def situation_of(config):
    """The convention situation of a world: SITUATION, plus ECOLOGY_BANDS when the ecology is on."""
    return SITUATION + (ECOLOGY_BANDS if config.ecology_on() else ())


def sender_information(signals, *, reps=100, seed=0, state=SENDER_STATE):
    """signals: (tick, sender, state bands, symbol) rows. The pooled test and the
    within-sender test (shuffles only inside each sender's calls); the criterion uses the latter."""
    joint = [s if isinstance(s, tuple) else _bands(s, state) for _, _, s, _ in signals]  # tuple: already banded
    symbols = [row[3] for row in signals]
    return {"signals": len(signals), "senders": len({row[1] for row in signals}), "state": list(state),
            **permutation_mi(joint, symbols, reps=reps, seed=seed),
            "within_sender": permutation_mi(joint, symbols, reps=reps, seed=seed + 1,
                                            groups=[row[1] for row in signals])}


def receiver_information(heard, *, reps=100, seed=0):
    """heard: (symbol, next_action) pairs."""
    return permutation_mi([h[0] for h in heard], [h[1] for h in heard], reps=reps, seed=seed)


def convention_agreement(signals, founder_of, *, situation=SITUATION, min_calls=5, reps=100, seed=0):
    """Situation-specific symbol agreement within founder lineages minus across them
    (the statistic of lineage.convention_agreement, pure python).

    A pair's agreement is P(same symbol) in each situation both used, averaged; its blind
    agreement is the same for the two senders' symbol distributions averaged over those
    situations, i.e. what shared symbol habits give without any situation mapping (round-8
    critic F2: inherited bias rows). difference = mean(agreement - blind) within lineages
    minus across, tested against label shuffles; within/across report the raw agreement."""
    counts = defaultdict(lambda: defaultdict(Counter))
    for _, sender, state, symbol in signals:
        counts[sender][_bands(state, situation)][symbol] += 1
    senders = sorted(s for s, m in counts.items() if sum(sum(c.values()) for c in m.values()) >= min_calls
                     and s in founder_of)
    if len(senders) < 3:
        return {"senders": len(senders), "within": None, "across": None, "difference": None, "p_value": None}
    probs = [{k: {sym: c / sum(cnt.values()) for sym, c in cnt.items()} for k, cnt in counts[s].items()}
             for s in senders]
    pairs = []
    for i in range(len(senders)):
        for j in range(i + 1, len(senders)):
            shared = [k for k in probs[i] if k in probs[j]]
            if shared:
                pi, pj = probs[i], probs[j]
                agree = sum(sum(p * pj[k].get(sym, 0.) for sym, p in pi[k].items()) for k in shared) / len(shared)
                mi, mj = Counter(), Counter()
                for k in shared:
                    mi.update(pi[k])
                    mj.update(pj[k])
                blind = sum(p * mj.get(sym, 0.) for sym, p in mi.items()) / len(shared) ** 2
                pairs.append((i, j, agree, blind, agree - blind))
    labels = [founder_of[s] for s in senders]

    def split(lab, index):  # index 2: agreement, 3: blind agreement, 4: situation-specific agreement
        w = [p[index] for p in pairs if lab[p[0]] == lab[p[1]]]
        x = [p[index] for p in pairs if lab[p[0]] != lab[p[1]]]
        return (sum(w) / len(w) if w else None), (sum(x) / len(x) if x else None)

    within, across = split(labels, 2)
    within_blind, across_blind = split(labels, 3)
    r = lambda v: None if v is None else round(v, 5)
    out = {"senders": len(senders), "lineages": len(set(labels)), "pairs": len(pairs), "situation": list(situation),
           "within": r(within), "across": r(across), "within_blind": r(within_blind), "across_blind": r(across_blind)}
    if within is None or across is None:
        return {**out, "difference": None, "difference_raw": None, "p_value": None}
    w, a = split(labels, 4)
    observed, rng, null, shuffled = w - a, random.Random(seed), [], list(labels)
    for _ in range(reps):
        rng.shuffle(shuffled)
        w, a = split(shuffled, 4)
        if w is not None and a is not None:
            null.append(w - a)
    return {**out, "difference": round(observed, 5), "difference_raw": round(within - across, 5),
            "p_value": round((1 + sum(v >= observed - 1e-12 for v in null)) / (len(null) + 1), 5)}


def reciprocity(shares, *, reps=100, seed=0):
    """shares: (tick, giver, receiver_gave_back, [gave_back flag of each candidate]) with
    >= 2 candidates. Observed vs a proximity-matched uniform choice among candidates."""
    rows = [s for s in shares if len(s[3]) >= 2]
    if not rows:
        return {"shares": 0, "observed": None, "null": None, "excess": None, "p_value": None}
    observed = sum(r[2] for r in rows) / len(rows)
    expected = sum(sum(r[3]) / len(r[3]) for r in rows) / len(rows)
    rng = random.Random(seed)
    null = [sum(rng.choice(r[3]) for r in rows) / len(rows) for _ in range(reps)]
    return {"shares": len(rows), "observed": round(observed, 5), "null": round(expected, 5),
            "excess": round(observed - expected, 5),
            "p_value": round((1 + sum(v >= observed - 1e-12 for v in null)) / (reps + 1), 5)}


# ---------------------------------------------------------------- plan and worlds

def parse_option(text):
    key, sep, value = text.partition("=")
    if not sep or not key:
        raise argparse.ArgumentTypeError("options are key=value")
    try:
        return key, json.loads(value)
    except json.JSONDecodeError:
        return key, value


SOURCES = ("plain", "synthetic", "world7", "mixed", "bridged")


def resolve_options(preset=None, options=None):
    """The Config fields of a search: the preset's (if any), then the --option values on top."""
    if preset is not None and preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; known: {', '.join(sorted(PRESETS))}")
    return {**(PRESETS[preset] if preset else {}), **(options or {})}


def _base_config(options):
    values = {**DEFAULT_OPTIONS, **(options or {})}
    values["log"] = "compact"
    return Config(**values)


def make_plan(source, worlds, start_seed=0, options=None, *, world7_workers=1, preset=None):
    """The candidate list: one entry per world with its id, seed, Config dict and founders.

    ``mixed`` alternates plain and synthetic worlds, ``bridged`` alternates world7 worlds
    that reach bodies (bridge.candidate_worlds from ``start_seed``) and synthetic planets.
    With the ecology on (e.g. preset ``living``), bridged worlds get their planet's
    predators, predator damage and visibility and ambient temperature (bridge.connect_ecology)."""
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}")
    base = _base_config(resolve_options(preset, options))
    plan = []
    if source != "plain":
        from . import bridge
    if source in ("world7", "bridged"):
        wanted = worlds if source == "world7" else (worlds + 1) // 2
        records = bridge.candidate_worlds(wanted, start_seed=start_seed, workers=world7_workers)
        if len(records) < wanted:
            raise ValueError(f"only {len(records)} of {wanted} world7 candidates found")
    for index in range(worlds):
        seed = start_seed + index
        kind = {"mixed": ("plain", "synthetic"), "bridged": ("world7", "synthetic")}.get(source, (source,))[
            index % (2 if source in ("mixed", "bridged") else 1)]
        if kind == "plain":
            plan.append({"id": f"plain-{seed}", "source": "plain", "seed": seed, "config": base.to_dict(),
                         "founders": None})
            continue
        record = (records[index // 2 if source == "bridged" else index] if kind == "world7"
                  else bridge.synthetic_world(seed))
        config, founders = bridge.world_config(record, base=base)
        if base.ecology_on():
            config, founders = bridge.connect_ecology(config, founders)
        plan.append({"id": f"{kind}-{record['seed']}", "source": kind, "seed": record["seed"],
                     "config": config.to_dict(), "founders": founders})
    if len({entry["id"] for entry in plan}) != len(plan):
        raise ValueError("duplicate world ids in the plan")
    return plan


def split_plan(plan, counts):
    """Consecutive slices of ``plan`` with the given sizes (one per host); they must use it all."""
    if sum(counts) != len(plan) or min(counts, default=1) < 1:
        raise ValueError(f"part sizes {counts} do not add up to the plan's {len(plan)} worlds")
    parts, start = [], 0
    for count in counts:
        parts.append(plan[start:start + count])
        start += count
    return parts


def build_world(entry):
    config = Config.from_dict({**entry["config"], "log": "compact"})
    if entry.get("founders"):
        from . import bridge
        return bridge.create_world(entry["seed"], config, entry["founders"])
    return World.create(seed=entry["seed"], config=config)


def _write_gz(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=6) as stream:
        json.dump(data, stream, separators=(",", ":"))
    os.replace(tmp, path)


def _read_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


class Tracker:
    """Founder of every individual (dead ones included) and recent shares, carried in the bundle."""

    def __init__(self, world=None, data=None):
        if data is not None:
            self.founder = {int(k): v for k, v in data["founder"].items()}
            self.shares = [tuple(s) for s in data["shares"]]
        else:
            self.founder, self.shares = {}, []
            for agent in sorted(world.agents.values(), key=lambda a: a.id):
                self.founder[agent.id] = self.founder.get(agent.parent_id, agent.id) \
                    if agent.parent_id is not None else agent.id

    def to_dict(self):
        return {"founder": {str(k): v for k, v in sorted(self.founder.items())}, "shares": [list(s) for s in self.shares]}

    def record(self, world, events, measured):
        tick = world.tick
        for event in events:
            kind = event["type"]
            if kind == "birth":
                self.founder[event["agent"]] = self.founder.get(event["parent"], event["parent"])
            elif kind == "act" and event["action"] == "share" and event.get("success"):
                giver, receiver = event["agent"], event["target"]
                recent = {(g, r) for t, g, r in self.shares if t >= tick - RECIPROCITY_WINDOW}
                if giver in world.agents:
                    me = world.agents[giver]
                    near = [a.id for a in world.agents.values() if a.id != giver and
                            world.distance(me.position, a.position) <= world.config.interaction_range]
                    if receiver not in near:
                        near.append(receiver)
                    measured.append((tick, giver, (receiver, giver) in recent,
                                     [(c, giver) in recent for c in sorted(near)]))
                self.shares.append((tick, giver, receiver))
        self.shares = [s for s in self.shares if s[0] > tick - RECIPROCITY_WINDOW]


def _bundle(entry, world, tracker, round_index):
    return {"schema": BUNDLE_SCHEMA, "id": entry["id"], "round": round_index, "tick": world.tick,
            "world": world.to_dict(), "tracker": tracker.to_dict()}


def _restore(bundle):
    if bundle.get("schema") != BUNDLE_SCHEMA:
        raise ValueError("not a life8 search bundle")
    return World.from_dict(bundle["world"]), Tracker(data=bundle["tracker"])


def run_segment(entry, ticks, *, start=None, perm_reps=100, seed=0):
    """Run one world for ``ticks`` (from ``start`` bundle or tick 0) and measure it.

    Returns (row, end bundle)."""
    started = time.perf_counter()
    if start is None:
        world = build_world(entry)
        tracker = Tracker(world)
    else:
        world, tracker = _restore(start)
    first = world.tick
    window = first + ticks - ticks // 2
    signals, heard, shares = [], [], []
    agent_ticks = window_agent_ticks = 0
    food0, gain0, births0 = world.ledger["energy_from_food"], world.counters.get("tool_damage_gain", 0.), \
        world.counters.get("births", 0)
    gain_w = food_w = None
    while world.tick < first + ticks and not world.stop_reason:
        if world.tick == window:
            gain_w, food_w = world.counters.get("tool_damage_gain", 0.), world.ledger["energy_from_food"]
        alive = len(world.agents)
        agent_ticks += alive
        events = world.step()
        in_window = world.tick > window
        window_agent_ticks += alive if in_window else 0
        before = len(shares)
        tracker.record(world, events, shares)
        if not in_window:
            del shares[before:]
            continue
        for event in events:
            if event["type"] == "signal":
                signals.append((event["tick"], event["agent"], event["sender"], event["symbol"]))
            elif event["type"] == "heard" and event.get("next_action") is not None:
                heard.append((event["symbol"], event["next_action"]))
    gain_w = world.counters.get("tool_damage_gain", 0.) if gain_w is None else gain_w
    food_w = world.ledger["energy_from_food"] if food_w is None else food_w
    measures = {
        "sender": sender_information(signals, reps=perm_reps, seed=seed, state=sender_state(world.config)),
        "receiver": receiver_information(heard, reps=perm_reps, seed=seed + 1),
        "convention": convention_agreement(signals, tracker.founder, situation=situation_of(world.config),
                                           reps=perm_reps, seed=seed + 2),
        "cooperation": reciprocity(shares, reps=perm_reps, seed=seed + 3),
        "tool_damage_gain_per_agent_tick": round((world.counters.get("tool_damage_gain", 0.) - gain_w)
                                                 / max(1, window_agent_ticks), 6),
        "food_energy_per_agent_tick": round((world.ledger["energy_from_food"] - food_w) / max(1, window_agent_ticks), 6),
    }
    row = {"id": entry["id"], "source": entry["source"], "seed": entry["seed"], "start_tick": first,
           "end_tick": world.tick, "window_start": window, "stop_reason": world.stop_reason,
           "ecology": world.config.ecology_on(), "kin_credit": world.config.kin_credit,
           "predators": world.config.predators, "population": len(world.agents), "mean_population": round(agent_ticks / max(1, world.tick - first), 2),
           "births": world.counters.get("births", 0) - births0, "lineages_alive": len({tracker.founder.get(a)
                                                                                       for a in world.agents}),
           "food_energy_per_agent_tick_all": round((world.ledger["energy_from_food"] - food0) / max(1, agent_ticks), 6),
           "measures": measures, "seconds": round(time.perf_counter() - started, 2)}
    return row, _bundle(entry, world, tracker, None)


def run_branch(bundle, arm, rep, ticks):
    """One paired branch from ``bundle`` with the arm's Config overrides; the world RNG
    and the scramble RNG are reseeded per replicate identically in every arm."""
    data = json.loads(json.dumps(bundle["world"]))
    data["config"].update(ARMS[arm])
    data["config"]["log"] = "compact"
    data["pending_heard"] = []
    world = World.from_dict(data)
    world.rng = random.Random(BRANCH_SALT + rep)
    world.channel_rng = random.Random(BRANCH_SALT * 7 + rep)
    starting, state = set(world.agents), sender_state(world.config)
    start, food, births, agent_ticks, heard, signals = world.tick, world.ledger["energy_from_food"], \
        world.counters.get("births", 0), 0, [], []
    while world.tick < start + ticks and not world.stop_reason:
        agent_ticks += len(world.agents)
        for event in world.step():
            if event["type"] == "heard" and event.get("next_action") is not None:
                heard.append((event["symbol"], event["next_action"]))
            elif event["type"] == "signal":  # kept small: the joint state as a tuple
                signals.append((event["tick"], event["agent"], _bands(event["sender"], state), event["symbol"]))
    return {"arm": arm, "rep": rep, "ticks": world.tick - start, "stop_reason": world.stop_reason,
            "population": len(world.agents), "births": world.counters.get("births", 0) - births,
            "survival": len(starting & set(world.agents)) / max(1, len(starting)),
            "food_energy_per_agent_tick": (world.ledger["energy_from_food"] - food) / max(1, agent_ticks),
            "heard": heard, "signals": signals}


def paired_measures(branches, *, perm_reps=100, seed=0, state=SENDER_STATE):
    """Paired effects (normal minus each other arm), receiver information over the scrambled
    arm, and the sender control: the normal arm's within-sender excess minus that of the
    first measured arm of SENDER_CONTROLS (masked, else scrambled), where content cannot matter."""
    by = {(b["arm"], b["rep"]): b for b in branches}
    reps = sorted({b["rep"] for b in branches})
    out = {}
    for arm in ("masked", "scrambled", "tools_off"):
        both = [r for r in reps if ("normal", r) in by and (arm, r) in by]
        out[f"normal_minus_{arm}"] = {m: paired([by["normal", r][m] - by[arm, r][m] for r in both])
                                      for m in ("food_energy_per_agent_tick", "population", "survival", "births")}
    normal = receiver_information([h for r in reps if ("normal", r) in by for h in by["normal", r]["heard"]],
                                  reps=perm_reps, seed=seed)
    scrambled = receiver_information([h for r in reps if ("scrambled", r) in by for h in by["scrambled", r]["heard"]],
                                     reps=perm_reps, seed=seed)
    over = None if normal["excess_bits"] is None or scrambled["excess_bits"] is None \
        else round(normal["excess_bits"] - scrambled["excess_bits"], 6)
    out["receiver"] = {"normal": normal, "scrambled": scrambled, "excess_over_scrambled_bits": over}
    sender = {}
    for arm in ("normal",) + SENDER_CONTROLS:
        if any((arm, r) in by for r in reps):
            rows = [s for r in reps if (arm, r) in by for s in by[arm, r].get("signals", [])]
            sender[arm] = sender_information(rows, reps=perm_reps, seed=seed + 17, state=state)
    within = {arm: (info.get("within_sender") or {}).get("excess_bits") for arm, info in sender.items()}
    control = next((arm for arm in SENDER_CONTROLS if within.get(arm) is not None), None)
    differences = {f"normal_minus_{arm}_bits": None if within.get("normal") is None or within.get(arm) is None
                   else round(within["normal"] - within[arm], 6) for arm in SENDER_CONTROLS}
    out["sender"] = {**sender, **differences, "control": control,
                     "excess_over_control_bits": differences[f"normal_minus_{control}_bits"] if control else None}
    return out


# ---------------------------------------------------------------- score

def _component(x, scale, p, unpaired=False):
    if x is None:
        return 0.
    factor = .25 if unpaired or p is None or p > ALPHA else 1.
    return min(2., max(0., x) / scale) * factor


def score(row):
    """Composite score and per-component inputs for one world row (documented in the module docstring)."""
    m, paired_ = row["measures"], row.get("paired")
    within = m["sender"].get("within_sender") or {}
    pooled_excess, within_excess = m["sender"]["excess_bits"], within.get("excess_bits")
    sender_x = None if pooled_excess is None or within_excess is None else min(pooled_excess, within_excess)
    sender_p = None if sender_x is None else max(m["sender"]["p_value"], within["p_value"])
    sender_part = _component(sender_x, SCALE["sender"], sender_p)
    if paired_:  # paired rounds: only what the normal arm adds over the masked (or scrambled) control
        over = (paired_.get("sender") or {}).get("excess_over_control_bits")
        sender_part = _component(over, SCALE["sender_over_control"], within.get("p_value"))
    parts = {"sender": sender_part,
             "convention": _component(m["convention"]["difference"], SCALE["convention"], m["convention"]["p_value"]),
             "cooperation": _component(m["cooperation"]["excess"], SCALE["cooperation"], m["cooperation"]["p_value"])}
    if paired_:
        rec = paired_["receiver"]
        parts["receiver"] = _component(rec["excess_over_scrambled_bits"], SCALE["receiver"],
                                       rec["normal"]["p_value"])
        fit = paired_["normal_minus_scrambled"]["food_energy_per_agent_tick"]
        parts["scramble_fitness"] = _component(fit["mean"], SCALE["scramble_fitness"], fit["sign_p"])
        tool = paired_["normal_minus_tools_off"]["food_energy_per_agent_tick"]
        parts["tools"] = _component(tool["mean"], SCALE["tools"], tool["sign_p"])
    else:
        parts["receiver"] = _component(m["receiver"]["excess_bits"], SCALE["receiver"], m["receiver"]["p_value"])
        parts["scramble_fitness"] = 0.
        parts["tools"] = _component(m["tool_damage_gain_per_agent_tick"], SCALE["tools_screen"], None, unpaired=True)
    total = sum(WEIGHTS[k] * v for k, v in parts.items())
    return round(total, 6), {k: round(v, 4) for k, v in parts.items()}


def criteria(row):
    """Strict pass/fail flags; the language ones use discovery.THRESHOLDS' values."""
    c, m, p = CRITERIA, row["measures"], row.get("paired")
    within = m["sender"].get("within_sender") or {}
    sender_ok = within.get("excess_bits") is not None and within["excess_bits"] >= c["sender_excess_bits"] \
        and within["p_value"] <= c["sender_p"]
    if p:  # and the normal arm's within-sender excess must beat the control arm's by sender_over_control_bits
        over = (p.get("sender") or {}).get("excess_over_control_bits")
        sender_ok = sender_ok and over is not None and over >= c["sender_over_control_bits"]
    flags = {"sender_information": sender_ok,
             "convention": m["convention"]["difference"] is not None
             and m["convention"]["difference"] >= c["convention_difference"] and m["convention"]["p_value"] <= c["convention_p"],
             "cooperation": m["cooperation"]["excess"] is not None and m["cooperation"]["shares"] >= c["cooperation_min_shares"]
             and m["cooperation"]["excess"] >= c["cooperation_excess"] and m["cooperation"]["p_value"] <= c["cooperation_p"],
             "receiver_information": None, "fitness_cost_of_scrambling": None, "tool_benefit": None}
    if p:
        rn, over = p["receiver"]["normal"], p["receiver"]["excess_over_scrambled_bits"]
        flags["receiver_information"] = rn["excess_bits"] is not None and rn["excess_bits"] >= c["receiver_excess_bits"] \
            and rn["p_value"] <= c["receiver_p"] and over is not None and over >= c["receiver_over_scrambled_bits"]
        fit = p["normal_minus_scrambled"]["food_energy_per_agent_tick"]
        flags["fitness_cost_of_scrambling"] = fit["n"] >= c["fitness_min_reps"] and fit["mean"] is not None \
            and fit["mean"] > 0 and fit["sign_p"] <= c["fitness_sign_p"]
        tool = p["normal_minus_tools_off"]["food_energy_per_agent_tick"]
        flags["tool_benefit"] = tool["n"] >= c["tool_min_reps"] and tool["mean"] is not None and tool["mean"] > 0 \
            and tool["sign_p"] <= c["tool_sign_p"]
        language = [flags["sender_information"], flags["receiver_information"], flags["fitness_cost_of_scrambling"]]
        flags["language_verdict"] = "all_three" if all(language) else "partial" if any(language) else "null"
    else:
        flags["language_verdict"] = "screen_only"
    return flags


# ---------------------------------------------------------------- jobs

def _segment_job(task):
    entry, out, round_index, ticks, perm_reps, start_path = task
    try:
        start = _read_gz(Path(start_path)) if start_path else None
        row, bundle = run_segment(entry, ticks, start=start, perm_reps=perm_reps, seed=entry["seed"] + 7919 * round_index)
        bundle["round"] = round_index
        _write_gz(_pending(out, round_index, entry["id"]), bundle)
        return row
    except Exception as exc:  # a failing world keeps a row; it is never silently dropped
        return {"id": entry["id"], "source": entry["source"], "seed": entry["seed"], "error": f"{type(exc).__name__}: {exc}"}


def _branch_job(task):
    path, arm, rep, ticks = task
    try:
        return run_branch(_read_gz(Path(path)), arm, rep, ticks)
    except Exception as exc:
        return {"arm": arm, "rep": rep, "error": f"{type(exc).__name__}: {exc}"}


def _pending(out, round_index, identity):
    return Path(out) / "pending" / f"r{round_index}" / f"{identity}.json.gz"


def _checkpoint(out, identity, round_index):
    return Path(out) / "checkpoints" / f"{identity}.r{round_index}.json.gz"


def _map(fn, tasks, workers):
    if workers <= 1 or len(tasks) <= 1:
        for task in tasks:
            yield fn(task)
        return
    with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
        yield from pool.map(fn, tasks)


def output_bytes(out):
    return sum(f.stat().st_size for f in Path(out).rglob("*") if f.is_file())


# ---------------------------------------------------------------- the search

def round_ticks(args, round_index):
    return int(args["ticks"] * args["growth"] ** round_index)


def read_results(out):
    path = Path(out) / "results.jsonl"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _append(out, row):
    with (Path(out) / "results.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
        stream.flush()


def _status(out, meta, **changes):
    meta.update(changes)
    tmp = Path(out) / "search.json.tmp"
    tmp.write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, Path(out) / "search.json")


def _utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def run_search(out, *, worlds=8, rounds=2, ticks=200, keep=.5, workers=1, source="plain", start_seed=0,
               growth=2., reps=6, branch_ticks=200, perm_reps=100, options=None, max_output_mb=None,
               world7_workers=None, plan=None, preset=None, log=print):
    """Run (or resume) a search in ``out``. Returns the search.json metadata.

    ``preset`` (a PRESETS key, e.g. "living") gives Config fields under ``options``.
    ``plan`` (a list from make_plan, e.g. built on another machine) replaces the
    candidate generation; worlds, source, start_seed and options are then taken from it."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "search.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("schema") != SCHEMA:
            raise ValueError("not a life8 search folder")
        args, plan = meta["args"], meta["plan"]
        if max_output_mb is not None:  # a resumed search may get a new output cap
            args["max_output_mb"] = max_output_mb
        meta["status"] = "running"
        log(json.dumps({"resume": str(out), "status": meta.get("status"), "utc": _utc()}))
    else:
        args = {"worlds": worlds, "rounds": rounds, "ticks": ticks, "keep": keep, "source": source,
                "start_seed": start_seed, "growth": growth, "reps": reps, "branch_ticks": branch_ticks,
                "perm_reps": perm_reps, "preset": preset, "options": dict(options or {}),
                "max_output_mb": max_output_mb}
        if not 0 < keep <= 1 or rounds < 1 or ticks < 2 or worlds < 1:
            raise ValueError("need worlds >= 1, rounds >= 1, ticks >= 2 and 0 < keep <= 1")
        if plan is None:
            plan = make_plan(source, worlds, start_seed, args["options"], world7_workers=world7_workers or workers,
                             preset=preset)
        else:
            args.update(worlds=len(plan), source="plan:" + ",".join(sorted({e["source"] for e in plan})),
                        start_seed=min(e["seed"] for e in plan))
        meta = {"schema": SCHEMA, "args": args, "plan": plan, "status": "running", "python": sys.version.split()[0],
                "weights": WEIGHTS, "scale": SCALE, "alpha": ALPHA, "criteria": CRITERIA,
                "interpretation": INTERPRETATION, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
        _status(out, meta)
    cap = None if args.get("max_output_mb") is None else args["max_output_mb"] * 2 ** 20
    entries = {entry["id"]: entry for entry in plan}
    rows = read_results(out)
    done = {(r["round"], r["id"]) for r in rows if r.get("kind") == "world"}
    selections = {r["round"]: r for r in rows if r.get("kind") == "selection"}
    alive = [entry["id"] for entry in plan]
    for round_index in range(args["rounds"]):
        if round_index in selections:
            alive = selections[round_index]["kept"]
            continue
        if not alive:
            break
        if cap is not None:  # projected: current folder plus one end state per world of this round
            sizes = [f.stat().st_size for f in (out / "checkpoints").glob("*.gz")] if (out / "checkpoints").exists() else []
            projected = output_bytes(out) + len(alive) * max(sizes + [2 ** 20])
            if projected > cap:
                _status(out, meta, status="stopped", stop_reason="output_cap", stopped_round=round_index,
                        projected_bytes=projected, cap_bytes=cap)
                log(json.dumps({"stopped": "output_cap", "round": round_index, "projected_bytes": projected,
                                "utc": _utc()}))
                return meta
        length = round_ticks(args, round_index)
        paired_stage = round_index >= 1
        todo = [i for i in alive if (round_index, i) not in done]
        tasks = [(entries[i], str(out), round_index, length, args["perm_reps"],
                  None if round_index == 0 else str(_checkpoint(out, i, round_index - 1))) for i in todo]
        log(json.dumps({"round": round_index, "ticks": length, "worlds": len(alive), "to_run": len(tasks),
                        "utc": _utc()}))
        segment_rows = {}
        for row in _map(_segment_job, tasks, workers):
            segment_rows[row["id"]] = row
            if not paired_stage or "error" in row:
                _finish(out, row, round_index, length, None)
        if paired_stage:
            ready = [i for i in todo if "error" not in segment_rows[i] and not segment_rows[i]["stop_reason"]]
            branch_tasks = [(str(_pending(out, round_index, i)), arm, rep, args["branch_ticks"])
                            for i in ready for rep in range(args["reps"]) for arm in ARMS]
            # Results come in task order, world by world: each world's row is finished as soon
            # as its own branches are in, so only one world's branches are held at a time.
            log(json.dumps({"round": round_index, "branches": len(branch_tasks), "worlds": len(ready), "utc": _utc()}))
            results, per_world = _map(_branch_job, branch_tasks, workers), args["reps"] * len(ARMS)
            for i in todo:
                if "error" in segment_rows[i]:
                    continue
                found = [next(results) for _ in range(per_world)] if i in ready else []
                errors = [b["error"] for b in found if "error" in b]
                good = [b for b in found if "error" not in b]
                segment_rows[i]["paired"] = paired_measures(
                    good, perm_reps=args["perm_reps"], seed=entries[i]["seed"] + 104729 * round_index,
                    state=sender_state(Config.from_dict(entries[i]["config"]))) if good else None
                segment_rows[i]["paired_errors"] = errors
                segment_rows[i]["branch_ticks"], segment_rows[i]["branch_reps"] = args["branch_ticks"], args["reps"]
                _finish(out, segment_rows[i], round_index, length, None)
            assert next(results, None) is None  # every branch was used; this also closes the pool
        rows = read_results(out)
        current = [r for r in rows if r.get("kind") == "world" and r["round"] == round_index and r["id"] in alive]
        eligible = sorted((r for r in current if "error" not in r and not r.get("stop_reason")),
                          key=lambda r: (-r["score"], r["id"]))
        last = round_index == args["rounds"] - 1
        count = max(1, math.ceil(args["keep"] * len(alive))) if eligible else 0
        kept = [r["id"] for r in eligible[:count]]
        cutoff = eligible[count - 1]["score"] if kept else None
        for identity in kept:  # promote; regenerate a lost end state deterministically
            target = _checkpoint(out, identity, round_index)
            if not target.exists():
                pending = _pending(out, round_index, identity)
                if not pending.exists():
                    start = None if round_index == 0 else str(_checkpoint(out, identity, round_index - 1))
                    _segment_job((entries[identity], str(out), round_index, length, args["perm_reps"], start))
                target.parent.mkdir(parents=True, exist_ok=True)
                os.replace(pending, target)
        dropped = [i for i in alive if i not in kept]
        for identity in alive:  # earlier checkpoints of every world in this round, and dropped worlds' end states
            stale = [_pending(out, round_index, identity)] + ([_checkpoint(out, identity, round_index - 1)]
                                                              if round_index else [])
            for path in stale:
                if path.exists():
                    path.unlink()
        shutil.rmtree(Path(out) / "pending" / f"r{round_index}", ignore_errors=True)
        _append(out, {"kind": "selection", "round": round_index, "ticks": length, "final": last, "kept": kept,
                      "dropped": dropped, "cutoff_score": cutoff, "eligible": len(eligible),
                      "stopped_or_failed": [r["id"] for r in current if "error" in r or r.get("stop_reason")]})
        log(json.dumps({"round": round_index, "kept": kept[:10], "n_kept": len(kept), "cutoff": cutoff, "utc": _utc()}))
        alive = kept
    _status(out, meta, status="complete", stop_reason="rounds_done",
            finished=time.strftime("%Y-%m-%dT%H:%M:%S"), output_bytes=output_bytes(out))
    return meta


def _finish(out, row, round_index, length, _):
    row = {"kind": "world", "round": round_index, "round_ticks": length, **row}
    if "error" not in row:
        row["score"], row["components"] = score(row)
        row["criteria"] = criteria(row)
    else:
        row["score"], row["components"], row["criteria"] = 0., None, None
    _append(out, row)


# ---------------------------------------------------------------- report and collect

def report(rows, *, top=10):
    worlds = [r for r in rows if r.get("kind") == "world"]
    selections = [r for r in rows if r.get("kind") == "selection"]
    out = {"interpretation": INTERPRETATION, "world_rows": len(worlds),
           "worlds": len({(r.get("host"), r["id"]) for r in worlds}), "rounds": {}}
    for index in sorted({r["round"] for r in worlds}):
        here = [r for r in worlds if r["round"] == index]
        ok = [r for r in here if "error" not in r]
        flags = Counter()
        for r in ok:
            for name, value in (r.get("criteria") or {}).items():
                if value is True:
                    flags[name] += 1
        scores = sorted(r["score"] for r in ok)
        out["rounds"][index] = {
            "ticks": here[0]["round_ticks"], "worlds": len(here), "errors": len(here) - len(ok),
            "stopped": dict(Counter(r["stop_reason"] for r in ok if r.get("stop_reason"))),
            "score_median": scores[len(scores) // 2] if scores else None, "score_max": scores[-1] if scores else None,
            "passed": dict(flags), "language_verdicts": dict(Counter((r.get("criteria") or {}).get("language_verdict")
                                                                      for r in ok)),
            "kept": sum(len(s["kept"]) for s in selections if s["round"] == index),
            "mean_seconds": round(sum(r["seconds"] for r in ok) / len(ok), 2) if ok else None}
    final = max((r["round"] for r in worlds), default=None)
    best = sorted((r for r in worlds if r["round"] == final and "error" not in r), key=lambda r: (-r["score"], r["id"]))
    out["best"] = [{"host": r.get("host"), "id": r["id"], "round": r["round"], "end_tick": r["end_tick"],
                    "score": r["score"], "components": r["components"], "criteria": r["criteria"],
                    "population": r["population"], "sender_excess_bits": r["measures"]["sender"]["excess_bits"],
                    "sender_within_excess_bits": (r["measures"]["sender"].get("within_sender") or {}).get("excess_bits"),
                    "sender_over_control_bits": ((r.get("paired") or {}).get("sender") or {}).get(
                        "excess_over_control_bits"),
                    "convention_difference": r["measures"]["convention"]["difference"],
                    "reciprocity_excess": r["measures"]["cooperation"]["excess"],
                    "shares": r["measures"]["cooperation"]["shares"]} for r in best[:top]]
    return out


def collect(target, sources):
    """Merge results.jsonl of several search folders (one per host) into ``target``,
    tagging each row with its folder name as ``host``."""
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    merged, counts = [], {}
    for folder in map(Path, sources):
        rows = read_results(folder)
        meta_path = folder / "search.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        host = folder.name
        counts[host] = {"rows": len(rows), "status": meta.get("status"), "stop_reason": meta.get("stop_reason"),
                        "args": meta.get("args")}
        merged += [{**row, "host": host} for row in rows]
    with (target / "results.jsonl").open("w", encoding="utf-8") as stream:
        for row in merged:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    summary = {"sources": counts, **report(merged)}
    (target / "report.json").write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run or resume a search")
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--worlds", type=int, default=8)
    run.add_argument("--rounds", type=int, default=2)
    run.add_argument("--ticks", type=int, default=200, help="ticks of round 0; round r runs ticks*growth**r")
    run.add_argument("--keep", type=float, default=.5)
    run.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    run.add_argument("--source", choices=SOURCES, default="plain")
    run.add_argument("--preset", choices=sorted(PRESETS), help="Config preset under the --option values "
                     "('living' = config.LIVING, the worlds where signalling can pay)")
    run.add_argument("--start-seed", type=int, default=0)
    run.add_argument("--growth", type=float, default=2.)
    run.add_argument("--reps", type=int, default=6, help="paired branch replicates per arm (rounds >= 1)")
    run.add_argument("--branch-ticks", type=int, default=200)
    run.add_argument("--perm-reps", type=int, default=100)
    run.add_argument("--max-output-mb", type=float, help="stop (stop_reason output_cap, resumable) before a round whose projected folder size exceeds this; on resume it replaces the saved cap")
    run.add_argument("--option", type=parse_option, action="append", default=[], help="Config field key=value (JSON value)")
    run.add_argument("--plan", type=Path, help="candidate list written by 'plan' (needs no numpy or world7 on this host)")
    pln = sub.add_parser("plan", help="write the candidate list (bridged worlds need haishool.evo and numpy here)")
    pln.add_argument("--out", type=Path, required=True)
    pln.add_argument("--worlds", type=int, default=8)
    pln.add_argument("--source", choices=SOURCES, default="plain")
    pln.add_argument("--preset", choices=sorted(PRESETS))
    pln.add_argument("--start-seed", type=int, default=0)
    pln.add_argument("--workers", type=int, default=1)
    pln.add_argument("--option", type=parse_option, action="append", default=[])
    spl = sub.add_parser("split", help="cut a plan into consecutive parts, one per host")
    spl.add_argument("--plan", type=Path, required=True)
    spl.add_argument("--out-dir", type=Path, required=True)
    spl.add_argument("parts", nargs="+", help="NAME=COUNT (writes OUT_DIR/NAME.plan.json)")
    rep = sub.add_parser("report", help="summarise a search folder")
    rep.add_argument("--out", type=Path, required=True)
    rep.add_argument("--top", type=int, default=10)
    col = sub.add_parser("collect", help="merge several search folders")
    col.add_argument("--out", type=Path, required=True)
    col.add_argument("folders", nargs="+", type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        plan_file = json.loads(args.plan.read_text(encoding="utf-8")) if args.plan else {}
        meta = run_search(args.out, worlds=args.worlds, rounds=args.rounds, ticks=args.ticks, keep=args.keep,
                          workers=args.workers, source=args.source, start_seed=args.start_seed, growth=args.growth,
                          reps=args.reps, branch_ticks=args.branch_ticks, perm_reps=args.perm_reps,
                          options=dict(args.option) or plan_file.get("options"), max_output_mb=args.max_output_mb,
                          plan=plan_file.get("plan"), preset=args.preset or plan_file.get("preset"))
        print(json.dumps({k: meta.get(k) for k in ("status", "stop_reason", "output_bytes")}))
        print(json.dumps(report(read_results(args.out), top=5), indent=1))
    elif args.command == "plan":
        plan = make_plan(args.source, args.worlds, args.start_seed, dict(args.option), world7_workers=args.workers,
                         preset=args.preset)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"schema": SCHEMA + "-plan", "source": args.source, "preset": args.preset,
                                        "options": dict(args.option), "plan": plan}) + chr(10), encoding="utf-8")
        print(json.dumps({"worlds": len(plan), "first": plan[0]["id"], "last": plan[-1]["id"],
                          "sources": dict(Counter(e["source"] for e in plan))}))
    elif args.command == "split":
        whole = json.loads(args.plan.read_text(encoding="utf-8"))
        names, counts = zip(*((name, int(count)) for name, _, count in (p.partition("=") for p in args.parts)))
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for name, part in zip(names, split_plan(whole["plan"], list(counts))):
            (args.out_dir / f"{name}.plan.json").write_text(json.dumps({**whole, "plan": part}) + chr(10),
                                                            encoding="utf-8")
            print(json.dumps({"part": name, "worlds": len(part), "first": part[0]["id"], "last": part[-1]["id"],
                              "sources": dict(Counter(e["source"] for e in part))}))
    elif args.command == "report":
        print(json.dumps(report(read_results(args.out), top=args.top), indent=1))
    else:
        print(json.dumps(collect(args.out, args.folders), indent=1))


if __name__ == "__main__":
    main()
