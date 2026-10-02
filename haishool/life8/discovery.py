"""Signalling discovery measurements for life8 compact-log runs (analysis code; numpy allowed).

Three questions, each answered with an intervention on the same seeds:

1. Sender information: does a sender's symbol depend on its own local state?
   I(symbol; sender state bands) from compact ``signal`` events, minus a permutation
   null, with a permutation p-value. The criterion uses the within-sender null
   (symbols shuffled only among one individual's calls), because individual symbol
   habits that correlate with state pass the pooled test (round-8 critic B2), and, when
   a control run is given (``discover``: the scrambled run, where content cannot pay),
   the within-sender excess must beat the control's by ``sender_over_control_bits``.
2. Receiver information: does a receiver's next action depend on the heard symbol
   more than when the content is destroyed? I(heard symbol; next action) from
   compact ``heard`` events in the normal run, against a paired run of the same seed
   with ``Config(channel="scrambled")``. Location and neighbour effects that create
   symbol/action dependencies without content are present in both runs, so only the
   difference counts.
3. Fitness effect: from one common checkpoint, paired branches run with the normal and
   the scrambled channel, using the same reseeded world RNG per replicate. Effects are
   normal minus scrambled: final population, food energy per agent-tick and survival
   of the individuals alive at the branch point.

A single verdict combines the three with the explicit thresholds in THRESHOLDS.
"all_three" means: symbols carry information about the sender's state, receivers act
on that content, and destroying the content costs measurable fitness in this toy
world. It is not evidence of language, and the numbers are toy-world outcomes, not
probabilities of anything real.

A positive and a negative control use the same functions: ``signalling_game`` is a
two-player game played by life8 learners in which the sender either shares the
receiver's payoff (a credit route for informative sending exists) or only pays for
the call (life8's rule). The first should reach about 0.5 bits or more of sender
information, the second should stay low (review: 0.55-0.66 vs 0.005-0.098 bits).

usage: python -m haishool.life8.discovery --seeds 1-6 --ticks 1500 [--branch-at 1000]
           [--branch-ticks 200] [--reps 8] [--jobs 6] [--config JSON] [--out FILE]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from . import learning
from .config import Config
from .world import World

# Joint sender state used for the main sender test (the review's "joint state").
SENDER_STATE = ("food_near", "food_seen", "energy_band", "neighbor_near")
# Per-band breakdown reported next to the joint test.
SENDER_BANDS = ("food_near", "food_seen", "food_direction", "energy_band", "health_band", "neighbor_near",
                "object_near", "tool_available")
KEPT_EVENTS = ("signal", "heard", "birth", "death", "capacity_pause")
SIGNALS = ("signal_0", "signal_1", "signal_2", "signal_3")
BRANCH_SALT = 0x51A7E5  # replicate r reseeds the world RNG with BRANCH_SALT + r in every arm
THRESHOLDS = {
    # sender: within-sender I(symbol; joint state) above its permutation null, and
    # (with a control run) at least sender_over_control_bits above the control's
    "sender_excess_bits": 0.05,
    "sender_p": 0.01,
    "sender_over_control_bits": 0.02,
    # receiver: I(heard; next action) above its own permutation null, and above the scrambled run
    "receiver_excess_bits": 0.02,
    "receiver_p": 0.01,
    "receiver_over_scrambled_bits": 0.02,
    # fitness: normal minus scrambled food energy per agent-tick over paired branch replicates
    "fitness_min_reps": 6,
    "fitness_sign_p": 0.05,
}
INTERPRETATION = ("Toy-world measurement. 'all_three' means signals carry sender-state information, receivers act "
                  "on their content and scrambling the content costs fitness here; it is not evidence of language.")


# ---------------------------------------------------------------- information measures

def _codes(values):
    table = {}
    return np.fromiter((table.setdefault(json.dumps(v, sort_keys=True), len(table)) for v in values),
                       dtype=np.int64, count=len(values)), len(table)


def _mi_codes(x, nx, y, ny):
    n = len(x)
    if not n:
        return 0.
    joint = np.bincount(x * ny + y, minlength=nx * ny).reshape(nx, ny).astype(float)
    px, py = joint.sum(1), joint.sum(0)
    rows, cols = np.nonzero(joint)
    c = joint[rows, cols]
    return float(np.sum(c / n * np.log2(c * n / (px[rows] * py[cols]))))


def mutual_information(xs, ys):
    """Plug-in mutual information in bits between two equally long label sequences."""
    if len(xs) != len(ys):
        raise ValueError("label sequences differ in length")
    x, nx = _codes(xs)
    y, ny = _codes(ys)
    return _mi_codes(x, nx, y, ny)


def permutation_test(xs, ys, *, reps=200, seed=0, groups=None):
    """I(X;Y) against shuffles of Y (within each group when ``groups`` is given).

    excess_bits = observed minus the null mean; p = (1 + #null >= observed) / (reps + 1).
    """
    n = len(xs)
    if n != len(ys) or (groups is not None and len(groups) != n):
        raise ValueError("label sequences differ in length")
    if n == 0:
        return {"n": 0, "mi_bits": None, "null_mean": None, "null_p95": None, "excess_bits": None, "p_value": None}
    x, nx = _codes(xs)
    y, ny = _codes(ys)
    observed = _mi_codes(x, nx, y, ny)
    rng = np.random.default_rng(seed)
    if groups is not None:
        g, _ = _codes(groups)
        slots = np.argsort(g, kind="stable")  # each group's positions, groups in code order
    null = np.empty(reps)
    for index in range(reps):
        if groups is None:
            shuffled = y[rng.permutation(n)]
        else:
            # Same group blocks in random order within each block: y is permuted only inside groups.
            order = np.lexsort((rng.random(n), g))
            shuffled = np.empty(n, dtype=np.int64)
            shuffled[slots] = y[order]
        null[index] = _mi_codes(x, nx, shuffled, ny)
    null.sort()
    return {"n": n, "mi_bits": round(observed, 6), "null_mean": round(float(null.mean()), 6),
            "null_p95": round(float(null[max(0, int(.95 * reps) - 1)]), 6),
            "excess_bits": round(observed - float(null.mean()), 6),
            "p_value": round((1 + int(np.sum(null >= observed - 1e-12))) / (reps + 1), 6)}


def _entropy(values):
    counts = Counter(values)
    total = sum(counts.values())
    return -sum(c / total * math.log2(c / total) for c in counts.values()) if total else 0.


# ---------------------------------------------------------------- rows from compact events

def _sender_view(bands):
    view = dict(bands)
    view["food_seen"] = bands.get("food_direction") not in (None, "none")
    return view


def sender_rows(events, *, start=0, stop=None):
    """(tick, sender, state dict, symbol) for every call sent, from compact ``signal`` events."""
    rows = []
    for event in events:
        if event.get("type") == "signal" and event["tick"] >= start and (stop is None or event["tick"] < stop):
            rows.append((event["tick"], event["agent"], _sender_view(event["sender"]), event["symbol"]))
    return rows


def receiver_rows(events, *, start=0, stop=None):
    """(tick, receiver, heard symbol, next action) for each delivery whose receiver chose again."""
    return [(e["tick"], e["agent"], e["symbol"], e["next_action"]) for e in events
            if e.get("type") == "heard" and e.get("next_action") is not None
            and e["tick"] >= start and (stop is None or e["tick"] < stop)]


def sender_information(events, *, state=SENDER_STATE, bands=SENDER_BANDS, reps=200, seed=0, start=0, stop=None):
    """Pooled I(symbol; joint sender state) against a permutation null, plus per-band tests
    and a within-sender test (symbols shuffled only among one individual's calls)."""
    rows = sender_rows(events, start=start, stop=stop)
    symbols = [row[3] for row in rows]
    joint = [tuple(row[2].get(name) for name in state) for row in rows]
    result = {"state": list(state), "signals": len(rows), "senders": len({row[1] for row in rows}),
              "symbol_entropy_bits": round(_entropy(symbols), 4),
              "joint": permutation_test(joint, symbols, reps=reps, seed=seed),
              "within_sender": permutation_test(joint, symbols, reps=reps, seed=seed + 1,
                                                groups=[row[1] for row in rows]),
              "bands": {}}
    for offset, name in enumerate(bands):
        result["bands"][name] = permutation_test([row[2].get(name) for row in rows], symbols,
                                                 reps=max(50, reps // 4), seed=seed + 2 + offset)
    return result


def receiver_information(events, *, reps=200, seed=0, start=0, stop=None):
    rows = receiver_rows(events, start=start, stop=stop)
    return {"deliveries": len(rows), "receivers": len({row[1] for row in rows}),
            "pooled": permutation_test([row[2] for row in rows], [row[3] for row in rows], reps=reps, seed=seed),
            "within_receiver": permutation_test([row[2] for row in rows], [row[3] for row in rows], reps=reps,
                                                seed=seed + 1, groups=[row[1] for row in rows])}


def receiver_contrast(normal_events, scrambled_events, *, reps=200, seed=0, start=0, stop=None):
    """Receiver information in the normal run minus the same in the paired scrambled run."""
    normal = receiver_information(normal_events, reps=reps, seed=seed, start=start, stop=stop)
    scrambled = receiver_information(scrambled_events, reps=reps, seed=seed, start=start, stop=stop)
    a, b = normal["pooled"]["excess_bits"], scrambled["pooled"]["excess_bits"]
    return {"normal": normal, "scrambled": scrambled,
            "excess_over_scrambled_bits": None if a is None or b is None else round(a - b, 6)}


# ---------------------------------------------------------------- running worlds

def _config(overrides=None, **forced):
    values = dict(overrides or {})
    values.update(forced)
    return Config(**values)


def run_events(seed, config=None, ticks=600, *, channel="normal", keep_state_at=None):
    """Run a compact-log world and keep only signal/heard/birth/death/capacity events.

    Returns (events, world, state_at) where state_at is world.to_dict() at tick
    ``keep_state_at`` (None if not requested or not reached)."""
    world = World.create(seed=seed, config=_config(config, log="compact", channel=channel))
    events, state_at = [], None
    while world.tick < ticks and not world.stop_reason:
        if keep_state_at is not None and world.tick == keep_state_at:
            state_at = world.to_dict()
        events.extend(e for e in world.step() if e["type"] in KEPT_EVENTS)
    if keep_state_at is not None and world.tick == keep_state_at and state_at is None:
        state_at = world.to_dict()
    return events, world, state_at


def load_events(run_dir):
    """Events of a life8 run directory written with --log compact (full logs are refused)."""
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("config", {}).get("log") != "compact":
        raise ValueError("discovery measurements need a compact-log run (--log compact)")
    with (run_dir / "events.jsonl").open(encoding="utf-8") as stream:
        return [event for event in map(json.loads, stream) if event.get("type") in KEPT_EVENTS]


def branch(state, channel, rep, ticks):
    """Continue ``state`` for ``ticks`` with the given channel; the world RNG (and the
    scramble RNG) are reseeded per replicate identically in every arm."""
    data = json.loads(json.dumps(state))
    data["config"]["channel"] = channel
    data["config"]["log"] = "compact"
    data["pending_heard"] = []
    world = World.from_dict(data)
    world.rng = random.Random(BRANCH_SALT + rep)
    world.channel_rng = random.Random(BRANCH_SALT * 7 + rep)
    starting = set(world.agents)
    start_tick, food = world.tick, world.ledger["energy_from_food"]
    births, agent_ticks = world.counters.get("births", 0), 0
    while world.tick < start_tick + ticks and not world.stop_reason:
        agent_ticks += len(world.agents)
        world.step()
    return {"channel": channel, "rep": rep, "ticks": world.tick - start_tick, "stop_reason": world.stop_reason,
            "start_population": len(starting), "population": len(world.agents),
            "births": world.counters.get("births", 0) - births,
            "food_energy_per_agent_tick": (world.ledger["energy_from_food"] - food) / max(1, agent_ticks),
            "survival": len(starting & set(world.agents)) / max(1, len(starting)),
            "signals_sent": world.counters.get("signals_sent", 0)}


def sign_test(differences):
    """Two-sided exact binomial sign test over nonzero paired differences."""
    wins = sum(d > 0 for d in differences)
    losses = sum(d < 0 for d in differences)
    n = wins + losses
    if not n:
        return 1.
    k = max(wins, losses)
    return min(1., 2 * sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n)


def _paired(values):
    n = len(values)
    mean = sum(values) / n if n else None
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1)) if n > 1 else None
    return {"n": n, "mean": None if mean is None else round(mean, 6), "sd": None if sd is None else round(sd, 6),
            "positive": sum(v > 0 for v in values), "negative": sum(v < 0 for v in values),
            "sign_p": round(sign_test(values), 6)}


def branch_fitness(state, *, ticks=200, reps=8, arms=("normal", "scrambled"), jobs=1):
    """Paired branches from one state. Differences are normal minus each other arm."""
    tasks = [(state, arm, rep, ticks) for rep in range(reps) for arm in arms]
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            rows = list(pool.map(_branch_job, tasks))
    else:
        rows = [branch(*task) for task in tasks]
    result = {"branch_tick": state["tick"], "ticks": ticks, "reps": reps, "runs": rows, "effects": {}}
    by = {(row["channel"], row["rep"]): row for row in rows}
    for arm in arms[1:]:
        result["effects"][f"normal_minus_{arm}"] = {
            metric: _paired([by["normal", rep][metric] - by[arm, rep][metric] for rep in range(reps)])
            for metric in ("population", "food_energy_per_agent_tick", "survival", "births")}
    return result


def _branch_job(task):
    return branch(*task)


# ---------------------------------------------------------------- verdict

def _within_excess(sender):
    return ((sender or {}).get("within_sender") or {}).get("excess_bits")


def verdict(sender, receiver, fitness, thresholds=None, sender_control=None):
    """One discovery verdict from the three measurements with explicit thresholds.

    The sender criterion uses the within-sender test (``sender["within_sender"]``); with
    ``sender_control`` (sender_information of a run where symbol content cannot matter,
    e.g. the scrambled or masked channel) its excess must also beat the control's."""
    t = {**THRESHOLDS, **(thresholds or {})}
    within = sender.get("within_sender") or {}
    s_ok = within.get("excess_bits") is not None and within["excess_bits"] >= t["sender_excess_bits"] \
        and within["p_value"] <= t["sender_p"]
    control = _within_excess(sender_control) if sender_control is not None else None
    over_control = None if control is None or within.get("excess_bits") is None \
        else round(within["excess_bits"] - control, 6)
    if sender_control is not None:
        s_ok = s_ok and over_control is not None and over_control >= t["sender_over_control_bits"]
    pooled = receiver["normal"]["pooled"]
    over = receiver["excess_over_scrambled_bits"]
    r_ok = pooled["excess_bits"] is not None and pooled["excess_bits"] >= t["receiver_excess_bits"] \
        and pooled["p_value"] <= t["receiver_p"] and over is not None and over >= t["receiver_over_scrambled_bits"]
    effect = (fitness or {}).get("effects", {}).get("normal_minus_scrambled", {}).get("food_energy_per_agent_tick")
    f_ok = bool(effect) and effect["n"] >= t["fitness_min_reps"] and effect["mean"] > 0 \
        and effect["sign_p"] <= t["fitness_sign_p"]
    criteria = {
        "sender_information": {"passed": s_ok, "within_sender_excess_bits": within.get("excess_bits"),
                               "within_sender_p_value": within.get("p_value"),
                               "pooled_excess_bits": (sender.get("joint") or {}).get("excess_bits"),
                               "excess_over_control_bits": over_control,
                               "needs": f"within-sender excess >= {t['sender_excess_bits']} bits and p <= "
                                        f"{t['sender_p']}" + ("" if sender_control is None else
                                                              f", and >= {t['sender_over_control_bits']} bits "
                                                              "above the control run's")},
        "receiver_information": {"passed": r_ok, "excess_bits": pooled["excess_bits"], "p_value": pooled["p_value"],
                                 "excess_over_scrambled_bits": over,
                                 "needs": f"excess >= {t['receiver_excess_bits']} bits, p <= {t['receiver_p']} and "
                                          f">= {t['receiver_over_scrambled_bits']} bits above the scrambled run"},
        "fitness_cost_of_scrambling": {"passed": f_ok, "food_energy_per_agent_tick": effect,
                                       "needs": f"normal - scrambled food energy per agent-tick > 0 over >= "
                                                f"{t['fitness_min_reps']} paired branches, sign test p <= {t['fitness_sign_p']}"}}
    passed = sum(c["passed"] for c in criteria.values())
    return {"verdict": "all_three" if passed == 3 else "partial" if passed else "null",
            "criteria": criteria, "thresholds": t, "interpretation": INTERPRETATION}


def discover(seed, config=None, ticks=1500, *, branch_at=None, branch_ticks=200, reps=8, window_start=None,
             perm_reps=200, jobs=1):
    """Normal run, paired scrambled run (same seed) and paired branches from the normal run.

    Information is measured over ticks [window_start, ticks) (default: the last third)."""
    started = time.perf_counter()
    branch_at = ticks * 2 // 3 if branch_at is None else branch_at
    window_start = ticks * 2 // 3 if window_start is None else window_start
    normal, world, state = run_events(seed, config, ticks, keep_state_at=branch_at)
    scrambled, scrambled_world, _ = run_events(seed, config, ticks, channel="scrambled")
    sender = sender_information(normal, reps=perm_reps, seed=seed, start=window_start)
    sender_scrambled = sender_information(scrambled, reps=perm_reps, seed=seed, start=window_start)
    receiver = receiver_contrast(normal, scrambled, reps=perm_reps, seed=seed, start=window_start)
    fitness = branch_fitness(state, ticks=branch_ticks, reps=reps, jobs=jobs) if state and reps else None
    result = verdict(sender, receiver, fitness, sender_control=sender_scrambled)
    return {"seed": seed, "config_overrides": config or {}, "ticks": ticks, "window_start": window_start,
            "normal": {"ticks_run": world.tick, "stop_reason": world.stop_reason or "step_budget",
                       "population": len(world.agents), "births": world.counters.get("births", 0),
                       "signals_sent": world.counters.get("signals_sent", 0)},
            "scrambled": {"ticks_run": scrambled_world.tick, "stop_reason": scrambled_world.stop_reason or "step_budget",
                          "population": len(scrambled_world.agents),
                          "births": scrambled_world.counters.get("births", 0)},
            "sender": sender, "sender_scrambled_run": sender_scrambled, "receiver": receiver, "fitness": fitness,
            **result, "seconds": round(time.perf_counter() - started, 1)}


def _discover_job(args):
    seed, config, ticks, branch_at, branch_ticks, reps, perm_reps = args
    return discover(seed, config, ticks, branch_at=branch_at, branch_ticks=branch_ticks, reps=reps,
                    perm_reps=perm_reps)


def measure(seeds, config=None, ticks=1500, *, branch_at=None, branch_ticks=200, reps=8, perm_reps=200, jobs=6):
    """Many seeds; per-seed verdicts plus sign tests across seeds for each criterion."""
    tasks = [(seed, config, ticks, branch_at, branch_ticks, reps, perm_reps) for seed in seeds]
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            rows = list(pool.map(_discover_job, tasks))
    else:
        rows = [_discover_job(task) for task in tasks]
    def values(path):
        out = []
        for row in rows:
            value = row
            for key in path:
                value = None if value is None else value.get(key)
            if value is not None:
                out.append(value)
        return out
    across = {
        "sender_excess_bits": _paired(values(("sender", "joint", "excess_bits"))),
        "sender_within_excess_bits": _paired(values(("sender", "within_sender", "excess_bits"))),
        "receiver_excess_over_scrambled_bits": _paired(values(("receiver", "excess_over_scrambled_bits"))),
        "fitness_food_energy_per_agent_tick": _paired(values(("fitness", "effects", "normal_minus_scrambled",
                                                              "food_energy_per_agent_tick", "mean"))),
        "fitness_population": _paired(values(("fitness", "effects", "normal_minus_scrambled", "population", "mean")))}
    return {"measurement": "signalling discovery: sender information, receiver information vs scrambled, "
                           "fitness cost of scrambling", "seeds": list(seeds), "ticks": ticks,
            "config_overrides": config or {}, "verdicts": dict(Counter(row["verdict"] for row in rows)),
            "seeds_passing": {name: sum(row["criteria"][name]["passed"] for row in rows)
                              for name in ("sender_information", "receiver_information", "fitness_cost_of_scrambling")},
            "across_seeds": across, "thresholds": THRESHOLDS, "interpretation": INTERPRETATION, "runs": rows}


# ---------------------------------------------------------------- controls: a two-player signalling game

def signalling_game(seed, payoff="shared", channel="normal", *, rounds=4000, population=8, cost=.33,
                    window=1000, epsilon=None, profile=None, cue="object_near"):
    """Two-player game with life8 learners (unchanged learning.py), as in the review's probe.

    A random sender perceives which side (east or west) pays and sends one of 4 symbols;
    a random receiver (who perceives nothing) hears it and moves east or west. Success
    pays the receiver 1. ``cue`` is the sender's observation bin that carries the side:
    "object_near" (default, True for east) or "food_direction" (the review's framing,
    "e"/"w"). An egocentric learner (learning profile "multistep") rotates every
    direction bin so that visible food is ahead, so with cue="food_direction" both sides
    look the same to it and no convention can form; the default cue is frame-free.
    payoff="shared": the sender also gets the receiver's payoff (a credit route exists).
    payoff="individual": the sender only pays the call cost (life8's actual rule).
    channel="scrambled": the receiver hears a uniform random symbol from a separate RNG.
    ``profile`` names a learning.PROFILES entry (e.g. "one_step", the review's rule);
    None uses the learner's default. ``epsilon`` overrides the exploration rate.
    Returns compact-style ``signal`` and ``heard`` events for the last ``window`` rounds.
    """
    if payoff not in ("shared", "individual") or channel not in ("normal", "scrambled"):
        raise ValueError("unknown payoff or channel")
    rng, scramble = random.Random(seed), random.Random(seed + 999)
    options = {} if epsilon is None else {"epsilon": epsilon}
    if profile is None:
        agents = [learning.new_controller(**options) for _ in range(population)]
    else:
        with learning.use_profile(profile):
            agents = [learning.new_controller(**options) for _ in range(population)]
    moves = ("move_e", "move_w")
    events, successes = [], []
    for tick in range(rounds):
        s, r = rng.sample(range(population), 2)
        side = rng.choice("ew")
        if cue == "food_direction":
            seen = {"food_direction": side, "food_near": False}
        else:
            seen = {"food_direction": "none", "food_near": False, cue: side == "e"}
        call = learning.choose_action(agents[s], seen, rng, actions=SIGNALS)
        symbol = int(call[-1])
        heard = symbol if channel == "normal" else scramble.randrange(4)
        listening = {"food_direction": "none", "food_near": False, "messages": [{"symbol": heard, "age": 0}]}
        move = learning.choose_action(agents[r], listening, rng, actions=moves)
        success = move == "move_" + side
        receiver_reward = 1. if success else 0.
        sender_reward = (receiver_reward if payoff == "shared" else 0.) - cost
        learning.learn(agents[s], seen, call, sender_reward, seen, terminal=True, actions=SIGNALS)
        learning.learn(agents[r], listening, move, receiver_reward, listening, terminal=True, actions=moves)
        if tick >= rounds - window:
            events.append({"tick": tick, "type": "signal", "agent": s + 1, "symbol": symbol,
                           "sender": {"side": side, **seen}})
            events.append({"tick": tick, "type": "heard", "agent": r + 1, "sender": s + 1, "symbol": heard,
                           "sent_symbol": symbol, "next_action": move})
            successes.append(success)
    return {"seed": seed, "payoff": payoff, "channel": channel, "rounds": rounds, "profile": profile, "cue": cue,
            "events": events,
            "success_late": sum(successes) / len(successes) if successes else None}


def game_report(seeds=range(1, 7), rounds=4000, perm_reps=100, profile=None, cue="object_near"):
    rows = []
    for payoff in ("shared", "individual"):
        for seed in seeds:
            normal = signalling_game(seed, payoff, rounds=rounds, profile=profile, cue=cue)
            scrambled = signalling_game(seed, payoff, "scrambled", rounds=rounds, profile=profile, cue=cue)
            sender = sender_information(normal["events"], state=("side",), bands=(), reps=perm_reps, seed=seed)
            receiver = receiver_contrast(normal["events"], scrambled["events"], reps=perm_reps, seed=seed)
            rows.append({"payoff": payoff, "seed": seed, "profile": profile, "cue": cue, "success_late": round(normal["success_late"], 3),
                         "success_late_scrambled": round(scrambled["success_late"], 3),
                         "sender_mi_bits": sender["joint"]["mi_bits"], "sender_excess_bits": sender["joint"]["excess_bits"],
                         "receiver_mi_bits": receiver["normal"]["pooled"]["mi_bits"],
                         "receiver_excess_over_scrambled_bits": receiver["excess_over_scrambled_bits"]})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", default="1-6")
    parser.add_argument("--ticks", type=int, default=1500)
    parser.add_argument("--branch-at", type=int)
    parser.add_argument("--branch-ticks", type=int, default=200)
    parser.add_argument("--reps", type=int, default=8)
    parser.add_argument("--perm-reps", type=int, default=200)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--config", default="{}", help="JSON Config overrides, e.g. '{\"food_patches\": 30}'")
    parser.add_argument("--game", action="store_true", help="run only the signalling-game controls")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    low, _, high = args.seeds.partition("-")
    seeds = range(int(low), int(high or low) + 1)
    if args.game:
        report = {"measurement": "signalling-game controls (life8 learner)",
                  "runs": game_report(seeds) + game_report(seeds, profile="one_step")
                          + game_report(seeds, cue="food_direction") + game_report(seeds, profile="one_step",
                                                                                    cue="food_direction")}
        summary = report["runs"]
    else:
        report = measure(seeds, json.loads(args.config), args.ticks, branch_at=args.branch_at,
                         branch_ticks=args.branch_ticks, reps=args.reps, perm_reps=args.perm_reps, jobs=args.jobs)
        summary = {k: report[k] for k in ("verdicts", "seeds_passing", "across_seeds")}
    text = json.dumps(report, indent=1)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
