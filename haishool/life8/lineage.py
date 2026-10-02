"""Lineage measurements for life8 runs (analysis code; numpy allowed).

* ``LineageTable``: a persistent record of every individual a run has had: founder,
  parent, generation depth, birth and death tick, anatomy at birth and a set of
  neutral marker genes. The engine deletes dead organisms (and with them their
  ``parent_id``); this table keeps them, so parent chains of the living reach back
  to the founders. It is rebuilt from a fresh world plus the birth/death events of
  each step (compact logs keep both), and it serializes to JSON next to a checkpoint.
* Neutral markers: each founder gets ``markers`` values drawn like its anatomy genes
  (uniform 0.75-1.25); a child copies its parent's markers and mutates each with the
  world's mutation_rate and mutation_scale, clamped like the genes. They come from
  their own string-seeded RNG and act on nothing, so their drift is the baseline
  against which selection on a real gene is judged (``trait_selection``).
* ``convention_agreement``: does a founder lineage share a symbol-to-situation mapping?
  For each sender, its distribution of symbols in each situation (from compact
  ``signal`` events); a pair's agreement is the probability that both pick the same
  symbol in a situation both used, averaged over those situations. Within-lineage
  pairs against across-lineage pairs, with a null that shuffles lineage labels among
  senders. A uniform random sender pair agrees with probability 0.25. The tested
  difference subtracts each pair's situation-blind agreement first, so a shared or
  inherited symbol habit without a situation mapping is not a convention (critic F2).
* ``mapping_persistence``: for each founder lineage that outlived its founder, the
  lineage's pooled mapping before the founder died against the mapping its
  descendants used afterwards, with the information I(symbol; situation) on each side
  (a "mapping" with no information is not a convention) and a cross-lineage control.

None of these numbers is evidence of language; they are toy-world measurements.
"""
from __future__ import annotations

import math
import random
import argparse
import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .discovery import permutation_test, sender_rows

GENES = ("body_mass", "speed", "sensing", "manipulation", "metabolism")
SITUATION = ("food_seen", "energy_band")
SCHEMA = "life8-lineage-table-v1"


class LineageTable:
    """Every individual of one run, alive or dead, with founder and parent chain."""

    def __init__(self, seed, *, markers=8, mutation_rate=.08, mutation_scale=.12):
        self.seed, self.markers = seed, markers
        self.mutation_rate, self.mutation_scale = mutation_rate, mutation_scale
        self.rows = {}

    # ------------------------------------------------------------ building
    @classmethod
    def from_world(cls, world, *, markers=8):
        """Start a table at ``world``. Agents without a parent are founders; an agent whose
        parent is not alive here (a table started mid-run) is treated as a root, flagged
        ``root_known=False``."""
        table = cls(world.seed, markers=markers, mutation_rate=world.config.mutation_rate,
                    mutation_scale=world.config.mutation_scale)
        for agent in sorted(world.agents.values(), key=lambda a: a.id):
            parent = table.rows.get(agent.parent_id) if agent.parent_id is not None else None
            if parent is None:
                table._add_root(agent, known=agent.parent_id is None)
            else:
                table._add_child(agent.id, parent, agent.born_tick, agent.anatomy.to_dict())
        return table

    def _rng(self, identity):
        return random.Random(f"life8-lineage-markers:{self.seed}:{identity}")

    def _add_root(self, agent, known=True):
        rng = self._rng(agent.id)
        self.rows[agent.id] = {"id": agent.id, "parent": agent.parent_id, "founder": agent.id, "depth": 0,
                               "root_known": known, "born_tick": agent.born_tick, "death_tick": None,
                               "death_reason": None, "anatomy": agent.anatomy.to_dict(),
                               "markers": [rng.uniform(.75, 1.25) for _ in range(self.markers)]}

    def _add_child(self, identity, parent, tick, anatomy):
        rng = self._rng(identity)
        markers = []
        for value in parent["markers"]:
            if rng.random() < self.mutation_rate:
                value = min(2.5, max(.3, value * math.exp(rng.uniform(-self.mutation_scale, self.mutation_scale))))
            markers.append(value)
        self.rows[identity] = {"id": identity, "parent": parent["id"], "founder": parent["founder"],
                               "depth": parent["depth"] + 1, "root_known": parent["root_known"], "born_tick": tick,
                               "death_tick": None, "death_reason": None, "anatomy": anatomy, "markers": markers}

    def record(self, world, events):
        """Add the births and deaths of one step's events (call after world.step())."""
        for event in events:
            kind = event.get("type")
            if kind == "birth":
                child = world.agents.get(event["agent"])
                anatomy = child.anatomy.to_dict() if child else None
                self._add_child(event["agent"], self.rows[event["parent"]], event["tick"], anatomy)
            elif kind == "death":
                row = self.rows[event["agent"]]
                row["death_tick"], row["death_reason"] = event["tick"], event.get("reason")

    # ------------------------------------------------------------ queries
    def alive(self):
        return [identity for identity, row in self.rows.items() if row["death_tick"] is None]

    def chain(self, identity):
        """Parent chain from ``identity`` back to its root (inclusive)."""
        out = []
        while identity is not None and identity in self.rows:
            out.append(identity)
            identity = self.rows[identity]["parent"]
        return out

    def founder(self, identity):
        return self.rows[identity]["founder"]

    def lineages(self):
        """Per founder: members, living members, deepest generation, founder death tick."""
        out = {}
        for row in self.rows.values():
            entry = out.setdefault(row["founder"], {"founder": row["founder"], "members": 0, "alive": 0,
                                                    "max_depth": 0, "founder_death_tick": None})
            entry["members"] += 1
            entry["alive"] += row["death_tick"] is None
            entry["max_depth"] = max(entry["max_depth"], row["depth"])
        for founder, entry in out.items():
            entry["founder_death_tick"] = self.rows[founder]["death_tick"]
        return out

    def summary(self):
        lineages = self.lineages().values()
        living = self.alive()
        return {"individuals": len(self.rows), "alive": len(living), "founders": len(lineages),
                "surviving_lineages": sum(e["alive"] > 0 for e in lineages),
                "max_depth": max((e["max_depth"] for e in lineages), default=0),
                "mean_depth_alive": round(sum(self.rows[i]["depth"] for i in living) / len(living), 3) if living else None}

    # ------------------------------------------------------------ persistence
    def to_dict(self):
        return {"schema": SCHEMA, "seed": self.seed, "markers": self.markers, "mutation_rate": self.mutation_rate,
                "mutation_scale": self.mutation_scale, "rows": [self.rows[k] for k in sorted(self.rows)]}

    @classmethod
    def from_dict(cls, data):
        if data.get("schema") != SCHEMA:
            raise ValueError("not a life8 lineage table")
        table = cls(data["seed"], markers=data["markers"], mutation_rate=data["mutation_rate"],
                    mutation_scale=data["mutation_scale"])
        table.rows = {row["id"]: dict(row) for row in data["rows"]}
        return table


def run_with_lineage(seed, config=None, ticks=600, *, keep=("signal", "heard", "birth", "death")):
    """Run a compact-log world from tick 0 with a lineage table. Returns (events, world, table)."""
    from .config import Config
    from .world import World
    values = dict(config or {})
    values["log"] = "compact"
    world = World.create(seed=seed, config=Config(**values))
    table = LineageTable.from_world(world)
    kept = []
    while world.tick < ticks and not world.stop_reason:
        events = world.step()
        table.record(world, events)
        kept.extend(e for e in events if e["type"] in keep)
    return kept, world, table


# ---------------------------------------------------------------- selection against neutral markers

def trait_selection(table, gene="body_mass", *, at_tick=None):
    """Change of the mean log gene value from the founders to the individuals alive at
    ``at_tick`` (default: the living), against the same change in each neutral marker.

    z = (gene change - mean marker change) / sd of marker changes. Markers drift by
    founder sampling and mutation only, so |z| well above 2 points to selection."""
    founders = [row for row in table.rows.values() if row["depth"] == 0 and row["anatomy"]]
    if at_tick is None:
        later = [row for row in table.rows.values() if row["death_tick"] is None]
    else:
        later = [row for row in table.rows.values() if row["born_tick"] <= at_tick
                 and (row["death_tick"] is None or row["death_tick"] > at_tick)]
    later = [row for row in later if row["anatomy"]]
    if not founders or not later:
        return {"gene": gene, "n_founders": len(founders), "n_later": len(later), "z": None}
    mean_log = lambda rows, get: sum(math.log(get(r)) for r in rows) / len(rows)
    change = mean_log(later, lambda r: r["anatomy"][gene]) - mean_log(founders, lambda r: r["anatomy"][gene])
    markers = [mean_log(later, lambda r: r["markers"][k]) - mean_log(founders, lambda r: r["markers"][k])
               for k in range(table.markers)]
    centre = sum(markers) / len(markers)
    sd = math.sqrt(sum((m - centre) ** 2 for m in markers) / max(1, len(markers) - 1))
    return {"gene": gene, "n_founders": len(founders), "n_later": len(later), "log_change": round(change, 5),
            "marker_changes": [round(m, 5) for m in markers], "marker_sd": round(sd, 5),
            "z": round((change - centre) / sd, 3) if sd > 0 else None,
            "beyond_all_markers": change < min(markers) or change > max(markers)}


# ---------------------------------------------------------------- conventions

def _situation(state, situation):
    return tuple(state.get(name) for name in situation)


def sender_mappings(events, *, situation=SITUATION, start=0, stop=None, min_calls=1):
    """{sender: {situation: Counter(symbol)}} from compact signal events."""
    out = defaultdict(lambda: defaultdict(Counter))
    for _, sender, state, symbol in sender_rows(events, start=start, stop=stop):
        out[sender][_situation(state, situation)][symbol] += 1
    return {s: dict(m) for s, m in out.items() if sum(sum(c.values()) for c in m.values()) >= min_calls}


def _agreement_matrix(mappings, senders, situations):
    """(agreement, blind) sender x sender matrices, NaN where two senders share no situation.

    agreement: P(same symbol) per shared situation, averaged over the shared situations.
    blind: the same for the two senders' symbol distributions averaged over those shared
    situations, i.e. what shared symbol habits alone give without any situation mapping."""
    index = {s: i for i, s in enumerate(situations)}
    probs = np.zeros((len(senders), len(situations), 4))
    for row, sender in enumerate(senders):
        for key, counts in mappings[sender].items():
            total = sum(counts.values())
            for symbol, count in counts.items():
                probs[row, index[key], symbol] = count / total
    used = (probs.sum(2) > 0).astype(float)
    shared = used @ used.T
    same = np.einsum("asi,bsi->ab", probs, probs)
    # mixed[a, b, i]: sender a's symbol-i probability summed over the situations b also used
    mixed = np.einsum("bs,asi->abi", used, probs)
    blind = np.einsum("abi,bai->ab", mixed, mixed)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (np.where(shared > 0, same / np.maximum(shared, 1), np.nan),
                np.where(shared > 0, blind / np.maximum(shared, 1) ** 2, np.nan))


def convention_agreement(events, table, *, situation=SITUATION, start=0, stop=None, min_calls=5, reps=200, seed=0):
    """Agreement on symbol use within founder lineages versus across them.

    ``within``/``across`` are the raw mean pair agreements. The tested ``difference`` is
    situation-specific: each pair's agreement minus its situation-blind agreement
    (see _agreement_matrix), within minus across, so lineages that merely share or inherit
    a symbol habit (e.g. inherited bias rows under the evolving genome; round-8 critic F2)
    score 0. It is tested against shuffles of lineage labels among senders (p = share of
    shuffles with a difference at least as large); ``difference_raw`` = within - across."""
    mappings = sender_mappings(events, situation=situation, start=start, stop=stop, min_calls=min_calls)
    senders = sorted(s for s in mappings if s in table.rows)
    if len(senders) < 3:
        return {"senders": len(senders), "within": None, "across": None, "difference": None, "p_value": None}
    situations = sorted({key for s in senders for key in mappings[s]}, key=repr)
    agreement, blind = _agreement_matrix(mappings, senders, situations)
    specific = agreement - blind
    upper = np.triu(np.ones_like(agreement, dtype=bool), 1) & ~np.isnan(agreement)
    labels = np.array([table.founder(s) for s in senders])

    def split(lab, matrix):
        same = lab[:, None] == lab[None, :]
        w, a = matrix[upper & same], matrix[upper & ~same]
        return (float(w.mean()) if w.size else None), (float(a.mean()) if a.size else None), int(w.size), int(a.size)

    within, across, n_within, n_across = split(labels, agreement)
    within_blind, across_blind, _, _ = split(labels, blind)
    r = lambda v: None if v is None else round(v, 5)
    result = {"senders": len(senders), "lineages": len(set(labels.tolist())), "situation": list(situation),
              "within": r(within), "across": r(across), "within_blind": r(within_blind), "across_blind": r(across_blind),
              "within_pairs": n_within, "across_pairs": n_across, "uniform_chance": .25}
    if within is None or across is None:
        result.update(difference=None, difference_raw=None, p_value=None)
        return result
    w, a, _, _ = split(labels, specific)
    observed = w - a
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(reps):
        w, a, _, _ = split(rng.permutation(labels), specific)
        if w is not None and a is not None:
            null.append(w - a)
    null = np.array(null)
    result.update(difference=round(observed, 5), difference_raw=round(within - across, 5),
                  null_mean=round(float(null.mean()), 5) if null.size else None,
                  p_value=round((1 + int(np.sum(null >= observed - 1e-12))) / (len(null) + 1), 5))
    return result


def _pooled(mappings, members):
    pooled = defaultdict(Counter)
    for member in members:
        for key, counts in mappings.get(member, {}).items():
            pooled[key].update(counts)
    return pooled


def _modes(pooled):
    return {key: min(counts, key=lambda s: (-counts[s], s)) for key, counts in pooled.items() if counts}


def _info(pooled, reps, seed):
    xs, ys = [], []
    for key, counts in pooled.items():
        for symbol, count in counts.items():
            xs += [repr(key)] * count
            ys += [symbol] * count
    return permutation_test(xs, ys, reps=reps, seed=seed)


def mapping_persistence(events, table, *, situation=SITUATION, min_calls=20, reps=200, seed=0, min_excess=.05,
                        alpha=.01):
    """For each founder lineage that outlived its founder: does the lineage's mapping
    from situation to symbol persist among descendants after the founder died?

    same_mode: share of situations used on both sides whose most frequent symbol is
    the same before and after; cross_same_mode: the same against other lineages'
    before-mappings (control); info_before/info_after: I(symbol; situation) on each
    side against a permutation null. ``persists`` needs a significant mapping on both
    sides (excess >= ``min_excess`` bits, p <= ``alpha``) and same_mode above both the
    cross-lineage control and 0.25. With ``reps`` permutations the smallest p is
    1 / (reps + 1), so ``alpha`` = 0.01 needs reps >= 99."""
    death = {f: e["founder_death_tick"] for f, e in table.lineages().items() if e["founder_death_tick"] is not None}
    rows = sender_rows(events)
    before, after = defaultdict(lambda: defaultdict(Counter)), defaultdict(lambda: defaultdict(Counter))
    for tick, sender, state, symbol in rows:
        if sender not in table.rows:
            continue
        founder = table.founder(sender)
        if founder not in death:
            continue
        side = before if tick < death[founder] else after
        side[founder][_situation(state, situation)][symbol] += 1
    total = lambda pooled: sum(sum(c.values()) for c in pooled.values())
    eligible = sorted(f for f in death if total(before[f]) >= min_calls and total(after[f]) >= min_calls)
    modes_before = {f: _modes(before[f]) for f in eligible}
    out = []
    for offset, founder in enumerate(eligible):
        b, a = modes_before[founder], _modes(after[founder])
        shared = [k for k in a if k in b]
        same = sum(a[k] == b[k] for k in shared) / len(shared) if shared else None
        cross = []
        for other in eligible:
            if other != founder:
                keys = [k for k in a if k in modes_before[other]]
                if keys:
                    cross.append(sum(a[k] == modes_before[other][k] for k in keys) / len(keys))
        info_b = _info(before[founder], reps, seed + 2 * offset)
        info_a = _info(after[founder], reps, seed + 2 * offset + 1)
        cross_mean = sum(cross) / len(cross) if cross else None
        significant = lambda info: info["excess_bits"] is not None and info["excess_bits"] >= min_excess \
            and info["p_value"] <= alpha
        persists = bool(significant(info_b) and significant(info_a) and same is not None and same > .25
                        and (cross_mean is None or same > cross_mean))
        out.append({"founder": founder, "founder_death_tick": death[founder], "calls_before": total(before[founder]),
                    "calls_after": total(after[founder]), "situations_shared": len(shared),
                    "same_mode": None if same is None else round(same, 4),
                    "cross_same_mode": None if cross_mean is None else round(cross_mean, 4),
                    "info_before": info_b, "info_after": info_a, "persists": persists})
    return {"situation": list(situation), "lineages_measured": len(out),
            "lineages_persisting": sum(r["persists"] for r in out), "lineages": out}


def lineage_report(seed, config=None, ticks=1500, *, window_start=None, reps=200):
    """One run: lineage summary, body-mass selection against markers, convention agreement
    in the late window and mapping persistence after founder death."""
    events, world, table = run_with_lineage(seed, config, ticks)
    window_start = ticks * 2 // 3 if window_start is None else window_start
    return {"seed": seed, "config_overrides": config or {}, "ticks_run": world.tick,
            "stop_reason": world.stop_reason or "step_budget", "lineage": table.summary(),
            "selection_body_mass": trait_selection(table, "body_mass"),
            "convention_late_window": convention_agreement(events, table, start=window_start, reps=reps, seed=seed),
            "persistence": mapping_persistence(events, table, reps=reps, seed=seed)}


def _report_job(args):
    seed, config, ticks, reps = args
    return lineage_report(seed, config, ticks, reps=reps)


def measure(seeds, config=None, ticks=1500, *, reps=200, jobs=6):
    tasks = [(seed, config, ticks, reps) for seed in seeds]
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            rows = list(pool.map(_report_job, tasks))
    else:
        rows = [_report_job(task) for task in tasks]
    return {"measurement": "lineages: body-mass selection vs neutral markers, convention agreement within vs "
                           "across founder lineages, mapping persistence after founder death",
            "seeds": list(seeds), "ticks": ticks, "config_overrides": config or {}, "runs": rows}


def main(argv=None):
    """usage: python -m haishool.life8.lineage --seeds 1-6 --ticks 1500 [--config JSON] [--out FILE]"""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--seeds", default="1-6")
    parser.add_argument("--ticks", type=int, default=1500)
    parser.add_argument("--reps", type=int, default=200)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--config", default="{}")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    low, _, high = args.seeds.partition("-")
    report = measure(range(int(low), int(high or low) + 1), json.loads(args.config), args.ticks, reps=args.reps,
                     jobs=args.jobs)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    for row in report["runs"]:
        c, sel, per = row["convention_late_window"], row["selection_body_mass"], row["persistence"]
        print(row["seed"], row["lineage"], "body_mass z", sel["z"], "within", c["within"], "across", c["across"],
              "p", c["p_value"], "persisting", per["lineages_persisting"], "/", per["lineages_measured"])


if __name__ == "__main__":
    main()
