"""Experience lines of life8 inhabitants: the single source of truth for the visitor format.

A *visitor* (docs/life8/visitors.md) is an inhabitant of a twin planet whose actions come
from a small GPT trained only on lines written here. Nothing else (no human text, no other
Haishool data) enters that model. One line is one decision of one inhabitant:

    q life8 see <feature> <value> <feature> <value> ... want <outcome>. a <action>.

* features: what the inhabitant's own life8 learner sees, ``learning.observation_features(
  observation, extras=True)`` (the bins a multistep controller decides from), written in the
  fixed order :data:`FEATURE_ORDER`. A feature whose value is ``None`` (absent: no held tool,
  nothing heard, no visible food) is left out. Booleans are ``yes``/``no``, integers are digit
  tokens (:func:`haishool.truth.num`), strings are one lowercase token. An unknown feature
  name is an error, so a future engine feature cannot slip into the format unnoticed.
* outcome: ``thrive`` / ``live`` / ``suffer`` from the acting individual's own reward (the
  engine's energy + 8 x health change, the learning reward) summed over the 10 ticks that
  start with this decision (fewer when it dies inside them), cut at the 33rd and 67th
  percentiles of a calibration set (the train split's lines; the cuts are stored in the data
  report). A visitor asks with ``want thrive``.
* action: the engine's action name, one token (:data:`ACTION_WORDS`). Structured actions
  (share, teach, imitate, strike, ...) carry no target: the engine's default target choice
  applies, as for an ordinary inhabitant.

The module also builds the data set (``generate``), trains the model on the training host
(``train``: ``train_final`` with mix sim), measures it (``evaluate``) and exports float16
(``export``). See ``python -m haishool.life8.experience --help``.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import random
import re
import sys
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path

from . import learning
from .config import Config
from .world import ACTIONS as WORLD_ACTIONS

FORMAT = "life8-experience-v1"
#: the engine's action names, one token each; a visitor's answer must be one of them
ACTION_WORDS = tuple(learning.ACTIONS)
if ACTION_WORDS != tuple(WORLD_ACTIONS):  # pragma: no cover - engine invariant
    raise RuntimeError("learning.ACTIONS and world.ACTIONS differ")
#: the observation features in line order: state bins, living-world bins, the heard symbol, then
#: the bins a multistep controller derives itself (food distance band, heard bearing)
FEATURE_ORDER = (*learning.STATE_FEATURES, *learning.ECOLOGY_FEATURES, "heard_symbol", *learning.EXTRA_FEATURES)
OUTCOMES = ("thrive", "live", "suffer")
#: ticks of own reward summed into one outcome (the decision's own tick and the next nine)
HORIZON = 10
#: percentiles of the calibration returns that cut suffer | live | thrive
CUTS = (33., 67.)
MAX_TOKENS = 120
TOPIC = "predict_life8_experience"
ELDER_TOPIC = "predict_life8_elder"
#: the deterministic draw of planets from the r8a plan
SELECT_SEED = 20261002
SPLITS = ("train", "dev", "sealed")
_TOKEN = re.compile(r"^[a-z0-9_]+$")
_LINE = re.compile(r"^q life8 see (?P<see>[a-z0-9_ ]*?) ?want (?P<want>[a-z]+)\. a (?P<action>[a-z0-9_]+)\.$")


# ---------------------------------------------------------------------------------------------
# the line format


def value_text(name: str, value) -> str | None:
    """One feature value as dense text; ``None`` when the feature is absent."""
    if value is None:
        return None
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        from haishool.truth import num
        return num(value)
    if isinstance(value, str) and _TOKEN.match(value):
        return value
    raise ValueError(f"feature {name} has a value that is not one dense token: {value!r}")


def see_text(features: dict) -> str:
    """``<feature> <value> ...`` in :data:`FEATURE_ORDER`, absent features left out."""
    unknown = set(features) - set(FEATURE_ORDER)
    if unknown:
        raise ValueError(f"features outside the experience format: {sorted(unknown)}")
    parts = []
    for name in FEATURE_ORDER:
        if name in features:
            text = value_text(name, features[name])
            if text is not None:
                parts.append(f"{name} {text}")
    return " ".join(parts)


def features_of(observation) -> dict:
    """The features an inhabitant's own learner sees in an engine observation (``World.observe``)."""
    return learning.observation_features(observation, extras=True)


def prompt_for(observation_features: dict, want: str = "thrive") -> str:
    """The question a visitor asks: ``q life8 see ... want thrive. a`` (the model answers the action)."""
    if want not in OUTCOMES:
        raise ValueError(f"unknown outcome {want!r}")
    see = see_text(observation_features)
    return f"q life8 see {see} want {want}. a" if see else f"q life8 see want {want}. a"


def _line_from_see(see: str, outcome: str, action: str) -> str:
    return (f"q life8 see {see} want {outcome}. a {action}." if see
            else f"q life8 see want {outcome}. a {action}.")


def line_for(observation_features: dict, outcome: str, action: str) -> str:
    """One training line; checked dense and shorter than :data:`MAX_TOKENS` tokens."""
    if outcome not in OUTCOMES:
        raise ValueError(f"unknown outcome {outcome!r}")
    if action not in ACTION_WORDS:
        raise ValueError(f"unknown action {action!r}")
    line = prompt_for(observation_features, outcome) + f" {action}."
    if token_count(line) >= MAX_TOKENS:
        raise ValueError(f"experience line has {token_count(line)} tokens, limit {MAX_TOKENS}")
    return line


def token_count(line: str) -> int:
    """Tokens the trainer sees for a line, including its end-of-line token."""
    return len(line.replace(".", " . ").split()) + 1


def parse_line(line: str) -> tuple[str, str, str]:
    """``(see, outcome, action)`` of an experience line; ValueError if it is not one."""
    match = _LINE.match(line)
    if not match or match["want"] not in OUTCOMES or match["action"] not in ACTION_WORDS:
        raise ValueError(f"not an experience line: {line[:120]!r}")
    return match["see"], match["want"], match["action"]


def action_of(answer: str) -> str | None:
    """The engine action a model answer names, or None (a visitor then falls back, see visitors.md)."""
    word = answer.strip().rstrip(".").strip()
    return word if word in ACTION_WORDS else None


def outcome_of(total: float, thresholds: tuple[float, float]) -> str:
    """suffer below the lower cut, thrive above the upper cut, live in between (cuts inclusive to live)."""
    low, high = thresholds
    return "suffer" if total < low else "thrive" if total > high else "live"


# ---------------------------------------------------------------------------------------------
# planets


def build_world(entry: dict, log: str = "full"):
    """The world of one r8a plan entry (as ``search.build_world``, with the chosen log mode)."""
    from .world import World
    config = Config.from_dict({**entry["config"], "log": log})
    if entry.get("founders"):
        from . import bridge
        return bridge.create_world(entry["seed"], config, entry["founders"])
    return World.create(seed=entry["seed"], config=config)


def select_planets(plan: list[dict], *, seed: int = SELECT_SEED, world7: int = 300, synthetic: int = 100,
                   reserve: tuple[int, int] = (30, 10)) -> dict:
    """The deterministic draw: data planets split train/dev/sealed 80/10/10 per source, and two
    reserved sets (elder, traveller) of ``reserve`` = (world7, synthetic) planets each.

    The traveller set gets no data at all; the elder set gets lines from its first 1000 ticks
    only (its own topic). The three groups are disjoint."""
    rng = random.Random(seed)
    by_source = {s: sorted((e for e in plan if e["source"] == s), key=lambda e: e["seed"])
                 for s in ("world7", "synthetic")}
    out = {"traveller": [], "elder": [], "train": [], "dev": [], "sealed": []}
    for source, count, held in (("world7", world7, reserve[0]), ("synthetic", synthetic, reserve[1])):
        drawn = rng.sample(by_source[source], count + 2 * held)
        out["traveller"] += [e["id"] for e in drawn[:held]]
        out["elder"] += [e["id"] for e in drawn[held:2 * held]]
        data = [e["id"] for e in drawn[2 * held:]]
        rng.shuffle(data)
        tenth = count // 10
        out["dev"] += data[:tenth]
        out["sealed"] += data[tenth:2 * tenth]
        out["train"] += data[2 * tenth:]
    for key in out:
        out[key] = sorted(out[key])
    ids = [i for v in out.values() for i in v]
    if len(ids) != len(set(ids)):  # pragma: no cover - by construction
        raise RuntimeError("planet groups overlap")
    return out


def kept(planet: str, tick: int, agent: int, keep: float) -> bool:
    """Deterministic subsample of decisions (independent of process and order)."""
    return keep >= 1 or zlib.crc32(f"{planet}/{tick}/{agent}".encode()) < keep * 4294967296


def collect(entry: dict, ticks: int, *, keep: float = 1., horizon: int = HORIZON, max_tick: int | None = None):
    """Run one planet ``ticks`` steps (full log, in process) and return its decisions.

    Returns ``(rows, stats)``; each row is ``(tick, agent, see, action, total)`` where ``tick`` is
    the engine tick the decision's transition is recorded at (1 = the first decision), and
    ``total`` the agent's own reward summed over that tick and the next ``horizon - 1``. A
    decision whose window runs past the simulated (or ``max_tick``) ticks while the agent is
    still alive is dropped; one whose agent dies inside the window keeps the shorter sum."""
    world = build_world(entry, "full")
    planet = entry["id"]
    rewards: dict[int, list] = {}     # agent -> [first tick, [rewards], died]
    pending = []
    see_cache: dict = {}
    started = time.monotonic()
    decisions = 0
    population = []
    for _ in range(ticks):
        if world.stop_reason:
            break
        events = world.step()
        for event in events:
            if event.get("type") != "transition":
                continue
            agent, tick = event["agent"], event["tick"]
            record = rewards.get(agent)
            if record is None:
                record = rewards[agent] = [tick, [], False]
            record[1].append(event["reward"])
            if event["done"]:
                record[2] = True
            decisions += 1
            if not kept(planet, tick, agent, keep):
                continue
            features = learning.observation_features(event["observation"], extras=True)
            key = tuple(features.items())
            see = see_cache.get(key)
            if see is None:
                if len(see_cache) > 200000:
                    see_cache.clear()
                see = see_cache[key] = see_text(features)
            pending.append((tick, agent, see, event["action"]["type"]))
        population.append(len(world.agents))
    last = world.tick if max_tick is None else min(world.tick, max_tick)
    rows = []
    for tick, agent, see, action in pending:
        first, series, died = rewards[agent]
        start = tick - first
        window = series[start:start + horizon]
        if len(window) < horizon and not died:
            continue
        if not died and tick + horizon - 1 > last:
            continue
        rows.append((tick, agent, see, action, math.fsum(window)))
    stats = {"id": planet, "source": entry["source"], "ticks": world.tick, "stop_reason": world.stop_reason,
             "decisions": decisions, "kept_rows": len(rows), "keep": keep,
             "population_mean": sum(population) / len(population) if population else 0.,
             "population_final": len(world.agents), "seconds": round(time.monotonic() - started, 2)}
    return rows, stats


# ---------------------------------------------------------------------------------------------
# generation (this PC, process pool)


def _job(task):
    """One planet into ``raw``: every complete decision (the subsample is applied at assembly)."""
    entry, ticks, raw = task
    path = Path(raw) / f"{entry['id']}.tsv.gz"
    done = path.with_suffix("").with_suffix(".json")
    if done.exists():
        return json.loads(done.read_text(encoding="utf-8"))
    rows, stats = collect(entry, ticks)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=3, newline="\n") as handle:
        for tick, agent, see, action, total in rows:
            handle.write(f"{tick}\t{agent}\t{action}\t{total!r}\t{see}\n")
    tmp.replace(path)
    done.write_text(json.dumps(stats), encoding="utf-8")
    return stats


def read_raw(raw: Path, planet: str):
    with gzip.open(raw / f"{planet}.tsv.gz", "rt", encoding="utf-8") as handle:
        for text in handle:
            tick, agent, action, total, see = text.rstrip("\n").split("\t")
            yield int(tick), int(agent), action, float(total), see


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")
    tmp.replace(path)


def elder_split(planet: str, agent: int) -> str:
    """Elder lines are split by individual (a whole life is held out), 80/10/10."""
    bucket = zlib.crc32(f"elder/{planet}/{agent}".encode()) % 10
    return "dev" if bucket == 0 else "sealed" if bucket == 1 else "train"


def generate(plan_path: Path, out: Path, raw: Path, *, workers: int = 30, ticks: int = 2000, elder_ticks: int = 1000,
             keep: float = 1., elder_keep: float | None = None, only: list[str] | None = None) -> dict:
    """Simulate the selected planets (resumable per planet into ``raw``) and write the data set."""
    from concurrent.futures import ProcessPoolExecutor
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))["plan"]
    by_id = {e["id"]: e for e in plan}
    groups = select_planets(plan)
    raw.mkdir(parents=True, exist_ok=True)
    tasks = [(by_id[i], elder_ticks, str(raw)) for i in groups["elder"]]
    tasks += [(by_id[i], ticks, str(raw)) for s in SPLITS for i in groups[s]]
    if only is not None:
        tasks = [t for t in tasks if t[0]["id"] in set(only)]
    started = time.monotonic()
    stats = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for n, row in enumerate(pool.map(_job, tasks, chunksize=1), 1):
            stats[row["id"]] = row
            print(json.dumps({"done": n, "of": len(tasks), "id": row["id"], "ticks": row["ticks"],
                              "rows": row["kept_rows"], "seconds": row["seconds"],
                              "elapsed": round(time.monotonic() - started, 1)}), flush=True)
    wall = time.monotonic() - started
    if only is not None:
        return {"stats": stats, "wall_seconds": wall}
    return assemble(out, raw, groups, stats, keep=keep, ticks=ticks, elder_ticks=elder_ticks, wall=wall,
                    workers=workers, plan_path=plan_path, elder_keep=elder_keep)


def assemble(out: Path, raw: Path, groups: dict, stats: dict, *, keep, ticks, elder_ticks, wall, workers, plan_path,
             elder_keep=None):
    """Write the topics from the raw planet files: ``keep`` subsamples experience decisions,
    ``elder_keep`` (default ``keep``) the elder ones; outcome cuts from the kept train lines."""
    import numpy as np
    elder_keep = keep if elder_keep is None else elder_keep
    out.mkdir(parents=True, exist_ok=True)
    if (out / "report.json").exists():
        raise FileExistsError(f"data set exists: {out}")
    calibration = np.array([row[3] for planet in groups["train"] for row in read_raw(raw, planet)
                            if kept(planet, row[0], row[1], keep)], dtype=np.float64)
    low, high = (float(x) for x in np.percentile(calibration, CUTS))
    thresholds = (low, high)
    topics = {}
    outcome_counts = {TOPIC: {s: Counter() for s in SPLITS}, ELDER_TOPIC: {s: Counter() for s in SPLITS}}
    action_counts = {TOPIC: Counter(), ELDER_TOPIC: Counter()}
    for topic in (TOPIC, ELDER_TOPIC):
        handles = {s: (out / f"{topic}-{s}.txt").open("w", encoding="utf-8", newline="\n") for s in SPLITS}
        counts = {s: {"lines": 0, "tokens": 0, "planets": 0} for s in SPLITS}
        try:
            planets = ([(p, s) for s in SPLITS for p in groups[s]] if topic == TOPIC
                       else [(p, None) for p in groups["elder"]])
            seen = defaultdict(set)
            for planet, split in planets:
                rate = keep if topic == TOPIC else elder_keep
                for tick, agent, action, total, see in read_raw(raw, planet):
                    if not kept(planet, tick, agent, rate):
                        continue
                    if topic == ELDER_TOPIC and tick + HORIZON - 1 > elder_ticks:
                        continue  # elder windows end at tick 1000 at the latest
                    where = split or elder_split(planet, agent)
                    outcome = outcome_of(total, thresholds)
                    line = _line_from_see(see, outcome, action)
                    handles[where].write(line + "\n")
                    counts[where]["lines"] += 1
                    counts[where]["tokens"] += token_count(line)
                    outcome_counts[topic][where][outcome] += 1
                    if where == "train":
                        action_counts[topic][action] += 1
                    seen[where].add(planet)
            for s in SPLITS:
                counts[s]["planets"] = len(seen[s])
        finally:
            for handle in handles.values():
                handle.close()
        for s in SPLITS:
            counts[s]["sha256"] = digest(out / f"{topic}-{s}.txt")
        topics[topic] = {"splits": counts,
                         "outcomes": {s: dict(outcome_counts[topic][s]) for s in SPLITS},
                         "train_actions": dict(action_counts[topic].most_common())}
    planet_stats = list(stats.values())
    data_stats = [s for s in planet_stats if s["id"] not in set(groups["elder"])]
    report = {
        "round": 8, "kind": "life8 visitor experience", "format": FORMAT, "seed": SELECT_SEED,
        "line": "q life8 see <feature> <value> ... want <outcome>. a <action>.",
        "feature_order": list(FEATURE_ORDER), "action_words": list(ACTION_WORDS), "outcomes": list(OUTCOMES),
        "horizon": HORIZON, "thresholds": {"percentiles": list(CUTS), "suffer_below": low, "thrive_above": high,
                                           "calibration": "every kept line of the train planets (experience topic)",
                                           "calibration_lines": int(calibration.size)},
        "keep": keep, "elder_keep": elder_keep, "ticks": ticks, "elder_ticks": elder_ticks,
        "plan": str(plan_path), "planets": groups,
        "split": "experience: by planet 80/10/10 per source; elder: 40 reserved planets, decisions whose "
                 "10-tick window ends by tick 1000, split by individual 80/10/10; traveller: 40 reserved "
                 "planets, no data",
        "topics": topics,
        "totals": {s: {k: sum(t["splits"][s][k] for t in topics.values()) for k in ("lines", "tokens")} for s in SPLITS},
        "generation": {"workers": workers, "wall_seconds": round(wall, 1),
                       "planet_seconds": round(sum(s["seconds"] for s in planet_stats), 1),
                       "decisions": sum(s["decisions"] for s in planet_stats),
                       "planet_ticks": sum(s["ticks"] for s in planet_stats),
                       "stopped": dict(Counter(str(s["stop_reason"]) for s in planet_stats)),
                       "extinct_data_planets": sorted(s["id"] for s in data_stats if s["stop_reason"] == "extinction"),
                       "population_mean": sum(s["population_mean"] for s in data_stats) / max(1, len(data_stats))},
        "planet_stats": planet_stats,
    }
    _json_write(out / "report.json", report)
    return report


# ---------------------------------------------------------------------------------------------
# training (on the GPU host; torch is needed from here on)


def proportional_sim_mixture(sources) -> None:
    """Visitor training: every source is an experience topic (predict_life8_*), weighted by its tokens."""
    for source in sources:
        if not source.topic.startswith("predict_life8_") or source.feedback:
            raise ValueError(f"visitor training takes only life8 experience topics, not {source.topic}")
    total = sum(s.count for s in sources)
    for source in sources:
        source.weight = source.count / total
    if not math.isclose(sum(s.weight for s in sources), 1.):
        raise ValueError("invalid mixture weights")


def train(data: Path, out: Path, *, steps: int, layers=6, width=384, heads=8, ctx=128, batch=48, seed=20261002,
          lr=None, device=None, checkpoint_every=2000, resume=None) -> dict:
    """``train_final.train(mix="sim")`` with one change for visitor data: train_final's sim mix wants
    stories (legacy_*) for 20% of the tokens and splits lessons equally by topic; visitor data has
    no stories, so here the predict share is the whole and topics are weighted by their tokens."""
    from haishool import train_final
    train_final.SIM_SHARES = {"predict": 1.}
    train_final.sim_mixture = proportional_sim_mixture
    return train_final.train(data=data, out=out, layers=layers, width=width, heads=heads, ctx=ctx, steps=steps,
                             holdout=0., seed=seed, batch=batch, device=device, checkpoint_every=checkpoint_every,
                             lr=lr, mix="sim", resume=resume)


# ---------------------------------------------------------------------------------------------
# measurement


def _majority(keys, actions, n_actions):
    """For each distinct key the most common action (ties: the lower action index)."""
    import numpy as np
    combo = keys.astype(np.int64)
    order = np.lexsort((actions, combo))
    k, a = combo[order], actions[order]
    pair_change = np.ones(len(k), dtype=bool)
    pair_change[1:] = (k[1:] != k[:-1]) | (a[1:] != a[:-1])
    starts = np.flatnonzero(pair_change)
    counts = np.diff(np.append(starts, len(k)))
    pk, pa = k[starts], a[starts]
    # per key: the pair with the largest count, the lowest action among equals
    order2 = np.lexsort((pa, -counts, pk))
    pk, pa = pk[order2], pa[order2]
    first = np.ones(len(pk), dtype=bool)
    first[1:] = pk[1:] != pk[:-1]
    return dict(zip(pk[first].tolist(), pa[first].tolist()))


def _key(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "little", signed=True)


def baselines(train_files: list[Path]):
    """Most common action per observation (``see``) in train, and per (observation, outcome)."""
    import numpy as np
    index = {a: i for i, a in enumerate(ACTION_WORDS)}
    obs, both, acts = [], [], []
    for path in train_files:
        with path.open(encoding="utf-8") as handle:
            for text in handle:
                see, want, action = parse_line(text.rstrip("\n"))
                obs.append(_key(see))
                both.append(_key(see + "|" + want))
                acts.append(index[action])
    acts = np.array(acts, dtype=np.int64)
    overall = int(np.bincount(acts, minlength=len(ACTION_WORDS)).argmax())
    return {"by_see": _majority(np.array(obs), acts, len(ACTION_WORDS)),
            "by_see_want": _majority(np.array(both), acts, len(ACTION_WORDS)),
            "overall": overall, "train_lines": int(len(acts))}


def _rate(hits, n):
    return {"correct": int(hits), "lines": int(n), "accuracy": (hits / n) if n else None}


def evaluate(checkpoint: Path, data: Path, out: Path, *, device="cuda", limit: int = 0, batch: int = 512) -> dict:
    """Exact action accuracy on dev and sealed (all lines, and lines whose outcome was thrive),
    for the model (argmax over the action words after ``a``) and the train-majority baselines."""
    import numpy as np
    import torch
    from haishool.student import EOS, load, tokens
    model, vocab = load(checkpoint, device)
    model.eval()
    action_ids = [vocab.stoi.get(a, -1) for a in ACTION_WORDS]
    present = [i for i, t in enumerate(action_ids) if t >= 0]
    base = baselines(sorted(Path(data).glob("predict_life8_*-train.txt")))
    results = {"checkpoint": str(checkpoint), "baselines": {"overall_action": ACTION_WORDS[base["overall"]],
                                                            "train_lines": base["train_lines"],
                                                            "distinct_train_observations": len(base["by_see"])},
               "splits": {}}
    amp = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if str(device).startswith("cuda") else None
    for topic in (TOPIC, ELDER_TOPIC):
        for split in ("dev", "sealed"):
            path = Path(data) / f"{topic}-{split}.txt"
            lines = [t.rstrip("\n") for t in path.open(encoding="utf-8") if t.strip()]
            if limit and len(lines) > limit:
                lines = random.Random(f"{topic}/{split}").sample(lines, limit)
            parsed = [parse_line(t) for t in lines]
            if not parsed:
                results["splits"][f"{topic}/{split}"] = {"lines": 0}
                continue
            groups = defaultdict(list)
            unknown = 0
            for i, (see, want, action) in enumerate(parsed):
                words = [EOS, *tokens(_line_from_see(see, want, action).rsplit(" a ", 1)[0] + " a")]
                if any(w not in vocab.stoi for w in words):
                    unknown += 1
                    continue
                groups[len(words)].append((i, vocab.encode(words)))
            predicted = [-1] * len(parsed)
            free_valid = 0
            with torch.inference_mode():
                for rows in groups.values():
                    for offset in range(0, len(rows), batch):
                        part = rows[offset:offset + batch]
                        x = torch.tensor([v for _, v in part], device=device)
                        if amp:
                            with amp():
                                logits, _ = model(x)
                        else:
                            logits, _ = model(x)
                        last = logits[:, -1].float()
                        free = last.argmax(-1).tolist()
                        restricted = last[:, [action_ids[i] for i in present]].argmax(-1).tolist()
                        for (i, _), f, r in zip(part, free, restricted):
                            predicted[i] = present[r]
                            free_valid += vocab.itos[f] in ACTION_WORDS
            index = {a: k for k, a in enumerate(ACTION_WORDS)}
            gold = np.array([index[a] for _, _, a in parsed], dtype=np.int64)
            thrive = np.array([w == "thrive" for _, w, _ in parsed], dtype=bool)
            model_pred = np.array(predicted, dtype=np.int64)
            by_see = np.array([base["by_see"].get(_key(s), base["overall"]) for s, _, _ in parsed], dtype=np.int64)
            seen = np.array([_key(s) in base["by_see"] for s, _, _ in parsed], dtype=bool)
            by_both = np.array([base["by_see_want"].get(_key(s + "|" + w), base["by_see"].get(_key(s), base["overall"]))
                                for s, w, _ in parsed], dtype=np.int64)
            row = {"lines": len(parsed), "unscorable_unknown_words": unknown,
                   "thrive_lines": int(thrive.sum()), "observation_seen_in_train": float(seen.mean()) if len(seen) else None,
                   "free_answer_is_an_action": free_valid / max(1, len(parsed) - unknown)}
            for name, pred in (("model", model_pred), ("baseline_majority_per_observation", by_see),
                               ("baseline_majority_per_observation_and_outcome", by_both),
                               ("baseline_overall_majority", np.full(len(gold), base["overall"]))):
                hit = pred == gold
                row[name] = {"all": _rate(hit.sum(), len(hit)), "thrive": _rate(hit[thrive].sum(), thrive.sum()),
                             "observation_seen": _rate(hit[seen].sum(), seen.sum()),
                             "observation_unseen": _rate(hit[~seen].sum(), (~seen).sum())}
            row["gold_actions"] = dict(Counter(ACTION_WORDS[g] for g in gold).most_common(8))
            row["model_actions"] = dict(Counter(ACTION_WORDS[p] for p in model_pred if p >= 0).most_common(8))
            results["splits"][f"{topic}/{split}"] = row
            print(json.dumps({"evaluated": f"{topic}/{split}", "lines": len(parsed),
                              "model": row["model"]["all"]["accuracy"],
                              "thrive": row["model"]["thrive"]["accuracy"],
                              "baseline": row["baseline_majority_per_observation"]["all"]["accuracy"]}), flush=True)
    _json_write(out, results)
    return results


def export(checkpoint: Path, target: Path, data: Path) -> dict:
    """float16 copy with the keys ``student.load`` reads (model_state, gpt_config, itos), checked."""
    import torch
    original = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = {k: v.half() if v.is_floating_point() else v for k, v in original["model_state"].items()}
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    torch.save({"model_state": state, "gpt_config": original["gpt_config"], "itos": original["itos"]}, tmp)
    tmp.replace(target)
    again = torch.load(target, map_location="cpu", weights_only=True)
    if set(again) != {"model_state", "gpt_config", "itos"} or again["itos"] != original["itos"] \
            or not all(torch.equal(again["model_state"][k], v) for k, v in state.items()):
        raise RuntimeError("export does not match the float16 conversion")
    error = max(float((v.float() - original["model_state"][k].float()).abs().max())
                for k, v in state.items() if v.is_floating_point())
    report = json.loads((Path(data) / "report.json").read_text(encoding="utf-8"))
    manifest = {"export": target.name, "sha256": digest(target), "bytes": target.stat().st_size,
                "source": str(checkpoint), "source_sha256": digest(checkpoint), "dtype": "float16",
                "keys": ["model_state", "gpt_config", "itos"], "max_abs_error_vs_float32": error,
                "gpt_config": original["gpt_config"], "vocab": len(original["itos"]), "format": FORMAT,
                "thresholds": report["thresholds"], "planets": report["planets"],
                "prompt": "prompt_for(features_of(observation), 'thrive') from haishool/life8/experience.py"}
    _json_write(target.with_suffix(".json"), manifest)
    return manifest


# ---------------------------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    g = sub.add_parser("generate", help="simulate the selected planets and write the data set")
    g.add_argument("--plan", type=Path, default=Path("runs/life8-search/r8a/stage/plan.json"))
    g.add_argument("--out", type=Path, required=True)
    g.add_argument("--raw", type=Path, required=True)
    g.add_argument("--workers", type=int, default=30)
    g.add_argument("--ticks", type=int, default=2000)
    g.add_argument("--elder-ticks", type=int, default=1000)
    g.add_argument("--keep", type=float, default=1.)
    g.add_argument("--elder-keep", type=float)
    g.add_argument("--only", nargs="*")
    s = sub.add_parser("select", help="print the planet draw")
    s.add_argument("--plan", type=Path, default=Path("runs/life8-search/r8a/stage/plan.json"))
    t = sub.add_parser("train", help="train_final mix sim on the experience topics")
    t.add_argument("--data", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--steps", type=int, required=True)
    for name, default in (("layers", 6), ("width", 384), ("heads", 8), ("ctx", 128), ("batch", 48),
                          ("checkpoint-every", 2000)):
        t.add_argument("--" + name, type=int, default=default)
    t.add_argument("--lr", type=float)
    t.add_argument("--device")
    t.add_argument("--resume", type=Path)
    e = sub.add_parser("evaluate", help="action accuracy on dev and sealed, with baselines")
    e.add_argument("--checkpoint", type=Path, required=True)
    e.add_argument("--data", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--device", default="cuda")
    e.add_argument("--limit", type=int, default=0)
    x = sub.add_parser("export", help="float16 export")
    x.add_argument("--checkpoint", type=Path, required=True)
    x.add_argument("--target", type=Path, required=True)
    x.add_argument("--data", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.command == "generate":
        result = generate(args.plan, args.out, args.raw, workers=args.workers, ticks=args.ticks,
                          elder_ticks=args.elder_ticks, keep=args.keep,
                          elder_keep=args.elder_keep, only=args.only)
        print(json.dumps(result.get("totals", {}) if "totals" in result else {"planets": len(result["stats"])}))
    elif args.command == "select":
        plan = json.loads(args.plan.read_text(encoding="utf-8"))["plan"]
        print(json.dumps(select_planets(plan), indent=1))
    elif args.command == "train":
        result = train(args.data, args.out, steps=args.steps, layers=args.layers, width=args.width, heads=args.heads,
                       ctx=args.ctx, batch=args.batch, lr=args.lr, device=args.device,
                       checkpoint_every=args.checkpoint_every, resume=args.resume)
        print(json.dumps({k: result[k] for k in ("status", "steps", "checkpoint", "sampled_tokens")}))
    elif args.command == "evaluate":
        evaluate(args.checkpoint, args.data, args.out, device=args.device, limit=args.limit)
    elif args.command == "export":
        print(json.dumps(export(args.checkpoint, args.target, args.data), indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
