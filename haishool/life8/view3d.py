"""3D browser viewer for life8 run archives (round 8, "living worlds").

    python -m haishool.life8.view3d build RUN_DIR --out FILE.html [--every K] [--max-ticks N]
                                         [--replay auto|on|off] [--max-mb 12]

Reads a run archive written by ``cli.run_world`` (manifest.json, snapshots.jsonl,
events.jsonl, summary.json, checkpoint.json) and writes ONE self-contained HTML page:
``view3d_template.html`` with the data embedded as compact JSON. The page loads only
two pinned scripts (Three.js r128 and its OrbitControls) from public CDNs.

Size: positions are stored in hundredths of a world unit, energy in tenths, health in
hundredths. ``--every K`` keeps one snapshot per K ticks; if the page would exceed
``--max-mb`` (default 12 MB) K is doubled until it fits, and as a last resort the
per-individual list of heard calls is reduced to counts.

What the snapshots do not hold is computed here from the archive, never by changing
the engine:

* blooms: food ids named by compact ``bloom`` events (else ids above
  ``Config.food_patches`` when blooms are on), with spoil tick = bloom tick + bloom_ttl;
* permanent scars: ``Config.permanent_injury`` x every wound, from ``predation``
  events and successful ``strike_agent`` acts (wound = strike_injury x damage, x
  attacker / target body mass with ``injury_mass_scaling``), capped at 1;
* call power: a call's range / ``Config.call_reach`` (signal events), else the final
  checkpoint's ecology record;
* lineage: founder = root of the parent chain (snapshots' ``parent_id`` and births);
* predators: their positions live only in the ecology record, not in snapshots. When
  the archive starts at tick 0 (or its resume checkpoint still exists) the run is
  replayed from the manifest's seed and config, and the replay is used only if it
  matches every snapshot (ids, positions, energy and health). Otherwise predators are
  shown only where and when they struck (``predation`` events).

The viewer shows recorded data; it adds no behaviour. Standard library only (the
search hosts have no numpy). This module is not imported by the engine.
"""
from __future__ import annotations

import argparse
import bisect
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys
import time

TEMPLATE = Path(__file__).resolve().parent / "view3d_template.html"
DATA_TOKEN = "__LIFE8_VIEW3D_DATA__"
MAX_BYTES = 12_000_000
ALLOWED_SCRIPTS = ("https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js",
                   "https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js")
DATA_VERSION = "life8-view3d-v1"
MATERIALS = ("stone", "wood", "fiber", "clay", "metal", "aggregate")
REASONS = ("starvation", "predation", "injury", "age")
# Visual event kinds (index = code in the page)
EVENT_KINDS = ("birth", "death", "predation", "strike_agent", "strike_food", "strike_object", "share",
               "teach", "imitate", "collect", "drop", "combine", "dismantle", "heat")
NOTABLE = {"strike", "strike_agent", "strike_object", "share", "teach", "imitate", "collect", "drop",
           "combine", "dismantle", "heat"}
IND_COLUMNS = ("id", "parent", "founder", "generation", "born", "died", "reason", "body_mass", "speed",
               "sensing", "manipulation", "metabolism", "voice", "hearing", "call_power", "lifespan",
               "alpha", "epsilon", "bias_entries", "innate_eat", "innate_call", "visitor", "x", "y", "presence")
CONFIG_KEYS = ("width", "height", "population", "food_patches", "initial_objects", "population_limit",
               "max_energy", "interaction_range", "message_ttl", "signal_noise", "signal_cost", "call_power",
               "call_reach", "bloom_interval", "bloom_amount", "bloom_ttl", "bloom_richness", "bloom_partners",
               "predators", "predator_speed", "predator_sight", "predator_cooldown", "predator_damage",
               "permanent_injury", "strike_injury", "injury_mass_scaling", "health_ability", "kin_credit",
               "genome", "reproduction", "channel", "log", "visibility", "max_age", "maturity_age",
               "tools", "communication", "culture", "learning")


class ArchiveError(ValueError):
    pass


def _json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ArchiveError(f"missing archive file: {path}") from exc


def _jsonl(path):
    """Rows of a JSON-lines file; a truncated last line (run still writing) is skipped."""
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                if line.endswith("\n"):
                    raise
                return


def _r(value, digits=3):
    return None if value is None else round(float(value), digits)


def _i(value, scale):
    return int(round(float(value) * scale))


def _genome_summary(genome, config):
    """alpha, epsilon, innate bias entry count, innate P(eat | food in reach) and
    innate P(any call | food in reach) of a newborn from its genome alone."""
    genome = genome or {}
    bias = genome.get("bias") or {}
    out = {"alpha": genome.get("alpha"), "epsilon": genome.get("epsilon"),
           "bias_entries": sum(len(row) for row in bias.values() if isinstance(row, dict)),
           "innate_eat": None, "innate_call": None}
    try:  # learning/genome are engine modules; the viewer degrades if they cannot be imported
        from . import genome as genes
        from . import learning
        expressed = genes.expressed(genome, config)
        out["alpha"], out["epsilon"] = expressed["alpha"], expressed["epsilon"]
        actions = learning.ACTIONS
        epsilon, eat, call = expressed["epsilon"], 0., 0.
        for context in genes.NEAR_CONTEXTS:
            values = genes.innate_values(genome, context)
            best = max(values[a] for a in actions)
            tied = [a for a in actions if values[a] == best]
            share = (1 - epsilon) / len(tied)
            eat += ("eat" in tied) * share + epsilon / len(actions)
            call += sum(share for a in tied if a.startswith("signal_")) + \
                4 * epsilon / len(actions)
        out["innate_eat"] = eat / len(genes.NEAR_CONTEXTS)
        out["innate_call"] = call / len(genes.NEAR_CONTEXTS)
    except Exception:  # noqa: BLE001 - optional detail only
        pass
    return out


class _Scan:
    """Everything read from one archive, before sampling and serialization."""

    def __init__(self):
        self.ticks = []                    # every snapshot tick in the window
        self.frames = {}                   # tick -> (agents, foods, objects) flat int lists
        self.agent_pos = {}                # tick -> {id: (x, y, energy, health)}
        self.object_pos = {}               # tick -> {id: (x, y)}
        self.population = []               # per snapshot tick
        self.ind = {}                      # id -> static record
        self.acts = defaultdict(list)      # id -> [[t0, action index, t1], ...]
        self.foods = {}                    # id -> static record
        self.objects = {}                  # id -> static record
        self.calls = {}                    # message id -> [tick, sender, symbol, range, x, y]
        self.actions = []                  # action names (index table)
        self.action_index = {}


def _action(scan, name):
    if name not in scan.action_index:
        scan.action_index[name] = len(scan.actions)
        scan.actions.append(name)
    return scan.action_index[name]


def _scan_snapshots(path, config, max_ticks, genome_config):
    scan = _Scan()
    previous = None
    for snap in _jsonl(path):
        tick = snap["tick"]
        if scan.ticks and max_ticks is not None and tick > scan.ticks[0] + max_ticks:
            break
        first = not scan.ticks
        scan.ticks.append(tick)
        agents_flat, where = [], {}
        for agent in snap.get("agents", []):
            identity = agent["id"]
            x, y = agent.get("x", agent["position"][0]), agent.get("y", agent["position"][1])
            record = scan.ind.get(identity)
            if record is None:
                anatomy = agent.get("anatomy") or {}
                record = scan.ind[identity] = {
                    "id": identity, "parent": agent.get("parent_id"), "born": agent.get("born_tick", tick),
                    "anatomy": anatomy, "lifespan": agent.get("lifespan"), "visitor": False, "presence": 0,
                    "genome": _genome_summary(agent.get("genome"), genome_config)}
            if agent.get("visitor") is True:
                record["visitor"] = True
                record["visitor_label"] = str(agent.get("visitor_label") or agent.get("name") or "visitor")[:40]
            record["presence"] += 1
            record["last"] = (tick, x, y)
            where[identity] = (x, y, agent.get("energy", 0.), agent.get("health", 0.))
            name = agent.get("last_action") or "rest"
            index = _action(scan, name)
            acted = not first and tick != record["born"]  # the first snapshot and a birth tick show no action yet
            if acted:
                runs = scan.acts[identity]
                if runs and runs[-1][1] == index and runs[-1][2] == previous:
                    runs[-1][2] = tick
                else:
                    runs.append([tick, index, tick])
            agents_flat += [identity, _i(x, 100), _i(y, 100), _i(agent.get("energy", 0.), 10),
                            _i(agent.get("health", 0.), 100), index if acted else -1]
        foods_flat = []
        for food in snap.get("foods", []):
            identity = food["id"]
            if identity not in scan.foods:
                scan.foods[identity] = {"x": food.get("x", food["position"][0]), "y": food.get("y", food["position"][1]),
                                        "capacity": food.get("capacity", 1.), "hardness": food.get("hardness", 0.),
                                        "regrowth": food.get("regrowth", 0.), "first": tick}
            scan.foods[identity]["last"] = tick
            hardness = food.get("hardness", 0.)
            opening = food.get("opening", 0.) / hardness if hardness > 0 else 0.
            foods_flat += [identity, _i(food.get("amount", 0.), 10), _i(min(1., opening), 100)]
        objects_flat, placed = [], {}
        for obj in snap.get("objects", []):
            identity = obj["id"]
            x, y = obj.get("x", obj["position"][0]), obj.get("y", obj["position"][1])
            if identity not in scan.objects:
                composition = (obj.get("properties") or {}).get("composition") or {obj.get("material"): obj.get("mass", 1.)}
                scan.objects[identity] = {"material": obj.get("material"), "mass": obj.get("mass", 1.),
                                          "composition": composition, "components": len(obj.get("components") or []),
                                          "maker": obj.get("maker_id"), "first": tick}
            holder = obj.get("holder")
            placed[identity] = (x, y)
            objects_flat += [identity, _i(x, 100), _i(y, 100), holder if holder is not None else 0]
        for message in snap.get("messages", []):
            if message["id"] not in scan.calls:
                position = message.get("position") or [0., 0.]
                scan.calls[message["id"]] = [message.get("created_tick", tick), message.get("sender"),
                                             message.get("symbol", 0), message.get("range", 0.),
                                             position[0], position[1]]
        scan.frames[tick] = (agents_flat, foods_flat, objects_flat)
        scan.agent_pos[tick] = where
        scan.object_pos[tick] = placed
        scan.population.append(len(where))
        previous = tick
    if not scan.ticks:
        raise ArchiveError(f"no snapshots in {path}")
    return scan


def _where(scan, identity, tick):
    """Last recorded position of an individual at or before ``tick`` (snapshots)."""
    index = bisect.bisect_right(scan.ticks, tick) - 1
    for back in range(index, max(-1, index - 64), -1):
        row = scan.agent_pos[scan.ticks[back]].get(identity)
        if row is not None:
            return row[0], row[1]
    record = scan.ind.get(identity)
    return (record["last"][1], record["last"][2]) if record and "last" in record else (None, None)


def _object_where(scan, identity, tick):
    index = bisect.bisect_right(scan.ticks, tick) - 1
    for back in range(index, max(-1, index - 64), -1):
        row = scan.object_pos[scan.ticks[back]].get(identity)
        if row is not None:
            return row
    return None, None


def _scan_events(path, scan, config):
    """Births, deaths, wounds, calls, hearings and interactions inside the snapshot window."""
    first, last = scan.ticks[0], scan.ticks[-1]
    out = {"births": [], "deaths": {}, "wounds": [], "heard": defaultdict(list), "visual": [],
           "blooms": {}, "call_power": {}, "counts": defaultdict(int), "capacity_pause": None}
    if not Path(path).exists():
        return out
    skip = ('"type":"transition"', '"type":"environment"', '"type":"observation"', '"type":"kin_credit"')
    reach = config.get("call_reach", 8.) or 8.
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if any(token in line for token in skip):
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                if line.endswith("\n"):
                    raise
                break
            tick, kind = event.get("tick"), event.get("type")
            if tick is None or tick < first or tick > last:
                continue
            out["counts"][kind] += 1
            if kind == "birth":
                out["births"].append((tick, event["agent"], event["parent"], event.get("partner")))
            elif kind == "death":
                out["deaths"][event["agent"]] = (tick, event.get("reason"))
            elif kind == "predation":
                out["wounds"].append((tick, event["agent"], event.get("wound", 0.), "predation", event.get("predator")))
            elif kind == "bloom":
                position = event.get("position") or [None, None]
                out["blooms"][event["food"]] = (tick, position[0], position[1])
            elif kind == "capacity_pause":
                out["capacity_pause"] = tick
            elif kind == "signal" or (kind == "action" and str(event.get("action", "")).startswith("signal_")
                                      and event.get("success") and "message_id" in event):
                sender, message = event["agent"], event["message_id"]
                if config.get("call_power") and event.get("range"):
                    out["call_power"].setdefault(sender, event["range"] / reach)
                if message not in scan.calls:
                    x, y = _where(scan, sender, tick)
                    if x is not None:
                        scan.calls[message] = [tick, sender, event.get("symbol", int(str(event.get("action", "_0"))[-1])),
                                               event.get("range", 0.), x, y]
            elif kind == "heard":
                out["heard"][event["agent"]].append((tick, event.get("sender"), event.get("symbol"),
                                                     event.get("sent_symbol", -1)))
            elif kind == "message_received":
                out["heard"][event["agent"]].append((tick, event.get("sender"), event.get("symbol"), -1))
            if kind in ("act", "action") and event.get("success") and event.get("action") in NOTABLE:
                _interaction(out, scan, event, config)
    return out


def _interaction(out, scan, event, config):
    tick, agent, action = event["tick"], event["agent"], event["action"]
    ax, ay = _where(scan, agent, tick)
    target, bx, by, value = 0, None, None, 0.
    kind = action
    if action in ("strike", "strike_agent", "strike_object"):
        target_kind = event.get("target_kind", "food")
        target = event.get("target") or 0
        value = event.get("damage", 0.)
        kind = {"agent": "strike_agent", "object": "strike_object"}.get(target_kind, "strike_food")
        if target_kind == "agent":
            bx, by = _where(scan, target, tick)
            attacker, victim = scan.ind.get(agent), scan.ind.get(target)
            per = config.get("strike_injury", .08)
            if config.get("injury_mass_scaling") and attacker and victim:
                per *= attacker["anatomy"].get("body_mass", 1.) / max(1e-9, victim["anatomy"].get("body_mass", 1.))
            out["wounds"].append((tick, target, per * value, "strike", agent))
        elif target_kind == "object":
            bx, by = _object_where(scan, target, tick)
        else:
            food = scan.foods.get(target)
            bx, by = (food["x"], food["y"]) if food else (None, None)
    elif action == "share":
        target, value = event.get("target") or 0, event.get("energy", 0.)
        bx, by = _where(scan, target, tick)
    elif action in ("teach", "imitate"):
        other = event.get("learner") if action == "teach" else event.get("teacher")
        target = other or 0
        bx, by = _where(scan, target, tick)
    elif action in ("collect", "drop"):
        target = event.get("target") or 0
    elif action in ("combine", "heat"):
        target = event.get("artifact") or 0
    elif action == "dismantle":
        target = event.get("consumed") or 0
    # row: tick, kind, agent, target, ax, ay, bx, by, value, tool (0 = bare hand / none)
    out["visual"].append([tick, EVENT_KINDS.index(kind), agent, target, *(_pos(ax, ay)), *(_pos(bx, by)),
                          _r(value, 2), event.get("tool") or 0])


def _pos(x, y):
    return (-1, -1) if x is None or y is None else (_i(x, 100), _i(y, 100))


class _ReplayMismatch(Exception):
    pass


def _replay_predators(run, manifest, scan, log):
    """Exact predator positions per snapshot tick by replaying the run (see module doc).
    Returns ({tick: [id, x100, y100, cooldown, ...]}, note)."""
    from .config import Config
    from .world import World
    resumed = manifest.get("resumed_from")
    if resumed:
        from .checkpoint import load
        source = Path(resumed.get("checkpoint", ""))
        if not source.is_file():
            return None, "resume checkpoint not found; predators shown at their strikes only"
        world, _ = load(source, check_source=False)
    else:
        if manifest.get("initial_tick", 0) != 0:
            return None, "archive does not start at tick 0; predators shown at their strikes only"
        values = dict(manifest["config"])
        values["log"] = "compact"  # the log mode changes records only, never the dynamics
        world = World.create(seed=manifest["seed"], config=Config.from_dict(values))
    if world.eco is None or not world.config.predators:
        return None, "no predators in this world"
    wanted = set(scan.ticks)
    out, started = {}, time.monotonic()
    while True:
        tick = world.tick
        if tick in wanted:
            recorded = scan.agent_pos[tick]
            if set(recorded) != set(world.agents):
                raise _ReplayMismatch(f"tick {tick}: individuals differ")
            for identity, (x, y, energy, health) in recorded.items():
                agent = world.agents[identity]
                if abs(agent.position[0] - x) > 1e-9 or abs(agent.position[1] - y) > 1e-9 \
                        or abs(agent.energy - energy) > 1e-9 or abs(agent.health - health) > 1e-9:
                    raise _ReplayMismatch(f"tick {tick}: individual {identity} differs")
            flat = []
            for predator in world.eco["predators"]:
                flat += [predator["id"], _i(predator["position"][0], 100), _i(predator["position"][1], 100),
                         predator.get("cooldown", 0)]
            out[tick] = flat
        if tick >= scan.ticks[-1] or world.stop_reason:
            break
        world.step()
        if world.tick % 200 == 0:
            log(f"  replay tick {world.tick}/{scan.ticks[-1]} ({time.monotonic() - started:.0f} s)")
    if set(out) != wanted:
        raise _ReplayMismatch("replay stopped before the last snapshot")
    return out, f"replayed from seed {manifest.get('seed')} and checked against all {len(out)} snapshots"


def _founders(scan, births):
    parent = {identity: record["parent"] for identity, record in scan.ind.items()}
    born = {identity: record["born"] for identity, record in scan.ind.items()}
    for tick, child, mother, _ in births:
        parent.setdefault(child, mother)
        born.setdefault(child, tick)
    founder, depth = {}, {}
    for identity in sorted(parent):
        chain, node = [], identity
        while node not in founder:
            up = parent.get(node)
            if up is None or up not in parent:  # a founder, or a parent from before the archive
                founder[node], depth[node] = node, 0
                break
            chain.append(node)
            node = up
        for node in reversed(chain):
            founder[node], depth[node] = founder[parent[node]], depth[parent[node]] + 1
    return parent, born, founder, depth


def _scars(wounds, config):
    """Permanent injury per individual as [tick, scar x 1000, ...] change points."""
    share = config.get("permanent_injury", 0.) or 0.
    if not share:
        return {}
    level, points = defaultdict(float), defaultdict(list)
    for tick, identity, wound, _, _ in sorted(wounds, key=lambda row: row[0]):
        if wound <= 0:
            continue
        level[identity] = min(1., level[identity] + share * wound)
        points[identity] += [tick, _i(level[identity], 1000)]
    return dict(points)


def _select(ticks, every):
    """Snapshot ticks kept for the page: one per ``every`` ticks, always the first and last."""
    kept, due = [], None
    for tick in ticks:
        if due is None or tick >= due:
            kept.append(tick)
            due = tick + every
    if kept[-1] != ticks[-1]:
        kept.append(ticks[-1])
    return kept


def _checkpoint_ecology(run):
    path = Path(run) / "checkpoint.json"
    if not path.is_file():
        return {}
    try:
        return (json.loads(path.read_text(encoding="utf-8")).get("state") or {}).get("ecology") or {}
    except (OSError, ValueError):
        return {}


def collect(run_dir, *, max_ticks=None, replay="auto", log=print):
    """Read an archive into the page's data tables (all snapshot ticks; sampling happens later)."""
    run = Path(run_dir)
    manifest = _json(run / "manifest.json")
    summary = _json(run / "summary.json") if (run / "summary.json").is_file() else {}
    config = dict(manifest.get("config") or {})
    try:
        from .config import Config
        genome_config = Config.from_dict(config)
    except Exception:  # noqa: BLE001 - genome detail falls back to raw values
        genome_config = None
    started = time.monotonic()
    scan = _scan_snapshots(run / "snapshots.jsonl", config, max_ticks, genome_config)
    log(f"snapshots: {len(scan.ticks)} ticks {scan.ticks[0]}-{scan.ticks[-1]}, {len(scan.ind)} individuals "
        f"({time.monotonic() - started:.1f} s)")
    events = _scan_events(run / "events.jsonl", scan, config)
    log(f"events: {sum(events['counts'].values())} in window ({time.monotonic() - started:.1f} s)")
    predators, predator_source, predator_note = None, "none", "no predators in this world"
    if config.get("predators") and replay != "off":
        try:
            predators, predator_note = _replay_predators(run, manifest, scan, log)
            predator_source = "replay" if predators else "events"
        except _ReplayMismatch as exc:
            predators, predator_source = None, "events"
            predator_note = f"replay did not match the archive ({exc}); predators shown at their strikes only"
        except Exception as exc:  # noqa: BLE001 - the engine may have changed or be mid-edit
            predators, predator_source = None, "events"
            predator_note = f"replay unavailable ({type(exc).__name__}: {exc}); predators shown at their strikes only"
        if replay == "on" and predators is None:
            raise ArchiveError(predator_note)
        log(f"predators: {predator_note}")
    elif config.get("predators"):
        predator_source, predator_note = "events", "replay switched off; predators shown at their strikes only"
    return {"run": run, "manifest": manifest, "summary": summary, "config": config, "scan": scan,
            "events": events, "predators": predators, "predator_source": predator_source,
            "predator_note": predator_note, "ecology": _checkpoint_ecology(run)}


def assemble(data, every=1, heard_detail=True):
    """The page's JSON payload for one sampling step."""
    scan, events, config, manifest = data["scan"], data["events"], data["config"], data["manifest"]
    parent, born, founder, depth = _founders(scan, events["births"])
    kept = _select(scan.ticks, every)
    frames = {"t": kept, "a": [scan.frames[t][0] for t in kept], "f": [scan.frames[t][1] for t in kept],
              "o": [scan.frames[t][2] for t in kept]}
    if data["predators"]:
        frames["p"] = [data["predators"][t] for t in kept]
    final_power = (data["ecology"].get("call_power") or {})
    material_index = {name: i for i, name in enumerate(MATERIALS)}
    presence = defaultdict(int)
    for tick in kept:
        for identity in scan.agent_pos[tick]:
            presence[identity] += 1
    rows = []
    for identity in sorted(set(parent) | set(scan.ind)):
        record = scan.ind.get(identity, {})
        anatomy = record.get("anatomy") or {}
        genome = record.get("genome") or {}
        death = events["deaths"].get(identity)
        power = events["call_power"].get(identity)
        if power is None and str(identity) in final_power:
            power = final_power[str(identity)]
        last = record.get("last")
        if death is not None:
            x, y = _where(scan, identity, death[0])
        elif last:
            x, y = last[1], last[2]
        else:
            x, y = _where(scan, parent.get(identity), born.get(identity, scan.ticks[0]))
        reason = REASONS.index(death[1]) if death and death[1] in REASONS else (-1 if not death else len(REASONS))
        rows.append([identity, parent.get(identity), founder.get(identity, identity), depth.get(identity, 0),
                     born.get(identity), death[0] if death else None, reason,
                     *(_r(anatomy.get(name)) for name in ("body_mass", "speed", "sensing", "manipulation", "metabolism")),
                     anatomy.get("voice"), anatomy.get("hearing"), _r(power), record.get("lifespan"),
                     _r(genome.get("alpha")), _r(genome.get("epsilon")), genome.get("bias_entries"),
                     _r(genome.get("innate_eat")), _r(genome.get("innate_call")), 1 if record.get("visitor") else 0,
                     *(_pos(x, y)), presence.get(identity, 0)])
    # lineage colour slots: the eight founders with the most individual-frames keep slots 1-8
    weight = defaultdict(int)
    for row in rows:
        weight[row[2]] += row[-1]
    ranked = sorted((f for f in weight if weight[f] > 0), key=lambda f: (-weight[f], f))
    visitors = {row[0]: scan.ind[row[0]].get("visitor_label", "visitor") for row in rows if row[21]}
    visual = list(events["visual"])  # interactions; births, deaths and predation are added here
    for death_id, (tick, reason) in events["deaths"].items():
        x, y = _where(scan, death_id, tick)
        visual.append([tick, EVENT_KINDS.index("death"), death_id, REASONS.index(reason) if reason in REASONS else 4,
                       *(_pos(x, y)), -1, -1, 0, 0])
    for tick, child, mother, partner in events["births"]:
        x, y = _where(scan, child, tick)
        if x is None:
            x, y = _where(scan, mother, tick)
        visual.append([tick, EVENT_KINDS.index("birth"), child, mother, *(_pos(x, y)), -1, -1, 0, partner or 0])
    for tick, victim, wound, cause, predator in events["wounds"]:
        if cause != "predation":
            continue
        x, y = _where(scan, victim, tick)
        px = py = None
        if data["predators"]:
            flat = data["predators"].get(tick) or []
            for i in range(0, len(flat), 4):
                if flat[i] == predator:
                    px, py = flat[i + 1] / 100, flat[i + 2] / 100
        visual.append([tick, EVENT_KINDS.index("predation"), victim, predator or 0, *(_pos(x, y)),
                       *(_pos(px, py)), _r(wound, 3), 0])
    visual.sort(key=lambda row: (row[0], row[1], row[2]))
    calls = sorted(([message, *row[:3], _r(row[3], 2), _i(row[4], 100), _i(row[5], 100)]
                    for message, row in scan.calls.items() if scan.ticks[0] <= row[0] <= scan.ticks[-1]),
                   key=lambda row: (row[1], row[0]))
    heard = {}
    for identity, items in events["heard"].items():
        items.sort()
        if heard_detail:
            heard[identity] = [value for item in items for value in
                               (item[0], item[1] or 0, item[2] if item[2] is not None else -1, item[3])]
        else:
            heard[identity] = [len(items)]
    blooms = events["blooms"]
    bloom_ttl = config.get("bloom_ttl", 25)
    foods = []
    for identity, food in sorted(scan.foods.items()):
        is_bloom = identity in blooms or (config.get("bloom_interval") and identity > config.get("food_patches", 0))
        start = blooms[identity][0] if identity in blooms else (food["first"] if is_bloom else -1)
        foods.append([identity, _i(food["x"], 100), _i(food["y"], 100), _i(food["capacity"], 10),
                      _i(food["hardness"], 100), start, start + bloom_ttl if is_bloom else -1])
    objects = []
    for identity, obj in sorted(scan.objects.items()):
        composition = [[material_index.get(name, 0), _i(mass, 100)] for name, mass in sorted((obj["composition"] or {}).items())
                       if name is not None]
        objects.append([identity, material_index.get(obj["material"], 0), _i(obj["mass"], 100), obj["components"],
                        obj["maker"] or 0, composition])
    counters = dict(sorted(((data["summary"] or {}).get("counters") or {}).items()))
    meta = {
        "title": "Haishool Living Worlds", "version": DATA_VERSION, "run": data["run"].name,
        "seed": manifest.get("seed"), "status": manifest.get("status"), "stop_reason": manifest.get("stop_reason"),
        "final_tick": manifest.get("final_tick"), "sample_every": manifest.get("sample_every"), "every": every,
        "first": scan.ticks[0], "last": scan.ticks[-1], "frames": len(kept), "snapshots": len(scan.ticks),
        "config": {key: config.get(key) for key in CONFIG_KEYS if key in config},
        "living": bool(config.get("predators") or config.get("bloom_interval") or config.get("call_power")),
        "predator_source": data["predator_source"], "predator_note": data["predator_note"],
        "heard_detail": heard_detail, "counters": counters, "capacity_pause": events["capacity_pause"],
        "lineage_slots": ranked[:8], "lineages": len(ranked), "visitors": visitors,
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "limits": ("Snapshots hold positions, energy, health, anatomy, genome and the last action of each "
                   "individual; they do not hold the ecology record. Blooms, scars, call power and lineage "
                   "are computed from events; predator positions come from a verified replay or only from "
                   "predation events. Ticks are simulation steps, not calibrated time."),
    }
    return {"meta": meta, "actions": scan.actions, "materials": list(MATERIALS), "reasons": list(REASONS),
            "kinds": list(EVENT_KINDS), "ind_columns": list(IND_COLUMNS), "ind": rows, "frames": frames,
            "foods": foods, "objects": objects, "calls": calls, "events": visual, "heard": heard,
            "acts": {identity: [v for run in runs for v in run] for identity, runs in scan.acts.items()},
            "scars": _scars(events["wounds"], config),
            "population": [scan.ticks, scan.population]}


def render(payload, template=TEMPLATE):
    text = Path(template).read_text(encoding="utf-8")
    if text.count(DATA_TOKEN) != 1:
        raise ValueError("view3d template must contain the data token exactly once")
    encoded = json.dumps(payload, allow_nan=False, separators=(",", ":"), ensure_ascii=False)
    encoded = encoded.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    return text.replace(DATA_TOKEN, encoded)


def build(run_dir, out, *, every=1, max_ticks=None, replay="auto", max_bytes=MAX_BYTES, log=print):
    """Write the page; returns a report (path, bytes, frames, every, predator source)."""
    if every < 1:
        raise ValueError("--every must be positive")
    data = collect(run_dir, max_ticks=max_ticks, replay=replay, log=log)
    step, heard_detail, html = every, True, None
    while True:
        payload = assemble(data, step, heard_detail)
        html = render(payload)
        size = len(html.encode("utf-8"))
        if size <= max_bytes:
            break
        if payload["meta"]["frames"] > 2:
            log(f"page would be {size / 1e6:.2f} MB with every={step}; sampling more sparsely")
            step *= 2
            continue
        if heard_detail:
            heard_detail = False
            continue
        raise ArchiveError(f"page cannot be kept under {max_bytes} bytes")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8", newline="\n")
    meta = payload["meta"]
    report = {"out": str(out.resolve()), "bytes": size, "megabytes": round(size / 1e6, 3), "frames": meta["frames"],
              "every": step, "first_tick": meta["first"], "last_tick": meta["last"], "individuals": len(payload["ind"]),
              "calls": len(payload["calls"]), "events": len(payload["events"]),
              "predator_source": meta["predator_source"], "predator_note": meta["predator_note"],
              "heard_detail": heard_detail}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m haishool.life8.view3d", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    make = sub.add_parser("build", help="write a self-contained 3D viewer page for a run archive")
    make.add_argument("run", type=Path, help="run archive folder (manifest.json, snapshots.jsonl, events.jsonl)")
    make.add_argument("--out", type=Path, required=True, help="HTML file to write")
    make.add_argument("--every", type=int, default=1, help="keep one snapshot per K ticks (default 1: all)")
    make.add_argument("--max-ticks", type=int, default=None, help="only the first N ticks of the archive")
    make.add_argument("--replay", choices=("auto", "on", "off"), default="auto",
                      help="replay the run to recover predator positions (auto: use it only if it matches)")
    make.add_argument("--max-mb", type=float, default=MAX_BYTES / 1e6, help="size bound in MB (default 12)")
    args = parser.parse_args(argv)
    try:
        report = build(args.run, args.out, every=args.every, max_ticks=args.max_ticks, replay=args.replay,
                       max_bytes=int(args.max_mb * 1e6), log=lambda text: print(text, file=sys.stderr, flush=True))
    except (ArchiveError, ValueError) as exc:
        parser.exit(2, f"view3d: {exc}\n")
    print(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    main()
