"""Command line of life9 v3 (PLANET-V3-SPEC), and the experiment protocol (world3.PROVENANCE['protocol']).

    python -m haishool.life9.v3 describe --seed 781 [--source world7|synthetic|earth] [--G 32] [--patches 4]
    python -m haishool.life9.v3 run --sample RULE [--source world7|synthetic|earth] --days N --out DIR [--part K:N]
                                    [--device cpu] [--every 1] [--view-every 0] [--save-every 0] [--chunk-days 0]
                                    [--set key=value ...]
    python -m haishool.life9.v3 run --seeds 781 6 --explicit "WHY" --source world7 --days N --out DIR [...]
    python -m haishool.life9.v3 run --resume --out DIR [--chunk-days 0] [--device cpu] [--allow-device-change WHY]
    python -m haishool.life9.v3 bench [--seed 781] [--source earth] [--days 2] [--device cpu] [--set key=value ...]

``describe`` prints formation3's description of the planet and the patches World3 would draw (no spin-up), or the
seed's outcome when its chain gives no planet.

``run`` is one experiment, under the owner's rules (4 Oct 2026):

* The sample is a rule fixed before any outcome is known: ``range:LO:HI`` (every seed LO..HI, inclusive, e.g.
  ``range:0:63``), ``hash:SALT:N:LO:HI`` (the N seeds of LO..HI with the smallest sha256 of "life9-v3-sample:SALT:seed")
  or ``earth:K`` (K Earth replicates, seeds 0..K-1, source earth). ``--source`` must be given for range and hash (the
  experiment's choice; never substituted). An explicit seed list (``--seeds``) may have been picked from known
  outcomes, so it is refused unless ``--explicit WHY`` declares it a separate, not pre-registered experiment.
  ``--part K:N`` runs the K-th (0-based) of N contiguous parts of the sample (one manifest per part's DIR).
* ``DIR/manifest.json`` is written before the worlds are built, so before the first step: the sampling rule and its
  seeds, the part, the source, the full config, the days, the rules identity (``world3.rules_identity``: sha256 over the
  source of haishool/life9/v3 and haishool/life9/planet and every in-repo module they import, the config, the rule
  dataclasses, the module constants and the Earth reference), the environment and the start time. A DIR that holds a
  manifest is never reused for a new experiment. ``--set rng_seed`` (the set-up draws' salt) redraws the worlds, so
  it is refused unless ``--explicit WHY`` makes the experiment a separate, not pre-registered one.
* ``DIR/inputs.json`` pins every seed's starting conditions before the first step: the digest of its chain inputs (and
  its planet pick), or of its absent outcome, or the error that kept it from being resolved. Every resume and every
  part recomputes them and refuses a mismatch (a seed that was an error may be resolved by a rerun, which is recorded).
  A source ``synthetic`` sample is habitable by construction: the manifest and the outcomes say so.
* Every sampled seed is run and recorded: one without a star, a planet, a habitable planet, a bodies era or land is an
  outcome, never skipped or replaced; ``DIR/outcomes.json`` lists every sampled seed's outcome at the end.
* ``--resume`` continues the experiment of DIR from ``DIR/checkpoint.pt`` (or from the start when no checkpoint was
  saved; the earlier partial outputs are then kept aside, not overwritten) up to the manifest's days, and refuses to
  run under a different rules hash (a rule change is a new experiment; the checkpoint carries its own), with a
  different sample, source, config or length, or on another device or host unless ``--allow-device-change WHY``
  declares it (GPU runs are not bit-exact). A complete experiment is not stepped, saved or rewritten again.
* Errors are not outcomes and are not dropped: a seed whose chain inputs cannot be resolved, a failed set-up or day is
  written to ``DIR/errors.json`` with its cause and traceback, and ``outcomes.json`` is then incomplete. Every run and
  resume is appended to ``DIR/runs.jsonl`` with its environment, mode and days.

It writes ``DIR/config.json`` (config and set-up reports), ``DIR/reports.json`` (the reports at the end of each
invocation: per-arena extinction days, the founding, the bound days), ``DIR/summary.jsonl`` (every ``--every`` days: summary and
the worst ledger errors), ``DIR/static-SEED.json`` and ``DIR/frames-SEED.jsonl`` (one frame3 every ``--view-every``
bouts; fields and the global layer every tenth frame) for the seeds with a world, and the checkpoint every
``--save-every`` days, when ``--chunk-days`` days of this invocation are done, and at the end. ``bench`` times the
set-up and the days at the given config (default V3Config) with one world and prints seconds per day, bouts, bodies
alive and memory. ``--set`` overrides V3Config fields (e.g. ``--set veg_spin_days=200 --set capacity=2048``).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import torch

from . import formation3 as f3
from . import world3 as w3

SAMPLE_SALT_PREFIX = "life9-v3-sample"
#: what every report of a v3 experiment must state (PROTOCOL.md 'How results are reported', step 7)
KNOWN_GAPS = ["rule 3 not satisfied: the chain's state is handed over as era values, and the founders are new",
              "'no bodies era' seeds are excluded by the chain's outcome (an open owner decision; counted apart)",
              "the global biosphere spins up with the chain's air held (a set-up replenishment, reported per world)"]


class ProtocolError(SystemExit):
    """A request the experiment protocol refuses (the process exits with the reason, status 1)."""

    def __init__(self, message: str):
        super().__init__(f"protocol: {message}")


def _value(text: str):
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    low = text.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    if "," in text:
        return [_value(t) for t in text.split(",")]
    return text


def _config(args, base: w3.V3Config | None = None) -> w3.V3Config:
    """V3Config (or ``base``) with the command line's --set, --source, --G and --patches."""
    cfg = w3.V3Config.from_dict(base.to_dict()) if base is not None else w3.V3Config()
    names = {f for f in cfg.to_dict()}
    for item in getattr(args, "set", None) or []:
        key, _, raw = item.partition("=")
        key = key.strip()
        if key not in names:
            raise SystemExit(f"unknown config field {key!r} (fields: {sorted(names)})")
        setattr(cfg, key, _value(raw.strip()))
    if getattr(args, "source", None):
        cfg.source = args.source
    for key in ("G", "patches"):
        v = getattr(args, key, None)
        if v is not None:
            setattr(cfg, key, v)
    cfg.__post_init__()
    cfg.check()
    return cfg


def _json_default(o):
    if torch.is_tensor(o):
        return o.tolist()
    if isinstance(o, float) and not math.isfinite(o):
        return str(o)
    return str(o)


def _write_json(path: Path, value) -> None:
    """Write JSON atomically (a temporary file, then a replace)."""
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=1, default=_json_default), encoding="utf-8")
    os.replace(tmp, path)


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


# ============================================================================================ sampling
def sample_seeds(rule: str) -> dict:
    """The seeds of a sampling rule fixed before any outcome is known: {"rule", "kind", "seeds", "source" (None, or
    "earth" for earth:K), "pre_registered": True}.

    ``range:LO:HI``: every seed LO..HI (inclusive). ``hash:SALT:N:LO:HI``: the N seeds of LO..HI with the smallest
    sha256("life9-v3-sample:SALT:seed") (a draw no outcome can steer), in ascending order. ``earth:K``: K Earth
    replicates, seeds 0..K-1 (the seed varies only the world's draws)."""
    head, _, rest = str(rule).partition(":")
    parts = rest.split(":") if rest else []
    try:
        if head == "range" and len(parts) == 2:
            lo, hi = int(parts[0]), int(parts[1])
            if not 0 <= lo <= hi:
                raise ValueError
            return {"rule": rule, "kind": "range", "seeds": list(range(lo, hi + 1)), "source": None,
                    "pre_registered": True}
        if head == "hash" and len(parts) == 4 and parts[0]:
            salt, n, lo, hi = parts[0], int(parts[1]), int(parts[2]), int(parts[3])
            if not (0 <= lo <= hi and 1 <= n <= hi - lo + 1):
                raise ValueError

            def key(s):
                return hashlib.sha256(f"{SAMPLE_SALT_PREFIX}:{salt}:{s}".encode()).hexdigest()

            drawn = sorted(range(lo, hi + 1), key=lambda s: (key(s), s))[:n]
            return {"rule": rule, "kind": "hash", "seeds": sorted(drawn), "source": None, "pre_registered": True}
        if head == "earth" and len(parts) == 1:
            k = int(parts[0])
            if k < 1:
                raise ValueError
            return {"rule": rule, "kind": "earth", "seeds": list(range(k)), "source": "earth", "pre_registered": True}
    except ValueError:
        pass
    raise ProtocolError(f"unknown sampling rule {rule!r}: use range:LO:HI, hash:SALT:N:LO:HI or earth:K")


def _part(seeds: list, spec: str | None) -> dict:
    """The K-th (0-based) of N contiguous parts of the sampled seeds (``--part K:N``)."""
    if not spec:
        return {"index": 0, "of": 1, "seeds": list(seeds)}
    try:
        k, n = (int(v) for v in spec.split(":"))
    except ValueError:
        raise ProtocolError(f"--part is K:N, not {spec!r}") from None
    if not 0 <= k < n or n > len(seeds):
        raise ProtocolError(f"--part {spec}: K must lie in 0..N-1 and N at most the {len(seeds)} sampled seeds")
    bounds = [round(i * len(seeds) / n) for i in range(n + 1)]
    return {"index": k, "of": n, "seeds": list(seeds[bounds[k]:bounds[k + 1]])}


def sampling_of(args) -> dict:
    """The experiment's sampling from the command line, refusing an explicit seed list that is not declared a
    separate experiment, and a source the sample does not allow."""
    if getattr(args, "sample", None) and getattr(args, "seeds", None):
        raise ProtocolError("give --sample or --seeds, not both")
    if getattr(args, "sample", None):
        s = sample_seeds(args.sample)
    elif getattr(args, "seeds", None):
        if not getattr(args, "explicit", None):
            raise ProtocolError("an explicit seed list may have been chosen from known outcomes; use a sampling rule "
                                "(--sample range:LO:HI, hash:SALT:N:LO:HI or earth:K), or declare a separate, not "
                                "pre-registered experiment with --explicit WHY")
        seeds = [int(v) for v in args.seeds]
        if len(set(seeds)) != len(seeds):
            raise ProtocolError("the explicit seed list repeats a seed")
        s = {"rule": "explicit:" + ",".join(str(v) for v in seeds), "kind": "explicit", "seeds": seeds,
             "source": None, "pre_registered": False, "note": str(args.explicit)}
    else:
        raise ProtocolError("run needs a sampling rule fixed before the run: --sample range:LO:HI, "
                            "hash:SALT:N:LO:HI or earth:K")
    given = getattr(args, "source", None)
    if s["source"] is not None:
        if given is not None and given != s["source"]:
            raise ProtocolError(f"the sample {s['rule']} is of source {s['source']}, not {given}")
        source = s["source"]
    else:
        if given is None:
            raise ProtocolError("--source must be chosen by the experiment (world7, synthetic or earth); it is never "
                                "substituted")
        source = given
    s = dict(s, source=source)
    if source == "synthetic":
        # chain.synthetic_inputs draws temperate liquid-water planets with a bodies era: habitable by construction
        s["habitability_conditioned"] = True
        s["note"] = (s.get("note", "") + "; " if s.get("note") else "") + (
            "source synthetic: every planet is temperate with liquid water and a bodies era by construction, so the "
            "sample cannot give the no-star, no-planet, no-habitable or no-bodies outcomes")
    s["part"] = _part(s["seeds"], getattr(args, "part", None))
    return s


# ============================================================================================ the manifest
def environment(device: str) -> dict:
    """Where the experiment runs (the manifest's environment)."""
    env = {"python": sys.version.split()[0], "torch": torch.__version__, "platform": platform.platform(),
           "machine": platform.machine(), "host": platform.node(), "cpus": os.cpu_count(), "device": str(device),
           "cuda": torch.version.cuda, "hip": getattr(torch.version, "hip", None),
           "threads": torch.get_num_threads(), "argv": list(sys.argv)}
    try:
        import numpy
        env["numpy"] = numpy.__version__
    except ImportError:
        env["numpy"] = None
    return env


def make_manifest(sampling: dict, cfg: w3.V3Config, days: int, device: str) -> dict:
    """The experiment's manifest (written before the worlds are built): protocol, sampling and part, source, config,
    days, rules identity, environment and start time."""
    return {"protocol": w3.PROTOCOL_VERSION, "sampling": {k: v for k, v in sampling.items() if k != "part"},
            "part": sampling["part"], "source": cfg.source, "config": cfg.to_dict(), "days": int(days),
            "rules": w3.rules_identity(cfg), "environment": environment(device), "started": _now(),
            "habitability_conditioned": bool(sampling.get("habitability_conditioned", False)),
            "carry_state_between_stages": "deferred: a known gap (world3.PROVENANCE['protocol'] rule 3)",
            "known_gaps": KNOWN_GAPS}


def check_rules(manifest: dict) -> w3.V3Config:
    """The manifest's config, after checking that the rules in effect are the manifest's (a rule change is a new
    experiment)."""
    try:
        cfg = w3.V3Config.from_dict(manifest["config"])
    except ValueError as e:
        raise ProtocolError(str(e)) from None
    now = w3.rules_identity(cfg)
    was = manifest["rules"]
    if now["sha256"] != was["sha256"]:
        changed = sorted(p for p in set(now["files"]) | set(was.get("files", {}))
                         if now["files"].get(p) != was.get("files", {}).get(p))
        parts = [name for name in ("sources_sha256", "values_sha256") if now[name] != was.get(name)]
        raise ProtocolError(f"the rules changed since the manifest ({was['sha256'][:12]} -> {now['sha256'][:12]}; "
                            f"differs: {parts}; files: {changed or 'none'}): a rule change is a new experiment, so "
                            "start it in a new DIR")
    return cfg


def _trim_jsonl(path: Path, keep) -> None:
    """Keep the lines of a JSON-lines file for which keep(record) is true (a resume drops what came after the
    checkpoint, so no day is written twice)."""
    if not path.exists():
        return
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    kept = [ln for ln in lines if keep(json.loads(ln))]
    if len(kept) != len(lines):
        tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
        tmp.write_text("".join(ln + "\n" for ln in kept), encoding="utf-8")
        os.replace(tmp, path)


def _read_json(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def record_error(out: Path, stage: str, exc: BaseException | None = None, *, seed=None, day=None, cause=None,
                 trace=None) -> dict:
    """Append an error to ``DIR/errors.json`` (seed, stage, day, cause, traceback, time): errors are not outcomes
    and are not dropped (PROTOCOL.md rule 1)."""
    import traceback
    row = {"seed": seed, "stage": stage, "day": day, "time": _now(),
           "cause": cause if cause is not None else (f"{type(exc).__name__}: {exc}" if exc is not None else None),
           "traceback": trace if trace is not None else (
               "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)) if exc is not None else None)}
    rows = _read_json(out / "errors.json", [])
    rows.append(row)
    _write_json(out / "errors.json", rows)
    return row


def write_outcomes(out: Path, manifest: dict, world: w3.World3) -> dict:
    """``DIR/outcomes.json``: every sampled seed of the part with its outcome (world3.World3.outcomes), the errors
    (an experiment with errors is incomplete), the exclusions counted apart and the known gaps."""
    rows = world.outcomes()
    errors = _read_json(out / "errors.json", [])
    days_done = world.day >= manifest["days"]
    doc = {"protocol": w3.PROTOCOL_VERSION, "rules_sha256": manifest["rules"]["sha256"],
           "sampling": manifest["sampling"], "part": manifest["part"], "source": manifest["source"],
           "days_planned": manifest["days"], "days_run": world.day, "days_complete": days_done,
           "complete": days_done and not errors, "errors": errors, "day_base": "days_completed",
           "finished": _now(), "capacity_bound_day": world.reports.get("capacity_bound_day"),
           "items_bound_day": world.reports.get("items_bound_day"),
           "fires_bound_day": world.reports.get("fires_bound_day"),
           "excluded_by_chain_outcome": [r["seed"] for r in rows if r.get("excluded")],
           "habitability_conditioned": bool(manifest.get("habitability_conditioned", False)),
           "known_gaps": manifest.get("known_gaps", KNOWN_GAPS),
           "state_mb": world.reports.get("state_mb"), "outcomes": rows}
    _write_json(out / "outcomes.json", doc)
    return doc


def write_reports(out: Path, world: w3.World3) -> None:
    """``DIR/reports.json``: the world's reports now (per-arena extinction days, the founding, the bound days, the
    spin-ups and the set-up), so they reach a JSON output, not only the checkpoint."""
    _write_json(out / "reports.json", {"day": world.day, "day_base": "days_completed", "reports": world.reports})


def pin_inputs(out: Path, manifest: dict, seeds: list, resolved: list) -> list:
    """Write or check ``DIR/inputs.json``, every seed's starting conditions pinned before the first step
    (world3.resolved_digest); a mismatch is refused, except a seed pinned as an error that now resolves (a rerun,
    returned so the caller records it)."""
    pinned = [w3.resolved_digest(sd, r) for sd, r in zip(seeds, resolved)]
    path = out / "inputs.json"
    was = _read_json(path, None)
    if was is None:
        _write_json(path, {"protocol": w3.PROTOCOL_VERSION, "rules_sha256": manifest["rules"]["sha256"],
                           "pinned": _now(), "seeds": pinned})
        return []
    old = was["seeds"]
    if [r["seed"] for r in old] != [r["seed"] for r in pinned]:
        raise ProtocolError("inputs.json pins other seeds: this is another experiment")
    reruns, bad = [], []
    for a, b in zip(old, pinned):
        if a == b:
            continue
        if a.get("kind") == w3.ERROR_KIND:
            reruns.append({"seed": b["seed"], "was": a, "now": b})
        else:
            bad.append(b["seed"])
    if bad:
        raise ProtocolError(f"the starting conditions of seeds {bad} differ from those pinned in {path} (another chain "
                            "cache?): this is another experiment")
    if reruns:
        _write_json(path, dict(was, seeds=pinned, reruns=was.get("reruns", []) + [dict(r, time=_now())
                                                                                   for r in reruns]))
    return reruns


# ============================================================================================ commands
def cmd_describe(args) -> int:
    cfg = _config(args)
    got = w3.choose_patches(cfg, [args.seed])
    for r in got["absent"]:
        print(f"seed {r['seed']}: {r['outcome']} ({r['note']})")
    if not got["specs"]:
        return 0
    spec = got["specs"][0]
    print(f3.describe3(spec))
    fails = f3.check_spec3(spec)
    print(f"check_spec3: {'ok' if not fails else '; '.join(fails)}")
    globe, terrain = got["globe"], got["terrain"]
    where = "over all cells: no land, so they are sea" if got["no_land"][0] else "over land"
    print(f"patches of seed {args.seed} ({cfg.patches} at random by area {where}, G = {cfg.G}):")
    for p, c in enumerate(got["cells"][0]):
        x, y, z = globe.centers[c].tolist()
        lat, lon = math.degrees(math.asin(max(-1.0, min(1.0, z)))), math.degrees(math.atan2(y, x))
        h = float(terrain["elevation_m"][0, c] - terrain["sea_level_m"][0])
        print(f"  {p}: cell {c}, lat {lat:+.1f}, lon {lon:+.1f}, {h:.0f} m above the sea")
    return 0


def _worst(ledgers: dict) -> dict:
    return {k: v["rel"] for k, v in ledgers.items() if isinstance(v, dict)}


def _resume_manifest(args, man_path: Path) -> tuple[dict, w3.V3Config]:
    """The manifest of the experiment to resume, after the protocol's checks."""
    if not man_path.exists():
        raise ProtocolError(f"--resume needs {man_path} (the experiment's manifest)")
    manifest = json.loads(man_path.read_text(encoding="utf-8"))
    if manifest.get("protocol") != w3.PROTOCOL_VERSION:
        raise ProtocolError(f"manifest protocol {manifest.get('protocol')!r}, expected {w3.PROTOCOL_VERSION!r}")
    cfg = check_rules(manifest)
    if args.source and args.source != manifest["source"]:
        raise ProtocolError(f"the experiment's source is {manifest['source']}, not {args.source}")
    if args.sample or args.seeds:
        if not args.source:
            args.source = manifest["source"]                 # the manifest holds it
        s = sampling_of(args)
        if s["rule"] != manifest["sampling"]["rule"] or s["part"] != manifest["part"] \
                or s["source"] != manifest["source"]:
            raise ProtocolError("--resume continues the manifest's sample, part and source; this is another experiment")
    if args.set:
        try:
            given = _config(args, base=cfg)
        except ValueError as e:
            raise ProtocolError(f"--resume runs the manifest's config ({e})") from None
        if given.to_dict() != cfg.to_dict():
            raise ProtocolError("--resume runs the manifest's config; a changed config is a new experiment")
    if args.days is not None and int(args.days) != int(manifest["days"]):
        raise ProtocolError(f"the experiment's length is {manifest['days']} days (its manifest), not {args.days}")
    return manifest, cfg


def _last_run(out: Path) -> dict | None:
    path = out / "runs.jsonl"
    if not path.exists():
        return None
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return json.loads(lines[-1]) if lines else None


def _append_run(out: Path, row: dict) -> None:
    with open(out / "runs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=_json_default) + "\n")


def _check_device(out: Path, manifest: dict, args) -> dict | None:
    """A resume on another device or host is refused unless declared (GPU runs are not bit-exact); a declared change
    is returned for the run record."""
    last = _last_run(out) or {"environment": manifest.get("environment", {})}
    env = last.get("environment", {})
    now = environment(args.device)
    changed = {k: [env.get(k), now[k]] for k in ("device", "host") if env.get(k) != now[k]}
    if changed and not getattr(args, "allow_device_change", None):
        raise ProtocolError(f"the experiment last ran on {changed}: a resume on another device or host must be declared "
                            "with --allow-device-change WHY (GPU runs are not bit-exact)")
    return {"changed": changed, "why": args.allow_device_change} if changed else None


def _set_aside(out: Path, names: list) -> list:
    """Keep the earlier partial outputs of a resume without a checkpoint (renamed, never overwritten)."""
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%S")
    kept = []
    for name in names:
        src = out / name
        if src.exists():
            dst = out / f"{name}.discarded-{stamp}"
            os.replace(src, dst)
            kept.append(dst.name)
    return kept


def cmd_run(args) -> int:
    out = Path(args.out)
    man_path = out / "manifest.json"
    ck = out / "checkpoint.pt"
    t0 = time.time()
    run = {"started": _now(), "environment": environment(args.device), "argv": list(sys.argv)}
    if args.resume:
        manifest, cfg = _resume_manifest(args, man_path)
        run["device_change"] = _check_device(out, manifest, args)
    else:
        if man_path.exists():
            raise ProtocolError(f"{out} already holds an experiment (manifest.json): --resume it, or use a new DIR")
        if args.days is None:
            raise ProtocolError("--days (the experiment's length) is part of the manifest")
        sampling = sampling_of(args)
        args.source = sampling["source"]
        cfg = _config(args)
        if cfg.rng_seed is not None:
            # the set-up draws' salt redraws every world: a retry under the same sampling rule (world3.SETUP_SALT)
            if not getattr(args, "explicit", None):
                raise ProtocolError("--set rng_seed redraws the worlds of the sample; only a declared, not "
                                    "pre-registered experiment (--explicit WHY) may set it")
            sampling = dict(sampling, pre_registered=False,
                            note=(sampling.get("note", "") + "; " if sampling.get("note") else "")
                            + f"rng_seed {cfg.rng_seed}: {args.explicit}")
        out.mkdir(parents=True, exist_ok=True)              # after every check: a refused call leaves no DIR
        manifest = make_manifest(sampling, cfg, args.days, args.device)
        _write_json(man_path, manifest)                    # before the worlds are built, so before the first step
    seeds = list(manifest["part"]["seeds"])
    target = int(manifest["days"])
    log = lambda m: print(m, flush=True)                   # noqa: E731
    # the starting conditions, pinned before the first step (and checked on every resume and part)
    resolved = w3.resolve_inputs(seeds, cfg)
    run["reruns"] = pin_inputs(out, manifest, seeds, resolved)
    known = {(e.get("seed"), e.get("stage")) for e in _read_json(out / "errors.json", [])}
    for sd, (x, note) in zip(seeds, resolved):
        if x is None and note.get("kind") == w3.ERROR_KIND and (sd, "inputs") not in known:
            record_error(out, "inputs", seed=sd, cause=note.get("cause"), trace=note.get("traceback"))
    have_ck = args.resume and ck.exists()
    if have_ck:
        try:
            world = w3.World3.load(ck, args.device, expect_rules=manifest["rules"]["sha256"])
        except ValueError as e:
            raise ProtocolError(f"{ck}: {e}") from None
        if world.config.to_dict() != cfg.to_dict() or world.sampled != seeds:
            raise ProtocolError(f"{ck} is not this experiment's checkpoint (config or seeds differ from the manifest)")
        run["mode"] = "resume"
        print(f"resumed day {world.day} from {ck}", flush=True)
        if world.day >= target and (out / "outcomes.json").exists():
            run.update(day_from=world.day, day_to=world.day, finished=_now(), note="already complete: nothing run")
            _append_run(out, run)
            print(f"the experiment is complete ({world.day} of {target} days): nothing to run", flush=True)
            return 0
        _trim_jsonl(out / "summary.jsonl", lambda r: r.get("day", 0) <= world.day)
        for seed in world.seeds:
            _trim_jsonl(out / f"frames-{seed}.jsonl", lambda r: r.get("day", 0) < world.day)
    else:
        if args.resume:                                   # no checkpoint: start over, keeping what came before
            run["mode"] = "resume_from_start"
            run["set_aside"] = _set_aside(out, ["summary.jsonl", "config.json", "outcomes.json", "reports.json"]
                                          + [p.name for p in sorted(out.glob("frames-*.jsonl"))])
        else:
            run["mode"] = "fresh"
        try:
            world = w3.World3(cfg, seeds, args.device, log=log, resolved=resolved)
        except Exception as e:                            # noqa: BLE001
            record_error(out, "set-up", e)
            _append_run(out, dict(run, finished=_now(), error=f"{type(e).__name__}: {e}"))
            raise
        _write_json(out / "config.json", {"config": cfg.to_dict(), "sampled": world.sampled, "seeds": world.seeds,
                                          "reports": world.reports})
        for w, seed in enumerate(world.seeds):
            (out / f"static-{seed}.json").write_text(json.dumps(world.static3(w), default=_json_default))
    run["day_from"] = world.day
    absent = ", ".join(f"{r['seed']} ({r['outcome']})" for r in world.absent) or "none"
    print(f"set-up {time.time() - t0:.1f} s; worlds {world.W}; arenas {world.A}; state {world.reports.get('state_mb')} "
          f"MB; no world: {absent}; no land (sea patches): {world.reports.get('no_land')}", flush=True)
    mode = "w" if run["mode"] != "resume" else "a"
    frames = {seed: open(out / f"frames-{seed}.jsonl", mode, encoding="utf-8") for seed in world.seeds}
    count = {"n": 0}

    def on_bout(wd, k):
        if args.view_every and (wd.day * wd.config.bouts + k) % args.view_every == 0:
            rich = count["n"] % 10 == 0
            for w, seed in enumerate(wd.seeds):
                frames[seed].write(json.dumps(wd.frame3(w, fields=rich, global_fields=rich)) + "\n")
            count["n"] += 1

    summary = open(out / "summary.jsonl", mode, encoding="utf-8")
    stop = target if not args.chunk_days else min(target, world.day + int(args.chunk_days))
    stepped = False
    try:
        while world.day < stop:
            t = time.time()
            try:
                world.step_day(on_bout)
            except Exception as e:                        # noqa: BLE001
                record_error(out, "day", e, day=world.day)
                raise
            stepped = True
            if args.every and world.day % args.every == 0:
                s = world.summary()
                s["seconds"] = round(time.time() - t, 3)
                s["ledgers"] = _worst(world.ledgers())
                summary.write(json.dumps(s, default=_json_default) + "\n")
                summary.flush()
                print(f"day {world.day}: population {s.get('population')} ({s['seconds']} s)", flush=True)
                if s.get("capacity_bound"):
                    print(f"  capacity bound since day {s.get('capacity_bound_day')} (days completed): births found no "
                          "free slot (a numerical limit, reported; world3.PROVENANCE['capacity_bound'])", flush=True)
            if args.save_every and world.day % args.save_every == 0:
                world.save(ck)
        if stepped or not ck.exists():
            world.save(ck)
    except Exception as e:                                # noqa: BLE001
        _append_run(out, dict(run, day_to=world.day, finished=_now(), error=f"{type(e).__name__}: {e}"))
        try:
            write_outcomes(out, manifest, world)
            write_reports(out, world)
        except Exception:                                 # noqa: BLE001  (the error is already recorded)
            pass
        raise
    finally:
        summary.close()
        for f in frames.values():
            f.close()
    write_reports(out, world)
    run.update(day_to=world.day, finished=_now())
    _append_run(out, run)
    if world.day >= target:
        doc = write_outcomes(out, manifest, world)
        for r in doc["outcomes"]:
            print(f"seed {r['seed']}: {r['outcome']}", flush=True)
        if doc["errors"]:
            print(f"{len(doc['errors'])} error(s) recorded in {out / 'errors.json'}: the experiment is incomplete",
                  flush=True)
    else:
        print(f"stopped at day {world.day} of {target}: --resume continues", flush=True)
    return 0


def cmd_bench(args) -> int:
    cfg = _config(args)
    t0 = time.time()
    world = w3.World3(cfg, [args.seed], args.device, log=lambda m: print(m, flush=True))
    setup = time.time() - t0
    if not world.W:
        print(json.dumps({"seed": args.seed, "outcome": world.absent[0]["outcome"], "note": world.absent[0]["note"]}))
        return 0
    print(f"set-up {setup:.1f} s; arenas {world.A}; bodies {int(world.b.alive.sum()) if world.A else 0}", flush=True)
    rows = []
    for d in range(args.days):
        t = time.time()
        world.step_day()
        dt = time.time() - t
        alive = int(world.b.alive.sum()) if world.A else 0
        rows.append(dt)
        print(f"day {d}: {dt:.2f} s, bodies alive {alive}", flush=True)
    rss = w3.peak_rss_bytes()
    report = {"seed": args.seed, "source": cfg.source, "device": str(world.device), "setup_s": round(setup, 2),
              "seconds_per_day": [round(v, 3) for v in rows],
              "mean_s_per_day": round(sum(rows) / max(len(rows), 1), 3), "bouts": cfg.bouts, "arenas": world.A,
              "capacity": cfg.capacity, "slots": world.A * cfg.capacity,
              "bodies_alive": int(world.b.alive.sum()) if world.A else 0,
              "state_mb": round(world.memory_bytes() / 1e6, 1),
              "bodies_state_mb": round(w3._tensor_bytes(world.b.state_dict()) / 1e6, 1),
              "capacity_bound": bool(world.A) and "capacity_bound_day" in world.reports,
              "outcome": world.outcomes()[0]["outcome"], "threads": torch.get_num_threads(),
              "peak_rss_mb": None if rss is None else round(rss / 1e6, 1), "ledgers_worst": world.ledgers()["worst"]}
    if torch.cuda.is_available() and world.device.type == "cuda":
        report["cuda_peak_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
    print(json.dumps(report))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m haishool.life9.v3", description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("describe", help="the planet and its patches (no spin-up)")
    d.add_argument("--seed", type=int, default=781)
    d.add_argument("--source", choices=w3.SOURCES)
    d.add_argument("--G", type=int)
    d.add_argument("--patches", type=int)
    d.add_argument("--set", action="append", default=[])
    r = sub.add_parser("run", help="one experiment: build or resume its worlds and step them")
    r.add_argument("--sample", help="range:LO:HI, hash:SALT:N:LO:HI or earth:K (fixed before the run)")
    r.add_argument("--seeds", type=int, nargs="+", help="an explicit seed list (needs --explicit)")
    r.add_argument("--explicit", help="why an explicit seed list: a separate, not pre-registered experiment")
    r.add_argument("--part", help="K:N, the K-th (0-based) of N contiguous parts of the sample")
    r.add_argument("--days", type=int, help="the experiment's length in days (in the manifest)")
    r.add_argument("--device", default="cpu")
    r.add_argument("--out", required=True)
    r.add_argument("--every", type=int, default=1)
    r.add_argument("--view-every", type=int, default=0)
    r.add_argument("--save-every", type=int, default=0)
    r.add_argument("--chunk-days", type=int, default=0, help="step at most this many days now, save, and stop")
    r.add_argument("--resume", action="store_true")
    r.add_argument("--allow-device-change", help="why a resume runs on another device or host (recorded)")
    r.add_argument("--source", choices=w3.SOURCES)
    r.add_argument("--set", action="append", default=[])
    b = sub.add_parser("bench", help="seconds per day at a config")
    b.add_argument("--seed", type=int, default=781)
    b.add_argument("--days", type=int, default=2)
    b.add_argument("--device", default="cpu")
    b.add_argument("--source", choices=w3.SOURCES)
    b.add_argument("--set", action="append", default=[])
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return {"describe": cmd_describe, "run": cmd_run, "bench": cmd_bench}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
