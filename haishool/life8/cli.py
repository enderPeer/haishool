"""Run persistent worlds, resume exact checkpoints, and inspect measured outcomes.

Usage: python -m haishool.life8 run --seed 1 --steps 600 --out runs/seed1 [--log compact]
"""
from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import platform
import time

from . import __version__
from .checkpoint import RUN_SCHEMA, digest, file_digest, load, save, source_identity, utc, write_json


def _positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def run_world(world, out, *, steps, sample_every=20, checkpoint_every=200, resumed_from=None):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    source = source_identity()
    started = time.monotonic()
    initial_tick = world.tick
    manifest = {"schema": RUN_SCHEMA, "version": __version__, "status": "running",
                "started_utc": utc(), "source": source, "python": platform.python_version(),
                "seed": world.seed, "config": world.config.to_dict(), "initial_tick": initial_tick,
                "requested_steps": steps, "sample_every": sample_every,
                "resumed_from": resumed_from,
                "limits": "Bounded artificial-life model; ticks are not calibrated years. Observed signalling is not evidence of human language."}
    write_json(out / "manifest.json", manifest)
    frames = deque(maxlen=500)
    try:
        with (out / "events.jsonl").open("w", encoding="utf-8") as events, (out / "snapshots.jsonl").open("w", encoding="utf-8") as snapshots:
            def sample():
                state = world.snapshot()
                state["metrics"] = world.summary()
                frames.append(state)
                snapshots.write(json.dumps(state, allow_nan=False, separators=(",", ":")) + "\n")
                snapshots.flush()
                write_json(out / "summary.json", world.summary())
            sample()
            save(world, out / "checkpoint.json", source=source)
            for index in range(steps):
                if world.stop_reason:
                    break
                records = world.step()
                for record in records:
                    events.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
                if (index + 1) % sample_every == 0:
                    world.validate()
                    sample()
                    events.flush()
                if (index + 1) % checkpoint_every == 0:
                    save(world, out / "checkpoint.json", source=source)
                    print(json.dumps({"tick": world.tick, "summary": world.summary()}), flush=True)
            if not frames or frames[-1].get("tick") != world.tick:
                sample()
        world.validate()
        if source_identity() != source:
            raise RuntimeError("life8 source changed during this run; output preserved but not verified as a stable-code experiment")
        receipt = save(world, out / "checkpoint.json", source=source)
        restored, _ = load(out / "checkpoint.json")
        if restored.to_dict() != world.to_dict():
            raise RuntimeError("saved checkpoint does not reproduce the final state")
        summary = world.summary()
        elapsed = time.monotonic() - started
        manifest.update(status="complete", completed_utc=utc(), final_tick=world.tick,
                        executed_steps=world.tick - initial_tick,
                        stop_reason=summary.get("stop_reason") or "step_budget",
                        wall_seconds=elapsed, ticks_per_second=(world.tick - initial_tick) / max(elapsed, 1e-9),
                        checkpoint_state_sha256=receipt["state_sha256"], checkpoint_roundtrip_verified=True)
        write_json(out / "summary.json", summary)
        write_json(out / "manifest.json", manifest)
        from .server import offline_viewer
        offline_viewer(out / "viewer.html", {"manifest": manifest, "summary": summary, "frames": list(frames)})
        manifest["files"] = {name: file_digest(out / name) for name in
                             ("checkpoint.json", "summary.json", "events.jsonl", "snapshots.jsonl", "viewer.html")}
        write_json(out / "manifest.json", manifest)
        manifest["bytes_written"] = {name: (out / name).stat().st_size for name in ("events.jsonl", "snapshots.jsonl")}
        write_json(out / "manifest.json", manifest)
        print(json.dumps({"run": str(out), "status": manifest["status"], "stop_reason": manifest["stop_reason"],
                          "tick": world.tick, "wall_seconds": round(elapsed, 3), "summary": summary}), flush=True)
        return manifest
    except BaseException as exc:
        # Keep the last atomically completed checkpoint and all flushed records.
        manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                        error=f"{type(exc).__name__}: {exc}", stopped_utc=utc())
        write_json(out / "manifest.json", manifest)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="create a new independent ecology")
    run.add_argument("--seed", type=int, default=85)
    run.add_argument("--config", type=Path)
    run.add_argument("--population", type=_positive)
    run.add_argument("--without", choices=("learning", "tools", "communication", "culture"), action="append", default=[])
    run.add_argument("--log", choices=("full", "compact"),
                     help="full: every observation/transition; compact: births, deaths, signals, tool and social events")
    resume = sub.add_parser("resume", help="continue a checkpoint without resetting learned or world state")
    resume.add_argument("checkpoint", type=Path)
    for item in (run, resume):
        item.add_argument("--steps", type=_positive, default=2000)
        item.add_argument("--out", type=Path, required=True)
        item.add_argument("--sample-every", type=_positive, default=20)
        item.add_argument("--checkpoint-every", type=_positive, default=200)
    verify = sub.add_parser("verify", help="verify checkpoint source, state and exact deserialization")
    verify.add_argument("checkpoint", type=Path)
    serve = sub.add_parser("serve", help="open a read-only local inspector for run archives")
    serve.add_argument("--root", type=Path, default=Path("runs"))
    serve.add_argument("--port", type=int, default=8654)
    export = sub.add_parser("export", help="export local observation/action/outcome examples from a verified run")
    export.add_argument("run", type=Path)
    export.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "serve":
        from .server import serve as serve_runs
        serve_runs(args.root, args.port)
        return
    if args.command == "verify":
        world, envelope = load(args.checkpoint)
        print(json.dumps({"verified": True, "tick": world.tick, "state_sha256": envelope["state_sha256"], "summary": world.summary()}))
        return
    if args.command == "export":
        from .dataset import export_transitions
        print(json.dumps(export_transitions(args.run, args.out)))
        return
    if args.command == "resume":
        world, envelope = load(args.checkpoint)
        resumed_from = {"checkpoint": str(args.checkpoint.resolve()), "state_sha256": envelope["state_sha256"]}
    else:
        from .config import Config
        from .world import World
        values = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
        if args.population is not None:
            values["population"] = args.population
        for name in args.without:
            values[name] = False
        if args.log is not None:
            values["log"] = args.log
        world = World.create(seed=args.seed, config=Config.from_dict(values))
        resumed_from = None
    run_world(world, args.out, steps=args.steps, sample_every=args.sample_every,
              checkpoint_every=args.checkpoint_every, resumed_from=resumed_from)


if __name__ == "__main__":
    main()
