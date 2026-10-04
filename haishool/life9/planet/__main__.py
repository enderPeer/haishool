"""life9 planet engine command line (PLANET-SPEC section 2.12).

    python -m haishool.life9.planet describe --seed 781 [--source auto]
    python -m haishool.life9.planet run --seeds 781,85 --days 365 --device cpu --out runs/life9/planet/a
                                        [--every 10] [--view-every 2 --view-days 200 --view-world 0]
                                        [--resume runs/life9/planet/a/state.pt] [--G 48 --capacity 1024 ...]
    python -m haishool.life9.planet bench --worlds 2 --capacity 1024 --days 5 --device cpu [--G 48] [--quick]

``describe`` prints the PlanetSpec of a seed with every field's provenance, the chain inputs' own
provenance and the spec checks. ``run`` writes run.json (config, specs, environment, set-up
reports, provenance), summary.jsonl (``world.summary()`` plus ``world.ledgers()`` every --every
days; a resume drops the rows after its day), state.pt (written atomically, resumable with --resume),
gates.json (``world.gates()`` at the end) and, with --view-every, view.html (``view.build_page``,
when that module is present; also view-static.json and view-frames.jsonl, which ``python -m
haishool.life9.planet.view`` turns into a page). ``bench`` times construction and days and reports
memory.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

from . import formation
from .world import PlanetConfig, PlanetWorld, planet_inputs

#: command-line options that set PlanetConfig fields (option name -> (field, type))
CONFIG_OPTIONS = {"G": int, "capacity": int, "founders": int, "items": int, "fires": int, "hidden": int,
                  "initial_hidden": int, "habitat_radius_m": float, "source": str, "rng_seed": int,
                  "bio_spin_days": int, "climate_fast_chunks": int, "climate_slow_chunks": int,
                  "fire_substeps": int, "weight_dtype": str, "founder_mass_kg": float}


def peak_rss_bytes() -> int | None:
    """Peak resident memory of this process (bytes), or None when the platform does not say."""
    try:
        import resource
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(peak if sys.platform == "darwin" else peak * 1024)
    except ImportError:
        pass
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        c = Counters()
        c.cb = ctypes.sizeof(Counters)
        kernel, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = (wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD)
        psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
        if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(c), c.cb):
            return int(c.PeakWorkingSetSize)
    return None


def environment(device) -> dict:
    info = {"python": platform.python_version(), "torch": torch.__version__, "platform": platform.platform(),
            "device": str(device), "threads": torch.get_num_threads()}
    if str(device).startswith("cuda") and torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(torch.device(device))
        info["hip"] = getattr(torch.version, "hip", None)
        info["cuda"] = torch.version.cuda
    try:
        info["git"] = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                     cwd=Path(__file__).resolve().parent, timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        info["git"] = None
    return info


def _sync(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def config_from(args, **extra) -> PlanetConfig:
    fields = {}
    for name in CONFIG_OPTIONS:
        value = getattr(args, name, None)
        if value is not None:
            fields[name] = value
    for item in getattr(args, "set", None) or []:
        key, _, value = item.partition("=")
        try:
            fields[key] = json.loads(value)
        except ValueError:
            fields[key] = value
    fields.update(extra)
    return PlanetConfig(**fields)


def parse_seeds(text: str) -> list[int]:
    return [int(s) for s in str(text).replace(" ", "").split(",") if s]


# ------------------------------------------------------------------------------------------- describe
def cmd_describe(args):
    inputs, note = planet_inputs(args.seed, args.source)
    spec = formation.build_planet(inputs, args.seed)
    print(f"# chain inputs of seed {args.seed}: {note} (reached {inputs.get('reached')}, "
          f"version {inputs.get('chain_version')})")
    for key, where in sorted((inputs.get("provenance") or {}).items()):
        print(f"#   {key}: {where}")
    print(formation.describe(spec))
    bad = formation.check_spec(spec)
    print("check_spec: " + ("passes" if not bad else "; ".join(bad)))


# ------------------------------------------------------------------------------------------- run
def write_view(out: Path, static: dict, frames: list) -> dict | None:
    (out / "view-static.json").write_text(json.dumps(static, separators=(",", ":")), encoding="utf-8")
    with (out / "view-frames.jsonl").open("w", encoding="utf-8") as f:
        for fr in frames:
            f.write(json.dumps(fr, separators=(",", ":")) + "\n")
    try:
        from . import view                                  # written by the viewer's owner; optional
    except ImportError as e:
        print(f"no view module ({e}); wrote view-static.json and view-frames.jsonl only")
        return None
    return view.build_page(static, frames, out / "view.html")


def save_state(world, path: Path):
    """torch.save the world's state atomically: a temporary file next to ``path``, then os.replace, so a job
    killed during the write leaves the previous checkpoint intact."""
    tmp = path.with_name(path.name + ".tmp")
    torch.save(world.state_dict(), tmp)
    os.replace(tmp, path)


def trim_summary(path: Path, day: int):
    """Drop the rows of summary.jsonl written after ``day`` (a resume repeats those days)."""
    if not path.exists():
        return
    keep = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            if int(json.loads(line).get("day", 0)) <= day:
                keep.append(line + "\n")
        except ValueError:
            continue
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(keep), encoding="utf-8")
    os.replace(tmp, path)


def cmd_run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=True)
        world = PlanetWorld.from_state(state, args.device, own=True)
        del state                                           # the world holds the tensors now: one copy in RAM
        trim_summary(out / "summary.jsonl", world.day)
    else:
        world = PlanetWorld(config_from(args), parse_seeds(args.seeds), args.device)
    built = time.time() - start
    run = {"schema": "life9-planet-run-v1", "seeds": world.seeds, "config": world.config.to_dict(),
           "inputs": world.input_notes, "specs": [s.to_dict() for s in world.specs],
           "spec_failures": world.spec_failures, "environment": environment(args.device),
           "reports": world.reports, "resumed_from": args.resume, "start_day": world.day,
           "build_seconds": round(built, 2), "command": sys.argv, "provenance": world.provenance()}
    name = "run.json" if not args.resume else f"run-resume-day{world.day}.json"
    (out / name).write_text(json.dumps(run, indent=1, default=str), encoding="utf-8")
    print(f"built {world.W} worlds (seeds {world.seeds}) in {built:.1f}s; day {world.day}", flush=True)
    frames = []
    w_view = args.view_world
    view_until = world.day + (args.view_days if args.view_days else args.days)
    t0 = time.time()
    with (out / "summary.jsonl").open("a", encoding="utf-8") as log:
        for i in range(args.days):
            if args.view_every and world.day < view_until and (world.day % args.view_every == 0):
                fields = len(frames) % max(1, args.field_every) == 0
                frames.append(world.frame(w_view, fields=fields))
            world.step()
            if world.day % args.every == 0 or i == args.days - 1:
                row = world.summary()
                row["ledgers"] = world.ledgers()
                row["seconds"] = round(time.time() - t0, 2)
                log.write(json.dumps(row) + "\n")
                log.flush()
                pops = [r["population"] for r in row["worlds"]]
                worst = world.worst_ledger(row["ledgers"])
                print(f"day {world.day}: population {pops}, worst checked ledger rel {worst:.2e} "
                      f"({row['seconds']}s)", flush=True)
            if args.save_every and world.day % args.save_every == 0:
                save_state(world, out / "state.pt")
    if args.view_every and world.day < view_until and world.day % args.view_every == 0:
        frames.append(world.frame(w_view, fields=True))
    save_state(world, out / "state.pt")
    (out / "gates.json").write_text(json.dumps(world.gates(), indent=1), encoding="utf-8")
    print(f"wrote {out / 'state.pt'} (day {world.day})")
    if frames:
        report = write_view(out, world.static(w_view), frames)
        if report is not None:
            print(f"wrote {out / 'view.html'} ({len(frames)} frames): {json.dumps(report, default=str)[:300]}")


# ------------------------------------------------------------------------------------------- bench
def cmd_bench(args):
    seeds = parse_seeds(args.seeds) if args.seeds else [781] * args.worlds
    if len(seeds) < args.worlds:
        seeds = (seeds * args.worlds)[:args.worlds]
    extra = {}
    if args.quick:
        extra = {"climate_fast_chunks": 2, "climate_slow_chunks": 1, "bio_spin_days": 60}
    cfg = config_from(args, **extra)
    t = time.time()
    world = PlanetWorld(cfg, seeds[:args.worlds], args.device)
    _sync(args.device)
    built = time.time() - t
    world.profile = True
    times = []
    for _ in range(args.days):
        t = time.time()
        world.step()
        _sync(args.device)
        times.append(time.time() - t)
    report = {"environment": environment(args.device), "worlds": world.W, "seeds": world.seeds,
              "G": cfg.G, "cells": world.globe.C, "capacity": cfg.capacity, "founders": cfg.founders,
              "items": cfg.items, "fires": cfg.fires, "hidden": cfg.hidden, "quick_spin_up": bool(args.quick),
              "build_seconds": round(built, 2), "days": args.days,
              "seconds_per_day": [round(x, 3) for x in times],
              "mean_seconds_per_day": round(sum(times) / max(1, len(times)), 3),
              "mean_seconds_per_day_after_first": round(sum(times[1:]) / max(1, len(times) - 1), 3),
              "phase_seconds_per_day": {k: round(v / max(1, args.days), 4)
                                        for k, v in getattr(world, "phase_seconds", {}).items()},
              "population": [int(x) for x in world.cr.alive.sum(1)],
              "state_tensor_mb": round(world.tensor_bytes() / 1e6, 1)}
    peak = peak_rss_bytes()
    report["peak_rss_mb"] = round(peak / 1e6, 1) if peak else None
    if str(args.device).startswith("cuda") and torch.cuda.is_available():
        report["cuda_max_allocated_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
    print(json.dumps(report, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m haishool.life9.planet")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("describe", help="print a seed's PlanetSpec with provenance")
    p.add_argument("--seed", type=int, default=781)
    p.add_argument("--source", default="auto", choices=("auto", "world7", "synthetic"))
    for name in ("run", "bench"):
        p = sub.add_parser(name)
        p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        p.add_argument("--days", type=int, default={"run": 100, "bench": 5}[name])
        for opt, typ in CONFIG_OPTIONS.items():
            p.add_argument("--" + opt.replace("_", "-"), dest=opt, type=typ, default=None)
        p.add_argument("--set", action="append", help="PlanetConfig field=value (JSON value)")
        if name == "run":
            p.add_argument("--seeds", default="781,85")
            p.add_argument("--out", required=True)
            p.add_argument("--every", type=int, default=10)
            p.add_argument("--view-every", type=int, default=0)
            p.add_argument("--view-days", type=int, default=0, help="record frames for this many days (0: all)")
            p.add_argument("--view-world", type=int, default=0)
            p.add_argument("--field-every", type=int, default=10, help="cell fields in one frame of this many")
            p.add_argument("--save-every", type=int, default=0)
            p.add_argument("--resume")
        else:
            p.add_argument("--worlds", type=int, default=2)
            p.add_argument("--seeds", default=None)
            p.add_argument("--quick", action="store_true", help="short spin-ups (construction time only)")
            p.add_argument("--out")
    args = parser.parse_args(argv)
    if args.command == "bench" and args.capacity is None:
        args.capacity = 1024
    {"describe": cmd_describe, "run": cmd_run, "bench": cmd_bench}[args.command](args)


if __name__ == "__main__":
    main()
