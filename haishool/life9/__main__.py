"""life9 command line.

    python -m haishool.life9 run   --seed 1 --ticks 2000 --worlds 16 --device cuda --out runs/life9/s1 [--view-every 2 --view-ticks 600]
    python -m haishool.life9 bench --worlds 64 --capacity 256 --ticks 50 --device cuda
    python -m haishool.life9 probe --seed 1 --ticks 3000 --worlds 32 --device cuda --out gate.json

``run`` writes summary.jsonl (every --every ticks), state.pt (resumable with --resume) and,
with --view-every, view.html: a 3D replay of world 0 on its real terrain.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch

from . import probe
from .config import Config9
from .world import World9

TEMPLATE = Path(__file__).with_name("viewer_template.html")


def config_from(args):
    fields = {"worlds": args.worlds, "capacity": args.capacity}
    for item in args.option or []:
        key, _, value = item.partition("=")
        fields[key] = json.loads(value) if value[:1] in "0123456789-[{tf" else value
    return Config9(**fields)


def environment(device):
    info = {"python": platform.python_version(), "torch": torch.__version__, "device": str(device)}
    if str(device).startswith("cuda") and torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(torch.device(device))
        info["hip"] = getattr(torch.version, "hip", None)
    return info


def write_view(path, world, frames):
    terrain = [[round(v, 2) for v in row] for row in world.terrain[0].tolist()]
    data = {"size": world.cfg.size, "relief": world.cfg.relief, "terrain": terrain,
            "config": world.cfg.to_dict(), "seed": world.seed, "frames": frames}
    page = TEMPLATE.read_text(encoding="utf-8").replace("/*LIFE9_DATA*/null", json.dumps(data, separators=(",", ":")))
    Path(path).write_text(page, encoding="utf-8")


def cmd_run(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.resume:
        world = World9.from_state(torch.load(args.resume, map_location=args.device, weights_only=False), args.device)
    else:
        world = World9(config_from(args), args.seed, args.device)
    (out / "run.json").write_text(json.dumps({"schema": "life9-run-v1", "seed": world.seed,
                                              "config": world.cfg.to_dict(), "environment": environment(args.device),
                                              "resumed_from": args.resume}, indent=1), encoding="utf-8")
    frames, start = [], time.time()
    with (out / "summary.jsonl").open("a", encoding="utf-8") as log:
        for _ in range(args.ticks):
            if args.view_every and world.tick % args.view_every == 0 and len(frames) * args.view_every < args.view_ticks:
                frames.append(world.frame(0))
            world.step()
            if world.tick % args.every == 0:
                row = world.summary()
                row["seconds"] = round(time.time() - start, 2)
                row["ledger_error"] = world.ledger_error().tolist()
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(f"tick {world.tick}: alive {row['alive']} ({row['seconds']}s)", flush=True)
            if not world.alive.any():
                break
    torch.save(world.state_dict(), out / "state.pt")
    if frames:
        write_view(out / "view.html", world, frames)
        print(f"wrote {out / 'view.html'} ({len(frames)} frames)")


def cmd_bench(args):
    world = World9(config_from(args), args.seed, args.device)
    world.step()
    if world.device.type == "cuda":
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(args.ticks):
        world.step()
    if world.device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.time() - start
    slots = world.cfg.worlds * world.cfg.capacity
    print(json.dumps({"environment": environment(args.device), "worlds": world.cfg.worlds,
                      "capacity": world.cfg.capacity, "ticks": args.ticks, "seconds": round(seconds, 3),
                      "world_ticks_per_hour": round(world.cfg.worlds * args.ticks / seconds * 3600),
                      "slot_ticks_per_second": round(slots * args.ticks / seconds),
                      "alive": int(world.alive.sum())}, indent=1))


def cmd_probe(args):
    report = probe.run(config_from(args), args.seed, args.ticks, args.device)
    print(json.dumps(report["verdict"], indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m haishool.life9")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "bench", "probe"):
        p = sub.add_parser(name)
        p.add_argument("--seed", type=int, default=1)
        p.add_argument("--ticks", type=int, default={"run": 2000, "bench": 50, "probe": 3000}[name])
        p.add_argument("--worlds", type=int, default={"run": 8, "bench": 64, "probe": 32}[name])
        p.add_argument("--capacity", type=int, default=128)
        p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
        p.add_argument("--option", action="append", help="Config9 field=value (JSON value)")
        p.add_argument("--out")
        if name == "run":
            p.add_argument("--every", type=int, default=100)
            p.add_argument("--view-every", type=int, default=0)
            p.add_argument("--view-ticks", type=int, default=600)
            p.add_argument("--resume")
    args = parser.parse_args(argv)
    if args.command == "run" and not args.out:
        parser.error("run needs --out")
    {"run": cmd_run, "bench": cmd_bench, "probe": cmd_probe}[args.command](args)


if __name__ == "__main__":
    main()
