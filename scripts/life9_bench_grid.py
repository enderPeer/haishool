"""life9 speed grid on one device, plus a statistics check against a CPU run of the same setup.

    python scripts/life9_bench_grid.py --device cuda --out bench.json [--grid 16x128,64x128,64x256,256x128,256x256] [--check-ticks 500]

For every WORLDSxCAPACITY entry: a few warm-up ticks, then ticks timed for about --seconds
(at least 5), peak device memory, world-ticks per hour. An entry that runs out of memory
is recorded as such. ``--check-ticks`` runs ``--check-worlds`` (32) worlds x 128 slots from seed 1 and records
alive, births, deaths and the ledger error, to compare with the same run on the CPU (the
random streams differ between devices, so only the statistics are comparable).
Compute is dense over all slots, so cost does not depend on how many slots are alive.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from haishool.life9.config import Config9  # noqa: E402
from haishool.life9.world import World9  # noqa: E402


def sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def environment(device):
    info = {"python": platform.python_version(), "torch": torch.__version__, "device": device,
            "host": platform.node(), "cuda": torch.version.cuda, "hip": getattr(torch.version, "hip", None)}
    if device.startswith("cuda"):
        info["gpu"] = torch.cuda.get_device_name(torch.device(device))
        info["memory_total_gb"] = round(torch.cuda.get_device_properties(torch.device(device)).total_memory / 2 ** 30, 1)
    return info


def bench(cfg, device, seconds):
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    world = World9(cfg, 1, device)
    for _ in range(3):
        world.step()
    sync(device)
    start = time.perf_counter()
    world.step()
    sync(device)
    one = time.perf_counter() - start
    ticks = max(5, min(200, int(seconds / max(one, 1e-4))))
    start = time.perf_counter()
    for _ in range(ticks):
        world.step()
    sync(device)
    elapsed = time.perf_counter() - start
    row = {"worlds": cfg.worlds, "capacity": cfg.capacity, "hidden": cfg.hidden, "ticks": ticks,
           "ms_per_tick": round(1000 * elapsed / ticks, 2),
           "world_ticks_per_hour": round(cfg.worlds * ticks / elapsed * 3600),
           "slot_ticks_per_second": round(cfg.worlds * cfg.capacity * ticks / elapsed),
           "alive": int(world.alive.sum())}
    if device.startswith("cuda"):
        row["peak_memory_gb"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    return row


def check(device, ticks, worlds=32):
    world = World9(Config9(worlds=worlds, capacity=128), 1, device)
    start = time.perf_counter()
    for _ in range(ticks):
        world.step()
    sync(device)
    s = world.summary()
    stats = s["stats"]
    mean = lambda values: round(sum(values) / len(values), 3)
    return {"worlds": worlds, "capacity": 128, "ticks": ticks, "seconds": round(time.perf_counter() - start, 1),
            "alive_mean": mean(s["alive"]), "alive_min": min(s["alive"]), "alive_max": max(s["alive"]),
            "births_mean": mean(stats["births"]), "starvation_mean": mean(stats["deaths_starvation"]),
            "predation_mean": mean(stats["deaths_predation"]), "calls_mean": mean(stats["calls"]),
            "mean_hidden": mean(s["mean_hidden"]),
            "ledger_error_max": float(world.ledger_error().abs().max()),
            "per_world": {"alive": s["alive"], "births": stats["births"], "starvation": stats["deaths_starvation"],
                          "predation": stats["deaths_predation"], "calls": stats["calls"]}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--grid", default="16x128,64x128,64x256,256x128,256x256")
    parser.add_argument("--brain-grid", default="64x128")
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--check-ticks", type=int, default=500)
    parser.add_argument("--check-worlds", type=int, default=32)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = {"schema": "life9-bench-v1", "environment": environment(args.device), "rows": [], "check": None}
    entries = [(entry, 32) for entry in args.grid.split(",") if entry] + \
              [(entry, 128) for entry in args.brain_grid.split(",") if entry]
    for entry, hidden in entries:
        worlds, capacity = (int(v) for v in entry.split("x"))
        cfg = Config9(worlds=worlds, capacity=capacity, hidden=hidden, initial_hidden=min(96, hidden // 4 * 3),
                      founders=capacity * 3 // 8)
        try:
            row = bench(cfg, args.device, args.seconds)
        except torch.OutOfMemoryError as error:
            row = {"worlds": worlds, "capacity": capacity, "hidden": hidden, "error": "out of memory: " + str(error)[:120]}
        except RuntimeError as error:
            row = {"worlds": worlds, "capacity": capacity, "hidden": hidden, "error": str(error)[:200]}
        report["rows"].append(row)
        print(json.dumps(row), flush=True)
        Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    if args.check_ticks:
        report["check"] = check(args.device, args.check_ticks, args.check_worlds)
        print(json.dumps(report["check"]), flush=True)
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
