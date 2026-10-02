"""Build the final curriculum as bounded files, with semantic train/dev/sealed splits.

The files are a large, varied source of examples, not 60 in-memory copies. The
trainer streams token windows from one encoded copy and controls replay by tokens.
Legacy seed questions measure recall; ``predict_*`` files measure input-conditioned
substeps. Table questions measure retrieval from records, not unseen knowledge.
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import random
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from haishool.student import load_records, split_pairs, split_extra, tokens
from haishool.truth import Line, split_line

SEED = 20261001
CONTEXT = 256


def key(prompt: str) -> str:
    from haishool.truth.splits import canonical_prompt
    return canonical_prompt(prompt)


def partition(prompt: str, seed: int = SEED) -> str:
    n = int.from_bytes(hashlib.sha256(f"{seed}:{key(prompt)}".encode()).digest()[:8], "big") % 100
    return "sealed" if n < 10 else "dev" if n < 20 else "train"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def all_gates():
    from haishool.truth.loop import all_gates as truth_gates
    from haishool.cosmos import predict
    return truth_gates() + [predict.gate(t) for t in ("nucleo", "gravity", "planets", "chem", "life", "world")]


def gate_for(topic):
    return next(g for g in all_gates() if g.topic == topic)


class Writer:
    def __init__(self, out: Path, topic: str, seed: int, eval_limit: int, earlier=None):
        self.out, self.topic, self.seed, self.eval_limit = out, topic, seed, eval_limit
        self.handles = {s: (out / f"{topic}-{s}.txt").open("w", encoding="utf-8", newline="\n")
                        for s in ("train", "dev", "sealed")}
        self.counts = {s: Counter() for s in self.handles}
        self.drops = Counter()
        self.seen = {}
        self.earlier = earlier or {}
        self.reservoirs = {s: [] for s in ("dev", "sealed")}

    def add(self, ln: Line, gate=None, split=None):
        if len(tokens(ln.text)) + 1 > CONTEXT:
            self.drops["long"] += 1
            return
        identity = ln.prompt
        signature = hashlib.sha256(ln.text.encode()).digest()
        if identity in self.seen:
            if self.seen[identity] != signature:
                raise ValueError(f"contradictory duplicate prompt in {self.topic}: {identity}")
            self.drops["duplicate"] += 1
            return
        self.seen[identity] = signature
        if ln.kind != "record" and identity in self.earlier and self.earlier[identity] != {ln.answer}:
            self.drops["earlier_collision"] += 1
            return
        split = split or ("train" if ln.kind == "record" else partition(ln.prompt, self.seed))
        if gate and ln.kind != "record" and not gate.check(ln.prompt, ln.answer).ok:
            raise ValueError(f"gate rejected generated gold: {ln.text}")
        if split != "train":
            # Select by independent hash rank across the complete generation, not
            # by arrival order: late worked examples and rollout seeds are eligible.
            rank = int.from_bytes(hashlib.sha256(f"eval:{self.seed}:{self.topic}:{identity}".encode()).digest(), "big")
            item = (-rank, identity, ln.text, ln.kind)
            heap = self.reservoirs[split]
            if len(heap) < self.eval_limit:
                heapq.heappush(heap, item)
            else:
                self.drops["reserved_not_saved"] += 1
                if item > heap[0]:
                    heapq.heapreplace(heap, item)
            return
        self.handles[split].write(ln.text + "\n")
        self.counts[split].update(lines=1, tokens=len(tokens(ln.text)) + 1)
        self.counts[split][ln.kind] += 1

    def close(self):
        for split, heap in self.reservoirs.items():
            for _, _, text, kind in sorted(heap, reverse=True):
                self.handles[split].write(text + "\n")
                self.counts[split].update(lines=1, tokens=len(tokens(text)) + 1)
                self.counts[split][kind] += 1
        for f in self.handles.values():
            f.close()
        return {"splits": {s: dict(c, sha256=digest(self.out / f"{self.topic}-{s}.txt"))
                           for s, c in self.counts.items()}, "dropped": dict(self.drops)}


def build_topic(args):
    topic, out, target, seed, eval_limit = args
    out = Path(out)
    gate = gate_for(topic)
    from haishool.round5 import earlier_answers
    w = Writer(out, topic, seed, eval_limit, earlier_answers())
    t0 = time.monotonic()
    for ln in gate.records():
        w.add(ln)
    # Fresh generator seeds; finite tables may saturate, which is reported rather than looped forever.
    rounds = stale = drawn = 0
    while (rounds == 0 or w.counts["train"]["lines"] < target) and rounds < 10000 and stale < 8:
        rng = random.Random(f"{seed}/{topic}/{rounds}")
        before = w.counts["train"]["lines"]
        n = min(20000, max(1000, target - before))
        lines = gate.generate(rng, n)
        if hasattr(gate, "generate_worked"):
            lines += gate.generate_worked(rng, n)
        rng.shuffle(lines)
        for ln in lines:
            drawn += 1
            w.add(ln, gate)
        stale = stale + 1 if w.counts["train"]["lines"] - before < 10 else 0
        rounds += 1
        if rounds % 10 == 0:
            print(json.dumps({"topic": topic, "round": rounds, "train": w.counts["train"]["lines"]}), flush=True)
    report = {**w.close(), "target": target, "drawn": drawn, "rounds": rounds,
              "status": "target_reached" if w.counts["train"]["lines"] >= target else "saturated" if stale >= 8 else "round_limit",
              "seconds": round(time.monotonic() - t0, 2)}
    json_write(out / f"{topic}-report.json", report)
    return topic, report


def rollout_job(args):
    name, seed = args
    from haishool.round5 import levels, thin
    lv = levels()[name]
    r = lv.rollout(seed)
    verdict = lv.sim.conserved(r)
    if not verdict.ok:
        raise ValueError(f"{name} seed {seed} conservation: {verdict}")
    kept, held = thin(lv.lines(r), random.Random(seed), 32)
    # The conservation check and existing unit tests cover the engine. Every selected QA is
    # additionally judged by the corresponding engine, using its cached rollout.
    for ln in kept + held:
        if ln.kind != "record" and not lv.sim.check(ln.prompt, ln.answer).ok:
            raise ValueError(f"{name}: rejected generated gold {ln.text}")
    return name, seed, kept, held, r.summary


def build(out: Path, *, seed=SEED, scale=1.0, world_seeds=512, workers=12, eval_limit=256):
    out = Path(out)
    if scale <= 0 or workers < 1 or eval_limit < 1:
        raise ValueError("scale, workers and eval_limit must be positive")
    if world_seeds < 1 or world_seeds + max(8, world_seeds // 8) > 9998:
        raise ValueError("world_seeds must leave room for at least eight distinct evaluation seeds within 1..9998")
    if (out / "report.json").exists():
        raise FileExistsError(f"completed dataset exists: {out}; choose a new destination")
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    targets = {"maths": 1500000, "forces": 300000, "elements": 10000,
               "substances": 25000, "reactions": 15000}
    targets.update({f"predict_{t}": 30000 for t in ("nucleo", "gravity", "planets", "chem", "life", "world")})
    jobs = [(t, str(out), max(50, int(n * scale)), seed, eval_limit) for t, n in targets.items()]
    report = {"seed": seed, "context": CONTEXT, "split": "semantic sha256: train80/dev10/sealed10",
              "evaluation": "table retrieval; analytic substeps; seed recall separately", "topics": {}}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for topic, result in pool.map(build_topic, jobs):
            report["topics"][topic] = result
            print(json.dumps({"built": topic, "counts": result["splits"]}), flush=True)
    names = ("nucleo", "gravity", "planets", "chem", "life", "world")
    # A whole seed stays on one side across ALL simulations, including the composed world.
    rng = random.Random(seed)
    seeds = rng.sample(range(1, 9999), min(9998, world_seeds + max(8, world_seeds // 8)))
    train_seeds = set(seeds[:world_seeds])
    remaining = seeds[world_seeds:]
    dev_seeds = set(remaining[:len(remaining)//2])
    writers = {n: Writer(out, "legacy_" + n, seed, eval_limit) for n in names}
    rolls = (out / "rollouts.jsonl").open("w", encoding="utf-8", newline="\n")
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for name, s, kept, held, summary in pool.map(rollout_job, [(n, s) for n in names for s in seeds]):
                split = "train" if s in train_seeds else "dev" if s in dev_seeds else "sealed"
                for ln in kept:
                    if ln.kind == "record":
                        if split == "train": writers[name].add(ln, split="train")
                    elif " seed " in ln.prompt:
                        writers[name].add(ln, split=split)
                    else:
                        # Seedless formula questions share the semantic partition of predict_world.
                        writers[name].add(ln)
                rolls.write(json.dumps({"sim": name, "seed": s, "split": split, "summary": summary}) + "\n")
        for n, w in writers.items():
            report["topics"]["legacy_" + n] = w.close()
    finally:
        rolls.close()
    report["rollouts"] = {"per_sim_train": world_seeds, "total": len(names) * len(seeds),
                          "train_seeds": sorted(train_seeds), "dev_seeds": sorted(dev_seeds),
                          "sealed_seeds": sorted(set(remaining) - dev_seeds)}
    # Prompt list is streamed to disk; semantic hash assignment also protects reserved-but-unsaved questions.
    with (out / "prompts.txt").open("w", encoding="utf-8", newline="\n") as f:
        for path in sorted(out / f"{topic}-{split}.txt" for topic in report["topics"] for split in ("train", "dev", "sealed")):
            for text in path.open(encoding="utf-8"):
                if parts := split_line(text.strip()): f.write(key(parts[0]) + "\n")
    report["seconds"] = round(time.monotonic() - started, 2)
    report["totals"] = {s: {k: sum(t["splits"][s].get(k, 0) for t in report["topics"].values())
                              for k in ("lines", "tokens")} for s in ("train", "dev", "sealed")}
    json_write(out / "report.json", report)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--scale", type=float, default=1.)
    ap.add_argument("--world-seeds", type=int, default=512)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--eval-limit", type=int, default=256)
    args = ap.parse_args()
    print(json.dumps(build(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
