"""Resumable final experiment: build, two sizes, dev/feedback, sealed evaluation.

Run on the training host. No public service is changed. Sealed answers never enter
training or feedback; development scores select before a one-time sealed report.
Production refinements restore old held-out facts and are reported separately.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from haishool.curriculum import SEED, all_gates, digest, json_write, key, partition
from haishool.student import EOS, tokens
from haishool.truth import Line, is_dense, split_line


def dataset_files(data: Path, split: str):
    """Validate each selected shard against the completed dataset manifest."""
    if split not in ("train", "dev", "sealed"):
        raise ValueError(f"unknown dataset split: {split}")
    report = json.loads((data / "report.json").read_text(encoding="utf-8"))
    paths = sorted(data / f"{topic}-{split}.txt" for topic in report["topics"])
    if set(paths) != set(data.glob(f"*-{split}.txt")):
        raise ValueError(f"{split} files do not match dataset manifest")
    hashes = {}
    for path in paths:
        topic = path.name.removesuffix(f"-{split}.txt")
        actual = digest(path)
        if actual != report["topics"][topic]["splits"][split]["sha256"]:
            raise ValueError(f"{split} source hash mismatch: {path}")
        hashes[path.name] = actual
    return report, paths, hashes


def evaluation_inputs(data: Path, split: str, old_dir: Path | None = None):
    _, paths, hashes = dataset_files(data, split)
    old_paths = [old_dir / f"{name}.txt" for name in ("old-recall", "old-heldout")] if old_dir else []
    # Requested old evaluations are required; silently omitting one would make
    # the preservation check and its reported denominator unreliable.
    old_hashes = {path.name: digest(path) for path in old_paths}
    return paths, old_paths, {"data": digest(data / "report.json"), "split": split,
                             "files": hashes, "old_files": old_hashes}


def ask_many(model, vocab, prompts: list[str], device="cuda", max_new=256, batch=16):
    """Batch only equal-length prefixes: no padding or shifted learned positions."""
    import torch
    groups = defaultdict(list)
    for i, p in enumerate(prompts):
        words = [EOS, *tokens(f"q {p}. a")]
        missing = set(words) - vocab.stoi.keys()
        if missing:
            raise ValueError(f"evaluation input has unknown tokens: {sorted(missing)}")
        groups[len(words)].append((i, vocab.encode(words)))
    results = [""] * len(prompts)
    model.eval()
    amp = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if str(device).startswith("cuda") else contextlib.nullcontext
    with torch.inference_mode():
        for rows in groups.values():
            for offset in range(0, len(rows), batch):
                part = rows[offset:offset + batch]
                x = torch.tensor([v for _, v in part], device=device)
                output = [[] for _ in part]
                done = [False] * len(part)
                for _ in range(max_new):
                    with amp():
                        logits, _ = model(x[:, -model.config.ctx:])
                    nxt = logits[:, -1].argmax(-1)
                    for j, token in enumerate(nxt.tolist()):
                        if done[j]: continue
                        word = vocab.itos[token] if token < len(vocab.itos) else EOS
                        if word in (EOS, ".", "<pad>"):
                            done[j] = True
                        else:
                            output[j].append(word)
                    if all(done): break
                    x = torch.cat((x, nxt[:, None]), dim=1)
                for (i, _), words in zip(part, output):
                    results[i] = " ".join(words)
    return results


def score_lines(model, vocab, path: Path, gate=None, device="cuda", limit=None):
    pairs = [p for t in path.read_text(encoding="utf-8").splitlines() if (p := split_line(t))]
    if limit and len(pairs) > limit:
        pairs = random.Random(SEED).sample(pairs, limit)
    answers = ask_many(model, vocab, [p for p, _ in pairs], device)
    counts = defaultdict(Counter)
    errors = []
    for (prompt, gold), got in zip(pairs, answers):
        kind = "worked" if prompt.endswith(" steps") else "direct"
        exact = got == gold
        accepted = exact
        if gate:
            valid = gate.check(prompt, gold)
            if not valid.ok:
                raise ValueError(f"evaluation gold rejected by {gate}: {prompt}: {gold}")
            accepted = bool(gate.check(prompt, got).ok)
        for bucket in ("all", kind):
            counts[bucket].update(n=1, exact=int(exact), gate_ok=int(accepted))
        if not accepted and len(errors) < 8:
            errors.append({"prompt": prompt, "expected": gold, "answer": got})
    return {"scores": {k: {"n": v["n"], "exact": v["exact"] / v["n"],
                            "gate_ok": v["gate_ok"] / v["n"]} for k, v in counts.items()}, "examples": errors}


def evaluate(checkpoint: Path, data: Path, split: str, out: Path, *, device="cuda", old_dir: Path | None = None):
    paths, old_paths, inputs = evaluation_inputs(data, split, old_dir)
    fingerprint = {"checkpoint": digest(checkpoint), **inputs}
    if out.exists():
        report = json.loads(out.read_text())
        if report["fingerprint"] != fingerprint:
            raise ValueError(f"refusing to overwrite evaluation for different inputs: {out}")
        return report
    marker = out.with_suffix(".started.json")
    # A crash can resume the SAME locked checkpoint, never a revised selection using observed sealed scores.
    if marker.exists() and json.loads(marker.read_text()) != fingerprint:
        raise ValueError("evaluation already started with a different checkpoint")
    json_write(marker, fingerprint)
    from haishool.student import load
    from haishool.round5 import levels
    import torch
    torch.set_num_threads(4)
    model, vocab = load(checkpoint, device)
    gates = {g.topic: g for g in all_gates()}
    sims = levels()
    report = {"fingerprint": fingerprint, "topics": {}, "old": {}}
    for path in paths:
        topic = path.name.removesuffix(f"-{split}.txt")
        gate = sims[topic.removeprefix("legacy_")].sim if topic.startswith("legacy_") else gates[topic]
        report["topics"][topic] = score_lines(model, vocab, path, gate, device)
        print(json.dumps({"evaluation": split, "topic": topic, "score": report["topics"][topic]["scores"]}), flush=True)
    for path in old_paths:
        report["old"][path.stem] = score_lines(model, vocab, path, device=device)
    # Macro average input-conditioned and truth topic acceptance. Legacy identifiers are never
    # used to call a model a better physical predictor.
    scores = [r["scores"]["all"]["gate_ok"] for t, r in report["topics"].items()
              if not t.startswith("legacy_") and r["scores"].get("all", {}).get("n")]
    report["selection_score"] = sum(scores) / len(scores) if scores else 0
    json_write(out, report)
    return report


def feedback(checkpoint: Path, data: Path, out: Path, *, n=256, device="cuda"):
    if n < 1:
        raise ValueError("feedback n must be positive")
    dataset, _, _ = dataset_files(data, "train")
    seed = dataset["seed"]
    fingerprint = {"checkpoint": digest(checkpoint), "dataset": digest(data / "report.json"),
                   "prompts": digest(data / "prompts.txt"), "n": n, "seed": seed}
    from haishool.student import load
    from haishool.truth.loop import check_judge
    import torch
    torch.set_num_threads(4)
    if (out / "report.json").exists():
        report = json.loads((out / "report.json").read_text())
        if report["fingerprint"] != fingerprint: raise ValueError("different feedback inputs")
        return report
    out.mkdir(parents=True, exist_ok=True)
    seen = set((data / "prompts.txt").read_text().splitlines())
    gates = all_gates()
    selected = []
    for gate in gates:
        rng = random.Random(f"feedback/{seed}/{gate.topic}")
        fresh = []
        for _ in range(12):
            if len(fresh) >= n: break
            lines = gate.generate(rng, max(n * 2, 1000))
            if hasattr(gate, "generate_worked"):
                lines += gate.generate_worked(rng, max(n, 500))
            rng.shuffle(lines)
            for ln in lines:
                if ln.kind == "record" or len(fresh) >= n: continue
                k = key(ln.prompt)
                if k in seen or partition(ln.prompt, seed) != "train": continue
                if len(tokens(ln.text)) + 1 > 256: continue
                if not gate.check(ln.prompt, ln.answer).ok: raise ValueError("invalid feedback gold")
                seen.add(k)
                fresh.append(ln)
        selected.extend((gate, ln) for ln in fresh)
    model, vocab = load(checkpoint, device)
    got = ask_many(model, vocab, [ln.prompt for _, ln in selected], device)
    counts = defaultdict(Counter)
    handles = {g.topic: (out / f"{g.topic}-train.txt").open("w", encoding="utf-8", newline="\n") for g in gates}
    try:
        for (gate, ln), answer in zip(selected, got):
            verdict = gate.check(ln.prompt, answer)
            # Corrections are always canonical ground truth, including when a rounded answer passes.
            if any(g.owns(ln.prompt) and not g.check(ln.prompt, ln.answer).ok for g in gates):
                raise ValueError("feedback gates disagree on gold")
            handles[gate.topic].write(ln.text + "\n")
            counts[gate.topic]["confirmed" if verdict.ok else "corrected"] += 1
            label = "right" if verdict.ok else "wrong"
            jp = f"judge {ln.prompt} answer {answer}" if answer else ""
            if jp and is_dense(f"q {jp}. a {label}.") and len(tokens(f"q {jp}. a {label}.")) + 1 <= 256:
                if not check_judge(jp, label, gates).ok: raise ValueError("judge disagreement")
                handles[gate.topic].write(f"q {jp}. a {label}.\n")
                counts[gate.topic]["judge_" + label] += 1
    finally:
        for f in handles.values(): f.close()
    report = {"fingerprint": fingerprint, "topics": dict(counts), "questions": len(selected),
              "reserved_questions": "excluded by the same semantic hash split as the dataset"}
    json_write(out / "report.json", report)
    return report


def command(args, log: Path):
    log.parent.mkdir(parents=True, exist_ok=True)
    args = list(map(str, args))
    print(json.dumps({"command": args, "log": str(log)}), flush=True)
    with log.open("a", encoding="utf-8") as f:
        subprocess.run([sys.executable, "-u", *map(str, args)], stdout=f, stderr=subprocess.STDOUT, check=True)


def pipeline(root: Path, data: Path, *, steps=30000, feedback_steps=3000, production_steps=3000,
             workers=12, scale=1., world_seeds=512, batch=48, device="cuda"):
    root.mkdir(parents=True, exist_ok=True)
    status = root / "status.json"
    settings = {"steps": steps, "feedback_steps": feedback_steps, "production_steps": production_steps,
                "scale": scale, "world_seeds": world_seeds, "batch": batch, "device": device, "data": str(data)}
    config_path = root / "pipeline.json"
    if config_path.exists() and json.loads(config_path.read_text()) != settings:
        raise ValueError("pipeline settings changed; use a new output directory")
    json_write(config_path, settings)
    def state(phase, **more):
        json_write(status, {"phase": phase, "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **more})
    def train(out, layers, width, *, count, holdout, init=None, corrections=None):
        if (out / "training-report.json").exists(): return
        args = ["-m", "haishool.train_final", "--data", data, "--out", out,
                "--records", "data/records-r3-all.jsonl", "--hops", "data/hops-v4b/hops-train-r4.txt",
                "--layers", layers, "--width", width, "--heads", 8, "--ctx", 256,
                "--steps", count, "--holdout", holdout, "--batch", batch, "--device", device]
        if init: args += ["--init", init]
        if corrections: args += ["--feedback", corrections]
        if (out / "latest.pt").exists(): args += ["--resume", out / "latest.pt"]
        command(args, out / "train.log")
    summary = {"models": {}, "sealed_policy": "select on development, lock checkpoint, evaluate sealed once"}
    try:
        if not (data / "report.json").exists():
            state("build")
            command(["-m", "haishool.curriculum", "--out", data, "--scale", scale,
                     "--world-seeds", world_seeds, "--workers", workers], root / "build.log")
        for name, layers, width in (("6x384", 6, 384), ("8x512", 8, 512)):
            base, corrected, production = (root / f"{name}-{s}" for s in ("eval", "feedback", "full"))
            state("train_evaluation", model=name)
            train(base, layers, width, count=steps, holdout=.1)
            state("development", model=name)
            command(["-m", "haishool.final_run", "evaluate", "--checkpoint", base / "student.pt", "--data", data,
                     "--split", "dev", "--out", base / "dev.json", "--old-dir", base, "--device", device], base / "evaluate.log")
            state("feedback_generation", model=name)
            fb = root / f"{name}-corrections"
            command(["-m", "haishool.final_run", "feedback", "--checkpoint", base / "student.pt", "--data", data,
                     "--out", fb, "--device", device], fb / "feedback.log")
            state("feedback_training", model=name)
            train(corrected, layers, width, count=feedback_steps, holdout=.1, init=base / "student.pt", corrections=fb)
            command(["-m", "haishool.final_run", "evaluate", "--checkpoint", corrected / "student.pt", "--data", data,
                     "--split", "dev", "--out", corrected / "dev.json", "--old-dir", corrected, "--device", device], corrected / "evaluate.log")
            scores = {p: json.loads((p / "dev.json").read_text()) for p in (base, corrected)}
            # Preserve old fact recall within one percentage point before accepting the feedback model.
            def recall(r): return r["old"]["old-recall"]["scores"]["all"]["exact"]
            chosen = corrected if (scores[corrected]["selection_score"] >= scores[base]["selection_score"]
                                   and recall(scores[corrected]) >= recall(scores[base]) - .01) else base
            selection_path = root / f"{name}-selection.json"
            selection = {"selected": str(chosen), "sha256": digest(chosen / "student.pt"),
                         "base_dev": scores[base]["selection_score"], "feedback_dev": scores[corrected]["selection_score"]}
            if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
                raise ValueError("locked selection changed")
            json_write(selection_path, selection)
            state("sealed_evaluation", model=name)
            command(["-m", "haishool.final_run", "evaluate", "--checkpoint", chosen / "student.pt", "--data", data,
                     "--split", "sealed", "--out", root / f"{name}-sealed.json", "--old-dir", chosen,
                     "--device", device], root / f"{name}-sealed.log")
            state("production_refinement", model=name)
            train(production, layers, width, count=production_steps, holdout=0, init=chosen / "student.pt", corrections=fb)
            # Production is a separate model. Its development report is not substituted for the locked sealed score.
            command(["-m", "haishool.final_run", "evaluate", "--checkpoint", production / "student.pt", "--data", data,
                     "--split", "dev", "--out", production / "dev.json", "--old-dir", production,
                     "--device", device], production / "evaluate.log")
            summary["models"][name] = {**selection, "sealed": str(root / f"{name}-sealed.json"),
                                       "production": str(production / "student.pt"), "production_dev": str(production / "dev.json")}
            json_write(root / "summary.json", summary)
        state("complete", summary=str(root / "summary.json"))
    except BaseException as e:
        state("failed", error=f"{type(e).__name__}: {e}")
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("evaluate", "feedback"):
        p = sub.add_parser(name)
        p.add_argument("--checkpoint", type=Path, required=True)
        p.add_argument("--data", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--device", default="cuda")
        if name == "evaluate":
            p.add_argument("--split", choices=("dev", "sealed"), required=True)
            p.add_argument("--old-dir", type=Path)
        else: p.add_argument("--n", type=int, default=256)
    p = sub.add_parser("pipeline")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--feedback-steps", type=int, default=3000)
    p.add_argument("--production-steps", type=int, default=3000)
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--world-seeds", type=int, default=512)
    p.add_argument("--scale", type=float, default=1.)
    p.add_argument("--batch", type=int, default=48)
    p.add_argument("--device", default="cuda")
    args = vars(ap.parse_args())
    cmd = args.pop("cmd")
    globals()[cmd](**args)


if __name__ == "__main__":
    main()
