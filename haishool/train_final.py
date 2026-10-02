"""Train from one disk-backed token copy with a fixed replay mixture.

Only curriculum *-train.txt files enter the vocabulary or token cache. The
curriculum's existing dev/sealed partition is retained; --holdout applies to old
facts and hops with student.split_pairs/split_extra. Evaluation is a separate
process, after a transactional checkpoint. Resume keeps the original total-step
schedule and rejects any configuration, source, vocabulary, or cache change.

--mix final (the default) is the round-5 replay mixture above. --mix sim trains on
simulated worlds only: no old facts or hops (records and hops are neither read nor
hashed), only predict_* and legacy_* topics, lessons 80% and stories 20% of the tokens.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from array import array
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from haishool import student
from haishool.truth import is_dense, split_line

TRUTH_SHARES = {"maths": .5, "forces": .2, "elements": .1, "substances": .1, "reactions": .1}
VERSION = 1
#: training mixtures: "final" (round 5, the default) and "sim" (simulated worlds only)
MIXES = ("final", "sim")
#: mix "sim": token share of lessons (predict_*) and stories (legacy_*), each split equally among its topics
SIM_SHARES = {"predict": .8, "legacy": .2}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def atomic_checkpoint(path: Path, value) -> None:
    import torch
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def normalized_topic(filename: str) -> str:
    name = filename.removesuffix("-train.txt")
    return name.removeprefix("truth_")


def category(topic: str) -> str:
    if topic in ("old_facts", "old_hops"):
        return "old"
    if topic in TRUTH_SHARES:
        return "truth"
    if topic.startswith("predict_"):
        return "predict"
    if topic.startswith("legacy_"):
        return "legacy"
    raise ValueError(f"unrecognized training topic: {topic}")


def check_mix(mix: str) -> None:
    if mix not in MIXES:
        raise ValueError(f"unknown mix: {mix!r}; expected one of {', '.join(MIXES)}")


def sim_topic(topic: str) -> None:
    """Mix "sim" accepts lessons (predict_*) and stories (legacy_*) and nothing else."""
    if category(topic) not in SIM_SHARES:
        raise ValueError(f"mix sim accepts only predict_* and legacy_* topics, not {topic}")


def input_manifest(data: Path, records: Path | None, hops: Path | None, init: Path | None,
                   feedback: Path | None, mix: str = "final"):
    check_mix(mix)
    if mix == "sim" and feedback:
        raise ValueError("mix sim has no feedback sources")
    report_path = data / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    paths = sorted(data.glob("*-train.txt"))
    if not paths:
        raise ValueError("curriculum contains no *-train.txt files")
    actual_topics = {p.name.removesuffix("-train.txt") for p in paths}
    if actual_topics != set(report["topics"]):
        raise ValueError("curriculum manifest topics do not match training files")
    hashes = {str(report_path.resolve()): digest(report_path)}
    for path in paths:
        topic = path.name.removesuffix("-train.txt")
        category(normalized_topic(path.name))
        if mix == "sim":
            sim_topic(normalized_topic(path.name))
        got = digest(path)
        if got != report["topics"][topic]["splits"]["train"]["sha256"]:
            raise ValueError(f"training source hash mismatch: {path}")
        hashes[str(path.resolve())] = got
    # Mix "sim" neither reads nor hashes the old records and hops.
    extras = ([records, hops] if mix == "final" else []) + ([init] if init else [])
    if feedback:
        extras += sorted(feedback.glob("*-train.txt")) if feedback.is_dir() else [feedback]
        if feedback.is_dir() and not list(feedback.glob("*-train.txt")):
            raise ValueError("feedback directory contains no *-train.txt files")
    for path in extras:
        hashes[str(path.resolve())] = digest(path)
    return report, paths, hashes


def _lines(path: Path):
    with path.open(encoding="utf-8-sig") as handle:
        for raw in handle:
            if text := raw.strip():
                yield text


def _write_lines(path: Path, lines) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for line in lines:
            handle.write(line + "\n")


@dataclass
class Source:
    name: str
    topic: str
    path: Path
    feedback: bool = False
    count: int = 0
    weight: float = 0.
    bin_path: Path | None = None
    sha256: str = ""


def sim_mixture(sources: list[Source]) -> None:
    """Mix "sim": lessons (predict_*) 80% and stories (legacy_*) 20% of the tokens, each share
    split equally among the topics of its group; no old facts, truth topics or feedback."""
    for source in sources:
        sim_topic(source.topic)
        if source.feedback:
            raise ValueError("mix sim has no feedback sources")
    groups = {g: [s for s in sources if category(s.topic) == g and s.count > 0] for g in SIM_SHARES}
    if any(not rows for rows in groups.values()):
        raise ValueError("mix sim requires nonempty predict and legacy sources")
    for group, share in SIM_SHARES.items():
        topics = sorted({s.topic for s in groups[group]})
        for topic in topics:
            rows = [s for s in groups[group] if s.topic == topic]
            total = sum(s.count for s in rows)
            for source in rows:
                source.weight = share / len(topics) * source.count / total
    if not math.isclose(sum(s.weight for s in sources), 1.):
        raise ValueError("invalid mixture weights")


def mixture(sources: list[Source], mix: str = "final") -> None:
    """Weights are token shares because every sampled row has the same context.

    Within a topic with feedback, feedback receives 20% and base receives 80%.
    Truth topics have fixed shares; cosmos topics share each subgroup equally.
    Mix "sim" is :func:`sim_mixture`.
    """
    check_mix(mix)
    if mix == "sim":
        return sim_mixture(sources)
    groups = {g: [s for s in sources if category(s.topic) == g and s.count > 0]
              for g in ("old", "truth", "predict", "legacy")}
    if any(not rows for rows in groups.values()):
        raise ValueError("mixture requires nonempty old, truth, predict and legacy sources")
    if set(s.topic for s in groups["truth"]) != set(TRUTH_SHARES):
        raise ValueError("mixture requires all five truth topics")
    old_total = sum(s.count for s in groups["old"])
    for source in groups["old"]:
        source.weight = .4 * source.count / old_total
    for group, share in (("truth", .4), ("predict", .15), ("legacy", .05)):
        topics = sorted({s.topic for s in groups[group]})
        for topic in topics:
            topic_share = share * (TRUTH_SHARES[topic] if group == "truth" else 1 / len(topics))
            rows = [s for s in groups[group] if s.topic == topic]
            base = [s for s in rows if not s.feedback]
            extra = [s for s in rows if s.feedback]
            if not base:
                raise ValueError(f"feedback topic lacks base curriculum: {topic}")
            for part, fraction in ((base, .8 if extra else 1.), (extra, .2)):
                total = sum(s.count for s in part)
                for source in part:
                    source.weight = topic_share * fraction * source.count / total
    if not math.isclose(sum(s.weight for s in sources), 1.):
        raise ValueError("invalid mixture weights")


def prepare(data: Path, out: Path, records: Path | None, hops: Path | None, *, holdout: float, seed: int,
            ctx: int, paths: list[Path], report: dict, old_itos=None, feedback: Path | None = None,
            mix: str = "final"):
    check_mix(mix)
    if mix == "sim" and feedback:
        raise ValueError("mix sim has no feedback sources")
    cache = out / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    sources = []
    if mix == "final":
        recs, held = student.split_pairs(student.load_records(records), holdout, seed)
        hop_train, hop_held = student.split_extra(student.load_extra(hops), holdout, seed)
        _write_lines(out / "old-heldout.txt", [f"q {obj} {key}. a {gold}." for obj, key, gold in held] + hop_held)
        recall = [q for rec in recs for q in rec.queries() if len(student.tokens(q)) + 1 <= ctx]
        _write_lines(out / "old-recall.txt", random.Random(seed + 1).sample(recall, min(1000, len(recall))))
        _write_lines(cache / "old_facts.txt", (line for rec in recs for line in [rec.line(), *rec.queries()]))
        _write_lines(cache / "old_hops.txt", hop_train)
        sources = [Source("old_facts", "old_facts", cache / "old_facts.txt"),
                   Source("old_hops", "old_hops", cache / "old_hops.txt")]
    sources += [Source(p.stem, normalized_topic(p.name), p) for p in paths]
    if feedback:
        from haishool.curriculum import all_gates, partition
        from haishool.truth.loop import check_judge, split_judge
        gates = all_gates()
        routes = {}
        feedback_paths = sorted(feedback.glob("*-train.txt")) if feedback.is_dir() else [feedback]
        try:
            for path in feedback_paths:
                for line in _lines(path):
                    parts = split_line(line)
                    if not parts:
                        raise ValueError("feedback must contain verified question/answer lines")
                    prompt, answer = parts
                    judged = split_judge(prompt, gates)
                    if judged:
                        gate, identity, _ = judged
                        valid = check_judge(prompt, answer, gates).ok
                    else:
                        gate = next((g for g in gates if g.owns(prompt)), None)
                        identity = prompt
                        valid = gate is not None and gate.check(prompt, answer).ok
                    if not valid:
                        raise ValueError(f"unverified feedback: {prompt}")
                    if partition(identity, report["seed"]) != "train":
                        raise ValueError(f"feedback touches reserved dev/sealed input: {prompt}")
                    topic = gate.topic
                    category(topic)
                    if topic not in routes:
                        dest = cache / f"feedback_{topic}.txt"
                        routes[topic] = (dest, dest.open("w", encoding="utf-8", newline="\n"))
                    routes[topic][1].write(line + "\n")
        finally:
            for _, handle in routes.values():
                handle.close()
        sources += [Source("feedback_" + topic, topic, dest, True) for topic, (dest, _) in routes.items()]
    words = set()
    dropped = Counter()
    def accepted(source):
        for line in _lines(source.path):
            if not is_dense(line):
                raise ValueError(f"not dense in {source.path}: {line[:100]}")
            row = student.tokens(line) + [student.EOS]
            if len(row) > ctx:
                continue
            yield row
    for source in sources:
        for row in accepted(source):
            words.update(row)
            source.count += len(row)
        dropped[source.name] = sum(len(student.tokens(line)) + 1 > ctx for line in _lines(source.path))
    vocab = student.Vocab.grown(old_itos, sorted(words)) if old_itos is not None else student.Vocab(sorted(words))
    for index, source in enumerate(sources):
        source.bin_path = cache / f"source-{index:02d}.u32"
        with source.bin_path.open("wb") as handle:
            chunk = array("I")
            if chunk.itemsize != 4:
                raise RuntimeError("token encoding requires 32 bit unsigned array")
            for row in accepted(source):
                chunk.extend(vocab.encode(row))
                if len(chunk) >= 262144:
                    chunk.tofile(handle)
                    chunk = array("I")
            chunk.tofile(handle)
        if source.bin_path.stat().st_size != source.count * 4:
            raise RuntimeError("encoded token count mismatch")
        source.sha256 = digest(source.bin_path)
    mixture(sources, mix)
    meta = {"itos": vocab.itos, "dropped_long_lines": dict(dropped), "sources": [
        {"name": s.name, "topic": s.topic, "tokens": s.count, "weight": s.weight,
         "feedback": s.feedback, "bin": s.bin_path.name, "sha256": s.sha256} for s in sources]}
    atomic_json(cache / "prepared.json", meta)
    return meta


def train(*, data: Path, out: Path, records: Path | None = None, hops: Path | None = None, layers=6,
          width=384, heads=8, ctx=256, steps=30000, holdout=.1, seed=20261001, init: Path | None = None,
          feedback: Path | None = None, batch=48, resume: Path | None = None, device=None,
          checkpoint_every=1000, lr=None, mix: str = "final") -> dict:
    import torch
    from haishool.model import GPT, GPTConfig
    check_mix(mix)
    if mix == "final":
        if records is None or hops is None:
            raise ValueError("mix final requires records and hops")
        data, out, records, hops = map(Path, (data, out, records, hops))
    else:
        # Mix "sim" neither reads nor hashes the old records and hops, whatever was passed.
        data, out, records, hops = Path(data), Path(out), None, None
    init, feedback, resume = (Path(p) if p else None for p in (init, feedback, resume))
    if steps < 1 or batch < 1 or ctx < 2 or not 0 <= holdout < 1 or checkpoint_every < 1:
        raise ValueError("invalid steps, batch, context, holdout or checkpoint interval")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA training requires bf16 support")
    torch.set_num_threads(4)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    report, paths, hashes = input_manifest(data, records, hops, init, feedback, mix)
    peak_lr = lr if lr is not None else (3e-4 if init else 1e-3)
    config = {"version": VERSION, "layers": layers, "width": width, "heads": heads,
              "ctx": ctx, "steps": steps, "holdout": holdout, "seed": seed, "batch": batch,
              "device": device, "lr": peak_lr, "checkpoint_every": checkpoint_every,
              "sources": hashes,
              "code": {str(p.relative_to(Path(__file__).parent)): digest(p)
                       for p in sorted(Path(__file__).parent.rglob("*.py"))}}
    if mix != "final":  # the default configuration (and so its fingerprint) has no mix key
        config["mix"] = mix
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    out.mkdir(parents=True, exist_ok=True)
    resumed = torch.load(resume, map_location="cpu", weights_only=True) if resume else None
    if resumed and resumed.get("fingerprint") != fingerprint:
        raise ValueError("resume configuration or source fingerprint mismatch")
    if not resume and (out / "latest.pt").exists():
        raise FileExistsError("output already has a checkpoint; use --resume or a new --out")
    initial = torch.load(init, map_location="cpu", weights_only=True) if init and not resume else None
    try:
        atomic_json(out / "status.json", {"status": "preparing", "step": resumed["step"] if resumed else 0,
                                          "fingerprint": fingerprint})
        if resumed:
            meta = json.loads((out / "cache" / "prepared.json").read_text(encoding="utf-8"))
            if meta != resumed["prepared"]:
                raise ValueError("resume prepared cache metadata mismatch")
            for source in meta["sources"]:
                if digest(out / "cache" / source["bin"]) != source["sha256"]:
                    raise ValueError(f"resume token cache hash mismatch: {source['name']}")
        else:
            meta = prepare(data, out, records, hops, holdout=holdout, seed=seed, ctx=ctx,
                           paths=paths, report=report, old_itos=initial["itos"] if initial else None,
                           feedback=feedback, mix=mix)
        # Detect source changes while a large corpus was being encoded.
        if any(digest(Path(path)) != sha for path, sha in hashes.items()):
            raise ValueError("input source changed during preparation")
        cfg = GPTConfig(vocab_size=max(32, len(meta["itos"])), n_layer=layers, n_head=heads,
                        d_model=width, ctx=ctx, dropout=.1)
        if initial:
            old_cfg = GPTConfig.from_json(initial["gpt_config"])
            if (old_cfg.n_layer, old_cfg.d_model, old_cfg.n_head) != (layers, width, heads) or old_cfg.ctx > ctx:
                raise ValueError("initial checkpoint architecture does not match requested model")
            cfg = GPTConfig.from_json({**initial["gpt_config"], "ctx": ctx,
                                       "vocab_size": max(old_cfg.vocab_size, len(meta["itos"]))})
        if resumed:
            cfg = GPTConfig.from_json(resumed["gpt_config"])
        model = GPT(cfg).to(device)
        if initial:
            model.load_state_dict(student.grown_state(initial["model_state"], model.state_dict()))
        if resumed:
            model.load_state_dict(resumed["model_state"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=.1)
        sampler = random.Random(seed + 31)
        observed = Counter()
        start_step = 0
        log = []
        if resumed:
            optimizer.load_state_dict(resumed["optimizer_state"])
            sampler.setstate(resumed["sampler_rng"])
            random.setstate(resumed["python_rng"])
            torch.set_rng_state(resumed["torch_rng"])
            if device.startswith("cuda"):
                torch.cuda.set_rng_state_all(resumed["cuda_rng"])
            start_step = resumed["step"]
            observed.update(resumed["sampled_windows"])
            log = resumed["log"]
        sources = [s for s in meta["sources"] if s["tokens"] > 0 and s["weight"] > 0]
        maps = [np.memmap(out / "cache" / s["bin"], dtype=np.uint32, mode="r") for s in sources]
        weights = [s["weight"] for s in sources]
        offsets = np.arange(ctx + 1)
        started = time.monotonic()
        last_loss = None
        step = start_step
        def checkpoint(path):
            if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
                raise FloatingPointError("nonfinite model parameters; checkpoint not replaced")
            atomic_checkpoint(path, {"model_state": model.state_dict(), "gpt_config": cfg.to_json(),
                "itos": meta["itos"], "optimizer_state": optimizer.state_dict(), "step": step,
                "fingerprint": fingerprint, "run_config": config, "prepared": meta,
                "sampler_rng": sampler.getstate(), "python_rng": random.getstate(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.startswith("cuda") else [],
                "sampled_windows": dict(observed), "log": log})
        if not resumed:
            checkpoint(out / "initialized.pt")
            checkpoint(out / "latest.pt")
        model.train()
        for step in range(start_step + 1, steps + 1):
            rows = []
            for index in sampler.choices(range(len(sources)), weights=weights, k=batch):
                stream = maps[index]
                start = sampler.randrange(len(stream))
                rows.append(np.asarray(stream[(start + offsets) % len(stream)], dtype=np.int64))
                observed[sources[index]["name"]] += 1
            packed = torch.from_numpy(np.stack(rows)).to(device)
            lr_now = peak_lr * min(1., step / student.WARMUP) * (.1 + .45 * (1 + math.cos(math.pi * step / steps)))
            for group in optimizer.param_groups:
                group["lr"] = lr_now
            optimizer.zero_grad(set_to_none=True)
            context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.startswith("cuda") else nullcontext()
            with context:
                _, loss = model(packed[:, :-1], packed[:, 1:])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"nonfinite training loss at step {step}")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            last_loss = float(loss.detach())
            if step == 1 or step % 100 == 0 or step == steps:
                status = {"status": "training", "step": step, "steps": steps, "loss": last_loss,
                          "gradient_norm": float(norm), "seconds": round(time.monotonic() - started, 2),
                          "fingerprint": fingerprint}
                log.append(status)
                atomic_json(out / "status.json", status)
                print(json.dumps(status), flush=True)
            if step % checkpoint_every == 0 or step == steps:
                checkpoint(out / "latest.pt")
        checkpoint(out / "student.pt")
        total_windows = sum(observed.values())
        result = {"status": "completed", "step": step, "steps": steps, "loss": last_loss,
                  "fingerprint": fingerprint, "config": cfg.to_json(), "run_config": config,
                  "vocab": len(meta["itos"]), "parameters": model.num_params(),
                  "source_mixture": [{**s, "observed_windows": observed[s["name"]],
                                      "observed_token_fraction": observed[s["name"]] / total_windows if total_windows else 0}
                                     for s in meta["sources"]],
                  "sampled_tokens": total_windows * ctx, "single_copy_tokens": sum(s["tokens"] for s in sources),
                  "dropped_long_lines": meta["dropped_long_lines"], "log": log,
                  "evaluation": "none; evaluate checkpoint in a separate process",
                  "checkpoint": str((out / "student.pt").resolve()),
                  "phase": "continued_from_init" if init else "from_scratch",
                  "holdout_scope": "old facts and hops; curriculum uses its fixed train/dev/sealed partitions"}
        if mix != "final":
            result["mix"] = mix
            result["holdout_scope"] = "none: mix sim has no old facts or hops; curriculum uses its fixed train/dev/sealed partitions"
        atomic_json(out / "training-report.json", result)
        atomic_json(out / "status.json", {"status": "completed", "step": step, "fingerprint": fingerprint,
                                          "checkpoint": result["checkpoint"]})
        return result
    except Exception as exc:
        atomic_json(out / "status.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                                          "fingerprint": fingerprint})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--records", type=Path, default=Path("data/records-r3-all.jsonl"))
    parser.add_argument("--hops", type=Path, default=Path("data/hops-v4b/hops-train-r4.txt"))
    for name, default in (("layers", 6), ("width", 384), ("heads", 8), ("ctx", 256),
                          ("steps", 30000), ("seed", 20261001), ("batch", 48), ("checkpoint-every", 1000)):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--holdout", type=float, default=.1)
    parser.add_argument("--mix", choices=MIXES, default="final",
                        help="final: the round-5 replay mixture; sim: simulated worlds only (predict_*, legacy_*)")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--device")
    for name in ("init", "feedback", "resume"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    result = train(**vars(args))
    print(json.dumps({k: result[k] for k in ("status", "steps", "checkpoint", "sampled_tokens")}), flush=True)


if __name__ == "__main__":
    main()
