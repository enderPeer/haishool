"""Train Haishool on the simulated worlds of round 7 only, then evaluate and export, resumably.

    python -m haishool.train_sim run --name sim7 --data data/evo-v7 --steps 60000

Run on the training host. Every phase can be restarted with the same command; runs/<name>/status.json
says which phase is running (the trainer's own step and loss are in runs/<name>/train/status.json).

1. build    ``data/.../report.json`` missing: ``python -m haishool.curriculum7 build`` writes the
            24 topics (log: runs/<name>/build.log). An existing dataset must have the requested
            scale and number of world seeds.
2. gates    every dev and sealed gold answer is judged by the gate the evaluation will use, before
            any GPU time is spent; a gate that rejects its own gold stops the run here.
3. train    ``train_final.train(mix="sim")``: lessons (predict_*) 80% and stories (legacy_*) 20% of
            the sampled tokens, each share split equally among its topics; no old facts, no truth
            topics, holdout 0. Resumes from runs/<name>/train/latest.pt.
4. dev, sealed  one evaluation each (fingerprinted; a started evaluation is never redone with
            another checkpoint). ``final_run.score_lines`` scores, as in rounds 5 and 6, with the
            round-7 gates: a lesson by its LessonGate (``curriculum7.lesson_gates``), a story by its
            level's ``simulation().check``; a corrected round-6 story (``legacy_<old>_rules7``) by
            the round-6 module's simulation, which judges ``rules 7`` prompts.
5. recall   stories of the train split: a sample per story topic, judged by the exact trained
            answer, and every world7 ``say`` and ``meaning`` line of the train worlds, judged by
            the world7 gate. Dev and sealed stories name seeds the model never saw, so recall of
            memorised seeds can only be measured here.
6. summary  runs/<name>/summary.json: lessons (computable from the question) apart from stories
            (memorised seeds), stories by seeded and seedless lines, world7 language counts.
7. export   runs/<name>/export/<name>-<layers>x<width>-fp16.pt: model_state in float16, gpt_config,
            itos (the keys ``student.load`` reads), checked against the float32 checkpoint.

A dev or sealed prompt with a word that is not in the model's vocabulary (a world7 era such as
``replicators_3`` that only a held-out world reaches) cannot be asked: ``final_run.ask_many``
refuses it. Such lines are left out of the scored file, listed under ``unscorable`` and counted
as wrong in ``all_lines`` and in the summary.
"""
from __future__ import annotations

import argparse
import importlib
import json
import multiprocessing
import re
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from haishool import curriculum7, final_run, train_final
from haishool.curriculum import SEED, digest, json_write
from haishool.student import tokens
from haishool.truth import Verdict, split_line

#: a language question about a seed's world: ``world7 seed 8 5 say predator small``
_LANGUAGE_Q = re.compile(r"^[a-z0-9]+ seed (?:[0-9] )*[0-9] (word|meaning|say|order)(?: |$)")
#: the round-7 levels (haishool.evo) and the corrected round-6 levels (haishool.cosmos)
OLD = curriculum7.OLD


# ---------------------------------------------------------------------------------------------
# gates


def story_gate(topic: str):
    """The simulation that judges a story topic: ``legacy_<level>`` by ``haishool.evo.<level>``,
    ``legacy_<old>_rules7`` by the round-6 module ``haishool.cosmos.<old>`` (its simulation
    replays ``rules 7`` prompts under the corrected rules)."""
    from haishool import evo
    name = topic.removeprefix("legacy_")
    if not topic.startswith("legacy_"):
        raise ValueError(f"not a story topic: {topic}")
    if name.endswith("_rules7"):
        old = name.removesuffix("_rules7")
        if old not in OLD:
            raise ValueError(f"no corrected round-6 level for {topic}")
        return importlib.import_module(f"haishool.cosmos.{old}").simulation()
    if name not in evo.LEVELS:
        raise ValueError(f"no round-7 level for {topic}")
    return importlib.import_module(f"haishool.evo.{name}").simulation()


def gate_for(topic: str):
    """The gate of a round-7 topic: a lesson topic's LessonGate, a story topic's simulation."""
    if topic.startswith("predict_"):
        gates = curriculum7.lesson_gates()
        if topic not in gates:
            raise ValueError(f"no round-7 lesson gate for {topic}")
        return gates[topic]
    if topic.startswith("legacy_"):
        return story_gate(topic)
    raise ValueError(f"not a round-7 topic: {topic}")


class Recorder:
    """A gate that remembers its verdicts. ``final_run.score_lines`` judges each line's gold and
    then the model's answer, in file order; :meth:`judged` pairs them up again."""

    def __init__(self, gate) -> None:
        self.gate = gate
        self.calls: list[tuple[str, str, bool]] = []

    def check(self, prompt: str, answer: str) -> Verdict:
        verdict = self.gate.check(prompt, answer)
        self.calls.append((prompt, answer, bool(verdict.ok)))
        return verdict

    def judged(self) -> list[tuple[str, str, str, bool]]:
        """(prompt, gold, answer, ok) per scored line."""
        if len(self.calls) % 2:
            raise RuntimeError("score_lines judged an odd number of answers")
        out = []
        for (prompt, gold, gold_ok), (again, answer, ok) in zip(self.calls[::2], self.calls[1::2]):
            if prompt != again or not gold_ok:
                raise RuntimeError(f"unexpected judging order at {prompt!r}")
            out.append((prompt, gold, answer, ok))
        return out


class Exact:
    """Recall of a trained line: right when the answer is the line's own answer, word for word."""

    def __init__(self, golds: dict[str, str]) -> None:
        self.golds = golds

    def check(self, prompt: str, answer: str) -> Verdict:
        gold = self.golds.get(prompt)
        return Verdict(gold is not None and answer == gold, gold, "the trained answer")


def gold_job(args) -> tuple[str, str, int, list[str]]:
    """(topic, split, lines, rejected golds) for one dataset file."""
    topic, path = args
    gate = gate_for(topic)
    pairs = [p for t in Path(path).read_text(encoding="utf-8").splitlines() if (p := split_line(t))]
    bad = [f"{p} -> {a}" for p, a in pairs if not gate.check(p, a).ok]
    return topic, Path(path).name, len(pairs), bad


def check_golds(data: Path, out: Path, *, workers: int = 12) -> dict:
    """Judge every dev and sealed gold with the evaluation's gate (in a spawned pool)."""
    files = {}
    for split in ("dev", "sealed"):
        _, paths, hashes = final_run.dataset_files(data, split)
        files.update(hashes)
    fingerprint = {"data": digest(data / "report.json"), "files": files}
    if out.exists():
        report = json.loads(out.read_text(encoding="utf-8"))
        if report["fingerprint"] == fingerprint and report["ok"]:
            return report
    jobs = [(name.rsplit("-", 1)[0], str(data / name)) for name in sorted(files)]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(jobs))), mp_context=context) as pool:
        results = list(pool.map(gold_job, jobs))
    report = {"fingerprint": fingerprint, "files": {name: {"topic": topic, "lines": n, "rejected": bad[:5],
                                                           "rejected_count": len(bad)}
                                                    for topic, name, n, bad in results}}
    report["ok"] = not any(r["rejected_count"] for r in report["files"].values())
    json_write(out, report)
    if not report["ok"]:
        raise ValueError(f"evaluation gates reject dataset gold; see {out}")
    return report


# ---------------------------------------------------------------------------------------------
# evaluation


def scorable(path: Path, vocab, work: Path) -> tuple[Path, list[str], Counter]:
    """(the file to score, the lines that cannot be asked, their unknown words). A prompt with a
    word outside the model's vocabulary is refused by ``final_run.ask_many``."""
    known, unknown, words = [], [], Counter()
    for text in path.read_text(encoding="utf-8").splitlines():
        parts = split_line(text)
        if not parts:
            continue
        missing = set(tokens(f"q {parts[0]}. a")) - vocab.stoi.keys()
        (unknown if missing else known).append(text)
        words.update(missing)
    if not unknown:
        return path, [], words
    work.mkdir(parents=True, exist_ok=True)
    target = work / path.name
    target.write_text("".join(t + "\n" for t in known), encoding="utf-8")
    return target, unknown, words


def _count(rows) -> dict:
    rows = list(rows)
    return {"n": len(rows), "right": sum(ok for ok in rows)}


def breakdown(topic: str, judged: list[tuple[str, str, str, bool]], unscorable: list[str]) -> dict:
    """Counts over every line of a topic file, unscorable lines as wrong: all lines, and for a story
    topic its seeded lines (a seed's rollout) apart from its seedless ones (tables, hand-off
    lessons), and its language questions by kind (word, meaning, say, order)."""
    rows = [(prompt, ok, answer == gold) for prompt, gold, answer, ok in judged]
    rows += [(split_line(t)[0], False, False) for t in unscorable]
    n = len(rows)
    out = {"all_lines": {"n": n, "right": sum(ok for _, ok, _ in rows), "exact": sum(ex for *_, ex in rows),
                         "unscorable": len(unscorable)}}
    out["all_lines"]["gate_ok"] = out["all_lines"]["right"] / n if n else 0.
    if topic.startswith("legacy_"):
        out["seeded"] = _count(ok for prompt, ok, _ in rows if curriculum7.seed_of(prompt) is not None)
        out["seedless"] = _count(ok for prompt, ok, _ in rows if curriculum7.seed_of(prompt) is None)
        language = {}
        for prompt, ok, _ in rows:
            if m := _LANGUAGE_Q.match(prompt):
                language.setdefault(m[1], []).append(ok)
        if language:
            out["language"] = {kind: _count(oks) for kind, oks in sorted(language.items())}
    return out


def score_topic(model, vocab, topic: str, path: Path, gate, work: Path, device: str, limit=None) -> dict:
    """``final_run.score_lines`` of one file (scorable lines only) plus :func:`breakdown`."""
    target, unknown, words = scorable(path, vocab, work)
    recorder = Recorder(gate)
    result = final_run.score_lines(model, vocab, target, recorder, device, limit)
    result.update(breakdown(topic, recorder.judged(), [] if limit else unknown))
    if unknown:
        result["unscorable"] = {"lines": len(unknown), "words": dict(words.most_common(20)),
                                "examples": unknown[:4], "counted": "wrong" if not limit else "not sampled"}
    return result


def evaluate(checkpoint: Path, data: Path, split: str, out: Path, *, device: str = "cuda") -> dict:
    """One evaluation of a dev or sealed split, as ``final_run.evaluate`` but with round-7 gates."""
    if split not in ("dev", "sealed"):
        raise ValueError(f"evaluate dev or sealed, not {split}")
    paths, _, inputs = final_run.evaluation_inputs(data, split)
    fingerprint = {"checkpoint": digest(checkpoint), **inputs}
    if out.exists():
        report = json.loads(out.read_text(encoding="utf-8"))
        if report["fingerprint"] != fingerprint:
            raise ValueError(f"refusing to overwrite evaluation for different inputs: {out}")
        return report
    marker = out.with_suffix(".started.json")
    # A crash can resume the SAME checkpoint, never a revised one chosen after seeing the scores.
    if marker.exists() and json.loads(marker.read_text(encoding="utf-8")) != fingerprint:
        raise ValueError("evaluation already started with a different checkpoint")
    topics = [p.name.removesuffix(f"-{split}.txt") for p in paths]
    gates = {topic: gate_for(topic) for topic in topics}
    json_write(marker, fingerprint)
    import torch
    from haishool.student import load
    torch.set_num_threads(4)
    model, vocab = load(checkpoint, device)
    report = {"fingerprint": fingerprint, "topics": {}, "old": {}}
    for topic, path in zip(topics, paths):
        result = score_topic(model, vocab, topic, path, gates[topic], out.parent / f"{split}-inputs", device)
        report["topics"][topic] = result
        print(json.dumps({"evaluation": split, "topic": topic, "score": result["scores"],
                          "all_lines": result["all_lines"]}), flush=True)
    # As in final_run: macro average over the lessons, the input-conditioned topics.
    lessons = [r["all_lines"]["gate_ok"] for t, r in report["topics"].items()
               if t.startswith("predict_") and r["all_lines"]["n"]]
    report["selection_score"] = sum(lessons) / len(lessons) if lessons else 0.
    json_write(out, report)
    return report


def language_lines(path: Path, kinds=("say", "meaning")) -> list[str]:
    out = []
    for text in path.read_text(encoding="utf-8").splitlines():
        parts = split_line(text)
        if parts and (m := _LANGUAGE_Q.match(parts[0])) and m[1] in kinds:
            out.append(text)
    return out


def recall(checkpoint: Path, data: Path, out: Path, *, device: str = "cuda", limit: int = 256,
           language_limit: int = 4096) -> dict:
    """Recall of the stories the model trained on (train split; no dev or sealed line is read)."""
    _, paths, hashes = final_run.dataset_files(data, "train")
    fingerprint = {"checkpoint": digest(checkpoint), "data": digest(data / "report.json"),
                   "files": {k: v for k, v in hashes.items() if k.startswith("legacy_")},
                   "limit": limit, "language_limit": language_limit}
    if out.exists():
        report = json.loads(out.read_text(encoding="utf-8"))
        if report["fingerprint"] != fingerprint:
            raise ValueError(f"refusing to overwrite recall for different inputs: {out}")
        return report
    import torch
    from haishool.student import load
    torch.set_num_threads(4)
    model, vocab = load(checkpoint, device)
    work = out.parent / "recall-inputs"
    report = {"fingerprint": fingerprint, "judged_by": "the exact trained answer; a seeded sample of "
              f"{limit} lines per story topic", "topics": {}}
    for path in paths:
        topic = path.name.removesuffix("-train.txt")
        if not topic.startswith("legacy_"):
            continue
        golds = dict(p for t in path.read_text(encoding="utf-8").splitlines() if (p := split_line(t)))
        result = score_topic(model, vocab, topic, path, Exact(golds), work, device, limit)
        report["topics"][topic] = result
        print(json.dumps({"recall": topic, "score": result["scores"]}), flush=True)
    world = data / "legacy_world7-train.txt"
    lines = language_lines(world) if world.exists() else []
    if lines:
        work.mkdir(parents=True, exist_ok=True)
        target = work / "legacy_world7-language-train.txt"
        target.write_text("".join(t + "\n" for t in lines), encoding="utf-8")
        result = score_topic(model, vocab, "legacy_world7", target, story_gate("legacy_world7"), work / "language",
                             device, language_limit if len(lines) > language_limit else None)
        result["judged_by"] = "the world7 gate (any utterance the world's language accepts)"
        report["world7_language"] = result
        print(json.dumps({"recall": "world7_language", "language": result.get("language")}), flush=True)
    json_write(out, report)
    return report


# ---------------------------------------------------------------------------------------------
# summary and export


def _group(rows: dict[str, dict]) -> dict:
    per = {t: r["all_lines"]["gate_ok"] for t, r in sorted(rows.items())}
    n = sum(r["all_lines"]["n"] for r in rows.values())
    right = sum(r["all_lines"]["right"] for r in rows.values())
    return {"topics": per, "macro_gate_ok": sum(per.values()) / len(per) if per else 0.,
            "micro_gate_ok": right / n if n else 0., "n": n, "right": right,
            "unscorable": sum(r["all_lines"]["unscorable"] for r in rows.values())}


def _sum(rows, key: str) -> dict:
    n = sum(r.get(key, {}).get("n", 0) for r in rows)
    right = sum(r.get(key, {}).get("right", 0) for r in rows)
    return {"n": n, "right": right, "share": right / n if n else 0.}


def _language(result: dict | None) -> dict:
    """say and meaning counts (zero when none was asked), then any other language kind asked."""
    kinds = (result or {}).get("language", {})
    out = {kind: kinds.get(kind, {"n": 0, "right": 0}) for kind in ("say", "meaning")}
    out.update({kind: counts for kind, counts in kinds.items() if kind not in out})
    return out


def split_summary(report: dict) -> dict:
    topics = report["topics"]
    lessons = {t: r for t, r in topics.items() if t.startswith("predict_")}
    stories = {t: r for t, r in topics.items() if t.startswith("legacy_")}
    out = {"lessons": _group(lessons), "stories": _group(stories)}
    out["stories"]["seeded"] = _sum(stories.values(), "seeded")
    out["stories"]["seedless"] = _sum(stories.values(), "seedless")
    out["world7_language"] = _language(topics.get("legacy_world7"))
    return out


def summarize(root: Path, checkpoint: Path, settings: dict) -> dict:
    summary = {
        "name": settings["name"], "checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint),
        "model": f"{settings['layers']}x{settings['width']}, {settings['heads']} heads, context {settings['ctx']}",
        "steps": settings["steps"],
        "definitions": {
            "lessons": "predict_* topics: the answer is computable from the question; dev and sealed are "
                       "unseen inputs, so these scores measure whether the rules were learned",
            "stories": "legacy_* topics: stories of seeded rollouts, to be memorised; dev and sealed seeded "
                       "lines name worlds the model never saw (only seedless lines, tables and hand-off "
                       "lessons, are computable), so memorisation is measured by train_recall",
            "gate_ok": "accepted by the topic's gate (lessons: LessonGate; stories: the level's simulation)",
            "macro_gate_ok": "mean over topics", "micro_gate_ok": "over all lines of the group",
            "unscorable": "dev or sealed lines with a word outside the vocabulary, counted as wrong",
            "world7_language": "world7 say and meaning questions answered right (n asked, right)"},
        "dev": split_summary(json.loads((root / "dev.json").read_text(encoding="utf-8"))),
        "sealed": split_summary(json.loads((root / "sealed.json").read_text(encoding="utf-8"))),
    }
    recalled = json.loads((root / "train-recall.json").read_text(encoding="utf-8"))
    summary["train_recall"] = {"stories": _group(recalled["topics"]),
                               "world7_language": _language(recalled.get("world7_language")),
                               "judged_by": {"stories": recalled["judged_by"],
                                             "world7_language": "the world7 gate, every say and meaning line "
                                                                "of the train worlds"}}
    json_write(root / "summary.json", summary)
    return summary


def export(checkpoint: Path, target: Path, data: Path, *, device: str = "cuda") -> dict:
    """float16 copy of a checkpoint with the keys ``student.load`` reads, checked and probed."""
    import torch
    from haishool.student import load
    original = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = {k: v.half() if v.is_floating_point() else v for k, v in original["model_state"].items()}
    target.parent.mkdir(parents=True, exist_ok=True)
    train_final.atomic_checkpoint(target, {"model_state": state, "gpt_config": original["gpt_config"],
                                           "itos": original["itos"]})
    exported = torch.load(target, map_location="cpu", weights_only=True)
    if set(exported) != {"model_state", "gpt_config", "itos"} or exported["gpt_config"] != original["gpt_config"] \
            or exported["itos"] != original["itos"] or set(exported["model_state"]) != set(state) \
            or not all(torch.equal(exported["model_state"][k], v) for k, v in state.items()):
        raise RuntimeError("exported checkpoint does not match the float16 conversion")
    error = max(float((v.float() - original["model_state"][k].float()).abs().max())
                for k, v in state.items() if v.is_floating_point())
    del exported, original
    first = sorted(data.glob("predict_*-train.txt"))[0]
    pairs = [p for t in first.read_text(encoding="utf-8").splitlines() if (p := split_line(t))][:4]
    model, vocab = load(target, device)
    answers = final_run.ask_many(model, vocab, [p for p, _ in pairs], device, max_new=64)
    manifest = {"export": str(target), "sha256": digest(target), "bytes": target.stat().st_size,
                "source": str(checkpoint), "source_sha256": digest(checkpoint), "dtype": "float16",
                "keys": ["model_state", "gpt_config", "itos"], "max_abs_error_vs_float32": error,
                "config": model.config.to_json(), "vocab": len(vocab.itos),
                "probe": [{"prompt": p, "expected": g, "got": a} for (p, g), a in zip(pairs, answers)]}
    json_write(target.with_suffix(".json"), manifest)
    return manifest


# ---------------------------------------------------------------------------------------------
# the run


def _lock(path: Path):
    """An exclusive lock for the run's lifetime (POSIX); a second launch fails instead of interleaving."""
    handle = path.open("w")
    try:
        import fcntl
    except ImportError:  # Windows: no advisory lock
        return handle
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise RuntimeError(f"another run holds {path}") from None
    return handle


def run(name: str, *, data: Path, root: Path = Path("runs"), steps: int = 60000, layers: int = 8,
        width: int = 512, heads: int = 8, ctx: int = 256, batch: int = 48, seed: int = SEED,
        scale: float = 1., world_seeds: int = 256, workers: int = 12, device: str = "cuda",
        checkpoint_every: int = 1000, recall_limit: int = 256) -> dict:
    data, root = Path(data), Path(root) / name
    root.mkdir(parents=True, exist_ok=True)
    lock = _lock(root / "run.lock")
    settings = {"name": name, "data": str(data), "steps": steps, "layers": layers, "width": width, "heads": heads,
                "ctx": ctx, "batch": batch, "seed": seed, "scale": scale, "world_seeds": world_seeds,
                "device": device, "checkpoint_every": checkpoint_every, "recall_limit": recall_limit,
                "mix": "sim", "holdout": 0}
    config_path = root / "settings.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != settings:
        raise ValueError(f"run settings changed; use a new --name: {config_path}")
    json_write(config_path, settings)
    status = root / "status.json"
    train_dir = root / "train"

    def state(phase: str, **more) -> None:
        json_write(status, {"phase": phase, "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            "name": name, "train_status": str(train_dir / "status.json"), **more})
        print(json.dumps({"phase": phase, **more}), flush=True)

    try:
        if not (data / "report.json").exists():
            state("build", data=str(data))
            final_run.command(["-m", "haishool.curriculum7", "build", "--out", data, "--scale", scale,
                               "--world-seeds", world_seeds, "--workers", workers, "--seed", seed],
                              root / "build.log")
        dataset = json.loads((data / "report.json").read_text(encoding="utf-8"))
        if dataset.get("round") != 7 or dataset.get("scale") != scale \
                or len(dataset["rollouts"]["train_seeds"]) != world_seeds or dataset["seed"] != seed:
            raise ValueError(f"{data} is not the round-7 dataset of scale {scale}, {world_seeds} world seeds, "
                             f"seed {seed}")
        for topic in dataset["topics"]:
            train_final.sim_topic(topic)
            gate_for(topic)
        state("gates")
        check_golds(data, root / "gates.json", workers=workers)
        if not (train_dir / "training-report.json").exists():
            resume = train_dir / "latest.pt" if (train_dir / "latest.pt").exists() else None
            state("train", steps=steps, resume=str(resume) if resume else None)
            train_final.train(data=data, out=train_dir, layers=layers, width=width, heads=heads, ctx=ctx,
                              steps=steps, holdout=0., seed=seed, batch=batch, device=device,
                              checkpoint_every=checkpoint_every, resume=resume, mix="sim")
        checkpoint = train_dir / "student.pt"
        for split in ("dev", "sealed"):
            state(f"evaluate_{split}")
            evaluate(checkpoint, data, split, root / f"{split}.json", device=device)
        state("recall")
        recall(checkpoint, data, root / "train-recall.json", device=device, limit=recall_limit)
        state("summary")
        summary = summarize(root, checkpoint, settings)
        state("export")
        target = root / "export" / f"{name}-{layers}x{width}-fp16.pt"
        done = target.with_suffix(".json")
        manifest = json.loads(done.read_text(encoding="utf-8")) if done.exists() and target.exists() else None
        if not manifest or manifest["sha256"] != digest(target) or manifest["source_sha256"] != digest(checkpoint):
            manifest = export(checkpoint, target, data, device=device)
        state("complete", summary=str(root / "summary.json"), export=manifest["export"],
              dev_lessons=summary["dev"]["lessons"]["macro_gate_ok"],
              sealed_lessons=summary["sealed"]["lessons"]["macro_gate_ok"])
        return summary
    except BaseException as exc:
        state("failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        lock.close()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="build (if needed), train, evaluate dev and sealed once, summarise, export")
    r.add_argument("--name", required=True)
    r.add_argument("--data", type=Path, default=Path("data/evo-v7"))
    r.add_argument("--root", type=Path, default=Path("runs"))
    for flag, default in (("steps", 60000), ("layers", 8), ("width", 512), ("heads", 8), ("ctx", 256),
                          ("batch", 48), ("seed", SEED), ("world-seeds", 256), ("workers", 12),
                          ("checkpoint-every", 1000), ("recall-limit", 256)):
        r.add_argument("--" + flag, type=int, default=default)
    r.add_argument("--scale", type=float, default=1.)
    r.add_argument("--device", default="cuda")
    args = vars(ap.parse_args(argv))
    args.pop("command")
    summary = run(**args)
    print(json.dumps({"dev": summary["dev"]["lessons"]["macro_gate_ok"],
                      "sealed": summary["sealed"]["lessons"]["macro_gate_ok"]}), flush=True)


if __name__ == "__main__":
    main()
