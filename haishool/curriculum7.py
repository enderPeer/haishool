"""Build the round-7 curriculum: lessons and stories of the evolution ladder, with semantic splits.

The layout and report format are those of :mod:`haishool.curriculum` (round 5/6), whose
``Writer``, ``partition``, ``key`` and ``digest`` are reused, so ``haishool.train_final`` reads the
result like any other data directory: ``<topic>-train.txt``, ``<topic>-dev.txt``,
``<topic>-sealed.txt`` per topic, a ``<topic>-report.json`` per lesson topic, and ``report.json``.

Topics:

* ``predict_<level>`` for every level of :data:`haishool.evo.LEVELS` (stars ... world7): lessons
  from the level's :class:`~haishool.evo.LessonGate` (target 30,000 training lines at scale 1),
  with the gate's records. ``predict_<level>_rules7`` does the same for the lesson gates of the
  corrected round-6 levels (nucleo7, gravity, planets7, chem, life7), so their rules-7 lessons
  reach training too. A lesson's split comes from ``curriculum.partition`` of the gate's
  ``split_key``, so two spellings of one question land on one side, in every topic.
* ``legacy_<level>`` for every level: the stories of the canonical rollouts of a common set of
  seeds (signals and world7 include their language lines), and ``legacy_<level>_rules7`` for the
  corrected round-6 levels (``<sim> seed <s> rules 7 ...``). A seed is train, dev or sealed for
  ALL levels at once. Records of train seeds go to train, records of held-out seeds are not
  written (round 6 did the same); of the questions about one step, 32 per rollout are kept, as
  in round 6. Seedless questions (tables, the hand-off lessons in world7's stories) take the
  semantic split of their key, the same split the lesson topics give them.

``--with-truth`` also writes the five round-5 truth topics (maths ... reactions) with
``curriculum.build_topic`` at the round-6 targets times ``--scale``: ``haishool.train_final``'s
mixture requires them in the same directory. Off by default.

Every gold answer is judged by its gate before it is written: lessons by their LessonGate in the
writer, stories by the level's own ``check`` in the worker that made the rollout (which replays
the cached canonical run), after ``conserved`` has passed. Rollouts run in a process pool.

    python -m haishool.curriculum7 build --out data/evo-v7 --scale 1 --world-seeds 256 --workers 12
"""
from __future__ import annotations

import argparse
import importlib
import json
import random
import re
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path

from haishool.curriculum import CONTEXT, SEED, Writer, build_topic, json_write, key, partition
from haishool.truth import Line, split_line

#: the round-5 truth topics and their round-6 targets: ``haishool.train_final``'s mixture needs all
#: five in the same directory, so ``build(..., with_truth=True)`` writes them too (curriculum.build_topic)
TRUTH_TARGETS = {"maths": 1500000, "forces": 300000, "elements": 10000, "substances": 25000, "reactions": 15000}
#: corrected round-6 levels, in the order of the chain
OLD = ("nucleo", "gravity", "planets", "chem", "life")
#: lessons per lesson topic at scale 1 (training lines)
LESSON_TARGET = 30000
#: questions about one step kept per rollout, as in round 6 (``haishool.round5.STEP_QUESTIONS``)
STEP_QUESTIONS = 32
#: a question about one step of one seed: ``gravity seed 5 rules 7 step 1 2 next clumps``
_STEP_Q = re.compile(r"^[a-z0-9]+ seed [0-9 ]+ (?:rules 7 )?step [0-9 ]+ (?:next )?([a-z0-9_]+)$")
#: the seed a story prompt names
_SEED = re.compile(r"^[a-z0-9]+ seed ((?:[0-9] )*[0-9])(?: |$)")
#: language questions are kept whole: they are what the language levels say, not one step's metric
_LANGUAGE = {"word", "meaning", "say"}


# ---------------------------------------------------------------------------------------------
# the levels as the builder sees them


def lesson_gates() -> dict[str, object]:
    """Topic -> lesson gate: ``predict_<level>`` for the round-7 levels, ``predict_<old>_rules7``
    for the corrected round-6 levels."""
    from haishool import evo
    from haishool.cosmos import chem, gravity, life, nucleo, planets
    out = {f"predict_{name}_rules7": mod.LESSONS for name, mod in
           zip(OLD, (nucleo, gravity, planets, chem, life))}
    for name in evo.LEVELS:
        out[f"predict_{name}"] = importlib.import_module(f"haishool.evo.{name}").LESSONS
    return out


@lru_cache(maxsize=1)
def _gates_by_sim() -> dict[str, object]:
    return {g.sim: g for g in lesson_gates().values()}


def key7(prompt: str) -> str:
    """The split identity of a prompt: a round-7 lesson's ``split_key``, else ``curriculum.key``."""
    words = prompt.split()
    if len(words) > 2 and words[1] == "predict":
        gate = _gates_by_sim().get(words[0])
        if gate is not None and gate.owns(prompt):
            return gate.split_key(prompt)
    return key(prompt)


def semantic(prompt: str, seed: int = SEED) -> str:
    """train, dev or sealed for a prompt, by ``curriculum.partition`` of its split key."""
    return partition(key7(prompt), seed)


def seed_of(prompt: str) -> int | None:
    """The seed a story prompt names (``world7 seed 8 5 era star age`` -> 85), else ``None``."""
    m = _SEED.match(prompt)
    return int(m.group(1).replace(" ", "")) if m else None


def story_topics() -> list[str]:
    """``legacy_<level>`` for the round-7 levels, then ``legacy_<old>_rules7``. world7 first: its
    rollouts are the slowest, so the pool starts on them."""
    from haishool import evo
    names = ["world7"] + [n for n in evo.LEVELS if n != "world7"]
    return [f"legacy_{n}" for n in names] + [f"legacy_{n}_rules7" for n in OLD]


def _level(topic: str):
    """(rollout(seed), lines(r), check, conserved, seedless extra lines) of a story topic."""
    name = topic.removeprefix("legacy_").removesuffix("_rules7")
    if topic.endswith("_rules7"):
        from haishool.cosmos import chem, gravity, life, nucleo, planets
        if name == "nucleo":
            return (lambda s: nucleo.rollout(s, rules=7)), nucleo.lines, nucleo.check, nucleo.conserved, \
                (lambda: nucleo.records(rules=7))
        if name == "gravity":
            return (lambda s: gravity.rollout(s, rules=7)), gravity.lines, gravity.check, gravity.conserved, list
        if name == "planets":
            return (lambda s: planets.rollout(s, rules=7)), planets.lines, planets.check, planets.conserved, list
        if name == "chem":
            sim = chem.simulation()
            return (lambda s: sim.rollout(s, rules=7)), chem.lines, chem.check, chem.conserved, \
                (lambda: sim.records(rules=7) + sim.table_lines(rules=7))
        if name == "life":
            sim = life.simulation()
            return (lambda s: sim.rollout(s, rules=7)), life.lines, life.check, life.conserved, list
        raise ValueError(topic)
    mod = importlib.import_module(f"haishool.evo.{name}")
    sim = mod.simulation()
    if name == "world7":
        # the lesson records live in predict_world7; the ladder, the spans and the tables stay here
        def extra():
            return [ln for ln in mod.records() + mod.table_lines() if not ln.prompt.startswith(f"{mod.SIM} predict ")]
    else:
        extra = sim.records
    return sim.rollout, mod.lines, mod.check, mod.conserved, extra


def thin(lines: list[Line], rng: random.Random, k: int = STEP_QUESTIONS) -> tuple[list[Line], list[Line]]:
    """(the lines of one rollout to train on, step questions held out), as ``round5.thin`` but
    for round-7 prompts (``rules 7``, ``world7``); language questions are never thinned."""
    step = [i for i, ln in enumerate(lines) if ln.kind != "record"
            and (m := _STEP_Q.match(ln.prompt)) and m.group(1) not in _LANGUAGE]
    drawn = rng.sample(step, min(len(step), 2 * k))
    out, held = set(step) - set(drawn[:k]), sorted(drawn[k:])
    return [ln for i, ln in enumerate(lines) if i not in out], [lines[i] for i in held]


def _plain(x: object) -> object:
    if hasattr(x, "item"):
        return x.item()
    raise TypeError(f"not JSON: {x!r}")


# ---------------------------------------------------------------------------------------------
# pool jobs


def rollout_job(args):
    """One canonical rollout: conserved, thinned, every kept or held question judged by the level."""
    topic, seed = args
    rollout, lines_of, check, conserved, _ = _level(topic)
    r = rollout(seed)
    verdict = conserved(r)
    if not verdict.ok:
        raise ValueError(f"{topic} seed {seed} conservation: {verdict}")
    kept, held = thin(lines_of(r), random.Random(seed))
    for ln in kept + held:
        if ln.kind != "record" and not check(ln.prompt, ln.answer).ok:
            raise ValueError(f"{topic}: rejected generated gold {ln.text}")
    summary = json.loads(json.dumps(r.summary, default=_plain))
    return topic, seed, kept, summary


def lesson_job(args):
    """One lesson topic, as ``curriculum.build_topic`` but split by the gate's ``split_key``."""
    topic, out, target, seed, eval_limit = args
    out = Path(out)
    gate = lesson_gates()[topic]
    from haishool.round5 import earlier_answers
    w = Writer(out, topic, seed, eval_limit, earlier_answers())
    t0 = time.monotonic()
    for ln in gate.records():
        w.add(ln)
    rounds = stale = drawn = 0
    while (rounds == 0 or w.counts["train"]["lines"] < target) and rounds < 10000 and stale < 8:
        rng = random.Random(f"{seed}/{topic}/{rounds}")
        before = w.counts["train"]["lines"]
        lines = gate.generate(rng, min(20000, max(1000, target - before)))
        rng.shuffle(lines)
        for ln in lines:
            drawn += 1
            w.add(ln, gate, split=partition(gate.split_key(ln.prompt), seed))
        stale = stale + 1 if w.counts["train"]["lines"] - before < 10 else 0
        rounds += 1
    report = {**w.close(), "target": target, "drawn": drawn, "rounds": rounds,
              "status": "target_reached" if w.counts["train"]["lines"] >= target
              else "saturated" if stale >= 8 else "round_limit",
              "seconds": round(time.monotonic() - t0, 2)}
    json_write(out / f"{topic}-report.json", report)
    return topic, report


# ---------------------------------------------------------------------------------------------
# the build


def seed_sets(seed: int, world_seeds: int, eval_seeds: int | None = None) -> tuple[list[int], list[int], list[int]]:
    """(train, dev, sealed) rollout seeds, drawn as round 6 draws them: one sample from 1..9998,
    the first ``world_seeds`` train, the rest halved into dev and sealed."""
    extra = max(8, world_seeds // 8) if eval_seeds is None else eval_seeds
    if world_seeds < 1 or extra < 2 or world_seeds + extra > 9998:
        raise ValueError("world_seeds and eval_seeds must leave at least two evaluation seeds within 1..9998")
    seeds = random.Random(seed).sample(range(1, 9999), world_seeds + extra)
    train, rest = seeds[:world_seeds], seeds[world_seeds:]
    return train, rest[:len(rest) // 2], rest[len(rest) // 2:]


def build(out: Path, *, seed: int = SEED, scale: float = 1.0, world_seeds: int = 256, workers: int = 12,
          eval_limit: int = 256, eval_seeds: int | None = None, with_truth: bool = False) -> dict:
    out = Path(out)
    if scale <= 0 or workers < 1 or eval_limit < 1:
        raise ValueError("scale, workers and eval_limit must be positive")
    train_seeds, dev_seeds, sealed_seeds = seed_sets(seed, world_seeds, eval_seeds)
    if (out / "report.json").exists():
        raise FileExistsError(f"completed dataset exists: {out}; choose a new destination")
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = {"seed": seed, "context": CONTEXT, "split": "semantic sha256: train80/dev10/sealed10",
              "evaluation": "analytic lessons by split_key; seed recall separately", "round": 7,
              "scale": scale, "topics": {}}
    jobs = [(t, str(out), max(50, int(LESSON_TARGET * scale)), seed, eval_limit) for t in lesson_gates()]
    truth = [(t, str(out), max(50, int(n * scale)), seed, eval_limit) for t, n in TRUTH_TARGETS.items()]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        done = list(pool.map(build_topic, truth)) if with_truth else []
        for topic, result in done + list(pool.map(lesson_job, jobs)):
            report["topics"][topic] = result
            print(json.dumps({"built": topic, "status": result["status"],
                              "train": result["splits"]["train"]["lines"]}), flush=True)
    side = {s: "train" for s in train_seeds} | {s: "dev" for s in dev_seeds} | {s: "sealed" for s in sealed_seeds}
    order = train_seeds + dev_seeds + sealed_seeds
    topics = story_topics()
    writers = {t: Writer(out, t, seed, eval_limit) for t in topics}
    for t in topics:
        _, _, check, _, extra = _level(t)
        for ln in extra():
            if ln.kind == "record":
                writers[t].add(ln, split="train")
            else:
                if not check(ln.prompt, ln.answer).ok:
                    raise ValueError(f"{t}: rejected table gold {ln.text}")
                writers[t].add(ln, split=semantic(ln.prompt, seed))
    t0 = time.monotonic()
    rolls = (out / "rollouts.jsonl").open("w", encoding="utf-8", newline="\n")
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for topic, s, kept, summary in pool.map(rollout_job, [(t, s) for t in topics for s in order]):
                split = side[s]
                for ln in kept:
                    if ln.kind == "record":
                        if split == "train":
                            writers[topic].add(ln, split="train")
                    elif seed_of(ln.prompt) is not None:
                        writers[topic].add(ln, split=split)
                    else:
                        writers[topic].add(ln, split=semantic(ln.prompt, seed))
                rolls.write(json.dumps({"topic": topic, "seed": s, "split": split, "summary": summary}) + "\n")
        for t, w in writers.items():
            report["topics"][t] = w.close()
    finally:
        rolls.close()
    report["rollouts"] = {"per_topic_train": len(train_seeds), "total": len(topics) * len(order),
                          "topics": topics, "seconds": round(time.monotonic() - t0, 2),
                          "train_seeds": sorted(train_seeds), "dev_seeds": sorted(dev_seeds),
                          "sealed_seeds": sorted(sealed_seeds)}
    with (out / "prompts.txt").open("w", encoding="utf-8", newline="\n") as f:
        for path in sorted(out / f"{t}-{s}.txt" for t in report["topics"] for s in ("train", "dev", "sealed")):
            for text in path.open(encoding="utf-8"):
                if parts := split_line(text.strip()):
                    f.write(key7(parts[0]) + "\n")
    report["seconds"] = round(time.monotonic() - started, 2)
    report["totals"] = {s: {k: sum(t["splits"][s].get(k, 0) for t in report["topics"].values())
                            for k in ("lines", "tokens")} for s in ("train", "dev", "sealed")}
    json_write(out / "report.json", report)
    return report


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="write the round-7 dataset into a new directory")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--seed", type=int, default=SEED)
    b.add_argument("--scale", type=float, default=1.0)
    b.add_argument("--world-seeds", type=int, default=256)
    b.add_argument("--eval-seeds", type=int, default=None,
                   help="held-out rollout seeds, halved into dev and sealed (default max(8, world_seeds // 8))")
    b.add_argument("--workers", type=int, default=12)
    b.add_argument("--eval-limit", type=int, default=256)
    b.add_argument("--with-truth", action="store_true",
                   help="also write the five round-5 truth topics, which haishool.train_final's mixture requires")
    args = vars(ap.parse_args(argv))
    args.pop("command")
    report = build(**args)
    print(json.dumps({"seconds": report["seconds"], "totals": report["totals"]}, indent=2))


if __name__ == "__main__":
    main()
