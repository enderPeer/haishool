"""Rounds 5 and 6 in one build: the truth gates' lines and the toy universe's lines.

Round 5 has five closed topics with a gate each (:mod:`haishool.truth`: maths, elements,
substances, reactions, forces); round 6 has six simulations, one on top of the other
(:mod:`haishool.cosmos`: nucleo, gravity, planets, chem, life, world). Every one of them writes
its own lines and judges any answer, so this module adds no knowledge: it calls them, checks
each line once more, counts tokens and writes the files the trainer reads.

    python -m haishool.round5 build --out-truth data/truth-v5 --out-cosmos data/cosmos-v6 --seed 20261001
    python -m haishool.round5 judge "calc 1 2 plus 7" "1 9"
    python -m haishool.round5 stats
    python -m haishool.round5 evaluate --model runs/r5/student.pt --lines data/cosmos-v6/nucleo-sealed.txt

``build`` writes, per topic of round 5 (``data/truth-v5``):

    <topic>-train.txt     the topic's records, then questions from its ``generate``
    <topic>-sealed.txt    questions from another seed that training does not hold (no records)
    prompts.txt           every training and sealed prompt, for the feedback loop's ``--exclude``
    report.json           lines by kind, tokens, sealed lines, drops and seconds per topic

and per simulation of round 6 (``data/cosmos-v6``):

    <sim>-train.txt       the sim's tables, then rollout after rollout on the training seeds
    <sim>-sealed.txt      questions about rollouts on other seeds (no records)
    <sim>-heldout.txt     questions about the training seeds that are in no training line
    <sim>-rollouts.jsonl  ``{"sim", "seed", "split", "params", "steps", "summary"}`` per rollout
    prompts.txt, report.json

Lines are the dense lines of the gates, unchanged:

    iron element. number 2 6. symbol fe. protons 2 6. ...          record
    q calc 1 2 plus 7. a 1 9.                                      question
    gravity seed 5 step 1 2. time 1 point 2. radius 0 point 3 6 5. ...
    q gravity seed 5 step 1 2 next clumps. a 8.

**Budget.** The trainer's stream is tokens, so the sizes are set in tokens, counted as the
student counts them (:func:`haishool.student.tokens`, plus one ``<eos>`` per line): about
350,000 for round 5 and 350,000 for round 6, split by :data:`TRUTH_SHARE` and
:data:`COSMOS_SHARE`. A topic's ``n`` for ``generate`` is fitted to its share (:func:`fit`); a
simulation takes training seeds until its share is used. Every seed's lines depend only on the
build seed, the topic and the seed itself, never on the clock or on another topic.

**A rollout's lines** are thinned, since one rollout asks every metric of every step twice
(600 to 1200 questions): all records are kept (the whole timeline), all questions that are not
about one step (``final``, ``param``, the world's eras and hand-offs), and
:data:`STEP_QUESTIONS` of the step questions, drawn by seed. The rest are never trained; as
many of them again go to ``<sim>-heldout.txt``, which asks whether the model can turn the
records it saw into an answer it did not see. ``<sim>-sealed.txt`` asks about seeds it never
saw at all. All simulations share one list of training seeds and one of sealed seeds (each
takes its share from the front), so ``nucleo seed s`` is the first hour of ``world seed s``.

**Every line is checked before it is written** (:func:`screen`), and dropped and counted when it

- is longer than :data:`MAX_TOKENS` tokens (``long``),
- is not accepted by the gate that wrote it (``rejected``),
- is owned by another gate that judges it differently (``conflict``; elements and substances
  both own ``iron state`` and agree),
- asks a prompt rounds 1-4 already answer otherwise (``collision``), or
- repeats a prompt (``repeated``).

A rollout whose conservation gate fails is not written at all (``gate_failed``).

**Sealed means not given away.** A sealed question is dropped (``seen``) when the training file
of any topic holds its prompt (``iron state`` belongs to two), or holds a check line that
restates it: ``q check 1 2 plus 7 equals 1 9. a yes.`` gives ``q calc 1 2 plus 7`` away, and
the other way round (:func:`asked`). Record lines are always trained, so a sealed table fact
(``q iron protons``) is still in its record: there the sealed set asks for the step from
record to answer, and for the calculations it asks the rule. In round 6 a sealed question is
about a seed training never saw; only the seedless ones (the world's hand-offs) can repeat a
training prompt, and those are dropped the same way.

``judge`` prints who owns a prompt and the verdict, over the five gates, the six simulations
and the loop's ``judge ... answer ...`` lines. ``evaluate`` scores a trained model on a line
file with the same judges (:func:`haishool.truth.loop.evaluate`; the loop's own command knows
the five gates only). ``stats`` prints the two reports.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from haishool.cosmos import Rollout, chem, gravity, life, nucleo, planets, seeds, world
from haishool.schema import Record
from haishool.student import tokens
from haishool.truth import Line, Verdict, loop, split_line
from haishool.truth.splits import canonical_prompt

ROOT = Path(__file__).resolve().parents[1]
OUT_TRUTH = ROOT / "data" / "truth-v5"
OUT_COSMOS = ROOT / "data" / "cosmos-v6"
SEED = 20261001
#: the longest line the student trains on (context 96, with room for the ``<eos>``)
MAX_TOKENS = 80
#: tokens of training lines per round, ``<eos>`` included
TRUTH_TOKENS = 350_000
COSMOS_TOKENS = 350_000
#: share of the round-5 budget per topic. Maths has no table to lean on and gets the most; the
#: others are sized so that ``generate`` still gives its documented mix (their table questions
#: run out at 2,000 to 4,000 lines, after which only calculations and checks are left)
TRUTH_SHARE: dict[str, float] = {"maths": 0.30, "elements": 0.165, "substances": 0.215, "reactions": 0.165,
                                 "forces": 0.155}
#: share of the round-6 budget per simulation. A world is short (70 lines) and is the chain
#: itself, so it gets the most seeds; the levels get 15 to 20 rollouts each
COSMOS_SHARE: dict[str, float] = {"nucleo": 0.15, "gravity": 0.17, "planets": 0.14, "chem": 0.15, "life": 0.16,
                                  "world": 0.23}
#: questions drawn per topic for the sealed set, before those training gives away are dropped
SEALED_QUESTIONS = 1000
#: step questions kept per rollout (of 590 to 1170), and as many again for the held-out file
STEP_QUESTIONS = 32
#: sealed rollouts per simulation: a quarter of its training rollouts, this many at least
SEALED_ROLLOUTS = 3
#: seedless hand-off questions added per world (:func:`haishool.cosmos.world.practice_lines`)
PRACTICE = world.PRACTICE_PER_SEED
#: no simulation takes more seeds than this, whatever the budget (a world costs about 2 s)
MAX_ROLLOUTS = 2000
#: tries of :func:`fit`, and how close to the budget is close enough
FIT_ROUNDS = 6
FIT_WITHIN = 0.01
#: what rounds 1-4 trained: a new prompt may not give one of their prompts a second answer
EARLIER = (ROOT / "data" / "records-r3-all.jsonl", ROOT / "data" / "hops-v4b" / "hops-train-r4.txt")
DROPS = ("long", "rejected", "conflict", "collision", "repeated")

_STEP_Q = re.compile(r"^[a-z]+ seed [0-9 ]+ step ")


@dataclass(frozen=True)
class Judge:
    """A gate or a simulation under one name. The loop's functions want ``topic``, which half of
    the simulations do not carry."""
    topic: str
    gate: object

    def owns(self, prompt: str) -> bool:
        return self.gate.owns(prompt)

    def check(self, prompt: str, answer: str) -> Verdict:
        return self.gate.check(prompt, answer)


@dataclass(frozen=True)
class Level:
    """One simulation as the build sees it."""
    name: str
    sim: object
    #: the rollout a seed names, the only one the simulation's ``check`` replays
    rollout: Callable[[int], Rollout]
    #: every line of one rollout
    lines: Callable[[Rollout], list[Line]]
    #: lines that belong to no seed: table records and table questions
    tables: Callable[[], list[Line]] = list


def levels() -> dict[str, Level]:
    """The six simulations, in the order of the chain."""
    ch, li, pl = chem.simulation(), life.simulation(), planets.simulation()
    out = [
        Level("nucleo", nucleo.simulation(), nucleo.rollout, nucleo.lines),
        Level("gravity", gravity.simulation(), gravity.rollout, gravity.lines),
        Level("planets", pl, pl.rollout, planets.lines),
        Level("chem", ch, ch.run, chem.lines, lambda: ch.records() + ch.table_lines()),
        Level("life", li, li.rollout, life.lines),
        Level("world", world.simulation(), world.rollout, world.lines, lambda: world.records() + world.table_lines()),
    ]
    return {lv.name: lv for lv in out}


def judges() -> list[Judge]:
    """The five gates and the six simulations, each under its name, gates first."""
    return [Judge(g.topic, g) for g in loop.all_gates()] + [Judge(lv.name, lv.sim) for lv in levels().values()]


def n_tokens(text: str) -> int:
    return len(tokens(text))


def cost(lines: Iterable[Line | str]) -> int:
    """Tokens of the lines in the training stream: each line's own and its ``<eos>``."""
    return sum(n_tokens(ln if isinstance(ln, str) else ln.text) + 1 for ln in lines)


def rng_for(seed: int, *names: object) -> random.Random:
    """A generator of its own for one purpose (``rng_for(seed, "maths", "train")``): what one
    topic draws never moves what another gets. SHA-256, so the same on every machine."""
    text = " ".join(str(x) for x in (seed, *names))
    return random.Random(int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big"))


def earlier_answers(paths: Iterable[Path] = EARLIER) -> dict[str, set[str]]:
    """Prompt -> the answers rounds 1-4 trained for it, from record files (``.jsonl``) and line
    files; a file that is not there is skipped."""
    out: dict[str, set[str]] = {}
    for path in paths:
        if not path.is_file():
            continue
        if path.suffix == ".jsonl":
            rows = (json.loads(ln) for ln in path.open(encoding="utf-8") if ln.strip())
            texts = [q for row in rows for q in Record(row["obj"], row["values"]).queries()]
        else:
            texts = loop.read_lines(path)
        for text in texts:
            parts = split_line(text)
            if parts:
                out.setdefault(parts[0], set()).add(parts[1])
    return out


def screen(lines: list[Line], own: Judge, others: list[Judge], earlier: dict[str, set[str]],
           seen: set[str] | None = None) -> tuple[list[Line], Counter]:
    """The lines that may be written, and why the others may not (:data:`DROPS`).

    ``seen`` holds the prompts already kept (it is added to), so a prompt is asked once across
    calls. A record is only measured: its gate wrote it and no one is asked about it.
    """
    kept: list[Line] = []
    drops: Counter = Counter()
    seen = set() if seen is None else seen
    for ln in lines:
        if n_tokens(ln.text) > MAX_TOKENS:
            drops["long"] += 1
        elif ln.kind == "record":
            kept.append(ln)
        elif ln.prompt in seen:
            drops["repeated"] += 1
        elif not loop.gate_verdict(own, ln.prompt, ln.answer).ok:
            drops["rejected"] += 1
        elif any(not loop.gate_verdict(j, ln.prompt, ln.answer).ok for j in loop.owners(ln.prompt, others)):
            drops["conflict"] += 1
        elif earlier.get(ln.prompt, {ln.answer}) != {ln.answer}:
            drops["collision"] += 1
        else:
            seen.add(ln.prompt)
            kept.append(ln)
    return kept, drops


def asked(gate: object, ln: Line) -> str:
    """The question a line is about: its own prompt, or for a check line the question it restates.

    ``check 1 2 plus 7 equals 1 9`` is about ``calc 1 2 plus 7``, ``check 3 x equals 1 2 x 4``
    about ``solve 3 x equals 1 2`` (the last ``x`` parts equation and value), ``check iron
    protons 2 6`` about ``iron protons``: the shortest start of the claim that the gate can
    answer. The forces gate names it itself (``meta["inner"]``). A check that restates no
    single question (``check ... balanced``, ``check a heavier_than b``) is about itself.
    """
    if ln.kind == "record":
        return ln.prompt
    return canonical_prompt(str(ln.meta.get("inner", ln.prompt)), (gate,))


def fit(make: Callable[[int], list[Line]], budget: int, start: int) -> tuple[int, list[Line]]:
    """The ``n`` for which ``make(n)`` costs closest to ``budget`` tokens, and its lines.

    ``make`` is a gate's ``generate`` behind :func:`screen`; what it gives for ``n`` is not the
    start of what it gives for ``2 n`` (the mix shifts as the table questions run out), so the
    size is found by trying: scale ``n`` by the miss, :data:`FIT_ROUNDS` times at most.
    """
    n, best = max(1, start), None
    for _ in range(FIT_ROUNDS):
        lines = make(n)
        miss = abs(cost(lines) - budget)
        if best is None or miss < best[0]:
            best = (miss, n, lines)
        if miss <= FIT_WITHIN * budget or not lines:
            break
        n = max(1, round(n * budget / cost(lines)))
    return best[1], best[2]


def thin(lines: list[Line], rng: random.Random, k: int = STEP_QUESTIONS) -> tuple[list[Line], list[Line]]:
    """(the lines of one rollout to train on, questions held out of training).

    Records and the questions that are not about one step all stay; of the step questions ``k``
    stay and ``k`` more are held out, both drawn with ``rng``; the others are dropped.
    """
    step = [i for i, ln in enumerate(lines) if ln.kind != "record" and _STEP_Q.match(ln.prompt)]
    drawn = rng.sample(step, min(len(step), 2 * k))
    out, held = set(step) - set(drawn[:k]), sorted(drawn[k:])
    return [ln for i, ln in enumerate(lines) if i not in out], [lines[i] for i in held]


def write_lines(path: Path, lines: Iterable[Line | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join((ln if isinstance(ln, str) else ln.text) + "\n" for ln in lines)
    path.write_text(text, encoding="utf-8", newline="\n")


def _plain(x: object) -> object:
    """JSON for a numpy scalar a simulation may have left in a rollout."""
    if hasattr(x, "item"):
        return x.item()
    raise TypeError(f"not JSON: {x!r}")


def _tally(lines: list[Line]) -> dict:
    return {"lines": len(lines), "tokens": cost(lines), "by_kind": dict(sorted(Counter(ln.kind for ln in lines).items()))}


def _drops(*counters: Counter, **more: int) -> dict[str, int]:
    total = sum(counters, Counter())
    return {**{k: total[k] for k in DROPS}, **more}


def _totals(parts: dict[str, dict]) -> dict:
    out = {"lines": 0, "tokens": 0, "sealed_lines": 0, "sealed_tokens": 0, "dropped": 0, "seconds": 0.0}
    for rep in parts.values():
        out["lines"] += rep["lines"]
        out["tokens"] += rep["tokens"]
        out["sealed_lines"] += rep["sealed"]["lines"]
        out["sealed_tokens"] += rep["sealed"]["tokens"]
        out["dropped"] += sum(v for k, v in rep["dropped"].items() if k != "seen")
        out["seconds"] = round(out["seconds"] + rep["seconds"], 2)
    return out


def _prompts(*groups: list[Line]) -> list[str]:
    return sorted({ln.prompt for lines in groups for ln in lines if ln.kind != "record"})


def build_truth(out: Path = OUT_TRUTH, seed: int = SEED, budget: int = TRUTH_TOKENS,
                sealed: int = SEALED_QUESTIONS, topics: Iterable[str] | None = None,
                earlier: dict[str, set[str]] | None = None, log: Callable[[str], None] = print) -> dict:
    """Write the training and sealed lines of the round-5 topics into ``out``; the report."""
    out, everyone = Path(out), judges()
    by_name = {j.topic: j for j in everyone}
    earlier = earlier_answers() if earlier is None else earlier
    report: dict[str, dict] = {}
    trained: dict[str, list[Line]] = {}
    drops: dict[str, Counter] = {}
    given: set[str] = set()
    topics = tuple(topics or TRUTH_SHARE)
    for topic in topics:
        began = time.perf_counter()
        own, share = by_name[topic], round(budget * TRUTH_SHARE[topic])
        others = [j for j in everyone if j is not own]
        records, drops[topic] = screen(own.gate.records(), own, others, earlier)
        tried: dict[int, Counter] = {0: Counter()}

        def make(n: int) -> list[Line]:
            lines, tried[n] = screen(own.gate.generate(rng_for(seed, topic, "train"), n), own, others, earlier)
            return lines

        left = max(0, share - cost(records))
        n, questions = fit(make, left, left // 16) if left else (0, [])
        drops[topic] += tried[n]
        given |= {asked(own.gate, ln) for ln in questions}
        trained[topic] = records + questions
        write_lines(out / f"{topic}-train.txt", trained[topic])
        report[topic] = {**_tally(trained[topic]), "budget": share, "n": n, "records": len(records),
                         "seconds": time.perf_counter() - began}
    # the sealed sets come last: what one topic trains may give away another's question (iron state)
    prompts: list[str] = []
    sealed_once: set[str] = set()
    for topic in topics:
        began = time.perf_counter()
        own = by_name[topic]
        others = [j for j in everyone if j is not own]
        drawn, more = screen(own.gate.generate(rng_for(seed, topic, "sealed"), sealed), own, others, earlier)
        test = [ln for ln in drawn if ln.kind != "record" and ln.prompt not in sealed_once
                and asked(own.gate, ln) not in given]
        sealed_once |= {ln.prompt for ln in test}
        write_lines(out / f"{topic}-sealed.txt", test)
        prompts += _prompts(trained[topic], test)
        rep = report[topic]
        rep.update({"sealed": {**_tally(test), "drawn": sealed},
                    "dropped": _drops(drops[topic], more, seen=len(drawn) - len(test)),
                    "seconds": round(rep["seconds"] + time.perf_counter() - began, 2)})
        log(f"{topic}: {rep['lines']} lines ({rep['records']} records, n {rep['n']}), {rep['tokens']} tokens "
            f"of {rep['budget']}; sealed {len(test)}; dropped {rep['dropped']}")
    write_lines(out / "prompts.txt", sorted(set(prompts)))
    full = {"round": 5, "seed": seed, "budget_tokens": budget, "max_tokens": MAX_TOKENS,
            "earlier_prompts": len(earlier), "topics": report, "total": _totals(report)}
    (out / "report.json").write_text(json.dumps(full, indent=1) + "\n", encoding="utf-8", newline="\n")
    log(f"round 5: {full['total']}")
    return full


def cosmos_seeds(seed: int) -> tuple[list[int], list[int]]:
    """(training seeds, sealed seeds) of all simulations: two lists with no seed in common, each
    simulation taking as many as it needs from the front."""
    train = seeds(rng_for(seed, "cosmos", "train"), MAX_ROLLOUTS)
    taken = set(train)
    return train, [s for s in seeds(rng_for(seed, "cosmos", "sealed"), 2 * MAX_ROLLOUTS) if s not in taken][:MAX_ROLLOUTS]


@dataclass
class _Seed:
    """One rollout as the build writes it."""
    train: list[Line]
    held: list[Line]
    drops: Counter
    #: the prompts asked so far, this seed's included
    known: set[str]
    row: dict


def _seed_lines(lv: Level, seed: int, s: int, own: Judge, others: list[Judge], earlier: dict[str, set[str]],
                known: set[str]) -> _Seed | None:
    """The rollout of seed ``s``, thinned and screened; ``None`` when its conservation gate fails."""
    r = lv.rollout(s)
    if not lv.sim.conserved(r).ok:
        return None
    lines = lv.lines(r)
    if lv.name == "world":
        lines = lines + world.practice_lines(rng_for(seed, lv.name, s, "practice"), PRACTICE)
    kept, more = thin(lines, rng_for(seed, lv.name, s, "thin"))
    after = set(known)
    kept, d1 = screen(kept, own, others, earlier, after)
    more, d2 = screen(more, own, others, earlier, after)
    return _Seed(kept, more, d1 + d2, after, {"params": r.params, "steps": r.steps, "summary": r.summary})


def build_cosmos(out: Path = OUT_COSMOS, seed: int = SEED, budget: int = COSMOS_TOKENS,
                 sims: Iterable[str] | None = None, sealed_rollouts: int = SEALED_ROLLOUTS,
                 earlier: dict[str, set[str]] | None = None, log: Callable[[str], None] = print) -> dict:
    """Write the training, sealed and held-out lines and the rollouts of the simulations into ``out``."""
    out, everyone, all_levels = Path(out), judges(), levels()
    by_name = {j.topic: j for j in everyone}
    earlier = earlier_answers() if earlier is None else earlier
    train_seeds, sealed_seeds = cosmos_seeds(seed)
    report: dict[str, dict] = {}
    prompts: list[str] = []
    for name in sims or COSMOS_SHARE:
        began = time.perf_counter()
        lv, own, share = all_levels[name], by_name[name], round(budget * COSMOS_SHARE[name])
        others = [j for j in everyone if j is not own]
        known: set[str] = set()
        train, drops = screen(lv.tables(), own, others, earlier, known)
        held: list[Line] = []
        rows: list[str] = []
        used: list[int] = []
        failed = seen = 0
        spent = cost(train)
        for s in train_seeds:
            if spent >= share:
                break
            got = _seed_lines(lv, seed, s, own, others, earlier, known)
            if got is None:
                failed += 1
                continue
            if used and spent + cost(got.train) - share > share - spent:  # the miss would only grow: stop short
                break
            train += got.train
            held += got.held
            drops, known = drops + got.drops, got.known
            rows.append(json.dumps({"sim": name, "seed": s, "split": "train", **got.row}, default=_plain))
            spent += cost(got.train)
            used.append(s)
        test: list[Line] = []
        tested: list[int] = []
        for s in sealed_seeds:
            if len(tested) >= max(sealed_rollouts, len(used) // 4):
                break
            got = _seed_lines(lv, seed, s, own, others, earlier, known)
            if got is None:
                failed += 1
                continue
            test += [ln for ln in got.train if ln.kind != "record"]
            seen += got.drops.pop("repeated", 0)  # a seedless question training, or the sealed set, already asks
            drops, known = drops + got.drops, got.known
            rows.append(json.dumps({"sim": name, "seed": s, "split": "sealed", **got.row}, default=_plain))
            tested.append(s)
        write_lines(out / f"{name}-train.txt", train)
        write_lines(out / f"{name}-sealed.txt", test)
        write_lines(out / f"{name}-heldout.txt", held)
        write_lines(out / f"{name}-rollouts.jsonl", rows)
        prompts += _prompts(train, test, held)
        report[name] = {
            **_tally(train), "budget": share, "rollouts": len(used), "seeds": used,
            "sealed": {**_tally(test), "rollouts": len(tested), "seeds": tested},
            "heldout": _tally(held),
            "dropped": _drops(drops, gate_failed=failed, seen=seen),
            "seconds": round(time.perf_counter() - began, 2),
        }
        log(f"{name}: {len(used)} rollouts, {len(train)} lines, {report[name]['tokens']} tokens of {share}; "
            f"sealed {len(tested)} rollouts, {len(test)} lines; held out {len(held)}; "
            f"dropped {report[name]['dropped']}; {report[name]['seconds']} s")
    write_lines(out / "prompts.txt", sorted(set(prompts)))
    full = {"round": 6, "seed": seed, "budget_tokens": budget, "max_tokens": MAX_TOKENS,
            "step_questions": STEP_QUESTIONS, "earlier_prompts": len(earlier), "sims": report,
            "total": {**_totals(report), "rollouts": sum(r["rollouts"] for r in report.values()),
                      "sealed_rollouts": sum(r["sealed"]["rollouts"] for r in report.values()),
                      "heldout_lines": sum(r["heldout"]["lines"] for r in report.values())}}
    (out / "report.json").write_text(json.dumps(full, indent=1) + "\n", encoding="utf-8", newline="\n")
    log(f"round 6: {full['total']}")
    return full


def judge(prompt: str, answer: str | None = None, everyone: list | None = None) -> dict:
    """Who owns a prompt and what they say to an answer.

    ``prompt`` may be a whole line (``q calc 1 2 plus 7. a 1 9.``) or a bare prompt with the
    answer beside it. The first owner's verdict counts; ``owners`` lists all of them and
    ``agree`` says whether they are of one mind. No owner: ``Verdict(False, None, "no gate")``.
    """
    parts = split_line(prompt.strip()) if answer is None else None
    if parts:
        prompt, answer = parts
    prompt = " ".join(prompt.split()).rstrip(".").strip()
    prompt = prompt[2:] if prompt.startswith("q ") else prompt
    answer = " ".join((answer or "").split()).rstrip(".").strip()
    answer = answer[2:] if answer.startswith("a ") else answer
    inner = judges() if everyone is None else everyone
    owning = loop.owners(prompt, [loop.JudgeGate(inner), *inner])
    verdicts = [loop.gate_verdict(g, prompt, answer) for g in owning]
    first = verdicts[0] if verdicts else Verdict(False, None, "no gate")
    return {"prompt": prompt, "answer": answer, "owner": owning[0].topic if owning else None,
            "owners": [g.topic for g in owning], "ok": first.ok, "expected": first.expected, "reason": first.reason,
            "agree": len({v.ok for v in verdicts}) <= 1}


def score(ask: Callable[[str], str], lines: list[str], everyone: list | None = None) -> dict:
    """:func:`haishool.truth.loop.evaluate` over the gates, the simulations and the judge lines."""
    inner = judges() if everyone is None else everyone
    return loop.evaluate(ask, lines, [loop.JudgeGate(inner), *inner])


def stats(out_truth: Path = OUT_TRUTH, out_cosmos: Path = OUT_COSMOS) -> dict:
    """The two reports of the last build, ``None`` for one that is not there."""
    out = {}
    for name, path in (("truth", out_truth / "report.json"), ("cosmos", out_cosmos / "report.json")):
        out[name] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m haishool.round5", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="write the training and sealed lines of rounds 5 and 6")
    for p in (b, sub.add_parser("stats", help="print the reports of the last build")):
        p.add_argument("--out-truth", type=Path, default=OUT_TRUTH)
        p.add_argument("--out-cosmos", type=Path, default=OUT_COSMOS)
    b.add_argument("--seed", type=int, default=SEED)
    b.add_argument("--truth-tokens", type=int, default=TRUTH_TOKENS, help="training tokens of round 5")
    b.add_argument("--cosmos-tokens", type=int, default=COSMOS_TOKENS, help="training tokens of round 6")
    b.add_argument("--sealed", type=int, default=SEALED_QUESTIONS, help="sealed questions drawn per topic")
    b.add_argument("--only", choices=("truth", "cosmos"), help="build one round only")
    j = sub.add_parser("judge", help="who owns a prompt, and the verdict on an answer")
    j.add_argument("prompt", help="a prompt (calc 1 2 plus 7) or a whole line (q calc 1 2 plus 7. a 1 9.)")
    j.add_argument("answer", nargs="?", help="the answer to judge, without a and the final .")
    e = sub.add_parser("evaluate", help="score a trained model on a line file, simulations included")
    e.add_argument("--model", type=Path, required=True, help="student checkpoint (student.pt)")
    e.add_argument("--lines", type=Path, required=True, help="dense q/a lines, e.g. a sealed set")
    e.add_argument("--device", default="cpu")
    return ap


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    if args.cmd == "build":
        def log(text: str) -> None:  # progress on stderr: stdout carries the report
            print(text, file=sys.stderr, flush=True)

        earlier = earlier_answers()
        out = {}
        if args.only != "cosmos":
            out["truth"] = build_truth(args.out_truth, args.seed, args.truth_tokens, args.sealed, earlier=earlier, log=log)
        if args.only != "truth":
            out["cosmos"] = build_cosmos(args.out_cosmos, args.seed, args.cosmos_tokens, earlier=earlier, log=log)
    elif args.cmd == "judge":
        out = judge(args.prompt, args.answer)
    elif args.cmd == "stats":
        out = stats(args.out_truth, args.out_cosmos)
    else:
        if not args.lines.is_file():
            raise SystemExit(f"no such line file: {args.lines}")
        student = importlib.import_module("haishool.student")  # torch: only needed to talk to the model
        try:
            model, vocab = student.load(args.model, args.device)
        except Exception as e:
            raise SystemExit(f"cannot load {args.model}: {type(e).__name__}: {e}") from e
        out = score(lambda p: student.generate(model, vocab, f"q {p}. a", max_new=MAX_TOKENS, device=args.device),
                    loop.read_lines(args.lines))
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main()
