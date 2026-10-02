"""Round 5's feedback loop: the trained model is asked fresh questions and the gates judge them.

Every gate in :mod:`haishool.truth` writes its own questions and knows their answers, so a
model's answer to a question it has never seen can be judged by code, not by a teacher model.
The loop asks, judges and writes the outcome back as training lines for the next pass:

    q calc 1 2 plus 7. a 1 9.                          confirmed: the model was right
    q calc 1 2 plus 7. a 1 9.                          corrected: the model was wrong, the gate's value
    q judge calc 1 2 plus 7 answer 1 9. a right.       the model's own answer, judged
    q judge calc 1 2 plus 7 answer 2 0. a wrong.

A confirmed or corrected line always carries the gate's own value where the gate gives one, so
an answer a gate accepts loosely (rounded, in another order) is trained in the gate's form and
only the judge line keeps the model's wording. A corrected line is only written when the gate
can say what the answer should have been. The judge lines teach the model to tell a right
answer from a wrong one; :class:`JudgeGate` makes them a closed topic of their own by handing
the inner question back to the gate that owns it, so a judge line is as checkable as the line
it judges.

Nothing the model says is trusted, and no gate is trusted further than it can be checked
(:func:`gate_verdict`): a number that is not finite (``1 e 9 9 9 9`` parses to infinity, which
passes any comparison within a relative tolerance) is wrong whatever the gate says, a gate that
raises on an answer has not judged it, and a line is written only if it is dense, at most
:data:`MAX_TOKENS` tokens long and accepted once more by every gate that owns its question (two
topics may own the same one, ``silver state``; where they disagree nothing is written).

    python -m haishool.truth.loop run --model runs/r5/student.pt --out data/loop-r5.txt \\
        --n 200 --seed 1 --exclude data/truth-train-r5.txt data/truth-sealed-r5.txt
    python -m haishool.truth.loop evaluate --model runs/r5/student.pt --lines data/truth-sealed-r5.txt

Both print a JSON report with, per topic and kind, ``n``, ``exact`` (the model's answer equals
the expected one word for word) and ``gate_ok`` (the gate accepts it, which may be looser, e.g.
within a rounding tolerance), plus ``overall``, ``gate_errors`` (topic -> how often its gate
raised, with the first message) and the counts of lines written. For ``run`` these are, in
``lines``: ``confirmed``, ``corrected``, ``judge_right`` and ``judge_wrong`` (written; their sum
is ``total``), ``loose`` (the confirmed lines whose accepted answer was not the gate's own
value), ``uncorrected`` (wrong, and the gate gave no value), ``skipped`` (not dense or too
long), ``unverified`` (a gate that owns the question would not accept the line), ``gate_errors``
(questions whose gate raised; nothing is written for them) and ``disowned`` (questions their
own gate does not own). Every question ends in two counted outcomes, one for its line and one
for its judge line, so the counters add up to twice the questions that were judged. The model
is only reached through ``ask(prompt) -> dense answer``, so the loop itself needs no torch.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable

from haishool.truth import DIGITS, NUMBER_WORDS, Gate, Line, Verdict, parse_num, split_line

#: the closed topics of round 5, in the order their gates are asked
TOPICS = ("maths", "elements", "substances", "reactions", "forces")
#: the longest line the student trains on (context 96, with room for the ``<eos>``)
MAX_TOKENS = 80
#: a generated question the model has already seen is replaced by another, this many rounds at most
MAX_ROUNDS = 4
#: how a verdict's reason starts when the gate raised instead of judging
GATE_ERROR = "gate error"
#: the counters of ``run``'s report, in the order they are printed
COUNTS = ("confirmed", "corrected", "uncorrected", "loose", "judge_right", "judge_wrong", "skipped",
          "unverified", "gate_errors", "disowned")

KEYS: dict[str, str] = {
    "judge": "whether a given answer to a round-5 question is right or wrong, by that question's gate",
}


def n_tokens(line: str) -> int:
    """Token count as the student sees it (``student.tokens`` without importing torch)."""
    return len(line.replace(".", " . ").split())


def read_lines(path: Path) -> list[str]:
    return [ln.strip() for ln in path.open(encoding="utf-8") if ln.strip()]


def prompts_in(path: Path) -> set[str]:
    """The prompts of a line file's questions; a record line counts whole, without its final ``.``."""
    out: set[str] = set()
    for text in read_lines(path):
        parts = split_line(text)
        out.add(parts[0] if parts else text.rstrip("."))
    return out


def load_gates(topics: tuple[str, ...] = TOPICS) -> tuple[list[Gate], dict[str, str]]:
    """The gates of the topic modules that exist, and topic -> reason for each that could not load.

    A module another round is still writing (missing, half-written, without ``gate()``) is
    reported, never raised, so the loop runs with the gates it has.
    """
    gates: list[Gate] = []
    missing: dict[str, str] = {}
    for topic in topics:
        try:
            module = importlib.import_module(f"haishool.truth.{topic}")
            gate = module.gate()
            if not isinstance(gate, Gate):
                raise TypeError(f"{topic}.gate() gave a {type(gate).__name__}, not a Gate")
        except Exception as e:
            missing[topic] = f"{type(e).__name__}: {e}"
            continue
        gates.append(gate)
    return gates, missing


def all_gates(topics: tuple[str, ...] = TOPICS) -> list[Gate]:
    return load_gates(topics)[0]


def answer_text(got: object) -> str:
    """What ``ask`` gave, as single-spaced text; ``None`` is an empty answer."""
    return "" if got is None else " ".join(str(got).split())


def finite(answer: str) -> bool:
    """False when ``answer`` holds a number too large to be one.

    ``1 e 9 9 9 9`` parses to infinity, and infinity is within any relative tolerance of every
    value, so a gate that compares that way would accept it for any numeric question.

    >>> finite("1 8 point 0 1 5"), finite("1 e 9 9 9 9"), finite("x minus 1 e 9 9 9 9"), finite("h 2 o 1")
    (True, False, False, True)
    """
    number: list[str] = []
    for word in [*answer.split(), ""]:
        if word in DIGITS or word in NUMBER_WORDS:
            number.append(word)
            continue
        value = parse_num(number) if number else None
        if isinstance(value, float) and not math.isfinite(value):
            return False
        number = []
    return True


def gate_verdict(gate: Gate, prompt: str, answer: str) -> Verdict:
    """``gate.check`` behind the loop's two guards.

    A gate that raises on an answer has not judged it: the verdict is not ok, without an
    expected value, and its reason starts with :data:`GATE_ERROR`. An accepted answer that is
    not a finite number is rejected; the gate's own value is kept as the expected one.
    """
    try:
        verdict = gate.check(prompt, answer)
        if not isinstance(verdict, Verdict):
            raise TypeError(f"check() gave a {type(verdict).__name__}, not a Verdict")
    except Exception as e:
        return Verdict(False, None, f"{GATE_ERROR} {type(e).__name__}: {str(e)[:200]}")
    if verdict.ok and not finite(answer):
        own = verdict.expected is not None and verdict.expected != answer and finite(verdict.expected)
        return Verdict(False, verdict.expected if own else None, "not a finite number")
    return verdict


def _owns(gate: Gate, prompt: str) -> bool:
    try:
        return bool(gate.owns(prompt))
    except Exception:
        return False


def owners(prompt: str, gates: list[Gate]) -> list[Gate]:
    """Every gate that owns ``prompt``, in the order of ``gates``."""
    return [g for g in gates if _owns(g, prompt)]


def owner(prompt: str, gates: list[Gate]) -> Gate | None:
    """The first gate that owns ``prompt``."""
    return next(iter(owners(prompt, gates)), None)


def judge(prompt: str, answer: str, gates: list[Gate]) -> tuple[str, Verdict]:
    """The owning gate's topic and verdict; ``("", Verdict(False, None, "no gate"))`` when none owns it."""
    gate = owner(prompt, gates)
    if gate is None:
        return "", Verdict(False, None, "no gate")
    return gate.topic, gate_verdict(gate, prompt, answer)


def judge_prompt(prompt: str, answer: str) -> str:
    """``calc 1 2 plus 7``, ``2 0`` -> ``judge calc 1 2 plus 7 answer 2 0`` (an empty answer ends at ``answer``)."""
    return f"judge {prompt} answer {answer}".rstrip()


def is_judge_prompt(prompt: str) -> bool:
    """True for the form ``judge <question> answer [<given>]``, whoever owns the question."""
    words = prompt.split()
    return len(words) >= 3 and words[0] == "judge" and "answer" in words[2:]


def split_judge(prompt: str, gates: list[Gate]) -> tuple[Gate, str, str] | None:
    """The gate, inner question and judged answer of a judge prompt; ``None`` when no gate owns it.

    The word ``answer`` separates question from answer. Should a question or an answer contain
    it, the split whose question a gate owns wins (the first such split from the left).
    """
    if not is_judge_prompt(prompt):
        return None
    words = prompt.split()
    for i in range(2, len(words)):
        if words[i] == "answer":
            inner = " ".join(words[1:i])
            gate = owner(inner, gates)
            if gate is not None:
                return gate, inner, " ".join(words[i + 1:])
    return None


def check_judge(prompt: str, answer: str, gates: list[Gate]) -> Verdict:
    """Judge ``judge <question> answer <given>``: ``right`` when the question's gate accepts ``given``.

    A prompt that is not of that form, or whose question none of ``gates`` owns, is not the
    judge's question; a question whose gate raises on ``given`` has no expected value.
    """
    found = split_judge(prompt, gates)
    if found is None:
        return Verdict(False, None, "not my question")
    gate, inner, given = found
    inner_verdict = gate_verdict(gate, inner, given)
    reason = " ".join(w for w in (gate.topic, inner_verdict.reason) if w)
    if inner_verdict.reason.startswith(GATE_ERROR):
        return Verdict(False, None, reason)
    expected = "right" if inner_verdict.ok else "wrong"
    return Verdict(answer == expected, expected, reason)


class JudgeGate:
    """The loop's own gate: ``judge <question> answer <given>`` -> ``right`` or ``wrong``.

    It writes no questions of its own (those come out of :func:`run`) and judges by delegating
    to the gate that owns the inner question, so it is exactly as strong as the gates it holds
    and owns a judge prompt only when one of them owns its question. Without a list of gates it
    loads :func:`all_gates` on first use.
    """

    topic = "judge"
    KEYS = KEYS

    def __init__(self, gates: list[Gate] | None = None) -> None:
        self._gates = gates

    @property
    def gates(self) -> list[Gate]:
        if self._gates is None:
            self._gates = all_gates()
        return self._gates

    def records(self) -> list[Line]:
        return []

    def generate(self, rng: random.Random, n: int) -> list[Line]:
        return []

    def check(self, prompt: str, answer: str) -> Verdict:
        return check_judge(prompt, answer, self.gates)

    def owns(self, prompt: str) -> bool:
        return split_judge(prompt, self.gates) is not None


def kind_of(prompt: str, answer: str) -> str:
    """The bucket of a bare line, by its form alone: ``judge``, ``yesno``, ``number`` or ``words``."""
    if is_judge_prompt(prompt):
        return "judge"
    if answer in ("yes", "no"):
        return "yesno"
    if parse_num(answer) is not None:
        return "number"
    return "words"


def _line(prompt: str, answer: str) -> str | None:
    """``q prompt. a answer.`` when that is dense and short enough to train on, else ``None``."""
    try:
        text = Line(prompt, answer).text
    except ValueError:
        return None
    return text if n_tokens(text) <= MAX_TOKENS else None


def _rates(tally: dict[str, int]) -> dict:
    n = tally["n"]
    return {"n": n, "exact": round(tally["exact"] / n, 4) if n else None,
            "gate_ok": round(tally["gate_ok"] / n, 4) if n else None}


def _tally() -> dict[str, int]:
    return {"n": 0, "exact": 0, "gate_ok": 0}


def _bump(tally: dict[str, int], exact: bool, ok: bool) -> None:
    tally["n"] += 1
    tally["exact"] += exact
    tally["gate_ok"] += ok


def _score(by_kind: dict[str, dict[str, int]]) -> dict:
    out = {kind: _rates(t) for kind, t in sorted(by_kind.items())}
    total = _tally()
    for t in by_kind.values():
        for k in total:
            total[k] += t[k]
    out["all"] = _rates(total)
    return out


def _error(errors: dict[str, dict], topic: str, reason: str) -> None:
    errors.setdefault(topic, {"n": 0, "first": reason})["n"] += 1


def run(ask: Callable[[str], str], gates: list[Gate], rng: random.Random, n_per_topic: int,
        exclude: set[str], out_path: Path | None) -> dict:
    """Ask the model ``n_per_topic`` fresh questions per gate, judge them and write the lines back.

    ``exclude`` holds the prompts the model has already seen (training and sealed sets); the
    same underlying question in worked, check or judge form is excluded too. Such a
    question, like one asked twice, is replaced by another, up to :data:`MAX_ROUNDS` rounds of
    generation. Each question is judged by the gate that wrote it. ``exact`` is the model's
    answer against the gate's expected value (the verdict itself when the gate gives none).
    Every line written has passed once more: the question line through the gate that wrote it
    and every other gate of ``gates`` that owns it, the judge line through :func:`check_judge`
    over ``gates``, with all of those gates of one mind about the model's answer.
    """
    from haishool.truth.splits import canonical_prompt
    excluded_keys = {canonical_prompt(prompt, gates) for prompt in exclude}
    lines: list[str] = []
    counts = dict.fromkeys(COUNTS, 0)
    errors: dict[str, dict] = {}
    topics: dict[str, dict] = {}
    questions: dict[str, dict[str, int]] = {}
    overall: dict[str, dict[str, int]] = defaultdict(_tally)
    asked: set[str] = set()
    for gate in gates:
        fresh: list[Line] = []
        generated = excluded = 0
        for _ in range(MAX_ROUNDS):
            if len(fresh) >= n_per_topic:
                break
            try:
                batch = list(gate.generate(rng, n_per_topic))
            except Exception as e:
                _error(errors, gate.topic, f"{GATE_ERROR} {type(e).__name__}: {str(e)[:200]}")
                break
            for ln in batch:
                generated += 1
                if ln.kind == "record" or len(fresh) >= n_per_topic:
                    continue
                key = canonical_prompt(ln.prompt, gates)
                if key in excluded_keys:
                    excluded += 1
                elif key not in asked:
                    asked.add(key)
                    fresh.append(ln)
        by_kind: dict[str, dict[str, int]] = defaultdict(_tally)
        for ln in fresh:
            got = answer_text(ask(ln.prompt))
            verdict = gate_verdict(gate, ln.prompt, got)
            exact = got == verdict.expected if verdict.expected is not None else verdict.ok
            _bump(by_kind[ln.kind], exact, verdict.ok)
            _bump(overall[ln.kind], exact, verdict.ok)
            counts["disowned"] += not _owns(gate, ln.prompt)
            if verdict.reason.startswith(GATE_ERROR):
                counts["gate_errors"] += 1
                _error(errors, gate.topic, verdict.reason)
                continue
            # the gate's own value where it gives one; the model's only when the gate accepts it as it is
            value = verdict.expected if verdict.expected is not None else (got if verdict.ok else None)
            judges = [g for g in gates if g is gate or _owns(g, ln.prompt)]
            if value is not None and not all(gate_verdict(g, ln.prompt, value).ok for g in judges):
                counts["unverified"] += 2  # an owner rejects the value: neither line can be trusted
                continue
            if value is None:
                counts["uncorrected"] += 1
            elif (text := _line(ln.prompt, value)) is None:
                counts["skipped"] += 1
            else:
                lines.append(text)
                counts["confirmed" if verdict.ok else "corrected"] += 1
                counts["loose"] += verdict.ok and value != got
            label = "right" if verdict.ok else "wrong"
            judged = judge_prompt(ln.prompt, got)
            if (text := _line(judged, label)) is None:
                counts["skipped"] += 1
            elif not (check_judge(judged, label, gates).ok
                      and all(gate_verdict(g, ln.prompt, got).ok == verdict.ok for g in judges)):
                counts["unverified"] += 1
            else:
                lines.append(text)
                counts[f"judge_{label}"] += 1
        topics[gate.topic] = _score(by_kind)
        questions[gate.topic] = {"generated": generated, "excluded": excluded, "asked": len(fresh)}
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("".join(ln + "\n" for ln in lines), encoding="utf-8", newline="\n")
    counts["total"] = len(lines)
    return {"topics": topics, "overall": _score(overall), "questions": questions, "lines": counts,
            "gate_errors": errors, "out": str(out_path) if out_path is not None else None}


def evaluate(ask: Callable[[str], str], lines: list[str], gates: list[Gate]) -> dict:
    """Ask the model every question of a line file (a sealed test set) and score it per topic and kind.

    ``exact`` is the model's answer against the line's own, ``gate_ok`` the owning gate's verdict
    on the model's answer; the kind is read off the line's form (:func:`kind_of`). Record lines
    and questions no gate owns are counted, not asked. A question two gates own is scored by
    the first that accepts the line's own answer. ``gold_rejected`` counts the lines whose own
    answer no owning gate accepts, which says the file and the gates have drifted apart. Judge
    lines need a :class:`JudgeGate` among ``gates``.
    """
    topics: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(_tally))
    overall: dict[str, dict[str, int]] = defaultdict(_tally)
    errors: dict[str, dict] = {}
    records = unowned = gold_rejected = 0
    for text in lines:
        parts = split_line(text.strip())
        if parts is None:
            records += 1
            continue
        prompt, gold = parts
        owning = owners(prompt, gates)
        if not owning:
            unowned += 1
            continue
        gate = next((g for g in owning if gate_verdict(g, prompt, gold).ok), None)
        gold_rejected += gate is None
        gate = gate or owning[0]
        got = answer_text(ask(prompt))
        verdict = gate_verdict(gate, prompt, got)
        if GATE_ERROR in verdict.reason:
            _error(errors, gate.topic, verdict.reason)
        kind = kind_of(prompt, gold)
        _bump(topics[gate.topic][kind], got == gold, verdict.ok)
        _bump(overall[kind], got == gold, verdict.ok)
    return {"topics": {t: _score(k) for t, k in sorted(topics.items())}, "overall": _score(overall),
            "lines": len(lines), "records": records, "unowned": unowned, "gold_rejected": gold_rejected,
            "gate_errors": errors}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m haishool.truth.loop", description=__doc__.split("\n\n")[0])
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", type=Path, required=True, help="student checkpoint (student.pt)")
    common.add_argument("--device", default="cpu")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", parents=[common], help="ask fresh questions, judge, write the lines back")
    r.add_argument("--out", type=Path, required=True, help="where the confirmed, corrected and judge lines go")
    r.add_argument("--n", type=int, default=100, help="questions per topic")
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--exclude", type=Path, nargs="*", default=[],
                   help="line files whose prompts are not asked again (training set, sealed set)")
    e = sub.add_parser("evaluate", parents=[common], help="score the model on a line file")
    e.add_argument("--lines", type=Path, required=True, help="dense q/a lines, e.g. the sealed test set")
    return ap


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    for path in args.exclude if args.cmd == "run" else [args.lines]:
        if not path.is_file():
            raise SystemExit(f"no such line file: {path}")
    student = importlib.import_module("haishool.student")  # torch: only needed to talk to the model
    try:
        model, vocab = student.load(args.model, args.device)
    except Exception as e:
        raise SystemExit(f"cannot load {args.model}: {type(e).__name__}: {e} "
                         "(the loop needs torch and a trained student.pt to ask the model)") from e

    def ask(prompt: str) -> str:
        # room for the longest answer a line can hold; the default of 40 tokens would cut a long sum short
        return student.generate(model, vocab, f"q {prompt}. a", max_new=MAX_TOKENS, device=args.device)

    gates, missing = load_gates()
    if args.cmd == "run":
        exclude: set[str] = set().union(*(prompts_in(p) for p in args.exclude))
        report = run(ask, gates, random.Random(args.seed), args.n, exclude, args.out)
    else:
        report = evaluate(ask, read_lines(args.lines), [JudgeGate(gates), *gates])
    report["missing_gates"] = missing
    print(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    main()
