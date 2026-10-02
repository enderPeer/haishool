import json
import random
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest

from haishool.truth import Gate, Line, Verdict, is_dense, num, parse_num, split_line
from haishool.truth import loop
from haishool.truth.loop import (COUNTS, GATE_ERROR, MAX_TOKENS, JudgeGate, all_gates, answer_text, build_parser,
                                 check_judge, evaluate, finite, gate_verdict, is_judge_prompt, judge, judge_prompt,
                                 kind_of, load_gates, n_tokens, owner, owners, prompts_in, run, split_judge)


class FakeGate:
    """A closed toy topic: ``fake sum 1 2 plus 3 4`` -> ``4 6`` and ``fake even 7`` -> ``yes`` / ``no``."""

    topic = "fake"
    KEYS = {"sum": "the sum of two whole numbers", "even": "whether a whole number is even"}

    def records(self):
        return [Line("fake table. sum plus. even parity", topic="fake", kind="record")]

    def generate(self, rng, n):
        out = []
        for i in range(n):
            if i % 2 == 0:
                a, b = rng.randint(0, 99), rng.randint(0, 99)
                out.append(Line(f"fake sum {num(a)} plus {num(b)}", num(a + b), "fake", "calc"))
            else:
                a = rng.randint(0, 99)
                out.append(Line(f"fake even {num(a)}", "yes" if a % 2 == 0 else "no", "fake", "yesno"))
        return out

    def expected(self, prompt):
        words = prompt.split()
        if words[:2] == ["fake", "sum"] and "plus" in words:
            i = words.index("plus")
            a, b = parse_num(words[2:i]), parse_num(words[i + 1:])
            return None if a is None or b is None else num(a + b)
        if words[:2] == ["fake", "even"]:
            a = parse_num(words[2:])
            return None if a is None else ("yes" if a % 2 == 0 else "no")
        return None

    def check(self, prompt, answer):
        exp = self.expected(prompt)
        if exp is None:
            return Verdict(False, None, "not my question")
        return Verdict(answer == exp, exp, "by rule")

    def owns(self, prompt):
        return self.expected(prompt) is not None


class OtherGate(FakeGate):
    """A second toy topic, so routing between gates is tested: ``other double 7`` -> ``1 4``."""

    topic = "other"
    KEYS = {"double": "twice a whole number"}

    def generate(self, rng, n):
        return [Line(f"other double {num(a)}", num(2 * a), "other", "calc") for a in rng.sample(range(100), n)]

    def expected(self, prompt):
        words = prompt.split()
        if words[:2] != ["other", "double"]:
            return None
        a = parse_num(words[2:])
        return None if a is None else num(2 * a)


class TolerantGate(FakeGate):
    """``tol half 9`` -> ``4 point 5``, compared within five percent the way a careless gate does it:
    infinity passes, and a number too large for a float raises."""

    topic = "tol"
    KEYS = {"half": "half of a whole number"}

    def generate(self, rng, n):
        return [Line(f"tol half {num(a)}", num(a / 2), "tol", "calc") for a in rng.sample(range(1, 500), n)]

    def expected(self, prompt):
        words = prompt.split()
        if words[:2] != ["tol", "half"]:
            return None
        a = parse_num(words[2:])
        return None if a is None else num(a / 2)

    def check(self, prompt, answer):
        exp = self.expected(prompt)
        if exp is None:
            return Verdict(False, None, "not my question")
        got, want = parse_num(answer), parse_num(exp)
        if got is None:
            return Verdict(False, exp, "not a number")
        return Verdict(abs(got - want) <= 0.05 * max(abs(got), abs(want)), exp, "within 5 percent")


class LongGate(FakeGate):
    """One fixed question so long that its judge line would pass 80 tokens."""

    topic = "long"
    KEYS = {"count": "how many words"}
    PROMPT = "long count " + " ".join(["w"] * 72)

    def generate(self, rng, n):
        return [Line(self.PROMPT, num(72), "long", "calc")]

    def expected(self, prompt):
        words = prompt.split()
        return num(len(words) - 2) if words[:2] == ["long", "count"] else None


class ChainGate(FakeGate):
    """``chain ones 4 5`` -> forty-five times ``1``: answers longer than the 40 tokens ``generate`` gives by default."""

    topic = "chain"
    KEYS = {"ones": "that many ones"}

    def generate(self, rng, n):
        return [Line(f"chain ones {num(k)}", " ".join(["1"] * k), "chain", "calc") for k in rng.sample(range(41, 70), n)]

    def expected(self, prompt):
        words = prompt.split()
        if words[:2] != ["chain", "ones"]:
            return None
        k = parse_num(words[2:])
        return None if k is None else " ".join(["1"] * k)


HUGE = " ".join("9" * 400)
#: what a broken model may say: nothing, specials, capitals, full stops, other alphabets, numbers
#: that are none, the loop's own words, far too much
MESS = ["", None, "<pad> 1", "Foo.", "a . b", "ß", 0, "1 e 9 9 9 9", "answer 3", "  1   2  ", "w " * 200, "right",
        "q x. a y.", HUGE]


def wrong(expected):
    if expected in ("yes", "no"):
        return "no" if expected == "yes" else "yes"
    if expected in ("right", "wrong"):
        return "wrong" if expected == "right" else "right"
    return num(parse_num(expected) + 1)


def half_right(gates, log):
    """An ask() that answers every other question right (by the gates' own rules) and logs its answers.

    A prompt none of the gates owns (a judge line, ``turkey capital``) gets an empty answer."""
    def ask(prompt):
        exp = next((g.expected(prompt) for g in gates if g.owns(prompt)), None)
        if exp is None:
            return ""
        got = exp if len(log) % 2 == 0 else wrong(exp)
        log.append((prompt, got))
        return got
    return ask


def messy(gates, log):
    """An ask() that goes round: the right answer, a wrong one, the next piece of :data:`MESS`."""
    def ask(prompt):
        exp = next(g.expected(prompt) for g in gates if g.owns(prompt))
        i = len(log)
        got = exp if i % 3 == 0 else wrong(exp) if i % 3 == 1 else MESS[(i // 3) % len(MESS)]
        log.append((prompt, got))
        return got
    return ask


def check_report(report, n_lines):
    """The sums every ``run`` report must satisfy."""
    c = report["lines"]
    assert tuple(c) == COUNTS + ("total",)
    asked = sum(q["asked"] for q in report["questions"].values())
    assert c["total"] == n_lines == c["confirmed"] + c["corrected"] + c["judge_right"] + c["judge_wrong"]
    written_or_not = c["total"] + c["uncorrected"] + c["skipped"] + c["unverified"]
    assert written_or_not == 2 * (asked - c["gate_errors"]), "every question ends in two counted outcomes"
    assert 0 <= c["loose"] <= c["confirmed"]
    assert sum(t["all"]["n"] for t in report["topics"].values()) == asked == report["overall"]["all"]["n"]
    for topic, q in report["questions"].items():
        assert report["topics"][topic]["all"]["n"] == q["asked"] <= q["generated"]
        assert sum(k["n"] for kind, k in report["topics"][topic].items() if kind != "all") == q["asked"]
    assert c["gate_errors"] <= sum(e["n"] for e in report["gate_errors"].values())


def check_lines(lines, gates):
    """What holds for every line the loop writes: dense, short, single, finite and true by its gate."""
    jg = JudgeGate(gates)
    assert len(set(lines)) == len(lines)
    for text in lines:
        assert is_dense(text) and n_tokens(text) <= MAX_TOKENS, text
        prompt, answer = split_line(text)
        assert "." not in prompt and "." not in answer
        if is_judge_prompt(prompt):
            assert answer in ("right", "wrong") and jg.owns(prompt), text
            assert jg.check(prompt, answer) == Verdict(True, answer, jg.check(prompt, answer).reason), text
            bad = jg.check(prompt, wrong(answer))
            assert not bad.ok and bad.expected == answer, text
        else:
            assert finite(answer) and not jg.owns(prompt), text
            owning = owners(prompt, gates)
            assert owning and all(gate_verdict(g, prompt, answer).ok for g in owning), text


@pytest.fixture
def gates():
    return [FakeGate(), OtherGate()]


def test_fake_gates_follow_the_protocol(gates):
    for g in gates + [TolerantGate(), LongGate()]:
        assert isinstance(g, Gate)
        for ln in g.generate(random.Random(3), 20):
            assert g.owns(ln.prompt) and g.check(ln.prompt, ln.answer).ok
    for g in gates:
        for ln in g.generate(random.Random(3), 20):
            bad = g.check(ln.prompt, wrong(ln.answer))
            assert not bad.ok and bad.expected == ln.answer
    assert not gates[0].owns("turkey capital") and not gates[0].owns("other double 3")


def test_judge_routes_to_the_owner(gates):
    assert judge("fake sum 1 plus 2", "3", gates) == ("fake", Verdict(True, "3", "by rule"))
    assert judge("other double 7", "1 5", gates) == ("other", Verdict(False, "1 4", "by rule"))
    assert judge("turkey capital", "ankara", gates) == ("", Verdict(False, None, "no gate"))
    assert owner("other double 7", gates) is gates[1] and owner("q spoon color", gates) is None

    class Touchy(FakeGate):
        def owns(self, prompt):
            raise KeyError(prompt)

    assert owner("other double 7", [Touchy(), *gates]) is gates[1]  # a gate that cannot say is not the owner


def test_judge_prompts(gates):
    assert judge_prompt("calc 1 2 plus 7", "2 0") == "judge calc 1 2 plus 7 answer 2 0"
    assert judge_prompt("calc 1 2 plus 7", "") == "judge calc 1 2 plus 7 answer"
    assert is_judge_prompt("judge calc 1 2 plus 7 answer 2 0") and is_judge_prompt("judge x answer")
    assert not is_judge_prompt("judge") and not is_judge_prompt("judge answer 3") and not is_judge_prompt("calc 1 plus 1")
    assert split_judge("judge fake sum 1 plus 2 answer 3", gates) == (gates[0], "fake sum 1 plus 2", "3")
    assert split_judge("judge fake sum 1 plus 2 answer", gates) == (gates[0], "fake sum 1 plus 2", "")
    # an answer that holds the word ``answer`` still splits after the question
    assert split_judge("judge other double 7 answer answer 3 answer", gates) == (gates[1], "other double 7", "answer 3 answer")
    assert split_judge("judge turkey capital answer ankara", gates) is None
    assert split_judge("fake sum 1 plus 2", gates) is None


def test_check_judge_right_and_wrong(gates):
    assert check_judge("judge fake sum 1 plus 2 answer 3", "right", gates) == Verdict(True, "right", "fake by rule")
    assert check_judge("judge fake sum 1 plus 2 answer 3", "wrong", gates) == Verdict(False, "right", "fake by rule")
    assert check_judge("judge fake sum 1 plus 2 answer 4", "wrong", gates).ok
    assert check_judge("judge fake sum 1 plus 2 answer 4", "right", gates) == Verdict(False, "wrong", "fake by rule")
    assert check_judge("judge other double 7 answer 1 4", "right", gates).ok
    assert check_judge("judge fake sum 1 plus 2 answer", "wrong", gates).ok  # an empty answer is wrong
    assert check_judge("judge fake sum 1 plus 2 answer 3", "", gates) == Verdict(False, "right", "fake by rule")
    assert check_judge("judge fake sum 1 plus 2 answer 3", "yes", gates) == Verdict(False, "right", "fake by rule")
    # the contract's words for anything the judge cannot judge
    assert check_judge("judge turkey capital answer ankara", "right", gates) == Verdict(False, None, "not my question")
    assert check_judge("fake sum 1 plus 2", "3", gates) == Verdict(False, None, "not my question")


def test_judge_gate_owns_what_it_can_judge(gates):
    jg = JudgeGate(gates)
    assert isinstance(jg, Gate) and jg.topic == "judge" and "judge" in jg.KEYS
    assert jg.generate(random.Random(1), 10) == [] and jg.records() == []
    for prompt in ("judge fake sum 1 plus 2 answer 3", "judge fake even 7 answer", "judge other double 7 answer x y"):
        assert jg.owns(prompt), prompt
    for prompt in ("fake sum 1 plus 2", "other double 7", "turkey capital", "q spoon color", "judge", "judge answer 3",
                   "carbon protons", "calc 1 2 plus 7", "judge turkey capital answer ankara",
                   "judge judge fake sum 1 plus 2 answer 3 answer right"):
        assert not jg.owns(prompt), prompt
        assert jg.check(prompt, "right") == Verdict(False, None, "not my question"), prompt
    assert not any(g.owns("judge fake sum 1 plus 2 answer 3") for g in gates)
    assert jg.check("judge fake sum 1 plus 2 answer 3", "right").ok
    lazy = JudgeGate()
    assert isinstance(lazy.gates, list)  # loads whatever real gates exist, never crashes
    assert not lazy.owns("turkey capital") and not lazy.owns("q spoon color")


def test_answers_are_taken_as_text():
    assert answer_text(None) == "" and answer_text("") == "" and answer_text("  1   2 \n") == "1 2"
    assert answer_text(0) == "0" and answer_text(19) == "19" and answer_text("yes") == "yes"


def test_finite_numbers():
    for ok in ("", "1 8 point 0 1 5", "minus 4", "6 point 6 7 4 e minus 1 1", "h 2 o 1", "x 5", "e", "yes",
               "1 e 3 0 8", HUGE, "1 e 9 9 9 9 e 1", "minus point"):
        assert finite(ok), ok
    for bad in ("1 e 9 9 9 9", "minus 1 e 9 9 9 9", "x 1 e 9 9 9 9", "1 e 9 9 9 9 newtons", "1 e 3 0 9",
                HUGE + " point 5", "2 and 1 e 9 9 9 9 and 3"):
        assert not finite(bad), bad
    assert parse_num("1 e 9 9 9 9") == float("inf")  # why the guard is needed


def test_gate_verdict_guards_the_gate():
    tol = TolerantGate()
    assert tol.check("tol half 9", "1 e 9 9 9 9").ok  # the hole: infinity is within five percent of anything
    assert gate_verdict(tol, "tol half 9", "1 e 9 9 9 9") == Verdict(False, "4 point 5", "not a finite number")
    assert gate_verdict(tol, "tol half 9", "minus 1 e 9 9 9 9") == Verdict(False, "4 point 5", "not a finite number")
    assert gate_verdict(tol, "tol half 9", "4 point 5") == Verdict(True, "4 point 5", "within 5 percent")
    assert gate_verdict(tol, "tol half 9", "4 point 6").ok  # loose but finite stays the gate's business
    assert gate_verdict(tol, "tol half 9", "9") == Verdict(False, "4 point 5", "within 5 percent")
    with pytest.raises(OverflowError):
        tol.check("tol half 9", HUGE)
    verdict = gate_verdict(tol, "tol half 9", HUGE)
    assert not verdict.ok and verdict.expected is None and verdict.reason.startswith(GATE_ERROR + " OverflowError")

    class Silent(FakeGate):
        def check(self, prompt, answer):
            return None

    class Echo(FakeGate):
        def check(self, prompt, answer):
            return Verdict(True, answer, "whatever you say")

    assert gate_verdict(Silent(), "fake sum 1 plus 2", "3").reason.startswith(GATE_ERROR + " TypeError")
    # a gate that hands the infinite answer back as its own value: rejected, and nothing to correct with
    assert gate_verdict(Echo(), "fake sum 1 plus 2", "1 e 9 9 9 9") == Verdict(False, None, "not a finite number")


def test_run_half_right(gates, tmp_path):
    out = tmp_path / "loop.txt"
    log = []
    report = run(half_right(gates, log), gates, random.Random(1), 10, set(), out)
    assert len(log) == 20 and len({p for p, _ in log}) == 20
    for topic in ("fake", "other"):
        assert report["topics"][topic]["all"] == {"n": 10, "exact": 0.5, "gate_ok": 0.5}
        assert report["questions"][topic] == {"generated": 10, "excluded": 0, "asked": 10}
    assert report["topics"]["fake"]["calc"]["n"] == 5 and report["topics"]["fake"]["yesno"]["n"] == 5
    assert report["overall"]["all"] == {"n": 20, "exact": 0.5, "gate_ok": 0.5}
    assert report["lines"] == {"confirmed": 10, "corrected": 10, "uncorrected": 0, "loose": 0, "judge_right": 10,
                               "judge_wrong": 10, "skipped": 0, "unverified": 0, "gate_errors": 0, "disowned": 0,
                               "total": 40}
    assert report["gate_errors"] == {} and report["out"] == str(out)
    lines = out.read_text(encoding="utf-8").split("\n")
    assert lines[-1] == "" and len(lines) == 41
    lines = lines[:-1]
    check_report(report, len(lines))
    for i, (prompt, got) in enumerate(log):
        exp = next(g.expected(prompt) for g in gates if g.owns(prompt))
        right = i % 2 == 0
        assert lines[2 * i] == f"q {prompt}. a {got if right else exp}."
        assert lines[2 * i + 1] == f"q judge {prompt} answer {got}. a {'right' if right else 'wrong'}."
    assert run(half_right(gates, []), gates, random.Random(1), 10, set(), None)["out"] is None  # no file wanted


def test_every_written_line_is_dense_short_and_true(gates, tmp_path):
    out = tmp_path / "loop.txt"
    report = run(half_right(gates, []), gates, random.Random(5), 15, set(), out)
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 60
    check_report(report, 60)
    check_lines(lines, gates)
    jg = JudgeGate(gates)
    for text in lines:
        prompt, answer = split_line(text)
        gate = jg if jg.owns(prompt) else next(g for g in gates if g.owns(prompt))
        assert gate.check(prompt, answer).ok, text
        # a changed word or digit is rejected, with the expected value filled in
        bad = gate.check(prompt, wrong(answer))
        assert not bad.ok and bad.expected == answer, text


def test_run_survives_a_messy_model(tmp_path):
    gates = [FakeGate(), OtherGate(), TolerantGate()]
    first = [ln.prompt for g in gates for ln in g.generate(random.Random(1), 5)]
    exclude = set(first[::2]) | {"turkey capital", "q spoon color"}
    out, log = tmp_path / "loop.txt", []
    report = run(messy(gates, log), gates, random.Random(1), 40, exclude, out)  # nothing raises
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(log) == 120 and {type(got) for _, got in log} == {str, int, type(None)}
    assert all(report["questions"][g.topic]["asked"] == 40 for g in gates)
    check_report(report, len(lines))
    check_lines(lines, gates)
    # no excluded prompt is asked, written or judged
    assert not exclude & {p for p, _ in log}
    for prompt, _ in map(split_line, lines):
        assert prompt not in exclude and not any(prompt.startswith(f"judge {seen} answer") for seen in exclude)
    # every outcome follows from what the model said, by the gates' own verdicts
    want = dict.fromkeys(COUNTS, 0)
    for prompt, said in log:
        got, gate = answer_text(said), owner(prompt, gates)
        verdict = gate_verdict(gate, prompt, got)
        if verdict.reason.startswith(GATE_ERROR):
            want["gate_errors"] += 1
            assert not any(f" {prompt}." in t or f" {prompt} answer" in t for t in lines), "nothing is written for it"
            continue
        want["confirmed" if verdict.ok else "corrected"] += 1
        want["loose"] += verdict.ok and got != verdict.expected
        assert f"q {prompt}. a {verdict.expected}." in lines, "the gate's own value is what is trained"
        label = "right" if verdict.ok else "wrong"
        judged = f"q judge {prompt} answer {got}. a {label}.".replace(" answer .", " answer.")
        if "." not in got and is_dense(judged) and n_tokens(judged) <= MAX_TOKENS:  # a full stop would end the question
            want[f"judge_{label}"] += 1
            assert judged in lines
        else:
            want["skipped"] += 1
    assert {k: v for k, v in report["lines"].items() if k != "total"} == want
    assert want["gate_errors"] == 1 and report["gate_errors"]["tol"]["n"] == 1  # the 400-digit answer
    assert report["gate_errors"]["tol"]["first"].startswith(GATE_ERROR + " OverflowError")
    assert want["skipped"] >= 12 and want["loose"] >= 1 and want["judge_right"] >= 40
    assert not any(" 1 e 9 9 9 9. a right." in t or t.endswith(" a 1 e 9 9 9 9.") for t in lines)


def test_an_infinite_answer_is_never_confirmed(tmp_path):
    tol, out = TolerantGate(), tmp_path / "loop.txt"
    report = run(lambda p: "1 e 9 9 9 9", [tol], random.Random(3), 8, set(), out)
    lines = out.read_text(encoding="utf-8").splitlines()
    assert report["lines"]["confirmed"] == 0 and report["lines"]["corrected"] == 8
    assert report["lines"]["judge_wrong"] == 8 and report["lines"]["judge_right"] == 0 and len(lines) == 16
    assert report["topics"]["tol"]["all"] == {"n": 8, "exact": 0.0, "gate_ok": 0.0}
    jg = JudgeGate([tol])
    for asked, judged in zip(lines[::2], lines[1::2]):
        prompt, answer = split_line(asked)
        assert answer == tol.expected(prompt) and finite(answer)
        assert judged == f"q judge {prompt} answer 1 e 9 9 9 9. a wrong." and jg.check(*split_line(judged)).ok
    assert jg.check("judge tol half 9 answer 1 e 9 9 9 9", "right") == Verdict(False, "wrong", "tol not a finite number")
    assert evaluate(lambda p: "1 e 9 9 9 9", ["q tol half 9. a 4 point 5."], [tol])["overall"]["all"]["gate_ok"] == 0.0


def test_a_loose_answer_is_trained_in_the_gates_form(tmp_path):
    tol, out = TolerantGate(), tmp_path / "loop.txt"

    def ask(prompt):  # one percent high: accepted, but not the gate's value
        return num(parse_num(tol.expected(prompt)) * 1.01, 6)

    report = run(ask, [tol], random.Random(3), 6, set(), out)
    lines = out.read_text(encoding="utf-8").splitlines()
    assert report["topics"]["tol"]["all"] == {"n": 6, "exact": 0.0, "gate_ok": 1.0}
    assert report["lines"]["confirmed"] == 6 and report["lines"]["loose"] == 6 and report["lines"]["judge_right"] == 6
    check_report(report, 12)
    check_lines(lines, [tol])
    for asked, judged in zip(lines[::2], lines[1::2]):
        prompt, answer = split_line(asked)
        assert answer == tol.expected(prompt) != ask(prompt)
        assert judged == f"q judge {prompt} answer {ask(prompt)}. a right."


def test_run_skips_excluded_prompts(gates, tmp_path):
    first = gates[0].generate(random.Random(1), 10)
    exclude = {ln.prompt for ln in first[:3]} | {"turkey capital"}
    report = run(half_right(gates, []), gates, random.Random(1), 10, exclude, tmp_path / "loop.txt")
    prompts = {split_line(t)[0] for t in (tmp_path / "loop.txt").read_text(encoding="utf-8").splitlines()}
    for ln in first[:3]:
        assert ln.prompt not in prompts and f"judge {ln.prompt}" not in " ".join(prompts)
    assert report["questions"]["fake"]["excluded"] >= 3 and report["questions"]["fake"]["asked"] == 10
    assert report["questions"]["fake"]["generated"] > 10  # topped up from a second round
    # with everything excluded nothing is asked and nothing written
    everything = {ln.prompt for seed in range(40) for ln in LongGate().generate(random.Random(seed), 1)}
    asked = []
    report = run(asked.append, [LongGate()], random.Random(1), 5, everything, tmp_path / "none.txt")
    assert asked == [] and report["lines"]["total"] == 0 and (tmp_path / "none.txt").read_text(encoding="utf-8") == ""
    assert report["questions"]["long"] == {"generated": 4, "excluded": 4, "asked": 0}
    assert report["topics"]["long"] == {"all": {"n": 0, "exact": None, "gate_ok": None}}


def test_run_is_deterministic(gates, tmp_path):
    reports, texts = [], []
    for seed in (7, 7, 8):
        out = tmp_path / f"loop-{len(texts)}.txt"
        reports.append(run(half_right(gates, []), gates, random.Random(seed), 12, set(), out))
        texts.append(out.read_text(encoding="utf-8"))
    assert texts[0] == texts[1] and texts[0] != texts[2]
    reports[0]["out"] = reports[1]["out"] = None
    assert reports[0] == reports[1]


def test_run_skips_lines_it_cannot_write(gates, tmp_path):
    out = tmp_path / "loop.txt"
    report = run(lambda p: "<pad> 1", [gates[0]], random.Random(1), 6, set(), out)
    lines = out.read_text(encoding="utf-8").splitlines()
    assert report["lines"]["confirmed"] == 0 and report["lines"]["corrected"] == 6
    assert report["lines"]["skipped"] == 6 and report["lines"]["judge_wrong"] == 0 and len(lines) == 6
    assert all(is_dense(t) and gates[0].check(*split_line(t)).ok for t in lines)
    # an empty answer still gives a judge line, which the judge gate accepts
    report = run(lambda p: "", [gates[0]], random.Random(1), 4, set(), out)
    lines = out.read_text(encoding="utf-8").splitlines()
    assert report["lines"] == {"confirmed": 0, "corrected": 4, "uncorrected": 0, "loose": 0, "judge_right": 0,
                               "judge_wrong": 4, "skipped": 0, "unverified": 0, "gate_errors": 0, "disowned": 0,
                               "total": 8}
    assert lines[1].endswith(" answer. a wrong.") and JudgeGate(gates).check(*split_line(lines[1])).ok
    # a judge line that would pass 80 tokens is not written, the confirmed line still is
    long = LongGate()
    report = run(lambda p: num(72), [long], random.Random(1), 1, set(), out)
    lines = out.read_text(encoding="utf-8").splitlines()
    assert report["lines"]["confirmed"] == 1 and report["lines"]["skipped"] == 1 and len(lines) == 1
    assert n_tokens(lines[0]) <= 80 and n_tokens(Line(judge_prompt(long.PROMPT, num(72)), "right").text) > 80
    check_report(report, 1)


def test_run_reports_a_gate_that_cannot_correct(tmp_path):
    class Mute(FakeGate):
        def check(self, prompt, answer):
            v = super().check(prompt, answer)
            return v if v.ok else Verdict(False, None, "wrong, but not saying what is right")

    report = run(lambda p: "1 0 0 0", [Mute()], random.Random(2), 4, set(), tmp_path / "loop.txt")
    assert report["lines"]["uncorrected"] == 4 and report["lines"]["corrected"] == 0
    assert report["lines"]["judge_wrong"] == 4 and report["lines"]["total"] == 4
    assert report["topics"]["fake"]["all"]["gate_ok"] == 0.0
    check_report(report, 4)


def test_a_gate_that_rejects_its_own_value_writes_nothing(tmp_path):
    class Confused(FakeGate):
        """Says what the answer should be, then refuses that very answer."""

        def check(self, prompt, answer):
            exp = self.expected(prompt)
            return Verdict(False, exp, "never satisfied") if exp is not None else Verdict(False, None, "not my question")

    out = tmp_path / "loop.txt"
    report = run(lambda p: "7", [Confused()], random.Random(2), 5, set(), out)
    assert report["lines"]["unverified"] == 10 and report["lines"]["total"] == 0
    assert out.read_text(encoding="utf-8") == ""
    check_report(report, 0)

    class Disowning(FakeGate):
        def owns(self, prompt):
            return False

    # a gate that does not own what it asks: counted, and its judge lines cannot be checked, so none is written
    report = run(lambda p: "7", [Disowning()], random.Random(2), 5, set(), out)
    assert report["lines"]["disowned"] == 5 and report["lines"]["unverified"] == 5
    assert report["lines"]["judge_wrong"] + report["lines"]["judge_right"] == 0
    check_report(report, len(out.read_text(encoding="utf-8").splitlines()))


def test_two_gates_that_own_one_question_must_agree(gates, tmp_path):
    class Twin(FakeGate):
        """Owns the fake topic's questions as well, asks none, says the same."""

        topic = "twin"

        def generate(self, rng, n):
            return []

    class Rival(Twin):
        """Owns ``fake even ...`` only, and says the opposite."""

        topic = "rival"

        def expected(self, prompt):
            exp = super().expected(prompt)
            return wrong(exp) if exp in ("yes", "no") else None

    fake, out = gates[0], tmp_path / "loop.txt"
    assert owners("fake even 7", [fake, Rival(), Twin()]) != [fake] and owners("fake sum 1 plus 2", [fake, Rival()]) == [fake]
    # of one mind: everything is written, once
    report = run(half_right([fake], []), [Twin(), fake], random.Random(1), 10, set(), out)
    assert report["lines"]["total"] == 20 and report["lines"]["unverified"] == 0 and report["questions"]["twin"]["asked"] == 0
    check_lines(out.read_text(encoding="utf-8").splitlines(), [Twin(), fake])
    # at odds over the even questions: none of them is written, the sums still are
    log = []
    report = run(half_right([fake], log), [fake, Rival()], random.Random(1), 10, set(), out)
    lines = out.read_text(encoding="utf-8").splitlines()
    assert report["lines"]["unverified"] == 10 and report["lines"]["total"] == 10 == len(lines)
    assert all("fake sum" in t for t in lines) and sum("fake even" in p for p, _ in log) == 5
    check_report(report, 10)
    # a sealed line both own is scored by the gate that accepts its answer; one that neither accepts is counted
    sealed = ["q fake even 7. a no.", "q fake even 8. a yes.", "q fake even 9. a maybe."]
    report = evaluate(lambda p: "no", sealed, [Rival(), fake])
    assert report["topics"]["fake"]["all"] == {"n": 2, "exact": 0.5, "gate_ok": 0.5}
    assert report["topics"]["rival"]["all"] == {"n": 1, "exact": 0.0, "gate_ok": 0.0} and report["gold_rejected"] == 1
    assert evaluate(lambda p: "no", sealed, [Rival()])["gold_rejected"] == 3  # the rival alone accepts none of them


def test_a_gate_that_raises_is_reported_not_raised(gates, tmp_path):
    class Brittle(FakeGate):
        topic = "brittle"

        def check(self, prompt, answer):
            if answer == "boom":
                raise RuntimeError("cannot parse boom")
            return super().check(prompt, answer)

    class Broken(OtherGate):
        topic = "broken"

        def generate(self, rng, n):
            raise ZeroDivisionError("half written")

    out = tmp_path / "loop.txt"
    report = run(lambda p: "boom", [Brittle(), Broken(), gates[1]], random.Random(1), 4, set(), out)
    assert report["gate_errors"] == {"brittle": {"n": 4, "first": "gate error RuntimeError: cannot parse boom"},
                                     "broken": {"n": 1, "first": "gate error ZeroDivisionError: half written"}}
    assert report["lines"]["gate_errors"] == 4 and report["questions"]["broken"] == {"generated": 0, "excluded": 0, "asked": 0}
    assert report["topics"]["brittle"]["all"] == {"n": 4, "exact": 0.0, "gate_ok": 0.0}
    # the gate after the broken ones is still asked, and only its lines are written
    assert report["questions"]["other"]["asked"] == 4 and report["lines"]["corrected"] == 4
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 8 and all("other double" in t for t in lines)
    check_report(report, 8)
    # what the gate could not judge, the judge gate cannot judge either
    jg = JudgeGate([Brittle()])
    unjudged = Verdict(False, None, "brittle gate error RuntimeError: cannot parse boom")
    assert jg.check("judge fake sum 1 plus 2 answer boom", "wrong") == unjudged
    report = evaluate(lambda p: "boom", ["q fake sum 1 plus 2. a 3.", "q judge fake sum 1 plus 2 answer boom. a wrong."],
                      [Brittle(), jg])
    assert report["gate_errors"] == {"brittle": {"n": 1, "first": "gate error RuntimeError: cannot parse boom"},
                                     "judge": {"n": 1, "first": "brittle gate error RuntimeError: cannot parse boom"}}
    assert report["topics"]["brittle"]["all"] == {"n": 1, "exact": 0.0, "gate_ok": 0.0}


def test_evaluate(gates):
    sealed = [ln.text for ln in gates[0].generate(random.Random(9), 8) + gates[1].generate(random.Random(9), 4)]
    sealed += ["fake table. sum plus. even parity.", "q turkey capital. a ankara.",
               "q judge fake sum 1 plus 2 answer 3. a right.", "q judge fake sum 1 plus 2 answer 4. a wrong."]
    log = []
    report = evaluate(half_right(gates, log), sealed, gates + [JudgeGate(gates)])
    assert len(log) == 12
    assert report["lines"] == 16 and report["records"] == 1 and report["unowned"] == 1
    assert report["gold_rejected"] == 0 and report["gate_errors"] == {}
    assert report["topics"]["fake"]["all"] == {"n": 8, "exact": 0.5, "gate_ok": 0.5}
    assert report["topics"]["other"]["all"] == {"n": 4, "exact": 0.5, "gate_ok": 0.5}
    assert report["topics"]["fake"]["number"]["n"] == 4 and report["topics"]["fake"]["yesno"]["n"] == 4
    assert report["topics"]["judge"]["judge"] == {"n": 2, "exact": 0.0, "gate_ok": 0.0}  # answered ""
    assert report["overall"]["all"] == {"n": 14, "exact": round(6 / 14, 4), "gate_ok": round(6 / 14, 4)}
    # the judge lines are scored by the judge gate, with an ask that answers them by their gate
    report = evaluate(lambda p: check_judge(p, "", gates).expected, sealed[-2:], gates + [JudgeGate(gates)])
    assert report["topics"]["judge"]["judge"] == {"n": 2, "exact": 1.0, "gate_ok": 1.0}
    # without a judge gate among the gates, judge lines are unowned; an unowned question is not even asked
    asked = []
    assert evaluate(asked.append, sealed[-2:] + ["q turkey capital. a ankara."], gates)["unowned"] == 3 and asked == []
    # a model that answers nothing, None or rubbish is scored, not crashed on
    for said in (None, "", "<pad>", "Foo.", HUGE, 0):
        report = evaluate(lambda p: said, sealed, gates + [JudgeGate(gates)])
        assert report["overall"]["all"]["n"] == 14 and report["overall"]["all"]["exact"] == 0.0
    # a line file whose own answers the gates reject has drifted from them
    stale = ["q fake sum 1 plus 2. a 4.", "q judge fake sum 1 plus 2 answer 3. a wrong.", "q other double 7. a 1 4."]
    assert evaluate(lambda p: "", stale, gates + [JudgeGate(gates)])["gold_rejected"] == 2


def test_kinds_tokens_and_prompt_files(tmp_path):
    assert kind_of("judge calc 1 plus 1 answer 2", "right") == "judge"
    assert kind_of("check 1 plus 1 equals 2", "yes") == "yesno"
    assert kind_of("calc 1 plus 1", "2") == "number" and kind_of("water formula", "h 2 o 1") == "words"
    assert n_tokens("q calc 1 2 plus 7. a 1 9.") == 11  # q calc 1 2 plus 7 . a 1 9 .
    path = tmp_path / "lines.txt"
    path.write_text("q carbon protons. a 6.\n\ncarbon element. number 6.\nq judge x answer 1. a right.\n", encoding="utf-8")
    assert prompts_in(path) == {"carbon protons", "carbon element. number 6", "judge x answer 1"}


def test_tokens_are_counted_as_the_student_counts_them():
    student = pytest.importorskip("haishool.student")  # imports without torch
    for text in ("q calc 1 2 plus 7. a 1 9.", "carbon element. number 6. symbol c.", "q judge x answer. a wrong.",
                 "q a b. a " + " ".join(["1"] * 90) + "."):
        assert n_tokens(text) == len(student.tokens(text))
    assert MAX_TOKENS == 80 and MAX_TOKENS + 1 < student.CTX  # a line and its <eos> fit one window


def test_all_gates_never_crashes():
    gates, missing = load_gates(("nonexistent_topic_xyz", "formula"))
    assert gates == [] and missing["nonexistent_topic_xyz"].startswith("ModuleNotFoundError")
    assert missing["formula"].startswith("AttributeError")  # a module without gate()
    real = all_gates()  # whatever topic modules exist right now; none is also fine
    assert isinstance(real, list) and all(isinstance(g, Gate) for g in real)


def test_real_gates_through_the_loop(tmp_path):
    """The round's own gates, as far as they are written: a model that is right, wrong, infinite and
    rubbish in turn leaves only lines its gates accept."""
    real, _ = load_gates()
    if not real:
        pytest.skip("no topic module is here yet")
    said = []

    def ask(prompt):
        gate = owner(prompt, real)
        right = gate_verdict(gate, prompt, "").expected if gate is not None else None
        said.append(prompt)
        turn = len(said) % 4
        return right if turn == 0 else "1 e 9 9 9 9" if turn == 1 else "7 7 7" if turn == 2 else MESS[len(said) % len(MESS)]

    texts = []
    for name in ("one", "two"):
        report = run(ask, real, random.Random(11), 24, {"turkey capital"}, tmp_path / f"{name}.txt")
        texts.append((tmp_path / f"{name}.txt").read_text(encoding="utf-8"))
        said.clear()
    lines = texts[0].splitlines()
    assert texts[0] == texts[1], "the same seed and the same answers give the same lines"
    check_report(report, len(lines))
    check_lines(lines, real)
    assert not any(t.endswith(" a 1 e 9 9 9 9.") or t.endswith(" answer 1 e 9 9 9 9. a right.") for t in lines)
    jg = JudgeGate(real)
    assert not jg.owns("turkey capital") and not jg.owns("q spoon color")
    for prompt, _ in map(split_line, lines):
        assert jg.owns(prompt) == is_judge_prompt(prompt), prompt
    # what the loop cannot mend is the topic's own business and its own tests'; here it is only said aloud
    foreign = [g.topic for g in real if any(map(g.owns, ("turkey capital", "q spoon color", "judge calc 1 2 plus 7 answer 1 9")))]
    if report["lines"]["unverified"] or report["lines"]["disowned"] or report["gate_errors"] or foreign:
        warnings.warn(f"gates at odds: {report['lines']}, errors {report['gate_errors']}, owning foreign prompts: {foreign}")


def test_cli_parses_arguments():
    args = build_parser().parse_args(["run", "--model", "m.pt", "--out", "o.txt", "--n", "5", "--seed", "3",
                                      "--exclude", "a.txt", "b.txt", "--device", "cuda"])
    assert args.cmd == "run" and args.model == Path("m.pt") and args.out == Path("o.txt")
    assert args.n == 5 and args.seed == 3 and args.exclude == [Path("a.txt"), Path("b.txt")] and args.device == "cuda"
    args = build_parser().parse_args(["run", "--model", "m.pt", "--out", "o.txt"])
    assert args.n == 100 and args.seed == 1 and args.exclude == [] and args.device == "cpu"
    args = build_parser().parse_args(["evaluate", "--model", "m.pt", "--lines", "sealed.txt"])
    assert args.cmd == "evaluate" and args.lines == Path("sealed.txt")
    with pytest.raises(SystemExit):
        build_parser().parse_args(["evaluate", "--model", "m.pt"])
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run", "--model", "m.pt"])
    with pytest.raises(SystemExit):
        build_parser().parse_args(["train", "--model", "m.pt"])


def test_cli_end_to_end_with_a_fake_student(gates, tmp_path, monkeypatch, capsys):
    """The CLI only needs ``student.load`` and ``student.generate``; a stand-in replaces torch."""
    def generate(model, vocab, prompt, *, device="cpu", **_):
        inner = prompt[2:-3]  # "q <prompt>. a"
        if is_judge_prompt(inner):
            return check_judge(inner, "", gates).expected
        return next((g.expected(inner) for g in gates if g.owns(inner)), "")

    fake_student = SimpleNamespace(load=lambda path, device: ("model", "vocab"), generate=generate)
    monkeypatch.setitem(sys.modules, "haishool.student", fake_student)
    monkeypatch.setattr(loop, "load_gates", lambda topics=loop.TOPICS: (gates, {"maths": "ModuleNotFoundError: x"}))
    exclude = tmp_path / "train.txt"
    exclude.write_text("".join(ln.text + "\n" for ln in gates[0].generate(random.Random(4), 10)), encoding="utf-8")
    out = tmp_path / "loop.txt"
    loop.main(["run", "--model", "m.pt", "--out", str(out), "--n", "6", "--seed", "4", "--exclude", str(exclude)])
    report = json.loads(capsys.readouterr().out)
    assert report["missing_gates"] == {"maths": "ModuleNotFoundError: x"}
    assert report["overall"]["all"] == {"n": 12, "exact": 1.0, "gate_ok": 1.0}
    assert report["questions"]["fake"]["excluded"] >= 1 and report["lines"]["total"] == 24
    assert out.read_text(encoding="utf-8").count("\n") == 24
    assert not prompts_in(exclude) & prompts_in(out)
    loop.main(["evaluate", "--model", "m.pt", "--lines", str(out)])
    report = json.loads(capsys.readouterr().out)
    assert report["lines"] == 24 and report["overall"]["all"]["exact"] == 1.0 and report["topics"]["judge"]["judge"]["n"] == 12
    assert report["gold_rejected"] == 0 and report["unowned"] == 0
    # a line file that is not there is one clear message, before any model is loaded
    with pytest.raises(SystemExit, match="no such line file"):
        loop.main(["evaluate", "--model", "m.pt", "--lines", str(tmp_path / "nowhere.txt")])
    with pytest.raises(SystemExit, match="no such line file"):
        loop.main(["run", "--model", "m.pt", "--out", str(out), "--exclude", str(exclude), str(tmp_path / "nowhere.txt")])
    # a model that cannot be loaded (no torch, no checkpoint) is one clear message, not a traceback
    def no_load(path, device):
        raise AttributeError("'NoneType' object has no attribute 'load'")

    monkeypatch.setitem(sys.modules, "haishool.student", SimpleNamespace(load=no_load, generate=generate))
    with pytest.raises(SystemExit, match="cannot load m.pt: AttributeError.*torch"):
        loop.main(["evaluate", "--model", "m.pt", "--lines", str(out)])


def test_cli_gives_the_model_room_for_a_long_answer(tmp_path, monkeypatch, capsys):
    chain, seen = ChainGate(), []

    def generate(model, vocab, prompt, *, stop=".", max_new=40, device="cpu"):  # the student's own signature
        seen.append(max_new)
        return " ".join(chain.expected(prompt[2:-3]).split()[:max_new])  # like the model: no further than max_new

    fake_student = SimpleNamespace(load=lambda path, device: ("model", "vocab"), generate=generate)
    monkeypatch.setitem(sys.modules, "haishool.student", fake_student)
    monkeypatch.setattr(loop, "load_gates", lambda topics=loop.TOPICS: ([chain], {}))
    out = tmp_path / "loop.txt"
    loop.main(["run", "--model", "m.pt", "--out", str(out), "--n", "8", "--seed", "2"])
    report = json.loads(capsys.readouterr().out)
    assert set(seen) == {MAX_TOKENS} and report["overall"]["all"] == {"n": 8, "exact": 1.0, "gate_ok": 1.0}
    assert report["lines"]["confirmed"] == 8 and report["lines"]["judge_right"] == 8 and report["lines"]["skipped"] == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    assert max(map(n_tokens, lines)) <= MAX_TOKENS and min(len(split_line(t)[1].split()) for t in lines[::2]) > 40
    check_lines(lines, [chain])


def test_cli_without_torch_says_so(tmp_path):
    """On a machine without torch the real student module loads, and the loop stops with one message."""
    student = pytest.importorskip("haishool.student")
    if student.torch is not None:
        pytest.skip("torch is installed here")
    lines = tmp_path / "sealed.txt"
    lines.write_text("q calc 1 2 plus 7. a 1 9.\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="cannot load .*needs torch"):
        loop.main(["evaluate", "--model", str(tmp_path / "student.pt"), "--lines", str(lines)])
