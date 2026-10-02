import json
import random
import re
from collections import Counter
from pathlib import Path

import pytest

from haishool import round5
from haishool.cosmos import close, world
from haishool.round5 import (COSMOS_SHARE, MAX_TOKENS, STEP_QUESTIONS, TRUTH_SHARE, Judge, asked, build_cosmos,
                             build_truth, cosmos_seeds, cost, earlier_answers, fit, judge, judges, levels, rng_for,
                             score, screen, stats, thin)
from haishool.student import tokens
from haishool.truth import Line, Verdict, is_dense, is_finite, loop, parse_num, split_line

SEED = 7
FAST = ("nucleo", "planets", "chem", "life")
QUIET = lambda text: None  # noqa: E731
BIG = " ".join("7" * 400)


def read(path: Path) -> list[str]:
    return [ln for ln in path.read_text(encoding="utf-8").split("\n") if ln]


def questions(path: Path) -> list[tuple[str, str]]:
    return [parts for parts in map(split_line, read(path)) if parts]


def wrong(answer: str) -> str:
    """An answer no gate may accept: yes for no, a leading digit moved by five, or another word."""
    if answer in ("yes", "no"):
        return "no" if answer == "yes" else "yes"
    words = answer.split()
    if parse_num(answer) is not None:
        i = next(i for i, w in enumerate(words) if w.isdigit())
        words[i] = str((int(words[i]) + 5) % 10)
        return " ".join(words)
    return "zzz"


@pytest.fixture(scope="module")
def everyone():
    return judges()


@pytest.fixture(scope="module")
def truth(tmp_path_factory):
    out = tmp_path_factory.mktemp("truth")
    return out, build_truth(out, SEED, log=QUIET)


@pytest.fixture(scope="module")
def cosmos(tmp_path_factory):
    out = tmp_path_factory.mktemp("cosmos")
    return out, build_cosmos(out, SEED, budget=50_000, sims=FAST, log=QUIET)


# ---------------------------------------------------------------------------------------------
# the pieces


def test_shares_cover_every_topic_and_simulation():
    assert tuple(TRUTH_SHARE) == loop.TOPICS
    assert list(COSMOS_SHARE) == list(levels())
    assert sum(TRUTH_SHARE.values()) == pytest.approx(1) and sum(COSMOS_SHARE.values()) == pytest.approx(1)
    assert [j.topic for j in judges()] == [*TRUTH_SHARE, *COSMOS_SHARE]


def test_rng_for_is_pinned_and_separate():
    assert rng_for(20261001, "maths", "train").random() == 0.5772400175141115
    assert rng_for(1, "a").randint(0, 10 ** 9) == 232633270
    assert rng_for(1, "a").random() == rng_for(1, "a").random() != rng_for(1, "b").random()
    assert rng_for(1, "a", 2).random() != rng_for(2, "a", 1).random()


def test_cost_counts_the_eos():
    text = "q carbon protons. a 6."
    assert cost([text]) == len(tokens(text)) + 1 == 8
    assert cost([Line("carbon protons", "6"), text]) == 16 and cost([]) == 0


def test_fit_finds_the_budget():
    def flat(n):
        return [Line(f"calc {' '.join(str(i))}", "1") for i in range(n)]

    n, lines = fit(flat, 10_000, 5)
    assert len(lines) == n and abs(cost(lines) - 10_000) <= 100

    def capped(n):  # a gate with 50 questions and no more
        return flat(min(n, 50))

    n, lines = fit(capped, 10_000, 5)
    assert len(lines) == 50


def test_thin_keeps_records_and_draws_step_questions():
    lines = [Line("toy seed 3 params. a 1", kind="record")]
    for t in range(30):
        lines.append(Line(f"toy seed 3 step {' '.join(str(t))}. x 1", kind="record"))
        lines += [Line(f"toy seed 3 step {' '.join(str(t))} {k}", "1") for k in ("x", "y", "next x", "next y")]
    lines += [Line("toy seed 3 final x", "1"), Line("toy seed 3 param a", "1"), Line("toy handoff a 1", "2")]
    kept, held = thin(lines, random.Random(1), 10)
    assert [ln for ln in kept if ln.kind == "record"] == [ln for ln in lines if ln.kind == "record"]
    assert all(ln in kept for ln in lines[-3:])
    step = [ln for ln in kept if ln.kind != "record" and " step " in ln.prompt]
    assert len(step) == 10 and len(held) == 10 and not set(step) & set(held)
    assert all(" step " in ln.prompt and ln.kind != "record" for ln in held)
    assert kept == [ln for ln in lines if ln in kept], "the order of the rollout is kept"
    assert thin(lines, random.Random(1), 10) == (kept, held)
    assert thin(lines, random.Random(2), 10) != (kept, held)
    everything, none = thin(lines, random.Random(1), 1000)
    assert everything == lines and none == []


def test_asked_names_the_question_a_check_restates(everyone):
    gate = {j.topic: j.gate for j in everyone}
    cases = [("maths", "check 1 2 plus 7 equals 1 9", "calc 1 2 plus 7"),
             ("maths", "check 3 x equals 1 2 x 4", "solve 3 x equals 1 2"),
             ("maths", "calc 1 2 plus 7", "calc 1 2 plus 7"),
             ("elements", "check iron protons 2 6", "iron protons"),
             ("elements", "check iron heavier_than carbon", "check iron heavier_than carbon"),
             ("substances", "check water formula h 2 o 2", "water formula"),
             ("substances", "check mass_percent water oxygen 8 8 point 8 1", "mass_percent water oxygen"),
             ("reactions", "check 2 x h 2 plus 1 x o 2 gives 2 x h 2 o 1 balanced",
              "check 2 x h 2 plus 1 x o 2 gives 2 x h 2 o 1 balanced")]
    for topic, prompt, inner in cases:
        assert asked(gate[topic], Line(prompt, "yes", topic, "yesno")) == inner, prompt
    forces = [ln for ln in gate["forces"].generate(random.Random(3), 400) if ln.kind == "yesno"]
    assert forces and all(asked(gate["forces"], ln) == ln.meta["inner"] != ln.prompt for ln in forces)
    record = Line("check this. a 1", kind="record")
    assert asked(gate["maths"], record) == record.prompt


class Fake:
    """A gate that owns ``toy <n>`` and answers ``n + 1``; ``other`` says ``n + 2`` to ``toy 5``."""

    def __init__(self, topic: str, add: int = 1, only: str | None = None) -> None:
        self.topic, self.add, self.only = topic, add, only

    def owns(self, prompt: str) -> bool:
        words = prompt.split()
        return len(words) == 2 and words[0] == "toy" and words[1].isdigit() and self.only in (None, prompt)

    def check(self, prompt: str, answer: str) -> Verdict:
        if not self.owns(prompt):
            return Verdict(False, None, "not my question")
        want = str(int(prompt.split()[1]) + self.add)
        return Verdict(answer == want, want)


def test_screen_drops_and_counts():
    own, other = Judge("toy", Fake("toy")), Judge("other", Fake("other", 2, "toy 5"))
    lines = [Line("toy 1", "2"), Line("toy 1", "2"), Line("toy 2", "9"), Line("toy 5", "6"), Line("toy 7", "8"),
             Line("toy 8", "9"), Line("toy record. a 1", kind="record"),
             Line("toy long. " + " ".join(["a 1."] * 40), kind="record")]
    seen: set[str] = set()
    kept, drops = screen(lines, own, [other], {"toy 7": {"0"}, "toy 8": {"9"}}, seen)
    assert [ln.text for ln in kept] == ["q toy 1. a 2.", "q toy 8. a 9.", "toy record. a 1."]
    assert drops == {"repeated": 1, "rejected": 1, "conflict": 1, "collision": 1, "long": 1}
    assert seen == {"toy 1", "toy 8"}
    again, drops = screen(lines[:1], own, [other], {}, seen)
    assert again == [] and drops == {"repeated": 1}


def test_earlier_answers_reads_records_and_lines(tmp_path):
    rec = tmp_path / "records.jsonl"
    rec.write_text(json.dumps({"obj": "spoon", "values": {"kind": "tool", "color": "silver"}}) + "\n", encoding="utf-8")
    txt = tmp_path / "hops.txt"
    txt.write_text("q gold found_in. a bank.\nq gold found_in. a mine.\ngold. kind metal.\n", encoding="utf-8")
    got = earlier_answers([rec, txt, tmp_path / "missing.txt"])
    assert got["spoon color"] == {"silver"} and got["gold found_in"] == {"bank", "mine"}
    assert "gold" not in got and len(got) == 3


def test_cosmos_seeds_are_two_separate_lists():
    train, sealed = cosmos_seeds(20261001)
    assert train[:5] == [5440, 6892, 8574, 6256, 95] and sealed[:5] == [3398, 8807, 2574, 8043, 8890]
    assert len(set(train)) == len(train) and len(set(sealed)) == len(sealed) and not set(train) & set(sealed)
    assert all(1 <= s <= 9998 for s in train + sealed)
    assert cosmos_seeds(20261001) == (train, sealed) and cosmos_seeds(1)[0] != train


def test_a_number_too_large_is_no_number(everyone):
    assert is_finite(3) and is_finite(0.5) and not is_finite(None)
    assert not is_finite(float("inf")) and not is_finite(float("nan")) and not is_finite(10 ** 400)
    assert close(100, 104) and not close(100, 106)
    assert not close(float("inf"), 5.0) and not close(5.0, float("inf")) and not close(10 ** 400, 5.0)
    assert not close(float("inf"), float("inf")) and not close(float("nan"), 1.0)


# ---------------------------------------------------------------------------------------------
# round 5


def test_truth_files_and_report(truth):
    out, report = truth
    assert set(report["topics"]) == set(TRUTH_SHARE)
    assert json.loads((out / "report.json").read_text(encoding="utf-8")) == report
    for topic, rep in report["topics"].items():
        train, sealed = read(out / f"{topic}-train.txt"), read(out / f"{topic}-sealed.txt")
        assert len(train) == rep["lines"] == sum(rep["by_kind"].values()) and cost(train) == rep["tokens"]
        assert len(sealed) == rep["sealed"]["lines"] > 100 and cost(sealed) == rep["sealed"]["tokens"]
        records = [ln for ln in train if split_line(ln) is None]
        assert len(records) == rep["records"] == rep["by_kind"].get("record", 0)
        assert train[:len(records)] == records, "records first"
        assert all(split_line(ln) for ln in sealed), "no records in the sealed set"
        assert rep["n"] == len(train) - len(records) > 500
        assert abs(rep["tokens"] - rep["budget"]) <= 0.02 * rep["budget"]
        assert rep["dropped"] == {**dict.fromkeys(round5.DROPS, 0), "seen": rep["sealed"]["drawn"] - len(sealed)}
        assert set(rep["by_kind"]) <= {"record", "fact", "calc", "yesno"} and rep["seconds"] >= 0
    total = report["total"]
    assert total["lines"] == sum(r["lines"] for r in report["topics"].values())
    assert total["sealed_lines"] == sum(r["sealed"]["lines"] for r in report["topics"].values())
    assert abs(total["tokens"] - round5.TRUTH_TOKENS) <= 0.01 * round5.TRUTH_TOKENS


def test_truth_lines_are_dense_short_and_owned(truth, everyone):
    out, _ = truth
    for topic in TRUTH_SHARE:
        for name in (f"{topic}-train.txt", f"{topic}-sealed.txt"):
            lines = read(out / name)
            assert all(is_dense(ln) for ln in lines), name
            assert max(len(tokens(ln)) for ln in lines) <= MAX_TOKENS, name
            asked_here = questions(out / name)
            assert len({p for p, _ in asked_here}) == len(asked_here), f"a prompt twice in {name}"
            for prompt, answer in asked_here:
                verdict = judge(prompt, answer, everyone)
                assert verdict["ok"] and topic in verdict["owners"] and verdict["agree"], (name, prompt, verdict)


def test_gates_reject_a_changed_answer(truth, everyone):
    out, _ = truth
    rng = random.Random(5)
    for topic in TRUTH_SHARE:
        lines = questions(out / f"{topic}-train.txt") + questions(out / f"{topic}-sealed.txt")
        for prompt, answer in rng.sample(lines, 400):
            verdict = judge(prompt, wrong(answer), everyone)
            assert not verdict["ok"] and verdict["expected"] is not None, (prompt, answer, verdict)
            for huge in (BIG, "1 e 9 9 9 9", "minus 1 e 9 9 9 9", ""):
                assert not judge(prompt, huge, everyone)["ok"], (prompt, huge)


def test_sealed_set_is_not_given_away(truth, everyone):
    out, _ = truth
    gate = {j.topic: j.gate for j in everyone}
    everything = Counter(p for topic in TRUTH_SHARE for p, _ in questions(out / f"{topic}-train.txt"))
    assert sum(n > 1 for n in everything.values()) < 30, "only the element states are asked by two topics"
    sealed_twice = Counter(p for topic in TRUTH_SHARE for p, _ in questions(out / f"{topic}-sealed.txt"))
    assert max(sealed_twice.values()) == 1
    for topic in TRUTH_SHARE:
        train, sealed = questions(out / f"{topic}-train.txt"), questions(out / f"{topic}-sealed.txt")
        assert not set(everything) & {p for p, _ in sealed}, "a sealed prompt in some topic's training file"
        if topic == "forces":  # its gate names the restated question in the line's meta, which a file does not keep
            continue
        given = {asked(gate[topic], Line(p, a, topic, "yesno" if a in ("yes", "no") else "fact")) for p, a in train}
        for p, a in sealed:
            assert asked(gate[topic], Line(p, a, topic, "yesno" if a in ("yes", "no") else "fact")) not in given, p
    forces = gate["forces"]
    drawn = forces.generate(rng_for(SEED, "forces", "sealed"), round5.SEALED_QUESTIONS)
    trained = forces.generate(rng_for(SEED, "forces", "train"), truth[1]["topics"]["forces"]["n"])
    given = {asked(forces, ln) for ln in trained}
    kept = {p for p, _ in questions(out / "forces-sealed.txt")}
    assert kept == {ln.prompt for ln in drawn if asked(forces, ln) not in given}  # no other topic asks these
    assert any(ln.prompt not in kept for ln in drawn)


def test_truth_prompts_file_feeds_the_loop(truth):
    out, _ = truth
    listed = read(out / "prompts.txt")
    assert listed == sorted(set(listed))
    want = set()
    for topic in TRUTH_SHARE:
        want |= {p for p, _ in questions(out / f"{topic}-train.txt") + questions(out / f"{topic}-sealed.txt")}
    assert set(listed) == want == loop.prompts_in(out / "prompts.txt")
    # the loop asks nothing that is listed
    gates = loop.all_gates()
    report = loop.run(lambda prompt: "", gates, random.Random(1), 40, set(listed), None)
    assert all(q["asked"] == 40 for q in report["questions"].values())
    assert sum(q["excluded"] for q in report["questions"].values()) > 0


def test_truth_build_is_deterministic(truth, tmp_path):
    out, report = truth
    again = build_truth(tmp_path / "a", SEED, log=QUIET)
    names = sorted(p.name for p in out.glob("*.txt"))
    assert names == sorted(p.name for p in (tmp_path / "a").glob("*.txt")) and len(names) == 11
    for name in names:
        assert (out / name).read_bytes() == (tmp_path / "a" / name).read_bytes(), name
    strip = lambda rep: {t: {k: v for k, v in r.items() if k != "seconds"} for t, r in rep["topics"].items()}  # noqa: E731
    assert strip(again) == strip(report)
    other = build_truth(tmp_path / "b", SEED + 1, budget=40_000, sealed=100, topics=("maths", "forces"), log=QUIET)
    assert set(other["topics"]) == {"maths", "forces"} and other["topics"]["maths"]["budget"] == 12_000
    assert (tmp_path / "b" / "maths-train.txt").read_bytes()[:2000] != (out / "maths-train.txt").read_bytes()[:2000]
    assert b"\r" not in (out / "maths-train.txt").read_bytes()


def test_no_prompt_of_rounds_1_to_4_gets_a_second_answer(truth, cosmos):
    earlier = earlier_answers()
    if not earlier:
        pytest.skip("the round 1-4 data files are not here")
    assert truth[1]["earlier_prompts"] == len(earlier) > 50_000
    for out in (truth[0], cosmos[0]):
        for path in out.glob("*-train.txt"):
            for prompt, answer in questions(path):
                assert earlier.get(prompt, {answer}) == {answer}, (path.name, prompt)


# ---------------------------------------------------------------------------------------------
# round 6


def seed_of(prompt: str) -> int | None:
    m = re.match(r"^[a-z]+ seed ((?:\d )*\d) ", prompt)
    return int(m.group(1).replace(" ", "")) if m else None


def test_cosmos_files_and_report(cosmos):
    out, report = cosmos
    assert tuple(report["sims"]) == FAST
    assert json.loads((out / "report.json").read_text(encoding="utf-8")) == report
    train_seeds, sealed_seeds = cosmos_seeds(SEED)
    for name, rep in report["sims"].items():
        train, sealed, held = (read(out / f"{name}-{part}.txt") for part in ("train", "sealed", "heldout"))
        assert len(train) == rep["lines"] and cost(train) == rep["tokens"]
        assert len(sealed) == rep["sealed"]["lines"] and len(held) == rep["heldout"]["lines"]
        assert all(split_line(ln) for ln in sealed + held), "questions only"
        n = rep["rollouts"]
        assert n >= 1 and rep["seeds"] == train_seeds[:n]
        assert rep["sealed"]["seeds"] == sealed_seeds[:rep["sealed"]["rollouts"]] and rep["sealed"]["rollouts"] == 3
        assert rep["tokens"] <= 1.6 * rep["budget"]
        assert {seed_of(ln[2:]) for ln in sealed} == set(rep["sealed"]["seeds"])
        assert {seed_of(ln[2:]) for ln in held} == set(rep["seeds"])
        assert {seed_of(ln[2:] if ln.startswith("q ") else ln) for ln in train} - {None} == set(rep["seeds"])
        assert len(held) == n * STEP_QUESTIONS
        step = [ln for ln in train if split_line(ln) and " step " in split_line(ln)[0]]
        assert len(step) == n * STEP_QUESTIONS
        prompts = [split_line(ln)[0] for ln in train + sealed + held if split_line(ln)]
        assert len(prompts) == len(set(prompts)), "a prompt in two files or twice in one"
        assert rep["dropped"] == {**dict.fromkeys(round5.DROPS, 0), "gate_failed": 0, "seen": 0}
    assert report["total"]["rollouts"] == sum(r["rollouts"] for r in report["sims"].values())
    assert sorted(set(read(out / "prompts.txt"))) == read(out / "prompts.txt")
    assert not (out / "gravity-train.txt").exists()


def test_cosmos_lines_are_dense_short_and_judged(cosmos, everyone):
    out, _ = cosmos
    rng = random.Random(9)
    for name in FAST:
        every = []
        for part in ("train", "sealed", "heldout"):
            lines = read(out / f"{name}-{part}.txt")
            assert all(is_dense(ln) for ln in lines), (name, part)
            assert max(len(tokens(ln)) for ln in lines) <= MAX_TOKENS
            every += questions(out / f"{name}-{part}.txt")
        for prompt, answer in every:
            verdict = judge(prompt, answer, everyone)
            assert verdict["ok"] and verdict["owners"] == [name], (prompt, verdict)
        for prompt, answer in rng.sample(every, min(len(every), 300)):
            verdict = judge(prompt, wrong(answer), everyone)
            assert not verdict["ok"] and verdict["expected"] is not None, (prompt, answer, verdict)
            for huge in (BIG, "1 e 9 9 9 9", ""):
                assert not judge(prompt, huge, everyone)["ok"], (prompt, huge)


def test_cosmos_tables_are_trained_once(cosmos):
    out, _ = cosmos
    train = read(out / "chem-train.txt")
    sim = levels()["chem"]
    tables = [ln.text for ln in sim.tables()]
    assert train[:len(tables)] == tables and len(tables) > 200
    assert not any(ln.startswith("q chem valence") for ln in read(out / "chem-sealed.txt"))


def test_rollout_rows_are_the_truth_of_the_lines(cosmos):
    out, report = cosmos
    for name, lv in levels().items():
        if name not in FAST:
            continue
        rows = [json.loads(ln) for ln in read(out / f"{name}-rollouts.jsonl")]
        rep = report["sims"][name]
        assert [(r["seed"], r["split"]) for r in rows] == [(s, "train") for s in rep["seeds"]] + \
            [(s, "sealed") for s in rep["sealed"]["seeds"]]
        for row in rows:
            assert set(row) == {"sim", "seed", "split", "params", "steps", "summary"} and row["sim"] == name
            r = lv.rollout(row["seed"])
            plain = json.loads(json.dumps({"params": r.params, "steps": r.steps, "summary": r.summary}))
            assert {k: row[k] for k in plain} == plain
            assert lv.sim.conserved(r).ok


def test_cosmos_build_is_deterministic_and_per_simulation(cosmos, tmp_path):
    out, report = cosmos
    build_cosmos(tmp_path / "a", SEED, budget=50_000, sims=("planets", "nucleo"), log=QUIET)
    for name in ("nucleo", "planets"):
        for part in ("train.txt", "sealed.txt", "heldout.txt", "rollouts.jsonl"):
            assert (tmp_path / "a" / f"{name}-{part}").read_bytes() == (out / f"{name}-{part}").read_bytes(), (name, part)
    other = build_cosmos(tmp_path / "b", SEED + 1, budget=50_000, sims=("nucleo",), sealed_rollouts=1, log=QUIET)
    assert other["sims"]["nucleo"]["seeds"] != report["sims"]["nucleo"]["seeds"]
    assert other["sims"]["nucleo"]["sealed"]["rollouts"] == 1


def test_gravity_and_world_through_the_build(tmp_path, everyone):
    report = build_cosmos(tmp_path, SEED, budget=12_000, sims=("gravity", "world"), sealed_rollouts=1, log=QUIET)
    gravity_rep, world_rep = report["sims"]["gravity"], report["sims"]["world"]
    assert gravity_rep["rollouts"] == 1 and world_rep["rollouts"] >= 1
    assert gravity_rep["dropped"]["gate_failed"] == world_rep["dropped"]["gate_failed"] == 0
    assert world_rep["seeds"][0] == gravity_rep["seeds"][0] == cosmos_seeds(SEED)[0][0]
    train = read(tmp_path / "world-train.txt")
    tables = [ln.text for ln in world.records() + world.table_lines()]
    assert train[:len(tables)] == tables
    assert sum(ln.startswith("q world handoff ") for ln in train) >= 7 + round5.PRACTICE - world_rep["dropped"]["repeated"]
    assert read(tmp_path / "world-heldout.txt") == [] and world_rep["heldout"]["lines"] == 0
    assert len(read(tmp_path / "gravity-heldout.txt")) == STEP_QUESTIONS
    for name in ("gravity", "world"):
        for part in ("train", "sealed"):
            lines = read(tmp_path / f"{name}-{part}.txt")
            assert all(is_dense(ln) and len(tokens(ln)) <= MAX_TOKENS for ln in lines)
            for prompt, answer in questions(tmp_path / f"{name}-{part}.txt"):
                verdict = judge(prompt, answer, everyone)
                assert verdict["ok"] and verdict["owners"] == [name], (prompt, verdict)
                assert not judge(prompt, wrong(answer), everyone)["ok"], (prompt, answer)
                assert not judge(prompt, "1 e 9 9 9 9", everyone)["ok"] and not judge(prompt, BIG, everyone)["ok"]
    asked_twice = [p for p, n in Counter(p for part in ("train", "sealed")
                                         for p, _ in questions(tmp_path / f"world-{part}.txt")).items() if n > 1]
    assert asked_twice == []
    rows = [json.loads(ln) for ln in read(tmp_path / "world-rollouts.jsonl")]
    assert Counter(r["split"] for r in rows) == {"train": world_rep["rollouts"], "sealed": 1}
    assert all(r["summary"]["outcome"] in world.OUTCOMES for r in rows)


# ---------------------------------------------------------------------------------------------
# judge, evaluate, stats, command line


def test_judge_names_the_owner(everyone):
    cases = [("calc 1 2 plus 7", "1 9", "maths", True), ("calc 1 2 plus 7", "2 0", "maths", False),
             ("iron protons", "2 6", "elements", True), ("water molar_mass", "1 8 point 0 1 5", "substances", True),
             ("balance h 2 plus o 2 gives h 2 o 1", "2 1 2", "reactions", True),
             ("weight m 5", "4 9 point 0 5", "forces", True),
             ("nucleo seed 7 final helium", "0 point 2 0 3", "nucleo", True),
             ("planets seed 3 final planets", "7", "planets", True), ("chem valence c", "4", "chem", True),
             ("life seed 7 param gap", "0", "life", True),
             ("world handoff fusion star_mass 0 point 8 8 5", "yes", "world", True),
             ("world constant cloud_mass", "1 point 5", "world", True)]
    for prompt, answer, owner, ok in cases:
        got = judge(prompt, answer, everyone)
        assert (got["owner"], got["ok"]) == (owner, ok), got
        assert got["expected"] is not None and got["prompt"] == prompt and got["answer"] == answer
    for prompt in ("turkey capital", "q spoon color", "gold found_in", "gravity", "judge this answer"):
        got = judge(prompt, "x", everyone)
        assert got == {"prompt": prompt[2:] if prompt.startswith("q ") else prompt, "answer": "x", "owner": None,
                       "owners": [], "ok": False, "expected": None, "reason": "no gate", "agree": True}
        assert not any(j.owns(prompt) for j in everyone)


def test_judge_reads_whole_lines_and_judge_lines(everyone):
    assert judge("q calc 1 2 plus 7. a 1 9.", None, everyone)["ok"]
    line = judge("q calc 1 2 plus 7. a 2 0.", None, everyone)
    assert (line["prompt"], line["answer"], line["ok"], line["expected"]) == ("calc 1 2 plus 7", "2 0", False, "1 9")
    assert judge("q calc 1 2 plus 7.", "a 1 9.", everyone)["ok"]
    right = judge("judge calc 1 2 plus 7 answer 2 0", "wrong", everyone)
    assert right["owner"] == "judge" and right["ok"] and right["expected"] == "wrong"
    assert not judge("judge calc 1 2 plus 7 answer 2 0", "right", everyone)["ok"]
    assert judge("judge nucleo seed 7 final helium answer 0 point 2 0 3", "right", everyone)["ok"]
    shared = judge("iron state", "solid", everyone)
    assert shared["owners"] == ["elements", "substances"] and shared["ok"] and shared["agree"]


def test_no_judge_raises_on_any_answer(everyone):
    rng = random.Random(2)
    prompts = [ln.prompt for j in everyone[:5] for ln in j.gate.generate(random.Random(4), 300)]
    prompts += ["nucleo seed 7 final helium", "gravity seed 5 final clumps", "planets seed 3 final largest_mass",
                "chem seed 4 final h2o", "life seed 7 final replicators", "world seed 3 final metallicity",
                "nucleo seed 7 step 3 next temperature", "life seed 7 step 2 0 next replicators"]
    answers = [BIG, "minus " + BIG, BIG + " point 5", "1 e 9 9 9 9", "minus 1 e 9 9 9 9", "x " + BIG, "", "point",
               "e e e", "minus", "yes no", "²", "1 e " + BIG]
    for prompt in prompts:
        owner = next(j for j in everyone if j.owns(prompt))
        for answer in answers:
            verdict = owner.check(prompt, answer)  # the gate itself, without the loop's guard
            assert isinstance(verdict, Verdict) and not verdict.ok, (prompt, answer[:40])
        assert isinstance(owner.check(prompt, "0 0"), Verdict)  # some simulations read it as 0
    for _ in range(300):  # and no owner is confused by word soup
        soup = " ".join(rng.choice(["calc", "seed", "step", "world", "iron", "1", "x", "check", "final", "e"])
                        for _ in range(rng.randint(1, 8)))
        assert isinstance(judge(soup, "1", everyone)["ok"], bool)


def test_score_asks_every_sealed_question(truth, cosmos, everyone):
    lines = read(truth[0] / "maths-sealed.txt")[:150] + read(cosmos[0] / "nucleo-sealed.txt")[:150]
    lines += ["q judge calc 1 2 plus 7 answer 2 0. a wrong.", "iron element. number 2 6.", "q turkey capital. a ankara."]
    gold = dict(parts for parts in map(split_line, lines) if parts)
    n = len(lines) - 2
    perfect = score(gold.__getitem__, lines, everyone)
    assert n > 250 and perfect["overall"]["all"] == {"n": n, "exact": 1.0, "gate_ok": 1.0}
    assert (perfect["records"], perfect["unowned"], perfect["gold_rejected"]) == (1, 1, 0)
    assert set(perfect["topics"]) == {"maths", "nucleo", "judge"}
    silent = score(lambda prompt: "", lines, everyone)
    assert silent["overall"]["all"] == {"n": n, "exact": 0.0, "gate_ok": 0.0}


def test_stats_and_command_line(truth, cosmos, tmp_path, capsys):
    both = stats(truth[0], cosmos[0])
    assert both == {"truth": truth[1], "cosmos": cosmos[1]}
    assert stats(tmp_path, tmp_path) == {"truth": None, "cosmos": None}
    args = round5.build_parser().parse_args(["build"])
    assert (args.seed, args.out_truth.name, args.out_cosmos.name) == (20261001, "truth-v5", "cosmos-v6")
    assert (args.truth_tokens, args.cosmos_tokens, args.sealed, args.only) == (350_000, 350_000, 1000, None)
    capsys.readouterr()
    assert round5.main(["stats", "--out-truth", str(truth[0]), "--out-cosmos", str(cosmos[0])]) == both
    assert json.loads(capsys.readouterr().out) == both
    got = round5.main(["judge", "calc 1 2 plus 7", "1 9"])
    assert json.loads(capsys.readouterr().out) == got and got["owner"] == "maths" and got["ok"]
    assert round5.main(["judge", "q water molar_mass. a 1 8."])["expected"] == "1 8 point 0 1 5"
    capsys.readouterr()
    built = round5.main(["build", "--only", "truth", "--out-truth", str(tmp_path / "t"), "--seed", str(SEED),
                         "--truth-tokens", "350000"])
    shown = capsys.readouterr()
    assert json.loads(shown.out) == built and "round 5" in shown.err and set(built) == {"truth"}
    assert (tmp_path / "t" / "maths-train.txt").read_bytes() == (truth[0] / "maths-train.txt").read_bytes()
    with pytest.raises(SystemExit):
        round5.main(["evaluate", "--model", "nowhere.pt", "--lines", str(tmp_path / "missing.txt")])
    with pytest.raises(SystemExit):
        round5.main(["evaluate", "--model", "nowhere.pt", "--lines", str(truth[0] / "maths-sealed.txt")])
    with pytest.raises(SystemExit):
        round5.main([])


# ---------------------------------------------------------------------------------------------
# the files of the real build, when they are there


def shipped(folder: Path) -> dict | None:
    path = folder / "report.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def test_shipped_truth_files_are_what_the_build_gives(tmp_path):
    report = shipped(round5.OUT_TRUTH)
    if report is None:
        pytest.skip("no build in data/truth-v5 (python -m haishool.round5 build)")
    sealed = {rep["sealed"]["drawn"] for rep in report["topics"].values()}
    assert len(sealed) == 1
    again = build_truth(tmp_path, report["seed"], report["budget_tokens"], sealed.pop(), tuple(report["topics"]), log=QUIET)
    for name in [f"{t}-{part}.txt" for t in report["topics"] for part in ("train", "sealed")] + ["prompts.txt"]:
        assert (round5.OUT_TRUTH / name).read_bytes() == (tmp_path / name).read_bytes(), \
            f"{name} is stale: the gates changed since the build; run python -m haishool.round5 build"
    assert again["total"]["tokens"] == report["total"]["tokens"] and again["total"]["lines"] == report["total"]["lines"]


def test_shipped_cosmos_files_match_their_report_and_the_fast_simulations(tmp_path):
    report = shipped(round5.OUT_COSMOS)
    if report is None:
        pytest.skip("no build in data/cosmos-v6 (python -m haishool.round5 build)")
    train_seeds, sealed_seeds = cosmos_seeds(report["seed"])
    for name, rep in report["sims"].items():
        train, sealed = read(round5.OUT_COSMOS / f"{name}-train.txt"), read(round5.OUT_COSMOS / f"{name}-sealed.txt")
        assert len(train) == rep["lines"] and cost(train) == rep["tokens"] and len(sealed) == rep["sealed"]["lines"]
        assert all(is_dense(ln) and len(tokens(ln)) <= MAX_TOKENS for ln in train + sealed)
        assert rep["seeds"] == train_seeds[:rep["rollouts"]]
        assert rep["sealed"]["seeds"] == sealed_seeds[:rep["sealed"]["rollouts"]]
        assert len(read(round5.OUT_COSMOS / f"{name}-rollouts.jsonl")) == rep["rollouts"] + rep["sealed"]["rollouts"]
    fast = [name for name in ("nucleo", "planets", "life") if name in report["sims"]]
    build_cosmos(tmp_path, report["seed"], report["budget_tokens"], fast, log=QUIET)
    for name in fast:
        for part in ("train.txt", "sealed.txt", "heldout.txt", "rollouts.jsonl"):
            assert (round5.OUT_COSMOS / f"{name}-{part}").read_bytes() == (tmp_path / f"{name}-{part}").read_bytes(), \
                f"{name}-{part} is stale: run python -m haishool.round5 build"
