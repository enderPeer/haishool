"""Independent integration checks for curriculum identities, gates and shipped splits."""

import random
from collections import defaultdict
from pathlib import Path

import pytest

from haishool import curriculum
from haishool.cosmos import predict, world
from haishool.student import _held_key
from haishool.truth import Line, split_line
from haishool.truth import forces, loop, maths, substances
from haishool.truth.splits import canonical_prompt


ROOT = Path(__file__).resolve().parents[1]


GROUPS = [
    ("calc 1 2 plus 7", "calc 1 2 plus 7 steps", "check 1 2 plus 7 equals 1 9",
     "judge calc 1 2 plus 7 steps answer incorrect", "judge check 1 2 plus 7 equals 1 9 answer yes"),
    ("solve 3 x plus 4 equals 1 9", "check 3 x plus 4 equals 1 9 x 5",
     "judge check 3 x plus 4 equals 1 9 x 5 answer yes"),
    ("weight m 5 on moon", "weight m 5 on moon steps", "check weight m 5 on moon equals 8 point 1",
     "judge weight m 5 on moon steps answer incorrect"),
    ("molar_mass water", "molar_mass water steps", "check molar_mass water 1 8 point 0 1 5",
     "judge molar_mass water steps answer incorrect"),
    ("mass_percent h 2 o 1 oxygen", "mass_percent h 2 o 1 oxygen steps",
     "check mass_percent h 2 o 1 oxygen 8 8 point 8 1", "judge mass_percent h 2 o 1 oxygen answer 0"),
    ("life predict expected_mutations length 8 mu 0 point 2",
     "life predict expected_mutations length 8 point 0 mu 2 e minus 1",
     "judge life predict expected_mutations length 8 point 0 mu 2 e minus 1 answer 1 point 6"),
    ("world handoff temperature star_mass 1 orbit 1",
     "world predict handoff temperature star_mass 1 orbit 1",
     "judge world handoff temperature star_mass 1 orbit 1 answer 3 1 1",
     "judge world predict handoff temperature star_mass 1 orbit 1 answer 3 1 1"),
]


@pytest.mark.parametrize("variants", GROUPS)
def test_all_split_entrypoints_share_wrapped_question_identity(variants):
    expected = canonical_prompt(variants[0])
    for prompt in variants:
        assert canonical_prompt(prompt) == expected
        assert curriculum.key(prompt) == expected
        assert _held_key(f"q {prompt}. a 0.") == "q " + expected
    for seed in (1, 7, 20261001):
        assert len({curriculum.partition(prompt, seed) for prompt in variants}) == 1


def test_writer_keeps_worked_check_and_judge_variants_together(tmp_path):
    writer = curriculum.Writer(tmp_path, "integration", seed=20261001, eval_limit=10000)
    gate = forces.gate()
    judge = loop.JudgeGate([gate])
    for worked in gate.generate_worked(random.Random(715), 200):
        plain = worked.meta["base_prompt"]
        gold = gate.check(plain, "").expected
        rows = [worked, Line(plain, gold, "forces", "calc"),
                Line(f"check {plain} equals {gold}", "yes", "forces", "yesno")]
        for row in rows:
            writer.add(row, gate)
            writer.add(Line(loop.judge_prompt(row.prompt, row.answer), "right", "judge", "yesno"), judge)
    writer.close()
    groups = defaultdict(set)
    seen_splits = set()
    for split in ("train", "dev", "sealed"):
        for text in (tmp_path / f"integration-{split}.txt").read_text().splitlines():
            prompt, _ = split_line(text)
            groups[canonical_prompt(prompt)].add(split)
            seen_splits.add(split)
    assert seen_splits == {"train", "dev", "sealed"}
    assert len(groups) == 200
    assert all(len(splits) == 1 for splits in groups.values())


def test_every_operator_routes_through_judge_and_rejects_nonfinite_answers():
    gates = [forces.gate(), substances.gate(), maths.gate()]
    gates += [predict.gate(topic) for topic in predict.TOPICS]
    judge = loop.JudgeGate(gates)
    categories = defaultdict(set)
    for gate in gates:
        rows = gate.generate(random.Random(9801), 150)
        if hasattr(gate, "generate_worked"):
            rows += gate.generate_worked(random.Random(9802), 150)
        for row in rows:
            assert loop.gate_verdict(gate, row.prompt, row.answer).ok
            assert judge.check(loop.judge_prompt(row.prompt, row.answer), "right").ok
            for invalid in ("nan", "inf", "minus inf", "1 e 9 9 9"):
                assert not loop.gate_verdict(gate, row.prompt, invalid).ok
                assert judge.check(loop.judge_prompt(row.prompt, invalid), "wrong").ok
            words = row.prompt.split()
            if gate.topic.startswith("predict_"):
                categories[gate.topic].add(words[3] if words[2] == "handoff" else words[2])
            elif gate.topic == "forces" and words[-1] == "steps":
                categories[gate.topic].add(words[0])
            elif gate.topic == "substances" and words[-1] == "steps":
                categories[gate.topic].add(words[0])
            elif gate.topic == "maths":
                categories[gate.topic].add("steps" if words[-1] == "steps" else words[0])
    assert categories["forces"] == set(forces.CALC)
    assert categories["substances"] == {"molar_mass", "mass_percent"}
    assert categories["maths"] == {"calc", "steps", "solve", "compare", "check"}
    for topic in predict.TOPICS:
        expected = set(world.HANDOFFS) if topic == "world" else {
            name for name, rule in predict.RULES.items() if rule.topic == topic}
        assert categories["predict_" + topic] == expected


def test_predictor_rounding_does_not_accept_several_wrong_final_digits():
    row = predict._line("expected_mutations", [10, .9999])
    assert row.answer == "9 point 9 9 9"
    assert predict.check(row.prompt, row.answer).ok
    assert not predict.check(row.prompt, "9 point 9 9 5").ok


def test_world_noncanonical_numbers_are_rejected_in_both_prompt_forms():
    for head in ("world handoff", "world predict handoff"):
        prompt = f"{head} temperature star_mass 1 point 0 orbit 1"
        gate = predict.gate("world") if "predict" in head else world
        assert not gate.owns(prompt)
        verdict = gate.check(prompt, "3 1 1")
        assert not verdict.ok and verdict.expected is None


def test_worked_chemical_rounding_uses_exact_total_not_rounded_subtotals():
    gate = substances.gate()
    prompt = "mass_percent c 9 h 7 n 1 o 2 oxygen"
    answer = gate.check(prompt + " steps", "").expected
    assert answer is not None
    # C9H7NO2: 108.099 + 7.056 + 14.007 + 31.998 = 161.160 exactly.
    assert "is 1 6 1 point 1 6" in answer
    assert "substitute 1 0 0 times 3 1 point 9 9 8 over 1 6 1 point 1 6" in answer
    assert answer.endswith("result 1 9 point 8 5")
    # Preserve the result while corrupting an intermediate: the judge must detect it.
    bad = answer.replace("is 1 0 8 point 0 9 9", "is 1 0 8 point 1")
    assert bad != answer
    assert not gate.check(prompt + " steps", bad).ok
    assert loop.JudgeGate([gate]).check(loop.judge_prompt(prompt + " steps", bad), "wrong").ok


@pytest.mark.parametrize("directory", ["truth-v5", "cosmos-v6"])
def test_shipped_training_and_sealed_questions_have_no_semantic_overlap(directory):
    root = ROOT / "data" / directory
    keys = {}
    for split in ("train", "sealed"):
        files = sorted(root.glob(f"*-{split}.txt"))
        assert files, f"missing shipped {directory} {split} data"
        keys[split] = {canonical_prompt(parts[0]) for path in files
                       for text in path.read_text(encoding="utf-8").splitlines()
                       if (parts := split_line(text))}
    assert not keys["train"] & keys["sealed"]
