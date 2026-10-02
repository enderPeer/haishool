"""The round-7 dataset builder: a tiny build (scale 0.01, 8 rollout seeds), built twice."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import pytest

from haishool import curriculum7 as c7
from haishool.evo import LEVELS
from haishool.truth import Line, is_dense, split_line

SPLITS = ("train", "dev", "sealed")
TINY = dict(seed=20261001, scale=0.01, world_seeds=4, eval_seeds=4)


def ntokens(text: str) -> int:
    return len(text.replace(".", " . ").split())


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    first = tmp_path_factory.mktemp("evo7a")
    second = tmp_path_factory.mktemp("evo7b")
    # different pool sizes: the output may not depend on how the work is scheduled
    report = c7.build(first, workers=6, **TINY)
    c7.build(second, workers=3, **TINY)
    return first, second, report


def lines_of(out: Path, topic: str, split: str) -> list[str]:
    return (out / f"{topic}-{split}.txt").read_text(encoding="utf-8").splitlines()


def test_topics_and_files_exist(built):
    out, _, report = built
    lessons = [f"predict_{n}" for n in LEVELS] + [f"predict_{n}_rules7" for n in c7.OLD]
    stories = [f"legacy_{n}" for n in LEVELS] + [f"legacy_{n}_rules7" for n in c7.OLD]
    assert sorted(report["topics"]) == sorted(lessons + stories)
    for topic in report["topics"]:
        for split in SPLITS:
            assert (out / f"{topic}-{split}.txt").is_file()
        assert lines_of(out, topic, "train"), topic
    for topic in lessons:
        assert (out / f"{topic}-report.json").is_file()
        assert report["topics"][topic]["status"] == "target_reached"
        assert report["topics"][topic]["splits"]["train"]["lines"] >= 300
    for name in ("report.json", "rollouts.jsonl", "prompts.txt"):
        assert (out / name).is_file()
    rolls = [json.loads(x) for x in (out / "rollouts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rolls) == len(stories) * 8 == report["rollouts"]["total"]
    assert {r["topic"] for r in rolls} == set(stories)


def test_every_line_dense_and_short(built):
    out, _, report = built
    for topic in report["topics"]:
        for split in SPLITS:
            for text in lines_of(out, topic, split):
                assert is_dense(text), text
                assert ntokens(text) <= 256, text
                parts = split_line(text)
                if parts:
                    assert "." not in parts[0] and "." not in parts[1], text


def test_lessons_are_judged_true(built):
    out, _, report = built
    gates = c7.lesson_gates()
    for topic, gate in gates.items():
        for split in SPLITS:
            for text in lines_of(out, topic, split):
                parts = split_line(text)
                if parts is None:
                    assert text.startswith(f"{gate.sim} predict ")
                    continue
                assert gate.check(*parts).ok, text
                # an altered answer is not
                assert not gate.check(parts[0], "9 " + parts[1]).ok, text


def test_splits_disjoint_by_split_key(built):
    out, _, report = built
    side: dict[str, str] = {}
    for topic in report["topics"]:
        for split in SPLITS:
            for text in lines_of(out, topic, split):
                parts = split_line(text)
                if parts is None:
                    continue
                k = c7.key7(parts[0])
                assert side.setdefault(k, split) == split, f"{k!r} in {side[k]} and {split} ({topic})"
    assert set(side.values()) == set(SPLITS)
    # spellings of one lesson share a key
    gate = c7.lesson_gates()["predict_stars"]
    ln = gate.generate(random.Random(5), 1)[0]
    assert c7.key7(ln.prompt) == gate.split_key(ln.prompt)


def test_splits_disjoint_by_seed(built):
    out, _, report = built
    seeds = {s: set(report["rollouts"][f"{s}_seeds"]) for s in SPLITS}
    assert len(seeds["train"]) == 4 and len(seeds["dev"]) == len(seeds["sealed"]) == 2
    assert not (seeds["train"] & seeds["dev"]) and not (seeds["train"] & seeds["sealed"]) \
        and not (seeds["dev"] & seeds["sealed"])
    for topic in report["topics"]:
        if not topic.startswith("legacy_"):
            continue
        found = {s: set() for s in SPLITS}
        for split in SPLITS:
            for text in lines_of(out, topic, split):
                prompt = split_line(text)[0] if text.startswith("q ") else text
                seed = c7.seed_of(prompt)
                if seed is None:
                    continue
                assert seed in seeds[split], (topic, split, text)
                if not text.startswith("q "):
                    assert split == "train", text  # records of held-out seeds are not written
                found[split].add(seed)
        assert found["train"] == seeds["train"], topic
        assert found["dev"] == seeds["dev"] and found["sealed"] == seeds["sealed"], topic


def test_report_totals_consistent(built):
    out, _, report = built
    totals = {s: {"lines": 0, "tokens": 0} for s in SPLITS}
    for topic, entry in report["topics"].items():
        for split in SPLITS:
            path = out / f"{topic}-{split}.txt"
            texts = lines_of(out, topic, split)
            counts = entry["splits"][split]
            assert counts.get("lines", 0) == len(texts), (topic, split)
            assert counts.get("tokens", 0) == sum(ntokens(t) + 1 for t in texts), (topic, split)
            assert counts["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
            totals[split]["lines"] += len(texts)
            totals[split]["tokens"] += counts.get("tokens", 0)
    assert report["totals"] == totals
    assert json.loads((out / "report.json").read_text(encoding="utf-8"))["totals"] == totals
    prompts = (out / "prompts.txt").read_text(encoding="utf-8").splitlines()
    assert len(prompts) == sum(1 for t in report["topics"] for s in SPLITS
                               for x in lines_of(out, t, s) if split_line(x))


def _without_seconds(obj):
    if isinstance(obj, dict):
        return {k: _without_seconds(v) for k, v in obj.items() if k != "seconds"}
    if isinstance(obj, list):
        return [_without_seconds(v) for v in obj]
    return obj


def test_second_build_is_byte_identical(built):
    first, second, _ = built
    names = sorted(p.name for p in first.iterdir())
    assert names == sorted(p.name for p in second.iterdir())
    for name in names:
        a, b = (first / name).read_bytes(), (second / name).read_bytes()
        if name.endswith(".json"):  # reports carry wall-clock seconds and nothing else that may differ
            assert _without_seconds(json.loads(a)) == _without_seconds(json.loads(b)), name
        else:
            assert a == b, name


def test_a_finished_build_is_not_overwritten(built):
    out, _, _ = built
    with pytest.raises(FileExistsError):
        c7.build(out, workers=1, **TINY)


def test_with_truth_writes_what_train_final_needs(tmp_path):
    report = c7.build(tmp_path, seed=7, scale=0.002, world_seeds=2, eval_seeds=2, workers=4, with_truth=True)
    assert set(c7.TRUTH_TARGETS) <= set(report["topics"])
    from haishool import train_final
    groups = {train_final.category(train_final.normalized_topic(p.name)) for p in tmp_path.glob("*-train.txt")}
    assert groups == {"truth", "predict", "legacy"}
    for topic in c7.TRUTH_TARGETS:
        assert report["topics"][topic]["status"] == "target_reached"
        assert all(is_dense(x) for x in lines_of(tmp_path, topic, "train"))


def test_thin_keeps_language_and_records():
    lines = [Line("signals seed 3 step 2 0. success 0 point 5.", topic="signals", kind="record")]
    lines += [Line(f"signals seed 3 step {i} success", "1", "signals") for i in range(1, 10)]
    lines += [Line("signals seed 3 step 2 0 word fire", "ka", "signals"),
              Line("signals seed 3 meaning ka", "fire", "signals"),
              Line("nucleo seed 3 rules 7 step 4 helium", "1", "nucleo"),
              Line("signals seed 3 final words", "4", "signals")]
    kept, held = c7.thin(lines, random.Random(1), k=2)
    texts = {ln.text for ln in kept}
    assert lines[0].text in texts and all(ln.text in texts for ln in lines[-4:-1] if "word" in ln.prompt
                                          or "meaning" in ln.prompt) and lines[-1].text in texts
    steps = [ln for ln in kept if "step" in ln.prompt and ln.kind != "record" and "word" not in ln.prompt]
    assert len(steps) == 2 and len(held) == 2


def test_seed_of():
    assert c7.seed_of("world7 seed 8 5 era star age") == 85
    assert c7.seed_of("nucleo seed 1 2 rules 7 step 2 2 helium") == 12
    assert c7.seed_of("world7 seed 3 timeline") == 3
    assert c7.seed_of("chem rules 7 bond c o order 2 stable_below_k") is None
    assert c7.seed_of("stars predict lifetime mass 2") is None
