import json
from pathlib import Path

import pytest

from haishool import curriculum
from haishool.truth import Line, Verdict


def test_eval_sample_is_independent_of_arrival_order_and_includes_late_examples(tmp_path):
    lines = [Line(f"calc {i // 10} {i % 10} plus 1", "1", kind="calc") for i in range(100)]
    paths = [tmp_path / name for name in ("forward", "reverse")]
    reports = []
    for path, group in zip(paths, (lines, list(reversed(lines)))):
        path.mkdir()
        writer = curriculum.Writer(path, "maths", 7, 8)
        for line in group:
            writer.add(line, split="dev")
        reports.append(writer.close())
    first = (paths[0] / "maths-dev.txt").read_bytes()
    assert first == (paths[1] / "maths-dev.txt").read_bytes()
    assert reports[0] == reports[1]
    assert reports[0]["splits"]["dev"]["lines"] == 8
    assert reports[0]["dropped"]["reserved_not_saved"] == 92
    selected = first.decode().splitlines()
    assert any(line.text in selected for line in lines[50:])


def test_writer_rejects_contradictions_and_preserves_earlier_answers(tmp_path):
    writer = curriculum.Writer(tmp_path, "toy", 7, 8, {"toy 2": {"4"}})
    writer.add(Line("toy 2", "9"), split="train")
    writer.add(Line("toy 1", "2"), split="train")
    writer.add(Line("toy 1", "2"), split="train")
    with pytest.raises(ValueError, match="contradictory duplicate"):
        writer.add(Line("toy 1", "3"), split="train")
    report = writer.close()
    assert report["dropped"] == {"earlier_collision": 1, "duplicate": 1}
    assert report["splits"]["train"]["lines"] == 1
    assert (tmp_path / "toy-train.txt").read_text().strip() == "q toy 1. a 2."


def test_saturated_generator_is_bounded_and_reported(tmp_path, monkeypatch):
    from haishool import round5

    prompt = next(f"toy {i}" for i in range(100) if curriculum.partition(f"toy {i}", 7) == "train")

    class Finite:
        def records(self): return []
        def generate(self, rng, n): return [Line(prompt, "1")]
        def check(self, prompt, answer): return Verdict(answer == "1", "1")

    monkeypatch.setattr(curriculum, "gate_for", lambda topic: Finite())
    monkeypatch.setattr(round5, "earlier_answers", lambda: {})
    topic, report = curriculum.build_topic(("toy", str(tmp_path), 100, 7, 8))
    assert topic == "toy"
    assert report["status"] == "saturated"
    assert report["rounds"] == 8
    assert report["splits"]["train"]["lines"] == 1
    assert json.loads((tmp_path / "toy-report.json").read_text()) == report


@pytest.mark.parametrize("kwargs", [{"scale": 0}, {"workers": 0}, {"eval_limit": 0},
                                    {"world_seeds": 0}, {"world_seeds": 9998}])
def test_invalid_build_sizes_fail_before_creating_output(tmp_path, kwargs):
    destination = tmp_path / "out"
    with pytest.raises(ValueError):
        curriculum.build(destination, **kwargs)
    assert not destination.exists()
