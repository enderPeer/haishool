import json
import sys
from types import SimpleNamespace

import pytest

from haishool import curriculum, final_run, student
from haishool.truth import Line, maths, split_line


def test_command_accepts_path_arguments_and_propagates_failure(tmp_path):
    script = tmp_path / "child.py"
    script.write_text("print('child ran')\n", encoding="utf-8")
    log = tmp_path / "child.log"
    final_run.command([script], log)
    assert "child ran" in log.read_text()
    script.write_text("raise SystemExit(3)\n", encoding="utf-8")
    with pytest.raises(final_run.subprocess.CalledProcessError):
        final_run.command([script], log)


def test_pipeline_completes_both_sizes_and_locks_before_sealed(tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    (data / "report.json").write_text("{}")
    root = tmp_path / "runs"
    calls = []
    def fake_command(args, log):
        from pathlib import Path
        args = list(map(str, args))
        calls.append(args)
        out = Path(args[args.index("--out") + 1])
        if args[1] == "haishool.train_final":
            out.mkdir(parents=True, exist_ok=True)
            (out / "student.pt").write_bytes(str(out).encode())
            (out / "training-report.json").write_text("{}")
        elif args[2] == "feedback":
            out.mkdir(parents=True, exist_ok=True)
            (out / "report.json").write_text("{}")
        else:
            if args[args.index("--split") + 1] == "sealed":
                name = out.name.removesuffix("-sealed.json")
                assert (root / f"{name}-selection.json").exists()
            curriculum.json_write(out, {"selection_score": .5, "old": {
                "old-recall": {"scores": {"all": {"exact": .99}}}}})
    monkeypatch.setattr(final_run, "command", fake_command)
    final_run.pipeline(root, data, steps=2, feedback_steps=1, production_steps=1, device="cpu")
    assert json.loads((root / "status.json").read_text())["phase"] == "complete"
    assert set(json.loads((root / "summary.json").read_text())["models"]) == {"6x384", "8x512"}
    assert len([a for a in calls if "--holdout" in a and a[a.index("--holdout") + 1] == "0"]) == 2
    assert len([a for a in calls if "sealed" in a]) == 2


def fixture_data(path, seed=7):
    path.mkdir()
    splits = {}
    for split in ("train", "dev", "sealed"):
        source = path / f"maths-{split}.txt"
        source.write_text("q calc 1 plus 2. a 3.\n", encoding="utf-8")
        splits[split] = {"sha256": curriculum.digest(source)}
    (path / "prompts.txt").write_text("", encoding="utf-8")
    curriculum.json_write(path / "report.json", {"seed": seed, "topics": {"maths": {"splits": splits}}})
    return path


def test_evaluation_fingerprint_includes_verified_shards_and_old_files(tmp_path):
    data = fixture_data(tmp_path / "data")
    old = tmp_path / "old"
    old.mkdir()
    for name in ("old-recall", "old-heldout"):
        (old / f"{name}.txt").write_text("q toy color. a red.\n")
    _, _, first = final_run.evaluation_inputs(data, "dev", old)
    assert set(first["files"]) == {"maths-dev.txt"}
    assert set(first["old_files"]) == {"old-recall.txt", "old-heldout.txt"}
    (old / "old-heldout.txt").write_text("q toy color. a blue.\n")
    _, _, second = final_run.evaluation_inputs(data, "dev", old)
    assert first != second


def test_invalid_gold_file_does_not_create_evaluation_start_marker(tmp_path):
    data = fixture_data(tmp_path / "data")
    (data / "maths-sealed.txt").write_text("q calc 1 plus 2. a 4.\n")
    output = tmp_path / "sealed.json"
    with pytest.raises(ValueError, match="hash mismatch"):
        final_run.evaluate(tmp_path / "missing.pt", data, "sealed", output)
    assert not output.exists()
    assert not output.with_suffix(".started.json").exists()


def test_extra_eval_file_is_not_silently_scored(tmp_path):
    data = fixture_data(tmp_path / "data")
    (data / "stale-dev.txt").write_text("q calc 1 plus 2. a 3.\n")
    with pytest.raises(ValueError, match="files do not match"):
        final_run.evaluation_inputs(data, "dev")


def test_feedback_uses_custom_dataset_seed(monkeypatch, tmp_path):
    data = fixture_data(tmp_path / "data", 7)
    gate = maths.gate()
    candidates = [f"calc {i // 10} {i % 10} plus 2" for i in range(10, 100)]
    wanted = next(p for p in candidates if curriculum.partition(p, 7) == "train"
                  and curriculum.partition(p) != "train")
    reserved = next(p for p in candidates if curriculum.partition(p, 7) != "train"
                    and curriculum.partition(p) == "train")
    generated = [Line(p, gate.check(p, "").expected, "maths", "calc") for p in (reserved, wanted)]
    monkeypatch.setattr(gate, "generate", lambda rng, n: list(generated))
    monkeypatch.setattr(final_run, "all_gates", lambda: [gate])
    monkeypatch.setattr(final_run, "ask_many", lambda model, vocab, prompts, device: [""] * len(prompts))
    monkeypatch.setattr(student, "load", lambda checkpoint, device: (None, None))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(set_num_threads=lambda n: None))
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"test checkpoint")
    out = tmp_path / "feedback"
    report = final_run.feedback(checkpoint, data, out, n=1, device="cpu")
    lines = (out / "maths-train.txt").read_text().splitlines()
    assert [split_line(line)[0] for line in lines] == [wanted]
    assert report["fingerprint"]["seed"] == 7
    assert report["questions"] == 1
