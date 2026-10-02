import json
import random
from collections import defaultdict
from pathlib import Path

import pytest

from haishool import student, train_final


def fixture_files(tmp_path):
    data = tmp_path / "curriculum"
    data.mkdir()
    records = tmp_path / "records.jsonl"
    records.write_text(json.dumps({"obj": "apple", "values": {"kind": "fruit", "color": "red"}}) + "\n",
                       encoding="utf-8")
    hops = tmp_path / "hops.txt"
    hops.write_text("q apple borders pear. a yes.\nq apple hop pear. a apple borders pear.\n"
                    "q apple hops pear. a 1.\n", encoding="utf-8")
    report = {"seed": 20261001, "topics": {}}
    for topic in [*train_final.TRUTH_SHARES, "predict_life", "legacy_life"]:
        path = data / f"{topic}-train.txt"
        path.write_text(f"q {topic} input 1. a 2.\n", encoding="utf-8")
        report["topics"][topic] = {"splits": {"train": {"sha256": train_final.digest(path)}}}
        (data / f"{topic}-dev.txt").write_text("q secret_dev. a secret_dev.\n", encoding="utf-8")
        (data / f"{topic}-sealed.txt").write_text("q secret_sealed. a secret_sealed.\n", encoding="utf-8")
    (data / "report.json").write_text(json.dumps(report), encoding="utf-8")
    return data, records, hops


def prepare_fixture(tmp_path, feedback=None):
    data, records, hops = fixture_files(tmp_path)
    report, paths, hashes = train_final.input_manifest(data, records, hops, None, feedback)
    out = tmp_path / "run"
    out.mkdir()
    prepared = train_final.prepare(data, out, records, hops, holdout=.1, seed=20261001, ctx=32,
                                   paths=paths, report=report, feedback=feedback)
    return data, records, hops, out, prepared, hashes


def test_preparation_excludes_reserved_vocab_and_uses_one_token_copy(tmp_path):
    _, _, _, out, prepared, hashes = prepare_fixture(tmp_path)
    assert "secret_dev" not in prepared["itos"]
    assert "secret_sealed" not in prepared["itos"]
    assert not any("-dev.txt" in p or "-sealed.txt" in p for p in hashes)
    totals = defaultdict(float)
    for source in prepared["sources"]:
        totals[train_final.category(source["topic"])] += source["weight"]
        path = out / "cache" / source["bin"]
        assert path.stat().st_size == 4 * source["tokens"]
        assert train_final.digest(path) == source["sha256"]
    assert dict(totals) == pytest.approx({"old": .4, "truth": .4, "predict": .15, "legacy": .05})
    truth = {s["topic"]: s["weight"] for s in prepared["sources"] if s["topic"] in train_final.TRUTH_SHARES}
    assert truth == pytest.approx({k: .4 * v for k, v in train_final.TRUTH_SHARES.items()})
    assert (out / "old-heldout.txt").exists()
    assert (out / "old-recall.txt").read_text().startswith("q ")


def test_source_hash_mismatch_rejected_before_training(tmp_path):
    data, records, hops = fixture_files(tmp_path)
    (data / "maths-train.txt").write_text("q changed. a source.\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        train_final.input_manifest(data, records, hops, None, None)


def test_feedback_keeps_owning_topic_share_and_gets_twenty_percent(tmp_path):
    from haishool.curriculum import partition
    from haishool.truth.maths import gate
    row = next(row for row in gate().generate(random.Random(7), 100)
               if row.kind != "record" and partition(row.prompt) == "train")
    feedback = tmp_path / "feedback.txt"
    feedback.write_text(row.text + f"\nq judge {row.prompt} answer {row.answer}. a right.\n", encoding="utf-8")
    _, _, _, _, prepared, _ = prepare_fixture(tmp_path, feedback)
    maths = [s for s in prepared["sources"] if s["topic"] == "maths"]
    assert sum(s["weight"] for s in maths) == pytest.approx(.2)
    assert sum(s["weight"] for s in maths if s["feedback"]) == pytest.approx(.04)


def test_empty_feedback_is_valid(tmp_path):
    feedback = tmp_path / "feedback"
    feedback.mkdir()
    (feedback / "maths-train.txt").write_text("", encoding="utf-8")
    _, _, _, _, prepared, _ = prepare_fixture(tmp_path, feedback)
    assert not any(s["feedback"] for s in prepared["sources"])


def test_feedback_reserved_inputs_rejected(tmp_path):
    from haishool.curriculum import partition
    from haishool.truth.maths import gate
    row = next(row for row in gate().generate(random.Random(7), 100)
               if row.kind != "record" and partition(row.prompt) != "train")
    feedback = tmp_path / "feedback.txt"
    feedback.write_text(row.text + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="reserved"):
        prepare_fixture(tmp_path, feedback)


def test_tiny_cpu_checkpoint_load_and_exact_resume(tmp_path):
    torch = pytest.importorskip("torch")
    data, records, hops = fixture_files(tmp_path)
    out = tmp_path / "run"
    args = dict(data=data, out=out, records=records, hops=hops, layers=1, width=16,
                heads=2, ctx=32, steps=2, batch=2, device="cpu", checkpoint_every=1)
    result = train_final.train(**args)
    assert result["sampled_tokens"] == 2 * 2 * 32
    assert result["steps"] == 2
    model, vocab = student.load(out / "student.pt")
    assert "secret_sealed" not in vocab.itos
    first = {k: v.clone() for k, v in model.state_dict().items()}
    initialized = torch.load(out / "initialized.pt", weights_only=True)
    assert initialized["step"] == 0
    final = torch.load(out / "latest.pt", weights_only=True)
    assert final["step"] == 2
    assert "optimizer_state" in final and "torch_rng" in final and "sampler_rng" in final
    # Restart from the transactional pre-step checkpoint with the same total-step schedule.
    train_final.train(**args, resume=out / "initialized.pt")
    restored, _ = student.load(out / "student.pt")
    for key, tensor in first.items():
        assert torch.equal(tensor, restored.state_dict()[key])
    assert json.loads((out / "status.json").read_text())["status"] == "completed"
    with pytest.raises(ValueError, match="fingerprint"):
        train_final.train(**{**args, "batch": 3}, resume=out / "latest.pt")


def test_resume_rejects_mutated_token_cache(tmp_path):
    pytest.importorskip("torch")
    data, records, hops = fixture_files(tmp_path)
    out = tmp_path / "run"
    args = dict(data=data, out=out, records=records, hops=hops, layers=1, width=16,
                heads=2, ctx=32, steps=1, batch=1, device="cpu", checkpoint_every=1)
    train_final.train(**args)
    cache_file = next((out / "cache").glob("*.u32"))
    with cache_file.open("r+b") as handle:
        handle.write(b"\xff\xff\xff\xff")
    with pytest.raises(ValueError, match="cache hash mismatch"):
        train_final.train(**args, resume=out / "latest.pt")
