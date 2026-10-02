import json
from pathlib import Path

import pytest

from haishool.life8.checkpoint import load, save
from haishool.life8.cli import run_world
from haishool.life8.config import Config
from haishool.life8.dataset import export_transitions
from haishool.life8.world import World


def small_world():
    return World.create(seed=7, config=Config(population=4, food_patches=4, initial_objects=4))


def test_verified_checkpoint_rejects_altered_state(tmp_path):
    path = tmp_path / "checkpoint.json"
    save(small_world(), path)
    data = json.loads(path.read_text())
    data["state"]["tick"] += 1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="hash mismatch"):
        load(path)


def test_source_identity_is_verified_before_resume(tmp_path):
    path = tmp_path / "checkpoint.json"
    save(small_world(), path)
    data = json.loads(path.read_text())
    data["source"]["files"]["world.py"] = "altered"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="source manifest"):
        load(path)


def test_run_archive_and_resume_preserve_final_state(tmp_path):
    direct = small_world()
    for _ in range(10):
        direct.step()
    first = tmp_path / "first"
    run_world(small_world(), first, steps=5, sample_every=2, checkpoint_every=3)
    resumed, _ = load(first / "checkpoint.json")
    second = tmp_path / "second"
    result = run_world(resumed, second, steps=5, sample_every=2, checkpoint_every=3)
    restored, _ = load(second / "checkpoint.json")
    assert restored.to_dict() == direct.to_dict()
    assert result["checkpoint_roundtrip_verified"] is True
    assert (second / "viewer.html").is_file()
    assert "window.MATRIX_EMBED=" in (second / "viewer.html").read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        run_world(restored, second, steps=1)


def test_experience_export_rejects_modified_observer_log(tmp_path):
    archive = tmp_path / "run"
    run_world(small_world(), archive, steps=3, sample_every=1)
    result = export_transitions(archive, tmp_path / "experience.jsonl")
    assert result["examples"] > 0
    rows = [json.loads(line) for line in (tmp_path / "experience.jsonl").read_text().splitlines()]
    assert all("executed" in row and "observation" in row and "next_observation" in row for row in rows)
    assert all("rng_state" not in row and "world_state" not in row for row in rows)
    with (archive / "events.jsonl").open("a") as stream:
        stream.write('{"type":"invented milestone"}\n')
    with pytest.raises(ValueError, match="hash mismatch"):
        export_transitions(archive, tmp_path / "altered.jsonl")
