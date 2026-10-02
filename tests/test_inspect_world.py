import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from haishool import inspect_world as inspector


TEMPLATE = '<html><script type="application/json" id="inhabitant-data">{}</script><script id="inhabitant-config" type="application/json">{}</script></html>'


def source_tree(root):
    files = {"haishool/__init__.py": "", "haishool/evo/__init__.py": "",
             "haishool/inspect_world.py": "# frozen worker\n",
             "haishool/evo/world7.py": "MARKER = 'original'\n",
             "haishool/evo/inhabitants.py": "# observer\n",
             "data/truth-v5/elements.json": '{"example": 1}',
             "docs/inhabitants.html": TEMPLATE}
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def fake_results(directory):
    req = json.loads((directory / "request.json").read_text())
    data = {"schema_version": "inhabitants-v1", "mapping_version": "phenotype-display-v1",
            "source": {"world_seed": req["seed"]}, "genotypes": {}, "phenotypes": {}, "planets": []}
    inspector.write_json(directory / "world.json", {"seed": req["seed"], "params": req["params"]})
    inspector.write_json(directory / "inhabitants.json", data)
    inspector.write_json(directory / "receipt.json", {
        "validation": {"world_conserved": True, "reason": "fixture"},
        "inhabitants_content_sha256": hashlib.sha256(inspector.canonical(data)).hexdigest()})


@pytest.fixture
def archive(tmp_path, monkeypatch):
    source = source_tree(tmp_path / "source")
    destination = tmp_path / "archive"
    def worker(args, **kwargs):
        fake_results(destination)
        return subprocess.CompletedProcess(args, 0)
    monkeypatch.setattr(inspector.subprocess, "run", worker)
    inspector.run_world(85, destination, source_root=source)
    return destination


def test_freeze_captures_bytes_without_importing_simulation(tmp_path):
    source = source_tree(tmp_path / "source")
    world = source / "haishool/evo/world7.py"
    world.write_text("raise RuntimeError('must not import evolving source')\n")
    before = set(sys.modules)
    profile = inspector.freeze_source(source, tmp_path / "frozen")
    assert set(sys.modules) == before
    assert "data/truth-v5/elements.json" in profile["files"]
    saved = (tmp_path / "frozen/haishool/evo/world7.py").read_bytes()
    world.write_text("# edited later\n")
    assert (tmp_path / "frozen/haishool/evo/world7.py").read_bytes() == saved
    assert profile["sha256"] == hashlib.sha256(inspector.canonical(profile["files"])).hexdigest()


def test_changing_source_during_capture_is_rejected(tmp_path, monkeypatch):
    source = source_tree(tmp_path / "source")
    monkeypatch.setattr(inspector, "sha", lambda path: "not-the-captured-digest")
    with pytest.raises(RuntimeError, match="changing during capture"):
        inspector.freeze_source(source, tmp_path / "frozen")
    assert not (tmp_path / "frozen").exists()


def test_worker_runs_only_from_frozen_tree(tmp_path, monkeypatch):
    source = source_tree(tmp_path / "source")
    destination = tmp_path / "archive"
    def worker(args, **kwargs):
        assert args[:4] == [sys.executable, "-m", "haishool.inspect_world", "_worker"]
        assert kwargs["cwd"] == destination / "code"
        assert kwargs["env"]["PYTHONPATH"] == str(destination / "code")
        assert kwargs["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
        assert kwargs["check"] and kwargs["timeout"] == 180
        (source / "haishool/evo/world7.py").write_text("MARKER = 'edited'\n")
        assert "original" in (kwargs["cwd"] / "haishool/evo/world7.py").read_text()
        fake_results(destination)
    monkeypatch.setattr(inspector.subprocess, "run", worker)
    result = inspector.run_world(85, destination, source_root=source, steps=[0])
    assert result["manifest"]["status"] == "complete"
    assert inspector.verify_archive(destination) == result["manifest"]


def test_complete_archive_roundtrip(archive):
    manifest = inspector.verify_archive(archive)
    assert manifest["seed"] == 85
    assert set(manifest["files"]) == set(inspector.ARCHIVE_FILES)
    assert set(inspector.REQUIRED_SOURCE) <= manifest["source_profile"]["files"].keys()


@pytest.mark.parametrize("name", ["request.json", "receipt.json", "world.json", "inhabitants.json", "viewer.html",
                                  "code/haishool/evo/world7.py", "code/source-profile.json"])
def test_tampered_archive_is_rejected(archive, name):
    with (archive / name).open("a", encoding="utf-8") as handle:
        handle.write("tampered")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        inspector.verify_archive(archive)


@pytest.mark.parametrize("field", ["files", "source_profile"])
def test_manifest_cannot_omit_required_hashes(archive, field):
    manifest = json.loads((archive / "manifest.json").read_text())
    manifest[field] = {} if field == "files" else {"files": {}, "sha256": hashlib.sha256(inspector.canonical({})).hexdigest()}
    inspector.write_json(archive / "manifest.json", manifest)
    with pytest.raises(ValueError, match="missing required"):
        inspector.verify_archive(archive)


def test_manifest_seed_must_match_archived_records(archive):
    manifest = json.loads((archive / "manifest.json").read_text())
    manifest["seed"] = 86
    inspector.write_json(archive / "manifest.json", manifest)
    with pytest.raises(ValueError, match="seed provenance"):
        inspector.verify_archive(archive)


def test_manifest_cannot_verify_file_outside_archive(archive, monkeypatch):
    outside = archive.parent / "outside.txt"
    outside.write_text("outside")
    manifest = json.loads((archive / "manifest.json").read_text())
    manifest["files"]["../outside.txt"] = inspector.sha(outside)
    inspector.write_json(archive / "manifest.json", manifest)
    real_sha = inspector.sha
    def safe_sha(path):
        assert Path(path).resolve().is_relative_to(archive)
        return real_sha(path)
    monkeypatch.setattr(inspector, "sha", safe_sha)
    with pytest.raises(ValueError, match="file mismatch"):
        inspector.verify_archive(archive)


def test_unhashed_source_cannot_be_added(archive):
    (archive / "code/haishool/new_logic.py").write_text("# new code")
    with pytest.raises(ValueError, match="source inventory"):
        inspector.verify_archive(archive)


def test_code_symlink_cannot_escape_archive(archive):
    code = archive / "code"
    outside = archive.parent / "external-code"
    code.rename(outside)
    try:
        code.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("platform does not permit symlink creation")
    with pytest.raises(ValueError, match="outside the archive"):
        inspector.verify_archive(archive)


def test_source_symlink_cannot_escape_source_root(tmp_path):
    source = source_tree(tmp_path / "source")
    outside = tmp_path / "private.py"
    outside.write_text("# not a source file")
    try:
        (source / "haishool/external.py").symlink_to(outside)
    except OSError:
        pytest.skip("platform does not permit symlink creation")
    with pytest.raises(ValueError, match="outside the source root"):
        inspector.freeze_source(source, tmp_path / "frozen")


def test_baked_json_cannot_escape_script_element():
    payload = {"message": '</script><script>alert("x")</script><img src=x onerror=alert(1)>\u2028'}
    baked = inspector.bake_viewer(TEMPLATE, payload, live=True)
    assert baked.count("<script") == 2 and baked.count("</script>") == 2
    assert "\\u003c/script>" in baked
    assert '<img src=x' not in baked
    assert '"live": true' in baked
    with pytest.raises(ValueError, match="missing data element"):
        inspector.bake_viewer("<html></html>", payload)


@pytest.mark.parametrize("seed", [True, False, "85", 85.0, None, -1, 10 ** 9 + 1])
def test_invalid_seed_does_not_create_output(tmp_path, seed):
    destination = tmp_path / "archive"
    with pytest.raises(ValueError, match="seed"):
        inspector.run_world(seed, destination)
    assert not destination.exists()


def test_existing_destination_is_never_overwritten(tmp_path):
    destination = tmp_path / "archive"
    destination.mkdir()
    sentinel = destination / "keep.txt"
    sentinel.write_text("precious")
    with pytest.raises(FileExistsError):
        inspector.run_world(85, destination)
    assert sentinel.read_text() == "precious"
    assert list(destination.iterdir()) == [sentinel]


def test_worker_failure_preserves_manifest_request_and_log(tmp_path, monkeypatch):
    source = source_tree(tmp_path / "source")
    destination = tmp_path / "archive"
    def worker(args, **kwargs):
        kwargs["stdout"].write("simulator failed here\n")
        raise subprocess.CalledProcessError(2, args)
    monkeypatch.setattr(inspector.subprocess, "run", worker)
    with pytest.raises(subprocess.CalledProcessError):
        inspector.run_world(85, destination, source_root=source)
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["status"] == "failed" and "CalledProcessError" in manifest["error"]
    assert (destination / "request.json").exists()
    assert "simulator failed" in (destination / "run.log").read_text()
    assert (destination / "code/source-profile.json").exists()
    with pytest.raises(ValueError, match="not complete"):
        inspector.verify_archive(destination)
