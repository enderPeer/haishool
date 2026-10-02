import hashlib
import json
import struct
import subprocess
import sys

import pytest

from haishool import anatomy_run as runner
from haishool.evo.anatomy import build_anatomy, validate_anatomy
from haishool.inspect_world import REQUIRED_SOURCE, canonical, sha, write_json


@pytest.fixture
def environment(tmp_path, monkeypatch):
    source = tmp_path / "source"
    for name in {*REQUIRED_SOURCE, "haishool/anatomy_run.py", "haishool/evo/anatomy.py",
                 "haishool/__init__.py", "haishool/evo/__init__.py", "scripts/render_anatomy_blender.py"}:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# frozen fixture\n", encoding="utf-8")
    archive = tmp_path / "world"
    archive.mkdir()
    phenotype = {"id": "p-fixture", "kind": "senses_agent", "traits": {
        "body_size": 2, "body_size_unit": "cells", "cell_types": 1, "neuron_count": 0, "speed": 1,
        "genes": {"smell": 0, "eyes": 0, "touch": 0, "ears": 0, "neuron_gene": 0, "speed": 1, "voice": 0}}}
    data = {"phenotypes": {phenotype["id"]: phenotype}, "planets": [{"id": 1, "snapshots": [
        {"step": 0, "generation": 0, "groups": [{"phenotype_id": phenotype["id"], "count": 2, "count_unit": "agents"}]}]}]}
    write_json(archive / "manifest.json", {"seed": 85})
    write_json(archive / "inhabitants.json", data)
    monkeypatch.setattr(runner, "ROOT", source)
    monkeypatch.setattr(runner, "verify_archive", lambda path: {"seed": 85})
    return source, archive, tmp_path / "anatomy"


def worker_result(out):
    request = json.loads((out / "request.json").read_text())
    data = build_anatomy(request["phenotype"], seed=request["seed"], config=request["config"])
    write_json(out / "anatomy.json", data)
    write_json(out / "anatomy-validation.json", {"validation": validate_anatomy(data), "dependencies": {},
               "content_sha256": hashlib.sha256(canonical(data)).hexdigest()})


@pytest.fixture
def archive(environment, monkeypatch):
    source, world, out = environment
    def process(args, **kwargs):
        assert args[:4] == [sys.executable, "-m", "haishool.anatomy_run", "_worker"]
        assert kwargs["cwd"] == out / "code"
        assert kwargs["env"]["PYTHONPATH"] == str(out / "code")
        worker_result(out)
        return subprocess.CompletedProcess(args, 0)
    monkeypatch.setattr(runner.subprocess, "run", process)
    manifest = runner.build(world, out)
    assert runner.verify(out) == manifest
    return out


def update_request_hash(out, request):
    write_json(out / "request.json", request)
    manifest = json.loads((out / "manifest.json").read_text())
    manifest["files"]["request.json"] = sha(out / "request.json")
    write_json(out / "manifest.json", manifest)


def test_completed_archive_binds_source_and_development(archive):
    manifest = runner.verify(archive)
    data = json.loads((archive / "anatomy.json").read_text())
    assert manifest["world_seed"] == 85 and data["seed"] == 85
    assert data["source"]["body_size"] == len(data["cells"]) == 2
    assert data["source"]["neuron_count"] == 0
    assert all("eyes" not in cell["roles"] for cell in data["cells"])


@pytest.mark.parametrize("change", ["seed", "phenotype", "config"])
def test_changed_request_cannot_relabel_existing_anatomy(archive, change):
    request = json.loads((archive / "request.json").read_text())
    if change == "seed": request["seed"] = 99
    elif change == "phenotype": request["phenotype"]["id"] = "another-phenotype"
    else: request["config"] = {"field_curvature": .2}
    update_request_hash(archive, request)
    with pytest.raises(ValueError, match="seed mismatch|source phenotype|configuration differs"):
        runner.verify(archive)


def test_world_label_cannot_change_independently(archive):
    manifest = json.loads((archive / "manifest.json").read_text())
    manifest["world_seed"] = 999
    write_json(archive / "manifest.json", manifest)
    with pytest.raises(ValueError, match="world provenance"):
        runner.verify(archive)


@pytest.mark.parametrize("filename", ["anatomy.json", "code/haishool/evo/anatomy.py", "code/scripts/render_anatomy_blender.py"])
def test_data_core_and_renderer_tampering_are_rejected(archive, filename):
    with (archive / filename).open("a", encoding="utf-8") as handle:
        handle.write("changed")
    with pytest.raises(ValueError, match="mismatch|source changed"):
        runner.verify(archive)


def test_worker_refuses_execution_from_live_workspace(environment):
    _, _, out = environment
    with pytest.raises(RuntimeError, match="frozen source"):
        runner.worker(out)


def test_failed_build_retains_source_request_and_log(environment, monkeypatch):
    _, world, out = environment
    def failed(args, **kwargs):
        kwargs["stdout"].write("development failed\n")
        raise subprocess.CalledProcessError(1, args)
    monkeypatch.setattr(runner.subprocess, "run", failed)
    with pytest.raises(subprocess.CalledProcessError):
        runner.build(world, out)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert (out / "request.json").exists() and (out / "code/source-profile.json").exists()
    assert "development failed" in (out / "build.log").read_text()


def test_output_is_never_overwritten(environment):
    _, world, out = environment
    out.mkdir()
    (out / "keep.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        runner.build(world, out)
    assert [p.name for p in out.iterdir()] == ["keep.txt"]


def rendering_result(out, bad_input=False, bad_output=False):
    destination = out / "preview"
    outputs = {"png": destination / "anatomy-preview.png", "blend": destination / "preview.blend", "mesh": destination / "surface.glb"}
    outputs["png"].write_bytes(b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", 960, 540))
    outputs["blend"].write_bytes(b"BLENDER fixture")
    outputs["mesh"].write_bytes(b"glTF fixture")
    receipt = {"source": {"sha256": "bad" if bad_input else sha(out / "anatomy.json"), "cell_count": 2},
               "geometry": {"source_coordinate_transform": "identity", "added_anatomical_features": []},
               "outputs": {k: {"path": str(p), "sha256": "bad" if bad_output and k == "mesh" else sha(p)} for k, p in outputs.items()}}
    write_json(destination / "preview-receipt.json", receipt)


@pytest.mark.parametrize("fault", [None, "input", "mesh"])
def test_render_requires_receipt_binding_not_just_png_dimensions(archive, monkeypatch, fault):
    def render_process(args, **kwargs):
        assert "--factory-startup" in args and "--background" in args
        assert str(archive / "code/scripts/render_anatomy_blender.py") in args
        rendering_result(archive, fault == "input", fault == "mesh")
    monkeypatch.setattr(runner.subprocess, "run", render_process)
    if fault:
        with pytest.raises(ValueError, match="different anatomy input|does not match receipt"):
            runner.render(archive, archive / "blender", preview=True)
        assert not (archive / "preview/verified-output.json").exists()
    else:
        result = runner.render(archive, archive / "blender", preview=True)
        assert result["verified"] and (result["width"], result["height"]) == (960, 540)
        assert (archive / "preview/verified-output.json").exists()


def test_render_receipt_must_name_the_delivered_png(archive, monkeypatch):
    def render_process(args, **kwargs):
        rendering_result(archive)
        destination = archive / "preview"
        alternate = destination / "another-image.png"
        alternate.write_bytes((destination / "anatomy-preview.png").read_bytes())
        receipt_path = destination / "preview-receipt.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["outputs"]["png"] = {"path": str(alternate), "sha256": sha(alternate)}
        write_json(receipt_path, receipt)
    monkeypatch.setattr(runner.subprocess, "run", render_process)
    with pytest.raises(ValueError, match="different delivered PNG"):
        runner.render(archive, archive / "blender", preview=True)
    assert not (archive / "preview/verified-output.json").exists()
