"""Checks for isolated launch planning and immutable source snapshots (no SSH/jobs)."""
import argparse
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile

import pytest


spec = importlib.util.spec_from_file_location("life10_cluster", Path(__file__).resolve().parents[2] / "scripts" / "life10_cluster.py")
cluster = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cluster)


def sources(tmp_path):
    data = tmp_path / "data" / "truth-v5"
    data.mkdir(parents=True)
    (data / "atomic_weights.json").write_text('{"H": 1}')
    (data / "large-private.json").write_text("must not leave machine")
    package = tmp_path / "haishool"
    package.mkdir()
    (package / "__init__.py").write_text('"""package"""')
    for name in ("life10", "cosmos", "evo", "truth"):
        folder = package / name
        folder.mkdir()
        (folder / "__init__.py").write_text(f'"""{name}"""')
        (folder / "private.data").write_text("must not leave machine")
    return tmp_path


@pytest.mark.parametrize("value", ("../other", "a/b", "x;touch z", "..", "", "x'bad"))
def test_experiment_names_reject_traversal_and_shell_syntax(value):
    with pytest.raises(ValueError):
        cluster.valid_name(value)


def test_lanes_exclude_integrated_and_insufficient_memory_and_reserve_threads():
    record = {"alias": "falke64", "cpu_count": 12, "gpus": [
        {"index": 0, "discrete": True, "free_bytes": 10 * 1024**3},
        {"index": 1, "discrete": True, "free_bytes": 100 * 1024**2},
        {"index": 2, "discrete": False, "free_bytes": 30 * 1024**3},
    ]}
    lanes = cluster.choose_lanes([record], 10, 2, 256)["falke64"]
    assert [lane["device"] for lane in lanes] == ["cuda:0", "cpu"]
    assert lanes[-1]["threads"] == 9
    assert [lane["seeds"] for lane in lanes] == [[10, 11], [12, 13]]


def test_unreachable_or_torchless_hosts_do_not_get_jobs():
    assert cluster.choose_lanes([{"alias": "adler40", "error": "offline"},
                                 {"alias": "knecht24", "torch_error": "missing"}], 0, 2, 256) == {}


def test_snapshot_is_deterministic_and_contains_only_selected_python_sources(tmp_path):
    sources(tmp_path)
    one, hashes = cluster.snapshot_sources(tmp_path)
    two, _ = cluster.snapshot_sources(tmp_path)
    assert one == two
    with tarfile.open(fileobj=io.BytesIO(one)) as archive:
        assert set(archive.getnames()) == set(hashes)
        assert all((member.name.endswith(".py") or member.name == "data/truth-v5/atomic_weights.json") and member.isfile() for member in archive.getmembers())
        for name, expected in hashes.items():
            assert hashlib.sha256(archive.extractfile(name).read()).hexdigest() == expected


def test_missing_new_engine_prevents_partial_snapshot(tmp_path):
    data = tmp_path / "data" / "truth-v5"
    data.mkdir(parents=True)
    (data / "atomic_weights.json").write_text("{}")
    (tmp_path / "haishool").mkdir()
    (tmp_path / "haishool" / "__init__.py").write_text("")
    with pytest.raises(ValueError, match="source package is missing"):
        cluster.snapshot_sources(tmp_path)


def test_manifest_source_and_worker_tampering_are_detected(tmp_path):
    folder = tmp_path / "trial"
    folder.mkdir()
    payload = b"source"
    worker = b"worker"
    manifest = {"source_sha256": hashlib.sha256(payload).hexdigest(),
                "worker_sha256": hashlib.sha256(worker).hexdigest()}
    manifest["manifest_sha256"] = hashlib.sha256(cluster.canonical(manifest)).hexdigest()
    (folder / "manifest.json").write_text(json.dumps(manifest))
    (folder / "source.tar.gz").write_bytes(payload)
    (folder / "worker.py").write_bytes(worker)
    args = argparse.Namespace(experiment="trial", output_root=tmp_path)
    assert cluster.load_experiment(args)[1] == manifest
    (folder / "worker.py").write_bytes(b"changed")
    with pytest.raises(ValueError, match="worker checksum mismatch"):
        cluster.load_experiment(args)
    (folder / "worker.py").write_bytes(worker)
    (folder / "source.tar.gz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="source archive checksum mismatch"):
        cluster.load_experiment(args)


def test_embedded_remote_scripts_compile():
    compile(cluster.PROBE, "probe.py", "exec")
    compile(cluster.WORKER, "worker.py", "exec")


def test_shared_chain_seeds_use_independent_repeat_and_paired_profiles():
    records = [{"alias": "adler40", "cpu_count": 24, "gpus": [
        {"index": 0, "discrete": True, "free_bytes": 10*1024**3}]}]
    jobs = cluster.choose_lanes(records, 0, 2, 256, seeds=[0, 3, 15], repeat=7)["adler40"]
    assert [job["seeds"] for job in jobs] == [[0, 3, 15], [0, 3, 15]]
    assert [job["repeat"] for job in jobs] == [7, 8]


def test_cli_defaults_do_not_launch(monkeypatch, capsys):
    monkeypatch.setattr(cluster, "inventory", lambda hosts: {"hosts": hosts})
    monkeypatch.setattr(cluster, "launch", lambda args: pytest.fail("unexpected job launch"))
    assert cluster.main(["inventory"]) == 0
    assert "adler40" in capsys.readouterr().out
