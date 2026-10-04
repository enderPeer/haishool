import pytest

from haishool.life10.protocol import digest, experiment_identity, require_source, write_json


def test_identity_changes_with_laws_seed_and_controls():
    a = experiment_identity({"rate": 1}, [0], {"sha256": "a"}, ["normal"])
    assert a != experiment_identity({"rate": 2}, [0], {"sha256": "a"}, ["normal"])
    assert a != experiment_identity({"rate": 1}, [1], {"sha256": "a"}, ["normal"])
    assert a != experiment_identity({"rate": 1}, [0], {"sha256": "a"}, ["dark"])


def test_source_mismatch_refuses_continuation():
    with pytest.raises(RuntimeError, match="Source changed"):
        require_source({"sha256": "different"})


def test_atomic_json_rejects_invalid_numbers(tmp_path):
    p = tmp_path / "out.json"
    write_json(p, {"valid": 1})
    old = p.read_bytes()
    with pytest.raises(ValueError):
        write_json(p, {"bad": float("nan")})
    assert p.read_bytes() == old
