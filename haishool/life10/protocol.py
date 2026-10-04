"""Immutable experiment identities and atomic, resumable evidence files."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def source_identity():
    paths = sorted((ROOT / "haishool" / "life10").glob("*.py"))
    for package in ("cosmos", "evo", "truth"):
        paths += sorted((ROOT / "haishool" / package).glob("*.py"))
    paths += [ROOT / "data" / "truth-v5" / "atomic_weights.json"]
    files = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    return {"sha256": digest(files), "files": files}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def experiment_identity(config, seeds, source, profiles, *, chain_inputs_sha256=None):
    return digest({"config": config, "seeds": seeds, "source": source, "profiles": profiles,
                   "chain_inputs_sha256": chain_inputs_sha256})


def require_source(expected):
    if source_identity() != expected:
        raise RuntimeError("Source changed: this experiment cannot continue under different laws")
