"""Portable JSON checkpoints, verified without executing serialized code."""
from __future__ import annotations

import hashlib
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = "life8-checkpoint-v2"  # Matrix 2cf045a wrote matrix-checkpoint-v1; refused
RUN_SCHEMA = "life8-run-v2"
ROOT = Path(__file__).resolve().parent.parent


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(part)
    return value.hexdigest()


def source_identity():
    """Hash of the engine modules (top level of the package). life8 keeps its
    analysis scripts in a subpackage so editing them does not invalidate runs."""
    package = Path(__file__).resolve().parent
    files = {p.relative_to(package).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(package.glob("*.py"))}
    return {"sha256": digest(files), "files": files}


def utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = canonical(data) + b"\n"
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def save(world, path, *, source=None):
    world.validate()
    state = world.to_dict()
    envelope = {"schema": SCHEMA, "created_utc": utc(), "python": platform.python_version(),
                "source": source or source_identity(), "state_sha256": digest(state), "state": state}
    write_json(path, envelope)
    return {k: v for k, v in envelope.items() if k != "state"}


def load(path, *, check_source=True):
    from .world import World
    envelope = json.loads(Path(path).read_text(encoding="utf-8"))
    if envelope.get("schema") != SCHEMA:
        raise ValueError("unsupported checkpoint schema")
    if digest(envelope["state"]) != envelope.get("state_sha256"):
        raise ValueError("checkpoint state hash mismatch")
    source = envelope.get("source", {})
    if not isinstance(source.get("files"), dict) or digest(source["files"]) != source.get("sha256"):
        raise ValueError("checkpoint source manifest mismatch")
    if check_source and source != source_identity():
        raise ValueError("checkpoint code differs from installed life8; use the original Git revision for exact continuation")
    world = World.from_dict(envelope["state"])
    world.validate()
    if digest(world.to_dict()) != envelope["state_sha256"]:
        raise ValueError("checkpoint is not an exact state roundtrip")
    return world, envelope
