"""Merge record files into one training file; later files override earlier ones key by key.

    python -m haishool.combine data/all.jsonl data/records-r1.jsonl data/records-r2.jsonl \
        data/records-r3.jsonl data/manual.jsonl data/identity.jsonl

A typed record (``type country``) replaces an earlier record of the same name with a different
type instead of merging into it: round 1 knew ``brazil`` only as the nut.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable
from pathlib import Path

from haishool.schema import Record


def merge_into(merged: dict[str, dict[str, str]], obj: str, values: dict[str, str]) -> bool:
    """Merge one record; return True when it replaced (or created) the entry rather than extending it."""
    old = merged.get(obj)
    if old is not None and "type" in values and old.get("type") not in (values["type"], "self"):
        old = None
    merged[obj] = {**(old or {}), **values}
    return old is None


def merge(rows: Iterable[dict]) -> dict[str, dict[str, str]]:
    merged: dict[str, dict[str, str]] = {}
    for row in rows:
        merge_into(merged, row["obj"], row["values"])
    return merged


def read(paths: Iterable[Path]) -> Iterable[dict]:
    for path in paths:
        with path.open(encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)


def combine(out: Path, inputs: list[Path]) -> int:
    merged = merge(read(inputs))
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for obj, values in merged.items():
            f.write(json.dumps({"obj": obj, "values": values, "line": Record(obj, values).line()}) + "\n")
    return len(merged)


if __name__ == "__main__":
    n = combine(Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]])
    print(f"{n} records -> {sys.argv[1]}")
