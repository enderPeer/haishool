"""Build docs/data/records.json for the data browser from the per-round record files.

    python -m haishool.site_data docs/data/records.json 3 \
        1=data/records-r1.jsonl 2=data/records-r2.jsonl 3=data/records-r3.jsonl 3=data/manual.jsonl 1=data/identity.jsonl

Each ``round=path`` is merged in order (later files override earlier ones key by key, as in
``haishool.combine``); a record keeps the round in which it first appeared. ``facts`` counts the
queryable values, so ``type`` and ``aliases`` are not counted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from haishool.combine import merge_into, read


def build(out: Path, current_round: int, inputs: list[tuple[int, Path]]) -> dict:
    values: dict[str, dict[str, str]] = {}
    rounds: dict[str, int] = {}
    for rnd, path in inputs:
        for row in read([path]):
            if merge_into(values, row["obj"], row["values"]):
                rounds[row["obj"]] = rnd
    records = [{"obj": o, "values": v, "round": rounds[o]} for o, v in values.items()]
    facts = sum(1 for r in records for k in r["values"] if k not in ("type", "aliases"))
    data = {"round": current_round, "objects": len(records), "facts": facts, "records": records}
    out.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    return data


if __name__ == "__main__":
    pairs = [(int(a.split("=", 1)[0]), Path(a.split("=", 1)[1])) for a in sys.argv[3:]]
    d = build(Path(sys.argv[1]), int(sys.argv[2]), pairs)
    print(f"{d['objects']} records, {d['facts']} facts -> {sys.argv[1]}")
