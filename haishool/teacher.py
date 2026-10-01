"""Generate dense object records with a local teacher model (ollama), standard library only.

    python3 -m haishool.teacher objects --out data/objects.txt --per-category 40
    python3 -m haishool.teacher records --objects data/objects.txt --out data/records.jsonl

Step 1 asks for everyday object names per category. Step 2 asks, for batches of objects, for the
nine attributes of ``schema.ATTRIBUTES`` as JSON; every value is normalised to the dense alphabet
and records with fewer than six attributes are dropped. Resumable: objects already in the output
are skipped. The teacher writes the facts; the student model only ever sees the dense records.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

from haishool.schema import ATTRIBUTES, ORDER, Record, normalise_value, object_key

MODEL = "qwen2.5:14b-instruct-q4_K_M"
CATEGORIES = [
    "kitchen items", "furniture", "tools", "clothing", "fruits", "vegetables", "animals",
    "vehicles", "toys", "school supplies", "electronic devices", "musical instruments",
    "sports equipment", "plants and trees", "body parts", "buildings and places",
    "bathroom items", "office items", "food and dishes", "drinks", "weather and nature things",
    "materials and substances", "containers", "jewelry and accessories", "garden items",
    "insects and small animals", "birds", "sea animals", "household appliances", "shapes and signs",
]


def ask(host: str, prompt: str, seed: int, num_predict: int = 900) -> str:
    body = json.dumps({"model": MODEL, "prompt": prompt, "stream": False, "format": "json",
                       "options": {"seed": seed, "temperature": 0.4, "num_predict": num_predict,
                                   "num_ctx": 4096}}).encode()
    req = urllib.request.Request(f"{host}/api/generate", data=body,
                                 headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:  # noqa: S310 (LAN teacher)
        return str(json.load(r).get("response", ""))


ROUND2 = [
    "hand tools", "power tools", "construction materials", "farm animals", "wild animals",
    "fish", "reptiles and amphibians", "flowers", "trees", "rocks and minerals",
    "space objects", "camping gear", "baby items", "pet supplies", "art supplies",
    "cleaning supplies", "baked goods", "spices and herbs", "nuts and seeds", "desserts",
    "sauces and spreads", "dishes and tableware", "lights and lamps", "computer parts",
    "car parts", "bicycle parts", "boats and ships", "aircraft", "hats and shoes", "fabrics",
    "small kitchen appliances", "board games and puzzles", "playground equipment",
    "exotic fruits", "root vegetables", "medical items", "farm machines", "weather instruments",
    "musical accessories", "party items",
]


def objects(host: str, out: Path, per_category: int, categories: list[str] | None = None,
            exclude: set[str] | None = None) -> None:
    names: list[str] = []
    exclude = exclude or set()
    for i, cat in enumerate(categories or CATEGORIES):
        prompt = (f"List {per_category} different common, concrete everyday {cat} that a child "
                  f"would know. Use simple singular English names of one or two words. "
                  f'Reply as JSON: {{"items": ["...", "..."]}}')
        try:
            items = json.loads(ask(host, prompt, seed=1000 + i)).get("items", [])
        except (ValueError, OSError) as exc:
            print(f"category {cat}: {exc!r}", file=sys.stderr)
            continue
        for item in items:
            key = object_key(str(item))
            if key and len(key) <= 24 and key not in names and key not in exclude:
                names.append(key)
        print(f"{cat}: {len(items)} -> total {len(names)}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(names) + "\n", encoding="utf-8")


ATTR_TEXT = "\n".join(f'- "{k}": {v}' for k, v in ATTRIBUTES.items())


def records(host: str, objects_file: Path, out: Path, batch: int) -> None:
    names = [n.strip() for n in objects_file.read_text(encoding="utf-8").splitlines() if n.strip()]
    done: set[str] = set()
    if out.exists():
        done = {json.loads(line)["obj"] for line in out.open(encoding="utf-8")}
    todo = [n for n in names if n not in done]
    start = time.monotonic()
    kept = dropped = 0
    with out.open("a", encoding="utf-8") as f:
        for i in range(0, len(todo), batch):
            group = todo[i:i + batch]
            listing = ", ".join(n.replace("_", " ") for n in group)
            prompt = (
                "For each object below give short, true, typical facts. For every object return "
                "these keys. Each value is a list of 1 to 4 short items separated by spaces; an "
                "item of several words is joined with underscores (for example fighting_fires "
                "carrying_water, or kitchen_utensil). Plain lowercase English, no sentences, no "
                f"numbers:\n{ATTR_TEXT}\n"
                f'Reply as JSON: {{"objects": [{{"name": "...", "kind": "...", ...}}]}}\n'
                f"Objects: {listing}")
            try:
                reply = json.loads(ask(host, prompt, seed=7 + i)).get("objects", [])
            except (ValueError, OSError) as exc:
                print(f"batch {i}: {exc!r}", file=sys.stderr, flush=True)
                continue
            by_name = {object_key(str(o.get("name", ""))): o for o in reply if isinstance(o, dict)}
            for name in group:
                o = by_name.get(name)
                if not o:
                    dropped += 1
                    continue
                values = {k: normalise_value(str(o.get(k, ""))) for k in ORDER}
                if values.get("alive") not in ("yes", "no"):
                    values["alive"] = "yes" if "yes" in values.get("alive", "") else "no"
                values = {k: v for k, v in values.items() if v}
                if len(values) < 6:
                    dropped += 1
                    continue
                rec = Record(name, values)
                f.write(json.dumps({"obj": name, "values": values, "line": rec.line()}) + "\n")
                kept += 1
            f.flush()
            rate = (i + len(group)) / max(time.monotonic() - start, 1e-9)
            print(f"{i + len(group)}/{len(todo)} kept {kept} dropped {dropped} {rate:.2f} obj/s",
                  flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("objects")
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--per-category", type=int, default=40)
    a.add_argument("--round", type=int, default=1, help="1: base categories, 2: second round")
    a.add_argument("--exclude", type=Path, default=None, help="object list to skip")
    b = sub.add_parser("records")
    b.add_argument("--objects", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--batch", type=int, default=6)
    for p in (a, b):
        p.add_argument("--host", default="http://127.0.0.1:11434")
    args = ap.parse_args()
    if args.cmd == "objects":
        exclude = set()
        if args.exclude and args.exclude.exists():
            exclude = {n.strip() for n in args.exclude.read_text(encoding="utf-8").splitlines() if n.strip()}
        objects(args.host, args.out, args.per_category, ROUND2 if args.round == 2 else None, exclude)
    else:
        records(args.host, args.objects, args.out, args.batch)


if __name__ == "__main__":
    main()
