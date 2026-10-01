"""Round 3 teacher: the knowledge map (people, countries, religions, history, concepts, works, lists).

    python3 -m haishool.teacher3 entities --out data/entities-r3.txt --exclude data/known.txt
    python3 -m haishool.teacher3 records --entities data/entities-r3.txt --out data/records-r3.jsonl

Categories come from what people ask a chat model first (who are you, money, markets and their
history, politics and geography, religions, history books, biology, software, television, games,
pop culture, music, film, systems, game theory, maths, physics, coding). Each category has an entity
type from ``schema.TYPES``; the teacher writes that type's keys plus ``aliases`` (other names,
including German ones) that only the parser uses. Same normalisation and resumability as round 1.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from haishool.schema import TYPES, Record, normalise_value, object_key
from haishool.teacher import ask

CATEGORIES: list[tuple[str, str, int]] = [
    ("famous scientists", "person", 40), ("important world leaders in history", "person", 40),
    ("chancellors of germany since 1949", "person", 12), ("famous inventors", "person", 30),
    ("famous musicians and bands", "person", 40), ("famous film actors and directors", "person", 40),
    ("famous writers and poets", "person", 30), ("famous philosophers", "person", 30),
    ("famous entrepreneurs and business people", "person", 30), ("founders of world religions and religious figures", "person", 20),
    ("famous painters and artists", "person", 30), ("famous athletes", "person", 30),
    ("countries of the world, the most important ones", "country", 80),
    ("world religions and belief systems", "religion", 20),
    ("major events of world history", "event", 50), ("major events of german history", "event", 30),
    ("famous wars and revolutions", "event", 30), ("great scientific discoveries and inventions", "event", 30),
    ("stock market crashes, financial crises and famous market events", "event", 25),
    ("milestones of space exploration", "event", 20),
    ("basic biology concepts", "concept", 40), ("basic physics concepts", "concept", 40),
    ("basic mathematics concepts", "concept", 40), ("game theory concepts", "concept", 25),
    ("systems theory and cybernetics concepts", "concept", 25), ("computer science and coding concepts", "concept", 40),
    ("economics and money concepts", "concept", 40), ("stock market and investing concepts", "concept", 30),
    ("basic chemistry concepts", "concept", 30), ("psychology concepts", "concept", 25),
    ("philosophy concepts", "concept", 25), ("internet and technology concepts", "concept", 30),
    ("most used software and apps", "work", 40), ("programming languages", "work", 30),
    ("famous video games", "work", 40), ("famous movies", "work", 40), ("famous tv series", "work", 30),
    ("famous songs and albums", "work", 30), ("most influential books in history", "work", 30),
    ("operating systems and social media platforms", "work", 25),
    ("common general knowledge lists, for example the continents, the planets of the solar system, "
     "the major world religions, the german federal states, the oceans, the g7 countries", "list", 25),
]


#: core entries of the knowledge map the category lists missed, and names a plain object had taken
#: (round 1 knew "brazil" only as the nut); ``combine`` lets the typed record replace the object.
EXTRA: list[tuple[str, str, str]] = [
    (n, "concept", "core science, maths and systems ideas") for n in (
        "game_theory", "systems_theory", "black_hole", "pythagorean_theorem", "gravity", "evolution",
        "theory_of_relativity", "quantum_mechanics", "big_bang", "atom", "cell", "prime_number", "algorithm",
        "artificial_intelligence", "machine_learning", "neural_network", "internet", "electricity", "energy",
        "democracy", "capitalism", "communism", "climate_change", "vaccine", "virus", "bacteria", "calculus",
        "probability", "statistics", "logic", "feedback_loop", "supply_and_demand", "compound_interest",
        "bitcoin", "blockchain", "stock_exchange", "bull_market", "bear_market", "dividend", "bond", "interest_rate")
] + [
    (n, "work", "famous software, games, films and music") for n in (
        "minecraft", "tetris", "super_mario_bros", "fortnite", "grand_theft_auto_v", "pokemon", "the_legend_of_zelda",
        "star_wars", "the_godfather", "titanic", "harry_potter", "the_lord_of_the_rings", "game_of_thrones",
        "the_simpsons", "friends", "linux", "google_search", "excel", "photoshop", "iphone", "chatgpt", "wikipedia",
        "facebook", "instagram", "tiktok", "spotify", "netflix", "thriller", "bohemian_rhapsody", "the_bible", "the_quran")
] + [
    (n, "country", "countries of the world") for n in (
        "brazil", "turkey", "china", "chile", "jordan", "georgia", "guinea", "chad", "japan", "india", "france",
        "united_states", "united_kingdom", "russia", "italy", "spain", "canada", "australia", "mexico", "egypt")
] + [
    (n, "event", "major events of history") for n in (
        "world_war_i", "world_war_ii", "fall_of_the_berlin_wall", "german_reunification", "moon_landing",
        "industrial_revolution", "renaissance", "reformation", "cold_war", "great_depression", "dot_com_bubble",
        "september_11_attacks", "covid_19_pandemic", "french_revolution", "american_revolution", "holocaust")
]


def extra(entities_file: Path, have: set[str]) -> int:
    """Re-key the entity file with the current ``object_key`` and append EXTRA names not yet present."""
    rows = [json.loads(line) for line in entities_file.open(encoding="utf-8")]
    for r in rows:
        r["obj"] = object_key(r["obj"].replace("_", " "))
    present = {r["obj"] for r in rows}
    added = 0
    for name, etype, cat in EXTRA:
        if name not in present and name not in have:
            rows.append({"obj": name, "type": etype, "category": cat})
            present.add(name)
            added += 1
    entities_file.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return added


def entities(host: str, out: Path, exclude: set[str]) -> None:
    rows: list[dict[str, str]] = []
    seen = set(exclude)
    for i, (cat, etype, n) in enumerate(CATEGORIES):
        prompt = (f"List {n} different well-known {cat}. Use their usual English names, short. "
                  f'Reply as JSON: {{"items": ["...", "..."]}}')
        try:
            items = json.loads(ask(host, prompt, seed=3000 + i, num_predict=1500)).get("items", [])
        except (ValueError, OSError) as exc:
            print(f"category {cat}: {exc!r}", file=sys.stderr, flush=True)
            continue
        added = 0
        for item in items:
            key = object_key(str(item))
            if key and len(key) <= 40 and key not in seen:
                seen.add(key)
                rows.append({"obj": key, "type": etype, "category": cat})
                added += 1
        print(f"{cat}: {len(items)} items, {added} new, total {len(rows)}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def records(host: str, entities_file: Path, out: Path, batch: int) -> None:
    todo = [json.loads(line) for line in entities_file.open(encoding="utf-8")]
    done: set[str] = set()
    if out.exists():
        done = {json.loads(line)["obj"] for line in out.open(encoding="utf-8")}
    todo = [t for t in todo if t["obj"] not in done]
    by_type: dict[str, list[dict[str, str]]] = {}
    for t in todo:
        by_type.setdefault(t["type"], []).append(t)
    kept = dropped = processed = 0
    start = time.monotonic()
    with out.open("a", encoding="utf-8") as f:
        for etype, group_all in by_type.items():
            keys = TYPES[etype]
            spec = "\n".join(f'- "{k}": {v}' for k, v in keys.items())
            for i in range(0, len(group_all), batch):
                group = group_all[i:i + batch]
                listing = "; ".join(f'{g["obj"].replace("_", " ")} ({g["category"]})' for g in group)
                prompt = (
                    f"For each {etype} below give short, true, widely known facts. Return these keys; each "
                    "value is a list of 1 to 4 short items separated by spaces, an item of several words "
                    "joined with underscores (for example theory_of_relativity). Write years and numbers "
                    "with digits. Plain lowercase English, no sentences. If you are not sure, write unknown.\n"
                    f"{spec}\n"
                    '- "aliases": other common names for it, including the German name, joined with underscores\n'
                    f'Reply as JSON: {{"items": [{{"name": "...", ...}}]}}\nItems: {listing}')
                try:
                    reply = json.loads(ask(host, prompt, seed=9000 + processed, num_predict=1400)).get("items", [])
                except (ValueError, OSError) as exc:
                    print(f"batch {etype} {i}: {exc!r}", file=sys.stderr, flush=True)
                    continue
                reply = [o for o in reply if isinstance(o, dict)]
                by_name = {object_key(str(o.get("name", ""))): o for o in reply}
                for n, g in enumerate(group):
                    o = by_name.get(g["obj"])
                    if o is None and len(reply) == len(group):
                        o = reply[n]  # same order, name spelled differently (accents, initials)
                    processed += 1
                    if not o:
                        dropped += 1
                        continue
                    values = {"type": etype}
                    for k in keys:
                        v = normalise_value(str(o.get(k, "")))
                        if v and v != "unknown":
                            values[k] = v
                    aliases = normalise_value(str(o.get("aliases", "")), max_words=6)
                    if aliases and aliases != "unknown":
                        values["aliases"] = aliases
                    if len(values) < 4:
                        dropped += 1
                        continue
                    rec = Record(g["obj"], values)
                    f.write(json.dumps({"obj": g["obj"], "values": values, "line": rec.line(),
                                        "category": g["category"]}) + "\n")
                    kept += 1
                f.flush()
                rate = processed / max(time.monotonic() - start, 1e-9)
                print(f"{processed}/{len(todo)} kept {kept} dropped {dropped} {rate:.2f} items/s", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("entities")
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--exclude", type=Path, default=None)
    b = sub.add_parser("records")
    b.add_argument("--entities", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--batch", type=int, default=5)
    c = sub.add_parser("extra")
    c.add_argument("--entities", type=Path, required=True)
    c.add_argument("--have", type=Path, required=True, help="records file whose typed records count as present")
    for p in (a, b):
        p.add_argument("--host", default="http://127.0.0.1:11434")
    args = ap.parse_args()
    if args.cmd == "extra":
        have = set()
        for line in args.have.open(encoding="utf-8"):
            row = json.loads(line)
            if "type" in row["values"]:
                have.add(row["obj"])
        print(f"{extra(args.entities, have)} extra entities added")
    elif args.cmd == "entities":
        exclude = set()
        if args.exclude and args.exclude.exists():
            exclude = {n.strip() for n in args.exclude.read_text(encoding="utf-8").splitlines() if n.strip()}
        entities(args.host, args.out, exclude)
    else:
        records(args.host, args.entities, args.out, args.batch)


if __name__ == "__main__":
    main()
