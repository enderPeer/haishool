"""Write the data of the world7 explorer page: 36 worlds told era by era, and a census of rungs.

    python scripts/build_world7_page.py --out path/to/world7-page.json [--worlds 36] [--census 200] [--workers 12]

For each of the worlds 1 to ``--worlds`` (``haishool.evo.world7.rollout(seed)``, the world its seed
names) the JSON holds its parameters, every era with its metrics, the clock (age at the start of
each era, its duration and end, the star's end and the end of the climb, in years since the Big
Bang), the highest rung, the outcome and the rung reached by now, the ``why`` line (the links of
the chain that held), and for a world that speaks: its lexicon, its grammar and three example
utterances with their meanings. Every printed answer is the one world7's own gate accepts.

Then the distribution of the highest rung (and of the outcome and the rung by now) over the
worlds 1 to ``--census``. The output depends only on the code: no time, no machine, so two runs
write the same bytes. It does not touch ``docs/``: the page that reads it is built later.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from haishool.evo import world7  # noqa: E402
from haishool.truth import num, split_line  # noqa: E402

EXAMPLES = 3


def _plain(x):
    if hasattr(x, "item"):
        return x.item()
    raise TypeError(f"not JSON: {x!r}")


def _clean(obj):
    return json.loads(json.dumps(obj, default=_plain))


def _judged(prompt: str, answer: str) -> str:
    if not world7.check(prompt, answer).ok:
        raise ValueError(f"world7 rejects its own answer: {prompt} / {answer}")
    return answer


def language_of(r) -> dict | None:
    """The lexicon, grammar and example utterances of a speaking world, read from the lines
    world7 trains on (so the page shows what the model is taught), or ``None``."""
    lines = world7.language_lines(r)
    if not lines:
        return None
    head = f"world7 seed {num(r.seed)} "
    lexicon: dict[str, str] = {}
    grammar: dict[str, str] = {}
    phrases: dict[str, str] = {}
    says, words = [], []
    for ln in lines:
        if ln.kind == "record":
            kind, _, rest = ln.prompt[len(head):].partition(". ")
            fields = [f.strip() for f in rest.rstrip(".").split(".") if f.strip()]
            if kind == "lexicon":
                for f in fields:
                    name, _, form = f.partition(" ")
                    lexicon[name] = form
            elif kind == "grammar":
                for f in fields:
                    name, _, value = f.partition(" ")
                    grammar[name] = value
            elif kind == "phrases":
                for f in fields:
                    a, b, form = f.split(" ", 2)
                    phrases[f"{a} {b}"] = form
            continue
        ask = ln.prompt[len(head):]
        if ln.answer == "none":  # a meaning the population has no word for
            continue
        if ask.startswith("say "):
            says.append({"utterance": _judged(ln.prompt, ln.answer), "meaning": ask[4:]})
        elif ask.startswith("word "):
            words.append({"utterance": _judged(ln.prompt, ln.answer), "meaning": ask[5:]})
    examples = (says or words)[:EXAMPLES]
    for ex in examples:  # the other way round: the utterance means the pair (or the word)
        prompt = f"world7 seed {num(r.seed)} meaning {ex['utterance']}"
        verdict = world7.check(prompt, ex["meaning"])
        ex["understood"] = bool(verdict.ok)
        if not verdict.ok:
            ex["heard_as"] = verdict.expected
    return {"lexicon": lexicon, "grammar": grammar, "phrases": phrases, "examples": examples,
            "sentence": r.summary.get("sentence"), "sentence_meaning": r.summary.get("meaning")}


def world_entry(seed: int) -> dict:
    r = world7.rollout(seed)
    verdict = world7.conserved(r)
    if not verdict.ok:
        raise ValueError(f"world7 seed {seed}: {verdict}")
    eras, clock = [], []
    star_end = None
    for step in r.steps:
        metrics = {k: v for k, v in step.items() if k != "era"}
        eras.append({"era": step["era"], "metrics": metrics})
        duration = step.get("duration", 0)
        clock.append({"era": step["era"], "age": step["age"], "duration": duration, "end": step["age"] + duration})
        if step["era"] == "star":
            star_end = step.get("star_end")
    why = world7.why_line(r)
    rung = r.summary["rung"]
    entry = {
        "seed": seed,
        "params": dict(r.params),
        "eras": eras,
        "clock": {"now": world7.NOW, "first_stars": world7.FIRST_STARS, "star_end": star_end,
                  "age_at_end": r.summary["age_at_end"], "eras": clock},
        "rung": rung,
        "rung_index": world7.LADDER.index(rung),
        "outcome": r.summary["outcome"],
        "by_now": r.summary["by_now"],
        "why": {"line": why.text, "links": why.answer.split() if why.answer != "nothing" else []},
        "summary": dict(r.summary),
        "language": language_of(r),
    }
    _judged(why.prompt, why.answer)
    return _clean(entry)


def census_entry(seed: int) -> dict:
    s = world7.rollout(seed).summary
    return {"seed": seed, "rung": s["rung"], "outcome": s["outcome"], "by_now": s["by_now"]}


def build(worlds: int = 36, census: int = 200, workers: int = 12) -> dict:
    with ProcessPoolExecutor(max_workers=workers) as pool:
        detailed = list(pool.map(world_entry, range(1, worlds + 1), chunksize=1))
        rest = list(pool.map(census_entry, range(worlds + 1, census + 1), chunksize=1))
    rows = [{"seed": w["seed"], "rung": w["rung"], "outcome": w["outcome"], "by_now": w["by_now"]}
            for w in detailed[:census]] + rest

    def tally(field: str, order: tuple[str, ...]) -> list[dict]:
        counts = Counter(row[field] for row in rows)
        return [{"word": w, "worlds": counts.get(w, 0)} for w in order]

    return {
        "sim": world7.SIM,
        "description": "toy worlds of round 7: each era is one level of the chain on one clock; "
                       "numbers are the simulation's, not measurements of the real universe",
        "ladder": list(world7.LADDER),
        "links": list(world7.LINKS),
        "spans": dict(world7.SPAN),
        "worlds": detailed,
        "census": {
            "seeds": [1, census],
            "rung": tally("rung", tuple(world7.LADDER)),
            "outcome": tally("outcome", tuple(world7.OUTCOMES)),
            "by_now": tally("by_now", tuple(world7.LADDER)),
            "speaking_of_detailed": sum(1 for w in detailed if w["language"]),
        },
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--worlds", type=int, default=36)
    ap.add_argument("--census", type=int, default=200)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args(argv)
    if args.worlds < 1 or args.census < args.worlds:
        raise SystemExit("--worlds must be at least 1 and --census at least --worlds")
    data = build(args.worlds, args.census, args.workers)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8", newline="\n")
    tmp.replace(args.out)
    print(json.dumps({"out": str(args.out), "worlds": len(data["worlds"]), "census": data["census"]["rung"],
                      "speaking": data["census"]["speaking_of_detailed"]}))


if __name__ == "__main__":
    main()
