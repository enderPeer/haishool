"""Round 4: hops, the relations between things.

Up to round 3 every record stood alone (``turkey. capital ankara. ...``). Round 4 links the
records to each other, so that every thing reaches every other thing in a few hops:

    turkey has_city istanbul birthplace_of ahmet_ertegun founded atlantic_records signed led_zeppelin

The seed (``data/hops-seed-r4.jsonl``) is small and written by hand. It declares nodes
(``{"node", "type"}``), aliases (``{"alias", "of"}``), dropped record links
(``{"drop": "jesus country israel", "why"}``) and edges
(``{"a", "rel", "b", "view", "why"}``). ``view`` says who holds the edge true: ``fact`` for
what is broadly agreed, otherwise the person, people, state or religion whose view it is
(``jerusalem capital_of palestine`` is held by ``palestine``). A disputed relation is not
flattened into one answer; each side is kept with its holder. ``why`` is for people and is
not trained.

Edges the round-3 records already imply between seed nodes (``turkey known_for istanbul``)
are added with ``src records``, unless the seed already holds the same link under a view (the
teacher states contested things as plain facts) or drops it. The build fails if any node cannot reach every other node.

Vectors: every node gets a vector from the graph alone (multi-hop co-occurrence, PPMI, SVD);
every relation and every view gets the mean offset ``vec(b) - vec(a)`` of its edges.

    python -m haishool.relations build --seed data/hops-seed-r4.jsonl \
        --records data/records-r3-all.jsonl --out data
"""

from __future__ import annotations

import argparse
import heapq
import json
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: relation -> inverse; symmetric relations map to themselves. Every relation in the seed must
#: be listed here (as a key or as an inverse).
INVERSE: dict[str, str] = {
    "borders": "borders", "allied_with": "allied_with", "relations_with": "relations_with",
    "related_to": "related_to",
    "coast_on": "coast_of", "part_of": "has_part", "city_in": "has_city", "capital_of": "has_capital",
    "holy_to": "holds_holy", "father_figure_of": "has_father_figure", "prophet_of": "has_prophet",
    "son_of_god_in": "holds_son_of_god", "lived_in": "home_of", "had_capital": "was_capital_of",
    "old_name_from": "gave_old_name_to", "spoke_language_of": "language_spoken_by",
    "state_religion": "state_religion_of", "built": "built_by", "used_by": "used", "ended": "ended_by",
    "led_by": "led", "happened_in": "scene_of", "ruled": "ruled_by", "followed_by": "followed",
    "fought_in": "had_side", "recognised": "recognised_by", "signed": "signed_by",
    "origin_in": "origin_of", "popularised_in": "popularised", "brought_by": "brought",
    "word_from": "gave_word", "culture_in": "has_culture_of", "national_style_of": "has_national_style",
    "drunk_in": "drinks", "origin_claimed_by": "claims_origin_of", "refined_in": "refined",
    "eaten_in": "eats", "grown_in": "grows", "born_in": "birthplace_of", "founded": "founded_by",
    "based_in": "base_of", "honoured": "honoured_by", "headlined_by": "headlined",
    "formed_in": "formed_here", "member_of": "has_member", "plays": "played_by",
    "influenced_by": "influenced", "made": "made_by", "inspired_by": "inspired",
    "travelled_in": "visited_by", "recorded_in": "recording_place_of", "played_with": "played_on",
    "played_in": "hosted",
    # from round-3 record keys
    "known_for": "makes_known", "country": "country_of", "involved": "involved_in",
    "caused_by": "caused", "resulted_in": "result_of", "holder": "held",
}
_BACK = {inv: rel for rel, inv in INVERSE.items()}

#: round-3 record key -> relation it implies when its value names another node
FROM_RECORD_KEY = {
    "known_for": "known_for", "country": "country", "capital": "has_capital", "origin": "origin_in",
    "founder": "founded_by", "maker": "made_by", "people": "involved", "place": "happened_in",
    "related": "related_to", "members": "has_member", "holder": "holder", "cause": "caused_by",
    "result": "resulted_in",
}

_KEY = re.compile(r"^[a-z0-9_]+$")


def inverse(rel: str) -> str:
    if rel in INVERSE:
        return INVERSE[rel]
    if rel in _BACK:
        return _BACK[rel]
    raise KeyError(f"relation {rel!r} has no inverse in INVERSE")


def canonical(a: str, rel: str, b: str) -> tuple[str, str, str]:
    """Store every edge in its forward direction (``has_capital`` -> flipped ``capital_of``)."""
    if rel in INVERSE:
        return a, rel, b
    if rel in _BACK:
        return b, _BACK[rel], a
    raise KeyError(f"relation {rel!r} has no inverse in INVERSE")


@dataclass(frozen=True)
class Edge:
    a: str
    rel: str
    b: str
    view: str = "fact"
    src: str = "seed"
    why: str = ""


@dataclass
class Graph:
    types: dict[str, str]
    edges: list[Edge]

    def neighbours(self) -> dict[str, list[tuple[str, str, str, str]]]:
        """node -> [(relation as read from node, other node, view, src)], both directions."""
        out: dict[str, list[tuple[str, str, str, str]]] = {n: [] for n in self.types}
        for e in self.edges:
            out[e.a].append((e.rel, e.b, e.view, e.src))
            out[e.b].append((inverse(e.rel), e.a, e.view, e.src))
        for n in out:
            out[n].sort(key=lambda t: (t[1], t[0], t[2]))
        return out


def load_seed(path: Path) -> tuple[dict[str, str], dict[str, str], list[Edge], set[tuple[str, str, str]]]:
    types: dict[str, str] = {}
    aliases: dict[str, str] = {}
    edges: list[Edge] = []
    drops: set[tuple[str, str, str]] = set()
    for n, line in enumerate(path.open(encoding="utf-8"), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if "node" in row:
            types[row["node"]] = row["type"]
        elif "alias" in row:
            aliases[row["alias"]] = row["of"]
        elif "drop" in row:
            drops.add(canonical(*row["drop"].split()))
        else:
            a, rel, b = canonical(row["a"], row["rel"], row["b"])
            edges.append(Edge(a, rel, b, row.get("view", "fact"), "seed", row.get("why", "")))
    for e in edges:
        for name in (e.a, e.b, e.view, e.rel):
            if not _KEY.match(name):
                raise ValueError(f"{path}: {name!r} is not a plain key")
        missing = {e.a, e.b} - set(types)
        if missing:
            raise ValueError(f"{path}: edge {e.a} {e.rel} {e.b} names undeclared node(s) {sorted(missing)}")
    return types, aliases, edges, drops


def _mentions(value: str, names: set[str]) -> list[str]:
    """Node keys inside a record value: whole words, or runs of ``_`` parts (``jerusalem_dead_sea``)."""
    found: list[str] = []
    for word in value.split():
        parts = word.split("_")
        for i in range(len(parts)):
            for j in range(len(parts), i, -1):
                cand = "_".join(parts[i:j])
                if cand in names and cand not in found:
                    found.append(cand)
    return found


def edges_from_records(types: dict[str, str], aliases: dict[str, str], records_path: Path) -> list[Edge]:
    names = set(types) | set(aliases)
    out: list[Edge] = []
    for line in records_path.open(encoding="utf-8"):
        row = json.loads(line)
        obj = aliases.get(row["obj"], row["obj"])
        if obj not in types:
            continue
        for key, value in row["values"].items():
            rel = FROM_RECORD_KEY.get(key)
            if not rel:
                continue
            for hit in _mentions(value, names):
                other = aliases.get(hit, hit)
                if other != obj:
                    a, r, b = canonical(obj, rel, other)
                    out.append(Edge(a, r, b, "fact", "records", f"round-3 record of {obj}: {key} {value}"))
    return out


def build_graph(seed: Path, records: Path | None) -> Graph:
    types, aliases, edges, drops = load_seed(seed)
    if records:
        held = {(e.a, e.rel, e.b) for e in edges} | drops
        edges = edges + [e for e in edges_from_records(types, aliases, records) if (e.a, e.rel, e.b) not in held]
    seen: set[tuple[str, str, str, str]] = set()
    unique = []
    for e in edges:
        k = (e.a, e.rel, e.b, e.view)
        if k not in seen and (e.b, inverse(e.rel), e.a, e.view) not in seen:
            seen.add(k)
            unique.append(e)
    return Graph(types, unique)


def components(g: Graph) -> list[set[str]]:
    nb = g.neighbours()
    left, comps = set(g.types), []
    while left:
        start = min(left)
        comp, todo = {start}, [start]
        while todo:
            for _, m, _, _ in nb[todo.pop()]:
                if m not in comp:
                    comp.add(m)
                    todo.append(m)
        comps.append(comp)
        left -= comp
    return comps


#: cost of one hop: an agreed seed fact is cheapest, a link read from the round-3 records costs a
#: little more (the teacher was not checked), a view costs most, so paths only lean on a disputed
#: link when nothing agreed connects the two things
COST = {"fact": 1.0, "records": 1.1, "view": 1.6}


def _cost(view: str, src: str) -> float:
    return COST["view"] if view != "fact" else COST["records"] if src == "records" else COST["fact"]


def shortest_paths(g: Graph) -> dict[tuple[str, str], list[tuple[str, str, str]]]:
    """(x, y) -> [(rel, next node, view), ...] for every ordered pair (Dijkstra over :data:`COST`).

    Ties go to fewer hops, then to the alphabetically first route, so every run gives the same
    paths.
    """
    nb = g.neighbours()
    out = {}
    for src in sorted(g.types):
        best: dict[str, tuple[float, int, tuple]] = {}
        heap = [(0.0, 0, (), src)]
        while heap:
            cost, n_hops, steps, node = heapq.heappop(heap)
            if node in best:
                continue
            best[node] = (cost, n_hops, steps)
            for rel, m, view, esrc in nb[node]:
                if m not in best:
                    heapq.heappush(heap, (round(cost + _cost(view, esrc), 6), n_hops + 1, steps + ((rel, m, view),), m))
        for dst, (_, _, steps) in best.items():
            if dst != src:
                out[(src, dst)] = list(steps)
    return out


def step_text(steps: list[tuple[str, str, str]]) -> str:
    """``capital_of palestine according_to palestine`` for a step that only one side holds."""
    return " ".join(f"{r} {m}" + (f" according_to {v}" if v != "fact" else "") for r, m, v in steps)


def embed(g: Graph, dim: int = 16, hops: int = 4, decay: float = 0.6) -> tuple[list[str], np.ndarray]:
    """Node vectors from the graph alone: weighted k-hop walk co-occurrence -> PPMI -> SVD."""
    nodes = sorted(g.types)
    ix = {n: i for i, n in enumerate(nodes)}
    A = np.zeros((len(nodes), len(nodes)))
    for e in g.edges:
        w = 1.0 if e.view == "fact" else 0.5  # a view links two things more weakly than a fact
        A[ix[e.a], ix[e.b]] += w
        A[ix[e.b], ix[e.a]] += w
    P = A / A.sum(1, keepdims=True)
    M, Pk = np.zeros_like(P), np.eye(len(nodes))
    for k in range(1, hops + 1):
        Pk = Pk @ P
        M += decay ** (k - 1) * Pk
    joint = M / M.sum()
    with np.errstate(divide="ignore"):
        pmi = np.log(joint / (joint.sum(1, keepdims=True) * joint.sum(0, keepdims=True)))
    ppmi = np.maximum(pmi, 0)
    U, S, _ = np.linalg.svd(ppmi)
    V = U[:, :dim] * np.sqrt(S[:dim])
    V *= np.sign(V[np.abs(V).argmax(0), range(V.shape[1])])  # fixed sign per dimension
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    return nodes, V


def offsets(g: Graph, nodes: list[str], V: np.ndarray, by: str) -> dict[str, list[float]]:
    ix = {n: i for i, n in enumerate(nodes)}
    groups: dict[str, list[np.ndarray]] = defaultdict(list)
    for e in g.edges:
        groups[getattr(e, by)].append(V[ix[e.b]] - V[ix[e.a]])
    return {k: np.round(np.mean(v, axis=0), 4).tolist() for k, v in sorted(groups.items())}


def training_lines(g: Graph, paths: dict[tuple[str, str], list[tuple[str, str]]]) -> list[str]:
    """Dense lines in the round-3 alphabet: relations per node, relation queries, hop paths.

    * ``turkey. borders greece syria. coast_on mediterranean_sea. ...``
    * ``q turkey borders. a greece syria.``
    * ``q islam has_prophet. a jesus moses muhammad according_to islam.`` (one side holds it)
    * ``q jerusalem capital_of. a israel palestine contested.`` (sides differ)
    * ``q jerusalem capital_of according_to palestine. a palestine.``
    * ``q turkey hop led_zeppelin. a turkey has_city istanbul ... signed led_zeppelin.``
    * ``q turkey hops led_zeppelin. a 4.``
    """
    by_rel: dict[tuple[str, str], dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for n, links in g.neighbours().items():
        for rel, m, view, _ in links:
            by_rel[(n, rel)][view].append(m)
    lines: list[str] = []
    per_node: dict[str, list[str]] = defaultdict(list)
    for (n, rel), views in sorted(by_rel.items()):
        all_targets = list(dict.fromkeys(m for ms in views.values() for m in sorted(ms)))
        per_node[n].append(f"{rel} {' '.join(all_targets)}.")
        held = sorted(views)
        tail = "" if held == ["fact"] else f" according_to {held[0]}" if len(held) == 1 else " contested"
        lines.append(f"q {n} {rel}. a {' '.join(all_targets)}{tail}.")
        if len(held) > 1:
            for view, ms in sorted(views.items()):
                lines.append(f"q {n} {rel} according_to {view}. a {' '.join(sorted(ms))}.")
    for n in sorted(per_node):
        lines.append(f"{n}. " + " ".join(per_node[n]))
    for (x, y), steps in sorted(paths.items()):
        lines.append(f"q {x} hop {y}. a {x} {step_text(steps)}.")
        lines.append(f"q {x} hops {y}. a {len(steps)}.")
    return lines


def build(seed: Path, records: Path | None, out: Path, dim: int = 16) -> dict:
    g = build_graph(seed, records)
    comps = components(g)
    if len(comps) != 1:
        small = sorted(comps, key=len)[:-1]
        raise SystemExit(f"not fully connected: {len(comps)} parts; cut off: {[sorted(c) for c in small]}")
    paths = shortest_paths(g)
    nodes, V = embed(g, dim=dim)
    sims = V @ V.T
    ix = {n: i for i, n in enumerate(nodes)}
    lengths = [len(s) for s in paths.values()]
    out.mkdir(parents=True, exist_ok=True)

    with (out / "hops-edges-r4.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for e in g.edges:
            f.write(json.dumps({"a": e.a, "rel": e.rel, "inverse": inverse(e.rel), "b": e.b,
                                "view": e.view, "src": e.src, "why": e.why}) + "\n")
    with (out / "hops-paths-r4.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for (x, y), steps in sorted(paths.items()):
            f.write(json.dumps({"from": x, "to": y, "hops": len(steps),
                                "path": [{"rel": r, "to": m, "view": v} for r, m, v in steps],
                                "cosine": round(float(sims[ix[x], ix[y]]), 4)}) + "\n")
    vectors = {
        "dim": dim,
        "method": "graph only: 4-hop random-walk co-occurrence (decay 0.6, views weigh 0.5), PPMI, SVD, unit length",
        "nodes": {n: {"type": g.types[n], "vec": np.round(V[i], 4).tolist(),
                      "nearest": [nodes[j] for j in np.argsort(-sims[i]) if j != i][:5]}
                  for i, n in enumerate(nodes)},
        "relations": offsets(g, nodes, V, "rel"),
        "views": offsets(g, nodes, V, "view"),
    }
    (out / "hops-vectors-r4.json").write_text(json.dumps(vectors, indent=1), encoding="utf-8")
    lines = training_lines(g, paths)
    (out / "hops-train-r4.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    views = sorted({e.view for e in g.edges} - {"fact"})
    report = {"nodes": len(g.types), "edges": len(g.edges),
              "edges_from_records": sum(e.src == "records" for e in g.edges),
              "edges_with_a_view": sum(e.view != "fact" for e in g.edges), "views": views,
              "relations": len({e.rel for e in g.edges}), "connected": True,
              "pairs": len(paths), "paths_using_a_view": sum(any(v != "fact" for _, _, v in s) for s in paths.values()),
              "max_hops": max(lengths),
              "mean_hops": round(sum(lengths) / len(lengths), 2),
              "hops_histogram": {h: lengths.count(h) for h in sorted(set(lengths))},
              "training_lines": len(lines)}
    (out / "hops-report-r4.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--seed", type=Path, required=True)
    b.add_argument("--records", type=Path)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--dim", type=int, default=16)
    p = sub.add_parser("path")
    p.add_argument("--seed", type=Path, required=True)
    p.add_argument("--records", type=Path)
    p.add_argument("a")
    p.add_argument("b")
    args = ap.parse_args()
    if args.cmd == "build":
        print(json.dumps(build(args.seed, args.records, args.out, args.dim), indent=1))
    else:
        g = build_graph(args.seed, args.records)
        steps = shortest_paths(g).get((args.a, args.b))
        print(args.a, step_text(steps) if steps else "(no path)")


if __name__ == "__main__":
    main()
