"""Round 4: hops, the relations between things.

Up to round 3 every record stood alone (``turkey. capital ankara. ...``). Round 4 links the
records to each other, so that every thing reaches every other thing in a few hops:

    turkey has_city istanbul birthplace_of ahmet_ertegun founded atlantic_records signed led_zeppelin

Four sources of links, cheapest to most expensive when a path is chosen:

1. **Seed** (``data/hops-seed-r4.jsonl``), small and written by hand. It declares nodes
   (``{"node", "type"}``), aliases (``{"alias", "of"}``), dropped record links
   (``{"drop": "jesus country israel", "why"}``) and edges (``{"a", "rel", "b", "view", "why"}``).
   ``view`` says who holds the edge true: ``fact`` for what is broadly agreed, otherwise the
   person, people, state or religion whose view it is (``jerusalem capital_of palestine`` is held
   by ``palestine``). A disputed relation is not flattened into one answer; each side is kept with
   its holder. ``why`` is for people and is not trained.
2. **Records**: a round-3 value that names another record (``turkey known_for istanbul``,
   ``spoon made metal``). Spelling slips of the teacher are resolved to the nearest record name
   (``led_zepellin`` -> ``led_zeppelin``). The seed wins: a link it holds under a view, or
   drops, is not added again as a plain fact (the teacher states contested things as facts).
3. **Rules**, derived from 1 and 2 and therefore checkable: ``located_in`` (a city in a city in a
   country), ``shares_food`` (two countries with the same dish), ``bandmate``.
4. **Hubs**: shared values become nodes (``europe``, ``euro``, ``1970s``, ``kitchen``), plus one
   node per kind and per type, so that nothing is left on its own.

Every node must reach every other node; the build fails otherwise.

Vectors: every node gets a vector from the graph alone (multi-hop co-occurrence, PPMI, SVD);
every relation and every view gets the mean offset ``vec(b) - vec(a)`` of its edges.

    python -m haishool.relations build --seed data/hops-seed-r4.jsonl \
        --records data/records-r3-all.jsonl --out data/hops-v4b
"""

from __future__ import annotations

import argparse
import heapq
import json
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: relation -> inverse; symmetric relations map to themselves. Every relation must be listed
#: here (as a key or as an inverse).
INVERSE: dict[str, str] = {
    "borders": "borders", "allied_with": "allied_with", "relations_with": "relations_with",
    "related_to": "related_to", "shares_food": "shares_food", "bandmate": "bandmate",
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
    "made_of": "material_of", "found_in": "place_of", "has_component": "component_of",
    # rules
    "located_in": "location_of",
    # hubs
    "is_a": "has_instance", "of_type": "type_of", "continent": "continent_of", "currency": "currency_of",
    "language": "language_of", "era": "era_of", "genre": "genre_of", "field": "field_of", "role": "role_of",
}
_BACK = {inv: rel for rel, inv in INVERSE.items()}

#: round-3 record key -> relation it implies when its value names another record
FROM_RECORD_KEY = {
    "known_for": "known_for", "country": "country", "capital": "has_capital", "origin": "origin_in",
    "founder": "founded_by", "maker": "made_by", "people": "involved", "place": "found_in",
    "related": "related_to", "members": "has_member", "holder": "holder", "cause": "caused_by",
    "result": "resulted_in", "made": "made_of", "parts": "has_component",
}
#: round-3 record key -> relation to a shared-value hub node (``europe``, ``1970s``, ``kitchen``)
HUB_KEY = {"kind": "is_a", "continent": "continent", "currency": "currency", "language": "language",
           "era": "era", "genre": "genre", "field": "field", "role": "role", "made": "made_of",
           "place": "found_in", "country": "country"}
#: keys whose values are lists of things, often joined with ``_`` by the teacher
LIST_KEYS = ("made", "parts", "place")
#: event records use ``place`` for where it happened, not where a thing is kept
EVENT_PLACE = "happened_in"

#: cost of one hop when a path is chosen: agreed links are cheap, hubs and views expensive, so a
#: path only leans on "both are in Europe" or on a disputed link when nothing better connects
COST = {"seed": 1.0, "records": 1.1, "rule": 1.2, "view": 1.6, "hub": 1.8, "kind": 2.2, "type": 3.0}
HUB_SRC = ("hub", "kind", "type")

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

    @property
    def cost(self) -> float:
        return COST["view"] if self.view != "fact" and self.src not in HUB_SRC else COST[self.src]


@dataclass
class Graph:
    types: dict[str, str]
    edges: list[Edge]
    seed_nodes: frozenset[str] = frozenset()

    def neighbours(self) -> dict[str, list[tuple[str, str, str, float]]]:
        """node -> [(relation as read from node, other node, view, cost)], both directions."""
        if getattr(self, "_nb", None) is None:
            out: dict[str, list[tuple[str, str, str, float]]] = {n: [] for n in self.types}
            for e in self.edges:
                out[e.a].append((e.rel, e.b, e.view, e.cost))
                out[e.b].append((inverse(e.rel), e.a, e.view, e.cost))
            for n in out:
                out[n].sort(key=lambda t: (t[1], t[0], t[2]))
            self._nb = out
        return self._nb

    def hubs(self) -> set[str]:
        return {n for n, t in self.types.items() if t in ("value", "kind", "type")}


def load_seed(path: Path) -> tuple[dict[str, str], dict[str, str], list[Edge], set[tuple[str, str, str]]]:
    types: dict[str, str] = {}
    aliases: dict[str, str] = {}
    edges: list[Edge] = []
    drops: set[tuple[str, str, str]] = set()
    for line in path.open(encoding="utf-8"):
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


def _doubled(short: str, long: str) -> bool:
    """True when ``long`` is ``short`` with one letter doubled (``zeppelin`` / ``zepelin``)."""
    if len(long) != len(short) + 1:
        return False
    for i in range(len(long)):
        if long[:i] + long[i + 1:] == short:
            return (i > 0 and long[i - 1] == long[i]) or (i + 1 < len(long) and long[i + 1] == long[i])
    return False


class Names:
    """Resolve a word from a record value to a record name: exact, alias, or a harmless slip.

    Only two slips are forgiven, because a general one-letter distance links real but different
    words (``waistband`` / ``wristband``, ``toy_store`` / ``toy_story``): a missing or extra ``_``
    (``fire_station`` / ``firestation``), and a doubled letter in the name of a person, work or
    other named thing (``beatels`` is not forgiven, ``zepelin`` is).
    """

    def __init__(self, types: dict[str, str], aliases: dict[str, str]) -> None:
        self.names, self.aliases = set(types), aliases
        self.named = {n for n, t in types.items() if t != "object"}
        self.flat: dict[str, list[str]] = defaultdict(list)
        for n in self.names:
            self.flat[n.replace("_", "")].append(n)
        self._cache: dict[str, str | None] = {}

    def resolve(self, word: str) -> str | None:
        if word in self.aliases:
            return self.aliases[word]
        if word in self.names:
            return word
        if word not in self._cache:
            hit = None
            same = self.flat.get(word.replace("_", ""), [])
            if len(same) == 1 and len(word) >= 6:
                hit = same[0]
            elif len(word) >= 7:
                near = [n for n in self.named if abs(len(n) - len(word)) == 1
                        and (_doubled(word, n) or _doubled(n, word))]
                hit = near[0] if len(near) == 1 else None
            self._cache[word] = hit
        return self._cache[word]

    def mentions(self, word: str, list_key: bool = False) -> list[str]:
        """Record names inside one value word: the whole word, else runs of its ``_`` parts.

        In a list key (``made``, ``parts``, ``place``) the teacher often joined several things
        with ``_`` (``shell_padding_straps``), so a single part of 4+ letters may link any record.
        Elsewhere a single part only links a named thing (``medina_of_marrakesh`` -> ``marrakesh``,
        but ``open_world_series`` does not link the object ``series``).
        """
        whole = self.resolve(word)
        if whole:
            return [whole]
        parts, found = word.split("_"), []
        i = 0
        while i < len(parts):
            for j in range(len(parts), i, -1):
                cand = "_".join(parts[i:j])
                hit = None
                if j - i > 1:
                    hit = self.resolve(cand)
                elif len(cand) >= 4 and cand in self.names and (list_key or cand in self.named):
                    hit = cand
                if hit:
                    found.append(hit)
                    i = j - 1
                    break
            i += 1
        return list(dict.fromkeys(found))


def edges_from_records(types: dict[str, str], aliases: dict[str, str], rows: list[dict]) -> tuple[list[Edge], dict[str, str]]:
    """Links between records, and links to shared-value hubs. Returns (edges, new hub node types)."""
    names = Names(types, aliases)
    out: list[Edge] = []
    hub_types: dict[str, str] = {}
    for row in rows:
        obj = names.resolve(row["obj"]) or row["obj"]
        if obj not in types:
            continue
        etype = row["values"].get("type", "object")
        for key, value in row["values"].items():
            if key == "type":
                continue
            rel = EVENT_PLACE if key == "place" and etype == "event" else FROM_RECORD_KEY.get(key)
            hub_rel = HUB_KEY.get(key)
            for word in value.split():
                hits = [h for h in names.mentions(word, key in LIST_KEYS) if h != obj] if rel else []
                for hit in hits:
                    a, r, b = canonical(obj, rel, hit)
                    out.append(Edge(a, r, b, "fact", "records", f"round-3 record of {obj}: {key} {value}"))
                if hub_rel and not hits and _KEY.match(word) and word != obj:
                    src = "kind" if key == "kind" else "hub"
                    hub_types.setdefault(word, "kind" if key == "kind" else "value")
                    rr = EVENT_PLACE if key == "place" and etype == "event" else hub_rel
                    out.append(Edge(obj, rr, word, "fact", src, f"round-3 record of {obj}: {key} {value}"))
        out.append(Edge(obj, "of_type", f"type_{etype}", "fact", "type", ""))
        hub_types.setdefault(f"type_{etype}", "type")
    return out, hub_types


#: relations that put a place inside a larger place, for the ``located_in`` rule
_INSIDE = ("city_in", "capital_of", "part_of", "located_in")
_FOOD_LINK = ("eaten_in", "drunk_in", "grown_in", "origin_in", "national_style_of", "origin_claimed_by")


def rule_edges(g: Graph) -> list[Edge]:
    """Links that follow from other links. Only agreed facts (no views) feed a rule."""
    facts = [e for e in g.edges if e.view == "fact" and e.src not in HUB_SRC]
    have = {(e.a, e.b) for e in g.edges} | {(e.b, e.a) for e in g.edges}
    out: list[Edge] = []
    inside = defaultdict(set)
    for e in facts:
        if e.rel in _INSIDE:
            inside[e.a].add(e.b)
    for x, ys in sorted(inside.items()):
        for y in sorted(ys):
            for z in sorted(inside.get(y, ())):
                if z != x and (x, z) not in have:
                    out.append(Edge(x, "located_in", z, "fact", "rule", f"{x} is in {y}, {y} is in {z}"))
                    have.add((x, z))
                    have.add((z, x))
    food_in = defaultdict(set)
    for e in facts:
        if e.rel in _FOOD_LINK and g.types.get(e.b) == "country":
            food_in[e.a].add(e.b)
    shared = defaultdict(list)
    for food, places in food_in.items():
        ps = sorted(places)
        for i, p in enumerate(ps):
            for q in ps[i + 1:]:
                shared[(p, q)].append(food)
    for (p, q), foods in sorted(shared.items()):
        out.append(Edge(p, "shares_food", q, "fact", "rule", "both: " + " ".join(sorted(foods))))
    bands = {n for n, t in g.types.items() if t == "band"} | {e.a for e in g.edges if e.rel == "is_a" and e.b == "band"}
    members = defaultdict(set)
    for e in facts:
        if e.rel == "member_of" and e.b in bands:
            members[e.b].add(e.a)
    for band, ms in sorted(members.items()):
        ms_ = sorted(ms)
        for i, p in enumerate(ms_):
            for q in ms_[i + 1:]:
                out.append(Edge(p, "bandmate", q, "fact", "rule", f"both in {band}"))
    return out


def _dedupe(edges: list[Edge]) -> list[Edge]:
    seen: set[tuple[str, str, str, str]] = set()
    out = []
    for e in edges:
        k = (e.a, e.rel, e.b, e.view)
        if e.a != e.b and k not in seen and (e.b, inverse(e.rel), e.a, e.view) not in seen:
            seen.add(k)
            out.append(e)
    return out


def build_graph(seed: Path, records: Path | None) -> Graph:
    types, aliases, edges, drops = load_seed(seed)
    seed_nodes = frozenset(types)
    if records:
        rows = [json.loads(ln) for ln in records.open(encoding="utf-8")]
        for row in rows:
            name = aliases.get(row["obj"], row["obj"])
            if name not in types and _KEY.match(name):
                types[name] = row["values"].get("type", "object")
        rec_edges, hub_types = edges_from_records(types, aliases, rows)
        for n, t in hub_types.items():
            types.setdefault(n, t)
        held = {(e.a, e.rel, e.b) for e in edges} | drops
        edges = edges + [e for e in rec_edges if (e.a, e.rel, e.b) not in held]
        edges = _dedupe(edges)
        # a hub only one record points at links nothing
        degree = defaultdict(int)
        for e in edges:
            degree[e.a] += 1
            degree[e.b] += 1
        dead = {n for n, t in types.items() if t in ("value", "kind") and degree[n] < 2}
        edges = [e for e in edges if e.a not in dead and e.b not in dead]
        types = {n: t for n, t in types.items() if n not in dead}
    g = Graph(types, _dedupe(edges), seed_nodes)
    if records:
        g = Graph(types, _dedupe(g.edges + rule_edges(g)), seed_nodes)
    return g


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


def paths_from(g: Graph, src: str, targets: set[str] | None = None) -> dict[str, list[tuple[str, str, str]]]:
    """Cheapest path (over :data:`COST`) from ``src`` to every node, or to ``targets`` only.

    Ties go to fewer hops, then to the alphabetically first route, so every run gives the same
    paths. A path never passes through a hub it does not end at more than once.
    """
    nb = g.neighbours()
    best: dict[str, list[tuple[str, str, str]]] = {}
    heap = [(0.0, 0, (), src)]
    left = set(targets) if targets else None
    while heap:
        cost, n_hops, steps, node = heapq.heappop(heap)
        if node in best:
            continue
        best[node] = list(steps)
        if left is not None:
            left.discard(node)
            if not left:
                break
        for rel, m, view, c in nb[node]:
            if m not in best:
                heapq.heappush(heap, (round(cost + c, 6), n_hops + 1, steps + ((rel, m, view),), m))
    best.pop(src, None)
    return best if targets is None else {t: best[t] for t in targets if t in best}


def shortest_paths(g: Graph, pairs: list[tuple[str, str]] | None = None) -> dict[tuple[str, str], list[tuple[str, str, str]]]:
    """(x, y) -> [(rel, next node, view), ...] for the given pairs, or for every ordered pair."""
    by_src: dict[str, set[str]] = defaultdict(set)
    for x, y in pairs if pairs is not None else ((x, y) for x in g.types for y in g.types if x != y):
        by_src[x].add(y)
    out = {}
    for src in sorted(by_src):
        for dst, steps in paths_from(g, src, by_src[src]).items():
            out[(src, dst)] = steps
    return out


def sample_pairs(g: Graph, n_sources: int, per_source: int, seed: int) -> list[tuple[str, str]]:
    """Every pair among the seed nodes, plus random pairs among all non-hub nodes."""
    rng = random.Random(seed)
    seed_nodes = sorted(g.seed_nodes)
    pairs = [(x, y) for x in seed_nodes for y in seed_nodes if x != y]
    pool = sorted(set(g.types) - g.hubs())
    for x in rng.sample(pool, min(n_sources, len(pool))):
        for y in rng.sample(pool, min(per_source + 1, len(pool))):
            if y != x:
                pairs.append((x, y))
    return list(dict.fromkeys(pairs))


def step_text(steps: list[tuple[str, str, str]]) -> str:
    """``capital_of palestine according_to palestine`` for a step that only one side holds."""
    return " ".join(f"{r} {m}" + (f" according_to {v}" if v != "fact" else "") for r, m, v in steps)


def embed(g: Graph, dim: int = 16, hops: int = 4, decay: float = 0.6) -> tuple[list[str], np.ndarray]:
    """Node vectors from the graph alone: weighted k-hop walk co-occurrence -> PPMI -> SVD."""
    nodes = sorted(g.types)
    ix = {n: i for i, n in enumerate(nodes)}
    A = np.zeros((len(nodes), len(nodes)), dtype=np.float32)
    for e in g.edges:
        w = 1.0 / e.cost  # a view or a shared hub links two things more weakly than a fact
        A[ix[e.a], ix[e.b]] += w
        A[ix[e.b], ix[e.a]] += w
    P = A / A.sum(1, keepdims=True)
    M, Pk = np.zeros_like(P), np.eye(len(nodes), dtype=np.float32)
    for k in range(1, hops + 1):
        Pk = Pk @ P
        M += decay ** (k - 1) * Pk
    joint = M / M.sum()
    with np.errstate(divide="ignore"):
        pmi = np.log(joint / (joint.sum(1, keepdims=True) * joint.sum(0, keepdims=True)))
    ppmi = np.maximum(pmi, 0).astype(np.float64)
    U, S, _ = np.linalg.svd(ppmi, full_matrices=False)
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


#: an answer lists at most this many things; a hub's fan (``kitchen place_of ...``) is not asked
MAX_TARGETS = 6
#: a node's link line lists at most this many links
MAX_LINKS = 10


def training_lines(g: Graph, paths: dict[tuple[str, str], list[tuple[str, str, str]]], seed: int = 1) -> list[str]:
    """Dense lines in the round-3 alphabet.

    * ``turkey links. borders greece syria. coast_on mediterranean_sea. ...``
    * ``q turkey borders. a greece syria.``
    * ``q islam has_prophet. a jesus moses muhammad according_to islam.`` (one side holds it)
    * ``q jerusalem capital_of. a israel palestine contested.`` (sides differ)
    * ``q jerusalem capital_of according_to palestine. a palestine.``
    * ``q turkey borders greece. a yes.`` / ``q turkey borders israel. a no.``
    * ``q jerusalem capital_of palestine. a according_to palestine.``
    * ``q turkey hop led_zeppelin. a turkey has_city istanbul ... signed led_zeppelin.``
    * ``q turkey hops led_zeppelin. a 4.``
    """
    rng = random.Random(seed)
    hubs = g.hubs()
    by_rel: dict[tuple[str, str], dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for n, links in g.neighbours().items():
        if n in hubs:
            continue
        for rel, m, view, _ in links:
            by_rel[(n, rel)][view].append(m)
    lines: list[str] = []
    per_node: dict[str, list[str]] = defaultdict(list)
    for (n, rel), views in sorted(by_rel.items()):
        all_targets = list(dict.fromkeys(m for ms in views.values() for m in sorted(ms)))
        if len(all_targets) > MAX_TARGETS:
            continue
        if len(per_node[n]) < MAX_LINKS:
            per_node[n].append(f"{rel} {' '.join(all_targets)}.")
        held = sorted(views)
        tail = "" if held == ["fact"] else f" according_to {held[0]}" if len(held) == 1 else " contested"
        lines.append(f"q {n} {rel}. a {' '.join(all_targets)}{tail}.")
        if len(held) > 1:
            for view, ms in sorted(views.items()):
                lines.append(f"q {n} {rel} according_to {view}. a {' '.join(sorted(ms))}.")
    for n in sorted(per_node):
        lines.append(f"{n} links. " + " ".join(per_node[n]))
    lines += yes_no_lines(g, rng)
    for (x, y), steps in sorted(paths.items()):
        lines.append(f"q {x} hop {y}. a {x} {step_text(steps)}.")
        lines.append(f"q {x} hops {y}. a {len(steps)}.")
    return lines


def yes_no_lines(g: Graph, rng: random.Random) -> list[str]:
    """One ``yes`` (or ``according_to``) per non-hub link, and one ``no`` with a wrong partner of
    the same type, so the model learns that most things are *not* linked."""
    hubs = g.hubs()
    linked: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    for e in g.edges:
        prev = linked[(e.a, e.rel)].get(e.b)
        linked[(e.a, e.rel)][e.b] = e.view if prev in (None, e.view) else "contested"
    by_type: dict[str, list[str]] = defaultdict(list)
    for n, t in sorted(g.types.items()):
        if n not in hubs:
            by_type[t].append(n)
    out = []
    for (a, rel), targets in sorted(linked.items()):
        if a in hubs or rel in ("of_type",):
            continue
        for b, view in sorted(targets.items()):
            if b in hubs:
                continue
            answer = "yes" if view == "fact" else view if view == "contested" else "according_to " + view
            out.append(f"q {a} {rel} {b}. a {answer}.")
            pool = by_type[g.types[b]]
            for _ in range(5):
                c = rng.choice(pool)
                symmetric_hit = inverse(rel) == rel and a in linked.get((c, rel), {})
                if c not in targets and c != a and not symmetric_hit:
                    out.append(f"q {a} {rel} {c}. a no.")
                    break
    return out


def build(seed: Path, records: Path | None, out: Path, dim: int = 16, n_sources: int = 400,
          per_source: int = 40) -> dict:
    g = build_graph(seed, records)
    comps = components(g)
    if len(comps) != 1:
        small = sorted(comps, key=len)[:-1]
        raise SystemExit(f"not fully connected: {len(comps)} parts; cut off: {[sorted(c)[:8] for c in small[:20]]}")
    pairs = sample_pairs(g, n_sources, per_source, seed=20261001)
    paths = shortest_paths(g, pairs)
    nodes, V = embed(g, dim=dim)
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
                                "cosine": round(float(V[ix[x]] @ V[ix[y]]), 4)}) + "\n")
    nearest = {}
    hubs = g.hubs()
    keep = np.array([n not in hubs for n in nodes])
    for i, n in enumerate(nodes):
        sims = V @ V[i]
        sims[~keep] = -9
        sims[i] = -9
        nearest[n] = [nodes[j] for j in np.argsort(-sims)[:5]]
    vectors = {
        "dim": dim,
        "method": "graph only: 4-hop random-walk co-occurrence (decay 0.6, links weigh 1/cost), PPMI, SVD, unit length",
        "nodes": {n: {"type": g.types[n], "vec": np.round(V[i], 4).tolist(), "nearest": nearest[n]}
                  for i, n in enumerate(nodes)},
        "relations": offsets(g, nodes, V, "rel"),
        "views": offsets(g, nodes, V, "view"),
    }
    (out / "hops-vectors-r4.json").write_text(json.dumps(vectors), encoding="utf-8")
    lines = training_lines(g, paths)
    (out / "hops-train-r4.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

    src_count = defaultdict(int)
    for e in g.edges:
        src_count[e.src] += 1
    kinds = defaultdict(int)
    for ln in lines:
        w = ln.split(".")[0].split()
        kinds["links" if w[-1] == "links" else "hop" if len(w) == 4 and w[2] in ("hop", "hops") else
              "yes_no" if len(w) == 4 and w[0] == "q" else "view" if "according_to" in w else "relation"] += 1
    report = {"nodes": len(g.types), "hub_nodes": len(hubs), "seed_nodes": len(g.seed_nodes),
              "edges": len(g.edges), "edges_by_source": dict(src_count),
              "edges_with_a_view": sum(e.view != "fact" for e in g.edges),
              "views": sorted({e.view for e in g.edges} - {"fact"}),
              "relations": len({e.rel for e in g.edges}), "connected": True,
              "pairs": len(paths), "paths_using_a_view": sum(any(v != "fact" for _, _, v in s) for s in paths.values()),
              "max_hops": max(lengths), "mean_hops": round(sum(lengths) / len(lengths), 2),
              "hops_histogram": {h: lengths.count(h) for h in sorted(set(lengths))},
              "training_lines": len(lines), "training_lines_by_kind": dict(kinds)}
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
    b.add_argument("--sources", type=int, default=400, help="random start nodes for sampled hop paths")
    b.add_argument("--per-source", type=int, default=40)
    p = sub.add_parser("path")
    p.add_argument("--seed", type=Path, required=True)
    p.add_argument("--records", type=Path)
    p.add_argument("a")
    p.add_argument("b")
    args = ap.parse_args()
    if args.cmd == "build":
        print(json.dumps(build(args.seed, args.records, args.out, args.dim, args.sources, args.per_source), indent=1))
    else:
        g = build_graph(args.seed, args.records)
        steps = paths_from(g, args.a, {args.b}).get(args.b)
        print(args.a, step_text(steps) if steps else "(no path)")


if __name__ == "__main__":
    main()
