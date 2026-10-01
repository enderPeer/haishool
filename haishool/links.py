"""Chat side of round 4: ask the model about the links between things, and say its answers in English.

The round-4 model (``haishool.relations``) learned, for a few dozen linked things,

    q turkey borders. a greece syria.                              a relation
    q jerusalem capital_of. a israel palestine contested.          a disputed one ...
    q jerusalem capital_of according_to palestine. a palestine.    ... and each side's view
    q abraham hop led_zeppelin. a abraham father_figure_of christianity according_to christianity ...
    q abraham hops led_zeppelin. a 6.                              a path and its length

This module decides when a question is one of those (which relation, which view, which two
things) and turns the dense answer into English. Which relations a thing has comes from the
training lines; every value in the answer is written by the model.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from haishool.translate import _title, candidates

PREPS = ("of", "in", "by", "on", "from", "to", "with")
#: question words for relation names whose own words are not how people ask
SYNONYMS: dict[str, tuple[str, ...]] = {
    "borders": ("border", "borders", "neighbour", "neighbours", "neighbor", "neighbors", "next"),
    "capital_of": ("capital",), "has_capital": ("capital",), "was_capital_of": ("capital",), "had_capital": ("capital",),
    "founded": ("found", "founded", "start", "started", "create", "created"),
    "founded_by": ("founded", "founder", "started", "created"),
    "born_in": ("born", "birthplace", "from"), "birthplace_of": ("born",),
    "has_member": ("member", "members", "band", "plays", "played"), "member_of": ("member", "band"),
    "origin_in": ("origin", "come", "comes", "from", "originate"), "origin_of": ("origin", "invented"),
    "origin_claimed_by": ("origin", "invented", "claim", "claims", "whose", "from", "come", "comes"),
    "claims_origin_of": ("claim", "claims", "origin"),
    "holy_to": ("holy", "sacred"), "holds_holy": ("holy", "sacred"),
    "coast_on": ("coast", "sea"), "coast_of": ("coast", "coasts"),
    "city_in": ("city", "where", "located"), "has_city": ("city", "cities"), "located_in": ("where", "located"),
    "eats": ("eat", "eats", "food", "dish", "dishes"), "eaten_in": ("eaten", "eat", "where"),
    "drinks": ("drink", "drinks"), "drunk_in": ("drunk", "drink", "where"),
    "spoke_language_of": ("language", "speak", "spoke"), "language_spoken_by": ("language", "spoken"),
    "influenced": ("influence", "influenced"), "influenced_by": ("influence", "influenced", "influences"),
    "signed": ("signed", "sign", "label"), "signed_by": ("signed", "label"),
    "fought_in": ("fought", "war", "wars", "fight"), "had_side": ("side", "sides", "fought"),
    "ruled": ("ruled", "rule"), "ruled_by": ("ruled", "rule", "ruler"),
    "recognised": ("recognise", "recognised", "recognize", "recognized"),
    "recognised_by": ("recognise", "recognised", "recognize", "recognized"),
    "father_figure_of": ("father",), "has_father_figure": ("father",),
    "prophet_of": ("prophet",), "has_prophet": ("prophet", "prophets"),
    "son_of_god_in": ("son", "god"), "holds_son_of_god": ("son", "god"),
    "state_religion": ("religion",), "state_religion_of": ("religion",),
}
CONNECT = re.compile(r"\b(?:connect(?:ed|ion|s)?|link(?:ed|s)?|relat(?:ed|ion|ionship)|between|hops?|path|lead(?:s)? to|get from)\b")
ACCORDING = re.compile(r"\baccording to ([a-z ]+?)(?:[?.!,]|$)|\bin the view of ([a-z ]+?)(?:[?.!,]|$)|\bfor ([a-z]+)s?\b")


def _stem(word: str) -> str:
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


@dataclass
class Graph:
    rels: dict[str, list[str]] = field(default_factory=dict)          # node -> relations it has
    views: dict[tuple[str, str], list[str]] = field(default_factory=dict)  # (node, relation) -> views held
    all_views: set[str] = field(default_factory=set)
    relation_names: set[str] = field(default_factory=set)
    edges: dict[tuple[str, str, str], list[str]] = field(default_factory=dict)  # (a, rel, b) -> views ("fact")

    @property
    def nodes(self) -> set[str]:
        return set(self.rels)

    def named(self, question: str, aliases: dict[str, str] | None = None) -> list[str]:
        """Linked things named in the question, in the order they appear; a rare word of a name
        counts too ("einstein" -> albert_einstein)."""
        q = question.lower()
        found = {c[2] for c in candidates(q, self.nodes, aliases) if c[2] in self.rels}
        if not hasattr(self, "_words"):
            index: dict[str, set[str]] = {}
            for n in self.rels:
                for w in n.split("_"):
                    if len(w) >= 4:
                        index.setdefault(w, set()).add(n)
            self._words = {w: next(iter(ns)) for w, ns in index.items() if len(ns) == 1}
        # a one-word thing that is only the asked relation's word ("capital") is not a subject
        rel_words = {w for r in self.relation_names for w in r.split("_")}
        if len(found) > 1:
            found = {n for n in found if "_" in n or n not in rel_words} or found
        if len(found) < 2:
            covered = {w for n in found for w in n.split("_")}
            for w in re.findall(r"[a-z0-9]+", q):
                if w in self._words and w not in covered and w not in rel_words:
                    found.add(self._words[w])
        return sorted(found, key=lambda n: min((q.find(w) for w in n.split("_") if q.find(w) >= 0), default=len(q)))


def load_graph(train_lines: Path, edges: Path | None = None) -> Graph:
    """Node lines of the training file (``turkey. borders greece syria. coast_on ...``) give each
    node's relations; the edge list gives which views hold a disputed link."""
    g = Graph()
    for line in train_lines.open(encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("q "):
            continue
        chunks = [c.strip() for c in line.split(".") if c.strip()]
        node = chunks[0][: -len(" links")] if chunks[0].endswith(" links") else chunks[0]  # 4b: "x links. rel b."
        if " " in node:
            continue
        rels = [c.split()[0] for c in chunks[1:] if c.split()]
        g.rels.setdefault(node, [])
        g.rels[node] += [r for r in rels if r not in g.rels[node]]
        g.relation_names.update(rels)
    if edges and edges.exists():
        for line in edges.open(encoding="utf-8"):
            e = json.loads(line)
            view = e.get("view", "fact")
            for key in ((e["a"], e["rel"], e["b"]), (e["b"], e.get("inverse", e["rel"]), e["a"])):
                g.edges.setdefault(key, [])
                if view not in g.edges[key]:
                    g.edges[key].append(view)
            if view == "fact":
                continue
            g.all_views.add(e["view"])
            for key in ((e["a"], e["rel"]), (e["b"], e.get("inverse", e["rel"]))):
                g.views.setdefault(key, [])
                if e["view"] not in g.views[key]:
                    g.views[key].append(e["view"])
    return g


NICE: dict[str, str] = {
    "origin_claimed_by": "is claimed as their own by", "claims_origin_of": "claims to be the origin of",
    "old_name_from": "has an old name from", "gave_old_name_to": "gave an old name to", "has_city": "has the city",
    "makes_known": "makes famous", "has_culture_of": "has a culture of", "culture_in": "is part of the culture in",
    "coast_on": "has a coast on", "coast_of": "is on the coast of", "city_in": "is a city in",
    "capital_of": "is the capital of", "has_capital": "has the capital", "was_capital_of": "was the capital of",
    "had_capital": "had the capital", "birthplace_of": "is the birthplace of", "father_figure_of": "is a father figure of",
    "has_father_figure": "has the father figure", "holy_to": "is holy to", "holds_holy": "holds holy",
    "prophet_of": "is a prophet of", "has_prophet": "has the prophet", "state_religion_of": "was the state religion of",
    "state_religion": "had the state religion", "lived_in": "lived in", "born_in": "was born in",
    "has_member": "has the members", "member_of": "is a member of", "gave_word": "gave the word",
    "followed_by": "was followed by", "origin_in": "has its origin in", "origin_of": "is the origin of",
    "influenced_by": "was influenced by", "refined_in": "was refined in", "had_side": "had on its side",
    "ruled_by": "was ruled by", "located_in": "is located in", "spoke_language_of": "spoke the language of",
    "has_national_style": "has its own style of", "national_style_of": "is a national style of",
    "based_in": "is based in", "son_of_god_in": "is the son of God in", "holds_son_of_god": "holds as the son of God",
    "happened_in": "happened in", "scene_of": "was the scene of", "home_of": "is home to", "eaten_in": "is eaten in",
    "drunk_in": "is drunk in", "grown_in": "is grown in", "formed_in": "was formed in", "signed_by": "was signed by",
    "country": "is from", "continent": "is in", "known_for": "is known for", "made_of": "is made of", "is_a": "is a",
    "of_type": "is of the type", "type_of": "is a type of", "genre": "has the genre", "field": "belongs to the field",
    "era": "belongs to the era", "currency": "uses the currency", "language": "speaks", "role": "has the role",
    "holder": "is held by", "played_by": "is played by", "has_instance": "has the example", "has_part": "has the part",
    "part_of": "is part of", "has_component": "has the part", "component_of": "is part of",
}


GENERIC = {"is_a", "of_type", "type_of", "has_instance"}
WEAK_WORDS = {"is", "a", "the", "type", "kind", "has", "was", "had", "of"}


def _clause_free(question: str) -> str:
    """The question without its "according to ..." clause (that names the side, not the subject)."""
    return ACCORDING.sub(" ", question.lower())


def relation_for(question: str, obj: str, g: Graph, among: list[str] | None = None) -> str | None:
    """The relation of ``obj`` (or, with ``among``, any of those relations) the question asks about."""
    q = question.lower()
    words = {_stem(w) for w in re.findall(r"[a-z]+", q)}
    if re.search(r"\bwho (?:is|are|was|were|plays?|played) in\b", q) and "has_member" in g.rels.get(obj, []):
        return "has_member"
    best, best_score = None, 0.0
    for rel in (among if among is not None else g.rels.get(obj, [])):
        if rel in GENERIC:
            continue
        own = {_stem(w) for w in rel.split("_") if w not in PREPS and w not in WEAK_WORDS}
        syn = {_stem(w) for w in SYNONYMS.get(rel, ())}
        score = len(words & own) * 2 + len(words & syn)
        if score > best_score:
            best, best_score = rel, score
    return best


VIEW_ALIASES = {"jews": "judaism", "jewish": "judaism", "christians": "christianity", "christian": "christianity",
                "muslims": "islam", "muslim": "islam", "israelis": "israel", "palestinians": "palestine",
                "greeks": "greece", "greek": "greece", "turks": "turkey", "turkish": "turkey", "americans": "united_states",
                "the us": "united_states", "the usa": "united_states", "historian": "historians"}


def view_for(question: str, g: Graph) -> str | None:
    m = ACCORDING.search(question.lower())
    if not m:
        return None
    phrase = next(x for x in m.groups() if x).strip()
    for word, view in VIEW_ALIASES.items():
        if re.search(rf"\b{word}\b", phrase) and view in g.all_views:
            return view
    for v in sorted(g.all_views, key=len, reverse=True):
        if v in phrase.replace(" ", "_") or v.replace("_", " ") in phrase or (len(phrase) >= 4 and phrase.rstrip("s") in v):
            return v
    return None


def two_nodes(question: str, g: Graph, aliases: dict[str, str] | None = None) -> tuple[str, str] | None:
    """The two linked things a "how is X connected to Y" question names, in the order asked."""
    if not CONNECT.search(question.lower()):
        return None
    found = g.named(question, aliases)
    return (found[0], found[-1]) if len(found) >= 2 else None


def node_in(question: str, g: Graph, aliases: dict[str, str] | None = None) -> str | None:
    """The linked thing the question is about: one that has the asked relation first ("the capital
    of israel": israel, not the concept capital), then the longest name."""
    q = _clause_free(question)
    found = [c for c in candidates(q, g.nodes, aliases) if c[2] in g.nodes]
    if found:
        return max(found, key=lambda c: (relation_for(q, c[2], g) is not None, c[0], c[1]))[2]
    for word, view in VIEW_ALIASES.items():  # "what is holy to the jews" -> judaism
        if view in g.nodes and re.search(rf"\b{word}\b", question.lower()):
            return view
    return None


YESNO = re.compile(r"^\s*(?:is|are|was|were|does|do|did|has|have|had|can)\b")


def yes_no(question: str, g: Graph, aliases: dict[str, str] | None = None) -> tuple[str, str, str] | None:
    """``("turkey", "borders", "greece")`` for "does turkey border greece?", else None."""
    q = question.lower()
    if not YESNO.search(q):
        return None
    found = g.named(q, aliases)
    if len(found) < 2:
        return None
    pairs = [(a, b) for i, a in enumerate(found) for b in found[i + 1:]]
    for a, b in pairs:  # the asked relation links them
        rel = relation_for(q, a, g)
        if rel and (a, rel, b) in g.edges:
            return a, rel, b
    for a, b in pairs:  # any link between them that the question words fit
        rels = [r for (x, r, y) in g.edges if x == a and y == b]
        rel = relation_for(q, a, g, among=rels)
        if rel:
            return a, rel, b
    a, b = found[0], found[-1]  # the asked relation, even if they are not linked by it: "no"
    rel = relation_for(q, a, g) or relation_for(q, a, g, among=sorted(g.relation_names))
    return (a, rel, b) if rel else None


def edge_truth(g: Graph, a: str, rel: str, b: str) -> list[str]:
    """What the links data says: ``["fact"]``, the views that hold it, or ``[]`` (not a link)."""
    return g.edges.get((a, rel, b), [])


def yes_no_sentence(a: str, rel: str, b: str, truth: list[str], model_said: str) -> str:
    claim = f"{_title(a)} {rel_phrase(rel)} {_title(b)}"
    if "fact" in truth:
        text = f"Yes: {claim}."
    elif truth:
        text = f"It depends on whom you ask: {claim} according to {', '.join(_title(v) for v in truth)}; others disagree."
    else:
        text = f"Not in what I learned: I have no link saying that {claim}."
    said = model_said.split()[0] if model_said.split() else ""
    agrees = (said == "yes") == ("fact" in truth) if said in ("yes", "no") else True
    return text if agrees else text + " (The model guessed otherwise; the answer follows the links it was trained on.)"


def rel_phrase(rel: str) -> str:
    if rel in NICE:
        return NICE[rel]
    words = rel.split("_")
    if words[0] == "has":
        return "has " + " ".join(words[1:])
    if words[-1] == "by" or (words[-1] == "in" and (words[0].endswith("ed") or words[0] in ("born", "held"))):
        return "was " + " ".join(words)
    if words[-1] in PREPS:
        return "is " + " ".join(words)
    return " ".join(words)


def relation_sentence(obj: str, rel: str, value: str, g: Graph) -> str:
    v = value.split()
    contested = "contested" in v
    v = [w for w in v if w != "contested"]
    parts: list[str] = []
    i = 0
    while i < len(v):
        if v[i] == "according_to" and i + 1 < len(v) and parts:
            parts[-1] += f" (according to {_title(v[i + 1])})"
            i += 2
            continue
        parts.append(_title(v[i]))
        i += 1
    if not parts:
        return f"I don't know what {_title(obj)} {rel_phrase(rel)}."
    joined = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
    text = f"{_title(obj)} {rel_phrase(rel)} {joined}."
    if contested:
        sides = g.views.get((obj, rel), [])
        text += " This is contested" + (f"; sides with their own view: {', '.join(_title(s) for s in sides)}. Ask \"according to ...\" for one side." if sides else ".")
    return text


def path_text(path: str, hops: str, g: Graph) -> str:
    """``abraham father_figure_of christianity according_to christianity state_religion_of ...`` ->
    "Abraham → father figure of → Christianity (according to Christianity) → ... (6 hops)"."""
    toks = path.split()
    if not toks:
        return "I can't find a path between them."
    out = [_title(toks[0])]
    i = 1
    while i < len(toks):
        if toks[i] == "according_to" and i + 1 < len(toks):
            out[-1] += f" (according to {_title(toks[i + 1])})"
            i += 2
            continue
        if toks[i] in g.relation_names and i + 1 < len(toks):
            out.append(toks[i].replace("_", " "))
            out.append(_title(toks[i + 1]))
            i += 2
            continue
        out.append(_title(toks[i]))
        i += 1
    if len(toks) == 5 and toks[1] == "of_type" and toks[3] == "type_of":  # only a shared kind links them
        kind = toks[2].removeprefix("type_").replace("_", " ")
        return (f"Both are {kind}s ({_title(toks[0])} and {_title(toks[4])}); I learned no closer link between them.")
    n = hops.split()[0] if hops.split() else ""
    tail = f" That is {n} hop{'s' if n != '1' else ''}." if n.isdigit() else ""
    return " → ".join(out) + "." + tail
