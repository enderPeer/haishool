"""The dense object-fact language (DOF) shared by data, model, parser and translator.

One record per object, one line, only lowercase letters, spaces and periods (the Homunculi
29-symbol alphabet without commas), attributes in a fixed order:

    apple. kind fruit. color red green yellow. shape round. size hand. made skin flesh seeds.
    parts stem skin core. use eating cooking juice. place tree kitchen market. alive no.

A query asks for one attribute of one object and the answer is that attribute's value:

    q apple color. a red green yellow.

Values are short lists of single words or hyphen-free compounds joined by spaces; multi-word
object names are joined with underscores in the record (``fire_truck``) so the parser can match
them, and rendered with spaces by the translator.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

#: attribute key -> (question words the parser looks for, English template parts)
ATTRIBUTES: dict[str, str] = {
    "kind": "what kind of thing it is",
    "color": "usual colors",
    "shape": "usual shape",
    "size": "usual size, compared to something familiar",
    "made": "what it is made of",
    "parts": "its main parts",
    "use": "what it is used for",
    "place": "where it is usually found",
    "alive": "whether it is alive (yes or no)",
}
#: attributes only the identity record carries (not asked from the teacher)
EXTRA = ("name",)
ORDER = tuple(ATTRIBUTES)

#: Round 3 (knowledge map): entity types and their keys. A record carries ``type`` plus the keys of
#: its type; values follow the same rules as the object attributes (digits allowed for years and
#: counts).
TYPES: dict[str, dict[str, str]] = {
    "person": {"kind": "what they are, one or two words", "role": "their main role or job",
               "country": "their country", "era": "century or decade they lived or worked in",
               "known_for": "what they are known for"},
    "country": {"kind": "country", "continent": "continent", "capital": "capital city",
                "language": "main languages", "currency": "currency", "states": "number of states or regions",
                "government": "form of government", "known_for": "what it is known for"},
    "religion": {"kind": "religion", "founder": "founder or origin figure", "holy_book": "holy book",
                 "god": "belief about god or gods", "origin": "place of origin",
                 "followers": "approximate number of followers"},
    "event": {"kind": "kind of event", "time": "year or century", "place": "where it happened",
              "cause": "main cause", "result": "main result", "people": "main people involved"},
    "concept": {"kind": "kind of concept", "field": "field of knowledge", "meaning": "short meaning",
                "example": "a simple example", "related": "related ideas"},
    "work": {"kind": "kind of work or product", "maker": "creator, company or artist",
             "year": "year it appeared", "genre": "genre or category", "known_for": "what it is known for"},
    "list": {"kind": "list", "members": "the members", "count": "how many members"},
    "office": {"kind": "office", "holder": "current holder", "since": "year since when",
               "country": "country"},
}
EXTRA_KEYS = ("type", "name", "desires", "feelings", "advice", "answer", "reason", "aliases")
ALL_KEYS = ORDER + tuple(dict.fromkeys(k for keys in TYPES.values() for k in keys if k not in ORDER)) + EXTRA_KEYS
QUERY_KEYS = tuple(k for k in ALL_KEYS if k not in ("aliases",))

_WORD = re.compile(r"[a-z0-9_]+")


def normalise_value(text: str, max_words: int = 6) -> str:
    """Lowercase words only, at most ``max_words``, joined by single spaces."""
    words = _WORD.findall(text.lower().replace("-", "_").replace(" and ", " "))
    words = [w.strip("_") for w in words if w.strip("_") and w not in {"a", "an", "the", "and", "or"}]
    seen: list[str] = []
    for w in words:
        if w not in seen:
            seen.append(w)
    return " ".join(seen[:max_words])


def object_key(name: str) -> str:
    """``"André-Marie Ampère"`` -> ``andre_marie_ampere``, ``"J. K. Rowling"`` -> ``j_k_rowling``."""
    plain = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    plain = re.sub(r"['’]", "", plain)
    return re.sub(r"_+", "_", "_".join(_WORD.findall(re.sub(r"[^a-z0-9]+", " ", plain)))).strip("_")


@dataclass(frozen=True)
class Record:
    obj: str
    values: dict[str, str]

    def line(self) -> str:
        parts = [f"{self.obj}."]
        for key in ALL_KEYS:
            value = self.values.get(key)
            if value:
                parts.append(f"{key} {value}.")
        return " ".join(parts)

    def queries(self) -> list[str]:
        return [f"q {self.obj} {key}. a {self.values[key]}." for key in QUERY_KEYS if self.values.get(key)]


def parse_record(line: str) -> Record | None:
    chunks = [c.strip() for c in line.strip().split(".") if c.strip()]
    if not chunks:
        return None
    obj, values = chunks[0], {}
    for chunk in chunks[1:]:
        key, _, value = chunk.partition(" ")
        if key in ALL_KEYS and value:
            values[key] = value
    return Record(obj, values)
