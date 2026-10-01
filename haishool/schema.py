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
ALL_KEYS = ORDER + ("name",)

_WORD = re.compile(r"[a-z_]+")


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
    return "_".join(_WORD.findall(name.lower().replace("-", " ").replace(" ", "_").replace("__", "_"))).strip("_")


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
        return [f"q {self.obj} {key}. a {self.values[key]}." for key in ALL_KEYS if self.values.get(key)]


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
