"""English in, dense query out (parser); dense answer in, English out (translator). Rules only.

The translator never adds a fact: every content word of its English sentence comes from the
model's dense answer, so a wrong answer stays visibly wrong.
"""

from __future__ import annotations

import re

from haishool.schema import ORDER

SELF = "homunculi"

#: question patterns -> attribute, checked in order
ATTR_PATTERNS: list[tuple[str, str]] = [
    (r"\bname\b|\bwho are you\b|\bwho r u\b", "name"),
    (r"\bcolou?rs?\b", "color"),
    (r"\bshape\b|\blook like\b", "shape"),
    (r"\bsize\b|\bhow big\b|\bhow large\b|\bhow small\b", "size"),
    (r"\bmade (?:of|from|out of)\b|\bmaterial\b|\bconsist\b", "made"),
    (r"\bparts?\b|\bconsists? of parts\b", "parts"),
    (r"\bused? (?:for|to)\b|\bwhat (?:is|are) (?:it|they|an?|the)? ?\w* ?for\b|\bpurpose\b|\bwhat do you do with\b|\buse\b", "use"),
    (r"\bwhere\b|\bfound\b|\blive\b|\brun\b|\blocated\b", "place"),
    (r"\balive\b|\bliving\b|\bliving thing\b", "alive"),
    (r"\bwhat kind\b|\bwhat type\b|\bwhat sort\b|\bwhat (?:is|are) (?:an?|the)?\b|\bwhat'?s (?:an?|the)?\b|\bwho\b", "kind"),
]
DESCRIBE = re.compile(r"\btell me (?:about|everything)|\bdescribe\b|\bexplain\b|\binfo(?:rmation)? (?:about|on)\b")
SELF_WORDS = re.compile(r"\byou\b|\byour\b|\byourself\b|\bhomunculi\b|\bu\b|\bur\b")


def _singular(word: str) -> list[str]:
    forms = [word]
    if word.endswith("ies"):
        forms.append(word[:-3] + "y")
    if word.endswith("es"):
        forms.append(word[:-2])
    if word.endswith("s"):
        forms.append(word[:-1])
    return forms


def find_object(text: str, known: set[str]) -> str | None:
    words = re.findall(r"[a-z]+", text.lower())
    best = None
    for n in (3, 2, 1):
        for i in range(len(words) - n + 1):
            head, last = words[i:i + n - 1], words[i + n - 1]
            for form in _singular(last):
                key = "_".join([*head, form])
                if key in known and (best is None or len(key) > len(best)):
                    best = key
        if best:
            return best
    return None


def parse(question: str, known: set[str]) -> dict:
    q = question.lower().strip()
    obj = find_object(q, known)
    if obj is None and SELF_WORDS.search(q):
        obj = SELF
    attr = None
    if not DESCRIBE.search(q):
        for pattern, key in ATTR_PATTERNS:
            if re.search(pattern, q):
                attr = key
                break
    if obj == SELF and attr == "kind" and re.search(r"\bwho\b", q):
        attr = "name"
    return {"obj": obj, "attr": attr}


def _article(word: str) -> str:
    if word.startswith(("uni", "use", "usu", "ut", "eu", "one")):
        return "a"
    if word.startswith(("hour", "honest", "honor")):
        return "an"
    return "an" if word[:1] in "aeiou" else "a"


def _join(words: list[str], last: str = "and") -> str:
    words = [w.replace("_", " ") for w in words]
    if len(words) <= 1:
        return "".join(words)
    return ", ".join(words[:-1]) + f" {last} " + words[-1]


def sentence(obj: str, attr: str, value: str) -> str:
    v = value.split()
    name = obj.replace("_", " ")
    if not v:
        return f"I don't know the {attr} of {_article(name)} {name}."
    if obj == SELF:
        if attr == "name":
            return f"My name is {value.capitalize()}."
        if attr == "place":
            return f"I run locally in {_join([w.capitalize() for w in v])}."
        if attr == "kind":
            return f"I am {_article(v[0])} {' '.join(v)}."
        if attr == "made":
            return f"I am made of {_join(v)}."
        if attr == "use":
            return f"I am used for {_join(v)}."
        if attr == "alive":
            return "I am not alive." if v[0] == "no" else "I am alive."
    subject = f"{_article(name).capitalize()} {name}"
    if attr == "kind":
        return f"{subject} is {_article(v[0])} {' '.join(w.replace('_', ' ') for w in v)}."
    if attr == "color":
        return f"{subject} is usually {_join(v, 'or')}."
    if attr == "shape":
        return f"{subject} is usually {_join(v, 'or')} in shape."
    if attr == "size":
        return f"Its size: {_join(v, 'or')}."
    if attr == "made":
        return f"{subject} is made of {_join(v)}."
    if attr == "parts":
        return f"Its main parts are {_join(v)}."
    if attr == "use":
        return f"{subject} is used for {_join(v)}."
    if attr == "place":
        return f"You usually find it in places like {_join(v, 'or')}."
    if attr == "alive":
        return f"{subject} is not alive." if v[0] == "no" else f"{subject} is alive."
    if attr == "name":
        return f"Its name is {value}."
    return f"{attr}: {value}."


def record_text(line: str) -> list[str]:
    chunks = [c.strip() for c in line.split(".") if c.strip()]
    if not chunks:
        return []
    obj, out = chunks[0], []
    name = obj.replace("_", " ")
    subject = f"{_article(name).capitalize()} {name}"
    for chunk in chunks[1:]:
        key, _, value = chunk.partition(" ")
        if key in ORDER or key == "name":
            text = sentence(obj, key, value)
            if out and text.startswith(subject + " "):
                text = "It " + text[len(subject) + 1:]
            out.append(text)
    return out
