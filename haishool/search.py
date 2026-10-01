"""Find the best record for any question, so Homunculi always answers from what it learned.

The parser (``translate.parse``) handles questions that name a record. Everything else comes here,
in this order, and the model still writes every fact of the answer:

1. **name words**   "who is einstein"            -> albert_einstein (one rare word of a record's name)
2. **typos**        "who was einstien"           -> einstein, then 1
3. **value words**  "who invented the telephone" -> alexander_graham_bell, key known_for (the words
                    appear in that record's facts; the model answers that key)
4. **small talk**   "hello", "how are you"       -> the identity record
5. **closest name** "what is a quasar"           -> the record whose name is spelled most alike,
                    said plainly ("I learned nothing about quasar; the closest thing I know is ...")

Matching uses inverse document frequency, so a word that appears in few records counts far more
than one that appears everywhere ("country", "used").
"""

from __future__ import annotations

import difflib
import math
import re
from collections import defaultdict
from dataclasses import dataclass

from haishool.translate import (PROBLEM, RICH, SELF, SELF_WORDS, UNKNOWABLE, _keys_for, choose_attrs, has_wanted,
                                parse, subject, wanted_keys)

STOP = set("""a an the of in on at to for from by with and or is are was were be been being am do does did done
have has had what which who whom whose when where why how whats who's what's that this these those it its it's
i me my mine you your yours u ur we us our they them their he him his she her hers there here can could would
should will shall may might must tell know about please give show explain describe say name list some any all
much many more most very really just also than then so if not no yes ok okay hey hi hello thanks thank
something anything thing things kind type sort one ones get got make made""".split())
SMALL_TALK = {
    "feelings": re.compile(r"\bhow are you\b|\bhow do you feel\b|\bhow is it going\b|\bhow'?s it going\b"),
    "desires": re.compile(r"\bthanks?\b|\bthank you\b|\bwhat do you want\b|\bwhat can you do\b|\bhelp\b"),
    "greeting": re.compile(r"^\s*(?:hi|hello|hey|hallo|moin|servus|good (?:morning|evening|day)|yo)\b"),
}


@dataclass
class Hit:
    obj: str
    key: str | None    # the key whose value matched (None: matched by name)
    how: str           # "name", "value", "closest"
    matched: str       # the words that matched, for the "what the model said" line
    score: float = 0.0


def _forms(word: str, stems: bool = True) -> list[str]:
    """The word, its singular and (with ``stems``) its stem: painted -> paint, planes -> plane."""
    forms = [word]
    endings = (("ies", "y"), ("es", ""), ("s", "")) + ((("ed", ""), ("ing", ""), ("er", "")) if stems else ())
    for suffix, repl in endings:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            forms.append(word[: -len(suffix)] + repl)
    return forms


def content_words(question: str) -> list[str]:
    words = re.findall(r"[a-z0-9]+", question.lower().replace("'s ", " "))
    return [w for w in words if w not in STOP and (len(w) >= 3 or w.isdigit())]


class Index:
    def __init__(self, rows: list[dict]) -> None:
        self.types = {r["obj"]: r["values"].get("type") for r in rows}
        self.names: dict[str, set[str]] = defaultdict(set)              # name word -> records
        self.alias: dict[str, set[str]] = defaultdict(set)              # alias word -> records (weaker)
        self.values: dict[str, set[tuple[str, str]]] = defaultdict(set)  # value word or item -> (record, key)
        for r in rows:
            obj, values = r["obj"], r["values"]
            for w in obj.split("_"):
                if w not in STOP and (len(w) >= 3 or w.isdigit()):
                    self.names[w].add(obj)
            for key, value in values.items():
                if key == "type":
                    continue
                for item in value.split():
                    words = [w for w in item.split("_") if w not in STOP and (len(w) >= 3 or w.isdigit())]
                    if key == "aliases":
                        for w in words:
                            self.alias[w].add(obj)
                        continue
                    self.values[item].add((obj, key))
                    for w in words:
                        self.values[w].add((obj, key))
        self.n = max(1, len(rows))
        self.vocab = sorted(set(self.names) | set(self.alias) | {w for w in self.values if "_" not in w})
        self.record_names = list(self.types)

    def idf(self, df: int) -> float:
        return math.log(1 + self.n / max(1, df))

    def _phrases(self, words: list[str]) -> list[str]:
        """Two- and three-word phrases as value items ("light bulb" -> light_bulb)."""
        return ["_".join(words[i:i + n]) for n in (3, 2) for i in range(len(words) - n + 1)]

    def by_name(self, words: list[str], wanted: set[str], keys_for) -> Hit | None:
        score: dict[str, float] = defaultdict(float)
        why: dict[str, list[str]] = defaultdict(list)
        for w in words:
            for table, weight in ((self.names, 1.0),):  # teacher aliases are too noisy word by word
                done = False
                for f in _forms(w, stems=False):
                    objs = table.get(f)
                    if objs and len(objs) <= 6:
                        for o in objs:
                            score[o] += weight * self.idf(len(objs))
                            why[o].append(f)
                        done = True
                        break
                if done:
                    break
        if not score:
            return None
        best = max(score, key=lambda o: (score[o], bool(wanted & set(keys_for(self.types.get(o)))),
                                         self.types.get(o) is not None, -len(o)))
        return Hit(best, None, "name", " ".join(why[best]), score[best])

    def by_value(self, words: list[str], wanted: set[str], exclude: str | None = None,
                 prefer: tuple[str, ...] = (), raw: list[str] | None = None) -> Hit | None:
        """``raw`` is the full word sequence, for phrases with small words ("romeo and juliet")."""
        score: dict[tuple[str, str], float] = defaultdict(float)
        why: dict[tuple[str, str], list[str]] = defaultdict(list)
        for term in dict.fromkeys(self._phrases(words) + self._phrases(raw or [])):
            for hit in self.values.get(term, ()):
                score[hit] += 3 * self.idf(len(self.values[term]))
                why[hit].append(term)
        for w in words:
            for f in _forms(w):
                hits = self.values.get(f)
                if hits and len(hits) <= 60:
                    for hit in hits:
                        score[hit] += self.idf(len(hits))
                        why[hit].append(f)
                    break
        score = {h: s for h, s in score.items() if h[0] != exclude}
        if not score:
            return None

        def rank(h: tuple[str, str]) -> tuple:
            obj, key = h
            return (score[h] + (1.0 if key in wanted else 0) + (0.5 if self.types.get(obj) in prefer else 0),
                    key not in ("kind", "field", "related", "example"), -len(obj))

        obj, key = max(score, key=rank)
        return Hit(obj, key, "value", " ".join(dict.fromkeys(why[(obj, key)])), score[(obj, key)])

    def correct(self, words: list[str]) -> tuple[list[str], str]:
        fixed, notes = [], []
        for w in words:
            if len(w) >= 5 and w not in self.names and w not in self.alias and w not in self.values:
                close = difflib.get_close_matches(w, self.vocab, n=1, cutoff=0.86)
                if close:
                    fixed.append(close[0])
                    notes.append(f"{w}->{close[0]}")
                    continue
            fixed.append(w)
        return fixed, " ".join(notes)

    def closest(self, words: list[str]) -> Hit:
        """The record whose name (or one of its name words) is spelled most like a question word."""
        words = [w for w in words if w not in VERBS] or words or ["homunculi"]
        best, best_ratio = "homunculi", 0.0
        for obj in self.record_names:
            parts = obj.split("_")
            for w in words:
                for target in (obj, *parts):
                    m = difflib.SequenceMatcher(None, w, target)
                    if m.real_quick_ratio() <= best_ratio or m.quick_ratio() <= best_ratio:
                        continue
                    r = m.ratio() - (0.02 if target != obj else 0)
                    if r > best_ratio:
                        best, best_ratio = obj, r
        return Hit(best, None, "closest", " ".join(words), best_ratio)


#: question verbs that name no thing ("who wrote ...", "what does ... mean")
VERBS = set("""wrote write written writes invented invent painted paint made created create discovered discover
founded found built build directed sang sing composed designed developed mean means meant happened happen
live lived use used called call located find work works play played start started""".split())


AGENT_KEYS = {"maker", "founder", "people", "holder"}
SPECIAL = {SELF, UNKNOWABLE, RICH, PROBLEM}
#: records whose one-word name is also an everyday helper word ("can you tell me ...")
WEAK = {"can", "will", "may", "match", "watch", "fly", "kind", "type", "mean", "set", "light", "well", "rock", "bat"}


def _answer(obj: str, attrs: list[str] | None, note: str = "", prefix: str = "") -> dict:
    return {"obj": obj, "attrs": attrs, "note": note, "prefix": prefix}


def resolve(question: str, known: dict, aliases: dict[str, str], index: Index) -> dict:
    """Always return a record to answer from: ``{"obj", "attrs", "note", "prefix"}``.

    ``note`` says how the record was found (shown under the answer); ``prefix`` is a plain
    sentence put before the answer when the record is only the nearest thing Homunculi knows.
    """
    q = question.lower().strip()
    words = content_words(q)
    wanted = wanted_keys(q)
    who = bool(re.search(r"\bwho\b", q))
    prefer = ("person", "work") if who else ()
    raw = re.findall(r"[a-z0-9]+", q.replace("'s ", " "))

    small = next((k for k, pat in SMALL_TALK.items() if pat.search(q)), None)
    if small and set(words) <= WEAK:
        attrs = {"feelings": ["feelings"], "desires": ["desires"], "greeting": ["name", "place"]}[small]
        return _answer(SELF, attrs, "small talk: identity record")

    p = parse(q, known, aliases, self_fallback=False)
    obj = p["obj"]
    if obj in WEAK and any(w != obj for w in words):
        better = index.by_name([w for w in words if w != obj], wanted, _keys_for)
        if better:
            obj = None
    if obj:
        if obj in SPECIAL or has_wanted(q, obj, known.get(obj)):
            return _answer(obj, p["attrs"])
        # the record has no key the question asks for: maybe another record's facts answer it
        # ("which country has the capital tokyo", "who invented the telephone")
        hit = index.by_value(words, wanted, exclude=obj, prefer=prefer, raw=raw)
        if hit and (hit.key in wanted or ((who or wanted & AGENT_KEYS) and set(obj.split("_")) & set(
                hit.matched.replace("_", " ").split()))):
            return _answer(hit.obj, [hit.key], f"found by: {hit.matched} in {hit.obj}.{hit.key}")
        return _answer(obj, p["attrs"])

    if not words:
        if SELF_WORDS.search(q):
            return _answer(SELF, choose_attrs(q, SELF, "self") or ["name", "desires"])
        return _answer(SELF, ["name", "desires"], "no content words: identity record")

    fixed, notes = index.correct(words)
    if notes:
        q2 = q
        for pair in notes.split():
            bad, good = pair.split("->")
            q2 = re.sub(rf"\b{re.escape(bad)}\b", good, q2)
        p = parse(q2, known, aliases, self_fallback=False)
        if p["obj"] and p["obj"] not in WEAK:
            return _answer(p["obj"], p["attrs"], f"spelling: {notes}")
        q = q2
        raw = re.findall(r"[a-z0-9]+", q.replace("'s ", " "))
    spell = f"spelling: {notes}; " if notes else ""

    name = index.by_name(fixed, wanted, _keys_for)
    value = index.by_value(fixed, wanted, prefer=prefer, raw=raw)
    # a phrase or rare words in someone's facts beat a weak name match ("who painted the mona lisa")
    if name and not (value and value.score > 2 * name.score):
        return _answer(name.obj, choose_attrs(q, name.obj, known.get(name.obj)),
                       f"{spell}found by name word: {name.matched}")
    # "where do you run": a question to Homunculi itself, unless a phrase clearly names something else
    if re.search(r"\b(?:you|your|yourself|u|ur)\b", q) and not (value and "_" in value.matched):
        return _answer(SELF, choose_attrs(q, SELF, "self") or ["name", "desires"])
    if value:
        return _answer(value.obj, [value.key], f"{spell}found by: {value.matched} in {value.obj}.{value.key}",
                       "" if "_" in value.matched or value.score >= 7.0 else "The nearest fact I have: ")

    hit = index.closest(fixed)
    etype = known.get(hit.obj)
    name_text = subject(hit.obj, etype if etype != "self" else None)
    if etype is None or name_text.startswith("The "):
        name_text = name_text[:1].lower() + name_text[1:]
    return _answer(hit.obj, None, f"{spell}closest name to {hit.matched}",
                   f"I learned nothing about {hit.matched}. The closest thing I know is {name_text}: ")
