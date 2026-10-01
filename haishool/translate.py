"""English in, dense query out (parser); dense answer in, English out (translator). Rules only.

The translator never adds a fact: every content word of its English sentence comes from the
model's dense answer, so a wrong answer stays visibly wrong. The parser only picks *which*
record and key to ask; the record's entity type (from the records file) decides how a question
word maps to a key ("when" is ``time`` for an event, ``year`` for a work, ``era`` for a person).
"""

from __future__ import annotations

import re

from haishool.schema import ALL_KEYS, TYPES

SELF = "homunculi"
#: special records the parser routes to (written by hand in data/manual.jsonl)
UNKNOWABLE = "question_about_you"
RICH = "getting_rich"
PROBLEM = "solving_a_problem"

#: questions about the asker's present situation or the future: nothing in the facts can answer them
CATCH = re.compile(
    r"\b(?:am i|i am|i'm|im)\b.*\b(?:wearing|thinking|holding|doing|eating|drinking|feeling|sitting|standing|"
    r"looking|listening|watching|reading|carrying|hiding)\b"
    r"|\bwhat (?:is|'s) (?:in )?my\b|\bwhats my\b|\bwhich \w+ (?:am i|do i|did i|have i)\b|\bwhat \w+ (?:am i|do i have|did i)\b"
    r"|\bright now\b|\bwhat time is it\b|\bwhat day is (?:it|today)\b|\btoday'?s (?:date|news|weather)\b"
    r"|\bweather (?:today|now|tomorrow)\b|\bwhere am i\b|\bhow old am i\b|\bwho am i\b|\bin my (?:hand|pocket|room|head)\b"
    r"|\bnumber (?:am i|i am|i'm) thinking\b|\blottery numbers?\b|\bnext week'?s\b|\btomorrow'?s (?:price|numbers|news)\b"
    r"|\bwill (?:the )?(?:price|stock|market|bitcoin|dollar|euro)\b.*\b(?:go up|rise|fall|crash|drop)\b")
RICH_Q = re.compile(r"\b(?:rich|wealthy|millionaire|billionaire|financial freedom|get money|make (?:me )?(?:some )?money|"
                    r"earn (?:more )?money|grow my money|make a fortune)\b")
PROBLEM_Q = re.compile(r"\b(?:solve|fix|handle|deal with) (?:an?|my|this|the|our) (?:\w+ )?(?:issue|problem|trouble)\b"
                       r"|\bhelp me with (?:an?|my|this) (?:\w+ )?(?:issue|problem)\b|\bhow do (?:i|you) solve problems\b")

#: hand-written aliases (German, short forms); record aliases from the teacher are added at load time
ALIASES: dict[str, str] = {
    "kanzler": "chancellor_of_germany", "bundeskanzler": "chancellor_of_germany", "german_chancellor": "chancellor_of_germany",
    "chancellor": "chancellor_of_germany", "bundespraesident": "federal_president_of_germany",
    "german_president": "federal_president_of_germany", "president_of_germany": "federal_president_of_germany",
    "us_president": "president_of_the_united_states", "american_president": "president_of_the_united_states",
    "president": "president_of_the_united_states", "pope": "pope",
    "big_religions": "major_world_religions", "main_religions": "major_world_religions",
    "world_religions": "major_world_religions", "biggest_religions": "major_world_religions",
    "history_books": "well_known_history_books", "most_used_software": "most_used_software",
    "bundeslaender": "german_federal_states", "german_states": "german_federal_states",
    "states_of_germany": "german_federal_states", "deutschland": "germany",
}

#: question pattern -> intent, checked in order; an intent maps to keys by entity type below
INTENTS: list[tuple[str, str]] = [
    (r"\bwhat is your name\b|\byour name\b|\bwho are you\b|\bwho r u\b|\bwhat are you called\b|\bwhat'?s your name\b", "name"),
    (r"\bdesires?\b|\bwhat do you want\b|\bwish(?:es)?\b|\bdreams?\b|\bgoals?\b|\bwhat drives you\b", "desires"),
    (r"\bfeel(?:ings?)?\b|\bemotions?\b|\bare you (?:happy|sad|conscious|angry)\b", "feelings"),
    (r"\badvice\b|\bshould i\b|\bhow (?:do|can|could) i\b|\btips?\b", "advice"),
    (r"\bhow many (?:states|regions|provinces|federal states)\b|\bnumber of (?:states|regions|provinces)\b", "states"),
    (r"\bhow many (?:followers|believers|people (?:believe|follow))\b|\bfollowers\b|\bbelievers\b", "followers"),
    (r"\bhow many\b|\bnumber of\b|\bcount\b", "count"),
    (r"\bcapital\b", "capital"),
    (r"\blanguages?\b|\bspeak\b|\bspoken\b", "language"),
    (r"\bcurrency\b|\bpay with\b|\bmoney (?:in|of|do they use)\b", "currency"),
    (r"\bcontinent\b", "continent"),
    (r"\bgovernment\b|\bpolitical system\b|\bruled\b", "government"),
    (r"\bholy book\b|\bscriptures?\b|\bsacred (?:book|text)\b", "holy_book"),
    (r"\bgods?\b|\bbelieve in\b|\bbeliefs?\b", "god"),
    (r"\bfounded\b|\bfounder\b|\bwho started\b|\bwho (?:made|created|built|developed|wrote|directed|invented|designed|composed|sang|painted)\b"
     r"|\bmade by\b|\bcreator\b|\bauthor\b|\bdeveloper\b", "maker"),
    (r"\bwho is (?:the )?(?:current )?\w*\s?(?:chancellor|kanzler|bundeskanzler|president|pope|king|queen|leader|prime minister|holder)\b|\bholder\b|\bwho holds\b", "holder"),
    (r"\bsince when\b|\bhow long has\b", "since"),
    (r"\bwhy did\b|\bcause[sd]?\b|\bwhy was\b|\breason for\b", "cause"),
    (r"\bresults?\b|\boutcome\b|\bconsequences?\b|\bwhat happened after\b|\blead to\b|\bled to\b|\beffects?\b", "result"),
    (r"\bwho (?:was|were) involved\b|\bwho fought\b|\bwhich people\b|\bkey (?:people|figures)\b", "people"),
    (r"\bmeans?\b|\bmeaning\b|\bdefin(?:e|ition)\b", "meaning"),
    (r"\bexamples?\b", "example"),
    (r"\brelated\b|\bsimilar (?:ideas|concepts)\b", "related"),
    (r"\bfield\b|\bwhich (?:area|branch|subject)\b", "field"),
    (r"\bgenre\b|\bwhat kind of (?:game|movie|film|song|book|music|show|series|app)\b", "genre"),
    (r"\bknown for\b|\bfamous for\b|\bwhy (?:is|was) \w+(?: \w+)? (?:famous|known|important)\b|\bachievements?\b", "known_for"),
    (r"\bjob\b|\brole\b|\bprofession\b|\bdo for a living\b|\bwork as\b|\boccupation\b", "role"),
    (r"\bwhen\b|\bwhat year\b|\bwhich year\b|\bwhat century\b|\bwhich era\b", "when"),
    (r"\bwhere (?:is|was) \w+(?: \w+)? from\b|\bnationality\b|\bwhich country\b|\bwhat country\b|\bwhere from\b", "from"),
    (r"\bmembers?\b|\blist\b|\bname (?:all )?the\b|\bwhich are the\b|\bwhat are the\b", "members"),
    (r"\bcolou?rs?\b", "color"),
    (r"\bshape\b|\blook like\b", "shape"),
    (r"\bsize\b|\bhow big\b|\bhow large\b|\bhow small\b", "size"),
    (r"\bmade (?:of|from|out of)\b|\bmaterial\b|\bconsist\b", "made"),
    (r"\bparts?\b", "parts"),
    (r"\bused? (?:for|to)\b|\bwhat (?:is|are) (?:it|they|an?|the)? ?\w* ?for\b|\bpurpose\b|\bwhat do you do with\b|\buse\b", "use"),
    (r"\bwhere\b|\bfound\b|\blive\b|\brun\b|\blocated\b|\bhappen(?:ed)?\b", "where"),
    (r"\balive\b|\bliving\b", "alive"),
    (r"\bwhat kind\b|\bwhat type\b|\bwhat sort\b", "kind"),
    (r"\bwho\b|\bwhat (?:is|are|was|were)\b|\bwhat'?s\b", "what"),
]
#: intent -> candidate keys, first one the record's type carries wins
INTENT_KEYS: dict[str, tuple[str, ...]] = {
    "when": ("time", "year", "era", "since"),
    "where": ("place", "origin", "country", "continent"),
    "from": ("country", "origin", "continent", "place"),
    "maker": ("maker", "founder", "people"),
    "holder": ("holder",),
    "count": ("count", "states", "followers"),
    "members": ("members",),
    "god": ("god",),
}
OBJECT_KEYS = ("kind", "color", "shape", "size", "made", "parts", "use", "place", "alive")
DESCRIBE = re.compile(r"\btell me (?:about|everything)|\bdescribe\b|\bexplain\b|\binfo(?:rmation)? (?:about|on)\b|\bwhat do you know about\b")
SELF_WORDS = re.compile(r"\byou\b|\byour\b|\byourself\b|\bhomunculi\b|\bu\b|\bur\b")
PROPER = {"person", "country", "religion", "work", "event", "office"}
SMALL = {"of", "the", "and", "in", "on", "a", "an", "to", "for", "at", "by"}


def _singular(word: str) -> list[str]:
    forms = [word]
    if word.endswith("ies"):
        forms.append(word[:-3] + "y")
    if word.endswith("es"):
        forms.append(word[:-2])
    if word.endswith("s"):
        forms.append(word[:-1])
    return forms


def alias_index(records: list[dict]) -> dict[str, str]:
    """Teacher aliases that name exactly one record, plus the hand-written ones (known objects win)."""
    known = {r["obj"] for r in records}
    seen: dict[str, set[str]] = {}
    for r in records:
        for alias in r.get("values", {}).get("aliases", "").split():
            if len(alias) >= 3 and alias not in known:
                seen.setdefault(alias, set()).add(r["obj"])
    index = {a: next(iter(objs)) for a, objs in seen.items() if len(objs) == 1}
    index.update({a: o for a, o in ALIASES.items() if o in known})
    return index


def candidates(text: str, known: set[str] | dict, aliases: dict[str, str] | None = None) -> list[tuple[int, int, str]]:
    """Every record named in the text as ``(words, key length, record)``."""
    words = re.findall(r"[a-z0-9]+", text.lower().replace("'s ", " ").replace("ä", "ae").replace("ö", "oe").replace("ü", "ue"))
    aliases = aliases or {}
    found: dict[str, tuple[int, int, str]] = {}
    for n in range(6, 0, -1):
        for i in range(len(words) - n + 1):
            head, last = words[i:i + n - 1], words[i + n - 1]
            for form in _singular(last):
                key = "_".join([*head, form])
                hit = key if key in known else aliases.get(key)
                if hit and (hit not in found or (n, len(key)) > found[hit][:2]):
                    found[hit] = (n, len(key), hit)
    return list(found.values())


def find_object(text: str, known: set[str] | dict, aliases: dict[str, str] | None = None,
                wanted: set[str] | None = None) -> str | None:
    """The record the question is about: one whose type has the asked-for key first ("the capital
    of japan": japan, not the economics concept capital), then the longest name, then a named
    entity over a plain object."""
    types = known if isinstance(known, dict) else {}
    found = candidates(text, known, aliases)
    if not found:
        return None
    wanted = wanted or set()

    def score(c: tuple[int, int, str]) -> tuple:
        return (bool(wanted & set(_keys_for(types.get(c[2])))), c[0], types.get(c[2]) is not None, c[1])

    return max(found, key=score)[2]


def _keys_for(etype: str | None) -> tuple[str, ...]:
    if etype in TYPES:
        return tuple(TYPES[etype])
    return OBJECT_KEYS


def wanted_keys(question: str) -> set[str]:
    """Every key the question words could be asking for."""
    q = question.lower()
    return {k for pattern, intent in INTENTS if intent not in ("what", "kind") and re.search(pattern, q)
            for k in INTENT_KEYS.get(intent, (intent,))}


def parse(question: str, known: set[str] | dict, aliases: dict[str, str] | None = None,
          self_fallback: bool = True) -> dict:
    """Return ``{"obj", "attrs"}``; ``attrs`` None means describe the whole record.

    ``known`` maps object -> entity type (None for plain objects); a set is accepted too. With
    ``self_fallback`` a question that names nothing but says "you" is about Homunculi itself.
    """
    q = question.lower().strip()
    types = known if isinstance(known, dict) else dict.fromkeys(known)
    if CATCH.search(q) and UNKNOWABLE in types:
        return {"obj": UNKNOWABLE, "attrs": ["answer", "reason"]}
    if RICH_Q.search(q) and RICH in types:
        return {"obj": RICH, "attrs": ["answer", "advice"]}
    if PROBLEM_Q.search(q) and PROBLEM in types:
        return {"obj": PROBLEM, "attrs": ["advice"]}
    obj = find_object(q, types, aliases, wanted_keys(q))
    if obj is None and self_fallback and SELF_WORDS.search(q):
        obj = SELF
    if obj is None:
        return {"obj": None, "attrs": None}
    return {"obj": obj, "attrs": choose_attrs(q, obj, types.get(obj))}


def choose_attrs(question: str, obj: str, etype: str | None) -> list[str] | None:
    """Which keys of ``obj`` the question asks for; None means describe the whole record."""
    q = question.lower().strip()
    if DESCRIBE.search(q):
        return None
    keys = ("name", "kind", "made", "use", "place", "alive", "desires", "feelings") if obj == SELF else _keys_for(etype)
    for pattern, intent in INTENTS:
        if not re.search(pattern, q):
            continue
        if intent == "what":
            if obj == SELF:
                return ["name", "kind"]
            return ["kind"] if etype is None else None
        for key in INTENT_KEYS.get(intent, (intent,)):
            if key in keys:
                return [key]
        if intent == "kind":
            return ["kind"]
    return None if etype else ["kind"]


def has_wanted(question: str, obj: str, etype: str | None) -> bool:
    """Whether the record's type carries a key the question asks for (no key asked counts as yes)."""
    wanted = wanted_keys(question)
    keys = set(_keys_for(etype)) | ({"name", "desires", "feelings"} if obj == SELF else set())
    return not wanted or bool(wanted & keys)


def _article(word: str) -> str:
    if word.startswith(("uni", "use", "usu", "ut", "eu", "one")):
        return "a"
    if word.startswith(("hour", "honest", "honor")):
        return "an"
    return "an" if word[:1] in "aeiou" else "a"


ROMAN = re.compile(r"^(?=[ivx])x{0,3}(?:ix|iv|v?i{0,3})$")
ACRONYMS = {"uk", "usa", "us", "eu", "un", "uae", "ussr", "nato", "dna", "rna", "ai", "os", "gdp", "nasa", "ibm",
            "bmw", "amc", "hbo", "bbc", "cnn", "fbi", "cia", "ios", "pc", "tv", "usb", "cpu", "gpu", "html", "css",
            "sql", "api", "http", "www", "etf", "ipo", "ecb", "fed", "imf", "opec", "covid", "hiv", "mri", "gps", "ddr"}


def _word(p: str) -> str:
    return p.upper() if p in ACRONYMS else p


def _title(word: str) -> str:
    parts = word.replace("_", " ").split()
    return " ".join(p if (i and p in SMALL) else p.upper() if (i and ROMAN.match(p)) or p in ACRONYMS
                    else p[:1].upper() + p[1:] for i, p in enumerate(parts))


def _plain(word: str) -> str:
    word = re.sub(r"^(\d{3,4})_(\d{3,4})$", r"\1 to \2", word)
    return " ".join(_word(p) for p in word.replace("_", " ").split())


def _join(words: list[str], last: str = "and", proper: bool = False) -> str:
    words = [_title(w) if proper else _plain(w) for w in words]
    if len(words) <= 1:
        return "".join(words)
    return ", ".join(words[:-1]) + f" {last} " + words[-1]


def _sentences(words: list[str]) -> str:
    out = []
    for w in words:
        text = re.sub(r"\bi\b", "I", w.replace("_", " "))
        out.append(text[:1].upper() + text[1:] + ".")
    return " ".join(out)


def subject(obj: str, etype: str | None = None) -> str:
    name = obj.replace("_", " ")
    if etype == "office":
        return "The " + _title(obj)
    if etype == "list":
        return "The " + name
    if etype == "event" and not obj.startswith(("world_war", "the_")):
        return "The " + _title(obj)
    if etype in PROPER:
        return _title(obj)
    if etype == "concept":
        name = _plain(obj)
        return name[:1].upper() + name[1:]
    return f"{_article(name).capitalize()} {name}"


def _self_sentence(attr: str, v: list[str], value: str) -> str | None:
    if attr == "name":
        return f"My name is {value.capitalize()}."
    if attr == "place":
        return f"I run locally in {_join([w.capitalize() for w in v])}."
    if attr == "kind":
        return f"I am {_article(v[0])} {' '.join(w.replace('_', ' ') for w in v)}."
    if attr == "made":
        return f"I am made of {_join(v)}."
    if attr == "use":
        return f"I am used for {_join(v)}."
    if attr == "alive":
        return "I am not alive." if v[0] == "no" else "I am alive."
    if attr in ("desires", "feelings"):
        noun = "desires" if attr == "desires" else "feelings"
        if v[0] in ("none", "no"):
            rest = v[1:]
            return f"I have no {noun} of my own." + (f" {_sentences(rest)}" if rest else "")
        return f"My {noun}: {_join(v)}."
    return None


def sentence(obj: str, attr: str, value: str, etype: str | None = None) -> str:
    v = value.split()
    s = subject(obj, etype)
    if not v:
        return f"I don't know the {attr.replace('_', ' ')} of {s[:1].lower() + s[1:] if etype is None else s}."
    if obj == SELF:
        text = _self_sentence(attr, v, value)
        if text:
            return text
    names ={"maker", "founder", "people", "holder", "capital", "country", "continent", "currency", "language", "origin", "members"}
    def j(last: str = "and") -> str:
        return _join(v, last, proper=attr in names and attr != "currency")
    if attr == "kind":
        return f"{s} is {_article(v[0])} {' '.join(w.replace('_', ' ') for w in v)}."
    if attr == "color":
        return f"{s} is usually {j('or')}."
    if attr == "shape":
        return f"{s} is usually {j('or')} in shape."
    if attr == "size":
        return f"Its size: {j('or')}."
    if attr == "made":
        return f"{s} is made of {j()}."
    if attr == "parts":
        return f"Its main parts are {j()}."
    if attr == "use":
        return f"{s} is used for {j()}."
    if attr == "place":
        if etype == "event":
            return f"{s} took place in {_join(v, proper=True)}."
        return f"You usually find it in places like {j('or')}."
    if attr == "alive":
        return f"{s} is not alive." if v[0] == "no" else f"{s} is alive."
    if attr == "name":
        return f"Its name is {value}."
    if attr == "role":
        return f"Main role of {s}: {j()}."
    if attr == "country":
        return f"{s} belongs to {j()}." if etype == "office" else f"{s} is from {j()}."
    if attr == "era":
        return f"{s} lived around {j()}." if re.fullmatch(r"\d{3,4}", v[0]) else f"{s} lived in the {j()}."
    if attr == "known_for":
        return f"{s} is known for {j()}."
    if attr == "continent":
        return f"{s} is in {j()}."
    if attr == "capital":
        return f"The capital of {s} is {j()}."
    if attr == "language":
        return f"In {s} people mainly speak {j()}."
    if attr == "currency":
        return f"The currency of {s} is the {j()}."
    if attr == "states":
        return f"{s} has {v[0]} states or regions." if v[0].isdigit() else f"{s} is divided into {j()}."
    if attr == "government":
        return f"{s} is {_article(v[0])} {j()}."
    if attr == "founder":
        return f"{s} goes back to {j()}."
    if attr == "holy_book":
        return f"The holy book of {s} is {_join(v, proper=True)}."
    if attr == "god":
        return f"Belief about god in {s}: {j()}."
    if attr == "origin":
        return f"{s} began in {j()}."
    if attr == "followers":
        return f"{s} has about {j()} followers."
    if attr == "time":
        return f"{s} happened in {j()}."
    if attr == "cause":
        return f"Its main cause was {j()}."
    if attr == "result":
        return f"It led to {j()}."
    if attr == "people":
        return f"The main people involved were {j()}."
    if attr == "field":
        return f"{s} belongs to the field of {j()}."
    if attr == "meaning":
        return f"{s} means {j()}."
    if attr == "example":
        return f"An example: {j()}."
    if attr == "related":
        return f"Related ideas: {j()}."
    if attr == "maker":
        return f"{s} was made by {j()}."
    if attr == "year":
        return f"{s} came out in {j()}."
    if attr == "genre":
        return f"Its genre is {j()}."
    if attr == "members":
        return f"{s} are {j()}."
    if attr == "count":
        return f"There are {v[0]} of them." if v[0].isdigit() else f"Count: {j()}."
    if attr == "holder":
        return f"{s} is {j()}, as far as my data goes."
    if attr == "since":
        return f"Holding the office since {j()}."
    if attr == "advice":
        return f"What generally helps: {j()}."
    if attr == "answer":
        return _sentences(v)
    if attr == "reason":
        return re.sub(r"\bi\b", "I", f"Because {_join(v)}.")
    if attr in ("desires", "feelings"):
        return f"{attr.capitalize()}: {j()}."
    return f"{attr.replace('_', ' ')}: {value.replace('_', ' ')}."


def _person_text(obj: str, values: dict[str, str]) -> list[str]:
    """One sentence for a person instead of five that each start with the name."""
    def j(key: str, proper: bool = False) -> str:
        return _join(values[key].split(), proper=proper)
    text = _title(obj)
    if "kind" in values:
        text += f" is {_article(values['kind'])} {j('kind')}"
    else:
        text += " is a person"
    if "role" in values:
        text += f" ({j('role')})"
    if "country" in values:
        text += f" from {j('country', proper=True)}"
    if "era" in values:
        era = values["era"].split()[0]
        text += f", active around {j('era')}" if re.fullmatch(r"\d{3,4}", era) else f", active in the {j('era')}"
    out = [text + "."]
    if "known_for" in values:
        out.append(f"Known for {j('known_for')}.")
    return out


def record_text(line: str, etype: str | None = None) -> list[str]:
    chunks = [c.strip() for c in line.split(".") if c.strip()]
    if not chunks:
        return []
    obj, out = chunks[0], []
    pairs = dict(c.partition(" ")[::2] for c in chunks[1:])
    if (etype or pairs.get("type")) == "person":
        return _person_text(obj, {k: v for k, v in pairs.items() if v})
    s = subject(obj, etype)
    for chunk in chunks[1:]:
        key, _, value = chunk.partition(" ")
        if key == "type" and etype is None and value in TYPES:
            etype, s = value, subject(obj, value)
            continue
        if key in ALL_KEYS and key not in ("type", "aliases"):
            text = sentence(obj, key, value, etype)
            if out and text.startswith(s + " "):
                text = ("They " if etype == "list" else "It ") + text[len(s) + 1:]
            out.append(text)
    return out
