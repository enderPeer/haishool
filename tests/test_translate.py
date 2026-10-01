"""Parser and translator rules (no model needed): python -m pytest tests"""

from haishool.combine import merge
from haishool.search import Index, resolve
from haishool.schema import Record, object_key
from haishool.translate import alias_index, parse, record_text, sentence

ROWS = [
    {"obj": "banana", "values": {"kind": "fruit", "color": "yellow green"}},
    {"obj": "capital", "values": {"type": "concept", "kind": "economic_resource", "field": "finance"}},
    {"obj": "dna", "values": {"type": "concept", "kind": "molecule"}},
    {"obj": "japan", "values": {"type": "country", "kind": "country", "capital": "tokyo", "states": "47"}},
    {"obj": "germany", "values": {"type": "country", "kind": "country", "states": "16"}},
    {"obj": "french_revolution", "values": {"type": "event", "kind": "revolution", "time": "1789_1799"}},
    {"obj": "albert_einstein", "values": {"type": "person", "kind": "scientist", "country": "germany",
                                          "era": "1900s", "known_for": "theory_of_relativity"}},
    {"obj": "chancellor_of_germany", "values": {"type": "office", "holder": "friedrich_merz", "since": "2025"}},
    {"obj": "major_world_religions", "values": {"type": "list", "members": "christianity islam"}},
    {"obj": "question_about_you", "values": {"type": "concept", "answer": "i_cannot_know_that"}},
    {"obj": "getting_rich", "values": {"type": "concept", "answer": "there_is_no_quick_safe_way"}},
    {"obj": "homunculi", "values": {"type": "self", "name": "homunculi", "place": "berlin"}},
]
KNOWN = {r["obj"]: r["values"].get("type") for r in ROWS}
ALIASES = alias_index(ROWS)


def ask(q):
    p = parse(q, KNOWN, ALIASES)
    return p["obj"], p["attrs"]


def test_routing():
    assert ask("what color is a banana") == ("banana", ["color"])
    assert ask("what is the capital of japan") == ("japan", ["capital"])
    assert ask("how many states does germany have") == ("germany", ["states"])
    assert ask("when did the french revolution happen") == ("french_revolution", ["time"])
    assert ask("who is the kanzler") == ("chancellor_of_germany", ["holder"])
    assert ask("what are the big religions") == ("major_world_religions", ["members"])
    assert ask("who was albert einstein") == ("albert_einstein", None)
    assert ask("which shoe am i wearing right now") == ("question_about_you", ["answer", "reason"])
    assert ask("make me rich") == ("getting_rich", ["answer", "advice"])
    assert ask("who are you") == ("homunculi", ["name"])
    assert ask("what is a quasar") == (None, None)
    assert ask("what is capital") == ("capital", None)


def test_sentences():
    assert sentence("banana", "color", "yellow green") == "A banana is usually yellow or green."
    assert sentence("french_revolution", "time", "1789_1799", "event") == "The French Revolution happened in 1789 to 1799."
    assert sentence("chancellor_of_germany", "holder", "friedrich_merz", "office").startswith(
        "The Chancellor of Germany is Friedrich Merz")
    assert sentence("question_about_you", "answer", "i_cannot_know_that", "concept") == "I cannot know that."
    assert sentence("homunculi", "place", "berlin") == "I run locally in Berlin."
    assert sentence("dna", "kind", "molecule", "concept") == "DNA is a molecule."


def test_person_is_one_sentence():
    line = Record("albert_einstein", next(r["values"] for r in ROWS if r["obj"] == "albert_einstein")).line()
    assert record_text(line, "person") == [
        "Albert Einstein is a scientist from Germany, active in the 1900s.", "Known for theory of relativity."]


def test_keys_and_merge():
    assert object_key("J. K. Rowling") == "j_k_rowling"
    assert object_key("Thirty Years' War") == "thirty_years_war"
    merged = merge([{"obj": "brazil", "values": {"kind": "nut", "color": "brown"}},
                    {"obj": "brazil", "values": {"type": "country", "capital": "brasilia"}},
                    {"obj": "brazil", "values": {"type": "country", "states": "26"}}])
    assert merged["brazil"] == {"type": "country", "capital": "brasilia", "states": "26"}


def test_always_answers():
    index = Index(ROWS)

    def route(q):
        r = resolve(q, KNOWN, ALIASES, index)
        return r["obj"], r["attrs"], r["prefix"]

    assert route("who is einstein")[0] == "albert_einstein"
    assert route("who was einstien")[0] == "albert_einstein"
    assert route("which country has the capital tokyo")[:2] == ("japan", ["capital"])
    assert route("hello")[:2] == ("homunculi", ["name", "place"])
    assert route("where do you run")[:2] == ("homunculi", ["place"])
    obj, attrs, prefix = route("what is a quasar")
    assert obj and prefix.startswith("I learned nothing about quasar.")
