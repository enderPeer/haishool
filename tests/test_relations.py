import json
import re
from pathlib import Path

import pytest

from haishool.relations import (INVERSE, Names, build_graph, canonical, components, embed, inverse,
                                paths_from, rule_edges, sample_pairs, shortest_paths, step_text,
                                training_lines)

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "data" / "hops-seed-r4.jsonl"
RECORDS = ROOT / "data" / "records-r3-all.jsonl"


@pytest.fixture(scope="module")
def seed_graph():
    return build_graph(SEED, None)


@pytest.fixture(scope="module")
def graph():
    return build_graph(SEED, RECORDS)


def test_inverses_are_one_to_one():
    for rel, inv in INVERSE.items():
        assert inverse(inverse(rel)) == rel
    backs = [inv for rel, inv in INVERSE.items() if inv != rel]
    assert len(backs) == len(set(backs))
    assert not set(backs) & set(INVERSE), "an inverse name is also used as a forward relation"


def test_canonical_flips_inverse_names():
    assert canonical("israel", "has_capital", "jerusalem") == ("jerusalem", "capital_of", "israel")
    assert canonical("turkey", "borders", "greece") == ("turkey", "borders", "greece")


def test_seed_alone_reaches_everything(seed_graph):
    assert len(components(seed_graph)) == 1
    n = len(seed_graph.types)
    assert len(shortest_paths(seed_graph)) == n * (n - 1)


def test_all_records_are_one_connected_graph(graph):
    assert len(components(graph)) == 1
    rows = [json.loads(ln) for ln in RECORDS.open(encoding="utf-8")]
    assert sum(r["obj"] in graph.types for r in rows) >= len(rows) - 5


def test_paths_read_back_and_forth(graph):
    edges = {(e.a, e.rel, e.b) for e in graph.edges}
    pairs = sample_pairs(graph, 20, 10, seed=1)
    for (x, _), steps in shortest_paths(graph, pairs).items():
        cur = x
        for rel, m, _ in steps:
            assert (cur, rel, m) in edges or (m, inverse(rel), cur) in edges
            cur = m


def test_known_hops(graph):
    assert step_text(paths_from(graph, "turkey", {"israel"})["israel"]) == "recognised israel"
    # Turkey reaches Led Zeppelin through agreed facts only
    assert all(v == "fact" for _, _, v in paths_from(graph, "turkey", {"led_zeppelin"})["led_zeppelin"])


def test_seed_view_beats_teacher_fact(graph):
    capital = [e for e in graph.edges if (e.a, e.rel, e.b) == ("jerusalem", "capital_of", "israel")]
    assert capital and all(e.view != "fact" for e in capital)
    assert not any((e.a, e.rel, e.b) == ("jesus", "country", "israel") for e in graph.edges)


def test_name_slips_are_forgiven_only_when_harmless():
    names = Names({"led_zeppelin": "person", "firestation": "object", "wristband": "object",
                   "toy_story": "work", "united_kingdom": "country"}, {"uk": "united_kingdom"})
    assert names.resolve("uk") == "united_kingdom"
    assert names.resolve("fire_station") == "firestation"
    assert names.resolve("led_zepelin") == "led_zeppelin"
    assert names.resolve("waistband") is None
    assert names.resolve("toy_store") is None
    assert names.mentions("jerusalem_dead_sea") == []  # not a record here
    assert Names({"marrakesh": "city"}, {}).mentions("medina_of_marrakesh") == ["marrakesh"]
    assert Names({"series": "object"}, {}).mentions("open_world_series") == []
    assert Names({"handle": "object"}, {}).mentions("shakers_handle", list_key=True) == ["handle"]


def test_rules_only_link_what_follows(graph):
    rules = rule_edges(graph)
    assert ("hagia_sophia", "located_in", "turkey") in {(e.a, e.rel, e.b) for e in graph.edges}
    for e in [e for e in graph.edges if e.rel == "bandmate"]:
        assert {e.a, e.b} <= {"jimmy_page", "robert_plant"}
    assert all(e.src == "rule" for e in rules)


def test_training_lines_use_the_dense_alphabet(seed_graph):
    lines = training_lines(seed_graph, shortest_paths(seed_graph))
    for line in lines:
        assert re.fullmatch(r"[a-z0-9_ .]+", line), line
    assert "q jerusalem capital_of. a israel palestine contested." in lines
    assert "q jerusalem capital_of according_to palestine. a palestine." in lines
    assert "q turkey hops israel. a 1." in lines
    assert "q turkey borders greece. a yes." in lines
    assert any(ln.startswith("q turkey borders ") and ln.endswith(" a no.") for ln in lines)
    assert "q jerusalem capital_of palestine. a according_to palestine." in lines
    assert any(ln.startswith("turkey links. ") for ln in lines)


def test_no_line_is_a_real_link(seed_graph):
    real = {(e.a, e.rel, e.b) for e in seed_graph.edges} | {(e.b, e.rel, e.a) for e in seed_graph.edges if inverse(e.rel) == e.rel}
    for ln in training_lines(seed_graph, {}):
        if ln.endswith(" a no."):
            _, a, rel, b = ln.split(".")[0].split()
            assert (a, rel, b) not in real, ln


def test_vectors_are_stable_and_unit(seed_graph):
    nodes, v1 = embed(seed_graph)
    _, v2 = embed(seed_graph)
    assert (v1 == v2).all()
    assert abs(float((v1 ** 2).sum(1).mean()) - 1) < 1e-6
    sims = v1 @ v1.T
    i = nodes.index("led_zeppelin")
    near = [nodes[j] for j in (-sims[i]).argsort()[1:6]]
    assert {"jimmy_page", "robert_plant"} & set(near)


def test_seed_is_valid_json_lines():
    for line in SEED.open(encoding="utf-8"):
        json.loads(line)
