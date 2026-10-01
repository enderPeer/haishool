import json
import re
from pathlib import Path

import pytest

from haishool.relations import (INVERSE, build_graph, canonical, components, embed, inverse,
                                shortest_paths, step_text, training_lines)

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "data" / "hops-seed-r4.jsonl"
RECORDS = ROOT / "data" / "records-r3-all.jsonl"


@pytest.fixture(scope="module")
def graph():
    return build_graph(SEED, RECORDS)


@pytest.fixture(scope="module")
def paths(graph):
    return shortest_paths(graph)


def test_inverses_are_one_to_one():
    for rel, inv in INVERSE.items():
        assert inverse(inverse(rel)) == rel
    backs = [inv for rel, inv in INVERSE.items() if inv != rel]
    assert len(backs) == len(set(backs))
    assert not set(backs) & set(INVERSE), "an inverse name is also used as a forward relation"


def test_canonical_flips_inverse_names():
    assert canonical("israel", "has_capital", "jerusalem") == ("jerusalem", "capital_of", "israel")
    assert canonical("turkey", "borders", "greece") == ("turkey", "borders", "greece")


def test_everything_reaches_everything(graph, paths):
    assert len(components(graph)) == 1
    n = len(graph.types)
    assert len(paths) == n * (n - 1)


def test_paths_read_back_and_forth(graph, paths):
    # every step is a real edge in one direction or the other
    edges = {(e.a, e.rel, e.b) for e in graph.edges}
    for (x, _), steps in list(paths.items())[:500]:
        cur = x
        for rel, m, _ in steps:
            assert (cur, rel, m) in edges or (m, inverse(rel), cur) in edges
            cur = m


def test_known_hops(paths):
    assert step_text(paths[("turkey", "israel")]) == "recognised israel"
    # Turkey reaches Led Zeppelin through agreed facts only
    assert all(v == "fact" for _, _, v in paths[("turkey", "led_zeppelin")])


def test_seed_view_beats_teacher_fact(graph):
    capital = [e for e in graph.edges if (e.a, e.rel, e.b) == ("jerusalem", "capital_of", "israel")]
    assert capital and all(e.view != "fact" for e in capital)
    assert not any((e.a, e.rel, e.b) == ("jesus", "country", "israel") for e in graph.edges)


def test_training_lines_use_the_dense_alphabet(graph, paths):
    lines = training_lines(graph, paths)
    for line in lines:
        assert re.fullmatch(r"[a-z0-9_ .]+", line), line
    assert "q jerusalem capital_of. a israel palestine contested." in lines
    assert "q jerusalem capital_of according_to palestine. a palestine." in lines
    assert "q turkey hops israel. a 1." in lines


def test_vectors_are_stable_and_unit(graph):
    nodes, v1 = embed(graph)
    _, v2 = embed(graph)
    assert (v1 == v2).all()
    assert abs(float((v1 ** 2).sum(1).mean()) - 1) < 1e-9
    sims = v1 @ v1.T
    i = nodes.index("led_zeppelin")
    near = [nodes[j] for j in (-sims[i]).argsort()[1:6]]
    assert {"jimmy_page", "robert_plant"} & set(near)


def test_seed_is_valid_json_lines():
    for line in SEED.open(encoding="utf-8"):
        json.loads(line)
