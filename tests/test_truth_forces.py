import json
import math
import random
import re
from collections import Counter
from pathlib import Path

import pytest

from haishool.truth import Gate, is_dense, num, parse_num
from haishool.truth.forces import (BODIES, CALC, CONTACT_RANGE_M, KEYS, MAX_LIST, OPERANDS, PROPORTIONS, RANGE_FACTOR,
                                   ROW_KEYS, SET_KEYS, SHARED, SURFACE_G, TABLE, UNCHECKED_KEYS, VERBS, ForcesGate,
                                   calc_answer, compute, gate, matches, n_tokens, range_order, record_lines,
                                   same_number, sig4, tidy_number)

ROOT = Path(__file__).resolve().parents[1]
JSONL = ROOT / "data" / "truth-v5" / "forces.jsonl"
FUNDAMENTAL = ["gravity", "electromagnetism", "strong", "weak"]
CONTACT = ["friction", "static_friction", "kinetic_friction", "normal", "tension", "drag", "lift", "buoyancy", "spring",
           "pressure_force", "surface_tension", "capillary", "thrust"]
BONDS = ["hydrogen_bond", "ionic_bond", "covalent_bond", "metallic_bond"]


@pytest.fixture(scope="module")
def g():
    return gate()


@pytest.fixture(scope="module")
def lines(g):
    return g.generate(random.Random(1), 2000)


@pytest.fixture(scope="module")
def big(g):
    return g.generate(random.Random(8), 20000)


def mutations(answer: str) -> list[str]:
    """Wrong answers: every digit changed in turn, yes/no flipped, or a word dropped or spoilt."""
    words = answer.split()
    if parse_num(answer) is not None:
        out = []
        for i, w in enumerate(words):
            if w.isdigit():
                out.append(" ".join(words[:i] + [str(int(w) % 9 + 1)] + words[i + 1:]))
        return out
    if answer in ("yes", "no"):
        return ["no" if answer == "yes" else "yes"]
    return ["wrong_" + words[0]] if len(words) == 1 else [" ".join(words[1:]), " ".join(words[:-1] + ["wrong_word"])]


def test_is_a_gate(g):
    assert isinstance(g, Gate)
    assert g.topic == "forces" and g.KEYS is KEYS
    assert gate() is g
    assert KEYS == {**ROW_KEYS, **VERBS} and not set(ROW_KEYS) & set(VERBS)
    assert set(CALC) < set(VERBS) and {op for ops in CALC.values() for op in ops} == set(OPERANDS)
    assert not set(OPERANDS) & set(ROW_KEYS)  # "<force> <key>" and "<verb> <operand>" never look alike


def test_determinism(g):
    a = [ln.text for ln in g.generate(random.Random(11), 400)]
    b = [ln.text for ln in g.generate(random.Random(11), 400)]
    c = [ln.text for ln in g.generate(random.Random(12), 400)]
    assert a == b and a != c
    assert len(a) == 400 == len(set(a))
    assert [ln.text for ln in ForcesGate().generate(random.Random(11), 400)] == a
    assert [ln.text for ln in g.records()] == [ln.text for ln in ForcesGate().records()]


def test_records_dense_and_short(g):
    recs = g.records()
    assert len(recs) >= len(TABLE)
    heads = [ln.text.split(".")[0] for ln in recs]
    for row in TABLE:
        assert f"{row['name']} force" in heads
    for ln in recs:
        assert ln.kind == "record" and ln.topic == "forces"
        assert is_dense(ln.text) and n_tokens(ln.text) <= 80, ln.text
        name, word = ln.text.split(".")[0].split()
        assert name in g.rows and word in ("force", "force_more")
    assert "gravity force. kind fundamental. carrier graviton_hypothetical. acts_on mass_energy. range_m infinite." in recs[0].text
    assert "constant 6 point 6 7 4 e minus 1 1." in recs[0].text


def test_split_rule_for_a_long_row():
    row = dict(TABLE[0], name="longwinded", examples=[f"example_{i}" for i in range(30)])
    two = record_lines(row)
    assert len(two) == 2
    assert two[0].startswith("longwinded force. kind fundamental.") and two[1].startswith("longwinded force_more. formula")
    assert all(is_dense(ln) and n_tokens(ln) <= 80 for ln in two)
    assert len(record_lines(TABLE[0])) == 1


def test_generated_lines_dense_and_short(lines):
    assert len(lines) == 2000
    for ln in lines:
        assert ln.topic == "forces" and ln.kind in ("fact", "calc", "yesno")
        assert "." not in ln.prompt and "." not in ln.answer
        assert is_dense(ln.text) and n_tokens(ln.text) <= 80, ln.text
        assert ln.text.startswith("q ") and ". a " in ln.text


def test_gate_agrees_with_its_own_answers(g, lines):
    for ln in lines:
        v = g.check(ln.prompt, ln.answer)
        assert v.ok and v.expected == ln.answer, (ln.text, v)


def test_gate_rejects_wrong_answers(g, lines):
    """Every single changed digit, not only the first, and every changed word."""
    tried = 0
    for ln in lines:
        for wrong in mutations(ln.answer):
            v = g.check(ln.prompt, wrong)
            assert not v.ok and v.expected == ln.answer, (ln.text, wrong, v)
            assert v.reason.startswith("expected ")
            tried += 1
    assert tried > 4000


def test_owns(g):
    for p in ["gravity carrier", "forces kind fundamental", "stronger gravity strong", "longer_range strong weak",
              "weight m 5", "pressure f 1 0 0 area 2", "check gravity carrier photon", "check weight m 5 equals 4 9",
              "static_friction formula", "centripetal m 2 v 3 r 1 point 5", "nuclear_binding range_m",
              "pressure_force unit", "weight formula", "friction mu 0 point 3 n 5 0", "friction unit",
              "check forces kind role equals centripetal", "tidal formula", "tidal m 1 m 1 r 1 d 1 0"]:
        assert g.owns(p), p
    for p in ["turkey capital", "q spoon color", "carbon protons", "water molar_mass", "calc 1 2 plus 7",
              "solve 3 x plus 4 equals 1 9", "check 1 2 plus 7 equals 1 9", "check carbon protons 6", "", "check",
              "gravity_well m 5", "photon carrier", "balance h 2 plus o 2 gives h 2 o 1", "element number 6",
              "molar_mass h 2 o 1", "count_atoms h 2 o 1", "check water formula h 2 o 1"]:
        assert not g.owns(p), p
    assert g.check("turkey capital", "ankara").reason == "not my question"
    assert g.check("turkey capital", "ankara").expected is None


def test_a_prompt_that_only_starts_with_a_force_name_is_not_owned(g):
    for p in ["gravity", "gravity seed 7 step 2 0 clumps", "gravity seed 7 final clumps", "gravity seed 5 step 1 2 next clumps",
              "gravity field", "friction is_a", "weight meaning", "weight kind", "acceleration kind", "pressure related",
              "pressure kind", "pressure unit", "normal distribution mean", "weak acid", "strong acid ph", "spring season",
              "drag queen", "magnetic field", "covalent_bond is_a", "gravity related_to mass", "stronger acid base",
              "check gravity field physics", "check pressure unit pascal", "friction", "weight", "forces", "stronger",
              "check check weight m 5 equals 4 9 point 0 5 equals yes"]:
        assert not g.owns(p), p
        assert g.check(p, "1") == g.check("turkey capital", "1")


def test_no_prompt_is_shared_with_the_earlier_rounds(g):
    """``gravity kind`` and its like have a round-3 answer in other words: neither asked nor owned."""
    from haishool.schema import QUERY_KEYS

    asked = set(g.table_prompts()["fact"])
    old: set[str] = set()
    for name in ("records-r1.jsonl", "records-r2.jsonl", "records-r3.jsonl", "records-r3-all.jsonl"):
        path = ROOT / "data" / name
        if path.exists():
            for raw in path.open(encoding="utf-8"):
                rec = json.loads(raw)
                old |= {f"{rec['obj']} {key}" for key in QUERY_KEYS if rec["values"].get(key)}
    if not old:
        pytest.skip("no round 1-3 records in this checkout")
    assert not asked & old
    assert not [p for p in old if g.owns(p)]
    kinds = {p.split()[0] for p in old if p.endswith(" kind")} & set(g.rows)
    assert kinds == set(SHARED)
    for name in SHARED:
        assert not g.owns(f"{name} kind") and not g.owns(f"check {name} kind effective")
        assert g.check(f"{name} kind", "effective").reason == "not my question"
        assert f"kind {g.rows[name]['kind']}." in record_lines(g.rows[name])[0]  # the record still says it
    for ln in g.generate(random.Random(2), 5000):
        assert ln.prompt not in old and ln.meta.get("inner") not in old
    hops = ROOT / "data" / "hops-v4b" / "hops-train-r4.txt"
    if hops.exists():
        prompts = {ln[2:].split(". a ")[0] for ln in hops.open(encoding="utf-8") if ln.startswith("q ")}
        assert len(prompts) > 1000 and not [p for p in prompts if g.owns(p)]


def test_no_prompt_is_owned_by_another_round_5_gate(g, lines):
    import importlib

    for topic in ("maths", "elements", "substances", "reactions"):
        try:
            other = importlib.import_module(f"haishool.truth.{topic}").gate()
        except Exception:  # a module another agent is still writing
            continue
        assert not [ln.prompt for ln in lines if other.owns(ln.prompt)], topic
        assert not [ln.prompt for ln in other.generate(random.Random(1), 500) if g.owns(ln.prompt)], topic


@pytest.mark.parametrize("prompt,answer", [
    ("gravity_force m 1 0 m 2 0 r 2", "3 point 3 3 7 e minus 9"),
    ("coulomb_force charge 1 e minus 6 charge 2 e minus 6 r 0 point 1", "1 point 7 9 8"),
    ("weight m 5", "4 9 point 0 5"), ("weight m 5 on moon", "8 point 1"), ("weight m 5 on earth", "4 9 point 0 5"),
    ("weight m 1 0 on mars", "3 7 point 3"), ("weight m 1 on jupiter", "2 4 point 7 9"),
    ("weight m 6 6 1 on earth", "6 4 8 4"), ("weight m 1 0 0 0", "9 8 1 0"),
    ("spring_force k 2 0 0 x 0 point 0 5", "1 0"), ("friction mu 0 point 3 n 5 0", "1 5"),
    ("pressure f 1 0 0 area 2", "5 0"), ("acceleration f 1 0 m 2", "5"),
    ("net_force f 1 0 f 2 0 opposite", "1 0"), ("net_force f 1 0 f 2 0 same", "3 0"),
    ("net_force f 5 f 5 opposite", "0"),
    ("buoyancy rho 1 0 0 0 vol 0 point 0 0 2", "1 9 point 6 2"), ("buoyancy rho 1 0 0 0 vol 1 point 8", "1 7 6 5 8"),
    ("centripetal m 2 v 3 r 1 point 5", "1 2"), ("centripetal m 3 5 v 2 7 r 0 point 5", "5 1 0 3 0"),
    ("orbital_speed m 5 point 9 7 e 2 4 r 6 point 7 7 e 6", "7 6 7 2"),
    ("escape_speed m 5 point 9 7 e 2 4 r 6 point 3 7 e 6", "1 1 1 8 0"),
    ("surface_gravity m 5 point 9 7 e 2 4 r 6 point 3 7 e 6", "9 point 8 1 9"),
    ("drag rho 1 point 2 v 1 0 c 1 area 2", "1 2 0"), ("drag rho 1 0 0 0 v 2 3 c 0 point 2 5 area 1", "6 6 1 2 5"),
    ("drag rho 1 v 4 4 c 1 point 2 area 1 0", "1 1 6 1 6"),
    ("lift rho 1 point 2 v 5 0 c 0 point 5 area 2 0", "1 5 0 0 0"),
    ("magnetic_force charge 2 v 1 0 b 0 point 5", "1 0"), ("thrust mdot 2 5 0 ve 3 0 0 0", "7 5 0 0 0 0"),
    ("tidal m 7 point 3 5 e 2 2 m 1 r 6 point 3 7 e 6 d 3 point 8 4 e 8", "1 point 1 0 4 e minus 6"),
    ("coulomb_force charge minus 1 e minus 6 charge 2 e minus 6 r 0 point 1", "minus 1 point 7 9 8"),
    ("check gravity carrier photon", "no"), ("check gravity carrier graviton_hypothetical", "yes"),
    ("check weight m 5 equals 4 9 point 0 5", "yes"), ("check weight m 5 equals 5 0", "no"),
    ("check weight m 2 0 0 equals 1 9 7 0", "no"), ("check weight m 2 0 0 equals 1 9 6 2", "yes"),
    ("check stronger gravity strong equals gravity", "no"),
    ("check forces kind fundamental equals weak strong electromagnetism gravity", "yes"),
    ("check forces kind fundamental equals weak strong gravity", "no"),
    ("check gravity described_by newton einstein", "yes"), ("check gravity described_by einstein newton", "yes"),
    ("check gravity described_by newton coulomb", "no"), ("check gravity described_by equals newton", "no"),
    ("stronger gravity strong", "strong"), ("stronger weak gravity", "weak"),
    ("stronger electromagnetism weak", "electromagnetism"),
    ("longer_range strong weak", "strong"), ("longer_range friction gravity", "gravity"),
    ("longer_range normal friction", "same"), ("longer_range gravity electromagnetism", "same"),
    ("longer_range strong friction", "friction"), ("longer_range weak normal", "normal"),
    ("longer_range nuclear_binding spring", "spring"), ("longer_range ionic_bond strong", "ionic_bond"),
    ("longer_range nuclear_binding weak", "nuclear_binding"), ("longer_range van_der_waals tidal", "tidal"),
    ("forces kind fundamental", "gravity electromagnetism strong weak"),
    ("forces carrier pion", "none"), ("forces kind role", "centripetal"), ("forces kind inertial", "centrifugal coriolis"),
    ("forces arises_from gravity", "buoyancy tidal weight"), ("forces carrier photon", "electromagnetism"),
    ("gravity carrier", "graviton_hypothetical"), ("weak carrier", "z_boson w_boson"),
    ("electromagnetism constant", "0 point 0 0 7 2 9 7 4"), ("electromagnetism constant", "7 point 2 9 7 4 e minus 3"),
    ("electromagnetism constant", "0 point 0 0 7 2 9 7 3 5"), ("gravity formula", "big_g m_1 m_2 over r squared"),
    ("buoyancy described_by", "archimedes"), ("spring described_by", "hooke"), ("strong range_m", "1 e minus 1 5"),
    ("metallic_bond range_m", "2 point 5 6 e minus 1 0"), ("ionic_bond range_m", "2 point 8 2 e minus 1 0"),
    ("covalent_bond range_m", "1 point 5 4 e minus 1 0"), ("nuclear_binding range_m", "1 point 4 1 e minus 1 5"),
    ("hydrogen_bond range_m", "2 point 7 6 e minus 1 0"), ("centripetal discovered", "1 6 7 3"),
    ("magnetic discovered", "1 8 2 0"), ("pressure_force unit", "newton"), ("pressure_force formula", "pressure area"),
    ("weight formula", "m g"), ("weight constant", "9 point 8 1"), ("weight arises_from", "gravity"),
    ("coriolis described_by", "coriolis"), ("centrifugal kind", "inertial"),
])
def test_worked_examples(g, prompt, answer):
    v = g.check(prompt, answer)
    assert v.ok, (prompt, answer, v)


@pytest.mark.parametrize("prompt,wrong,gold", [
    # a changed third, fourth or last digit is wrong, also on whole numbers and with a more exact prompt
    ("weight m 1 0 0 0", "9 8 0 0", "9 8 1 0"), ("net_force f 1 0 0 f 1 0 0 same", "2 0 1", "2 0 0"),
    ("net_force f 1 0 0 f 1 0 0 same", "1 9 9", "2 0 0"), ("net_force f 1 0 0 f 9 0 0 same", "1 0 0 4", "1 0 0 0"),
    ("acceleration f 1 0 0 0 m 1", "9 9 6", "1 0 0 0"), ("weight m 5", "4 9 point 2", "4 9 point 0 5"),
    ("weight m 5", "4 9 point 0 6", "4 9 point 0 5"), ("weight m 5", "4 9", "4 9 point 0 5"),
    ("centripetal m 3 5 v 2 7 r 0 point 5", "5 1 0 3 9", "5 1 0 3 0"),
    ("orbital_speed m 5 point 9 7 e 2 4 r 6 point 7 7 e 6", "7 6 7 0", "7 6 7 2"),
    ("orbital_speed m 5 point 9 7 e 2 4 r 6 point 7 7 e 6", "7 7 0 0", "7 6 7 2"),
    ("drag rho 1 0 0 0 v 2 3 c 0 point 2 5 area 1", "6 6 1 2 0", "6 6 1 2 5"),
    ("gravity constant", "6 point 7 e minus 1 1", "6 point 6 7 4 e minus 1 1"),
    ("gravity constant", "6 point 6 7 e minus 1 1", "6 point 6 7 4 e minus 1 1"),
    ("radiation_pressure constant", "3 e 8", "2 point 9 9 7 9 e 8"),
    ("gravity relative_strength", "1 e minus 3 8", "1 e minus 3 9"),
    ("metallic_bond range_m", "2 point 5 e minus 1 0", "2 point 5 6 e minus 1 0"),
    ("gravity discovered", "1 6 9 0", "1 6 8 7"), ("centripetal discovered", "1 6 8 7", "1 6 7 3"),
    ("gravity formula", "big_g m_2 m_1 over r squared", "big_g m_1 m_2 over r squared"),  # formulas keep their order
    # nothing infinite or untidy passes as a number, no repeat passes in a list
    ("weight m 5", "1 e 9 9 9", "4 9 point 0 5"), ("weight m 5", "minus 1 e 9 9 9", "4 9 point 0 5"),
    ("gravity constant", "9 e 9 9 9", "6 point 6 7 4 e minus 1 1"), ("strong range_m", "1 e 4 0 0", "1 e minus 1 5"),
    ("weight m 5", "0 4 9 point 0 5", "4 9 point 0 5"), ("net_force f 5 f 5 opposite", "minus 0", "0"),
    ("forces kind fundamental", "gravity gravity strong weak electromagnetism", "gravity electromagnetism strong weak"),
    ("gravity described_by", "einstein newton newton", "newton einstein"),
    ("gravity_force m 1 m 1 r 1 0 0", "6 point 6 7 4 e minus 1 4", "6 point 6 7 4 e minus 1 5"),
])
def test_wrong_answers_the_old_tolerance_let_pass(g, prompt, wrong, gold):
    v = g.check(prompt, wrong)
    assert not v.ok and v.expected == gold, (prompt, wrong, v)
    assert g.check(prompt, gold).ok


def test_numbers_are_judged_by_their_printed_digits(g):
    # more digits than the gold shows are rounded, so a more exact answer passes
    assert g.check("weight m 5", "4 9 point 0 5 1").ok and g.check("weight m 5", "4 9 point 0 5 0").ok
    assert g.check("gravity constant", "6 point 6 7 4 3 e minus 1 1").ok  # codata
    assert g.check("weight constant", "9 point 8 0 6 6 5").ok and not g.check("weight constant", "9 point 8").ok
    assert g.check("ionic_bond range_m", "2 point 8 2 0 1 e minus 1 0").ok
    # a round order-of-magnitude figure is judged at its one digit
    assert g.check("strong relative_strength", "1 point 0 0 4").ok
    assert not g.check("strong relative_strength", "2").ok
    assert g.check("strong range_m", "1 e minus 1 5").ok and not g.check("strong range_m", "1 e minus 1 4").ok
    assert g.check("check weight m 5 equals 1 e 9 9 9", "no").ok and g.check("check gravity constant 1 e 9 9 9", "no").ok
    assert same_number("2 0 0", "2 0 0 point 0 4", 4) and not same_number("2 0 0", "2 0 0 point 4", 4)
    assert same_number("1 e minus 1 5", "1 point 4 e minus 1 5") and not same_number("1 e minus 1 5", "2 e minus 1 5")
    assert not same_number("0", "minus 0") and same_number("0", "0", 4) and not same_number("0", "1 e minus 3 0", 4)
    assert tidy_number("minus 1 point 7 9 8") and tidy_number("0 point 0 5") and tidy_number("1 e minus 1 5")
    for bad in ("0 4 9", "minus 0", "1 e 9 9 9", "1 point", "point 5", "1 e 0 5", "newton", ""):
        assert not tidy_number(bad), bad
    assert matches("a b", "b a", "set") and not matches("a b", "a b b", "set") and not matches("a b", "a", "set")


def test_whole_results_are_written_in_full():
    assert calc_answer(66125.0) == "6 6 1 2 5" and calc_answer(17658.000000000004) == "1 7 6 5 8"
    assert calc_answer(15.000000000000002) == "1 5" and calc_answer(0.0) == "0"
    assert calc_answer(6484.41) == "6 4 8 4" and calc_answer(11184.8) == "1 1 1 8 0"
    assert calc_answer(1234567.0) == "1 point 2 3 5 e 6" and calc_answer(3.337e-9) == "3 point 3 3 7 e minus 9"
    assert calc_answer(-1.7976) == "minus 1 point 7 9 8"


def test_cannot_judge_but_owned(g):
    for p in ["weak formula", "centripetal range_m", "weight m 5 on pluto", "gravity_force m 1 m 2", "pressure f 1 area 0",
              "net_force f 1 f 2", "stronger gravity friction", "longer_range centripetal gravity",
              "forces relative_strength 1", "forces range_m 1", "forces range_m e", "check weight m 5",
              "check gravity carrier", "normal formula", "weak holds_together", "electrostatic carrier",
              "stronger electrostatic magnetic", "weight m minus 5", "friction mu minus 0 point 3 n 5 0",
              "pressure f 1 area minus 2", "orbital_speed m minus 1 r minus 1", "orbital_speed m 1 r 0",
              "weight m 1 e 9 9 9"]:
        v = g.check(p, "1")
        assert g.owns(p) and not v.ok and v.expected is None and v.reason.startswith("cannot judge"), (p, v)


def test_huge_operands_get_a_verdict_not_an_exception(g):
    for p in ["gravity_force m 1 m 1 r 1 e 2 0 0", "centripetal m 1 v 1 e 2 0 0 r 1", "drag rho 1 v 1 e 2 0 0 c 1 area 1",
              "lift rho 1 v 1 e 2 0 0 c 1 area 1", "surface_gravity m 1 r 1 e 2 0 0", "tidal m 1 m 1 r 1 d 1 e 2 0 0",
              "thrust mdot 1 e 3 0 0 ve 1 e 3 0 0", "check gravity_force m 1 m 1 r 1 e 2 0 0 equals 1",
              "gravity_force m 1 e 3 0 0 m 1 e 3 0 0 r 1"]:
        v = g.check(p, "1")
        assert not v.ok and v.expected is None and v.reason.startswith("cannot judge"), (p, v)


def test_compute_matches_physics():
    assert compute("gravity_force", "m 1 0 m 2 0 r 2".split()) == pytest.approx(6.674e-11 * 200 / 4)
    assert compute("escape_speed", "m 5 point 9 7 e 2 4 r 6 point 3 7 e 6".split()) == pytest.approx(
        math.sqrt(2 * 6.674e-11 * 5.97e24 / 6.37e6))
    assert compute("orbital_speed", "m 1 r 1".split()) == pytest.approx(math.sqrt(6.674e-11))
    assert compute("surface_gravity", "m 2 r 4".split()) == pytest.approx(6.674e-11 * 2 / 16)
    assert compute("tidal", "m 2 m 3 r 5 d 1 0".split()) == pytest.approx(2 * 6.674e-11 * 2 * 3 * 5 / 1000)
    assert compute("lift", "rho 2 v 3 c 5 area 7".split()) == compute("drag", "rho 2 v 3 c 5 area 7".split()) == 315
    assert compute("thrust", "mdot 3 ve 7".split()) == 21
    assert compute("buoyancy", "rho 1 0 0 0 vol 2".split()) == pytest.approx(19620)
    assert compute("coulomb_force", "charge minus 1 charge 1 r 1".split()) == pytest.approx(-8.988e9)
    assert sig4(11184.8) == 11180 and sig4(3.33700001e-9) == 3.337e-9
    for verb, words in (("weight", "m 5 m 6"), ("friction", "mu 0 point 3"), ("buoyancy", "rho 1 v 2"),
                        ("pressure", "f 1 a 2"), ("coulomb_force", "q 1 q 1 r 1"), ("weight", "m minus 1")):
        with pytest.raises(ValueError):
            compute(verb, words.split())


def test_constants_agree_with_codata(g):
    c = pytest.importorskip("scipy.constants")
    from haishool.truth.forces import K_COULOMB, LIGHT_SPEED, G
    pairs = [(G, c.G), (K_COULOMB, 1 / (4 * math.pi * c.epsilon_0)), (SURFACE_G["earth"], c.g), (LIGHT_SPEED, c.c),
             (g.rows["gravity"]["constant"], c.G), (g.rows["tidal"]["constant"], c.G),
             (g.rows["electrostatic"]["constant"], 1 / (4 * math.pi * c.epsilon_0)),
             (g.rows["ionic_bond"]["constant"], 1 / (4 * math.pi * c.epsilon_0)),
             (g.rows["magnetic"]["constant"], c.mu_0), (g.rows["radiation_pressure"]["constant"], c.c),
             (g.rows["weight"]["constant"], c.g), (g.rows["buoyancy"]["constant"], c.g),
             (g.rows["weak"]["constant"], c.physical_constants["Fermi coupling constant"][0]),
             (g.rows["electromagnetism"]["constant"], c.alpha)]
    for mine, codata in pairs:
        assert mine == pytest.approx(codata, rel=1e-3)
    # the gate accepts the codata value itself wherever the table shows fewer digits
    for name, codata in (("gravity", c.G), ("magnetic", c.mu_0), ("radiation_pressure", c.c), ("electromagnetism", c.alpha),
                         ("electrostatic", 1 / (4 * math.pi * c.epsilon_0))):
        digits = len(g.value(name, "constant").split(" e ")[0].replace(" point", "").replace("0 0 0 ", "").split())
        assert g.check(f"{name} constant", num(codata, sig=digits)).ok, name
    assert {name for name in g.names if "constant" in g.rows[name]} == {
        "gravity", "electromagnetism", "weak", "electrostatic", "magnetic", "buoyancy", "tidal", "radiation_pressure",
        "ionic_bond", "weight"}
    # yukawa: the nuclear force reaches about one pion compton wavelength, hbar c / (m_pi c^2), m_pi 139.57 MeV
    pion_range = c.hbar * c.c / (139.57e6 * c.e)
    assert g.rows["nuclear_binding"]["range_m"] == float(f"{pion_range:.3g}")


def test_bond_lengths_follow_from_lattice_constants(g):
    """Lattice constants in angstrom: rock salt 5.640, diamond 3.567, copper 3.615 (from memory, 4 digits)."""
    assert g.rows["ionic_bond"]["range_m"] == float(f"{5.640e-10 / 2:.3g}")
    assert g.rows["covalent_bond"]["range_m"] == float(f"{3.567e-10 * math.sqrt(3) / 4:.3g}")
    assert g.rows["metallic_bond"]["range_m"] == float(f"{3.615e-10 / math.sqrt(2):.3g}")
    mendeleev = pytest.importorskip("mendeleev")
    assert mendeleev.element("Cu").lattice_constant == pytest.approx(3.615, rel=5e-3)
    assert mendeleev.element("C").lattice_constant == pytest.approx(3.567, rel=5e-3)


def test_surface_gravity_agrees_with_the_bodies():
    from haishool.truth.forces import G
    for body, listed in SURFACE_G.items():
        m, r = BODIES[body]
        assert listed == pytest.approx(G * m / r ** 2, rel=1e-3), body


def test_fundamental_rows(g):
    assert [r["name"] for r in TABLE if r["kind"] == "fundamental"] == FUNDAMENTAL
    for name in FUNDAMENTAL:
        row = g.rows[name]
        assert row["carrier"] and row["acts_on"] and "relative_strength" in row and "range_m" in row
        assert "arises_from" not in row
    # relative strength and carrier are facts of the four interactions only
    assert [n for n in g.names if "relative_strength" in g.rows[n]] == FUNDAMENTAL
    assert [n for n in g.names if "carrier" in g.rows[n]] == FUNDAMENTAL
    s = {n: g.rows[n]["relative_strength"] for n in FUNDAMENTAL}
    assert s["strong"] > s["electromagnetism"] > s["weak"] > s["gravity"]
    assert g.rows["strong"]["range_m"] > g.rows["weak"]["range_m"]
    assert g.rows["gravity"]["range_m"] == g.rows["electromagnetism"]["range_m"] == "infinite"
    assert g.rows["centripetal"]["kind"] == "role"
    assert [n for n in g.names if g.rows[n]["kind"] == "inertial"] == ["centrifugal", "coriolis"]
    assert "holds_together" not in g.rows["weak"] and ("holds_together", "none") not in g.list_pairs


def test_effective_rows_arise_from_a_fundamental_interaction(g):
    for row in TABLE:
        if row["kind"] == "effective":
            assert set(row["arises_from"]) <= set(FUNDAMENTAL), row["name"]
        assert row["unit"] == "newton"
        assert row.get("attractive", "both") in ("attractive", "repulsive", "both", "neither")
        for key, value in row.items():
            assert key in ROW_KEYS or key in ("name", "notes"), (row["name"], key)
    assert g.rows["nuclear_binding"]["arises_from"] == ["strong"]
    assert g.rows["tidal"]["arises_from"] == g.rows["weight"]["arises_from"] == ["gravity"]
    assert [n for n in g.names if g.rows[n].get("range_m") == "contact"] == CONTACT
    # a formula that holds only in a special case is left out
    assert "formula" not in g.rows["normal"] and "formula" not in g.rows["tension"]
    # every row says which of its values are remembered rather than computed, so a checker can target them
    for row in TABLE:
        tail = row["notes"].rsplit("; from memory, not computed: ", 1)
        assert len(tail) == 2 and "described_by" in tail[1].split(), row["name"]
        for key in ("relative_strength", "discovered", "formula", "examples", "acts_on", "holds_together", "direction"):
            assert (key in tail[1].split()) == (key in row), (row["name"], key)
        assert ("range_m" in tail[1].split()) == (isinstance(row.get("range_m"), (int, float))), row["name"]


def test_formulas_use_one_token_for_one_quantity(g):
    words = Counter(w for row in TABLE for w in row.get("formula", "").split())
    assert not {"q", "a", "e", "p", "1", "3", "7 "} & set(words)  # no markers, no bare exponents or subscripts
    assert g.rows["gravity"]["formula"] == "big_g m_1 m_2 over r squared"
    assert g.rows["tidal"]["formula"] == "2 big_g m_1 m_2 r over d cubed"
    assert g.rows["weight"]["formula"] == "m g" and g.rows["buoyancy"]["formula"] == "rho g vol"
    assert g.rows["electrostatic"]["formula"] == g.rows["ionic_bond"]["formula"] == "k_e charge_1 charge_2 over r squared"
    assert g.rows["spring"]["formula"] == "k x" and g.rows["radiation_pressure"]["formula"] == "power over c"
    assert g.rows["centripetal"]["formula"] == g.rows["centrifugal"]["formula"] == "m v squared over r"
    for ops in CALC.values():
        assert not {"q", "a", "e"} & set(ops)


def test_comparisons_are_antisymmetric(g):
    for verb in ("stronger", "longer_range"):
        pool = [n for n in g.names if ("relative_strength" if verb == "stronger" else "range_m") in g.rows[n]]
        for a in pool:
            for b in pool:
                if a != b:
                    x, _, _ = g.expected(f"{verb} {a} {b}")
                    y, _, _ = g.expected(f"{verb} {b} {a}")
                    assert x == y and x in (a, b, "same", None), (verb, a, b, x, y)
                    assert (x is not None) == (f"{verb} {a} {b}" in g.comparisons[verb])


def test_ranges_are_compared_by_order_of_magnitude_only(g):
    assert CONTACT_RANGE_M == 1e-10 and RANGE_FACTOR == 100
    nuclear = ["strong", "weak", "nuclear_binding"]
    for contact in CONTACT:
        for n in nuclear:  # touching atoms are 1e-10 m apart, the nuclear forces reach 1e-15 m and less
            assert g.expected(f"longer_range {n} {contact}")[0] == contact
            assert g.expected(f"longer_range {contact} {n}")[0] == contact
        for bond in BONDS + ["van_der_waals"]:  # the same scale: not compared
            assert g.expected(f"longer_range {contact} {bond}")[0] is None
        assert g.expected(f"longer_range {contact} gravity")[0] == "gravity"
    for a in BONDS + ["van_der_waals"]:
        for b in BONDS + ["van_der_waals"]:
            if a != b:  # one textbook example each is no ordering of the bond types
                v = g.check(f"longer_range {a} {b}", a)
                assert not v.ok and v.expected is None and v.reason.startswith("cannot judge")
        for n in nuclear:
            assert g.expected(f"longer_range {a} {n}")[0] == a
    assert g.expected("longer_range strong nuclear_binding")[0] is None  # 1e-15 against 1.41e-15
    assert g.expected("longer_range strong weak")[0] == "strong"
    assert range_order("infinite", "contact") == 1 and range_order("contact", "infinite") == -1
    assert range_order("contact", "contact") == 0 and range_order("1 e minus 1 5", "contact") == -1
    assert range_order("2 point 8 2 e minus 1 0", "contact") is None and range_order("newton", "contact") is None
    for prompt in g.comparisons["longer_range"]:
        a, b = (g.value(n, "range_m") for n in prompt.split()[1:])
        assert a == b or range_order(a, b) in (1, -1)


def test_lists_name_matching_forces_only(g):
    assert g.list_pairs
    for key, value in g.list_pairs:
        names = g.expected(f"forces {key} {value}")[0].split()
        assert 1 <= len(names) <= MAX_LIST
        for name in names:
            assert value in g.value(name, key).split()
        for name in set(g.names) - set(names):
            assert g.value(name, key) is None or value not in g.value(name, key).split()


def test_proportions(g):
    n = 600
    sample = g.generate(random.Random(5), n)
    for sub, share in PROPORTIONS.items():
        got = sum(ln.meta["sub"] == sub for ln in sample) / n
        assert abs(got - share) < 0.05, (sub, got, share)
    kinds = {ln.kind for ln in sample}
    assert kinds == {"fact", "calc", "yesno"}
    verbs = {ln.meta["verb"] for ln in g.generate(random.Random(5), 2000) if ln.kind == "calc"}
    assert verbs == set(CALC)


def test_yes_and_no_stay_balanced_in_every_kind_of_check(g, big):
    for n in (300, 1000, 20000):
        yesno = [ln for ln in (big if n == 20000 else g.generate(random.Random(8), n)) if ln.kind == "yesno"]
        count = Counter((ln.meta["of"], ln.answer) for ln in yesno)
        assert {of for of, _ in count} == {"fact", "calc", "compare", "list"}
        for of in ("fact", "calc", "compare", "list"):
            assert abs(count[of, "yes"] - count[of, "no"]) <= 1, (n, of, count)
    for ln in yesno:
        words = ln.prompt.split()
        assert ln.meta["inner"] and ln.prompt.startswith("check " + ln.meta["inner"] + " ")
        assert g.expected(ln.meta["inner"])[0] is not None
        if words[1] in ("stronger", "longer_range"):  # check stronger a b equals <a, b or same>
            assert words[5] in (words[2], words[3], "same"), ln.text


def test_a_wrong_value_is_never_part_of_the_truth(g, big):
    """No ``no`` for a statement that is partly or arguably true (``check gravity described_by newton``)."""
    for ln in big:
        if ln.kind != "yesno" or ln.meta["of"] != "fact":
            continue
        name, key = ln.meta["name"], ln.meta["key"]
        assert key not in UNCHECKED_KEYS and key not in SHARED.get(name, ())
        gold, value = g.value(name, key), " ".join(ln.prompt.split()[3:])
        if ln.answer == "yes":
            assert value == gold
            continue
        if key in SET_KEYS:
            assert not set(value.split()) & set(gold.split()), ln.text
        if key == "discovered":
            assert value not in [" ".join(y) for y in re.findall(r"\b[0-9]{4}\b", g.rows[name]["notes"])], ln.text
        if key == "range_m":
            assert range_order(gold, value) in (1, -1), ln.text
        if key == "constant":
            ratio = parse_num(value) / parse_num(gold)
            assert ratio >= 1.99 or ratio <= 0.501, ln.text
    for name, key in g.facts:
        gold = set(g.value(name, key).split())
        for wrong in g.wrong_values(name, key):
            assert wrong != g.value(name, key) and g.expected(f"check {name} {key} {wrong}")[0] == "no"
            if key in SET_KEYS:
                assert not set(wrong.split()) & gold, (name, key, wrong)
            if key in ("holds_together", "described_by", "formula"):  # nor true of any force of its family
                for kin in g.names:
                    if g.related(name, kin) and g.value(kin, key) is not None:
                        assert wrong != g.value(kin, key), (name, key, wrong, kin)
                        assert key == "formula" or not set(wrong.split()) & set(g.value(kin, key).split()), (name, key, wrong, kin)
    assert g.wrong_values("centripetal", "arises_from") == [] and g.wrong_values("gravity", "acts_on") == []
    assert g.wrong_values("spring", "attractive") == ["neither"]  # both is not wrong by attractive or repulsive
    assert "infinite" not in g.wrong_values("ionic_bond", "range_m") + g.wrong_values("van_der_waals", "range_m")
    assert "infinite" in g.wrong_values("strong", "range_m") and "infinite" in g.wrong_values("friction", "range_m")
    assert g.wrong_values("electrostatic", "described_by") and "oersted ampere lorentz" not in g.wrong_values(
        "electrostatic", "described_by")
    for year in ("1 6 6 3", "1 6 4 3"):
        assert year not in g.wrong_values("pressure_force", "discovered")
    assert "1 9 3 4" not in g.wrong_values("weak", "discovered") and "1 8 0 6" not in g.wrong_values("surface_tension", "discovered")
    assert g.wrong_values("gravity", "relative_strength") == ["0 point 0 1", "1", "1 e minus 6"]


def test_part_of_a_list_is_not_called_wrong(g):
    for p in ["check gravity described_by newton", "check buoyancy arises_from gravity", "check friction described_by coulomb",
              "check buoyancy arises_from electromagnetism", "check lift acts_on bodies_in_fluids",
              "check van_der_waals acts_on atoms", "check weak carrier w_boson"]:
        for answer in ("yes", "no"):
            v = g.check(p, answer)
            assert g.owns(p) and not v.ok and v.expected is None and v.reason.startswith("cannot judge"), (p, v)
    assert g.check("check gravity described_by newton einstein", "yes").ok
    assert g.check("check gravity described_by newton einstein coulomb", "no").ok  # coulomb did not
    assert g.check("check friction described_by newton", "no").ok


def test_a_wrong_list_drops_or_adds_exactly_one_name(g):
    rng = random.Random(3)
    drops = adds = 0
    for _ in range(3000):
        key, value = rng.choice(g.list_pairs)
        inner = f"forces {key} {value}"
        gold = g.expected(inner)[0].split()
        wrong = g._wrong_answer(rng, inner, " ".join(gold), "set", "list").split()
        assert len(wrong) == len(set(wrong)) and wrong
        if len(wrong) == len(gold) - 1:
            assert set(wrong) < set(gold)
            drops += 1
        else:
            assert len(wrong) == len(gold) + 1 and set(gold) < set(wrong)
            adds += 1
        assert g.expected(f"check {inner} equals {' '.join(wrong)}")[0] == "no"
    assert drops > 100 and adds > 1500


def test_orbits_stay_newtonian(g):
    lines = [ln for ln in g.generate(random.Random(4), 20000) if ln.meta.get("verb") in ("orbital_speed", "escape_speed")]
    assert len(lines) > 1000
    for ln in lines:
        if ln.kind == "calc":
            assert parse_num(ln.answer) < 3e7, ln.text
    # a body inside its own schwarzschild radius: newton would say faster than light, the gate says nothing
    v = g.check("escape_speed m 9 point 9 e 3 0 r 1 e 4", "3 point 6 3 5 e 8")
    assert not v.ok and v.expected is None and "faster than light" in v.reason
    assert g.check("escape_speed m 1 point 9 9 e 3 0 r 6 point 9 6 e 8", "6 1 7 8 0 0").ok  # the sun


def test_many_distinct_lines(g):
    # every large run holds the whole table (all facts and lists), so runs overlap there by design
    seen = set()
    for seed in range(1, 4):
        seen |= {ln.text for ln in g.generate(random.Random(seed), 10000)}
    assert len(seen) >= 20000
    one = g.generate(random.Random(99), 20000)
    assert len({ln.text for ln in one}) == 20000
    assert all(is_dense(ln.text) and n_tokens(ln.text) <= 80 for ln in one)
    assert all(g.check(ln.prompt, ln.answer).ok for ln in one)
    for n in (0, 1, 2, 5):
        assert len(g.generate(random.Random(n), n)) == n


def test_a_large_run_asks_the_whole_table_once(g):
    pools = g.table_prompts()
    assert {k: len(v) for k, v in pools.items()} == {"fact": 365, "stronger": 12, "longer_range": 550, "list": 230}
    for row in TABLE:
        for key in ROW_KEYS:
            assert (f"{row['name']} {key}" in pools["fact"]) == (key in row and key not in SHARED.get(row["name"], ()))
    for seed in (1, 2, 99):
        prompts = [ln.prompt for ln in g.generate(random.Random(seed), 5000)]
        assert len(prompts) == len(set(prompts))
        assert set(pools["fact"]) | set(pools["list"]) | set(pools["stronger"]) <= set(prompts)
    assert set(pools["longer_range"]) <= {ln.prompt for ln in g.generate(random.Random(1), 20000)}
    # a small run still mixes both comparison verbs
    small = [ln.meta["verb"] for ln in g.generate(random.Random(3), 600) if ln.meta["sub"] == "compare"]
    assert small.count("stronger") >= 10 and small.count("longer_range") >= 10


def test_docstring_examples_are_what_the_gate_says(g):
    import haishool.truth.forces as module
    pairs = re.findall(r"q ([a-z0-9_ ]+)\. a ([a-z0-9_ ]+)\.", module.__doc__)
    assert len(pairs) >= 30
    for prompt, answer in pairs:
        v = g.check(prompt, answer)
        assert v.ok and v.expected == answer, (prompt, answer, v)
    shown = re.findall(r"^    (strong force\..*(?:\n        .*)*)", module.__doc__, flags=re.M)
    assert len(shown) == 1 and " ".join(shown[0].split()) == record_lines(g.rows["strong"])[0]
    assert max(n_tokens(ln.text) for ln in g.records()) == 73  # "the longest, gravity, is 73 tokens"
    counts = {k: len(v) for k, v in g.table_prompts().items()}
    assert (f"({counts['fact']} facts, {counts['stronger'] + counts['longer_range']} comparisons, {counts['list']} lists)"
            in module.__doc__)


def test_jsonl_matches_table():
    rows = [json.loads(ln) for ln in JSONL.open(encoding="utf-8")]
    assert len(rows) == len(TABLE) == 31
    for row, got in zip(TABLE, rows):
        assert got["lines"] == record_lines(row)
        assert {k: v for k, v in got.items() if k != "lines"} == row
        assert all(is_dense(ln) and n_tokens(ln) <= 80 for ln in got["lines"])


def test_a_fresh_gate_validates_its_table():
    with pytest.raises(ValueError):
        ForcesGate([dict(TABLE[0], kind="Fundamental")])
    with pytest.raises(ValueError):
        ForcesGate([dict(TABLE[4], arises_from=["magic"])])
    with pytest.raises(ValueError):
        ForcesGate([TABLE[0], TABLE[0]])
    with pytest.raises(ValueError):
        ForcesGate(TABLE[:4] + [dict(TABLE[4], carrier="photon")])  # only an interaction has a carrier
    with pytest.raises(ValueError):
        ForcesGate([dict(TABLE[0], range="infinite")])  # the key is range_m
