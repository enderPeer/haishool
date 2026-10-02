import json
import random
import re
import warnings
from collections import Counter
from pathlib import Path

import pytest

from haishool.truth import Line, Verdict, is_dense, num, parse_num
from haishool.truth.elements import (FACT_KEYS, FLOAT_KEYS, INT_KEYS, KEYS, LIST_KEYS, MASS_TOLERANCE, MAX_LIST,
                                     ORDER_KEYS, RECORD_KEYS, RECORD_MORE_KEYS, RULE_KEYS, TOLERANCE, _sig5, _signed,
                                     build, dense_value, gate)

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "data" / "truth-v5" / "elements.jsonl"
MAX_TOKENS = 80
NOBLE = (2, 10, 18, 36, 54, 86, 118)  # the last atomic number of each period
BULK_KEYS = ("state", "melting_k", "boiling_k", "sublimation_k", "density_g_cm3")


def tokens(line: str) -> int:
    return len(line.replace(".", " . ").split())


def flip(line: Line) -> str:
    """A wrong answer: the first significant digit changed, or the first word swapped."""
    words = line.answer.split()
    for i, w in enumerate(words):
        if w.isdigit() and w != "0":
            words[i] = "9" if w != "9" else "8"
            return " ".join(words)
        if w == "0" and len(words) == 1:
            return "1"
    words[0] = "no" if words[0] == "yes" else "yes" if words[0] == "no" else "xenon" if words[0] != "xenon" else "argon"
    return " ".join(words)


def position(z: int) -> tuple[int, int | str]:
    """(period, group) from the atomic number alone, by the shape of the periodic table."""
    period = next(i for i, last in enumerate(NOBLE, 1) if z <= last)
    offset = z - (NOBLE[period - 2] + 1 if period > 1 else 1)
    if period == 1:
        return period, 1 if offset == 0 else 18
    if period <= 3:
        return period, offset + 1 if offset < 2 else offset + 11
    if period <= 5:
        return period, offset + 1
    if offset < 3:
        return period, offset + 1
    if offset < 17:
        return period, "lanthanide" if period == 6 else "actinide"
    return period, offset - 13


def electrons_in(econf: str) -> int:
    cores = {"he": 2, "ne": 10, "ar": 18, "kr": 36, "xe": 54, "rn": 86}
    return sum(cores[t] if t in cores else int(re.fullmatch(r"\d[spdf](\d+)", t)[1]) for t in econf.split())


@pytest.fixture(scope="module")
def g():
    return gate(TABLE)


@pytest.fixture(scope="module")
def lines(g):
    return g.generate(random.Random(1), 4000)


@pytest.fixture(scope="module")
def many(g):
    return g.generate(random.Random(4), 30000)


def test_table_is_complete_and_consistent():
    with TABLE.open(encoding="utf-8") as f:
        rows = [json.loads(ln) for ln in f]
    assert len(rows) == 118
    assert sorted(r["number"] for r in rows) == list(range(1, 119))
    assert len({r["name"] for r in rows}) == 118 and len({r["symbol"] for r in rows}) == 118
    for r in rows:
        z = r["number"]
        for key in RECORD_KEYS:
            assert key in r, (r["name"], key)
        assert r["protons"] == z == r["electrons"]
        assert r["neutrons"] == r["mass_number"] - r["protons"]
        assert abs(r["mass"] - r["mass_number"]) < 3
        assert (r["period"], r["group"]) == position(z), r["name"]  # recomputed from the number alone
        assert r["block"] in "spdf"
        assert (r["block"] == "f") == (r["group"] in ("lanthanide", "actinide"))
        assert r["metal"] == ("nonmetal" if r["category"] in ("nonmetal", "halogen", "noble_gas")
                              else "metalloid" if r["category"] == "metalloid" else "metal")
        assert (r["category"] == "lanthanide") == (57 <= z <= 71)
        assert (r["category"] == "actinide") == (89 <= z <= 103)
        if r["category"] == "transition_metal":
            assert r["block"] == "d"
        if z >= 84 or r["symbol"] in ("tc", "pm"):
            assert r["radioactive"] == "yes"
        if isinstance(r["mass"], int):  # no standard atomic weight: the mass number stands in
            assert r["mass"] == r["mass_number"] and r["radioactive"] == "yes"
            assert "no standard atomic weight" in r["notes"]
        else:
            assert r["mass"] != int(r["mass"]) and _sig5(r["mass"]) == r["mass"]
        if "melting_k" in r and "boiling_k" in r:
            assert r["melting_k"] <= r["boiling_k"], r["name"]
        if "sublimation_k" in r:
            assert "melting_k" not in r and "boiling_k" not in r and "sublimes" in r["notes"]
        if "state" in r:
            if r["state"] == "solid":
                assert r.get("melting_k", r.get("sublimation_k")) > 293
            if r["state"] == "liquid":
                assert r["melting_k"] <= 293 < r["boiling_k"]
            if r["state"] == "gas":
                assert r["boiling_k"] <= 293
        if z in (85, 87) or z >= 100:  # never weighed: the source only has predictions
            assert not any(k in r for k in BULK_KEYS), r["name"]
            assert "predictions" in r["notes"]
        if "valence" in r:
            assert r["valence"] == sorted(set(r["valence"])) and len(r["valence"]) >= 1
            assert set(r["valence"]) <= set(r["valence_known"])
        assert electrons_in(r["econf"]) == z, r["name"]
        for key in r:
            assert key in KEYS or key in ("name", "notes", "valence_known"), key
        assert all(k != "notes" or isinstance(r[k], str) for k in r)
        for key in FLOAT_KEYS - {"mass"}:
            if key in r:
                assert isinstance(r[key], float) and _sig5(r[key]) == r[key], (r["name"], key)
    assert sum(r["category"] == "noble_gas" for r in rows) == 7
    assert sum(r["category"] == "lanthanide" for r in rows) == 15
    assert sum(r["category"] == "actinide" for r in rows) == 15
    assert [r["name"] for r in rows if r.get("state") == "liquid"] == ["bromine", "mercury"]
    assert [r["name"] for r in rows if "sublimation_k" in r] == ["carbon", "arsenic"]
    assert sum(isinstance(r["mass"], int) for r in rows) == 34
    for name in ("neodymium", "yttrium", "radon", "nihonium", "darmstadtium", "roentgenium"):
        assert "from memory" in next(r for r in rows if r["name"] == name)["notes"]


def test_reference_values_checked_by_hand(g):
    """Values checked against a printed periodic table, not against the gate's own source."""
    masses = {"hydrogen": 1.008, "carbon": 12.011, "nitrogen": 14.007, "oxygen": 15.999, "sodium": 22.99,
              "chlorine": 35.45, "vanadium": 50.942, "iron": 55.845, "copper": 63.546, "silver": 107.87,
              "ytterbium": 173.05, "gold": 196.97, "lead": 207.2, "uranium": 238.03}
    isotopes = {"hydrogen": 1, "helium": 4, "carbon": 12, "nitrogen": 14, "oxygen": 16, "chlorine": 35, "argon": 40,
                "potassium": 39, "iron": 56, "nickel": 58, "copper": 63, "zinc": 64, "selenium": 80, "tin": 120,
                "tellurium": 130, "xenon": 132, "lead": 208, "bismuth": 209, "uranium": 238,
                "technetium": 98, "promethium": 145, "polonium": 209, "radon": 222, "radium": 226, "plutonium": 244,
                "darmstadtium": 281, "flerovium": 289, "oganesson": 294}
    years = {"neodymium": 1885, "praseodymium": 1885, "yttrium": 1794, "radon": 1900, "nihonium": 2004,
             "oxygen": 1774, "hydrogen": 1766, "polonium": 1898}
    for name, mass in masses.items():
        assert g.by_name[name]["mass"] == mass, name
    for name, a in isotopes.items():
        assert g.by_name[name]["mass_number"] == a, name
    for name, year in years.items():
        assert g.by_name[name]["discovered"] == year, name
    assert g.by_name["chlorine"]["valence"] == [-1, 1, 3, 5, 7] == g.by_name["iodine"]["valence"]
    assert g.by_name["phosphorus"]["melting_k"] == 317.3 and g.by_name["phosphorus"]["boiling_k"] == 553.65
    assert "melting_k" not in g.by_name["arsenic"] and g.by_name["arsenic"]["sublimation_k"] == 889.15
    assert g.by_name["boron"]["boiling_k"] == 4273.2 and g.by_name["iron"]["melting_k"] == 1811.2  # ties round up
    assert g.groups[("state", "gas")] == ["hydrogen", "helium", "nitrogen", "oxygen", "fluorine", "neon", "chlorine",
                                           "argon", "krypton", "xenon", "radon"]
    # a gas density is the ideal-gas value at 25 C and 1 atm: molar mass / 24.465 l
    for name, atoms in (("hydrogen", 2), ("helium", 1), ("nitrogen", 2), ("oxygen", 2), ("argon", 1)):
        r = g.by_name[name]
        assert r["density_g_cm3"] == pytest.approx(atoms * r["mass"] / 24465, rel=0.01), name


def test_table_is_a_fresh_build(tmp_path):
    pytest.importorskip("mendeleev")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        build(tmp_path / "elements.jsonl")
    assert (tmp_path / "elements.jsonl").read_text(encoding="utf-8") == TABLE.read_text(encoding="utf-8")


def test_sig5_rounds_decimal_ties_up():
    assert _sig5(50.9415) == 50.942 and _sig5(173.045) == 173.05
    assert _sig5(4273.15) == 4273.2 and _sig5(1811.15) == 1811.2 and _sig5(1000.15) == 1000.2
    assert _sig5(286.34999999999997) == 286.35 and _sig5(8.2e-05) == 8.2e-05 and _sig5(55.845) == 55.845


def test_records_are_two_dense_short_lines_per_element(g):
    recs = g.records()
    assert len(recs) == 236
    for ln in recs:
        assert ln.kind == "record" and ln.topic == "elements"
        assert is_dense(ln.text) and tokens(ln.text) <= MAX_TOKENS, ln.text
        assert "valence_known" not in ln.text and "notes" not in ln.text
    iron = [ln.text for ln in recs if ln.text.startswith("iron element")]
    assert iron[0] == ("iron element. number 2 6. symbol fe. protons 2 6. electrons 2 6. neutrons 3 0. "
                       "mass_number 5 6. mass 5 5 point 8 4 5. group 8. period 4. block d. "
                       "category transition_metal. metal metal. radioactive no.")
    assert iron[1] == ("iron element_more. state solid. melting_k 1 8 1 1 point 2. boiling_k 3 1 3 4 point 2. "
                       "density_g_cm3 7 point 8 7. electronegativity 1 point 8 3. econf ar 3d6 4s2. "
                       "valence plus 2 plus 3. discovered ancient.")
    texts = {ln.text for ln in recs}
    assert ("carbon element_more. state solid. sublimation_k 4 0 9 8 point 2. density_g_cm3 2 point 2. "
            "electronegativity 2 point 5 5. econf he 2s2 2p2. valence minus 4 plus 4. discovered ancient.") in texts
    assert "oganesson element_more. econf rn 5f14 6d10 7s2 7p6. discovered 2 0 0 2." in texts
    # every key of a row appears in exactly one of its two lines
    for r in g.rows:
        both = " ".join(ln.text for ln in recs if ln.meta["element"] == r["name"])
        for key in KEYS:
            if key in r:
                assert f" {key} {dense_value(r, key)}." in both, (r["name"], key)


def test_generate_is_deterministic(g):
    a = [ln.text for ln in g.generate(random.Random(7), 500)]
    b = [ln.text for ln in g.generate(random.Random(7), 500)]
    c = [ln.text for ln in g.generate(random.Random(8), 500)]
    assert a == b and len(a) == 500 and len(set(a)) == 500
    assert a != c


def test_generated_lines_are_dense_short_and_mixed(lines, many):
    kinds = Counter(ln.kind for ln in lines)
    assert set(kinds) == {"fact", "calc", "yesno"}
    assert 0.25 <= kinds["fact"] / len(lines) <= 0.45
    assert 0.25 <= kinds["calc"] / len(lines) <= 0.45
    assert 0.2 <= kinds["yesno"] / len(lines) <= 0.4
    heads = Counter(ln.prompt.split()[0] for ln in lines)
    assert {"element", "elements", "count", "heavier", "lighter", "higher", "lower", "neutrons", "check"} <= set(heads)
    for batch in (lines, many):
        for ln in batch:
            assert ln.topic == "elements"
            assert is_dense(ln.text) and tokens(ln.text) <= MAX_TOKENS, ln.text
            assert "." not in ln.prompt and "." not in ln.answer
        # yes and no stay level however many lines are asked for
        yes_no = [ln.answer for ln in batch if ln.kind == "yesno"]
        assert 0.45 <= yes_no.count("yes") / len(yes_no) <= 0.55
        value = [ln.answer for ln in batch if ln.kind == "yesno" and "key" in ln.meta]
        assert abs(value.count("yes") - value.count("no")) <= 1
    assert len(many) == 30000 and len({ln.text for ln in many}) == 30000
    assert len({ln.prompt for ln in many}) == 30000  # no prompt with two answers


def test_gate_agrees_with_its_own_answers(g, lines, many):
    for ln in lines + many:
        v = g.check(ln.prompt, ln.answer)
        assert v.ok and v.expected == ln.answer, (ln.text, v)
        assert g.owns(ln.prompt), ln.prompt


def test_gate_rejects_wrong_answers(g, lines, many):
    for ln in lines + many[:6000]:
        wrong = flip(ln)
        assert wrong != ln.answer
        v = g.check(ln.prompt, wrong)
        assert not v.ok and v.expected == ln.answer, (ln.text, wrong, v)


def test_gate_rejects_near_misses(g, many):
    """A whole number one off is wrong, also for the mass numbers that stand in for a weight."""
    seen = 0
    for ln in many:
        n = parse_num(ln.answer)
        key = ln.meta.get("key")
        if not isinstance(n, int) or (key in FLOAT_KEYS and isinstance(g.by_name[ln.meta["element"]][key], float)):
            continue
        seen += 1
        for wrong in (num(n + 1), "0 " + ln.answer, ln.answer + " point 0", ln.answer + " point"):
            v = g.check(ln.prompt, wrong)
            assert not v.ok and v.expected == ln.answer, (ln.text, wrong, v)
    assert seen > 1000


def test_owns(g):
    for prompt in ("iron protons", "helium category", "oganesson mass", "element number 2 6", "element symbol fe",
                   "element after iron", "element before iron", "elements group 1 8", "elements period 1",
                   "count elements category noble_gas", "heavier iron copper", "lighter iron copper",
                   "check iron protons 2 7", "check helium category noble_gas", "check iron heavier_than copper",
                   "element group 8 period 4", "higher melting_k iron copper", "lower number iron copper",
                   "neutrons mass_number 5 6 protons 2 6", "carbon sublimation_k"):
        assert g.owns(prompt), prompt
    for prompt in ("turkey capital", "q spoon color", "calc 1 2 plus 7", "solve 3 x plus 4 equals 1 9",
                   "check 1 2 plus 7 equals 2 0", "water molar_mass", "check water formula h 2 o", "gold color",
                   "iron", "iron made", "element", "elements", "count elements", "heavier iron", "iron protons 2 6",
                   "gravity seed 7 step 2 0 clumps", "iron compound water", "lighter color", "higher iron copper",
                   "higher mass iron copper", "higher melting_k iron water", "neutrons", "neutrons mass_number 5 6",
                   "iron valence_known", "iron notes", "heavier_molecule water methane", "chem valence c",
                   "force gravity carrier", "balance h 2 plus o 2"):
        assert not g.owns(prompt), prompt
    assert g.check("turkey capital", "ankara") == Verdict(False, None, "not my question")
    assert g.check("helium electronegativity", "1").expected is None  # no value in the table
    assert g.check("element number 1 1 9", "x").expected is None
    for prompt in ("iron", "element", "check iron", "neutrons mass_number 5 protons", "", "check", "higher"):
        assert g.expected(prompt) is None  # never raises on a prompt that is not the gate's


def test_other_gates_agree_on_shared_prompts(g):
    """``<name> state`` is also a substances prompt for the elements that are substances too
    (iron, oxygen ...); the two gates must then give the same answer."""
    try:
        from haishool.truth import substances
        other = substances.gate()
    except Exception as e:  # another topic's module is being rewritten
        pytest.skip(f"substances gate not loadable: {e}")
    shared = [f"{r['name']} {key}" for r in g.rows for key in KEYS if key in r and other.owns(f"{r['name']} {key}")]
    assert all(p.endswith(" state") for p in shared), shared
    for prompt in shared:
        answer = g.check(prompt, "x").expected
        assert other.check(prompt, answer).ok, (prompt, answer)


def test_masses_agree_with_the_substances_weights(g):
    path = ROOT / "data" / "truth-v5" / "atomic_weights.json"
    if not path.exists():
        pytest.skip("no atomic_weights.json")
    weights = {k.lower(): v for k, v in json.loads(path.read_text(encoding="utf-8")).items()}
    differ = set()
    for r in g.rows:
        w = weights[r["symbol"]]
        if isinstance(r["mass"], float):
            assert r["mass"] == _sig5(w), r["name"]
        elif r["mass"] != w:
            differ.add(r["name"])
    # the three bracket isotopes the isotope table shows to be outlived beyond doubt
    assert differ <= {"meitnerium", "roentgenium", "moscovium"}, differ


def test_known_values(g):
    assert g.check("iron protons", "2 6").ok
    assert g.check("iron neutrons", "3 0").ok
    assert g.check("iron electrons", "2 6").ok
    assert g.check("iron mass", "5 5 point 8 4 5").ok
    assert g.check("iron symbol", "fe").ok
    assert g.check("iron econf", "ar 3d6 4s2").ok
    assert g.check("iron valence", "plus 3 plus 2").ok  # oxidation states compare as sets
    assert g.check("chlorine valence", "minus 1 plus 1 plus 3 plus 5 plus 7").ok
    assert not g.check("chlorine valence", "minus 1 plus 1 plus 3 plus 5").ok
    assert g.check("element number 2 6", "iron").ok
    assert g.check("element symbol fe", "iron").ok
    assert g.check("element after iron", "cobalt").ok
    assert g.check("element before iron", "manganese").ok
    assert g.check("element group 8 period 4", "iron").ok
    assert g.check("element group 3 period 6", "lanthanum").ok
    assert g.check("element group lanthanide period 6", "cerium").expected is None  # fourteen elements there
    assert g.check("elements group 1 8", "helium neon argon krypton xenon radon oganesson").ok
    assert not g.check("elements group 1 8", "neon helium argon krypton xenon radon oganesson").ok  # ordered
    assert g.check("elements period 1", "hydrogen helium").ok
    assert g.check("count elements category noble_gas", "7").ok
    assert g.check("count elements category lanthanide", "1 5").ok
    assert g.check("count elements category actinide", "1 5").ok
    assert g.check("lutetium category", "lanthanide").ok and g.check("lawrencium category", "actinide").ok
    assert g.check("heavier iron copper", "copper").ok
    assert g.check("lighter iron copper", "iron").ok
    assert g.check("heavier argon potassium", "argon").ok  # a real inversion
    assert g.check("higher melting_k iron copper", "iron").ok
    assert g.check("lower density_g_cm3 iron gold", "iron").ok
    assert g.check("higher number bohrium hassium", "hassium").ok
    assert g.check("neutrons mass_number 5 6 protons 2 6", "3 0").ok
    assert g.check("neutrons mass_number 1 protons 1", "0").ok
    assert g.check("neutrons mass_number 2 6 protons 5 6", "x").expected is None
    assert g.check("check iron protons 2 7", "no").ok
    assert g.check("check helium category noble_gas", "yes").ok
    assert g.check("check iron heavier_than copper", "no").ok
    assert g.check("check iron lighter_than copper", "yes").ok
    assert g.check("hydrogen neutrons", "0").ok and g.check("hydrogen state", "gas").ok
    assert g.check("carbon discovered", "ancient").ok and g.check("mercury state", "liquid").ok
    assert g.check("technetium radioactive", "yes").ok and g.check("bismuth radioactive", "yes").ok
    assert g.check("neodymium discovered", "1 8 8 5").ok and g.check("oganesson neutrons", "1 7 6").ok
    assert g.check("carbon sublimation_k", "4 0 9 8 point 2").ok
    assert g.check("arsenic melting_k", "1 0 9 0 point 2").expected is None  # it sublimes at 1 atm
    assert g.check("francium state", "solid").expected is None  # never seen in bulk
    assert g.check("check francium state solid", "no").expected is None
    assert g.check("nobelium symbol", "no").ok and g.check("check nobelium symbol no", "yes").ok


def test_check_tolerance(g):
    assert g.check("iron mass", "5 5 point 8 5").ok  # 0.009 % off: within 0.05 %
    assert g.check("iron mass", "5 5 point 8 6").ok  # 0.03 % off
    assert not g.check("iron mass", "5 5 point 8").ok  # 0.08 % off
    assert not g.check("iron mass", "5 6 point 2").ok
    assert not g.check("iron mass", "fifty").ok
    assert g.check("iron melting_k", "1 8 1 1").ok and g.check("iron density_g_cm3", "7 point 9").ok  # within 0.5 %
    assert not g.check("iron melting_k", "1 8 3 0").ok  # 1 % off
    assert g.check("iron protons", "2 6").ok and not g.check("iron protons", "2 7").ok  # integers exact
    assert not g.check("hydrogen discovered", "1 7 6 7").ok
    assert not g.check("iron number", "26").ok  # not digit tokens
    assert g.check("check iron mass 5 5 point 8 5", "yes").ok  # the same tolerance inside a check
    assert g.check("check iron mass 5 5 point 8", "no").ok
    assert g.check("heavier iron copper", "copper").expected == "copper"
    assert g.check("heavier iron copper", "iron") == Verdict(False, "copper", "expected copper")
    # a neighbour's weight never passes, and a mass number that stands in for a weight is exact
    assert not g.check("nickel mass", dense_value(g.by_name["cobalt"], "mass")).ok
    assert not g.check("uranium mass", "2 3 7").ok and not g.check("lead mass", "2 0 8").ok
    assert not g.check("gold mass", "1 9 6").ok
    assert g.check("polonium mass", "2 0 9").ok and not g.check("polonium mass", "2 0 8").ok
    assert not g.check("plutonium mass", "2 4 3").ok and not g.check("polonium mass", "2 0 9 point 1").ok
    assert g.check("check oganesson mass 2 9 4", "yes").ok and g.check("check oganesson mass 2 9 5", "no").ok
    for a in g.weighed:
        for b in g.weighed:
            if a is not b:
                assert not g.check(f"{a['name']} mass", dense_value(b, "mass")).ok, (a["name"], b["name"])
    assert 0 < MASS_TOLERANCE < TOLERANCE


def test_numbers_must_be_written_as_num_writes_them(g):
    for prompt, answer in (("iron group", "0 8"), ("hydrogen neutrons", "0 0"), ("iron protons", "0 2 6"),
                           ("iron protons", "2 6 point 0"), ("iron protons", "2 point 6 e 1"),
                           ("iron melting_k", "1 8 1 1 point"), ("iron melting_k", "0 1 8 1 1 point 2"),
                           ("iron mass", "1 e 9 9 9 9"), ("iron valence", "plus 2 plus 3 plus 2"),
                           ("iron valence", "plus 0 2 plus 3"), ("iron valence", "2 3"),
                           ("count elements period 1", "0 2"), ("iron discovered", "ancient ancient")):
        v = g.check(prompt, answer)
        assert not v.ok and v.expected is not None, (prompt, answer)
    assert g.check("element number 0 2 6", "iron").expected is None
    assert g.check("neutrons mass_number 5 6 protons 0 2 6", "3 0").expected is None
    assert g.check("check iron protons 0 2 6", "no").ok
    assert _signed("plus 2 minus 1") == {2, -1} and _signed("plus 2 plus 2") is None and _signed("") is None


def test_lists_and_counts_follow_the_table(g):
    rows = g.rows
    for (key, value), names in g.groups.items():
        assert key in LIST_KEYS
        expect = [r["name"] for r in rows if key in r and dense_value(r, key) == value]
        assert names == expect  # ordered by atomic number
        assert g.check(f"count elements {key} {value}", num(len(names))).ok
        if len(names) <= MAX_LIST:
            assert g.check(f"elements {key} {value}", " ".join(names)).ok
        else:
            assert g.check(f"elements {key} {value}", " ".join(names)).expected is None
    assert len(g.groups[("period", "6")]) == 32 and len(g.groups[("period", "7")]) == 32
    assert len(g.groups[("block", "d")]) > MAX_LIST  # asked only as a count
    assert g.groups[("category", "lanthanide")][0] == "lanthanum" and g.groups[("category", "lanthanide")][-1] == "lutetium"
    assert g.groups[("group", "lanthanide")][0] == "cerium" and len(g.groups[("group", "lanthanide")]) == 14
    assert g.groups[("group", "3")] == ["scandium", "yttrium", "lanthanum", "actinium"]
    assert sum(len(names) for (key, _), names in g.groups.items() if key == "state") == 97


def test_rules_hold_for_every_element(g):
    for r in g.rows:
        name = r["name"]
        assert g.check(f"{name} electrons", dense_value(r, "protons")).ok
        assert not g.check(f"{name} neutrons", dense_value(r, "mass_number")).ok  # not the mass number
        assert parse_num(dense_value(r, "neutrons")) == r["mass_number"] - r["protons"]
        assert g.check(f"neutrons mass_number {dense_value(r, 'mass_number')} protons {dense_value(r, 'protons')}",
                       dense_value(r, "neutrons")).ok
        assert g.check(f"element number {dense_value(r, 'number')}", name).ok
        assert g.check(f"element symbol {r['symbol']}", name).ok
        if isinstance(r["group"], int):
            assert g.check(f"element group {num(r['group'])} period {num(r['period'])}", name).ok
        for key in FACT_KEYS + RULE_KEYS:
            if key in r:
                assert g.check(f"{name} {key}", dense_value(r, key)).ok, (name, key)
                assert g.check(f"check {name} {key} {dense_value(r, key)}", "yes").ok, (name, key)
    assert set(RECORD_KEYS) | set(RECORD_MORE_KEYS) == set(KEYS)
    assert not (INT_KEYS & FLOAT_KEYS) and set(ORDER_KEYS) <= INT_KEYS | FLOAT_KEYS


def test_wrong_check_values_are_plausible_and_really_wrong(g, lines, many):
    valence_no = 0
    for ln in lines + many:
        w = ln.prompt.split()
        if ln.kind != "yesno" or w[2] in ("heavier_than", "lighter_than"):
            continue
        name, key, value = w[1], w[2], " ".join(w[3:])
        row = g.by_name[name]
        truth = dense_value(row, key)
        if ln.answer == "yes":
            assert value == truth
            continue
        assert not g._agree(key, truth, value)
        if key in INT_KEYS or key in FLOAT_KEYS:
            assert parse_num(value) is not None  # a number, just a wrong one
            assert not value.startswith("0 ") or "point" in value
        if key == "valence":  # never a subset of the real states: it names a state the element lacks
            valence_no += 1
            assert not _signed(value) <= set(row["valence_known"]), ln.text
    assert valence_no > 20
    assert g.check("check iron valence plus 3", "no").ok  # judged as the whole set, but never generated


def test_mass_comparisons_need_two_standard_weights(g, many):
    weighed = {r["name"] for r in g.weighed}
    assert len(weighed) == 84 and "uranium" in weighed and "neptunium" not in weighed
    seen = Counter()
    for ln in many:
        w = ln.prompt.split()
        if w[0] in ("heavier", "lighter"):
            assert {w[1], w[2]} <= weighed, ln.text
            seen["compare"] += 1
        elif w[0] == "check" and w[2] in ("heavier_than", "lighter_than"):
            assert {w[1], w[3]} <= weighed, ln.text
            seen["check"] += 1
        elif w[0] in ("higher", "lower"):
            a, b = g.by_name[w[2]][w[1]], g.by_name[w[3]][w[1]]
            assert abs(a - b) > TOLERANCE * max(abs(a), abs(b)), ln.text  # never a near tie
            assert ln.answer == (w[2] if (a > b) == (w[0] == "higher") else w[3])
            seen[w[1]] += 1
    assert set(seen) == {"compare", "check", *ORDER_KEYS} and min(seen.values()) > 100
    for prompt in ("heavier bohrium hassium", "lighter plutonium americium", "heavier uranium neptunium",
                   "check bohrium heavier_than hassium", "heavier hydrogen oganesson",
                   "higher electronegativity helium iron", "higher density_g_cm3 hassium iron"):
        assert g.owns(prompt) and g.check(prompt, "x") == Verdict(False, None, "no answer in the table"), prompt


def test_many_distinct_lines(g, many):
    facts = {ln.text for ln in many if ln.kind == "fact"}
    assert len(facts) > 2000  # nearly the whole table
    assert "q iron protons. a 2 6." in facts and "q vanadium mass. a 5 0 point 9 4 2." in facts
    assert g.generate(random.Random(3), 0) == []


def test_generate_warns_when_the_topic_runs_out(g):
    with pytest.warns(RuntimeWarning, match="lines asked for exist"):
        got = g.generate(random.Random(5), 200000)
    assert 130000 < len(got) < 200000 and len({ln.text for ln in got}) == len(got)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert len(g.generate(random.Random(5), 100000)) == 100000


def test_sample_lines_verbatim(g, many):
    texts = {ln.text for ln in many}
    for text in ("q element after iron. a cobalt.", "q element before iron. a manganese.",
                 "q elements group 1 8. a helium neon argon krypton xenon radon oganesson.",
                 "q count elements category noble_gas. a 7.", "q count elements category lanthanide. a 1 5.",
                 "q element number 2 6. a iron.", "q element symbol fe. a iron.",
                 "q element group 8 period 4. a iron.", "q iron neutrons. a 3 0.", "q iron electrons. a 2 6.",
                 "q check iron valence plus 2 plus 3. a yes."):
        assert text in texts, text
    for start, end in (("q check ", " a no."), ("q check ", " a yes."), ("q heavier ", "."), ("q lighter ", "."),
                       ("q higher melting_k ", "."), ("q lower density_g_cm3 ", "."),
                       ("q neutrons mass_number ", "."), ("q check iron heavier_than ", ".")):
        assert any(t.startswith(start) and t.endswith(end) for t in texts), (start, end)
