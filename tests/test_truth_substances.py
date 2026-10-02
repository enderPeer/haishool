import json
import random
import re
from decimal import ROUND_HALF_UP, Decimal, localcontext
from fractions import Fraction
from pathlib import Path

import pytest

import haishool.truth.substances as module
from haishool.truth import Gate, Line, Verdict, is_dense, parse_num, split_line
from haishool.truth.formula import dense, parse, parse_dense, same
from haishool.truth.substances import (BASES, BOND_DOUBT, BONDS, CLASSES, CLOSED_PLACES, ELEMENTS, INSIDE, KEYS,
                                       MATERIALS, MAX_LIST, METALLOIDS, MIX_KEYS, MIXTURES, NAME_OF, NONMETALS,
                                       PART_DOUBT, PLACES, PURE, PURE_KEYS, STATE_DOUBT, STATES, SYMBOL_OF, UNITS,
                                       Substances, basis_truth, bond_truth, class_truth, describe, exact, gate,
                                       has_standard_weight, impossible_places, load_rows, load_weights, sig,
                                       state_truth, strict_formula, table_rows)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "truth-v5"

#: what the task names; every one must be a row
REQUIRED = """
nitrogen oxygen argon carbon_dioxide water ozone methane neon helium hydrogen nitrous_oxide water_ice
sodium_chloride magnesium_chloride magnesium_sulfate calcium_carbonate
quartz orthoclase albite anorthite muscovite biotite forsterite fayalite enstatite diopside hornblende calcite
aragonite dolomite gypsum anhydrite halite sylvite hematite magnetite goethite pyrite galena sphalerite
chalcopyrite corundum rutile zircon apatite kaolinite talc graphite diamond gibbsite
bridgmanite periclase iron nickel iron_nickel
atomic_hydrogen carbon_monoxide ammonia methanol formaldehyde hydrogen_cyanide acetylene naphthalene
sulfuric_acid sulfur_dioxide hydrogen_sulfide phosphine carbon_dioxide_ice
glucose fructose galactose sucrose lactose maltose ribose deoxyribose glycogen starch cellulose glycerol
palmitic_acid stearic_acid oleic_acid cholesterol
glycine alanine valine leucine isoleucine proline phenylalanine tryptophan methionine serine threonine cysteine
tyrosine asparagine glutamine aspartic_acid glutamic_acid lysine arginine histidine
adenine guanine cytosine thymine uracil atp adp nadh urea uric_acid creatine lactic_acid pyruvic_acid citric_acid
acetic_acid ethanol caffeine nicotine chlorophyll_a heme_b vitamin_c vitamin_d3 aspirin paracetamol ibuprofen
penicillin_g dopamine adrenaline serotonin testosterone oestradiol cortisol
nitric_acid hydrogen_chloride phosphoric_acid sodium_hydroxide potassium_hydroxide sodium_carbonate
sodium_bicarbonate calcium_oxide calcium_hydroxide alite sodium_hypochlorite hydrogen_peroxide nitrogen_dioxide
sulfur_trioxide cfc_12 polyethylene polypropylene polyvinyl_chloride polystyrene polyethylene_terephthalate ptfe
nylon_6 polydimethylsiloxane tnt nitroglycerin ammonium_nitrate potassium_nitrate ethane propane butane octane
ethylene benzene toluene acetone formic_acid silicon silicon_carbide silica gallium_arsenide lithium_cobalt_oxide
lithium titanium_dioxide zinc_oxide aluminum_oxide copper_sulfate silver_chloride gold silver copper aluminum tin
lead zinc tungsten uranium_dioxide plutonium_dioxide fullerene boron_nitride carbon
air seawater steel stainless_steel bronze brass solder gunpowder granite basalt limestone sandstone wood bone
blood_plasma milk honey petrol natural_gas crude_oil concrete soda_lime_glass sun earth_crust human_body universe
""".split()

NOT_MINE = ["turkey capital", "q spoon color", "spoon color", "carbon protons", "iron protons", "hydrogen number",
            "calc 1 2 plus 7", "solve 3 x plus 4 equals 1 9", "check 1 2 plus 7 equals 2 0", "check turkey capital ankara",
            "gravity seed 7 step 2 0 clumps", "elements group 1 8", "water color", "air formula", "turkey borders greece",
            "balance h 2 plus o 2", "check", "check check water formula h 2 o 1", "water", "substance", "",
            # the prompts of rounds 1-4 that an earlier version of this gate claimed
            "water found_in", "gold found_in", "gold made_of", "diamond made_of", "milk parts", "sun parts",
            "water found_in sink", "check water found_in ocean", "air parts", "water made_of",
            # a question inside a check has no second ``q``; a verb needs a row or a symbol after it
            "check q atoms water 3", "check q water formula h 2 o 1", "atoms is_a", "molar_mass", "atoms part_of molecule",
            "heavier_molecule", "water main_parts", "air class", "air composition"]


def n_tokens(text: str) -> int:
    return len(text.replace(".", " . ").split())


@pytest.fixture(scope="module")
def g() -> Substances:
    return gate()


@pytest.fixture(scope="module")
def lines(g) -> list[Line]:
    return g.generate(random.Random(11), 4000)


def wrong(answer: str, last: bool = False) -> str:
    """The answer with one thing changed: yes/no flipped, one digit raised (the first of its first
    number, or with ``last`` the very last digit of the answer), or its first word replaced."""
    if answer in ("yes", "no"):
        return "no" if answer == "yes" else "yes"
    words = answer.split()
    for i in (reversed(range(len(words))) if last else range(len(words))):
        if words[i].isdigit():
            words[i] = str(int(words[i]) % 9 + 1)
            return " ".join(words)
    return " ".join(["zzz"] + words[1:])


def half_up(x: Fraction, digits: int) -> float:
    """``x`` to ``digits`` significant digits with decimal ties rounded up, by the decimal module
    (an independent path: the module rounds with integer arithmetic on fractions)."""
    with localcontext() as ctx:
        ctx.prec = 60
        d = Decimal(x.numerator) / Decimal(x.denominator)
        return float(d.quantize(Decimal(1).scaleb(d.adjusted() - digits + 1), rounding=ROUND_HALF_UP))


def exact_mass(counts: dict[str, int], weights: dict) -> Fraction:
    return sum(Fraction(str(weights[el])) * n for el, n in counts.items())


# ---- the tables -------------------------------------------------------------------------------------

def test_atomic_weights_file():
    weights = load_weights()
    assert list(weights) == [sym for sym, _ in ELEMENTS] and len(weights) == 118
    for z, (sym, name) in enumerate(ELEMENTS, start=1):
        # a float is a standard atomic weight, an integer a mass number
        assert isinstance(weights[sym], float if has_standard_weight(z) else int), sym
        assert name == name.lower() and name.isalpha()
    assert (weights["H"], weights["C"], weights["O"], weights["Fe"]) == (1.008, 12.011, 15.999, 55.845)
    assert (weights["Tc"], weights["Pm"], weights["Rn"], weights["Pu"]) == (98, 145, 222, 244)
    # the three isotopes the source's own half-lives show to outlive its bracket weight
    assert (weights["Mt"], weights["Rg"], weights["Mc"]) == (278, 282, 290)
    assert weights["U"] == pytest.approx(238.03, abs=0.01) and weights["Th"] == pytest.approx(232.04, abs=0.01)
    # the tokens of the elements table and of rounds 1-4, not the British spelling
    assert NAME_OF["Al"] == "aluminum" and NAME_OF["Cs"] == "cesium" and NAME_OF["S"] == "sulfur"


def test_atomic_weights_match_mendeleev():
    pytest.importorskip("mendeleev")
    from haishool.truth.substances import build_weights
    assert build_weights() == load_weights()


def test_weights_and_names_agree_with_the_elements_table():
    path = DATA / "elements.jsonl"
    if not path.exists():
        pytest.skip("elements table not built")
    elements = {r["symbol"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())}
    weights = load_weights()
    for sym, name in ELEMENTS:
        row = elements[sym.lower()]
        assert row["name"] == name, sym
        # a float there is the weight to 5 significant digits, an integer the mass number
        assert row["mass"] == (weights[sym] if isinstance(weights[sym], int) else sig(weights[sym], 5)), sym


def test_file_is_the_table():
    rows = load_rows()
    assert rows == table_rows(load_weights()), "run: python -m haishool.truth.substances build"
    assert [r["name"] for r in rows] == [r[0] for r in PURE] + [r[0] for r in MIXTURES]
    for line in (DATA / "substances.jsonl").read_text(encoding="utf-8").splitlines():
        json.loads(line)


def test_table_size_and_coverage(g):
    assert 400 <= len(g.pure) <= 460
    assert 35 <= len(g.mixtures) <= 60
    assert not [n for n in REQUIRED if n not in g.rows]
    names = list(g.rows)
    assert len(names) == len(set(names))
    for n in names:
        assert re.fullmatch(r"[a-z][a-z0-9_]{2,}", n) and is_dense(n + "."), n
        assert parse_dense(n) is None, "a name must not read as a dense formula"
    amino = [r["name"] for r in g.pure if r["class"] == "amino_acid"]
    assert len(amino) == 20
    assert sorted(r["name"] for r in g.pure if r["class"] == "nucleobase") == ["adenine", "cytosine", "guanine", "thymine", "uracil"]
    assert "insulin" not in g.rows  # no single formula
    assert not [n for n in names if "aluminium" in n or "caesium" in n]


def test_vocabulary(g):
    for r in g.pure:
        assert r["class"] in CLASSES and r["bonds"] in BONDS and r["state"] in STATES, r["name"]
        assert r["occurs_in"] and set(r["occurs_in"]) <= set(PLACES) and len(set(r["occurs_in"])) == len(r["occurs_in"])
        assert r["unit"] == (r["name"] in UNITS)
        assert (r["class"] == "polymer_unit") == r["unit"]
        if r["class"] == "ice":
            assert r["state"] == "solid"
        if r["class"] == "metal":
            assert r["bonds"] == "metallic" and r["elements"] == 1
            assert not set(parse(r["formula"])) & (NONMETALS | METALLOIDS), r["name"]
        if r["bonds"] == "none":
            assert r["elements"] == 1 and r["atoms"] == 1 and r["state"] == "gas", r["name"]
        if r["elements"] == 1:  # an element carries one of the three labels an element can have
            assert r["class"] in ("element", "metal", "semiconductor"), r["name"]
    assert "gas" not in CLASSES and "molecular" not in BONDS  # a state is no class; ``none`` for free atoms
    assert g.rows["antimony"]["class"] == "element"  # a metalloid, as the elements table says
    assert {r["class"] for r in g.pure} == set(CLASSES), "every class is used"
    assert {p for r in g.rows.values() for p in r["occurs_in"]} == set(PLACES), "every place is used"
    for r in g.mixtures:
        assert r["basis"] in BASES and r["state"] in STATES and set(r["occurs_in"]) <= set(PLACES)
        assert 90 <= sum(r["main_parts"].values()) <= 101, r["name"]
        assert all(v > 0 for v in r["main_parts"].values()) and len(r["main_parts"]) >= 2
        assert list(r["main_parts"].values()) == sorted(r["main_parts"].values(), reverse=True), r["name"]
        assert r["notes"], "every mixture says how exact its shares are"
        for comp in r["main_parts"]:
            if r["basis"] == "element_mass":
                assert comp in SYMBOL_OF, (r["name"], comp)
            else:
                assert comp in g.rows or comp in MATERIALS, (r["name"], comp)
    assert not set(MATERIALS) & set(g.rows)
    assert {m for m, classes in MATERIALS.items() for c in classes if c not in CLASSES} == set()
    assert set(STATE_DOUBT) | set(BOND_DOUBT) | set(PART_DOUBT) <= set(g.rows)
    by_element = {r["name"] for r in g.mixtures if r["basis"] == "element_mass"}
    assert {"earth_crust", "earth_core", "sun", "universe", "human_body", "coal", "crude_oil", "steel", "cast_iron"} <= by_element


def test_places_are_completed_by_rule(g):
    for r in g.rows.values():
        for inner, outer in INSIDE.items():
            if inner in r["occurs_in"]:
                assert outer in r["occurs_in"], (r["name"], inner, outer)
    for m in g.mixtures:
        if m["basis"] == "element_mass":
            continue
        for comp in m["main_parts"]:
            row = g.rows.get(comp)
            if row is not None and not row["mixture"]:  # what a mixture holds occurs where the mixture occurs
                assert set(m["occurs_in"]) <= set(row["occurs_in"]), (m["name"], comp)
    assert "mars" in g.rows["oxygen"]["occurs_in"] and "bones" in g.rows["water"]["occurs_in"]
    assert "crust" in g.rows["methane"]["occurs_in"] and "animals" in g.rows["glucose"]["occurs_in"]
    # molecular hydrogen is not in the sun (atomic_hydrogen is), bromine and iodine are not in the sea as elements
    assert not {"sun", "stars"} & set(g.rows["hydrogen"]["occurs_in"])
    assert "ocean" not in g.rows["bromine"]["occurs_in"] + g.rows["iodine"]["occurs_in"]
    assert g.rows["sun"]["occurs_in"] == ["stars"]  # not ``sun occurs_in sun``


def test_computed_fields_follow_from_the_formula(g):
    weights = load_weights()
    for r in g.pure:
        counts = parse(r["formula"])
        assert all(n >= 1 for n in counts.values())
        mass = exact_mass(counts, weights)
        assert r["molar_mass"] == half_up(mass, 5), r["name"]
        assert r["atoms"] == sum(counts.values()) == sum(r["composition"].values())
        assert r["elements"] == len(counts) == len(r["composition"]) == len(r["mass_percent"])
        assert list(r["composition"]) == [NAME_OF[el] for el in counts]
        for el, n in counts.items():
            assert r["mass_percent"][NAME_OF[el]] == half_up(100 * Fraction(str(weights[el])) * n / mass, 4), r["name"]
        assert sum(r["mass_percent"].values()) == pytest.approx(100, abs=0.05), r["name"]
        assert r["dense"] == dense(r["formula"]) and same(parse_dense(r["dense"]), counts)
        fresh = describe(r["formula"], weights)
        assert all(r[k] == fresh[k] for k in fresh)


def test_ties_round_up_and_do_not_depend_on_the_order(g):
    # exact decimal ties at 5 significant digits: a float sum lands on either side of them
    for name, value in [("thymine", 126.12), ("creatine", 131.14), ("petn", 316.14), ("paracetamol", 151.17),
                        ("ibuprofen", 206.29), ("heptane", 100.21), ("pyrite", 119.97), ("zircon", 183.31),
                        ("nitroglycerin", 227.09)]:
        mass = exact_mass(parse(g.rows[name]["formula"]), g.weights)
        assert mass * 1000 % 10 == 5, name  # the sixth digit is a 5 and nothing follows
        assert g.rows[name]["molar_mass"] == value, name
    assert g.check("molar_mass v 1", "5 0 point 9 4 2").ok and g.check("molar_mass yb 1", "1 7 3 point 0 5").ok
    assert g.check("molar_mass c 5 h 6 n 2 o 2", "1 2 6 point 1 2").ok
    assert g.check("molar_mass c 9 h 2", "1 1 0 point 1 2").ok and g.check("molar_mass fe 1 1", "6 1 4 point 3").ok
    a = g.check("molar_mass c 1 h 1 3 cl 4", "x").expected
    assert a == g.check("molar_mass cl 4 h 1 3 c 1", "x").expected == g.check("molar_mass h 1 3 cl 4 c 1", "x").expected
    assert sig(126.115, 5) == 126.12 and sig(50.9415, 5) == 50.942 and sig(173.045, 5) == 173.05
    assert sig(Fraction(126115, 1000), 5) == 126.12 and sig(0.0, 3) == 0.0 and sig(99.996, 4) == 100.0
    assert sig(18.01528, 5) == 18.015 and sig(1000, 3) == 1000.0 and sig(0.001, 2) == 0.001
    rng = random.Random(4)
    for _ in range(300):  # generated molar masses are the half-up rounding of the exact sum
        f = g.random_formula(rng)
        expected = g.check(f"molar_mass {dense(f)}", "x").expected
        assert parse_num(expected) == pytest.approx(half_up(exact_mass(parse(f), g.weights), 5), rel=0, abs=0), f


#: the usual oxidation states, for the charge balance of the minerals, salts and ceramics
OXIDATION = {"H": [1], "Li": [1], "Na": [1], "K": [1], "Ag": [1], "Be": [2], "Mg": [2], "Ca": [2], "Sr": [2], "Ba": [2],
             "Zn": [2], "Cd": [2], "Ni": [2], "Hg": [2], "B": [3], "Al": [3], "Ga": [3], "In": [3], "Cr": [3], "Sb": [3],
             "Si": [4], "Zr": [4], "U": [4], "Pu": [4], "Mo": [4], "Ti": [4, 3], "O": [-2], "F": [-1], "Br": [-1],
             "I": [-1], "Te": [-2], "Cl": [-1, 1, 5, 7], "C": [4, -4], "N": [-3, 3, 5], "P": [5, -3], "S": [-2, -1, 4, 6],
             "Fe": [2, 3], "Mn": [2, 4], "Cu": [1, 2], "Pb": [2, 4], "Sn": [4, 2], "Co": [3, 2], "W": [6, 4],
             "As": [3, -3, -1, 2]}
#: not ionic in this simple sense: a lower oxide, an interstitial carbide, the azide, acetylide, peroxide and
#: superoxide ions, an organic anion
NO_BALANCE = {"silicon_monoxide", "cementite", "lead_azide", "sodium_azide", "calcium_carbide", "sodium_peroxide",
              "potassium_superoxide", "monosodium_glutamate"}


def test_ionic_and_network_compounds_are_charge_balanced(g):
    checked = 0
    for r in g.pure:
        if r["bonds"] not in ("ionic", "network") or r["elements"] == 1 or r["name"] in NO_BALANCE:
            continue
        sums = {0}
        for el, n in parse(r["formula"]).items():
            for _ in range(n):  # each atom of an element may take any of its usual states (magnetite)
                sums = {total + state for total in sums for state in OXIDATION[el]}
        assert 0 in sums, (r["name"], r["formula"])
        checked += 1
    assert checked > 160


def test_organic_molecules_have_a_whole_degree_of_unsaturation(g):
    organic = set("C H N O S P F Cl Br I Si".split())
    checked = 0
    for r in g.pure:
        c = parse(r["formula"])
        if r["bonds"] != "covalent" or "C" not in c or r["elements"] == 1 or not set(c) <= organic:
            continue
        halogens = sum(c.get(x, 0) for x in ("F", "Cl", "Br", "I"))
        twice = 2 * c["C"] + 2 * c.get("Si", 0) + 2 + c.get("N", 0) + c.get("P", 0) - c.get("H", 0) - halogens
        assert twice >= 0 and twice % 2 == 0, (r["name"], r["formula"])  # a miscounted hydrogen shows here
        checked += 1
    assert checked > 170


def test_known_values(g):
    water = g.rows["water"]
    assert (water["molar_mass"], water["atoms"], water["elements"]) == (18.015, 3, 2)
    assert water["mass_percent"] == {"hydrogen": 11.19, "oxygen": 88.81}
    assert g.rows["glucose"]["molar_mass"] == 180.16 and g.rows["glucose"]["atoms"] == 24
    assert g.rows["sodium_chloride"]["molar_mass"] == 58.44
    assert g.rows["carbon_dioxide"]["molar_mass"] == 44.009
    assert g.rows["muscovite"]["composition"] == {"potassium": 1, "aluminum": 3, "silicon": 3, "oxygen": 12, "hydrogen": 2}
    assert g.rows["gypsum"]["composition"] == {"calcium": 1, "sulfur": 1, "oxygen": 6, "hydrogen": 4}
    assert g.rows["plutonium_dioxide"]["molar_mass"] == 276.0  # 244 + 2 * 15.999, plutonium by mass number
    assert g.rows["air"]["main_parts"] == {"nitrogen": 78.1, "oxygen": 20.9, "argon": 0.93, "carbon_dioxide": 0.04}
    assert g.rows["air"]["basis"] == "volume" and g.rows["bronze"]["basis"] == "mass"
    assert g.rows["steel"]["basis"] == "element_mass" and g.rows["steel"]["main_parts"]["carbon"] == 0.2  # mild steel
    concrete = g.rows["concrete"]["main_parts"]  # by mass: about half as much water as cement
    assert concrete == {"gravel": 47, "sand": 30, "portland_cement": 15, "water": 8}
    assert 0.4 <= concrete["water"] / concrete["portland_cement"] <= 0.6
    assert g.rows["iron_nickel"]["main_parts"]["nickel"] == 6  # kamacite holds 5 to 7.5 percent
    # molar masses known from the literature, to the rounding of the weights
    for name, mass in [("caffeine", 194.19), ("sucrose", 342.3), ("cholesterol", 386.66), ("atp", 507.18),
                       ("nad", 663.43), ("fad", 785.55), ("acetyl_coa", 809.57), ("gtp", 523.18),
                       ("progesterone", 314.46), ("noradrenaline", 169.18), ("thymidine", 242.23),
                       ("phosphorus_pentoxide", 283.89), ("iron_oxide", 159.69), ("hydrazine", 32.045)]:
        assert g.rows[name]["molar_mass"] == pytest.approx(mass, rel=1e-4), name


def test_states_agree_with_the_elements_table(g):
    path = DATA / "elements.jsonl"
    if not path.exists():
        pytest.skip("elements table not built")
    elements = {r["name"]: r for r in map(json.loads, path.read_text(encoding="utf-8").splitlines())}
    shared = [n for n in g.rows if n in elements and "state" in elements[n]]
    assert len(shared) > 30 and "aluminum" in shared and "carbon" in shared
    for n in shared:  # both gates answer ``q <name> state``; they must say the same
        assert g.rows[n]["state"] == elements[n]["state"], n
    assert elements["antimony"]["metal"] == "metalloid" and g.rows["antimony"]["class"] != "metal"


def test_species_of_the_reactions_table_have_the_same_formula_here(g):
    path = DATA / "reactions.jsonl"
    if not path.exists():
        pytest.skip("reactions table not built")
    species: dict[str, str] = {}
    for r in map(json.loads, path.read_text(encoding="utf-8").splitlines()):
        if r.get("nuclear"):
            continue
        for (_, formula), name in zip(r["reactants"] + r["products"], r["reactant_names"] + r["product_names"]):
            species[name] = formula
    shared = [n for n in species if n in g.rows and not g.rows[n]["mixture"]]
    assert len(shared) > 100 or len(shared) > 0.9 * len(species)
    for n in shared:
        assert same(parse(species[n]), parse(g.rows[n]["formula"])), n
    assert {"carbon", "aluminum", "iron_oxide", "aluminum_oxide", "barium_sulfate"} <= set(g.rows)


# ---- no prompt of rounds 1-4 --------------------------------------------------------------------------

def test_keys_are_no_relation_and_no_attribute_of_rounds_1_to_4():
    from haishool.relations import INVERSE
    from haishool.schema import ALL_KEYS
    taken = set(INVERSE) | set(INVERSE.values()) | set(ALL_KEYS)
    assert not taken & (set(PURE_KEYS) | set(MIX_KEYS) | {"part"})
    assert {"found_in", "made_of", "parts"} <= taken and not {"found_in", "made_of", "parts"} & set(KEYS)


def test_no_prompt_of_rounds_1_to_4_is_owned(g):
    from haishool.student import load_records
    prompts = []
    hops = ROOT / "data" / "hops-v4b" / "hops-train-r4.txt"
    if hops.exists():
        prompts += [pair[0] for pair in map(split_line, hops.read_text(encoding="utf-8").splitlines()) if pair]
    for name in ("records-r1.jsonl", "records-r2.jsonl", "records-r3-all.jsonl"):
        path = ROOT / "data" / name
        if path.exists():
            prompts += [split_line(q)[0] for rec in load_records(path) for q in rec.queries()]
    if not prompts:
        pytest.skip("no data of rounds 1-4")
    assert len(prompts) > 1000
    assert not [p for p in prompts if g.owns(p)][:5]


def test_lines_fall_into_round_5_buckets(g, lines):
    from haishool.student import CLASSIC_KINDS, extra_kind
    for ln in lines + g.facts():
        kind = extra_kind(ln.text)
        assert kind not in CLASSIC_KINDS, ln.text  # else the line is never held out and counts as round 4
        assert kind == ("fact" if len(ln.prompt.split()) == 2 or ln.prompt.split()[0] in g.rows else ln.prompt.split()[0])


# ---- records ----------------------------------------------------------------------------------------

def test_records_are_dense_and_short(g):
    recs = g.records()
    assert all(r.kind == "record" and r.topic == "substances" and not r.answer for r in recs)
    for r in recs:
        assert is_dense(r.text), r.text
        assert n_tokens(r.text) <= 80, r.text
    heads = [r.text.split(".")[0] for r in recs]
    assert len(heads) == len(set(heads))
    assert {h.split()[0] for h in heads} == set(g.rows)
    assert {h.split()[1] for h in heads} == {"substance", "substance_more", "mixture"}
    assert len(recs) == len(g.rows) + sum(h.endswith("_more") for h in heads)


def test_record_examples(g):
    text = {r.text.split(".")[0]: r.text for r in g.records()}
    assert text["water substance"] == (
        "water substance. formula h 2 o 1. composition hydrogen 2 oxygen 1. atoms 3. elements 2. "
        "molar_mass 1 8 point 0 1 5. mass_percent hydrogen 1 1 point 1 9 oxygen 8 8 point 8 1. class oxide. "
        "bonds covalent. state liquid. occurs_in ocean rivers atmosphere comets mars cells blood plants animals food "
        "bones industry buildings.")
    assert text["air mixture"] == ("air mixture. main_parts nitrogen 7 8 point 1 oxygen 2 0 point 9 argon 0 point 9 3 "
                                   "carbon_dioxide 0 point 0 4. basis volume. state gas. occurs_in atmosphere.")
    assert text["earth_crust mixture"].endswith("basis element_mass. state solid. occurs_in crust.")
    assert " unit yes. " in text["cellulose substance"] and " unit " not in text["glucose substance"]


def test_long_rows_split_at_a_field(g):
    split = [r for r in g.rows.values() if len(g.record_lines(r)) > 1]
    assert split and "vitamin_b12" in {r["name"] for r in split}
    for row in split:
        first, more = [ln.text for ln in g.record_lines(row)]
        assert first.startswith(f"{row['name']} substance. formula ") and more.startswith(f"{row['name']} substance_more. ")
        fields = first.split(". ")[1:] + more.split(". ")[1:]
        keys = [f.split()[0] for f in fields]
        assert keys == [k for k in ("formula", "unit", "composition", "atoms", "elements", "molar_mass", "mass_percent",
                                    "class", "bonds", "state", "occurs_in") if k != "unit" or row["unit"]]


def test_every_record_field_is_a_fact_the_gate_accepts(g):
    for r in g.records():
        name, head = r.text.split(".")[0].split()
        assert head in KEYS or head == "mixture"
        for field in r.text.rstrip(".").split(". ")[1:]:
            key, _, value = field.partition(" ")
            assert key in KEYS
            verdict = g.check(f"{name} {key}", value)
            assert verdict.ok and verdict.expected == value, (r.text, key)


# ---- questions --------------------------------------------------------------------------------------

def test_gate_protocol(g):
    assert isinstance(g, Gate) and g.topic == "substances" and g.KEYS is KEYS
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in KEYS.items())
    # every key of a row and every first word of a question is documented
    assert set(PURE_KEYS) | set(MIX_KEYS) | {"part", "substance", "substances"} | set(module.CALC_HEADS) <= set(KEYS)


def test_generate_is_deterministic(g):
    a = [ln.text for ln in g.generate(random.Random(5), 800)]
    assert a == [ln.text for ln in g.generate(random.Random(5), 800)]
    assert a == [ln.text for ln in gate().generate(random.Random(5), 800)]
    assert a[:200] != [ln.text for ln in g.generate(random.Random(6), 200)]
    assert [r.text for r in g.records()] == [r.text for r in gate().records()]
    assert [f.text for f in g.facts()] == [f.text for f in gate().facts()]


def test_generated_lines_are_dense_short_and_distinct(g, lines):
    assert len(lines) == 4000 == len({ln.text for ln in lines})
    for ln in lines + g.facts():
        assert is_dense(ln.text), ln.text
        assert n_tokens(ln.text) <= 80, ln.text
        assert "." not in ln.prompt and "." not in ln.answer and ln.answer
        assert ln.topic == "substances" and ln.kind in ("fact", "calc", "yesno")
        assert (ln.kind == "yesno") == ln.prompt.startswith("check ")
        if ln.answer in ("yes", "no"):  # a check, or the two yes/no keys of the table
            assert ln.kind == "yesno" or ln.prompt.split()[-1] in ("mixture", "unit"), ln.text
        else:
            assert ln.kind != "yesno", ln.text


def test_no_prompt_has_two_answers(g, lines):
    answers: dict[str, str] = {}
    for ln in lines + g.facts() + g.generate(random.Random(12), 4000):
        assert answers.setdefault(ln.prompt, ln.answer) == ln.answer, ln.prompt


def test_mix_of_kinds(g):
    some = g.generate(random.Random(2), 2000)
    share = {k: sum(ln.kind == k for ln in some) / len(some) for k in ("fact", "calc", "yesno")}
    assert 0.38 <= share["fact"] <= 0.52 and 0.28 <= share["calc"] <= 0.42 and 0.14 <= share["yesno"] <= 0.26
    heads = {ln.prompt.split()[0] for ln in some}
    assert {"substance", "substances", "molar_mass", "atoms", "mass_percent", "element_count", "heavier_molecule",
            "check"} <= heads
    yesno = [ln.answer for ln in some if ln.kind == "yesno"]
    assert 0.35 <= yesno.count("yes") / len(yesno) <= 0.65
    zeros = [ln for ln in some if ln.kind == "calc" and ln.answer == "0"]
    assert zeros and all(ln.prompt.split()[0] in ("element_count", "mass_percent") for ln in zeros)


def test_twenty_thousand_distinct_lines(g):
    a = g.generate(random.Random(1), 20000)
    assert len({ln.text for ln in a}) == 20000
    b = g.generate(random.Random(2), 20000)
    assert len({ln.text for ln in a} | {ln.text for ln in b}) > 30000


def test_generate_says_when_it_runs_out(g):
    short = gate()
    short._makers = [(1, short._list)]  # a finite kind alone: 17 or so lists
    with pytest.warns(UserWarning, match="distinct lines"):
        made = short.generate(random.Random(1), 500)
    assert len(made) == len(short.lists) < 500


def test_gate_agrees_with_its_own_answers(g, lines):
    for ln in lines + g.facts():
        assert g.owns(ln.prompt), ln.text
        verdict = g.check(ln.prompt, ln.answer)
        assert verdict.ok and verdict.expected == ln.answer, (ln.text, verdict)


def test_gate_rejects_wrong_answers(g, lines):
    for ln in lines + g.facts():
        for last in (False, True):  # the leading digit, and the very last one: no tolerance hides it
            changed = wrong(ln.answer, last)
            verdict = g.check(ln.prompt, changed)
            assert verdict == Verdict(False, ln.answer, verdict.reason) and verdict.reason, (ln.text, changed)
    # a missing or an extra member of a list is wrong too
    assert not g.check("substances occurs_in dna", "adenine cytosine guanine thymine").ok
    assert not g.check("substances class ice", "ammonia_ice carbon_dioxide_ice methane_ice water_ice water").ok
    assert not g.check("water occurs_in", "ocean rivers").ok
    assert not g.check("water composition", "hydrogen 2").ok
    assert not g.check("water formula", "").ok and not g.check("water molar_mass", "heavy").ok


def test_owns(g, lines):
    for ln in lines:
        assert g.owns(ln.prompt) and g.owns("q " + ln.prompt)
    for prompt in NOT_MINE:
        assert not g.owns(prompt), prompt
        assert g.check(prompt, "yes") == Verdict(False, None, "not my question")
    assert g.owns("water formula") and g.owns("air main_parts") and g.owns("check gold class metal")
    assert g.owns("iron state") and not g.owns("iron melting_k")


def test_examples_from_the_brief(g):
    for prompt, answer in [
        ("water formula", "h 2 o 1"), ("water molar_mass", "1 8 point 0 1 5"), ("water class", "oxide"),
        ("water bonds", "covalent"), ("water state", "liquid"), ("water composition", "hydrogen 2 oxygen 1"),
        ("substance formula h 2 o 1", "water"), ("molar_mass c 6 h 1 2 o 6", "1 8 0 point 1 6"),
        ("atoms c 6 h 1 2 o 6", "2 4"), ("mass_percent water oxygen", "8 8 point 8 1"),
        ("element_count water hydrogen", "2"), ("heavier_molecule water methane", "water"),
        ("substances occurs_in dna", "adenine cytosine deoxyribose guanine thymidine thymine"),
        ("check water formula h 2 o 2", "no"),
        ("check gold class metal", "yes"), ("steel mixture", "yes"), ("gold mixture", "no"),
        ("substances class element state liquid", "bromine mercury"),
        ("substance formula c 6 h 1 2 o 6", "fructose galactose glucose"),
        ("check water occurs_in ocean", "yes"), ("check glucose occurs_in sun", "no"),
        ("check steel main_parts nickel", "no"), ("check stainless_steel main_parts nickel", "yes"),
        ("check element_count water hydrogen 2", "yes"), ("check element_count h 2 o 1 hydrogen 3", "no"),
        ("check mass_percent water oxygen 8 8 point 8 1", "yes"), ("check heavier_molecule methane water", "no"),
        ("cellulose unit", "yes"), ("air basis", "volume"), ("air part oxygen", "2 0 point 9"),
        ("earth_crust basis", "element_mass"), ("argon bonds", "none"), ("oxygen class", "element"),
        ("mass_percent corundum aluminum", "5 2 point 9 3"), ("sun occurs_in", "stars"),
    ]:
        assert g.check(prompt, answer) == Verdict(True, answer, ""), prompt


def test_answers_compare_by_value_not_by_spelling(g):
    assert g.check("water formula", "o 1 h 2").ok and g.check("water formula", "h 2 o").ok
    assert g.check("water occurs_in", " ".join(reversed(g.rows["water"]["occurs_in"]))).ok
    assert g.check("water composition", "oxygen 1 hydrogen 2").ok
    assert g.check("substances class ice", "water_ice methane_ice carbon_dioxide_ice ammonia_ice").ok
    assert g.check("molar_mass h 2 o", "1 8 point 0 1 5").ok and g.check("molar_mass water", "1 8 point 0 1 5").ok
    assert g.check("mass_percent h 2 o 1 oxygen", "8 8 point 8 1").ok
    assert g.check("element_count water carbon", "0").ok and g.check("mass_percent water carbon", "0").ok
    assert g.check("bronze main_parts", "tin 1 2 copper 8 8").ok
    assert g.check("sodium_chloride molar_mass", "5 8 point 4 4 0").ok  # a trailing zero changes no value


def test_numbers_are_exact(g):
    # the values are computed, not measured: every digit counts
    assert not g.check("water molar_mass", "1 8 point 0 2").ok and not g.check("water molar_mass", "1 8 point 0 1 6").ok
    assert not g.check("water molar_mass", "1 8 point 1").ok and not g.check("water molar_mass", "1 8").ok
    assert not g.check("water molar_mass", "1 8 point 0 9 9").ok and not g.check("water molar_mass", "1 8 0 point 1 5").ok
    assert not g.check("glucose molar_mass", "1 8 0 point 9").ok and not g.check("vitamin_b12 molar_mass", "1 3 6 0").ok
    assert not g.check("mass_percent water oxygen", "8 8 point 8").ok and not g.check("mass_percent water oxygen", "8 9").ok
    assert not g.check("air part oxygen", "2 1").ok
    assert g.check("check water molar_mass 1 8 point 0 9 9", "no").ok and g.check("check water molar_mass 1 8 point 0 1 5", "yes").ok
    assert g.check("check mass_percent water oxygen 8 8 point 8 2", "no").ok
    big = g.rows["vitamin_b12"]
    assert big["atoms"] == 181 and g.check("vitamin_b12 atoms", "1 8 1").ok and not g.check("vitamin_b12 atoms", "1 8 2").ok
    assert not g.check("atoms c 6 0", "6 1").ok and g.check("atoms c 6 0", "6 0").ok
    assert parse_num(g.check("molar_mass c 6 0", "0").expected) == pytest.approx(720.66)
    assert exact("1 8 point 0 1 5") == Fraction(18015, 1000) and exact("minus 4") == -4
    assert exact("6 point 6 7 4 e minus 1 1") == Fraction(6674, 10 ** 14)
    for bad in ("", "1 8 point", "point 5", "1 point 2 point 3", "heavy", "1 e 9 9 9 9 9", "minus"):
        assert exact(bad) is None, bad


def test_sloppy_answers_are_wrong(g):
    assert not g.check("water occurs_in", "ocean " + " ".join(g.rows["water"]["occurs_in"])).ok  # a repeated word
    assert not g.check("water formula", "h 2 o 1 o 0").ok and not g.check("water formula", "h 1 h 1 o 1").ok
    assert not g.check("water formula", "h 2 o 1 c 0").ok and not g.check("water formula", "h 0 2 o 1").ok
    assert not g.check("water atoms", "3 point 0").ok and not g.check("water atoms", "0 3").ok
    assert not g.check("water composition", "hydrogen 2 oxygen 1 oxygen 1").ok
    assert not g.check("water composition", "hydrogen 2 point 0 oxygen 1").ok
    assert not g.check("water mass_percent", "hydrogen 1 1 point 1 9 oxygen 8 8 point 8 1 oxygen 8 8 point 8 1").ok
    assert strict_formula("h 2 o 1") == {"H": 2, "O": 1} == strict_formula("h 2 o")
    for bad in ("h 2 o 0", "h 2 h 1", "h 2 o 1 2 3 4 5 6 7", "water", "h 0 2", "", "2 h", "h 12"):
        assert strict_formula(bad) is None, bad


def test_check_never_raises(g):
    for prompt in ["check q atoms water 3", "mass_percent h 0 hydrogen", "check mass_percent h 0 hydrogen 5",
                   "molar_mass h 0", "atoms h 0", "molar_mass c " + "9 " * 400, "atoms c " + "9 " * 30,
                   "check water", "check water formula", "check air part", "check air part oxygen",
                   "check air main_parts", "check heavier_molecule water", "check substances class ice water_ice",
                   "substances class", "substances class ice class ice", "heavier_molecule water", "water part oxygen",
                   "check water unit", "check air mixture", "mass_percent water oxygen oxygen", "element_count oxygen"]:
        for answer in ("yes", "0", "", "zzz"):
            assert isinstance(g.check(prompt, answer), Verdict), prompt
    words = ["check", "water", "air", "steel", "gold", "formula", "class", "bonds", "state", "occurs_in", "main_parts",
             "part", "basis", "mixture", "unit", "composition", "atoms", "elements", "molar_mass", "mass_percent",
             "element_count", "heavier_molecule", "substance", "substances", "h", "o", "c", "1", "2", "0", "point",
             "minus", "e", "oxygen", "hydrogen", "ocean", "sun", "metal", "yes", "no", "q", "zzz"]
    rng = random.Random(3)
    for _ in range(20000):
        prompt = " ".join(rng.choice(words) for _ in range(rng.randint(1, 7)))
        answer = " ".join(rng.choice(words) for _ in range(rng.randint(0, 4)))
        verdict = g.check(prompt, answer)
        assert g.owns(prompt) or verdict == Verdict(False, None, "not my question"), prompt
        if verdict.expected is not None:  # what the gate would write back is a training line
            assert is_dense(f"q {prompt}. a {verdict.expected}."), prompt


# ---- what a yes/no line may say ---------------------------------------------------------------------

def test_a_class_beside_the_label_is_never_a_no(g):
    yes = [("gold", "element"), ("oxygen", "element"), ("mercury", "element"), ("silicon", "element"),
           ("iron", "element"), ("helium", "element"), ("quartz", "oxide"), ("hematite", "oxide"),
           ("corundum", "oxide"), ("water_ice", "oxide"), ("glucose", "organic"), ("glycine", "organic"),
           ("benzene", "organic"), ("caffeine", "organic"), ("ethanol", "organic"), ("cortisol", "organic"),
           ("heptane", "organic"), ("lysine", "organic"), ("polystyrene", "organic"), ("polyethylene", "hydrocarbon"),
           ("gold", "inorganic"), ("water", "inorganic"), ("gold", "metal"), ("antimony", "element")]
    open_ = [("halite", "salt"), ("calcite", "salt"), ("diamond", "mineral"), ("graphite", "mineral"),
             ("aspirin", "acid"), ("vitamin_c", "acid"), ("silicon_carbide", "semiconductor"), ("cholesterol", "alcohol"),
             ("antimony", "metal"), ("silicon", "metal"), ("carbon_dioxide", "organic"), ("carbon_dioxide", "inorganic"),
             ("nitrous_oxide", "drug"), ("gaba", "amino_acid"), ("glycogen", "sugar"), ("hydrogen_cyanide", "inorganic"),
             ("aluminum_hydroxide", "acid"), ("ammonium_nitrate", "explosive"), ("spinel", "oxide"),
             ("methane", "gas"), ("water", "zzz")]
    no = [("water", "element"), ("gold", "salt"), ("glucose", "salt"), ("glucose", "amino_acid"), ("quartz", "organic"),
          ("oxygen", "metal"), ("tungsten_carbide", "metal"), ("ethanol", "oxide"), ("sodium_chloride", "hydrocarbon"),
          ("ethylene", "polymer_unit"), ("water", "ice"), ("caffeine", "nucleobase"), ("ammonia", "acid"),
          ("lactic_acid", "base"), ("argon", "explosive"), ("gold", "ceramic"), ("water", "semiconductor"),
          ("methane", "inorganic"), ("oxygen", "mineral"), ("tungsten", "drug")]
    for name, cls in yes:
        assert class_truth(g.rows[name], cls) is True, (name, cls)
        assert g.check(f"check {name} class {cls}", "yes").ok
    for name, cls in open_:
        assert class_truth(g.rows[name], cls) is None, (name, cls)
        assert g.check(f"check {name} class {cls}", "no") == Verdict(False, None, "cannot judge that claim")
    for name, cls in no:
        assert class_truth(g.rows[name], cls) is False, (name, cls)
        assert g.check(f"check {name} class {cls}", "no").ok
    for r in g.pure:  # the rules themselves, once more from the formula
        els = set(parse(r["formula"]))
        assert class_truth(r, r["class"]) is True
        assert class_truth(r, "element") is (len(els) == 1)
        if len(els) == 2 and "O" in els:
            assert class_truth(r, "oxide") is True, r["name"]
        if {"C", "H"} <= els:
            assert class_truth(r, "organic") is not False and class_truth(r, "inorganic") is not True, r["name"]
        if "medicine" in r["occurs_in"]:
            assert class_truth(r, "drug") is not False, r["name"]
        if r["bonds"] == "ionic":
            assert class_truth(r, "salt") is not False, r["name"]
        if r["name"].endswith("_acid"):
            assert class_truth(r, "acid") is not False, r["name"]
        if r["state"] == "solid" and r["bonds"] in ("ionic", "network"):
            assert class_truth(r, "mineral") is not False and class_truth(r, "ceramic") is not False, r["name"]


def test_bonds_state_and_basis_claims(g):
    for name in ("diamond", "quartz", "silicon", "graphite", "forsterite"):
        assert bond_truth(g.rows[name], "covalent") is True and g.check(f"check {name} bonds covalent", "yes").ok
    assert bond_truth(g.rows["water"], "network") is False and bond_truth(g.rows["water"], "none") is False
    assert bond_truth(g.rows["argon"], "none") is True and bond_truth(g.rows["argon"], "covalent") is False
    assert bond_truth(g.rows["sodium_chloride"], "covalent") is False and bond_truth(g.rows["calcium_oxide"], "network") is False
    assert bond_truth(g.rows["quartz"], "ionic") is False and bond_truth(g.rows["gold"], "ionic") is False
    for name, bond in [("calcite", "covalent"), ("pyrite", "network"), ("pyrite", "covalent"), ("forsterite", "ionic"),
                       ("belite", "ionic"), ("cementite", "metallic"), ("antimony", "network"), ("glycine", "ionic"),
                       ("aluminum_chloride", "covalent"), ("polyethylene", "network"), ("sodium_peroxide", "covalent"),
                       ("water", "molecular")]:
        assert bond_truth(g.rows[name], bond) is None, (name, bond)
    for r in g.pure:
        assert bond_truth(r, r["bonds"]) is True
        if r["bonds"] == "network":  # every silicate is a network, by the documented rule
            assert bond_truth(r, "covalent") is True
        els = set(parse(r["formula"]))
        if {"Si", "O"} <= els and "C" not in els and r["elements"] >= 3:
            assert r["bonds"] == "network", r["name"]
    for name in STATE_DOUBT:
        for state in STATES:
            assert state_truth(g.rows[name], state) is (True if state == g.rows[name]["state"] else None)
    assert state_truth(g.rows["water"], "gas") is False and state_truth(g.rows["water"], "plasma") is None
    assert basis_truth(g.rows["air"], "mass") is False and basis_truth(g.rows["bronze"], "volume") is False
    assert basis_truth(g.rows["earth_crust"], "mass") is None and basis_truth(g.rows["bronze"], "element_mass") is None
    assert basis_truth(g.rows["earth_crust"], "volume") is False and basis_truth(g.rows["earth_crust"], "element_mass") is True


def test_places_are_an_open_list(g):
    for r in g.pure:
        impossible = impossible_places(r)
        assert impossible <= set(PLACES) and not impossible & set(r["occurs_in"]), r["name"]
        for place in PLACES:
            want = True if place in r["occurs_in"] else False if place in impossible else None
            assert g.place_truth(r, place) is want
    # listed: yes. Not listed: undecided, unless a rule excludes the place
    assert g.check("check oxygen occurs_in mars", "yes").ok and g.check("check water occurs_in bones", "yes").ok
    for claim in ("water occurs_in core", "neon occurs_in sun", "atp occurs_in plants", "iron occurs_in crust",
                  "water occurs_in interstellar_clouds", "helium occurs_in interstellar_clouds", "air occurs_in core",
                  "phosphoric_acid occurs_in dna", "water occurs_in zzz",
                  # the name of an element: the sun and the crust list their elements under it
                  "silicon occurs_in sun", "iron occurs_in sun", "oxygen occurs_in crust", "gold occurs_in dna",
                  "argon occurs_in bones"):
        assert g.check(f"check {claim}", "no") == Verdict(False, None, "cannot judge that claim"), claim
    for claim in ("glucose occurs_in sun", "aspirin occurs_in core", "quartz occurs_in dna", "water_ice occurs_in blood",
                  "atomic_hydrogen occurs_in bones", "methane occurs_in sun", "polyethylene occurs_in dna"):
        assert g.check(f"check {claim}", "no").ok, claim
    for m in g.mixtures:  # no ``no`` about an element where a mixture lists that element
        for comp in m["main_parts"]:
            if comp in g.rows and not g.rows[comp]["mixture"] and m["basis"] == "element_mass":
                assert not impossible_places(g.rows[comp]), comp
    # several places: all listed is a yes, one excluded a no
    assert g.check("check water occurs_in ocean rivers", "yes").ok
    assert g.check("check water occurs_in " + " ".join(g.rows["water"]["occurs_in"]), "yes").ok
    assert g.check("check glucose occurs_in food sun", "no").ok
    assert g.check("check water occurs_in ocean ocean", "yes").expected is None


def test_main_parts_claims(g):
    for claim in ("venus_atmosphere main_parts sulfur_dioxide", "human_body main_parts water", "air main_parts neon",
                  "steel main_parts cementite", "seawater main_parts sodium_bromide", "blood_plasma main_parts sodium_chloride",
                  "blood_plasma main_parts glucose", "sun main_parts atomic_hydrogen", "steel main_parts silicon",
                  "earth_core main_parts carbon", "brass main_parts tin", "air part oxygen", "air main_parts zzz"):
        assert g.check(f"check {claim}", "no") == Verdict(False, None, "cannot judge that claim"), claim
    for claim in ("steel main_parts nickel", "sun main_parts gold", "earth_crust main_parts uranium",
                  "air main_parts glucose", "bronze main_parts glycine", "bronze main_parts copper 9 0 tin 1 0",
                  "air part oxygen 2 1"):
        assert g.check(f"check {claim}", "no").ok, claim
    for claim in ("bronze main_parts copper", "bronze main_parts copper tin", "bronze main_parts copper 8 8 tin 1 2",
                  "bronze main_parts tin 1 2 copper 8 8", "air part oxygen 2 0 point 9", "earth_crust main_parts oxygen"):
        assert g.check(f"check {claim}", "yes").ok, claim
    for m in g.mixtures:
        not_parts = g.not_parts[m["name"]]
        assert len(not_parts) >= 40 and not set(not_parts) & set(m["main_parts"]), m["name"]
        if m["basis"] == "element_mass":
            assert all(n in SYMBOL_OF for n in not_parts)
            assert not set(not_parts) & set(PART_DOUBT.get(m["name"], ()))
        else:  # only what shares no place with the mixture and belongs to none of its materials
            families = {c for comp in m["main_parts"] for c in MATERIALS.get(comp, ())}
            for n in not_parts:
                assert not set(g.rows[n]["occurs_in"]) & set(m["occurs_in"]) and g.rows[n]["class"] not in families


def test_yes_and_no_lines_tell_the_truth(g, lines):
    holders: dict[str, set[str]] = {}  # row -> the places of the mixtures that hold it
    for m in g.mixtures:
        for comp in m["main_parts"]:
            holders.setdefault(comp, set()).update(m["occurs_in"])
    seen = set()
    for ln in lines + g.generate(random.Random(21), 20000):
        if ln.kind != "yesno":
            continue
        words = ln.prompt.split()[1:]
        yes = ln.answer == "yes"
        if words[0] == "heavier_molecule":
            rows = [g.rows[w] for w in words[1:]]
            assert all(r["bonds"] == "covalent" and not r["unit"] and not r["mixture"] for r in rows), ln.text
            a, b = (exact_mass(parse(r["formula"]), g.weights) for r in rows)
            assert (a > b) == yes and abs(a - b) > Fraction(1, 100) * max(a, b), ln.text
            continue
        if words[0] not in g.rows:
            continue
        row, key, value = g.rows[words[0]], words[1], words[2:]
        seen.add((key, ln.answer))
        if key == "class":
            assert class_truth(row, value[0]) is yes, ln.text
            if not yes:  # never a class that may apply
                els = set(parse(row["formula"]))
                assert value[0] != "element" or len(els) > 1
                assert value[0] != "oxide" or "O" not in els or len(els) == 1 or {"C", "H"} <= els
                assert value[0] != "organic" or "C" not in els
                assert value[0] != "salt" or row["bonds"] != "ionic"
        elif key == "bonds":
            assert bond_truth(row, value[0]) is yes, ln.text
            assert yes or (row["bonds"], value[0]) != ("network", "covalent")
        elif key == "state":
            assert (row["state"] == value[0]) == yes and (yes or row["name"] not in STATE_DOUBT), ln.text
        elif key == "basis":
            assert (row["basis"] == value[0]) == yes and (yes or {row["basis"], value[0]} != {"mass", "element_mass"})
        elif key == "occurs_in":
            assert (value[0] in row["occurs_in"]) == yes, ln.text
            if not yes:  # never a place of a mixture that holds the row, never the outside of a place it is in
                assert value[0] not in holders.get(row["name"], ()), ln.text
                assert value[0] not in {INSIDE.get(p) for p in row["occurs_in"]}, ln.text
                assert not row["mixture"] and value[0] in impossible_places(row), ln.text
        elif key == "part":
            assert (parse_num(value[1:]) == row["main_parts"][value[0]]) == yes, ln.text
        elif key == "main_parts":
            assert (value[0] in row["main_parts"]) == yes, ln.text
            if not yes and row["basis"] != "element_mass":
                assert not set(g.rows[value[0]]["occurs_in"]) & set(row["occurs_in"]), ln.text
            if not yes and row["basis"] == "element_mass":
                assert value[0] in SYMBOL_OF, ln.text
    assert {(k, a) for k in ("class", "bonds", "state", "basis", "occurs_in", "main_parts", "part")
            for a in ("yes", "no")} <= seen


def test_heavier_molecule_compares_molecules_only(g):
    assert len(g.molecules) > 150 and all(r["bonds"] == "covalent" and not r["unit"] for r in g.molecules)
    for prompt in ("heavier_molecule cellulose glucose", "heavier_molecule water diamond", "heavier_molecule polyethylene water",
                   "heavier_molecule sulfur iron", "heavier_molecule sodium_chloride water", "heavier_molecule argon water",
                   "heavier_molecule u water", "heavier_molecule h 2 o 1 methane"):
        verdict = g.check(prompt, prompt.split()[1])
        assert not verdict.ok and verdict.expected is None, prompt
        assert g.check("check " + prompt, "yes").expected is None
    assert g.check("heavier_molecule sulfur water", "sulfur").ok  # s8 rings are molecules
    assert g.check("heavier_molecule glucose fructose", "same").ok  # equal formulas; never generated


def test_lists_are_short_and_whole(g):
    assert 10 <= len(g.lists) <= 60
    prompts = [p for p, _ in g.lists]
    assert len(prompts) == len(set(prompts))
    for prompt, members in g.lists:
        assert 1 <= len(members) <= MAX_LIST and members == sorted(members)
        words = prompt.split()[1:]
        for r in g.pure:
            verdicts = []
            for k, v in zip(words[::2], words[1::2]):
                if k == "occurs_in":  # an open place counts only where the other condition rules the rest out
                    verdicts.append(v in r["occurs_in"] if v in CLOSED_PLACES else g.place_truth(r, v))
                else:
                    verdicts.append({"class": class_truth, "bonds": bond_truth, "state": state_truth}[k](r, v))
            assert (r["name"] in members) == all(v is True for v in verdicts), (prompt, r["name"])
            assert False in verdicts or None not in verdicts, (prompt, r["name"])  # nothing is left undecided
    lists = dict(g.lists)
    assert lists["substances occurs_in dna"] == ["adenine", "cytosine", "deoxyribose", "guanine", "thymidine", "thymine"]
    assert lists["substances class element state liquid"] == ["bromine", "mercury"]
    assert lists["substances class nucleobase occurs_in dna"] == ["adenine", "cytosine", "guanine", "thymine"]
    # a list over an open place, or with a class that leaves rows undecided, is not asked and not answered
    for prompt in ("substances occurs_in core", "substances occurs_in rivers", "substances class sugar occurs_in animals",
                   "substances class oxide occurs_in crust", "substances class organic state gas",
                   "substances class element state gas", "substances state solid", "substances class amino_acid",
                   "substances bonds covalent", "substances class mineral", "substances class ice class ice"):
        assert prompt not in lists
        verdict = g.check(prompt, "x")
        assert not verdict.ok and verdict.expected is None and verdict.reason == "no answer in the table", prompt


def test_reverse_lookup_leaves_out_ices_and_repeat_units(g):
    assert g.check("substance formula c 2 h 4", "ethylene").ok and g.check("substance formula c 8 h 8", "styrene").ok
    assert g.check("substance formula h 2 o 1", "water").ok and g.check("substance formula c 1", "carbon charcoal diamond graphite").ok
    assert g.check("substance formula fe 2 o 3", "hematite iron_oxide").ok
    assert g.check("substance formula h 1 cl 1", "hydrogen_chloride").ok
    assert g.check("substance formula c 6 h 1 0 o 5", "cellulose").expected is None  # only repeat units
    assert g.check("substance formula water", "water").expected is None  # a name is no formula


def test_facts_cover_every_row_and_key(g):
    prompts = {f.prompt for f in g.facts()}
    assert all(f.kind == "fact" for f in g.facts())
    for r in g.pure:
        for key in ("formula", "composition", "atoms", "elements", "molar_mass", "mass_percent", "class", "bonds", "state",
                    "occurs_in", "mixture"):
            assert f"{r['name']} {key}" in prompts
        assert (f"{r['name']} unit" in prompts) == r["unit"]
    for r in g.mixtures:
        assert {f"{r['name']} {k}" for k in ("main_parts", "basis", "state", "occurs_in", "mixture")} <= prompts
        assert {f"{r['name']} part {c}" for c in r["main_parts"]} <= prompts
    assert {p for p, _ in g.lists} <= prompts


def test_docstring_examples_are_real_lines(g):
    doc = module.__doc__
    examples = re.findall(r"q [a-z0-9_ ]+\. a [a-z0-9_ ]+\.", doc)
    assert len(examples) >= 25
    for example in examples:
        assert is_dense(example)
        prompt, answer = split_line(example)
        assert g.check(prompt, answer) == Verdict(True, answer, ""), example
    records = {r.text for r in g.records()}
    assert "air mixture. main_parts nitrogen 7 8 point 1 oxygen 2 0 point 9 argon 0 point 9 3 carbon_dioxide 0 point 0 4. " \
           "basis volume. state gas. occurs_in atmosphere." in records


def test_owned_but_unanswerable_prompts_fail_without_a_guess(g):
    for prompt, answer in [("substance formula xx 9", "water"), ("substance formula c 9 9", "water"),
                           ("heavier_molecule water air", "water"), ("molar_mass air", "2 9"),
                           ("check molar_mass c 1 2 point 0 1 1", "yes"), ("check water occurs_in", "yes"),
                           ("check substance formula h 2 o 1 water", "yes"), ("air part gold", "1"),
                           ("substances class sugar occurs_in core", "glucose"), ("mass_percent water", "5 0"),
                           ("substances state solid", "water"), ("substances occurs_in core", "iron nickel"),
                           ("heavier_molecule gold water", "gold"), ("molar_mass h 0", "0"),
                           ("mass_percent h 0 hydrogen", "0"), ("check halite class salt", "yes"),
                           ("check methane class gas", "yes"), ("check air part oxygen", "yes"),
                           ("check nitrogen_dioxide state gas", "no"), ("check earth_crust basis mass", "yes"),
                           ("atoms xx 9", "9"), ("molar_mass c " + "9 " * 300, "1")]:
        assert g.owns(prompt), prompt
        verdict = g.check(prompt, answer)
        assert not verdict.ok and verdict.expected is None and verdict.reason not in ("", "not my question"), prompt
