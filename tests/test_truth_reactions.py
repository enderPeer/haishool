import hashlib
import importlib
import json
import os
import random
import re
import subprocess
import sys
from collections import Counter
from math import gcd
from pathlib import Path

import pytest

from haishool.truth import Gate, Verdict, is_dense, num, parse_num, split_line
from haishool.truth import formula as fm
from haishool.truth.reactions import (ALIASES, ATOM_MASS_U, ELECTRON_MASS_U, ELEMENT_NAMES, FILTERS, KEYS, KINDS,
                                      LEPTONS, MAX_LIST, NEEDS, NEUTRON_MASS_U, NUCLEAR_KEYS, NUCLIDE_KEYS, PARTICLES,
                                      RECORD_KEYS, SHARED, SUBSTANCES, SUMS, SYMBOLS, TABLE, U_MEV, UNSURE,
                                      YES_NO_SLACK, Z, R, ReactionsGate, balance, count_tokens, gate, is_balanced,
                                      mass_u, names_asked, nuclide, nuclide_name, parse_equation, parse_species,
                                      rows, species, untrained)

ROOT = Path(__file__).resolve().parents[1]
JSONL = ROOT / "data" / "truth-v5" / "reactions.jsonl"
NOT_MINE = ["turkey capital", "q spoon color", "spoon color", "carbon protons", "water molar_mass", "calc 1 2 plus 7",
            "check 1 2 plus 7 equals 2 0", "solve 3 x plus 4 equals 1 9", "check carbon protons 6",
            "gravity seed 7 step 2 0 clumps", "element number 6", "turkey borders greece", "turkey hop led_zeppelin",
            "photosynthesis meaning", "photosynthesis kind", "thermite colour", "methane plus spoon gives",
            "count_atoms 2 x he4 helium", "balance", "reactions", "reactions colour red", "gives", "",
            # a value the table does not hold, a list too long to be a line, a filter that is none
            "reactions kind spoon", "reactions where moon", "reactions kind combustion", "reactions needs nothing",
            "reactions energy exothermic", "reactions kind photosynthesis",
            # keys that are not trained, keys of the other kind of row
            "chlorine_and_water energy", "marble_and_sulfuric_acid energy", "methane_combustion q_mev",
            # numbers that are not written digit by digit, from 1 on, without a leading zero
            "check 0 1 x h 2 gives 1 x h 2 balanced", "check 0 x h 2 gives 1 x h 2 balanced",
            "check 12 x h 2 gives 1 x h 2 balanced", "check 1 x h 0 gives 1 x o 0 balanced",
            "check 1 x h 0 2 gives 1 x h 2 balanced", "check ² x h 2 gives 1 x h 2 balanced",
            "balance h ٢ plus o 2 gives h 2 o 1", "count_atoms ² x h 2 o 1 hydrogen",
            # no nuclides: the mass number is below the charge, or has a leading zero
            "check 1 x h2 plus 1 x o2 gives 1 x o2 plus 1 x h2 balanced", "check 1 x h0 gives 1 x h0 balanced",
            "nuclide o2 name", "nuclide he1 charge", "nuclide h02 name", "nuclide u235 colour", "nuclide u235",
            "count_nuclear 2 x h 2 o 1 charge", "count_nuclear 2 x he4 helium", "count_nuclear 2 x he4"]


def tokens(line: str) -> int:
    """As ``haishool.student.tokens`` counts (that module needs torch, which this machine lacks)."""
    return len(line.replace(".", " . ").split())


@pytest.fixture(scope="module")
def g():
    return gate()


@pytest.fixture(scope="module")
def lines(g):
    return g.generate(random.Random(11), 6000)


# ----- the table


def test_gate_follows_the_protocol(g):
    assert isinstance(g, Gate)
    assert g.topic == "reactions"
    assert set(NUCLEAR_KEYS) <= set(KEYS) and set(RECORD_KEYS) < set(NUCLEAR_KEYS)
    assert {"balance", "check", "count_atoms", "count_nuclear", "nuclide", "reactions", "gives"} <= set(KEYS)
    assert set(FILTERS) <= {"reactant", "product"} | set(RECORD_KEYS) and set(NUCLIDE_KEYS) == {"name"} | set(SUMS)


def test_table_shape():
    assert 160 <= len(TABLE) <= 190
    names = [r.name for r in TABLE]
    assert len(set(names)) == len(names)
    for r in TABLE:
        assert re.fullmatch(r"[a-z][a-z0-9_]*", r.name), r.name
        assert r.kind in KINDS, r.name
        assert r.energy in ("exothermic", "endothermic"), r.name
        assert r.needs and set(r.needs) <= set(NEEDS), r.name
        assert ("nothing" not in r.needs) or r.needs == ("nothing",), r.name
        assert r.where and all(re.fullmatch(r"[a-z0-9_]+", w) for w in r.where), r.name
        assert len({s.nuclear for s in r.species}) == 1, f"{r.name} mixes chemical and nuclear species"
        assert r.keys == (NUCLEAR_KEYS if r.nuclear else RECORD_KEYS)
    assert sum(r.nuclear for r in TABLE) >= 40
    wanted = {"methane_combustion", "photosynthesis", "rusting", "sodium_and_water", "thermite", "limestone_calcination",
              "haber_process", "electrolysis_of_water", "hydrogen_combustion", "respiration_of_glucose",
              "fermentation_of_glucose", "neutralisation_hcl_naoh", "baking_soda_and_vinegar",
              # nuclear
              "deuterium_formation", "helium_3_formation", "helium_3_fusion", "triple_alpha_process",
              "cno_carbon_12_proton_capture", "carbon_12_alpha_capture", "uranium_235_fission", "neutron_beta_decay",
              "carbon_14_decay", "uranium_238_alpha_decay", "tritium_decay", "deuterium_tritium_fusion",
              # added by the fact check: second branches, the closed cno cycle, the first nuclei
              "deuterium_deuterium_fusion_tritium_branch", "cno_carbon_13_proton_capture", "proton_neutron_capture",
              "potassium_40_electron_capture", "sodium_peroxide_formation", "malachite_decomposition"}
    assert wanted <= set(names)
    assert set(KINDS) == {r.kind for r in TABLE}, "a documented kind has no reaction, or the other way round"


def test_labels_the_fact_check_settled(g):
    by = g.by_name
    # plaster of paris is the half hydrate, not the dry sulfate
    assert by["plaster_setting"].reactant_names == ["plaster_of_paris", "water"]
    assert by["plaster_setting"].equation == "1 x ca 2 s 2 o 9 h 2 plus 3 x h 2 o 1 gives 2 x ca 1 s 1 o 6 h 4"
    assert fm.parse("(CaSO4)2H2O") == {"Ca": 2, "S": 2, "O": 9, "H": 2}
    assert by["thermite"].needs == ("heat",) and by["phosphorus_combustion"].needs == ("nothing",)
    # every row called a precipitation is one; ammonia and an acid is acid_base whatever the acid
    for r in TABLE:
        if "precipitation" in r.name:
            assert r.kind == "precipitation", r.name
    assert {by[n].kind for n in ("ammonium_chloride_formation", "ammonium_nitrate_formation",
                                 "ammonium_sulfate_formation")} == {"acid_base"}
    assert by["sabatier_reaction"].kind == "redox" and by["magnesium_hydroxide_from_seawater"].kind == "precipitation"
    # every synthesis but urea's has one product
    assert [r.name for r in TABLE if r.kind == "synthesis" and len(r.products) != 1] == ["urea_synthesis"]
    # a word of ``where`` has one meaning: the two kinds of flares are not one place
    assert g.index["where"]["flares"] == ["magnesium_combustion"]


def test_every_reaction_passes_its_own_balance_test():
    for r in TABLE:
        assert is_balanced(list(r.reactants), list(r.products)), r.name
        g_ = 0
        for s in r.species:
            g_ = gcd(g_, s.coef)
        assert g_ == 1, f"{r.name}: coefficients are not the smallest"


def test_chemical_balance_is_counted_from_the_formulas():
    """Independent of the module's own totals: count the atoms again from the standard formulas."""
    for row in rows():
        if row["nuclear"]:
            continue
        sides = []
        for side in (row["reactants"], row["products"]):
            total = Counter()
            for coef, formula in side:
                for el, n in fm.parse(formula).items():
                    total[el] += coef * n
            sides.append(total)
        assert sides[0] == sides[1], row["name"]


def _nuclear_numbers(token: str) -> tuple[int, int, int]:
    """(A, Z, L) of a nuclide or particle token, read without the module's own parser."""
    particle = {"n1": (1, 0, 0), "e": (0, -1, 1), "positron": (0, 1, -1), "neutrino": (0, 0, 1),
                "antineutrino": (0, 0, -1), "gamma": (0, 0, 0)}
    if token in particle:
        return particle[token]
    m = re.fullmatch(r"([a-z]{1,2})([0-9]+)", token)
    return int(m.group(2)), SYMBOLS.index(m.group(1).capitalize()) + 1, 0


def test_nuclear_balance_keeps_mass_number_charge_and_lepton_number():
    assert PARTICLES == {"n1": (1, 0), "e": (0, -1), "positron": (0, 1), "neutrino": (0, 0), "antineutrino": (0, 0),
                         "gamma": (0, 0)}
    assert LEPTONS == {"e": 1, "positron": -1, "neutrino": 1, "antineutrino": -1}
    seen = 0
    for row in rows():
        if not row["nuclear"]:
            continue
        seen += 1
        sums = []
        for side in (row["reactants"], row["products"]):
            total = [0, 0, 0]
            for coef, token in side:
                azl = _nuclear_numbers(token)
                assert azl[0] >= azl[1] or token in PARTICLES, token
                total = [t + coef * x for t, x in zip(total, azl)]
            sums.append(total)
        assert sums[0] == sums[1], row["name"]
    assert seen >= 40
    assert nuclide("n1") == (1, 0) and nuclide("n14") == (14, 7) and nuclide("he4") == (4, 2)
    assert nuclide("u235") == (235, 92) and nuclide("e") == (0, -1) and nuclide("xx3") is None
    # no nuclide has fewer nucleons than protons, a leading zero or digits outside 0-9
    assert nuclide("o2") is None and nuclide("he1") is None and nuclide("h0") is None and nuclide("h02") is None
    assert nuclide("h²") is None and nuclide("h٢") is None and nuclide("h2") == (2, 1)
    assert species("antineutrino").counts == {"a": 0, "z": 0, "l": -1} and species("he4").counts == {"a": 4, "z": 2, "l": 0}
    assert species("gamma").free and not species("neutrino").free and not species("H2O").free


def test_nuclear_energy_is_computed_from_the_masses(g):
    """``q_mev`` again from the atomic masses, with the textbook rules for each kind of row."""
    atom = lambda t: NEUTRON_MASS_U if t == "n1" else ATOM_MASS_U[t]  # noqa: E731
    for r in TABLE:
        if not r.nuclear:
            assert r.q_mev is None
            continue
        assert r.energy == ("exothermic" if r.q_mev > 0 else "endothermic"), r.name
        assert parse_num(r.value("q_mev")) == pytest.approx(r.q_mev, rel=6e-4), r.name
        nuclei = [[(s.coef, s.text) for s in side if s.text not in LEPTONS and s.text != "gamma"]
                  for side in (r.reactants, r.products)]
        # in atomic masses the electrons of a beta minus decay and of an electron capture are
        # already counted; a positron costs two electron masses
        positrons = sum(s.coef for s in r.products if s.text == "positron")
        q = (sum(c * atom(t) for c, t in nuclei[0]) - sum(c * atom(t) for c, t in nuclei[1])
             - 2 * positrons * ELECTRON_MASS_U) * U_MEV
        assert q == pytest.approx(r.q_mev, abs=1e-6), r.name
    known = {"deuterium_tritium_fusion": 17.59, "proton_neutron_capture": 2.224, "uranium_238_alpha_decay": 4.270,
             "carbon_14_decay": 0.1565, "neutron_beta_decay": 0.782, "triple_alpha_process": 7.275,
             "deuterium_formation": 0.420, "tritium_decay": 0.01859, "beryllium_8_formation": -0.0918}
    for name, q in known.items():
        assert g.by_name[name].q_mev == pytest.approx(q, rel=2e-3), name
    assert [r.name for r in TABLE if r.energy == "endothermic" and r.nuclear] == ["beryllium_8_formation"]
    assert g.by_name["beryllium_8_formation"].value("q_mev") == "minus 0 point 0 9 1 8 4"
    assert g.by_name["deuterium_tritium_fusion"].value("q_mev") == "1 7 point 5 9"
    assert mass_u("gamma") == 0 and mass_u("positron") == mass_u("e") == ELECTRON_MASS_U
    with pytest.raises(ValueError, match="masses give"):  # a nuclear row cannot claim the wrong sign
        R("x", "fusion", "2 he4", "be8", "exothermic", "heat", "stars")
    assert set(ATOM_MASS_U) == {s.text for r in TABLE for s in r.species if s.nuclear} - set(PARTICLES)


def test_masses_match_mendeleev_and_scipy():
    pytest.importorskip("mendeleev")
    constants = pytest.importorskip("scipy.constants")
    from mendeleev.db import get_session
    from mendeleev.models import Isotope

    table = {(i.atomic_number, i.mass_number): i.mass for i in get_session().query(Isotope).all()}
    for token, mass in ATOM_MASS_U.items():
        a, z = nuclide(token)
        assert table[(z, a)] == pytest.approx(mass, rel=1e-10), token
    physical = constants.physical_constants
    assert physical["neutron mass in u"][0] == pytest.approx(NEUTRON_MASS_U, rel=1e-9)
    assert physical["electron mass in u"][0] == pytest.approx(ELECTRON_MASS_U, rel=1e-9)
    assert physical["atomic mass constant energy equivalent in MeV"][0] == pytest.approx(U_MEV, rel=1e-9)


def test_chemical_rows_have_exactly_one_smallest_balance(g):
    for r in TABLE:
        coefs = [s.coef for s in r.species]
        found = balance(list(r.reactants), list(r.products))
        if not r.nuclear:
            assert found == coefs, r.name
        else:
            assert found in (None, coefs), r.name
    assert len(g.balanceable) >= 140
    # a free species (gamma) or a species on both sides leaves no unique balance
    assert g.balanced["helium_3_formation"] is None and g.balanced["uranium_235_fission"] is None
    assert g.balanced["helium_3_fusion"] == [2, 1, 2]
    # with the lepton number a beta decay has one balance
    assert g.balanced["neutron_beta_decay"] == [1, 1, 1, 1] and g.balanced["deuterium_formation"] == [2, 1, 1, 1]
    # two rows with the same species on the same sides are the same equation
    for r in TABLE:
        first = g.by_species[(frozenset(s.key for s in r.reactants), frozenset(s.key for s in r.products))]
        assert sorted((s.key, s.coef) for s in first.species) == sorted((s.key, s.coef) for s in r.species), r.name


def test_balance_agrees_with_sympy(g):
    sympy = pytest.importorskip("sympy")
    checked = 0
    for r in TABLE:
        keys = sorted({k for s in r.species for k in s.counts})
        m = sympy.Matrix([[(1 if i < len(r.reactants) else -1) * s.counts.get(k, 0) for i, s in enumerate(r.species)]
                          for k in keys])
        null = m.nullspace()
        if g.balanced[r.name] is None:
            assert len(null) != 1, r.name
            continue
        assert len(null) == 1, r.name
        v = null[0] * sympy.ilcm(*[x.q for x in null[0]])
        v = [int(x) for x in v]
        v = [abs(x) for x in v] if all(x <= 0 for x in v) else v
        d = 0
        for x in v:
            d = gcd(d, x)
        assert [x // d for x in v] == g.balanced[r.name], r.name
        checked += 1
    assert checked >= 140


def test_symbols_and_element_names_match_mendeleev():
    pytest.importorskip("mendeleev")
    from mendeleev.db import get_session
    from mendeleev.models import Element

    table = {sym: (number, name.lower()) for number, sym, name in
             get_session().query(Element.atomic_number, Element.symbol, Element.name).all()}
    for sym in SYMBOLS:
        assert Z[sym.lower()] == table[sym][0], sym
    for sym, name in ELEMENT_NAMES.items():
        assert table[sym.capitalize()][1] == name, sym
    assert set(ELEMENT_NAMES) == set(Z), "an element has a symbol and no name"
    elements = ROOT / "data" / "truth-v5" / "elements.jsonl"
    if elements.exists():  # one spelling across round 5
        for ln in elements.open(encoding="utf-8"):
            row = json.loads(ln)
            if row["symbol"] in ELEMENT_NAMES:
                assert ELEMENT_NAMES[row["symbol"]] == row["name"], row["symbol"]


def test_substances(g):
    assert len(set(SUBSTANCES.values())) == len(SUBSTANCES), "two formulas share a name"
    compositions = {}
    for formula, name in SUBSTANCES.items():
        counts = fm.parse(formula)
        assert re.fullmatch(r"[a-z][a-z0-9_]*", name), name
        for el in counts:
            assert el.lower() in Z and el.lower() in ELEMENT_NAMES, f"{formula}: {el} has no name"
        key = tuple(sorted(counts.items()))
        assert key not in compositions, f"{formula} and {compositions.get(key)} have the same atoms"
        compositions[key] = formula
    used = {s.formula for r in TABLE for s in r.species if not s.nuclear}
    assert used == set(SUBSTANCES), f"unused or missing: {used ^ set(SUBSTANCES)}"
    assert not g.names & set(g.by_name), "a substance has the name of a reaction"
    assert not (g.names | set(g.by_name)) & {"balance", "check", "count_atoms", "count_nuclear", "nuclide", "reactions",
                                             "plus", "gives", "x", "q", "a", "none"}
    # a second name for a formula: used by a row, and not a name of another formula
    for spelling, (formula, name) in ALIASES.items():
        assert formula in SUBSTANCES and name not in SUBSTANCES.values() and name in g.names, spelling
        assert species(spelling).name == name and species(spelling).text == species(formula).text
    gas = [r.name for r in TABLE if "hydrogen_chloride" in r.reactant_names + r.product_names]
    assert gas == ["hydrogen_chloride_synthesis", "ammonium_chloride_formation"]
    assert len(g.index["reactant"]["hydrochloric_acid"]) >= 8 and "hydrochloric_acid" in g.index["product"]
    # a species read from a line is named by its atoms; an unknown one has no name
    assert parse_species("c 1 h 4", with_coef=False).name == "methane"
    assert parse_species("2 x n 2 h 4 c 1 o 1", with_coef=True).name == "urea"  # the atoms in any order
    with pytest.raises(KeyError):
        parse_species("c 7 h 1", with_coef=False).name
    assert nuclide_name("u235") == "uranium_235" and nuclide_name("sc45") == "scandium_45"
    assert nuclide_name("h1") == "proton" and nuclide_name("n1") == "neutron" and nuclide_name("e") == "electron"
    with pytest.raises(ValueError):
        nuclide_name("spoon")


def test_names_agree_with_the_substances_table():
    """One name, one formula across round 5: a name both tables hold has the same atoms and the
    same dense tokens in both."""
    path = ROOT / "data" / "truth-v5" / "substances.jsonl"
    if not path.exists():
        pytest.skip("the substances table is not here")
    theirs = {}
    for ln in path.open(encoding="utf-8"):
        row = json.loads(ln)
        if row.get("formula") and not row.get("mixture"):
            theirs[row["name"]] = row
    mine = dict({name: formula for formula, name in SUBSTANCES.items()},
                **{name: formula for formula, name in ALIASES.values()})
    shared = set(mine) & set(theirs)
    assert len(shared) >= 60 and {"urea", "hydrogen_chloride", "gypsum", "ethanol", "acetic_acid", "sulfur"} <= shared
    for name in sorted(shared):
        assert fm.same(fm.parse(mine[name]), fm.parse(theirs[name]["formula"])), name
        assert fm.dense(mine[name]) == fm.dense(theirs[name]["formula"]), name
        if "dense" in theirs[name]:
            assert fm.parse_dense(theirs[name]["dense"]) == fm.parse(mine[name]), name


def test_jsonl_is_the_table(g):
    built = [json.loads(ln) for ln in JSONL.open(encoding="utf-8")]
    assert built == rows()
    asked = {r.name for r in g.names_unique}
    for row in built:
        left = [species(f, c) for c, f in row["reactants"]]
        right = [species(f, c) for c, f in row["products"]]
        assert is_balanced(left, right), row["name"]
        assert row["equation"] == " plus ".join(s.dense() for s in left) + " gives " + " plus ".join(s.dense() for s in right)
        assert parse_equation(row["equation"], with_coef=True) is not None, row["name"]
        assert {"name", "kind", "reactants", "products", "energy", "needs", "where", "reactant_names",
                "product_names", "notes", "names", "q_mev", "untrained"} <= set(row)
        # the ``gives`` line is there only where it is asked, so a build cannot write two answers
        assert bool(row["names"]) == (row["name"] in asked), row["name"]
        if row["names"]:
            prompt, _, answer = row["names"].partition(" gives ")
            assert g.check(prompt + " gives", answer).ok, row["name"]
        assert (row["q_mev"] is not None) == row["nuclear"]
        assert row["untrained"] == list(untrained(row["name"]))
    assert sum(not row["names"] for row in built) >= 15


def test_shared_names_leave_earlier_rounds_alone(g):
    """A reaction that is also a round 1-3 record must not ask a key that record already answers."""
    files = [ROOT / "data" / f for f in ("records-r1.jsonl", "records-r2.jsonl", "records-r3-all.jsonl")]
    if not all(f.exists() for f in files):
        pytest.skip("round 1-3 records not here")
    for f in files:
        for ln in f.open(encoding="utf-8"):
            row = json.loads(ln)
            if row["obj"] in g.by_name:
                clash = set(row["values"]) & set(NUCLEAR_KEYS)
                assert clash <= set(SHARED.get(row["obj"], ())), f"{row['obj']}: {clash}"


def test_untrained_keys_are_nowhere(g):
    """A key left to another round, or without a defensible value, is not in the record, not a
    question, not owned and not in a list."""
    assert SHARED == {"photosynthesis": ("kind",)}
    assert UNSURE == {"chlorine_and_water": ("energy",), "marble_and_sulfuric_acid": ("energy",)}
    records = {ln.meta["name"]: ln.text for ln in g.records()}
    for name, keys in {**SHARED, **UNSURE}.items():
        assert untrained(name) == keys and name in g.by_name
        for key in keys:
            assert f" {key} " not in records[name], records[name]
            assert not g.owns(f"{name} {key}")
            assert all(ln.prompt != f"{name} {key}" for ln in g.all_facts())
            if key in g.index:
                assert all(name not in names for names in g.index[key].values())
    assert records["photosynthesis"].startswith("photosynthesis reaction. reactants carbon_dioxide water. products ")
    assert "photosynthesis" not in g.index["kind"] and not g.owns("reactions kind photosynthesis")
    assert " energy " in records["limewater_test"] and untrained("limewater_test") == ()


# ----- records and generated lines


def test_records(g):
    recs = g.records()
    assert len(recs) >= len(TABLE)
    assert len({ln.text for ln in recs}) == len(recs)
    for ln in recs:
        assert ln.kind == "record" and ln.topic == "reactions"
        assert is_dense(ln.text), ln.text
        assert tokens(ln.text) <= 80 and count_tokens(ln.text) == tokens(ln.text), ln.text
        assert split_line(ln.text) is None
        name, second = ln.text.split(".")[0].split()
        assert name in g.by_name and second in ("reaction", "reaction_more")
    assert recs[0].text == ("methane_combustion reaction. kind combustion. reactants methane oxygen. "
                            "products carbon_dioxide water. equation 1 x c 1 h 4 plus 2 x o 2 gives "
                            "1 x c 1 o 2 plus 2 x h 2 o 1. energy exothermic. needs spark. where stoves heaters engines.")
    by_name = {ln.meta["name"]: ln.text for ln in recs}
    assert "equation 2 x h1 gives 1 x h2 plus 1 x positron plus 1 x neutrino." in by_name["deuterium_formation"]
    assert "equation 1 x u235 plus 1 x n1 gives 1 x ba141 plus 1 x kr92 plus 3 x n1." in by_name["uranium_235_fission"]
    assert "reactants uranium_235 neutron. products barium_141 krypton_92 neutron." in by_name["uranium_235_fission"]
    assert "equation 2 x c 8 h 1 8 plus 2 5 x o 2 gives 1 6 x c 1 o 2 plus 1 8 x h 2 o 1." in by_name["octane_combustion"]
    assert by_name["deuterium_tritium_fusion"] == (
        "deuterium_tritium_fusion reaction. kind fusion. reactants deuterium tritium. products helium_4 neutron. "
        "equation 1 x h2 plus 1 x h3 gives 1 x he4 plus 1 x n1. energy exothermic. q_mev 1 7 point 5 9. "
        "needs heat pressure. where fusion_reactors early_universe.")
    for r in TABLE:
        assert (" q_mev " in by_name[r.name]) == r.nuclear, r.name
    # urea as the substances table writes it; the gas and the acid by their own names
    assert "gives 1 x c 1 h 4 n 2 o 1 plus 1 x h 2 o 1." in by_name["urea_synthesis"]
    assert "products hydrogen_chloride. equation 1 x h 2 plus 1 x cl 2 gives 2 x h 1 cl 1." in by_name["hydrogen_chloride_synthesis"]
    assert "reactants zinc hydrochloric_acid. " in by_name["zinc_and_hydrochloric_acid"]


def test_a_long_record_is_split_in_two(g):
    r = g.by_name["rusting"]
    long = type(r)(r.name, r.kind, r.reactants, r.products, r.energy, r.needs, tuple(f"place_{i}" for i in range(40)))
    first, more = long.record_lines()
    assert first.text.startswith("rusting reaction. kind corrosion. reactants ") and " equation " not in first.text
    assert more.text.startswith("rusting reaction_more. equation ")
    assert tokens(first.text) <= 80 and tokens(more.text) <= 80
    longer = type(r)(r.name, r.kind, r.reactants, r.products, r.energy, r.needs, tuple(f"place_{i}" for i in range(60)))
    with pytest.raises(ValueError, match="80 tokens"):  # a part over the limit is a table error, not a line
        longer.record_lines()


def test_generate_is_deterministic(g):
    a = [ln.text for ln in g.generate(random.Random(5), 800)]
    b = [ln.text for ln in gate().generate(random.Random(5), 800)]
    c = [ln.text for ln in g.generate(random.Random(6), 800)]
    assert a == b
    assert a != c and len(set(a) - set(c)) > 100
    assert [ln.text for ln in g.records()] == [ln.text for ln in gate().records()]


def test_generate_does_not_depend_on_the_hash_seed():
    """Sets of strings are ordered by a per-process hash seed; no line may depend on that order."""
    code = ("import hashlib, random; from haishool.truth.reactions import gate; g = gate(); "
            "t = [ln.text for ln in g.generate(random.Random(5), 1500) + g.all_facts() + g.records()]; "
            "print(hashlib.sha256('|'.join(t).encode()).hexdigest())")
    digests = set()
    for seed in ("0", "1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONPATH=str(ROOT))
        digests.add(subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True,
                                   text=True, check=True).stdout.strip())
    g_ = gate()
    here = [ln.text for ln in g_.generate(random.Random(5), 1500) + g_.all_facts() + g_.records()]
    assert digests == {hashlib.sha256("|".join(here).encode()).hexdigest()}


def test_generated_lines_are_dense_and_short(lines):
    assert len(lines) == 6000
    assert len({ln.text for ln in lines}) == len(lines)
    for ln in lines:
        assert ln.topic == "reactions" and ln.kind in ("fact", "calc", "yesno"), ln.text
        assert is_dense(ln.text), ln.text
        assert tokens(ln.text) <= 80, ln.text
        assert "." not in ln.prompt and "." not in ln.answer
        assert split_line(ln.text) == (ln.prompt, ln.answer)
    kinds = {ln.meta["q"]: ln.kind for ln in lines}
    assert kinds == {"fact": "fact", "nuclide": "fact", "names": "fact", "list": "fact", "balance": "calc",
                     "count_atoms": "calc", "count_nuclear": "calc", "check": "yesno"}
    # no prompt has two answers
    answers = {}
    for ln in lines:
        assert answers.setdefault(ln.prompt, ln.answer) == ln.answer, ln.prompt


def test_gate_agrees_with_its_own_answers(g, lines):
    for ln in lines + g.all_facts():
        v = g.check(ln.prompt, ln.answer)
        assert v.ok is True, (ln.text, v)
        assert v.expected == ln.answer, (ln.text, v)


def _wrong(ln) -> str:
    """A changed digit or word."""
    words = ln.answer.split()
    if ln.kind == "yesno":
        return "no" if ln.answer == "yes" else "yes"
    if ln.meta["q"] in ("count_atoms", "count_nuclear", "nuclide", "balance", "fact") and words[0].isdigit():
        words[0] = str(int(words[0]) % 9 + 1)  # another digit, never 0
        return " ".join(words)
    words[0] = "spoon"
    return " ".join(words)


def test_gate_rejects_wrong_answers(g, lines):
    for ln in lines + g.all_facts():
        wrong = _wrong(ln)
        assert wrong != ln.answer
        v = g.check(ln.prompt, wrong)
        assert v.ok is False, (ln.text, wrong)
        assert v.expected == ln.answer, (ln.text, v)
        assert g.check(ln.prompt, "").ok is False
        if ln.prompt.endswith(" q_mev"):  # a rounded number: a last digit more or less may be within the tolerance
            assert g.check(ln.prompt, ln.answer + " e 2").ok is False
            continue
        assert g.check(ln.prompt, ln.answer + " " + ln.answer.split()[-1]).ok is False  # the last word twice
        if ln.prompt.endswith(" equation"):  # there a dropped final count of 1 is the same formula
            assert g.check(ln.prompt, ln.answer.partition(" gives ")[0]).ok is False  # the right side dropped
        else:
            assert g.check(ln.prompt, " ".join(ln.answer.split()[:-1])).ok is False  # the last word dropped


def test_owns(g, lines):
    for ln in lines + g.all_facts():
        assert g.owns(ln.prompt), ln.prompt
    for prompt in NOT_MINE:
        assert not g.owns(prompt), prompt
        assert g.check(prompt, "yes") == Verdict(False, None, "not my question")
    assert g.owns("methane_combustion products") and g.owns("photosynthesis products")
    assert g.owns("balance h 2 plus o 2 gives h 2 o 1")
    assert g.owns("check 2 x h 2 plus 1 x o 2 gives 2 x h 2 o 1 balanced")
    assert g.owns("count_atoms 2 x h 2 o 1 hydrogen") and g.owns("reactions where airbags")
    assert g.owns("methane plus oxygen gives") and g.owns("water gives")
    assert g.owns("deuterium_tritium_fusion q_mev") and g.owns("nuclide he4 charge")
    assert g.owns("count_nuclear 2 x he4 plus 1 x n1 charge") and g.owns("reactions reactant gypsum")


def test_no_other_gate_owns_these_questions(g, lines):
    """Checked against the other round 5 gates as they are in this checkout."""
    mine = [ln.prompt for ln in lines + g.all_facts()]
    seen = 0
    for topic in ("maths", "elements", "substances", "forces"):
        try:
            other = importlib.import_module(f"haishool.truth.{topic}").gate()
            theirs = [ln.prompt for ln in other.generate(random.Random(3), 1500)]
        except Exception:  # another topic that is not there or does not load is not this test's business
            continue
        seen += 1
        assert [p for p in mine if other.owns(p)] == [], topic
        assert [p for p in theirs if g.owns(p)] == [], topic
    if not seen:
        pytest.skip("no other gate loads here")


# ----- the question kinds


def test_balance(g):
    q = "balance c 1 h 4 plus o 2 gives c 1 o 2 plus h 2 o 1"
    assert g.check(q, "1 2 1 2") == Verdict(True, "1 2 1 2", "")
    scaled = g.check(q, "2 4 2 4")
    assert not scaled.ok and scaled.expected == "1 2 1 2" and "smallest" in scaled.reason
    assert not g.check(q, "1 1 1 2").ok and not g.check(q, "1 2 1").ok and not g.check(q, "1 2 1 0").ok
    assert g.check(q, "1 x c 1 h 4 plus 2 x o 2 gives 1 x c 1 o 2 plus 2 x h 2 o 1").ok
    assert not g.check(q, "1 x c 1 h 4 plus 2 x o 2 gives 2 x h 2 o 1 plus 1 x c 1 o 2").ok  # the question's order
    # the answer follows the order of the species in the question
    assert g.check("balance o 2 plus c 1 h 4 gives h 2 o 1 plus c 1 o 2", "2 1 2 1").ok
    assert g.check("balance h 2 o gives h 2 plus o 2", "2 2 1").ok  # a missing count is 1
    # not in the table: the rule decides
    assert g.check("balance fe 1 plus cl 2 gives fe 1 cl 3", "2 3 2").ok
    assert g.check("balance c 3 h 8 plus o 2 gives c 1 o 2 plus h 2 o 1", "1 5 3 4").ok
    assert g.check("balance he3 gives he4 plus h1", "2 1 2").ok
    # two-digit coefficients: the expected answer is the full equation, digits alone are ambiguous
    octane = "balance c 8 h 1 8 plus o 2 gives c 1 o 2 plus h 2 o 1"
    full = "2 x c 8 h 1 8 plus 2 5 x o 2 gives 1 6 x c 1 o 2 plus 1 8 x h 2 o 1"
    assert g.check(octane, "2 2 5 1 6 1 8") == Verdict(False, full, "not a coefficient for every species")
    assert g.check(octane, "2 25 16 18") == Verdict(False, full, "not a coefficient for every species")  # not dense
    assert g.check(octane, full).ok
    assert g.check(q, "01 2 1 2").ok is False and g.check(q, "¹ 2 1 2").ok is False
    assert g.check("balance h 2 gives o 2", "1 1") == Verdict(False, None, "does not balance")
    assert parse_equation("c 1 h 4 plus he4 gives c 1 o 2", with_coef=False) is None  # mixed
    assert parse_equation("xx 2 plus o 2 gives xx 1 o 1", with_coef=False) is None  # not an element
    assert parse_species("2 x c 1 h 4", with_coef=False) is None and parse_species("c 1 h 4", with_coef=True) is None
    assert parse_species("1 2 x c 6 h 1 2 o 6", with_coef=True).coef == 12
    for bad in ("0 x h 2", "0 1 x h 2", "12 x h 2", "1 x h 0", "1 x h 0 2", "² x h 2", "1 x h ٢", "1 x H 2"):
        assert parse_species(bad, with_coef=True) is None, bad


def test_balance_without_a_unique_answer(g):
    # a reaction of the table: its own coefficients, whatever else the rule would allow
    free = "balance h2 plus h3 gives he4 plus n1"
    assert g.check(free, "1 1 1 1") == Verdict(True, "1 1 1 1", "")
    assert g.check(free, "2 2 2 2") == Verdict(False, "1 1 1 1", "balances, but not with the smallest coefficients")
    assert g.check(free, "1 1 1 2") == Verdict(False, "1 1 1 1", "does not balance")
    gamma = "balance h2 plus h1 gives he3 plus gamma"
    assert g.check(gamma, "1 1 1 1") == Verdict(True, "1 1 1 1", "")
    assert g.check(gamma, "1 1 1 5") == Verdict(False, "1 1 1 1", "balances, but the reaction of the table is another")
    assert g.check("balance gamma plus he3 gives h1 plus h2", "1 1 1 1").expected is None  # backwards: not the table's
    fission = "balance n1 plus u235 gives kr92 plus ba141 plus n1"
    assert g.check(fission, "1 1 1 1 3").ok and g.check(fission, "2 1 1 1 4").expected == "1 1 1 1 3"
    # not in the table, several balances: any smallest one is right, and it is the expected one
    two = "balance c 1 plus o 2 gives c 1 o 1 plus c 1 o 2"
    assert g.check(two, "4 3 2 2") == Verdict(True, "4 3 2 2", "balances; this species set has more than one balance")
    assert g.check(two, "3 2 2 1").ok
    assert g.check(two, "8 6 4 4") == Verdict(False, "4 3 2 2", "balances, but not with the smallest coefficients")
    # not in the table, with a gamma: nothing to judge, nothing confirmed
    v = g.check("balance e plus positron gives gamma", "1 1 2")
    assert v.ok is False and v.expected is None and "cannot be judged" in v.reason
    # the lepton number makes a decay a question with one answer, and a missing neutrino a wrong one
    assert g.check("balance n1 gives h1 plus e plus antineutrino", "1 1 1 1") == Verdict(True, "1 1 1 1", "")
    assert g.check("balance n1 gives h1 plus e plus antineutrino", "1 1 1 3").ok is False
    assert g.check("balance n1 gives h1 plus e", "1 1 1") == Verdict(False, None, "does not balance")
    assert g.check("balance n1 gives h1 plus e plus neutrino", "1 1 1 1") == Verdict(False, None, "does not balance")


def test_balance_questions_are_unique_and_single_digit(g, lines):
    asked = [ln for ln in lines if ln.meta["q"] == "balance"]
    assert len(asked) > 300
    for ln in asked:
        left, right = parse_equation(ln.prompt.removeprefix("balance "), with_coef=False)
        coefs = balance(left, right)
        assert coefs is not None and max(coefs) < 10
        assert ln.answer == " ".join(str(c) for c in coefs)
        assert len(ln.answer.split()) == len(left) + len(right)
        assert not any(s.free for s in left + right)
    assert g.by_name["helium_3_fusion"] in g.balanceable and g.by_name["carbon_14_decay"] in g.balanceable
    # every question runs in the table's direction: a balance question still says "gives"
    for ln in asked:
        r = g.by_name[ln.meta["name"]]
        left, right = parse_equation(ln.prompt.removeprefix("balance "), with_coef=False)
        assert sorted(s.key for s in left) == sorted(s.key for s in r.reactants)
        assert sorted(s.key for s in right) == sorted(s.key for s in r.products)
    assert g.by_name["octane_combustion"] not in g.balanceable
    assert g.by_name["helium_3_formation"] not in g.balanceable  # a gamma


def test_check(g, lines):
    assert g.check("check 1 x c 1 h 4 plus 1 x o 2 gives 1 x c 1 o 2 plus 2 x h 2 o 1 balanced", "no").ok
    assert g.check("check 1 x c 1 h 4 plus 2 x o 2 gives 1 x c 1 o 2 plus 2 x h 2 o 1 balanced", "yes").ok
    assert g.check("check 4 x h 2 plus 2 x o 2 gives 4 x h 2 o 1 balanced", "yes").ok  # scaled is still balanced
    assert g.check("check 2 x c 8 h 1 8 plus 2 5 x o 2 gives 1 6 x c 1 o 2 plus 1 8 x h 2 o 1 balanced", "yes").ok
    assert g.check("check 2 x c 8 h 1 8 plus 2 4 x o 2 gives 1 6 x c 1 o 2 plus 1 8 x h 2 o 1 balanced", "no").ok
    assert g.check("check 1 x u235 plus 1 x n1 gives 1 x ba141 plus 1 x kr92 plus 3 x n1 balanced", "yes").ok
    assert g.check("check 1 x u235 plus 1 x n1 gives 1 x ba141 plus 1 x kr92 plus 2 x n1 balanced", "no").ok
    assert g.check("check 1 x n1 gives 1 x h1 plus 1 x e plus 1 x antineutrino balanced", "yes").ok
    assert g.check("check 1 x n1 gives 1 x h1 plus 1 x antineutrino balanced", "no").ok  # charge is not kept
    assert g.check("check 1 x c14 gives 1 x n14 plus 1 x positron plus 1 x neutrino balanced", "no").ok
    assert g.check("check 4 x h 2 plus 2 x o 2 gives 4 x h 2 o 1 balanced", "no") == Verdict(False, "yes", "it is yes")
    # the lepton number is kept too: a decay without its neutrino, or with the wrong one, is not balanced
    for wrong in ("1 x n1 gives 1 x h1 plus 1 x e", "1 x c14 gives 1 x n14 plus 1 x e plus 1 x neutrino",
                  "2 x h1 gives 1 x h2 plus 1 x positron plus 1 x antineutrino", "1 x gamma gives 2 x neutrino",
                  "1 x k40 plus 1 x e gives 1 x ar40 plus 1 x antineutrino"):
        assert g.check(f"check {wrong} balanced", "no").ok, wrong
    assert g.check("check 1 x k40 plus 1 x e gives 1 x ar40 plus 1 x neutrino balanced", "yes").ok
    assert g.check("check 1 x e plus 1 x positron gives 2 x gamma balanced", "yes").ok
    asked = [ln for ln in lines if ln.meta["q"] == "check"]
    answers = Counter(ln.answer for ln in asked)
    assert set(answers) == {"yes", "no"} and abs(answers["yes"] - answers["no"]) <= YES_NO_SLACK
    shapes = Counter()
    for ln in asked:
        left, right = parse_equation(ln.prompt.removeprefix("check ").removesuffix(" balanced"), with_coef=True)
        assert (ln.answer == "yes") == is_balanced(left, right)
        assert len({s.key for s in left}) == len(left) and len({s.key for s in right}) == len(right), ln.prompt
        r = g.by_name[ln.meta["name"]]
        same = sorted(s.key for s in left) == sorted(s.key for s in r.reactants) and \
            sorted(s.key for s in right) == sorted(s.key for s in r.products)
        shapes["coefficients" if same else "species"] += 1
    # wrong equations differ from the table in a coefficient, or in a species
    assert shapes["species"] > 100 and shapes["coefficients"] > 1000


def test_check_questions_survive_a_row_of_massless_species():
    """A side made only of gammas has no coefficient to change; that must not stop ``generate``."""
    table = [R("electron_positron_annihilation", "fusion", "e + positron", "2 gamma", "exothermic", "nothing", "stars"),
             R("pair_of_gammas", "fusion", "gamma", "gamma", "endothermic", "nothing", "stars")]
    small = ReactionsGate(table)
    for seed in range(8):
        out = small.generate(random.Random(seed), 80)
        assert len(out) == 80 and {"check", "count_nuclear"} <= {ln.meta["q"] for ln in out}
        for ln in out:
            assert small.check(ln.prompt, ln.answer).ok, ln.text
            if ln.meta["q"] == "check":  # no species twice on a side, however the equation was changed
                sides = parse_equation(ln.prompt.removeprefix("check ").removesuffix(" balanced"), with_coef=True)
                assert all(len({s.key for s in side}) == len(side) for side in sides), ln.prompt
    assert table[0].q_mev == pytest.approx(1.022, rel=1e-3)


def test_count_atoms(g, lines):
    assert g.check("count_atoms 2 x h 2 o 1 hydrogen", "4") == Verdict(True, "4", "")
    assert g.check("count_atoms 2 x h 2 o 1 h", "4").ok and g.check("count_atoms 2 x h 2 o 1 oxygen", "2").ok
    assert g.check("count_atoms 1 x c 1 h 4 plus 2 x o 2 oxygen", "4").ok
    assert g.check("count_atoms 2 x h 2 o 1 carbon", "0").ok
    assert g.check("count_atoms 1 2 x c 6 h 1 2 o 6 hydrogen", "1 4 4").ok
    assert g.check("count_atoms 3 x ca 1 o 2 h 2 oxygen", num(6)).ok  # 3 Ca(OH)2
    v = g.check("count_atoms 1 2 x c 6 h 1 2 o 6 hydrogen", "1 4 5")
    assert not v.ok and v.expected == "1 4 4"
    assert not g.check("count_atoms 2 x h 2 o 1 hydrogen", "four").ok
    # a whole number is compared digit for digit: no other spelling of 4 is the answer
    for other in ("0 4", "4 point 0", "4 e 0", "04", "4 4"):
        assert g.check("count_atoms 2 x h 2 o 1 hydrogen", other) == Verdict(False, "4", "count is 4"), other
    # every element is known by its name, not only those of the table's substances
    assert g.check("count_atoms 1 x v 2 o 5 vanadium", "2").ok and g.check("count_atoms 3 x ga 2 o 3 gallium", "6").ok
    assert not g.owns("count_atoms 1 x v 2 o 5 spoonium")
    asked = [ln for ln in lines if ln.meta["q"] == "count_atoms"]
    for ln in asked:
        words = ln.prompt.split()
        el = {v: k for k, v in ELEMENT_NAMES.items()}[words[-1]]
        total = 0
        for term in " ".join(words[1:-1]).split(" plus "):
            coef, _, dense = term.partition(" x ")
            total += int(coef.replace(" ", "")) * fm.parse_dense(dense).get(el.capitalize(), 0)
        assert ln.answer == num(total), ln.text
    assert any(len(ln.prompt.split(" x ")[0].split()) == 3 for ln in asked)  # two-digit coefficients are there


def test_count_nuclear_and_nuclides(g, lines):
    assert g.check("count_nuclear 1 x u235 plus 1 x n1 mass_number", "2 3 6") == Verdict(True, "2 3 6", "")
    assert g.check("count_nuclear 1 x u235 plus 1 x n1 charge", "9 2").ok
    assert g.check("count_nuclear 1 x ba141 plus 1 x kr92 plus 3 x n1 mass_number", "2 3 6").ok
    assert g.check("count_nuclear 1 x h1 plus 1 x e plus 1 x antineutrino charge", "0").ok
    assert g.check("count_nuclear 1 x h1 plus 1 x e plus 1 x antineutrino lepton_number", "0").ok
    assert g.check("count_nuclear 2 x positron plus 2 x neutrino lepton_number", "0").ok
    assert g.check("count_nuclear 3 x e charge", "minus 3").ok and g.check("count_nuclear 3 x e lepton_number", "3").ok
    assert g.check("count_nuclear 1 x u235 plus 1 x n1 mass_number", "2 3 5") == Verdict(False, "2 3 6", "count is 2 3 6")
    asked = [ln for ln in lines if ln.meta["q"] == "count_nuclear"]
    assert len(asked) > 200 and {ln.prompt.split()[-1] for ln in asked} == set(SUMS)
    for ln in asked:
        words = ln.prompt.split()
        total = 0
        for term in " ".join(words[1:-1]).split(" plus "):
            coef, _, token = term.partition(" x ")
            total += int(coef.replace(" ", "")) * _nuclear_numbers(token)[list(SUMS).index(words[-1])]
        assert ln.answer == num(total), ln.text
    # every nuclide and particle of the table: its name and the three numbers a balance keeps
    assert g.check("nuclide u235 name", "uranium_235").ok and g.check("nuclide u235 mass_number", "2 3 5").ok
    assert g.check("nuclide u235 charge", "9 2").ok and g.check("nuclide u235 lepton_number", "0").ok
    assert g.check("nuclide e charge", "minus 1").ok and g.check("nuclide antineutrino lepton_number", "minus 1").ok
    assert g.check("nuclide n1 name", "neutron").ok and g.check("nuclide h1 name", "proton").ok
    assert g.check("nuclide sc45 charge", "2 1").ok  # by rule: not a nuclide of the table
    assert g.check("nuclide u235 charge", "9 3") == Verdict(False, "9 2", "it is 9 2")
    assert g.check("nuclide u235 charge", "9 2 point 0").ok is False
    facts = {ln.prompt: ln.answer for ln in g.all_facts() if ln.meta["q"] == "nuclide"}
    tokens_ = {s.text for r in TABLE for s in r.species if s.nuclear}
    assert set(facts) == {f"nuclide {t} {k}" for t in tokens_ for k in NUCLIDE_KEYS}
    for t in tokens_:
        a, z, lep = _nuclear_numbers(t)
        assert [facts[f"nuclide {t} {k}"] for k in ("mass_number", "charge", "lepton_number")] == [num(a), num(z), num(lep)]
    assert len({facts[f"nuclide {t} name"] for t in tokens_}) == len(tokens_)  # one name per token


def test_names_form(g, lines):
    assert g.check("calcium_oxide plus water gives", "calcium_hydroxide").ok
    assert g.check("water plus calcium_oxide gives", "calcium_hydroxide").ok
    assert g.check("deuterium plus tritium gives", "neutron helium_4").ok
    assert g.check("iron_oxide plus aluminum gives", "aluminum_oxide iron").ok
    # two reactions share these reactants: either product set is right
    assert g.check("methane plus oxygen gives", "carbon_dioxide water").ok
    assert g.check("methane plus oxygen gives", "carbon_monoxide water").ok
    assert not g.check("methane plus oxygen gives", "carbon_dioxide").ok
    assert not g.check("methane plus oxygen gives", "carbon_dioxide water water").ok  # each name once
    assert g.check("methane plus water gives", "carbon_monoxide hydrogen").ok
    assert g.check("gypsum plus water gives", "x") == Verdict(False, None, "no reaction with these reactants in the table")
    asked = {ln.prompt for ln in lines if ln.meta["q"] == "names"}
    assert "methane plus oxygen gives" not in asked and "carbon plus oxygen gives" not in asked
    assert "glucose gives" not in asked and "proton gives" not in asked
    for prompt in asked:
        assert len({frozenset(r.product_names) for r in TABLE if r.names_prompt == prompt}) == 1
        reactants = set(prompt.removesuffix(" gives").split(" plus "))
        same = {frozenset(r.product_names) for r in TABLE if not r.nuclear and set(r.reactant_names) == reactants}
        assert len(same) <= 1
    # a nuclear equation repeats a reactant by its coefficient: fusion must not read as decay
    assert g.check("proton plus proton gives", "deuterium positron neutrino").ok
    assert g.check("proton plus proton plus proton plus proton gives", "helium_4 positron neutrino").ok
    assert g.check("helium_4 plus helium_4 gives", "beryllium_8").ok
    assert g.check("helium_4 plus helium_4 plus helium_4 gives", "gamma carbon_12").ok
    assert not g.check("helium_4 plus helium_4 gives", "carbon_12 gamma").ok
    assert g.check("deuterium gives", "helium_3 neutron") == Verdict(False, None, "no reaction with these reactants in the table")
    assert g.check("neutron gives", "proton electron antineutrino").ok
    none = Verdict(False, None, "no reaction with these reactants in the table")
    assert g.check("neutron plus neutron gives", "proton electron antineutrino") == none  # two neutrons are no decay
    assert g.check("carbon_14 plus carbon_14 gives", "nitrogen_14 electron antineutrino") == none
    assert g.check("uranium_238 plus uranium_238 gives", "thorium_234 helium_4") == none
    every = {ln.prompt for ln in g.all_facts()}
    assert {"proton plus proton gives", "helium_4 plus helium_4 plus helium_4 gives", "carbon_14 gives",
            "potassium_40 gives", "potassium_40 plus electron gives", "proton plus neutron gives"} <= every
    assert g.check("methane plus methane plus water gives", "carbon_monoxide hydrogen").ok  # chemical: each name once
    # branches: both are right, so the question is not asked
    assert g.check("deuterium plus deuterium gives", "helium_3 neutron").ok
    assert g.check("deuterium plus deuterium gives", "tritium proton").ok
    assert g.check("carbon_12 plus carbon_12 gives", "sodium_23 proton").ok
    assert g.check("sodium plus oxygen gives", "sodium_oxide").ok and g.check("sodium plus oxygen gives", "sodium_peroxide").ok
    assert not every & {"deuterium plus deuterium gives", "carbon_12 plus carbon_12 gives", "sodium plus oxygen gives"}
    # a fission has many channels: its products are judged against the table, and never asked as the one answer
    assert not [p for p in every if p.endswith(" gives") and ("uranium_235" in p or "plutonium_239" in p)]
    assert g.check("plutonium_239 plus neutron gives", "xenon_134 zirconium_103 neutron").ok
    assert {r.name for r in names_asked(TABLE)} == {r.name for r in g.names_unique}
    assert all(r.kind != "fission" for r in g.names_unique)
    # the gas and the acid
    assert g.check("hydrogen plus chlorine gives", "hydrogen_chloride").ok
    assert not g.check("hydrogen plus chlorine gives", "hydrochloric_acid").ok
    assert g.check("ammonia plus hydrogen_chloride gives", "ammonium_chloride").ok
    assert g.check("zinc plus hydrochloric_acid gives", "zinc_chloride hydrogen").ok
    assert g.check("plaster_of_paris plus water gives", "gypsum").ok


def test_lists(g, lines):
    v = g.check("reactions kind fission", "uranium_235_fission plutonium_239_fission uranium_235_fission_xenon_channel")
    assert v.ok and v.expected == "plutonium_239_fission uranium_235_fission uranium_235_fission_xenon_channel"
    assert g.check("reactions where airbags", "sodium_azide_decomposition").ok
    assert g.check("reactions needs enzyme", "x").expected.split() == sorted(r.name for r in TABLE if "enzyme" in r.needs)
    assert not g.check("reactions kind fission", "uranium_235_fission").ok
    assert g.check("reactions product hydrogen_chloride", "hydrogen_chloride_synthesis").ok
    # none: a substance of the table that no reaction has in that role
    assert g.check("reactions reactant gypsum", "none") == Verdict(True, "none", "")
    assert g.check("reactions reactant gypsum", "plaster_setting") == Verdict(False, "none", "table lists none")
    assert g.check("reactions product gypsum", "plaster_setting").ok
    assert g.check("reactions product uranium_238", "none").ok and g.check("reactions reactant gamma", "none").ok
    # what the table does not hold, or cannot say in six names, is not a question
    for prompt in ("reactions where moon", "reactions kind spoon", "reactions reactant spoon", "reactions kind combustion",
                   "reactions energy exothermic", "reactions reactant oxygen"):
        assert g.check(prompt, "none") == Verdict(False, None, "not my question"), prompt
    table = rows()
    field = {"kind": "kind", "where": "where", "needs": "needs", "reactant": "reactant_names", "product": "product_names"}
    asked = [ln for ln in lines if ln.meta["q"] == "list"]
    for ln in asked:
        _, f, value = ln.prompt.split()
        assert f in FILTERS
        want = sorted(r["name"] for r in table if f not in r["untrained"]
                      and (value == r[field[f]] or (isinstance(r[field[f]], list) and value in r[field[f]])))
        assert ln.answer.split() == (want or ["none"]) and len(want) <= MAX_LIST, ln.text
        assert want or f in ("reactant", "product"), ln.text
    every = [ln for ln in g.all_facts() if ln.meta["q"] == "list"]
    none = [ln for ln in every if ln.answer == "none"]
    assert 50 <= len(none) < len(every) / 3 and tokens(max(every, key=lambda ln: tokens(ln.text)).text) <= 80
    for f in FILTERS:  # every value with a short enough list is asked, and nothing else is owned
        for value in g.values[f]:
            assert g.owns(f"reactions {f} {value}") == (len(g.index[f].get(value, ())) <= MAX_LIST)


def test_facts(g):
    assert g.check("methane_combustion products", "water carbon_dioxide").ok  # a set
    assert g.check("methane_combustion kind", "combustion").ok and not g.check("methane_combustion kind", "synthesis").ok
    assert g.check("photosynthesis needs", "light").ok and g.check("photosynthesis energy", "endothermic").ok
    assert g.check("haber_process needs", "heat pressure catalyst").ok and not g.check("haber_process needs", "heat").ok
    assert g.check("methane_combustion needs", "spark spark") == Verdict(False, "spark", "table says spark")
    eq = "methane_combustion equation"
    assert g.check(eq, "1 x c 1 h 4 plus 2 x o 2 gives 1 x c 1 o 2 plus 2 x h 2 o 1").ok
    assert g.check(eq, "2 x o 2 plus 1 x c 1 h 4 gives 2 x h 2 o plus 1 x c 1 o 2").ok  # order and a missing 1
    assert not g.check(eq, "1 x c 1 h 4 plus 1 x o 2 gives 1 x c 1 o 2 plus 2 x h 2 o 1").ok
    assert not g.check(eq, "2 x c 1 h 4 plus 4 x o 2 gives 2 x c 1 o 2 plus 4 x h 2 o 1").ok  # not the smallest
    assert not g.check(eq, "1 x c 1 o 2 plus 2 x h 2 o 1 gives 1 x c 1 h 4 plus 2 x o 2").ok  # backwards
    assert not g.check("helium_3_formation equation", "1 x h2 plus 1 x h1 gives 1 x he3 plus 1 x neutrino").ok
    assert g.check("urea_synthesis equation", "2 x n 1 h 3 plus 1 x c 1 o 2 gives 1 x c 1 o 1 n 2 h 4 plus 1 x h 2 o 1").ok
    # q_mev: within the tolerance of the rounded value, with its sign
    q = "deuterium_tritium_fusion q_mev"
    assert g.check(q, "1 7 point 5 9") == Verdict(True, "1 7 point 5 9", "")
    assert g.check(q, "1 7 point 6").ok and not g.check(q, "1 7").ok and not g.check(q, "minus 1 7 point 5 9").ok
    assert not g.check(q, "exothermic").ok and not g.check(q, "1 e 9 9 9 9").ok
    assert g.check("beryllium_8_formation q_mev", "minus 0 point 0 9 1 8 4").ok
    assert not g.check("beryllium_8_formation q_mev", "0 point 0 9 1 8 4").ok
    assert g.check("beryllium_8_formation energy", "endothermic").ok
    facts = g.all_facts()
    assert len({ln.text for ln in facts}) == len(facts)
    asked = {ln.prompt for ln in facts}
    for r in TABLE:
        for key in NUCLEAR_KEYS:
            assert (f"{r.name} {key}" in asked) == (key in r.keys and key not in untrained(r.name)), (r.name, key)
    for ln in facts:
        assert is_dense(ln.text) and tokens(ln.text) <= 80 and ln.kind == "fact"


def test_mix_and_capacity(g):
    first = Counter(ln.meta["q"] for ln in g.generate(random.Random(1), 1000))
    share = {k: v / 1000 for k, v in first.items()}
    # drawn 35 / 10 / 15 / 15 / 15 / 10 %; the small pools (names 149, lists 484, balance 603) thin out first
    assert 0.30 <= share["fact"] + share["nuclide"] <= 0.42 and 0.06 <= share["names"] <= 0.12
    assert 0.10 <= share["balance"] <= 0.18 and 0.12 <= share["check"] <= 0.20 and 0.07 <= share["list"] <= 0.13
    assert 0.12 <= share["count_atoms"] + share["count_nuclear"] <= 0.20
    assert g.sizes == {"fact": len([ln for ln in g.all_facts() if ln.meta["q"] in ("fact", "nuclide")]),
                       "names": len([ln for ln in g.all_facts() if ln.meta["q"] == "names"]),
                       "balance": g.sizes["balance"], "list": len(g.lists)}
    assert 550 <= g.sizes["balance"] <= 700
    many = g.generate(random.Random(2), 20000)
    assert len({ln.text for ln in many}) == 20000
    other = {ln.text for ln in g.generate(random.Random(3), 20000)}
    assert len(other - {ln.text for ln in many}) > 5000
    # every finite kind is given out in full, and a large n still comes back whole
    kinds = Counter(ln.meta["q"] for ln in many)
    assert kinds["fact"] + kinds["nuclide"] == g.sizes["fact"] and kinds["names"] == g.sizes["names"]
    assert kinds["list"] == g.sizes["list"] and kinds["balance"] >= g.sizes["balance"] - 20
    big = g.generate(random.Random(4), 70000)
    assert len(big) == 70000 == len({ln.text for ln in big}) and max(tokens(ln.text) for ln in big) <= 80
