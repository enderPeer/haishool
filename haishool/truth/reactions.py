"""Round 5: reactions, a closed topic whose truth is balance.

A reaction is a table row that code can check: atoms are neither made nor lost, so every
element count must be the same on both sides of a chemical equation, and a nuclear equation
must keep the mass number ``A``, the charge ``Z`` and the lepton number ``L``. The table
(``TABLE`` below, written to ``data/truth-v5/reactions.jsonl`` by
``python -m haishool.truth.reactions build``) holds 173 hand-curated reactions, 44 of them
nuclear; the gate verifies every one of them against its own balance rule, computes the energy
of every nuclear row from the masses of its two sides, and judges any balance the model
proposes by recomputing it.

Dense forms (alphabet ``[a-z0-9_ .]``, numbers digit by digit via :func:`haishool.truth.num`):

    species     ``<coefficient> x <dense formula>``: ``2 x h 2 o 1`` (two water molecules,
                :func:`haishool.truth.formula.dense`), or ``2 x h1`` for a nuclide. A nuclide
                token is ``<symbol><mass number>`` (``h1`` proton, ``h2`` deuterium, ``he4``,
                ``c12``, ``u235``); ``n1`` is a free neutron; the particles are ``e`` (electron),
                ``positron``, ``neutrino``, ``antineutrino`` and ``gamma``. Molecules are kept
                neutral: there are no ions and no charge field.
    equation    species joined by ``plus``, sides joined by ``gives``:
                ``1 x c 1 h 4 plus 2 x o 2 gives 1 x c 1 o 2 plus 2 x h 2 o 1``
    names       ``calcium_oxide plus water gives`` -> ``calcium_hydroxide``: one name per species;
                a nuclear equation repeats a reactant by its coefficient, so that fusion does
                not read as decay: ``proton plus proton gives`` -> ``deuterium positron neutrino``

Lines (``kind`` in brackets):

    methane_combustion reaction. kind combustion. reactants methane oxygen. products
        carbon_dioxide water. equation 1 x c 1 h 4 plus 2 x o 2 gives 1 x c 1 o 2 plus
        2 x h 2 o 1. energy exothermic. needs spark. where stoves heaters engines.   [record]
    deuterium_tritium_fusion reaction. kind fusion. reactants deuterium tritium. products
        helium_4 neutron. equation 1 x h2 plus 1 x h3 gives 1 x he4 plus 1 x n1. energy
        exothermic. q_mev 1 7 point 5 9. needs heat pressure. where fusion_reactors
        early_universe.                                                              [record]
    q methane_combustion products. a carbon_dioxide water.                           [fact]
    q methane_combustion equation. a 1 x c 1 h 4 plus 2 x o 2 gives ... .            [fact]
    q deuterium_tritium_fusion q_mev. a 1 7 point 5 9.                               [fact]
    q nuclide u235 name. a uranium_235.                                              [fact]
    q nuclide u235 charge. a 9 2.       (also ``mass_number`` and ``lepton_number``) [fact]
    q calcium_oxide plus water gives. a calcium_hydroxide.                           [fact]
    q balance c 1 h 4 plus o 2 gives c 1 o 2 plus h 2 o 1. a 1 2 1 2.               [calc]
    q check 1 x c 1 h 4 plus 1 x o 2 gives 1 x c 1 o 2 plus 2 x h 2 o 1 balanced. a no.  [yesno]
    q count_atoms 2 x h 2 o 1 hydrogen. a 4.                                         [calc]
    q count_atoms 1 x c 1 h 4 plus 2 x o 2 oxygen. a 4.                              [calc]
    q count_nuclear 1 x u235 plus 1 x n1 mass_number. a 2 3 6.                       [calc]
    q reactions kind fission. a plutonium_239_fission uranium_235_fission ... .      [fact]
    q reactions where airbags. a sodium_azide_decomposition.                         [fact]
    q reactions reactant gypsum. a none.                                             [fact]
        (also ``needs`` and ``product``; only lists of at most six)

A record line that would pass 80 tokens is split into ``<name> reaction.`` (kind, reactants,
products) and ``<name> reaction_more.`` (the other keys); the longest current row has 71
tokens, so none is split today.

Rules of the gate:

* chemical balance: every element count equal on both sides; nuclear balance: ``A``, ``Z`` and
  ``L`` equal on both sides (``e`` has ``A 0 Z -1 L 1``, ``positron`` ``A 0 Z 1 L -1``,
  ``neutrino`` ``L 1``, ``antineutrino`` ``L -1``, ``gamma`` nothing, ``n1`` ``A 1``; a nucleus
  has ``L 0``). The ``nuclide`` questions give every token of the table its name and its three
  numbers, and ``count_nuclear`` sums one of them over a side, so that a nuclear ``check`` can
  be worked out and not only remembered. Energy is not part of the balance.
* ``balance`` answers are the smallest whole coefficients, in the order of the species in the
  question. The gate recomputes them from the nullspace of the element (or ``A``/``Z``/``L``)
  matrix with exact integer arithmetic. A question is only asked when that nullspace is
  one-dimensional (a unique balance: every chemical row, and the nuclear rows without a
  ``gamma`` and without a species on both sides) and every coefficient is below 10, so that the
  digit-token answer stays unambiguous; equations with larger coefficients (octane, butane,
  candle wax) are still trained as ``equation`` facts and judged in ``check`` questions. A
  ``balance`` answer may also be given as a full equation with ``x`` coefficients. Where the
  balance is not unique, a reaction of the table is right only with the table's coefficients
  (one ``gamma``, not five); a species set the table does not hold is right with any smallest
  balance, and cannot be judged at all if it has a ``gamma``.
* ``check`` questions ask only whether the written coefficients balance; a scaled equation
  (``4 x h 2 plus 2 x o 2 gives 4 x h 2 o 1``) is balanced. The wrong equations of ``generate``
  have one coefficient changed, one species exchanged (a lepton for its opposite, any other for
  another species of the table) or one species left out.
* ``count_atoms`` sums coefficient times atom count over the listed (chemical) species for
  one element, named (``hydrogen``) or by symbol (``h``); ``count_nuclear`` does the same for
  ``mass_number``, ``charge`` or ``lepton_number`` over nuclides and particles. These are whole
  numbers and are compared exactly, digit for digit (the 0.5 % tolerance of the other topics
  has no rounded value to serve here); so are the numbers of the ``nuclide`` questions.
* ``q_mev`` is computed, not curated: the rest mass lost between the two sides times 931.494
  MeV/u, with the nuclear masses taken as the atomic masses of ``mendeleev`` less the electrons
  (their binding energy, some keV at most, is left out) and written with 4 significant digits.
  It is the energy of the equation as written: a ``gamma`` carries it away, the later
  annihilation of a positron is not in it. An answer is right within 0.5 %. The ``energy`` of a
  nuclear row is the sign of ``q_mev``; a row that says otherwise does not load.
* list answers (``reactants``, ``products``, ``needs``, ``where``, ``reactions ...``) are
  compared as sets, each word once; ``kind`` and ``energy`` exactly; an ``equation`` answer must
  carry the same coefficient per species on each side, species order free, a dense formula
  being read by its atom counts (``h 2 o`` equals ``h 2 o 1``).
* ``reactions <filter> <value>`` is a question only for a value the table holds (a kind, a
  place, a need, a substance name) and a list of at most six names. A substance that is never
  a reactant (or never a product) of the table has the answer ``none``.
* ``<names> gives`` is asked only when the reactants have one product set in the table
  (``carbon plus oxygen gives`` is not asked: carbon dioxide or carbon monoxide; neither is
  ``deuterium plus deuterium gives``, which has two branches) and never for a fission, which
  has hundreds of channels; the gate accepts any product set the table holds for those
  reactants. The answer is the ``products`` field, each name once.
* a species is written digit by digit: a coefficient or an atom count is one or more single
  digit tokens, at least 1, without a leading zero; a nuclide's mass number is at least its
  charge (``o2`` is no nuclide). Anything else is not a question of this gate.
* ``owns``: a reaction name followed by one of its record keys, ``balance`` or ``check ...
  balanced`` with a parseable equation, ``count_atoms``, ``count_nuclear``, ``nuclide <token>
  <key>``, ``reactions <filter> <value>`` as above, or ``<names> gives`` with known substance
  names. Three ``<name> <key>`` pairs are left out of the records, the questions and the lists:
  ``photosynthesis kind`` belongs to round 3, where photosynthesis is a concept whose kind is
  ``process`` (:data:`SHARED`), and the ``energy`` of ``chlorine_and_water`` and of
  ``marble_and_sulfuric_acid`` has no defensible sign (:data:`UNSURE`).

``generate`` draws question kinds in the proportions fact 35 % (reaction keys and nuclide
keys), names 10 %, balance 15 %, check 15 %, counts 15 % (``count_atoms`` for a chemical row,
``count_nuclear`` for a nuclear one), reactions lists 10 %, keeps only distinct lines and keeps
the ``yes`` and ``no`` answers of its check questions even. The pools differ in size (facts
1,464, names 149, lists 484, balance 603, check about 7,000; the count questions do not run
out, because one in six of them has two-digit coefficients), so the documented mix holds for
the first thousand lines or so; a kind whose lines are all out hands its share to the count
questions, and ``generate`` always returns ``n`` lines (60,000 take a few seconds; at that
size five lines in six are count questions). Every generated equation runs in the table's
direction, with the species of a side in any order. :meth:`ReactionsGate.all_facts` gives
every table question once, for a build that wants them all.

Names across topics: the dense tokens follow the elements table and the round 1-4 corpus
(``aluminum``, ``sulfur``, ``cesium``, ``jewelry``, ``fertilizer_plants``); only the reaction
names the round was given keep ``neutralisation`` and their formula fragments
(``neutralisation_h2so4_naoh``). A substance that the substances table also holds has the same
name and the same dense formula here (the tests compare the two tables). ``HCl`` has two
names: ``hydrogen_chloride`` for the gas, as in the substances table, and ``hydrochloric_acid``
for its solution in water, the acid of the acid rows (:data:`ALIASES`). Compounds are named as
compounds (``iron_oxide``, ``barium_sulfate``, ``carbon``), not as the minerals of the same
atoms (``hematite``, ``barite``, ``graphite``).

What is checked and what is curated: every equation is verified by the balance rule, the
nuclear energies are computed from masses that the tests compare with ``mendeleev`` and
``scipy.constants``, and the symbol table and element names are checked against ``mendeleev``
in the tests. ``kind``, ``needs`` and ``where``, and the ``energy`` of the chemical rows, are
curated labels (textbook knowledge, not computed); a row's ``notes`` flags the labels the
curator is less sure of (``sign from memory``, ``channel from memory``), so that a checker can
target them. Notes are for people and are not trained.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import permutations
from math import gcd
from pathlib import Path
from typing import Sequence

from haishool.truth import DIGITS, Line, Verdict, is_finite, num, parse_num
from haishool.truth import formula as fm

TOPIC = "reactions"
MAX_TOKENS = 80
MAX_LIST = 6
#: ``generate`` lets ``yes`` and ``no`` among its check questions differ by at most this many
YES_NO_SLACK = 20
#: reaction name -> record keys left to an earlier round, where the same name is a record with
#: another answer (round 3: ``photosynthesis. type concept. kind process.``); such a key is
#: left out of the record line, is neither asked nor owned here and feeds no list. The tests
#: check this against the round 1-3 records.
SHARED: dict[str, tuple[str, ...]] = {"photosynthesis": ("kind",)}
#: reaction name -> record keys whose table value cannot be defended and is therefore not
#: trained, asked or owned (the equation of the row still is). Both rows are nearly
#: thermoneutral: the sign of the heat depends on the state of the chlorine (about -2 kJ/mol
#: from the gas, about +21 kJ/mol from dissolved chlorine) and on the acid strength and the
#: form of the calcium sulfate (about +3 kJ/mol as written, about -14 kJ/mol if gypsum forms).
UNSURE: dict[str, tuple[str, ...]] = {"chlorine_and_water": ("energy",), "marble_and_sulfuric_acid": ("energy",)}

#: key -> meaning (and unit where there is one); the question forms are keys too
KEYS: dict[str, str] = {
    "kind": "reaction kind: combustion, synthesis, decomposition, single_replacement, double_replacement, "
            "acid_base, redox, precipitation, corrosion, electrolysis, photosynthesis, respiration, "
            "fermentation, fusion, fission, beta_decay, alpha_decay, electron_capture, neutron_capture. "
            "Where two kinds fit, the one that says more wins: precipitation (a solid falls out of a "
            "solution) and acid_base (an acid or an acid oxide meets a base or a carbonate) before "
            "double_replacement and synthesis, combustion and corrosion before redox; synthesis is for "
            "reactants that join into one product (urea_synthesis also sets water free)",
    "reactants": "substance names of the reactants, one per species, table order",
    "products": "substance names of the products, one per species, table order",
    "equation": "balanced equation with the smallest whole coefficients: <coefficient> x <dense formula or "
                "nuclide> joined by plus, sides joined by gives",
    "energy": "exothermic or endothermic; for a nuclear reaction the sign of q_mev",
    "q_mev": "nuclear reactions only: energy set free by the equation as written, from the rest masses of "
             "both sides, MeV (negative: energy has to be put in)",
    "needs": "what starts or drives it: spark (any small flame or spark, then it burns on by itself), heat "
             "(it has to be heated strongly or kept hot), light, catalyst, enzyme, electricity, pressure, nothing",
    "where": "where it happens: stoves, engines, cells, leaves, stars, reactors, kitchens, labs, ...",
    "gives": "<reactant names joined by plus> gives -> the product names",
    "balance": "balance <species without coefficients, plus and gives> -> smallest whole coefficients in "
               "species order, every coefficient a single digit",
    "check": "check <equation with coefficients> balanced -> yes or no",
    "count_atoms": "count_atoms <species with coefficients joined by plus> <element name or symbol> -> "
                   "number of atoms of that element, count",
    "count_nuclear": "count_nuclear <nuclides and particles with coefficients joined by plus> "
                     "<mass_number|charge|lepton_number> -> the sum over the listed species, count",
    "nuclide": "nuclide <nuclide or particle token> <name|mass_number|charge|lepton_number> -> its name, its "
               "nucleons (A), its charge in proton charges (Z), its lepton number",
    "reactions": "reactions <kind|where|needs|reactant|product> <value> -> reaction names, at most 6; none "
                 "when a substance of the table is never a reactant (or never a product)",
}
RECORD_KEYS = ("kind", "reactants", "products", "equation", "energy", "needs", "where")
#: a nuclear record carries its energy in MeV after ``energy``
NUCLEAR_KEYS = ("kind", "reactants", "products", "equation", "energy", "q_mev", "needs", "where")
FILTERS = ("kind", "where", "needs", "reactant", "product")
KINDS = ("combustion", "synthesis", "decomposition", "single_replacement", "double_replacement", "acid_base",
         "redox", "precipitation", "corrosion", "electrolysis", "photosynthesis", "respiration", "fermentation",
         "fusion", "fission", "beta_decay", "alpha_decay", "electron_capture", "neutron_capture")
NEEDS = ("spark", "heat", "light", "catalyst", "enzyme", "electricity", "pressure", "nothing")
NUCLIDE_KEYS = ("name", "mass_number", "charge", "lepton_number")
#: ``count_nuclear`` and ``nuclide`` key -> the conserved quantity it reads
SUMS: dict[str, str] = {"mass_number": "a", "charge": "z", "lepton_number": "l"}
#: relative tolerance for ``q_mev`` answers, as the other topics judge rounded numbers
TOLERANCE = 0.005

#: the elements in order of atomic number (Z = index + 1); the tests check this against mendeleev
SYMBOLS = ("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
           "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba "
           "La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn "
           "Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr").split()
Z: dict[str, int] = {s.lower(): i + 1 for i, s in enumerate(SYMBOLS)}
#: lowercase symbol -> element name, for ``count_atoms`` and nuclide names; spelt as the elements
#: table and the round 1-4 corpus spell them (``aluminum``, ``sulfur``, ``cesium``)
ELEMENT_NAMES: dict[str, str] = {
    "h": "hydrogen", "he": "helium", "li": "lithium", "be": "beryllium", "b": "boron", "c": "carbon",
    "n": "nitrogen", "o": "oxygen", "f": "fluorine", "ne": "neon", "na": "sodium", "mg": "magnesium",
    "al": "aluminum", "si": "silicon", "p": "phosphorus", "s": "sulfur", "cl": "chlorine", "ar": "argon",
    "k": "potassium", "ca": "calcium", "sc": "scandium", "ti": "titanium", "v": "vanadium", "cr": "chromium",
    "mn": "manganese", "fe": "iron", "co": "cobalt", "ni": "nickel", "cu": "copper", "zn": "zinc",
    "ga": "gallium", "ge": "germanium", "as": "arsenic", "se": "selenium", "br": "bromine", "kr": "krypton",
    "rb": "rubidium", "sr": "strontium", "y": "yttrium", "zr": "zirconium", "nb": "niobium",
    "mo": "molybdenum", "tc": "technetium", "ru": "ruthenium", "rh": "rhodium", "pd": "palladium",
    "ag": "silver", "cd": "cadmium", "in": "indium", "sn": "tin", "sb": "antimony", "te": "tellurium",
    "i": "iodine", "xe": "xenon", "cs": "cesium", "ba": "barium", "la": "lanthanum", "ce": "cerium",
    "pr": "praseodymium", "nd": "neodymium", "pm": "promethium", "sm": "samarium", "eu": "europium",
    "gd": "gadolinium", "tb": "terbium", "dy": "dysprosium", "ho": "holmium", "er": "erbium",
    "tm": "thulium", "yb": "ytterbium", "lu": "lutetium", "hf": "hafnium", "ta": "tantalum", "w": "tungsten",
    "re": "rhenium", "os": "osmium", "ir": "iridium", "pt": "platinum", "au": "gold", "hg": "mercury",
    "tl": "thallium", "pb": "lead", "bi": "bismuth", "po": "polonium", "at": "astatine", "rn": "radon",
    "fr": "francium", "ra": "radium", "ac": "actinium", "th": "thorium", "pa": "protactinium",
    "u": "uranium", "np": "neptunium", "pu": "plutonium", "am": "americium", "cm": "curium",
    "bk": "berkelium", "cf": "californium", "es": "einsteinium", "fm": "fermium", "md": "mendelevium",
    "no": "nobelium", "lr": "lawrencium",
}
_NAME_TO_SYMBOL = {name: sym for sym, name in ELEMENT_NAMES.items()}

#: particle token -> (A, Z); ``n1`` is the free neutron, not nitrogen-1
PARTICLES: dict[str, tuple[int, int]] = {
    "n1": (1, 0), "e": (0, -1), "positron": (0, 1), "neutrino": (0, 0), "antineutrino": (0, 0), "gamma": (0, 0),
}
#: particle token -> lepton number; every nucleus, the neutron and the gamma have 0
LEPTONS: dict[str, int] = {"e": 1, "positron": -1, "neutrino": 1, "antineutrino": -1}
#: lepton -> the particle a wrong equation is most likely to hold in its place
OPPOSITE: dict[str, str] = {"e": "positron", "positron": "e", "neutrino": "antineutrino", "antineutrino": "neutrino"}
PARTICLE_NAMES = {"n1": "neutron", "e": "electron", "positron": "positron", "neutrino": "neutrino",
                  "antineutrino": "antineutrino", "gamma": "gamma"}
NUCLIDE_NAMES = {"h1": "proton", "h2": "deuterium", "h3": "tritium"}
_NUCLIDE = re.compile(r"^([a-z]{1,2})([1-9][0-9]{0,2})$")

#: rest masses in u (CODATA, as ``scipy.constants`` holds them) and the MeV in one u; the tests
#: check the three against scipy
NEUTRON_MASS_U = 1.00866491606
ELECTRON_MASS_U = 5.485799090441e-4
U_MEV = 931.49410372
#: nuclide token -> mass of the neutral atom in u, copied from the isotope table of ``mendeleev``
#: (the atomic mass evaluation) and checked against it in the tests; every nuclide of the table
ATOM_MASS_U: dict[str, float] = {
    "h1": 1.007825031898, "h2": 2.014101777844, "h3": 3.01604928132, "he3": 3.01602932197,
    "he4": 4.00260325413, "li7": 7.01600343426, "be7": 7.016928714, "be8": 8.005305102, "c12": 12.0,
    "c13": 13.00335483534, "c14": 14.00324198862, "n13": 13.005738609, "n14": 14.00307400425,
    "n15": 15.00010889827, "o15": 15.003065636, "o16": 15.99491461926, "o18": 17.99915961214,
    "f18": 18.000937324, "ne20": 19.99244017525, "na23": 22.98976928195, "mg24": 23.985041689,
    "k40": 39.963998165, "ar40": 39.96238312204, "ca40": 39.96259085, "kr92": 91.926173092,
    "sr94": 93.915355641, "zr103": 102.927204054, "i131": 130.906126375, "xe131": 130.90508412808,
    "xe134": 133.90539303, "xe140": 139.921645814, "cs137": 136.907089296, "ba137": 136.905827207,
    "ba141": 140.914403653, "po218": 218.008971234, "rn222": 222.017576017, "ra226": 226.025408186,
    "th234": 234.043599801, "u234": 234.040950296, "u235": 235.043928117, "u238": 238.050786936,
    "u239": 239.054291989, "np237": 237.04817164, "np239": 239.052937538, "pu238": 238.049558175,
    "pu239": 239.052161596, "am241": 241.056827343,
}

#: chemical formula (standard case, for :func:`haishool.truth.formula.parse`) -> substance name,
#: spelt and written as the substances table has them where it has them (``CH4N2O`` urea,
#: ``C2H3NaO2`` sodium acetate, ``NaClO``). ``C`` is ``carbon`` whatever its form, sulfur is the ``S8`` ring as in the
#: substances table, ``(CaSO4)2H2O`` is the half hydrate (plaster of paris) doubled to whole atoms.
SUBSTANCES: dict[str, str] = {
    "CH4": "methane", "O2": "oxygen", "CO2": "carbon_dioxide", "H2O": "water", "CO": "carbon_monoxide",
    "H2": "hydrogen", "C3H8": "propane", "C4H10": "butane", "C8H18": "octane", "C12H26": "dodecane",
    "C2H5OH": "ethanol", "CH3OH": "methanol", "C2H2": "acetylene", "C2H4": "ethylene", "C": "carbon",
    "S8": "sulfur", "SO2": "sulfur_dioxide", "SO3": "sulfur_trioxide", "H2S": "hydrogen_sulfide",
    "Mg": "magnesium", "MgO": "magnesium_oxide", "Al": "aluminum", "Al2O3": "aluminum_oxide", "Fe": "iron",
    "Fe2O3": "iron_oxide", "Fe3O4": "magnetite", "P4": "white_phosphorus", "P4O10": "phosphorus_pentoxide",
    "Na": "sodium", "Na2O": "sodium_oxide", "NH3": "ammonia", "N2": "nitrogen", "N2H4": "hydrazine",
    "C25H52": "paraffin_wax", "C6H10O5": "cellulose", "C12H22O11": "sucrose", "C6H12O6": "glucose",
    "HCl": "hydrochloric_acid", "NH4Cl": "ammonium_chloride", "H2SO4": "sulfuric_acid", "NO": "nitric_oxide",
    "NO2": "nitrogen_dioxide", "N2O4": "dinitrogen_tetroxide", "O3": "ozone", "NaCl": "sodium_chloride",
    "Cl2": "chlorine", "FeS": "iron_sulfide", "CaO": "calcium_oxide", "Ca(OH)2": "calcium_hydroxide",
    "H2CO3": "carbonic_acid", "H2SO3": "sulfurous_acid", "H3PO4": "phosphoric_acid", "CH4N2O": "urea",
    "CaSO4": "calcium_sulfate", "(CaSO4)2H2O": "plaster_of_paris",
    "CaSO4(H2O)2": "gypsum", "CaCO3": "calcium_carbonate", "Ca(HCO3)2": "calcium_bicarbonate",
    "H2O2": "hydrogen_peroxide", "NaHCO3": "sodium_bicarbonate", "Na2CO3": "sodium_carbonate",
    "KClO3": "potassium_chlorate", "KCl": "potassium_chloride", "NH4NO3": "ammonium_nitrate",
    "N2O": "nitrous_oxide", "HgO": "mercury_oxide", "Hg": "mercury", "CuCO3": "copper_carbonate",
    "CuO": "copper_oxide", "NaN3": "sodium_azide", "NaOH": "sodium_hydroxide", "K": "potassium",
    "KOH": "potassium_hydroxide", "Li": "lithium", "LiOH": "lithium_hydroxide", "Ca": "calcium",
    "ZnCl2": "zinc_chloride", "Zn": "zinc", "MgCl2": "magnesium_chloride", "AlCl3": "aluminum_chloride",
    "ZnSO4": "zinc_sulfate", "CuSO4": "copper_sulfate", "FeSO4": "iron_sulfate", "Cu": "copper",
    "AgNO3": "silver_nitrate", "Cu(NO3)2": "copper_nitrate", "Ag": "silver", "KBr": "potassium_bromide",
    "Br2": "bromine", "NaI": "sodium_iodide", "I2": "iodine", "ZnO": "zinc_oxide", "Si": "silicon",
    "SiO2": "silica", "HNO3": "nitric_acid", "HOCl": "hypochlorous_acid", "Pb": "lead",
    "PbO2": "lead_dioxide", "PbSO4": "lead_sulfate", "Na2O2": "sodium_peroxide", "KO2": "potassium_superoxide",
    "K2CO3": "potassium_carbonate", "Fe(OH)3": "iron_hydroxide", "Cu2CO3(OH)2": "malachite",
    "Ag2S": "silver_sulfide", "Mg(OH)2": "magnesium_hydroxide", "Al(OH)3": "aluminum_hydroxide",
    "CaCl2": "calcium_chloride", "CH3COOH": "acetic_acid", "C2H3NaO2": "sodium_acetate",
    "KNO3": "potassium_nitrate", "Na2SO4": "sodium_sulfate", "(NH4)2SO4": "ammonium_sulfate",
    "Li2CO3": "lithium_carbonate", "AgCl": "silver_chloride", "NaNO3": "sodium_nitrate",
    "BaCl2": "barium_chloride", "BaSO4": "barium_sulfate", "Pb(NO3)2": "lead_nitrate", "KI": "potassium_iodide",
    "PbI2": "lead_iodide", "Cu(OH)2": "copper_hydroxide", "FeCl3": "iron_chloride", "CaC2": "calcium_carbide",
    "C3H6O3": "lactic_acid", "NaClO": "sodium_hypochlorite",
}
#: table spelling -> (formula, name): a second name for a formula of :data:`SUBSTANCES`.
#: ``HCl`` is ``hydrochloric_acid`` in the rows where it is the acid in water and
#: ``hydrogen_chloride``, as the substances table names it, in the two rows where it is the gas.
ALIASES: dict[str, tuple[str, str]] = {"HCl_gas": ("HCl", "hydrogen_chloride")}


def nuclide(token: str) -> tuple[int, int] | None:
    """``he4`` -> ``(4, 2)``; a particle by its table; ``None`` if ``token`` is neither (a mass
    number below the charge, as in ``o2``, is no nuclide)."""
    if token in PARTICLES:
        return PARTICLES[token]
    m = _NUCLIDE.match(token)
    if m and m.group(1) in Z and int(m.group(2)) >= Z[m.group(1)]:
        return int(m.group(2)), Z[m.group(1)]
    return None


def nuclide_name(token: str) -> str:
    """``u235`` -> ``uranium_235``, ``h1`` -> ``proton``, ``n1`` -> ``neutron``."""
    if token in PARTICLE_NAMES:
        return PARTICLE_NAMES[token]
    if token in NUCLIDE_NAMES:
        return NUCLIDE_NAMES[token]
    if nuclide(token) is None:
        raise ValueError(f"not a nuclide: {token!r}")
    m = _NUCLIDE.match(token)
    return f"{ELEMENT_NAMES[m.group(1)]}_{m.group(2)}"


def mass_u(token: str) -> float:
    """Rest mass in u of a particle or of a bare nucleus (the atom's mass less its electrons;
    their binding energy, some keV at most, is left out). Only for the nuclides of the table."""
    if token == "n1":
        return NEUTRON_MASS_U
    if token in PARTICLES:
        return ELECTRON_MASS_U if token in ("e", "positron") else 0.0
    return ATOM_MASS_U[token] - nuclide(token)[1] * ELECTRON_MASS_U


@dataclass(frozen=True)
class Species:
    """One term of an equation: ``2 x h 2 o 1`` or ``2 x h1``.

    ``counts`` is element -> atoms for a chemical species and ``{"a": A, "z": Z, "l": L}``
    (mass number, charge, lepton number) for a nuclear one; ``key`` identifies the substance
    (atom counts, or the nuclide token) for comparisons. ``alias`` is the name of a species
    written with one of the :data:`ALIASES`.
    """
    coef: int
    text: str
    counts: dict[str, int]
    nuclear: bool
    formula: str = ""
    alias: str = ""

    @property
    def key(self) -> tuple:
        return (self.text,) if self.nuclear else tuple(sorted(self.counts.items()))

    @property
    def name(self) -> str:
        if self.nuclear:
            return nuclide_name(self.text)
        if self.alias:
            return self.alias
        if self.formula:
            return SUBSTANCES[self.formula]
        for formula, name in SUBSTANCES.items():  # a species read from a line: found by its atoms
            if fm.same(fm.parse(formula), {el.capitalize(): n for el, n in self.counts.items()}):
                return name
        raise KeyError(f"no substance name for {self.text!r}")

    @property
    def free(self) -> bool:
        """True for a species the balance cannot count: ``gamma`` has no mass number, charge or
        lepton number."""
        return not any(self.counts.values())

    def times(self, coef: int) -> Species:
        return replace(self, coef=coef)

    def dense(self) -> str:
        return f"{num(self.coef)} x {self.text}"


def species(formula: str, coef: int = 1) -> Species:
    """A species from the table's spelling: ``H2O`` (standard case) or ``he4`` (nuclide token)."""
    az = nuclide(formula)
    if az is not None:
        return Species(coef, formula, {"a": az[0], "z": az[1], "l": LEPTONS.get(formula, 0)}, True)
    formula, alias = ALIASES.get(formula, (formula, ""))
    counts = fm.parse(formula)
    return Species(coef, fm.dense(formula), {el.lower(): n for el, n in counts.items()}, False, formula, alias)


def _whole(digits: list[str]) -> int | None:
    """A whole number of one or more digit tokens, at least 1, without a leading zero."""
    if not digits or digits[0] == "0" or any(d not in DIGITS for d in digits):
        return None
    return int("".join(digits))


def _atoms(words: list[str]) -> dict[str, int] | None:
    """Atom counts of a dense formula given as tokens (``c 1 h 4``; a missing count is 1). Every
    token is an element symbol or a single digit, and a count is at least 1."""
    counts: dict[str, int] = {}
    i = 0
    while i < len(words):
        el = words[i]
        if el not in Z:
            return None
        j = i + 1
        while j < len(words) and words[j] in DIGITS:
            j += 1
        n = _whole(words[i + 1:j]) if j > i + 1 else 1
        if n is None:
            return None
        counts[el] = counts.get(el, 0) + n
        i = j
    return counts or None


def parse_species(text: str, with_coef: bool) -> Species | None:
    """``2 x c 1 h 4`` (``with_coef``) or ``c 1 h 4`` -> :class:`Species`; ``None`` when malformed
    (a coefficient or count that is zero, has a leading zero or is not written digit by digit)."""
    words = text.split()
    coef = 1
    if with_coef:
        if "x" not in words:
            return None
        i = words.index("x")
        coef = _whole(words[:i])
        words = words[i + 1:]
        if coef is None:
            return None
    elif "x" in words:
        return None
    if not words:
        return None
    if len(words) == 1:
        az = nuclide(words[0])
        if az is not None:
            return Species(coef, words[0], {"a": az[0], "z": az[1], "l": LEPTONS.get(words[0], 0)}, True)
    counts = _atoms(words)
    if counts is None:
        return None
    return Species(coef, " ".join(f"{el} {num(n)}" for el, n in counts.items()), counts, False)


def parse_equation(text: str, with_coef: bool) -> tuple[list[Species], list[Species]] | None:
    """Both sides of ``... plus ... gives ... plus ...``; ``None`` if malformed or mixed."""
    if " gives " not in text:
        return None
    left, _, right = text.partition(" gives ")
    sides = []
    for side in (left, right):
        out = [parse_species(s, with_coef) for s in side.split(" plus ")]
        if not out or any(s is None for s in out):
            return None
        sides.append(out)
    kinds = {s.nuclear for side in sides for s in side}
    if len(kinds) != 1:
        return None
    return sides[0], sides[1]


def equation_text(left: Sequence[Species], right: Sequence[Species]) -> str:
    """``1 x c 1 h 4 plus 2 x o 2 gives 1 x c 1 o 2 plus 2 x h 2 o 1`` from two lists of species."""
    return " plus ".join(s.dense() for s in left) + " gives " + " plus ".join(s.dense() for s in right)


def totals(side: list[Species]) -> dict[str, int]:
    out: dict[str, int] = {}
    for s in side:
        for k, n in s.counts.items():
            out[k] = out.get(k, 0) + s.coef * n
    return {k: v for k, v in out.items() if v}


def is_balanced(left: list[Species], right: list[Species]) -> bool:
    return totals(left) == totals(right)


def _nullspace(rows: list[list[int]], ncols: int) -> list[list[Fraction]]:
    """Basis of the nullspace of an integer matrix, exact (reduced row echelon form)."""
    m = [[Fraction(x) for x in r] for r in rows]
    pivots: list[int] = []
    r = 0
    for c in range(ncols):
        p = next((i for i in range(r, len(m)) if m[i][c] != 0), None)
        if p is None:
            continue
        m[r], m[p] = m[p], m[r]
        pv = m[r][c]
        m[r] = [x / pv for x in m[r]]
        for i in range(len(m)):
            if i != r and m[i][c] != 0:
                f = m[i][c]
                m[i] = [a - f * b for a, b in zip(m[i], m[r])]
        pivots.append(c)
        r += 1
        if r == len(m):
            break
    basis = []
    for fc in [c for c in range(ncols) if c not in pivots]:
        v = [Fraction(0)] * ncols
        v[fc] = Fraction(1)
        for i, pc in enumerate(pivots):
            v[pc] = -m[i][fc]
        basis.append(v)
    return basis


def balance(left: list[Species], right: list[Species]) -> list[int] | None:
    """The smallest whole coefficients that balance ``left gives right`` (species order kept),
    or ``None`` when there is no unique positive balance (a free species such as ``gamma``, or a
    species set that balances in more than one way)."""
    keys = sorted({k for s in left + right for k in s.counts})
    cols = [s.counts for s in left] + [s.counts for s in right]
    signs = [1] * len(left) + [-1] * len(right)
    rows = [[sg * c.get(k, 0) for sg, c in zip(signs, cols)] for k in keys]
    basis = _nullspace(rows, len(cols))
    if len(basis) != 1:
        return None
    v = basis[0]
    lcm = 1
    for x in v:
        lcm = lcm * x.denominator // gcd(lcm, x.denominator)
    ints = [int(x * lcm) for x in v]
    if all(x <= 0 for x in ints):
        ints = [-x for x in ints]
    if any(x <= 0 for x in ints):
        return None
    g = 0
    for x in ints:
        g = gcd(g, x)
    return [x // g for x in ints]


def minimal(coefs: list[int]) -> list[int]:
    g = 0
    for x in coefs:
        g = gcd(g, x)
    return [x // g for x in coefs] if g else list(coefs)


@dataclass(frozen=True)
class Reaction:
    name: str
    kind: str
    reactants: tuple[Species, ...]
    products: tuple[Species, ...]
    energy: str
    needs: tuple[str, ...]
    where: tuple[str, ...]
    notes: str = ""

    @property
    def nuclear(self) -> bool:
        return self.reactants[0].nuclear

    @property
    def keys(self) -> tuple[str, ...]:
        """The record keys of this row, in record order."""
        return NUCLEAR_KEYS if self.nuclear else RECORD_KEYS

    @property
    def q_mev(self) -> float | None:
        """Energy set free by a nuclear equation as written, in MeV: the rest mass that is lost."""
        if not self.nuclear:
            return None
        before, after = (sum(s.coef * mass_u(s.text) for s in side) for side in (self.reactants, self.products))
        return (before - after) * U_MEV

    @property
    def species(self) -> tuple[Species, ...]:
        return self.reactants + self.products

    @property
    def equation(self) -> str:
        return equation_text(self.reactants, self.products)

    @property
    def reactant_names(self) -> list[str]:
        return list(dict.fromkeys(s.name for s in self.reactants))

    @property
    def product_names(self) -> list[str]:
        return list(dict.fromkeys(s.name for s in self.products))

    @property
    def names_key(self) -> tuple[str, ...]:
        """The reactant names of the ``gives`` form, sorted: one per species, but a nuclear
        equation repeats a name by its coefficient (two protons are ``proton plus proton``)."""
        if self.nuclear:
            return tuple(sorted(s.name for s in self.reactants for _ in range(s.coef)))
        return tuple(sorted(self.reactant_names))

    @property
    def names_prompt(self) -> str:
        names = [s.name for s in self.reactants for _ in range(s.coef)] if self.nuclear else self.reactant_names
        return " plus ".join(names) + " gives"

    def value(self, key: str) -> str:
        """The dense answer text of one record key."""
        if key == "kind":
            return self.kind
        if key == "reactants":
            return " ".join(self.reactant_names)
        if key == "products":
            return " ".join(self.product_names)
        if key == "equation":
            return self.equation
        if key == "energy":
            return self.energy
        if key == "q_mev" and self.nuclear:
            return num(self.q_mev, sig=4)
        if key == "needs":
            return " ".join(self.needs)
        if key == "where":
            return " ".join(self.where)
        raise KeyError(key)

    def record_lines(self) -> list[Line]:
        """The record line, without the keys this row leaves untrained; two lines when one would
        pass :data:`MAX_TOKENS` (a row too long even for that is a table error)."""
        keys = [k for k in self.keys if k not in untrained(self.name)]
        head = [f"{k} {self.value(k)}." for k in keys if k in RECORD_KEYS[:3]]
        tail = [f"{k} {self.value(k)}." for k in keys if k not in RECORD_KEYS[:3]]
        texts = [f"{self.name} reaction. " + " ".join(head + tail)]
        if count_tokens(texts[0]) > MAX_TOKENS:
            texts = [f"{self.name} reaction. " + " ".join(head), f"{self.name} reaction_more. " + " ".join(tail)]
        if any(count_tokens(t) > MAX_TOKENS for t in texts):
            raise ValueError(f"{self.name}: a record line passes {MAX_TOKENS} tokens")
        return [Line(t, topic=TOPIC, kind="record", meta={"name": self.name}) for t in texts]


def untrained(name: str) -> tuple[str, ...]:
    """The record keys of reaction ``name`` that are neither trained nor asked nor owned."""
    return SHARED.get(name, ()) + UNSURE.get(name, ())


def count_tokens(line: str) -> int:
    """Tokens as :func:`haishool.student.tokens` counts them (``.`` is a token)."""
    return len(line.replace(".", " . ").split())


def _side(text: str) -> tuple[Species, ...]:
    """``CH4 + 2 O2`` -> species; a leading integer is the coefficient."""
    out = []
    for term in text.split(" + "):
        words = term.split()
        coef, formula = (int(words[0]), words[1]) if len(words) == 2 else (1, words[0])
        out.append(species(formula, coef))
    return tuple(out)


def R(name: str, kind: str, reactants: str, products: str, energy: str, needs: str, where: str,
      notes: str = "") -> Reaction:
    """One table row; ``needs`` and ``where`` are space-separated lists. The ``energy`` of a
    nuclear row is not taken on trust: it must be the sign of the energy its masses give."""
    row = Reaction(name, kind, _side(reactants), _side(products), energy, tuple(needs.split()),
                   tuple(where.split()), notes)
    if row.nuclear and energy != ("exothermic" if row.q_mev > 0 else "endothermic"):
        raise ValueError(f"{name}: the masses give {row.q_mev:.4g} MeV, the row says {energy}")
    return row


#: the table: name, kind, reactants, products, energy, needs, where, notes
TABLE: list[Reaction] = [
    # combustion
    R("methane_combustion", "combustion", "CH4 + 2 O2", "CO2 + 2 H2O", "exothermic", "spark",
      "stoves heaters engines", "natural gas burning"),
    R("methane_incomplete_combustion", "combustion", "2 CH4 + 3 O2", "2 CO + 4 H2O", "exothermic", "spark",
      "faulty_heaters", "too little oxygen gives poisonous carbon monoxide"),
    R("hydrogen_combustion", "combustion", "2 H2 + O2", "2 H2O", "exothermic", "spark", "rockets"),
    R("propane_combustion", "combustion", "C3H8 + 5 O2", "3 CO2 + 4 H2O", "exothermic", "spark", "grills heaters"),
    R("butane_combustion", "combustion", "2 C4H10 + 13 O2", "8 CO2 + 10 H2O", "exothermic", "spark",
      "lighters camping_stoves"),
    R("octane_combustion", "combustion", "2 C8H18 + 25 O2", "16 CO2 + 18 H2O", "exothermic", "spark",
      "engines cars", "petrol stands in for octane"),
    R("dodecane_combustion", "combustion", "2 C12H26 + 37 O2", "24 CO2 + 26 H2O", "exothermic", "heat pressure",
      "diesel_engines trucks", "diesel stands in for dodecane; compression ignites it"),
    R("ethanol_combustion", "combustion", "C2H5OH + 3 O2", "2 CO2 + 3 H2O", "exothermic", "spark", "burners engines"),
    R("methanol_combustion", "combustion", "2 CH3OH + 3 O2", "2 CO2 + 4 H2O", "exothermic", "spark",
      "racing_cars burners"),
    R("acetylene_combustion", "combustion", "2 C2H2 + 5 O2", "4 CO2 + 2 H2O", "exothermic", "spark", "welding_torches"),
    R("carbon_combustion", "combustion", "C + O2", "CO2", "exothermic", "heat", "coal_fires furnaces power_stations"),
    R("charcoal_incomplete_combustion", "combustion", "2 C + O2", "2 CO", "exothermic", "heat",
      "charcoal_grills", "carbon monoxide forms when air is short"),
    R("carbon_monoxide_combustion", "combustion", "2 CO + O2", "2 CO2", "exothermic", "spark",
      "catalytic_converters blast_furnaces"),
    R("sulfur_combustion", "combustion", "S8 + 8 O2", "8 SO2", "exothermic", "heat",
      "volcanoes sulfuric_acid_plants matches", "sulfur as its s8 ring"),
    R("hydrogen_sulfide_combustion", "combustion", "2 H2S + 3 O2", "2 SO2 + 2 H2O", "exothermic", "spark",
      "refineries", "burnt off in refinery gas flares"),
    R("magnesium_combustion", "combustion", "2 Mg + O2", "2 MgO", "exothermic", "heat",
      "fireworks flares sparklers", "bright white light"),
    R("aluminum_combustion", "combustion", "4 Al + 3 O2", "2 Al2O3", "exothermic", "heat", "rocket_boosters fireworks"),
    R("iron_combustion", "combustion", "4 Fe + 3 O2", "2 Fe2O3", "exothermic", "heat",
      "sparklers steel_wool", "the oxide written as fe2o3; burning iron also gives fe3o4"),
    R("phosphorus_combustion", "combustion", "P4 + 5 O2", "P4O10", "exothermic", "nothing",
      "old_matches labs", "white phosphorus lights by itself in air at about 30 c"),
    R("sodium_combustion", "combustion", "4 Na + O2", "2 Na2O", "exothermic", "heat",
      "labs", "with little oxygen; in plenty of air see sodium_peroxide_formation"),
    R("sodium_peroxide_formation", "combustion", "2 Na + O2", "Na2O2", "exothermic", "heat",
      "labs", "sodium burning in plenty of air"),
    R("ammonia_combustion", "combustion", "4 NH3 + 3 O2", "2 N2 + 6 H2O", "exothermic", "spark",
      "ammonia_engines ships"),
    R("hydrazine_combustion", "combustion", "N2H4 + O2", "N2 + 2 H2O", "exothermic", "spark", "rockets"),
    R("candle_combustion", "combustion", "C25H52 + 38 O2", "25 CO2 + 26 H2O", "exothermic", "spark",
      "candles", "paraffin wax as c25h52"),
    R("cellulose_combustion", "combustion", "C6H10O5 + 6 O2", "6 CO2 + 5 H2O", "exothermic", "spark",
      "campfires fireplaces forest_fires", "wood as one cellulose unit"),
    R("sucrose_combustion", "combustion", "C12H22O11 + 12 O2", "12 CO2 + 11 H2O", "exothermic", "heat",
      "demonstrations"),
    # synthesis
    R("haber_process", "synthesis", "N2 + 3 H2", "2 NH3", "exothermic", "catalyst pressure heat",
      "fertilizer_plants", "iron catalyst"),
    R("sulfur_trioxide_formation", "synthesis", "2 SO2 + O2", "2 SO3", "exothermic", "catalyst heat",
      "sulfuric_acid_plants", "contact process, vanadium oxide catalyst"),
    R("sulfuric_acid_formation", "synthesis", "SO3 + H2O", "H2SO4", "exothermic", "nothing",
      "sulfuric_acid_plants acid_rain"),
    R("sulfurous_acid_formation", "synthesis", "SO2 + H2O", "H2SO3", "exothermic", "nothing", "acid_rain clouds"),
    R("nitric_oxide_formation", "synthesis", "N2 + O2", "2 NO", "endothermic", "heat", "lightning engines"),
    R("nitrogen_dioxide_formation", "synthesis", "2 NO + O2", "2 NO2", "exothermic", "nothing",
      "smog exhaust_pipes", "brown gas"),
    R("dinitrogen_tetroxide_formation", "synthesis", "2 NO2", "N2O4", "exothermic", "nothing",
      "rockets labs", "rocket oxidiser"),
    R("ozone_formation", "synthesis", "3 O2", "2 O3", "endothermic", "light electricity",
      "stratosphere lightning photocopiers", "ultraviolet light or sparks"),
    R("sodium_chloride_synthesis", "synthesis", "2 Na + Cl2", "2 NaCl", "exothermic", "heat", "demonstrations"),
    R("hydrogen_chloride_synthesis", "synthesis", "H2 + Cl2", "2 HCl_gas", "exothermic", "light",
      "chemical_plants", "explosive in sunlight; the product is the gas, hydrochloric acid is its solution in water"),
    R("iron_sulfide_synthesis", "synthesis", "8 Fe + S8", "8 FeS", "exothermic", "heat",
      "classrooms", "sulfur as its s8 ring"),
    R("lime_slaking", "synthesis", "CaO + H2O", "Ca(OH)2", "exothermic", "nothing",
      "builders cement_works", "quicklime to slaked lime"),
    R("carbonic_acid_formation", "synthesis", "CO2 + H2O", "H2CO3", "exothermic", "nothing",
      "soda_bottles oceans blood", "energy sign from memory, small"),
    R("phosphoric_acid_formation", "synthesis", "P4O10 + 6 H2O", "4 H3PO4", "exothermic", "nothing",
      "phosphoric_acid_plants"),
    R("methanol_synthesis", "synthesis", "CO + 2 H2", "CH3OH", "exothermic", "catalyst pressure heat",
      "chemical_plants"),
    R("ethylene_hydration", "synthesis", "C2H4 + H2O", "C2H5OH", "exothermic", "catalyst heat pressure",
      "chemical_plants", "phosphoric acid catalyst"),
    R("urea_synthesis", "synthesis", "2 NH3 + CO2", "CH4N2O + H2O", "exothermic", "heat pressure",
      "fertilizer_plants", "urea is co(nh2)2, written as the substances table writes it"),
    R("plaster_setting", "synthesis", "(CaSO4)2H2O + 3 H2O", "2 CaSO4(H2O)2", "exothermic", "nothing",
      "builders plaster_casts", "plaster of paris, the half hydrate caso4 with half a water, takes water "
      "back and sets as gypsum; written doubled to keep whole atoms"),
    # decomposition
    R("limestone_calcination", "decomposition", "CaCO3", "CaO + CO2", "endothermic", "heat", "lime_kilns cement_works"),
    R("hydrogen_peroxide_decomposition", "decomposition", "2 H2O2", "2 H2O + O2", "exothermic", "catalyst",
      "first_aid demonstrations", "manganese dioxide or catalase"),
    R("baking_soda_decomposition", "decomposition", "2 NaHCO3", "Na2CO3 + H2O + CO2", "endothermic", "heat",
      "ovens cakes"),
    R("potassium_chlorate_decomposition", "decomposition", "2 KClO3", "2 KCl + 3 O2", "exothermic", "heat catalyst",
      "labs oxygen_generators", "manganese dioxide catalyst; energy sign from memory"),
    R("ammonium_nitrate_decomposition", "decomposition", "NH4NO3", "N2O + 2 H2O", "exothermic", "heat",
      "labs", "gentle heating gives laughing gas"),
    R("ozone_decomposition", "decomposition", "2 O3", "3 O2", "exothermic", "light catalyst",
      "stratosphere", "chlorine atoms from cfcs catalyse it"),
    R("mercury_oxide_decomposition", "decomposition", "2 HgO", "2 Hg + O2", "endothermic", "heat",
      "labs", "how oxygen was first isolated"),
    R("copper_carbonate_decomposition", "decomposition", "CuCO3", "CuO + CO2", "endothermic", "heat",
      "classrooms", "school book equation; the green powder of the classroom is mostly the basic "
      "carbonate, see malachite_decomposition"),
    R("malachite_decomposition", "decomposition", "Cu2CO3(OH)2", "2 CuO + CO2 + H2O", "endothermic", "heat",
      "classrooms", "basic copper carbonate, green to black; energy sign from memory"),
    R("sodium_azide_decomposition", "decomposition", "2 NaN3", "2 Na + 3 N2", "exothermic", "heat", "airbags"),
    R("kettle_scale_formation", "decomposition", "Ca(HCO3)2", "CaCO3 + H2O + CO2", "endothermic", "heat",
      "kettles boilers", "hard water leaves limescale"),
    R("carbonic_acid_decomposition", "decomposition", "H2CO3", "H2O + CO2", "endothermic", "nothing",
      "soda_bottles", "energy sign from memory, small"),
    # electrolysis
    R("electrolysis_of_water", "electrolysis", "2 H2O", "2 H2 + O2", "endothermic", "electricity",
      "electrolysers classrooms"),
    R("electrolysis_of_brine", "electrolysis", "2 NaCl + 2 H2O", "2 NaOH + H2 + Cl2", "endothermic", "electricity",
      "chlor_alkali_plants"),
    R("electrolysis_of_molten_salt", "electrolysis", "2 NaCl", "2 Na + Cl2", "endothermic", "electricity heat",
      "sodium_plants", "downs cell"),
    R("hall_heroult_process", "electrolysis", "2 Al2O3 + 3 C", "4 Al + 3 CO2", "endothermic", "electricity heat",
      "aluminum_smelters", "carbon anodes burn away"),
    # single replacement
    R("sodium_and_water", "single_replacement", "2 Na + 2 H2O", "2 NaOH + H2", "exothermic", "nothing",
      "demonstrations", "the hydrogen may catch fire"),
    R("potassium_and_water", "single_replacement", "2 K + 2 H2O", "2 KOH + H2", "exothermic", "nothing",
      "demonstrations", "lilac flame"),
    R("lithium_and_water", "single_replacement", "2 Li + 2 H2O", "2 LiOH + H2", "exothermic", "nothing",
      "demonstrations"),
    R("calcium_and_water", "single_replacement", "Ca + 2 H2O", "Ca(OH)2 + H2", "exothermic", "nothing", "labs"),
    R("magnesium_and_steam", "single_replacement", "Mg + H2O", "MgO + H2", "exothermic", "heat", "labs"),
    R("iron_and_steam", "single_replacement", "3 Fe + 4 H2O", "Fe3O4 + 4 H2", "exothermic", "heat", "labs"),
    R("zinc_and_hydrochloric_acid", "single_replacement", "Zn + 2 HCl", "ZnCl2 + H2", "exothermic", "nothing", "labs"),
    R("magnesium_and_hydrochloric_acid", "single_replacement", "Mg + 2 HCl", "MgCl2 + H2", "exothermic", "nothing",
      "classrooms"),
    R("aluminum_and_hydrochloric_acid", "single_replacement", "2 Al + 6 HCl", "2 AlCl3 + 3 H2", "exothermic", "nothing",
      "labs"),
    R("zinc_and_sulfuric_acid", "single_replacement", "Zn + H2SO4", "ZnSO4 + H2", "exothermic", "nothing", "labs"),
    R("iron_and_copper_sulfate", "single_replacement", "Fe + CuSO4", "FeSO4 + Cu", "exothermic", "nothing",
      "classrooms copper_mines"),
    R("zinc_and_copper_sulfate", "single_replacement", "Zn + CuSO4", "ZnSO4 + Cu", "exothermic", "nothing",
      "batteries classrooms", "the daniell cell"),
    R("copper_and_silver_nitrate", "single_replacement", "Cu + 2 AgNO3", "Cu(NO3)2 + 2 Ag", "exothermic", "nothing",
      "demonstrations", "silver crystals grow on copper"),
    R("thermite", "single_replacement", "Fe2O3 + 2 Al", "Al2O3 + 2 Fe", "exothermic", "heat",
      "rail_welding", "molten iron; a spark does not light it, a burning magnesium ribbon does"),
    R("chlorine_and_potassium_bromide", "single_replacement", "Cl2 + 2 KBr", "2 KCl + Br2", "exothermic", "nothing",
      "labs"),
    R("chlorine_and_sodium_iodide", "single_replacement", "Cl2 + 2 NaI", "2 NaCl + I2", "exothermic", "nothing",
      "labs"),
    R("magnesium_and_carbon_dioxide", "single_replacement", "2 Mg + CO2", "2 MgO + C", "exothermic", "heat",
      "demonstrations", "magnesium burns in carbon dioxide"),
    R("copper_oxide_and_hydrogen", "single_replacement", "CuO + H2", "Cu + H2O", "exothermic", "heat", "labs"),
    # redox
    R("iron_oxide_reduction", "redox", "Fe2O3 + 3 CO", "2 Fe + 3 CO2", "exothermic", "heat", "blast_furnaces"),
    R("zinc_oxide_reduction", "redox", "ZnO + C", "Zn + CO", "endothermic", "heat", "zinc_smelters"),
    R("silicon_production", "redox", "SiO2 + 2 C", "Si + 2 CO", "endothermic", "heat electricity",
      "silicon_plants", "electric arc furnace"),
    R("ammonia_oxidation", "redox", "4 NH3 + 5 O2", "4 NO + 6 H2O", "exothermic", "catalyst heat",
      "nitric_acid_plants", "ostwald process, platinum catalyst"),
    R("nitric_acid_formation", "redox", "3 NO2 + H2O", "2 HNO3 + NO", "exothermic", "nothing",
      "nitric_acid_plants acid_rain"),
    R("steam_reforming", "redox", "CH4 + H2O", "CO + 3 H2", "endothermic", "catalyst heat",
      "hydrogen_plants", "nickel catalyst"),
    R("water_gas_shift", "redox", "CO + H2O", "CO2 + H2", "exothermic", "catalyst", "hydrogen_plants"),
    R("sabatier_reaction", "redox", "CO2 + 4 H2", "CH4 + 2 H2O", "exothermic", "catalyst heat",
      "space_stations", "hydrogen reduces carbon dioxide; nickel catalyst"),
    R("water_gas_reaction", "redox", "C + H2O", "CO + H2", "endothermic", "heat",
      "gas_works", "steam over red hot coke"),
    R("copper_and_nitric_acid", "redox", "Cu + 4 HNO3", "Cu(NO3)2 + 2 NO2 + 2 H2O", "exothermic", "nothing",
      "labs", "concentrated acid, brown fumes"),
    R("chlorine_and_water", "redox", "Cl2 + H2O", "HCl + HOCl", "exothermic", "nothing",
      "swimming_pools water_works", "nearly thermoneutral; the energy sign is not trained, see UNSURE"),
    R("bleach_formation", "redox", "Cl2 + 2 NaOH", "NaCl + NaClO + H2O", "exothermic", "nothing",
      "chemical_plants", "chlorine into cold sodium hydroxide; about minus 100 kj, from formation "
      "enthalpies from memory"),
    R("lead_acid_battery_discharge", "redox", "Pb + PbO2 + 2 H2SO4", "2 PbSO4 + 2 H2O", "exothermic", "nothing",
      "car_batteries", "runs backwards when charged; energy sign from memory"),
    R("hydrogen_fuel_cell", "redox", "2 H2 + O2", "2 H2O", "exothermic", "catalyst",
      "fuel_cells buses", "same sum as hydrogen combustion, energy as electricity"),
    R("sodium_peroxide_and_carbon_dioxide", "redox", "2 Na2O2 + 2 CO2", "2 Na2CO3 + O2", "exothermic", "nothing",
      "submarines"),
    R("potassium_superoxide_and_carbon_dioxide", "redox", "4 KO2 + 2 CO2", "2 K2CO3 + 3 O2", "exothermic", "nothing",
      "rebreathers spacecraft"),
    # corrosion
    R("rusting", "corrosion", "4 Fe + 3 O2 + 6 H2O", "4 Fe(OH)3", "exothermic", "nothing",
      "cars bridges ships", "needs both air and water; rust written as iron hydroxide"),
    R("copper_patina", "corrosion", "2 Cu + O2 + CO2 + H2O", "Cu2CO3(OH)2", "exothermic", "nothing",
      "roofs statues", "the green of old copper roofs; energy sign from memory"),
    R("silver_tarnishing", "corrosion", "4 Ag + 2 H2S + O2", "2 Ag2S + 2 H2O", "exothermic", "nothing",
      "cutlery jewelry"),
    R("aluminum_passivation", "corrosion", "4 Al + 3 O2", "2 Al2O3", "exothermic", "nothing",
      "cans aircraft window_frames", "a thin oxide skin stops further attack"),
    R("zinc_corrosion", "corrosion", "2 Zn + O2", "2 ZnO", "exothermic", "nothing",
      "galvanized_steel roofs", "zinc corrodes instead of the steel under it"),
    # acid and base
    R("neutralisation_hcl_naoh", "acid_base", "HCl + NaOH", "NaCl + H2O", "exothermic", "nothing", "labs"),
    R("neutralisation_h2so4_naoh", "acid_base", "H2SO4 + 2 NaOH", "Na2SO4 + 2 H2O", "exothermic", "nothing", "labs"),
    R("neutralisation_hno3_koh", "acid_base", "HNO3 + KOH", "KNO3 + H2O", "exothermic", "nothing", "labs"),
    R("antacid_magnesium_hydroxide", "acid_base", "Mg(OH)2 + 2 HCl", "MgCl2 + 2 H2O", "exothermic", "nothing",
      "stomachs antacids", "milk of magnesia"),
    R("antacid_aluminum_hydroxide", "acid_base", "Al(OH)3 + 3 HCl", "AlCl3 + 3 H2O", "exothermic", "nothing",
      "stomachs antacids"),
    R("calcium_carbonate_and_hydrochloric_acid", "acid_base", "CaCO3 + 2 HCl", "CaCl2 + H2O + CO2", "exothermic",
      "nothing", "stomachs antacids limestone", "fizzing test for carbonates"),
    R("baking_soda_and_vinegar", "acid_base", "NaHCO3 + CH3COOH", "C2H3NaO2 + H2O + CO2", "endothermic", "nothing",
      "kitchens volcano_models", "the mixture gets cold"),
    R("baking_soda_and_hydrochloric_acid", "acid_base", "NaHCO3 + HCl", "NaCl + H2O + CO2", "endothermic", "nothing",
      "stomachs antacids", "energy sign from memory"),
    R("calcium_hydroxide_and_hydrochloric_acid", "acid_base", "Ca(OH)2 + 2 HCl", "CaCl2 + 2 H2O", "exothermic",
      "nothing", "labs"),
    R("marble_and_sulfuric_acid", "acid_base", "CaCO3 + H2SO4", "CaSO4 + H2O + CO2", "exothermic", "nothing",
      "acid_rain statues", "nearly thermoneutral; the energy sign is not trained, see UNSURE"),
    R("ammonium_sulfate_formation", "acid_base", "2 NH3 + H2SO4", "(NH4)2SO4", "exothermic", "nothing",
      "fertilizer_plants"),
    R("ammonium_nitrate_formation", "acid_base", "NH3 + HNO3", "NH4NO3", "exothermic", "nothing", "fertilizer_plants"),
    R("ammonium_chloride_formation", "acid_base", "NH3 + HCl_gas", "NH4Cl", "exothermic", "nothing",
      "labs", "white smoke from two gases"),
    R("carbon_dioxide_scrubbing_naoh", "acid_base", "CO2 + 2 NaOH", "Na2CO3 + H2O", "exothermic", "nothing",
      "submarines labs"),
    R("carbon_dioxide_scrubbing_lioh", "acid_base", "CO2 + 2 LiOH", "Li2CO3 + H2O", "exothermic", "nothing",
      "spacecraft", "the apollo scrubbers"),
    R("limestone_dissolving", "acid_base", "CaCO3 + H2O + CO2", "Ca(HCO3)2", "exothermic", "nothing",
      "caves rainwater", "carves caves; energy sign from memory"),
    # precipitation and double replacement
    R("silver_chloride_precipitation", "precipitation", "AgNO3 + NaCl", "AgCl + NaNO3", "exothermic", "nothing",
      "labs photography", "white solid, test for chloride"),
    R("barium_sulfate_precipitation", "precipitation", "BaCl2 + Na2SO4", "BaSO4 + 2 NaCl", "exothermic", "nothing",
      "labs hospitals", "test for sulfate"),
    R("lead_iodide_precipitation", "precipitation", "Pb(NO3)2 + 2 KI", "PbI2 + 2 KNO3", "exothermic", "nothing",
      "demonstrations", "golden rain"),
    R("calcium_carbonate_precipitation", "precipitation", "CaCl2 + Na2CO3", "CaCO3 + 2 NaCl", "endothermic", "nothing",
      "water_softening labs", "small heat; energy sign from memory"),
    R("limewater_test", "precipitation", "Ca(OH)2 + CO2", "CaCO3 + H2O", "exothermic", "nothing",
      "classrooms", "limewater turns milky"),
    R("copper_hydroxide_precipitation", "precipitation", "CuSO4 + 2 NaOH", "Cu(OH)2 + Na2SO4", "exothermic",
      "nothing", "labs", "pale blue solid"),
    R("iron_hydroxide_precipitation", "precipitation", "FeCl3 + 3 NaOH", "Fe(OH)3 + 3 NaCl", "exothermic",
      "nothing", "labs water_treatment", "rust brown solid"),
    R("magnesium_hydroxide_from_seawater", "precipitation", "MgCl2 + Ca(OH)2", "Mg(OH)2 + CaCl2", "exothermic",
      "nothing", "magnesium_plants seawater", "small heat; energy sign from memory"),
    R("calcium_carbide_and_water", "double_replacement", "CaC2 + 2 H2O", "C2H2 + Ca(OH)2", "exothermic", "nothing",
      "miners_lamps", "acetylene for the lamp flame"),
    # life
    R("photosynthesis", "photosynthesis", "6 CO2 + 6 H2O", "C6H12O6 + 6 O2", "endothermic", "light",
      "leaves algae", "chlorophyll catches the light"),
    R("respiration_of_glucose", "respiration", "C6H12O6 + 6 O2", "6 CO2 + 6 H2O", "exothermic", "enzyme",
      "cells mitochondria"),
    R("fermentation_of_glucose", "fermentation", "C6H12O6", "2 C2H5OH + 2 CO2", "exothermic", "enzyme",
      "breweries bakeries wineries", "yeast"),
    R("lactic_acid_fermentation", "fermentation", "C6H12O6", "2 C3H6O3", "exothermic", "enzyme", "muscles yogurt"),
    R("vinegar_fermentation", "fermentation", "C2H5OH + O2", "CH3COOH + H2O", "exothermic", "enzyme",
      "vinegar_makers", "acetobacter"),
    # nuclear: fusion in stars and reactors
    R("deuterium_formation", "fusion", "2 h1", "h2 + positron + neutrino", "exothermic", "heat pressure",
      "sun stars", "first step of the proton proton chain; the positron's later annihilation adds 1.022 mev"),
    R("helium_3_formation", "fusion", "h2 + h1", "he3 + gamma", "exothermic", "heat pressure",
      "sun stars", "second step of the proton proton chain"),
    R("helium_3_fusion", "fusion", "2 he3", "he4 + 2 h1", "exothermic", "heat pressure",
      "sun stars", "last step of the proton proton chain"),
    R("hydrogen_burning", "fusion", "4 h1", "he4 + 2 positron + 2 neutrino", "exothermic", "heat pressure",
      "sun stars", "the chain's sum; the two positrons' later annihilation adds 2.044 mev"),
    R("beryllium_7_formation", "fusion", "he3 + he4", "be7 + gamma", "exothermic", "heat pressure",
      "sun stars", "opens the second branch of the proton proton chain"),
    R("lithium_7_proton_capture", "fusion", "li7 + h1", "2 he4", "exothermic", "heat pressure",
      "sun stars", "closes the second branch of the proton proton chain"),
    R("beryllium_8_formation", "fusion", "2 he4", "be8", "endothermic", "heat pressure",
      "red_giants", "beryllium 8 is unbound and falls apart again"),
    R("triple_alpha_process", "fusion", "3 he4", "c12 + gamma", "exothermic", "heat pressure", "red_giants stars"),
    R("carbon_12_alpha_capture", "fusion", "c12 + he4", "o16 + gamma", "exothermic", "heat pressure",
      "red_giants stars"),
    R("carbon_burning", "fusion", "2 c12", "ne20 + he4", "exothermic", "heat pressure",
      "massive_stars", "one of three branches"),
    R("carbon_burning_sodium_branch", "fusion", "2 c12", "na23 + h1", "exothermic", "heat pressure",
      "massive_stars", "one of three branches"),
    R("carbon_burning_magnesium_branch", "fusion", "2 c12", "mg24 + gamma", "exothermic", "heat pressure",
      "massive_stars", "one of three branches, the rarest"),
    R("cno_carbon_12_proton_capture", "fusion", "c12 + h1", "n13 + gamma", "exothermic", "heat pressure",
      "massive_stars sun", "first step of the cno cycle"),
    R("cno_carbon_13_proton_capture", "fusion", "c13 + h1", "n14 + gamma", "exothermic", "heat pressure",
      "massive_stars sun", "the cno cycle after nitrogen_13_decay"),
    R("cno_nitrogen_14_proton_capture", "fusion", "n14 + h1", "o15 + gamma", "exothermic", "heat pressure",
      "massive_stars sun", "slowest step of the cno cycle"),
    R("cno_nitrogen_15_proton_capture", "fusion", "n15 + h1", "c12 + he4", "exothermic", "heat pressure",
      "massive_stars sun", "closes the cno cycle"),
    R("deuterium_tritium_fusion", "fusion", "h2 + h3", "he4 + n1", "exothermic", "heat pressure",
      "fusion_reactors early_universe", "also how the first minutes made helium"),
    R("deuterium_deuterium_fusion", "fusion", "2 h2", "he3 + n1", "exothermic", "heat pressure",
      "fusion_reactors early_universe", "one of two branches, about half each"),
    R("deuterium_deuterium_fusion_tritium_branch", "fusion", "2 h2", "h3 + h1", "exothermic", "heat pressure",
      "fusion_reactors early_universe", "one of two branches, about half each"),
    R("deuterium_helium_3_fusion", "fusion", "h2 + he3", "he4 + h1", "exothermic", "heat pressure",
      "fusion_reactors early_universe", "no neutron; also how the first minutes made helium"),
    # nuclear: fission
    R("uranium_235_fission", "fission", "u235 + n1", "ba141 + kr92 + 3 n1", "exothermic", "nothing",
      "reactors", "one channel of many"),
    R("uranium_235_fission_xenon_channel", "fission", "u235 + n1", "xe140 + sr94 + 2 n1", "exothermic", "nothing",
      "reactors", "channel from memory"),
    R("plutonium_239_fission", "fission", "pu239 + n1", "xe134 + zr103 + 3 n1", "exothermic", "nothing",
      "reactors", "channel from memory"),
    # nuclear: neutron capture
    R("proton_neutron_capture", "neutron_capture", "h1 + n1", "h2 + gamma", "exothermic", "nothing",
      "early_universe reactors", "the first nuclei of the first minutes; also in the water of a reactor"),
    R("carbon_14_formation", "neutron_capture", "n14 + n1", "c14 + h1", "exothermic", "nothing",
      "atmosphere", "neutrons from cosmic rays; a proton leaves"),
    R("uranium_238_neutron_capture", "neutron_capture", "u238 + n1", "u239 + gamma", "exothermic", "nothing",
      "reactors", "first step of breeding plutonium"),
    # nuclear: decay
    R("neutron_beta_decay", "beta_decay", "n1", "h1 + e + antineutrino", "exothermic", "nothing",
      "reactors early_universe", "a free neutron lasts about fifteen minutes"),
    R("carbon_14_decay", "beta_decay", "c14", "n14 + e + antineutrino", "exothermic", "nothing",
      "archaeology bones", "radiocarbon dating"),
    R("tritium_decay", "beta_decay", "h3", "he3 + e + antineutrino", "exothermic", "nothing",
      "exit_signs fusion_reactors"),
    R("potassium_40_decay", "beta_decay", "k40", "ca40 + e + antineutrino", "exothermic", "nothing",
      "rocks bananas", "main branch, about nine in ten; the rest is potassium_40_electron_capture"),
    R("uranium_239_decay", "beta_decay", "u239", "np239 + e + antineutrino", "exothermic", "nothing",
      "reactors", "second step of breeding plutonium"),
    R("neptunium_239_decay", "beta_decay", "np239", "pu239 + e + antineutrino", "exothermic", "nothing",
      "reactors", "last step of breeding plutonium"),
    R("cesium_137_decay", "beta_decay", "cs137", "ba137 + e + antineutrino", "exothermic", "nothing",
      "reactor_waste", "gamma omitted"),
    R("iodine_131_decay", "beta_decay", "i131", "xe131 + e + antineutrino", "exothermic", "nothing",
      "hospitals", "gamma omitted"),
    R("nitrogen_13_decay", "beta_decay", "n13", "c13 + positron + neutrino", "exothermic", "nothing",
      "stars", "beta plus, inside the cno cycle"),
    R("oxygen_15_decay", "beta_decay", "o15", "n15 + positron + neutrino", "exothermic", "nothing",
      "stars", "beta plus, inside the cno cycle"),
    R("fluorine_18_decay", "beta_decay", "f18", "o18 + positron + neutrino", "exothermic", "nothing",
      "pet_scanners hospitals", "beta plus"),
    # nuclear: electron capture
    R("potassium_40_electron_capture", "electron_capture", "k40 + e", "ar40 + neutrino", "exothermic", "nothing",
      "rocks", "about one decay in ten; the argon dates rocks"),
    R("beryllium_7_electron_capture", "electron_capture", "be7 + e", "li7 + neutrino", "exothermic", "nothing",
      "sun stars", "second branch of the proton proton chain"),
    R("uranium_238_alpha_decay", "alpha_decay", "u238", "th234 + he4", "exothermic", "nothing", "rocks"),
    R("radium_226_alpha_decay", "alpha_decay", "ra226", "rn222 + he4", "exothermic", "nothing", "rocks old_watches"),
    R("radon_222_alpha_decay", "alpha_decay", "rn222", "po218 + he4", "exothermic", "nothing", "basements rocks"),
    R("americium_241_alpha_decay", "alpha_decay", "am241", "np237 + he4", "exothermic", "nothing", "smoke_detectors"),
    R("plutonium_238_alpha_decay", "alpha_decay", "pu238", "u234 + he4", "exothermic", "nothing",
      "space_probes", "heat source of deep space probes"),
]


def names_asked(table: Sequence[Reaction]) -> list[Reaction]:
    """The rows whose ``<names> gives`` form has one answer: the table holds a single product
    set for their reactants, and the row is not a fission (a heavy nucleus splits in hundreds
    of ways, the table holds a channel or two)."""
    products: dict[tuple[str, ...], set[frozenset[str]]] = {}
    for r in table:
        products.setdefault(r.names_key, set()).add(frozenset(r.product_names))
    return [r for r in table if len(products[r.names_key]) == 1 and r.kind != "fission"]


def rows() -> list[dict]:
    """The table as JSON rows for ``data/truth-v5/reactions.jsonl``. ``names`` is the ``gives``
    line of a row and empty where that form is not asked (:func:`names_asked`); ``untrained``
    lists the record keys that are not trained; ``q_mev`` is ``null`` for a chemical row."""
    asked = {r.name for r in names_asked(TABLE)}
    return [{"name": r.name, "kind": r.kind, "nuclear": r.nuclear,
             "reactants": [[s.coef, s.formula or s.text] for s in r.reactants],
             "products": [[s.coef, s.formula or s.text] for s in r.products],
             "reactant_names": r.reactant_names, "product_names": r.product_names,
             "equation": r.equation,
             "names": r.names_prompt + " " + " ".join(r.product_names) if r.name in asked else "",
             "energy": r.energy, "q_mev": float(f"{r.q_mev:.4g}") if r.nuclear else None,
             "needs": list(r.needs), "where": list(r.where), "untrained": list(untrained(r.name)),
             "notes": r.notes}
            for r in TABLE]


def _ok_set(answer: str, expected: str) -> bool:
    """The same words in any order, each once."""
    got = answer.split()
    return bool(got) and len(set(got)) == len(got) and set(got) == set(expected.split())


def _ok_equation(answer: str, left: list[Species], right: list[Species]) -> bool:
    got = parse_equation(answer, with_coef=True)
    if got is None:
        return False
    want = [sorted((s.key, s.coef) for s in side) for side in (left, right)]
    have = [sorted((s.key, s.coef) for s in side) for side in got]
    return want == have


def _coef_text(coefs: list[int], left: list[Species], right: list[Species]) -> str:
    """Coefficients as a ``balance`` answer: digit tokens while every one is a single digit, the
    full equation otherwise."""
    if max(coefs) < 10:
        return " ".join(num(c) for c in coefs)
    terms = [f"{num(c)} x {s.text}" for c, s in zip(coefs, left + right)]
    return " plus ".join(terms[:len(left)]) + " gives " + " plus ".join(terms[len(left):])


class ReactionsGate:
    """The gate: owns the reaction questions, generates them and judges answers."""

    topic = TOPIC
    KEYS = KEYS

    def __init__(self, table: list[Reaction] | None = None) -> None:
        self.table = list(table) if table is not None else list(TABLE)
        self.by_name = {r.name: r for r in self.table}
        self.names: set[str] = {s.name for r in self.table for s in r.species}
        # the smallest coefficients, where the balance is unique; None otherwise
        self.balanced: dict[str, list[int] | None] = {r.name: balance(list(r.reactants), list(r.products))
                                                      for r in self.table}
        self.by_reactants: dict[tuple[str, ...], list[Reaction]] = {}
        self.by_species: dict[tuple[frozenset, frozenset], Reaction] = {}
        for r in self.table:
            self.by_reactants.setdefault(r.names_key, []).append(r)
            self.by_species.setdefault((frozenset(s.key for s in r.reactants), frozenset(s.key for s in r.products)), r)
        self.names_unique = names_asked(self.table)
        self.index: dict[str, dict[str, list[str]]] = {f: {} for f in FILTERS}
        for r in self.table:
            for f, values in (("kind", [r.kind]), ("where", r.where), ("needs", r.needs),
                              ("reactant", r.reactant_names), ("product", r.product_names)):
                if f in untrained(r.name):
                    continue
                for v in values:
                    self.index[f].setdefault(v, []).append(r.name)
        # the values a list question can name: what the table holds; a substance may have no
        # reaction in one of its two roles, and the answer is then ``none``
        self.values: dict[str, set[str]] = {f: set(self.index[f]) for f in FILTERS}
        self.values["reactant"] = self.values["product"] = self.names
        self.facts = [(r, k) for r in self.table for k in r.keys if k not in untrained(r.name)]
        self.nuclides = sorted({s.text for r in self.table for s in r.species if s.nuclear})
        self.nuclide_facts = [(t, k) for t in self.nuclides for k in NUCLIDE_KEYS]
        self.lists = [(f, v) for f in FILTERS for v in sorted(self.values[f])
                      if len(self.index[f].get(v, ())) <= MAX_LIST]
        self.chemical = [r for r in self.table if not r.nuclear]
        self.balanceable = [r for r in self.table
                            if self.balanced[r.name] == [s.coef for s in r.species] and max(self.balanced[r.name]) < 10]
        # every species of the table once, for the wrong equations of the check questions
        once = {(s.nuclear, s.key): s.times(1) for r in self.table for s in r.species}
        self.pool: dict[bool, list[Species]] = {
            nuclear: sorted((s for s in once.values() if s.nuclear == nuclear), key=lambda s: s.text)
            for nuclear in (False, True)}
        # how many distinct lines each finite kind of question has; ``generate`` stops drawing
        # from a kind once it has given them all
        orders = {(tuple(s.text for s in left), tuple(s.text for s in right))
                  for r in self.balanceable for left in permutations(r.reactants) for right in permutations(r.products)}
        self.sizes: dict[str, int] = {"fact": len({ln.text for ln in self._fact_lines()}),
                                      "names": len({r.names_prompt for r in self.names_unique}),
                                      "balance": len(orders), "list": len(self.lists)}

    # -- records and questions -----------------------------------------------------------------

    def records(self) -> list[Line]:
        return [ln for r in self.table for ln in r.record_lines()]

    def generate(self, rng: random.Random, n: int) -> list[Line]:
        out: list[Line] = []
        seen: set[str] = set()
        makers = ((0.35, "fact", self._fact), (0.45, "names", self._names), (0.60, "balance", self._balance),
                  (0.75, "check", self._check), (0.90, "count", self._count), (1.01, "list", self._list))
        attempts = 0
        yes_no = {"yes": 0, "no": 0}
        left = dict(self.sizes)
        while len(out) < n and attempts < 80 * n + 1000:
            attempts += 1
            u = rng.random()
            source, make = next((s, m) for limit, s, m in makers if u < limit)
            if left.get(source, 1) <= 0:  # every line of that kind is out: count questions never run out
                source, make = "count", self._count
            ln = make(rng)
            if ln is None or ln.text in seen or count_tokens(ln.text) > MAX_TOKENS:
                continue
            if ln.kind == "yesno":
                # distinct wrong equations far outnumber the right ones; keep the answers even
                other = "no" if ln.answer == "yes" else "yes"
                if yes_no[ln.answer] >= yes_no[other] + YES_NO_SLACK:
                    continue
                yes_no[ln.answer] += 1
            if source in left:
                left[source] -= 1
            seen.add(ln.text)
            out.append(ln)
        if len(out) < n:
            raise ValueError(f"only {len(out)} distinct lines found for n = {n}")
        return out

    def all_facts(self) -> list[Line]:
        """Every table question once: each key of each reaction, each nuclide key, each
        unambiguous ``gives`` form, each list of at most six names. The same lines ``generate``
        draws its fact kinds from."""
        out = self._fact_lines()
        out += [self._line(r.names_prompt, r.value("products"), "fact", "names", r.name) for r in self.names_unique]
        out += [self._list_line(f, v) for f, v in self.lists]
        seen: set[str] = set()
        return [ln for ln in out if not (ln.text in seen or seen.add(ln.text))]

    def _line(self, prompt: str, answer: str, kind: str, q: str, name: str = "") -> Line:
        return Line(prompt, answer, TOPIC, kind, {"q": q, "name": name})

    def _fact_lines(self) -> list[Line]:
        return ([self._line(f"{r.name} {k}", r.value(k), "fact", "fact", r.name) for r, k in self.facts]
                + [self._nuclide_line(t, k) for t, k in self.nuclide_facts])

    def _nuclide_line(self, token: str, key: str) -> Line:
        return self._line(f"nuclide {token} {key}", self._nuclide_value(token, key), "fact", "nuclide")

    def _list_line(self, f: str, value: str) -> Line:
        names = sorted(self.index[f].get(value, ()))
        return self._line(f"reactions {f} {value}", " ".join(names) or "none", "fact", "list")

    @staticmethod
    def _nuclide_value(token: str, key: str) -> str:
        if key == "name":
            return nuclide_name(token)
        return num(species(token).counts[SUMS[key]])

    def _fact(self, rng: random.Random) -> Line:
        i = rng.randrange(len(self.facts) + len(self.nuclide_facts))
        if i < len(self.facts):
            r, key = self.facts[i]
            return self._line(f"{r.name} {key}", r.value(key), "fact", "fact", r.name)
        return self._nuclide_line(*self.nuclide_facts[i - len(self.facts)])

    def _names(self, rng: random.Random) -> Line:
        r = rng.choice(self.names_unique)
        return self._line(r.names_prompt, r.value("products"), "fact", "names", r.name)

    def _balance(self, rng: random.Random) -> Line:
        # the table's coefficients are the unique smallest balance (checked in __init__), and they
        # stay so when the species of a side are reordered. The sides are never swapped: a balance
        # question still says ``gives``, and the reverse of a reaction is not in the table.
        r = rng.choice(self.balanceable)
        left, right = list(r.reactants), list(r.products)
        if rng.random() < 0.5:
            rng.shuffle(left)
        if rng.random() < 0.5:
            rng.shuffle(right)
        coefs = [s.coef for s in left + right]
        prompt = "balance " + " plus ".join(s.text for s in left) + " gives " + " plus ".join(s.text for s in right)
        return self._line(prompt, " ".join(num(c) for c in coefs), "calc", "balance", r.name)

    def _check(self, rng: random.Random) -> Line:
        r = rng.choice(self.table)
        left, right = list(r.reactants), list(r.products)
        if rng.random() < 0.5:
            rng.shuffle(left)
            rng.shuffle(right)
        u = rng.random()
        if u < 0.25:
            k = rng.randint(2, 5)
            left = [s.times(s.coef * k) for s in left]
            right = [s.times(s.coef * k) for s in right]
        elif u < 0.65:
            # one coefficient changed; never that of a species the balance cannot count (gamma)
            spots = [(side, i) for side in (left, right) for i, s in enumerate(side) if not s.free]
            if spots:
                side, i = rng.choice(spots)
                s = side[i]
                new = rng.randint(1, max(9, 2 * s.coef))
                while new == s.coef:
                    new = rng.randint(1, max(9, 2 * s.coef))
                side[i] = s.times(new)
        elif u < 0.80:
            # one species changed: a lepton for its opposite, any other for another species of the
            # table, or one species of several left out
            side = left if rng.random() < 0.5 else right
            i = rng.randrange(len(side))
            s = side[i]
            present = {x.key for x in left + right}
            others = [x for x in self.pool[r.nuclear] if x.key not in present]
            if s.text in OPPOSITE and (OPPOSITE[s.text],) not in present:
                side[i] = species(OPPOSITE[s.text], s.coef)
            elif len(side) > 1 and rng.random() < 0.4:
                del side[i]
            elif others:
                side[i] = rng.choice(others).times(s.coef)
        answer = "yes" if is_balanced(left, right) else "no"
        return self._line(f"check {equation_text(left, right)} balanced", answer, "yesno", "check", r.name)

    def _count(self, rng: random.Random) -> Line:
        r = rng.choice(self.table)
        side = list(r.reactants if rng.random() < 0.5 else r.products)
        if rng.random() < 0.6:
            side = [rng.choice(side)]
        u = rng.random()
        if u < 0.6:
            # mostly single digits; the two-digit coefficients keep the supply of questions open
            lo, hi = (1, 9) if u < 0.5 else (10, 99)
            side = [s.times(rng.randint(lo, hi)) for s in side]
        terms = " plus ".join(s.dense() for s in side)
        if r.nuclear:
            keys = [k for k, c in SUMS.items() if c != "l" or any(s.counts["l"] for s in side) or rng.random() < 0.1]
            key = rng.choice(keys)
            total = sum(s.coef * s.counts[SUMS[key]] for s in side)
            return self._line(f"count_nuclear {terms} {key}", num(total), "calc", "count_nuclear", r.name)
        present = sorted({el for s in side for el in s.counts})
        others = sorted({el for s in r.species for el in s.counts} - set(present))
        el = rng.choice(others) if others and rng.random() < 0.1 else rng.choice(present)
        total = sum(s.coef * s.counts.get(el, 0) for s in side)
        return self._line(f"count_atoms {terms} {ELEMENT_NAMES[el]}", num(total), "calc", "count_atoms", r.name)

    def _list(self, rng: random.Random) -> Line:
        return self._list_line(*rng.choice(self.lists))

    # -- judging ---------------------------------------------------------------------------------

    def owns(self, prompt: str) -> bool:
        words = prompt.split()
        if not words:
            return False
        w0 = words[0]
        rest = " ".join(words[1:])
        if w0 in self.by_name:
            return rest in self.by_name[w0].keys and rest not in untrained(w0)
        if w0 == "balance":
            return parse_equation(rest, with_coef=False) is not None
        if w0 == "check":
            return words[-1] == "balanced" and parse_equation(" ".join(words[1:-1]), with_coef=True) is not None
        if w0 in ("count_atoms", "count_nuclear"):
            return self._count_parts(words) is not None
        if w0 == "nuclide":
            return len(words) == 3 and nuclide(words[1]) is not None and words[2] in NUCLIDE_KEYS
        if w0 == "reactions":
            return (len(words) == 3 and words[1] in FILTERS and words[2] in self.values[words[1]]
                    and len(self.index[words[1]].get(words[2], ())) <= MAX_LIST)
        if words[-1] == "gives" and len(words) >= 2:
            return all(n in self.names for n in " ".join(words[:-1]).split(" plus "))
        return False

    @staticmethod
    def _count_parts(words: list[str]) -> tuple[list[Species], str] | None:
        """``count_atoms 2 x h 2 o 1 hydrogen`` -> (species, element symbol), chemical species only;
        ``count_nuclear 2 x he4 plus 1 x n1 charge`` -> (species, ``z``), nuclear species only."""
        if len(words) < 4:
            return None
        nuclear = words[0] == "count_nuclear"
        what = SUMS.get(words[-1]) if nuclear else _NAME_TO_SYMBOL.get(words[-1], words[-1])
        if what is None or (not nuclear and what not in Z):
            return None
        parts: list[Species] = []
        for term in " ".join(words[1:-1]).split(" plus "):
            sp = parse_species(term, with_coef=True)
            if sp is None or sp.nuclear != nuclear:
                return None
            parts.append(sp)
        return parts, what

    def check(self, prompt: str, answer: str) -> Verdict:
        words = prompt.split()
        answer = answer.strip()
        if not self.owns(prompt):
            return Verdict(False, None, "not my question")
        w0 = words[0]
        if w0 in self.by_name:
            r = self.by_name[w0]
            key = words[1]
            want = r.value(key)
            if key in ("kind", "energy"):
                ok = answer == want
            elif key == "equation":
                ok = _ok_equation(answer, list(r.reactants), list(r.products))
            elif key == "q_mev":
                got, value = parse_num(answer), parse_num(want)
                ok = is_finite(got) and abs(got - value) <= TOLERANCE * abs(value)
            else:
                ok = _ok_set(answer, want)
            return Verdict(ok, want, "" if ok else f"table says {want}")
        if w0 == "balance":
            return self._check_balance(*parse_equation(" ".join(words[1:]), with_coef=False), answer)
        if w0 == "check":
            left, right = parse_equation(" ".join(words[1:-1]), with_coef=True)
            want = "yes" if is_balanced(left, right) else "no"
            return Verdict(answer == want, want, "" if answer == want else f"it is {want}")
        if w0 in ("count_atoms", "count_nuclear"):
            parts, what = self._count_parts(words)
            want = num(sum(s.coef * s.counts.get(what, 0) for s in parts))
            return Verdict(answer == want, want, "" if answer == want else f"count is {want}")
        if w0 == "nuclide":
            want = self._nuclide_value(words[1], words[2])
            return Verdict(answer == want, want, "" if answer == want else f"it is {want}")
        if w0 == "reactions":
            names = self.index[words[1]].get(words[2], [])
            want = " ".join(sorted(names)) if names else "none"
            ok = _ok_set(answer, want) if names else answer == "none"
            return Verdict(ok, want, "" if ok else f"table lists {want}")
        # names form: the names as written; a chemical equation names each reactant once, however
        # often it is written, a nuclear one counts them (two protons are not one)
        names = " ".join(words[:-1]).split(" plus ")
        matches = self.by_reactants.get(tuple(sorted(names)))
        if not matches:
            matches = [r for r in self.by_reactants.get(tuple(sorted(set(names))), []) if not r.nuclear]
        if not matches:
            return Verdict(False, None, "no reaction with these reactants in the table")
        want = matches[0].value("products")
        ok = any(_ok_set(answer, r.value("products")) for r in matches)
        return Verdict(ok, want, "" if ok else f"table says {want}")

    def _check_balance(self, left: list[Species], right: list[Species], answer: str) -> Verdict:
        """Judge the coefficients given for ``left gives right``. The rule decides where the
        balance is unique; where it is not, the table's own reaction does; a species set with
        several balances that the table does not hold is right with any smallest balance, unless
        it has a species the balance cannot count."""
        coefs = balance(left, right)
        if coefs is None:
            coefs = self._table_coefs(left, right)
        want = _coef_text(coefs, left, right) if coefs is not None else None
        got = self._coefs(answer, left, right)
        if got is None or min(got) < 1:
            return Verdict(False, want, "not a coefficient for every species")
        scaled = [[s.times(c) for c, s in zip(cs, side)]
                  for cs, side in ((got[:len(left)], left), (got[len(left):], right))]
        if not is_balanced(*scaled):
            return Verdict(False, want, "does not balance")
        if coefs is not None:
            if got == coefs:
                return Verdict(True, want, "")
            reason = "balances, but not with the smallest coefficients" if minimal(got) == coefs else \
                "balances, but the reaction of the table is another"
            return Verdict(False, want, reason)
        free = [s.text for s in left + right if s.free]
        if free:
            return Verdict(False, None, f"cannot be judged: a balance does not count {free[0]}")
        smallest = _coef_text(minimal(got), left, right)
        if minimal(got) != got:
            return Verdict(False, smallest, "balances, but not with the smallest coefficients")
        return Verdict(True, smallest, "balances; this species set has more than one balance")

    def _table_coefs(self, left: list[Species], right: list[Species]) -> list[int] | None:
        """The coefficients of the table's reaction with these species on these sides, in the
        order of the question; ``None`` when the table has none."""
        keys = [s.key for s in left], [s.key for s in right]
        if any(len(set(side)) != len(side) for side in keys):
            return None
        r = self.by_species.get((frozenset(keys[0]), frozenset(keys[1])))
        if r is None:
            return None
        coef = [{s.key: s.coef for s in side} for side in (r.reactants, r.products)]
        return [coef[0][k] for k in keys[0]] + [coef[1][k] for k in keys[1]]

    @staticmethod
    def _coefs(answer: str, left: list[Species], right: list[Species]) -> list[int] | None:
        """Coefficients from a digit list (one single-digit token per species) or from a full equation."""
        words = answer.split()
        if len(words) == len(left) + len(right) and all(w in DIGITS for w in words):
            return [int(w) for w in words]
        eq = parse_equation(answer, with_coef=True)
        if eq is None or [[s.key for s in side] for side in eq] != [[s.key for s in side] for side in (left, right)]:
            return None
        return [s.coef for s in eq[0] + eq[1]]


def gate() -> ReactionsGate:
    return ReactionsGate()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="write the table as JSON lines")
    b.add_argument("--out", type=Path, default=Path("data/truth-v5/reactions.jsonl"))
    s = sub.add_parser("sample", help="print records and generated lines")
    s.add_argument("--n", type=int, default=30)
    s.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    g = gate()
    if args.cmd == "build":
        bad = [r.name for r in g.table if not is_balanced(list(r.reactants), list(r.products))]
        if bad:
            raise SystemExit(f"unbalanced table rows: {bad}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8", newline="\n") as f:
            for row in rows():
                f.write(json.dumps(row) + "\n")
        print(json.dumps({"rows": len(g.table), "nuclear": sum(r.nuclear for r in g.table),
                          "records": len(g.records()), "out": str(args.out)}))
    else:
        for ln in g.records()[:5]:
            print(ln.text)
        for ln in g.generate(random.Random(args.seed), args.n):
            print(ln.text)


if __name__ == "__main__":
    main()
