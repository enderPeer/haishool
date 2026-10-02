"""Round 5, topic ``substances``: what the world is made of.

Worked curriculum: ``gate().generate_worked(rng, n)`` appends ``steps`` to molar-mass
and mass-percentage questions. Each answer shows the element counts times atomic
weights, their exact products and sum, then the rounded result (and percentage
formula/substitution where needed). Every segment must pass the gate. The separate
192-token budget requires a context of at least 192; use 256 to leave ample room.
The old answer-only generator and 80-token budget are unchanged.

A table of pure substances (air, water and ice, the ocean's salts, the crust's minerals, the
mantle and core, stars and interstellar ices, other planets, the molecules of life, industry and
everyday things, the species of the reactions table) and of mixtures (air, seawater, steel,
granite, wood, blood plasma, the sun, the crust, the human body, the universe ...). Only the
name, the formula, the class, the bond type, the state, where it occurs and the notes are curated
by hand. Everything numeric is *computed* from the formula and
``data/truth-v5/atomic_weights.json`` in exact decimal arithmetic: the elements by name with
their counts, the number of atoms, the number of different elements, the molar mass (5
significant digits) and the mass percent of each element (4 significant digits); a decimal tie
rounds up (126.115 -> 126.12), as in the elements table. A mixture carries its main components
with their approximate share in percent and its basis: ``mass`` or ``volume`` when the
components are substances, ``element_mass`` when they are elements whatever they are bound in
(the crust, the sun, the human body, steel).

Build the two data files once (mendeleev is only needed here; the gate reads only the json files):

    python -m haishool.truth.substances build            # atomic_weights.json, substances.jsonl
    python -m haishool.truth.substances sample --seed 3  # print a few lines of every kind

``atomic_weights.json`` maps every element symbol to its standard atomic weight (a float), or,
for the elements without one (technetium, promethium, polonium and beyond, except thorium,
protactinium and uranium), to a mass number (an integer): the source's bracket weight, unless
its isotope table shows another isotope to be longer-lived beyond doubt (meitnerium 278,
roentgenium 282, moscovium 290). This is the rule of the elements table, so the two agree.

Record lines (kind ``record``), one per table row; a row whose line would pass 80 tokens is split
into ``<name> substance.`` and ``<name> substance_more.`` at a field boundary:

    water substance. formula h 2 o 1. composition hydrogen 2 oxygen 1. atoms 3. elements 2.
      molar_mass 1 8 point 0 1 5. mass_percent hydrogen 1 1 point 1 9 oxygen 8 8 point 8 1.
      class oxide. bonds covalent. state liquid. occurs_in ocean rivers atmosphere ...
    cellulose substance. formula c 6 h 1 0 o 5. unit yes. composition ... (a polymer repeat unit)
    air mixture. main_parts nitrogen 7 8 point 1 oxygen 2 0 point 9 argon 0 point 9 3
      carbon_dioxide 0 point 0 4. basis volume. state gas. occurs_in atmosphere.

Question lines, mixed by :meth:`Substances.generate` in the target shares fact 45 % (facts per key
37, reverse lookups 6, lists 2), calc 35 % and yes/no 20 % (the finite kinds saturate when many
lines are asked for; calc and yes/no lines are unbounded):

    fact   q water formula. a h 2 o 1.                 q water molar_mass. a 1 8 point 0 1 5.
           q water composition. a hydrogen 2 oxygen 1.  q gold class. a metal.
           q bronze main_parts. a copper 8 8 tin 1 2.   q bronze basis. a mass.
           q air part oxygen. a 2 0 point 9.            q steel mixture. a yes.
           q cellulose unit. a yes.                     q earth_crust basis. a element_mass.
           q substance formula h 2 o 1. a water.        (reverse; ices and repeat units are left out)
           q substance formula c 6 h 1 2 o 6. a fructose galactose glucose.
           q substances class element state liquid. a bromine mercury.    (whole lists of at most 6 rows)
           q substances occurs_in dna. a adenine cytosine deoxyribose guanine thymidine thymine.
           q substances class ice. a ammonia_ice carbon_dioxide_ice methane_ice water_ice.
    calc   q molar_mass c 6 h 1 2 o 6. a 1 8 0 point 1 6.       (any formula, also random ones)
           q atoms c 6 h 1 2 o 6. a 2 4.
           q mass_percent water oxygen. a 8 8 point 8 1.          (by name or by formula)
           q element_count water hydrogen. a 2.         q element_count water carbon. a 0.
           q heavier_molecule water methane. a water.   (molecules only)
    yesno  q check water formula h 2 o 2. a no.         q check gold class metal. a yes.
           q check gold class element. a yes.           q check glucose class salt. a no.
           q check diamond bonds covalent. a yes.       q check water occurs_in ocean. a yes.
           q check glucose occurs_in sun. a no.         q check steel main_parts nickel. a no.
           q check air part oxygen 2 0 point 9. a yes.  q check air basis mass. a no.
           q check element_count water hydrogen 2. a yes.
           q check mass_percent water oxygen 8 8 point 8 1. a yes.
           q check heavier_molecule methane water. a no.

The keys ``composition``, ``occurs_in`` and ``main_parts`` are named so that no prompt of this
topic is a prompt of rounds 1-4 (``made_of`` and ``found_in`` are relations there, ``parts`` is
an object attribute, and ``q gold found_in`` already has another answer). Element names are
spelt as the elements table and rounds 1-4 spell them (``aluminum``, ``cesium``, ``sulfur``).

What the hand-made columns mean, and what a yes/no line may say about them:

* ``class`` is one label per row, the narrowest that fits, but classes overlap: gold is a
  ``metal`` and also an element, quartz a ``mineral`` and also an oxide, glucose a ``sugar``
  and also organic. :func:`class_truth` knows three answers: yes (the label, or a class that
  follows from the formula), no (a class the formula or the bonds rule out) and undecided. A
  ``no`` line is only ever written for a definite no; an undecided claim gets no verdict.
* ``bonds``: ``metallic`` for metals, ``ionic`` for salts, hydroxides and most oxides,
  ``covalent`` for molecular substances (gases, liquids, sugars and all organic molecules),
  ``network`` for covalent networks (diamond, graphite, quartz and every silicate, carbides,
  nitrides, zinc blende sulfides) and ``none`` for the noble gases and free atoms. A network is
  covalent too; between ionic and network many solids are a matter of judgement (sulfides, the
  silicates with their metal ions), so :func:`bond_truth` is undecided there.
* ``state`` is the one at 293 K and 1 atm; an ice (class ``ice``) is the frozen form of a
  substance and is listed solid. A row within a few kelvin of 293 K (:data:`STATE_DOUBT`) keeps
  its value but gets no ``no`` line.
* ``occurs_in`` is an open list from a fixed vocabulary of places. The build completes it by
  two rules: a substance occurs wherever a mixture that holds it occurs, and blood and bones are
  in animals, dna is in cells, the sun is a star. A place that is not listed is undecided,
  except where :func:`impossible_places` rules it out (no sugar in the sun, no mineral in dna);
  a row with the name of an element (``iron``, ``oxygen``) has no such place, because the
  mixtures given by element use the same names.
* ``main_parts`` are the main components only. Another element is not a main part of a mixture
  given by element; another substance is not a main part when it shares no place with the
  mixture; everything else is undecided.

A list (``q substances ...``) names the rows of this table and is only asked, and only answered,
when it is whole: at most 6 members and no row left undecided. A list over a place is whole when
the place is closed (:data:`CLOSED_PLACES`: dna, whose parts are known) or when the other
condition settles every row (all four ices occur in comets).

Numbers are computed, so an answer must have exactly the digits of the table (trailing zeros
aside); counts compare exactly; formulas compare by their atom counts (each element once, no
zero counts), lists as sets without repeats. A prompt of the topic that the table cannot answer
(an unknown formula in a reverse lookup, a list of more than 6 members, a metal in
``heavier_molecule``, ``check molar_mass`` followed by a formula, where the claimed number
cannot be told from the last count) gets a failing verdict with no expected value and the
reason. :meth:`Substances.owns` claims a name only together with one of this topic's keys,
because many names are shared with other topics (``iron``, ``oxygen``, ``water``):
``iron state`` is owned here (and agrees with the elements table), ``iron protons`` and
``water color`` are not.

Beside the :class:`~haishool.truth.Gate` methods there is :meth:`Substances.facts`, every table
fact exactly once, for a build that must not leave the coverage of the table to chance.

Nothing in this module comes from a teacher model: the formulas are standard, where a
representative formula stands for a family (biotite, hornblende, brownmillerite, paraffin wax)
or a value is approximate or from memory (every mixture, a few places) the row's ``notes`` say so.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import warnings
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path

from haishool.truth import Line, Verdict, dedupe, num, parse_num
from haishool.truth.formula import dense, parse
from haishool.truth.worked import check_steps, decimal_tokens

DATA = Path(__file__).resolve().parents[2] / "data" / "truth-v5"
TOPIC = "substances"
MAX_TOKENS = 80
# Element subtotals and exact summation need more context than answer-only lessons.
# A context of 256 keeps every generated prompt and its complete worked answer.
WORKED_MAX_TOKENS = 192
#: an answer lists at most this many substances
MAX_LIST = 6
#: a count in a dense formula has at most this many digits (longer ones are not formulas)
MAX_COUNT_DIGITS = 6

#: symbol and name of every element in order of atomic number; the build checks it against
#: mendeleev (the names are the tokens of the elements table: aluminum, cesium, sulfur)
ELEMENTS: tuple[tuple[str, str], ...] = (
    ("H", "hydrogen"), ("He", "helium"), ("Li", "lithium"), ("Be", "beryllium"), ("B", "boron"),
    ("C", "carbon"), ("N", "nitrogen"), ("O", "oxygen"), ("F", "fluorine"), ("Ne", "neon"),
    ("Na", "sodium"), ("Mg", "magnesium"), ("Al", "aluminum"), ("Si", "silicon"),
    ("P", "phosphorus"), ("S", "sulfur"), ("Cl", "chlorine"), ("Ar", "argon"), ("K", "potassium"),
    ("Ca", "calcium"), ("Sc", "scandium"), ("Ti", "titanium"), ("V", "vanadium"),
    ("Cr", "chromium"), ("Mn", "manganese"), ("Fe", "iron"), ("Co", "cobalt"), ("Ni", "nickel"),
    ("Cu", "copper"), ("Zn", "zinc"), ("Ga", "gallium"), ("Ge", "germanium"), ("As", "arsenic"),
    ("Se", "selenium"), ("Br", "bromine"), ("Kr", "krypton"), ("Rb", "rubidium"),
    ("Sr", "strontium"), ("Y", "yttrium"), ("Zr", "zirconium"), ("Nb", "niobium"),
    ("Mo", "molybdenum"), ("Tc", "technetium"), ("Ru", "ruthenium"), ("Rh", "rhodium"),
    ("Pd", "palladium"), ("Ag", "silver"), ("Cd", "cadmium"), ("In", "indium"), ("Sn", "tin"),
    ("Sb", "antimony"), ("Te", "tellurium"), ("I", "iodine"), ("Xe", "xenon"), ("Cs", "cesium"),
    ("Ba", "barium"), ("La", "lanthanum"), ("Ce", "cerium"), ("Pr", "praseodymium"),
    ("Nd", "neodymium"), ("Pm", "promethium"), ("Sm", "samarium"), ("Eu", "europium"),
    ("Gd", "gadolinium"), ("Tb", "terbium"), ("Dy", "dysprosium"), ("Ho", "holmium"),
    ("Er", "erbium"), ("Tm", "thulium"), ("Yb", "ytterbium"), ("Lu", "lutetium"),
    ("Hf", "hafnium"), ("Ta", "tantalum"), ("W", "tungsten"), ("Re", "rhenium"), ("Os", "osmium"),
    ("Ir", "iridium"), ("Pt", "platinum"), ("Au", "gold"), ("Hg", "mercury"), ("Tl", "thallium"),
    ("Pb", "lead"), ("Bi", "bismuth"), ("Po", "polonium"), ("At", "astatine"), ("Rn", "radon"),
    ("Fr", "francium"), ("Ra", "radium"), ("Ac", "actinium"), ("Th", "thorium"),
    ("Pa", "protactinium"), ("U", "uranium"), ("Np", "neptunium"), ("Pu", "plutonium"),
    ("Am", "americium"), ("Cm", "curium"), ("Bk", "berkelium"), ("Cf", "californium"),
    ("Es", "einsteinium"), ("Fm", "fermium"), ("Md", "mendelevium"), ("No", "nobelium"),
    ("Lr", "lawrencium"), ("Rf", "rutherfordium"), ("Db", "dubnium"), ("Sg", "seaborgium"),
    ("Bh", "bohrium"), ("Hs", "hassium"), ("Mt", "meitnerium"), ("Ds", "darmstadtium"),
    ("Rg", "roentgenium"), ("Cn", "copernicium"), ("Nh", "nihonium"), ("Fl", "flerovium"),
    ("Mc", "moscovium"), ("Lv", "livermorium"), ("Ts", "tennessine"), ("Og", "oganesson"),
)
NAME_OF = {sym: name for sym, name in ELEMENTS}
SYMBOL_OF = {name: sym for sym, name in ELEMENTS}
#: the elements that are neither metals nor metalloids, and the metalloids (for the bond and
#: class rules; every other element counts as a metal there)
NONMETALS = frozenset("H He C N O F Ne P S Cl Ar Se Br Kr I Xe At Rn".split())
METALLOIDS = frozenset("B Si Ge As Sb Te".split())
#: the cations of the plainly ionic compounds, and their anions
STRONG_CATIONS = frozenset("Li Na K Rb Cs Mg Ca Sr Ba".split())
HALOGENS = frozenset("F Cl Br I".split())


def has_standard_weight(z: int) -> bool:
    """IUPAC gives a standard atomic weight for the elements up to bismuth (83) except technetium
    and promethium, and for thorium, protactinium and uranium."""
    return (z <= 83 and z not in (43, 61)) or z in (90, 91, 92)


CLASSES = ("element", "metal", "mineral", "salt", "acid", "base", "oxide", "hydrocarbon", "alcohol",
           "sugar", "amino_acid", "nucleobase", "lipid", "polymer_unit", "organic", "inorganic", "drug",
           "explosive", "ceramic", "semiconductor", "ice")
#: the classes whose every member is an organic compound
ORGANIC_CLASSES = frozenset({"hydrocarbon", "alcohol", "sugar", "amino_acid", "nucleobase", "lipid", "drug",
                             "polymer_unit", "organic"})
BONDS = ("ionic", "covalent", "metallic", "network", "none")
STATES = ("solid", "liquid", "gas")
BASES = ("mass", "volume", "element_mass")
PLACES = ("atmosphere", "ocean", "rivers", "crust", "mantle", "core", "sun", "stars", "interstellar_clouds",
          "comets", "mars", "venus", "jupiter", "titan", "cells", "plants", "animals", "blood", "bones", "dna",
          "food", "fuel", "industry", "buildings", "batteries", "medicine")
#: a place that lies inside another: what occurs in the first occurs in the second
INSIDE = {"blood": "animals", "bones": "animals", "dna": "cells", "sun": "stars"}
#: the places whose rows the table lists completely, so a list over them is whole (dna is made of
#: deoxyribose, phosphate and four bases; every other place is open)
CLOSED_PLACES = ("dna",)
#: mixture components that are materials without one formula (allowed beside the table's names),
#: each with the classes of the rows that belong to it
MATERIALS: dict[str, tuple[str, ...]] = {
    "proteins": ("amino_acid",), "collagen": ("amino_acid",), "fats": ("lipid",),
    "salts": ("salt", "mineral"), "minerals": ("salt", "mineral", "oxide"), "sugars": ("sugar",),
    "carbohydrates": ("sugar", "polymer_unit"), "hemicellulose": ("sugar", "polymer_unit"), "lignin": (),
    "extractives": ("organic", "lipid", "hydrocarbon", "acid", "alcohol", "drug"),
    "sand": ("mineral", "ceramic", "oxide"), "gravel": ("mineral",),
}

KEYS: dict[str, str] = {
    "steps": "append to molar_mass or mass_percent: exact element products and sum, then rounded result",
    "formula": "the formula in dense form: each element symbol followed by its count in digit tokens",
    "unit": "yes when the formula is the repeat unit of a polymer (only written when yes)",
    "composition": "the elements by name, each with its atoms per formula unit",
    "atoms": "atoms per formula unit (count); as a question head: the atoms of any dense formula",
    "elements": "number of different elements in the formula (count)",
    "molar_mass": "molar mass in g per mol, 5 significant digits, from atomic_weights.json; as a question head: "
                  "of any dense formula",
    "mass_percent": "share of the molar mass that each element contributes, in percent, 4 significant digits; "
                    "as a question head: of one element in a substance or a dense formula",
    "class": "kind of substance, the narrowest label that fits: " + " ".join(CLASSES),
    "bonds": "bond type that holds the substance together: " + " ".join(BONDS),
    "state": "state of matter at 293 k and 1 atm (an ice is listed solid)",
    "occurs_in": "where it occurs, an open list from the fixed vocabulary: " + " ".join(PLACES),
    "mixture": "yes for a mixture of several substances, no for a pure substance",
    "main_parts": "(mixture) the main components with their approximate share in percent, 3 significant digits",
    "part": "(mixture) the share of one main component in percent",
    "basis": "(mixture) what the shares are percent of: mass or volume of the component substances, or "
             "element_mass, the mass of each element whatever it is bound in",
    "element_count": "(question head) atoms of one element per formula unit of a substance or a dense formula",
    "heavier_molecule": "(question head) which of two molecular substances has the larger molar mass",
    "substance": "the record of a pure row; as a question head (substance formula ...): the rows with that formula",
    "substance_more": "the second record line of a pure row",
    "substances": "(question head) the rows that share a class, bond type, state or place",
}
PURE_KEYS = ("formula", "unit", "composition", "atoms", "elements", "molar_mass", "mass_percent", "class", "bonds",
             "state", "occurs_in", "mixture")
MIX_KEYS = ("main_parts", "basis", "state", "occurs_in", "mixture")
COUNT_KEYS = ("atoms", "elements")
LIST_KEYS = ("class", "bonds", "state", "occurs_in")
CALC_HEADS = ("molar_mass", "atoms", "mass_percent", "element_count", "heavier_molecule")
#: elements a random formula outside the organic pattern is drawn from
COMMON = ("H", "Li", "B", "C", "N", "O", "F", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "K", "Ca", "Ti", "Cr", "Mn",
          "Fe", "Co", "Ni", "Cu", "Zn", "Br", "Ag", "Sn", "I", "Ba", "W", "Au", "Pb", "U")

#: rows whose formula is a repeat unit
UNITS = frozenset({"cellulose", "starch", "glycogen", "chitin", "polyethylene", "polypropylene",
                   "polyvinyl_chloride", "polystyrene", "polyethylene_terephthalate", "ptfe", "nylon_6",
                   "polydimethylsiloxane"})
#: rows whose state is right by the 293 K rule but not beyond doubt: they melt or boil within a
#: few kelvin of 293 K (which ones is from memory), the name covers a liquid and a solid form, or
#: a mixture is partly one and partly the other; no ``no`` line about their state
STATE_DOUBT = frozenset({"nitrogen_dioxide", "dinitrogen_tetroxide", "acetaldehyde", "hydrogen_fluoride",
                         "hydrogen_cyanide", "sulfur_trioxide", "hexadecane", "glycerol", "acetic_acid", "oleic_acid",
                         "nitroglycerin", "lactic_acid", "polydimethylsiloxane", "human_body", "earth_core",
                         "bulk_earth", "lpg"})
#: molecules that carry a plus and a minus end (inner salts), beside the amino acids: whether
#: they are ionic, or salts, is left open
ZWITTERIONS = frozenset({"nad", "gaba", "creatine"})
#: rows whose bond type is a judgement: a metalloid, interstitial carbides and nitrides, a halide
#: that is nearly covalent; no ``no`` line about them
BOND_DOUBT = frozenset({"antimony", "cementite", "tungsten_carbide", "titanium_nitride", "aluminum_chloride"})
#: rows for which a class beside the label is a judgement: hydrogen cyanide is filed organic and
#: often called inorganic, some hydroxides react as acids too; no ``no`` line about these classes
CLASS_DOUBT: dict[str, tuple[str, ...]] = {
    "hydrogen_cyanide": ("inorganic",), "aluminum_hydroxide": ("acid",), "copper_hydroxide": ("acid",),
    "iron_hydroxide": ("acid",),
}
#: elements that may well be a main part of a mixture given by element although the row does not
#: list them (the share is close to the smallest listed one, or uncertain)
PART_DOUBT: dict[str, tuple[str, ...]] = {
    "steel": ("silicon",), "cast_iron": ("manganese", "phosphorus"), "earth_core": ("carbon", "hydrogen"),
}

#: (name, formula, class, bonds, state at 293 K, occurs_in, notes). Organic formulas are in Hill
#: order (C, H, then alphabetical); minerals and salts in their conventional order.
PURE: list[tuple[str, str, str, str, str, str, str]] = [
    # the atmosphere
    ("nitrogen", "N2", "element", "covalent", "gas", "atmosphere ocean titan mars venus industry", ""),
    ("oxygen", "O2", "element", "covalent", "gas", "atmosphere ocean rivers blood cells industry medicine", ""),
    ("argon", "Ar", "element", "none", "gas", "atmosphere mars industry", "monatomic"),
    ("carbon_dioxide", "CO2", "oxide", "covalent", "gas", "atmosphere ocean mars venus comets cells blood food", ""),
    ("water", "H2O", "oxide", "covalent", "liquid", "ocean rivers atmosphere comets mars cells blood plants animals food",
     "vapour in the atmosphere"),
    ("ozone", "O3", "element", "covalent", "gas", "atmosphere industry", ""),
    ("methane", "CH4", "hydrocarbon", "covalent", "gas", "atmosphere titan jupiter comets interstellar_clouds fuel", ""),
    ("neon", "Ne", "element", "none", "gas", "atmosphere stars industry", "monatomic"),
    ("helium", "He", "element", "none", "gas", "atmosphere sun stars jupiter industry", "monatomic"),
    ("hydrogen", "H2", "element", "covalent", "gas", "atmosphere jupiter interstellar_clouds fuel industry",
     "molecular hydrogen; the hydrogen of the sun and the stars is atomic_hydrogen"),
    ("nitrous_oxide", "N2O", "oxide", "covalent", "gas", "atmosphere medicine", ""),
    ("krypton", "Kr", "element", "none", "gas", "atmosphere industry", "monatomic"),
    ("xenon", "Xe", "element", "none", "gas", "atmosphere industry medicine", "monatomic"),
    ("radon", "Rn", "element", "none", "gas", "atmosphere crust", "monatomic; radioactive, its weight is the mass number 222"),
    ("nitric_oxide", "NO", "oxide", "covalent", "gas", "atmosphere blood industry", ""),
    ("nitrogen_dioxide", "NO2", "oxide", "covalent", "liquid", "atmosphere industry",
     "boils at 294 k; a brown gas when thinned in air"),
    ("sulfur_dioxide", "SO2", "oxide", "covalent", "gas", "atmosphere venus industry", "volcanic gas"),
    ("carbon_monoxide", "CO", "oxide", "covalent", "gas", "atmosphere interstellar_clouds comets industry fuel", ""),
    ("atomic_hydrogen", "H", "element", "none", "gas", "sun stars interstellar_clouds",
     "free atoms; they pair up to hydrogen below a few thousand kelvin"),
    ("water_ice", "H2O", "ice", "covalent", "solid", "comets mars interstellar_clouds atmosphere", "frozen water"),
    ("carbon_dioxide_ice", "CO2", "ice", "covalent", "solid", "mars comets interstellar_clouds", "dry ice; the polar caps of mars"),
    ("ammonia_ice", "NH3", "ice", "covalent", "solid", "jupiter comets interstellar_clouds", "a cloud layer of jupiter"),
    ("methane_ice", "CH4", "ice", "covalent", "solid", "comets interstellar_clouds", ""),
    # the ocean
    ("sodium_chloride", "NaCl", "salt", "ionic", "solid", "ocean crust food blood medicine", ""),
    ("magnesium_chloride", "MgCl2", "salt", "ionic", "solid", "ocean industry", ""),
    ("magnesium_sulfate", "MgSO4", "salt", "ionic", "solid", "ocean medicine industry", ""),
    ("calcium_carbonate", "CaCO3", "salt", "ionic", "solid", "ocean crust animals buildings medicine", "shells and limestone"),
    ("calcium_sulfate", "CaSO4", "salt", "ionic", "solid", "ocean crust buildings industry", ""),
    ("calcium_chloride", "CaCl2", "salt", "ionic", "solid", "ocean industry food", ""),
    ("potassium_chloride", "KCl", "salt", "ionic", "solid", "ocean crust food medicine", ""),
    ("sodium_sulfate", "Na2SO4", "salt", "ionic", "solid", "ocean industry", ""),
    ("sodium_bromide", "NaBr", "salt", "ionic", "solid", "ocean industry", ""),
    ("hydrogen_sulfide", "H2S", "inorganic", "covalent", "gas", "ocean crust jupiter industry",
     "volcanic gas; from decay in mud; a weak acid in water"),
    ("magnesium_hydroxide", "Mg(OH)2", "base", "ionic", "solid", "medicine industry",
     "milk of magnesia; made from seawater, not found in it"),
    # the crust's minerals
    ("quartz", "SiO2", "mineral", "network", "solid", "crust rivers buildings industry", ""),
    ("orthoclase", "KAlSi3O8", "mineral", "network", "solid", "crust", "potassium feldspar"),
    ("albite", "NaAlSi3O8", "mineral", "network", "solid", "crust", "sodium feldspar"),
    ("anorthite", "CaAl2Si2O8", "mineral", "network", "solid", "crust", "calcium feldspar"),
    ("muscovite", "KAl3Si3O10(OH)2", "mineral", "network", "solid", "crust", "white mica"),
    ("biotite", "KMg2FeAlSi3O10(OH)2", "mineral", "network", "solid", "crust",
     "representative composition between phlogopite and annite"),
    ("forsterite", "Mg2SiO4", "mineral", "network", "solid", "crust mantle mars comets", "magnesium olivine; also in comet dust"),
    ("fayalite", "Fe2SiO4", "mineral", "network", "solid", "crust mantle", "iron olivine"),
    ("enstatite", "MgSiO3", "mineral", "network", "solid", "crust mantle mars", "a pyroxene"),
    ("diopside", "CaMgSi2O6", "mineral", "network", "solid", "crust mantle", "a pyroxene"),
    ("hornblende", "Ca2Mg4Al2Si7O22(OH)2", "mineral", "network", "solid", "crust",
     "representative magnesio hornblende; an amphibole"),
    ("tremolite", "Ca2Mg5Si8O22(OH)2", "mineral", "network", "solid", "crust", "an amphibole"),
    ("calcite", "CaCO3", "mineral", "ionic", "solid", "crust ocean buildings animals", "limestone and marble"),
    ("aragonite", "CaCO3", "mineral", "ionic", "solid", "ocean animals crust", "shells and coral"),
    ("dolomite", "CaMg(CO3)2", "mineral", "ionic", "solid", "crust buildings", ""),
    ("gypsum", "CaSO4(H2O)2", "mineral", "ionic", "solid", "crust buildings industry", "plaster; two waters per formula unit"),
    ("anhydrite", "CaSO4", "mineral", "ionic", "solid", "crust", ""),
    ("halite", "NaCl", "mineral", "ionic", "solid", "crust ocean", "rock salt"),
    ("sylvite", "KCl", "mineral", "ionic", "solid", "crust", ""),
    ("hematite", "Fe2O3", "mineral", "ionic", "solid", "crust mars industry",
     "the mineral of iron_oxide; the colour of mars"),
    ("magnetite", "Fe3O4", "mineral", "ionic", "solid", "crust industry", ""),
    ("goethite", "FeO(OH)", "mineral", "ionic", "solid", "crust mars", "rust and ochre"),
    ("pyrite", "FeS2", "mineral", "ionic", "solid", "crust", "fools gold"),
    ("galena", "PbS", "mineral", "ionic", "solid", "crust industry", "lead ore"),
    ("sphalerite", "ZnS", "mineral", "network", "solid", "crust industry", "zinc ore"),
    ("chalcopyrite", "CuFeS2", "mineral", "network", "solid", "crust industry", "copper ore"),
    ("corundum", "Al2O3", "mineral", "ionic", "solid", "crust industry", "ruby and sapphire"),
    ("rutile", "TiO2", "mineral", "ionic", "solid", "crust industry", ""),
    ("zircon", "ZrSiO4", "mineral", "network", "solid", "crust", ""),
    ("apatite", "Ca5(PO4)3F", "mineral", "ionic", "solid", "crust industry", "fluorapatite; phosphate ore"),
    ("hydroxyapatite", "Ca5(PO4)3(OH)", "mineral", "ionic", "solid", "bones animals medicine", "the mineral of bone and teeth"),
    ("kaolinite", "Al2Si2O5(OH)4", "mineral", "network", "solid", "crust buildings industry", "clay"),
    ("talc", "Mg3Si4O10(OH)2", "mineral", "network", "solid", "crust industry medicine", ""),
    ("graphite", "C", "element", "network", "solid", "crust industry batteries", "layered carbon"),
    ("diamond", "C", "element", "network", "solid", "crust mantle industry", ""),
    ("gibbsite", "Al(OH)3", "mineral", "ionic", "solid", "crust industry", "bauxite; aluminium ore"),
    ("ilmenite", "FeTiO3", "mineral", "ionic", "solid", "crust industry", "titanium ore"),
    ("chromite", "FeCr2O4", "mineral", "ionic", "solid", "crust industry", "chromium ore"),
    ("cassiterite", "SnO2", "mineral", "ionic", "solid", "crust industry", "tin ore"),
    ("cinnabar", "HgS", "mineral", "ionic", "solid", "crust industry", "mercury ore"),
    ("fluorite", "CaF2", "mineral", "ionic", "solid", "crust industry", ""),
    ("barite", "BaSO4", "mineral", "ionic", "solid", "crust industry medicine", ""),
    ("malachite", "Cu2CO3(OH)2", "mineral", "ionic", "solid", "crust", ""),
    ("azurite", "Cu3(CO3)2(OH)2", "mineral", "ionic", "solid", "crust", ""),
    ("cuprite", "Cu2O", "mineral", "ionic", "solid", "crust", ""),
    ("chalcocite", "Cu2S", "mineral", "ionic", "solid", "crust industry", ""),
    ("bornite", "Cu5FeS4", "mineral", "network", "solid", "crust", ""),
    ("molybdenite", "MoS2", "mineral", "network", "solid", "crust industry", ""),
    ("scheelite", "CaWO4", "mineral", "ionic", "solid", "crust industry", "tungsten ore"),
    ("uraninite", "UO2", "mineral", "ionic", "solid", "crust industry", "uranium ore"),
    ("beryl", "Be3Al2Si6O18", "mineral", "network", "solid", "crust", "emerald and aquamarine"),
    ("topaz", "Al2SiO4F2", "mineral", "network", "solid", "crust", ""),
    ("pyrope", "Mg3Al2Si3O12", "mineral", "network", "solid", "crust mantle", "a garnet"),
    ("almandine", "Fe3Al2Si3O12", "mineral", "network", "solid", "crust", "a garnet"),
    ("spinel", "MgAl2O4", "mineral", "ionic", "solid", "crust mantle", ""),
    ("wollastonite", "CaSiO3", "mineral", "network", "solid", "crust industry", ""),
    ("serpentine", "Mg3Si2O5(OH)4", "mineral", "network", "solid", "crust", "lizardite"),
    ("nepheline", "NaAlSiO4", "mineral", "network", "solid", "crust", ""),
    ("leucite", "KAlSi2O6", "mineral", "network", "solid", "crust", ""),
    ("kyanite", "Al2SiO5", "mineral", "network", "solid", "crust", ""),
    ("magnesite", "MgCO3", "mineral", "ionic", "solid", "crust industry", ""),
    ("siderite", "FeCO3", "mineral", "ionic", "solid", "crust", ""),
    ("rhodochrosite", "MnCO3", "mineral", "ionic", "solid", "crust", ""),
    ("smithsonite", "ZnCO3", "mineral", "ionic", "solid", "crust", ""),
    ("cerussite", "PbCO3", "mineral", "ionic", "solid", "crust", ""),
    ("celestine", "SrSO4", "mineral", "ionic", "solid", "crust", ""),
    ("borax", "Na2B4O7(H2O)10", "mineral", "ionic", "solid", "crust industry", "ten waters per formula unit"),
    ("arsenopyrite", "FeAsS", "mineral", "network", "solid", "crust", ""),
    ("stibnite", "Sb2S3", "mineral", "network", "solid", "crust industry", ""),
    ("realgar", "As4S4", "mineral", "covalent", "solid", "crust", "molecular cages"),
    ("orpiment", "As2S3", "mineral", "network", "solid", "crust", ""),
    ("perovskite", "CaTiO3", "mineral", "ionic", "solid", "crust", ""),
    ("troilite", "FeS", "mineral", "ionic", "solid", "crust", "rare on earth; common in meteorites"),
    ("wustite", "FeO", "mineral", "ionic", "solid", "mantle industry", ""),
    ("stishovite", "SiO2", "mineral", "network", "solid", "mantle crust", "high pressure form of quartz"),
    ("coesite", "SiO2", "mineral", "network", "solid", "mantle crust", "high pressure form of quartz"),
    ("sulfur", "S8", "element", "covalent", "solid", "crust industry", "rings of eight atoms; around volcanoes"),
    ("white_phosphorus", "P4", "element", "covalent", "solid", "industry", ""),
    ("iodine", "I2", "element", "covalent", "solid", "medicine industry", "in the sea it is iodide, not the element"),
    ("bromine", "Br2", "element", "covalent", "liquid", "industry", "in the sea it is bromide, not the element"),
    ("chlorine", "Cl2", "element", "covalent", "gas", "industry", ""),
    ("fluorine", "F2", "element", "covalent", "gas", "industry", ""),
    ("boron", "B", "element", "network", "solid", "industry", ""),
    # mantle and core
    ("bridgmanite", "MgSiO3", "mineral", "network", "solid", "mantle", "the most abundant mineral of the earth"),
    ("post_perovskite", "MgSiO3", "mineral", "network", "solid", "mantle", "lowest mantle"),
    ("periclase", "MgO", "mineral", "ionic", "solid", "mantle industry", "ferropericlase holds iron too"),
    ("ringwoodite", "Mg2SiO4", "mineral", "network", "solid", "mantle", "transition zone"),
    ("wadsleyite", "Mg2SiO4", "mineral", "network", "solid", "mantle", "transition zone"),
    ("majorite", "Mg4Si4O12", "mineral", "network", "solid", "mantle", "a garnet"),
    ("davemaoite", "CaSiO3", "mineral", "network", "solid", "mantle", "calcium silicate perovskite"),
    ("iron", "Fe", "metal", "metallic", "solid", "core industry buildings", "as metal; in the crust it is bound in minerals"),
    ("nickel", "Ni", "metal", "metallic", "solid", "core industry batteries", ""),
    # stars and space
    ("ammonia", "NH3", "base", "covalent", "gas", "jupiter comets interstellar_clouds industry", ""),
    ("methanol", "CH4O", "alcohol", "covalent", "liquid", "interstellar_clouds comets industry fuel", ""),
    ("formaldehyde", "CH2O", "organic", "covalent", "gas", "interstellar_clouds comets industry", ""),
    ("hydrogen_cyanide", "CHN", "organic", "covalent", "liquid", "titan comets interstellar_clouds industry",
     "boils at 299 k; a gas in space and on titan"),
    ("acetylene", "C2H2", "hydrocarbon", "covalent", "gas", "titan jupiter interstellar_clouds industry", ""),
    ("naphthalene", "C10H8", "hydrocarbon", "covalent", "solid", "interstellar_clouds industry",
     "smallest polycyclic aromatic; this family is seen in space, naphthalene itself only tentatively"),
    ("silicon_monoxide", "SiO", "oxide", "network", "solid", "stars interstellar_clouds industry",
     "a gas around old stars; a solid on earth"),
    ("silicon_carbide", "SiC", "ceramic", "network", "solid", "stars industry", "dust from carbon stars"),
    ("cyanoacetylene", "C3HN", "organic", "covalent", "liquid", "titan interstellar_clouds", "boils at 316 k; a gas in space"),
    ("carbonyl_sulfide", "COS", "inorganic", "covalent", "gas", "venus atmosphere", ""),
    ("fullerene", "C60", "element", "covalent", "solid", "industry interstellar_clouds", "sixty carbon atoms in a ball"),
    # other planets
    ("sulfuric_acid", "H2SO4", "acid", "covalent", "liquid", "venus industry batteries", "the clouds of venus"),
    ("phosphine", "PH3", "inorganic", "covalent", "gas", "jupiter industry", ""),
    ("hydrogen_chloride", "HCl", "acid", "covalent", "gas", "venus industry",
     "the gas; the mixture hydrochloric_acid is its solution in water"),
    ("hydrogen_fluoride", "HF", "acid", "covalent", "gas", "venus industry", "boils at 293 k"),
    ("ammonium_hydrosulfide", "NH4SH", "salt", "ionic", "solid", "jupiter", "a cloud layer of jupiter"),
    ("magnesium_perchlorate", "Mg(ClO4)2", "salt", "ionic", "solid", "mars industry", "in martian soil"),
    # life: sugars and polysaccharides
    ("glucose", "C6H12O6", "sugar", "covalent", "solid", "cells blood plants food", ""),
    ("fructose", "C6H12O6", "sugar", "covalent", "solid", "plants food", "fruit sugar"),
    ("galactose", "C6H12O6", "sugar", "covalent", "solid", "animals food", "half of milk sugar"),
    ("sucrose", "C12H22O11", "sugar", "covalent", "solid", "plants food", "table sugar"),
    ("lactose", "C12H22O11", "sugar", "covalent", "solid", "animals food", "milk sugar"),
    ("maltose", "C12H22O11", "sugar", "covalent", "solid", "plants food", "malt sugar"),
    ("ribose", "C5H10O5", "sugar", "covalent", "solid", "cells", "the sugar of rna and atp"),
    ("deoxyribose", "C5H10O4", "sugar", "covalent", "solid", "dna cells", ""),
    ("cellulose", "C6H10O5", "polymer_unit", "covalent", "solid", "plants buildings food", "repeat unit"),
    ("starch", "C6H10O5", "polymer_unit", "covalent", "solid", "plants food", "repeat unit"),
    ("glycogen", "C6H10O5", "polymer_unit", "covalent", "solid", "animals cells", "repeat unit"),
    ("chitin", "C8H13NO5", "polymer_unit", "covalent", "solid", "animals", "repeat unit; shells of insects and crabs"),
    # life: lipids
    ("glycerol", "C3H8O3", "alcohol", "covalent", "liquid", "cells plants animals food", ""),
    ("palmitic_acid", "C16H32O2", "lipid", "covalent", "solid", "animals plants cells food", ""),
    ("stearic_acid", "C18H36O2", "lipid", "covalent", "solid", "animals food", ""),
    ("oleic_acid", "C18H34O2", "lipid", "covalent", "liquid", "plants animals food", ""),
    ("linoleic_acid", "C18H32O2", "lipid", "covalent", "liquid", "plants food", ""),
    ("linolenic_acid", "C18H30O2", "lipid", "covalent", "liquid", "plants food", "an omega 3 fatty acid"),
    ("cholesterol", "C27H46O", "lipid", "covalent", "solid", "animals cells blood", ""),
    ("triolein", "C57H104O6", "lipid", "covalent", "liquid", "plants food", "fat of three oleic acids on glycerol"),
    ("tristearin", "C57H110O6", "lipid", "covalent", "solid", "animals food", "fat of three stearic acids on glycerol"),
    ("tripalmitin", "C51H98O6", "lipid", "covalent", "solid", "animals plants food", "fat of three palmitic acids on glycerol"),
    ("squalene", "C30H50", "hydrocarbon", "covalent", "liquid", "animals plants cells", ""),
    # life: the twenty standard amino acids
    ("glycine", "C2H5NO2", "amino_acid", "covalent", "solid", "cells plants animals blood food", ""),
    ("alanine", "C3H7NO2", "amino_acid", "covalent", "solid", "cells plants animals blood food", ""),
    ("valine", "C5H11NO2", "amino_acid", "covalent", "solid", "cells plants animals blood food", ""),
    ("leucine", "C6H13NO2", "amino_acid", "covalent", "solid", "cells plants animals blood food", ""),
    ("isoleucine", "C6H13NO2", "amino_acid", "covalent", "solid", "cells plants animals blood food", ""),
    ("proline", "C5H9NO2", "amino_acid", "covalent", "solid", "cells plants animals blood food", ""),
    ("phenylalanine", "C9H11NO2", "amino_acid", "covalent", "solid", "cells plants animals blood food", ""),
    ("tryptophan", "C11H12N2O2", "amino_acid", "covalent", "solid", "cells plants animals blood food", ""),
    ("methionine", "C5H11NO2S", "amino_acid", "covalent", "solid", "cells plants animals blood food", ""),
    ("serine", "C3H7NO3", "amino_acid", "covalent", "solid", "cells plants animals blood food", ""),
    ("threonine", "C4H9NO3", "amino_acid", "covalent", "solid", "cells plants animals blood food", ""),
    ("cysteine", "C3H7NO2S", "amino_acid", "covalent", "solid", "cells plants animals blood food", ""),
    ("tyrosine", "C9H11NO3", "amino_acid", "covalent", "solid", "cells plants animals blood food", ""),
    ("asparagine", "C4H8N2O3", "amino_acid", "covalent", "solid", "cells plants animals blood food", ""),
    ("glutamine", "C5H10N2O3", "amino_acid", "covalent", "solid", "cells plants animals blood food", ""),
    ("aspartic_acid", "C4H7NO4", "amino_acid", "covalent", "solid", "cells plants animals blood food", ""),
    ("glutamic_acid", "C5H9NO4", "amino_acid", "covalent", "solid", "cells plants animals blood food", ""),
    ("lysine", "C6H14N2O2", "amino_acid", "covalent", "solid", "cells plants animals blood food", ""),
    ("arginine", "C6H14N4O2", "amino_acid", "covalent", "solid", "cells plants animals blood food", ""),
    ("histidine", "C6H9N3O2", "amino_acid", "covalent", "solid", "cells plants animals blood food", ""),
    # life: nucleobases and nucleotides
    ("adenine", "C5H5N5", "nucleobase", "covalent", "solid", "dna cells", ""),
    ("guanine", "C5H5N5O", "nucleobase", "covalent", "solid", "dna cells", ""),
    ("cytosine", "C4H5N3O", "nucleobase", "covalent", "solid", "dna cells", ""),
    ("thymine", "C5H6N2O2", "nucleobase", "covalent", "solid", "dna cells", ""),
    ("uracil", "C4H4N2O2", "nucleobase", "covalent", "solid", "cells", "in rna, not in dna"),
    ("atp", "C10H16N5O13P3", "organic", "covalent", "solid", "cells", "adenosine triphosphate"),
    ("adp", "C10H15N5O10P2", "organic", "covalent", "solid", "cells", "adenosine diphosphate"),
    ("amp", "C10H14N5O7P", "organic", "covalent", "solid", "cells", "adenosine monophosphate"),
    ("nadh", "C21H29N7O14P2", "organic", "covalent", "solid", "cells",
     "reduced nicotinamide adenine dinucleotide; formula from memory"),
    ("nad", "C21H27N7O14P2", "organic", "covalent", "solid", "cells",
     "oxidised nicotinamide adenine dinucleotide, written as the neutral inner salt; formula from memory"),
    ("fad", "C27H33N9O15P2", "organic", "covalent", "solid", "cells", "flavin adenine dinucleotide; formula from memory"),
    ("gtp", "C10H16N5O14P3", "organic", "covalent", "solid", "cells", "guanosine triphosphate; formula from memory"),
    ("acetyl_coa", "C23H38N7O17P3S", "organic", "covalent", "solid", "cells", "acetyl coenzyme a; formula from memory"),
    ("adenosine", "C10H13N5O4", "organic", "covalent", "solid", "cells medicine", "adenine on ribose"),
    ("guanosine", "C10H13N5O5", "organic", "covalent", "solid", "cells", "guanine on ribose"),
    ("cytidine", "C9H13N3O5", "organic", "covalent", "solid", "cells", "cytosine on ribose"),
    ("uridine", "C9H12N2O6", "organic", "covalent", "solid", "cells", "uracil on ribose"),
    ("thymidine", "C10H14N2O5", "organic", "covalent", "solid", "dna cells", "thymine on deoxyribose"),
    # life: metabolites, pigments, hormones, vitamins
    ("urea", "CH4N2O", "organic", "covalent", "solid", "animals blood industry", "also a fertiliser"),
    ("uric_acid", "C5H4N4O3", "organic", "covalent", "solid", "animals blood", ""),
    ("creatine", "C4H9N3O2", "organic", "covalent", "solid", "animals cells", ""),
    ("lactic_acid", "C3H6O3", "acid", "covalent", "solid", "cells blood food",
     "the l form of muscle melts at 326 k; the racemic mix near 290 k"),
    ("pyruvic_acid", "C3H4O3", "acid", "covalent", "liquid", "cells", ""),
    ("citric_acid", "C6H8O7", "acid", "covalent", "solid", "cells plants food", ""),
    ("acetic_acid", "C2H4O2", "acid", "covalent", "liquid", "food industry", "vinegar"),
    ("ethanol", "C2H6O", "alcohol", "covalent", "liquid", "food fuel industry medicine", ""),
    ("caffeine", "C8H10N4O2", "drug", "covalent", "solid", "plants food", ""),
    ("nicotine", "C10H14N2", "drug", "covalent", "liquid", "plants", ""),
    ("chlorophyll_a", "C55H72MgN4O5", "organic", "covalent", "solid", "plants cells", "the green pigment"),
    ("heme_b", "C34H32FeN4O4", "organic", "covalent", "solid", "blood animals cells",
     "the iron group of haemoglobin; a cofactor, not part of the protein chain"),
    ("vitamin_c", "C6H8O6", "organic", "covalent", "solid", "plants food medicine", "ascorbic acid"),
    ("vitamin_d3", "C27H44O", "lipid", "covalent", "solid", "animals food medicine", "cholecalciferol; a steroid"),
    ("retinol", "C20H30O", "organic", "covalent", "solid", "animals food", "vitamin a"),
    ("beta_carotene", "C40H56", "hydrocarbon", "covalent", "solid", "plants food", "the orange of carrots"),
    ("riboflavin", "C17H20N4O6", "organic", "covalent", "solid", "food cells", "vitamin b2"),
    ("niacin", "C6H5NO2", "organic", "covalent", "solid", "food cells", "vitamin b3"),
    ("folic_acid", "C19H19N7O6", "organic", "covalent", "solid", "food cells medicine", "vitamin b9"),
    ("vitamin_b12", "C63H88CoN14O14P", "organic", "covalent", "solid", "food cells medicine", "cyanocobalamin; formula from memory"),
    ("biotin", "C10H16N2O3S", "organic", "covalent", "solid", "food cells", "vitamin b7"),
    ("pyridoxine", "C8H11NO3", "organic", "covalent", "solid", "food cells medicine", "vitamin b6"),
    ("vitamin_e", "C29H50O2", "organic", "covalent", "liquid", "plants food medicine",
     "alpha tocopherol, a thick oil; formula from memory"),
    ("vitamin_k1", "C31H46O2", "organic", "covalent", "liquid", "plants food medicine",
     "phylloquinone, a thick oil; formula from memory"),
    ("dopamine", "C8H11NO2", "organic", "covalent", "solid", "animals cells blood medicine", ""),
    ("adrenaline", "C9H13NO3", "organic", "covalent", "solid", "animals blood medicine", ""),
    ("noradrenaline", "C8H11NO3", "organic", "covalent", "solid", "animals blood cells medicine", ""),
    ("serotonin", "C10H12N2O", "organic", "covalent", "solid", "animals cells blood", ""),
    ("melatonin", "C13H16N2O2", "organic", "covalent", "solid", "animals blood medicine", ""),
    ("histamine", "C5H9N3", "organic", "covalent", "solid", "animals cells", ""),
    ("gaba", "C4H9NO2", "organic", "covalent", "solid", "animals cells", "gamma aminobutyric acid"),
    ("thyroxine", "C15H11I4NO4", "organic", "covalent", "solid", "animals blood medicine", ""),
    ("testosterone", "C19H28O2", "lipid", "covalent", "solid", "animals blood medicine", "a steroid"),
    ("oestradiol", "C18H24O2", "lipid", "covalent", "solid", "animals blood medicine", "a steroid"),
    ("cortisol", "C21H30O5", "lipid", "covalent", "solid", "animals blood medicine", "a steroid"),
    ("progesterone", "C21H30O2", "lipid", "covalent", "solid", "animals blood medicine", "a steroid"),
    ("indole_3_acetic_acid", "C10H9NO2", "organic", "covalent", "solid", "plants", "auxin, the growth hormone of plants"),
    # drugs
    ("aspirin", "C9H8O4", "drug", "covalent", "solid", "medicine", ""),
    ("paracetamol", "C8H9NO2", "drug", "covalent", "solid", "medicine", ""),
    ("ibuprofen", "C13H18O2", "drug", "covalent", "solid", "medicine", ""),
    ("penicillin_g", "C16H18N2O4S", "drug", "covalent", "solid", "medicine", ""),
    ("amoxicillin", "C16H19N3O5S", "drug", "covalent", "solid", "medicine", ""),
    ("morphine", "C17H19NO3", "drug", "covalent", "solid", "plants medicine", ""),
    ("codeine", "C18H21NO3", "drug", "covalent", "solid", "plants medicine", ""),
    ("cocaine", "C17H21NO4", "drug", "covalent", "solid", "plants", ""),
    ("diazepam", "C16H13ClN2O", "drug", "covalent", "solid", "medicine", ""),
    ("metformin", "C4H11N5", "drug", "covalent", "solid", "medicine", ""),
    ("lidocaine", "C14H22N2O", "drug", "covalent", "solid", "medicine", ""),
    ("warfarin", "C19H16O4", "drug", "covalent", "solid", "medicine", ""),
    ("atorvastatin", "C33H35FN2O5", "drug", "covalent", "solid", "medicine", ""),
    ("omeprazole", "C17H19N3O3S", "drug", "covalent", "solid", "medicine", ""),
    ("chloroquine", "C18H26ClN3", "drug", "covalent", "solid", "medicine", ""),
    ("quinine", "C20H24N2O2", "drug", "covalent", "solid", "plants medicine", ""),
    ("theobromine", "C7H8N4O2", "drug", "covalent", "solid", "plants food", "in cocoa"),
    # food and flavour
    ("monosodium_glutamate", "C5H8NNaO4", "salt", "ionic", "solid", "food", ""),
    ("vanillin", "C8H8O3", "organic", "covalent", "solid", "plants food", ""),
    ("capsaicin", "C18H27NO3", "organic", "covalent", "solid", "plants food", "the heat of chillies"),
    ("menthol", "C10H20O", "organic", "covalent", "solid", "plants food medicine", ""),
    ("limonene", "C10H16", "hydrocarbon", "covalent", "liquid", "plants food", "the smell of oranges"),
    ("aspartame", "C14H18N2O5", "organic", "covalent", "solid", "food", ""),
    ("sucralose", "C12H19Cl3O8", "organic", "covalent", "solid", "food", ""),
    ("saccharin", "C7H5NO3S", "organic", "covalent", "solid", "food", ""),
    ("tartaric_acid", "C4H6O6", "acid", "covalent", "solid", "plants food", ""),
    ("malic_acid", "C4H6O5", "acid", "covalent", "solid", "plants food", ""),
    ("oxalic_acid", "C2H2O4", "acid", "covalent", "solid", "plants", ""),
    ("benzoic_acid", "C7H6O2", "acid", "covalent", "solid", "plants food industry", ""),
    ("salicylic_acid", "C7H6O3", "acid", "covalent", "solid", "plants medicine", ""),
    ("formic_acid", "CH2O2", "acid", "covalent", "liquid", "animals industry", "the sting of ants"),
    ("calcium_phosphate", "Ca3(PO4)2", "salt", "ionic", "solid", "bones food industry", ""),
    ("sodium_nitrite", "NaNO2", "salt", "ionic", "solid", "food industry", "cures meat"),
    # industry: acids, bases, salts, oxides
    ("nitric_acid", "HNO3", "acid", "covalent", "liquid", "industry", ""),
    ("phosphoric_acid", "H3PO4", "acid", "covalent", "solid", "industry food", "melts at 315 k"),
    ("sodium_hydroxide", "NaOH", "base", "ionic", "solid", "industry", ""),
    ("potassium_hydroxide", "KOH", "base", "ionic", "solid", "industry batteries", ""),
    ("sodium_carbonate", "Na2CO3", "salt", "ionic", "solid", "industry", "soda"),
    ("sodium_bicarbonate", "NaHCO3", "salt", "ionic", "solid", "food blood medicine industry", "baking soda"),
    ("calcium_oxide", "CaO", "oxide", "ionic", "solid", "industry buildings", "quicklime"),
    ("calcium_hydroxide", "Ca(OH)2", "base", "ionic", "solid", "buildings industry", "slaked lime"),
    ("alite", "Ca3SiO5", "ceramic", "network", "solid", "buildings industry", "the main phase of cement"),
    ("belite", "Ca2SiO4", "ceramic", "network", "solid", "buildings industry", "a phase of cement"),
    ("tricalcium_aluminate", "Ca3Al2O6", "ceramic", "ionic", "solid", "buildings industry", "a phase of cement"),
    ("brownmillerite", "Ca2AlFeO5", "ceramic", "ionic", "solid", "buildings industry", "a phase of cement; representative formula"),
    ("sodium_hypochlorite", "NaClO", "salt", "ionic", "solid", "industry medicine", "bleach; used as a solution in water"),
    ("hydrogen_peroxide", "H2O2", "oxide", "covalent", "liquid", "medicine industry cells", "a peroxide"),
    ("sulfur_trioxide", "SO3", "oxide", "covalent", "liquid", "industry", "melts at 290 k"),
    ("cfc_12", "CCl2F2", "organic", "covalent", "gas", "atmosphere industry", "dichlorodifluoromethane"),
    ("ammonium_nitrate", "NH4NO3", "salt", "ionic", "solid", "industry", "fertiliser and explosive"),
    ("potassium_nitrate", "KNO3", "salt", "ionic", "solid", "industry food crust", "saltpetre"),
    ("ammonium_chloride", "NH4Cl", "salt", "ionic", "solid", "batteries industry", ""),
    ("ammonium_sulfate", "(NH4)2SO4", "salt", "ionic", "solid", "industry", "fertiliser"),
    ("sodium_nitrate", "NaNO3", "salt", "ionic", "solid", "crust industry", ""),
    ("potassium_carbonate", "K2CO3", "salt", "ionic", "solid", "industry", "potash"),
    ("lithium_carbonate", "Li2CO3", "salt", "ionic", "solid", "medicine batteries industry", ""),
    ("sodium_silicate", "Na2SiO3", "salt", "network", "solid", "industry", "water glass"),
    ("sodium_fluoride", "NaF", "salt", "ionic", "solid", "medicine industry", "in toothpaste"),
    ("copper_sulfate", "CuSO4", "salt", "ionic", "solid", "industry", ""),
    ("silver_chloride", "AgCl", "salt", "ionic", "solid", "industry", "photographic film"),
    ("sodium_oxide", "Na2O", "oxide", "ionic", "solid", "industry buildings", "in glass"),
    ("magnesium_oxide", "MgO", "oxide", "ionic", "solid", "industry medicine", ""),
    ("titanium_dioxide", "TiO2", "oxide", "ionic", "solid", "industry medicine", "white pigment; sunscreen"),
    ("zinc_oxide", "ZnO", "oxide", "ionic", "solid", "industry medicine", ""),
    ("aluminum_oxide", "Al2O3", "ceramic", "ionic", "solid", "industry", "alumina; the compound of the mineral corundum"),
    ("manganese_dioxide", "MnO2", "oxide", "ionic", "solid", "batteries crust industry", ""),
    ("lead_dioxide", "PbO2", "oxide", "ionic", "solid", "batteries", ""),
    ("nickel_hydroxide", "Ni(OH)2", "base", "ionic", "solid", "batteries", ""),
    ("lithium_cobalt_oxide", "LiCoO2", "oxide", "ionic", "solid", "batteries", ""),
    ("lithium_iron_phosphate", "LiFePO4", "salt", "ionic", "solid", "batteries", ""),
    ("lithium_hexafluorophosphate", "LiPF6", "salt", "ionic", "solid", "batteries", ""),
    ("uranium_dioxide", "UO2", "oxide", "ionic", "solid", "fuel industry", "nuclear fuel"),
    ("plutonium_dioxide", "PuO2", "oxide", "ionic", "solid", "fuel industry",
     "nuclear fuel; plutonium has no standard weight: 244 is its longest-lived isotope, reactor plutonium is mostly 239"),
    # industry: polymers
    ("polyethylene", "C2H4", "polymer_unit", "covalent", "solid", "industry buildings", "repeat unit"),
    ("polypropylene", "C3H6", "polymer_unit", "covalent", "solid", "industry", "repeat unit"),
    ("polyvinyl_chloride", "C2H3Cl", "polymer_unit", "covalent", "solid", "buildings industry", "repeat unit"),
    ("polystyrene", "C8H8", "polymer_unit", "covalent", "solid", "industry buildings", "repeat unit"),
    ("polyethylene_terephthalate", "C10H8O4", "polymer_unit", "covalent", "solid", "industry", "repeat unit; bottles and fibres"),
    ("ptfe", "C2F4", "polymer_unit", "covalent", "solid", "industry", "repeat unit; polytetrafluoroethylene"),
    ("nylon_6", "C6H11NO", "polymer_unit", "covalent", "solid", "industry", "repeat unit"),
    ("polydimethylsiloxane", "C2H6OSi", "polymer_unit", "covalent", "solid", "industry medicine",
     "repeat unit; silicone: a rubber when cross linked, an oil when not"),
    # industry: explosives
    ("tnt", "C7H5N3O6", "explosive", "covalent", "solid", "industry", "trinitrotoluene"),
    ("nitroglycerin", "C3H5N3O9", "explosive", "covalent", "liquid", "industry medicine", ""),
    ("rdx", "C3H6N6O6", "explosive", "covalent", "solid", "industry", ""),
    ("petn", "C5H8N4O12", "explosive", "covalent", "solid", "industry medicine", ""),
    ("picric_acid", "C6H3N3O7", "explosive", "covalent", "solid", "industry", ""),
    ("lead_azide", "Pb(N3)2", "explosive", "ionic", "solid", "industry", ""),
    # industry: hydrocarbons and solvents
    ("ethane", "C2H6", "hydrocarbon", "covalent", "gas", "titan fuel industry", ""),
    ("propane", "C3H8", "hydrocarbon", "covalent", "gas", "fuel industry", ""),
    ("butane", "C4H10", "hydrocarbon", "covalent", "gas", "fuel industry", ""),
    ("pentane", "C5H12", "hydrocarbon", "covalent", "liquid", "fuel industry", ""),
    ("hexane", "C6H14", "hydrocarbon", "covalent", "liquid", "fuel industry", ""),
    ("heptane", "C7H16", "hydrocarbon", "covalent", "liquid", "fuel", ""),
    ("octane", "C8H18", "hydrocarbon", "covalent", "liquid", "fuel", ""),
    ("dodecane", "C12H26", "hydrocarbon", "covalent", "liquid", "fuel", "in kerosene and diesel"),
    ("hexadecane", "C16H34", "hydrocarbon", "covalent", "liquid", "fuel", "cetane; melts at 291 k"),
    ("cyclohexane", "C6H12", "hydrocarbon", "covalent", "liquid", "industry", ""),
    ("ethylene", "C2H4", "hydrocarbon", "covalent", "gas", "industry plants", "also a plant hormone"),
    ("propylene", "C3H6", "hydrocarbon", "covalent", "gas", "industry", ""),
    ("butadiene", "C4H6", "hydrocarbon", "covalent", "gas", "industry", ""),
    ("isoprene", "C5H8", "hydrocarbon", "covalent", "liquid", "plants atmosphere industry",
     "given off by trees; the unit of natural rubber; boils at 307 k"),
    ("benzene", "C6H6", "hydrocarbon", "covalent", "liquid", "industry fuel titan", ""),
    ("toluene", "C7H8", "hydrocarbon", "covalent", "liquid", "industry fuel", ""),
    ("xylene", "C8H10", "hydrocarbon", "covalent", "liquid", "industry fuel", ""),
    ("styrene", "C8H8", "hydrocarbon", "covalent", "liquid", "industry", ""),
    ("phenol", "C6H6O", "organic", "covalent", "solid", "industry", ""),
    ("aniline", "C6H7N", "organic", "covalent", "liquid", "industry", ""),
    ("acetone", "C3H6O", "organic", "covalent", "liquid", "industry blood", "a ketone body in blood"),
    ("acetaldehyde", "C2H4O", "organic", "covalent", "liquid", "industry", "boils at 293 k"),
    ("ethylene_glycol", "C2H6O2", "alcohol", "covalent", "liquid", "industry", "antifreeze"),
    ("isopropanol", "C3H8O", "alcohol", "covalent", "liquid", "industry medicine", ""),
    ("ethyl_acetate", "C4H8O2", "organic", "covalent", "liquid", "industry food", ""),
    ("diethyl_ether", "C4H10O", "organic", "covalent", "liquid", "industry medicine", ""),
    ("chloroform", "CHCl3", "organic", "covalent", "liquid", "industry", ""),
    ("dichloromethane", "CH2Cl2", "organic", "covalent", "liquid", "industry", ""),
    ("carbon_tetrachloride", "CCl4", "organic", "covalent", "liquid", "industry", ""),
    # industry: semiconductors and ceramics
    ("silicon", "Si", "semiconductor", "network", "solid", "industry", "chips and solar cells"),
    ("germanium", "Ge", "semiconductor", "network", "solid", "industry", ""),
    ("gallium_arsenide", "GaAs", "semiconductor", "network", "solid", "industry", ""),
    ("gallium_nitride", "GaN", "semiconductor", "network", "solid", "industry", "blue leds"),
    ("indium_phosphide", "InP", "semiconductor", "network", "solid", "industry", ""),
    ("cadmium_telluride", "CdTe", "semiconductor", "network", "solid", "industry", "thin film solar cells"),
    ("silica", "SiO2", "ceramic", "network", "solid", "industry buildings", "amorphous silicon dioxide; fused silica glass"),
    ("boron_nitride", "BN", "ceramic", "network", "solid", "industry", ""),
    ("zirconia", "ZrO2", "ceramic", "ionic", "solid", "industry medicine", ""),
    ("tungsten_carbide", "WC", "ceramic", "network", "solid", "industry", ""),
    ("titanium_nitride", "TiN", "ceramic", "network", "solid", "industry", ""),
    ("silicon_nitride", "Si3N4", "ceramic", "network", "solid", "industry", ""),
    ("mullite", "Al6Si2O13", "ceramic", "network", "solid", "industry buildings", "in fired clay"),
    ("cementite", "Fe3C", "ceramic", "network", "solid", "industry", "iron carbide, the hard phase of steel"),
    # metals
    ("gold", "Au", "metal", "metallic", "solid", "crust rivers industry", ""),
    ("silver", "Ag", "metal", "metallic", "solid", "crust industry", ""),
    ("copper", "Cu", "metal", "metallic", "solid", "crust industry buildings batteries", ""),
    ("platinum", "Pt", "metal", "metallic", "solid", "crust industry", ""),
    ("aluminum", "Al", "metal", "metallic", "solid", "industry buildings", ""),
    ("tin", "Sn", "metal", "metallic", "solid", "industry", ""),
    ("lead", "Pb", "metal", "metallic", "solid", "industry batteries buildings", ""),
    ("zinc", "Zn", "metal", "metallic", "solid", "industry batteries buildings", ""),
    ("tungsten", "W", "metal", "metallic", "solid", "industry", ""),
    ("titanium", "Ti", "metal", "metallic", "solid", "industry medicine", ""),
    ("chromium", "Cr", "metal", "metallic", "solid", "industry", ""),
    ("cobalt", "Co", "metal", "metallic", "solid", "industry batteries", ""),
    ("manganese", "Mn", "metal", "metallic", "solid", "industry batteries", ""),
    ("magnesium", "Mg", "metal", "metallic", "solid", "industry", ""),
    ("sodium", "Na", "metal", "metallic", "solid", "industry", ""),
    ("potassium", "K", "metal", "metallic", "solid", "industry", ""),
    ("calcium", "Ca", "metal", "metallic", "solid", "industry", ""),
    ("lithium", "Li", "metal", "metallic", "solid", "batteries industry", ""),
    ("mercury", "Hg", "metal", "metallic", "liquid", "industry", "the liquid metal"),
    ("antimony", "Sb", "element", "metallic", "solid", "industry",
     "a metalloid, as the elements table says; brittle, its layers are held partly by covalent bonds"),
    ("uranium", "U", "metal", "metallic", "solid", "industry fuel", ""),
    ("charcoal", "C", "element", "network", "solid", "fuel industry", "idealised as pure carbon"),
    ("carbon", "C", "element", "network", "solid", "crust stars interstellar_clouds industry fuel",
     "the element whatever its form: graphite, diamond, charcoal, soot; dust of carbon stars"),
    # the species of the reactions table that have no row above (same names, same formulas)
    ("iron_oxide", "Fe2O3", "oxide", "ionic", "solid", "crust mars industry",
     "iron three oxide; the compound of the mineral hematite; rust is close to it"),
    ("iron_sulfide", "FeS", "salt", "ionic", "solid", "industry", "iron two sulfide; the compound of the mineral troilite"),
    ("iron_chloride", "FeCl3", "salt", "ionic", "solid", "industry", "iron three chloride"),
    ("iron_hydroxide", "Fe(OH)3", "base", "ionic", "solid", "industry", "iron three hydroxide; a brown precipitate"),
    ("iron_sulfate", "FeSO4", "salt", "ionic", "solid", "industry medicine", "iron two sulfate"),
    ("aluminum_chloride", "AlCl3", "salt", "ionic", "solid", "industry", "nearly covalent; sublimes at 453 k"),
    ("aluminum_hydroxide", "Al(OH)3", "base", "ionic", "solid", "industry medicine",
     "the compound of the mineral gibbsite; reacts with acids and with bases"),
    ("barium_chloride", "BaCl2", "salt", "ionic", "solid", "industry", ""),
    ("barium_sulfate", "BaSO4", "salt", "ionic", "solid", "industry medicine", "the compound of the mineral barite"),
    ("calcium_carbide", "CaC2", "salt", "ionic", "solid", "industry", "gives acetylene with water"),
    ("copper_oxide", "CuO", "oxide", "ionic", "solid", "industry", "copper two oxide, black"),
    ("copper_hydroxide", "Cu(OH)2", "base", "ionic", "solid", "industry", ""),
    ("copper_carbonate", "CuCO3", "salt", "ionic", "solid", "industry", "the green of old copper is close to it"),
    ("copper_nitrate", "Cu(NO3)2", "salt", "ionic", "solid", "industry", ""),
    ("silver_nitrate", "AgNO3", "salt", "ionic", "solid", "industry medicine", ""),
    ("silver_sulfide", "Ag2S", "salt", "ionic", "solid", "industry", "the tarnish on silver"),
    ("lead_nitrate", "Pb(NO3)2", "salt", "ionic", "solid", "industry", ""),
    ("lead_iodide", "PbI2", "salt", "ionic", "solid", "industry", "a yellow precipitate"),
    ("lead_sulfate", "PbSO4", "salt", "ionic", "solid", "batteries industry", "forms when a lead battery discharges"),
    ("mercury_oxide", "HgO", "oxide", "ionic", "solid", "industry batteries", "mercury two oxide"),
    ("zinc_chloride", "ZnCl2", "salt", "ionic", "solid", "industry batteries", ""),
    ("zinc_sulfate", "ZnSO4", "salt", "ionic", "solid", "industry medicine", ""),
    ("lithium_hydroxide", "LiOH", "base", "ionic", "solid", "industry", "takes carbon dioxide out of air"),
    ("potassium_bromide", "KBr", "salt", "ionic", "solid", "industry medicine", ""),
    ("potassium_iodide", "KI", "salt", "ionic", "solid", "medicine food industry", "in iodised salt"),
    ("sodium_iodide", "NaI", "salt", "ionic", "solid", "medicine industry", ""),
    ("potassium_chlorate", "KClO3", "salt", "ionic", "solid", "industry", "matches and fireworks"),
    ("potassium_superoxide", "KO2", "oxide", "ionic", "solid", "industry", "a superoxide; gives oxygen in breathing sets"),
    ("sodium_peroxide", "Na2O2", "oxide", "ionic", "solid", "industry", "a peroxide"),
    ("sodium_azide", "NaN3", "salt", "ionic", "solid", "industry", "fills airbags"),
    ("sodium_acetate", "C2H3NaO2", "salt", "ionic", "solid", "food industry", ""),
    ("phosphorus_pentoxide", "P4O10", "oxide", "covalent", "solid", "industry", "molecular cages; takes up water"),
    ("dinitrogen_tetroxide", "N2O4", "oxide", "covalent", "liquid", "fuel industry",
     "boils at 294 k; rocket oxidiser; it is two nitrogen_dioxide joined"),
    ("hydrazine", "N2H4", "inorganic", "covalent", "liquid", "fuel industry", "rocket fuel; a weak base"),
    ("paraffin_wax", "C25H52", "hydrocarbon", "covalent", "solid", "industry fuel",
     "candle wax; a mix of long alkanes, pentacosane stands for it"),
]

#: (name, main parts as "component share ...", basis, state, occurs_in, notes); shares in percent,
#: largest first; with basis element_mass the components are elements, else rows or materials
MIXTURES: list[tuple[str, str, str, str, str, str]] = [
    ("air", "nitrogen 78.1 oxygen 20.9 argon 0.93 carbon_dioxide 0.04", "volume", "gas", "atmosphere", "dry air"),
    ("seawater", "water 96.5 sodium_chloride 2.7 magnesium_chloride 0.4 magnesium_sulfate 0.2 calcium_sulfate 0.1 potassium_chloride 0.1",
     "mass", "liquid", "ocean", "the salts are dissolved as ions; approximate"),
    ("steel", "iron 99 manganese 0.8 carbon 0.2", "element_mass", "solid", "industry buildings",
     "mild steel by element; the carbon sits in cementite; approximate"),
    ("stainless_steel", "iron 70 chromium 19 nickel 9 manganese 2", "mass", "solid", "industry buildings", "grade 304; approximate"),
    ("cast_iron", "iron 94 carbon 3.5 silicon 2", "element_mass", "solid", "industry buildings",
     "grey iron by element; approximate"),
    ("bronze", "copper 88 tin 12", "mass", "solid", "industry", "classic tin bronze"),
    ("brass", "copper 65 zinc 35", "mass", "solid", "industry", "common brass"),
    ("solder", "tin 63 lead 37", "mass", "solid", "industry", "eutectic tin lead solder"),
    ("sterling_silver", "silver 92.5 copper 7.5", "mass", "solid", "industry", "925 parts of silver in 1000 by definition"),
    ("gold_18k", "gold 75 silver 12.5 copper 12.5", "mass", "solid", "industry", "yellow 18 carat gold; approximate"),
    ("cupronickel", "copper 75 nickel 25", "mass", "solid", "industry", "coins"),
    ("nichrome", "nickel 80 chromium 20", "mass", "solid", "industry", "heating wire"),
    ("invar", "iron 64 nickel 36", "mass", "solid", "industry", "36 percent nickel by definition; barely expands with heat"),
    ("duralumin", "aluminum 94 copper 4 magnesium 1 manganese 1", "mass", "solid", "industry", "approximate"),
    ("pewter", "tin 92 antimony 6 copper 2", "mass", "solid", "industry", "modern pewter; approximate"),
    ("dental_amalgam", "mercury 50 silver 35 tin 13 copper 2", "mass", "solid", "medicine", "approximate"),
    ("iron_nickel", "iron 94 nickel 6", "mass", "solid", "core",
     "kamacite, the alloy of iron meteorites, and close to the alloy of the core; approximate"),
    ("gunpowder", "potassium_nitrate 75 charcoal 15 sulfur 10", "mass", "solid", "industry", "the classic black powder recipe"),
    ("granite", "quartz 30 orthoclase 30 albite 25 biotite 10 hornblende 5", "mass", "solid", "crust buildings",
     "by mineral; typical, approximate"),
    ("basalt", "anorthite 45 diopside 35 forsterite 10 magnetite 5 ilmenite 5", "mass", "solid", "crust mars",
     "by mineral; typical, approximate"),
    ("limestone", "calcite 95 dolomite 3 quartz 2", "mass", "solid", "crust buildings", "approximate"),
    ("sandstone", "quartz 85 orthoclase 5 kaolinite 5 calcite 5", "mass", "solid", "crust buildings", "approximate"),
    ("upper_mantle", "forsterite 60 enstatite 20 diopside 10 pyrope 10", "mass", "solid", "mantle",
     "olivine as forsterite, pyroxenes, garnet; approximate"),
    ("lower_mantle", "bridgmanite 75 periclase 18 davemaoite 7", "mass", "solid", "mantle",
     "periclase stands for ferropericlase; approximate"),
    ("earth_core", "iron 85 nickel 5 silicon 4 oxygen 3 sulfur 2", "element_mass", "liquid", "core",
     "by element; the outer core is liquid, the inner solid; the light elements are uncertain; from memory"),
    ("earth_crust", "oxygen 46.6 silicon 27.7 aluminum 8.1 iron 5 calcium 3.6 sodium 2.8 potassium 2.6 magnesium 2.1",
     "element_mass", "solid", "crust", "by element; clarke values from memory"),
    ("sun", "hydrogen 73.5 helium 24.9 oxygen 0.8 carbon 0.3 iron 0.2 neon 0.1 nitrogen 0.1 silicon 0.1", "element_mass", "gas", "stars",
     "by element, photosphere; a plasma, listed as gas; the sun is a star, so its place is stars; from memory"),
    ("universe", "hydrogen 74 helium 24 oxygen 1 carbon 0.5", "element_mass", "gas", "stars interstellar_clouds",
     "ordinary matter by element; from memory"),
    ("interstellar_medium", "hydrogen 70 helium 28 oxygen 1 carbon 0.5", "element_mass", "gas", "interstellar_clouds",
     "by element; from memory"),
    ("comet_ice", "water 80 carbon_dioxide 10 carbon_monoxide 5 methanol 2 ammonia 1 methane 1", "mass", "solid", "comets",
     "the volatile ices of a nucleus; varies, approximate"),
    ("mars_atmosphere", "carbon_dioxide 95.3 nitrogen 2.7 argon 1.6 oxygen 0.1", "volume", "gas", "mars",
     "viking lander values, rounded; from memory"),
    ("venus_atmosphere", "carbon_dioxide 96.5 nitrogen 3.5", "volume", "gas", "venus", "sulfur dioxide in traces"),
    ("titan_atmosphere", "nitrogen 95 methane 5", "volume", "gas", "titan", "near the surface; approximate"),
    ("jupiter_atmosphere", "hydrogen 89.5 helium 10.2 methane 0.3", "volume", "gas", "jupiter", "approximate"),
    ("human_body", "oxygen 65 carbon 18 hydrogen 10 nitrogen 3 calcium 1.5 phosphorus 1", "element_mass", "solid", "animals",
     "by element; about six tenths of the body is water, solid is the nearest state; approximate"),
    ("bone", "hydroxyapatite 65 collagen 25 water 10", "mass", "solid", "bones animals", "approximate"),
    ("tooth_enamel", "hydroxyapatite 96 water 3 proteins 1", "mass", "solid", "bones animals", "approximate"),
    ("blood_plasma", "water 92 proteins 7 salts 1", "mass", "liquid", "blood animals", "approximate"),
    ("milk", "water 87 lactose 5 fats 4 proteins 3 minerals 1", "mass", "liquid", "food animals", "cow milk; approximate"),
    ("butter", "fats 81 water 16 proteins 1 lactose 1", "mass", "solid", "food", "approximate"),
    ("honey", "fructose 38 glucose 31 water 17 maltose 7 sucrose 1", "mass", "liquid", "food", "approximate"),
    ("olive_oil", "oleic_acid 72 palmitic_acid 13 linoleic_acid 9 stearic_acid 3", "mass", "liquid", "food plants",
     "as fatty acids; the oil holds them as triglycerides; approximate"),
    ("vinegar", "water 95 acetic_acid 5", "mass", "liquid", "food", "table vinegar; approximate"),
    ("wine", "water 86 ethanol 11 sugars 2 glycerol 1", "mass", "liquid", "food", "approximate"),
    ("wood", "cellulose 45 hemicellulose 25 lignin 25 extractives 5", "mass", "solid", "plants buildings fuel",
     "dry wood; approximate"),
    ("cotton", "cellulose 95 water 5", "mass", "solid", "plants industry", "approximate"),
    ("coal", "carbon 80 oxygen 10 hydrogen 5 nitrogen 1.5 sulfur 1.5", "element_mass", "solid", "crust fuel",
     "bituminous coal by element, dry and ash free; approximate"),
    ("petrol", "octane 40 heptane 20 toluene 15 xylene 10 hexane 10", "mass", "liquid", "fuel",
     "representative blend; real petrol holds hundreds of hydrocarbons"),
    ("lpg", "propane 60 butane 40", "mass", "gas", "fuel", "varies with season and country"),
    ("natural_gas", "methane 90 ethane 5 propane 2 nitrogen 1 carbon_dioxide 1", "volume", "gas", "fuel crust", "approximate"),
    ("crude_oil", "carbon 85 hydrogen 12 sulfur 2 nitrogen 0.5 oxygen 0.5", "element_mass", "liquid", "crust fuel",
     "by element; approximate"),
    ("portland_cement", "alite 60 belite 18 tricalcium_aluminate 8 brownmillerite 8 gypsum 5", "mass", "solid",
     "buildings industry", "approximate"),
    ("concrete", "gravel 47 sand 30 portland_cement 15 water 8", "mass", "solid", "buildings",
     "a usual mix: half as much water as cement by mass; approximate"),
    ("soda_lime_glass", "silica 73 sodium_oxide 14 calcium_oxide 9 magnesium_oxide 3 aluminum_oxide 1", "mass", "solid",
     "buildings industry", "by oxide; approximate"),
    ("hydrochloric_acid", "water 63 hydrogen_chloride 37", "mass", "liquid", "industry",
     "the concentrated acid; the reactions table writes it by its formula h cl"),
    ("bulk_earth", "iron 32 oxygen 30 silicon 16 magnesium 15 nickel 1.8 calcium 1.7 aluminum 1.6", "element_mass", "solid",
     "crust mantle core", "the whole planet by element; from memory"),
]


_NAME = re.compile(r"^[a-z][a-z0-9_]{2,}$")
_NUMBER = re.compile(r"^-?(0|[1-9]\d*)(\.\d+)?(e-?\d{1,3})?$")


# ---- exact numbers ----------------------------------------------------------------------------------

def sig(x: float | int | Fraction, digits: int) -> float:
    """``x`` to ``digits`` significant digits; a decimal tie rounds up (126.115 -> 126.12,
    50.9415 -> 50.942). A float is read as the decimal it prints as, so its binary noise is gone."""
    q = x if isinstance(x, Fraction) else Fraction(Decimal(f"{x:.12g}"))
    if q == 0:
        return 0.0
    sign, q = (-1 if q < 0 else 1), abs(q)
    exp = math.floor(math.log10(q))
    exp += (q >= Fraction(10) ** (exp + 1)) - (q < Fraction(10) ** exp)  # log10 may be one off at a power of ten
    scale = Fraction(10) ** (digits - 1 - exp)
    return sign * float(Fraction(math.floor(q * scale + Fraction(1, 2))) / scale)


def exact(text: str | list[str]) -> Fraction | None:
    """The exact value of a number in digit tokens (``1 8 point 0 1 5``), or ``None``."""
    words = text.split() if isinstance(text, str) else list(text)
    if not words or parse_num(words) is None:
        return None
    plain = "".join({"point": ".", "minus": "-"}.get(w, w) for w in words)
    if not _NUMBER.match(plain):
        return None
    try:
        return Fraction(Decimal(plain))
    except (InvalidOperation, ValueError, OverflowError):
        return None


def load_weights(data_dir: Path = DATA) -> dict[str, float | int]:
    return json.loads((data_dir / "atomic_weights.json").read_text(encoding="utf-8"))


def formula_mass(counts: dict[str, int], weights: dict[str, float | int]) -> Fraction:
    """The molar mass as an exact fraction: the weights are decimals, so the sum has no float noise
    and does not depend on the order of the elements."""
    return sum((Fraction(str(weights[sym])) * n for sym, n in counts.items()), Fraction(0))


def describe(formula: str, weights: dict[str, float | int]) -> dict:
    """The computed fields of a formula: dense form, elements by name, atoms, elements, molar
    mass (5 significant digits) and mass percent of each element (4)."""
    counts = parse(formula)
    mass = formula_mass(counts, weights)
    return {
        "dense": dense(formula),
        "composition": {NAME_OF[sym]: n for sym, n in counts.items()},
        "atoms": sum(counts.values()),
        "elements": len(counts),
        "molar_mass": sig(mass, 5),
        "mass_percent": {NAME_OF[sym]: sig(100 * Fraction(str(weights[sym])) * n / mass, 4)
                         for sym, n in counts.items()},
    }


def _parts(text: str) -> dict[str, float | int]:
    """``copper 88 tin 12`` -> ``{"copper": 88, "tin": 12}``; raises on a malformed or repeated part."""
    words = text.split()
    out: dict[str, float | int] = {}
    if len(words) % 2:
        raise ValueError(f"parts need a share for every component: {text!r}")
    for comp, share in zip(words[::2], words[1::2]):
        if comp in out or not _NAME.match(comp):
            raise ValueError(f"bad or repeated component {comp!r} in {text!r}")
        out[comp] = float(share) if "." in share else int(share)
    return out


def _complete(places: list[str]) -> list[str]:
    """The places plus the places they lie inside (blood is in animals, the sun is a star)."""
    out = list(places)
    for p in places:
        if p in INSIDE and INSIDE[p] not in out:
            out.append(INSIDE[p])
    return out


def table_rows(weights: dict[str, float | int]) -> list[dict]:
    """The curated tables as rows with their computed fields; raises on a table mistake.

    ``occurs_in`` is completed here: every place brings the place it lies inside, and a pure row
    that is a main part of a mixture (by mass or volume) occurs wherever the mixture occurs."""
    rows: list[dict] = []
    by_name: dict[str, dict] = {}
    for name, formula, cls, bonds, state, occurs_in, notes in PURE:
        if name in by_name or not _NAME.match(name):
            raise ValueError(f"bad or repeated name {name!r}")
        places = occurs_in.split()
        bad = [p for p in places if p not in PLACES]
        if bad or not places or len(set(places)) != len(places) or cls not in CLASSES or bonds not in BONDS \
                or state not in STATES:
            raise ValueError(f"{name}: bad vocabulary {bad or (cls, bonds, state)}")
        row = {"name": name, "formula": formula, "mixture": False, "unit": name in UNITS, "class": cls,
               "bonds": bonds, "state": state, "occurs_in": _complete(places), "notes": notes}
        row.update(describe(formula, weights))
        if row["unit"] != (cls == "polymer_unit"):
            raise ValueError(f"{name}: class polymer_unit and UNITS disagree")
        rows.append(row)
        by_name[name] = row
    for name, parts, basis, state, occurs_in, notes in MIXTURES:
        if name in by_name or not _NAME.match(name):
            raise ValueError(f"bad or repeated name {name!r}")
        places = occurs_in.split()
        shares = _parts(parts)
        if basis == "element_mass":
            bad = [c for c in shares if c not in SYMBOL_OF]
        else:  # a component is a row declared before (a mixture may hold a mixture) or a material
            bad = [c for c in shares if c not in by_name and c not in MATERIALS]
        bad += [p for p in places if p not in PLACES]
        if bad or not places or basis not in BASES or state not in STATES or not 90 <= sum(shares.values()) <= 101:
            raise ValueError(f"{name}: bad parts or vocabulary {bad}")
        row = {"name": name, "mixture": True, "main_parts": shares, "basis": basis, "state": state,
               "occurs_in": _complete(places), "notes": notes}
        rows.append(row)
        by_name[name] = row
        if basis != "element_mass":
            for comp in shares:
                held = by_name.get(comp)
                if held is not None and not held["mixture"]:
                    held["occurs_in"] = _complete(held["occurs_in"] + [p for p in row["occurs_in"]
                                                                       if p not in held["occurs_in"]])
    return rows


def load_rows(data_dir: Path = DATA) -> list[dict]:
    text = (data_dir / "substances.jsonl").read_text(encoding="utf-8")
    return [json.loads(ln) for ln in text.splitlines() if ln.strip()]


def n_tokens(text: str) -> int:
    return len(text.replace(".", " . ").split())


# ---- what a yes/no line may say about the hand-made columns ----------------------------------------

def class_truth(row: dict, cls: str) -> bool | None:
    """Whether class ``cls`` applies to a pure row: ``True`` for its label and for a class that
    follows from the formula (a metal is an element, quartz an oxide, glucose organic), ``False``
    where the formula or the bonds rule the class out, ``None`` where chemistry leaves it open
    (is halite a salt, is cholesterol an alcohol) or ``cls`` is no class at all."""
    if cls == row["class"]:
        return True
    if cls not in CLASSES or cls in CLASS_DOUBT.get(row["name"], ()):
        return None
    els = set(parse(row["formula"]))
    single = len(els) == 1
    organic = {"C", "H"} <= els and row["bonds"] == "covalent"
    if cls == "element":
        return single
    if cls == "metal":  # every metal row is labelled metal; a metalloid is left open
        return None if single and els <= METALLOIDS else False
    if cls in ("ice", "polymer_unit", "nucleobase"):  # every ice, repeat unit and base carries the label
        return False
    if cls == "hydrocarbon":
        return els == {"C", "H"}
    if cls == "inorganic":
        return True if "C" not in els else False if organic else None
    if cls == "drug":  # a use, not a structure: ruled out only for what is no medicine at all
        hard = row["class"] in ("mineral", "metal", "ceramic", "semiconductor", "ice") or row["bonds"] == "none"
        return False if hard and "medicine" not in row["occurs_in"] else None
    if cls in ORGANIC_CLASSES:
        if "C" not in els:
            return False
        if cls == "organic":
            return True if organic or row["class"] in ORGANIC_CLASSES else None
        need = {"alcohol": {"H", "O"}, "sugar": {"H", "O"}, "amino_acid": {"H", "N"}, "lipid": {"H"}}[cls]
        return None if need <= els else False
    if cls == "oxide":
        if single or "O" not in els or organic:
            return False
        return True if len(els) == 2 else None
    if cls == "salt":
        if row["class"] == "amino_acid" or row["name"] in ZWITTERIONS:
            return None
        return False if single or row["bonds"] in ("covalent", "metallic", "none") else None
    if cls in ("acid", "base"):
        other = "base" if cls == "acid" else "acid"
        return False if single or els == {"C", "H"} or row["class"] == other else None
    if cls == "mineral":
        return False if organic or row["state"] == "gas" else None
    if cls == "ceramic":
        return False if organic or row["state"] != "solid" or row["bonds"] in ("covalent", "metallic", "none") else None
    if cls == "semiconductor":
        return False if row["state"] != "solid" or row["bonds"] in ("metallic", "none") else None
    # explosive
    return False if row["bonds"] == "none" or row["class"] in ("mineral", "ceramic", "semiconductor") else None


def _plainly_ionic(counts: dict[str, int]) -> bool:
    """A halide or a plain oxide of an alkali or alkaline earth metal: ionic beyond doubt."""
    if len(counts) != 2:
        return False
    cation = next((el for el in counts if el in STRONG_CATIONS), None)
    anion = next((el for el in counts if el != cation), None)
    if cation is None:
        return False
    charge = 1 if cation in ("Li", "Na", "K", "Rb", "Cs") else 2
    if anion in HALOGENS:
        return counts[cation] * charge == counts[anion]
    return anion == "O" and counts[cation] * charge == 2 * counts["O"]  # not a peroxide or superoxide


def bond_truth(row: dict, bond: str) -> bool | None:
    """Whether bond type ``bond`` describes a pure row: ``True`` for its label (and ``covalent``
    for a network), ``False`` where it is ruled out, ``None`` for a judgement call."""
    own = row["bonds"]
    if bond == own or (own == "network" and bond == "covalent"):
        return True
    if bond not in BONDS or row["name"] in BOND_DOUBT:
        return None
    if bond in ("metallic", "none") or own in ("metallic", "none"):
        return False
    counts = parse(row["formula"])
    if own == "covalent":
        # a molecular substance; a polymer may be cross-linked into a network, and the crystal of an
        # amino acid is held by the charges of its two ends
        inner_salt = row["class"] == "amino_acid" or row["name"] in ZWITTERIONS
        return None if (bond == "network" and row["unit"]) or (bond == "ionic" and inner_salt) else False
    if own == "ionic":  # against covalent or network: only the plain salts are beyond doubt
        return False if _plainly_ionic(counts) else None
    # a network against ionic: ruled out only where no metal is in the formula
    return False if all(el in NONMETALS or el in METALLOIDS for el in counts) else None


def state_truth(row: dict, state: str) -> bool | None:
    if state == row["state"]:
        return True
    return None if state not in STATES or row["name"] in STATE_DOUBT else False


def basis_truth(row: dict, basis: str) -> bool | None:
    """Mass against volume is a clear yes or no; shares by element are shares by mass too."""
    if basis == row["basis"]:
        return True
    if basis not in BASES or {basis, row["basis"]} == {"mass", "element_mass"}:
        return None
    return False


def impossible_places(row: dict) -> frozenset[str]:
    """The places a pure row cannot occur in, by a few safe rules; ``occurs_in`` is an open list,
    so only these places are a ``no``.

    * the molecules of life, drugs, polymers and explosives are in no star, mantle or core; any
      molecule with carbon and hydrogen is neither in the sun nor in the core;
    * a mineral, a ceramic or a semiconductor is not in the sun and not in dna;
    * an ice is not in the sun, the core, blood, bones or dna;
    * dna holds no hydrocarbon, lipid, amino acid, polymer, explosive or free atom; bones hold no
      explosive, semiconductor or free atom.

    A row that carries the name of an element has no such place: ``iron`` is the metal here, but
    the sun, the crust and the human body list their elements under the same names.
    """
    cls, els = row["class"], set(parse(row["formula"]))
    if row["name"] in SYMBOL_OF:
        return frozenset()
    out: set[str] = set()
    if cls in ("sugar", "amino_acid", "nucleobase", "lipid", "drug", "polymer_unit", "explosive"):
        out |= {"sun", "stars", "mantle", "core"}
    if {"C", "H"} <= els and row["bonds"] == "covalent":
        out |= {"sun", "core"}
    if cls in ("mineral", "ceramic", "semiconductor"):
        out |= {"sun", "dna"}
    if cls == "ice":
        out |= {"sun", "core", "blood", "bones", "dna"}
    if cls in ("hydrocarbon", "lipid", "amino_acid", "polymer_unit", "explosive") or row["bonds"] == "none":
        out.add("dna")
    if cls in ("explosive", "semiconductor") or row["bonds"] == "none":
        out.add("bones")
    return frozenset(out)


def _pairs(text: str, counts: bool = False) -> dict[str, Fraction] | None:
    """``hydrogen 2 oxygen 1`` -> ``{"hydrogen": 2, "oxygen": 1}`` with exact values; ``None`` when
    malformed, when a key comes twice or, with ``counts``, when a value is not a plain count."""
    out: dict[str, Fraction] = {}
    key, digits = "", []
    for w in text.split() + [""]:
        if w and (w.isdigit() or w in ("point", "minus", "e")) and key:
            digits.append(w)
            continue
        if key:
            value = exact(digits)
            if value is None or key in out or (counts and " ".join(digits) != num(int(value))):
                return None
            out[key] = value
        key, digits = w, []
    return out or None


def strict_formula(text: str) -> dict[str, int] | None:
    """Atom counts of a dense formula written without slack: every element once, a missing count
    is 1, no zero count, no count of more than :data:`MAX_COUNT_DIGITS` digits; else ``None``."""
    words = text.split() if isinstance(text, str) else list(text)
    counts: dict[str, int] = {}
    i = 0
    while i < len(words):
        el = words[i]
        if not el.isalpha() or not el.isascii() or len(el) > 2 or el.capitalize() in counts:
            return None
        digits = ""
        i += 1
        while i < len(words) and len(words[i]) == 1 and words[i].isdigit():
            digits += words[i]
            i += 1
        if digits.startswith("0") or len(digits) > MAX_COUNT_DIGITS:
            return None
        counts[el.capitalize()] = int(digits) if digits else 1
    return counts or None


class Substances:
    """The gate: records, questions and the judge for topic ``substances``."""

    topic = TOPIC
    KEYS = KEYS

    def __init__(self, data_dir: Path = DATA) -> None:
        self.weights = load_weights(data_dir)
        self.rows = {r["name"]: r for r in load_rows(data_dir)}
        self.pure = [r for r in self.rows.values() if not r["mixture"]]
        self.mixtures = [r for r in self.rows.values() if r["mixture"]]
        self.by_formula: dict[str, list[str]] = {}
        for r in self.pure:
            # an ice is a form of its substance and a repeat unit is not the formula of its polymer
            if r["class"] != "ice" and not r["unit"]:
                self.by_formula.setdefault(self._key(parse(r["formula"])), []).append(r["name"])
        self.table_elements = sorted({sym for r in self.pure for sym in parse(r["formula"])})
        #: the rows that are molecules: covalent, and not the repeat unit of a polymer
        self.molecules = [r for r in self.pure if r["bonds"] == "covalent" and not r["unit"]]
        self.impossible = {r["name"]: impossible_places(r) for r in self.pure}
        for r in self.pure:
            clash = self.impossible[r["name"]] & set(r["occurs_in"])
            if clash:
                raise ValueError(f"{r['name']}: occurs_in holds a place the rules exclude: {sorted(clash)}")
        self.not_parts = {m["name"]: self._not_parts(m) for m in self.mixtures}
        self.lists = self._lists()
        self._makers = [
            (37, self._fact), (6, self._reverse), (2, self._list),
            (10, self._calc_mass), (6, self._calc_atoms), (8, self._calc_percent), (6, self._calc_count),
            (5, self._calc_heavier),
            (12, self._check_fact), (5, self._check_calc), (3, self._check_heavier),
        ]

    # ---- the table as lines -------------------------------------------------------------------

    @staticmethod
    def _key(counts: dict[str, int]) -> str:
        return " ".join(f"{el}{n}" for el, n in sorted(counts.items()))

    def value(self, row: dict, key: str) -> str | None:
        """The dense answer to ``q <name> <key>``, or ``None`` when the row has no such key."""
        if key == "mixture":
            return "yes" if row["mixture"] else "no"
        if key == "state":
            return row["state"]
        if key == "occurs_in":
            return " ".join(row["occurs_in"])
        if row["mixture"]:
            if key == "main_parts":
                return " ".join(f"{c} {num(v, 3)}" for c, v in row["main_parts"].items())
            if key == "basis":
                return row["basis"]
            return None
        if key == "formula":
            return row["dense"]
        if key == "unit":
            return "yes" if row["unit"] else "no"
        if key == "composition":
            return " ".join(f"{el} {num(n)}" for el, n in row["composition"].items())
        if key in COUNT_KEYS:
            return num(row[key])
        if key == "molar_mass":
            return num(row["molar_mass"], 5)
        if key == "mass_percent":
            return " ".join(f"{el} {num(p, 4)}" for el, p in row["mass_percent"].items())
        if key in ("class", "bonds"):
            return row[key]
        return None

    def record_lines(self, row: dict) -> list[Line]:
        if row["mixture"]:
            fields = [f"{k} {self.value(row, k)}." for k in ("main_parts", "basis", "state", "occurs_in")]
            head = f"{row['name']} mixture."
        else:
            keys = [k for k in PURE_KEYS if k != "mixture" and (k != "unit" or row["unit"])]
            fields = [f"{k} {self.value(row, k)}." for k in keys]
            head = f"{row['name']} substance."
        lines, current = [], head
        for f in fields:
            if n_tokens(f"{current} {f}") > MAX_TOKENS:
                lines.append(current)
                current = f"{head[:-1]}_more. {f}"
            else:
                current = f"{current} {f}"
        lines.append(current)
        if len(lines) > 2 or any(n_tokens(text) > MAX_TOKENS for text in lines):
            raise ValueError(f"{row['name']}: the record does not fit two lines of {MAX_TOKENS} tokens")
        return [Line(text, topic=self.topic, kind="record") for text in lines]

    def records(self) -> list[Line]:
        return [ln for row in self.rows.values() for ln in self.record_lines(row)]

    def facts(self) -> list[Line]:
        """Every table fact once, in table order: each key of each row, each mixture share, each
        reverse lookup and each list. :meth:`generate` samples from this same pool, so a build
        that wants every fact trained takes this list and adds generated calc and yes/no lines."""
        out: list[Line] = []
        for row in self.rows.values():
            keys = MIX_KEYS if row["mixture"] else tuple(k for k in PURE_KEYS if k != "unit" or row["unit"])
            out += [Line(f"{row['name']} {k}", self.value(row, k), self.topic, "fact") for k in keys]
            if row["mixture"]:
                out += [Line(f"{row['name']} part {c}", num(v, 3), self.topic, "fact")
                        for c, v in row["main_parts"].items()]
        for row in self.pure:
            names = self.by_formula.get(self._key(parse(row["formula"])))
            if names:
                out.append(Line(f"substance formula {row['dense']}", " ".join(sorted(names)), self.topic, "fact"))
        out += [Line(prompt, " ".join(members), self.topic, "fact") for prompt, members in self.lists]
        long = [ln.text for ln in out if n_tokens(ln.text) > MAX_TOKENS]
        if long:
            raise ValueError(f"a fact line passes {MAX_TOKENS} tokens: {long[0]}")
        return dedupe(out)

    def place_truth(self, row: dict, place: str) -> bool | None:
        """Whether a row occurs in a place: listed, ruled out (:func:`impossible_places`), or
        undecided, as the list is open."""
        if place in row["occurs_in"]:
            return True
        if place not in PLACES or row["mixture"]:
            return None
        return False if place in self.impossible[row["name"]] else None

    def _member(self, row: dict, key: str, value: str) -> bool | None:
        """Whether a pure row belongs to the list ``substances <key> <value>``, or ``None`` when
        that is undecided. Under one of the :data:`CLOSED_PLACES` a row that does not list the
        place is no member."""
        if key == "occurs_in":
            return value in row["occurs_in"] if value in CLOSED_PLACES else self.place_truth(row, value)
        return {"class": class_truth, "bonds": bond_truth, "state": state_truth}[key](row, value)

    def list_members(self, words: list[str]) -> list[str] | None:
        """Members of ``substances <key> <value> [<key> <value>]`` in alphabetical order; ``None``
        when the prompt is malformed or the list is not whole: some row is undecided for one
        condition and not ruled out by the other."""
        if len(words) not in (2, 4) or any(k not in LIST_KEYS for k in words[::2]) or words[0] in words[2:3]:
            return None
        out = []
        for r in self.pure:
            verdicts = [self._member(r, key, value) for key, value in zip(words[::2], words[1::2])]
            if False in verdicts:
                continue
            if None in verdicts:
                return None
            out.append(r["name"])
        return sorted(out)

    def _lists(self) -> list[tuple[str, list[str]]]:
        """``(prompt, members)`` for every whole list question with 1 to MAX_LIST pure members."""
        values = {"class": CLASSES, "bonds": BONDS, "state": STATES, "occurs_in": PLACES}
        prompts = [f"{k} {v}" for k in LIST_KEYS for v in values[k]]
        prompts += [f"{k1} {v1} {k2} {v2}" for i, k1 in enumerate(LIST_KEYS) for k2 in LIST_KEYS[i + 1:]
                    for v1 in values[k1] for v2 in values[k2]]
        out = []
        for prompt in prompts:
            members = self.list_members(prompt.split())
            if members and len(members) <= MAX_LIST:
                out.append((f"substances {prompt}", members))
        return out

    def _not_parts(self, mixture: dict) -> list[str]:
        """The names that are definitely no main part of a mixture, in table order."""
        if mixture["basis"] == "element_mass":
            doubt = PART_DOUBT.get(mixture["name"], ())
            names = [NAME_OF[sym] for sym in self.table_elements]
            return [n for n in names if n not in mixture["main_parts"] and n not in doubt]
        places = set(mixture["occurs_in"])
        families = {cls for comp in mixture["main_parts"] for cls in MATERIALS.get(comp, ())}
        return [r["name"] for r in self.pure if r["name"] not in mixture["main_parts"]
                and not places & set(r["occurs_in"]) and r["class"] not in families]

    def part_truth(self, mixture: dict, name: str) -> bool | None:
        """Whether ``name`` is a main part of a mixture: listed, definitely not, or undecided."""
        if name in mixture["main_parts"]:
            return True
        return False if name in self.not_parts[mixture["name"]] else None

    # ---- calculations on any formula ----------------------------------------------------------

    def counts_of(self, words: list[str]) -> dict[str, int] | None:
        """Atom counts of a dense formula of known elements; ``None`` for anything else."""
        counts = strict_formula(words) if words else None
        if counts is None or any(el not in self.weights for el in counts):
            return None
        return counts

    def subject_counts(self, words: list[str]) -> dict[str, int] | None:
        """Atom counts of the name of a pure substance or of a dense formula."""
        if len(words) == 1 and words[0] in self.rows:
            row = self.rows[words[0]]
            return None if row["mixture"] else parse(row["formula"])
        return self.counts_of(words)

    def mass(self, counts: dict[str, int]) -> Fraction:
        return formula_mass(counts, self.weights)

    def molar_mass(self, counts: dict[str, int]) -> float:
        return sig(self.mass(counts), 5)

    def mass_percent(self, counts: dict[str, int], element: str) -> float:
        sym = SYMBOL_OF.get(element)
        if sym not in counts:
            return 0.0
        return sig(100 * Fraction(str(self.weights[sym])) * counts[sym] / self.mass(counts), 4)

    def worked_answer(self, words: list[str]) -> str | None:
        """Work a molar mass or mass percentage from exact atomic-weight decimals.

        Counts, weights, each product and the total remain exact. Only the final
        mass (five significant digits) or percentage (four) is rounded. Element
        order is sorted by symbol, making the trace independent of formula order.
        ``words`` is the original calculation, without the trailing ``steps``.
        """
        if not words:
            return None
        head, rest = words[0], words[1:]
        element = None
        if head == "molar_mass":
            counts = self.subject_counts(rest)
        elif head == "mass_percent":
            split = self._split_element(rest)
            if split is None or split[2]:
                return None
            counts, element, _ = split
        else:
            return None
        if counts is None:
            return None
        steps = ["formula sum count times atomic_weight"]
        products = {}
        for symbol, count in sorted(counts.items()):
            weight = Fraction(str(self.weights[symbol]))
            products[symbol] = weight * count
            steps.append(f"{symbol.lower()} {num(count)} times {decimal_tokens(weight)} "
                         f"is {decimal_tokens(products[symbol])}")
        total = self.mass(counts)
        terms = " plus ".join(decimal_tokens(value) for value in products.values())
        steps.append(f"sum {terms} is {decimal_tokens(total)}")
        if element is None:
            result = num(self.molar_mass(counts), 5)
        else:
            steps.append("formula 1 0 0 times element_mass over total_mass")
            part = products.get(SYMBOL_OF[element], Fraction(0))
            steps.append(f"substitute 1 0 0 times {decimal_tokens(part)} over {decimal_tokens(total)}")
            result = num(self.mass_percent(counts, element), 4)
        steps.append("result " + result)
        return " then ".join(steps)

    def _split_element(self, words: list[str]) -> tuple[dict[str, int], str, list[str]] | None:
        """``water oxygen ...`` / ``h 2 o 1 oxygen ...`` -> (counts, element name, the rest)."""
        if words and words[0] in self.rows:
            cut = 1
        else:
            cut = next((i for i, w in enumerate(words) if w in SYMBOL_OF), 0)
        if cut == 0 or cut >= len(words) or words[cut] not in SYMBOL_OF:
            return None
        counts = self.subject_counts(words[:cut])
        return None if counts is None else (counts, words[cut], words[cut + 1:])

    def _molecule(self, name: str) -> Fraction | None:
        """The exact molar mass of a row that is a molecule, else ``None``."""
        row = self.rows.get(name)
        if row is None or row["mixture"] or row["bonds"] != "covalent" or row["unit"]:
            return None
        return self.mass(parse(row["formula"]))

    # ---- judging --------------------------------------------------------------------------------

    def owns(self, prompt: str) -> bool:
        return self._owns(self._words(prompt))

    def _owns(self, words: list[str]) -> bool:
        if not words:
            return False
        head, rest = words[0], words[1:]
        if head == "check":
            return bool(rest) and rest[0] != "check" and self._owns(rest)
        if head in CALC_HEADS:  # followed by a row or by what may be an element symbol
            return bool(rest) and (rest[0] in self.rows or (rest[0].isalpha() and len(rest[0]) <= 2))
        if head == "substance":
            return len(rest) > 1 and rest[0] == "formula"
        if head == "substances":
            return len(rest) in (2, 4) and all(k in LIST_KEYS for k in rest[::2])
        if head in self.rows and rest:
            keys = MIX_KEYS + ("part",) if self.rows[head]["mixture"] else PURE_KEYS
            return rest[0] in keys
        return False

    @staticmethod
    def _words(prompt: str) -> list[str]:
        words = prompt.strip().rstrip(".").split()
        return words[1:] if words[:1] == ["q"] else words

    def check(self, prompt: str, answer: str) -> Verdict:
        words = self._words(prompt)
        if not self._owns(words):
            return Verdict(False, None, "not my question")
        answer = answer.strip().rstrip(".")
        head, rest = words[0], words[1:]
        if head == "check":
            expected = self._truth(rest)
            if expected is None:
                return Verdict(False, None, "cannot judge that claim")
            return Verdict(answer == expected, expected, "" if answer == expected else "checked by table or rule")
        result = self.expected(words)
        if result is None:
            return Verdict(False, None, "no answer in the table")
        expected, how = result
        if how == "steps":
            return check_steps(expected, answer)
        ok = self._agree(how, expected, answer)
        return Verdict(ok, expected, "" if ok else f"compared as {how}")

    def expected(self, words: list[str]) -> tuple[str, str] | None:
        """The canonical answer to a non-check prompt and how answers to it compare
        (``exact``, ``count``, ``number``, ``set``, ``formula``, ``pairs`` or ``numbers``)."""
        head, rest = words[0], words[1:]
        if head in ("molar_mass", "mass_percent") and rest[-1:] == ["steps"]:
            trace = self.worked_answer(words[:-1])
            return (trace, "steps") if trace is not None else None
        if head == "substance":
            counts = self.counts_of(rest[1:])
            names = self.by_formula.get(self._key(counts), []) if counts else []
            return (" ".join(sorted(names)), "set") if names else None
        if head == "substances":
            members = self.list_members(rest)
            return (" ".join(members), "set") if members and len(members) <= MAX_LIST else None
        if head in ("molar_mass", "atoms"):  # a dense formula, or the name of a pure substance
            counts = self.subject_counts(rest)
            if counts is None:
                return None
            if head == "atoms":
                return num(sum(counts.values())), "count"
            return num(self.molar_mass(counts), 5), "number"
        if head in ("mass_percent", "element_count"):
            split = self._split_element(rest)
            if split is None or split[2]:
                return None
            counts, element, _ = split
            if head == "element_count":
                return num(counts.get(SYMBOL_OF[element], 0)), "count"
            return num(self.mass_percent(counts, element), 4), "number"
        if head == "heavier_molecule":
            if len(rest) != 2 or any(self._molecule(w) is None for w in rest):
                return None
            a, b = (self._molecule(w) for w in rest)
            # ``same`` only for equal formulas (glucose and fructose); generate() never asks those
            return (rest[0] if a > b else rest[1] if b > a else "same"), "exact"
        row = self.rows[head]
        key = rest[0]
        if key == "part":
            if len(rest) != 2 or rest[1] not in row["main_parts"]:
                return None
            return num(row["main_parts"][rest[1]], 3), "number"
        if len(rest) != 1:
            return None
        value = self.value(row, key)
        if value is None:
            return None
        how = {"formula": "formula", "composition": "pairs", "mass_percent": "numbers", "main_parts": "numbers",
               "occurs_in": "set", "atoms": "count", "elements": "count", "molar_mass": "number"}.get(key, "exact")
        return value, how

    def _agree(self, how: str, expected: str, answer: str) -> bool:
        if how == "exact":
            return answer == expected
        if how == "set":
            words = answer.split()
            return len(words) == len(set(words)) and set(words) == set(expected.split())
        if how == "count":  # digits only, written as num() writes them
            return answer == expected
        if how == "number":
            got = exact(answer)
            return got is not None and got == exact(expected)
        if how == "formula":
            got = strict_formula(answer)
            return got is not None and got == strict_formula(expected)
        got, want = _pairs(answer, counts=how == "pairs"), _pairs(expected)
        return got is not None and got == want

    def _truth(self, claim: list[str]) -> str | None:
        """``yes``/``no`` for the claim behind ``check ...``; ``None`` when it cannot be judged."""
        head = claim[0]
        if head == "heavier_molecule":
            if len(claim) != 3 or any(self._molecule(w) is None for w in claim[1:]):
                return None
            a, b = (self._molecule(w) for w in claim[1:])
            return "yes" if a > b else "no"
        if head in ("element_count", "mass_percent"):
            split = self._split_element(claim[1:])
            if split is None or not split[2]:
                return None
            counts, element, value = split
            how = "count" if head == "element_count" else "number"
            right = (num(counts.get(SYMBOL_OF[element], 0)) if head == "element_count"
                     else num(self.mass_percent(counts, element), 4))
            return "yes" if self._agree(how, right, " ".join(value)) else "no"
        if head in ("molar_mass", "atoms"):
            # only by name: after a dense formula the claimed number cannot be told from a count
            if len(claim) < 3 or claim[1] not in self.rows or self.rows[claim[1]]["mixture"]:
                return None
            expected, how = self.expected([head, claim[1]])
            return "yes" if self._agree(how, expected, " ".join(claim[2:])) else "no"
        if head in ("substance", "substances"):
            return None
        row, key, value = self.rows[head], claim[1], claim[2:]
        if not value:
            return None
        if key in ("class", "bonds", "state", "basis", "occurs_in", "main_parts"):
            return self._claim(row, key, value)
        if key == "part":
            value = value[1:]
            if not value:
                return None
        result = self.expected([head, key] + (claim[2:3] if key == "part" else []))
        if result is None:
            return None
        expected, how = result
        return "yes" if self._agree(how, expected, " ".join(value)) else "no"

    def _claim(self, row: dict, key: str, value: list[str]) -> str | None:
        """A claim about a hand-made column: yes, no, or ``None`` where the table leaves it open."""
        if key not in (MIX_KEYS if row["mixture"] else PURE_KEYS):
            return None
        if key == "main_parts" and any(w.isdigit() for w in value):  # the whole value with its shares
            return "yes" if self._agree("numbers", self.value(row, key), " ".join(value)) else "no"
        if key in ("occurs_in", "main_parts"):  # one member, or several: all of them must hold
            if len(set(value)) != len(value):
                return None
            if key == "main_parts":
                verdicts = [self.part_truth(row, w) for w in value]
            else:
                verdicts = [self.place_truth(row, w) for w in value]
            verdict = False if False in verdicts else None if None in verdicts else True
        elif len(value) != 1:
            return None
        elif key == "basis":
            verdict = basis_truth(row, value[0])
        elif key == "state":
            verdict = state_truth(row, value[0])
        else:
            verdict = {"class": class_truth, "bonds": bond_truth}[key](row, value[0])
        return None if verdict is None else "yes" if verdict else "no"

    # ---- question makers ----------------------------------------------------------------------

    def generate_worked(self, rng: random.Random, n: int, max_tokens: int = WORKED_MAX_TOKENS) -> list[Line]:
        """Opt-in worked molar-mass/percentage lessons, half each before length filtering.

        The existing generator is unchanged. Set max_tokens=96 for short-context
        lessons; use the default and context 256 to retain multi-element work.
        """
        out, seen = [], set()
        for _ in range(100 * n + 1000):
            if len(out) == n:
                return out
            base, _, _ = (self._calc_mass if rng.random() < 0.5 else self._calc_percent)(rng)
            prompt = base + " steps"
            answer = self.worked_answer(base.split())
            if answer is None or prompt in seen:
                continue
            line = Line(prompt, answer, self.topic, "calc", {"form": "steps", "base_prompt": base})
            if n_tokens(line.text) > max_tokens:
                continue
            seen.add(prompt)
            out.append(line)
        raise ValueError(f"cannot generate {n} distinct worked substances lines within {max_tokens} tokens")

    def generate(self, rng: random.Random, n: int) -> list[Line]:
        weights = [w for w, _ in self._makers]
        makers = [m for _, m in self._makers]
        out, seen, tries = [], set(), 0
        while len(out) < n and tries < 60 * n + 1000:
            tries += 1
            made = rng.choices(makers, weights)[0](rng)
            if made is None:
                continue
            prompt, answer, kind = made
            text = f"q {prompt}. a {answer}."
            if text in seen or n_tokens(text) > MAX_TOKENS:
                continue
            seen.add(text)
            out.append(Line(prompt, answer, self.topic, kind))
        if len(out) < n:
            warnings.warn(f"substances: only {len(out)} of {n} distinct lines could be made", stacklevel=2)
        return out

    def _fact(self, rng: random.Random) -> tuple[str, str, str] | None:
        row = rng.choice(self.pure) if rng.random() < 0.85 else rng.choice(self.mixtures)
        if row["mixture"]:
            key = rng.choice(["main_parts", "main_parts", "basis", "state", "occurs_in", "mixture", "part", "part"])
            if key == "part":
                comp = rng.choice(list(row["main_parts"]))
                return f"{row['name']} part {comp}", num(row["main_parts"][comp], 3), "fact"
        else:
            keys = ["formula", "formula", "composition", "atoms", "elements", "molar_mass", "molar_mass", "mass_percent",
                    "class", "bonds", "state", "occurs_in"] + (["unit"] if row["unit"] else []) + ["mixture"]
            key = rng.choice(keys)
        return f"{row['name']} {key}", self.value(row, key), "fact"

    def _reverse(self, rng: random.Random) -> tuple[str, str, str] | None:
        row = rng.choice(self.pure)
        names = self.by_formula.get(self._key(parse(row["formula"])))
        return (f"substance formula {row['dense']}", " ".join(sorted(names)), "fact") if names else None

    def _list(self, rng: random.Random) -> tuple[str, str, str]:
        prompt, members = rng.choice(self.lists)
        return prompt, " ".join(members), "fact"

    def random_formula(self, rng: random.Random) -> str:
        """A small formula in standard case, mostly organic looking; it need not exist."""
        if rng.random() < 0.6:
            els = ["C", "H"] + rng.sample(["O", "N", "S", "P", "Cl", "F", "Br"], rng.choice([0, 1, 1, 2]))
            counts = [rng.randint(1, 20), rng.randint(1, 40)] + [rng.randint(1, 8) for _ in els[2:]]
        else:
            els = rng.sample(COMMON, rng.choice([1, 2, 2, 3, 3, 4]))
            counts = [rng.randint(1, 12) for _ in els]
        return "".join(f"{el}{n if n > 1 else ''}" for el, n in zip(els, counts))

    def _formula(self, rng: random.Random) -> str:
        return rng.choice(self.pure)["formula"] if rng.random() < 0.5 else self.random_formula(rng)

    def _calc_mass(self, rng: random.Random) -> tuple[str, str, str]:
        f = self._formula(rng)
        return f"molar_mass {dense(f)}", num(self.molar_mass(parse(f)), 5), "calc"

    def _calc_atoms(self, rng: random.Random) -> tuple[str, str, str]:
        f = self._formula(rng)
        return f"atoms {dense(f)}", num(sum(parse(f).values())), "calc"

    def _subject(self, rng: random.Random) -> tuple[str, dict[str, int]]:
        """A substance name or a dense formula, with its counts."""
        if rng.random() < 0.5:
            row = rng.choice(self.pure)
            return row["name"], parse(row["formula"])
        f = self._formula(rng)
        return dense(f), parse(f)

    def _element(self, rng: random.Random, counts: dict[str, int]) -> str:
        """An element of the formula by name; about one time in eight one that is not in it."""
        if rng.random() < 0.12:
            return NAME_OF[rng.choice([sym for sym in COMMON if sym not in counts])]
        return NAME_OF[rng.choice(list(counts))]

    def _calc_percent(self, rng: random.Random) -> tuple[str, str, str]:
        subject, counts = self._subject(rng)
        element = self._element(rng, counts)
        return f"mass_percent {subject} {element}", num(self.mass_percent(counts, element), 4), "calc"

    def _calc_count(self, rng: random.Random) -> tuple[str, str, str]:
        subject, counts = self._subject(rng)
        element = self._element(rng, counts)
        return f"element_count {subject} {element}", num(counts.get(SYMBOL_OF[element], 0)), "calc"

    def _heavier_pair(self, rng: random.Random) -> tuple[dict, dict] | None:
        """Two molecules whose molar masses differ by more than 1 %, so the answer is plain."""
        a, b = rng.sample(self.molecules, 2)
        if abs(a["molar_mass"] - b["molar_mass"]) < 0.01 * max(a["molar_mass"], b["molar_mass"]):
            return None
        return a, b

    def _calc_heavier(self, rng: random.Random) -> tuple[str, str, str] | None:
        pair = self._heavier_pair(rng)
        if pair is None:
            return None
        a, b = pair
        return f"heavier_molecule {a['name']} {b['name']}", (a if a["molar_mass"] > b["molar_mass"] else b)["name"], "calc"

    @staticmethod
    def _off(rng: random.Random, right: float, digits: int) -> float:
        """A wrong number near ``right``: far off, or one or two units off in its last digit."""
        if rng.random() < 0.7:
            return sig(right * rng.choice([0.5, 0.75, 0.9, 1.1, 1.3, 2.0]), digits)
        unit = Fraction(10) ** (math.floor(math.log10(right)) - digits + 1) if right > 0 else Fraction(1)
        return sig(Fraction(Decimal(f"{right:.12g}")) + unit * rng.choice([-2, -1, 1, 2]), digits)

    def _right(self, rng: random.Random, row: dict, key: str) -> str:
        """A true value for ``q check <name> <key> ...``."""
        if key == "occurs_in":
            return rng.choice(row["occurs_in"])
        if key == "main_parts":
            return rng.choice(list(row["main_parts"]))
        if key == "part":
            comp = rng.choice(list(row["main_parts"]))
            return f"{comp} {num(row['main_parts'][comp], 3)}"
        if key in ("class", "bonds") and rng.random() < 0.4:  # also a class or bond type beside the label
            truth = class_truth if key == "class" else bond_truth
            return rng.choice([v for v in (CLASSES if key == "class" else BONDS) if truth(row, v)])
        return self.value(row, key)

    def _wrong(self, rng: random.Random, row: dict, key: str) -> str | None:
        """A wrong value for ``q check <name> <key> ...``: never one that may be true; or ``None``."""
        if key == "molar_mass":
            wrong = self._off(rng, row["molar_mass"], 5)
            return None if wrong == row["molar_mass"] or wrong <= 0 else num(wrong, 5)
        if key in COUNT_KEYS:
            wrong = row[key] + rng.choice([-2, -1, 1, 2, 3])
            return num(wrong) if wrong >= 1 and wrong != row[key] else None
        if key == "formula":
            counts = dict(parse(row["formula"]))
            el = rng.choice(list(counts))
            if rng.random() < 0.6:
                counts[el] = max(1, counts[el] + rng.choice([-2, -1, 1, 2]))
            else:
                other = rng.choice(self.table_elements)
                if other in counts:
                    return None
                counts[other] = counts.pop(el)
            if counts == parse(row["formula"]):
                return None
            return " ".join(f"{e.lower()} {num(n)}" for e, n in counts.items())
        if key == "composition":
            made = dict(row["composition"])
            el = rng.choice(list(made))
            made[el] = max(1, made[el] + rng.choice([-2, -1, 1, 2, 3]))
            return None if made == row["composition"] else " ".join(f"{e} {num(n)}" for e, n in made.items())
        if key == "part":
            comp = rng.choice(list(row["main_parts"]))
            wrong = self._off(rng, row["main_parts"][comp], 3)
            return None if wrong == row["main_parts"][comp] or not 0 < wrong <= 100 else f"{comp} {num(wrong, 3)}"
        if key == "class":
            pool = [c for c in CLASSES if class_truth(row, c) is False]
        elif key == "bonds":
            pool = [b for b in BONDS if bond_truth(row, b) is False]
        elif key == "state":
            pool = [s for s in STATES if state_truth(row, s) is False]
        elif key == "basis":
            pool = [b for b in BASES if basis_truth(row, b) is False]
        elif key == "occurs_in":
            pool = [p for p in PLACES if self.place_truth(row, p) is False]
        elif key == "main_parts":
            pool = self.not_parts[row["name"]]
        else:
            return None
        return rng.choice(pool) if pool else None

    def _check_fact(self, rng: random.Random) -> tuple[str, str, str] | None:
        yes = rng.random() < 0.5
        for _ in range(40):  # a row may have no safe wrong value for a key: draw again, keep yes or no
            row = rng.choice(self.pure) if rng.random() < 0.85 else rng.choice(self.mixtures)
            if row["mixture"]:
                key = rng.choice(["main_parts", "main_parts", "part", "basis", "state", "occurs_in"])
            else:
                key = rng.choice(["formula", "formula", "class", "bonds", "state", "occurs_in", "occurs_in", "atoms",
                                  "elements", "molar_mass", "composition"])
            value = self._right(rng, row, key) if yes else self._wrong(rng, row, key)
            if value is not None:
                return f"check {row['name']} {key} {value}", "yes" if yes else "no", "yesno"
        return None

    def _check_calc(self, rng: random.Random) -> tuple[str, str, str] | None:
        subject, counts = self._subject(rng)
        element = NAME_OF[rng.choice(list(counts))]
        head = rng.choice(["element_count", "mass_percent"])
        if head == "element_count":
            right = counts[SYMBOL_OF[element]]
            wrong = max(1, right + rng.choice([-2, -1, 1, 2, 3]))
            if wrong == right:
                return None
            value, truth = (right, "yes") if rng.random() < 0.5 else (wrong, "no")
            return f"check element_count {subject} {element} {num(value)}", truth, "yesno"
        right = self.mass_percent(counts, element)
        wrong = self._off(rng, right, 4)
        if wrong == right or not 0 < wrong <= 100:
            return None
        value, truth = (right, "yes") if rng.random() < 0.5 else (wrong, "no")
        return f"check mass_percent {subject} {element} {num(value, 4)}", truth, "yesno"

    def _check_heavier(self, rng: random.Random) -> tuple[str, str, str] | None:
        pair = self._heavier_pair(rng)
        if pair is None:
            return None
        a, b = pair
        return f"check heavier_molecule {a['name']} {b['name']}", "yes" if a["molar_mass"] > b["molar_mass"] else "no", "yesno"


def gate(data_dir: Path = DATA) -> Substances:
    return Substances(data_dir)


# ---- build ----------------------------------------------------------------------------------------

_SECONDS = {"ysec": 1e-24, "zsec": 1e-21, "asec": 1e-18, "fsec": 1e-15, "psec": 1e-12, "nsec": 1e-9, "usec": 1e-6,
            "msec": 1e-3, "sec": 1.0, "minute": 60.0, "hour": 3600.0, "day": 86400.0, "year": 31557600.0,
            "kyear": 31557600.0e3, "Myear": 31557600.0e6, "Gyear": 31557600.0e9, "Tyear": 31557600.0e12,
            "Pyear": 31557600.0e15, "Eyear": 31557600.0e18, "Zyear": 31557600.0e21, "Yyear": 31557600.0e24}


def _mass_number(element) -> int:
    """The mass number that stands for an element without a standard weight: the source's bracket
    weight, unless its isotope table shows another isotope to be longer-lived beyond doubt (a
    half-life of at least twice its own uncertainty that exceeds the bracket isotope's by more
    than both uncertainties). The elements table uses the same rule."""
    bracket = int(round(element.atomic_weight))

    def seconds(i) -> float:
        return i.half_life * _SECONDS[i.half_life_unit]

    def spread(i) -> float:
        return (i.half_life_uncertainty or 0.0) * _SECONDS[i.half_life_unit]

    timed = sorted((i for i in element.isotopes if i.half_life is not None and i.half_life_unit in _SECONDS),
                   key=lambda i: (-seconds(i), i.mass_number))
    own = next((i for i in timed if i.mass_number == bracket), None)
    sure = next((i for i in timed if i.half_life_uncertainty is not None and seconds(i) >= 2 * spread(i)), None)
    if own is None or sure is None or sure is own:
        return bracket
    return sure.mass_number if seconds(sure) - spread(sure) > seconds(own) + spread(own) else bracket


def build_weights() -> dict[str, float | int]:
    """Symbol -> standard atomic weight from mendeleev, or a mass number (an integer) where there
    is no standard weight (:func:`_mass_number`). Checks :data:`ELEMENTS` on the way."""
    from mendeleev import get_all_elements  # only the build needs it

    out: dict[str, float | int] = {}
    for e in sorted(get_all_elements(), key=lambda e: e.atomic_number):
        if ELEMENTS[e.atomic_number - 1] != (e.symbol, e.name.lower()):
            raise ValueError(f"ELEMENTS disagrees with mendeleev at {e.atomic_number}: {e.symbol} {e.name.lower()}")
        out[e.symbol] = float(e.atomic_weight) if has_standard_weight(e.atomic_number) else _mass_number(e)
    return out


def build(data_dir: Path = DATA) -> dict:
    data_dir.mkdir(parents=True, exist_ok=True)
    weights = build_weights()
    (data_dir / "atomic_weights.json").write_text(json.dumps(weights, indent=1) + "\n", encoding="utf-8", newline="\n")
    rows = table_rows(weights)
    with (data_dir / "substances.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    g = gate(data_dir)
    recs = g.records()
    return {"elements": len(weights), "mass_number_only": sum(isinstance(w, int) for w in weights.values()),
            "pure": len(g.pure), "mixtures": len(g.mixtures), "record_lines": len(recs),
            "split_rows": sum(ln.text.split(". ")[0].endswith("_more") for ln in recs),
            "facts": len(g.facts()), "list_questions": len(g.lists),
            "longest_record_tokens": max(n_tokens(ln.text) for ln in recs)}


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--data", type=Path, default=DATA)
    s = sub.add_parser("sample")
    s.add_argument("--data", type=Path, default=DATA)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("-n", type=int, default=40)
    args = ap.parse_args()
    if args.cmd == "build":
        print(json.dumps(build(args.data), indent=1))
    else:
        g = gate(args.data)
        for ln in g.records()[:3]:
            print(ln.text)
        for ln in g.generate(random.Random(args.seed), args.n):
            print(ln.kind.ljust(6), ln.text)


if __name__ == "__main__":
    main()
