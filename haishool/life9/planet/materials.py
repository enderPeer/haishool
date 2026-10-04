"""Materials of the planet engine: species, reference properties, thermochemistry and transforms
(PLANET-SPEC section 2.4).

Every material is a mix of ``SPECIES`` given as mass fractions. Each species row carries its
provenance (:class:`Val`: ``tag`` is ``reference``, ``derived`` or ``new_rule`` and ``source``
says where the number comes from). The main sources are:

* CRC Handbook of Chemistry and Physics (97th ed., 2016): melting points, standard
  thermodynamic properties, molar heat capacities, Mohs hardness of the elements;
* NIST-JANAF Thermochemical Tables (Chase 1998, J. Phys. Chem. Ref. Data Monograph 9): the gases;
* Incropera, DeWitt, Bergman & Lavine, Fundamentals of Heat and Mass Transfer (6th ed., 2007),
  Tables A.1 and A.3: density, cp and conductivity of metals, rocks, brick, glass, sand, clay;
* Anthony, Bideaux, Bladh & Nichols, Handbook of Mineralogy (Mineralogical Society of America):
  mineral density and Mohs hardness;
* Atkinson & Meredith 1987 (Fracture Mechanics of Rock) and Ashby 2011 (Materials
  Selection in Mechanical Design, 4th ed.): fracture toughness;
* IT'IS Foundation tissue property database v4.1 (Hasgall et al. 2022): meat, fat, bone, skin;
* Tylecote 1992 (A History of Metallurgy, 2nd ed.), Pleiner 2000 (Iron in Archaeology), Rice 1987
  (Pottery Analysis), Boynton 1980 (Chemistry and Technology of Lime and Limestone): the practical
  onset temperatures of the transforms;
* Rudnick & Gao 2003 (Treatise on Geochemistry vol. 3, Table 3, upper continental crust): the
  crust abundances of Fe, Cu, Sn, S.

Rules (``new_rule``) are stated where they are made. ``CLASS`` is a label for senses and stats;
no rule reads it. Nothing here names a recipe: a transform is a chemical or physical change with
its conditions, and :func:`props` turns any composition into physical properties.
:func:`provenance` lists the tag and source of every number here that sets up a world.

Rules crafting must keep when it reads these tables:

* ``hardness`` is the scratch hardness of the grains (it rules abrasion). Hammering, knapping,
  chopping, digging and striking use ``tool_hardness = hardness x (1 - granular)``: loose sand or
  ash has no cohesion and cannot act as a solid tool. The resistance of a deposit to digging is
  the same quantity of the deposit species.
* A transform applies when its temperature and atmosphere conditions hold (:func:`atmosphere_ok`).
  A fuel item above its ``IGNITION_K`` in a fire burns, unless a transform of its own species
  applies there (smothered wood chars instead of burning).
* :func:`burn` takes fuels only; other items in a fire heat up, transform, cook or stay.

Units are SI (kg, m, K, J, Pa) unless a name says otherwise; ``MOHS`` is the Mohs scale and
``TOUGHNESS`` is in MPa m^0.5.
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass

import torch

from haishool.life9.planet.constants import SOLAR_LOG_EPS, Z_SUN, atomic_weights, molar_mass
from haishool.truth.formula import parse

REFERENCE, DERIVED, NEW_RULE, CHAIN = "reference", "derived", "new_rule", "chain"
TAGS = (CHAIN, DERIVED, REFERENCE, NEW_RULE)


class Val(float):
    """A number with its provenance: it is a ``float``; ``.tag`` and ``.source`` say where it comes from."""

    def __new__(cls, value: float, tag: str = REFERENCE, source: str = ""):
        if tag not in TAGS:
            raise ValueError(f"unknown provenance tag {tag!r}")
        x = super().__new__(cls, value)
        x.tag, x.source = tag, source
        return x

    def __reduce__(self):
        return (Val, (float(self), self.tag, self.source))


# ============================================================================== species
SPECIES: tuple[str, ...] = (
    # rocks and minerals (the ground)
    "basalt", "granite", "flint", "sandstone", "limestone", "clay", "sand", "salt",
    "hematite", "magnetite", "malachite", "cassiterite", "native_copper", "pyrite",
    # organics
    "wood", "plant_fiber", "resin", "meat", "fat", "bone", "hide", "charcoal", "ash",
    # products
    "ceramic", "lime", "copper", "tin", "bronze", "iron", "steel", "glass",
)
S = len(SPECIES)
IDX = {s: i for i, s in enumerate(SPECIES)}
#: species that occur in ground deposits (``globe.make_deposits``); pyrite is added to the
#: spec's list because crafting's spark ignition needs it (flint + pyrite, section 2.11)
CRUST_SPECIES: tuple[str, ...] = SPECIES[:14]
CLASSES = ("stone", "ore", "clay", "wood", "organic", "metal", "product")
CLASS: dict[str, str] = {
    **{s: "stone" for s in ("basalt", "granite", "flint", "sandstone", "limestone", "sand", "salt")},
    "clay": "clay",
    **{s: "ore" for s in ("hematite", "magnetite", "malachite", "cassiterite", "native_copper", "pyrite")},
    "wood": "wood",
    **{s: "organic" for s in ("plant_fiber", "resin", "meat", "fat", "bone", "hide")},
    **{s: "product" for s in ("charcoal", "ash", "ceramic", "lime", "glass")},
    **{s: "metal" for s in ("copper", "tin", "bronze", "iron", "steel")},
}
#: temperature used for "never at 1 atm below any fire" (no melting point: the species chars,
#: decomposes or calcines first; or it is not a fuel). A sentinel, not a property.
NEVER_K = Val(5000.0, NEW_RULE, "sentinel: no melting or ignition at 1 atm below any fire temperature")


def _never(why: str) -> Val:
    """A NEVER_K sentinel row (tagged as NEVER_K, new_rule) with the reason the species has no value."""
    return Val(float(NEVER_K), NEVER_K.tag, f"sentinel NEVER_K: {why}")


def _table(rows: dict) -> dict[str, Val]:
    missing = set(SPECIES) ^ set(rows)
    if missing:
        raise KeyError(f"table rows do not match SPECIES: {sorted(missing)}")
    return {s: rows[s] if isinstance(rows[s], Val) else Val(*rows[s]) for s in SPECIES}


R, D, N = REFERENCE, DERIVED, NEW_RULE
_INC1 = "Incropera et al. 2007, Table A.1"
_INC3 = "Incropera et al. 2007, Table A.3"
_ITIS = "IT'IS tissue database v4.1 (Hasgall et al. 2022)"
_HOM = "Handbook of Mineralogy (Anthony et al.)"


def _per_kg(formula: str, molar: float) -> float:
    """A molar quantity (per mol) as a quantity per kg of the formula."""
    return molar / molar_mass(formula)


# ============================================================================== composition
#: average elemental composition of protein by mass (C 50-55 %, H 6.5-7.3 %, N 15-18 %
#: (Kjeldahl factor 6.25 -> 16 %), O 20-24 %, S 0.5-2 %; Jones 1931, USDA Circular 183)
PROTEIN = {"C": 0.53, "H": 0.07, "O": 0.23, "N": 0.16, "S": 0.01}
PROTEIN_SOURCE = (REFERENCE, "protein C 53, H 7, O 23, N 16, S 1 % by mass, inside C 50-55, H 6.5-7.3, N 15-18 "
                             "(Kjeldahl 6.25), O 20-24, S 0.5-2 % (Jones 1931, USDA Circular 183)")
#: idealised formulas: dry wood C6H9O4 (C 49.7, H 6.2, O 44.1 % by mass, the typical ultimate
#: analysis of dry hardwood, Jenkins et al. 1998, Fuel Proc. Tech. 54, 17); cellulose C6H10O5;
#: abietic acid C20H30O2 (the main rosin acid); tripalmitin C51H98O6 (an animal fat)
WOOD, CELLULOSE, ROSIN, LIPID = "C6H9O4", "C6H10O5", "C20H30O2", "C51H98O6"
ORGANIC = (WOOD, CELLULOSE, ROSIN, LIPID, "C")


def _oxides(wt: dict[str, float]) -> tuple:
    total = sum(wt.values())
    return tuple((f, v / total) for f, v in wt.items())


def _moles(mol: dict[str, float]) -> tuple:
    mass = {f: n * molar_mass(f) for f, n in mol.items()}
    total = sum(mass.values())
    return tuple((f, m / total) for f, m in mass.items())


#: wood ash: CaCO3 and K2CO3, 3:1 by mass (ash made at 500-600 C is mostly calcium and potassium
#: carbonates; Misra, Ragland & Baker 1993, Biomass & Bioenergy 4, 103). new_rule: two carbonates only
ASH_MAKEUP = (("CaCO3", 0.75), ("K2CO3", 0.25))
ASH_MAKEUP_SOURCE = (NEW_RULE, "wood ash as CaCO3 75 % + K2CO3 25 % by mass: ash made at 500-600 C is mostly Ca and K "
                               "carbonates, Ca the larger (Misra, Ragland & Baker 1993); Mg, P and the rest left out")
#: ash content of dry wood: 1 % (hardwoods 0.3-1.5 %, Misra et al. 1993)
WOOD_ASH_FRACTION = Val(0.01, REFERENCE, "dry hardwood ash content 0.3-1.5 % (Misra et al. 1993)")
#: glass batch: 1 kg wood ash per kg sand. With the 3:1 carbonate ash the melt is SiO2 62.9, CaO 26.4,
#: K2O 10.7 % by mass: richer in lime and silica and poorer in potash than Central European forest
#: glass (SiO2 about 50-60, CaO 15-25, K2O 10-20 %; Wedepohl 2003, Glas in Antike und Mittelalter).
#: No batch ratio of this ash reaches that range (CaO:K2O stays 2.5:1); 1:1 is the nearest (more ash
#: pushes CaO further out, less ash leaves too little flux), so the onset is raised (sand_ash_to_glass)
GLASS_ASH_PER_SAND = Val(1.0, NEW_RULE, "1 kg wood ash per kg sand: melt SiO2 62.9, CaO 26.4, K2O 10.7 wt%, more lime and "
                                        "less potash than forest glass (SiO2 50-60, CaO 15-25, K2O 10-20 %, Wedepohl 2003)")
#: alloy compositions by mass: tin bronze 89/11 (the Incropera 'phosphor gear bronze' row; ancient
#: bronzes 8-14 % Sn, Tylecote 1992); eutectoid steel 0.8 % C (plain carbon steels 0.05-2.1 % C)
BRONZE_SN = Val(0.11, REFERENCE, "tin bronze Cu 89, Sn 11 (Incropera Table A.1; ancient bronzes 8-14 % Sn, Tylecote 1992)")
STEEL_C = Val(0.008, NEW_RULE, "eutectoid carbon steel, 0.8 % C (plain carbon steel range 0.05-2.1 %)")


def _glass_oxides() -> tuple:
    sand, ash = 1.0, float(GLASS_ASH_PER_SAND)
    cao = ash * 0.75 / molar_mass("CaCO3") * molar_mass("CaO")
    k2o = ash * 0.25 / molar_mass("K2CO3") * molar_mass("K2O")
    total = sand + cao + k2o
    return (("SiO2", sand / total), ("CaO", cao / total), ("K2O", k2o / total))


#: species -> components with mass fractions. A component is a formula or ``protein``.
MAKEUP: dict[str, tuple] = {
    # all-MORB mean (Gale et al. 2013, G-cubed 14, 489): the oxides above 0.2 % except TiO2 (1.68 %;
    # Ti is not in ELEMENTS), plus K2O 0.16 %, renormalised
    "basalt": _oxides({"SiO2": 50.47, "Al2O3": 14.70, "FeO": 10.43, "MgO": 7.58, "CaO": 11.39, "Na2O": 2.79, "K2O": 0.16}),
    # average granite (Le Maitre 1976, J. Petrology 17, 589): the oxides above 0.2 % except TiO2
    # (0.31 %; Ti is not in ELEMENTS) and the volatiles, renormalised
    "granite": _oxides({"SiO2": 71.30, "Al2O3": 14.32, "Fe2O3": 1.21, "FeO": 1.64, "MgO": 0.71, "CaO": 1.84,
                        "Na2O": 3.68, "K2O": 4.07}),
    "flint": (("SiO2", 1.0),),          # microcrystalline quartz (real flint 95-99 % SiO2)
    "sandstone": (("SiO2", 1.0),),      # quartz arenite (> 90 % quartz, Pettijohn 1975)
    "limestone": (("CaCO3", 1.0),),
    "clay": (("Al2Si2O5(OH)4", 1.0),),  # kaolinite
    "sand": (("SiO2", 1.0),),
    "salt": (("NaCl", 1.0),),
    "hematite": (("Fe2O3", 1.0),),
    "magnetite": (("Fe3O4", 1.0),),
    "malachite": (("Cu2CO3(OH)2", 1.0),),
    "cassiterite": (("SnO2", 1.0),),
    "native_copper": (("Cu", 1.0),),
    "pyrite": (("FeS2", 1.0),),
    "wood": ((WOOD, 1.0 - float(WOOD_ASH_FRACTION)),
             *((f, x * float(WOOD_ASH_FRACTION)) for f, x in ASH_MAKEUP)),
    "plant_fiber": ((CELLULOSE, 1.0),),
    "resin": ((ROSIN, 1.0),),
    # lean game muscle: water 73.5 %, protein 21.5 %, fat 5 % (USDA FoodData Central, raw venison/beef)
    "meat": (("H2O", 0.735), ("protein", 0.215), (LIPID, 0.05)),
    "fat": ((LIPID, 1.0),),
    # fresh bone: mineral (hydroxyapatite) 60 %, collagen 22 %, marrow fat 6 %, water 12 % (Currey 2002, Bones)
    "bone": (("Ca5(PO4)3(OH)", 0.60), ("protein", 0.22), (LIPID, 0.06), ("H2O", 0.12)),
    # fresh hide: water 64 %, protein 33 %, fat 3 % (Covington 2009, Tanning Chemistry)
    "hide": (("H2O", 0.64), ("protein", 0.33), (LIPID, 0.03)),
    "charcoal": (("C", 1.0),),           # new_rule: pure carbon (real charcoal 75-95 % fixed carbon)
    "ash": ASH_MAKEUP,
    # fired kaolinite: mullite + silica (3 Al2Si2O5(OH)4 -> Al6Si2O13 + 4 SiO2 + 6 H2O)
    "ceramic": _moles({"Al6Si2O13": 1.0, "SiO2": 4.0}),
    "lime": (("CaO", 1.0),),
    "copper": (("Cu", 1.0),),
    "tin": (("Sn", 1.0),),
    "bronze": (("Cu", 1.0 - float(BRONZE_SN)), ("Sn", float(BRONZE_SN))),
    "iron": (("Fe", 1.0),),              # bloomery (wrought) iron, idealised as pure Fe
    "steel": (("Fe", 1.0 - float(STEEL_C)), ("C", float(STEEL_C))),
    "glass": _glass_oxides(),
}
_FORMULA_HOM = "the mineral's formula (Handbook of Mineralogy)"
#: provenance of each MAKEUP row: (tag, source)
MAKEUP_SOURCE: dict[str, tuple[str, str]] = {
    "basalt": (REFERENCE, "all-MORB mean (Gale et al. 2013, G-cubed 14, 489), oxides above 0.2 % except TiO2 "
                          "(Ti not tracked) plus K2O, renormalised"),
    "granite": (REFERENCE, "average granite (Le Maitre 1976, J. Petrology 17, 589), oxides above 0.2 % except TiO2 "
                           "(Ti not tracked) and volatiles, renormalised"),
    "flint": (NEW_RULE, "pure SiO2 (real flint is 95-99 % SiO2)"),
    "sandstone": (NEW_RULE, "pure SiO2 (quartz arenite, > 90 % quartz, Pettijohn 1975)"),
    "limestone": (NEW_RULE, "pure CaCO3 (calcite; real limestones carry clay, dolomite and silica)"),
    "clay": (NEW_RULE, "pure kaolinite Al2Si2O5(OH)4 (real clays mix kaolinite, illite, quartz)"),
    "sand": (NEW_RULE, "pure SiO2 (quartz sand)"),
    **{s: (REFERENCE, _FORMULA_HOM) for s in ("salt", "hematite", "magnetite", "malachite", "cassiterite",
                                            "native_copper", "pyrite")},
    "wood": (DERIVED, "dry wood C6H9O4 (C 49.7, H 6.2, O 44.1 %, ultimate analysis of dry hardwood, Jenkins et al. "
                      "1998, Fuel Proc. Tech. 54, 17) with WOOD_ASH_FRACTION of ASH_MAKEUP"),
    "plant_fiber": (NEW_RULE, "pure cellulose C6H10O5"),
    "resin": (NEW_RULE, "pure abietic acid C20H30O2, the main rosin acid"),
    "meat": (REFERENCE, "lean game muscle: water 73.5 %, protein 21.5 %, fat 5 % (USDA FoodData Central, raw venison/beef)"),
    "fat": (NEW_RULE, "pure tripalmitin C51H98O6 (an animal fat)"),
    "bone": (REFERENCE, "fresh bone: hydroxyapatite 60 %, collagen 22 %, marrow fat 6 %, water 12 % (Currey 2002, Bones)"),
    "hide": (REFERENCE, "fresh hide: water 64 %, protein 33 %, fat 3 % (Covington 2009, Tanning Chemistry)"),
    "charcoal": (NEW_RULE, "pure carbon (real charcoal 75-95 % fixed carbon)"),
    "ash": ASH_MAKEUP_SOURCE,
    "ceramic": (DERIVED, "fired kaolinite: 3 Al2Si2O5(OH)4 -> Al6Si2O13 (mullite) + 4 SiO2 + 6 H2O"),
    "lime": (NEW_RULE, "pure CaO"),
    "copper": (REFERENCE, "Cu"),
    "tin": (REFERENCE, "Sn"),
    "bronze": (BRONZE_SN.tag, BRONZE_SN.source),
    "iron": (NEW_RULE, "bloomery (wrought) iron idealised as pure Fe"),
    "steel": (STEEL_C.tag, STEEL_C.source),
    "glass": (DERIVED, "the sand_ash_to_glass melt (GLASS_ASH_PER_SAND, ASH_MAKEUP)"),
}

ELEMENTS: tuple[str, ...] = ("H", "C", "N", "O", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "K", "Ca", "Fe", "Cu", "Sn")
GASES: tuple[str, ...] = ("O2", "CO2", "H2O", "CO")


def formula_elements(formula: str) -> dict[str, float]:
    """Element mass fractions of a formula (or of ``protein``)."""
    if formula == "protein":
        return dict(PROTEIN)
    w = atomic_weights()
    counts = parse(formula)
    total = sum(w[e] * n for e, n in counts.items())
    return {e: w[e] * n / total for e, n in counts.items()}


def species_elements(species: str) -> dict[str, float]:
    """Element mass fractions of one species (from its :data:`MAKEUP`)."""
    out: dict[str, float] = {}
    for comp, x in MAKEUP[species]:
        for e, f in formula_elements(comp).items():
            out[e] = out.get(e, 0.0) + x * f
    return out


#: [S][E] element mass fractions of each species (rows sum to 1)
ELEMENT_FRACTION: tuple[tuple[float, ...], ...] = tuple(
    tuple(species_elements(s).get(e, 0.0) for e in ELEMENTS) for s in SPECIES)
GAS_ELEMENT_FRACTION: tuple[tuple[float, ...], ...] = tuple(
    tuple(formula_elements(g).get(e, 0.0) for e in ELEMENTS) for g in GASES)
GAS_MOLAR_MASS = {g: molar_mass(g) for g in GASES}

# ============================================================================== property tables
DENSITY = _table({  # kg/m^3
    "basalt": (2900.0, R, "basalt 2.7-3.1 g/cm3, mean 2.9 (Schon 2015, Physical Properties of Rocks)"),
    "granite": (2630.0, R, f"Barre granite ({_INC3})"),
    "flint": (2600.0, R, "chert/flint about 2.6 g/cm3 (Schon 2015, Physical Properties of Rocks)"),
    "sandstone": (2150.0, R, f"Berea sandstone ({_INC3})"),
    "limestone": (2320.0, R, f"Salem limestone ({_INC3})"),
    "clay": (1460.0, R, f"clay ({_INC3})"),
    "sand": (1515.0, R, f"sand, bulk ({_INC3})"),
    "salt": (2165.0, R, "halite NaCl 2.165 g/cm3 (CRC Handbook)"),
    "hematite": (5260.0, R, f"hematite 5.26 g/cm3 ({_HOM})"),
    "magnetite": (5175.0, R, f"magnetite 5.175 g/cm3 ({_HOM})"),
    "malachite": (4000.0, R, f"malachite 3.6-4.05 g/cm3 ({_HOM})"),
    "cassiterite": (6990.0, R, f"cassiterite 6.98-7.01 g/cm3 ({_HOM})"),
    "native_copper": (8933.0, R, f"pure copper ({_INC1})"),
    "pyrite": (5010.0, R, f"pyrite 5.01 g/cm3 ({_HOM})"),
    "wood": (545.0, R, f"oak, cross grain ({_INC3})"),
    "plant_fiber": (1500.0, R, "flax fibre 1.5 g/cm3 (Bledzki & Gassan 1999, Prog. Polym. Sci. 24, 221)"),
    "resin": (1070.0, R, "rosin 1.07-1.09 g/cm3 (Merck Index)"),
    "meat": (1090.0, R, f"skeletal muscle ({_ITIS})"),
    "fat": (911.0, R, f"fat ({_ITIS})"),
    "bone": (1908.0, R, f"cortical bone ({_ITIS})"),
    "hide": (1109.0, R, f"skin ({_ITIS})"),
    "charcoal": (400.0, R, "hardwood charcoal apparent density 0.3-0.5 g/cm3 (Antal & Gronli 2003, Ind. Eng. Chem. Res. 42, 1619)"),
    "ash": (600.0, R, "wood ash bulk density 0.5-0.7 g/cm3 (Etiegni & Campbell 1991, Bioresource Tech. 37, 173)"),
    "ceramic": (1920.0, R, f"common brick ({_INC3})"),
    "lime": (2320.0 * molar_mass("CaO") / molar_mass("CaCO3"), D, "limestone density x mass yield of CaCO3 -> CaO (a burnt lump keeps its volume; quicklime lumps 1.3-1.6 g/cm3, Boynton 1980)"),
    "copper": (8933.0, R, f"pure copper ({_INC1})"),
    "tin": (7310.0, R, f"tin ({_INC1})"),
    "bronze": (8780.0, R, f"phosphor gear bronze 89 Cu 11 Sn ({_INC1})"),
    "iron": (7870.0, R, f"pure iron ({_INC1})"),
    "steel": (7854.0, R, f"plain carbon steel ({_INC1})"),
    "glass": (2500.0, R, f"plate (soda-lime) glass ({_INC3})"),
})

MOHS = _table({
    "basalt": (6.5, N, "hardest abundant phases of basalt: plagioclase 6-6.5, olivine 6.5-7, augite 5.5-6.5 (Handbook of "
                       "Mineralogy); 6.5, the top of the 6-6.5 plagioclase matrix, chosen so that a basalt hammer knaps "
                       "flint under the x 1.1 rule (PLANET-SPEC 2.11, 4)"),
    "granite": (6.5, R, "quartz 7 in feldspar 6-6.5 (Handbook of Mineralogy)"),
    "flint": (7.0, R, "microcrystalline quartz (chalcedony) 6.5-7; quartz is Mohs 7"),
    "sandstone": (6.0, N, "quartz grains (7) in a weaker cement: scratch hardness below quartz (estimate)"),
    "limestone": (3.0, R, "calcite 3 (Mohs scale mineral)"),
    "clay": (2.0, R, f"kaolinite 2-2.5 ({_HOM})"),
    "sand": (7.0, R, "quartz grains 7 (Mohs scale mineral)"),
    "salt": (2.5, R, f"halite 2.5 ({_HOM})"),
    "hematite": (5.5, R, f"hematite 5-6 ({_HOM})"),
    "magnetite": (5.75, R, f"magnetite 5.5-6 ({_HOM})"),
    "malachite": (3.75, R, f"malachite 3.5-4 ({_HOM})"),
    "cassiterite": (6.5, R, f"cassiterite 6-7 ({_HOM})"),
    "native_copper": (2.75, R, f"native copper 2.5-3 ({_HOM})"),
    "pyrite": (6.25, R, f"pyrite 6-6.5 ({_HOM})"),
    "wood": (1.5, N, "dry hardwood is indented by a fingernail (2.5); the Mohs scale is not defined for wood (estimate)"),
    "plant_fiber": (1.0, N, "softer than wood (estimate)"),
    "resin": (2.0, R, "amber/rosin about 2-2.5 (gemmological tables)"),
    "meat": (0.3, N, "soft tissue, far below talc (estimate)"),
    "fat": (0.2, N, "soft tissue, far below talc (estimate)"),
    "bone": (2.5, R, "cortical bone Vickers about 50 HV (Currey 2002, Bones), about Mohs 2.5"),
    "hide": (0.5, N, "soft tissue, below talc (estimate)"),
    "charcoal": (1.0, N, "charcoal marks paper like graphite (1-2) (estimate)"),
    "ash": (1.0, N, "loose powder (estimate)"),
    "ceramic": (4.0, R, "low-fired earthenware sherds Mohs 3-5 (Rice 1987, Pottery Analysis)"),
    "lime": (3.0, N, "porous quicklime CaO lump, about 3 (estimate)"),
    "copper": (3.0, R, "copper 3.0 (CRC Handbook, Mohs hardness of the elements)"),
    "tin": (1.5, R, "tin 1.5 (CRC Handbook, Mohs hardness of the elements)"),
    "bronze": (3.75, N, "cast tin bronze HB 90-110 (ASM C90700), about Mohs 3.5-4 (conversion estimate)"),
    "iron": (4.0, R, "iron 4.0 (CRC Handbook, Mohs hardness of the elements)"),
    "steel": (5.5, N, "as-forged 0.8 % C pearlitic steel HB 250-300, about Mohs 5-5.5 (conversion estimate)"),
    "glass": (5.5, R, "soda-lime glass 5.5 (standard Mohs comparison tables)"),
})

#: mass fraction that is loose, cohesionless grains or powder (0 or 1 per species). A granular
#: species scratches with its grains' hardness but cannot act as a solid hammer, blade or digging
#: edge: :func:`props` gives ``tool_hardness = hardness x (1 - granular)`` for those uses.
_COHERENT = "coherent solid (rock, crystal, lump, metal or tissue)"
GRANULAR = _table({
    **{s: (0.0, R, _COHERENT) for s in SPECIES},
    "sand": (1.0, R, "clean sand is cohesionless, c' = 0 (Terzaghi, Peck & Mesri 1996, Soil Mechanics in "
                     "Engineering Practice)"),
    "ash": (1.0, R, "wood ash is a fine loose powder (Etiegni & Campbell 1991, Bioresource Tech. 37, 173)"),
    "clay": (0.0, R, "clay is a cohesive soil, c' > 0, and a worked lump holds together (Terzaghi, Peck & Mesri 1996)"),
})

TOUGHNESS = _table({  # MPa m^0.5, mode-I fracture toughness
    "basalt": (2.6, R, "basalts 1.8-3.0 (Atkinson & Meredith 1987, compiled rock K_IC data)"),
    "granite": (1.5, R, "Westerly granite 1.4-1.9 (Atkinson & Meredith 1987)"),
    "flint": (1.3, R, "flint/chert 1.2-1.5 (Domanski, Webb & Boland 1994, Archaeometry 36, 177)"),
    "sandstone": (0.6, R, "sandstones 0.2-1.0 (Atkinson & Meredith 1987)"),
    "limestone": (1.0, R, "limestones 0.7-1.3 (Atkinson & Meredith 1987)"),
    "clay": (0.1, N, "dry unfired clay lump (estimate)"),
    "sand": (0.0, N, "cohesionless grains"),
    "salt": (0.3, N, "halite crystal 0.2-0.4 (estimate)"),
    "hematite": (1.5, N, "oxide ceramic range 1-2 (estimate)"),
    "magnetite": (1.5, N, "oxide ceramic range 1-2 (estimate)"),
    "malachite": (0.8, N, "soft carbonate mineral (estimate)"),
    "cassiterite": (1.5, N, "oxide ceramic range 1-2 (estimate)"),
    "native_copper": (100.0, R, "pure annealed copper about 100 (Ashby 2011, K_IC chart)"),
    "pyrite": (1.0, N, "brittle sulfide (estimate)"),
    "wood": (3.0, N, "between across-grain 0.5-1 and along-grain 5-10 (Ashby 2011)"),
    "plant_fiber": (1.0, N, "fibre bundle (estimate)"),
    "resin": (0.5, R, "glassy polymers 0.5-1 (Ashby 2011)"),
    "meat": (0.5, N, "soft tissue, tears rather than cracks (estimate)"),
    "fat": (0.2, N, "soft tissue (estimate)"),
    "bone": (5.0, R, "cortical bone 2-7 (Ritchie, Buehler & Hansma 2009, Physics Today 62(6), 41)"),
    "hide": (3.0, N, "skin tears tough (estimate)"),
    "charcoal": (0.2, N, "friable (estimate)"),
    "ash": (0.0, N, "loose powder"),
    "ceramic": (1.0, R, "brick and earthenware about 1-2 (Ashby 2011)"),
    "lime": (0.3, N, "porous quicklime (estimate)"),
    "copper": (100.0, R, "pure copper about 100 (Ashby 2011)"),
    "tin": (30.0, N, "ductile soft metal (estimate)"),
    "bronze": (40.0, R, "copper alloys 30-90 (Ashby 2011); cast tin bronze near the low end"),
    "iron": (80.0, R, "low-carbon iron and steel 41-82 (Ashby 2011)"),
    "steel": (40.0, R, "high-carbon steel 27-92, as-forged eutectoid near 40 (Ashby 2011)"),
    "glass": (0.75, R, "soda-lime glass 0.55-0.77 (Ashby 2011)"),
})

_CONCHOIDAL = {"flint", "basalt", "resin", "glass"}
BRITTLE = _table({s: (1.0 if s in _CONCHOIDAL else 0.0, R,
                      "conchoidal fracture, knappable (Whittaker 1994, Flintknapping: flint, fine-grained basalt, glass; amber/rosin)"
                      if s in _CONCHOIDAL else "no conchoidal fracture (granular, cleavage, ductile or soft)")
                  for s in SPECIES})

MELT_K = _table({
    "basalt": (1473.0, R, "tholeiitic basalt liquidus about 1200 C at 1 atm (eruption temperatures 1100-1250 C, USGS)"),
    "granite": (1488.0, R, "dry granite melts about 1215-1260 C (standard petrology references)"),
    "flint": (1986.0, R, "SiO2 (cristobalite) melts at 1713 C (CRC Handbook)"),
    "sandstone": (1986.0, R, "quartz arenite: SiO2 1713 C (CRC Handbook)"),
    "limestone": _never("calcines (CaCO3 -> CaO + CO2) before it melts; melts only under CO2 pressure (1330 C at 102.5 atm, CRC Handbook)"),
    "clay": _never("kaolinite dehydroxylates near 800 K and fires to ceramic before it melts (Rice 1987)"),
    "sand": (1986.0, R, "SiO2 1713 C (CRC Handbook)"),
    "salt": (1073.8, R, "NaCl 800.7 C (CRC Handbook)"),
    "hematite": (1838.0, R, "Fe2O3 1565 C, decomposes (CRC Handbook)"),
    "magnetite": (1870.0, R, "Fe3O4 1597 C (CRC Handbook)"),
    "malachite": _never("decomposes near 600 K to CuO + CO2 + H2O before it melts (CRC Handbook)"),
    "cassiterite": (1903.0, R, "SnO2 1630 C (CRC Handbook)"),
    "native_copper": (1357.8, R, "Cu 1357.8 K (data/truth-v5/elements.jsonl, mendeleev)"),
    "pyrite": _never("decomposes near 900 K to pyrrhotite + S before it melts (CRC Handbook)"),
    "wood": _never("chars, does not melt"),
    "plant_fiber": _never("chars, does not melt"),
    "resin": (353.0, R, "rosin softening point 70-80 C (ring and ball)"),
    "meat": _never("dries and chars, does not melt"),
    "fat": (318.0, R, "beef tallow melts 40-50 C"),
    "bone": _never("chars and calcines, does not melt below any fire"),
    "hide": _never("dries and chars, does not melt"),
    "charcoal": _never("carbon does not melt at 1 atm, it sublimes near 4098 K (data/truth-v5/elements.jsonl)"),
    "ash": _never("carbonate ash fluxes silica (sand_ash_to_glass) rather than melting alone"),
    "ceramic": (2023.0, R, "fired kaolin, pyrometric cone equivalent 33-35 (1743-1785 C)"),
    "lime": (2886.0, R, "CaO 2613 C (CRC Handbook)"),
    "copper": (1357.8, R, "Cu 1357.8 K (data/truth-v5/elements.jsonl, mendeleev)"),
    "tin": (505.08, R, "Sn 505.08 K (data/truth-v5/elements.jsonl, mendeleev)"),
    "bronze": (1273.0, R, "tin bronze C90700 liquidus about 1000 C (ASM Handbook vol. 2; Cu-Sn diagram, ASM vol. 3)"),
    "iron": (1811.2, R, "Fe 1811.2 K (data/truth-v5/elements.jsonl, mendeleev)"),
    "steel": (1753.0, R, "Fe-C liquidus at 0.8 % C about 1480 C (ASM Handbook vol. 3)"),
    "glass": (1000.0, R, "softening point of soda/potash-lime glass about 700-730 C (viscosity 10^6.6 Pa s)"),
})

CP = _table({  # J/(kg K) near 298 K
    "basalt": (840.0, R, "basalt 0.84-0.90 kJ/(kg K) (Waples & Waples 2004, Nat. Resour. Res. 13, 97)"),
    "granite": (775.0, R, f"Barre granite ({_INC3})"),
    "flint": (_per_kg("SiO2", 44.4), D, "quartz Cp 44.4 J/(mol K) (CRC) / molar mass"),
    "sandstone": (745.0, R, f"Berea sandstone ({_INC3})"),
    "limestone": (810.0, R, f"Salem limestone ({_INC3})"),
    "clay": (880.0, R, f"clay ({_INC3})"),
    "sand": (800.0, R, f"sand ({_INC3})"),
    "salt": (_per_kg("NaCl", 50.5), D, "NaCl Cp 50.5 J/(mol K) (CRC) / molar mass"),
    "hematite": (_per_kg("Fe2O3", 103.9), D, "Fe2O3 Cp 103.9 J/(mol K) (CRC) / molar mass"),
    "magnetite": (_per_kg("Fe3O4", 143.4), D, "Fe3O4 Cp 143.4 J/(mol K) (CRC) / molar mass"),
    "malachite": (_per_kg("Cu2CO3(OH)2", 2 * 26.0 + 7.5 + 5 * 16.7 + 2 * 9.6), D,
                  "Kopp's rule (atomic contributions Cu 26, C 7.5, O 16.7, H 9.6 J/(mol K); Perry's Chemical Engineers' Handbook)"),
    "cassiterite": (_per_kg("SnO2", 52.6), D, "SnO2 Cp 52.6 J/(mol K) (CRC) / molar mass"),
    "native_copper": (385.0, R, f"pure copper ({_INC1})"),
    "pyrite": (_per_kg("FeS2", 62.2), D, "FeS2 Cp 62.2 J/(mol K) (CRC) / molar mass"),
    "wood": (103.1 + 3.867 * 298.15, D, "dry wood cp0 = 0.1031 + 0.003867 T kJ/(kg K) at 298 K (Wood Handbook FPL-GTR-190, 2010)"),
    "plant_fiber": (1300.0, N, "cellulose about 1.2-1.35 kJ/(kg K) (estimate)"),
    "resin": (1500.0, N, "organic solids 1.2-1.8 kJ/(kg K) (estimate)"),
    "meat": (3421.0, R, f"skeletal muscle ({_ITIS})"),
    "fat": (2348.0, R, f"fat ({_ITIS})"),
    "bone": (1313.0, R, f"cortical bone ({_ITIS})"),
    "hide": (3391.0, R, f"skin ({_ITIS})"),
    "charcoal": (_per_kg("C", 8.517), D, "graphite Cp 8.517 J/(mol K) (CRC) / molar mass"),
    "ash": (0.75 * _per_kg("CaCO3", 83.47) + 0.25 * _per_kg("K2CO3", 114.4), D,
            "Neumann-Kopp: CaCO3 Cp 83.47, K2CO3 114.4 J/(mol K) (CRC), mass-weighted"),
    "ceramic": (835.0, R, f"common brick ({_INC3})"),
    "lime": (_per_kg("CaO", 42.0), D, "CaO Cp 42.0 J/(mol K) (CRC) / molar mass"),
    "copper": (385.0, R, f"pure copper ({_INC1})"),
    "tin": (227.0, R, f"tin ({_INC1})"),
    "bronze": (355.0, R, f"phosphor gear bronze ({_INC1})"),
    "iron": (447.0, R, f"pure iron ({_INC1})"),
    "steel": (434.0, R, f"plain carbon steel ({_INC1})"),
    "glass": (750.0, R, f"plate glass ({_INC3})"),
})

CONDUCTIVITY = _table({  # W/(m K)
    "basalt": (1.69, R, "basalt mean 1.69 (Clauser & Huenges 1995, AGU Ref. Shelf 3)"),
    "granite": (2.79, R, f"Barre granite ({_INC3})"),
    "flint": (3.0, N, "within the chert range 1.4-5.7 (Clauser & Huenges 1995) (estimate)"),
    "sandstone": (2.90, R, f"Berea sandstone ({_INC3})"),
    "limestone": (2.15, R, f"Salem limestone ({_INC3})"),
    "clay": (1.3, R, f"clay ({_INC3})"),
    "sand": (0.27, R, f"sand ({_INC3})"),
    "salt": (6.0, R, "rock salt 5.4-7 (Clauser & Huenges 1995)"),
    "hematite": (11.28, R, "hematite (Horai 1971, J. Geophys. Res. 76, 1278)"),
    "magnetite": (5.10, R, "magnetite (Horai 1971)"),
    "malachite": (2.0, N, "carbonate mineral (estimate)"),
    "cassiterite": (30.0, N, "rutile-structure oxide (estimate)"),
    "native_copper": (401.0, R, f"pure copper ({_INC1})"),
    "pyrite": (19.21, R, "pyrite (Horai 1971)"),
    "wood": (0.17, R, f"oak, cross grain ({_INC3})"),
    "plant_fiber": (0.2, N, "cellulose solid (estimate)"),
    "resin": (0.15, N, "organic glass (estimate)"),
    "meat": (0.49, R, f"skeletal muscle ({_ITIS})"),
    "fat": (0.21, R, f"fat ({_ITIS})"),
    "bone": (0.32, R, f"cortical bone ({_ITIS})"),
    "hide": (0.37, R, f"skin ({_ITIS})"),
    "charcoal": (0.1, N, "porous char 0.05-0.1 (estimate)"),
    "ash": (0.1, N, "loose powder (estimate)"),
    "ceramic": (0.72, R, f"common brick ({_INC3})"),
    "lime": (0.5, N, "porous quicklime (estimate)"),
    "copper": (401.0, R, f"pure copper ({_INC1})"),
    "tin": (66.6, R, f"tin ({_INC1})"),
    "bronze": (54.0, R, f"phosphor gear bronze ({_INC1})"),
    "iron": (80.2, R, f"pure iron ({_INC1})"),
    "steel": (60.5, R, f"plain carbon steel ({_INC1})"),
    "glass": (1.4, R, f"plate glass ({_INC3})"),
})

# ---------------------------------------------------------------------------- fuel and food
#: latent heat of water at 25 C (J/kg), for lower heating values (IAPWS; CRC 2442 J/g)
H_FG_WATER = Val(2.442e6, REFERENCE, "enthalpy of vaporisation of water at 25 C, 2442 J/g (CRC Handbook; IAPWS-95)")


def _hhv_correlation(formula: str) -> float:
    """Higher heating value (J/kg) from the ultimate analysis, Channiwala & Parikh 2002 (Fuel 81, 1051):
    HHV MJ/kg = 0.3491 C + 1.1783 H + 0.1005 S - 0.1034 O - 0.0151 N (mass %)."""
    e = {k: 100 * v for k, v in formula_elements(formula).items()}
    return 1e6 * (0.3491 * e.get("C", 0) + 1.1783 * e.get("H", 0) + 0.1005 * e.get("S", 0)
                  - 0.1034 * e.get("O", 0) - 0.0151 * e.get("N", 0))


def _fuel_lhv(species: str) -> float:
    """Lower heating value (J/kg, water leaves as vapour) of a species' burnable components."""
    total = 0.0
    for comp, x in MAKEUP[species]:
        if comp == "C":
            total += x * 393.522e3 / molar_mass("C")   # C + O2 -> CO2, NIST-JANAF dHf(CO2)
        elif comp in ORGANIC:
            water = formula_elements(comp).get("H", 0.0) * molar_mass("H2O") / (2 * molar_mass("H"))  # kg per kg
            total += x * (_hhv_correlation(comp) - H_FG_WATER * water)
    return total


_FUELS = ("wood", "plant_fiber", "resin", "fat", "charcoal")
COMBUSTION_J_KG = _table({s: (_fuel_lhv(s) if s in _FUELS else 0.0,
                              D if s in _FUELS else R,
                              ("lower heating value: C + O2 -> CO2 from NIST-JANAF dHf(CO2) -393.522 kJ/mol" if s == "charcoal" else
                               "lower heating value: Channiwala & Parikh 2002 HHV of the makeup minus 2.442 MJ per kg of water formed")
                              if s in _FUELS else "not a fuel")
                          for s in SPECIES})

IGNITION_K = _table({
    **{s: _never("not a fuel") for s in SPECIES if s not in _FUELS},
    "wood": (573.0, R, "piloted ignition of wood about 250-365 C (Babrauskas 2003, Ignition Handbook)"),
    "plant_fiber": (533.0, R, "cellulosic tinder about 260 C (Babrauskas 2003, Ignition Handbook)"),
    "resin": (623.0, N, "rosin, about 350 C (estimate from safety data sheets)"),
    "fat": (616.0, N, "within the 340-450 C autoignition range of animal fats and oils (NFPA 325) (estimate)"),
    "charcoal": (623.0, R, "charcoal ignition about 340-400 C (Babrauskas 2003, Ignition Handbook)"),
})

#: gross energy of nutrients, J/kg (bomb calorimetry; Blaxter 1989, Energy Metabolism in Animals and
#: Man: protein 23.6, fat 39.5, carbohydrate 17.5 kJ/g)
_BLAXTER = "gross energy by bomb calorimetry (Blaxter 1989, Energy Metabolism in Animals and Man)"
GROSS_J_KG = {"protein": Val(23.6e6, R, f"protein 23.6 kJ/g, {_BLAXTER}"),
              LIPID: Val(39.5e6, R, f"fat 39.5 kJ/g, {_BLAXTER}"),
              CELLULOSE: Val(17.5e6, R, f"carbohydrate 17.5 kJ/g, {_BLAXTER}")}
_FOODS = ("meat", "fat", "bone", "hide", "plant_fiber")
FOOD_J_KG = _table({s: (sum(x * GROSS_J_KG.get(c, 0.0) for c, x in MAKEUP[s]) if s in _FOODS else 0.0,
                        D if s in _FOODS else R,
                        "gross energy of the makeup (protein 23.6, fat 39.5, carbohydrate 17.5 MJ/kg; Blaxter 1989)"
                        if s in _FOODS else "not food")
                    for s in SPECIES})
#: digestible share of the gross energy for a pure plant eater (diet 0) and a pure meat eater
#: (diet 1); creatures interpolate linearly in the diet gene (PLANET-SPEC 2.8)
_DIG = {
    "meat": (0.4, 0.9, R, "meat: 0.4 + 0.5 diet (PLANET-SPEC 2.8; carnivore protein digestibility about 0.9)"),
    "fat": (0.4, 0.9, N, "as meat"),
    "bone": (0.0, 0.4, N, "only marrow and grease, open to bone-cracking meat eaters"),
    "hide": (0.0, 0.3, N, "raw collagen is poorly digested"),
    "plant_fiber": (0.3, 0.0, N, "fibre fermentation by plant eaters (Van Soest 1994, Nutritional Ecology of the Ruminant: 0.3-0.5)"),
}
DIGESTIBLE = {
    "plant": _table({s: (_DIG[s][0], _DIG[s][2], _DIG[s][3]) if s in _DIG else (0.0, R, "not food") for s in SPECIES}),
    "meat": _table({s: (_DIG[s][1], _DIG[s][2], _DIG[s][3]) if s in _DIG else (0.0, R, "not food") for s in SPECIES}),
}
COOK_GAIN = _table({s: (1.3, R, "cooking raises the energy gained from meat (Carmody et al. 2011, PNAS 108, 19199; PLANET-SPEC 2.8)")
                    if s == "meat" else (1.0, N, "no cooking gain modelled") for s in SPECIES})

TABLES = {"DENSITY": DENSITY, "MOHS": MOHS, "GRANULAR": GRANULAR, "TOUGHNESS": TOUGHNESS, "BRITTLE": BRITTLE,
          "MELT_K": MELT_K, "CP": CP, "CONDUCTIVITY": CONDUCTIVITY, "COMBUSTION_J_KG": COMBUSTION_J_KG, "IGNITION_K": IGNITION_K,
          "FOOD_J_KG": FOOD_J_KG, "DIGESTIBLE_PLANT": DIGESTIBLE["plant"], "DIGESTIBLE_MEAT": DIGESTIBLE["meat"],
          "COOK_GAIN": COOK_GAIN}


FUELS: tuple[str, ...] = _FUELS
_ASH_FORMULAS = {f for f, _ in ASH_MAKEUP}


def burn(species: str) -> dict[str, float]:
    """Complete combustion of 1 kg of a fuel species: kg of O2 used and of CO2, H2O and ash made.

    Each organic component CxHyOz takes x + y/4 - z/2 mol O2 and gives x CO2 and y/2 H2O; the
    carbonates of wood pass to ash. Element-balanced by construction (tested). Raises ValueError
    for a species that is not in :data:`FUELS` (meat, hide or bone in a fire do not burn to ash:
    they heat, cook or char; booking them as carbonate ash would create Ca and K and destroy N)."""
    if species not in FUELS:
        raise ValueError(f"burn: {species!r} is not a fuel (FUELS = {FUELS})")
    out = {"O2": 0.0, "CO2": 0.0, "H2O": 0.0, "ash": 0.0}
    for comp, x in MAKEUP[species]:
        if comp in ORGANIC:
            c = parse(comp)
            n = x / molar_mass(comp)
            out["O2"] += n * (c.get("C", 0) + c.get("H", 0) / 4 - c.get("O", 0) / 2) * molar_mass("O2")
            out["CO2"] += n * c.get("C", 0) * molar_mass("CO2")
            out["H2O"] += n * c.get("H", 0) / 2 * molar_mass("H2O")
        elif comp in _ASH_FORMULAS:
            out["ash"] += x
        else:
            raise ValueError(f"burn: component {comp!r} of {species!r} is neither organic nor ash")
    return out


# ============================================================================== thermochemistry
_CRC = "CRC Handbook, standard thermodynamic properties of chemical substances (298.15 K)"
_JANAF = "NIST-JANAF Thermochemical Tables (Chase 1998)"
_RH = "Robie & Hemingway 1995, USGS Bulletin 2131 (approximate, about +-1 %)"
#: formula -> (dHf J/mol, S J/(mol K) or None, source); H2O is the gas, H2O(l) the liquid
THERMO: dict[str, tuple[float, float | None, str]] = {
    "C": (0.0, 5.740, f"graphite, {_JANAF}"),
    "O2": (0.0, 205.147, _JANAF),
    "CO": (-110.527e3, 197.653, _JANAF),
    "CO2": (-393.522e3, 213.795, _JANAF),
    "H2O": (-241.826e3, 188.834, f"gas, {_JANAF}"),
    "H2O(l)": (-285.830e3, 69.95, f"liquid, {_JANAF}"),
    "CaCO3": (-1207.6e3, 91.7, f"calcite, {_CRC}"),
    "CaO": (-634.9e3, 38.1, _CRC),
    "Fe": (0.0, 27.28, _CRC),
    "Fe2O3": (-824.248e3, 87.400, f"hematite, {_JANAF}"),
    "Fe3O4": (-1118.4e3, 146.4, f"magnetite, {_CRC}"),
    "Cu": (0.0, 33.15, _CRC),
    "Cu2CO3(OH)2": (-1051.4e3, 186.2, f"malachite, {_CRC}"),
    "Sn": (0.0, 51.18, f"white tin, {_CRC}"),
    "SnO2": (-577.6e3, 49.04, f"cassiterite, {_CRC}"),
    "SiO2": (-910.7e3, 41.5, f"quartz, {_CRC}"),
    "CaSiO3": (-1634.9e3, 81.9, f"wollastonite, {_CRC}"),
    "Na2CO3": (-1130.7e3, 135.0, _CRC),
    "Na2SiO3": (-1554.9e3, 113.9, _CRC),
    "Al2Si2O5(OH)4": (-4119.6e3, 200.9, f"kaolinite, {_RH}"),
    "Al6Si2O13": (-6816.2e3, 274.9, f"3:2 mullite, {_RH}"),
}

_TERM = re.compile(r"^\s*(\d+(?:\.\d+)?)?\s*([A-Z][A-Za-z0-9()]*?)(\((?:g|l|s|cr)\))?\s*$")


def reaction_terms(reaction: str) -> tuple[list[tuple[float, str]], list[tuple[float, str]]]:
    """``"Fe2O3 + 3C -> 2Fe + 3CO"`` -> ([(1, 'Fe2O3'), (3, 'C')], [(2, 'Fe'), (3, 'CO')]).
    A phase suffix ``(l)`` selects a THERMO row (``H2O(l)``); ``(g)``, ``(s)`` and ``(cr)`` are dropped."""
    left, arrow, right = reaction.partition("->")
    if not arrow:
        raise ValueError(f"no '->' in {reaction!r}")
    sides = []
    for side in (left, right):
        terms = []
        for t in side.split(" + "):
            m = _TERM.match(t)
            if m is None:
                raise ValueError(f"bad term {t!r} in {reaction!r}")
            coef, formula, phase = m.groups()
            key = formula + ("(l)" if phase == "(l)" else "")
            terms.append((float(coef) if coef else 1.0, key))
        sides.append(terms)
    return sides[0], sides[1]


def _bare(key: str) -> str:
    return key[:-3] if key.endswith("(l)") else key


def element_totals(terms: list[tuple[float, str]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for coef, key in terms:
        for e, n in parse(_bare(key)).items():
            out[e] = out.get(e, 0.0) + coef * n
    return out


def is_balanced(reaction: str) -> bool:
    left, right = reaction_terms(reaction)
    a, b = element_totals(left), element_totals(right)
    return set(a) == set(b) and all(abs(a[e] - b[e]) < 1e-9 for e in a)


def reaction_thermo(reaction: str) -> tuple[float, float]:
    """(dH J, dS J/K) of a reaction as written, from :data:`THERMO` at 298.15 K."""
    if not is_balanced(reaction):
        raise ValueError(f"reaction not element-balanced: {reaction!r}")
    left, right = reaction_terms(reaction)
    dh = sum(c * THERMO[k][0] for c, k in right) - sum(c * THERMO[k][0] for c, k in left)
    if any(THERMO[k][1] is None for _, k in left + right):
        return dh, float("nan")
    ds = sum(c * THERMO[k][1] for c, k in right) - sum(c * THERMO[k][1] for c, k in left)
    return dh, ds


def delta_g(reaction: str, t_k: float) -> float:
    """dG = dH - T dS (J) in the Ellingham approximation (dH, dS fixed at their 298 K values)."""
    dh, ds = reaction_thermo(reaction)
    return dh - t_k * ds


def equilibrium_temperature(reaction: str) -> float:
    """Temperature (K) where dG = dH - T dS = 0 for the reaction as written (298 K dH and dS).

    For an endothermic reaction with an entropy gain the products are favoured above it. 0 means
    favoured at every temperature (dH <= 0, dS >= 0); ``inf`` means never (dH > 0, dS <= 0). An
    exothermic reaction with an entropy loss is favoured *below* the returned temperature."""
    dh, ds = reaction_thermo(reaction)
    if dh <= 0 and ds >= 0:
        return 0.0
    if dh > 0 and ds <= 0:
        return math.inf
    return dh / ds


# ============================================================================== transforms
#: the atmosphere conditions a transform can need (checked by :func:`atmosphere_ok`):
#: ``any``; ``reducing``: charcoal is present in the same fire (PLANET-SPEC 2.4); ``smothered``:
#: the fire has no forced air (``air == 0``) and its bed of wood and charcoal holds at least
#: :data:`SMOTHER_COVER` x the item's mass, so the burning bed takes the O2 and a buried wood item
#: chars instead of burning (new_rule: a pit or earth kiln in miniature)
ATMOSPHERES = ("any", "reducing", "smothered")
SMOTHER_COVER = Val(1.0, NEW_RULE, "smothered charring: fire bed wood + charcoal >= 1 x the item's mass and no forced "
                                   "air (a pit or earth kiln in miniature, FAO 1987; estimate)")


@dataclass(frozen=True)
class Transform:
    """A physically conditioned change. Quantities are kg per kg of ``basis`` (inputs[basis] = 1);
    gases are separate (``gas_in`` taken from the air, ``gas_out`` released). ``enthalpy_j_kg`` is
    the heat absorbed per kg of basis at 298 K (positive: endothermic; latent heats excluded).
    ``min_temp_k = max(t_eq_k, onset_k)``, where ``t_eq_k`` (dG = 0) is a lower bound because every
    transform here is endothermic with an entropy gain (:func:`_from_reaction` refuses the other
    kinds). ``atmosphere`` is one of :data:`ATMOSPHERES`; ``days`` is the time held at or above
    ``min_temp_k`` to complete. ``onset_k`` and ``days`` are :class:`Val` with their provenance.

    Precedence (crafting keeps it): an item whose transform's conditions hold transforms; a fuel
    item above its ``IGNITION_K`` whose own transform does not apply burns as fuel. So a wood item
    smothered in a fire bed chars, and a wood item in a fire with forced air or a thin bed burns."""
    name: str
    basis: str
    inputs: dict
    outputs: dict
    gas_in: dict
    gas_out: dict
    atmosphere: str
    t_eq_k: float | None
    onset_k: float
    min_temp_k: float
    enthalpy_j_kg: float
    days: float
    reaction: str | None
    source: str
    enthalpy_source: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _signed(reaction: str, basis_formula: str) -> dict[str, float]:
    """kg of each formula per kg of the basis formula (negative = consumed)."""
    left, right = reaction_terms(reaction)
    coef = next(c for c, k in left if _bare(k) == basis_formula)
    scale = 1.0 / (coef * molar_mass(basis_formula))
    out: dict[str, float] = {}
    for sign, side in ((-1.0, left), (1.0, right)):
        for c, k in side:
            f = _bare(k)
            out[f] = out.get(f, 0.0) + sign * c * molar_mass(f) * scale
    return out


def _from_reaction(name, basis, reaction, basis_formula, solids, atmosphere, onset, days, source, scale=1.0,
                   passthrough=None, enthalpy=None, enthalpy_source=None):
    """A Transform from a balanced reaction: formulas in ``solids`` map to species, the rest are gases.

    ``min_temp_k = max(t_eq, onset)`` treats the dG = 0 temperature as a lower bound, which holds
    only for dH > 0 and dS > 0 (products favoured above it); dH <= 0, dS >= 0 gives no bound (0).
    A reaction favoured only below t_eq (dH < 0, dS < 0) or never (dH > 0, dS <= 0) raises."""
    if atmosphere not in ATMOSPHERES:
        raise ValueError(f"{name}: unknown atmosphere {atmosphere!r}")
    kg = _signed(reaction, basis_formula)
    inputs, outputs, gas_in, gas_out = {basis: 1.0}, {}, {}, {}
    for f, m in kg.items():
        if f == basis_formula:
            continue
        m *= scale
        if f in solids:
            side = inputs if m < 0 else outputs
            side[solids[f]] = side.get(solids[f], 0.0) + abs(m)
        else:
            (gas_in if m < 0 else gas_out)[f] = abs(m)
    for s, m in (passthrough or {}).items():
        outputs[s] = outputs.get(s, 0.0) + m
    t_eq = None
    if enthalpy is None:   # from THERMO; otherwise the caller derived it (wood has no THERMO row)
        dh, ds = reaction_thermo(reaction)
        if not math.isnan(ds):
            if dh > 0 and ds > 0:
                t_eq = dh / ds
            elif dh <= 0 and ds >= 0:
                t_eq = 0.0
            else:
                raise ValueError(f"{name}: dH = {dh:.4g} J, dS = {ds:.4g} J/K: the reaction is favoured only below "
                                 f"its dG = 0 temperature or never; a maximum temperature is not modelled")
        left, _ = reaction_terms(reaction)
        coef = next(c for c, k in left if _bare(k) == basis_formula)
        enthalpy = dh / (coef * molar_mass(basis_formula)) * scale
        enthalpy_source = enthalpy_source or f"derived: dH of {reaction} from THERMO (298 K)"
    min_t = max(t_eq or 0.0, float(onset))
    return Transform(name, basis, inputs, outputs, gas_in, gas_out, atmosphere, t_eq, onset, min_t, enthalpy, days,
                     reaction, source, enthalpy_source or "")


def _wood_enthalpy() -> float:
    """Heat absorbed (J per kg of wood) by charring with partial combustion,
    4 C6H9O4 + 13 O2 -> 12 C + 18 H2O(g) + 12 CO2. dHf of the wood follows from its HHV
    (complete combustion to CO2 and liquid water), so the value is derived, not looked up."""
    m = molar_mass(WOOD)
    dhf_wood = 6 * THERMO["CO2"][0] + 4.5 * THERMO["H2O(l)"][0] + _hhv_correlation(WOOD) * m
    dh = (3 * THERMO["C"][0] + 4.5 * THERMO["H2O"][0] + 3 * THERMO["CO2"][0]) - dhf_wood  # per mol C6H9O4
    return dh / m * (1.0 - float(WOOD_ASH_FRACTION))


def _glass_enthalpy() -> float:
    """Heat absorbed per kg sand: CaCO3 + SiO2 -> CaSiO3 + CO2 for the lime of the ash, and for the
    potash the soda analogue Na2CO3 + SiO2 -> Na2SiO3 + CO2 (no K2SiO3 row in the tables; new_rule)."""
    ash = float(GLASS_ASH_PER_SAND)
    ca = reaction_thermo("CaCO3 + SiO2 -> CaSiO3 + CO2")[0] * ash * 0.75 / molar_mass("CaCO3")
    k = reaction_thermo("Na2CO3 + SiO2 -> Na2SiO3 + CO2")[0] * ash * 0.25 / molar_mass("K2CO3")
    return ca + k


def _build_transforms() -> list[Transform]:
    cu_melt = MELT_K["copper"]
    spec = "PLANET-SPEC 2.4"
    ts = [
        _from_reaction("clay_to_ceramic", "clay", "3Al2Si2O5(OH)4 -> Al6Si2O13 + 4SiO2 + 6H2O", "Al2Si2O5(OH)4",
                       {"Al6Si2O13": "ceramic", "SiO2": "ceramic"}, "any",
                       Val(1150.0, R, f"{spec}: 1150 K (877 C), inside the 800-1000 C of earthenware firing (Rice 1987)"),
                       Val(0.5, N, "open firings take hours (Rice 1987); half a day (estimate)"),
                       "dehydroxylation and sintering of kaolinite; earthenware fired 800-1000 C, open firings take hours "
                       "(Rice 1987, Pottery Analysis); onset 1150 K (PLANET-SPEC 2.4)"),
        _from_reaction("limestone_to_lime", "limestone", "CaCO3 -> CaO + CO2", "CaCO3", {"CaCO3": "limestone", "CaO": "lime"},
                       "any", Val(1073.0, R, "CaCO3 dissociates in air from about 800 C (Boynton 1980)"),
                       Val(1.0, N, "lime kilns burn 900-1100 C for a day or more (Boynton 1980); one day (estimate)"),
                       "calcination: dG = 0 at the 298 K Ellingham approximation (about 1118 K); dissociation in air from "
                       "about 800 C, kilns 900-1100 C for a day or more (Boynton 1980)"),
        _from_reaction("wood_to_charcoal", "wood", "4C6H9O4 + 13O2 -> 12C + 18H2O + 12CO2", WOOD, {"C": "charcoal"},
                       "smothered", Val(600.0, R, "carbonisation of wood from about 300 C (Antal & Gronli 2003)"),
                       Val(1.0, N, "earth kilns take days for a full charge; one day for one item in a fire bed (estimate)"),
                       "charring with partial combustion of the volatiles (pit and earth kilns, charcoal yield about 25 % "
                       "of dry wood: FAO 1987, Simple technologies for charcoal making); carbonisation from about 300 C "
                       "(Antal & Gronli 2003). atmosphere 'smothered' (new_rule, SMOTHER_COVER): the spec's 'reducing' "
                       "(charcoal present) would need charcoal before the first charcoal exists, and charring needs "
                       "restricted air, not a reductant; a wood item that is not smothered burns as fuel",
                       scale=1.0 - float(WOOD_ASH_FRACTION), passthrough={"ash": float(WOOD_ASH_FRACTION)},
                       enthalpy=_wood_enthalpy(),
                       enthalpy_source="derived: dHf of C6H9O4 from its Channiwala & Parikh 2002 HHV, then Hess's law"),
        _from_reaction("malachite_to_copper", "malachite", "Cu2CO3(OH)2 + 2C -> 2Cu + CO2 + H2O + 2CO", "Cu2CO3(OH)2",
                       {"C": "charcoal", "Cu": "copper"}, "reducing",
                       Val(1000.0, R, f"{spec}: 1000 K (727 C); crucible smelting of oxide copper ores from 700-800 C "
                                      "(Tylecote 1992)"),
                       Val(0.25, N, "a crucible smelt of a few hours (estimate)"),
                       "malachite decomposes near 600 K and CuO is reduced by CO; crucible smelting of oxide ores at "
                       "700-800 C and above (Tylecote 1992); onset 1000 K (PLANET-SPEC 2.4)"),
        _from_reaction("cassiterite_to_tin", "cassiterite", "SnO2 + 2C -> Sn + 2CO", "SnO2", {"C": "charcoal", "Sn": "tin"},
                       "reducing", Val(1100.0, N, f"{spec}: 1100 K, above the 924 K Ellingham temperature (estimate)"),
                       Val(0.25, N, "a crucible smelt of a few hours (estimate)"),
                       "carbothermic reduction of cassiterite needs a hot reducing charcoal fire (Tylecote 1992); "
                       "onset 1100 K (PLANET-SPEC 2.4)"),
        _from_reaction("hematite_to_iron", "hematite", "Fe2O3 + 3C -> 2Fe + 3CO", "Fe2O3", {"C": "charcoal", "Fe": "iron"},
                       "reducing", Val(1400.0, R, f"{spec}: 1400 K (1127 C), in the bloomery reduction zone 1100-1300 C "
                                                  "(Pleiner 2000)"),
                       Val(0.5, R, "a bloomery smelt lasts 6-12 h (Pleiner 2000, Iron in Archaeology)"),
                       "bloomery: reduction zone 1100-1300 C, a smelt lasts 6-12 h (Pleiner 2000, Iron in Archaeology; "
                       "Tylecote 1992); onset 1400 K (PLANET-SPEC 2.4)"),
        _from_reaction("magnetite_to_iron", "magnetite", "Fe3O4 + 4C -> 3Fe + 4CO", "Fe3O4", {"C": "charcoal", "Fe": "iron"},
                       "reducing", Val(1400.0, R, f"{spec}: as hematite_to_iron (Pleiner 2000)"),
                       Val(0.5, R, "a bloomery smelt lasts 6-12 h (Pleiner 2000)"),
                       "bloomery, as hematite_to_iron (Pleiner 2000)"),
    ]
    sn = float(BRONZE_SN)
    ts.append(Transform("copper_tin_to_bronze", "copper", {"copper": 1.0, "tin": sn / (1 - sn)}, {"bronze": 1.0 / (1 - sn)},
                        {}, {}, "any", None, cu_melt, float(cu_melt), 0.0,
                        Val(0.1, N, "tin stirred into molten copper: a melt of a couple of hours (estimate)"), None,
                        "tin added to molten copper (Tylecote 1992); onset = copper melting point 1357.8 K; "
                        "mixing enthalpy neglected (new_rule)", "new_rule: mixing enthalpy of Cu-Sn neglected"))
    ts.append(Transform("native_copper_to_copper", "native_copper", {"native_copper": 1.0}, {"copper": 1.0}, {}, {}, "any",
                        None, cu_melt, float(cu_melt), 0.0,
                        Val(0.1, N, "melting and casting a lump: a couple of hours (estimate)"), None,
                        "native copper melted and cast (Tylecote 1992); no chemical change",
                        "no chemical change (the latent heat of melting is excluded, as for every transform)"))
    c = float(STEEL_C)
    ts.append(Transform("iron_to_steel", "iron", {"iron": 1.0, "charcoal": c / (1 - c)}, {"steel": 1.0 / (1 - c)}, {}, {},
                        "reducing", None,
                        Val(1200.0, R, f"{spec}: 1200 K; case carburising at 900-950 C (ASM Handbook vol. 4)"), 1200.0, 0.0,
                        Val(2.0, N, "carburising a piece through takes hours to days (ASM Handbook vol. 4); two days "
                                    "(estimate)"), None,
                        "carburising in charcoal: austenite above 1000 K (A1 727 C), case carburising at 900-950 C "
                        "(ASM Handbook vol. 4); onset 1200 K (PLANET-SPEC 2.4); dissolution enthalpy neglected (new_rule)",
                        "new_rule: dissolution enthalpy of C in Fe neglected"))
    ash = float(GLASS_ASH_PER_SAND)
    co2 = ash * (0.75 / molar_mass("CaCO3") + 0.25 / molar_mass("K2CO3")) * molar_mass("CO2")
    glass_onset = Val(1500.0, N, "the lime-rich, potash-poor melt (SiO2 62.9, CaO 26.4, K2O 10.7 wt%) needs more heat "
                                 "than forest glass melted at 1100-1200 C = 1373-1473 K (Wedepohl 2003); 1500 K, above "
                                 "the spec's 1400 K and below the CaO-SiO2 eutectic 1709 K (Osborn & Muan 1960) (estimate)")
    ts.append(Transform("sand_ash_to_glass", "sand", {"sand": 1.0, "ash": ash}, {"glass": 1.0 + ash - co2}, {},
                        {"CO2": co2}, "any", None, glass_onset, float(glass_onset), _glass_enthalpy(),
                        Val(1.0, N, "forest glass was melted over a day or more (Wedepohl 2003); one day (estimate)"), None,
                        "1 kg wood ash per kg sand (GLASS_ASH_PER_SAND) melts to a potash-lime glass of SiO2 62.9, CaO "
                        "26.4, K2O 10.7 wt%; onset 1500 K (new_rule, raised from the spec's 1400 K for this lime-rich "
                        "melt). sand + lime alone does not melt below the CaO-SiO2 eutectic, about 1709 K "
                        "(Osborn & Muan 1960), so that route is left out",
                        "new_rule: CaCO3 + SiO2 -> CaSiO3 + CO2 for the lime, the soda analogue Na2CO3 + SiO2 -> "
                        "Na2SiO3 + CO2 for the potash (no K2SiO3 row in THERMO)"))
    for t in ts:
        if t.atmosphere not in ATMOSPHERES:
            raise ValueError(f"{t.name}: unknown atmosphere {t.atmosphere!r}")
    return ts


TRANSFORMS: list[Transform] = _build_transforms()
TRANSFORM_INDEX = {t.name: i for i, t in enumerate(TRANSFORMS)}
#: the gases the climate ledger tracks (no CO): the order of ``air_out`` in :func:`transform_tensors`
AIR_GASES: tuple[str, ...] = ("O2", "CO2", "H2O")


def gas_to_air(t: Transform) -> dict[str, float]:
    """Net exchange of a transform with the air (kg per kg of basis; O2 negative = taken from the
    air) when its CO burns to CO2 above the charge, 2 CO + O2 -> 2 CO2 (new_rule: an open fire
    has air above the bed). The climate gas ledger tracks O2, CO2 and H2O but not CO."""
    co = t.gas_out.get("CO", 0.0) / molar_mass("CO")
    return {"O2": 0.0 - t.gas_in.get("O2", 0.0) - 0.5 * co * molar_mass("O2"),
            "CO2": t.gas_out.get("CO2", 0.0) + co * molar_mass("CO2"),
            "H2O": t.gas_out.get("H2O", 0.0)}


# ============================================================================== torch helpers
_CACHE: dict = {}


def _vec(table: dict, device, dtype=torch.float32) -> torch.Tensor:
    key = (id(table), str(device), dtype)
    t = _CACHE.get(key)
    if t is None:
        t = torch.tensor([float(table[s]) for s in SPECIES], dtype=dtype, device=device)
        _CACHE[key] = t
    return t


def element_matrix(device="cpu", dtype=torch.float64) -> torch.Tensor:
    """[S, E] element mass fractions (float64 by default, for ledgers). Cached: do not modify."""
    key = ("elements", str(device), dtype)
    if key not in _CACHE:
        _CACHE[key] = torch.tensor(ELEMENT_FRACTION, dtype=dtype, device=device)
    return _CACHE[key]


def gas_element_matrix(gases=GASES, device="cpu", dtype=torch.float64) -> torch.Tensor:
    """[G, E] element mass fractions of ``gases`` (GASES by default; AIR_GASES for ``air_out``). Cached."""
    gases = tuple(gases)
    key = ("gas_elements", gases, str(device), dtype)
    if key not in _CACHE:
        _CACHE[key] = torch.tensor([[formula_elements(g).get(e, 0.0) for e in ELEMENTS] for g in gases],
                                   dtype=dtype, device=device)
    return _CACHE[key]


def element_mass(species_kg: torch.Tensor) -> torch.Tensor:
    """[..., S] species masses -> [..., E] element masses (same dtype)."""
    return species_kg @ element_matrix(species_kg.device, species_kg.dtype)


def species_vector(values: dict[str, float], device="cpu", dtype=torch.float32) -> torch.Tensor:
    """A dict species -> value as an [S] tensor in SPECIES order (missing species are 0)."""
    unknown = set(values) - set(SPECIES)
    if unknown:
        raise KeyError(f"unknown species {sorted(unknown)}")
    return torch.tensor([float(values.get(s, 0.0)) for s in SPECIES], dtype=dtype, device=device)


#: a phase is continuous (a matrix that carries scratches) above the site-percolation threshold
#: of the simple cubic lattice, 0.3116 (Stauffer & Aharony 1994, Introduction to Percolation Theory)
CONTINUOUS_FRACTION = Val(0.3116, REFERENCE, "site percolation threshold, simple cubic lattice (Stauffer & Aharony 1994)")
#: melting and ignition count only species above this mass fraction (PLANET-SPEC 2.4)
PRESENT_FRACTION = Val(0.10, NEW_RULE, "PLANET-SPEC 2.4: melting is the minimum over species above 10 %")


def _clean(x: torch.Tensor) -> torch.Tensor:
    """float32 with NaN and +-inf as 0 and negatives clamped to 0."""
    return torch.nan_to_num(x.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)


def props(comp: torch.Tensor, mass: torch.Tensor) -> dict[str, torch.Tensor]:
    """Physical properties of items from their composition, batched over any leading shape.

    comp [..., S] (mass fractions; renormalised here), mass [...] kg. Returns float32 tensors [...]:
    ``mass``, ``volume_m3``, ``density`` (volume-additive), ``hardness`` (Mohs scratch hardness:
    mass-weighted mean, capped at the hardest continuous phase, i.e. one above the percolation
    fraction, or the dominant phase), ``granular`` (mass fraction of loose grains or powder),
    ``tool_hardness`` (``hardness x (1 - granular)``: what hammering, knapping, chopping, digging and
    striking use, so a pile of sand is no hammer), ``toughness`` (mass-weighted), ``brittle`` (mass
    fraction with conchoidal fracture), ``melt_k`` (lowest melting point among species above 10 %),
    ``cp`` (J/(kg K), Neumann-Kopp), ``heat_capacity_j_k``, ``conductivity``, ``fuel_j`` (lower
    heating value of the whole item), ``ignition_k`` (lowest among fuels above 10 %), ``food_j``
    (gross), ``food_plant_j`` and ``food_meat_j`` (digestible at diet 0 and 1), ``cook_gain``
    (food-energy weighted). An alloy is its own species (bronze, steel), so a copper + tin mix is
    not bronze until it is transformed.

    Every output is finite and non-negative: a negative or non-finite fraction counts as 0 (crafting
    leaves float32 residues such as -1e-9 after subtracting consumed kg) and so does a negative or
    non-finite mass. A row without a positive fraction is empty: zeros, ``melt_k`` and
    ``ignition_k`` NEVER_K, ``cook_gain`` 1."""
    comp = _clean(comp)
    mass = _clean(mass)
    dev = comp.device
    total = comp.sum(-1, keepdim=True)
    has = (total > 0) & torch.isfinite(total)
    w = torch.where(has, comp / torch.where(has, total, torch.ones_like(total)), torch.zeros_like(comp))
    has = has[..., 0]
    v = lambda t: _vec(t, dev)
    # the mass-weighted sums of every linear property in one product w @ TABLE [S, P]
    lin = torch.matmul(w, _props_table(dev))
    (spec_vol, mean_h, granular, toughness, brittle, fuel_kg, cp, conductivity, food_kg, food_p, food_m,
     food_cook) = lin.unbind(-1)
    density = torch.where(spec_vol > 0, 1.0 / spec_vol.clamp_min(1e-30), torch.zeros_like(spec_vol))
    dominant = (w == w.amax(-1, keepdim=True)) & (w > 0)
    hard = v(MOHS)
    cont = (w >= float(CONTINUOUS_FRACTION)) | dominant
    cap = torch.where(cont, hard.expand_as(w), torch.full_like(w, -1.0)).amax(-1)
    hardness = torch.where(has, torch.minimum(mean_h, cap), torch.zeros_like(mean_h))
    granular = granular.clamp(0.0, 1.0)
    present = (w > float(PRESENT_FRACTION)) | dominant
    never = float(NEVER_K)
    melt = torch.where(present, v(MELT_K).expand_as(w), torch.full_like(w, never)).amin(-1)
    is_fuel = present & (v(COMBUSTION_J_KG) > 0)
    ign = torch.where(is_fuel, v(IGNITION_K).expand_as(w), torch.full_like(w, never)).amin(-1)
    cook = torch.where(food_kg > 0, food_cook / food_kg.clamp_min(1e-30), torch.ones_like(food_kg))
    return {
        "mass": mass,
        "volume_m3": mass * spec_vol,
        "density": density,
        "hardness": hardness,
        "granular": granular,
        "tool_hardness": hardness * (1.0 - granular),
        "toughness": toughness,
        "brittle": brittle,
        "melt_k": melt,
        "cp": cp,
        "heat_capacity_j_k": cp * mass,
        "conductivity": conductivity,
        "fuel_j": fuel_kg * mass,
        "ignition_k": ign,
        "food_j": food_kg * mass,
        "food_plant_j": food_p * mass,
        "food_meat_j": food_m * mass,
        "cook_gain": cook,
    }


def _props_table(device) -> torch.Tensor:
    """[S, 12] float32 per-species columns of the linear properties of :func:`props`: 1 / density, Mohs,
    granular, toughness, brittle, combustion J/kg, cp, conductivity, food J/kg, food x plant and meat
    digestibility, food x cook gain. Cached per device."""
    key = ("props_table", str(device))
    t = _CACHE.get(key)
    if t is None:
        cols = []
        for s in SPECIES:
            food = float(FOOD_J_KG[s])
            cols.append([1.0 / float(DENSITY[s]), float(MOHS[s]), float(GRANULAR[s]), float(TOUGHNESS[s]),
                         float(BRITTLE[s]), float(COMBUSTION_J_KG[s]), float(CP[s]), float(CONDUCTIVITY[s]), food,
                         food * float(DIGESTIBLE["plant"][s]), food * float(DIGESTIBLE["meat"][s]),
                         food * float(COOK_GAIN[s])])
        t = torch.tensor(cols, dtype=torch.float64).float().to(device)
        _CACHE[key] = t
    return t


def digestible_j(p: dict[str, torch.Tensor], diet: torch.Tensor) -> torch.Tensor:
    """Digestible energy (J) of items with properties ``p`` for an eater with diet gene ``diet`` in [0, 1]."""
    return p["food_plant_j"] * (1 - diet) + p["food_meat_j"] * diet


def transform_tensors(device="cpu") -> dict:
    """The transforms as tensors for vectorised crafting (built once per device and cached; the dict
    is a fresh copy, the tensors are shared and must not be modified in place):

    ``consume`` and ``produce`` [T, S] (kg per kg basis); ``gas_out`` [T, G] (kg per kg basis,
    G = GASES, negative = taken from the air, CO included); ``air_out`` [T, 3] (AIR_GASES = O2, CO2,
    H2O: the net exchange with the climate gas ledger once the CO has burnt to CO2, O2 negative;
    :func:`gas_to_air`); ``basis`` [T] long; ``atmosphere`` [T] long (index in ATMOSPHERES);
    ``reducing`` and ``smothered`` [T] bool; ``min_temp_k``, ``enthalpy_j_kg``, ``days`` [T] float32.
    Element closure per transform: ``produce @ E + gas_out @ E_gas - consume @ E = 0`` and
    ``produce @ E + air_out @ E_air - consume @ E = 0`` (:func:`element_matrix`, :func:`gas_element_matrix`)."""
    key = ("transforms", str(device))
    tt = _CACHE.get(key)
    if tt is None:
        T = len(TRANSFORMS)
        consume = torch.zeros(T, S, dtype=torch.float64)
        produce = torch.zeros(T, S, dtype=torch.float64)
        gas = torch.zeros(T, len(GASES), dtype=torch.float64)
        air = torch.zeros(T, len(AIR_GASES), dtype=torch.float64)
        for i, t in enumerate(TRANSFORMS):
            for s, m in t.inputs.items():
                consume[i, IDX[s]] += m
            for s, m in t.outputs.items():
                produce[i, IDX[s]] += m
            for g, m in t.gas_out.items():
                gas[i, GASES.index(g)] += m
            for g, m in t.gas_in.items():
                gas[i, GASES.index(g)] -= m
            for g, m in gas_to_air(t).items():
                air[i, AIR_GASES.index(g)] = m
        f32 = lambda x: torch.tensor(x, dtype=torch.float32, device=device)
        atm = [ATMOSPHERES.index(t.atmosphere) for t in TRANSFORMS]
        tt = {
            "names": tuple(t.name for t in TRANSFORMS),
            "consume": consume.float().to(device), "produce": produce.float().to(device),
            "gas_out": gas.float().to(device), "air_out": air.float().to(device),
            "basis": torch.tensor([IDX[t.basis] for t in TRANSFORMS], dtype=torch.long, device=device),
            "atmosphere": torch.tensor(atm, dtype=torch.long, device=device),
            "reducing": torch.tensor([a == ATMOSPHERES.index("reducing") for a in atm], dtype=torch.bool, device=device),
            "smothered": torch.tensor([a == ATMOSPHERES.index("smothered") for a in atm], dtype=torch.bool, device=device),
            "min_temp_k": f32([float(t.min_temp_k) for t in TRANSFORMS]),
            "enthalpy_j_kg": f32([float(t.enthalpy_j_kg) for t in TRANSFORMS]),
            "days": f32([float(t.days) for t in TRANSFORMS]),
        }
        _CACHE[key] = tt
    return dict(tt)


def fire_atmosphere(fuel_kg: torch.Tensor, air: torch.Tensor, item_mass: torch.Tensor,
                    item_comp: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """The atmosphere an item meets in a fire, batched over any leading shape: ``fuel_kg`` [..., S]
    (the fire's bed), ``air`` [...] (forced-air effort), ``item_mass`` [...] kg and optionally the
    item's ``item_comp`` [..., S]. Returns bool tensors [...]: ``reducing`` (charcoal in the fire,
    in its bed or mixed into the item: PLANET-SPEC 2.4) and ``smothered`` (no forced air and a bed
    of wood and charcoal of at least :data:`SMOTHER_COVER` x the item's mass)."""
    fuel = _clean(fuel_kg)
    charcoal = fuel[..., IDX["charcoal"]]
    bed = charcoal + fuel[..., IDX["wood"]]
    reducing = charcoal > 0
    if item_comp is not None:
        reducing = reducing | (_clean(item_comp)[..., IDX["charcoal"]] > 0)
    need = float(SMOTHER_COVER) * _clean(torch.as_tensor(item_mass, device=fuel.device))
    smothered = (torch.as_tensor(air, device=fuel.device) <= 0) & (bed > 0) & (bed >= need)
    return {"reducing": reducing, "smothered": smothered}


def atmosphere_ok(fuel_kg: torch.Tensor, air: torch.Tensor, item_mass: torch.Tensor,
                  item_comp: torch.Tensor | None = None) -> torch.Tensor:
    """[..., T] bool: whether each transform's atmosphere condition holds for an item in a fire
    (arguments as :func:`fire_atmosphere`). Temperature and inputs are checked by the caller."""
    cond = fire_atmosphere(fuel_kg, air, item_mass, item_comp)
    tt = transform_tensors(fuel_kg.device)
    return (~tt["reducing"] | cond["reducing"][..., None]) & (~tt["smothered"] | cond["smothered"][..., None])


# ============================================================================== crust
#: hydrogen mass fraction of the present-day solar photosphere (Asplund et al. 2009, ARA&A 47, 481, Table 4)
X_SUN = Val(0.7381, REFERENCE, "present-day solar photosphere X = 0.7381 (Asplund et al. 2009, ARA&A 47, 481, Table 4)")
_CLOUD_EL = {"carbon": "C", "nitrogen": "N", "oxygen": "O", "neon": "Ne", "magnesium": "Mg", "silicon": "Si", "iron": "Fe"}


def solar_cloud() -> dict[str, float]:
    """Solar mass fractions in the chain's cloud keys (Asplund et al. 2009 log eps with X_sun);
    ``other`` = Z_sun minus the seven named metals, as in ``haishool/evo/stars.py``."""
    w = atomic_weights()
    out = {k: X_SUN * 10 ** (SOLAR_LOG_EPS[e] - 12.0) * w[e] / w["H"] for k, e in _CLOUD_EL.items()}
    out["other"] = Z_SUN - sum(out.values())
    out["metallicity"] = Z_SUN
    return out


#: upper continental crust (Rudnick & Gao 2003, Treatise on Geochemistry vol. 3, Table 3)
UCC = {"FeO_total": Val(5.04e-2, R, "UCC FeOT 5.04 wt% (Rudnick & Gao 2003)"),
       "Cu": Val(28e-6, R, "UCC Cu 28 ppm (Rudnick & Gao 2003)"),
       "Sn": Val(2.1e-6, R, "UCC Sn 2.1 ppm (Rudnick & Gao 2003)"),
       "S": Val(621e-6, R, "UCC S 621 ppm (Rudnick & Gao 2003)")}
#: Earth's accessible surface deposits by mass, before the chain scalings (new_rule: rounded from
#: the exposed continental lithology, Blatt & Jones 1975, GSA Bull. 86, 1085, and Amiotte Suchet,
#: Probst & Ludwig 2003, Global Biogeochem. Cycles 17, 1038: shield and acid plutonic rocks about a
#: quarter, basalts about 7 %, sands and sandstones about a quarter, carbonates about 13 %; shales
#: count only as their workable clay, the rest of their mass is spread by renormalising)
EARTH_SHARE = {
    "granite": Val(0.25, N, "shield, acid plutonic and acid volcanic rocks (Amiotte Suchet et al. 2003)"),
    "basalt": Val(0.07, N, "basalts (Amiotte Suchet et al. 2003)"),
    "sandstone": Val(0.15, N, "sandstones (Amiotte Suchet et al. 2003)"),
    "sand": Val(0.10, N, "unconsolidated sands (Amiotte Suchet et al. 2003)"),
    "clay": Val(0.08, N, "workable surface clays (a part of the shales; Blatt & Jones 1975)"),
    "limestone": Val(0.13, N, "carbonate rocks (Amiotte Suchet et al. 2003)"),
    "flint": Val(0.01, N, "chert nodules and beds in carbonates"),
    "salt": Val(0.01, N, "evaporites"),
}
#: share of crustal Fe accessible as oxide ore (laterite, bog ore, banded iron), split 2:1 hematite:magnetite
FE_ORE_SHARE = Val(0.10, N, "accessible share of crustal Fe in oxide ores (estimate)")
HEMATITE_OF_FE_ORE = Val(2 / 3, N, "hematite ores dominate world iron ore, magnetite about a third (USGS Mineral Commodity Summaries)")
#: crustal Cu is placed in its oxidised ores at the surface: malachite 80 %, native copper 20 % (by Cu)
MALACHITE_OF_CU = Val(0.8, N, "surface oxidised copper: malachite with some native copper (estimate)")
CRUST_RULE = ("materials.crust_fractions: Earth's surface lithology shares (new_rule, EARTH_SHARE) and upper "
              "continental crust abundances (reference, Rudnick & Gao 2003) scaled by the chain cloud's ratios "
              "relative to solar (Asplund et al. 2009): basalt x Mg/Si, felsic and silica rocks / Mg/Si, iron ores x "
              "Fe/Si, limestone x C/O x s_Ca, salt x s_Na, pyrite x s_S, malachite and native copper x s_Cu, "
              "cassiterite x s_Sn; s_Ca = s_Na = s_K = s_S = the cloud's other/Si (Ca, Na, K, S, Al are in the chain's "
              "'other' bin: derived); s_Cu = s_Sn = other/Si x Z/Z_sun (new_rule: secondary heavy elements fall with "
              "metallicity, PLANET-SPEC 2.3)")


def crust_scales(inputs: dict) -> dict[str, float]:
    """The chain-cloud ratios (relative to solar) that shape the deposit mix."""
    c, sun = inputs["cloud"], solar_cloud()
    tiny = 1e-30

    def rel(a, b):
        return (max(c[a], tiny) / max(c[b], tiny)) / (sun[a] / sun[b])

    other = rel("other", "silicon")
    z = max(c["metallicity"], tiny) / Z_SUN
    return {"mg_si": rel("magnesium", "silicon"), "fe_si": rel("iron", "silicon"), "c_o": rel("carbon", "oxygen"),
            "other_si": other, "z": z, "ca": other, "na": other, "k": other, "s": other, "cu": other * z, "sn": other * z}


def crust_fractions(inputs: dict, spec=None) -> dict[str, float]:
    """Mass fractions of the ground-deposit species (CRUST_SPECIES, summing to 1) for a planet
    from its chain inputs (``inputs['cloud']``). ``spec`` is accepted for the formation API and
    not used: the mix depends on the chain composition only (rule: :data:`CRUST_RULE`)."""
    k = crust_scales(inputs)
    w = {s: float(EARTH_SHARE.get(s, 0.0)) for s in CRUST_SPECIES}
    w["basalt"] *= k["mg_si"]
    for s in ("granite", "sandstone", "sand", "flint"):
        w[s] /= k["mg_si"]
    w["limestone"] *= k["c_o"] * k["ca"]
    w["salt"] *= k["na"]
    fe_ore = float(UCC["FeO_total"]) * molar_mass("Fe") / molar_mass("FeO") * float(FE_ORE_SHARE) * k["fe_si"]
    h = float(HEMATITE_OF_FE_ORE)
    w["hematite"] = fe_ore * h * molar_mass("Fe2O3") / (2 * molar_mass("Fe"))
    w["magnetite"] = fe_ore * (1 - h) * molar_mass("Fe3O4") / (3 * molar_mass("Fe"))
    cu = float(UCC["Cu"]) * k["cu"]
    m = float(MALACHITE_OF_CU)
    w["malachite"] = cu * m * molar_mass("Cu2CO3(OH)2") / (2 * molar_mass("Cu"))
    w["native_copper"] = cu * (1 - m)
    w["cassiterite"] = float(UCC["Sn"]) * k["sn"] * molar_mass("SnO2") / molar_mass("Sn")
    w["pyrite"] = float(UCC["S"]) * k["s"] * molar_mass("FeS2") / (2 * molar_mass("S"))
    total = sum(w.values())
    return {s: w[s] / total for s in CRUST_SPECIES}


# ============================================================================== provenance
def provenance() -> dict[str, tuple[str, str]]:
    """Dotted key -> (tag, source) for every number in this module that sets up a world."""
    out: dict[str, tuple[str, str]] = {}
    for name, table in TABLES.items():
        for s, val in table.items():
            out[f"materials.{name}.{s}"] = (val.tag, val.source)
    for name in ("NEVER_K", "WOOD_ASH_FRACTION", "GLASS_ASH_PER_SAND", "BRONZE_SN", "STEEL_C", "CONTINUOUS_FRACTION",
                 "PRESENT_FRACTION", "FE_ORE_SHARE", "HEMATITE_OF_FE_ORE", "MALACHITE_OF_CU", "H_FG_WATER", "X_SUN",
                 "SMOTHER_COVER"):
        val = globals()[name]
        out[f"materials.{name}"] = (val.tag, val.source)
    for s, (tag, src) in MAKEUP_SOURCE.items():
        out[f"materials.MAKEUP.{s}"] = (tag, src)
    out["materials.PROTEIN"] = PROTEIN_SOURCE
    out["materials.ASH_MAKEUP"] = ASH_MAKEUP_SOURCE
    for c, val in GROSS_J_KG.items():
        out[f"materials.GROSS_J_KG.{c}"] = (val.tag, val.source)
    for s, val in EARTH_SHARE.items():
        out[f"materials.EARTH_SHARE.{s}"] = (val.tag, val.source)
    for s, val in UCC.items():
        out[f"materials.UCC.{s}"] = (val.tag, val.source)
    for f, (_, _, src) in THERMO.items():
        out[f"materials.THERMO.{f}"] = (REFERENCE, src)
    for t in TRANSFORMS:
        key = f"materials.TRANSFORMS.{t.name}"
        out[key] = (DERIVED if t.reaction else NEW_RULE, t.source)
        for field in ("onset_k", "days"):
            val = getattr(t, field)
            out[f"{key}.{field}"] = (val.tag, val.source)
        tag = NEW_RULE if t.enthalpy_source.startswith("new_rule") else DERIVED
        out[f"{key}.enthalpy_j_kg"] = (tag, t.enthalpy_source)
    out["materials.solar_cloud"] = (DERIVED, "Asplund et al. 2009 photospheric log eps (constants.SOLAR_LOG_EPS) with "
                                             "X_SUN and Z_SUN, in the chain's cloud keys (as haishool/evo/stars.py)")
    out["materials.crust_fractions"] = (DERIVED, CRUST_RULE)
    return out
