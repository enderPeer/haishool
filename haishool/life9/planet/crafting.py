"""Crafting: the actions on items and fires, fire physics, item heating, transforms and decay
(PLANET-SPEC section 2.11).

The pools are ``items.ItemPool`` [W, I] and ``items.FirePool`` [W, F]; every function here changes
them in place (as ``items.spawn`` does) and returns a dict of outcomes and ledger flows. Actors
are plain tensors with a leading worlds axis (``creatures`` owns them):

* ``inv`` [W, N, K] long: the item index in each of K hand slots, -1 empty. Changed in place and
  kept in step with ``pool.holder = n * K + k`` (``check_inventory`` verifies the pair).
* ``act`` [W, N] bool: the actors that take this action this tick (the caller masks the dead).
* ``pos`` [W, N, 3] unit vectors (habitat globe); ``cell`` [W, N] long (``globe.cell_of(pos)``).
* slot arguments [W, N] long, a hand slot 0..K-1 or -1; ``None`` picks the documented default
  (for example the held item with the highest working hardness as the tool).
* ``effort`` [W, N] in [0, 1] (None = 1), already scaled by the caller's aerobic capacity;
  ``body_mass_kg`` [W, N].

Outcomes are [W, N] tensors; ledger flows are per world in float64. ``collect``, ``knap`` and
``make_fire`` report the mechanical work they took (``work_j``, effort x arm power x ``ACTION_S``);
``action_work_j`` gives the same for any other action, for the energy ledger.

Ledgers. Items plus fire beds hold the item species mass (``ledger_mass``). Flows in: ``collect``
(``ground_kg`` from the deposit stock, ``wood_kg`` from the plants). Flows out: ``fire_step`` and
``cook`` (``air_kg`` [W, 3] net exchange with the air in ``materials.AIR_GASES`` order, O2 negative
= taken; ``to_soil_kg`` fire residue with no free item slot and tissue charred in fires) and
``decay`` (``decayed_kg``, its carbon to the cell's litter). Element closure: E(items + fires)
changes by E(in) - E(out) - E(air_kg); species mass is conserved through every transform and burn
up to float32 rounding. Energy: the items in a fire take at most LOAD_SHARE of the heat its bed
releases (``item_heat_j``).

No meaning is supplied. A tool's effect comes from its hardness, sharpness, mass and lever; a
transform happens when temperature, atmosphere and inputs hold (``materials.TRANSFORMS``); fire
needs fuel, ignition and oxygen. ``materials.CLASS`` is read only to group what the brain's four
collect outputs reach (the same grouping its senses use), never to decide an outcome.

Every number that sets up a world is a ``materials.Val`` with its provenance; ``provenance()``
lists them. Units are SI. One generator is used (``collect``'s choice of lump), so CPU runs are
exactly reproducible.
"""
from __future__ import annotations

import math

import torch

from haishool.life9.planet import items as it
from haishool.life9.planet import materials as mat
from haishool.life9.planet.constants import DAY_S, SIGMA, molar_mass
from haishool.life9.planet.materials import DERIVED, NEW_RULE, REFERENCE, Val

R_, D_, N_ = REFERENCE, DERIVED, NEW_RULE
S = mat.S
IDX = mat.IDX
CHAR, WOOD, PYRITE, HIDE, FIBER = (IDX[s] for s in ("charcoal", "wood", "pyrite", "hide", "plant_fiber"))
COLLECT_CLASSES = ("stone", "ore", "clay", "wood")          # brain.COLLECT_CLASSES, in that order
FLAMING, GLOWING = 1, 2

# ============================================================================== reach and geometry
REACH_M = Val(5.0, N_, "an individual reaches items, fires and kin within 5 m, the reach creatures.REACH_M gives for "
                       "food items on the ground (one reach for the engine)")
FIRE_RADIUS_M = Val(0.5, N_, "an item on the ground within 0.5 m of a fire's centre lies in its bed (hearths 0.5-1 m across)")
WARMTH_RADIUS_M = Val(20.0, N_, "PLANET-SPEC 2.11: a fire warms everyone within 20 m")
NEAR_FIRE_M = Val(1.0, N_, "the closest an individual sits to a fire: the point-source flux is capped at 1 m")
SPHERE_AREA = Val((36 * math.pi) ** (1 / 3), D_, "surface of a sphere per volume^(2/3), (36 pi)^(1/3): items and fire "
                                                 "beds are taken as compact lumps (new_rule geometry)")
MIN_ITEM_KG = Val(1e-4, N_, "numerics: a piece below 0.1 g is not an item (its mass is booked where it goes)")

# ============================================================================== fire
X_O2_MIN = Val(0.15, R_, "no fire spreads below an O2 mole fraction of about 0.15 (Belcher et al. 2010, PNAS 107, 22448)")
X_O2_REF = Val(0.21, N_, "PLANET-SPEC 2.11: the fire formulas are calibrated at Earth's O2 mole fraction 0.21")
FIRE_DT0_K = Val(812.0, N_, "excess temperature of an open wood fire at x_O2 0.21: calibrated so T_fire = 1100 K at "
                            "T_air 288 K, the spec's own anchor (open wood fire about 1100 K). The spec's formula "
                            "constant 1000 K would give 1288 K open and 2043 K forced charcoal, above iron's melting "
                            "point, which a bloomery never reaches; with 812 K a fully forced charcoal fire is 1713 K, "
                            "a hot forge, and the spec's 1500 K forced charcoal fire is air effort 0.33")
AIR_GAIN = Val(0.3, N_, "PLANET-SPEC 2.11: forced air raises the fire's excess temperature by 30 % at full effort")
FUEL_FACTOR = {s: Val(1.35, N_, "PLANET-SPEC 2.11: charcoal fuel factor 1.35") if s == "charcoal" else
               Val(1.0, N_, "PLANET-SPEC 2.11: wood fuel factor 1.0; fibre, resin and fat taken as flaming fuels like wood")
               for s in mat.FUELS}
MIN_FIRE_KG = Val(0.01, N_, "a fire with less than 10 g of fuel goes out")
EXIT_CHECK = Val(8, N_, "numerics: fire_step asks every 8 substeps whether any fire still burns; once none does, "
                        "the rest of the step is one relaxation of the items toward their air")
BETA_FLAMING = Val(0.50e-3 / 60.0, R_, "flaming fuels regress at 0.50 mm/min, the one-dimensional charring rate "
                                       "beta_0 of hardwood with a characteristic density of 450 kg/m3 or more (EN "
                                       "1995-1-2:2004, Eurocode 5, Table 3.1; the wood species is oak, 545 kg/m3; "
                                       "softwood 0.65); used for every flaming fuel of a bed (new_rule), scaled by "
                                       "x_O2/0.21 and (1 + air)")
#: a fire bed is a pile, not a solid lump: species s fills the share PACKING[s] of the pile's
#: volume, so the pile's envelope is V = sum m_s / (PACKING_s rho_s) and burns back at beta / PACKING
#: (only that share of the swept volume is fuel); compact lumps and non-fuels count as solid
PACKING_SOLID = Val(1.0, N_, "a compact lump (resin, fat) or a non-fuel in a fire bed fills its volume")
PACKING = {s: PACKING_SOLID for s in mat.SPECIES}
PACKING.update({
    "wood": Val(0.4, R_, "loosely piled firewood holds about 0.4-0.5 m3 of solid wood per m3 of pile, stacked about "
                         "0.6-0.7 (forestry conversion factors, e.g. Kofman 2010, COFORD Connects, Units, conversion "
                         "factors and formulae for wood for energy); a campfire pile is loose"),
    "plant_fiber": Val(0.07, N_, "loose fibre and straw lie at roughly 50-150 kg/m3 (baled straw about 100-150) against "
                                 "1500 kg/m3 of the fibre itself: 100 / 1500 (estimate)"),
    "charcoal": Val(0.55, N_, "lump charcoal heaps at roughly 200-250 kg/m3 against its piece density of 400 kg/m3 "
                              "(materials.DENSITY) (estimate)"),
})
H_CONV = Val(10.0, R_, "free-convection heat transfer coefficient of gases 2-25 W/(m2 K) (Incropera et al. 2007, Table 1.1)")
EMISSIVITY = Val(0.9, R_, "total emissivity of rock, brick, char and oxidised surfaces 0.8-0.95 (Incropera et al. 2007, Table A.11)")
CP_GAS_HOT = Val(1141.0, R_, "cp of air at 1000 K, 1.141 kJ/(kg K) (Incropera et al. 2007, Table A.4)")
M_AIR = Val(0.0289644, R_, "molar mass of dry air 28.9644 g/mol (U.S. Standard Atmosphere 1976)")
#: glowing char burns at the diffusion limit: m'' = h_m rho_g Y_O2 / s with h_m = h / (rho_g cp_g)
#: (heat and mass transfer analogy, Le ~ 1), s = kg O2 per kg C; about 0.11 mm/min of charcoal in still air
S_CARBON = Val(molar_mass("O2") / molar_mass("C"), D_, "kg O2 per kg C in C + O2 -> CO2")
#: flaming wood pyrolyses to char with the yield of materials' wood_to_charcoal reaction and burns its
#: volatiles; the char then glows away (two-stage combustion of wood, Drysdale 2011, An Introduction
#: to Fire Dynamics, 3rd ed., ch. 5)
WOOD_CHAR_RULE = (N_, "flaming wood leaves char with the wood_to_charcoal stoichiometry (0.246 kg C per kg wood; "
                      "conventional carbonisation yields 25-35 %, Antal & Gronli 2003) and the char glows away; heat "
                      "COMBUSTION_J_KG[wood] - yield x COMBUSTION_J_KG[charcoal] then COMBUSTION_J_KG[charcoal] (Hess)")
RADIANT_FRACTION = Val(0.3, R_, "radiative fraction of the heat release of wood fires 0.2-0.4 (SFPE Handbook of Fire "
                                "Protection Engineering, point-source radiation model q = chi_r Q / (4 pi d^2))")
LOAD_SHARE = Val(0.3, N_, "the items lying in a fire take at most 30 % of the heat its bed releases in a step (sensible "
                          "heat m cp dT, endothermic transforms, drying of charring tissue); open three-stone cooking "
                          "fires pass about 10-20 % of their heat to the pot (MacCarty, Still & Ogle 2010, Energy for "
                          "Sustainable Development 14, 161), a load buried in the bed takes more (estimate)")
FAN_S = Val(3600.0, N_, "forced air from one feed_fire action lasts its ACTION_S (an hour of fanning or blowing)")
#: meat, hide and bone in a fire: their organic matter dries, chars and burns away (no species
#: holds its N, S, P or Ca, so the charred tissue goes to the soil at the fire, ``to_soil_kg``)
TISSUE = tuple(s for s in mat.SPECIES if s not in mat.FUELS and any(c == "protein" for c, _ in mat.MAKEUP[s]))
TISSUE_CHAR_K = Val(573.0, R_, "tissue chars from about 300 C: bone blackens at 285-525 C (Shipman, Foster & Schoeninger "
                               "1984, J. Archaeol. Sci. 11, 307) and collagen decomposes from about 250 C")
TISSUE_CHAR_RULE = (N_, "an item's meat, hide and bone above TISSUE_CHAR_K are lost from its surface inward at "
                        "BETA_FLAMING (the item shrinks as a sphere, m (1 - beta t / r)^3), drying their water first "
                        "(H_FG_WATER per kg of MAKEUP water, charged to the fire's LOAD_SHARE); the charred tissue goes "
                        "to the soil at the fire (to_soil_kg, also reported as charred_kg): element-exact, but its "
                        "volatiles' CO2 and H2O reach the air through the soil carbon rather than at once")
ROAST_SHARE = Val(0.2, N_, "cook holds the item at roasting distance, where it meets T_air + 0.2 (T_fire - T_air): 450 K "
                           "beside an open wood fire (1100 K), 507 K beside an open charcoal fire, the hot air of an oven "
                           "roast at 150-230 C (estimate)")

# ============================================================================== ignition
ARM_POWER_W = Val(75.0, N_, "sustained arm work of a 70 kg adult: arm-crank ergometry gives about 50-100 W sustained "
                            "(Sawka 1986, Exerc. Sport Sci. Rev. 14, 175); scaled as (M / 70 kg)^0.75 (Kleiber)")
ACTION_S = Val(3600.0, N_, "one craft action is an hour of work at the actor's effort")
FRICTION_EFFICIENCY = Val(0.1, N_, "share of a fire drill's work dissipated at the tip; the rest goes into the bow, the "
                                   "bearing block and the dust (estimate)")
DRILL_RADIUS_M = Val(0.005, N_, "tip radius of a hand or bow drill (spindles 1-2 cm across)")
FRICTION_RULE = (D_, "contact temperature of a drill: T = T_air + eta P / (8 a k), the constriction resistance "
                     "1/(4 a k) of an isothermal disc of radius a on a half-space (Carslaw & Jaeger 1959, Conduction "
                     "of Heat in Solids, sec. 8.2), drill and board in parallel; k = the fuel's conductivity")
PYRITE_MIN = Val(0.5, N_, "a spark stone is an item at least half pyrite")
SPARK_TINDER_K = Val(550.0, N_, "a pyrite spark carries millijoules and lights only tinder: fuels igniting at 550 K or "
                                "below (cellulosic tinder 533 K; wood 573 K needs an ember) (estimate)")

# ============================================================================== tools
KNAP_HARDNESS_RATIO = Val(1.1, N_, "PLANET-SPEC 2.11: a core knaps when its hardness <= the hammer's x 1.1")
BRITTLE_MIN = Val(0.5, N_, "a core knaps when at least half of it fractures conchoidally (materials.BRITTLE)")
KNAP_LOSS = Val(0.2, N_, "one knap action (ACTION_S of work) at full effort removes a fifth of the core's knappable "
                         "mass as debitage (biface reduction removes most of a nodule's mass over several sessions, "
                         "Newcomer 1971, World Archaeology 3, 85)")
KNAP_EDGE_SHARE = Val(0.05, N_, "removing 5 % of the core brings the edge 63 % of the way to its limit (estimate)")
K_EDGE = Val(2.0, N_, "sharpness limit 1 / (1 + (K_IC / 2 MPa m^0.5)^2): the edge radius left by conchoidal fracture "
                      "grows with the process zone, which scales as K_IC^2 (Lawn 1993, Fracture of Brittle Solids)")
HEAD_CORE = Val(0.5, N_, "an assembly works with the hardness of its head: of the species making up at least half of "
                         "its head mass, the one whose mass is closest to the head mass (the hardest on a tie)")
BOND_FIT = Val(0.2, N_, "bond of a head wedged in a handle with no binder (estimate)")
BINDER_SHARE = Val(0.05, N_, "a binder of 5 % of the head mass brings the bond 63 % of the way to full (estimate)")
HANDLE_ASPECT = Val(15.0, N_, "a handle is a rod 15 times as long as it is thick (estimate)")
K_ROD = Val(2.0, N_, "a slender rod survives a swing in proportion to its toughness up to 2 MPa m^0.5 (wood 3 and bone 5 "
                     "pass; flint 1.3 gives 65 % of its length)")
ARM_LENGTH_M = Val(0.6, N_, "PLANET-SPEC 2.11: v = v_arm (1 + handle_len / 0.6 m)")
V_ARM = Val(5.0, N_, "speed of a hand-held striking stone, a few m/s in experimental knapping (Bril et al. 2010, "
                     "J. Exp. Psychol. Hum. Percept. Perform. 36, 825)")
LIMB_FRACTION = Val(0.022, R_, "hand 0.006 + forearm 0.016 of body mass (Winter 2009, Biomechanics and Motor Control of "
                               "Human Movement, Table 4.1, after Dempster 1955): the striking mass of a bare limb")
BARE_HARDNESS = Val(2.5, R_, "fingernail 2.5 on the Mohs scale (standard Mohs comparison)")
CUT_GAIN = Val(2.0, N_, "a fully sharp edge triples a blow's damage (estimate)")
K_SOFT = Val(1.0, N_, "damage falls as 1 / (1 + K_IC / 1 MPa m^0.5) of the target (estimate)")
BUTCHER_BASE = Val(0.3, N_, "without a cutting edge an eater frees 30 % of what a sharp edge frees (estimate)")
CHOP_BLUNT = Val(0.25, N_, "a blunt tool gathers and breaks 25 % of the wood an edge cuts (estimate)")
CARRY_FRACTION = Val(0.1, N_, "one collected lump or armful is up to 10 % of the body mass")
HANDLE_MAX_K = Val(333.15, R_, "bare skin is burnt within seconds by a surface at about 60 C or more (ASTM C1055, Standard "
                               "Guide for Heated System Surface Conditions that Produce Contact Burn Injuries): collect "
                               "does not pick up a hotter item")
GARMENT_INSULATION = Val(1.0, N_, "a full hide garment adds 1 to the insulation gene (fur garments about 1.5-2.5 clo, "
                                  "ISO 9920; the gene runs 0.2-3)")
HIDE_COVER = Val(0.06, D_, "hide mass covering a body: PLANET-SPEC 2.8 corpses give 0.06 M of hide")
FIBER_PER_HIDE = Val(0.05, N_, "a garment needs fibre of 5 % of its hide mass to be lashed on (estimate)")

# ============================================================================== cooking and decay
COOK_K = Val(343.15, R_, "meat is cooked at 70 C: actin denatures at 66-73 C (Tornberg 2005, Meat Science 70, 493)")
COOK_S = Val(3600.0, N_, "cook holds the item in the fire for an hour")
Q10 = Val(2.0, N_, "PLANET-SPEC 2.7: decomposition Q10 = 2")
DECAY_REF_K = Val(288.0, N_, "PLANET-SPEC 2.7: decomposition rates are given at 288 K")
DECAY_MIN_K = Val(263.15, R_, "microbial growth in foods stops below about -10 C (Jay, Loessner & Golden 2005, Modern Food "
                              "Microbiology, 7th ed.)")
DECAY_MAX_K = Val(333.15, R_, "60 C and above kill vegetative bacteria (pasteurisation, Jay et al. 2005)")
DECAY_OPT_K = Val(310.15, N_, "the Q10 rise is capped at 37 C, the optimum of mesophiles")
_YEAR_D = 365.25
DECAY_PER_DAY = {
    "meat": Val(0.1, N_, "carrion soft tissue is lost within weeks at temperate temperatures (Carter, Yellowlees & "
                         "Tibbett 2007, Naturwissenschaften 94, 12); 0.1/day at 288 K (estimate)"),
    "fat": Val(0.03, N_, "adipose tissue outlasts muscle (adipocere; Carter et al. 2007); 0.03/day (estimate)"),
    "hide": Val(0.05, N_, "raw skin rots slower than muscle (Carter et al. 2007); 0.05/day (estimate)"),
    "bone": Val(0.1 / _YEAR_D, N_, "bones on the surface weather away over 10-15 years (Behrensmeyer 1978, Paleobiology "
                                   "4, 150); 0.1/yr"),
    "wood": Val(0.1 / _YEAR_D, R_, "dead wood decays at 0.01-0.5 /yr (Harmon et al. 1986, Adv. Ecol. Res. 15, 133); 0.1/yr"),
    "plant_fiber": Val(0.3 / _YEAR_D, N_, "PLANET-SPEC 2.7 litter decomposition 0.3/yr"),
}
BINDS = {
    "plant_fiber": Val(1.0, R_, "fibre lashing (flax fibre tensile strength 345-1035 MPa, Bledzki & Gassan 1999)"),
    "resin": Val(1.0, R_, "resin adhesive hafting (compound adhesives in the Middle Stone Age, Wadley 2010, Curr. "
                          "Anthropol. 51, S111)"),
    "hide": Val(1.0, N_, "rawhide lashing, which shrinks tight as it dries"),
}
#: what each of the four collect outputs reaches on the ground (CLASS grouping; products by kind)
PICKUP_GROUP = {s: {"stone": "stone", "ore": "ore", "metal": "ore", "clay": "clay", "wood": "wood", "organic": "wood"}
                .get(mat.CLASS[s], None) for s in mat.SPECIES}
PICKUP_GROUP.update({"ceramic": "stone", "lime": "stone", "glass": "stone", "ash": "clay", "charcoal": "wood"})
PICKUP_RULE = (N_, "collect reaches a loose item whose largest species falls in the chosen group: stone (stone, ceramic, "
                   "lime, glass), ore (ore, metal), clay (clay, ash), wood (wood, organic, charcoal); else the deposit")


# ============================================================================== tables as tensors
_CACHE: dict = {}


def _vec(rows: dict, device, dtype=torch.float32, default=0.0) -> torch.Tensor:
    key = (id(rows), str(device), dtype)
    if key not in _CACHE:
        _CACHE[key] = torch.tensor([float(rows.get(s, default)) for s in mat.SPECIES], dtype=dtype, device=device)
    return _CACHE[key]


def _bed_tables(device) -> dict:
    """Per species of a fire bed: kg of each species made per kg burnt [S, S], net air exchange
    [S, 3] (AIR_GASES, O2 negative), heat released J/kg [S] and the regime (0 inert, 1 flaming,
    2 glowing) [S]. Wood burns in two stages (WOOD_CHAR_RULE); other fuels by ``materials.burn``."""
    key = ("bed", str(device))
    if key in _CACHE:
        return _CACHE[key]
    produce = torch.zeros(S, S, dtype=torch.float64)
    air = torch.zeros(S, 3, dtype=torch.float64)
    heat = torch.zeros(S, dtype=torch.float64)
    regime = torch.zeros(S, dtype=torch.long)
    char = mat.TRANSFORMS[mat.TRANSFORM_INDEX["wood_to_charcoal"]]
    comb = mat.COMBUSTION_J_KG
    for s in mat.FUELS:
        i = IDX[s]
        if s == "wood":
            for o, m in char.outputs.items():
                produce[i, IDX[o]] += m
            for g, m in mat.gas_to_air(char).items():
                air[i, mat.AIR_GASES.index(g)] = m
            heat[i] = float(comb["wood"]) - char.outputs["charcoal"] * float(comb["charcoal"])
            regime[i] = FLAMING
        else:
            b = mat.burn(s)
            produce[i, IDX["ash"]] += b["ash"]
            air[i] = torch.tensor([-b["O2"], b["CO2"], b["H2O"]], dtype=torch.float64)
            heat[i] = float(comb[s])
            regime[i] = GLOWING if s == "charcoal" else FLAMING
    out = {"produce": produce.to(device), "air": air.to(device), "heat": heat.to(device), "regime": regime.to(device),
           "rho": _vec(mat.DENSITY, device, torch.float64), "phi": _vec(PACKING, device, torch.float64),
           "fuel": torch.tensor([s in mat.FUELS for s in mat.SPECIES], device=device),
           "factor": _vec(FUEL_FACTOR, device, torch.float64)}
    _CACHE[key] = out
    return out


def _tt64(device) -> dict:
    """materials.transform_tensors in float64, plus per transform (as Python numbers, so the loop
    needs no device sync) its basis index and its co-inputs [(species index, kg per kg basis)], and
    as [T] tensors its main product (the largest output) ``main`` and that output's yield ``main_yield``."""
    key = ("tt64", str(device))
    if key not in _CACHE:
        tt = mat.transform_tensors(device)
        main = [max(t.outputs, key=t.outputs.get) for t in mat.TRANSFORMS]
        _CACHE[key] = {**tt, **{k: tt[k].double() for k in ("consume", "produce", "air_out", "min_temp_k",
                                                              "enthalpy_j_kg", "days")},
                       "basis_i": [IDX[t.basis] for t in mat.TRANSFORMS],
                       "co": [[(IDX[s], float(v)) for s, v in t.inputs.items() if s != t.basis] for t in mat.TRANSFORMS],
                       "main": torch.tensor([IDX[s] for s in main], dtype=torch.long, device=device),
                       "main_yield": torch.tensor([float(t.outputs[s]) for t, s in zip(mat.TRANSFORMS, main)],
                                                  dtype=torch.float64, device=device)}
    return _CACHE[key]


def _tissue_tables(device) -> dict:
    """[S] float64: 1 for the TISSUE species, and their drying heat (J/kg) H_FG_WATER x MAKEUP water share."""
    key = ("tissue", str(device))
    if key not in _CACHE:
        water = {s: sum(x for c, x in mat.MAKEUP[s] if c == "H2O") for s in TISSUE}
        _CACHE[key] = {"mask": torch.tensor([float(s in TISSUE) for s in mat.SPECIES], dtype=torch.float64, device=device),
                       "dry_j_kg": torch.tensor([float(mat.H_FG_WATER) * water.get(s, 0.0) for s in mat.SPECIES],
                                                dtype=torch.float64, device=device)}
    return _CACHE[key]


def _work_hardness_vec(device) -> torch.Tensor:
    """[S] MOHS x (1 - GRANULAR): what hammering, digging and chopping use (materials' rule)."""
    key = ("work_h", str(device))
    if key not in _CACHE:
        _CACHE[key] = _vec(mat.MOHS, device) * (1 - _vec(mat.GRANULAR, device))
    return _CACHE[key]


def _group_of_species(device) -> torch.Tensor:
    key = ("group", str(device))
    if key not in _CACHE:
        _CACHE[key] = torch.tensor([COLLECT_CLASSES.index(PICKUP_GROUP[s]) if PICKUP_GROUP[s] else -1
                                    for s in mat.SPECIES], dtype=torch.long, device=device)
    return _CACHE[key]


def _class_members(Sd: int, device) -> torch.Tensor:
    """[4, Sd] bool: the deposit species (SPECIES[:Sd]) each collect class digs (by materials.CLASS)."""
    key = ("members", Sd, str(device))
    if key not in _CACHE:
        rows = [[mat.CLASS[mat.SPECIES[s]] == c and c != "wood" for s in range(Sd)] for c in COLLECT_CLASSES]
        _CACHE[key] = torch.tensor(rows, dtype=torch.bool, device=device)
    return _CACHE[key]


WOOD_C_FRACTION = mat.ELEMENT_FRACTION[WOOD][mat.ELEMENTS.index("C")]   # kg C per kg of the wood species


# ============================================================================== small helpers
def _per_world(x, W: int, device, dtype=torch.float32) -> torch.Tensor:
    t = torch.as_tensor(x, dtype=dtype, device=device)
    return t.expand(W) if t.dim() == 0 else t.reshape(W)


def _like(x, shape, device, default=None, dtype=torch.float32) -> torch.Tensor:
    """A scalar, [W] or full-shape value broadcast to ``shape``."""
    if x is None:
        x = default
    t = torch.as_tensor(x, dtype=dtype, device=device)
    if t.dim() == 1 and len(shape) > 1 and t.shape[0] == shape[0]:
        t = t.reshape(shape[0], *([1] * (len(shape) - 1)))
    return t.expand(shape)


def _effort(effort, W, N, device) -> torch.Tensor:
    return _like(effort, (W, N), device, 1.0).clamp(0.0, 1.0)


def _gather(t: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """t [W, I, ...] read at item indices idx [W, ...] (clamped; mask the result where idx < 0)."""
    W = t.shape[0]
    flat = idx.clamp_min(0).reshape(W, -1)
    if t.dim() == 2:
        return t.gather(1, flat).reshape(idx.shape)
    tail = t.shape[2:]
    g = t.gather(1, flat.reshape(W, -1, *([1] * len(tail))).expand(W, flat.shape[1], *tail))
    return g.reshape(*idx.shape, *tail)


def held(inv: torch.Tensor, slot: torch.Tensor) -> torch.Tensor:
    """[W, N] item index in hand ``slot`` [W, N] (-1 where the slot is -1 or empty)."""
    K = inv.shape[2]
    got = inv.gather(2, slot.clamp(0, K - 1)[..., None]).squeeze(-1)
    return torch.where(slot >= 0, got, torch.full_like(got, -1))


def free_slot(inv: torch.Tensor) -> torch.Tensor:
    """[W, N] the first empty hand slot, -1 when all are full."""
    empty = inv < 0
    k = empty.long().argmax(-1)
    return torch.where(empty.any(-1), k, torch.full_like(k, -1))


def _best_slot(score: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """[W, N] the valid slot with the highest score (first on ties), -1 if none."""
    s = torch.where(valid, score.float(), torch.full_like(score.float(), -math.inf))
    k = s.argmax(-1)
    return torch.where(valid.any(-1), k, torch.full_like(k, -1))


def _wn(W, N, device):
    return (torch.arange(W, device=device)[:, None].expand(W, N), torch.arange(N, device=device).expand(W, N))


def arm_power_w(body_mass_kg: torch.Tensor) -> torch.Tensor:
    """Sustained arm power (W) of a body of mass M: ARM_POWER_W (M / 70 kg)^0.75."""
    return float(ARM_POWER_W) * (body_mass_kg.clamp_min(0) / 70.0) ** 0.75


def action_work_j(effort: torch.Tensor, body_mass_kg: torch.Tensor, seconds=ACTION_S) -> torch.Tensor:
    """Mechanical work (J) of an action: effort x arm power x duration."""
    return effort * arm_power_w(body_mass_kg) * float(seconds)


def _nearest(src, dst, src_mask, dst_mask, max_chord: float, pair=None, chunk: int = 1 << 22, compact: bool = True):
    """For each source point src [W, A, 3] the nearest destination dst [W, B, 3] (dst_mask, within
    the chord max_chord, and ``pair(w0, w1)`` [w, A, B] if given). Returns (index [W, A], chord
    [W, A]); -1 and inf where none. Exact Euclidean differences (no matmul expansion), in chunks
    of worlds to bound memory; ties go to the lowest index. When at most half of the sources act or
    half of the destinations are valid, only those rows and columns are compared (the result is the
    same; one host sync for the two counts)."""
    W, A = src.shape[:2]
    B = dst.shape[1]
    idx = torch.full((W, A), -1, dtype=torch.long, device=src.device)
    dist = torch.full((W, A), math.inf, dtype=torch.float32, device=src.device)
    if A == 0 or B == 0:
        return idx, dist
    na, nb = (torch.stack((src_mask.sum(1).max(), dst_mask.sum(1).max())).tolist() if compact   # one host sync
              else (A, B))
    if na == 0 or nb == 0:
        return idx, dist
    if compact and (2 * na <= A or 2 * nb <= B):
        # only the acting sources and the valid destinations (stable order: ties still go to the lowest index)
        rows = torch.sort((~src_mask).to(torch.int8), dim=1, stable=True).indices[:, :na]
        cols = torch.sort((~dst_mask).to(torch.int8), dim=1, stable=True).indices[:, :nb]
        sub_pair = None
        if pair is not None:
            def sub_pair(w0, w1):
                full = pair(w0, w1).expand(w1 - w0, A, B)
                r = rows[w0:w1, :, None].expand(-1, -1, B)
                return full.gather(1, r).gather(2, cols[w0:w1, None, :].expand(-1, na, -1))
        j, d = _nearest(src.gather(1, rows[..., None].expand(-1, -1, 3)), dst.gather(1, cols[..., None].expand(-1, -1, 3)),
                        src_mask.gather(1, rows), dst_mask.gather(1, cols), max_chord, pair=sub_pair, chunk=chunk,
                        compact=False)
        got = torch.where(j >= 0, cols.gather(1, j.clamp_min(0)), torch.full_like(j, -1))
        idx.scatter_(1, rows, got)
        dist.scatter_(1, rows, d)
        return idx, dist
    step = max(1, chunk // (A * B))
    for w0 in range(0, W, step):          # chunks of worlds, not individuals
        w1 = min(W, w0 + step)
        d = torch.cdist(src[w0:w1].float(), dst[w0:w1].float(), compute_mode="donot_use_mm_for_euclid_dist")
        ok = dst_mask[w0:w1, None, :] & src_mask[w0:w1, :, None] & (d <= max_chord)
        if pair is not None:
            ok = ok & pair(w0, w1)
        d = torch.where(ok, d, torch.full_like(d, math.inf))
        dmin, j = d.min(-1)
        found = torch.isfinite(dmin)
        idx[w0:w1] = torch.where(found, j, torch.full_like(j, -1))
        dist[w0:w1] = dmin
    return idx, dist


def _nearest_fire(fires, pos, act, reach_m, radius_m):
    """[W, N] the nearest live fire within reach of each acting actor (-1 none)."""
    f, _ = _nearest(pos, fires.pos, act, fires.alive, float(reach_m) / float(radius_m))
    return f


def _take(pool, inv, w, n, k):
    """Free hand slot k of actors (w, n): the item stays alive (the caller places it)."""
    i = inv[w, n, k]
    inv[w, n, k] = -1
    pool.holder[w, i] = -1
    return i


def _give(pool, inv, w, n, k, i):
    K = inv.shape[2]
    inv[w, n, k] = i
    pool.holder[w, i] = n * K + k


def _remove(pool, inv, w, i) -> torch.Tensor:
    """items.remove that also clears the hand slot of a held item. [M, S] float64 removed."""
    if inv is not None and w.numel():
        K = inv.shape[2]
        code = pool.holder[w, i]
        h = (code >= 0) & pool.alive[w, i]
        inv[w[h], torch.div(code[h], K, rounding_mode="floor"), code[h] % K] = -1
    return it.remove(pool, w, i)


def check_inventory(pool, inv) -> list[str]:
    """Problems between ``inv`` and the pool's ``holder`` codes (empty list = consistent)."""
    W, N, K = inv.shape
    out = []
    wq, nq, kq = (inv >= 0).nonzero(as_tuple=True)
    i = inv[wq, nq, kq]
    if bool((~pool.alive[wq, i]).any()):
        out.append("a hand holds a dead item slot")
    if bool((pool.holder[wq, i] != nq * K + kq).any()):
        out.append("a held item's holder code does not name its hand")
    held_items = pool.alive & (pool.holder >= 0)
    if int(held_items.sum()) != int(wq.numel()):
        out.append("held items and filled hands differ in number")
    return out


def sync_held(pool, pos: torch.Tensor, K: int) -> None:
    """Move every held item to its holder's position pos [W, N, 3] (call after movement)."""
    w, i = (pool.alive & (pool.holder >= 0)).nonzero(as_tuple=True)
    n = torch.div(pool.holder[w, i], K, rounding_mode="floor")
    pool.pos[w, i] = pos[w, n]


def ledger_mass(pool, fires) -> torch.Tensor:
    """[W, S] float64 kg of each species in items plus fire beds (the item ledger)."""
    return it.species_mass(pool) + it.fire_species_mass(fires)


# ============================================================================== item properties
def working_hardness(comp: torch.Tensor, mass: torch.Tensor, head_mass: torch.Tensor,
                     plain: torch.Tensor | None = None) -> torch.Tensor:
    """Hardness a tool works with [...]: an assembly (head_mass > 0) works with its head species,
    taken as the species making up at least HEAD_CORE of its head mass whose mass is closest to the
    head mass (the hardest on a tie), so a heavy handle does not lend the head its hardness; a plain
    item with props' ``tool_hardness`` (hardness x (1 - granular); pass it as ``plain`` when the
    caller already has the props)."""
    hard = _work_hardness_vec(comp.device)
    kg = comp.clamp_min(0) * mass.clamp_min(0)[..., None]
    hm = head_mass[..., None]
    core = (kg >= float(HEAD_CORE) * hm) & (kg > 0)
    dist = torch.where(core, (kg - hm).abs(), torch.full_like(kg, math.inf))
    near = core & (dist <= dist.amin(-1, keepdim=True) + 1e-4 * hm)          # float32 ties
    head_h = torch.where(near, hard.expand_as(kg), torch.full_like(kg, -1.0)).amax(-1).clamp_min(0)
    if plain is None:
        plain = mat.props(comp, mass)["tool_hardness"]
    return torch.where(head_mass > 0, head_h, plain)


def item_props(pool, idx: torch.Tensor) -> dict:
    """materials.props of the items idx [W, ...] (zeros where idx < 0) plus ``present``,
    ``work_hardness``, ``sharp``, ``head_mass``, ``handle_len_m``, ``bond``, ``temp_k``, ``comp``.
    When at most half of the entries hold an item, the properties are worked out for those only and
    the empty entries take an empty item's (one host sync)."""
    ok = (idx >= 0) & _gather(pool.alive, idx)
    flat = ok.reshape(-1)
    if flat.numel() >= 64:
        sel = flat.nonzero(as_tuple=True)[0]
        if 2 * sel.numel() <= flat.numel():
            W = idx.shape[0]
            wsel = torch.div(sel, flat.numel() // W, rounding_mode="floor")
            isel = idx.reshape(-1)[sel]
            q = _item_props_rows(pool, wsel, isel)
            empty = _empty_props(pool.comp.device)
            out = {}
            for k, v in q.items():
                base = empty[k]
                full = base.expand(flat.numel(), *base.shape[1:]).clone()
                full[sel] = v
                out[k] = full.reshape(*idx.shape, *base.shape[1:])
            out["present"] = ok
            return out
    comp = _gather(pool.comp, idx) * ok[..., None]
    mass = _gather(pool.mass, idx) * ok
    head = _gather(pool.head_mass, idx) * ok
    p = mat.props(comp, mass)
    p.update(present=ok, comp=comp, head_mass=head, sharp=_gather(pool.sharp, idx) * ok,
             handle_len_m=_gather(pool.handle_len_m, idx) * ok, bond=_gather(pool.bond, idx) * ok,
             temp_k=_gather(pool.temp_k, idx) * ok,
             work_hardness=working_hardness(comp, mass, head, plain=p["tool_hardness"]) * ok)
    return p


def _item_props_rows(pool, w, i) -> dict:
    """item_props of the live items (w [M], i [M]) as [M] rows (no ``present``)."""
    comp, mass, head = pool.comp[w, i], pool.mass[w, i], pool.head_mass[w, i]
    p = mat.props(comp, mass)
    p.update(comp=comp, head_mass=head, sharp=pool.sharp[w, i], handle_len_m=pool.handle_len_m[w, i],
             bond=pool.bond[w, i], temp_k=pool.temp_k[w, i],
             work_hardness=working_hardness(comp, mass, head, plain=p["tool_hardness"]))
    return p


def _empty_props(device) -> dict:
    """item_props of an empty entry, [1] rows (cached)."""
    key = ("empty_props", str(device))
    if key not in _CACHE:
        p = mat.props(torch.zeros(1, S, device=device), torch.zeros(1, device=device))
        z = torch.zeros(1, device=device)
        p.update(comp=torch.zeros(1, S, device=device), head_mass=z, sharp=z, handle_len_m=z, bond=z, temp_k=z,
                 work_hardness=z)
        _CACHE[key] = p
    return _CACHE[key]


def cooked(pool) -> torch.Tensor:
    """[W, I] bool: the item has been at COOK_K or above (its peak temperature)."""
    return pool.alive & (pool.peak_k >= float(COOK_K))


def cook_factor(pool) -> torch.Tensor:
    """[W, I] multiplier of digestible food energy: props' ``cook_gain`` once cooked, else 1."""
    gain = mat.props(pool.comp, pool.mass * pool.alive)["cook_gain"]
    return torch.where(cooked(pool), gain, torch.ones_like(gain))


# ============================================================================== tool effects
def tool_of(pool, inv, slot: torch.Tensor | None, body_mass_kg: torch.Tensor) -> dict:
    """What an actor strikes, cuts or digs with, [W, N] each: the item in ``slot`` (None = the held
    item with the highest working hardness) or the bare limb (mass LIMB_FRACTION x M, hardness
    BARE_HARDNESS, no edge). Keys: item, head_mass (kg), handle_len_m, sharp, hardness, toughness, mass."""
    p = item_props(pool, inv)
    if slot is None:
        slot = _best_slot(p["work_hardness"], p["present"])
    i = held(inv, slot)
    # the chosen hand's properties, read from the props of all hands (no second props pass)
    k = slot.clamp(0, inv.shape[2] - 1)[..., None]
    q = {name: torch.where(slot >= 0, p[name].gather(-1, k).squeeze(-1), torch.zeros_like(p[name][..., 0]))
         for name in ("head_mass", "mass", "handle_len_m", "sharp", "work_hardness", "toughness")}
    q["present"] = (slot >= 0) & p["present"].gather(-1, k).squeeze(-1)
    bare = ~q["present"]
    head = torch.where(q["head_mass"] > 0, q["head_mass"], q["mass"])
    M = torch.as_tensor(body_mass_kg, dtype=torch.float32, device=inv.device).expand(i.shape)
    return {"item": torch.where(bare, torch.full_like(i, -1), i),
            "head_mass": torch.where(bare, float(LIMB_FRACTION) * M, head),
            "handle_len_m": torch.where(bare, 0.0, q["handle_len_m"]),
            "sharp": torch.where(bare, 0.0, q["sharp"]),
            "hardness": torch.where(bare, float(BARE_HARDNESS), q["work_hardness"]),
            "toughness": q["toughness"], "mass": q["mass"]}


def strike_energy(head_mass: torch.Tensor, handle_len_m: torch.Tensor, v_arm=V_ARM) -> torch.Tensor:
    """Kinetic energy (J) of a blow: 1/2 m_head v^2 with v = v_arm (1 + handle_len / ARM_LENGTH_M)."""
    v = (float(v_arm) if not torch.is_tensor(v_arm) else v_arm) * (1.0 + handle_len_m / float(ARM_LENGTH_M))
    return 0.5 * head_mass * v * v


def strike_damage(energy_j, sharp, tool_hardness, target_hardness, target_toughness) -> torch.Tensor:
    """Damage (J delivered into the target) of a blow: energy x min(1, H_tool / H_target) x
    (1 + CUT_GAIN sharp) x K_SOFT / (K_SOFT + K_IC_target). A softer tool deforms instead of
    transmitting; an edge concentrates the blow; a tough target resists."""
    dev = energy_j.device if torch.is_tensor(energy_j) else None
    th = torch.as_tensor(target_hardness, dtype=torch.float32, device=dev)
    ratio = torch.where(th > 0, (tool_hardness / th.clamp_min(1e-6)).clamp(0, 1), torch.ones_like(th))
    return energy_j * ratio * (1 + float(CUT_GAIN) * sharp) * float(K_SOFT) / (float(K_SOFT) + target_toughness)


def dig_yield(tool_hardness, target_hardness) -> torch.Tensor:
    """Share of a lump that a tool breaks free: min(1, H_tool / H_target) (1 for loose ground)."""
    dev = next((x.device for x in (tool_hardness, target_hardness) if torch.is_tensor(x)), None)
    th = torch.as_tensor(target_hardness, dtype=torch.float32, device=dev)
    tool = torch.as_tensor(tool_hardness, dtype=torch.float32, device=dev)
    return torch.where(th > 0, (tool / th.clamp_min(1e-6)).clamp(0, 1), torch.ones_like(tool * th))


def chop_yield(tool_hardness, sharp, wood_hardness=None) -> torch.Tensor:
    """Share of an armful of wood a tool cuts: min(1, H_tool / H_wood) x (CHOP_BLUNT + (1 - CHOP_BLUNT) sharp)."""
    wh = float(mat.MOHS["wood"]) if wood_hardness is None else wood_hardness
    return dig_yield(tool_hardness, wh) * (float(CHOP_BLUNT) + (1 - float(CHOP_BLUNT)) * sharp)


def butcher_yield(sharp, tool_hardness) -> torch.Tensor:
    """Multiplier (BUTCHER_BASE..1) of the meat an eater frees per day: BUTCHER_BASE + (1 -
    BUTCHER_BASE) x sharp x min(1, H_tool / H_bone) (an edge harder than bone cuts through joints)."""
    return float(BUTCHER_BASE) + (1 - float(BUTCHER_BASE)) * sharp * dig_yield(tool_hardness, float(mat.MOHS["bone"]))


def worn_insulation(pool, inv, body_mass_kg) -> torch.Tensor:
    """[W, N] insulation added by held garments (hide + fibre in one item): GARMENT_INSULATION x
    min(1, hide / (HIDE_COVER M)) x min(1, fibre / (FIBER_PER_HIDE hide)), summed over the hands and
    capped at GARMENT_INSULATION."""
    p = item_props(pool, inv)
    kg = p["comp"] * p["mass"][..., None]
    hide, fiber = kg[..., HIDE], kg[..., FIBER]
    M = torch.as_tensor(body_mass_kg, dtype=torch.float32, device=inv.device)[..., None]
    cover = (hide / (float(HIDE_COVER) * M).clamp_min(1e-9)).clamp(0, 1)
    bind = torch.where(hide > 0, (fiber / (float(FIBER_PER_HIDE) * hide).clamp_min(1e-12)).clamp(0, 1),
                       torch.zeros_like(hide))
    return (float(GARMENT_INSULATION) * cover * bind).sum(-1).clamp_max(float(GARMENT_INSULATION))


# ============================================================================== fire physics
def fire_temperature(fuel_kg: torch.Tensor, air: torch.Tensor, x_o2, t_air_k) -> torch.Tensor:
    """T_fire = T_air + FIRE_DT0_K sqrt(x_O2 / 0.21) x fuel factor x (1 + AIR_GAIN air), batched
    over [W, F] (x_o2 [W], t_air_k scalar, [W] or [W, F]). The fuel factor is the mass-weighted
    FUEL_FACTOR of the bed's fuels (charcoal 1.35, others 1.0)."""
    W = fuel_kg.shape[0]
    bt = _bed_tables(fuel_kg.device)
    kg = fuel_kg.double().clamp_min(0) * bt["fuel"]
    tot = kg.sum(-1)
    factor = torch.where(tot > 0, (kg * bt["factor"]).sum(-1) / tot.clamp_min(1e-30), torch.ones_like(tot))
    x = _per_world(x_o2, W, fuel_kg.device, torch.float64).clamp_min(0)
    shape = fuel_kg.shape[:-1]
    ta = _like(t_air_k, shape, fuel_kg.device, dtype=torch.float64)
    a = torch.as_tensor(air, dtype=torch.float64, device=fuel_kg.device).expand(shape).clamp(0, 1)
    x = x.reshape(W, *([1] * (len(shape) - 1)))
    return (ta + float(FIRE_DT0_K) * (x / float(X_O2_REF)).sqrt() * factor * (1 + float(AIR_GAIN) * a)).float()


def _burn_constant(bed: torch.Tensor, air: torch.Tensor, x_o2: torch.Tensor) -> torch.Tensor:
    """[W, F, S] float64 first-order burning constant (1/s) of each species of the beds.

    The bed is a compact pile of envelope volume V = sum m_s / (PACKING_s rho_s); species s has the
    surface share V_s / V of A = SPHERE_AREA V^(2/3) and loses rho_s beta m^-2 s^-1 (flaming:
    BETA_FLAMING x x_O2/0.21 x (1 + air); the envelope burns back at beta / PACKING) or the
    diffusion-limited char flux h (1 + air) Y_O2 / (cp_g S_CARBON) (glowing). So dm_s/dt = -k_s m_s
    with k_s = flux_s SPHERE_AREA V^(-1/3) / (PACKING_s rho_s): a bed of one fuel loses mass as
    m^(2/3), its rate set by its size."""
    bt = _bed_tables(bed.device)
    W = bed.shape[0]
    kg = bed.clamp_min(0)
    rho = bt["rho"]
    bulk = rho * bt["phi"]
    vtot = (kg / bulk).sum(-1, keepdim=True)
    inv_cbrt = torch.where(vtot > 0, vtot.clamp_min(1e-30) ** (-1.0 / 3.0), torch.zeros_like(vtot))
    x = x_o2.double().reshape(W, 1, 1).clamp_min(0)
    forced = 1 + air.double().clamp(0, 1)[..., None]
    y_o2 = x * molar_mass("O2") / float(M_AIR)
    flux_flame = rho * float(BETA_FLAMING) * (x / float(X_O2_REF)) * forced
    flux_glow = float(H_CONV) * forced * y_o2 / (float(CP_GAS_HOT) * float(S_CARBON))
    reg = bt["regime"]
    flux = torch.where(reg == FLAMING, flux_flame, torch.where(reg == GLOWING, flux_glow, torch.zeros_like(flux_flame)))
    return flux * float(SPHERE_AREA) * inv_cbrt / bulk


def heat_release_w(fires, x_o2) -> torch.Tensor:
    """[W, F] instantaneous heat release (W) of the live fires."""
    W = fires.shape[0]
    bed = fires.fuel_kg.double()
    k = _burn_constant(bed, fires.air, _per_world(x_o2, W, bed.device))
    hrr = (k * bed.clamp_min(0)) @ _bed_tables(bed.device)["heat"]
    return (hrr * fires.alive).float()


def warmth_within(fire_pos, hrr_w, pos, radius_m, *, fire_alive=None, within_m=WARMTH_RADIUS_M) -> torch.Tensor:
    """[W, N] radiant flux (W/m^2) at the actors pos [W, N, 3] from fires within ``within_m``:
    sum of RADIANT_FRACTION x HRR / (4 pi d^2), d at least NEAR_FIRE_M (point-source model).
    ``hrr_w`` [W, F] is the heat release (e.g. ``fire_step``'s ``mean_hrr_w`` for the day)."""
    alive = hrr_w > 0 if fire_alive is None else fire_alive & (hrr_w > 0)
    if pos.shape[1] == 0 or fire_pos.shape[1] == 0 or not bool(alive.any()):
        return torch.zeros(pos.shape[:2], device=pos.device)
    d = torch.cdist(pos.float(), fire_pos.float(), compute_mode="donot_use_mm_for_euclid_dist") * float(radius_m)
    near = alive[:, None, :] & (d <= float(within_m))
    q = float(RADIANT_FRACTION) * hrr_w[:, None, :] / (4 * math.pi * d.clamp_min(float(NEAR_FIRE_M)) ** 2)
    return torch.where(near, q, torch.zeros_like(q)).sum(-1)


def warmth(fires, pos, radius_m, x_o2, within_m=WARMTH_RADIUS_M) -> torch.Tensor:
    """warmth_within from the fires' instantaneous heat release."""
    return warmth_within(fires.pos, heat_release_w(fires, x_o2), pos, radius_m, fire_alive=fires.alive,
                         within_m=within_m)


# ============================================================================== item heating
def _h_coef(t_env, t_item):
    """Convective plus linearised radiative heat transfer coefficient (W/(m^2 K))."""
    return float(H_CONV) + float(EMISSIVITY) * SIGMA * (t_env ** 2 + t_item ** 2) * (t_env + t_item)


def _geometry(p: dict) -> tuple:
    """(m cp, surface, internal time constant) of items as compact spheres, float64.

    tau = m cp / (h A) + r^2 / (pi^2 alpha): the external (lumped) time constant plus the first
    conduction mode of a sphere with a held surface temperature (Carslaw & Jaeger 1959, sec. 9.3);
    exact in both limits of the Biot number."""
    m = p["mass"].double()
    rho = p["density"].double().clamp_min(1e-9)
    cp = p["cp"].double().clamp_min(1e-9)
    k = p["conductivity"].double().clamp_min(1e-9)
    r = (3 * m / (4 * math.pi * rho)).clamp_min(0) ** (1.0 / 3.0)
    area = 4 * math.pi * r * r
    alpha = k / (rho * cp)
    return m * cp, area, r * r / (math.pi ** 2 * alpha)


def _relax(t0, t_env, mcp, area, tau_int, h):
    """Temperature after h seconds of exponential relaxation toward t_env, and the time constant."""
    tau = mcp / (_h_coef(t_env, t0) * area).clamp_min(1e-30) + tau_int
    tau = tau.clamp_min(1e-6)
    return t_env + (t0 - t_env) * torch.exp(-h / tau), tau


def time_above(t0, t_env, tau, h, t_min):
    """Seconds (0..h) that T(t) = t_env + (t0 - t_env) exp(-t / tau) spends at or above t_min."""
    tiny = 1e-30
    above = t0 >= t_min
    up = t_env > t_min
    reach = tau * torch.log((t_env - t0).clamp_min(tiny) / (t_env - t_min).clamp_min(tiny))
    heating = torch.where(above, torch.full_like(reach, h), (h - reach).clamp(0, h))
    leave = tau * torch.log((t0 - t_env).clamp_min(tiny) / (t_min - t_env).clamp_min(tiny))
    cooling = torch.where(above, leave.clamp(0, h), torch.zeros_like(leave))
    level = torch.where(above, torch.full_like(reach, h), torch.zeros_like(reach))  # t_env == t_min
    return torch.where(up, heating, torch.where(t_env == t_min, level, cooling))


def _new_flows(W, F, device) -> dict:
    f64 = lambda *s: torch.zeros(*s, dtype=torch.float64, device=device)
    T = len(mat.TRANSFORMS)
    return {"air_kg": f64(W, 3), "heat_j": f64(W), "heat_j_fire": f64(W, F), "burnt_kg": f64(W, S),
            "transformed_kg": f64(W, T), "transform_heat_j": f64(W), "item_heat_j": f64(W), "to_soil_kg": f64(W, S),
            "charred_kg": f64(W, S), "fires_out": f64(W)}


def _fire_items(pool, inv, bed, fire_t, fire_air, w, i, f, h, flows, budget=None) -> None:
    """Advance items (w, i) in fires (w, f) by h seconds, in place. ``bed`` is the [W, F, S]
    float64 working copy of the fire beds, ``fire_t`` [W, F] the temperature the items meet,
    ``fire_air`` [W, F] the forced air of the step and ``budget`` [W, F] float64 the heat (J) each
    fire can give its items in the step (None = unlimited).

    1. Unloaded, an item relaxes toward fire_t (``_geometry``, ``_relax``); each transform whose
       atmosphere holds (``materials.atmosphere_ok``) converts basis at charge / days for the time
       spent at or above its min_temp_k, the charge being the basis the item started with (basis
       kg + main product kg / its yield); its TISSUE above TISSUE_CHAR_K chars (TISSUE_CHAR_RULE).
    2. Heat: the items of one fire need m cp (T1 - T0) (heating only), the endothermic transform
       heat and the drying heat of charring tissue. Where that exceeds the fire's budget, each of
       its items gets the share s = budget / demand: its rise is scaled by s (a relaxation toward
       T0 + s (fire_t - T0) with the same time constant, so the time above each onset follows that
       slower curve) and its transforms and charring are capped at s x their unloaded amounts. No
       fire heats or converts more than its heat allows (``item_heat_j``).
    3. Transforms are limited by the basis and co-inputs; charcoal comes from the item, then from
       the fire's bed (shared in proportion among the items of one fire). Gases go to
       ``flows['air_kg']``; charred tissue to ``flows['to_soil_kg']`` (and ``charred_kg``).
    4. Fuel species at or above their IGNITION_K move into the bed and burn there, except a
       transform's basis whose atmosphere holds and, in a smothered item
       (``materials.fire_atmosphere``), its charcoal: smothered wood chars and its char stays.
    5. A transformed, charred or melted item loses its edge; one that lost fuel or tissue to the
       fire or melted is no longer an assembly; an item below MIN_ITEM_KG goes into the bed whole."""
    if w.numel() == 0:
        return
    dev = pool.device
    W, F = bed.shape[:2]
    tt = _tt64(dev)
    tis = _tissue_tables(dev)
    T = len(tt["names"])
    comp32, mass32 = pool.comp[w, i], pool.mass[w, i]
    kg = comp32.double().clamp_min(0) * mass32.double()[:, None]
    p = mat.props(comp32, mass32)
    mcp, area, tau_int = _geometry(p)
    radius = (area / (4 * math.pi)).sqrt().clamp_min(1e-9)
    t0 = pool.temp_k[w, i].double()
    te = fire_t[w, f].double()
    t1u, tau = _relax(t0, te, mcp, area, tau_int, h)
    bed_wf = bed[w, f].float()
    atm = mat.atmosphere_ok(bed_wf, fire_air[w, f], mass32, comp32)                               # [M, T]
    smothered = mat.fire_atmosphere(bed_wf, fire_air[w, f], mass32, comp32)["smothered"]          # [M]
    basis0 = kg[:, tt["basis"]]                                                                   # [M, T]
    rate = (basis0 + kg[:, tt["main"]] / tt["main_yield"]) / (tt["days"] * DAY_S)
    tissue = kg * tis["mask"]
    dry_j = tissue @ tis["dry_j_kg"]
    beta = float(BETA_FLAMING)

    def plan(t_env):        # time above each onset, transform amounts and charred share toward t_env
        hot_ = time_above(t0[:, None], t_env[:, None], tau[:, None], h, tt["min_temp_k"][None, :])
        x_ = torch.where(atm, torch.minimum(basis0, rate * hot_), torch.zeros_like(hot_))
        t_ch = time_above(t0, t_env, tau, h, float(TISSUE_CHAR_K))
        return hot_, x_, 1 - (1 - beta * t_ch / radius).clamp_min(0) ** 3

    hot_u, x_u, lost_u = plan(te)
    key = w * F + f
    s = torch.ones_like(t0)
    if budget is not None:
        demand = ((mcp * (t1u - t0)).clamp_min(0) + (x_u * tt["enthalpy_j_kg"].clamp_min(0)).sum(-1)
                  + lost_u * dry_j)
        need = torch.zeros(W * F, dtype=torch.float64, device=dev).index_add_(0, key, demand)
        avail = budget.reshape(-1).double().clamp_min(0)
        share = torch.where(need > avail, avail / need.clamp_min(1e-300), torch.ones_like(need))
        s = share[key]
    heating = te > t0
    t1 = torch.where(heating, t0 + s * (t1u - t0), t1u)
    hot, x_cap, lost = plan(torch.where(heating, t0 + s * (te - t0), te))
    x_cap = torch.minimum(x_cap, s[:, None] * x_u)
    lost = torch.minimum(lost, s * lost_u)
    bed_flat = bed.view(W * F, S)
    converted = torch.zeros(w.numel(), T, dtype=torch.float64, device=dev)
    present0 = basis0 > 0                                                                         # [M, T]
    for t in range(T):            # the transforms (a short fixed list), each vectorised over items
        b = tt["basis_i"][t]
        x = torch.where(atm[:, t], torch.minimum(kg[:, b], x_cap[:, t]), torch.zeros_like(t0))
        c = 0.0
        for sp, per in tt["co"][t]:
            avail_s = kg[:, sp] + (bed_flat[key, CHAR] if sp == CHAR else 0.0)
            x = torch.minimum(x, avail_s / per)
            c = per if sp == CHAR else c
        bed_take = torch.zeros_like(x)
        if c > 0:
            own = kg[:, CHAR]
            want = (x * c - own).clamp_min(0)
            total = torch.zeros(W * F, dtype=torch.float64, device=dev).index_add_(0, key, want)
            ratio = torch.where(total > 0, (bed_flat[:, CHAR] / total.clamp_min(1e-300)).clamp(max=1.0),
                                torch.ones_like(total))
            x = torch.minimum(x, (own + want * ratio[key]) / c)
            bed_take = (x * c - torch.minimum(own, x * c)).clamp_min(0)
            bed_flat[:, CHAR] -= torch.zeros(W * F, dtype=torch.float64, device=dev).index_add_(0, key, bed_take)
            bed_flat[:, CHAR].clamp_min_(0)
        kg = kg - x[:, None] * tt["consume"][t][None, :] + x[:, None] * tt["produce"][t][None, :]
        kg[:, CHAR] += bed_take              # the bed's share of the reductant was not the item's
        kg = kg.clamp_min(0)                 # float64 rounding only
        converted[:, t] = x
        flows["air_kg"].index_add_(0, w, x[:, None] * tt["air_out"][t][None, :])
    flows["transformed_kg"].index_add_(0, w, converted)
    flows["transform_heat_j"].index_add_(0, w, converted @ tt["enthalpy_j_kg"])
    # tissue chars away to the soil at the fire
    charred = tissue * lost[:, None]
    kg = (kg - charred).clamp_min(0)
    flows["to_soil_kg"].index_add_(0, w, charred)
    flows["charred_kg"].index_add_(0, w, charred)
    taken = (mcp * (t1 - t0)).clamp_min(0) + converted @ tt["enthalpy_j_kg"].clamp_min(0) + lost * dry_j
    flows["item_heat_j"].index_add_(0, w, taken)
    # fuels at or above their ignition temperature burn in the bed unless their own transform's
    # atmosphere holds, or they are the char of a smothered item
    bt = _bed_tables(dev)
    protect = atm.double() @ torch.nn.functional.one_hot(tt["basis"], S).double() > 0               # [M, S]
    protect[:, CHAR] |= smothered
    ign = _vec(mat.IGNITION_K, dev, torch.float64)
    burns = bt["fuel"][None, :] & (torch.maximum(t0, t1)[:, None] >= ign[None, :]) & ~protect
    moved = torch.where(burns, kg, torch.zeros_like(kg))
    kg = kg - moved
    mass = kg.sum(-1)
    gone = mass < float(MIN_ITEM_KG)
    moved = moved + torch.where(gone[:, None], kg, torch.zeros_like(kg))
    bed_flat.index_add_(0, key, moved)
    # write back
    keep = ~gone
    wk, ik = w[keep], i[keep]
    mk = mass[keep]
    pool.comp[wk, ik] = (kg[keep] / mk[:, None]).float()
    pool.mass[wk, ik] = mk.float()
    pool.temp_k[wk, ik] = t1[keep].float()
    pool.peak_k[wk, ik] = torch.maximum(pool.peak_k[wk, ik], torch.maximum(t0, t1)[keep].float())
    applicable = torch.where(present0, hot, torch.zeros_like(hot)).amax(-1)
    pool.hot_days[wk, ik] += (applicable[keep] / DAY_S).float()
    melt = torch.maximum(t0, t1) >= p["melt_k"].double()
    lost_kg = charred.sum(-1)
    changed = (converted.sum(-1) > 0) | melt | (lost_kg > 0)
    pool.sharp[wk, ik] = torch.where(changed[keep], 0.0, pool.sharp[wk, ik])
    broke = ((moved.sum(-1) > 0) | melt | (lost_kg > 0))[keep] & (pool.handle_len_m[wk, ik] > 0)
    pool.handle_len_m[wk, ik] = torch.where(broke, 0.0, pool.handle_len_m[wk, ik])
    pool.bond[wk, ik] = torch.where(broke, 0.0, pool.bond[wk, ik])
    pool.head_mass[wk, ik] = torch.where(melt[keep], 0.0, torch.minimum(pool.head_mass[wk, ik], mk.float()))
    if bool(gone.any()):
        _remove(pool, inv, w[gone], i[gone])


def _kill_fires(pool, fires, bed, x, flows, assoc=None) -> tuple:
    """Put out fires with less than MIN_FIRE_KG of fuel or with x_O2 below X_O2_MIN: the residue of
    the bed becomes an item at the fire (holder -1, at the fire's temperature), or goes to
    ``flows['to_soil_kg']`` when it is below MIN_ITEM_KG or the item pool is full. Returns the
    (world, slot) of the residue items."""
    bt = _bed_tables(bed.device)
    fuel = (bed.clamp_min(0) * bt["fuel"]).sum(-1)
    dead = fires.alive & ((fuel < float(MIN_FIRE_KG)) | (x[:, None] < float(X_O2_MIN)))
    if not bool(dead.any()):
        return bed.new_zeros(0, dtype=torch.long), bed.new_zeros(0, dtype=torch.long)
    w, f = dead.nonzero(as_tuple=True)
    residue = bed[w, f].clamp_min(0)
    total = residue.sum(-1)
    big = total >= float(MIN_ITEM_KG)
    idx = torch.full_like(w, -1)
    if bool(big.any()):
        idx[big] = it.spawn(pool, w[big], residue[big].float(), total[big].float(), fires.pos[w[big], f[big]],
                            fires.temp_k[w[big], f[big]].clamp_min(1.0), -1)
        if assoc is not None:
            ok = idx >= 0
            assoc[w[ok], idx[ok]] = -1
    lost = idx < 0
    flows["to_soil_kg"].index_add_(0, w[lost], residue[lost])
    flows["fires_out"].index_add_(0, w, torch.ones_like(total))
    bed[w, f] = 0.0
    it.remove_fire(fires, w, f)
    ok = idx >= 0
    return w[ok], idx[ok]


def fire_step(pool, fires, x_o2, dt_s=DAY_S, *, ambient_item_k, ambient_fire_k, radius_m, inv=None,
              substeps: int = 48, fire_radius_m=FIRE_RADIUS_M) -> dict:
    """Advance every fire and every item by dt_s seconds in ``substeps`` steps (in place). Every
    EXIT_CHECK steps it asks whether a fire is still alive; once none is, the remaining time is one
    relaxation step of the items toward their air (one host sync per check).

    Per step: fires below MIN_FIRE_KG of fuel or at x_O2 < X_O2_MIN go out (``_kill_fires``);
    each live fire takes ``fire_temperature``; each bed burns (``_burn_constant``, exact
    exponential over the step): wood to char and gases, fibre, resin, fat and char to gases and ash,
    releasing COMBUSTION_J_KG; then ground items within ``fire_radius_m`` of a live fire (found
    once, at the start) heat, transform, char and feed the bed (``_fire_items``), all the items of
    one fire taking at most LOAD_SHARE of the heat its bed released in the step; every other item
    relaxes toward ``ambient_item_k`` ([W, I], [W] or scalar: the air at the item; keep held items'
    positions current with ``sync_held``). Forced air (``fires.air``, set by ``feed_fire``) blows for
    the first FAN_S seconds of the call (a step partly inside takes the time-weighted air) and is
    then reset to 0, so call fire_step once after each round of actions.

    x_o2 [W] mole fraction; ambient_fire_k [W, F], [W] or scalar; radius_m the habitat radius.
    Returns float64 flows: ``air_kg`` [W, 3] (O2, CO2, H2O; O2 negative = taken from the air),
    ``o2_mol``/``co2_mol``/``h2o_mol`` [W] (O2 consumed, CO2 and vapour made), ``heat_j`` [W] and
    ``heat_j_fire`` [W, F] combustion heat, ``burnt_kg`` [W, S], ``transformed_kg`` [W, T] (basis kg
    per transform), ``transform_heat_j`` [W], ``item_heat_j`` [W] (heat the items took from the
    fires), ``to_soil_kg`` [W, S] (fire residue with no item slot, and charred tissue),
    ``charred_kg`` [W, S] (the charred tissue alone), ``fires_out`` [W]; and for ``warmth_within``
    the fires' ``fire_pos`` [W, F, 3] and ``mean_hrr_w`` [W, F] of the step."""
    W, I, _ = pool.shape
    F = fires.shape[1]
    dev = pool.device
    flows = _new_flows(W, F, dev)
    x = _per_world(x_o2, W, dev)
    fire_pos = fires.pos.clone()
    ta_item = _like(ambient_item_k, (W, I), dev).double()
    ta_fire = _like(ambient_fire_k, (W, F), dev)
    ground = pool.alive & (pool.holder < 0)
    assoc, _ = _nearest(pool.pos, fires.pos, ground, fires.alive, float(fire_radius_m) / float(radius_m))
    bed = fires.fuel_kg.double()
    bt = _bed_tables(dev)
    h = float(dt_s) / substeps
    geo = [g.clone() for g in _geometry(mat.props(pool.comp, pool.mass * pool.alive))]
    def refresh(wq, iq):            # geometry of items that changed or are new
        if wq.numel():
            g = _geometry(mat.props(pool.comp[wq, iq], pool.mass[wq, iq]))
            for full, part in zip(geo, g):
                full[wq, iq] = part

    fan = fires.air.clone()
    for step in range(substeps):
        refresh(*_kill_fires(pool, fires, bed, x, flows, assoc))
        air = fan * min(1.0, max(0.0, (float(FAN_S) - step * h) / h))
        tf = torch.where(fires.alive, fire_temperature(bed, air, x, ta_fire), torch.zeros_like(ta_fire))
        fires.temp_k.copy_(tf)
        # burning
        k = _burn_constant(bed, air, x)
        burnt = bed.clamp_min(0) * (1 - torch.exp(-k * h)) * fires.alive[..., None]
        bed = bed - burnt + burnt @ bt["produce"]
        flows["burnt_kg"] += burnt.sum(1)
        flows["air_kg"] += (burnt @ bt["air"]).sum(1)
        heat = burnt @ bt["heat"]
        flows["heat_j_fire"] += heat
        flows["heat_j"] += heat.sum(1)
        # the items in the fires, on a share of that heat
        inside = (assoc >= 0) & pool.alive & (pool.holder < 0) & fires.alive.gather(1, assoc.clamp_min(0))
        w, i = inside.nonzero(as_tuple=True)
        _fire_items(pool, inv, bed, tf, air, w, i, assoc[w, i], h, flows, budget=float(LOAD_SHARE) * heat)
        # every other item relaxes toward its ambient air
        ok = pool.alive[w, i]
        refresh(w[ok], i[ok])
        amb = pool.alive & ~inside
        t0 = pool.temp_k.double()
        t1, _ = _relax(t0, ta_item, geo[0], geo[1], geo[2], h)
        t1 = torch.where(amb & (geo[0] > 0), t1, t0)
        pool.temp_k.copy_(t1.float())
        pool.peak_k.copy_(torch.where(amb, torch.maximum(pool.peak_k, pool.temp_k), pool.peak_k))
        done = step + 1
        if done < substeps and done % int(EXIT_CHECK) == 0 and not bool(fires.alive.any()):
            # every fire is out: the rest of the time only relaxes the items toward their air, in one step
            t0 = pool.temp_k.double()
            t1, _ = _relax(t0, ta_item, geo[0], geo[1], geo[2], (substeps - done) * h)
            t1 = torch.where(pool.alive & (geo[0] > 0), t1, t0)
            pool.temp_k.copy_(t1.float())
            pool.peak_k.copy_(torch.where(pool.alive, torch.maximum(pool.peak_k, pool.temp_k), pool.peak_k))
            break
    _kill_fires(pool, fires, bed, x, flows, assoc)
    fires.fuel_kg.copy_(torch.where(fires.alive[..., None], bed, torch.zeros_like(bed)).float())
    fires.air.zero_()
    flows["o2_mol"] = -flows["air_kg"][:, 0] / molar_mass("O2")
    flows["co2_mol"] = flows["air_kg"][:, 1] / molar_mass("CO2")
    flows["h2o_mol"] = flows["air_kg"][:, 2] / molar_mass("H2O")
    flows["fire_pos"] = fire_pos
    flows["mean_hrr_w"] = (flows["heat_j_fire"] / float(dt_s)).float()
    return flows


def _split_fuel(kg: torch.Tensor):
    """Split item species kg [M, S] (float64) into the part that goes into a fire bed (its fuel
    species; everything when the rest is below MIN_ITEM_KG) and the rest. Returns (to_bed, rest,
    rest_kg, whole) with whole [M] true where the item goes in entirely."""
    to_bed = torch.where(_bed_tables(kg.device)["fuel"], kg, torch.zeros_like(kg))
    rest = kg - to_bed
    rest_kg = rest.sum(-1)
    whole = rest_kg < float(MIN_ITEM_KG)
    return torch.where(whole[:, None], kg, to_bed), rest, rest_kg, whole


def _leave_rest(pool, w, i, rest, rest_kg, whole, where_pos) -> None:
    """After its fuel went into a fire: the non-fuel rest of item (w, i) lies in the fire at
    where_pos (an assembly no longer, its handle burnt), or the item is gone when it went in whole.
    The item must already be out of the hand."""
    stay = ~whole
    ws, iS = w[stay], i[stay]
    pool.comp[ws, iS] = (rest[stay] / rest_kg[stay, None]).float()
    pool.mass[ws, iS] = rest_kg[stay].float()
    pool.pos[ws, iS] = where_pos[stay]
    pool.handle_len_m[ws, iS] = 0.0
    pool.bond[ws, iS] = 0.0
    pool.head_mass[ws, iS] = torch.minimum(pool.head_mass[ws, iS], pool.mass[ws, iS])
    it.remove(pool, w[whole], i[whole])


# ============================================================================== actions
def collect(pool, inv, act, choice, pos, cell, stock, wood_kg_m2, cell_area_m2, gen, *, body_mass_kg, item_temp_k,
            radius_m, effort=None, tool_slot=None, reach_m=REACH_M) -> dict:
    """Collect into the first free hand: the nearest loose item within reach whose largest species
    is in the chosen group (PICKUP_RULE; the lowest actor index wins an item two reach for), else a
    lump from the cell's ground. An item hotter than HANDLE_MAX_K (in or fresh from a fire) is not
    taken into the hand: an actor that holds another item rakes it out with that item onto the
    ground at its feet (``raked``), where it cools; a bare-handed actor leaves it.

    choice [W, N] long: 0 stone, 1 ore, 2 clay, 3 wood (COLLECT_CLASSES). stock [W, C, Sd] kg/m^2 of
    SPECIES[:Sd] (``globe.make_deposits`` order), reduced in place (keep it float64: a lump of a few kg
    is below float32 resolution of a land stock of 1e3 kg/m^2 on a 1e5 m^2 cell); wood_kg_m2 [W, C] collectable
    wood (kg of the wood species per m^2), not changed: subtract the returned ``wood_kg`` (its carbon
    is WOOD_C_FRACTION). cell_area_m2 [C] or [W, C]; item_temp_k [W, N] the temperature of a new
    lump. The species of a ground lump is drawn in proportion to the cell's stock of the class
    (one ``gen`` draw per actor every call). Its mass is CARRY_FRACTION x M x effort x yield:
    ``dig_yield(tool, MOHS x (1 - granular) of the species)`` or ``chop_yield`` for wood, where the
    tool is ``tool_slot`` (None = the held item with the highest working hardness, or the bare
    hand). Actors of one cell share a short stock in proportion to their demand.

    Returns: item [W, N] (-1 none; the raked item for a rake), kg [W, N], picked [W, N] bool (a loose
    item, now in the hand), raked [W, N] bool, species [W, N] (-1 none), ground_kg [W, S] float64
    taken from the deposits, wood_kg [W, C] float64, work_j."""
    W, N, K = inv.shape
    dev = pool.device
    u = torch.rand(W, N, generator=gen, device=dev)
    free = free_slot(inv)
    act = act & (free >= 0)
    eff = _effort(effort, W, N, dev)
    M = _like(body_mass_kg, (W, N), dev)
    choice = choice.long().clamp(0, 3)
    tool = tool_of(pool, inv, tool_slot, M)
    _, ni = _wn(W, N, dev)
    out_item = torch.full((W, N), -1, dtype=torch.long, device=dev)
    out_kg = torch.zeros(W, N, device=dev)
    out_species = torch.full((W, N), -1, dtype=torch.long, device=dev)
    # 1. loose items within reach
    dom = pool.comp.argmax(-1)
    group = torch.where(pool.alive, _group_of_species(dev)[dom], torch.full_like(dom, -1))
    ground = pool.alive & (pool.holder < 0)
    cool = pool.temp_k <= float(HANDLE_MAX_K)
    holds = (inv >= 0).any(-1)
    cand, _ = _nearest(pos, pool.pos, act, ground, float(reach_m) / float(radius_m),
                       pair=lambda w0, w1: (group[w0:w1, None, :] == choice[w0:w1, :, None])
                       & (cool[w0:w1, None, :] | holds[w0:w1, :, None]))
    valid = cand >= 0
    win = torch.full(pool.alive.shape, N, dtype=torch.long, device=dev)
    win.scatter_reduce_(1, cand.clamp_min(0), torch.where(valid, ni, torch.full_like(ni, N)), reduce="amin")
    got_one = valid & (win.gather(1, cand.clamp_min(0)) == ni)
    hot = ~cool.gather(1, cand.clamp_min(0))
    picked, raked = got_one & ~hot, got_one & hot
    wp, np_ = picked.nonzero(as_tuple=True)
    ip = cand[wp, np_]
    _give(pool, inv, wp, np_, free[wp, np_], ip)
    wr, nr = raked.nonzero(as_tuple=True)
    pool.pos[wr, cand[wr, nr]] = pos[wr, nr]
    out_item = torch.where(got_one, cand, out_item)
    out_kg = torch.where(got_one, pool.mass.gather(1, cand.clamp_min(0)), out_kg)
    out_species = torch.where(got_one, dom.gather(1, cand.clamp_min(0)), out_species)
    # 2. a lump from the ground
    dig = act & ~got_one
    Sd = stock.shape[-1]
    C = stock.shape[1]
    local = stock.gather(1, cell.long()[..., None].expand(W, N, Sd)).clamp_min(0)          # [W, N, Sd]
    member = _class_members(Sd, dev)[choice]                                                  # [W, N, Sd]
    weight = (local * member).double()
    cum = weight.cumsum(-1)
    tot = cum[..., -1]
    pick = (cum <= (u.double() * tot)[..., None]).sum(-1).clamp(max=Sd - 1)
    is_wood = choice == 3
    sp = torch.where(is_wood, torch.full_like(pick, WOOD), pick)
    hard = _work_hardness_vec(dev)[sp]
    yld = torch.where(is_wood, chop_yield(tool["hardness"], tool["sharp"]), dig_yield(tool["hardness"], hard))
    want = (float(CARRY_FRACTION) * M * eff * yld).double()
    area = torch.as_tensor(cell_area_m2, dtype=torch.float64, device=dev)
    area = area.reshape(1, C).expand(W, C) if area.dim() == 1 else area.expand(W, C)
    a_cell = area.gather(1, cell.long())
    wood_here = wood_kg_m2.double().gather(1, cell.long()).clamp_min(0)
    stock_here = local.double().gather(2, pick[..., None]).squeeze(-1)
    avail = torch.where(is_wood, wood_here, stock_here) * a_cell
    has = dig & (torch.where(is_wood, wood_here, tot) > 0) & (want > 0)
    wd, nd = has.nonzero(as_tuple=True)
    col = torch.where(is_wood, torch.full_like(pick, Sd), pick)[wd, nd]
    key = (wd * C + cell.long()[wd, nd]) * (Sd + 1) + col
    uk, inv_k = torch.unique(key, return_inverse=True)
    demand = torch.zeros(uk.numel(), dtype=torch.float64, device=dev).index_add_(0, inv_k, want[wd, nd])
    avail_k = torch.zeros_like(demand).scatter_(0, inv_k, avail[wd, nd])      # the same value within a key
    ratio = torch.where(demand > 0, (avail_k / demand.clamp_min(1e-300)).clamp(max=1.0), torch.zeros_like(demand))
    take = want[wd, nd] * ratio[inv_k]
    ok = take >= float(MIN_ITEM_KG)
    comp = torch.nn.functional.one_hot(sp[wd, nd], S).float()
    idx = torch.full_like(wd, -1)
    if bool(ok.any()):
        idx[ok] = it.spawn(pool, wd[ok], comp[ok], take[ok].float(), pos[wd[ok], nd[ok]],
                           _like(item_temp_k, (W, N), dev)[wd[ok], nd[ok]], -1)
    got = idx >= 0
    wg, ng, ig = wd[got], nd[got], idx[got]
    _give(pool, inv, wg, ng, free[wg, ng], ig)
    kg_got = pool.mass[wg, ig].double()          # what the item holds (float32), booked exactly
    out_item[wg, ng] = ig
    out_kg[wg, ng] = pool.mass[wg, ig]
    out_species[wg, ng] = sp[wg, ng]
    taken_k = torch.zeros_like(demand).index_add_(0, inv_k[got], kg_got)
    ground_kg = torch.zeros(W, S, dtype=torch.float64, device=dev)
    rock = ~is_wood[wg, ng]
    ground_kg.index_put_((wg[rock], sp[wg[rock], ng[rock]]), kg_got[rock], accumulate=True)
    wood_kg = torch.zeros(W, C, dtype=torch.float64, device=dev)
    wood_kg.index_put_((wg[~rock], cell.long()[wg[~rock], ng[~rock]]), kg_got[~rock], accumulate=True)
    # the stock of each touched (world, cell, species) keeps what was not taken
    kw = torch.div(uk, C * (Sd + 1), rounding_mode="floor")
    kc = torch.div(uk, Sd + 1, rounding_mode="floor") % C
    ks = uk % (Sd + 1)
    rk = ks < Sd
    left = (avail_k - taken_k).clamp_min(0) / area[kw, kc].clamp_min(1e-30)
    stock[kw[rk], kc[rk], ks[rk]] = left[rk].to(stock.dtype)       # keep a float64 stock in float64
    work = action_work_j(eff, M) * act
    return {"item": out_item, "kg": out_kg, "picked": picked, "raked": raked, "species": out_species,
            "ground_kg": ground_kg,
            "wood_kg": wood_kg, "work_j": work}


def drop(pool, inv, act, pos, slot=None) -> dict:
    """Put the item in ``slot`` (None = the first filled hand) on the ground at the actor."""
    W, N, K = inv.shape
    dev = pool.device
    if slot is None:
        slot = _best_slot(-torch.arange(K, device=dev).float().expand(W, N, K), inv >= 0)
    item = held(inv, slot)
    go = act & (item >= 0)
    w, n = go.nonzero(as_tuple=True)
    i = _take(pool, inv, w, n, slot[w, n])
    pool.pos[w, i] = pos[w, n]
    return {"item": torch.where(go, item, torch.full_like(item, -1))}


def _knappable(p: dict) -> tuple:
    """The part of items (item_props ``p``) that knapping works on: the whole item, or for an
    assembly (head_mass > 0) its brittle species only (materials.BRITTLE), so its handle and binder
    are never knapped. Returns (part kg [..., S], part mass [...], props of the part, knappable
    [...] bool: brittle kg >= BRITTLE_MIN x the item's mass (an assembly: x its head mass), edge
    limit [...] 1 / (1 + (K_IC_part / K_EDGE)^2))."""
    brit = _vec(mat.BRITTLE, p["comp"].device)
    kg = p["comp"] * p["mass"][..., None]
    assembly = p["head_mass"] > 0
    part = torch.where(assembly[..., None], kg * brit, kg)
    part_kg = part.sum(-1)
    q = mat.props(part, part_kg)
    basis = torch.where(assembly, p["head_mass"], p["mass"])
    ok = p["present"] & (part_kg > 0) & ((kg * brit).sum(-1) >= float(BRITTLE_MIN) * basis)
    return part, part_kg, q, ok, 1.0 / (1.0 + (q["toughness"] / float(K_EDGE)) ** 2)


def knap(pool, inv, act, pos, *, body_mass_kg, hammer_slot=None, core_slot=None, effort=None) -> dict:
    """Strike a held brittle core with a held hammer.

    The core's knappable part is the whole item, or for an assembly only its brittle species
    (``_knappable``: retouching a hafted head leaves the handle alone). It works when that part is
    brittle enough and its scratch hardness is at most KNAP_HARDNESS_RATIO x the hammer's working
    hardness. Then a share KNAP_LOSS x effort of the knappable mass goes to a debris item on the
    ground (a fresh flake of the part: sharpness at its limit; an assembly's head mass falls by as
    much) and the core's sharpness rises toward the part's limit lim = 1 / (1 + (K_IC / K_EDGE)^2)
    as lim - (lim - s) exp(-share / KNAP_EDGE_SHARE). Defaults: the core is the knappable held item
    with the highest sharpness limit (the one that takes the best edge), the hammer the hardest
    other held item. No debris slot (pool full): nothing happens. body_mass_kg [W, N] gives the
    work. Returns ok, debris, removed_kg, sharp, work_j [W, N]."""
    W, N, K = inv.shape
    dev = pool.device
    p = item_props(pool, inv)
    if core_slot is None:
        _, _, _, can, edge = _knappable(p)
        core_slot = _best_slot(edge, can)
    if hammer_slot is None:
        other = p["present"] & (torch.arange(K, device=dev) != core_slot[..., None])
        hammer_slot = _best_slot(p["work_hardness"], other)
    core, ham = held(inv, core_slot), held(inv, hammer_slot)
    c, hm = item_props(pool, core), item_props(pool, ham)
    part, part_kg, cq, can, lim = _knappable(c)
    eff = _effort(effort, W, N, dev)
    ok = (act & can & hm["present"] & (core_slot != hammer_slot)
          & (cq["hardness"] <= float(KNAP_HARDNESS_RATIO) * hm["work_hardness"]) & (eff > 0))
    share = float(KNAP_LOSS) * eff
    removed = share * part_kg
    ok = ok & (removed >= float(MIN_ITEM_KG)) & (c["mass"] - removed >= float(MIN_ITEM_KG))
    flake = part / part_kg.clamp_min(1e-30)[..., None]
    w, n = ok.nonzero(as_tuple=True)
    debris = torch.full((W, N), -1, dtype=torch.long, device=dev)
    if w.numel():
        d = it.spawn(pool, w, flake[w, n], removed[w, n], pos[w, n], c["temp_k"][w, n], -1, sharp=lim[w, n])
        debris[w, n] = d
    done = debris >= 0
    w, n = done.nonzero(as_tuple=True)
    i = core[w, n]
    rem = removed[w, n]
    m_new = c["mass"][w, n] - rem
    assembly = c["head_mass"][w, n] > 0
    kg_new = (c["comp"][w, n] * c["mass"][w, n][:, None] - rem[:, None] * flake[w, n]).clamp_min(0)
    pool.comp[w, i] = torch.where(assembly[:, None], kg_new / m_new[:, None], pool.comp[w, i])
    pool.mass[w, i] = m_new
    head = torch.where(assembly, (c["head_mass"][w, n] - rem).clamp_min(float(MIN_ITEM_KG)), torch.zeros_like(rem))
    pool.head_mass[w, i] = torch.minimum(head, m_new)
    new_sharp = lim - (lim - c["sharp"]) * torch.exp(-share / float(KNAP_EDGE_SHARE))
    pool.sharp[w, i] = torch.maximum(pool.sharp[w, i], new_sharp[w, n]).clamp(0, 1)
    work = action_work_j(eff, _like(body_mass_kg, (W, N), dev)) * act
    return {"ok": done, "debris": debris, "removed_kg": torch.where(done, removed, torch.zeros_like(removed)),
            "sharp": torch.where(done, _gather(pool.sharp, core), c["sharp"]), "work_j": work}


def handle_length(mass, density, toughness) -> torch.Tensor:
    """Lever length (m) of a handle: a rod of HANDLE_ASPECT, L = (4 V lambda^2 / pi)^(1/3), shortened
    by min(1, K_IC / K_ROD) for brittle stuff."""
    vol = mass / density.clamp_min(1e-9)
    rod = (4 * vol * float(HANDLE_ASPECT) ** 2 / math.pi).clamp_min(0) ** (1.0 / 3.0)
    return rod * (toughness / float(K_ROD)).clamp(0, 1)


def combine(pool, inv, act, *, head_slot=None, handle_slot=None, binder_slot=None) -> dict:
    """Join a head, a handle and optionally a binder held in three hands into one item (in the
    head's hand). Mass and species are summed; the heat content is kept (temperature by m cp).
    ``head_mass`` is the head's (its own head mass if it is already an assembly), ``handle_len_m``
    is ``handle_length`` of the handle, ``bond`` = BOND_FIT + (1 - BOND_FIT) x b x (1 - exp(-m_binder
    / (BINDER_SHARE m_head))) with b the binder's mass-weighted BINDS; ``sharp`` stays the head's.
    Defaults: the head is the hardest held item, the handle the toughest other one, the binder the
    third hand if it binds at all (binder_slot -1 = none). Returns item, bond, handle_len_m [W, N]."""
    W, N, K = inv.shape
    dev = pool.device
    p = item_props(pool, inv)
    ks = torch.arange(K, device=dev)
    if head_slot is None:
        head_slot = _best_slot(p["work_hardness"], p["present"])
    if handle_slot is None:
        handle_slot = _best_slot(p["toughness"], p["present"] & (ks != head_slot[..., None]))
    binds = (p["comp"] * _vec(BINDS, dev)).sum(-1)
    if binder_slot is None:
        rest = p["present"] & (ks != head_slot[..., None]) & (ks != handle_slot[..., None]) & (binds > 0)
        binder_slot = _best_slot(binds, rest)
    hi, di, bi = held(inv, head_slot), held(inv, handle_slot), held(inv, binder_slot)
    hp, dp, bp = item_props(pool, hi), item_props(pool, di), item_props(pool, bi)
    bi = torch.where((binder_slot != head_slot) & (binder_slot != handle_slot), bi, torch.full_like(bi, -1))
    has_b = bi >= 0
    ok = act & hp["present"] & dp["present"] & (head_slot != handle_slot)
    mb = torch.where(has_b, bp["mass"], torch.zeros_like(bp["mass"]))
    mass = hp["mass"] + dp["mass"] + mb
    kg = hp["comp"] * hp["mass"][..., None] + dp["comp"] * dp["mass"][..., None] + bp["comp"] * mb[..., None]
    head = torch.where(hp["head_mass"] > 0, hp["head_mass"], hp["mass"])
    b = (bp["comp"] * _vec(BINDS, dev)).sum(-1) * has_b
    bond = float(BOND_FIT) + (1 - float(BOND_FIT)) * b * (1 - torch.exp(-mb / (float(BINDER_SHARE) * head).clamp_min(1e-12)))
    length = handle_length(dp["mass"], dp["density"], dp["toughness"])
    hc, dc, bc = (q["heat_capacity_j_k"] for q in (hp, dp, bp))
    bc = bc * has_b
    temp = (hc * hp["temp_k"] + dc * dp["temp_k"] + bc * bp["temp_k"]) / (hc + dc + bc).clamp_min(1e-12)
    w, n = ok.nonzero(as_tuple=True)
    i = hi[w, n]
    pool.comp[w, i] = kg[w, n] / mass[w, n][:, None]
    pool.mass[w, i] = mass[w, n]
    pool.head_mass[w, i] = head[w, n]
    pool.handle_len_m[w, i] = length[w, n]
    pool.bond[w, i] = bond[w, n]
    pool.temp_k[w, i] = temp[w, n]
    peaks = torch.stack([_gather(pool.peak_k, x) * (x >= 0) for x in (hi, di, bi)], -1).amax(-1)
    ages = torch.stack([_gather(pool.age_d, x) * (x >= 0) for x in (hi, di, bi)], -1).amax(-1)
    pool.peak_k[w, i] = peaks[w, n]
    pool.age_d[w, i] = ages[w, n]
    _remove(pool, inv, w, di[w, n])
    wb = ok & has_b
    _remove(pool, inv, wb.nonzero(as_tuple=True)[0], bi[wb])
    return {"item": torch.where(ok, hi, torch.full_like(hi, -1)), "bond": torch.where(ok, bond, torch.zeros_like(bond)),
            "handle_len_m": torch.where(ok, length, torch.zeros_like(length))}


def make_fire(pool, fires, inv, act, pos, x_o2, air_k, *, body_mass_kg, effort=None, fuel_slot=None) -> dict:
    """Light a fire from a held fuel item at the actor's position.

    Ignition: the friction of a drill brings its tip to T = air_k + FRICTION_EFFICIENCY x effort x
    arm power / (8 DRILL_RADIUS_M k_fuel) (FRICTION_RULE), which must reach the item's
    ``ignition_k``; or the actor holds a pyrite item (PYRITE_MIN) and, in another hand, a striker as
    hard as pyrite (working hardness >= MOHS pyrite), and the fuel catches a spark (ignition_k <=
    SPARK_TINDER_K). Either way the O2 mole fraction x_o2 [W] must be at least X_O2_MIN (Belcher
    et al. 2010); otherwise the fire fails and the item stays held. The item's fuel species become
    the fire's bed (a non-fuel rest of at least MIN_ITEM_KG, such as a hafted head, lies in the new
    fire), the fire starts at ``fire_temperature``. fuel_slot None = the held item with the most fuel
    energy. An item with less than MIN_FIRE_KG of fuel species makes no fire (``no_fuel``; it would
    go out at once). air_k [W, N] the air at the actor; no fire pool slot left: the fire fails too.
    Returns fire [W, N] (slot or -1), lit, no_fuel, no_oxygen, no_ignition, spark [W, N] bool,
    contact_k, work_j."""
    W, N, K = inv.shape
    dev = pool.device
    p = item_props(pool, inv)
    if fuel_slot is None:
        fuel_slot = _best_slot(p["fuel_j"], p["present"] & (p["fuel_j"] > 0))
    item = held(inv, fuel_slot)
    q = item_props(pool, item)
    eff = _effort(effort, W, N, dev)
    M = _like(body_mass_kg, (W, N), dev)
    ta = _like(air_k, (W, N), dev)
    x = _per_world(x_o2, W, dev)
    fuel_kg = (q["comp"] * q["mass"][..., None] * _bed_tables(dev)["fuel"]).sum(-1)
    fuel = act & q["present"] & (q["fuel_j"] > 0) & (fuel_kg >= float(MIN_FIRE_KG))
    contact = ta + float(FRICTION_EFFICIENCY) * eff * arm_power_w(M) / (8 * float(DRILL_RADIUS_M)
                                                                          * q["conductivity"].clamp_min(1e-6))
    others = p["present"] & (torch.arange(K, device=dev) != fuel_slot[..., None])
    pyrite = others & (p["comp"][..., PYRITE] >= float(PYRITE_MIN))
    striker = others & (p["work_hardness"] >= float(mat.MOHS["pyrite"]))
    pair = (pyrite[..., :, None] & striker[..., None, :] & ~torch.eye(K, dtype=torch.bool, device=dev)).any(-1).any(-1)
    spark = pair & (q["ignition_k"] <= float(SPARK_TINDER_K))
    ignites = (contact >= q["ignition_k"]) | spark
    oxygen = (x >= float(X_O2_MIN))[:, None].expand(W, N)
    go = fuel & ignites & oxygen
    w, n = go.nonzero(as_tuple=True)
    fire = torch.full((W, N), -1, dtype=torch.long, device=dev)
    if w.numel():
        to_bed, rest, rest_kg, whole = _split_fuel(q["comp"][w, n].double() * q["mass"][w, n].double()[:, None])
        temp = fire_temperature(to_bed[:, None, :], torch.zeros(w.numel(), 1, device=dev), x[w], ta[w, n][:, None])[:, 0]
        fire[w, n] = it.spawn_fire(fires, w, pos[w, n], to_bed.float(), temp, air=0.0)
        lit = fire[w, n] >= 0
        wl, nl = w[lit], n[lit]
        il = _take(pool, inv, wl, nl, fuel_slot[wl, nl])
        _leave_rest(pool, wl, il, rest[lit], rest_kg[lit], whole[lit], pos[wl, nl])
    lit = fire >= 0
    return {"fire": fire, "lit": lit, "no_fuel": act & ~fuel, "no_oxygen": fuel & ~oxygen,
            "no_ignition": fuel & oxygen & ~ignites,
            "spark": go & spark & (contact < q["ignition_k"]), "contact_k": torch.where(fuel, contact, ta),
            "work_j": action_work_j(eff, M) * act}


def feed_fire(pool, fires, inv, act, pos, *, radius_m, effort=None, fan=None, slot=None, reach_m=REACH_M) -> dict:
    """Feed or fan the nearest live fire within reach. The fuel species of the item in ``slot``
    (None = the held item with the most fuel energy) go into the bed; a non-fuel rest of at least
    MIN_ITEM_KG stays an item lying in the fire, a smaller rest goes into the bed too. ``fan``
    [W, N] in [0, 1] is the forced air the actor blows (bellows, blowpipe, fanning) for FAN_S of the
    next ``fire_step``; the fire takes the strongest fan on it. None: an actor that feeds does not
    fan, one with no fuel to feed fans at its ``effort`` (None = 1). Returns fire, fed_kg, fan [W, N]."""
    W, N, K = inv.shape
    dev = pool.device
    eff = _effort(effort, W, N, dev)
    p = item_props(pool, inv)
    if slot is None:
        slot = _best_slot(p["fuel_j"], p["present"] & (p["fuel_j"] > 0))
    fire = _nearest_fire(fires, pos, act, reach_m, radius_m)
    F = fires.shape[1]
    at = act & (fire >= 0)
    item = held(inv, slot)
    q = item_props(pool, item)
    feed = at & q["present"] & (q["fuel_j"] > 0)
    blow = torch.where(feed, torch.zeros_like(eff), eff) if fan is None else _effort(fan, W, N, dev)
    blow = torch.where(at, blow, torch.zeros_like(blow))
    wa, na = at.nonzero(as_tuple=True)
    fires.air.view(-1).scatter_reduce_(0, wa * F + fire[wa, na], blow[wa, na], reduce="amax")
    w, n = feed.nonzero(as_tuple=True)
    fed = torch.zeros(W, N, device=dev)
    if w.numel():
        f = fire[w, n]
        i = item[w, n]
        to_bed, rest, rest_kg, whole = _split_fuel(q["comp"][w, n].double() * q["mass"][w, n].double()[:, None])
        fires.fuel_kg.view(-1, S).index_add_(0, w * F + f, to_bed.float())
        fed[w, n] = to_bed.sum(-1).float()
        _take(pool, inv, w, n, slot[w, n])
        _leave_rest(pool, w, i, rest, rest_kg, whole, fires.pos[w, f])
    return {"fire": torch.where(at, fire, torch.full_like(fire, -1)), "fed_kg": fed, "fan": blow}


def heat_item(pool, fires, inv, act, pos, *, radius_m, slot=None, reach_m=REACH_M) -> dict:
    """Put the item in ``slot`` (None = the first filled hand) into the nearest live fire within
    reach: it lies in the bed at the fire's centre, where ``fire_step`` heats it, transforms it or
    burns its fuel. Pick it up again with ``collect``. Returns fire, item [W, N]."""
    W, N, K = inv.shape
    dev = pool.device
    if slot is None:
        slot = _best_slot(-torch.arange(K, device=dev).float().expand(W, N, K), inv >= 0)
    item = held(inv, slot)
    fire = _nearest_fire(fires, pos, act & (item >= 0), reach_m, radius_m)
    go = act & (item >= 0) & (fire >= 0)
    w, n = go.nonzero(as_tuple=True)
    i = _take(pool, inv, w, n, slot[w, n])
    pool.pos[w, i] = fires.pos[w, fire[w, n]]
    return {"fire": torch.where(go, fire, torch.full_like(fire, -1)), "item": torch.where(go, item, torch.full_like(item, -1))}


def cook(pool, fires, inv, act, pos, x_o2, *, radius_m, ambient_fire_k, slot=None, seconds=COOK_S,
         reach_m=REACH_M) -> dict:
    """Hold the item in ``slot`` (None = the held item with the most food energy) at roasting
    distance from the nearest live fire within reach for ``seconds``: the physics of an item in a
    fire (``_fire_items``) toward T_air + ROAST_SHARE (T_fire - T_air), on LOAD_SHARE of the fire's
    present heat release (shared by everyone cooking at it; the bed's own burning is left to
    ``fire_step``). Meat that reaches COOK_K is cooked (``cook_factor``). The item stays in the hand
    unless it burns away. Returns fire, item, cooked [W, N] and the flows of ``fire_step``'s kinds
    (book ``air_kg`` and ``to_soil_kg`` as for fire_step)."""
    W, N, K = inv.shape
    dev = pool.device
    F = fires.shape[1]
    p = item_props(pool, inv)
    if slot is None:
        slot = _best_slot(p["food_j"], p["present"] & (p["food_j"] > 0))
    item = held(inv, slot)
    fire = _nearest_fire(fires, pos, act & (item >= 0), reach_m, radius_m)
    go = act & (item >= 0) & (fire >= 0)
    flows = _new_flows(W, F, dev)
    x = _per_world(x_o2, W, dev)
    bed = fires.fuel_kg.double()
    ta = _like(ambient_fire_k, (W, F), dev)
    tf = fire_temperature(bed, fires.air, x, ta)
    roast = torch.where(fires.alive, ta + float(ROAST_SHARE) * (tf - ta), torch.zeros(W, F, device=dev))
    steps = max(1, math.ceil(float(seconds) / 900.0))
    h = float(seconds) / steps
    budget = float(LOAD_SHARE) * heat_release_w(fires, x).double() * h
    w, n = go.nonzero(as_tuple=True)
    for _ in range(steps):
        i = item[w, n]
        live = pool.alive[w, i] & (pool.holder[w, i] >= 0)
        _fire_items(pool, inv, bed, roast, fires.air, w[live], i[live], fire[w[live], n[live]], h, flows, budget=budget)
    fires.fuel_kg.copy_(torch.where(fires.alive[..., None], bed, torch.zeros_like(bed)).float())
    still = go & _gather(pool.alive, item) & (_gather(pool.holder, item) >= 0)
    flows.update(fire=torch.where(go, fire, torch.full_like(fire, -1)), item=torch.where(still, item, torch.full_like(item, -1)),
                 cooked=still & (_gather(pool.peak_k, item) >= float(COOK_K)))
    return flows


def share(pool, inv, act, pos, uid, parent, alive, *, radius_m, slot=None, reach_m=REACH_M) -> dict:
    """Give the item in ``slot`` (None = the first filled hand) to the nearest living relative
    within reach that has a free hand. Relatives: parent and child (``parent`` [W, N] holds the
    parent's ``uid``) and siblings (same parent). A receiver takes gifts into its free hands in the
    givers' index order; a gift it has no hand for is not given. Returns to, item [W, N]."""
    W, N, K = inv.shape
    dev = pool.device
    if slot is None:
        slot = _best_slot(-torch.arange(K, device=dev).float().expand(W, N, K), inv >= 0)
    item = held(inv, slot)
    giver = act & (item >= 0) & alive
    nfree = (inv < 0).sum(-1)
    eye = torch.eye(N, dtype=torch.bool, device=dev)

    def kin(w0, w1):
        u, pa = uid[w0:w1], parent[w0:w1]
        child = (pa[:, None, :] == u[:, :, None]) & (u[:, :, None] >= 0)     # j's parent is i
        par = (pa[:, :, None] == u[:, None, :]) & (u[:, None, :] >= 0)       # i's parent is j
        sib = (pa[:, :, None] == pa[:, None, :]) & (pa[:, :, None] >= 0)
        return (child | par | sib) & ~eye

    tgt, _ = _nearest(pos, pos, giver, alive & (nfree > 0), float(reach_m) / float(radius_m), pair=kin)
    go = giver & (tgt >= 0)
    wg, ng = go.nonzero(as_tuple=True)
    r = tgt[wg, ng]
    key = wg * N + r
    order = torch.argsort(key, stable=True)
    ks = key[order]
    first = torch.ones_like(ks, dtype=torch.bool)
    first[1:] = ks[1:] != ks[:-1]
    ar = torch.arange(ks.numel(), device=dev)
    start = torch.cummax(torch.where(first, ar, torch.zeros_like(ar)), 0).values
    rank = torch.empty_like(ar)
    rank[order] = ar - start
    kk = torch.arange(K, device=dev)
    free_order = torch.sort(torch.where(inv < 0, kk, kk + K), dim=-1).values
    ok = rank < nfree[wg, r]
    rslot = free_order[wg, r, rank.clamp(max=K - 1)]
    wg, ng, r, rslot = wg[ok], ng[ok], r[ok], rslot[ok]
    i = _take(pool, inv, wg, ng, slot[wg, ng])
    _give(pool, inv, wg, r, rslot, i)
    pool.pos[wg, i] = pos[wg, r]
    to = torch.full((W, N), -1, dtype=torch.long, device=dev)
    to[wg, ng] = r
    return {"to": to, "item": torch.where(to >= 0, item, torch.full_like(item, -1))}


# ============================================================================== decay
def decay(pool, dt_days=1.0, *, inv=None, item_cell=None, n_cells: int | None = None) -> dict:
    """Rot meat, fat, hide, bone, wood and fibre for dt_days (in place): species s loses
    1 - exp(-k_s f(T) dt) of its mass, k_s = DECAY_PER_DAY at 288 K, f = Q10^((min(T, DECAY_OPT_K) -
    288) / 10) between DECAY_MIN_K and DECAY_MAX_K and 0 outside (frozen or too hot for microbes),
    T the item's own temperature. An item left below MIN_ITEM_KG rots away whole. Every item ages.

    Returns decayed_kg [W, S] float64, decayed_el [W, E], litter_c_kg [W, C] float64 (the decayed
    carbon per cell when item_cell [W, I] and n_cells are given; else None) and removed [W]."""
    W, I, _ = pool.shape
    dev = pool.device
    k = _vec(DECAY_PER_DAY, dev, torch.float64)
    temp = pool.temp_k.double()
    f = torch.where((temp > float(DECAY_MIN_K)) & (temp < float(DECAY_MAX_K)),
                    float(Q10) ** ((temp.clamp(max=float(DECAY_OPT_K)) - float(DECAY_REF_K)) / 10.0),
                    torch.zeros_like(temp)) * pool.alive
    kg = pool.comp.double().clamp_min(0) * (pool.mass.double() * pool.alive)[..., None]
    lost = kg * (1 - torch.exp(-k * f[..., None] * float(dt_days)))
    kg = kg - lost
    mass = kg.sum(-1)
    gone = pool.alive & (mass < float(MIN_ITEM_KG))
    lost = lost + torch.where(gone[..., None], kg, torch.zeros_like(kg))
    keep = pool.alive & ~gone & (lost.sum(-1) > 0)
    w, i = keep.nonzero(as_tuple=True)
    pool.comp[w, i] = (kg[w, i] / mass[w, i, None]).float()
    pool.mass[w, i] = mass[w, i].float()
    pool.head_mass[w, i] = torch.minimum(pool.head_mass[w, i], pool.mass[w, i])
    wg, ig = gone.nonzero(as_tuple=True)
    _remove(pool, inv, wg, ig)
    pool.age_d += float(dt_days) * pool.alive
    per_item_c = lost @ mat.element_matrix(dev)[:, mat.ELEMENTS.index("C")]
    litter = None
    if item_cell is not None and n_cells is not None:
        litter = torch.zeros(W * n_cells, dtype=torch.float64, device=dev)
        cells = item_cell.long().clamp(0, n_cells - 1)
        litter.index_add_(0, (torch.arange(W, device=dev)[:, None] * n_cells + cells).reshape(-1),
                          per_item_c.reshape(-1))
        litter = litter.reshape(W, n_cells)
    decayed = lost.sum(1)
    return {"decayed_kg": decayed, "decayed_el": mat.element_mass(decayed), "litter_c_kg": litter,
            "removed": gone.sum(1)}


# ============================================================================== provenance
def provenance() -> dict[str, tuple[str, str]]:
    """Dotted key -> (tag, source) for every number of this module that sets up a world."""
    out: dict[str, tuple[str, str]] = {}
    for name, val in globals().items():
        if isinstance(val, Val):
            out[f"crafting.{name}"] = (val.tag, val.source)
    for table_name in ("FUEL_FACTOR", "DECAY_PER_DAY", "BINDS"):
        for s, val in globals()[table_name].items():
            out[f"crafting.{table_name}.{s}"] = (val.tag, val.source)
    for s, val in PACKING.items():
        out[f"crafting.PACKING.{s}"] = (val.tag, val.source)
    out["crafting.WOOD_CHAR_RULE"] = WOOD_CHAR_RULE
    out["crafting.TISSUE_CHAR_RULE"] = TISSUE_CHAR_RULE
    out["crafting.FRICTION_RULE"] = FRICTION_RULE
    out["crafting.PICKUP_RULE"] = PICKUP_RULE
    out["crafting.WOOD_C_FRACTION"] = (DERIVED, "carbon mass fraction of the wood species (materials.MAKEUP)")
    return out
