"""Bodies of life9 v3: physics, not life-history laws (PLANET-V3-SPEC section 3).

State lives in :class:`Bodies`, tensors with a leading ``[A, N]`` (arenas, slots); a slot is used while ``alive``.
Positions are metres on the arena's periodic square of side ``L``; ``heading`` is radians from +x (east) toward +y
(north). ``mass_kg`` is lean tissue at normal hydration, ``frame_kg`` the largest lean the body reached (scaled down
when it divides), ``reserve_j`` the fat store (tripalmitin), ``water_kg`` all body water, ``n_kg`` free amino nitrogen
awaiting use or excretion, ``salt_kg`` salt awaiting excretion, ``body_k`` the body temperature, ``damage`` in [0, 1)
while alive and ``wear`` its oxidative part (which repair cannot undo), ``gut`` [A, N, G, Q] the gut contents per
digestive class (:data:`GUT_CLASSES`) and quantity (:data:`GUT_Q`: kg dry, J gross, kg C, kg N), ``held`` [A, N, 2] the
item slots in the two grips (-1 empty), ``age_bouts``, lineage ids (``uid``, ``parent``, ``founder``, ``generation``),
the genome (brain3's weight genes, ``k``, its section-3 body genes and this module's :data:`EXTRA_SPECS`) and,
optionally, the live recurrent weights and hidden state. ``transit`` holds the float64 per-cell residuals of the
exchanges with the patch (below).

What is supplied here is physics and chemistry with measured constants, the body's physical abilities (move and turn,
open the mouth on what it touches, vibrate the air, divide), replication with heritable variation (``brain3.mutate3``)
and death when physics says so. Nothing names a purpose: the motor outputs are intensities and every one has a
physical cost. There is no age at which division is allowed, no fixed span of life, no scaling law of metabolic
rate with mass, no automatic reproduction, and no supplied homeostat: how much fat a body stores before it grows, how
much water it holds, how strongly its kidney concentrates salt and how much air it holds (lungs, air sacs, a swim
bladder: buoyancy and the O2 of a breath-hold) are heritable genes with physical costs.

One bout (:func:`bout`; dt = a day / bouts), in order:

* :func:`move`: heading += pi x turn; the muscles deliver |thrust| x the sustained power, SUSTAINED_SHARE x
  a(pO2) x muscle x muscle mass x (1 - damage) x the Arrhenius factor of T_b (muscle power is chemistry), a = pO2 /
  (pO2 + O2_HALF_PA), at most MUSCLE_EFF x what the O2 supply leaves after the resting costs
  (:func:`sustained_power_w`); the speed solves P = c_m M g v + the limbs' internal kinetic work + 1/2 rho C_d A_x v^3 on land
  and P = 1/2 rho_w C_d A_x v^3 in water (the path's submerged share from path samples); climbing M g dh is paid from
  the same power (the travel time shrinks); the metabolic cost is work / MUSCLE_EFF; drag work and potential energy
  leave the body, the rest is heat in it. ``dt_s`` lets a world sub-step movement within a bout (:func:`sum_loco`).
* :func:`ingest`: the mouth (at the front of the ellipsoid, reaching one body radius) closes on what it touches, a
  body first, then an item, then the ground: a bite cuts tissue with force F = muscle stress x jaw-muscle
  cross-section x lever against the hide's cutting toughness, one gape-sized chunk per jaw cycle (cycles run at the
  Arrhenius rate of T_b) while contact lasts (reach / relative speed), a prey that fits the gape is swallowed whole;
  items are bitten the same way against their species' toughness (or swallowed whole); on the ground the mouth crops
  plant soft tissue from the area it sweeps in the bout (2 x reach x path + pi reach^2, the disc once a bout however
  many sub-steps the world takes, at most the fine cell) into the gut's free space, and fresh water (pond, soil above
  the bucket) or sea water is drunk at the gut's emptying rate.
* :func:`metabolism`: the gut passes contents with a mean retention time (at the Arrhenius rate of T_b) and absorbs
  enz / (enz + K) of each class; absorbed energy becomes fat (carbon-limited, the surplus carbon leaves as CO2); the fat
  above the heritable store level ``fat_store`` x lean, after the bout's committed costs, builds tissue toward
  ``size_kg`` (nitrogen-limited); nothing is burnt to keep the store below a cap. The heat balance of the ellipsoid
  (spherical fur shell, forced and free convection, linearised radiation with sky and ground, evaporation through skin,
  fur and boundary layer in series, respiratory water) is integrated exactly per sub-step with maintenance per tissue
  under the Arrhenius factor, repair, thermogenesis gain x lean x (setpoint - T_b) capped by the aerobic capacity of
  the muscles at T_b and by the O2 supply, the heat of locomotion and muscle work, the heat of growth (conversion and synthesis
  overhead), of a child's development and of digestion's conversions; the respiratory water of every joule follows
  the sub-step's T_b. The fuel
  comes from the fat, then from lean tissue; tripalmitin's stoichiometry books O2, CO2 and metabolic water; free
  nitrogen leaves as urea; salt leaves in urine at most at the heritable kidney's concentration and in at most the
  urine its filtration makes (unflushed salt stays and harms); water above the normal plus the heritable bladder
  store leaves as urine. Oxidative wear grows with the O2
  used x (1 - the share the repair budget prevents) and is permanent; the repair budget, the share ``repair`` on top
  of the bout's running costs, first heals wounds, osmotic harm and anoxia at the cost of re-synthesising the harmed
  protein.
* :func:`deaths`: no fuel (lean below 1 - LEAN_LOSS_LETHAL of the frame), dehydration (water below 0.6 x normal),
  freezing (271 K), denaturation (318 K) or damage 1; the body becomes meat, bone, hide and fat items at its position
  (``items.spawn``); what finds no slot goes to the litter, the rest of its water to the soil. ``selection`` False (the
  no-selection control): physics still removes exactly its dead, but first the genome and lineage each dying body
  carries are swapped with those of a uniformly random living body, so which genomes are lost is random.
* :class:`Shadow` (:func:`shadow_of`, :func:`shadow_step`): the neutral shadow, a gene-only population that takes the
  run's own deaths and births per bout at random, the drift baseline of PLANET-V3-SPEC 10.
* :func:`sense_view` and :func:`manipulation_view` hand senses3 and manipulate these bodies' mass, mouth, reach and
  power under their field names; manipulate's mechanical work comes back as ``work_j``, and its item ledger can take
  the item changes (``take_items``, ``spawn_items``).
* Division is a physical process with a rate, not an event: the divide output is the share of the body's protein
  synthesis capacity (:func:`develop_power_w`: PROTEIN_FSR_MAX_DAY of its own protein a day at the Arrhenius rate of
  T_b) it spends making its next child anew (DEVELOP_J_PER_KG per kg of child, oxidised and released as heat in
  :func:`metabolism`, which adds what was paid to ``brood_j``). :func:`divide` splits off the children whose
  development is complete: a body whose child and remainder are both at least MIN_BODY_KG gives ``offspring_share``
  of its lean mass, frame, store, water and free nitrogen to a free slot of its arena; the child lies touching the
  parent's rear; its genome is ``brain3.mutate3`` of the parent's (with :data:`GENE_SPECS`). A birth without a free
  slot (``capacity_full``) or too small (``too_small``) fails and its development is lost. Because development is a
  power, the number of children per day does not depend on the bouts per day.
* Air and water (``PROVENANCE['breathing']``): a body takes up O2 through its exchange organs, its lung (the
  tissue that holds its air, ``air_l_kg`` x lean x AIR_TISSUE_KG_PER_L, taking its share of the tissue budget and
  costing C_LUNG_W_KG) at LUNG_O2_MOL_S_KG per kg at the reference inspired pO2 and in proportion to its own, and
  its skin (SKIN_O2_MOL_S_M2_PA x area x pO2): :func:`o2_supply_mol_s`. The supply caps what it can oxidise: the
  sustained muscle power (:func:`sustained_power_w`) and thermogenesis take only what is left after its resting
  costs, and a body whose costs exceed its supply runs an O2 debt (hypoxia). A body denser than the water at its
  position (:func:`water_density`: sea water RHO_SEA, ponds fresh) in water deeper than its height has its airway
  under water (:func:`sunk`) unless it treads water, which costs :func:`tread_power_w` and needs its sustained
  power; a floating body holds its density's share under water (:func:`submerged_share`). With the airway under
  water a body lives on its O2 store (:func:`o2_store_mol`: the air it holds from the alveolar pO2 down to
  O2_HOLD_FLOOR_SHARE of it, and its heritable O2 carrier, ``o2_carrier``, loaded and unloaded between the same two
  pressures by the aerobic law): ``o2_debt_mol`` grows with its own O2 use and is repaid at its next breath; the
  store spent, the body is unconscious (world3 lets it act no more) and the debt beyond the store is anoxia, damage
  1 after ANOXIA_TOLERANCE_S of its resting O2 use at T_REF (``anoxia`` keeps its share of the damage, and a death
  it dominates is booked as ``anoxia``); there is no fixed time to drowning. The air carries no mass to the books
  (the O2 it gives is booked with the oxidation that uses it).

Ledgers (float64 [A] in ``Bodies.ledger``): energy (J), carbon (kg C), water (kg), nitrogen (kg N), inert matter (gut
contents and the lean tissue's bone mineral, kg), salt (kg), and the gases of oxidation (O2, CO2, H2O moles);
:func:`ledger_errors` closes each one against :func:`stocks`. The exchanges with the patch (plant tissue, plant water,
drinking, excreta, carcass remains) move float64 amounts into per-cell residuals in ``Bodies.transit``, which are
settled into the patch's float32 fields as far as float32 resolves them (and booked in the patch's own counters), so
neither side loses or creates matter by rounding; :func:`exchange_errors` closes the bodies' books against the patch's.
Every constant has a ``PROVENANCE`` entry.

Every tensor has the leading [A, N] axes; there are no Python loops over bodies or cells (the heat balance loops over
a fixed number of sub-steps). Randomness comes only from the generator passed in (division draws, mutation, headings,
the no-selection control), so the CPU is exactly reproducible. State is float32, ledgers float64.
"""
from __future__ import annotations

import ctypes
import hashlib
import math
from dataclasses import dataclass

import torch

from haishool.life9.planet import climate as cl
from haishool.life9.planet import constants as K
from haishool.life9.planet import crafting as cf
from haishool.life9.planet import items as itm
from haishool.life9.planet import materials as mt
from haishool.truth.formula import parse as _parse

from . import brain3 as b3
from . import optics
from . import patch as pt

R_, D_, N_, C_ = "reference", "derived", "new_rule", "chain"
RN, DN = "reference+new_rule", "derived+new_rule"

# =========================================================================================== chemistry
DAY_S = K.DAY_S
BOUTS = 4
M_C = K.element_mass("C")
M_N = K.element_mass("N")
M_H2O = K.molar_mass("H2O")
LIPID = mt.LIPID                                               # tripalmitin C51H98O6, the fat store
M_FAT = K.molar_mass(LIPID)
FAT_J_KG = float(mt.GROSS_J_KG[LIPID])                         # 39.5 MJ/kg
FAT_C_KG = mt.formula_elements(LIPID)["C"]
FAT_C_PER_J = FAT_C_KG / FAT_J_KG
_FAT_ATOMS = _parse(LIPID)
O2_MOL_PER_KG_FAT = (_FAT_ATOMS["C"] + _FAT_ATOMS["H"] / 4 - _FAT_ATOMS["O"] / 2) / M_FAT
CO2_MOL_PER_KG_FAT = _FAT_ATOMS["C"] / M_FAT
H2O_MOL_PER_KG_FAT = _FAT_ATOMS["H"] / 2 / M_FAT
O_MOL_PER_KG_FAT = _FAT_ATOMS["O"] / M_FAT
OXY_J_PER_MOL_O2 = FAT_J_KG / O2_MOL_PER_KG_FAT               # about 440 kJ per mol O2
PROTEIN_J_KG = float(mt.GROSS_J_KG["protein"])                 # 23.6 MJ/kg
PROTEIN_C_KG = mt.PROTEIN["C"]
PROTEIN_N_KG = mt.PROTEIN["N"]
UREA = "CH4N2O"
UREA_J_MOL = 632.7e3                                           # J/mol, heat of combustion of solid urea
UREA_J_PER_KG_N = UREA_J_MOL / (2 * M_N)                       # 22.6 MJ per kg N excreted as urea
UREA_C_PER_KG_N = M_C / (2 * M_N)                              # 0.43 kg C per kg N


def _hhv(formula: str) -> float:
    """Higher heating value (J/kg) from the ultimate analysis, Channiwala & Parikh 2002 (as materials uses)."""
    e = {k: 100 * v for k, v in mt.formula_elements(formula).items()}
    return 1e6 * (0.3491 * e.get("C", 0) + 1.1783 * e.get("H", 0) + 0.1005 * e.get("S", 0)
                  - 0.1034 * e.get("O", 0) - 0.0151 * e.get("N", 0))


# =========================================================================================== body plan
BODY_PLAN = {"meat": 0.84, "bone": 0.10, "hide": 0.06}         # mass shares of lean tissue (carcass species)


def _makeup(species: str, comp: str) -> float:
    return dict(mt.MAKEUP[species]).get(comp, 0.0)


LEAN_WATER = sum(x * _makeup(s, "H2O") for s, x in BODY_PLAN.items())          # 0.668 kg water per kg lean
LEAN_PROTEIN = sum(x * _makeup(s, "protein") for s, x in BODY_PLAN.items())    # 0.222
LEAN_LIPID = sum(x * _makeup(s, LIPID) for s, x in BODY_PLAN.items())          # 0.050
LEAN_MINERAL = 1.0 - LEAN_WATER - LEAN_PROTEIN - LEAN_LIPID                     # 0.060 (bone apatite)
E_LEAN = LEAN_PROTEIN * PROTEIN_J_KG + LEAN_LIPID * FAT_J_KG                     # 7.2 MJ/kg = sum share x FOOD_J_KG
C_LEAN = LEAN_PROTEIN * PROTEIN_C_KG + LEAN_LIPID * FAT_C_KG                    # 0.156 kg C/kg
N_LEAN = LEAN_PROTEIN * PROTEIN_N_KG                                             # 0.036 kg N/kg
FOUNDER_FAT_PER_LEAN = 0.15                                     # kg fat per kg lean a founder starts with
FAT_SCALE_PER_LEAN = 0.3                                        # scale of the interoceptive fat channel (no limit)
RHO_TISSUE = 1050.0                                             # kg/m^3
CP_TISSUE = 3500.0                                              # J/(kg K)
ASPECT = 2.0                                                    # prolate spheroid, length / width
_ECC = math.sqrt(1.0 - 1.0 / ASPECT ** 2)
#: area of the prolate spheroid over the area of the sphere of the same volume
SHAPE_FACTOR = (2 * math.pi * (1 + ASPECT / _ECC * math.asin(_ECC))) / (4 * math.pi * ASPECT ** (2.0 / 3.0))

# =========================================================================================== tissue upkeep
E_ARRHENIUS_EV = 0.65
K_B_EV = K.K_B / 1.602176634e-19                                # eV/K
T_REF_K = 310.15
_KCAL_DAY = 4184.0 / DAY_S                                      # W/kg per kcal/(kg day)
C_MUSCLE_W_KG = 13.0 * _KCAL_DAY                                # 0.63 W/kg resting skeletal muscle at T_REF
C_GUT_W_KG = 200.0 * _KCAL_DAY                                  # 9.7 W/kg splanchnic tissue (liver's rate)
C_NERVE_W_KG = 240.0 * _KCAL_DAY                                # 11.6 W/kg brain; eyes and ears as nervous tissue
C_REST_W_KG = 12.0 * _KCAL_DAY                                  # 0.58 W/kg residual tissue
C_ADIPOSE_W_KG = 4.5 * _KCAL_DAY                                # 0.22 W/kg adipose tissue
C_KIDNEY_W_KG = 440.0 * _KCAL_DAY                               # 21.3 W/kg kidney
C_LUNG_W_KG = 28.0 * _KCAL_DAY                                  # 1.36 W/kg wet lung tissue (holds the body's air)
MUSCLE_POWER_REF_W_KG = 100.0                                   # muscle whose resting rate is C_MUSCLE_W_KG
BRAIN_KG_PER_UNIT = 1e-6                                        # brain tissue per active hidden unit
RHO_PELAGE = 1300.0 * 0.05                                      # kg/m^3 of coat: keratin x hair volume share
SYNTH_OVERHEAD = 1.0 / 3.0                                      # heat of synthesis per J of tissue built
PROTEIN_SYNTH_J_KG = 3.6e6                                      # fuel to synthesise 1 kg of protein from amino acids
DEVELOP_J_PER_KG = PROTEIN_SYNTH_J_KG * LEAN_PROTEIN            # 0.8 MJ: a child's protein made anew, per kg child
PROTEIN_FSR_MAX_DAY = 0.25                                      # most of its own protein a body makes per day (T_REF)
DEVELOP_PASSES = 8                                              # numerics: births one body may complete in one call
RHO_FAT = 900.0                                                 # kg/m^3, the fat store
AIR_TISSUE_KG_PER_L = 11.3 / 53.5                               # kg of (wet) lung tissue per litre of air it holds
ML_PER_MOL_STP = 22414.0                                        # mL of an ideal gas per mol at 273.15 K, 101.325 kPa
O2_HOLD_FLOOR_SHARE = 4.0 / 13.3                                # the store is used down to this share of the lung pO2
HUFNER_ML_PER_G = 1.34                                          # mL O2 bound per g of carrier protein (haemoglobin)
CARRIER_TURNOVER_S = 120.0 * DAY_S                              # a carrier protein's life (red cells)
#: carrier upkeep: residual tissue's rate plus re-making the carrier protein over its life (0.93 W/kg)
C_CARRIER_W_KG = C_REST_W_KG + PROTEIN_SYNTH_J_KG / CARRIER_TURNOVER_S
ANOXIA_TOLERANCE_S = 180.0                                      # anoxia beyond the store: damage 1 (resting O2 use)
VO2MAX_1KG_ML_S = 1.92                                          # the most O2 a 1 kg mammal takes up (mL/s)
LUNG_KG_PER_KG = 11.3e-3                                        # a mammal's lungs per kg of body
LUNG_O2_MOL_S_KG = VO2MAX_1KG_ML_S / ML_PER_MOL_STP / LUNG_KG_PER_KG    # 7.6e-3 mol/s per kg lung at P_O2_REF_PA
SKIN_O2_MOL_S_M2_PA = 2e-5 / 0.003 * 1e4 / 101325.0 / ML_PER_MOL_STP / 60.0   # 4.9e-10: Krogh's K over 30 um

# =========================================================================================== heat exchange
K_FUR = 0.04                                                    # W/(m K), pelage with its air
EMISSIVITY_BODY = float(optics.EMISSIVITY["tissue:fur"])
ALBEDO_SW_BODY = float(optics.SW["tissue:fur"])
EMISSIVITY_GROUND = float(optics.EMISSIVITY["ground:soil"])
SW_VIEW = 0.5                                                   # share of the body's area receiving the ground flux
BRUTSAERT = (1.24, 1.0 / 7.0)                                   # clear-sky emissivity 1.24 (e_hPa / T)^(1/7)
M_AIR = 0.028965                                                # kg/mol, dry air
CP_AIR = 1007.0                                                 # J/(kg K)
MU_AIR_300 = 1.846e-5                                           # Pa s at 300 K
MU_AIR_EXP = 0.7
K_AIR_300 = 0.0263                                              # W/(m K) at 300 K
K_AIR_EXP = 0.8
PR_AIR = 0.707
DV_AIR_273 = 2.178e-5                                           # m^2/s, water vapour in air at 273.15 K, 1 atm
DV_EXP = 1.81
K_WATER = 0.598                                                 # W/(m K) at 293 K
NU_WATER = 1.004e-6                                             # m^2/s at 293 K
PR_WATER = 7.0
RHO_WATER = K.RHO_WATER
RANZ_MARSHALL = (2.0, 0.6, 0.5, 1.0 / 3.0)                      # Nu = 2 + 0.6 Re^0.5 Pr^(1/3)
CHURCHILL_SPHERE = (0.589, 0.469)                               # free convection Nu = 2 + 0.589 Ra^1/4 / [...]
NU_MIX_EXP = 3.0                                                # forced and free convection combined (cube law)
ALPHA_WATER = NU_WATER / PR_WATER                               # m^2/s, thermal diffusivity of water
WATER_DRIFT = 0.03                                              # surface water drift per unit of wind speed
THIESEN_WATER = (288.9414, 508929.2, 68.12963, 3.9863)          # fresh water density law (celsius)
LV_273 = K.L_VAPORIZATION                                       # J/kg at 273.15 K
LV_SLOPE = 2370.0                                               # J/(kg K): L(T) = L(273.15) - 2370 (T - 273.15)
RESP_EXTRACTION = 0.25                                          # share of inspired O2 taken up
#: the inspired pO2 of the reference (Earth's 21.2 kPa at 101,325 Pa, saturated with vapour at T_REF): 19.9 kPa
P_O2_REF_PA = 21200.0 * (1.0 - float(cl.e_sat_pa(torch.tensor(310.15))) / 101325.0)
T_FREEZE_K, T_DENATURE_K = b3.SETPOINT_BOUNDS_K                 # 271 K and 318 K: death thresholds
SUBSTEPS = 8                                                    # numerics: heat-balance sub-steps per bout

# =========================================================================================== locomotion
C_M = 0.1                                                       # mechanical work per metre = c_m M g
FEDAK_INTERNAL = (0.478, 1.53)                                  # limbs' internal kinetic power 0.478 v^1.53 W/kg
MUSCLE_EFF = 0.25
SUSTAINED_SHARE = 0.3                                           # sustained (aerobic) over peak muscle power
O2_HALF_PA = 3000.0                                             # Michaelis constant of the aerobic capacity
PADDLE_EFF = 0.25                                               # drag-based paddling: thrust (or lift) per limb work
C_DRAG = 0.47                                                   # sphere drag coefficient, both media
TURN_SPAN = math.pi                                             # a bout's turn spans any direction
PATH_SAMPLES = 16                                               # numerics: samples along a bout's path

# =========================================================================================== gut
GUT_CLASSES = ("plant", "protein", "lipid", "cooked", "inert")
G_PLANT, G_PROTEIN, G_LIPID, G_COOKED, G_INERT = range(len(GUT_CLASSES))
GUT_Q = ("kg", "j", "c", "n")
Q_KG, Q_J, Q_C, Q_N = range(len(GUT_Q))
GUT_WET_PER_TISSUE = 3.0                                        # kg wet digesta per kg gut tissue
DIGESTA_DRY = 0.15                                              # dry share of digesta
RETENTION_S = 12 * 3600.0                                       # mean retention time of digesta
K_ENZ_PLANT = 0.05                                              # kg gut tissue per kg body at half efficiency
K_ENZ_MEAT = 0.005
COOK_GAIN = float(mt.COOK_GAIN["meat"])                         # 1.3: cooked flesh digests better
COOKED_K = float(cf.COOK_K)                                     # an item's flesh is cooked once it reached 70 C
PLANT_J_KG = 18.5e6                                             # gross energy of plant dry matter
PLANT_WATER_PER_DRY = 3.0                                       # kg water per kg dry soft plant tissue
WATER_EMPTY_S = 900.0                                           # gastric emptying time of drunk water
FAECES_WATER_PER_DRY = 2.0                                      # kg water per kg dry faeces
SEA_SALINITY = 0.035                                            # kg salt per kg sea water
SEA_DENSITY_PER_SALT = 0.77                                     # kg/m^3 of sea water per g/kg of salt (EOS-80)
RHO_SEA = RHO_WATER + SEA_DENSITY_PER_SALT * SEA_SALINITY * 1e3   # 1,027 kg/m^3
KIDNEY_REF_SHARE = 0.31 / 73.0                                  # human kidneys per kg body (ICRP 89)
URINE_SALT_REF = 0.035                                          # kg salt per kg urine water at that kidney share
URINE_SALT_PER_KIDNEY = URINE_SALT_REF / KIDNEY_REF_SHARE       # concentrating capacity per kidney share
NACL_SOLUBILITY = 0.36                                          # kg NaCl per kg water at 25 C: no urine holds more
GFR_KG_PER_KG_S = 180.0 / 0.31 / DAY_S                          # filtrate per kg kidney (human 180 L/day, 0.31 kg)
URINE_FLOW_SHARE = 0.15                                         # the most urine a kidney makes, share of its filtrate
OSMOTIC_LETHAL = 0.01                                           # kg unexcreted salt per kg body water = damage 1
OSMOTIC_TIME_S = 6 * 3600.0                                     # ... held for this long

# =========================================================================================== bites
MUSCLE_STRESS_PA = 3.0e5                                        # peak isometric stress of muscle
JAW_SHARE = 0.03                                                # jaw-closing share of the muscle mass
JAW_LEVER = 0.4                                                 # mechanical advantage at the teeth
GAPE_SHARE = 0.15                                               # gape over body length
BITE_CYCLE_S = 1.0                                              # one jaw cycle
V_CONTACT_FLOOR = 1e-6                                          # m/s, numerics: contact time = gape / speed
ENAMEL_MOHS = 5.0                                               # teeth cannot cut what is harder than enamel
E_MINERAL_PA = 5.0e10                                           # Young's modulus of rocks, minerals, ceramics
E_METAL_PA = 1.1e11                                             # Young's modulus of the metals
WOUND_LETHAL_SHARE = 0.3                                        # a wound removing this share of tissue: damage 1
CONTACT_K = 8                                                   # numerics: nearest bodies checked for contact
ITEM_SCAN = 16                                                  # numerics: items scanned per fine cell
BRUTE_PAIRS = 1 << 21                                           # numerics: few contact queries are checked directly
#: cutting toughness of soft organic species (J/m^2): energy per area of new crack a tooth must supply
ORGANIC_TOUGHNESS_J_M2 = {"meat": 1500.0, "fat": 300.0, "hide": 15000.0, "bone": 1700.0, "plant_fiber": 2000.0,
                          "wood": 10000.0, "resin": 100.0, "charcoal": 50.0}

# =========================================================================================== damage, death
ROS_PER_O2 = 0.002                                              # mol reactive oxygen per mol O2 consumed
PROTEIN_KG_PER_MOL_HIT = 50.0                                   # a hit inactivates one 50 kDa protein
LIFETIME_J_PER_KG = 0.92e9                                      # mammals' lifetime energy per kg (Rubner)
OXIDISED_LETHAL_SHARE = 0.3                                     # share of the protein oxidised at damage 1
#: kg of protein irreversibly oxidised per mol O2 with no repair: OXIDISED_LETHAL_SHARE of the protein over the
#: O2 of a lifetime's energy
WEAR_PROTEIN_KG_PER_MOL_O2 = OXIDISED_LETHAL_SHARE * LEAN_PROTEIN / (LIFETIME_J_PER_KG / OXY_J_PER_MOL_O2)
ROS_ESCAPE = WEAR_PROTEIN_KG_PER_MOL_O2 / (ROS_PER_O2 * PROTEIN_KG_PER_MOL_HIT)   # share escaping the antioxidants
HEAL_J_PER_KG = OXIDISED_LETHAL_SHARE * LEAN_PROTEIN * PROTEIN_SYNTH_J_KG        # per unit damage and kg lean
LEAN_LOSS_LETHAL = 0.4                                          # no fuel: lean below 0.6 x the frame
DEHYDRATION_LETHAL = 0.6                                        # water below 0.6 x normal
MIN_BODY_KG = b3.SIZE_BOUNDS_KG[0]                              # 1 mg: the smallest body the physics covers
MIN_KG = 1e-9                                                   # numerics: geometry floor of empty slots
CARCASS_MIN_KG = 1e-6                                           # numerics: carcass parts below 1 mg go to litter
CAUSES = ("none", "no_fuel", "dehydration", "freezing", "denaturation", "damage", "anoxia")
CAUSE = {c: i for i, c in enumerate(CAUSES)}

# =========================================================================================== calls
VOICE_W_PER_KG = 1.0                                            # acoustic watts per kg of sound organ at full effort
VOCAL_EFF = 0.01                                                # acoustic over metabolic power of a call

# =========================================================================================== genes this module adds
FAT_STORE_BOUNDS = (1e-3, 3.0)                                  # kg fat per kg lean
FAT_STORE_FOUNDER = (0.03, 0.6)
BLADDER_BOUNDS = (1e-6, 1.0)                                    # kg water per kg lean held above the normal
BLADDER_FOUNDER = (1e-3, 0.1)
KIDNEY_BOUNDS = (b3.TISSUE_MIN, 0.1)                            # kg kidney per kg lean
KIDNEY_FOUNDER = (1e-3, 1e-2)
AIR_BOUNDS_L_KG = (0.0, 1.0)                                    # L of air per kg lean: none to the lean's own volume
AIR_FOUNDER_L_KG = AIR_BOUNDS_L_KG                             # founders: over the gene's physical bounds
AIR_UNIT_L_KG = (1.0 / RHO_WATER - 1.0 / RHO_TISSUE) * 1e3      # 0.048 L/kg: the air that floats lean tissue
CARRIER_BOUNDS_ML_KG = (0.0, LEAN_PROTEIN * HUFNER_ML_PER_G * 1e3)   # none to all lean protein as carrier: 298
CARRIER_FOUNDER_ML_KG = CARRIER_BOUNDS_ML_KG                    # founders: over the gene's physical bounds
CARRIER_UNIT_ML_KG = 15.0                                       # the lin step's unit: a land mammal's carrier
#: genes added after the founders' draw order was fixed: drawn after the headings, in name order (see found)
FOUNDER_LATE_GENES = ("air_l_kg", "o2_carrier")
#: heritable body genes of this module's physiology (brain3.GeneSpec3: mutated by brain3.mutate3 like every gene)
EXTRA_SPECS = {
    "fat_store": b3.GS3(FAT_STORE_BOUNDS[0], FAT_STORE_BOUNDS[1], "log", b3.LOG_UNIT, "loguniform",
                        *FAT_STORE_FOUNDER, "kg fat per kg lean",
                        "the store level above which fat (beyond the bout's costs) builds lean tissue toward "
                        "size_kg; below it, or at size, fat accumulates: no cap and no automatic burning"),
    "bladder": b3.GS3(BLADDER_BOUNDS[0], BLADDER_BOUNDS[1], "log", b3.LOG_UNIT, "loguniform", *BLADDER_FOUNDER,
                      "kg water per kg lean", "water held above the normal before the kidney excretes it (bladder "
                      "storage); its cost is the mass carried"),
    "kidney": b3.GS3(KIDNEY_BOUNDS[0], KIDNEY_BOUNDS[1], "log", b3.LOG_UNIT, "loguniform", *KIDNEY_FOUNDER,
                     "kg per kg body", "kidney tissue: it costs C_KIDNEY_W_KG and takes its share of the tissue "
                     "budget; the most concentrated urine scales with it (URINE_SALT_PER_KIDNEY)"),
    "air_l_kg": b3.GS3(AIR_BOUNDS_L_KG[0], AIR_BOUNDS_L_KG[1], "lin", AIR_UNIT_L_KG, "uniform", *AIR_FOUNDER_L_KG,
                       "L of air per kg lean",
                       "air the body holds (lungs, air sacs, swim bladder) at the ambient pressure: it lowers the "
                       "body's density (density: floating or lying on the bottom) and holds O2 for a breath-hold "
                       "(o2_store_mol, breath_hold_s); its lung tissue (AIR_TISSUE_KG_PER_L) takes its share of the "
                       "tissue budget and costs C_LUNG_W_KG; it is the body's exchange organ for O2 "
                       "(o2_supply_mol_s); 0 is a body without an air store or a lung"),
    "o2_carrier": b3.GS3(CARRIER_BOUNDS_ML_KG[0], CARRIER_BOUNDS_ML_KG[1], "lin", CARRIER_UNIT_ML_KG, "uniform",
                         *CARRIER_FOUNDER_ML_KG, "mL O2 (STP) per kg lean",
                         "the O2 the body's carrier proteins (haemoglobin, myoglobin) bind when saturated: an O2 "
                         "store for a breath-hold (o2_store_mol), loaded and unloaded by the aerobic law between "
                         "the lung pO2 and its floor; its protein (HUFNER_ML_PER_G) takes its share of the tissue "
                         "budget and costs C_CARRIER_W_KG; 0 is a body without a carrier"),
}
b3.check_specs(EXTRA_SPECS)
#: every float gene a v3 body genome carries: brain3's and this module's
GENE_SPECS = {**b3.GENE_SPECS3, **EXTRA_SPECS}

LEDGER_KEYS = ("e_founded", "e_food", "e_oxidised", "e_conversion", "e_faeces", "e_urine", "e_dead",
               "c_founded", "c_food", "c_co2", "c_faeces", "c_urine", "c_dead",
               "w_founded", "w_drunk", "w_food", "w_metabolic", "w_evap", "w_resp", "w_faeces", "w_urine", "w_dead",
               "n_founded", "n_food", "n_urine", "n_faeces", "n_dead",
               "i_founded", "i_food", "i_built", "i_faeces", "i_urine", "i_dead",
               "salt_in", "salt_out", "salt_dead", "salt_land", "salt_sea", "sea_in_kg", "sea_out_kg",
               "o2_mol", "co2_mol", "co2_ox_mol", "h2o_ox_mol", "fat_ox_kg",
               "births", "capacity_full", "divide_fired", "too_small", "control_swaps",
               *(f"deaths_{c}" for c in CAUSES[1:]))

PROVENANCE = {
    # chemistry
    "DAY_S": (R_, "constants.DAY_S, 86,400 s"),
    "BOUTS": (N_, "PLANET-V3-SPEC 1: 4 bouts a day (6 h); dt is passed with the environment"),
    "M_C": (R_, "molar mass of C, the repo's IUPAC table (constants.element_mass)"),
    "M_N": (R_, "molar mass of N, the repo's IUPAC table"),
    "M_H2O": (R_, "molar mass of H2O, the repo's IUPAC table"),
    "LIPID": (R_, "materials.LIPID, tripalmitin C51H98O6: the fat store is animal fat"),
    "M_FAT": (D_, "molar mass of tripalmitin from the IUPAC table, 0.807 kg/mol"),
    "FAT_J_KG": (R_, "gross energy of fat 39.5 MJ/kg (Blaxter 1989, Energy Metabolism in Animals and Man; "
                     "materials.GROSS_J_KG)"),
    "FAT_C_KG": (D_, "carbon mass fraction of tripalmitin, 0.759"),
    "FAT_C_PER_J": (D_, "FAT_C_KG / FAT_J_KG: carbon held per joule of the store"),
    "O2_MOL_PER_KG_FAT": (D_, "C51H98O6 + 72.5 O2 -> 51 CO2 + 49 H2O: 72.5 mol O2 per mol fat (chemistry)"),
    "CO2_MOL_PER_KG_FAT": (D_, "51 mol CO2 per mol tripalmitin (RQ 0.70)"),
    "H2O_MOL_PER_KG_FAT": (D_, "49 mol metabolic water per mol tripalmitin (1.09 kg per kg fat)"),
    "O_MOL_PER_KG_FAT": (D_, "6 O atoms per mol tripalmitin, for the oxygen balance of oxidation"),
    "OXY_J_PER_MOL_O2": (D_, "FAT_J_KG / O2_MOL_PER_KG_FAT = 440 kJ per mol O2 (the oxycaloric equivalent of fat; "
                             "450 kJ/mol for mixed fuel, Schmidt-Nielsen 1997)"),
    "PROTEIN_J_KG": (R_, "gross energy of protein 23.6 MJ/kg (Blaxter 1989; materials.GROSS_J_KG)"),
    "PROTEIN_C_KG": (R_, "protein C 53 % by mass (materials.PROTEIN; Jones 1931)"),
    "PROTEIN_N_KG": (R_, "protein N 16 % by mass (materials.PROTEIN; Kjeldahl factor 6.25)"),
    "UREA": (R_, "mammals excrete amino nitrogen as urea CO(NH2)2"),
    "UREA_J_MOL": (R_, "standard enthalpy of combustion of solid urea -632.7 kJ/mol (NIST Chemistry WebBook)"),
    "UREA_J_PER_KG_N": (D_, "UREA_J_MOL / (2 M_N): energy leaving with 1 kg of urea nitrogen, 22.6 MJ"),
    "UREA_C_PER_KG_N": (D_, "M_C / (2 M_N): carbon leaving with 1 kg of urea nitrogen, 0.43 kg"),
    "hhv": (R_, "wood's gross energy by the Channiwala & Parikh 2002 correlation (Fuel 81, 1051), as materials"),
    # body plan
    "BODY_PLAN": (RN, "lean tissue is meat 84 %, bone 10 %, hide 6 % by mass (version 2's creatures.BODY_MIX: "
                      "skeleton about 10 %, ICRP 23 Reference Man bone 5 kg and marrow 4.5 kg of 70 kg; skin and "
                      "coat 6 %); every functional tissue (muscle, gut, organs, brain, coat) has this chemistry, so "
                      "the carcass items carry exactly the body's energy, carbon, nitrogen and water"),
    "LEAN_WATER": (D_, "sum of BODY_PLAN x makeup water (materials.MAKEUP): 0.668, the body's normal water per kg"),
    "LEAN_PROTEIN": (D_, "sum of BODY_PLAN x makeup protein: 0.222 kg/kg"),
    "LEAN_LIPID": (D_, "sum of BODY_PLAN x makeup lipid: 0.050 kg/kg"),
    "LEAN_MINERAL": (D_, "the rest of the makeup, bone apatite 0.060 kg/kg (no energy, carbon or nitrogen)"),
    "E_LEAN": (D_, "7.2 MJ per kg lean = sum of BODY_PLAN x materials.FOOD_J_KG (the spec's tissue cost of 7 MJ/kg)"),
    "C_LEAN": (D_, "0.156 kg C per kg lean from the makeup"),
    "N_LEAN": (D_, "0.036 kg N per kg lean from the makeup"),
    "FOUNDER_FAT_PER_LEAN": (N_, "a founder starts with 0.15 kg fat per kg lean (half of version 2's store, "
                                 "RESERVE_MAX_FRACTION 0.3), the same for every founder so the founding endowment "
                                 "favours no gene (PLANET-V3-SPEC 6: half a reserve)"),
    "FAT_SCALE_PER_LEAN": (N_, "the interoceptive fat channel reads reserve / (0.3 kg/kg x lean x 39.5 MJ/kg) (fat "
                               "reaches 20-50 % of body mass before hibernation or migration, Pond 1998, The Fats of "
                               "Life); a physical scale for senses3 only: the store has no cap and may exceed it"),
    "RHO_TISSUE": (R_, "PLANET-V3-SPEC 3: 1,050 kg/m^3 (soft tissue 1,030-1,070, ICRP 89)"),
    "CP_TISSUE": (R_, "PLANET-V3-SPEC 3: 3,500 J/(kg K), the mean specific heat of the body (3.47 kJ/(kg K), "
                      "Gephart & Thomas; tissue values 3.4-3.7 in the IT'IS database)"),
    "ASPECT": (N_, "the body is a prolate spheroid of its volume with length twice its width"),
    "SHAPE_FACTOR": (D_, "the spheroid's area over the area of the equal-volume sphere (1.08): heat and vapour "
                         "conductances of the sphere model are scaled by it"),
    # upkeep
    "E_ARRHENIUS_EV": (R_, "PLANET-V3-SPEC 3: E = 0.65 eV, the mean activation energy of metabolism (Gillooly, "
                           "Brown, West, Savage & Charnov 2001, Science 293, 2248); chemistry, not size"),
    "K_B_EV": (D_, "Boltzmann constant in eV/K: CODATA K_B / e (exact)"),
    "T_REF_K": (R_, "37 C, the temperature of the tissue rates (Elia 1992, human tissue measurements)"),
    "C_MUSCLE_W_KG": (R_, "resting skeletal muscle 13 kcal/(kg day) (Elia 1992, in Kinney & Tucker eds., Energy "
                          "Metabolism: Tissue Determinants and Cellular Corollaries); also the sound organ"),
    "C_GUT_W_KG": (RN, "liver 200 kcal/(kg day) (Elia 1992), taken for gut tissue: splanchnic tissues use 20-25 % "
                       "of the body's O2 from about 5 % of its mass"),
    "C_NERVE_W_KG": (RN, "brain 240 kcal/(kg day) (Elia 1992); eyes and ears are costed as nervous tissue (the "
                         "retina is among the most O2-hungry tissues)"),
    "C_REST_W_KG": (R_, "residual tissues 12 kcal/(kg day) (Elia 1992)"),
    "C_ADIPOSE_W_KG": (R_, "adipose tissue 4.5 kcal/(kg day) (Elia 1992), paid on the fat store"),
    "C_KIDNEY_W_KG": (R_, "kidney 440 kcal/(kg day) (Elia 1992), paid on the kidney gene's tissue"),
    "C_LUNG_W_KG": (RN, "lung tissue 28 kcal/(kg day), 1.36 W/kg of WET lung (with its blood), paid on the tissue "
                        "that holds the body's air (air_l_kg x AIR_TISSUE_KG_PER_L, itself a wet lung weight): the "
                        "human lung's own O2 use at rest is about 1-3 % of the body's (Loer, Scheeren & Tarnow 1997, "
                        "Anesthesiology 86, 532; from memory: verify), about 5 mL O2/min = 1.64 W, from lungs of "
                        "1.2 kg with their blood (ICRP 89): 1.37 W/kg, about 0.27 W per litre of a 6 L lung. The "
                        "earlier 1.9 W/kg was per blood-free tissue and was charged on the wet mass (review of 4 Oct "
                        "2026); Elia 1992 lumps the lungs into his residual tissues (12 kcal/(kg day))"),
    "C_CARRIER_W_KG": (D_, "the O2 carrier's protein is residual tissue (C_REST_W_KG) re-made over its life: "
                           "PROTEIN_SYNTH_J_KG / CARRIER_TURNOVER_S on top, 0.93 W/kg in all"),
    "CARRIER_TURNOVER_S": (R_, "a red cell and its haemoglobin live about 120 days (human; quoted from memory: "
                               "verify)"),
    "HUFNER_ML_PER_G": (R_, "Huefner's number: 1.34 mL O2 bound per g of haemoglobin when saturated (myoglobin "
                            "about the same per g; quoted from memory: verify)"),
    "ML_PER_MOL_STP": (R_, "22,414 mL per mol of an ideal gas at 273.15 K and 101.325 kPa (CODATA molar volume)"),
    "PROTEIN_SYNTH_J_KG": (R_, "protein synthesis costs about 4 ATP per peptide bond: 9.1 mol bonds per kg x 4 x "
                               "about 0.1 MJ of fuel per mol ATP = 3.6 MJ per kg protein (Waterlow 1995, Annu. Rev. "
                               "Nutr. 15, 57; Reeds, Fuller & Nicholson 1985); repair re-synthesises harmed protein "
                               "at this cost"),
    "MUSCLE_POWER_REF_W_KG": (RN, "muscle at 100 W/kg peak costs C_MUSCLE_W_KG at rest (human muscle peaks near "
                                  "100-200 W/kg); resting cost scales with the power gene (mitochondrial volume "
                                  "follows aerobic capacity, Weibel & Hoppeler 2005, J. Exp. Biol. 208, 1635), so "
                                  "power costs maintenance"),
    "BRAIN_KG_PER_UNIT": (N_, "one active hidden unit stands for the neurons of 1 mg of brain (10^5-10^6 neurons "
                              "at the densities of Herculano-Houzel 2011, Front. Neuroanat. 5, 46): 32 units weigh "
                              "32 mg; maintenance is paid per unit through its tissue"),
    "RHO_PELAGE": (RN, "a coat is keratin (1,300 kg/m^3) at about 5 % hair volume (pelage hair volume shares of a few "
                       "per cent, Cena & Monteith 1975, Proc. R. Soc. B 188, 395): 65 kg per m^3 of coat; its mass "
                       "takes its share of the tissue budget (brain3.tissue_shares)"),
    "SYNTH_OVERHEAD": (R_, "synthesis heat 1/3 of the energy deposited: net efficiency of tissue deposition about "
                           "0.75 (Blaxter 1989); paid by oxidation when lean tissue grows; its heat enters the bout's "
                           "heat balance (a child's development is paid separately: DEVELOP_J_PER_KG)"),
    "DEVELOP_J_PER_KG": (D_, "PROTEIN_SYNTH_J_KG x LEAN_PROTEIN = 0.8 MJ per kg of child: a child is a new organised "
                             "body, so its protein is made anew from the parent's amino acids before it separates "
                             "(the mass is the parent's; the synthesis is the cost of building a body from it); "
                             "oxidised and released as heat in the parent over the time the development takes"),
    "PROTEIN_FSR_MAX_DAY": (RN, "the fastest a body makes protein: whole-body fractional protein synthesis of young "
                                "growing small mammals is about 20-30 % of body protein a day (Waterlow, Garlick & "
                                "Millward 1978, Protein Turnover in Mammalian Tissues and in the Whole Body; tissue "
                                "rates up to about 100 %/day in liver and gut mucosa, Garlick, McNurlan & Preedy "
                                "1980; values from memory: verify); 0.25/day of the parent's own protein at T_REF, "
                                "with the Arrhenius factor of T_b (a rate per kg of tissue, the same at every size, "
                                "as the tissue maintenance rates). The divide output is the share of this rate the "
                                "body spends on its next child, so a child of offspring_share s takes at least s / "
                                "0.25 days at 310 K; the rate is per second, so it does not depend on the decision "
                                "interval"),
    "DEVELOP_PASSES": (N_, "numerics: divide completes at most this many births per body per call (development that "
                           "finished faster than that waits for the next call)"),
    "RHO_FAT": (R_, "density of animal fat about 900 kg/m^3 (adipose tissue 0.90-0.92 g/cm^3; tripalmitin 0.85-0.88 "
                    "liquid, CRC Handbook): the body's density for buoyancy is its mass over lean / RHO_TISSUE + fat / "
                    "RHO_FAT + (gut contents and water beyond the normal) / RHO_WATER + the air it holds "
                    "(air_volume_m3) (body.density); the geometry keeps RHO_TISSUE (shape)"),
    "AIR_TISSUE_KG_PER_L": (RN, "kg of (wet) tissue that holds a litre of the body's air: mammal lungs weigh 11.3 g "
                                "per kg of body and hold 53.5 mL per kg at total lung capacity (Stahl 1967, J. Appl. "
                                "Physiol. 22, 453; from memory: verify), 0.21 kg per L, the 1 kg value of his "
                                "size laws (about 0.16 kg/L at 70 kg and 0.27 at 20 g by his exponents: the same "
                                "ratio is taken at every size, a simplification); taken for every air store "
                                "(new_rule: the thin walls of bird air sacs and fish swim bladders weigh less per "
                                "litre, so large stores are charged as lungs). The air volume is the total lung "
                                "capacity at all times: a body cannot breathe out to sink or in to float (lungs "
                                "swing about 4x between residual volume and total capacity; a stated "
                                "simplification, review of 4 Oct 2026)"),
    "O2_HOLD_FLOOR_SHARE": (RN, "the store is used from the lung's alveolar pO2 (the inspired pO2 x (1 - "
                                "RESP_EXTRACTION), the exhaled level) down to 4 / 13.3 = 0.30 of it: breath-hold "
                                "divers lose consciousness at an alveolar pO2 of about 25-30 mmHg, 3.3-4 kPa "
                                "(Lindholm & Lundgren 2009, J. Appl. Physiol. 106, 284), against a normal alveolar "
                                "pO2 of about 100 mmHg, 13.3 kPa (from memory: verify). Relative, so that one "
                                "O2-transport rule holds for breathing and breath-holding on every planet: a body "
                                "breathing 2.2 kPa air uses its store from its own operating pO2 (a fixed 4 kPa "
                                "floor gave no lung store at all on world7 781 while the same bodies breathed). A "
                                "BODY-PLAN CONSTANT STILL TO BE DECIDED BY THE OWNER (it could be heritable)"),
    "ANOXIA_TOLERANCE_S": (RN, "anoxia beyond the O2 store (the body unconscious): damage 1 when the O2 debt beyond "
                               "the store reaches 180 s of the body's resting O2 use at T_REF (maintenance at "
                               "T_REF / OXY_J_PER_MOL_O2), so a cold body, using less, lasts longer. Drowning: "
                               "consciousness is lost within about 2 min, the brain irreversibly injured after 4-6 "
                               "min (Szpilman et al. 2012, N. Engl. J. Med. 366, 2102; quoted from memory: verify), "
                               "so 2-4 min of anoxia after the store: 180 s. A BODY-PLAN CONSTANT STILL TO BE DECIDED "
                               "BY THE OWNER (anoxia-tolerant ectotherms last far longer)"),
    "VO2MAX_1KG_ML_S": (R_, "maximal O2 uptake of mammals 1.92 M^0.809 mL O2/s (Taylor, Maloiy, Weibel, Langman, "
                            "Kamau, Seeherman & Heglund 1981, Respir. Physiol. 44, 25; quoted from memory: verify): "
                            "1.92 mL/s at 1 kg"),
    "LUNG_KG_PER_KG": (R_, "mammal lungs 11.3 g per kg of body (Stahl 1967; quoted from memory: verify), the same "
                           "wet lung as AIR_TISSUE_KG_PER_L"),
    "LUNG_O2_MOL_S_KG": (RN, "the O2 a lung takes up at most per kg of its (wet) tissue: a 1 kg mammal's VO2max over "
                             "its lungs, 1.92 mL/s / 11.3 g = 170 mL/(s kg), 7.6e-3 mol/(s kg), at the reference "
                             "inspired pO2 (P_O2_REF_PA); in proportion to the inspired pO2 (ventilation and "
                             "diffusion both carry O2 in proportion to its partial pressure, Fick; new_rule: no "
                             "carrier affinity shifts it); a mammal-like lung (0.054 L/kg) supplies about 38 W/kg, "
                             "about its muscles' sustained metabolic power (symmorphosis). At 70 kg the same law "
                             "overstates VO2max about 2x (lungs scale as M^0.99, VO2max as M^0.81)"),
    "P_O2_REF_PA": (D_, "the inspired pO2 of the reference: Earth's 21.2 kPa at 101,325 Pa, saturated with water "
                        "vapour at T_REF (21.2 x (1 - e_sat(310.15 K) / 101,325) = 19.9 kPa)"),
    "SKIN_O2_MOL_S_M2_PA": (RN, "O2 through the skin, 4.9e-10 mol/(s m^2 Pa): Krogh's diffusion constant of O2 in "
                                "tissue, about 2e-5 mL O2 cm / (cm^2 min atm) (Krogh 1919, J. Physiol. 52, 391), "
                                "over a 30 um barrier to the capillaries (amphibian skin capillaries lie a few tens "
                                "of um deep; Feder & Burggren 1985, Biol. Rev. 60, 1) (quoted from memory: verify); "
                                "on the skin area at the ambient pO2, in air only; about 4.6 W/m^2 at Earth's pO2: "
                                "enough for a resting 10 g body, not for a 1 kg one (lungless salamanders are small "
                                "ectotherms)"),
    "DIVE_KEYS": (N_, "the dive structure of a stretch of time, in order: the airway under water from its start "
                      "until the first breath, the longest stretch between two breaths, the stretch since the last "
                      "breath at its end, and whether it was under water all along (dive_piece, dive_merge)"),
    "breathing": (DN, "o2_supply_mol_s = LUNG_O2_MOL_S_KG x lung tissue x p_insp / P_O2_REF_PA + "
                      "SKIN_O2_MOL_S_M2_PA x skin area x pO2, p_insp = pO2 x (1 - e_sat(T_b) / p_air). The supply "
                      "caps the body's aerobic metabolism (review of 4 Oct 2026: before, gas exchange did not "
                      "depend on the lung, so a body without one breathed at full capacity): the sustained muscle "
                      "power is the least of SUSTAINED_SHARE x a(pO2) x peak and MUSCLE_EFF x (supply - resting "
                      "costs), thermogenesis takes at most supply / (1 + repair) - maintenance - the bout's work, "
                      "and what the costs need beyond the supply in air is an O2 debt as under water (hypoxia: "
                      "anoxia damage once it passes the store). Under water the supply is 0"),
    # heat exchange
    "K_FUR": (R_, "pelage conductivity with its air 0.035-0.07 W/(m K) (Cena & Monteith 1975; Scholander, "
                  "Walters, Hock & Irving 1950, Biol. Bull. 99, 225); spherical shell (1/r1 - 1/r2) / (4 pi k) "
                  "with convection at r2 (brain3's fur_m contract); wet fur conducts as water"),
    "EMISSIVITY_BODY": (R_, "optics.EMISSIVITY['tissue:fur'] (pelage 0.95-0.99, Hammel 1956)"),
    "ALBEDO_SW_BODY": (R_, "optics.SW['tissue:fur'] (coats 0.1-0.4, Walsberg 1983)"),
    "EMISSIVITY_GROUND": (R_, "optics.EMISSIVITY['ground:soil']; the ground radiates at the air temperature "
                              "(new_rule)"),
    "SW_VIEW": (N_, "the shortwave per horizontal m^2 at the body's cell falls on half its area (a sphere's "
                    "projected area is a quarter of its area per unit of normal beam; beam per horizontal m^2 over a "
                    "day is about twice the normal flux per m^2 projected, and diffuse light comes from the sky "
                    "half)"),
    "BRUTSAERT": (R_, "clear-sky emissivity 1.24 (e / T)^(1/7), e in hPa (Brutsaert 1975, Water Resour. Res. 11, "
                      "742); the body sees half sky, half ground"),
    "M_AIR": (R_, "molar mass of dry air 28.965 g/mol (U.S. Standard Atmosphere 1976)"),
    "CP_AIR": (R_, "1,007 J/(kg K) at 300 K (Incropera et al. 2007, Table A.4)"),
    "MU_AIR_300": (R_, "dynamic viscosity of air 184.6e-7 Pa s at 300 K (Incropera Table A.4)"),
    "MU_AIR_EXP": (D_, "power-law fit mu ~ T^0.7 of Incropera Table A.4 over 250-350 K"),
    "K_AIR_300": (R_, "conductivity of air 26.3 mW/(m K) at 300 K (Incropera Table A.4)"),
    "K_AIR_EXP": (D_, "power-law fit k ~ T^0.8 of Incropera Table A.4 over 250-350 K"),
    "PR_AIR": (R_, "Prandtl number of air 0.707 (Incropera Table A.4)"),
    "DV_AIR_273": (R_, "diffusivity of water vapour in air 21.78e-6 m^2/s at 273.15 K and 101.325 kPa, x (T / "
                       "273.15)^1.81 x (p0 / p) (Massman 1998, Atmos. Environ. 32, 1111)"),
    "DV_EXP": (R_, "Massman 1998 temperature exponent 1.81"),
    "K_WATER": (R_, "conductivity of water 0.598 W/(m K) at 293 K (Incropera Table A.6)"),
    "NU_WATER": (R_, "kinematic viscosity of water 1.004e-6 m^2/s at 293 K (Incropera Table A.6)"),
    "PR_WATER": (R_, "Prandtl number of water 7.0 at 293 K (Incropera Table A.6)"),
    "RHO_WATER": (R_, "constants.RHO_WATER 1,000 kg/m^3"),
    "RANZ_MARSHALL": (R_, "forced convection from a sphere Nu = 2 + 0.6 Re^0.5 Pr^(1/3) (Ranz & Marshall 1952, "
                          "Chem. Eng. Prog. 48, 141), at the outer (fur) diameter; vapour by the Chilton-Colburn "
                          "analogy g_v = g_h (D_v / alpha)^(2/3)"),
    "CHURCHILL_SPHERE": (R_, "free convection from a sphere Nu = 2 + 0.589 Ra^(1/4) / [1 + (0.469 / Pr)^(9/16)]^(4/9) "
                             "for Ra up to 1e11 (Churchill 1983, in Heat Exchanger Design Handbook 2.5.7; quoted "
                             "from memory: verify), Ra = g beta |T_b - T_fluid| d^3 / (nu alpha) at the outer "
                             "diameter (the core-to-fluid difference: an upper bound of the surface's); beta = 1 / T "
                             "in air, the fresh-water law (THIESEN_WATER) in water"),
    "NU_MIX_EXP": (R_, "forced and free convection combined Nu = 2 + ((Nu_F - 2)^3 + (Nu_N - 2)^3)^(1/3) "
                       "(Churchill 1983's combination for mixed convection; quoted from memory: verify). Before, a "
                       "floating or still body in water had Nu = 2 (pure conduction), below its conductance in air "
                       "(review of 4 Oct 2026)"),
    "ALPHA_WATER": (D_, "thermal diffusivity of water nu / Pr = 1.43e-7 m^2/s at 293 K (Incropera Table A.6)"),
    "WATER_DRIFT": (R_, "the water a body is in moves at about 3 % of the wind speed (wind drift of the sea surface, "
                        "Wu 1983, J. Phys. Oceanogr. 13, 1441; quoted from memory: verify), added to the body's speed "
                        "for forced convection in water as the wind is in air"),
    "THIESEN_WATER": (R_, "fresh water's density rho(t) = 1000 (1 - (t + 288.9414) (t - 3.9863)^2 / (508929.2 (t + "
                          "68.12963))), t in C (Tilton & Taylor 1937 after Thiesen; quoted from memory: verify), "
                          "whose expansion coefficient beta = -d ln rho / dt drives free convection in water (2.1e-4 "
                          "/K at 20 C, 0 at 4 C; |beta| is used)"),
    "LV_273": (R_, "latent heat of vaporisation 2.501 MJ/kg at 273.15 K (constants.L_VAPORIZATION)"),
    "LV_SLOPE": (R_, "L(T) = 2.501e6 - 2370 (T - 273.15) J/kg (Rogers & Yau 1989, A Short Course in Cloud "
                     "Physics, table 2.1)"),
    "RESP_EXTRACTION": (R_, "mammals take up 20-25 % of the O2 they inspire (Schmidt-Nielsen 1997, Animal "
                            "Physiology, ch. 2); exhaled air is saturated at T_b, so respiratory water per mol O2 = "
                            "(e_sat(T_b) - e_air) / (pO2 x extraction), and its latent heat leaves the body"),
    "T_FREEZE_K": (C_, "brain3.SETPOINT_BOUNDS_K[0] = 271 K: body fluids freeze (PLANET-V3-SPEC 3)"),
    "T_DENATURE_K": (C_, "brain3.SETPOINT_BOUNDS_K[1] = 318 K: protein denaturation (PLANET-V3-SPEC 3)"),
    "SUBSTEPS": (N_, "numerics: the bout's heat balance in 8 exactly integrated linear sub-steps (45 min)"),
    "heat_balance": (D_, "C dT/dt = P(T) - G (T - T_e) - L_v E(T): G = 1 / (R_fur + 1 / G_out), G_out = (1 - s) "
                         "(h_c + h_r) A2 + s h_w A2 (s the submerged share), T_e the operative temperature T_air + "
                         "(Q_abs - eps sigma A2 T_air^4) / ((h_c + h_r) A2) (Campbell & Norman 1998, An "
                         "Introduction to Environmental Biophysics, ch. 12), water at the air temperature (new_rule); "
                         "P holds maintenance, thermogenesis, repair, the heat of locomotion, muscle work and calls "
                         "(less the work that leaves the body), the heat of growth (conversion and synthesis "
                         "overhead) and of digestion's conversions, less the latent heat of the respiratory water of "
                         "every joule oxidised; maintenance, thermogenesis, its cap and evaporation are linearised "
                         "about each sub-step's start, so the sub-step is solved exactly (exponential relaxation), "
                         "and the respiratory water booked is the sub-steps' own"),
    # locomotion
    "C_M": (R_, "PLANET-V3-SPEC 3: c_m = 0.1, mechanical work per metre 0.1 M g, about 1 J/(kg m) at every size "
                "(Heglund, Fedak, Taylor & Cavagna 1982, J. Exp. Biol. 97, 57)"),
    "FEDAK_INTERNAL": (R_, "the limbs' internal kinetic work, 0.478 v^1.53 W per kg of body on land, independent of "
                           "size (Fedak, Heglund & Taylor 1982, J. Exp. Biol. 97, 23): the work of swinging the limbs, "
                           "added to the centre-of-mass work c_m M g v; not in water"),
    "MUSCLE_EFF": (R_, "PLANET-V3-SPEC 3: muscle efficiency 0.25 (Margaria 1976, Biomechanics and Energetics of "
                       "Muscular Exercise; Smith, Barclay & Loiselle 2005, Prog. Biophys. Mol. Biol. 88, 1)"),
    "SUSTAINED_SHARE": (RN, "aerobic (sustained) mechanical power is about 30 % of peak muscle power (human "
                            "peak sprint power about 3 x the power at VO2max; Bundle, Hoyt & Weyand 2003, J. Appl. "
                            "Physiol. 95, 1955); the same capacity caps thermogenesis (shivering)"),
    "O2_HALF_PA": (RN, "aerobic capacity a = pO2 / (pO2 + 3 kPa), the spec's Michaelis term (version 2's "
                       "AEROBIC_HALF_PA, PLANET-SPEC 2.8); the same law loads and unloads the O2 carrier "
                       "(o2_store_mol)"),
    "PADDLE_EFF": (RN, "drag-based paddling converts at most about a third of the limbs' work into thrust (propulsive "
                       "efficiency 0.33 for the muskrat, Fish 1984, J. Comp. Physiol. B 154, 467; paddlers 0.2-0.33 "
                       "against 0.8 for lift-based swimmers, Fish 1996, Am. Zool. 36, 628; from memory: verify); "
                       "0.25, also the figure of merit of treading water (tread_power_w)"),
    "tread": (DN, "a body denser than the water holds its airway up while it moves (its locomotion share) when its "
                  "moving power covers tread_power_w: the vertical force of its weight in water F = (rho - rho_w) / "
                  "rho x m g at the actuator-disc power F^1.5 / sqrt(2 rho_w A_x) (Rankine-Froude momentum theory, "
                  "the ideal), over PADDLE_EFF (the paddling limb's figure of merit) and MUSCLE_EFF; the rest of "
                  "its power propels it. Standing (no thrust) it does not tread, and in deep water it sinks. Before, "
                  "sinking was a switch at the water's density whatever the body's power (review of 4 Oct 2026)"),
    "C_DRAG": (R_, "sphere drag coefficient 0.47 at Re 1e3-1e5 (White 2011, Fluid Mechanics, 7th ed., fig. 7.16) "
                   "on the frontal area pi b^2; swimming costs 1/2 rho_w C_d A_x v^2 per metre (spec)"),
    "TURN_SPAN": (N_, "turn in [-1, 1] x pi per bout: a bout may start in any direction (version 2's MAX_TURN)"),
    "PATH_SAMPLES": (N_, "numerics: the path's submerged share and its climb profile from 16 samples"),
    "locomotion": (D_, "speed solves P = c_m M g v + 0.478 M v^1.53 + 1/2 rho_air C_d A_x v^3 (land) or 1/2 rho_w "
                       "C_d A_x v^3 (water) by Newton from above; the path in water and on land is crossed at each "
                       "medium's speed (distance = t / ((1 - s) / v_land + s / v_water)); the net climb M g "
                       "max(0, dz) is paid from the same power: the body stops where P t + M g max(0, z - z0) along "
                       "the path's samples reaches P dt (linear between samples); metabolic cost = work / MUSCLE_EFF; "
                       "the drag work (air and water) and the climb's potential energy leave the body, the rest is "
                       "heat in it. Muscle power, and so speed, carries the Arrhenius factor of T_b"),
    # gut
    "GUT_CLASSES": (N_, "digestive classes by chemistry: plant (carbohydrate polymers: plant dry matter, "
                        "cellulose, wood), protein, lipid, cooked (protein and lipid of cooked flesh), inert "
                        "(minerals, metals, charcoal, rosin: no animal digests them; they cost gut volume)"),
    "GUT_Q": (N_, "per class: kg dry, J gross, kg C, kg N, so mixed sources keep their own chemistry"),
    "GUT_WET_PER_TISSUE": (RN, "a gut holds about 3 kg of wet digesta per kg of gut tissue (ruminant reticulorumen "
                               "contents 10-15 % of body mass with gut tissue 4-5 %; Van Soest 1994, Nutritional "
                               "Ecology of the Ruminant)"),
    "DIGESTA_DRY": (R_, "digesta are 10-20 % dry matter (rumen 10-15 %, Van Soest 1994)"),
    "RETENTION_S": (RN, "mean retention time of digesta 12 h (mammals 5-70 h, Clauss et al. 2007, Comp. Biochem. "
                        "Physiol. A 148, 249): each bout passes 1 - exp(-dt / 12 h) of every class"),
    "K_ENZ_PLANT": (RN, "plant matter is digested at efficiency enz_plant / (enz_plant + 0.05): a ruminant's 5 % gut "
                        "tissue digests about half of the plant dry matter it eats (digestibility 0.5-0.6, Van Soest "
                        "1994)"),
    "K_ENZ_MEAT": (RN, "protein at enz_meat / (enz_meat + 0.005): carnivores digest 90 % of meat with 3-4 % gut "
                       "tissue; lipid uses both investments (lipase is secreted by every gut, new_rule)"),
    "COOK_GAIN": (R_, "materials.COOK_GAIN['meat'] 1.3 (Carmody et al. 2011, PNAS 108, 19199): cooked flesh "
                      "digests at min(1, 1.3 x enz / (enz + K))"),
    "COOKED_K": (R_, "flesh is cooked once it reached 70 C: crafting.COOK_K (actin denatures at 66-73 C, Tornberg "
                     "2005), the temperature manipulate.py and version 2 use"),
    "PLANT_J_KG": (R_, "PLANET-V3-SPEC 3: gross energy of plant dry matter 18.5 MJ/kg (grasses and herbage "
                       "17-19.5 MJ/kg, Blaxter 1989); with the patch's carbon fraction and leaf C:N"),
    "PLANT_WATER_PER_DRY": (R_, "green herbage is 75-80 % water (version 2's plant water per kg dry, 3 kg); taken "
                                "from the "
                                "cell's soil with the plant"),
    "WATER_EMPTY_S": (R_, "drunk water empties from the stomach with a half-time of 10-20 min (Hunt & "
                          "Spurrell 1951, J. Physiol. 113, 157): a bout can drink m x gut volume x dt / 900 s"),
    "FAECES_WATER_PER_DRY": (R_, "mammal faeces are 60-75 % water: 2 kg per kg dry (Schmidt-Nielsen 1997)"),
    "SEA_SALINITY": (R_, "mean ocean salinity 35 g/kg"),
    "SEA_DENSITY_PER_SALT": (R_, "sea water is denser than fresh by about 0.77 kg/m^3 per g/kg of salt (the haline "
                                 "contraction of the UNESCO EOS-80 equation of state, Millero & Poisson 1981, "
                                 "Deep-Sea Res. 28, 625; quoted from memory: verify)"),
    "RHO_SEA": (D_, "RHO_WATER + 0.77 x 35 = 1,027 kg/m^3 (sea water at 35 g/kg is 1,025-1,028 kg/m^3 between 20 "
                    "and 0 C): the water a body floats or sinks in at sea; ponds are fresh (RHO_WATER)"),
    "KIDNEY_REF_SHARE": (R_, "human kidneys 310 g in a 73 kg body (ICRP 89, Reference Man)"),
    "URINE_SALT_REF": (R_, "maximal urine concentration of the human kidney about 1,200 mOsm/kg (Schmidt-Nielsen "
                           "1997, Animal Physiology, ch. 9): 0.6 mol NaCl = 35 g per kg urine water, so a human "
                           "kidney gains no water from sea water"),
    "URINE_SALT_PER_KIDNEY": (DN, "the most concentrated urine = URINE_SALT_REF x kidney share / KIDNEY_REF_SHARE "
                                  "(8.2 kg salt per kg water per unit share), at most NACL_SOLUBILITY: concentrating "
                                  "power grows with medullary tissue (relative medullary thickness, Schmidt-Nielsen & "
                                  "O'Dell 1961, Am. J. Physiol. 200, 1119; Beuchat 1990, Am. J. Physiol. 258, R298); "
                                  "desert rodents reach 5,000-9,000 mOsm/kg (Schmidt-Nielsen 1997) with kidneys of "
                                  "about 1 % of body mass; linear in the share (new_rule), and the kidney's tissue "
                                  "costs C_KIDNEY_W_KG"),
    "NACL_SOLUBILITY": (R_, "solubility of NaCl 360 g per kg water at 25 C (CRC Handbook)"),
    "GFR_KG_PER_KG_S": (R_, "glomerular filtration 125 mL/min, about 180 L/day, by human kidneys of 0.31 kg "
                            "(Guyton & Hall, Textbook of Medical Physiology, ch. 26; ICRP 89): 6.7 g of filtrate per "
                            "kg of kidney per second"),
    "URINE_FLOW_SHARE": (R_, "maximal urine flow, in water diuresis, 15-20 mL/min, about 15 % of the filtrate "
                             "(Guyton & Hall, ch. 28): the urine that carries salt in a bout is at most this share "
                             "of the kidney's filtrate, so a small kidney cannot spend the body's water on salt it "
                             "cannot concentrate (the salt stays and harms)"),
    "OSMOTIC_LETHAL": (RN, "salt that cannot be excreted (no water left to carry it) stays and raises plasma "
                           "osmolality; 1 % of body water as extra NaCl is about +170 mOsm, lethal hypernatraemia "
                           "within hours; damage += unexcreted salt / (0.01 x body water) x dt / OSMOTIC_TIME_S, "
                           "repairable like a wound"),
    "OSMOTIC_TIME_S": (RN, "severe hypernatraemia kills within about 6 h to a day (Adrogue & Madias 2000, N. Engl. "
                           "J. Med. 342, 1493): the retention that gives damage 1 per 6 h"),
    # bites
    "MUSCLE_STRESS_PA": (R_, "peak isometric stress of vertebrate muscle 200-300 kPa (Medler 2002, Am. J. "
                             "Physiol. 283, R368)"),
    "JAW_SHARE": (RN, "jaw-closing muscles are 1-5 % of the musculature (masseter and temporalis); 3 %; their "
                      "cross-section is (volume)^(2/3) (geometry)"),
    "JAW_LEVER": (R_, "mechanical advantage of mammal jaws at the teeth 0.3-0.5 (Greaves 1983, J. Zool. 199, 23)"),
    "GAPE_SHARE": (RN, "gape about 15 % of body length (mammal gapes 10-20 % of head-body length); a prey whose "
                       "volume fits a sphere of the gape's diameter is swallowed whole"),
    "BITE_CYCLE_S": (RN, "one jaw cycle a second (mammal chewing 1-7 Hz, Ross et al. 2007, J. Exp. Biol. 210, "
                         "3384); bites in a bout = mouth x contact time / cycle"),
    "V_CONTACT_FLOOR": (N_, "numerics: contact lasts the mouth's reach / relative speed, at most the bout"),
    "mouth_reach": (N_, "the mouth sits at the front of the spheroid, pos + a (cos h, sin h); a surface within one "
                        "equal-volume radius of it is in contact (a mouth, neck or limb reaches about its body's "
                        "size: manipulate.REACH_RULE, senses3's default), so the three modules agree on contact"),
    "ENAMEL_MOHS": (R_, "tooth enamel is Mohs 5: teeth cannot cut harder species (they may be swallowed whole)"),
    "E_MINERAL_PA": (RN, "Young's moduli of rocks and minerals 10-100 GPa (Schon 2015, Physical Properties of "
                         "Rocks); cutting toughness of a brittle species = K_IC^2 / E (materials.TOUGHNESS)"),
    "E_METAL_PA": (R_, "metals 50-200 GPa (Incropera Table A.1; copper 117 GPa): ductile metals are far too tough "
                       "to bite"),
    "ORGANIC_TOUGHNESS_J_M2": (RN, "cutting toughness of soft tissues and plant matter (Lucas 2004, Dental "
                                   "Functional Morphology, ch. 4; Atkins 2009, The Science and Engineering of "
                                   "Cutting): muscle 0.3-3 kJ/m^2, adipose 0.1-0.5, skin 10-30, cortical bone "
                                   "1-2 (Currey 2002, Bones), plant fibre 1-5, wood across the grain 8-20 (Ashby "
                                   "2011), glassy rosin 0.1, friable charcoal about 0.05"),
    "bite_toughness": (DN, "per species: ORGANIC_TOUGHNESS_J_M2; granular species 0; species harder than "
                           "ENAMEL_MOHS infinite; other inorganic species K_IC^2 / E. A bite cuts a chunk "
                           "rho (pi/6) g^3 x min(1, F / (Gamma g)): the force per unit cut width a tooth needs is "
                           "the toughness (Atkins 2009); a body is cut through its hide"),
    "WOUND_LETHAL_SHARE": (RN, "losing 30-40 % of blood volume is lethal (haemorrhage class IV, ATLS); a bite that "
                               "removes the share phi of the body's tissue adds phi / 0.3 damage"),
    "CONTACT_K": (N_, "numerics: the 8 nearest bodies (patch.neighbours) are checked for contact with the mouth"),
    "ITEM_SCAN": (N_, "numerics: at most 16 ground items per fine cell are checked (stratified when there are "
                      "more); the mouth's reach is clamped to half a fine cell"),
    "BRUTE_PAIRS": (N_, "numerics: when the mouths that act (queries x slots of their arena) are at most 2^21 pairs, "
                        "as in a world's later movement sub-steps, contact is found by checking them directly (the "
                        "same rule: the nearest of the CONTACT_K nearest bodies, the nearest ground item within the "
                        "reach) instead of rebuilding the spatial hashes of every body and item"),
    # damage
    "ROS_PER_O2": (R_, "0.1-0.2 % of the O2 consumed becomes superoxide / H2O2 in vivo (St-Pierre, Buckingham, "
                       "Roebuck & Brand 2002, J. Biol. Chem. 277, 44784); most is scavenged (ROS_ESCAPE)"),
    "PROTEIN_KG_PER_MOL_HIT": (N_, "an oxidative hit that escapes the antioxidants inactivates one protein of 50 kDa "
                                   "(the median protein mass of 40-50 kDa)"),
    "LIFETIME_J_PER_KG": (R_, "non-primate mammals spend about 220 kcal (0.92 MJ) per gram of body mass over their "
                              "lives, at every size (Rubner 1908, Das Problem der Lebensdauer; reviewed by Speakman "
                              "2005, J. Exp. Biol. 208, 1717): the calibration datum of the wear rate, not a law the "
                              "bodies follow (how long they live comes from their own O2 use, repair and causes of "
                              "death)"),
    "OXIDISED_LETHAL_SHARE": (R_, "30-50 % of cellular protein is oxidatively modified in old animals (Stadtman "
                                  "1992, Science 257, 1220): damage 1 when 30 % of the body's protein is oxidised, "
                                  "the same share as WOUND_LETHAL_SHARE"),
    "WEAR_PROTEIN_KG_PER_MOL_O2": (D_, "OXIDISED_LETHAL_SHARE x LEAN_PROTEIN / (LIFETIME_J_PER_KG / OXY_J_PER_MOL_O2) "
                                       "= 3.2e-5 kg protein per mol O2 (calibration, new_rule): without repair a body "
                                       "dies of wear after spending Rubner's lifetime energy per kg lean; wear per "
                                       "bout = this x O2 x (1 - prevented share) / (OXIDISED_LETHAL_SHARE x "
                                       "LEAN_PROTEIN x lean), permanent (PLANET-V3-SPEC 3: wear ~ O2 x (1 - repair)), "
                                       "so the "
                                       "length of life varies continuously with repair and running cost, with no "
                                       "repair share at which damage stops"),
    "ROS_ESCAPE": (D_, "WEAR_PROTEIN_KG_PER_MOL_O2 / (ROS_PER_O2 x PROTEIN_KG_PER_MOL_HIT) = 3e-4: the share of the "
                       "ROS that escape the antioxidants and inactivate a protein (a check, not an input)"),
    "HEAL_J_PER_KG": (D_, "repair spends the share repair on top of the bout's running costs (maintenance, "
                          "thermogenesis, muscular work, calls); that budget first heals wounds and osmotic harm "
                          "(the damage above the permanent wear) at OXIDISED_LETHAL_SHARE x LEAN_PROTEIN x "
                          "PROTEIN_SYNTH_J_KG = 0.24 MJ per unit damage and kg lean (re-synthesising the harmed "
                          "protein), and the rest prevents its share of the bout's oxidative wear (prevention costs "
                          "in proportion to the O2 that causes the wear)"),
    "LEAN_LOSS_LETHAL": (R_, "no fuel: death once 40 % of the frame (the largest lean mass reached, scaled by "
                             "division) is lost (starvation kills after about 40-50 % loss of body mass; Keys et al. "
                             "1950, The Biology of Human Starvation)"),
    "DEHYDRATION_LETHAL": (R_, "PLANET-V3-SPEC 3: death when water < 0.6 x the body's normal water (lethal "
                               "dehydration of mammals)"),
    "MIN_BODY_KG": (C_, "brain3.SIZE_BOUNDS_KG[0] = 1 mg: a child smaller than this is not made (below it the "
                        "body physics does not apply)"),
    "MIN_KG": (N_, "numerics: a geometry floor so empty slots have finite shapes"),
    "CARCASS_MIN_KG": (N_, "numerics: carcass parts under 1 mg go to the litter"),
    "CAUSES": (N_, "death causes by physics (no_fuel, dehydration, freezing, denaturation, damage, anoxia: damage "
                   "1 of which anoxia, the O2 debt beyond the store, makes at least half); the no-selection control "
                   "keeps them and randomises only which genome dies"),
    "CAUSE": (N_, "index of each cause"),
    # calls
    "VOICE_W_PER_KG": (RN, "acoustic power of a sound organ at full effort about 1 W per kg of organ (a 30 kg "
                           "mammal's 100 dB call at 1 m radiates about 0.1 W from a larynx of about 0.1 kg)"),
    "VOCAL_EFF": (R_, "vocalising converts about 1 % of its metabolic power to sound (version 2's VOCAL_EFF; "
                      "Ryan 1988, in The Evolution of the Amphibian Auditory System)"),
    # genes this module adds
    "FAT_STORE_BOUNDS": (N_, "1e-3 to 3 kg fat per kg lean (fat reaches 20-50 % of body mass before hibernation or "
                             "migration, Pond 1998; 3 kg/kg is 75 % fat), log step mut"),
    "FAT_STORE_FOUNDER": (N_, "founders log-uniform on 0.03-0.6 kg/kg (lean to fat mammals)"),
    "BLADDER_BOUNDS": (RN, "1e-6 to 1 kg water per kg lean held above the normal (desert tortoises store water of "
                           "up to about 40 % of body mass in the bladder, Nagy & Medica 1986, Herpetologica 42, 73)"),
    "BLADDER_FOUNDER": (N_, "founders log-uniform on 1e-3-0.1 kg/kg"),
    "KIDNEY_BOUNDS": (N_, "brain3.TISSUE_MIN (no kidney) to 0.1 kg per kg body"),
    "KIDNEY_FOUNDER": (N_, "founders log-uniform on 1e-3-1e-2 kg/kg (mammal kidneys 0.4-1.5 % of body mass)"),
    "AIR_BOUNDS_L_KG": (RN, "0 (no air store) to 1 L of air per kg lean, about the lean tissue's own volume (0.95 "
                            "L/kg), a body half air; far above the largest measured stores: mammal lungs 0.05-0.06 "
                            "L/kg at total lung capacity (Stahl 1967), teleost swim bladders about 5 % of body volume "
                            "in sea water and 7 % in fresh water (Jones & Marshall 1953, Biol. Rev. 28, 16; Alexander "
                            "1966, Biol. Rev. 41, 141), birds' lungs and air sacs 0.1-0.2 L/kg, 10-20 % of body "
                            "volume (Duncker 1971, Ergeb. Anat. Entwicklungsgesch. 45; Maina 2005, The Lung-Air Sac "
                            "System of Birds) (all from memory: verify); a lin gene so that 0 itself is a value"),
    "AIR_FOUNDER_L_KG": (N_, "founders uniform over the gene's physical bounds, 0-1 L/kg (the owner's decision 3: "
                             "'random among founders over a physically broad range including zero'; the earlier "
                             "0-0.2 L/kg was set from evolved animals, review of 4 Oct 2026); a starting range, not a "
                             "target"),
    "CARRIER_BOUNDS_ML_KG": (D_, "0 (no carrier) to all the lean's protein as carrier, LEAN_PROTEIN x 1.34 mL/g = "
                                 "298 mL O2 per kg lean; measured: a land mammal's blood and muscle hold about 15 "
                                 "mL/kg, diving mammals 40-90 mL/kg (Kooyman 1989, Diverse Divers; Ponganis 2011, "
                                 "Compr. Physiol. 1, 447; quoted from memory: verify)"),
    "CARRIER_FOUNDER_ML_KG": (N_, "founders uniform over the gene's bounds (as the air gene's): a starting range, "
                                  "not a target"),
    "CARRIER_UNIT_ML_KG": (N_, "the lin step of o2_carrier is mut x 15 mL/kg, a land mammal's carrier store"),
    "AIR_UNIT_L_KG": (D_, "the lin step of air_l_kg is mut x (1 / RHO_WATER - 1 / RHO_TISSUE) = mut x 0.048 L/kg, "
                          "the air that makes lean tissue as dense as fresh water: a step's effect on buoyancy is "
                          "the same at every value; neutral law uniform on AIR_BOUNDS_L_KG"),
    "FOUNDER_LATE_GENES": (N_, "genes added after the founders' draw order was fixed (air_l_kg and o2_carrier, 4 Oct "
                               "2026) are drawn after the headings in name order, so the founders of a seed keep the "
                               "genes, positions and headings they had before; complete_genome gives an older "
                               "body state the gene's lower bound, and refuses (strict) in a protocol resume"),
    "EXTRA_SPECS": (N_, "the heritable genes this module's physiology adds (fat_store, bladder, kidney, air_l_kg, "
                        "o2_carrier) as "
                        "brain3.GeneSpec3: same bounds-reflected mutation, drawn for founders after brain3's genome; "
                        "they replace version 2's fixed store cap, the urination of all water above normal, the "
                        "human kidney's fixed concentration and v3's fixed 300 s to drowning, so none of them is a "
                        "supplied homeostat"),
    "GENE_SPECS": (N_, "brain3.GENE_SPECS3 and EXTRA_SPECS: the table divide passes to brain3.mutate3"),
    # tables and names
    "GUT_CLASS_INDEX": (N_, "G_PLANT, G_PROTEIN, G_LIPID, G_COOKED, G_INERT index GUT_CLASSES"),
    "GUT_Q_INDEX": (N_, "Q_KG, Q_J, Q_C, Q_N index GUT_Q"),
    "LEDGER_KEYS": (N_, "the float64 per-arena ledger entries (energy e_, carbon c_, water w_, nitrogen n_, inert "
                        "i_ (with the lean's bone mineral: i_built when growth makes it from the diet's ash, which "
                        "the food chemistry does not carry, i_urine when catabolism excretes it), salt and where it "
                        "went (land, sea), sea water drunk and returned, gases of oxidation, events)"),
    "STATE_FIELDS": (N_, "the per-slot state tensors of Bodies (o2_debt_mol: the O2 drawn from the store since the "
                         "last breath that met the body's need; anoxia: the share of damage anoxia made, healed in "
                         "proportion)"),
    "LINEAGE_FIELDS": (N_, "the lineage ids the no-selection control swaps with the genome, live weights and hidden "
                           "state, so a genome keeps its identity when it moves to another body"),
    "TRANSIT_KEYS": (N_, "numerics: the float64 per-cell residuals [A, n*n] of the exchanges with the patch's float32 "
                         "fields (soil water given and taken, pond water taken, plant carbon taken, litter carbon and "
                         "mineral nitrogen given); each exchange adds its exact amount and is settled into the field "
                         "as far as float32 resolves it (the patch books what the field moved), so the residual stays "
                         "within half a float32 step of the field per cell and no matter is lost or made"),
}


# =========================================================================================== state
STATE_FIELDS = ("alive", "pos", "heading", "vel", "mass_kg", "frame_kg", "reserve_j", "water_kg", "n_kg", "salt_kg",
                "body_k", "damage", "wear", "brood_j", "gut", "held", "age_bouts", "uid", "parent", "founder",
                "generation", "o2_debt_mol", "anoxia")
#: the patch field each transit residual settles into, and its direction (+1 given to the field, -1 taken)
TRANSIT_KEYS = {"soil_in": ("soil", 1), "soil_out": ("soil", -1), "pond_out": ("pond", -1),
                "plant_out": ("plant", -1), "litter_in": ("litter", 1), "nutrient_in": ("nutrient", 1)}
#: the heritable and lineage content the no-selection control swaps between a dying and a random living body
LINEAGE_FIELDS = ("uid", "parent", "founder", "generation")


@dataclass
class Bodies:
    """The bodies of A arenas in N slots each (see the module docstring); ``L`` is the arena side in metres;
    ``transit`` the float64 per-cell residuals of the exchanges with the patch (:data:`TRANSIT_KEYS`, created when
    first used)."""
    L: float
    alive: torch.Tensor
    pos: torch.Tensor
    heading: torch.Tensor
    vel: torch.Tensor
    mass_kg: torch.Tensor
    frame_kg: torch.Tensor
    reserve_j: torch.Tensor
    water_kg: torch.Tensor
    n_kg: torch.Tensor
    salt_kg: torch.Tensor
    body_k: torch.Tensor
    damage: torch.Tensor
    wear: torch.Tensor
    brood_j: torch.Tensor
    gut: torch.Tensor
    held: torch.Tensor
    age_bouts: torch.Tensor
    uid: torch.Tensor
    parent: torch.Tensor
    founder: torch.Tensor
    generation: torch.Tensor
    o2_debt_mol: torch.Tensor
    anoxia: torch.Tensor
    next_uid: torch.Tensor
    genome: dict
    wh_live: torch.Tensor | None
    hidden: torch.Tensor | None
    ledger: dict
    transit: dict | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.alive.shape)

    @property
    def device(self):
        return self.alive.device

    @property
    def speed(self) -> torch.Tensor:
        """The last bout's mean speed (m/s) [A, N]."""
        return self.vel.norm(dim=-1)

    def state_dict(self) -> dict:
        out = {name: getattr(self, name).clone() for name in STATE_FIELDS}
        out["L"] = float(self.L)
        out["next_uid"] = self.next_uid.clone()
        out["genome"] = {k: v.clone() for k, v in self.genome.items()}
        out["wh_live"] = None if self.wh_live is None else self.wh_live.clone()
        out["hidden"] = None if self.hidden is None else self.hidden.clone()
        out["ledger"] = {k: v.clone() for k, v in self.ledger.items()}
        out["transit"] = None if self.transit is None else {k: v.clone() for k, v in self.transit.items()}
        return out

    @classmethod
    def from_state(cls, d: dict, copy: bool = True, strict: bool = False) -> "Bodies":
        """Bodies from :meth:`state_dict`. ``copy`` False takes the tensors as they are (a load that owns them:
        no second copy of the state in memory); ``strict`` refuses a state without every gene and field (a protocol
        resume) instead of completing it."""
        def c(v):
            return None if v is None else (v.clone() if copy else v)
        tr = d.get("transit")
        missing = [name for name in STATE_FIELDS if name not in d]
        if strict and missing:
            raise ValueError(f"body state without {missing}: a checkpoint of other rules")
        fields = {name: c(d[name]) if name in d else torch.zeros_like(d["damage"]) for name in STATE_FIELDS}
        genome = {k: c(v) for k, v in d["genome"].items()}
        A, N = fields["alive"].shape
        complete_genome(genome, A, N, fields["alive"].device, strict=strict)   # a state from before EXTRA_SPECS
        ledger = new_ledger(A, fields["alive"].device)
        ledger.update({k: c(v) for k, v in d["ledger"].items()})
        return cls(L=float(d["L"]), **fields, next_uid=c(d["next_uid"]), genome=genome,
                   wh_live=c(d["wh_live"]), hidden=c(d["hidden"]), ledger=ledger,
                   transit=None if tr is None else {k: c(v) for k, v in tr.items()})


def state_hash(b: Bodies) -> str:
    """SHA-256 over every tensor of the state (fields, genome, brain, ledger, transit) for determinism checks."""
    h = hashlib.sha256(b"Bodies")
    tensors = [(name, getattr(b, name)) for name in STATE_FIELDS] + [("next_uid", b.next_uid)]
    tensors += [(f"genome.{k}", b.genome[k]) for k in sorted(b.genome)]
    tensors += [(f"ledger.{k}", b.ledger[k]) for k in sorted(b.ledger)]
    tensors += [(f"transit.{k}", b.transit[k]) for k in sorted(b.transit or {})]
    for name in ("wh_live", "hidden"):
        if getattr(b, name) is not None:
            tensors.append((name, getattr(b, name)))
    for name, t in tensors:
        t = t.detach().cpu().contiguous()
        h.update(f"|{name}|{t.dtype}|{tuple(t.shape)}|".encode())
        if t.numel():
            h.update(ctypes.string_at(t.data_ptr(), t.numel() * t.element_size()))
    return h.hexdigest()


def new_ledger(A: int, device="cpu") -> dict:
    d = {k: torch.zeros(A, dtype=torch.float64, device=device) for k in LEDGER_KEYS}
    d["items_eaten_kg"] = torch.zeros(A, mt.S, dtype=torch.float64, device=device)
    d["carcass_kg"] = torch.zeros(A, mt.S, dtype=torch.float64, device=device)
    return d


def complete_genome(genome: dict, A: int, N: int, device="cpu", strict: bool = False) -> dict:
    """Add any :data:`EXTRA_SPECS` gene a genome lacks, at its lower bound (the organ physically absent), in place.
    ``strict`` (a protocol resume) refuses a genome without them: filling a gene would change the bodies' physics
    silently."""
    missing = [name for name in EXTRA_SPECS if name not in genome]
    if strict and missing:
        raise ValueError(f"genome without the genes {missing}: a checkpoint of other rules")
    for name in missing:
        genome[name] = torch.full((A, N), float(EXTRA_SPECS[name].lo), dtype=torch.float32, device=device)
    return genome


def empty(A: int, N: int, L: float, *, genome: dict | None = None, wh_live=None, hidden=None, device="cpu") -> Bodies:
    """A container with no living body. ``genome`` (dict of [A, N, ...] tensors) defaults to zero genes; a genome
    without this module's :data:`EXTRA_SPECS` genes gets them at their lower bounds (:func:`complete_genome`)."""
    f = lambda *s: torch.zeros(*s, dtype=torch.float32, device=device)
    lng = lambda *s: torch.zeros(*s, dtype=torch.long, device=device)
    if genome is None:
        genome = {name: f(A, N, *((b3.N_INTERO,) if s.per_intero else ())) for name, s in GENE_SPECS.items()}
        genome["k"] = lng(A, N)
    complete_genome(genome, A, N, device)
    return Bodies(L=float(L), alive=torch.zeros(A, N, dtype=torch.bool, device=device), pos=f(A, N, 2),
                  heading=f(A, N), vel=f(A, N, 2), mass_kg=f(A, N), frame_kg=f(A, N), reserve_j=f(A, N),
                  water_kg=f(A, N), n_kg=f(A, N), salt_kg=f(A, N), body_k=f(A, N), damage=f(A, N), wear=f(A, N),
                  brood_j=f(A, N), gut=f(A, N, len(GUT_CLASSES), len(GUT_Q)), held=torch.full((A, N, 2), -1, dtype=torch.long,
                                                                               device=device),
                  age_bouts=lng(A, N), uid=torch.full((A, N), -1, dtype=torch.long, device=device),
                  parent=torch.full((A, N), -1, dtype=torch.long, device=device),
                  founder=torch.full((A, N), -1, dtype=torch.long, device=device), generation=lng(A, N),
                  o2_debt_mol=f(A, N), anoxia=f(A, N), next_uid=lng(A), genome=genome, wh_live=wh_live,
                  hidden=hidden, ledger=new_ledger(A, device))


def _book(b: Bodies, key: str, x) -> None:
    """Add the arena sums of x [A, N] (or [A]) to ledger[key] in float64."""
    x = torch.as_tensor(x, device=b.device).double()
    b.ledger[key] += x.reshape(x.shape[0], -1).sum(-1)


# =========================================================================================== environment
@dataclass
class Env:
    """Conditions of one bout. Each field is a number, [A], [A, N] (per body) or a fine field [A, n, n] (sampled at
    the bodies' positions, needs the geometry): air temperature (K), water-vapour pressure (Pa), shortwave at the
    ground per horizontal m^2 (W/m^2), O2 partial pressure (Pa), wind (m/s), gravity (m/s^2), air pressure (Pa);
    ``dt_s`` the bout's length; ``heat_w`` optional extra heat absorbed by each body (W, e.g. from a fire)."""
    t_air_k: object
    vapour_pa: object
    sw_w_m2: object
    p_o2_pa: object
    wind_m_s: object
    gravity_m_s2: object
    p_air_pa: object
    dt_s: float = DAY_S / BOUTS
    heat_w: object = None


def constant_env(*, t_air_k=293.0, rel_humidity=0.5, vapour_pa=None, sw_w_m2=0.0, p_o2_pa=21200.0,
                 wind_m_s=None, gravity_m_s2=K.G_STANDARD, p_air_pa=K.P_STANDARD, dt_s=DAY_S / BOUTS,
                 heat_w=None) -> Env:
    """An environment from numbers or tensors (tests, stand-alone arenas); the vapour pressure defaults to
    ``rel_humidity`` x e_sat(t_air) and the wind to the climate's bulk-formula wind."""
    t = torch.as_tensor(t_air_k, dtype=torch.float32)
    if vapour_pa is None:
        vapour_pa = rel_humidity * cl.e_sat_pa(t)
    if wind_m_s is None:
        wind_m_s = cl.ClimateRules().wind_m_s
    return Env(t_air_k=t_air_k, vapour_pa=vapour_pa, sw_w_m2=sw_w_m2, p_o2_pa=p_o2_pa, wind_m_s=wind_m_s,
               gravity_m_s2=gravity_m_s2, p_air_pa=p_air_pa, dt_s=float(dt_s), heat_w=heat_w)


def patch_env(geom: pt.PatchGeometry, forcing: pt.PatchForcing, bout_sw, *, gravity_m_s2=K.G_STANDARD,
              p_air_pa=K.P_STANDARD, wind_m_s=None, dt_s=None) -> Env:
    """The environment of one bout from a patch forcing: the fine air temperature field, the cell's vapour pressure,
    ``bout_sw`` [A, n, n] (``patch.light`` or ``patch.step``'s ``bout_sw[:, bout]``), the cell's O2 partial pressure.
    ``gravity_m_s2`` and ``p_air_pa`` are numbers or [A] (the planet's), the wind the climate's by default."""
    return Env(t_air_k=pt.fine_temperature(geom, forcing), vapour_pa=forcing.vapour_pa(), sw_w_m2=bout_sw,
               p_o2_pa=forcing.p_o2_pa, wind_m_s=cl.ClimateRules().wind_m_s if wind_m_s is None else wind_m_s,
               gravity_m_s2=gravity_m_s2, p_air_pa=p_air_pa, dt_s=float(DAY_S / forcing.bouts if dt_s is None
                                                                            else dt_s))


def _at(x, b: Bodies, geom=None) -> torch.Tensor:
    """An environment value at every body, [A, N] float32."""
    A, N = b.shape
    t = torch.as_tensor(x, dtype=torch.float32, device=b.device)
    if t.dim() == 0:
        return t.expand(A, N)
    if t.dim() == 1:
        return t.reshape(-1).expand(A)[:, None].expand(A, N)
    if t.dim() == 2:
        return t.expand(A, N)
    if t.dim() == 3:
        if geom is None:
            raise ValueError("a fine field [A, n, n] needs the patch geometry")
        return pt.sample(geom, t, b.pos)
    raise ValueError(f"environment value of shape {tuple(t.shape)}")


def _env(b: Bodies, env: Env, geom=None) -> dict:
    d = {k: _at(getattr(env, k), b, geom) for k in ("t_air_k", "vapour_pa", "sw_w_m2", "p_o2_pa", "wind_m_s",
                                                    "gravity_m_s2", "p_air_pa")}
    d["heat_w"] = None if env.heat_w is None else _at(env.heat_w, b, geom)
    d["dt"] = float(env.dt_s)
    return d


# =========================================================================================== geometry and tissues
def fat_kg(b: Bodies) -> torch.Tensor:
    return b.reserve_j.clamp_min(0) / FAT_J_KG


def normal_water(b: Bodies) -> torch.Tensor:
    """The body's normal water [A, N]: LEAN_WATER x lean."""
    return LEAN_WATER * b.mass_kg


def reserve_capacity(b: Bodies) -> torch.Tensor:
    """The scale of the interoceptive fat channel (J) [A, N]: FAT_SCALE_PER_LEAN x lean x 39.5 MJ/kg. The store has
    no cap: it may exceed this scale."""
    return FAT_SCALE_PER_LEAN * b.mass_kg * FAT_J_KG


def store_level_j(b: Bodies) -> torch.Tensor:
    """The heritable store level (J) [A, N], ``fat_store`` x lean x 39.5 MJ/kg: fat beyond it (and beyond the bout's
    committed costs) builds lean tissue toward ``size_kg``."""
    return b.genome["fat_store"].float() * b.mass_kg * FAT_J_KG


def shape(b: Bodies) -> dict:
    """The ellipsoid of the body's whole mass (lean, fat, gut contents, water beyond the normal) at RHO_TISSUE:
    total mass, volume, equal-volume sphere radius r1, semi-axes (a, b), length, skin area, frontal area, fur radius.
    The geometry keeps RHO_TISSUE: the fat's lower density and the air the body holds enter its buoyancy
    (:func:`density`), not its size."""
    gut = b.gut[..., Q_KG].sum(-1)
    m = (b.mass_kg + fat_kg(b) + gut + (b.water_kg - normal_water(b))).clamp_min(MIN_KG)
    vol = m / RHO_TISSUE
    r1 = (3 * vol / (4 * math.pi)) ** (1.0 / 3.0)
    minor = r1 / ASPECT ** (1.0 / 3.0)
    major = ASPECT * minor
    fur = b.genome["fur_m"].float().clamp_min(0)
    return {"m": m, "volume": vol, "r1": r1, "a": major, "b": minor, "length": 2 * major,
            "area": 4 * math.pi * r1 ** 2 * SHAPE_FACTOR, "frontal": math.pi * minor ** 2, "r2": r1 + fur, "fur": fur}


def air_volume_m3(b: Bodies, tis: dict | None = None) -> torch.Tensor:
    """The air the body holds (m^3) [A, N], at the ambient pressure: the capacity of its lung tissue
    (``tissues()['lung'] / AIR_TISSUE_KG_PER_L``), i.e. ``air_l_kg`` x lean, less in proportion when the tissue
    budget is over-asked (a genome asking for more tissue than the body has holds the air its lung tissue holds)."""
    tis = tissues(b) if tis is None else tis
    return tis["lung"] / AIR_TISSUE_KG_PER_L * 1e-3


def density(b: Bodies, tis: dict | None = None) -> torch.Tensor:
    """The body's density for buoyancy (kg/m^3) [A, N]: its whole mass over lean / RHO_TISSUE + fat / RHO_FAT + (gut
    contents and water beyond the normal) / RHO_WATER + the air it holds (:func:`air_volume_m3`; the air's own mass,
    a thousandth of tissue's per m^3, is left out) (``PROVENANCE['RHO_FAT']``). A lean body without air (1,050
    kg/m^3) sinks; 0.048 L of air per kg lean (AIR_UNIT_L_KG) or about 0.45 kg of fat per kg lean float it in fresh
    water. The volume is the one at the surface: under water the air is compressed (Boyle), so a body that sinks at
    the surface sinks deeper down too, and one that floats stays at the surface."""
    gut = b.gut[..., Q_KG].sum(-1).clamp_min(0)
    extra = (b.water_kg - normal_water(b)).clamp_min(0)
    fat = fat_kg(b)
    lean = b.mass_kg.clamp_min(MIN_KG)
    m = lean + fat + gut + extra
    vol = lean / RHO_TISSUE + fat / RHO_FAT + (gut + extra) / RHO_WATER + air_volume_m3(b, tis).clamp_min(0)
    return m / vol


def active_units(b: Bodies, tis: dict | None = None) -> torch.Tensor:
    """Hidden units [A, N] (long) the body can run: k, but no more than its brain tissue holds (floor of brain mass /
    BRAIN_KG_PER_UNIT): a genome asking for more tissue than the body has (tissues' closure) computes with the brain
    it has, not the one it asks for."""
    tis = tissues(b) if tis is None else tis
    have = torch.floor(tis["brain"].double() / BRAIN_KG_PER_UNIT + 1e-3).long()
    return torch.minimum(b.genome["k"].long(), have.clamp_min(0)) * b.alive


def tissues(b: Bodies, sh: dict | None = None) -> dict:
    """Tissue masses (kg) [A, N]: muscle, enz_plant, enz_meat, gut (their sum), eye, ear, voice, brain (k x
    BRAIN_KG_PER_UNIT), fur (coat area x thickness x RHO_PELAGE), kidney (the gene's share), lung (the tissue that
    holds the body's air, ``air_l_kg`` x lean x AIR_TISSUE_KG_PER_L), carrier (the O2 carrier's protein,
    ``o2_carrier`` / HUFNER_ML_PER_G x lean), rest; closed by brain3.tissue_shares (a genome
    asking for more than the body gets every tissue scaled down), and ``asked`` (the unscaled sum of shares)."""
    sh = shape(b) if sh is None else sh
    g = b.genome
    lean = b.mass_kg.clamp_min(MIN_KG)
    brain = g["k"].float() * BRAIN_KG_PER_UNIT
    fur = sh["area"] * sh["fur"] * RHO_PELAGE
    kidney = g["kidney"].float().clamp_min(0) * b.mass_kg
    lung = g["air_l_kg"].float().clamp_min(0) * AIR_TISSUE_KG_PER_L * b.mass_kg
    carrier = g["o2_carrier"].float().clamp_min(0) / HUFNER_ML_PER_G * 1e-3 * b.mass_kg
    extra = (brain + fur + kidney + lung + carrier) / lean
    s = b3.tissue_shares({name: g[name] for name in b3.TISSUE_GENES}, brain_frac=extra)
    split = torch.where(extra > 0, s["brain"] / extra.clamp_min(1e-30), torch.ones_like(extra))
    out = {name: s[name] * b.mass_kg for name in b3.TISSUE_GENES}
    out["muscle"] = out.pop("muscle_frac")
    out["gut"] = out["enz_plant"] + out["enz_meat"]
    out["brain"] = brain * split * (b.mass_kg > 0)
    out["fur"] = fur * split * (b.mass_kg > 0)
    out["kidney"] = kidney * split
    out["lung"] = lung * split
    out["carrier"] = carrier * split
    out["rest"] = s["rest"] * b.mass_kg
    out["asked"] = s["asked"]
    return out


def arrhenius(t_k) -> torch.Tensor:
    """exp(-E / k (1 / T - 1 / T_ref)): the temperature factor of every tissue rate (maintenance, muscle power, jaw
    cycles, gut passage)."""
    t = torch.as_tensor(t_k, dtype=torch.float32).clamp_min(1.0)
    return torch.exp(-E_ARRHENIUS_EV / K_B_EV * (1.0 / t - 1.0 / T_REF_K))


def maintenance_ref_w(b: Bodies, tis: dict | None = None) -> torch.Tensor:
    """Maintenance power at T_REF (W) [A, N]: each tissue's mass x its rate, muscle at C_MUSCLE x muscle /
    MUSCLE_POWER_REF, the kidney at C_KIDNEY, the lung tissue at C_LUNG, the carrier at C_CARRIER, the fat store at
    the adipose rate; the coat
    costs nothing."""
    tis = tissues(b) if tis is None else tis
    muscle_rate = C_MUSCLE_W_KG * b.genome["muscle"].float() / MUSCLE_POWER_REF_W_KG
    p = (tis["muscle"] * muscle_rate + tis["gut"] * C_GUT_W_KG
         + (tis["eye"] + tis["ear"] + tis["brain"]) * C_NERVE_W_KG + tis["voice"] * C_MUSCLE_W_KG
         + tis["kidney"] * C_KIDNEY_W_KG + tis["lung"] * C_LUNG_W_KG + tis["carrier"] * C_CARRIER_W_KG
         + tis["rest"] * C_REST_W_KG
         + fat_kg(b) * C_ADIPOSE_W_KG)
    return p * b.alive


def muscle_ref_w(b: Bodies, tis: dict | None = None) -> torch.Tensor:
    """Peak mechanical power at T_REF (W) [A, N]: muscle gene x muscle mass x (1 - damage)."""
    tis = tissues(b) if tis is None else tis
    return b.genome["muscle"].float() * tis["muscle"] * damage_factor(b)


def muscle_peak_w(b: Bodies, tis: dict | None = None) -> torch.Tensor:
    """Peak mechanical power (W) [A, N] at the body's temperature: :func:`muscle_ref_w` x arrhenius(T_b) (muscle
    power is enzyme kinetics, Q10 about 2-3: Bennett 1985; E = 0.65 eV gives Q10 2.2 near 305 K)."""
    return muscle_ref_w(b, tis) * arrhenius(b.body_k)


def urine_salt_max(b: Bodies, tis: dict | None = None) -> torch.Tensor:
    """The most concentrated urine the kidney makes (kg salt per kg urine water) [A, N]: URINE_SALT_PER_KIDNEY x the
    kidney's share of lean mass, at most NACL_SOLUBILITY."""
    tis = tissues(b) if tis is None else tis
    share = tis["kidney"] / b.mass_kg.clamp_min(MIN_KG)
    return (URINE_SALT_PER_KIDNEY * share).clamp(max=NACL_SOLUBILITY)


def damage_factor(b: Bodies) -> torch.Tensor:
    """1 - damage, clamped to [0, 1] [A, N]: the share of muscle power (and, in senses3, of sense acuity) a damaged
    body keeps (PLANET-V3-SPEC 3)."""
    return (1 - b.damage).clamp(0, 1) * b.alive


def aerobic(p_o2_pa) -> torch.Tensor:
    p = torch.as_tensor(p_o2_pa, dtype=torch.float32).clamp_min(0)
    return p / (p + O2_HALF_PA)


def gut_capacity_kg(b: Bodies, tis: dict | None = None) -> torch.Tensor:
    """Dry matter the gut holds [A, N]: GUT_WET_PER_TISSUE x DIGESTA_DRY x gut tissue."""
    tis = tissues(b) if tis is None else tis
    return GUT_WET_PER_TISSUE * DIGESTA_DRY * tis["gut"]


def call_power_w(b: Bodies, loudness, tis: dict | None = None) -> torch.Tensor:
    """Acoustic power of a call (W) [A, N]: loudness x VOICE_W_PER_KG x sound-organ mass (senses3 propagates it)."""
    tis = tissues(b) if tis is None else tis
    return torch.as_tensor(loudness, dtype=torch.float32, device=b.device).clamp(0, 1) * VOICE_W_PER_KG \
        * tis["voice"] * b.alive


# =========================================================================================== air and water
def air_props(t_k, p_pa) -> dict:
    """Density, conductivity, kinematic viscosity, vapour diffusivity and thermal diffusivity of air."""
    t = torch.as_tensor(t_k, dtype=torch.float32).clamp_min(150.0)
    p = torch.as_tensor(p_pa, dtype=torch.float32).clamp_min(1.0)
    rho = p * M_AIR / (K.R_GAS * t)
    mu = MU_AIR_300 * (t / 300.0) ** MU_AIR_EXP
    k = K_AIR_300 * (t / 300.0) ** K_AIR_EXP
    dv = DV_AIR_273 * (t / 273.15) ** DV_EXP * (K.P_STANDARD / p)
    return {"rho": rho, "k": k, "nu": mu / rho, "dv": dv, "alpha": k / (rho * CP_AIR)}


def nusselt(re, pr) -> torch.Tensor:
    """Forced convection from a sphere (Ranz-Marshall)."""
    c0, c1, e_re, e_pr = RANZ_MARSHALL
    return c0 + c1 * re.clamp_min(0) ** e_re * pr ** e_pr


def nusselt_free(ra, pr) -> torch.Tensor:
    """Free convection from a sphere (Churchill 1983): 2 + 0.589 Ra^(1/4) / [1 + (0.469 / Pr)^(9/16)]^(4/9)."""
    a, c = CHURCHILL_SPHERE
    return 2.0 + a * ra.clamp_min(0) ** 0.25 / (1.0 + (c / pr) ** (9.0 / 16.0)) ** (4.0 / 9.0)


def nusselt_mixed(re, pr, ra) -> torch.Tensor:
    """Forced and free convection combined (``PROVENANCE['NU_MIX_EXP']``): 2 + ((Nu_F - 2)^3 + (Nu_N - 2)^3)^(1/3)."""
    n = NU_MIX_EXP
    return 2.0 + ((nusselt(re, pr) - 2.0) ** n + (nusselt_free(ra, pr) - 2.0) ** n) ** (1.0 / n)


def water_expansion(t_k) -> torch.Tensor:
    """|beta| (1/K) of fresh water at t_k (``PROVENANCE['THIESEN_WATER']``): -d ln rho / dT of the Thiesen law."""
    a, bb, d, c = THIESEN_WATER
    t = (torch.as_tensor(t_k, dtype=torch.float32) - 273.15).clamp(-5.0, 100.0)
    f = (t + a) * (t - c) ** 2 / (bb * (t + d))
    df = ((t - c) ** 2 + 2 * (t + a) * (t - c)) / (bb * (t + d)) - (t + a) * (t - c) ** 2 / (bb * (t + d) ** 2)
    return (df / (1 - f)).abs()


def latent_heat(t_k) -> torch.Tensor:
    return LV_273 - LV_SLOPE * (torch.as_tensor(t_k, dtype=torch.float32) - 273.15)


def e_sat(t_k) -> torch.Tensor:
    """Saturation vapour pressure (Pa) over water, the climate's Magnus form, at T clamped to [150, 400] K (empty
    slots hold 0 K)."""
    return cl.e_sat_pa(torch.as_tensor(t_k, dtype=torch.float32).clamp(150.0, 400.0))


def _de_sat_dt(t_k) -> torch.Tensor:
    t = torch.as_tensor(t_k, dtype=torch.float32).clamp(150.0, 400.0)
    return e_sat(t) * 17.625 * 243.04 / (t - 273.15 + 243.04) ** 2


def sky_emissivity(t_k, vapour_pa) -> torch.Tensor:
    a, e = BRUTSAERT
    return (a * (torch.as_tensor(vapour_pa).clamp_min(0) / 100.0 / t_k) ** e).clamp(0, 1)


def heat_exchange(b: Bodies, e: dict, submerged=None, sh: dict | None = None) -> dict:
    """Conductances of the body to its surroundings (``PROVENANCE['heat_balance']``): G (W/K, core to the operative
    temperature), T_e (K), G_v (kg s^-1 Pa^-1, water vapour, skin + fur + boundary layer in series, in air), the
    convective h (forced and free convection, :func:`nusselt_mixed`; in water at the body's speed plus the wind's
    drift), the fur resistance and the heat capacity C (J/K), all [A, N]."""
    sh = shape(b) if sh is None else sh
    s = torch.zeros_like(b.mass_kg) if submerged is None else submerged.clamp(0, 1)
    ta = e["t_air_k"]
    air = air_props(ta, e["p_air_pa"])
    r1, r2 = sh["r1"], sh["r2"]
    d2 = 2 * r2
    a2 = 4 * math.pi * r2 ** 2 * SHAPE_FACTOR
    a1 = sh["area"]
    u_air = e["wind_m_s"] + b.speed
    g = e["gravity_m_s2"]
    dtk = (b.body_k - ta).abs() * b.alive                     # core to fluid: an upper bound of the surface's
    ra_air = g / ta.clamp_min(1.0) * dtk * d2 ** 3 / (air["nu"] * air["alpha"])
    h_c = nusselt_mixed(u_air * d2 / air["nu"], PR_AIR, ra_air) * air["k"] / d2
    h_r = 4 * EMISSIVITY_BODY * K.SIGMA * ta ** 3
    u_w = WATER_DRIFT * e["wind_m_s"] + b.speed               # the water moves with the wind's drift
    ra_w = g * water_expansion(ta) * dtk * d2 ** 3 / (NU_WATER * ALPHA_WATER)
    h_w = nusselt_mixed(u_w * d2 / NU_WATER, PR_WATER, ra_w) * K_WATER / d2
    eps_sky = sky_emissivity(ta, e["vapour_pa"])
    lw_net = EMISSIVITY_BODY * K.SIGMA * ta ** 4 * (1 - 0.5 * eps_sky - 0.5 * EMISSIVITY_GROUND)
    sw_abs = (1 - ALBEDO_SW_BODY) * SW_VIEW * e["sw_w_m2"].clamp_min(0)
    te_air = ta + (sw_abs - lw_net) / (h_c + h_r)
    g_air = (1 - s) * (h_c + h_r) * a2
    g_wat = s * h_w * a2
    g_out = (g_air + g_wat).clamp_min(1e-12)
    t_e = (g_air * te_air + g_wat * ta) / g_out
    shell = (1.0 / r1 - 1.0 / r2).clamp_min(0)
    k_fur = (1 - s) * K_FUR + s * K_WATER
    r_fur = shell / (4 * math.pi * k_fur * SHAPE_FACTOR)
    G = 1.0 / (r_fur + 1.0 / g_out)
    if e.get("heat_w") is not None:
        t_e = t_e + e["heat_w"] / G.clamp_min(1e-12)
    # water vapour: skin, fur (diffusion through the shell) and boundary layer (Chilton-Colburn) in series
    g_skin = b.genome["skin_perm"].float() * a1
    r_fur_v = shell / (4 * math.pi * air["dv"] * SHAPE_FACTOR * b3.WATER_KG_PER_J)
    g_bl = b3.WATER_KG_PER_J * h_c / (air["rho"] * CP_AIR) * (air["dv"] / air["alpha"]) ** (2.0 / 3.0) * a2
    g_v = 1.0 / (1.0 / g_skin.clamp_min(1e-30) + r_fur_v + 1.0 / g_bl.clamp_min(1e-30))
    return {"G": G * b.alive, "T_e": t_e, "G_v": g_v * (1 - s) * b.alive, "h_c": h_c, "h_r": h_r, "R_fur": r_fur,
            "C": CP_TISSUE * sh["m"], "area_out": a2}


# =========================================================================================== breathing
def inspired_po2(e: dict, t_k) -> torch.Tensor:
    """The O2 partial pressure in the lungs as air is breathed in (Pa) [A, N]: the ambient pO2 saturated with water
    vapour at the body's temperature, pO2 x (1 - e_sat(T_b) / p_air) (``e`` an environment dict of :func:`_env`)."""
    t = torch.as_tensor(t_k, dtype=torch.float32).clamp_min(150.0)
    p_air = e["p_air_pa"].clamp_min(1.0)
    return e["p_o2_pa"].clamp_min(0) * (1 - e_sat(t) / p_air).clamp(0, 1)


def _o2_store_parts(b: Bodies, e: dict, tis: dict, t_k) -> dict:
    """:func:`o2_store_parts` from an environment dict of :func:`_env` at body temperature ``t_k``."""
    t = torch.as_tensor(t_k, dtype=torch.float32, device=b.device).clamp_min(150.0)
    p_alv = inspired_po2(e, t) * (1 - RESP_EXTRACTION)                    # the exhaled (alveolar) level
    floor = O2_HOLD_FLOOR_SHARE * p_alv
    lung = air_volume_m3(b, tis).clamp_min(0) * (p_alv - floor).clamp_min(0) / (K.R_GAS * t)
    bound = tis["carrier"].clamp_min(0) * HUFNER_ML_PER_G * 1e3 / ML_PER_MOL_STP    # mol O2 when saturated
    carrier = bound * (aerobic(p_alv) - aerobic(floor)).clamp_min(0)
    return {"lung": lung * b.alive, "carrier": carrier * b.alive}


def _o2_store(b: Bodies, e: dict, tis: dict, t_k) -> torch.Tensor:
    parts = _o2_store_parts(b, e, tis, t_k)
    return parts["lung"] + parts["carrier"]


def o2_store_parts(b: Bodies, env: Env, geom=None, tis: dict | None = None, t_k=None) -> dict:
    """The O2 a body can use with its airway under water (mol) [A, N], by store (``PROVENANCE['breath_hold']``):
    ``lung``, the air it holds (:func:`air_volume_m3`, ideal gas at ``t_k``, default T_b) from the alveolar pO2 (the
    inspired pO2 x (1 - RESP_EXTRACTION)) down to O2_HOLD_FLOOR_SHARE of it, and ``carrier``, its O2 carrier
    (``o2_carrier``) unloading between the same two pressures by the aerobic law a(p) = p / (p + O2_HALF_PA)."""
    e = _env(b, env, geom)
    tis = tissues(b) if tis is None else tis
    return _o2_store_parts(b, e, tis, b.body_k if t_k is None else t_k)


def o2_store_mol(b: Bodies, env: Env, geom=None, tis: dict | None = None, t_k=None) -> torch.Tensor:
    """The whole O2 store (mol) [A, N]: :func:`o2_store_parts` summed."""
    parts = o2_store_parts(b, env, geom, tis, t_k)
    return parts["lung"] + parts["carrier"]


def _o2_supply(b: Bodies, e: dict, tis: dict, sh: dict, t_k) -> torch.Tensor:
    """:func:`o2_supply_mol_s` from an environment dict."""
    p_insp = inspired_po2(e, torch.as_tensor(t_k, dtype=torch.float32, device=b.device))
    lung = LUNG_O2_MOL_S_KG * tis["lung"].clamp_min(0) * p_insp / P_O2_REF_PA
    skin = SKIN_O2_MOL_S_M2_PA * sh["area"] * e["p_o2_pa"].clamp_min(0)
    return (lung + skin) * b.alive


def o2_supply_mol_s(b: Bodies, env: Env, geom=None, tis: dict | None = None, sh: dict | None = None,
                    t_k=None) -> torch.Tensor:
    """The most O2 a breathing body takes up (mol/s) [A, N] (``PROVENANCE['breathing']``): its lung,
    LUNG_O2_MOL_S_KG x lung tissue x the inspired pO2 / P_O2_REF_PA, and its skin, SKIN_O2_MOL_S_M2_PA x area x
    pO2. A body without a lung breathes through its skin only."""
    e = _env(b, env, geom)
    sh = shape(b) if sh is None else sh
    tis = tissues(b, sh) if tis is None else tis
    return _o2_supply(b, e, tis, sh, b.body_k if t_k is None else t_k)


def resting_power_w(b: Bodies, tis: dict | None = None, t_k=None) -> torch.Tensor:
    """What the body spends standing still (W) [A, N]: maintenance at T_b (or ``t_k``) with its repair share on
    top."""
    tis = tissues(b) if tis is None else tis
    t = b.body_k if t_k is None else t_k
    return maintenance_ref_w(b, tis) * arrhenius(t) * (1 + b.genome["repair"].float().clamp(0, 1))


def _sustained(b: Bodies, e: dict, tis: dict, sh: dict) -> torch.Tensor:
    muscle = SUSTAINED_SHARE * aerobic(e["p_o2_pa"]) * muscle_peak_w(b, tis)
    spare = (_o2_supply(b, e, tis, sh, b.body_k) * OXY_J_PER_MOL_O2 - resting_power_w(b, tis)).clamp_min(0)
    return torch.minimum(muscle, MUSCLE_EFF * spare) * b.alive


def sustained_power_w(b: Bodies, env: Env, geom=None, tis: dict | None = None, sh: dict | None = None
                      ) -> torch.Tensor:
    """The mechanical power a body sustains (W) [A, N] (``PROVENANCE['breathing']``): its muscles' SUSTAINED_SHARE
    x a(pO2) x peak power, at most MUSCLE_EFF x what its O2 supply leaves after its resting costs."""
    e = _env(b, env, geom)
    sh = shape(b) if sh is None else sh
    tis = tissues(b, sh) if tis is None else tis
    return _sustained(b, e, tis, sh)


def breath_hold_s(b: Bodies, env: Env, geom=None, power_w=None, tis: dict | None = None) -> torch.Tensor:
    """How long a body lasts with its airway under water before its store is spent and it loses consciousness (s)
    [A, N]: its O2 store (:func:`o2_store_mol`) over its O2 use, power / OXY_J_PER_MOL_O2. ``power_w`` [A, N] is the
    body's metabolic power; by default :func:`resting_power_w`. :func:`metabolism` uses the bout's own power.
    Anoxia damage starts after it (ANOXIA_TOLERANCE_S). Infinite for empty slots (and a body that uses no O2)."""
    tis = tissues(b) if tis is None else tis
    if power_w is None:
        power_w = resting_power_w(b, tis)
    rate = torch.as_tensor(power_w, dtype=torch.float32, device=b.device).clamp_min(0) / OXY_J_PER_MOL_O2
    store = o2_store_mol(b, env, geom, tis)
    return torch.where(b.alive & (rate > 0), store / rate.clamp_min(1e-30), torch.full_like(store, math.inf))


#: the dive structure of a stretch of time (``PROVENANCE['breath_hold']``): the time the airway was under water from
#: its start until the first breath, the longest stretch between two breaths, the time since the last breath at its
#: end, and whether it was under water all along (then lead = trail = the whole stretch)
DIVE_KEYS = ("dive_lead_s", "dive_inner_s", "dive_trail_s", "dive_all")


def dive_piece(sunk_s, total_s=None) -> dict:
    """The dive structure of one stretch of time with ``sunk_s`` under water in one piece (``total_s`` its length:
    all under water when sunk_s >= total_s, else the dive counted as inner, between breaths)."""
    sk = torch.as_tensor(sunk_s, dtype=torch.float32)
    tot = sk if total_s is None else torch.as_tensor(total_s, dtype=torch.float32, device=sk.device).expand_as(sk)
    whole = sk >= tot * (1 - 1e-6)
    z = torch.zeros_like(sk)
    return {"dive_lead_s": torch.where(whole, sk, z), "dive_inner_s": torch.where(whole, z, sk),
            "dive_trail_s": torch.where(whole, sk, z), "dive_all": whole.float()}


def dive_merge(a: dict, b: dict) -> dict:
    """The dive structure of stretch ``a`` followed by stretch ``b`` (:data:`DIVE_KEYS`; a stretch of no time is
    'all under water' with zeros, so it changes nothing)."""
    aa, ba = a["dive_all"] > 0.5, b["dive_all"] > 0.5
    lead = torch.where(aa, a["dive_lead_s"] + b["dive_lead_s"], a["dive_lead_s"])
    trail = torch.where(ba, a["dive_trail_s"] + b["dive_trail_s"], b["dive_trail_s"])
    inner = torch.where(aa, b["dive_inner_s"], torch.where(ba, a["dive_inner_s"], torch.maximum(
        torch.maximum(a["dive_inner_s"], b["dive_inner_s"]), a["dive_trail_s"] + b["dive_lead_s"])))
    inner = torch.where(aa & ba, torch.zeros_like(inner), inner)
    return {"dive_lead_s": lead, "dive_inner_s": inner, "dive_trail_s": trail, "dive_all": (aa & ba).float()}


PROVENANCE["breath_hold"] = (DN, "under water (and in air when its costs exceed its supply) a body draws its O2 "
                                 "store, the air it holds and its carrier (o2_store_parts), at its own O2 use "
                                 "(metabolic power / OXY_J_PER_MOL_O2): o2_debt_mol grows and is repaid at the next "
                                 "breath that meets the need. Nothing is harmed while the debt is within the store "
                                 "(a breath-hold shorter than the store is harmless; review of 4 Oct 2026: before, "
                                 "every second under water was damage); beyond it the body is unconscious and the "
                                 "debt beyond the store is anoxia: damage = that O2 / (ANOXIA_TOLERANCE_S x the "
                                 "resting O2 use at T_REF), healed as a wound. A bout's sunk time is placed by its "
                                 "dive structure (DIVE_KEYS: the dive continued from the last bout, the longest dive "
                                 "within it, the dive it ends in); a bout reported only as sunk_s is one dive. A "
                                 "resting 70 kg mammal at 310 K with 0.054 L/kg of air and 15 mL/kg of carrier holds "
                                 "about 0.4 mmol/kg and is unconscious after about 1.5-2 min, dead about 3 min later; "
                                 "cold bodies last longer (lower use), working ones and bodies in thin or O2-poor air "
                                 "shorter")


# =========================================================================================== chemistry of the body
def _to_reserve(b: Bodies, j, c) -> torch.Tensor:
    """Absorbed or released matter (J, kg C; [A, N], >= 0) becomes fat: the store gains min(J, C / FAT_C_PER_J);
    the energy beyond it leaves as heat, the carbon beyond it as CO2 (no O2 is taken). Returns that heat (J) [A, N]
    so the caller can put it into the heat balance."""
    j = j.clamp_min(0)
    c = c.clamp_min(0)
    gain = torch.minimum(j, c / FAT_C_PER_J)
    b.reserve_j = b.reserve_j + gain
    co2_c = c - gain * FAT_C_PER_J
    heat = j - gain
    _book(b, "e_conversion", heat)
    _book(b, "c_co2", co2_c)
    _book(b, "co2_mol", co2_c / M_C)
    return heat


def resp_water_per_j(t_k, e: dict) -> torch.Tensor:
    """Respiratory water lost per joule of fat oxidised (kg/J) [A, N]: exhaled air is saturated at T_b, so the water
    per mol O2 is (e_sat(T_b) - e_air) / (pO2 x RESP_EXTRACTION) mol, at OXY_J_PER_MOL_O2 per mol O2."""
    deficit = (e_sat(t_k) - e["vapour_pa"]).clamp_min(0)
    return deficit / (e["p_o2_pa"].clamp_min(1.0) * RESP_EXTRACTION) * M_H2O / OXY_J_PER_MOL_O2


def _oxidise(b: Bodies, j, e: dict, water_per_j=None) -> None:
    """Book the oxidation of j joules of fat (already taken from the store; [A, N] >= 0): O2, CO2 and metabolic
    water by tripalmitin's stoichiometry, and the respiratory water lost with the air breathed for that O2
    (``water_per_j`` kg/J, the heat balance's own; :func:`resp_water_per_j` at the body's temperature otherwise)."""
    j = j.clamp_min(0) * b.alive
    kg = j / FAT_J_KG
    o2 = kg * O2_MOL_PER_KG_FAT
    h2o = kg * H2O_MOL_PER_KG_FAT
    co2 = kg * CO2_MOL_PER_KG_FAT
    metabolic = h2o * M_H2O
    b.water_kg = b.water_kg + metabolic
    wpj = resp_water_per_j(b.body_k, e) if water_per_j is None else water_per_j
    resp = torch.minimum(j * wpj, b.water_kg.clamp_min(0))
    b.water_kg = b.water_kg - resp
    _book(b, "e_oxidised", j)
    _book(b, "fat_ox_kg", kg)
    _book(b, "o2_mol", o2)
    _book(b, "co2_mol", co2)
    _book(b, "co2_ox_mol", co2)
    _book(b, "h2o_ox_mol", h2o)
    _book(b, "c_co2", co2 * M_C)
    _book(b, "w_metabolic", metabolic)
    _book(b, "w_resp", resp)


#: per kg of lean tissue catabolised: nitrogen to the free pool; the rest becomes fat (carbon-limited)
_CAT_J = E_LEAN - N_LEAN * UREA_J_PER_KG_N
_CAT_C = C_LEAN - N_LEAN * UREA_C_PER_KG_N
CATABOLISM_GAIN_J_KG = min(_CAT_J, _CAT_C / FAT_C_PER_J)
#: fat needed per kg of lean tissue built from the free nitrogen pool (carbon- or energy-limited)
BUILD_FAT_J_KG = max(_CAT_J, _CAT_C / FAT_C_PER_J)
# catabolism is energy-limited, so it releases no conversion heat (its surplus carbon leaves as CO2): the heat balance
# needs no term for it
assert _CAT_J <= _CAT_C / FAT_C_PER_J
PROVENANCE["CATABOLISM_GAIN_J_KG"] = (D_, "fat gained per kg of lean tissue burnt: its N goes to the free pool "
                                          "(with the energy and carbon urea will carry), the rest becomes fat, "
                                          "energy-limited (6.4 MJ; the surplus carbon leaves as CO2, no heat)")
PROVENANCE["BUILD_FAT_J_KG"] = (D_, "fat used per kg of lean tissue built with N from the free pool: enough for both "
                                    "its energy and its carbon (7.3 MJ); the surplus leaves as heat (in the heat "
                                    "balance) or CO2")


def _spend(b: Bodies, j, e: dict, water_per_j=None) -> torch.Tensor:
    """Pay j joules [A, N] by oxidising fat; a store that runs short is refilled by burning lean tissue (its N to the
    free pool, its bone mineral excreted: ``i_urine``). Returns the joules actually oxidised (less than j only when
    the body has nothing left)."""
    j = j.clamp_min(0) * b.alive
    short = (j - b.reserve_j).clamp_min(0)
    burn = torch.minimum(short / CATABOLISM_GAIN_J_KG, b.mass_kg.clamp_min(0)) * b.alive
    b.mass_kg = b.mass_kg - burn
    b.n_kg = b.n_kg + burn * N_LEAN
    _book(b, "i_urine", burn.double() * LEAN_MINERAL)
    _to_reserve(b, burn * _CAT_J, burn * _CAT_C)
    paid = torch.minimum(j, b.reserve_j.clamp_min(0))
    b.reserve_j = b.reserve_j - paid
    _oxidise(b, paid, e, water_per_j)
    return paid


# =========================================================================================== food chemistry tables
def _component_class(comp: str):
    """(gut class or None for water, J/kg, kg C/kg, kg N/kg) of one makeup component."""
    if comp == "H2O":
        return None, 0.0, 0.0, 0.0
    el = mt.formula_elements(comp)
    c, n = el.get("C", 0.0), el.get("N", 0.0)
    if comp == "protein":
        return G_PROTEIN, PROTEIN_J_KG, c, n
    if comp == LIPID:
        return G_LIPID, FAT_J_KG, c, n
    if comp == mt.CELLULOSE:
        return G_PLANT, float(mt.GROSS_J_KG[mt.CELLULOSE]), c, n
    if comp == mt.WOOD:
        return G_PLANT, _hhv(mt.WOOD), c, n
    return G_INERT, 0.0, c, n


def _species_tables():
    """[S, G, Q] per kg of each species (raw and cooked) and [S] its water share."""
    G, Q = len(GUT_CLASSES), len(GUT_Q)
    raw = torch.zeros(mt.S, G, Q, dtype=torch.float64)
    water = torch.zeros(mt.S, dtype=torch.float64)
    for i, s in enumerate(mt.SPECIES):
        for comp, x in mt.MAKEUP[s]:
            g, j, c, n = _component_class(comp)
            if g is None:
                water[i] += x
                continue
            raw[i, g] += torch.tensor([x, x * j, x * c, x * n], dtype=torch.float64)
    cooked = raw.clone()
    gain = torch.tensor([float(mt.COOK_GAIN[s]) > 1.0 for s in mt.SPECIES])
    for g in (G_PROTEIN, G_LIPID):
        cooked[gain, G_COOKED] += cooked[gain, g]
        cooked[gain, g] = 0.0
    return raw, cooked, water


def _bite_toughness() -> torch.Tensor:
    vals = []
    for s in mt.SPECIES:
        if s in ORGANIC_TOUGHNESS_J_M2:
            vals.append(ORGANIC_TOUGHNESS_J_M2[s])
        elif float(mt.GRANULAR[s]) > 0:
            vals.append(0.0)
        elif float(mt.MOHS[s]) > ENAMEL_MOHS:
            vals.append(math.inf)
        else:
            e_mod = E_METAL_PA if (mt.CLASS[s] == "metal" or s == "native_copper") else E_MINERAL_PA
            vals.append((float(mt.TOUGHNESS[s]) * 1e6) ** 2 / e_mod)
    return torch.tensor(vals, dtype=torch.float64)


SPECIES_GUT, SPECIES_GUT_COOKED, SPECIES_WATER = _species_tables()
BITE_TOUGHNESS_J_M2 = _bite_toughness()
HIDE_TOUGHNESS_J_M2 = ORGANIC_TOUGHNESS_J_M2["hide"]
PROVENANCE["SPECIES_GUT"] = (D_, "per species and kg: the makeup (materials.MAKEUP) split into gut classes with "
                                 "each component's gross energy, carbon and nitrogen; water apart")
PROVENANCE["SPECIES_GUT_COOKED"] = (D_, "as SPECIES_GUT with the protein and lipid of species with a cooking gain "
                                        "in the cooked class")
PROVENANCE["SPECIES_WATER"] = (D_, "water share of each species' makeup")
PROVENANCE["BITE_TOUGHNESS_J_M2"] = PROVENANCE["bite_toughness"]
PROVENANCE["HIDE_TOUGHNESS_J_M2"] = (RN, "a bite on a body must cut its hide: ORGANIC_TOUGHNESS_J_M2['hide']")


def _plant_row(prules: pt.PatchRules, device) -> torch.Tensor:
    """[G, Q] per kg dry of the patch's soft plant tissue: plant class, 18.5 MJ, the patch's carbon fraction and its
    leaf C:N (as patch.take_plant books them)."""
    br = prules.bio
    row = torch.zeros(len(GUT_CLASSES), len(GUT_Q), dtype=torch.float64, device=device)
    row[G_PLANT] = torch.tensor([1.0, PLANT_J_KG, br.plant_carbon_fraction,
                                 br.plant_carbon_fraction / br.cn_leaf], dtype=torch.float64)
    return row


def _lean_row(device) -> torch.Tensor:
    """[G, Q] per kg of lean tissue (its dry part; water apart): protein, lipid and bone mineral."""
    row = torch.zeros(len(GUT_CLASSES), len(GUT_Q), dtype=torch.float64, device=device)
    row[G_PROTEIN] = torch.tensor([LEAN_PROTEIN, LEAN_PROTEIN * PROTEIN_J_KG, LEAN_PROTEIN * PROTEIN_C_KG,
                                   LEAN_PROTEIN * PROTEIN_N_KG], dtype=torch.float64)
    row[G_LIPID] = torch.tensor([LEAN_LIPID, LEAN_LIPID * FAT_J_KG, LEAN_LIPID * FAT_C_KG, 0.0], dtype=torch.float64)
    row[G_INERT, Q_KG] = LEAN_MINERAL
    return row


def _fat_row(device) -> torch.Tensor:
    row = torch.zeros(len(GUT_CLASSES), len(GUT_Q), dtype=torch.float64, device=device)
    row[G_LIPID] = torch.tensor([1.0, FAT_J_KG, FAT_C_KG, 0.0], dtype=torch.float64)
    return row


# =========================================================================================== stocks and ledgers
def stocks(b: Bodies) -> dict:
    """What the living hold, [A] float64: energy (lean + fat + gut + the urea energy of free N), carbon, water,
    nitrogen, inert matter (gut contents and the lean's bone mineral), salt."""
    a = b.alive.double()
    lean = b.mass_kg.double()
    gut = b.gut.double()
    n_free = b.n_kg.double()

    def tot(x):
        return (x * a).sum(-1)

    return {"energy": tot(lean * E_LEAN + b.reserve_j.double() + gut[..., Q_J].sum(-1) + n_free * UREA_J_PER_KG_N),
            "carbon": tot(lean * C_LEAN + b.reserve_j.double() * FAT_C_PER_J + gut[..., Q_C].sum(-1)
                          + n_free * UREA_C_PER_KG_N),
            "water": tot(b.water_kg.double()),
            "nitrogen": tot(lean * N_LEAN + gut[..., Q_N].sum(-1) + n_free),
            "inert": tot(gut[..., G_INERT, Q_KG] + lean * LEAN_MINERAL),
            "salt": tot(b.salt_kg.double())}


def ledger_errors(b: Bodies) -> dict:
    """Stock minus (inflows - outflows) since the founding, [A] float64, for energy, carbon, water, nitrogen, inert
    matter and salt, each with its scale (the largest of the stock and the gross flows); ``oxygen``: the O-atom
    balance of oxidation, 2 O2 + O of the fat oxidised - 2 CO2 - H2O (mol), with its scale."""
    L, s = b.ledger, stocks(b)
    flows = {"energy": (L["e_founded"] + L["e_food"],
                        L["e_oxidised"] + L["e_conversion"] + L["e_faeces"] + L["e_urine"] + L["e_dead"]),
             "carbon": (L["c_founded"] + L["c_food"], L["c_co2"] + L["c_faeces"] + L["c_urine"] + L["c_dead"]),
             "water": (L["w_founded"] + L["w_drunk"] + L["w_food"] + L["w_metabolic"],
                       L["w_evap"] + L["w_resp"] + L["w_faeces"] + L["w_urine"] + L["w_dead"]),
             "nitrogen": (L["n_founded"] + L["n_food"], L["n_urine"] + L["n_faeces"] + L["n_dead"]),
             "inert": (L["i_founded"] + L["i_food"] + L["i_built"], L["i_faeces"] + L["i_urine"] + L["i_dead"]),
             "salt": (L["salt_in"], L["salt_out"] + L["salt_dead"])}
    out = {}
    for k, (inn, outf) in flows.items():
        out[k] = s[k] - (inn - outf)
        out[f"{k}_scale"] = torch.maximum(s[k].abs(), inn + outf)
    o_in = 2 * L["o2_mol"] + L["fat_ox_kg"] * O_MOL_PER_KG_FAT
    out["oxygen"] = o_in - 2 * L["co2_ox_mol"] - L["h2o_ox_mol"]
    out["oxygen_scale"] = o_in
    # the carbon booked as CO2 is the CO2 booked in moles
    out["co2"] = L["c_co2"] - L["co2_mol"] * M_C
    out["co2_scale"] = L["c_co2"].abs()
    return out


def reset_ledgers(b: Bodies) -> None:
    """Re-base every ledger on the current stocks (counters restart at zero; the stocks are booked as founded)."""
    A = b.shape[0]
    b.ledger = new_ledger(A, b.device)
    s = stocks(b)
    for key, name in (("e_founded", "energy"), ("c_founded", "carbon"), ("w_founded", "water"),
                      ("n_founded", "nitrogen"), ("i_founded", "inert")):
        b.ledger[key] += s[name]
    b.ledger["salt_in"] += s["salt"]


def gas_mol(b: Bodies) -> dict:
    """Cumulative gas exchange of the bodies per arena [A] float64 (mol): O2 taken, CO2 given (oxidation and
    conversion), metabolic water made; for the world's air ledger (patch.per_world sums arenas into worlds)."""
    return {"o2_mol": b.ledger["o2_mol"].clone(), "co2_mol": b.ledger["co2_mol"].clone(),
            "h2o_mol": b.ledger["h2o_ox_mol"].clone()}


# =========================================================================================== founders
def empty_genome(A: int, N: int, in_dim: int, hidden: int = b3.HIDDEN, weight_dtype=torch.float32, device="cpu",
                 n_intero: int = b3.N_INTERO) -> dict:
    """A zero genome [A, N, ...] holding every gene a founder draws (brain3.random_genome3's weights in
    ``weight_dtype``, ``k`` long, every :data:`GENE_SPECS` gene float32): the container :func:`found_into` fills."""
    o = b3.LAYOUT3.out_dim
    w = dict(dtype=weight_dtype, device=device)
    g = {"Wx": torch.zeros(A, N, in_dim, hidden, **w), "Wh": torch.zeros(A, N, hidden, hidden, **w),
         "b": torch.zeros(A, N, hidden, **w), "Wo": torch.zeros(A, N, hidden, o, **w), "bo": torch.zeros(A, N, o, **w),
         "k": torch.zeros(A, N, dtype=torch.long, device=device)}
    for name, spec in GENE_SPECS.items():
        g[name] = torch.zeros(A, N, *((n_intero,) if spec.per_intero else ()), dtype=torch.float32, device=device)
    return g


def found_into(b: Bodies, a0: int, a1: int, F: int, gen: torch.Generator, *, in_dim: int, hidden: int = b3.HIDDEN,
               k_range=None, t_k=293.0) -> None:
    """Put the founders of arenas a0..a1-1 into the container ``b`` (slots 0..F-1, empty before), drawn on ``gen``'s
    device in :func:`found`'s order and copied to ``b``'s device: a world's founders come from its own generator, and
    no second copy of the whole state is made (a GPU build holds the state once). The ledgers are not touched
    (:func:`reset_ledgers` after the last call books the founders)."""
    A_w, N = a1 - a0, b.shape[1]
    if not 0 < F <= N:
        raise ValueError("founders F must satisfy 0 < F <= N")
    k_range = (2, min(32, hidden)) if k_range is None else k_range
    gd = gen.device
    dev = b.device
    gf = b3.random_genome3(gen, (A_w, F), in_dim, hidden=hidden, k_range=k_range, device=gd,
                           dtype=b.genome["Wh"].dtype)
    gf.update(b3.random_body_genes(gen, (A_w, F), gd,
                                   {k: s for k, s in EXTRA_SPECS.items() if k not in FOUNDER_LATE_GENES}))
    pos = torch.rand(A_w, F, 2, generator=gen, device=gd) * b.L
    heading = torch.rand(A_w, F, generator=gen, device=gd) * (2 * math.pi)
    gf.update(b3.random_body_genes(gen, (A_w, F), gd, {k: EXTRA_SPECS[k] for k in FOUNDER_LATE_GENES}))
    for name, t in gf.items():
        b.genome[name][a0:a1, :F] = t.to(dev, b.genome[name].dtype)
    if b.wh_live is not None:
        b.wh_live[a0:a1, :F] = gf["Wh"].to(dev, b.wh_live.dtype)
    b.alive[a0:a1, :F] = True
    b.pos[a0:a1, :F] = pos.to(dev)
    b.heading[a0:a1, :F] = heading.to(dev)
    size = b.genome["size_kg"][a0:a1, :F].float()
    b.mass_kg[a0:a1, :F] = size
    b.frame_kg[a0:a1, :F] = size
    b.reserve_j[a0:a1, :F] = FOUNDER_FAT_PER_LEAN * size * FAT_J_KG
    b.water_kg[a0:a1, :F] = LEAN_WATER * size
    tk = torch.as_tensor(t_k, dtype=torch.float32, device=dev)
    b.body_k[a0:a1, :F] = tk.reshape(A_w, -1).expand(A_w, F) if tk.dim() > 0 else tk
    ids = torch.arange(F, device=dev).expand(A_w, F)
    b.uid[a0:a1, :F] = ids
    b.founder[a0:a1, :F] = ids
    b.next_uid[a0:a1] = F


def found(A: int, N: int, F: int, L: float, gen: torch.Generator, *, in_dim: int, hidden: int = b3.HIDDEN,
          k_range=None, t_k=293.0, weight_dtype=torch.float32, brain: bool = True, device="cpu") -> Bodies:
    """The founders of A arenas (PLANET-V3-SPEC 6): F per arena in slots 0..F-1 of N, each with a random genome
    (brain3.random_genome3: small random weights, g = 0, eta at its floor, body genes drawn broadly), at a uniformly
    random position with a uniformly random heading, at its size with FOUNDER_FAT_PER_LEAN of fat (half a version-2
    store, the same for every genome) and normal water, at body temperature ``t_k`` (a number, [A] or [A, F]). Draws
    (fixed order): brain3's genome, this module's EXTRA_SPECS genes but FOUNDER_LATE_GENES, positions, headings,
    FOUNDER_LATE_GENES (air_l_kg, o2_carrier). The founders' stocks are booked as ``*_founded``. ``brain`` False
    leaves the live weights and hidden state out. It is :func:`empty_genome` and :func:`found_into` over all arenas."""
    if not 0 < F <= N:
        raise ValueError("founders F must satisfy 0 < F <= N")
    genome = empty_genome(A, N, in_dim, hidden, weight_dtype, device)
    wh_live = torch.zeros(A, N, hidden, hidden, dtype=torch.float32, device=device) if brain else None
    hid = b3.init_state((A, N), hidden, device) if brain else None
    b = empty(A, N, L, genome=genome, wh_live=wh_live, hidden=hid, device=device)
    found_into(b, 0, A, F, gen, in_dim=in_dim, hidden=hidden, k_range=k_range, t_k=t_k)
    reset_ledgers(b)
    return b


# =========================================================================================== locomotion
def _speed(k1, k3, p, k_int=None, e_int: float = FEDAK_INTERNAL[1]) -> torch.Tensor:
    """The speed v >= 0 with k3 v^3 + k_int v^e_int + k1 v = p (float64 Newton from above; every term is convex and
    increasing, so it converges monotonically from the smallest one-term solution)."""
    k1, k3, p = k1.double(), k3.double().clamp_min(1e-300), p.double().clamp_min(0)
    v = (p / k3) ** (1.0 / 3.0)
    v = torch.where(k1 > 0, torch.minimum(v, p / k1.clamp_min(1e-300)), v)
    if k_int is None:                                    # no internal-work term: the cubic alone
        for _ in range(40):
            v2 = v * v
            v = (v - (k3 * v2 * v + k1 * v - p) / (3 * k3 * v2 + k1).clamp_min(1e-300)).clamp_min(0)
        return v
    ki = torch.as_tensor(k_int).double().expand_as(p)
    v = torch.where(ki > 0, torch.minimum(v, (p / ki.clamp_min(1e-300)) ** (1.0 / e_int)), v)
    for _ in range(40):
        f = k3 * v ** 3 + ki * v ** e_int + k1 * v - p
        df = 3 * k3 * v ** 2 + e_int * ki * v.clamp_min(1e-300) ** (e_int - 1) + k1
        v = (v - f / df.clamp_min(1e-300)).clamp_min(0)
    return v


def _water_depth(geom, pstate, pos) -> torch.Tensor:
    """Water depth (m) at positions [A, ..., 2]: the sea (deep: infinite) or the pond."""
    sea = pt.sample(geom, geom.sea.float(), pos) > 0.5
    pond = torch.zeros_like(sea, dtype=torch.float32) if pstate is None else \
        pt.sample(geom, pstate.pond, pos) / RHO_WATER
    return torch.where(sea, torch.full_like(pond, float("inf")), pond)


def _surface_m(geom, pstate, pos) -> torch.Tensor:
    """Height (m) of the surface a body stands or floats on: ground + pond, or the sea level."""
    z = pt.bilinear(geom, geom.elev, pos)
    sea = pt.sample(geom, geom.sea.float(), pos) > 0.5
    pond = 0.0 if pstate is None else pt.sample(geom, pstate.pond, pos) / RHO_WATER
    lvl = geom.sea_level_m.reshape((-1,) + (1,) * (z.dim() - 1)).expand_as(z)
    return torch.where(sea, lvl, z + pond)


def water_density(geom, pos) -> torch.Tensor:
    """The density of the water at positions [A, ..., 2] (kg/m^3): sea water (RHO_SEA) on sea cells, fresh water
    (RHO_WATER, the ponds) elsewhere."""
    sea = pt.sample(geom, geom.sea.float(), pos) > 0.5
    return torch.where(sea, torch.full_like(sea, RHO_SEA, dtype=torch.float32),
                       torch.full_like(sea, RHO_WATER, dtype=torch.float32))


def tread_power_w(b: Bodies, sh: dict | None = None, rho=None, rho_w=RHO_WATER, gravity=K.G_STANDARD,
                  tis: dict | None = None) -> torch.Tensor:
    """The metabolic power (W) [A, N] a body denser than the water needs to hold its airway above it
    (``PROVENANCE['tread']``): its weight in water F = (rho - rho_w) / rho x m g at the actuator-disc power F^1.5 /
    sqrt(2 rho_w A_x), over PADDLE_EFF and MUSCLE_EFF; 0 for a body that floats."""
    sh = shape(b) if sh is None else sh
    rho = density(b, tis) if rho is None else rho
    rw = torch.as_tensor(rho_w, dtype=torch.float32, device=b.device)
    g = torch.as_tensor(gravity, dtype=torch.float32, device=b.device)
    force = ((rho - rw) / rho.clamp_min(1e-9)).clamp_min(0) * sh["m"] * g
    ideal = force ** 1.5 / torch.sqrt(2 * rw * sh["frontal"].clamp_min(1e-12))
    return ideal / (PADDLE_EFF * MUSCLE_EFF) * b.alive


def submerged_share(b: Bodies, geom=None, pstate=None, pos=None, sh=None, rho=None) -> torch.Tensor:
    """Share of the body under water at its position (or ``pos`` [A, N, ...] with one body height each): water depth
    over the body's height, at most the share a floating body holds under water (its density over the water's at
    the position, :func:`water_density`; 1 for a body that sinks; ``rho`` = :func:`density`)."""
    if geom is None:
        return torch.zeros_like(b.mass_kg)
    sh = shape(b) if sh is None else sh
    pos = b.pos if pos is None else pos
    depth = _water_depth(geom, pstate, pos)
    trail = (1,) * (depth.dim() - 2)
    height = (2 * sh["b"]).reshape(sh["b"].shape + trail)
    rho = density(b) if rho is None else rho
    floats = (rho.reshape(rho.shape + trail) / water_density(geom, pos)).clamp(0, 1)
    return torch.minimum(depth / height.clamp_min(1e-9), floats).clamp(0, 1)


def sunk(b: Bodies, geom=None, pstate=None, pos=None, sh=None, rho=None, treads=None) -> torch.Tensor:
    """[A, N] (or [A, N, ...] at ``pos``) bool: the airway is under water. A body denser than the water at its
    position (:func:`density`, with the air it holds; :func:`water_density`) in water deeper than its height sinks,
    unless it treads water (``treads`` [A, N] bool: it moves with the power :func:`tread_power_w` needs); it lives on
    its O2 store (``PROVENANCE['breath_hold']``)."""
    if geom is None:
        return torch.zeros_like(b.alive)
    sh = shape(b) if sh is None else sh
    pos = b.pos if pos is None else pos
    depth = _water_depth(geom, pstate, pos)
    trail = (1,) * (depth.dim() - 2)
    height = (2 * sh["b"]).reshape(sh["b"].shape + trail)
    rho = density(b) if rho is None else rho
    out = (depth >= height) & (rho.reshape(rho.shape + trail) > water_density(geom, pos))
    if treads is not None:
        out = out & ~treads.reshape(treads.shape + trail)
    return out


def _land_terms(sh: dict, air: dict, g) -> tuple:
    """(k1, k3, k_int) of the land power balance P = k1 v + k_int v^1.53 + k3 v^3 [A, N]."""
    m = sh["m"]
    return C_M * m * g, 0.5 * air["rho"] * C_DRAG * sh["frontal"], FEDAK_INTERNAL[0] * m


def top_speed(b: Bodies, env: Env, geom=None) -> torch.Tensor:
    """Fastest land speed (m/s) [A, N] at the muscles' peak power (at T_b) against c_m M g v + the limbs' internal
    work + 1/2 rho C_d A_x v^3."""
    e = _env(b, env, geom)
    sh = shape(b)
    air = air_props(e["t_air_k"], e["p_air_pa"])
    k1, k3, ki = _land_terms(sh, air, e["gravity_m_s2"])
    return _speed(k1, k3, muscle_peak_w(b), ki).float()


def move(b: Bodies, turn, thrust, env: Env, *, geom=None, pstate=None, dt_s=None) -> dict:
    """Turn and move every living body for one bout, or for ``dt_s`` seconds of it (a world may sub-step movement
    within a bout, re-sensing between the parts, and add them with :func:`sum_loco`) (``PROVENANCE['locomotion']``).
    Returns per body [A, N]: work_j (mechanical), metabolic_j (work / MUSCLE_EFF), heat_j (metabolic less the work
    that leaves the body: drag in air and water, the climb's potential energy), dist_m, submerged (the path's share in
    water), climb_j, drag_j, sunk_s (the bout if the airway ends under water) and its dive structure (DIVE_KEYS),
    dt_s. The power is :func:`sustained_power_w`; a body denser than the water treads it while it moves with the
    power :func:`tread_power_w` needs (the rest propels it). Positions wrap on the periodic arena."""
    e = _env(b, env, geom)
    dt = float(e["dt"] if dt_s is None else dt_s)
    alive = b.alive
    turn = torch.as_tensor(turn, dtype=torch.float32, device=b.device).clamp(-1, 1) * alive
    thrust = torch.as_tensor(thrust, dtype=torch.float32, device=b.device).clamp(-1, 1) * alive
    b.heading = torch.remainder(b.heading + TURN_SPAN * turn, 2 * math.pi)
    sh = shape(b)
    tis = tissues(b, sh)
    rho = density(b, tis)
    p_sus = _sustained(b, e, tis, sh)
    power = (thrust.abs() * p_sus).double()
    m, g = sh["m"], e["gravity_m_s2"]
    air = air_props(e["t_air_k"], e["p_air_pa"])
    k1, k3_air, k_int = _land_terms(sh, air, g)
    v_land = _speed(k1, k3_air, power, k_int)
    # a body denser than the water treads it while it moves with the power that needs (PROVENANCE['tread'])
    rho_w = water_density(geom, b.pos) if geom is not None else torch.full_like(m, RHO_WATER)
    tread = (tread_power_w(b, sh, rho, rho_w, g) * MUSCLE_EFF).double()
    treads = (rho > rho_w) & (power > tread) & alive
    v_water = _speed(torch.zeros_like(m), 0.5 * RHO_WATER * C_DRAG * sh["frontal"],
                     torch.where(treads, power - tread, power))
    direction = torch.stack((torch.cos(b.heading), torch.sin(b.heading)), -1) * torch.sign(thrust)[..., None]
    if geom is not None:
        frac = (torch.arange(PATH_SAMPLES, device=b.device, dtype=torch.float32) + 0.5) / PATH_SAMPLES
        reach = (v_land * dt).float()
        pts = b.pos[..., None, :] + reach[..., None, None] * frac[:, None] * direction[..., None, :]
        s = submerged_share(b, geom, pstate, pts, sh, rho).mean(-1)
        # the path's share in water (its medium), not the body's floating share: a floater's path at sea is all water
        height = (2 * sh["b"])[..., None]
        wet = (_water_depth(geom, pstate, pts) / height.clamp_min(1e-9)).clamp(0, 1).mean(-1)
    else:
        s = wet = torch.zeros_like(m)
    s64 = wet.double()
    slow = (1 - s64) / v_land.clamp_min(1e-300) + s64 / v_water.clamp_min(1e-300)       # s per metre
    moving = (power > 0) & (v_land > 0) & (v_water > 0) & alive
    old = b.pos
    mg = m.double() * g.double()
    share = moving.double()                       # share of the bout spent travelling (the rest pays the climb)
    if geom is not None:
        # the climb is paid from the same power: along the path's samples the energy to reach the share f of the
        # level distance is P dt f + M g max(0, z(f) - z0); the body stops where it reaches P dt
        z0 = _surface_m(geom, pstate, old).double()
        fk = torch.arange(1, PATH_SAMPLES + 1, device=b.device, dtype=torch.float64) / PATH_SAMPLES
        reach = (dt / slow.clamp_min(1e-300)) * moving
        pts = old[..., None, :].double() + (reach[..., None] * fk)[..., None] * direction[..., None, :].double()
        zk = _surface_m(geom, pstate, torch.remainder(pts, b.L).float()).double()
        budget = (power * dt)[..., None]
        ek = budget * fk + (zk - z0[..., None]).clamp_min(0) * mg[..., None]
        e_prev = torch.cat((torch.zeros_like(ek[..., :1]), ek[..., :-1]), -1)
        f_prev = torch.cat((torch.zeros(1, dtype=fk.dtype, device=fk.device), fk[:-1]))
        over = ek > budget
        k = over.to(torch.int8).argmax(-1, keepdim=True)
        e1, e0 = ek.gather(-1, k)[..., 0], e_prev.gather(-1, k)[..., 0]
        f1, f0 = fk[k[..., 0]], f_prev[k[..., 0]]
        cross = f0 + (budget[..., 0] - e0) / (e1 - e0).clamp_min(1e-300) * (f1 - f0)
        share = torch.where(over.any(-1), cross.clamp(0, 1), share) * moving
    t_go = share * dt
    dist = t_go / slow.clamp_min(1e-300)
    b.pos = torch.remainder(old + dist.float()[..., None] * direction, b.L)
    if geom is not None:
        climb = (_surface_m(geom, pstate, b.pos).double() - z0).clamp_min(0) * mg * moving
    else:
        climb = torch.zeros_like(power)
    b.vel = (dist.float()[..., None] * direction / dt) * alive[..., None]
    t_water = dist * s64 / v_water.clamp_min(1e-300)
    t_land = dist * (1 - s64) / v_land.clamp_min(1e-300)
    work = power * t_go + climb
    metabolic = work / MUSCLE_EFF
    drag = power * t_water + k3_air.double() * v_land ** 3 * t_land                      # leaves into water and air
    heat = (metabolic - drag - climb).clamp_min(0)
    af = alive.float()
    under = sunk(b, geom, pstate, sh=sh, rho=rho, treads=treads).float() * dt if geom is not None \
        else torch.zeros_like(af)
    return {"work_j": work.float() * af, "metabolic_j": metabolic.float() * af, "heat_j": heat.float() * af,
            "dist_m": dist.float() * af, "submerged": s.float() * af, "climb_j": climb.float() * af,
            "drag_j": drag.float() * af, "sunk_s": under * af, "dt_s": dt, **dive_piece(under * af, dt)}


def sum_loco(parts: list) -> dict:
    """The bout's locomotion from the results of several :func:`move` calls (sub-steps): energies, distances and the
    time sunk add, the dives merge in order (:func:`dive_merge`), the submerged share is weighted by the time of
    each part; ``dt_s`` the sum."""
    out = {k: sum(p[k] for p in parts) for k in ("work_j", "metabolic_j", "heat_j", "dist_m", "climb_j", "drag_j")}
    if all("sunk_s" in p for p in parts):
        out["sunk_s"] = sum(p["sunk_s"] for p in parts)
    if all(k in p for p in parts for k in DIVE_KEYS):
        d = {k: parts[0][k] for k in DIVE_KEYS}
        for p in parts[1:]:
            d = dive_merge(d, p)
        out.update(d)
    total = sum(float(p["dt_s"]) for p in parts)
    out["submerged"] = sum(p["submerged"] * float(p["dt_s"]) for p in parts) / max(total, 1e-30)
    out["dt_s"] = total
    return out


# =========================================================================================== contact under the mouth
def mouth_position(b: Bodies, sh: dict | None = None) -> torch.Tensor:
    """The mouth at the front of the ellipsoid: pos + a (cos h, sin h), wrapped [A, N, 2]."""
    sh = shape(b) if sh is None else sh
    d = torch.stack((torch.cos(b.heading), torch.sin(b.heading)), -1)
    return torch.remainder(b.pos + sh["a"][..., None] * d, b.L)


def mouth_reach_m(b: Bodies, sh: dict | None = None) -> torch.Tensor:
    """How far from the mouth a surface is in contact (m) [A, N]: one radius of the equal-volume sphere (the rule
    manipulate.py and senses3.py use by default; :func:`sense_view` and :func:`manipulation_view` pass it)."""
    sh = shape(b) if sh is None else sh
    return sh["r1"] * b.alive


def bite_force_n(b: Bodies, tis: dict | None = None) -> torch.Tensor:
    """F = MUSCLE_STRESS x (jaw-muscle volume)^(2/3) x JAW_LEVER x (1 - damage) [A, N] (N). Isometric force hardly
    depends on temperature (Q10 1.0-1.3, Bennett 1985, J. Exp. Biol. 115, 333); the rate of jaw cycles does
    (:func:`ingest` runs them at the Arrhenius rate of T_b)."""
    tis = tissues(b) if tis is None else tis
    vol = JAW_SHARE * tis["muscle"] / RHO_TISSUE
    return MUSCLE_STRESS_PA * vol ** (2.0 / 3.0) * JAW_LEVER * (1 - b.damage).clamp(0, 1) * b.alive


def body_contact(b: Bodies, geom, sh=None, nbr=None, query=None) -> torch.Tensor:
    """The body under each mouth [A, N] (slot, -1 none): the nearest of the CONTACT_K nearest bodies (patch.neighbours,
    or ``nbr`` = its (idx, offset, ...) output) whose equal-volume sphere lies within the mouth's reach
    (:func:`mouth_reach_m`). ``query`` [A, N] bool: only these mouths are asked (others get -1); few of them are
    checked directly (BRUTE_PAIRS)."""
    sh = shape(b) if sh is None else sh
    A, N = b.shape
    if nbr is None and query is not None:
        qa, qn = query.nonzero(as_tuple=True)
        if qa.numel() * N <= BRUTE_PAIRS:
            return _body_contact_direct(b, sh, qa, qn)
    if nbr is None:
        nbr = pt.neighbours(geom, b.pos, b.alive, K=CONTACT_K)
    idx, off = nbr[0], nbr[1]
    d = torch.stack((torch.cos(b.heading), torch.sin(b.heading)), -1)
    mouth = sh["a"][..., None] * d
    dist = (off - mouth[..., None, :]).norm(dim=-1)
    safe = idx.clamp_min(0)
    r_t = sh["r1"].gather(1, safe.reshape(A, -1)).reshape(idx.shape)
    reach = mouth_reach_m(b, sh)[..., None]
    gap = dist - r_t
    ok = (idx >= 0) & (gap <= reach) & b.alive[..., None]
    gap = torch.where(ok, gap, torch.full_like(gap, float("inf")))
    best = gap.argmin(-1, keepdim=True)
    hit = ok.gather(-1, best)[..., 0]
    return torch.where(hit, idx.gather(-1, best)[..., 0], torch.full_like(hit, -1, dtype=torch.long))


def _body_contact_direct(b: Bodies, sh: dict, qa, qn) -> torch.Tensor:
    """body_contact for the mouths (qa, qn) [M], checked against every slot of their arena."""
    A, N = b.shape
    out = torch.full((A, N), -1, dtype=torch.long, device=b.device)
    if not qa.numel():
        return out
    me = b.pos[qa, qn]
    d_c = pt.wrap(b.pos[qa] - me[:, None, :], b.L).norm(dim=-1)                        # [M, N] centre distances
    valid = b.alive[qa] & (torch.arange(N, device=b.device)[None] != qn[:, None])
    d_c = torch.where(valid, d_c, torch.full_like(d_c, float("inf")))
    k = min(CONTACT_K, N)
    dk, ik = torch.topk(d_c, k, dim=-1, largest=False, sorted=True)
    head = torch.stack((torch.cos(b.heading[qa, qn]), torch.sin(b.heading[qa, qn])), -1)
    mouth = sh["a"][qa, qn][:, None] * head                                            # relative to the centre
    off = pt.wrap(b.pos[qa[:, None], ik] - me[:, None, :], b.L)
    gap = (off - mouth[:, None, :]).norm(dim=-1) - sh["r1"][qa[:, None], ik]
    ok = torch.isfinite(dk) & (gap <= mouth_reach_m(b, sh)[qa, qn][:, None]) & b.alive[qa, qn][:, None]
    gap = torch.where(ok, gap, torch.full_like(gap, float("inf")))
    best = gap.argmin(-1, keepdim=True)
    hit = ok.gather(-1, best)[:, 0]
    out[qa, qn] = torch.where(hit, ik.gather(-1, best)[:, 0], torch.full_like(qa, -1))
    return out


def _item_contact_direct(b: Bodies, geom, pool, sh: dict, props: dict, qa, qn) -> torch.Tensor:
    """item_contact for the mouths (qa, qn) [M], checked against every ground item of their arena."""
    A, N = b.shape
    out = torch.full((A, N), -1, dtype=torch.long, device=b.device)
    if not qa.numel() or pool.alive.shape[1] == 0:
        return out
    r_item = (3 * props["volume_m3"] / (4 * math.pi)) ** (1.0 / 3.0)
    ground = pool.alive & (pool.holder < 0)
    pm = mouth_position(b, sh)[qa, qn]
    ipos = torch.remainder(pool.pos[qa][..., :2].float(), geom.L)
    gap = pt.wrap(ipos - pm[:, None, :], geom.L).norm(dim=-1) - r_item[qa]
    reach = mouth_reach_m(b, sh)[qa, qn].clamp_max(geom.dx / 2)[:, None]
    ok = ground[qa] & (gap <= reach) & b.alive[qa, qn][:, None]
    gap = torch.where(ok, gap, torch.full_like(gap, float("inf")))
    best = gap.argmin(-1, keepdim=True)
    hit = ok.gather(-1, best)[:, 0]
    out[qa, qn] = torch.where(hit, best[:, 0], torch.full_like(qa, -1))
    return out


def item_contact(b: Bodies, geom, pool, sh=None, props=None, query=None) -> torch.Tensor:
    """The ground item under each mouth [A, N] (slot, -1 none): items on the ground (holder -1) bucketed by fine cell,
    the 2 x 2 cells nearest the mouth scanned (ITEM_SCAN each, stratified), the nearest whose sphere lies within the
    mouth's reach (:func:`mouth_reach_m`, clamped to half a fine cell). ``pool`` holds the arenas' items with patch
    coordinates in pos[..., :2] (metres). ``query`` [A, N] bool: only these mouths are asked; few of them are checked
    directly against every ground item (BRUTE_PAIRS)."""
    sh = shape(b) if sh is None else sh
    A, N = b.shape
    I = pool.alive.shape[1]
    n, dx, L = geom.n, geom.dx, geom.L
    dev = b.device
    props = mt.props(pool.comp, pool.mass * pool.alive) if props is None else props
    if query is not None:
        qa, qn = query.nonzero(as_tuple=True)
        if qa.numel() * max(I, 1) <= BRUTE_PAIRS:
            return _item_contact_direct(b, geom, pool, sh, props, qa, qn)
    r_item = (3 * props["volume_m3"] / (4 * math.pi)) ** (1.0 / 3.0)
    ground = pool.alive & (pool.holder < 0)
    ipos = torch.remainder(pool.pos[..., :2].float(), L)
    ic = torch.floor(ipos / dx).long().clamp(0, n - 1)
    ar = torch.arange(A, device=dev)[:, None]
    sentinel = A * n * n
    key = torch.where(ground, ar * n * n + ic[..., 0] * n + ic[..., 1], torch.full_like(ic[..., 0], sentinel))
    sk, order = torch.sort(key.reshape(-1), stable=True)
    pm = mouth_position(b, sh)
    c = torch.floor(pm / dx).long().clamp(0, n - 1)
    frac = pm / dx - c.float()
    ox = torch.where(frac[..., 0] < 0.5, -1, 1)
    oy = torch.where(frac[..., 1] < 0.5, -1, 1)
    zero = torch.zeros_like(ox)
    offs = torch.stack((torch.stack((zero, zero), -1), torch.stack((ox, zero), -1), torch.stack((zero, oy), -1),
                        torch.stack((ox, oy), -1)), -2)                                   # [A, N, 4, 2]
    q = torch.remainder(c[..., None, :] + offs, n)
    qkey = ar[..., None] * n * n + q[..., 0] * n + q[..., 1]                              # [A, N, 4]
    start = torch.searchsorted(sk, qkey.reshape(-1)).reshape(A, N, 4)
    end = torch.searchsorted(sk, qkey.reshape(-1), right=True).reshape(A, N, 4)
    cnt = end - start
    j = torch.arange(ITEM_SCAN, device=dev)
    pick = torch.where(cnt[..., None] <= ITEM_SCAN, j.expand(A, N, 4, ITEM_SCAN),
                       torch.floor((j.double() + 0.5) * cnt[..., None].double() / ITEM_SCAN).long())
    valid = j < torch.minimum(cnt, torch.full_like(cnt, ITEM_SCAN))[..., None]
    p = (start[..., None] + pick).clamp(0, max(sk.numel() - 1, 0))
    flat = order[p] if sk.numel() else torch.zeros_like(p)                               # a I + i
    valid = valid & (sk[p] < sentinel) if sk.numel() else valid & False
    other = ipos.reshape(-1, 2)[flat]
    dist = pt.wrap(other - pm[..., None, None, :], L).norm(dim=-1)
    rad = r_item.reshape(-1)[flat]
    reach = mouth_reach_m(b, sh).clamp_max(dx / 2)[..., None, None]
    gap = dist - rad
    ok = valid & (gap <= reach) & b.alive[..., None, None]
    gap = torch.where(ok, gap, torch.full_like(gap, float("inf"))).reshape(A, N, -1)
    best = gap.argmin(-1, keepdim=True)
    hit = ok.reshape(A, N, -1).gather(-1, best)[..., 0]
    slot = (flat % max(I, 1)).reshape(A, N, -1).gather(-1, best)[..., 0]
    return torch.where(hit, slot, torch.full_like(slot, -1))


# =========================================================================================== exchanges with the patch
def _residual(b: Bodies, geom, key: str) -> torch.Tensor:
    """The float64 residual [A, n*n] (kg per fine cell) of one exchange (:data:`TRANSIT_KEYS`), created at zero."""
    if key not in TRANSIT_KEYS:
        raise KeyError(key)
    if b.transit is None:
        b.transit = {}
    want = (int(geom.A), int(geom.n) * int(geom.n))
    r = b.transit.get(key)
    if r is None:
        r = torch.zeros(want, dtype=torch.float64, device=b.device)
        b.transit[key] = r
    elif tuple(r.shape) != want:
        raise ValueError(f"transit {key} has shape {tuple(r.shape)}, the patch {want}")
    return r


def _settle(b: Bodies, pstate, geom, key: str, prules=None) -> torch.Tensor:
    """Move the residual of ``key`` into its float32 patch field as far as float32 resolves it (cells whose residual
    is 0 are untouched); the patch books what its field moved, as its own helpers do (soil given: w_given; soil and
    pond taken: w_taken; plant taken: c_taken and its nitrogen n_removed; litter given: c_added; nitrogen given:
    n_added). What an emptied pond still owes passes to its cell's soil. Returns the amount moved per arena [A]
    (float64, kg)."""
    field, sign = TRANSIT_KEYS[key]
    R = _residual(b, geom, key)
    fld = getattr(pstate, field)
    flat = fld.reshape(R.shape)
    before = flat.double()
    target = (before + sign * R / geom.cell_m2).clamp_min(0)
    after = torch.where(R != 0, target.float(), flat)
    moved = sign * (after.double() - before) * geom.cell_m2
    R.sub_(moved)
    setattr(pstate, field, after.view_as(fld))
    if key == "pond_out":
        # a pond that is gone (dried or drunk) hands what it still owes to its cell's soil, the store it drained into
        gone = (after == 0) & (R != 0)
        if bool(gone.any()):
            _residual(b, geom, "soil_out").add_(torch.where(gone, R, torch.zeros_like(R)))
            R.masked_fill_(gone, 0.0)
            _settle(b, pstate, geom, "soil_out", prules)
    tot = moved.sum(-1)
    if key == "soil_in":
        pstate.w_given = pstate.w_given + tot
    elif key in ("soil_out", "pond_out"):
        pstate.w_taken = pstate.w_taken + tot
    elif key == "plant_out":
        pstate.c_taken = pstate.c_taken + tot
        pstate.n_removed = pstate.n_removed + tot / (prules or pt.PatchRules()).bio.cn_leaf
    elif key == "litter_in":
        pstate.c_added = pstate.c_added + tot
    else:
        pstate.n_added = pstate.n_added + tot
    return tot


def _stock(b: Bodies, pstate, geom, field: str) -> torch.Tensor:
    """What a patch field holds per fine cell as the bodies see it [A * n * n] (float64, kg per cell): the field plus
    the residuals given into it, less those taken from it."""
    x = getattr(pstate, field).reshape(-1).double() * geom.cell_m2
    for key, (f, sign) in TRANSIT_KEYS.items():
        if f == field and b.transit is not None and key in b.transit:
            x = x + sign * b.transit[key].reshape(-1)
    return x.clamp_min(0)


def _share(idx, req, avail, size: int) -> torch.Tensor:
    """Requests req [M] (float64 >= 0) on flat cells idx [M] share the cells' avail [size] in proportion when they ask
    for more than is there. Returns what each request gets [M]."""
    total = torch.zeros(size, dtype=torch.float64, device=req.device).index_add_(0, idx, req)
    return req * (avail[idx] / total[idx].clamp_min(1e-300)).clamp(max=1.0)


def _arena(a, x, A: int) -> torch.Tensor:
    return torch.zeros(A, dtype=torch.float64, device=x.device).index_add_(0, a, x.double())


def _give_water(b: Bodies, pstate, geom, a, cell, water_kg, salt_kg=None) -> None:
    """Water a body returns (urine, faeces water, a dead body's water) with the salt it carries, per request (arena a
    [M], flat fine cell [M]): on land into the cell's soil through the residual ``soil_in`` (the salt stays on the
    land, ``salt_land``: the patch has no salt field); on a sea cell into the sea, water and salt (``sea_water``, the
    global ocean's, and the bodies' ``sea_out_kg`` and ``salt_sea``)."""
    nn = geom.n * geom.n
    idx = a * nn + cell
    sea = geom.sea.reshape(-1)[idx]
    w = torch.as_tensor(water_kg, device=b.device).double().reshape(-1).clamp_min(0)
    s = torch.zeros_like(w) if salt_kg is None else torch.as_tensor(salt_kg, device=b.device).double().reshape(-1)
    zero = torch.zeros_like(w)
    _residual(b, geom, "soil_in").view(-1).index_add_(0, idx, torch.where(sea, zero, w))
    to_sea = _arena(a, torch.where(sea, w + s, zero), geom.A)
    pstate.sea_water = pstate.sea_water - to_sea
    b.ledger["sea_out_kg"] += to_sea
    b.ledger["salt_sea"] += _arena(a, torch.where(sea, s, zero), geom.A)
    b.ledger["salt_land"] += _arena(a, torch.where(sea, zero, s), geom.A)
    _settle(b, pstate, geom, "soil_in")


def _give_litter(b: Bodies, pstate, geom, a, cell, kg_c, kg_n, prules=None) -> None:
    """Organic carbon (faeces, urea, carcass remains) into the cell's litter and its nitrogen into the cell's mineral
    pool (as patch.add_litter), through the residuals ``litter_in`` and ``nutrient_in``."""
    idx = a * geom.n * geom.n + cell
    _residual(b, geom, "litter_in").view(-1).index_add_(0, idx, torch.as_tensor(kg_c).double().reshape(-1)
                                                        .clamp_min(0))
    _residual(b, geom, "nutrient_in").view(-1).index_add_(0, idx, torch.as_tensor(kg_n).double().reshape(-1)
                                                          .clamp_min(0))
    _settle(b, pstate, geom, "litter_in", prules)
    _settle(b, pstate, geom, "nutrient_in", prules)


def settle_transit(b: Bodies, pstate, geom, prules=None) -> dict:
    """Settle every residual into its field (e.g. after the patch's daily step changed the fields); returns the
    amounts moved per key [A]. The residuals stay within half a float32 step of their fields per cell."""
    return {key: _settle(b, pstate, geom, key, prules) for key in TRANSIT_KEYS if b.transit and key in b.transit}


def transit_kg(b: Bodies) -> dict:
    """Each residual's total per arena [A] (float64, kg; kg C for plant and litter, kg N for nutrient)."""
    return {key: v.sum(-1) for key, v in (b.transit or {}).items()}


_SPECIES_C = SPECIES_GUT[:, :, Q_C].sum(-1)
_SPECIES_N = SPECIES_GUT[:, :, Q_N].sum(-1)
_PATCH_COUNTERS = ("w_taken", "w_given", "sea_water", "c_taken", "c_added", "n_removed", "n_added")


def exchange_snapshot(b: Bodies, pstate) -> dict:
    """The bodies' ledger, the patch's counters and the residuals now, for :func:`exchange_errors`."""
    return {"ledger": {k: v.clone() for k, v in b.ledger.items()},
            "patch": {k: getattr(pstate, k).clone() for k in _PATCH_COUNTERS},
            "transit": {k: v.clone() for k, v in transit_kg(b).items()}}


def exchange_errors(b: Bodies, pstate, geom, snap: dict, prules=None) -> dict:
    """Bodies' books minus the patch's (+ the change of the residuals in transit) since ``snap``, per arena [A]
    (float64), each with its ``_scale``: ``water_in`` (fresh and plant water taken), ``water_out`` (urine, faeces
    water and dead bodies' water not held by carcass items, on land), ``sea`` (sea water drunk less returned),
    ``carbon_in`` / ``nitrogen_in`` (plant tissue), ``carbon_out`` / ``nitrogen_out`` (excreta and carcass remains
    not held by items), ``salt`` (excreted and dead salt against where it went). Every body exchange with the patch
    must go through this module while the books run."""
    cn = (prules or pt.PatchRules()).bio.cn_leaf
    L0, P0, T0 = snap["ledger"], snap["patch"], snap["transit"]
    T1 = transit_kg(b)

    def dl(k):
        return b.ledger[k] - L0[k]

    def dp(k):
        return getattr(pstate, k) - P0[k]

    def dt(k):
        z = torch.zeros(b.shape[0], dtype=torch.float64, device=b.device)
        return T1.get(k, z) - T0.get(k, z)

    dev = b.device
    eaten, carc = dl("items_eaten_kg"), dl("carcass_kg")
    sw, sc, sn = SPECIES_WATER.to(dev), _SPECIES_C.to(dev), _SPECIES_N.to(dev)
    pairs = {
        "water_in": (dl("w_drunk") + dl("w_food") - eaten @ sw - (dl("sea_in_kg") * (1 - SEA_SALINITY)),
                     dp("w_taken") + dt("soil_out") + dt("pond_out")),
        "water_out": (dl("w_urine") + dl("w_faeces") + dl("w_dead") - carc @ sw
                      - (dl("sea_out_kg") - dl("salt_sea")), dp("w_given") + dt("soil_in")),
        "sea": (dl("sea_in_kg") - dl("sea_out_kg"), dp("sea_water")),
        "carbon_in": (dl("c_food") - eaten @ sc, dp("c_taken") + dt("plant_out")),
        "carbon_out": (dl("c_faeces") + dl("c_urine") + dl("c_dead") - carc @ sc, dp("c_added") + dt("litter_in")),
        "nitrogen_in": (dl("n_food") - eaten @ sn, dp("n_removed") + dt("plant_out") / cn),
        "nitrogen_out": (dl("n_urine") + dl("n_faeces") + dl("n_dead") - carc @ sn,
                         dp("n_added") + dt("nutrient_in")),
        "salt": (dl("salt_out") + dl("salt_dead"), dl("salt_land") + dl("salt_sea")),
    }
    out = {}
    for k, (body_side, patch_side) in pairs.items():
        out[k] = body_side - patch_side
        out[f"{k}_scale"] = torch.maximum(body_side.abs(), patch_side.abs())
    return out


PROVENANCE["exchange_books"] = (N_, "exchange_errors: what the bodies booked as taken from and given to the patch "
                                    "equals what the patch's counters booked plus the change of the residuals in "
                                    "transit (water, sea water, carbon, nitrogen, salt)")


# =========================================================================================== ingestion
def take_item_mass(pool, a, i, kg) -> torch.Tensor:
    """Take up to ``kg`` [M] (float64) from items (arena a [M], slot i [M]) in proportion to their species; requests on
    one item share it in proportion to what they ask, never more than it holds (what the float32 pool really lost is
    what is given); an item eaten to nothing is removed. Returns the species taken [M, S] float64. ``ingest`` takes
    another function of this signature as ``take_items`` (e.g. ``manipulate.remove_mass`` with its item ledger)."""
    dev = pool.alive.device
    I = pool.alive.shape[1]
    a = torch.as_tensor(a, device=dev).long().reshape(-1)
    i = torch.as_tensor(i, device=dev).long().reshape(-1)
    req = torch.as_tensor(kg, dtype=torch.float64, device=dev).reshape(-1).clamp_min(0)
    comp = pool.comp[a, i].double()
    comp = comp / comp.sum(-1, keepdim=True).clamp_min(1e-30)
    flat = a * I + i
    total = torch.zeros(pool.alive.numel(), dtype=torch.float64, device=dev).index_add_(0, flat, req)
    have = (pool.mass.double() * pool.alive).reshape(-1)
    left = torch.where(total >= have, torch.zeros_like(have), have - total).float()
    removed = torch.where(total > 0, have - left.double(), torch.zeros_like(have))
    taken = req / total[flat].clamp_min(1e-300) * removed[flat]
    asked = (total > 0).view_as(pool.mass)
    pool.mass.copy_(torch.where(asked, left.view_as(pool.mass), pool.mass))
    gone = asked & pool.alive & (left.view_as(pool.mass) <= 0)
    if bool(gone.any()):
        gw, gi = gone.nonzero(as_tuple=True)
        itm.remove(pool, gw, gi)
    return comp * taken[:, None]


def spawn_carcass(pool, a, comp, mass, pos, temp_k) -> torch.Tensor:
    """Default carcass placement: ``items.spawn`` on the ground (holder -1). ``deaths`` takes another function of this
    signature as ``spawn_items`` (e.g. ``manipulate.spawn_items`` with its item ledger). Returns the slots [M]."""
    return itm.spawn(pool, a, comp, mass, pos, temp_k, -1)


def ingest(b: Bodies, mouth, env: Env, *, geom=None, pstate=None, pool=None, nbr=None, prules=None,
           take_items=None, dt_body=None, disc_share=None, settle: bool = True) -> dict:
    """The mouth closes on what it touches for the bout (see the module docstring); ``mouth`` [A, N] in [0, 1] is the
    share of the bout the jaws work, and jaw cycles run at the Arrhenius rate of the body's temperature. Contact
    order: a body, else a ground item, else the fine cell (plant tissue from the area the mouth sweeps this bout, and
    water). Food enters the gut by class (water straight into the body), bites wound. Items are taken through
    ``take_items`` (default :func:`take_item_mass`); plant tissue, plant water and drinking water through the transit
    residuals into the patch's fields. Returns per body [A, N]: contact (0 none, 1 ground, 2 item, 3 body), plant_kg
    (dry), drunk_kg, sea_kg, item_kg, bitten_kg (tissue the body removed from another), bite_work_j, wound (damage
    added to this body), and item / target slots.

    A world that sub-steps the bout calls ingest once per sub-step: ``dt_body`` [A, N] is each body's own sub-step
    (default env.dt_s for all), and ``disc_share`` (a number or [A, N], default 1) the share of the still mouth's
    disc pi reach^2 this call may crop: the disc is the area a still mouth reaches once in a bout, so the sub-steps of
    one bout pass shares that add up to 1 (the strip a moving mouth sweeps is its path, which the sub-steps split by
    themselves). ``settle`` False skips settling the transit residuals first (needed once after the patch's day)."""
    A, N = b.shape
    dev = b.device
    e = _env(b, env, geom)
    dt = e["dt"]
    dtb = None if dt_body is None else torch.as_tensor(dt_body, dtype=torch.float64, device=dev).expand(A, N)
    disc = torch.ones(A, N, dtype=torch.float64, device=dev) if disc_share is None else \
        torch.as_tensor(disc_share, dtype=torch.float64, device=dev).expand(A, N)

    def dt_of(ai, ni, like):
        return torch.full_like(like, dt) if dtb is None else dtb[ai, ni]

    pr = prules or pt.PatchRules()
    if settle and pstate is not None and geom is not None and b.transit:
        settle_transit(b, pstate, geom, pr)                 # the patch's day may have moved its fields
    m = torch.as_tensor(mouth, dtype=torch.float32, device=dev).clamp(0, 1) * b.alive
    sh = shape(b)
    tis = tissues(b, sh)
    free = (gut_capacity_kg(b, tis) - b.gut[..., Q_KG].sum(-1)).clamp_min(0).double()
    force = bite_force_n(b, tis).double()
    cycles = (arrhenius(b.body_k) / BITE_CYCLE_S).double()                 # jaw cycles per second at T_b
    gape = (GAPE_SHARE * sh["length"]).double()
    whole = RHO_TISSUE * math.pi / 6 * gape ** 3
    zeros = torch.zeros(A, N, dtype=torch.float64, device=dev)
    out = {"contact": torch.zeros(A, N, dtype=torch.long, device=dev), "plant_kg": zeros.clone(),
           "drunk_kg": zeros.clone(), "sea_kg": zeros.clone(), "item_kg": zeros.clone(), "bitten_kg": zeros.clone(),
           "bite_work_j": zeros.clone(), "wound": zeros.clone(),
           "target": torch.full((A, N), -1, dtype=torch.long, device=dev),
           "item": torch.full((A, N), -1, dtype=torch.long, device=dev)}
    add = torch.zeros(A, N, len(GUT_CLASSES), len(GUT_Q), dtype=torch.float64, device=dev)
    water_in = zeros.clone()
    active = m > 0
    # ---------------------------------------------------------------- bodies
    target = torch.full((A, N), -1, dtype=torch.long, device=dev)
    if geom is not None and bool(active.any()):
        target = torch.where(active, body_contact(b, geom, sh, nbr, query=active), target)
    bit = target >= 0
    if bool(bit.any()):
        ai, ni = bit.nonzero(as_tuple=True)
        tj = target[ai, ni]
        lean_t = b.mass_kg[ai, tj].double()
        fat_t = fat_kg(b)[ai, tj].double()
        wat_t = b.water_kg[ai, tj].double().clamp_min(0)
        body_t = (lean_t + fat_t + (wat_t - LEAN_WATER * lean_t).clamp_min(0)).clamp_min(1e-30)
        dry_t = lean_t * (1 - LEAN_WATER) + fat_t
        v_rel = (b.vel[ai, ni] - b.vel[ai, tj]).norm(dim=-1).double().clamp_min(V_CONTACT_FLOOR)
        g = gape[ai, ni]
        t_c = torch.minimum(sh["r1"][ai, ni].double() / v_rel, dt_of(ai, ni, v_rel))
        n_b = m[ai, ni].double() * t_c * cycles[ai, ni]
        q = (force[ai, ni] / (HIDE_TOUGHNESS_J_M2 * g).clamp_min(1e-30)).clamp(0, 1)
        chunk = whole[ai, ni] * q
        swallow = (sh["m"][ai, tj].double() <= whole[ai, ni]) & (n_b >= 1)
        req = torch.where(swallow, body_t, (n_b * chunk).clamp_max(body_t))
        req = torch.minimum(req, free[ai, ni] * body_t / dry_t.clamp_min(1e-30))
        phi = req / body_t                                                     # share of the target's tissue
        flat_t = ai * N + tj
        total = torch.zeros(A * N, dtype=torch.float64, device=dev).index_add_(0, flat_t, phi)
        phi = phi / total[flat_t].clamp_min(1.0)                               # biters of one target share it
        req = phi * body_t                                                     # what each really takes
        tot_t = torch.zeros(A * N, dtype=torch.float64, device=dev).index_add_(0, flat_t, phi).view(A, N)
        lean_cut = phi * lean_t
        fat_cut = phi * fat_t
        wat_cut = phi * wat_t
        gain = lean_cut[:, None, None] * _lean_row(dev) + fat_cut[:, None, None] * _fat_row(dev)
        add[ai, ni] += gain                    # the target's bone mineral moves into the biter's gut (inert)
        water_in[ai, ni] += wat_cut
        # the target loses the same shares (computed once per target from its own state)
        keep = (1 - tot_t).clamp_min(0)
        hit_t = tot_t > 0
        b.mass_kg = torch.where(hit_t, (b.mass_kg.double() * keep).float(), b.mass_kg)
        b.reserve_j = torch.where(hit_t, (b.reserve_j.double() * keep).float(), b.reserve_j)
        b.water_kg = torch.where(hit_t, (b.water_kg.double() * keep).float(), b.water_kg)
        wound = tot_t / WOUND_LETHAL_SHARE
        b.damage = b.damage + wound.float()
        out["wound"] = wound
        n_used = torch.where(chunk > 0, torch.minimum(n_b, req / chunk.clamp_min(1e-30)), n_b)
        n_used = torch.where(swallow, torch.ones_like(n_used), n_used)
        out["bite_work_j"][ai, ni] = n_used * force[ai, ni] * g
        out["bitten_kg"][ai, ni] = req
        out["contact"][ai, ni] = 3
        out["target"][ai, ni] = tj
        free[ai, ni] = (free[ai, ni] - gain[:, :, Q_KG].sum(-1)).clamp_min(0)
    # ---------------------------------------------------------------- items
    rest = active & ~bit
    if pool is not None and geom is not None and bool(rest.any()):
        props = mt.props(pool.comp, pool.mass * pool.alive)
        slot = torch.where(rest, item_contact(b, geom, pool, sh, props, query=rest), torch.full_like(target, -1))
        got = slot >= 0
        if bool(got.any()):
            ai, ni = got.nonzero(as_tuple=True)
            si = slot[ai, ni]
            comp = pool.comp[ai, si].double()
            comp = comp / comp.sum(-1, keepdim=True).clamp_min(1e-30)
            mass = pool.mass[ai, si].double()
            rho = props["density"][ai, si].double().clamp_min(1e-30)
            tough = BITE_TOUGHNESS_J_M2.to(dev)
            hard = (comp >= float(mt.CONTINUOUS_FRACTION)) & torch.isinf(tough)
            gamma = torch.where(hard.any(-1), torch.full_like(mass, float("inf")),
                                (comp * torch.where(torch.isinf(tough), torch.zeros_like(tough), tough)).sum(-1))
            g = gape[ai, ni]
            n_b = m[ai, ni].double() * dt_of(ai, ni, mass) * cycles[ai, ni]
            q = torch.where(torch.isinf(gamma), torch.zeros_like(gamma),
                            (force[ai, ni] / (gamma * g).clamp_min(1e-30)).clamp(0, 1))
            q = torch.where(gamma <= 0, torch.ones_like(q), q)
            cap = rho * math.pi / 6 * g ** 3
            swallow = (mass <= cap) & (n_b >= 1)
            req = torch.where(swallow, mass, (n_b * cap * q).clamp_max(mass))
            dry = (1 - comp @ SPECIES_WATER.to(dev)).clamp_min(1e-12)
            req = torch.minimum(req, free[ai, ni] / dry)
            cooked = pool.peak_k[ai, si] >= COOKED_K                      # read before the item may be removed
            species = (take_items or take_item_mass)(pool, ai, si, req)    # [M, S] kg actually taken
            taken = species.sum(-1)
            table = torch.where(cooked[:, None, None, None], SPECIES_GUT_COOKED.to(dev)[None],
                                SPECIES_GUT.to(dev)[None])                         # [M, S, G, Q]
            gain = (species[:, :, None, None] * table).sum(1)
            wat = species @ SPECIES_WATER.to(dev)
            add[ai, ni] += gain
            water_in[ai, ni] += wat
            b.ledger["items_eaten_kg"].index_add_(0, ai, species)
            _book_rows(b, "e_food", ai, gain[:, :, Q_J].sum(-1))
            _book_rows(b, "c_food", ai, gain[:, :, Q_C].sum(-1))
            _book_rows(b, "n_food", ai, gain[:, :, Q_N].sum(-1))
            _book_rows(b, "i_food", ai, gain[:, G_INERT, Q_KG])
            _book_rows(b, "w_food", ai, wat)
            n_used = torch.where(swallow, torch.ones_like(n_b),
                                 torch.where(cap * q > 0, torch.minimum(n_b, taken / (cap * q).clamp_min(1e-30)), n_b))
            out["bite_work_j"][ai, ni] += n_used * force[ai, ni] * g
            out["item_kg"][ai, ni] = taken
            out["contact"][ai, ni] = 2
            out["item"][ai, ni] = si
            free[ai, ni] = (free[ai, ni] - gain[:, :, Q_KG].sum(-1)).clamp_min(0)
            rest = rest & ~got
    # ---------------------------------------------------------------- the ground: plant tissue and water
    if pstate is not None and geom is not None and bool(rest.any()):
        ai, ni = rest.nonzero(as_tuple=True)
        cells = geom.A * geom.n * geom.n
        cell = pt.cell_index(geom, mouth_position(b, sh))[ai, ni]
        idx = ai * geom.n * geom.n + cell
        cf = pr.bio.plant_carbon_fraction
        mm = m[ai, ni].double()
        # plant soft tissue: the cell's density over the area the mouth sweeps in the bout (a stationary mouth crops
        # the disc it reaches, once a bout: disc_share), at most the cell; requests on one cell share what it holds
        stock_c = _stock(b, pstate, geom, "plant")
        reach = mouth_reach_m(b, sh)[ai, ni].double()
        dt_i = dt_of(ai, ni, reach)
        path = b.speed[ai, ni].double() * dt_i
        area = (2 * reach * path + disc[ai, ni] * math.pi * reach ** 2).clamp(max=geom.cell_m2)
        avail = stock_c[idx] / geom.cell_m2 / cf * area
        want = mm * torch.minimum(avail, free[ai, ni])
        got_c = _share(idx, want * cf, stock_c, cells)
        _residual(b, geom, "plant_out").view(-1).index_add_(0, idx, got_c)
        _settle(b, pstate, geom, "plant_out", pr)
        dry = got_c / cf
        gain = dry[:, None, None] * _plant_row(pr, dev)
        add[ai, ni] += gain
        # the plant's water comes from the cell's soil (never more than is there)
        pw = _share(idx, dry * PLANT_WATER_PER_DRY, _stock(b, pstate, geom, "soil"), cells)
        _residual(b, geom, "soil_out").view(-1).index_add_(0, idx, pw)
        water_in[ai, ni] += pw
        _book_rows(b, "e_food", ai, gain[:, :, Q_J].sum(-1))
        _book_rows(b, "c_food", ai, gain[:, :, Q_C].sum(-1))
        _book_rows(b, "n_food", ai, gain[:, :, Q_N].sum(-1))
        _book_rows(b, "w_food", ai, pw)
        # drinking: fresh water (pond, then soil above the bucket) or sea water, at the gut's emptying rate
        sea = geom.sea.reshape(-1)[idx]
        pond = _stock(b, pstate, geom, "pond")
        above = (_stock(b, pstate, geom, "soil") - pr.climate.bucket_kg_m2 * geom.cell_m2).clamp_min(0)
        fresh = pond + above
        rate = GUT_WET_PER_TISSUE * tis["gut"][ai, ni].double() * dt_i / WATER_EMPTY_S
        drink = mm * torch.minimum(torch.where(sea, torch.full_like(rate, float("inf")), fresh[idx]), rate)
        got_fresh = _share(idx, torch.where(sea, torch.zeros_like(drink), drink), fresh, cells)
        per_cell = torch.zeros(cells, dtype=torch.float64, device=dev).index_add_(0, idx, got_fresh)
        from_pond = torch.minimum(per_cell, pond)
        _residual(b, geom, "pond_out").view(-1).add_(from_pond)
        _residual(b, geom, "soil_out").view(-1).add_(per_cell - from_pond)
        _settle(b, pstate, geom, "pond_out", pr)
        _settle(b, pstate, geom, "soil_out", pr)
        got_sea = torch.where(sea, drink, torch.zeros_like(drink))
        sea_kg = _arena(ai, got_sea, A)
        pstate.sea_water = pstate.sea_water + sea_kg
        b.ledger["sea_in_kg"] += sea_kg
        drunk = got_fresh + got_sea * (1 - SEA_SALINITY)
        salt = got_sea * SEA_SALINITY
        water_in[ai, ni] += drunk
        b.salt_kg[ai, ni] = b.salt_kg[ai, ni] + salt.float()
        _book_rows(b, "w_drunk", ai, drunk)
        _book_rows(b, "salt_in", ai, salt)
        out["plant_kg"][ai, ni] = dry
        out["drunk_kg"][ai, ni] = drunk
        out["sea_kg"][ai, ni] = got_sea
        out["contact"][ai, ni] = 1
    b.gut = (b.gut.double() + add).float()
    b.water_kg = (b.water_kg.double() + water_in).float()
    return out


def _book_rows(b: Bodies, key: str, arena, x) -> None:
    b.ledger[key].index_add_(0, arena, x.double())


# =========================================================================================== metabolism
def _phi(x):
    """(1 - exp(-x)) / x, 1 at 0."""
    small = x.abs() < 1e-4
    xs = torch.where(small, torch.ones_like(x), x)
    return torch.where(small, 1 - x / 2 + x * x / 6, -torch.expm1(-xs) / xs)


def _psi(x):
    """(1 - phi(x)) / x = (x - 1 + exp(-x)) / x^2, 1/2 at 0."""
    small = x.abs() < 1e-3
    xs = torch.where(small, torch.ones_like(x), x)
    return torch.where(small, 0.5 - x / 6 + x * x / 24, (1 - _phi(xs)) / xs)


def _log_ratio(q):
    """-log(1 - q) / q for q < 1, 1 at 0: the time to a boundary over the time at the initial rate."""
    small = q.abs() < 1e-4
    qs = torch.where(small, torch.full_like(q, 0.5), q)
    return torch.where(small, 1 + q / 2 + q * q / 3, -torch.log1p(-qs.clamp_max(1 - 1e-7)) / qs)


def digest(b: Bodies, dt: float) -> dict:
    """One bout of the gut: each class passes 1 - exp(-dt x arrhenius(T_b) / RETENTION_S) (gut motility and
    enzyme kinetics are chemistry: a cold gut passes and absorbs less per bout); of what passes, enz / (enz + K) is
    absorbed (plant: enz_plant, protein: enz_meat, lipid: both, cooked: min(1, COOK_GAIN x the meat efficiency),
    inert: none). Absorbed nitrogen joins the free pool (with the urea energy and carbon it will carry); the rest of
    the absorbed energy and carbon becomes fat (``_to_reserve``). Returns the faeces per body: kg dry, J, C, N, inert
    kg; the absorbed J and N; and ``heat_j``, the conversion heat the heat balance must hold."""
    tis = tissues(b)
    lean = b.mass_kg.clamp_min(MIN_KG)
    ep = tis["enz_plant"] / lean
    em = tis["enz_meat"] / lean
    eff_m = em / (em + K_ENZ_MEAT)
    eff = torch.stack((ep / (ep + K_ENZ_PLANT), eff_m, (ep + em) / (ep + em + K_ENZ_MEAT),
                       (COOK_GAIN * eff_m).clamp_max(1.0), torch.zeros_like(ep)), -1).double() * b.alive[..., None]
    gut = b.gut.double()
    share = -torch.expm1(-(arrhenius(b.body_k).double() * dt / RETENTION_S))
    passed = gut * share[..., None, None]
    absorbed = passed * eff[..., None]
    faeces = passed - absorbed
    b.gut = (gut - passed).float()
    j_a, c_a, n_a = absorbed[..., Q_J].sum(-1), absorbed[..., Q_C].sum(-1), absorbed[..., Q_N].sum(-1)
    b.n_kg = (b.n_kg.double() + n_a).float()
    heat = _to_reserve(b, (j_a - n_a * UREA_J_PER_KG_N).float(), (c_a - n_a * UREA_C_PER_KG_N).float())
    return {"kg": faeces[..., Q_KG].sum(-1), "j": faeces[..., Q_J].sum(-1), "c": faeces[..., Q_C].sum(-1),
            "n": faeces[..., Q_N].sum(-1), "inert": faeces[..., G_INERT, Q_KG],
            "absorbed_j": j_a, "absorbed_n": n_a, "heat_j": heat}


def child_cost_j(b: Bodies) -> torch.Tensor:
    """The development a body's next child needs (J) [A, N]: DEVELOP_J_PER_KG x offspring_share x lean."""
    return DEVELOP_J_PER_KG * b.genome["offspring_share"].float().clamp(0, 1) * b.mass_kg.clamp_min(0) * b.alive


def develop_power_w(b: Bodies, t_k=None) -> torch.Tensor:
    """The most development power a body can spend (W) [A, N]: PROTEIN_FSR_MAX_DAY of its own protein per day made
    anew at PROTEIN_SYNTH_J_KG, with the Arrhenius factor of ``t_k`` (default T_b)."""
    t = b.body_k if t_k is None else t_k
    rate = PROTEIN_FSR_MAX_DAY / DAY_S * LEAN_PROTEIN * b.mass_kg.clamp_min(0) * PROTEIN_SYNTH_J_KG
    return rate * arrhenius(t) * b.alive


def metabolism(b: Bodies, env: Env, *, geom=None, pstate=None, loco: dict | None = None, eat: dict | None = None,
               work_j=None, loudness=None, develop=None, prules=None) -> dict:
    """One bout of physiology for every living body (see the module docstring): digestion, growth from the store
    above its heritable level, the heat balance (maintenance, repair, thermogenesis, and the heat of the bout's
    work, conversions and synthesis), paying all costs (``loco`` from :func:`move`, bites from :func:`ingest`,
    ``work_j`` [A, N] of other muscular work, calls at ``loudness``, the next child's development at ``develop``
    [A, N] in [0, 1], the share of :func:`develop_power_w` spent on it), urea, water and salt (kidney and bladder),
    wear, healing, osmotic harm and anoxia (``loco['sunk_s']`` and its dive structure, the time the airway was under
    water, and any need beyond the O2 supply in air, at the bout's own O2 use against the body's O2 store and its
    debt: ``PROVENANCE['breath_hold']``; thermogenesis is capped by the supply), and the excreta into the patch (litter
    carbon and nitrogen, water into the soil, through the transit residuals). The development paid is added to
    ``brood_j``. Returns per-body diagnostics (with ``o2_store_mol`` and ``breath_hold_s``, the bout's)."""
    A, N = b.shape
    e = _env(b, env, geom)
    dt = e["dt"]
    alive = b.alive
    af = alive.float()
    pr = prules or pt.PatchRules()
    zero = torch.zeros_like(b.mass_kg)
    r = b.genome["repair"].float().clamp(0, 1)
    rr = 1 + r                                    # repair spends the share r on top of every running cost
    # ------------------------------------------------------------------ gut (rates at the body's temperature)
    fae = digest(b, dt)
    # ------------------------------------------------------------------ the bout's work and the heat it leaves
    sh = shape(b)
    tis = tissues(b, sh)
    loco_met = zero if loco is None else loco["metabolic_j"]
    loco_heat = zero if loco is None else loco["heat_j"]
    bite_j = zero if eat is None else eat["bite_work_j"].float()
    other_j = zero if work_j is None else torch.as_tensor(work_j, dtype=torch.float32, device=b.device) * af
    call_ac = zero if loudness is None else call_power_w(b, loudness, tis) * dt       # sound leaves the body
    call_met = call_ac / VOCAL_EFF
    muscle_met = (bite_j + other_j) / MUSCLE_EFF                  # the work itself goes into what is bitten or held
    t_start = torch.where(alive, b.body_k, torch.full_like(b.body_k, T_REF_K))
    dev_i = zero if develop is None else torch.as_tensor(develop, dtype=torch.float32, device=b.device).clamp(0, 1)
    dev_j = dev_i * develop_power_w(b, t_start) * dt * af       # synthesis: all of it heat in the body
    held_met_j = loco_met + muscle_met + call_met + dev_j
    held_heat_j = loco_heat + (muscle_met - bite_j - other_j) + (call_met - call_ac) + dev_j
    # ------------------------------------------------------------------ growth from the store above its level
    committed = (held_met_j + maintenance_ref_w(b, tis) * arrhenius(t_start) * dt) * rr
    over = (b.reserve_j - store_level_j(b) - committed).clamp_min(0) * af
    per_kg = BUILD_FAT_J_KG + SYNTH_OVERHEAD * E_LEAN
    room = (b.genome["size_kg"].float() - b.mass_kg).clamp_min(0)
    dm = torch.minimum(torch.minimum(room, over / per_kg), b.n_kg.clamp_min(0) / N_LEAN) * af
    b.reserve_j = b.reserve_j - dm * BUILD_FAT_J_KG
    b.n_kg = b.n_kg - dm * N_LEAN
    b.mass_kg = b.mass_kg + dm
    conv_g = dm * (BUILD_FAT_J_KG + N_LEAN * UREA_J_PER_KG_N - E_LEAN)
    _book(b, "e_conversion", conv_g)
    co2_build = dm * (BUILD_FAT_J_KG * FAT_C_PER_J + N_LEAN * UREA_C_PER_KG_N - C_LEAN)
    _book(b, "c_co2", co2_build)
    _book(b, "co2_mol", co2_build / M_C)
    _book(b, "i_built", dm.double() * LEAN_MINERAL)
    overhead = dm * SYNTH_OVERHEAD * E_LEAN
    b.reserve_j = b.reserve_j - overhead          # oxidised: booked after the heat balance, with its breath
    b.frame_kg = torch.maximum(b.frame_kg, b.mass_kg) * af
    growth_heat = conv_g + overhead
    # ------------------------------------------------------------------ powers held over the bout
    sh = shape(b)
    tis = tissues(b, sh)
    held_heat_w = (held_heat_j + fae["heat_j"].float() + growth_heat) / dt
    held_met_w = held_met_j / dt
    oh_w = overhead / dt
    s = submerged_share(b, geom, pstate, None, sh, density(b, tis))
    parts = _o2_store_parts(b, e, tis, t_start)          # the O2 a breath-hold has (PROVENANCE['breath_hold'])
    store = parts["lung"] + parts["carrier"]
    sunk_s = zero if loco is None or "sunk_s" not in loco else \
        torch.as_tensor(loco["sunk_s"], dtype=torch.float32, device=b.device).clamp(0, dt)
    supply = _o2_supply(b, e, tis, sh, t_start)          # mol/s while breathing (PROVENANCE['breathing'])
    supply_w = supply * OXY_J_PER_MOL_O2 * (1 - sunk_s / dt)                 # the bout's mean, none under water
    hx = heat_exchange(b, e, s, sh)
    C = hx["C"]
    G = hx["G"]
    t_e = hx["T_e"]
    gv = hx["G_v"]
    pm_ref = maintenance_ref_w(b, tis)
    gain_th = b.genome["thermo_gain"].float() * b.mass_kg
    setpoint = b.genome["setpoint_k"].float()
    cap_ref = SUSTAINED_SHARE * aerobic(e["p_o2_pa"]) * muscle_ref_w(b, tis) / MUSCLE_EFF
    deficit_air = e["vapour_pa"]
    h = dt / SUBSTEPS
    T = t_start
    maint_j, thermo_j, evap_kg, resp_kg, ox_j = (zero.clone() for _ in range(5))
    # thermogenesis is continuous and piecewise linear in T: the cap below T_c, gain x (setpoint - T) up to the
    # setpoint, nothing above; each sub-step is integrated exactly through its regime changes (at most two); the cap
    # (the muscles' aerobic capacity) follows the Arrhenius factor of the sub-step's start
    inf = torch.full_like(T, float("inf"))
    for _ in range(SUBSTEPS):
        T0 = T
        f0 = arrhenius(T0)
        beta = E_ARRHENIUS_EV / (K_B_EV * T0.clamp_min(1.0) ** 2)
        pm0 = pm_ref * f0
        es0 = e_sat(T0)
        wet = es0 > deficit_air
        e0 = gv * (es0 - deficit_air).clamp_min(0)
        e1 = torch.where(wet, gv * _de_sat_dt(T0), zero)
        lv = latent_heat(T0)
        resp = (lv * resp_water_per_j(T0, e)).clamp(0, 1)          # latent share of each joule oxidised
        wpj = resp / lv                                             # its water (kg/J)
        keep = rr * (1 - resp)
        held = held_heat_w + r * held_met_w - resp * (rr * held_met_w + oh_w)
        cap_th = torch.minimum(cap_ref * f0, (supply_w - oh_w) / rr - pm0).sub(held_met_w).clamp_min(0)
        t_c = setpoint - cap_th / gain_th.clamp_min(1e-30)
        base_a = pm0 * (1 - beta * T0) * keep + held + G * t_e - lv * (e0 - e1 * T0)
        base_b = G + lv * e1 - pm0 * beta * keep
        regime = torch.where(T0 < t_c, 0, torch.where(T0 < setpoint, 1, 2))      # capped, linear, off
        rem = torch.full_like(T, h)
        for _seg in range(3):
            lin, cap = regime == 1, regime == 0
            a_c = base_a + torch.where(lin, gain_th * setpoint * keep, torch.where(cap, cap_th * keep, zero))
            b_c = base_b + torch.where(lin, gain_th * keep, zero)
            rate = (a_c - b_c * T) / C
            up = rate > 0
            bnd = torch.where(up, torch.where(cap, t_c, torch.where(lin, setpoint, inf)),
                              torch.where(regime == 2, setpoint, torch.where(lin, t_c, -inf)))
            gap = bnd - T
            moving = (rate != 0) & torch.isfinite(bnd)
            q = torch.where(moving, gap * b_c / (C * torch.where(moving, rate, torch.ones_like(rate))), zero)
            t_hit = torch.where(moving & (q < 1), gap / torch.where(moving, rate, torch.ones_like(rate))
                                * _log_ratio(q), inf)
            tau = torch.minimum(t_hit, rem).clamp_min(0)
            x = (b_c * tau / C).clamp(-30.0, 1e6)
            t_new = T + rate * tau * _phi(x)
            t_mean = T + rate * tau * _psi(x)
            dT = t_mean - T0
            m_seg = (pm0 * (1 + beta * dT)).clamp_min(0)
            th_seg = torch.where(lin, (gain_th * (setpoint - t_mean)).clamp(min=0),
                                 torch.where(cap, cap_th, zero)).clamp_max(cap_th)
            ox_seg = rr * (m_seg + th_seg + held_met_w) + oh_w
            maint_j = maint_j + m_seg * tau * af
            thermo_j = thermo_j + th_seg * tau * af
            evap_kg = evap_kg + (e0 + e1 * dT).clamp_min(0) * tau * af
            resp_kg = resp_kg + wpj * ox_seg * tau * af
            ox_j = ox_j + ox_seg * tau * af
            hit = t_hit <= rem
            T = torch.where(hit, bnd, t_new).clamp(100.0, 500.0)
            rem = rem - tau
            regime = torch.where(hit, regime + torch.where(up, 1, -1), regime).clamp(0, 2)
        T = torch.where(alive, T, torch.full_like(T, T_REF_K))
    b.body_k = torch.where(alive, T, b.body_k)
    # ------------------------------------------------------------------ fuel, with the breath of the heat balance
    running = maint_j + thermo_j + held_met_j
    repair_j = r * running
    spend = running + repair_j
    wpj_bout = torch.where(ox_j > 0, resp_kg / ox_j.clamp_min(1e-30), resp_water_per_j(b.body_k, e))
    paid = _spend(b, spend, e, wpj_bout)
    _oxidise(b, overhead, e, wpj_bout)
    # the development paid (in proportion when the body could not pay all its costs) goes to the next child
    got = torch.where(spend > 0, (paid / spend.clamp_min(1e-30)).clamp(0, 1), torch.ones_like(spend))
    b.brood_j = torch.where(alive, b.brood_j + dev_j * got, b.brood_j)
    # ------------------------------------------------------------------ urea: free nitrogen leaves
    urea_n = b.n_kg.clamp_min(0) * af
    b.n_kg = b.n_kg - urea_n
    _book(b, "n_urine", urea_n)
    _book(b, "e_urine", urea_n.double() * UREA_J_PER_KG_N)
    _book(b, "c_urine", urea_n.double() * UREA_C_PER_KG_N)
    # ------------------------------------------------------------------ water and salt: skin, faeces, kidney, bladder
    skin = torch.minimum(evap_kg, b.water_kg.clamp_min(0))
    b.water_kg = b.water_kg - skin
    fw = torch.minimum((fae["kg"] * FAECES_WATER_PER_DRY).float(), b.water_kg.clamp_min(0))
    b.water_kg = b.water_kg - fw
    c_max = urine_salt_max(b, tis)
    flow = torch.minimum(b.water_kg.clamp_min(0), URINE_FLOW_SHARE * GFR_KG_PER_KG_S * tis["kidney"] * dt)
    salt_ex = torch.minimum(b.salt_kg.clamp_min(0), flow * c_max) * af
    flush_w = torch.minimum(salt_ex / c_max.clamp_min(1e-30), flow) * (salt_ex > 0)
    b.water_kg = b.water_kg - flush_w
    b.salt_kg = b.salt_kg - salt_ex
    osmotic = b.salt_kg.clamp_min(0) / (OSMOTIC_LETHAL * b.water_kg.clamp_min(1e-9)) * (dt / OSMOTIC_TIME_S) * af
    hold = normal_water(b) + b.genome["bladder"].float().clamp_min(0) * b.mass_kg
    excess = (b.water_kg - hold).clamp_min(0) * af
    b.water_kg = b.water_kg - excess
    urine_w = flush_w + excess
    _book(b, "w_evap", skin)
    _book(b, "w_faeces", fw)
    _book(b, "w_urine", urine_w)
    _book(b, "salt_out", salt_ex)
    # ------------------------------------------------------------------ faeces
    _book(b, "e_faeces", fae["j"])
    _book(b, "c_faeces", fae["c"])
    _book(b, "n_faeces", fae["n"])
    _book(b, "i_faeces", fae["inert"])
    # ------------------------------------------------------------------ damage: healing, permanent wear, salt
    lean = b.mass_kg.clamp_min(MIN_KG)
    o2_bout = (paid + overhead) / OXY_J_PER_MOL_O2
    healable = (b.damage - b.wear).clamp_min(0)
    heal = torch.minimum(repair_j / (lean * HEAL_J_PER_KG), healable) * af
    used = heal * lean * HEAL_J_PER_KG
    prevented = torch.where(running > 0, (repair_j - used) / running.clamp_min(1e-30), zero).clamp(0, 1)
    wear = WEAR_PROTEIN_KG_PER_MOL_O2 * o2_bout * (1 - prevented) / (OXIDISED_LETHAL_SHARE * LEAN_PROTEIN * lean) * af
    b.wear = torch.where(alive, b.wear + wear, b.wear)
    # O2: the debt grows under water and wherever the need passes the supply; beyond the store it is anoxia
    # (PROVENANCE['breath_hold'])
    o2_rate = (spend + overhead) / dt / OXY_J_PER_MOL_O2
    hold = torch.where(alive & (o2_rate > 0), store / o2_rate.clamp_min(1e-30), torch.full_like(store, math.inf))
    if loco is not None and all(k in loco for k in DIVE_KEYS):
        dv = {k: torch.as_tensor(loco[k], dtype=torch.float32, device=b.device) for k in DIVE_KEYS}
    else:
        dv = dive_piece(sunk_s, dt)
    air_s = (dt - sunk_s).clamp_min(0)
    debt0 = b.o2_debt_mol.clamp_min(0)
    deficit = (o2_rate - supply).clamp_min(0)                    # hypoxia: the need beyond the supply in air
    relu = lambda x: x.clamp_min(0)                              # noqa: E731
    beyond0 = relu(debt0 - store)                                # already booked as anoxia
    end_hyp = debt0 + o2_rate * sunk_s + deficit * air_s         # no breath meets the need: no repayment
    end_all = debt0 + o2_rate * dt
    whole = dv["dive_all"] > 0.5
    hypoxic = deficit > 0
    anoxic_dive = relu(debt0 + o2_rate * dv["dive_lead_s"] - store) - beyond0 \
        + relu(o2_rate * dv["dive_inner_s"] - store) + relu(o2_rate * dv["dive_trail_s"] - store)
    debt_end = torch.where(hypoxic, end_hyp, torch.where(whole, end_all, o2_rate * dv["dive_trail_s"]))
    anoxic = torch.where(hypoxic | whole, relu(debt_end - store) - beyond0, relu(anoxic_dive)) * af
    q_ref = maintenance_ref_w(b, tis) / OXY_J_PER_MOL_O2       # the resting O2 use at T_REF
    anoxia = torch.where(q_ref > 0, anoxic / (ANOXIA_TOLERANCE_S * q_ref).clamp_min(1e-30), zero) * af
    b.o2_debt_mol = torch.where(alive, debt_end, zero)
    kept = torch.where(healable > 0, (1 - heal / healable.clamp_min(1e-30)).clamp(0, 1), torch.ones_like(heal))
    b.damage = torch.where(alive, (b.damage - heal).clamp_min(0) + wear + osmotic + anoxia, b.damage)
    b.anoxia = torch.where(alive, torch.minimum(b.anoxia * kept + anoxia, b.damage), b.anoxia)
    # ------------------------------------------------------------------ excreta into the patch
    if pstate is not None and geom is not None:
        cell = pt.cell_index(geom, b.pos).reshape(-1)
        ar = torch.arange(A, device=b.device).repeat_interleave(N)
        _give_litter(b, pstate, geom, ar, cell, (fae["c"] + urea_n.double() * UREA_C_PER_KG_N).reshape(-1),
                     (fae["n"] + urea_n.double()).reshape(-1), pr)
        _give_water(b, pstate, geom, ar, cell, (fw + urine_w).double().reshape(-1), salt_ex.double().reshape(-1))
    return {"maintenance_j": maint_j, "repair_j": repair_j, "thermo_j": thermo_j, "spent_j": spend, "paid_j": paid,
            "growth_kg": dm, "growth_heat_j": growth_heat, "conversion_heat_j": fae["heat_j"], "evap_kg": skin,
            "faeces_kg": fae["kg"], "urine_kg": urine_w, "urea_n_kg": urea_n, "salt_out_kg": salt_ex,
            "wear": wear, "heal": heal, "prevented": prevented, "osmotic": osmotic, "anoxia": anoxia,
            "develop_j": dev_j * got, "submerged": s, "o2_store_mol": store, "o2_store_lung_mol": parts["lung"],
            "o2_store_carrier_mol": parts["carrier"], "o2_supply_w": supply * OXY_J_PER_MOL_O2 * af,
            "anoxic_mol": anoxic, "o2_debt_mol": b.o2_debt_mol, "breath_hold_s": hold,
            "resp_kg_per_j": wpj_bout, "metabolic_w": (paid + overhead) / dt, "G": G, "T_e": t_e, "C": C}


# =========================================================================================== deaths
def death_cause(b: Bodies) -> torch.Tensor:
    """Cause code [A, N] (CAUSES; 0 lives): damage >= 1 (anoxia when anoxia made at least half of it), T_b above
    318 K, below 271 K, water below 0.6 x normal, lean below 1 - LEAN_LOSS_LETHAL of the frame (no fuel). The first
    that applies in that order."""
    cause = torch.zeros(b.shape, dtype=torch.long, device=b.device)
    flags = ((b.mass_kg < (1 - LEAN_LOSS_LETHAL) * b.frame_kg) | (b.mass_kg <= 0), "no_fuel"), \
        (b.water_kg < DEHYDRATION_LETHAL * normal_water(b), "dehydration"), \
        (b.body_k < T_FREEZE_K, "freezing"), (b.body_k > T_DENATURE_K, "denaturation"), \
        (b.damage >= 1.0, "damage"), ((b.damage >= 1.0) & (b.anoxia >= 0.5 * b.damage), "anoxia")
    for flag, name in flags:
        cause = torch.where(flag, torch.full_like(cause, CAUSE[name]), cause)
    return torch.where(b.alive, cause, torch.zeros_like(cause))


def _swap_rows(t: torch.Tensor, a1, n1, a2, n2) -> None:
    x = t[a1, n1].clone()
    t[a1, n1] = t[a2, n2]
    t[a2, n2] = x


def _control_swap(b: Bodies, dying, gen: torch.Generator) -> dict:
    """The no-selection control (PLANET-V3-SPEC 9): the genomes that die are a uniformly random subset of the
    living, as many per arena as physics kills. Each dying body whose genome is not among them swaps its genome,
    live weights, hidden state and lineage ids (:data:`LINEAGE_FIELDS`) with a surviving body whose genome is (paired
    in slot order within the arena), so physics still removes exactly its dead bodies while fate is decoupled from
    the genome. Draws: one uniform per slot. Returns ``genome_lost`` [A, N] (the slots, before the swap, whose genome
    died) and the swapped pairs. A genome that random survival keeps stays expressed in its new body, so a lethal
    one goes on causing deaths and is never purged: such a population melts down (a property of any control with
    expressed genes); :class:`Shadow` is the drift baseline."""
    A, N = b.shape
    count = dying.sum(1)
    u = torch.rand(A, N, generator=gen, device=b.device)
    u = torch.where(b.alive, u, torch.full_like(u, 2.0))
    rank = torch.argsort(torch.argsort(u, dim=1, stable=True), dim=1, stable=True)
    lost = b.alive & (rank < count[:, None])
    keep_from = dying & ~lost                     # a dying body whose genome survives ...
    keep_to = lost & ~dying                       # ... moves it into a surviving body whose genome dies
    fa, fn = keep_from.nonzero(as_tuple=True)     # row-major: by arena, then slot; equal counts per arena
    ta, tn = keep_to.nonzero(as_tuple=True)
    if fa.numel():
        for name in b.genome:
            _swap_rows(b.genome[name], fa, fn, ta, tn)
        for t in (b.wh_live, b.hidden):
            if t is not None:
                _swap_rows(t, fa, fn, ta, tn)
        for name in LINEAGE_FIELDS:
            _swap_rows(getattr(b, name), fa, fn, ta, tn)
    b.ledger["control_swaps"] += keep_from.sum(1).double()
    return {"genome_lost": lost, "swap_from": torch.stack((fa, fn), -1), "swap_to": torch.stack((ta, tn), -1)}


def deaths(b: Bodies, *, geom=None, pstate=None, pool=None, selection: bool = True, gen=None,
           spawn_items=None, prules=None) -> dict:
    """Remove the dead (:func:`death_cause`), always exactly the bodies physics kills. ``selection`` False (the
    no-selection control) first decouples genome from fate (:func:`_control_swap`, drawn from ``gen``): the genomes
    lost are a uniformly random subset of the living, as many as the deaths. Each dead body becomes items at its
    position on the ground (``pool``, patch coordinates, at its body temperature): meat, bone and hide by BODY_PLAN x
    lean (scaled down to the water it holds) and fat = reserve / 39.5 MJ/kg; parts without a slot (or without a
    pool), the gut contents and the free nitrogen go to the cell's litter (carbon, nitrogen), the water the items do
    not hold to the soil (or the sea), its salt to the land (or the sea). Held items drop where it lies. Carcass
    items are placed through ``spawn_items`` (default :func:`spawn_carcass`). Returns dead, cause (the physical
    one), the per-body item_j, item_c, item_w, litter_c, litter_n, soil_w [A, N], and in the control
    ``genome_lost`` and the swapped pairs."""
    A, N = b.shape
    dev = b.device
    if not selection and gen is None:
        raise ValueError("the no-selection control needs the world generator")
    cause = death_cause(b)
    dead = cause > 0
    z = torch.zeros(A, N, dtype=torch.float64, device=dev)
    out = {"dead": dead, "cause": cause, "item_j": z.clone(), "item_c": z.clone(), "item_w": z.clone(),
           "litter_c": z.clone(), "litter_n": z.clone(), "soil_w": z.clone()}
    if not selection:
        out.update(_control_swap(b, dead, gen))
    for c in range(1, len(CAUSES)):
        b.ledger[f"deaths_{CAUSES[c]}"] += (cause == c).sum(1).double()
    if not bool(dead.any()):
        return out
    ai, ni = dead.nonzero(as_tuple=True)
    lean = b.mass_kg[ai, ni].double().clamp_min(0)
    reserve = b.reserve_j[ai, ni].double().clamp_min(0)
    water = b.water_kg[ai, ni].double().clamp_min(0)
    gut = b.gut[ai, ni].double()
    n_free = b.n_kg[ai, ni].double().clamp_min(0)
    salt = b.salt_kg[ai, ni].double().clamp_min(0)
    body_j = lean * E_LEAN + reserve + gut[..., Q_J].sum(-1) + n_free * UREA_J_PER_KG_N
    body_c = lean * C_LEAN + reserve * FAT_C_PER_J + gut[..., Q_C].sum(-1) + n_free * UREA_C_PER_KG_N
    body_n = lean * N_LEAN + gut[..., Q_N].sum(-1) + n_free
    parts = list(BODY_PLAN) + ["fat"]
    sp = torch.tensor([mt.IDX[p] for p in parts], device=dev)
    wat = SPECIES_WATER.to(dev)[sp]
    mass = torch.stack([lean * BODY_PLAN[p] for p in BODY_PLAN] + [reserve / FAT_J_KG], -1)        # [D, 4]
    need = (mass * wat).sum(-1)
    fill = torch.where(need > water, water / need.clamp_min(1e-300) * (1 - 1e-6), torch.ones_like(need))
    mass = torch.where(wat > 0, mass * fill[:, None], mass)
    mass = torch.where(mass >= CARCASS_MIN_KG, mass, torch.zeros_like(mass))
    made = torch.zeros_like(mass, dtype=torch.bool)
    if pool is not None:
        Dn = ai.shape[0]
        rw = ai.repeat_interleave(len(parts))
        comp = torch.nn.functional.one_hot(sp, mt.S).float().repeat(Dn, 1)
        pos = torch.cat((b.pos[ai, ni], torch.zeros(Dn, 1, device=dev)), -1).repeat_interleave(len(parts), 0)
        temp = b.body_k[ai, ni].repeat_interleave(len(parts))
        req = mass.reshape(-1).float()
        want = req > 0
        slot = torch.full_like(rw, -1)
        if bool(want.any()):
            slot[want] = (spawn_items or spawn_carcass)(pool, rw[want], comp[want], req[want], pos[want],
                                                        temp[want])
        made = (slot >= 0).reshape(Dn, len(parts))
        stored = torch.where(made, pool.mass[rw, slot.clamp_min(0)].double().reshape(Dn, len(parts)),
                             torch.zeros_like(mass))
        mass = stored
        b.ledger["carcass_kg"].index_add_(0, ai, mass @ torch.nn.functional.one_hot(sp, mt.S).double())
    got = mass * made
    table = SPECIES_GUT.to(dev)[sp]                                                         # [4, G, Q]
    item_j = (got * table[:, :, Q_J].sum(-1)).sum(-1)
    item_c = (got * table[:, :, Q_C].sum(-1)).sum(-1)
    item_n = (got * table[:, :, Q_N].sum(-1)).sum(-1)
    item_w = (got * wat).sum(-1)
    litter_c = body_c - item_c
    litter_n = body_n - item_n
    soil_w = water - item_w
    out["item_j"][ai, ni], out["item_c"][ai, ni], out["item_w"][ai, ni] = item_j, item_c, item_w
    out["litter_c"][ai, ni], out["litter_n"][ai, ni], out["soil_w"][ai, ni] = litter_c, litter_n, soil_w
    _book_rows(b, "e_dead", ai, body_j)
    _book_rows(b, "c_dead", ai, body_c)
    _book_rows(b, "n_dead", ai, body_n)
    _book_rows(b, "w_dead", ai, water)
    _book_rows(b, "i_dead", ai, gut[..., G_INERT, Q_KG] + lean * LEAN_MINERAL)
    _book_rows(b, "salt_dead", ai, salt)
    if pstate is not None and geom is not None:
        cell = pt.cell_index(geom, b.pos[ai, ni][None])[0]
        _give_litter(b, pstate, geom, ai, cell, litter_c.clamp_min(0), litter_n.clamp_min(0), prules)
        _give_water(b, pstate, geom, ai, cell, soil_w.clamp_min(0), salt)
    if pool is not None:
        held = b.held[ai, ni]                                                               # [D, 2]
        hw = ai[:, None].expand_as(held)[held >= 0]
        hi = held[held >= 0]
        if hi.numel():
            pool.holder[hw, hi] = -1
            hp = torch.cat((b.pos[ai, ni], torch.zeros(ai.shape[0], 1, device=dev)), -1)
            pool.pos[hw, hi] = hp[:, None].expand(-1, 2, -1)[held >= 0].to(pool.pos.dtype)
    _clear(b, ai, ni)
    return out


def _clear(b: Bodies, ai, ni) -> None:
    b.alive[ai, ni] = False
    for name in ("mass_kg", "frame_kg", "reserve_j", "water_kg", "n_kg", "salt_kg", "damage", "wear", "brood_j",
                 "body_k", "heading", "o2_debt_mol", "anoxia"):
        getattr(b, name)[ai, ni] = 0.0
    b.vel[ai, ni] = 0.0
    b.gut[ai, ni] = 0.0
    b.held[ai, ni] = -1


# =========================================================================================== division
def divide(b: Bodies, gen: torch.Generator, env: Env | None = None, *, geom=None) -> dict:
    """Births: every living body whose development for its next child is complete (``brood_j`` at least
    :func:`child_cost_j`; the development is paid in :func:`metabolism` at the divide output's share of
    :func:`develop_power_w`) divides now, when its child of ``offspring_share`` x lean and what it keeps are both at
    least MIN_BODY_KG (else ``too_small``). The child takes offspring_share of the parent's lean mass, store, water and
    free nitrogen, and the parent's frame shrinks by the same share (so a split is not a starvation); the mass is the
    parent's own, and its making anew was the development paid before. The parent keeps its gut, salt, grips, damage
    and wear, and the development beyond this child's cost for its next one. Children fill the lowest free slots of
    their arena in a random order of the requests (drawn from ``gen``). A birth that fails (no free slot:
    ``capacity_full``; too small: ``too_small``) loses the whole development paid for it, as a real one would have
    spent it. ``divide_fired`` counts completed developments (births attempted). Up to DEVELOP_PASSES births per
    body and call. The child: touching the parent's rear (its centre at the parent's semi-major axis plus its own
    radius behind the parent, wrapped), the parent's body temperature, a random heading, damage, wear and development
    0, empty gut, its frame its lean, the genome ``brain3.mutate3`` of the parent's (with :data:`GENE_SPECS`; live
    weights from its Wh, zero hidden state), generation + 1. Draws (fixed order, per pass): the order of the
    requests, mutation, headings. ``env`` and ``geom`` are accepted for call compatibility and not used."""
    if not isinstance(gen, torch.Generator):
        raise TypeError("divide(b, gen): births come from the development paid in metabolism(develop=...)")
    A, N = b.shape
    dev = b.device
    empty_l = torch.empty(0, dtype=torch.long, device=dev)
    born = torch.zeros(A, dtype=torch.long, device=dev)
    got_a, got_n, got_p = [empty_l], [empty_l], [empty_l]
    for _ in range(DEVELOP_PASSES):
        cost = child_cost_j(b)
        ready = b.alive & (cost > 0) & (b.brood_j >= cost)
        if not bool(ready.any()):
            break
        _book(b, "divide_fired", ready)
        share = b.genome["offspring_share"].float()
        child_kg = share * b.mass_kg
        big = (child_kg >= MIN_BODY_KG) & (b.mass_kg - child_kg >= MIN_BODY_KG)
        small = ready & ~big
        _book(b, "too_small", small)
        b.brood_j = torch.where(small, torch.zeros_like(b.brood_j), b.brood_j)
        pa, pn = (ready & big).nonzero(as_tuple=True)
        if not pa.numel():
            continue
        order = torch.argsort(torch.rand(pa.numel(), generator=gen, device=dev), stable=True)
        pa, pn = pa[order], pn[order]
        slot = itm.allocate(b.alive, pa)
        full = slot < 0
        if bool(full.any()):
            b.ledger["capacity_full"].index_add_(0, pa[full], torch.ones(int(full.sum()), dtype=torch.float64,
                                                                         device=dev))
            b.brood_j[pa[full], pn[full]] = 0.0
        ok = ~full
        pa, pn, slot = pa[ok], pn[ok], slot[ok]
        if not pa.numel():
            continue
        b.brood_j[pa, pn] = (b.brood_j[pa, pn] - cost[pa, pn]).clamp_min(0)
        n_a = _birth(b, gen, pa, pn, slot, share)
        born += n_a
        got_a.append(pa)
        got_n.append(slot)
        got_p.append(pn)
    return {"born": born, "child_a": torch.cat(got_a), "child_n": torch.cat(got_n), "parent_n": torch.cat(got_p)}


def _birth(b: Bodies, gen: torch.Generator, pa, pn, slot, share) -> torch.Tensor:
    """Split the children of parents (pa, pn) into the free slots (pa, slot) (see :func:`divide`). Returns the births
    per arena [A]."""
    A = b.shape[0]
    dev = b.device
    child = b3.mutate3(gen, b3.rows(b.genome, pa, pn), specs=GENE_SPECS)
    heading = torch.rand(pa.numel(), generator=gen, device=dev) * (2 * math.pi)
    s = share[pa, pn]
    c_mass = s * b.mass_kg[pa, pn]
    c_res = s * b.reserve_j[pa, pn]
    c_wat = s * b.water_kg[pa, pn]
    c_n = s * b.n_kg[pa, pn]
    b.mass_kg[pa, pn] -= c_mass
    b.frame_kg[pa, pn] *= (1 - s)
    b.reserve_j[pa, pn] -= c_res
    b.water_kg[pa, pn] -= c_wat
    b.n_kg[pa, pn] -= c_n
    for name in b.genome:
        b.genome[name][pa, slot] = child[name].to(b.genome[name].dtype)
    if b.wh_live is not None:
        b.wh_live[pa, slot] = child["Wh"].to(b.wh_live.dtype)
    if b.hidden is not None:
        b.hidden[pa, slot] = 0.0
    b.alive[pa, slot] = True
    b.heading[pa, slot] = heading
    b.vel[pa, slot] = 0.0
    b.mass_kg[pa, slot] = c_mass
    b.frame_kg[pa, slot] = c_mass
    b.reserve_j[pa, slot] = c_res
    b.water_kg[pa, slot] = c_wat
    b.n_kg[pa, slot] = c_n
    b.salt_kg[pa, slot] = 0.0
    b.body_k[pa, slot] = b.body_k[pa, pn]
    b.damage[pa, slot] = 0.0
    b.wear[pa, slot] = 0.0
    b.o2_debt_mol[pa, slot] = 0.0
    b.anoxia[pa, slot] = 0.0
    b.brood_j[pa, slot] = 0.0
    b.gut[pa, slot] = 0.0
    b.held[pa, slot] = -1
    b.age_bouts[pa, slot] = 0
    # the child lies touching the parent's rear, not inside it
    sh = shape(b)
    back = (sh["a"][pa, pn] + sh["r1"][pa, slot])[:, None]
    rear = torch.stack((torch.cos(b.heading[pa, pn]), torch.sin(b.heading[pa, pn])), -1)
    b.pos[pa, slot] = torch.remainder(b.pos[pa, pn] - back * rear, b.L)
    onehot = torch.nn.functional.one_hot(pa, A)
    rank = (onehot.cumsum(0) * onehot).sum(1) - 1
    b.uid[pa, slot] = b.next_uid[pa] + rank
    b.parent[pa, slot] = b.uid[pa, pn]
    b.founder[pa, slot] = b.founder[pa, pn]
    b.generation[pa, slot] = b.generation[pa, pn] + 1
    born = onehot.sum(0)
    b.next_uid += born
    b.ledger["births"] += born.double()
    return born


# =========================================================================================== the neutral shadow
@dataclass
class Shadow:
    """The neutral shadow of a run (Bedau & Brown 1999, Artificial Life 5, 17): genomes without the network weights
    (every body and learning gene and ``k``) in [A, N] slots that undergo the run's own numbers of deaths and births
    per arena and bout, but whose members die and reproduce uniformly at random, mutating as bodies do. Nothing is
    expressed, so its gene trajectories are drift and mutation only: the baseline against which a run's change
    measures selection (PLANET-V3-SPEC 9 and 10). ``hidden`` bounds k as the network size does; ``founder`` [A, N]
    the founder each member descends from (its lineage count is the drift baseline of the run's lineages_alive)."""
    alive: torch.Tensor
    genes: dict
    hidden: int
    founder: torch.Tensor | None = None

    def state_dict(self) -> dict:
        return {"alive": self.alive.clone(), "genes": {k: v.clone() for k, v in self.genes.items()},
                "hidden": int(self.hidden), "founder": None if self.founder is None else self.founder.clone()}

    @classmethod
    def from_state(cls, d: dict) -> "Shadow":
        f = d.get("founder")
        return cls(alive=d["alive"].clone(), genes={k: v.clone() for k, v in d["genes"].items()},
                   hidden=int(d["hidden"]), founder=None if f is None else f.clone())


def shadow_of(b: Bodies) -> Shadow:
    """A shadow starting from the bodies' current genomes (normally the founders') and their founder ids."""
    genes = {k: v.clone() for k, v in b.genome.items() if k not in b3.WEIGHT_GENES}
    hidden = int(b.genome["Wh"].shape[-1]) if "Wh" in b.genome else b3.HIDDEN
    return Shadow(alive=b.alive.clone(), genes=genes, hidden=hidden, founder=b.founder.clone())


def _mutate_genes(gen: torch.Generator, parent: dict, hidden: int) -> dict:
    """brain3.mutate3's mutation of the genes without network weights, in its order: ``mut`` (its fixed log step),
    ``k`` (+-1 with probability K_MUT_RATE within [2, hidden]), then every other gene of GENE_SPECS in sorted order
    by its GeneSpec3 (scaled by the child's mut)."""
    k = parent["k"]
    lead = tuple(k.shape)
    dev = k.device
    s_mut = GENE_SPECS["mut"]
    mut_p = parent["mut"].float()
    child = {"mut": b3._gene_step(s_mut, mut_p, torch.randn(lead, generator=gen, device=dev),
                                  mut_p if s_mut.by_mut else 1.0).to(parent["mut"].dtype)}
    mut = child["mut"].float()
    step = torch.where(torch.rand(lead, generator=gen, device=dev) < .5, -1, 1)
    change = torch.rand(lead, generator=gen, device=dev) < b3.K_MUT_RATE
    child["k"] = (k + step * change).clamp(2, hidden).to(k.dtype)
    for name in sorted(n for n in parent if n in GENE_SPECS and n != "mut"):
        t, s = parent[name], GENE_SPECS[name]
        scale = b3._trail(mut, t) if s.by_mut else 1.0
        child[name] = b3._gene_step(s, t, torch.randn(t.shape, generator=gen, device=dev), scale).to(t.dtype)
    return child


def shadow_step(sh: Shadow, deaths, births, gen: torch.Generator) -> dict:
    """One bout of the shadow: ``deaths`` [A] members chosen uniformly at random die (at most all), then ``births`` [A]
    children are born to parents chosen uniformly at random among the living, each living member parent of at most
    one child until every one has had one (as a body completes at most one birth per pass), into the lowest free
    slots (a birth without a free slot fails), each with :func:`_mutate_genes` of its parent and its parent's founder.
    Pass the run's counts of the bout (``deaths(...)['dead'].sum(1)`` and ``divide(...)['born']``) and a generator of
    the shadow's own, so the run's draws do not change. Draws: one uniform per slot (deaths), one per slot (the order
    of the parents), the mutation. Returns died, born [A]."""
    A, N = sh.alive.shape
    dev = sh.alive.device
    d = torch.as_tensor(deaths, device=dev).long().reshape(A)
    n_b = torch.as_tensor(births, device=dev).long().reshape(A)
    u = torch.rand(A, N, generator=gen, device=dev)
    u = torch.where(sh.alive, u, torch.full_like(u, 2.0))
    rank = torch.argsort(torch.argsort(u, dim=1, stable=True), dim=1, stable=True)
    kill = sh.alive & (rank < d[:, None])
    sh.alive = sh.alive & ~kill
    living = sh.alive.sum(1)
    want = torch.where(living > 0, n_b, torch.zeros_like(n_b))
    born = torch.zeros(A, dtype=torch.long, device=dev)
    top = int(want.max()) if want.numel() else 0
    if top > 0:
        v = torch.rand(A, N, generator=gen, device=dev)
        v = torch.where(sh.alive, v, torch.full_like(v, 2.0))
        order = torch.argsort(v, dim=1, stable=True)                         # the living first, in a random order
        j = torch.arange(top, device=dev)[None].expand(A, top)
        par = order.gather(1, j % living.clamp_min(1)[:, None])             # without replacement until all had one
        sel = j < want[:, None]
        pa = torch.arange(A, device=dev)[:, None].expand(A, top)[sel]
        pn = par[sel]
        slot = itm.allocate(sh.alive, pa)
        ok = slot >= 0
        pa, pn, slot = pa[ok], pn[ok], slot[ok]
        if pa.numel():
            child = _mutate_genes(gen, {k: v[pa, pn] for k, v in sh.genes.items()}, sh.hidden)
            for k in sh.genes:
                sh.genes[k][pa, slot] = child[k].to(sh.genes[k].dtype)
            if sh.founder is not None:
                sh.founder[pa, slot] = sh.founder[pa, pn]
            sh.alive[pa, slot] = True
            born = torch.zeros(A, dtype=torch.long, device=dev).index_add_(0, pa, torch.ones_like(pa))
    return {"died": kill.sum(1), "born": born}


PROVENANCE["Shadow"] = (R_, "the neutral shadow model of Bedau & Brown 1999 (Artificial Life 5, 17; Bedau, Snyder & "
                            "Packard 1998): the run's own demography with random deaths and births, so drift and "
                            "mutation alone move its genes. A control in which genes stay expressed cannot do this: "
                            "a lethal genome that random survival keeps (deaths with selection False) goes on causing "
                            "deaths, so such a population melts down; the shadow is the drift baseline")


# =========================================================================================== the bout
def bout(b: Bodies, motor: dict, env: Env, gen: torch.Generator, *, geom=None, pstate=None, pool=None, nbr=None,
         work_j=None, selection: bool = True, prules=None, take_items=None, spawn_items=None, loco=None) -> dict:
    """One bout for every body: move, ingest, metabolism, deaths, division, age (see the module docstring).
    ``motor`` holds brain3.decode's turn, thrust, mouth, loudness and divide ([A, N] each; missing ones are 0);
    ``work_j`` [A, N] is other mechanical work of the bout (manipulate.bout's), paid at MUSCLE_EFF; ``take_items``
    and ``spawn_items`` route item changes through another item ledger (manipulate's). ``loco`` (a :func:`move` or
    :func:`sum_loco` result) means the world has already moved the bodies in sub-steps, and ``turn`` and ``thrust``
    are not applied again."""
    z = torch.zeros(b.shape, device=b.device)
    get = lambda k: motor.get(k, z) if motor is not None else z
    if loco is None:
        loco = move(b, get("turn"), get("thrust"), env, geom=geom, pstate=pstate)
    eat = ingest(b, get("mouth"), env, geom=geom, pstate=pstate, pool=pool, nbr=nbr, prules=prules,
                 take_items=take_items)
    met = metabolism(b, env, geom=geom, pstate=pstate, loco=loco, eat=eat, work_j=work_j, loudness=get("loudness"),
                     develop=get("divide"), prules=prules)
    dead = deaths(b, geom=geom, pstate=pstate, pool=pool, selection=selection, gen=gen, spawn_items=spawn_items,
                  prules=prules)
    div = divide(b, gen)
    b.age_bouts = b.age_bouts + b.alive.long()
    return {"move": loco, "ingest": eat, "metabolism": met, "deaths": dead, "divide": div}


# =========================================================================================== views for the senses
def sense_view(b: Bodies, loudness=None, calls=None) -> dict:
    """The body tensors senses3 reads, by its field names (``senses3.Bodies.from_mapping``): the whole body's mass
    (lean, fat, gut, water beyond the normal), speed, store and its channel scale (:func:`reserve_capacity`, which
    the store may exceed), water and its normal, dry gut contents
    and the gut's capacity, the genes it uses, the mouth point and reach of this module, the radius of the
    equal-volume sphere and the call's acoustic power. senses3 owns the interoceptive channels themselves."""
    sh = shape(b)
    tis = tissues(b, sh)
    A, N = b.shape
    if loudness is None:
        loud = torch.zeros(A, N, device=b.device)
    else:
        loud = torch.as_tensor(loudness, dtype=torch.float32, device=b.device) * b.alive
    voc = torch.zeros(A, N, b3.VOCAL_DIMS, device=b.device) if calls is None else calls
    g = b.genome
    return {"alive": b.alive, "pos": b.pos, "heading": b.heading, "mass_kg": sh["m"] * b.alive,
            "speed_m_s": b.speed, "body_k": b.body_k, "damage": b.damage, "reserve_j": b.reserve_j,
            "reserve_cap_j": reserve_capacity(b), "water_kg": b.water_kg, "water_norm_kg": normal_water(b),
            "gut_kg": b.gut[..., Q_KG].sum(-1), "gut_cap_kg": gut_capacity_kg(b, tis), "eye": g["eye"],
            "ear": g["ear"], "voice": g["voice"], "muscle": g["muscle"], "loudness": loud, "calls": voc,
            "held": b.held, "fur_m": g["fur_m"], "radius_m": sh["r1"] * b.alive,
            "call_w": call_power_w(b, loud, tis), "mouth_pos": mouth_position(b, sh),
            "reach_m": mouth_reach_m(b, sh)}


def manipulation_view(b: Bodies, env: Env, thrust=None, geom=None) -> dict:
    """The body tensors manipulate.bout reads: alive, pos, mouth_pos, body_mass_kg, power_w (the sustained power,
    :func:`sustained_power_w`, less the share |thrust| the bout's locomotion takes) and reach_m; its
    ``work_j`` goes back into :func:`metabolism` (or :func:`bout`) as ``work_j``."""
    e = _env(b, env, geom)
    sh = shape(b)
    tis = tissues(b, sh)
    if thrust is None:
        used = torch.zeros_like(b.mass_kg)
    else:
        used = torch.as_tensor(thrust, dtype=torch.float32, device=b.device).abs().clamp(0, 1)
    power = _sustained(b, e, tis, sh) * (1 - used)
    return {"alive": b.alive, "pos": b.pos, "mouth_pos": mouth_position(b, sh), "body_mass_kg": sh["m"] * b.alive,
            "power_w": power, "reach_m": mouth_reach_m(b, sh), "gravity": e["gravity_m_s2"]}


PROVENANCE["views"] = (N_, "sense_view and manipulation_view hand senses3 and manipulate this module's own body "
                           "quantities under their field names, so mass, mouth, reach and power are defined once")
