"""Bodies, energetics and life history of the individuals (PLANET-SPEC section 2.8).

State lives in :class:`Creatures`, tensors with a leading ``[W, N]`` (worlds, slots). A slot is
used when ``alive`` is true. One tick is one day. The functions of a day, in the order a world
calls them:

* :func:`move`: the day's travel along a great circle of the habitat globe (turn and speed from
  the brain), with the cost of walking (Taylor, Heglund & Maloiy 1982), climbing and swimming.
* :func:`eat_plants`, :func:`eat_meat`, :func:`drink`: intake through a harvest callback (the
  biosphere's scatter), from food items of the item pool, and from soil, lakes or the sea.
* :func:`physiology`: basal (Kleiber), brain, senses, thermoregulation, calling, juvenile growth
  (von Bertalanffy), starvation catabolism, water turnover (Nagy & Peterson 1988), hypoxia and
  hypothermia; it returns the respiration O2 and CO2 moles for the gas ledger.
* :func:`deaths`: starvation, dehydration, injury, age, hypoxia or cold; a body becomes meat,
  bone, hide and fat items (:func:`items.spawn`), the rest goes to litter.
* :func:`reproduce`: asexual litters with allometric life history; the genome mutates through
  :func:`brain.mutate`.

Fields the climate and biosphere own are plain arguments (no import of those modules):
``t_air_k [W, C]`` K, ``p_o2_pa [W]`` or ``[W, C]`` Pa, ``soil_kg_m2 [W, C]``, ``land [W, C]``
bool, ``lake [W, C]`` bool (optional), the terrain dict of :func:`globe.make_terrain` and
``gravity_m_s2 [W]``. Plants are eaten through ``harvest(w [M], cell [M], kg_dry [M]) -> kg_dry
given [M]`` (for example the biosphere's ``harvest`` or :func:`field_harvester`).

Body model (``new_rule``, see ``PROVENANCE``). ``mass_kg`` is the lean body at normal
hydration, made of the carcass species in :data:`BODY_MIX` (meat 0.84, bone 0.10, hide 0.06 by
mass: the spec's carcass items plus the viscera counted as meat-like tissue). Its gross energy,
carbon and water per kg (:data:`TISSUE_J_KG`, :data:`TISSUE_C_KG`, :data:`WATER_FRACTION`) follow
from the materials makeup. ``reserve_j`` is fat (39.5 MJ/kg, at most 0.3 M); its carbon is
``reserve_j x FAT_C_PER_J`` exactly. Every conversion into or out of the reserve is booked at
those ratios: digested food carries more carbon per joule than fat, and the surplus leaves as
CO2 (lipogenesis); lean tissue burnt in starvation does the same; tissue built from fat
(growth, offspring) takes the fat that holds its carbon, :data:`GROWTH_J_KG` per kg, and the
energy beyond the tissue's own leaves as heat (a 12 % cost of growth). So the energy, carbon and
water ledgers of the individuals close exactly (to float32 rounding): see :func:`ledger_errors`.
Water also moves between bodies and food items: a corpse's meat, bone and hide carry their
water into the item pool (:func:`deaths` returns only the rest as ``water_kg``) and eating an
item takes its water into the body (``food_w``). The world's water ledger must therefore count
the water held in items (materials makeup x item species mass) as a store of its own.

Respiration O2 is the reserve energy oxidised / 4.5e5 J/mol (spec; the heat of tissue synthesis
takes none); CO2 is the carbon actually oxidised (fat RQ 0.72, plus the conversion CO2), not the
spec's fixed 0.85 x O2 (reported deviation): a grazer's overall RQ comes out near 0.95, as for
real herbivores. The oxygen of the O2 not returned as CO2 becomes metabolic water in the body
(``metabolic_w``), so O atoms taken from the air come back as CO2 or water (``o2_mol``,
``oxidised_c`` and ``metabolic_w`` are booked for that check). Grazing brings the forage's water
(:data:`FORAGE_WATER_KG`) through the world's soil-water callback.

Every tensor has the leading ``[W, N]`` axes; there are no Python loops over individuals,
cells or items. State is float32 whatever the dtype of the fields passed in (climate gives
float64 pressures), ledger sums float64. The only randomness (founders, the breeding order
at capacity, mutation, headings of the newborn) is drawn from the generator passed in.

The day's functions rebind ``reserve_j``, ``water_kg``, ``mass_kg``, ``health`` and ``harm``
(out of place); :func:`deaths`, :func:`reproduce` and :func:`place` write the slots they
empty or fill in place. A caller that wants a snapshot (for the brain's outcome) clones it.

Strikes (:func:`injure`, from crafting) are remembered in ``struck`` until the next
:func:`physiology`: no healing on a day with a strike or at health <= 0, and the cause code
``harm`` names the largest of the day's harms (strike, hypoxia, cold).
"""
from __future__ import annotations

import ctypes
import hashlib
import math
from dataclasses import dataclass, field

import torch

from . import brain, items, materials
from .constants import DAY_S, RHO_WATER, YEAR_S, molar_mass

TWO_PI = 2 * math.pi
_C = materials.ELEMENTS.index("C")
M_C = molar_mass("C")                      # kg/mol
M_H2O = molar_mass("H2O")                  # kg/mol


def _carbon(s):
    return materials.ELEMENT_FRACTION[materials.IDX[s]][_C]


def _water(s):
    return dict(materials.MAKEUP[s]).get("H2O", 0.0)


# ------------------------------------------------------------------------------------ energetics
KLEIBER_W = brain.KLEIBER_W                # W kg^-0.75 (Kleiber 1947)
COT_J_KG_M, COT_EXP = 10.7, -0.316         # J kg^-1 m^-1 x M^-0.316 (Taylor, Heglund & Maloiy 1982)
MUSCLE_EFF = 0.25                          # climbing M g dh / 0.25 (spec; Margaria 1976)
V_TROT, V_TROT_EXP = 1.53, 0.24            # m/s x M^0.24, trot-gallop transition (Heglund, Taylor & McMahon 1974)
MOVE_S = 7200.0                            # new_rule: a day's travel is at most 2 h at that speed
MAX_TURN = math.pi                         # new_rule: a day's travel may start in any direction
PATH_SAMPLES_MAX = 256                     # numerics: path samples, one per smallest cell spacing, at most 256
PATH_BLOCK = 32                            # numerics: path samples evaluated together (memory)
MEEH_K = 0.1                               # m^2 kg^-2/3: body surface A = 0.1 M^(2/3) (Meeh 1879; spec)
SWIM_CD = 0.04                             # drag coefficient on the wetted (Meeh) area, surface swimming
ETA_PADDLE, ETA_LIFT = 0.33, 0.85          # propulsive efficiency, paddling vs lift-based swimmers
SWIM_LAND_PENALTY = 1.0                    # new_rule: land cost x (1 + swim): flippers walk badly
AEROBIC_SCOPE = 10.0                       # maximal over basal aerobic power
T_LC0, T_LC_PER_INS = 303.0, 8.0           # K: lower critical T = 303 - 8 insulation (spec)
#: W/K x M^0.5 / insulation: Herreid & Kessel 1967 minimal conductance 1.02 W_g^-0.505 ml O2 g^-1 h^-1 C^-1,
#: i.e. 1.02 (1000 M)^0.495 ml O2 / (h K) x 20.1 J/ml / 3600 s = 0.174 M^0.495 (spec 2.8 had 1.0: deviation)
C_TH = 1.02 * 1000.0 ** 0.495 * 20.1 / 3600.0
T_BODY = 310.0                             # K: above it heat is shed by evaporation (spec)
LATENT_J_KG = 2.43e6                       # J/kg, latent heat of water at 30-35 C (spec)
NEURON_COST_W = 0.002                      # W per active brain unit (spec, new_rule)
SENSE_UNIT = {"acuity": 2.0, "hearing": 1.0}   # world7 unit costs, eyes 2 ears 1 (evo/senses.py UNIT_COST)
#: chain: the 781 senses era spent 2,613 of 35,297 intake units on sensors (7.4 %); founders' eyes
#: 2.04 and ears 0.685 cost that share of basal with cost ~ unit x level^2 (evo/senses.py:472)
SENSE_SHARE_781 = 2613.0 / 35297.0
SENSE_W = SENSE_SHARE_781 * KLEIBER_W / (SENSE_UNIT["acuity"] * 2.04 ** 2 + SENSE_UNIT["hearing"] * 0.685 ** 2)
AEROBIC_HALF_PA = 3000.0                   # a = pO2 / (pO2 + 3 kPa) (spec)
HYPOXIA_PA = 1000.0                        # health falls below 1 kPa O2 (spec)
HYPOXIA_PER_DAY = 2.0                      # new_rule: health lost per day x (1 - pO2 / 1 kPa)
COLD_PER_DAY = 1.0                         # new_rule: health lost per day x the uncovered share of heat loss
HEAL_PER_DAY = 0.05                        # new_rule: healing when fed, watered, alive and unharmed that day
OXY_J_MOL = 4.5e5                          # J per mol O2 (spec; 20.1 kJ/L x 22.4 L/mol)
# ------------------------------------------------------------------------------------ water
WATER_TURNOVER, WATER_EXP = 0.12, 0.82     # kg/day x M^0.82 (Nagy & Peterson 1988)
DEHYDRATION_LETHAL = 0.8                   # death below 0.8 x the norm (spec)
DRINK_SOIL_MIN = 50.0                      # kg/m^2: soil wet enough to drink from (spec)
SWIM_SALT = 0.5                            # sea water rehydrates only above this swim gene (spec)
# ------------------------------------------------------------------------------------ food
PLANT_INTAKE = 0.08                        # kg dry matter per day x M^0.75 (spec); also the meat cap
PLANT_J_KG = 18.5e6                        # J/kg dry forage, gross (spec)
PLANT_DIG0, PLANT_DIG_DIET = 0.75, 0.5     # digestible share 0.75 - 0.5 diet (spec)
PLANT_C_FRACTION = 0.47                    # kg C per kg dry plant (IPCC 2006 vol. 4 default 0.47)
FORAGE_WATER_KG = 3.0                      # kg water per kg dry forage eaten (green forage about 75 % water)
FAT_J_KG = float(materials.FOOD_J_KG["fat"])   # 39.5e6 J/kg (Blaxter 1989)
FAT_C_PER_J = _carbon("fat") / FAT_J_KG    # derived: kg C per J of reserve (tripalmitin)
RESERVE_MAX_FRACTION = 0.3                 # fat at most 0.3 M (spec)
COOKED_K = 343.15                          # an item counts as cooked once its peak reached 70 C
REACH_M = 5.0                              # new_rule: food items on the ground within this reach
SCRAP_KG = 1e-3                            # numerics: an item eaten down below 1 g is cleared
# ------------------------------------------------------------------------------------ body
BODY_MIX = {"meat": 0.84, "bone": 0.10, "hide": 0.06}
CORPSE = {"meat": 0.45, "bone": 0.10, "hide": 0.06}   # spec; the rest of the lean body goes to litter
TISSUE_J_KG = sum(x * float(materials.FOOD_J_KG[s]) for s, x in BODY_MIX.items())   # 7.22e6
TISSUE_C_KG = sum(x * _carbon(s) for s, x in BODY_MIX.items())                     # 0.156
WATER_FRACTION = sum(x * _water(s) for s, x in BODY_MIX.items())                   # 0.668
GROWTH_J_KG = TISSUE_C_KG / FAT_C_PER_J    # 8.10e6: the fat that holds a kg of tissue's carbon
FAT_MIN_KG = SCRAP_KG                      # numerics: corpse parts below a gram go to litter
# ------------------------------------------------------------------------------------ life history
LIFESPAN_YR, LIFESPAN_EXP = brain.LIFESPAN_YR, brain.LIFESPAN_EXP   # 11.6 yr M^0.20 (Calder 1984)
MATURITY_YR, MATURITY_EXP = 0.6, 0.27      # Calder 1984 / Peters 1983 (spec)
GESTATION_D, GESTATION_EXP = 65.0, 0.25    # days (spec)
LITTER_MASS, LITTER_EXP = 0.1, 0.92        # kg x M^0.92 per litter (spec)
VB_MATURE_FRACTION = 0.9                   # new_rule: von Bertalanffy reaches 90 % of adult at maturity
GROWTH_FLOOR = 0.25                        # new_rule: juveniles grow only from reserve above 1/4 of max
CHILD_RESERVE = 0.5                        # new_rule: a newborn's reserve, share of its own maximum
LEAN_LOSS_LETHAL = 0.4                     # starvation kills at 40 % lean mass lost
FOUNDER_MASS_KG = 30.0                     # new_rule: founders' adult mass (brain.REF_MASS_KG)
FOUNDER_RESERVE = 0.5                      # new_rule: founders start at half their maximum reserve
FOUNDER_SPEED, FOUNDER_SWIM, FOUNDER_LITTER = 1.0, 0.0, 2
SENSES_DEFAULT = {"eyes": 1.0, "ears": 1.0, "talkers": 0.1}   # new_rule when a chain has no senses era
# ------------------------------------------------------------------------------------ calls
CALL_P1M_PA = 2.0                          # Pa at 1 m (100 dB SPL), full effort, 30 kg, Earth air
CALL_REF_KG, CALL_MASS_EXP = 30.0, 0.5     # pressure ~ M^0.5 (acoustic power ~ M)
CALL_S = 600.0                             # new_rule: seconds of calling per day at the chosen loudness
VOCAL_EFF = 0.01                           # acoustic over metabolic power of vocalising
RHO_AIR_EARTH, C_SOUND_EARTH = 1.204, 343.2   # kg/m^3, m/s: dry air at 20 C, 101.325 kPa (CRC)

CAUSES = ("none", "starvation", "dehydration", "injury", "age", "hypoxia", "cold")
CAUSE = {c: i for i, c in enumerate(CAUSES)}
BODY_GENES = ("adult_mass_kg", "diet", "insulation", "speed", "acuity", "hearing", "voice", "swim", "litter")
GS = brain.GeneSpec
BODY_SPECS = {
    "adult_mass_kg": GS(0.01, 5000.0, 0.03, "log", "new_rule: 3 % log-normal step, bounds 10 g to 5 t"),
    "diet": GS(0.0, 1.0, 0.03, "range", "spec 2.8: 0 plant to 1 meat"),
    "insulation": GS(0.2, 3.0, 0.03, "range", "spec 2.8: [0.2, 3]"),
    "speed": GS(0.3, 1.2, 0.03, "range", "spec 2.8: [0.3, 1.2]"),
    "acuity": GS(0.0, 5.0, 0.03, "range", "world7 sensor level scale 0-5 (evo/senses.py)"),
    "hearing": GS(0.0, 5.0, 0.03, "range", "world7 sensor level scale 0-5 (evo/senses.py)"),
    "voice": GS(0.0, 1.5, 0.03, "range", "flat prototype voice 0-1.5 (docs/life9/PLAN.md A2)"),
    "swim": GS(0.0, 1.0, 0.03, "range", "spec 2.8: [0, 1]"),
    "litter": GS(1, 6, 0.03, "step", "spec 2.8: 1-6 offspring, +-1 with probability 0.03"),
}
GENE_SPECS = {**brain.BRAIN_SPECS, **BODY_SPECS}
LEDGER = ("spawned_j", "food_j", "respired_j", "dead_j",
          "spawned_c", "food_c", "co2_c", "faeces_c", "dead_c",
          "spawned_w", "drunk_w", "food_w", "lost_w", "dead_w", "metabolic_w",
          "o2_mol", "oxidised_c",
          "births", *(f"deaths_{c}" for c in CAUSES[1:]))

R_, D_, N_, C_ = "reference", "derived", "new_rule", "chain"
PROVENANCE = {
    "KLEIBER_W": (R_, "Kleiber 1947, Physiol. Rev. 27:511: 3.4 W x M^0.75 (spec 2.8)"),
    "COT_J_KG_M": (R_, "Taylor, Heglund & Maloiy 1982, J. Exp. Biol. 97:1: minimum cost of transport "
                       "10.7 M^-0.316 J/(kg m)"),
    "MUSCLE_EFF": (R_, "positive-work efficiency of muscle about 0.25 (Margaria 1976, Biomechanics and Energetics "
                       "of Muscular Exercise; spec 2.8 climbing M g dh / 0.25); also the muscle factor of swimming"),
    "V_TROT": (R_, "Heglund, Taylor & McMahon 1974, Science 186:1112: trot-gallop transition 1.53 M^0.24 m/s, "
                   "the top economical speed; x speed gene x aerobic capacity (spec 2.8)"),
    "MOVE_S": (N_, "2 h of travel per day at most: 24.6 km/day at 30 kg (Garland 1983, Am. Nat. 121:571, gives a "
                   "mean daily movement of 1.04 M^0.25 km, 2.4 km at 30 kg)"),
    "MAX_TURN": (N_, "a day's travel may start in any direction: turn in [-1, 1] x pi"),
    "shore_stop": (N_, "move(shore_stop=True): a land animal (swim gene below SWIM_SALT) that starts on land does "
                       "not walk out to sea: its day's travel ends at the last path sample before the shore"),
    "PATH_SAMPLES_MAX": (N_, "numerics: the path is sampled once per smallest cell spacing (globe.min_spacing) of the "
                             "longest travel that day, at most 256 samples, evaluated PATH_BLOCK = 32 at a time"),
    "climb_surface": (N_, "the climb is the sum of rises on land of the smooth surface sight lines see (senses."
                          "surface_at, bilinear within cells from globe.at_vertices), so movement and sight share one "
                          "terrain and the climb converges with the samples (cell steps would count every corner a "
                          "path clips)"),
    "MEEH_K": (R_, "Meeh 1879 body surface A = k M^(2/3), k about 0.1 m^2 kg^-2/3 for mammals (spec 2.8)"),
    "SWIM_CD": (R_, "surface swimming drag on the wetted area: human active drag about 29 v^2 N over 1.8 m^2 of "
                    "skin, C_d about 0.03 (Toussaint et al. 1988, J. Appl. Physiol. 65:2506); semi-aquatic "
                    "mammals paddle with more wave drag (Fish 1993, Aust. J. Zool. 42:79); 0.04 (approximate)"),
    "ETA_PADDLE": (R_, "paddling propulsive efficiency about 0.33 (Fish 1984, J. Comp. Physiol. B 154:91, muskrat)"),
    "ETA_LIFT": (R_, "lift-based swimmers 0.75-0.9 (Fish 1993, Aust. J. Zool. 42:79); swim gene interpolates"),
    "swim_cost": ("derived", "metabolic swim cost = 0.5 rho_w C_d A v^2 d / (MUSCLE_EFF x eta): the spec's drag "
                             "work turned into metabolic energy (deviation: the spec books the bare drag work)"),
    "SWIM_LAND_PENALTY": (N_, "land cost of transport x (1 + swim): the swim gene's trade-off (seals walk badly)"),
    "AEROBIC_SCOPE": (R_, "maximal aerobic power about 10 x basal in mammals (Taylor et al. 1981, Respir. Physiol. "
                          "44:25); x a(pO2) caps thermogenesis and the swimming speed"),
    "T_LC0": (N_, "spec 2.8 modelling choice: lower critical temperature 303 K - 8 x insulation (T_LC_PER_INS), "
                  "independent of body mass"),
    "C_TH": (R_, "minimal thermal conductance of mammals, Herreid & Kessel 1967, Comp. Biochem. Physiol. 21:405: "
                 "C = 1.02 W_g^-0.505 ml O2 g^-1 h^-1 C^-1 = 0.174 M^0.495 W/K at 20.1 J/ml O2, used as "
                 "0.174 M^0.5 / insulation. Deviation: spec 2.8 has 1.0 M^0.5 / insulation, about 5.7 x the "
                 "published conductance, which made cold about 5.7 x too costly"),
    "T_BODY": (R_, "mammalian core about 37 C = 310 K (Clarke & Rothery 2008, Funct. Ecol. 22:58, mean of mammals "
                   "36.6 C); at or above it no heat can be lost dry"),
    "evaporation": (D_, "above T_BODY every joule respired that day plus the heat gained from the air "
                        "C_th (T - T_BODY) leaves as evaporated water at LATENT_J_KG (deviation: spec 2.8 evaporates "
                        "only the environmental excess; dry heat loss is impossible once the air is as warm as the "
                        "body, so the metabolic heat must be evaporated too)"),
    "LATENT_J_KG": (R_, "latent heat of vaporisation of water at 30-35 C, 2.43e6 J/kg (CRC; spec 2.8)"),
    "NEURON_COST_W": (N_, "spec 2.8: 0.002 W per active brain unit"),
    "SENSE_W": (C_, "sensor upkeep SENSE_W x M^0.75 x (2 acuity^2 + hearing^2): world7's L^2 sensor cost and unit "
                    "costs (evo/senses.py:472, UNIT_COST eyes 2, ears 1), scaled so the 781 founders' senses "
                    "(eyes 2.04, ears 0.685) cost the senses era's 7.4 % (e_sensors 2613 / e_intake 35297) of basal"),
    "AEROBIC_HALF_PA": (N_, "spec 2.8 modelling choice: aerobic capacity a = pO2 / (pO2 + 3 kPa) scales top speed "
                            "and strike power"),
    "HYPOXIA_PA": (N_, "spec 2.8 modelling choice: below 1 kPa O2 health falls"),
    "HYPOXIA_PER_DAY": (N_, "health lost per day = 2 x (1 - pO2 / 1 kPa): death within a day below 0.5 kPa"),
    "COLD_PER_DAY": (N_, "hypothermia: health lost per day = the share of the heat loss the aerobic ceiling cannot "
                         "cover (cause 'cold', added to the spec's causes)"),
    "HEAL_PER_DAY": (N_, "0.05 health per day (about 3 weeks to heal) when alive at health > 0, fed (reserve > 0), "
                         "watered (>= 0.9 x norm) and unharmed that day: no strike since the last physiology "
                         "(``struck``), no hypoxia, no cold; so a lethal blow is never healed away before deaths()"),
    "OXY_J_MOL": (R_, "oxycaloric equivalent of a mixed diet 20.1 kJ per litre O2 (Schmidt-Nielsen 1997, Animal "
                      "Physiology, 5th ed., ch. 5) x 22.4 L/mol = 4.5e5 J/mol (spec 2.8)"),
    "respiration_co2": ("derived", "CO2 = the carbon oxidised: reserve joules x FAT_C_PER_J (RQ 0.72) plus the "
                                   "conversion surplus of food and catabolised tissue (deviation from the spec's "
                                   "fixed RQ 0.85: the carbon ledger closes; a grazer's overall RQ is about 0.95)"),
    "respiration_o2": ("derived", "O2 = the reserve joules oxidised / OXY_J_MOL; the heat of tissue synthesis (growth "
                                  "and offspring, GROWTH_J_KG - TISSUE_J_KG per kg) takes no O2: no carbon is oxidised "
                                  "there, the oxygen the tissue gains over fat comes with the food (untracked like its N)"),
    "metabolic_water": ("derived", "the oxygen of the respired O2 that does not return as CO2 leaves as water, "
                                   "2 (O2 - CO2_oxidised) mol H2O (x M_H2O kg/mol), into body water (booked metabolic_w), with hydrogen "
                                   "from the oxidised fat; the fat's own oxygen is not tracked, so this is about 0.56 "
                                   "mol per mol O2 against 0.66 for tripalmitin (1.07 g water per g fat, Schmidt-Nielsen "
                                   "1997, Animal Physiology): a lower bound. Nagy & Peterson's flux includes it"),
    "FORAGE_WATER_KG": (R_, "fresh green forage is about 75 % water, 3 kg per kg dry matter (pasture herbage 15-25 % "
                            "dry matter, e.g. NRC 2001 Nutrient Requirements of Dairy Cattle; approximate, from "
                            "memory); the eater takes it with the plants from the cell's soil water (the world's "
                            "callback), so the water ledger closes"),
    "WATER_TURNOVER": (R_, "Nagy & Peterson 1988, Scaling of Water Flux Rate in Animals (UC Publ. Zool. 120): "
                           "0.12 M^0.82 kg/day (spec 2.8)"),
    "DEHYDRATION_LETHAL": (N_, "spec 2.8 modelling choice: death below 0.8 x the water norm"),
    "DRINK_SOIL_MIN": (N_, "spec 2.8 modelling choice: drinking where soil > 50 kg/m^2 or a lake"),
    "excretion": (N_, "after the day's turnover and evaporation, body water above the norm leaves as urine "
                      "(physiology's excreted_kg, to the ground): forage and metabolic water cannot fill a body "
                      "beyond its norm"),
    "drink_day": (N_, "a drinker fills to the water norm plus the day's turnover 0.12 M^0.82 (drinking is spread over "
                      "the day the physiology then books as lost) plus, given the air temperature, the evaporation of "
                      "its basal heat above T_BODY; otherwise animals below about 1 kg would lose more than the lethal "
                      "20 % between two daily drinks, and any animal in air above 310 K would dry out though watered"),
    "SWIM_SALT": (N_, "spec 2.8 modelling choice: sea water rehydrates only with swim > 0.5 (salt glands); "
                      "otherwise no gain"),
    "PLANT_INTAKE": (N_, "spec 2.8 modelling choice: 0.08 M^0.75 kg dry matter per day; the same dry-matter cap "
                         "applies to items"),
    "PLANT_J_KG": (R_, "spec 2.8: forage gross energy 18.5 MJ/kg dry (forages 17.5-19.5, Minson 1990)"),
    "PLANT_DIG0": (N_, "spec 2.8 modelling choice: plant digestibility 0.75 - 0.5 diet"),
    "PLANT_C_FRACTION": (R_, "dry plant matter 47 % carbon (IPCC 2006 Guidelines vol. 4, Table 4.3 default); "
                             "the biosphere must convert kg dry and kg C with the same fraction"),
    "FAT_J_KG": (R_, "fat 39.5 MJ/kg (Blaxter 1989); materials.FOOD_J_KG['fat']"),
    "FAT_C_PER_J": ("derived", "tripalmitin carbon 0.759 kg/kg over 39.5 MJ/kg: the reserve's carbon per joule"),
    "RESERVE_MAX_FRACTION": (N_, "spec 2.8 modelling choice: fat reserve at most 0.3 M x 39.5 MJ/kg (the "
                                 "harvester's float32 rounding of one day's intake may overshoot it slightly)"),
    "COOKED_K": (R_, "meat counts as cooked once its peak temperature reached 70 C (USDA FSIS safe minimum "
                     "internal temperatures 63-74 C); cooking gain from materials.COOK_GAIN"),
    "meat_digestible": ("derived", "items: materials.DIGESTIBLE interpolated in diet (meat 0.4 + 0.5 diet, spec) x "
                                   "COOK_GAIN if cooked, capped at 1 of the gross energy (deviation: the spec's "
                                   "0.9 x 1.3 = 1.17 would digest more than the gross energy)"),
    "REACH_M": (N_, "food items on the ground within 5 m can be eaten (held items always)"),
    "SCRAP_KG": (N_, "numerics: an item eaten below 1 g is removed (its remainder reported as scraps); corpse parts "
                     "below 1 g (FAT_MIN_KG) go to litter"),
    "BODY_MIX": (N_, "the lean body as carcass species: meat 0.84 (muscle 0.45 plus viscera 0.39 counted as "
                     "meat-like tissue), bone 0.10, hide 0.06 (the spec's corpse shares)"),
    "TISSUE_J_KG": ("derived", f"gross energy of BODY_MIX from materials.FOOD_J_KG: {TISSUE_J_KG:.4e} J/kg"),
    "TISSUE_C_KG": ("derived", f"carbon of BODY_MIX from materials.MAKEUP: {TISSUE_C_KG:.5f} kg C/kg"),
    "WATER_FRACTION": ("derived", f"water of BODY_MIX: {WATER_FRACTION:.4f} (total body water of mammals about "
                                  "0.6-0.7 of body mass, 0.73 of fat-free mass, Pace & Rathbun 1945)"),
    "GROWTH_J_KG": ("derived", f"{GROWTH_J_KG:.4e} J/kg: the fat holding a kg of tissue's carbon; "
                               f"{GROWTH_J_KG / TISSUE_J_KG - 1:.3f} of it is the heat cost of growth"),
    "CORPSE": (N_, "spec 2.8 modelling choice: meat 0.45 M, bone 0.1 M, hide 0.06 M, fat from the reserve; the "
                   "remaining viscera go to litter carbon (and their water to the environment)"),
    "LIFESPAN_YR": (R_, "Calder 1984, Size, Function and Life History: maximum lifespan 11.6 yr x M^0.20; death at "
                        "that age (deterministic)"),
    "MATURITY_YR": (R_, "spec 2.8 (Calder 1984; Peters 1983): maturity 0.6 yr x M^0.27"),
    "GESTATION_D": (N_, "spec 2.8 modelling choice: gestation 65 d x M^0.25 (quarter-power like the other life-history "
                        "times), the cooldown after a birth"),
    "LITTER_MASS": (R_, "spec 2.8 (Peters 1983): litter mass 0.1 M^0.92"),
    "VB_MATURE_FRACTION": (N_, "von Bertalanffy growth (dm/dt = 3k m^2/3 (M^1/3 - m^1/3), stepped exactly per day) "
                               "with k such that 90 % of adult mass is reached at the maturity age"),
    "GROWTH_FLOOR": (N_, "juveniles grow only from reserve above 1/4 of its maximum"),
    "CHILD_RESERVE": (N_, "a newborn gets half its own maximum reserve, paid by the parent"),
    "breed_water": (N_, "a parent breeds only if its water after paying the litter's (WATER_FRACTION x litter mass) "
                        "stays at or above DEHYDRATION_LETHAL x its norm"),
    "breed_order": (N_, "when a world has fewer free slots than newborns, the ready parents are served in a random "
                        "order drawn from the world's generator (not by slot index)"),
    "LEAN_LOSS_LETHAL": (R_, "death when 40 % of the largest lean mass is lost: starvation kills after about a "
                             "third to a half of body protein is gone (Cahill 2006, Annu. Rev. Nutr. 26:1)"),
    "FOUNDER_MASS_KG": (N_, "founders' adult mass 30 kg: world7 bodies have cells, not kg (256 cells for 781); the "
                            "brain's learning-rate reference body (brain.REF_MASS_KG)"),
    "FOUNDER_RESERVE": (N_, "founders start at half their maximum reserve, full water, health 1, as young adults "
                            "(age = maturity) with a uniform random cooldown within one gestation"),
    "founder_diet": (N_, "proxy: the bodies era's predator_share (781: 0.034), which world7 defines as the share "
                         "of ALL biomass in predators (world7.py:574; 781: 2.9 / (56.4 + 26.1 + 2.9)). The consumers' "
                         "meat share would be predators / (grazers + predators) (781: 2.9 / 29.0 = 0.10), but the "
                         "chain export (chain.py) does not carry the guild biomasses"),
    "founder_insulation": ("derived", "(303 - chain T_s) / 8 clamped to [0.2, 3]: lower critical temperature = "
                                      "the chain's surface temperature (781: 1.875)"),
    "founder_acuity": (C_, "senses era eyes level (781: 2.04); world7 sensor range = reach x level "
                           "(evo/senses.py:393), so range is proportional to acuity"),
    "founder_hearing": (C_, "senses era ears level (781: 0.685)"),
    "founder_voice": (C_, "senses era talkers share (781: 0.08) as the founders' mean voice"),
    "founder_speed": (N_, "1.0: the allometric animal (no speed in the chain's senses era record)"),
    "founder_swim": (N_, "0: land animals"),
    "founder_litter": (N_, "2 offspring per litter"),
    "founder_innate": (C_, "brain.random_genome innate forage bias on the local plant input (spec 2.10)"),
    "SENSES_DEFAULT": (N_, "eyes 1, ears 1, talkers 0.1 when the chain reached no senses era"),
    "BODY_SPECS": (N_, "body genes mutate at a scale of 3 % (spec 2.8); bounds as in BODY_SPECS sources"),
    "CALL_P1M_PA": (R_, "a loud mammal call is about 100 dB SPL at 1 m (2 Pa; dog barks 100-110 dB) for 30 kg; "
                        "pressure ~ M^0.5 since acoustic power ~ M (Fletcher 2004, JASA 115:2334), approximate"),
    "call_air": ("derived", "monopole source: p(r) = rho omega Q / (4 pi r), so the pressure scales with air density "
                            "(x rho / 1.204); acoustic power 4 pi p1^2 / (rho c)"),
    "CALL_S": (N_, "600 s of calling per day at the chosen loudness"),
    "VOCAL_EFF": (R_, "vocal efficiency of mammals about 0.1-2 % (Fletcher 1992, Acoustic Systems in Biology); 1 %"),
    "RHO_AIR_EARTH": (R_, "dry air at 20 C and 101.325 kPa: 1.204 kg/m^3, c = 343.2 m/s (CRC Handbook)"),
}


# ------------------------------------------------------------------------------------ state
@dataclass
class Creatures:
    """Individuals of W worlds in N slots: per-slot tensors [W, N] (``pos`` [W, N, 3] unit vectors,
    ``inv`` [W, N, K] item indices or -1, ``calls`` [W, N, D] the last vocal vector, ``loud`` its
    loudness in [0, 1]), the genome dict (brain genes and :data:`BODY_GENES`, each [W, N, ...]),
    live recurrent weights, brain state and modulator baseline, the next uid per world [W] and
    the cumulative ledger dict (float64 [W], keys :data:`LEDGER`). ``mass_kg`` is the lean body,
    ``frame_kg`` the largest lean mass reached, ``harm`` the cause code of the last health loss,
    ``struck`` the health lost to strikes since the last :func:`physiology`."""
    alive: torch.Tensor
    pos: torch.Tensor
    heading: torch.Tensor
    mass_kg: torch.Tensor
    frame_kg: torch.Tensor
    reserve_j: torch.Tensor
    water_kg: torch.Tensor
    health: torch.Tensor
    age_d: torch.Tensor
    cooldown_d: torch.Tensor
    uid: torch.Tensor
    parent: torch.Tensor
    founder: torch.Tensor
    generation: torch.Tensor
    harm: torch.Tensor
    struck: torch.Tensor
    inv: torch.Tensor
    calls: torch.Tensor
    loud: torch.Tensor
    genome: dict
    wh_live: torch.Tensor
    brain_state: torch.Tensor
    baseline: torch.Tensor
    next_uid: torch.Tensor
    ledger: dict = field(default_factory=dict)

    TENSORS = ("alive", "pos", "heading", "mass_kg", "frame_kg", "reserve_j", "water_kg", "health", "age_d",
               "cooldown_d", "uid", "parent", "founder", "generation", "harm", "struck", "inv", "calls", "loud",
               "wh_live", "brain_state", "baseline", "next_uid")

    @property
    def shape(self):
        return tuple(self.alive.shape)

    @property
    def device(self):
        return self.alive.device

    @property
    def K(self):
        return self.inv.shape[-1]

    def gene(self, name):
        return self.genome[name]

    def state_dict(self):
        out = {name: getattr(self, name).clone() for name in self.TENSORS}
        out["genome"] = {k: v.clone() for k, v in self.genome.items()}
        out["ledger"] = {k: v.clone() for k, v in self.ledger.items()}
        return out

    @classmethod
    def from_state(cls, state, device=None):
        mv = (lambda t: t.to(device).clone()) if device is not None else (lambda t: t.clone())
        kw = {name: mv(state[name]) for name in cls.TENSORS}
        return cls(**kw, genome={k: mv(v) for k, v in state["genome"].items()},
                   ledger={k: mv(v) for k, v in state["ledger"].items()})


def state_hash(cr: Creatures) -> str:
    """SHA-256 over every tensor of the state (name, dtype, shape and bytes), for determinism checks."""
    h = hashlib.sha256(b"Creatures")
    sd = cr.state_dict()
    flat = [(n, sd[n]) for n in Creatures.TENSORS] + [(f"genome.{k}", v) for k, v in sorted(sd["genome"].items())]
    for name, t in flat:
        t = t.detach().cpu().contiguous()
        h.update(f"|{name}|{t.dtype}|{tuple(t.shape)}|".encode())
        if t.numel():
            h.update(ctypes.string_at(t.data_ptr(), t.numel() * t.element_size()))
    return h.hexdigest()


# ------------------------------------------------------------------------------------ allometry
def basal_w(m):
    return KLEIBER_W * m.clamp_min(1e-9) ** 0.75


def reserve_max(m):
    """Largest fat reserve (J): 0.3 M x 39.5 MJ/kg."""
    return RESERVE_MAX_FRACTION * m * FAT_J_KG


def water_norm(m):
    return WATER_FRACTION * m


def lifespan_d(adult):
    return LIFESPAN_YR * adult.clamp_min(1e-9) ** LIFESPAN_EXP * YEAR_S / DAY_S


def maturity_d(adult):
    return MATURITY_YR * adult.clamp_min(1e-9) ** MATURITY_EXP * YEAR_S / DAY_S


def gestation_d(adult):
    return GESTATION_D * adult.clamp_min(1e-9) ** GESTATION_EXP


def litter_mass_kg(m):
    return LITTER_MASS * m.clamp_min(1e-9) ** LITTER_EXP


def vb_k(adult, m0):
    """von Bertalanffy rate (per day) reaching VB_MATURE_FRACTION x adult at the maturity age from m0."""
    b = 1 - (m0 / adult).clamp(1e-9, 0.999) ** (1 / 3)
    gap = 1 - VB_MATURE_FRACTION ** (1 / 3)
    return torch.log((b / gap).clamp_min(1.0 + 1e-6)) / maturity_d(adult)


def aerobic(p_o2_pa):
    return p_o2_pa / (p_o2_pa + AEROBIC_HALF_PA)


def call_pressure_1m(loud, voice, mass_kg, rho_air):
    """Sound pressure (Pa) at 1 m of a call: CALL_P1M_PA x loud x voice x (M / 30)^0.5 x rho / rho_Earth."""
    size = (mass_kg.clamp_min(1e-9) / CALL_REF_KG) ** CALL_MASS_EXP
    return CALL_P1M_PA * loud.clamp(0, 1) * voice.clamp_min(0) * size * rho_air / RHO_AIR_EARTH


def body_energy_j(mass, reserve):
    return mass.double() * TISSUE_J_KG + reserve.double()


def body_carbon_kg(mass, reserve):
    return mass.double() * TISSUE_C_KG + reserve.double() * FAT_C_PER_J


# ------------------------------------------------------------------------------------ helpers
def per_world(x, W, device, dtype=torch.float32):
    """A number or [W] tensor as [W, 1]."""
    return torch.as_tensor(x, dtype=dtype, device=device).reshape(-1).expand(W).reshape(W, 1)


def local(globe, value, pos):
    """``value`` at each individual: [W] per world or [W, C] per cell (the containing cell) -> [W, N]
    in the state's dtype (float32), whatever the dtype passed in (climate pressures are float64)."""
    value = torch.as_tensor(value, device=pos.device).to(pos.dtype)
    if value.dim() <= 1:
        W, N = pos.shape[:2]
        return value.reshape(-1).expand(W).reshape(W, 1).expand(W, N)
    return globe.sample(value, pos)


def _f32(x, device=None):
    """A field or argument as float32 (the state's dtype); None stays None."""
    return None if x is None else torch.as_tensor(x, device=device).float()


def surface_m(terrain):
    """Height of the walkable or swimmable surface [W, C]: the ground, or the sea surface over the ocean."""
    return torch.maximum(terrain["elevation_m"], terrain["sea_level_m"][:, None])


def _book(cr, name, value):
    cr.ledger[name] += value.double().reshape(value.shape[0], -1).sum(-1)


def new_ledger(W, device):
    return {k: torch.zeros(W, dtype=torch.float64, device=device) for k in LEDGER}


def share_scatter(stock, w, cell, request):
    """Fair share of a per-cell stock (kg [W, C], modified in place) among requests (w [M], cell [M],
    request [M] kg): each gets request x min(1, stock / total requested in its cell). Returns the
    kg given [M] (never negative, the cell never below zero)."""
    W, C = stock.shape
    key = w.long() * C + cell.long()
    req = request.clamp_min(0).to(stock.dtype)
    total = torch.zeros(W * C, dtype=torch.float64, device=stock.device).index_add_(0, key, req.double())
    avail = stock.reshape(-1).double().clamp_min(0)
    frac = torch.where(total > 0, (avail / total.clamp_min(1e-300)).clamp(max=1.0), torch.zeros_like(total))
    given = (req.double() * frac[key]).to(stock.dtype)
    taken = torch.zeros(W * C, dtype=torch.float64, device=stock.device).index_add_(0, key, given.double())
    stock.copy_((avail - taken).clamp_min(0).reshape(W, C).to(stock.dtype))
    return given


def field_harvester(stock_kg):
    """A harvest callback over a per-cell stock of kg dry plant matter [W, C] (for tests and tools)."""
    def harvest(w, cell, kg):
        return share_scatter(stock_kg, w, cell, kg)
    return harvest


# ------------------------------------------------------------------------------------ founders
def founder_genes(inputs: dict) -> dict:
    """Founder body genes of one planet from its chain inputs: name -> (value, tag, note)."""
    bio = inputs.get("biosphere") or {}
    sen = bio.get("senses") or None
    t_s = float(inputs["planet"]["t_surface_k"])
    src = "chain" if sen else "new_rule"
    sen = sen or SENSES_DEFAULT
    return {
        "adult_mass_kg": (FOUNDER_MASS_KG, N_, PROVENANCE["FOUNDER_MASS_KG"][1]),
        "diet": (float(bio.get("predator_share", 0.0)), N_, PROVENANCE["founder_diet"][1]),
        "insulation": (min(3.0, max(0.2, (T_LC0 - t_s) / T_LC_PER_INS)), D_, PROVENANCE["founder_insulation"][1]),
        "speed": (FOUNDER_SPEED, N_, PROVENANCE["founder_speed"][1]),
        "acuity": (float(sen.get("eyes", SENSES_DEFAULT["eyes"])), src, PROVENANCE["founder_acuity"][1]),
        "hearing": (float(sen.get("ears", SENSES_DEFAULT["ears"])), src, PROVENANCE["founder_hearing"][1]),
        "voice": (float(sen.get("talkers", SENSES_DEFAULT["talkers"])), src, PROVENANCE["founder_voice"][1]),
        "swim": (FOUNDER_SWIM, N_, PROVENANCE["founder_swim"][1]),
        "litter": (FOUNDER_LITTER, N_, PROVENANCE["founder_litter"][1]),
    }


def empty(W, N, *, in_dim=None, hidden=None, genome=None, initial_hidden=2, K=3, D=brain.VOCAL_DIMS, device="cpu",
          params=None, weight_dtype=torch.float32):
    """W x N empty slots (all dead). ``genome`` (brain genes [W, N, ...]) is used as given and gets the
    body genes added; without it a genome is drawn from a fixed generator (seed 0, for tests and
    tools), never from a world's generator."""
    if genome is None:
        gen = torch.Generator(device=device).manual_seed(0)
        genome = brain.random_genome(gen, (W, N), in_dim, hidden, brain.OUT_DIM, initial_hidden, device,
                                     dtype=weight_dtype, params=params)
    g = genome
    hidden = g["Wh"].shape[-1]
    f = lambda: torch.zeros(W, N, device=device)
    lg = lambda v=-1: torch.full((W, N), v, dtype=torch.long, device=device)
    for name in BODY_GENES:
        if name not in g:
            g[name] = torch.ones(W, N, dtype=torch.long, device=device) if name == "litter" else f()
    pos = torch.zeros(W, N, 3, device=device)
    pos[..., 2] = 1.0
    return Creatures(alive=torch.zeros(W, N, dtype=torch.bool, device=device), pos=pos,
                     heading=f(), mass_kg=f(), frame_kg=f(), reserve_j=f(), water_kg=f(), health=f(), age_d=f(),
                     cooldown_d=f(), uid=lg(), parent=lg(), founder=lg(), generation=lg(0), harm=lg(0), struck=f(),
                     inv=torch.full((W, N, K), -1, dtype=torch.long, device=device),
                     calls=torch.zeros(W, N, D, device=device), loud=f(), genome=g,
                     wh_live=brain.init_live(g, dtype=weight_dtype),
                     brain_state=brain.init_state((W, N), hidden, device), baseline=f(),
                     next_uid=torch.zeros(W, dtype=torch.long, device=device), ledger=new_ledger(W, device))


def place(cr, w, n, *, pos, heading, mass, reserve, water, age, cooldown, genes=None):
    """Make slots (w, n) alive with the given body (values broadcast to [M]); books 'spawned' ledgers.
    The slots' live weights restart from their genome's Wh, brain state and baseline from zero, so
    nothing a previous occupant learned stays in the slot."""
    dev = cr.device
    w, n = w.long(), n.long()
    M = w.shape[0]
    b = lambda v: torch.as_tensor(v, dtype=torch.float32, device=dev).expand(M)
    cr.alive[w, n] = True
    cr.pos[w, n] = pos
    cr.heading[w, n] = b(heading)
    cr.mass_kg[w, n] = b(mass)
    cr.frame_kg[w, n] = b(mass)
    cr.reserve_j[w, n] = b(reserve)
    cr.water_kg[w, n] = b(water)
    cr.health[w, n] = 1.0
    cr.age_d[w, n] = b(age)
    cr.cooldown_d[w, n] = b(cooldown)
    cr.harm[w, n] = 0
    cr.struck[w, n] = 0.0
    cr.inv[w, n] = -1
    cr.calls[w, n] = 0.0
    cr.loud[w, n] = 0.0
    for name, v in (genes or {}).items():
        cr.genome[name][w, n] = torch.as_tensor(v, device=dev).to(cr.genome[name].dtype).expand(M)
    cr.wh_live[w, n] = cr.genome["Wh"][w, n].to(cr.wh_live.dtype)
    cr.brain_state[w, n] = 0.0
    cr.baseline[w, n] = 0.0
    rank = torch.nn.functional.one_hot(w, cr.shape[0])
    order = (rank.cumsum(0) * rank).sum(1) - 1
    cr.uid[w, n] = cr.next_uid[w] + order
    cr.founder[w, n] = cr.uid[w, n]
    cr.parent[w, n] = -1
    cr.generation[w, n] = 0
    cr.next_uid += rank.sum(0)
    _book_w(cr, w, "spawned_j", body_energy_j(cr.mass_kg[w, n], cr.reserve_j[w, n]))
    _book_w(cr, w, "spawned_c", body_carbon_kg(cr.mass_kg[w, n], cr.reserve_j[w, n]))
    _book_w(cr, w, "spawned_w", cr.water_kg[w, n])


def _book_w(cr, w, name, value):
    """Add per-request values [M] to the ledger of their worlds w [M]."""
    cr.ledger[name].index_add_(0, w.long(), value.double())


def spawn_founders(inputs, globe, terrain, gen, *, capacity, founders, in_dim=None, hidden=128, initial_hidden=48,
                   params=None, K=3, D=brain.VOCAL_DIMS, adult_mass_kg=FOUNDER_MASS_KG, weight_dtype=torch.float32,
                   cell_weight=None):
    """Founders of W worlds (one chain-input dict, or a list of W of them, one planet per world).

    ``terrain`` is :func:`globe.make_terrain` output for the same W. Each world gets ``founders``
    individuals in its first slots, on land cells drawn by area (anywhere if a world has no
    land), or, when ``cell_weight`` [W, C] (>= 0, e.g. a habitability mask from the world) is given
    and a world has a positive weight, on cells drawn by area x that weight, with the
    chain-derived genes of :func:`founder_genes` and a brain from
    :func:`brain.random_genome` with the innate forage wiring on ``senses.PLANT_INDEX``. Each
    founder lies at a uniform point of its cell's gnomonic square (the cell centre in the rare
    case float32 rounding puts the point across the cell's edge), so it is on the drawn cell. Draw
    order: cells, jitter, headings, cooldowns, then the genome.
    """
    from . import senses
    dev = globe.device
    W = terrain["land"].shape[0]
    if isinstance(inputs, dict):
        inputs = [inputs] * W
    if len(inputs) != W:
        raise ValueError(f"{len(inputs)} chain inputs for {W} worlds")
    if not 1 <= founders <= capacity:
        raise ValueError("founders must lie within 1 and capacity")
    in_dim = senses.IN_DIM if in_dim is None else in_dim
    land = terrain["land"].double() * globe.area64
    weight = torch.where(land.sum(-1, keepdim=True) > 0, land, globe.area64.expand(W, -1))
    if cell_weight is not None:
        habit = torch.as_tensor(cell_weight, device=dev).double().clamp_min(0) * globe.area64
        weight = torch.where(habit.sum(-1, keepdim=True) > 0, habit, weight)
    cells = torch.multinomial(weight, founders, replacement=True, generator=gen)              # [W, F]
    jit = torch.rand(W, founders, 2, generator=gen, device=dev)
    frame = globe._frame                                                                     # face (n, u, v)
    face = globe.face[cells]
    a = -math.pi / 4 + (globe.i[cells] + jit[..., 0]) * globe.delta
    b = -math.pi / 4 + (globe.j[cells] + jit[..., 1]) * globe.delta
    pos = frame[face, 0] + torch.tan(a)[..., None] * frame[face, 1] + torch.tan(b)[..., None] * frame[face, 2]
    pos = pos / pos.norm(dim=-1, keepdim=True)
    pos = torch.where((globe.cell_of(pos) == cells)[..., None], pos, globe.centers[cells])
    heading = torch.rand(W, founders, generator=gen, device=dev) * TWO_PI
    phase = torch.rand(W, founders, generator=gen, device=dev)
    innate = {"plant_index": senses.PLANT_INDEX} if in_dim == senses.IN_DIM else None
    g = brain.random_genome(gen, (W, capacity), in_dim, hidden, brain.OUT_DIM, initial_hidden, dev, innate=innate,
                            dtype=weight_dtype, params=params)
    table = [founder_genes(x) for x in inputs]
    for name in BODY_GENES:
        vals = [adult_mass_kg] * W if name == "adult_mass_kg" else [t[name][0] for t in table]
        dtype = torch.long if name == "litter" else torch.float32
        g[name] = torch.tensor(vals, dtype=dtype, device=dev)[:, None].expand(W, capacity).clone()
    cr = empty(W, capacity, genome=g, K=K, D=D, device=dev, weight_dtype=weight_dtype)
    w = torch.arange(W, device=dev).repeat_interleave(founders)
    n = torch.arange(founders, device=dev).repeat(W)
    adult = g["adult_mass_kg"][w, n]
    place(cr, w, n, pos=pos.reshape(-1, 3), heading=heading.reshape(-1), mass=adult,
          reserve=FOUNDER_RESERVE * reserve_max(adult), water=water_norm(adult), age=maturity_d(adult),
          cooldown=phase.reshape(-1) * gestation_d(adult))
    return cr


# ------------------------------------------------------------------------------------ movement
def path_samples(globe, dist_m, radius_m, samples=None):
    """Samples of the day's paths: ``samples`` if given, else one per smallest cell spacing
    (``globe.min_spacing``) of the longest path, at least 1 and at most PATH_SAMPLES_MAX."""
    if samples is not None:
        return max(1, int(samples))
    theta = float((dist_m / radius_m).max()) if dist_m.numel() else 0.0
    return max(1, min(PATH_SAMPLES_MAX, math.ceil(theta / globe.min_spacing)))


def move(cr, globe, terrain, turn, speed, *, gravity_m_s2, p_o2_pa, radius_m, samples=None, surf_v=None,
         seconds=None, shore_stop=False):
    """The day's travel. turn [W, N] in [-1, 1] (x MAX_TURN), speed [W, N] in [0, 1] of the top speed.

    Top speed v = 1.53 M^0.24 x speed gene x a(pO2); distance = v x MOVE_S along the great circle
    of the new heading. The path is sampled at :func:`path_samples` points (one per smallest cell
    spacing of the day's longest path, unless ``samples`` is given): the climb is the sum of
    rises between the samples, on land, of the same smooth surface sight lines see
    (``senses.surface_at``: the ground, or sea level over the ocean, bilinear within each cell
    from its vertex values), and the share of samples over ocean cells is swum. In water the
    speed is capped where the drag power reaches the aerobic ceiling, which shortens the swum
    part. Costs (J): land COT x M x d x (1 + swim), climb M g rise / 0.25, swim 0.5 rho C_d A v^2
    d / (0.25 eta(swim)). Updates ``pos`` and ``heading`` of the living (one that does not move
    keeps its exact position); returns dist_m, land_m, swim_m, climb_m, speed_m_s, cost_j,
    in_water (bool, at the end point) and samples (int). Inputs of any float dtype. ``surf_v`` may
    pass ``globe.at_vertices(surface_m(terrain))`` when the caller keeps it (the terrain is static).
    ``seconds`` ([W, N] or a number) replaces MOVE_S as the day's travel time (the world's time
    budget). With ``shore_stop`` a land animal (swim gene below SWIM_SALT) that starts on land ends its
    travel at the last path sample before the sea (PROVENANCE['shore_stop']).
    """
    W, N = cr.shape
    dev = cr.device
    alive = cr.alive
    g = cr.genome
    M = cr.mass_kg.clamp_min(1e-6)
    radius = per_world(radius_m, W, dev)
    a = aerobic(local(globe, p_o2_pa, cr.pos))
    v_top = V_TROT * M ** V_TROT_EXP * g["speed"] * a
    s = _f32(speed, dev).clamp(0, 1) * alive
    v = s * v_top
    dist = v * (MOVE_S if seconds is None else _f32(seconds, dev))
    heading = torch.remainder(cr.heading + _f32(turn, dev).clamp(-1, 1) * MAX_TURN * alive, TWO_PI)
    from .senses import surface_at
    if surf_v is None:
        surf_v = globe.at_vertices(surface_m(terrain).float())
    ocean = ~terrain["land"]
    swim_g = g["swim"]
    P = path_samples(globe, dist, radius, samples)
    wn = torch.arange(W, device=dev)[:, None, None]
    h_prev = surface_at(globe, surf_v, wn[..., 0].expand(W, N), cr.pos)
    climb = torch.zeros(W, N, device=dev)
    wet = torch.zeros(W, N, device=dev)
    stopper = alive & (swim_g < SWIM_SALT) & ~ocean.gather(1, globe.cell_of(cr.pos)) if shore_stop else None
    met = torch.zeros(W, N, dtype=torch.bool, device=dev)          # a stopper's path has reached the sea
    stop_k = torch.zeros(W, N, device=dev)                         # its last land sample (index of 1..P)
    for k0 in range(1, P + 1, PATH_BLOCK):                          # blocks of samples (memory), not individuals
        frac = torch.arange(k0, min(P, k0 + PATH_BLOCK - 1) + 1, device=dev, dtype=torch.float32) / P
        B = frac.numel()
        pts, _ = globe.move(cr.pos[:, :, None, :].expand(W, N, B, 3), heading[..., None].expand(W, N, B),
                            dist[..., None] * frac, radius[..., None])
        sea = ocean.gather(1, globe.cell_of(pts).reshape(W, -1)).reshape(W, N, B)
        h = surface_at(globe, surf_v, wn.expand(W, N, B), pts)
        keep = torch.ones_like(sea)
        if shore_stop:
            hit = sea & stopper[..., None]
            keep = ~(met[..., None] | (hit.cumsum(-1) > 0))         # the samples before a stopper's shore
            new = hit.any(-1) & ~met
            stop_k = torch.where(new, (k0 - 1 + hit.float().argmax(-1)).float(), stop_k)
            met = met | new
        climb = climb + (torch.diff(h, dim=-1, prepend=h_prev[..., None]).clamp_min(0) * ~sea * keep).sum(-1)
        wet = wet + (sea & keep).float().sum(-1)
        h_prev = h[..., -1]
    wet = wet / P
    if shore_stop:
        dist = torch.where(met, dist * stop_k / P, dist)            # the walk ends at the shore
    eta = ETA_PADDLE + (ETA_LIFT - ETA_PADDLE) * swim_g
    area = MEEH_K * M ** (2 / 3)
    drag_k = 0.5 * RHO_WATER * SWIM_CD * area                       # N per (m/s)^2
    p_free = ((AEROBIC_SCOPE * a - 1) * basal_w(M)).clamp_min(0)
    v_cap = (p_free * MUSCLE_EFF * eta / drag_k).pow(1 / 3)
    v_w = torch.minimum(v, v_cap)
    swim_planned = dist * wet
    swim_m = swim_planned * torch.where(v > 0, v_w / v.clamp_min(1e-9), torch.zeros_like(v))
    land_m = dist - swim_planned
    dist_eff = land_m + swim_m
    climb = climb * torch.where(dist > 0, dist_eff / dist.clamp_min(1e-9), torch.zeros_like(dist))
    g_w = per_world(gravity_m_s2, W, dev)
    cost = (COT_J_KG_M * M ** COT_EXP * M * land_m * (1 + SWIM_LAND_PENALTY * swim_g)
            + M * g_w * climb / MUSCLE_EFF
            + drag_k * v_w ** 2 * swim_m / (MUSCLE_EFF * eta)) * alive
    p2, h2 = globe.move(cr.pos, heading, dist_eff, radius)
    went = alive & (dist_eff > 0)                                   # who stays keeps its exact position
    cr.pos = torch.where(went[..., None], p2, cr.pos)
    cr.heading = torch.where(went, h2, torch.where(alive, heading, cr.heading))
    in_water = ocean.gather(1, globe.cell_of(cr.pos)) & alive
    return {"dist_m": dist_eff * alive, "land_m": land_m * alive, "swim_m": swim_m * alive, "climb_m": climb * alive,
            "speed_m_s": v * alive, "cost_j": cost, "in_water": in_water, "samples": P}


# ------------------------------------------------------------------------------------ eating and drinking
def eat_plants(cr, mask, globe, harvest, *, c_fraction=PLANT_C_FRACTION, share=None, water=None, cell=None):
    """Graze where ``mask`` [W, N] (and alive). Each asks for min(0.08 M^0.75 kg dry x ``share``,
    what fits in the reserve) from ``harvest(w [M], cell [M], kg_dry [M]) -> kg_dry given [M]`` at
    its cell (``cell`` [W, N] if given, else the containing cell); ``share`` [W, N] in [0, 1] (default 1)
    is the part of a full grazing day spent grazing (the world's time budget).
    What the harvester gives is eaten and booked in full (biosphere.harvest returns each request's
    share of what left its float32 pool, which can exceed the request by that pool's rounding; so
    the biosphere's ``c_harvested`` and the eaters' ``food_c`` are the same carbon, and the reserve
    may overshoot its maximum by that rounding). Digested energy 18.5e6 x (0.75 - 0.5 diet) J/kg
    goes to the reserve; the undigested carbon is faeces (to the cell's litter) and the digested
    carbon beyond fat's is CO2. The forage's water (FORAGE_WATER_KG per kg dry eaten) comes from
    ``water(w [M], cell [M], kg_water [M]) -> kg given [M]`` when given (the world takes it from the
    cell's soil water) into body water, booked as ``food_w``. Returns kg_dry, energy_j, food_c,
    faeces_c, co2_mol, water_kg ([W, N]) and cell [W, N]."""
    W, N = cr.shape
    eat = mask & cr.alive
    M = cr.mass_kg.clamp_min(1e-6)
    dig = (PLANT_DIG0 - PLANT_DIG_DIET * cr.genome["diet"]).clamp(0, 1)
    room = (reserve_max(M) - cr.reserve_j).clamp_min(0)
    full = PLANT_INTAKE * M ** 0.75
    if share is not None:
        full = full * _f32(share, cr.device).clamp(0, 1)
    want = torch.minimum(full, room / (PLANT_J_KG * dig).clamp_min(1e-9)) * eat
    eat = eat & (want > 0)
    cell = globe.cell_of(cr.pos) if cell is None else cell
    w, n = eat.nonzero(as_tuple=True)
    kg = torch.zeros(W, N, device=cr.device)
    wat = torch.zeros(W, N, device=cr.device)
    if w.numel():
        given = torch.as_tensor(harvest(w, cell[w, n], want[w, n]), dtype=torch.float32, device=cr.device)
        kg[w, n] = given.clamp_min(0)
        if water is not None:
            got = torch.as_tensor(water(w, cell[w, n], FORAGE_WATER_KG * kg[w, n]), dtype=torch.float32,
                                  device=cr.device)
            wat[w, n] = got.clamp_min(0)
    energy = kg * PLANT_J_KG * dig
    food_c = kg * c_fraction
    faeces = food_c * (1 - dig)
    co2_c = food_c - faeces - energy * FAT_C_PER_J          # digested carbon beyond the fat's (>= 0)
    cr.reserve_j = cr.reserve_j + energy
    cr.water_kg = cr.water_kg + wat
    _book(cr, "food_j", energy)
    _book(cr, "food_c", food_c)
    _book(cr, "faeces_c", faeces)
    _book(cr, "co2_c", co2_c)
    _book(cr, "food_w", wat)
    return {"kg_dry": kg, "energy_j": energy, "food_c": food_c, "faeces_c": faeces, "co2_mol": co2_c / M_C,
            "water_kg": wat, "cell": cell}


_VEC_CACHE: dict = {}


def _species_vec(fn, device, name=None):
    """[S] float32 of fn over materials.SPECIES; with ``name`` cached per device (no host-to-device copy a day)."""
    key = (name, str(device))
    if name is not None and key in _VEC_CACHE:
        return _VEC_CACHE[key]
    t = torch.tensor([fn(s) for s in materials.SPECIES], dtype=torch.float32, device=device)
    if name is not None:
        _VEC_CACHE[key] = t
    return t


def eat_meat(cr, mask, pool, *, radius_m, reach_m=REACH_M, cooked_k=COOKED_K, chunk=256):
    """Eat food items where ``mask`` [W, N]: the held item or ground item within ``reach_m`` with the
    most digestible energy for the eater's diet. Intake is capped like grazing (0.08 M^0.75 kg dry
    matter) and by the room in the reserve; eaters of one item share it in proportion to their
    requests. Per species: digestible share = materials.DIGESTIBLE in diet x COOK_GAIN if the item
    is cooked (peak_k >= cooked_k), capped at 1; its water goes to body water, undigested carbon
    to faeces, the digested carbon beyond fat's to CO2. Items eaten below SCRAP_KG are removed.
    Returns energy_j, food_c, faeces_c, co2_mol, water_kg ([W, N]), item [W, N] (-1 none),
    eaten_species_kg and scraps_species_kg ([W, S] float64, for the item ledger)."""
    W, N = cr.shape
    I, S = pool.shape[1], pool.shape[2]
    dev = cr.device
    eat = mask & cr.alive
    out = {k: torch.zeros(W, N, device=dev) for k in ("energy_j", "food_c", "faeces_c", "co2_mol", "water_kg")}
    out["item"] = torch.full((W, N), -1, dtype=torch.long, device=dev)
    out["eaten_species_kg"] = torch.zeros(W, S, dtype=torch.float64, device=dev)
    out["scraps_species_kg"] = torch.zeros(W, S, dtype=torch.float64, device=dev)
    w, n = eat.nonzero(as_tuple=True)
    if not w.numel():
        return out
    food = _species_vec(lambda s: float(materials.FOOD_J_KG[s]), dev, "food")
    dig_p = _species_vec(lambda s: float(materials.DIGESTIBLE["plant"][s]), dev, "dig_plant")
    dig_m = _species_vec(lambda s: float(materials.DIGESTIBLE["meat"][s]), dev, "dig_meat")
    cook = _species_vec(lambda s: float(materials.COOK_GAIN[s]), dev, "cook")
    carbon = _species_vec(_carbon, dev, "carbon")
    water = _species_vec(_water, dev, "water")
    edible = (pool.comp * food).sum(-1) > 0
    cooked = pool.peak_k >= cooked_k
    radius = per_world(radius_m, W, dev)[:, 0]
    K = cr.K
    E = w.numel()
    # candidates (eater, item) in reach or held: a cheap [chunk, I] mask, then only the candidate pairs
    ce, ci = [], []
    for s0 in range(0, E, chunk):                       # chunks of eaters (memory), not single individuals
        ws, ns = w[s0:s0 + chunk], n[s0:s0 + chunk]
        held = pool.holder[ws] >= 0
        mine = held & (torch.div(pool.holder[ws], K, rounding_mode="floor") == ns[:, None])
        chord = (pool.pos[ws] - cr.pos[ws, ns][:, None]).norm(dim=-1)
        d = 2 * radius[ws][:, None] * torch.asin((chord / 2).clamp(max=1.0))
        ok = pool.alive[ws] & edible[ws] & (mine | (~held & (d <= reach_m)))
        e, i = ok.nonzero(as_tuple=True)
        ce.append(e + s0)
        ci.append(i)
    ce, ci = torch.cat(ce), torch.cat(ci)
    best = torch.full_like(w, -1)
    if ce.numel():
        cw = w[ce]
        diet = cr.genome["diet"][cw, n[ce]][:, None]
        gain = torch.where(cooked[cw, ci][:, None], cook, torch.ones_like(cook))            # [P, S]
        dg = ((dig_p * (1 - diet) + dig_m * diet) * gain).clamp(max=1.0)
        score = (pool.comp[cw, ci] * food * dg).sum(-1) * pool.mass[cw, ci]
        pos_ = score > 0
        ce, ci, score = ce[pos_], ci[pos_], score[pos_]
        top = torch.full((E,), -1.0, device=dev).scatter_reduce(0, ce, score, "amax", include_self=True)
        at_top = score == top[ce]                       # the most digestible energy; ties: the lowest item index
        first = torch.full((E,), I, dtype=torch.long, device=dev).scatter_reduce(0, ce[at_top], ci[at_top], "amin",
                                                                                include_self=True)
        best = torch.where(first < I, first, best)
    has = best >= 0
    w, n, idx = w[has], n[has], best[has]
    if not w.numel():
        return out
    comp = pool.comp[w, idx]                                             # [M, S]
    gain = torch.where(cooked[w, idx][:, None], cook, torch.ones_like(cook))
    frac = ((dig_p * (1 - cr.genome["diet"][w, n, None]) + dig_m * cr.genome["diet"][w, n, None]) * gain).clamp(max=1.0)
    j_kg = (comp * food * frac).sum(-1)
    dry = (comp * (1 - water)).sum(-1).clamp_min(1e-6)
    M = cr.mass_kg[w, n].clamp_min(1e-6)
    room = (reserve_max(M) - cr.reserve_j[w, n]).clamp_min(0)
    want = torch.minimum(PLANT_INTAKE * M ** 0.75 / dry, room / j_kg.clamp_min(1e-9))
    stock = pool.mass.clone()
    kg = share_scatter(stock, w, idx, want)
    kg = torch.minimum(kg, want)
    eaten_s = comp * kg[:, None]
    energy = (eaten_s * food * frac).sum(-1)
    food_c = (eaten_s * carbon).sum(-1)
    digested_c = (eaten_s * carbon * frac).sum(-1)
    fat_c = energy * FAT_C_PER_J
    co2_c = digested_c - fat_c
    wat = (eaten_s * water).sum(-1)
    pool.mass.index_put_((w, idx), -kg, accumulate=True)
    out["eaten_species_kg"].index_add_(0, w, eaten_s.double())
    for key, v in (("energy_j", energy), ("food_c", food_c), ("faeces_c", food_c - digested_c),
                   ("co2_mol", co2_c / M_C), ("water_kg", wat)):
        out[key][w, n] = v
    out["item"][w, n] = idx
    cr.reserve_j = cr.reserve_j.index_put((w, n), energy, accumulate=True)       # out of place, like eat_plants
    cr.water_kg = cr.water_kg.index_put((w, n), wat, accumulate=True)
    _book_w(cr, w, "food_j", energy)
    _book_w(cr, w, "food_c", food_c)
    _book_w(cr, w, "faeces_c", food_c - digested_c)
    _book_w(cr, w, "co2_c", co2_c)
    _book_w(cr, w, "food_w", wat)
    gone = torch.zeros_like(pool.alive)
    gone[w, idx] = True
    gone &= pool.alive & (pool.mass < SCRAP_KG)
    if bool(gone.any()):
        gw, gi = gone.nonzero(as_tuple=True)
        code = pool.holder[gw, gi]
        hn, hk = items.holder_split(code, K)
        held = code >= 0
        cr.inv[gw[held], hn[held], hk[held]] = -1
        out["scraps_species_kg"].index_add_(0, gw, items.remove(pool, gw, gi))
    return out


def evaporation_kg(c_th, t_k, heat_j):
    """Water evaporated in a day (kg): above T_BODY no heat can be lost dry, so the day's heat
    ``heat_j`` (J) and the heat gained from the air c_th (T - T_BODY) x 86400 s leave at LATENT_J_KG;
    0 at or below T_BODY."""
    return torch.where(t_k > T_BODY, (heat_j + c_th * (t_k - T_BODY) * DAY_S) / LATENT_J_KG, torch.zeros_like(t_k))


def drink(cr, mask, globe, soil_kg_m2, land, *, radius_m, lake=None, t_air_k=None):
    """Drink where ``mask`` up to the water norm plus the day's turnover (0.12 M^0.82 kg: drinking is
    spread over the day that :func:`physiology` then books as lost) plus, when ``t_air_k`` ([W, C]
    or [W] K) is given, the day's expected evaporation at rest (:func:`evaporation_kg` of the basal
    heat), from a lake (land cell with ``lake``), else from soil above DRINK_SOIL_MIN kg/m^2 (shared
    among the cell's drinkers; the caller subtracts ``soil_taken_kg_m2``), else from the sea, which
    rehydrates only with swim > SWIM_SALT. Returns from_lake, from_soil, from_sea (kg [W, N]), salt
    (bool [W, N]: drank sea water to no avail) and soil_taken_kg_m2 [W, C]."""
    W, N = cr.shape
    dev = cr.device
    soil_kg_m2 = _f32(soil_kg_m2, dev)
    d = mask & cr.alive
    M = cr.mass_kg.clamp_min(1e-9)
    turnover = WATER_TURNOVER * M ** WATER_EXP
    if t_air_k is not None:
        c_th = C_TH * M ** 0.5 / cr.genome["insulation"].clamp_min(0.05)
        turnover = turnover + evaporation_kg(c_th, local(globe, t_air_k, cr.pos), basal_w(M) * DAY_S)
    need = (water_norm(cr.mass_kg) + turnover - cr.water_kg).clamp_min(0) * d
    cell = globe.cell_of(cr.pos)
    on_land = land.gather(1, cell)
    lake_here = (lake.gather(1, cell) & on_land) if lake is not None else torch.zeros_like(on_land)
    soil_here = on_land & ~lake_here
    sea = ~on_land
    from_lake = need * lake_here
    area = globe.area[None] * per_world(radius_m, W, dev) ** 2                     # m^2 [W, C]
    avail = ((soil_kg_m2 - DRINK_SOIL_MIN).clamp_min(0) * area).clone()
    ws, ns = (d & soil_here & (need > 0)).nonzero(as_tuple=True)
    from_soil = torch.zeros(W, N, device=dev)
    if ws.numel():
        from_soil[ws, ns] = share_scatter(avail, ws, cell[ws, ns], need[ws, ns])
    taken = torch.zeros(W, globe.C, dtype=torch.float64, device=dev)
    taken.view(-1).index_add_(0, (torch.arange(W, device=dev)[:, None] * globe.C + cell).reshape(-1),
                              from_soil.reshape(-1).double())
    salty = cr.genome["swim"] > SWIM_SALT
    from_sea = need * sea * salty
    salt = d & sea & ~salty
    got = from_lake + from_soil + from_sea
    cr.water_kg = cr.water_kg + got
    _book(cr, "drunk_w", got)
    return {"from_lake": from_lake, "from_soil": from_soil, "from_sea": from_sea, "salt": salt,
            "soil_taken_kg_m2": (taken / area.double()).float()}


# ------------------------------------------------------------------------------------ physiology
def physiology(cr, globe, env, travel=None, *, loud=None, heat_k=None, insulation_add=None,
               neuron_cost_w=NEURON_COST_W):
    """One day of metabolism, growth, water and health for every living individual.

    ``env``: ``t_air_k`` [W, C] (or [W]) K, ``p_o2_pa`` [W] or [W, C] Pa; optional
    ``air_density_kg_m3`` [W] (calls). ``travel`` is :func:`move`'s output (its cost_j is paid
    here), ``loud`` [W, N] the call loudness (default ``cr.loud``), ``heat_k`` [W, N] warming by
    fires, ``insulation_add`` [W, N] worn insulation. Inputs of any float dtype; the state stays
    float32.

    Costs: basal 3.4 M^0.75, brain neuron_cost_w x k, senses SENSE_W M^0.75 (2 acuity^2 +
    hearing^2), thermoregulation C_th (T_lc - T) below T_lc (capped by the aerobic ceiling 10 x
    basal x a; the uncovered share harms health as cold), calling, travel. Juveniles grow by the
    exact daily von Bertalanffy step, paid from reserve above GROWTH_FLOOR. A negative reserve is
    covered by burning lean tissue. Water: 0.12 M^0.82 kg/day, plus, above T_BODY, the day's
    respired heat and the heat gained from the air C_th (T - T_BODY) evaporated at LATENT_J_KG.
    Respiration: O2 = the reserve joules spent / OXY_J_MOL (tissue synthesis takes none), CO2 = the
    fat carbon oxidised plus the catabolism surplus, and the O2 not returned as CO2 makes metabolic
    water, 2 (O2 - CO2_oxidised) mol, added to body water (``metabolic_w``) before the day's loss.
    Health: the day's harms are the strikes since the last call (``struck``, already taken off
    health by :func:`injure`), hypoxia below 1 kPa O2 and cold; ``harm`` becomes the cause of the
    largest of them, and only an individual with none of them, health > 0, reserve > 0 and water
    >= 0.9 x norm heals. ``struck`` is then cleared. Age +1 day, cooldown -1 day.
    Water above the norm after the day's loss is excreted (``excreted_kg``, booked in ``lost_w``).
    Returns o2_mol, co2_mol, water_loss_kg (breath and sweat), excreted_kg (urine), metabolic_w and
    the energy terms (J, [W, N] each).
    """
    W, N = cr.shape
    dev = cr.device
    alive = cr.alive
    af = alive.float()
    g = cr.genome
    M = cr.mass_kg.clamp_min(1e-6)
    T = local(globe, env["t_air_k"], cr.pos)
    if heat_k is not None:
        T = T + _f32(heat_k, dev)
    p_o2 = local(globe, env["p_o2_pa"], cr.pos)
    a = aerobic(p_o2)
    basal = basal_w(M)
    brain_w = neuron_cost_w * g["k"].float()
    levels = SENSE_UNIT["acuity"] * g["acuity"] ** 2 + SENSE_UNIT["hearing"] * g["hearing"] ** 2
    sense_w = SENSE_W * M ** 0.75 * levels
    ins = g["insulation"] + (_f32(insulation_add, dev) if insulation_add is not None else 0.0)
    ins = ins.clamp_min(0.05)
    c_th = C_TH * M ** 0.5 / ins
    p_th = c_th * (T_LC0 - T_LC_PER_INS * ins - T).clamp_min(0)
    room = (AEROBIC_SCOPE * a * basal - basal - brain_w - sense_w).clamp_min(0)
    p_th_paid = torch.minimum(p_th, room)
    cold_gap = torch.where(p_th > 0, (p_th - p_th_paid) / p_th.clamp_min(1e-12), torch.zeros_like(p_th))
    loud = cr.loud if loud is None else _f32(loud, dev)
    rho = per_world(env.get("air_density_kg_m3", RHO_AIR_EARTH), W, dev)
    c_snd = per_world(env.get("sound_speed_m_s", C_SOUND_EARTH), W, dev)
    p1 = call_pressure_1m(loud, g["voice"], M, rho)
    call_j = 4 * math.pi * p1 ** 2 / (rho * c_snd) / VOCAL_EFF * CALL_S
    loco = _f32(travel["cost_j"], dev) if travel is not None else torch.zeros_like(M)
    terms = {"basal_j": basal * DAY_S, "brain_j": brain_w * DAY_S, "sense_j": sense_w * DAY_S,
             "thermo_j": p_th_paid * DAY_S, "call_j": call_j, "travel_j": loco}
    terms = {k: v * af for k, v in terms.items()}
    spent = sum(terms.values())
    reserve = cr.reserve_j - spent
    # juvenile growth (exact daily von Bertalanffy step toward the adult mass)
    adult = g["adult_mass_kg"]
    m0 = litter_mass_kg(adult) / g["litter"].float()
    k = vb_k(adult, m0)
    m = cr.mass_kg
    target = (adult ** (1 / 3) - (adult ** (1 / 3) - m.clamp_min(0) ** (1 / 3)) * torch.exp(-k)) ** 3
    afford = (reserve - GROWTH_FLOOR * reserve_max(m)).clamp_min(0) / GROWTH_J_KG
    dm = torch.minimum((target - m).clamp_min(0), afford) * (alive & (m < adult))
    reserve = reserve - dm * GROWTH_J_KG
    overhead = dm * (GROWTH_J_KG - TISSUE_J_KG)
    m = m + dm
    # starvation: lean tissue covers a negative reserve
    burn = torch.minimum((-reserve).clamp_min(0) / TISSUE_J_KG, m) * af
    m = m - burn
    reserve = reserve + burn * TISSUE_J_KG
    cat_c = burn * TISSUE_C_KG - burn * TISSUE_J_KG * FAT_C_PER_J
    cr.mass_kg = m
    cr.frame_kg = torch.maximum(cr.frame_kg, m)
    cr.reserve_j = reserve
    resp = spent + overhead
    ox_c = spent * FAT_C_PER_J                              # the fat oxidised (RQ 0.72)
    co2_c = ox_c + cat_c
    o2 = spent / OXY_J_MOL                                  # tissue synthesis (overhead) takes no O2
    metabolic = 2 * (o2 - ox_c / M_C).clamp_min(0) * M_H2O  # PROVENANCE['metabolic_water']
    _book(cr, "respired_j", resp)
    _book(cr, "co2_c", co2_c)
    _book(cr, "o2_mol", o2)
    _book(cr, "oxidised_c", ox_c)
    _book(cr, "metabolic_w", metabolic)
    cr.water_kg = cr.water_kg + metabolic
    # water: turnover, and above body temperature all the day's heat leaves by evaporation
    evap = evaporation_kg(c_th, T, resp)
    loss = torch.minimum((WATER_TURNOVER * M ** WATER_EXP + evap) * af, cr.water_kg.clamp_min(0))
    cr.water_kg = cr.water_kg - loss
    # the kidneys pass what the day left above the norm (PROVENANCE['excretion'])
    excreted = (cr.water_kg - water_norm(cr.mass_kg)).clamp_min(0) * af
    cr.water_kg = cr.water_kg - excreted
    _book(cr, "lost_w", loss + excreted)
    # health: the day's harms (strikes since the last call, hypoxia, cold) and healing
    hyp = HYPOXIA_PER_DAY * (1 - p_o2 / HYPOXIA_PA).clamp(0, 1) * af
    cold = COLD_PER_DAY * cold_gap * af
    struck = cr.struck * af
    well = (alive & (struck == 0) & (hyp == 0) & (cold == 0) & (cr.health > 0) & (cr.reserve_j > 0)
            & (cr.water_kg >= 0.9 * water_norm(cr.mass_kg)))
    cr.health = torch.where(well, (cr.health + HEAL_PER_DAY).clamp(max=1.0), cr.health - hyp - cold)
    harms = torch.stack((struck, hyp, cold), -1)
    codes = torch.tensor([CAUSE["injury"], CAUSE["hypoxia"], CAUSE["cold"]], device=dev)
    cr.harm = torch.where(harms.sum(-1) > 0, codes[harms.argmax(-1)], cr.harm)   # ties: injury, hypoxia, cold
    cr.struck = torch.zeros_like(cr.struck)
    cr.age_d = cr.age_d + af
    cr.cooldown_d = (cr.cooldown_d - af).clamp_min(0)
    return {"o2_mol": o2, "co2_mol": co2_c / M_C, "water_loss_kg": loss, "excreted_kg": excreted,
            "evap_kg": evap * af, "metabolic_w": metabolic, "aerobic": a * af, "cold_gap": cold_gap * af, "growth_kg": dm, "burnt_kg": burn,
            "growth_overhead_j": overhead, **terms}


def injure(cr, w, n, damage):
    """Health loss from a strike: health and ``struck`` (the day's strikes, which block that day's
    healing) fall by ``damage``; the cause code becomes 'injury' (physiology may name a larger harm
    of the same day)."""
    w, n = w.long(), n.long()
    dmg = torch.as_tensor(damage, dtype=torch.float32, device=cr.device).expand(w.shape)
    cr.health = cr.health.index_put((w, n), -dmg, accumulate=True)
    cr.struck = cr.struck.index_put((w, n), dmg.clamp_min(0), accumulate=True)
    cr.harm = cr.harm.index_put((w, n), torch.full_like(w, CAUSE["injury"]))


# ------------------------------------------------------------------------------------ deaths
def death_cause(cr):
    """Cause code [W, N] (CAUSES; 0 = lives): health <= 0 (the last harm), dehydration (water below
    0.8 x norm), starvation (lean mass below 0.6 x frame), age (beyond the maximum lifespan)."""
    alive = cr.alive
    hurt = cr.health <= 0
    dry = cr.water_kg < DEHYDRATION_LETHAL * water_norm(cr.mass_kg)
    starved = cr.mass_kg < (1 - LEAN_LOSS_LETHAL) * cr.frame_kg
    old = cr.age_d >= lifespan_d(cr.genome["adult_mass_kg"])
    harm = torch.where(cr.harm > 0, cr.harm, torch.full_like(cr.harm, CAUSE["injury"]))
    cause = torch.zeros_like(cr.harm)
    for flag, code in ((old, CAUSE["age"]), (starved, CAUSE["starvation"]), (dry, CAUSE["dehydration"])):
        cause = torch.where(flag, torch.full_like(cause, code), cause)
    cause = torch.where(hurt, harm, cause)
    return torch.where(alive, cause, torch.zeros_like(cause))


def deaths(cr, pool, *, temp_k=T_BODY):
    """Remove the dead and turn each body into items at its position (items.spawn, on the ground at
    ``temp_k``): meat 0.45 M, bone 0.10 M, hide 0.06 M and fat = reserve / 39.5 MJ/kg. Held items
    drop where the body lies. A part that finds no free item slot goes to litter with the viscera.
    ``pos`` keeps the death position (``globe.cell_of(cr.pos)`` gives the cells for the litter).
    The parts hold their makeup water (materials.MAKEUP): when the body holds less, the meat, bone
    and hide are scaled down to what its water fills and the rest of their dry matter goes to
    litter, so ``water_kg`` is never negative.
    Returns dead (bool), cause (long), litter_c (kg C to the cell's litter), litter_j (J),
    water_kg (to the environment, >= 0; the water of the meat, bone and hide stays in the items, a
    store the world's water ledger must count), item_j (gross food energy in the items), all
    [W, N], and items_species_kg [W, S] float64 (what entered the item pool)."""
    W, N = cr.shape
    S = pool.shape[2]
    dev = cr.device
    cause = death_cause(cr)
    dead = cause > 0
    out = {"dead": dead, "cause": cause}
    for k in ("litter_c", "litter_j", "water_kg", "item_j"):
        out[k] = torch.zeros(W, N, device=dev)
    out["items_species_kg"] = torch.zeros(W, S, dtype=torch.float64, device=dev)
    w, n = dead.nonzero(as_tuple=True)
    if not w.numel():
        return out
    D = w.shape[0]
    M = cr.mass_kg[w, n]
    reserve = cr.reserve_j[w, n]
    parts = list(CORPSE) + ["fat"]
    sp = torch.tensor([materials.IDX[s] for s in parts], device=dev)
    wat = torch.tensor([_water(s) for s in parts], device=dev)
    mass = torch.stack([M * CORPSE[s] for s in CORPSE] + [reserve.clamp_min(0) / FAT_J_KG], -1)     # [D, 4]
    # the parts hold their makeup water: a body drier than that leaves smaller parts (the dry matter of the
    # rest goes to litter with the viscera), so a corpse never hands out water it does not hold
    need_w = (mass * wat).sum(-1)
    budget = cr.water_kg[w, n].clamp_min(0) * (1 - 1e-6)
    fill = torch.where(need_w > budget, budget / need_w.clamp_min(1e-30), torch.ones_like(need_w))
    mass = torch.where(wat > 0, mass * fill[:, None], mass)
    mass = torch.where(mass >= FAT_MIN_KG, mass, torch.zeros_like(mass))
    comp = torch.nn.functional.one_hot(sp, S).float()                                              # [4, S]
    rw = w.repeat_interleave(len(parts))
    pos = cr.pos[w, n].repeat_interleave(len(parts), 0)
    req_mass = mass.reshape(-1)
    want = req_mass > 0
    slot = torch.full_like(rw, -1)
    if bool(want.any()):
        slot[want] = items.spawn(pool, rw[want], comp.repeat(D, 1)[want], req_mass[want], pos[want], temp_k, -1)
    made = (slot >= 0).reshape(D, len(parts))
    got = mass * made
    food = torch.tensor([float(materials.FOOD_J_KG[s]) for s in parts], device=dev)
    carb = torch.tensor([_carbon(s) for s in parts], device=dev)
    body_j = body_energy_j(M, reserve)
    body_c = body_carbon_kg(M, reserve)
    item_j = (got * food).sum(-1)
    item_c = (got * carb).sum(-1)
    item_w = (got * wat).sum(-1)
    out["items_species_kg"].index_add_(0, w, (got.double()[..., None] * comp.double()).sum(1))
    out["item_j"][w, n] = item_j
    out["litter_j"][w, n] = (body_j - item_j.double()).float()
    out["litter_c"][w, n] = (body_c - item_c.double()).float()
    out["water_kg"][w, n] = cr.water_kg[w, n] - item_w
    _book_w(cr, w, "dead_j", body_j)
    _book_w(cr, w, "dead_c", body_c)
    _book_w(cr, w, "dead_w", cr.water_kg[w, n])
    for c in range(1, len(CAUSES)):
        cr.ledger[f"deaths_{CAUSES[c]}"] += ((cause == c) & dead).sum(1).double()
    # held items drop where the body lies
    held = cr.inv[w, n]                                                                             # [D, K]
    hw = w[:, None].expand_as(held)[held >= 0]
    hi = held[held >= 0]
    if hi.numel():
        pool.holder[hw, hi] = -1
        pool.pos[hw, hi] = cr.pos[w, n][:, None].expand(-1, held.shape[1], -1)[held >= 0]
    cr.inv[w, n] = -1
    cr.alive[w, n] = False
    cr.reserve_j[w, n] = 0.0
    cr.mass_kg[w, n] = 0.0
    cr.frame_kg[w, n] = 0.0
    cr.water_kg[w, n] = 0.0
    cr.struck[w, n] = 0.0
    cr.calls[w, n] = 0.0
    cr.loud[w, n] = 0.0
    return out


# ------------------------------------------------------------------------------------ reproduction
def reproduce(cr, gen, *, params=None, specs=None):
    """Asexual litters. A parent breeds when alive, at least ``maturity_d`` old, its reserve above
    half its maximum, its cooldown 0, and it can pay the litter (reserve, and water staying at or
    above DEHYDRATION_LETHAL x its norm). Its litter (``litter`` gene) of total mass 0.1 M^0.92
    fills free slots of its world; when there are fewer free slots than newborns, the ready
    parents are served in a random order (drawn from ``gen``), each taking the lowest free slots
    left. Each newborn's tissue (GROWTH_J_KG per kg) and starting reserve (CHILD_RESERVE of its
    maximum) are paid from the parent's reserve and its water from the parent's water. Genomes come
    from :func:`brain.mutate` of the parent's genome (GENE_SPECS), so nothing learned is inherited;
    live weights start from the child's own Wh. The parent's cooldown becomes the gestation time.
    Draws: the parents' order, mutation, then headings. Returns born (number per world [W]), o2_mol
    [W, N] (zeros: the synthesis heat of the newborn tissue, booked as respired energy, takes no O2;
    PROVENANCE['respiration_o2']), child_w and child_n (slots)."""
    W, N = cr.shape
    dev = cr.device
    g = cr.genome
    specs = GENE_SPECS if specs is None else specs
    M = cr.mass_kg
    adult = g["adult_mass_kg"]
    ready = (cr.alive & (cr.age_d >= maturity_d(adult)) & (cr.reserve_j > 0.5 * reserve_max(M))
             & (cr.cooldown_d <= 0))
    litter = g["litter"].long().clamp(1, None)
    m0 = litter_mass_kg(M) / litter.float()
    child_res = CHILD_RESERVE * reserve_max(m0)
    each = m0 * GROWTH_J_KG + child_res
    litter_w = water_norm(m0) * litter.float()
    ready = (ready & (cr.reserve_j - each * litter.float() > 0)
             & (cr.water_kg - litter_w >= DEHYDRATION_LETHAL * water_norm(M)))
    out = {"born": torch.zeros(W, dtype=torch.long, device=dev), "o2_mol": torch.zeros(W, N, device=dev),
           "child_w": torch.empty(0, dtype=torch.long, device=dev),
           "child_n": torch.empty(0, dtype=torch.long, device=dev)}
    pw, pn = ready.nonzero(as_tuple=True)
    if not pw.numel():
        return out
    order = torch.argsort(torch.rand(pw.numel(), generator=gen, device=dev))     # who is served first at capacity
    pw, pn = pw[order], pn[order]
    count = litter[pw, pn]
    cw, cp = pw.repeat_interleave(count), torch.arange(pw.numel(), device=dev).repeat_interleave(count)
    slot = items.allocate(cr.alive, cw)
    ok = slot >= 0
    cw, cp, slot = cw[ok], cp[ok], slot[ok]
    if not cw.numel():
        return out
    cn = pn[cp]
    child = brain.mutate(gen, brain.rows(g, cw, cn), params, specs)
    heading = torch.rand(cw.numel(), generator=gen, device=dev) * TWO_PI
    brain.install(g, cr.wh_live, cr.brain_state, cw, slot, child, baseline=cr.baseline)
    cm = m0[cw, cn]
    cres = child_res[cw, cn]
    cwat = water_norm(cm)
    # parent pays
    pay = torch.zeros(W, N, device=dev).index_put_((cw, cn), cm * GROWTH_J_KG + cres, accumulate=True)
    water_out = torch.zeros(W, N, device=dev).index_put_((cw, cn), cwat, accumulate=True)
    heat = torch.zeros(W, N, device=dev).index_put_((cw, cn), cm * (GROWTH_J_KG - TISSUE_J_KG), accumulate=True)
    cr.reserve_j = cr.reserve_j - pay
    cr.water_kg = cr.water_kg - water_out
    bred = pay > 0
    cr.cooldown_d = torch.where(bred, gestation_d(adult), cr.cooldown_d)
    _book(cr, "respired_j", heat)
    # the child
    cr.alive[cw, slot] = True
    cr.pos[cw, slot] = cr.pos[cw, cn]
    cr.heading[cw, slot] = heading
    cr.mass_kg[cw, slot] = cm
    cr.frame_kg[cw, slot] = cm
    cr.reserve_j[cw, slot] = cres
    cr.water_kg[cw, slot] = cwat
    cr.health[cw, slot] = 1.0
    cr.age_d[cw, slot] = 0.0
    cr.cooldown_d[cw, slot] = 0.0
    cr.harm[cw, slot] = 0
    cr.struck[cw, slot] = 0.0
    cr.inv[cw, slot] = -1
    cr.calls[cw, slot] = 0.0
    cr.loud[cw, slot] = 0.0
    onehot = torch.nn.functional.one_hot(cw, W)
    rank = (onehot.cumsum(0) * onehot).sum(1) - 1
    cr.uid[cw, slot] = cr.next_uid[cw] + rank
    cr.parent[cw, slot] = cr.uid[cw, cn]
    cr.founder[cw, slot] = cr.founder[cw, cn]
    cr.generation[cw, slot] = cr.generation[cw, cn] + 1
    born = onehot.sum(0)
    cr.next_uid += born
    cr.ledger["births"] += born.double()
    out.update(born=born, o2_mol=torch.zeros_like(heat), child_w=cw, child_n=slot)
    return out


# ------------------------------------------------------------------------------------ ledgers
def stocks(cr):
    """Energy (J), carbon (kg C) and water (kg) held by the living, [W] float64 each."""
    a = cr.alive.double()
    return {"energy": (body_energy_j(cr.mass_kg, cr.reserve_j) * a).sum(1),
            "carbon": (body_carbon_kg(cr.mass_kg, cr.reserve_j) * a).sum(1),
            "water": (cr.water_kg.double() * a).sum(1)}


def gas_mol(cr):
    """Cumulative respiration of the individuals per world [W] float64: O2 consumed (reserve energy
    oxidised / 4.5e5 J/mol), CO2 released (carbon oxidised, including the conversion of food and
    tissue) and metabolic water made (mol), for the atmosphere's oxygen and carbon ledgers."""
    return {"o2_mol": cr.ledger["o2_mol"].clone(), "co2_mol": cr.ledger["co2_c"] / M_C,
            "h2o_mol": cr.ledger["metabolic_w"] / M_H2O}


def ledger_errors(cr):
    """Stock minus (inflows - outflows) since the start, per world [W] float64, for energy, carbon,
    water; and the scale (largest of the stock and the gross flows) to judge them against."""
    L, s = cr.ledger, stocks(cr)
    flows = {"energy": (L["spawned_j"] + L["food_j"], L["respired_j"] + L["dead_j"]),
             "carbon": (L["spawned_c"] + L["food_c"], L["co2_c"] + L["faeces_c"] + L["dead_c"]),
             "water": (L["spawned_w"] + L["drunk_w"] + L["food_w"] + L["metabolic_w"], L["lost_w"] + L["dead_w"])}
    out = {}
    for k, (inn, outf) in flows.items():
        out[k] = s[k] - (inn - outf)
        out[f"{k}_scale"] = torch.maximum(s[k].abs(), inn + outf)
    return out
