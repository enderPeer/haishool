"""The planet world: every module of the engine put together (PLANET-SPEC section 2.12).

:class:`PlanetWorld` builds W worlds (one planet per seed) and steps them one day at a time.

Construction (``PlanetWorld(config, seeds, device)``)
-----------------------------------------------------
1. Chain inputs per seed (:func:`planet_inputs`): world7's chain (``chain.chain_inputs``); a seed
   whose chain has no habitable planet or does not reach the ``bodies`` era falls back to
   ``chain.synthetic_inputs`` (``config.source = "auto"``; "world7" or "synthetic" force one).
2. ``formation.build_planet`` -> PlanetSpec per world, stacked to [W] tensors (``spec_t``).
3. The set-up generator (a CPU ``torch.Generator``; its seed is ``config.rng_seed`` or derived from
   the seeds) draws, on the CPU, the habitat terrain (``globe.make_terrain`` with the water volume
   of ``globe.habitat_water_volume``), the ground deposits (``globe.make_deposits`` of the spec's
   crust mix over ``materials.CRUST_SPECIES``, kept in float64) and, after the spin-ups, the
   founders; the results move to the device, so a seed builds the same world on every device. The
   day's generator (on the device) is seeded from the set-up generator's next draw.
4. Climate: ``climate.make_params``, ``init_state`` and ``spin_up`` (cloud albedo calibrated to the
   chain's T_s). Biosphere: ``biosphere.make_params``, ``init_state`` and ``spin_up`` (air held).
5. Empty item and fire pools, founders (``creatures.spawn_founders``: bodies from the chain,
   brains from ``brain.random_genome`` with the innate forage wiring on ``senses.PLANT_INDEX``),
   placed on habitable cells (:meth:`habitable`: land, drinkable soil, air between the founders'
   lower critical temperature and the body temperature).
6. The ledger baseline: the climate's water ledger and the biosphere's carbon and oxygen ledgers
   are rebased and every store of the world is recorded; from here every flow is booked.

One day (``step``), in this order (``WORLD_PROVENANCE['world.order']``)
------------------------------------------------------------------------
 1. climate.step (vegetation cover from the biosphere) -> the day's T, water, light.
 2. biosphere.step (that day's surface light, T, soil) -> plants, litter, the gas exchange.
 3. senses.observe at the start-of-day position (the new fields, last day's calls).
 4. brain.think and brain.act (day generator): turn, speed, loudness, vocal vector, one focus
    action and a collect class per individual. The calls are written to ``loud``/``calls``.
 5. Craft actions where the individual looked, in an order that lets same-day actions see each
    other: collect -> drop -> knap -> combine -> make_fire -> feed_fire -> heat_item -> cook ->
    share. An individual whose action needs a fire or a relative within reach first walks to the
    nearest one it can see (``WORLD_PROVENANCE['world.approach']``).
 6. Strikes: the striker walks to the nearest individual it sees and strikes it
    (``crafting.tool_of``, ``strike_energy``, ``strike_damage``, ``creatures.injure``), before
    physiology, so a strike blocks that day's healing.
 7. Eating and drinking where the individual is: eat_meat (it walks to the nearest food item it
    sees), grazing (``eat_plants`` through ``biosphere.harvest``) for everyone but the resters, for
    the share of the active day the focus action and the travel leave (``world.time_budget``),
    with the forage's water from the cell's soil; drinking for those whose action is drink (soil,
    sea) and for everyone at fresh water (``world.background_drink``).
 8. crafting.fire_step (beds burn, items in fires heat and transform, other items relax toward
    the air at their cell) and crafting.decay (carrion and wood rot).
 9. creatures.move: the day's travel, last (resters stay; ``world.travel`` sets its time, land
    animals stop at the shore); held items follow.
10. creatures.physiology at the night's position: the travel, the walks to targets and the work
    (craft and strike work / muscle efficiency) are paid, fires warm, worn hides insulate.
11. creatures.deaths: bodies become meat, bone, hide and fat items; the rest goes to litter.
12. Plasticity: brain.outcome over the day (body energy: the reserve change less the lean tissue
    burnt, plus growth paid from the reserve; water against the lethal margin; health), brain.
    modulate (the running mean starts at an individual's first outcome) and brain.hebbian for those
    alive at the start and the end.
13. creatures.reproduce.
14. The day's routed flows are settled (see below) and the day counter advances.

Flows and ledgers (PLANET-SPEC section 0.3)
-------------------------------------------
* Gas: respiration O2 and CO2 (physiology, the conversion CO2 of eating), the fires' and
  cooking's ``air_kg`` (O2, CO2) and the CO2 of charred tissue go into the climate's gas column
  through ``climate.add_gas`` (booked in ``gas_external``).
* Water: drinking and the forage's water take from the soil bucket or the ocean store
  (swimmers); breath and sweat go to the vapour of the cell, urine (water above the norm) to the
  ground (soil on land, vapour over the ocean); a body's
  water not held by its corpse items, the water of rotted items and of eaten-down scraps go to the
  soil on land or the vapour over the ocean; the water made by fires and transforms (``air_kg``
  H2O) and the water of charred tissue go to the vapour at the fires. Water held in items
  (``materials.MAKEUP`` H2O) is a store of its own.
* Carbon: faeces, a body's carbon not held by its items and the organic carbon of rotted or
  scrap items go to the litter of their cells (``biosphere.add_litter``); carbonate and metal
  carbon of items leaving the pool go to a mineral sink (``flows['mineral_c_kg']``); collected
  wood leaves ``wood_c`` (``collect_wood``); deposits (limestone, malachite) bring carbon from
  the ground stock, which is a store of the items ledger (``deposits``).
* Per-cell float32 stores cannot take every small exchange at once: a drink of 2 kg from a soil
  cell of 1e5-1e6 m^2 is below the bucket's float32 rounding. Each world keeps float64 per-cell
  carries (soil and vapour water, litter carbon, wood owed by the plants): the day's exchanges go
  into them and each day they are flushed into the fields as far as float32 resolves; what is
  left waits in its cell (``WORLD_PROVENANCE['world.settle']``). The carries are stores of the
  ledgers, so the cross-module ledgers close to float64 rounding and nothing moves between cells.
* Non-carbon, non-water elements of items that rot, char or are eaten (N, P, Ca, ...) go to a
  soil-mineral sink (counted, ``flows['soil_sink_el']``).

``ledgers()`` returns per world error, scale and relative error of: ``energy`` (the individuals;
``bodies_carbon`` and ``bodies_water`` are their own carbon and water closures), ``carbon``
(= ``carbon_bio`` + ``carbon_mobile``), ``oxygen`` (= ``oxygen_bio`` + ``oxygen_mobile``), ``water``
(= ``water_climate`` + ``water_mobile``), ``oxygen_atoms`` (the O of the O2 the individuals took
from the air, by the climate's record, against the O they returned as CO2 of oxidised fat and as
metabolic water, by the creatures' record), ``items`` (element mass of items and fire beds against
the deposits, wood, corpses in and eaten, scraps, soil, rot and air out) and ``deposits`` (the
ground stock's loss against what collect made of it). The ``*_bio`` and ``water_climate`` parts are
the modules' own closures, the ``*_mobile`` parts check the bookings between modules. Errors are
absolute (kg, mol, J per world); ``rel`` is the error over the ledger's scale (its stores plus gross
flows), float32 tolerance is about 1e-5. The aggregates ``carbon``, ``oxygen`` and ``water``
(``INFO_LEDGERS``) are dominated by the air's CO2 and O2 columns and the ocean, so their ``rel``
cannot see a booking error between modules: they are reported for information; the checked
ledgers are the others (``CHECKED_LEDGERS``).

Frames for the viewer (``static(w)``, ``frame(w)``) follow ``FRAME_SCHEMA`` (the viewer is
``view.build_page``). An item's ``class_index`` is the ``materials.CLASSES`` index of its dominant
species (``static['item_classes']`` lists them); agent rows carry the optional 13th value ``diet``.

State: ``state_dict()`` (host copies) / ``PlanetWorld.from_state(state, device)`` restore a world
exactly on the CPU (static parts are rebuilt deterministically from the stored chain inputs and
terrain); ``state_hash()`` hashes every dynamic tensor and the generator. ``gates()`` reports the
spec's world gates (ledgers, founder and newborn viability).
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import math
import time
import warnings
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields

import torch

from . import biosphere as bs
from . import brain as br
from . import chain as ch
from . import climate as cl
from . import constants as K
from . import crafting as cf
from . import creatures as cr_
from . import formation as fm
from . import globe as gb
from . import items as it
from . import materials as mat
from . import senses as sn

STATE_FORMAT = "life9-planet-world-v2"
FRAME_SCHEMA = "life9-globe-v1"
FOUR_PI = 4.0 * math.pi
DAY_S = K.DAY_S
M_C = K.element_mass("C")
M_O2 = K.molar_mass("O2")
M_CO2 = K.molar_mass("CO2")
M_H2O = K.molar_mass("H2O")
S = mat.S
SD = len(mat.CRUST_SPECIES)
E = len(mat.ELEMENTS)
I_C = mat.ELEMENTS.index("C")
ACTIONS = br.ACTIONS
ACT = br.ACT
A_N = len(ACTIONS)
HANDS = sn.K_SLOTS
#: planet fields of the viewer's static block (PlanetSpec names)
STATIC_PLANET = ("radius_m", "gravity_m_s2", "surface_pressure_pa", "tidally_locked", "year_s", "star_teff_k",
                 "insolation_w_m2", "t_surface_target_k")
MAX_FRAME_ITEMS = 2000
#: the aggregate ledgers (information only: the air's columns and the ocean dominate their scales)
INFO_LEDGERS = ("carbon", "oxygen", "water")
CHECKED_LEDGERS = ("energy", "bodies_carbon", "bodies_water", "carbon_bio", "carbon_mobile", "oxygen_bio",
                   "oxygen_mobile", "oxygen_atoms", "water_climate", "water_mobile", "items", "deposits")
LEDGER_TOL = 1e-5

# ------------------------------------------------------------------------------------------- world rules
#: approximate, from memory: the "58 ft-lbf" (79 J) kinetic-energy casualty criterion of fragments for a
#: man (US Army Ballistic Research Laboratory, Sperrazza & Kokinakis 1967); new_rule: health 1 is lost to a
#: blow that delivers 79 J x (M / 70 kg) into the body (strike_damage)
STRIKE_LETHAL_J = 79.0
STRIKE_REF_KG = 70.0
#: a convex body's mean projected area is a quarter of its surface (Cauchy's surface-area formula)
PROJECTED_SHARE = 0.25
#: new_rule: the active day of an individual (time budget of the focus action, travel and grazing)
ACTIVE_S = 12 * 3600.0
#: the actions that take an hour of the active day (crafting.ACTION_S); forage, drink and rest take none
TIMED_ACTIONS = tuple(a for a in ACTIONS if a not in ("rest", "forage", "drink"))
#: reference (approximate, from memory): mammals' mean daily movement distance 1.04 M^0.25 km (Garland 1983,
#: Am. Nat. 121:571); the day's travel time is set from it (travel_seconds)
DAILY_KM, DAILY_EXP = 1.04, 0.25
#: new_rule: an individual walking to a target stops this far from it (within crafting.REACH_M)
STAND_M = 0.5 * float(cf.REACH_M)
#: the farthest any individual can see (the acuity gene's upper bound x senses.VISION_BASE_M)
SIGHT_MAX_M = sn.VISION_BASE_M * cr_.BODY_SPECS["acuity"].hi
#: numerics: the item pool keeps at least this share of its slots free (oldest rotting ground items go first)
POOL_FREE_SHARE = 1.0 / 8.0
#: gates: founders alive after GATE_DAYS, newborns surviving NEWBORN_D days
GATE_DAYS = 30
FOUNDER_GATE = 0.5
NEWBORN_D = 30.0
NEWBORN_GATE = 0.5
#: numerics: a routing buffer below this (kg) is a sign error (strict mode raises)
NEG_TOL_KG = 1e-6

WORLD_PROVENANCE = {
    "world.habitat": ("new_rule", "PLANET-SPEC 1: the habitat globe of config.habitat_radius_m carries the planet's "
                                  "derived surface values; geography, curvature and distances are the habitat's"),
    "world.synthetic_fallback": ("new_rule", "config.source 'auto': a world7 seed without a habitable planet or "
                                             "without the bodies era uses chain.synthetic_inputs(seed) (labelled "
                                             "'synthetic')"),
    "world.order": ("new_rule", "the day's order of the module docstring: climate, biosphere, senses, brain, then "
                                "the actions where the individual looked (craft actions collect, drop, knap, combine, "
                                "make_fire, feed_fire, heat_item, cook, share; strikes; eat_meat, grazing, drinking), "
                                "fire_step and decay, then the day's travel (movement last, so what is done is done "
                                "where it was seen), physiology at the night's position, deaths, plasticity, "
                                "reproduction"),
    "world.one_action": ("new_rule", "one sampled focus action per individual and day (PLANET-SPEC 2.10); the rest of "
                                     "the active day is the time budget's (world.time_budget); travel every day but "
                                     "for the resters"),
    "world.time_budget": ("new_rule", "an individual is active 12 h a day (ACTIVE_S; grazing ungulates feed about "
                                      "7-12 h a day, approximate, from memory). The focus action takes crafting."
                                      "ACTION_S (1 h) unless it is forage, drink or rest; the travel takes speed x "
                                      "creatures.MOVE_S. Everyone but the resters grazes for the time left, a share of "
                                      "the full daily intake creatures.PLANT_INTAKE: forage grazes the whole active "
                                      "day (on the move too), others (12 h - travel - action) / 12 h. 'rest' stays "
                                      "put: no travel, no grazing, no action"),
    "world.background_drink": ("new_rule", "drinking takes minutes, so everyone on fresh water (soil >= creatures."
                                           "DRINK_SOIL_MIN on land, as the senses' 'fresh') drinks each day whatever "
                                           "its action; the drink action also drinks the sea (swimmers) and dry land "
                                           "(nothing)"),
    "world.forage_water": ("reference", "creatures.FORAGE_WATER_KG (3 kg per kg dry, green forage about 75 % water) "
                                        "taken from the cell's soil water (with what is owed to it) in proportion "
                                        "among the cell's grazers"),
    "world.travel": ("derived+reference", "the day's travel time is the time in which the top economical speed "
                                          "(creatures.V_TROT M^0.24, speed gene 1, a = 1) covers twice Garland's (1983) "
                                          "mean daily movement distance of mammals 1.04 M^0.25 km (about 23 min at 30 "
                                          "kg), at most creatures.MOVE_S: an untrained brain's half speed moves the "
                                          "mean distance (5-10 km a day before, 2-4 times the mammals' mean)"),
    "world.shore": ("new_rule", "land animals do not walk out to sea (creatures PROVENANCE['shore_stop'])"),
    "world.approach": ("new_rule", "strike, share, eat_meat, feed_fire, heat_item and cook act within crafting."
                                   "REACH_M (5 m); the actor first walks to the nearest target it can see: an "
                                   "individual (strike), a relative by descent with a free hand (share), a food item "
                                   "on the ground (eat_meat) within its sight range r_v, or a live fire within its "
                                   "lamp range (senses.VISION_BASE_M x acuity), with a clear senses.line_of_sight; it "
                                   "stops STAND_M = 2.5 m short and pays the walk at creatures' land cost of transport"),
    "world.founder_habitat": ("new_rule", "founders are drawn by area over habitable cells: land, soil >= creatures."
                                          "DRINK_SOIL_MIN and the spun-up air between the founders' lower critical "
                                          "temperature (creatures.T_LC0 - T_LC_PER_INS x insulation) and creatures."
                                          "T_BODY; when a world has none, land below T_BODY and above T_lc - 20 K; "
                                          "then any land"),
    "world.effort": ("derived", "craft and strike effort = aerobic capacity a = pO2 / (pO2 + 3 kPa) (creatures."
                                "aerobic; PLANET-SPEC 2.8: a scales strike power): an action is done at full effort, "
                                "limited by oxygen"),
    "world.work_cost": ("derived", "the mechanical work of craft actions and strikes (crafting.action_work_j, an hour "
                                   "of arm work) is paid as metabolic energy work / creatures.MUSCLE_EFF (0.25) in "
                                   "physiology's travel term"),
    "world.strike": ("new_rule", "a strike hits the nearest other living individual the striker sees (world.approach) "
                                 "with its hardest held item or its bare limb (crafting.tool_of); the target's surface "
                                 "is hide (materials MOHS and TOUGHNESS of hide); one blow per strike action"),
    "world.strike_lethal": ("reference+new_rule", "approximate, from memory: 79 J (58 ft-lbf) casualty criterion of "
                                                  "fragments for a 70 kg man (Sperrazza & Kokinakis 1967, BRL); "
                                                  "health lost = damage_J / (79 J x M / 70 kg)"),
    "world.fire_warmth": ("derived+reference", "fire warmth as an air-temperature increment: the radiant flux of "
                                               "crafting.warmth_within x absorptivity (crafting.EMISSIVITY, Kirchhoff) x "
                                               "the mean projected area (a quarter of the Meeh surface 0.1 M^(2/3), "
                                               "Cauchy) over the body's conductance C_th = creatures.C_TH M^0.5 / "
                                               "insulation (Herreid & Kessel 1967); new_rule: it cuts "
                                               "thermoregulation (PLANET-SPEC 2.11), so it raises the air an "
                                               "individual feels at most to its lower critical temperature "
                                               "303 K - 8 insulation (an individual keeps its distance; without "
                                               "the cap a day 1 m from a 10 kW fire would be +100 K)"),
    "world.routing": ("new_rule", "litter carbon and water go to the cell where they arise (eater, corpse, item); "
                                  "per-world fire and cooking residues go to the cells of the day's fires in proportion "
                                  "to their heat release (the cooks' cells for cooking); ground water to the soil on "
                                  "land and to the vapour over the ocean; breath, sweat and charred tissue's water to "
                                  "the vapour; organic carbon of items leaving the pool to litter, their carbonate and "
                                  "metal carbon to a mineral sink; the carbon of charred tissue to CO2 (one O2 per C "
                                  "from the air: the combustion its drying heat in the fire implies); the elements of "
                                  "eaten items other than carbon and water to the soil-mineral sink (the eaters' "
                                  "excreta; bodies carry no N ledger, so the biosphere's nutrient pool does not get "
                                  "them back)"),
    "world.settle": ("new_rule", "numerics: the day's per-cell litter, wood and water exchanges go into float64 "
                                 "per-cell carries, flushed each day into the float32 fields as far as their rounding "
                                 "resolves; the rest waits in its cell (a store of the ledgers), so nothing is moved "
                                 "between cells or to the global stores"),
    "world.fire_substeps": ("new_rule", "numerics: fire_step runs config.fire_substeps substeps while any fire lives "
                                        "(ending early once none does), else one step (items only relax toward the "
                                        "air)"),
    "world.ledger_baseline": ("new_rule", "the ledgers start after the spin-ups and the founders: the founders' bodies "
                                          "are part of the starting stores"),
    "world.rng": ("new_rule", "a CPU set-up generator (terrain, deposits, founders) and a day generator on the device "
                              "seeded from it (the brain's draws, collect's lumps, reproduction; PLANET-SPEC 0.7)"),
    "world.outcome": ("new_rule", "brain.outcome with the body-energy change (the reserve change less the lean tissue "
                                  "burnt x creatures.TISSUE_J_KG, so fasting at an empty reserve is felt) plus the "
                                  "growth paid from the reserve, the water change over the lethal margin (1 - "
                                  "creatures.DEHYDRATION_LETHAL) x the norm (brain PROVENANCE['outcome_water_margin']) "
                                  "and the health change"),
    "world.baseline_start": ("new_rule", "the modulator's running mean starts at an individual's first outcome "
                                         "(founders and newborns), not at 0, so the first weeks carry no systematic "
                                         "anti-Hebbian bias"),
    "world.default_outputs": ("new_rule", "with zero output biases a founder travels at sigmoid(0) = half its top "
                                          "speed and calls at half loudness each day: the default behaviour of an "
                                          "untrained brain (brain.random_genome)"),
    "world.pool_pressure": ("new_rule", "numerics: when fewer than 1/8 of a world's item slots are free, its oldest "
                                        "ground items made only of rotting species (crafting.DECAY_PER_DAY: bone, "
                                        "hide, meat, fat, wood, fibre) rot away at once (to litter, soil and the "
                                        "mineral sink) until 1/8 is free, so a die-off cannot block new items"),
    "world.gates": ("new_rule", "world.gates(): checked ledgers within 1e-5; after 30 days at least half of the "
                                "founders alive; at least half of the newborns live 30 days"),
}


# ------------------------------------------------------------------------------------------- configuration
@dataclass
class PlanetConfig:
    """Settings of a planet world (SI units). Brain settings use :class:`brain.BrainParams` names."""
    G: int = 48                          # cube-sphere cells per face edge (C = 6 G^2)
    habitat_radius_m: float = 1.0e4      # PLANET-SPEC 1
    capacity: int = 1024                 # individual slots per world
    founders: int = 256                  # founders per world
    items: int = 4096                    # item slots per world
    fires: int = 256                     # fire slots per world
    hidden: int = 128                    # brain units at most (PLANET-SPEC 2.10)
    initial_hidden: int = 48             # founders' active units
    vocal_dims: int = 8                  # must be brain.VOCAL_DIMS (senses' channels)
    dt_days: int = 1                     # the tick: one day (PLANET-SPEC 0.2)
    source: str = "auto"                 # chain inputs: auto (world7, synthetic fallback), world7, synthetic, earth (reference control)
    rng_seed: int | None = None          # world generator seed (None: derived from the seeds)
    founder_mass_kg: float = cr_.FOUNDER_MASS_KG
    founder_genes: dict = field(default_factory=dict)   # overrides of founder body genes (tests, experiments)
    allowed_actions: tuple | None = None # restrict the brain's actions (None: all 14)
    weight_dtype: str = "float32"        # brain weight genes and live weights: float32 or bfloat16
    climate_fast_chunks: int = 8         # climate.spin_up
    climate_slow_chunks: int = 4
    climate_tol_k: float = 0.1
    bio_spin_days: int = 730             # biosphere.spin_up
    bio_spin_segments: int = 2
    fire_substeps: int = 48              # crafting.fire_step substeps per day
    neuron_cost_w: float = cr_.NEURON_COST_W
    relief_exaggeration: float = 1.0     # the viewer's relief scale (static frames)
    # brain.BrainParams (brain.brain_params reads these names)
    weight_mutation_rel: float = br.BrainParams.weight_mutation_rel
    bias_mutation_sd: float = br.BrainParams.bias_mutation_sd
    hidden_mutation_rate: float = br.BrainParams.hidden_mutation_rate
    weight_clip: float = br.BrainParams.weight_clip
    baseline_rate: float = br.BrainParams.baseline_rate
    eta0: float = br.BrainParams.eta0
    temperature0: float = br.BrainParams.temperature0

    def __post_init__(self):
        if self.allowed_actions is not None:
            self.allowed_actions = tuple(self.allowed_actions)
        self.founder_genes = dict(self.founder_genes or {})

    def to_dict(self) -> dict:
        d = asdict(self)
        d["allowed_actions"] = list(self.allowed_actions) if self.allowed_actions is not None else None
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PlanetConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    def check(self):
        if self.vocal_dims != br.VOCAL_DIMS:
            raise ValueError(f"vocal_dims must be {br.VOCAL_DIMS} (senses has that many call channels)")
        if self.dt_days != 1:
            raise ValueError("dt_days must be 1: every module steps one day")
        if not 1 <= self.founders <= self.capacity:
            raise ValueError("founders must lie within 1 and capacity")
        if self.source not in ("auto", "world7", "synthetic", "earth"):
            raise ValueError("source is auto, world7, synthetic or earth")
        if self.weight_dtype not in ("float32", "bfloat16"):
            raise ValueError("weight_dtype is float32 or bfloat16")
        bad = sorted(set(self.allowed_actions or ()) - set(ACTIONS))
        if bad:
            raise ValueError(f"unknown actions {bad}")
        bad = sorted(set(self.founder_genes) - (set(cr_.BODY_GENES) - {"adult_mass_kg"}))
        if bad:
            raise ValueError(f"founder_genes takes body genes other than adult_mass_kg (use founder_mass_kg): {bad}")


def planet_inputs(seed: int, source: str = "auto") -> tuple[dict, str]:
    """Chain inputs of one seed and a note on where they came from (``WORLD_PROVENANCE['world.synthetic_fallback']``)."""
    if source == "synthetic":
        return ch.synthetic_inputs(seed), "synthetic"
    if source == "earth":   # reference control (Earth and the sun); the seed only varies the world's random draws
        return ch.earth_inputs(), "earth reference control (not chain-derived)"
    if source == "world7":
        return ch.chain_inputs(seed), "world7"
    try:
        x = ch.chain_inputs(seed)
    except ch.NoHabitablePlanet as e:
        return ch.synthetic_inputs(seed), f"synthetic fallback: {e}"
    if "bodies" not in (x.get("links") or ()):
        return ch.synthetic_inputs(seed), (f"synthetic fallback: world7 seed {seed} reached {x.get('reached')}, "
                                           "not the bodies era")
    return x, "world7"


def _derived_rng_seed(seeds) -> int:
    h = hashlib.sha256(("life9-planet-world:" + ",".join(str(int(s)) for s in seeds)).encode()).digest()
    return int.from_bytes(h[:8], "little") & ((1 << 63) - 1)


def travel_seconds(mass_kg):
    """The day's travel time (s) of bodies of mass_kg (``WORLD_PROVENANCE['world.travel']``)."""
    m = mass_kg.clamp_min(1e-6)
    t = 2 * DAILY_KM * 1000.0 * m ** DAILY_EXP / (cr_.V_TROT * m ** cr_.V_TROT_EXP)
    return t.clamp(max=cr_.MOVE_S)


def _species_vec(fn, device, dtype=torch.float64):
    return torch.tensor([float(fn(s)) for s in mat.SPECIES], dtype=dtype, device=device)


def _water_fraction(s):
    return dict(mat.MAKEUP[s]).get("H2O", 0.0)


def _mineral_carbon(s):
    """kg of carbonate or metal carbon per kg of species s (the rest of its carbon is organic)."""
    if mat.CLASS[s] == "metal":
        return mat.species_elements(s).get("C", 0.0)
    return sum(x * mat.formula_elements(f).get("C", 0.0) for f, x in mat.MAKEUP[s] if "CO3" in f)


def _host(obj, device="cpu"):
    """A copy of a nest of dicts and tensors with every tensor copied to ``device``."""
    if torch.is_tensor(obj):
        return obj.detach().to(device, copy=True)
    if isinstance(obj, dict):
        return {k: _host(v, device) for k, v in obj.items()}
    return obj


class _Today(Mapping):
    """The day's counts per world, converted to Python lists on first access (no host sync before)."""

    def __init__(self, tensors: dict, day: int):
        self._t, self._day, self._d = tensors, day, None

    def _dict(self) -> dict:
        if self._d is None:
            t = self._t
            out = {"day": self._day, "actions": t["actions"].tolist(), "births": t["births"].tolist(),
                   "deaths": {c: v for c, v in zip(cr_.CAUSES[1:], t["deaths"].T.tolist())}}
            for k in ("fires_lit", "strikes", "npp_kg_c_m2_day", "gpp_kg_c_m2_day", "mean_t_k",
                      "precipitation_kg_m2_day", "mean_reward"):
                out[k] = t[k].tolist()
            self._d = out
        return self._d

    def __getitem__(self, key):
        return self._dict()[key]

    def __iter__(self):
        return iter(self._dict())

    def __len__(self):
        return len(self._dict())


# ------------------------------------------------------------------------------------------- the world
class PlanetWorld:
    """W planet worlds stepped together, one day per ``step`` (see the module docstring)."""

    def __init__(self, config: PlanetConfig | None = None, seeds=(781,), device="cpu", *, inputs=None):
        cfg = config or PlanetConfig()
        cfg.check()
        self.config = cfg
        self.seeds = [int(s) for s in seeds]
        if not self.seeds:
            raise ValueError("at least one seed")
        self.W = len(self.seeds)
        self.device = torch.device(device)
        if inputs is None:
            got = [planet_inputs(s, cfg.source) for s in self.seeds]
            self.inputs = [x for x, _ in got]
            self.input_notes = [n for _, n in got]
        else:
            self.inputs = [dict(x) for x in inputs]
            self.input_notes = ["given"] * self.W
        self._build_specs()
        self.rng_seed = int(cfg.rng_seed) if cfg.rng_seed is not None else _derived_rng_seed(self.seeds)
        cpu = torch.device("cpu")
        setup = torch.Generator().manual_seed(self.rng_seed)           # set-up draws, on the CPU on every device
        dev, W, R = self.device, self.W, cfg.habitat_radius_m
        self.globe = gb.Globe(cfg.G, dev)
        globe0 = self.globe if dev.type == "cpu" else gb.Globe(cfg.G, cpu)
        water = torch.stack([gb.habitat_water_volume(s.ocean_mass_kg, s.radius_m, s.gravity_m_s2, s.relief_m, R)
                             for s in self.specs]).cpu()
        relief = torch.tensor([s.relief_m for s in self.specs], dtype=torch.float64)
        terrain0 = gb.make_terrain(globe0, W, setup, relief, water, R)
        dep_report: dict = {}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            stock0 = gb.make_deposits(globe0, W, setup, terrain0, self.spec_t["crust"].double().cpu(),
                                      mat.CRUST_SPECIES, report=dep_report)
        self.terrain = {k: v.to(dev) for k, v in terrain0.items()}
        self.stock = stock0.to(dev, torch.float64)                       # float64: a lump is below float32 ulp
        self.stock0 = self.stock.clone()
        self._build_params()
        clim0 = cl.init_state(self.globe, self.P)
        clim, crep = cl.spin_up(clim0, self.globe, self.P, fast_chunks=cfg.climate_fast_chunks,
                                slow_chunks=cfg.climate_slow_chunks, tol_k=cfg.climate_tol_k)
        bio = bs.init_state(self.globe, self.P, self.B, clim)
        bio, clim, brep = bs.spin_up(bio, clim, self.globe, self.P, self.B, days=cfg.bio_spin_days,
                                     segments=cfg.bio_spin_segments)
        self.clim = cl.reset_water_ledger(clim, self.globe)
        self.bio = bs.reset_ledgers(bio, self.clim)
        self.pool = it.new_item_pool(W, cfg.items, device=dev)
        self.fires = it.new_fire_pool(W, cfg.fires, device=dev)
        habit, habit_note = self.habitable()
        cr0 = cr_.spawn_founders(self.inputs, globe0, terrain0, setup, capacity=cfg.capacity,
                                 founders=cfg.founders, in_dim=sn.IN_DIM, hidden=cfg.hidden,
                                 initial_hidden=cfg.initial_hidden, params=self.bparams, K=HANDS,
                                 D=cfg.vocal_dims, adult_mass_kg=cfg.founder_mass_kg,
                                 weight_dtype=self._wdtype, cell_weight=habit.cpu())
        self.cr = cr0 if dev.type == "cpu" else self._creatures_on(cr0, dev)
        del cr0
        for name, value in cfg.founder_genes.items():
            g = self.cr.genome[name]
            g[...] = torch.as_tensor(value, device=dev).to(g.dtype)
        # the running mean starts at each founder's first outcome (WORLD_PROVENANCE['world.baseline_start'])
        self.cr.baseline = torch.where(self.cr.alive, torch.full_like(self.cr.baseline, math.nan), self.cr.baseline)
        day_seed = int(torch.randint(0, 1 << 62, (1,), generator=setup))
        self.gen = torch.Generator(device=dev).manual_seed(day_seed)
        self.day = 0
        self.last_action = torch.full((W, cfg.capacity), -1, dtype=torch.long, device=dev)
        self._new_carry()
        f_cells = self.globe.cell_of(self.cr.pos)
        a = self.cr.alive
        self.reports = {
            "inputs": self.input_notes, "rng_seed": self.rng_seed,
            "climate_spin_up": {k: crep[k] for k in ("cloud_albedo", "t_eq", "error_k", "chunks_run", "slow_chunk_days")},
            "biosphere_spin_up": {k: brep[k] for k in ("days", "plant_c", "mean_npp_kg_c_m2_day", "mean_t")},
            "deposits": {"support_level": dep_report["support_level"].tolist(),
                         "land_error": dep_report["land_error"].tolist(),
                         "ocean_error": dep_report["ocean_error"].tolist(),
                         "species": list(mat.CRUST_SPECIES), "warnings": [str(w.message) for w in caught]},
            "terrain": {"ocean_fraction": self.terrain["ocean_fraction"].tolist(),
                        "sea_level_m": self.terrain["sea_level_m"].tolist()},
            "founders": {"habitat": habit_note,
                         "on_habitable_cells": (habit.gather(1, f_cells) & a).sum(1).tolist(),
                         "habitable_land_share": ((habit.double() * self.globe.area64).sum(-1)
                                                  / ((self.P["land"].double() * self.globe.area64).sum(-1))
                                                  .clamp_min(1e-30)).tolist()},
        }
        self._new_flows()
        self._new_day()
        self.base = self._stores()

    # --------------------------------------------------------------------------------------- set-up pieces
    @property
    def _wdtype(self):
        return torch.bfloat16 if self.config.weight_dtype == "bfloat16" else torch.float32

    @staticmethod
    def _creatures_on(cr, dev, own=True):
        """A Creatures object with every tensor on ``dev`` (no copy where it already is when ``own``)."""
        mv = (lambda t: t.to(dev)) if own else (lambda t: t.to(dev, copy=True))
        return cr_.Creatures(**{n: mv(getattr(cr, n)) for n in cr_.Creatures.TENSORS},
                             genome={k: mv(v) for k, v in cr.genome.items()},
                             ledger={k: mv(v) for k, v in cr.ledger.items()})

    def _build_specs(self):
        self.specs = [fm.build_planet(x, s) for x, s in zip(self.inputs, self.seeds)]
        bad = {s.seed: fm.check_spec(s) for s in self.specs}
        self.spec_failures = {k: v for k, v in bad.items() if v}
        self.spec_t = fm.stack_specs(self.specs, self.device)

    def _build_params(self):
        """Climate and biosphere parameters, brain settings and per-world constants (from specs and terrain)."""
        cfg, dev, R = self.config, self.device, self.config.habitat_radius_m
        self.P = cl.make_params(self.specs, self.globe, self.terrain, R)
        self.B = bs.make_params(self.P)
        self.bparams = br.brain_params(cfg)
        self.area_m2 = FOUR_PI * R * R                                       # the habitat's surface
        self.cell_m2 = self.globe.area64 * R * R                             # [C] float64
        self._carbon = mat.element_matrix(dev)[:, I_C].clone()               # [S] kg C per kg
        self._mineral_c = _species_vec(_mineral_carbon, dev)                 # [S] carbonate and metal C per kg
        self._organic_c = self._carbon - self._mineral_c
        self._water = _species_vec(_water_fraction, dev)                     # [S] kg H2O per kg (MAKEUP)
        self._el = mat.element_matrix(dev)                                   # [S, E]
        self._air_el = mat.gas_element_matrix(mat.AIR_GASES, dev)            # [3, E]
        self._h2o_el = mat.gas_element_matrix(("H2O",), dev)[0]              # [E]
        self._food_j = _species_vec(lambda s: mat.FOOD_J_KG[s], dev, torch.float32)
        rot = set(cf.DECAY_PER_DAY)
        self._rots = torch.tensor([s in rot for s in mat.SPECIES], device=dev)
        self._north = torch.tensor([0.0, 0.0, 1.0], device=dev)
        allowed = torch.ones(len(ACTIONS), dtype=torch.bool, device=dev)
        if cfg.allowed_actions is not None:
            allowed = torch.tensor([a in cfg.allowed_actions for a in ACTIONS], device=dev)
        self._allowed = allowed
        self._timed = torch.tensor([a in TIMED_ACTIONS for a in ACTIONS], device=dev)
        self._hide_h = float(mat.MOHS["hide"])
        self._hide_k = float(mat.TOUGHNESS["hide"])
        self._species_class = torch.tensor([mat.CLASSES.index(mat.CLASS[s]) for s in mat.SPECIES], device=dev)
        self.surf_v = self.globe.at_vertices(cr_.surface_m(self.terrain).float())      # the static terrain
        W, C = self.W, self.globe.C
        self._wc = (torch.arange(W, device=dev).repeat_interleave(C), torch.arange(C, device=dev).repeat(W))
        self._chord_sight = 2 * math.sin(min(math.pi, SIGHT_MAX_M / R) / 2)

    def habitable(self):
        """[W, C] bool cells the founders may start on (``WORLD_PROVENANCE['world.founder_habitat']``) and a note
        per world on which rule applied."""
        T = self.clim.T
        land = self.P["land"]
        ins = torch.tensor([float(self.config.founder_genes.get("insulation", cr_.founder_genes(x)["insulation"][0]))
                            for x in self.inputs], device=self.device)
        t_lc = (cr_.T_LC0 - cr_.T_LC_PER_INS * ins)[:, None]
        wet = self.clim.soil >= cr_.DRINK_SOIL_MIN
        strict = land & wet & (T >= t_lc) & (T < cr_.T_BODY)
        loose = land & (T >= t_lc - 20.0) & (T < cr_.T_BODY)
        has_s, has_l = strict.any(-1), loose.any(-1)
        out = torch.where(has_s[:, None], strict, torch.where(has_l[:, None], loose, land))
        notes = ["habitable" if bool(s) else ("loose" if bool(l) else "land") for s, l in zip(has_s, has_l)]
        return out, notes

    def _new_flows(self):
        W, dev = self.W, self.device
        T = len(mat.TRANSFORMS)
        z = lambda *s: torch.zeros(*s, dtype=torch.float64, device=dev)
        self.flows = {
            # item pool (kg per species)
            "dep_in_kg": z(W, S), "wood_in_kg": z(W), "corpse_in_kg": z(W, S), "eaten_kg": z(W, S),
            "scraps_kg": z(W, S), "to_soil_kg": z(W, S), "decayed_kg": z(W, S), "air_kg": z(W, 3),
            "transformed_kg": z(W, T), "soil_sink_el": z(W, E), "mineral_c_kg": z(W), "items_recycled": z(W),
            # fires and work
            "fire_heat_j": z(W), "item_heat_j": z(W), "fires_lit": z(W), "fires_out": z(W), "work_j": z(W),
            "strikes": z(W), "strike_damage_j": z(W), "char_c_kg": z(W), "char_o2_mol": z(W),
            # gas from the individuals (mol)
            "creature_o2_mol": z(W), "creature_co2_mol": z(W),
            # routed amounts
            "litter_c_kg": z(W), "ground_water_kg": z(W), "vapour_kg": z(W), "drunk_soil_kg": z(W),
            "drunk_sea_kg": z(W), "forage_water_kg": z(W), "walk_m": z(W),
            # actions: tries and successes of the reach-limited ones, young deaths
            "tries": z(W, A_N), "done": z(W, A_N), "deaths_young": z(W),
        }

    def _new_carry(self):
        """The float64 per-cell carries (``WORLD_PROVENANCE['world.settle']``): kg of water owed to the soil
        (negative: owed by it) and to the vapour, kg C owed to the litter, kg of wood owed by the plants (<= 0)."""
        z = lambda: torch.zeros(self.W, self.globe.C, dtype=torch.float64, device=self.device)
        self.carry = {"soil": z(), "vapour": z(), "litter": z(), "wood": z()}

    def _new_day(self):
        """The day's routing buffers (dense [W, C] float64) and gas delta [W, 4] (mol per m^2 of column)."""
        W, C, dev = self.W, self.globe.C, self.device
        self._litter = torch.zeros(W, C, dtype=torch.float64, device=dev)    # kg C to litter
        self._wet = torch.zeros(W, C, dtype=torch.float64, device=dev)       # kg water to the ground (soil / sea vapour)
        self._vap = torch.zeros(W, C, dtype=torch.float64, device=dev)       # kg water to vapour
        self._gas = torch.zeros(W, 4, dtype=torch.float64, device=dev)

    # --------------------------------------------------------------------------------------- helpers
    def _cells(self, pos, alive):
        """Cell index of positions [W, X, 3]; dead slots (zero vectors) read the north pole."""
        safe = torch.where(alive[..., None], pos, self._north.expand_as(pos))
        return self.globe.cell_of(safe)

    def _spread(self, buf, total, weight, cells):
        """Add per-world totals [W] (float64) to buf [W, C], split over cells [W, X] in proportion to
        weight [W, X] >= 0 (equally over weight > 0 ... or over every listed slot when all weights are 0)."""
        W, C = buf.shape
        if cells.shape[1] == 0:
            buf[:, 0] += total
            return
        wt = weight.double().clamp_min(0)
        s = wt.sum(1, keepdim=True)
        wt = torch.where(s > 0, wt / s.clamp_min(1e-300), torch.full_like(wt, 1.0 / cells.shape[1]))
        idx = (torch.arange(W, device=buf.device)[:, None] * C + cells.long()).reshape(-1)
        buf.view(-1).index_add_(0, idx, (wt * total[:, None]).reshape(-1))

    def _to_ground(self, species_kg, weight, cells):
        """Items' species mass [W, S] leaving the pool to the ground at cells: organic carbon to litter, carbonate
        and metal carbon to the mineral sink, MAKEUP water to the ground water, the other elements to the
        soil-mineral sink."""
        kg = species_kg.double()
        c_org = kg @ self._organic_c
        c_min = kg @ self._mineral_c
        h2o = kg @ self._water
        self._spread(self._litter, c_org, weight, cells)
        self._spread(self._wet, h2o, weight, cells)
        self.flows["mineral_c_kg"] += c_min
        sink = mat.element_mass(kg)
        sink[:, I_C] -= c_org + c_min
        sink -= h2o[:, None] * self._h2o_el
        self.flows["soil_sink_el"] += sink

    def _char(self, species_kg, weight, cells):
        """Charred tissue [W, S] leaving the pool in fires: its makeup water to the vapour at the fires, its
        carbon to CO2 (one O2 from the air per C), the other elements to the soil-mineral sink."""
        kg = species_kg.double()
        c = kg @ self._carbon
        h2o = kg @ self._water
        self._spread(self._vap, h2o, weight, cells)
        mol = c / M_C
        self._gas[:, cl.I_CO2] += mol / self.area_m2
        self._gas[:, cl.I_O2] -= mol / self.area_m2
        self.flows["char_c_kg"] += c
        self.flows["char_o2_mol"] += mol
        sink = mat.element_mass(kg)
        sink[:, I_C] -= c
        sink -= h2o[:, None] * self._h2o_el
        self.flows["soil_sink_el"] += sink

    def _add_cells(self, buf, mask, cells, kg):
        """buf [W, C] += kg [W, N] at cells [W, N] where mask (dense: no host sync)."""
        W, C = buf.shape
        idx = (torch.arange(W, device=buf.device)[:, None] * C + cells.long()).reshape(-1)
        buf.view(-1).index_add_(0, idx, (kg.double() * mask).reshape(-1))

    def _sync_held(self):
        """Held items at their holders' positions (dense: no host sync)."""
        pool, cr = self.pool, self.cr
        held = pool.alive & (pool.holder >= 0)
        n = torch.div(pool.holder.clamp_min(0), HANDS, rounding_mode="floor").clamp(max=cr.shape[1] - 1)
        p = cr.pos.gather(1, n[..., None].expand(-1, -1, 3))
        pool.pos.copy_(torch.where(held[..., None], p, pool.pos))

    def _env(self, press):
        """Environment fields for senses."""
        return {"elevation_m": self.terrain["elevation_m"], "sea_level_m": self.terrain["sea_level_m"],
                "land": self.terrain["land"], "t_air_k": self.clim.T, "soil_kg_m2": self.clim.soil,
                "plant_c": self.bio.plant_c, "insolation_w_m2": self._insolation,
                "surface_pressure_pa": press["total"], "air_density_kg_m3": self.spec_t["air_density_kg_m3"],
                "sound_speed_m_s": self.spec_t["sound_speed_m_s"], "stock_kg_m2": self.stock,
                "deposit_species": mat.CRUST_SPECIES, "wood_c": self.bio.wood_c, "surface_v": self.surf_v}

    # --------------------------------------------------------------------------------------- the day
    def step(self) -> Mapping:
        """One day (see the module docstring). Returns the day's counts per world (``today``, converted to
        lists when first read)."""
        cfg, globe, P, B = self.config, self.globe, self.P, self.B
        W, R = self.W, cfg.habitat_radius_m
        dev = self.device
        cr, pool, fires = self.cr, self.pool, self.fires
        N = cr.shape[1]
        self._new_day()
        before_ledger = {k: cr.ledger[k].clone() for k in ("births", *(f"deaths_{c}" for c in cr_.CAUSES[1:]))}
        before_flows = {k: self.flows[k].clone() for k in ("fires_lit", "strikes", "creature_o2_mol",
                                                            "creature_co2_mol")}

        lap = self._lap()
        self._relieve_pool()
        # 1-2. climate and biosphere
        self.clim, cdiag = cl.step(self.clim, globe, P, bs.cover(self.bio, B))
        self.bio, self.clim, bdiag = bs.step(self.bio, self.clim, globe, P, B, cdiag)
        self._insolation = cdiag["insolation"]
        press = cl.pressures(self.clim, globe, P)
        p_o2 = press["O2"]
        x_o2 = (press["O2"] / press["total"]).float()
        T = self.clim.T

        lap("climate_biosphere")
        # 3-4. senses and brain, at the start-of-day position
        alive0 = cr.alive.clone()
        af = alive0.float()
        r0, w0, h0 = cr.reserve_j.clone(), cr.water_kg.clone(), cr.health.clone()
        x, seen = sn.observe(cr, globe, self._env(press), pool, fires, radius_m=R, details=True)
        r_v = seen["r_v"]
        r_lum = sn.VISION_BASE_M * cr.genome["acuity"].clamp_min(0) * af
        width = self._brain_width()
        before = cr.brain_state
        out, after = br.think(cr.genome, cr.wh_live, before, x, width=width)
        allowed = self._allowed.expand(W, N, -1)
        d = br.act(out, cr.genome, self.gen, allowed=allowed)
        cr.brain_state = after * af[..., None]
        cr.loud = d["loud"] * af
        cr.calls = d["vocal"] * af[..., None]
        action = torch.where(alive0, d["action"], torch.full_like(d["action"], -1))
        self.last_action = action
        onehot = torch.nn.functional.one_hot(action.clamp_min(0), A_N).bool() & alive0[..., None]   # [W, N, A]
        counts = onehot.sum(1)                                                                      # [W, A]
        self.flows["tries"] += counts.double()
        any_act = (counts.sum(0) > 0).tolist()
        mask = {name: onehot[..., ACT[name]] for name in ACTIONS}
        rest = mask["rest"]

        cell = globe.cell_of(cr.pos)
        M = cr.mass_kg
        eff = cr_.aerobic(cr_.local(globe, p_o2, cr.pos)) * af
        t_here = T.gather(1, cell)
        work = torch.zeros(W, N, device=dev)
        walk = torch.zeros(W, N, device=dev)                                     # m walked to targets
        hour = cf.action_work_j(eff, M)                                          # an hour of arm work

        def on(name):
            return any_act[ACT[name]]

        lap("senses_brain")
        # 5. craft actions where the individual looked
        if on("collect"):
            wood_kg_m2 = self.bio.wood_c / cf.WOOD_C_FRACTION
            col = cf.collect(pool, cr.inv, mask["collect"], d["collect_class"], cr.pos, cell, self.stock,
                             wood_kg_m2, self.cell_m2, self.gen, body_mass_kg=M, item_temp_k=t_here, radius_m=R,
                             effort=eff)
            work = work + col["work_j"]
            self.flows["dep_in_kg"] += col["ground_kg"]
            self.flows["wood_in_kg"] += col["wood_kg"].sum(1)
            self.carry["wood"] -= col["wood_kg"]                     # the plants owe the items' wood (settle)
        if on("drop"):
            cf.drop(pool, cr.inv, mask["drop"], cr.pos)
            work = work + hour * mask["drop"]
        if on("knap"):
            kn = cf.knap(pool, cr.inv, mask["knap"], cr.pos, body_mass_kg=M, effort=eff)
            work = work + kn["work_j"]
        if on("combine"):
            cf.combine(pool, cr.inv, mask["combine"])
            work = work + hour * mask["combine"]
        if on("make_fire"):
            mf = cf.make_fire(pool, fires, cr.inv, mask["make_fire"], cr.pos, x_o2, t_here, body_mass_kg=M,
                              effort=eff)
            work = work + mf["work_j"]
            self.flows["fires_lit"] += mf["lit"].sum(1).double()
            self.flows["done"][:, ACT["make_fire"]] += mf["lit"].sum(1).double()
        for name in ("feed_fire", "heat_item", "cook"):
            if on(name):
                _, went = self._approach(mask[name], fires.pos, fires.alive, r_lum, sn.FIRE_HEIGHT_M)
                walk = walk + went
        if on("feed_fire"):
            ff = cf.feed_fire(pool, fires, cr.inv, mask["feed_fire"], cr.pos, radius_m=R, effort=eff)
            work = work + hour * mask["feed_fire"]
            self.flows["done"][:, ACT["feed_fire"]] += (ff["fire"] >= 0).sum(1).double()
        if on("heat_item"):
            hi = cf.heat_item(pool, fires, cr.inv, mask["heat_item"], cr.pos, radius_m=R)
            work = work + hour * mask["heat_item"]
            self.flows["done"][:, ACT["heat_item"]] += (hi["fire"] >= 0).sum(1).double()
        if on("cook"):
            fire_t = T.gather(1, self._cells(fires.pos, fires.alive))
            ck = cf.cook(pool, fires, cr.inv, mask["cook"], cr.pos, x_o2, radius_m=R, ambient_fire_k=fire_t)
            work = work + hour * mask["cook"]
            cooks = mask["cook"] & (ck["fire"] >= 0)
            self._fire_flows(ck, cooks.float(), globe.cell_of(cr.pos))
            self.flows["done"][:, ACT["cook"]] += cooks.sum(1).double()
        if on("share"):
            _, went = self._approach(mask["share"], cr.pos, cr.alive & ((cr.inv < 0).sum(-1) > 0),
                                     r_v, None, pair=self._kin_pair())
            walk = walk + went
            sh = cf.share(pool, cr.inv, mask["share"], cr.pos, cr.uid, cr.parent, cr.alive, radius_m=R)
            work = work + hour * mask["share"]
            self.flows["done"][:, ACT["share"]] += (sh["to"] >= 0).sum(1).double()

        lap("craft")
        # 6. strikes
        if on("strike"):
            blow, went = self._strikes(mask["strike"], eff, r_v)
            work = work + blow + hour * mask["strike"]
            walk = walk + went

        lap("strikes")
        # 7. eating and drinking where the individual now is
        if on("eat_meat"):
            food = pool.alive & (pool.holder < 0) & ((pool.comp * self._food_j).sum(-1) > 0)
            _, went = self._approach(mask["eat_meat"], pool.pos, food, r_v, sn.ITEM_HEIGHT_M)
            walk = walk + went
            em = cr_.eat_meat(cr, mask["eat_meat"], pool, radius_m=R)
            ate = mask["eat_meat"] & (em["item"] >= 0)
            ecell = globe.cell_of(cr.pos)
            self._add_cells(self._litter, ate, ecell, em["faeces_c"])
            self.flows["creature_co2_mol"] += em["co2_mol"].double().sum(1)
            self.flows["eaten_kg"] += em["eaten_species_kg"]
            self.flows["scraps_kg"] += em["scraps_species_kg"]
            self.flows["done"][:, ACT["eat_meat"]] += ate.sum(1).double()
            self._to_ground(em["scraps_species_kg"], em["food_c"] * ate, ecell)
            # the eaten elements other than carbon and water leave with the excreta (soil-mineral sink)
            eaten = em["eaten_species_kg"]
            sink = mat.element_mass(eaten)
            sink[:, I_C] -= eaten @ self._carbon
            sink -= (eaten @ self._water)[:, None] * self._h2o_el
            self.flows["soil_sink_el"] += sink
        cell = globe.cell_of(cr.pos)
        # grazing for the time the day leaves (WORLD_PROVENANCE['world.time_budget'])
        travel_secs = travel_seconds(cr.mass_kg)
        travel_s = d["speed"].clamp(0, 1) * travel_secs * ~rest
        act_s = float(cf.ACTION_S) * self._timed[action.clamp_min(0)]
        share = torch.where(mask["forage"], torch.ones_like(travel_s),
                            ((ACTIVE_S - travel_s - act_s) / ACTIVE_S).clamp(0, 1)) * ~rest
        grazers = alive0 & ~rest & (share > 0)
        avail_w = (self.clim.soil.double() * self.cell_m2 + self.carry["soil"]).clamp_min(0)     # [W, C] kg

        def harvest(w, c, kg):
            return bs.harvest(self.bio, w, c, kg, carbon_fraction=cr_.PLANT_C_FRACTION, rules=B)

        def forage_water(w, c, kg):
            given = cr_.share_scatter(avail_w, w, c, kg.double()).float().double()    # what the body gets (f32)
            self.carry["soil"].view(-1).index_add_(0, w * globe.C + c, -given)
            self.flows["forage_water_kg"].index_add_(0, w, given)
            return given

        ep = cr_.eat_plants(cr, grazers, globe, harvest, share=share, water=forage_water, cell=cell)
        self._add_cells(self._litter, grazers, cell, ep["faeces_c"])
        self.flows["creature_co2_mol"] += ep["co2_mol"].double().sum(1)
        self.flows["done"][:, ACT["forage"]] += ((ep["kg_dry"] > 0) & mask["forage"]).sum(1).double()
        # drinking: the drink action anywhere, everyone at fresh water (WORLD_PROVENANCE['world.background_drink'])
        self._drink(mask["drink"], alive0, cell, T)

        lap("eat_drink")
        # 8. fires, item heat and transforms, decay
        fires_alive0 = fires.alive.clone()
        fire_cells = self._cells(fires.pos, fires.alive)
        item_cells = self._cells(pool.pos, pool.alive)
        substeps = cfg.fire_substeps if bool(fires.alive.any()) else 1
        fs = cf.fire_step(pool, fires, x_o2, DAY_S, ambient_item_k=T.gather(1, item_cells),
                          ambient_fire_k=T.gather(1, fire_cells), radius_m=R, inv=cr.inv, substeps=substeps)
        self._fire_flows(fs, fs["heat_j_fire"] * fires_alive0 + 1e-30 * fires_alive0, fire_cells)
        self.flows["fires_out"] += fs["fires_out"]
        self._decay(item_cells)

        lap("fires_decay")
        # 9. the day's travel, last (resters stay)
        speed = d["speed"] * ~rest
        mv = cr_.move(cr, globe, self.terrain, d["turn"], speed, gravity_m_s2=self.spec_t["gravity_m_s2"],
                      p_o2_pa=p_o2, radius_m=R, surf_v=self.surf_v, seconds=travel_secs, shore_stop=True)
        self._sync_held()
        cell = globe.cell_of(cr.pos)

        lap("movement")
        # 10. physiology at the night's position
        warm = cf.warmth_within(fs["fire_pos"], fs["mean_hrr_w"], cr.pos, R, fire_alive=fires_alive0)
        worn = cf.worn_insulation(pool, cr.inv, cr.mass_kg)
        m = cr.mass_kg.clamp_min(1e-6)
        ins = (cr.genome["insulation"] + worn).clamp_min(0.05)
        c_th = cr_.C_TH * m ** 0.5 / ins
        heat_k = warm * float(cf.EMISSIVITY) * PROJECTED_SHARE * cr_.MEEH_K * m ** (2 / 3) / c_th
        # warmth cuts thermoregulation (PLANET-SPEC 2.11): at most up to the lower critical temperature
        t_body_air = T.gather(1, cell)
        heat_k = torch.minimum(heat_k, (cr_.T_LC0 - cr_.T_LC_PER_INS * ins - t_body_air).clamp_min(0))
        walk_j = cr_.COT_J_KG_M * m ** cr_.COT_EXP * m * walk
        travel = {"cost_j": mv["cost_j"] + (walk_j + work / cr_.MUSCLE_EFF) * af}
        env = {"t_air_k": T, "p_o2_pa": p_o2, "air_density_kg_m3": self.spec_t["air_density_kg_m3"],
               "sound_speed_m_s": self.spec_t["sound_speed_m_s"]}
        ph = cr_.physiology(cr, globe, env, travel, heat_k=heat_k * af, insulation_add=worn,
                            neuron_cost_w=cfg.neuron_cost_w)
        self.flows["work_j"] += (work * af).double().sum(1)
        self.flows["walk_m"] += (walk * af).double().sum(1)
        self.flows["creature_o2_mol"] += ph["o2_mol"].double().sum(1)
        self.flows["creature_co2_mol"] += ph["co2_mol"].double().sum(1)
        self._add_cells(self._vap, alive0, cell, ph["water_loss_kg"])
        self._add_cells(self._wet, alive0, cell, ph["excreted_kg"])

        lap("physiology")
        # 11. deaths
        age0 = cr.age_d.clone()
        dd = cr_.deaths(cr, pool)
        dead = dd["dead"]
        self._add_cells(self._litter, dead, cell, dd["litter_c"])
        self._add_cells(self._wet, dead, cell, dd["water_kg"])
        self.flows["corpse_in_kg"] += dd["items_species_kg"]
        self.flows["deaths_young"] += (dead & (age0 <= NEWBORN_D) & (cr.generation > 0)).sum(1).double()

        lap("deaths")
        # 12. plasticity (WORLD_PROVENANCE['world.outcome'])
        alive1 = cr.alive & alive0
        invested = ph["growth_kg"] * cr_.GROWTH_J_KG
        d_energy = cr.reserve_j - r0 - ph["burnt_kg"] * cr_.TISSUE_J_KG
        margin = (1 - cr_.DEHYDRATION_LETHAL) * cr_.water_norm(cr.mass_kg)
        reward = br.outcome(d_energy, cr.water_kg - w0, cr.mass_kg, cr.health - h0, invested_j=invested,
                            basal_j=br.basal_j_day(cr.mass_kg), water_margin_kg=margin)
        reward = torch.where(alive1, reward, torch.zeros_like(reward))
        base = torch.where(torch.isnan(cr.baseline), reward, cr.baseline)
        mod, cr.baseline = br.modulate(reward, base, alive1, rate=self.bparams.baseline_rate)
        br.hebbian(cr.wh_live, cr.genome["eta"], mod, before, after, alive1, self.bparams.weight_clip,
                   width=width, gen=self.gen)

        lap("plasticity")
        # 13. reproduction
        rp = cr_.reproduce(cr, self.gen, params=self.bparams, specs=cr_.GENE_SPECS)
        self.flows["creature_o2_mol"] += rp["o2_mol"].double().sum(1)
        if rp["child_w"].numel():
            cr.baseline[rp["child_w"], rp["child_n"]] = math.nan
            self._width = None                  # a child's k may exceed the cached width (brain's install duty)

        lap("reproduction")
        # 14. settle the day's flows
        self._settle(before_flows)
        lap("settle")
        self.day += 1
        self._today_t = self._today(before_ledger, before_flows, counts, bdiag, cdiag, reward, alive1)
        self._today_d = None
        return self._today_t

    # --------------------------------------------------------------------------------------- day pieces
    def _lap(self):
        """Phase timer of ``step``: with ``self.profile`` set, each call records the seconds since the last one
        (device synchronised) in ``self.phase_seconds`` (summed over days); otherwise it does nothing."""
        if not getattr(self, "profile", False):
            return lambda name: None
        sync = (torch.cuda.synchronize if self.device.type == "cuda" else (lambda: None))
        sync()
        last = [time.perf_counter()]
        times = self.phase_seconds = getattr(self, "phase_seconds", {})

        def lap(name):
            sync()
            now = time.perf_counter()
            times[name] = times.get(name, 0.0) + now - last[0]
            last[0] = now
        return lap

    def _brain_width(self) -> int:
        """The active brain width for think and hebbian: max(k) over every slot (dead ones too, so it is at
        least the living's max), cached until a birth installs new genomes (one host sync then)."""
        if getattr(self, "_width", None) is None:
            self._width = max(2, int(self.cr.genome["k"].max()))
        return self._width

    def _kin_pair(self):
        """pair(w0, w1) -> [w, N, N] bool: j is a relative of i by descent (parent, child, sibling), not i."""
        cr = self.cr
        N = cr.shape[1]
        eye = torch.eye(N, dtype=torch.bool, device=self.device)

        def kin(w0, w1):
            u, pa = cr.uid[w0:w1], cr.parent[w0:w1]
            child = (pa[:, None, :] == u[:, :, None]) & (u[:, :, None] >= 0)
            par = (pa[:, :, None] == u[:, None, :]) & (u[:, None, :] >= 0)
            sib = (pa[:, :, None] == pa[:, None, :]) & (pa[:, :, None] >= 0)
            return (child | par | sib) & ~eye
        return kin

    def _approach(self, act, dst_pos, dst_mask, sight_m, dst_height, pair=None):
        """Actors (act [W, N]) walk to the nearest destination (dst_pos [W, X, 3] where dst_mask) they can see:
        within sight_m [W, N] and with a clear line of sight (``WORLD_PROVENANCE['world.approach']``). They stop
        STAND_M short of it. ``dst_height`` is the target's height above the surface (m; None: an individual's
        eye). Returns (target index [W, N], -1 none; metres walked [W, N])."""
        cr, globe, R = self.cr, self.globe, self.config.habitat_radius_m
        W, N = cr.shape
        idx, chord = cf._nearest(cr.pos, dst_pos, act, dst_mask, self._chord_sight, pair=pair)
        found = idx >= 0
        dist = torch.where(found, 2 * R * torch.asin((chord / 2).clamp(0, 1)), torch.zeros_like(chord))
        ok = found & (dist <= sight_m)
        w, n = ok.nonzero(as_tuple=True)                          # the actors with a target in range
        went = torch.zeros(W, N, device=self.device)
        if w.numel():
            i = idx[w, n]
            p, q = cr.pos[w, n], dst_pos[w, i]
            eye = sn.EYE_K * cr.mass_kg.clamp_min(1e-6) ** (1 / 3)
            h_p = sn.surface_at(globe, self.surf_v, w, p) + eye[w, n]
            h_q = sn.surface_at(globe, self.surf_v, w, q) + (eye[w, i] if dst_height is None else dst_height)
            clear = sn.line_of_sight(globe, self.surf_v, w, p, q, h_p, h_q, torch.full_like(h_p, float(R)),
                                     floor=self.surf_v.amin(-1))
            d = dist[w, n]
            step = (d - STAND_M).clamp_min(0) * clear
            t = torch.where(d > 0, step / d.clamp_min(1e-9), torch.zeros_like(d))
            cr.pos = cr.pos.index_put((w, n), torch.where((step > 0)[:, None], gb.slerp(p, q, t), p))
            went = went.index_put((w, n), step)
            ok = ok.index_put((w, n), clear)
            self._sync_held()
        return torch.where(ok, idx, torch.full_like(idx, -1)), went

    def _drink(self, act, alive0, cell, T):
        """The drink action anywhere and every living individual at fresh water (background); soil water is owed
        by the cell's carry, sea water comes from the ocean store."""
        cr, R, globe = self.cr, self.config.habitat_radius_m, self.globe
        W, N = cr.shape
        land = self.P["land"]
        soil_eff = (self.clim.soil.double() + self.carry["soil"] / self.cell_m2).clamp_min(0)
        fresh = (land & (soil_eff >= cr_.DRINK_SOIL_MIN)).gather(1, cell)
        dk = cr_.drink(cr, act | (alive0 & fresh), globe, soil_eff.float(), land, radius_m=R, t_air_k=T)
        soil = dk["from_soil"].double()
        self.carry["soil"].view(-1).index_add_(0, (torch.arange(W, device=self.device)[:, None] * globe.C
                                                   + cell).reshape(-1), -soil.reshape(-1))
        self.flows["drunk_soil_kg"] += soil.sum(1)
        sea = dk["from_sea"].double()
        wn = torch.arange(W, device=self.device).repeat_interleave(N)
        cl.take_water(self.clim, globe, self.P, wn, cell.reshape(-1), sea.reshape(-1), store="ocean")
        self.flows["drunk_sea_kg"] += sea.sum(1)
        self.flows["done"][:, ACT["drink"]] += ((dk["from_soil"] + dk["from_sea"] + dk["from_lake"] > 0)
                                                & act).sum(1).double()

    def _strikes(self, act, eff, sight_m):
        """Each striker walks to the nearest other living individual it sees and strikes it
        (WORLD_PROVENANCE['world.strike']). Returns the blow's kinetic energy [W, N] (mechanical work, J) and
        the metres walked."""
        cr, R = self.cr, self.config.habitat_radius_m
        W, N = cr.shape
        dev = self.device
        eye = torch.eye(N, dtype=torch.bool, device=dev)
        tgt, went = self._approach(act, cr.pos, cr.alive, sight_m, None, pair=lambda w0, w1: ~eye)
        hit = act & (tgt >= 0)
        tool = cf.tool_of(self.pool, cr.inv, None, cr.mass_kg)
        energy = cf.strike_energy(tool["head_mass"], tool["handle_len_m"]) * eff
        damage = cf.strike_damage(energy, tool["sharp"], tool["hardness"], self._hide_h,
                                  torch.full_like(energy, self._hide_k))
        w, n = hit.nonzero(as_tuple=True)
        if w.numel():
            t = tgt[w, n]
            lethal = STRIKE_LETHAL_J * cr.mass_kg[w, t].clamp_min(1e-6) / STRIKE_REF_KG
            cr_.injure(cr, w, t, damage[w, n] / lethal)
        hits = hit.sum(1).double()
        self.flows["strikes"] += hits
        self.flows["done"][:, ACT["strike"]] += hits
        self.flows["strike_damage_j"] += (damage * hit).double().sum(1)
        return energy * act, went

    def _fire_flows(self, fl, weight, cells):
        """Book fire_step's or cook's flows: air_kg to the gas column (O2, CO2) and the vapour (H2O), charred
        tissue to CO2 and vapour, the rest of to_soil_kg to the ground; ``weight`` [W, X] and ``cells`` [W, X]
        place the per-world amounts."""
        air = fl["air_kg"]
        self.flows["air_kg"] += air
        self._gas[:, cl.I_O2] += air[:, 0] / M_O2 / self.area_m2
        self._gas[:, cl.I_CO2] += air[:, 1] / M_CO2 / self.area_m2
        self._spread(self._vap, air[:, 2], weight, cells)
        self.flows["to_soil_kg"] += fl["to_soil_kg"]
        charred = fl.get("charred_kg")
        if charred is None:
            charred = torch.zeros_like(fl["to_soil_kg"])
        self._char(charred, weight, cells)
        self._to_ground(fl["to_soil_kg"] - charred, weight, cells)
        self.flows["transformed_kg"] += fl["transformed_kg"]
        self.flows["fire_heat_j"] += fl["heat_j"]
        self.flows["item_heat_j"] += fl["item_heat_j"]

    def _decay(self, cells):
        """crafting.decay with the rotted organic carbon to litter, carbonate carbon to the mineral sink, water and
        minerals to the items' cells (``cells`` [W, I], the items' cells today)."""
        pool = self.pool
        kg0 = pool.comp.double() * (pool.mass.double() * pool.alive)[..., None]
        dc = cf.decay(pool, 1.0, inv=self.cr.inv)
        kg1 = pool.comp.double() * (pool.mass.double() * pool.alive)[..., None]
        lost = (kg0 - kg1).clamp_min(0)
        decayed = dc["decayed_kg"]
        self.flows["decayed_kg"] += decayed
        self._rot(decayed, lost, cells)

    def _rot(self, decayed, lost, cells):
        """Route rotted species mass (decayed [W, S], booked) placed by lost [W, I, S] at item cells [W, I]."""
        c_org = decayed @ self._organic_c
        c_min = decayed @ self._mineral_c
        h2o = decayed @ self._water
        self._spread(self._litter, c_org, lost @ self._organic_c, cells)
        self._spread(self._wet, h2o, lost @ self._water, cells)
        self.flows["mineral_c_kg"] += c_min
        sink = mat.element_mass(decayed)
        sink[:, I_C] -= c_org + c_min
        sink -= h2o[:, None] * self._h2o_el
        self.flows["soil_sink_el"] += sink

    def _relieve_pool(self):
        """Keep POOL_FREE_SHARE of the item slots free (``WORLD_PROVENANCE['world.pool_pressure']``): the oldest
        ground items made only of rotting species rot away at once."""
        pool = self.pool
        I = pool.shape[1]
        need = math.ceil(POOL_FREE_SHARE * I)
        free = I - pool.alive.sum(1)
        if not bool((free < need).any()):
            return
        rots = ((pool.comp > 0) & ~self._rots).sum(-1) == 0
        ok = pool.alive & (pool.holder < 0) & rots
        age = torch.where(ok, pool.age_d.double(), torch.full_like(pool.age_d.double(), -1.0))
        order = torch.argsort(age, dim=1, descending=True, stable=True)
        rank = torch.empty_like(order)
        rank.scatter_(1, order, torch.arange(I, device=self.device).expand_as(order))
        go = ok & (rank < (need - free).clamp_min(0)[:, None])
        cells = self._cells(pool.pos, pool.alive)
        kg = pool.comp.double() * (pool.mass.double() * go)[..., None]
        w, i = go.nonzero(as_tuple=True)
        it.remove(pool, w, i)
        decayed = kg.sum(1)
        self.flows["decayed_kg"] += decayed
        self.flows["items_recycled"] += go.sum(1).double()
        self._rot(decayed, kg, cells)

    def _flush_give(self, store, carry):
        """Give the positive part of a water carry [W, C] to the climate's ``store`` as far as its float32 field
        resolves; the carry keeps the rest."""
        before = getattr(self.clim, store).double()
        w, c = self._wc
        cl.give_water(self.clim, self.globe, self.P, w, c, carry.clamp_min(0).reshape(-1), store=store)
        carry -= (getattr(self.clim, store).double() - before) * self.cell_m2

    def _settle(self, before_flows):
        """Move the day's litter, water and wood exchanges into the carries and flush them into the biosphere and
        climate (WORLD_PROVENANCE['world.settle']); add the day's gas."""
        A = self.area_m2
        w, c = self._wc
        if getattr(self, "strict", False):
            for name, buf in (("litter", self._litter), ("ground water", self._wet), ("vapour", self._vap)):
                if bool((buf < -NEG_TOL_KG).any()):
                    raise RuntimeError(f"negative {name} routed: {float(buf.min()):.3g} kg")
        land = self.P["land"]
        car = self.carry
        car["litter"] += self._litter
        car["soil"] += torch.where(land, self._wet, torch.zeros_like(self._wet))
        car["vapour"] += torch.where(land, torch.zeros_like(self._wet), self._wet) + self._vap
        self.flows["litter_c_kg"] += self._litter.sum(1)
        self.flows["ground_water_kg"] += self._wet.sum(1)
        self.flows["vapour_kg"] += self._vap.sum(1)
        # litter carbon
        before = self.bio.litter_c.double()
        bs.add_litter(self.bio, w, c, car["litter"].clamp_min(0).reshape(-1))
        car["litter"] -= (self.bio.litter_c.double() - before) * self.cell_m2
        # wood owed by the plants
        given = bs.collect_wood(self.bio, w, c, (-car["wood"]).clamp_min(0).reshape(-1))
        car["wood"] += given.reshape(self.W, -1)
        # water: soil (both ways) and vapour
        self._flush_give("soil", car["soil"])
        given = cl.take_water(self.clim, self.globe, self.P, w, c, (-car["soil"]).clamp_min(0).reshape(-1),
                              store="soil")
        car["soil"] += given.reshape(self.W, -1)
        self._flush_give("vapour", car["vapour"])
        # gas: the individuals' respiration plus the fires' and the charred tissue's (already in self._gas)
        self._gas[:, cl.I_O2] -= (self.flows["creature_o2_mol"] - before_flows["creature_o2_mol"]) / A
        self._gas[:, cl.I_CO2] += (self.flows["creature_co2_mol"] - before_flows["creature_co2_mol"]) / A
        cl.add_gas(self.clim, self._gas)

    def _today(self, before_ledger, before_flows, counts, bdiag, cdiag, reward, alive1):
        L = self.cr.ledger
        deaths = torch.stack([L[f"deaths_{c}"] - before_ledger[f"deaths_{c}"] for c in cr_.CAUSES[1:]], -1)
        precip = (cdiag["precipitation"].double() * self.globe.area64).sum(-1) / FOUR_PI
        t = {"actions": counts, "births": L["births"] - before_ledger["births"], "deaths": deaths,
             "fires_lit": self.flows["fires_lit"] - before_flows["fires_lit"],
             "strikes": self.flows["strikes"] - before_flows["strikes"],
             "npp_kg_c_m2_day": bdiag["mean_npp"], "gpp_kg_c_m2_day": bdiag["mean_gpp"],
             "mean_t_k": cdiag["mean_t"], "precipitation_kg_m2_day": precip,
             "mean_reward": (reward * alive1).sum(1) / alive1.sum(1).clamp_min(1)}
        return _Today(t, self.day)

    @property
    def today(self):
        """The last day's counts per world (lists), or None before the first step."""
        if getattr(self, "_today_d", None) is None:
            t = getattr(self, "_today_t", None)
            if t is None:
                return None
            self._today_d = dict(t._dict()) if isinstance(t, _Today) else dict(t)
        return self._today_d

    # --------------------------------------------------------------------------------------- stores and ledgers
    def _stores(self) -> dict:
        """Every store the ledgers count, float64 [W] (absolute: kg, mol, J per world)."""
        A = self.area_m2
        clim, bio = self.clim, self.bio
        body = cr_.stocks(self.cr)
        L = self.cr.ledger
        lm = cf.ledger_mass(self.pool, self.fires)                              # [W, S]
        el = mat.element_mass(lm)                                               # [W, E]
        car = self.carry
        return {"co2_c": clim.gas[:, cl.I_CO2] * M_C * A, "organic_c": bs.organic_carbon(bio) * A,
                "body_c": body["carbon"], "item_c": el[:, I_C], "o2_mol": clim.gas[:, cl.I_O2] * A,
                "climate_w": cl.water_total(clim, self.globe) * A, "body_w": body["water"],
                "item_w": lm @ self._water, "item_el": el, "body_j": body["energy"],
                "gas_ext": clim.gas_external * A, "water_ext": clim.water_external * A,
                "c_harvested": bio.c_harvested * A, "c_wood": bio.c_wood_collected * A,
                "c_buried": bio.c_buried * A, "c_litter": bio.c_litter_added * A,
                "co2_out": clim.co2_outgassed * A, "co2_weath": clim.co2_weathered * A,
                "carry_w": (car["soil"] + car["vapour"]).sum(1),
                "carry_c": car["litter"].sum(1) + car["wood"].sum(1) * cf.WOOD_C_FRACTION,
                "cr_o2": L["o2_mol"].clone(), "cr_ox_c": L["oxidised_c"].clone(),
                "cr_met_w": L["metabolic_w"].clone()}

    def ledgers(self, tensors: bool = False) -> dict:
        """Per-world ledger errors (see the module docstring): {name: {"error", "scale", "rel"}} with [W]
        lists (tensors with ``tensors=True``); ``items`` and ``deposits`` also have ``elements`` / ``species``."""
        s, b, f = self._stores(), self.base, self.flows
        A = self.area_m2
        d = {k: s[k] - b[k] for k in s}
        tiny = 1e-30
        out = {}

        def put(name, err, scale, **extra):
            scale = scale.abs().clamp_min(tiny)
            out[name] = {"error": err, "scale": scale, "rel": err.abs() / scale, **extra}

        # energy, carbon and water of the individuals (creatures' own ledger)
        e = cr_.ledger_errors(self.cr)
        put("energy", e["energy"], e["energy_scale"])
        put("bodies_carbon", e["carbon"], e["carbon_scale"])
        put("bodies_water", e["water"], e["water_scale"])
        # carbon: the biosphere's own closure (air CO2 + organic pools against its counters) plus the bookings
        # between modules (bodies + items + carries against harvest, wood, deposits, litter, the gas added and
        # the mineral sink), computed directly: the total as a difference of stores would lose the small flows
        dep_c = f["dep_in_kg"] @ self._carbon
        c_bio = bs.carbon_ledger(self.bio, self.clim) * A
        gas_c = d["gas_ext"][:, cl.I_CO2] * M_C
        flows_c = d["c_harvested"] + d["c_wood"] + dep_c - d["c_litter"] - gas_c - f["mineral_c_kg"]
        c_mob = d["body_c"] + d["item_c"] + d["carry_c"] - flows_c
        stores = s["co2_c"] + s["organic_c"] + s["body_c"] + s["item_c"]
        put("carbon", c_bio + c_mob, stores + d["c_buried"].abs() + dep_c)
        put("carbon_bio", c_bio, s["co2_c"] + s["organic_c"])
        put("carbon_mobile", c_mob, s["body_c"] + s["item_c"] + d["c_harvested"] + d["c_wood"] + dep_c
            + d["c_litter"] + gas_c.abs() + s["carry_c"].abs() + f["mineral_c_kg"])
        # oxygen (mol O2): the biosphere's closure plus the climate's record of what other modules added against
        # the individuals' own record (creatures' o2_mol), the fires' (items-checked air_kg) and charred tissue
        o_bio = bs.oxygen_ledger(self.bio, self.clim) * A
        fire_o2 = f["air_kg"][:, 0] / M_O2
        o_mob = d["gas_ext"][:, cl.I_O2] - (-d["cr_o2"] + fire_o2 - f["char_o2_mol"])
        put("oxygen", o_bio + o_mob, s["o2_mol"])
        put("oxygen_bio", o_bio, s["o2_mol"])
        put("oxygen_mobile", o_mob, d["cr_o2"] + fire_o2.abs() + f["char_o2_mol"] + d["gas_ext"][:, cl.I_O2].abs())
        # O atoms the individuals took from the air as O2 (climate's record) = O in the CO2 of the fat they
        # oxidised + O in their metabolic water (creatures' record)
        o2_air = -(d["gas_ext"][:, cl.I_O2] - fire_o2 + f["char_o2_mol"])
        o_ret = 2 * d["cr_ox_c"] / M_C + d["cr_met_w"] / M_H2O
        put("oxygen_atoms", 2 * o2_air - o_ret, 2 * o2_air.abs() + o_ret)
        # water (kg): the climate's closure plus bodies, items and carries against what the climate handed out
        # (water_external), the water fires and transforms made and the metabolic water bodies made
        chem = f["air_kg"][:, 2]
        w_clim = cl.water_ledger(self.clim, self.globe) * A
        w_mob = d["body_w"] + d["item_w"] + d["carry_w"] - chem - d["cr_met_w"] - d["water_ext"]
        put("water", w_clim + w_mob, s["climate_w"] + s["body_w"] + s["item_w"])
        put("water_climate", w_clim, s["climate_w"])
        put("water_mobile", w_mob, s["body_w"] + s["item_w"] + d["water_ext"].abs() + chem.abs()
            + f["drunk_soil_kg"] + f["drunk_sea_kg"] + f["vapour_kg"] + f["ground_water_kg"].abs()
            + f["forage_water_kg"] + d["cr_met_w"] + s["carry_w"].abs())
        # items: element mass of items and fire beds
        wood = torch.zeros_like(f["dep_in_kg"])
        wood[:, mat.IDX["wood"]] = f["wood_in_kg"]
        inflow = f["dep_in_kg"] + wood + f["corpse_in_kg"]
        outflow = f["eaten_kg"] + f["scraps_kg"] + f["to_soil_kg"] + f["decayed_kg"]
        flow_el = mat.element_mass(inflow) - mat.element_mass(outflow) - f["air_kg"] @ self._air_el
        el_err = d["item_el"] - flow_el
        put("items", el_err.sum(-1), s["item_el"].sum(-1) + inflow.sum(-1) + outflow.sum(-1)
            + (f["air_kg"].abs().sum(-1)), elements=el_err)
        # deposits: the ground stock's loss against what collect made of it (kg per crust species)
        lost = -((self.stock - self.stock0) * self.cell_m2[None, :, None]).sum(1)          # [W, SD]
        dep_err = f["dep_in_kg"][:, :SD] - lost
        put("deposits", dep_err.sum(-1), f["dep_in_kg"][:, :SD].sum(-1) + lost.abs().sum(-1) + 1e-9,
            species=dep_err)
        if tensors:
            return out
        return {k: {kk: vv.tolist() for kk, vv in v.items()} for k, v in out.items()}

    def worst_ledger(self, led: dict | None = None) -> float:
        """The largest relative error among the checked ledgers (CHECKED_LEDGERS) over the worlds."""
        led = self.ledgers() if led is None else led
        return max(max(led[k]["rel"]) for k in CHECKED_LEDGERS if k in led)

    def gates(self) -> dict:
        """The world gates (``WORLD_PROVENANCE['world.gates']``): checked ledgers within LEDGER_TOL; after
        GATE_DAYS days at least FOUNDER_GATE of the founders alive; at least NEWBORN_GATE of the newborns that
        died or reached NEWBORN_D days lived that long. A gate not yet decidable (too early, no births) is None.
        Returns per-world values and the pass flags."""
        led = self.ledgers()
        worst = [max(led[k]["rel"][w] for k in CHECKED_LEDGERS) for w in range(self.W)]
        cr = self.cr
        founders = (cr.alive & (cr.generation == 0)).sum(1).double() / self.config.founders
        births = cr.ledger["births"]
        old = (cr.alive & (cr.generation > 0) & (cr.age_d > NEWBORN_D)).sum(1).double()
        young_dead = self.flows["deaths_young"]
        decided = old + young_dead
        newborn = torch.where(decided > 0, old / decided.clamp_min(1), torch.full_like(old, math.nan))
        out = {"day": self.day, "worst_ledger_rel": worst, "founders_alive_share": founders.tolist(),
               "births": births.tolist(), "newborn_survival": newborn.tolist(),
               "ledgers_closed": [x < LEDGER_TOL for x in worst]}
        out["founders_viable"] = ([x >= FOUNDER_GATE for x in out["founders_alive_share"]]
                                  if self.day >= GATE_DAYS else None)
        out["newborns_viable"] = [None if math.isnan(x) else x >= NEWBORN_GATE for x in out["newborn_survival"]]
        return out

    # --------------------------------------------------------------------------------------- reports
    def summary(self) -> dict:
        """JSON-ready state of every world (population, births and deaths, brains, bodies, climate, plants,
        fires, items, transforms, encounters) plus the share of CO2-capped planets."""
        cr, pool, fires = self.cr, self.pool, self.fires
        L = cr.ledger
        a = cr.alive
        n = a.sum(1)
        nf = n.clamp_min(1).double()
        g = cr.genome

        def mean(t):
            return ((t.double() * a).sum(1) / nf).tolist()

        means = {"k": mean(g["k"].float()), "diet": mean(g["diet"]), "mass": mean(cr.mass_kg),
                 "reserve": mean(cr.reserve_j / cr_.reserve_max(cr.mass_kg.clamp_min(1e-6))),
                 "water": mean(cr.water_kg / cr_.water_norm(cr.mass_kg.clamp_min(1e-6))),
                 "health": mean(cr.health), "insulation": mean(g["insulation"]), "speed": mean(g["speed"]),
                 "age_d": mean(cr.age_d)}
        press = cl.pressures(self.clim, self.globe, self.P)
        area = self.globe.area64
        T = self.clim.T.double()
        land = self.P["land"]
        land_a = (land.double() * area).sum(-1).clamp_min(1e-30)
        dom = pool.comp.argmax(-1)
        icls = torch.where(pool.alive, self._species_class[dom], torch.full_like(dom, -1))
        by_class = [{c: int((icls[w] == j).sum()) for j, c in enumerate(mat.CLASSES)} for w in range(self.W)]
        tnames = [t.name for t in mat.TRANSFORMS]
        f = self.flows
        tries, done = f["tries"].tolist(), f["done"].tolist()
        reach = ("strike", "share", "eat_meat", "feed_fire", "heat_item", "cook", "make_fire", "drink", "forage")
        gross_w = f["drunk_soil_kg"] + f["forage_water_kg"] + f["ground_water_kg"].abs() + f["vapour_kg"]
        carry_w = (self.carry["soil"].abs() + self.carry["vapour"].abs()).sum(1)
        carry_c = (self.carry["litter"].abs().sum(1) + self.carry["wood"].abs().sum(1) * cf.WOOD_C_FRACTION)
        worlds = []
        today = self.today
        founders_alive = (a & (cr.generation == 0)).sum(1).tolist()
        for w in range(self.W):
            spec = self.specs[w]
            items_alive = int(pool.alive[w].sum())
            row = {
                "seed": self.seeds[w], "source": spec.source, "population": int(n[w]),
                "founders_alive": founders_alive[w],
                "births": int(L["births"][w]),
                "deaths": {c: int(L[f"deaths_{c}"][w]) for c in cr_.CAUSES[1:]},
                "generation_max": int((cr.generation[w] * a[w]).max()) if int(n[w]) else 0,
                "mean_brain_units": means["k"][w], "mean_diet": means["diet"][w], "mean_mass_kg": means["mass"][w],
                "mean_reserve_frac": means["reserve"][w], "mean_water_frac": means["water"][w],
                "mean_health": means["health"][w],
                "mean_insulation": means["insulation"][w], "mean_speed_gene": means["speed"][w],
                "mean_age_d": means["age_d"][w],
                "mean_t_k": float((T[w] * area).sum() / FOUR_PI), "land_t_k": float((T[w] * area * land[w]).sum()
                                                                                    / land_a[w]),
                "min_t_k": float(T[w].min()), "max_t_k": float(T[w].max()),
                "p_o2_pa": float(press["O2"][w]), "p_co2_pa": float(press["CO2"][w]),
                "p_total_pa": float(press["total"][w]),
                "npp_kg_c_m2_yr": (today["npp_kg_c_m2_day"][w] * 365.25) if today else None,
                "plant_c_kg_m2": float((self.bio.plant_c[w].double() * area).sum() / FOUR_PI),
                "wood_c_kg_m2": float((self.bio.wood_c[w].double() * area).sum() / FOUR_PI),
                "fires_alive": int(fires.alive[w].sum()), "fires_lit": int(f["fires_lit"][w]),
                "items_alive": items_alive, "item_pool_fill": items_alive / pool.shape[1],
                "items_recycled": int(f["items_recycled"][w]), "items_by_class": by_class[w],
                "transforms_kg": {nm: round(float(f["transformed_kg"][w, j]), 6) for j, nm in enumerate(tnames)},
                "strikes": int(f["strikes"][w]), "co2_capped": bool(spec.co2_capped),
                "tidally_locked": bool(spec.tidally_locked),
                "encounters": {nm: {"tries": int(tries[w][ACT[nm]]), "done": int(done[w][ACT[nm]])} for nm in reach},
                "walk_km": round(float(f["walk_m"][w]) / 1000.0, 3),
                "carry_share": {"water": float(carry_w[w] / gross_w[w].clamp_min(1e-30)),
                                "carbon": float(carry_c[w] / f["litter_c_kg"][w].clamp_min(1e-30))},
            }
            if today:
                row["today"] = {"births": today["births"][w],
                                "deaths": {c: today["deaths"][c][w] for c in today["deaths"]},
                                "actions": dict(zip(ACTIONS, today["actions"][w])),
                                "fires_lit": today["fires_lit"][w], "strikes": today["strikes"][w],
                                "mean_reward": today["mean_reward"][w]}
            worlds.append(row)
        capped = sum(bool(s.co2_capped) for s in self.specs)
        return {"day": self.day, "worlds": worlds, "co2_capped_share": capped / self.W}

    def provenance(self) -> dict:
        """Every module's provenance merged (dotted keys -> (tag, note)), plus each world's PlanetSpec and the
        innate wiring's tag per world (chain where the inputs reach the bodies era, else new_rule)."""
        out = dict(WORLD_PROVENANCE)
        for prefix, table in (("climate", cl.PROVENANCE), ("climate.params", self.P["provenance"]),
                              ("biosphere", bs.PROVENANCE), ("creatures", cr_.PROVENANCE), ("senses", sn.PROVENANCE),
                              ("brain", br.PROVENANCE), ("globe", gb.PROVENANCE)):
            for k, v in table.items():
                out[k if k.startswith(prefix.split(".")[0] + ".") else f"{prefix}.{k}"] = tuple(v)
        out.update(mat.provenance())
        out.update(cf.provenance())
        for w, spec in enumerate(self.specs):
            for k, v in spec.provenance.items():
                out[f"spec[{w}].{k}"] = tuple(v)
            bodies = "bodies" in (self.inputs[w].get("links") or ())
            out[f"world[{w}].innate"] = (("chain", "the bodies era gives grazer lineages: the innate forage relay "
                                                   "(brain PROVENANCE['innate'])") if bodies else
                                         ("new_rule", "these inputs have no bodies era: the innate forage relay is "
                                                      "applied as a rule, not taken from the chain"))
        return out

    # --------------------------------------------------------------------------------------- frames
    def static(self, w: int = 0) -> dict:
        """The viewer's static block of world w (FRAME_SCHEMA)."""
        spec = self.specs[w].to_dict()
        centers = self.globe.centers.detach().cpu()
        elev = self.terrain["elevation_m"][w].detach().cpu()
        return {"schema": FRAME_SCHEMA, "seed": self.seeds[w],
                "planet": {k: spec[k] for k in STATIC_PLANET},
                "habitat_radius_m": float(self.config.habitat_radius_m), "G": self.globe.G, "cells": self.globe.C,
                "centers": [[round(v, 4) for v in row] for row in centers.tolist()],
                "elevation_m": [int(round(v)) for v in elev.tolist()],
                "land": [int(v) for v in self.terrain["land"][w].detach().cpu().tolist()],
                "sea_level_m": round(float(self.terrain["sea_level_m"][w]), 2),
                "relief_exaggeration": float(self.config.relief_exaggeration),
                "species": list(mat.SPECIES), "actions": list(ACTIONS), "item_classes": list(mat.CLASSES)}

    def sun(self, w: int = 0) -> list:
        """Unit vector toward the star at the middle of the current day (planet coordinates: the climate's
        substellar point of a locked planet is lon 0, lat 0; a rotating planet turns once per solar day)."""
        spec = self.specs[w]
        dec = float(cl.declination(self.P, self.day)[w])
        if spec.tidally_locked or not math.isfinite(spec.day_length_s):
            lon = 0.0
        else:
            t = (self.day + 0.5) * DAY_S
            lon = -2 * math.pi * ((t / spec.day_length_s) % 1.0)
        return [round(math.cos(dec) * math.cos(lon), 4), round(math.cos(dec) * math.sin(lon), 4),
                round(math.sin(dec), 4)]

    def frame(self, w: int = 0, fields: bool = False) -> dict:
        """The viewer's frame of world w for the current day (FRAME_SCHEMA); ``fields`` adds the cell fields."""
        cr, pool, fires = self.cr, self.pool, self.fires
        a = cr.alive[w]
        vocal = cr.calls[w].argmax(-1)
        cols = torch.stack([cr.uid[w].double(), cr.pos[w, :, 0].double(), cr.pos[w, :, 1].double(),
                            cr.pos[w, :, 2].double(), cr.heading[w].double(), cr.mass_kg[w].double(),
                            cr.founder[w].double(), cr.loud[w].double(), vocal.double(),
                            cr.genome["k"][w].double(), self.last_action[w].double(), cr.health[w].double(),
                            cr.genome["diet"][w].double()], -1)
        rows = cols[a].cpu().tolist()
        digits = (None, 4, 4, 4, 3, 3, None, 2, None, None, None, 3, 3)
        agents = [[int(v) if dg is None else round(v, dg) for v, dg in zip(r, digits)] for r in rows]
        fa = fires.alive[w]
        fr = torch.cat([fires.pos[w], fires.temp_k[w, :, None]], -1)[fa].cpu().tolist()
        fire_rows = [[round(r[0], 4), round(r[1], 4), round(r[2], 4), round(r[3], 1)] for r in fr]
        ground = pool.alive[w] & (pool.holder[w] < 0)
        klass = self._species_class[pool.comp[w].argmax(-1)]
        ir = torch.cat([pool.pos[w], klass[:, None].float(), pool.mass[w, :, None]], -1)[ground][:MAX_FRAME_ITEMS]
        item_rows = [[round(r[0], 4), round(r[1], 4), round(r[2], 4), int(r[3]), round(r[4], 3)]
                     for r in ir.cpu().tolist()]
        out = {"day": self.day, "sun": self.sun(w), "agents": agents, "fires": fire_rows, "items": item_rows}
        if fields:
            clim, bio = self.clim, self.bio
            rules = self.P["rules"]
            cover = 1 - torch.exp(-bio.plant_c[w].double() / self.B["rules"].fapar_kg_c)
            ice = cl.ice_fraction(clim.T[w], rules) > 0.5
            snow = (clim.snow[w] >= 0.5 * rules.snow_mask_kg_m2) | (ice & ~self.P["land"][w])
            out["fields"] = {"T_k": [int(round(v)) for v in clim.T[w].cpu().tolist()],
                             "plant": (cover * 255).round().clamp(0, 255).int().cpu().tolist(),
                             "snow": snow.int().cpu().tolist(),
                             "soil": (clim.soil[w].double() / rules.bucket_kg_m2 * 255).round().clamp(0, 255)
                             .int().cpu().tolist()}
        return out

    # --------------------------------------------------------------------------------------- state
    def _live(self) -> dict:
        """The world's dynamic tensors by reference (no copies), in the layout of :meth:`state_dict`."""
        cr = self.cr
        return {"gen": self.gen.get_state(), "terrain": dict(self.terrain), "stock": self.stock,
                "stock0": self.stock0,
                "climate": {f.name: getattr(self.clim, f.name) for f in fields(self.clim)},
                "bio": {f.name: getattr(self.bio, f.name) for f in fields(self.bio)},
                "items": {n: getattr(self.pool, n) for n in it.ITEM_FIELDS},
                "fires": {n: getattr(self.fires, n) for n in it.FIRE_FIELDS},
                "creatures": {**{n: getattr(cr, n) for n in cr_.Creatures.TENSORS}, "genome": dict(cr.genome),
                              "ledger": dict(cr.ledger)},
                "flows": dict(self.flows), "base": dict(self.base), "carry": dict(self.carry),
                "last_action": self.last_action}

    def state_dict(self, device="cpu") -> dict:
        """Everything a resume needs, copied straight to ``device`` (the host by default, so a GPU world needs
        no second copy of its state on the GPU): config, seeds, chain inputs, day, generator, terrain, deposits,
        climate, biosphere, items, fires, individuals, flows, carries and the ledger baseline."""
        live = self._live()
        out = {k: _host(v, device) for k, v in live.items()}
        out.update({"format": STATE_FORMAT, "config": self.config.to_dict(), "seeds": list(self.seeds),
                    "inputs": json.loads(json.dumps(self.inputs)), "input_notes": list(self.input_notes),
                    "rng_seed": self.rng_seed, "day": self.day,
                    "reports": json.loads(json.dumps(self.reports)),
                    "today": json.loads(json.dumps(self.today))})
        return out

    @classmethod
    def from_state(cls, state: dict, device=None, *, own: bool = False) -> "PlanetWorld":
        """A world restored from :meth:`state_dict` (exact on the CPU). The static parts (specs, globe,
        parameters) are rebuilt from the stored chain inputs and terrain; nothing is spun up again. With
        ``own`` the world takes the state's tensors without copying them where they already are on the
        device (the caller hands the dict over and must not use it again)."""
        if state.get("format") != STATE_FORMAT:
            raise ValueError(f"not a {STATE_FORMAT} state")
        self = cls.__new__(cls)
        self.config = PlanetConfig.from_dict(state["config"])
        self.config.check()
        self.seeds = [int(s) for s in state["seeds"]]
        self.W = len(self.seeds)
        dev = torch.device(device) if device is not None else state["stock"].device
        self.device = dev
        self.inputs = state["inputs"]
        self.input_notes = list(state["input_notes"])
        self._build_specs()
        self.rng_seed = int(state["rng_seed"])
        self.gen = torch.Generator(device=dev)
        gstate = state["gen"].cpu()
        try:
            self.gen.set_state(gstate)
        except RuntimeError as e:                        # another device type: the stream cannot continue
            warnings.warn(f"generator state not restorable on {dev} ({e}); reseeded, the resume is not exact")
            self.gen.manual_seed(self.rng_seed + int(state["day"]))
        if own:
            mv = lambda t: t.to(dev) if torch.is_tensor(t) else t
        else:
            mv = lambda t: t.to(dev, copy=True) if torch.is_tensor(t) else t
        self.globe = gb.Globe(self.config.G, dev)
        self.terrain = {k: mv(v) for k, v in state["terrain"].items()}
        self.stock = mv(state["stock"])
        self.stock0 = mv(state["stock0"])
        self._build_params()
        self.clim = cl.ClimateState(**{k: mv(v) for k, v in state["climate"].items()})
        self.bio = bs.BioState(**{k: mv(v) for k, v in state["bio"].items()})
        self.pool = it.ItemPool(**{k: mv(state["items"][k]) for k in it.ITEM_FIELDS})
        self.fires = it.FirePool(**{k: mv(state["fires"][k]) for k in it.FIRE_FIELDS})
        c = state["creatures"]
        self.cr = cr_.Creatures(**{n: mv(c[n]) for n in cr_.Creatures.TENSORS},
                                genome={k: mv(v) for k, v in c["genome"].items()},
                                ledger={k: mv(v) for k, v in c["ledger"].items()})
        self.flows = {k: mv(v) for k, v in state["flows"].items()}
        self.base = {k: mv(v) for k, v in state["base"].items()}
        self.carry = {k: mv(v) for k, v in state["carry"].items()}
        self.day = int(state["day"])
        self.last_action = mv(state["last_action"])
        self.reports = state["reports"]
        if state.get("today") is not None:
            self._today_t = dict(state["today"])
            self._today_d = dict(state["today"])
        self._new_day()
        return self

    def state_hash(self) -> str:
        """SHA-256 over every dynamic tensor of the world (sorted keys; name, dtype, shape, bytes) and the day,
        streamed from the live tensors (one host copy at a time)."""
        h = hashlib.sha256(STATE_FORMAT.encode())
        h.update(f"|day={self.day}|".encode())
        live = self._live()

        def walk(prefix, obj):
            if torch.is_tensor(obj):
                t = obj.detach().cpu().contiguous()
                h.update(f"|{prefix}|{t.dtype}|{tuple(t.shape)}|".encode())
                if t.numel():
                    h.update(ctypes.string_at(t.data_ptr(), t.numel() * t.element_size()))
            elif isinstance(obj, dict):
                for k in sorted(obj):
                    walk(f"{prefix}.{k}", obj[k])
            elif isinstance(obj, (int, float, bool, str)) or obj is None:
                h.update(f"|{prefix}={obj!r}|".encode())

        for key in ("gen", "terrain", "stock", "stock0", "climate", "bio", "items", "fires", "creatures", "flows",
                    "base", "carry", "last_action"):
            walk(key, live[key])
        return h.hexdigest()

    def tensor_bytes(self) -> int:
        """Bytes held by the world's dynamic tensors (walked live, no copies)."""
        total = 0
        seen = set()

        def walk(obj):
            nonlocal total
            if torch.is_tensor(obj):
                if id(obj) not in seen:
                    seen.add(id(obj))
                    total += obj.numel() * obj.element_size()
            elif isinstance(obj, dict):
                for v in obj.values():
                    walk(v)
        live = self._live()
        for key in ("terrain", "stock", "stock0", "climate", "bio", "items", "fires", "creatures", "flows", "base",
                    "carry"):
            walk(live[key])
        return total
