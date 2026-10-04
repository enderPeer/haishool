"""The v3 world: a real-size planet, true-scale habitat patches and the bodies' one feedback loop
(PLANET-V3-SPEC sections 0, 1, 6, 9 and 10).

:class:`World3` puts the v3 modules together and adds nothing behavioural. What it supplies is the planet (the chain's
starting conditions through ``formation3``), the physics of every layer (version 2's climate and coarse global
biosphere, ``patch``, ``body``, ``manipulate``, ``senses3``), the founders of the replication loop (``body.found``:
random genomes, uniformly random positions and headings, nothing pre-installed) and the books. It never decides what
a body does: every motor output comes from the body's own network, every event is a physical one.

Set-up (each world's draws from its own generator, seeded by its seed alone, ``PROVENANCE['SETUP_SALT']``; the
order is fixed):

1. Chain inputs per seed from the source the experiment chose (``planet.chain``: ``world7`` the cached chain,
   ``synthetic`` or ``earth``; never substituted), ``formation3.build_planet3`` and ``check_spec3``. A world7 seed whose
   chain gives no planet to found on (no star, no planet, no habitable planet, or a planet whose chain made no bodies
   era) is recorded with that outcome and gets no world (:func:`chain_outcome`, ``PROVENANCE['absent']``; the caches
   are read first, so a host without world7 runs every cached seed); it is never skipped and never replaced; a seed
   that cannot be resolved is an error (``PROVENANCE['ERROR_KIND']``).
2. The globe (``planet.globe``, G cells per face edge) at each planet's real radius, its real-scale terrain
   (``formation3.make_terrain3``; draw 1), version 2's climate parameters with each planet's own radius for the
   ocean. The climate spins up (version 2's procedure); its cloud albedo is calibrated toward formation's T_s only
   for planets that are neither ``frozen_mean`` nor ``runaway``: for those the formation T_s is no target (formation
   has no ice-albedo feedback), so the climate keeps the spec's planetary albedo and finds its own state. Version 2's
   global biosphere then spins up with the chain's air held. It is the coarse global layer of the planet, run without
   version 2's seed rain (its plants are sown once and live on their own production); the patches have their own.
3. Patches: ``patches`` global land cells per world drawn at random by area over land (``patch.choose_patch_cells``;
   draw 2), with no habitability bias. A world without land draws its patches over all its cells: they are sea
   (``PROVENANCE['patch_cells']``). Fine elevation =
   the cell's elevation + periodic fractal detail from the planet's own relief (draw 3), conditioned as
   ``patch.make_geometry`` does. Mineral deposits of the chain's crust (version 2's ``globe.make_deposits``; draw 4)
   are the ground that strokes can break pieces off, kept per fine cell. The plants are sown sparsely and uniformly
   (draw 5; their carbon is taken from the climate's air) and grow with the coupled daily climate for
   ``veg_spin_days`` (seed dispersal is the only way they spread).
4. Founders (``body.found_into``, draws 6): ``founders`` per arena with random genomes (``brain3.random_genome3`` and
   ``random_body_genes``), drawn on the CPU per world into the state allocated once on the device, at the air
   temperature where they stand. Empty item and fire pools. The world generator, the shadow's generator and the
   behavioural baseline's seed come from the run seed (derived from the sampled seeds: draws shared by the worlds
   stepped together). The founding day (its day of the year and the declination), whether each world's air allows
   fire, and what the global biosphere's spin-up made or held (``PROVENANCE['spin_up_replenishment']``) are
   reported.

A day: the climate and the global biosphere step once; the patches' forcing comes from their global cell; the patches
step once (water, then plants); the arenas' air is reduced to each fine cell's height (``PROVENANCE['barometric']``);
then ``bouts`` bouts; then items rot (``manipulate.decay_step``) and every flow of the day is booked to the global
layer.

A bout (dt = a day / ``bouts``), for every arena at once (tensors [A, N]; no loop over bodies):

1. Senses: ``senses3.observe`` of ``body.sense_view`` (mass, mouth, reach, call power: one definition) for the living
   slots, with the bout's light (``scene_from_patch``), the arena's air, smells, items and fires. Lineage ids never
   reach it. On one bout a day the senses report their cost bounds, and the founders' networks are replayed on the
   bodies' inputs as the behavioural baseline.
2. Brain: ``brain3.think`` and ``decode`` with the units each body's brain tissue holds (``body.active_units``): motor
   intensities, nothing else.
3. Contact physics: ``manipulate.bout`` (grip, press, rub, force, then the grips that open: continuous-time events at
   rates set by the intensities), the pieces broken off the ground booked out of the patch (``take_ground``) and the
   struck cells' deposits; ``manipulate.heat_step`` (fires, heating, transforms, cooking); strokes wound the struck,
   a bare limb's charred skin is a burn, and the heat the hands leave in the body is kept.
4. Locomotion: the locomotion share of the bout's time (``thrust_share``) at the muscles' full sustained power, with
   the force-generation cost of running and the held load, bottom walking or paddling in water (:func:`move_kt`),
   in each body's own sub-steps; in each sub-step the mouth acts on what it touches (``body.ingest`` for the
   sub-step's time, the still mouth's disc once a bout), held items follow the mouth, and the fires' flux at the
   body is averaged.
5. ``body.metabolism`` (with the bout's locomotion and time under water, bites, manipulation work, calls, the fires'
   and the hands' heat, and the development of the next child at the divide output), then ``body.deaths`` (dead
   bodies become carcass items through ``manipulate.spawn_items``).
6. Hebbian plasticity of the living (``brain3.hebbian``), modulated by the evolved readout of the body's own state
   after the bout.
7. ``body.divide``: the bodies whose child's development is complete divide; children are mutated copies. A birth
   that finds no slot fails, loses its development and marks the run ``capacity_bound``.
8. The neutral shadow (``body.Shadow``) takes the bout's numbers of deaths and births; the smell fields advance.

Decision interval and movement sub-steps (``PROVENANCE['bouts']``, ``['substeps']``). Physically a body must look
again before it has passed what it could sense: a decision interval dt needs v dt below the senses' range. The
default is 24 bouts (1 h): the brain senses and acts hourly and the sun angle is resolved hourly (dawn, dusk and
night are distinct, so nocturnality can emerge). Every rate of the bodies is per second (development, digestion,
drinking, the manipulation events, locomotion's time share), so the bout length changes numerics, not physics. Within a
bout each body moves in its own K sub-steps, K = ceil(its longest possible bout travel / ``substep_m``), at most
``max_substeps``: each sub-step moves it at most ``substep_m`` (two fine cells, the reach of the fine 3 x 3 contact
hash) and its mouth meets the plants, water, items and bodies along the path. A body that would need more sub-steps
is not slowed: it takes longer sub-steps, counted as ``coarse_moves``; the moving bodies whose bout travel passes the
senses' far range (128 m) are counted too (``travel_beyond_far``).

Locomotion cost (:func:`move_kt`, ``PROVENANCE['KT_COST_J_PER_N']``, a contract change the owner has not yet
decided). ``body.move`` charges c_m M g per metre over a muscle efficiency, which gives about 7-9 J/(kg m) at every
size, against 35-75 measured for 2-20 g runners. The metabolic cost of running is set by the force that supports the
body's weight during each foot contact (Kram & Taylor 1990): E/t = c W / t_c, W = M g, t_c = L_c / v the time of one
ground contact, L_c the distance the body travels during it, which scales with leg length; here the body's own length
(the ellipsoid's 2a: the body plan has no separate leg). Per metre that is c M g / L_c: each contact costs c M g.
Air drag (1/2 rho C_d A v^3, absent on a treadmill) and climbing (M g dh) are added at the muscle efficiency. In water
deeper than its height a body denser than the water there treads it while it moves when its power covers the tread
power (``body.tread_power_w``), else walks the bottom (the same law with its weight in water, water's drag) with its
airway under water, as it is whenever it stands there (its O2 store against its O2 use, ``body.breath_hold_s``; an
unconscious body acts no more, ``PROVENANCE['drowning_gate']``); one that floats paddles at the surface at a
propulsive efficiency PADDLE_EFF against the drag of its bow wave (``WAVE_DRAG_MAX``). The metabolic power while
moving is the sustained power (capped by the body's O2 supply) over the muscle efficiency.

Books (:meth:`World3.ledgers`). Every layer closes on its own (climate water; biosphere carbon and oxygen; patch carbon,
water and nitrogen; bodies' energy, carbon, water, nitrogen, inert matter, salt, O atoms, CO2; items' mass and
elements), the bodies' exchanges close against the patch's counters (``body.exchange_errors``), and the local carbon and
nitrogen of each arena (patch + bodies + items + what is in transit between them) close against the flows across its
boundary: carbon from the air (the patch's net fixation), the bodies' CO2, the fires' CO2, the ground's minerals;
nitrogen fixed. What crosses to the global layer is booked there: the patches' net fixation, the bodies' and the
fires' gases with ``climate.add_gas`` (and nitrogen fixed as N2 taken), the water the bodies and items give to the air
into the cell's vapour, the sea water drunk from the ocean. Float32 fields cannot hold a patch's kilograms on a
planet's columns, so those waters are kept in float64 carries and released into the fields as far as the fields
resolve them (``PROVENANCE['carries']``); the ``air_booked`` and ``water_booked`` ledgers check that what was booked is
what the climate received plus what waits.

Viewer data (``static3``, ``frame3``): the contract in ``VIEW_SCHEMA``'s docstring of :meth:`World3.static3`.

Protocol (the owner's five rules of 4 Oct 2026; ``PROVENANCE['protocol']``). Every outcome is kept: no star, no
planet, no habitable planet, no bodies era (an exclusion by the chain's outcome, counted apart), extinction and
survival are all results (:meth:`World3.outcomes` lists one row per sampled seed, the summary keeps every world);
errors are recorded, not outcomes. Nothing rescues a run: no synthetic planet replaces a world7 seed, patches are
placed by area only (sea patches on a world without land), founders are placed uniformly, nothing is replenished
outside the modelled processes (the spin-up's held air is reported), and an extinct arena stays extinct. The rules
are fixed per experiment: :func:`rules_identity` hashes the source of ``haishool/life9/v3`` and
``haishool/life9/planet`` and every in-repo module they import, the config, the rule dataclasses, every module
constant in effect and the Earth reference; the command line (``__main__``) writes it into the experiment's manifest
before the first step, pins every seed's starting conditions, and refuses to resume under a different hash.
Carrying state between stages (the chemistry layer, the hand-offs) is a known gap: v3 starts at the chain's bodies
era.

Every constant carries a ``PROVENANCE`` entry. One world generator drives every draw of the run (senses' scan offsets,
events, division, mutation, the no-selection control), the shadow has its own, so the CPU run is exactly
reproducible and resumable (:meth:`World3.state_dict`, :meth:`World3.from_state`). Ledgers are float64.
"""
from __future__ import annotations

import ast
import copy as copy_mod
import ctypes
import dataclasses
import hashlib
import importlib
import json
import math
import os
import pkgutil
import sys
import time
import types
import warnings
from dataclasses import asdict, dataclass, fields, replace
from functools import partial
from pathlib import Path

import torch

from haishool.life9.planet import biosphere as bs
from haishool.life9.planet import chain as ch
from haishool.life9.planet import climate as cl
from haishool.life9.planet import constants as K
from haishool.life9.planet import crafting as cr
from haishool.life9.planet import globe as gb
from haishool.life9.planet import items as itm
from haishool.life9.planet import materials as mat

from . import body as B
from . import brain3 as b3
from . import formation3 as f3
from . import manipulate as mp
from . import patch as pt
from . import senses3 as s3

VIEW_SCHEMA = "life9-v3-view-1"
STATE_VERSION = "life9-v3-world3-2"
DAY_S = K.DAY_S
FOUR_PI = 4.0 * math.pi
M_C = K.element_mass("C")
M_N = K.element_mass("N")
M_O2 = K.molar_mass("O2")
M_CO2 = K.molar_mass("CO2")
_C, _N = mat.ELEMENTS.index("C"), mat.ELEMENTS.index("N")
_H, _O = mat.ELEMENTS.index("H"), mat.ELEMENTS.index("O")
_GAS = {g: i for i, g in enumerate(mat.AIR_GASES)}                # O2, CO2, H2O of the items' air ledger

# ============================================================================================ physics constants
KT_COST_J_PER_N = 0.2            # J per N of body weight per ground contact (Kram & Taylor 1990)
SUBSTEP_CELLS = 2.0              # a movement sub-step moves at most this many fine cells (default substep_m)
PADDLE_EFF = B.PADDLE_EFF        # propulsive (Froude) efficiency of drag-based paddling (body.PADDLE_EFF)
WAVE_DRAG_MAX = 4.0              # surface wave drag up to 4 x the deep drag near and above hull speed
FR_HULL = 1.0 / math.sqrt(2.0 * math.pi)                         # hull speed: the bow wave as long as the body
BASELINE_GENOMES = 64            # founders' networks kept as the behavioural baseline, per arena
REPLAY_BODIES = 64               # living bodies per arena whose inputs the baseline is replayed on
SENSE_DETAIL_BOUT = 0            # the bout of each day whose senses report their cost bounds
FLESH_SPECIES = ("meat", "fat", "hide")                          # rotting animal matter for the carcass smell
GENE_REPORT = ("size_kg", "fur_m", "thermo_gain", "setpoint_k", "enz_plant", "enz_meat", "eye", "ear", "voice",
               "muscle", "muscle_frac", "repair", "offspring_share", "skin_perm", "fat_store", "bladder", "kidney",
               "air_l_kg", "o2_carrier", "eta", "mut", "g", "k")
BEHAVIOUR_KEYS = ("mouth_plant", "mouth_water", "mouth_body", "mouth_item", "mouth_events", "mouth_sum",
                  "mouth_on_plant", "n_on_plant", "mouth_off_plant", "n_off_plant", "mouth_on_water", "n_on_water",
                  "mouth_off_water", "n_off_water", "plant_kg", "water_kg", "bitten_kg", "item_kg", "rubs",
                  "rubs_on_partner", "rub_heat_j", "embers", "fires_born", "grips", "joins", "placed", "released",
                  "strikes", "hits", "wound_j", "knaps", "broke", "transform_kg", "call_j", "loudness", "distance_m",
                  "coarse_moves", "moving", "substeps", "body_substeps", "travel_beyond_far", "skin_char_kg",
                  "fire_heat_j", "hand_heat_j", "sunk_s", "develop_j", "nonfinite", "no_item_slot", "no_fire_slot",
                  "carcass_no_slot", "grip_overflow", "anoxic_mol", "unconscious", "treading", "sense_bodies",
                  "sense_bound", "sense_far_bound",
                  "sense_beyond", "sense_item_beyond", "fires_culled", *(f"replay_{w}_{o}_{q}" for q in ("plant", "water")
                                                                          for o in ("on", "off")
                                                                          for w in ("n", "real", "base")))
#: the per-arena ledger entries whose change over the day the summary reports
DAY_KEYS = ("births", "divide_fired", "capacity_full", "too_small", "e_food", "e_oxidised",
            *(f"deaths_{c}" for c in B.CAUSES[1:]))
SOURCES = ("world7", "synthetic", "earth")                        # chain inputs an experiment may choose
#: outcomes of a sampled seed that gets no world: kind -> the outcome's words
ABSENT_KINDS = {"no_star": "no star", "no_planet": "no planet", "no_habitable_planet": "no habitable planet",
                "no_bodies_era": "no bodies era (excluded by the chain's outcome)"}
#: absent kinds that are not physical outcomes of the planet but a rule that conditions on an earlier stage's
#: biological outcome (an open owner decision, PROVENANCE['ABSENT_KINDS']): counted apart in the reports
EXCLUDED_KINDS = ("no_bodies_era",)
ERROR_KIND = "error"                                              # a seed that could not be resolved (not an outcome)
ABSENT_VERSION = "life9-v3-absent-1"                              # cache format of chain_outcome
PROTOCOL_VERSION = "life9-v3-protocol-1"                          # manifest and outcomes format
RULE_PACKAGES = ("haishool.life9.v3", "haishool.life9.planet")    # whose source and constants are the rules
#: modules outside RULE_PACKAGES whose constants join the values hash (their source joins through the import closure)
RULE_VALUE_MODULES = ("haishool.life9.brain", "haishool.truth.formula", "haishool.cosmos.planets")
SETUP_SALT = "life9-v3-setup"                                     # the per-seed set-up generators' salt
REPO_ROOT = Path(__file__).resolve().parents[3]

R_, D_, N_, C_ = "reference", "derived", "new_rule", "chain"
RN, DN = "reference+new_rule", "derived+new_rule"
PROVENANCE = {
    "KT_COST_J_PER_N": (RN, "the metabolic rate of running per unit body weight is c / t_c, t_c the time of one foot "
                            "contact, with c nearly the same across a 4,000-fold range of size and a fourfold range of "
                            "speed (Kram & Taylor 1990, Nature 346, 265; Roberts, Kram, Weyand & Taylor 1998, J. Exp. "
                            "Biol. 201, 2745: mammals of about 0.1-200 kg, so the 2-200 g founders lie below the "
                            "measured range, an extrapolation); c about 0.2 J/N, the value quoted from memory (the "
                            "form is confirmed by those sources, the number still to verify). A contract "
                            "change the owner has not yet decided (PLANET-V3-SPEC 11, 'Locomotion cost': body.move's "
                            "c_m M g law gives 7-9 J/(kg m) at every size): the gates test the physics (energy c M g "
                            "per ground contact, the contact length the body's), never a fit to a measured size curve; "
                            "cost_of_transport reports the comparison with Taylor, Heglund & Maloiy 1982 (J. Exp. "
                            "Biol. 97, 1), within about 12 % of their 10.7 M^-0.316 J/(kg m) from 2 g to 70 kg"),
    "contact_length": (N_, "the distance a body travels during one ground contact is its body length (the ellipsoid's "
                           "2a): contact length scales with leg length, and the body plan has no separate leg"),
    "locomotion": (N_, "move_kt. The locomotion share of the bout (manipulate.time_split: a share of the bout's time) is "
                       "spent moving at the muscles' full sustained power over body.MUSCLE_EFF; the rest of the bout "
                       "the body stands. On land the speed solves KT c M g v / L_c + 1/2 rho C_d A_x v^3 / eff = P (air "
                       "drag added: a treadmill has none; M includes the held items); a body denser than water "
                       "(body.density, against body.water_density at its position: sea water 1,027 kg/m^3, ponds "
                       "fresh) treads it while it moves when its moving power covers body.tread_power_w (it paddles "
                       "at the surface with the rest), else walks on the bottom of water deeper than its height: the "
                       "same law with its weight in water and water's drag; standing, it sinks there; one that floats "
                       "paddles at the surface: PADDLE_EFF x eff x P = (1 + wave drag(Fr)) 1/2 rho_w C_d A_x v^3. The "
                       "moving power is body.sustained_power_w (its O2 supply caps it). The path is marched in "
                       "samples: each segment is "
                       "crossed at the speed of its medium (submerged share from body.submerged_share), the climb M g "
                       "max(0, z - z0) is paid from the same power, and the body stops where its locomotion time is "
                       "spent; the time its airway spends under water (body.sunk), moving or standing, is sunk_s, "
                       "with its dive structure in order (body.DIVE_KEYS: the dive continued from the bout before, "
                       "the longest between breaths, the dive it ends in)"),
    "PADDLE_EFF": (RN, "body.PADDLE_EFF: drag-based paddling converts at most about a third of the limbs' work into "
                       "thrust (Fish 1984, 1996; from memory: verify); 0.25"),
    "WAVE_DRAG_MAX": (RN, "a body swimming at the surface makes a bow wave: its drag rises to about 5 times the drag "
                          "deep under water near hull speed and stays high above it (Hertel 1966, Structure, Form, "
                          "Movement; Vogel 1994, Life in Moving Fluids, ch. 13); drag factor 1 + 4 Fr^4 / (Fr^4 + "
                          "FR_HULL^4), Fr = v / sqrt(g L) with L the body's length (the smooth rise is a new_rule "
                          "shape)"),
    "FR_HULL": (D_, "hull speed: a gravity wave as long as the body travels at sqrt(g L / 2 pi), Fr = 1 / sqrt(2 pi) "
                    "= 0.40"),
    "BASELINE_GENOMES": (N_, "numerics: founders' networks kept per arena as the behavioural baseline "
                             "(behaviour_baseline)"),
    "REPLAY_BODIES": (N_, "numerics: living bodies per arena (the lowest living slots) whose inputs the baseline and "
                          "their own networks are replayed on"),
    "SENSE_DETAIL_BOUT": (N_, "numerics: the bout of each day whose senses report their cost bounds (sense_details)"),
    "FLESH_SPECIES": (N_, "the rotting animal matter whose items feed the carcass smell"),
    "bouts": (N_, "decision interval: 24 bouts a day (1 h), so that the sun angle is resolved hourly (dawn, dusk and "
                  "night are distinct, so nocturnality can emerge) and a body looks again before most of its travel "
                  "passes the senses' guaranteed far range (128 m); the share of moving bodies whose bout travel "
                  "passes it is reported (travel_beyond_far; measured on the default Earth config, days 0-2: 0-7 % "
                  "of the moving bodies per arena, coarse sub-steps 0-2.6 %; the fastest founders reach 2 m/s at "
                  "full power on land, so a whole hourly bout of travel can pass several kilometres). PLANET-V3-SPEC "
                  "1 keeps bouts a config value; every rate of the bodies (development, digestion, drinking, the "
                  "manipulation events, locomotion) is per second, so the bout length changes numerics, not physics"),
    "substeps": (N_, "numerics: each body's own K = ceil(its longest possible bout travel at max(v_land, v_water) / "
                     "substep_m) movement sub-steps, at most max_substeps; sub-step s moves only the bodies with K > s, "
                     "each for its own bout / K, and their mouths act in it for that time (body.ingest with dt_body "
                     "and disc_share 1 / K, so the still mouth's disc is cropped once a bout); held items follow; the "
                     "brain's command is held for the bout. Bodies that would need more are not slowed: they take "
                     "longer sub-steps, and the realised sub-steps longer than substep_m are counted (coarse_moves)"),
    "SUBSTEP_CELLS": (N_, "a sub-step moves at most two fine cells (32 m on the default grid): within the fine 3 x 3 "
                          "contact hash and 1/64 of the arena"),
    "capacity": (N_, "body slots per arena: 4,096 (the owner's decision of 4 Oct 2026: keep the 2 km patches, 4,096 "
                     "slots per arena; was 1,024), below the spec's 8,192 (PLANET-V3-SPEC 6). No slot count that fits "
                     "in memory exceeds a 4.2 km^2 patch's carrying capacity for 2-200 g bodies (of order 1e8 bodies), "
                     "so the cap binds once a lineage multiplies: a binding cap is a numerical limit, not ecology. "
                     "Births into a full arena fail and lose their development (body.divide), capacity_full is "
                     "reported as a share of the births attempted, and the summary and the outcomes mark the run "
                     "capacity_bound from the first day it binds. Memory: about 223 kB of state per slot at hidden "
                     "128 with float32 weights (the founders' Wx, Wh, the live Wh and Wo; weight_dtype bfloat16 "
                     "halves them), so 16 x 4,096 slots hold about 14.6 GB per world; World3.memory_bytes and "
                     "reports['state_mb'] report it. Measured on the model Earth (bench, CPU, 24 threads, 4 Oct "
                     "2026): 14.7 GB of state, peak resident 16.4 GB. The review of 4 Oct 2026 ran it longer: 4,096 "
                     "slots BIND TOO, on the model Earth on day index 4 (arena 5 grew 482, 1,027, 1,935, 3,115, "
                     "4,096 over days 1-5; capacity_full 538), about 2 days later than 1,024 slots (bound on day "
                     "index 2), at 4x the memory and about 1.6x the CPU time per day at equal population (35 s "
                     "against 22 s on day 1: the empty slots cost); seconds per day at 4,096 rose 35, 38, 47, 60, 78 "
                     "over days 1-5 as the slots filled. Decision (1) assumed more headroom than this. GPU hosts: "
                     "the float32 body state of the default world is about 14.6 GB, so the RTX 3060 (12 GB) cannot "
                     "hold it, the RTX 4090 (24 GB) holds it with little room, the R9700 (32 GB) with some; "
                     "weight_dtype bfloat16 halves it (7.3 GB measured). Whether GPU experiments use bfloat16 or "
                     "fewer slots is AN OWNER'S DECISION STILL OPEN (it changes the rules hash). The founders are "
                     "drawn per world on the CPU and copied into the device's state, so set-up holds the state "
                     "once (body.found_into)"),
    "patches": (N_, "16 habitat patches per world by default (PLANET-V3-SPEC 1 says 4): a scale parameter. With 4 the "
                    "random draw by area alone often put every patch outside the bodies' temperature window "
                    "(271-318 K) and the world died at founding; 16 patches sample the land better, with no "
                    "habitability bias (16 x 4,096 slots: PROVENANCE['capacity'])"),
    "climate_target": (N_, "the climate's cloud albedo is calibrated toward formation's T_s only for planets that are "
                           "not frozen_mean and not runaway; for those formation's T_s (no ice-albedo feedback, liquid "
                           "vapour) is no target and the climate keeps the spec's planetary albedo (PLANET-V3-SPEC 2, "
                           "formation3's docstring)"),
    "ocean0": (D_, "the climate's ocean = terrain ocean volume x 1000 kg/m^3 over 4 pi R^2 with each planet's own "
                   "radius (climate.make_params takes one radius for all worlds)"),
    "global_biosphere": (N_, "version 2's global biosphere on the globe is the coarse global layer, run without its "
                             "seed rain (the floor that refilled soft tissue toward 1 g C/m^2 is 0): its plants are "
                             "sown once at set-up (biosphere.init_state) and live on their own production; the "
                             "patches' plants have no minimum stock anywhere and spread only by seed"),
    "patch_cells": (N_, "patch.choose_patch_cells for every world: global land cells at random by area (no "
                        "habitability bias); a world without land draws its patches over all its cells, so they are "
                        "sea (patch.py's own rule): founders float or sink there and physics decides (rule 4: no "
                        "placement by habitability; before, World3 gave a landless world no patch and recorded 'no "
                        "land', a habitability prior for land bodies, review of 4 Oct 2026). Such a world's row "
                        "carries no_land True"),
    "deposits": (C_, "version 2's globe.make_deposits of the chain's crust: the mineral ground, of which a stroke can "
                     "break pieces off its top manipulate.SURFACE_LAYER_M; kept per fine cell (float64, kg/m^2, "
                     "initialised from the global cell's column), and what is broken off a cell is booked out of "
                     "that cell"),
    "founder_temperature": (N_, "founders start at the fine air temperature where they stand (an ectotherm founder's "
                                "body is at ambient)"),
    "founding_day": (N_, "the founding day is where the climate and vegetation spin-ups end (veg_spin_days after the "
                         "climate's own spin-up): a starting condition, reported with its day of the year and the "
                         "declination (reports['vegetation_spin_up']['founding']); the share of each arena's "
                         "founders dead in the first bout is reported (reports['founding'])"),
    "barometric": (D_, "the global layer's partial pressures are at sea level; a patch's air is reduced by exp(-g M h / "
                       "(R T_mean)) at each fine cell's height above the sea (T_mean the mean of the air column with "
                       "the patch's lapse rate): the bodies' O2 and air pressure per fine cell, and the senses' air at "
                       "the arena's base height. Mole fractions (fire's x_O2) do not change with height"),
    "fire_heat": (DN, "the fires' heat on a body, averaged over the positions of its movement sub-steps: flux q = "
                      "min(crafting.RADIANT_FRACTION x HRR / (4 pi d^2), eps sigma T_f^4) (the point source of version "
                      "2 near the fire is capped by the flame's own emissive power, so it rises continuously into "
                      "the bed instead of jumping at its rim), absorbed over the body's cross-section pi r1^2 with "
                      "its emissivity, into body.metabolism as Env.heat_w"),
    "fire_possible": (C_, "fire needs crafting.X_O2_MIN of O2 by mole fraction (a 1-bar flammability limit, version 2): "
                          "the summary and the set-up reports say whether a world's air allows any fire (None, "
                          "uncertain, with fire_beyond_fit when the total pressure lies outside formation3.FIRE_FIT_PA, "
                          "0.5-2 bar: the 1-bar limit is applied by the fires at any pressure, and world7 66 passes it "
                          "at 0.24 bar with dry x_O2 0.157 and pO2 3.7 kPa, outside its calibration). With "
                          "formation3's separate degassed share for nitrogen (the owner's decision of 4 Oct 2026, an "
                          "Earth calibration) the model Earth's air is about 1 bar (1.026 bar measured on 4 Oct 2026) "
                          "with x_O2 about 0.21, so fire is physically possible on it; before, its 2.05 bar held "
                          "10 % O2 and every fire went out"),
    "hand_heat": (D_, "heat a body's own manipulation leaves in it, into Env.heat_w: the heat the hot rubbing contact "
                      "conducts into a bare limb and the energy that chars its skin (manipulate.rub skin_heat_j, "
                      "skin_char_j), a free limb's swing (rub limb_j) and the swing of strokes that hit nothing "
                      "(force (strokes - hits) x swing_j): body.metabolism counts work_j as leaving the body"),
    "burns": (N_, "a bare limb's skin charred by rubbing (manipulate.rub skin_char_kg) is a wound of that share of the "
                  "lean mass over body.WOUND_LETHAL_SHARE (the charred mass is not removed from the body's books: "
                  "body.py has no tissue-removal path besides bites; it is counted as skin_char_kg)"),
    "wounds": (N_, "manipulate.force damage of the struck bodies adds to body damage"),
    "drowning_gate": (DN, "within a bout a body whose O2 store is spent is unconscious: from that sub-step on it "
                          "neither moves nor eats (it stands where it is, sunk if it is in deep water and denser "
                          "than it). The gate follows its O2 debt: the debt carried from the last bout while no "
                          "breath has come, plus the time of its current dive x its O2 use, against body.o2_store_mol. "
                          "The O2 use is the larger of the previous bout's metabolic O2 rate (body.metabolism's "
                          "metabolic_w, with thermogenesis, development and every cost) and the resting use plus the "
                          "bout-mean use of its locomotion, so the gate and metabolism agree (review of 4 Oct 2026: "
                          "the gate's resting rate left thermogenesis out). body.metabolism books the anoxia beyond "
                          "the store itself (body.PROVENANCE['breath_hold'])"),
    "carry_load": (N_,"held items add their mass to the weight the legs support and lift (KT cost, climb, bottom "
                       "walking; Taylor, Heglund, McMahon & Looney 1980, J. Exp. Biol. 86, 9: the cost of carrying a "
                       "load is in proportion to its weight) and follow the mouth after each sub-step"),
    "development": (N_, "the divide output is the share of the body's protein synthesis spent on its next child "
                        "(body.metabolism develop=, body.divide): a rate per second, so births per day do not depend "
                        "on the bouts"),
    "brain_units": (N_, "a body computes with body.active_units: k, but no more units than its brain tissue holds when "
                        "the tissue closure scaled its organs down (computation needs tissue)"),
    "item_residue": (N_, "carbon and nitrogen of items that leave the pool to the ground (rot, fire residue, charred "
                         "tissue) go to the fine cell's litter and mineral nitrogen through float64 residuals settled "
                         "as far as the float32 fields resolve them; their makeup water goes to the air (the cell's "
                         "vapour); of the rest, by element, the H and O of the organic part are counted as "
                         "organic_ho_kg and the other elements as mineral_sink_el_kg (version 2's world)"),
    "carries": (N_, "numerics: water given to the global air and taken from the global ocean is kept per world and "
                    "cell in float64 carries and released into the climate's float32 vapour and float64 ocean as far "
                    "as they resolve it, the climate's water_external booking each release"),
    "exchange_scale": (N_, "the scale of each body <-> patch book's error (ledgers' exchange_*) is the larger of "
                           "body.exchange_errors' own scale and what passed through the book (the dead bodies', "
                           "excreta's and food's amounts since the snapshot): the body side of carbon_out is the dead "
                           "carbon less the carcass items' carbon, a difference of near-equal numbers whose float32 "
                           "rounding (micrograms in a planet-built arena whose 32 founders froze in the first bout) "
                           "is relative to the carcasses, not to the difference"),
    "gas_booking": (D_, "per world and day: CO2 down and O2 up by the patches' net fixation (patch c_air, the sown "
                        "carbon included), the bodies' O2 and CO2 (body.gas_mol), the fires' O2 and CO2 "
                        "(manipulate's air ledger), N2 down by the patches' nitrogen fixation; each over 4 pi R^2 of "
                        "the planet, through climate.add_gas"),
    "patch_water_sample": (N_, "the patches' precipitation, evaporation and outflow are a sample of their global "
                               "cell's (inside the climate's own water ledger) and are not booked again; the water "
                               "the bodies, items and fires give to the air is booked into the cell's vapour, so the "
                               "global layer gains the water of the patch-grown matter and of the patch water the "
                               "bodies drank (net external water, reported as water_to_air_kg)"),
    "smells": (N_, "patch.volatile_step per bout: plant volatiles from the soft tissue, carcass volatiles from the "
                   "flesh, fat and hide items on the ground, smoke from the fires' burning, CO2 from the bodies' "
                   "metabolism and the fires"),
    "post_bout_state": (N_, "the plasticity's readout m = tanh(sum g_i x intero_i) reads the body's own state after the "
                            "bout (senses3's interoceptive code, its light level of the bout), so what a gene g "
                            "weighs is the consequence of the bout"),
    "lineages": (N_, "lineages alive: distinct founder ids among the living (bookkeeping only; never sensed); the "
                     "neutral shadow's count is the drift baseline"),
    "gene_coord": (N_, "gene statistics in each gene's mutation coordinate (log10 for log genes, logit for logit, "
                       "linear for lin): mean and sd over the living, the same for the neutral shadow"),
    "behaviour_stats": (N_, "mouth statistics weighted by the mouth intensity (its output is a sigmoid, never exactly "
                            "0, so a boolean 'touched plants' would count every body on a vegetated cell): the summed "
                            "intensity on each kind of contact, the amounts taken, and the mean intensity where the "
                            "mouth is on plants (water) against where it is not, a measure of behaviour against "
                            "chance"),
    "behaviour_baseline": (N_, "BASELINE_GENOMES founders' networks per arena are kept unchanged (no selection, no "
                               "drift); on the senses' sampled bout of each day the networks of REPLAY_BODIES living "
                               "bodies and the baseline networks are run one step from a zero hidden state on the "
                               "same bodies' inputs, and the mouth intensity on plants (water) against off them is "
                               "reported for both"),
    "sense_details": (N_, "on bout SENSE_DETAIL_BOUT of each day senses3.observe returns its cost bounds (bound, "
                          "far_bound, beyond, item_beyond, fires_culled), summed per arena (PLANET-V3-SPEC 0: every "
                          "cost-bounding cap reports when it binds); the item and fire pools' overflows "
                          "(no_item_slot, no_fire_slot, carcass parts without a slot) and the grips' second scans "
                          "are counted every bout"),
    "capacity_bound": (N_, "the run is marked capacity_bound (a numerical limit binding, reported, not ecology) from "
                           "the first day a birth found no free slot; each world's outcome carries it, with the first "
                           "day per arena (reports['arena_capacity_bound_day'])"),
    "protocol": (N_, "the owner's five rules of 4 Oct 2026 for an experiment: (1) keep every outcome (no star, barren "
                     "planets, extinction are results); (2) follow every sampled seed under a sampling rule fixed "
                     "before its biological outcome is known (the command line's --sample); (3) carry state between "
                     "stages: deferred, a known gap (v3 starts at the chain's bodies era; the chemistry layer and the "
                     "hand-offs come later); (4) no automatic rescue or replacement: no synthetic planet in place of a "
                     "world7 seed, no habitability-based placement (patches by area, founders uniform), nothing "
                     "replenished outside the modelled processes, an extinct arena is never re-founded; (5) the rules "
                     "are fixed before each experiment (rules_identity in the manifest; a rule change is a new "
                     "experiment)"),
    "SOURCES": (N_, "the chain inputs an experiment chooses: world7 (the cached chain), synthetic (chain.synthetic_inputs) "
                    "or earth (chain.earth_inputs, the calibration case); the choice is the experiment's, never "
                    "substituted"),
    "ABSENT_KINDS": (C_, "the chain's own reasons a world7 seed gives v3 no planet to found on (chain_outcome): the "
                         "star era did not fuse (no star); no planets era, or none formed (no planet); no temperate "
                         "rocky planet, or none with liquid water (chain.NoHabitablePlanet: no habitable planet); the "
                         "chain's planet did not reach the bodies era (no bodies era: the origin of v3's replicators "
                         "is the chain's earlier rungs, PLANET-V3-SPEC 0, so v3 founds no bodies where the chain made "
                         "none). 'no bodies era' is NOT a physical outcome of the planet: it conditions the body stage "
                         "on an earlier stage's biological outcome while rule 3 is deferred (v3's founders carry "
                         "nothing from the chain), so it is labelled 'excluded by the chain's outcome' "
                         "(EXCLUDED_KINDS) and counted apart; whether to build these planets and let physics decide "
                         "is AN OWNER'S DECISION STILL OPEN (review of 4 Oct 2026)"),
    "EXCLUDED_KINDS": (N_, "the absent kinds that are exclusions by an earlier stage's outcome, not outcomes "
                           "(ABSENT_KINDS); reports['excluded'] and the outcomes count them apart"),
    "ERROR_KIND": (N_, "a seed whose chain inputs could not be resolved (a missing chain cache on a host without "
                       "world7, a crash) is recorded as an error with its cause, never as an outcome; the experiment "
                       "is then incomplete (PROTOCOL.md rule 1)"),
    "ABSENT_VERSION": (N_, "numerics: format of chain_outcome's cache (absent-world7-SEED.json next to the chain cache, "
                           "so hosts without world7's scipy read the outcome as they read the chain inputs)"),
    "PROTOCOL_VERSION": (N_, "format of the manifest and the outcomes files"),
    "RULE_PACKAGES": (N_, "the packages whose source and module constants make the rules hash (haishool/life9/v3 and "
                          "haishool/life9/planet), with every in-repo module they import, found statically "
                          "(rule_source_files: haishool/truth/formula.py, haishool/cosmos/planets.py, "
                          "haishool/life9/brain.py, the chain's world7 path ...), so the hash is the same on every "
                          "host; the chain cache is the inputs, pinned per seed in DIR/inputs.json before the first "
                          "step"),
    "RULE_VALUE_MODULES": (N_, "modules outside RULE_PACKAGES whose constants (public and private upper-case) join the "
                               "values hash: life9.brain (unit_mask), truth.formula (the formula parser of the molar "
                               "masses and the fat), cosmos.planets (formation3's level-3 constants and the Earth "
                               "reference)"),
    "SETUP_SALT": (N_, "each world's set-up generator is seeded by sha256(SETUP_SALT:salt:seed), salt the config's "
                       "rng_seed (default 'default'): a world is a function of its seed and the rules alone, not of "
                       "the other seeds of its sample or part (review of 4 Oct 2026: one generator for all worlds "
                       "made a seed's terrain, patches and founders depend on the list); the run's generators (day, "
                       "shadow, baseline) are drawn once for the worlds stepped together, so the dynamics of a part "
                       "are shared draws (statistically, not bitwise, independent of the partition)"),
    "REPO_ROOT": (N_, "the repository root: source paths in the rules identity are relative to it"),
    "absent": (N_, "a sampled seed that gets no world is recorded with its outcome (ABSENT_KINDS, the chain's note) "
                   "in reports['absent'], the summary's worlds and the outcomes; it is neither skipped nor replaced. "
                   "A world7 seed reaches it only through the chain's own eras, read from the chain cache first (a "
                   "host without world7's scipy reads absent-world7-SEED.json before trying to run the chain); "
                   "synthetic and earth always give a planet"),
    "outcomes": (N_, "one row per sampled seed in the sampled order (World3.outcomes): the absent kinds (no bodies era "
                     "flagged excluded); an error kind for a seed that could not be resolved; 'no patches' (patches "
                     "= 0); 'extinct at day d' when every arena of the world is empty, d the last arena's day of "
                     "extinction; else 'alive at day d' after d days; with the population, the arenas alive, every "
                     "arena's extinction day, the founding report (founders dead in the first bout), the first day "
                     "the body slots, the item pool and the fire pool bound, the non-finite counts, the spin-up's "
                     "replenishment, the planet pick, check_spec3's failures and the seed's inputs digest. Every day "
                     "in the outputs is a count of days completed (day_base 'days_completed')"),
    "extinction": (N_, "an arena is extinct at day d, d the days completed at the end of the first day at whose end no "
                       "body is alive in it (reports['extinct_day']: founders all dead in the first day give 1); it "
                       "is never re-founded"),
    "day_base": (N_, "every day in the summary, the outcomes and the reports counts the days completed when it "
                     "happened (the row's 'day' is the days run): 'extinct at day 1' is an arena empty at the end of "
                     "its first day (review of 4 Oct 2026: the outputs mixed a 0-based day index with days run)"),
    "pools_bound": (N_, "the first day (days completed) each arena's item pool and fire pool overflowed (a carcass "
                        "part, a broken piece or a fire found no slot: reports['arena_items_bound_day'], "
                        "['arena_fires_bound_day']); a numerical limit reported beside capacity_bound: after it, "
                        "carcass parts go to the litter for want of a slot (V3Config.items, fires)"),
    "spin_up_replenishment": (N_, "the global biosphere spins up with the chain's air held (biosphere.spin_up "
                                  "hold_gas) and jumps its wood and litter to their steady state: carbon is not "
                                  "conserved there. The carbon it made or removed (organic plus air carbon, after "
                                  "minus before, kg per world) and the air's moles before and after are reported "
                                  "(reports['spin_up_replenishment'], the outcome rows): a set-up replenishment of "
                                  "rule 4, stated in PROTOCOL.md"),
    "rules_hash": (N_, "rules_identity: sha256 over the source of every .py file of RULE_PACKAGES and of every in-repo "
                       "module they import (rule_source_files; line endings normalised to LF, so a Windows and a Linux "
                       "checkout agree), the config, the rule dataclasses as World3 uses them (Formation3Rules, "
                       "FormationRules, ClimateRules, BioRules without its seed floor, PatchRules at the config's "
                       "patch size, SenseRules), every upper-case module constant (private ones too, caches excepted) "
                       "of RULE_PACKAGES and RULE_VALUE_MODULES in effect (a constant changed at run time changes the "
                       "hash too), and the digest of the Earth reference that calibrates formation3 "
                       "(chain.earth_inputs, computed from the code); a path constant counts by the content of the "
                       "file it names (the atomic weights), a folder not at all, so the hash does not depend on where "
                       "the checkout lives"),
    "viability": (N_, "the summary judges the loop by births and deaths per day, turnover, the food energy taken "
                      "against the energy spent, the fat runway (days the living could pay their maintenance from "
                      "their fat alone) and the arenas without plants, next to the lineage counts"),
    "view": (N_, "viewer rounding: positions 0.01 m, heading 1e-3 rad, mass 1e-4 g, intensities 1e-3; items capped at "
                 "1,500 per patch (ground items, heaviest first); plants 255 x fAPAR, water 255 x pond / (pond + 100 "
                 "kg/m^2)"),
}


# ============================================================================================ config
@dataclass
class V3Config:
    """Settings of a v3 world (SI units). Physics lives in the modules; these are sizes, schedules and numerics."""
    G: int = 32                          # cube-sphere cells per face edge (C = 6 G^2)
    patches: int = 16                    # habitat patches per world (PROVENANCE['patches'])
    patch_m: float = 2048.0              # patch side (m)
    cells: int = 128                     # fine cells per patch side
    capacity: int = 4096                 # body slots per arena (PROVENANCE['capacity'])
    founders: int = 512                  # founders per arena
    items: int = 1024                    # item slots per arena
    fires: int = 64                      # fire slots per arena
    hidden: int = 128                    # brain units at most
    k_founder: tuple = (2, 32)           # founders' active units
    bouts: int = 24                      # bouts per day (PROVENANCE['bouts'])
    substep_m: float | None = None       # movement sub-step length (None: SUBSTEP_CELLS fine cells)
    max_substeps: int = 8                # movement sub-steps per bout at most
    fire_substeps: int = 4               # crafting.fire_step sub-steps per bout
    weight_dtype: str = "float32"        # brain weights and live weights: float32 or bfloat16
    source: str = "world7"               # chain inputs: world7 (cached chain), synthetic, earth (PROVENANCE['SOURCES'])
    rng_seed: int | None = None          # set-up generator seed (None: derived from the sampled seeds)
    selection: bool = True               # False: body.deaths' no-selection control
    shadow: bool = True                  # run the neutral shadow (drift baseline)
    climate_fast_chunks: int = 8
    climate_slow_chunks: int = 4
    climate_tol_k: float = 0.1
    bio_spin_days: int = 730
    bio_spin_segments: int = 2
    veg_spin_days: int = 1800            # patch plants grow from sparse seed (patch's open issue: 1,500-2,000 days)
    veg_spin_bouts: int = 4              # light resolution of the spin-up days (bouts of sun samples)

    def __post_init__(self):
        self.k_founder = tuple(int(v) for v in self.k_founder)

    @property
    def substep_len_m(self) -> float:
        return float(self.substep_m) if self.substep_m is not None else SUBSTEP_CELLS * self.patch_m / self.cells

    def to_dict(self) -> dict:
        d = asdict(self)
        d["k_founder"] = list(self.k_founder)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "V3Config":
        if d.get("synthetic_fallback"):
            raise ValueError("synthetic_fallback is gone (PROVENANCE['protocol']: no synthetic planet replaces a "
                             "world7 seed); this config belongs to an experiment under other rules")
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    def check(self) -> None:
        if not 1 <= self.founders <= self.capacity:
            raise ValueError("founders must lie within 1 and capacity")
        if self.source not in SOURCES:
            raise ValueError("source is world7, synthetic or earth")
        if self.weight_dtype not in ("float32", "bfloat16"):
            raise ValueError("weight_dtype is float32 or bfloat16")
        if self.bouts < 1 or self.max_substeps < 1 or self.fire_substeps < 1:
            raise ValueError("bouts, max_substeps and fire_substeps must be at least 1")
        if not 2 <= self.k_founder[0] <= self.k_founder[1] <= self.hidden:
            raise ValueError("k_founder must lie within [2, hidden]")
        if self.cells < 3 or self.patches < 0:
            raise ValueError("cells must be at least 3 and patches at least 0")


# ============================================================================================ inputs and seeds
def planet_inputs(seed: int, config: V3Config) -> tuple[dict | None, object]:
    """Chain inputs of one seed from the source the config names, and a note on them: ``(inputs, note)``, or
    ``(None, outcome)`` for a world7 seed whose chain gives no planet to found on (:func:`chain_outcome`'s
    {"kind", "outcome", "note"}; ``PROVENANCE['absent']``). The source is never substituted."""
    if config.source == "synthetic":
        return ch.synthetic_inputs(seed), "synthetic"
    if config.source == "earth":
        return ch.earth_inputs(), "earth reference (not chain-derived; the seed varies only the world's draws)"
    if config.source != "world7":
        raise ValueError(f"source is world7, synthetic or earth, not {config.source!r}")
    # the caches first (planet, then absent): a host without world7's scipy runs nothing it has cached
    x = ch.cached_chain_inputs(seed)
    if x is None:
        known = cached_chain_outcome(seed)
        if known is not None:
            return None, known
        try:
            x = ch.chain_inputs(seed)
        except ch.NoHabitablePlanet:
            return None, chain_outcome(seed)
    if "bodies" not in (x.get("links") or ()):
        return None, {"kind": "no_bodies_era", "outcome": ABSENT_KINDS["no_bodies_era"], "excluded": True,
                      "note": f"world7 seed {seed}: the chain's planet {x.get('planet_index')} reached "
                              f"{(list(x.get('links') or []) or ['surface'])[-1]!r} (links "
                              f"{list(x.get('links') or [])}), not the bodies era; v3 founds no bodies where the chain "
                              f"made none (an exclusion by the chain's outcome, PROVENANCE['ABSENT_KINDS'])",
                      "pick": pick_record(x)}
    return x, "world7"


def pick_record(inputs: dict | None) -> dict | None:
    """The planet pick of a seed's chain inputs (rule 2): the planet index, the rule, the bridge's pick of chain
    version 2 and whether they agree, and the number of temperate ocean planets."""
    if not inputs or inputs.get("source") != "world7":
        return None
    return {k: inputs.get(k) for k in ("planet_index", "pick_rule", "planet_index_bridge", "pick_agrees_with_bridge",
                                       "temperate_ocean_planets")}


def cached_chain_outcome(seed: int, cache_dir=None) -> dict | None:
    """:func:`chain_outcome` from its cache alone, or None (never runs world7)."""
    path = _absent_path(seed, cache_dir)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if data.get("version") == ABSENT_VERSION and data.get("chain_version") == ch.CHAIN_VERSION \
            and data.get("seed") == int(seed):
        return {k: data[k] for k in ("seed", "kind", "outcome", "note", "eras")}
    return None


def _absent_path(seed: int, cache_dir=None) -> Path:
    return Path(cache_dir if cache_dir is not None else ch.CACHE_DIR) / f"absent-world7-{int(seed)}.json"


def chain_outcome(seed: int, cache_dir=None, refresh: bool = False) -> dict:
    """Why world7 seed ``seed`` has no habitable planet, from the chain's own eras (``PROVENANCE['ABSENT_KINDS']``):
    {"seed", "kind", "outcome", "note", "eras"}. Read from ``absent-world7-SEED.json`` next to the chain cache, or
    worked out with world7 (``chain._scan``, which needs world7's levels) and cached there."""
    path = _absent_path(seed, cache_dir)
    if not refresh:
        known = cached_chain_outcome(seed, cache_dir)
        if known is not None:
            return known
    steps, _, _ = ch._scan(int(seed))
    eras = {s["era"]: s for s in steps}
    star = eras.get("star") or {}
    planets = eras.get("planets")
    surfaces = [s for s in steps if s["era"].rpartition("_")[0] == "surface"]
    if star.get("fusion") != "yes":
        kind, note = "no_star", (f"world7 seed {seed}: the star era's fusion is {star.get('fusion')!r} (star mass "
                                 f"{star.get('star_mass')} M_sun)")
    elif planets is None or not planets.get("planets"):
        kind, note = "no_planet", (f"world7 seed {seed}: the star fuses but no planet formed (disc solids "
                                   f"{star.get('disc_solids')} earth masses)")
    elif not surfaces:
        kind, note = "no_habitable_planet", (f"world7 seed {seed}: {planets.get('planets')} planets, "
                                             f"{planets.get('rocky')} rocky, none temperate")
    else:
        kind, note = "no_habitable_planet", (f"world7 seed {seed}: {len(surfaces)} temperate rocky planet(s), none "
                                             f"with liquid water (chain.NoHabitablePlanet)")
    out = {"seed": int(seed), "kind": kind, "outcome": ABSENT_KINDS[kind], "note": note,
           "eras": [s["era"] for s in steps]}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps({"version": ABSENT_VERSION, "chain_version": ch.CHAIN_VERSION, **out}, indent=1),
                   encoding="utf-8")
    os.replace(tmp, path)
    return out


def inputs_digest(inputs: dict | None) -> str | None:
    """sha256 of one world's chain inputs (canonical JSON): the experiment's starting conditions per seed."""
    if inputs is None:
        return None
    return hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def resolve_inputs(seeds, config: V3Config) -> list:
    """``(inputs, note)`` of every seed (:func:`planet_inputs`), an error recorded in the seed's place instead of
    raised: {"kind": ERROR_KIND, "outcome", "note", "cause", "traceback"} (``PROVENANCE['ERROR_KIND']``)."""
    import traceback
    out = []
    for s in seeds:
        try:
            out.append(planet_inputs(int(s), config))
        except Exception as e:                                                    # noqa: BLE001
            out.append((None, {"kind": ERROR_KIND, "outcome": "error", "cause": f"{type(e).__name__}: {e}",
                               "note": f"seed {int(s)}: its chain inputs could not be resolved "
                                       f"({type(e).__name__}: {e}): an error, not an outcome",
                               "traceback": traceback.format_exc()}))
    return out


def resolved_digest(seed: int, resolved: tuple) -> dict:
    """What a seed starts from, pinned before the first step (DIR/inputs.json): its inputs digest, or the digest of
    its absent outcome (kind, outcome, note, eras) or error."""
    x, note = resolved
    if x is not None:
        return {"seed": int(seed), "kind": "world", "inputs_sha256": inputs_digest(x), "pick": pick_record(x)}
    body = {k: note.get(k) for k in ("kind", "outcome", "note", "eras", "cause")}
    return {"seed": int(seed), "kind": note.get("kind"), "outcome": note.get("outcome"),
            "absent_sha256": hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()}


# ============================================================================================ the rules in effect
def _canon(v, depth: int = 0):
    """A JSON-ready, address-free form of a rule value (``PROVENANCE['rules_hash']``): tensors and arrays by the
    sha256 of their bytes, dataclasses by their fields, floats exactly (repr); callables, classes and modules are
    left out (their source is hashed)."""
    if depth > 12:
        return "<deep>"
    if v is None or isinstance(v, (bool, int, str)):
        return v
    if isinstance(v, float):
        return repr(v)
    if torch.is_tensor(v):
        t = v.detach().cpu().contiguous()
        h = hashlib.sha256(f"{t.dtype}|{tuple(t.shape)}|".encode())
        if t.numel():
            h.update(ctypes.string_at(t.data_ptr(), t.numel() * t.element_size()))
        return {"tensor": h.hexdigest()}
    if type(v).__module__ == "numpy" or type(v).__module__.startswith("numpy."):
        if hasattr(v, "tobytes") and hasattr(v, "shape") and getattr(v, "ndim", 0) > 0:
            return {"array": hashlib.sha256(f"{v.dtype}|{v.shape}|".encode() + v.tobytes()).hexdigest()}
        return repr(v.item()) if hasattr(v, "item") else repr(v)
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        return {"__dataclass__": type(v).__name__,
                **{f.name: _canon(getattr(v, f.name), depth + 1) for f in dataclasses.fields(v)}}
    if isinstance(v, dict):
        return {str(k): _canon(x, depth + 1) for k, x in sorted(v.items(), key=lambda kv: str(kv[0]))}
    if isinstance(v, (list, tuple)):
        return [_canon(x, depth + 1) for x in v]
    if isinstance(v, (set, frozenset)):
        return sorted((_canon(x, depth + 1) for x in v), key=repr)
    if isinstance(v, os.PathLike):                    # a data file by its content; a folder is no rule
        p = Path(v)
        return {"file": hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()} if p.is_file() else None
    if isinstance(v, (types.ModuleType, type)) or callable(v):
        return None
    text = repr(v)
    return type(v).__name__ if " at 0x" in text else text


def rule_modules() -> list:
    """Every module of RULE_PACKAGES (their command lines excepted), imported, in name order."""
    mods = []
    for pkg_name in RULE_PACKAGES:
        pkg = importlib.import_module(pkg_name)
        for info in sorted(pkgutil.iter_modules(pkg.__path__), key=lambda i: i.name):
            if info.name == "__main__":
                continue
            mods.append(importlib.import_module(f"{pkg_name}.{info.name}"))
    return mods


def _module_path(root: Path, name: str) -> Path | None:
    """The source file of dotted module ``name`` under ``root`` (a module or a package's __init__), or None."""
    base = root.joinpath(*name.split("."))
    for cand in (base.with_suffix(".py"), base / "__init__.py"):
        if cand.is_file():
            return cand
    return None


def _module_name(root: Path, path: Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    parts = list(rel.parts)
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _imports(root: Path, path: Path) -> set:
    """The in-repo (haishool.*) modules a source file imports anywhere in it (relative imports resolved; ``from X
    import Y`` counts Y when it is a module, else X), with their parent packages."""
    name = _module_name(root, path)
    pkg = name if path.name == "__init__.py" else name.rpartition(".")[0]
    tree = ast.parse(path.read_bytes())
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = pkg.split(".")
                base = ".".join(parts[:len(parts) - node.level + 1])
                mod = f"{base}.{node.module}" if node.module else base
            else:
                mod = node.module or ""
            found.add(mod)
            found.update(f"{mod}.{a.name}" for a in node.names)
    out = set()
    for m in found:
        if not m.startswith("haishool"):
            continue
        parts = m.split(".")
        for i in range(1, len(parts) + 1):
            out.add(".".join(parts[:i]))
    return {m for m in out if _module_path(root, m) is not None}


def rule_source_files(root=None) -> list:
    """Every source file of the rules (``PROVENANCE['RULE_PACKAGES']``): the .py files of RULE_PACKAGES and, found
    statically, every in-repo module they import, transitively (no import is run, so every host finds the same)."""
    root = Path(root) if root is not None else REPO_ROOT
    todo = []
    for pkg_name in RULE_PACKAGES:
        folder = root.joinpath(*pkg_name.split("."))
        todo += sorted(folder.glob("*.py"))
    seen = set()
    while todo:
        p = todo.pop()
        if p in seen:
            continue
        seen.add(p)
        for m in _imports(root, p):
            q = _module_path(root, m)
            if q is not None and q not in seen:
                todo.append(q)
    return sorted(seen, key=lambda q: q.relative_to(root).as_posix())


def rule_sources(root=None) -> dict:
    """{repo-relative path: sha256} of every rule source file (:func:`rule_source_files`) under ``root`` (default the
    repository), with line endings normalised to LF."""
    root = Path(root) if root is not None else REPO_ROOT
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for p in rule_source_files(root)}


def rule_values(config: V3Config) -> dict:
    """The rules in effect for ``config`` (``PROVENANCE['rules_hash']``): the config, the rule dataclasses as World3
    uses them and every upper-case module constant of RULE_PACKAGES, canonical."""
    from haishool.life9.planet import formation as fm
    consts = {}
    for mod in rule_modules() + [importlib.import_module(m) for m in RULE_VALUE_MODULES]:
        row = {}
        for name, v in sorted(vars(mod).items()):
            bare = name.lstrip("_")
            if not bare or not bare.isupper() or "CACHE" in bare:
                continue
            c = _canon(v)
            if c is not None:
                row[name] = c
        consts[mod.__name__] = row
    rules = {"Formation3Rules": f3.Formation3Rules(), "FormationRules": fm.FormationRules(),
             "ClimateRules": cl.ClimateRules(), "BioRules": bs.BioRules(seed_floor_kg_m2=0.0),
             "PatchRules": pt.PatchRules(patch_m=float(config.patch_m), cells=int(config.cells)),
             "SenseRules": s3.SenseRules()}
    return {"config": _canon(config.to_dict()), "dataclasses": {k: _canon(v) for k, v in rules.items()},
            "constants": consts, "earth_reference": inputs_digest(ch.earth_inputs_computed())}


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def rules_identity(config: V3Config, root=None) -> dict:
    """The rules hash of an experiment (``PROVENANCE['rules_hash']``): {"sha256" (over the next two), "sources_sha256",
    "values_sha256", "files" ({path: sha256})}. Equal hashes mean the same source, config, rule dataclasses and module
    constants."""
    files = rule_sources(root)
    src = _digest(files)
    val = _digest(rule_values(config))
    return {"sha256": _digest({"protocol": PROTOCOL_VERSION, "sources": src, "values": val}), "sources_sha256": src,
            "values_sha256": val, "files": files}


def derived_seed(seeds) -> int:
    """The seed of the run's generators (day, shadow, baseline) of the worlds stepped together."""
    h = hashlib.sha256(("life9-v3-world3:" + ",".join(str(int(s)) for s in seeds)).encode()).digest()
    return int.from_bytes(h[:8], "little") & ((1 << 63) - 1)


def setup_seed(seed: int, salt=None) -> int:
    """The seed of one world's set-up generator (``PROVENANCE['SETUP_SALT']``): sha256 of SETUP_SALT, the salt (the
    config's rng_seed, default 'default') and the world's seed alone."""
    tag = "default" if salt is None else str(int(salt))
    h = hashlib.sha256(f"{SETUP_SALT}:{tag}:{int(seed)}".encode()).digest()
    return int.from_bytes(h[:8], "little") & ((1 << 63) - 1)


# ============================================================================================ locomotion
def _world_rows(terrain: dict, w: int) -> dict:
    """World w's rows of a terrain dict (tensors with a leading world axis)."""
    return {k: (v[w:w + 1] if torch.is_tensor(v) and v.dim() >= 1 else v) for k, v in terrain.items()}


def make_terrain_per_world(globe, gens, specs) -> dict:
    """formation3.make_terrain3 of each world on its own generator, joined along the world axis
    (``PROVENANCE['SETUP_SALT']``)."""
    parts = [f3.make_terrain3(globe, g, [spec]) for g, spec in zip(gens, specs)]
    return {k: torch.cat([p[k] for p in parts]) for k in parts[0]}


def wave_factor(fr):
    """Drag of a body swimming at the surface over its drag deep under water, at Froude number Fr = v / sqrt(g L):
    1 + WAVE_DRAG_MAX Fr^4 / (Fr^4 + FR_HULL^4) (``PROVENANCE['WAVE_DRAG_MAX']``)."""
    f4 = fr ** 4
    return 1.0 + WAVE_DRAG_MAX * f4 / (f4 + FR_HULL ** 4)


def paddle_speed(p_thrust, k3, hull_v, iters: int = 48) -> torch.Tensor:
    """The surface swimming speed v >= 0 with wave_factor(v / hull_v) k3 v^3 = p_thrust (float64 bisection between
    the speeds without and with the full wave drag; the left side increases with v)."""
    p = torch.as_tensor(p_thrust).double().clamp_min(0)
    k3 = torch.as_tensor(k3).double().clamp_min(1e-300)
    hv = torch.as_tensor(hull_v).double().clamp_min(1e-300)
    hi = (p / k3) ** (1.0 / 3.0)
    lo = (p / (k3 * (1.0 + WAVE_DRAG_MAX))) ** (1.0 / 3.0)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        over = wave_factor(mid / hv) * k3 * mid ** 3 > p
        hi = torch.where(over, mid, hi)
        lo = torch.where(over, lo, mid)
    return 0.5 * (lo + hi)


def held_mass_kg(items, held) -> torch.Tensor:
    """Mass of the items each body holds [A, N] (kg)."""
    ok = held >= 0
    m = items.mass.gather(1, held.clamp_min(0).reshape(held.shape[0], -1)).reshape(held.shape)
    alive = items.alive.gather(1, held.clamp_min(0).reshape(held.shape[0], -1)).reshape(held.shape)
    return (m * (ok & alive)).sum(-1)


def locomotion_terms(b: B.Bodies, thrust, env: B.Env, geom=None, carry_kg=None) -> dict:
    """The bout's locomotion terms [A, N] for :func:`move_kt` (worked out once per bout, at its start;
    ``PROVENANCE['locomotion']``): the locomotion time share |thrust| and its sign, the metabolic power while moving
    (the muscles' full sustained power over MUSCLE_EFF), the land speed (KT force cost with the held load + air drag),
    the speed in water (bottom walking for a body denser than the water at its position that cannot tread it,
    paddling at the surface with wave drag for one that floats or treads, the tread power taken from its propulsion),
    the drag coefficients, the weight with the load (mg) and the body's shape and density. The moving power is
    body's sustained power, capped by its O2 supply."""
    e = B._env(b, env, geom)
    alive = b.alive
    thrust = torch.as_tensor(thrust, dtype=torch.float32, device=b.device).clamp(-1, 1) * alive
    share = thrust.abs().double()
    sh = B.shape(b)
    tis = B.tissues(b, sh)
    eff = B.MUSCLE_EFF
    p_sus = B._sustained(b, e, tis, sh)                       # the O2 supply caps it (body.PROVENANCE['breathing'])
    p_met = p_sus.double() / eff * (share > 0)
    m_body = sh["m"].double()
    load = torch.zeros_like(m_body) if carry_kg is None else torch.as_tensor(carry_kg, device=b.device).double()
    m = m_body + load
    g = e["gravity_m_s2"].double()
    l_c = sh["length"].double().clamp_min(1e-9)
    air = B.air_props(e["t_air_k"], e["p_air_pa"])
    frontal = sh["frontal"].double()
    k1 = KT_COST_J_PER_N * m * g / l_c
    k3_air = 0.5 * air["rho"].double() * B.C_DRAG * frontal
    rho = B.density(b, tis)
    rho_w = (B.water_density(geom, b.pos) if geom is not None else torch.full_like(rho, B.RHO_WATER)).double()
    sinks = rho.double() > rho_w                                                 # against the water where it is
    k3_w = 0.5 * rho_w * B.C_DRAG * frontal
    in_water = (1.0 - rho_w / rho.double()).clamp_min(0)                        # the share of its weight in water
    k1_bottom = KT_COST_J_PER_N * (m_body * in_water + load) * g / l_c
    tread = B.tread_power_w(b, sh, rho, rho_w.float(), e["gravity_m_s2"], tis).double()   # body.PROVENANCE['tread']
    treads = sinks & (p_met > tread) & alive
    # the speed solves for the moving bodies only, gathered to the front of each arena
    zero = torch.zeros_like(p_met)
    v_land, v_water = zero.clone(), zero.clone()
    go = alive & (share > 0)
    n_c = int(go.sum(1).max()) if go.numel() else 0
    if n_c:
        order = torch.argsort((~go).to(torch.int8), dim=1, stable=True)[:, :n_c]
        r = lambda x: _rows(x, order)                                                   # noqa: E731
        pm = r(p_met)
        vl = B._speed(r(k1), r(k3_air) / eff, pm)
        vb = B._speed(r(k1_bottom), r(k3_w) / eff, pm)
        tr = r(treads)
        vp = paddle_speed((pm - torch.where(tr, r(tread), torch.zeros_like(pm))) * eff * PADDLE_EFF, r(k3_w),
                          (r(g) * r(l_c)).sqrt())
        v_land.scatter_(1, order, vl)
        v_water.scatter_(1, order, torch.where(r(sinks) & ~tr, vb, vp))
    return {"thrust": thrust, "share": share, "p_met": p_met, "v_land": v_land, "v_water": v_water, "sinks": sinks,
            "treads": treads, "tread_w": tread, "rho": rho, "k3_air": k3_air, "k3_w": k3_w, "mg": m * g, "shape": sh}


def bout_reach(terms: dict, dt_s, b: B.Bodies | None = None, geom=None, pstate=None, probes: int = 16
               ) -> torch.Tensor:
    """The longest a body could travel this bout (m) [A, N], the sub-step budget's measure (:func:`move_kt` gives the
    realised path): its locomotion time at its land speed, or at the faster of its land and water speeds when water
    lies within its land reach along its heading (``probes`` points; without the geometry, always the faster)."""
    t = terms["share"] * float(dt_s) * (terms["p_met"] > 0)
    v_max = torch.maximum(terms["v_land"], terms["v_water"])
    if b is None or geom is None:
        return t * v_max
    land = t * terms["v_land"]
    sign = torch.sign(terms["thrust"]).double()
    direction = torch.stack((torch.cos(b.heading), torch.sin(b.heading)), -1).double() * sign[..., None]
    f = torch.arange(probes + 1, device=b.device, dtype=torch.float64) / probes
    pts = torch.remainder(b.pos.double()[..., None, :] + (land[..., None] * f)[..., None] * direction[..., None, :],
                          b.L).float()
    wet = (B._water_depth(geom, pstate, pts) > 0).any(-1)
    return torch.where(wet, t * v_max, land)


def _march(geom, pstate, L: float, pos0, direction, t_av, v_land, v_water, mg, p_met, height, sinks, moving,
           n_s: int, treads=None) -> dict:
    """The path of one sub-step for bodies [A, M] (float64 except the masks): marched in n_s segments up to the
    longest reach t_av x max(v_land, v_water); each segment crossed at its medium's speed (its wet share: water depth
    over the body's height), the net climb paid from the same power, the body stopping where its locomotion time
    runs out. Returns dist, t_land, t_water, t_sunk (time with the airway under water while moving), s_path (the
    path's wet share), climb (J), under_end (the airway under water where it stops), z0."""
    eff = B.MUSCLE_EFF
    dev = pos0.device
    vl, vw = v_land.clamp_min(1e-300), v_water.clamp_min(1e-300)
    reach = t_av * torch.maximum(v_land, v_water) * moving
    seg = reach / n_s
    mid = (torch.arange(n_s, device=dev, dtype=torch.float64) + 0.5) / n_s
    end = torch.arange(1, n_s + 1, device=dev, dtype=torch.float64) / n_s
    p_mid = torch.remainder(pos0[..., None, :] + (reach[..., None] * mid)[..., None] * direction[..., None, :],
                            L).float()
    depth = B._water_depth(geom, pstate, p_mid).double()                               # [A, M, n_s]
    s_k = (depth / height[..., None].clamp_min(1e-9)).clamp(0, 1)
    afloat = sinks if treads is None else sinks & ~treads           # a treading body holds its airway up while moving
    sunk_k = ((depth >= height[..., None]) & afloat[..., None]).double()
    slow_k = (1 - s_k) / vl[..., None] + s_k / vw[..., None]                          # s per metre
    lvl = torch.cumsum(seg[..., None] * slow_k, -1)                                     # level time to each end
    start = pos0.float()
    z0 = B._surface_m(geom, pstate, start).double()
    p_end = torch.remainder(pos0[..., None, :] + (reach[..., None] * end)[..., None] * direction[..., None, :],
                            L).float()
    zk = B._surface_m(geom, pstate, p_end).double()
    climb_t = (zk - z0[..., None]).clamp_min(0) * mg[..., None] / (eff * p_met.clamp_min(1e-300))[..., None]
    tot = lvl + climb_t                                                                 # time to reach each end
    over = tot > t_av[..., None]
    k = over.to(torch.int8).argmax(-1, keepdim=True)
    prev = torch.cat((torch.zeros_like(tot[..., :1]), tot[..., :-1]), -1)
    t1, t0 = tot.gather(-1, k)[..., 0], prev.gather(-1, k)[..., 0]
    cross = (k[..., 0].double() + ((t_av - t0) / (t1 - t0).clamp_min(1e-300)).clamp(0, 1)) / n_s
    frac = torch.where(over.any(-1), cross, torch.ones_like(cross)) * moving
    trav = (frac[..., None] * n_s - torch.arange(n_s, device=dev, dtype=torch.float64)).clamp(0, 1)
    part = trav * seg[..., None]                                                        # metres crossed per segment
    t_water = (part * s_k / vw[..., None]).sum(-1)
    t_land = (part * (1 - s_k) / vl[..., None]).sum(-1)
    t_sunk = (part * sunk_k * slow_k).sum(-1)
    crossed = part.sum(-1)
    dist = frac * reach
    end_pos = torch.remainder(pos0 + dist[..., None] * direction, L)
    here = B._water_depth(geom, pstate, start).double()
    s_here = (here / height.clamp_min(1e-9)).clamp(0, 1)
    s_path = torch.where(crossed > 0, (part * s_k).sum(-1) / crossed.clamp_min(1e-300), s_here)
    ef = end_pos.float()
    climb = (B._surface_m(geom, pstate, ef).double() - z0).clamp_min(0) * mg * moving
    d_end = B._water_depth(geom, pstate, ef).double()
    under_end = ((d_end >= height) & sinks).double()
    return {"dist": dist, "t_land": t_land, "t_water": t_water, "t_sunk": t_sunk, "s_path": s_path, "climb": climb,
            "under_end": under_end, "end_pos": end_pos, "seg_t": part * slow_k, "seg_sunk": sunk_k}


def move_kt(b: B.Bodies, turn, thrust, env: B.Env, *, geom=None, pstate=None, dt_s=None, terms: dict | None = None,
            samples: int = B.PATH_SAMPLES, active=None, still=None) -> dict:
    """One movement sub-step (``PROVENANCE['locomotion']``): turn by pi x turn, then, for ``dt_s`` (a number or
    [A, N]; default the bout), move for the locomotion share |thrust| of it at the full sustained power and stand for
    the rest. ``terms`` (:func:`locomotion_terms` at the bout's start) lets the sub-steps share one speed solve;
    ``active`` [A, N] bool limits the call to some bodies (the others neither move nor spend time). The path is
    marched in ``samples`` segments up to the longest reach (the faster medium's speed): each segment is crossed at
    its own medium's speed, the net climb is paid from the same power, and the body stops where its locomotion time
    runs out (:func:`_march`, worked out for the active bodies gathered to the front of each arena). ``still`` [A, N]
    bool: bodies that spend the time standing whatever their thrust (unconscious ones, ``PROVENANCE
    ['drowning_gate']``). Returns body.move's dict (work_j, metabolic_j, heat_j, dist_m, submerged (the path's share
    in water), climb_j, drag_j) plus sunk_s (time with the airway under water) and its dive structure in order
    (body.DIVE_KEYS: the path's segments, then the standing), moving_s and time_s [A, N] (the time this call stands
    for)."""
    A, N = b.shape
    dev = b.device
    act = b.alive if active is None else (b.alive & active.bool())
    dtt = torch.as_tensor(env.dt_s if dt_s is None else dt_s, dtype=torch.float64, device=dev).expand(A, N) * act
    turn = torch.as_tensor(turn, dtype=torch.float32, device=dev).clamp(-1, 1) * act
    b.heading = torch.remainder(b.heading + B.TURN_SPAN * turn, 2 * math.pi)
    t = locomotion_terms(b, thrust, env, geom) if terms is None else terms
    p_met, share, v_land, v_water, mg, sh = (t[k] for k in ("p_met", "share", "v_land", "v_water", "mg", "shape"))
    eff = B.MUSCLE_EFF
    sign = torch.sign(t["thrust"]).double()
    direction = torch.stack((torch.cos(b.heading), torch.sin(b.heading)), -1).double() * sign[..., None]
    if still is not None:
        share = torch.where(still.bool(), torch.zeros_like(share), share)
    t_av = share * dtt                                                      # locomotion time of this sub-step
    moving = act & (p_met > 0) & (t_av > 0)
    zero = torch.zeros(A, N, dtype=torch.float64, device=dev)
    pos0 = b.pos.double()
    if geom is not None:
        n_c = int(act.sum(1).max()) if act.numel() else 0
        if n_c:
            order = torch.argsort((~act).to(torch.int8), dim=1, stable=True)[:, :n_c]
            g = lambda x: _rows(x, order)                                               # noqa: E731
            treads = t.get("treads", torch.zeros_like(t["sinks"]))
            r = _march(geom, pstate, b.L, g(pos0), g(direction), g(t_av), g(v_land), g(v_water), g(mg), g(p_met),
                       g((2 * sh["b"]).double()), g(t["sinks"]), g(moving), max(1, int(samples)), treads=g(treads))
            full = lambda x: zero.clone().scatter_(1, order, x.double())               # noqa: E731
            dist, t_land, t_water, t_sunk = full(r["dist"]), full(r["t_land"]), full(r["t_water"]), full(r["t_sunk"])
            s_path, climb, under_end = full(r["s_path"]), full(r["climb"]), full(r["under_end"])
            end_pos = pos0.clone().scatter_(1, order[..., None].expand(-1, -1, 2), r["end_pos"])
            # the path's dive structure, worked out for the gathered bodies and scattered back (neutral elsewhere)
            moved = _dive_of(r["seg_t"], r["seg_sunk"], g(act))
            path_dive = {key: (torch.ones if key == "dive_all" else torch.zeros)(A, N, device=dev).scatter_(
                1, order, v) for key, v in moved.items()}
        else:
            dist = t_land = t_water = t_sunk = s_path = climb = under_end = zero
            end_pos = pos0
            path_dive = None
    else:
        dist = t_av * v_land * moving
        t_land, t_water, t_sunk, s_path, climb, under_end = t_av * moving, zero, zero, zero, zero, zero
        end_pos = torch.remainder(pos0 + dist[..., None] * direction, b.L)
        path_dive = None
    b.pos = torch.where(act[..., None], end_pos.float(), b.pos)
    t_move = (t_land + t_water + climb / (eff * p_met.clamp_min(1e-300))) * moving
    t_stand = (dtt - t_move).clamp_min(0)
    sunk_s = (t_sunk + t_stand * under_end) * act
    # the dive structure of this call: the path's segments in order, then the standing (body.DIVE_KEYS)
    stand_sunk = (t_stand * under_end * act).float()
    dive = path_dive if path_dive is not None else _dive_of(None, None, act)
    dive = B.dive_merge(dive, _dive_piece(stand_sunk, (t_stand * act).float(), under_end > 0.5))
    vel = (dist[..., None] * direction / dtt.clamp_min(1e-300)[..., None]).float()
    b.vel = torch.where(act[..., None], vel, b.vel)
    metabolic = p_met * (t_land + t_water) + climb / eff
    paddling = ~t["sinks"] | t.get("treads", torch.zeros_like(t["sinks"]))
    drag_w = torch.where(paddling, p_met * eff * t_water, t["k3_w"] * v_water ** 3 * t_water)
    drag = drag_w + t["k3_air"] * v_land ** 3 * t_land                              # leaves into water and air
    heat = (metabolic - drag - climb).clamp_min(0)
    af = act.float()
    return {"work_j": (metabolic * eff).float() * af, "metabolic_j": metabolic.float() * af,
            "heat_j": heat.float() * af, "dist_m": dist.float() * af, "submerged": s_path.float() * af,
            "climb_j": climb.float() * af, "drag_j": drag.float() * af, "sunk_s": sunk_s.float() * af,
            "moving_s": t_move.float() * af, "time_s": dtt.float(), "dt_s": dtt.float(),
            "treading": (t.get("treads", torch.zeros_like(act)) & moving & (t_water > 0)), **dive}


def _dive_piece(sunk_s, total_s, sunk) -> dict:
    """The dive structure (body.DIVE_KEYS) of one stretch spent wholly under water (``sunk`` True) or in air: a
    stretch of no time is neutral."""
    z = torch.zeros_like(sunk_s)
    whole = sunk | (total_s <= 0)
    return {"dive_lead_s": torch.where(whole, sunk_s, z), "dive_inner_s": z,
            "dive_trail_s": torch.where(whole, sunk_s, z), "dive_all": whole.float()}


def _dive_of(seg_t, seg_sunk, act) -> dict:
    """The dive structure of a path's segments [A, N, S] in order (times and 0/1 sunk flags)."""
    zero = torch.zeros(act.shape, dtype=torch.float32, device=act.device)
    out = {"dive_lead_s": zero, "dive_inner_s": zero, "dive_trail_s": zero, "dive_all": torch.ones_like(zero)}
    if seg_t is None:
        return out
    for k in range(seg_t.shape[-1]):
        tk = (seg_t[..., k] * act).float()
        out = B.dive_merge(out, _dive_piece(tk * (seg_sunk[..., k] > 0.5), tk, seg_sunk[..., k] > 0.5))
    return out


def sum_loco(parts: list, dt_s: float) -> dict:
    """The bout's locomotion from :func:`move_kt`'s sub-steps (per-body times): energies, distances, time sunk add;
    the dive structures merge in order (body.dive_merge); the submerged share is weighted by each part's time;
    ``dt_s`` the bout."""
    out = {k: sum(p[k] for p in parts) for k in ("work_j", "metabolic_j", "heat_j", "dist_m", "climb_j", "drag_j",
                                                  "sunk_s", "moving_s")}
    if all(k in p for p in parts for k in B.DIVE_KEYS):
        d = {k: parts[0][k] for k in B.DIVE_KEYS}
        for p in parts[1:]:
            d = B.dive_merge(d, p)
        out.update(d)
    time = sum(p["time_s"] for p in parts)
    out["submerged"] = sum(p["submerged"] * p["time_s"] for p in parts) / time.clamp_min(1e-30)
    out["dt_s"] = float(dt_s)
    return out


def cost_of_transport(mass_kg, gravity=K.G_STANDARD) -> float:
    """Level-ground metabolic cost of transport (J per kg and m) of move_kt for a body of mass_kg at low speed:
    KT_COST_J_PER_N g / body length (the body's ellipsoid at body.RHO_TISSUE and body.ASPECT). For the comparison
    with measured costs (PROVENANCE['KT_COST_J_PER_N']), not a target."""
    r1 = (3 * float(mass_kg) / (4 * math.pi * B.RHO_TISSUE)) ** (1.0 / 3.0)
    length = 2 * B.ASPECT * r1 / B.ASPECT ** (1.0 / 3.0)
    return KT_COST_J_PER_N * gravity / length


# ============================================================================================ the climate's spin-up
def spin_climate(state, globe, P, calibrate, plant_cover=None, fast_chunks: int = 8, slow_chunks: int = 4,
                 tol_k: float = 0.1):
    """``climate.spin_up`` with a per-world calibration mask ``calibrate`` [W] bool (``PROVENANCE['climate_target']``):
    calibrated worlds move their cloud albedo toward spec.t_surface_target_k as version 2 does; the others keep their
    cloud albedo and are run until their own projected equilibrium (T_eq - mean T) settles. Returns (state, report)."""
    r = P["rules"]
    W = P["W"]
    dev = globe.device
    mask = torch.as_tensor(calibrate, dtype=torch.bool, device=dev).reshape(W)
    years = P["year_days"].double().round().clamp_min(1)
    rotating = ~P["locked"]
    slow_days = r.spin_chunk_days
    if bool(rotating.any()):
        slow_days = int(min(730, max(r.spin_chunk_days, float(years[rotating].max()))))
    whole = torch.where(rotating & (years <= slow_days), torch.floor(slow_days / years) * years,
                        torch.full_like(years, float(slow_days)))
    target = P["t_target64"]
    wv = r.wv_lr_feedback / r.planck_feedback if r.h2o_feedback else 0.0
    report = {"chunks": [], "slow_chunk_days": slow_days, "calibrated": mask.tolist()}
    fast = slow = 0
    t_eq = err = None
    if fast_chunks + slow_chunks <= 0:
        report.update(cloud_albedo=state.cloud_albedo.tolist(), t_eq=None, error_k=None, chunks_run=0)
        return state, report
    while True:
        accelerate = fast < fast_chunks
        days = r.spin_chunk_days if accelerate else slow_days
        window = torch.full_like(years, float(days)) if accelerate else whole
        sums = torch.zeros(5, W, dtype=torch.float64, device=dev)
        for d in range(days):
            state, diag = cl.step(state, globe, P, plant_cover, accelerate)
            use = (days - d <= window).double()
            sums += use * torch.stack((diag["mean_t"], diag["toa_imbalance"], diag["planck"],
                                       diag["mean_insolation"], diag["dalbedo_dcloud"]))
        mean_t, imbalance, planck, mean_i, slope = sums / window
        lam = (planck * (1 - wv)).clamp_min(0.3)
        t_eq = mean_t + imbalance / lam
        err = torch.where(mask, t_eq - target, t_eq - mean_t)
        report["chunks"].append({"accelerated": accelerate, "cloud_albedo": state.cloud_albedo.tolist(),
                                 "mean_t": mean_t.tolist(), "t_eq": t_eq.tolist(), "error_k": err.tolist()})
        step_c = lam * (t_eq - target) / mean_i.clamp_min(1e-9) / slope.clamp_min(1e-3)
        cloud = (state.cloud_albedo.double() + step_c).clamp(0.0, r.cloud_max).float()
        state = replace(state, cloud_albedo=torch.where(mask, cloud, state.cloud_albedo))
        worst = float(err.abs().max())
        if accelerate:
            near = torch.where(mask, (mean_t - target).abs(), torch.zeros_like(mean_t))
            settled = worst < 0.25 and float(near.max()) < 0.3
            fast = fast_chunks if settled else fast + 1
        else:
            slow += 1
            if worst < tol_k or slow >= slow_chunks:
                break
        if fast >= fast_chunks and slow_chunks <= 0:
            break
    report.update(cloud_albedo=state.cloud_albedo.tolist(), t_eq=t_eq.tolist(), error_k=err.tolist(),
                  chunks_run=len(report["chunks"]))
    return state, report


# ============================================================================================ small helpers
def _host(obj, device="cpu", copy: bool = True):
    """A nest of dicts, lists and tensors with every tensor on ``device``: copied, or (``copy`` False) moved only
    where it is not there yet (a load that owns the tensors makes no second copy)."""
    if torch.is_tensor(obj):
        return obj.detach().to(device, copy=True) if copy else obj.detach().to(device)
    if isinstance(obj, dict):
        return {k: _host(v, device, copy) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)) and obj and any(torch.is_tensor(v) for v in obj):
        return type(obj)(_host(v, device, copy) for v in obj)
    return obj


def _ref(obj):
    """A nest of dicts with its tensors referenced (not copied)."""
    if isinstance(obj, dict):
        return {k: _ref(v) for k, v in obj.items()}
    return obj


def _clone(obj):
    """A nest of dicts with every tensor cloned on its own device."""
    if torch.is_tensor(obj):
        return obj.clone()
    if isinstance(obj, dict):
        return {k: _clone(v) for k, v in obj.items()}
    return obj


def _hash_update(h, name: str, t) -> None:
    if torch.is_tensor(t):
        t = t.detach().cpu().contiguous()
        h.update(f"|{name}|{t.dtype}|{tuple(t.shape)}|".encode())
        if t.numel():
            h.update(ctypes.string_at(t.data_ptr(), t.numel() * t.element_size()))
    elif isinstance(t, dict):
        for k in sorted(t, key=str):
            _hash_update(h, f"{name}.{k}", t[k])
    elif isinstance(t, (list, tuple)):
        for i, v in enumerate(t):
            _hash_update(h, f"{name}[{i}]", v)
    else:
        h.update(f"|{name}|{t!r}|".encode())


def _tensor_bytes(obj) -> int:
    if torch.is_tensor(obj):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(_tensor_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_tensor_bytes(v) for v in obj)
    return 0


def peak_rss_bytes() -> int | None:
    """The process's peak resident memory (bytes), where the platform tells it."""
    try:
        import resource                                   # POSIX
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        import sys
        return int(r) if sys.platform == "darwin" else int(r) * 1024
    except Exception:  # noqa: BLE001
        pass
    try:                                                  # Windows
        from ctypes import wintypes

        class _PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
        pmc = _PMC()
        pmc.cb = ctypes.sizeof(_PMC)
        k32 = ctypes.WinDLL("kernel32")
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi")
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]
        if psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
            return int(pmc.PeakWorkingSetSize)
    except Exception:  # noqa: BLE001
        pass
    return None


def _rows(t, order):
    """t [A, N, ...] at the slots order [A, M] -> [A, M, ...] (other values unchanged)."""
    if not torch.is_tensor(t) or t.dim() < 2:
        return t
    idx = order.reshape(order.shape + (1,) * (t.dim() - 2)).expand(order.shape + tuple(t.shape[2:]))
    return t.gather(1, idx)


class _BodySide:
    """The patch's exchange counters as the bodies see them: the counters less what the items' flows booked
    (``body.exchange_errors`` reads these attributes)."""

    def __init__(self, ps, items_side: dict):
        for k in ("w_taken", "w_given", "sea_water", "c_taken", "c_added", "n_removed", "n_added"):
            setattr(self, k, getattr(ps, k) - items_side.get(k, 0.0))


# ============================================================================================ the world
class World3:
    """W planets (one per seed) with their habitat patches, stepped together one day per :meth:`step_day`."""

    def __init__(self, config: V3Config | None = None, seeds=(781,), device="cpu", *, inputs=None, log=None,
                 resolved=None):
        cfg = config or V3Config()
        cfg.check()
        self.config = cfg
        self.sampled = [int(s) for s in seeds]
        if not self.sampled:
            raise ValueError("at least one seed")
        self.device = torch.device(device)
        self._log = log or (lambda msg: None)
        t0 = time.time()
        if resolved is not None:                             # (inputs, note) per seed, resolved before (the CLI)
            if len(resolved) != len(self.sampled):
                raise ValueError("one resolved (inputs, note) per seed")
            got = [(None if x is None else dict(x), note) for x, note in resolved]
        elif inputs is None:
            got = [planet_inputs(s, cfg) for s in self.sampled]
        else:
            if len(inputs) != len(self.sampled):
                raise ValueError("one inputs dict per seed")
            got = [(dict(x), "given") for x in inputs]
        # a seed whose chain gives no planet is recorded with its outcome, never skipped or replaced
        self.absent = [{"index": i, "seed": s, "source": cfg.source, **note}
                       for i, (s, (x, note)) in enumerate(zip(self.sampled, got)) if x is None]
        built = [(s, x, note) for s, (x, note) in zip(self.sampled, got) if x is not None]
        self.seeds = [s for s, _, _ in built]
        self.inputs = [x for _, x, _ in built]
        self.input_notes = [n for _, _, n in built]
        self.W = len(self.seeds)
        self.rng_seed = int(cfg.rng_seed) if cfg.rng_seed is not None else derived_seed(self.sampled)
        run_gen = torch.Generator().manual_seed(self.rng_seed)          # the run's generators (shared draws)
        # every world's set-up draws come from its own generator (PROVENANCE['SETUP_SALT'])
        self.setup_seeds = [setup_seed(s, cfg.rng_seed) for s in self.seeds]
        gens = [torch.Generator().manual_seed(v) for v in self.setup_seeds]
        self.reports = {"inputs": self.input_notes, "rng_seed": self.rng_seed, "setup_seeds": list(self.setup_seeds),
                        "absent": [{k: r.get(k) for k in ("seed", "kind", "outcome", "note")} for r in self.absent],
                        "excluded": [r["seed"] for r in self.absent if r.get("kind") in EXCLUDED_KINDS],
                        "errors": [{k: r.get(k) for k in ("seed", "cause", "note")} for r in self.absent
                                   if r.get("kind") == ERROR_KIND]}
        for r in self.absent:
            self._log(f"seed {r['seed']}: {r['outcome']} ({r['note']})")
        if not self.W:
            self._init_without_worlds()
            self.reports["setup_s"] = round(time.time() - t0, 2)
            return
        self._build_specs()
        # ---- draw 1: the globe's terrain
        self.globe0 = gb.Globe(cfg.G, torch.device("cpu"))
        terrain0 = make_terrain_per_world(self.globe0, gens, self.specs)
        self._build_static(terrain0)
        # ---- climate and the coarse global biosphere
        clim0 = cl.init_state(self.globe, self.P)
        calibrate = [not (s.frozen_mean or s.runaway) for s in self.specs]
        clim, crep = spin_climate(clim0, self.globe, self.P, calibrate, fast_chunks=cfg.climate_fast_chunks,
                                  slow_chunks=cfg.climate_slow_chunks, tol_k=cfg.climate_tol_k)
        self._log(f"climate spun up ({time.time() - t0:.1f} s)")
        c_air0 = clim.gas[:, cl.I_CO2].double() * M_C
        gas0 = clim.gas.clone()
        bio = bs.init_state(self.globe, self.P, self.Bp, clim)
        sown = bs.organic_carbon(bio).double()
        if cfg.bio_spin_days > 0:
            bio, clim, brep = bs.spin_up(bio, clim, self.globe, self.P, self.Bp, days=cfg.bio_spin_days,
                                         segments=cfg.bio_spin_segments)
        else:
            brep = {"days": 0}
        self.clim, self.bio = clim, bio
        self._spin_up_report(c_air0, gas0, sown)
        self._log(f"global biosphere spun up ({time.time() - t0:.1f} s)")
        # ---- draws 2-4: patches, their detail, the deposits
        self._build_patches(gens, terrain0)
        # ---- draw 5: the plants of the patches
        self.ps = pt.init_state(self.geom, self.prules) if self.A else None
        veg = self._spin_vegetation(gens)
        self._log(f"patch vegetation spun up ({time.time() - t0:.1f} s)")
        # ---- draws 6: founders; the run's generators from the run seed
        self._found(gens)
        day_seed, shadow_seed, base_seed = (int(v) for v in torch.randint(0, 1 << 62, (3,), generator=run_gen))
        self.gen = torch.Generator(device=self.device).manual_seed(day_seed)
        self.shadow_gen = torch.Generator(device=self.device).manual_seed(shadow_seed)
        self.baseline_seed = base_seed
        self.reports.update({
            "climate_spin_up": {k: crep.get(k) for k in ("cloud_albedo", "t_eq", "error_k", "chunks_run",
                                                         "slow_chunk_days", "calibrated")},
            "biosphere_spin_up": {k: brep.get(k) for k in ("days", "plant_c", "mean_npp_kg_c_m2_day", "mean_t")},
            "global_biosphere": "coarse global layer: version 2's biosphere without seed rain (sown once)",
            "vegetation_spin_up": veg, "fire": self._fire_report(), "setup_s": None,
            "extinct_day": [None] * self.A, "arena_capacity_bound_day": [None] * self.A,
            "arena_items_bound_day": [None] * self.A, "arena_fires_bound_day": [None] * self.A,
            "day_base": "days_completed"})
        self._rebase()
        self.reports["state_mb"] = round(self.memory_bytes() / 1e6, 1)              # PROVENANCE['capacity']
        self.reports["setup_s"] = round(time.time() - t0, 2)

    def _init_without_worlds(self) -> None:
        """No sampled seed gave a planet: the experiment still runs (its days pass) and records every outcome."""
        self.specs, self.spec_failures, self.no_land = [], {}, []
        self.A = 0
        self.arena_world = torch.zeros(0, dtype=torch.long, device=self.device)
        self.arena_cell = torch.zeros(0, dtype=torch.long, device=self.device)
        self.day, self.bout = 0, 0
        self.setup_seeds = []
        self.reports.update({"no_land": [], "patches": [], "extinct_day": [], "arena_capacity_bound_day": [],
                             "arena_items_bound_day": [], "arena_fires_bound_day": [], "state_mb": 0.0,
                             "day_base": "days_completed", "spin_up_replenishment": []})

    # --------------------------------------------------------------------------------------- set-up pieces
    def _build_specs(self):
        self.specs = [f3.build_planet3(x, s) for x, s in zip(self.inputs, self.seeds)]
        self.spec_failures = {s.seed: f for s, x in zip(self.specs, self.inputs)
                              if (f := f3.check_spec3(s, inputs=x))}
        self.radius_w = torch.tensor([s.radius_m for s in self.specs], dtype=torch.float64)
        self.gravity_w = torch.tensor([s.gravity_m_s2 for s in self.specs], dtype=torch.float32)

    def _build_static(self, terrain0):
        """Globe on the device, terrain, climate and biosphere parameters (deterministic from specs and terrain)."""
        dev = self.device
        cfg = self.config
        self.globe = self.globe0 if dev.type == "cpu" else gb.Globe(cfg.G, dev)
        self.terrain0 = terrain0
        self.terrain = {k: v.to(dev) for k, v in terrain0.items()}
        self.P = cl.make_params(self.specs, self.globe, self.terrain, float(self.radius_w[0]))
        r = self.radius_w.to(dev)
        self.P["ocean0"] = self.terrain["ocean_volume_m3"].double() * K.RHO_WATER / (FOUR_PI * r ** 2)
        self.Bp = bs.make_params(self.P, bs.BioRules(seed_floor_kg_m2=0.0))     # PROVENANCE['global_biosphere']
        self.prules = pt.PatchRules(patch_m=float(cfg.patch_m), cells=int(cfg.cells))
        self.srules = s3.SenseRules()

    def _spin_up_report(self, c_air0, gas0, sown) -> None:
        """What the global biosphere's spin-up made or held outside the books (``PROVENANCE
        ['spin_up_replenishment']``), per world: the carbon (organic plus air, after minus before the sowing and the
        spin-up) and the air's moles before and after."""
        area = (FOUR_PI * self.radius_w.to(self.clim.gas.device) ** 2).double()
        c1 = bs.organic_carbon(self.bio).double() + self.clim.gas[:, cl.I_CO2].double() * M_C
        made = (c1 - c_air0) * area
        names = ("N2", "O2", "CO2", "Ar")
        self.reports["spin_up_replenishment"] = [
            {"seed": self.seeds[w], "carbon_made_kg": float(made[w]), "sown_carbon_kg": float(sown[w] * area[w]),
             "air_mol_before": {n: float(gas0[w, cl.GASES.index(n)] * area[w]) for n in names},
             "air_mol_after": {n: float(self.clim.gas[w, cl.GASES.index(n)] * area[w]) for n in names}}
            for w in range(self.W)]

    def _build_patches(self, gens, terrain0):
        """Arenas: patch cells (draw 2), fine detail (draw 3), deposits (draw 4), each world from its own generator;
        a world without land draws its patches over all its cells (sea patches, ``PROVENANCE['patch_cells']``)."""
        cfg, dev = self.config, self.device
        land = terrain0["land"]
        has_land = land.any(-1)
        self.no_land = [not bool(v) for v in has_land]
        if self.W and cfg.patches > 0:
            cells = torch.stack([pt.choose_patch_cells(self.globe0, land[w:w + 1], self.globe0.area64, gens[w],
                                                       cfg.patches)[0] for w in range(self.W)])
            world = torch.arange(self.W).repeat_interleave(cfg.patches)
            cell = cells.reshape(-1)
        else:
            world = torch.zeros(0, dtype=torch.long)
            cell = torch.zeros(0, dtype=torch.long)
        self.A = int(world.numel())
        self.arena_world = world.to(dev)
        self.arena_cell = cell.to(dev)
        self.reports["patches"] = [{"world": int(w), "cell": int(c)} for w, c in zip(world, cell)]
        self.reports["no_land"] = [self.seeds[w] for w in range(self.W) if self.no_land[w]]
        if not self.A:
            self.geom = None
            self.deposit = None
            return
        r = self.prules
        g0 = self.globe0
        elev = terrain0["elevation_m"]
        rms = pt.detail_rms(g0, elev, self.radius_w, r)[world, cell].float()
        base = elev[world, cell].float()
        detail = torch.cat([pt.fractal_detail(cfg.patches, r.cells, gens[w], r.hurst) for w in range(self.W)]
                           ).double() * rms.double()[:, None, None]
        sea = terrain0["sea_level_m"].float()[world]
        geom = pt.geometry_from_elevation(base.double()[:, None, None] + detail, r.patch_m, sea, self.radius_w[world],
                                          world=world, cell=cell, W=self.W, patches=cfg.patches,
                                          basis=(g0.east[cell], g0.north[cell], g0.centers[cell]), base_m=base,
                                          detail_rms_m=rms, condition_depth_m=r.lake_keep_rms * rms.double(), rules=r)
        self.geom = pt.PatchGeometry.from_state(_host(geom.state_dict(), dev))
        crust = torch.tensor([[float(s.crust.get(name, 0.0)) for name in mat.CRUST_SPECIES] for s in self.specs],
                             dtype=torch.float64)
        dep_report: dict = {}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            stock = torch.cat([gb.make_deposits(g0, 1, gens[w], _world_rows(terrain0, w), crust[w:w + 1],
                                                mat.CRUST_SPECIES, report=dep_report) for w in range(self.W)])
        column = stock.double()[world, cell].to(dev)                            # [A, Sd] kg/m^2, the cell's column
        self.deposit = column[:, None, :].expand(self.A, r.cells * r.cells, column.shape[-1]).clone()  # per fine cell
        self.reports["deposits"] = {"warnings": [str(w.message) for w in caught],
                                    "species": list(mat.CRUST_SPECIES)}

    def _arena_forcing(self, cdiag, day: int, bouts: int):
        return pt.forcing_from_climate(self.geom, self.clim, cdiag, self.P, day, bouts=bouts, specs=self.specs,
                                       rules=self.prules)

    def _global_day(self):
        """One day of the climate and the coarse global biosphere; returns (day index, climate diagnostics)."""
        day = int(self.clim.day)
        clim, cdiag = cl.step(self.clim, self.globe, self.P, bs.cover(self.bio, self.Bp))
        bio, clim, _ = bs.step(self.bio, clim, self.globe, self.P, self.Bp, cdiag)
        self.clim, self.bio = clim, bio
        return day, cdiag

    def _spin_vegetation(self, gens) -> dict:
        """Sow the patches (draw 5, each world's arenas from its own generator) and grow them with the coupled daily
        climate for veg_spin_days; the patches' exchange with the air is booked to the climate each day."""
        if not self.A:
            return {"days": 0, "note": "no patches"}
        cfg = self.config
        c_before = self.ps.c_air.clone()
        n = self.prules.cells
        u = torch.cat([torch.rand(cfg.patches, n, n, generator=gens[w]) for w in range(self.W)])
        self.ps = pt.sow(self.ps, self.geom, None, self.prules, u=u)
        sown = self.ps.c_air - c_before
        self._add_gas_arena(sown, torch.zeros_like(sown))         # the sown carbon is taken from the air
        for _ in range(cfg.veg_spin_days):
            day, cdiag = self._global_day()
            f = self._arena_forcing(cdiag, day, max(1, cfg.veg_spin_bouts))
            c0, n0 = self.ps.c_air.clone(), self.ps.n_fixed.clone()
            self.ps, _ = pt.step(self.ps, self.geom, f, self.prules)
            self._add_gas_arena(self.ps.c_air - c0, self.ps.n_fixed - n0)
        cm2 = self.geom.cell_m2
        return {"days": cfg.veg_spin_days, "sown_kg_c": sown.tolist(),
                "plant_kg_c": (self.ps.plant.double().sum((1, 2)) * cm2).tolist(),
                "covered_share": ((self.ps.plant > 1e-6) & ~self.geom.sea).float().mean((1, 2)).tolist(),
                "lake_share": pt.lake_share(self.geom).tolist(), "founding": self._founding_day()}

    def _founding_day(self) -> list:
        """The founding day of each world (``PROVENANCE['founding_day']``): the climate's day, its day of the year and
        the declination (degrees)."""
        day = int(self.clim.day)
        dec = cl.declination(self.P, day).double()
        years = self.P["year_days"].double()
        return [{"day": day, "year_days": round(float(years[w]), 3),
                 "day_of_year": round(float(day % max(float(years[w]), 1.0)), 3),
                 "declination_deg": round(math.degrees(float(dec[w])), 3)} for w in range(self.W)]

    def _fire_report(self) -> list:
        """Whether each world's air allows fire (``PROVENANCE['fire_possible']``): the O2 mole fraction of the dry air
        against crafting.X_O2_MIN, uncertain (None) outside formation3.FIRE_FIT_PA of total pressure."""
        pp = cl.partial_pressures(self.clim, self.P)
        x = pp[:, cl.I_O2] / pp.sum(-1).clamp_min(1e-300)
        total = cl.pressures(self.clim, self.globe, self.P)["total"]
        out = []
        for w in range(self.W):
            fire = f3.fire_possible(float(x[w]), float(total[w]))
            out.append({"seed": self.seeds[w], "x_o2_dry": round(float(x[w]), 5), "x_o2_min": float(cr.X_O2_MIN),
                        "pressure_pa": round(float(total[w]), 1), **fire})
        return out

    def _add_gas_arena(self, c_fixed_kg, n_fixed_kg, body_o2=None, body_co2=None, items_air=None):
        """Book per-arena gas exchanges [A] (float64) to the climate's column (PROVENANCE['gas_booking'])."""
        dev = self.device
        mol = torch.zeros(self.A, 4, dtype=torch.float64, device=dev)
        cfix = c_fixed_kg.double() / M_C
        mol[:, cl.I_CO2] -= cfix
        mol[:, cl.I_O2] += cfix
        mol[:, cl.I_N2] -= n_fixed_kg.double() / (2 * M_N)
        if body_o2 is not None:
            mol[:, cl.I_O2] -= body_o2
            mol[:, cl.I_CO2] += body_co2
        if items_air is not None:
            mol[:, cl.I_O2] += items_air[:, _GAS["O2"]] / M_O2
            mol[:, cl.I_CO2] += items_air[:, _GAS["CO2"]] / M_CO2
        per_w = torch.zeros(self.W, 4, dtype=torch.float64, device=dev).index_add_(0, self.arena_world, mol)
        col = per_w / (FOUR_PI * self.radius_w.to(dev) ** 2)[:, None]
        self.clim = cl.add_gas(self.clim, col)
        if hasattr(self, "booked"):
            self.booked["gas_mol_m2"] += col

    def _found(self, gens):
        """The founders (draws 6), each world's from its own generator, drawn on the CPU and put into the state
        allocated once on the world's device (``body.found_into``; ``PROVENANCE['capacity']``)."""
        cfg = self.config
        dt = torch.bfloat16 if cfg.weight_dtype == "bfloat16" else torch.float32
        L = float(cfg.patch_m)
        A = self.A
        N = cfg.capacity
        dev = self.device
        if A:
            genome = B.empty_genome(A, N, s3.IN_DIM, cfg.hidden, dt, dev)
            wh = torch.zeros(A, N, cfg.hidden, cfg.hidden, dtype=dt, device=dev)
            b = B.empty(A, N, L, genome=genome, wh_live=wh, hidden=b3.init_state((A, N), cfg.hidden, dev), device=dev)
            for w in range(self.W):
                a0 = w * cfg.patches
                B.found_into(b, a0, a0 + cfg.patches, cfg.founders, gens[w], in_dim=s3.IN_DIM, hidden=cfg.hidden,
                             k_range=cfg.k_founder, t_k=293.0)
            B.reset_ledgers(b)
        else:
            b = B.empty(0, N, L, device=dev)
        self.b = b
        if A:
            # founders at the fine air temperature where they stand (PROVENANCE['founder_temperature'])
            f = self._arena_forcing(self._last_cdiag(), int(self.clim.day), 1)
            t_fine = pt.fine_temperature(self.geom, f)
            t_here = pt.sample(self.geom, t_fine, b.pos)
            b.body_k = torch.where(b.alive, t_here, b.body_k)
        self.shadow = B.shadow_of(b) if cfg.shadow else None
        dev = self.device
        self.items, self.fires = mp.new_pools(A, cfg.items, cfg.fires, device=dev)
        self.iled = mp.new_ledger(self.items, self.fires)
        self.vol = pt.init_volatiles(self.geom) if A else None
        z = torch.zeros(A, N, device=dev)
        self.last = {"loudness": z.clone(), "vocal": torch.zeros(A, N, b3.VOCAL_DIMS, device=dev),
                     "mouth": z.clone(), "thrust": z.clone(), "o2_mol_s": z.clone()}

    def _last_cdiag(self):
        """Climate diagnostics of the present state without stepping it (a step on a copy)."""
        _, cdiag = cl.step(self.clim.clone(), self.globe, self.P, bs.cover(self.bio, self.Bp))
        return cdiag

    def _rebase(self):
        """Start every ledger of the bodies' era (after the spin-ups and the founding)."""
        dev, A, W = self.device, self.A, self.W
        self.clim = cl.reset_water_ledger(self.clim, self.globe)
        self.bio = bs.reset_ledgers(self.bio, self.clim)
        if A:
            self.ps = pt.reset_ledgers(self.ps, self.geom, self.prules)
        z = lambda *s: torch.zeros(*s, dtype=torch.float64, device=dev)   # noqa: E731
        nn = self.config.cells * self.config.cells
        self.carry = {"vapour": z(W, self.globe.C), "ocean": z(W)}
        self.residual = {"c": z(A, nn), "n": z(A, nn)}
        self.items_side = {k: z(A) for k in ("c_taken", "c_added", "n_removed", "n_added")}
        self.counters = {"crust_in_kg": z(A, mat.S), "mineral_sink_el_kg": z(A, len(mat.ELEMENTS)),
                         "organic_ho_kg": z(A), "items_water_air_kg": z(A), "water_air_kg": z(A),
                         "skin_char_kg": z(A), "deposit_out_kg": z(A, len(mat.CRUST_SPECIES))}
        self.booked = {"gas_mol_m2": z(W, 4), "vapour_kg": z(W), "ocean_kg": z(W),
                       "vapour_released_kg": z(W), "ocean_released_kg": z(W)}
        self.gas_external0 = self.clim.gas_external.clone()
        self.water_external0 = self.clim.water_external.clone()
        self.stats_day = {k: z(A) for k in BEHAVIOUR_KEYS}
        self.stats_total = {k: z(A) for k in BEHAVIOUR_KEYS}
        self.day = 0
        self.bout = 0
        self.cdiag = None
        self.forcing = None
        self.bout_sw = None
        self.day0 = self._day_marks()
        self._mark = self._marks()
        self.base = {"local": self._local_stocks(), "marks": {k: v.clone() for k, v in self._mark.items()}}
        self.exchange_snap = B.exchange_snapshot(self.b, _BodySide(self.ps, self.items_side)) if A else None

    def _day_marks(self) -> dict:
        """The ledger entries of DAY_KEYS now [A] (float64), for the day's differences in the summary."""
        if not self.A:
            return {}
        return {k: self.b.ledger[k].clone() for k in DAY_KEYS}

    # --------------------------------------------------------------------------------------- the day
    def step_day(self, bout_callback=None) -> None:
        """One day: climate and global biosphere, the patches' day, the bouts, rot, the books. ``bout_callback``
        (world, bout) is called after each bout (viewer frames)."""
        if not self.W:                                   # no sampled seed gave a planet: only the days pass
            self.day += 1
            return
        for k in self.stats_total:                       # yesterday's behaviour joins the totals
            self.stats_total[k] += self.stats_day[k]
            self.stats_day[k].zero_()
        self.day0 = self._day_marks()
        day, cdiag = self._global_day()
        self.cdiag = cdiag
        if self.A:
            self.forcing = self._arena_forcing(cdiag, day, self.config.bouts)
            self.ps, pdiag = pt.step(self.ps, self.geom, self.forcing, self.prules)
            self.bout_sw = pdiag["bout_sw"]
            self._air_today()
            for k in range(self.config.bouts):
                self.bout = k
                self._bout(k)
                if bout_callback is not None:
                    bout_callback(self, k)
            self.bout = 0
            self._rot()
        self._book_day()
        if self.A:
            self._mark_ends()
        self.day += 1

    def _mark_ends(self) -> None:
        """The day each arena went extinct, the day its slots first bound and the days its item and fire pools first
        overflowed, as days completed (``PROVENANCE['extinction']``, ``['capacity_bound']``, ``['pools_bound']``,
        ``['day_base']``); one host sync a day."""
        pop = self.b.alive.sum(1)
        full = self.b.ledger["capacity_full"]
        st = self.stats_day
        items = st["no_item_slot"] + st["carcass_no_slot"]
        rows = torch.stack((pop.double(), full.double(), items.double(), st["no_fire_slot"].double())).tolist()
        done = self.day + 1                                    # days completed at the end of this day
        ext = self.reports.setdefault("extinct_day", [None] * self.A)
        cap = self.reports.setdefault("arena_capacity_bound_day", [None] * self.A)
        itb = self.reports.setdefault("arena_items_bound_day", [None] * self.A)
        frb = self.reports.setdefault("arena_fires_bound_day", [None] * self.A)
        for a in range(self.A):
            if ext[a] is None and rows[0][a] == 0:
                ext[a] = done
            if cap[a] is None and rows[1][a] > 0:
                cap[a] = done
            if itb[a] is None and rows[2][a] > 0:
                itb[a] = done
            if frb[a] is None and rows[3][a] > 0:
                frb[a] = done
        if "capacity_bound_day" not in self.reports and any(v is not None for v in cap):
            self.reports["capacity_bound_day"] = done          # PROVENANCE['capacity_bound']
        if "items_bound_day" not in self.reports and any(v is not None for v in itb):
            self.reports["items_bound_day"] = done             # PROVENANCE['pools_bound']
        if "fires_bound_day" not in self.reports and any(v is not None for v in frb):
            self.reports["fires_bound_day"] = done

    def run(self, days: int, callback=None, bout_callback=None) -> None:
        """``days`` days; ``callback`` (world) after each day, ``bout_callback`` (world, bout) after each bout."""
        for _ in range(int(days)):
            self.step_day(bout_callback)
            if callback is not None:
                callback(self)

    def _air_today(self):
        """The arenas' air for the day (``PROVENANCE['barometric']``): senses3.Air at each arena's base height, the
        O2 partial pressure and the air pressure per fine cell [A, n, n] for the bodies, and the fine air
        temperature."""
        dev = self.device
        w = self.arena_world
        geom = self.geom
        pp = cl.partial_pressures(self.clim, self.P)[w].to(dev)                        # [A, 4] Pa at the sea level
        f = self.forcing
        g = self.gravity_w.to(dev)[w].double()
        gases = torch.tensor([K.molar_mass(x) for x in ("N2", "O2", "CO2", "Ar")], dtype=torch.float64, device=dev)
        m_air = (pp * gases).sum(-1) / pp.sum(-1).clamp_min(1e-300)                      # kg/mol of the dry air
        self.t_fine = pt.fine_temperature(geom, f)
        lapse = f.lapse_k_m.double()
        sea = geom.sea_level_m.double()
        h_fine = (geom.elev.double() - sea[:, None, None]).clamp_min(0)
        t_col = self.t_fine.double() + 0.5 * lapse[:, None, None] * h_fine
        fac = torch.exp(-(g * m_air)[:, None, None] * h_fine / (K.R_GAS * t_col.clamp_min(1.0)))
        h_base = (geom.base_m.double() - sea).clamp_min(0)
        t_base = f.t_air_k.double() + 0.5 * lapse * h_base
        fac_a = torch.exp(-g * m_air * h_base / (K.R_GAS * t_base.clamp_min(1.0)))
        vapour = f.vapour_pa()
        self.air = s3.Air.from_partial(pp * fac_a[:, None], f.t_air_k, self.gravity_w.to(dev)[w], vapour)
        self.p_o2_fine = (pp[:, cl.I_O2][:, None, None] * fac).float()
        self.p_air_fine = (pp.sum(-1)[:, None, None] * fac + vapour.double()[:, None, None]).float()
        self.p_factor_arena = fac_a.float()

    # --------------------------------------------------------------------------------------- one bout
    def _bout(self, k: int) -> None:
        cfg, b, geom, ps = self.config, self.b, self.geom, self.ps
        if not bool(b.alive.any()):
            self._smells(None, None)
            return
        dev = self.device
        A, N = b.shape
        dt = DAY_S / cfg.bouts
        L = float(cfg.patch_m)
        n = cfg.cells
        gen = self.gen
        st = self.stats_day
        air = self.air
        g_a = air.g_m_s2.to(dev)
        env = replace(B.patch_env(geom, self.forcing, self.bout_sw[:, k], gravity_m_s2=g_a, dt_s=dt),
                      p_o2_pa=self.p_o2_fine, p_air_pa=self.p_air_fine)
        scene = s3.scene_from_patch(geom, ps, self.forcing, k, bout_sw=self.bout_sw, rules=self.srules,
                                    prules=self.prules)
        detail = k == SENSE_DETAIL_BOUT
        # ---- 1. senses
        x = self.sense(scene, details=detail)
        # ---- 2. brain, with the units its brain tissue holds (PROVENANCE['brain_units'])
        sh = B.shape(b)
        tis = B.tissues(b, sh)
        brain = dict(b.genome)
        brain["k"] = B.active_units(b, tis)
        width = int(b.genome["k"].max())
        h_before = b.hidden
        out, h_new = b3.think(brain, b.wh_live, b.hidden, x, width=width)
        motor = b3.decode(out)
        # a body whose O2 store is spent is unconscious: its outputs act on nothing (PROVENANCE['drowning_gate'])
        store = B.o2_store_mol(b, env, geom, tis)
        awake0 = b.alive & ~((b.o2_debt_mol > 0) & (b.o2_debt_mol >= store))   # a body without a store breathes
        motor = {key: v * (awake0 if v.dim() == 2 else awake0[..., None]) if torch.is_tensor(v) and
                 v.shape[:2] == awake0.shape else v for key, v in motor.items()}
        b.hidden = torch.where(b.alive[..., None], h_new, torch.zeros_like(h_new))
        alive0 = b.alive.clone()
        slot_id0 = b.uid.clone()
        mouth = B.mouth_position(b, sh)
        reach = B.mouth_reach_m(b, sh)
        depth = s3.water_depth(geom, ps, self.prules)
        plant_here = (pt.sample(geom, ps.plant, mouth) > 0) & alive0
        water_here = (pt.sample(geom, depth, mouth) > 0) & alive0
        if detail:
            self._replay(x, brain, mouth, depth)
        # ---- 3. contact physics: grip, press, rub, force, place, release; fire and heat
        peak = B.muscle_peak_w(b, tis)
        sustained = B.sustained_power_w(b, env, geom, tis, sh)          # the O2 supply caps it
        t_mouth = pt.sample(geom, self.t_fine, mouth)
        wet = water_here
        g_k, g_c = mp.ground_thermal(pt.sample(geom, ps.soil, mouth), self.prules.climate.bucket_kg_m2)
        ground = {"kg_m2": mp.ground_stock(ps, geom, self.deposit), "cells": n}
        res = mp.bout(self.items, self.fires, b.held, motor, gen, alive=b.alive, pos=b.pos, mouth_pos=mouth,
                      body_mass_kg=sh["m"] * b.alive, peak_w=peak, sustained_w=sustained, gravity=g_a,
                      x_o2=air.x_o2, air_k=t_mouth, L=L, dt_s=dt, thrust=motor["thrust"], vel=b.vel, reach_m=reach,
                      skin_k=b.body_k, wet=wet, ground_k=g_k, ground_c=g_c, ground=ground, ledger=self.iled, cells=n)
        self._ground_taken(res["force"].get("ground_kg"))
        self._to_patch(res["rub"].get("soil_c_kg"), res["rub"].get("soil_n_kg"), res["rub"].get("to_soil_kg"))
        b.damage = b.damage + res["damage"] * b.alive
        lean = b.mass_kg.clamp_min(B.MIN_KG)
        rub, force = res["rub"], res["force"]
        char = rub["skin_char_kg"] * b.alive
        b.damage = b.damage + char / (B.WOUND_LETHAL_SHARE * lean)
        # heat the body's own manipulation leaves in it (PROVENANCE['hand_heat'])
        hand_j = (rub["skin_heat_j"] + rub["skin_char_j"] + rub["limb_j"]
                  + (force["strokes"] - force["hits"]).clamp_min(0) * force["swing_j"]) * b.alive
        t_items = pt.sample(geom, self.t_fine, self.items.pos[..., :2])
        t_fires = pt.sample(geom, self.t_fine, self.fires.pos[..., :2])
        fire_t0 = self.fires.temp_k.clone()
        fire_on0 = self.fires.alive.clone()
        fl = mp.heat_step(self.items, self.fires, b.held, x_o2=air.x_o2, air_k=t_items, fire_air_k=t_fires, L=L,
                          dt_s=dt, ledger=self.iled, cells=n, substeps=cfg.fire_substeps, mouth_pos=mouth,
                          body_pos=b.pos, alive=b.alive)
        self._to_patch(fl.get("soil_c_kg"), fl.get("soil_n_kg"), fl.get("to_soil_kg"))
        fire_live = fire_on0 & (fl["mean_hrr_w"] > 0)
        burning = bool(fire_live.any())
        # ---- 4. locomotion in each body's own sub-steps, the mouth on what it touches (PROVENANCE['substeps'])
        thrust = res["thrust_share"]
        terms = locomotion_terms(b, thrust, env, geom, carry_kg=held_mass_kg(self.items, b.held))
        far = bout_reach(terms, dt, b, geom, ps)
        bad = ~torch.isfinite(far) & b.alive
        if bool(bad.any()):                                   # counted, never a crash: such a body does not move
            st["nonfinite"] += bad.double().sum(1)
            terms["share"] = torch.where(bad, torch.zeros_like(terms["share"]), terms["share"])
            far = torch.where(bad, torch.zeros_like(far), far)
        sub_len = cfg.substep_len_m
        k_i = torch.ceil(far / sub_len).clamp(1, cfg.max_substeps).long()
        k_i = torch.where(b.alive, k_i, torch.ones_like(k_i))
        k_max = int(k_i.max())
        dt_i = dt / k_i.double()
        per_step = float((far / k_i.double()).max())
        samples = int(min(B.PATH_SAMPLES, max(2, math.ceil(2 * per_step / geom.dx))))
        st["substeps"] += torch.where(b.alive, k_i, torch.zeros_like(k_i)).amax(1).double()
        st["body_substeps"] += (k_i * b.alive).double().sum(1)
        take = lambda pool, a, i, kg: mp.remove_mass(pool, b.held, a, i, kg, ledger=self.iled)["species_kg"]  # noqa
        parts = []
        bite = torch.zeros(A, N, device=dev)
        dist = torch.zeros(A, N, dtype=torch.float64, device=dev)
        coarse = torch.zeros(A, N, dtype=torch.bool, device=dev)
        flux = torch.zeros(A, N, dtype=torch.float64, device=dev)
        got = {key: torch.zeros(A, N, dtype=torch.float64, device=dev) for key in ("plant", "water", "bitten", "item")}
        touched = {key: torch.zeros(A, N, dtype=torch.bool, device=dev) for key in ("body", "item")}
        zero_turn = torch.zeros_like(thrust)
        sunk = torch.zeros(A, N, device=dev)
        use = self._o2_use(b, tis, terms)                                  # mol/s (PROVENANCE['drowning_gate'])
        debt0 = b.o2_debt_mol.clamp_min(0)
        dive = {"dive_lead_s": torch.zeros_like(sunk), "dive_inner_s": torch.zeros_like(sunk),
                "dive_trail_s": torch.zeros_like(sunk), "dive_all": torch.ones_like(sunk)}
        out_cold = torch.zeros_like(b.alive)
        treading = torch.zeros_like(b.alive)
        for s in range(k_max):
            # the debt of the dive in progress: carried from the last bout while no breath has come; a body whose store
            # is spent is unconscious and only stands where it is (PROVENANCE['drowning_gate'])
            cur = torch.where(dive["dive_all"] > 0.5, debt0 + use * dive["dive_lead_s"], use * dive["dive_trail_s"])
            cold = b.alive & (cur > 0) & (cur >= store)
            out_cold |= cold
            act = b.alive & (k_i > s)
            pos_s = b.pos.clone()
            loco = move_kt(b, (motor["turn"] if s == 0 else zero_turn) * ~cold, thrust, env, geom=geom, pstate=ps,
                           dt_s=dt_i, terms=terms, samples=samples, active=act, still=cold)
            parts.append(loco)
            dive = B.dive_merge(dive, {key: loco[key] for key in B.DIVE_KEYS})
            treading |= loco["treading"]
            sunk = sunk + loco["sunk_s"]
            dist = dist + loco["dist_m"].double()
            coarse |= act & (loco["dist_m"] > sub_len * (1 + 1e-6))
            mp.carry(self.items, b.held, B.mouth_position(b), L)
            if burning:                                       # the mean of the flux at the sub-step's two ends
                q = self._fire_flux(fl, fire_t0, fire_live, pos_s, L) + self._fire_flux(fl, fire_t0, fire_live, b.pos, L)
                flux = flux + 0.5 * q * dt_i * act
            eat = B.ingest(b, motor["mouth"] * (act & ~cold), env, geom=geom, pstate=ps, pool=self.items,
                           prules=self.prules,
                           take_items=take, dt_body=dt_i, disc_share=1.0 / k_i.double(), settle=s == 0)
            bite = bite + eat["bite_work_j"].float()
            got["plant"] += eat["plant_kg"]
            got["water"] += eat["drunk_kg"] + eat["sea_kg"]
            got["bitten"] += eat["bitten_kg"]
            got["item"] += eat["item_kg"]
            touched["body"] |= eat["contact"] == 3
            touched["item"] |= eat["contact"] == 2
        loco = sum_loco(parts, dt)
        direction = torch.stack((torch.cos(b.heading), torch.sin(b.heading)), -1) * torch.sign(thrust)[..., None]
        b.vel = (dist.float() / dt)[..., None] * direction * b.alive[..., None]
        # ---- 5. physiology and death
        cross = math.pi * sh["r1"].double() ** 2
        heat_w = ((B.EMISSIVITY_BODY * flux / dt * cross).float() + hand_j / dt) * b.alive
        met = B.metabolism(b, replace(env, heat_w=heat_w), geom=geom, pstate=ps, loco=loco,
                           eat={"bite_work_j": bite}, work_j=res["work_j"], loudness=motor["loudness"],
                           develop=motor["divide"], prules=self.prules)
        spawn = partial(self._spawn_counted, L=L)
        dead = B.deaths(b, geom=geom, pstate=ps, pool=self.items, selection=cfg.selection, gen=gen,
                        spawn_items=spawn, prules=self.prules)
        if "swap_from" in dead and dead["swap_from"].numel():
            fa, fn = dead["swap_from"].unbind(-1)
            ta, tn = dead["swap_to"].unbind(-1)
            B._swap_rows(h_before, fa, fn, ta, tn)
        if self.day == 0 and k == 0 and "founding" not in self.reports:
            self._founding_report(alive0, dead["dead"])
        # ---- 6. plasticity of the living, modulated by the state after the bout
        lvl = x[..., s3.INTERO_SLICE][..., b3.INTERO_NAMES.index("light")]
        intero = self.interoception(lvl)
        bf16 = b.wh_live.dtype == torch.bfloat16
        brain = dict(b.genome)
        brain["k"] = B.active_units(b)
        b3.hebbian(b.wh_live, brain, h_before, b.hidden, intero, b.alive, width=width, gen=gen if bf16 else None)
        # ---- 7. division: the children whose development is complete (PROVENANCE['development'])
        div = B.divide(b, gen)
        b.age_bouts = b.age_bouts + b.alive.long()
        # ---- 8. the shadow, smells, the bout's record
        if self.shadow is not None:
            B.shadow_step(self.shadow, dead["dead"].sum(1), div["born"], self.shadow_gen)
        self._smells(met, fl)
        keep = b.alive & alive0 & (b.uid == slot_id0)            # a newborn in a freed slot made no call yet
        self.last = {"loudness": motor["loudness"] * keep, "vocal": motor["vocal"] * keep[..., None],
                     "mouth": motor["mouth"] * keep, "thrust": thrust * keep,
                     "o2_mol_s": (met["metabolic_w"] / B.OXY_J_PER_MOL_O2) * keep}
        acoustic = B.call_power_w(b, motor["loudness"] * keep) * dt
        m = motor["mouth"].double() * alive0
        on = {"plant": got["plant"] > 0, "water": got["water"] > 0, "body": touched["body"],
              "item": touched["item"]}
        rubbing = rub["strokes"] >= 1
        partner = (rub["partner_a"] >= 0) | (rub["partner_b"] >= 0)
        moving = (thrust != 0) & alive0
        add = {"mouth_plant": m * on["plant"], "mouth_water": m * on["water"], "mouth_body": m * on["body"],
               "mouth_item": m * on["item"], "mouth_events": m * (on["plant"] | on["water"] | on["body"] | on["item"]),
               "mouth_sum": m, "mouth_on_plant": m * plant_here, "n_on_plant": plant_here,
               "mouth_off_plant": m * (alive0 & ~plant_here), "n_off_plant": alive0 & ~plant_here,
               "mouth_on_water": m * water_here, "n_on_water": water_here,
               "mouth_off_water": m * (alive0 & ~water_here), "n_off_water": alive0 & ~water_here,
               "plant_kg": got["plant"], "water_kg": got["water"], "bitten_kg": got["bitten"], "item_kg": got["item"],
               "rubs": rubbing, "rubs_on_partner": rubbing & partner, "rub_heat_j": rub["heat_j"],
               "embers": rub["ember"], "fires_born": rub["ignited"],
               "grips": (res["grip"]["item"] >= 0).sum(-1), "joins": res["press"]["item"] >= 0,
               "placed": res["place"]["item"] >= 0, "released": (res["release"]["item"] >= 0).sum(-1),
               "strikes": force["strokes"], "hits": force["hit_body"] >= 0, "wound_j": force["wound_j"],
               "knaps": force["knapped"], "call_j": acoustic, "loudness": motor["loudness"] * alive0,
               "distance_m": dist, "coarse_moves": coarse & moving, "moving": moving,
               "travel_beyond_far": moving & (dist > self.srules.far_cells * geom.dx), "skin_char_kg": char,
               "hand_heat_j": hand_j, "sunk_s": loco["sunk_s"], "develop_j": met["develop_j"],
               "grip_overflow": res["grip"]["overflow"], "anoxic_mol": met["anoxic_mol"] * alive0,
               "unconscious": (out_cold | ~awake0) & alive0, "treading": treading & alive0}
        if "broke" in force:
            add["broke"] = force["broke"] >= 0
        for key, v in add.items():
            st[key] += v.double().reshape(A, -1).sum(-1)
        st["transform_kg"] += fl["transformed_kg"].sum(-1)
        st["fire_heat_j"] += fl["heat_j"].double()
        st["no_item_slot"] += force["no_item_slot"].double()
        st["no_fire_slot"] += rub["no_fire_slot"].double()
        self.counters["skin_char_kg"] += char.double().sum(-1)

    def _o2_use(self, b, tis, terms) -> torch.Tensor:
        """Each body's O2 use this bout (mol/s) [A, N] for the unconsciousness gate (``PROVENANCE
        ['drowning_gate']``): the larger of the previous bout's metabolic O2 rate (thermogenesis, development and
        every cost included) and the resting use plus the bout-mean use of its locomotion."""
        rest = B.resting_power_w(b, tis) / B.OXY_J_PER_MOL_O2
        rate = rest + (terms["p_met"] * terms["share"]).float() / B.OXY_J_PER_MOL_O2
        last = self.last.get("o2_mol_s")
        if last is not None:
            rate = torch.maximum(rate, last.to(rate.dtype))
        return rate * b.alive

    def _spawn_counted(self, pool, a, comp, mass, pos, temp_k, *, L: float):
        """manipulate.spawn_items for carcass parts, counting the parts that found no item slot (they go to the
        litter: ``carcass_no_slot``)."""
        idx = mp.spawn_items(pool, a, comp, mass, pos, temp_k, ledger=self.iled, L=L)
        miss = idx < 0
        if bool(miss.any()):
            self.stats_day["carcass_no_slot"].index_add_(0, a[miss], torch.ones(int(miss.sum()), dtype=torch.float64,
                                                                                device=a.device))
        return idx

    def _founding_report(self, alive0, dead) -> None:
        """The share of each arena's founders dead in the first bout, and the arenas that lost them all
        (``PROVENANCE['founding_day']``)."""
        n0 = alive0.sum(1).double().clamp_min(1)
        share = dead.sum(1).double() / n0
        self.reports["founding"] = {"dead_first_bout_share": [round(float(v), 4) for v in share],
                                    "extinct_first_bout": [a for a in range(self.A) if float(share[a]) >= 1.0],
                                    "extinct_first_bout_share": round(float((share >= 1.0).double().mean()), 4)}

    def _fire_flux(self, fl: dict, temp0, live, pos, L: float) -> torch.Tensor:
        """The fires' heat flux at positions pos [A, N, 2] (W/m^2, float64; ``PROVENANCE['fire_heat']``): the point
        source RADIANT_FRACTION x HRR / (4 pi d^2), at most the flame's emissive power sigma T_f^4, within
        crafting.WARMTH_RADIUS_M."""
        A, N = pos.shape[:2]
        fpos, hrr = fl["fire_pos"], fl["mean_hrr_w"]
        F = fpos.shape[1]
        out = torch.zeros(A, N, dtype=torch.float64, device=pos.device)
        xy = pos[..., :2].double()
        within = float(cr.WARMTH_RADIUS_M)
        for f0 in range(0, F, 64):                                          # chunks of fire slots
            f1 = min(F, f0 + 64)
            d = pt.wrap(fpos[:, None, f0:f1, :2].double() - xy[:, :, None, :], L).norm(dim=-1)
            point = float(cr.RADIANT_FRACTION) * hrr[:, None, f0:f1].double() / (4 * math.pi * d.clamp_min(1e-6) ** 2)
            flame = K.SIGMA * temp0[:, None, f0:f1].double().clamp_min(0) ** 4
            q = torch.minimum(point, flame)
            out += torch.where(live[:, None, f0:f1] & (d <= within), q, torch.zeros_like(q)).sum(-1)
        return out

    def sense(self, scene, details: bool = False) -> torch.Tensor:
        """The bodies' input x [A, N, IN_DIM] this bout: senses3 of body.sense_view (no lineage id is in it), computed
        for the living slots only (gathered to the front of each arena and scattered back). With ``details`` the
        senses' cost bounds join the day's statistics (``PROVENANCE['sense_details']``)."""
        b = self.b
        A, N = b.shape
        alive = b.alive
        n_live = int(alive.sum(1).max()) if alive.numel() else 0
        x = torch.zeros(A, N, s3.IN_DIM, device=b.device)
        if n_live == 0:
            return x
        view = B.sense_view(b, loudness=self.last["loudness"], calls=self.last["vocal"])
        order = None
        if n_live < N:
            order = torch.argsort((~alive).to(torch.int8), dim=1, stable=True)[:, :n_live]
            view = {key: _rows(v, order) for key, v in view.items()}
        got = s3.observe(self.geom, s3.Bodies.from_mapping(view), scene, air=self.air, state=self.ps, vol=self.vol,
                         items=s3.Items.from_pool(self.items), fires=s3.Fires.from_pool(self.fires), gen=self.gen,
                         rules=self.srules, prules=self.prules, details=details)
        xs, info = got if details else (got, None)
        if order is None:
            x = xs
        else:
            x.scatter_(1, order[..., None].expand(-1, -1, s3.IN_DIM), xs)
        if info is not None:
            st = self.stats_day
            live = view["alive"].bool()
            st["sense_bodies"] += live.double().sum(1)
            for key, name in (("sense_bound", "bound"), ("sense_far_bound", "far_bound"), ("sense_beyond", "beyond")):
                st[key] += (info[name].bool() & live).double().sum(1)
            st["sense_item_beyond"] += info["item_beyond"].double()
            st["fires_culled"] += info["fires_culled"].double().sum(1)
        return x

    def _replay(self, x, brain: dict, mouth, depth) -> None:
        """The behavioural baseline (``PROVENANCE['behaviour_baseline']``): the networks of REPLAY_BODIES living bodies
        per arena (the lowest living slots) and as many founders' networks (drawn anew from the founders'
        distribution with the world's baseline seed, never selected) run one step from a zero hidden state on those
        bodies' inputs; their mouth intensities on and off plants and water join the day's statistics."""
        b = self.b
        cfg = self.config
        A, N = b.shape
        dev = b.device
        R = min(REPLAY_BODIES, N)
        order = torch.argsort((~b.alive).to(torch.int8), dim=1, stable=True)[:, :R]
        ar = torch.arange(A, device=dev)[:, None]
        live = b.alive[ar, order]
        if not bool(live.any()):
            return
        xs = x[ar, order]
        real = {key: v[ar, order] for key, v in brain.items() if key in b3.WEIGHT_GENES or key == "k"}
        h0 = torch.zeros(A, R, b.hidden.shape[-1], device=dev)
        width = int(b.genome["k"].max())
        out_r, _ = b3.think(real, b.wh_live[ar, order], h0, xs, width=width)
        S = min(BASELINE_GENOMES, R)
        gb_ = torch.Generator(device=dev).manual_seed(int(self.baseline_seed))
        base = b3.random_genome3(gb_, (A, S), s3.IN_DIM, hidden=cfg.hidden, k_range=cfg.k_founder, device=dev,
                                 dtype=b.genome["Wh"].dtype)
        pick = torch.arange(R, device=dev) % S
        base = {key: v[:, pick] for key, v in base.items() if key in b3.WEIGHT_GENES or key == "k"}
        out_b, _ = b3.think(base, base["Wh"], h0, xs, width=max(width, int(base["k"].max())))
        m_r, m_b = b3.decode(out_r)["mouth"].double(), b3.decode(out_b)["mouth"].double()
        mpos = mouth[ar, order]
        here = {"plant": pt.sample(self.geom, self.ps.plant, mpos) > 0, "water": pt.sample(self.geom, depth, mpos) > 0}
        st = self.stats_day
        for q, h in here.items():
            for o, sel in (("on", live & h), ("off", live & ~h)):
                st[f"replay_n_{o}_{q}"] += sel.double().sum(1)
                st[f"replay_real_{o}_{q}"] += (m_r * sel).sum(1)
                st[f"replay_base_{o}_{q}"] += (m_b * sel).sum(1)

    def interoception(self, light_level) -> torch.Tensor:
        """The body's own state now [A, N, N_INTERO] in senses3's code (``PROVENANCE['post_bout_state']``)."""
        b = self.b
        view = B.sense_view(b)
        rad = view["radius_m"].clamp_min(1e-9)
        g = self.air.g_m_s2.to(b.device)[:, None]
        froude = b.speed / torch.sqrt(g * 2 * rad).clamp_min(1e-9)
        x = torch.stack((view["reserve_j"] / view["reserve_cap_j"].clamp_min(1e-12),
                         view["water_kg"] / view["water_norm_kg"].clamp_min(1e-12),
                         (b.body_k - s3.T_LIFE_MID_K) / s3.T_LIFE_HALF_K, b.damage.clamp(0, 1),
                         view["gut_kg"] / view["gut_cap_kg"].clamp_min(1e-12), light_level, s3.saturate(froude)), -1)
        return torch.where(b.alive[..., None], x, torch.zeros_like(x))

    def _smells(self, met, fl) -> None:
        """Advance the smell fields by one bout (``PROVENANCE['smells']``)."""
        geom, dt = self.geom, DAY_S / self.config.bouts
        items, fires = self.items, self.fires
        A = self.A
        flesh = torch.tensor([mat.IDX[s] for s in FLESH_SPECIES], device=self.device)
        on_ground = items.alive & (items.holder < 0)
        kg = (items.comp[..., flesh].sum(-1) * items.mass * on_ground).reshape(-1)
        a_i = torch.arange(A, device=self.device).repeat_interleave(items.alive.shape[1])
        carcass = pt.deposit(geom, a_i, items.pos[..., :2].reshape(-1, 2), kg)
        co2 = torch.zeros_like(geom.elev)
        smoke = None
        if met is not None:
            b = self.b
            a_b = torch.arange(A, device=self.device).repeat_interleave(b.alive.shape[1])
            rate = pt.metabolic_co2_emission(met["metabolic_w"] * b.alive, self.prules).reshape(-1)
            co2 = co2 + pt.deposit(geom, a_b, b.pos.reshape(-1, 2), rate)
        if fl is not None and fires.alive.shape[1]:
            burn = fl["burn_kg_s"].double() * fires.alive
            a_f = torch.arange(A, device=self.device).repeat_interleave(fires.alive.shape[1])
            fpos = fires.pos[..., :2].reshape(-1, 2)
            smoke = pt.deposit(geom, a_f, fpos, pt.smoke_emission(burn, self.prules).reshape(-1))
            co2 = co2 + pt.deposit(geom, a_f, fpos, pt.fire_co2_emission(burn, self.prules).reshape(-1))
        self.vol = pt.volatile_step(self.vol, geom, dt, plant_c=self.ps.plant, carcass_kg_m2=carcass,
                                    smoke_kg_m2_s=smoke, co2_kg_m2_s=co2, rules=self.prules)

    # --------------------------------------------------------------------------------------- items and the patch
    def _ground_taken(self, ground_kg) -> None:
        """Pieces broken off the ground: wood and litter out of the patch, minerals out of the deposit."""
        if ground_kg is None or not bool((ground_kg > 0).any()):
            return
        ps = self.ps
        before = {k: getattr(ps, k).clone() for k in self.items_side}
        crust = mp.take_ground(ps, self.geom, ground_kg, self.prules)
        for k in self.items_side:
            self.items_side[k] += getattr(ps, k) - before[k]
        self.counters["crust_in_kg"] += crust
        Sd = len(mat.CRUST_SPECIES)
        per_cell = ground_kg.to(self.device, torch.float64)[..., :Sd]                  # [A, n*n, Sd] kg
        self.deposit = (self.deposit - per_cell / self.geom.cell_m2).clamp_min(0)     # the struck cells deplete
        self.counters["deposit_out_kg"] += crust[:, :Sd]

    def _to_patch(self, c_cells, n_cells, species_kg=None) -> None:
        """Carbon and nitrogen per fine cell [A, n*n] (float64) of items that left the pool to the ground into the
        litter and mineral nitrogen (through float64 residuals); the rest of their mass (``species_kg`` [A, S]):
        makeup water to the air; by element, the organic H and O to ``organic_ho_kg`` and the other elements to
        ``mineral_sink_el_kg`` (``PROVENANCE['item_residue']``)."""
        if c_cells is not None:
            self.residual["c"] += c_cells.double()
        if n_cells is not None:
            self.residual["n"] += n_cells.double()
        if species_kg is not None:
            kg = species_kg.double()
            water = kg @ B.SPECIES_WATER.to(kg.device).double()
            self.counters["items_water_air_kg"] += water
            el = kg @ mat.element_matrix(kg.device).double()                            # [A, E] kg of each element
            m_h, m_o = K.element_mass("H"), K.element_mass("O")
            m_w = 2 * m_h + m_o
            el[:, _H] -= water * 2 * m_h / m_w                                          # the water's H and O
            el[:, _O] -= water * m_o / m_w
            el[:, _C] = 0.0                                                             # carbon to the litter
            el[:, _N] = 0.0                                                             # nitrogen to the mineral pool
            el = el.clamp_min(0)
            self.counters["organic_ho_kg"] += el[:, _H] + el[:, _O]
            el[:, _H] = 0.0
            el[:, _O] = 0.0
            self.counters["mineral_sink_el_kg"] += el
        self._settle_items()

    def _settle_items(self) -> None:
        ps, geom = self.ps, self.geom
        cm2 = geom.cell_m2
        for key, fld, counter in (("c", "litter", "c_added"), ("n", "nutrient", "n_added")):
            R = self.residual[key]
            if not bool((R != 0).any()):
                continue
            f = getattr(ps, fld)
            flat = f.reshape(R.shape)
            before = flat.double()
            after = torch.where(R != 0, (before + R / cm2).clamp_min(0).float(), flat)
            moved = (after.double() - before) * cm2
            R.sub_(moved)
            setattr(ps, fld, after.view_as(f))
            tot = moved.sum(-1)
            setattr(ps, counter, getattr(ps, counter) + tot)
            self.items_side[counter] += tot

    def _rot(self) -> None:
        """A day of rot for the items (``manipulate.decay_step``): carbon and nitrogen to the cell's ground."""
        out = mp.decay_step(self.items, self.b.held, dt_days=1.0, L=float(self.config.patch_m),
                            cells=self.config.cells, ledger=self.iled)
        self._to_patch(out.get("litter_c_kg"), out.get("litter_n_kg"), out.get("decayed_kg"))

    # --------------------------------------------------------------------------------------- the global books
    def _marks(self) -> dict:
        """The cumulative counters the day's booking takes differences of."""
        dev = self.device
        A = self.A
        if not A:
            z = torch.zeros(0, dtype=torch.float64, device=dev)
            return {"c_air": z, "n_fixed": z, "o2": z, "co2": z, "items_air": torch.zeros(0, 3, dtype=torch.float64,
                                                                                          device=dev),
                    "water_air": z, "sea": z}
        L = self.b.ledger
        gas = B.gas_mol(self.b)
        return {"c_air": self.ps.c_air.clone(), "n_fixed": self.ps.n_fixed.clone(), "o2": gas["o2_mol"],
                "co2": gas["co2_mol"], "items_air": self.iled.air_kg.clone(),
                "water_air": (L["w_evap"] + L["w_resp"]).clone() + self.counters["items_water_air_kg"]
                + self.iled.air_kg[:, _GAS["H2O"]],
                "sea": self.ps.sea_water.clone()}

    def _book_day(self) -> None:
        """Book the day's exchanges of the patches, bodies and items with the global layer."""
        if self.A:
            now = self._marks()
            d = {k: now[k] - self._mark[k] for k in now}
            self._mark = now
            self._add_gas_arena(d["c_air"], d["n_fixed"], d["o2"], d["co2"], d["items_air"])
            dev = self.device
            w = self.arena_world
            cell = self.arena_cell
            C = self.globe.C
            self.carry["vapour"].view(-1).index_add_(0, w * C + cell, d["water_air"])
            per_w = torch.zeros(self.W, dtype=torch.float64, device=dev)
            self.booked["vapour_kg"] += per_w.index_add(0, w, d["water_air"])
            sea = per_w.index_add(0, w, d["sea"])
            self.carry["ocean"] -= sea
            self.booked["ocean_kg"] -= sea
        self._release()

    def _release(self) -> None:
        """Release the carries into the climate's fields as far as they resolve them (``PROVENANCE['carries']``)."""
        clim = self.clim
        dev = self.device
        R2 = (FOUR_PI * self.radius_w.to(dev) ** 2)                                     # [W] m^2
        area = self.globe.area64[None, :] * (self.radius_w.to(dev) ** 2)[:, None]       # [W, C] m^2
        cv = self.carry["vapour"]
        if bool((cv != 0).any()):
            before = clim.vapour.double()
            after = torch.where(cv != 0, (before + cv / area).clamp_min(0).float(), clim.vapour)
            moved = (after.double() - before) * area
            cv.sub_(moved)
            tot = moved.sum(-1)
            clim.vapour = after
            clim.water_external = clim.water_external - tot / R2
            self.booked["vapour_released_kg"] += tot
        co = self.carry["ocean"]
        if bool((co != 0).any()):
            before = clim.ocean
            after = before + co / R2
            moved = (after - before) * R2
            co.sub_(moved)
            clim.ocean = after
            clim.water_external = clim.water_external - moved / R2
            self.booked["ocean_released_kg"] += moved

    # --------------------------------------------------------------------------------------- ledgers
    def _local_stocks(self) -> dict:
        """Carbon and nitrogen [A] (float64 kg) held by each arena's patch, bodies, items and transits."""
        if not self.A:
            z = torch.zeros(0, dtype=torch.float64, device=self.device)
            return {"carbon": z, "nitrogen": z}
        ps, geom, b = self.ps, self.geom, self.b
        cn = self.prules.bio.cn_leaf
        st = B.stocks(b)
        em = mat.element_matrix(self.device).double()
        items = cr.ledger_mass(self.items, self.fires)
        tr = B.transit_kg(b)
        z = torch.zeros(self.A, dtype=torch.float64, device=self.device)
        t = lambda k: tr.get(k, z)                                                                   # noqa: E731
        carbon = (pt.carbon_stores(ps, geom) + st["carbon"] + items @ em[:, _C] + t("litter_in") - t("plant_out")
                  + self.residual["c"].sum(-1))
        nitrogen = (pt.nitrogen_stores(ps, geom, self.prules) + st["nitrogen"] + items @ em[:, _N]
                    + t("nutrient_in") - t("plant_out") / cn + self.residual["n"].sum(-1))
        return {"carbon": carbon, "nitrogen": nitrogen}

    def ledgers(self) -> dict:
        """Every book with its error and scale (float64) and the worst relative error ``rel``:

        climate_water [W], biosphere_carbon / biosphere_oxygen [W], air_booked [W] (gas booked = the climate's
        gas_external since the start), water_booked [W] (vapour and ocean booked = released + carried), patch_carbon /
        patch_water / patch_nitrogen [A], bodies_* [A] (body.ledger_errors), exchange_* [A] (body.exchange_errors
        against the patch's counters less the items' flows), items_mass / items_elements [A]
        (manipulate.ledger_error), local_carbon / local_nitrogen [A] (patch + bodies + items + transits against the
        flows across the arena's boundary)."""
        out: dict = {}
        if not self.W:
            out["worst"] = 0.0
            return out

        def put(name, err, scale):
            err = torch.as_tensor(err, dtype=torch.float64).reshape(-1).cpu()
            scale = torch.as_tensor(scale, dtype=torch.float64).reshape(-1).cpu().expand_as(err)
            rel = float((err.abs() / scale.abs().clamp_min(1e-30)).max()) if err.numel() else 0.0
            out[name] = {"error": err.tolist(), "scale": scale.tolist(), "rel": rel}

        clim, bio = self.clim, self.bio
        wt = cl.water_total(clim, self.globe)
        put("climate_water", cl.water_ledger(clim, self.globe), wt)
        org = bs.organic_carbon(bio) + clim.gas[:, cl.I_CO2] * M_C
        put("biosphere_carbon", bs.carbon_ledger(bio, clim), org)
        put("biosphere_oxygen", bs.oxygen_ledger(bio, clim), clim.gas[:, cl.I_O2])
        booked = self.booked["gas_mol_m2"]
        put("air_booked", ((clim.gas_external - self.gas_external0) - booked).abs().sum(-1),
            booked.abs().sum(-1).clamp_min(1e-300))
        bk = self.booked
        vap = bk["vapour_kg"] - (bk["vapour_released_kg"] + self.carry["vapour"].sum(-1))
        oce = bk["ocean_kg"] - (bk["ocean_released_kg"] + self.carry["ocean"])
        put("water_booked", vap.abs() + oce.abs(), (bk["vapour_kg"].abs() + bk["ocean_kg"].abs()).clamp_min(1e-300))
        R2 = FOUR_PI * self.radius_w.to(self.device) ** 2
        ext = (clim.water_external - self.water_external0) * R2
        released = bk["vapour_released_kg"] + bk["ocean_released_kg"]
        put("water_external", ext + released, released.abs().clamp_min(1e-300) + wt.abs() * R2 * 1e-15)
        if not self.A:
            out["worst"] = max(v["rel"] for v in out.values())
            return out
        ps, geom, b = self.ps, self.geom, self.b
        put("patch_carbon", pt.carbon_ledger(ps, geom), pt.carbon_stores(ps, geom).clamp_min(1.0))
        put("patch_water", pt.water_ledger(ps, geom), pt.water_stores(ps, geom).clamp_min(1.0))
        put("patch_nitrogen", pt.nitrogen_ledger(ps, geom, self.prules),
            pt.nitrogen_stores(ps, geom, self.prules).clamp_min(1e-3))
        errs = B.ledger_errors(b)
        for k in ("energy", "carbon", "water", "nitrogen", "inert", "salt", "oxygen", "co2"):
            put(f"bodies_{k}", errs[k], errs[f"{k}_scale"].clamp_min(1e-30))
        ex = B.exchange_errors(b, _BodySide(ps, self.items_side), geom, self.exchange_snap, self.prules)
        gross = self._exchange_gross()
        for k in ("water_in", "water_out", "sea", "carbon_in", "carbon_out", "nitrogen_in", "nitrogen_out", "salt"):
            put(f"exchange_{k}", ex[k], torch.maximum(ex[f"{k}_scale"], gross[k]).clamp_min(1e-12))
        ie = mp.ledger_error(self.iled, self.items, self.fires)
        tot = ie["total_kg"].clamp_min(1e-6)
        put("items_mass", ie["mass_kg"], tot)
        put("items_elements", ie["elements_kg"].abs().max(-1).values, tot)
        now = self._local_stocks()
        m0 = self.base["marks"]
        L = b.ledger
        em = mat.element_matrix(self.device).double()
        ga = mat.gas_element_matrix(mat.AIR_GASES, self.device).double()
        items_air_c = self.iled.air_kg @ ga[:, _C]
        flows_c = ((ps.c_air - m0["c_air"]) - L["c_co2"] - items_air_c + self.counters["crust_in_kg"] @ em[:, _C])
        base_c = self.base["local"]["carbon"]
        put("local_carbon", now["carbon"] - base_c - flows_c,
            torch.maximum(now["carbon"].abs(), base_c.abs()) + flows_c.abs())
        flows_n = (ps.n_fixed - m0["n_fixed"]) + self.counters["crust_in_kg"] @ em[:, _N]
        base_n = self.base["local"]["nitrogen"]
        put("local_nitrogen", now["nitrogen"] - base_n - flows_n,
            torch.maximum(now["nitrogen"].abs(), base_n.abs()) + flows_n.abs())
        out["worst"] = max(v["rel"] for k, v in out.items() if isinstance(v, dict))
        return out

    _GROSS = {"water_in": ("w_drunk", "w_food"), "water_out": ("w_urine", "w_faeces", "w_dead"),
              "sea": ("sea_in_kg", "sea_out_kg"), "carbon_in": ("c_food",),
              "carbon_out": ("c_faeces", "c_urine", "c_dead"), "nitrogen_in": ("n_food",),
              "nitrogen_out": ("n_urine", "n_faeces", "n_dead"), "salt": ("salt_out", "salt_dead")}

    def _exchange_gross(self) -> dict:
        """What passed through each body <-> patch book since the snapshot [A] (float64; ``PROVENANCE
        ['exchange_scale']``): the scale of the book's error, since its body side is a difference (the dead bodies'
        carbon less what their carcass items hold) whose float32 rounding is relative to these amounts."""
        L, L0 = self.b.ledger, self.exchange_snap["ledger"]
        z = torch.zeros(self.A, dtype=torch.float64, device=self.device)
        return {k: sum(((L[n] - L0[n]).double().abs() if n in L and n in L0 else z for n in names), z)
                for k, names in self._GROSS.items()}

    # --------------------------------------------------------------------------------------- summary
    def _gene_stats(self, genes: dict, alive) -> dict:
        out = {}
        specs = B.GENE_SPECS
        for name in GENE_REPORT:
            if name not in genes:
                continue
            v = genes[name].double()
            if name == "k":
                c, coord = v, "lin"
            else:
                s = specs[name]
                v = v.clamp(s.lo, s.hi)
                c, coord = (v.log10(), "log10") if s.kind == "log" else \
                    ((v / (1 - v)).log(), "logit") if s.kind == "logit" else (v, "lin")
            if c.dim() == 3:                                     # the readout gains: one value per channel
                m = alive[..., None].double()
                cnt = m.sum(1).clamp_min(1)
                mean = (c * m).sum(1) / cnt
                sd = (((c - mean[:, None]) ** 2 * m).sum(1) / cnt).sqrt()
                out[name] = {"coord": coord, "channels": list(b3.INTERO_NAMES), "mean": mean.tolist(),
                             "sd": sd.tolist()}
                continue
            m = alive.double()
            cnt = m.sum(1).clamp_min(1)
            mean = (c * m).sum(1) / cnt
            sd = (((c - mean[:, None]) ** 2 * m).sum(1) / cnt).sqrt()
            out[name] = {"coord": coord, "mean": mean.tolist(), "sd": sd.tolist()}
        return out

    def summary(self) -> dict:
        """Population, lineages (and the shadow's), births and deaths by cause (since the founding and today),
        capacity_full and whether the run is capacity_bound, gene means and spreads (with the neutral shadow's),
        behaviour statistics (today and since the founding, the mouth's intensity on and off plants and water against
        the founders' baseline), the cost bounds, the viability measures (``PROVENANCE['viability']``), the climate,
        whether fire is possible, and the patches."""
        cfg = self.config
        outcomes = self.outcomes()
        out = {"day": self.day, "seeds": self.seeds, "sampled": self.sampled, "worlds": [],
               "absent": [{k: r[k] for k in ("seed", "kind", "outcome", "note")} for r in self.absent],
               "outcomes": [{k: r[k] for k in ("seed", "outcome")} for r in outcomes]}
        if not self.W:
            out["arenas"] = []
            return out
        built = [r for r in outcomes if r["world"] is not None]
        clim, P = self.clim, self.P
        area = self.globe.area64
        ice = cl.ice_fraction(clim.T, P["rules"])
        pp = cl.partial_pressures(clim, P)
        total = cl.pressures(clim, self.globe, P)["total"]
        for w in range(self.W):
            out["worlds"].append({
                "seed": self.seeds[w], "outcome": built[w]["outcome"], "no_land": self.no_land[w],
                "mean_t_k": float((clim.T[w].double() * area).sum() / area.sum()),
                "ice_share": float((ice[w].double() * area).sum() / area.sum()),
                "p_co2_pa": float(pp[w, cl.I_CO2]), "p_o2_pa": float(pp[w, cl.I_O2]),
                "x_o2_dry": float(pp[w, cl.I_O2] / pp[w].sum().clamp_min(1e-300)),
                "pressure_pa": float(total[w]),
                **f3.fire_possible(float(pp[w, cl.I_O2] / pp[w].sum().clamp_min(1e-300)), float(total[w])),
                "water_to_air_kg": float(self.booked["vapour_kg"][w]),
                "global_plant_c_kg_m2": float(bs.organic_carbon(self.bio)[w]),
                "arenas": [int(a) for a in range(self.A) if int(self.arena_world[a]) == w]})
        if not self.A:
            out["arenas"] = []
            return out
        b = self.b
        L = b.ledger
        alive = b.alive
        pop = alive.sum(1)
        F = int(cfg.founders)
        present = torch.zeros(self.A, F + 1, dtype=torch.bool, device=self.device)
        fid = torch.where(alive, b.founder.clamp(0, F - 1), torch.full_like(b.founder, F))
        present.scatter_(1, fid, torch.ones_like(fid, dtype=torch.bool))
        lineages = present[:, :F].sum(1)
        genes = self._gene_stats(b.genome, alive)
        shadow = self._gene_stats(self.shadow.genes, self.shadow.alive) if self.shadow is not None else None
        ps, geom = self.ps, self.geom
        cm2 = geom.cell_m2
        ground_items = (self.items.alive & (self.items.holder < 0)).sum(1)

        def shares(st):
            ev = st["mouth_events"].clamp_min(1)
            return {k: (st[f"mouth_{k}"] / ev).tolist() for k in ("plant", "water", "body", "item")}

        out["arenas"] = [{"arena": a, "world": int(self.arena_world[a]), "cell": int(self.arena_cell[a])}
                         for a in range(self.A)]
        out["population"] = pop.tolist()
        out["shadow_population"] = self.shadow.alive.sum(1).tolist() if self.shadow is not None else None
        out["lineages_alive"] = lineages.tolist()
        out["births"] = L["births"].tolist()
        out["deaths"] = {c: L[f"deaths_{c}"].tolist() for c in B.CAUSES[1:]}
        out["divide_fired"] = L["divide_fired"].tolist()
        out["too_small"] = L["too_small"].tolist()
        out["capacity_full"] = L["capacity_full"].tolist()
        out["control_swaps"] = L["control_swaps"].tolist()
        out["genes"] = genes
        out["shadow_genes"] = shadow
        out.update(self._demography(lineages))
        out["behaviour_today"] = {k: v.tolist() for k, v in self.stats_day.items()}
        out["behaviour_total"] = {k: (v + self.stats_day[k]).tolist() for k, v in self.stats_total.items()}
        out["mouth_shares_today"] = shares(self.stats_day)
        out["mouth_shares_total"] = shares({k: v + self.stats_day[k] for k, v in self.stats_total.items()})
        out["mean_mass_g"] = ((B.shape(b)["m"] * alive).sum(1) / pop.clamp_min(1) * 1e3).tolist()
        out["mean_generation"] = ((b.generation.double() * alive).sum(1) / pop.clamp_min(1)).tolist()
        out["max_generation"] = torch.where(alive, b.generation, torch.zeros_like(b.generation)).amax(1).tolist()
        out["mean_body_k"] = ((b.body_k.double() * alive).sum(1) / pop.clamp_min(1)).tolist()
        out["patch"] = {"plant_kg_c": (ps.plant.double().sum((1, 2)) * cm2).tolist(),
                        "wood_kg_c": (ps.wood.double().sum((1, 2)) * cm2).tolist(),
                        "water_share": pt.water_share(ps, geom).tolist(),
                        "air_t_k": self.forcing.t_air_k.tolist() if self.forcing is not None else None}
        out["items"] = {"alive": self.items.alive.sum(1).tolist(), "on_ground": ground_items.tolist(),
                        "fires": self.fires.alive.sum(1).tolist()}
        out["mouth_association_today"] = self._association(self.stats_day)
        out["bounds_today"] = self._bounds(self.stats_day)
        out["o2_store"] = self._o2_store_report()
        out["excluded"] = list(self.reports.get("excluded", []))
        out["errors"] = list(self.reports.get("errors", []))
        out["day_base"] = "days_completed"
        for key in ("extinct_day", "arena_capacity_bound_day", "arena_items_bound_day", "arena_fires_bound_day",
                    "founding", "items_bound_day", "fires_bound_day"):
            out[key] = copy_mod.deepcopy(self.reports.get(key))
        out["books"] = {"mineral_sink_el_kg": {e: self.counters["mineral_sink_el_kg"][:, i].tolist()
                                               for i, e in enumerate(mat.ELEMENTS)},
                        "organic_ho_kg": self.counters["organic_ho_kg"].tolist(),
                        "items_water_air_kg": self.counters["items_water_air_kg"].tolist()}
        return out

    def _o2_store_report(self) -> dict | None:
        """The O2 store of the living per arena (``body.PROVENANCE['breath_hold']``): the mean lung and carrier store
        per kg lean (mol/kg) and the lung's share, at the arena's air of the last day stepped (None before it)."""
        if not self.A or getattr(self, "forcing", None) is None or self.bout_sw is None:
            return None
        b = self.b
        env = replace(B.patch_env(self.geom, self.forcing, self.bout_sw[:, 0], gravity_m_s2=self.air.g_m_s2,
                                  dt_s=DAY_S / self.config.bouts), p_o2_pa=self.p_o2_fine, p_air_pa=self.p_air_fine)
        parts = B.o2_store_parts(b, env, self.geom)
        lean = (b.mass_kg.double() * b.alive).sum(1).clamp_min(1e-30)
        lung = (parts["lung"].double() * b.alive).sum(1)
        car = (parts["carrier"].double() * b.alive).sum(1)
        return {"lung_mol_per_kg": (lung / lean).tolist(), "carrier_mol_per_kg": (car / lean).tolist(),
                "lung_share": (lung / (lung + car).clamp_min(1e-300)).tolist()}

    def _demography(self, lineages) -> dict:
        """Births and deaths today and their turnover, capacity_full as a share of the births attempted, whether
        the run is capacity_bound (``PROVENANCE['capacity_bound']``), the shadow's lineages, the share of bodies
        whose tissues ask for more than the body (brain units scaled), and the viability measures
        (``PROVENANCE['viability']``): food energy taken against energy spent today, the fat runway, arenas without
        plants."""
        b = self.b
        L = b.ledger
        alive = b.alive
        pop = alive.sum(1).double()
        d0 = self.day0 or {k: torch.zeros_like(L[k]) for k in DAY_KEYS}
        today = {k: L[k] - d0[k] for k in DAY_KEYS}
        deaths_today = sum(today[f"deaths_{c}"] for c in B.CAUSES[1:])
        attempts = L["divide_fired"]
        out = {"births_today": today["births"].tolist(), "deaths_today": deaths_today.tolist(),
               "deaths_today_by_cause": {c: today[f"deaths_{c}"].tolist() for c in B.CAUSES[1:]},
               "turnover_today": ((today["births"] + deaths_today) / 2 / pop.clamp_min(1)).tolist(),
               "capacity_full_share": (L["capacity_full"] / attempts.clamp_min(1)).tolist(),
               "capacity_full_share_today": (today["capacity_full"] / today["divide_fired"].clamp_min(1)).tolist(),
               "capacity_bound": "capacity_bound_day" in self.reports,
               "capacity_bound_day": self.reports.get("capacity_bound_day"),
               "items_bound_day": self.reports.get("items_bound_day"),
               "fires_bound_day": self.reports.get("fires_bound_day")}
        if self.shadow is not None and self.shadow.founder is not None:
            F = int(self.config.founders)
            sa = self.shadow.alive
            present = torch.zeros(self.A, F + 1, dtype=torch.bool, device=self.device)
            fid = torch.where(sa, self.shadow.founder.clamp(0, F - 1), torch.full_like(self.shadow.founder, F))
            present.scatter_(1, fid, torch.ones_like(fid, dtype=torch.bool))
            out["shadow_lineages_alive"] = present[:, :F].sum(1).tolist()
        tis = B.tissues(b)
        over = (tis["asked"] > 1) & alive
        out["tissue_overasked_share"] = (over.double().sum(1) / pop.clamp_min(1)).tolist()
        maint = B.maintenance_ref_w(b, tis) * B.arrhenius(b.body_k)
        runway = torch.where(alive & (maint > 0), b.reserve_j / maint.clamp_min(1e-30) / DAY_S, torch.zeros_like(maint))
        out["fat_runway_days_mean"] = (runway.double().sum(1) / pop.clamp_min(1)).tolist()
        out["food_j_today"] = today["e_food"].tolist()
        out["spent_j_today"] = today["e_oxidised"].tolist()
        out["food_over_spent_today"] = (today["e_food"] / today["e_oxidised"].clamp_min(1e-30)).tolist()
        plant = self.ps.plant.double().sum((1, 2)) * self.geom.cell_m2
        out["arenas_without_plants"] = [a for a in range(self.A) if float(plant[a]) <= 0.0]
        out["development_today_j"] = self.stats_day["develop_j"].tolist()
        out["brood_j_mean"] = ((b.brood_j.double() * alive).sum(1) / pop.clamp_min(1)).tolist()
        return out

    @staticmethod
    def _association(st: dict) -> dict:
        """Mean mouth intensity where the mouth is on plants (water) and where it is not, for the living, and the
        same for their own networks and the founders' baseline replayed on the same inputs
        (``PROVENANCE['behaviour_stats']``, ``['behaviour_baseline']``)."""
        out = {}
        for q in ("plant", "water"):
            row = {}
            for o in ("on", "off"):
                row[f"mouth_{o}"] = (st[f"mouth_{o}_{q}"] / st[f"n_{o}_{q}"].clamp_min(1)).tolist()
                row[f"n_{o}"] = st[f"n_{o}_{q}"].tolist()
                for w in ("real", "base"):
                    row[f"replay_{w}_{o}"] = (st[f"replay_{w}_{o}_{q}"] / st[f"replay_n_{o}_{q}"].clamp_min(1)).tolist()
            out[q] = row
        return out

    @staticmethod
    def _bounds(st: dict) -> dict:
        """The cost bounds of the day (``PROVENANCE['sense_details']``): shares of the bodies sensed on the sampled
        bout whose fine hash (bound) or far tier (far_bound) was full, whose reach passed the far tier (beyond), the
        items beyond it and the fires culled; the pools' overflows; sub-steps longer than substep_m; moving bodies
        whose bout travel passed the far range; non-finite locomotion."""
        n = st["sense_bodies"].clamp_min(1)
        moving = st["moving"].clamp_min(1)
        return {"sense_bound_share": (st["sense_bound"] / n).tolist(),
                "sense_far_bound_share": (st["sense_far_bound"] / n).tolist(),
                "sense_beyond_share": (st["sense_beyond"] / n).tolist(),
                "sense_item_beyond": st["sense_item_beyond"].tolist(), "fires_culled": st["fires_culled"].tolist(),
                "no_item_slot": st["no_item_slot"].tolist(), "no_fire_slot": st["no_fire_slot"].tolist(),
                "carcass_no_slot": st["carcass_no_slot"].tolist(), "grip_overflow": st["grip_overflow"].tolist(),
                "coarse_moves_share": (st["coarse_moves"] / moving).tolist(),
                "travel_beyond_far_share": (st["travel_beyond_far"] / moving).tolist(),
                "nonfinite": st["nonfinite"].tolist()}

    def outcomes(self) -> list:
        """One row per sampled seed, in the sampled order (``PROVENANCE['outcomes']``): {"index", "seed", "source",
        "world" (None for an absent seed), "kind", "outcome", "note", "excluded" (an exclusion by the chain's outcome,
        EXCLUDED_KINDS), "error" (the cause of an error kind), "day" (days run), "day_base", "arenas",
        "arenas_alive", "population", "no_land", "extinct_day", "arena_extinct_day", "founding",
        "capacity_bound", "capacity_bound_day", "arena_capacity_bound_day", "items_bound_day",
        "arena_items_bound_day", "fires_bound_day", "arena_fires_bound_day", "nonfinite", "spin_up_replenishment",
        "pick", "spec_failures", "input_note", "inputs_sha256"}. Kinds: ABSENT_KINDS, error, no_patches, extinct,
        alive; every day is a count of days completed (``PROVENANCE['day_base']``)."""
        rows: list = [None] * len(self.sampled)
        blank = {"world": None, "day": self.day, "day_base": "days_completed", "arenas": 0, "arenas_alive": 0,
                 "population": 0, "no_land": None, "extinct_day": None, "arena_extinct_day": [], "founding": None,
                 "capacity_bound": False, "capacity_bound_day": None, "arena_capacity_bound_day": [],
                 "items_bound_day": None, "arena_items_bound_day": [], "fires_bound_day": None,
                 "arena_fires_bound_day": [], "nonfinite": 0, "spin_up_replenishment": None, "spec_failures": [],
                 "input_note": None, "inputs_sha256": None}
        for r in self.absent:
            rows[r["index"]] = {"index": r["index"], "seed": r["seed"], "source": r.get("source", self.config.source),
                                **blank, "kind": r["kind"], "outcome": r["outcome"], "note": r["note"],
                                "excluded": r["kind"] in EXCLUDED_KINDS,
                                "error": r.get("cause") if r["kind"] == ERROR_KIND else None, "pick": r.get("pick")}
        pop = self.b.alive.sum(1).tolist() if self.W and self.A else []
        ext = self.reports.get("extinct_day") or [None] * self.A
        cap = self.reports.get("arena_capacity_bound_day") or [None] * self.A
        itb = self.reports.get("arena_items_bound_day") or [None] * self.A
        frb = self.reports.get("arena_fires_bound_day") or [None] * self.A
        found = self.reports.get("founding") or {}
        share = found.get("dead_first_bout_share") or []
        nonfinite = ((self.stats_total["nonfinite"] + self.stats_day["nonfinite"]).tolist()
                     if self.W and self.A else [])
        repl = self.reports.get("spin_up_replenishment") or []
        free = [i for i, r in enumerate(rows) if r is None]

        def first(v):
            v = [x for x in v if x is not None]
            return min(v) if v else None

        for w, i in enumerate(free[:self.W]):
            arenas = self._arenas_of(w)
            alive = [a for a in arenas if pop[a] > 0]
            e = [ext[a] for a in arenas]
            if not arenas:
                kind, outcome, note = "no_patches", "no patches", "the config has patches = 0"
            elif not alive and all(d is not None for d in e):
                kind, outcome = "extinct", f"extinct at day {max(e)}"
                note = f"every arena empty; arenas extinct at days {e} (days completed)"
            else:
                kind, outcome = "alive", f"alive at day {self.day}"
                note = f"{len(alive)} of {len(arenas)} arenas alive"
            if self.no_land[w]:
                note += "; no land: its patches are sea"
            c = [cap[a] for a in arenas]
            rows[i] = {"index": i, "seed": self.seeds[w], "source": self.config.source, "world": w, "kind": kind,
                       "outcome": outcome, "note": note, "excluded": False, "error": None, "day": self.day,
                       "day_base": "days_completed", "arenas": len(arenas), "arenas_alive": len(alive),
                       "population": int(sum(pop[a] for a in arenas)), "no_land": self.no_land[w],
                       "extinct_day": max(e) if kind == "extinct" else None, "arena_extinct_day": e,
                       "founding": None if not share else {
                           "dead_first_bout_share": [share[a] for a in arenas],
                           "arenas_extinct_first_bout": sum(1 for a in arenas if share[a] >= 1.0)},
                       "capacity_bound": any(v is not None for v in c), "capacity_bound_day": first(c),
                       "arena_capacity_bound_day": c, "items_bound_day": first([itb[a] for a in arenas]),
                       "arena_items_bound_day": [itb[a] for a in arenas],
                       "fires_bound_day": first([frb[a] for a in arenas]),
                       "arena_fires_bound_day": [frb[a] for a in arenas],
                       "nonfinite": int(sum(nonfinite[a] for a in arenas)) if nonfinite else 0,
                       "spin_up_replenishment": repl[w] if w < len(repl) else None,
                       "pick": pick_record(self.inputs[w]),
                       "spec_failures": list(self.spec_failures.get(self.seeds[w], [])),
                       "input_note": self.input_notes[w], "inputs_sha256": inputs_digest(self.inputs[w])}
        return rows

    def bound_report(self) -> dict:
        """brain3.bound_report of the living (the share at each gene bound, k at its top, weights at the clip; one
        host sync per entry: for analysis, not every day)."""
        if not self.A:
            return {}
        b = self.b
        return b3.bound_report(b.genome, b.alive, specs=B.GENE_SPECS, hidden=self.config.hidden, wh_live=b.wh_live)

    # --------------------------------------------------------------------------------------- viewer data
    def static3(self, w: int = 0) -> dict:
        """Static viewer data of world w (schema VIEW_SCHEMA):

        {"schema", "seed", "planet": {radius_m, gravity_m_s2, surface_pressure_pa, tidally_locked, year_s,
        star_teff_k, t_surface_k, ocean_fraction, frozen_mean}, "globe": {"G", "centers" [[x, y, z]] (4 dp),
        "elevation_m" [C ints], "land" [C 0/1], "sea_level_m"}, "patches": [{"index", "cell", "center", "L", "n",
        "elevation_m" [n*n ints], "pond" [n*n 0/1]}], "species": materials.SPECIES, "item_classes":
        materials.CLASSES}. Frames (:meth:`frame3`) index items by ``item_classes``."""
        s = self.specs[w]
        row = f3.summary_row(s)
        planet = {"radius_m": float(s.radius_m), "gravity_m_s2": float(s.gravity_m_s2),
                  "surface_pressure_pa": float(s.surface_pressure_pa), "tidally_locked": bool(s.tidally_locked),
                  "year_s": float(s.year_s), "star_teff_k": float(s.star_teff_k), "t_surface_k": float(row["t_s_k"]),
                  "ocean_fraction": float(row["ocean_fraction"]), "frozen_mean": bool(row["frozen_mean"])}
        t = self.terrain0
        g0 = self.globe0
        globe = {"G": int(self.config.G), "centers": [[round(float(v), 4) for v in c] for c in g0.centers.tolist()],
                 "elevation_m": [int(round(v)) for v in t["elevation_m"][w].tolist()],
                 "land": [int(v) for v in t["land"][w].tolist()], "sea_level_m": float(t["sea_level_m"][w])}
        patches = []
        for p, a in enumerate(self._arenas_of(w)):
            geom = self.geom
            pond = (self.ps.pond[a] > 0) | geom.sea[a]
            patches.append({"index": p, "arena": a, "cell": int(self.arena_cell[a]),
                            "center": [round(float(v), 4) for v in g0.centers[int(self.arena_cell[a])].tolist()],
                            "L": float(geom.L), "n": int(geom.n),
                            "elevation_m": [int(round(v)) for v in geom.elev[a].reshape(-1).tolist()],
                            "pond": [int(v) for v in pond.reshape(-1).tolist()]})
        return {"schema": VIEW_SCHEMA, "seed": int(self.seeds[w]), "planet": planet, "globe": globe,
                "patches": patches, "species": list(mat.SPECIES), "item_classes": list(mat.CLASSES),
                "no_land": self.no_land[w]}

    def _arenas_of(self, w: int) -> list:
        return [a for a, x in enumerate(self.arena_world.tolist()) if x == w]

    def sun_vector(self, w: int, day: int | None = None, bout: int | None = None) -> list:
        """Unit vector toward the star in planet coordinates at the middle of a bout."""
        day = int(self.clim.day if day is None else day)
        bout = int(self.bout if bout is None else bout)
        dec = cl.declination(self.P, day).double()[w].reshape(1)
        lock = self.P["locked"][w].reshape(1)
        sd = pt.solar_days([self.specs[w]])[0]
        t = (day + (bout + 0.5) / self.config.bouts) * DAY_S
        frac = torch.tensor([0.0 if not math.isfinite(sd) else (t / sd) % 1.0], dtype=torch.float64)
        v = pt._sun_vectors(dec.cpu(), lock.cpu(), frac)[0]
        return [round(float(x), 5) for x in v]

    def frame3(self, w: int = 0, *, fields: bool = False, global_fields: bool = False) -> dict:
        """One viewer frame of world w (schema VIEW_SCHEMA): {"day", "bout", "sun", optional "global": {"T_k" [C
        ints], "ice" [C 0/1]}, "patches": [{"index", "bodies": [[uid, x, y, heading, mass_g, lineage, k_active, mouth,
        thrust, loud, vocal_argmax, damage]], optional "plants" [n*n 0-255], optional "water" [n*n 0-255], "items":
        [[x, y, class_index, mass_kg]] (at most 1,500), "fires": [[x, y, temp_k]]}]}."""
        fr = {"day": int(self.day), "bout": int(self.bout), "sun": self.sun_vector(w)}
        if global_fields:
            ice = (cl.ice_fraction(self.clim.T[w], self.P["rules"]) >= 0.5) | \
                (self.clim.snow[w] >= self.P["rules"].snow_mask_kg_m2)
            fr["global"] = {"T_k": [int(round(v)) for v in self.clim.T[w].tolist()],
                            "ice": [int(v) for v in ice.tolist()]}
        pats = []
        cls_of = torch.tensor([mat.CLASSES.index(mat.CLASS[s]) for s in mat.SPECIES])
        for p, a in enumerate(self._arenas_of(w)):
            b = self.b
            al = b.alive[a]
            idx = al.nonzero(as_tuple=True)[0]
            sh_m = B.shape(b)["m"][a]
            cols = torch.stack((b.uid[a].double(), b.pos[a, :, 0].double(), b.pos[a, :, 1].double(),
                                b.heading[a].double(), sh_m.double() * 1e3, b.founder[a].double(),
                                b.genome["k"][a].double(), self.last["mouth"][a].double(),
                                self.last["thrust"][a].double(), self.last["loudness"][a].double(),
                                self.last["vocal"][a].argmax(-1).double(), b.damage[a].double()), -1)[idx].cpu()
            nd = (0, 2, 2, 3, 4, 0, 0, 3, 3, 3, 0, 3)
            rows = [[int(v) if d == 0 else round(float(v), d) for v, d in zip(r, nd)] for r in cols.tolist()]
            it = self.items
            ground = (it.alive[a] & (it.holder[a] < 0)).nonzero(as_tuple=True)[0]
            if ground.numel() > 1500:
                order = torch.argsort(it.mass[a, ground], descending=True, stable=True)[:1500]
                ground = ground[order]
            dom = it.comp[a, ground].argmax(-1).cpu()
            items = [[round(float(x), 2), round(float(y), 2), int(cls_of[d]), round(float(m), 6)]
                     for (x, y), d, m in zip(it.pos[a, ground, :2].tolist(), dom.tolist(), it.mass[a, ground].tolist())]
            fi = self.fires
            on = fi.alive[a].nonzero(as_tuple=True)[0]
            fires = [[round(float(x), 2), round(float(y), 2), int(round(t))]
                     for (x, y), t in zip(fi.pos[a, on, :2].tolist(), fi.temp_k[a, on].tolist())]
            pat = {"index": p, "arena": a, "bodies": rows, "items": items, "fires": fires}
            if fields:
                cover = pt.fapar(self.ps.plant[a], self.prules)
                pat["plants"] = [int(v) for v in (cover * 255).round().clamp(0, 255).reshape(-1).tolist()]
                pond = self.ps.pond[a].double()
                water = torch.where(self.geom.sea[a], torch.ones_like(pond), pond / (pond + 100.0))
                pat["water"] = [int(v) for v in (water * 255).round().clamp(0, 255).reshape(-1).tolist()]
            pats.append(pat)
        fr["patches"] = pats
        return fr

    # --------------------------------------------------------------------------------------- state
    def state_dict(self, copy: bool = True) -> dict:
        """Everything a resume needs (tensors copied unless ``copy`` is False; the set-up is not re-run)."""
        cp = _clone if copy else _ref

        def t(x):
            return x.clone() if (copy and torch.is_tensor(x)) else x

        if not self.W:
            return {"version": STATE_VERSION, "config": self.config.to_dict(), "seeds": [],
                    "sampled": list(self.sampled), "absent": [dict(r) for r in self.absent], "rng_seed": self.rng_seed,
                    "day": self.day, "reports": copy_mod.deepcopy(self.reports) if copy else self.reports,
                    "without_worlds": True}
        d = {"version": STATE_VERSION, "config": self.config.to_dict(), "seeds": list(self.seeds),
             "sampled": list(self.sampled), "absent": [dict(r) for r in self.absent],
             "inputs": self.inputs, "input_notes": self.input_notes, "rng_seed": self.rng_seed,
             "setup_seeds": list(getattr(self, "setup_seeds", [])),
             "terrain0": {k: t(v) for k, v in self.terrain0.items()},
             "A": self.A, "arena_world": t(self.arena_world), "arena_cell": t(self.arena_cell), "no_land": self.no_land,
             "geom": None if self.geom is None else (self.geom.state_dict() if copy else
                                                      {f.name: getattr(self.geom, f.name)
                                                       for f in fields(self.geom)}),
             "deposit": t(self.deposit), "clim": self.clim.state_dict(), "bio": self.bio.state_dict(),
             "ps": None if self.ps is None else self.ps.state_dict(),
             "vol": None if self.vol is None else self.vol.state_dict(),
             "items": itm.state_dict(self.items), "fires": itm.state_dict(self.fires), "iled": self.iled.state_dict(),
             "bodies": self.b.state_dict() if copy else self._bodies_ref(),
             "shadow": None if self.shadow is None else self.shadow.state_dict(),
             "gen": self.gen.get_state(), "shadow_gen": self.shadow_gen.get_state(),
             "day": self.day, "bout": self.bout, "carry": cp(self.carry), "residual": cp(self.residual),
             "items_side": cp(self.items_side), "counters": cp(self.counters), "booked": cp(self.booked),
             "gas_external0": t(self.gas_external0), "water_external0": t(self.water_external0),
             "stats_day": cp(self.stats_day), "stats_total": cp(self.stats_total), "last": cp(self.last),
             "mark": cp(self._mark), "base": cp(self.base), "day0": cp(self.day0),
             "baseline_seed": int(self.baseline_seed),
             "exchange_snap": None if self.exchange_snap is None else cp(self.exchange_snap),
             "reports": copy_mod.deepcopy(self.reports) if copy else self.reports,
             "spec_failures": copy_mod.deepcopy(self.spec_failures) if copy else self.spec_failures}
        return d

    def _bodies_ref(self) -> dict:
        b = self.b
        out = {name: getattr(b, name) for name in B.STATE_FIELDS}
        out.update(L=float(b.L), next_uid=b.next_uid, genome=dict(b.genome), wh_live=b.wh_live, hidden=b.hidden,
                   ledger=dict(b.ledger), transit=None if b.transit is None else dict(b.transit))
        return out

    @classmethod
    def from_state(cls, d: dict, device="cpu", copy: bool = True, expect_rules: str | None = None) -> "World3":
        """A world from :meth:`state_dict` (on ``device``); continues exactly where the saved one stopped. ``copy``
        False takes the tensors as they are (a load that owns them makes no second copy of the state); with
        ``expect_rules`` a state saved under another rules hash is refused."""
        if d.get("version") != STATE_VERSION:
            raise ValueError(f"state version {d.get('version')!r}, expected {STATE_VERSION!r}")
        if expect_rules is not None and d.get("rules_sha256") != expect_rules:
            raise ValueError(f"the checkpoint was saved under rules {str(d.get('rules_sha256'))[:12]}, not "
                             f"{expect_rules[:12]}: a rule change is a new experiment")
        mv = partial(_host, copy=copy)
        self = cls.__new__(cls)
        dev = torch.device(device)
        self.device = dev
        self._log = lambda msg: None
        self.config = V3Config.from_dict(d["config"])
        self.seeds = list(d["seeds"])
        self.sampled = list(d.get("sampled", self.seeds))
        self.absent = [dict(r) for r in d.get("absent", [])]
        self.W = len(self.seeds)
        if d.get("without_worlds"):
            self.rng_seed = d["rng_seed"]
            self.reports = copy_mod.deepcopy(d["reports"])
            self.inputs, self.input_notes = [], []
            self._init_without_worlds()
            self.day = int(d["day"])
            return self
        self.inputs = d["inputs"]
        self.input_notes = d["input_notes"]
        self.rng_seed = d["rng_seed"]
        self.setup_seeds = list(d.get("setup_seeds", []))
        self._build_specs()
        self.globe0 = gb.Globe(self.config.G, torch.device("cpu"))
        terrain0 = {k: v.cpu().clone() for k, v in d["terrain0"].items()}
        self._build_static(terrain0)
        self.A = int(d["A"])
        self.arena_world = d["arena_world"].to(dev).clone()
        self.arena_cell = d["arena_cell"].to(dev).clone()
        self.no_land = list(d["no_land"])
        self.geom = None if d["geom"] is None else pt.PatchGeometry.from_state(mv(d["geom"], dev))
        self.deposit = None if d["deposit"] is None else d["deposit"].to(dev).clone()
        if self.deposit is not None and self.deposit.dim() == 2:          # a checkpoint with one column per arena
            nn = int(self.config.cells) ** 2
            self.deposit = self.deposit[:, None, :].expand(self.deposit.shape[0], nn, self.deposit.shape[1]).clone()
        self.clim = cl.ClimateState.from_state(mv(d["clim"], dev))
        self.bio = bs.BioState.from_state(mv(d["bio"], dev))
        self.ps = None if d["ps"] is None else pt.PatchState.from_state(mv(d["ps"], dev))
        self.vol = None if d["vol"] is None else pt.Volatiles.from_state(mv(d["vol"], dev))
        self.items = itm.from_state(mv(d["items"], dev))
        self.fires = itm.from_state(mv(d["fires"], dev))
        self.iled = mp.ItemLedger.from_state(mv(d["iled"], dev))
        self.b = B.Bodies.from_state(mv(d["bodies"], dev), copy=copy, strict=True)
        self.shadow = None if d["shadow"] is None else B.Shadow.from_state(mv(d["shadow"], dev))
        self.gen = torch.Generator(device=dev)
        self.gen.set_state(d["gen"])
        self.shadow_gen = torch.Generator(device=dev)
        self.shadow_gen.set_state(d["shadow_gen"])
        self.day, self.bout = int(d["day"]), int(d["bout"])
        for name in ("carry", "residual", "items_side", "counters", "booked", "stats_day", "stats_total", "last",
                     "base"):
            setattr(self, name, mv(d[name], dev))
        # a checkpoint from before the review's books and statistics: the new entries start at zero
        z = lambda *s: torch.zeros(*s, dtype=torch.float64, device=dev)            # noqa: E731
        self.counters.setdefault("mineral_sink_el_kg", z(self.A, len(mat.ELEMENTS)))
        self.counters.setdefault("organic_ho_kg", z(self.A))
        for st in (self.stats_day, self.stats_total):
            for key in BEHAVIOUR_KEYS:
                st.setdefault(key, z(self.A))
        self._mark = mv(d["mark"], dev)
        self.day0 = mv(d.get("day0") or {}, dev)
        self.baseline_seed = int(d.get("baseline_seed", 0))
        self.gas_external0 = d["gas_external0"].to(dev).clone()
        self.water_external0 = d["water_external0"].to(dev).clone()
        self.exchange_snap = None if d["exchange_snap"] is None else mv(d["exchange_snap"], dev)
        self.reports = copy_mod.deepcopy(d["reports"])           # a copy never shares its books
        self.spec_failures = copy_mod.deepcopy(d["spec_failures"])
        self.cdiag = self.forcing = self.bout_sw = None
        return self

    def save(self, path) -> None:
        """Write a checkpoint (atomic replace) with the rules identity it was made under (``rules_sha256``)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        d = self.state_dict(copy=False)
        d["rules_sha256"] = rules_identity(self.config)["sha256"]
        torch.save(d, tmp)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path, device="cpu", expect_rules: str | None = None) -> "World3":
        """A world from a checkpoint, holding its state once (no copies: ``from_state(copy=False)``). With
        ``expect_rules`` a checkpoint of other rules is refused; without it, one that differs from the rules in
        effect gives a warning."""
        d = torch.load(path, map_location="cpu", weights_only=False)
        if expect_rules is None and d.get("version") == STATE_VERSION and d.get("config") is not None:
            now = rules_identity(V3Config.from_dict(d["config"]))["sha256"]
            if d.get("rules_sha256") != now:
                warnings.warn(f"{path}: saved under rules {str(d.get('rules_sha256'))[:12]}, the rules in effect are "
                              f"{now[:12]}")
        return cls.from_state(d, device, copy=False, expect_rules=expect_rules)

    def state_hash(self) -> str:
        """SHA-256 of the whole dynamic state (bodies, patches, items, climate, generators, books)."""
        h = hashlib.sha256(b"World3")
        if not self.W:
            _hash_update(h, "without_worlds", {"sampled": self.sampled, "day": self.day})
            return h.hexdigest()
        h.update(B.state_hash(self.b).encode())
        for name, v in (("clim", self.clim.state_dict()), ("bio", self.bio.state_dict()),
                        ("ps", None if self.ps is None else self.ps.state_dict()),
                        ("vol", None if self.vol is None else self.vol.state_dict()),
                        ("items", itm.state_dict(self.items)), ("fires", itm.state_dict(self.fires)),
                        ("iled", self.iled.state_dict()),
                        ("shadow", None if self.shadow is None else self.shadow.state_dict()),
                        ("gen", self.gen.get_state()), ("shadow_gen", self.shadow_gen.get_state()),
                        ("carry", self.carry), ("residual", self.residual), ("counters", self.counters),
                        ("booked", self.booked), ("stats", self.stats_day), ("last", self.last),
                        ("deposit", self.deposit), ("day0", self.day0), ("day", self.day)):
            _hash_update(h, name, v)
        return h.hexdigest()

    def memory_bytes(self) -> int:
        """Bytes held by the world's state tensors."""
        return _tensor_bytes(self.state_dict(copy=False))

    # --------------------------------------------------------------------------------------- text
    def describe(self) -> str:
        lines = [f"seed {r['seed']}: {r['outcome']} ({r['note']})" for r in self.absent]
        for w, s in enumerate(self.specs):
            lines.append(f3.describe3(s))
            lines.append(patch_lines(self, w))
        return "\n".join(lines)


def patch_lines(world: World3, w: int) -> str:
    """The patches of world w, one line each (cell, latitude, longitude, elevation above the sea, detail rms)."""
    out = [f"patches of seed {world.seeds[w]}" + (" (no land: its patches are sea):" if world.no_land[w] else ":")]
    for p, a in enumerate(world._arenas_of(w)):
        c = int(world.arena_cell[a])
        x, y, z = world.globe0.centers[c].tolist()
        lat, lon = math.degrees(math.asin(max(-1.0, min(1.0, z)))), math.degrees(math.atan2(y, x))
        h = float(world.terrain0["elevation_m"][w, c] - world.terrain0["sea_level_m"][w])
        line = f"  {p}: cell {c}, lat {lat:+.1f}, lon {lon:+.1f}, {h:.0f} m above the sea"
        if world.geom is not None:
            line += (f", detail rms {float(world.geom.detail_rms_m[a]):.1f} m, "
                     f"sea {float(world.geom.sea[a].float().mean()) * 100:.1f} % of the patch")
        out.append(line)
    return "\n".join(out)


def choose_patches(config: V3Config, seeds, inputs=None) -> dict:
    """The set-up's draws 1-2 without the spin-ups (for ``describe``): specs, terrain and the patch cells World3
    would get for these seeds and config (each world from its own generator; a world without land gets sea patches).
    Returns {specs, terrain, globe, cells: [[cell]], no_land, seeds (those with a planet), absent ([{seed, kind,
    outcome, note}] of the seeds without one)}; lists are per world with a planet."""
    sampled = [int(s) for s in seeds]
    got = [(x, None) for x in inputs] if inputs is not None else [planet_inputs(s, config) for s in sampled]
    absent = [{"seed": s, **note} for s, (x, note) in zip(sampled, got) if x is None]
    built = [(s, x) for s, (x, _) in zip(sampled, got) if x is not None]
    seeds = [s for s, _ in built]
    out = {"seeds": seeds, "absent": absent, "specs": [], "terrain": None, "globe": None, "cells": [], "no_land": []}
    if not built:
        return out
    specs = [f3.build_planet3(x, s) for s, x in built]
    gens = [torch.Generator().manual_seed(setup_seed(s, config.rng_seed)) for s in seeds]
    globe = gb.Globe(config.G, torch.device("cpu"))
    terrain = make_terrain_per_world(globe, gens, specs)
    land = terrain["land"]
    no_land = [not bool(v) for v in land.any(-1)]
    cells = [[] for _ in seeds]
    if config.patches > 0:
        for w in range(len(seeds)):
            got_cells = pt.choose_patch_cells(globe, land[w:w + 1], globe.area64, gens[w], config.patches)
            cells[w] = [int(c) for c in got_cells[0].tolist()]
    out.update(specs=specs, terrain=terrain, globe=globe, cells=cells, no_land=no_land)
    return out
