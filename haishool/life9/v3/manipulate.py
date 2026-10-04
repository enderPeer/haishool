"""Contact physics of the body's abilities on things, and fire, heat, transforms and decay of items in the habitat
patches of life9 v3 (PLANET-V3-SPEC section 8).

Nothing here names a purpose. A body has two grips and a mouth; what it does is set by its motor outputs
(``brain3.decode``: grip_left, grip_right, release, force, rub, press, place) and what happens is set by physics:

* :func:`grip` closes a grip on the nearest loose item in contact with the mouth and lifts it when the muscles can
  (weight m g at most the force peak_w / ``LIFT_SPEED_M_S``, less what the grips already carry); else it fails.
* :func:`release` opens both grips; :func:`place` opens one (the left first). The item lands at the mouth, so in a
  fire's bed when the mouth is within ``crafting.FIRE_RADIUS_M`` of a fire's centre.
* :func:`force` strikes, for its share of the bout, with what the first filled grip holds (or a bare limb) at what is
  in contact with the mouth: the nearest body or loose item, else the item in the other grip, else the ground. Each
  stroke is constant peak power over the limb (``STROKE_RULE``) with every held item a rod (``ROD_RULE``); strokes
  follow one another at most as fast as the stroke cycle and the sustained power allow, while the target stays in
  contact (``CONTACT_TIME_RULE``). On a brittle item softer than the striker (version 2's knapping rule) flakes come
  off and the edge sharpens (``KNAP_STROKES_RULE``); on a body the strokes wound (``crafting.strike_damage``,
  ``WOUND_J_PER_KG``); on the ground they break pieces off the plants, the litter or the mineral ground
  (``BREAK_RULE``): the only source of loose material besides carcasses.
* :func:`rub` slides the held item against the other held item, or else against what is in contact (an item, or the
  ground); with no held item a bare limb rubs the item in contact. A share of the work becomes heat at the contact
  spot, whose temperature follows from conduction into the partner the spot stays on and the partner it slides over
  (``FRICTION_RULE``), capped at the partners' decomposition or melting (``CAP_RULE``). A fuel partner above its
  ignition temperature in enough O2 makes an ember, and an ember on tinder becomes a new fire (``IGNITION_RULE``).
  No fire is lit any other way.
* :func:`press` joins the two held items when a separate binding item (fibre, resin, hide: ``crafting.BINDS``) is in
  contact with the mouth (``PRESS_RULE``): the left grip's item is the handle end, the right grip's the far end.
* :func:`heat_step` runs version 2's fire and item physics (``crafting.fire_step``: burning, heating, transforms by
  temperature and atmosphere, charring) for a bout on patch coordinates, with held items in a bed in the fire and the
  fires' radiation on every other item; :func:`decay_step` rots organic items (``crafting.decay``). Cooking is flesh
  that has been above ``crafting.COOK_K`` (:func:`denatured`, :func:`digestible_gain`), nothing more.
* :func:`bout` runs the primitives of one bout in the order grip, press, rub, force, then the grips that open,
  with the discrete ones (grips, press, place, release) as continuous-time events at rates set by their intensities
  (``EVENT_RULE``: the same per-second rates for any bout length) and the bout's time shared among press, rub,
  force and locomotion (:func:`time_split`, ``SHARE_RULE``).

Shapes and units. A arenas, N body slots, I item slots, F fire slots. Pools are version 2's ``items.ItemPool``
[A, I] and ``items.FirePool`` [A, F]; ``held`` [A, N, 2] long holds the item in each grip (-1 empty), and an item's
``holder`` is ``n * 2 + k`` (version 2's inventory code). Positions are patch metres, x east and y north in
[0, L) with a periodic boundary; the pools' ``pos`` is (x, y, z) with z the height above the ground (0 for loose
items and fires; held items carry the mouth's z when one is given). An item's ``handle_len_m`` is its extent from the
grip to its tip: a plain item's own length (0 for a compact one), an assembly's lever. Body quantities are plain
tensors from ``body.py``: ``alive`` [A, N], ``pos`` (body centre) and ``mouth_pos`` [A, N, 2 or 3],
``body_mass_kg`` [A, N], ``peak_w`` (``body.muscle_peak_w``) and ``sustained_w`` (``SUSTAINED_SHARE`` x aerobic x
peak) [A, N] (``POWER_RULE``), optionally ``vel`` [A, N, 2], ``reach_m``, ``limb_m``. Scalars or [A] values
broadcast. Work returned (``work_j``) is mechanical; ``body.py`` charges its metabolic cost. ``time_s`` is the part
of the bout each primitive used, and ``bout`` returns the locomotion share left for ``body.move``.

Contact. What touches the mouth is found by :func:`contact`: a spatial hash on a grid of cells at least as wide as
the largest contact distance (one host sync), 3 x 3 cells per query, nearest cells first, ``CONTACT_SCAN``
candidates per pass and a full second pass for the queries that had more. A body's mouth reaches one body radius
(``REACH_RULE``); a body or item is in contact when its surface (an item's half extent) is within that reach. When
several bodies claim the same item in one primitive, the one exerting the most gets it; ties go to the lowest slot.

Ledgers (float64, per arena, :class:`ItemLedger`). Items plus fire beds hold species mass. Flows in: items spawned
from outside (:func:`spawn_items`: carcasses; pieces broken off the ground by :func:`force`). Flows out:
:func:`remove_mass` (taken by mouths), charred tissue and residue with no slot (``to_soil_kg``), rot
(``decayed_kg``). Gases: ``air_kg`` [A, 3] (``materials.AIR_GASES``, O2 negative = taken from the air). Every
primitive conserves mass and elements; the residuals are :func:`ledger_error`. Per fine cell, :func:`heat_step`,
:func:`rub` and :func:`decay_step` return the carbon and nitrogen that went to the soil, for ``patch.add_litter``;
:func:`take_ground` books the ground's loss to the patch. Rubbing reports where its energy went (``energy``).

Every number carries a ``PROVENANCE`` entry; :func:`provenance` adds version 2's entries this module relies on. One
generator is used (:func:`bout`'s event draws, and :func:`force`'s head strokes and ground draw), so CPU runs are
exactly reproducible. No Python loop runs over bodies, items or cells: the loops are over the two grips, the two rubbing
partners, the contact scan's chunks of candidates, chunks of fire slots and ``fire_step``'s substeps.

Statistics of behaviour should use the energies (blow_j x strokes, wound_j, heat_j), not the partner indices alone:
an index is reported only where at least one stroke happened.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields

import torch

from haishool.life9.planet import crafting as cr
from haishool.life9.planet import items as it
from haishool.life9.planet import materials as mat

from . import brain3
from . import patch as pt

S = mat.S
GRIPS = 2
E = len(mat.ELEMENTS)
_C, _N = mat.ELEMENTS.index("C"), mat.ELEMENTS.index("N")
#: the 3 x 3 block of the contact hash, own cell first (as ``patch.BLOCK9``)
BLOCK9 = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
MAX_HASH_CELLS = 1 << 24
WOOD, FIBER = mat.IDX["wood"], mat.IDX["plant_fiber"]
NEVER = float(mat.NEVER_K)

# ============================================================================== constants
TISSUE_DENSITY_KG_M3 = 1050.0
LIFT_SPEED_M_S = 1.0
KNAP_J_PER_KG = 500.0
MAX_FLAKE_SHARE = 0.5
WOUND_J_PER_KG = 0.7
WOUND_BODY_J_PER_KG = 4.0
HEAD_SHARE = 0.09
LIMB_SUSTAINED_SHARE = 0.25
PRESS_S = 600.0
GRASP_S = 1.0
RUB_SPEED_M_S = 0.5
MOVING_FLASH = 0.31
PECLET_BRIDGE = (0.25 / MOVING_FLASH) ** 2
PYROLYSIS_K = 673.0
EMBER_KG = 0.002
TINDER_MIN_SHARE = 0.5
TINDER_SPECIES = ("plant_fiber",)
PYROLYSING_SPECIES = ("wood", "plant_fiber")
GROUND_K_W_MK = 1.0
GROUND_C_J_M3K = 2.2e6
GROUND_K_DRY_W_MK = 0.28
GROUND_K_WET_W_MK = 1.89
GROUND_C_DRY_J_M3K = 1.35e6
GROUND_C_WET_J_M3K = 3.03e6
SURFACE_LAYER_M = 0.05
CONTACT_V_FLOOR_M_S = 1e-6
CONTACT_SCAN = 64
SCAN_CHUNK = 16

R_, D_, N_, RN, DN = "reference", "derived", "new_rule", "reference+new_rule", "derived+new_rule"
PROVENANCE = {
    "TISSUE_DENSITY_KG_M3": (R_, "PLANET-V3-SPEC 3: a body is an ellipsoid of its mass at tissue density 1,050 kg/m3 "
                                 "(soft tissue about 1.05 g/cm3)"),
    "REACH_RULE": (N_, "default: a mouth touches what lies within one body radius of it, the radius of the sphere of "
                       "the body's mass at TISSUE_DENSITY_KG_M3 (a mouth, neck or limb reaches about its body's size); "
                       "a body or an item is in contact when its surface (an item's half extent) is within that "
                       "reach; body.py passes its own reach_m when the body plan carries it"),
    "LIMB_RULE": (N_, "default: a striking or rubbing limb is a uniform rod (m/3 at the hand) as long as the diameter "
                      "of the sphere of its body's mass (0.51 m at 70 kg, against version 2's human arm of 0.6 m, "
                      "crafting.ARM_LENGTH_M) with the mass crafting.LIMB_FRACTION of the body (a human hand and "
                      "forearm); body.py passes limb_m when the body plan carries the limb as a gene"),
    "POWER_RULE": (N_, "two powers come from body.py: peak_w (body.muscle_peak_w) for the brief acts, the grip's lift "
                       "and each stroke of force; sustained_w (body.py's SUSTAINED_SHARE x aerobic x peak) for the "
                       "time-integrated ones, of which a limb draws LIMB_SUSTAINED_SHARE: rubbing runs at the limb's "
                       "sustained power, a press holds it for PRESS_S, and the strokes of force are spaced so that "
                       "their mean power is at most it"),
    "LIMB_SUSTAINED_SHARE": (RN, "default: a limb's sustained work draws a quarter of the body's sustained power: arm "
                                 "cranking sustains 50-100 W (crafting.ARM_POWER_W) against 200-300 W of whole-body "
                                 "(leg) work in adults (estimate from the two); body.py passes limb_share when the "
                                 "body plan carries it"),
    "SHARE_RULE": (N_, "the bout's time is shared: a press that happens takes PRESS_S; the rest is split among rub, "
                       "force and locomotion (|thrust|, body.move) in proportion to their intensities when together "
                       "they ask for more than is left, else each gets its intensity; bout returns the locomotion "
                       "share (thrust_share, signed) for body.move, so that locomotion and manipulation together "
                       "never use more than the bout's sustained energy"),
    "STROKE_RULE": (D_, "a stroke is constant peak power P from rest over the limb's length a: m_eff v^2 dv = P dx "
                        "gives v^3 = 3 P a / m_eff at the end of the stroke, reached after 3 a / (2 v) (x = 2 v t / 3, "
                        "Newton's second law at constant power); the return is taken to last as long, a cycle of "
                        "3 a / v; striking runs at min(1 / cycle, sustained_w / swing_j) strokes per second of its "
                        "share; strokes = floor(share x contact time x rate): a fraction of a stroke does nothing"),
    "ROD_RULE": (D_, "every held item is a rod from the hand to its tip at handle_len_m (plain items: their own "
                     "length, 0 when compact; assemblies: the lever), swung about the shoulder: the tip is lambda = "
                     "1 + handle_len / a times as far out as the hand; an assembly's head sits at the tip (m_head "
                     "lambda^2 at the hand) and the rest is a uniform rod from the hand to the tip (m (lambda^2 + "
                     "lambda + 1) / 3 at the hand, a rod's moment of inertia); the blow delivers the held item's "
                     "kinetic energy 1/2 (m_head lambda^2 + m_rod (lambda^2 + lambda + 1) / 3) v^2, a bare limb its "
                     "own 1/2 (m_limb / 3) v^2 (version 2's strike_energy with the speed from power instead of "
                     "crafting.V_ARM)"),
    "LIFT_SPEED_M_S": (N_, "an item is liftable when its weight, with what the grips already carry, is at most the "
                           "force the muscles give at their peak power at a deliberate lifting speed of 1 m/s, F = "
                           "peak_w / v (power = force x speed): 28 kg of human muscle at a peak of 100 W/kg gives "
                           "2.8 kN, near the isometric forces of the human legs and back (2-3 kN) (estimate); a lift "
                           "is brief, so the peak power applies, not the sustained"),
    "LIFT_WORK_RULE": (D_, "gripping lifts the item through the mouth's reach: work m g reach"),
    "KNAP_J_PER_KG": (N_, "a hard-hammer blow of a few to about 10 J (hammerstones of 0.3-1 kg at 2-5 m/s, Bril et "
                          "al. 2010, J. Exp. Psychol. Hum. Percept. Perform. 36, 825) detaches a flake of a few to "
                          "tens of grams: about 500 J of blow per kg of flake (estimate). The fracture surface energy "
                          "alone, G_c = K_IC^2 / E (24 J/m2 for flint), is about 5 J/kg of 4 mm flakes: most of a blow "
                          "goes into rebound, vibration and crushing. The rest of version 2's knapping rule applies "
                          "unchanged (crafting._knappable, KNAP_HARDNESS_RATIO, KNAP_EDGE_SHARE, K_EDGE); the same "
                          "energy per kg breaks a piece off the ground (BREAK_RULE)"),
    "MAX_FLAKE_SHARE": (N_, "one stroke takes at most half of the knappable part (a blow with more energy splits the "
                            "core; the larger part stays the core)"),
    "KNAP_STROKES_RULE": (D_, "the strokes of one bout on one core act one after another: each takes a flake of blow / "
                              "KNAP_J_PER_KG while that is at most MAX_FLAKE_SHARE of the knappable part left, then "
                              "MAX_FLAKE_SHARE of what is left while that is at least crafting.MIN_ITEM_KG (closed "
                              "forms of both phases); the edge sharpens as lim - (lim - s) (P_end / P_start)^(1 / "
                              "KNAP_EDGE_SHARE), version 2's per-blow rule summed over many small flakes (the removed "
                              "shares sum to ln(P_start / P_end))"),
    "HEAP_RULE": (N_, "the flakes one striker detaches from one core in one bout, and the pieces it breaks off the "
                      "ground, lie as one heap item of their summed mass (the item pool has no count field, and one "
                      "slot per flake would fill it); a heap of flakes keeps the edge limit as its sharpness; flakes "
                      "and strokes are counted in the results"),
    "KNAP_DEBRIS_RULE": (N_, "a flake below crafting.MIN_ITEM_KG is not detached (version 2: no debris slot, nothing "
                             "happens); the heap lies where the core is (a held core's at the mouth), at the core's "
                             "temperature; several strikers on one loose core in one call: the one delivering the most "
                             "energy acts"),
    "WOUND_J_PER_KG": (RN, "a stroke on the head: damage 1 - exp(-wound_j / (0.7 J/kg x body mass)), calibrated on the "
                           "wound energy after the skin (crafting.strike_damage: x K_SOFT / (K_SOFT + K_IC of hide) = "
                           "0.25 for a blunt stone): a blow at the top of the skull-fracture range, 69 J (14-69 J, "
                           "Yoganandan et al. 1995, J. Neurotrauma 12, 659), puts 17 J into a 70 kg body and does "
                           "damage 0.30, a grave injury (estimate); strokes sum their wound energy before the "
                           "exponential"),
    "WOUND_BODY_J_PER_KG": (RN, "a stroke elsewhere: wound / (4 J/kg x body mass) in the same exponential: blunt "
                                "thoracic impacts of a 23 kg pendulum at 4.9-7 m/s (0.3-0.6 kJ) gave serious (AIS 3) "
                                "injury (Kroell, Schneider & Nahum 1974, Stapp Car Crash Conf. 18, SAE 741187): 400 J "
                                "of blow, 100 J into the tissue after the skin, does damage 0.3 to 70 kg (estimate)"),
    "HEAD_SHARE": (R_, "a stroke lands on the head with probability 0.09, the head and neck's share of the body "
                       "surface (rule of nines, Wallace 1951, Lancet 257, 501); a bout's head strokes are "
                       "Binomial(strokes, 0.09) drawn with the generator (their expectation without one)"),
    "CONTACT_TIME_RULE": (RN, "a struck or rubbed loose item or body stays in contact for (reach + its radius) / "
                              "relative speed, at most the bout (body.py's bite: gape / speed); with no velocities "
                              "given, for the bout; a held partner and the ground for the bout; strokes and rubbing "
                              "are spread over the bout, so they fill their share of the contact time"),
    "CONTACT_V_FLOOR_M_S": (N_, "numerics: relative speeds below 1e-6 m/s count as 1e-6 m/s (contact lasts the bout)"),
    "PRESS_S": (N_, "pressing two held items together holds the limb's sustained power for 10 minutes: hafting with "
                    "a lashing or adhesive takes minutes of handwork (estimate)"),
    "PRESS_RULE": (N_, "a press joins the two held items only when a separate binding item (crafting.BINDS > 0: "
                       "fibre, resin, hide) lies loose in contact with the mouth; it is consumed into the joint; "
                       "species inside the pressed items do not bind them. The left grip's item is the near end (the "
                       "handle) and the joint stays in that grip; the right grip's item is the far end (the head), "
                       "whatever their hardness. The lever is the near item's own handle_len_m plus half the far "
                       "item's (its centre of mass), times min(1, K_IC / crafting.K_ROD) of the near item (version "
                       "2's survival of a brittle rod): a compact near item gives no lever. Head mass = the far "
                       "item's mass; bond BOND_FIT + (1 - BOND_FIT) b (1 - exp(-m_bind / (BINDER_SHARE m_head))) as "
                       "version 2's combine, m_bind the binder item's binding kg and b their mass-weighted BINDS"),
    "RUB_RULE": (N_, "the rubbing item (A) is the first filled grip's (left first); it rubs the other grip's item, "
                     "else the nearest loose item in contact with the mouth, else the ground; with no held item a bare "
                     "limb (skin: materials hide) rubs the item in contact, and with nothing in contact it moves in "
                     "the air: no friction work (LIMB_SWING_RULE only). Rubbing runs at the limb's sustained power "
                     "(POWER_RULE) for its share of the bout in strokes as long as the limb (the partner's extent when "
                     "shorter) at RUB_SPEED_M_S; "
                     "less than one stroke does nothing"),
    "RUB_SPEED_M_S": (N_, "sliding speed of a rubbing stroke 0.5 m/s: hand-drill spindles 1-2 cm across turning at "
                          "5-15 rev/s slide at 0.2-0.9 m/s, fire-plough and saw strokes at 0.5-1 m/s (estimate from "
                          "experimental fire making)"),
    "LIMB_SWING_RULE": (D_, "a limb moved with nothing to rub accelerates and brakes its own rod mass each half "
                            "stroke: (m_limb / 3) V^3 / (2 a) per second (0.2 W at 70 kg), reported as limb_j and "
                            "charged by bout; it is not friction work"),
    "FRICTION_RULE": (D_, "a share crafting.FRICTION_EFFICIENCY of the rubbing work becomes heat Q at a contact spot "
                          "of radius a = min(crafting.DRILL_RADIUS_M, the partners' tip radii). The spot stays on A: "
                          "its resistance is the isothermal disc's 1 / (4 a k_A) (Carslaw & Jaeger 1959, sec. 8.2) "
                          "times the transient rise g(kappa_A t / a^2) of a disc source after t seconds of rubbing (g "
                          "= 2 sqrt(x) (1/sqrt(pi) - ierfc(1 / (2 sqrt(x)))), Carslaw & Jaeger 1959, sec. 10.5). B is "
                          "slid over: its flash resistance 1 / (4 a k_B sqrt(1 + PECLET_BRIDGE Pe)), Pe = V a / "
                          "kappa_B, in series with the track the strokes sweep, a stationary disc of the track's area "
                          "2 a l_s (radius sqrt(2 a l_s / pi)) with its transient. T_c = (Q + T_A / R_A + T_B / R_B) "
                          "/ (1 / R_A + 1 / R_B)"),
    "MOVING_FLASH": (D_, "0.31: mean temperature rise of a disc of uniform flux moving at high Peclet number, "
                         "0.31 Q / (a k) Pe^-1/2, from the one-dimensional semi-infinite solid under each point for "
                         "its dwell time averaged over the disc (0.30995; Archard 1959, Wear 2, 438 gives 0.31)"),
    "PECLET_BRIDGE": (DN, "(0.25 / 0.31)^2 = 0.650: the flash resistance 1 / (4 a k sqrt(1 + c Pe)) equals the "
                          "stationary disc's 1 / (4 a k) at Pe -> 0 and Archard's 0.31 / (a k sqrt(Pe)) at high Pe; "
                          "the join between is the stated rule"),
    "CAP_RULE": (RN, "the spot cannot be hotter than the lowest decomposition or melting temperature of its partners "
                     "(over their species above materials.PRESENT_FRACTION): materials.MELT_K, tissue chars at "
                     "crafting.TISSUE_CHAR_K, wood and plant fibre pyrolyse at PYROLYSIS_K; above it the excess heat "
                     "goes into that partner: its tissue chars (H_FG_WATER x its water per kg, version 2's "
                     "TISSUE_CHAR_RULE) to the soil, a bare limb's skin chars as a burn (skin_char_j and skin_char_kg "
                     "for body.py), a melting or pyrolysing partner takes it into its bulk"),
    "PYROLYSIS_K": (R_, "673 K: cellulose decomposes over 315-400 C and hemicellulose over 220-315 C (Yang et al. "
                        "2007, Fuel 86, 1781); a wood or fibre surface under a strong heat flux pyrolyses at about "
                        "400 C"),
    "PYROLYSING_SPECIES": (N_, "the cellulosic fuels whose surface pyrolyses at PYROLYSIS_K (wood, plant_fiber); "
                               "charcoal is char already, resin and fat melt first"),
    "IGNITION_RULE": (N_, "a fuel partner whose spot is above its ignition temperature (materials.IGNITION_K via "
                          "props; at the temperature itself the heat goes into the decomposing partner) in an O2 mole "
                          "fraction of at least crafting.X_O2_MIN, with the spot dry, makes an ember of EMBER_KG of "
                          "its fuel (the slid-over partner's first). The ember grows into a fire only on tinder "
                          "touching it: the igniting partner itself, the other partner, or else the nearest loose item "
                          "in contact with the mouth, at least TINDER_MIN_SHARE of TINDER_SPECIES; the ember and the "
                          "tinder's fuel species become the bed of a new fire at the mouth (the tinder's non-fuel rest "
                          "lies in it, crafting._split_fuel and _leave_rest) when the bed holds at least "
                          "crafting.MIN_FIRE_KG of fuel and a fire slot is free. Otherwise the ember dies: no matter "
                          "moves"),
    "EMBER_KG": (N_, "the coal of a hand or bow drill is the 1-3 g of hot dust gathered in the notch (estimate from "
                     "experimental fire making)"),
    "TINDER_SPECIES": (N_, "thermally thin fuel: fibres and dry plant litter (plant_fiber, 0.1-1 mm thick) catch from "
                           "a glowing ember, a compact stick or lump is thermally thick and does not (the tinder step "
                           "of every friction method)"),
    "TINDER_MIN_SHARE": (N_, "an item at least half tinder catches from an ember"),
    "RUB_HEAT_RULE": (D_, "a partner item that stays an item relaxes over the rubbing time toward T_air + Q_i / (h A) "
                          "(version 2's heat-transfer coefficient and time constant, crafting._h_coef, _geometry, "
                          "_relax), never below min(T_air, T_c); heat_step then relaxes it with every other item for "
                          "the bout; a loose item rubbed by several bodies takes the sum of their Q_i"),
    "RUB_ENERGY_RULE": (D_, "energy of rubbing: friction work = the limb's sustained power x rubbing time; "
                            "FRICTION_EFFICIENCY of it "
                            "is the spot's heat Q t, split into the items (items_j), the ground (ground_j), the limb's "
                            "skin (skin_j) and charring (char_j); the rest dissipates in the limb and the partners' "
                            "vibration (dissipated_j)"),
    "GROUND_K_W_MK": (RN, "default conductivity of the ground surface: soils 0.25-2.2 W/(m K) from dry to saturated "
                          "(Oke 1987, Boundary Layer Climates, 2nd ed., table 2.1); 1.0, a moist soil"),
    "GROUND_C_J_M3K": (RN, "default volumetric heat capacity of the ground, 2.2 MJ/(m3 K), a moist soil between dry "
                           "1.3-1.4 and saturated 3.0-3.1 (Oke 1987, table 2.1)"),
    "GROUND_K_DRY_W_MK": (R_, "dry soil 0.28 W/(m K): mean of Oke's sandy (0.30) and clay (0.25) soils of 40 % pore "
                              "space (Oke 1987, table 2.1)"),
    "GROUND_K_WET_W_MK": (R_, "saturated soil 1.89 W/(m K): mean of Oke's sandy (2.20) and clay (1.58) soils"),
    "GROUND_C_DRY_J_M3K": (R_, "dry soil 1.35 MJ/(m3 K): mean of Oke's sandy (1.28) and clay (1.42) soils"),
    "GROUND_C_WET_J_M3K": (R_, "saturated soil 3.03 MJ/(m3 K): mean of Oke's sandy (2.96) and clay (3.10) soils"),
    "GROUND_MOISTURE_RULE": (N_, "ground_thermal: conductivity and heat capacity interpolate linearly in the soil "
                                 "water over the bucket between the dry and saturated soils"),
    "BREAK_RULE": (N_, "a stroke with nothing in contact lands on the ground under the mouth (when the caller passes "
                       "the cells' stock, ground_stock): on a species drawn with the generator in proportion to its "
                       "mass within the mouth's reach (pi reach^2 x kg/m^2), and breaks off blow x yield / "
                       "KNAP_J_PER_KG of it, yield = version 2's chop_yield for standing wood and dig_yield (striker "
                       "against the species' working hardness) for the rest, a piece below crafting.MIN_ITEM_KG not "
                       "detached (the stroke only scuffs the ground); at most the reachable mass, a cell's stock "
                       "shared in proportion to demand. Wood comes off as a stick (a rod of crafting.HANDLE_ASPECT: "
                       "its length is the stick's handle_len_m), litter as plant_fiber, mineral pieces as compact "
                       "lumps, at the mouth at the air temperature. The items are booked in, and ground_kg goes back "
                       "to the caller (take_ground), so the patch and the deposits lose exactly that; nothing is "
                       "placed near bodies by any other rule"),
    "SURFACE_LAYER_M": (N_, "ground_stock: the loose top 5 cm of the deposits' column (globe.ACCESSIBLE_DEPTH_M) is "
                            "what a stroke can break off (gravel and clasts lie in the top centimetres; estimate)"),
    "GROUND_STOCK_RULE": (D_, "ground_stock: standing wood = patch wood / C fraction of materials wood; litter = patch "
                              "litter / C fraction of plant_fiber (dead plant matter as cellulose); mineral = deposits "
                              "x SURFACE_LAYER_M / globe.ACCESSIBLE_DEPTH_M; take_ground books wood with "
                              "patch.take_plant (its nitrogen goes back to the cell's mineral pool: a stick carries "
                              "none) and litter carbon in the patch's c_taken"),
    "SET_DOWN_RULE": (N_, "an item set down lands at the mouth (z = 0), so in a fire's bed when the mouth is within "
                          "crafting.FIRE_RADIUS_M of its centre; it is not moved to the centre (items spread over the "
                          "bed)"),
    "GRIP_ORDER_RULE": (N_, "within a call the left grip closes before the right; within a bout the primitives act "
                            "in the order grip, press, rub, force, then the grips that open (place, release) (bout)"),
    "GRASP_S": (RN, "one reach-and-grasp or let-go movement takes about a second (human reach-to-grasp 0.5-1.5 s, "
                    "Jeannerod 1984, J. Mot. Behav. 16, 235; from memory: verify): the shortest time of a grip event, "
                    "so an output of intensity i commands grips at the rate i / GRASP_S (EVENT_RULE)"),
    "EVENT_RULE": (N_, "the discrete abilities are continuous-time events, not one draw per bout: a grip closes at the "
                       "rate grip / GRASP_S (per empty grip, on what touches the mouth), opens at (release + place for "
                       "the grip place opens) / GRASP_S, and a press happens at press / PRESS_S. Over a bout dt the "
                       "two-state process of each grip ends closed with probability c / (c + o) (1 - exp(-(c + o) dt)) "
                       "from empty and open with o / (c + o) (1 - exp(-(c + o) dt)) from full (c, o the closing and "
                       "opening rates), and a press happens with 1 - exp(-press dt / PRESS_S): every per-second rate, "
                       "and so every per-day frequency, is the same for any bout length (the old per-bout Bernoulli "
                       "draw made it bouts x p). Which ability opened a grip is drawn in proportion to its rate"),
    "CLAIM_RULE": (N_, "several bodies acting on one loose item in one call (a grip, a strike on a core, an ember "
                       "taken from it, tinder, a binder): the largest grip force, rubbing heat, blow energy or power "
                       "gets it; ties go to the lowest slot"),
    "CONTACT_SCAN": (N_, "numerics: a contact query scans 64 candidates of its 3 x 3 hash cells per pass, nearest "
                         "cells first; queries with more are rescanned in full in a second pass (overflow reports "
                         "them)"),
    "SCAN_CHUNK": (N_, "numerics: candidates compared per chunk of a contact query (memory bound)"),
    "MAX_HASH_CELLS": (N_, "numerics: at most 2^24 hash cells per side (keys stay below 2^63)"),
    "HELD_FIRE_RULE": (N_, "heat_step: an item held by a body whose mouth lies within crafting.FIRE_RADIUS_M of a "
                           "live fire's centre is in that fire: it heats, transforms, chars and feeds the bed as an "
                           "item lying there (version 2's fire_step) and stays in the grip unless it went into the bed "
                           "whole; the body's exposure (mouth_fire_k, radiant_w_m2) is returned for body.py's burns"),
    "RADIANT_ITEM_RULE": (D_, "items outside a bed relax toward T_air + q / (4 h): q the radiant flux at the item from "
                              "the fires' heat release at the start of the bout (radiant_flux, version 2's point "
                              "source within WARMTH_RADIUS_M), absorbed on the projected area A / 4 of a sphere and "
                              "lost by crafting._h_coef at the air temperature"),
    "PERIODIC_FIRE_RULE": (D_, "crafting.fire_step finds the items in a bed by straight distance; heat_step first "
                               "moves each loose item within FIRE_RADIUS_M of a fire across the periodic boundary to "
                               "that fire's image (the same physical point), runs fire_step with metres as its "
                               "length unit (radius_m = 1) and wraps positions back into [0, L)"),
    "BURN_RATE_RULE": (N_, "heat_step's burn_kg_s of a fire is the heat it released in the step over the heat of "
                           "combustion per kg of its burning bed at the start (crafting's bed table), an estimate of "
                           "its mean burning rate for the smoke source (patch.smoke_emission); the ledgers use the "
                           "exact flows of crafting.fire_step"),
    "SOIL_CELL_RULE": (N_, "the residue a bout's fires send to the soil (charred tissue, fire residue with no item "
                           "slot) is spread over the bout's fires by the heat each released (equally over the fires "
                           "alive at the start when none released heat), into their fine cells; tissue charred by "
                           "rubbing goes to the rubbing mouth's cell"),
    "Z_RULE": (N_, "pos[..., 2] is height above the ground: 0 for loose items and fires (heat_step sets it), the "
                   "mouth's z for held items when one is given"),
    "REMOVE_RULE": (D_, "remove_mass books what the pool lost: (float64 mass before - float64 of the float32 mass "
                        "written back) x composition, given to the requests in proportion to their demand"),
}

#: version 2 entries (crafting.provenance keys) this module relies on
USED_CRAFTING = (
    "FIRE_RADIUS_M", "X_O2_MIN", "X_O2_REF", "FIRE_DT0_K", "AIR_GAIN", "MIN_FIRE_KG", "MIN_ITEM_KG", "EXIT_CHECK",
    "BETA_FLAMING", "PACKING_SOLID", "H_CONV", "EMISSIVITY", "CP_GAS_HOT", "M_AIR", "S_CARBON", "RADIANT_FRACTION",
    "LOAD_SHARE", "TISSUE_CHAR_K", "FRICTION_EFFICIENCY", "DRILL_RADIUS_M", "KNAP_HARDNESS_RATIO", "BRITTLE_MIN",
    "KNAP_EDGE_SHARE", "K_EDGE", "HEAD_CORE", "BOND_FIT", "BINDER_SHARE", "HANDLE_ASPECT", "K_ROD", "LIMB_FRACTION",
    "BARE_HARDNESS", "CUT_GAIN", "K_SOFT", "COOK_K", "Q10", "DECAY_REF_K", "DECAY_MIN_K", "DECAY_MAX_K",
    "DECAY_OPT_K", "WARMTH_RADIUS_M", "NEAR_FIRE_M", "SPHERE_AREA", "WOOD_CHAR_RULE", "TISSUE_CHAR_RULE",
    "FRICTION_RULE", "WOOD_C_FRACTION", "CHOP_BLUNT",
)
USED_CRAFTING_TABLES = ("FUEL_FACTOR", "DECAY_PER_DAY", "BINDS", "PACKING")


def provenance() -> dict[str, tuple[str, str]]:
    """Dotted key -> (tag, source): this module's numbers and rules, and the version 2 entries it relies on."""
    out = {f"manipulate.{k}": v for k, v in PROVENANCE.items()}
    v2 = cr.provenance()
    for name in USED_CRAFTING:
        out[f"crafting.{name}"] = v2[f"crafting.{name}"]
    for key, val in v2.items():
        if key.split(".")[1] in USED_CRAFTING_TABLES:
            out[key] = val
    return out


# ============================================================================== pools, ledgers, small helpers
def new_pools(A: int, items_per_arena: int, fires_per_arena: int, device="cpu"):
    """(ItemPool [A, I], FirePool [A, F]): empty pools on patch coordinates."""
    return it.new_item_pool(A, items_per_arena, device=device), it.new_fire_pool(A, fires_per_arena, device=device)


def new_held(A: int, N: int, device="cpu") -> torch.Tensor:
    """[A, N, 2] long, every grip empty (-1)."""
    return torch.full((A, N, GRIPS), -1, dtype=torch.long, device=device)


@dataclass
class ItemLedger:
    """Float64 item ledger per arena: ``base_kg`` [A, S] (items plus fire beds at the start), ``in_kg`` and
    ``out_kg`` [A, S] (species brought in and taken out), ``air_kg`` [A, 3] (net to the air, AIR_GASES)."""
    base_kg: torch.Tensor
    in_kg: torch.Tensor
    out_kg: torch.Tensor
    air_kg: torch.Tensor

    def book_in(self, kg: torch.Tensor) -> None:
        self.in_kg += kg.double()

    def book_out(self, kg: torch.Tensor) -> None:
        self.out_kg += kg.double()

    def book_air(self, kg: torch.Tensor) -> None:
        self.air_kg += kg.double()

    def state_dict(self) -> dict:
        return {f.name: getattr(self, f.name).clone() for f in fields(self)}

    @classmethod
    def from_state(cls, d: dict, device=None) -> "ItemLedger":
        return cls(**{f.name: d[f.name].clone().to(device) if device is not None else d[f.name].clone()
                      for f in fields(cls)})


def new_ledger(items, fires) -> ItemLedger:
    """An item ledger whose baseline is the pools' present content."""
    base = cr.ledger_mass(items, fires)
    z = torch.zeros_like(base)
    air = torch.zeros(base.shape[0], 3, dtype=torch.float64, device=base.device)
    return ItemLedger(base, z.clone(), z.clone(), air)


def ledger_error(ledger: ItemLedger, items, fires) -> dict:
    """Residuals (float64) of the item ledger: ``mass_kg`` [A] = M(now) - M(base) - (in - out - air) and
    ``elements_kg`` [A, E] the same per element (gases by ``materials.gas_element_matrix(AIR_GASES)``); and the
    ``total_kg`` [A] the pools hold now, to judge the residuals against."""
    now = cr.ledger_mass(items, fires)
    dev = now.device
    em = mat.element_matrix(dev)
    ga = mat.gas_element_matrix(mat.AIR_GASES, dev)
    d = now - ledger.base_kg - ledger.in_kg + ledger.out_kg
    return {"mass_kg": d.sum(-1) + ledger.air_kg.sum(-1),
            "elements_kg": d @ em + ledger.air_kg @ ga,
            "total_kg": now.sum(-1)}


def body_radius_m(mass_kg) -> torch.Tensor:
    """Radius (m) of the sphere of a body's mass at TISSUE_DENSITY_KG_M3."""
    m = torch.as_tensor(mass_kg, dtype=torch.float32).clamp_min(0)
    return (3.0 * m / (4.0 * math.pi * TISSUE_DENSITY_KG_M3)) ** (1.0 / 3.0)


def _radius_of(mass, density) -> torch.Tensor:
    return (3.0 * mass.clamp_min(0) / (4.0 * math.pi * density.clamp_min(1e-9))) ** (1.0 / 3.0)


def _shape_of(p: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """(tip radius, extent) [...] of items with item_props ``p``: a compact item is a sphere (tip = its radius,
    extent = its diameter); a plain item with a length is a rod of that length (tip = the rod's radius); an assembly
    is a head sphere at the end of its lever (extent = lever + head diameter)."""
    rho = p["density"].clamp_min(1e-9)
    r_sph = _radius_of(p["mass"], rho)
    ell = p["handle_len_m"].clamp_min(0)
    asm = p["head_mass"] > 0
    r_rod = torch.where(ell > 0, (p["mass"].clamp_min(0) / rho / (math.pi * ell.clamp_min(1e-9))).sqrt(), r_sph)
    r_rod = torch.minimum(r_rod, r_sph)
    r_head = _radius_of(p["head_mass"], rho)
    tip = torch.where(asm, r_head, torch.where(ell > 0, r_rod, r_sph))
    ext = torch.where(asm, ell + 2 * r_head, torch.where(ell > 0, torch.maximum(ell, 2 * r_rod), 2 * r_sph))
    return tip, ext


def item_radius_m(items) -> torch.Tensor:
    """[A, I] contact radius (m) of each item: half its extent (a compact item's sphere radius, half a stick's
    length; 0 for empty slots)."""
    p = mat.props(items.comp, items.mass * items.alive)
    p.update(handle_len_m=items.handle_len_m, head_mass=items.head_mass)
    _, ext = _shape_of(p)
    return torch.where(items.alive, 0.5 * ext, torch.zeros_like(items.mass))


def stick_length_m(mass, density) -> torch.Tensor:
    """Length (m) of a stick of a given mass: a rod of crafting.HANDLE_ASPECT, (4 V lambda^2 / pi)^(1/3)
    (version 2's handle_length without the brittleness factor)."""
    vol = torch.as_tensor(mass, dtype=torch.float32).clamp_min(0) / torch.as_tensor(density).clamp_min(1e-9)
    return (4 * vol * float(cr.HANDLE_ASPECT) ** 2 / math.pi) ** (1.0 / 3.0)


def wrap(d, L: float) -> torch.Tensor:
    """Periodic displacement wrapped into [-L/2, L/2) (as ``patch.wrap``)."""
    return torch.remainder(d + 0.5 * L, L) - 0.5 * L


def _ground3(xy, L: float) -> torch.Tensor:
    """(x, y, 0) with x and y wrapped into [0, L)."""
    xy = torch.remainder(xy[..., :2].float(), L)
    return torch.cat((xy, torch.zeros_like(xy[..., :1])), -1)


def _held3(mouth_pos, L: float | None = None) -> torch.Tensor:
    """The mouth as (x, y, z): z from a 3-vector, else 0; x and y wrapped when L is given."""
    p = mouth_pos.float()
    xy = torch.remainder(p[..., :2], L) if L is not None else p[..., :2]
    z = p[..., 2:3] if p.shape[-1] >= 3 else torch.zeros_like(p[..., :1])
    return torch.cat((xy, z), -1)


def _like(x, shape, device, default=0.0) -> torch.Tensor:
    return cr._like(default if x is None else x, shape, device)


def _first_filled(held: torch.Tensor) -> torch.Tensor:
    """[A, N] the first filled grip (left first), -1 when both are empty."""
    return torch.where(held[..., 0] >= 0, torch.zeros_like(held[..., 0]),
                       torch.where(held[..., 1] >= 0, torch.ones_like(held[..., 0]), torch.full_like(held[..., 0], -1)))


def _in_grip(held: torch.Tensor, slot: torch.Tensor) -> torch.Tensor:
    """[A, N] the item in grip ``slot`` (-1 where slot is -1 or the grip is empty)."""
    return cr.held(held, slot)


def _other_slot(slot: torch.Tensor) -> torch.Tensor:
    return torch.where(slot >= 0, 1 - slot, slot)


def _read_at(t: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """t [A, M] read at candidates m [A, ...] (valid indices)."""
    A = t.shape[0]
    return t.gather(1, m.reshape(A, -1)).reshape(m.shape)


def _reach(reach_m, body_mass_kg, shape, device) -> torch.Tensor:
    if reach_m is not None:
        return _like(reach_m, shape, device)
    if body_mass_kg is None:
        raise ValueError("pass reach_m or body_mass_kg (REACH_RULE)")
    return body_radius_m(_like(body_mass_kg, shape, device))


def _limb(limb_m, body_mass_kg, shape, device) -> tuple[torch.Tensor, torch.Tensor]:
    """(limb length a, limb mass) [shape] (LIMB_RULE)."""
    if body_mass_kg is None:
        raise ValueError("pass body_mass_kg (LIMB_RULE: the limb's mass)")
    M = _like(body_mass_kg, shape, device)
    a = _like(limb_m, shape, device) if limb_m is not None else 2.0 * body_radius_m(M)
    return a, float(cr.LIMB_FRACTION) * M


def _species_vec(names, device, value=1.0, default=0.0, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor([value if s in names else default for s in mat.SPECIES], dtype=dtype, device=device)


def _decomp_vec(device) -> torch.Tensor:
    """[S] decomposition temperature (CAP_RULE): tissue chars, cellulosic fuels pyrolyse, others NEVER."""
    v = torch.full((S,), NEVER, device=device)
    for s in cr.TISSUE:
        v[mat.IDX[s]] = float(cr.TISSUE_CHAR_K)
    for s in PYROLYSING_SPECIES:
        v[mat.IDX[s]] = PYROLYSIS_K
    return v


def cap_k(comp: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(cap [...], chars [...] bool) of compositions comp [..., S] (CAP_RULE): the lowest melting or decomposition
    temperature over the species above materials.PRESENT_FRACTION (or the dominant one), and whether that cap is
    tissue charring (the excess heat chars tissue)."""
    c = comp.float().clamp_min(0)
    tot = c.sum(-1, keepdim=True)
    w = torch.where(tot > 0, c / tot.clamp_min(1e-30), torch.zeros_like(c))
    present = (w > float(mat.PRESENT_FRACTION)) | ((w == w.amax(-1, keepdim=True)) & (w > 0))
    dev = comp.device
    melt = torch.where(present, cr._vec(mat.MELT_K, dev).expand_as(w), torch.full_like(w, NEVER)).amin(-1)
    dec = torch.where(present, _decomp_vec(dev).expand_as(w), torch.full_like(w, NEVER)).amin(-1)
    cap = torch.minimum(melt, dec)
    chars = (dec <= melt) & (dec == float(cr.TISSUE_CHAR_K))
    return cap, chars


def _disc_rise(x: torch.Tensor) -> torch.Tensor:
    """Transient rise of a disc source on a half-space, as a share of its steady rise, at x = kappa t / a^2
    (FRICTION_RULE): g = 2 sqrt(x) (1/sqrt(pi) - ierfc(1 / (2 sqrt(x)))), 0 at x = 0, 1 as x -> inf (float64)."""
    x = x.double().clamp_min(1e-30)
    sx = x.sqrt()
    u = 0.5 / sx
    ierfc = torch.exp(-u * u) / math.sqrt(math.pi) - u * torch.special.erfc(u)
    return (2 * sx * (1.0 / math.sqrt(math.pi) - ierfc)).clamp(0.0, 1.0)


def ground_thermal(soil_kg_m2, bucket_kg_m2) -> tuple[torch.Tensor, torch.Tensor]:
    """(conductivity W/(m K), volumetric heat capacity J/(m3 K)) of the ground from its soil water
    (GROUND_MOISTURE_RULE): linear between the dry and saturated soils in soil / bucket."""
    w = (torch.as_tensor(soil_kg_m2, dtype=torch.float32) / float(bucket_kg_m2)).clamp(0, 1)
    k = GROUND_K_DRY_W_MK + (GROUND_K_WET_W_MK - GROUND_K_DRY_W_MK) * w
    c = GROUND_C_DRY_J_M3K + (GROUND_C_WET_J_M3K - GROUND_C_DRY_J_M3K) * w
    return k, c


def _contact_time(reach, radius, v_rel, dt: float) -> torch.Tensor:
    """CONTACT_TIME_RULE: min(dt, (reach + radius) / max(v_rel, floor))."""
    v = v_rel.clamp_min(CONTACT_V_FLOOR_M_S)
    return torch.minimum((reach + radius) / v, torch.full_like(v, float(dt)))


# ============================================================================== contact search
def contact(q_pos, q_mask, q_reach, t_pos, t_mask, t_radius, L: float, *, exclude=None, pair_ok=None,
            scan: int = CONTACT_SCAN, chunk: int = SCAN_CHUNK):
    """For each query point q_pos [A, N, 2+] (q_mask, reach q_reach [A, N]) the target t_pos [A, M, 2+] (t_mask,
    radius t_radius [A, M]) in contact with it, |wrap(t - q)| <= reach + radius, with the smallest surface gap
    |wrap(t - q)| - radius (the lowest index on a tie). ``exclude`` [A, N] is a target index a query never takes
    (itself); ``pair_ok(m)`` maps candidate indices [A, N, C] to bool [A, N, C]. A query with more than ``scan``
    candidates in its 3 x 3 cells is rescanned in full (CONTACT_SCAN). Returns (index [A, N] long, -1 none; gap
    [A, N], inf none; overflow [A, N] bool: the queries that needed the second pass)."""
    dev = q_mask.device
    A, N = q_mask.shape
    M = t_mask.shape[1]
    idx = torch.full((A, N), -1, dtype=torch.long, device=dev)
    gap = torch.full((A, N), math.inf, dtype=torch.float32, device=dev)
    over = torch.zeros(A, N, dtype=torch.bool, device=dev)
    if N == 0 or M == 0:
        return idx, gap, over
    qm, tm = q_mask.bool(), t_mask.bool()
    reach = _like(q_reach, (A, N), dev).clamp_min(0)
    rad = _like(t_radius, (A, M), dev).clamp_min(0)
    stats = torch.stack((torch.where(qm, reach, torch.zeros_like(reach)).amax(),
                         torch.where(tm, rad, torch.zeros_like(rad)).amax(), qm.any().float(), tm.any().float()))
    r_max, t_max, any_q, any_t = stats.tolist()                                       # one host sync
    if not any_q or not any_t:
        return idx, gap, over
    h = r_max + t_max
    G = MAX_HASH_CELLS if h <= 0 else int(min(MAX_HASH_CELLS, max(1, math.floor(L / h))))
    if G < 3:
        G = 1
    cs = L / G
    ar = torch.arange(A, device=dev)[:, None]
    t_xy = torch.remainder(t_pos[..., :2].float(), L)
    tc = torch.floor(t_xy / cs).long().clamp(0, G - 1)
    sentinel = A * G * G
    key = torch.where(tm, ar * (G * G) + tc[..., 0] * G + tc[..., 1], torch.full_like(tc[..., 0], sentinel))
    sk, order = torch.sort(key.reshape(-1), stable=True)
    q_xy = torch.remainder(q_pos[..., :2].float(), L)
    qc = torch.floor(q_xy / cs).long().clamp(0, G - 1)
    if G >= 3:
        off = torch.tensor(BLOCK9, device=dev)
        frac = (q_xy - qc.float() * cs).clamp(0, cs)
        gaps = []
        for k in range(2):
            o = off[:, k].float()
            fk = frac[..., k, None]
            gaps.append(torch.where(o > 0, cs - fk, torch.where(o < 0, fk, torch.zeros_like(fk))))
        offo = off[torch.argsort(gaps[0] ** 2 + gaps[1] ** 2, dim=-1, stable=True)]   # [A, N, 9, 2], nearest first
    else:
        offo = torch.zeros(A, N, 1, 2, dtype=torch.long, device=dev)
    nb = offo.shape[2]
    cx = torch.remainder(qc[..., None, 0] + offo[..., 0], G)
    cy = torch.remainder(qc[..., None, 1] + offo[..., 1], G)
    qkey = (ar[..., None] * (G * G) + cx * G + cy).reshape(-1)
    start = torch.searchsorted(sk, qkey).reshape(A, N, nb)
    end = torch.searchsorted(sk, qkey, right=True).reshape(A, N, nb)
    cnt = torch.where(qm[..., None], end - start, torch.zeros_like(start))
    cum = torch.cumsum(cnt, -1).contiguous()
    total = cum[..., -1]
    before = cum - cnt
    over = total > scan
    t_max_cnt = int(total.max())                                                      # one host sync
    budget = min(scan, t_max_cnt)
    t_flat = t_pos[..., :2].reshape(-1, 2).float()
    r_flat = rad.reshape(-1)
    for s0 in range(0, budget, chunk):                          # chunks of candidate slots, not bodies
        slot = torch.arange(s0, min(s0 + chunk, budget), device=dev).expand(A, N, -1).contiguous()
        b = torch.searchsorted(cum, slot, right=True).clamp_max(nb - 1)
        valid = slot < total[..., None]
        p = torch.where(valid, start.gather(-1, b) + slot - before.gather(-1, b), torch.zeros_like(slot))
        j = order[p.clamp(0, sk.numel() - 1)]
        m = j % M
        d = wrap(t_flat[j] - q_xy[..., None, :], L).norm(dim=-1)
        rr = r_flat[j]
        ok = valid & (d <= reach[..., None] + rr)
        if exclude is not None:
            ok = ok & (m != exclude[..., None])
        if pair_ok is not None:
            ok = ok & pair_ok(m)
        g = torch.where(ok, d - rr, torch.full_like(d, math.inf))
        gmin, k = g.min(-1)
        cand = m.gather(-1, k[..., None]).squeeze(-1)
        better = (gmin < gap) | ((gmin == gap) & torch.isfinite(gmin) & (cand < idx))
        gap = torch.where(better, gmin, gap)
        idx = torch.where(better, cand, idx)
    if t_max_cnt > scan:
        # second pass: the overflowing queries scan every candidate of their cells
        i2, g2, _ = contact(q_pos, qm & over, reach, t_pos, tm, rad, L, exclude=exclude, pair_ok=pair_ok,
                            scan=t_max_cnt, chunk=chunk)
        idx = torch.where(over, i2, idx)
        gap = torch.where(over, g2, gap)
    return idx, gap, over


def _claim(target: torch.Tensor, strength: torch.Tensor, M: int) -> torch.Tensor:
    """[A, N] bool: the queries that get their target [A, N] (-1 none) when several want one (CLAIM_RULE)."""
    A, N = target.shape
    dev = target.device
    valid = target >= 0
    if M == 0 or not bool(valid.any()):
        return valid
    t = target.clamp_min(0)
    s = torch.where(valid, strength.float(), torch.full_like(strength.float(), -math.inf))
    best = torch.full((A, M), -math.inf, device=dev).scatter_reduce(1, t, s, reduce="amax", include_self=True)
    top = valid & (s >= best.gather(1, t))
    n = torch.arange(N, device=dev).expand(A, N)
    first = torch.full((A, M), N, dtype=torch.long, device=dev).scatter_reduce(
        1, t, torch.where(top, n, torch.full_like(n, N)), reduce="amin", include_self=True)
    return top & (first.gather(1, t) == n)


def _claim2(t1: torch.Tensor, t2: torch.Tensor, strength: torch.Tensor, M: int) -> torch.Tensor:
    """[A, N] bool: bodies that win every loose item they need (t1, t2 [A, N], -1 none) against all other claims
    on those items in the call, by strength (CLAIM_RULE; the lowest slot on a tie)."""
    A, N = t1.shape
    both = torch.cat((t1, t2), 1)
    s = torch.cat((strength, strength), 1)
    win = _claim(both, s, M)
    # a body's two claims on the same item count once
    same = (t1 >= 0) & (t1 == t2)
    w1, w2 = win[:, :N], win[:, N:]
    ok1 = (t1 < 0) | w1 | (same & w2)
    ok2 = (t2 < 0) | w2 | (same & w1)
    return ok1 & ok2


def fire_at(fires, mouth_pos, mask, L: float) -> torch.Tensor:
    """[A, N] the live fire whose bed holds the mouth (within crafting.FIRE_RADIUS_M of its centre), -1 none."""
    A, N = mask.shape
    F = fires.alive.shape[1]
    zero = torch.zeros(A, N, device=mask.device)
    f, _, _ = contact(mouth_pos, mask, zero, fires.pos, fires.alive, torch.full((A, F), float(cr.FIRE_RADIUS_M),
                                                                                device=mask.device), L)
    return f


def contacts(items, fires, *, alive, pos, mouth_pos, body_mass_kg, L: float, reach_m=None,
             scan: int = CONTACT_SCAN) -> dict:
    """What touches each living body's mouth: ``item`` (nearest loose item) and ``item_gap``, ``body`` (nearest
    other living body, by its centre ``pos``) and ``body_gap``, ``fire`` (the fire whose bed holds the mouth), all
    [A, N] with -1 / inf for none. For the mouth (``body.py``) and touch (senses)."""
    A, N = alive.shape
    dev = alive.device
    alive = alive.bool()
    reach = _reach(reach_m, body_mass_kg, (A, N), dev)
    ground = items.alive & (items.holder < 0)
    ci, cg, _ = contact(mouth_pos, alive, reach, items.pos, ground, item_radius_m(items), L, scan=scan)
    rb = body_radius_m(_like(body_mass_kg, (A, N), dev))
    cb, bg, _ = contact(mouth_pos, alive, reach, pos, alive, rb, L,
                        exclude=torch.arange(N, device=dev).expand(A, N), scan=scan)
    return {"item": ci, "item_gap": cg, "body": cb, "body_gap": bg, "fire": fire_at(fires, mouth_pos, alive, L)}


def touch(items, idx: torch.Tensor) -> dict:
    """What a body feels of items idx [A, ...] (-1 none): present, mass (kg), hardness (working Mohs), sharp,
    temp_k. For the held items pass ``held``; for the item at the mouth, ``contacts(...)['item']``."""
    p = cr.item_props(items, idx)
    return {"present": p["present"], "mass": p["mass"], "hardness": p["work_hardness"], "sharp": p["sharp"],
            "temp_k": p["temp_k"]}


def carry(items, held, mouth_pos, L: float | None = None) -> None:
    """Move every held item to its holder's mouth (call after movement)."""
    cr.sync_held(items, _held3(mouth_pos, L), held.shape[2])


# ============================================================================== shares and events
def time_split(out: dict, dt_s: float, *, alive=None, pressed=None, thrust=None) -> dict:
    """Time shares of the bout (SHARE_RULE), [A, N] each: ``press`` (PRESS_S / dt_s where a press happened,
    ``pressed`` [A, N] bool), and ``rub``, ``force``, ``locomotion`` from the intensities out['rub'], out['force']
    and |thrust| ([A, N], optional), scaled together to what the press left when they ask for more."""
    rub_i = torch.as_tensor(out["rub"], dtype=torch.float32).clamp(0, 1)
    force_i = torch.as_tensor(out["force"], dtype=torch.float32).clamp(0, 1)
    shape = torch.broadcast_shapes(rub_i.shape, force_i.shape)
    dev = rub_i.device
    live = torch.ones(shape, dtype=torch.bool, device=dev) if alive is None else alive.bool()
    rub_i, force_i = rub_i.expand(shape) * live, force_i.expand(shape) * live
    loco = torch.zeros(shape, device=dev) if thrust is None else (
        torch.as_tensor(thrust, dtype=torch.float32, device=dev).clamp(-1, 1).abs().expand(shape) * live)
    p = torch.zeros(shape, device=dev) if pressed is None else pressed.float() * min(1.0, PRESS_S / float(dt_s))
    left = (1.0 - p).clamp_min(0)
    total = rub_i + force_i + loco
    k = torch.where(total > left, left / total.clamp_min(1e-30), torch.ones_like(total))
    return {"press": p, "rub": rub_i * k, "force": force_i * k, "locomotion": loco * k}


def events(out: dict, alive, gen, *, dt_s: float | None = None, held=None) -> dict:
    """The discrete events of a bout from the decoded outputs (``brain3.decode``), drawn with the world generator in a
    fixed order. With ``dt_s`` and ``held`` (as :func:`bout` calls it): continuous-time events over the bout
    (EVENT_RULE): ``grip`` [A, N, 2] the empty grips that end the bout closed, ``open`` [A, N, 2] the full grips that
    end it open and ``open_place`` the ones of those place opened (the rest release opened), ``press`` [A, N]; and,
    per body, ``place`` and ``release`` (it opened a grip by that ability). Without them: one Bernoulli draw per
    intensity (brain3.fire), grip [A, N, 2], press, place, release [A, N]."""
    alive = alive.bool()
    grip_i = out["grip"] if "grip" in out else torch.stack((out["grip_left"], out["grip_right"]), -1)
    if dt_s is None or held is None:
        return {"grip": brain3.fire(grip_i * alive[..., None], gen), "press": brain3.fire(out["press"] * alive, gen),
                "place": brain3.fire(out["place"] * alive, gen), "release": brain3.fire(out["release"] * alive, gen)}
    dt = float(dt_s)
    A, N, K = held.shape
    dev = held.device
    gi = torch.as_tensor(grip_i, dtype=torch.float32, device=dev).clamp(0, 1).double() * alive[..., None]
    ri = torch.as_tensor(out["release"], dtype=torch.float32, device=dev).clamp(0, 1).double() * alive
    pi_ = torch.as_tensor(out["place"], dtype=torch.float32, device=dev).clamp(0, 1).double() * alive
    # the grip place would open once this one holds: the left always, the right while the left is empty
    by_slot = torch.ones(A, N, K, dtype=torch.bool, device=dev)
    by_slot[..., 1:] = (held[..., :1] < 0).expand(A, N, K - 1)
    lam_c = gi / GRASP_S
    lam_p = pi_[..., None] * by_slot / GRASP_S
    lam_o = ri[..., None] / GRASP_S + lam_p
    tot = lam_c + lam_o
    settle = -torch.expm1(-tot * dt)
    share_c = torch.where(tot > 0, lam_c / tot.clamp_min(1e-300), torch.zeros_like(tot))
    holding = held >= 0
    u = torch.rand(A, N, K, generator=gen, device=dev).double()
    close = ~holding & (u < share_c * settle) & alive[..., None]
    opened = holding & (u < (1 - share_c) * settle) & (tot > 0) & alive[..., None]
    v = torch.rand(A, N, K, generator=gen, device=dev).double()
    by_place = opened & (v * lam_o < lam_p)
    pr = torch.as_tensor(out["press"], dtype=torch.float32, device=dev).clamp(0, 1).double()
    press = (torch.rand(A, N, generator=gen, device=dev).double() < -torch.expm1(-pr * dt / PRESS_S)) & alive
    return {"grip": close, "press": press, "open": opened, "open_place": by_place, "place": by_place.any(-1),
            "release": (opened & ~by_place).any(-1)}


# ============================================================================== grip, release, place
def grip(items, held, act, *, mouth_pos, peak_w, gravity, L: float, body_mass_kg=None, alive=None, reach_m=None,
         scan: int = CONTACT_SCAN) -> dict:
    """Close the grips that fire (act [A, N, 2] bool, left and right) on the nearest loose item in contact with the
    mouth, lifted when the muscles can: m g <= peak_w / LIFT_SPEED_M_S - the weight already held (a heavier nearest
    item is not lifted and the grip fails; it does not reach past it for a lighter one). An empty grip only;
    the left grip closes first. Returns item [A, N, 2] (the item each grip took, -1 none), work_j [A, N] (lift
    work), overflow [A, N]."""
    A, N, K = held.shape
    dev = items.device
    I = items.alive.shape[1]
    alive = torch.ones(A, N, dtype=torch.bool, device=dev) if alive is None else alive.bool()
    act = act.bool()
    g = _like(gravity, (A, N), dev)
    strength = _like(peak_w, (A, N), dev).clamp_min(0) / LIFT_SPEED_M_S
    reach = _reach(reach_m, body_mass_kg, (A, N), dev)
    rad = item_radius_m(items)
    where3 = _held3(mouth_pos, L)
    took = torch.full((A, N, K), -1, dtype=torch.long, device=dev)
    work = torch.zeros(A, N, device=dev)
    over = torch.zeros(A, N, dtype=torch.bool, device=dev)
    for k in range(K):                                           # the two grips, not bodies
        hp = cr.item_props(items, held)
        spare = strength - (hp["mass"] * hp["present"]).sum(-1) * g
        go = act[..., k] & alive & (held[..., k] < 0) & (spare > 0)
        ground = items.alive & (items.holder < 0)
        mass = items.mass

        i, _, ov = contact(mouth_pos, go, reach, items.pos, ground, rad, L, scan=scan)
        over |= ov & go
        # the grip closes on the nearest item in contact; one too heavy for the muscles is not lifted (the grip fails)
        i = torch.where((i >= 0) & (_read_at(mass, i.clamp_min(0)) * g <= spare), i, torch.full_like(i, -1))
        win = _claim(i, spare, I)
        a_, n_ = win.nonzero(as_tuple=True)
        i_ = i[a_, n_]
        cr._give(items, held, a_, n_, k, i_)
        items.pos[a_, i_] = where3[a_, n_]
        took[a_, n_, k] = i_
        work[a_, n_] += items.mass[a_, i_] * g[a_, n_] * reach[a_, n_]
    return {"item": took, "work_j": work, "overflow": over}


def _set_down(items, fires, held, go, slot, mouth_pos, L: float) -> tuple:
    """Open grip ``slot`` [A, N] of the bodies ``go`` [A, N]: the item lands at the mouth (SET_DOWN_RULE), in the
    bed of the fire that holds the mouth. Returns (item [A, N], fire [A, N]), -1 none."""
    item = _in_grip(held, slot)
    go = go & (item >= 0)
    f = fire_at(fires, mouth_pos, go, L)
    a, n = go.nonzero(as_tuple=True)
    i = cr._take(items, held, a, n, slot[a, n])
    items.pos[a, i] = _ground3(mouth_pos[a, n], L)
    return torch.where(go, item, torch.full_like(item, -1)), torch.where(go, f, torch.full_like(f, -1))


def release(items, fires, held, act, *, mouth_pos, L: float) -> dict:
    """Open both grips of the bodies ``act`` [A, N]: the items land at the mouth (in a fire's bed when the mouth is
    in it). Returns item [A, N, 2] and fire [A, N, 2] (-1 none)."""
    A, N, K = held.shape
    out_i, out_f = [], []
    for k in range(K):                                           # the two grips
        i, f = _set_down(items, fires, held, act.bool(), torch.full((A, N), k, dtype=torch.long, device=held.device),
                         mouth_pos, L)
        out_i.append(i)
        out_f.append(f)
    return {"item": torch.stack(out_i, -1), "fire": torch.stack(out_f, -1)}


def _open_grips(items, fires, held, opened, by_place, alive, mouth_pos, L: float) -> tuple:
    """Open the grips the bout's events opened (opened, by_place [A, N, 2]; EVENT_RULE): each item lands at the mouth
    (SET_DOWN_RULE). Returns place's dict (item, fire [A, N]: the grip place opened) and release's (item, fire
    [A, N, 2])."""
    A, N, K = held.shape
    dev = held.device
    none = torch.full((A, N), -1, dtype=torch.long, device=dev)
    p_item, p_fire = none.clone(), none.clone()
    r_item, r_fire = [], []
    for k in range(K):                                           # the two grips
        i, f = _set_down(items, fires, held, opened[..., k] & alive, torch.full((A, N), k, dtype=torch.long,
                                                                               device=dev), mouth_pos, L)
        mine = by_place[..., k]
        p_item = torch.where(mine & (i >= 0), i, p_item)
        p_fire = torch.where(mine & (i >= 0), f, p_fire)
        r_item.append(torch.where(mine, none, i))
        r_fire.append(torch.where(mine, none, f))
    return {"item": p_item, "fire": p_fire}, {"item": torch.stack(r_item, -1), "fire": torch.stack(r_fire, -1)}


def place(items, fires, held, act, *, mouth_pos, L: float) -> dict:
    """Open one grip (the left when it holds something, else the right) of the bodies ``act`` [A, N]: the item
    lands at the mouth, in a fire's bed when the mouth is in it. Returns item, fire [A, N] (-1 none)."""
    i, f = _set_down(items, fires, held, act.bool(), _first_filled(held), mouth_pos, L)
    return {"item": i, "fire": f}


# ============================================================================== force
def swing(items, held, *, body_mass_kg, peak_w, alive=None, limb_m=None) -> dict:
    """One stroke at peak power (STROKE_RULE, ROD_RULE, LIMB_RULE) with the first filled grip's item, else the bare
    limb (crafting.tool_of). Returns the tool dict of crafting.tool_of plus slot, bare, limb_m, m_eff (kg at the
    hand), v_hand (m/s), swing_j (the stroke's work), blow_j (the held item's or bare limb's kinetic energy),
    stroke_s and cycle_s, [A, N] each."""
    A, N, K = held.shape
    dev = items.device
    alive = torch.ones(A, N, dtype=torch.bool, device=dev) if alive is None else alive.bool()
    M = _like(body_mass_kg, (A, N), dev)
    P = torch.where(alive, _like(peak_w, (A, N), dev).clamp_min(0), torch.zeros(A, N, device=dev))
    a, m_limb = _limb(limb_m, M, (A, N), dev)
    slot = _first_filled(held)
    tool = cr.tool_of(items, held, slot, M)
    item = tool["item"]
    bare = item < 0
    head = cr._gather(items.head_mass, item) * (~bare)                        # 0 for a plain item
    rod = torch.where(bare, torch.zeros_like(head), (tool["mass"] - head).clamp_min(0))
    lam = 1.0 + tool["handle_len_m"] / a.clamp_min(1e-9)
    i_item = head * lam * lam + rod * (lam * lam + lam + 1) / 3.0
    m_limb_eff = m_limb / 3.0
    m_eff = m_limb_eff + i_item
    v = (3.0 * P * a / m_eff.clamp_min(1e-12)) ** (1.0 / 3.0)
    swing_j = 0.5 * m_eff * v * v
    blow = torch.where(bare, 0.5 * m_limb_eff * v * v, 0.5 * i_item * v * v)
    stroke = torch.where(v > 0, 1.5 * a / v.clamp_min(1e-30), torch.full_like(v, math.inf))
    tool.update(slot=slot, bare=bare, v_hand=v, swing_j=swing_j, blow_j=blow, limb_m=a, m_eff=m_eff,
                stroke_s=stroke, cycle_s=2.0 * stroke)
    return tool


def _flake_strokes(n, flake, part):
    """KNAP_STROKES_RULE for n strokes of flake size ``flake`` on a knappable part ``part`` (float64 [...]).
    Returns (removed kg, flakes, part left)."""
    s = MAX_FLAKE_SHARE
    mi = float(cr.MIN_ITEM_KG)
    ok = (flake >= mi) & (part > 0) & (n >= 1)
    f = flake.clamp_min(1e-30)
    lin_max = torch.where(part >= f / s, torch.floor((part - f / s) / f) + 1, torch.zeros_like(part))
    k1 = torch.where(ok, torch.minimum(n, lin_max), torch.zeros_like(part))
    p1 = (part - k1 * f).clamp_min(0)
    geo = ok & (s * p1 >= mi)
    ratio = torch.log((mi / (s * p1).clamp_min(1e-300)).clamp_max(1.0)) / math.log(1 - s)
    k2_max = torch.where(geo, torch.floor(ratio) + 1, torch.zeros_like(part))
    k2 = torch.where(ok, torch.minimum((n - k1).clamp_min(0), k2_max), torch.zeros_like(part))
    p2 = p1 * (1 - s) ** k2
    return torch.where(ok, part - p2, torch.zeros_like(part)), k1 + k2, torch.where(ok, p2, part)


def force(items, held, intensity, *, alive, pos, mouth_pos, body_mass_kg, peak_w, sustained_w, L: float,
          dt_s: float, share=None, vel=None, reach_m=None, limb_m=None, limb_share=None, air_k=None, ground=None,
          gen=None, ledger: ItemLedger | None = None, scan: int = CONTACT_SCAN) -> dict:
    """Strike for ``share`` of the bout (default the intensity) on what is in contact with the mouth: the nearer of
    a living body (centre ``pos``) and a loose item, else the item in the other grip, else the ground (when
    ``ground`` and ``gen`` are given). Strokes (STROKE_RULE) at peak power, spaced to the sustained power, while the
    target stays in contact (CONTACT_TIME_RULE, ``vel`` [A, N, 2] the bodies' velocities over the bout); the limb
    draws ``limb_share`` (default LIMB_SUSTAINED_SHARE) of sustained_w.

    A struck item knaps when version 2's rule holds (crafting._knappable; its scratch hardness <= KNAP_HARDNESS_RATIO
    x the striker's working hardness): KNAP_STROKES_RULE, the flakes as one heap item at the edge limit (HEAP_RULE).
    A struck body takes strokes x crafting.strike_damage(blow_j, sharpness, hardness, skin hardness, skin toughness)
    as wound energy, its head strokes drawn with ``gen`` (HEAD_SHARE). On the ground the strokes break pieces off
    (BREAK_RULE): ``ground`` = dict(kg_m2 [A, C, S]
    float64, cells: fine cells per side), :func:`ground_stock`; the piece is at ``air_k``.

    Returns work_j (strokes x swing_j), time_s, strokes, hits (strokes on the target), blow_j, swing_j, hit_body,
    hit_item (-1 none), knapped, flakes, debris, removed_kg, and for the struck bodies wound_j and wound_head_j
    [A, N] (summed over strikers) and damage = 1 - exp(-(wound_head_j / WOUND_J_PER_KG + (wound_j - wound_head_j) /
    WOUND_BODY_J_PER_KG) / mass), [A, N] each; broke (the heap from the ground, -1 none), broke_kg, ground_species,
    ground_kg [A, C, S] float64 (with ``ground``); no_item_slot [A]."""
    A, N, K = held.shape
    dev = items.device
    I = items.alive.shape[1]
    dt = float(dt_s)
    alive = alive.bool()
    M = _like(body_mass_kg, (A, N), dev)
    s = _like(intensity if share is None else share, (A, N), dev).clamp(0, 1) * alive
    sw = swing(items, held, body_mass_kg=M, peak_w=peak_w, alive=alive, limb_m=limb_m)
    sus = _like(sustained_w, (A, N), dev).clamp_min(0) * _like(limb_share, (A, N), dev, default=LIMB_SUSTAINED_SHARE)
    rate = torch.where(sw["swing_j"] > 0, torch.minimum(1.0 / sw["cycle_s"], sus / sw["swing_j"].clamp_min(1e-30)),
                       torch.zeros_like(sus))
    n_total = torch.floor(s * dt * rate)
    going = alive & (n_total >= 1)
    blow = sw["blow_j"]
    reach = _reach(reach_m, M, (A, N), dev)
    ground_items = items.alive & (items.holder < 0)
    i_rad = item_radius_m(items)
    b_rad = body_radius_m(M)
    ci, cgap, _ = contact(mouth_pos, going, reach, items.pos, ground_items, i_rad, L, scan=scan)
    cb, bgap, _ = contact(mouth_pos, going, reach, pos, alive, b_rad, L,
                          exclude=torch.arange(N, device=dev).expand(A, N), scan=scan)
    hit_body = (cb >= 0) & ((ci < 0) | (bgap < cgap))
    on_item = (ci >= 0) & ~hit_body
    slot = sw["slot"]
    other = _in_grip(held, _other_slot(slot))
    on_held = going & (ci < 0) & (cb < 0) & (slot >= 0) & (other >= 0)
    # ---- how long the target stays in contact
    if vel is None:
        t_c = torch.full((A, N), dt, device=dev)
    else:
        v = vel[..., :2].float()
        v_t = v.gather(1, cb.clamp_min(0)[..., None].expand(A, N, 2))
        t_body = _contact_time(reach, _read_at(b_rad, cb.clamp_min(0)), (v - v_t).norm(dim=-1), dt)
        t_item = _contact_time(reach, _read_at(i_rad, ci.clamp_min(0)), v.norm(dim=-1), dt)
        t_c = torch.where(hit_body, t_body, torch.where(on_item, t_item, torch.full_like(t_body, dt)))
    n_hit = torch.minimum(n_total, torch.floor(s * t_c * rate))
    n_hit = torch.where(going, n_hit, torch.zeros_like(n_hit))
    core = torch.where(on_item, ci, torch.where(on_held, other, torch.full_like(ci, -1)))
    core = torch.where(n_hit >= 1, core, torch.full_like(core, -1))
    # ---- knapping (version 2's rule; the amount from the strokes' energy)
    c = cr.item_props(items, core)
    part, part_kg, cq, can, lim = cr._knappable(c)
    hard_ok = cq["hardness"] <= float(cr.KNAP_HARDNESS_RATIO) * sw["hardness"]
    flake = blow.double() / KNAP_J_PER_KG
    removed, flakes, left = _flake_strokes(n_hit.double(), flake, part_kg.double())
    mi = float(cr.MIN_ITEM_KG)
    ok = (core >= 0) & can & hard_ok & (removed >= mi) & (c["mass"].double() - removed >= mi)
    ok = _claim(torch.where(ok, core, torch.full_like(core, -1)), blow * n_hit, I)
    flake_comp = part / part_kg.clamp_min(1e-30)[..., None]
    debris = torch.full((A, N), -1, dtype=torch.long, device=dev)
    w, n = ok.nonzero(as_tuple=True)
    if w.numel():
        i = core[w, n]
        spot = torch.where(on_held[w, n][:, None], _ground3(mouth_pos[w, n], L), _ground3(items.pos[w, i], L))
        debris[w, n] = it.spawn(items, w, flake_comp[w, n], removed[w, n].float(), spot, c["temp_k"][w, n], -1,
                                sharp=lim[w, n])
    lost = ok & (debris < 0)
    done = debris >= 0
    w, n = done.nonzero(as_tuple=True)
    i = core[w, n]
    rem = items.mass[w, debris[w, n]]                                     # what the heap holds (float32)
    m_new = c["mass"][w, n] - rem
    assembly = c["head_mass"][w, n] > 0
    kg_new = (c["comp"][w, n] * c["mass"][w, n][:, None] - rem[:, None] * flake_comp[w, n]).clamp_min(0)
    items.comp[w, i] = torch.where(assembly[:, None], kg_new / m_new[:, None], items.comp[w, i])
    items.mass[w, i] = m_new
    head = torch.where(assembly, (c["head_mass"][w, n] - rem).clamp_min(mi), torch.zeros_like(rem))
    items.head_mass[w, i] = torch.minimum(head, m_new)
    keep = (left / part_kg.double().clamp_min(1e-30)).clamp(0, 1) ** (1.0 / float(cr.KNAP_EDGE_SHARE))
    new_sharp = (lim - (lim - c["sharp"]) * keep.float())
    items.sharp[w, i] = torch.maximum(items.sharp[w, i], new_sharp[w, n]).clamp(0, 1)
    # ---- wounds
    skin_h, skin_k = float(mat.MOHS["hide"]), float(mat.TOUGHNESS["hide"])
    hit = going & hit_body & (n_hit >= 1)
    per = cr.strike_damage(blow, sw["sharp"], sw["hardness"], skin_h, skin_k)
    if gen is not None:                                   # every body draws, so the stream does not hang on contacts
        n_head = torch.binomial(torch.where(hit, n_hit, torch.zeros_like(n_hit)), torch.full_like(n_hit, HEAD_SHARE),
                                generator=gen)
    else:
        n_head = n_hit * HEAD_SHARE
    wj = torch.where(hit, n_hit * per, torch.zeros_like(blow))
    wh = torch.where(hit, n_head * per, torch.zeros_like(blow))
    ar = torch.arange(A, device=dev)[:, None]
    flat = (ar * N + cb.clamp_min(0)).reshape(-1)
    hf = hit.reshape(-1)
    wound = torch.zeros(A * N, device=dev).index_add_(0, flat[hf], wj.reshape(-1)[hf]).view(A, N)
    wound_h = torch.zeros(A * N, device=dev).index_add_(0, flat[hf], wh.reshape(-1)[hf]).view(A, N)
    Mc = M.clamp_min(1e-12)
    damage = 1.0 - torch.exp(-(wound_h / (WOUND_J_PER_KG * Mc) + (wound - wound_h).clamp_min(0)
                               / (WOUND_BODY_J_PER_KG * Mc)))
    out = {"work_j": torch.where(going, n_total * sw["swing_j"], torch.zeros_like(blow)),
           "time_s": s * dt, "strokes": torch.where(going, n_total, torch.zeros_like(n_total)), "hits": n_hit,
           "blow_j": torch.where(going, blow, torch.zeros_like(blow)), "swing_j": sw["swing_j"],
           "hit_body": torch.where(hit, cb, torch.full_like(cb, -1)), "hit_item": core, "knapped": done,
           "flakes": torch.where(done, flakes.float(), torch.zeros_like(blow)), "debris": debris,
           "removed_kg": torch.where(done, items.mass.gather(1, debris.clamp_min(0)), torch.zeros_like(blow)),
           "wound_j": wound, "wound_head_j": wound_h, "damage": damage, "no_item_slot": lost.sum(1)}
    # ---- the ground (BREAK_RULE)
    free = going & (ci < 0) & (cb < 0) & ~on_held
    if ground is not None and gen is not None:
        out.update(_break_off(items, free, n_total, blow, sw, reach, mouth_pos, ground, gen, L, air_k, ledger))
    return out


def _break_off(items, free, n, blow, sw, reach, mouth_pos, ground: dict, gen, L: float, air_k, ledger) -> dict:
    """BREAK_RULE: strokes of the bodies ``free`` [A, N] land on the ground under their mouths."""
    A, N = free.shape
    dev = items.device
    kg_m2 = ground["kg_m2"].to(dev, torch.float64)
    cells = int(ground["cells"])
    C = kg_m2.shape[1]
    u = torch.rand(A, N, generator=gen, device=dev).double()                         # one draw per body per call
    cell = _cells(mouth_pos, L, cells)
    stock = kg_m2.gather(1, cell[..., None].expand(A, N, S))                         # [A, N, S]
    area = math.pi * reach.double() ** 2
    reach_kg = stock * area[..., None]
    hard = cr._work_hardness_vec(dev).double()
    tool_h = sw["hardness"].double()[..., None]
    dig = torch.where(hard > 0, (tool_h / hard.clamp_min(1e-6)).clamp(0, 1), torch.ones_like(tool_h * hard))
    chop = cr.chop_yield(sw["hardness"], sw["sharp"]).double()
    yld = dig.clone()
    yld[..., WOOD] = chop
    weight = torch.where(yld > 0, reach_kg, torch.zeros_like(reach_kg))
    cum = weight.cumsum(-1)
    tot = cum[..., -1]
    pick = (cum <= (u * tot)[..., None]).sum(-1).clamp(max=S - 1)
    y = yld.gather(-1, pick[..., None]).squeeze(-1)
    avail = reach_kg.gather(-1, pick[..., None]).squeeze(-1)
    piece = blow.double() * y / KNAP_J_PER_KG
    want = torch.minimum(n.double() * piece, avail)
    go = free & (tot > 0) & (piece >= float(cr.MIN_ITEM_KG)) & (want > 0)
    # requests on one cell and species share its stock in proportion to demand
    cell_m2 = (L / cells) ** 2
    key = (torch.arange(A, device=dev)[:, None] * C + cell) * S + pick
    demand = torch.zeros(A * C * S, dtype=torch.float64, device=dev).index_add_(
        0, key[go], want[go])
    have = kg_m2.reshape(-1) * cell_m2
    ratio = torch.where(demand > have, have / demand.clamp_min(1e-300), torch.ones_like(demand))
    take = want * ratio[key]
    go = go & (take >= float(cr.MIN_ITEM_KG))
    broke = torch.full((A, N), -1, dtype=torch.long, device=dev)
    ground_kg = torch.zeros(A, C, S, dtype=torch.float64, device=dev)
    w, n_ = go.nonzero(as_tuple=True)
    if w.numel():
        sp = pick[w, n_]
        comp = torch.nn.functional.one_hot(sp, S).float()
        kg = take[w, n_].float()
        rho = cr._vec(mat.DENSITY, dev)[sp]
        length = torch.where(sp == WOOD, stick_length_m(kg, rho), torch.zeros_like(kg))
        if air_k is None:
            raise ValueError("pass air_k: a piece broken off the ground is at the air temperature (BREAK_RULE)")
        t_air = _like(air_k, (A, N), dev)[w, n_]
        idx = it.spawn(items, w, comp, kg, _ground3(mouth_pos[w, n_], L), t_air, -1, handle_len_m=length)
        broke[w, n_] = idx
        ok = idx >= 0
        got = items.mass[w[ok], idx[ok]].double()
        ground_kg.view(A * C, S).index_put_((w[ok] * C + cell[w[ok], n_[ok]], sp[ok]), got, accumulate=True)
        if ledger is not None:
            ledger.book_in(ground_kg.sum(1))
    got_kg = torch.where(broke >= 0, items.mass.gather(1, broke.clamp_min(0)), torch.zeros(A, N, device=dev))
    return {"broke": broke, "broke_kg": got_kg, "ground_kg": ground_kg, "ground_species": torch.where(
        broke >= 0, pick, torch.full_like(pick, -1))}


# ============================================================================== rub
def rub(items, fires, held, intensity, *, alive, mouth_pos, sustained_w, x_o2, air_k, L: float, dt_s: float,
        share=None, vel=None, body_mass_kg=None, reach_m=None, limb_m=None, limb_share=None, skin_k=None, wet=None,
        ground_k=None, ground_c=None, ledger: ItemLedger | None = None, cells: int | None = None,
        scan: int = CONTACT_SCAN) -> dict:
    """Rub for ``share`` of the bout (default the intensity) at the limb's share of sustained_w (``limb_share``,
    default LIMB_SUSTAINED_SHARE; RUB_RULE): friction heats the contact
    spot (FRICTION_RULE) up to the partners' cap (CAP_RULE); a fuel partner above its ignition temperature makes an
    ember, and an ember on tinder a new fire (IGNITION_RULE); partners that stay items warm (RUB_HEAT_RULE).

    x_o2 [A] O2 mole fraction; air_k the air at the mouth ([A] or [A, N]); skin_k the limb's surface temperature
    (None = the air); wet [A, N] bool, the mouth in standing water (None = dry); ground_k, ground_c the ground's
    conductivity and heat capacity (ground_thermal; default a moist soil); vel [A, N, 2] for the contact time of a
    loose partner. Returns work_j (friction work, 0 with no partner), limb_j (LIMB_SWING_RULE), time_s, strokes,
    heat_j (Q t), contact_k, partner_a and partner_b (items, -1 = limb or ground), ember, ignited (a fire was born),
    fire [A, N] (-1 none), tinder (the tinder item, -1 none), no_fire_slot [A]; for body.py's heat and burn physics
    skin_contact_k (the temperature a bare rubbing limb's skin meets, else skin_k), skin_heat_j (conducted into the
    limb) and skin_char_j, skin_char_kg (the burn); char_kg (tissue items charred), to_soil_kg [A, S] float64 (booked
    out of the ledger), with ``cells`` soil_c_kg and soil_n_kg [A, cells^2]; energy (RUB_ENERGY_RULE)."""
    A, N, K = held.shape
    dev = items.device
    I = items.alive.shape[1]
    dt = float(dt_s)
    alive = alive.bool()
    sh = _like(intensity if share is None else share, (A, N), dev).clamp(0, 1) * alive
    P = torch.where(alive, _like(sustained_w, (A, N), dev).clamp_min(0), torch.zeros(A, N, device=dev))
    P = P * _like(limb_share, (A, N), dev, default=LIMB_SUSTAINED_SHARE)
    a_limb, m_limb = _limb(limb_m, body_mass_kg, (A, N), dev)
    t_air = _like(air_k, (A, N), dev)
    t_skin = t_air if skin_k is None else _like(skin_k, (A, N), dev)
    x = cr._per_world(x_o2, A, dev)
    slot = _first_filled(held)
    ia = _in_grip(held, slot)
    other = _in_grip(held, _other_slot(slot))
    loose = items.alive & (items.holder < 0)
    reach = _reach(reach_m, body_mass_kg, (A, N), dev)
    i_rad = item_radius_m(items)
    want = alive & (sh > 0) & (P > 0)
    ci, _, _ = contact(mouth_pos, want & (other < 0), reach, items.pos, loose, i_rad, L, scan=scan)
    ib = torch.where(other >= 0, other, ci)
    has_a, has_b = ia >= 0, ib >= 0
    b_loose = has_b & (other < 0)
    pa, pb = cr.item_props(items, ia), cr.item_props(items, ib)
    tip_a, _ = _shape_of(pa)
    tip_b, ext_b = _shape_of(pb)
    # ---- strokes and time (CONTACT_TIME_RULE for a loose partner)
    t_av = sh * dt
    if vel is not None:
        t_loose = _contact_time(reach, _read_at(i_rad, ib.clamp_min(0)), vel[..., :2].float().norm(dim=-1), dt)
        t_av = torch.where(b_loose, sh * t_loose, t_av)
    stroke = torch.where(has_b, torch.minimum(a_limb, ext_b.clamp_min(1e-6)), a_limb).clamp_min(1e-6)
    n_str = torch.where(want, torch.floor(t_av * RUB_SPEED_M_S / stroke), torch.zeros_like(t_av))
    t_r = n_str * stroke / RUB_SPEED_M_S
    active = (n_str >= 1) & (has_a | has_b)
    free = (n_str >= 1) & ~(has_a | has_b)
    # ---- thermal properties of the partners (A: held item or skin; B: item or ground)
    hide = mat.props(mat.species_vector({"hide": 1.0}, device=dev)[None], torch.ones(1, device=dev))
    k_a = torch.where(has_a, pa["conductivity"], hide["conductivity"].expand(A, N)).double().clamp_min(1e-6)
    c_a = torch.where(has_a, pa["density"] * pa["cp"], (hide["density"] * hide["cp"]).expand(A, N)).double()
    gk = _like(ground_k, (A, N), dev, default=GROUND_K_W_MK)
    gc = _like(ground_c, (A, N), dev, default=GROUND_C_J_M3K)
    k_b = torch.where(has_b, pb["conductivity"], gk).double().clamp_min(1e-6)
    c_b = torch.where(has_b, pb["density"] * pb["cp"], gc).double().clamp_min(1.0)
    T_a = torch.where(has_a, pa["temp_k"], t_skin).double()
    T_b = torch.where(has_b, pb["temp_k"], t_air).double()
    cap_a, char_a = cap_k(pa["comp"])
    skin_cap, _ = cap_k(mat.species_vector({"hide": 1.0}, device=dev))
    cap_a = torch.where(has_a, cap_a, skin_cap.expand(A, N)).double()
    char_a = torch.where(has_a, char_a, torch.ones_like(char_a))
    cap_b, char_b = cap_k(pb["comp"])
    cap_b = torch.where(has_b, cap_b, torch.full_like(cap_b, NEVER)).double()
    char_b = has_b & char_b
    # ---- the contact spot (FRICTION_RULE)
    inf = torch.full((A, N), math.inf, device=dev)
    spot = torch.minimum(torch.where(has_a, tip_a, inf), torch.where(has_b, tip_b, inf))
    spot = torch.minimum(spot, torch.full_like(spot, float(cr.DRILL_RADIUS_M))).clamp_min(1e-6).double()
    t = t_r.double()
    kap_a, kap_b = k_a / c_a.clamp_min(1.0), k_b / c_b
    r_a = _disc_rise(kap_a * t / spot ** 2) / (4 * spot * k_a)
    pe = RUB_SPEED_M_S * spot / kap_b
    r_flash = 1.0 / (4 * spot * k_b * (1 + PECLET_BRIDGE * pe).sqrt())
    a_tr = (2 * spot * stroke.double() / math.pi).sqrt().clamp_min(spot)
    r_track = _disc_rise(kap_b * t / a_tr ** 2) / (4 * a_tr * k_b)
    r_b = r_flash + r_track
    g_a, g_b = 1.0 / r_a.clamp_min(1e-30), 1.0 / r_b.clamp_min(1e-30)
    Q = (float(cr.FRICTION_EFFICIENCY) * P).double()
    t_free = (Q + g_a * T_a + g_b * T_b) / (g_a + g_b)
    cap = torch.minimum(cap_a, cap_b)
    capped = active & (t_free > cap)
    t_c = torch.where(capped, cap, t_free)
    q_a = g_a * (t_c - T_a)
    q_b = g_b * (t_c - T_b)
    q_ex = torch.where(capped, (Q - q_a - q_b).clamp_min(0), torch.zeros_like(Q))
    q_b = torch.where(capped, q_b, Q - q_a)
    ex_on_a = capped & (cap_a <= cap_b)
    ex_on_b = capped & ~ex_on_a
    zero = torch.zeros_like(Q)
    t_c = torch.where(active, t_c, t_air.double())
    q_a, q_b = torch.where(active, q_a, zero), torch.where(active, q_b, zero)
    # ---- ignition: ember and tinder (IGNITION_RULE)
    fuel_vec = cr._bed_tables(dev)["fuel"]
    tinder_vec = _species_vec(TINDER_SPECIES, dev, dtype=torch.float64)
    kg_a = pa["comp"].double() * pa["mass"].double()[..., None]
    kg_b = pb["comp"].double() * pb["mass"].double()[..., None]
    fuel_a, fuel_b = (kg_a * fuel_vec).sum(-1), (kg_b * fuel_vec).sum(-1)
    dry = torch.ones_like(active) if wet is None else ~wet.bool()
    air_ok = (x[:, None] >= float(cr.X_O2_MIN)) & dry
    ign_a = active & has_a & (fuel_a > 0) & (t_c > pa["ignition_k"].double())
    ign_b = active & has_b & (fuel_b > 0) & (t_c > pb["ignition_k"].double())
    ember = (ign_a | ign_b) & air_ok
    src_b = ign_b
    src = torch.where(src_b, ib, ia)
    kg_src = torch.where(src_b[..., None], kg_b, kg_a)
    opp = torch.where(src_b, ia, ib)
    kg_opp = torch.where(src_b[..., None], kg_a, kg_b)

    def fine_part(kg):
        return (kg * tinder_vec).sum(-1) / kg.sum(-1).clamp_min(1e-30)

    src_tinder = fine_part(kg_src) >= TINDER_MIN_SHARE
    opp_tinder = (opp >= 0) & (fine_part(kg_opp) >= TINDER_MIN_SHARE) & ((kg_opp * fuel_vec).sum(-1) > 0)
    need_loose = ember & ~src_tinder & ~opp_tinder
    tin_mask = loose & ((items.comp.double() * tinder_vec).sum(-1) >= TINDER_MIN_SHARE)
    lt, _, _ = contact(mouth_pos, need_loose, reach, items.pos, tin_mask, i_rad, L, exclude=ib, scan=scan)
    kind = torch.where(src_tinder, 0, torch.where(opp_tinder, 1, torch.where(lt >= 0, 2, -1)))
    kind = torch.where(ember, kind, torch.full_like(kind, -1))
    tin = torch.where(kind == 0, src, torch.where(kind == 1, opp, torch.where(kind == 2, lt, torch.full_like(lt, -1))))
    pt_ = cr.item_props(items, tin)
    kg_tin = pt_["comp"].double() * pt_["mass"].double()[..., None]
    # the ember: EMBER_KG of the source's fuel (all of it when the rest would be below MIN_ITEM_KG)
    mi = float(cr.MIN_ITEM_KG)
    src_mass = kg_src.sum(-1)
    e_kg = torch.minimum(torch.full_like(src_mass, EMBER_KG), (kg_src * fuel_vec).sum(-1))
    src_whole = (kind == 0) | (src_mass - e_kg < mi)
    ember_sp = torch.where(src_whole[..., None], kg_src,
                           kg_src * fuel_vec * (e_kg / (kg_src * fuel_vec).sum(-1).clamp_min(1e-300))[..., None])
    to_bed_src, rest_src, rest_kg_src, whole_src = (t.reshape(A, N, *t.shape[1:])
                                                    for t in cr._split_fuel(ember_sp.reshape(-1, S)))
    to_bed_tin, rest_tin, rest_kg_tin, whole_tin = (t.reshape(A, N, *t.shape[1:])
                                                    for t in cr._split_fuel(kg_tin.reshape(-1, S)))
    use_tin = (kind == 1) | (kind == 2)
    bed = to_bed_src + torch.where(use_tin[..., None], to_bed_tin, torch.zeros_like(to_bed_tin))
    bed_fuel = (bed * fuel_vec).sum(-1)
    wantf = (kind >= 0) & (bed_fuel >= float(cr.MIN_FIRE_KG))
    # loose items taken (the ember's source, the tinder): the strongest heat gets each (CLAIM_RULE)
    l_src = torch.where(wantf & (src == ib) & b_loose, src, torch.full_like(src, -1))
    l_tin = torch.where(wantf & use_tin & ((kind == 2) | ((kind == 1) & (opp == ib) & b_loose)), tin,
                        torch.full_like(tin, -1))
    wantf = wantf & _claim2(l_src, l_tin, Q.float(), I)
    fire = torch.full((A, N), -1, dtype=torch.long, device=dev)
    w, n = wantf.nonzero(as_tuple=True)
    if w.numel() and fires.alive.shape[1]:
        where3 = _ground3(mouth_pos[w, n], L)
        temp = cr.fire_temperature(bed[w, n][:, None, :], torch.zeros(w.numel(), 1, device=dev), x[w],
                                   t_air[w, n][:, None])[:, 0]
        fire[w, n] = it.spawn_fire(fires, w, where3, bed[w, n].float(), temp, air=0.0)
    born = fire >= 0
    # matter into the new fires: the ember's source and the tinder
    grip_of = lambda item: torch.where(item == ia, slot, torch.where((item == other) & (item >= 0), _other_slot(slot),
                                                                     torch.full_like(slot, -1)))
    w, n = born.nonzero(as_tuple=True)
    if w.numel():
        fpos = fires.pos[w, fire[w, n]]
        # the source: whole (into the bed, its non-fuel rest left at the fire) or less its ember
        sw_ = src_whole[w, n]
        i_src = src[w, n]
        k_src = grip_of(src)[w, n]
        h = sw_ & (k_src >= 0)
        cr._take(items, held, w[h], n[h], k_src[h])
        cr._leave_rest(items, w[sw_], i_src[sw_], rest_src[w, n][sw_], rest_kg_src[w, n][sw_], whole_src[w, n][sw_],
                       fpos[sw_])
        part = ~sw_
        if bool(part.any()):
            wp, np_, ip = w[part], n[part], i_src[part]
            kg_new = (kg_src[wp, np_] - ember_sp[wp, np_]).clamp_min(0)
            m_new = kg_new.sum(-1)
            items.comp[wp, ip] = (kg_new / m_new[:, None]).float()
            items.mass[wp, ip] = m_new.float()
            items.head_mass[wp, ip] = torch.minimum(items.head_mass[wp, ip], items.mass[wp, ip])
        # the tinder (kinds 1 and 2): its fuel into the bed, its rest left at the fire
        ut = use_tin[w, n]
        if bool(ut.any()):
            wt, nt = w[ut], n[ut]
            i_t = tin[wt, nt]
            k_t = grip_of(tin)[wt, nt]
            h = k_t >= 0
            cr._take(items, held, wt[h], nt[h], k_t[h])
            cr._leave_rest(items, wt, i_t, rest_tin[wt, nt], rest_kg_tin[wt, nt], whole_tin[wt, nt], fpos[ut])
    # ---- the excess heat at the cap (CAP_RULE)
    dry_vec = cr._tissue_tables(dev)["dry_j_kg"]
    tis_mask = cr._tissue_tables(dev)["mask"]
    char_heat = q_ex * t
    limb = active & ~has_a
    skin_char_j = torch.where(limb & ex_on_a, char_heat, zero)
    skin_dry = float((mat.species_vector({"hide": 1.0}, dtype=torch.float64) * dry_vec.cpu()).sum())
    skin_char_kg = skin_char_j / max(skin_dry, 1e-30)
    item_char_a = has_a & ex_on_a & char_a & ~born
    item_char_b = has_b & ex_on_b & char_b & ~born
    ar = torch.arange(A, device=dev)[:, None].expand(A, N)
    to_soil = torch.zeros(A, S, dtype=torch.float64, device=dev)
    char_kg = torch.zeros(A, N, dtype=torch.float64, device=dev)
    keys = torch.cat(((ar * I + ia.clamp_min(0))[item_char_a], (ar * I + ib.clamp_min(0))[item_char_b]))
    soil_c = soil_n = None
    if keys.numel():
        heat_i = torch.zeros(A * I, dtype=torch.float64, device=dev).index_add_(
            0, keys, torch.cat((char_heat[item_char_a], char_heat[item_char_b])))
        touched = torch.zeros(A * I, dtype=torch.bool, device=dev)
        touched[keys] = True
        wt, itm = touched.view(A, I).nonzero(as_tuple=True)
        kg_i = items.comp[wt, itm].double() * items.mass[wt, itm].double()[:, None]
        tissue = kg_i * tis_mask
        need = (tissue * dry_vec).sum(-1)
        frac = torch.where(need > 0, (heat_i[wt * I + itm] / need.clamp_min(1e-300)).clamp(max=1.0),
                           torch.ones_like(need))
        charred = tissue * frac[:, None]
        left = (kg_i - charred).clamp_min(0)
        m_left = left.sum(-1)
        gone = m_left < mi
        charred = torch.where(gone[:, None], kg_i, charred)
        keep = ~gone
        items.comp[wt[keep], itm[keep]] = (left[keep] / m_left[keep, None]).float()
        items.mass[wt[keep], itm[keep]] = m_left[keep].float()
        items.head_mass[wt[keep], itm[keep]] = torch.minimum(items.head_mass[wt[keep], itm[keep]],
                                                             items.mass[wt[keep], itm[keep]])
        items.sharp[wt[keep], itm[keep]] = 0.0
        cr._remove(items, held, wt[gone], itm[gone])
        to_soil.index_add_(0, wt, charred)
        # each rubbing body's share of what its item lost, for the soil of its mouth's cell
        lost_i = torch.zeros(A * I, dtype=torch.float64, device=dev)
        lost_i[wt * I + itm] = charred.sum(-1)
        for mask, idx in ((item_char_a, ia), (item_char_b, ib)):
            key = ar * I + idx.clamp_min(0)
            share = torch.where(mask, char_heat / heat_i[key].clamp_min(1e-300), zero)
            char_kg += share * lost_i[key]
        if cells is not None:
            el = mat.element_matrix(dev)
            per_kg = torch.zeros(A * I, E, dtype=torch.float64, device=dev)
            per_kg[wt * I + itm] = (charred @ el) / charred.sum(-1, keepdim=True).clamp_min(1e-300)
            body_el = torch.zeros(A, N, E, dtype=torch.float64, device=dev)
            for mask, idx in ((item_char_a, ia), (item_char_b, ib)):
                key = ar * I + idx.clamp_min(0)
                share = torch.where(mask, char_heat / heat_i[key].clamp_min(1e-300), zero)
                body_el += (share * lost_i[key])[..., None] * per_kg[key]
            where = _cells(mouth_pos, L, cells)
            soil_c = _per_cell(where, body_el[..., _C], cells)
            soil_n = _per_cell(where, body_el[..., _N], cells)
        if ledger is not None:
            ledger.book_out(to_soil)
    # ---- partners that stay items warm (RUB_HEAT_RULE), the excess of a non-charring cap into its bulk
    bulk_a = q_a + torch.where(ex_on_a & ~char_a, q_ex, zero)
    bulk_b = q_b + torch.where(ex_on_b & ~char_b, q_ex, zero)
    still_a = active & has_a & _read_at(items.alive.long(), ia.clamp_min(0)).bool()
    still_b = active & has_b & _read_at(items.alive.long(), ib.clamp_min(0)).bool()
    still_a = still_a & ~(born & ((src == ia) & src_whole | (tin == ia) & use_tin))
    still_b = still_b & ~(born & ((src == ib) & src_whole | (tin == ib) & use_tin))
    keys = torch.cat(((ar * I + ia.clamp_min(0))[still_a], (ar * I + ib.clamp_min(0))[still_b]))
    if keys.numel():
        q_item = torch.zeros(A * I, dtype=torch.float64, device=dev).index_add_(
            0, keys, torch.cat((bulk_a[still_a], bulk_b[still_b])))
        t_env_air = torch.full((A * I,), -math.inf, device=dev).scatter_reduce(
            0, keys, torch.cat((t_air[still_a], t_air[still_b])), reduce="amax", include_self=True)
        low = torch.minimum(t_air.double(), t_c)
        t_floor = torch.full((A * I,), math.inf, dtype=torch.float64, device=dev).scatter_reduce(
            0, keys, torch.cat((low[still_a], low[still_b])), reduce="amin", include_self=True)
        t_len = torch.zeros(A * I, dtype=torch.float64, device=dev).scatter_reduce(
            0, keys, torch.cat((t[still_a], t[still_b])), reduce="amax", include_self=True)
        touched = torch.zeros(A * I, dtype=torch.bool, device=dev)
        touched[keys] = True
        wt, itm = touched.view(A, I).nonzero(as_tuple=True)
        flat = wt * I + itm
        p = mat.props(items.comp[wt, itm], items.mass[wt, itm])
        mcp, area, tau_int = cr._geometry(p)
        t0 = items.temp_k[wt, itm].double()
        ta = t_env_air[flat].double()
        t_ss = torch.maximum(ta + q_item[flat] / (cr._h_coef(ta, t0) * area).clamp_min(1e-30), t_floor[flat])
        tau = mcp / (cr._h_coef(ta, t0) * area).clamp_min(1e-30) + tau_int
        t1 = t_ss + (t0 - t_ss) * torch.exp(-t_len[flat] / tau.clamp_min(1e-6))
        items.temp_k[wt, itm] = t1.float()
        items.peak_k[wt, itm] = torch.maximum(items.peak_k[wt, itm], torch.maximum(t0, t1).float())
    # ---- results
    work = torch.where(active, P.double() * t, zero)
    limb_j = torch.where(free, (m_limb / 3.0).double() * RUB_SPEED_M_S ** 3 / (2 * a_limb.double()) * t, zero)
    spot_j = torch.where(active, Q * t, zero)
    items_j = torch.where(has_a & active, bulk_a * t, zero) + torch.where(has_b & active, bulk_b * t, zero)
    ground_j = torch.where(active & ~has_b, q_b * t, zero)
    skin_j = torch.where(limb, q_a * t, zero)
    char_j = torch.where(active, torch.where(ex_on_a & char_a, char_heat, zero)
                         + torch.where(ex_on_b & char_b, char_heat, zero), zero)
    out = {"work_j": work.float(), "limb_j": limb_j.float(), "time_s": t_r, "strokes": n_str,
           "heat_j": spot_j.float(), "contact_k": t_c.float(),
           "skin_contact_k": torch.where(limb, t_c.float(), t_skin),
           "skin_heat_j": skin_j.float(), "skin_char_j": skin_char_j.float(), "skin_char_kg": skin_char_kg.float(),
           "partner_a": torch.where(active, ia, torch.full_like(ia, -1)),
           "partner_b": torch.where(active, ib, torch.full_like(ib, -1)),
           "ember": ember, "ignited": born, "fire": fire, "tinder": torch.where(born, tin, torch.full_like(tin, -1)),
           "no_fire_slot": (wantf & ~born).sum(1), "char_kg": char_kg.float(), "to_soil_kg": to_soil,
           "energy": {"work_j": work, "spot_j": spot_j, "items_j": items_j, "ground_j": ground_j, "skin_j": skin_j,
                      "char_j": char_j, "dissipated_j": work - spot_j, "limb_j": limb_j}}
    if cells is not None:
        z = torch.zeros(A, cells * cells, dtype=torch.float64, device=dev)
        out["soil_c_kg"] = z if soil_c is None else soil_c
        out["soil_n_kg"] = z.clone() if soil_n is None else soil_n
    return out


# ============================================================================== press
def press(items, held, act, *, alive, mouth_pos, sustained_w, L: float, body_mass_kg=None, reach_m=None,
          limb_share=None, scan: int = CONTACT_SCAN) -> dict:
    """Press the two held items together (act [A, N] bool, both grips full) and join them when a separate binding
    item lies loose in contact with the mouth (PRESS_RULE). The joint stays in the left grip (the near end) with the
    summed species and mass, the far item's sharpness as the head's, the heat content kept (temperature by heat
    capacity), the highest peak temperature and age. Returns pressed (a press happened), item (the joint, -1 none),
    bond, handle_len_m, binder (the binder item consumed, -1 none), work_j and time_s, [A, N] each."""
    A, N, K = held.shape
    dev = items.device
    I = items.alive.shape[1]
    alive = alive.bool()
    power = _like(sustained_w, (A, N), dev).clamp_min(0) * _like(limb_share, (A, N), dev, default=LIMB_SUSTAINED_SHARE)
    both = (held >= 0).all(-1)
    go = act.bool() & alive & both
    ni, fi = held[..., 0], held[..., 1]                     # near end (handle), far end (head)
    binds = cr._vec(cr.BINDS, dev)
    item_binds = (items.comp * binds).sum(-1)
    loose_binder = items.alive & (items.holder < 0) & (item_binds > 0)
    reach = _reach(reach_m, body_mass_kg, (A, N), dev)
    bi, _, _ = contact(mouth_pos, go, reach, items.pos, loose_binder, item_radius_m(items), L, scan=scan)
    bi = torch.where(_claim(bi, power, I), bi, torch.full_like(bi, -1))
    np_, fp, bp = cr.item_props(items, ni), cr.item_props(items, fi), cr.item_props(items, bi)
    has_b = bi >= 0
    mb = torch.where(has_b, bp["mass"], torch.zeros_like(bp["mass"]))
    kg_b = bp["comp"] * mb[..., None]
    bind_species = binds > 0
    m_bind = (kg_b * bind_species).sum(-1)
    b = torch.where(m_bind > 0, (kg_b * binds).sum(-1) / m_bind.clamp_min(1e-30), torch.zeros_like(m_bind))
    joined = go & has_b & (m_bind > 0)
    head = fp["mass"]
    bond = float(cr.BOND_FIT) + (1 - float(cr.BOND_FIT)) * b * (
        1 - torch.exp(-m_bind / (float(cr.BINDER_SHARE) * head).clamp_min(1e-12)))
    lever = (np_["handle_len_m"] + 0.5 * fp["handle_len_m"]) * (np_["toughness"] / float(cr.K_ROD)).clamp(0, 1)
    kg = np_["comp"] * np_["mass"][..., None] + fp["comp"] * fp["mass"][..., None] + kg_b
    mass = np_["mass"] + fp["mass"] + mb
    nc, fc = np_["heat_capacity_j_k"], fp["heat_capacity_j_k"]
    bc = bp["heat_capacity_j_k"] * has_b
    temp = (nc * np_["temp_k"] + fc * fp["temp_k"] + bc * bp["temp_k"]) / (nc + fc + bc).clamp_min(1e-12)
    w, n = joined.nonzero(as_tuple=True)
    i = ni[w, n]
    peaks = torch.stack([cr._gather(items.peak_k, x) * (x >= 0) for x in (ni, fi, bi)], -1).amax(-1)
    ages = torch.stack([cr._gather(items.age_d, x) * (x >= 0) for x in (ni, fi, bi)], -1).amax(-1)
    items.comp[w, i] = kg[w, n] / mass[w, n][:, None]
    items.mass[w, i] = mass[w, n]
    items.head_mass[w, i] = head[w, n]
    items.handle_len_m[w, i] = lever[w, n]
    items.bond[w, i] = bond[w, n]
    items.sharp[w, i] = fp["sharp"][w, n]
    items.temp_k[w, i] = temp[w, n]
    items.peak_k[w, i] = peaks[w, n]
    items.age_d[w, i] = ages[w, n]
    cr._remove(items, held, w, fi[w, n])
    wb = joined & has_b
    cr._remove(items, held, wb.nonzero(as_tuple=True)[0], bi[wb])
    none = torch.full_like(ni, -1)
    return {"pressed": go, "item": torch.where(joined, ni, none),
            "bond": torch.where(joined, bond, torch.zeros_like(bond)),
            "handle_len_m": torch.where(joined, lever, torch.zeros_like(lever)),
            "binder": torch.where(joined, bi, none),
            "work_j": torch.where(go, power * PRESS_S, torch.zeros_like(power)),
            "time_s": torch.where(go, torch.full_like(power, PRESS_S), torch.zeros_like(power))}


# ============================================================================== one bout of the primitives
def bout(items, fires, held, out: dict, gen, *, alive, pos, mouth_pos, body_mass_kg, peak_w, sustained_w, gravity,
         x_o2, air_k, L: float, dt_s: float, thrust=None, vel=None, reach_m=None, limb_m=None, limb_share=None,
         skin_k=None, wet=None, ground_k=None, ground_c=None, ground=None, ledger: ItemLedger | None = None,
         cells: int | None = None, scan: int = CONTACT_SCAN) -> dict:
    """The primitives of one bout in the order grip, press, rub, force, then the grips that open (place, release;
    GRIP_ORDER_RULE), the discrete ones as continuous-time events over the bout (:func:`events`, EVENT_RULE), the
    continuous ones on their time shares (:func:`time_split`). ``out`` holds the decoded motor outputs (grip or
    grip_left/grip_right, press, rub, force, place, release, each in [0, 1]); ``thrust`` [A, N] the locomotion output.
    Held items are first carried to the mouth.

    Bodies that are not alive first let go of what they hold (it falls at their mouth). Returns the work of each
    primitive and their sum ``work_j`` [A, N] (mechanical, for body.py's energy, rub's limb_j included), ``time_s``
    (the manipulation time), ``shares``, ``thrust_share`` (the signed locomotion share for body.move; None without
    thrust), the struck bodies' ``wound_j`` and ``damage`` [A, N], the events, each primitive's own result under its
    name, and ``dropped_by_dead`` [A, N, 2]."""
    A, N = alive.shape
    dev = items.device
    alive = alive.bool()
    reach = _reach(reach_m, body_mass_kg, (A, N), dev)
    carry(items, held, mouth_pos, L)
    dead = release(items, fires, held, ~alive & (held >= 0).any(-1), mouth_pos=mouth_pos, L=L)
    ev = events(out, alive, gen, dt_s=dt_s, held=held)
    g = grip(items, held, ev["grip"], mouth_pos=mouth_pos, peak_w=peak_w, gravity=gravity, L=L, alive=alive,
             reach_m=reach, scan=scan)
    pr = press(items, held, ev["press"], alive=alive, mouth_pos=mouth_pos, sustained_w=sustained_w, L=L,
               reach_m=reach, limb_share=limb_share, scan=scan)
    sh = time_split(out, dt_s, alive=alive, pressed=pr["pressed"], thrust=thrust)
    r = rub(items, fires, held, out["rub"], share=sh["rub"], alive=alive, mouth_pos=mouth_pos,
            sustained_w=sustained_w, x_o2=x_o2, air_k=air_k, L=L, dt_s=dt_s, vel=vel, body_mass_kg=body_mass_kg,
            reach_m=reach, limb_m=limb_m, limb_share=limb_share, skin_k=skin_k, wet=wet, ground_k=ground_k,
            ground_c=ground_c, ledger=ledger, cells=cells, scan=scan)
    f = force(items, held, out["force"], share=sh["force"], alive=alive, pos=pos, mouth_pos=mouth_pos,
              body_mass_kg=body_mass_kg, peak_w=peak_w, sustained_w=sustained_w, L=L, dt_s=dt_s, vel=vel,
              reach_m=reach, limb_m=limb_m, limb_share=limb_share, air_k=air_k, ground=ground, gen=gen,
              ledger=ledger, scan=scan)
    pl, rl = _open_grips(items, fires, held, ev["open"], ev["open_place"], alive, mouth_pos, L)
    work = g["work_j"] + pr["work_j"] + r["work_j"] + r["limb_j"] + f["work_j"]
    time = pr["time_s"] + r["time_s"] + f["time_s"]
    ts = None
    if thrust is not None:
        th = torch.as_tensor(thrust, dtype=torch.float32, device=dev)
        ts = torch.sign(th) * sh["locomotion"]
    return {"work_j": work, "time_s": time, "shares": sh, "thrust_share": ts, "wound_j": f["wound_j"],
            "damage": f["damage"], "events": ev, "grip": g, "press": pr, "rub": r, "force": f, "place": pl,
            "release": rl, "dropped_by_dead": dead["item"]}


# ============================================================================== matter in and out
def spawn_items(items, a, comp, mass, pos, temp_k, *, ledger: ItemLedger | None = None, L: float | None = None,
                **fields_) -> torch.Tensor:
    """Loose items brought in from outside (carcasses, loose material): ``items.spawn`` at (x, y, 0) (wrapped when L
    is given) with the species kept booked in ``ledger.in_kg`` exactly as the pool stores them. ``handle_len_m``
    gives an elongated item its length. Returns the slots [M] (-1: the arena's pool is full or the request is
    invalid; nothing is booked for it)."""
    a = torch.as_tensor(a, device=items.device).long().reshape(-1)
    p = torch.as_tensor(pos, dtype=torch.float32, device=items.device)
    p = p.expand(a.shape[0], p.shape[-1]) if p.dim() == 1 else p
    where3 = _ground3(p, L) if L is not None else torch.cat((p[..., :2], torch.zeros_like(p[..., :1])), -1)
    idx = it.spawn(items, a, comp, mass, where3, temp_k, -1, **fields_)
    if ledger is not None:
        ok = idx >= 0
        kg = items.comp[a[ok], idx[ok]].double() * items.mass[a[ok], idx[ok]].double()[:, None]
        ledger.book_in(torch.zeros_like(ledger.in_kg).index_add_(0, a[ok], kg))
    return idx


def remove_mass(items, held, a, i, kg, *, ledger: ItemLedger | None = None) -> dict:
    """Take up to ``kg`` [M] from items (a [M], i [M]; loose or held) in proportion to their species (a mouth
    biting into an item). Requests on one item share what it lost in proportion to what they ask; an item left
    below crafting.MIN_ITEM_KG goes whole (and leaves its grip). Booked in ``ledger.out_kg`` as what the pool lost
    (REMOVE_RULE). Returns species_kg [M, S] float64, cooked [M] bool (the item had been at crafting.COOK_K or
    above), cook_gain [M] (materials' gain when cooked, else 1)."""
    dev = items.device
    A, I = items.alive.shape
    a = torch.as_tensor(a, device=dev).long().reshape(-1)
    i = torch.as_tensor(i, device=dev).long().reshape(-1)
    want = torch.as_tensor(kg, dtype=torch.float64, device=dev).reshape(-1).expand(a.shape[0]).clamp_min(0)
    ic = i.clamp_min(0)
    valid = (i >= 0) & items.alive[a, ic] & (want > 0)
    key = a * I + ic
    demand = torch.zeros(A * I, dtype=torch.float64, device=dev).index_add_(0, key[valid], want[valid])
    before = (items.mass.double() * items.alive).reshape(-1)
    comp_all = items.comp.double().reshape(A * I, S).clone()
    take = torch.minimum(before, demand)
    gone = (demand > 0) & (before - take < float(cr.MIN_ITEM_KG))
    p = mat.props(items.comp[a, ic], items.mass[a, ic])
    cooked = valid & (items.peak_k[a, ic] >= float(cr.COOK_K))
    gain = torch.where(cooked, p["cook_gain"], torch.ones_like(p["cook_gain"]))
    # write back: shrink the items, remove the ones taken whole
    left = torch.where(gone, torch.zeros_like(before), before - take).view(A, I)
    shrink = (demand > 0).view(A, I) & ~gone.view(A, I)
    ws, iS = shrink.nonzero(as_tuple=True)
    items.mass[ws, iS] = left[ws, iS].float()
    items.head_mass[ws, iS] = torch.minimum(items.head_mass[ws, iS], items.mass[ws, iS])
    wg, ig = gone.view(A, I).nonzero(as_tuple=True)
    cr._remove(items, held, wg, ig)
    after = (items.mass.double() * items.alive).reshape(-1)
    lost = before - after                                                # what the pool really lost
    got = torch.where(valid, want / demand[key].clamp_min(1e-300) * lost[key], torch.zeros_like(want))
    species = comp_all[key] * got[:, None]
    if ledger is not None:
        ledger.book_out(torch.zeros(A, S, dtype=torch.float64, device=dev).index_add_(
            0, torch.arange(A, device=dev).repeat_interleave(I), comp_all * lost[:, None]))
    return {"species_kg": species, "cooked": cooked, "cook_gain": gain}


def ground_stock(pstate, geom, deposits_kg_m2=None) -> torch.Tensor:
    """[A, C, S] float64 kg/m^2 a stroke at the ground can break off (GROUND_STOCK_RULE): the patch's standing wood
    as wood, its litter as plant_fiber, and with ``deposits_kg_m2`` ([A, Sd] or [A, C, Sd], the deposits' column of
    the crust species, materials.CRUST_SPECIES order) its top SURFACE_LAYER_M."""
    from haishool.life9.planet import globe
    A, n = geom.A, geom.n
    C = n * n
    dev = geom.device
    out = torch.zeros(A, C, S, dtype=torch.float64, device=dev)
    cf_wood = mat.species_elements("wood")["C"]
    cf_fiber = mat.species_elements("plant_fiber")["C"]
    out[..., WOOD] = pstate.wood.reshape(A, C).double().clamp_min(0) / cf_wood
    out[..., FIBER] = pstate.litter.reshape(A, C).double().clamp_min(0) / cf_fiber
    if deposits_kg_m2 is not None:
        d = torch.as_tensor(deposits_kg_m2, dtype=torch.float64, device=dev)
        Sd = d.shape[-1]
        d = d[:, None, :].expand(A, C, Sd) if d.dim() == 2 else d
        out[..., :Sd] += d.clamp_min(0) * (SURFACE_LAYER_M / float(globe.ACCESSIBLE_DEPTH_M))
    return out


def take_ground(pstate, geom, ground_kg, prules=None) -> torch.Tensor:
    """Book what strokes broke off the ground (``ground_kg`` [A, C, S] kg, :func:`force`) out of the patch
    (GROUND_STOCK_RULE): wood by patch.take_plant (pool 'wood'; its nitrogen back to the cell's mineral pool),
    litter carbon out of the litter (``c_taken``). Returns the crust species' kg [A, S] float64 for the caller's
    deposit stock. The patch state's fields are rebound."""
    A, C, _ = ground_kg.shape
    dev = geom.device
    g = ground_kg.to(dev, torch.float64)
    a_idx = torch.arange(A, device=dev)[:, None].expand(A, C).reshape(-1)
    c_idx = torch.arange(C, device=dev)[None, :].expand(A, C).reshape(-1)
    wood = g[..., WOOD].reshape(-1)
    sel = wood > 0
    if bool(sel.any()):
        r = prules or pt.PatchRules()
        given = pt.take_plant(pstate, geom, a_idx[sel], c_idx[sel], wood[sel], pool="wood", rules=r)
        pt.give_nitrogen(pstate, geom, a_idx[sel], c_idx[sel], pt.plant_nitrogen(given, "wood", r))
    lit = g[..., FIBER].reshape(-1)
    sel = lit > 0
    if bool(sel.any()):
        cf = mat.species_elements("plant_fiber")["C"]
        idx = a_idx[sel] * C + c_idx[sel]
        new, removed, _ = pt._take(pstate.litter, idx, lit[sel] * cf / geom.cell_m2)
        pstate.litter = new
        pstate.c_taken = pstate.c_taken + pt._arena_sum(geom, removed)
    crust = g.sum(1)
    crust[..., WOOD] = 0
    crust[..., FIBER] = 0
    return crust


# ============================================================================== heat, fire, transforms, decay
def _image_near_fires(items, fires, L: float) -> None:
    """Loose items to (x, y, 0) in [0, L); those within FIRE_RADIUS_M of a fire across the boundary to that fire's
    image (PERIODIC_FIRE_RULE)."""
    r = float(cr.FIRE_RADIUS_M)
    ground = items.alive & (items.holder < 0)
    xy = torch.remainder(items.pos[..., :2], L)
    items.pos[..., :2] = torch.where(ground[..., None], xy, items.pos[..., :2])
    items.pos[..., 2] = torch.where(ground, torch.zeros_like(items.pos[..., 2]), items.pos[..., 2])
    fires.pos[..., :2] = torch.where(fires.alive[..., None], torch.remainder(fires.pos[..., :2], L), fires.pos[..., :2])
    edge = ground & ((xy < r) | (xy > L - r)).any(-1)
    if not bool((edge.any(1) & fires.alive.any(1)).any()):
        return
    a, i = edge.nonzero(as_tuple=True)
    fxy = fires.pos[a][..., :2]                                                    # [M, F, 2]
    d = wrap(xy[a, i][:, None, :] - fxy, L)
    dist = torch.where(fires.alive[a], d.norm(dim=-1), torch.full_like(d[..., 0], math.inf))
    dmin, f = dist.min(-1)
    near = dmin <= r
    m = torch.arange(a.numel(), device=a.device)[near]
    items.pos[a[near], i[near], :2] = fxy[m, f[near]] + d[m, f[near]]


def _cells(pos, L: float, n: int) -> torch.Tensor:
    """Flat fine cell ix n + iy of positions (as ``patch.cell_index``)."""
    c = torch.floor(torch.remainder(pos[..., :2].float(), L) / (L / n)).long().clamp(0, n - 1)
    return c[..., 0] * n + c[..., 1]


def _per_cell(where: torch.Tensor, amount: torch.Tensor, cells: int) -> torch.Tensor:
    """Sum amounts [A, X] (float64) into the flat fine cells where [A, X] of each arena -> [A, cells^2]."""
    A = where.shape[0]
    key = (torch.arange(A, device=where.device)[:, None] * cells * cells + where).reshape(-1)
    return torch.zeros(A * cells * cells, dtype=torch.float64, device=where.device).index_add_(
        0, key, amount.double().reshape(-1)).view(A, cells * cells)


def heat_step(items, fires, held, *, x_o2, air_k, L: float, dt_s: float, fire_air_k=None, ledger=None,
              cells: int | None = None, substeps: int = 48, mouth_pos=None, body_pos=None, alive=None) -> dict:
    """Fire and heat physics of one bout (crafting.fire_step on patch metres, PERIODIC_FIRE_RULE): beds burn, items
    in a bed heat, transform by temperature and atmosphere, char and feed it, every other item relaxes toward
    ``air_k`` ([A] or [A, I]) plus the fires' radiation (RADIANT_ITEM_RULE); fires below the fuel or O2 limits go
    out. With ``mouth_pos`` [A, N, 2+] the items held by bodies whose mouth is in a bed are in that fire
    (HELD_FIRE_RULE), and the bodies' exposure is returned: ``mouth_fire_k`` [A, N] (that fire's temperature at the
    start, else the air) and ``radiant_w_m2`` [A, N] at ``body_pos`` (default the mouth) from the bout's mean heat
    release. ``fire_air_k`` ([A] or [A, F]) defaults to air_k when that is per arena. Books ``air_kg`` and
    ``to_soil_kg`` in the ledger. Returns fire_step's flows plus ``burn_kg_s`` [A, F] (BURN_RATE_RULE); with
    ``cells`` also soil_c_kg and soil_n_kg [A, cells^2] (SOIL_CELL_RULE)."""
    A, I = items.alive.shape
    F = fires.alive.shape[1]
    dev = items.device
    if fire_air_k is None:
        t = torch.as_tensor(air_k)
        if t.dim() > 1:
            raise ValueError("pass fire_air_k when air_k is per item")
        fire_air_k = air_k
    x = cr._per_world(x_o2, A, dev)
    _image_near_fires(items, fires, L)
    temp0 = fires.temp_k.clone()
    hrr0 = cr.heat_release_w(fires, x)
    # ---- held items in a bed (HELD_FIRE_RULE)
    loosened = None
    f_m = None
    if mouth_pos is not None:
        Nb = held.shape[1]
        balive = torch.ones(A, Nb, dtype=torch.bool, device=dev) if alive is None else alive.bool()
        f_m = fire_at(fires, mouth_pos, balive, L)
        carry(items, held, mouth_pos, L)
        hk = (held >= 0) & (f_m >= 0)[..., None]
        a_, n_, k_ = hk.nonzero(as_tuple=True)
        if a_.numel():
            i_ = held[a_, n_, k_]
            fpos = fires.pos[a_, f_m[a_, n_], :2]
            items.pos[a_, i_, :2] = fpos + wrap(torch.remainder(mouth_pos[a_, n_, :2].float(), L) - fpos, L)
            items.pos[a_, i_, 2] = 0.0
            age = items.age_d[a_, i_].clone()
            loosened = (a_, n_, k_, i_, items.holder[a_, i_].clone(), age)
            items.holder[a_, i_] = -1
            items.age_d[a_, i_] = -1.0 - age                         # identity mark: spawn resets age_d to 0
    # ---- the fires' radiation on every item outside a bed (RADIANT_ITEM_RULE)
    t_items = _like(air_k, (A, I), dev)
    if F and bool(fires.alive.any()):
        q = radiant_flux(fires, hrr0, items.pos, L)
        h = cr._h_coef(t_items.double(), t_items.double())
        amb = (t_items.double() + q.double() / (4 * h)).float()
    else:
        amb = t_items
    alive0 = fires.alive.clone()
    bt = cr._bed_tables(dev)
    bed0 = fires.fuel_kg.double().clamp_min(0) * (bt["heat"] > 0) * alive0[..., None]
    q_bed = torch.where(bed0.sum(-1) > 0, (bed0 @ bt["heat"]) / bed0.sum(-1).clamp_min(1e-30),
                        torch.zeros_like(bed0[..., 0]))
    flows = cr.fire_step(items, fires, x, float(dt_s), ambient_item_k=amb, ambient_fire_k=fire_air_k, radius_m=1.0,
                         inv=held, substeps=substeps)
    if loosened is not None:
        a_, n_, k_, i_, code, age = loosened
        same = items.alive[a_, i_] & (items.age_d[a_, i_] < 0)
        items.holder[a_[same], i_[same]] = code[same]
        items.age_d[a_[same], i_[same]] = age[same]
        held[a_[~same], n_[~same], k_[~same]] = -1
    ground = items.alive & (items.holder < 0)
    items.pos[..., :2] = torch.where(ground[..., None], torch.remainder(items.pos[..., :2], L), items.pos[..., :2])
    if mouth_pos is not None:
        carry(items, held, mouth_pos, L)
        at = mouth_pos if body_pos is None else body_pos
        a_air = torch.as_tensor(air_k, dtype=torch.float32, device=dev)
        if a_air.dim() > 1:                                       # per item: the bodies meet the fires' air
            a_air = _like(fire_air_k, (A, max(F, 1)), dev).mean(1)
        t_body_air = _like(a_air, f_m.shape, dev)
        in_fire = temp0.gather(1, f_m.clamp_min(0)) if F else t_body_air
        flows["mouth_fire_k"] = torch.where(f_m >= 0, in_fire, t_body_air)
        flows["radiant_w_m2"] = _radiant(flows["fire_pos"], alive0, flows["mean_hrr_w"], at, L)
    flows["burn_kg_s"] = torch.where(q_bed > 0, flows["heat_j_fire"] / (q_bed.clamp_min(1e-30) * float(dt_s)),
                                     torch.zeros_like(q_bed))
    if ledger is not None:
        ledger.book_air(flows["air_kg"])
        ledger.book_out(flows["to_soil_kg"])
    if cells is not None:
        heat = flows["heat_j_fire"]
        weight = torch.where(heat.sum(1, keepdim=True) > 0, heat, alive0.double())
        weight = weight / weight.sum(1, keepdim=True).clamp_min(1e-300)
        el = mat.element_mass(flows["to_soil_kg"])                                     # [A, E]
        out = [_per_cell(_cells(flows["fire_pos"], L, cells), weight * el[:, e:e + 1], cells) for e in (_C, _N)]
        flows["soil_c_kg"], flows["soil_n_kg"] = out
    return flows


def decay_step(items, held, *, dt_days: float, L: float, cells: int | None = None, ledger=None) -> dict:
    """Rot organic items for dt_days (crafting.decay at each item's own temperature; an item below
    crafting.MIN_ITEM_KG rots away and leaves its grip). Books ``decayed_kg`` in the ledger. Returns decay's
    decayed_kg [A, S], decayed_el [A, E] and removed [A]; with ``cells`` (cells per side) also litter_c_kg and
    litter_n_kg [A, cells^2] (the carbon and nitrogen each fine cell received, for patch.add_litter)."""
    dev = items.device
    cn = mat.element_matrix(dev)[:, [_C, _N]]

    def item_cn():
        return (items.comp.double() * (items.mass.double() * items.alive)[..., None]) @ cn          # [A, I, 2]

    before = item_cn() if cells is not None else None
    where = _cells(items.pos, L, cells) if cells is not None else None
    out = cr.decay(items, float(dt_days), inv=held)
    if ledger is not None:
        ledger.book_out(out["decayed_kg"])
    if cells is not None:
        lost = before - item_cn()
        res = [_per_cell(where, lost[..., e], cells) for e in range(2)]
        out["litter_c_kg"], out["litter_n_kg"] = res
    return out


def _radiant(fire_pos, live, hrr_w, pos, L: float, within_m=None, chunk: int = 64) -> torch.Tensor:
    """[A, N] radiant flux (W/m^2) at pos [A, N, 2+] from fires at fire_pos [A, F, 2+] (live [A, F], hrr_w [A, F])."""
    within = float(cr.WARMTH_RADIUS_M if within_m is None else within_m)
    A, N = pos.shape[:2]
    F = fire_pos.shape[1]
    live = live & (hrr_w > 0)
    out = torch.zeros(A, N, device=pos.device)
    if F == 0 or N == 0 or not bool(live.any()):
        return out
    xy = pos[..., :2].float()
    for f0 in range(0, F, chunk):                                       # chunks of fire slots
        f1 = min(F, f0 + chunk)
        d = wrap(fire_pos[:, None, f0:f1, :2].float() - xy[:, :, None, :], L).norm(dim=-1)
        q = float(cr.RADIANT_FRACTION) * hrr_w[:, None, f0:f1] / (4 * math.pi * d.clamp_min(float(cr.NEAR_FIRE_M)) ** 2)
        out += torch.where(live[:, None, f0:f1] & (d <= within), q, torch.zeros_like(q)).sum(-1)
    return out


def radiant_flux(fires, hrr_w, pos, L: float, within_m=None, chunk: int = 64) -> torch.Tensor:
    """[A, N] radiant flux (W/m^2) at points pos [A, N, 2+] from the live fires, periodic: sum of
    crafting.RADIANT_FRACTION x HRR / (4 pi d^2), d at least crafting.NEAR_FIRE_M, within crafting.WARMTH_RADIUS_M
    (version 2's point-source model; hrr_w [A, F], e.g. heat_step's mean_hrr_w)."""
    return _radiant(fires.pos, fires.alive, hrr_w, pos, L, within_m, chunk)


def denatured(items) -> torch.Tensor:
    """[A, I] bool: the item has been at crafting.COOK_K or above (flesh heated past the denaturation of its
    proteins: cooked)."""
    return cr.cooked(items)


def digestible_gain(items) -> torch.Tensor:
    """[A, I] multiplier of the digestible food energy: materials' cook gain once denatured, else 1."""
    return cr.cook_factor(items)


def check(items, held) -> list[str]:
    """Problems between ``held`` and the pool's holder codes (empty = consistent)."""
    return cr.check_inventory(items, held)
