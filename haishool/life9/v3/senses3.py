"""Senses of life9 v3 (PLANET-V3-SPEC section 4): physical channels only. :func:`observe` -> x [A, N, IN_DIM].

Nothing here names what a thing is. The input carries angular sizes, contrasts, image motion, reflectances, received
light and sound, chemical concentrations, contact masses, hardness, temperatures and the body's own state; never
"this is something to eat", "this is a relative", "this is alive" or "this is stone". A body has to learn (or evolve)
what a percept means.

Layout (:data:`CHANNELS`, IN_DIM = 157). ``SECTORS`` = 8 sectors around the heading (sector 0 centred straight ahead,
then clockwise), each with :data:`SECTOR_CHANNELS` (7 vision + D = 8 call features + 1 broadband noise); then the
smell of each volatile field (concentration and gradient along the heading), touch under the mouth, the two held items
and the interoception (:data:`INTERO_SLICE`, in ``brain3.INTERO_NAMES`` order, the modulator's input). Dead slots get
zeros. Every channel is dimensionless and of order 1: the sector, touch and held channels and the coded interoception
(temperature, light, speed) lie in [-1, 1]; the smell codes and the reserve, water and gut ratios are of order 1.

Objects. Bodies and the items lying on the ground are one kind of thing to the senses: an object is a sphere of its
volume with a reflectance (plus its own glow), a speed over the ground, a mass, a hardness and a temperature. No
channel tells a body from a stone: a still body of a stone's size and reflectance looks like that stone; what differs
is only what physics gives (it moves, it calls, it is warm and soft to the touch).

Conventions. Positions are metres (x east, y north) in an arena of ``patch`` (periodic). A heading h points along
(cos h, sin h): 0 is east, angles grow counter-clockwise (:func:`heading_vector`); a target's sector comes from its
clockwise bearing relative to the heading (:func:`sector_of`). A body is a sphere of its mass at tissue density
(:func:`body_radius`); eyes and ears sit at the top of it, the mouth at its front (:func:`mouth_point`). A body on
water floats: its eyes are counted from the water surface. Items denser than water lie on the bed under it.

Vision (an eye of the ``eye`` gene's mass, as one sphere at tissue density, diameter d):

* optics: pupil D_p = ``pupil_share`` d; resolution theta_res = max(1.22 lambda / D_p (diffraction), 2 x receptor
  spacing / (``nodal_share`` d) (Nyquist sampling of the receptor mosaic)); photon catch per unit irradiance
  G = (pi D_p^2 / 4) x integration time x quantum efficiency x (1 - damage) x lambda / (h c) (:func:`eye_optics`).
* light: the visible illuminance of each fine cell this bout (the sun angle per bout, slope and aspect from
  ``patch.light`` x the PAR share, plus the moonless night sky; :func:`scene_from_patch`), plus fires (blocked by the
  terrain like sight).
* detection: a target of solid angle Omega and luminance L_t against a background L_b collects dN = |L_t - L_b|
  Omega G photons more than its background, whose photons are counted over the larger of the target and the blur
  spot (Omega_res = pi theta_res^2 / 4): N_b = L_b max(Omega, Omega_res) G. It is seen when the photon noise allows it
  (Rose 1948: dN / sqrt(N_b + dN) >= k), its contrast diluted over the blur spot clears the eye's intrinsic contrast
  threshold (|L_t - L_b| Omega / (L_b max(Omega, Omega_res)) >= ``c_min``) and the terrain line of sight
  (``patch.line_of_sight``: fine elevation and the real-radius bulge) is clear.
* per sector: ``size``, ``contrast`` and ``motion`` of the nearest seen object: size n / (1 + n) of its angular
  diameter in resolution elements n, contrast the Michelson contrast (L_t - L_b) / (L_t + L_b) against the ground
  behind it (reflectances from ``optics``), motion m / (1 + m) of the resolution elements its image crosses per
  integration time (its speed over the ground, which is the background); ``cover`` the summed angular width of the
  seen objects over the sector width (clamped to 1); ``flow`` the summed image motion of the seen objects, x / (1 + x);
  ``ground`` the mean reflectance of the ground (or water surface) seen along the sector centre line (samples at 1/2 to
  8 fine cells, each weighted by its photon-limited visibility N / (N + k^2), unseen samples 0); ``emitted`` the
  light received from fires and glowing objects: log10(1 + x) / ``emit_span_dec`` of x, the sources' photons over the
  background photons behind them (counted over the larger of the source's image and the blur spot), summing the
  sources that pass the same Rose and contrast tests.

Hearing (an ear of the ``ear`` gene's mass): a call radiates acoustic power loudness x vocal efficiency x muscle
specific power x voice organ mass (:func:`call_acoustic_power`) at the quarter-wave resonance c / 4 l of the voice
organ, a tube of its volume and aspect ratio ``voice_aspect`` (:func:`call_frequency`); locomotion radiates noise
proportional to mass x speed^2 (:func:`footfall_power`). Intensity at the listener: P / (4 pi r^2) (spherical
spreading) x 10^(-alpha r / 10) with alpha the atmospheric absorption of ISO 9613-1 (classical plus the O2 and N2
relaxations, from the arena's pressure, temperature, humidity and composition; :func:`absorption_db_m`), x 0.25
(pressure x 0.5) when terrain blocks the line; nothing below 1 kPa of air. The ear's threshold intensity falls with
its collecting area (:func:`ear_threshold`). Per sector: the D call features, each call weighted by its sensation
level over ``level_full_db`` (their weighted sum, or their weighted mean once the summed weight exceeds 1), and
``noise``, the sensation level of everything heard there (calls and locomotion).

Smell (chemistry; ``patch.Volatiles``): for each field the concentration code log10(1 + c / c_th) / span with c_th
the field's odour threshold as a number density (an animal nose: ``nose_factor`` x the human-panel thresholds), and
its gradient along the heading: the code ahead minus the code behind, half a fine cell each way (the field's own
resolution).

Touch, under the mouth: ``plant`` the share of the ground covered by leaf (``patch.fapar`` of the soft tissue mass),
``damp`` the soil water over the bucket's capacity (1 under water), ``wet`` the free water depth (ponds, soil water
above the bucket, the sea) relative to the body's radius, x / (1 + x); the nearest object in contact (smallest
surface gap): ``mass`` its mass ratio x / (1 + x), ``hard`` its hardness (Mohs of what pressing feels: an item's
``materials.props`` ``tool_hardness``, a body's hide) / 10 and ``temp`` its temperature relative to the body,
tanh(dT / 10 K). Held items (two grips): mass ratio x / (1 + x), hardness / 10, sharpness, temperature tanh(dT / 10 K).

Interoception (``brain3.INTERO_NAMES``): reserve / capacity, water / normal, body temperature placed symmetrically in
the physical range of life ((T_b - 294.5 K) / 23.5 K: -1 at freezing, +1 at denaturation), damage, gut fill, light
level (Weber-Fechner: log10(1 + E / E_night) / 8 decades) and own speed as Fr / (1 + Fr), Fr the Froude speed
v / sqrt(g x body length).

Neighbours. Two spatial hashes, both scanned nearest cell first with at most ``max_scan`` candidates: the fine one
(``patch.neighbours`` for bodies, :func:`cross_neighbours` for items; 3 x 3 fine cells, the K = 32 nearest) and a
coarse one of ``far_cells`` x ``far_cells`` fine cells (3 x 3 coarse cells, ``far_k`` candidates) holding only the
targets whose physical range could reach past the fine block: an upper bound of the distance at which the arena's best
eye or ear could detect them (:func:`vision_reach`, :func:`sound_reach`), ranked by distance over that range. A far
candidate counts only when it is detected on its own. Range is therefore physical out to at least ``far_cells`` fine
cells (128 m on the default grid); ``details`` reports the targets whose range is longer (``beyond``). Fires: at each
body (and item) the fires that matter (at least ``c_min`` of the sky's light there, or seen by the body's eye) are
ranked per sector by the light they shed on it, nearest first (streamed over the fires), and line-tested ``fire_k``
ranks at a time until the sector has a clear one (at most ``fire_rounds`` rounds); the untested ones still light it,
unoccluded, and are left out of the emitted sum. There are no O(N^2) pair sets and no Python loops over bodies, items or cells.
Randomness only in the neighbour scans' offsets (on the generator passed in); the CPU result is exactly reproducible.

Inputs are plain tensors (:class:`Bodies`, :class:`Items`, :class:`Fires`, :class:`Scene`, :class:`Air`), so this
module does not import ``body.py``. Every constant carries a provenance entry in ``PROVENANCE``.
"""
from __future__ import annotations

import math
from dataclasses import MISSING, dataclass, fields

import torch

from haishool.life9.planet import constants as K
from haishool.life9.planet import materials

from . import brain3, optics
from . import patch as pt

# ============================================================================================ layout
SECTORS = 8
SECTOR_WIDTH = 2 * math.pi / SECTORS
VOCAL = brain3.VOCAL_DIMS
GRIPS = 2
SMELL_FIELDS = ("plant", "carcass", "smoke", "co2")            # the fields of patch.Volatiles
VISION_CHANNELS = ("size", "contrast", "motion", "cover", "flow", "ground", "emitted")
HEARING_CHANNELS = (*(f"call{d}" for d in range(VOCAL)), "noise")
SECTOR_CHANNELS = VISION_CHANNELS + HEARING_CHANNELS
SECTOR_DIM = len(SECTOR_CHANNELS)
TOUCH_CHANNELS = ("plant", "damp", "wet", "mass", "hard", "temp")
HELD_CHANNELS = ("mass", "hard", "sharp", "temp")
INTERO_NAMES = brain3.INTERO_NAMES

CHANNELS = (tuple(f"s{s}_{c}" for s in range(SECTORS) for c in SECTOR_CHANNELS)
            + tuple(name for f in SMELL_FIELDS for name in (f"smell_{f}", f"smell_{f}_grad"))
            + tuple(f"touch_{c}" for c in TOUCH_CHANNELS)
            + tuple(f"held{g}_{c}" for g in range(GRIPS) for c in HELD_CHANNELS)
            + tuple(f"intero_{c}" for c in INTERO_NAMES))
IN_DIM = len(CHANNELS)
SMELL0 = SECTORS * SECTOR_DIM
TOUCH0 = SMELL0 + 2 * len(SMELL_FIELDS)
HELD0 = TOUCH0 + len(TOUCH_CHANNELS)
INTERO0 = HELD0 + GRIPS * len(HELD_CHANNELS)
INTERO_SLICE = slice(INTERO0, INTERO0 + len(INTERO_NAMES))
assert INTERO_SLICE.stop == IN_DIM

# ============================================================================================ physical constants
TISSUE_DENSITY = 1050.0               # kg/m^3 (PLANET-V3-SPEC 3)
T_COLD_DEATH_K = 271.0                # K, tissue freezes (PLANET-V3-SPEC 3)
T_HOT_DEATH_K = 318.0                 # K, protein denaturation (PLANET-V3-SPEC 3)
T_LIFE_MID_K = 0.5 * (T_COLD_DEATH_K + T_HOT_DEATH_K)
T_LIFE_HALF_K = 0.5 * (T_HOT_DEATH_K - T_COLD_DEATH_K)
MAX_EFFICACY_LM_W = 683.0             # lm/W at 540 THz (the SI definition of the candela)
V_INTEGRAL_M = 106.86e-9              # m, the integral of the CIE 1924 photopic luminosity function V(lambda)
VISIBLE_BAND_M = optics.VISIBLE_BAND_M
#: lm per W of a spectrally flat 400-700 nm light: 683 x (integral of V) / band width = 243 lm/W
LUMEN_PER_W = MAX_EFFICACY_LM_W * V_INTEGRAL_M / (VISIBLE_BAND_M[1] - VISIBLE_BAND_M[0])
STP_T_K = 298.15                      # K, room temperature of the odour-threshold measurements
SMELL_FORMULA = {"plant": "C5H8", "carcass": "C2H6S2", "smoke": "CO", "co2": "CO2"}
HUMAN_PANEL_FIELDS = ("plant", "carcass", "smoke")   # thresholds measured on human panels (nose_factor applies)
ISO_P_REF_PA = 101325.0               # ISO 9613-1 reference pressure
ISO_T0_K = 293.15                     # ISO 9613-1 reference temperature (20 C)
ISO_T01_K = 273.16                    # ISO 9613-1: the triple point of water


def _dry_air_reference():
    """Mole fractions of O2 and N2, molar mass and gamma of Earth's dry air (constants.EARTH_AIR_MOLE_FRACTION):
    the air the ISO 9613-1 coefficients belong to."""
    x = K.EARTH_AIR_MOLE_FRACTION
    tot = sum(x.values())
    m = sum(v / tot * K.molar_mass(g) for g, v in x.items())
    cp = sum(v / tot * K.CP_MOLAR_298[g] for g, v in x.items())
    return x["O2"] / tot, x["N2"] / tot, m, cp / (cp - K.R_GAS)


ISO_X_O2, ISO_X_N2, ISO_MOLAR_KG, ISO_GAMMA = _dry_air_reference()


def _hide_hardness() -> float:
    c = torch.zeros(1, materials.S)
    c[0, materials.IDX["hide"]] = 1.0
    return float(materials.props(c, torch.ones(1))["tool_hardness"][0])


#: Mohs hardness of a body's surface to the touch: its hide (materials.props of "hide")
BODY_HARDNESS = _hide_hardness()


@dataclass(frozen=True)
class SenseRules:
    """Modelling choices and reference values of the senses (sources in ``PROVENANCE``)."""
    K: int = 32
    max_scan: int = 128
    far_cells: int = 8
    far_k: int = 16
    fire_k: int = 2
    fire_rounds: int = 3
    fire_chunk: int = 16
    eye_height_radii: float = 2.0
    mouth_reach_radii: float = 1.0
    pupil_share: float = 0.35
    nodal_share: float = 0.7
    receptor_m: float = 2.5e-6
    wavelength_m: float = 555e-9
    integration_s: float = 0.1
    quantum_eff: float = 0.1
    rose_k: float = 3.0
    c_min: float = 0.02
    night_sky_lux: float = 0.002
    light_span_dec: float = 8.0
    emit_span_dec: float = 8.0
    fur_cover_m: float = 1e-3
    water_kd_m: float = 0.5
    ray_step_cells: float = 0.5
    ray_steps: int = 16
    ray_fields: tuple = (1, 2, 4, 8, 16)
    ground_tolerance_m: float = 1e-3
    vocal_eff: float = 0.005
    voice_aspect: float = 4.4
    footfall_db: float = 45.0
    footfall_kg: float = 70.0
    footfall_m_s: float = 1.4
    noise_hz: float = 1000.0
    hear_i0_w_m2: float = 1e-12
    ear_ref_kg: float = 0.02
    level_full_db: float = 60.0
    p_min_pa: float = 1000.0
    occlusion: float = 0.5
    smell_plant_ppm: float = 0.048
    smell_carcass_ppm: float = 0.0022
    smell_smoke_ppm: float = 2.0
    smell_co2_ppm: float = 100.0
    nose_factor: float = 0.01
    smell_span_dec: float = 3.0
    thermal_k: float = 10.0
    hardness_full: float = 10.0


R_, D_, N_, C_ = "reference", "derived", "new_rule", "chain"
RN_ = "reference+new_rule"
PROVENANCE = {
    # ---- layout and conventions
    "SECTORS": (N_, "PLANET-V3-SPEC 4: S = 8 sectors around the heading, sector 0 centred ahead, then clockwise "
                    "(version 2's layout)"),
    "SECTOR_WIDTH": (D_, "2 pi / SECTORS"),
    "VOCAL": (N_, "PLANET-V3-SPEC 4: calls are D = 8 dimensional vectors (brain3.VOCAL_DIMS)"),
    "GRIPS": (N_, "PLANET-V3-SPEC 3: two grips (held [A, N, 2])"),
    "SMELL_FIELDS": (N_, "the volatile fields of patch.Volatiles (chemistry: plant volatiles, sulfur volatiles of "
                         "rotting animal matter, smoke CO, CO2 from point sources); channel names are for people, "
                         "the body only gets the numbers"),
    "layout": (N_, f"{SECTORS} sectors x {SECTOR_DIM} channels ({len(VISION_CHANNELS)} vision, {VOCAL} call "
                   f"features, 1 noise) + {2 * len(SMELL_FIELDS)} smell + {len(TOUCH_CHANNELS)} touch + "
                   f"{GRIPS * len(HELD_CHANNELS)} held + {len(INTERO_NAMES)} interoception = IN_DIM {IN_DIM}; "
                   "no channel names a class of thing"),
    "objects": (N_, "PLANET-V3-SPEC 0 over section 4's channel list: bodies and items on the ground share the object "
                    "channels (nearest seen object's size, contrast and image motion, summed cover and summed image "
                    "motion per sector; touch of the object in contact by mass ratio, hardness and temperature), so "
                    "no channel marks animate against inanimate; section 4's 'nearest moving body' and separate "
                    "item channels would be an engine-supplied class label"),
    "heading": (N_, "a heading h points along (cos h, sin h) in (east, north): 0 east, counter-clockwise; body.py and "
                    "manipulate.py must share it (heading_vector)"),
    "sector_of": (N_, "sector = floor(((h - bearing) + w/2) mod 2 pi / w), bearing = atan2(dy, dx): clockwise from "
                      "ahead, as version 2"),
    # ---- body geometry
    "TISSUE_DENSITY": (R_, "1,050 kg/m^3, soft tissue (PLANET-V3-SPEC 3; ICRP 1975 reference man tissue 1.03-1.06)"),
    "body_radius": (D_, "a body seen and heard as the sphere of its mass at tissue density: r = (3 M / 4 pi rho)^(1/3) "
                        "(the spec's ellipsoid has the same volume; its axis ratios are body.py's, an optional "
                        "radius_m input overrides)"),
    "eye_height_radii": (N_, "eyes and ears at the top of the body's sphere, 2 r above the ground (or the water "
                             "surface for a floating body)"),
    "mouth_reach_radii": (N_, "the mouth at the front of the sphere, pos + r (cos h, sin h) (or body.py's mouth_pos); "
                              "it touches an object whose surface is within one body radius of it (or body.py's "
                              "reach_m): manipulate.REACH_RULE, so touch reports what the mouth can act on"),
    "fire_area": (N_, "a fire pool (version 2's FirePool) carries no flame size: a fire radiates from its bed's area "
                      "pi R^2, R = crafting.FIRE_RADIUS_M (0.5 m, a hearth), unless the caller passes area_m2"),
    "water_surface": (D_, "a body on water floats: its eye height is counted from the water surface: free water depth "
                          "(pond / 1000 kg/m^3 plus the sea depth max(0, sea level - elevation)) interpolated "
                          "bilinearly like the elevation, so eye = bilinear(elevation) + bilinear(depth) + 2 r, the "
                          "same height in the line-of-sight test and the ground rays"),
    "item_buoyancy": (D_, "an item denser than water (materials.props density > 1,000 kg/m^3) lies on the bed under "
                          "the water (sight height 2 r above the bed), a lighter one floats (2 r above the surface)"),
    "water_kd_m": (RN_, "diffuse attenuation of visible light in natural waters K_d = 0.03-0.1 m^-1 (clear ocean), "
                        "0.1-1 (coastal water, clear lakes), 1-10 (turbid ponds) (Kirk 2011, Light and "
                        "Photosynthesis in Aquatic Ecosystems, 3rd ed., ch. 6); new_rule 0.5 m^-1: a submerged item "
                        "and the bed behind it are seen through exp(-2 K_d depth) (down and back up)"),
    "T_COLD_DEATH_K": (R_, "271 K, tissue freezing (PLANET-V3-SPEC 3)"),
    "T_HOT_DEATH_K": (R_, "318 K, protein denaturation (PLANET-V3-SPEC 3)"),
    "T_LIFE_MID_K": (D_, "the middle of the physical range of life, (271 + 318) / 2 = 294.5 K"),
    "T_LIFE_HALF_K": (D_, "half the physical range of life, 23.5 K"),
    "damage_senses": (N_, "PLANET-V3-SPEC 3, damage reduces senses: the eye's quantum catch, the ear's sensitivity "
                          "and the nose's sensitivity scale with the intact share (1 - damage); touch and "
                          "interoception are not reduced"),
    # ---- vision
    "eye_optics": (D_, "the eye gene's mass as one spherical eye at tissue density (paired eyes split it: the "
                       "collecting area changes by 2^(1/3)), diameter d = (6 m / pi rho)^(1/3); pupil D_p = "
                       "pupil_share d; theta_res = max(1.22 lambda / D_p, 2 receptor_m / (nodal_share d)); photon "
                       "catch G = pi D_p^2 / 4 x integration_s x quantum_eff x (1 - damage) x lambda / (h c)"),
    "pupil_share": (RN_, "vertebrate eyes work at f-numbers of about 0.9-2.5 (Land & Nilsson 2012, Animal Eyes, 2nd "
                         "ed., ch. 4); new_rule: f = 2 at a posterior nodal distance of 0.7 d, D_p = 0.35 d (a human "
                         "eye of 24 mm: 8.4 mm, the dark-adapted pupil)"),
    "nodal_share": (R_, "posterior nodal distance of vertebrate eyes about 0.6-0.7 of the axial length (human 17 of "
                        "24 mm; Land & Nilsson 2012)"),
    "airy": (R_, "1.22 lambda / D, the Rayleigh resolution of a circular aperture (the first zero of the Airy "
                 "pattern; Born & Wolf, Principles of Optics, ch. 8)"),
    "receptor_m": (R_, "foveal cone spacing 2.5 um in the human retina (Curcio, Sloan, Kalina & Hendrickson 1990, "
                       "J. Comp. Neurol. 292, 497); the Nyquist limit 2 s / f gives 1 arcmin for a 24 mm eye. "
                       "Receptors cannot be much finer (waveguide limit 1-2 um, Snyder 1977)"),
    "wavelength_m": (R_, "555 nm, the peak of the photopic luminosity function (CIE 1924), the middle of the "
                         "visible band; used for photon energy and diffraction"),
    "integration_s": (R_, "photoreceptor integration time about 0.1 s (Bloch's law critical duration; Barlow 1958, "
                          "J. Physiol. 141, 337)"),
    "quantum_eff": (R_, "overall quantum efficiency of a vertebrate eye about 0.1 (Rose 1948, J. Opt. Soc. Am. 38, "
                        "196; Barlow 1956: 0.05-0.1 for the dark-adapted human eye)"),
    "rose_k": (R_, "Rose's criterion: a target is detected when its photon signal exceeds about 3-5 times the "
                   "photon noise (Rose 1948; Burgess 1999, J. Opt. Soc. Am. A 16, 633); k = 3"),
    "c_min": (RN_, "the intrinsic (neural) contrast threshold of an eye in bright light: humans 0.5-2 % Weber "
                   "contrast for large targets (Blackwell 1946, J. Opt. Soc. Am. 36, 624), vertebrates' peak contrast "
                   "sensitivity 10-100, i.e. 1-10 % (cat: Uhlrich, Essock & Lehmkuhle 1981, Behav. Brain Res. 2, "
                   "291; rat: Prusky et al. 2000, Behav. Brain Res. 116, 135; birds: Ghim & Hodos 2006, Vision Res. "
                   "46, 1242); new_rule 2 %, the photon noise of small eyes adds to it (Rose test)"),
    "rose_test": (D_, "dN = |L_t - L_b| Omega G, N_b = L_b max(Omega, pi theta_res^2 / 4) G; seen when dN / sqrt(N_b "
                      "+ dN) >= rose_k, |L_t - L_b| Omega >= c_min L_b max(Omega, Omega_res) and the line of sight is "
                      "clear; L = (rho E + M_glow) / pi (Lambertian), Omega = 2 pi s^2 / (1 + sqrt(1 - s^2)), s = "
                      "r/d, the solid angle of a sphere (the cancellation-free form of 2 pi (1 - sqrt(1 - s^2)))"),
    "angular_size": (D_, "angular diameter theta = 2 asin(min(1, r / d)); channel n / (1 + n), n = theta / theta_res "
                         "(resolution elements covered, the Naka-Rushton saturation of a receptor array, Naka & "
                         "Rushton 1966, J. Physiol. 185, 536)"),
    "contrast": (D_, "Michelson contrast (L_t - L_b) / (L_t + L_b) against the ground at the target's fine cell under "
                     "the same light, so the reflectance table (optics) decides it: green plants and water dark, "
                     "sand and snow bright"),
    "motion": (D_, "image motion m = speed x integration_s / (d theta_res): resolution elements the image crosses per "
                   "integration time (speed over the ground, which is the background); channel m / (1 + m), a value "
                   "not a gate"),
    "cover": (D_, "summed angular widths of the seen objects in the sector / the sector width, clamped to 1"),
    "flow": (D_, "summed image motion m of the seen objects in the sector, x / (1 + x): what an array of elementary "
                 "motion detectors adds up"),
    "body_surface": (RN_, "a body's visible reflectance: optics tissue:skin under its coat, covered by fur as "
                          "1 - exp(-fur_m / fur_cover_m) (optics tissue:fur); an optional surface_vis input "
                          "overrides"),
    "fur_cover_m": (N_, "a coat of 1 mm hides about 63 % of the skin from view"),
    "ground_rays": (N_, "numerics: the visible surface (ground, or the water surface: elevation + free water depth, "
                        "bilinear) is sampled every ray_step_cells (half a fine cell) out to ray_steps (8 cells) "
                        "along each sector's centre line; a sample is seen when its elevation angle (z + tolerance - "
                        "z_eye) / t - t / 2R clears the running horizon of the samples before it (the line-of-sight "
                        "test with the samples shared along the line); the ground channel uses the samples at 1/2, 1, "
                        "2, 4 and 8 cells (ray_fields)"),
    "ray_step_cells": (N_, "see ground_rays"), "ray_steps": (N_, "see ground_rays"),
    "ray_fields": (N_, "see ground_rays"),
    "ground_tolerance_m": (N_, "numerics: a ground sample may graze the horizon by 1 mm (patch.sight_tolerance_m)"),
    "ground_visibility": (D_, "a ground sample (one fine cell of surface) subtends Omega = dx^2 / n_up x max(0, n . u) "
                              "/ R^2 (its projected solid angle: surface normal n from patch.normal, u the unit "
                              "vector to the eye at distance R); its photon catch N = rho E / pi x Omega x G and its "
                              "visibility N / (N + k^2) (the Rose test for a contrast of 1)"),
    "night_sky_lux": (R_, "a clear moonless night sky with airglow gives about 0.002 lux at the ground (Schlyter, "
                          "Radiometry and photometry in astronomy, table of natural illuminances; the optics "
                          "module's full moon is about 0.25 lux); added to every fine cell's visible light"),
    "MAX_EFFICACY_LM_W": (R_, "683 lm/W at 540 THz, the SI definition of the candela (CGPM 1979)"),
    "V_INTEGRAL_M": (R_, "the CIE 1924 photopic luminosity function integrates to 106.86 nm"),
    "LUMEN_PER_W": (D_, "683 x 106.86 nm / 300 nm = 243 lm per W of a flat 400-700 nm spectrum (sunlight: 1e5 lux "
                        "is about 410 W/m^2 of visible light)"),
    "VISIBLE_BAND_M": (R_, "400-700 nm (optics.VISIBLE_BAND_M, the PAR band)"),
    "visible_light": (D_, "visible illuminance of a fine cell = patch.light's bout shortwave x the PAR share "
                          "(400-700 nm, the same band) + the night sky (night_sky_lux / LUMEN_PER_W)"),
    "light_span_dec": (D_, "light level = log10(1 + E / E_night) / 8: the noon sun over the moonless night sky is "
                           "about 10^7.7 (Weber-Fechner coding of intensity, Fechner 1860)"),
    "emit_span_dec": (D_, "emitted = log10(1 + x) / 8, clamped to 1: the same 8 decades as the light level, so a "
                          "beacon's distance stays readable at night (a flame over the night ground is about 10^5-10^6 "
                          "in luminance)"),
    "fire_light": (D_, "a fire radiates visible power optics.visible_exitance(T, FLAME_EMISSIVITY) x its radiating "
                       "area (Fires.area_m2), isotropically from a sphere of that area (r_f = sqrt(area / 4 pi), "
                       "luminance M / pi): irradiance P / (4 pi max(d, r_f)^2) on the bodies and items whose line to "
                       "it is clear (from a body's eye or an item's top to the flame centre; the ground samples get "
                       "only the sky's light). A fire under c_min of the sky's light at a point is not line-tested "
                       "there: it changes the light by less than an eye's contrast threshold"),
    "emitted": (D_, "per sector x = sum over detected sources of N_s / N_b: source photons N_s = irradiance x G over "
                    "the background photons behind the source N_b = L_b max(Omega_source, Omega_res) G (a fire's "
                    "image is the sphere r_f, a glowing object its own), so a resolved source gives its luminance "
                    "over the background's; a source counts when N_s / sqrt(N_s + N_b) >= rose_k, N_s >= c_min N_b "
                    "and its line of sight is clear"),
    "fire_k": (N_, "numerics (a cost bound on the line-of-sight tests, not on fires): the fires that matter at a "
                   "point (irradiance at least c_min of the sky's, or seen by its eye: the unoccluded Rose and "
                   "contrast tests) are ranked per sector by irradiance, nearest first (streamed over the fires "
                   "fire_chunk at a time); each round line-tests the next fire_k ranks of the sectors that have no "
                   "clear source yet, so the nearest visible fire of a sector is found unless fire_k x fire_rounds "
                   "nearer ones are all hidden. The untested ones are counted in details['fires_culled'], still light "
                   "the point unoccluded and are left out of the emitted sum; items are lit the same way"),
    "fire_rounds": (N_, "numerics: at most 3 rounds of fire_k line tests per sector (at most 48 fire sight lines per "
                        "body, whatever the number of fires)"),
    "fire_chunk": (N_, "numerics: fires streamed 16 at a time (memory bound)"),
    "glow": (D_, "an item at temperature T radiates M = optics.visible_exitance(T, emissivity) from its surface (a "
                 "sphere of its volume, emissivity optics.mix of its composition): irradiance M r^2 / d^2; its "
                 "luminance adds M / pi to the reflected rho E / pi"),
    "item_props": (D_, "items from materials.props of their composition: radius of the sphere of their volume, "
                       "density, tool_hardness (scratch hardness x (1 - granular): what pressing and striking feel), "
                       "reflectance and emissivity optics.mix"),
    "BODY_HARDNESS": (D_, "a body's surface feels like hide: materials.props tool_hardness of 'hide' (Mohs 0.5)"),
    # ---- hearing
    "call_power": (RN_, "acoustic power of a call = loudness x vocal_eff x muscle (W per kg) x voice organ mass: "
                        "the sound organ is muscle of the body's own specific power"),
    "vocal_eff": (RN_, "radiated over mechanical power of sound production 0.1-2 % (Titze 1994, Principles of Voice "
                       "Production, ch. 9; Brackenbury 1979 for birds); new_rule 0.5 %. A 30 kg body with 3 g/kg of "
                       "voice organ at 200 W/kg calls at about 100 dB at 1 m, version 2's reference loud call"),
    "call_frequency": (RN_, "the dominant frequency is the quarter-wave resonance c / (4 l) of the vocal tract (F1 = c "
                            "/ 4 VTL, Fitch 1997, J. Acoust. Soc. Am. 102, 1213); new_rule: the voice organ is a "
                            "tube of its own volume V = m / rho and aspect ratio a = length / width, so l = (4 a^2 V / "
                            "pi)^(1/3)"),
    "voice_aspect": (RN_, "a vocal tract is a tube 4-6 times longer than wide (human about 17 cm long; Fitch & Giedd "
                          "1999, J. Acoust. Soc. Am. 106, 1511); new_rule a = 4.4, calibrated so that the loudness "
                          "calibration's 3 g/kg voice organ gives a 70 kg body a 17 cm tract (F1 about 500 Hz) and a "
                          "30 kg one 12.8 cm (children of 30 kg: about 13 cm, Fitch & Giedd 1999)"),
    "footfall": (RN_, "locomotion noise acoustic power = kappa M v^2 (PLANET-V3-SPEC 4: noise proportional to speed^2 "
                      "x mass), kappa calibrated on footsteps of about 45 dB at 1 m for 70 kg at 1.4 m/s (walking "
                      "on hard ground 40-60 dB, on soft ground less)"),
    "footfall_db": (RN_, "see footfall"), "footfall_kg": (RN_, "see footfall"), "footfall_m_s": (RN_, "see footfall"),
    "noise_hz": (N_, "locomotion noise is broadband; its absorption is taken at 1 kHz"),
    "absorption": (R_, "ISO 9613-1:1993 eq. 5: alpha = 8.686 f^2 [1.84e-11 (p/p_r)^-1 (T/T_0)^(1/2) + (T/T_0)^(-5/2) "
                       "(0.01275 e^(-2239.1/T) / (f_rO + f^2/f_rO) + 0.1068 e^(-3352/T) / (f_rN + f^2/f_rN))] dB/m, "
                       "f_rO = (p/p_r)(24 + 4.04e4 h (0.02 + h) / (0.391 + h)), f_rN = (p/p_r)(T/T_0)^(-1/2) (9 + 280 "
                       "h e^(-4.170 ((T/T_0)^(-1/3) - 1))), h the molar concentration of water vapour in %; it "
                       "reproduces ISO 9613-2 table 2 within 2 %"),
    "absorption_other_air": (N_, "for another planet's air: each relaxation term scales with its gas's mole fraction "
                                 "(dry basis) over Earth's, and the classical term with 1 / (rho c^3) at the same p "
                                 "and T, (gamma_air / gamma)^1.5 (M / M_air)^0.5 (viscosity differences and CO2's own "
                                 "vibrational relaxation are not included)"),
    "ISO_P_REF_PA": (R_, "ISO 9613-1 reference pressure 101.325 kPa"),
    "ISO_T0_K": (R_, "ISO 9613-1 reference temperature 293.15 K"),
    "ISO_T01_K": (R_, "ISO 9613-1: triple-point isotherm 273.16 K of the saturation vapour pressure formula"),
    "ISO_X_O2": (D_, "the O2 mole fraction of Earth's dry air (constants.EARTH_AIR_MOLE_FRACTION), the air of the ISO "
                     "coefficients"),
    "ISO_X_N2": (D_, "the N2 mole fraction of Earth's dry air"),
    "ISO_MOLAR_KG": (D_, "the molar mass of Earth's dry air"),
    "ISO_GAMMA": (D_, "the heat capacity ratio of Earth's dry air (constants.CP_MOLAR_298)"),
    "saturation_vapour": (R_, "ISO 9613-1 annex B: p_sat / p_r = 10^C, C = -6.8346 (T_01 / T)^1.261 + 4.6151"),
    "spreading": (D_, "spherical spreading: intensity P / (4 pi r^2), r at least the source's radius"),
    "occlusion": (N_, "PLANET-V3-SPEC 4: terrain between halves the sound pressure (intensity x 0.25), using the "
                      "same terrain line of sight as vision"),
    "p_min_pa": (N_, "PLANET-V3-SPEC 4: nothing is heard below 1 kPa of air"),
    "air": (D_, "air density p M / (R T) and sound speed sqrt(gamma R T / M), gamma = Cp / (Cp - R) of the mixture "
                "(constants.CP_MOLAR_298, NIST-JANAF); the caller passes each arena's air (observe has no default)"),
    "hear_i0_w_m2": (R_, "1e-12 W/m^2 = 0 dB, the human threshold near 1-4 kHz (ISO 226:2003)"),
    "ear_threshold": (RN_, "the threshold intensity scales inversely with the ear's collecting area (a fixed "
                           "receptor noise power over the area): I_th = I0 x (ear_ref_kg / m_ear)^(2/3) / (1 - "
                           "damage)"),
    "ear_ref_kg": (N_, "the human ear (two pinnae, middle and inner ears) taken as 20 g for the 0 dB threshold"),
    "level_full_db": (N_, "version 2's weight: sensation level 10 log10(I / I_th) / 60 dB, clamped to [0, 1]"),
    "calls_heard": (N_, "version 2's rule: the sector's calls are the weighted sum of the vocal vectors while the "
                        "summed weight is at most 1, else the weighted mean, so a crowd's calls keep their content"),
    # ---- smell
    "smell_plant_ppm": (R_, "human odour threshold of isoprene 0.048 ppm (Nagata 2003, Measurement of odor threshold "
                            "by triangle odor bag method, Japan Ministry of the Environment)"),
    "smell_carcass_ppm": (R_, "human odour threshold of dimethyl disulfide 0.0022 ppm (Nagata 2003)"),
    "smell_smoke_ppm": (RN_, "CO is odourless; smoke is smelt by its phenolic co-emissions (human phenol threshold "
                             "0.0056 ppm, Nagata 2003), emitted at a few thousandths of CO by moles in wood smoke "
                             "(grams of phenols and methoxyphenols against about 100 g CO per kg of fuel; Schauer et "
                             "al. 2001, Environ. Sci. Technol. 35, 1716; Andreae & Merlet 2001); new_rule: 2 ppm of "
                             "the CO tracer"),
    "smell_co2_ppm": (RN_, "CO2-sensing animals respond to rises of about 0.01 % (mosquitoes: Gillies 1980, Bull. "
                           "Entomol. Res. 70, 525); new_rule: 100 ppm above the background (the co2 field is the "
                           "excess); not a human-panel value, so nose_factor does not apply"),
    "nose_factor": (RN_, "animal noses detect many odorants, sulfides and amines among them, at 10-1,000 times lower "
                         "concentrations than the human panels (Laska 2017, Human and animal olfactory capabilities "
                         "compared, in Springer Handbook of Odor, ch. 32; dogs: Walker et al. 2006, Appl. Anim. "
                         "Behav. Sci. 97, 241); new_rule 0.01 (two orders) on the human-panel thresholds. It "
                         "reconciles the thresholds with patch.volatile_step: a 10 kg carcass alone in a 16 m cell "
                         "codes about 0.2 there and 0.1 three cells away, dense plant cover about 0.8"),
    "HUMAN_PANEL_FIELDS": (N_, "the fields whose thresholds come from human panels (plant, carcass, smoke)"),
    "SMELL_FORMULA": (R_, "the tracer molecules: isoprene C5H8, dimethyl disulfide C2H6S2, CO, CO2"),
    "STP_T_K": (R_, "odour thresholds are mixing ratios at about 1 atm and 25 C; receptors bind molecules, so the "
                    "threshold is the number density ppm x 1e-6 x P_STANDARD / (R x 298.15 K)"),
    "smell_span_dec": (RN_, "olfactory receptor neurons and their populations code over about 2-4 decades of "
                            "concentration (Firestein 2001, Nature 413, 211); new_rule: code = log10(1 + c / c_th) "
                            "/ 3"),
    "smell_gradient": (N_, "the code half a fine cell ahead minus half a cell behind along the heading (bilinear "
                           "in the field): the field is resolved at the fine cell, and a body moving through a bout "
                           "samples it along its path"),
    # ---- touch
    "touch_plant": (D_, "share of the ground under the mouth covered by leaf: patch.fapar of the soft tissue (1 - "
                        "exp(-k LAI), Monsi & Saeki), which grows with the tissue's mass per m^2"),
    "touch_damp": (D_, "soil water / the bucket's capacity (patch rules, 150 kg/m^2), 1 under standing water or sea"),
    "touch_wet": (D_, "free water depth h (pond, soil water above the bucket, the sea depth max(0, sea level - "
                      "elevation)) relative to the body's radius, x / (1 + x)"),
    "touch_contact": (N_, "an object (body or item on the ground) touches the mouth when its surface gap (distance "
                          "from the mouth point to its centre less its radius) is at most mouth_reach_radii x r; the "
                          "one with the smallest gap is felt"),
    "hardness_full": (R_, "the Mohs scale tops at 10 (diamond): hardness / 10"),
    "thermal_k": (RN_, "warm and cold receptors code skin temperature changes over about +-10 K and heat "
                       "nociceptors start near 43 C, about 10 K above skin (Schepers & Ringkamp 2010, Neurosci. "
                       "Biobehav. Rev. 34, 177); new_rule: tanh((T - T_body) / 10 K)"),
    "mass_ratio": (N_, "masses felt relative to the body's own mass, x / (1 + x) (1/2 for an equal mass)"),
    # ---- interoception
    "intero": (N_, "brain3.INTERO_NAMES: reserve / capacity, water / normal, (T_b - 294.5 K) / 23.5 K (signed, -1 at "
                   "freezing and +1 at denaturation, so a bias-free readout gain can value one side against the "
                   "other), damage, gut contents / gut capacity, the light level at the body (sky and the fires "
                   "whose line is clear), Fr / (1 + Fr) with Fr = speed / sqrt(g x 2 r) (the Froude speed of a body "
                   "of length 2 r; walking turns to running near 0.7, Alexander 1989, Physiol. Rev. 69, 1199): "
                   "every channel within [-1, 1]"),
    # ---- neighbours
    "K": (N_, "PLANET-V3-SPEC 4: at most K = 32 candidates from the fine spatial hash (patch.neighbours)"),
    "max_scan": (N_, "numerics: each hash scans at most 4 K = 128 candidates (patch.neighbours' default)"),
    "cross_neighbours": (N_, "items found by the same spatial hash as bodies (patch.PROVENANCE['neighbour_scan']) "
                             "between two sets: the 3 x 3 cells nearest first, a stratified uniform sample past the "
                             "scan budget, the K nearest kept (or the K smallest distance / reach for the far hash)"),
    "far_cells": (N_, "PLANET-V3-SPEC 4 'out to the physical range of each organ': a coarse hash of 8 x 8 fine cells "
                      "(128 m on the default grid; 3 x 3 of them, so at least 128 m all round) holds the targets "
                      "whose physical range could pass the fine block; details['beyond'] counts the ones whose range "
                      "is longer still"),
    "far_k": (N_, "numerics: at most 16 far candidates each of bodies and items per body, the ones with the smallest "
                  "distance over their reach bound (vision_reach, sound_reach); a far candidate counts only when it "
                  "is detected on its own (seen, or heard above threshold)"),
    "reach_bound": (D_, "upper bounds of a target's detection distance for the arena's best eye (largest G, smallest "
                        "Omega_res) and best ear (smallest threshold), with the line clear: sight from the Rose and "
                        "contrast tests with Omega <= 2 pi (r/d)^2, sound from P / (4 pi d^2) 10^(-alpha d / 10) >= "
                        "I_th (bisection); the exact per-pair tests then decide"),
}
for _f in fields(SenseRules):
    assert _f.name in PROVENANCE, _f.name


# ============================================================================================ inputs
@dataclass
class Bodies:
    """The body slots of A arenas [A, N] (float32 unless noted), as plain tensors:

    ``alive`` bool; ``pos`` [A, N, 2] m; ``heading`` rad (:func:`heading_vector`); ``mass_kg`` the body's mass (its
    size); ``speed_m_s`` speed over the ground in the last bout; ``body_k`` K; ``damage`` in [0, 1]; ``reserve_j`` and
    ``reserve_cap_j`` J; ``water_kg`` and ``water_norm_kg``; ``gut_kg`` [A, N] or [A, N, G] and ``gut_cap_kg``;
    genes ``eye``, ``ear``, ``voice`` (kg per kg body) and ``muscle`` (W per kg muscle); ``loudness`` in [0, 1] and
    ``calls`` [A, N, D] of the last bout. Optional: ``held`` [A, N, 2] long item slots (-1 empty), ``fur_m`` (the
    coat, for reflectance), ``radius_m`` (overrides the sphere of the mass), ``surface_vis`` (overrides the coat's
    reflectance) and ``call_w`` (overrides :func:`call_acoustic_power`). Nothing else is read: lineage ids and other
    labels may be carried by the caller's state and never reach the input. ``mouth_pos`` [A, N, 2] and ``reach_m``
    (optional) override the mouth point and its reach (``PROVENANCE['mouth_reach_radii']``)."""
    alive: torch.Tensor
    pos: torch.Tensor
    heading: torch.Tensor
    mass_kg: torch.Tensor
    speed_m_s: torch.Tensor
    body_k: torch.Tensor
    damage: torch.Tensor
    reserve_j: torch.Tensor
    reserve_cap_j: torch.Tensor
    water_kg: torch.Tensor
    water_norm_kg: torch.Tensor
    gut_kg: torch.Tensor
    gut_cap_kg: torch.Tensor
    eye: torch.Tensor
    ear: torch.Tensor
    voice: torch.Tensor
    muscle: torch.Tensor
    loudness: torch.Tensor
    calls: torch.Tensor
    held: torch.Tensor | None = None
    fur_m: torch.Tensor | None = None
    radius_m: torch.Tensor | None = None
    surface_vis: torch.Tensor | None = None
    call_w: torch.Tensor | None = None
    mouth_pos: torch.Tensor | None = None
    reach_m: torch.Tensor | None = None

    @classmethod
    def from_mapping(cls, src, **override) -> "Bodies":
        """Pick the fields above by name from a mapping or an object's attributes (a genome dict under ``genome`` is
        searched for the genes); every other entry (lineage ids, ages, anything) is ignored."""
        def get(name):
            if name in override:
                return override[name]
            for holder in (src, _get(src, "genome")):
                v = _get(holder, name)
                if v is not None:
                    return v
            return None

        out = {}
        for f in fields(cls):
            v = get(f.name)
            if v is None and f.default is MISSING:
                raise KeyError(f"Bodies.from_mapping: missing {f.name!r}")
            out[f.name] = v
        return cls(**out)


def _get(holder, name):
    if holder is None:
        return None
    if isinstance(holder, dict):
        return holder.get(name)
    return getattr(holder, name, None)


@dataclass
class Items:
    """Items of A arenas in I slots [A, I]: ``pos`` [A, I, 2] m, ``alive`` bool, ``mass`` kg, ``temp_k`` K,
    ``sharp`` in [0, 1], ``comp`` [A, I, S] mass fractions in ``materials.SPECIES`` order. ``on_ground`` (bool) marks
    the items lying on the ground (default: every live item that no live body holds in ``Bodies.held``). Optional
    precomputed ``radius_m``, ``vis``, ``emissivity``, ``hardness`` (tool hardness, Mohs) and ``density`` (kg/m^3)
    replace the values derived from ``comp``."""
    pos: torch.Tensor
    alive: torch.Tensor
    mass: torch.Tensor
    temp_k: torch.Tensor
    sharp: torch.Tensor
    comp: torch.Tensor | None = None
    on_ground: torch.Tensor | None = None
    radius_m: torch.Tensor | None = None
    vis: torch.Tensor | None = None
    emissivity: torch.Tensor | None = None
    hardness: torch.Tensor | None = None
    density: torch.Tensor | None = None

    @classmethod
    def from_pool(cls, pool) -> "Items":
        """From version 2's ``items.ItemPool`` [A, I] on patch coordinates (manipulate.py): positions (x, y), and the
        items with ``holder`` -1 lying on the ground."""
        return cls(pos=pool.pos[..., :2], alive=pool.alive, mass=pool.mass, temp_k=pool.temp_k, sharp=pool.sharp,
                   comp=pool.comp, on_ground=pool.alive & (pool.holder < 0))


@dataclass
class Fires:
    """Fires of A arenas in F slots [A, F]: ``pos`` [A, F, 2] m, ``alive`` bool, ``temp_k`` K (flame temperature)
    and ``area_m2`` (the flame's radiating area)."""
    pos: torch.Tensor
    alive: torch.Tensor
    temp_k: torch.Tensor
    area_m2: torch.Tensor

    @classmethod
    def from_pool(cls, pool, area_m2=None) -> "Fires":
        """From version 2's ``items.FirePool`` [A, F] on patch coordinates: the radiating area is the bed's,
        pi crafting.FIRE_RADIUS_M^2, unless ``area_m2`` is given (``PROVENANCE['fire_area']``)."""
        if area_m2 is None:
            from haishool.life9.planet import crafting
            area_m2 = torch.full_like(pool.temp_k, math.pi * float(crafting.FIRE_RADIUS_M) ** 2)
        return cls(pos=pool.pos[..., :2], alive=pool.alive, temp_k=pool.temp_k,
                   area_m2=torch.as_tensor(area_m2, dtype=torch.float32, device=pool.temp_k.device).expand_as(
                       pool.temp_k))


@dataclass
class Scene:
    """This bout's light and ground per fine cell [A, n, n]: ``light_vis`` the visible illuminance (W/m^2, sky and
    sun including the night sky) and ``ground_vis`` the ground's visible reflectance (``patch.ground_reflectance``)."""
    light_vis: torch.Tensor
    ground_vis: torch.Tensor


def saturation_vapour_pa(t_k) -> torch.Tensor:
    """Saturation vapour pressure of water (Pa, float64) at t_k (ISO 9613-1 annex B, ``PROVENANCE['saturation_vapour']``)."""
    t = torch.as_tensor(t_k, dtype=torch.float64)
    return ISO_P_REF_PA * torch.pow(10.0, -6.8346 * (ISO_T01_K / t) ** 1.261 + 4.6151)


@dataclass
class Air:
    """Near-surface air per arena [A] (float32): pressure (Pa), temperature (K), molar mass (kg/mol), the heat
    capacity ratio, gravity (m/s^2) and the mole fractions of O2, N2 and water vapour (of the total)."""
    pressure_pa: torch.Tensor
    t_k: torch.Tensor
    molar_kg: torch.Tensor
    gamma: torch.Tensor
    g_m_s2: torch.Tensor
    x_o2: torch.Tensor
    x_n2: torch.Tensor
    x_h2o: torch.Tensor

    def density(self) -> torch.Tensor:
        return self.pressure_pa * self.molar_kg / (K.R_GAS * self.t_k)

    def sound_speed(self) -> torch.Tensor:
        return torch.sqrt(self.gamma * K.R_GAS * self.t_k / self.molar_kg)

    @classmethod
    def from_partial(cls, partial_pa, t_k, g_m_s2, vapour_pa) -> "Air":
        """From the partial pressures [A, 4] (Pa) of N2, O2, CO2, Ar (``climate.partial_pressures`` mapped to the
        arenas) and the water vapour pressure [A] (``patch.PatchForcing.vapour_pa``; 0 for dry air): total pressure,
        mean molar mass, gamma = Cp / (Cp - R) of the mixture and the mole fractions (``PROVENANCE['air']``)."""
        p = torch.as_tensor(partial_pa, dtype=torch.float64)
        A = p.shape[0]
        w = torch.as_tensor(vapour_pa, dtype=torch.float64, device=p.device).reshape(-1).expand(A)
        p = torch.cat((p, w.reshape(-1, 1)), -1)
        gases = ("N2", "O2", "CO2", "Ar", "H2O")
        m = torch.tensor([K.molar_mass(g) for g in gases], dtype=torch.float64, device=p.device)
        cp = torch.tensor([K.CP_MOLAR_298[g] for g in gases], dtype=torch.float64, device=p.device)
        total = p.sum(-1)
        x = p / total.clamp_min(1e-300)[:, None]
        cpm = (x * cp).sum(-1)
        f = lambda v: torch.as_tensor(v, dtype=torch.float32, device=p.device).reshape(-1).expand(A).clone()  # noqa
        return cls(pressure_pa=total.float(), t_k=f(t_k), molar_kg=(x * m).sum(-1).float(),
                   gamma=(cpm / (cpm - K.R_GAS)).float(), g_m_s2=f(g_m_s2), x_o2=x[:, 1].float(),
                   x_n2=x[:, 0].float(), x_h2o=x[:, 4].float())

    @classmethod
    def earth(cls, A: int, device="cpu", pressure_pa=K.P_STANDARD, t_k=288.15, g_m_s2=K.G_STANDARD,
              rel_humidity=0.7) -> "Air":
        """Earth's air (constants.EARTH_AIR_MOLE_FRACTION) at the given pressure, temperature and relative humidity
        (an explicit helper: observe never assumes it)."""
        x = K.EARTH_AIR_MOLE_FRACTION
        t = torch.as_tensor(t_k, dtype=torch.float64).reshape(-1).expand(A)
        ptot = torch.as_tensor(pressure_pa, dtype=torch.float64).reshape(-1).expand(A)
        vap = torch.minimum(float(rel_humidity) * saturation_vapour_pa(t), ptot)
        frac = torch.tensor([[x["N2"], x["O2"], x["CO2"], x["Ar"]]], dtype=torch.float64)
        dry = frac / frac.sum() * (ptot - vap)[:, None]
        return cls.from_partial(dry.to(device), t.to(device), g_m_s2, vap.to(device))


# ============================================================================================ geometry helpers
def heading_vector(heading) -> torch.Tensor:
    """(cos h, sin h) [..., 2] in (east, north): the direction a heading points (``PROVENANCE['heading']``)."""
    h = torch.as_tensor(heading)
    return torch.stack((torch.cos(h), torch.sin(h)), -1)


def sector_of(heading, dvec) -> torch.Tensor:
    """Sector (long) of the displacement dvec [..., 2] seen from a body with heading [...] (broadcast): 0 centred
    ahead, then clockwise (``PROVENANCE['sector_of']``)."""
    rel = heading - torch.atan2(dvec[..., 1], dvec[..., 0])
    return torch.floor(torch.remainder(rel + SECTOR_WIDTH / 2, 2 * math.pi) / SECTOR_WIDTH).long().clamp(0, SECTORS - 1)


def body_radius(mass_kg) -> torch.Tensor:
    """Radius (m) of the sphere of a body's mass at tissue density (``PROVENANCE['body_radius']``)."""
    m = torch.as_tensor(mass_kg, dtype=torch.float32).clamp_min(0)
    return (3 * m / (4 * math.pi * TISSUE_DENSITY)) ** (1.0 / 3.0)


def mouth_point(pos, heading, radius) -> torch.Tensor:
    """The mouth [..., 2]: the front of the body's sphere, pos + r (cos h, sin h) (not wrapped)."""
    return pos + torch.as_tensor(radius)[..., None] * heading_vector(heading)


def surface_reflectance(fur_m=None, like=None, rules: SenseRules | None = None) -> torch.Tensor:
    """Visible reflectance of a body's surface: skin covered by its coat as 1 - exp(-fur_m / fur_cover_m)
    (``PROVENANCE['body_surface']``); bare skin without a coat."""
    r = rules or SenseRules()
    skin, fur = optics.vis("tissue:skin"), optics.vis("tissue:fur")
    if fur_m is None:
        return torch.full_like(like, skin)
    cover = 1 - torch.exp(-torch.as_tensor(fur_m, dtype=torch.float32).clamp_min(0) / r.fur_cover_m)
    return skin + (fur - skin) * cover


def water_depth(geom: pt.PatchGeometry, state=None, prules: pt.PatchRules | None = None) -> torch.Tensor:
    """Free water depth (m) per fine cell [A, n, n]: the sea depth max(0, sea level - elevation) plus the pond and
    the soil water above the bucket (``PROVENANCE['touch_wet']``); no label is read."""
    sea = (geom.sea_level_m[:, None, None] - geom.elev).clamp_min(0)
    if state is None:
        return sea
    bucket = (prules or pt.PatchRules()).climate.bucket_kg_m2
    return sea + (state.pond.clamp_min(0) + (state.soil - bucket).clamp_min(0)) / K.RHO_WATER


# ============================================================================================ physics of the organs
def eye_optics(eye_kg, intact=None, rules: SenseRules | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """(theta_res rad, photon catch G photons per (W/m^2)) of eyes of mass eye_kg (``PROVENANCE['eye_optics']``);
    ``intact`` (1 - damage) scales the catch."""
    r = rules or SenseRules()
    m = torch.as_tensor(eye_kg, dtype=torch.float32).clamp_min(0)
    d = (6 * m / (math.pi * TISSUE_DENSITY)) ** (1.0 / 3.0)
    d = d.clamp_min(1e-9)
    dp = r.pupil_share * d
    theta = torch.maximum(1.22 * r.wavelength_m / dp, 2 * r.receptor_m / (r.nodal_share * d))
    photon_j = K.H_PLANCK * K.C_LIGHT / r.wavelength_m
    G = (math.pi / 4) * dp * dp * (r.integration_s * r.quantum_eff / photon_j)
    G = torch.where(m > 0, G, torch.zeros_like(G))
    if intact is not None:
        G = G * torch.as_tensor(intact, dtype=torch.float32).clamp(0, 1)
    return theta, G


def rose_snr(delta_n, n_b) -> torch.Tensor:
    """Photon signal over photon noise dN / sqrt(N_b + dN) (Rose 1948); 0 when nothing is collected."""
    return delta_n / torch.sqrt((n_b + delta_n).clamp_min(1e-30))


def sphere_solid_angle(radius, d) -> torch.Tensor:
    """Solid angle (sr) of a sphere of radius r at centre distance d: 2 pi s^2 / (1 + sqrt(1 - s^2)), s = r / d (the
    cancellation-free form of 2 pi (1 - sqrt(1 - s^2)), exact in float32 down to s ~ 1e-19); 2 pi inside."""
    s = (torch.as_tensor(radius) / torch.as_tensor(d).clamp_min(1e-12)).clamp(0, 1)
    return 2 * math.pi * s * s / (1 + torch.sqrt((1 - s * s).clamp_min(0)))


def night_light(rules: SenseRules | None = None) -> float:
    """Visible illuminance of the moonless night sky (W/m^2, ``PROVENANCE['night_sky_lux']``)."""
    return (rules or SenseRules()).night_sky_lux / LUMEN_PER_W


def visible_light(bout_sw, par_fraction, rules: SenseRules | None = None) -> torch.Tensor:
    """Visible illuminance [A, n, n] (W/m^2) from a bout's shortwave [A, n, n] (``patch.light``) and the PAR share
    [A], plus the night sky (``PROVENANCE['visible_light']``)."""
    par = torch.as_tensor(par_fraction, dtype=torch.float32, device=bout_sw.device).reshape(-1, 1, 1)
    return bout_sw.float().clamp_min(0) * par + night_light(rules)


def light_level(e_vis, rules: SenseRules | None = None) -> torch.Tensor:
    """Light level log10(1 + E / E_night) / span (``PROVENANCE['light_span_dec']``)."""
    r = rules or SenseRules()
    return torch.log10(1 + torch.as_tensor(e_vis).clamp_min(0) / night_light(r)) / r.light_span_dec


def emitted_code(x, rules: SenseRules | None = None) -> torch.Tensor:
    """log10(1 + x) / emit_span_dec clamped to [0, 1] (``PROVENANCE['emit_span_dec']``)."""
    r = rules or SenseRules()
    return (torch.log10(1 + torch.as_tensor(x).clamp_min(0)) / r.emit_span_dec).clamp(0, 1)


def call_acoustic_power(loudness, voice, muscle_w_kg, mass_kg, rules: SenseRules | None = None) -> torch.Tensor:
    """Acoustic power (W) of a call: loudness x vocal_eff x muscle x voice x mass (``PROVENANCE['call_power']``).
    body.py pays the mechanical power (this / vocal_eff) for it."""
    r = rules or SenseRules()
    return (torch.as_tensor(loudness).clamp(0, 1) * r.vocal_eff * torch.as_tensor(muscle_w_kg).clamp_min(0)
            * torch.as_tensor(voice).clamp_min(0) * torch.as_tensor(mass_kg).clamp_min(0))


def vocal_tract_m(voice, mass_kg, rules: SenseRules | None = None) -> torch.Tensor:
    """Length (m) of the voice organ as a tube of its volume and aspect ratio a: (4 a^2 V / pi)^(1/3)
    (``PROVENANCE['call_frequency']``)."""
    r = rules or SenseRules()
    vol = (torch.as_tensor(voice) * torch.as_tensor(mass_kg)).clamp_min(1e-15) / TISSUE_DENSITY
    return (4 * r.voice_aspect ** 2 * vol / math.pi) ** (1.0 / 3.0)


def call_frequency(voice, mass_kg, sound_speed, rules: SenseRules | None = None) -> torch.Tensor:
    """Dominant call frequency (Hz): the quarter-wave resonance c / (4 l) of the voice organ's tube
    (:func:`vocal_tract_m`)."""
    return torch.as_tensor(sound_speed) / (4 * vocal_tract_m(voice, mass_kg, rules))


def footfall_power(mass_kg, speed_m_s, rules: SenseRules | None = None) -> torch.Tensor:
    """Acoustic power (W) of locomotion noise, kappa M v^2 (``PROVENANCE['footfall']``)."""
    r = rules or SenseRules()
    p_ref = 4 * math.pi * r.hear_i0_w_m2 * 10 ** (r.footfall_db / 10)          # W radiated, from the level at 1 m
    kappa = p_ref / (r.footfall_kg * r.footfall_m_s ** 2)
    return kappa * torch.as_tensor(mass_kg).clamp_min(0) * torch.as_tensor(speed_m_s) ** 2


def absorption_db_m(freq_hz, air: Air) -> torch.Tensor:
    """Atmospheric absorption (dB/m, float32) of ISO 9613-1 at frequency f (a number, [A] or [A, ...]) in the air of
    each arena [A] (``PROVENANCE['absorption']`` and ``['absorption_other_air']``)."""
    dev = air.pressure_pa.device
    f = torch.as_tensor(freq_hz, dtype=torch.float64, device=dev)
    A = air.pressure_pa.shape[0]
    if f.dim() == 0:
        f = f.expand(A)
    shape = (A,) + (1,) * (f.dim() - 1)

    def col(v):
        return torch.as_tensor(v, dtype=torch.float64, device=dev).reshape(shape)

    T = col(air.t_k)
    pa = col(air.pressure_pa).clamp_min(1.0) / ISO_P_REF_PA
    tr = T / ISO_T0_K
    xw = col(air.x_h2o).clamp(0, 1)
    dry = (1 - xw).clamp_min(1e-12)
    h = 100.0 * xw
    fro = pa * (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h))
    frn = pa * tr ** -0.5 * (9.0 + 280.0 * h * torch.exp(-4.170 * (tr ** (-1.0 / 3.0) - 1)))
    classical = 1.84e-11 / pa * tr ** 0.5 * (ISO_GAMMA / col(air.gamma)) ** 1.5 * (col(air.molar_kg) / ISO_MOLAR_KG) ** 0.5
    o2 = 0.01275 * (col(air.x_o2) / dry / ISO_X_O2) * torch.exp(-2239.1 / T) / (fro + f * f / fro)
    n2 = 0.1068 * (col(air.x_n2) / dry / ISO_X_N2) * torch.exp(-3352.0 / T) / (frn + f * f / frn)
    return (8.686 * f * f * (classical + tr ** -2.5 * (o2 + n2))).float()


def ear_threshold(ear_kg, intact=None, rules: SenseRules | None = None) -> torch.Tensor:
    """Threshold intensity (W/m^2) of ears of mass ear_kg (``PROVENANCE['ear_threshold']``); inf without ears."""
    r = rules or SenseRules()
    m = torch.as_tensor(ear_kg, dtype=torch.float32).clamp_min(0)
    th = r.hear_i0_w_m2 * (r.ear_ref_kg / m.clamp_min(1e-30)) ** (2.0 / 3.0)
    if intact is not None:
        th = th / torch.as_tensor(intact, dtype=torch.float32).clamp(0, 1).clamp_min(1e-30)
    return torch.where(m > 0, th, torch.full_like(th, math.inf))


def received_intensity(power_w, r_m, alpha_db_m, clear, rules: SenseRules | None = None) -> torch.Tensor:
    """Intensity (W/m^2) at distance r of a source of acoustic power P: P / (4 pi r^2) x 10^(-alpha r / 10), x
    occlusion^2 where the line is blocked (``PROVENANCE['spreading']``)."""
    r = rules or SenseRules()
    occ = torch.where(clear, torch.ones_like(r_m), torch.full_like(r_m, r.occlusion ** 2))
    return power_w / (4 * math.pi * r_m * r_m) * torch.pow(10.0, -alpha_db_m * r_m / 10) * occ


def sensation(intensity, threshold, rules: SenseRules | None = None) -> torch.Tensor:
    """Sensation level 10 log10(I / I_th) / level_full_db, clamped to [0, 1]."""
    r = rules or SenseRules()
    lvl = 10 * torch.log10(intensity.clamp_min(1e-38) / threshold.clamp_min(1e-38))
    return (lvl / r.level_full_db).clamp(0, 1)


def vision_reach(dl, lb, radius, G, omega_res, rules: SenseRules | None = None) -> torch.Tensor:
    """Upper bound (m) of the distance at which a target of radius r, luminance difference dl = |L_t - L_b| and
    background luminance lb can pass the Rose and contrast tests of an eye (G, Omega_res), the line clear; 0 if it
    never can (``PROVENANCE['reach_bound']``)."""
    r = rules or SenseRules()
    f64 = torch.float64
    dl, lb = torch.as_tensor(dl, dtype=f64), torch.as_tensor(lb, dtype=f64)
    G, om = torch.as_tensor(G, dtype=f64), torch.as_tensor(omega_res, dtype=f64)
    k = r.rose_k
    b = lb * om * G
    a_min = 0.5 * (k * k + torch.sqrt(k ** 4 + 4 * k * k * b))
    dls = dl.clamp_min(1e-300)
    om_min = torch.maximum(r.c_min * om * lb / dls, a_min / (dls * G).clamp_min(1e-300))
    ok = (dl > 0) & (dl >= r.c_min * lb) & (G > 0) & torch.isfinite(om)
    # the sphere's solid angle inverted exactly: Omega = 2 pi c, c = 1 - sqrt(1 - s^2), so s = sqrt(c (2 - c))
    c = (om_min / (2 * math.pi)).clamp(1e-300, 1.0)
    s_min = torch.sqrt(c * (2 - c))
    d = torch.as_tensor(radius, dtype=f64) / s_min
    return torch.where(ok, d, torch.zeros_like(d)).float()


def sound_reach(power_w, alpha_db_m, i_th, r0, iters: int = 48) -> torch.Tensor:
    """Upper bound (m) of the distance d >= r0 at which a source of acoustic power P is heard at threshold i_th
    unoccluded: the largest d with P / (4 pi d^2) 10^(-alpha d / 10) >= i_th (bisection in log d); 0 if not even at
    r0 (``PROVENANCE['reach_bound']``)."""
    f64 = torch.float64
    P = torch.as_tensor(power_w, dtype=f64)
    a = torch.as_tensor(alpha_db_m, dtype=f64)
    I = torch.as_tensor(i_th, dtype=f64)
    lo = torch.as_tensor(r0, dtype=f64).clamp_min(1e-6)
    P, a, I, lo = torch.broadcast_tensors(P, a, I, lo)
    lp = torch.log10(P.clamp_min(1e-300) / (4 * math.pi * I.clamp_min(1e-300)))

    def g(d):
        return lp - 2 * torch.log10(d) - a * d / 10

    ok = (P > 0) & torch.isfinite(I) & (g(lo) >= 0)
    hi = torch.maximum(torch.pow(10.0, 0.5 * lp).clamp_max(1e12), lo)      # spreading alone
    lo = lo.clone()
    for _ in range(iters):
        mid = torch.sqrt(lo * hi)
        up = g(mid) >= 0
        lo = torch.where(up, mid, lo)
        hi = torch.where(up, hi, mid)
    return torch.where(ok, hi, torch.zeros_like(hi)).float()


def smell_thresholds(rules: SenseRules | None = None) -> dict:
    """Odour thresholds (kg/m^3) of the volatile fields: ppm x 1e-6 x P_STANDARD / (R x 298.15 K) x M, the human-panel
    ones x nose_factor (``PROVENANCE['STP_T_K']`` and ``['nose_factor']``)."""
    r = rules or SenseRules()
    n_air = K.P_STANDARD / (K.R_GAS * STP_T_K)
    ppm = {"plant": r.smell_plant_ppm, "carcass": r.smell_carcass_ppm, "smoke": r.smell_smoke_ppm,
           "co2": r.smell_co2_ppm}
    return {f: ppm[f] * (r.nose_factor if f in HUMAN_PANEL_FIELDS else 1.0) * 1e-6 * n_air
            * K.molar_mass(SMELL_FORMULA[f]) for f in SMELL_FIELDS}


def smell_code(c, c_th, rules: SenseRules | None = None) -> torch.Tensor:
    """log10(1 + c / c_th) / smell_span_dec (``PROVENANCE['smell_span_dec']``)."""
    r = rules or SenseRules()
    return torch.log10(1 + torch.as_tensor(c).clamp_min(0) / c_th) / r.smell_span_dec


def michelson(l_t, l_b) -> torch.Tensor:
    """(L_t - L_b) / (L_t + L_b); 0 when both are dark."""
    s = l_t + l_b
    return torch.where(s > 0, (l_t - l_b) / s.clamp_min(1e-30), torch.zeros_like(s))


def saturate(x) -> torch.Tensor:
    """x / (1 + x) for x >= 0."""
    x = torch.as_tensor(x).clamp_min(0)
    return x / (1 + x)


# ============================================================================================ scene from the patch
def scene_from_patch(geom: pt.PatchGeometry, state, forcing, bout: int, bout_sw=None, rules: SenseRules | None = None,
                     prules: pt.PatchRules | None = None) -> Scene:
    """This bout's :class:`Scene` from the patch: ``patch.light`` (pass ``bout_sw`` [A, B, n, n] to reuse it) x the PAR
    share + the night sky, and ``patch.ground_reflectance`` at the fine temperature."""
    if bout_sw is None:
        _, bout_sw = pt.light(geom, forcing)
    light = visible_light(bout_sw[:, bout], forcing.par_fraction, rules)
    ground = pt.ground_reflectance(state, geom, prules, t_k=pt.fine_temperature(geom, forcing))
    return Scene(light_vis=light, ground_vis=ground)


# ============================================================================================ neighbour search
def _block_offsets(nc: int):
    """The 3 x 3 block of cells on a periodic grid of nc cells per side, own cell first, without repeats when
    nc < 3 (the block is then the whole grid)."""
    one = (0, 1, -1) if nc >= 3 else tuple(range(nc))
    return [(0, 0)] + [(a, b) for a in one for b in one if (a, b) != (0, 0)]


def cross_neighbours(geom: pt.PatchGeometry, qpos, qalive, tpos, talive, K: int = 32, max_scan: int | None = None,
                     chunk: int | None = None, gen=None, *, cell: int = 1, reach=None, exclude_block: bool = False):
    """The spatial hash of ``patch.neighbours`` between queries qpos [A, N, 2] (alive qalive) and targets tpos
    [A, M, 2] (present talive): targets are bucketed by cells of ``cell`` x ``cell`` fine cells; each live query scans
    its 3 x 3 cells nearest first up to ``max_scan`` (4 K) targets, the last cell reached by a stratified uniform
    sample (offset on ``gen``, 0.5 without), and keeps the K nearest by periodic distance. With ``reach`` [A, M] a
    target farther than its reach is skipped and the K smallest distance / reach are kept; ``exclude_block`` skips the
    targets in the query's own 3 x 3 fine cells (the far hash). Returns (idx [A, N, K] long, -1 padded; offset
    [A, N, K, 2]; dist [A, N, K], inf for padding; count [A, N], the qualifying targets in the scanned cells plus the
    unscanned ones) (``PROVENANCE['cross_neighbours']``)."""
    qpos = torch.as_tensor(qpos, dtype=torch.float32, device=geom.device)
    tpos = torch.as_tensor(tpos, dtype=torch.float32, device=geom.device)
    qalive = torch.as_tensor(qalive, device=geom.device).bool()
    talive = torch.as_tensor(talive, device=geom.device).bool()
    A, N = qalive.shape
    M = talive.shape[1]
    n, L = geom.n, geom.L
    nc = max(1, n // max(1, int(cell)))
    dxc = L / nc
    dev = geom.device
    Mb = 4 * K if max_scan is None else int(max_scan)
    C = max(1, 2 * K if chunk is None else int(chunk))
    best_k = torch.full((A, N, K), float("inf"), device=dev)
    best_d = torch.full((A, N, K), float("inf"), device=dev)
    best_i = torch.full((A, N, K), -1, dtype=torch.long, device=dev)
    best_o = torch.zeros(A, N, K, 2, device=dev)
    if M == 0 or K == 0:
        return best_i, best_o, best_d, torch.zeros(A, N, dtype=torch.long, device=dev)
    ar = torch.arange(A, device=dev)[:, None]
    tw = torch.remainder(tpos, L)
    tc = torch.floor(tw / dxc).long().clamp(0, nc - 1)
    sentinel = A * nc * nc
    tkey = torch.where(talive, ar * (nc * nc) + tc[..., 0] * nc + tc[..., 1], torch.full_like(tc[..., 0], sentinel))
    sk, order = torch.sort(tkey.reshape(-1), stable=True)
    wrapped = torch.remainder(qpos, L)
    c = torch.floor(wrapped / dxc).long().clamp(0, nc - 1)
    frac = (wrapped - c.float() * dxc).clamp(0, dxc)
    blk = _block_offsets(nc)
    B = len(blk)
    off = torch.tensor(blk, device=dev)
    gap = []
    for k in range(2):
        o = off[:, k].float()
        fk = frac[..., k, None]
        if nc >= 3:
            gk = torch.where(o > 0, dxc - fk, torch.where(o < 0, fk, torch.zeros_like(fk)))
        else:
            gk = torch.where(o != 0, torch.minimum(fk, dxc - fk), torch.zeros_like(fk))
        gap.append(gk)
    nine = torch.argsort(gap[0] ** 2 + gap[1] ** 2, dim=-1, stable=True)
    offo = off[nine]
    qx = torch.remainder(c[..., None, 0] + offo[..., 0], nc)
    qy = torch.remainder(c[..., None, 1] + offo[..., 1], nc)
    qkey = ar[..., None] * (nc * nc) + qx * nc + qy
    start = torch.searchsorted(sk, qkey.reshape(-1)).reshape(A, N, B)
    end = torch.searchsorted(sk, qkey.reshape(-1), right=True).reshape(A, N, B)
    cnt = torch.where(qalive[..., None], end - start, torch.zeros_like(start))
    cum = torch.cumsum(cnt, -1)
    total = cum[..., -1]
    before = cum - cnt
    if gen is None:
        u = torch.full((A, N), 0.5, device=dev)
    else:
        u = torch.rand(A, N, generator=gen, device=gen.device).to(dev)
    scan = torch.minimum(total, torch.full_like(total, Mb))
    flat_pos = tpos.reshape(-1, 2)
    if exclude_block:
        qf = torch.floor(wrapped / geom.dx).long().clamp(0, n - 1)                      # [A, N, 2]
        tf = torch.floor(tw / geom.dx).long().clamp(0, n - 1).reshape(-1, 2)
    flat_reach = None if reach is None else torch.as_tensor(reach, dtype=torch.float32, device=dev).reshape(-1)
    qual = torch.zeros(A, N, dtype=torch.long, device=dev)
    for s0 in range(0, Mb, C):
        slot = torch.arange(s0, min(s0 + C, Mb), device=dev).expand(A, N, -1).contiguous()
        b = torch.searchsorted(cum.contiguous(), slot, right=True).clamp_max(B - 1)
        bef = before.gather(-1, b)
        cn = cnt.gather(-1, b)
        inside = cum.gather(-1, b) <= Mb
        avail = (Mb - bef).clamp_min(1)
        strat = torch.floor(((slot - bef).double() + u[..., None].double()) * cn.double() / avail.double()).long()
        pick = torch.where(inside, slot - bef, torch.minimum(strat, (cn - 1).clamp_min(0)))
        valid = slot < scan[..., None]
        p = torch.where(valid, start.gather(-1, b) + pick, torch.zeros_like(slot)).clamp(0, sk.numel() - 1)
        j = order[p]
        d = pt.wrap(flat_pos[j] - qpos[..., None, :], L)
        dist = d.norm(dim=-1)
        if exclude_block:
            dc = torch.remainder(tf[j] - qf[..., None, :], n)
            near = ((dc == 0) | (dc == 1) | (dc == n - 1)).all(-1)
            valid = valid & ~near
        key = dist
        if flat_reach is not None:
            rj = flat_reach[j]
            valid = valid & (dist <= rj)
            key = dist / rj.clamp_min(1e-30)
        qual = qual + valid.sum(-1)
        cand = torch.where(valid, j % M, torch.full_like(j, -1))
        key = torch.where(valid, key, torch.full_like(key, float("inf")))
        dist = torch.where(valid, dist, torch.full_like(dist, float("inf")))
        all_k = torch.cat((best_k, key), -1)
        all_d = torch.cat((best_d, dist), -1)
        all_i = torch.cat((best_i, cand), -1)
        all_o = torch.cat((best_o, d), -2)
        srt, o = torch.sort(all_k, dim=-1, stable=True)
        o = o[..., :K]
        best_k = srt[..., :K]
        best_d = all_d.gather(-1, o)
        best_i = all_i.gather(-1, o)
        best_o = all_o.gather(-2, o[..., None].expand(*o.shape, 2))
    ok = torch.isfinite(best_k)
    best_i = torch.where(ok, best_i, torch.full_like(best_i, -1))
    best_o = torch.where(ok[..., None], best_o, torch.zeros_like(best_o))
    best_d = torch.where(ok, best_d, torch.full_like(best_d, float("inf")))
    return best_i, best_o, best_d, qual + (total - scan)


# ============================================================================================ internal helpers
def _gather(t, j):
    """t [A, X, ...] gathered at j [A, ...] (long, >= 0) along axis 1 -> [*j.shape, ...]."""
    A = t.shape[0]
    flat = j.reshape(A, -1)
    extra = t.shape[2:]
    if not extra:
        return t.gather(1, flat).reshape(j.shape)
    idx = flat.reshape(A, -1, *([1] * len(extra))).expand(A, flat.shape[1], *extra)
    return t.gather(1, idx).reshape(*j.shape, *extra)


def _los_masked(geom, p, q, h_p, h_q, mask, prules):
    """patch.line_of_sight for the lines where mask [A, M] is set (the others False); p, q [A, M, 2], heights
    [A, M]. The masked lines are packed per arena first, so dead or padded pairs cost nothing."""
    A, M = mask.shape
    out = torch.zeros(A, M, dtype=torch.bool, device=mask.device)
    if M == 0 or not bool(mask.any()):
        return out
    P = int(mask.sum(1).max())
    order = torch.sort((~mask).to(torch.int8), dim=1, stable=True).indices[:, :P]
    sel = mask.gather(1, order)
    pp = p.gather(1, order[..., None].expand(-1, -1, 2))
    qq = torch.where(sel[..., None], q.gather(1, order[..., None].expand(-1, -1, 2)), pp)
    clear = pt.line_of_sight(geom, pp, qq, h_p.gather(1, order), h_q.gather(1, order), rules=prules)
    out.scatter_(1, order, clear & sel)
    return out


def _sector_sum(sector, vals):
    """vals [A, N, C] (or [A, N, C, D]) summed per sector of sector [A, N, C] -> [A, N, S] (or [A, N, S, D])."""
    outs = []
    for s in range(SECTORS):
        m = sector == s
        if vals.dim() == m.dim():
            outs.append(torch.where(m, vals, torch.zeros_like(vals)).sum(-1))
        else:
            outs.append(torch.where(m[..., None], vals, torch.zeros_like(vals)).sum(-2))
    return torch.stack(outs, 2)


def _first_per_sector(cand, sector, *values):
    """For each (arena, body, sector) the first candidate along the last axis (the nearest when the candidates are
    sorted by distance): values [A, N, C] there -> [A, N, S] each (0 where the sector has none)."""
    outs = [[] for _ in values]
    for s in range(SECTORS):
        m = cand & (sector == s)
        has = m.any(-1)
        k = m.to(torch.uint8).argmax(-1, keepdim=True)
        for o, v in zip(outs, values):
            o.append(torch.where(has, v.gather(-1, k)[..., 0], torch.zeros_like(has, dtype=v.dtype)))
    return [torch.stack(o, -1) for o in outs]


def _item_table(items: Items):
    """Per-item radius, reflectance, emissivity, tool hardness and density [A, I] (``PROVENANCE['item_props']``)."""
    mass = torch.as_tensor(items.mass, dtype=torch.float32).clamp_min(0)
    given = (items.radius_m, items.vis, items.emissivity, items.hardness)
    props = None
    if any(v is None for v in given) or items.density is None:
        if items.comp is not None:
            props = materials.props(items.comp, mass)
        elif any(v is None for v in given):
            raise ValueError("Items needs comp, or radius_m, vis, emissivity and hardness")
    radius = items.radius_m if items.radius_m is not None else \
        (3 * props["volume_m3"] / (4 * math.pi)) ** (1.0 / 3.0)
    vis = items.vis if items.vis is not None else optics.mix(items.comp)
    em = items.emissivity if items.emissivity is not None else optics.mix(items.comp, table=optics.EMISSIVITY)
    hard = items.hardness if items.hardness is not None else props["tool_hardness"]
    if items.density is not None:
        dens = torch.as_tensor(items.density, dtype=torch.float32)
    elif props is not None:
        dens = props["density"]
    else:
        dens = mass / (4.0 / 3.0 * math.pi * torch.as_tensor(radius, dtype=torch.float32) ** 3).clamp_min(1e-30)
    f = lambda t: torch.as_tensor(t, dtype=torch.float32)        # noqa: E731
    return f(radius), f(vis), f(em), f(hard), f(dens)


def _compact(mask):
    """Indices [A, P] of the set entries of mask [A, X] first (P the largest count) and their validity."""
    A, X = mask.shape
    P = int(mask.sum(1).max()) if mask.numel() else 0
    order = torch.sort((~mask).to(torch.int8), dim=1, stable=True).indices[:, :P]
    return order, mask.gather(1, order)


def _fire_table(fires: Fires, geom, light, ground, depth, F):
    """The live fires compacted per arena [A, P]: position, visible power (W), flame radius, validity, the background
    luminance at the fire and the flame centre's height above the ground (``PROVENANCE['fire_light']``)."""
    falive = torch.as_tensor(fires.alive, device=geom.device).bool()
    if falive.numel() == 0:
        return None
    fidx, fval = _compact(falive)
    if fidx.shape[1] == 0:
        return None
    L = geom.L
    fpos = torch.remainder(F(fires.pos).gather(1, fidx[..., None].expand(-1, -1, 2)), L)
    farea = F(fires.area_m2).gather(1, fidx).clamp_min(0)
    fpow = optics.visible_exitance(F(fires.temp_k).gather(1, fidx), optics.FLAME_EMISSIVITY).to(geom.device) \
        * farea * fval
    frad = torch.sqrt(farea / (4 * math.pi))
    lb = pt.sample(geom, ground, fpos) * pt.sample(geom, light, fpos) / math.pi
    h = 2 * frad + pt.bilinear(geom, depth, fpos)
    return dict(pos=fpos, pow=fpow, rad=frad, val=fval, lb=lb, h=h)


def _fire_light(geom, fire, pos, h, alive, e_sky, r: SenseRules, pr, G=None, om_res=None, head=None):
    """Light of the fires at points pos [A, X, 2] (height h above the ground, present alive, sky light e_sky [A, X]).

    The fires are streamed fire_chunk at a time. A fire matters at a point when its irradiance there is at least
    c_min of the sky's (it changes the light by more than an eye's contrast threshold) or, with an eye (G, om_res,
    head: the bodies), when it passes that eye's unoccluded Rose and contrast tests. The fires that matter are ranked
    per sector (one sector without an eye) by irradiance, nearest first, and line-tested fire_k ranks per round in the
    sectors without a clear source yet (``PROVENANCE['fire_k']``); the light on the point does not depend on its
    eye. Returns (x [A, X, S] the summed N_s / N_b of the sources the eye sees per sector (None without an eye),
    E [A, X] the irradiance (W/m^2) of every fire less the tested ones whose line is blocked, culled [A, X] the fires
    that mattered but were not line-tested)."""
    dev = pos.device
    A, X = alive.shape
    L = geom.L
    fpos, fpow, frad, fval, flb = fire["pos"], fire["pow"], fire["rad"], fire["val"], fire["lb"]
    P = fpos.shape[1]
    eye = G is not None
    S_ = SECTORS if eye else 1
    k1 = max(1, int(r.fire_k))
    kk = max(1, min(k1 * max(1, int(r.fire_rounds)), P))                 # ranks kept per sector
    best_k = torch.full((A, X, S_, kk), -1.0, device=dev)
    best_x = torch.zeros(A, X, S_, kk, device=dev)
    best_j = torch.full((A, X, S_, kk), -1, dtype=torch.long, device=dev)
    best_e = torch.zeros(A, X, S_, kk, device=dev)
    e_all = torch.zeros(A, X, device=dev)
    n_ok = torch.zeros(A, X, dtype=torch.long, device=dev)
    srange = torch.arange(S_, device=dev)[:, None]
    step = max(1, int(r.fire_chunk))
    for c0 in range(0, P, step):
        c1 = min(P, c0 + step)
        dvec = pt.wrap(fpos[:, None, c0:c1] - pos[:, :, None], L)                          # [A, X, C, 2]
        d = dvec.norm(dim=-1)
        fr = frad[:, None, c0:c1]
        irr = fpow[:, None, c0:c1] / (4 * math.pi * torch.maximum(d, fr).clamp_min(1e-6) ** 2)
        e_all = e_all + irr.sum(-1)
        base = fval[:, None, c0:c1] & alive[..., None] & (irr > 0)
        ok = base & (irr >= r.c_min * e_sky[..., None])
        if eye:
            n_s = irr * G[..., None]
            n_b = flb[:, None, c0:c1] * torch.maximum(sphere_solid_angle(fr, d), om_res[..., None]) * G[..., None]
            seen = base & (rose_snr(n_s, n_b) >= r.rose_k) & (n_s >= r.c_min * n_b)
            ok = ok | seen
            xs = torch.where(seen, (n_s / n_b.clamp_min(1e-30)).clamp_max(1e30), torch.zeros_like(n_s))
            sec = sector_of(head[..., None], dvec)
        else:
            xs = torch.zeros_like(irr)
            sec = torch.zeros_like(d, dtype=torch.long)
        n_ok = n_ok + ok.sum(-1)
        key = torch.where(ok, irr, torch.full_like(irr, -1.0))         # rank by received light: nearest first
        ins = sec[..., None, :] == srange                                                  # [A, X, S_, C]
        jj = torch.arange(c0, c1, device=dev).expand(A, X, S_, c1 - c0)
        ck = torch.where(ins, key[..., None, :], torch.full_like(ins, -1.0, dtype=key.dtype))
        cj = torch.where(ins, jj, torch.full_like(jj, -1))
        ce = irr[..., None, :].expand(A, X, S_, c1 - c0)
        cx = xs[..., None, :].expand(A, X, S_, c1 - c0)
        srt, o = torch.sort(torch.cat((best_k, ck), -1), dim=-1, descending=True, stable=True)
        o = o[..., :kk]
        best_k = srt[..., :kk]
        best_j = torch.cat((best_j, cj), -1).gather(-1, o)
        best_e = torch.cat((best_e, ce), -1).gather(-1, o)
        best_x = torch.cat((best_x, cx), -1).gather(-1, o)
    ranked = (best_k >= 0) & (best_j >= 0)
    tested = torch.zeros_like(ranked)
    clear = torch.zeros_like(ranked)
    found = torch.zeros(A, X, S_, dtype=torch.bool, device=dev)
    hq_all = fire["h"]
    for r0 in range(0, kk, k1):                     # rounds: sectors with no clear source yet test the next ranks
        r1 = min(kk, r0 + k1)
        test = ranked[..., r0:r1] & ~found[..., None]
        if not bool(test.any()):
            break
        w = r1 - r0
        M = X * S_ * w
        jf = best_j[..., r0:r1].clamp_min(0).reshape(A, M)
        p = pos[:, :, None, None, :].expand(A, X, S_, w, 2).reshape(A, M, 2)
        q = fpos.gather(1, jf[..., None].expand(-1, -1, 2))
        hp = h[:, :, None, None].expand(A, X, S_, w).reshape(A, M)
        cl = _los_masked(geom, p, q, hp, hq_all.gather(1, jf), test.reshape(A, M), pr).reshape(A, X, S_, w)
        tested[..., r0:r1] = test
        clear[..., r0:r1] = cl
        found = found | cl.any(-1)
    blocked = torch.where(tested & ~clear, best_e, torch.zeros_like(best_e)).sum((-1, -2))
    E = (e_all - blocked).clamp_min(0)
    culled = (n_ok - tested.sum((-1, -2))).clamp_min(0)
    x = torch.where(clear, best_x, torch.zeros_like(best_x)).sum(-1) if eye else None
    return x, E, culled


def _cand(idx, off, dist, table: dict, fine: bool, snd: bool):
    """A candidate list (idx [A, N, K], -1 padded) with the target attributes of table ([A, X] each) gathered."""
    valid = idx >= 0
    j = idx.clamp_min(0)
    out = {k: _gather(v, j) for k, v in table.items()}
    out["valid"] = valid
    out["off"] = off
    out["dist"] = torch.where(valid, dist, torch.full_like(dist, math.inf))
    out["fine"] = torch.full_like(valid, fine)
    out["snd"] = torch.full_like(valid, snd)
    return out


# ============================================================================================ observe
def observe(geom: pt.PatchGeometry, bodies: Bodies, scene: Scene, *, air: Air, state=None, vol=None,
            items: Items | None = None, fires: Fires | None = None, gen=None, rules: SenseRules | None = None,
            prules: pt.PatchRules | None = None, details: bool = False):
    """The input x [A, N, IN_DIM] (float32) of every body slot (zeros for the dead); the module docstring gives the
    channels. ``geom`` the arenas, ``bodies`` (:class:`Bodies`), ``scene`` this bout's light and ground
    (:func:`scene_from_patch`), ``air`` each arena's :class:`Air` (required: there is no default planet), ``state``
    the patch state (plants, ponds, soil; touch), ``vol`` the ``patch.Volatiles`` (smell), ``items`` (:class:`Items`),
    ``fires`` (:class:`Fires`), ``gen`` the generator of the neighbour scans. With ``details`` it returns (x, info):
    theta_res, the light level, how many bodies and items were seen and heard, the hash counts (to see when K
    binds), the reach bounds and the targets whose range passes the far hash (``beyond``), and the fires culled from
    the line-of-sight tests."""
    if air is None:
        raise ValueError("observe needs each arena's Air (Air.from_partial; Air.earth is an explicit helper)")
    r = rules or SenseRules()
    pr = prules or pt.PatchRules()
    dev = geom.device
    f32 = torch.float32
    alive = torch.as_tensor(bodies.alive, device=dev).bool()
    A, N = alive.shape
    S, D, L, dx = SECTORS, VOCAL, geom.L, geom.dx
    F = lambda t: torch.as_tensor(t, dtype=f32, device=dev)          # noqa: E731
    pos = torch.remainder(F(bodies.pos), L)
    head = F(bodies.heading)
    M = F(bodies.mass_kg).clamp_min(1e-12)
    rad = F(bodies.radius_m) if bodies.radius_m is not None else body_radius(M)
    intact = 1 - F(bodies.damage).clamp(0, 1)
    hdir = heading_vector(head)
    c_air = air.sound_speed().to(dev)
    theta_res, G = eye_optics(F(bodies.eye) * M, intact, r)
    om_res = math.pi / 4 * theta_res * theta_res
    light, ground = F(scene.light_vis), F(scene.ground_vis)
    depth = water_depth(geom, state, pr)
    h_eye = r.eye_height_radii * rad + pt.bilinear(geom, depth, pos)
    surf = F(bodies.surface_vis) if bodies.surface_vis is not None else surface_reflectance(
        None if bodies.fur_m is None else F(bodies.fur_m), like=M, rules=r)
    speed = F(bodies.speed_m_s).abs()
    sound_on = air.pressure_pa.to(dev) >= r.p_min_pa                                  # [A]
    i_th = ear_threshold(F(bodies.ear) * M, intact, r)                                # [A, N]
    T_b = F(bodies.body_k)
    inf = torch.full_like(G, math.inf)
    G_best = torch.where(alive, G, torch.zeros_like(G)).amax(1)                      # the arena's best organs
    om_best = torch.where(alive, om_res, inf).amin(1)
    ith_best = torch.where(alive, i_th, inf).amin(1)
    nc = max(1, geom.n // max(1, int(r.far_cells)))
    far_radius = L / nc                                                               # the far block's sure range

    # ---- fires: what each body receives from them, and the light they shed on it
    X_emit = torch.zeros(A, N, S, device=dev)
    E_fire_body = torch.zeros(A, N, device=dev)
    fires_culled = torch.zeros(A, N, dtype=torch.long, device=dev)
    fire = None if fires is None else _fire_table(fires, geom, light, ground, depth, F)
    if fire is not None:
        X_emit, E_fire_body, fires_culled = _fire_light(geom, fire, pos, h_eye, alive, pt.sample(geom, light, pos),
                                                        r, pr, G, om_res, head)

    # ---- bodies as targets
    E_b = pt.sample(geom, light, pos) + E_fire_body
    lb_b = pt.sample(geom, ground, pos) * E_b / math.pi
    lt_b = surf * E_b / math.pi
    p_call = F(bodies.call_w) if bodies.call_w is not None else call_acoustic_power(
        F(bodies.loudness), F(bodies.voice), F(bodies.muscle), M, r)
    a_call = absorption_db_m(call_frequency(F(bodies.voice), M, c_air[:, None], r), air).to(dev)    # [A, N]
    a_noise = absorption_db_m(r.noise_hz, air).to(dev)                                               # [A]
    p_noise = footfall_power(M, speed, r)
    on2 = sound_on[:, None]
    reach_b = torch.maximum(
        vision_reach((lt_b - lb_b).abs(), lb_b, rad, G_best[:, None], om_best[:, None], r),
        torch.where(on2, torch.maximum(sound_reach(p_call, a_call, ith_best[:, None], rad),
                                       sound_reach(p_noise, a_noise[:, None], ith_best[:, None], rad)),
                    torch.zeros_like(rad)))
    reach_b = torch.where(alive, reach_b, torch.zeros_like(reach_b))
    zb = torch.zeros_like(M)
    body_tab = dict(r=rad, lt=lt_b, lb=lb_b, h=h_eye, speed=speed, mass=M, temp=T_b,
                    hard=torch.full_like(M, BODY_HARDNESS), meff=zb, pcall=p_call, acall=a_call, pnoise=p_noise)
    calls_tab = F(bodies.calls)
    idx, off, dist, count = pt.neighbours(geom, pos, alive, K=r.K, max_scan=r.max_scan, gen=gen)
    cands = [_cand(idx, off, dist, body_tab, True, True)]
    calls_c = [_gather(calls_tab, idx.clamp_min(0))]
    far_count = torch.zeros(A, N, dtype=torch.long, device=dev)
    if r.far_k > 0:
        fi, fo, fd, far_count = cross_neighbours(geom, pos, alive, pos, alive & (reach_b > dx), K=r.far_k,
                                                 max_scan=r.max_scan, gen=gen, cell=r.far_cells, reach=reach_b,
                                                 exclude_block=True)
        cands.append(_cand(fi, fo, fd, body_tab, False, True))
        calls_c.append(_gather(calls_tab, fi.clamp_min(0)))
    n_body_cols = sum(c["valid"].shape[-1] for c in cands)

    # ---- items lying on the ground as targets
    held_x = torch.zeros(A, N, GRIPS, len(HELD_CHANNELS), device=dev)
    icount = torch.zeros(A, N, dtype=torch.long, device=dev)
    ifar_count = torch.zeros(A, N, dtype=torch.long, device=dev)
    item_beyond = torch.zeros(A, dtype=torch.long, device=dev)
    if items is not None and torch.as_tensor(items.alive).shape[1] > 0:
        ipos = torch.remainder(F(items.pos), L)
        ialive = torch.as_tensor(items.alive, device=dev).bool()
        I = ialive.shape[1]
        if items.on_ground is not None:
            ground_mask = ialive & torch.as_tensor(items.on_ground, device=dev).bool()
        else:                                    # every live item no live body holds
            ground_mask = ialive.clone()
            if bodies.held is not None:
                hl = torch.as_tensor(bodies.held, device=dev).long()
                hv = (hl >= 0) & (hl < I) & alive[..., None]
                taken = torch.zeros(A, I + 1, dtype=torch.long, device=dev)
                taken.scatter_add_(1, torch.where(hv, hl, torch.full_like(hl, I)).reshape(A, -1),
                                   torch.ones_like(hl).reshape(A, -1))
                ground_mask = ground_mask & (taken[:, :I] == 0)
        irad, ivis, iem, ihard, idens = (t.to(dev) for t in _item_table(items))
        itemp = F(items.temp_k)
        imass = F(items.mass).clamp_min(0)
        dep_i = pt.sample(geom, depth, ipos)
        sunk = (dep_i > 0) & (idens > K.RHO_WATER)
        t_w = torch.where(sunk, torch.exp(-2 * r.water_kd_m * dep_i), torch.ones_like(dep_i))
        h_item = 2 * irad + torch.where(sunk, torch.zeros_like(dep_i), pt.bilinear(geom, depth, ipos))
        E_fire_item = torch.zeros_like(dep_i)
        if fire is not None:
            _, E_fire_item, _ = _fire_light(geom, fire, ipos, h_item, ground_mask, pt.sample(geom, light, ipos), r, pr)
        E_i = pt.sample(geom, light, ipos) + E_fire_item
        glow = optics.visible_exitance(itemp, iem).to(dev) * ialive * t_w
        lt_i = (ivis * E_i * t_w + glow) / math.pi
        lb_i = pt.sample(geom, ground, ipos) * E_i * t_w / math.pi
        reach_i = vision_reach(torch.maximum((lt_i - lb_i).abs(), glow / math.pi), lb_i, irad, G_best[:, None],
                               om_best[:, None], r)
        reach_i = torch.where(ground_mask, reach_i, torch.zeros_like(reach_i))
        item_beyond = (reach_i > far_radius).sum(1)
        zi = torch.zeros_like(irad)
        item_tab = dict(r=irad, lt=lt_i, lb=lb_i, h=h_item, speed=zi, mass=imass, temp=itemp, hard=ihard, meff=glow,
                        pcall=zi, acall=zi, pnoise=zi)
        ii, ioff, idist, icount = cross_neighbours(geom, pos, alive, ipos, ground_mask, K=r.K, max_scan=r.max_scan,
                                                   gen=gen)
        cands.append(_cand(ii, ioff, idist, item_tab, True, False))
        if r.far_k > 0:
            fi, fo, fd, ifar_count = cross_neighbours(geom, pos, alive, ipos, ground_mask & (reach_i > dx),
                                                      K=r.far_k, max_scan=r.max_scan, gen=gen, cell=r.far_cells,
                                                      reach=reach_i, exclude_block=True)
            cands.append(_cand(fi, fo, fd, item_tab, False, False))
        # held items
        if bodies.held is not None:
            hl = torch.as_tensor(bodies.held, device=dev).long()
            hv = (hl >= 0) & (hl < I)
            hj = hl.clamp(0, I - 1)
            hv = hv & _gather(ialive, hj) & alive[..., None]
            cols = (saturate(_gather(imass, hj) / M[..., None]), _gather(ihard, hj) / r.hardness_full,
                    _gather(F(items.sharp), hj).clamp(0, 1), torch.tanh((_gather(itemp, hj) - T_b[..., None])
                                                                         / r.thermal_k))
            held_x = torch.where(hv[..., None], torch.stack(cols, -1), torch.zeros_like(held_x))

    # ---- every candidate object: sight, glow, sound, contact
    U = {k: torch.cat([c[k] for c in cands], 2) for k in cands[0]}
    valid = U["valid"]
    d = torch.where(valid, U["dist"], torch.ones_like(U["dist"]))
    rj = U["r"]
    G3, om3, th3, ith3 = G[..., None], om_res[..., None], theta_res[..., None], i_th[..., None]
    omega = sphere_solid_angle(rj, d)
    om_bg = torch.maximum(omega, om3)
    lt, lb = U["lt"], U["lb"]
    dL = (lt - lb).abs()
    n_b = lb * om_bg * G3
    seen_u = valid & (rose_snr(dL * omega * G3, n_b) >= r.rose_k) & (dL * omega >= r.c_min * lb * om_bg)
    n_s = U["meff"] * (rj / d).clamp_max(1) ** 2 * G3
    glow_u = valid & (U["meff"] > 0) & (rose_snr(n_s, n_b) >= r.rose_k) & (n_s >= r.c_min * n_b)
    on3 = sound_on[:, None, None]
    rs = torch.maximum(d, rj)
    spread = 1.0 / (4 * math.pi * rs * rs)
    i_call_u = U["pcall"] * spread * torch.pow(10.0, -U["acall"] * rs / 10)
    i_noise_u = U["pnoise"] * spread * torch.pow(10.0, -a_noise[:, None, None] * rs / 10)
    snd = U["snd"] & valid
    heard_u = snd & on3 & ((i_call_u >= ith3) | (i_noise_u >= ith3))
    need = valid & (seen_u | glow_u | (snd & (U["fine"] | heard_u)))
    C = valid.shape[-1]
    clear = _los_masked(geom, pos[:, :, None].expand(-1, -1, C, -1).reshape(A, -1, 2),
                        (pos[:, :, None] + U["off"]).reshape(A, -1, 2),
                        h_eye[..., None].expand(-1, -1, C).reshape(A, -1), U["h"].reshape(A, -1),
                        need.reshape(A, -1), pr).reshape(A, N, C)
    seen = seen_u & clear
    glow_seen = glow_u & clear
    occ = torch.where(clear, torch.ones_like(d), torch.full_like(d, r.occlusion ** 2))
    i_call, i_noise = i_call_u * occ, i_noise_u * occ
    contrib = snd & on3 & (U["fine"] | (i_call >= ith3) | (i_noise >= ith3))
    sec = sector_of(head[..., None], U["off"])
    theta = 2 * torch.asin((rj / d).clamp(0, 1))
    # angular speed over the threshold; a target closer than its own radius (overlapping, or at the same point)
    # sweeps at most the rate at its own radius (d = 0 would make 0 / 0)
    mot = U["speed"] * r.integration_s / (rs * th3)
    order = torch.argsort(U["dist"], dim=-1, stable=True)                               # nearest first

    def srt(t):
        return t.gather(-1, order)

    o_size, o_con, o_mot = _first_per_sector(srt(seen), srt(sec), srt(saturate(theta / th3)), srt(michelson(lt, lb)),
                                             srt(saturate(mot)))
    cover = (_sector_sum(sec, torch.where(seen, theta, torch.zeros_like(theta))) / SECTOR_WIDTH).clamp(0, 1)
    flow = saturate(_sector_sum(sec, torch.where(seen, mot, torch.zeros_like(mot))))
    x_glow = torch.where(glow_seen, (n_s / n_b.clamp_min(1e-30)).clamp_max(1e30), torch.zeros_like(n_s))
    emitted = emitted_code(X_emit + _sector_sum(sec, x_glow), r)

    # hearing (the sound sources are the body columns)
    w = sensation(i_call, ith3, r) * contrib
    wb, secb = w[..., :n_body_cols], sec[..., :n_body_cols]
    calls_in = _sector_sum(secb, wb[..., None] * torch.cat(calls_c, 2))
    wsum = _sector_sum(secb, wb)
    calls_in = (calls_in / wsum.clamp_min(1.0)[..., None]).clamp(-1, 1)
    i_sec = _sector_sum(sec, torch.where(contrib, i_call + i_noise, torch.zeros_like(i_call)))
    noise = sensation(i_sec, ith3, r) * on3

    # contact under the mouth: the object with the smallest surface gap within reach
    mvec = hdir * rad[..., None] if bodies.mouth_pos is None else pt.wrap(F(bodies.mouth_pos)[..., :2] - pos, L)
    reach = (r.mouth_reach_radii * rad if bodies.reach_m is None else F(bodies.reach_m).expand_as(rad))[..., None]
    gap = (U["off"] - mvec[:, :, None]).norm(dim=-1) - rj
    touching = valid & (gap <= reach)
    kc = torch.where(touching, gap, torch.full_like(gap, math.inf)).argmin(-1, keepdim=True)
    has_c = touching.any(-1)
    t_mass = torch.where(has_c, saturate(U["mass"].gather(-1, kc)[..., 0] / M), torch.zeros_like(M))
    t_hard = torch.where(has_c, U["hard"].gather(-1, kc)[..., 0] / r.hardness_full, torch.zeros_like(M))
    t_temp = torch.where(has_c, torch.tanh((U["temp"].gather(-1, kc)[..., 0] - T_b) / r.thermal_k),
                         torch.zeros_like(M))

    # ---- the ground along each sector line
    gsec = _ground_rays(geom, pos, head, h_eye, G, light, ground, geom.elev + depth, r)

    vision = torch.stack((o_size, o_con, o_mot, cover, flow, gsec, emitted), -1)          # [A, N, S, 7]
    hearing = torch.cat((calls_in, noise[..., None]), -1)                                 # [A, N, S, D + 1]
    sector = torch.cat((vision, hearing), -1)

    # ---- smell
    smells = []
    th = smell_thresholds(r)
    step = hdir * (dx / 2)
    for name in SMELL_FIELDS:
        fld = None if vol is None else getattr(vol, name)
        if fld is None:
            smells += [torch.zeros(A, N, device=dev)] * 2
            continue
        fld = F(fld)
        cth = th[name] / intact.clamp_min(1e-6)
        c0 = pt.bilinear(geom, fld, pos)
        ahead, behind = pt.bilinear(geom, fld, pos + step), pt.bilinear(geom, fld, pos - step)
        smells += [smell_code(c0, cth, r), smell_code(ahead, cth, r) - smell_code(behind, cth, r)]
    smell = torch.stack(smells, -1)

    # ---- touch under the mouth
    mouth = torch.remainder(pos + mvec, L)
    d_mouth = pt.sample(geom, depth, mouth)
    if state is not None:
        t_plant = pt.sample(geom, pt.fapar(F(state.plant), pr), mouth)
        bucket = pr.climate.bucket_kg_m2
        soil = pt.sample(geom, F(state.soil), mouth)
        pond = pt.sample(geom, F(state.pond), mouth)
        t_damp = torch.where(d_mouth > 0, torch.ones_like(soil), (soil / bucket).clamp(0, 1))
        t_damp = torch.where(pond > 0, torch.ones_like(t_damp), t_damp)
    else:
        t_plant = torch.zeros(A, N, device=dev)
        t_damp = (d_mouth > 0).to(f32)
    t_wet = saturate(d_mouth / rad.clamp_min(1e-9))
    touch = torch.stack((t_plant, t_damp, t_wet, t_mass, t_hard, t_temp), -1)

    # ---- interoception (brain3.INTERO_NAMES order)
    gut = F(bodies.gut_kg)
    if gut.dim() == 3:
        gut = gut.sum(-1)
    E_obs = pt.sample(geom, light, pos) + E_fire_body
    lvl = light_level(E_obs, r)
    g_acc = air.g_m_s2.to(dev)[:, None]
    froude = speed / torch.sqrt(g_acc * 2 * rad).clamp_min(1e-9)
    intero = torch.stack((F(bodies.reserve_j) / F(bodies.reserve_cap_j).clamp_min(1e-12),
                          F(bodies.water_kg) / F(bodies.water_norm_kg).clamp_min(1e-12),
                          (T_b - T_LIFE_MID_K) / T_LIFE_HALF_K,
                          1 - intact, gut / F(bodies.gut_cap_kg).clamp_min(1e-12), lvl, saturate(froude)), -1)

    x = torch.cat((sector.reshape(A, N, S * SECTOR_DIM), smell, touch, held_x.reshape(A, N, -1), intero), -1)
    x = torch.where(alive[..., None], x, torch.zeros_like(x))
    if details:
        af = alive.to(f32)
        body_cols = torch.zeros_like(valid)
        body_cols[..., :n_body_cols] = True
        info = {"theta_res": theta_res * af, "light": lvl * af,
                "seen_bodies": (seen & body_cols).sum(-1).to(f32) * af,
                "seen_items": (seen & ~body_cols).sum(-1).to(f32) * af, "heard": (w > 0).sum(-1).to(f32) * af,
                "body_count": count, "item_count": icount, "K": r.K, "far_k": r.far_k,
                "bound": ((count > r.K) | (icount > r.K)) & alive,
                "far_bound": ((far_count > r.far_k) | (ifar_count > r.far_k)) & alive,
                "reach": reach_b, "beyond": alive & (reach_b > far_radius), "item_beyond": item_beyond,
                "far_radius_m": far_radius, "fires_culled": fires_culled * alive}
        return x, info
    return x


def _ground_rays(geom, pos, head, h_eye, G, light, ground, surf_z, r: SenseRules):
    """Per sector [A, N, S] the mean reflectance of the visible surface (ground or water surface surf_z [A, n, n])
    seen along the sector's centre line (``PROVENANCE['ground_rays']`` and ``['ground_visibility']``). The eye is at
    bilinear(elevation) + h_eye, as in the line-of-sight test (h_eye counts the water depth)."""
    A, N = pos.shape[:2]
    dev = pos.device
    S, T = SECTORS, int(r.ray_steps)
    dx = geom.dx
    t = (torch.arange(T, device=dev, dtype=torch.float32) + 1) * (r.ray_step_cells * dx)          # [T]
    phi = head[..., None] - torch.arange(S, device=dev) * SECTOR_WIDTH                            # [A, N, S]
    dirs = torch.stack((torch.cos(phi), torch.sin(phi)), -1)                                       # [A, N, S, 2]
    pts = pos[:, :, None, None, :] + t[:, None] * dirs[..., None, :]                               # [A, N, S, T, 2]
    z = pt.bilinear(geom, surf_z, pts)
    z_eye = (pt.bilinear(geom, geom.elev, pos) + h_eye)[..., None, None]                           # [A, N, 1, 1]
    R = geom.radius_m.to(torch.float32).reshape(A, 1, 1, 1)
    ang = (z - z_eye) / t - t / (2 * R)
    horizon = torch.cummax(ang, -1).values
    horizon = torch.cat((torch.full_like(horizon[..., :1], -math.inf), horizon[..., :-1]), -1)
    vis_ok = ang + r.ground_tolerance_m / t >= horizon
    sel = torch.tensor([int(k) - 1 for k in r.ray_fields], device=dev)
    p_sel = pts.index_select(3, sel)                                                               # [A, N, S, F, 2]
    z_sel, t_sel = z.index_select(3, sel), t.index_select(0, sel)
    rho = pt.sample(geom, ground, p_sel)
    E = pt.sample(geom, light, p_sel)
    nrm = torch.stack([pt.sample(geom, geom.normal[..., c], p_sel) for c in range(3)], -1)         # [A, N, S, F, 3]
    dz = z_eye - z_sel                                                                             # eye above sample
    to_eye = torch.cat((-dirs[..., None, :] * t_sel[:, None], dz[..., None]), -1)
    R_k = to_eye.norm(dim=-1).clamp_min(1e-6)
    cosn = ((nrm * to_eye).sum(-1) / R_k).clamp_min(0)
    area = dx * dx / nrm[..., 2].clamp_min(1e-3)
    omega = (area * cosn / (R_k * R_k)).clamp_max(2 * math.pi)
    nphot = rho * E / math.pi * omega * G[:, :, None, None]
    v = nphot / (nphot + r.rose_k ** 2)
    seen = vis_ok.index_select(3, sel).to(torch.float32)
    return (seen * v * rho).mean(-1)
