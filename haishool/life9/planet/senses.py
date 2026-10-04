"""What an individual perceives (PLANET-SPEC section 2.9): :func:`observe` -> x [W, N, IN_DIM].

The vector is the only input to the brain. It has ``SECTORS`` = 8 sectors around the heading
(sector 0 centred straight ahead, then clockwise), each with :data:`SECTOR_CHANNELS`, followed by
the individual's own state :data:`SELF_CHANNELS`. Dead slots get zeros.

Vision. Range r_v = VISION_BASE_M x acuity x light, with light from the cell's daily mean
insolation (relative to Earth's global mean S0/4) and a floor at night or on a locked planet's
dark side. Every sight line is tested against the surface of the habitat globe (the ground, or
sea level over the ocean, interpolated bilinearly within each cell from its corner values) at
points along the great circle, with the sphere's curvature (the chord between two eyes sags
R theta^2 t(1-t)/2 below their heights), so the horizon is the real one: on level ground two
animals with eyes at h see each other out to 2 sqrt(2 R h) (223 m for 30 kg animals on the
10 km habitat), and high ground sees further. Fires are seen by their own light out to
VISION_BASE_M x acuity.

Per sector:
* ``kin``: proximity (1 - d / r_v) of the nearest visible individual within a mass ratio of 2;
* ``threat``: the largest heavier one, proximity x (1 - m_self / m_other);
* ``prey``: the largest lighter one, proximity x (m_other / m_self);
* ``plant``, ``water``, ``stone``, ``ore``, ``clay``, ``wood``: cell fields sampled at 1/16, 1/8,
  1/4, 1/2 and 1 of r_v along the sector's centre line (seen at FIELD_HEIGHT_M), averaged over
  the samples (an unseen sample counts 0): plant cover 1 - exp(-plant_c / 0.5 kg C/m^2) (the
  biosphere's fAPAR), fresh water (soil >= 50 kg/m^2 on land, or a lake), deposit richness
  (:func:`richness_fields`, relative to the world's typical deposit of the class);
* ``meat``: the best visible food item on the ground, proximity x (1 - exp(-E / 10 MJ));
* ``fire``: the hottest visible fire, proximity x (T_fire - T_air) / 1000 K;
* ``call0..7``: the vocal vectors heard, each weighted by its sensation level: their weighted sum
  while the summed weight in the sector is at most 1, else their weighted mean (loudness
  saturates, the content does not).

Hearing. A call's pressure at 1 m (``creatures.call_pressure_1m``: loudness, voice, body mass and
air density) spreads spherically (pressure ~ 1/r, intensity ~ 1/r^2), is absorbed by the air
(ISO 9613-1 at 1 kHz for Earth air, scaled as the classical absorption 1/(rho c^3) with the
planet's density and speed of sound) and is halved in pressure behind terrain. The sensation
level is 20 log10(p / p_th) with the threshold p_th = 20 uPa / hearing; the weight is the level
over 60 dB, clamped to [0, 1]. Below 1 kPa of surface pressure there is no sound. Callers and
their loudness are those of the last tick (``cr.calls``, ``cr.loud``). Only pairs whose level
without occlusion (spreading and absorption, an upper bound) is above threshold, or that are in
sight range, have their line of sight tested.

Fields come as plain tensors in ``env`` (no import of climate or biosphere):
``elevation_m`` [W, C], ``sea_level_m`` [W], ``land`` [W, C] bool (globe.make_terrain), optional
``surface_v`` [W, V] (``globe.at_vertices`` of the walkable surface, kept by a caller whose terrain is static),
``t_air_k`` [W, C] K, ``soil_kg_m2`` [W, C], ``plant_c`` [W, C] kg C/m^2, ``insolation_w_m2``
[W, C] W/m^2, ``surface_pressure_pa``, ``air_density_kg_m3``, ``sound_speed_m_s`` [W]; optional
``lake`` [W, C] bool, ``richness`` [W, C, 4] in [0, 1] (stone, ore, clay, wood; else built by
:func:`richness_fields` from ``stock_kg_m2`` [W, C, S] with its ``deposit_species`` (default
materials.CRUST_SPECIES) and ``wood_c`` [W, C] when given, else 0). Fields of any float dtype
are read as float32.

Pairs of individuals are dense O(N^2) per world; sight lines are tested only for the pairs in
range, LOS_CHUNK lines at a time. No Python loops over individuals, cells or items; no
randomness.
"""
from __future__ import annotations

import math

import torch

from . import brain, materials
from . import items as items_mod
from . import globe as globe_mod
from .constants import S0

SECTORS = 8
VOCAL = brain.VOCAL_DIMS
K_SLOTS = 3
RICH_CLASSES = ("stone", "ore", "clay", "wood")
SECTOR_CHANNELS = ("kin", "threat", "prey", "plant", "water", "meat", *RICH_CLASSES, "fire",
                   *(f"call{d}" for d in range(VOCAL)))
SECTOR_DIM = len(SECTOR_CHANNELS)
SLOT_CHANNELS = ("hardness", "sharp", "mass", "fuel", "food", "temp")
SELF_CHANNELS = ("reserve", "water", "health", "age", "mass", "temp", "plant", "fresh", "light", "slope",
                 *(f"slot{k}_{c}" for k in range(K_SLOTS) for c in SLOT_CHANNELS), "in_water", "bias")
SELF0 = SECTORS * SECTOR_DIM
IN_DIM = SELF0 + len(SELF_CHANNELS)
PLANT_INDEX = SELF0 + SELF_CHANNELS.index("plant")

# ------------------------------------------------------------------------------------ set-up numbers
VISION_BASE_M = 1000.0          # new_rule: sight range per acuity level in full light
LIGHT_REF_W_M2 = S0 / 4         # reference: Earth's global daily mean insolation, 340 W/m^2
LIGHT_FLOOR = 0.05              # new_rule: night or dark-side vision, a twentieth of daylight range
EYE_K = 0.2                     # new_rule: eye height 0.2 m x M^(1/3) (geometric similarity; 0.62 m at 30 kg)
ITEM_HEIGHT_M, FIRE_HEIGHT_M = 0.1, 1.0   # new_rule: heights of a lying item and of flames
FIELD_HEIGHT_M = 0.5            # new_rule: height of what is seen of a field sample (ground cover)
LOS_SAMPLES = 8                 # numerics: terrain samples per sight line
LOS_CHUNK = 1 << 17             # numerics: sight lines tested together (memory)
RAY_FRACTIONS = (1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0)   # numerics: field samples along a sector, fractions of r_v
#: numerics: terrain samples along a sector line (fractions of r_v, sorted; they include RAY_FRACTIONS), whose
#: running horizon decides which field samples are seen
RAY_TERRAIN = tuple(sorted({1 / 64, 1 / 32, 3 / 64, *(k / 16 for k in range(1, 17))}))
KIN_RATIO = 2.0                 # new_rule: kin are within a mass ratio of 2
PLANT_COVER_C = 0.5             # derived: plant cover 1 - exp(-plant_c / 0.5), the biosphere's fAPAR (spec 2.7)
FRESH_SOIL = 50.0               # new_rule: drinkable soil (spec 2.8), as creatures.DRINK_SOIL_MIN
SOIL_FULL = 150.0               # reference: the full soil bucket, 150 kg/m^2 (Manabe 1969; spec 2.6)
MEAT_SCALE_J = 1e7              # new_rule: food item energy scale
FIRE_SCALE_K = 1000.0           # new_rule: fire excess temperature scale
WOOD_SCALE_C = 5.0              # new_rule: woody carbon scale, kg C/m^2 (forests hold about 5-20)
SLOPE_PROBE_M = 100.0           # new_rule: slope ahead measured over 100 m
TEMP_REF_K, TEMP_SCALE_K = 288.15, 30.0   # new_rule: local temperature as (T - 288.15) / 30
HARD_SCALE, ITEM_MASS_KG, ITEM_J, ITEM_TEMP_K = 10.0, 1.0, 1e7, 500.0   # new_rule: held-item scales
P_MIN_PA = 1000.0               # spec: no sound below 1 kPa surface pressure
P_HEAR_PA = 20e-6               # reference: 0 dB SPL, the threshold of human hearing near 1-4 kHz (ISO 226)
SL_FULL_DB = 60.0               # new_rule: a call 60 dB above threshold is heard at full weight
ABSORB_DB_M = 0.005             # reference: about 5 dB/km at 1 kHz, 20 C, 70 % RH (ISO 9613-1:1993)
OCCLUSION = 0.5                 # spec: terrain between halves the pressure
RHO_EARTH, C_EARTH = 1.204, 343.2  # reference: dry air at 20 C, 101.325 kPa (CRC Handbook)

PROVENANCE = {
    "layout": ("new_rule", f"PLANET-SPEC 2.9: {SECTORS} sectors x {SECTOR_DIM} channels + {len(SELF_CHANNELS)} "
                           f"self channels = IN_DIM {IN_DIM}; PLANT_INDEX {PLANT_INDEX} is the local plant input "
                           "of the innate forage wiring"),
    "VISION_BASE_M": ("new_rule", "1 km per acuity level in full light (world7 range = reach x level, "
                                  "evo/senses.py:393, has no metres)"),
    "LIGHT_REF_W_M2": ("reference", "S0 / 4 = 340 W/m^2, Earth's global daily mean insolation (Kopp & Lean 2011)"),
    "LIGHT_FLOOR": ("new_rule", "light = 0.05 + 0.95 min(1, I / 340 W/m^2): starlight and twilight vision"),
    "EYE_K": ("new_rule", "eye height 0.2 M^(1/3) m (geometric similarity: 0.2 m at 1 kg, 0.62 m at 30 kg)"),
    "horizon": ("derived", "sight lines sag R theta^2 t(1-t)/2 below the eyes' heights on the habitat sphere, so "
                           "on level ground the horizon is sqrt(2 R h) (spec 2.9)"),
    "ITEM_HEIGHT_M": ("new_rule", "a lying item 0.1 m high, flames 1 m, ground cover 0.5 m (FIELD_HEIGHT_M)"),
    "LOS_SAMPLES": ("new_rule", "numerics: 8 terrain samples per sight line, tested LOS_CHUNK = 131072 lines at a "
                                "time"),
    "terrain_surface": ("new_rule", "sight lines see the surface (ground, or sea level over the ocean) as a smooth "
                                    "surface, interpolated bilinearly within each cell from globe.at_vertices, so cell "
                                    "steps do not act as cliffs. A one-cell feature keeps only about 1/4 of its height "
                                    "(its corner vertices average it with its neighbours: a lone 10 m cell hides like a "
                                    "2.5 m rise). creatures.move climbs the same surface, so sight and movement share "
                                    "one terrain"),
    "RAY_FRACTIONS": ("new_rule", "numerics: fields sampled at 1/16, 1/8, 1/4, 1/2 and 1 of the sight range along "
                                  "each sector (geometric, since on level ground the 10 km habitat's horizon is "
                                  "about 200 m)"),
    "RAY_TERRAIN": ("new_rule", "numerics: a field sample is seen when its elevation angle (h + 0.5 m - eye) / d - d / "
                                "2R clears the running horizon of the terrain samples before it on the same sector line "
                                "(1/64, 1/32, 3/64 and k/16 of the sight range): the same test as line_of_sight at "
                                "those points, with the samples shared along the line"),
    "KIN_RATIO": ("new_rule", "kin: mass ratio within 2; threat: heavier, (1 - m/m_o); prey: lighter, m_o/m"),
    "PLANT_COVER_C": ("derived", "fAPAR = 1 - exp(-plant_c / 0.5) (spec 2.7)"),
    "FRESH_SOIL": ("new_rule", "spec 2.8 modelling choice: drinkable where soil > 50 kg/m^2 or a lake"),
    "SOIL_FULL": ("reference", "the soil bucket's field capacity 150 kg/m^2 = 15 cm of water (Manabe 1969, Mon. Wea. "
                               "Rev. 97:739; spec 2.6): the local water channel is soil / 150"),
    "MEAT_SCALE_J": ("new_rule", "food items: 1 - exp(-E / 10 MJ)"),
    "FIRE_SCALE_K": ("new_rule", "fires: (T_fire - T_air) / 1000 K"),
    "WOOD_SCALE_C": ("new_rule", "woody carbon 1 - exp(-wood_c / 5 kg C/m^2)"),
    "richness": ("new_rule", "stone, ore, clay (by materials.CLASS of the deposit species): 1 - exp(-x / x_typ), x "
                             "the class's accessible stock (kg/m^2) in the cell and x_typ its mean over the world's "
                             "cells that hold any, so a typical deposit reads 0.63 whatever its share of the column "
                             "(ores are about 1e-3 of it)"),
    "calls_heard": ("new_rule", "the calls in a sector: sum of weight x vocal vector while the summed weight is at "
                                "most 1, else the weighted mean (sum / summed weight), so a crowd's calls keep their "
                                "content instead of saturating at +-1"),
    "SLOPE_PROBE_M": ("new_rule", "slope ahead = tanh(rise over 100 m / 100 m)"),
    "TEMP_SCALE_K": ("new_rule", "local temperature (T - 288.15) / 30 K"),
    "held_items": ("new_rule", "per slot: tool hardness / 10 (Mohs), sharp, tanh(mass / 1 kg), tanh(fuel / 10 MJ), "
                               "tanh(digestible food for the holder's diet / 10 MJ), tanh((T_item - T_air) / 500 K)"),
    "P_MIN_PA": ("new_rule", "spec 2.9: no sound below 1 kPa"),
    "P_HEAR_PA": ("reference", "20 uPa = 0 dB SPL, the human threshold near 1-4 kHz (ISO 226:2003); the hearing "
                               "gene divides it (world7 ears level: range proportional to level)"),
    "SL_FULL_DB": ("new_rule", "weight = sensation level / 60 dB, clamped to [0, 1]"),
    "ABSORB_DB_M": ("reference", "about 5 dB/km at 1 kHz, 20 C, 70 % RH (ISO 9613-1:1993); scaled by "
                                 "(rho_E c_E^3) / (rho c^3) (classical Stokes-Kirchhoff absorption, new_rule)"),
    "spreading": ("derived", "spherical spreading: pressure 1/r (intensity 1/r^2) from the call's 1 m pressure"),
    "OCCLUSION": ("new_rule", "spec 2.9: terrain between halves the sound pressure; sight is blocked"),
    "RHO_EARTH": ("reference", "dry air at 20 C, 101.325 kPa: 1.204 kg/m^3, 343.2 m/s (CRC Handbook)"),
}


def sector_index(sector: int, channel: str) -> int:
    """Input index of a sector channel."""
    return sector * SECTOR_DIM + SECTOR_CHANNELS.index(channel)


def self_index(channel: str) -> int:
    """Input index of a self channel."""
    return SELF0 + SELF_CHANNELS.index(channel)


# ------------------------------------------------------------------------------------ geometry
def surface_at(globe, surf_v, w, pts):
    """Height (m) of the surface at points pts [..., 3] of worlds w [...] (broadcast): bilinear in the
    gnomonic angles of each cube-sphere cell between its four corner values surf_v [W, V]
    (``globe.at_vertices`` of the cell field), so it is continuous across cells."""
    G_ = globe.G
    frame = globe._frame.to(pts.dtype)            # face frames (normal, u, v), as Globe.cell_of uses
    face = (pts @ frame[:, 0].T).argmax(-1)
    n, u, v = frame[face, 0], frame[face, 1], frame[face, 2]
    pn = (pts * n).sum(-1)
    step = math.pi / (2 * G_)
    a = (torch.atan((pts * u).sum(-1) / pn) + math.pi / 4) / step
    b = (torch.atan((pts * v).sum(-1) / pn) + math.pi / 4) / step
    i, j = a.floor().clamp(0, G_ - 1), b.floor().clamp(0, G_ - 1)
    fa, fb = (a - i).clamp(0, 1), (b - j).clamp(0, 1)
    corner = globe.corner_id[face * G_ * G_ + i.long() * G_ + j.long()]          # [..., 4]: v00 v10 v11 v01
    val = surf_v[w[..., None].expand(corner.shape), corner]
    return ((1 - fa) * (1 - fb) * val[..., 0] + fa * (1 - fb) * val[..., 1] + fa * fb * val[..., 2]
            + (1 - fa) * fb * val[..., 3])


def line_of_sight(globe, surf_v, w, p, q, h_p, h_q, radius, samples=LOS_SAMPLES, chunk=LOS_CHUNK, floor=None):
    """bool [P]: the sight line from height h_p (m above the habitat sphere) at p [P, 3] to h_q at q
    clears the surface (vertex values surf_v [W, V], see :func:`surface_at`) of world w [P] at
    ``samples`` interior points of the great circle. The straight chord sags radius x theta^2
    t(1 - t) / 2 below the linear interpolation of the two heights (curvature of the sphere).
    Lines are tested ``chunk`` at a time (memory). ``floor`` [W] (the lowest vertex value of each world,
    ``surf_v.amin(-1)``) lets a line whose sampled heights dip below its world's lowest surface fail at once
    (the surface there is at least the floor, so the full test would fail it too); only the others are
    sampled against the surface."""
    P = p.shape[0]
    if P > chunk and floor is None:
        return torch.cat([line_of_sight(globe, surf_v, w[i:i + chunk], p[i:i + chunk], q[i:i + chunk],
                                        h_p[i:i + chunk], h_q[i:i + chunk], radius[i:i + chunk], samples, chunk)
                          for i in range(0, P, chunk)])
    if P == 0:
        return torch.zeros(0, dtype=torch.bool, device=p.device)
    t = (torch.arange(samples, device=p.device, dtype=p.dtype) + 1) / (samples + 1)
    theta = globe_mod.angle(p, q)[:, None]
    line = (1 - t) * h_p[:, None] + t * h_q[:, None] - radius[:, None] * theta ** 2 * t * (1 - t) / 2
    if floor is not None:
        maybe = (line >= floor[w][:, None]).all(-1)
        out = torch.zeros(P, dtype=torch.bool, device=p.device)
        k = maybe.nonzero(as_tuple=True)[0]
        if k.numel():
            out[k] = line_of_sight(globe, surf_v, w[k], p[k], q[k], h_p[k], h_q[k], radius[k], samples, chunk)
        return out
    pts = globe_mod.slerp(p[:, None], q[:, None], t[None, :])                    # [P, K, 3]
    terr = surface_at(globe, surf_v, w[:, None].expand(-1, samples), pts)         # [P, K]
    return (terr <= line).all(-1)


def sector_of(bearing):
    """Sector index of a bearing relative to the heading (sector 0 centred ahead, clockwise)."""
    width = 2 * math.pi / SECTORS
    return (torch.remainder(bearing + width / 2, 2 * math.pi) / width).long().clamp(0, SECTORS - 1)


def _pair_angle(pos):
    """[W, N, N] angle between all pairs (chord formula; for range pre-selection)."""
    dot = torch.bmm(pos, pos.transpose(1, 2)).clamp(-1.0, 1.0)
    return 2 * torch.asin((((2 - 2 * dot).clamp_min(0)).sqrt() / 2).clamp(max=1.0))


def _cross_angle(a, b):
    """[W, N, M] angle between points a [W, N, 3] and b [W, M, 3]."""
    dot = torch.bmm(a, b.transpose(1, 2)).clamp(-1.0, 1.0)
    return 2 * torch.asin((((2 - 2 * dot).clamp_min(0)).sqrt() / 2).clamp(max=1.0))


def _compact(mask):
    """Indices [W, P] of the True entries of mask [W, X] first (P = the largest count) and their validity."""
    W, X = mask.shape
    P = int(mask.sum(1).max()) if mask.numel() else 0
    order = torch.sort((~mask).to(torch.int8), dim=1, stable=True).indices[:, :P]
    return order, mask.gather(1, order)


def richness_fields(stock_kg_m2=None, wood_c=None, W=None, C=None, device="cpu", species=None):
    """[W, C, 4] richness in [0, 1] of stone, ore, clay and wood.

    ``stock_kg_m2`` [W, C, S] is the accessible ground stock of ``species`` (the sequence passed to
    globe.make_deposits; default materials.CRUST_SPECIES, or materials.SPECIES, whichever has S
    entries, else ValueError). Each class (materials.CLASS) reads 1 - exp(-x / x_typ): x its stock
    in the cell, x_typ its mean over the world's cells that hold any. Wood: 1 - exp(-wood_c /
    WOOD_SCALE_C)."""
    if stock_kg_m2 is not None:
        W, C = stock_kg_m2.shape[:2]
        device = stock_kg_m2.device
    elif wood_c is not None:
        W, C = wood_c.shape
        device = wood_c.device
    out = torch.zeros(W, C, 4, device=device)
    if stock_kg_m2 is not None:
        S = stock_kg_m2.shape[-1]
        if species is None:
            species = next((names for names in (materials.CRUST_SPECIES, materials.SPECIES) if len(names) == S), None)
        if species is None or len(species) != S:
            raise ValueError(f"stock_kg_m2 has {S} species; pass the species sequence it was made with")
        stock = stock_kg_m2.float().clamp_min(0)
        for j, cls in enumerate(RICH_CLASSES[:3]):
            sel = torch.tensor([materials.CLASS[s] == cls for s in species], dtype=torch.bool, device=device)
            x = stock[..., sel].sum(-1)                                                    # [W, C]
            some = x > 0
            typ = (x.sum(-1, keepdim=True) / some.sum(-1, keepdim=True).clamp_min(1)).clamp_min(1e-30)
            out[..., j] = torch.where(some, 1 - torch.exp(-x / typ), torch.zeros_like(x))
    if wood_c is not None:
        out[..., 3] = 1 - torch.exp(-wood_c.float().clamp_min(0) / WOOD_SCALE_C)
    return out


def field_rays(globe, surf_v, pos, heading, r_v, eye_h, radius):
    """The field samples of each sector line and whether they are seen (PROVENANCE['RAY_TERRAIN']).

    pos [W, N, 3], heading [W, N], r_v [W, N] the sight range, eye_h [W, N] the eyes' height (m above the
    sphere), radius [W]. The surface is sampled at RAY_TERRAIN x r_v along each of the SECTORS lines; a
    sample's elevation angle is (h - eye) / d - d / (2 R) (the curvature drop), and a field sample (seen at
    FIELD_HEIGHT_M above the surface) is seen when its angle is at least the running maximum of the terrain
    angles before it: the same decision as :func:`line_of_sight` would make with those earlier samples.
    Returns pts [W, N, S, Rr, 3] (RAY_FRACTIONS), seen [W, N, S, Rr] bool and, for checks, the terrain
    samples ``pts_all`` [W, N, S, T, 3], ``h_all`` [W, N, S, T] and their distances ``d_all`` [W, N, 1, T]."""
    W, N = pos.shape[:2]
    dev = pos.device
    T_ = len(RAY_TERRAIN)
    tfrac = torch.tensor(RAY_TERRAIN, device=dev)
    at_field = torch.tensor([RAY_TERRAIN.index(f) for f in RAY_FRACTIONS], device=dev)
    sec_head = heading[..., None] + torch.arange(SECTORS, device=dev) * (2 * math.pi / SECTORS)   # [W, N, S]
    d_ray = r_v[..., None, None] * tfrac                                                         # [W, N, 1, T]
    R4 = torch.as_tensor(radius, dtype=pos.dtype, device=dev).reshape(-1).expand(W).reshape(W, 1, 1, 1)
    pts_all, _ = globe_mod.move(pos[:, :, None, None, :].expand(W, N, SECTORS, T_, 3),
                                sec_head[..., None].expand(W, N, SECTORS, T_), d_ray, R4)
    wn = torch.arange(W, device=dev)[:, None, None, None].expand(W, N, SECTORS, T_)
    h_all = surface_at(globe, surf_v, wn, pts_all)                                              # [W, N, S, T]
    eye4 = eye_h[..., None, None]
    d_safe = d_ray.clamp_min(1e-6)
    sag = d_ray / (2 * R4)                                                                      # curvature drop / d
    terrain_ang = (h_all - eye4) / d_safe - sag
    target_ang = (h_all + FIELD_HEIGHT_M - eye4) / d_safe - sag
    horizon = torch.cummax(terrain_ang, -1).values
    horizon = torch.cat((torch.full_like(horizon[..., :1], -math.inf), horizon[..., :-1]), -1)
    return {"pts": pts_all.index_select(-2, at_field), "seen": (target_ang >= horizon).index_select(-1, at_field),
            "pts_all": pts_all, "h_all": h_all, "d_all": d_ray}


# ------------------------------------------------------------------------------------ observe
def observe(cr, globe, env, pool=None, fires=None, *, radius_m, details=False):
    """The brain input x [W, N, IN_DIM] of every slot (zeros for the dead); see the module doc for
    the channels and ``env``. ``pool`` (items.ItemPool) gives the food items on the ground and
    the held items' properties; ``fires`` (items.FirePool) the fires. With ``details`` it returns
    (x, info) with the sight range r_v, light, the summed call weight heard and the number of
    individuals seen, per slot."""
    from . import creatures as C_
    W, N = cr.shape
    dev = cr.device
    env = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in env.items()}
    S, D = SECTORS, VOCAL
    alive = cr.alive
    af = alive.float()
    g = cr.genome
    pos, head = cr.pos, cr.heading
    R = C_.per_world(radius_m, W, dev)                                           # [W, 1]
    Rf = R[:, 0]
    M = cr.mass_kg.clamp_min(1e-6)
    surf_v = env.get("surface_v")                    # the caller may keep it (the terrain is static)
    if surf_v is None:
        surf_v = globe.at_vertices(C_.surface_m(env))
    land = env["land"]
    cell = globe.cell_of(pos)
    wn = torch.arange(W, device=dev)[:, None].expand(W, N)
    ground = surface_at(globe, surf_v, wn, pos)
    floor = surf_v.amin(-1)                                  # the lowest surface: lines below it fail at once
    eye = EYE_K * M ** (1 / 3)
    t_loc = env["t_air_k"].gather(1, cell)
    insol = env["insolation_w_m2"].gather(1, cell)
    light = LIGHT_FLOOR + (1 - LIGHT_FLOOR) * (insol / LIGHT_REF_W_M2).clamp(0, 1)
    r_v = VISION_BASE_M * g["acuity"].clamp_min(0) * light * af
    r_lum = VISION_BASE_M * g["acuity"].clamp_min(0) * af
    flat = W * N * S
    sec = {name: torch.zeros(flat, device=dev) for name in ("kin", "threat", "prey", "meat", "fire")}
    calls_in = torch.zeros(flat, D, device=dev)

    def amax(name, w, n, s, v):
        sec[name].scatter_reduce_(0, (w * N + n) * S + s, v, "amax", include_self=True)

    # ---- individuals: sight and sound
    dist = _pair_angle(pos) * R[:, :, None]                                      # [W, N, N]
    others = alive[:, :, None] & alive[:, None, :] & ~torch.eye(N, dtype=torch.bool, device=dev)
    see = others & (dist < r_v[..., None])
    rho = C_.per_world(env.get("air_density_kg_m3", RHO_EARTH), W, dev)
    c_snd = C_.per_world(env.get("sound_speed_m_s", C_EARTH), W, dev)
    p_surf = C_.per_world(env.get("surface_pressure_pa", 101325.0), W, dev)
    p1 = C_.call_pressure_1m(cr.loud, g["voice"], M, rho) * (p_surf >= P_MIN_PA) * af   # [W, N] callers
    hearing = g["hearing"].clamp_min(0)
    p_th = P_HEAR_PA / hearing.clamp_min(1e-6)
    alpha = ABSORB_DB_M * (RHO_EARTH * C_EARTH ** 3) / (rho[:, 0] * c_snd[:, 0] ** 3)          # dB/m [W]
    # the level without occlusion (an upper bound): only pairs above threshold need a line of sight
    r_all = dist.clamp_min(1.0)
    level_open = (20 * (torch.log10(p1.clamp_min(1e-30))[:, None, :] - torch.log10(p_th)[:, :, None]
                        - torch.log10(r_all)) - alpha[:, None, None] * r_all)
    hear = others & (p1[:, None, :] > 0) & (hearing[..., None] > 0) & (level_open > 0)
    del level_open, r_all
    wi, ni, mi = (see | hear).nonzero(as_tuple=True)
    heard_sum = torch.zeros(W, N, device=dev)
    seen_n = torch.zeros(W, N, device=dev)
    call_w = torch.zeros(flat, device=dev)
    if wi.numel():
        p, q = pos[wi, ni], pos[wi, mi]
        d = globe_mod.angle(p, q) * Rf[wi]
        clear = line_of_sight(globe, surf_v, wi, p, q, ground[wi, ni] + eye[wi, ni], ground[wi, mi] + eye[wi, mi],
                              Rf[wi], floor=floor)
        s = sector_of(globe_mod.bearing(p, q, head[wi, ni]))
        vis = see[wi, ni, mi] & clear & (d < r_v[wi, ni])
        prox = (1 - d / r_v[wi, ni].clamp_min(1e-6)).clamp(0, 1) * vis
        ratio = M[wi, mi] / M[wi, ni]
        kin = (ratio <= KIN_RATIO) & (ratio >= 1 / KIN_RATIO)
        amax("kin", wi, ni, s, prox * kin)
        amax("threat", wi, ni, s, prox * (ratio > 1) * (1 - 1 / ratio))
        amax("prey", wi, ni, s, prox * (ratio < 1) * ratio)
        seen_n.index_put_((wi, ni), vis.float(), accumulate=True)
        # sound: spherical spreading, absorption, occlusion
        r = d.clamp_min(1.0)
        occ = torch.where(clear, torch.ones_like(r), torch.full_like(r, OCCLUSION))
        level = 20 * torch.log10((p1[wi, mi] * occ / (r * p_th[wi, ni])).clamp_min(1e-30)) - alpha[wi] * r
        wgt = (level / SL_FULL_DB).clamp(0, 1) * hear[wi, ni, mi]
        key = (wi * N + ni) * S + s
        calls_in.index_add_(0, key, wgt[:, None] * cr.calls[wi, mi])
        call_w.index_add_(0, key, wgt)
        heard_sum.index_put_((wi, ni), wgt, accumulate=True)
    # the weighted sum while the sector's summed weight is at most 1, else the weighted mean
    calls_in = (calls_in / call_w.clamp_min(1.0)[:, None]).clamp(-1, 1)
    pairs = torch.bincount(wi, minlength=W)

    # ---- food items on the ground (any item with food energy: carcass parts, fibre)
    props = items_mod.item_props(pool) if pool is not None else None
    if pool is not None:
        food = pool.alive & (pool.holder < 0) & (props["food_j"] > 0)
        idx, valid = _compact(food)
        if idx.shape[1]:
            ipos = pool.pos.gather(1, idx[..., None].expand(-1, -1, 3))
            d_all = _cross_angle(pos, ipos) * R[:, :, None]
            cand = alive[..., None] & valid[:, None, :] & (d_all < r_v[..., None])
            wi, ni, ji = cand.nonzero(as_tuple=True)
            if wi.numel():
                it = idx[wi, ji]
                p, q = pos[wi, ni], pool.pos[wi, it]
                d = globe_mod.angle(p, q) * Rf[wi]
                h_q = surface_at(globe, surf_v, wi, q) + ITEM_HEIGHT_M
                clear = line_of_sight(globe, surf_v, wi, p, q, ground[wi, ni] + eye[wi, ni], h_q, Rf[wi], floor=floor)
                s = sector_of(globe_mod.bearing(p, q, head[wi, ni]))
                prox = (1 - d / r_v[wi, ni].clamp_min(1e-6)).clamp(0, 1) * clear
                amax("meat", wi, ni, s, prox * (1 - torch.exp(-props["food_j"][wi, it] / MEAT_SCALE_J)))

    # ---- fires, seen by their own light
    if fires is not None and fires.alive.shape[1]:
        fpos = fires.pos
        d_all = _cross_angle(pos, fpos) * R[:, :, None]
        cand = alive[..., None] & fires.alive[:, None, :] & (d_all < r_lum[..., None])
        wi, ni, fi = cand.nonzero(as_tuple=True)
        if wi.numel():
            p, q = pos[wi, ni], fpos[wi, fi]
            d = globe_mod.angle(p, q) * Rf[wi]
            h_q = surface_at(globe, surf_v, wi, q) + FIRE_HEIGHT_M
            clear = line_of_sight(globe, surf_v, wi, p, q, ground[wi, ni] + eye[wi, ni], h_q, Rf[wi], floor=floor)
            s = sector_of(globe_mod.bearing(p, q, head[wi, ni]))
            prox = (1 - d / r_lum[wi, ni].clamp_min(1e-6)).clamp(0, 1) * clear
            heat = ((fires.temp_k[wi, fi] - t_loc[wi, ni]) / FIRE_SCALE_K).clamp(0, 1)
            amax("fire", wi, ni, s, prox * heat)

    # ---- cell fields along each sector's centre line: terrain samples, their running horizon (RAY_TERRAIN)
    Rr = len(RAY_FRACTIONS)
    rays = field_rays(globe, surf_v, pos, head, r_v, ground + eye, Rf)
    seen = (rays["seen"] & alive[..., None, None]).float()                                     # [W, N, S, Rr]
    rcell = globe.cell_of(rays["pts"])                                                          # [W, N, S, Rr]
    flat_cell = rcell.reshape(W, -1)

    def along(field):
        return (field.gather(1, flat_cell).reshape(W, N, S, Rr) * seen).mean(-1)

    lake = env.get("lake")
    fresh = land & (env["soil_kg_m2"] >= FRESH_SOIL)
    if lake is not None:
        fresh = fresh | (lake & land)
    plant_cover = 1 - torch.exp(-env["plant_c"].clamp_min(0) / PLANT_COVER_C)
    rich = env.get("richness")
    if rich is None:
        rich = richness_fields(env.get("stock_kg_m2"), env.get("wood_c"), W, globe.C, dev,
                               species=env.get("deposit_species"))
    sector = torch.zeros(W, N, S, SECTOR_DIM, device=dev)
    ch = SECTOR_CHANNELS.index
    for name in ("kin", "threat", "prey", "meat", "fire"):
        sector[..., ch(name)] = sec[name].reshape(W, N, S)
    sector[..., ch("plant")] = along(plant_cover)
    sector[..., ch("water")] = along(fresh.float())
    for j, name in enumerate(RICH_CLASSES):
        sector[..., ch(name)] = along(rich[..., j])
    sector[..., ch("call0"):ch("call0") + D] = calls_in.reshape(W, N, S, D)

    # ---- self
    adult = g["adult_mass_kg"].clamp_min(1e-6)
    ahead, _ = globe_mod.move(pos, head, torch.full_like(head, SLOPE_PROBE_M), R)
    slope = torch.tanh((surface_at(globe, surf_v, wn, ahead) - ground) / SLOPE_PROBE_M)
    soil_loc = env["soil_kg_m2"].gather(1, cell)
    land_loc = land.gather(1, cell)
    fresh_loc = torch.where(land_loc, (soil_loc / SOIL_FULL).clamp(0, 1), torch.zeros_like(soil_loc))
    if lake is not None:
        fresh_loc = torch.where(lake.gather(1, cell) & land_loc, torch.ones_like(fresh_loc), fresh_loc)
    own = [cr.reserve_j / C_.reserve_max(M), cr.water_kg / C_.water_norm(M), cr.health,
           cr.age_d / C_.lifespan_d(adult), cr.mass_kg / adult, (t_loc - TEMP_REF_K) / TEMP_SCALE_K,
           plant_cover.gather(1, cell), fresh_loc, light, slope]
    slots = torch.zeros(W, N, K_SLOTS, len(SLOT_CHANNELS), device=dev)
    if pool is not None:
        inv = cr.inv[..., :K_SLOTS]
        has = inv >= 0
        ix = inv.clamp_min(0).reshape(W, -1)
        get = lambda t: t.gather(1, ix).reshape(W, N, -1) * has
        diet = g["diet"][..., None]
        food = get(props["food_plant_j"]) * (1 - diet) + get(props["food_meat_j"]) * diet
        temp = get(pool.temp_k) - t_loc[..., None] * has
        cols = [get(props["tool_hardness"]) / HARD_SCALE, get(pool.sharp), torch.tanh(get(pool.mass) / ITEM_MASS_KG),
                torch.tanh(get(props["fuel_j"]) / ITEM_J), torch.tanh(food / ITEM_J), torch.tanh(temp / ITEM_TEMP_K)]
        slots[:, :, :inv.shape[-1]] = torch.stack(cols, -1)
    own_t = torch.stack(own, -1)
    tail = torch.stack([(~land_loc).float(), torch.ones_like(light)], -1)
    x = torch.cat((sector.reshape(W, N, S * SECTOR_DIM), own_t, slots.reshape(W, N, -1), tail), -1)
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0) * af[..., None]
    if details:
        return x, {"r_v": r_v, "light": light * af, "heard": heard_sum, "seen": seen_n, "pairs": pairs}
    return x
