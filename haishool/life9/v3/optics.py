"""Optical table of life9 v3 (PLANET-V3-SPEC section 4): reflectance and emission of everything an eye can see.

For each surface the table gives three numbers, each a :class:`Num` (a float like ``materials.Val``) with its
provenance tag and source:

* ``VIS``: visible-band (400-700 nm) reflectance, what an eye compares against the background (contrast);
* ``SW``: broadband shortwave albedo (0.3-3 um), for heat balances in sunlight;
* ``EMISSIVITY``: thermal-infrared emissivity near 300 K (8-14 um window or total hemispherical).

Entries cover the ground covers (green plant cover, plant litter, open water, snow, ice, a generic bare soil), every
``materials.SPECIES`` (rocks, ores, organics and products: what items and the ground's crust are made of) and the
body tissues (fur, skin, flesh, fat, bone). Fire is not a reflector: it emits. :func:`visible_exitance` gives the
visible light (W/m^2) a surface of temperature T and emissivity e radiates (e sigma T^4 times the Planck share in
400-700 nm), so a flame of about 1,100 K glows and a warm stone does not. ``FLAME_EMISSIVITY`` is the emissivity of a
small open flame.

Nothing here labels a surface for a body: the senses get reflectances and emitted light, never names. Values are
mid-range picks from published ranges; where a pick inside the range is ours the tag is ``reference+new_rule``.

Mixtures (:func:`mix`) reflect by area share; an item's or a deposit's area share is taken as its mass share
(new_rule: grains of similar size).
"""
from __future__ import annotations

import math

import numpy as np
import torch

from haishool.life9.planet import constants as K
from haishool.life9.planet import materials

R, N, D = materials.REFERENCE, materials.NEW_RULE, materials.DERIVED
RN = "reference+new_rule"

_OKE = "Oke 1987, Boundary Layer Climates, 2nd ed., table 1.1"
_INC = "Incropera, DeWitt, Bergman & Lavine 2007, Fundamentals of Heat and Mass Transfer, 6th ed., table A.11"
_SD = "Salisbury & D'Aria 1992, Remote Sens. Environ. 42, 83 (8-14 um emissivity of terrestrial materials)"
_USGS = "USGS Spectral Library version 7 (Kokaly et al. 2017, USGS Data Series 1035)"
_METAL = ("polished metals reflect most visible light and emit little: copper 0.03-0.05, tin 0.04-0.06, iron and "
          f"steel 0.05-0.1 polished, 0.6-0.8 oxidised ({_INC})")
_FRESNEL = "Fresnel reflectance ((n - 1) / (n + 1))^2 = 0.04 per surface at n = 1.5, two surfaces"


class Num(float):
    """A number with its provenance like ``materials.Val``, whose tag may combine tags ("reference+new_rule")."""

    def __new__(cls, value: float, tag: str, source: str):
        if not tag or any(part not in materials.TAGS for part in tag.split("+")):
            raise ValueError(f"unknown provenance tag {tag!r}")
        x = super().__new__(cls, value)
        x.tag, x.source = tag, source
        return x

    def __reduce__(self):
        return (Num, (float(self), self.tag, self.source))


def _v(value: float, tag: str, source: str) -> Num:
    return Num(value, tag, source)


def _rows(rows: dict) -> tuple[dict, dict, dict]:
    vis, sw, em = {}, {}, {}
    for name, (a_vis, a_sw, e, tag, source) in rows.items():
        vis[name] = _v(a_vis, tag, source)
        sw[name] = _v(a_sw, tag, source)
        em[name] = _v(e, tag, source)
    return vis, sw, em


# ---------------------------------------------------------------------------------------------- ground covers
GROUND_ROWS = {
    # name: (visible reflectance, shortwave albedo, IR emissivity, tag, source)
    "plant": (0.05, 0.15, 0.97, RN,
              "green leaves absorb most visible light (reflectance about 0.05, 0.10 at the 550 nm green peak; Gates, "
              "Keegan, Schleter & Weidner 1965, Appl. Opt. 4, 11); vegetated land shortwave 0.15 (forests 0.10-0.18, "
              "grass 0.16-0.26; Hartmann 2016, Global Physical Climatology, table 4.2; the climate's "
              "albedo_vegetation); "
              f"emissivity 0.97 (green foliage 0.96-0.99, {_SD})"),
    "litter": (0.25, 0.25, 0.95, RN,
               "dry dead leaves and straw 0.2-0.3 in the visible (senescent foliage loses chlorophyll absorption; "
               f"{_USGS}); emissivity 0.92-0.96 (dry vegetation, {_SD})"),
    "water": (0.06, 0.06, 0.96, R,
              "open water 0.06 for diffuse light (Payne 1972, J. Atmos. Sci. 29, 959; the direct beam follows "
              "climate.ocean_albedo by sun angle); emissivity 0.96 (" + _INC + ")"),
    "snow": (0.95, 0.80, 0.98, R,
             "fresh snow: visible 0.95-0.98, broadband 0.8-0.9, aged snow lower (Warren 1982, Rev. Geophys. 20, 67); "
             "snow is nearly black in the thermal infrared, 0.97-0.99 (Warren 1982). The climate's 0.6 is its mean for "
             "aged and patchy snow and ice"),
    "ice": (0.50, 0.40, 0.97, RN, f"glacier ice 0.20-0.40 broadband, sea ice 0.30-0.45 ({_OKE}); emissivity 0.95-0.98 "
                                  f"({_INC})"),
    "soil": (0.20, 0.25, 0.95, RN,
             f"bare soils 0.05 (dark, wet) to 0.40 (light, dry) ({_OKE}); bare land 0.25 broadband (Hartmann 2016, the "
             f"climate's albedo_land); emissivity 0.93-0.96 ({_INC})"),
}

# ---------------------------------------------------------------------------------------------- materials.SPECIES
SPECIES_ROWS = {
    "basalt": (0.07, 0.10, 0.93, RN, f"dark mafic rock, visible 0.05-0.10 ({_USGS}); emissivity 0.90-0.95 ({_SD})"),
    "granite": (0.25, 0.30, 0.90, RN, f"light felsic rock 0.2-0.35 ({_USGS}); emissivity 0.88-0.92 ({_SD})"),
    "flint": (0.10, 0.12, 0.88, RN, f"chert and flint are dark grey to black, 0.05-0.15 ({_USGS}); quartz-rich rocks "
                                    f"have "
                                    f"a lower 8-14 um emissivity, 0.80-0.90 ({_SD})"),
    "sandstone": (0.30, 0.35, 0.88, RN, f"quartz arenite 0.25-0.40 ({_USGS}); quartz-rich 0.80-0.90 ({_SD})"),
    "limestone": (0.45, 0.50, 0.94, RN, f"light carbonate rock 0.35-0.60 ({_USGS}); carbonates 0.92-0.96 ({_SD})"),
    "clay": (0.55, 0.50, 0.94, RN, f"kaolinite is white, 0.5-0.8 in the visible ({_USGS}); clay soils 0.16-0.23 when "
                                   f"wet "
                                   f"({_OKE}); emissivity 0.92-0.96 ({_SD})"),
    "sand": (0.40, 0.40, 0.90, R, f"dry sand 0.35-0.45 ({_OKE}); sand 0.90 ({_INC})"),
    "salt": (0.75, 0.70, 0.90, RN, f"halite crusts and salt flats 0.5-0.9 ({_USGS}); emissivity of salt crusts about "
                                   f"0.9 "
                                   "(new_rule: a crystal of halite is transparent in the infrared, a crust is not)"),
    "hematite": (0.12, 0.15, 0.90, RN, f"red-brown to steel-grey, 0.05-0.30 rising to the red ({_USGS}); oxides "
                                       f"0.85-0.95 "
                                       f"({_SD})"),
    "magnetite": (0.05, 0.05, 0.90, RN, f"black, 0.03-0.07 ({_USGS}); emissivity about 0.9 ({_SD})"),
    "malachite": (0.12, 0.15, 0.90, RN, f"green, 0.05-0.25 with its peak in the green ({_USGS}); emissivity about 0.9 "
                                        f"({_SD})"),
    "cassiterite": (0.10, 0.12, 0.90, RN, f"brown-black, 0.05-0.15 ({_USGS}); emissivity about 0.9 ({_SD})"),
    "native_copper": (0.60, 0.70, 0.05, RN, "copper reflects 0.5 at 500 nm rising to 0.95 at 700 nm (Johnson & Christy "
                                            f"1972, Phys. Rev. B 6, 4370); {_METAL}"),
    "pyrite": (0.50, 0.50, 0.40, RN, "pyrite reflects about 0.54 at 589 nm (Criddle & Stanley 1993, Quantitative Data "
                                     "File for Ore Minerals, 3rd ed.); emissivity: new_rule 0.4, between metals and "
                                     "oxides (a metallic-lustred sulfide)"),
    "wood": (0.35, 0.40, 0.90, RN, f"fresh-cut wood 0.3-0.5, bark 0.1-0.25 ({_USGS}); wood 0.82-0.92 ({_INC})"),
    "plant_fiber": (0.50, 0.50, 0.90, RN, f"dry plant fibre and straw 0.4-0.6 ({_USGS}); dry vegetation 0.88-0.94 "
                                          f"({_SD})"),
    "resin": (0.20, 0.25, 0.92, RN, "amber-coloured, translucent rosin about 0.1-0.3 (new_rule: no spectral "
                                    "measurement "
                                    "used); organic solids 0.9-0.95"),
    "meat": (0.20, 0.25, 0.95, RN, "skeletal muscle reflects 0.1-0.3 across the visible, red-weighted (Jacques 2013, "
                                   "Phys. Med. Biol. 58, R37); moist tissue emits like water, 0.95-0.98"),
    "fat": (0.60, 0.60, 0.95, RN, "adipose tissue is white to yellow, 0.5-0.7 (Jacques 2013); moist tissue 0.95"),
    "bone": (0.60, 0.60, 0.90, RN, "fresh and dry bone 0.5-0.7 in the visible (Jacques 2013); emissivity about 0.9"),
    "hide": (0.25, 0.25, 0.97, RN, "mammal coats reflect 0.1 (dark) to 0.4 (light) of visible light (Walsberg 1983, "
                                   "BioScience 33, 88); pelage emissivity 0.95-0.99 (Hammel 1956, J. Mammal. 37, 375)"),
    "charcoal": (0.04, 0.05, 0.95, RN, f"carbon black and charcoal 0.03-0.05 ({_USGS}); charcoal 0.9-0.96 ({_INC})"),
    "ash": (0.40, 0.40, 0.90, RN, f"grey-white wood ash 0.3-0.5 ({_USGS}); powders about 0.9 ({_INC})"),
    "ceramic": (0.30, 0.35, 0.93, RN, f"fired clay and brick 0.20-0.40 ({_OKE}); red brick 0.93-0.96 ({_INC})"),
    "lime": (0.85, 0.85, 0.90, RN, f"quicklime powder is white, 0.8-0.9 ({_USGS}); emissivity about 0.9 ({_INC})"),
    "copper": (0.60, 0.70, 0.05, RN, f"as native_copper (Johnson & Christy 1972); {_METAL}"),
    "tin": (0.65, 0.70, 0.05, RN, f"tin reflects 0.6-0.75 across the visible; {_METAL}"),
    "bronze": (0.50, 0.55, 0.05, RN, f"tin bronze 0.4-0.6, between copper and tin; {_METAL}"),
    "iron": (0.35, 0.35, 0.30, RN, f"bloomery iron is grey: polished iron 0.55-0.6, a rough bloom far less (new_rule "
                                   f"0.35); {_METAL} (new_rule 0.3 for a rough, partly oxidised surface)"),
    "steel": (0.50, 0.55, 0.10, RN, f"steel 0.5-0.6 polished; {_METAL}"),
    "glass": (0.08, 0.08, 0.92, D, f"{_FRESNEL}; soda-lime glass emissivity 0.90-0.95 ({_INC})"),
}

# ---------------------------------------------------------------------------------------------- body tissues
TISSUE_ROWS = {
    "fur": SPECIES_ROWS["hide"],
    "skin": (0.30, 0.35, 0.98, RN, "human skin 0.1 (dark) to 0.45 (light) in the visible (Jacques 2013); emissivity "
                                   "0.98 (Steketee 1973, Phys. Med. Biol. 18, 686)"),
    "flesh": SPECIES_ROWS["meat"],
    "fat": SPECIES_ROWS["fat"],
    "bone": SPECIES_ROWS["bone"],
}

VIS, SW, EMISSIVITY = _rows({**{f"ground:{k}": v for k, v in GROUND_ROWS.items()},
                             **{f"species:{k}": v for k, v in SPECIES_ROWS.items()},
                             **{f"tissue:{k}": v for k, v in TISSUE_ROWS.items()}})

FLAME_EMISSIVITY = _v(0.5, RN, "luminous wood flames: e = 1 - exp(-kappa L), soot kappa about 1 m^-1, so a flame "
                               "0.3-1 m thick has 0.3-0.6 (Drysdale 2011, An Introduction to Fire Dynamics, 3rd ed., "
                               "ch. 2); new_rule: 0.5 for every fire")
VISIBLE_BAND_M = (400e-9, 700e-9)
_NODES = 48   # numerics: Gauss-Legendre nodes of the Planck band integral


def vis(name: str) -> float:
    """Visible reflectance of a ground cover, species or tissue: ``vis("species:flint")``, ``vis("ground:plant")``."""
    return float(VIS[name])


def species_vector(table: dict = VIS, device="cpu", dtype=torch.float32) -> torch.Tensor:
    """[S] values of ``table`` (VIS, SW or EMISSIVITY) in ``materials.SPECIES`` order."""
    return torch.tensor([float(table[f"species:{s}"]) for s in materials.SPECIES], dtype=dtype, device=device)


def mix(shares: torch.Tensor, species=materials.SPECIES, table: dict = VIS) -> torch.Tensor:
    """Reflectance (or emissivity) of a mixture: shares [..., len(species)] (masses or mass fractions) in the order
    of ``species``; the area share is the mass share (module docstring). An all-zero mixture gives 0."""
    vals = torch.tensor([float(table[f"species:{s}"]) for s in species], dtype=torch.float32, device=shares.device)
    s = shares.float().clamp_min(0)
    total = s.sum(-1)
    return torch.where(total > 0, (s * vals).sum(-1) / total.clamp_min(1e-30), torch.zeros_like(total))


def visible_share(t_k) -> torch.Tensor:
    """Share of a black body's radiant exitance in 400-700 nm at temperature t_k (tensor, K): the Planck integral by
    Gauss-Legendre over the band divided by sigma T^4 / pi (the same integral as ``formation.band_fraction``)."""
    t = torch.as_tensor(t_k, dtype=torch.float64)
    x, w = np.polynomial.legendre.leggauss(_NODES)
    lo, hi = VISIBLE_BAND_M
    lam = torch.tensor(0.5 * (hi - lo) * x + 0.5 * (hi + lo), dtype=torch.float64, device=t.device)
    wt = torch.tensor(w, dtype=torch.float64, device=t.device)
    hc, kb, c = K.H_PLANCK, K.K_B, K.C_LIGHT
    tt = t.clamp_min(1.0)[..., None]
    radiance = 2.0 * hc * c ** 2 / lam ** 5 / torch.expm1(hc * c / (lam * kb * tt))
    band = 0.5 * (hi - lo) * (wt * radiance).sum(-1)
    return torch.where(t > 1.0, band / (K.SIGMA * t.clamp_min(1.0) ** 4 / math.pi), torch.zeros_like(t))


def visible_exitance(t_k, emissivity=1.0) -> torch.Tensor:
    """Visible light (W/m^2, float32) radiated by a grey surface at t_k: emissivity sigma T^4 x visible_share(T).
    A 1,100 K flame of emissivity 0.5 radiates about 0.4 W/m^2 of visible light (a full moon gives about 10^-3 at the
    ground, the noon sun about 4 x 10^2) and a 1,300 K one about 8; a 300 K body about 10^-23 (dark)."""
    t = torch.as_tensor(t_k, dtype=torch.float64)
    e = torch.as_tensor(emissivity, dtype=torch.float64, device=t.device)
    return (e * K.SIGMA * t ** 4 * visible_share(t)).float()


def provenance() -> dict[str, tuple[str, str]]:
    """{name: (tag, source)} of every number in this module."""
    out = {}
    for label, table in (("vis", VIS), ("sw", SW), ("emissivity", EMISSIVITY)):
        for name, v in table.items():
            out[f"{label}:{name}"] = (v.tag, v.source)
    out["FLAME_EMISSIVITY"] = (FLAME_EMISSIVITY.tag, FLAME_EMISSIVITY.source)
    out["VISIBLE_BAND_M"] = (R, "the visible / photosynthetically active band 400-700 nm (McCree 1972, Agric. "
                                "Meteorol. 10, 443), as formation.par_fraction")
    out["_NODES"] = (N, "numerics: Gauss-Legendre nodes of the band integral")
    return out
