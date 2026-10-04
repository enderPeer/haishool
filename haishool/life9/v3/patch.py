"""Habitat patches of life9 v3 (PLANET-V3-SPEC sections 1 and 7): true-scale arenas on the real-size planet, their
terrain, light, water, plants and smells.

Arenas. A world holds ``patches`` arenas, A = worlds x patches (arena a = w x patches + p). Each is a square of side L
(``patch_m``, default 2,048 m) on an n x n fine grid (``cells``, default 128: 16 m cells) with a periodic boundary: the
patch is a representative sample of its global cell, standing for homogeneous surroundings (the standard boundary of
physics simulations of an infinite homogeneous medium, not a fence). Positions are metres (x east, y north) in
[0, L); fields are [A, n, n] indexed [a, ix, iy], flat cell = ix n + iy. The patch's global cell is drawn at random by
area over land on the set-up generator (:func:`choose_patch_cells`, no habitability bias).

Terrain. Fine elevation = the global cell's elevation + periodic fractal detail (:func:`fractal_detail`): spectral
synthesis of a self-affine surface, power spectrum P(q) ~ q^-2(H+1) with the short-range Hurst exponent H = 0.8. Its
rms comes from the planet's own relief (:func:`detail_rms`): the rms height difference D between the global cell and
its side neighbours at their spacing l_g, carried down to the patch by a two-regime structure function, D(r) ~ r^0.5
from l_g down to the scale break (5 km) and D(r) ~ r^0.8 below it, divided by sqrt(2). Gaussian noise is not an eroded
landscape: every local minimum of it is a closed pit. :func:`condition_terrain` therefore fills (sediment infill) every
closed depression shallower than ``lake_keep_rms`` x the detail rms, so only the deep ones stay as lake basins (the
constant is calibrated on Earth's lake share, 3.7 % of non-glaciated land). Slope, aspect and the surface normal come
from centred differences. :func:`sagitta` and :func:`line_of_sight` use the real planet radius.

Light. The sun direction per bout comes from the global sun angle (declination, the subsolar point at the bout's time
with version 2's phase: noon at longitude 0 at t = 0; a locked planet's substellar point is fixed), sampled
``light_samples`` times within each bout (no daily averaging: night is dark). The climate's daily-mean flat-ground
shortwave is turned into the instantaneous flux by the flat ground's mean incidence over one whole solar day
(:func:`flat_mean_incidence`), so the bouts carry the physical instantaneous light and the mean over many days is the
climate's, whatever the length of the solar day. The light is split into beam and diffuse (Erbs, Klein & Duffie's daily
diffuse fraction of the clearness index): the beam falls per horizontal m^2 as max(0, s_z - z_x s_x - z_y s_y), the
diffuse by the sky view factor (1 + cos slope) / 2 per sloped m^2, normalised over the patch so that it is conserved.

Water (:func:`step`, the hydrology part). Per fine cell a soil bucket (the climate's Manabe bucket, 150 kg/m^2), snow
(degree-day melt, glacier discharge above 1,000 kg/m^2) and open water (``pond``). Precipitation is the global cell's,
as snow below 273.15 K at the fine cell's temperature (the lapse rate g/c_p applied to its height above the cell's
mean). Water above the bucket runs off downhill to where it collects, on the depression hierarchy of the terrain
(fill-spill-merge; Barnes, Callaghan & Wickert 2020, Earth Surf. Dynam. 8, 431): every closed depression keeps its
own water and level until it is full to its sill, then it spills into its neighbour across the sill; two neighbours
that are both full share one level above their merge sill. The patch drains at its exits: the sea (fine cells below the
sea level) and its lowest fine cell, a sink at its own level (the patch's link to the regional drainage the global
layer books as runoff). Ponded water seeps to the deep drainage (``seep_kg_m2_day``). Ponds evaporate as open water
(none when frozen); the soil by the climate's bulk formula with Manabe's beta, none under snow. Ponds wet the soil
under them. Everything that leaves (exits, seepage, glacier) is booked as ``w_out``.

Plants. Per fine cell soft tissue, wood, litter, mineral nitrogen and a seed bank, with version 2's production
equations (``planet.biosphere``: GPP = LUE fAPAR PAR f_T f_W f_CO2 f_N, maintenance per tissue N acclimated to the
running-mean temperature, the payment order, nitrogen-limited growth, resprouting, turnover, decomposition) and
nitrogen fixation by the plants' symbionts (``bnf_share`` of the N the growth asks for, where the mineral pool limits
it), with the fine cell's light, temperature and soil water, except fAPAR, which follows the leaf area (Beer's law
on LAI = SLA x leaf share x dry soft tissue, Monsi & Saeki), and standing water, which drowns production and
germination. There is no minimum plant stock anywhere: plants spread only by seed. A share
``seed_share`` of growth becomes seeds; ``seed_export`` of them land in the 8 neighbouring cells (weights 1 /
distance), the rest at home. Seeds germinate (an e-folding of ``germination_days``) only where a seedling's net
production would be positive; elsewhere they wait in the seed bank and die (half-life ``seed_half_life_yr``) to litter.
:func:`sow` sows seeds sparsely and uniformly for the spin-up; plants end up where they can grow.

Smells (:class:`Volatiles`). Chemistry-based channels: plant volatiles (from soft tissue), sulfur volatiles of rotting
animal matter (carcass flesh, faeces), smoke (CO from fires) and CO2 from point sources (bodies' metabolism and fires),
as near-surface concentrations (kg/m^3) in a surface layer: dc/dt = K lap c - c / tau + E / h, solved exactly per step
on the periodic grid in Fourier space (:func:`volatile_step`).

Neighbours. :func:`neighbours` is a spatial hash: bodies are bucketed by fine cell (one sort); each looks in its 3 x 3
cells, nearest cells first, and keeps the K nearest of what it scans; past its scan budget it samples a cell uniformly,
so no direction or slot is favoured.

Ledgers (float64 [A], whole-patch totals): carbon (kg C: plant + wood + litter + seed - carbon from the air + carbon
taken by bodies - carbon added), water (kg: soil + pond + snow - precipitation + evaporation + outflow + taken - given)
and nitrogen (kg N: stores - fixed + removed by bodies - returned by bodies). Each is the error against the baseline of
:func:`init_state` (or :func:`reset_ledgers`). The patch's net carbon exchange with the air (including what
:func:`sow` took) is in each step's diagnostics (``exchange_mol`` [A]); :func:`gas_delta` turns it into the [W, 4]
column change for ``climate.add_gas``. Patch precipitation and evaporation are a sample of the global cell's (already
inside the climate's water ledger); what bodies take and give is what the world books against the global ledgers.

Every constant carries a provenance entry in ``PROVENANCE``. States are values: :func:`step` returns a new state;
the take/give helpers rebind the state's fields to new tensors (an earlier ``state_dict`` never changes). The set-up
(:func:`drainage`, :func:`condition_terrain`) walks the depression tree in Python over depressions and sill edges (a
few hundred per arena), never over cells or bodies; the daily routing is vectorised per tree level.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, replace
from functools import lru_cache

import torch

from haishool.life9.planet import biosphere as bs
from haishool.life9.planet import climate as cl
from haishool.life9.planet import constants as K
from haishool.life9.planet import materials

from . import optics

DAY_S = K.DAY_S
DAYS_PER_YEAR = K.YEAR_S / K.DAY_S
MJ_PER_W_DAY = K.DAY_S / 1e6
M_C = K.element_mass("C")
M_H2O = K.molar_mass("H2O")
M_CO2 = K.molar_mass("CO2")
RHO_W = K.RHO_WATER
T_FREEZE = 273.15
SQRT2 = math.sqrt(2.0)
ERBS_WS_RAD = 1.4208          # 81.4 degrees: the sunset hour angle that splits Erbs et al.'s two daily correlations
#: the 8 neighbour offsets (dx, dy) on the fine grid: 4 sides, then 4 diagonals
OFFSETS8 = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
#: the 3 x 3 block of the spatial hash, own cell first
BLOCK9 = ((0, 0),) + OFFSETS8


@dataclass(frozen=True)
class PatchRules:
    """Modelling choices and reference values of the patches (sources in ``PROVENANCE``)."""
    patch_m: float = 2048.0
    cells: int = 128
    hurst: float = 0.8
    hurst_large: float = 0.5
    relief_break_m: float = 5000.0
    lake_keep_rms: float = 0.4
    condition_rounds: int = 16
    fill_eps_m: float = 1e-9
    seep_kg_m2_day: float = 2.0
    light_samples: int = 8
    seed_share: float = 0.1
    seed_export: float = 0.5
    germination_days: float = 10.0
    seed_half_life_yr: float = 1.0
    sow_share: float = 1.0 / 64.0
    sow_kg_c_m2: float = 1e-3
    sla_m2_kg: float = 20.0
    leaf_share: float = 0.5
    light_extinction: float = 0.5
    flood_depth_m: float = 0.1
    germination_max_pond_kg_m2: float = 10.0
    pond_mask_kg_m2: float = 10.0
    litter_cover_m2_kg: float = 4.0
    eddy_diffusivity_m2_s: float = 10.0
    smell_layer_m: float = 10.0
    ventilation_s: float = 3600.0
    plant_volatile_life_s: float = 1.4 * 3600.0
    carcass_volatile_life_s: float = 0.6 * 3600.0
    plant_emission_kg_kg_s: float = 10e-9 / (1e-3 * 3600.0)
    flesh_decay_day: float = 0.05
    carcass_volatile_share: float = 0.01
    smoke_kg_per_kg_fuel: float = 0.1
    co2_kg_per_kg_fuel: float = 1.58
    oxycaloric_j_mol_o2: float = 4.5e5
    respiratory_quotient: float = 0.85
    sight_tolerance_m: float = 1e-3
    sight_samples_per_cell: float = 3.0
    sight_chunk: int = 1 << 22
    bnf_share: float = 0.5
    bio: bs.BioRules = field(default_factory=bs.BioRules)
    climate: cl.ClimateRules = field(default_factory=cl.ClimateRules)


PROVENANCE = {
    "patch_m": ("new_rule", "patch side L = 2,048 m (PLANET-V3-SPEC 1, config patch_m)"),
    "cells": ("new_rule", "fine grid n = 128 per side: 16 m cells (PLANET-V3-SPEC 1)"),
    "hurst": ("reference+new_rule", "self-affine topography at short range: the variograms of topographic surfaces "
              "show a low fractal dimension D = 3 - H of about 2.2 at short lags, separated by scale breaks from "
              "rougher behaviour at longer ones (Mark & Aronson 1984, Math. Geol. 16, 671); new_rule: H = 0.8 below "
              "the break, the patch's own spectrum q^-2(H+1)"),
    "hurst_large": ("reference", "above the break topography is close to Brownian: profile spectra S(k) ~ k^-2 over "
                    "tens to thousands of km (Newman & Turcotte 1990, Geophys. J. Int. 100, 433; Turcotte 1997, "
                    "Fractals and Chaos in Geology and Geophysics, 2nd ed., ch. 7), H = (2 - 1) / 2 = 0.5"),
    "relief_break_m": ("reference+new_rule", "the scale breaks of Mark & Aronson (1984) lie at characteristic "
                       "horizontal scales of hundreds of metres to kilometres (ridge-valley spacing); new_rule: 5 km, "
                       "so a 2 km patch lies in the H = 0.8 range"),
    "detail_rms": ("new_rule", "rms of the fine detail = D(l_g) (r_b / l_g)^H_large (L / r_b)^H / sqrt(2), r_b clamped "
                   "to [L, l_g]: D(l_g) the rms height difference between the global cell and its 4 side neighbours "
                   "at their mean centre distance l_g (the planet's own relief), carried down by the two-regime "
                   "structure function; a stationary field has D(r)^2 -> 2 sigma^2 at large lags"),
    "lake_keep_rms": ("reference+new_rule", "Gaussian detail is not an eroded landscape: each of its local minima is a "
                      "closed pit (hundreds per patch, 27-29 % of the cells under their spill levels). Closed "
                      "depressions shallower than this x the detail rms are filled (sediment infill of shallow "
                      "basins; erosion and sediment supply scale with the relief, so the rule is scale-free); "
                      "calibration: on the synthetic detail this leaves about 3-4 % of the land in lake basins, "
                      "Earth's lakes cover 3.7 % of the non-glaciated land (Verpoorter et al. 2014, Geophys. Res. "
                      "Lett. 41, 6396)"),
    "condition_rounds": ("new_rule", "numerics: the depression conditioning repeats until no shallow depression is "
                         "left, at most 16 rounds"),
    "fill_eps_m": ("new_rule", "numerics: filled depressions and flats get this rise per cell toward their outlet "
                   "(Planchon & Darboux 2002, Catena 46, 159), so they drain by strict descent"),
    "seep_kg_m2_day": ("reference+new_rule", "lake beds exchange water with the ground at millimetres to centimetres a "
                       "day (published lake-groundwater exchange: median 6.8 L m^-2 d^-1, Rosenberry, Lewandowski, "
                       "Meinikmann & Nuetzmann 2015, Hydrol. Process. 29, 2895); new_rule: ponded water seeps down at "
                       "2 kg/m^2 (2 mm) a day to the deep drainage, booked as outflow"),
    "light_samples": ("new_rule", "numerics: sun directions sampled per bout (the bout's mean incidence)"),
    "bnf_share": ("reference+new_rule", "biological N fixation: symbionts fix up to about half of the nitrogen the "
                  "plants' growth asks for where the mineral pool limits it (legume stands take 40-80 % of their N "
                  "from fixation; global terrestrial fixation of the order of 0.1-0.2 Pg N/yr against about 60 Pg C/yr "
                  "of NPP, Cleveland et al. 1999, Global Biogeochem. Cycles 13, 623; Vitousek et al. 2013, Phil. "
                  "Trans. R. Soc. B 368, 20130119; from memory: verify): n_fixed = bnf_share x (N demand of the day's "
                  "growth) x (1 - f_N), so it follows production and N limitation, with no reference to the starting "
                  "stock (version 2's rule refilled every cell toward its starting N, a target acting as ecology). "
                  "The respiratory cost of fixation (about 6-7 g C per g N) is not charged"),
    "seed_share": ("reference+new_rule", "reproductive allocation of plants is commonly 5-20 % of production (Bazzaz, "
                   "Ackerly & Reekie 2000, in Fenner ed., Seeds: the Ecology of Regeneration in Plant Communities, "
                   "ch. 1; Obeso 2002, New Phytol. 155, 321); new_rule: 10 % of growth becomes seeds, which carry the "
                   "soft tissue's C:N"),
    "seed_export": ("reference+new_rule", "most seeds fall within metres to tens of metres of the parent (Willson "
                    "1993, Vegetatio 107/108, 261; Nathan & Muller-Landau 2000, Trends Ecol. Evol. 15, 278); "
                    "new_rule: half of a 16 m cell's seeds land in its 8 neighbours, weighted by 1 / distance "
                    "(sides 1, diagonals 1 / sqrt 2)"),
    "germination_days": ("reference+new_rule", "seeds germinate within days to weeks once conditions allow (Baskin & "
                         "Baskin 2014, Seeds, 2nd ed.); new_rule: an e-folding of 10 days where a seedling's net "
                         "production is positive"),
    "seed_half_life_yr": ("reference+new_rule", "soil seed banks: transient seeds live under a year, short-term "
                          "persistent ones 1-5 years (Thompson, Bakker & Bekker 1997, The Soil Seed Banks of North "
                          "West Europe); new_rule: a 1-year half-life, dead seeds become litter"),
    "sow_share": ("new_rule", "spin-up sowing: 1 / 64 of the fine cells, drawn uniformly at random (sparse, no "
                  "habitability bias); the starting condition of the plants' spread, not a target"),
    "sow_kg_c_m2": ("new_rule", "1 g C/m^2 of seed in a sown cell (seed rain is of the order of grams of dry matter "
                    "per m^2 per year), its carbon from the air, its nitrogen from the mineral pool"),
    "sla_m2_kg": ("reference", "specific leaf area 20 m^2 per kg dry leaf: leaf mass per area of herbs about 50 g/m^2 "
                  "(Wright et al. 2004, Nature 428, 821, the global leaf economics spectrum, LMA 14-1,500 g/m^2; "
                  "Poorter et al. 2009, New Phytol. 182, 565)"),
    "leaf_share": ("reference", "leaves are about half of a herbaceous plant's soft (non-woody) dry mass (leaf mass "
                   "fraction 0.4-0.6; Poorter et al. 2012, New Phytol. 193, 30)"),
    "light_extinction": ("reference", "canopy extinction coefficient k = 0.5 of a spherical leaf-angle distribution "
                         "(Monsi & Saeki 1953, Jpn. J. Bot. 14, 22; Campbell & Norman 1998, An Introduction to "
                         "Environmental Biophysics, ch. 15): fAPAR = 1 - exp(-k LAI), LAI = SLA x leaf share x "
                         "plant_c / 0.47"),
    "flood_depth_m": ("reference+new_rule", "submerged leaves of terrestrial plants photosynthesise at a few per cent "
                      "of their rate in air (CO2 diffuses 10^4 times slower in water; Mommer & Visser 2005, Ann. "
                      "Bot. 96, 581); new_rule: production x exp(-pond depth / 0.1 m), the emergent share of a low "
                      "canopy"),
    "germination_max_pond_kg_m2": ("reference+new_rule", "seeds of terrestrial plants do not germinate under standing "
                                   "water (anoxia; Baskin & Baskin 2014); new_rule: none under more than 1 cm"),
    "pond_mask_kg_m2": ("new_rule", "10 kg/m^2 (1 cm) of open water covers the cell fully for its reflectance (as the "
                        "climate's snow_mask_kg_m2)"),
    "litter_cover_m2_kg": ("reference", "ground covered by plant residue: 1 - exp(-A_m M) with A_m of 1.5-6 m^2 per kg "
                           "dry residue (Gregory 1982, Trans. ASAE 25, 1333); 4 m^2/kg on the litter's dry mass "
                           "(litter carbon / 0.47)"),
    "eddy_diffusivity_m2_s": ("reference+new_rule", "horizontal eddy diffusivity of the surface layer, 1-100 m^2/s "
                              "(Seinfeld & Pandis 2016, Atmospheric Chemistry and Physics, 3rd ed., ch. 18); "
                              "new_rule: 10 m^2/s (the mean wind's transport is not modelled)"),
    "smell_layer_m": ("new_rule", "a smell is the concentration in the lowest 10 m of air (the surface layer is about "
                      "a tenth of a ~1 km boundary layer, Stull 1988, An Introduction to Boundary Layer Meteorology)"),
    "ventilation_s": ("reference+new_rule", "air leaves the surface layer to the boundary layer on the convective "
                      "time scale, 10-30 min by day and longer at night (Stull 1988); new_rule: 1 h for every "
                      "volatile"),
    "plant_volatile_life_s": ("reference", "isoprene's lifetime against OH (2e6 cm^-3) is 1.4 h (Atkinson & Arey "
                              "2003, Chem. Rev. 103, 4605)"),
    "carcass_volatile_life_s": ("reference", "dimethyl disulfide, a main volatile of rotting animal matter, reacts "
                                "with OH at about 2.3e-10 cm^3/s (Atkinson et al. 2004, Atmos. Chem. Phys. 4, 1461): "
                                "0.6 h at 2e6 cm^-3"),
    "plant_emission_kg_kg_s": ("reference+new_rule", "leaf volatile emission factors span 0.1-70 ug per g dry leaf "
                               "per hour (Guenther et al. 1995, J. Geophys. Res. 100, 8873); new_rule: 10 ug/g/h from "
                               "the dry soft tissue (plant_c / 0.47)"),
    "flesh_decay_day": ("reference+new_rule", "carcasses lose mass at a few to 10 % a day in active decay at 20-25 C "
                        "(Carter, Yellowlees & Tibbett 2007, Naturwissenschaften 94, 12); new_rule: 5 %/day for any "
                        "rotting animal matter (flesh, faeces)"),
    "carcass_volatile_share": ("new_rule", "1 % of the decaying animal matter leaves as volatiles (amines, sulfides)"),
    "smoke_kg_per_kg_fuel": ("reference", "wood burning emits 60-110 g CO per kg of fuel (Andreae & Merlet 2001, "
                             "Global Biogeochem. Cycles 15, 955): smoke as 0.1 kg per kg burnt (CO; it does not decay "
                             "chemically within hours)"),
    "co2_kg_per_kg_fuel": ("reference", "biomass burning emits 1.55-1.65 kg CO2 per kg of dry fuel (Andreae & Merlet "
                           "2001)"),
    "oxycaloric_j_mol_o2": ("reference", "aerobic metabolism releases about 450 kJ per mol of O2 (20.1 kJ per litre O2 "
                            "at STP for mixed fuels; Schmidt-Nielsen 1997, Animal Physiology, 5th ed., ch. 5)"),
    "respiratory_quotient": ("reference", "CO2 out per O2 in: 0.7 (fat) to 1.0 (carbohydrate), 0.85 for mixed fuel "
                             "(Schmidt-Nielsen 1997)"),
    "sight_tolerance_m": ("new_rule", "numerics: a sight line may graze the terrain by 1 mm"),
    "sight_samples_per_cell": ("new_rule", "numerics: terrain samples along a sight line, 3 per fine cell of its "
                               "length (at least 2), so the cost follows the distance"),
    "sight_chunk": ("new_rule", "numerics: at most 2^22 sight-line samples are held at once (memory bound)"),
    "bio": ("reference+new_rule", "version 2's biosphere rules and references (planet.biosphere.PROVENANCE); its "
            "seed rain and its fapar_kg_c are not used"),
    "climate": ("reference+new_rule", "version 2's climate rules (bucket, Manabe beta, bulk transfer, degree-day melt, "
                "glacier, snow mask, ice ramp; planet.climate.PROVENANCE)"),
    "fine_temperature": ("derived", "T = T_cell - (g / c_p)(z - z_cell): the climate's dry adiabat applied per fine "
                         "cell (PLANET-V3-SPEC 1)"),
    "light": ("derived", "flat-ground instantaneous shortwave = the climate's daily mean x cos zenith / the flat "
              "ground's mean cos zenith over a whole solar day; beam per horizontal m^2 x max(0, s_z - z_x s_x - "
              "z_y s_y) / s_z, diffuse x v / mean(v), v = (1 + cos slope) / (2 cos slope) (isotropic sky view factor "
              "per horizontal m^2, normalised so the patch mean is kept)"),
    "diffuse_share": ("reference", "daily diffuse fraction of the clearness index K_T = surface_sw / insolation (Erbs, "
                      "Klein & Duffie 1982, Solar Energy 28, 293): sunset hour angle <= 81.4 deg: 1 - 0.2727 K + "
                      "2.4495 K^2 - 11.9514 K^3 + 9.3879 K^4 (K < 0.715), else 0.143; above 81.4 deg (and a locked "
                      "planet's permanent day): 1 + 0.2832 K - 2.5557 K^2 + 0.8448 K^3 (K < 0.722), else 0.175"),
    "sun_time": ("derived", "version 2's phase (world.sun): the subsolar longitude is 0 at t = 0 and moves west by "
                 "2 pi per solar day (the spec's day_length_s; infinite or locked: fixed); declination from "
                 "climate.declination"),
    "flat_mean": ("derived", "the flat ground's mean max(0, cos zenith) over one whole solar day: the midpoint rule "
                  "at the bouts' sample density, at least 8 samples per solar day (the sim days' own samples cover the "
                  "same phases when the solar day is commensurate with 24 h), the analytic diurnal mean m1 / pi "
                  "(Berger 1978, as planet.climate) for days longer than 4,096 samples"),
    "evaporation": ("derived", "the climate's bulk formula at the fine cell's temperature: min(1, C_E U dt / H_w) x "
                    "(v_sat(T) - v), v the global cell's column; ponds evaporate freely (none when frozen, the "
                    "climate's ice ramp), soil with Manabe's beta, none under snow"),
    "seedling_test": ("derived", "a seedling's net production is positive where the potential GPP per kg of soft "
                      "tissue at small cover (potential f_N f_flood k SLA leaf share / 0.47) exceeds its maintenance "
                      "per kg (r_N q / CN_leaf)"),
    "neighbour_scan": ("new_rule", "numerics: the spatial hash scans the 3 x 3 cells nearest first (by the body's "
                       "distance to each cell) up to a budget of 4 K candidates, sampling the last cell it reaches "
                       "uniformly (stratified), and keeps the K nearest (K = 32, PLANET-V3-SPEC 4); ``count`` reports "
                       "how many were there"),
    "T_FREEZE": ("reference", "273.15 K, the melting point of ice at 1 atm (as planet.climate)"),
    "RHO_W": ("reference", "constants.RHO_WATER, 1000 kg/m^3: pond depth = kg/m^2 / 1000"),
    "ERBS_WS_RAD": ("reference", "81.4 deg, Erbs, Klein & Duffie 1982"),
    "flood_check": ("new_rule", "numerics: the depression filling checks its fixed point every 16 sweeps"),
    "hierarchy": ("reference", "depression hierarchy: watersheds of the pits by steepest descent, merged in the order "
                  "of their sill heights (lowest first; a depression that meets the exits drains there), fill-spill-"
                  "merge routing (Barnes, Callaghan & Wickert 2020, Earth Surf. Dynam. 8, 431; Barnes, Callaghan & "
                  "Wickert 2021, Earth Surf. Dynam. 9, 105)"),
}
for _f in fields(PatchRules):
    assert _f.name in PROVENANCE, _f.name


@lru_cache(maxsize=8)
def _bio_params(rules: bs.BioRules) -> dict:
    """version 2's derived biosphere numbers (r_n, leaf_wood) for these rules."""
    return bs.make_params({"par_fraction": None, "land": None, "W": 0}, rules)


def leaf_area_per_kg_c(rules: PatchRules | None = None) -> float:
    """k x LAI per kg C/m^2 of soft tissue (m^2 per kg C): extinction x SLA x leaf share / carbon fraction."""
    r = rules or PatchRules()
    return r.light_extinction * r.sla_m2_kg * r.leaf_share / r.bio.plant_carbon_fraction


def fapar(plant_c, rules: PatchRules | None = None):
    """Absorbed share of PAR (and the canopy's ground cover) of soft tissue plant_c (kg C/m^2): 1 - exp(-k LAI)
    (``PROVENANCE['light_extinction']``)."""
    return 1 - torch.exp(-leaf_area_per_kg_c(rules) * plant_c.clamp_min(0))


# ============================================================================================ geometry
@dataclass
class PatchGeometry:
    """Terrain and set-up of A arenas (see the module docstring). ``hydro`` holds the drainage tables."""
    L: float
    n: int
    W: int
    patches: int
    world: torch.Tensor          # [A] long
    cell: torch.Tensor           # [A] long (global cell)
    radius_m: torch.Tensor       # [A] float64, planet radius
    east: torch.Tensor           # [A, 3] the global cell's local basis (unit vectors)
    north: torch.Tensor
    up: torch.Tensor
    base_m: torch.Tensor         # [A] the global cell's elevation
    sea_level_m: torch.Tensor    # [A]
    detail_rms_m: torch.Tensor   # [A]
    bare_vis: torch.Tensor       # [A] visible reflectance of the bare ground
    elev: torch.Tensor           # [A, n, n] m
    dzdx: torch.Tensor           # [A, n, n]
    dzdy: torch.Tensor
    slope: torch.Tensor          # [A, n, n] rad
    aspect: torch.Tensor         # [A, n, n] rad, compass direction the slope faces (downhill), 0 north, pi/2 east
    normal: torch.Tensor         # [A, n, n, 3] (east, north, up)
    sea: torch.Tensor            # [A, n, n] bool, fine cells below the sea level
    hydro: dict

    @property
    def A(self) -> int:
        return int(self.elev.shape[0])

    @property
    def dx(self) -> float:
        return self.L / self.n

    @property
    def cell_m2(self) -> float:
        return self.dx * self.dx

    @property
    def device(self):
        return self.elev.device

    def state_dict(self) -> dict:
        out = {f.name: getattr(self, f.name) for f in fields(self)}
        out = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in out.items()}
        out["hydro"] = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in self.hydro.items()}
        return out

    @classmethod
    def from_state(cls, d: dict) -> "PatchGeometry":
        return cls(**d)


def choose_patch_cells(globe, land, area, gen, patches: int) -> torch.Tensor:
    """Global cells [W, patches] (long) of each world's patches: drawn at random by area over land (with
    replacement, on ``gen``), with no habitability bias. A world without land draws over all its cells (its patches
    are sea: a finding, reported by ``land[w, cell]``). ``area`` is [C] or [W, C] (steradians or m^2)."""
    land = torch.as_tensor(land).bool()
    W, C = land.shape
    area = torch.as_tensor(area, dtype=torch.float64, device=land.device).expand(W, C)
    weight = torch.where(land, area, torch.zeros_like(area))
    empty = weight.sum(-1, keepdim=True) <= 0
    weight = torch.where(empty, area, weight)
    gdev = land.device if gen is None else gen.device
    return torch.multinomial(weight.to(gdev), int(patches), replacement=True, generator=gen).to(land.device)


def relief_structure(globe, elevation_m, radius_m):
    """(D, l) [W, C] float64: the rms elevation difference D (m) between each global cell and its 4 side neighbours
    and their mean centre distance l (m) on a planet of radius_m ([W] or a number)."""
    e = torch.as_tensor(elevation_m).double()
    W = e.shape[0]
    nb = globe.nbr4
    d2 = (e[:, :, None] - e[:, nb]) ** 2
    ang = torch.acos((globe.centers[:, None].double() * globe.centers[nb].double()).sum(-1).clamp(-1, 1)).mean(-1)
    radius = torch.as_tensor(radius_m, dtype=torch.float64, device=e.device).reshape(-1, 1).expand(W, 1)
    return d2.mean(-1).sqrt(), ang[None] * radius


def structure_scale(lag_m, patch_m: float, rules: PatchRules | None = None):
    """D(L) / D(lag) of the two-regime structure function (``PROVENANCE['detail_rms']``): (r_b / lag)^H_large x
    (L / r_b)^H with the break r_b clamped to [L, lag]; (L / lag)^H when the lag is shorter than the patch."""
    r = rules or PatchRules()
    lag = torch.as_tensor(lag_m, dtype=torch.float64).clamp_min(1e-9)
    L = torch.full_like(lag, float(patch_m))
    hi = torch.maximum(lag, L)
    rb = torch.minimum(torch.maximum(torch.full_like(lag, r.relief_break_m), L), hi)
    return (rb / hi) ** r.hurst_large * (L / rb) ** r.hurst * (L / torch.minimum(lag, L)) ** r.hurst


def detail_rms(globe, elevation_m, radius_m, rules: PatchRules | None = None) -> torch.Tensor:
    """rms (m) of the fine detail of a patch of side ``patch_m`` at each global cell [W, C]
    (``PROVENANCE['detail_rms']``)."""
    r = rules or PatchRules()
    D, lag = relief_structure(globe, elevation_m, radius_m)
    return D * structure_scale(lag, r.patch_m, r) / SQRT2


def fractal_detail(A: int, n: int, gen, hurst: float, device=None) -> torch.Tensor:
    """A periodic self-affine surface [A, n, n] (float32) of zero mean and unit standard deviation per arena:
    Gaussian white noise on ``gen`` filtered in Fourier space by |q|^-(H+1) (power ~ q^-2(H+1)), the mean mode
    removed (spectral synthesis; Saupe 1988, in Peitgen & Saupe eds., The Science of Fractal Images, ch. 2)."""
    device = gen.device if device is None else torch.device(device)
    noise = torch.randn(A, n, n, generator=gen, device=gen.device, dtype=torch.float32).to(device).double()
    kx = torch.fft.fftfreq(n, d=1.0 / n, dtype=torch.float64, device=device)
    ky = torch.fft.rfftfreq(n, d=1.0 / n, dtype=torch.float64, device=device)
    q = torch.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)
    amp = torch.where(q > 0, q.clamp_min(1e-12) ** (-(hurst + 1.0)), torch.zeros_like(q))
    z = torch.fft.irfft2(torch.fft.rfft2(noise) * amp, s=(n, n))
    z = z - z.mean((1, 2), keepdim=True)
    std = z.pow(2).mean((1, 2), keepdim=True).sqrt()
    return (z / std.clamp_min(1e-30)).float()


def slope_aspect(elev, dx: float):
    """Centred periodic differences of elev [A, n, n]: (dz/dx, dz/dy, slope (rad), aspect (rad, the compass direction
    of steepest descent, 0 north, pi/2 east), unit normal [A, n, n, 3] in (east, north, up))."""
    z = elev.double()
    dzdx = (torch.roll(z, -1, 1) - torch.roll(z, 1, 1)) / (2 * dx)
    dzdy = (torch.roll(z, -1, 2) - torch.roll(z, 1, 2)) / (2 * dx)
    g = torch.sqrt(dzdx ** 2 + dzdy ** 2)
    slope = torch.atan(g)
    aspect = torch.remainder(torch.atan2(-dzdx, -dzdy), 2 * math.pi)
    normal = torch.stack((-dzdx, -dzdy, torch.ones_like(z)), -1) / torch.sqrt(1 + g * g)[..., None]
    return dzdx.float(), dzdy.float(), slope.float(), aspect.float(), normal.float()


def make_geometry(globe, elevation_m, sea_level_m, land, radius_m, gen, patches: int = 4,
                  rules: PatchRules | None = None, cells=None, ground_vis=None) -> PatchGeometry:
    """Patches of W worlds on the global grid: their cells (``choose_patch_cells`` unless ``cells`` [W, patches] is
    given), fine elevation = the cell's elevation_m [W, C] + fractal detail (``detail_rms``) conditioned by
    :func:`condition_terrain` (depressions shallower than ``lake_keep_rms`` x the rms filled), slopes, the sea below
    sea_level_m [W] and the drainage tables. ``radius_m`` is the planet radius ([W] or a number); ``ground_vis``
    [W, C] the bare ground's visible reflectance (``optics.mix`` of the cell's deposits; the optics table's bare
    soil when not given). Every draw is on ``gen``."""
    r = rules or PatchRules()
    elevation_m = torch.as_tensor(elevation_m)
    W = elevation_m.shape[0]
    dev = elevation_m.device
    if cells is None:
        cells = choose_patch_cells(globe, land, globe.area64, gen, patches)
    cells = torch.as_tensor(cells, device=dev).long().reshape(W, -1)
    patches = cells.shape[1]
    world = torch.arange(W, device=dev).repeat_interleave(patches)
    cell = cells.reshape(-1)
    radius = torch.as_tensor(radius_m, dtype=torch.float64, device=dev).reshape(-1).expand(W)
    rms = detail_rms(globe, elevation_m, radius, r)[world, cell].float()
    base = elevation_m[world, cell].float()
    detail = fractal_detail(world.shape[0], r.cells, gen, r.hurst, dev).double() * rms.double()[:, None, None]
    sea = torch.as_tensor(sea_level_m, dtype=torch.float32, device=dev).reshape(-1).expand(W)[world]
    vis = None if ground_vis is None else torch.as_tensor(ground_vis, dtype=torch.float32, device=dev)[world, cell]
    return geometry_from_elevation(base.double()[:, None, None] + detail, r.patch_m, sea, radius[world], world=world,
                                   cell=cell, W=W, patches=patches,
                                   basis=(globe.east[cell], globe.north[cell], globe.centers[cell]),
                                   base_m=base, detail_rms_m=rms, bare_vis=vis,
                                   condition_depth_m=r.lake_keep_rms * rms.double(), rules=r)


def geometry_from_elevation(elev, L: float, sea_level_m, radius_m, *, world=None, cell=None, W=None, patches=None,
                            basis=None, base_m=None, detail_rms_m=None, bare_vis=None, condition_depth_m=None,
                            rules: PatchRules | None = None) -> PatchGeometry:
    """A geometry from a given fine elevation [A, n, n] (m) of side L: slopes, the sea and the drainage tables.
    ``condition_depth_m`` ([A] or a number) fills every closed depression shallower than it first
    (:func:`condition_terrain`; none when not given). Defaults for a stand-alone arena: world = arena, one patch per
    world, the basis of (lat 0, lon 0) (east = +y, north = +z, up = +x), base = the mean elevation, bare ground = the
    optics table's soil."""
    r = rules or PatchRules()
    z = torch.as_tensor(elev).double()
    A, n = z.shape[0], z.shape[1]
    if z.shape != (A, n, n):
        raise ValueError("elev must be [A, n, n]")
    if n < 3:
        raise ValueError("the fine grid needs at least 3 cells per side")
    dev = z.device
    f32 = torch.float32
    world = torch.arange(A, device=dev) if world is None else torch.as_tensor(world, device=dev).long()
    cell = torch.zeros(A, dtype=torch.long, device=dev) if cell is None else torch.as_tensor(cell, device=dev).long()
    W = int(world.max()) + 1 if W is None else int(W)
    patches = 1 if patches is None else int(patches)
    if basis is None:
        e = torch.tensor([0.0, 1.0, 0.0], device=dev).expand(A, 3)
        nn_ = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(A, 3)
        u = torch.tensor([1.0, 0.0, 0.0], device=dev).expand(A, 3)
        basis = (e, nn_, u)
    east, north, up = (torch.as_tensor(b, dtype=f32, device=dev).reshape(A, 3).clone() for b in basis)
    sea_level = torch.as_tensor(sea_level_m, dtype=f32, device=dev).reshape(-1).expand(A).clone()
    radius = torch.as_tensor(radius_m, dtype=torch.float64, device=dev).reshape(-1).expand(A).clone()
    sea = z < sea_level.double()[:, None, None]
    rounds = 0
    if condition_depth_m is not None:
        z, rounds = condition_terrain(z, sea, condition_depth_m, r)
    elev32 = z.float()
    base = elev32.mean((1, 2)) if base_m is None else torch.as_tensor(base_m, dtype=f32, device=dev).reshape(A)
    rms = (elev32 - elev32.mean((1, 2), keepdim=True)).pow(2).mean((1, 2)).sqrt() if detail_rms_m is None else \
        torch.as_tensor(detail_rms_m, dtype=f32, device=dev).reshape(A)
    vis = torch.full((A,), optics.vis("ground:soil"), device=dev) if bare_vis is None else \
        torch.as_tensor(bare_vis, dtype=f32, device=dev).reshape(A)
    dx = L / n
    dzdx, dzdy, slope, aspect, normal = slope_aspect(z, dx)
    hydro = drainage(z, sea, r)
    hydro["condition_rounds"] = rounds
    return PatchGeometry(L=float(L), n=int(n), W=W, patches=patches, world=world, cell=cell, radius_m=radius,
                         east=east, north=north, up=up, base_m=base.float(), sea_level_m=sea_level,
                         detail_rms_m=rms.float(), bare_vis=vis, elev=elev32, dzdx=dzdx, dzdy=dzdy, slope=slope,
                         aspect=aspect, normal=normal, sea=sea, hydro=hydro)


# ============================================================================================ drainage
def _neighbour_index(n: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """[n*n, 8] flat indices of the 8 periodic neighbours (``OFFSETS8`` order) and their distances in cells [8]."""
    ix, iy = torch.meshgrid(torch.arange(n, device=device), torch.arange(n, device=device), indexing="ij")
    off = torch.tensor(OFFSETS8, device=device)
    jx = (ix.reshape(-1, 1) + off[:, 0]) % n
    jy = (iy.reshape(-1, 1) + off[:, 1]) % n
    dist = torch.tensor([1.0, 1.0, 1.0, 1.0, SQRT2, SQRT2, SQRT2, SQRT2], dtype=torch.float64, device=device)
    return jx * n + jy, dist


def _flood(z, fixed, level, eps: float = 0.0, check_every: int = 16):
    """Depression filling (Planchon & Darboux 2002) of z [A, n, n] (float64) from the cells ``fixed`` held at
    ``level``: S = max(z, min over the 8 neighbours of S + eps x distance), iterated from +inf to the fixed point
    (checked every ``check_every`` sweeps, ``PROVENANCE['flood_check']``)."""
    big = torch.finfo(torch.float64).max / 4
    S = torch.where(fixed, level, torch.full_like(z, big))
    offs = [(dx, dy, (SQRT2 if dx and dy else 1.0) * eps) for dx, dy in OFFSETS8]
    it = 0
    while True:
        prev = S
        for _ in range(check_every):
            m = None
            for dx, dy, e in offs:
                r = torch.roll(S, (dx, dy), (1, 2)) + e
                m = r if m is None else torch.minimum(m, r)
            S = torch.where(fixed, S, torch.maximum(z, torch.minimum(S, m)))
        it += check_every
        if torch.equal(S, prev):
            return S
        if it > 4 * z.shape[1] * z.shape[2] + 64:
            raise RuntimeError("depression filling did not converge")


def _jump(ptr: torch.Tensor) -> torch.Tensor:
    """Pointer jumping to the fixed points: ptr <- ptr[ptr] until stable (log2 of the longest path steps)."""
    for _ in range(64):
        nxt = ptr[ptr]
        if torch.equal(nxt, ptr):
            return ptr
        ptr = nxt
    raise RuntimeError("pointer jumping did not converge (a cycle)")


def _exits(z, sea):
    """Exit cells [A, n, n]: the sea and each arena's lowest fine cell (a sink at its own level)."""
    A = z.shape[0]
    low = z.reshape(A, -1).argmin(-1)
    outlet = torch.zeros(A, z.shape[1] * z.shape[2], dtype=torch.bool, device=z.device)
    outlet[torch.arange(A, device=z.device), low] = True
    outlet = outlet.reshape(z.shape)
    return sea | outlet, outlet


def _receivers(z, exits, eps: float):
    """Steepest-descent receiver of every cell (flat [G]; exits and pits point to themselves) and the [G, 8] global
    neighbour table. A cell without a strictly lower neighbour that has equal neighbours (a flat) drains to an equal
    neighbour that comes before it in the strict order (z, eps-filled surface toward the exits, flat index), so flats
    drain to their edge or end in one pit."""
    A, n = z.shape[0], z.shape[1]
    N, G, dev = n * n, A * n * n, z.device
    zf = z.reshape(-1)
    gidx = torch.arange(G, device=dev)
    nbr, dist = _neighbour_index(n, dev)
    nb = (nbr[None] + (torch.arange(A, device=dev) * N)[:, None, None]).reshape(G, 8)
    zn = zf[nb]
    drop = (zf[:, None] - zn) / dist
    best = drop.argmax(-1, keepdim=True)
    bdrop = drop.gather(1, best)[:, 0]
    ex = exits.reshape(-1)
    rec = torch.where(bdrop > 0, nb.gather(1, best)[:, 0], gidx)
    tie = zn == zf[:, None]
    flat = (bdrop <= 0) & ~ex & tie.any(-1)
    if bool(flat.any()):
        Se = _flood(z, exits, z, eps).reshape(-1)
        Sn = Se[nb]
        lower = (Sn < Se[:, None]) | ((Sn == Se[:, None]) & (nb < gidx[:, None]))   # strict order (z, Se, index)
        Sm = torch.where(tie & lower, Sn, torch.full_like(Sn, math.inf))
        j = Sm.argmin(-1, keepdim=True)
        ok = torch.isfinite(Sm.gather(1, j)[:, 0])
        rec = torch.where(flat & ok, nb.gather(1, j)[:, 0], rec)
    return torch.where(ex, gidx, rec), nb


def _merge_tree(u, v, w, P: int, A: int):
    """Kruskal merge of the watershed graph (labels 0..P-1 the pits' watersheds, P + a the exits of arena a), edges
    (u, v) at sill height w sorted ascending: two depressions meeting at a sill become the children of a new node (each
    spills across the sill into the other's watershed); a depression meeting the exits' component becomes a top node
    that spills into the watershed across its sill. Returns Python lists (parent, ch0, ch1, spill, tgt) over the nodes
    (leaves first, every node after its children); tgt is a label (leaf or P + a). Set-up only: a loop over the sill
    edges (a few per depression), never over cells."""
    uf = list(range(P + A))
    ocean = [False] * P + [True] * A
    top = list(range(P)) + [-1] * A
    parent, ch0, ch1 = [-1] * P, [-1] * P, [-1] * P
    spill, tgt = [math.inf] * P, [-1] * P

    def find(x):
        while uf[x] != x:
            uf[x] = uf[uf[x]]
            x = uf[x]
        return x

    for a, b, h in zip(u, v, w):
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        if ocean[ra] and ocean[rb]:
            uf[ra] = rb
            continue
        if ocean[ra]:
            a, b, ra, rb = b, a, rb, ra
        ta = top[ra]
        if not ocean[rb]:
            tb = top[rb]
            m = len(parent)
            parent.append(-1)
            ch0.append(ta)
            ch1.append(tb)
            spill.append(math.inf)
            tgt.append(-1)
            parent[ta] = parent[tb] = m
            spill[ta] = spill[tb] = h
            tgt[ta], tgt[tb] = b, a
            uf[ra] = rb
            top[rb] = m
        else:
            spill[ta], tgt[ta] = h, b
            uf[ra] = rb
    for x in range(P):
        if not ocean[find(x)]:
            raise RuntimeError("a depression never reached the exits")
    return parent, ch0, ch1, spill, tgt


def _by_depth(depth: torch.Tensor):
    """(order, start) of nodes sorted by depth: nodes at depth d are order[start[d]:start[d + 1]]."""
    D = int(depth.max()) + 1 if depth.numel() else 0
    order = torch.argsort(depth, stable=True)
    count = torch.bincount(depth, minlength=D) if D else torch.zeros(0, dtype=torch.long, device=depth.device)
    start = torch.cat((torch.zeros(1, dtype=torch.long, device=depth.device), torch.cumsum(count, 0)))
    return order, start.tolist()


def drainage(z, sea, rules: PatchRules | None = None) -> dict:
    """Drainage tables of the arenas (set-up, float64): the depression hierarchy of z [A, n, n] with the exits (the
    sea and each arena's lowest cell). Every cell drains (steepest descent) to a pit's watershed (a leaf) or to the
    exits (``label`` [G]: leaf id, or P + a). Nodes: leaves (0..P-1) and merged depressions, each with its sill
    height ``spill``, capacity ``cap`` (kg/m^2 summed over its cells: the water it holds full to its sill),
    ``parent``, children ``ch`` [Nn, 2] and the label ``tgt`` its overflow goes to. ``band`` [G]: the lowest node
    whose water can cover the cell (-1: never under water). See the module docstring and ``PROVENANCE['hierarchy']``."""
    r = rules or PatchRules()
    z = torch.as_tensor(z).double()
    A, n = z.shape[0], z.shape[1]
    N, G, dev = n * n, A * n * n, z.device
    f64, lng = torch.float64, torch.long
    zf = z.reshape(-1)
    gidx = torch.arange(G, device=dev)
    exits, outlet = _exits(z, sea)
    ex = exits.reshape(-1)
    rec, nb = _receivers(z, exits, r.fill_eps_m)
    root = _jump(rec)
    pits = gidx[(rec == gidx) & ~ex]
    P = int(pits.numel())
    leaf_id = torch.full((G,), -1, dtype=lng, device=dev)
    leaf_id[pits] = torch.arange(P, device=dev)
    label = torch.where(ex[root], P + root // N, leaf_id[root])
    # sill edges between neighbouring watersheds: the lowest max(z, z') over their 8-connected cell pairs
    us, vs, ws = [], [], []
    for k in (0, 2, 4, 5):                                           # (1,0), (0,1), (1,1), (1,-1): each pair once
        j = nb[:, k]
        la, lb = label, label[j]
        m = la != lb
        us.append(torch.minimum(la, lb)[m])
        vs.append(torch.maximum(la, lb)[m])
        ws.append(torch.maximum(zf, zf[j])[m])
    u, v, w = torch.cat(us), torch.cat(vs), torch.cat(ws)
    key = u * (P + A) + v
    if key.numel():
        o = torch.argsort(w, stable=True)
        o = o[torch.argsort(key[o], stable=True)]
        ks = key[o]
        first = torch.ones_like(ks, dtype=torch.bool)
        first[1:] = ks[1:] != ks[:-1]
        sel = o[first]
        u, v, w, key = u[sel], v[sel], w[sel], key[sel]
        o = torch.argsort(key, stable=True)
        o = o[torch.argsort(w[o], stable=True)]
        u, v, w = u[o], v[o], w[o]
    parent, ch0, ch1, spill, tgt = _merge_tree(u.tolist(), v.tolist(), w.tolist(), P, A)
    Nn = len(parent)
    depth = [0] * Nn
    for m in range(Nn - 1, -1, -1):                                  # parents come after their children
        if parent[m] >= 0:
            depth[m] = depth[parent[m]] + 1
    pit_z = zf[pits].tolist()
    bottom = pit_z + [0.0] * (Nn - P)
    arena = (pits // N).tolist() + [0] * (Nn - P)
    for m in range(P, Nn):
        bottom[m] = min(bottom[ch0[m]], bottom[ch1[m]])
        arena[m] = arena[ch0[m]]
    T = lambda x, dt=lng: torch.tensor(x, dtype=dt, device=dev)       # noqa: E731
    parent_t, ch_t = T(parent), torch.stack((T(ch0), T(ch1)), 1) if Nn else torch.zeros(0, 2, dtype=lng, device=dev)
    spill_t, tgt_t, depth_t = T(spill, f64), T(tgt), T(depth)
    bottom_t, arena_t = T(bottom, f64), T(arena)
    Dmax = int(depth_t.max()) if Nn else 0
    order, start = _by_depth(depth_t) if Nn else (torch.zeros(0, dtype=lng, device=dev), [0, 0])
    # band of each cell: the lowest node (from its leaf up) whose sill is above it
    band = torch.full((G,), -1, dtype=lng, device=dev)
    cur = torch.where(label < P, label, torch.full_like(label, -1))
    for _ in range(Dmax + 1):
        ok = (band < 0) & (cur >= 0)
        if not bool(ok.any()):
            break
        cc = cur.clamp_min(0)
        hit = ok & (spill_t[cc] > zf) if Nn else ok & False
        band = torch.where(hit, cur, band)
        cur = torch.where(ok & ~hit, parent_t[cc] if Nn else cur, torch.full_like(cur, -1))
    inb = band >= 0
    cnt = torch.zeros(Nn, dtype=f64, device=dev).index_add_(0, band[inb], torch.ones_like(zf[inb]))
    zs = torch.zeros(Nn, dtype=f64, device=dev).index_add_(0, band[inb], zf[inb])
    own = cnt.clone()
    for d in range(Dmax, 0, -1):
        idx = order[start[d]:start[d + 1]]
        cnt.index_add_(0, parent_t[idx], cnt[idx])
        zs.index_add_(0, parent_t[idx], zs[idx])
    cap = ((cnt * spill_t - zs) * RHO_W).clamp_min(0) if Nn else cnt
    meta = (ch_t[:, 0] >= 0) if Nn else torch.zeros(0, dtype=torch.bool, device=dev)
    c0, c1 = ch_t[:, 0].clamp_min(0), ch_t[:, 1].clamp_min(0)
    V0 = torch.where(meta, cap[c0] + cap[c1], torch.zeros_like(cap)) if Nn else cap
    n0 = torch.where(meta, cnt[c0] + cnt[c1], torch.zeros_like(cnt)) if Nn else cnt
    base = torch.where(meta, spill_t[c0], bottom_t) if Nn else bottom_t
    cap = torch.maximum(cap, V0)
    # ancestors by depth: anc[m, d] the ancestor of m at depth d (m itself at its own depth), -1 padded
    anc = torch.full((Nn, Dmax + 1), -1, dtype=lng, device=dev)
    for d in range(Dmax + 1):
        idx = order[start[d]:start[d + 1]]
        if d > 0:
            anc[idx, :d] = anc[parent_t[idx], :d]
        anc[idx, d] = idx
    # band hypsometry: band cells sorted by (node, z); vz = water (kg/m^2 summed) that brings the level to a cell
    cells_b = gidx[inb]
    o1 = torch.argsort(zf[cells_b], stable=True)
    o2 = torch.argsort(band[cells_b][o1], stable=True)
    h_cells = cells_b[o1[o2]]
    h_node = band[h_cells]
    h_rel = (zf[h_cells] - base[h_node]).clamp_min(0) if Nn else zf[h_cells]
    count = torch.bincount(h_node, minlength=Nn)
    h_start = torch.cumsum(count, 0) - count
    csum = torch.cumsum(h_rel, 0)
    before = (csum - h_rel)[h_start.clamp_max(max(h_rel.numel() - 1, 0))] if h_rel.numel() else csum
    cum = csum - before[h_node]
    pos = (torch.arange(h_cells.numel(), device=dev) - h_start[h_node]).double()
    vz = RHO_W * ((n0[h_node] + pos + 1) * h_rel - cum)
    # top nodes and where they spill (the top holding the target leaf, or an exit slot after the tops)
    tops = (parent_t < 0).nonzero().reshape(-1) if Nn else torch.zeros(0, dtype=lng, device=dev)
    Tn = int(tops.numel())
    top_pos = torch.full((Nn,), -1, dtype=lng, device=dev)
    top_pos[tops] = torch.arange(Tn, device=dev)
    tt = tgt_t[tops]
    tleaf = tt < P
    top_dest = torch.where(tleaf, top_pos[anc[tt.clamp_max(max(P - 1, 0)), 0]] if Nn else tt,
                           Tn + (tt - P).clamp_min(0))
    rounds, x = 0, torch.arange(Tn, device=dev)
    dest_full = torch.cat((top_dest, Tn + torch.arange(A, device=dev)))
    while bool((x < Tn).any()):
        x = dest_full[x]
        rounds += 1
        if rounds > Tn + 1:
            raise RuntimeError("the top-level spill graph has a cycle")
    top_of_cell = torch.where(inb, anc[band.clamp_min(0), 0] if Nn else band, torch.full_like(band, -1))
    spill_m = torch.where(inb, spill_t[top_of_cell.clamp_min(0)] if Nn else zf, zf)
    land = ~sea.reshape(A, N)
    lake_share = (inb.reshape(A, N) & land).double().sum(-1) / land.double().sum(-1).clamp_min(1)
    return {"z": z, "exits": exits, "outlet": outlet, "label": label, "leaves": P, "nodes": Nn, "depth": Dmax,
            "parent": parent_t, "ch": ch_t, "spill": spill_t, "tgt": tgt_t, "node_depth": depth_t, "arena": arena_t,
            "bottom": bottom_t, "base": base, "cap": cap, "V0": V0, "n0": n0, "cells_in": cnt, "cells_own": own,
            "anc": anc, "order": order, "start": torch.tensor(start, dtype=lng, device=dev),
            "band": band, "h_cells": h_cells, "h_node": h_node, "h_rel": h_rel, "h_cum": cum, "h_vz": vz,
            "h_start": h_start, "tops": tops, "top_dest": top_dest, "top_rounds": rounds,
            "lake": top_of_cell.reshape(A, n, n), "spill_m": spill_m.float().reshape(A, n, n),
            "lake_share": lake_share}


def condition_terrain(z, sea, keep_m, rules: PatchRules | None = None):
    """Fill (sediment infill) every closed depression shallower than keep_m ([A] or a number, m): on the depression
    hierarchy a node is shallow when its sill lies less than keep_m above its lowest ground once its shallow children
    are filled; the cells of shallow nodes are raised to their sill (Planchon & Darboux filling with ``fill_eps_m`` per
    cell, so they drain), and the hierarchy is rebuilt until nothing shallow is left (``condition_rounds``). Returns
    (z float64 [A, n, n], rounds used). ``PROVENANCE['lake_keep_rms']``."""
    r = rules or PatchRules()
    z = torch.as_tensor(z).double().clone()
    A = z.shape[0]
    keep = torch.as_tensor(keep_m, dtype=torch.float64, device=z.device).reshape(-1).expand(A).tolist()
    for it in range(r.condition_rounds):
        H = drainage(z, sea, r)
        Nn, P = H["nodes"], H["leaves"]
        if Nn == 0:
            return z, it
        spill, bottom = H["spill"].tolist(), H["bottom"].tolist()
        ch, arena = H["ch"].tolist(), H["arena"].tolist()
        eb, filled = [0.0] * Nn, [False] * Nn
        for m in range(Nn):                                          # children before parents
            if m < P:
                eb[m] = bottom[m]
            else:
                c0, c1 = ch[m]
                eb[m] = min(spill[c0] if filled[c0] else eb[c0], spill[c1] if filled[c1] else eb[c1])
            filled[m] = spill[m] - eb[m] < keep[arena[m]]
        if not any(filled):
            return z, it
        band = H["band"]
        f = torch.tensor(filled, dtype=torch.bool, device=z.device)
        fill = ((band >= 0) & f[band.clamp_min(0)]).reshape(z.shape)
        z = _flood(z, ~fill, z, r.fill_eps_m)
    return z, r.condition_rounds


def _land_water(ext, anc, leaves, amount, above: int):
    """Add amount [M] to every ancestor of the leaves [M] deeper than ``above`` (water landing in their subtrees)."""
    if leaves.numel() == 0:
        return
    rows = anc[leaves]
    cols = torch.arange(rows.shape[1], device=rows.device)
    m = (rows >= 0) & (cols[None] > above) & (amount[:, None] > 0)
    ext.index_add_(0, rows[m], amount[:, None].expand_as(rows)[m])


def _solve(water_in, H):
    """Fill-spill-merge: the water F [Nn] each node's subtree holds (float64) and what leaves each arena [A], for
    water_in [P] poured into the leaves' watersheds. Bottom-up subtree sums; the top nodes cascade over their spill
    graph; then each merged node, top-down per depth, lets a child above its capacity spill into its sibling across
    their sill (both full: the rest stands in the parent above the sill)."""
    Nn, P, Dmax = H["nodes"], H["leaves"], H["depth"]
    dev = water_in.device
    A = H["exits"].shape[0]
    order, start = H["order"], H["start"].tolist()
    parent, ch, cap, anc, tgt = H["parent"], H["ch"], H["cap"], H["anc"], H["tgt"]
    sub = torch.zeros(Nn, dtype=torch.float64, device=dev)
    sub[:P] = water_in
    for d in range(Dmax, 0, -1):
        idx = order[start[d]:start[d + 1]]
        sub.index_add_(0, parent[idx], sub[idx])
    ext = torch.zeros_like(sub)
    tops = H["tops"]
    Tn = int(tops.numel())
    V = torch.cat((sub[tops], torch.zeros(A, dtype=torch.float64, device=dev)))
    cap_t = torch.cat((cap[tops], torch.full((A,), math.inf, dtype=torch.float64, device=dev)))
    sent = torch.zeros(Tn, dtype=torch.float64, device=dev)
    for _ in range(H["top_rounds"]):
        ex = (V[:Tn] - cap_t[:Tn]).clamp_min(0)
        V[:Tn] = V[:Tn] - ex
        V.index_add_(0, H["top_dest"], ex)
        sent = sent + ex
    tt = tgt[tops]
    leaf = tt < P
    _land_water(ext, anc, tt[leaf], sent[leaf], -1)
    for d in range(Dmax):
        idx = order[start[d]:start[d + 1]]
        idx = idx[ch[idx, 0] >= 0]
        if idx.numel() == 0:
            continue
        X, Y = ch[idx, 0], ch[idx, 1]
        tX, tY = sub[X] + ext[X], sub[Y] + ext[Y]
        ovX, ovY = (tX - cap[X]).clamp_min(0), (tY - cap[Y]).clamp_min(0)
        _land_water(ext, anc, tgt[X], ovX, d)
        _land_water(ext, anc, tgt[Y], ovY, d)
    return torch.minimum(sub + ext, cap), V[Tn:]


def _pond(F, H):
    """Pond water (kg/m^2, float64 [G]) of nodes holding F [Nn]: a node whose water rises above its children's sill
    (or a leaf with any water) stands at one level over its band, filled from its lowest cells; a node full to its sill
    stands at the sill; every cell takes the highest level among its band node and that node's ancestors."""
    z = H["z"].reshape(-1)
    G, Nn = z.numel(), H["nodes"]
    pond = torch.zeros(G, dtype=torch.float64, device=z.device)
    if Nn == 0:
        return pond
    w = (F - H["V0"]).clamp_min(0)
    active = (w > 0) & (F > 0)
    full = F >= H["cap"]
    hn = H["h_node"]
    under = (H["h_vz"] <= w[hn]).double()
    k = torch.zeros(Nn, dtype=torch.float64, device=z.device).index_add_(0, hn, under)
    pos = (H["h_start"] + k.long() - 1).clamp_min(0).clamp_max(max(hn.numel() - 1, 0))
    cumk = torch.where(k > 0, H["h_cum"][pos] if hn.numel() else k, torch.zeros_like(k))
    rel = (w / RHO_W + cumk) / (H["n0"] + k).clamp_min(1e-300)
    rel = torch.where(full, H["spill"] - H["base"], rel)
    surf = torch.where(active | full, H["base"] + rel, torch.full_like(rel, -math.inf))
    order, start = H["order"], H["start"].tolist()
    for d in range(1, H["depth"] + 1):
        idx = order[start[d]:start[d + 1]]
        surf[idx] = torch.maximum(surf[idx], surf[H["parent"][idx]])
    band = H["band"]
    inb = band >= 0
    depth = (surf[band.clamp_min(0)] - z).clamp_min(0)
    return torch.where(inb, depth * RHO_W, pond)


def route(surplus, pond, geom: PatchGeometry):
    """Route the day's runoff ``surplus`` and the standing ``pond`` water ([A, n, n] kg/m^2, float64) through the
    depression hierarchy (each cell's water enters the watershed it lies in). Returns (new pond [A, n, n] float64,
    water that left each arena [A] in kg/m^2 summed over cells)."""
    H = geom.hydro
    A, n = geom.A, geom.n
    P = H["leaves"]
    tot = torch.zeros(P + A, dtype=torch.float64, device=pond.device).index_add_(
        0, H["label"], (surplus + pond).reshape(-1).double())
    if H["nodes"] == 0:
        return torch.zeros(A, n, n, dtype=torch.float64, device=pond.device), tot[P:]
    F, out = _solve(tot[:P], H)
    return _pond(F, H).reshape(A, n, n), tot[P:] + out


def lake_share(geom: PatchGeometry) -> torch.Tensor:
    """Share [A] of each arena's land cells that lie in a lake basin (below a depression's sill: standing water
    covers them once the basins are full)."""
    return geom.hydro["lake_share"].clone()


def water_share(state, geom: PatchGeometry, min_kg_m2: float = 10.0) -> torch.Tensor:
    """Share [A] of each arena's land cells under more than ``min_kg_m2`` of standing water (default 1 cm)."""
    land = ~geom.sea
    wet = (state.pond > min_kg_m2) & land
    return wet.double().sum((1, 2)) / land.double().sum((1, 2)).clamp_min(1)


# ============================================================================================ light and forcing
def _sun_vectors(dec, lock, frac):
    """Unit vectors toward the sun in planet coordinates [..., 3] at day fractions frac (version 2's phase: the
    subsolar longitude is -2 pi frac); a locked planet's sun stands at (lat 0, lon 0)."""
    lam = -2 * math.pi * frac
    cd = torch.cos(dec)
    sg = torch.stack((cd * torch.cos(lam), cd * torch.sin(lam), torch.sin(dec).expand_as(lam)), -1)
    fixed = torch.tensor([1.0, 0.0, 0.0], dtype=sg.dtype, device=sg.device).expand_as(sg)
    return torch.where(lock[..., None], fixed, sg)


def _per_arena(x, A: int, device, dtype=torch.float64):
    return torch.as_tensor(x, dtype=dtype, device=device).reshape(-1).expand(A)


def sun_local(geom: PatchGeometry, declination, locked, day: int, bouts: int = 4, samples: int = 8,
              solar_day_s=None) -> torch.Tensor:
    """Unit sun directions [A, bouts, samples, 3] in each arena's local (east, north, up) frame, sampled at the
    middle of ``samples`` equal parts of each bout of day ``day``. ``declination`` [A] (rad) and ``locked`` [A]
    bool per arena; ``solar_day_s`` [A] or a number (DAY_S when not given; infinite: the sun stands still) is the
    length of the solar day (``PROVENANCE['sun_time']``)."""
    dev, f64 = geom.device, torch.float64
    dec = _per_arena(declination, geom.A, dev)
    lock = torch.as_tensor(locked, device=dev).bool().reshape(-1).expand(geom.A)
    sd = _per_arena(DAY_S if solar_day_s is None else solar_day_s, geom.A, dev)
    b = torch.arange(bouts, dtype=f64, device=dev)[:, None]
    k = torch.arange(samples, dtype=f64, device=dev)[None, :]
    t_s = (float(day) + (b + (k + 0.5) / samples) / bouts) * DAY_S   # [B, S]
    frac = torch.where(torch.isfinite(sd)[:, None, None], torch.remainder(t_s[None] / sd[:, None, None], 1.0),
                       torch.zeros(geom.A, bouts, samples, dtype=f64, device=dev))
    sg = _sun_vectors(dec[:, None, None], lock[:, None, None], frac)
    basis = torch.stack((geom.east, geom.north, geom.up), 1).double()  # [A, 3 (e, n, u), 3 (xyz)]
    return torch.einsum("absx,acx->absc", sg, basis).float()


def flat_mean_incidence(geom: PatchGeometry, declination, locked, solar_day_s=None, per_day: int = 32,
                        max_samples: int = 4096) -> torch.Tensor:
    """The flat ground's mean max(0, cos zenith) over one whole solar day [A] (float64, ``PROVENANCE['flat_mean']``):
    the midpoint rule at ``per_day`` samples per DAY_S (at least 8 per solar day), so the sim days' own samples cover
    the same phases (exactly, when the solar day is commensurate with DAY_S); the analytic diurnal mean m1 / pi for
    days longer than ``max_samples`` samples; a locked planet's up . substellar point."""
    dev, f64 = geom.device, torch.float64
    A = geom.A
    dec = _per_arena(declination, A, dev)
    lock = torch.as_tensor(locked, device=dev).bool().reshape(-1).expand(A)
    sd = _per_arena(DAY_S if solar_day_s is None else solar_day_s, A, dev)
    up = geom.up.double()
    lat = torch.asin(up[:, 2].clamp(-1, 1))
    h0 = torch.acos((-torch.tan(lat) * torch.tan(dec)).clamp(-1.0, 1.0))
    m1 = (h0 * torch.sin(lat) * torch.sin(dec) + torch.cos(lat) * torch.cos(dec) * torch.sin(h0)).clamp_min(0)
    out = m1 / math.pi
    M = torch.where(torch.isfinite(sd), torch.round(sd / DAY_S * per_day), torch.full_like(sd, math.inf))
    M = M.clamp_min(8)
    use = torch.isfinite(M) & (M <= max_samples) & ~lock
    if bool(use.any()):
        Mmax = int(M[use].max())
        j = torch.arange(Mmax, dtype=f64, device=dev)
        Mu = torch.where(use, M, torch.ones_like(M))
        frac = (j[None] + 0.5) / Mu[:, None]
        valid = j[None] < Mu[:, None]
        sg = _sun_vectors(dec[:, None], torch.zeros(A, 1, dtype=torch.bool, device=dev), frac)
        mu = (sg * up[:, None, :]).sum(-1).clamp_min(0)
        sampled = torch.where(valid, mu, torch.zeros_like(mu)).sum(-1) / Mu
        out = torch.where(use, sampled, out)
    return torch.where(lock, up[:, 0].clamp_min(0), out)


def erbs_diffuse_share(k_t, sunset_rad):
    """Erbs, Klein & Duffie's (1982) daily diffuse fraction of the clearness index K_T and the sunset hour angle
    (``PROVENANCE['diffuse_share']``)."""
    k = torch.as_tensor(k_t, dtype=torch.float64).clamp(0, 1)
    ws = torch.as_tensor(sunset_rad, dtype=torch.float64, device=k.device)
    low = torch.where(k < 0.715, 1 - 0.2727 * k + 2.4495 * k ** 2 - 11.9514 * k ** 3 + 9.3879 * k ** 4,
                      torch.full_like(k, 0.143))
    high = torch.where(k < 0.722, 1 + 0.2832 * k - 2.5557 * k ** 2 + 0.8448 * k ** 3, torch.full_like(k, 0.175))
    return torch.where(ws.abs() < ERBS_WS_RAD, low, high).clamp(0, 1)


def sky_view(geom: PatchGeometry) -> torch.Tensor:
    """Relative diffuse light per horizontal m^2 [A, n, n]: v = (1 + cos slope) / (2 cos slope), divided by its patch
    mean (``PROVENANCE['light']``)."""
    c = torch.cos(geom.slope.double())
    v = (1 + c) / (2 * c)
    return (v / v.mean((1, 2), keepdim=True)).float()


def light(geom: PatchGeometry, forcing) -> tuple[torch.Tensor, torch.Tensor]:
    """(day_sw [A, n, n], bout_sw [A, B, n, n]) W/m^2 per horizontal m^2 of fine ground (``PROVENANCE['light']``):
    the flat-ground instantaneous light sw_day x cos zenith / flat_mu, its beam on each cell's own incidence and its
    diffuse by the sky view. Over a whole solar day a flat patch gets exactly ``sw_day_w_m2``."""
    sun = forcing.sun.float()                                          # [A, B, S, 3]
    B = sun.shape[1]
    sz = sun[..., 2]
    above = sz > 0
    mu = forcing.flat_mu.double()
    g0 = torch.where(mu > 0, forcing.sw_day_w_m2.double() / mu.clamp_min(1e-300), torch.zeros_like(mu))   # [A]
    fd = forcing.diffuse_share.double()
    beam_w = ((1 - fd) * g0).float()[:, None, None]
    sky = (sz.clamp_min(0) * above).double().mean(2)                                        # [A, B]
    diffuse = (sky * (fd * g0)[:, None]).float()
    view = sky_view(geom)
    zx, zy = geom.dzdx[:, None], geom.dzdy[:, None]
    out = []
    for b in range(B):                                                 # per bout: [A, S, n, n] at a time
        s = sun[:, b]
        inc = (s[..., 2, None, None] - zx * s[..., 0, None, None] - zy * s[..., 1, None, None]).clamp_min(0)
        inc = torch.where(above[:, b, :, None, None], inc, torch.zeros_like(inc))
        out.append(inc.mean(1) * beam_w + diffuse[:, b, None, None] * view)
    bout_sw = torch.stack(out, 1)
    return bout_sw.mean(1), bout_sw


@dataclass
class PatchForcing:
    """The global cell's conditions of one day, per arena [A] (float32 unless noted)."""
    t_air_k: torch.Tensor          # the cell's air temperature at its mean ground
    lapse_k_m: torch.Tensor        # g / c_p
    precip_kg_m2: torch.Tensor     # precipitation that day
    sw_day_w_m2: torch.Tensor      # daily-mean shortwave at flat ground
    sun: torch.Tensor              # [A, B, S, 3] local sun directions
    flat_mu: torch.Tensor          # [A] float64, flat ground's mean cos zenith over a whole solar day
    diffuse_share: torch.Tensor    # [A] diffuse share of the shortwave
    par_fraction: torch.Tensor
    p_co2_pa: torch.Tensor
    p_o2_pa: torch.Tensor
    vapour_kg_m2: torch.Tensor     # the cell's column water vapour
    vapour_h_ref_m: torch.Tensor   # vapour scale height at t_ref_k
    t_ref_k: torch.Tensor

    @property
    def bouts(self) -> int:
        return int(self.sun.shape[1])

    def vapour_pa(self) -> torch.Tensor:
        """Surface water vapour pressure [A] (Pa) of the cell's air: column / H_w(T) x R T / M_w, as
        ``climate.vapour_pressure`` (for evaporation from bodies)."""
        h_w = self.vapour_h_ref_m * (self.t_air_k / self.t_ref_k) ** 2
        return self.vapour_kg_m2.clamp_min(0) / h_w * K.R_GAS * self.t_air_k / M_H2O


def solar_days(specs) -> list:
    """The solar day (s) of each spec: ``day_length_s`` (infinite for a locked planet)."""
    return [math.inf if getattr(s, "tidally_locked", False) else float(s.day_length_s) for s in specs]


def forcing_from_climate(geom: PatchGeometry, clim, cdiag, P, day: int, bouts: int = 4, samples: int | None = None,
                         solar_day_s=None, specs=None, rules: PatchRules | None = None) -> PatchForcing:
    """The forcing of day ``day`` from version 2's climate after its step (``clim`` the new state, ``cdiag`` its
    diagnostics, ``P`` its parameters). The solar day comes from ``specs`` (their ``day_length_s``) or
    ``solar_day_s`` [W] (or a number); one of them is required (``PROVENANCE['sun_time']``)."""
    r = rules or PatchRules()
    w, c = geom.world, geom.cell
    S = r.light_samples if samples is None else int(samples)
    if solar_day_s is None:
        if specs is None:
            raise ValueError("forcing_from_climate needs the solar day: pass specs (day_length_s) or solar_day_s")
        solar_day_s = solar_days(specs)
    sd = torch.as_tensor(solar_day_s, dtype=torch.float64, device=geom.device).reshape(-1).expand(geom.W)[w]
    dec = cl.declination(P, day).to(geom.device)[w]
    locked = P["locked"][w]
    pp = cl.partial_pressures(clim, P)
    f32 = torch.float32
    sw = cdiag["surface_sw"][w, c].double()
    toa = cdiag["insolation"][w, c].double()
    lat = torch.asin(geom.up.double()[:, 2].clamp(-1, 1))
    ws = torch.where(locked, torch.full_like(lat, math.pi),
                     torch.acos((-torch.tan(lat) * torch.tan(dec)).clamp(-1.0, 1.0)))
    fd = torch.where(toa > 0, erbs_diffuse_share(sw / toa.clamp_min(1e-300), ws), torch.ones_like(sw))
    return PatchForcing(t_air_k=clim.T[w, c].float(), lapse_k_m=P["lapse"][w].float(),
                        precip_kg_m2=cdiag["precipitation"][w, c].float(),
                        sw_day_w_m2=sw.float(),
                        sun=sun_local(geom, dec, locked, day, bouts, S, sd),
                        flat_mu=flat_mean_incidence(geom, dec, locked, sd, bouts * S),
                        diffuse_share=fd.float(),
                        par_fraction=P["par_fraction"][w].float(), p_co2_pa=pp[w, cl.I_CO2].to(f32),
                        p_o2_pa=pp[w, cl.I_O2].to(f32), vapour_kg_m2=clim.vapour[w, c].float(),
                        vapour_h_ref_m=P["vapour_h_ref"][w].float(), t_ref_k=P["t_target"][w].float())


def constant_forcing(geom: PatchGeometry, day: int = 0, t_air_k=293.0, precip_kg_m2=2.0, sw_day_w_m2=200.0,
                     declination=0.0, locked=False, bouts: int = 4, samples: int = 8, par_fraction=0.37,
                     p_co2_pa=28.0, p_o2_pa=21200.0, vapour_kg_m2=20.0, vapour_h_ref_m=2000.0, t_ref_k=288.0,
                     lapse_k_m=9.8e-3, solar_day_s=None, diffuse_share=0.0) -> PatchForcing:
    """A forcing from given numbers (tests, stand-alone arenas); each is a number or [A]. ``diffuse_share`` 0 puts
    all light in the beam."""
    A, dev = geom.A, geom.device

    def v(x):
        return torch.as_tensor(x, dtype=torch.float32, device=dev).reshape(-1).expand(A).clone()

    return PatchForcing(t_air_k=v(t_air_k), lapse_k_m=v(lapse_k_m), precip_kg_m2=v(precip_kg_m2),
                        sw_day_w_m2=v(sw_day_w_m2),
                        sun=sun_local(geom, declination, locked, day, bouts, samples, solar_day_s),
                        flat_mu=flat_mean_incidence(geom, declination, locked, solar_day_s, bouts * samples),
                        diffuse_share=v(diffuse_share),
                        par_fraction=v(par_fraction), p_co2_pa=v(p_co2_pa), p_o2_pa=v(p_o2_pa),
                        vapour_kg_m2=v(vapour_kg_m2), vapour_h_ref_m=v(vapour_h_ref_m), t_ref_k=v(t_ref_k))


def fine_temperature(geom: PatchGeometry, forcing: PatchForcing) -> torch.Tensor:
    """Air temperature [A, n, n] at each fine cell: the cell's temperature less g / c_p times the height above the
    cell's mean ground (``PROVENANCE['fine_temperature']``)."""
    return forcing.t_air_k[:, None, None] - forcing.lapse_k_m[:, None, None] * (geom.elev - geom.base_m[:, None, None])


def evaporation_demand(t_k, forcing: PatchForcing, crules: cl.ClimateRules) -> torch.Tensor:
    """Evaporation of a free water surface (kg/m^2 per day) at the fine temperature t_k [A, n, n]: the climate's
    bulk formula (``PROVENANCE['evaporation']``)."""
    t = t_k.double()
    h_w = forcing.vapour_h_ref_m.double()[:, None, None] * (t / forcing.t_ref_k.double()[:, None, None]) ** 2
    v_sat = cl.e_sat_pa(t) * M_H2O / (K.R_GAS * t) * h_w
    rate = (crules.transfer_coeff * crules.wind_m_s * DAY_S / h_w).clamp_max(1.0)
    return rate * (v_sat - forcing.vapour_kg_m2.double()[:, None, None]).clamp_min(0)


# ============================================================================================ state
@dataclass
class PatchState:
    """Fields [A, n, n] float32 (kg/m^2: plant, wood, litter, seed as kg C; nutrient kg N; soil, pond, snow kg
    water; t_acclim K) and the ledger counters [A] float64 (kg)."""
    plant: torch.Tensor
    wood: torch.Tensor
    litter: torch.Tensor
    seed: torch.Tensor
    nutrient: torch.Tensor
    t_acclim: torch.Tensor
    soil: torch.Tensor
    pond: torch.Tensor
    snow: torch.Tensor
    carbon0: torch.Tensor
    c_air: torch.Tensor            # carbon taken from the air (fixed - respired - decomposed + sown)
    c_pending: torch.Tensor        # carbon taken from the air outside a step (sowing), reported by the next step
    c_taken: torch.Tensor          # carbon taken by bodies (eaten, gathered)
    c_added: torch.Tensor          # carbon added (faeces, carcasses, items)
    water0: torch.Tensor
    w_precip: torch.Tensor
    w_evap: torch.Tensor
    w_out: torch.Tensor            # exits, seepage and glacier discharge
    w_taken: torch.Tensor          # fresh water drunk
    w_given: torch.Tensor          # water returned by bodies
    sea_water: torch.Tensor        # sea water drunk less returned to the sea (the global ocean's, not this ledger)
    nitrogen0: torch.Tensor
    n_fixed: torch.Tensor
    n_removed: torch.Tensor        # nitrogen taken by bodies (in what they ate)
    n_added: torch.Tensor          # nitrogen returned by bodies (faeces, urine, carcasses)
    day: int = 0

    def clone(self) -> "PatchState":
        return replace(self, **{f.name: _copy(getattr(self, f.name)) for f in fields(self)})

    def state_dict(self) -> dict:
        return {f.name: _copy(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_state(cls, d: dict) -> "PatchState":
        return cls(**{k: _copy(v) for k, v in d.items()})


def _copy(v):
    return v.clone() if isinstance(v, torch.Tensor) else v


_FIELDS = ("plant", "wood", "litter", "seed", "nutrient", "t_acclim", "soil", "pond", "snow")


def _arena_total(x, cell_m2: float) -> torch.Tensor:
    return x.double().sum((1, 2)) * cell_m2


def carbon_stores(state: PatchState, geom: PatchGeometry) -> torch.Tensor:
    """Organic carbon of each arena [A] (float64 kg C)."""
    return _arena_total(state.plant.double() + state.wood.double() + state.litter.double() + state.seed.double(),
                        geom.cell_m2)


def water_stores(state: PatchState, geom: PatchGeometry) -> torch.Tensor:
    """Water of each arena [A] (float64 kg): soil + pond + snow."""
    return _arena_total(state.soil.double() + state.pond.double() + state.snow.double(), geom.cell_m2)


def nitrogen_stores(state: PatchState, geom: PatchGeometry, rules: PatchRules | None = None) -> torch.Tensor:
    """Nitrogen of each arena [A] (float64 kg N): mineral + soft tissue + wood + seeds."""
    b = (rules or PatchRules()).bio
    x = state.nutrient.double() + (state.plant.double() + state.seed.double()) / b.cn_leaf \
        + state.wood.double() / b.cn_wood
    return _arena_total(x, geom.cell_m2)


def carbon_ledger(state: PatchState, geom: PatchGeometry) -> torch.Tensor:
    """Carbon ledger error [A] (kg C): stores - carbon from the air + taken - added - baseline."""
    return carbon_stores(state, geom) - state.c_air + state.c_taken - state.c_added - state.carbon0


def water_ledger(state: PatchState, geom: PatchGeometry) -> torch.Tensor:
    """Water ledger error [A] (kg): stores - precipitation + evaporation + outflow + taken - given - baseline."""
    return (water_stores(state, geom) - state.w_precip + state.w_evap + state.w_out + state.w_taken - state.w_given
            - state.water0)


def nitrogen_ledger(state: PatchState, geom: PatchGeometry, rules: PatchRules | None = None) -> torch.Tensor:
    """Nitrogen ledger error [A] (kg N): stores - fixed + removed by bodies - returned by bodies - baseline."""
    return nitrogen_stores(state, geom, rules) - state.n_fixed + state.n_removed - state.n_added - state.nitrogen0


def reset_ledgers(state: PatchState, geom: PatchGeometry, rules: PatchRules | None = None) -> PatchState:
    """Re-base the three ledgers on the current stores (the counters restart at zero; pending exchange is kept)."""
    z = torch.zeros_like(state.c_air)
    return replace(state, carbon0=carbon_stores(state, geom), water0=water_stores(state, geom),
                   nitrogen0=nitrogen_stores(state, geom, rules), c_air=z.clone(), c_taken=z.clone(),
                   c_added=z.clone(), w_precip=z.clone(), w_evap=z.clone(), w_out=z.clone(), w_taken=z.clone(),
                   w_given=z.clone(), n_fixed=z.clone(), n_removed=z.clone(), n_added=z.clone())


def init_state(geom: PatchGeometry, rules: PatchRules | None = None) -> PatchState:
    """Bare ground (no plants, no seeds, no litter), the biosphere's starting mineral N and the climate's
    half-full bucket on every land fine cell, no ponds or snow; the ledgers are based on it."""
    r = rules or PatchRules()
    land = (~geom.sea).float()
    zeros = torch.zeros_like(geom.elev)
    z64 = torch.zeros(geom.A, dtype=torch.float64, device=geom.device)
    st = PatchState(plant=zeros.clone(), wood=zeros.clone(), litter=zeros.clone(), seed=zeros.clone(),
                    nutrient=land * r.bio.nutrient_start_kg_m2, t_acclim=zeros.clone(),
                    soil=land * (r.climate.soil_start * r.climate.bucket_kg_m2), pond=zeros.clone(),
                    snow=zeros.clone(), carbon0=z64.clone(), c_air=z64.clone(), c_pending=z64.clone(),
                    c_taken=z64.clone(), c_added=z64.clone(), water0=z64.clone(), w_precip=z64.clone(),
                    w_evap=z64.clone(), w_out=z64.clone(), w_taken=z64.clone(), w_given=z64.clone(),
                    sea_water=z64.clone(), nitrogen0=z64.clone(), n_fixed=z64.clone(), n_removed=z64.clone(),
                    n_added=z64.clone())
    return reset_ledgers(st, geom, r)


def sow(state: PatchState, geom: PatchGeometry, gen, rules: PatchRules | None = None, u=None) -> PatchState:
    """Sparse uniform sowing: each land fine cell gets ``sow_kg_c_m2`` of seed with probability ``sow_share``
    (drawn on ``gen``, or the uniform draws ``u`` [A, n, n] given, e.g. one world's arenas from that world's own
    generator; no habitability bias); the carbon comes from the air (booked in ``c_air`` and in ``c_pending``, which
    the next :func:`step` adds to its ``exchange_mol`` so the world removes it from the global air), the nitrogen
    from the cell's mineral pool (less seed where it is short)."""
    r = rules or PatchRules()
    if u is None:
        u = torch.rand(geom.elev.shape, generator=gen, device=gen.device)
    u = u.to(geom.device)
    want = torch.where((u < r.sow_share) & ~geom.sea, r.sow_kg_c_m2, 0.0)
    give = torch.minimum(want, state.nutrient.clamp_min(0) * r.bio.cn_leaf)
    seed = state.seed + give
    added = seed.double() - state.seed.double()
    nutrient = (state.nutrient.double() - added / r.bio.cn_leaf).float()
    kg = _arena_total(added, geom.cell_m2)
    return replace(state, seed=seed, nutrient=nutrient, c_air=state.c_air + kg, c_pending=state.c_pending + kg)


def _spread(x, side: float, diag: float):
    """Each cell's x [A, n, n] sent to its 8 periodic neighbours with weights side (4 sides) and diag (4 diagonals);
    returns what each cell receives."""
    out = torch.zeros_like(x)
    for dx, dy in OFFSETS8:
        out = out + torch.roll(x, (dx, dy), (1, 2)) * (diag if dx and dy else side)
    return out


# ============================================================================================ the day
def step(state: PatchState, geom: PatchGeometry, forcing: PatchForcing, rules: PatchRules | None = None):
    """One day of the patches: water, then plants. Returns (new state, diagnostics).

    Diagnostics: t_k [A, n, n] (fine air temperature), day_sw [A, n, n] and bout_sw [A, B, n, n] (W/m^2 at the
    ground), the plant fluxes gpp, npp, respiration, decomposition, litterfall, seeds_made, seeds_landed,
    germinated, seeds_died, resprout, n_fixed, potential, seedling_ok ([A, n, n], kg C/m^2 that day), the water
    fluxes rain, snowfall, melt, surplus (runoff above the bucket), evaporation, seepage ([A, n, n] kg/m^2), and per
    arena [A] float64: exchange_kg_c and exchange_mol (carbon fixed from the air, net, with any carbon sown since the
    last step), precip_kg, evap_kg (what left the float32 stores as vapour, with the cast's rounding),
    evap_physical_kg (the evaporation field itself), out_kg."""
    r = rules or PatchRules()
    br, cr = r.bio, r.climate
    BP = _bio_params(br)
    cm2 = geom.cell_m2
    land = ~geom.sea
    T = fine_temperature(geom, forcing)
    day_sw, bout_sw = light(geom, forcing)

    # ------------------------------------------------------------------ water (float64 working copies)
    soil, pond, snow = state.soil.double(), state.pond.double(), state.snow.double()
    T64 = T.double()
    precip = forcing.precip_kg_m2.double()[:, None, None] * land
    cold = T64 < T_FREEZE
    snowfall = torch.where(cold, precip, torch.zeros_like(precip))
    rain = precip - snowfall
    snow = snow + snowfall
    melt = torch.minimum(snow, cr.degree_day_kg_m2_k * (T64 - T_FREEZE).clamp_min(0))
    snow = snow - melt
    glacier = (snow - cr.glacier_kg_m2).clamp_min(0)
    snow = snow - glacier
    soil = soil + rain + melt
    surplus = (soil - cr.bucket_kg_m2).clamp_min(0)
    soil = soil - surplus
    pond, left = route(surplus, pond, geom)
    fill = torch.minimum(pond, (cr.bucket_kg_m2 - soil).clamp_min(0))
    pond, soil = pond - fill, soil + fill
    demand = evaporation_demand(T, forcing, cr) * land
    f_snow = (snow / cr.snow_mask_kg_m2).clamp(0, 1)
    open_water = 1 - cl.ice_fraction(T64, cr)
    e_pond = torch.minimum(pond, demand * open_water)
    beta = (soil / (cr.beta_share * cr.bucket_kg_m2)).clamp(0, 1)
    e_soil = torch.minimum(soil, (demand - e_pond).clamp_min(0) * beta * (1 - f_snow))
    pond, soil = pond - e_pond, soil - e_soil
    evap = e_pond + e_soil
    seep = torch.where(land, torch.minimum(pond, torch.full_like(pond, r.seep_kg_m2_day)), torch.zeros_like(pond))
    pond = pond - seep
    new_soil, new_pond, new_snow = soil.float(), pond.float(), snow.float()
    # evaporation is booked as what left the float32 stores: the cast's rounding (under half a float32 unit per
    # cell) goes with it, so the water ledger closes to float64; evap_physical_kg reports the field itself
    evap_booked = evap + (soil + pond + snow) - (new_soil.double() + new_pond.double() + new_snow.double())

    # ------------------------------------------------------------------ plants (version 2's equations, float32)
    dt_yr = 1.0 / DAYS_PER_YEAR
    a_w, s_sh = br.wood_allocation, r.seed_share
    par_mj = day_sw * forcing.par_fraction[:, None, None] * MJ_PER_W_DAY
    f_co2 = bs.co2_factor(forcing.p_co2_pa.double(), br).float()[:, None, None]
    f_w = (new_soil / cr.bucket_kg_m2).clamp(0, 1)
    f_t = bs.temperature_factor(T, br)
    f_flood = torch.exp(-new_pond / (RHO_W * r.flood_depth_m))
    nut = state.nutrient.clamp_min(0)
    f_n = nut / (nut + br.nutrient_half_kg_m2)
    plant0, wood0, seed0 = state.plant, state.wood, state.seed
    potential = torch.where(land, br.lue_g_c_mj * 1e-3 * par_mj * f_t * f_w * f_co2 * f_flood, torch.zeros_like(T))
    gpp = potential * fapar(plant0, r) * f_n
    q10 = br.q10 ** ((T - br.t_q10_k) / 10.0)
    share = max(1.0 - math.exp(-1.0 / br.acclimation_days), 1.0 / (state.day + 1))
    t_acclim = state.t_acclim + share * (T - state.t_acclim)
    maint = BP["r_n"] * br.q10 ** ((T - t_acclim) / 10.0) * dt_yr
    rm_leaf = maint * plant0 / br.cn_leaf
    rm_wood = maint * wood0 / br.cn_wood
    pay_leaf = torch.minimum(gpp, rm_leaf)
    pay_wood = torch.minimum(gpp - pay_leaf, rm_wood)
    surplus_c = gpp - pay_leaf - pay_wood
    n_per_c = (1 - s_sh) * ((1 - a_w) / br.cn_leaf + a_w / br.cn_wood) + s_sh / br.cn_leaf
    need = surplus_c * n_per_c
    phi = torch.where(need > nut, nut / need.clamp_min(1e-30), torch.ones_like(need))
    grow = surplus_c * phi
    gpp = pay_leaf + pay_wood + grow
    def_leaf, def_wood = rm_leaf - pay_leaf, rm_wood - pay_wood
    own_wood = torch.minimum(def_wood, wood0)
    own_leaf = torch.minimum(def_leaf, plant0)
    rest = (def_wood - own_wood) + (def_leaf - own_leaf)
    cross_wood = torch.minimum(rest, wood0 - own_wood)
    cross_leaf = torch.minimum(rest - cross_wood, plant0 - own_leaf)
    burn_wood, burn_leaf = own_wood + cross_wood, own_leaf + cross_leaf
    resp = pay_leaf + pay_wood + burn_leaf + burn_wood
    plant = plant0 + grow * (1 - s_sh) * (1 - a_w) - burn_leaf
    wood = wood0 + grow * (1 - s_sh) * a_w - burn_wood
    made = grow * s_sh
    nutrient = state.nutrient - grow * n_per_c + burn_leaf / br.cn_leaf + burn_wood / br.cn_wood
    # resprouting from the wood's reserves where the plant could photosynthesise
    dn = 1.0 / br.cn_leaf - 1.0 / br.cn_wood
    active = land & (par_mj > 0) & (f_t > 0) & (f_w > 0)
    short = (BP["leaf_wood"] * wood - plant).clamp_min(0)
    sprout = torch.where(active, short * (1 - math.exp(-1.0 / br.resprout_days)), torch.zeros_like(short))
    sprout = torch.minimum(torch.minimum(sprout, wood), nutrient.clamp_min(0) / dn)
    plant, wood = plant + sprout, wood - sprout
    nutrient = nutrient - sprout * dn
    # seeds: some stay, the rest land in the 8 neighbours (they carry their nitrogen)
    out = made * r.seed_export
    w_side = 1.0 / (4.0 + 4.0 / SQRT2)
    landed = _spread(out, w_side, w_side / SQRT2)
    seed = seed0 + (made - out) + landed
    # germination only where a seedling's net production is positive, never under standing water
    # (``PROVENANCE['seedling_test']``)
    seedling_ok = active & (new_pond <= r.germination_max_pond_kg_m2) & \
        (potential * f_n * leaf_area_per_kg_c(r) > maint / br.cn_leaf)
    germ = torch.where(seedling_ok, seed * (1 - math.exp(-1.0 / r.germination_days)), torch.zeros_like(seed))
    seed, plant = seed - germ, plant + germ
    died = seed * (1 - math.exp(-math.log(2.0) / r.seed_half_life_yr * dt_yr))
    seed = seed - died
    nutrient = nutrient + died / br.cn_leaf
    # turnover and decomposition (exact daily exponentials)
    lit_leaf = plant * (1 - math.exp(-br.leaf_turnover_yr * dt_yr))
    lit_wood = wood * (1 - math.exp(-br.wood_turnover_yr * dt_yr))
    plant, wood = plant - lit_leaf, wood - lit_wood
    nutrient = nutrient + lit_leaf / br.cn_leaf + lit_wood / br.cn_wood
    # biological fixation by the plants' symbionts: a share of the nitrogen the day's growth asks for, as far as the
    # mineral pool does not supply it (fixers pay off where N limits); no reference to any starting stock
    n_fixed = torch.where(land, r.bnf_share * need * (1 - f_n), torch.zeros_like(need))
    nutrient = nutrient + n_fixed
    decay = 1 - torch.exp(-br.decomposition_yr * q10 * torch.where(land, f_w, torch.ones_like(f_w)) * dt_yr)
    litter = state.litter + lit_leaf + lit_wood + died
    decomp = litter * decay
    litter = litter - decomp
    exchange = (gpp.double() - resp.double() - decomp.double())
    ex_kg = _arena_total(exchange, cm2)
    ex_report = ex_kg + state.c_pending

    new = replace(state, plant=plant, wood=wood, litter=litter, seed=seed, nutrient=nutrient, t_acclim=t_acclim,
                  soil=new_soil, pond=new_pond, snow=new_snow,
                  c_air=state.c_air + ex_kg, c_pending=torch.zeros_like(state.c_pending),
                  w_precip=state.w_precip + _arena_total(precip, cm2),
                  w_evap=state.w_evap + _arena_total(evap_booked, cm2),
                  w_out=state.w_out + left * cm2 + _arena_total(glacier + seep, cm2),
                  n_fixed=state.n_fixed + _arena_total(n_fixed, cm2), day=state.day + 1)
    diag = {"t_k": T, "day_sw": day_sw, "bout_sw": bout_sw, "gpp": gpp, "npp": gpp - resp, "respiration": resp,
            "decomposition": decomp, "litterfall": lit_leaf + lit_wood, "seeds_made": made, "seeds_landed": landed,
            "germinated": germ, "seeds_died": died, "resprout": sprout, "n_fixed": n_fixed, "potential": potential,
            "seedling_ok": seedling_ok, "rain": rain.float(), "snowfall": snowfall.float(), "melt": melt.float(),
            "surplus": surplus.float(), "evaporation": evap.float(), "seepage": seep.float(),
            "exchange_kg_c": ex_report, "exchange_mol": ex_report / M_C, "precip_kg": _arena_total(precip, cm2),
            "evap_kg": _arena_total(evap_booked, cm2), "evap_physical_kg": _arena_total(evap, cm2),
            "out_kg": left * cm2 + _arena_total(glacier + seep, cm2)}
    return new, diag


def run(state: PatchState, geom: PatchGeometry, forcing, days: int, rules: PatchRules | None = None, stats=None):
    """``days`` steps; ``forcing`` is a PatchForcing (every day the same) or a callable day -> PatchForcing.
    ``stats`` (optional) is called with (day index, state, diag) after each day. Returns (state, last diag)."""
    diag = None
    for d in range(days):
        f = forcing(state.day) if callable(forcing) else forcing
        state, diag = step(state, geom, f, rules)
        if stats is not None:
            stats(d, state, diag)
    return state, diag


def spin_up(state: PatchState, geom: PatchGeometry, forcing, days: int, gen, rules: PatchRules | None = None):
    """Vegetation spin-up: :func:`sow`, then ``days`` days of :func:`run`. The ledgers keep running (the sown
    carbon is booked as taken from the air). Returns (state, report) with the sown and final plant carbon [A]
    (kg C), the cumulative exchange and ``exchange_mol`` [A] (float64: the whole spin-up's net carbon taken from the
    air, sowing included, for ``gas_delta`` when the steps' own exchanges were not forwarded)."""
    r = rules or PatchRules()
    c_before = state.c_air.clone()
    state = sow(state, geom, gen, r)
    sown = state.c_air - c_before
    state, _ = run(state, geom, forcing, days, r)
    covered = ((state.plant > 1e-6) & ~geom.sea).float().mean((1, 2))
    taken = state.c_air - c_before
    return state, {"days": days, "sown_kg_c": sown.tolist(),
                   "plant_kg_c": _arena_total(state.plant, geom.cell_m2).tolist(),
                   "covered_share": covered.tolist(), "c_air_kg": taken.tolist(), "exchange_mol": taken / M_C}


# ============================================================================================ exchanges with bodies
def _flat(geom: PatchGeometry, a, cell, amount):
    dev = geom.device
    a = torch.as_tensor(a, device=dev).long().reshape(-1)
    cell = torch.as_tensor(cell, device=dev).long().reshape(-1)
    amt = torch.as_tensor(amount, dtype=torch.float64, device=dev).reshape(-1).clamp_min(0)
    amt = amt.expand(a.shape[0]) if amt.numel() == 1 else amt
    return a, cell, a * geom.n * geom.n + cell, amt


def _take(stock: torch.Tensor, idx, per_m2):
    """Remove per_m2 [M] (float64) at flat cells idx [M] from the float32 stock [A, n, n], never below zero; requests
    on one cell share what is there in proportion. Returns (new stock, removed per cell [G] float64, given per
    request [M] float64 per m^2) so that taker and giver book the same amount."""
    G = stock.numel()
    total = torch.zeros(G, dtype=torch.float64, device=stock.device).index_add_(0, idx, per_m2)
    have = stock.reshape(-1).double()
    left = torch.where(total > have, torch.zeros_like(have), have - total).float()
    removed = have - left.double()
    given = per_m2 / total[idx].clamp_min(1e-300) * removed[idx]
    return left.view_as(stock), removed, given


def _arena_sum(geom: PatchGeometry, per_cell):
    return per_cell.view(geom.A, -1).sum(-1) * geom.cell_m2


_POOLS = ("soft", "seed", "wood")


def _pool(pool: str, rules: PatchRules):
    br = rules.bio
    if pool not in _POOLS:
        raise ValueError("pool is 'soft', 'wood' or 'seed'")
    if pool == "wood":
        return "wood", materials.species_elements("wood")["C"], br.cn_wood
    return ("plant" if pool == "soft" else "seed"), br.plant_carbon_fraction, br.cn_leaf


def plant_nitrogen(kg_dry, pool: str = "soft", rules: PatchRules | None = None) -> torch.Tensor:
    """kg N in kg_dry of plant matter of a pool (``take_plant``'s pools): dry x carbon fraction / C:N."""
    _, cf, cn = _pool(pool, rules or PatchRules())
    return torch.as_tensor(kg_dry, dtype=torch.float64) * cf / cn


def take_plant(state: PatchState, geom: PatchGeometry, a, cell, kg_dry, pool: str = "soft",
               rules: PatchRules | None = None) -> torch.Tensor:
    """Plant matter bitten or gathered: kg dry requested per request (arena a [M], flat fine cell [M]) from the soft
    tissue (``soft``), the wood (``wood``) or the seed bank (``seed``). Returns kg dry given [M] (float64), never more
    than is there (requests on one cell share it). The carbon is booked in ``c_taken`` (dry matter 47 % C; wood at
    the carbon fraction of materials 'wood'); all its nitrogen leaves the cell with it (``n_removed``;
    :func:`plant_nitrogen` gives the amount): the body carries it and returns it where it excretes or dies
    (:func:`give_nitrogen`, :func:`add_litter`). The state's fields are rebound."""
    r = rules or PatchRules()
    attr, cf, cn = _pool(pool, r)
    a, cell, idx, kg = _flat(geom, a, cell, kg_dry)
    new, removed, given = _take(getattr(state, attr), idx, kg * cf / geom.cell_m2)
    setattr(state, attr, new)
    state.c_taken = state.c_taken + _arena_sum(geom, removed)
    state.n_removed = state.n_removed + _arena_sum(geom, removed / cn)
    return given * geom.cell_m2 / cf


def water_at(state: PatchState, geom: PatchGeometry, a, cell,
             rules: PatchRules | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """(fresh water a mouth can take there, kg/m^2 [M]: the pond and any soil water above the bucket; sea [M]
    bool). The sea flag is for physiology only (salt intake): it is a class label and must never reach the senses,
    which find salinity by taste or touch physics."""
    dev = geom.device
    a = torch.as_tensor(a, device=dev).long().reshape(-1)
    cell = torch.as_tensor(cell, device=dev).long().reshape(-1)
    idx = a * geom.n * geom.n + cell
    bucket = (rules or PatchRules()).climate.bucket_kg_m2
    fresh = state.pond.reshape(-1)[idx] + (state.soil.reshape(-1)[idx] - bucket).clamp_min(0)
    return fresh, geom.sea.reshape(-1)[idx]


def take_water(state: PatchState, geom: PatchGeometry, a, cell, kg,
               rules: PatchRules | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Water drunk: kg requested per request. On a sea cell it is sea water (salt; unlimited, booked in
    ``sea_water``, the global ocean's); elsewhere fresh water from the pond (and soil above the bucket), shared in
    proportion and never more than is there (booked in ``w_taken``). Returns (fresh kg [M], sea kg [M]) float64."""
    a, cell, idx, req = _flat(geom, a, cell, kg)
    sea = geom.sea.reshape(-1)[idx]
    fresh_req = torch.where(sea, torch.zeros_like(req), req)
    new, removed, given = _take(state.pond, idx, fresh_req / geom.cell_m2)
    state.pond = new
    taken = given * geom.cell_m2
    # what the pond could not give, from soil water above the bucket (requests on a cell share it in proportion;
    # what the float32 soil actually lost is what is given and booked)
    rest = (fresh_req - taken).clamp_min(0) / geom.cell_m2
    bucket = (rules or PatchRules()).climate.bucket_kg_m2
    soil_before = state.soil.reshape(-1).double()
    want = torch.zeros_like(soil_before).index_add_(0, idx, rest)
    excess = (soil_before - bucket).clamp_min(0)
    soil_after = (soil_before - torch.minimum(want, excess)).float()
    lost = soil_before - soil_after.double()
    state.soil = soil_after.view_as(state.soil)
    taken = taken + rest / want[idx].clamp_min(1e-300) * lost[idx] * geom.cell_m2
    state.w_taken = state.w_taken + _arena_sum(geom, removed) + _arena_sum(geom, lost)
    salt = torch.where(sea, req, torch.zeros_like(req))
    state.sea_water = state.sea_water + torch.zeros(geom.A, dtype=torch.float64, device=geom.device).index_add_(
        0, a, salt)
    return taken, salt


def give_water(state: PatchState, geom: PatchGeometry, a, cell, kg) -> None:
    """Water returned by bodies (urine, a body's water at death): kg per request into the soil of the cell (above
    the bucket it runs off on the next step), booked in ``w_given``; on a sea cell it joins the sea (``sea_water``)."""
    a, cell, idx, kg = _flat(geom, a, cell, kg)
    sea = geom.sea.reshape(-1)[idx]
    land_kg = torch.where(sea, torch.zeros_like(kg), kg)
    before = state.soil.reshape(-1).double()
    after = before.index_add(0, idx, land_kg / geom.cell_m2).float()
    state.soil = after.view_as(state.soil)
    state.w_given = state.w_given + _arena_sum(geom, after.double() - before)
    state.sea_water = state.sea_water - torch.zeros(geom.A, dtype=torch.float64, device=geom.device).index_add_(
        0, a, torch.where(sea, kg, torch.zeros_like(kg)))


def give_nitrogen(state: PatchState, geom: PatchGeometry, a, cell, kg_n) -> None:
    """Nitrogen returned by bodies (urine, faeces, a carcass): kg N per request into the cell's mineral pool (the
    biosphere mineralises litter nitrogen at once, as version 2), booked in ``n_added``."""
    a, cell, idx, kg = _flat(geom, a, cell, kg_n)
    before = state.nutrient.reshape(-1).double()
    after = before.index_add(0, idx, kg / geom.cell_m2).float()
    state.nutrient = after.view_as(state.nutrient)
    state.n_added = state.n_added + _arena_sum(geom, after.double() - before)


def add_litter(state: PatchState, geom: PatchGeometry, a, cell, kg_c, kg_n=None) -> None:
    """Organic matter returned to the ground (faeces, carcass remains, items): kg C per request into the cell's
    litter, booked in ``c_added``; its nitrogen ``kg_n`` (optional, per request) into the cell's mineral pool
    (:func:`give_nitrogen`)."""
    a, cell, idx, kg = _flat(geom, a, cell, kg_c)
    before = state.litter.reshape(-1).double()
    after = before.index_add(0, idx, kg / geom.cell_m2).float()
    state.litter = after.view_as(state.litter)
    state.c_added = state.c_added + _arena_sum(geom, after.double() - before)
    if kg_n is not None:
        give_nitrogen(state, geom, a, cell, kg_n)


def per_world(geom: PatchGeometry, x) -> torch.Tensor:
    """Sum of a per-arena quantity [A] over each world's patches -> [W] (float64)."""
    x = torch.as_tensor(x, dtype=torch.float64, device=geom.device)
    return torch.zeros(geom.W, *x.shape[1:], dtype=torch.float64, device=geom.device).index_add_(0, geom.world, x)


def gas_delta(geom: PatchGeometry, exchange_mol) -> torch.Tensor:
    """The column change [W, 4] (mol per m^2 of the planet; N2, O2, CO2, Ar) of the patches' net carbon fixation
    ``exchange_mol`` [A]: CO2 down and O2 up by the same moles over 4 pi R^2 (pass it to ``climate.add_gas``)."""
    mol = per_world(geom, exchange_mol)
    radius = torch.zeros(geom.W, dtype=torch.float64, device=geom.device).index_copy_(0, geom.world, geom.radius_m)
    col = mol / (4 * math.pi * radius ** 2)
    out = torch.zeros(geom.W, 4, dtype=torch.float64, device=geom.device)
    out[:, cl.I_CO2] = -col
    out[:, cl.I_O2] = col
    return out


# ============================================================================================ fine-grid queries
def wrap(d, L: float):
    """Periodic displacement: d wrapped into [-L/2, L/2)."""
    return torch.remainder(d + 0.5 * L, L) - 0.5 * L


def cell_index(geom: PatchGeometry, pos) -> torch.Tensor:
    """Flat fine cell (long [A, ...]) of positions pos [A, ..., 2] (metres, wrapped into the patch)."""
    n = geom.n
    i = torch.floor(torch.remainder(pos, geom.L) / geom.dx).long().clamp(0, n - 1)
    return i[..., 0] * n + i[..., 1]


def sample(geom: PatchGeometry, fld, pos) -> torch.Tensor:
    """The value of fld [A, n, n] in the cell under each position pos [A, ..., 2] -> [A, ...]."""
    c = cell_index(geom, pos)
    A = fld.shape[0]
    return fld.reshape(A, -1).gather(1, c.reshape(A, -1)).reshape(c.shape)


def _bilinear_flat(flat, n: int, dx: float, L: float, a, pos):
    """flat [A n n] (cell-centred) interpolated bilinearly and periodically at pos [..., 2] in arenas a [...]."""
    u = torch.remainder(pos, L) / dx - 0.5
    i0 = torch.floor(u)
    f = (u - i0).to(flat.dtype)
    i0 = i0.long()
    x0, y0 = torch.remainder(i0[..., 0], n), torch.remainder(i0[..., 1], n)
    x1, y1 = (x0 + 1) % n, (y0 + 1) % n
    off = a * (n * n)
    fx, fy = f[..., 0], f[..., 1]
    return (flat[off + x0 * n + y0] * (1 - fx) * (1 - fy) + flat[off + x1 * n + y0] * fx * (1 - fy)
            + flat[off + x0 * n + y1] * (1 - fx) * fy + flat[off + x1 * n + y1] * fx * fy)


def bilinear(geom: PatchGeometry, fld, pos) -> torch.Tensor:
    """fld [A, n, n] (cell-centred) interpolated bilinearly and periodically at pos [A, ..., 2] -> [A, ...]."""
    A = fld.shape[0]
    a = torch.arange(A, device=pos.device).reshape((A,) + (1,) * (pos.dim() - 2)).expand(pos.shape[:-1])
    return _bilinear_flat(fld.reshape(-1), geom.n, geom.dx, geom.L, a, pos)


def sagitta(d_m, radius_m):
    """Drop (m) of the planet's surface below the tangent plane at distance d_m: R - sqrt(R^2 - d^2), about
    d^2 / 2R (evaluated as d^2 / (R + sqrt(R^2 - d^2)), stable for d << R)."""
    d = torch.as_tensor(d_m, dtype=torch.float64)
    R = torch.as_tensor(radius_m, dtype=torch.float64, device=d.device)
    return d * d / (R + torch.sqrt((R * R - d * d).clamp_min(0)))


def bulge(d1_m, d2_m, radius_m):
    """Height of the planet's surface above the straight chord between two points at the surface, at distances d1
    and d2 from them: d1 d2 / 2R (the curvature term of line of sight)."""
    return torch.as_tensor(d1_m, dtype=torch.float64) * torch.as_tensor(d2_m, dtype=torch.float64) / (
        2 * torch.as_tensor(radius_m, dtype=torch.float64))


def line_of_sight(geom: PatchGeometry, p, q, h_p, h_q, samples: int | None = None,
                  rules: PatchRules | None = None) -> torch.Tensor:
    """Whether a straight line from p (eye h_p above the ground) to q (target h_q above it) clears the fine
    terrain: positions [A, M, 2] (metres; the shortest periodic displacement is used), heights [A, M] or numbers.
    The terrain is interpolated bilinearly at points spaced along the line, ``sight_samples_per_cell`` per fine cell of
    its length (at least 2; ``samples`` fixes the count for every line), raised by the planet's curvature
    d1 d2 / 2R. Lines are grouped by sample count and processed in chunks of at most ``sight_chunk`` samples.
    Returns bool [A, M]."""
    r = rules or PatchRules()
    dev = geom.device
    p = torch.as_tensor(p, dtype=torch.float32, device=dev)
    q = torch.as_tensor(q, dtype=torch.float32, device=dev)
    A, M = p.shape[0], p.shape[1]
    d = wrap(q - p, geom.L)
    dist = d.norm(dim=-1).double()                                   # [A, M]
    flat = geom.elev.reshape(-1)
    a_idx = torch.arange(A, device=dev)[:, None].expand(A, M)
    hp = torch.as_tensor(h_p, dtype=torch.float64, device=dev).expand(A, M)
    hq = torch.as_tensor(h_q, dtype=torch.float64, device=dev).expand(A, M)
    zp = (_bilinear_flat(flat, geom.n, geom.dx, geom.L, a_idx, p).double() + hp).reshape(-1)
    zq = (_bilinear_flat(flat, geom.n, geom.dx, geom.L, a_idx, q).double() + hq).reshape(-1)
    if samples is None:
        S = torch.ceil(r.sight_samples_per_cell * dist / geom.dx).long().clamp_min(2).reshape(-1)
    else:
        S = torch.full((A * M,), int(samples), dtype=torch.long, device=dev)
    pf, df, distf, af = p.reshape(-1, 2), d.reshape(-1, 2), dist.reshape(-1), a_idx.reshape(-1)
    R = geom.radius_m[af]
    out = torch.zeros(A * M, dtype=torch.bool, device=dev)
    order = torch.argsort(S, stable=True)
    S_sorted = S[order]
    total, i, budget = A * M, 0, max(int(r.sight_chunk), 1)
    while i < total:
        j = min(total, i + max(1, budget // int(S_sorted[i])))
        s_hi = int(S_sorted[j - 1])
        if s_hi * (j - i) > budget:
            j = i + max(1, budget // s_hi)
            s_hi = int(S_sorted[j - 1])
        idx = order[i:j]
        k = torch.arange(s_hi, device=dev, dtype=torch.float64)[None, :]
        s_line = S[idx, None].double()
        t64 = (k + 0.5) / s_line                                     # each line its own samples
        used = k < s_line
        pts = pf[idx, None, :] + t64.float()[..., None] * df[idx, None, :]
        zt = _bilinear_flat(flat, geom.n, geom.dx, geom.L, af[idx, None].expand(-1, s_hi), pts).double()
        ray = zp[idx, None] + t64 * (zq[idx] - zp[idx])[:, None]
        lift = t64 * (1 - t64) * distf[idx, None] ** 2 / (2 * R[idx, None])
        out[idx] = (((zt + lift) <= ray + r.sight_tolerance_m) | ~used).all(-1)
        i = j
    return out.reshape(A, M)


def neighbours(geom: PatchGeometry, pos, alive, K: int = 32, max_scan: int | None = None, chunk: int | None = None,
               gen=None):
    """Spatial hash neighbour search on the fine grid (``PROVENANCE['neighbour_scan']``). Bodies pos [A, N, 2]
    (metres) with alive [A, N] are bucketed by fine cell (one sort of the cell keys). Each alive body orders its 3 x 3
    cells by its own distance to each cell (its own cell first) and scans them in that order up to ``max_scan``
    candidates (4 K by default): cells inside the budget completely, the cell where the budget ends by a stratified
    uniform sample of its bodies (offset drawn on ``gen``, 0.5 when not given), so no direction or slot is favoured.
    It keeps the K nearest by periodic distance (merged ``chunk`` candidates at a time, 2 K by default). Returns (idx
    [A, N, K] long, -1 padded; offset [A, N, K, 2] the wrapped displacement to each candidate, 0 for padding; dist
    [A, N, K], inf for padding; count [A, N] long, the alive bodies in the 3 x 3 cells other than itself, so a caller
    can see when K or the budget bound). With the budget not reached the K nearest within the 3 x 3 cells are exact;
    every body within one cell size (L / n) of another is in its 3 x 3 cells. Dead bodies neither search nor are
    found."""
    pos = torch.as_tensor(pos, dtype=torch.float32, device=geom.device)
    alive = torch.as_tensor(alive, device=geom.device).bool()
    A, N = alive.shape
    n, L, dx = geom.n, geom.L, geom.dx
    Mb = 4 * K if max_scan is None else int(max_scan)
    C = max(1, 2 * K if chunk is None else int(chunk))
    dev = geom.device
    wrapped = torch.remainder(pos, L)
    c = torch.floor(wrapped / dx).long().clamp(0, n - 1)                              # [A, N, 2]
    frac = (wrapped - c.float() * dx).clamp(0, dx)                                     # position inside the cell
    ar = torch.arange(A, device=dev)[:, None]
    sentinel = A * n * n
    key = torch.where(alive, ar * (n * n) + c[..., 0] * n + c[..., 1], torch.full_like(c[..., 0], sentinel))
    sk, order = torch.sort(key.reshape(-1), stable=True)
    off = torch.tensor(BLOCK9, device=dev)                                              # [9, 2]
    gap = []
    for k in range(2):
        o = off[:, k].float()
        fk = frac[..., k, None]
        gap.append(torch.where(o > 0, dx - fk, torch.where(o < 0, fk, torch.zeros_like(fk))))
    d2 = gap[0] ** 2 + gap[1] ** 2                                                     # [A, N, 9]
    nine = torch.argsort(d2, dim=-1, stable=True)
    offo = off[nine]                                                                   # [A, N, 9, 2]
    qx = torch.remainder(c[..., None, 0] + offo[..., 0], n)
    qy = torch.remainder(c[..., None, 1] + offo[..., 1], n)
    qkey = ar[..., None] * (n * n) + qx * n + qy                                        # [A, N, 9]
    start = torch.searchsorted(sk, qkey.reshape(-1)).reshape(A, N, 9)
    end = torch.searchsorted(sk, qkey.reshape(-1), right=True).reshape(A, N, 9)
    cnt = torch.where(alive[..., None], end - start, torch.zeros_like(start))
    cum = torch.cumsum(cnt, -1)
    total = cum[..., -1]
    before = cum - cnt
    if gen is None:
        u = torch.full((A, N), 0.5, device=dev)
    else:
        u = torch.rand(A, N, generator=gen, device=gen.device).to(dev)
    own = (ar * N + torch.arange(N, device=dev)[None, :])[..., None]
    best_d = torch.full((A, N, K), float("inf"), device=dev)
    best_i = torch.full((A, N, K), -1, dtype=torch.long, device=dev)
    best_o = torch.zeros(A, N, K, 2, device=dev)
    scan = torch.minimum(total, torch.full_like(total, Mb))
    for s0 in range(0, Mb, C):
        slot = (torch.arange(s0, min(s0 + C, Mb), device=dev)).expand(A, N, -1).contiguous()
        b = torch.searchsorted(cum.contiguous(), slot, right=True).clamp_max(8)            # bucket of each slot
        bef = before.gather(-1, b)
        cn = cnt.gather(-1, b)
        inside = cum.gather(-1, b) <= Mb
        avail = (Mb - bef).clamp_min(1)
        strat = torch.floor(((slot - bef).double() + u[..., None].double()) * cn.double() / avail.double()).long()
        pick = torch.where(inside, slot - bef, torch.minimum(strat, (cn - 1).clamp_min(0)))
        valid = slot < scan[..., None]
        p = torch.where(valid, start.gather(-1, b) + pick, torch.zeros_like(slot)).clamp(0, sk.numel() - 1)
        j = order[p]                                                                   # flat a N + i
        valid = valid & (j != own)
        cand = torch.where(valid, j % N, torch.full_like(j, -1))
        other = pos.reshape(-1, 2)[j]
        d = wrap(other - pos[..., None, :], L)
        dist = torch.where(valid, d.norm(dim=-1), torch.full_like(d[..., 0], float("inf")))
        all_d = torch.cat((best_d, dist), -1)
        all_i = torch.cat((best_i, cand), -1)
        all_o = torch.cat((best_o, d), -2)
        srt, o = torch.sort(all_d, dim=-1, stable=True)
        o = o[..., :K]
        best_d = srt[..., :K]
        best_i = all_i.gather(-1, o)
        best_o = all_o.gather(-2, o[..., None].expand(*o.shape, 2))
    ok = torch.isfinite(best_d)
    best_i = torch.where(ok, best_i, torch.full_like(best_i, -1))
    best_o = torch.where(ok[..., None], best_o, torch.zeros_like(best_o))
    count = (total - alive.long()).clamp_min(0)
    return best_i, best_o, best_d, count


# ============================================================================================ smells
@dataclass
class Volatiles:
    """Near-surface concentrations [A, n, n] (kg/m^3, float32): plant volatiles, sulfur volatiles of rotting animal
    matter, smoke (CO) and CO2 from point sources (bodies, fires)."""
    plant: torch.Tensor
    carcass: torch.Tensor
    smoke: torch.Tensor
    co2: torch.Tensor

    def state_dict(self) -> dict:
        return {f.name: getattr(self, f.name).clone() for f in fields(self)}

    @classmethod
    def from_state(cls, d: dict) -> "Volatiles":
        z = next(iter(d.values()))
        return cls(**{f.name: (d[f.name] if f.name in d else torch.zeros_like(z)).clone() for f in fields(cls)})


def init_volatiles(geom: PatchGeometry) -> Volatiles:
    z = torch.zeros_like(geom.elev)
    return Volatiles(plant=z.clone(), carcass=z.clone(), smoke=z.clone(), co2=z.clone())


def deposit(geom: PatchGeometry, a, pos, amount) -> torch.Tensor:
    """Amounts at points (arena a [M], pos [M, 2] metres, amount [M]) binned into their fine cells, per m^2
    ([A, n, n] float32): rotting animal matter (carcass flesh, faeces; kg) for ``volatile_step``'s carcass source,
    or an emission rate (kg/s: a fire's smoke, a body's or a fire's CO2)."""
    dev = geom.device
    a = torch.as_tensor(a, device=dev).long().reshape(-1)
    pos = torch.as_tensor(pos, dtype=torch.float32, device=dev).reshape(-1, 2)
    amt = torch.as_tensor(amount, dtype=torch.float64, device=dev).reshape(-1)
    n = geom.n
    i = torch.floor(torch.remainder(pos, geom.L) / geom.dx).long().clamp(0, n - 1)
    idx = a * n * n + i[:, 0] * n + i[:, 1]
    out = torch.zeros(geom.A * n * n, dtype=torch.float64, device=dev).index_add_(0, idx, amt)
    return (out / geom.cell_m2).float().view(geom.A, n, n)


def smoke_emission(fuel_burnt_kg_s, rules: PatchRules | None = None):
    """Smoke (kg/s) of a fire burning fuel_burnt_kg_s of fuel (``smoke_kg_per_kg_fuel``)."""
    r = rules or PatchRules()
    return torch.as_tensor(fuel_burnt_kg_s, dtype=torch.float64) * r.smoke_kg_per_kg_fuel


def fire_co2_emission(fuel_burnt_kg_s, rules: PatchRules | None = None):
    """CO2 (kg/s) of a fire burning fuel_burnt_kg_s of fuel (``co2_kg_per_kg_fuel``)."""
    r = rules or PatchRules()
    return torch.as_tensor(fuel_burnt_kg_s, dtype=torch.float64) * r.co2_kg_per_kg_fuel


def metabolic_co2_emission(metabolic_w, rules: PatchRules | None = None):
    """CO2 (kg/s) breathed out by a body of aerobic metabolic power metabolic_w (W): power / oxycaloric equivalent
    x respiratory quotient x M_CO2 (chemistry)."""
    r = rules or PatchRules()
    return torch.as_tensor(metabolic_w, dtype=torch.float64).clamp_min(0) / r.oxycaloric_j_mol_o2 \
        * r.respiratory_quotient * M_CO2


def _screened(c, e_kg_m2_s, dt_s: float, life_s: float, geom: PatchGeometry, r: PatchRules):
    """Exact solution over dt_s of dc/dt = K lap c - c / tau + E / h on the periodic grid (Fourier space, the
    discrete Laplacian's eigenvalues); 1 / tau = 1 / life + 1 / ventilation."""
    n, dx = geom.n, geom.dx
    kx = torch.arange(n, dtype=torch.float64, device=c.device)
    ky = torch.arange(n // 2 + 1, dtype=torch.float64, device=c.device)
    lap = (4.0 / dx ** 2) * (torch.sin(math.pi * kx / n)[:, None] ** 2 + torch.sin(math.pi * ky / n)[None, :] ** 2)
    loss = (0.0 if not math.isfinite(life_s) else 1.0 / life_s) + \
        (0.0 if not math.isfinite(r.ventilation_s) else 1.0 / r.ventilation_s)
    lam = loss + r.eddy_diffusivity_m2_s * lap
    decay = torch.exp(-lam * dt_s)
    gain = torch.where(lam > 0, (1 - decay) / lam.clamp_min(1e-300), torch.full_like(lam, float(dt_s)))
    ch = torch.fft.rfft2(c.double())
    eh = torch.fft.rfft2(e_kg_m2_s.double() / r.smell_layer_m)
    return torch.fft.irfft2(ch * decay + eh * gain, s=(n, n)).clamp_min(0).float()


def volatile_step(vol: Volatiles, geom: PatchGeometry, dt_s: float, plant_c=None, carcass_kg_m2=None,
                  smoke_kg_m2_s=None, co2_kg_m2_s=None, rules: PatchRules | None = None) -> Volatiles:
    """Advance the smell fields by dt_s seconds. Sources: plant volatiles from the soft tissue plant_c [A, n, n]
    (kg C/m^2; ``plant_emission_kg_kg_s`` per kg dry), sulfur volatiles from rotting animal matter carcass_kg_m2
    [A, n, n] (carcass flesh and faeces, ``flesh_decay_day`` x ``carcass_volatile_share``), smoke from smoke_kg_m2_s
    and CO2 from co2_kg_m2_s [A, n, n] (``deposit`` of ``smoke_emission``, ``fire_co2_emission`` and
    ``metabolic_co2_emission``; the field is the excess over the background air, inert within hours). Missing
    sources emit nothing. Returns new Volatiles."""
    r = rules or PatchRules()
    z = torch.zeros_like(geom.elev)
    e_plant = z if plant_c is None else plant_c.clamp_min(0) / r.bio.plant_carbon_fraction * r.plant_emission_kg_kg_s
    e_carc = z if carcass_kg_m2 is None else carcass_kg_m2.clamp_min(0) * (
        r.flesh_decay_day / DAY_S * r.carcass_volatile_share)
    e_smoke = z if smoke_kg_m2_s is None else smoke_kg_m2_s.clamp_min(0)
    e_co2 = z if co2_kg_m2_s is None else co2_kg_m2_s.clamp_min(0)
    return Volatiles(plant=_screened(vol.plant, e_plant, dt_s, r.plant_volatile_life_s, geom, r),
                     carcass=_screened(vol.carcass, e_carc, dt_s, r.carcass_volatile_life_s, geom, r),
                     smoke=_screened(vol.smoke, e_smoke, dt_s, float("inf"), geom, r),
                     co2=_screened(vol.co2, e_co2, dt_s, float("inf"), geom, r))


def gradient(geom: PatchGeometry, fld) -> tuple[torch.Tensor, torch.Tensor]:
    """Centred periodic gradient (d/dx east, d/dy north) of a field [A, n, n] per metre (for a smell's gradient
    along a heading)."""
    gx = (torch.roll(fld, -1, 1) - torch.roll(fld, 1, 1)) / (2 * geom.dx)
    gy = (torch.roll(fld, -1, 2) - torch.roll(fld, 1, 2)) / (2 * geom.dx)
    return gx, gy


# ======================================================================================= what the ground looks like
def ground_reflectance(state: PatchState, geom: PatchGeometry, rules: PatchRules | None = None,
                       t_k=None) -> torch.Tensor:
    """Visible reflectance [A, n, n] of each fine cell's surface: bare ground (``geom.bare_vis``) under its litter
    cover (1 - exp(-A_m dry litter)), under the plant canopy's cover (its fAPAR), then open water (the pond, full at
    ``pond_mask_kg_m2``; ice by the climate's ice ramp at the fine temperature ``t_k`` when given) and snow (full at
    the climate's snow mask); sea cells are water. Values from ``optics``."""
    r = rules or PatchRules()
    bare = geom.bare_vis[:, None, None]
    lit = 1 - torch.exp(-r.litter_cover_m2_kg * state.litter.clamp_min(0) / r.bio.plant_carbon_fraction)
    rho = bare + (optics.vis("ground:litter") - bare) * lit
    cover = fapar(state.plant, r)
    rho = rho + (optics.vis("ground:plant") - rho) * cover
    water = optics.vis("ground:water")
    if t_k is None:
        surface = torch.full_like(rho, water)
    else:
        ice = cl.ice_fraction(torch.as_tensor(t_k, dtype=torch.float32, device=rho.device), r.climate)
        surface = water + (optics.vis("ground:ice") - water) * ice
    f_pond = (state.pond / r.pond_mask_kg_m2).clamp(0, 1)
    rho = rho + (surface - rho) * f_pond
    f_snow = (state.snow / r.climate.snow_mask_kg_m2).clamp(0, 1)
    rho = rho + (optics.vis("ground:snow") - rho) * f_snow
    return torch.where(geom.sea, surface, rho)
