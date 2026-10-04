"""Biosphere of the habitat globe (PLANET-SPEC section 2.7): land plants, litter, nutrients and ocean
phytoplankton, coupled to the climate's gas column.

State (:class:`BioState`, float32 [W, C], kg per m^2 of cell):

* ``plant_c`` living soft tissue (leaves, fine roots, herbs; the forage) and ``wood_c`` living wood
  (the share that can be collected, with the plant's carbohydrate reserves), both kg C/m^2;
  ``litter_c`` dead organic carbon;
* ``nutrient`` the plant-available mineral nitrogen (kg N/m^2, the N/P proxy);
* ``phyto_c`` ocean phytoplankton (kg C/m^2 of ocean cell);
* ``t_acclim`` the running-mean temperature the plants' respiration is acclimated to (K) and ``day``;
* ledger counters (float64 [W], global-mean kg C/m^2) and the cell areas in m^2 (for the scatters).

One step is one day, after ``climate.step`` (it uses that day's surface shortwave, the new T, soil and
the gas column):

* GPP = LUE x fAPAR x PAR x f_T x f_W x f_CO2 x f_N, with LUE 1.8 g C per MJ of absorbed PAR, PAR the
  surface shortwave x the star's ``par_fraction``, fAPAR = 1 - exp(-plant_c / 0.5), f_T a beta-function
  bell zero at 268 and 318 K with its peak at 298 K, f_W = soil / 150, f_CO2 = (p / (p + 30 Pa)) /
  (28 / 58), f_N = N / (N + K_N).
* NPP = GPP - R_m (spec 2.7). Maintenance respiration per pool, R_leaf = r_N plant_c / CN_leaf q and
  R_wood = r_N wood_c / CN_wood q, q = 2^((T - T_acclim) / 10) (respiration per unit tissue nitrogen,
  Ryan 1991), with r_N derived so that a plant in steady state has NPP / GPP = 0.47 (Waring et al.
  1998) at the temperature it is acclimated to (its one-year running mean: Waring's ratio holds across
  climates; Atkin & Tjoelker 2003). GPP pays the soft tissue's maintenance first, then the wood's; the
  rest grows, 61 % to soft tissue and 39 % to wood (Malhi et al. 2011), taking its nitrogen from the
  mineral pool (growth stops when it is empty). Unpaid maintenance burns each pool's own tissue (then
  the other pool's, wood first), so a grazed plant does not burn its last leaves for its wood.
* resprouting: where the plant could photosynthesise (light, f_T > 0, f_W > 0) and its soft tissue is
  below its steady-state ratio to the wood, the shortfall is rebuilt from the wood's reserves with a
  30-day e-folding (refoliation after defoliation; its extra nitrogen from the mineral pool).
* seed rain: soft tissue never falls below 1 g C/m^2 on land that gets light during the year (taken
  from the air's CO2 like any fixation, its nitrogen from the mineral pool), so a stripped cell regrows.
* turnover to litter: soft tissue 1 / yr, wood 0.05 / yr; their nitrogen returns to the mineral pool
  at litterfall (instant mineralisation, new_rule), so nutrient + plant_c / CN_leaf + wood_c / CN_wood
  is constant in each cell except for what harvest and wood collection take away (an eaten plant's N goes
  back to the cell at 85 %, the eaters' dung and urine); nitrogen fixation refills such a deficit (at most
  0.39 g N/m^2/yr, falling to 0 at the cell's starting mineral N).
* decomposition k = 0.3 / yr x 2^((T - 288) / 10) x soil / 150 (Q10 = 2) to CO2 (on the ocean, where
  only corpses add litter, without the soil factor).
* ocean: phytoplankton production e_o x PAR x 1.066^(min(T, 293 K) - 291 K) (Eppley 1972, capped at the
  20 C optimum of the VGPM) on open water, e_o calibrated to Earth's ocean NPP (Field et al. 1998), so
  Earth's mean nutrient limitation is inside it (new_rule: no ocean nutrient field); the stock turns over
  in 7 days, a 0.33 % share is buried in sediments (the long-term O2 source), the rest is remineralised
  to CO2. It is not food for land individuals.
* gas exchange: each mol of C fixed takes one mol of CO2 from the climate's ``gas`` and gives one mol
  of O2; respiration, decomposition and remineralisation reverse it.

``spin_up`` grows the vegetation with the chain's air held and puts the slow pools (wood, litter) at their
steady state (wood has a 20-year turnover, so a short spin-up would leave it drawing CO2 from the air for
decades): see :func:`spin_up`.

Ledgers (float64 [W], global means): :func:`carbon_ledger` (kg C/m^2: CO2 column + organic pools +
what left - what came in) and :func:`oxygen_ledger` (mol/m^2: O2 column - organic carbon in moles -
organic carbon that left + organic carbon that came in - O2 added by other modules). Each returns the error against
the baseline set by :func:`init_state` or :func:`reset_ledgers`.

States are values: ``step`` returns a new state, and the scatters (``harvest``, ``collect_wood``,
``add_litter``) rebind the state's fields to new tensors, so an earlier state or ``state_dict`` (copies)
never changes afterwards. The water used by photosynthesis (one H2O per C, about 0.1 % of the
precipitation) is not booked in the water ledger. Every set-up number carries a provenance entry in
``PROVENANCE``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace

import torch

from . import climate as cl
from . import constants as K
from . import materials

DAYS_PER_YEAR = K.YEAR_S / K.DAY_S
MJ_PER_W_DAY = K.DAY_S / 1e6            # MJ per (W/m^2 x day)
M_C = K.element_mass("C")
FOUR_PI = 4.0 * math.pi


@dataclass(frozen=True)
class BioRules:
    """Modelling choices and reference values of the biosphere (sources in ``PROVENANCE``)."""
    lue_g_c_mj: float = 1.8
    fapar_kg_c: float = 0.5
    t_min_k: float = 268.0
    t_opt_k: float = 298.0
    t_max_k: float = 318.0
    co2_half_pa: float = 30.0
    co2_ref_pa: float = 28.0
    leaf_turnover_yr: float = 1.0
    wood_turnover_yr: float = 0.05
    decomposition_yr: float = 0.3
    q10: float = 2.0
    t_q10_k: float = 288.0
    acclimation_days: float = 365.25
    wood_allocation: float = 0.39
    npp_gpp: float = 0.47
    cn_leaf: float = 29.0
    cn_wood: float = 330.0
    nutrient_start_kg_m2: float = 0.03
    nutrient_half_kg_m2: float = 0.004
    n_fixation_kg_m2_yr: float = 3.9e-4
    seed_c_kg_m2: float = 0.05
    seed_floor_kg_m2: float = 1e-3
    excreted_n_share: float = 0.85
    resprout_days: float = 30.0
    plant_carbon_fraction: float = 0.47
    ocean_npp_pg_yr: float = 48.5
    earth_ocean_area_m2: float = 3.618e14
    earth_surface_sw_w_m2: float = 188.0
    eppley_base: float = 1.066
    eppley_t_k: float = 291.15
    eppley_t_max_k: float = 293.15
    phyto_turnover_days: float = 7.0
    burial_share: float = 0.0033


PROVENANCE = {
    "lue_g_c_mj": ("reference", "light-use efficiency 1.8 g C per MJ of absorbed PAR, gross (Monteith 1977, Phil. "
                   "Trans. R. Soc. B 281, 277; field values 1-3 g C/MJ APAR, Gower, Kucharik & Norman 1999, Remote "
                   "Sens. Environ. 70, 29)"),
    "fapar_kg_c": ("new_rule", "fAPAR = 1 - exp(-plant_c / 0.5 kg C/m^2) (Beer's law on the soft tissue, spec)"),
    "t_bell": ("reference+new_rule", "f_T = ((T_max - T)/(T_max - T_opt)) ((T - T_min)/(T_opt - T_min))^((T_opt - "
               "T_min)/(T_max - T_opt)), the beta function of Yan & Hunt 1999 (Ann. Bot. 84, 607), with the spec's "
               "268 / 298 / 318 K"),
    "co2": ("new_rule", "f_CO2 = (p / (p + 30 Pa)) / (28 / 58): Michaelis-Menten, 1 at Earth's pre-industrial "
            "28 Pa (spec); it is about 2.07 at the 1e4 Pa cap of formation"),
    "leaf_turnover_yr": ("reference", "soft tissue 1 / yr (deciduous leaves and fine roots; spec)"),
    "wood_turnover_yr": ("reference", "wood 0.05 / yr (20 yr residence; spec)"),
    "decomposition_yr": ("reference+new_rule", "litter 0.3 / yr at 288 K x Q10^((T - 288)/10) x soil / 150 (spec); "
                         "new_rule: litter on ocean cells (corpses) decays without the soil factor"),
    "q10": ("reference", "Q10 = 2 for decomposition and maintenance respiration (Lloyd & Taylor 1994, Funct. Ecol. "
            "8, 315: about 2 near 288 K)"),
    "acclimation_days": ("reference+new_rule", "NPP / GPP is nearly the same (0.47) in forests of very different "
                         "climates (Waring, Landsberg & Williams 1998), because plant respiration acclimates to the "
                         "growth temperature (Atkin & Tjoelker 2003, Trends Plant Sci. 8, 343; Gifford 2003, Funct. "
                         "Plant Biol. 30, 171); new_rule: maintenance is r_N at the cell's running-mean temperature "
                         "(one-year e-folding, a cumulative mean over the first year), with the short-term Q10 = 2 "
                         "around it (so the seasons keep their effect; decomposition is not acclimated)"),
    "wood_allocation": ("reference+new_rule", "39 % of NPP to wood, the rest (canopy 34 % + fine roots 27 %) to the "
                        "soft tissue: tropical forest allocation of Malhi, Doughty & Galbraith 2011 (Phil. Trans. R. "
                        "Soc. B 366, 3225: canopy 34 +/- 6, wood 39 +/- 10, fine roots 27 +/- 11 %); new_rule: one "
                        "allocation for every plant"),
    "npp_gpp": ("reference", "NPP / GPP = 0.47 (Waring, Landsberg & Williams 1998, Tree Physiol. 18, 129)"),
    "r_n": ("derived", "maintenance respiration per kg tissue N per yr at 288 K such that a steady-state plant has "
            "NPP / GPP = npp_gpp: r_N = (1 / npp_gpp - 1) / (a_leaf / (CN_leaf turnover_leaf) + a_wood / (CN_wood "
            "turnover_wood)); respiration proportional to tissue N (Ryan 1991, Ecol. Appl. 1, 157)"),
    "maintenance_order": ("new_rule", "each pool pays its own maintenance: GPP pays the soft tissue's first, then the "
                          "wood's; what is unpaid burns the pool's own tissue, then the other pool's (wood first, the "
                          "plant's reserves)"),
    "cn": ("reference", "C:N 29 for soft tissue, 330 for wood (sapwood; Sitch et al. 2003, Glob. Change Biol. 9, "
           "161)"),
    "nutrient_start_kg_m2": ("new_rule", "30 g N/m^2 plant-available mineral N on land at the start (vegetation N "
                             "stocks are 20-60 g N/m^2)"),
    "nutrient_half_kg_m2": ("new_rule", "f_N = N / (N + 4 g N/m^2)"),
    "n_fixation_kg_m2_yr": ("reference+new_rule", "biological N fixation 0.39 g N/m^2/yr: pre-industrial terrestrial "
                            "BNF of about 58 Tg N/yr (Vitousek, Menge, Reed & Cleveland 2013, Phil. Trans. R. Soc. B "
                            "368, 20130119) over 1.49e14 m^2 of land; new_rule: it falls linearly to 0 as the cell's "
                            "total N (mineral + tissue) reaches nutrient_start_kg_m2 (fixation is suppressed where N "
                            "is plentiful, and the model has no N losses to balance it), so it only refills what "
                            "harvest and wood collection took"),
    "excreted_n_share": ("reference+new_rule", "grazing animals return 75-95 % of the nitrogen they eat in dung and "
                         "urine (Haynes & Williams 1993, Adv. Agron. 49, 119); new_rule: 85 % of the harvested soft "
                         "tissue's N goes back to the cell's mineral pool at once (creatures carry no N ledger)"),
    "seed_c_kg_m2": ("new_rule", "0.05 kg C/m^2 of soft tissue seeded on land cells that get light during the year; "
                     "cells without any light (a locked planet's night side) start and stay bare"),
    "seed_floor_kg_m2": ("new_rule", "seed rain refills soft tissue toward 1 g C/m^2 (of the order of a year's seed "
                         "rain, a few g of dry matter per m^2) at that annual rate (time constant 1 yr), only on land "
                         "where a seedling could grow that day (light, f_T > 0, f_W > 0); the carbon comes from the "
                         "air's CO2 and is booked in the gas exchange, its nitrogen from the mineral pool (none when it "
                         "is empty). Fixed on 3 Oct 2026: it had refilled the floor in full every day everywhere lit, "
                         "a free food supply on barren planets"),
    "resprout_days": ("reference+new_rule", "plants refoliate from stored carbohydrates within weeks after "
                      "defoliation (Kozlowski 1992, Bot. Rev. 58, 107; the mobile carbon of trees could rebuild the "
                      "canopy several times, Hoch, Richter & Koerner 2003, Plant Cell Environ. 26, 1067); new_rule: "
                      "where the plant could photosynthesise, soft tissue below its steady-state ratio to the wood "
                      "((1 - a_w) / turnover_leaf) / (a_w / turnover_wood) is rebuilt from the wood with a 30-day "
                      "e-folding"),
    "plant_carbon_fraction": ("reference", "carbon fraction of dry plant matter 0.47 (IPCC 2006 Guidelines, vol. 4, "
                              "ch. 4, table 4.3)"),
    "wood_carbon_fraction": ("derived", "carbon mass fraction of materials 'wood' (C6H9O4 with 1 % ash), so the item "
                             "ledger books the same carbon"),
    "ocean_npp": ("reference+new_rule", "Earth's ocean NPP 48.5 Pg C/yr (Field et al. 1998, Science 281, 237) over "
                  "3.618e14 m^2 of ocean, surface shortwave 188 W/m^2 (Stephens et al. 2012, Nature Geosci. 5, 691) x "
                  "the Sun's PAR share (formation.par_fraction(5772 K)); derived: e_o g C per MJ PAR at 291 K; "
                  "new_rule: Earth's mean nutrient limitation of the ocean applies everywhere (no nutrient field)"),
    "eppley": ("reference+new_rule", "phytoplankton growth x 1.066^(T - T_ref) (Eppley 1972, Fish. Bull. 70, 1063), "
               "T_ref 291 K (Earth's mean sea surface, about 18 C); none under sea ice. Eppley's curve is the maximum "
               "growth rate; production in warm, stratified water is nutrient-limited and the optimum "
               "photosynthetic rate of the VGPM peaks near 20 C (Behrenfeld & Falkowski 1997, Limnol. Oceanogr. 42, "
               "1), so new_rule: the factor is held at its 293.15 K value above it (its further fall above about "
               "28 C is not modelled)"),
    "phyto_turnover_days": ("reference", "7 d: about 1 Pg C of phytoplankton for 48.5 Pg C/yr (Field et al. 1998)"),
    "burial_share": ("reference", "0.33 % of ocean NPP buried: 0.16 Pg C/yr organic carbon burial (Hedges & Keil "
                     "1995, Mar. Chem. 49, 81) of 48.5"),
    "nutrient_cycle": ("new_rule", "tissue N returns to the mineral pool at litterfall (instant mineralisation); "
                       "litter added by other modules (corpses) brings carbon only (bodies carry no N ledger)"),
    "spin_up": ("new_rule", "numerics: the spin-up holds the chain's air and, at the end of each of its segments, puts "
                "the wood (and before the last one the soft tissue) at the steady state of the segment's last year "
                "and the litter at that year's periodic steady state (see spin_up)"),
}
# every BioRules field has its entry (the shared notes under the field names)
for _names, _key in ((("t_min_k", "t_opt_k", "t_max_k"), "t_bell"), (("co2_half_pa", "co2_ref_pa"), "co2"),
                     (("t_q10_k",), "q10"), (("cn_leaf", "cn_wood"), "cn"),
                     (("ocean_npp_pg_yr", "earth_ocean_area_m2", "earth_surface_sw_w_m2"), "ocean_npp"),
                     (("eppley_base", "eppley_t_k", "eppley_t_max_k"), "eppley")):
    for _name in _names:
        PROVENANCE[_name] = PROVENANCE[_key]


def make_params(P, rules: BioRules | None = None) -> dict:
    """Biosphere parameters from the climate's parameters ``P`` (par_fraction, land): the rules, the derived
    respiration coefficient ``r_n`` (kg C per kg N per yr at 288 K) and ocean efficiency ``e_ocean`` (kg C per MJ
    PAR), the steady-state soft tissue to wood ratio ``leaf_wood``, and the carbon fraction of collected wood."""
    from .formation import par_fraction       # set-up only (CPU)
    r = rules or BioRules()
    a_w = r.wood_allocation
    respiring = (1 - a_w) / (r.cn_leaf * r.leaf_turnover_yr) + a_w / (r.cn_wood * r.wood_turnover_yr)
    par_sun = par_fraction(K.T_SUN)
    mj_yr = r.earth_surface_sw_w_m2 * par_sun * K.YEAR_S / 1e6       # MJ of PAR per m^2 per yr
    e_ocean = r.ocean_npp_pg_yr * 1e12 / r.earth_ocean_area_m2 / mj_yr  # kg C per MJ PAR
    leaf_wood = ((1 - a_w) / r.leaf_turnover_yr) / (a_w / r.wood_turnover_yr)
    return {"rules": r, "r_n": (1 / r.npp_gpp - 1) / respiring, "e_ocean": e_ocean, "leaf_wood": leaf_wood,
            "wood_carbon_fraction": materials.species_elements("wood")["C"],
            "par_fraction": P["par_fraction"], "land": P["land"], "W": P["W"], "par_sun": par_sun}


# ------------------------------------------------------------------------------------------- state
@dataclass
class BioState:
    """Biosphere state of W worlds (see the module docstring)."""
    plant_c: torch.Tensor           # [W, C] kg C/m^2, living soft tissue
    wood_c: torch.Tensor            # [W, C] kg C/m^2, living wood
    litter_c: torch.Tensor          # [W, C] kg C/m^2
    nutrient: torch.Tensor          # [W, C] kg N/m^2, mineral
    phyto_c: torch.Tensor           # [W, C] kg C/m^2 (ocean cells)
    cell_m2: torch.Tensor           # [C] float64, cell areas on the habitat globe
    area64: torch.Tensor            # [C] float64, steradians
    carbon0: torch.Tensor           # [W] float64 ledger baselines
    oxygen0: torch.Tensor
    c_harvested: torch.Tensor       # [W] float64 global-mean kg C/m^2, cumulative
    c_wood_collected: torch.Tensor
    c_buried: torch.Tensor
    c_litter_added: torch.Tensor
    t_acclim: torch.Tensor | None = None   # [W, C] K, running-mean temperature the respiration is acclimated to
                                    # (None: none yet, the first step starts it)
    day: int = 0                    # days stepped (the running mean starts as a cumulative mean)

    def clone(self) -> "BioState":
        return replace(self, **{f.name: _copy(getattr(self, f.name)) for f in fields(self)})

    def state_dict(self) -> dict:
        """The fields as a dict of copies (later changes of the state do not reach it)."""
        return {f.name: _copy(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_state(cls, d: dict) -> "BioState":
        return cls(**{k: _copy(v) for k, v in d.items()})


def _copy(v):
    return v.clone() if isinstance(v, torch.Tensor) else v


def lit_land(globe, P, B) -> torch.Tensor:
    """Land cells [W, C] that get light during the year (cached in B)."""
    if "_lit_land" not in B:
        B["_lit_land"] = P["land"] & (cl.annual_insolation(globe, P) > 0)
    return B["_lit_land"]


def init_state(globe, P, B, clim, radius_m: float | None = None) -> BioState:
    """Bare litter, seeded soft tissue on land cells with light during the year, the mineral N, no wood and no
    phytoplankton stock; the ledger baselines are taken against ``clim``."""
    r: BioRules = B["rules"]
    W, dev = P["W"], globe.device
    radius = float(P["radius_m"] if radius_m is None else radius_m)
    land = P["land"]
    zeros = torch.zeros(W, globe.C, device=dev)
    z64 = torch.zeros(W, dtype=torch.float64, device=dev)
    bio = BioState(plant_c=torch.where(lit_land(globe, P, B), r.seed_c_kg_m2, 0.0).float(), wood_c=zeros.clone(),
                   litter_c=zeros.clone(), nutrient=torch.where(land, r.nutrient_start_kg_m2, 0.0).float(),
                   phyto_c=zeros.clone(), cell_m2=globe.area64 * radius ** 2, area64=globe.area64,
                   carbon0=z64.clone(), oxygen0=z64.clone(), c_harvested=z64.clone(), c_wood_collected=z64.clone(),
                   c_buried=z64.clone(), c_litter_added=z64.clone(), t_acclim=clim.T.clone())
    return reset_ledgers(bio, clim)


def _rules(rules) -> BioRules:
    return rules["rules"] if isinstance(rules, dict) else rules


def cover(bio: BioState, rules) -> torch.Tensor:
    """Vegetation cover [W, C] in [0, 1] = fAPAR (the climate's vegetation albedo uses it); ``rules`` is the
    world's BioRules or the biosphere parameters B."""
    return 1 - torch.exp(-bio.plant_c / _rules(rules).fapar_kg_c)


def temperature_factor(t_k, r: BioRules):
    """f_T: the Yan & Hunt (1999) beta function, 0 at t_min and t_max, 1 at t_opt."""
    expo = (r.t_opt_k - r.t_min_k) / (r.t_max_k - r.t_opt_k)
    hi = ((r.t_max_k - t_k) / (r.t_max_k - r.t_opt_k)).clamp_min(0)
    lo = ((t_k - r.t_min_k) / (r.t_opt_k - r.t_min_k)).clamp_min(0)
    return hi * lo ** expo


def co2_factor(p_co2_pa, r: BioRules):
    return (p_co2_pa / (p_co2_pa + r.co2_half_pa)) / (r.co2_ref_pa / (r.co2_ref_pa + r.co2_half_pa))


def ocean_temperature_factor(t_k, r: BioRules):
    """Eppley's 1.066^(T - T_ref), held at its value at eppley_t_max_k above it (``PROVENANCE['eppley']``)."""
    return r.eppley_base ** (t_k.clamp_max(r.eppley_t_max_k) - r.eppley_t_k)


def nitrogen_total(bio: BioState, r: BioRules) -> torch.Tensor:
    """Nitrogen of each cell [W, C] (float64 kg N/m^2): mineral + soft tissue + wood."""
    return bio.nutrient.double() + bio.plant_c.double() / r.cn_leaf + bio.wood_c.double() / r.cn_wood


def _mean(x, area64):
    return (x.double() * area64).sum(-1) / FOUR_PI


# ------------------------------------------------------------------------------------------- the step
def step(bio: BioState, clim, globe, P, B, cdiag):
    """One day of the biosphere after ``climate.step`` (``cdiag`` its diagnostics). Returns (bio, clim, diag).

    The new climate state has the day's gas exchange in ``gas``. Diagnostics ([W, C] kg C/m^2 per day): gpp,
    npp, respiration (maintenance), decomposition, litterfall (litterfall_wood its wood part), seed (seed rain),
    resprout, phyto_production, phyto_loss, burial; n_fixed (kg N/m^2 that day); gpp_potential (GPP at full cover
    and f_N = 1), gpp_raw (GPP before the nitrogen down-regulation of growth), maintenance (r_N q dt, kg C per kg
    N), decay (the day's litter decay share), rm_leaf, rm_wood (maintenance due), burn_leaf, burn_wood (tissue
    burnt for what GPP did not pay), surplus (GPP after maintenance) and grow (the part of it that grew); and
    [W] global means (``mean_<name>``, float64) plus ``exchange_mol`` [W] (mol C fixed net of respiration).
    """
    r: BioRules = B["rules"]
    dt_yr = 1.0 / DAYS_PER_YEAR
    land = B["land"]
    lit = lit_land(globe, P, B)
    T, soil = clim.T, clim.soil
    area64 = bio.area64
    a_w = r.wood_allocation
    par_mj = cdiag["surface_sw"] * B["par_fraction"][:, None] * MJ_PER_W_DAY
    p_co2 = cl.gas_pressures(clim.gas, P)[:, cl.I_CO2]
    f_co2 = co2_factor(p_co2, r).float()[:, None]
    f_w = (soil / P["rules"].bucket_kg_m2).clamp(0, 1)
    f_t = temperature_factor(T, r)
    nut = bio.nutrient.clamp_min(0)
    f_n = nut / (nut + r.nutrient_half_kg_m2)
    fapar = 1 - torch.exp(-bio.plant_c / r.fapar_kg_c)
    potential = torch.where(land, r.lue_g_c_mj * 1e-3 * par_mj * f_t * f_w * f_co2, 0.0)
    gpp = potential * fapar * f_n
    f_ice = cl.ice_fraction(T, P["rules"])
    prod = torch.where(land, 0.0, B["e_ocean"] * par_mj * ocean_temperature_factor(T, r) * (1 - f_ice))
    # never take more than half of the CO2 column in a day (a guard; never active on a real planet)
    fix_mol = _mean(gpp + prod, area64) / M_C
    s_c = (0.5 * clim.gas[:, cl.I_CO2] / fix_mol.clamp_min(1e-30)).clamp_max(1.0).float()[:, None]
    gpp, prod, potential = gpp * s_c, prod * s_c, potential * s_c

    # land: maintenance respiration per pool, GPP pays the soft tissue's first, then the wood's
    plant0, wood0 = bio.plant_c, bio.wood_c
    q10 = r.q10 ** ((T - r.t_q10_k) / 10.0)                            # decomposition
    # maintenance acclimated to the running-mean temperature, with the short-term Q10 around it
    share = max(1.0 - math.exp(-1.0 / r.acclimation_days), 1.0 / (bio.day + 1))
    before = T if bio.t_acclim is None else bio.t_acclim
    t_acclim = before + share * (T - before)
    maint = B["r_n"] * r.q10 ** ((T - t_acclim) / 10.0) * dt_yr       # kg C per kg tissue N per day
    rm_leaf = maint * plant0 / r.cn_leaf
    rm_wood = maint * wood0 / r.cn_wood
    pay_leaf = torch.minimum(gpp, rm_leaf)
    pay_wood = torch.minimum(gpp - pay_leaf, rm_wood)
    surplus = gpp - pay_leaf - pay_wood
    n_per_c = (1 - a_w) / r.cn_leaf + a_w / r.cn_wood
    need = surplus * n_per_c
    phi = torch.where(need > nut, nut / need.clamp_min(1e-30), torch.ones_like(need))
    grow = surplus * phi
    gpp_raw = gpp
    gpp = pay_leaf + pay_wood + grow                                   # down-regulated by nitrogen
    # unpaid maintenance burns the pool's own tissue, then the other pool's (wood, the reserves, first)
    def_leaf, def_wood = rm_leaf - pay_leaf, rm_wood - pay_wood
    own_wood = torch.minimum(def_wood, wood0)
    own_leaf = torch.minimum(def_leaf, plant0)
    rest = (def_wood - own_wood) + (def_leaf - own_leaf)
    cross_wood = torch.minimum(rest, wood0 - own_wood)
    cross_leaf = torch.minimum(rest - cross_wood, plant0 - own_leaf)
    burn_wood, burn_leaf = own_wood + cross_wood, own_leaf + cross_leaf
    resp = pay_leaf + pay_wood + burn_leaf + burn_wood
    npp = gpp - resp
    plant = plant0 + grow * (1 - a_w) - burn_leaf
    wood = wood0 + grow * a_w - burn_wood
    nutrient = bio.nutrient - grow * n_per_c + burn_leaf / r.cn_leaf + burn_wood / r.cn_wood
    # resprouting from the wood's reserves where the plant could photosynthesise
    dn = 1.0 / r.cn_leaf - 1.0 / r.cn_wood                             # extra N per kg C moved to soft tissue
    active = land & (par_mj > 0) & (f_t > 0) & (f_w > 0)
    short = (B["leaf_wood"] * wood - plant).clamp_min(0)
    sprout = torch.where(active, short * (1 - math.exp(-1.0 / r.resprout_days)), 0.0)
    sprout = torch.minimum(torch.minimum(sprout, wood), nutrient.clamp_min(0) / dn)
    plant, wood = plant + sprout, wood - sprout
    nutrient = nutrient - sprout * dn
    # seed rain (carbon from the air, nitrogen from the mineral pool): it refills the floor at the rate of a year's seed
    # rain, and only where a seedling could grow today (light, warmth and water: ``active``). Topping the floor up
    # in full every day, as before, fed grazers 1 g C/m^2 a day on barren planets (up to 365 g C/m^2/yr, several
    # times a planet's real production) without any photosynthesis.
    seed = torch.where(active & lit, (r.seed_floor_kg_m2 - plant).clamp_min(0) * (1 - math.exp(-dt_yr)), 0.0)
    seed = torch.minimum(seed, nutrient.clamp_min(0) * r.cn_leaf)
    plant = plant + seed
    nutrient = nutrient - seed / r.cn_leaf
    # turnover and decomposition (exact daily exponentials)
    lit_leaf = plant * (1 - math.exp(-r.leaf_turnover_yr * dt_yr))
    lit_wood = wood * (1 - math.exp(-r.wood_turnover_yr * dt_yr))
    plant, wood = plant - lit_leaf, wood - lit_wood
    nutrient = nutrient + lit_leaf / r.cn_leaf + lit_wood / r.cn_wood
    # nitrogen fixation refills a cell below its starting N (harvest and wood collection take N away)
    n_tot = nutrient.double() + plant.double() / r.cn_leaf + wood.double() / r.cn_wood
    gap = (1 - n_tot / r.nutrient_start_kg_m2).clamp(0, 1).float()
    n_fixed = torch.where(land, r.n_fixation_kg_m2_yr * dt_yr * gap, 0.0)
    nutrient = nutrient + n_fixed
    decay = 1 - torch.exp(-r.decomposition_yr * q10 * torch.where(land, f_w, 1.0) * dt_yr)
    litter = bio.litter_c + lit_leaf + lit_wood
    decomp = litter * decay
    # ocean
    loss = bio.phyto_c * (1 - math.exp(-1.0 / r.phyto_turnover_days))
    burial = loss * r.burial_share
    remin = loss - burial
    # less decomposition and remineralisation when O2 runs short (a guard: anoxic decay is slower)
    o2_avail = clim.gas[:, cl.I_O2] + _mean(gpp + prod + seed - resp, area64) / M_C
    o2_need = _mean(decomp + remin, area64) / M_C
    s_o = (0.9 * o2_avail.clamp_min(0) / o2_need.clamp_min(1e-30)).clamp_max(1.0).float()[:, None]
    decomp, remin, decay = decomp * s_o, remin * s_o, decay * s_o
    loss = remin + burial
    litter = litter - decomp
    phyto = bio.phyto_c + prod - loss

    exchange = (_mean(gpp + prod + seed - resp - decomp - remin, area64)) / M_C        # mol C fixed, net
    gas = clim.gas.clone()
    gas[:, cl.I_CO2] -= exchange
    gas[:, cl.I_O2] += exchange
    new_bio = replace(bio, plant_c=plant, wood_c=wood, litter_c=litter, nutrient=nutrient, phyto_c=phyto,
                      c_buried=bio.c_buried + _mean(burial, area64), t_acclim=t_acclim, day=bio.day + 1)
    diag = {"gpp": gpp, "npp": npp, "respiration": resp, "decomposition": decomp, "litterfall": lit_leaf + lit_wood,
            "litterfall_wood": lit_wood,
            "seed": seed, "resprout": sprout, "n_fixed": n_fixed, "phyto_production": prod, "phyto_loss": loss,
            "burial": burial, "exchange_mol": exchange, "gpp_potential": potential, "gpp_raw": gpp_raw,
            "maintenance": maint, "decay": decay, "rm_leaf": rm_leaf, "rm_wood": rm_wood, "burn_leaf": burn_leaf,
            "burn_wood": burn_wood, "surplus": surplus, "grow": grow}
    for name in ("gpp", "npp", "respiration", "decomposition", "litterfall", "seed", "phyto_production", "burial"):
        diag["mean_" + name] = _mean(diag[name], area64)
    return new_bio, replace(clim, gas=gas), diag


# ------------------------------------------------------------------------------------------- scatters
def _take(bio: BioState, pool: str, counter: str, w, cell, kg_c_requested, n_back: float = 0.0):
    """Remove carbon (kg C per request [N]) from ``pool`` at (w, cell), never below zero; requests on one cell
    share what is there in proportion. The pool is rebound to a new tensor. Returns kg C given [N] (float64):
    each request's share of what actually left the float32 pool (within float32 resolution of the request), so
    giver and taker book the same carbon. ``n_back`` (kg N per kg C) of the carbon that left goes back to the cell's
    mineral nitrogen."""
    stock = getattr(bio, pool)
    W, C = stock.shape
    w = torch.as_tensor(w, device=stock.device).long().reshape(-1)
    cell = torch.as_tensor(cell, device=stock.device).long().reshape(-1)
    req = torch.as_tensor(kg_c_requested, dtype=torch.float64, device=stock.device).reshape(-1).clamp_min(0)
    req = req.expand(w.shape[0]) if req.numel() == 1 else req
    idx = w * C + cell
    per_m2 = req / bio.cell_m2[cell]
    total = torch.zeros(W * C, dtype=torch.float64, device=stock.device).index_add_(0, idx, per_m2)
    have = stock.reshape(-1).double()
    left = torch.where(total > have, torch.zeros_like(have), have - total).float()
    removed = have - left.double()                      # what left the float32 pool, exactly
    setattr(bio, pool, left.view(W, C))
    setattr(bio, counter, getattr(bio, counter) + (removed.view(W, C) * bio.area64).sum(-1) / FOUR_PI)
    if n_back > 0:
        bio.nutrient = (bio.nutrient.reshape(-1).double() + removed * n_back).float().view(W, C)
    # each request gets its share of what actually left the cell (so the taker books the same carbon)
    return per_m2 / total[idx].clamp_min(1e-300) * removed[idx] * bio.cell_m2[cell]


def harvest(bio: BioState, w, cell, kg_dry_requested, carbon_fraction: float = BioRules.plant_carbon_fraction,
            rules: BioRules | None = None):
    """Plant matter eaten or gathered: kg dry requested per request ([N] world index, cell index) from the soft
    tissue ``plant_c`` (the state's field is rebound). Returns kg dry given [N] (float64, exactly the booked carbon
    / carbon_fraction; a float32 cast of it differs by up to 6e-8 relative), never more than is there. Its carbon is
    booked in ``c_harvested`` (dry matter is 47 % carbon); ``excreted_n_share`` of its nitrogen goes back to the
    cell's mineral pool (what the eaters excrete), the rest leaves the cell (fixation refills it). ``rules`` are
    the world's BioRules (the defaults when not given)."""
    r = _rules(rules) if rules is not None else BioRules()
    kg = torch.as_tensor(kg_dry_requested, dtype=torch.float64, device=bio.plant_c.device)
    return _take(bio, "plant_c", "c_harvested", w, cell, kg * carbon_fraction,
                 r.excreted_n_share / r.cn_leaf) / carbon_fraction


def collect_wood(bio: BioState, w, cell, kg_requested):
    """Wood collected: kg of dry wood requested per request from ``wood_c`` (the state's field is rebound). Returns
    kg given [N] (float64, see ``harvest``). Its carbon (the carbon fraction of materials 'wood') is booked in
    ``c_wood_collected``; the item made from it carries the same carbon in the item ledger."""
    cf = materials.species_elements("wood")["C"]
    kg = torch.as_tensor(kg_requested, dtype=torch.float64, device=bio.wood_c.device)
    return _take(bio, "wood_c", "c_wood_collected", w, cell, kg * cf) / cf


def add_litter(bio: BioState, w, cell, kg_c):
    """Organic carbon returned to the ground (decayed corpses and items), kg C per request (the state's
    ``litter_c`` is rebound); booked in ``c_litter_added``. On an ocean cell it decays like litter without the
    soil-moisture factor. It brings no nitrogen (bodies carry no N ledger)."""
    W, C = bio.litter_c.shape
    dev = bio.litter_c.device
    w = torch.as_tensor(w, device=dev).long().reshape(-1)
    cell = torch.as_tensor(cell, device=dev).long().reshape(-1)
    add = torch.as_tensor(kg_c, dtype=torch.float64, device=dev).reshape(-1).clamp_min(0)
    add = add.expand(w.shape[0]) if add.numel() == 1 else add
    before = bio.litter_c.reshape(-1).double()
    after = (before.index_add(0, w * C + cell, add / bio.cell_m2[cell])).float()
    bio.litter_c = after.view(W, C)
    bio.c_litter_added = bio.c_litter_added + ((after.double() - before).view(W, C) * bio.area64).sum(-1) / FOUR_PI


# ------------------------------------------------------------------------------------------- ledgers
def organic_carbon(bio: BioState) -> torch.Tensor:
    """Organic carbon of the biosphere [W] (float64, global-mean kg C/m^2)."""
    return _mean(bio.plant_c + bio.wood_c + bio.litter_c + bio.phyto_c, bio.area64)


def _carbon_balance(bio: BioState, clim) -> torch.Tensor:
    stores = clim.gas[:, cl.I_CO2] * M_C + organic_carbon(bio)
    out = bio.c_harvested + bio.c_wood_collected + bio.c_buried + clim.co2_weathered * M_C
    into = bio.c_litter_added + (clim.co2_outgassed + clim.gas_external[:, cl.I_CO2]) * M_C
    return stores + out - into


def _oxygen_balance(bio: BioState, clim) -> torch.Tensor:
    exported = (bio.c_harvested + bio.c_wood_collected + bio.c_buried - bio.c_litter_added) / M_C
    return clim.gas[:, cl.I_O2] - organic_carbon(bio) / M_C - exported - clim.gas_external[:, cl.I_O2]


def carbon_ledger(bio: BioState, clim, globe=None) -> torch.Tensor:
    """Carbon ledger error [W] (float64, global-mean kg C/m^2): CO2 column + organic pools + carbon that left
    (harvest, wood, burial, weathering) - carbon that came in (litter added, outgassing, CO2 from other modules),
    against the baseline."""
    return _carbon_balance(bio, clim) - bio.carbon0


def oxygen_ledger(bio: BioState, clim, globe=None) -> torch.Tensor:
    """Oxygen ledger error [W] (float64, mol O2/m^2): O2 column - organic carbon (mol) - organic carbon that left
    (harvest, wood, burial) + organic carbon that came in (litter added) - O2 added by other modules
    (``climate.add_gas``), against the baseline. Each mol of carbon fixed gives one mol of O2 and each mol
    respired or decomposed takes one back, so this is constant."""
    return _oxygen_balance(bio, clim) - bio.oxygen0


def reset_ledgers(bio: BioState, clim) -> BioState:
    """Take the carbon and oxygen ledger baselines from the current states."""
    return replace(bio, carbon0=_carbon_balance(bio, clim), oxygen0=_oxygen_balance(bio, clim))


# ------------------------------------------------------------------------------------------- spin-up
def run(bio: BioState, clim, globe, P, B, days: int, hold_gas: bool = False, stats=None):
    """``days`` coupled days (climate with the vegetation cover, then the biosphere). With ``hold_gas`` the gas
    column is put back after every day (the spin-up keeps the chain's air). ``stats`` (optional) is a callable
    (day index, bio, cdiag, bdiag) called after each day. Returns (bio, clim, last diags)."""
    cdiag = bdiag = None
    for d in range(days):
        clim, cdiag = cl.step(clim, globe, P, cover(bio, B))
        gas = clim.gas
        bio, clim, bdiag = step(bio, clim, globe, P, B, cdiag)
        if hold_gas:
            clim = replace(clim, gas=gas)
        if stats is not None:
            stats(d, bio, cdiag, bdiag)
    return bio, clim, (cdiag, bdiag)


_STATS = ("gpp_raw", "gpp_potential", "rm_leaf", "rm_wood", "burn_leaf", "burn_wood", "surplus", "grow", "resprout",
          "decay", "plant", "wood", "litter", "nutrient", "decay_litter")


def _collect(sums, use, bio, bdiag):
    """Add one day to the spin-up's window sums (``use`` [W, 1] marks the worlds whose window it is in)."""
    day = {name: bdiag[name] for name in _STATS if name in bdiag}
    day.update(plant=bio.plant_c, wood=bio.wood_c, litter=bio.litter_c, nutrient=bio.nutrient,
               decay_litter=bdiag["decay"] * bio.litter_c)
    for name in _STATS:
        sums[name] = sums.get(name, 0.0) + use * day[name].double()
    # the litter's year as a linear map L_end = pi L_start + x (x split by the source of its litterfall), so that
    # the periodic steady state at the window's end is x / (1 - pi)
    keep = 1 - bdiag["decay"].double()
    fall_wood = bdiag["litterfall_wood"].double()
    on = use > 0
    sums["x_leaf"] = torch.where(on, (sums.get("x_leaf", 0.0) + bdiag["litterfall"].double() - fall_wood) * keep,
                                 sums.get("x_leaf", torch.zeros_like(keep)))
    sums["x_wood"] = torch.where(on, (sums.get("x_wood", 0.0) + fall_wood) * keep,
                                 sums.get("x_wood", torch.zeros_like(keep)))
    sums["pi"] = torch.where(on, sums.get("pi", 1.0) * keep, sums.get("pi", torch.ones_like(keep)))


def steady_state(bio: BioState, B, means: dict, iters: int = 60):
    """Steady-state soft tissue, wood and litter [W, C] (float64) of each cell under the window's conditions, plus
    the effective litter decay share. Returns (plant, wood, litter, decay).

    ``means`` holds window means (see ``_collect``) of the day's fluxes and pools. The cell's year is reduced to
    effective rates that keep the seasonal covariance of the pools with the conditions (a plant grows in the
    season that feeds it): G = mean(gpp_raw) / (fAPAR(mean plant) f_N(mean nutrient)), the maintenance per kg of
    each pool mu_x = mean(rm_x) / mean(pool), the share of it that went unpaid and burnt the pool s_x =
    mean(burn_x) / mean(rm_x), the growth share of the surplus phi (nitrogen), resprouting per kg of wood
    sigma = mean(resprout) / mean(wood), and the litter decay mean(decay x litter) / mean(litter). The steady
    state then solves
        soft tissue: (1 - a_w) phi S - s_l mu_l plant + sigma wood - k_l plant = 0,
        wood:        a_w phi S - (s_w mu_w + sigma + k_w) wood = 0,
        S = G (1 - exp(-plant / 0.5)) f_N(N - plant / CN_leaf - wood / CN_wood) - (1 - s_l) mu_l plant
            - (1 - s_w) mu_w wood,
    the cell's nitrogen N fixed: wood = rho' plant with rho' = (s_l mu_l + k_l) / ((1 - a_w) c_w / a_w + sigma),
    c_w = s_w mu_w + sigma + k_w, and the wood equation by bisection in plant (0 where the plant cannot grow).
    Without seasons (s = sigma = 0, phi = 1) this is NPP split a_w : 1 - a_w with wood = plant / rho. Litter is
    the steady-state litterfall over the decay."""
    r: BioRules = B["rules"]
    a = r.wood_allocation
    k_l = 1 - math.exp(-r.leaf_turnover_yr / DAYS_PER_YEAR)
    k_w = 1 - math.exp(-r.wood_turnover_yr / DAYS_PER_YEAR)
    tiny = 1e-12

    def ratio(num, den, fallback):
        return torch.where(den > tiny, num / den.clamp_min(tiny), fallback)

    def f_n(n):
        n = n.clamp_min(0)
        return n / (n + r.nutrient_half_kg_m2)

    zero = torch.zeros_like(means["plant"])
    p_bar, w_bar = means["plant"], means["wood"]
    g = ratio(means["gpp_raw"], (1 - torch.exp(-p_bar / r.fapar_kg_c)) * f_n(means["nutrient"]), means["gpp_potential"])
    maint = B["r_n"] / DAYS_PER_YEAR                                   # fallback: maintenance at 288 K
    mu_l = ratio(means["rm_leaf"], p_bar, zero + maint / r.cn_leaf)
    mu_w = ratio(means["rm_wood"], w_bar, zero + maint / r.cn_wood)
    s_l = ratio(means["burn_leaf"], means["rm_leaf"], zero).clamp(0, 1)
    s_w = ratio(means["burn_wood"], means["rm_wood"], zero).clamp(0, 1)
    phi = ratio(means["grow"], means["surplus"], zero + 1).clamp(0, 1)
    sigma = ratio(means["resprout"], w_bar, zero)
    decay = ratio(means["decay_litter"], means["litter"], means["decay"]).clamp_min(1e-12)
    c_w = s_w * mu_w + sigma + k_w
    rho_w = (s_l * mu_l + k_l) / ((1 - a) * c_w / a + sigma)          # wood per kg of soft tissue
    n_tot = nitrogen_total(bio, r)
    n_per_p = 1 / r.cn_leaf + rho_w / r.cn_wood                         # tissue N per kg C of soft tissue

    def f(p):
        s = g * (1 - torch.exp(-p / r.fapar_kg_c)) * f_n(n_tot - p * n_per_p) \
            - (1 - s_l) * mu_l * p - (1 - s_w) * mu_w * rho_w * p
        return a * phi * s - c_w * rho_w * p

    hi = n_tot / n_per_p
    lo = torch.full_like(hi, 1e-9)
    grows = f(lo) > 0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        up = f(mid) > 0
        lo, hi = torch.where(up, mid, lo), torch.where(up, hi, mid)
    plant = torch.where(grows, 0.5 * (lo + hi), zero)
    wood = rho_w * plant
    litter = (k_l * plant + k_w * wood) / decay
    return plant, wood, litter, decay


def spin_up(bio: BioState, clim, globe, P, B, days: int = 730, hold_gas: bool = True, segments: int = 2,
            reset_water: bool = False):
    """Grow the vegetation on the spun-up climate for ``days`` days, holding the chain's air composition
    (``hold_gas``), and put the slow pools at their steady state; then re-base the carbon and oxygen ledgers
    (the water ledger only with ``reset_water``: the biosphere moves no water). Returns (bio, clim, report).

    The days run in ``segments`` equal parts. At the end of each, the cell's steady state under the conditions of
    the part's last whole year (each world's own; the whole part when the year is longer) is worked out
    (:func:`steady_state`): the wood is set to it (its 20-year turnover would otherwise keep drawing CO2 from the
    air for decades after the spin-up, and nothing in the model buffers the air), and the litter to its steady
    state under the new litterfall. Before the last part the soft tissue is scaled so that its window mean is its
    steady state too (the slow mode of a stand is the soft tissue and wood together: a young stand carries too many
    leaves for its wood); in the last part it keeps its own (seasonal) state. The litter is set to the periodic
    steady state of the window's year (``_collect`` keeps its yearly linear map), which keeps the seasonal phase;
    where its yearly decay is below 1 % it is left as it is. The nitrogen moved into or out of the plant comes from
    or goes to the mineral pool (scaled down so that the pool stays >= 0). Carbon is not conserved by the jumps
    (the air is held; the ledgers are re-based at the end)."""
    r: BioRules = B["rules"]
    years = P["year_days"].double().round().clamp_min(1)
    cdiag = bdiag = None
    report = {"days": days, "segments": []}
    segments = max(int(segments), 1)
    base = days // segments
    k_l = 1 - math.exp(-r.leaf_turnover_yr / DAYS_PER_YEAR)
    k_w = 1 - math.exp(-r.wood_turnover_yr / DAYS_PER_YEAR)
    land = B["land"]
    for k in range(segments):
        n = base if k < segments - 1 else days - base * (segments - 1)
        if n <= 0:
            continue
        window = torch.where(years <= n, years, torch.full_like(years, float(n)))
        sums: dict = {}

        def collect(d, b, cd, bd, n=n, window=window, sums=sums):
            _collect(sums, ((n - d) <= window).double()[:, None], b, bd)

        bio, clim, (cdiag, bdiag) = run(bio, clim, globe, P, B, n, hold_gas, collect)
        means = {name: sums[name] / window[:, None] for name in _STATS}
        plant_ss, wood_ss, _, decay = steady_state(bio, B, means)
        # soft tissue scaled so that its window mean is the steady state (keeping its seasonal phase), wood at the
        # steady state; their nitrogen comes from or goes to the mineral pool, which stays >= 0
        plant0, wood0 = bio.plant_c.double(), bio.wood_c.double()
        p_bar = means["plant"]
        last = k == segments - 1
        scale = torch.where(land & (p_bar > 1e-9) & (not last), plant_ss / p_bar.clamp_min(1e-9),
                            torch.ones_like(p_bar))
        plant = plant0 * scale
        wood = torch.where(land, wood_ss, wood0)
        n_tot = nitrogen_total(bio, r)
        need = plant / r.cn_leaf + wood / r.cn_wood
        fit = torch.where(need > n_tot, n_tot / need.clamp_min(1e-30), torch.ones_like(need))
        plant, wood = plant * fit, wood * fit
        nutrient = n_tot - plant / r.cn_leaf - wood / r.cn_wood
        # litter at the periodic steady state of the window's year (exact for its decay and litterfall series, the
        # litterfall rescaled to the new soft tissue and wood); kept where it would take over a century to settle
        fall = sums["x_leaf"] * scale * fit + sums["x_wood"] * wood / means["wood"].clamp_min(1e-12)
        settles = land & (1 - sums["pi"] > 0.01)
        litter = torch.where(settles, fall / (1 - sums["pi"]).clamp_min(1e-12), bio.litter_c.double())
        m = lambda x: _mean(x, bio.area64).tolist()
        report["segments"].append({"days": n, "plant_before": m(plant0), "plant_after": m(plant),
                                   "wood_before": m(wood0), "wood_after": m(wood),
                                   "litter_before": m(bio.litter_c), "litter_after": m(litter)})
        bio = replace(bio, plant_c=plant.float(), wood_c=wood.float(), litter_c=litter.float(),
                      nutrient=nutrient.float().clamp_min(0))
    bio = reset_ledgers(bio, clim)
    if reset_water:
        clim = cl.reset_water_ledger(clim, globe)
    report.update(plant_c=organic_carbon(bio).tolist(),
                  mean_npp_kg_c_m2_day=bdiag["mean_npp"].tolist() if bdiag else None,
                  mean_t=cdiag["mean_t"].tolist() if cdiag else None)
    return bio, clim, report
