"""Climate of the habitat globe (PLANET-SPEC section 2.6): insolation, an energy-balance model of the
Budyko-Sellers form, the water cycle with a closed water ledger, the global gas column and the
carbonate-silicate cycle.

State (:class:`ClimateState`, leading W axis):

* ``T`` [W, C] surface air temperature (K) at the cell's ground, ``vapour`` column water (kg/m^2),
  ``soil`` the bucket (kg/m^2, at most 150), ``snow`` (kg/m^2 water equivalent): float32.
* ``gas`` [W, 4] the planet's well-mixed dry air as moles per m^2 of column (N2, O2, CO2, Ar), so
  that p_i = n_i g M_i; ``ocean`` [W] the ocean as a global-mean column (kg/m^2); the ledger
  counters. These are float64: a day's biological exchange is about 1e-6 of the gas column, below
  float32 resolution (the gas and ocean stores are [W] numbers, so the cost is nil).
* ``cloud_albedo`` [W], the one calibrated number (see below); ``runoff_mean`` [W, C] the running-mean runoff
  (kg/m^2 per day) that weathers.

One step is one day (``DAY_S``):

1. insolation, the daily mean: a locked planet gets S max(0, cos theta) around a substellar point
   fixed at lon 0, lat 0; a rotating planet the diurnal mean of Berger (1978) for the declination
   of its obliquity and orbital phase (circular orbit, vernal equinox at day 0).
2. evaporation by the bulk formula E = C_E U (rho_v,sat(T) - rho_v) (Dalton), the surface vapour
   density rho_v = vapour / H_w of an exponential profile of scale height H_w(T) = R_v T^2 / (L_v
   Gamma) (formation's vapour rule) and rho_v,sat from Magnus e_sat(T); ocean cells draw on the
   ocean (none under sea ice), land cells on the bucket with Manabe's (1969) beta = soil / (0.75 x
   150), none under snow.
3. radiation, linearised backward Euler (stable at any step):
   C dT/dt = I (1 - alpha_p) - sigma T^4 / (1 + 0.75 tau).
4. transport, on the sea-level equivalent theta = T + Gamma_d h (Gamma_d = g / c_p, h the ground
   above the sea): the Sellers diffusion D lap theta with D = 0.55 W m^-2 K^-1 x P / 1 bar
   (``globe.diffuse``, explicit sub-steps; latent heat is inside D, as in the Budyko-Sellers form),
   then Budyko's exchange with the free troposphere, gamma (H - theta_e), implicit and exactly
   energy-conserving. gamma is the bulk surface-air exchange rho c_p C_H U times the share
   w = Lambda^2 / (1 + Lambda^2) of the weak-temperature-gradient regime (Lambda = N H / (Omega R)):
   about 6 W m^-2 K^-1 for a locked planet, 0.2 for Earth. It acts on the moist static energy
   theta_e = theta + L_v q(r b e_sat(T)) / c_p at the formation's relative humidity r times the
   surface's evaporation efficiency b (Manabe's beta under no snow on land, open water on the ocean),
   so warm wet cells export heat as latent heat and dry ones do not (the substellar point of a
   locked planet stays near 310 K). Vapour is mixed in the same way (D and gamma over the air
   column's heat capacity).
5. precipitation: a background rate vapour / 8.9 d (the residence time of atmospheric water) plus
   everything above the saturation column; snow below 273.15 K. Rain fills the bucket, the
   overflow runs off to the ocean, snow melts by a degree-day rule into the bucket, and snow above
   1000 kg/m^2 leaves as glacier discharge to the ocean.
6. carbonate-silicate cycle (Walker, Hays & Kasting 1981 with the runoff law of Berner 1994): weathering
   removes CO2 at F0 exp((T - 288) / 13.7) (runoff / runoff0)^0.65 per m^2 of land, on the cell's
   running-mean runoff (``runoff_mean``, a one-year e-folding mean: the daily bucket overflow is spiky
   and the power law is concave); volcanoes add it in proportion to the radiogenic heat flow. runoff0 is
   the model's own Earth's, so that the model's Earth is in carbonate balance. Both are about 1e-6 of
   the column per year.

Albedo per cell: ocean by sun angle (Briegleb et al. 1986, at the day's insolation-weighted cos zenith;
0.06 when no sun angle is given), bare land 0.25, vegetation 0.15 (a cover passed in by the biosphere),
snow and ice 0.6 (cells below 263 K, ramped over 258-268 K, and snow-covered land). One layer of
cloud and air lies over the surface (adding method): it reflects ``cloud_albedo`` a_c, and the air
absorbs 22 % of what crosses it, so t = (1 - a_c)(1 - 0.22) and alpha_p = a_c + t^2 a_s / (1 - a_c a_s);
the ground gets I t / (1 - a_c a_s) (``surface_sw``, the light for plants and eyes). ``spin_up``
calibrates a_c per world so that the planet without life reaches the chain's T_s (the global mean;
the gate is 3 K).

Greenhouse: tau = tau_rest + tau_per_ln_co2 ln(p_CO2 / co2_ref_pa) + tau_w(vapour), so tau is the
formation's ``greenhouse_tau`` at the calibration point. The water term (``h2o_feedback``) is
max(-tau_w,ref, k_w ln(vapour / vapour_ref)): logarithmic like the CO2 term, with k_w set so that at
the calibration point it adds the IPCC AR6 ratio of the water vapour + lapse rate feedback to the
Planck feedback (1.30 / 3.22) of the model's own Planck response (so it amplifies the response by
1 / (1 - 0.40) and cannot run away); the water greenhouse vanishes with the vapour (the floor
-tau_w,ref). Without it the response is Planck-only.

Every set-up number carries a provenance entry in ``PROVENANCE`` (rules) or in the ``provenance``
of :func:`make_params` (per-world values). Water is booked as a global-mean column (kg per m^2 of
the habitat surface, sums of area x column / 4 pi in float64).

States are values: ``step`` returns a new state, and the exchanges with other modules (``take_water``,
``give_water``, ``add_gas``) rebind the state's fields to new tensors instead of writing into them, so an
earlier state or ``state_dict`` (which returns copies) never changes afterwards.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields, replace

import torch

from . import constants as K

DAY_S = K.DAY_S
FOUR_PI = 4.0 * math.pi
GASES = ("N2", "O2", "CO2", "Ar")
I_N2, I_O2, I_CO2, I_AR = range(4)
M_GAS = tuple(K.molar_mass(g) for g in GASES)      # kg/mol
M_H2O = K.molar_mass("H2O")
M_CO2 = K.molar_mass("CO2")
M_C = K.element_mass("C")
R_V = K.R_GAS / M_H2O                              # J/(kg K), water vapour gas constant
T_FREEZE = 273.15                                  # K
DAYS_PER_YEAR = K.YEAR_S / K.DAY_S                 # 365.25
EARTH_AREA = FOUR_PI * K.R_EARTH ** 2
EARTH_LAND_FRACTION = 1.0 - 0.709                  # Charette & Smith 2010 (globe.EARTH_OCEAN_FRACTION)


@dataclass(frozen=True)
class ClimateRules:
    """Modelling choices and reference values of the climate (sources in ``PROVENANCE``)."""
    heat_ocean: float = 2.1e8          # J m^-2 K^-1, 50 m mixed layer
    heat_land: float = 1.0e7           # J m^-2 K^-1, land plus air column
    heat_spin: float = 1.0e7           # J m^-2 K^-1, every cell during an accelerated spin-up
    spin_chunk_days: int = 60          # days per accelerated spin-up chunk (and per chunk of a locked planet)
    d_per_bar: float = 0.55            # W m^-2 K^-1 per bar of surface pressure
    air_absorption: float = 0.22       # share of the shortwave crossing the layer that the air absorbs
    albedo_ocean: float = 0.06
    ocean_albedo_zenith: bool = True   # ocean albedo by sun angle (Briegleb et al. 1986)
    albedo_ice: float = 0.60
    albedo_land: float = 0.25
    albedo_vegetation: float = 0.15
    ice_k: float = 263.0               # K, the ice line
    ice_ramp_k: float = 10.0           # K, width of the ice ramp around ice_k
    snow_mask_kg_m2: float = 10.0      # snow water equivalent that covers the ground fully
    bucket_kg_m2: float = 150.0
    soil_start: float = 0.5            # share of the bucket filled at the start
    beta_share: float = 0.75           # Manabe 1969: evaporation is free above 0.75 of the bucket
    transfer_coeff: float = 1.15e-3    # bulk transfer coefficient for vapour and heat
    wind_m_s: float = 5.0              # near-surface wind of the bulk formulas
    residence_days: float = 8.9        # atmospheric residence time of water
    degree_day_kg_m2_k: float = 4.0    # snow melt per K above freezing per day
    glacier_kg_m2: float = 1000.0      # snow above this flows off to the ocean (glacier discharge)
    h2o_feedback: bool = True
    wv_lr_feedback: float = 1.30       # W m^-2 K^-1, water vapour + lapse rate feedback
    planck_feedback: float = 3.22      # W m^-2 K^-1, the Planck feedback it is measured against
    water_share: float = 0.5           # water vapour share of the greenhouse at the calibration point
    outgas_earth_kg_co2_yr: float = 0.26e12
    runoff_earth_kg_yr: float = 3.7288e16
    runoff0_kg_m2_yr: float = 22.0     # weathering-effective land runoff of the model's own Earth
    runoff_mean_days: float = 365.25   # e-folding of the running-mean runoff that weathers
    weathering_k: float = 13.7         # K, e-folding of weathering with temperature
    weathering_runoff_exp: float = 0.65
    cloud_max: float = 0.95            # upper bound of the calibrated cloud albedo


PROVENANCE = {
    "heat_ocean": ("reference", "50 m ocean mixed layer, rho c_p h = 1025 x 3990 x 50 = 2.0e8 J m^-2 K^-1 (spec 2.1e8; "
                   "e.g. Hartmann 2016, Global Physical Climatology, 2nd ed., ch. 4)"),
    "heat_land": ("new_rule", "land surface plus the air column above it, 1e7 J m^-2 K^-1 (the air column alone is "
                  "P c_p / g = 1.0e7 for Earth)"),
    "heat_spin": ("new_rule", "numerics: during an accelerated spin-up every cell has this heat capacity and the "
                  "insolation is the year mean (the equilibrium does not depend on C; seasons come back after it)"),
    "spin_chunk_days": ("new_rule", "numerics: length of an accelerated spin-up chunk"),
    "d_per_bar": ("new_rule", "Sellers-type diffusion D = 0.55 W m^-2 K^-1 x P / 1 bar on the unit sphere (spec; "
                  "North, Cahalan & Coakley 1981, Rev. Geophys. 19, 91 fit D = 0.649 for Earth): heat transport "
                  "scales with the atmosphere's mass; it includes the latent heat transport"),
    "air_absorption": ("reference", "the atmosphere absorbs about 75 of 340 W/m^2 of sunlight (Stephens et al. 2012, "
                       "Nature Geosci. 5, 691): the layer's transmittance is (1 - a_c)(1 - 0.22); it sets the light "
                       "at the ground (PAR), the column's energy is I (1 - alpha_p) either way"),
    "albedo_ocean": ("reference", "open ocean 0.06 (Payne 1972, J. Atmos. Sci. 29, 959; the diffuse-light value of "
                     "Briegleb et al. 1986): used when no sun angle is given"),
    "ocean_albedo_zenith": ("reference+new_rule", "direct-beam ocean albedo 0.026 / (mu^1.7 + 0.065) + 0.15 (mu - 0.1)"
                            "(mu - 0.5)(mu - 1) (Briegleb, Minnis, Ramanathan & Harrison 1986, J. Clim. Appl. Meteor. "
                            "25, 214), 0.024 at overhead sun, 0.14 at mu = 0.3; new_rule: evaluated at the day's "
                            "insolation-weighted mean cos zenith (the year's for an accelerated spin-up), for all the "
                            "light under the layer"),
    "albedo_ice": ("reference", "snow and ice 0.6 (Budyko 1969, Tellus 21, 611 used 0.62)"),
    "albedo_land": ("reference", "bare soil and rock 0.25 (soils 0.10-0.35, deserts 0.30-0.40; Hartmann 2016, table 4.2)"),
    "albedo_vegetation": ("reference", "vegetated land 0.15 (forests 0.10-0.18, grass 0.16-0.26; Hartmann 2016)"),
    "ice_k": ("reference", "the ice line at -10 C, 263 K (Budyko 1969; North 1975, J. Atmos. Sci. 32, 2033)"),
    "ice_ramp_k": ("new_rule", "numerics: the ice albedo ramps linearly over 258-268 K instead of a step"),
    "snow_mask_kg_m2": ("new_rule", "10 kg/m^2 (1 cm water) of snow covers the ground fully (masking depths in land "
                        "models are of this order)"),
    "bucket_kg_m2": ("reference", "Manabe 1969 bucket, field capacity 15 cm (Mon. Weather Rev. 97, 739)"),
    "soil_start": ("new_rule", "the bucket starts half full on land (from the ocean)"),
    "beta_share": ("reference", "evaporation efficiency beta = W / (0.75 W_fc), capped at 1 (Manabe 1969)"),
    "transfer_coeff": ("reference", "bulk transfer coefficient 1.15e-3 for evaporation (Dalton number, Large & Pond "
                       "1982, J. Phys. Oceanogr. 12, 464), also used for sensible heat"),
    "wind_m_s": ("new_rule", "5 m/s near-surface wind (Earth: about 7 over the ocean, 3-4 over land; Archer & "
                 "Jacobson 2005, JGR 110, D12110)"),
    "residence_days": ("reference", "8.9 d mean residence time of atmospheric water (van der Ent & Tuinenburg 2017, "
                       "Hydrol. Earth Syst. Sci. 21, 779); background precipitation = vapour / 8.9 d"),
    "degree_day_kg_m2_k": ("reference", "degree-day factor of snow 4 mm per K per day (2.5-11.6 in the review of "
                           "Hock 2003, J. Hydrol. 282, 104)"),
    "glacier_kg_m2": ("new_rule", "snow above 1000 kg/m^2 (1 m water equivalent) leaves as glacier discharge to the "
                      "ocean the same day (no ice-sheet dynamics; it stops the cold trap of a night side or a pole "
                      "from draining the ocean)"),
    "h2o_feedback": ("new_rule", "the grey tau has a water vapour term (spec 2.6: add an H2O term where realistic "
                     "sensitivity matters)"),
    "planck_feedback": ("reference", "Planck feedback -3.22 W m^-2 K^-1 (IPCC AR6 WG1 ch. 7, table 7.10)"),
    "wv_lr_feedback": ("reference", "water vapour + lapse rate feedback 1.30 W m^-2 K^-1 (IPCC AR6 WG1 ch. 7, "
                       "table 7.10, very likely 1.13-1.47)"),
    "water_share": ("reference+new_rule", "water vapour is about half of Earth's greenhouse effect (Schmidt et al. "
                    "2010, JGR 115, D20106); new_rule: tau_w,ref = 0.5 x min(tau, tau_Earth)"),
    "outgas_earth_kg_co2_yr": ("reference", "present-day global volcanic CO2 0.26 Gt/yr (Gerlach 2011, Eos 92, 201); "
                               "in steady state silicate weathering removes the same (Walker, Hays & Kasting 1981, "
                               "JGR 86, 9776)"),
    "runoff_earth_kg_yr": ("reference", "continental discharge 37,288 km^3/yr (Dai & Trenberth 2002, J. Hydrometeor. "
                           "3, 660), 251 mm/yr per m^2 of land: the observed value, reported against the model's "
                           "runoff0 (not used by the weathering law)"),
    "runoff0_kg_m2_yr": ("derived", "22 mm/yr, the weathering-effective land runoff of the model's own Earth: "
                         "(time and land mean of exp((T - 288)/13.7) r^0.65)^(1/0.65) over the running-mean runoff r "
                         "of chain.earth_inputs alone after spin-up (terrain seed 1; 21.8 mm/yr at G = 12, 22.1 at "
                         "G = 24; other terrains give 13-29, so weathering is 0.7-1.2 x outgassing; the tests "
                         "recheck the balance): with it the model's Earth weathers what it outgasses. It is far below "
                         "the model's mean land runoff (about 75 mm/yr) because the power law is concave and many "
                         "land cells hardly run off; the bucket model's land is also drier than Earth's (observed "
                         "251 mm/yr: its land evaporation is potential evaporation without stomatal control)"),
    "runoff_mean_days": ("new_rule", "weathering acts on a running mean of the runoff with a one-year e-folding (a "
                         "cumulative mean over the first year): the daily bucket overflow is zero on most days, and "
                         "the mean of (r/r0)^0.65 of a spiky series is far below (mean r / r0)^0.65"),
    "weathering_k": ("reference", "weathering e-folds every 13.7 K (Walker, Hays & Kasting 1981, JGR 86, 9776); their "
                     "(pCO2 / pCO2,0)^0.3 factor is omitted (spec 2.6)"),
    "weathering_runoff_exp": ("reference", "(runoff / runoff0)^0.65 (Berner 1994, GEOCARB II, Am. J. Sci. 294, 56)"),
    "outgassing": ("new_rule", "volcanic CO2 = Earth's x radiogenic_heat_w_m2 / 0.047 W/m^2 (spec 2.6)"),
    "cloud_max": ("new_rule", "numerics: bound of the cloud albedo calibration"),
    "substellar_point": ("new_rule", "a locked planet's substellar point is lon 0, lat 0 (the terrain is random)"),
    "orbit": ("new_rule", "circular orbit, vernal equinox at day 0; daily-mean insolation over the planet's solar day "
              "(the tick resolves seasons, not the diurnal cycle)"),
    "wtg_exchange": ("derived+reference+new_rule",
                     "Budyko-type exchange gamma (H - theta_e) of every cell with a horizontally uniform free "
                     "troposphere (the weak-temperature-gradient regime of slow rotators), on the moist static energy "
                     "theta_e = theta + L_v q / c_p at the formation's relative humidity (as moist energy-balance "
                     "models diffuse it, Frierson, Held & Zurita-Gotor 2007, J. Atmos. Sci. 64, 1680) times the "
                     "surface's evaporation efficiency (new_rule: Manabe's beta x (1 - snow cover) on land, 1 - ice "
                     "on the ocean, the same factors as the evaporation, so dry land exports no latent heat), H set "
                     "so that "
                     "the exchange conserves energy: gamma = w rho c_p C_H U (the bulk surface flux, Large & Pond "
                     "1982), w = Lambda^2 / (1 + Lambda^2) with the WTG parameter Lambda = N H_p / (Omega R) "
                     "(Pierrehumbert & Hammond 2019, Annu. Rev. Fluid Mech. 51, 275), N^2 = (g / T)(g / c_p - Gamma), "
                     "Gamma = 6.5 K/km x g / g_E (formation's lapse rule), H_p the pressure scale height, Omega the "
                     "sidereal rotation, R the planet's radius; Earth gets w = 0.04, a locked planet w = 1. Without it "
                     "the spec's D alone leaves a locked planet's substellar point near 390 K and its mean 23 K "
                     "below T_s, beyond what a cloud albedo can calibrate"),
    "cloud_albedo": ("derived+new_rule", "calibrated by spin_up so that the lifeless planet's global mean T is "
                     "t_surface_target_k; the first guess makes the insolation-weighted planetary albedo the spec's "
                     "albedo (0.30)"),
}


# ------------------------------------------------------------------------------------------- physics helpers
def e_sat_pa(t_k):
    """Saturation vapour pressure over water (Pa), Magnus form (Alduchov & Eskridge 1996), as formation."""
    t = t_k - 273.15
    return 610.94 * torch.exp(17.625 * t / (t + 243.04))


def vapour_scale_height(t_k, P):
    """H_w(T) = R_v T^2 / (L_v Gamma) (formation's rule), scaled from the spec's value at T_s."""
    return P["vapour_h_ref"][:, None] * (t_k / P["t_target"][:, None]) ** 2


def saturation_column(t_k, P):
    """Saturated column water vapour (kg/m^2): e_sat M_w / (R T) x H_w(T)."""
    return e_sat_pa(t_k) * M_H2O / (K.R_GAS * t_k) * vapour_scale_height(t_k, P)


def latent_temperature(t_k, P, wet=None):
    """Latent part of the moist static energy as a temperature, L_v q / c_p (K), and its derivative in T.

    q = M_w e / (M_w e + mu_d p_d) is the specific humidity of air at relative humidity r b (formation's 0.77
    times the surface's evaporation efficiency b = ``wet`` [W, C] in [0, 1], 1 when not given: e = r b e_sat(T))
    over the planet's dry air pressure p_d; both are [W, C].
    """
    rh = P["rh"][:, None] if wet is None else P["rh"][:, None] * wet
    e = rh * e_sat_pa(t_k)
    t = t_k - 273.15
    de_dt = e * 17.625 * 243.04 / (t + 243.04) ** 2
    a = M_H2O
    b = (P["mu_dry"] * P["p_dry"])[:, None]
    q = a * e / (a * e + b)
    dq_dt = a * b / (a * e + b) ** 2 * de_dt
    scale = K.L_VAPORIZATION / P["air_cp"][:, None]
    return scale * q, scale * dq_dt


def ice_fraction(t_k, rules: ClimateRules):
    """Ice cover share of a cell from its temperature: 1 below ice_k - ramp/2, 0 above ice_k + ramp/2."""
    return ((rules.ice_k + 0.5 * rules.ice_ramp_k - t_k) / rules.ice_ramp_k).clamp(0.0, 1.0)


def planetary_albedo(cloud, surface, air_absorption: float = 0.0):
    """Adding method for one layer (cloud reflectance a_c, transmittance t = (1 - a_c)(1 - a_air)) over the
    surface: a_c + t^2 a_s / (1 - a_c a_s)."""
    t = (1 - cloud) * (1 - air_absorption)
    return cloud + t ** 2 * surface / (1 - cloud * surface)


def surface_shortwave(insolation_w_m2, cloud, surface, air_absorption: float = 0.0):
    """Shortwave reaching the ground through the layer, with the multiple reflections: I t / (1 - a_c a_s)."""
    return insolation_w_m2 * (1 - cloud) * (1 - air_absorption) / (1 - cloud * surface)


def _dalbedo_dcloud(cloud, surface, air_absorption: float = 0.0):
    d = 1 - cloud * surface
    t = (1 - cloud) * (1 - air_absorption)
    return 1 + surface * (-2 * t * (1 - air_absorption) * d + t ** 2 * surface) / d ** 2


def _area_mean(x, area64):
    """Area-weighted global mean (float64 [...]) of x [..., C] (area in steradians)."""
    return (x.double() * area64).sum(-1) / FOUR_PI


# ------------------------------------------------------------------------------------------- insolation
def declination(P, day):
    """Solar declination [W] (rad) at the middle of tick ``day``: sin d = sin(obliquity) sin(orbital phase)."""
    return _declination(P, 2 * math.pi * (float(day) + 0.5) / P["year_days"].double())


def _declination(P, phase):
    return torch.asin(torch.sin(P["obliquity"].double()) * torch.sin(phase))


def _insolation_mu(globe, P, dec):
    """(insolation, insolation-weighted cos zenith) [W, C] float64 for declination dec [W]."""
    S = P["insolation"].double()[:, None]
    x = globe.centers[:, 0].double().clamp_min(0)[None]
    lat = globe.lat.double()[None]
    dec = dec[:, None]
    h0 = torch.acos((-torch.tan(lat) * torch.tan(dec)).clamp(-1.0, 1.0))
    a = torch.sin(lat) * torch.sin(dec)
    b = torch.cos(lat) * torch.cos(dec)
    m1 = (h0 * a + b * torch.sin(h0)).clamp_min(0)                      # int_0^h0 mu dh
    m2 = a * a * h0 + 2 * a * b * torch.sin(h0) + b * b * (0.5 * h0 + 0.25 * torch.sin(2 * h0))   # int_0^h0 mu^2
    mu_rot = torch.where(m1 > 0, m2 / m1.clamp_min(1e-30), torch.zeros_like(m1))
    locked = P["locked"][:, None]
    I = torch.where(locked, S * x, S / math.pi * m1)
    mu = torch.where(locked, x.expand_as(mu_rot), mu_rot)
    return I, mu.clamp(0.0, 1.0)


def _insolation(globe, P, dec):
    return _insolation_mu(globe, P, dec)[0]


def insolation(globe, P, day):
    """Daily-mean top-of-atmosphere insolation [W, C] (W/m^2) on tick ``day``.

    Locked: S max(0, cos theta) from the substellar point (lon 0, lat 0). Rotating: the diurnal mean
    (S / pi)(h0 sin(lat) sin(d) + cos(lat) cos(d) sin(h0)), cos(h0) = -tan(lat) tan(d) clipped to
    [-1, 1] (Berger 1978, J. Atmos. Sci. 35, 2362; Hartmann 2016, eq. 2.15).
    """
    return _insolation(globe, P, declination(P, day)).float()


def sun_angle(globe, P, day):
    """(insolation [W, C], insolation-weighted daily-mean cos zenith [W, C]) on tick ``day`` (float32).

    mu = int mu^2 dh / int mu dh over the daylight hour angles (locked: cos theta); 0 where there is no sun.
    """
    I, mu = _insolation_mu(globe, P, declination(P, day))
    return I.float(), mu.float()


def annual_sun_angle(globe, P, samples: int = 24):
    """Year-mean insolation [W, C] and its insolation-weighted cos zenith [W, C] (float32), over ``samples``
    equally spaced orbital phases."""
    acc = acc_mu = 0.0
    for k in range(samples):
        phase = torch.full((P["W"],), 2 * math.pi * (k + 0.5) / samples, dtype=torch.float64, device=globe.device)
        I, mu = _insolation_mu(globe, P, _declination(P, phase))
        acc, acc_mu = acc + I, acc_mu + I * mu
    mu = torch.where(acc > 0, acc_mu / acc.clamp_min(1e-30), torch.zeros_like(acc))
    return (acc / samples).float(), mu.float()


def annual_insolation(globe, P, samples: int = 24):
    """Year-mean insolation [W, C] (W/m^2): the mean over ``samples`` equally spaced orbital phases."""
    return annual_sun_angle(globe, P, samples)[0]


def ocean_albedo(mu):
    """Direct-beam albedo of open water at cos zenith mu (Briegleb et al. 1986, J. Clim. Appl. Meteor. 25, 214)."""
    return 0.026 / (mu.clamp_min(0) ** 1.7 + 0.065) + 0.15 * (mu - 0.1) * (mu - 0.5) * (mu - 1.0)


# ------------------------------------------------------------------------------------------- parameters
def _e_sat_scalar(t_k: float) -> float:
    t = t_k - 273.15
    return 610.94 * math.exp(17.625 * t / (t + 243.04))


def wtg_weight(spec) -> float:
    """w = Lambda^2 / (1 + Lambda^2), Lambda = N H / (Omega R): the share of a planet's heat exchange that
    goes through a horizontally uniform free troposphere (``PROVENANCE['wtg_exchange']``)."""
    g, cp, T = spec.gravity_m_s2, spec.air_cp_j_kg_k, spec.t_surface_target_k
    gamma_env = K.EARTH_LAPSE_RATE * g / K.G_STANDARD
    n2 = g / T * (g / cp - gamma_env)
    c = math.sqrt(max(n2, 0.0)) * spec.scale_height_m
    omega = 2 * math.pi / spec.rotation_period_s
    lam = c / (omega * spec.radius_m)
    return lam * lam / (1.0 + lam * lam)


def make_params(specs, globe, terrain, radius_m: float = 1.0e4, rules: ClimateRules | None = None) -> dict:
    """Per-world climate parameters from PlanetSpecs (formation) and the habitat terrain (globe).

    Returns a dict of tensors on the globe's device: [W] scalars (float32, gas and ledger values
    float64), [W, C] fields (``land``, ``height_m`` above the sea, ``heat``), plus ``W``, ``rules``,
    ``radius_m`` and ``provenance`` ({name: (tag, note)} for each per-world value).
    """
    from .formation import earth_tau     # set-up only (CPU); formation may use NumPy
    r = rules or ClimateRules()
    specs = list(specs)
    W, dev = len(specs), globe.device
    tau_e = earth_tau()
    rows: dict[str, list] = {}

    def put(name, value):
        rows.setdefault(name, []).append(float(value))

    for s in specs:
        g, cp, T = s.gravity_m_s2, s.air_cp_j_kg_k, s.t_surface_target_k
        p_h2o = s.partial_pressure_pa.get("H2O", 0.0)
        put("insolation", s.insolation_w_m2)
        put("locked", float(s.tidally_locked))
        put("obliquity", s.obliquity_rad)
        put("year_days", s.year_s / DAY_S)
        put("par_fraction", s.par_fraction)
        put("gravity", g)
        put("lapse", g / cp)
        put("t_target", T)
        put("albedo_target", s.albedo)
        put("air_column", (s.surface_pressure_pa - p_h2o) / g)
        put("air_cp", cp)
        put("rh", p_h2o / _e_sat_scalar(T))
        put("p_dry", s.surface_pressure_pa - p_h2o)
        x_h2o = p_h2o / s.surface_pressure_pa
        put("mu_dry", (s.air_molar_mass_kg_mol - x_h2o * M_H2O) / (1.0 - x_h2o))
        put("diffusion", r.d_per_bar * s.surface_pressure_pa / 1e5)
        put("wtg", wtg_weight(s) * s.air_density_kg_m3 * cp * r.transfer_coeff * r.wind_m_s)
        put("vapour_h_ref", s.vapour_scale_height_m)
        put("vapour_ref", s.vapour_column_kg_m2)
        put("co2_ref", s.co2_ref_pa)
        put("tau_per_ln_co2", s.tau_per_ln_co2)
        # water vapour term: d tau / d ln vapour = feedback / (dOLR/dtau x d ln vapour / dT) at the calibration
        tau = s.greenhouse_tau
        if r.h2o_feedback:
            dolr_dtau = 0.75 * K.SIGMA * T ** 4 / (1 + 0.75 * tau) ** 2
            planck = 4 * K.SIGMA * T ** 3 / (1 + 0.75 * tau)
            t_c = T - 273.15
            dlnv_dt = 17.625 * 243.04 / (t_c + 243.04) ** 2 + 1.0 / T      # column ~ e_sat(T) T at fixed H_w/T^2
            k_w = r.wv_lr_feedback / r.planck_feedback * planck / (dolr_dtau * dlnv_dt)
            put("tau_w_ref", r.water_share * min(tau, tau_e))
            put("tau_w_k", k_w)
        else:
            put("tau_w_ref", 0.0)
            put("tau_w_k", 0.0)
        put("tau_rest", tau)
        mol_yr = r.outgas_earth_kg_co2_yr / M_CO2 / EARTH_AREA
        put("outgas", mol_yr * s.radiogenic_heat_w_m2 / K.EARTH_RADIOGENIC_FLUX / DAYS_PER_YEAR)
        put("weathering0", mol_yr / EARTH_LAND_FRACTION / DAYS_PER_YEAR)
        put("runoff0", r.runoff0_kg_m2_yr / DAYS_PER_YEAR)
        put("ocean_layer", s.ocean_layer_m * K.RHO_WATER)

    f32 = {k: torch.tensor(v, dtype=torch.float32, device=dev) for k, v in rows.items()}
    P = dict(f32)
    P["locked"] = f32["locked"] > 0.5
    for k in ("insolation", "outgas", "weathering0", "co2_ref", "tau_per_ln_co2", "t_target", "gravity",
              "air_column", "albedo_target", "ocean_layer"):
        P[k + "64"] = torch.tensor(rows[k], dtype=torch.float64, device=dev)
    P["gas0"] = torch.tensor([[s.partial_pressure_pa[gas] / (s.gravity_m_s2 * m) for gas, m in zip(GASES, M_GAS)]
                              for s in specs], dtype=torch.float64, device=dev)
    P["m_gas"] = torch.tensor(M_GAS, dtype=torch.float64, device=dev)
    land = terrain["land"].to(dev)
    P["land"] = land
    P["height_m"] = torch.where(land, terrain["elevation_m"] - terrain["sea_level_m"][:, None], 0.0).clamp_min(0).float()
    P["heat"] = torch.where(land, r.heat_land, r.heat_ocean).float()
    if "ocean_volume_m3" in terrain:
        P["ocean0"] = terrain["ocean_volume_m3"].double().to(dev) * K.RHO_WATER / (FOUR_PI * radius_m ** 2)
    else:
        P["ocean0"] = P["ocean_layer64"].clone()
    P.update(W=W, rules=r, radius_m=float(radius_m), area64=globe.area64, tau_earth=tau_e)
    P["provenance"] = {
        "insolation": ("derived", "spec insolation_w_m2 (S0 x flux_earth)"),
        "par_fraction": ("derived", "spec par_fraction (Planck integral 400-700 nm of the star)"),
        "gravity": ("derived", "spec gravity_m_s2"), "t_target": ("chain", "spec t_surface_target_k"),
        "albedo_target": ("reference", "spec albedo (0.30, world7's)"),
        "air_cp": ("derived+reference", "spec air_cp_j_kg_k (NIST-JANAF mixture)"),
        "rh": ("new_rule", "relative humidity of the moist static energy: spec p_H2O / e_sat(T_s) (formation's 0.77)"),
        "p_dry": ("derived", "spec surface pressure less p_H2O"),
        "mu_dry": ("derived", "molar mass of the dry air: (mu - x_H2O mu_w) / (1 - x_H2O)"),
        "ocean_layer": ("derived", "spec ocean_layer_m x 1000 kg/m^3 (the real planet's; used when the terrain has no "
                        "ocean volume)"),
        "locked": ("derived", "spec tidally_locked"), "obliquity": ("new_rule", "spec obliquity_rad"),
        "year_days": ("derived", "spec year_s / 86400 (Kepler III)"),
        "lapse": ("derived", "dry adiabat g / c_p (spec 2.6)"),
        "air_column": ("derived", "(P - p_H2O) / g: dry air column, kg/m^2"),
        "diffusion": PROVENANCE["d_per_bar"], "wtg": PROVENANCE["wtg_exchange"],
        "vapour_h_ref": ("derived", "spec vapour_scale_height_m at t_surface_target_k, scaled as T^2"),
        "vapour_ref": ("derived", "spec vapour_column_kg_m2: the initial vapour and the water term's reference"),
        "tau_rest": ("derived", "greenhouse_tau: tau at p_CO2 = co2_ref_pa and vapour = vapour_ref (tau_residual of "
                     "a capped planet is inside it)"),
        "tau_w_ref": PROVENANCE["water_share"],
        "tau_w_k": ("derived+reference", "k_w = (1.30 / 3.22) x Planck / (dOLR/dtau x dln(vapour)/dT) at T_s, Planck = "
                    "4 sigma T^3 / (1 + 0.75 tau), dln/dT from Magnus plus 1/T (the column scales as e_sat T)"),
        "co2_ref": ("derived+new_rule", "spec co2_ref_pa (a cap, not a calibration, when co2_capped)"),
        "tau_per_ln_co2": ("derived", "spec tau_per_ln_co2 (5.35 W/m^2 per e-fold at the calibration point)"),
        "gas0": ("derived", "n_i = p_i / (g M_i) from spec partial_pressure_pa (mol per m^2 of column)"),
        "outgas": PROVENANCE["outgassing"],
        "weathering0": ("reference+derived", "Earth's outgassing per m^2 of land (land 29.1 %): weathering balances "
                        "it at 288 K and runoff0"),
        "runoff0": PROVENANCE["runoff0_kg_m2_yr"],
        "m_gas": ("reference", "molar masses of N2, O2, CO2, Ar (kg/mol) from the IUPAC standard atomic weights, "
                  "conventional values (constants.molar_mass, the repo's data/truth-v5/atomic_weights.json)"),
        "tau_earth": ("derived", "formation.earth_tau(): Earth's grey optical depth (the cap of the water term)"),
        "land": ("derived", "terrain land mask (globe.make_terrain: elevation above the sea level)"),
        "radius_m": ("new_rule", "habitat globe radius (PLANET-SPEC section 1, Config.habitat_radius_m)"),
        "area64": ("derived", "cell solid angles of the habitat grid (globe.area64, steradians)"),
        "ocean0": ("derived", "terrain ocean_volume_m3 x 1000 kg/m^3 over 4 pi R_hab^2 (the habitat's ocean)"),
        "height_m": ("derived", "terrain elevation above the sea level (0 on the ocean)"),
        "heat": (PROVENANCE["heat_ocean"][0] + "+new_rule", "ocean 2.1e8, land 1e7 J m^-2 K^-1"),
    }
    return P


# ------------------------------------------------------------------------------------------- state
@dataclass
class ClimateState:
    """Climate state of W worlds (see the module docstring for units and dtypes)."""
    T: torch.Tensor                 # [W, C] K
    vapour: torch.Tensor            # [W, C] kg/m^2
    soil: torch.Tensor              # [W, C] kg/m^2
    snow: torch.Tensor              # [W, C] kg/m^2
    gas: torch.Tensor               # [W, 4] mol/m^2 (float64)
    ocean: torch.Tensor             # [W] kg/m^2 global mean (float64)
    cloud_albedo: torch.Tensor      # [W] float32
    water0: torch.Tensor            # [W] float64: total water at the ledger's start
    water_external: torch.Tensor    # [W] float64: water that left to other ledgers (bodies), kg/m^2
    gas_external: torch.Tensor      # [W, 4] float64: gas added by other modules (creatures, fires), mol/m^2
    co2_outgassed: torch.Tensor     # [W] float64 mol/m^2, cumulative
    co2_weathered: torch.Tensor     # [W] float64 mol/m^2, cumulative
    runoff_mean: torch.Tensor       # [W, C] kg/m^2 per day, running mean of the runoff (weathering)
    day: int = 0

    def clone(self) -> "ClimateState":
        return replace(self, **{f.name: getattr(self, f.name).clone() for f in fields(self)
                                if isinstance(getattr(self, f.name), torch.Tensor)})

    def state_dict(self) -> dict:
        """The fields as a dict of copies (later changes of the state do not reach it)."""
        return {f.name: (v.clone() if isinstance(v, torch.Tensor) else v)
                for f in fields(self) for v in (getattr(self, f.name),)}

    @classmethod
    def from_state(cls, d: dict) -> "ClimateState":
        return cls(**{k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in d.items()})


def init_state(globe, P) -> ClimateState:
    """The climate at the start: T = t_surface_target_k - Gamma_d h, vapour = the spec's column everywhere, the
    bucket ``soil_start`` full on land (taken from the ocean), no snow, the spec's gas columns, and the first-guess
    cloud albedo (the insolation-weighted planetary albedo is the spec's albedo)."""
    r: ClimateRules = P["rules"]
    W, dev = P["W"], globe.device
    T = P["t_target"][:, None] - P["lapse"][:, None] * P["height_m"]
    vapour = P["vapour_ref"][:, None].expand(W, globe.C).clone()
    soil = torch.where(P["land"], r.soil_start * r.bucket_kg_m2, 0.0).float()
    snow = torch.zeros(W, globe.C, device=dev)
    ocean = P["ocean0"] - _area_mean(soil, globe.area64)
    state = ClimateState(T=T.float(), vapour=vapour, soil=soil, snow=snow, gas=P["gas0"].clone(), ocean=ocean,
                         cloud_albedo=torch.zeros(W, device=dev),
                         water0=torch.zeros(W, dtype=torch.float64, device=dev),
                         water_external=torch.zeros(W, dtype=torch.float64, device=dev),
                         gas_external=torch.zeros(W, 4, dtype=torch.float64, device=dev),
                         co2_outgassed=torch.zeros(W, dtype=torch.float64, device=dev),
                         co2_weathered=torch.zeros(W, dtype=torch.float64, device=dev),
                         runoff_mean=torch.zeros(W, globe.C, device=dev))
    state.cloud_albedo = first_cloud_guess(globe, P, state)
    state.water0 = water_total(state, globe)
    return state


def first_cloud_guess(globe, P, state, plant_cover=None):
    """Cloud albedo [W] that makes the year-mean insolation-weighted planetary albedo the spec's albedo."""
    r: ClimateRules = P["rules"]
    I, mu = annual_sun_angle(globe, P)
    I = I.double() * globe.area64
    a_s = surface_albedo(state, P, plant_cover, mu).double()
    lo = torch.zeros(P["W"], dtype=torch.float64, device=globe.device)
    hi = torch.full_like(lo, r.cloud_max)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        ap = (planetary_albedo(mid[:, None], a_s, r.air_absorption) * I).sum(-1) / I.sum(-1).clamp_min(1e-30)
        low = ap < P["albedo_target64"]
        lo, hi = torch.where(low, mid, lo), torch.where(low, hi, mid)
    return (0.5 * (lo + hi)).float()


# ------------------------------------------------------------------------------------------- diagnostics
def partial_pressures(state: ClimateState, P) -> torch.Tensor:
    """Partial pressures [W, 4] (Pa, float64) of N2, O2, CO2, Ar: p_i = n_i g M_i."""
    return state.gas * P["gravity64"][:, None] * P["m_gas"]


def vapour_pressure(state: ClimateState, P) -> torch.Tensor:
    """Surface water vapour partial pressure [W, C] (Pa): vapour / H_w(T) x R T / M_w."""
    return state.vapour.clamp_min(0) / vapour_scale_height(state.T, P) * K.R_GAS * state.T / M_H2O


def pressures(state: ClimateState, globe, P) -> dict:
    """{gas: [W] Pa (float64)} for N2, O2, CO2, Ar, the global-mean H2O, and the total."""
    p = partial_pressures(state, P)
    out = {gas: p[:, i] for i, gas in enumerate(GASES)}
    out["H2O"] = _area_mean(vapour_pressure(state, P), globe.area64)
    out["total"] = p.sum(-1) + out["H2O"]
    return out


def surface_albedo(state: ClimateState, P, plant_cover=None, mu=None):
    """Surface albedo [W, C]: ocean (by the cos zenith ``mu`` [W, C] when given and ``ocean_albedo_zenith``,
    else 0.06), bare land and vegetation (``plant_cover`` in [0, 1]), snow, then ice."""
    r: ClimateRules = P["rules"]
    land_a = torch.full_like(state.T, r.albedo_land)
    if plant_cover is not None:
        land_a = land_a + (r.albedo_vegetation - r.albedo_land) * plant_cover.clamp(0, 1)
    f_snow = (state.snow / r.snow_mask_kg_m2).clamp(0, 1)
    land_a = land_a + (r.albedo_ice - land_a) * f_snow
    if mu is not None and r.ocean_albedo_zenith:
        sea = ocean_albedo(mu).to(land_a.dtype)
    else:
        sea = torch.full_like(land_a, r.albedo_ocean)
    a = torch.where(P["land"], land_a, sea)
    return a + (r.albedo_ice - a) * ice_fraction(state.T, r)


def optical_depth(state: ClimateState, P):
    """Grey optical depth [W, C]: tau_rest + k ln(p_CO2 / p_ref) + max(-tau_w,ref, k_w ln(vapour / vapour_ref))."""
    p_co2 = (state.gas[:, I_CO2] * P["gravity64"] * M_CO2).clamp_min(1e-12)
    tau_co2 = (P["tau_per_ln_co2"].double() * torch.log(p_co2 / P["co2_ref64"])).float()
    tau = (P["tau_rest"] + tau_co2)[:, None].expand_as(state.T)
    if P["rules"].h2o_feedback:
        rel = state.vapour.clamp_min(1e-9) / P["vapour_ref"][:, None]
        tau = tau + torch.maximum(P["tau_w_k"][:, None] * torch.log(rel), -P["tau_w_ref"][:, None])
    return tau.clamp_min(0)


def water_total(state: ClimateState, globe) -> torch.Tensor:
    """Total water [W] (float64, global-mean kg/m^2): ocean + area means of vapour, soil and snow."""
    return state.ocean + _area_mean(state.vapour + state.soil + state.snow, globe.area64)


def water_ledger(state: ClimateState, globe) -> torch.Tensor:
    """Water ledger error [W] (float64, kg/m^2): stores + water handed to other ledgers - the start."""
    return water_total(state, globe) + state.water_external - state.water0


def reset_water_ledger(state: ClimateState, globe) -> ClimateState:
    return replace(state, water0=water_total(state, globe), water_external=torch.zeros_like(state.water_external))


def _cells(state, w, cell, kg):
    dev = state.T.device
    w = torch.as_tensor(w, device=dev).long().reshape(-1)
    cell = torch.as_tensor(cell, device=dev).long().reshape(-1)
    kg = torch.as_tensor(kg, dtype=torch.float64, device=dev).reshape(-1).clamp_min(0)
    return w, cell, (kg.expand(w.shape[0]) if kg.numel() == 1 else kg)


def take_water(state: ClimateState, globe, P, w, cell, kg_requested, store: str = "soil"):
    """Water taken by other modules (drinking): kg per request ([N] world, cell) from ``soil`` (shared in
    proportion when requests on one cell exceed it) or from the ``ocean`` store. The state object is updated (its
    fields are rebound to new tensors; earlier states and state_dicts keep their values). Returns kg given [N]
    (float64), each request's share of what actually left the float32 bucket; the water is booked in
    ``water_external``."""
    w, cell, kg = _cells(state, w, cell, kg_requested)
    W, C = state.T.shape
    cell_m2 = globe.area64[cell] * P["radius_m"] ** 2
    per_m2 = kg / cell_m2
    if store == "ocean":
        out = torch.zeros(W, dtype=torch.float64, device=kg.device).index_add_(0, w, kg)
        out = out / (FOUR_PI * P["radius_m"] ** 2)
        state.ocean = state.ocean - out
        state.water_external = state.water_external + out
        return kg
    if store != "soil":
        raise ValueError("store is 'soil' or 'ocean'")
    idx = w * C + cell
    total = torch.zeros(W * C, dtype=torch.float64, device=kg.device).index_add_(0, idx, per_m2)
    have = state.soil.reshape(-1).double()
    left = torch.where(total > have, torch.zeros_like(have), have - total).float()
    removed = have - left.double()                      # what left the float32 bucket, exactly
    state.soil = left.view(W, C)
    state.water_external = state.water_external + (removed.view(W, C) * globe.area64).sum(-1) / FOUR_PI
    return per_m2 / total[idx].clamp_min(1e-300) * removed[idx] * cell_m2


def give_water(state: ClimateState, globe, P, w, cell, kg, store: str = "vapour"):
    """Water returned by other modules (breath, sweat, urine, a body's water at death): kg per request into
    ``vapour`` or ``soil`` of the cell (soil above the bucket runs off on the next step); booked as negative
    ``water_external``. The state object is updated by rebinding its fields (see ``take_water``)."""
    w, cell, kg = _cells(state, w, cell, kg)
    W, C = state.T.shape
    if store not in ("vapour", "soil"):
        raise ValueError("store is 'vapour' or 'soil'")
    before = getattr(state, store).reshape(-1).double()
    after = before.index_add(0, w * C + cell, kg / (globe.area64[cell] * P["radius_m"] ** 2)).float()
    setattr(state, store, after.view(W, C))
    added = ((after.double() - before).view(W, C) * globe.area64).sum(-1) / FOUR_PI
    state.water_external = state.water_external - added


def add_gas(state: ClimateState, delta_mol_m2) -> ClimateState:
    """Add gas [W, 4] (mol per m^2 of the global column, negative to remove) from another module (respiration,
    fires); it is booked in ``gas_external`` for the carbon and oxygen ledgers. The state object is updated by
    rebinding ``gas`` and ``gas_external`` (earlier states keep their values). Returns the state."""
    delta = torch.as_tensor(delta_mol_m2, dtype=torch.float64, device=state.gas.device)
    state.gas = state.gas + delta
    state.gas_external = state.gas_external + delta
    return state


# ------------------------------------------------------------------------------------------- the step
def _diffuse_steps(globe, P, kappa, key):
    """Sub-steps of the explicit diffusion, worked out once per ``key`` (the kappas only change with the
    accelerate flag) and cached in P, so the day's step does not ask the device for kappa's range again."""
    if key is None:
        return None
    if key not in P:
        P[key] = max(1, math.ceil(float(kappa.double().max()) * globe.lap_radius))
    return P[key]


def transport(T, vapour, heat, globe, P, wet=None, steps_key=None):
    """Horizontal transport of one day: (T, vapour) [W, C] -> (T, vapour).

    Heat moves as theta = T + Gamma_d h (the sea-level equivalent): Sellers diffusion D lap theta (explicit
    sub-steps in ``globe.diffuse``), then the implicit exchange with the free troposphere,
    C d theta = k (H - theta_e) with k = gamma dt, theta_e = theta + L_v q(r b e_sat(T)) / c_p linearised in T
    (b = ``wet``, the surface's evaporation efficiency, 1 when not given), and H chosen so that
    sum(area C d theta) = 0. Vapour moves with the same mixing over the air column's heat capacity. Both conserve
    their content (sum(area C theta), sum(area vapour)) up to float32 rounding. ``steps_key`` names the P entry
    that caches the diffusion's sub-step count (``step`` passes one per accelerate flag).
    """
    area64 = globe.area64
    lift = P["lapse"][:, None] * P["height_m"]
    D = P["diffusion"][:, None]
    air = (P["air_column"] * P["air_cp"])[:, None]
    kappa = torch.stack((D * DAY_S / heat, (D * DAY_S / air).expand_as(T)))
    lift64 = lift.double()
    # float64 throughout (globe.diffuse keeps the field's dtype), one rounding to float32 at the end
    steps = _diffuse_steps(globe, P, kappa, steps_key)
    both = globe.diffuse(torch.stack((T.double() + lift64, vapour.double())), kappa, steps=steps,
                         check=steps is None)
    theta, v = both[0], both[1]
    lat_k, dlat_dt = latent_temperature(theta - lift64, P, wet)
    k = (P["wtg"] * DAY_S).double()[:, None]
    h64 = heat.double()
    m = 1.0 + dlat_dt
    theta_e = theta + lat_k
    wgt = area64 * h64 / (h64 + k * m)
    big_h = (wgt * theta_e).sum(-1, keepdim=True) / wgt.sum(-1, keepdim=True)
    theta = theta + k * (big_h - theta_e) / (h64 + k * m)
    rv = k / air.double()
    v_bar = (v * area64).sum(-1, keepdim=True) / FOUR_PI
    v = (v + rv * v_bar) / (1 + rv)
    return (theta - lift64).to(T.dtype), v.to(vapour.dtype)


def step(state: ClimateState, globe, P, plant_cover=None, accelerate: bool = False):
    """One day of climate. Returns (new state, diagnostics).

    ``plant_cover`` [W, C] in [0, 1] is the vegetation cover (``biosphere.cover``); ``accelerate`` gives every
    cell the spin-up heat capacity and the year-mean insolation (spin-up only; cached in ``P``).
    Diagnostics: insolation, mu (insolation-weighted cos zenith), surface_sw (down at the surface), absorbed,
    albedo (surface), planetary_albedo [W], tau, olr (the linearised OLR at the step's new temperature, taken from
    the stored float32 temperature so that the column energy sum(area C T) changes by toa_imbalance x dt up to the
    transport's one float32 rounding), evaporation, precipitation, snowfall,
    melt, runoff, glacier ([W, C] kg/m^2 per day), weathering and outgassing [W] (mol CO2/m^2 that day),
    pressures {gas: [W]}, mean_t, toa_imbalance, planck, mean_insolation [W] (float64).
    """
    r: ClimateRules = P["rules"]
    area64 = globe.area64
    land = P["land"]
    T, v, soil, snow = state.T, state.vapour, state.soil, state.snow
    heat = torch.full_like(T, r.heat_spin) if accelerate else P["heat"]

    # 1. radiation inputs
    if accelerate:
        if "_annual_sun" not in P:
            P["_annual_sun"] = annual_sun_angle(globe, P)
        I, mu = P["_annual_sun"]
    else:
        I, mu = sun_angle(globe, P, state.day)
    a_s = surface_albedo(state, P, plant_cover, mu)
    cloud = state.cloud_albedo[:, None]
    a_p = planetary_albedo(cloud, a_s, r.air_absorption)
    absorbed = I * (1 - a_p)
    surface_sw = surface_shortwave(I, cloud, a_s, r.air_absorption)
    tau = optical_depth(state, P)
    eps = 1.0 / (1.0 + 0.75 * tau)
    olr0 = eps * K.SIGMA * T ** 4

    # 2. evaporation (the bulk formula, capped so that it cannot overshoot saturation)
    f_ice = ice_fraction(T, r)
    f_snow = (snow / r.snow_mask_kg_m2).clamp(0, 1)
    v_sat = saturation_column(T, P)
    rate = (r.transfer_coeff * r.wind_m_s * DAY_S / vapour_scale_height(T, P)).clamp_max(1.0)
    demand = rate * (v_sat - v).clamp_min(0)
    beta = (soil / (r.beta_share * r.bucket_kg_m2)).clamp(0, 1)
    wet = torch.where(land, beta * (1 - f_snow), 1 - f_ice)          # the surface's evaporation efficiency
    evap = torch.where(land, torch.minimum(demand * wet, soil), demand * wet)
    evap_land = torch.where(land, evap, 0.0)
    evap_ocean = evap - evap_land
    v = v + evap
    soil = soil - evap_land

    # 3. radiation, linearised backward Euler
    planck = 4 * eps * K.SIGMA * T ** 3
    T1 = T + DAY_S * (absorbed - olr0) / (heat + DAY_S * planck)
    # the OLR the step actually emits: the linearised OLR at the new temperature, olr0 + planck (T1 - T), taken
    # as absorbed - C (T1 - T) / dt with the stored float32 T1 (equal up to its rounding), so that
    # sum(area C dT) = sum(area (absorbed - olr)) dt holds to float64 (the transport conserves energy)
    olr64 = absorbed.double() - heat.double() * (T1.double() - T.double()) / DAY_S

    # 4. transport of heat and vapour
    T2, v = transport(T1, v, heat, globe, P, wet, "_diffuse_steps_spin" if accelerate else "_diffuse_steps")

    # 5. precipitation, snow, melt, runoff
    keep = math.exp(-1.0 / r.residence_days)
    p_bg = v.clamp_min(0) * (1 - keep)
    v = v - p_bg
    p_sat = (v - saturation_column(T2, P)).clamp_min(0)
    v = v - p_sat
    precip = p_bg + p_sat
    cold = T2 < T_FREEZE
    snowfall = torch.where(land & cold, precip, 0.0)
    rain_land = torch.where(land & ~cold, precip, 0.0)
    to_ocean = torch.where(land, 0.0, precip)
    snow = snow + snowfall
    melt = torch.minimum(snow, r.degree_day_kg_m2_k * (T2 - T_FREEZE).clamp_min(0))
    snow = snow - melt
    glacier = (snow - r.glacier_kg_m2).clamp_min(0)
    snow = snow - glacier
    soil = soil + rain_land + melt
    runoff = (soil - r.bucket_kg_m2).clamp_min(0)
    soil = soil - runoff
    ocean = state.ocean + _area_mean(to_ocean + runoff + glacier - evap_ocean, area64)

    # 6. carbonate-silicate cycle (global CO2 column), on the running-mean runoff (a cumulative mean over
    # the first year, then an exponential mean with the e-folding runoff_mean_days)
    share = max(1.0 - math.exp(-1.0 / r.runoff_mean_days), 1.0 / (state.day + 1))
    runoff_mean = state.runoff_mean + share * (runoff - state.runoff_mean)
    weath_cell = torch.where(land, P["weathering0"][:, None] * torch.exp((T2 - 288.0) / r.weathering_k)
                             * (runoff_mean / P["runoff0"][:, None]) ** r.weathering_runoff_exp, 0.0)
    weathering = torch.minimum(_area_mean(weath_cell, area64), state.gas[:, I_CO2])
    outgassing = P["outgas64"]
    gas = state.gas.clone()
    gas[:, I_CO2] = gas[:, I_CO2] + outgassing - weathering

    new = replace(state, T=T2, vapour=v, soil=soil, snow=snow, gas=gas, ocean=ocean,
                  co2_outgassed=state.co2_outgassed + outgassing, co2_weathered=state.co2_weathered + weathering,
                  runoff_mean=runoff_mean, day=state.day + 1)
    mean_i = _area_mean(I, area64)
    diag = {"insolation": I, "mu": mu, "surface_sw": surface_sw, "absorbed": absorbed, "albedo": a_s,
            "planetary_albedo": 1 - _area_mean(absorbed, area64) / mean_i.clamp_min(1e-30),
            "tau": tau, "olr": olr64.float(), "evaporation": evap, "precipitation": precip, "snowfall": snowfall,
            "melt": melt, "runoff": runoff, "glacier": glacier, "weathering": weathering, "outgassing": outgassing,
            "pressures": pressures(new, globe, P), "mean_t": _area_mean(T2, area64),
            "toa_imbalance": _area_mean(absorbed.double() - olr64, area64),
            "planck": _area_mean(planck, area64), "mean_insolation": mean_i,
            "dalbedo_dcloud": _area_mean(I * _dalbedo_dcloud(cloud, a_s, r.air_absorption), area64) / mean_i.clamp_min(1e-30)}
    return new, diag


def run(state: ClimateState, globe, P, days: int, plant_cover=None, accelerate: bool = False):
    """``days`` steps; returns (state, time-mean global T [W], time-mean TOA imbalance [W])."""
    t_sum = torch.zeros(P["W"], dtype=torch.float64, device=globe.device)
    n_sum = torch.zeros_like(t_sum)
    for _ in range(days):
        state, diag = step(state, globe, P, plant_cover, accelerate)
        t_sum += diag["mean_t"]
        n_sum += diag["toa_imbalance"]
    return state, t_sum / max(days, 1), n_sum / max(days, 1)


# ------------------------------------------------------------------------------------------- spin-up
def spin_up(state: ClimateState, globe, P, plant_cover=None, fast_chunks: int = 8, slow_chunks: int = 4,
            tol_k: float = 0.1, calibrate: bool = True):
    """Spin the climate up and calibrate the cloud albedo (PLANET-SPEC 2.6). Returns (state, report).

    First at most ``fast_chunks`` accelerated chunks of ``rules.spin_chunk_days`` (every cell at the spin-up heat
    capacity, year-mean insolation) until every world's projected error is below 0.25 K and its chunk-mean T within
    0.3 K of the target, then at most ``slow_chunks`` chunks with the
    real heat capacity and seasons until every world is within ``tol_k``. A slow chunk lasts the longest year of the
    rotating worlds (at most 730 days; ``spin_chunk_days`` when all are locked), and each world averages over the
    last whole number of its own years in it. After each chunk the equilibrium global mean is projected as
    T_eq = mean(T) + mean(N) / lambda (N the TOA imbalance, lambda the model's Planck response less the water
    vapour feedback) and, if ``calibrate``, the cloud albedo moves by lambda (T_eq - T_s) / mean(I) /
    (d alpha_p / d a_c), clipped to [0, cloud_max]. Water is conserved throughout (the ledger is untouched).
    ``report``: per chunk the cloud albedo used, mean_t, t_eq and error_k (T_eq - t_surface_target_k); at the end
    cloud_albedo, t_eq, error_k and chunks_run.
    """
    r: ClimateRules = P["rules"]
    years = P["year_days"].double().round().clamp_min(1)
    rotating = ~P["locked"]
    slow_days = r.spin_chunk_days
    if bool(rotating.any()):
        slow_days = int(min(730, max(r.spin_chunk_days, float(years[rotating].max()))))
    whole = torch.where(rotating & (years <= slow_days), torch.floor(slow_days / years) * years,
                        torch.full_like(years, float(slow_days)))
    report = {"chunks": [], "slow_chunk_days": slow_days}
    target = P["t_target64"]
    wv = r.wv_lr_feedback / r.planck_feedback if r.h2o_feedback else 0.0
    fast = slow = 0
    while True:
        accelerate = fast < fast_chunks
        days = r.spin_chunk_days if accelerate else slow_days
        window = torch.full_like(years, float(days)) if accelerate else whole
        sums = torch.zeros(5, P["W"], dtype=torch.float64, device=globe.device)
        for d in range(days):
            state, diag = step(state, globe, P, plant_cover, accelerate)
            use = (days - d <= window).double()
            sums += use * torch.stack((diag["mean_t"], diag["toa_imbalance"], diag["planck"],
                                       diag["mean_insolation"], diag["dalbedo_dcloud"]))
        mean_t, imbalance, planck, mean_i, slope = sums / window
        lam = (planck * (1 - wv)).clamp_min(0.3)
        t_eq = mean_t + imbalance / lam
        err = t_eq - target
        report["chunks"].append({"accelerated": accelerate, "cloud_albedo": state.cloud_albedo.tolist(),
                                 "mean_t": mean_t.tolist(), "t_eq": t_eq.tolist(), "error_k": err.tolist()})
        if calibrate:
            step_c = lam * err / mean_i.clamp_min(1e-9) / slope.clamp_min(1e-3)
            state = replace(state, cloud_albedo=(state.cloud_albedo.double() + step_c).clamp(0.0, r.cloud_max).float())
        worst = float(err.abs().max())
        if accelerate:
            # leave the fast phase once the cloud is calibrated and the state itself is near equilibrium (the
            # ocean's real heat capacity would take years to close a remaining gap)
            settled = worst < 0.25 and float((mean_t - target).abs().max()) < 0.3
            fast = fast_chunks if settled else fast + 1
        else:
            slow += 1
            if worst < tol_k or slow >= slow_chunks:
                break
    report.update(cloud_albedo=state.cloud_albedo.tolist(), t_eq=t_eq.tolist(), error_k=err.tolist(),
                  chunks_run=len(report["chunks"]))
    return state, report
