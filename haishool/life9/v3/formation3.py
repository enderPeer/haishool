"""Formation v3: derive what version 2 assumed (PLANET-V3-SPEC section 2).

:func:`build_planet3` turns chain inputs into a :class:`PlanetSpec3`, a :class:`planet.formation.PlanetSpec`
with the volatile budgets added. Star, orbit, rotation, crust and the interior formulas are version 2's
(``planet.formation``, imported, not edited). What version 2 assumed is derived here:

water      The accreted solids are hydrated by the disc temperature at the orbit,
           T = BLACK_BODY L^(1/4) / sqrt(a) (world7's own 278.3 K, ``haishool.cosmos.world``). Their
           water, carbon and nitrogen mass fractions are anchored to meteorite classes (enstatite,
           ordinary, CM and CI chondrites; reference contents, new_rule anchor temperatures) and
           interpolated in log10 against T (clamped beyond the outer anchors). Beyond world7's frost
           line (T < BLACK_BODY / sqrt(2.7 au) = 169.4 K) ice joins the rock in world7's own ice-to-rock
           proportion (ICE_FACTOR 3). Two Earth calibrations bring volatiles to the surface. The degassed
           share, calibrated so that chain.earth_inputs gets Earth's surface water (2.3e-4 of its mass),
           brings that share of the bulk water and carbon to the surface. Nitrogen has its own degassed
           share, calibrated so that the model Earth's air holds Earth's N2 partial pressure (x_N2 0.78084
           of 101,325 Pa = 79.1 kPa, dry air at sea level); the single water share gave it 2.3x Earth's N2
           (2.05 bar of air), ``nitrogen_calibration="water"`` keeps that earlier reading. The rest stays
           undegassed (kept in the interior or lost; not distinguished). Carbon and nitrogen are scaled by
           the chain cloud's C and N per unit of rock (Fe + MgO + SiO2) relative to the sun's.
           ``METEORITE_CLASSES_SITES`` is a reported sensitivity: anchors at the black-body temperatures
           where the classes formed.
atmosphere N2: all degassed nitrogen. Ar: K-40 decay over the planet's age. K (and U, Th, the radiogenic
           heat) per kg of rock is the bulk silicate Earth's x the cloud's lithophile scale
           (other / rock)_cloud / (other / rock)_sun, the same per-rock rule as C and N (the mantle is
           rock; version 2's Z / Z_sun also counts C, N, O and Ne). Half of the radiogenic argon is
           degassed (Earth's). O2: the chain's bodies era (as before). CO2: the carbon-silicate
           steady state: weathering W = outgassing V. V = V_E (q / q_E) (C_s / C_s,E) (radiogenic heat
           flux q, degassed carbon per area C_s). W (default ``weathering="land_seafloor"``) is
           continental weathering on the land, (1 - s) (land / land_E) (p / p_E)^0.3 exp((T - T_E) / 13.7 K)
           (Walker, Hays & Kasting 1981 per land area, Abbot et al. 2012), plus seafloor weathering
           s (q / q_E) (p / p_E)^0.25 exp(-E / R (1 / T - 1 / T_E)) with its own pCO2 and temperature laws
           (s = 0.15 of Earth's silicate sink). A planet without land therefore weathers only on its
           seafloor. ``weathering="global"`` (the published WHAK law on any planet, land or not) is
           reported in every spec; ``"land"`` drops the seafloor term (a bound). The solved p is capped
           by the degassed carbon (all of it in the air); the rest is carbonate rock. The weathering
           temperature is the planet's own T_s, the fixed point of the loop CO2 -> greenhouse -> T_s ->
           weathering -> CO2 (``weathering_t="self_consistent"``, the default); ``"chain"`` weathers at
           the chain's temperature (the spec's literal wording); both readings are reported. The dry
           gases are a well-mixed column: p_i = n_i g mu_dry.
greenhouse tau results from that CO2 and the water vapour at the planet's own temperature:
           tau = tau_cloud + tau_other + tau_c,E (u_c P^b / (u_c,E P_cal^b))^n_c
           + tau_w,E (u_w P^b / (u_w,E P_cal^b))^n_w (u: absorber columns, kg/m^2; P the dry pressure,
           b = 0.5 pressure broadening, Lorentz widths ~ P, checked against Goldblatt et al. 2009's warming
           of doubled N2). Earth's grey tau (world7's 287.6 K) is split by Schmidt et
           al. 2010 (vapour 0.50, CO2 0.20, clouds 0.25, other gases 0.05); the cloud term is held at
           Earth's (a placeholder: formation has no cloud physics), the other gases (O3, CH4, N2O, made by
           a biosphere) scale with the chain's oxygen_pal. P_cal is the model Earth's dry pressure (1.013
           bar with its N2 calibrated on Earth's), so the model Earth returns 287.6 K. n_c makes a
           doubling of Earth's CO2 worth Myhre's 5.35 ln 2 = 3.71 W/m^2 and n_w gives IPCC AR6's water
           vapour + lapse rate feedback (1.30 of the Planck 3.22) at Earth. N2-N2 and CO2 collision-induced
           absorption is missing: ``pressure_beyond_fit`` flags a dry pressure above twice P_cal.
           T_s is the lowest root of sigma T^4 = sigma T_eq^4 (1 + 0.75 tau(T)) above T_eq (a planet
           with none below ``t_max_k`` is flagged ``runaway``). The chain's T_s becomes a check:
           ``chain_t_s_mismatch_k`` = T_s - (chain t_eq + greenhouse). ``frozen_mean`` flags T_s below
           273.15 K: the formation keeps albedo 0.30, liquid-water vapour and active weathering there
           (no ice-albedo feedback, no frozen-surface shutdown), so the v3 climate, not formation,
           decides snowball outcomes and the formation T_s is no target for such planets. Each spec
           reports the warming of the clouds, of the other gases and of the broadening.
relief     Real scale. Peaks are strength-limited, h = 8,848.86 m (Everest) x g_E / g. Isostasy:
           the continental platform rides at sea level and the water-loaded ocean basins have a depth
           D(g) = D_E (s_r g_E / g + 1 - s_r): the ridge depth (2,600 m of Earth's 3,682 m, s_r) is an Airy
           balance of crust whose thickness is fixed in pressure (~1/g); the thermal subsidence of the
           plates (the rest) does not depend on g. D_E is calibrated so that Earth's ocean covers 70.9 %.
           The ocean volume sets the ocean's area: ocean_fraction = min(1, V / (4 pi R^2 D)); the readings
           with D fixed and D ~ 1/g are reported. relief_m = peak + D. :func:`make_terrain3` builds the
           globe's terrain with that ocean fraction (version 2's ``globe.make_terrain`` spreads relief_m
           linearly and puts the sea elsewhere; do not feed it relief_m).

:func:`check_spec3` keeps every version-2 check (``planet.formation.check_spec``) except the ones
listed in ``SUPERSEDED`` (the gates that tie T_s and tau backwards to the chain's T_s, the sign of the
signed mismatch, and the grey target of a flagged runaway planet), and adds the water, carbon, nitrogen
and argon mass budgets, the two degassed shares (on Earth's inputs also the two calibrations they meet:
Earth's surface water and Earth's N2), the potassium and the heat flux, the disc temperature, the steady
state, the greenhouse, the flags and the relief.

Python floats (float64) throughout; no RNG beyond version 2's rotation draw (and the terrain generator's
torch.Generator in :func:`make_terrain3`). Every constant has a ``PROVENANCE`` entry, every spec field a
``provenance`` entry.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np
import torch

from haishool.cosmos import planets as level3
#: world7's BLACK_BODY (haishool/cosmos/world.py:233), the same formula and rounding, computed here so that
#: formation3 does not import cosmos.world (which needs scipy, absent on some hosts); a test checks equality.
BLACK_BODY = round(5772.0 * math.sqrt(6.957e8 / (2.0 * 1.495978707e11)), 1)
from haishool.life9.planet import chain as chain9
from haishool.life9.planet import constants as C
from haishool.life9.planet import crafting as cr
from haishool.life9.planet import formation as F
from haishool.life9.planet import globe as gb
from haishool.life9.planet.globe import EARTH_OCEAN_FRACTION

FORMATION3_VERSION = "life9-formation-v3"

# ---------------------------------------------------------------------------------------------
# constants (each has a PROVENANCE entry)

#: world7's Earth: T_eq 254.6 K + greenhouse 33 K (haishool/cosmos/planets.py), the WHAK reference T
T_EARTH_K = level3.T_EQ_EARTH + level3.GREENHOUSE_EARTH
#: world7's frost line at 1 L_sun (2.7 au x sqrt(L)): the disc temperature there does not depend on L
FROST_T_K = BLACK_BODY / math.sqrt(level3.FROST_AU)
#: world7's solids beyond the frost line: rock x ICE_FACTOR, so ice is 1 - 1/ICE_FACTOR of them
ICE_SHARE = 1.0 - 1.0 / level3.ICE_FACTOR
#: Everest, 8,848.86 m (China-Nepal joint survey announced 8 Dec 2020)
EVEREST_M = 8848.86
#: Earth's mean surface gravity GM_E / R_E^2 (IAU 2015 GM, IUGG mean radius), the g of the Everest calibration
G_EARTH_SURFACE = C.GM_EARTH / C.R_EARTH ** 2
#: K-40 decay constants of geochronology (Steiger & Jaeger 1977, EPSL 36, 359): lambda_e 0.581e-10 /yr to Ar-40,
#: lambda_beta 4.962e-10 /yr to Ca-40; the branch to Ar-40 is their ratio
K40_AR_BRANCH = 0.581 / (0.581 + 4.962)
#: K-40 atom fraction of potassium today, 0.0117 % (IUPAC isotopic abundances, Meija et al. 2016, Pure Appl.
#: Chem. 88, 293)
K40_ATOM_FRACTION = 1.17e-4
#: mean depth of Earth's ocean, m (Charette & Smith 2010, Oceanography 23(2), 112): the reference the derived
#: basin depth is reported against, and the denominator of the ridge share
EARTH_MEAN_OCEAN_DEPTH_M = 3682.0
#: melting point of water ice at 1 atm, K (the frozen_mean flag)
T_FREEZE_K = 273.15
#: Earth's N2 partial pressure, Pa: dry-air mole fraction 0.78084 (U.S. Standard Atmosphere 1976) x 101,325 Pa,
#: 79.1 kPa at sea level; the target of the nitrogen degassed share on the model Earth
EARTH_N2_PA = C.EARTH_AIR_MOLE_FRACTION["N2"] * C.P_STANDARD
_P_CO2_EARTH = C.EARTH_AIR_MOLE_FRACTION["CO2"] * C.P_STANDARD   # 28.17 Pa, pre-industrial 278 ppm
#: total pressures (Pa) within which the 1-bar flammability limit crafting.X_O2_MIN is taken as measured: outside it
#: a world's fire_possible is reported as uncertain (None) with fire_beyond_fit
FIRE_FIT_PA = (5.0e4, 2.0e5)
_P_O2_EARTH = C.EARTH_AIR_MOLE_FRACTION["O2"] * C.P_STANDARD
_M = {gas: C.molar_mass(gas) for gas in F.GASES}
_M_C, _M_AR, _M_K = C.element_mass("C"), C.element_mass("Ar"), C.element_mass("K")
_DRY = ("N2", "O2", "CO2", "Ar")
_WEATHERING = ("land_seafloor", "global", "land")
_OTHER_GASES = ("oxygen_pal", "earth", "none")
_BASIN_RULES = ("split", "fixed", "inverse_g")
_NITROGEN_CALIBRATIONS = ("earth_n2", "water")


@dataclass(frozen=True)
class MeteoriteClass:
    """One anchor of the solids' volatile contents: a chondrite class, the disc temperature it stands for
    (new_rule) and its bulk water, carbon and nitrogen mass fractions (reference)."""
    name: str
    t_k: float
    h2o: float
    c: float
    n: float


#: the anchors (contents: reference, see PROVENANCE['meteorite_classes']; temperatures: new_rule)
METEORITE_CLASSES = (
    MeteoriteClass("EC", 400.0, 0.001, 0.0040, 0.00040),
    MeteoriteClass("OC", 350.0, 0.005, 0.0012, 0.00003),
    MeteoriteClass("CM", 160.0, 0.100, 0.0220, 0.00080),
    MeteoriteClass("CI", 120.0, 0.170, 0.0350, 0.00300),
)
#: sensitivity: the same contents anchored at the black-body temperatures where the classes formed in the solar
#: system (PROVENANCE['meteorite_classes_sites'])
METEORITE_CLASSES_SITES = (
    MeteoriteClass("EC", BLACK_BODY, 0.001, 0.0040, 0.00040),
    MeteoriteClass("OC", 190.0, 0.005, 0.0012, 0.00003),
    MeteoriteClass("CM", 160.0, 0.100, 0.0220, 0.00080),
    MeteoriteClass("CI", 120.0, 0.170, 0.0350, 0.00300),
)

PROVENANCE = {
    "black_body_k": ("chain", f"world7's BLACK_BODY {BLACK_BODY} K (haishool/cosmos/world.py: T_sun sqrt(R_sun / 2 au)); "
                     "disc temperature T = BLACK_BODY L^(1/4) / sqrt(a / au), the spec's T(r)"),
    "meteorite_classes": ("reference+new_rule",
                          "bulk mass fractions (reference, rounded, water as all bulk H counted as H2O): "
                          "EC enstatite chondrites H2O about 0.1 %, C 0.40 %, N 0.04 % (C and N: EH/EL, Grady, Wright, "
                          "Carr & Pillinger 1986, GCA 50, 2799; water: Piani et al. 2020, Science 369, 1110, find "
                          "tenths of a per cent at most); OC ordinary chondrites H2O about 0.5 %, C 0.12 %, N 0.003 % "
                          "(Jarosewich 1990, Meteoritics 25, 323; Moore & Lewis 1967, JGR 72, 6289; Kung & Clayton 1978, "
                          "EPSL 38, 421: a few tens of ppm N); CM H2O about 10 %, C 2.2 %, N 0.08 % (Kerridge 1985, "
                          "GCA 49, 1707; Alexander et al. 2012, Science 337, 721); CI H2O about 17 %, C 3.5 %, N 0.3 % "
                          "(Lodders 2003, ApJ 591, 1220: C 3.48 %, N 2950 ppm, H about 2 %). new_rule: the disc "
                          "temperatures they stand for, from the spec's ranges: EC 400 K (EC above 400 K), OC 350 K "
                          "(middle of 300-400 K), CM 160 K (CM/CI below 160 K), CI 120 K (colder than CM: CI parent "
                          "bodies formed farther out, Kruijer et al. 2017, PNAS 114, 6712; it is the black body near "
                          "5 au). Habitable planets (215-286 K) lie between OC and CM, so EC and CI only shape the "
                          "curve outside them. A spec-level choice: these temperatures are not where the classes "
                          "formed (see meteorite_classes_sites, reported)"),
    "meteorite_classes_sites": ("reference+new_rule",
                                f"sensitivity (not the default): the same contents at the black body where each class "
                                f"formed: EC at Earth's {BLACK_BODY} K (Earth is built from EC-like material, Piani et al. "
                                "2020, Science 369, 1110; Dauphas 2017, Nature 541, 521), OC 190 K (S-type parent bodies "
                                "of the inner main belt near 2.1-2.2 au), CM 160 K and CI 120 K (as the default)"),
    "interpolation": ("new_rule", "log10 of each content linear in T between neighbouring anchors, constant beyond the "
                      "outermost ones (the spec: interpolated in log against T)"),
    "frost_t_k": ("chain", f"world7's frost line 2.7 au x sqrt(L) (haishool/cosmos/planets.py FROST_AU) at the disc "
                  f"temperature: BLACK_BODY / sqrt(2.7) = {FROST_T_K:.2f} K for every L"),
    "ice_share": ("chain", f"world7's solids beyond the frost line are rock x ICE_FACTOR {level3.ICE_FACTOR:g} "
                  f"(haishool/cosmos/planets.py): ice is {ICE_SHARE:.4f} of them; the rock keeps its anchored contents"),
    "degassed_share": ("new_rule+derived", "one constant: the share of the bulk water and carbon at the surface, "
                       "calibrated so that chain.earth_inputs gets Earth's surface water (ocean and vapour "
                       f"{C.EARTH_OCEAN_MASS_FRACTION:g} of its mass, Charette & Smith 2010)"),
    "nitrogen_degassed_share": ("new_rule+derived+reference",
                                "one more Earth calibration (owner's decision, 4 Oct 2026): the share of the bulk "
                                "nitrogen in the air as N2, calibrated so that the model Earth (chain.earth_inputs) "
                                "holds Earth's N2 partial pressure earth_n2_pa; solved jointly with p_cal as a fixed "
                                "point of the model Earth's build. 0.43 of the water's share: chondritic N/H2O "
                                "under the water's share gave the model Earth 2.3x Earth's N2 (2.05 bar of air). The "
                                "bulk Earth is poorer in N than chondrites relative to H and C (Marty 2012, EPSL "
                                "313-314, 56: the 'missing nitrogen'; citation quoted from memory). The same share "
                                "applies on every planet"),
    "earth_n2_pa": ("reference", f"{EARTH_N2_PA:.1f} Pa: Earth's dry-air N2 mole fraction 0.78084 (U.S. Standard "
                    "Atmosphere 1976, constants.EARTH_AIR_MOLE_FRACTION) x the standard sea-level pressure 101,325 "
                    "Pa; with O2 at x_O2 x 101,325 Pa (oxygen_pal 1) the model Earth's air is then about 1.03 bar "
                    "(vapour included) with x_O2 about 0.21. THE TARGET IS AN OWNER'S CHOICE STILL OPEN (review of "
                    "4 Oct 2026): the owner asked for 'about 78 kPa'; 79.1 kPa is the dry standard atmosphere at sea "
                    "level, 78 kPa about the moist sea-level N2 partial pressure (0.78084 x about 100 kPa), and "
                    "Earth's N2 inventory (about 3.88e18 kg, Trenberth & Smith 2005, J. Climate 18, 864; quoted from "
                    "memory: verify) would give 75.5 kPa on the model Earth (R 6,353.6 km, g 9.874 m/s^2); the "
                    "calibrated inventory is about 3.93e18 kg (+1.4 %). A degassed share is a mass ratio, so a "
                    "mass target is the more direct reading; the partial pressure is kept until the owner decides"),
    "fire_fit_pa": ("new_rule", "crafting.X_O2_MIN (0.15, Belcher et al. 2010) is a flammability limit measured at "
                    "about 1 bar and applied by the fires at any pressure; flame spread over thin fuels fails at "
                    "low pressure (limits rise as the pressure falls; quoted from memory: verify). Outside 0.5-2 "
                    "bar of total pressure a world's fire_possible is reported as None (uncertain) with "
                    "fire_beyond_fit True (summary_row, world3's fire report); the fires' own gate is unchanged"),
    "surface_pressure_hydrostatic": ("derived", "surface_pressure_pa is the sum of the partial pressures with the "
                                     "local surface vapour pressure RH x e_sat(T_s); the column's weight is the dry "
                                     "pressure plus the vapour column x g (surface_pressure_hydrostatic_pa), lower "
                                     "because water vapour is not well mixed (model Earth: 1,260 Pa of surface "
                                     "vapour against a 219 Pa column, 1.026 against 1.016 bar). Mole fractions "
                                     "with vapour (oxygen_mole_fraction) use the local surface sum"),
    "cloud_scale": ("chain+reference", "carbon and nitrogen x (X / rock)_cloud / (X / rock)_sun, rock = Fe + MgO + SiO2 "
                    "(version 2's interior rule) from the chain's cloud and from the solar cloud of chain.earth_inputs "
                    "(Asplund et al. 2009): the meteorite contents are those of solar-composition solids"),
    "lithophile_scale": ("chain+reference+new_rule",
                         "K, U and Th per kg of rock x (other / rock)_cloud / (other / rock)_sun: the chain cloud's "
                         "'other' metals (all beyond H, He, C, N, O, Ne, Mg, Si and Fe: Na, Al, S, K, Ca, U, Th ...) per "
                         "unit rock against the solar cloud's (new_rule: K, U and Th track the other metals). The "
                         "mantle is rock, so the heat and the argon per kg of it follow this ratio, not Z / Z_sun "
                         "(version 2's rule, which also counts C, N, O and Ne)"),
    "k_bse": ("reference+new_rule", "K in the silicate: 240 ppm, the bulk silicate Earth (McDonough & Sun 1995, Chem. "
              "Geol. 120, 223), x lithophile_cloud_scale (new_rule, see lithophile_scale)"),
    "radiogenic_heat": ("reference+new_rule", "Earth's 0.047 W/m^2 (constants.EARTH_RADIOGENIC_FLUX) x "
                        "lithophile_cloud_scale (heat-producing elements per kg of rock) x the decay of K-40, U-238, "
                        "U-235 and Th-232 since formation relative to Earth (version 2's radiogenic_decay_factor) x "
                        "silicate mass per area relative to Earth (version 2's rule)"),
    "k40_ar_branch": ("reference", f"{K40_AR_BRANCH:.4f}: lambda_e / (lambda_e + lambda_beta) = 0.581 / 5.543 (Steiger & "
                      "Jaeger 1977, EPSL 36, 359); the decay rate is ln 2 / 1.248 Gyr (NUBASE2016, constants.HALF_LIFE_YR)"),
    "k40_atom_fraction": ("reference+new_rule", "K-40 / K = 1.17e-4 by atoms today on Earth (IUPAC); new_rule: every "
                          "planet starts with the solar system's initial ratio (decayed back over Earth's age)"),
    "argon_degassed": ("reference", "about half of Earth's radiogenic Ar-40 is in the atmosphere (Allegre, Hofmann & "
                       "O'Nions 1996, GRL 23, 3555); the same share on every planet"),
    "relative_humidity": ("new_rule", "0.77 near-surface (version 2's, Dai 2006, J. Climate 19, 3589: 0.75-0.8)"),
    "whak": ("reference", "Walker, Hays & Kasting 1981 (JGR 86, 9776): continental weathering ~ (p_CO2 / p_E)^0.3 "
             "exp((T - T_E) / 13.7 K), normalised on the model's own Earth at world7's 287.6 K and 278 ppm (28.17 Pa)"),
    "weathering": ("reference+new_rule",
                   "default 'land_seafloor': W / W_E = (1 - s) (land / land_E) WHAK(p, T) + s (q / q_E) (p / p_E)^b_sf "
                   "exp(-E_sf / R (1 / T - 1 / T_E)). Continental weathering scales with the land area (Abbot, Cowan & "
                   "Ciesla 2012, ApJ 756, 178; version 2's climate weathers per m^2 of land), seafloor weathering with "
                   "the crust production rate (~ heat flow q, the same tectonic rule as the outgassing). 'global': the "
                   "published WHAK law on the whole planet, land or not (it hands a continental thermostat to a planet "
                   "without continents; reported in every spec as t_surface_global_weathering_k, "
                   "no_land_weathering_assumed when it drives a landless spec). 'land': no seafloor term (a bound: a "
                   "planet without land keeps all its degassed carbon in the air)"),
    "seafloor_share": ("reference+new_rule", "0.15 of the model Earth's silicate weathering is on the seafloor: basalt "
                       "carbonation of the upper oceanic crust about 2e12 mol C/yr (Alt & Teagle 1999, GCA 63, 1527: "
                       "1.5-2.4e12) against continental silicate weathering 11.7e12 mol CO2/yr (Gaillardet, Dupre, "
                       "Louvat & Allegre 1999, Chem. Geol. 159, 3); 2 / (2 + 11.7) = 0.15"),
    "seafloor_beta": ("reference", "0.25: seafloor weathering ~ (p_CO2 / p_E)^0.25, the middle of the 0-0.5 range of "
                      "Krissansen-Totton & Catling 2017 (Nat. Commun. 8, 15423); weaker than the continental 0.3 "
                      "only slightly, the temperature law is what differs"),
    "seafloor_e_j_mol": ("reference+new_rule", "41 kJ/mol: the activation energy of seafloor basalt weathering (Brady & "
                         "Gislason 1997, GCA 61, 965), an e-folding of R T^2 / E = 16.8 K at 287.6 K against the "
                         "continental 13.7 K; new_rule: applied to the surface temperature (the pore water follows the "
                         "deep ocean, which follows the surface where it does not freeze; Krissansen-Totton, Arney & "
                         "Catling 2018, PNAS 115, 4105)"),
    "outgassing": ("new_rule", "outgassing x radiogenic heat flux / the model Earth's (version 2's rule: tectonic "
                   "recycling scales with heat flow; the heat flux per kg of rock follows lithophile_cloud_scale) x "
                   "degassed carbon per m^2 / the model Earth's (the reservoir volcanoes return)"),
    "weathering_t": ("derived+new_rule", "default: weathering at the planet's own T_s (the fixed point of the "
                     "carbon-silicate loop, found by bisection; T_s falls as the weathering temperature rises, so it is "
                     "unique); 'chain': at world7's t_eq + greenhouse (the spec's wording); both are reported"),
    "carbon_cap": ("derived", "the air holds at most all the degassed carbon as CO2 (co2_carbon_limited); the rest of "
                   "the degassed carbon is carbonate rock (ocean dissolved carbon is counted with it)"),
    "greenhouse_split": ("reference+new_rule", "Earth's greenhouse: water vapour 50 %, CO2 20 %, clouds 25 % and other "
                         "gases 5 % (Schmidt, Ruedy, Miller & Lacis 2010, JGR 115, D20106); new_rule: these shares of "
                         "Earth's grey tau (world7's 254.6 + 33 K at albedo 0.30); vapour and CO2 are computed forward, "
                         "clouds held at Earth's (cloud_share), other gases scaled (other_gases)"),
    "co2_exponent": ("reference+derived", "tau_c ~ u_c^n_c with n_c such that doubling Earth's CO2 lowers the grey OLR "
                     "by 5.35 ln 2 = 3.71 W/m^2 at Earth's T_s (Myhre et al. 1998, GRL 25, 2715); finite as p -> 0 "
                     "(removing Earth's CO2 gives about -18 W/m^2, the g(C) fit of Myhre et al. -22)"),
    "h2o_exponent": ("reference+derived", "tau_w ~ u_w^n_w with n_w such that the vapour term adds IPCC AR6's water "
                     "vapour + lapse rate feedback 1.30 W m^-2 K^-1 per 3.22 of Planck (WG1 ch. 7, table 7.10) to "
                     "the grey model's own Planck response at Earth, as version 2's climate does"),
    "absorber_column": ("reference+new_rule", "absorption depends on the pressure-weighted absorber column u (P / P_cal)^b "
                        "(u_c = n_CO2 M_CO2; u_w = the vapour column of version 2's rule; P the dry surface pressure, the "
                        "foreign broadener): see broadening_exponent. N2-N2 and CO2-CO2 collision-induced absorption "
                        "(~ P^2) is not modelled; pressure_beyond_fit flags where it would matter"),
    "p_cal": ("derived", "the broadening reference P_cal: the model Earth's dry surface pressure (chain.earth_inputs, "
              "found as a fixed point of its build together with the nitrogen degassed share: 1.013 bar; 2.04 bar "
              "under nitrogen_calibration 'water'), so that the model Earth keeps world7's 287.6 K"),
    "co2_fit_max_pa": ("reference", "the CO2 forcing fits are made near present concentrations; beyond 1e4 Pa "
                       "(Byrne & Goldblatt 2014, GRL 41, 152) the result is flagged co2_beyond_fit"),
    "t_max_k": ("new_rule", "numerics: the energy-balance root is searched up to 450 K; none below it = runaway"),
    "t_grid_k": ("new_rule", "numerics: 0.25 K scan for the lowest root, then float64 bisection"),
    "everest_m": ("reference", "peak relief: Everest 8,848.86 m (2020 survey) at Earth's g; strength-limited "
                  "h ~ sigma_rock / (rho g), so h = 8,848.86 m x g_E / g"),
    "basin_depth": ("derived+reference+new_rule",
                    "isostasy: the continental platform rides at sea level (constant freeboard, new_rule) and the "
                    "water-loaded basins have depth D(g) = D_E (s_r g_E / g + 1 - s_r). D_E is calibrated so that "
                    f"Earth's ocean volume covers {EARTH_OCEAN_FRACTION} of Earth (Charette & Smith 2010; reference mean "
                    f"depth {EARTH_MEAN_OCEAN_DEPTH_M:g} m, the model ocean is 1000 kg/m^3, not 1025). The ridge share "
                    "s_r (basin_depth_rule 'split', the default) scales as 1/g: the ridge stands below the platform by an "
                    "Airy balance of crust whose thickness is fixed in pressure (oceanic crust from mantle melting at a "
                    "fixed pressure, continental crust strength-limited); the thermal subsidence of the plates "
                    "(2 rho_m alpha dT sqrt(kappa t / pi) / (rho_m - rho_w), Parsons & Sclater 1977) does not depend on "
                    "g. 'fixed' (all g-independent) and 'inverse_g' (all ~ 1/g) are the bounds; their ocean fractions "
                    "are reported in every spec (781: 0.62 fixed, 0.48 inverse_g; super-Earths flood more with 1/g)"),
    "t_earth_k": ("chain", f"world7's Earth {T_EARTH_K} K (planets.T_EQ_EARTH + GREENHOUSE_EARTH)"),
    "g_earth_surface": ("reference+derived", "GM_E / R_E^2 = 9.820 m/s^2 (IAU 2015 GM, IUGG mean radius): the g of the "
                        "Everest calibration"),
    "earth_mean_ocean_depth_m": ("reference", "3,682 m (Charette & Smith 2010, Oceanography 23(2), 112): the check of "
                                 "the derived basin depth"),
    "t_freeze_k": ("reference", "273.15 K, the melting point of ice at 1 atm (frozen_mean)"),
    "frozen": ("new_rule", "frozen_mean = T_s < 273.15 K. The formation T_s keeps the Bond albedo 0.30 (no ice-albedo "
               "feedback), Magnus vapour over liquid water and active weathering below freezing (no frozen-surface "
               "shutdown), so for these planets the T_s and the CO2 steady state are not self-consistent: the v3 "
               "climate decides snowball outcomes, and the formation T_s is no target for them"),
    "terrain": ("derived+reference+new_rule",
                "make_terrain3: version 2's random height field (great-circle faults and plane waves, globe.make_terrain's "
                "generator and RNG draws) ranks the cells; the lowest ocean_fraction of the area is ocean. Ocean "
                "depth below the platform D_max (1 - (1 - w)^k) (w: area rank from the coast, 0-1; k = "
                "ocean_hypsometry_k), scaled so the area-mean depth is basin_depth_m (or the whole layer when the "
                "basins overflow). Land height above sea level a truncated exponential of the land's area rank, mean "
                "mean_land_height_m, top peak_height_m. The sea level is then solved for the spec's ocean volume "
                "(version 2's globe.sea_level, exact shells), so the ocean volume and area both match the spec"),
    # one entry per Formation3Rules field
    "classes": ("reference+new_rule", "the meteorite anchors (see meteorite_classes)"),
    "lapse_rate_k_m": ("reference+new_rule", "6.5 K/km (US Std. Atm. 1976) x g / g_E for the vapour scale height "
                       "(version 2's rule)"),
    "whak_beta": ("reference", "0.3: weathering ~ (p_CO2 / p_E)^0.3 (Walker, Hays & Kasting 1981, JGR 86, 9776)"),
    "whak_t_e_k": ("reference", "13.7 K e-folding of weathering with T (Walker, Hays & Kasting 1981; version 2's "
                   "climate weathering_k)"),
    "water_share": ("reference+new_rule", "0.50 of Earth's grey tau is water vapour (see greenhouse_split)"),
    "co2_share": ("reference+new_rule", "0.20 of Earth's grey tau is CO2 (see greenhouse_split)"),
    "cloud_share": ("reference+new_rule", "0.25 of Earth's grey tau is clouds (Schmidt et al. 2010), held at Earth's on "
                    "every planet: a stated physics placeholder (formation has no cloud model); each spec reports its "
                    "warming (t_surface_no_clouds_k)"),
    "other_gases": ("reference+new_rule", "the remaining 0.05 of Earth's grey tau (O3, CH4, N2O and aerosols, Schmidt et "
                    "al. 2010) is made by a biosphere (O3 from its O2). Default 'oxygen_pal': x the chain's oxygen_pal "
                    "(new_rule: the bodies era's O2 as the measure of its gas output); 'earth': held at Earth's (the "
                    "earlier rule); 'none': dropped. Each spec reports its warming (t_surface_no_other_gases_k)"),
    "co2_forcing_w_m2": ("reference", "5.35 W/m^2 per e-fold of CO2 (Myhre et al. 1998, GRL 25, 2715)"),
    "wv_lr_feedback": ("reference", "water vapour + lapse rate feedback 1.30 W m^-2 K^-1 (IPCC AR6 WG1 ch. 7, table 7.10)"),
    "planck_feedback": ("reference", "Planck feedback 3.22 W m^-2 K^-1 (IPCC AR6 WG1 ch. 7, table 7.10)"),
    "broadening_exponent": ("reference+new_rule", "b = 0.5. The Lorentz half-width grows in proportion to the pressure "
                            "(Goody & Yung 1989, Atmospheric Radiation, ch. 3): absorption depends on the pressure-weighted "
                            "path u P in the strong-line limit (b = 1, Pierrehumbert 2010, Principles of Planetary Climate, "
                            "section 4.4) and not on P in the weak-line limit (b = 0). new_rule: the grey power laws (n_c, "
                            "n_w, fitted at fixed P) take u (P / P_cal)^b with b = 0.5, inside the usual 0.5-1 range, because "
                            "it reproduces the warming of doubled N2 at fixed CO2 (vapour feedback included): about 5 K in "
                            "this grey model against about 4.4 K in Goldblatt et al. 2009 (Nature Geosci. 2, 891), where "
                            "b = 1 gives 13 K. b = 0 removes broadening; each spec reports its warming "
                            "(t_surface_no_broadening_k)"),
    "pressure_fit_factor": ("new_rule", "pressure_beyond_fit when the dry pressure exceeds 2 x P_cal (2.03 bar; 4.1 bar "
                            "under nitrogen_calibration 'water'): there the missing collision-induced absorption "
                            "(~ P^2) would add warming the model lacks"),
    "nitrogen_calibration": ("new_rule+reference",
                             "'earth_n2' (default, owner's decision 4 Oct 2026): nitrogen has its own degassed share, "
                             "calibrated on Earth's N2 (see nitrogen_degassed_share); 'water': the earlier single "
                             "share of water, carbon and nitrogen (the model Earth then has 2.05 bar of air), kept "
                             "as a reported reading"),
    "basin_depth_rule": ("derived+reference+new_rule", "'split' (default), 'fixed' or 'inverse_g' (see basin_depth)"),
    "ridge_depth_m": ("reference", "2,600 m: depth of the mid-ocean ridges below sea level (Stein & Stein 1992, Nature "
                      "359, 123, GDH1); s_r = 2,600 / 3,682 = 0.706 of Earth's mean ocean depth is the ridge's"),
    "mean_land_height_m": ("reference+new_rule", "about 840 m: Earth's mean land elevation above sea level (Earth's "
                           "hypsometric curve, e.g. Eakins & Sharman 2012, NOAA NGDC, ETOPO1), x g_E / g (new_rule: "
                           "continental freeboard and mountains are strength- and crust-limited, ~ 1/g like the peaks)"),
    "ocean_hypsometry_k": ("new_rule", "k = 4: the ocean floor's shape D_max (1 - (1 - w)^k), mean D: about 3/4 of the "
                           "floor deeper than 0.64 D_max (Earth: about three quarters of its ocean floor lies 3-6 km "
                           "deep, ETOPO1); a stated shape, the area and volume are what the spec fixes"),
    "base": ("new_rule", "version 2's FormationRules (planet.formation, its own provenance): star radius, PAR band, "
             "Murnaghan K', lock radius and the rotation draws"),
}


@dataclass(frozen=True)
class Formation3Rules:
    """The modelling choices of :func:`build_planet3` (sources in ``PROVENANCE``)."""
    classes: tuple = METEORITE_CLASSES
    relative_humidity: float = 0.77
    lapse_rate_k_m: float = C.EARTH_LAPSE_RATE
    whak_beta: float = 0.3
    whak_t_e_k: float = 13.7
    weathering_t: str = "self_consistent"  # or "chain" (the spec's literal reading); both are reported
    weathering: str = "land_seafloor"      # or "global" (published WHAK on any planet) or "land" (no seafloor)
    seafloor_share: float = 0.15
    seafloor_beta: float = 0.25
    seafloor_e_j_mol: float = 41.0e3
    k_bse: float = 240e-6
    argon_degassed: float = 0.5
    water_share: float = 0.50
    co2_share: float = 0.20
    cloud_share: float = 0.25
    other_gases: str = "oxygen_pal"        # or "earth" or "none"
    co2_forcing_w_m2: float = 5.35       # Myhre et al. 1998, per e-fold
    wv_lr_feedback: float = 1.30         # IPCC AR6
    planck_feedback: float = 3.22        # IPCC AR6
    broadening_exponent: float = 0.5
    pressure_fit_factor: float = 2.0
    co2_fit_max_pa: float = 1.0e4
    t_max_k: float = 450.0
    t_grid_k: float = 0.25
    everest_m: float = EVEREST_M
    basin_depth_rule: str = "split"        # or "fixed" or "inverse_g"
    ridge_depth_m: float = 2600.0
    mean_land_height_m: float = 840.0
    ocean_hypsometry_k: float = 4.0
    nitrogen_calibration: str = "earth_n2"   # or "water" (the earlier single share of water, C and N)
    base: F.FormationRules = F.FormationRules()   # version 2's star, orbit, rotation and interior choices

    def validate(self) -> None:
        for name, value, allowed in (("weathering_t", self.weathering_t, ("self_consistent", "chain")),
                                     ("weathering", self.weathering, _WEATHERING),
                                     ("other_gases", self.other_gases, _OTHER_GASES),
                                     ("basin_depth_rule", self.basin_depth_rule, _BASIN_RULES),
                                     ("nitrogen_calibration", self.nitrogen_calibration, _NITROGEN_CALIBRATIONS)):
            if value not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, not {value!r}")
        if not 0.0 <= self.seafloor_share < 1.0:
            raise ValueError("seafloor_share must lie in [0, 1)")
        if self.water_share + self.co2_share + self.cloud_share > 1.0:
            raise ValueError("water, CO2 and cloud shares exceed Earth's tau")


# ---------------------------------------------------------------------------------------------
# the solids


def disc_temperature_k(luminosity_lsun, orbit_au):
    """Disc (black-body) temperature at the orbit: world7's BLACK_BODY L^(1/4) / sqrt(a). Numbers or arrays."""
    return BLACK_BODY * np.power(luminosity_lsun, 0.25) / np.sqrt(orbit_au)


def anchored_fraction(t_k, key: str, classes=METEORITE_CLASSES):
    """The meteorite-anchored mass fraction ``key`` ('h2o', 'c' or 'n') of rock formed at disc temperature t_k:
    log10 interpolated against T between the anchors, constant beyond them. Numbers or arrays."""
    pts = sorted(classes, key=lambda m: m.t_k)
    ts = np.array([m.t_k for m in pts], dtype=np.float64)
    logs = np.log10(np.array([getattr(m, key) for m in pts], dtype=np.float64))
    out = np.power(10.0, np.interp(np.asarray(t_k, dtype=np.float64), ts, logs))
    return float(out) if np.ndim(out) == 0 else out


def rock_mass_fraction(cloud: dict) -> float:
    """Fe + MgO + SiO2 per unit cloud mass (version 2's interior rule: all Fe metal, Mg and Si as oxides)."""
    mgo = cloud["magnesium"] * C.molar_mass("MgO") / C.element_mass("Mg")
    sio2 = cloud["silicon"] * C.molar_mass("SiO2") / C.element_mass("Si")
    return cloud["iron"] + mgo + sio2


@lru_cache(maxsize=1)
def _solar_cloud() -> dict:
    return dict(chain9.earth_inputs()["cloud"])


def cloud_scale(cloud: dict, element: str) -> float:
    """(element / rock)_cloud over the sun's: the chain cloud's carbon, nitrogen or 'other' metals (the
    lithophile scale of K, U and Th) relative to solar solids."""
    sun = _solar_cloud()
    return (cloud[element] / rock_mass_fraction(cloud)) / (sun[element] / rock_mass_fraction(sun))


def solid_volatiles(t_k: float, cloud: dict, classes=METEORITE_CLASSES) -> dict:
    """Bulk mass fractions of water, carbon and nitrogen in solids accreted at disc temperature t_k from
    ``cloud``: the anchored rock contents (C and N x cloud_scale), plus world7's ice beyond the frost line."""
    ice = ICE_SHARE if t_k < FROST_T_K else 0.0
    rock = 1.0 - ice
    return {"h2o": ice + rock * anchored_fraction(t_k, "h2o", classes),
            "c": rock * anchored_fraction(t_k, "c", classes) * cloud_scale(cloud, "carbon"),
            "n": rock * anchored_fraction(t_k, "n", classes) * cloud_scale(cloud, "nitrogen"),
            "ice": ice}


@lru_cache(maxsize=None)
def degassed_share(classes=METEORITE_CLASSES) -> float:
    """The water (and carbon) degassed share: Earth's surface water over the water of solids at Earth's disc
    temperature."""
    e = chain9.earth_inputs()
    t = float(disc_temperature_k(e["star"]["luminosity_lsun"], e["planet"]["orbit_au"]))
    return C.EARTH_OCEAN_MASS_FRACTION / solid_volatiles(t, e["cloud"], classes)["h2o"]


def nitrogen_degassed_share(rules: Formation3Rules | None = None) -> float:
    """The nitrogen degassed share: the share of the bulk N that puts Earth's N2 partial pressure (EARTH_N2_PA) in
    the model Earth's air (a fixed point of its build, found with :func:`greenhouse`'s P_cal); the water's share
    under ``nitrogen_calibration="water"``."""
    return _earth_calibration(rules or Formation3Rules())[1]


def radiogenic_heat_w_m2(lithophile_scale: float, age_yr: float, mantle_kg: float, radius_m: float) -> float:
    """Radiogenic heat flux: Earth's 0.047 W/m^2 x heat-producing elements per kg of rock (lithophile scale)
    x their decay relative to Earth x silicate mass per area relative to Earth."""
    mantle_earth = (1.0 - C.EARTH_CORE_MASS_FRACTION) * (1.0 - C.EARTH_OCEAN_MASS_FRACTION) * C.M_EARTH
    size = (mantle_kg / mantle_earth) / (radius_m / C.R_EARTH) ** 2
    return C.EARTH_RADIOGENIC_FLUX * lithophile_scale * F.radiogenic_decay_factor(age_yr) * size


# ---------------------------------------------------------------------------------------------
# relief


def basin_depth_m() -> float:
    """Earth's calibrated water-loaded ocean basin depth below the continental platform, D_E: Earth's ocean
    volume (model density 1000 kg/m^3) over the area it covers, 0.709 of 4 pi R_E^2."""
    volume = C.EARTH_OCEAN_MASS_FRACTION * C.M_EARTH / C.RHO_WATER
    return volume / (4.0 * math.pi * C.R_EARTH ** 2 * EARTH_OCEAN_FRACTION)


def ridge_share(rules: Formation3Rules | None = None, rule: str | None = None) -> float:
    """The share of the basin depth that scales as 1/g: ridge depth / Earth's mean ocean depth ('split'), 0
    ('fixed') or 1 ('inverse_g')."""
    r = rules or Formation3Rules()
    rule = rule or r.basin_depth_rule
    return {"split": r.ridge_depth_m / EARTH_MEAN_OCEAN_DEPTH_M, "fixed": 0.0, "inverse_g": 1.0}[rule]


def basin_depth_at(g: float, share: float) -> float:
    """D(g) = D_E (share g_E / g + 1 - share)."""
    return basin_depth_m() * (share * G_EARTH_SURFACE / g + 1.0 - share)


def ocean_fraction(ocean_volume_m3: float, radius_m: float, depth_m: float | None = None) -> float:
    """Area share of the ocean: the basins fill to the platform, min(1, V / (4 pi R^2 D))."""
    d = basin_depth_m() if depth_m is None else depth_m
    return min(1.0, max(0.0, ocean_volume_m3) / (4.0 * math.pi * radius_m ** 2 * d))


def argon_produced_kg(potassium_kg: float, age_yr: float) -> float:
    """Radiogenic Ar-40 made over ``age_yr`` by K whose K-40 started at the solar system's initial ratio:
    branch x initial K-40 x (1 - exp(-lambda t)), lambda = ln 2 / 1.248 Gyr."""
    lam = math.log(2.0) / C.HALF_LIFE_YR["K40"]
    k40_initial_mol = potassium_kg / _M_K * K40_ATOM_FRACTION * math.exp(lam * C.EARTH_AGE_YR)
    return K40_AR_BRANCH * k40_initial_mol * (-math.expm1(-lam * age_yr)) * _M_AR


# ---------------------------------------------------------------------------------------------
# the greenhouse


def e_sat_pa(t_k):
    """Magnus saturation vapour pressure over water (version 2's, Alduchov & Eskridge 1996); arrays too."""
    t = np.asarray(t_k, dtype=np.float64) - 273.15
    out = 610.94 * np.exp(17.625 * t / (t + 243.04))
    return float(out) if np.ndim(out) == 0 else out


def vapour_column(t_k, g: float, rules: Formation3Rules):
    """Column water vapour (kg/m^2) at relative humidity r: version 2's p M_w / (R T) x R_v T^2 / (L_v Gamma),
    Gamma = lapse x g / g_E. Arrays too."""
    t = np.asarray(t_k, dtype=np.float64)
    gamma = rules.lapse_rate_k_m * g / C.G_STANDARD
    h = (C.R_GAS / _M["H2O"]) * t ** 2 / (C.L_VAPORIZATION * gamma)
    out = rules.relative_humidity * e_sat_pa(t) * _M["H2O"] / (C.R_GAS * t) * h
    return float(out) if np.ndim(out) == 0 else out


@dataclass(frozen=True)
class Greenhouse:
    """Earth's calibration of the forward grey tau (see the module docstring). ``p_rel`` is the dry pressure
    over ``p_cal_pa`` (the model Earth's)."""
    tau_earth: float
    tau_cloud: float
    tau_other_earth: float
    tau_c_earth: float
    tau_w_earth: float
    n_c: float
    n_w: float
    u_c_earth: float
    u_w_earth: float
    p_cal_pa: float
    broadening: float

    @property
    def tau_rest(self) -> float:
        """Earth's clouds and other gases."""
        return self.tau_cloud + self.tau_other_earth

    def tau_co2(self, u_c, p_rel=1.0):
        path = np.maximum(u_c, 0.0) * np.power(p_rel, self.broadening) / self.u_c_earth
        return self.tau_c_earth * np.power(path, self.n_c)

    def tau_h2o(self, u_w, p_rel=1.0):
        path = np.maximum(u_w, 0.0) * np.power(p_rel, self.broadening) / self.u_w_earth
        return self.tau_w_earth * np.power(path, self.n_w)

    def tau_other(self, oxygen_pal: float, mode: str) -> float:
        """The other gases (O3, CH4, N2O): Earth's x oxygen_pal, Earth's, or none (``Formation3Rules.other_gases``)."""
        return {"oxygen_pal": self.tau_other_earth * max(0.0, oxygen_pal), "earth": self.tau_other_earth,
                "none": 0.0}[mode]


def _greenhouse_at(rules: Formation3Rules, p_cal_pa: float) -> Greenhouse:
    t = T_EARTH_K
    tau = F.earth_tau()
    olr = C.SIGMA * t ** 4
    dolr_dtau = 0.75 * olr / (1.0 + 0.75 * tau) ** 2
    planck = 4.0 * olr / t / (1.0 + 0.75 * tau)
    tau_c, tau_w = rules.co2_share * tau, rules.water_share * tau
    tau_cloud = rules.cloud_share * tau
    tau_other = tau - tau_c - tau_w - tau_cloud
    # CO2: doubling at Earth's T_s cuts the grey OLR by Myhre's 5.35 ln 2 = 3.71 W/m^2
    chi2 = 1.0 / (1.0 + 0.75 * tau) - rules.co2_forcing_w_m2 * math.log(2.0) / olr
    n_c = math.log2(1.0 + ((1.0 / chi2 - 1.0) / 0.75 - tau) / tau_c)
    tc = t - 273.15
    dlnu_dt = 17.625 * 243.04 / (tc + 243.04) ** 2 + 1.0 / t   # Magnus + T^2 scale height - 1/T density
    n_w = rules.wv_lr_feedback / rules.planck_feedback * planck / dlnu_dt / (dolr_dtau * tau_w)
    mole = dict(C.EARTH_AIR_MOLE_FRACTION)
    mu_dry = sum(x * _M[g] for g, x in mole.items()) / sum(mole.values())
    u_c = _P_CO2_EARTH * _M["CO2"] / (mu_dry * C.G_STANDARD)
    u_w = vapour_column(t, C.G_STANDARD, rules)
    return Greenhouse(tau, tau_cloud, tau_other, tau_c, tau_w, n_c, n_w, u_c, u_w, float(p_cal_pa),
                      float(rules.broadening_exponent))


def greenhouse(rules: Formation3Rules = Formation3Rules()) -> Greenhouse:
    """The forward greenhouse calibrated on world7's Earth (287.6 K, tau_E of albedo 0.30 at S0), with its
    broadening reference at the model Earth's dry pressure (a fixed point of the model Earth's build)."""
    return _earth_calibration(rules)[0]


@lru_cache(maxsize=None)
def _earth_calibration(rules: Formation3Rules) -> tuple[Greenhouse, float]:
    """(greenhouse, nitrogen degassed share): the joint fixed point of the model Earth's build, its dry pressure
    (the broadening reference P_cal) and, under ``nitrogen_calibration="earth_n2"``, the nitrogen share that gives
    it Earth's N2 partial pressure. p_N2 = n_N2 g mu_dry is near proportional to the share (mu_dry and g move
    little), so the share is rescaled by EARTH_N2_PA / p_N2 each pass."""
    rules.validate()
    e = chain9.earth_inputs()
    base = F.build_planet(e, 0, rules.base)
    gh = _greenhouse_at(rules, C.P_STANDARD)
    n_share = degassed_share(rules.classes)
    calibrate = rules.nitrogen_calibration == "earth_n2"
    for _ in range(100):
        s = _solve(e, base, rules, gh, None, n_share)
        p = s["p_dry"]
        n_next = n_share
        if calibrate:
            p_n2 = s["st"]["n"]["N2"] * s["col"].g * s["st"]["mu_dry"]
            n_next = n_share * EARTH_N2_PA / p_n2
        done = abs(p - gh.p_cal_pa) <= 1e-13 * p and abs(n_next - n_share) <= 1e-13 * n_share
        gh = replace(gh, p_cal_pa=p)
        n_share = n_next
        if done:
            return gh, n_share
    raise RuntimeError("the model Earth's dry pressure and nitrogen share did not converge")


def surface_temperature(t_eq: float, tau_of_t, rules: Formation3Rules) -> tuple[float, bool]:
    """Lowest T >= t_eq with T^4 = t_eq^4 (1 + 0.75 tau(T)): a 0.25 K scan, then float64 bisection.
    Returns (T, runaway); runaway when no root lies below ``t_max_k`` (then T = t_max_k)."""
    grid = np.arange(t_eq, rules.t_max_k + rules.t_grid_k, rules.t_grid_k, dtype=np.float64)
    f = grid ** 4 - t_eq ** 4 * (1.0 + 0.75 * tau_of_t(grid))
    hit = np.flatnonzero(f >= 0.0)
    if hit.size == 0:
        return float(rules.t_max_k), True
    i = int(hit[0])
    if i == 0:
        return float(grid[0]), False
    lo, hi = float(grid[i - 1]), float(grid[i])
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            break
        if mid ** 4 - t_eq ** 4 * (1.0 + 0.75 * float(tau_of_t(mid))) >= 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi), False


# ---------------------------------------------------------------------------------------------
# the carbon-silicate steady state


def whak_co2_pa(t_k: float, outgassing_rel: float, land_rel: float, rules: Formation3Rules) -> float:
    """Continental Walker-Hays-Kasting steady state alone: p_E [V_rel / land_rel exp(-(T - T_E) / T_e)]^(1 / beta);
    inf without land, 0 without outgassing (pass land_rel = 1 for the published global law)."""
    if outgassing_rel <= 0.0:
        return 0.0
    if land_rel <= 0.0:
        return math.inf
    ln_p = math.log(_P_CO2_EARTH) + (math.log(outgassing_rel) - math.log(land_rel)
                                     - (t_k - T_EARTH_K) / rules.whak_t_e_k) / rules.whak_beta
    return math.exp(min(ln_p, 700.0))


def _weathering_logs(t_k: float, land_rel: float, heat_rel: float, rules: Formation3Rules, mode: str):
    """ln coefficients (a, b) of W / W_E = a (p / p_E)^beta + b (p / p_E)^beta_sf (-inf for an absent term)."""
    land_t = (t_k - T_EARTH_K) / rules.whak_t_e_k
    if mode == "global":
        return land_t, -math.inf
    if mode == "land":
        return (math.log(land_rel) + land_t if land_rel > 0.0 else -math.inf), -math.inf
    s = rules.seafloor_share
    ln_a = math.log(1.0 - s) + math.log(land_rel) + land_t if land_rel > 0.0 else -math.inf
    sea_t = -rules.seafloor_e_j_mol / C.R_GAS * (1.0 / t_k - 1.0 / T_EARTH_K)
    ln_b = math.log(s) + math.log(heat_rel) + sea_t if s > 0.0 and heat_rel > 0.0 else -math.inf
    return ln_a, ln_b


def weathering_terms(p_pa: float, t_k: float, land_rel: float, heat_rel: float, rules: Formation3Rules,
                     mode: str | None = None) -> tuple[float, float]:
    """(continental, seafloor) silicate weathering relative to the model Earth's total at CO2 ``p_pa`` and
    temperature ``t_k`` under ``mode`` (default ``rules.weathering``)."""
    ln_a, ln_b = _weathering_logs(t_k, land_rel, heat_rel, rules, mode or rules.weathering)
    if p_pa <= 0.0:
        return 0.0, 0.0
    x = math.log(p_pa / _P_CO2_EARTH)
    land = math.exp(ln_a + rules.whak_beta * x) if ln_a > -math.inf else 0.0
    sea = math.exp(ln_b + rules.seafloor_beta * x) if ln_b > -math.inf else 0.0
    return land, sea


def steady_co2_pa(t_k: float, outgassing_rel: float, land_rel: float, heat_rel: float, rules: Formation3Rules,
                  mode: str | None = None) -> float:
    """The CO2 at which weathering (``weathering_terms``) equals outgassing at temperature t_k: closed form for one
    term, else bisection on ln p (the sum of two exponentials in ln p is monotone). 0 without outgassing, inf
    without any weathering surface."""
    if outgassing_rel <= 0.0:
        return 0.0
    ln_a, ln_b = _weathering_logs(t_k, land_rel, heat_rel, rules, mode or rules.weathering)
    ln_v = math.log(outgassing_rel)
    beta, beta_sf = rules.whak_beta, rules.seafloor_beta
    if ln_a == -math.inf and ln_b == -math.inf:
        return math.inf
    if ln_b == -math.inf:
        x = (ln_v - ln_a) / beta
    elif ln_a == -math.inf:
        x = (ln_v - ln_b) / beta_sf
    else:
        # at hi one term alone equals V (sum in [V, 2V]); at lo both terms are at most V / 2
        hi = min((ln_v - ln_a) / beta, (ln_v - ln_b) / beta_sf)
        lo = hi - math.log(2.0) / min(beta, beta_sf)
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if mid <= lo or mid >= hi:
                break
            if np.logaddexp(ln_a + beta * mid, ln_b + beta_sf * mid) < ln_v:
                lo = mid
            else:
                hi = mid
        x = 0.5 * (lo + hi)
    return math.exp(min(math.log(_P_CO2_EARTH) + x, 700.0))


def dry_air(n_fixed: dict, p_target: dict, n_cap: dict, g: float) -> tuple[dict, float]:
    """A well-mixed dry column: gases given in mol/m^2 (``n_fixed``) and gases given as partial pressures
    (``p_target``, at most ``n_cap`` mol/m^2) share mu_dry, p_i = n_i g mu_dry. Returns ({gas: mol/m^2}, mu_dry).
    Raises RuntimeError when the molar mass does not converge."""
    mu = _M["N2"]
    for _ in range(1000):
        n = dict(n_fixed)
        for gas, p in p_target.items():
            n[gas] = min(p / (g * mu), n_cap.get(gas, math.inf))
        total = sum(n.values())
        if total <= 0.0:
            return {gas: 0.0 for gas in _DRY}, _M["N2"]
        mu_next = sum(n[gas] * _M[gas] for gas in n) / total
        if abs(mu_next - mu) <= 1e-15 * mu:
            return {gas: n.get(gas, 0.0) for gas in _DRY}, mu_next
        mu = mu_next
    raise RuntimeError("dry_air: the molar mass did not converge in 1000 iterations")


@dataclass(frozen=True)
class _Column:
    """One planet's air column at fixed geometry: CO2 -> dry air -> greenhouse -> T_s."""
    gh: Greenhouse
    r: Formation3Rules
    t_eq: float
    p_o2: float
    tau_cloud: float
    tau_other: float
    g: float
    n_n2: float
    n_ar: float
    n_c_max: float

    def climate(self, p_co2: float, *, clouds: bool = True, other: bool = True, broadening: bool = True) -> dict:
        n, mu_dry = dry_air({"N2": self.n_n2, "Ar": self.n_ar}, {"O2": self.p_o2, "CO2": p_co2},
                            {"CO2": self.n_c_max}, self.g)
        p_dry = sum(n.values()) * self.g * mu_dry
        p_rel = p_dry / self.gh.p_cal_pa if broadening else 1.0
        tau_c = float(self.gh.tau_co2(n["CO2"] * _M["CO2"], p_rel))
        fixed = (self.tau_cloud if clouds else 0.0) + (self.tau_other if other else 0.0)

        def tau_of_t(t):
            return fixed + tau_c + self.gh.tau_h2o(vapour_column(t, self.g, self.r), p_rel)

        t_s, runaway = surface_temperature(self.t_eq, tau_of_t, self.r)
        return {"n": n, "mu_dry": mu_dry, "p_dry": p_dry, "p_rel": p_rel, "tau_c": tau_c, "t_s": t_s,
                "runaway": runaway, "limited": n["CO2"] < p_co2 / (self.g * mu_dry) * (1.0 - 1e-12)}

    def fixed_point(self, steady, guess=None):
        """The weathering temperature T_w with T_s(p_CO2(T_w)) = T_w: T_s falls as T_w rises (warmer weathers
        faster, less CO2), so the root is unique; bisection on [T_eq, t_max] (or a 1 K bracket around ``guess``
        when it brackets)."""
        def h(t):
            return self.climate(steady(t))

        lo, hi = self.t_eq, self.r.t_max_k
        top = h(hi)
        if top["t_s"] >= hi:
            return hi, top
        if guess is not None and lo < guess - 0.5 and guess + 0.5 < hi \
                and h(guess - 0.5)["t_s"] > guess - 0.5 and h(guess + 0.5)["t_s"] <= guess + 0.5:
            lo, hi = guess - 0.5, guess + 0.5
        while hi - lo > 1e-10:
            mid = 0.5 * (lo + hi)
            if mid <= lo or mid >= hi:
                break
            if h(mid)["t_s"] > mid:
                lo = mid
            else:
                hi = mid
        t_w = 0.5 * (lo + hi)
        return t_w, h(t_w)


# ---------------------------------------------------------------------------------------------
# the spec


@dataclass(frozen=True)
class PlanetSpec3(F.PlanetSpec):
    """A version-2 :class:`PlanetSpec` (every field and meaning kept; ``t_surface_target_k`` is now the derived
    T_s, ``greenhouse_tau`` the forward tau, ``co2_ref_pa`` the steady-state CO2 and ``relief_m`` the real-scale
    peak + basin depth, for :func:`make_terrain3`) plus the v3 budgets."""
    # solids and the degassed share
    surface_pressure_hydrostatic_pa: float = 0.0
    disc_temperature_k: float = 0.0
    frost_temperature_k: float = 0.0
    ice_fraction: float = 0.0
    solid_water_fraction: float = 0.0
    solid_carbon_fraction: float = 0.0
    solid_nitrogen_fraction: float = 0.0
    carbon_cloud_scale: float = 0.0
    nitrogen_cloud_scale: float = 0.0
    lithophile_cloud_scale: float = 0.0
    degassed_share: float = 0.0
    nitrogen_degassed_share: float = 0.0
    # mass budgets (kg)
    bulk_water_kg: float = 0.0
    surface_water_kg: float = 0.0
    undegassed_water_kg: float = 0.0
    bulk_carbon_kg: float = 0.0
    surface_carbon_kg: float = 0.0
    air_carbon_kg: float = 0.0
    carbonate_carbon_kg: float = 0.0
    undegassed_carbon_kg: float = 0.0
    bulk_nitrogen_kg: float = 0.0
    air_nitrogen_kg: float = 0.0
    undegassed_nitrogen_kg: float = 0.0
    potassium_kg: float = 0.0
    argon_produced_kg: float = 0.0
    air_argon_kg: float = 0.0
    retained_argon_kg: float = 0.0
    dry_air_molar_mass_kg_mol: float = 0.0
    # carbon-silicate steady state
    heat_rel_earth: float = 0.0
    outgassing_rel_earth: float = 0.0
    land_rel_earth: float = 0.0
    weathering_land_scaled: bool = False
    seafloor_weathering_share: float = 0.0
    weathering_land_share: float = 0.0
    no_land_weathering_assumed: bool = False
    weathering_at_chain_t: bool = False
    weathering_t_k: float = 0.0
    co2_carbon_limited: bool = False
    co2_beyond_fit: bool = False
    # greenhouse and the chain check
    tau_h2o: float = 0.0
    tau_co2: float = 0.0
    tau_cloud: float = 0.0
    tau_other_gases: float = 0.0
    tau_rest: float = 0.0
    dry_pressure_rel_model_earth: float = 0.0
    pressure_beyond_fit: bool = False
    runaway: bool = False
    frozen_mean: bool = False
    chain_t_target_k: float = 0.0
    chain_t_s_mismatch_k: float = 0.0
    t_surface_selfconsistent_k: float = 0.0
    co2_selfconsistent_pa: float = 0.0
    t_surface_chain_weathering_k: float = 0.0
    co2_chain_weathering_pa: float = 0.0
    t_surface_global_weathering_k: float = 0.0
    co2_global_weathering_pa: float = 0.0
    t_surface_no_clouds_k: float = 0.0
    t_surface_no_other_gases_k: float = 0.0
    t_surface_no_broadening_k: float = 0.0
    fixed_point_converged: bool = False
    # relief and isostasy
    peak_height_m: float = 0.0
    mean_land_height_m: float = 0.0
    basin_depth_ridge_share: float = 0.0
    basin_depth_m: float = 0.0
    ocean_fraction: float = 0.0
    ocean_fraction_fixed_depth: float = 0.0
    ocean_fraction_inverse_g_depth: float = 0.0
    land_fraction: float = 0.0
    flood_depth_m: float = 0.0


#: version-2 fields taken unchanged from planet.formation.build_planet
_V2_KEPT = ("seed", "source", "star_mass_kg", "star_luminosity_w", "star_age_yr", "star_birth_yr", "star_lifetime_yr",
            "star_radius_m", "star_teff_k", "par_fraction", "orbit_m", "year_s", "flux_earth", "insolation_w_m2",
            "metallicity", "planet_mass_kg", "core_mass_fraction", "uncompressed_density_kg_m3", "compression",
            "oxygen_pal", "albedo", "chain_t_eq_k", "t_eq_k", "chain_t_surface_k", "lock_radius_m", "tidally_locked",
            "rotation_period_s", "obliquity_rad", "day_length_s", "crust")


@lru_cache(maxsize=None)
def _earth_reference(rules: Formation3Rules) -> tuple[float, float, float]:
    """(radiogenic heat flux, degassed carbon per m^2, land fraction) of the model's own Earth, which is in
    carbonate balance at 287.6 K and 278 ppm by definition."""
    spec = model_earth(rules)
    area = 4.0 * math.pi * spec.radius_m ** 2
    return spec.radiogenic_heat_w_m2, spec.surface_carbon_kg / area, spec.land_fraction


@lru_cache(maxsize=None)
def model_earth(rules: Formation3Rules = Formation3Rules()) -> PlanetSpec3:
    """The model's own Earth (chain.earth_inputs, the calibration planet, outgassing and land at their own
    reference): Earth's surface water and, with its own degassed share, Earth's N2 (79.1 kPa): about 1.03 bar of air
    with x_O2 about 0.21 (2.05 bar and 2.3x Earth's N2 under ``nitrogen_calibration="water"``)."""
    rules.validate()
    gh, n_share = _earth_calibration(rules)
    return _build(chain9.earth_inputs(), 0, rules, gh, n_share, reference=None)


def build_planet3(inputs: dict, seed: int, rules: Formation3Rules | None = None) -> PlanetSpec3:
    """The :class:`PlanetSpec3` of chain inputs (``chain.chain_inputs``, ``synthetic_inputs`` or ``earth_inputs``).
    ``seed`` seeds version 2's formation RNG (rotation and obliquity when the planet is not locked)."""
    r = rules or Formation3Rules()
    r.validate()
    gh, n_share = _earth_calibration(r)
    return _build(inputs, seed, r, gh, n_share, reference=_earth_reference(r))


def _solve(inputs: dict, base: F.PlanetSpec, r: Formation3Rules, gh: Greenhouse, reference, n_share: float) -> dict:
    """The fixed point of radius, gravity, atmosphere and climate (the atmosphere is ~1e-6 of M). ``n_share`` is
    the nitrogen degassed share (water and carbon take ``degassed_share``)."""
    star_in, planet_in, cloud = inputs["star"], inputs["planet"], inputs["cloud"]
    mass, cmf, age = base.planet_mass_kg, base.core_mass_fraction, base.star_age_yr
    t_disc = float(disc_temperature_k(float(star_in["luminosity_lsun"]), float(planet_in["orbit_au"])))
    solids = solid_volatiles(t_disc, cloud, r.classes)
    share = degassed_share(r.classes)
    water, carbon, nitrogen = share * solids["h2o"] * mass, share * solids["c"] * mass, n_share * solids["n"] * mass
    lith = cloud_scale(cloud, "other")
    rho0, comp = base.uncompressed_density_kg_m3, base.compression
    t_chain = float(planet_in["t_eq_k"]) + float(planet_in["greenhouse_k"])
    p_o2 = base.oxygen_pal * _P_O2_EARTH
    tau_other = gh.tau_other(base.oxygen_pal, r.other_gases)
    s_r = ridge_share(r)
    out = {"t_disc": t_disc, "solids": solids, "share": share, "n_share": n_share, "water": water, "carbon": carbon,
           "nitrogen": nitrogen, "lith": lith, "t_chain": t_chain, "tau_other": tau_other, "ridge_share": s_r}
    dry = vap = 0.0
    t_w = None
    converged = False
    for _ in range(100):
        ocean = water - vap
        solid = mass - water - dry
        r_solid = (3.0 * solid / (4.0 * math.pi * rho0 * comp)) ** (1.0 / 3.0)
        radius = (r_solid ** 3 + 3.0 * (ocean / C.RHO_WATER) / (4.0 * math.pi)) ** (1.0 / 3.0)
        g = C.G * mass / radius ** 2
        area = 4.0 * math.pi * radius ** 2
        mantle = (1.0 - cmf) * solid
        q = radiogenic_heat_w_m2(lith, age, mantle, radius)
        f_ocean = ocean_fraction(ocean / C.RHO_WATER, radius, basin_depth_at(g, s_r))
        land = 1.0 - f_ocean
        if reference is None:
            heat_rel = v_rel = land_rel = 1.0
        else:
            q_e, sigma_c_e, land_e = reference
            heat_rel = q / q_e
            v_rel = heat_rel * ((carbon / area) / sigma_c_e)
            land_rel = land / land_e
        potassium = r.k_bse * lith * mantle
        ar_made = argon_produced_kg(potassium, age)
        ar_air = r.argon_degassed * ar_made
        col = _Column(gh, r, base.t_eq_k, p_o2, gh.tau_cloud, tau_other, g,
                      nitrogen / _M["N2"] / area, ar_air / _M["Ar"] / area, carbon / _M_C / area)

        def steady(t, mode=r.weathering, v_rel=v_rel, land_rel=land_rel, heat_rel=heat_rel):
            return steady_co2_pa(t, v_rel, land_rel, heat_rel, r, mode)

        if r.weathering_t == "chain":
            t_w = t_chain
            st = col.climate(steady(t_w))
        else:
            t_w, st = col.fixed_point(steady, guess=t_w)
        t_s = st["t_s"]
        p_dry = st["p_dry"]
        column = vapour_column(t_s, g, r)
        dry_next, vap_next = p_dry * area / g, column * area
        done = abs(dry_next - dry) + abs(vap_next - vap) <= 1e-13 * mass
        dry, vap = dry_next, vap_next
        if done:
            converged = True
            break
    out.update(dry=dry, vap=vap, col=col, st=st, t_w=t_w, steady=steady, heat_rel=heat_rel, v_rel=v_rel,
               land_rel=land_rel, potassium=potassium, ar_made=ar_made, ar_air=ar_air, p_dry=p_dry, column=column,
               converged=converged)
    return out


def _build(inputs: dict, seed: int, r: Formation3Rules, gh: Greenhouse, n_share: float, reference) -> PlanetSpec3:
    base = F.build_planet(inputs, seed, r.base)
    v = {name: getattr(base, name) for name in _V2_KEPT}
    prov = {name: base.provenance[name] for name in _V2_KEPT}

    def put(name, value, tag, note):
        v[name] = value
        prov[name] = (tag, note)

    star_in, cloud = inputs["star"], inputs["cloud"]
    mass, cmf, age = base.planet_mass_kg, base.core_mass_fraction, base.star_age_yr
    z = base.metallicity
    S = _solve(inputs, base, r, gh, reference, n_share)
    solids, share, t_disc = S["solids"], S["share"], S["t_disc"]
    water, carbon, nitrogen, lith = S["water"], S["carbon"], S["nitrogen"], S["lith"]
    col, st, t_w, steady = S["col"], S["st"], S["t_w"], S["steady"]
    dry, vap = S["dry"], S["vap"]
    # ---- the solids and the degassed share
    put("disc_temperature_k", t_disc, "chain+derived",
        f"BLACK_BODY {BLACK_BODY} K x L^(1/4) / sqrt(a) with the chain's L {float(star_in['luminosity_lsun']):.6g} L_sun "
        f"and orbit {float(inputs['planet']['orbit_au']):.6g} au (world7's black body)")
    put("frost_temperature_k", FROST_T_K, "chain", PROVENANCE["frost_t_k"][1])
    put("ice_fraction", solids["ice"], "chain", PROVENANCE["ice_share"][1] + "; 0 inside the frost line")
    put("solid_water_fraction", solids["h2o"], "reference+new_rule",
        "ice + (1 - ice) x the meteorite-anchored water at the disc temperature (" + PROVENANCE["interpolation"][1] + ")")
    put("carbon_cloud_scale", cloud_scale(cloud, "carbon"), "chain+reference", PROVENANCE["cloud_scale"][1])
    put("nitrogen_cloud_scale", cloud_scale(cloud, "nitrogen"), "chain+reference", PROVENANCE["cloud_scale"][1])
    put("lithophile_cloud_scale", lith, "chain+reference+new_rule",
        PROVENANCE["lithophile_scale"][1] + f" (Z / Z_sun would be {z / C.Z_SUN:.4g})")
    put("solid_carbon_fraction", solids["c"], "reference+new_rule+chain",
        "(1 - ice) x the meteorite-anchored carbon at the disc temperature x carbon_cloud_scale")
    put("solid_nitrogen_fraction", solids["n"], "reference+new_rule+chain",
        "(1 - ice) x the meteorite-anchored nitrogen at the disc temperature x nitrogen_cloud_scale")
    put("degassed_share", share, "new_rule+derived", PROVENANCE["degassed_share"][1] + f" = {share:.6g}")
    put("nitrogen_degassed_share", S["n_share"], "new_rule+derived+reference",
        (PROVENANCE["nitrogen_degassed_share"][1] + f" = {S['n_share']:.6g}" if r.nitrogen_calibration == "earth_n2"
         else "the water's degassed share (nitrogen_calibration 'water', the earlier single share)"))
    bulk_w, bulk_c, bulk_n = solids["h2o"] * mass, solids["c"] * mass, solids["n"] * mass
    put("bulk_water_kg", bulk_w, "derived", "solid_water_fraction x planet mass")
    put("surface_water_kg", water, "derived", "degassed_share x bulk water: the ocean plus the vapour")
    put("undegassed_water_kg", bulk_w - water, "derived", "bulk minus surface water (kept in the interior or lost)")
    put("bulk_carbon_kg", bulk_c, "derived", "solid_carbon_fraction x planet mass")
    put("surface_carbon_kg", carbon, "derived", "degassed_share x bulk carbon: air CO2 plus carbonate rock")
    put("undegassed_carbon_kg", bulk_c - carbon, "derived", "bulk minus surface carbon")
    put("bulk_nitrogen_kg", bulk_n, "derived", "solid_nitrogen_fraction x planet mass")
    put("air_nitrogen_kg", nitrogen, "derived", "nitrogen_degassed_share x bulk nitrogen, all as N2 in the air")
    put("undegassed_nitrogen_kg", bulk_n - nitrogen, "derived", "bulk minus degassed nitrogen")
    put("water_mass_fraction", water / mass, "derived+reference+new_rule",
        "surface water / planet mass = degassed_share x solid_water_fraction (replaces version 2's assumed Earth value)")
    # ---- final geometry from the converged atmosphere
    rho0, comp = base.uncompressed_density_kg_m3, base.compression
    ocean = water - vap
    solid = mass - water - dry
    r_solid = (3.0 * solid / (4.0 * math.pi * rho0 * comp)) ** (1.0 / 3.0)
    radius = (r_solid ** 3 + 3.0 * (ocean / C.RHO_WATER) / (4.0 * math.pi)) ** (1.0 / 3.0)
    core = cmf * solid
    put("core_mass_kg", core, "derived", "core_mass_fraction x (planet - surface water - atmosphere)")
    put("mantle_mass_kg", solid - core, "derived", "silicate mantle and crust: (1 - core fraction) x solid mass "
        "(the undegassed volatiles are part of it)")
    put("ocean_mass_kg", ocean, "derived", "surface water less the vapour in the air")
    put("atmosphere_mass_kg", dry + vap, "derived",
        "dry air (P - p_H2O) 4 pi R^2 / g plus the vapour column x 4 pi R^2 (water vapour is not well mixed)")
    put("radius_m", radius, "derived+new_rule",
        "solid sphere of density uncompressed x compression, plus the ocean as a liquid-water shell (1000 kg/m^3)")
    put("core_radius_m", (3.0 * core / (4.0 * math.pi * C.RHO_IRON * comp)) ** (1.0 / 3.0), "derived+new_rule",
        "core of iron 7870 kg/m^3 x the same compression factor")
    put("mean_density_kg_m3", mass / (4.0 / 3.0 * math.pi * radius ** 3), "derived", "M / (4/3 pi R^3)")
    g = C.G * mass / radius ** 2
    area = 4.0 * math.pi * radius ** 2
    put("gravity_m_s2", g, "derived", "G M / R^2")
    put("escape_velocity_m_s", math.sqrt(2.0 * C.G * mass / radius), "derived", "sqrt(2 G M / R)")
    put("ocean_volume_m3", ocean / C.RHO_WATER, "derived", "ocean mass / 1000 kg/m^3")
    put("ocean_layer_m", v["ocean_volume_m3"] / area, "derived", "global equivalent layer: ocean volume / 4 pi R^2")
    # ---- atmosphere
    t_s = st["t_s"]
    mu_dry = st["mu_dry"]
    pp = {gas: st["n"][gas] * col.g * mu_dry for gas in _DRY}
    pp = {"N2": pp["N2"], "O2": pp["O2"], "CO2": pp["CO2"], "Ar": pp["Ar"],
          "H2O": r.relative_humidity * e_sat_pa(t_s)}
    put("oxygen_pal", base.oxygen_pal, "chain", base.provenance["oxygen_pal"][1])
    put("partial_pressure_pa", dict(pp), "derived+reference+new_rule+chain",
        "well-mixed dry column p_i = n_i g mu_dry: N2 all degassed nitrogen; Ar the degassed radiogenic Ar-40; O2 "
        "oxygen_pal x Earth's 21.2 kPa (chain); CO2 the carbon-silicate steady state (co2_ref_pa); H2O relative "
        f"humidity {r.relative_humidity:g} x Magnus e_sat at the derived T_s")
    pressure = sum(pp.values())
    put("surface_pressure_pa", pressure, "derived", "sum of the partial pressures (with the local surface vapour "
        "pressure; PROVENANCE['surface_pressure_hydrostatic'])")
    put("surface_pressure_hydrostatic_pa", sum(pp[gas] for gas in _DRY) + S["column"] * g, "derived",
        PROVENANCE["surface_pressure_hydrostatic"][1])
    x = {gas: p / pressure for gas, p in pp.items()}
    mu = sum(x[gas] * _M[gas] for gas in pp)
    cp_molar = sum(x[gas] * C.CP_MOLAR_298[gas] for gas in pp)
    put("dry_air_molar_mass_kg_mol", mu_dry, "derived", "molar mass of the dry column: sum n_i M_i / sum n_i")
    put("oxygen_mole_fraction", x["O2"], "derived", "p_O2 / P")
    put("air_molar_mass_kg_mol", mu, "derived", "sum x_i mu_i (molar masses from atomic_weights.json)")
    put("air_cp_j_kg_k", cp_molar / mu, "derived+reference", "sum x_i Cp_i (NIST-JANAF, 298 K) / mu")
    put("air_gamma", cp_molar / (cp_molar - C.R_GAS), "derived", "Cp / (Cp - R), ideal gas")
    put("scale_height_m", C.R_GAS * t_s / (mu * g), "derived", "R T_s / (mu g) at the derived T_s")
    put("air_density_kg_m3", pressure * mu / (C.R_GAS * t_s), "derived", "P mu / (R T_s) at the derived T_s")
    put("sound_speed_m_s", math.sqrt(v["air_gamma"] * C.R_GAS * t_s / mu), "derived", "sqrt(gamma R T_s / mu)")
    gamma_lapse = r.lapse_rate_k_m * g / C.G_STANDARD
    put("vapour_scale_height_m", F.vapour_scale_height_m(t_s, gamma_lapse), "derived+reference+new_rule",
        f"R_v T_s^2 / (L_v Gamma) at the derived T_s, Gamma = {r.lapse_rate_k_m * 1e3:g} K/km x g / g_E (version 2's rule)")
    put("vapour_column_kg_m2", S["column"], "derived+reference+new_rule",
        "p_H2O mu_w / (R T_s) x vapour_scale_height_m at the derived T_s (version 2's rule)")
    # ---- volatile budgets in the air
    area_loop = C.G * mass / col.g * 4.0 * math.pi
    air_c = st["n"]["CO2"] * _M_C * area_loop
    put("air_carbon_kg", air_c, "derived", "n_CO2 x M_C x 4 pi R^2: the carbon of the air's CO2")
    put("carbonate_carbon_kg", max(0.0, carbon - air_c), "derived", "surface carbon less the air's: carbonate rock "
        "(and ocean dissolved carbon); 0 when the air holds it all")
    put("potassium_kg", S["potassium"], "reference+new_rule", PROVENANCE["k_bse"][1] + " x mantle mass")
    put("argon_produced_kg", S["ar_made"], "derived+reference",
        f"{K40_AR_BRANCH:.4f} x initial K-40 (K x 1.17e-4 by atoms, decayed back over Earth's 4.567 Gyr) x "
        f"(1 - exp(-lambda t)) over the planet's age {age / 1e9:.4g} Gyr, x M_Ar")
    put("air_argon_kg", S["ar_air"], "reference+derived",
        f"{r.argon_degassed:g} x argon_produced_kg (Earth's degassed share)")
    put("retained_argon_kg", S["ar_made"] - S["ar_air"], "derived", "argon_produced_kg less the air's")
    # ---- carbon-silicate steady state
    v_rel, land_rel, heat_rel = S["v_rel"], S["land_rel"], S["heat_rel"]
    p_co2 = pp["CO2"]
    w_land, w_sea = weathering_terms(p_co2, t_w, land_rel, heat_rel, r)
    land_scaled = r.weathering != "global"
    put("heat_rel_earth", heat_rel, "derived", "radiogenic heat flux / the model Earth's")
    put("outgassing_rel_earth", v_rel, "new_rule+derived",
        "radiogenic heat flux / the model Earth's x degassed carbon per m^2 / the model Earth's")
    put("land_rel_earth", land_rel, "derived", "land_fraction / the model Earth's")
    put("weathering_land_scaled", land_scaled, "new_rule", PROVENANCE["weathering"][1])
    put("seafloor_weathering_share", r.seafloor_share if r.weathering == "land_seafloor" else 0.0,
        "reference+new_rule", PROVENANCE["seafloor_share"][1] + " (0 unless weathering is 'land_seafloor')")
    put("weathering_land_share", w_land / (w_land + w_sea) if w_land + w_sea > 0.0 else 0.0, "derived",
        "the continental share of the steady-state weathering (0: a planet without land weathers on its seafloor)")
    put("weathering_at_chain_t", r.weathering_t == "chain", "new_rule", PROVENANCE["weathering_t"][1])
    put("weathering_t_k", t_w, "chain" if r.weathering_t == "chain" else "derived",
        "the T at which weathering balances outgassing: the chain's t_eq + greenhouse (the spec's literal reading)"
        if r.weathering_t == "chain" else
        "the carbon-silicate fixed point: weathering at the planet's own derived T_s balances outgassing")
    put("co2_carbon_limited", bool(st["limited"]), "derived", PROVENANCE["carbon_cap"][1])
    put("co2_beyond_fit", bool(p_co2 > r.co2_fit_max_pa), "derived+reference", PROVENANCE["co2_fit_max_pa"][1])
    put("co2_ref_pa", p_co2, "derived+reference+new_rule",
        f"carbon-silicate steady state ({r.weathering}) at {t_w:.4g} K: weathering_terms(p, T) = outgassing_rel_earth, "
        "p_E 28.17 Pa, at most all degassed carbon")
    put("co2_capped", False, "derived", "version 2's CO2 cap is gone: CO2 comes from the carbon cycle, not from T_s "
        "(co2_beyond_fit reports a CO2 outside the forcing fits)")
    put("tau_residual", 0.0, "derived", "no greenhouse of no named gas: tau is computed forward")
    # ---- greenhouse
    p_rel = st["p_rel"]
    tau_w = float(gh.tau_h2o(S["column"], p_rel))
    tau_fixed = col.tau_cloud + col.tau_other
    tau = tau_fixed + st["tau_c"] + tau_w
    put("dry_pressure_rel_model_earth", p_rel, "derived",
        f"dry surface pressure / P_cal {gh.p_cal_pa:.6g} Pa (the model Earth's): the broadening factor's base")
    put("pressure_beyond_fit", bool(p_rel > r.pressure_fit_factor), "new_rule", PROVENANCE["pressure_fit_factor"][1])
    put("tau_h2o", tau_w, "derived+reference",
        f"{gh.tau_w_earth:.5f} (u_w (P / P_cal)^{gh.broadening:g} / {gh.u_w_earth:.4g} kg/m^2)^{gh.n_w:.4f}")
    put("tau_co2", st["tau_c"], "derived+reference",
        f"{gh.tau_c_earth:.5f} (u_c (P / P_cal)^{gh.broadening:g} / {gh.u_c_earth:.4g} kg/m^2)^{gh.n_c:.4f}")
    put("tau_cloud", col.tau_cloud, "reference+new_rule", PROVENANCE["cloud_share"][1])
    put("tau_other_gases", col.tau_other, "reference+new_rule+chain",
        PROVENANCE["other_gases"][1] + f" (rule {r.other_gases!r})")
    put("tau_rest", tau_fixed, "reference+new_rule", "tau_cloud + tau_other_gases")
    put("greenhouse_tau", tau, "derived", "tau_cloud + tau_other_gases + tau_co2 + tau_h2o (forward, at the derived T_s)")
    put("runaway", bool(st["runaway"]), "derived", f"no energy-balance root below {r.t_max_k:g} K")
    put("frozen_mean", bool(t_s < T_FREEZE_K), "derived+new_rule", PROVENANCE["frozen"][1])
    put("t_surface_target_k", t_s, "derived",
        "lowest root of T^4 = T_eq^4 (1 + 0.75 tau(T)): the planet's own T_s (no target when frozen_mean)")
    put("chain_t_target_k", S["t_chain"], "chain", "planet.t_eq_k + planet.greenhouse_k: world7's unrounded T_s")
    put("chain_t_s_mismatch_k", t_s - S["t_chain"], "derived",
        "t_surface_target_k - chain_t_target_k (a finding, not a fix)")
    put("tau_per_ln_co2", gh.n_c * st["tau_c"], "derived", "d tau / d ln p_CO2 = n_c tau_co2 at fixed T")
    # both readings of the steady state, whichever drives the spec (same geometry), and the global law
    if r.weathering_t == "self_consistent":
        st_sc, st_ch = st, col.climate(steady(S["t_chain"]))
    else:
        st_ch, st_sc = st, col.fixed_point(steady)[1]
    st_gl = st_sc if r.weathering == "global" else col.fixed_point(lambda t: steady(t, "global"))[1]

    def p_of(state):
        return state["n"]["CO2"] * col.g * state["mu_dry"]

    put("t_surface_selfconsistent_k", st_sc["t_s"], "derived",
        "T_s at the carbon-silicate fixed point (weathering at the planet's own T_s balances outgassing)")
    put("co2_selfconsistent_pa", p_of(st_sc), "derived", "the CO2 of that fixed point")
    put("t_surface_chain_weathering_k", st_ch["t_s"], "derived",
        "T_s with CO2 from weathering = outgassing at the chain's temperature (the spec's literal reading)")
    put("co2_chain_weathering_pa", p_of(st_ch), "derived", "the CO2 of weathering = outgassing at the chain's temperature")
    put("t_surface_global_weathering_k", st_gl["t_s"], "derived",
        "T_s at the fixed point of the published global WHAK law (a continental thermostat on the whole planet, "
        "land or not; the earlier default)")
    put("co2_global_weathering_pa", p_of(st_gl), "derived", "the CO2 of that fixed point")
    put("t_surface_no_clouds_k", col.climate(p_co2, clouds=False)["t_s"], "derived",
        "T_s without tau_cloud at the same CO2 and air (vapour follows T): the cloud placeholder's warming is "
        "t_surface_target_k - this")
    put("t_surface_no_other_gases_k", col.climate(p_co2, other=False)["t_s"], "derived",
        "T_s without tau_other_gases at the same CO2 and air: their warming is t_surface_target_k - this")
    put("t_surface_no_broadening_k", col.climate(p_co2, broadening=False)["t_s"], "derived",
        "T_s with the absorber paths at P_cal (no pressure broadening) at the same CO2 and air")
    put("fixed_point_converged", bool(S["converged"]), "derived",
        "the radius / atmosphere loop met 1e-13 of the planet's mass within 100 iterations")
    # ---- relief and isostasy
    peak = r.everest_m * G_EARTH_SURFACE / g
    s_r = S["ridge_share"]
    depth = basin_depth_at(g, s_r)
    f_ocean = ocean_fraction(v["ocean_volume_m3"], radius, depth)
    put("peak_height_m", peak, "reference+derived", PROVENANCE["everest_m"][1])
    put("mean_land_height_m", r.mean_land_height_m * G_EARTH_SURFACE / g, "reference+new_rule",
        PROVENANCE["mean_land_height_m"][1])
    put("basin_depth_ridge_share", s_r, "reference+derived", PROVENANCE["ridge_depth_m"][1]
        + f" (rule {r.basin_depth_rule!r})")
    put("basin_depth_m", depth, "derived+reference+new_rule",
        f"D_E {basin_depth_m():.1f} m x ({s_r:.4f} g_E / g + {1.0 - s_r:.4f}): " + PROVENANCE["basin_depth"][1])
    put("ocean_fraction", f_ocean, "derived+new_rule", "min(1, ocean volume / (4 pi R^2 basin_depth_m))")
    put("ocean_fraction_fixed_depth", ocean_fraction(v["ocean_volume_m3"], radius, basin_depth_at(g, 0.0)),
        "derived+new_rule", "the ocean fraction if the basin depth did not depend on g (D = D_E)")
    put("ocean_fraction_inverse_g_depth", ocean_fraction(v["ocean_volume_m3"], radius, basin_depth_at(g, 1.0)),
        "derived+new_rule", "the ocean fraction if the whole basin depth scaled as 1/g (D = D_E g_E / g)")
    put("land_fraction", 1.0 - f_ocean, "derived", "1 - ocean_fraction (the continental platform above the sea)")
    put("no_land_weathering_assumed", bool(not land_scaled and f_ocean >= 1.0), "derived",
        "a planet without land given continental weathering (the 'global' rule drives the spec)")
    put("flood_depth_m", max(0.0, v["ocean_layer_m"] - depth), "derived",
        "water over the continental platform when the basins overflow: ocean layer - basin depth, at least 0")
    put("relief_m", peak + depth, "derived+reference+new_rule",
        "real scale: peak_height_m (strength-limited) + basin_depth_m (isostatic): the scale of the relief. Build the "
        "terrain with make_terrain3 (version 2's globe.make_terrain spreads relief_m linearly and puts the sea "
        "elsewhere)")
    # ---- radiogenic heat (per kg of rock x the lithophile scale, on the v3 mantle)
    q = radiogenic_heat_w_m2(lith, age, v["mantle_mass_kg"], radius)
    put("radiogenic_heat_w_m2", q, "reference+new_rule",
        PROVENANCE["radiogenic_heat"][1] + f": lithophile scale {lith:.4g} (Z / Z_sun {z / C.Z_SUN:.4g} is not used), "
        f"decay {F.radiogenic_decay_factor(age):.4g}")
    return PlanetSpec3(**v, provenance=prov)


# ---------------------------------------------------------------------------------------------
# the terrain


def make_terrain3(globe, gen: torch.Generator, specs, rules: Formation3Rules | None = None,
                  iters: int = gb.SEA_LEVEL_ITERS) -> dict:
    """Real-scale terrain for W = len(specs) worlds whose ocean covers each spec's ``ocean_fraction``.

    Version 2's height field (``globe.great_circle_faults`` and ``globe.plane_waves``, the same RNG draws as
    ``globe.make_terrain``) ranks the cells by area. The lowest ``ocean_fraction`` of the area is ocean, with
    depth D_max (1 - (1 - w)^k) below the continental platform (w the area rank from the coast) scaled so its
    area mean is ``basin_depth_m`` (when the basins overflow, every cell is basin and the sea stands
    ``flood_depth_m`` over the platform). Land rises above the platform as a truncated exponential of its area
    rank (mean ``mean_land_height_m``, top ``peak_height_m``). The sea level is then solved for each spec's
    ``ocean_volume_m3`` with ``globe.sea_level`` (exact shells about ``radius_m``), so ocean volume and area
    both match the spec. Returns the keys of ``globe.make_terrain`` (elevation_m float32 from the lowest
    ground, sea_level_m, depth_m, land, coast, inland_rad, ocean_fraction, ocean_volume_m3)."""
    r = rules or Formation3Rules()
    specs = list(specs)
    W, dev, f64 = len(specs), globe.device, torch.float64

    def per(name):
        return torch.tensor([float(getattr(s, name)) for s in specs], dtype=f64, device=dev)[:, None]

    f, depth, peak, h_land = per("ocean_fraction"), per("basin_depth_m"), per("peak_height_m"), per("mean_land_height_m")
    height = gb.FAULT_SHARE * gb.great_circle_faults(globe, W, gen) + (1 - gb.FAULT_SHARE) * gb.plane_waves(globe, W, gen)
    order = torch.sort(height, dim=-1, stable=True).indices
    area = globe.area64
    a_sorted = area[order]
    u_sorted = (torch.cumsum(a_sorted, -1) - 0.5 * a_sorted) / area.sum()
    u = torch.empty_like(u_sorted).scatter_(-1, order, u_sorted)        # area rank of every cell, 0-1
    ocean = u < f
    w = ((f - u) / f.clamp_min(1e-300)).clamp(0.0, 1.0)
    shape = 1.0 - (1.0 - w) ** r.ocean_hypsometry_k
    a = area[None, :] * ocean
    mean_shape = (a * shape).sum(-1, keepdim=True) / a.sum(-1, keepdim=True).clamp_min(1e-300)
    basin = depth * shape / mean_shape.clamp_min(1e-300)
    v = ((u - f) / (1.0 - f).clamp_min(1e-300)).clamp(0.0, 1.0)
    rise = h_land * -torch.log1p(-v * -torch.expm1(-peak / h_land))
    z = torch.where(ocean, -basin, rise)
    elevation = (z - z.amin(-1, keepdim=True)).float()
    volume = per("ocean_volume_m3")[:, 0]
    radius = per("radius_m")[:, 0]
    sea = gb.sea_level(globe, elevation, volume, radius, iters)
    land = elevation >= sea[:, None]
    wet = ~land
    return {"elevation_m": elevation, "sea_level_m": sea, "depth_m": (sea[:, None] - elevation).clamp_min(0),
            "land": land, "coast": land & wet[:, globe.nbr].any(-1), "inland_rad": gb.inland_distance(globe, land),
            "ocean_fraction": ((wet.double() * area).sum(-1) / (4 * math.pi)).float(),
            "ocean_volume_m3": gb.ocean_volume(globe, elevation, sea, radius)}


# ---------------------------------------------------------------------------------------------
# checks and text

#: version-2 checks that compare against the chain's T_s backwards; v3 reports the mismatch instead
SUPERSEDED = {
    "grey T_s vs the chain's whole-K T_s": "v3 derives T_s forward; the difference is chain_t_s_mismatch_k",
    "CO2 tau + residual = greenhouse_tau": "v3's tau is forward (tau_cloud + tau_other + tau_co2 + tau_h2o, checked "
                                           "here)",
    "chain_t_s_mismatch_k is negative": "a signed difference by design",
    "grey T_s misses the target (runaway planets only)": "a runaway planet has no energy balance below t_max_k; "
                                                         "its T_s is that bound and the flag says so",
}


def _superseded(msg: str, runaway: bool = False) -> bool:
    return ((msg.startswith("grey T_s ") and "from the chain's" in msg) or msg.startswith("CO2 tau ")
            or msg.startswith("chain_t_s_mismatch_k is negative")
            or (runaway and msg.startswith("grey T_s ") and "misses the target" in msg))


def _mode_of(spec: PlanetSpec3) -> str:
    if not spec.weathering_land_scaled:
        return "global"
    return "land_seafloor" if spec.seafloor_weathering_share > 0.0 else "land"


def check_spec3(spec: PlanetSpec3, rules: Formation3Rules | None = None, inputs: dict | None = None) -> list[str]:
    """The failures of a v3 spec (empty when it passes): version 2's checks (less ``SUPERSEDED``) plus the
    water, carbon, nitrogen and argon budgets, the two degassed shares (on Earth's inputs, ``source`` 'earth_reference',
    also Earth's surface water and Earth's N2), the potassium and heat flux, the outgassing and land ratios, the
    disc temperature, the steady state and its temperature, the greenhouse, the flags and the relief. With
    ``inputs`` (the chain inputs the spec was built from) the cloud scales and the disc temperature are also
    recomputed from them."""
    r = rules or Formation3Rules()
    bad = [msg for msg in F.check_spec(spec) if not _superseded(msg, bool(spec.runaway))]

    def close(name, a, b, rel=1e-9, floor=0.0):
        if not abs(a - b) <= rel * max(abs(a), abs(b)) + floor:
            bad.append(f"{name}: {a!r} differs from {b!r}")

    def flag(name, value, expected):
        if bool(value) != bool(expected):
            bad.append(f"{name} is {value}, expected {expected}")

    if not spec.fixed_point_converged:
        bad.append("the radius / atmosphere fixed point did not converge")
    lum, orbit = spec.star_luminosity_w / C.L_SUN, spec.orbit_m / C.AU
    close("disc temperature", spec.disc_temperature_k, BLACK_BODY * lum ** 0.25 / math.sqrt(orbit))
    if inputs is not None:
        cloud = inputs["cloud"]
        close("carbon cloud scale (inputs)", spec.carbon_cloud_scale, cloud_scale(cloud, "carbon"))
        close("nitrogen cloud scale (inputs)", spec.nitrogen_cloud_scale, cloud_scale(cloud, "nitrogen"))
        close("lithophile cloud scale (inputs)", spec.lithophile_cloud_scale, cloud_scale(cloud, "other"))
        close("disc temperature (inputs)", spec.disc_temperature_k,
              float(disc_temperature_k(float(inputs["star"]["luminosity_lsun"]), float(inputs["planet"]["orbit_au"]))))
    ice = ICE_SHARE if spec.disc_temperature_k < FROST_T_K else 0.0
    close("ice fraction", spec.ice_fraction, ice, floor=1e-15)
    t = spec.disc_temperature_k
    close("solid water", spec.solid_water_fraction, ice + (1 - ice) * anchored_fraction(t, "h2o", r.classes))
    close("solid carbon", spec.solid_carbon_fraction,
          (1 - ice) * anchored_fraction(t, "c", r.classes) * spec.carbon_cloud_scale)
    close("solid nitrogen", spec.solid_nitrogen_fraction,
          (1 - ice) * anchored_fraction(t, "n", r.classes) * spec.nitrogen_cloud_scale)
    if not 0.0 < spec.degassed_share <= 1.0:
        bad.append(f"degassed share {spec.degassed_share} outside (0, 1]")
    close("degassed share", spec.degassed_share, degassed_share(r.classes))
    if not 0.0 < spec.nitrogen_degassed_share <= 1.0:
        bad.append(f"nitrogen degassed share {spec.nitrogen_degassed_share} outside (0, 1]")
    close("nitrogen degassed share", spec.nitrogen_degassed_share, nitrogen_degassed_share(r))
    if spec.source == "earth_reference":
        # the calibration planet holds the two Earth calibrations
        close("Earth's surface water", spec.water_mass_fraction, C.EARTH_OCEAN_MASS_FRACTION)
        if r.nitrogen_calibration == "earth_n2":
            close("Earth's N2", spec.partial_pressure_pa["N2"], EARTH_N2_PA)
    m = spec.planet_mass_kg
    area = 4.0 * math.pi * spec.radius_m ** 2
    g = spec.gravity_m_s2
    # water
    close("bulk water", spec.bulk_water_kg, spec.solid_water_fraction * m)
    close("water budget", spec.surface_water_kg + spec.undegassed_water_kg, spec.bulk_water_kg)
    close("surface water", spec.surface_water_kg, spec.degassed_share * spec.bulk_water_kg)
    close("ocean + vapour", spec.ocean_mass_kg + spec.vapour_column_kg_m2 * area, spec.surface_water_kg)
    close("water mass fraction", spec.water_mass_fraction * m, spec.surface_water_kg)
    # the dry column
    p_dry = sum(spec.partial_pressure_pa[gas] for gas in _DRY)
    mu_dry = (sum(spec.partial_pressure_pa[gas] * _M[gas] for gas in _DRY) / p_dry) if p_dry > 0 else _M["N2"]
    close("dry molar mass", spec.dry_air_molar_mass_kg_mol, mu_dry)

    def column_kg(gas, per_atom):
        return spec.partial_pressure_pa[gas] / (g * mu_dry) * per_atom * area

    # carbon
    close("bulk carbon", spec.bulk_carbon_kg, spec.solid_carbon_fraction * m)
    close("surface carbon", spec.surface_carbon_kg, spec.degassed_share * spec.bulk_carbon_kg)
    close("carbon budget", spec.air_carbon_kg + spec.carbonate_carbon_kg + spec.undegassed_carbon_kg,
          spec.bulk_carbon_kg)
    close("air carbon", spec.air_carbon_kg, column_kg("CO2", _M_C), floor=1e-30)
    if spec.carbonate_carbon_kg < -1e-9 * spec.surface_carbon_kg:
        bad.append(f"carbonate carbon is negative ({spec.carbonate_carbon_kg})")
    # nitrogen
    close("bulk nitrogen", spec.bulk_nitrogen_kg, spec.solid_nitrogen_fraction * m)
    close("nitrogen budget", spec.air_nitrogen_kg + spec.undegassed_nitrogen_kg, spec.bulk_nitrogen_kg)
    close("air nitrogen", spec.air_nitrogen_kg, spec.nitrogen_degassed_share * spec.bulk_nitrogen_kg)
    close("air N2 column", spec.air_nitrogen_kg, column_kg("N2", _M["N2"]))
    # potassium, heat and argon
    close("potassium", spec.potassium_kg, r.k_bse * spec.lithophile_cloud_scale * spec.mantle_mass_kg)
    close("radiogenic heat", spec.radiogenic_heat_w_m2,
          radiogenic_heat_w_m2(spec.lithophile_cloud_scale, spec.star_age_yr, spec.mantle_mass_kg, spec.radius_m))
    close("argon made", spec.argon_produced_kg, argon_produced_kg(spec.potassium_kg, spec.star_age_yr))
    close("argon budget", spec.air_argon_kg + spec.retained_argon_kg, spec.argon_produced_kg)
    close("air argon", spec.air_argon_kg, r.argon_degassed * spec.argon_produced_kg)
    close("air Ar column", spec.air_argon_kg, column_kg("Ar", _M["Ar"]))
    close("O2", spec.partial_pressure_pa["O2"], spec.oxygen_pal * _P_O2_EARTH, floor=1e-12)
    # the outgassing and land ratios against the model Earth
    q_e, sigma_e, land_e = _earth_reference(r)
    close("heat ratio", spec.heat_rel_earth, spec.radiogenic_heat_w_m2 / q_e)
    close("outgassing ratio", spec.outgassing_rel_earth, spec.heat_rel_earth * (spec.surface_carbon_kg / area) / sigma_e)
    close("land ratio", spec.land_rel_earth, spec.land_fraction / land_e, floor=1e-15)
    # the steady state and its temperature
    mode = _mode_of(spec)
    if mode == "land_seafloor" and spec.seafloor_weathering_share != r.seafloor_share:
        bad.append("the spec's seafloor share differs from the rules' (built with other rules?)")
    if spec.weathering_at_chain_t:
        close("weathering T (chain)", spec.weathering_t_k, spec.chain_t_target_k, floor=1e-12)
    else:
        close("weathering T (own T_s)", spec.weathering_t_k, spec.t_surface_target_k, rel=0.0, floor=1e-6)
    w_land, w_sea = weathering_terms(spec.partial_pressure_pa["CO2"], spec.weathering_t_k, spec.land_rel_earth,
                                     spec.heat_rel_earth, r, mode)
    p_steady = steady_co2_pa(spec.weathering_t_k, spec.outgassing_rel_earth, spec.land_rel_earth,
                             spec.heat_rel_earth, r, mode)
    if spec.co2_carbon_limited:
        if not w_land + w_sea <= spec.outgassing_rel_earth * (1 + 1e-9):
            bad.append("carbon-limited CO2 weathers faster than outgassing")
        if not spec.partial_pressure_pa["CO2"] <= p_steady * (1 + 1e-9):
            bad.append("carbon-limited CO2 above the steady state")
        close("carbon-limited air carbon", spec.air_carbon_kg, spec.surface_carbon_kg)
    else:
        close("weathering = outgassing", w_land + w_sea, spec.outgassing_rel_earth)
        close("steady-state CO2", spec.partial_pressure_pa["CO2"], p_steady, floor=1e-300)
    close("land share of weathering", spec.weathering_land_share,
          w_land / (w_land + w_sea) if w_land + w_sea > 0 else 0.0, floor=1e-12)
    flag("no_land_weathering_assumed", spec.no_land_weathering_assumed, mode == "global" and spec.ocean_fraction >= 1.0)
    close("co2_ref_pa", spec.co2_ref_pa, spec.partial_pressure_pa["CO2"])
    flag("co2_beyond_fit", spec.co2_beyond_fit, spec.partial_pressure_pa["CO2"] > r.co2_fit_max_pa)
    # greenhouse
    gh = greenhouse(r)
    p_rel = p_dry / gh.p_cal_pa
    close("dry pressure ratio", spec.dry_pressure_rel_model_earth, p_rel)
    flag("pressure_beyond_fit", spec.pressure_beyond_fit, p_rel > r.pressure_fit_factor)
    close("tau_cloud", spec.tau_cloud, gh.tau_cloud)
    close("tau_other_gases", spec.tau_other_gases, gh.tau_other(spec.oxygen_pal, r.other_gases), floor=1e-15)
    close("tau_rest", spec.tau_rest, spec.tau_cloud + spec.tau_other_gases)
    close("tau components", spec.tau_rest + spec.tau_co2 + spec.tau_h2o, spec.greenhouse_tau)
    close("tau_h2o", spec.tau_h2o, float(gh.tau_h2o(spec.vapour_column_kg_m2, p_rel)))
    close("tau_co2", spec.tau_co2, float(gh.tau_co2(spec.air_carbon_kg / _M_C * _M["CO2"] / area, p_rel)), floor=1e-15)
    close("vapour", spec.vapour_column_kg_m2, vapour_column(spec.t_surface_target_k, g, r))
    if spec.runaway:
        close("runaway T_s", spec.t_surface_target_k, r.t_max_k)
    else:
        close("energy balance", spec.t_surface_target_k ** 4,
              spec.t_eq_k ** 4 * (1 + 0.75 * spec.greenhouse_tau), rel=1e-9)
    flag("frozen_mean", spec.frozen_mean, spec.t_surface_target_k < T_FREEZE_K)
    for name in ("t_surface_no_clouds_k", "t_surface_no_other_gases_k"):
        if getattr(spec, name) > spec.t_surface_target_k + 1e-6:
            bad.append(f"{name} is warmer than with the term")
    close("chain mismatch", spec.chain_t_s_mismatch_k, spec.t_surface_target_k - spec.chain_t_target_k, floor=1e-12)
    close("p_H2O", spec.partial_pressure_pa["H2O"], r.relative_humidity * e_sat_pa(spec.t_surface_target_k))
    # relief
    close("peak", spec.peak_height_m, r.everest_m * G_EARTH_SURFACE / g)
    close("mean land height", spec.mean_land_height_m, r.mean_land_height_m * G_EARTH_SURFACE / g)
    close("ridge share", spec.basin_depth_ridge_share, ridge_share(r))
    close("basin depth", spec.basin_depth_m, basin_depth_at(g, spec.basin_depth_ridge_share))
    close("relief", spec.relief_m, spec.peak_height_m + spec.basin_depth_m)
    close("ocean fraction", spec.ocean_fraction, ocean_fraction(spec.ocean_volume_m3, spec.radius_m, spec.basin_depth_m))
    close("ocean fraction (fixed D)", spec.ocean_fraction_fixed_depth,
          ocean_fraction(spec.ocean_volume_m3, spec.radius_m, basin_depth_at(g, 0.0)))
    close("ocean fraction (D ~ 1/g)", spec.ocean_fraction_inverse_g_depth,
          ocean_fraction(spec.ocean_volume_m3, spec.radius_m, basin_depth_at(g, 1.0)))
    close("land fraction", spec.land_fraction + spec.ocean_fraction, 1.0)
    close("flood depth", spec.flood_depth_m, max(0.0, spec.ocean_layer_m - spec.basin_depth_m), floor=1e-9)
    if not 0.0 <= spec.ocean_fraction <= 1.0:
        bad.append(f"ocean fraction {spec.ocean_fraction} outside [0, 1]")
    return bad


def fire_possible(x_o2_dry: float, pressure_pa: float) -> dict:
    """Whether a world's air allows fire (``PROVENANCE['fire_fit_pa']``): {"fire_possible": the dry O2 mole fraction
    against crafting.X_O2_MIN, or None (uncertain) when it passes but the total pressure lies outside FIRE_FIT_PA,
    "fire_beyond_fit"}."""
    beyond = not FIRE_FIT_PA[0] <= float(pressure_pa) <= FIRE_FIT_PA[1]
    ok = float(x_o2_dry) >= float(cr.X_O2_MIN)
    return {"fire_possible": None if (ok and beyond) else ok, "fire_beyond_fit": beyond}


def describe3(spec: PlanetSpec3) -> str:
    """One line per field (``planet.formation.describe`` with the v3 header and a note on the model Earth's air)."""
    text = F.describe(spec).replace(F.FORMATION_VERSION, FORMATION3_VERSION, 1)
    head, _, rest = text.partition("\n")
    earth = model_earth()
    note = (f"note: the model Earth (chain.earth_inputs, the calibration planet) has {earth.surface_pressure_pa / 1e5:.3g} "
            f"bar of air, N2 {earth.partial_pressure_pa['N2'] / 1e3:.1f} kPa against Earth's N2 {EARTH_N2_PA / 1e3:.1f} kPa "
            f"(nitrogen degassed share {earth.nitrogen_degassed_share:.4g}; water and carbon {earth.degassed_share:.4g}); "
            f"this planet's dry air is {spec.dry_pressure_rel_model_earth:.3g}x the model Earth's")
    return head + "\n" + note + ("\n" + rest if rest else "")


def summary_row(spec: PlanetSpec3) -> dict:
    """The derived-against-chain numbers of one planet (for reports). Pressures are absolute; the model Earth's
    air is about 1.03 bar (``model_earth``: Earth's N2 by the nitrogen degassed share). ``x_o2_dry`` is the O2 mole
    fraction of the dry air (the measure of world3's fire report) and ``fire_possible`` whether it reaches
    crafting.X_O2_MIN's 0.15 (Belcher et al. 2010): None (uncertain) when the total pressure lies outside
    FIRE_FIT_PA (``fire_beyond_fit``, ``PROVENANCE['fire_fit_pa']``)."""
    dry = sum(spec.partial_pressure_pa[gas] for gas in _DRY)
    x_dry = spec.partial_pressure_pa["O2"] / dry if dry > 0.0 else 0.0
    fire = fire_possible(x_dry, spec.surface_pressure_pa)
    return {"seed": spec.seed, "source": spec.source, "flux": spec.flux_earth, "t_disc_k": spec.disc_temperature_k,
            "water_mass_fraction": spec.water_mass_fraction, "ocean_fraction": spec.ocean_fraction,
            "ocean_fraction_fixed_depth": spec.ocean_fraction_fixed_depth,
            "ocean_fraction_inverse_g_depth": spec.ocean_fraction_inverse_g_depth,
            "lithophile_scale": spec.lithophile_cloud_scale, "heat_w_m2": spec.radiogenic_heat_w_m2,
            "outgassing_rel": spec.outgassing_rel_earth, "weathering_land_share": spec.weathering_land_share,
            "p_n2_pa": spec.partial_pressure_pa["N2"], "p_co2_pa": spec.partial_pressure_pa["CO2"],
            "p_ar_pa": spec.partial_pressure_pa["Ar"], "p_o2_pa": spec.partial_pressure_pa["O2"],
            "pressure_pa": spec.surface_pressure_pa, "x_o2": spec.oxygen_mole_fraction, "x_o2_dry": x_dry,
            "fire_possible": fire["fire_possible"], "fire_beyond_fit": fire["fire_beyond_fit"],
            "pressure_rel_model_earth": spec.dry_pressure_rel_model_earth,
            "chain_t_k": spec.chain_t_target_k, "t_s_k": spec.t_surface_target_k,
            "mismatch_k": spec.chain_t_s_mismatch_k, "t_selfconsistent_k": spec.t_surface_selfconsistent_k,
            "co2_selfconsistent_pa": spec.co2_selfconsistent_pa,
            "t_chain_weathering_k": spec.t_surface_chain_weathering_k,
            "t_global_weathering_k": spec.t_surface_global_weathering_k,
            "co2_global_weathering_pa": spec.co2_global_weathering_pa,
            "cloud_warming_k": spec.t_surface_target_k - spec.t_surface_no_clouds_k,
            "other_gases_warming_k": spec.t_surface_target_k - spec.t_surface_no_other_gases_k,
            "broadening_warming_k": spec.t_surface_target_k - spec.t_surface_no_broadening_k,
            "carbon_limited": spec.co2_carbon_limited, "co2_beyond_fit": spec.co2_beyond_fit,
            "pressure_beyond_fit": spec.pressure_beyond_fit, "frozen_mean": spec.frozen_mean,
            "no_land_weathering_assumed": spec.no_land_weathering_assumed, "runaway": spec.runaway}
