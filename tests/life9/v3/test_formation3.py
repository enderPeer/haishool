"""formation3: water, C, N and Ar from the solids, CO2 from the carbon-silicate loop, a forward greenhouse,
real-scale relief (PLANET-V3-SPEC section 2). Earth calibration, planet 781, the formulas, the checks."""
from __future__ import annotations

import math
import re
from dataclasses import fields, replace
from pathlib import Path

import pytest
import torch
from scipy.integrate import quad

from haishool.cosmos import planets as level3
from haishool.life9.planet import chain
from haishool.life9.planet import constants as C
from haishool.life9.planet import formation as F
from haishool.life9.planet import globe as gb
from haishool.life9.v3 import formation3 as F3

TAGS = ("chain", "derived", "reference", "new_rule")
EARTH_AIR_N2_PA = C.EARTH_AIR_MOLE_FRACTION["N2"] * C.P_STANDARD
EARTH_AIR_AR_PA = C.EARTH_AIR_MOLE_FRACTION["Ar"] * C.P_STANDARD
#: Earth's surface carbon (sedimentary carbonate and organic carbon, about 1e20 kg), order of magnitude only
EARTH_SURFACE_CARBON_KG = 1.0e20
#: Goldblatt et al. 2009 (Nature Geosci. 2, 891): doubling Earth's N2 warms by about 4.4 K
GOLDBLATT_DOUBLED_N2_K = 4.4


@pytest.fixture(scope="module")
def earth():
    return F3.build_planet3(chain.earth_inputs(), 0)


@pytest.fixture(scope="module")
def inputs781():
    return chain.chain_inputs(781)


@pytest.fixture(scope="module")
def spec781(inputs781):
    return F3.build_planet3(inputs781, 781)


@pytest.fixture(scope="module")
def landless():
    """A synthetic planet whose basins overflow (no land)."""
    for seed in range(20):
        s = F3.build_planet3(chain.synthetic_inputs(seed), seed)
        if s.ocean_fraction >= 1.0:
            return s
    pytest.fail("no landless planet among synthetic seeds 0-19")


# ------------------------------------------------------------------------------------- the solids

def test_disc_temperature_is_world7s_black_body():
    from haishool.cosmos.world import BLACK_BODY
    assert BLACK_BODY == 278.3
    assert F3.disc_temperature_k(1.0, 1.0) == pytest.approx(278.3, abs=1e-12)
    assert F3.disc_temperature_k(16.0, 4.0) == pytest.approx(278.3 * 2 / 2, rel=1e-12)
    assert F3.disc_temperature_k(0.25, 0.25) == pytest.approx(278.3 * 0.25 ** 0.25 / 0.5, rel=1e-12)
    # world7's frost line (2.7 au sqrt L) sits at one disc temperature for every star
    for lum in (0.1, 1.0, 3.0):
        assert F3.disc_temperature_k(lum, level3.FROST_AU * math.sqrt(lum)) == pytest.approx(F3.FROST_T_K, rel=1e-12)
    assert F3.FROST_T_K == pytest.approx(169.368, abs=1e-3)


def test_anchors_are_reproduced_and_interpolated_in_log():
    for m in F3.METEORITE_CLASSES:
        for key in ("h2o", "c", "n"):
            assert F3.anchored_fraction(m.t_k, key) == pytest.approx(getattr(m, key), rel=1e-12)
    oc, cm = F3.METEORITE_CLASSES[1], F3.METEORITE_CLASSES[2]
    mid = 0.5 * (oc.t_k + cm.t_k)
    assert F3.anchored_fraction(mid, "h2o") == pytest.approx(math.sqrt(oc.h2o * cm.h2o), rel=1e-12)
    # constant beyond the outermost anchors
    assert F3.anchored_fraction(1000.0, "h2o") == pytest.approx(F3.METEORITE_CLASSES[0].h2o)
    assert F3.anchored_fraction(20.0, "c") == pytest.approx(F3.METEORITE_CLASSES[-1].c)
    # wetter and richer in C and N toward the cold side across the habitable range (215-286 K)
    ts = [215.0 + 5 * k for k in range(15)]
    for key in ("h2o", "c", "n"):
        vals = [F3.anchored_fraction(t, key) for t in ts]
        assert all(a > b for a, b in zip(vals, vals[1:]))
    # vectorised
    import numpy as np
    arr = F3.anchored_fraction(np.array([150.0, 300.0]), "n")
    assert arr.shape == (2,) and arr[1] == pytest.approx(F3.anchored_fraction(300.0, "n"))


def test_anchor_contents_are_the_cited_meteorite_values():
    by = {m.name: m for m in F3.METEORITE_CLASSES}
    assert set(by) == {"EC", "OC", "CM", "CI"}
    assert by["CI"].c == pytest.approx(0.035) and by["CI"].n == pytest.approx(0.003)      # spec / Lodders 2003
    assert by["EC"].c == pytest.approx(0.004) and by["EC"].n == pytest.approx(0.0004)     # spec / Grady et al. 1986
    assert by["EC"].h2o == pytest.approx(0.001) and by["OC"].h2o == pytest.approx(0.005) and by["CM"].h2o == pytest.approx(0.10)
    # anchor temperatures inside the spec's ranges: EC > 400 K edge, OC in 300-400 K, CM/CI at or below 160 K
    assert by["EC"].t_k >= 400 and 300 <= by["OC"].t_k <= 400 and by["CM"].t_k <= 160 and by["CI"].t_k < by["CM"].t_k
    tag, note = F3.PROVENANCE["meteorite_classes"]
    for cite in ("Lodders 2003", "Grady", "Jarosewich", "Kerridge", "Alexander"):
        assert cite in note
    # the formation-site anchors (a reported sensitivity) keep the contents and move only the temperatures
    sites = {m.name: m for m in F3.METEORITE_CLASSES_SITES}
    for name in by:
        assert (sites[name].h2o, sites[name].c, sites[name].n) == (by[name].h2o, by[name].c, by[name].n)
    assert sites["EC"].t_k == pytest.approx(278.3) and sites["OC"].t_k == pytest.approx(190.0)


def test_ice_joins_beyond_the_frost_line():
    sun = chain.earth_inputs()["cloud"]
    warm = F3.solid_volatiles(F3.FROST_T_K + 1.0, sun)
    cold = F3.solid_volatiles(F3.FROST_T_K - 1.0, sun)
    assert warm["ice"] == 0.0
    assert cold["ice"] == pytest.approx(1.0 - 1.0 / level3.ICE_FACTOR)
    rock = 1.0 - cold["ice"]
    t = F3.FROST_T_K - 1.0
    assert cold["h2o"] == pytest.approx(cold["ice"] + rock * F3.anchored_fraction(t, "h2o"), rel=1e-12)
    assert cold["c"] == pytest.approx(rock * F3.anchored_fraction(t, "c"), rel=1e-12)
    assert cold["h2o"] > 0.6 > warm["h2o"]


def test_cloud_scale_is_relative_to_solar_solids(inputs781):
    sun = chain.earth_inputs()["cloud"]
    for element in ("carbon", "nitrogen", "other"):
        assert F3.cloud_scale(sun, element) == pytest.approx(1.0, rel=1e-12)
    doubled = dict(sun, carbon=2.0 * sun["carbon"])
    assert F3.cloud_scale(doubled, "carbon") == pytest.approx(2.0, rel=1e-12)
    more_rock = dict(sun, iron=2 * sun["iron"], magnesium=2 * sun["magnesium"], silicon=2 * sun["silicon"])
    assert F3.cloud_scale(more_rock, "nitrogen") == pytest.approx(0.5, rel=1e-12)
    assert F3.cloud_scale(more_rock, "other") == pytest.approx(0.5, rel=1e-12)
    s781 = F3.cloud_scale(inputs781["cloud"], "carbon")
    assert 0.5 < s781 < 1.5


def test_degassed_shares_are_the_two_earth_calibrations(earth):
    # water (and carbon): Earth's surface water
    share = F3.degassed_share()
    t_e = F3.disc_temperature_k(1.0, 1.0)
    assert share == pytest.approx(C.EARTH_OCEAN_MASS_FRACTION / F3.anchored_fraction(t_e, "h2o"), rel=1e-12)
    assert 0.001 < share < 0.1
    assert earth.degassed_share == share
    assert earth.water_mass_fraction == pytest.approx(C.EARTH_OCEAN_MASS_FRACTION, rel=1e-12)
    assert earth.ocean_mass_kg + earth.vapour_column_kg_m2 * 4 * math.pi * earth.radius_m ** 2 == \
        pytest.approx(C.EARTH_OCEAN_MASS_FRACTION * earth.planet_mass_kg, rel=1e-12)
    assert earth.surface_carbon_kg == pytest.approx(share * earth.bulk_carbon_kg, rel=1e-12)
    # nitrogen: its own share, so that the model Earth's air holds Earth's N2 (x_N2 0.78084 of 101,325 Pa)
    n_share = F3.nitrogen_degassed_share()
    assert F3.EARTH_N2_PA == pytest.approx(EARTH_AIR_N2_PA, rel=1e-15)
    assert earth.nitrogen_degassed_share == n_share
    assert earth.partial_pressure_pa["N2"] == pytest.approx(F3.EARTH_N2_PA, rel=1e-9)
    assert earth.partial_pressure_pa["N2"] == pytest.approx(78.0e3, rel=0.05)           # the owner's "about 78 kPa"
    assert earth.air_nitrogen_kg == pytest.approx(n_share * earth.bulk_nitrogen_kg, rel=1e-12)
    # the column it needs is that share of the chondritic N: p_N2 = n_N2 g mu_dry
    area = 4 * math.pi * earth.radius_m ** 2
    n_n2 = earth.air_nitrogen_kg / C.molar_mass("N2") / area
    assert n_n2 * earth.gravity_m_s2 * earth.dry_air_molar_mass_kg_mol == pytest.approx(F3.EARTH_N2_PA, rel=1e-9)
    # chondritic N/H2O under the water's share gave 2.3x Earth's N2: the N share is 0.43 of the water's
    assert 0.40 < n_share / share < 0.46
    # Earth's air holds about 3.9e18 kg of N2 (dry mass 5.135e18 kg, Trenberth & Smith 2005, x N2 mass fraction
    # 0.755; quoted from memory): the model Earth's is 1.4 % more (sea-level, not mean surface, pressure)
    assert earth.air_nitrogen_kg == pytest.approx(3.9e18, rel=0.03)
    # the model Earth is the calibration: check_spec3 tests both on Earth's inputs
    assert F3.check_spec3(earth) == []
    off = replace(earth, partial_pressure_pa=dict(earth.partial_pressure_pa, N2=78.0e3))
    assert any(m.startswith("Earth's N2") for m in F3.check_spec3(off))
    off = replace(earth, water_mass_fraction=1.01 * earth.water_mass_fraction)
    assert any(m.startswith("Earth's surface water") for m in F3.check_spec3(off))


# ------------------------------------------------------------------------------------- potassium and heat

def test_potassium_and_heat_scale_per_kg_of_rock(spec781, inputs781, earth):
    """K, U and Th per kg of mantle follow the cloud's 'other' metals per rock, not Z / Z_sun."""
    cloud = inputs781["cloud"]
    lith = F3.cloud_scale(cloud, "other")
    assert spec781.lithophile_cloud_scale == pytest.approx(lith, rel=1e-12)
    z_rel = cloud["metallicity"] / C.Z_SUN
    # 781's cloud is near scaled-solar: rock 0.21 of solar, Z 0.23 of solar, but other metals per rock 0.83
    assert 0.75 < lith < 0.9 and z_rel < 0.3
    sun = chain.earth_inputs()["cloud"]
    rock_rel = F3.rock_mass_fraction(cloud) / F3.rock_mass_fraction(sun)
    assert lith == pytest.approx((cloud["other"] / sun["other"]) / rock_rel, rel=1e-12)
    assert spec781.potassium_kg == pytest.approx(240e-6 * lith * spec781.mantle_mass_kg, rel=1e-9)
    q = F3.radiogenic_heat_w_m2(lith, spec781.star_age_yr, spec781.mantle_mass_kg, spec781.radius_m)
    assert spec781.radiogenic_heat_w_m2 == pytest.approx(q, rel=1e-12)
    q_z = q * z_rel / lith                     # version 2's Z / Z_sun reading
    assert spec781.radiogenic_heat_w_m2 > 3 * q_z
    # Earth's solar cloud: scale 1, 0.047 W/m^2 x decay x size
    assert earth.lithophile_cloud_scale == pytest.approx(1.0, rel=1e-12)
    assert earth.radiogenic_heat_w_m2 == pytest.approx(0.047, rel=0.02)
    assert earth.heat_rel_earth == pytest.approx(1.0, rel=1e-9)
    # the argon follows the potassium
    assert spec781.argon_produced_kg == pytest.approx(F3.argon_produced_kg(spec781.potassium_kg, spec781.star_age_yr),
                                                      rel=1e-12)


# ------------------------------------------------------------------------------------- Earth

def test_earth_calibration(earth):
    assert F3.check_spec3(earth, inputs=chain.earth_inputs()) == []
    assert earth.disc_temperature_k == pytest.approx(278.3)
    # the forward greenhouse returns world7's Earth
    assert earth.t_surface_target_k == pytest.approx(F3.T_EARTH_K, abs=0.3)
    assert abs(earth.chain_t_s_mismatch_k) < 0.3
    assert earth.partial_pressure_pa["CO2"] == pytest.approx(C.EARTH_AIR_MOLE_FRACTION["CO2"] * C.P_STANDARD, rel=0.02)
    assert earth.outgassing_rel_earth == pytest.approx(1.0, rel=1e-9)
    assert earth.land_rel_earth == pytest.approx(1.0, rel=1e-9)
    assert not earth.co2_carbon_limited and not earth.runaway and not earth.co2_beyond_fit
    assert not earth.pressure_beyond_fit and not earth.frozen_mean and not earth.no_land_weathering_assumed
    assert earth.dry_pressure_rel_model_earth == pytest.approx(1.0, rel=1e-9)
    # the seafloor carries its share of Earth's weathering
    assert earth.weathering_land_share == pytest.approx(1.0 - 0.15, rel=1e-4)
    # isostasy: Earth's ocean covers about Earth's 70.9 % (the model Earth is 0.3 % smaller than R_E)
    assert earth.ocean_fraction == pytest.approx(0.709, abs=0.01)
    assert earth.peak_height_m == pytest.approx(8848.86, rel=0.01)
    # argon from K-40 lands near Earth's 946 Pa
    assert earth.partial_pressure_pa["Ar"] == pytest.approx(EARTH_AIR_AR_PA, rel=0.1)
    # chondritic C/H2O gives the model Earth more surface carbon than Earth has (a known finding, 3.2x: carbon
    # keeps the water's share). Nitrogen has its own share: Earth's N2, about 1 bar of air with x_O2 about 0.21.
    # Tight bounds, so drift is caught.
    assert 2.5 * EARTH_SURFACE_CARBON_KG < earth.surface_carbon_kg < 4.0 * EARTH_SURFACE_CARBON_KG
    assert earth.partial_pressure_pa["N2"] == pytest.approx(EARTH_AIR_N2_PA, rel=1e-9)
    assert earth.partial_pressure_pa["O2"] == pytest.approx(C.EARTH_AIR_MOLE_FRACTION["O2"] * C.P_STANDARD)
    assert earth.surface_pressure_pa == pytest.approx(1.0262e5, rel=1e-3)          # vapour 1.26 kPa included
    assert earth.surface_pressure_pa - earth.partial_pressure_pa["H2O"] == pytest.approx(C.P_STANDARD, rel=0.002)
    assert earth.oxygen_mole_fraction == pytest.approx(0.21, abs=0.005)            # 0.2068 with the vapour
    row = F3.summary_row(earth)
    assert row["x_o2_dry"] == pytest.approx(C.EARTH_AIR_MOLE_FRACTION["O2"], rel=2e-3)
    assert row["fire_possible"] and row["x_o2_dry"] >= 0.15                         # fire is physically possible
    assert row["fire_possible"] == (row["x_o2_dry"] >= float(F3.cr.X_O2_MIN))
    # the model Earth is what build_planet3 returns for Earth's inputs
    me = F3.model_earth()
    assert me.t_surface_target_k == pytest.approx(earth.t_surface_target_k, abs=1e-6)
    assert me.surface_pressure_pa == pytest.approx(earth.surface_pressure_pa, rel=1e-9)
    assert me.partial_pressure_pa["N2"] == pytest.approx(EARTH_AIR_N2_PA, rel=1e-12)
    text = F3.describe3(earth)
    assert "1.03 bar" in text.splitlines()[1] and "Earth's N2 79.1 kPa" in text.splitlines()[1]


def test_the_earlier_single_share_is_a_reading(earth, spec781):
    """nitrogen_calibration="water" is the earlier rule (one share for water, C and N): the model Earth gets 2.3x
    Earth's N2 and 2.05 bar, P_cal 2.04 bar, and 781 its earlier 1.20 bar and 270.77 K. Under either rule the model
    Earth keeps world7's T_s, water and carbon."""
    rules = F3.Formation3Rules(nitrogen_calibration="water")
    old = F3.model_earth(rules)
    assert F3.check_spec3(old, rules) == []
    assert old.nitrogen_degassed_share == old.degassed_share == earth.degassed_share
    assert F3.nitrogen_degassed_share(rules) == F3.degassed_share()
    assert old.partial_pressure_pa["N2"] == pytest.approx(2.2984 * EARTH_AIR_N2_PA, rel=1e-4)
    assert old.surface_pressure_pa == pytest.approx(2.0533e5, rel=1e-4)
    assert F3.greenhouse(rules).p_cal_pa == pytest.approx(2.0407e5, rel=1e-4)
    assert old.surface_carbon_kg == pytest.approx(earth.surface_carbon_kg, rel=1e-12)
    assert old.water_mass_fraction == pytest.approx(earth.water_mass_fraction, rel=1e-12)
    assert old.t_surface_target_k == pytest.approx(F3.T_EARTH_K, abs=0.3)
    assert old.oxygen_mole_fraction < 0.15 < earth.oxygen_mole_fraction             # no fire in the earlier air
    s = F3.build_planet3(chain.chain_inputs(781), 781, rules)
    assert F3.check_spec3(s, rules) == []
    assert s.surface_pressure_pa == pytest.approx(1.2013e5, rel=1e-4)
    assert s.partial_pressure_pa["N2"] == pytest.approx(116.64e3, rel=1e-4)
    assert s.t_surface_target_k == pytest.approx(270.769, abs=0.01)
    # the separate share takes every planet's N2 down by about the same factor (0.43-0.46: mu_dry moves)
    ratio = spec781.partial_pressure_pa["N2"] / s.partial_pressure_pa["N2"]
    assert 0.42 < ratio < 0.47
    assert spec781.air_nitrogen_kg / s.air_nitrogen_kg == pytest.approx(
        F3.nitrogen_degassed_share() / F3.degassed_share(), rel=1e-6)
    # water, carbon and the ocean do not move; the planet's dry air relative to its model Earth falls a little
    assert spec781.surface_water_kg == pytest.approx(s.surface_water_kg, rel=1e-12)
    assert spec781.surface_carbon_kg == pytest.approx(s.surface_carbon_kg, rel=1e-12)
    assert spec781.ocean_fraction == pytest.approx(s.ocean_fraction, abs=1e-4)
    assert spec781.dry_pressure_rel_model_earth < s.dry_pressure_rel_model_earth
    with pytest.raises(ValueError):
        F3.build_planet3(chain.earth_inputs(), 0, F3.Formation3Rules(nitrogen_calibration="guess"))


def test_greenhouse_is_calibrated_on_world7s_earth():
    rules = F3.Formation3Rules()
    gh = F3.greenhouse(rules)
    assert gh.tau_earth == pytest.approx(F.earth_tau(), rel=1e-12)
    assert gh.tau_cloud + gh.tau_other_earth + gh.tau_c_earth + gh.tau_w_earth == pytest.approx(gh.tau_earth, rel=1e-12)
    assert gh.tau_w_earth == pytest.approx(0.5 * gh.tau_earth) and gh.tau_c_earth == pytest.approx(0.2 * gh.tau_earth)
    assert gh.tau_cloud == pytest.approx(0.25 * gh.tau_earth) and gh.tau_other_earth == pytest.approx(0.05 * gh.tau_earth)
    # the broadening reference is the model Earth's dry pressure: with Earth's N2, about Earth's 1.013 bar
    me = F3.model_earth(rules)
    assert gh.p_cal_pa == pytest.approx(me.surface_pressure_pa - me.partial_pressure_pa["H2O"], rel=1e-9)
    assert gh.p_cal_pa == pytest.approx(C.P_STANDARD, rel=0.002)
    t_eq = F.equilibrium_temperature_k(C.S0, C.EARTH_ALBEDO)

    def solve(u_c, p_rel=1.0):
        tau_c = float(gh.tau_co2(u_c, p_rel))
        return F3.surface_temperature(
            t_eq, lambda t: gh.tau_rest + tau_c + gh.tau_h2o(F3.vapour_column(t, C.G_STANDARD, rules), p_rel), rules)

    t0, run = solve(gh.u_c_earth)
    assert not run and t0 == pytest.approx(F3.T_EARTH_K, abs=1e-6)
    # CO2 doubling: Myhre's 3.71 W/m^2 over Planck less the water vapour + lapse rate feedback
    t2, _ = solve(2 * gh.u_c_earth)
    olr = C.SIGMA * t0 ** 4
    planck = 4 * olr / t0 / (1 + 0.75 * gh.tau_earth)
    expected = 5.35 * math.log(2) / (planck * (1 - rules.wv_lr_feedback / rules.planck_feedback))
    assert t2 - t0 == pytest.approx(expected, rel=0.05)
    assert 1.5 < t2 - t0 < 2.5
    # no CO2 at all: finite (about -20 W/m^2 of forcing), colder
    tn, _ = solve(0.0)
    assert t0 - 15 < tn < t0 - 3
    # the vapour term vanishes with the vapour
    assert float(gh.tau_h2o(F3.vapour_column(180.0, C.G_STANDARD, rules))) < 0.02 * gh.tau_w_earth
    # pressure broadening: doubling the dry air warms about as Goldblatt et al. 2009 found for doubled N2
    tp, _ = solve(gh.u_c_earth, 2.0)
    assert 0.6 * GOLDBLATT_DOUBLED_N2_K < tp - t0 < 1.8 * GOLDBLATT_DOUBLED_N2_K
    assert solve(gh.u_c_earth, 0.5)[0] < t0
    # b = 0 removes it
    flat = F3.greenhouse(F3.Formation3Rules(broadening_exponent=0.0))
    assert float(flat.tau_h2o(gh.u_w_earth, 5.0)) == pytest.approx(flat.tau_w_earth, rel=1e-12)


def test_other_gases_scale_with_the_chains_oxygen():
    gh = F3.greenhouse()
    assert gh.tau_other(1.0, "oxygen_pal") == pytest.approx(gh.tau_other_earth, rel=1e-12)
    assert gh.tau_other(0.118, "oxygen_pal") == pytest.approx(0.118 * gh.tau_other_earth, rel=1e-12)
    assert gh.tau_other(0.118, "earth") == gh.tau_other_earth and gh.tau_other(0.118, "none") == 0.0
    with pytest.raises(ValueError):
        F3.build_planet3(chain.earth_inputs(), 0, F3.Formation3Rules(other_gases="guess"))


def test_surface_temperature_flags_runaway():
    rules = F3.Formation3Rules()
    t, run = F3.surface_temperature(250.0, lambda t: 1e3 + 0 * t, rules)
    assert run and t == rules.t_max_k
    t, run = F3.surface_temperature(250.0, lambda t: 0.0 * t, rules)
    assert not run and t == 250.0


def test_continental_whak_law():
    r = F3.Formation3Rules()
    p_e = C.EARTH_AIR_MOLE_FRACTION["CO2"] * C.P_STANDARD
    assert F3.whak_co2_pa(F3.T_EARTH_K, 1.0, 1.0, r) == pytest.approx(p_e, rel=1e-12)
    # weathering (p/p_E)^0.3 exp(dT/13.7) = outgassing
    for v, dt in ((2.0, 0.0), (0.5, 10.0), (3.0, -20.0)):
        p = F3.whak_co2_pa(F3.T_EARTH_K + dt, v, 1.0, r)
        assert (p / p_e) ** 0.3 * math.exp(dt / 13.7) == pytest.approx(v, rel=1e-12)
    # land divides, never silently ignored
    assert F3.whak_co2_pa(F3.T_EARTH_K, 1.0, 0.5, r) == pytest.approx(p_e * 2 ** (1 / 0.3), rel=1e-12)
    assert F3.whak_co2_pa(F3.T_EARTH_K, 1.0, 0.0, r) == math.inf
    assert F3.whak_co2_pa(F3.T_EARTH_K, 0.0, 1.0, r) == 0.0


def test_land_and_seafloor_weathering():
    r = F3.Formation3Rules()
    p_e = C.EARTH_AIR_MOLE_FRACTION["CO2"] * C.P_STANDARD
    s = r.seafloor_share
    # the model Earth: both terms at their Earth values, the total is 1 at p_E and T_E
    land, sea = F3.weathering_terms(p_e, F3.T_EARTH_K, 1.0, 1.0, r)
    assert land == pytest.approx(1 - s, rel=1e-12) and sea == pytest.approx(s, rel=1e-12)
    assert F3.steady_co2_pa(F3.T_EARTH_K, 1.0, 1.0, 1.0, r) == pytest.approx(p_e, rel=1e-12)
    # the steady state balances outgassing for any mix of land, heat and temperature
    for v, land_rel, heat_rel, dt in ((2.0, 1.0, 1.0, 0.0), (0.5, 1.7, 0.4, -15.0), (4.0, 0.2, 3.0, 12.0),
                                      (1.0, 0.0, 2.0, 5.0), (3.0, 1e-6, 1.0, -30.0)):
        p = F3.steady_co2_pa(F3.T_EARTH_K + dt, v, land_rel, heat_rel, r)
        assert sum(F3.weathering_terms(p, F3.T_EARTH_K + dt, land_rel, heat_rel, r)) == pytest.approx(v, rel=1e-10)
    # without land only the seafloor weathers: closed form with its own laws
    t = F3.T_EARTH_K + 10.0
    p = F3.steady_co2_pa(t, 1.0, 0.0, 1.0, r)
    arr = math.exp(-r.seafloor_e_j_mol / C.R_GAS * (1 / t - 1 / F3.T_EARTH_K))
    assert p == pytest.approx(p_e * (1.0 / (s * arr)) ** (1 / r.seafloor_beta), rel=1e-12)
    # the seafloor's temperature dependence is weaker than the continents' (e-folding 16.8 K against 13.7 K)
    efold = C.R_GAS * F3.T_EARTH_K ** 2 / r.seafloor_e_j_mol
    assert 15.0 < efold < 18.5 and efold > r.whak_t_e_k
    # the global reading is the published law with no land factor; 'land' drops the seafloor
    assert F3.steady_co2_pa(t, 1.0, 0.0, 1.0, r, "global") == pytest.approx(F3.whak_co2_pa(t, 1.0, 1.0, r), rel=1e-12)
    assert F3.steady_co2_pa(t, 1.0, 0.0, 1.0, r, "land") == math.inf
    assert F3.steady_co2_pa(t, 0.0, 1.0, 1.0, r) == 0.0
    # monotone: warmer weathers faster, so less CO2
    ps = [F3.steady_co2_pa(F3.T_EARTH_K + d, 1.0, 0.5, 1.0, r) for d in (-20, -10, 0, 10, 20)]
    assert all(a > b for a, b in zip(ps, ps[1:]))


def test_argon_matches_the_decay_integral():
    lam = math.log(2) / C.HALF_LIFE_YR["K40"]
    k = 1.0e20
    n0 = k / C.element_mass("K") * F3.K40_ATOM_FRACTION * math.exp(lam * C.EARTH_AGE_YR)
    for age in (1e9, C.EARTH_AGE_YR, 9e9):
        made, _ = quad(lambda t: F3.K40_AR_BRANCH * lam * n0 * math.exp(-lam * t), 0.0, age)
        assert F3.argon_produced_kg(k, age) == pytest.approx(made * C.element_mass("Ar"), rel=1e-9)
    assert F3.K40_AR_BRANCH == pytest.approx(0.1048, abs=1e-4)


def test_dry_air_is_well_mixed():
    g = 9.8
    n, mu = F3.dry_air({"N2": 300.0, "Ar": 3.0}, {"O2": 2.0e4, "CO2": 50.0}, {"CO2": math.inf}, g)
    total = sum(n.values())
    assert mu == pytest.approx(sum(n[k] * C.molar_mass(k) for k in n) / total, rel=1e-12)
    assert n["O2"] * g * mu == pytest.approx(2.0e4, rel=1e-12) and n["CO2"] * g * mu == pytest.approx(50.0, rel=1e-12)
    capped, _ = F3.dry_air({"N2": 300.0}, {"CO2": 1e9}, {"CO2": 7.0}, g)
    assert capped["CO2"] == 7.0
    # a loop that cannot converge raises instead of ending silently
    with pytest.raises(RuntimeError):
        F3.dry_air({"N2": math.nan}, {"CO2": 50.0}, {}, g)


# ------------------------------------------------------------------------------------- 781

def test_781_spec(spec781, inputs781):
    s = spec781
    assert F3.check_spec3(s, inputs=inputs781) == []
    assert s.disc_temperature_k == pytest.approx(278.3 * inputs781["star"]["luminosity_lsun"] ** 0.25
                                                 / math.sqrt(inputs781["planet"]["orbit_au"]), rel=1e-12)
    assert s.disc_temperature_k == pytest.approx(269.86, abs=0.01)
    assert s.radius_m == pytest.approx(5.674e6, rel=0.01)
    # cooler disc than Earth's: wetter solids, more water per unit mass
    assert s.solid_water_fraction > F3.anchored_fraction(278.3, "h2o")
    assert s.water_mass_fraction > C.EARTH_OCEAN_MASS_FRACTION
    # outcomes are reported, not gated: only their consistency is asserted
    assert 0.0 <= s.ocean_fraction <= 1.0
    assert s.flood_depth_m == pytest.approx(max(0.0, s.ocean_layer_m - s.basin_depth_m), abs=1e-9)
    assert s.runaway == (s.t_surface_target_k >= F3.Formation3Rules().t_max_k)
    assert s.frozen_mean == (s.t_surface_target_k < 273.15)
    assert s.chain_t_target_k == pytest.approx(inputs781["planet"]["t_eq_k"] + inputs781["planet"]["greenhouse_k"])
    assert s.chain_t_s_mismatch_k == pytest.approx(s.t_surface_target_k - s.chain_t_target_k, abs=1e-12)
    # the same atmosphere bookkeeping as version 2: O2 from the bodies era
    assert s.oxygen_pal == pytest.approx(inputs781["biosphere"]["oxygen_pal"])
    assert tuple(s.partial_pressure_pa) == F.GASES
    # the other gases follow 781's thin oxygen; the imposed parts are reported as warming
    gh = F3.greenhouse()
    assert s.tau_other_gases == pytest.approx(s.oxygen_pal * gh.tau_other_earth, rel=1e-12)
    assert s.tau_cloud == pytest.approx(gh.tau_cloud, rel=1e-12)
    row = F3.summary_row(s)
    assert row["cloud_warming_k"] > 0 and row["other_gases_warming_k"] >= 0
    assert row["frozen_mean"] == s.frozen_mean and row["pressure_beyond_fit"] == s.pressure_beyond_fit
    # pinned (nitrogen's own degassed share; the single share gave 1.20 bar, 116.6 kPa of N2 and 270.77 K)
    assert s.nitrogen_degassed_share == pytest.approx(0.0063547, rel=1e-4)
    assert s.partial_pressure_pa["N2"] == pytest.approx(50.223e3, rel=1e-4)
    assert s.partial_pressure_pa["O2"] == pytest.approx(2504.6, rel=1e-4)
    assert s.partial_pressure_pa["CO2"] == pytest.approx(12.771, rel=1e-3)
    assert s.partial_pressure_pa["Ar"] == pytest.approx(579.0, rel=1e-3)
    assert s.surface_pressure_pa == pytest.approx(53.707e3, rel=1e-4)
    assert s.oxygen_mole_fraction == pytest.approx(0.04663, abs=1e-5)
    assert s.t_surface_target_k == pytest.approx(270.499, abs=0.01)
    assert s.frozen_mean and not s.runaway and not s.pressure_beyond_fit
    assert s.ocean_fraction == pytest.approx(0.51285, abs=1e-4)
    assert s.dry_pressure_rel_model_earth == pytest.approx(0.52606, rel=1e-4)
    assert row["x_o2_dry"] == pytest.approx(0.04697, abs=1e-5) and not row["fire_possible"]   # O2 0.118 PAL
    print(f"\n781: q {s.radiogenic_heat_w_m2 * 1e3:.1f} mW/m^2, V_rel {s.outgassing_rel_earth:.3f}, P "
          f"{s.surface_pressure_pa / 1e5:.3f} bar (N2 {s.partial_pressure_pa['N2'] / 1e3:.1f} kPa, x_O2 "
          f"{s.oxygen_mole_fraction:.4f}), p_CO2 "
          f"{s.partial_pressure_pa['CO2']:.3g} Pa, p_Ar {s.partial_pressure_pa['Ar']:.0f} Pa, ocean {s.ocean_fraction:.3f} "
          f"(fixed D {s.ocean_fraction_fixed_depth:.3f}, D~1/g {s.ocean_fraction_inverse_g_depth:.3f}), T_s "
          f"{s.t_surface_target_k:.1f} K (chain {s.chain_t_target_k:.1f}), frozen_mean {s.frozen_mean}; warming: clouds "
          f"{row['cloud_warming_k']:.1f} K, other gases {row['other_gases_warming_k']:.2f} K, broadening "
          f"{row['broadening_warming_k']:+.1f} K; global-WHAK reading {s.t_surface_global_weathering_k:.1f} K")


def test_budgets_close(earth, spec781):
    for s in (earth, spec781):
        area = 4 * math.pi * s.radius_m ** 2
        assert s.surface_water_kg + s.undegassed_water_kg == pytest.approx(s.bulk_water_kg, rel=1e-12)
        assert s.ocean_mass_kg + s.vapour_column_kg_m2 * area == pytest.approx(s.surface_water_kg, rel=1e-9)
        assert s.air_carbon_kg + s.carbonate_carbon_kg + s.undegassed_carbon_kg == pytest.approx(s.bulk_carbon_kg, rel=1e-12)
        assert s.air_nitrogen_kg + s.undegassed_nitrogen_kg == pytest.approx(s.bulk_nitrogen_kg, rel=1e-12)
        assert s.air_nitrogen_kg == pytest.approx(s.nitrogen_degassed_share * s.bulk_nitrogen_kg, rel=1e-12)
        assert s.surface_water_kg == pytest.approx(s.degassed_share * s.bulk_water_kg, rel=1e-12)
        assert s.surface_carbon_kg == pytest.approx(s.degassed_share * s.bulk_carbon_kg, rel=1e-12)
        # the air's N2 column is all the degassed nitrogen: p_N2 = n_N2 g mu_dry
        n_n2 = s.air_nitrogen_kg / C.molar_mass("N2") / area
        assert n_n2 * s.gravity_m_s2 * s.dry_air_molar_mass_kg_mol == pytest.approx(s.partial_pressure_pa["N2"], rel=1e-9)
        assert s.air_argon_kg + s.retained_argon_kg == pytest.approx(s.argon_produced_kg, rel=1e-12)
        layers = s.core_mass_kg + s.mantle_mass_kg + s.ocean_mass_kg + s.atmosphere_mass_kg
        assert layers == pytest.approx(s.planet_mass_kg, rel=1e-12)
        # the dry column's weight is the dry pressure
        p_dry = s.surface_pressure_pa - s.partial_pressure_pa["H2O"]
        assert s.atmosphere_mass_kg - s.vapour_column_kg_m2 * area == pytest.approx(p_dry * area / s.gravity_m_s2, rel=1e-9)
        assert s.fixed_point_converged


def test_the_carbon_silicate_fixed_point(spec781, earth):
    r = F3.Formation3Rules()
    for s in (spec781, earth):
        assert s.weathering_t_k == pytest.approx(s.t_surface_target_k, abs=1e-6)
        assert not s.weathering_at_chain_t
        assert s.partial_pressure_pa["CO2"] == pytest.approx(
            F3.steady_co2_pa(s.weathering_t_k, s.outgassing_rel_earth, s.land_rel_earth, s.heat_rel_earth, r), rel=1e-9)
        assert sum(F3.weathering_terms(s.partial_pressure_pa["CO2"], s.weathering_t_k, s.land_rel_earth,
                                       s.heat_rel_earth, r)) == pytest.approx(s.outgassing_rel_earth, rel=1e-9)
        assert s.t_surface_selfconsistent_k == s.t_surface_target_k
        assert s.co2_selfconsistent_pa == pytest.approx(s.co2_ref_pa, rel=1e-12)
    # the spec's literal reading (weathering at the chain's T) is reported, and is what the "chain" rule builds
    rules = F3.Formation3Rules(weathering_t="chain")
    chain_rule = F3.build_planet3(chain.chain_inputs(781), 781, rules)
    assert F3.check_spec3(chain_rule, rules) == [] and chain_rule.weathering_at_chain_t
    assert chain_rule.weathering_t_k == pytest.approx(spec781.chain_t_target_k)
    assert chain_rule.t_surface_target_k == pytest.approx(spec781.t_surface_chain_weathering_k, abs=1e-4)
    assert chain_rule.co2_ref_pa == pytest.approx(spec781.co2_chain_weathering_pa, rel=1e-4)
    assert chain_rule.t_surface_selfconsistent_k == pytest.approx(spec781.t_surface_target_k, abs=1e-4)
    # the global WHAK reading is what the 'global' rule builds
    glob = F3.build_planet3(chain.chain_inputs(781), 781, F3.Formation3Rules(weathering="global"))
    assert glob.t_surface_target_k == pytest.approx(spec781.t_surface_global_weathering_k, abs=1e-4)
    assert glob.weathering_land_share == 1.0 and not glob.weathering_land_scaled
    with pytest.raises(ValueError):
        F3.build_planet3(chain.chain_inputs(781), 781, F3.Formation3Rules(weathering_t="guess"))
    with pytest.raises(ValueError):
        F3.build_planet3(chain.chain_inputs(781), 781, F3.Formation3Rules(weathering="guess"))


def test_a_planet_without_land_weathers_on_its_seafloor(landless):
    s = landless
    r = F3.Formation3Rules()
    assert s.ocean_fraction == 1.0 and s.land_rel_earth == 0.0
    assert s.weathering_land_share == 0.0 and not s.no_land_weathering_assumed
    assert s.seafloor_weathering_share == r.seafloor_share
    if not s.co2_carbon_limited:
        _, sea = F3.weathering_terms(s.partial_pressure_pa["CO2"], s.weathering_t_k, 0.0, s.heat_rel_earth, r)
        assert sea == pytest.approx(s.outgassing_rel_earth, rel=1e-9)
    # the published global law would hand it a continental thermostat: that reading is flagged
    rules = F3.Formation3Rules(weathering="global")
    g = F3.build_planet3(chain.synthetic_inputs(s.seed), s.seed, rules)
    assert g.no_land_weathering_assumed and F3.check_spec3(g, rules) == []
    assert g.t_surface_target_k == pytest.approx(s.t_surface_global_weathering_k, abs=1e-4)
    assert F3.summary_row(g)["no_land_weathering_assumed"]


def test_land_only_weathering_is_a_reported_bound(landless):
    rules = F3.Formation3Rules(weathering="land")
    s = F3.build_planet3(chain.synthetic_inputs(landless.seed), landless.seed, rules)
    assert s.ocean_fraction == 1.0 and s.land_rel_earth == 0.0
    assert s.co2_carbon_limited
    assert s.air_carbon_kg == pytest.approx(s.surface_carbon_kg, rel=1e-12)
    assert s.carbonate_carbon_kg <= 1e-12 * s.surface_carbon_kg
    assert F3.check_spec3(s, rules) == []


def test_relief_is_real_scale(earth, spec781):
    d = F3.basin_depth_m()
    assert d == pytest.approx(F3.EARTH_MEAN_OCEAN_DEPTH_M, rel=0.04)
    share = F3.ridge_share()
    assert share == pytest.approx(2600.0 / 3682.0, rel=1e-12)
    assert F3.ridge_share(rule="fixed") == 0.0 and F3.ridge_share(rule="inverse_g") == 1.0
    for s in (earth, spec781):
        g = s.gravity_m_s2
        assert s.peak_height_m * g == pytest.approx(8848.86 * F3.G_EARTH_SURFACE, rel=1e-12)
        assert s.basin_depth_m == pytest.approx(d * (share * F3.G_EARTH_SURFACE / g + 1 - share), rel=1e-12)
        assert s.relief_m == pytest.approx(s.peak_height_m + s.basin_depth_m, rel=1e-12)
        area = 4 * math.pi * s.radius_m ** 2
        assert s.ocean_fraction == pytest.approx(min(1.0, s.ocean_volume_m3 / (area * s.basin_depth_m)), rel=1e-12)
        assert s.ocean_fraction_fixed_depth == pytest.approx(min(1.0, s.ocean_volume_m3 / (area * d)), rel=1e-12)
        assert s.ocean_fraction_inverse_g_depth == pytest.approx(
            min(1.0, s.ocean_volume_m3 / (area * d * F3.G_EARTH_SURFACE / g)), rel=1e-12)
        assert s.mean_land_height_m * g == pytest.approx(840.0 * F3.G_EARTH_SURFACE, rel=1e-12)
    assert spec781.peak_height_m > earth.peak_height_m       # lower g, taller peaks and deeper basins
    assert spec781.basin_depth_m > earth.basin_depth_m
    # the split lies between the bounds
    lo, hi = sorted((spec781.ocean_fraction_fixed_depth, spec781.ocean_fraction_inverse_g_depth))
    assert lo <= spec781.ocean_fraction <= hi
    # basins overflow: everything is sea, the platform floods
    assert F3.ocean_fraction(2.0 * 4 * math.pi * 1e12 * d, 1e6, d) == 1.0
    # the rule is the spec's
    fixed = F3.build_planet3(chain.chain_inputs(781), 781, F3.Formation3Rules(basin_depth_rule="fixed"))
    assert fixed.ocean_fraction == pytest.approx(spec781.ocean_fraction_fixed_depth, rel=1e-6)
    assert fixed.basin_depth_m == pytest.approx(d, rel=1e-12)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_make_terrain3_matches_the_specs_ocean(seed, earth, spec781, landless):
    """At G = 32 the globe's ocean covers the spec's ocean_fraction and holds its volume."""
    specs = [earth, spec781, F3.build_planet3(chain.synthetic_inputs(4), 4), landless]
    globe = gb.Globe(32)
    t = F3.make_terrain3(globe, torch.Generator().manual_seed(seed), specs)
    assert set(t) == {"elevation_m", "sea_level_m", "depth_m", "land", "coast", "inland_rad", "ocean_fraction",
                      "ocean_volume_m3"}
    area = globe.area64
    for i, s in enumerate(specs):
        assert float(t["ocean_fraction"][i]) == pytest.approx(s.ocean_fraction, abs=2e-3), s.seed
        assert float(t["ocean_volume_m3"][i]) == pytest.approx(s.ocean_volume_m3, rel=1e-6)
        h = (t["elevation_m"][i] - t["sea_level_m"][i]).double()
        land = t["land"][i]
        if land.any():
            mean_land = float((h * area * land).sum() / (area * land).sum())
            assert mean_land == pytest.approx(s.mean_land_height_m, rel=0.05)
            assert float(h.max()) <= s.peak_height_m
        ocean = ~land
        mean_depth = float((-h * area * ocean).sum() / (area * ocean).sum())
        target = s.basin_depth_m if s.ocean_fraction < 1.0 else s.ocean_layer_m
        assert mean_depth == pytest.approx(target, rel=0.01)
    # deterministic
    again = F3.make_terrain3(globe, torch.Generator().manual_seed(seed), specs)
    assert torch.equal(again["elevation_m"], t["elevation_m"]) and torch.equal(again["sea_level_m"], t["sea_level_m"])


def test_check_spec3_catches_broken_budgets(spec781, inputs781):
    assert F3.check_spec3(spec781) == []
    broken = {
        "air_carbon_kg": spec781.air_carbon_kg * 1.01,
        "undegassed_water_kg": spec781.undegassed_water_kg * 1.01,
        "air_nitrogen_kg": spec781.air_nitrogen_kg * 1.01,
        "air_argon_kg": spec781.air_argon_kg * 1.01,
        "disc_temperature_k": spec781.disc_temperature_k + 1.0,
        "degassed_share": spec781.degassed_share * 1.01,
        "nitrogen_degassed_share": spec781.nitrogen_degassed_share * 1.01,
        "undegassed_nitrogen_kg": spec781.undegassed_nitrogen_kg * 1.01,
        "tau_h2o": spec781.tau_h2o * 1.01,
        "chain_t_s_mismatch_k": spec781.chain_t_s_mismatch_k + 1.0,
        "co2_ref_pa": spec781.co2_ref_pa * 2.0,
        "relief_m": spec781.relief_m + 10.0,
        "ocean_fraction": spec781.ocean_fraction * 0.9,
        # the inputs that drive the CO2
        "radiogenic_heat_w_m2": spec781.radiogenic_heat_w_m2 * 3.0,
        "land_rel_earth": 0.123,
        "potassium_kg": spec781.potassium_kg * 1.01,
        "lithophile_cloud_scale": spec781.lithophile_cloud_scale * 1.01,
        "heat_rel_earth": spec781.heat_rel_earth * 1.01,
        "outgassing_rel_earth": spec781.outgassing_rel_earth * 1.01,
        "weathering_t_k": spec781.chain_t_target_k,
        "weathering_land_share": spec781.weathering_land_share * 0.9,
        "seafloor_weathering_share": 0.3,
        # flags, readings and the greenhouse split
        "frozen_mean": not spec781.frozen_mean,
        "pressure_beyond_fit": not spec781.pressure_beyond_fit,
        "co2_beyond_fit": not spec781.co2_beyond_fit,
        "no_land_weathering_assumed": not spec781.no_land_weathering_assumed,
        "fixed_point_converged": False,
        "tau_other_gases": spec781.tau_other_gases * 1.5,
        "tau_cloud": spec781.tau_cloud * 1.01,
        "dry_pressure_rel_model_earth": spec781.dry_pressure_rel_model_earth * 1.01,
        "t_surface_no_clouds_k": spec781.t_surface_target_k + 1.0,
        # relief
        "basin_depth_m": spec781.basin_depth_m * 1.01,
        "basin_depth_ridge_share": 1.0,
        "ocean_fraction_fixed_depth": spec781.ocean_fraction_fixed_depth * 0.9,
        "ocean_fraction_inverse_g_depth": spec781.ocean_fraction_inverse_g_depth * 0.9,
        "mean_land_height_m": spec781.mean_land_height_m * 1.01,
        "flood_depth_m": 100.0,
    }
    for name, value in broken.items():
        assert F3.check_spec3(replace(spec781, **{name: value})), name
    # a spec built with weathering at the chain's T must say so
    assert F3.check_spec3(replace(spec781, weathering_at_chain_t=True))
    # with the inputs, the cloud scales are recomputed
    assert F3.check_spec3(spec781, inputs=inputs781) == []
    assert F3.check_spec3(spec781, inputs=chain.earth_inputs())
    # version 2's backwards T_s gate is superseded; the signed mismatch is a finding
    assert any("from the chain's" in m for m in F.check_spec(spec781))
    assert set(F3.SUPERSEDED)


def test_spec_keeps_version2_fields_and_works_with_its_consumers(spec781, earth):
    v2 = {f.name: f.type for f in fields(F.PlanetSpec)}
    v3 = {f.name: f.type for f in fields(F3.PlanetSpec3)}
    assert set(v2) <= set(v3)
    assert isinstance(spec781, F.PlanetSpec)
    stacked = F.stack_specs([earth, spec781])
    assert stacked["radius_m"].shape == (2,) and stacked["crust"].shape[0] == 2
    assert all(bool(torch.isfinite(t).all()) for k, t in stacked.items() if k != "day_length_s")
    d = spec781.to_dict()
    assert d["chain_t_s_mismatch_k"] == spec781.chain_t_s_mismatch_k and "provenance" in d
    assert F3.describe3(spec781).startswith(f"PlanetSpec seed 781 (world7), {F3.FORMATION3_VERSION}")
    # version 2's climate takes the spec and the v3 terrain, whose land is the spec's
    from haishool.life9.planet import climate
    g = gb.Globe(8)
    terrain = F3.make_terrain3(g, torch.Generator().manual_seed(0), [spec781])
    assert float(terrain["ocean_fraction"][0]) == pytest.approx(spec781.ocean_fraction, abs=0.01)
    params = climate.make_params([spec781], g, terrain, radius_m=spec781.radius_m)
    assert float(params["t_target"][0]) == pytest.approx(spec781.t_surface_target_k, rel=1e-6)
    assert float(params["rh"][0]) == pytest.approx(0.77, rel=1e-5)
    assert all(bool(torch.isfinite(params[k]).all()) for k in ("tau_rest", "tau_w_k", "tau_per_ln_co2", "co2_ref"))
    # land heights are real scale: mean land height above the sea, not a linear spread of relief_m
    land = terrain["land"][0]
    height = params["height_m"][0]
    mean_land = float((height * g.area * land).sum() / (g.area * land).sum())
    assert mean_land == pytest.approx(spec781.mean_land_height_m, rel=0.15)
    assert mean_land < 0.2 * spec781.relief_m


def test_provenance(spec781):
    for f in fields(spec781):
        if f.name == "provenance":
            continue
        tag, note = spec781.provenance[f.name]
        assert note and all(part in TAGS for part in tag.split("+")), f.name
    for name, (tag, note) in F3.PROVENANCE.items():
        assert note and all(part in TAGS for part in tag.split("+")), name
    # every modelling choice has its own entry
    missing = {f.name for f in fields(F3.Formation3Rules)} - set(F3.PROVENANCE)
    assert not missing
    # the sources the review asked for are stated
    assert "Z / Z_sun" in F3.PROVENANCE["lithophile_scale"][1]
    assert "1/g" in F3.PROVENANCE["basin_depth"][1] and "inverse_g" in F3.PROVENANCE["basin_depth"][1]
    assert "ice-albedo" in F3.PROVENANCE["frozen"][1]
    assert "collision-induced" in F3.PROVENANCE["absorber_column"][1]
    # the nitrogen calibration is stated, with its target and the earlier reading
    assert "Earth's N2" in F3.PROVENANCE["nitrogen_degassed_share"][1]
    assert "0.78084" in F3.PROVENANCE["earth_n2_pa"][1] and "101,325" in F3.PROVENANCE["earth_n2_pa"][1]
    assert "'water'" in F3.PROVENANCE["nitrogen_calibration"][1]
    assert "nitrogen_degassed_share" in spec781.provenance["air_nitrogen_kg"][1]


def test_determinism_and_seed(inputs781, spec781):
    again = F3.build_planet3(inputs781, 781)
    assert again.to_dict() == spec781.to_dict()
    other = F3.build_planet3(inputs781, 782)
    same = {f.name for f in fields(spec781)} - {"seed", "provenance", "rotation_period_s", "obliquity_rad",
                                                "day_length_s"}
    for name in same:
        assert getattr(other, name) == getattr(spec781, name), name


def test_no_forbidden_mechanisms_in_the_module():
    text = Path(F3.__file__).read_text(encoding="utf-8")
    for word in ("innate", "forage", "background_drink", "time_budget", "pedigree", "relatedness", "seed_floor",
                 "LIFESPAN", "MATURITY", "reward", "ACTIONS"):
        assert not re.search(word, text, flags=re.IGNORECASE if word.islower() else 0), word


def test_black_body_equals_world7s():
    pytest.importorskip("scipy")
    from haishool.cosmos.world import BLACK_BODY as world7_black_body
    from haishool.life9.v3 import formation3
    assert formation3.BLACK_BODY == world7_black_body
