"""Tests of the planet engine's constants, chain inputs and formation (PLANET-SPEC 2.1-2.3, 3, 4)."""
from __future__ import annotations

import json
import math
from dataclasses import fields, replace

import pytest
import scipy.constants as sc
import torch
from scipy.integrate import quad

from haishool.cosmos import planets as level3
from haishool.life9.planet import chain, constants as C, formation as F, materials

DAY = 86400.0
#: world7 seeds whose unrounded T_s lies closest to a whole-K half among seeds 12-59 (seed 34
#: failed the 0.5 K gate when t_eq came from the unrounded level flux; with the era flux 34, 19,
#: 26, 35 and 39 pass within 0.07 K of it), plus 44 (CO2 not capped, locked)
WORLD7_SEEDS = (19, 26, 34, 35, 39, 44)


@pytest.fixture(scope="module")
def inputs781():
    return chain.chain_inputs(781)


@pytest.fixture(scope="module")
def spec781(inputs781):
    return F.build_planet(inputs781, 781)


@pytest.fixture(scope="module")
def earth():
    return F.build_planet(chain.earth_inputs(), 0)


@pytest.fixture(scope="module")
def synthetic20():
    return [F.build_planet(chain.synthetic_inputs(s), s) for s in range(20)]


def _leaves(d, prefix=""):
    for key, value in d.items():
        if isinstance(value, dict) and key not in ("senses", "summary", "params") and value:
            yield from _leaves(value, f"{prefix}{key}.")
        else:
            yield f"{prefix}{key}"


# ------------------------------------------------------------------------------------- constants

def test_constants_match_scipy():
    assert C.G == sc.G
    assert C.K_B == sc.k and C.N_A == sc.N_A and C.H_PLANCK == sc.h and C.C_LIGHT == sc.c
    assert C.R_GAS == pytest.approx(sc.R, rel=1e-12)
    assert C.SIGMA == pytest.approx(sc.sigma, rel=1e-12)
    assert C.AU == sc.au
    assert C.DAY_S == sc.day and C.YEAR_S == sc.Julian_year


def test_masses_reproduce_the_iau_gm():
    # IAU 2015 B3 nominal GM values exactly, with the CODATA 2018 G
    assert C.G * C.M_SUN == pytest.approx(1.3271244e20, rel=1e-13)
    assert C.G * C.M_EARTH == pytest.approx(3.986004e14, rel=1e-13)
    assert C.M_SUN == pytest.approx(1.988410e30, rel=1e-6) and C.M_EARTH == pytest.approx(5.972168e24, rel=1e-6)
    # Kepler: 1 au around GM_sun is the Gaussian year 2 pi / k = 365.2568983 d (k = 0.01720209895)
    assert F.kepler_period_s(C.AU, C.M_SUN) / DAY == pytest.approx(2 * math.pi / 0.01720209895, abs=2e-6)


def test_solar_values_are_consistent():
    # IAU nominal L, R, T_eff obey Stefan-Boltzmann; S0 is L / (4 pi au^2) within 0.1 %
    assert 4 * math.pi * C.R_SUN ** 2 * C.SIGMA * C.T_SUN ** 4 == pytest.approx(C.L_SUN, rel=1e-3)
    assert C.L_SUN / (4 * math.pi * C.AU ** 2) == pytest.approx(C.S0, rel=1e-3)
    assert C.EARTH_BULK_DENSITY == pytest.approx(5514, abs=1)
    assert sum(C.EARTH_RADIOGENIC_SHARE.values()) == pytest.approx(1.0)


def test_molar_masses():
    assert C.molar_mass("CO2") == pytest.approx(0.044009, rel=1e-5)
    assert C.molar_mass("H2O") == pytest.approx(0.018015, rel=1e-4)
    assert C.molar_mass("Cu2CO3(OH)2") == pytest.approx(0.221114, rel=1e-5)
    assert C.element_mass("Fe") == pytest.approx(0.055845)


# ------------------------------------------------------------------------------------- chain

def test_chain_781_matches_section_3(inputs781):
    d = inputs781
    assert d["source"] == "world7" and d["seed"] == 781 and d["reached"] == "groups"
    assert d["chain_version"] == chain.CHAIN_VERSION
    star, planet, cloud = d["star"], d["planet"], d["cloud"]
    assert star["mass_msun"] == 0.617
    assert round(star["luminosity_lsun"], 4) == 0.1449
    assert star["age_yr"] == pytest.approx(3.5026e9, rel=1e-4)
    assert star["lifetime_yr"] == 4.26e10
    assert round(planet["mass_earth"], 3) == 0.609
    assert round(planet["orbit_au"], 5) == 0.40488
    assert round(planet["flux_earth"], 3) == 0.884
    # t_eq of the era flux (3 digits), which world7's temperature rule uses
    assert planet["era_flux_earth"] == 0.884 and planet["flux_earth"] != planet["era_flux_earth"]
    assert planet["t_eq_k"] == level3.equilibrium_temperature(0.884)
    assert round(planet["t_eq_k"], 1) == 246.9
    assert planet["greenhouse_k"] == 40.8 and planet["t_surface_k"] == 288 and planet["water_state"] == "liquid"
    expected = {"oxygen": 1.50e-3, "carbon": 4.58e-4, "iron": 1.87e-4, "magnesium": 1.82e-4, "silicon": 1.47e-4,
                "nitrogen": 1.37e-4, "metallicity": 3.05e-3}
    for key, value in expected.items():
        assert float(f"{cloud[key]:.3g}") == value, key
    ten = ("hydrogen", "helium", "carbon", "nitrogen", "oxygen", "neon", "magnesium", "silicon", "iron", "other")
    assert sum(cloud[k] for k in ten) == pytest.approx(1.0, abs=1e-9)
    assert d["biosphere"]["oxygen_pal"] == 0.118 and d["biosphere"]["trophic_levels"] == 3
    assert len(d["sky"]) == 5 and all({"orbit_au", "mass_earth", "t_k", "t_eq_k"} <= set(s) for s in d["sky"])


def test_sky_temperatures_mean_the_same_in_both_sources(inputs781):
    """t_k is level 3's whole-K temperature with its greenhouse (a gas giant: cloud tops), t_eq_k
    the bare equilibrium temperature, for world7 and synthetic siblings alike."""
    runaway = 0
    for d in (inputs781, *(chain.synthetic_inputs(s) for s in range(10))):
        for b in d["sky"]:
            flux = (b["t_eq_k"] / level3.T_EQ_EARTH) ** 4
            assert flux == pytest.approx(d["star"]["luminosity_lsun"] / b["orbit_au"] ** 2, rel=1e-9)
            warming = 0.0 if b["type"] == "gas" else level3.greenhouse(flux)
            assert b["t_k"] == math.floor(b["t_eq_k"] + warming + 0.5), (d["source"], b)
            runaway += warming == level3.GREENHOUSE_RUNAWAY
    assert runaway >= 2                       # 781's two inner siblings, 943 K and 843 K


def test_chain_provenance_covers_every_value(inputs781):
    for d in (inputs781, chain.synthetic_inputs(5), chain.earth_inputs()):
        prov = d["provenance"]
        missing = [k for k in _leaves({k: v for k, v in d.items() if k not in ("provenance", "seed", "source",
                                                                               "chain_version", "links",
                                                                               "truncated_after")})
                   if k not in prov and k.split(".")[0] not in prov]
        assert not missing, missing
        assert all(isinstance(v, str) and v for v in prov.values())
        assert set(d["planet"]) == set(inputs781["planet"])        # one structure for every source


def _close(a, b, rel=1e-12):
    """Nested equality with floats within ``rel``: world7 differs in the last bit between platforms (libm)."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k], rel) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_close(x, y, rel) for x, y in zip(a, b))
    if isinstance(a, float) or isinstance(b, float):
        return math.isclose(a, b, rel_tol=rel, abs_tol=1e-300)
    return a == b


def test_chain_cache_round_trip(tmp_path, inputs781, monkeypatch):
    fresh = chain.run_chain(781)
    values = {k: v for k, v in fresh.items() if k != "provenance"}
    assert _close(values, {k: v for k, v in inputs781.items() if k != "provenance"})   # the cache is current
    calls = []
    monkeypatch.setattr(chain, "run_chain", lambda seed: calls.append(seed) or fresh)
    cached = chain.chain_inputs(781, cache_dir=tmp_path)
    path = tmp_path / "world7-781.json"
    assert path.exists() and cached == fresh and calls == [781]
    assert chain.chain_inputs(781, cache_dir=tmp_path) == fresh and calls == [781]     # read back, not rerun
    stale = dict(fresh, chain_version="life9-chain-v1")
    path.write_text(json.dumps(stale), encoding="utf-8")
    assert chain.chain_inputs(781, cache_dir=tmp_path) == fresh and calls == [781, 781]  # old version reruns
    json.dumps(fresh, allow_nan=False)             # JSON-ready


def test_scan_is_the_bridge_prefix():
    from haishool.life8 import bridge
    steps, source, _ = chain._scan(781)
    assert steps == bridge.scan_world(781)
    assert "planets" in source.levels and "stars" in source.levels


def test_synthetic_inputs_are_deterministic_and_documented():
    a, b = chain.synthetic_inputs(11), chain.synthetic_inputs(11)
    assert a == b and a["source"] == "synthetic"
    assert a != chain.synthetic_inputs(12)
    for s in range(20):
        d = chain.synthetic_inputs(s)
        lo, hi = chain.SYNTHETIC_RANGES["star_mass_msun"]
        assert lo * 0.999 <= d["star"]["mass_msun"] <= hi * 1.001
        assert d["planet"]["water_state"] == "liquid" and d["planet"]["surface"] == "terran"
        assert d["planet"]["flux_earth"] * d["planet"]["orbit_au"] ** 2 == pytest.approx(d["star"]["luminosity_lsun"])
        assert d["cloud"]["metallicity"] >= 0.1 * C.Z_SUN
        assert d["star"]["elapsed_yr"] <= 0.9 * d["star"]["lifetime_yr"]
        p = d["planet"]
        assert p["t_surface_k"] == math.floor(p["t_eq_k"] + p["greenhouse_k"] + 0.5)   # world7's own rounding


# ------------------------------------------------------------------------------------- formulas

def test_par_fraction_is_the_planck_integral():
    for t in (3000.0, 4320.0, 5772.0, 7000.0):
        ref = quad(lambda lam: float(F.planck_lambda(lam, t)), 400e-9, 700e-9, epsrel=1e-12)[0]
        assert F.par_fraction(t) == pytest.approx(ref / (C.SIGMA * t ** 4 / math.pi), rel=1e-9)
    assert F.par_fraction(5772.0) == pytest.approx(0.366, abs=0.002)
    # the whole spectrum holds sigma T^4 / pi
    total = sum(F.band_fraction(5772.0, lo, hi) for lo, hi in ((5e-8, 4e-7), (4e-7, 2e-6), (2e-6, 2e-5), (2e-5, 1e-3)))
    assert total == pytest.approx(1.0, abs=1e-6)


def test_kasting_lock_radius_and_grey_tau():
    assert F.lock_radius_au(1.0, 4.5e9) == pytest.approx(0.027 * (13.5 * 4.5e9 / 100) ** (1 / 6))
    assert F.lock_radius_au(0.5, 4.5e9) < F.lock_radius_au(1.0, 4.5e9) < F.lock_radius_au(1.0, 9e9)
    tau = F.grey_tau(288.0, 255.0)
    assert F.grey_surface_k(255.0, tau) == pytest.approx(288.0)
    assert F.e_sat_pa(273.15) == pytest.approx(610.94)
    assert F.e_sat_pa(293.15) == pytest.approx(2339.2, rel=0.005)     # IAPWS-95 at 20 C (Wagner & Pruss 2002)


def test_compression_reproduces_rocky_planets():
    def radius(m, cmf):
        rho0 = F.uncompressed_density(cmf)
        return (3 * m * C.M_EARTH / (4 * math.pi * rho0 * F.compression(m, rho0))) ** (1 / 3) / C.R_EARTH
    assert F.compression(1.0) == pytest.approx(F.earth_compression(), rel=1e-12)
    assert radius(0.815, 0.32) == pytest.approx(0.9499, rel=0.02)       # Venus
    assert radius(0.107, 0.24) == pytest.approx(0.532, rel=0.03)        # Mars
    for m in (1.0, 3.0, 8.0):                                           # Zeng et al. 2016 (1-8 M_E)
        assert radius(m, 0.325) == pytest.approx((1.07 - 0.21 * 0.325) * m ** (1 / 3.7), rel=0.02)


def test_radiogenic_decay_factor():
    assert F.radiogenic_decay_factor(C.EARTH_AGE_YR) == pytest.approx(1.0, rel=1e-12)
    assert F.radiogenic_decay_factor(8.71e9) == pytest.approx(0.583, abs=1e-3)
    ages = [1e9 * k for k in range(1, 14)]
    values = [F.radiogenic_decay_factor(a) for a in ages]
    assert all(a > b > 0 for a, b in zip(values, values[1:]))           # older bodies are cooler
    # a single isotope decays by its half-life: the K-40 share halves its contribution in 1.248 Gyr
    k40 = C.EARTH_RADIOGENIC_SHARE["K40"] * 0.5
    assert F.radiogenic_decay_factor(C.EARTH_AGE_YR + C.HALF_LIFE_YR["K40"]) < 1 - k40 + 1e-12


def test_vapour_column_rule_on_earth(earth):
    # H_w = R_v T^2 / (L_v Gamma) about 2.3 km; the column within 15 % of the observed 24.6 kg/m^2
    assert 2000 < earth.vapour_scale_height_m < 2600
    assert earth.vapour_column_kg_m2 == pytest.approx(C.EARTH_VAPOUR_COLUMN, rel=0.15)
    area = 4 * math.pi * earth.radius_m ** 2
    vapour = earth.vapour_column_kg_m2 * area
    assert vapour == pytest.approx(1.27e16, rel=0.15)                   # Trenberth & Smith 2005, J. Climate 18, 864
    dry = (earth.surface_pressure_pa - earth.partial_pressure_pa["H2O"]) * area / earth.gravity_m_s2
    assert earth.atmosphere_mass_kg == pytest.approx(dry + vapour, rel=1e-12)
    assert vapour < 0.01 * earth.atmosphere_mass_kg
    assert earth.ocean_mass_kg + vapour == pytest.approx(C.EARTH_OCEAN_MASS_FRACTION * C.M_EARTH, rel=1e-12)


def test_tau_per_ln_co2_is_the_myhre_slope(spec781, earth, synthetic20):
    """Raising tau by tau_per_ln_co2 per e-fold of CO2 cuts the grey OLR at fixed T_s by 5.35 W/m^2
    per e-fold (finite difference)."""
    for s in (spec781, earth, *synthetic20[:5]):
        olr = lambda tau: C.SIGMA * s.t_surface_target_k ** 4 / (1 + 0.75 * tau)     # noqa: E731
        h = 1e-4
        slope = (olr(s.greenhouse_tau - h * s.tau_per_ln_co2) - olr(s.greenhouse_tau + h * s.tau_per_ln_co2)) / (2 * h)
        assert slope == pytest.approx(5.35, abs=1e-3), s.seed
    # a doubling at fixed T_eq: about 1.26 K for 781 (1.27 K linearised; Planck only, stated in the note)
    s = spec781
    dt = F.grey_surface_k(s.t_eq_k, s.greenhouse_tau + s.tau_per_ln_co2 * math.log(2)) - s.t_surface_target_k
    assert dt == pytest.approx(1.26, abs=0.01)
    assert "1.27 K per CO2 doubling" in s.provenance["tau_per_ln_co2"][1]


# ------------------------------------------------------------------------------------- planet 781

def test_spec_781(spec781):
    s = spec781
    assert F.check_spec(s) == []
    assert s.year_s / DAY == pytest.approx(119.8, abs=0.1)
    assert s.core_mass_fraction == pytest.approx(0.233, abs=5e-4)
    assert s.tidally_locked and math.isinf(s.day_length_s) and s.obliquity_rad == 0.0 and s.rotation_period_s == s.year_s
    assert 0.6 * C.AU < s.lock_radius_m < 0.8 * C.AU
    assert s.partial_pressure_pa["O2"] == pytest.approx(2.5e3, rel=0.01)
    assert s.oxygen_mole_fraction < 0.15                                 # no fire until oxygen rises
    assert s.greenhouse_tau == pytest.approx(1.13, abs=0.01)
    assert abs(s.t_eq_k - 246.9) < 1.0
    assert s.t_surface_target_k == pytest.approx(level3.equilibrium_temperature(0.884) + 40.8, abs=1e-12)
    assert s.insolation_w_m2 == pytest.approx(1203, abs=1)
    assert s.star_teff_k == pytest.approx(4320, abs=5) and 0.2 < s.par_fraction < 0.3
    assert s.relief_m == pytest.approx(600 * 9.81 / s.gravity_m_s2)
    assert 0 < s.radiogenic_heat_w_m2 < 0.047 * s.metallicity / C.Z_SUN
    assert s.surface_pressure_pa == pytest.approx(sum(s.partial_pressure_pa.values()))
    assert set(s.partial_pressure_pa) == set(F.GASES)
    assert not s.co2_capped and s.tau_residual == 0.0


#: planet 781's derived values (PLANET-SPEC section 3 and the formation rules), rel 1e-3; a wrong
#: compression, mixture rule or calibration moves them
KEY_781 = {"radius_m": 5674.4e3, "core_radius_m": 2742.8e3, "mean_density_kg_m3": 4755, "gravity_m_s2": 7.544,
           "atmosphere_mass_kg": 4.7156e18, "air_molar_mass_kg_mol": 0.029033, "air_cp_j_kg_k": 1018.4,
           "air_gamma": 1.3912, "scale_height_m": 10.92e3, "air_density_kg_m3": 1.080, "sound_speed_m_s": 338.5,
           "co2_ref_pa": 5130.5, "tau_per_ln_co2": 0.06248, "radiogenic_heat_w_m2": 0.005427,
           "vapour_scale_height_m": 3054, "vapour_column_kg_m2": 29.23, "surface_pressure_pa": 88971}


@pytest.mark.parametrize("name", sorted(KEY_781))
def test_key_values_781(spec781, name):
    assert getattr(spec781, name) == pytest.approx(KEY_781[name], rel=1e-3)


def test_earth_twin_is_earth(earth):
    e = earth
    assert F.check_spec(e) == []
    assert e.core_mass_fraction == pytest.approx(C.EARTH_CORE_MASS_FRACTION, abs=0.01)
    assert e.radius_m == pytest.approx(C.R_EARTH, rel=0.005)
    assert e.gravity_m_s2 == pytest.approx(9.82, rel=0.01)
    assert e.year_s / DAY == pytest.approx(365.25, abs=0.1)
    assert e.t_eq_k == pytest.approx(254.6, abs=0.1)
    assert e.co2_ref_pa == pytest.approx(278e-6 * C.P_STANDARD, rel=1e-9)
    assert not e.co2_capped and e.tau_residual == 0.0
    assert e.surface_pressure_pa == pytest.approx(101325 + 0.77 * F.e_sat_pa(287.6), rel=0.01)
    assert e.radiogenic_heat_w_m2 == pytest.approx(0.047, rel=0.02)
    assert e.star_teff_k == pytest.approx(C.T_SUN, rel=1e-12)
    assert e.sound_speed_m_s == pytest.approx(340, abs=3) and e.air_density_kg_m3 == pytest.approx(1.23, abs=0.02)
    assert e.air_molar_mass_kg_mol == pytest.approx(0.02897, rel=0.01)  # dry air 28.97 g/mol, a little vapour
    assert not e.tidally_locked
    assert 12 * 3600 <= e.rotation_period_s <= 48 * 3600 and e.day_length_s > e.rotation_period_s
    assert 0.0 <= e.obliquity_rad <= math.radians(35)


# ------------------------------------------------------------------------------------- gates

def test_check_spec_passes_for_20_synthetic_planets(synthetic20):
    for s in synthetic20:
        assert s.source == "synthetic"
        assert F.check_spec(s) == [], s.seed


@pytest.mark.parametrize("seed", WORLD7_SEEDS)
def test_check_spec_passes_for_world7_planets(seed):
    d = chain.chain_inputs(seed)
    s = F.build_planet(d, seed)
    assert F.check_spec(s) == [], seed
    p = d["planet"]
    assert p["t_surface_k"] == math.floor(p["t_eq_k"] + p["greenhouse_k"] + 0.5)   # world7's own rounding
    assert s.t_surface_target_k == pytest.approx(p["t_eq_k"] + p["greenhouse_k"], abs=1e-12)


def test_co2_cap_is_flagged(synthetic20):
    capped = [s for s in synthetic20 if s.co2_capped]
    free = [s for s in synthetic20 if not s.co2_capped]
    assert capped and free                               # the cap is common (about 70 % of planets)
    rules = F.FormationRules()
    for s in capped:
        assert s.co2_ref_pa == rules.co2_max_pa and s.tau_residual > 0
        assert "capped" in s.provenance["co2_ref_pa"][1]
        # the CO2 part plus the residual is the whole tau, and the CO2 part alone leaves T_s cooler
        assert F.co2_tau(s.co2_ref_pa, s.t_surface_target_k) + s.tau_residual == pytest.approx(s.greenhouse_tau)
        assert F.grey_surface_k(s.t_eq_k, s.greenhouse_tau - s.tau_residual) < s.t_surface_target_k
    for s in free:
        assert s.co2_ref_pa < rules.co2_max_pa and s.tau_residual == 0.0
        assert F.co2_tau(s.co2_ref_pa, s.t_surface_target_k) == pytest.approx(s.greenhouse_tau, rel=1e-9)
    # a higher cap removes the residual of a capped planet
    s = capped[0]
    loose = F.build_planet(chain.synthetic_inputs(s.seed), s.seed, replace(rules, co2_max_pa=1e9))
    assert not loose.co2_capped and loose.co2_ref_pa > s.co2_ref_pa and F.check_spec(loose) == []


def test_provenance_complete(spec781, synthetic20):
    for s in (spec781, *synthetic20):
        names = {f.name for f in fields(s)} - {"provenance"}
        assert set(s.provenance) == names
        for name, (tag, note) in s.provenance.items():
            assert tag and all(part in F.TAGS for part in tag.split("+")), (name, tag)
            assert isinstance(note, str) and note
    tags = {name: tag for name, (tag, _) in spec781.provenance.items()}
    assert tags["star_mass_kg"] == "chain" and tags["albedo"] == "reference"
    assert "new_rule" in tags["ocean_mass_kg"] and "new_rule" in tags["relief_m"]
    assert spec781.provenance["crust"][1] == materials.CRUST_RULE


def test_check_spec_catches_errors(spec781, synthetic20):
    s = spec781
    assert F.check_spec(replace(s, gravity_m_s2=s.gravity_m_s2 * 1.01))
    assert F.check_spec(replace(s, core_mass_kg=s.core_mass_kg * 1.001))
    assert F.check_spec(replace(s, greenhouse_tau=s.greenhouse_tau + 0.05))
    assert F.check_spec(replace(s, chain_t_eq_k=s.t_eq_k + 2))
    assert F.check_spec(replace(s, atmosphere_mass_kg=s.atmosphere_mass_kg * 1.01))
    assert F.check_spec(replace(s, vapour_column_kg_m2=s.vapour_column_kg_m2 * 5))
    assert F.check_spec(replace(s, co2_capped=True))
    assert F.check_spec(replace(s, co2_ref_pa=s.co2_ref_pa * 1.01))
    capped = next(x for x in synthetic20 if x.co2_capped)
    assert F.check_spec(replace(capped, tau_residual=0.0))
    reordered = dict(reversed(list(s.crust.items())))
    assert any("CRUST_SPECIES" in m for m in F.check_spec(replace(s, crust=reordered)))
    prov = dict(s.provenance)
    del prov["relief_m"]
    assert any("relief_m" in m for m in F.check_spec(replace(s, provenance=prov)))
    prov = dict(s.provenance, relief_m=("guess", "x"))
    assert F.check_spec(replace(s, provenance=prov))


def test_crust_fractions(inputs781, spec781, earth, synthetic20):
    for s in (spec781, earth, *synthetic20):
        assert tuple(s.crust) == tuple(materials.CRUST_SPECIES)          # the [W, S] order globe uses
        assert sum(s.crust.values()) == pytest.approx(1.0, abs=1e-6)
        assert min(s.crust.values()) >= 0
    assert F.crust_fractions(inputs781, spec781) == pytest.approx(spec781.crust)
    assert spec781.crust == pytest.approx(materials.crust_fractions(inputs781))
    # 781 is metal-poor (0.23 Z_sun) and Mg-rich: fewer trace ores, more basalt than the Earth twin
    a, e = spec781.crust, earth.crust
    assert a["malachite"] < e["malachite"] and a["cassiterite"] < e["cassiterite"]
    assert a["basalt"] > e["basalt"]


def test_to_dict_is_strict_json(spec781, earth):
    d = spec781.to_dict()
    text = json.dumps(d, allow_nan=False)                # a locked planet's infinite day is null
    assert d["day_length_s"] is None and d["tidally_locked"] is True
    assert json.loads(text)["radius_m"] == spec781.radius_m
    assert json.dumps(earth.to_dict(), allow_nan=False) and earth.to_dict()["day_length_s"] == earth.day_length_s


def test_tensors_and_stacking(spec781, synthetic20):
    t = spec781.tensors("cpu")
    assert set(t) == set(spec781.scalar_names()) | {"crust"}
    assert all(v.dtype == torch.float32 and v.dim() == 0 for k, v in t.items() if k != "crust")
    assert t["tidally_locked"].item() == 1.0 and math.isinf(t["day_length_s"].item())
    assert t["co2_capped"].item() == 0.0 and "vapour_column_kg_m2" in t
    assert t["p_O2_pa"].item() == pytest.approx(spec781.partial_pressure_pa["O2"], rel=1e-6)
    specs = [spec781, *synthetic20[:3]]
    stack = F.stack_specs(specs, "cpu")
    assert stack["gravity_m_s2"].shape == (4,) and stack["crust"].shape == (4, len(materials.CRUST_SPECIES))
    assert torch.allclose(stack["crust"].sum(1), torch.ones(4))
    assert stack["radius_m"][1].item() == pytest.approx(synthetic20[0].radius_m, rel=1e-6)
    d64 = F.stack_specs(specs, "cpu", torch.float64)
    assert d64["planet_mass_kg"].dtype == torch.float64 and d64["planet_mass_kg"][0].item() == spec781.planet_mass_kg


def test_build_is_deterministic(inputs781, spec781):
    again = F.build_planet(json.loads(json.dumps(inputs781)), 781)
    assert again.to_dict() == spec781.to_dict()
    e1, e2 = F.build_planet(chain.earth_inputs(), 5), F.build_planet(chain.earth_inputs(), 5)
    assert e1.rotation_period_s == e2.rotation_period_s and e1.obliquity_rad == e2.obliquity_rad
    assert F.build_planet(chain.earth_inputs(), 6).rotation_period_s != e1.rotation_period_s
    text = F.describe(spec781)
    assert "core_mass_fraction" in text and "[chain]" in text and "kg/m^2" in text
