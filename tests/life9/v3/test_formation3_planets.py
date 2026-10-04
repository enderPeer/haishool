"""formation3 over many planets: 781, Earth, 20 synthetic and 10 cached world7 planets pass check_spec3, close
their water, C, N and Ar budgets, and report the derived T_s against the chain's (PLANET-V3-SPEC section 9).

Outcomes (runaway, frozen, water-covered, the mismatch) are reported, never gated (spec section 10): the asserts
only check that each flag agrees with the numbers it summarises."""
from __future__ import annotations

import math

import pytest

from haishool.life9.planet import chain
from haishool.life9.planet import constants as C
from haishool.life9.v3 import formation3 as F3

#: world7 seeds with a cached chain (runs/life9/chain-cache); a missing cache file skips the seed instead of
#: running world7
WORLD7_SEEDS = (6, 7, 9, 10, 29, 35, 40, 54, 56, 63)
SYNTHETIC_SEEDS = tuple(range(20))


def _cached(seed: int) -> bool:
    return chain._cache_path(seed, None).exists()


def _inputs():
    out = [("earth", chain.earth_inputs(), 0)]
    if _cached(781):
        out.append(("world7", chain.chain_inputs(781), 781))
    out += [("synthetic", chain.synthetic_inputs(s), s) for s in SYNTHETIC_SEEDS]
    out += [("world7", chain.chain_inputs(s), s) for s in WORLD7_SEEDS if _cached(s)]
    return out


@pytest.fixture(scope="module")
def inputs():
    return _inputs()


@pytest.fixture(scope="module")
def planets(inputs):
    return [(kind, F3.build_planet3(inp, seed)) for kind, inp, seed in inputs]


def test_enough_planets(planets):
    kinds = [k for k, _ in planets]
    assert kinds.count("synthetic") == 20
    assert kinds.count("world7") >= 8, "the chain cache lost world7 seeds"


def test_every_planet_passes_check_spec3(planets, inputs):
    for (kind, s), (_, inp, _) in zip(planets, inputs):
        assert F3.check_spec3(s, inputs=inp) == [], (kind, s.seed, F3.check_spec3(s, inputs=inp))


def test_budgets_close_everywhere(planets):
    for kind, s in planets:
        area = 4 * math.pi * s.radius_m ** 2
        assert s.surface_water_kg + s.undegassed_water_kg == pytest.approx(s.bulk_water_kg, rel=1e-12)
        assert s.ocean_mass_kg + s.vapour_column_kg_m2 * area == pytest.approx(s.surface_water_kg, rel=1e-9)
        assert s.air_carbon_kg + s.carbonate_carbon_kg + s.undegassed_carbon_kg == pytest.approx(s.bulk_carbon_kg,
                                                                                                rel=1e-12)
        assert s.air_nitrogen_kg + s.undegassed_nitrogen_kg == pytest.approx(s.bulk_nitrogen_kg, rel=1e-12)
        # nitrogen has its own degassed share (Earth's N2), water and carbon theirs (Earth's surface water)
        assert s.nitrogen_degassed_share == F3.nitrogen_degassed_share()
        assert s.air_nitrogen_kg == pytest.approx(s.nitrogen_degassed_share * s.bulk_nitrogen_kg, rel=1e-12)
        assert s.surface_carbon_kg == pytest.approx(s.degassed_share * s.bulk_carbon_kg, rel=1e-12)
        assert s.air_argon_kg + s.retained_argon_kg == pytest.approx(s.argon_produced_kg, rel=1e-12)
        assert s.core_mass_kg + s.mantle_mass_kg + s.ocean_mass_kg + s.atmosphere_mass_kg == \
            pytest.approx(s.planet_mass_kg, rel=1e-12)
        assert s.water_mass_fraction == pytest.approx(s.degassed_share * s.solid_water_fraction, rel=1e-12)
        # K (and the heat flux) per kg of rock follow the cloud's other metals per rock
        assert s.potassium_kg == pytest.approx(240e-6 * s.lithophile_cloud_scale * s.mantle_mass_kg, rel=1e-9)
        assert s.fixed_point_converged


def test_flags_agree_with_the_numbers(planets):
    """Consistency, not outcomes: a runaway, frozen or water-covered planet is a result."""
    t_max = F3.Formation3Rules().t_max_k
    for kind, s in planets:
        assert 200.0 < s.disc_temperature_k < 300.0, (kind, s.seed)          # the chain's temperate inputs
        assert s.disc_temperature_k == pytest.approx(278.3 * s.flux_earth ** 0.25, rel=1e-9)
        assert s.t_eq_k <= s.t_surface_target_k <= t_max
        assert s.runaway == (s.t_surface_target_k >= t_max), (kind, s.seed)
        if not s.runaway:
            assert s.t_surface_target_k ** 4 == pytest.approx(s.t_eq_k ** 4 * (1 + 0.75 * s.greenhouse_tau), rel=1e-9)
        assert s.frozen_mean == (s.t_surface_target_k < 273.15)
        assert s.pressure_beyond_fit == (s.dry_pressure_rel_model_earth > 2.0)
        assert math.isfinite(s.chain_t_s_mismatch_k)
        assert 0.0 <= s.ocean_fraction <= 1.0 and s.relief_m > s.basin_depth_m > 0
        assert (s.land_fraction == 0.0) == (s.weathering_land_share == 0.0)
        assert not s.no_land_weathering_assumed
        assert s.partial_pressure_pa["CO2"] > 0 and s.partial_pressure_pa["N2"] > 0 and s.partial_pressure_pa["Ar"] > 0
        # both readings of the carbon-silicate steady state are there; the fixed point drives the spec
        assert s.t_surface_selfconsistent_k == s.t_surface_target_k
        assert s.t_surface_chain_weathering_k >= s.t_eq_k and s.t_surface_global_weathering_k >= s.t_eq_k
        # the fixed point is warmer than weathering at the chain's T where the planet is colder than the chain
        if s.t_surface_chain_weathering_k < s.chain_t_target_k - 0.5:
            assert s.t_surface_target_k > s.t_surface_chain_weathering_k
        # the imposed greenhouse terms only warm
        assert s.t_surface_no_clouds_k <= s.t_surface_target_k + 1e-6
        assert s.t_surface_no_other_gases_k <= s.t_surface_target_k + 1e-6


def test_wetter_solids_beyond_earths_orbit(planets):
    earth = planets[0][1]
    for kind, s in planets[1:]:
        if s.disc_temperature_k < earth.disc_temperature_k:
            assert s.water_mass_fraction > earth.water_mass_fraction
        else:
            assert s.water_mass_fraction <= earth.water_mass_fraction


def test_report_table(planets, capsys):
    """The derived T_s against the chain's, the flags and the sensitivities (``pytest -s`` prints it)."""
    rows = [F3.summary_row(s) for _, s in planets]
    lines = ["source     seed  flux  T_disc water/M  ocean (fixD  D~1/g) lith  q_mW  V_rel landW p_N2kPa  p_CO2Pa "
             "P/P_E  chainT   T_s  mism  T(chainW) T(globW) cloudK otherK broadK  x_O2 flags"]
    for (kind, s), r in zip(planets, rows):
        flags = "".join(c for c, b in (("L", r["carbon_limited"]), ("C", r["co2_beyond_fit"]),
                                       ("P", r["pressure_beyond_fit"]), ("F", r["frozen_mean"]), ("R", r["runaway"]),
                                       ("f", r["fire_possible"]))
                        if b)
        lines.append(f"{kind:10s} {r['seed']:4d} {r['flux']:.3f} {r['t_disc_k']:6.1f} {r['water_mass_fraction']:.2e} "
                     f"{r['ocean_fraction']:.3f} ({r['ocean_fraction_fixed_depth']:.3f} "
                     f"{r['ocean_fraction_inverse_g_depth']:.3f}) {r['lithophile_scale']:.2f} {r['heat_w_m2'] * 1e3:5.1f} "
                     f"{r['outgassing_rel']:5.2f} {r['weathering_land_share']:.2f} {r['p_n2_pa'] / 1e3:7.1f} "
                     f"{r['p_co2_pa']:9.3g} {r['pressure_rel_model_earth']:5.2f} {r['chain_t_k']:6.1f} {r['t_s_k']:6.1f} "
                     f"{r['mismatch_k']:+6.1f} {r['t_chain_weathering_k']:8.1f} {r['t_global_weathering_k']:8.1f} "
                     f"{r['cloud_warming_k']:6.1f} {r['other_gases_warming_k']:6.2f} {r['broadening_warming_k']:+6.1f} "
                     f"{r['x_o2_dry']:.3f} {flags}")
    others = rows[1:]
    count = {name: sum(bool(r[name]) for r in others) for name in
             ("frozen_mean", "runaway", "pressure_beyond_fit", "co2_beyond_fit", "carbon_limited", "fire_possible")}
    water = sum(r["ocean_fraction"] >= 1.0 for r in others)
    water_fixed = sum(r["ocean_fraction_fixed_depth"] >= 1.0 for r in others)
    water_1g = sum(r["ocean_fraction_inverse_g_depth"] >= 1.0 for r in others)
    lines.append(f"{len(others)} planets besides Earth: water-covered {water} (fixed D {water_fixed}, D ~ 1/g {water_1g}), "
                 + ", ".join(f"{k} {v}" for k, v in count.items())
                 + f"; model Earth air {rows[0]['pressure_pa'] / 1e5:.3f} bar, x_O2 {rows[0]['x_o2']:.4f} "
                   f"(N2 calibrated on Earth's {F3.EARTH_N2_PA / 1e3:.1f} kPa)")
    with capsys.disabled():
        print("\n" + "\n".join(lines))
    mism = [r["mismatch_k"] for r in others]
    assert all(math.isfinite(m) for m in mism)
    assert rows[0]["mismatch_k"] == pytest.approx(0.0, abs=0.3)          # Earth
    assert C.EARTH_OCEAN_MASS_FRACTION == pytest.approx(rows[0]["water_mass_fraction"], rel=1e-12)
    assert rows[0]["p_n2_pa"] == pytest.approx(F3.EARTH_N2_PA, rel=1e-9) and rows[0]["fire_possible"]
    assert all(r["fire_possible"] == (r["x_o2_dry"] >= 0.15) for r in rows)
    assert water_1g >= water >= water_fixed      # the split lies between the bounds


def test_formation_site_anchors_sensitivity(inputs, planets, capsys):
    """Meteorite anchors at the black body where each class formed (EC at Earth's 278 K, OC at 190 K): a reported
    sensitivity of water per mass and of the water-world count (spec-level choice, not the default)."""
    rules = F3.Formation3Rules(classes=F3.METEORITE_CLASSES_SITES)
    sites = [F3.build_planet3(inp, seed, rules) for _, inp, seed in inputs]
    earth_d, earth_s = planets[0][1], sites[0]
    assert earth_s.water_mass_fraction == pytest.approx(C.EARTH_OCEAN_MASS_FRACTION, rel=1e-12)   # same calibration
    for s in sites:
        assert F3.check_spec3(s, rules) == [], s.seed
    cold = [(d, s) for (_, d), s in zip(planets[1:], sites[1:]) if d.disc_temperature_k < 230.0]
    ratio_d = max(d.water_mass_fraction for d, _ in cold) / earth_d.water_mass_fraction if cold else float("nan")
    ratio_s = max(s.water_mass_fraction for _, s in cold) / earth_s.water_mass_fraction if cold else float("nan")
    water_d = sum(d.ocean_fraction >= 1.0 for _, d in planets[1:])
    water_s = sum(s.ocean_fraction >= 1.0 for s in sites[1:])
    with capsys.disabled():
        print(f"\nanchor sites: degassed share {earth_s.degassed_share:.4f} (default {earth_d.degassed_share:.4f}); "
              f"nitrogen share {earth_s.nitrogen_degassed_share:.4g} (default {earth_d.nitrogen_degassed_share:.4g}); "
              f"model Earth air {earth_s.surface_pressure_pa / 1e5:.3f} bar (default {earth_d.surface_pressure_pa / 1e5:.3f}); "
              f"coldest-orbit water per mass {ratio_s:.2f}x Earth's (default {ratio_d:.2f}x); "
              f"water-covered {water_s} (default {water_d}) of {len(sites) - 1}")
    assert earth_s.degassed_share > earth_d.degassed_share     # EC-like Earth needs a larger degassed share
    # nitrogen's own share keeps the model Earth's air at Earth's N2 under either anchors (one share for water, C
    # and N gave this sensitivity 107 bar of air)
    assert earth_s.partial_pressure_pa["N2"] == pytest.approx(F3.EARTH_N2_PA, rel=1e-9)
    assert earth_s.nitrogen_degassed_share == F3.nitrogen_degassed_share(rules) != earth_d.nitrogen_degassed_share
