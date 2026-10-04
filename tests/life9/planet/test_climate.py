"""Tests of the planet climate (PLANET-SPEC 2.6 and the climate gates of section 4)."""
from __future__ import annotations

import math
from dataclasses import fields

import pytest
import torch

from haishool.life9.planet import chain, climate as cl, constants as K, formation as F, globe as gb

R_HAB = 1.0e4
TAGS = {"chain", "derived", "reference", "new_rule"}
#: world7 planet 781 (locked, CO2 not capped) and synthetic locked planets: 6 (not capped), 9 (capped, cold),
#: 11 (not capped)
LOCKED = (781, 6, 9, 11)
#: rotating planets: Earth (chain.earth_inputs, seed 0) and synthetic 7 (obliquity 29 deg)
ROTATING = (0, 7)


def _spec(seed):
    if seed == 0:
        return F.build_planet(chain.earth_inputs(), 0)
    if seed >= 100:
        return F.build_planet(chain.chain_inputs(seed), seed)
    return F.build_planet(chain.synthetic_inputs(seed), seed)


def build(seeds, G=16, terrain_seed=1):
    specs = [_spec(s) for s in seeds]
    globe = gb.Globe(G)
    water = torch.stack([gb.habitat_water_volume(s.ocean_mass_kg, s.radius_m, s.gravity_m_s2, s.relief_m, R_HAB)
                         for s in specs])
    relief = torch.tensor([s.relief_m for s in specs])
    terrain = gb.make_terrain(globe, len(specs), torch.Generator().manual_seed(terrain_seed), relief, water, R_HAB)
    P = cl.make_params(specs, globe, terrain, R_HAB)
    return globe, specs, terrain, P


@pytest.fixture(scope="module")
def locked():
    globe, specs, terrain, P = build(LOCKED)
    state, report = cl.spin_up(cl.init_state(globe, P), globe, P)
    return globe, specs, P, state, report


@pytest.fixture(scope="module")
def rotating():
    globe, specs, terrain, P = build(ROTATING)
    # two seasonal chunks (one 484-day year each) instead of four keep the test short; the gate is 3 K
    state, report = cl.spin_up(cl.init_state(globe, P), globe, P, slow_chunks=2)
    return globe, specs, P, state, report


def _mean(globe, x):
    return (x.double() * globe.area64).sum(-1) / (4 * math.pi)


# ------------------------------------------------------------------------------------------- set-up
def test_rules_and_params_have_provenance():
    for f in fields(cl.ClimateRules):
        assert f.name in cl.PROVENANCE, f.name
    for name, (tag, note) in cl.PROVENANCE.items():
        assert note and all(part in TAGS for part in tag.split("+")), name
    globe, specs, terrain, P = build((781,), G=8)
    cl.step(cl.init_state(globe, P), globe, P)                        # fills the caches ("_" keys)
    assert "_diffuse_steps" in P
    # every number in P has its entry: tensors and floats, the float64 twins under their base name
    for name, value in P.items():
        if name in ("W", "rules", "provenance") or name.startswith("_"):
            continue
        assert isinstance(value, (torch.Tensor, float, int)), name
        key = name[:-2] if name.endswith("64") and name[:-2] in P else name
        tag, note = P["provenance"][key]
        assert note and all(part in TAGS for part in tag.split("+")), name
    assert P["provenance"]["m_gas"][0] == "reference" and P["provenance"]["tau_earth"][0] == "derived"


def test_initial_gas_columns_give_the_spec_pressures():
    globe, specs, terrain, P = build((781, 9), G=8)
    state = cl.init_state(globe, P)
    p = cl.partial_pressures(state, P)
    for w, s in enumerate(specs):
        for i, gas in enumerate(cl.GASES):
            assert float(p[w, i]) == pytest.approx(s.partial_pressure_pa[gas], rel=1e-12)
    assert state.gas.dtype == torch.float64 and state.T.dtype == torch.float32
    # vapour starts from the spec's column; its surface pressure is the spec's p_H2O at T_s on the ocean
    assert float(state.vapour[0].mean()) == pytest.approx(specs[0].vapour_column_kg_m2, rel=1e-6)
    ocean = ~P["land"][0]
    e = cl.vapour_pressure(state, P)[0][ocean]
    assert float(e.mean()) == pytest.approx(specs[0].partial_pressure_pa["H2O"], rel=1e-4)


def test_wtg_weight_locked_and_earth():
    earth, p781 = _spec(0), _spec(781)
    assert cl.wtg_weight(earth) < 0.06          # fast rotator: the Sellers diffusion carries the heat
    assert cl.wtg_weight(p781) > 0.99           # locked, 120-day rotation: weak temperature gradients


# ------------------------------------------------------------------------------------------- insolation
def test_insolation_locked_planet():
    globe, specs, terrain, P = build((781,), G=16)
    I = cl.insolation(globe, P, 5)
    S = specs[0].insolation_w_m2
    night = globe.centers[:, 0] <= 0
    assert float(I[0, night].abs().max()) == 0.0
    assert float(I.max()) == pytest.approx(S * float(globe.centers[:, 0].max()), rel=1e-6)
    assert float(_mean(globe, I)) == pytest.approx(S / 4, rel=2e-3)
    assert torch.equal(I, cl.insolation(globe, P, 77))          # no seasons
    assert torch.equal(I, cl.annual_insolation(globe, P))


def test_insolation_rotating_planet():
    globe, specs, terrain, P = build((0,), G=16)
    S = specs[0].insolation_w_m2
    year = float(P["year_days"][0])
    # day -0.5 is the vernal equinox: no declination, S / pi at the equator, hemispheres alike
    eq = cl.insolation(globe, P, -0.5)
    assert float(cl.declination(P, -0.5)[0]) == pytest.approx(0.0, abs=1e-12)
    assert float(_mean(globe, eq)) == pytest.approx(S / 4, rel=2e-3)
    lat = globe.lat
    expected = S / math.pi * torch.cos(lat.double())
    assert torch.allclose(eq[0].double(), expected, rtol=1e-5, atol=1e-3)
    # northern summer solstice: polar night in the south, the north pole lit all day
    sol = cl.insolation(globe, P, year / 4 - 0.5)
    dec = float(cl.declination(P, year / 4 - 0.5)[0])
    assert dec == pytest.approx(specs[0].obliquity_rad, rel=1e-6)
    south = lat < -(math.pi / 2 - dec) + 0.02
    assert float(sol[0, south].max()) == 0.0
    north = lat > (math.pi / 2 - dec) + 0.02
    assert float(sol[0, north].min()) > 0.0
    # the year mean is S / 4 over the sphere and symmetric between the hemispheres
    ann = cl.annual_insolation(globe, P)
    assert float(_mean(globe, ann)) == pytest.approx(S / 4, rel=2e-3)
    nh = (ann[0] * globe.area * (lat > 0)).sum() / (globe.area * (lat > 0)).sum()
    sh = (ann[0] * globe.area * (lat < 0)).sum() / (globe.area * (lat < 0)).sum()
    assert float(nh) == pytest.approx(float(sh), rel=1e-3)


# ------------------------------------------------------------------------------------------- numerics
def test_transport_conserves_heat_and_vapour():
    globe, specs, terrain, P = build((781, 0), G=16)
    gen = torch.Generator().manual_seed(3)
    T = 250 + 60 * torch.rand(2, globe.C, generator=gen)
    v = 40 * torch.rand(2, globe.C, generator=gen)
    T2, v2 = cl.transport(T, v, P["heat"], globe, P)
    lift = P["lapse"][:, None] * P["height_m"]
    e0 = _mean(globe, P["heat"].double() * (T + lift).double())
    e1 = _mean(globe, P["heat"].double() * (T2 + lift).double())
    assert torch.allclose(e1, e0, rtol=1e-7)
    assert torch.allclose(_mean(globe, v2), _mean(globe, v), rtol=1e-6)
    # it mixes: the spread falls
    assert bool((T2.std(-1) < T.std(-1)).all()) and bool((v2.std(-1) < v.std(-1)).all())


def test_column_energy_changes_by_the_toa_imbalance():
    """Transport conserves energy, so sum(area C T) changes by exactly the diagnosed TOA imbalance."""
    globe, specs, terrain, P = build((781, 0), G=16)
    state = cl.run(cl.init_state(globe, P), globe, P, 10)[0]
    new, diag = cl.step(state, globe, P)
    heat = P["heat"].double()
    change = _mean(globe, heat * (new.T.double() - state.T.double())) / K.DAY_S
    # up to the transport's one float32 rounding of T (an ulp of a 300 K ocean cell is 0.07 W/m^2 of a day's flux)
    assert torch.allclose(change, diag["toa_imbalance"], rtol=1e-4, atol=2e-3)       # W/m^2
    assert torch.allclose(diag["toa_imbalance"], _mean(globe, diag["absorbed"].double() - diag["olr"].double()),
                          rtol=0, atol=1e-4)


def test_step_is_stable_from_a_rough_start():
    globe, specs, terrain, P = build((781, 0), G=16)
    state = cl.init_state(globe, P)
    gen = torch.Generator().manual_seed(4)
    state.T = state.T + 40 * (2 * torch.rand(state.T.shape, generator=gen) - 1)
    noise0 = float((state.T - state.T.mean(-1, keepdim=True)).std())
    for _ in range(60):
        state, diag = cl.step(state, globe, P)
        assert bool(torch.isfinite(state.T).all())
        assert float(state.T.min()) > 150 and float(state.T.max()) < 400
    smooth = cl.run(cl.init_state(globe, P), globe, P, 60)[0]
    assert float((state.T - smooth.T).abs().mean()) < 0.1 * noise0
    assert float(state.vapour.min()) > -1e-3 and float(state.soil.min()) >= 0 and float(state.snow.min()) >= 0
    assert float(state.soil.max()) <= P["rules"].bucket_kg_m2 + 1e-3


def test_states_are_values():
    """step returns a new state and the exchanges rebind fields: earlier states and state_dicts never change."""
    globe, specs, terrain, P = build((781, 0), G=8)
    s0 = cl.reset_water_ledger(cl.run(cl.init_state(globe, P), globe, P, 5)[0], globe)
    sd0 = s0.state_dict()
    s1, _ = cl.step(s0, globe, P)
    sd1 = s1.state_dict()
    land = int(P["land"][0].nonzero()[0, 0])
    cl.take_water(s1, globe, P, [0], [land], [1e6])
    cl.take_water(s1, globe, P, [1], [0], [1e6], store="ocean")
    cl.give_water(s1, globe, P, [0, 1], [land, 3], [5e5, 2e5])
    cl.give_water(s1, globe, P, [1], [land], [2e5], store="soil")
    cl.add_gas(s1, torch.tensor([[0.0, -1.0, 0.8, 0.0]] * 2, dtype=torch.float64))
    assert float(cl.water_ledger(s0, globe).abs().max()) == 0.0
    assert float(cl.water_ledger(s1, globe).abs().max()) < 1e-6
    for name, value in sd0.items():
        same = getattr(s0, name)
        assert torch.equal(value, same) if isinstance(value, torch.Tensor) else value == same, name
    for name in ("soil", "vapour", "ocean", "gas", "gas_external", "water_external"):
        assert not torch.equal(sd1[name], getattr(s1, name)), name    # s1 moved on, its state_dict did not
    assert float(sd1["gas_external"].abs().max()) == 0.0
    # from_state copies too
    s2 = cl.ClimateState.from_state(sd1)
    s2.T += 1.0
    assert not torch.equal(s2.T, sd1["T"])


def test_resume_is_exact():
    globe, specs, terrain, P = build((781, 7), G=8)
    state = cl.run(cl.init_state(globe, P), globe, P, 30)[0]
    sd = state.state_dict()
    straight = cl.run(state, globe, P, 25)[0]
    for _ in range(2):
        resumed = cl.run(cl.ClimateState.from_state(sd), globe, P, 25)[0]
        for f in fields(straight):
            x, y = getattr(straight, f.name), getattr(resumed, f.name)
            assert torch.equal(x, y) if isinstance(x, torch.Tensor) else x == y, f.name


def test_running_mean_runoff_and_surface_wetness():
    globe, specs, terrain, P = build((0,), G=8)
    state = cl.init_state(globe, P)
    seen = []
    for _ in range(3):
        state, diag = cl.step(state, globe, P)
        seen.append(diag["runoff"])
        # a cumulative mean over the first year (then a one-year exponential mean)
        assert torch.allclose(state.runoff_mean, torch.stack(seen).mean(0), rtol=1e-5, atol=1e-9)
    # dry or snow-covered land has no latent part in the moist static energy
    T = torch.full((1, globe.C), 300.0)
    lat_wet, d_wet = cl.latent_temperature(T, P)
    lat_dry, d_dry = cl.latent_temperature(T, P, torch.zeros_like(T))
    assert float(lat_wet.min()) > 10 and float(lat_dry.abs().max()) == 0 and float(d_dry.abs().max()) == 0
    v = torch.full((1, globe.C), 20.0)
    wet = cl.transport(T, v, P["heat"], globe, P, torch.ones_like(T))
    assert torch.equal(cl.transport(T, v, P["heat"], globe, P)[0], wet[0])


def test_ocean_albedo_follows_the_sun_angle():
    assert float(cl.ocean_albedo(torch.tensor(1.0))) == pytest.approx(0.0244, abs=5e-4)    # Briegleb et al. 1986
    assert float(cl.ocean_albedo(torch.tensor(0.3))) == pytest.approx(0.138, abs=2e-3)
    globe, specs, terrain, P = build((781, 0), G=16)
    I, mu = cl.sun_angle(globe, P, -0.5)                              # Earth's vernal equinox
    assert torch.allclose(mu[0], globe.centers[:, 0].clamp_min(0), atol=1e-6)   # locked: cos of the substellar angle
    eq = globe.lat.abs() < 0.05
    assert torch.allclose(mu[1, eq], torch.full_like(mu[1, eq], math.pi / 4), atol=2e-3)   # int cos^2 / int cos
    assert torch.equal(I, cl.insolation(globe, P, -0.5))
    state = cl.init_state(globe, P)
    a = cl.surface_albedo(state, P, mu=mu)
    sea = (~P["land"][1]) & (cl.ice_fraction(state.T, P["rules"])[1] == 0)
    polar = (globe.lat.abs() > 1.2) & (globe.lat.abs() < 1.45)
    assert float(a[1][sea & eq].max()) < 0.06 < float(a[1][sea & polar].min())


def test_determinism():
    globe, specs, terrain, P = build((781, 7), G=8)
    a = cl.run(cl.init_state(globe, P), globe, P, 20)[0]
    b = cl.run(cl.init_state(globe, P), globe, P, 20)[0]
    for f in fields(a):
        x, y = getattr(a, f.name), getattr(b, f.name)
        assert torch.equal(x, y) if isinstance(x, torch.Tensor) else x == y, f.name


# ------------------------------------------------------------------------------------------- the gates
def test_gate_locked_planets_reach_the_chain_temperature(locked):
    globe, specs, P, state, report = locked
    assert max(abs(e) for e in report["error_k"]) < 0.5
    state, mean_t, _ = cl.run(state, globe, P, 60)
    for w, s in enumerate(specs):
        assert abs(float(mean_t[w]) - s.t_surface_target_k) < 3.0, (s.seed, float(mean_t[w]))
    clouds = state.cloud_albedo
    assert bool((clouds > 0).all()) and bool((clouds < P["rules"].cloud_max).all())


def test_gate_rotating_planets_reach_the_chain_temperature(rotating):
    globe, specs, P, state, report = rotating
    years = [int(round(float(y))) for y in P["year_days"]]
    sums = torch.zeros(len(specs), dtype=torch.float64)
    north = globe.lat > 1.0
    polar = []
    for d in range(max(years)):
        state, diag = cl.step(state, globe, P)
        sums += diag["mean_t"] * torch.tensor([float(d < y) for y in years], dtype=torch.float64)
        polar.append(float(state.T[0, north].mean()))
    for w, s in enumerate(specs):
        mean_t = float(sums[w]) / years[w]
        assert abs(mean_t - s.t_surface_target_k) < 3.0, (s.seed, mean_t)
    # seasons: Earth's north polar cap swings by tens of K over its year
    assert max(polar[:years[0]]) - min(polar[:years[0]]) > 15.0


def test_gate_stable_for_1000_days_and_water_ledger_closes(locked):
    globe, specs, P, state, report = locked
    assert float(cl.water_ledger(state, globe).abs().max()) < 1e-3
    worst = 0.0
    for chunk in range(10):
        state, mean_t, _ = cl.run(state, globe, P, 100)
        assert bool(torch.isfinite(state.T).all()) and bool(torch.isfinite(state.vapour).all())
        assert float(state.T.min()) > 180 and float(state.T.max()) < 360
        worst = max(worst, float((mean_t - P["t_target64"]).abs().max()))
        err = cl.water_ledger(state, globe)
        assert float(err.abs().max()) < 1e-3, err          # kg/m^2 of about 1e5 kg/m^2 of water
    assert worst < 3.0
    assert state.day >= 1000


def test_gate_rotating_stable_for_1000_days_with_balanced_carbonate_cycle(rotating):
    """Earth and synthetic 7 for 1000 days: stable, the water ledger closes, snow stays bounded, Earth weathers what
    it outgasses (within a factor 2: runoff0 is the model's own Earth's) and its equator is warmer than both polar
    caps (by about 25-30 K in this model; observed 40-60 K, see the module's report)."""
    globe, specs, P, state, report = rotating
    state = cl.reset_water_ledger(state, globe)
    r = P["rules"]
    year = int(round(float(P["year_days"][0])))
    weathered = torch.zeros(len(specs), dtype=torch.float64)
    outgassed = torch.zeros_like(weathered)
    t_sum = torch.zeros(globe.C, dtype=torch.float64)
    for day in range(1000):
        state, diag = cl.step(state, globe, P)
        if day >= 1000 - year:                                        # Earth's last whole year
            weathered += diag["weathering"]
            outgassed += diag["outgassing"]
            t_sum += state.T[0].double()
        if day % 100 == 99:
            assert bool(torch.isfinite(state.T).all()) and bool(torch.isfinite(state.vapour).all())
            assert float(state.T.min()) > 180 and float(state.T.max()) < 360
            assert float(cl.water_ledger(state, globe).abs().max()) < 1e-3
            assert float(state.snow.max()) <= r.glacier_kg_m2 + 1e-3 and float(state.soil.min()) >= 0
    ratio = float(weathered[0] / outgassed[0])
    assert 0.5 < ratio < 2.0, ratio
    t_year = t_sum / year

    def band(lo, hi):
        sel = (globe.lat >= math.radians(lo)) & (globe.lat < math.radians(hi))
        return float((t_year[sel] * globe.area64[sel]).sum() / globe.area64[sel].sum())

    equator = band(-15, 15)
    assert equator - band(60, 91) > 20 and equator - band(-91, -60) > 20, (equator, band(60, 91), band(-91, -60))


def test_locked_climate_pattern(locked):
    globe, specs, P, state, report = locked
    state, diag = cl.step(state, globe, P)
    x = globe.centers[:, 0]
    day, night = x > 0.9, x < -0.5
    t_day, t_night = state.T[:, day].mean(-1), state.T[:, night].mean(-1)
    assert bool((t_day > t_night + 10).all())
    assert bool((t_day < 330).all()) and bool((t_night > 200).all())
    assert float(diag["surface_sw"][:, x <= 0].abs().max()) == 0.0
    # water cycles: global evaporation and precipitation are a few mm per day and balance over time
    e, p = _mean(globe, diag["evaporation"]), _mean(globe, diag["precipitation"])
    assert bool(((e > 0.3) & (e < 8)).all()) and bool(((p > 0.3) & (p < 8)).all())


def test_water_exchange_with_other_modules(locked):
    globe, specs, P, state, report = locked
    state = state.clone()
    state = cl.reset_water_ledger(state, globe)
    land = P["land"][0].nonzero()[:, 0]
    cells = land[:3]
    got = cl.take_water(state, globe, P, [0, 0, 0, 0], torch.cat((cells, cells[:1])), [5.0, 1e12, 2.0, 3.0])
    assert float(got[1]) < 1e12 and float(state.soil.min()) >= 0
    cl.take_water(state, globe, P, [1], [0], [7.0], store="ocean")
    cl.give_water(state, globe, P, [0, 2], [cells[0], land[0]], [4.0, 1.0])
    cl.give_water(state, globe, P, [3], [land[0]], [9.0], store="soil")
    assert float(cl.water_ledger(state, globe).abs().max()) < 1e-6
    state, _ = cl.step(state, globe, P)
    assert float(cl.water_ledger(state, globe).abs().max()) < 1e-3


def test_carbonate_silicate_cycle_is_slow_and_booked(locked):
    globe, specs, P, state, report = locked
    co2 = state.gas[:, cl.I_CO2].clone()
    out0, wea0 = state.co2_outgassed.clone(), state.co2_weathered.clone()
    st, _, _ = cl.run(state, globe, P, 365)
    change = st.gas[:, cl.I_CO2] - co2
    booked = (st.co2_outgassed - out0) - (st.co2_weathered - wea0)
    assert torch.allclose(change, booked, rtol=1e-9, atol=1e-9)
    assert float((change.abs() / co2).max()) < 1e-4          # tiny at sim time scales (spec 2.6)
    assert bool((st.co2_outgassed > out0).all())


def test_co2_doubling_warms_more_than_planck_only():
    globe, specs, terrain, P = build((781, 781), G=12)
    state = cl.init_state(globe, P)
    state, _ = cl.spin_up(state, globe, P, slow_chunks=1)
    state.gas[1, cl.I_CO2] *= 2
    for _ in range(4):
        state, mean_t, _ = cl.run(state, globe, P, 60, accelerate=True)
    warming = float(mean_t[1] - mean_t[0])
    planck_only = specs[0].t_surface_target_k * 0.75 * specs[0].tau_per_ln_co2 * math.log(2) / (
        4 * (1 + 0.75 * specs[0].greenhouse_tau))
    assert 1.2 * planck_only < warming < 4.5, (warming, planck_only)
