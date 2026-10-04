"""Tests of the v3 patch ecology: runoff and ponds, seed dispersal, plant growth, ledgers, exchanges with bodies,
determinism and the coupling to version 2's climate (PLANET-V3-SPEC sections 1 and 7)."""
from __future__ import annotations

import math

import pytest
import torch

from haishool.life9.planet import chain, climate as cl, formation as F, globe as gb
from haishool.life9.v3 import optics
from haishool.life9.v3 import patch as pt

R_781 = 5.674e6
WET = dict(vapour_kg_m2=200.0)          # a saturated column: no evaporation at 293 K


def _pits(n=32, L=512.0):
    """Two Gaussian pits on a periodic plain: a shallow one at (L/4, L/2) and a deep one at (3L/4, L/2)."""
    c = (torch.arange(n, dtype=torch.float64) + 0.5) * L / n
    x, y = torch.meshgrid(c, c, indexing="ij")

    def d2(x0, y0):
        dx = pt.wrap(x - x0, L)
        dy = pt.wrap(y - y0, L)
        return dx ** 2 + dy ** 2

    s2 = 2 * 60.0 ** 2
    z = 100.0 - 6.0 * torch.exp(-d2(L / 4, L / 2) / s2) - 12.0 * torch.exp(-d2(3 * L / 4, L / 2) / s2)
    return pt.geometry_from_elevation(z[None], L, -1e4, R_781)


def level_checks(geom, pond, tol=1e-6):
    """Standing water as physics wants it: only in depressions; 8-adjacent wet cells share one surface; a dry cell
    next to a wet one is not below that surface (water would run into it)."""
    H = geom.hydro
    z = H["z"]
    p = pond.double().reshape(z.shape)
    band = H["band"].reshape(z.shape)
    assert float(p[band < 0].abs().max()) == 0.0
    wet = p > 0
    surf = z + p / pt.RHO_W
    for dx, dy in pt.OFFSETS8:
        wn = torch.roll(wet, (dx, dy), (1, 2))
        sn = torch.roll(surf, (dx, dy), (1, 2))
        zn = torch.roll(z, (dx, dy), (1, 2))
        both = wet & wn
        if bool(both.any()):
            assert float((surf - sn)[both].abs().max()) < tol
        edge = wet & ~wn
        if bool(edge.any()):
            assert float((surf - zn)[edge].max()) < tol


# ------------------------------------------------------------------------------------------------ hydrology
def test_drainage_tables_of_two_pits():
    geom = _pits()
    H = geom.hydro
    n = geom.n
    z = H["z"][0]
    # the deep pit holds the patch's lowest cell: a sink at its own level, so it holds no lake
    low = int(z.reshape(-1).argmin())
    assert bool(H["outlet"][0].reshape(-1)[low]) and int(H["outlet"].sum()) == 1
    assert int(H["band"].reshape(1, n, n)[0, 3 * n // 4, n // 2]) == -1
    # the shallow pit is the one depression; it spills into the exits over the saddle between the pits
    assert H["leaves"] == 1 and H["nodes"] == 1
    assert int(H["tgt"][0]) >= H["leaves"]
    saddle = float(z[n // 2 - 2:n // 2 + 2, n // 2].max())
    assert float(H["spill"][0]) <= saddle + 1e-9
    spill = float(H["spill"][0])
    inside = H["band"].reshape(n, n) == 0
    assert bool((z[inside] < spill).all())
    assert float(H["cap"][0]) == pytest.approx(float((spill - z[inside]).sum()) * pt.RHO_W, rel=1e-9)
    assert float(H["spill_m"][0, n // 4, n // 2]) == pytest.approx(spill, abs=1e-4)
    assert float(H["lake_share"][0]) == pytest.approx(float(inside.double().mean()))


def test_runoff_fills_the_lowest_cells_and_conserves_water():
    geom = _pits()
    H = geom.hydro
    n = geom.n
    pond = torch.zeros(1, n, n, dtype=torch.float64)
    total_in = left = 0.0
    cap = float(H["cap"][0])
    lake = H["band"] == 0
    v_before = 0.0
    for day in range(200):
        surplus = torch.full((1, n, n), 5.0, dtype=torch.float64)
        total_in += float(surplus.sum())
        pond, out = pt.route(surplus, pond, geom)
        left += float(out.sum())
        assert float(pond.sum()) + left == pytest.approx(total_in, rel=1e-12)     # nothing lost or made
        level_checks(geom, pond)
        v = float(pond.reshape(-1)[lake].sum())
        assert v_before - 1e-6 <= v <= cap * (1 + 1e-12)                     # fills, never beyond its sill
        v_before = v
    assert left > 0 and v == pytest.approx(cap, rel=1e-9)
    # small amounts sit in the very bottom of the pit
    pond, out = pt.route(torch.full((1, n, n), 0.01, dtype=torch.float64), torch.zeros(1, n, n, dtype=torch.float64),
                         geom)
    wet = pond[0] > 0
    z = H["z"][0]
    m = H["band"].reshape(n, n) == 0
    assert 0 < int(wet.sum()) < 10
    assert float(z[wet].max()) < float(z[m].min()) + 0.05


def test_ponds_form_where_water_collects_and_the_water_ledger_closes():
    geom = _pits()
    st = pt.init_state(geom)
    f = pt.constant_forcing(geom, t_air_k=293.0, precip_kg_m2=8.0, vapour_kg_m2=20.0)
    st, d = pt.run(st, geom, f, 60)
    assert float(st.pond.sum()) > 0
    # after the day's levelling the ponds wet the soil under newly flooded cells, evaporate and seep: level within
    # the bucket's 0.15 m (the next day's levelling makes it flat again)
    level_checks(geom, st.pond.double(), tol=0.15)
    under = st.pond > 0
    assert torch.allclose(st.soil[under], torch.full_like(st.soil[under], 150.0))
    err = pt.water_ledger(st, geom)
    assert float(err.abs().max()) < 1e-9 * float(pt.water_stores(st, geom).max())
    assert float(st.w_out.max()) > 0 and float(st.w_evap.max()) > 0
    # seepage: 2 mm a day under standing water, booked as outflow
    assert float(d["seepage"][under].max()) == pytest.approx(2.0, rel=1e-6)
    assert float(d["seepage"][~under].max()) <= 2.0
    # the booked evaporation is the physical field up to the float32 cast of the stores (an independent check)
    stores = st.soil.double() + st.pond.double() + st.snow.double()
    bound = float((stores.abs() * 2 ** -24).sum()) * geom.cell_m2 + 1e-9
    assert abs(float(d["evap_kg"][0] - d["evap_physical_kg"][0])) <= bound
    # cold: snow builds up on the ground and nothing runs off
    st2 = pt.init_state(geom)
    st2, _ = pt.run(st2, geom, pt.constant_forcing(geom, t_air_k=250.0, precip_kg_m2=8.0, **WET), 20)
    assert torch.allclose(st2.snow, torch.full_like(st2.snow, 160.0))
    assert float(st2.pond.sum()) == 0.0 and float(pt.water_ledger(st2, geom).abs().max()) < 1e-6


def test_sea_cells_take_the_runoff():
    n, L = 24, 384.0
    x = (torch.arange(n) + 0.5) * L / n
    z = (10.0 * torch.cos(2 * math.pi * x / L))[:, None].expand(n, n)[None].float()
    geom = pt.geometry_from_elevation(z, L, 0.0, R_781)
    assert bool(geom.sea.any()) and bool((~geom.sea).any())
    st = pt.init_state(geom)
    assert float(st.soil[geom.sea].abs().max()) == 0.0
    st, d = pt.run(st, geom, pt.constant_forcing(geom, precip_kg_m2=20.0, **WET), 30)
    assert float(st.w_out[0]) > 0
    assert float(st.soil[geom.sea].abs().max()) == 0.0 and float(st.pond[geom.sea].abs().max()) == 0.0
    assert float(pt.water_ledger(st, geom).abs().max()) < 1e-9 * float(pt.water_stores(st, geom).max())


# ------------------------------------------------------------------------------------------------ plants
def _barrier(n=48, L=768.0):
    """A plain with a 5 km high (cold, barren) strip at ix 4..9."""
    z = torch.zeros(1, n, n)
    z[:, 4:10, :] = 5000.0
    return pt.geometry_from_elevation(z, L, -1e4, R_781, base_m=torch.zeros(1))


def test_dispersal_spreads_into_fertile_ground_and_never_into_barren_ground():
    geom = _barrier()
    st = pt.init_state(geom)
    st.seed[:, 0:4, :] = 1e-2                         # sown on ix 0..3 only
    st = pt.reset_ledgers(st, geom)
    f = pt.constant_forcing(geom, t_air_k=295.0, precip_kg_m2=3.0, sw_day_w_m2=220.0, vapour_kg_m2=25.0)
    st, d = pt.run(st, geom, f, 300)
    col = st.plant[0].mean(-1)
    # the strip is barren (too cold for any production): seeds land there but nothing grows
    assert float(d["t_k"][0, 6].max()) < 268.0
    assert float(st.seed[0, 4:10].sum()) > 0
    assert float(st.plant[0, 4:10].abs().max()) == 0.0 and float(st.wood[0, 4:10].abs().max()) == 0.0
    # plants spread from ix 0 into the empty fertile ground on the other side (periodic: ix 47, 46, ...)
    assert float(col[47]) > 1e-4 and float(col[46]) > 1e-6
    # and cannot cross the barren strip (the far side, ix 10 on, stays empty; the spread around the periodic
    # boundary reaches only the high ix)
    assert float(st.plant[0, 10:30].abs().max()) == 0.0 and float(st.seed[0, 11:30].abs().max()) == 0.0
    assert float(pt.carbon_ledger(st, geom).abs().max()) < 1e-6 * float(pt.carbon_stores(st, geom).max())
    assert float(pt.nitrogen_ledger(st, geom).abs().max()) < 1e-6 * float(pt.nitrogen_stores(st, geom).max())


def test_spin_up_sows_sparsely_and_plants_grow_where_they_can():
    geom = _barrier(n=32, L=512.0)
    gen = torch.Generator().manual_seed(4)
    rules = pt.PatchRules(sow_share=0.25)
    st = pt.init_state(geom, rules)
    f = pt.constant_forcing(geom, t_air_k=295.0, precip_kg_m2=3.0, sw_day_w_m2=220.0, vapour_kg_m2=25.0)
    st, rep = pt.spin_up(st, geom, f, 120, gen, rules)
    assert rep["sown_kg_c"][0] > 0
    land = st.plant[0] > 0
    assert 0 < float(land.float().mean()) < 1
    assert float(st.plant[0, 4:10].abs().max()) == 0.0
    assert float(pt.carbon_ledger(st, geom).abs().max()) < 1e-6 * float(pt.carbon_stores(st, geom).max())
    # the report's exchange is everything taken from the air, sowing included
    assert float(rep["exchange_mol"][0]) == pytest.approx(rep["c_air_kg"][0] / pt.M_C, rel=1e-12)


def test_seedlings_need_positive_net_production():
    geom = pt.geometry_from_elevation(torch.zeros(3, 8, 8), 128.0, -1e4, R_781)
    st = pt.init_state(geom)
    st.seed = torch.full_like(st.seed, 1e-3)
    # arena 0 good, arena 1 dark, arena 2 dry (no rain, the soil emptied)
    st.soil[2] = 0.0
    st = pt.reset_ledgers(st, geom)
    f = pt.constant_forcing(geom, t_air_k=295.0, precip_kg_m2=torch.tensor([3.0, 3.0, 0.0]),
                            sw_day_w_m2=torch.tensor([220.0, 0.0, 220.0]), vapour_kg_m2=25.0)
    st, d = pt.run(st, geom, f, 20)
    assert float(st.plant[0].min()) > 0
    assert float(st.plant[1].abs().max()) == 0.0 and float(st.plant[2].abs().max()) == 0.0
    assert float(st.seed[1].min()) > 0                                # waiting in the seed bank
    assert float(st.seed[1].max()) < 1e-3                             # and dying slowly


def test_standing_water_drowns_seedlings_and_production():
    """A bowl half full of water: seeds do not germinate under more than 1 cm of it and terrestrial production
    fades with depth; the dry slopes around it grow."""
    n, L = 16, 256.0
    c = (torch.arange(n, dtype=torch.float64) + 0.5) * L / n
    x, y = torch.meshgrid(c, c, indexing="ij")
    r = torch.sqrt((x - L / 2) ** 2 + (y - L / 2) ** 2)
    z = (10.0 * (r / 100.0) ** 2).clamp_max(10.0) + 1e-6 * x
    z = z - 12.0 * torch.exp(-((x - 24.0) ** 2 + (y - 24.0) ** 2) / 200.0)   # the patch's lowest cell, outside the bowl
    geom = pt.geometry_from_elevation(z[None], L, -1e4, R_781)
    assert geom.hydro["nodes"] >= 1
    pond, _ = pt.route(torch.full((1, n, n), 600.0, dtype=torch.float64), torch.zeros(1, n, n, dtype=torch.float64),
                       geom)
    st = pt.init_state(geom)
    st.pond = pond.float()
    st.soil = torch.full_like(st.soil, 150.0)
    st.seed = torch.full_like(st.seed, 1e-3)
    st = pt.reset_ledgers(st, geom)
    deep = st.pond > 500.0
    dry = st.pond == 0
    assert bool(deep.any()) and bool(dry.any())
    f = pt.constant_forcing(geom, t_air_k=295.0, precip_kg_m2=0.0, sw_day_w_m2=220.0, vapour_kg_m2=200.0)
    st, d = pt.run(st, geom, f, 20)
    assert bool((st.pond[deep] > 10.0).all())
    assert float(st.plant[deep].abs().max()) == 0.0 and float(d["germinated"][deep].abs().max()) == 0.0
    assert float(st.plant[dry].min()) > 0
    # production under water: exp(-depth / 0.1 m) of the potential
    assert float(d["potential"][deep].max()) < 1e-2 * float(d["potential"][dry].min())
    assert float(pt.water_ledger(st, geom).abs().max()) < 1e-9 * float(pt.water_stores(st, geom).max())


def test_seedling_relative_growth_rate_is_in_the_herbaceous_range():
    """Sparse young plants under good conditions grow at a relative rate of herbaceous seedlings, 0.05-0.3 per day
    (Grime & Hunt 1975, J. Ecol. 63, 393): fAPAR follows the leaf area."""
    geom = pt.geometry_from_elevation(torch.zeros(1, 8, 8), 128.0, -1e4, R_781)
    rules = pt.PatchRules(seed_share=0.0)
    st = pt.init_state(geom, rules)
    st.plant = torch.full_like(st.plant, 1e-3)                       # 1 g C/m^2
    st.soil = torch.full_like(st.soil, 150.0)
    st = pt.reset_ledgers(st, geom, rules)
    f = pt.constant_forcing(geom, t_air_k=295.0, precip_kg_m2=3.0, sw_day_w_m2=220.0, par_fraction=0.45,
                            vapour_kg_m2=25.0)
    p0 = float(st.plant.double().mean())
    st, _ = pt.run(st, geom, f, 10, rules)
    rgr = math.log(float(st.plant.double().mean()) / p0) / 10
    assert 0.05 < rgr < 0.3
    # Beer's law on the leaf area: the initial slope is k SLA leaf share / 0.47 per kg C
    assert pt.leaf_area_per_kg_c(rules) == pytest.approx(0.5 * 20.0 * 0.5 / 0.47)
    small = torch.tensor([1e-4])
    assert float(pt.fapar(small, rules)) == pytest.approx(1e-4 * pt.leaf_area_per_kg_c(rules), rel=1e-3)
    assert float(pt.fapar(torch.tensor([0.5]), rules)) > 0.9               # a closed canopy (LAI about 5)


# ------------------------------------------------------------------------------------------------ ledgers and bodies
def _two_worlds(n=32, seed=3):
    """2 worlds x 2 patches of fractal terrain (conditioned); arena 3 partly below the sea."""
    gen = torch.Generator().manual_seed(seed)
    z = 50.0 + 8.0 * pt.fractal_detail(4, n, gen, 0.8)
    sea = torch.tensor([-1e4, -1e4, -1e4, 50.0])
    geom = pt.geometry_from_elevation(z, 512.0, sea, R_781, world=torch.tensor([0, 0, 1, 1]), W=2, patches=2,
                                      condition_depth_m=0.4 * 8.0)
    return geom, gen


def test_ledgers_close_over_100_days_with_bodies_taking_and_giving():
    geom, gen = _two_worlds()
    assert bool(geom.sea[3].any()) and not bool(geom.sea[0].any())
    rules = pt.PatchRules(sow_share=0.3)
    st = pt.init_state(geom, rules)
    st = pt.sow(st, geom, gen, rules)
    sown = st.c_pending.clone()
    assert float(sown.min()) > 0
    st.seed = st.seed * 20                            # bigger plants sooner (re-based below)
    st = pt.reset_ledgers(st, geom, rules)
    f = pt.constant_forcing(geom, t_air_k=294.0, precip_kg_m2=4.0, sw_day_w_m2=230.0, vapour_kg_m2=22.0)
    M = 64
    exchanged = torch.zeros(4, dtype=torch.float64)
    n_eaten = torch.zeros(4, dtype=torch.float64)
    for day in range(100):
        st, d = pt.step(st, geom, f, rules)
        exchanged += d["exchange_kg_c"]
        if day == 0:
            # the sown carbon (taken from the air before the ledgers were re-based) reaches the world's air
            # through the first step's exchange
            assert torch.allclose(d["exchange_kg_c"] - st.c_air, sown, rtol=1e-12)
            assert float(st.c_pending.abs().max()) == 0.0
        if day % 10 == 9:
            a = torch.randint(0, 4, (M,), generator=gen)
            cell = torch.randint(0, geom.n ** 2, (M,), generator=gen)
            kg = torch.rand(M, generator=gen, dtype=torch.float64) * 0.05
            got = pt.take_plant(st, geom, a, cell, kg, "soft", rules)
            n_eaten += torch.zeros(4, dtype=torch.float64).index_add_(0, a, pt.plant_nitrogen(got, "soft", rules))
            pt.take_plant(st, geom, a, cell, kg * 0.1, "wood", rules)
            pt.take_plant(st, geom, a, cell, kg * 0.01, "seed", rules)
            pt.take_water(st, geom, a, cell, kg * 20, rules)
            pt.give_water(st, geom, a, cell, kg * 5)
            # the bodies carry the nitrogen and return some of it elsewhere, with faeces and urine
            back = pt.plant_nitrogen(got, "soft", rules) * 0.5
            pt.add_litter(st, geom, a, (cell + 37) % geom.n ** 2, kg * 0.2, kg_n=back * 0.4)
            pt.give_nitrogen(st, geom, a, (cell + 91) % geom.n ** 2, back * 0.6)
    c_err = pt.carbon_ledger(st, geom)
    w_err = pt.water_ledger(st, geom)
    n_err = pt.nitrogen_ledger(st, geom, rules)
    assert float(c_err.abs().max()) < 1e-6 * float(pt.carbon_stores(st, geom).max())
    assert float(w_err.abs().max()) < 1e-9 * float(pt.water_stores(st, geom).max())
    assert float(n_err.abs().max()) < 1e-6 * float(pt.nitrogen_stores(st, geom, rules).max())
    assert float(st.c_taken.min()) > 0 and float(st.c_added.min()) > 0 and float(st.w_given.min()) > 0
    assert float(st.n_added.min()) > 0 and float((st.n_removed - n_eaten).min()) > 0   # wood and seeds too
    assert float(st.sea_water[3]) != 0.0 and float(st.sea_water[0]) == 0.0
    # every carbon the patches took from the air (sowing included) went through the steps' exchanges
    assert torch.allclose(exchanged, st.c_air + sown, rtol=1e-12)
    # the patches' exchange with the air goes to each world's column: CO2 down, O2 up by the same moles
    gd = pt.gas_delta(geom, d["exchange_mol"])
    assert gd.shape == (2, 4)
    assert torch.allclose(gd[:, cl.I_CO2], -gd[:, cl.I_O2])
    want = (d["exchange_mol"][0] + d["exchange_mol"][1]) / (4 * math.pi * R_781 ** 2)
    assert float(gd[0, cl.I_O2]) == pytest.approx(float(want), rel=1e-12)


def test_sown_carbon_is_reported_by_the_next_step():
    geom = pt.geometry_from_elevation(torch.zeros(2, 8, 8), 128.0, -1e4, R_781)
    rules = pt.PatchRules(sow_share=0.5)
    st = pt.init_state(geom, rules)
    st = pt.sow(st, geom, torch.Generator().manual_seed(1), rules)
    f = pt.constant_forcing(geom, sw_day_w_m2=0.0)                     # dark: no photosynthesis that day
    total = torch.zeros(2, dtype=torch.float64)
    for _ in range(3):
        st, d = pt.step(st, geom, f, rules)
        total += d["exchange_kg_c"]
    assert torch.allclose(total, st.c_air, rtol=1e-12, atol=1e-15)
    assert float(st.c_air.min()) > 0                                    # the sown seed came from the air


def test_take_plant_removes_its_nitrogen_and_never_overdraws():
    geom = pt.geometry_from_elevation(torch.zeros(1, 4, 4), 64.0, -1e4, R_781)
    st = pt.init_state(geom)
    st.plant[0, 1, 2] = 0.2                                          # kg C/m^2 in cell 6
    st = pt.reset_ledgers(st, geom)
    n_before = st.nutrient.clone()
    have_dry = 0.2 * geom.cell_m2 / 0.47
    got = pt.take_plant(st, geom, [0, 0], [6, 6], [have_dry, have_dry])       # 2 x what is there
    assert float(got.sum()) == pytest.approx(have_dry, rel=1e-6)
    assert float(got[0]) == pytest.approx(float(got[1]))
    assert float(st.plant.abs().max()) == 0.0
    assert float(st.c_taken[0]) == pytest.approx(0.2 * geom.cell_m2, rel=1e-6)
    assert abs(float(pt.carbon_ledger(st, geom)[0])) < 1e-9
    assert abs(float(pt.nitrogen_ledger(st, geom)[0])) < 1e-9
    # none of the eaten nitrogen comes back under the mouth: it leaves with the body
    assert torch.equal(st.nutrient, n_before)
    assert float(st.n_removed[0]) == pytest.approx(float(pt.plant_nitrogen(got, "soft").sum()), rel=1e-6)
    assert float(st.n_removed[0]) == pytest.approx(0.2 * geom.cell_m2 / 29.0, rel=1e-6)
    # returned where the body excretes
    pt.give_nitrogen(st, geom, [0], [0], [0.01])
    assert float(st.nutrient[0, 0, 0] - n_before[0, 0, 0]) == pytest.approx(0.01 / geom.cell_m2, rel=1e-3)
    assert abs(float(pt.nitrogen_ledger(st, geom)[0])) < 1e-9


def test_drinking_from_ponds_and_from_soil_above_the_bucket():
    geom = pt.geometry_from_elevation(torch.zeros(1, 4, 4), 64.0, -1e4, R_781)
    st = pt.init_state(geom)
    st.pond[0, 0, 1] = 3.0
    st.soil[0, 0, 2] = 170.0                                         # 20 kg/m^2 above the bucket
    st.soil[0, 0, 3] = 150.0                                         # full, none above
    st = pt.reset_ledgers(st, geom)
    fresh_w, there = pt.water_at(st, geom, [0, 0, 0], [1, 2, 3])
    assert fresh_w.tolist() == pytest.approx([3.0, 20.0, 0.0]) and not bool(there.any())
    fresh, salt = pt.take_water(st, geom, [0, 0, 0, 0], [1, 2, 2, 3], [1e6, 3000.0, 3000.0, 5.0])
    assert float(fresh[0]) == pytest.approx(3.0 * geom.cell_m2, rel=1e-6)
    # two mouths on cell 2 share its 20 kg/m^2 above the bucket (5,120 kg for 6,000 asked)
    assert float(fresh[1]) == pytest.approx(10.0 * geom.cell_m2, rel=1e-5)
    assert float(fresh[1]) == pytest.approx(float(fresh[2]))
    assert float(st.soil[0, 0, 2]) == pytest.approx(150.0, abs=1e-4)
    assert float(fresh[3]) == 0.0 and float(st.soil[0, 0, 3]) == 150.0  # the bucket's own water stays in the soil
    assert float(salt.abs().sum()) == 0.0
    assert float(st.w_taken[0]) == pytest.approx(float(fresh.sum()), rel=1e-9)
    assert abs(float(pt.water_ledger(st, geom)[0])) < 1e-6
    # a smaller request takes only what it asks
    st.soil[0, 1, 1] = 160.0
    st = pt.reset_ledgers(st, geom)
    fresh, _ = pt.take_water(st, geom, [0], [5], [256.0])
    assert float(fresh[0]) == pytest.approx(256.0, rel=1e-5)
    assert float(st.soil[0, 1, 1]) == pytest.approx(159.0, abs=1e-4)
    assert abs(float(pt.water_ledger(st, geom)[0])) < 1e-6


# ------------------------------------------------------------------------------------------------ determinism
def _run(seed, days=25):
    geom, gen = _two_worlds(n=24, seed=seed)
    rules = pt.PatchRules(sow_share=0.2)
    st = pt.sow(pt.init_state(geom, rules), geom, gen, rules)
    f = pt.constant_forcing(geom, precip_kg_m2=5.0, vapour_kg_m2=20.0)
    st, _ = pt.run(st, geom, f, days, rules)
    return geom, st, f, rules


def test_determinism_and_exact_resume():
    g1, s1, f, rules = _run(9)
    g2, s2, _, _ = _run(9)
    assert torch.equal(g1.elev, g2.elev)
    for k, v in s1.state_dict().items():
        if isinstance(v, torch.Tensor):
            assert torch.equal(v, getattr(s2, k)), k
    g3, s3, _, _ = _run(10)
    assert not torch.equal(g1.elev, g3.elev)
    # resume from a state_dict (and a stored geometry) gives the same days
    geom_r = pt.PatchGeometry.from_state(g1.state_dict())
    resumed = pt.PatchState.from_state(s1.state_dict())
    a, _ = pt.run(s1, g1, f, 5, rules)
    b, _ = pt.run(resumed, geom_r, f, 5, rules)
    for k, v in a.state_dict().items():
        if isinstance(v, torch.Tensor):
            assert torch.equal(v, getattr(b, k)), k


# ------------------------------------------------------------------------------------------------ coupling
@pytest.fixture(scope="module")
def v2_world():
    specs = [F.build_planet(chain.chain_inputs(781), 781), F.build_planet(chain.earth_inputs(), 0)]
    globe = gb.Globe(6)
    radius = torch.tensor([s.radius_m for s in specs], dtype=torch.float64)
    water = torch.stack([gb.habitat_water_volume(s.ocean_mass_kg, s.radius_m, s.gravity_m_s2, s.relief_m, s.radius_m)
                         for s in specs])
    terrain = gb.make_terrain(globe, 2, torch.Generator().manual_seed(1), torch.tensor([s.relief_m for s in specs]),
                              water, radius)
    P = cl.make_params(specs, globe, terrain, specs[0].radius_m)
    clim = cl.init_state(globe, P)
    clim, cdiag = cl.step(clim, globe, P)
    return specs, globe, terrain, P, clim, cdiag, radius


def test_forcing_from_the_global_climate(v2_world):
    specs, globe, terrain, P, clim, cdiag, radius = v2_world
    rules = pt.PatchRules(cells=16)
    geom = pt.make_geometry(globe, terrain["elevation_m"], terrain["sea_level_m"], terrain["land"], radius,
                            torch.Generator().manual_seed(2), patches=2, rules=rules)
    with pytest.raises(ValueError):
        pt.forcing_from_climate(geom, clim, cdiag, P, day=0, rules=rules)       # the solar day is required
    f = pt.forcing_from_climate(geom, clim, cdiag, P, day=0, specs=specs, rules=rules)
    w, c = geom.world, geom.cell
    assert torch.equal(f.t_air_k, clim.T[w, c]) and torch.equal(f.precip_kg_m2, cdiag["precipitation"][w, c])
    assert torch.allclose(f.vapour_pa(), cl.vapour_pressure(clim, P)[w, c], rtol=1e-5)
    assert f.sun.shape == (4, 4, rules.light_samples, 3)
    assert torch.allclose(f.sun.norm(dim=-1), torch.ones(4, 4, rules.light_samples), atol=1e-5)
    # 781 is locked: its sun stands still through the day
    assert specs[0].tidally_locked and not specs[1].tidally_locked
    assert torch.allclose(f.sun[:2, :1], f.sun[:2], atol=1e-6)
    # Earth rotates with its own solar day: the sun's height changes over the day
    assert float((f.sun[2:, :, :, 2].amax((1, 2)) - f.sun[2:, :, :, 2].amin((1, 2))).min()) > 0.5
    sd = pt.solar_days(specs)
    assert math.isinf(sd[0]) and sd[1] == pytest.approx(specs[1].day_length_s)
    f2 = pt.forcing_from_climate(geom, clim, cdiag, P, day=0, solar_day_s=sd, rules=rules)
    assert torch.equal(f2.sun, f.sun)
    # the diffuse share follows the clearness index
    assert bool(((f.diffuse_share > 0) & (f.diffuse_share <= 1)).all())
    st = pt.init_state(geom, rules)
    st, d = pt.step(st, geom, f, rules)
    assert all(bool(torch.isfinite(v).all()) for v in (st.plant, st.soil, st.pond, d["t_k"], d["day_sw"]))
    # a flat patch at the same cells gets exactly the global cell's flat-ground light over a whole solar day (here a
    # 24 h day; the model Earth's own 19.9 h day spreads it over the sim days, test_patch checks that energy)
    flat = pt.geometry_from_elevation(geom.base_m[:, None, None].expand(4, 16, 16).clone(), rules.patch_m,
                                      geom.sea_level_m, geom.radius_m, world=geom.world, cell=geom.cell, W=2,
                                      patches=2, basis=(geom.east, geom.north, geom.up), rules=rules)
    f24 = pt.forcing_from_climate(geom, clim, cdiag, P, day=0, solar_day_s=pt.DAY_S, rules=rules)
    day_sw, bout_sw = pt.light(flat, f24)
    assert torch.allclose(day_sw.mean((1, 2)), cdiag["surface_sw"][w, c], rtol=1e-5, atol=1e-4)
    # the fine temperature follows the lapse rate g / c_p
    t = pt.fine_temperature(geom, f)
    hi = geom.elev.reshape(4, -1).argmax(-1)
    dz = geom.elev.reshape(4, -1).gather(1, hi[:, None])[:, 0] - geom.base_m
    want = f.t_air_k - P["lapse"][w] * dz
    assert torch.allclose(t.reshape(4, -1).gather(1, hi[:, None])[:, 0], want, atol=1e-3)


def test_ground_reflectance_darkens_with_plants_and_water_and_shows_ice_and_litter():
    geom = pt.geometry_from_elevation(torch.zeros(1, 5, 5), 80.0, -1e4, R_781)
    st = pt.init_state(geom)
    st.plant[0, 0, 0] = 2.0
    st.pond[0, 1, 1] = 50.0
    st.snow[0, 2, 2] = 50.0
    st.litter[0, 3, 3] = 1.0
    rho = pt.ground_reflectance(st, geom)[0]
    bare = float(rho[4, 4])
    assert float(rho[0, 0]) < bare and float(rho[1, 1]) < bare and float(rho[2, 2]) > bare
    assert float(rho[1, 1]) == pytest.approx(optics.vis("ground:water"), abs=1e-6)
    assert float(rho[3, 3]) == pytest.approx(optics.vis("ground:litter"), abs=1e-3)   # a thick litter layer
    # a frozen pond looks like ice, a thawed one like water
    cold = pt.ground_reflectance(st, geom, t_k=torch.full((1, 5, 5), 250.0))[0]
    warm = pt.ground_reflectance(st, geom, t_k=torch.full((1, 5, 5), 290.0))[0]
    assert float(cold[1, 1]) == pytest.approx(optics.vis("ground:ice"), abs=1e-6)
    assert float(warm[1, 1]) == pytest.approx(optics.vis("ground:water"), abs=1e-6)


def test_nitrogen_fixation_follows_growth_and_n_limitation_not_the_starting_stock():
    """Review M7: fixation is bnf_share of the N the day's growth asks for, scaled by how much the mineral pool limits
    it; the starting stock (nutrient_start) plays no part, bare ground fixes nothing, and N-rich ground fixes less."""
    from dataclasses import replace as dc_replace
    geom = pt.geometry_from_elevation(torch.zeros(3, 8, 8), 128.0, -1e4, R_781)
    base = pt.PatchRules()
    other = dc_replace(base, bio=dc_replace(base.bio, nutrient_start_kg_m2=base.bio.nutrient_start_kg_m2 * 10))
    f = pt.constant_forcing(geom, t_air_k=295.0, precip_kg_m2=3.0, sw_day_w_m2=220.0, vapour_kg_m2=25.0)
    out = []
    for rules in (base, other):
        st = pt.init_state(geom, base)
        st.plant = torch.full_like(st.plant, 0.05)
        st.plant[1] = 0.0                                            # arena 1: bare ground
        st.nutrient[2] = 0.3                                         # arena 2: N-rich ground
        st = pt.reset_ledgers(st, geom, rules)
        st, d = pt.step(st, geom, f, rules)
        out.append(d["n_fixed"].clone())
        rel = pt.nitrogen_ledger(st, geom, rules).abs() / pt.nitrogen_stores(st, geom, rules).clamp_min(1e-9)
        assert float(rel.max()) < 1e-6
    assert torch.equal(out[0], out[1])                               # no reference to the starting stock
    fixed = out[0].sum((1, 2))
    assert float(fixed[0]) > 0 and float(fixed[1]) == 0.0 and float(fixed[2]) < 0.5 * float(fixed[0])
