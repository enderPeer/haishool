"""Tests of the planet biosphere (PLANET-SPEC 2.7 and the biosphere gates of section 4)."""
from __future__ import annotations

import math
from dataclasses import fields

import pytest
import torch

from haishool.life9.planet import biosphere as bs, chain, climate as cl, formation as F, globe as gb, materials

R_HAB = 1.0e4
TAGS = {"chain", "derived", "reference", "new_rule"}


def build(seeds, G=16, terrain_seed=1):
    """Globe, specs, terrain and climate parameters (seed 0 is Earth, >= 100 world7, else synthetic)."""
    specs = [F.build_planet(chain.earth_inputs(), 0) if s == 0 else
             F.build_planet(chain.chain_inputs(s) if s >= 100 else chain.synthetic_inputs(s), s) for s in seeds]
    globe = gb.Globe(G)
    water = torch.stack([gb.habitat_water_volume(s.ocean_mass_kg, s.radius_m, s.gravity_m_s2, s.relief_m, R_HAB)
                         for s in specs])
    relief = torch.tensor([s.relief_m for s in specs])
    terrain = gb.make_terrain(globe, len(specs), torch.Generator().manual_seed(terrain_seed), relief, water, R_HAB)
    return globe, specs, terrain, cl.make_params(specs, globe, terrain, R_HAB)

#: 781 (locked) and Earth (rotating, seed 0)
SEEDS = (781, 0)


@pytest.fixture(scope="module")
def world():
    globe, specs, terrain, P = build(SEEDS)
    clim, _ = cl.spin_up(cl.init_state(globe, P), globe, P, slow_chunks=1)
    B = bs.make_params(P)
    return globe, specs, P, B, clim


@pytest.fixture(scope="module")
def grown(world):
    globe, specs, P, B, clim = world
    bio = bs.init_state(globe, P, B, clim)
    bio, clim, (cdiag, bdiag) = bs.run(bio, clim, globe, P, B, 240)
    return bio, clim, cdiag, bdiag


def _mean(globe, x):
    return (x.double() * globe.area64).sum(-1) / (4 * math.pi)


# ------------------------------------------------------------------------------------------- rules
def test_rules_have_provenance():
    for f in fields(bs.BioRules):
        tag, note = bs.PROVENANCE[f.name]
        assert note and all(part in TAGS for part in tag.split("+")), f.name


def test_factors():
    r = bs.BioRules()
    T = torch.tensor([250.0, 268.0, 283.0, 298.0, 310.0, 318.0, 330.0])
    f = bs.temperature_factor(T, r)
    assert f[0] == 0 and f[1] == 0 and f[5] == 0 and f[6] == 0
    assert float(f[3]) == pytest.approx(1.0, abs=1e-6)
    assert 0 < float(f[2]) < 1 and 0 < float(f[4]) < 1
    assert float(bs.co2_factor(torch.tensor(28.0), r)) == pytest.approx(1.0, rel=1e-6)
    assert float(bs.co2_factor(torch.tensor(1e4), r)) == pytest.approx(2.07, abs=0.01)   # formation's cap note


def test_respiration_gives_npp_over_gpp_in_steady_state():
    """r_N is derived so that a plant in steady state at 288 K has NPP / GPP = 0.47 (Waring et al. 1998)."""
    globe, specs, terrain, P = build((781,), G=4)
    B = bs.make_params(P)
    r = B["rules"]
    npp = 1.0                                                   # kg C/m^2/yr
    leaf = (1 - r.wood_allocation) * npp / r.leaf_turnover_yr
    wood = r.wood_allocation * npp / r.wood_turnover_yr
    rm = B["r_n"] * (leaf / r.cn_leaf + wood / r.cn_wood)       # kg C/m^2/yr at 288 K
    assert npp / (npp + rm) == pytest.approx(r.npp_gpp, rel=1e-12)
    assert 10 < B["r_n"] < 60               # LPJ-like order: 0.066 g C/g N/d at 10 C is 34/yr at 288 K


# ------------------------------------------------------------------------------------------- the gates
def test_gate_npp_zero_on_the_night_side_of_a_locked_planet(world, grown):
    globe, specs, P, B, clim0 = world
    bio, clim, cdiag, bdiag = grown
    assert bool(P["locked"][0]) and not bool(P["locked"][1])
    night = globe.centers[:, 0] <= 0
    land_night = P["land"][0] & night
    assert int(land_night.sum()) > 10
    for name in ("gpp", "npp", "respiration"):
        assert float(bdiag[name][0, land_night].abs().max()) == 0.0, name
    assert float(bio.plant_c[0, land_night].abs().max()) == 0.0
    assert float(bdiag["phyto_production"][0, night].abs().max()) == 0.0
    # and the day side produces
    assert float(bdiag["gpp"][0, P["land"][0] & ~night].max()) > 0


def test_gate_npp_positive_where_warm_wet_and_lit(world):
    globe, specs, P, B, clim = world
    bio = bs.init_state(globe, P, B, clim)
    clim1, cdiag = cl.step(clim, globe, P, bs.cover(bio, B))
    bio1, clim2, bdiag = bs.step(bio, clim1, globe, P, B, cdiag)
    # light, water (half the bucket) and warmth (280-300 K: above that a seedling's maintenance, Q10 = 2, outgrows
    # its photosynthesis on a dim day, as it should)
    good = P["land"] & (cdiag["surface_sw"] > 150) & (clim1.T > 280) & (clim1.T < 300) & (clim1.soil > 75)
    for w in range(len(specs)):
        assert int(good[w].sum()) > 10, w
        assert float(bdiag["npp"][w, good[w]].min()) > 0, w
    # light is what drives it: in the same cells without light there is no production
    dark = dict(cdiag, surface_sw=torch.zeros_like(cdiag["surface_sw"]))
    _, _, dd = bs.step(bio, clim1, globe, P, B, dark)
    assert float(dd["gpp"].abs().max()) == 0.0 and float(dd["phyto_production"].abs().max()) == 0.0


def test_vegetation_grows_and_lowers_the_albedo(world, grown):
    globe, specs, P, B, clim0 = world
    bio, clim, cdiag, bdiag = grown
    assert bool((bs.organic_carbon(bio) > 0.02).all())
    assert bool((bdiag["mean_npp"] > 0).all())
    a_bare = cl.surface_albedo(clim, P)
    a_veg = cl.surface_albedo(clim, P, bs.cover(bio, B))
    green = bs.cover(bio, B) > 0.3
    assert bool(green.any()) and bool((a_veg[green] < a_bare[green]).all())
    # land NPP and ocean production of an Earth-like planet are of Earth's order (kg C per m^2 of surface per yr)
    year = 365.25
    assert 0.02 < float(bdiag["mean_npp"][1]) * year < 0.5
    assert 0.02 < float(bdiag["mean_phyto_production"][1]) * year < 0.3


def test_gate_carbon_and_oxygen_ledgers_close(world):
    globe, specs, P, B, clim = world
    clim = clim.clone()
    bio = bs.init_state(globe, P, B, clim)
    gen = torch.Generator().manual_seed(9)
    land = [P["land"][w].nonzero()[:, 0] for w in range(len(specs))]
    for day in range(300):
        clim, cdiag = cl.step(clim, globe, P, bs.cover(bio, B))
        bio, clim, bdiag = bs.step(bio, clim, globe, P, B, cdiag)
        if day % 3 == 0:
            # other modules: eat, collect wood, return corpses to litter, breathe and burn (O2 in, CO2 out)
            w = torch.randint(0, len(specs), (16,), generator=gen)
            cell = torch.stack([land[int(i)][torch.randint(0, len(land[int(i)]), (1,), generator=gen)][0] for i in w])
            bs.harvest(bio, w, cell, 1e5 * torch.rand(16, generator=gen))
            bs.collect_wood(bio, w, cell, 1e5 * torch.rand(16, generator=gen))
            bs.add_litter(bio, w[:4], cell[:4], 5e4 * torch.rand(4, generator=gen))
            o2 = 1e-3 * torch.rand(len(specs), generator=gen, dtype=torch.float64)
            delta = torch.zeros(len(specs), 4, dtype=torch.float64)
            delta[:, cl.I_O2], delta[:, cl.I_CO2] = -o2, 0.85 * o2
            cl.add_gas(clim, delta)
        if day % 50 == 49:
            c_err, o_err = bs.carbon_ledger(bio, clim), bs.oxygen_ledger(bio, clim)
            assert float(c_err.abs().max()) < 1e-6, c_err              # kg C/m^2 (the CO2 column holds 2-190)
            assert float(o_err.abs().max()) < 1e-4, o_err              # mol/m^2 (the O2 column holds 1e3-1e4)
    assert float(cl.water_ledger(clim, globe).abs().max()) < 1e-3
    assert bool((bio.c_harvested > 0).all()) and bool((bio.c_wood_collected >= 0).all())
    for name in ("plant_c", "wood_c", "litter_c", "nutrient", "phyto_c"):
        assert float(getattr(bio, name).min()) >= 0, name


def test_gas_exchange_is_one_to_one(world):
    globe, specs, P, B, clim = world
    bio = bs.init_state(globe, P, B, clim)
    bio, clim, _ = bs.run(bio, clim, globe, P, B, 30)
    clim1, cdiag = cl.step(clim, globe, P, bs.cover(bio, B))
    bio2, clim2, bdiag = bs.step(bio, clim1, globe, P, B, cdiag)
    d_o2 = clim2.gas[:, cl.I_O2] - clim1.gas[:, cl.I_O2]
    d_co2 = clim2.gas[:, cl.I_CO2] - clim1.gas[:, cl.I_CO2]
    assert torch.allclose(d_o2, bdiag["exchange_mol"], rtol=1e-9, atol=1e-12)
    assert torch.allclose(d_co2, -bdiag["exchange_mol"], rtol=1e-9, atol=1e-12)
    assert torch.equal(clim2.gas[:, cl.I_N2], clim1.gas[:, cl.I_N2])
    # organic carbon gained = O2 gained - carbon buried (mol)
    d_org = (bs.organic_carbon(bio2) - bs.organic_carbon(bio)) / bs.M_C
    buried = (bio2.c_buried - bio.c_buried) / bs.M_C
    assert torch.allclose(d_o2, d_org + buried, rtol=1e-5, atol=1e-7)


def test_nitrogen_is_conserved_in_each_cell(world):
    globe, specs, P, B, clim = world
    r = B["rules"]
    bio = bs.init_state(globe, P, B, clim)

    def total(b):
        return b.nutrient.double() + b.plant_c.double() / r.cn_leaf + b.wood_c.double() / r.cn_wood

    n0 = total(bio)
    bio, clim, _ = bs.run(bio, clim, globe, P, B, 120)
    assert float((total(bio) - n0).abs().max()) < 1e-6        # kg N/m^2 of 0.03 (float32 per step)


# ------------------------------------------------------------------------------------------- scatters
def test_harvest_never_goes_negative(world, grown):
    globe, specs, P, B, clim0 = world
    bio = grown[0].clone()
    cell, other = (int(i) for i in bio.plant_c[0].topk(2).indices)
    stock_kg_c = float(bio.plant_c[0, cell]) * float(bio.cell_m2[cell])
    other_kg_c = float(bio.plant_c[0, other]) * float(bio.cell_m2[other])
    c0 = bio.c_harvested.clone()
    cf = bs.BioRules().plant_carbon_fraction
    want = torch.tensor([stock_kg_c / cf, 3 * stock_kg_c / cf, 50.0])     # dry kg: 4x the stock, plus a meal
    got = bs.harvest(bio, [0, 0, 0], [cell, cell, other], want)
    assert float(bio.plant_c[0, cell]) == 0.0
    assert float(got[:2].sum()) * cf == pytest.approx(stock_kg_c, rel=1e-6)
    assert float(got[1] / got[0]) == pytest.approx(3.0, rel=1e-6)        # shared in proportion
    # a meal from a cell of about 1e6 kg: given what left the float32 pool, 50 kg up to its resolution
    other_after = float(bio.plant_c[0, other]) * float(bio.cell_m2[other])
    assert float(got[2]) == pytest.approx(50.0, rel=0.02)
    assert (other_kg_c - other_after) == pytest.approx(float(got[2]) * cf, rel=1e-6)
    taken = float((bio.c_harvested - c0)[0]) * 4 * math.pi / float(bio.area64.sum()) * float(bio.cell_m2.sum())
    # float64: what the eaters get is the booked carbon to float64 rounding (no float32 cast)
    assert got.dtype == torch.float64
    assert taken == pytest.approx(float(got.sum()) * cf, rel=1e-12)
    assert float(bio.plant_c.min()) >= 0
    # nothing left: nothing given
    assert float(bs.harvest(bio, [0], [cell], [5.0])[0]) == 0.0


def test_collect_wood_books_the_materials_carbon(world, grown):
    globe, specs, P, B, clim0 = world
    bio = grown[0].clone()
    cell = int(bio.wood_c[0].argmax())
    cf = materials.species_elements("wood")["C"]
    before = float(bio.wood_c[0, cell]) * float(bio.cell_m2[cell])
    c0 = bio.c_wood_collected.clone()
    got = bs.collect_wood(bio, [0], [cell], [20.0])
    after = float(bio.wood_c[0, cell]) * float(bio.cell_m2[cell])
    # what is given is what left the float32 pool: 20 kg up to the pool's resolution (a cell holds about 1e6 kg)
    assert float(got[0]) == pytest.approx(20.0, rel=0.02)
    assert before - after == pytest.approx(float(got[0]) * cf, rel=1e-6)
    booked = float((bio.c_wood_collected - c0)[0]) * 4 * math.pi / float(bio.area64.sum()) * float(bio.cell_m2.sum())
    assert booked == pytest.approx(float(got[0]) * cf, rel=1e-6)
    big = bs.collect_wood(bio, [0], [cell], [1e12])
    assert float(bio.wood_c[0, cell]) == 0.0 and float(big[0]) * cf == pytest.approx(after, rel=1e-4)




def test_spin_up_holds_the_air_and_keeps_the_water_ledger(world):
    globe, specs, P, B, clim = world
    bio = bs.init_state(globe, P, B, clim)
    gas = clim.gas.clone()
    water_before = cl.water_ledger(clim, globe)
    bio, clim2, report = bs.spin_up(bio, clim, globe, P, B, days=60)
    assert float((clim2.gas - gas)[:, cl.I_O2].abs().max()) == 0.0     # O2 is only touched by life, which is held
    assert report["days"] == 60 and len(report["segments"]) == 2
    # the biosphere moves no water: the water ledger is not re-based, and it still closes
    assert torch.equal(clim2.water0, clim.water0) and torch.equal(clim2.water_external, clim.water_external)
    assert float((cl.water_ledger(clim2, globe) - water_before).abs().max()) < 1e-4
    # the re-based carbon and oxygen ledgers close over the following days, with the air free
    bio, clim3, _ = bs.run(bio, clim2, globe, P, B, 30)
    assert float(bs.carbon_ledger(bio, clim3).abs().max()) < 1e-6
    assert float(bs.oxygen_ledger(bio, clim3).abs().max()) < 1e-4
    assert float((clim3.gas - clim2.gas)[:, cl.I_O2].abs().max()) > 0


@pytest.fixture(scope="module")
def spun(world):
    globe, specs, P, B, clim = world
    bio, clim, report = bs.spin_up(bs.init_state(globe, P, B, clim), clim, globe, P, B)
    return bio, clim, report


def test_steady_state_after_spin_up_keeps_the_air(world, spun):
    """The spin-up puts wood and litter at their steady state: afterwards the air's CO2 drifts by less than 1 % per
    year on Earth (it was 15 % per year with the slow pools still filling) and the stand neither grows nor shrinks
    much; 781's huge CO2 column hardly moves."""
    globe, specs, P, B, clim0 = world
    bio, clim, report = spun
    assert all(s["wood_after"][1] > 3 * s["wood_before"][1] for s in report["segments"][:1])   # the first jump fills
    co2 = clim.gas[:, cl.I_CO2].clone()
    org = bs.organic_carbon(bio)
    bio2, clim2, _ = bs.run(bio, clim, globe, P, B, 730)
    change = (clim2.gas[:, cl.I_CO2] / co2 - 1).abs()
    assert float(change[1]) < 0.02, float(change[1])                 # Earth: < 1 % per year over two years
    assert float(change[0]) < 1e-4, float(change[0])                 # 781
    assert float(((bs.organic_carbon(bio2) - org) / org).abs().max()) < 0.06
    assert float(bs.carbon_ledger(bio2, clim2).abs().max()) < 1e-6


def test_harvest_to_zero_regrows(world, spun):
    """A lit, wet land cell eaten bare regrows: the wood's reserves resprout the soft tissue, and the eaters' dung
    gives most of its nitrogen back; a cell stripped of wood too regrows from the seed rain."""
    globe, specs, P, B, clim = world
    r = B["rules"]
    bio, clim, _ = spun
    bio, clim = bio.clone(), clim.clone()
    day = globe.centers[:, 0] > 0.3
    cells = (P["land"][0] & day & (bio.plant_c[0] > 0.05) & (clim.soil[0] > 30)).nonzero()[:, 0][:8]
    assert len(cells) == 8
    bare, stripped = cells[:6], cells[6:]
    before = bio.plant_c[0, bare].clone()
    n0 = bs.nitrogen_total(bio, r)[0, bare]
    removed = before.double()
    zero = torch.zeros_like(cells)
    got = bs.harvest(bio, zero, cells, torch.full((8,), 1e12))
    bs.collect_wood(bio, zero[:2], stripped, torch.full((2,), 1e12))
    assert got.dtype == torch.float64
    assert float(bio.plant_c[0, cells].abs().max()) == 0.0 and float(bio.wood_c[0, stripped].abs().max()) == 0.0
    # 85 % of the eaten nitrogen goes back to the cell's mineral pool
    n_lost = n0 - bs.nitrogen_total(bio, r)[0, bare]
    assert torch.allclose(n_lost, (1 - r.excreted_n_share) * removed / r.cn_leaf, rtol=1e-4, atol=1e-9)
    bio, clim, (cd, bd) = bs.run(bio, clim, globe, P, B, 1)
    day_rain = r.seed_floor_kg_m2 * (1 - math.exp(-1 / 365.25))
    assert 0 < float(bio.plant_c[0, stripped].min())                                  # seed rain, day one ...
    assert float(bio.plant_c[0, stripped].max()) < 3 * day_rain                         # ... at a year's rate
    assert float(bd["resprout"][0, bare].min()) > 0
    bio, clim, _ = bs.run(bio, clim, globe, P, B, 29)
    seedlings = bio.plant_c[0, stripped].clone()
    bio, clim, _ = bs.run(bio, clim, globe, P, B, 30)
    assert float((bio.plant_c[0, bare] / before).min()) > 0.3                          # resprouted
    # from seed, without reserves, a stand grows back slowly (by photosynthesis alone), but it grows
    assert bool((bio.plant_c[0, stripped] > seedlings).all())
    assert float(bs.carbon_ledger(bio, clim).abs().max()) < 1e-6


def test_scatters_and_states_are_values(world, grown):
    globe, specs, P, B, clim0 = world
    bio0 = grown[0]
    sd = bio0.state_dict()
    bio1 = bs.BioState.from_state(sd)
    cell = int(bio1.plant_c[0].argmax())
    bs.harvest(bio1, [0], [cell], [1e9])
    bs.collect_wood(bio1, [0], [cell], [1e9])
    bs.add_litter(bio1, [0], [cell], [1e3])
    assert float(bio1.plant_c[0, cell]) == 0.0
    for name, value in sd.items():
        same = getattr(bio0, name)
        assert torch.equal(value, same) if isinstance(value, torch.Tensor) else value == same, name


def test_ocean_litter_decays_and_production_is_capped(world):
    globe, specs, P, B, clim = world
    r = B["rules"]
    bio = bs.init_state(globe, P, B, clim)
    sea = int((~P["land"][1]).nonzero()[0, 0])
    bs.add_litter(bio, [1], [sea], [1e5])                            # a swimmer's corpse on Earth's ocean
    before = float(bio.litter_c[1, sea])
    bio, clim, _ = bs.run(bio, clim, globe, P, B, 30)
    assert 0 < float(bio.litter_c[1, sea]) < before * (1 - 0.5 * 30 * r.decomposition_yr / 365.25)
    T = torch.tensor([280.0, 291.15, 293.15, 300.0, 315.0])
    f = bs.ocean_temperature_factor(T, r)
    assert float(f[1]) == pytest.approx(1.0) and float(f[0]) < 1
    assert float(f[3]) == float(f[2]) == float(f[4]) == pytest.approx(r.eppley_base ** 2, rel=1e-6)


def test_cover_takes_the_world_rules(world, grown):
    globe, specs, P, B, clim = world
    bio = grown[0]
    assert torch.equal(bs.cover(bio, B), bs.cover(bio, B["rules"]))
    other = bs.cover(bio, bs.BioRules(fapar_kg_c=0.2))
    assert float((other - bs.cover(bio, B)).max()) > 0.05
    with pytest.raises(TypeError):
        bs.cover(bio)


def test_biosphere_determinism_and_resume(world):
    globe, specs, P, B, clim = world
    a = bs.run(bs.init_state(globe, P, B, clim), clim, globe, P, B, 15)
    b = bs.run(bs.init_state(globe, P, B, clim), clim, globe, P, B, 15)
    for f in fields(a[0]):
        x, y = getattr(a[0], f.name), getattr(b[0], f.name)
        assert torch.equal(x, y) if isinstance(x, torch.Tensor) else x == y, f.name
    assert torch.equal(a[1].gas, b[1].gas) and torch.equal(a[1].T, b[1].T)
    # resume: from the state_dicts, twice, exactly as running on
    sd_bio, sd_clim = a[0].state_dict(), a[1].state_dict()
    straight = bs.run(a[0], a[1], globe, P, B, 12)
    for _ in range(2):
        again = bs.run(bs.BioState.from_state(sd_bio), cl.ClimateState.from_state(sd_clim), globe, P, B, 12)
        for k in range(2):
            for f in fields(straight[k]):
                x, y = getattr(straight[k], f.name), getattr(again[k], f.name)
                assert torch.equal(x, y) if isinstance(x, torch.Tensor) else x == y, f.name


def test_seed_rain_is_not_a_daily_food_supply(world, spun):
    """Regression (3 Oct 2026): the seed floor was refilled in full every day, so grazers stripping a planet bare got
    1 g C/m^2 a day of plants that no photosynthesis had made. Strip every plant and all wood every day for 60 days:
    what can be eaten is the seedlings' own growth plus about a year's seed rain pro rata, not 60 floors."""
    globe, specs, P, B, clim = world
    r = B["rules"]
    bio, clim, _ = spun
    bio, clim = bio.clone(), clim.clone()
    W, C = bio.plant_c.shape
    w = torch.arange(W).repeat_interleave(C)
    c = torch.arange(C).repeat(W)
    eaten = torch.zeros(W, dtype=torch.float64)
    for day in range(60):
        h0 = bio.c_harvested.clone()
        bs.harvest(bio, w, c, torch.full((W * C,), 1e15))
        bs.collect_wood(bio, w, c, torch.full((W * C,), 1e15))
        if day:
            eaten += bio.c_harvested - h0
        bio, clim, _ = bs.run(bio, clim, globe, P, B, 1)
    land = (P["land"].double() * globe.area64).sum(-1) / (4 * math.pi)
    old_rule = 59 * r.seed_floor_kg_m2 * land                     # what the daily refill would have handed out
    assert bool((eaten < 0.05 * old_rule).all()), (eaten, old_rule)
