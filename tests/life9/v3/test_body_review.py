"""Tests of the v3 body physics the review asked for: a parent survives its division, the newborn touches without
overlapping, the heat of growth and the respiratory water follow the heat balance, cold muscles and guts are slower,
water exchanged with the patch is neither lost nor made, wear has a measured magnitude and no repair threshold,
locomotion against measured data, the stores, bladder and kidney are genes, shared bites, the bone-mineral ledger,
and an independent heat-transfer reference."""
from __future__ import annotations

import math

import pytest
import torch

from haishool.life9.planet import constants as K
from haishool.life9.v3 import body as B
from haishool.life9.v3 import patch as pt

L = 256.0
BASE = dict(size_kg=0.02, fur_m=1e-3, skin_perm=1e-12, thermo_gain=1e-3, setpoint_k=310.0, enz_plant=0.02,
            enz_meat=0.02, repair=0.5, muscle=100.0, muscle_frac=0.3, offspring_share=0.5, eye=1e-4, ear=1e-4,
            voice=1e-4, fat_store=0.3, bladder=0.01, kidney=0.005, air_l_kg=0.2,
            o2_carrier=0.0)


def make(n=1, *, A=1, N=None, T0=300.0, seed=0, **genes):
    """n bodies per arena with the given genes (BASE otherwise), at their size, a founder's fat, normal water."""
    g = torch.Generator().manual_seed(seed)
    N = n if N is None else N
    b = B.found(A, N, n, L, g, in_dim=4, hidden=8, t_k=T0)
    for k, v in {**BASE, **genes}.items():
        b.genome[k][:, :n] = torch.as_tensor(v, dtype=torch.float32)
    b.genome["k"][:, :n] = 8
    size = b.genome["size_kg"][:, :n]
    b.mass_kg[:, :n] = size
    b.frame_kg[:, :n] = size
    b.reserve_j[:, :n] = B.FOUNDER_FAT_PER_LEAN * size * B.FAT_J_KG
    b.water_kg[:, :n] = B.LEAN_WATER * size
    b.body_k[:, :n] = T0
    B.reset_ledgers(b)
    return b


def flat_geom(A=1, n=16):
    return pt.geometry_from_elevation(torch.zeros(A, n, n), L, -5.0, 5.674e6)


def closed(b, keys=("energy", "carbon", "water", "nitrogen", "inert", "salt", "oxygen", "co2"), tol=1e-6):
    errs = B.ledger_errors(b)
    for k in keys:
        rel = (errs[k].abs() / errs[f"{k}_scale"].clamp_min(1e-30)).max()
        assert float(rel) < tol, (k, float(rel))


# ------------------------------------------------------------------------------------------------ division
@pytest.mark.parametrize("share", [0.3, 0.5, 0.9])
def test_a_parent_survives_its_division(share):
    b = make(1, N=2, size_kg=0.05, offspring_share=share, T0=305.0)
    b.brood_j[0, 0] = B.child_cost_j(b)[0, 0]
    out = B.divide(b, torch.Generator().manual_seed(1))
    assert out["born"].tolist() == [1]
    assert B.death_cause(b)[0].tolist() == [0, 0]
    assert float(b.frame_kg[0, 0]) == pytest.approx(0.05 * (1 - share), rel=1e-5)
    B.metabolism(b, B.constant_env(t_air_k=305.0, rel_humidity=1.0))
    assert B.death_cause(b)[0].tolist() == [0, 0]
    closed(b)


def test_the_newborn_touches_its_parent_without_overlap():
    geom = flat_geom()
    b = make(1, N=2, size_kg=0.2, offspring_share=0.3)
    b.pos[0, 0] = torch.tensor([100.0, 100.0])
    b.heading[0, 0] = 0.3
    b.brood_j[0, 0] = B.child_cost_j(b)[0, 0]
    B.divide(b, torch.Generator().manual_seed(3))
    sh = B.shape(b)
    d = pt.wrap(b.pos[0, 1] - b.pos[0, 0], L)
    assert float(d.norm()) == pytest.approx(float(sh["a"][0, 0] + sh["r1"][0, 1]), abs=3e-5)   # float32 metres
    rear = -torch.tensor([math.cos(0.3), math.sin(0.3)])
    assert float((d / d.norm()) @ rear) == pytest.approx(1.0, abs=1e-5)             # behind the parent
    # the parent's mouth faces away: it does not touch its newborn
    assert int(B.body_contact(b, geom)[0, 0]) == -1
    out = B.ingest(b, torch.tensor([[1.0, 0.0]]), B.constant_env(), geom=geom)
    assert float(out["bitten_kg"][0, 0]) == 0.0 and float(b.damage[0, 1]) == 0.0


# ------------------------------------------------------------------------------------------------ heat
def test_the_heat_of_growth_warms_the_body():
    """Twin ectotherms in still, saturated air; one grows. Its extra heat Q over the bout warms it by the linear
    response of the heat balance, (Q / dt) / G x (1 - exp(-G dt / C))."""
    b = make(2, size_kg=0.03, fur_m=1e-3, T0=300.0, fat_store=0.05)
    for i in range(2):
        b.mass_kg[0, i] = 0.02
        b.frame_kg[0, i] = 0.02
        b.water_kg[0, i] = 0.02 * B.LEAN_WATER
    b.reserve_j[0, :] = 0.3 * 0.02 * B.FAT_J_KG
    b.n_kg[0, 0] = 1e-5                                               # nitrogen for about 0.28 g of growth
    B.reset_ledgers(b)
    env = B.constant_env(t_air_k=300.0, rel_humidity=1.0, wind_m_s=1.0)
    m = B.metabolism(b, env)
    q = float(m["growth_heat_j"][0, 0])
    assert q > 0 and float(m["growth_heat_j"][0, 1]) == 0.0
    G, C = float(m["G"][0, 0]), float(m["C"][0, 0])
    want = q / env.dt_s / G * (1 - math.exp(-G * env.dt_s / C))
    rise = float(b.body_k[0, 0] - b.body_k[0, 1])
    assert rise == pytest.approx(want, rel=0.15)
    assert rise > 0.3
    closed(b)


def test_the_respiratory_water_follows_the_bouts_temperatures():
    """A warm ectotherm cooling in cold saturated air breathes out water at each sub-step's T_b: the booked water
    lies between what the start and the end temperatures would give, and the heat balance's latent share matches it."""
    b = make(1, size_kg=0.02, fur_m=1e-4, T0=310.0, repair=0.0)
    env = B.constant_env(t_air_k=280.0, rel_humidity=1.0, wind_m_s=2.0)
    e = B._env(b, env)
    f_start = float(B.resp_water_per_j(torch.tensor([[310.0]]), e))
    m = B.metabolism(b, env)
    f_end = float(B.resp_water_per_j(b.body_k, e))
    ox = float(b.ledger["e_oxidised"])
    w = float(b.ledger["w_resp"])
    assert f_end * ox < w < f_start * ox
    assert w == pytest.approx(float(m["resp_kg_per_j"][0, 0]) * ox, rel=1e-5)


def test_work_that_leaves_the_body_is_not_its_heat():
    geom = flat_geom()
    b = make(1, size_kg=50.0, muscle_frac=0.4, T0=310.0)
    env = B.constant_env(t_air_k=293.0)
    r = B.move(b, torch.zeros(1, 1), torch.ones(1, 1), env, geom=geom)
    v = float(r["dist_m"]) / env.dt_s
    air = B.air_props(torch.tensor(293.0), torch.tensor(K.P_STANDARD))
    sh = B.shape(b)
    drag = 0.5 * float(air["rho"]) * B.C_DRAG * float(sh["frontal"]) * v ** 3 * env.dt_s
    assert float(r["drag_j"]) == pytest.approx(drag, rel=1e-3) and drag > 0
    assert float(r["heat_j"]) == pytest.approx(float(r["metabolic_j"]) - drag, rel=1e-5)


# ------------------------------------------------------------------------------------------------ temperature
def test_cold_muscles_move_bite_and_digest_less():
    cold, warm = 280.0, 305.0
    ratio = float(B.arrhenius(torch.tensor(cold)) / B.arrhenius(torch.tensor(warm)))
    assert 0.1 < ratio < 0.5                                          # Q10 about 2.2 over these 25 K
    b = make(2, size_kg=0.02)
    b.body_k[0] = torch.tensor([cold, warm])
    b.heading[0] = 0.0
    assert float(B.muscle_peak_w(b)[0, 0] / B.muscle_peak_w(b)[0, 1]) == pytest.approx(ratio, rel=1e-5)
    r = B.move(b, torch.zeros(1, 2), torch.full((1, 2), 0.5), B.constant_env(t_air_k=293.0))
    assert float(r["dist_m"][0, 0]) < 0.6 * float(r["dist_m"][0, 1])
    # the gut passes (and so absorbs) less per bout in the cold
    d = make(2, size_kg=0.05, enz_plant=0.05)
    d.body_k[0] = torch.tensor([cold, warm])
    d.gut[0, :, B.G_PLANT] = torch.tensor([0.002, 0.002 * B.PLANT_J_KG, 9.4e-4, 4e-5])
    fae = B.digest(d, B.DAY_S / 4)
    passed = 0.002 - d.gut[0, :, B.G_PLANT, B.Q_KG].double()        # what left the gut this bout
    assert float(passed[0]) < 0.6 * float(passed[1])
    assert float(fae["absorbed_j"][0, 0]) < 0.6 * float(fae["absorbed_j"][0, 1])
    # jaw cycles: on a prey that runs, the tissue cut scales with the number of bites, so with the Arrhenius rate
    geom = flat_geom()
    c = make(4, size_kg=[1.0, 1.0, 1.0, 1.0])
    c.body_k[0] = torch.tensor([cold, warm, warm, warm])
    sh = B.shape(c)
    for biter, prey, y in ((0, 1, 60.0), (2, 3, 180.0)):
        c.pos[0, prey] = torch.tensor([100.0, y])
        c.heading[0, biter] = 0.0
        c.pos[0, biter] = torch.tensor([100.0 - 0.5 * float(sh["r1"][0, prey]) - float(sh["a"][0, biter]), y])
        c.heading[0, prey] = math.pi / 2
        c.vel[0, prey] = torch.tensor([0.0, 2.0])
    out = B.ingest(c, torch.tensor([[1.0, 0.0, 1.0, 0.0]]), B.constant_env(), geom=geom)
    assert float(out["bitten_kg"][0, 0] / out["bitten_kg"][0, 2]) == pytest.approx(ratio, rel=1e-3)


# ------------------------------------------------------------------------------------------------ water books
def test_water_given_to_the_patch_is_neither_lost_nor_made():
    geom = flat_geom()
    ps = pt.reset_ledgers(pt.init_state(geom), geom)
    soil0 = ps.soil.double().clone()
    b = make(1)
    a, cell = torch.zeros(1, dtype=torch.long), torch.tensor([5 * 16 + 5])
    # 0.1 g: far below the float32 step of 75 kg/m^2 over 256 m^2 (about 2 g); it waits, it is not dropped
    B._give_water(b, ps, geom, a, cell, torch.tensor([1e-4], dtype=torch.float64))
    tr = B.transit_kg(b)["soil_in"]
    assert float(ps.w_given + tr) == pytest.approx(1e-4, rel=1e-9)
    for _ in range(59):
        B._give_water(b, ps, geom, a, cell, torch.tensor([1e-4], dtype=torch.float64))
    tr = B.transit_kg(b)["soil_in"]
    gained = float(((ps.soil.double() - soil0) * geom.cell_m2).sum())
    assert gained + float(tr) == pytest.approx(6e-3, rel=1e-9)
    assert gained == pytest.approx(float(ps.w_given), rel=1e-12) and gained > 3e-3   # the field has received most
    # 1 g gives 1 g, not a whole float32 step
    w0 = float(ps.w_given + B.transit_kg(b)["soil_in"])
    B._give_water(b, ps, geom, a, cell, torch.tensor([1e-3], dtype=torch.float64))
    assert float(ps.w_given + B.transit_kg(b)["soil_in"]) - w0 == pytest.approx(1e-3, rel=1e-9)
    assert abs(float(pt.water_ledger(ps, geom))) < 1e-6


def test_a_small_grazer_gets_its_plant_water():
    geom = flat_geom()
    ps = pt.init_state(geom)
    ps.plant = torch.full_like(ps.plant, 0.3)
    ps = pt.reset_ledgers(ps, geom)
    b = make(1, size_kg=0.002, enz_plant=0.05)
    b.heading[0, 0] = 0.0
    b.pos[0, 0] = torch.tensor([88.0 - float(B.shape(b)["a"][0, 0]), 88.0])
    snap = B.exchange_snapshot(b, ps)
    w0 = float(b.water_kg[0, 0])
    out = B.ingest(b, torch.ones(1, 1), B.constant_env(), geom=geom, pstate=ps)
    dry = float(out["plant_kg"][0, 0])
    assert dry > 0
    assert float(b.water_kg[0, 0]) - w0 == pytest.approx(dry * B.PLANT_WATER_PER_DRY, rel=1e-3)
    errs = B.exchange_errors(b, ps, geom, snap)
    for k in ("water_in", "carbon_in", "nitrogen_in"):
        assert float(errs[k].abs()) <= 1e-9 * float(errs[f"{k}_scale"]), k


# ------------------------------------------------------------------------------------------------ wear
def test_wear_is_rubners_energy_per_kg_and_has_no_repair_threshold():
    env = B.constant_env(t_air_k=283.0, rel_humidity=1.0)
    shares = [0.0, 0.3, 0.45, 0.6, 0.9]
    b = make(len(shares), size_kg=0.02, thermo_gain=30.0, setpoint_k=310.0, fur_m=3e-3, T0=310.0, repair=shares)
    m = B.metabolism(b, env)
    burnt = (m["paid_j"] + m["growth_heat_j"] * 0)[0].double()
    lean = b.mass_kg[0].double()
    rate = m["wear"][0].double()
    # wear per bout = the joules oxidised / (Rubner's lifetime energy x lean) x (1 - the share repair prevented)
    want = burnt / (B.LIFETIME_J_PER_KG * lean) * (1 - torch.tensor(shares, dtype=torch.float64))
    assert torch.allclose(rate, want, rtol=1e-4)
    # no share stops the wear (it falls continuously), and the wear is permanent
    per_joule = rate / burnt
    assert bool((per_joule[1:] < per_joule[:-1]).all()) and bool((rate > 0).all())
    assert torch.allclose(b.wear[0], b.damage[0], rtol=1e-6)
    # a 20 g endotherm in the cold lives months to years without repair, not days
    days = 1.0 / float(rate[0]) * env.dt_s / B.DAY_S
    assert 60 < days < 3000
    # a 70 kg body at about 1 W/kg: decades
    big = 0.92e9 / (1.0 * B.DAY_S) / 365.25
    assert 15 < big < 60


def test_wounds_heal_but_wear_stays():
    b = make(1, size_kg=0.05, repair=0.8, T0=305.0, thermo_gain=10.0, setpoint_k=305.0)
    b.damage[0, 0] = 0.6
    b.wear[0, 0] = 0.2
    env = B.constant_env(t_air_k=290.0, rel_humidity=1.0)
    for _ in range(40):
        B.metabolism(b, env)
    assert float(b.damage[0, 0]) < 0.25 and float(b.damage[0, 0]) >= float(b.wear[0, 0]) - 1e-7
    assert float(b.wear[0, 0]) > 0.2


# ------------------------------------------------------------------------------------------------ locomotion
def test_locomotion_against_measured_speed_and_cost():
    """Measured data: humans run at VO2max at about 5 m/s; mammals' incremental cost of transport is 10.7 M^-0.316
    J/(kg m) (Taylor, Heglund & Maloiy 1982, J. Exp. Biol. 97, 1). The spec's c_m = 0.1 and 25 % muscle efficiency
    plus the limbs' internal work match it within a factor 2 from 1 to 30 kg; below about 0.1 kg the model's cost
    stays near 6-8 J/(kg m) while measured costs rise to 40-80 (a gap the spec's constant efficiency leaves)."""
    env = B.constant_env(t_air_k=293.0)
    human = make(1, size_kg=70.0, muscle=100.0, muscle_frac=0.4, T0=310.15, fur_m=1e-4)
    r = B.move(human, torch.zeros(1, 1), torch.ones(1, 1), env)
    v = float(r["dist_m"]) / env.dt_s
    assert 3.0 < v < 7.0
    for mass in (1.0, 30.0):
        body = make(1, size_kg=mass, muscle=100.0, muscle_frac=0.4, T0=310.15, fur_m=1e-4)
        rr = B.move(body, torch.zeros(1, 1), torch.full((1, 1), 0.15), env)
        m = float(B.shape(body)["m"])
        cot = float(rr["metabolic_j"]) / (m * float(rr["dist_m"]))
        taylor = 10.7 * m ** -0.316
        assert 0.5 < cot / taylor < 2.0, (mass, cot, taylor)


def test_movement_can_be_sub_stepped():
    env = B.constant_env(t_air_k=293.0)
    one = make(1, size_kg=0.05)
    two = make(1, size_kg=0.05)
    for b in (one, two):
        b.heading[0, 0] = 0.4
    r = B.move(one, torch.zeros(1, 1), torch.full((1, 1), 0.01), env)
    parts = [B.move(two, torch.zeros(1, 1), torch.full((1, 1), 0.01), env, dt_s=env.dt_s / 2) for _ in range(2)]
    s = B.sum_loco(parts)
    assert s["dt_s"] == pytest.approx(env.dt_s)
    for k in ("work_j", "metabolic_j", "heat_j", "dist_m"):
        assert float(s[k]) == pytest.approx(float(r[k]), rel=1e-5), k
    assert torch.allclose(one.pos, two.pos, atol=1e-3)
    # bout() takes the world's sub-stepped locomotion instead of moving again
    pos = two.pos.clone()
    B.bout(two, {}, env, torch.Generator().manual_seed(0), loco=s)
    assert torch.equal(two.pos, pos)


# ------------------------------------------------------------------------------------------------ stores and kidney
def test_the_store_level_is_a_gene():
    b = make(2, size_kg=0.05, fat_store=[0.1, 1.0], repair=0.0)
    for i in range(2):
        b.mass_kg[0, i] = 0.03
        b.frame_kg[0, i] = 0.03
        b.water_kg[0, i] = 0.03 * B.LEAN_WATER
    b.reserve_j[0, :] = 0.5 * 0.03 * B.FAT_J_KG
    b.n_kg[0, :] = 1e-5
    B.reset_ledgers(b)
    m = B.metabolism(b, B.constant_env(t_air_k=300.0, rel_humidity=1.0))
    assert float(m["growth_kg"][0, 0]) > 0 and float(m["growth_kg"][0, 1]) == 0.0
    closed(b)


def test_the_bladder_holds_water_by_its_gene():
    b = make(2, size_kg=0.05, bladder=[1e-6, 0.2])
    b.water_kg[0, :] = B.LEAN_WATER * 0.05 + 0.006                    # 6 g drunk beyond the normal
    B.reset_ledgers(b)
    m = B.metabolism(b, B.constant_env(t_air_k=300.0, rel_humidity=1.0))
    normal = B.LEAN_WATER * 0.05
    assert float(b.water_kg[0, 0]) == pytest.approx(normal, rel=1e-3)          # urinated
    assert float(b.water_kg[0, 1]) > normal + 0.005                             # held
    assert float(m["urine_kg"][0, 0]) > 0.005 and float(m["urine_kg"][0, 1]) < 1e-3
    closed(b)


def test_the_kidney_is_an_organ_and_unflushed_salt_stays():
    geom = flat_geom()
    ps = pt.reset_ledgers(pt.init_state(geom), geom)
    b = make(3, size_kg=0.05, kidney=[1e-6, 0.005, 0.03])
    tis = B.tissues(b)
    assert float(B.maintenance_ref_w(b, tis)[0, 2] - B.maintenance_ref_w(b, tis)[0, 1]) == pytest.approx(
        float(tis["kidney"][0, 2] - tis["kidney"][0, 1]) * (B.C_KIDNEY_W_KG - B.C_REST_W_KG), rel=1e-3)
    cmax = B.urine_salt_max(b, tis)[0]
    assert float(cmax[2]) > float(cmax[1]) > 1e3 * float(cmax[0])
    salt = 2e-4
    b.salt_kg[0, :] = salt
    B.reset_ledgers(b)
    snap = B.exchange_snapshot(b, ps)
    w0 = b.water_kg[0].clone()
    m = B.metabolism(b, B.constant_env(t_air_k=300.0, rel_humidity=1.0), geom=geom, pstate=ps)
    # a working kidney flushes the salt with the water its urine carries; a missing one keeps it, and it harms
    assert float(b.salt_kg[0, 1]) == 0.0 and float(b.salt_kg[0, 2]) == 0.0
    assert float(m["salt_out_kg"][0, 1]) / float(m["urine_kg"][0, 1]) <= float(cmax[1]) * (1 + 1e-5)
    assert float(b.salt_kg[0, 0]) > 0.9 * salt and float(m["osmotic"][0, 0]) > 0
    assert float(b.water_kg[0, 0]) > 0.9 * float(w0[0])                          # its water is not flushed away
    assert float(m["osmotic"][0, 1]) == 0.0
    # the excreted salt went to the land, booked
    assert float(b.ledger["salt_land"]) == pytest.approx(float(m["salt_out_kg"][0].sum()), rel=1e-6)
    closed(b)
    errs = B.exchange_errors(b, ps, geom, snap)
    for k in ("water_out", "salt", "carbon_out", "nitrogen_out"):
        assert float(errs[k].abs()) <= 1e-6 * float(errs[f"{k}_scale"]) + 1e-15, k


# ------------------------------------------------------------------------------------------------ bites and minerals
def test_biters_of_one_target_share_it():
    geom = flat_geom()
    b = make(3, size_kg=[1.0, 1.0, 0.01])
    b.pos[0, 2] = torch.tensor([100.0, 100.0])
    sh = B.shape(b)
    r_t = float(sh["r1"][0, 2])
    for i, (dx, h) in enumerate(((-1.0, 0.0), (1.0, math.pi))):
        b.heading[0, i] = h
        b.pos[0, i] = torch.tensor([100.0 + dx * (0.5 * r_t + float(sh["a"][0, i])), 100.0])
    body_t = float(b.mass_kg[0, 2] + B.fat_kg(b)[0, 2])
    s0 = B.stocks(b)
    out = B.ingest(b, torch.tensor([[1.0, 1.0, 0.0]]), B.constant_env(), geom=geom)
    assert out["target"][0, :2].tolist() == [2, 2]
    took = float(out["bitten_kg"][0, :2].sum())
    assert took == pytest.approx(body_t, rel=1e-4)                    # each takes its share, together the prey
    assert float(b.mass_kg[0, 2]) == pytest.approx(0.0, abs=1e-9)
    s1 = B.stocks(b)
    for k in ("energy", "carbon", "nitrogen", "water", "inert"):
        assert float(s1[k]) == pytest.approx(float(s0[k]), rel=1e-6), k    # bone mineral moved into the guts
    assert float(b.ledger["i_food"]) == 0.0


def test_bone_mineral_is_booked_when_tissue_is_burnt():
    b = make(1, size_kg=0.02, T0=303.0, repair=0.0)
    b.reserve_j[0, 0] = 10.0                                          # almost no fat: the tissue pays
    B.reset_ledgers(b)
    lean0 = float(b.mass_kg[0, 0])
    B.metabolism(b, B.constant_env(t_air_k=303.0, rel_humidity=1.0))
    burnt = lean0 - float(b.mass_kg[0, 0])
    assert burnt > 0
    assert float(b.ledger["i_urine"]) == pytest.approx(burnt * B.LEAN_MINERAL, rel=1e-3)
    closed(b)


# ------------------------------------------------------------------------------------------------ heat-transfer reference
def test_convection_matches_an_independent_sphere_correlation():
    """Ranz-Marshall at the fur's diameter against Whitaker's sphere correlation (Whitaker 1972, AIChE J. 18, 361:
    Nu = 2 + (0.4 Re^0.5 + 0.06 Re^(2/3)) Pr^0.4, mu/mu_s about 1, valid for Re 3.5 to 7.6e4): within 20 %."""
    for mass, wind in ((0.005, 0.5), (0.2, 1.0), (5.0, 3.0), (70.0, 1.5)):
        b = make(1, size_kg=mass, fur_m=1e-6)
        e = B._env(b, B.constant_env(t_air_k=293.0, wind_m_s=wind))
        hx = B.heat_exchange(b, e)
        air = B.air_props(torch.tensor(293.0), torch.tensor(K.P_STANDARD))
        d = 2 * float(B.shape(b)["r2"])
        re = wind * d / float(air["nu"])
        assert 100 < re < 7.6e4
        nu_w = 2 + (0.4 * re ** 0.5 + 0.06 * re ** (2 / 3)) * B.PR_AIR ** 0.4
        h_w = nu_w * float(air["k"]) / d
        assert 0.8 < float(hx["h_c"]) / h_w < 1.2, (mass, wind, float(hx["h_c"]), h_w)


# ------------------------------------------------------------------------------------------------ the neutral shadow
def test_the_neutral_shadow_takes_the_runs_demography_at_random():
    b = B.found(1, 800, 400, L, torch.Generator().manual_seed(2), in_dim=4, hidden=8)
    sh = B.shadow_of(b)
    assert int(sh.alive.sum()) == 400 and sh.hidden == 8
    assert set(sh.genes) == set(B.GENE_SPECS) | {"k"}                 # every gene but the network weights
    gen = torch.Generator().manual_seed(5)
    alive0 = sh.alive.clone()
    logsize = sh.genes["size_kg"].double().log()
    out = B.shadow_step(sh, torch.tensor([200]), torch.tensor([0]), gen)
    died = alive0 & ~sh.alive
    assert out["died"].tolist() == [200] and int(died.sum()) == 200
    # who dies does not depend on the genes: the dead are a random half
    pop = logsize[alive0]
    se = float(pop.std()) / math.sqrt(200)
    assert abs(float(logsize[died].mean() - pop.mean())) < 4 * se
    # births: random living parents, the lowest free slots, children mutated as bodies are
    parents = sh.genes["size_kg"][sh.alive].clone()
    out = B.shadow_step(sh, torch.tensor([0]), torch.tensor([300]), gen)
    assert out["born"].tolist() == [300] and int(sh.alive.sum()) == 500
    child_sizes = sh.genes["size_kg"][0, sh.alive[0]]
    assert float(child_sizes.min()) >= B.GENE_SPECS["size_kg"].lo and len(set(child_sizes.tolist())) > 450
    assert not set(child_sizes.tolist()) <= set(parents.tolist())    # mutated
    k = sh.genes["k"][sh.alive]
    assert int(k.min()) >= 2 and int(k.max()) <= 8
    # births beyond the free slots fail; an empty shadow has no births
    out = B.shadow_step(sh, torch.tensor([0]), torch.tensor([1000]), gen)
    assert out["born"].tolist() == [300] and int(sh.alive.sum()) == 800
    out = B.shadow_step(sh, torch.tensor([800]), torch.tensor([5]), gen)
    assert int(sh.alive.sum()) == 0 and out["born"].tolist() == [0]
    # deterministic, and it round-trips through its state
    s1, s2 = B.shadow_of(b), B.shadow_of(b)
    for s in (s1, s2):
        B.shadow_step(s, torch.tensor([50]), torch.tensor([70]), torch.Generator().manual_seed(9))
    assert torch.equal(s1.alive, s2.alive) and all(torch.equal(s1.genes[k], s2.genes[k]) for k in s1.genes)
    s3 = B.Shadow.from_state(s1.state_dict())
    assert torch.equal(s3.alive, s1.alive) and s3.hidden == s1.hidden


def test_the_shadows_parents_have_one_child_each_and_keep_their_founder():
    """Review L13: a real body completes at most one birth per pass, so the shadow's parents are drawn without
    replacement until every living member had one; children carry their parent's founder id."""
    b = B.found(1, 800, 300, L, torch.Generator().manual_seed(4), in_dim=4, hidden=8)
    sh = B.shadow_of(b)
    assert torch.equal(sh.founder, b.founder)
    before = sh.alive.clone()
    out = B.shadow_step(sh, torch.tensor([0]), torch.tensor([300]), torch.Generator().manual_seed(1))
    assert out["born"].tolist() == [300]
    kids = sh.alive & ~before
    # every founder id appears exactly twice now: each living member was the parent of exactly one child
    ids = sh.founder[sh.alive]
    assert torch.equal(torch.bincount(ids, minlength=300), torch.full((300,), 2))
    assert int(kids.sum()) == 300
    s2 = B.Shadow.from_state(sh.state_dict())
    assert torch.equal(s2.founder, sh.founder)


def test_a_still_mouth_crops_its_disc_once_a_bout_however_many_sub_steps():
    """Review M2: one ingest call over the bout and K calls over bout / K with disc_share 1 / K take the same plant
    tissue from a still mouth (before, every call cropped the disc again: K times as much)."""
    geom = flat_geom()
    out = {}
    for K_sub in (1, 2, 4, 8):
        ps = pt.init_state(geom)
        ps.plant = torch.full_like(ps.plant, 0.2)
        ps = pt.reset_ledgers(ps, geom)
        b = make(3, size_kg=0.05, enz_plant=0.05)
        b.pos[0] = torch.tensor([[40.0, 40.0], [120.0, 120.0], [200.0, 40.0]])
        env = B.constant_env(dt_s=B.DAY_S / 24)
        dt_i = torch.full((1, 3), env.dt_s / K_sub)
        got = torch.zeros(1, 3, dtype=torch.float64)
        drunk = torch.zeros(1, 3, dtype=torch.float64)
        for s in range(K_sub):
            eat = B.ingest(b, torch.tensor([[0.05, 0.5, 1.0]]), env, geom=geom, pstate=ps, dt_body=dt_i,
                           disc_share=1.0 / K_sub, settle=s == 0)
            got += eat["plant_kg"]
            drunk += eat["drunk_kg"]
        out[K_sub] = (got, drunk)
    # equal up to the body's own growth within the bout (the food and water it took enlarge its reach: < 2 %)
    for K_sub in (2, 4, 8):
        assert torch.allclose(out[K_sub][0], out[1][0], rtol=0.02), K_sub
        assert torch.allclose(out[K_sub][1], out[1][1], rtol=1e-5), K_sub
    assert float(out[1][0].min()) > 0


def test_a_lean_body_sinks_and_drowns_and_a_fat_one_floats():
    """Review M15 and the review of 4 Oct 2026: buoyancy from the body's composition decides, against the water where
    the body is (sea water 1,027 kg/m^3); a body whose airway is under water draws its O2 store at its own O2 use and
    is harmed only by the debt beyond it (anoxia, healed as a wound); a floating body holds its density's share
    under water."""
    geom = pt.geometry_from_elevation(torch.zeros(1, 16, 16), L, 5.0, 5.674e6)          # all sea
    lean = make(1, size_kg=0.02, air_l_kg=0.01)                         # a small lung and little fat: it sinks
    lean.reserve_j[:] = 0.02 * lean.mass_kg * B.FAT_J_KG
    B.reset_ledgers(lean)
    fat = make(1, size_kg=0.02)
    fat.reserve_j[:] = 0.8 * fat.mass_kg * B.FAT_J_KG
    rl, rf = float(B.density(lean)), float(B.density(fat))
    assert rl > B.RHO_SEA > B.RHO_WATER > rf
    assert bool(B.sunk(lean, geom)) and not bool(B.sunk(fat, geom))
    assert float(B.submerged_share(lean, geom)) == pytest.approx(1.0)
    assert float(B.submerged_share(fat, geom)) == pytest.approx(rf / B.RHO_SEA, rel=1e-5)
    env = B.constant_env(t_air_k=300.0, dt_s=600.0)
    m = B.move(lean, torch.zeros(1, 1), torch.zeros(1, 1), env, geom=geom)          # still: it does not tread
    assert float(m["sunk_s"]) == pytest.approx(600.0)
    out = B.metabolism(lean, env, geom=geom, loco=m)
    hold = float(out["breath_hold_s"])
    o2_use = float(out["metabolic_w"]) / B.OXY_J_PER_MOL_O2                          # mol/s, the bout's own
    assert hold == pytest.approx(float(out["o2_store_mol"]) / o2_use, rel=1e-5)
    assert hold < 600.0                                    # a 20 g body with a small lung lasts minutes, not ten
    beyond = (600.0 - hold) * o2_use                       # the O2 debt beyond the store, mol
    q_ref = float(B.maintenance_ref_w(lean)) / B.OXY_J_PER_MOL_O2
    assert float(out["anoxic_mol"]) == pytest.approx(beyond, rel=1e-4)
    assert float(out["anoxia"]) == pytest.approx(beyond / (B.ANOXIA_TOLERANCE_S * q_ref), rel=1e-4)
    assert float(lean.o2_debt_mol) == pytest.approx(600.0 * o2_use, rel=1e-5)      # still under water: carried
    # on land nothing sinks
    dry = make(1, size_kg=0.02)
    assert not bool(B.sunk(dry, flat_geom()))
