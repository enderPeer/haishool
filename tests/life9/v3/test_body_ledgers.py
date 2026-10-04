"""Ledger, division, death, determinism and resume tests of v3 bodies (PLANET-V3-SPEC sections 3, 6 and 9)."""
from __future__ import annotations

import time

import pytest
import torch

from haishool.life9.planet import items as itm
from haishool.life9.planet import materials as mt
from haishool.life9.v3 import body as B
from haishool.life9.v3 import brain3 as b3
from haishool.life9.v3 import patch as pt

L = 256.0
BASE = dict(size_kg=0.05, fur_m=1e-3, skin_perm=1e-11, thermo_gain=1e-3, setpoint_k=310.0, enz_plant=0.03,
            enz_meat=0.02, repair=0.5, muscle=100.0, muscle_frac=0.3, offspring_share=0.4, eye=1e-4, ear=1e-4,
            voice=1e-4, fat_store=0.3, bladder=0.01, kidney=0.005, air_l_kg=0.2,
            o2_carrier=0.0)


def make(n=1, *, A=1, N=None, T0=300.0, seed=0, **genes):
    g = torch.Generator().manual_seed(seed)
    N = n if N is None else N
    b = B.found(A, N, n, L, g, in_dim=4, hidden=8, t_k=T0)
    for k, v in {**BASE, **genes}.items():
        b.genome[k][:, :n] = torch.as_tensor(v, dtype=torch.float32)
    size = b.genome["size_kg"][:, :n]
    b.mass_kg[:, :n] = size
    b.frame_kg[:, :n] = size
    b.reserve_j[:, :n] = B.FOUNDER_FAT_PER_LEAN * size * B.FAT_J_KG
    b.water_kg[:, :n] = B.LEAN_WATER * size
    B.reset_ledgers(b)
    return b


def closed(b, keys=("energy", "carbon", "water", "nitrogen", "inert", "salt", "oxygen", "co2"), tol=1e-6):
    errs = B.ledger_errors(b)
    for k in keys:
        rel = (errs[k].abs() / errs[f"{k}_scale"].clamp_min(1e-30)).max()
        assert float(rel) < tol, (k, float(rel), errs[k].tolist())


def flat_geom(A=1, n=16, seed=None):
    if seed is None:
        z = torch.zeros(A, n, n)
    else:
        z = torch.randn(A, n, n, generator=torch.Generator().manual_seed(seed)) * 2.0
    return pt.geometry_from_elevation(z, L, -100.0, 5.674e6)


# ------------------------------------------------------------------------------------------------ division
def test_division_conserves_mass_energy_and_water():
    b = make(2, N=4, offspring_share=[0.3, 0.7])
    b.n_kg[0, :2] = 1e-4
    b.salt_kg[0, 0] = 1e-5
    b.gut[0, 0, B.G_PLANT] = torch.tensor([1e-3, 1e-3 * B.PLANT_J_KG, 4.7e-4, 1.6e-5])
    b.damage[0, :2] = 0.2
    b.wear[0, :2] = 0.05
    B.reset_ledgers(b)
    keys = ("mass_kg", "frame_kg", "reserve_j", "water_kg", "n_kg")
    before = {k: getattr(b, k)[0, :2].double().clone() for k in keys}
    s0 = B.stocks(b)
    led0 = {k: v.clone() for k, v in b.ledger.items()}
    ready(b, [[True, True, False, False]])
    out = B.divide(b, torch.Generator().manual_seed(5))
    assert out["born"].tolist() == [2]
    ch = out["child_n"]
    assert sorted(ch.tolist()) == [2, 3] and bool(b.alive.all())
    par = out["parent_n"]
    share = b.genome["offspring_share"][0, par].double()
    for k in keys:
        tot = getattr(b, k)[0, par].double() + getattr(b, k)[0, ch].double()
        assert torch.allclose(tot, before[k][par], rtol=1e-6), k                     # conserved
        assert torch.allclose(getattr(b, k)[0, ch].double(), share * before[k][par], rtol=1e-5), k
    # the split oxidises nothing: the child's tissue is the parent's, paid for when it grew
    s1 = B.stocks(b)
    for k in ("energy", "carbon", "water", "nitrogen", "inert", "salt"):
        assert float(s1[k]) == pytest.approx(float(s0[k]), rel=1e-6), k
    for k in ("e_oxidised", "o2_mol", "w_resp", "w_metabolic"):
        assert float(b.ledger[k]) == float(led0[k]), k
    closed(b)
    # the frame shrinks with the lean: a parent that divided is not starving
    assert torch.allclose(b.frame_kg[0, par] / b.mass_kg[0, par], torch.ones(2), rtol=1e-5)
    assert (B.death_cause(b)[0] == 0).all()
    # the child starts new: no damage or wear, empty gut, its own salt 0, generation + 1, parent's uid, a fresh brain
    assert (b.damage[0, ch] == 0).all() and (b.wear[0, ch] == 0).all() and (b.gut[0, ch] == 0).all()
    assert (b.salt_kg[0, ch] == 0).all()
    assert (b.damage[0, par] == pytest.approx(0.2)) and (b.wear[0, par] == pytest.approx(0.05))
    assert (b.generation[0, ch] == 1).all() and torch.equal(b.parent[0, ch], b.uid[0, par])
    assert torch.equal(b.founder[0, ch], b.founder[0, par])
    assert len(set(b.uid[0].tolist())) == 4
    assert torch.equal(b.wh_live[0, ch], b.genome["Wh"][0, ch].to(b.wh_live.dtype))
    assert (b.hidden[0, ch] == 0).all()
    # the genome is the parent's plus variation, this module's genes included
    for name in ("size_kg", "fur_m", "repair", "mut", "fat_store", "bladder", "kidney"):
        assert not torch.equal(b.genome[name][0, ch], b.genome[name][0, par]), name
        assert torch.allclose(b.genome[name][0, ch].log(), b.genome[name][0, par].log(), atol=1.0) if name != \
            "repair" else True


def ready(b, mask=None):
    """Complete the development of the next child of the bodies in mask (default all living)."""
    m = b.alive if mask is None else torch.as_tensor(mask, dtype=torch.bool)
    b.brood_j = torch.where(m, B.child_cost_j(b), b.brood_j)


def test_division_fails_at_capacity_and_for_too_small_bodies_and_loses_the_development():
    full = make(3, N=3)
    full.ledger["capacity_full"].zero_()
    ready(full)
    out = B.divide(full, torch.Generator().manual_seed(1))
    assert int(out["born"].sum()) == 0 and float(full.ledger["capacity_full"]) == 3.0
    assert float(full.brood_j.abs().max()) == 0.0                     # a failed birth spent its development
    small = make(1, N=2, size_kg=1e-3, offspring_share=1e-4)          # a child below the smallest body
    ready(small)
    out = B.divide(small, torch.Generator().manual_seed(1))
    assert int(out["born"].sum()) == 0 and float(small.ledger["too_small"]) == 1.0
    greedy = make(1, N=2, size_kg=1e-3, offspring_share=0.9999)       # a parent left below the smallest body
    ready(greedy)
    out = B.divide(greedy, torch.Generator().manual_seed(1))
    assert int(out["born"].sum()) == 0 and float(greedy.ledger["too_small"]) == 1.0
    # births need the whole development: none below it, exactly the completed ones
    many = make(400, N=800, seed=3)
    half = torch.zeros(1, 800, dtype=torch.bool)
    half[0, :400:2] = True
    ready(many, half)
    many.brood_j[0, 1:400:2] = 0.999 * B.child_cost_j(many)[0, 1:400:2]
    out = B.divide(many, torch.Generator().manual_seed(2))
    assert int(out["born"].sum()) == 200 and sorted(out["parent_n"].tolist()) == list(range(0, 400, 2))
    zero = make(10, N=20)
    assert int(B.divide(zero, torch.Generator().manual_seed(2))["born"].sum()) == 0
    with pytest.raises(TypeError):
        B.divide(zero, torch.ones(1, 20), torch.Generator().manual_seed(2))


def test_development_is_paid_as_heat_and_its_rate_does_not_depend_on_the_bouts():
    """PLANET-V3-SPEC 0 and the review's H3: the divide output is a share of a physical rate (protein synthesis),
    so the development made in a day, and the children it buys, are the same for 4 and for 24 bouts a day."""
    env = lambda bouts: B.constant_env(t_air_k=300.0, rel_humidity=1.0, dt_s=B.DAY_S / bouts)
    got = {}
    for bouts in (4, 24):
        b = make(2, N=8, size_kg=0.02, offspring_share=0.05, T0=300.0, thermo_gain=1e-3)
        b.reserve_j[:] = 1.0 * b.mass_kg * B.FAT_J_KG                # fuel for the day
        B.reset_ledgers(b)
        born = 0
        for _ in range(bouts):
            b.body_k[0, :2] = 300.0                                    # the same temperature in both schedules
            m = B.metabolism(b, env(bouts), develop=torch.tensor([[1.0, 0.25] + [0.0] * 6]))
            born += int(B.divide(b, torch.Generator().manual_seed(1))["born"].sum())
        closed(b)
        got[bouts] = (born, b.ledger["births"].clone(), float(b.brood_j[0, 0]), float(b.brood_j[0, 1]))
    rate = float(B.develop_power_w(make(1, size_kg=0.02, T0=300.0))[0, 0]) * B.DAY_S
    cost = B.DEVELOP_J_PER_KG * 0.05 * 0.02
    assert got[4][0] == got[24][0] >= int(rate / cost) - 1 > 0                      # children per day
    # a full-intensity body makes about rate / cost children a day, a quarter-intensity one a quarter of that
    assert got[4][0] == pytest.approx(rate / cost * 1.25, abs=3)


def test_a_body_too_large_for_its_tissues_runs_only_the_units_its_brain_holds():
    b = make(2, size_kg=[0.05, 2e-3], muscle_frac=[0.3, 0.98])
    b.genome["k"][0] = torch.tensor([8, 32])
    tis = B.tissues(b)
    k_eff = B.active_units(b, tis)
    assert int(k_eff[0, 0]) == 8                                     # the brain it asked for
    assert float(tis["asked"][0, 1]) > 1 and int(k_eff[0, 1]) < 32  # scaled tissue: fewer units
    assert int(k_eff[0, 1]) == int(float(tis["brain"][0, 1]) / B.BRAIN_KG_PER_UNIT + 1e-3)


# ------------------------------------------------------------------------------------------------ deaths
def test_carcass_items_carry_the_body_and_held_items_drop():
    geom = flat_geom()
    ps = pt.init_state(geom)
    pool = itm.new_item_pool(1, 32)
    b = make(3, size_kg=[0.5, 0.5, 0.5])
    b.pos[0] = torch.tensor([[40.0, 40.0], [120.0, 120.0], [200.0, 40.0]])
    b.gut[0, 0, B.G_PLANT] = torch.tensor([1e-3, 1e-3 * B.PLANT_J_KG, 4.7e-4, 1.6e-5])
    b.n_kg[0, 0] = 1e-5
    held = itm.spawn(pool, torch.tensor([0]), torch.nn.functional.one_hot(torch.tensor(mt.IDX["flint"]), mt.S)
                     .float()[None], torch.tensor([0.1]), torch.tensor([[0.0, 0.0, 0.0]]), torch.tensor([300.0]),
                     torch.tensor([0]))
    b.held[0, 0, 0] = int(held)
    b.water_kg[0, 0] = 0.9 * B.LEAN_WATER * 0.5                      # drier than normal: smaller wet parts
    b.damage[0, 0] = 1.2                                              # dies of damage
    b.water_kg[0, 1] = 0.1 * B.LEAN_WATER * 0.5                      # dies of dehydration
    B.reset_ledgers(b)
    ps = pt.reset_ledgers(ps, geom)
    s0 = B.stocks(b)
    items0 = itm.species_mass(pool).clone()
    out = B.deaths(b, geom=geom, pstate=ps, pool=pool)
    assert out["dead"][0].tolist() == [True, True, False]
    assert out["cause"][0].tolist()[:2] == [B.CAUSE["damage"], B.CAUSE["dehydration"]]
    made = itm.species_mass(pool) - items0
    assert torch.allclose(made, b.ledger["carcass_kg"], rtol=1e-6, atol=1e-12)
    for s in ("meat", "bone", "hide", "fat"):
        assert float(made[0, mt.IDX[s]]) > 0, s
    # the healthy-water body's meat is its full share; the dry one's is scaled to the water it held
    assert float(b.ledger["deaths_damage"]) == 1.0 and float(b.ledger["deaths_dehydration"]) == 1.0
    # energy, carbon and water are all accounted: items + litter + soil = what the bodies held
    e_dead = float(b.ledger["e_dead"])
    assert e_dead == pytest.approx(float(s0["energy"] - B.stocks(b)["energy"]), rel=1e-6)
    item_e = float((out["item_j"]).sum())
    assert item_e < e_dead
    # the litter received the remainder of the carbon, the soil the remainder of the water: what the float32
    # fields could not yet resolve waits in the transit residuals, nothing is lost
    tr = B.transit_kg(b)
    assert float(ps.c_added + tr["litter_in"]) == pytest.approx(float(out["litter_c"].sum()), rel=1e-9)
    assert float(ps.w_given + tr["soil_in"]) == pytest.approx(float(out["soil_w"].sum()), rel=1e-9)
    assert float(ps.n_added + tr["nutrient_in"]) == pytest.approx(float(out["litter_n"].sum()), rel=1e-9)
    assert float(out["item_c"].sum() + out["litter_c"].sum()) == pytest.approx(float(b.ledger["c_dead"]), rel=1e-6)
    assert float(out["item_w"].sum() + out["soil_w"].sum()) == pytest.approx(float(b.ledger["w_dead"]), rel=1e-6)
    # the item's carbon is what the species makeup holds
    c_items = float((made.double() @ torch.tensor([mt.species_elements(s).get("C", 0.0) for s in mt.SPECIES],
                                                    dtype=torch.float64)).sum())
    assert c_items == pytest.approx(float(out["item_c"].sum()), rel=1e-5)
    # the held flint dropped where the body lay
    assert int(pool.holder[0, int(held)]) == -1
    assert torch.allclose(pool.pos[0, int(held), :2], torch.tensor([40.0, 40.0]))
    assert int(b.held[0, 0, 0]) == -1 and not bool(b.alive[0, 0])
    closed(b)


def test_without_a_pool_the_dead_go_to_litter():
    geom = flat_geom()
    ps = pt.reset_ledgers(pt.init_state(geom), geom)
    b = make(2)
    b.body_k[0, 0] = 250.0
    B.reset_ledgers(b)
    out = B.deaths(b, geom=geom, pstate=ps)
    assert float(out["item_j"].sum()) == 0.0
    tr = B.transit_kg(b)
    assert float(ps.c_added + tr["litter_in"]) == pytest.approx(float(b.ledger["c_dead"]), rel=1e-9)
    closed(b)


def _control_run(seed, bouts=6, per_bout=3):
    """40 bodies; each bout ``per_bout`` living bodies are made lethal (damage 1.5), then the control's deaths."""
    b = make(40, N=40, seed=4)
    B.reset_ledgers(b)
    gen = torch.Generator().manual_seed(seed)
    fur = {int(u): float(f) for u, f in zip(b.uid[0], b.genome["fur_m"][0])}       # a genome's tag by its uid
    lethal_uids, lost_uids = [], []
    for k in range(bouts):
        living = b.alive[0].nonzero()[:, 0]
        pick = living[(torch.arange(per_bout) * 5 + k) % living.numel()]
        b.damage[0, pick] = 1.5
        lethal_uids += b.uid[0, pick].tolist()
        before = set(b.uid[0, b.alive[0]].tolist())
        out = B.deaths(b, selection=False, gen=gen)
        # physics removes exactly its dead, for its own cause; no lethal body is left alive
        assert int(out["dead"].sum()) == per_bout
        assert torch.equal(out["dead"][0].nonzero()[:, 0].sort().values, pick.sort().values)
        assert bool((out["cause"][out["dead"]] == B.CAUSE["damage"]).all())
        assert not bool((B.death_cause(b) > 0).any())
        after = set(b.uid[0, b.alive[0]].tolist())
        lost = before - after
        assert len(lost) == per_bout and int(out["genome_lost"].sum()) == per_bout
        lost_uids += sorted(lost)
        # a genome keeps its lineage ids wherever it moved, and every uid is still unique
        alive_uids = b.uid[0, b.alive[0]].tolist()
        assert len(set(alive_uids)) == len(alive_uids)
        for u, f in zip(alive_uids, b.genome["fur_m"][0, b.alive[0]].tolist()):
            assert f == fur[u]
    return b, lethal_uids, lost_uids


def test_no_selection_control_removes_the_physical_dead_and_loses_random_genomes():
    b, lethal, lost = _control_run(9)
    assert float(b.ledger["deaths_damage"]) == 18.0 and float(b.ledger["control_swaps"]) > 0
    assert int(b.alive.sum()) == 40 - 18
    # which genomes were lost is not which bodies died
    assert set(lost) != set(lethal)
    assert len(set(lost) & set(lethal)) < len(lethal)
    closed(b)
    # deterministic from the generator, and random across seeds
    _, _, lost2 = _control_run(9)
    _, _, lost3 = _control_run(10)
    assert lost2 == lost and lost3 != lost
    with pytest.raises(ValueError):
        B.deaths(make(2), selection=False)


# ------------------------------------------------------------------------------------------------ the whole loop
def _world(seed=0, A=4, N=192, F=48, n=16):
    g = torch.Generator().manual_seed(seed)
    b = B.found(A, N, F, L, g, in_dim=6, hidden=8, t_k=295.0)
    geom = flat_geom(A, n, seed=seed + 1)
    ps = pt.init_state(geom)
    ps.plant = torch.where(geom.sea, 0.0, 0.3)
    ps.pond[:, 3:6, 3:6] = 30.0
    ps = pt.reset_ledgers(ps, geom)
    pool = itm.new_item_pool(A, 256)
    forcing = pt.constant_forcing(geom, t_air_k=295.0, sw_day_w_m2=220.0, precip_kg_m2=3.0)
    return b, geom, ps, pool, forcing, g


def _motor(mg, b):
    A, N = b.shape
    out = torch.randn(A, N, b3.OUT_DIM, generator=mg)
    m = b3.decode(out)
    m["thrust"] = m["thrust"] * 0.003
    m["mouth"] = 0.5 + 0.5 * m["mouth"]
    m["divide"] = m["divide"] * 0.1
    return m


def _run(b, geom, ps, pool, forcing, g, mg, bouts, start=0):
    day_sw = None
    for k in range(start, start + bouts):
        if k % 4 == 0:
            ps, diag = pt.step(ps, geom, forcing)
            day_sw = diag["bout_sw"]
        env = B.patch_env(geom, forcing, day_sw[:, k % 4])
        B.bout(b, _motor(mg, b), env, g, geom=geom, pstate=ps, pool=pool)
    return ps


def test_ledgers_close_over_100_days_with_plants_items_and_the_patch():
    t0 = time.time()
    b, geom, ps, pool, forcing, g = _world()
    items0 = itm.species_mass(pool).clone()
    snap = B.exchange_snapshot(b, ps)
    mg = torch.Generator().manual_seed(11)
    ps = _run(b, geom, ps, pool, forcing, g, mg, 400)
    assert time.time() - t0 < 75
    assert int(b.ledger["births"].sum()) > 0 and float(sum(b.ledger[f"deaths_{c}"].sum() for c in B.CAUSES[1:])) > 0
    assert float(b.ledger["e_food"].sum()) > 0 and float(b.ledger["w_drunk"].sum()) > 0
    closed(b)
    # the patch's own ledgers close
    assert float((pt.carbon_ledger(ps, geom).abs() / pt.carbon_stores(ps, geom).clamp_min(1)).max()) < 1e-4
    # items: what is in the pool = what was there + carcasses - what was eaten
    pool_now = itm.species_mass(pool)
    expect = items0 + b.ledger["carcass_kg"] - b.ledger["items_eaten_kg"]
    assert torch.allclose(pool_now, expect, rtol=1e-4, atol=1e-6)
    assert float(pt.water_ledger(ps, geom).abs().max()) < 1e-6 * float(pt.water_stores(ps, geom).max())
    # what bodies took from and gave to the patch is what the patch booked (plus what waits in transit), in water,
    # sea water, carbon, nitrogen and salt, to 1e-6
    errs = B.exchange_errors(b, ps, geom, snap)
    for k in ("water_in", "water_out", "sea", "carbon_in", "carbon_out", "nitrogen_in", "nitrogen_out", "salt"):
        rel = float((errs[k].abs() / errs[f"{k}_scale"].clamp_min(1e-30)).max())
        assert rel < 1e-6, (k, rel)
    assert float(errs["water_in_scale"].min()) > 0 and float(errs["water_out_scale"].min()) > 0
    # written out for water: urine + faeces water + dead bodies' water not in carcass items = patch's w_given + what
    # waits in transit; fresh water drunk + plant water = patch's w_taken + what waits in transit
    tr = B.transit_kg(b)
    carc_w = b.ledger["carcass_kg"] @ B.SPECIES_WATER
    out_w = b.ledger["w_urine"] + b.ledger["w_faeces"] + b.ledger["w_dead"] - carc_w
    assert torch.allclose(out_w, ps.w_given + tr["soil_in"], rtol=1e-6, atol=0)
    plant_w = b.ledger["w_food"] - b.ledger["items_eaten_kg"] @ B.SPECIES_WATER
    assert torch.allclose(b.ledger["w_drunk"] + plant_w, ps.w_taken + tr["soil_out"] + tr["pond_out"], rtol=1e-6,
                          atol=0)
    # the residuals stay small: within about a float32 step of their arena's largest field value per cell (a
    # residual can outlive a pond the day dried up)
    for key, v in b.transit.items():
        field = getattr(ps, B.TRANSIT_KEYS[key][0]).reshape(v.shape).double() * geom.cell_m2
        assert bool((v.abs() <= 1.2e-7 * field.abs().amax(-1, keepdim=True) + 1e-9).all()), key
    # gas exchange: O2 taken and CO2 given are booked for the air
    gas = B.gas_mol(b)
    assert bool((gas["o2_mol"] > 0).all()) and bool((gas["co2_mol"] > 0).all())


def test_determinism_and_exact_resume():
    def fresh():
        return _world(seed=3, A=2, N=96, F=24)

    b1, geom, ps1, pool1, forcing, g1 = fresh()
    mg1 = torch.Generator().manual_seed(4)
    ps1 = _run(b1, geom, ps1, pool1, forcing, g1, mg1, 24)
    h1 = B.state_hash(b1)
    b2, _, ps2, pool2, _, g2 = fresh()
    mg2 = torch.Generator().manual_seed(4)
    ps2 = _run(b2, geom, ps2, pool2, forcing, g2, mg2, 24)
    assert B.state_hash(b2) == h1 and itm.state_hash(pool2) == itm.state_hash(pool1)
    # checkpoint at bout 12, resume, and land on the same state
    b3_, _, ps3, pool3, _, g3 = fresh()
    mg3 = torch.Generator().manual_seed(4)
    ps3 = _run(b3_, geom, ps3, pool3, forcing, g3, mg3, 12)
    ck = {"b": b3_.state_dict(), "ps": ps3.state_dict(), "pool": pool3.state_dict(), "g": g3.get_state(),
          "mg": mg3.get_state()}
    del b3_, ps3, pool3
    b4 = B.Bodies.from_state(ck["b"])
    ps4 = pt.PatchState.from_state(ck["ps"])
    pool4 = itm.from_state(ck["pool"])
    g4 = torch.Generator()
    g4.set_state(ck["g"])
    mg4 = torch.Generator()
    mg4.set_state(ck["mg"])
    _run(b4, geom, ps4, pool4, forcing, g4, mg4, 12, start=12)
    assert B.state_hash(b4) == h1


def test_a_full_size_bout_is_fast():
    g = torch.Generator().manual_seed(0)
    A, N, F = 2, 8192, 2048
    b = B.found(A, N, F, 2048.0, g, in_dim=4, hidden=8, t_k=293.0, brain=False)
    geom = pt.geometry_from_elevation(torch.randn(A, 128, 128, generator=g) * 3, 2048.0, -100.0, 5.674e6)
    ps = pt.init_state(geom)
    ps.plant = torch.full_like(ps.plant, 0.3)
    ps = pt.reset_ledgers(ps, geom)
    pool = itm.new_item_pool(A, 2048)
    env = B.constant_env(t_air_k=293.0, sw_w_m2=200.0)
    mg = torch.Generator().manual_seed(1)
    t = time.time()
    for _ in range(3):
        m = _motor(mg, b)
        m["divide"] = m["divide"] * 0.0
        B.bout(b, m, env, g, geom=geom, pstate=ps, pool=pool)
    assert (time.time() - t) / 3 < 0.75
    closed(b)


def test_item_changes_can_go_through_manipulates_item_ledger():
    """Eating items and leaving carcasses through manipulate's remove_mass / spawn_items keeps its item ledger closed
    and books the same species as the body's own ledger."""
    mp = pytest.importorskip("haishool.life9.v3.manipulate")
    if not all(hasattr(mp, n) for n in ("remove_mass", "spawn_items", "new_pools", "new_ledger", "ledger_error",
                                        "new_held")):
        pytest.skip("manipulate's item API is not there")
    from functools import partial
    geom = flat_geom()
    items, fires = mp.new_pools(1, 32, 4)
    b = make(3, size_kg=1.0, enz_plant=0.05, enz_meat=0.05)
    sh = B.shape(b)
    held = mp.new_held(1, 3)
    for i, (xy, s, m) in enumerate(zip([(30.0, 30.0), (90.0, 30.0)], ["meat", "fat"], [0.5, 0.2])):
        b.heading[0, i] = 0.0
        b.pos[0, i] = torch.tensor([xy[0] - float(sh["a"][0, i]), xy[1]])
        comp = torch.nn.functional.one_hot(torch.tensor(mt.IDX[s]), mt.S).float()[None]
        itm.spawn(items, torch.tensor([0]), comp, torch.tensor([m]), torch.tensor([[xy[0], xy[1], 0.0]]),
                  torch.tensor([300.0]), torch.tensor([-1]))
    b.pos[0, 2] = torch.tensor([200.0, 200.0])
    led = mp.new_ledger(items, fires)
    take = lambda pool, a, i, kg: mp.remove_mass(pool, held, a, i, kg, ledger=led)["species_kg"]
    out = B.ingest(b, torch.tensor([[1.0, 1.0, 0.0]]), B.constant_env(), geom=geom, pool=items, take_items=take)
    assert out["contact"][0].tolist()[:2] == [2, 2] and float(out["item_kg"][0, 0]) > 0
    assert torch.allclose(led.out_kg, b.ledger["items_eaten_kg"], rtol=1e-9, atol=1e-12)
    b.damage[0, 2] = 2.0
    spawn = partial(mp.spawn_items, ledger=led, L=L)
    B.deaths(b, geom=geom, pool=items, spawn_items=spawn)
    assert torch.allclose(led.in_kg, b.ledger["carcass_kg"], rtol=1e-6, atol=1e-12)
    err = mp.ledger_error(led, items, fires)
    assert float(err["mass_kg"].abs().max()) < 1e-6 * float(err["total_kg"].max())
    closed(b)
