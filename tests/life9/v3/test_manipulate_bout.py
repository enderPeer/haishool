"""Whole bouts of life9 v3's contact physics (PLANET-V3-SPEC section 8): random motor outputs on crowded patches,
item ledgers that close over days, determinism and exact resume on the CPU, the contact hash against brute force,
fires from random outputs at founder size, and speed."""
from __future__ import annotations

import math
import time

import pytest
import torch

from haishool.life9.planet import crafting as cr
from haishool.life9.planet import items as it
from haishool.life9.planet import materials as m
from haishool.life9.v3 import brain3
from haishool.life9.v3 import manipulate as mp

L = 2048.0
LOOSE = ("wood", "flint", "basalt", "clay", "plant_fiber", "resin", "meat", "fat", "bone", "hide", "granite",
         "charcoal")
CELLS = 128
SUSTAINED = 0.3                                 # body.py's SUSTAINED_SHARE: sustained over peak power


def world(A=2, N=64, I=512, F=32, seed=0, area_m=40.0, mass_kg=(0.5, 80.0), peak_w_kg=20.0, fires=4):
    """Bodies of mass_kg (log-uniform) and loose items crowded into an area_m square near the patch's corner (so the
    periodic boundary runs through it), a few fires, and a ground stock of wood, litter and basalt."""
    g = torch.Generator().manual_seed(seed)
    items, fp = mp.new_pools(A, I, F)
    held = mp.new_held(A, N)
    ledger = mp.new_ledger(items, fp)
    origin = L - area_m / 2
    pos = torch.remainder(origin + area_m * torch.rand(A, N, 2, generator=g), L)
    heading = 2 * torch.pi * torch.rand(A, N, generator=g)
    lo, hi = (torch.log(torch.tensor(x)) for x in mass_kg)
    mass = torch.exp(torch.empty(A, N).uniform_(0.0, 1.0, generator=g) * (hi - lo) + lo)
    n_loose = I // 2
    a = torch.arange(A).repeat_interleave(n_loose)
    species = torch.randint(len(LOOSE), (A * n_loose,), generator=g)
    comp = torch.stack([m.species_vector({LOOSE[int(s)]: 1.0}) for s in species])
    kg = torch.exp(torch.empty(A * n_loose).uniform_(-5.0, 1.0, generator=g))
    xy = torch.remainder(origin + area_m * torch.rand(A * n_loose, 2, generator=g), L)
    is_wood = species == LOOSE.index("wood")
    length = torch.where(is_wood, mp.stick_length_m(kg, torch.tensor(float(m.DENSITY["wood"]))), torch.zeros_like(kg))
    mp.spawn_items(items, a, comp, kg, xy, torch.full((A * n_loose,), 290.0), ledger=ledger, L=L, handle_len_m=length)
    for f in range(fires):
        bed = m.species_vector({"wood": 2.0, "charcoal": 1.0})[None]
        p = torch.remainder(origin + area_m * torch.rand(A, 2, generator=g), L)
        it.spawn_fire(fp, torch.arange(A), torch.cat((p, torch.zeros(A, 1)), -1), bed.expand(A, -1),
                      torch.full((A,), 1100.0))
        ledger.book_in(m.species_vector({"wood": 2.0, "charcoal": 1.0}, dtype=torch.float64)[None].expand(A, -1))
    stock = torch.zeros(A, CELLS * CELLS, m.S, dtype=torch.float64)
    stock[..., m.IDX["wood"]] = 8.0
    stock[..., m.IDX["plant_fiber"]] = 0.6
    stock[..., m.IDX["basalt"]] = 130.0
    return dict(items=items, fires=fp, held=held, ledger=ledger, pos=pos, heading=heading, mass=mass, gen=g,
                peak_w_kg=peak_w_kg, ground={"kg_m2": stock, "cells": CELLS})


def mouth_of(w):
    r = mp.body_radius_m(w["mass"])
    ahead = torch.stack((torch.cos(w["heading"]), torch.sin(w["heading"])), -1)
    return torch.remainder(w["pos"] + r[..., None] * ahead, L)


def one_bout(w, dt=21600.0, decay=True, raw_scale=2.0):
    A, N = w["mass"].shape
    g = w["gen"]
    # random motor outputs, as random founders' would be, and a random walk of a few metres
    raw = torch.randn(A, N, brain3.OUT_DIM, generator=g) * raw_scale
    out = brain3.decode(raw)
    step = torch.randn(A, N, 2, generator=g)
    w["pos"] = torch.remainder(w["pos"] + step, L)
    w["heading"] = w["heading"] + 0.5 * out["turn"]
    alive = torch.ones(A, N, dtype=torch.bool)
    peak = w["peak_w_kg"] * w["mass"]                                   # W per kg of body at peak
    mouth = mouth_of(w)
    res = mp.bout(w["items"], w["fires"], w["held"], out, g, alive=alive, pos=w["pos"], mouth_pos=mouth,
                  body_mass_kg=w["mass"], peak_w=peak, sustained_w=SUSTAINED * peak, gravity=9.81, x_o2=0.21,
                  air_k=290.0, L=L, dt_s=dt, thrust=out["thrust"], vel=step / dt, ground=w["ground"],
                  ledger=w["ledger"], cells=CELLS)
    fl = mp.heat_step(w["items"], w["fires"], w["held"], x_o2=0.21, air_k=290.0, L=L, dt_s=dt, ledger=w["ledger"],
                      cells=CELLS, substeps=16, mouth_pos=mouth, body_pos=w["pos"], alive=alive)
    if decay:
        mp.decay_step(w["items"], w["held"], dt_days=dt / 86400.0, L=L, cells=CELLS, ledger=w["ledger"])
    # mouths bite into what they touch, and a carcass arrives
    c = mp.contacts(w["items"], w["fires"], alive=alive, pos=w["pos"], mouth_pos=mouth, body_mass_kg=w["mass"], L=L)
    an, nn = (c["item"] >= 0).nonzero(as_tuple=True)
    mp.remove_mass(w["items"], w["held"], an, c["item"][an, nn], 0.01 * w["mass"][an, nn], ledger=w["ledger"])
    carcass = m.species_vector({"meat": 0.7, "fat": 0.1, "bone": 0.15, "hide": 0.05})[None].expand(A, -1)
    mp.spawn_items(w["items"], torch.arange(A), carcass, torch.full((A,), 2.0), w["pos"][:, 0], torch.full((A,), 310.0),
                   ledger=w["ledger"], L=L)
    return res, fl


def test_random_bouts_keep_the_item_ledgers_closed_and_the_grips_consistent():
    w = world()
    stats = {"grips": 0, "fires": 0, "embers": 0, "knaps": 0, "joins": 0, "placed": 0, "hits": 0, "broke": 0,
             "wound_j": 0.0}
    for _ in range(8):                                     # two days of four bouts
        res, fl = one_bout(w)
        stats["grips"] += int((res["grip"]["item"] >= 0).sum())
        stats["fires"] += int((res["rub"]["fire"] >= 0).sum())
        stats["embers"] += int(res["rub"]["ember"].sum())
        stats["knaps"] += int(res["force"]["knapped"].sum())
        stats["joins"] += int((res["press"]["item"] >= 0).sum())
        stats["placed"] += int((res["place"]["item"] >= 0).sum())
        stats["hits"] += int((res["force"]["hit_body"] >= 0).sum())
        stats["broke"] += int((res["force"]["broke"] >= 0).sum())
        stats["wound_j"] += float(res["force"]["wound_j"].sum())
        assert not mp.check(w["items"], w["held"])
        assert torch.isfinite(res["work_j"]).all() and (res["work_j"] >= 0).all()
        assert ((res["damage"] >= 0) & (res["damage"] <= 1)).all()
        assert (res["time_s"] <= 21600.0 * (1 - res["shares"]["locomotion"]) + 1e-2).all()
        # hits are reported only where a stroke landed; a body's blows carry energy
        hit = res["force"]["hit_body"] >= 0
        assert (res["force"]["hits"][hit] >= 1).all() and (res["force"]["blow_j"][hit] > 0).all()
        # the spot never passes its partners' melting or decomposition (CAP_RULE); skin chars at the tissue cap
        hottest = max(float(v) for v in m.MELT_K.values() if float(v) < float(m.NEVER_K))
        assert torch.isfinite(res["rub"]["contact_k"]).all() and (res["rub"]["contact_k"] <= hottest).all()
        assert (res["rub"]["skin_contact_k"] <= float(cr.TISSUE_CHAR_K) + 1e-3).all()
        assert torch.isfinite(fl["radiant_w_m2"]).all() and torch.isfinite(fl["mouth_fire_k"]).all()
        err = mp.ledger_error(w["ledger"], w["items"], w["fires"])
        scale = float(err["total_kg"].max())
        assert float(err["mass_kg"].abs().max()) < 1e-6 * scale
        assert float(err["elements_kg"].abs().max()) < 1e-6 * scale
        assert torch.isfinite(w["items"].temp_k).all() and torch.isfinite(w["items"].pos).all()
        ground = w["items"].alive & (w["items"].holder < 0)
        xy = w["items"].pos[..., :2][ground]
        assert ((xy >= 0) & (xy < L)).all()
    # the physics did happen on the crowded patch
    assert stats["grips"] > 0 and stats["placed"] > 0 and stats["knaps"] + stats["hits"] > 0, stats
    assert stats["broke"] > 0 and stats["wound_j"] > 0, stats


def test_random_outputs_at_founder_size_light_no_fire():
    """Founders weigh 2-200 g (PLANET-V3-SPEC 6): with random outputs and a peak of 100 W per kg of body (a body
    of muscle at its best) on a patch strewn with wood and tinder, no fire is born: rubbing that lights a fire is
    not behaviour random weights already have at founder size."""
    w = world(A=2, N=256, I=1024, seed=7, area_m=20.0, mass_kg=(0.002, 0.2), peak_w_kg=100.0, fires=0)
    fires = embers = rubs = 0
    for _ in range(8):
        res, _ = one_bout(w)
        fires += int((res["rub"]["fire"] >= 0).sum())
        embers += int(res["rub"]["ember"].sum())
        rubs += int((res["rub"]["partner_a"] >= 0).sum() + (res["rub"]["partner_b"] >= 0).sum())
    assert rubs > 100 and fires == 0 and embers == 0, (rubs, embers, fires)


def _hash(w):
    return (it.state_hash(w["items"]), it.state_hash(w["fires"]), w["held"].clone(),
            {k: v.clone() for k, v in w["ledger"].state_dict().items()})


def _same(h1, h2):
    assert h1[0] == h2[0] and h1[1] == h2[1] and torch.equal(h1[2], h2[2])
    assert all(torch.equal(h1[3][k], h2[3][k]) for k in h1[3])


def test_determinism_same_seed_same_state():
    w1, w2 = world(seed=3), world(seed=3)
    for _ in range(3):
        one_bout(w1)
        one_bout(w2)
    _same(_hash(w1), _hash(w2))


def test_exact_resume_on_the_cpu():
    w = world(seed=5)
    for _ in range(2):
        one_bout(w)
    saved = dict(items=it.state_dict(w["items"]), fires=it.state_dict(w["fires"]), held=w["held"].clone(),
                 ledger=w["ledger"].state_dict(), pos=w["pos"].clone(), heading=w["heading"].clone(),
                 gen=w["gen"].get_state())
    for _ in range(2):
        one_bout(w)
    after = _hash(w)
    g = torch.Generator()
    g.set_state(saved["gen"])
    r = dict(items=it.from_state(saved["items"]), fires=it.from_state(saved["fires"]), held=saved["held"].clone(),
             ledger=mp.ItemLedger.from_state(saved["ledger"]), pos=saved["pos"].clone(),
             heading=saved["heading"].clone(), mass=w["mass"], gen=g, peak_w_kg=w["peak_w_kg"], ground=w["ground"])
    for _ in range(2):
        one_bout(r)
    _same(after, _hash(r))


def test_contact_hash_matches_brute_force():
    g = torch.Generator().manual_seed(11)
    A, N, M = 3, 300, 400
    q = torch.remainder(L - 5.0 + 10.0 * torch.rand(A, N, 2, generator=g), L)       # across the boundary
    t = torch.remainder(L - 5.0 + 10.0 * torch.rand(A, M, 2, generator=g), L)
    reach = 0.05 + 0.4 * torch.rand(A, N, generator=g)
    rad = 0.3 * torch.rand(A, M, generator=g)
    qm = torch.rand(A, N, generator=g) < 0.8
    tm = torch.rand(A, M, generator=g) < 0.7
    d = mp.wrap(t[:, None, :, :] - q[:, :, None, :], L).norm(dim=-1)
    ok = qm[..., None] & tm[:, None, :] & (d <= reach[..., None] + rad[:, None, :])
    gg = torch.where(ok, d - rad[:, None, :], torch.full_like(d, float("inf")))
    gmin, j = gg.min(-1)
    want = torch.where(torch.isfinite(gmin), j, torch.full_like(j, -1))
    for scan in (4096, 8):                                          # one pass, and the second pass on overflow
        idx, gap, over = mp.contact(q, qm, reach, t, tm, rad, L, scan=scan)
        assert over.any() == (scan == 8)
        assert torch.equal(idx, want)
        assert torch.allclose(torch.where(torch.isfinite(gap), gap, 0), torch.where(torch.isfinite(gmin), gmin, 0))
    assert (idx >= 0).sum() > 50                                     # the test has many contacts


def test_contact_budget_reports_overflow_and_still_finds_the_nearest():
    A, N, M = 1, 4, 200
    q = torch.full((A, N, 2), 10.0)
    t = torch.full((A, M, 2), 10.0) + 0.01 * torch.arange(M).float()[None, :, None]
    idx, gap, over = mp.contact(q, torch.ones(A, N, dtype=torch.bool), torch.full((A, N), 0.5), t,
                                torch.ones(A, M, dtype=torch.bool), torch.zeros(A, M), L, scan=64)
    assert over.all() and (idx == 0).all()


def test_a_crowded_patch_bout_is_fast():
    w = world(A=4, N=2048, I=8192, F=64, seed=1, area_m=400.0)
    one_bout(w)                                                       # warm-up
    t0 = time.perf_counter()
    one_bout(w)
    assert time.perf_counter() - t0 < 5.0


def test_grip_release_and_press_are_rates_so_a_day_looks_the_same_at_any_bout_length():
    """Review H3: the discrete abilities are continuous-time events (EVENT_RULE). A grip ends a bout closed or open
    with the two-state process's probabilities, and over a day the chance of a press, and the share of grips that end
    the day closed, do not depend on how many bouts the day has (the old per-bout Bernoulli made them bouts x p)."""
    A, N = 1, 40000
    gen = torch.Generator().manual_seed(3)
    alive = torch.ones(A, N, dtype=torch.bool)
    out = {"grip": torch.full((A, N, 2), 0.4), "release": torch.full((A, N), 0.1), "place": torch.full((A, N), 0.3),
           "press": torch.full((A, N), 2e-3)}
    day = 86400.0
    res = {}
    for bouts in (4, 24):
        held = torch.full((A, N, 2), -1, dtype=torch.long)
        pressed = torch.zeros(A, N, dtype=torch.bool)
        for _ in range(bouts):
            ev = mp.events(out, alive, gen, dt_s=day / bouts, held=held)
            assert not (ev["grip"] & (held >= 0)).any() and not (ev["open"] & (held < 0)).any()
            held = torch.where(ev["grip"], 1, torch.where(ev["open"], -1, held))      # stand-in items
            pressed |= ev["press"]
        res[bouts] = ((held >= 0).double().mean((0, 1)), pressed.double().mean())
    # the left grip is the one place opens whenever it holds: c / (c + o) = 0.4 / 0.8; the right one sits between
    # 0.4 / 0.8 and 0.4 / 0.5 (place opens it when the left is empty); the same for 4 and 24 bouts
    for bouts in (4, 24):
        assert float(res[bouts][0][0]) == pytest.approx(0.5, abs=0.02)
        assert 0.5 < float(res[bouts][0][1]) < 0.8
        assert float(res[bouts][1]) == pytest.approx(1 - math.exp(-2e-3 * day / mp.PRESS_S), abs=0.02)
    assert float(res[4][0][1]) == pytest.approx(float(res[24][0][1]), abs=0.02)
