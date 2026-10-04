"""Tests of the v3 patch hydrology: the depression hierarchy (fill-spill-merge), the conditioning of the synthetic
detail, the lake share against Earth's and a temperate year (PLANET-V3-SPEC sections 1 and 7: fresh water exists
where it physically accumulates)."""
from __future__ import annotations

import math

import pytest
import torch

from haishool.life9.planet import globe as gb
from haishool.life9.v3 import patch as pt

R_781 = 5.674e6


def level_checks(geom, pond, tol=1e-6):
    """Standing water only in depressions; 8-adjacent wet cells share one surface; a dry cell next to a wet one is
    not below that surface."""
    H = geom.hydro
    z = H["z"]
    p = pond.double().reshape(z.shape)
    assert float(p[H["band"].reshape(z.shape) < 0].abs().max()) == 0.0
    wet = p > 0
    surf = z + p / pt.RHO_W
    for dx, dy in pt.OFFSETS8:
        wn = torch.roll(wet, (dx, dy), (1, 2))
        both = wet & wn
        if bool(both.any()):
            assert float((surf - torch.roll(surf, (dx, dy), (1, 2)))[both].abs().max()) < tol
        edge = wet & ~wn
        if bool(edge.any()):
            assert float((surf - torch.roll(z, (dx, dy), (1, 2)))[edge].max()) < tol


def _nested(n=48, L=768.0):
    """A bowl (rim about 100 m) on a plain at 80 m holding two pits: A (bottom 86.2 m) and B (bottom 82.3 m) under a
    sill between them at 93.2 m; the bowl spills at 99.5 m."""
    c = (torch.arange(n, dtype=torch.float64) + 0.5) * L / n
    x, y = torch.meshgrid(c, c, indexing="ij")

    def g(x0, y0, s):
        return torch.exp(-(pt.wrap(x - x0, L) ** 2 + pt.wrap(y - y0, L) ** 2) / (2 * s * s))

    r = torch.sqrt(pt.wrap(x - L / 2, L) ** 2 + pt.wrap(y - L / 2, L) ** 2)
    z = 80.0 + 20.0 * torch.exp(-((r - 220.0) / 60.0) ** 2)
    z = torch.where(r < 220.0, 100.0 - 6.0 * (1 - (r / 220.0) ** 2), z)
    z = z - 9.0 * g(L / 2 - 90.0, L / 2, 35.0) - 13.0 * g(L / 2 + 90.0, L / 2, 35.0) + 1e-6 * x / L
    return pt.geometry_from_elevation(z[None], L, -1e4, R_781)


def test_nested_pits_fill_and_spill_before_they_merge():
    """Water poured on pit A only stays in A until A is full to its sill; then it spills into B; A and B share one
    level only above their sill (fill-spill-merge, Barnes et al. 2020)."""
    geom = _nested()
    H = geom.hydro
    n = geom.n
    z = H["z"][0]
    ia, ib, j = int((geom.L / 2 - 90.0) / geom.dx), int((geom.L / 2 + 90.0) / geom.dx), n // 2
    band = H["band"].reshape(n, n)
    a_leaf = int(H["label"].reshape(n, n)[ia, j])
    b_leaf = int(H["label"].reshape(n, n)[ib, j])
    assert a_leaf != b_leaf and a_leaf < H["leaves"] and b_leaf < H["leaves"]
    sill = float(H["spill"][a_leaf])
    assert sill == pytest.approx(float(H["spill"][b_leaf]))
    parent = int(H["parent"][a_leaf])
    assert parent == int(H["parent"][b_leaf]) and parent >= 0
    top_spill = float(H["spill"][parent])
    assert float(z[ia, j]) < sill < top_spill
    in_a = band == a_leaf
    in_b = band == b_leaf
    pond = torch.zeros(1, n, n, dtype=torch.float64)
    total = 0.0
    seen = {"a_full_before_b": False, "merged": False}
    for day in range(400):
        s = torch.zeros(1, n, n, dtype=torch.float64)
        s[0, ia - 2:ia + 3, j - 2:j + 3] = 50.0
        total += float(s.sum())
        pond, out = pt.route(s, pond, geom)
        assert float(pond.sum()) + float(out.sum()) == pytest.approx(total, rel=1e-12)
        total -= float(out.sum())
        level_checks(geom, pond)
        p = pond[0]
        surf = z + p / pt.RHO_W
        wa, wb = float(p[in_a].sum()), float(p[in_b].sum())
        if float(surf[ia, j]) > sill + 1e-6:
            # above the sill A and B are one lake
            seen["merged"] = True
            assert float(surf[ib, j]) == pytest.approx(float(surf[ia, j]), abs=1e-9)
        elif wb > 0:
            # B gets water only once A stands at its sill, and stays below it
            assert float(surf[ia, j]) == pytest.approx(sill, abs=1e-9)
            assert float(surf[ib, j]) <= sill + 1e-9
            seen["a_full_before_b"] = True
        if day == 20:
            assert wa > 0 and wb == 0.0                               # A alone holds the water at first
    assert seen["a_full_before_b"] and seen["merged"]


def test_hundreds_of_sub_basins_keep_their_own_water():
    """Raw (unconditioned) Gaussian detail: hundreds of nested pits. Water poured into one pit's watershed only ever
    wets that pit (or what it spills into), each wet patch stands level, and nothing is lost."""
    gen = torch.Generator().manual_seed(21)
    z = pt.fractal_detail(2, 64, gen, 0.8).double() * 8.0 + 200.0
    geom = pt.geometry_from_elevation(z, 1024.0, -1e4, R_781)
    H = geom.hydro
    assert H["leaves"] > 60 and H["depth"] > 3
    lab = H["label"].reshape(2, 64, 64)
    # pour a little into one mid-sized leaf's watershed: only that leaf's own cells get wet
    caps = H["cap"][:H["leaves"]]
    leaf = int(torch.argsort(caps, descending=True)[H["leaves"] // 4])
    s = torch.where(lab == leaf, torch.full_like(z, 0.2 * float(caps[leaf]) / int((lab == leaf).sum())),
                    torch.zeros_like(z))
    pond, out = pt.route(s, torch.zeros_like(z), geom)
    wet = pond.reshape(-1) > 0
    assert bool(wet.any()) and bool((H["band"][wet] == leaf).all())
    assert float(pond.sum() + out.sum()) == pytest.approx(float(s.sum()), rel=1e-12)
    level_checks(geom, pond)
    # uneven random rain for many days, re-levelled each day
    pond = torch.zeros_like(z)
    total = 0.0
    for day in range(30):
        s = torch.rand(z.shape, generator=gen, dtype=torch.float64) * 20.0
        total += float(s.sum())
        pond, out = pt.route(s, pond, geom)
        total -= float(out.sum())
        level_checks(geom, pond)
        assert float(pond.sum()) == pytest.approx(total, rel=1e-11)


def test_conditioning_fills_only_shallow_depressions():
    gen = torch.Generator().manual_seed(5)
    rms = 10.0
    z = pt.fractal_detail(4, 64, gen, 0.8).double() * rms + 300.0
    sea = torch.zeros(4, 64, 64, dtype=torch.bool)
    keep = 0.4 * rms
    zc, rounds = pt.condition_terrain(z, sea, keep)
    assert 1 <= rounds < pt.PatchRules().condition_rounds
    assert bool((zc >= z).all())                                        # filling never lowers the ground
    assert torch.equal(zc.reshape(4, -1).argmin(-1), z.reshape(4, -1).argmin(-1))   # the outlet stays
    H = pt.drainage(zc, sea)
    raw = pt.drainage(z, sea)
    assert H["leaves"] < raw["leaves"] / 10
    # every depression left is at least keep deep (from its lowest ground to its sill)
    if H["nodes"]:
        assert float((H["spill"] - H["bottom"]).min()) >= keep - 1e-9
    # filling touches only cells inside the raw terrain's depressions and never rises above their spill level
    changed = (zc > z).reshape(-1)
    assert bool((raw["band"][changed] >= 0).all())
    assert bool((zc <= raw["spill_m"].double() + 1e-3).all())
    assert float(H["lake_share"].mean()) < float(raw["lake_share"].mean())


def test_lake_share_on_realistic_terrain_is_earth_like():
    """make_geometry on version 2's terrain: about 3.7 % of Earth's non-glaciated land is lake (Verpoorter et al.
    2014); the conditioned detail leaves a few per cent in lake basins (most patches few, some many)."""
    globe = gb.Globe(12)
    relief = torch.tensor([9000.0, 12000.0])
    water = 0.143 * 0.7 * 4 / 3 * math.pi * ((R_781 + relief) ** 3 - R_781 ** 3)
    terrain = gb.make_terrain(globe, 2, torch.Generator().manual_seed(1), relief, water, R_781)
    geom = pt.make_geometry(globe, terrain["elevation_m"], terrain["sea_level_m"], terrain["land"], R_781,
                            torch.Generator().manual_seed(3), patches=24)
    share = pt.lake_share(geom)
    assert geom.A == 48 and geom.n == 128
    assert 0.015 < float(share.mean()) < 0.07
    assert float(share.median()) < float(share.mean())                 # skewed: lake districts and dry patches
    assert float(share.max()) < 0.5
    # the scale-free rule gives the same statistics on synthetic detail of any relief
    gen = torch.Generator().manual_seed(8)
    z = pt.fractal_detail(48, 128, gen, 0.8).double()
    shares = []
    for rms in (2.0, 30.0):
        g = pt.geometry_from_elevation(z * rms + 500.0, 2048.0, -1e4, R_781, condition_depth_m=0.4 * rms)
        shares.append(pt.lake_share(g))
    assert torch.allclose(shares[0], shares[1], atol=1e-12)
    assert 0.015 < float(shares[0].mean()) < 0.07


def test_a_temperate_year_leaves_water_in_a_few_per_cent_of_the_land():
    """One year at 288 K with 2.7 kg/m^2 of rain a day: standing water (over 1 cm) covers a few per cent of the land
    (Earth: 3.7 % lakes), deep enough to drink from, and the patches drain."""
    gen = torch.Generator().manual_seed(5)
    rms = torch.tensor([1.0, 4.0, 12.0, 30.0, 12.0, 12.0, 4.0, 1.0], dtype=torch.float64)
    z = pt.fractal_detail(8, 128, gen, 0.8).double() * rms[:, None, None] + 300.0
    geom = pt.geometry_from_elevation(z, 2048.0, -1e4, R_781, condition_depth_m=0.4 * rms,
                                      detail_rms_m=rms.float())
    st = pt.init_state(geom)
    f = pt.constant_forcing(geom, t_air_k=288.0, precip_kg_m2=2.7, vapour_kg_m2=18.0, sw_day_w_m2=180.0)
    st, d = pt.run(st, geom, f, 365)
    share = pt.water_share(st, geom)
    assert 0.003 < float(share.mean()) < 0.08
    assert bool((share <= pt.lake_share(geom) + 1e-12).all())           # water stands only in lake basins
    wet = st.pond > 10.0
    depth = st.pond[wet].double().mean() / pt.RHO_W
    assert 0.05 < float(depth) < 10.0
    assert float(st.w_out.min()) > 0                                    # every patch drains
    assert float(pt.water_ledger(st, geom).abs().max()) < 1e-9 * float(pt.water_stores(st, geom).max())
    level_checks(geom, st.pond, tol=0.15)


def test_ponds_seep_to_the_deep_drainage():
    geom = _nested(n=32, L=512.0)
    H = geom.hydro
    assert H["nodes"] >= 1
    n = geom.n
    pond, _ = pt.route(torch.full((1, n, n), 300.0, dtype=torch.float64), torch.zeros(1, n, n, dtype=torch.float64),
                       geom)
    st = pt.init_state(geom)
    st.pond = pond.float()
    st.soil = torch.full_like(st.soil, 150.0)
    st = pt.reset_ledgers(st, geom)
    wet = st.pond > 2.0
    f = pt.constant_forcing(geom, t_air_k=290.0, precip_kg_m2=0.0, vapour_kg_m2=500.0, sw_day_w_m2=0.0)
    st2, d = pt.step(st, geom, f)
    assert torch.allclose(d["seepage"][wet], torch.full_like(d["seepage"][wet], 2.0))
    assert float(st2.w_out[0]) == pytest.approx(float(d["seepage"].double().sum()) * geom.cell_m2, rel=1e-6)
    assert float(d["evap_physical_kg"][0]) == 0.0                     # a saturated column: no evaporation
    bound = float(((st2.soil.double() + st2.pond.double()).abs() * 2 ** -24).sum()) * geom.cell_m2
    assert abs(float(st2.w_evap[0])) <= bound                          # only the float32 cast's rounding
    assert float(pt.water_ledger(st2, geom).abs().max()) < 1e-6
