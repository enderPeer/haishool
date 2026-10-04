"""Tests of the v3 habitat patches: geometry, light, line of sight, neighbour search, smells and the optics table
(PLANET-V3-SPEC sections 1, 4 and 7)."""
from __future__ import annotations

import math
import re
from dataclasses import fields
from pathlib import Path

import pytest
import torch

from haishool.life9.planet import formation as F
from haishool.life9.planet import globe as gb
from haishool.life9.planet import materials
from haishool.life9.v3 import optics
from haishool.life9.v3 import patch as pt

TAGS = {"chain", "derived", "reference", "new_rule"}
R_781 = 5.674e6
ROOT = Path(__file__).resolve().parents[3]


def _tag_ok(tag: str) -> bool:
    return bool(tag) and all(part in TAGS for part in tag.split("+"))


def _world_terrain(W=2, G=12, seed=1):
    globe = gb.Globe(G)
    relief = torch.tensor([9000.0, 12000.0][:W])
    water = 0.143 * 0.7 * 4 / 3 * math.pi * ((R_781 + relief) ** 3 - R_781 ** 3)
    terrain = gb.make_terrain(globe, W, torch.Generator().manual_seed(seed), relief, water, R_781)
    return globe, terrain


# ------------------------------------------------------------------------------------------------ provenance
def test_rules_have_provenance():
    for f in fields(pt.PatchRules):
        tag, note = pt.PROVENANCE[f.name]
        assert note and _tag_ok(tag), f.name
    for name, (tag, note) in pt.PROVENANCE.items():
        assert note and _tag_ok(tag), name


def test_no_forbidden_mechanisms_in_my_files():
    """The section-9 words must not appear in the patch modules (the package-wide grep is the gate)."""
    banned = ("seed_floor", "innate", "forage_bias", "background_drink", "time_budget", "pedigree", "relatedness",
              "LIFESPAN", "MATURITY", "reward")
    for rel in ("haishool/life9/v3/patch.py", "haishool/life9/v3/optics.py"):
        text = (ROOT / rel).read_text(encoding="utf-8")
        for word in banned:
            assert re.search(word, text, re.IGNORECASE) is None, (rel, word)


# ------------------------------------------------------------------------------------------------ optics
def test_optics_table_covers_every_species_and_tissue():
    for s in materials.SPECIES:
        for table in (optics.VIS, optics.SW, optics.EMISSIVITY):
            v = table[f"species:{s}"]
            assert 0.0 <= float(v) <= 1.0 and v.source and _tag_ok(v.tag), s
    for name in ("plant", "litter", "water", "snow", "ice", "soil"):
        assert f"ground:{name}" in optics.VIS
    for name in ("fur", "skin", "flesh", "fat", "bone"):
        assert f"tissue:{name}" in optics.VIS
    for key, (tag, source) in optics.provenance().items():
        assert source and _tag_ok(tag), key
    # plants dark, water dark, sand and snow bright, charcoal darkest
    v = optics.vis
    assert v("ground:plant") < v("species:sand") < v("ground:snow")
    assert v("ground:water") < v("species:granite")
    assert v("species:charcoal") < v("species:basalt") < v("species:limestone")
    assert optics.species_vector().shape == (materials.S,)


def test_mix_and_fire_emission():
    shares = torch.zeros(2, materials.S)
    shares[0, materials.IDX["sand"]] = 1.0
    shares[1, materials.IDX["sand"]] = 1.0
    shares[1, materials.IDX["charcoal"]] = 1.0
    out = optics.mix(shares)
    assert float(out[0]) == pytest.approx(optics.vis("species:sand"))
    assert float(out[1]) == pytest.approx(0.5 * (optics.vis("species:sand") + optics.vis("species:charcoal")))
    # the visible share is the same Planck integral as formation's
    assert float(optics.visible_share(torch.tensor(5772.0))) == pytest.approx(F.par_fraction(5772.0), rel=1e-6)
    assert float(optics.visible_share(torch.tensor(1100.0))) == pytest.approx(F.band_fraction(1100.0, 400e-9, 700e-9),
                                                                               rel=1e-6)
    e = optics.visible_exitance(torch.tensor([300.0, 1100.0, 1300.0]), optics.FLAME_EMISSIVITY)
    assert float(e[0]) < 1e-15 and 0.1 < float(e[1]) < 1.0 < float(e[2])


# ------------------------------------------------------------------------------------------------ patch cells
def test_choose_patch_cells_by_area_over_land():
    C = 6
    land = torch.tensor([[False, True, False, True, False, False], [False] * C])
    area = torch.tensor([1.0, 1.0, 1.0, 3.0, 1.0, 1.0])
    cells = pt.choose_patch_cells(None, land, area, torch.Generator().manual_seed(3), 4000)
    assert cells.shape == (2, 4000)
    assert bool(land[0, cells[0]].all())
    share = float((cells[0] == 3).double().mean())
    assert share == pytest.approx(0.75, abs=0.03)
    # a world without land draws over all its cells
    assert len(torch.unique(cells[1])) == C
    again = pt.choose_patch_cells(None, land, area, torch.Generator().manual_seed(3), 4000)
    assert torch.equal(cells, again)


# ------------------------------------------------------------------------------------------------ terrain
def test_fractal_detail_is_periodic_with_the_stated_spectrum():
    A, n, H = 16, 64, 0.8
    z = pt.fractal_detail(A, n, torch.Generator().manual_seed(5), H).double()
    assert z.shape == (A, n, n)
    assert float(z.mean((1, 2)).abs().max()) < 1e-5
    assert float(z.std((1, 2), unbiased=False).sub(1).abs().max()) < 1e-4
    # periodic: the step across the boundary is like any other step
    inner = (z[:, 1:] - z[:, :-1]).pow(2).mean().sqrt()
    seam = (z[:, 0] - z[:, -1]).pow(2).mean().sqrt()
    assert float(seam / inner) == pytest.approx(1.0, abs=0.3)
    inner_y = (z[:, :, 1:] - z[:, :, :-1]).pow(2).mean().sqrt()
    seam_y = (z[:, :, 0] - z[:, :, -1]).pow(2).mean().sqrt()
    assert float(seam_y / inner_y) == pytest.approx(1.0, abs=0.3)
    # radial power spectrum ~ q^-2(H+1)
    p = torch.fft.fft2(z).abs().pow(2).mean(0)
    k = torch.fft.fftfreq(n, d=1.0 / n, dtype=torch.float64)
    q = torch.sqrt(k[:, None] ** 2 + k[None, :] ** 2)
    xs, ys = [], []
    for lo in range(2, 24, 2):
        m = (q >= lo) & (q < lo + 2)
        xs.append(math.log(float(q[m].mean())))
        ys.append(math.log(float(p[m].mean())))
    xs, ys = torch.tensor(xs), torch.tensor(ys)
    slope = float(((xs - xs.mean()) * (ys - ys.mean())).sum() / ((xs - xs.mean()) ** 2).sum())
    assert slope == pytest.approx(-2 * (H + 1), abs=0.25)


def test_make_geometry_ties_detail_to_the_planets_relief():
    globe, terrain = _world_terrain()
    gen = torch.Generator().manual_seed(7)
    rules = pt.PatchRules(cells=32)
    geom = pt.make_geometry(globe, terrain["elevation_m"], terrain["sea_level_m"], terrain["land"], R_781, gen,
                            patches=3, rules=rules)
    assert geom.A == 6 and geom.n == 32 and geom.L == 2048.0
    assert torch.equal(geom.world, torch.tensor([0, 0, 0, 1, 1, 1]))
    assert bool(terrain["land"][geom.world, geom.cell].all())
    # the raw detail (same draws) has the derived rms on the cell's elevation; conditioning only fills depressions
    D, lag = pt.relief_structure(globe, terrain["elevation_m"], R_781)
    want = D * pt.structure_scale(lag, 2048.0, rules) / math.sqrt(2)
    g2 = torch.Generator().manual_seed(7)
    cells = pt.choose_patch_cells(globe, terrain["land"], globe.area64, g2, 3).reshape(-1)
    assert torch.equal(cells, geom.cell)
    base = terrain["elevation_m"][geom.world, geom.cell].double()
    rms = want[geom.world, geom.cell]
    raw = base[:, None, None] + pt.fractal_detail(6, 32, g2, 0.8).double() * rms.float().double()[:, None, None]
    assert torch.allclose(raw.sub(base[:, None, None]).pow(2).mean((1, 2)).sqrt(), rms, rtol=1e-3)
    assert torch.allclose(geom.detail_rms_m.double(), rms, rtol=1e-5)
    z = geom.hydro["z"]
    assert bool((z >= raw - 1e-6).all()) and bool((z > raw + 1e-6).any())
    assert torch.equal(geom.elev, z.float())
    # filling stays a small share of the relief
    assert float((z - raw).pow(2).mean().sqrt()) < 0.3 * float(rms.mean())
    # the lag is the global cell spacing at the real radius
    assert float(lag.mean()) == pytest.approx(R_781 * math.pi / 2 / 12, rel=0.15)
    # the same generator state gives the same patches
    again = pt.make_geometry(globe, terrain["elevation_m"], terrain["sea_level_m"], terrain["land"], R_781,
                             torch.Generator().manual_seed(7), patches=3, rules=rules)
    assert torch.equal(geom.elev, again.elev) and torch.equal(geom.cell, again.cell)


def test_detail_rms_follows_the_two_regime_structure_function():
    r = pt.PatchRules()
    lag = torch.tensor([290e3, 4e3, 1e3], dtype=torch.float64)
    s = pt.structure_scale(lag, 2048.0, r)
    # Brownian (H 0.5) from the global lag down to the 5 km break, H 0.8 below it
    assert float(s[0]) == pytest.approx((5e3 / 290e3) ** 0.5 * (2048.0 / 5e3) ** 0.8, rel=1e-12)
    assert float(s[1]) == pytest.approx((2048.0 / 4e3) ** 0.8, rel=1e-12)           # lag below the break
    assert float(s[2]) == pytest.approx((2048.0 / 1e3) ** 0.8, rel=1e-12)           # lag below the patch
    # an Earth-like neighbour difference (about 300 m between 290 km cells) gives Earth's hill-scale relief:
    # a 2 km window holds metres (plains) to tens of metres (hills) of rms relief
    rms = 300.0 * float(s[0]) / math.sqrt(2)
    assert 5.0 < rms < 40.0
    # the single H = 0.8 law shrank it 3-4 fold
    single = 300.0 * (2048.0 / 290e3) ** 0.8 / math.sqrt(2)
    assert rms > 3.0 * single
    globe, terrain = _world_terrain()
    D, lag_g = pt.relief_structure(globe, terrain["elevation_m"], R_781)
    assert torch.allclose(pt.detail_rms(globe, terrain["elevation_m"], R_781, r),
                          D * pt.structure_scale(lag_g, 2048.0, r) / math.sqrt(2))


def test_detail_rms_on_formation3_real_scale_relief_is_earth_like():
    """On formation3's real-scale terrain (Earth's model and planet 781) the derived 2 km rms is metres to tens of
    metres, as Earth's land within 2 km windows (plains a few metres, hills tens)."""
    try:
        from haishool.life9.planet import chain
        from haishool.life9.v3 import formation3 as F3
        specs = [F3.build_planet3(chain.chain_inputs(781), 781), F3.model_earth()]
        globe = gb.Globe(16)
        terrain = F3.make_terrain3(globe, torch.Generator().manual_seed(1), specs)
    except (ImportError, AttributeError, TypeError) as exc:                # formation3's own API, not this module
        pytest.skip(f"formation3 terrain not available: {exc!r}")
    radius = torch.tensor([s.radius_m for s in specs], dtype=torch.float64)
    rms = pt.detail_rms(globe, terrain["elevation_m"], radius)
    for w in range(2):
        land = terrain["land"][w]
        med = float(rms[w][land].median())
        assert 3.0 < med < 40.0, (w, med)


def test_slope_aspect_and_normal():
    n, L = 32, 640.0
    x = (torch.arange(n) + 0.5) * L / n
    amp = 20.0
    z = (amp * torch.sin(2 * math.pi * x / L))[None, :, None].expand(1, n, n)
    geom = pt.geometry_from_elevation(z, L, -1e4, R_781)
    dzdx = amp * 2 * math.pi / L * torch.cos(2 * math.pi * x / L)
    # centred differences of a sine: exact up to the factor sin(k dx) / (k dx)
    kdx = 2 * math.pi / n
    assert torch.allclose(geom.dzdx[0, :, 0].double(), (dzdx * math.sin(kdx) / kdx).double(), atol=1e-4)
    assert float(geom.dzdy.abs().max()) < 1e-6
    # rising to the east (dz/dx > 0) faces west (aspect 3 pi / 2), falling faces east (pi / 2)
    rising, falling = int(n * 0.0), int(n * 0.5)
    assert float(geom.aspect[0, rising, 0]) == pytest.approx(1.5 * math.pi, abs=1e-5)
    assert float(geom.aspect[0, falling, 0]) == pytest.approx(0.5 * math.pi, abs=1e-5)
    assert torch.allclose(geom.normal.norm(dim=-1), torch.ones(1, n, n), atol=1e-6)
    assert torch.allclose(torch.tan(geom.slope), geom.dzdx.abs(), atol=1e-6)


# ------------------------------------------------------------------------------------------------ light
def _ridge(n=32, L=640.0, amp=60.0, lat=0.0):
    y = (torch.arange(n) + 0.5) * L / n
    z = (amp * torch.sin(2 * math.pi * y / L))[None, None, :].expand(1, n, n)
    up = (math.cos(lat), 0.0, math.sin(lat))
    north = (-math.sin(lat), 0.0, math.cos(lat))
    east = (0.0, 1.0, 0.0)
    basis = tuple(torch.tensor([v]) for v in (east, north, up))
    return pt.geometry_from_elevation(z, L, -1e4, R_781, basis=basis)


def test_flat_patch_gets_the_global_cells_light_and_nights_are_dark():
    geom = pt.geometry_from_elevation(torch.zeros(1, 16, 16), 256.0, -1e4, R_781)
    f = pt.constant_forcing(geom, sw_day_w_m2=180.0, declination=0.0, locked=False)
    day_sw, bout_sw = pt.light(geom, f)
    assert torch.allclose(day_sw, torch.full_like(day_sw, 180.0), rtol=1e-5)
    # version 2's phase (world.sun): the sun stands over lon 0 at t = 0, so at lon 0 the first bout is the
    # afternoon, the second and third are night and the last is the morning
    assert float(bout_sw[:, 1].abs().max()) == 0.0 and float(bout_sw[:, 2].abs().max()) == 0.0
    assert float(bout_sw[:, 0].min()) > 0 and float(bout_sw[:, 3].min()) > 0
    assert torch.allclose(bout_sw[:, 0], bout_sw[:, 3], rtol=1e-5)          # symmetric about noon
    sun0 = pt.sun_local(geom, 0.0, False, 0, 4, 8)
    assert float(sun0[0, 0, 0, 2]) > 0.99                                   # just after noon
    # a locked planet's sun stands still: every bout alike
    f_l = pt.constant_forcing(geom, sw_day_w_m2=300.0, locked=True)
    _, bout_l = pt.light(geom, f_l)
    assert torch.allclose(bout_l, torch.full_like(bout_l, 300.0), rtol=1e-5)
    # a solar day of two sim days: day 0 is the afternoon and evening, day 1 the night and the morning
    sun = pt.sun_local(geom, 0.0, False, 0, 4, 8, solar_day_s=2 * pt.DAY_S)
    assert float(sun[0, 0, :, 2].min()) > 0.0 and float(sun[0, 3, :, 2].max()) <= 0.0
    sun1 = pt.sun_local(geom, 0.0, False, 1, 4, 8, solar_day_s=2 * pt.DAY_S)
    assert float(sun1[0, 0, :, 2].max()) <= 0.0 and float(sun1[0, 3, :, 2].min()) > 0.0


@pytest.mark.parametrize("hours", [12.0, 30.0, 36.0, 48.0])
def test_light_energy_over_many_days_for_any_solar_day(hours):
    """High latitude in winter (lat 60, lon 90, declination -20 deg): on a long solar day whole sim days are dark and
    others catch only a sliver of daylight; the instantaneous light is the physical one, so the mean over whole solar
    days is the climate's daily mean (no energy lost, none packed into a dawn)."""
    lat, lon = math.radians(60.0), math.radians(90.0)
    up = (math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat))
    east = (-math.sin(lon), math.cos(lon), 0.0)
    north = (-math.sin(lat) * math.cos(lon), -math.sin(lat) * math.sin(lon), math.cos(lat))
    basis = tuple(torch.tensor([v]) for v in (east, north, up))
    geom = pt.geometry_from_elevation(torch.zeros(1, 8, 8), 128.0, -1e4, R_781, basis=basis)
    dec, sd, sw = math.radians(-20.0), hours * 3600.0, 50.0
    days = {12.0: 2, 30.0: 20, 36.0: 6, 48.0: 4}[hours]                  # whole solar days
    daily, peak = [], 0.0
    for d in range(days):
        f = pt.constant_forcing(geom, day=d, sw_day_w_m2=sw, declination=dec, solar_day_s=sd)
        day_sw, bout_sw = pt.light(geom, f)
        daily.append(float(day_sw.double().mean()))
        peak = max(peak, float(bout_sw.max()))
    mean = sum(daily) / len(daily)
    assert mean == pytest.approx(sw, rel=1e-5)
    if hours == 48.0:
        assert min(daily) == 0.0                                            # a whole dark sim day
    if hours == 36.0:
        assert 0.0 < min(daily) < 0.5 * sw                                  # a sliver of daylight, not a day's worth
    # the instantaneous flux never exceeds the noon flux of the physical sun
    f = pt.constant_forcing(geom, sw_day_w_m2=sw, declination=dec, solar_day_s=sd)
    noon_mu = math.cos(lat - dec)
    assert peak <= sw / float(f.flat_mu) * noon_mu * (1 + 1e-5)


def test_slopes_facing_the_sun_get_more_light():
    lat = math.radians(45.0)
    geom = _ridge(lat=lat)
    f = pt.constant_forcing(geom, sw_day_w_m2=200.0, declination=0.0)
    day_sw, _ = pt.light(geom, f)
    south = geom.aspect.sub(math.pi).abs() < 0.3
    north = (geom.aspect < 0.3) | (geom.aspect > 2 * math.pi - 0.3)
    steep = geom.slope > 0.3
    assert float(day_sw[south & steep].mean()) > 1.3 * float(day_sw[north & steep].mean())
    # on the equator at equinox the ridge's two faces are alike
    geom0 = _ridge(lat=0.0)
    d0, _ = pt.light(geom0, pt.constant_forcing(geom0, sw_day_w_m2=200.0))
    s0 = geom0.aspect.sub(math.pi).abs() < 0.3
    n0 = ((geom0.aspect < 0.3) | (geom0.aspect > 2 * math.pi - 0.3)) & (geom0.slope > 0.3)
    assert float(d0[s0 & (geom0.slope > 0.3)].mean()) == pytest.approx(float(d0[n0].mean()), rel=1e-3)
    # diffuse light flattens the contrast
    fd = pt.constant_forcing(geom, sw_day_w_m2=200.0, declination=0.0, diffuse_share=0.6)
    dd, _ = pt.light(geom, fd)
    ratio_beam = float(day_sw[south & steep].mean()) / float(day_sw[north & steep].mean())
    ratio_mix = float(dd[south & steep].mean()) / float(dd[north & steep].mean())
    assert 1.0 < ratio_mix < ratio_beam


def test_light_is_per_horizontal_area_and_conserved_on_rough_ground():
    """With the sun overhead no slope faces away from it: the patch mean of the light per horizontal m^2 is the
    flat ground's (beam: s_z - z_x s_x - z_y s_y has the periodic mean s_z; diffuse: normalised view factor)."""
    n, L = 32, 512.0
    x = (torch.arange(n) + 0.5) * L / n
    z = 40.0 * torch.sin(2 * math.pi * x / L)[:, None] * torch.cos(2 * math.pi * x / L)[None, :]
    geom = pt.geometry_from_elevation(z[None], L, -1e4, R_781)
    assert float(geom.slope.max()) > 0.3
    for fd in (0.0, 0.4):
        f = pt.constant_forcing(geom, sw_day_w_m2=250.0, locked=True, diffuse_share=fd)   # sun overhead
        day_sw, _ = pt.light(geom, f)
        assert float(day_sw.double().mean()) == pytest.approx(250.0, rel=1e-5)
    # the beam per horizontal m^2 is the flat flux everywhere when the sun is overhead (n.s / n_z = 1)
    beam, _ = pt.light(geom, pt.constant_forcing(geom, sw_day_w_m2=250.0, locked=True))
    assert torch.allclose(beam, torch.full_like(beam, 250.0), rtol=1e-5)
    # the diffuse part favours steep cells per horizontal m^2 (more surface under the sky)
    dif, _ = pt.light(geom, pt.constant_forcing(geom, sw_day_w_m2=250.0, locked=True, diffuse_share=1.0))
    steep, gentle = geom.slope > geom.slope.quantile(0.75), geom.slope < geom.slope.quantile(0.25)
    assert float(dif[steep].mean()) > float(dif[gentle].mean())


def test_erbs_diffuse_share():
    k = torch.tensor([0.1, 0.5, 0.8])
    low = pt.erbs_diffuse_share(k, torch.full((3,), 1.0))
    high = pt.erbs_diffuse_share(k, torch.full((3,), 2.0))
    assert float(low[1]) == pytest.approx(1 - 0.2727 * 0.5 + 2.4495 * 0.25 - 11.9514 * 0.125 + 9.3879 * 0.0625)
    assert float(low[2]) == pytest.approx(0.143) and float(high[2]) == pytest.approx(0.175)
    assert float(high[1]) == pytest.approx(1 + 0.2832 * 0.5 - 2.5557 * 0.25 + 0.8448 * 0.125)
    assert bool((low[:-1] > low[1:]).all())                                 # clearer sky, less diffuse


# ------------------------------------------------------------------------------------------------ line of sight
def test_sagitta_and_bulge():
    assert float(pt.sagitta(1000.0, R_781)) == pytest.approx(1000.0 ** 2 / (2 * R_781), rel=1e-6)
    d = 2.0e6
    assert float(pt.sagitta(d, R_781)) == pytest.approx(R_781 - math.sqrt(R_781 ** 2 - d ** 2), rel=1e-9)
    assert float(pt.bulge(500.0, 500.0, R_781)) == pytest.approx(500.0 ** 2 / (2 * R_781))


def test_line_of_sight_curvature_terrain_and_wrap():
    n, L = 64, 2048.0
    flat = pt.geometry_from_elevation(torch.zeros(1, n, n), L, -1e4, R_781)
    p = torch.tensor([[[100.0, 300.0], [100.0, 300.0], [100.0, 300.0]]])
    q = torch.tensor([[[1100.0, 300.0], [500.0, 300.0], [1100.0, 300.0]]])
    h = torch.tensor([[0.01, 0.01, 1.0]])
    vis = pt.line_of_sight(flat, p, q, h, h)
    # 1 cm eyes 1 km apart: the bulge (2.2 cm) hides them; 400 m apart (3.5 mm) or at 1 m height they see
    assert vis.tolist() == [[False, True, True]]
    # a ridge across x = L / 2 blocks a sight line over it, not the one around the periodic boundary
    x = (torch.arange(n) + 0.5) * L / n
    ridge = (30.0 * torch.exp(-((x - L / 2) / 40.0) ** 2))[None, :, None].expand(1, n, n)
    g2 = pt.geometry_from_elevation(ridge, L, -1e4, R_781)
    p2 = torch.tensor([[[700.0, 500.0], [100.0, 500.0]]])
    q2 = torch.tensor([[[1400.0, 500.0], [1950.0, 500.0]]])
    assert pt.line_of_sight(g2, p2, q2, 1.0, 1.0).tolist() == [[False, True]]
    # high enough eyes see over it
    assert pt.line_of_sight(g2, p2[:, :1], q2[:, :1], 60.0, 60.0).tolist() == [[True]]


def test_line_of_sight_samples_follow_the_distance_and_match_a_dense_reference():
    n, L = 64, 1024.0
    gen = torch.Generator().manual_seed(4)
    z = pt.fractal_detail(2, n, gen, 0.8) * 15.0
    geom = pt.geometry_from_elevation(z, L, -1e4, R_781)
    M = 3000
    p = torch.rand(2, M, 2, generator=gen) * L
    q = p + (torch.rand(2, M, 2, generator=gen) - 0.5) * torch.rand(2, M, 1, generator=gen) * 700.0
    h = torch.rand(2, M, generator=gen) * 4.0
    fast = pt.line_of_sight(geom, p, q, h, h)
    dense = pt.line_of_sight(geom, p, q, h, h, samples=1500)
    agree = float((fast == dense).float().mean())
    assert agree > 0.99 and 0.05 < float(dense.float().mean()) < 0.95       # the rest graze the terrain
    # small chunks give the same answer
    small = pt.line_of_sight(geom, p, q, h, h, rules=pt.PatchRules(sight_chunk=257))
    assert torch.equal(small, fast)


# ------------------------------------------------------------------------------------------------ neighbours
def _brute(pos, alive, L, n):
    A, N = alive.shape
    dx = L / n
    c = torch.floor(torch.remainder(pos, L) / dx).long()
    out = []
    for a in range(A):
        rows = []
        for i in range(N):
            s = set()
            if bool(alive[a, i]):
                for j in range(N):
                    if j == i or not bool(alive[a, j]):
                        continue
                    ddx = (int(c[a, j, 0]) - int(c[a, i, 0])) % n
                    ddy = (int(c[a, j, 1]) - int(c[a, i, 1])) % n
                    if ddx in (0, 1, n - 1) and ddy in (0, 1, n - 1):
                        s.add(j)
            rows.append(s)
        out.append(rows)
    return out


def test_neighbour_search_equals_brute_force():
    gen = torch.Generator().manual_seed(11)
    A, N, n, L = 2, 300, 16, 128.0
    geom = pt.geometry_from_elevation(torch.zeros(A, n, n), L, -1e4, R_781)
    pos = torch.rand(A, N, 2, generator=gen) * L
    pos[0, :20, 0] = torch.rand(20, generator=gen) * 2.0              # crowd the seam
    pos[0, 20:40, 0] = L - torch.rand(20, generator=gen) * 2.0
    alive = torch.rand(A, N, generator=gen) < 0.8
    idx, off, dist, count = pt.neighbours(geom, pos, alive, K=N, max_scan=N)
    brute = _brute(pos, alive, L, n)
    for a in range(A):
        for i in range(N):
            got = {int(j) for j in idx[a, i] if int(j) >= 0}
            assert got == brute[a][i], (a, i)
            assert int(count[a, i]) == len(brute[a][i])
    # offsets are the wrapped displacements, distances their norms, sorted ascending
    valid = idx >= 0
    j = idx.clamp_min(0)
    other = torch.gather(pos, 1, j.reshape(A, -1, 1).expand(-1, -1, 2)).reshape(A, N, N, 2)
    want = pt.wrap(other - pos[:, :, None], L)
    assert torch.allclose(off[valid], want[valid], atol=1e-4)
    assert torch.allclose(dist[valid], want[valid].norm(dim=-1), atol=1e-4)
    assert bool((dist[..., 1:] >= dist[..., :-1]).all())
    assert bool((idx[~alive] == -1).all())
    # every pair closer than one cell is found
    d_all = pt.wrap(pos[:, None, :, :] - pos[:, :, None, :], L).norm(dim=-1)
    close = (d_all < L / n) & alive[:, :, None] & alive[:, None, :] & ~torch.eye(N, dtype=torch.bool)
    for a, i, jj in close.nonzero().tolist():
        assert jj in set(idx[a, i].tolist())
    # with a small K the K nearest are kept
    K = 4
    idx4, _, dist4, _ = pt.neighbours(geom, pos, alive, K=K, max_scan=N)
    assert idx4.shape == (A, N, K)
    assert torch.allclose(dist4, dist[..., :K])


def test_neighbour_search_is_deterministic_and_pads():
    geom = pt.geometry_from_elevation(torch.zeros(1, 8, 8), 64.0, -1e4, R_781)
    pos = torch.tensor([[[1.0, 1.0], [63.0, 63.0], [30.0, 30.0]]])
    alive = torch.tensor([[True, True, True]])
    idx, off, dist, count = pt.neighbours(geom, pos, alive, K=3)
    assert idx[0, 0].tolist() == [1, -1, -1]                           # across the corner seam
    assert torch.allclose(off[0, 0, 0], torch.tensor([-2.0, -2.0]))
    assert idx[0, 2].tolist() == [-1, -1, -1] and int(count[0, 2]) == 0
    assert math.isinf(float(dist[0, 0, 1]))
    again = pt.neighbours(geom, pos, alive, K=3)
    assert all(torch.equal(x, y) for x, y in zip((idx, off, dist, count), again))


def test_crowded_neighbour_search_is_isotropic_and_near_the_true_nearest():
    """443 bodies per 3 x 3 block (more than the scan budget): the found neighbours lie east and north half the
    time, and their distances are close to the true K nearest."""
    n, L, N, K = 16, 256.0, 12600, 32
    geom = pt.geometry_from_elevation(torch.zeros(1, n, n), L, -1e4, R_781)
    gen = torch.Generator().manual_seed(1)
    pos = torch.rand(1, N, 2, generator=gen) * L
    alive = torch.ones(1, N, dtype=torch.bool)
    idx, off, dist, count = pt.neighbours(geom, pos, alive, K=K, gen=torch.Generator().manual_seed(2))
    assert float(count.float().mean()) == pytest.approx(443.0, rel=0.05)
    v = idx >= 0
    assert bool(v.all())
    assert float((off[..., 0][v] > 0).float().mean()) == pytest.approx(0.5, abs=0.01)
    assert float((off[..., 1][v] > 0).float().mean()) == pytest.approx(0.5, abs=0.01)
    sub = torch.arange(0, N, 7)
    d = pt.wrap(pos[0, sub, None, :] - pos[0, None, :, :], L).norm(dim=-1)
    d[torch.arange(sub.numel()), sub] = float("inf")
    true = d.topk(K, largest=False).values
    assert float(dist[0, sub].mean()) < 1.06 * float(true.mean())
    assert float(dist[0, sub, -1].mean()) < 1.08 * float(true[:, -1].mean())
    # low-slot bodies are not favoured: found neighbours' slots are spread like all slots
    assert float(idx[v].double().mean()) == pytest.approx((N - 1) / 2, rel=0.03)


# ------------------------------------------------------------------------------------------------ smells
def test_volatiles_diffuse_periodically_and_conserve_without_loss():
    n, L = 32, 512.0
    geom = pt.geometry_from_elevation(torch.zeros(1, n, n), L, -1e4, R_781)
    rules = pt.PatchRules(ventilation_s=float("inf"))
    vol = pt.init_volatiles(geom)
    src = pt.deposit(geom, [0], [[1.0, 1.0]], [2.0])                   # kg/s of smoke in cell (0, 0)
    dt = 600.0
    vol = pt.volatile_step(vol, geom, dt, smoke_kg_m2_s=src, rules=rules)
    mass = float(vol.smoke.double().sum()) * geom.cell_m2 * rules.smell_layer_m
    assert mass == pytest.approx(2.0 * dt, rel=1e-5)
    s = vol.smoke[0]
    assert float(s[1, 0]) == pytest.approx(float(s[n - 1, 0]), rel=1e-4)
    assert float(s[0, 1]) == pytest.approx(float(s[0, n - 1]), rel=1e-4)
    assert float(s[0, 0]) > float(s[1, 0]) > float(s[2, 0]) > 0
    # no source: the content stays, it only spreads
    vol2 = pt.volatile_step(vol, geom, 3600.0, rules=rules)
    mass2 = float(vol2.smoke.double().sum()) * geom.cell_m2 * rules.smell_layer_m
    assert mass2 == pytest.approx(mass, rel=1e-5)
    assert float(vol2.smoke.max()) < float(vol.smoke.max())


def test_volatile_steady_state_and_sources():
    n, L = 16, 256.0
    geom = pt.geometry_from_elevation(torch.zeros(1, n, n), L, -1e4, R_781)
    r = pt.PatchRules()
    plant = torch.full((1, n, n), 0.3)
    vol = pt.volatile_step(pt.init_volatiles(geom), geom, pt.DAY_S, plant_c=plant)
    e = 0.3 / r.bio.plant_carbon_fraction * r.plant_emission_kg_kg_s
    tau = 1.0 / (1.0 / r.plant_volatile_life_s + 1.0 / r.ventilation_s)
    assert torch.allclose(vol.plant, torch.full_like(vol.plant, e * tau / r.smell_layer_m), rtol=1e-4)
    assert float(vol.carcass.abs().max()) == 0.0 and float(vol.smoke.abs().max()) == 0.0
    flesh = pt.deposit(geom, [0], [[100.0, 100.0]], [30.0])
    vol = pt.volatile_step(vol, geom, 3600.0, carcass_kg_m2=flesh)
    gx, gy = pt.gradient(geom, vol.carcass)
    i = int(100.0 // geom.dx)
    assert float(vol.carcass[0, i, i]) == float(vol.carcass.max())
    assert float(gx[0, i - 2, i]) > 0 > float(gx[0, i + 2, i])             # the gradient points to the carcass
    assert float(pt.smoke_emission(0.01)) == pytest.approx(0.01 * r.smoke_kg_per_kg_fuel)
    # CO2 from point sources: a body's metabolism (chemistry) and fires share the channel
    w = float(pt.metabolic_co2_emission(10.0))
    assert w == pytest.approx(10.0 / 4.5e5 * 0.85 * 0.04401, rel=1e-3)          # about 8e-7 kg/s for 10 W
    src = pt.deposit(geom, [0, 0], [[50.0, 50.0], [200.0, 200.0]], [w, float(pt.fire_co2_emission(1e-3))])
    vol = pt.volatile_step(vol, geom, 600.0, co2_kg_m2_s=src)
    assert float(vol.co2.max()) > 0 and float(vol.plant.max()) > 0
    i2 = int(200.0 // geom.dx)
    assert float(vol.co2[0, i2, i2]) > float(vol.co2[0, i, i]) > 0          # the fire outweighs the body
    assert set(pt.Volatiles.from_state({"plant": vol.plant, "carcass": vol.carcass, "smoke": vol.smoke})
               .state_dict()) == {"plant", "carcass", "smoke", "co2"}
