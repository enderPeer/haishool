"""Tests of the habitat globe: grid, motion, operators, terrain, sea level and deposits (PLANET-SPEC 2.5)."""
import math

import pytest

torch = pytest.importorskip("torch")

from haishool.life9.planet import constants  # noqa: E402
from haishool.life9.planet import globe as gb  # noqa: E402
from haishool.life9.planet.constants import EARTH_OCEAN_MASS_FRACTION, G, M_EARTH, R_EARTH  # noqa: E402

SPECIES = ("basalt", "granite", "flint", "sandstone", "limestone", "clay", "sand", "salt", "hematite", "magnetite",
           "malachite", "cassiterite", "native_copper", "obsidian")
MARINE = ("basalt", "limestone", "clay", "sand")
CRUST = torch.tensor([0.3, 0.25, 0.02, 0.1, 0.1, 0.1, 0.08, 0.01, 0.02, 0.01, 1e-3, 5e-4, 5e-4, 0.01],
                     dtype=torch.float64)
CRUST = CRUST / CRUST.sum()
R_HAB = 1.0e4
ROCK = gb.ROCK_DENSITY * gb.ACCESSIBLE_DEPTH_M


@pytest.fixture(scope="module")
def g48():
    return gb.Globe(48)


@pytest.fixture(scope="module")
def g24():
    return gb.Globe(24)


@pytest.fixture(scope="module")
def g16():
    return gb.Globe(16)


def unit_points(n, seed):
    gen = torch.Generator().manual_seed(seed)
    return gb._unit(torch.randn(n, 3, generator=gen))


def wrap(x):
    return torch.remainder(x + math.pi, 2 * math.pi) - math.pi


def area_mean(globe, field):
    return (field.double() * globe.area64).sum(-1) / (4 * math.pi)


def masked_mean(globe, field, mask):
    """Area mean of field [W, C] (or [W, C, S]) over the cells in mask [W, C]."""
    a = globe.area64 * mask
    if field.dim() == 3:
        return (field.double() * a[..., None]).sum(1) / a.sum(-1, keepdim=True)
    return (field.double() * a).sum(-1) / a.sum(-1)


def legendre(l, z):
    p0, p1 = torch.ones_like(z), z
    for n in range(1, l):
        p0, p1 = p1, ((2 * n + 1) * z * p1 - n * p0) / (n + 1)
    return p1


# ------------------------------------------------------------------------------------------- grid
@pytest.mark.parametrize("G", [2, 7, 48])
def test_area_sums_to_4pi_and_the_frame_is_orthonormal(G):
    g = gb.Globe(G)
    assert g.C == 6 * G * G and g.centers.shape == (g.C, 3)
    assert abs(float(g.area64.sum()) - 4 * math.pi) < 1e-12
    assert abs(float(g.area.double().sum()) - 4 * math.pi) < 1e-5
    assert float(g.area.min()) > 0 and float(g.area.max() / g.area.min()) < 1.45   # equiangular: nearly uniform
    assert torch.allclose(g.spacing, g.area.sqrt())
    assert torch.allclose(g.centers.norm(dim=-1), torch.ones(g.C), atol=1e-6)
    assert torch.allclose(g.lat.sin(), g.centers[:, 2], atol=1e-6)
    assert torch.allclose(torch.stack((g.lat.cos() * g.lon.cos(), g.lat.cos() * g.lon.sin()), -1),
                          g.centers[:, :2], atol=1e-6)
    for v in (g.east, g.north):
        assert torch.allclose(v.norm(dim=-1), torch.ones(g.C), atol=1e-6)
        assert float((v * g.centers).sum(-1).abs().max()) < 1e-6
    assert float((g.east * g.north).sum(-1).abs().max()) < 1e-6
    assert float(g.north[:, 2].min()) >= -1e-6 and float(g.east[:, 2].abs().max()) < 1e-6


def test_mean_cell_size_at_the_default_resolution(g48):
    # PLANET-SPEC section 1 table: 13,824 cells; mean side sqrt(4 pi R^2 / C) = 301 m on the 10 km habitat,
    # cell sizes sqrt(area) 277-327 m
    side = R_HAB * math.sqrt(4 * math.pi / g48.C)
    assert g48.C == 13824 and abs(side - 301.5) < 1
    assert 275 < R_HAB * float(g48.spacing.min()) and R_HAB * float(g48.spacing.max()) < 330


def test_cell_of_round_trips_centres_and_contains_points():
    for G in (7, 24):
        g = gb.Globe(G)
        assert torch.equal(g.cell_of(g.centers), torch.arange(g.C))
        assert torch.equal(g.cell_of(g.centers.double()), torch.arange(g.C))
        p = unit_points(20000, G)
        c = g.cell_of(p)
        assert c.dtype == torch.long and int(c.min()) >= 0 and int(c.max()) < g.C
        # the point lies inside its cell: on the inner side of all four great-circle sides
        corner = g.corners[c]
        for k in range(4):
            normal = torch.linalg.cross(corner[:, k], corner[:, (k + 1) % 4], dim=-1)
            assert float((normal * p).sum(-1).min()) > -1e-6
        # and its centre is the nearest one, or almost (the cells are quads, not Voronoi cells)
        ang = gb.angle(p[:, None], g.centers[None])
        nearest = ang.argmin(1)
        excess = ang.gather(1, c[:, None])[:, 0] - ang.min(1).values
        assert float((nearest == c).float().mean()) > 0.93
        assert float(excess.max()) < 0.35 * g.delta
        # any leading shape
        assert g.cell_of(p.reshape(4, 5000, 3)).shape == (4, 5000)


def test_neighbours_are_adjacent_symmetric_and_cross_faces():
    for G in (3, 16):
        g = gb.Globe(G)
        own = torch.arange(g.C)
        valid = g.nbr_valid
        assert int((~valid).sum()) == 24 and bool(valid[:, :4].all())        # 8 cube corners x 3 cells
        assert bool((g.nbr[~valid] == own[:, None].expand(-1, 8)[~valid]).all())
        assert not bool(((g.nbr == own[:, None]) & valid).any())
        distinct = torch.tensor([len(set(row[ok].tolist())) for row, ok in zip(g.nbr, valid)])
        assert bool((distinct == valid.sum(1)).all())                       # no duplicates
        assert int((distinct == 7).sum()) == 24 and int((distinct == 8).sum()) == g.C - 24
        d = gb.angle(g.centers[:, None], g.centers[g.nbr])
        assert float(d[valid].max()) < 1.6 * g.delta and float(d[:, :4].max()) < 1.2 * g.delta
        assert abs(float(d[:, :4].min()) - g.min_spacing) < 1e-6
        shared = (g.corner_id[:, None, :, None] == g.corner_id[g.nbr][:, :, None, :]).any(-1).sum(-1)
        assert bool((shared[:, :4] == 2).all())                             # a side neighbour shares a side
        assert bool((shared[:, 4:][valid[:, 4:]] == 1).all())               # a diagonal shares one vertex
        for c in range(g.C):
            for n in g.nbr[c][valid[c]].tolist():
                assert c in g.nbr[n][valid[n]].tolist()
        crossing = g.face[g.nbr4] != g.face[:, None]
        assert int(crossing.any(1).sum()) == 6 * 4 * (G - 1)               # the face rims
        assert int(crossing.sum()) == 6 * 4 * G                             # sides across the 12 cube edges, twice


def test_vertices_follow_euler_and_interpolate_linear_fields(g16):
    assert g16.V == g16.C + 2
    z = g16.centers[:, 2]
    assert float((g16.at_vertices(z) - g16.vertices[:, 2]).abs().max()) < 0.4 * g16.delta ** 2
    assert torch.allclose(g16.vertex_w.sum(-1), torch.ones(g16.V), atol=1e-5)


# ------------------------------------------------------------------------------------------- motion
def test_moving_once_round_returns_to_the_start():
    p = unit_points(4000, 1)
    gen = torch.Generator().manual_seed(2)
    h = torch.rand(4000, generator=gen) * 2 * math.pi
    keep = p[:, 2].abs() < 0.995
    p2, h2 = gb.move(p, h, 2 * math.pi * R_HAB, R_HAB)
    assert float(gb.angle(p, p2).max()) < 1e-4
    assert float(wrap(h2 - h)[keep].abs().max()) < 1e-3
    assert torch.allclose(p2.norm(dim=-1), torch.ones(4000), atol=1e-6)


def test_moving_keeps_the_heading_relative_to_north_and_reverses():
    gen = torch.Generator().manual_seed(3)
    p = unit_points(2000, 4)
    p = p[p[:, 2].abs() < 0.9]
    h = torch.rand(len(p), generator=gen) * 2 * math.pi
    d = torch.rand(len(p), generator=gen) * 0.9 * math.pi * R_HAB
    p2, h2 = gb.move(p, h, d, R_HAB)
    assert torch.allclose(gb.angle(p, p2), d / R_HAB, atol=2e-4)
    back, hb = gb.move(p2, h2 + math.pi, d, R_HAB)
    assert float(gb.angle(p, back).max()) < 2e-4
    assert float(wrap(hb - h - math.pi).abs().max()) < 2e-3
    # due north from the equator raises latitude by d/R; due east along the equator stays on it
    eq = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    north, hn = gb.move(eq, torch.zeros(2), 1000.0, R_HAB)
    assert torch.allclose(north[:, 2].asin(), torch.full((2,), 0.1), atol=1e-6) and float(hn.abs().max()) < 1e-6
    east, he = gb.move(eq, torch.full((2,), math.pi / 2), 3000.0, R_HAB)
    assert float(east[:, 2].abs().max()) < 1e-6 and torch.allclose(he, torch.full((2,), math.pi / 2), atol=1e-6)
    assert torch.allclose(torch.atan2(east[0, 1], east[0, 0]), torch.tensor(0.3), atol=1e-6)
    # over the pole the heading flips from north to south
    over, ho = gb.move(eq[:1], torch.zeros(1), math.pi * R_HAB, R_HAB)
    assert float(gb.angle(over, -eq[:1]).max()) < 1e-5 and abs(float(ho[0]) - math.pi) < 1e-4


def test_headings_and_bearings_stay_in_their_half_open_ranges():
    # a heading a hair west of north: torch.remainder rounds -1e-9 up to float32(2 pi)
    _, h = gb.move(torch.tensor([[1.0, 0.0, 0.0]]), torch.tensor([-1e-9]), 1000.0, R_HAB)
    assert 0.0 <= float(h[0]) < 2 * math.pi
    x = torch.tensor([-1e-9, -1e-30, 0.0, 1.0, 7.0])
    folded = gb._wrap(torch.remainder(x, 2 * math.pi), 2 * math.pi)
    assert bool((folded >= 0).all()) and bool((folded < 2 * math.pi).all())
    p, q = unit_points(5000, 12), unit_points(5000, 13)
    gen = torch.Generator().manual_seed(14)
    heading = (torch.rand(5000, generator=gen) - 0.5) * 40
    _, h2 = gb.move(p, heading, 3000.0, R_HAB)
    b = gb.bearing(p, q, heading)
    assert bool((h2 >= 0).all()) and bool((h2 < 2 * math.pi).all())
    assert bool((b >= -math.pi - 1e-6).all()) and bool((b < math.pi).all())


def test_bearing_points_where_the_move_went():
    gen = torch.Generator().manual_seed(5)
    p = unit_points(3000, 6)
    p = p[p[:, 2].abs() < 0.98]
    n = len(p)
    h = torch.rand(n, generator=gen) * 2 * math.pi
    b = (torch.rand(n, generator=gen) - 0.5) * 2 * 3.1
    d = 50.0 + torch.rand(n, generator=gen) * 20000.0
    q, _ = gb.move(p, h + b, d, R_HAB)
    assert float(wrap(gb.bearing(p, q, h) - b).abs().max()) < 1e-3
    # broadcasting: [N,1,3] against [1,M,3]
    bb = gb.bearing(p[:5, None], q[None, :7], h[:5, None])
    assert bb.shape == (5, 7) and float(bb.abs().max()) <= math.pi


def test_slerp_and_sample(g16):
    p, q = unit_points(100, 7), unit_points(100, 8)
    mid = gb.slerp(p, q, 0.5)
    assert torch.allclose(gb.angle(p, mid), gb.angle(mid, q), atol=1e-5)
    assert torch.allclose(gb.slerp(p, q, 0.0), p, atol=1e-6) and torch.allclose(gb.slerp(p, q, 1.0), q, atol=1e-5)
    third = gb.slerp(p, q, torch.full((100,), 1 / 3))
    assert torch.allclose(gb.angle(p, third), gb.angle(p, q) / 3, atol=1e-5)
    # nearly equal points, antipodal points (also at a pole) and nearly antipodal points
    close = gb._unit(p + 1e-4 * q)
    assert torch.allclose(gb.angle(p, gb.slerp(p, close, 0.5)), 0.5 * gb.angle(p, close), atol=1e-6)
    poles = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    for a, b in ((p, -p), (poles, -poles), (p, gb._unit(-p + 1e-3 * q))):
        m = gb.slerp(a, b, 0.5)
        assert torch.allclose(m.norm(dim=-1), torch.ones(len(a)), atol=1e-6)
        assert torch.allclose(gb.angle(a, m), gb.angle(m, b), atol=1e-5)
        assert torch.allclose(gb.angle(a, m), 0.5 * gb.angle(a, b), atol=1e-5)
        assert torch.equal(g16.cell_of(m), g16.cell_of(m))                     # no NaN
        assert bool(torch.isfinite(m).all())
    field = torch.arange(2 * g16.C, dtype=torch.float32).reshape(2, g16.C)
    pts = torch.stack((p, q))
    assert torch.equal(g16.sample(field, pts), field.gather(1, g16.cell_of(pts)))


# ------------------------------------------------------------------------------------------- operators
def test_laplacian_of_constants_and_spherical_harmonics(g48, g24):
    assert float(g48.laplacian(torch.full((3, g48.C), 288.0)).abs().max()) == 0.0
    z = g48.centers[:, 2]
    w = g48.area / g48.area.sum()
    err = g48.laplacian(z[None])[0] + 2 * z                       # Y_1^0 = cos(colatitude): eigenvalue -2
    assert float(((err ** 2) * w).sum().sqrt()) < 0.005 and float(err.abs().max()) < 0.06
    y2 = 3 * z ** 2 - 1                                           # Y_2^0: eigenvalue -6
    err2 = g48.laplacian(y2[None])[0] + 6 * y2
    assert float(((err2 ** 2) * w).sum().sqrt()) < 0.08
    # supraconvergent: the pointwise error on the face rims stays O(1) (stated in the docstring), only the RMS falls
    zc = g24.centers[:, 2]
    y2c = 3 * zc ** 2 - 1
    err2c = g24.laplacian(y2c[None])[0] + 6 * y2c
    assert float(((err2 ** 2) * w).sum().sqrt()) < float(((err2c ** 2) * g24.area / g24.area.sum()).sum().sqrt())
    assert float(err2.abs().max()) > 0.3


def test_diffusion_converges_at_second_order():
    errors = []
    for G in (12, 24, 48):
        g = gb.Globe(G)
        z = g.centers[:, 2].double()
        y2, y3 = 3 * z ** 2 - 1, 5 * z ** 3 - 3 * z                  # eigenvalues -6 and -12
        out = g.diffuse((y2 + y3).float()[None], 0.1)[0].double()
        exact = y2 * math.exp(-0.6) + y3 * math.exp(-1.2)
        errors.append(float(((out - exact) ** 2 * g.area64).sum().sqrt() / math.sqrt(4 * math.pi)))
    assert errors[0] < 4e-3
    assert errors[0] / errors[1] > 3 and errors[1] / errors[2] > 3


def test_laplacian_conserves_and_its_spectrum_is_the_spheres():
    g = gb.Globe(10)
    gen = torch.Generator().manual_seed(9)
    f = torch.randn(4, g.C, generator=gen)
    lap = g.laplacian(f)
    assert float((lap * g.area).sum(-1).abs().max()) < 1e-5 * float((lap.abs() * g.area).sum(-1).max())
    m = g.laplacian(torch.eye(g.C, dtype=torch.float64)).T          # column k = laplacian of the unit vector k
    ev = torch.linalg.eigvals(m)
    real = torch.sort(ev.real, descending=True).values
    assert float(real[0]) < 1e-6                                     # no growing mode: stable diffusion
    assert torch.allclose(real[1:4], torch.full((3,), -2.0, dtype=torch.float64), rtol=0.01)
    assert torch.allclose(real[4:9], torch.full((5,), -6.0, dtype=torch.float64), rtol=0.03)
    assert float(ev.abs().max()) < g.lap_radius


def test_diffuse_matches_the_heat_equation_and_conserves(g48):
    z = g48.centers[:, 2]
    for kappa in (1e-3, 4.75e-3):                                    # 4.75e-3 = 0.55 W/m2/K x 1 day / 1e7 J/m2/K
        out = g48.diffuse(z[None], kappa)[0]
        assert float((out - z * math.exp(-2 * kappa)).abs().max()) < 1e-4
    gen = torch.Generator().manual_seed(11)
    f = 250 + 40 * torch.rand(2, g48.C, generator=gen)
    kappa = torch.where(torch.rand(2, g48.C, generator=gen) < 0.3, 4.75e-3, 2.4e-4)    # land / ocean heat capacity
    k64 = kappa.double()
    # the anomaly content, not the content dominated by the 250 K offset, over 100 one-day calls
    scale = ((f.double() - 270).abs() * g48.area64 / k64).sum(-1)
    heat0 = ((f.double() - 270) * g48.area64 / k64).sum(-1)
    out = f
    for _ in range(100):
        out = g48.diffuse(out, kappa)
    assert out.dtype == torch.float32
    heat1 = ((out.double() - 270) * g48.area64 / k64).sum(-1)
    assert float(((heat1 - heat0) / scale).abs().max()) < 2e-6
    assert float(out.min()) >= float(f.min()) - 1e-3 and float(out.max()) <= float(f.max()) + 1e-3
    smooth = g48.smooth(f, 0.1)
    assert float(smooth.std(-1).max()) < 0.5 * float(f.std(-1).min())
    assert torch.allclose(area_mean(g48, smooth), area_mean(g48, f), rtol=1e-6)


def test_diffuse_refuses_unstable_or_negative_steps(g48):
    f = torch.where(g48.centers[:, 2] > 0, 300.0, 250.0)[None]
    with pytest.raises(ValueError):
        g48.diffuse(f, 4.75e-3, steps=1)
    with pytest.raises(ValueError):
        g48.diffuse(f, -1e-3)
    need = math.ceil(4.75e-3 * g48.lap_radius)
    out = g48.diffuse(f, 4.75e-3, steps=need + 5)
    assert float(out.min()) >= 250.0 - 1e-3 and float(out.max()) <= 300.0 + 1e-3
    assert torch.equal(g48.diffuse(f, 0.0), f)


def test_smoothing_is_the_same_at_every_resolution(g48, g24):
    # smoothing a Legendre mode P_l(z) multiplies it by exp(-l(l+1) L^2 / 2) whatever G is
    length, l = 0.1, 8
    factor = math.exp(-l * (l + 1) * length ** 2 / 2)
    errs = []
    for g in (g24, g48):
        pl = legendre(l, g.centers[:, 2].double())
        out = g.smooth(pl.float()[None], length)[0].double()
        errs.append(float((out - factor * pl).abs().max()))
    assert errs[0] < 0.03 and errs[1] < errs[0]


def test_hops_counts_neighbour_steps(g16):
    source = torch.zeros(1, g16.C, dtype=torch.bool)
    source[0, 0] = True
    hops = g16.hops(source, 3)
    assert int(hops[0, 0]) == 0 and bool((hops[0, g16.nbr[0][g16.nbr_valid[0]]] == 1).all())
    assert int((hops == 1).sum()) == 7 and int(hops.max()) == 4      # cell 0 sits at a cube corner


def test_distance_is_the_geodesic_angle_to_the_nearest_source(g48, g24):
    for g in (g24, g48):
        source = torch.zeros(2, g.C, dtype=torch.bool)
        source[0, [5, 777, g.C // 2]] = True
        source[1, g.C - 1] = True
        dist = g.distance(source, 0.4)
        assert dist.shape == (2, g.C) and dist.dtype == torch.float32
        for w in range(2):
            exact = gb.angle(g.centers[:, None], g.centers[source[w]][None]).amin(-1)
            near = exact < 0.4
            assert float((dist[w][near] - exact[near]).abs().max()) < 0.05 * g.delta
            assert bool((dist[w][~near] == 0.4).all())
    none = g24.distance(torch.zeros(1, g24.C, dtype=torch.bool), 0.2)
    assert bool((none == 0.2).all())


# ------------------------------------------------------------------------------------------- terrain
def test_terrain_sea_level_matches_the_water_volume(g48):
    W = 4
    relief = torch.tensor([600.0, 700.0, 450.0, 600.0])
    shell = 4 * math.pi / 3 * ((R_HAB + relief.double()) ** 3 - R_HAB ** 3)
    water = shell * torch.tensor([0.143, 0.05, 0.3, 0.143], dtype=torch.float64)
    ter = gb.make_terrain(g48, W, torch.Generator().manual_seed(781), relief, water, R_HAB)
    e = ter["elevation_m"]
    assert e.shape == (W, g48.C) and e.dtype == torch.float32 and ter["sea_level_m"].shape == (W,)
    assert torch.allclose(e.amin(-1), torch.zeros(W)) and torch.allclose(e.amax(-1), relief, rtol=1e-6)
    assert float(((ter["ocean_volume_m3"] - water) / water).abs().max()) < 1e-5
    exact = gb.ocean_volume(g48, e, ter["sea_level_m"], R_HAB)
    assert torch.equal(exact, ter["ocean_volume_m3"])
    assert torch.equal(gb.sea_level(g48, e, water, R_HAB), ter["sea_level_m"])     # the factored-out solver
    sea = ter["sea_level_m"][:, None]
    assert torch.equal(ter["land"], e >= sea)
    assert torch.allclose(ter["depth_m"], (sea - e).clamp_min(0))
    assert bool((ter["depth_m"][ter["land"]] == 0).all()) and bool((ter["depth_m"][~ter["land"]] > 0).all())
    coast = ter["land"] & (~ter["land"])[:, g48.nbr].any(-1)
    assert torch.equal(ter["coast"], coast) and int(coast.sum()) > 0
    frac = ter["ocean_fraction"]
    assert bool(((frac > 0.05) & (frac < 0.98)).all())
    assert float(frac[1]) < float(frac[0]) < float(frac[2])           # more water, more ocean
    # flat-earth check: a layer of depth h over area A holds about A R^2 h, but not exactly (ledgers use volume)
    planar = (ter["depth_m"].double() * g48.area64 * R_HAB ** 2).sum(-1)
    assert float(((planar - water) / water).abs().max()) < 0.08
    # inland distance: 0 on the ocean, under one cell on the coast, growing inland, capped
    inland = ter["inland_rad"]
    assert bool((inland[~ter["land"]] == 0).all()) and float(inland.max()) <= gb.INLAND_CAP_RAD + 1e-6
    assert float(inland[coast].max()) < 1.2 * g48.delta
    assert float(inland[ter["land"] & ~coast].min()) > 0.5 * g48.delta


def test_sea_level_round_trips_any_volume(g16):
    ter = gb.make_terrain(g16, 3, torch.Generator().manual_seed(2), 600.0, 0.0, R_HAB)
    e = ter["elevation_m"]
    level = torch.tensor([50.0, 300.0, 580.0])
    volume = gb.ocean_volume(g16, e, level, R_HAB)
    back = gb.sea_level(g16, e, volume, R_HAB)
    assert torch.allclose(back, level, atol=1e-3)
    assert torch.allclose(gb.ocean_volume(g16, e, back, R_HAB), volume, rtol=1e-6)


def test_terrain_extremes_and_determinism(g16):
    dry = gb.make_terrain(g16, 2, torch.Generator().manual_seed(1), 600.0, 0.0, R_HAB)
    assert bool(dry["land"].all()) and float(dry["ocean_fraction"].max()) == 0.0
    assert bool((dry["inland_rad"] == gb.INLAND_CAP_RAD).all())
    flood = gb.make_terrain(g16, 2, torch.Generator().manual_seed(1), 600.0, 1e12, R_HAB)
    assert not bool(flood["land"].any()) and bool((flood["sea_level_m"] > 600.0).all())
    assert float(((flood["ocean_volume_m3"] - 1e12) / 1e12).abs().max()) < 1e-5
    a = gb.make_terrain(g16, 3, torch.Generator().manual_seed(5), 600.0, 1e11, R_HAB)
    b = gb.make_terrain(g16, 3, torch.Generator().manual_seed(5), 600.0, 1e11, R_HAB)
    c = gb.make_terrain(g16, 3, torch.Generator().manual_seed(6), 600.0, 1e11, R_HAB)
    for k in a:
        assert torch.equal(a[k], b[k]), k
    assert not torch.equal(a["elevation_m"], c["elevation_m"])
    assert not torch.equal(a["elevation_m"][0], a["elevation_m"][1])  # worlds differ


def test_terrain_pattern_does_not_depend_on_resolution():
    coarse, fine = gb.Globe(16), gb.Globe(48)
    a = gb.make_terrain(coarse, 2, torch.Generator().manual_seed(3), 600.0, 0.0, R_HAB)["elevation_m"]
    b = gb.make_terrain(fine, 2, torch.Generator().manual_seed(3), 600.0, 0.0, R_HAB)["elevation_m"]
    b_at_coarse = fine.sample(b, coarse.centers[None].expand(2, -1, -1))
    corr = torch.corrcoef(torch.stack((a[0], b_at_coarse[0])))[0, 1]
    assert float(corr) > 0.95


def test_earth_water_covers_earths_ocean_fraction(g24):
    g_earth = G * M_EARTH / R_EARTH ** 2
    relief = 600.0 * 9.81 / g_earth
    volume = gb.habitat_water_volume(EARTH_OCEAN_MASS_FRACTION * M_EARTH, R_EARTH, g_earth, relief, R_HAB)
    shell = 4 * math.pi / 3 * ((R_HAB + relief) ** 3 - R_HAB ** 3)
    assert abs(float(volume) / shell - gb.EARTH_FILL) < 1e-9
    fractions = torch.cat([gb.make_terrain(g24, 32, torch.Generator().manual_seed(100 + s), relief, volume, R_HAB)
                           ["ocean_fraction"] for s in range(2)])
    assert abs(float(fractions.mean()) - gb.EARTH_OCEAN_FRACTION) < 0.03
    # twice the ocean on the same planet: twice the habitat water; vectorised over worlds
    two = gb.habitat_water_volume(torch.tensor([1.0, 2.0]) * EARTH_OCEAN_MASS_FRACTION * M_EARTH, R_EARTH, g_earth,
                                  relief, R_HAB)
    assert torch.allclose(two[1] / two[0], torch.tensor(2.0, dtype=torch.float64))


# ------------------------------------------------------------------------------------------- deposits
def deposits(globe, seed, W=2, water=0.12, crust=CRUST):
    gen = torch.Generator().manual_seed(seed)
    shell = 4 * math.pi / 3 * ((R_HAB + 600.0) ** 3 - R_HAB ** 3)
    ter = gb.make_terrain(globe, W, gen, 600.0, water * shell, R_HAB)
    crust = crust.repeat(W, 1)
    report = {}
    stock = gb.make_deposits(globe, W, gen, ter, crust, SPECIES, report=report)
    return ter, crust, stock, report


def expected_mix(crust):
    expect = ROCK * crust.double().clone()
    expect[:, SPECIES.index("clay")] *= gb.CLAY_START
    return expect


def test_the_land_holds_the_crust_mix_and_one_metre_of_rock(g48):
    ter, crust, stock, report = deposits(g48, 781, water=0.2)
    W, S = crust.shape
    land, ocean = ter["land"], ~ter["land"]
    assert stock.shape == (W, g48.C, S) and stock.dtype == torch.float32 and float(stock.min()) >= 0
    assert bool((report["support_level"] == 0).all())
    assert float(report["land_error"].max()) < 1e-8 and float(report["ocean_error"].max()) < 1e-8
    # land inventory: the land-area mean of every species is its crust share of 1 m of rock (clay x CLAY_START)
    assert torch.allclose(masked_mean(g48, stock, land), expected_mix(crust), rtol=1e-5)
    clay = SPECIES.index("clay")
    full = stock.double().clone()
    full[..., clay] /= gb.CLAY_START
    column = full.sum(-1)
    assert torch.allclose(column[land], torch.full_like(column[land], ROCK * float(crust[0].sum())), rtol=1e-5)
    assert torch.allclose(masked_mean(g48, stock.sum(-1), land), expected_mix(crust).sum(-1), rtol=1e-5)
    land_mean = masked_mean(g48, stock, land)
    ratio = land_mean[:, SPECIES.index("granite")] / land_mean[:, SPECIES.index("basalt")]
    assert torch.allclose(ratio, crust[:, 1] / crust[:, 0], rtol=1e-5)
    # sea floor: only the marine species, in their crust proportions renormalised, also 1 m of rock
    marine = torch.tensor([s in MARINE for s in SPECIES])
    assert float(stock[..., ~marine][ocean].abs().max()) == 0.0
    share = crust * marine
    share = share / share.sum(-1, keepdim=True) * crust.sum(-1, keepdim=True)
    assert torch.allclose(masked_mean(g48, stock, ocean), expected_mix(share), rtol=1e-5)
    assert torch.allclose(column[ocean], torch.full_like(column[ocean], ROCK * float(crust[0].sum())), rtol=1e-5)


def test_deposits_follow_the_ground(g48):
    ter, crust, stock, _ = deposits(g48, 781)
    land, coast = ter["land"], ter["coast"]
    e = ter["elevation_m"]
    s = {name: stock[..., i] for i, name in enumerate(SPECIES)}

    def land_elevation(x):
        return (x * e * land).sum(-1) / (x * land).sum(-1)

    assert bool((land_elevation(s["granite"]) > land_elevation(s["basalt"]) + 20).all())   # felsic high, mafic low
    inland = land & ~coast
    assert bool(((s["sand"] * coast).sum(-1) / coast.sum(-1) > 3 * (s["sand"] * inland).sum(-1) / inland.sum(-1)).all())
    # salt at least SALT_MIN_DIST_RAD inland: none on the coast cells (narrower than that at G = 48)
    assert float(s["salt"][coast].max()) == 0.0 and float(s["salt"].max()) > 0
    assert float(ter["inland_rad"][s["salt"] > 0].min()) > gb.SALT_MIN_DIST_RAD - g48.delta
    shelf = ~land & (ter["depth_m"] < 30.0)
    deep = ~land & (ter["depth_m"] > 100.0)
    for name, more_on_shelf in (("sand", True), ("limestone", True), ("basalt", False), ("clay", False)):
        on_shelf, in_deep = masked_mean(g48, s[name], shelf), masked_mean(g48, s[name], deep)
        assert bool(((on_shelf > in_deep) == more_on_shelf).all()), name
    for name in ("hematite", "malachite", "cassiterite"):
        x = torch.sort(s[name], -1, descending=True).values
        assert bool((x[:, : g48.C // 10].sum(-1) > 0.8 * x.sum(-1)).all()), name          # patches
    corr = torch.corrcoef(torch.stack((s["malachite"][0], s["native_copper"][0])))[0, 1]
    assert float(corr) > 0.5                                                             # copper minerals together


def test_deposit_patterns_do_not_depend_on_resolution(g24, g48):
    # where sand and salt lie, as their inventory's mean distance inland (rad), is the same at G = 24 and 48
    out = {}
    for g in (g24, g48):
        ter, _, stock, _ = deposits(g, 3, W=4)
        a = g.area64 * ter["land"]
        out[g.G] = {}
        for name in ("sand", "salt", "granite"):
            x = stock[..., SPECIES.index(name)].double() * a
            out[g.G][name] = float(((x * ter["inland_rad"]).sum(-1) / x.sum(-1)).mean())
    for name in ("sand", "salt", "granite"):
        assert abs(out[24][name] / out[48][name] - 1) < 0.12, (name, out)


def test_ore_patches_carry_the_same_mass_at_every_resolution(g24, g48):
    for r in (0.02, 0.05):
        for g in (g24, g48):
            where = torch.ones(2, g.C, dtype=torch.bool)
            field = gb.ore_patches(g, 2, torch.Generator().manual_seed(1), where, count=3, radius_rad=(r, r))
            mass = (field.double() * g.area64).sum(-1)
            assert torch.allclose(mass, torch.full((2,), 3 * math.pi * r * r, dtype=torch.float64), rtol=1e-5)
    # a resolved patch peaks at about 1
    field = gb.ore_patches(g48, 1, torch.Generator().manual_seed(2), torch.ones(1, g48.C, dtype=torch.bool),
                           count=1, radius_rad=(0.06, 0.06))
    assert 0.9 < float(field.max()) < 1.1


def test_a_flooded_world_has_only_a_sea_floor(g16):
    with pytest.warns(RuntimeWarning, match="no land"):
        ter, crust, stock, report = deposits(g16, 4, water=5.0)
    assert not bool(ter["land"].any())
    level = report["support_level"]
    marine = torch.tensor([s in MARINE for s in SPECIES])
    assert bool((level[:, ~marine] == 3).all())
    assert float(stock[..., ~marine].abs().max()) == 0.0
    share = crust * marine
    share = share / share.sum(-1, keepdim=True)
    assert torch.allclose(masked_mean(g16, stock, ~ter["land"]), expected_mix(share), rtol=1e-5)


def test_a_species_without_its_own_ground_widens_and_keeps_its_inventory(g48):
    # a land belt 0.05 rad wide either side of the equator: nowhere SALT_MIN_DIST_RAD inland
    lat = g48.lat
    elevation = (600.0 * (1 - lat.abs() / (math.pi / 2)))[None]
    sea = torch.tensor([600.0 * (1 - 0.05 / (math.pi / 2))])
    ter = {"elevation_m": elevation, "sea_level_m": sea, "depth_m": (sea[:, None] - elevation).clamp_min(0),
           "land": elevation >= sea[:, None]}
    crust = CRUST[None]
    report = {}
    stock = gb.make_deposits(g48, 1, torch.Generator().manual_seed(0), ter, crust, SPECIES, report=report)
    salt = SPECIES.index("salt")
    assert int(report["support_level"][0, salt]) == 1                   # low land instead of inland basins
    assert float(stock[0, :, salt].max()) > 0
    assert torch.allclose(masked_mean(g48, stock, ter["land"]), expected_mix(crust), rtol=1e-5)
    # a share too large for its own ground widens too (SUPPORT_MARGIN), and the inventory still holds
    big = CRUST.clone()
    big[salt] = 1.2                                                     # over half the mix after normalising
    big = (big / big.sum())[None].repeat(2, 1)
    shell = 4 * math.pi / 3 * ((R_HAB + 600.0) ** 3 - R_HAB ** 3)
    gen = torch.Generator().manual_seed(7)
    ter = gb.make_terrain(g48, 2, gen, 600.0, 0.12 * shell, R_HAB)
    report = {}
    stock = gb.make_deposits(g48, 2, gen, ter, big, SPECIES, report=report)
    assert bool((report["support_level"][:, salt] >= 1).all())
    assert torch.allclose(masked_mean(g48, stock, ter["land"]), expected_mix(big), rtol=1e-5)


def test_deposits_are_deterministic_and_checked(g16):
    a = deposits(g16, 4)[2]
    b = deposits(g16, 4)[2]
    c = deposits(g16, 5)[2]
    assert torch.equal(a, b) and not torch.equal(a, c)
    ter, crust, _, _ = deposits(g16, 4)
    with pytest.raises(ValueError):
        gb.make_deposits(g16, 2, torch.Generator().manual_seed(0), ter, crust[:, :3], SPECIES)
    with pytest.raises(ValueError):
        gb.make_deposits(g16, 2, torch.Generator().manual_seed(0), ter, -crust, SPECIES)


# ------------------------------------------------------------------------------------------- provenance
def _numeric(v):
    if isinstance(v, bool):
        return False
    if isinstance(v, (int, float)):
        return True
    if isinstance(v, (tuple, list)):
        return any(_numeric(x) for x in v)
    if isinstance(v, dict):
        return any(_numeric(x) for x in v.values())
    return False


def test_every_setup_number_has_a_provenance_tag():
    tags = {"chain", "derived", "reference", "new_rule"}
    assert all(tag in tags and note for tag, note in gb.PROVENANCE.values())
    notes = " | ".join(note for _, note in gb.PROVENANCE.values())
    geometry = {"TWO_PI", "OFFSETS", "SIDES"}                     # grid tables, not set-up numbers
    names = [name for name, value in vars(gb).items()
             if name.isupper() and not name.startswith("_") and name not in geometry
             and not hasattr(constants, name) and _numeric(value)]
    assert len(names) > 25
    missing = [name for name in names if f"{name} = {getattr(gb, name)}" not in notes]
    assert not missing, missing
    assert gb.PROVENANCE["terrain.water_volume"][0] == "new_rule" and "calibrated" in gb.PROVENANCE[
        "terrain.water_volume"][1]


def test_the_sparse_laplacian_is_the_flux_laplacian():
    g = gb.Globe(10)
    gen = torch.Generator().manual_seed(3)
    f = torch.randn(3, g.C, generator=gen, dtype=torch.float64)
    ref = gb._laplacian(f, *g._lap64)
    got = (g.laplacian_csr() @ f.T).T
    assert torch.allclose(got, ref, rtol=1e-10, atol=1e-9 * float(ref.abs().max()))
    kappa = torch.full((3, 1), 2e-3, dtype=torch.float64)
    steps = math.ceil(2e-3 * g.lap_radius)
    a = g.diffuse(f, kappa, steps=steps)
    b = g.diffuse(f, kappa, steps=steps, check=False)
    assert torch.equal(a, b)
