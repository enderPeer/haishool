"""Tests of haishool.life9.planet.senses (PLANET-SPEC 2.9): shapes, sight with the real horizon, sound
physics (spreading, absorption, occlusion, no sound below 1 kPa), sectors, fields and held items."""
import math

import pytest

torch = pytest.importorskip("torch")

from haishool.life9.planet import brain, items, materials, senses  # noqa: E402
from haishool.life9.planet import creatures as C  # noqa: E402
from haishool.life9.planet import globe as G  # noqa: E402

R = 10000.0
GLOBE = G.Globe(96)                       # cells of about 100 m on the 10 km habitat
P0 = GLOBE.centers[GLOBE.cell_of(torch.tensor([1.0, 0.0, 0.0]) + torch.tensor([0.0, 0.004, 0.004]))]
# two 30 kg animals (eyes 0.62 m up) see each other over level ground out to 2 sqrt(2 R h) = 223 m


@pytest.fixture(scope="module")
def glb():
    return GLOBE


def env_of(glb, W=1, *, pressure=101325.0, rho=1.204, c=343.2, insolation=400.0, plant=0.0, soil=0.0):
    Cc = glb.C
    return {"elevation_m": torch.zeros(W, Cc), "sea_level_m": torch.full((W,), -1.0),
            "land": torch.ones(W, Cc, dtype=torch.bool), "t_air_k": torch.full((W, Cc), 290.0),
            "soil_kg_m2": torch.full((W, Cc), float(soil)), "plant_c": torch.full((W, Cc), float(plant)),
            "insolation_w_m2": torch.full((W, Cc), float(insolation)),
            "surface_pressure_pa": torch.full((W,), float(pressure)), "air_density_kg_m3": torch.full((W,), float(rho)),
            "sound_speed_m_s": torch.full((W,), float(c))}


def population(N=4, W=1, hidden=8):
    return C.empty(W, N, in_dim=senses.IN_DIM, hidden=hidden, initial_hidden=4)


def at(dist_m, heading=0.0, base=P0):
    p, _ = G.move(base, torch.tensor(float(heading)), torch.tensor(float(dist_m)), R)
    return p


def put(cr, n, pos, *, heading=0.0, mass=30.0, acuity=1.0, hearing=1.0, voice=1.0, w=0):
    m = torch.tensor(float(mass))
    C.place(cr, torch.tensor([w]), torch.tensor([n]), pos=pos[None], heading=heading, mass=m,
            reserve=0.5 * C.reserve_max(m), water=C.water_norm(m), age=100.0, cooldown=0.0,
            genes={"adult_mass_kg": mass, "acuity": acuity, "hearing": hearing, "voice": voice, "diet": 0.0,
                   "insulation": 1.0, "speed": 1.0, "swim": 0.0, "litter": 1})


def ch(x, sector, name, w=0, n=0):
    return float(x[w, n, senses.sector_index(sector, name)])


def me(x, name, w=0, n=0):
    return float(x[w, n, senses.self_index(name)])


# ------------------------------------------------------------------------------------ layout
def test_layout_constants():
    assert senses.SECTOR_DIM == 11 + brain.VOCAL_DIMS
    assert senses.IN_DIM == senses.SECTORS * senses.SECTOR_DIM + len(senses.SELF_CHANNELS)
    assert senses.PLANT_INDEX == senses.self_index("plant")
    assert senses.sector_index(1, "kin") == senses.SECTOR_DIM
    assert len(set(senses.SECTOR_CHANNELS)) == senses.SECTOR_DIM and len(set(senses.SELF_CHANNELS)) == len(
        senses.SELF_CHANNELS)
    assert senses.sector_of(torch.tensor([0.0, 0.3, math.pi / 2, -math.pi / 2, math.pi - 0.01])).tolist() == \
        [0, 0, 2, 6, 4]


def test_observe_shapes_zero_for_the_dead_and_finite(glb):
    cr = population(N=6, W=2)
    put(cr, 0, P0)
    put(cr, 1, at(300.0), mass=10.0)
    put(cr, 0, at(500.0, 1.0), w=1)
    env = env_of(glb, W=2, plant=1.0, soil=100.0)
    pool = items.new_item_pool(2, 32)
    fires = items.new_fire_pool(2, 4)
    x = senses.observe(cr, glb, env, pool, fires, radius_m=R)
    assert x.shape == (2, 6, senses.IN_DIM) and x.dtype == torch.float32
    assert bool(torch.isfinite(x).all())
    assert float(x[~cr.alive].abs().max()) == 0.0
    assert bool((x[cr.alive][:, senses.self_index("bias")] == 1).all())
    assert me(x, "reserve") == pytest.approx(0.5) and me(x, "water") == pytest.approx(1.0)
    assert me(x, "mass") == pytest.approx(1.0) and me(x, "health") == pytest.approx(1.0)
    assert me(x, "plant") == pytest.approx(1 - math.exp(-2.0), rel=1e-5)
    assert me(x, "in_water") == 0.0
    # deterministic
    assert torch.equal(x, senses.observe(cr, glb, env, pool, fires, radius_m=R))


# ------------------------------------------------------------------------------------ sight
def test_the_horizon_of_the_habitat_globe_is_real(glb):
    eye = senses.EYE_K * 1.0 ** (1 / 3)                      # 1 kg animals: eyes 0.2 m up
    horizon = 2 * math.sqrt(2 * R * eye)                     # 126 m between two such eyes on level ground
    env = env_of(glb)
    for d, seen in ((0.8 * horizon, True), (1.25 * horizon, False)):
        cr = population(N=2)
        put(cr, 0, P0, mass=1.0)
        put(cr, 1, at(d), mass=1.0)
        x = senses.observe(cr, glb, env, radius_m=R)
        kin = ch(x, 0, "kin")
        assert (kin > 0) == seen, (d, kin)
        if seen:
            assert kin == pytest.approx(1 - d / 1000.0, rel=1e-3)
    # bigger animals have their eyes higher and see further over the curve
    cr = population(N=2)
    put(cr, 0, P0, mass=30.0)
    put(cr, 1, at(1.25 * horizon), mass=30.0)
    assert ch(senses.observe(cr, glb, env, radius_m=R), 0, "kin") > 0


def test_light_sets_the_range_with_a_night_floor(glb):
    cr = population(N=2)
    put(cr, 0, P0, acuity=2.0)
    put(cr, 1, at(200.0, math.pi / 2), mass=200.0)          # a heavy one, to the east
    x, info = senses.observe(cr, glb, env_of(glb, insolation=400.0), radius_m=R, details=True)
    assert float(info["r_v"][0, 0]) == pytest.approx(2000.0)
    assert ch(x, 2, "threat") == pytest.approx((1 - 200 / 2000) * (1 - 30 / 200), rel=1e-3)
    assert ch(x, 0, "threat") == 0.0 and ch(x, 2, "kin") == 0.0
    x, info = senses.observe(cr, glb, env_of(glb, insolation=0.0), radius_m=R, details=True)
    assert float(info["r_v"][0, 0]) == pytest.approx(senses.LIGHT_FLOOR * 2000.0)
    assert ch(x, 2, "threat") == 0.0                          # 200 m is beyond 100 m night sight
    # the heavy one sees the lighter one as prey, behind it (to the west = sector 6 when heading north)
    x = senses.observe(cr, glb, env_of(glb, insolation=400.0), radius_m=R)
    assert ch(x, 6, "prey", n=1) == pytest.approx((1 - 200 / 1000) * 30 / 200, rel=1e-3)
    # beyond the horizon nothing is seen however good the eyes
    far = population(N=2)
    put(far, 0, P0, acuity=5.0)
    put(far, 1, at(400.0))
    assert ch(senses.observe(far, glb, env_of(glb), radius_m=R), 0, "kin") == 0.0


def test_terrain_blocks_sight_and_halves_sound(glb):
    env = env_of(glb)
    d = 180.0
    cr = population(N=2)
    put(cr, 0, P0, hearing=1.0)
    put(cr, 1, at(d))
    cr.loud[0, 1] = 1.0
    cr.calls[0, 1] = torch.eye(brain.VOCAL_DIMS)[0]
    open_x = senses.observe(cr, glb, env, radius_m=R)
    wall = dict(env)
    wall["elevation_m"] = env["elevation_m"].clone()
    mid = glb.cell_of(at(d / 2))
    wall["elevation_m"][0, mid] = 10.0
    hid_x = senses.observe(cr, glb, wall, radius_m=R)
    assert ch(open_x, 0, "kin") > 0 and ch(hid_x, 0, "kin") == 0.0
    assert ch(open_x, 0, "call0") > ch(hid_x, 0, "call0") > 0
    # halving the pressure is 6.02 dB, a weight drop of 6.02 / 60
    assert ch(open_x, 0, "call0") - ch(hid_x, 0, "call0") == pytest.approx(20 * math.log10(2) / 60, rel=1e-3)


# ------------------------------------------------------------------------------------ sound
def expected_weight(d, p1=2.0, hearing=1.0, alpha=senses.ABSORB_DB_M):
    level = 20 * math.log10(p1 / (d * senses.P_HEAR_PA / hearing)) - alpha * d
    return min(1.0, max(0.0, level / senses.SL_FULL_DB))


def test_calls_spread_absorb_and_arrive_in_their_sector(glb):
    env = env_of(glb)
    cr = population(N=3)
    put(cr, 0, P0, hearing=1.0)
    put(cr, 1, at(150.0, math.pi / 2))                       # east of a listener heading north
    put(cr, 2, at(150.0, -math.pi / 2), voice=0.0)          # a silent one to the west
    cr.loud[0] = torch.tensor([0.0, 1.0, 1.0])
    vec = torch.linspace(-1, 1, brain.VOCAL_DIMS)
    cr.calls[0, 1] = vec
    cr.calls[0, 2] = torch.ones(brain.VOCAL_DIMS)
    x, info = senses.observe(cr, glb, env, radius_m=R, details=True)
    w = expected_weight(150.0)
    assert 0 < w < 1
    got = x[0, 0, senses.sector_index(2, "call0"):senses.sector_index(2, "call0") + brain.VOCAL_DIMS]
    assert torch.allclose(got, w * vec, atol=1e-4)
    for s in range(senses.SECTORS):
        if s != 2:
            assert ch(x, s, "call0") == 0.0
    assert float(info["heard"][0, 0]) == pytest.approx(w, rel=1e-4)
    # louder at half the distance (spherical spreading: +6 dB), quieter in thin air (pressure ~ density)
    near = population(N=2)
    put(near, 0, P0)
    put(near, 1, at(75.0, math.pi / 2))
    near.loud[0, 1] = 1.0
    near.calls[0, 1] = vec
    x_near = senses.observe(near, glb, env, radius_m=R)
    assert ch(x_near, 2, "call7") == pytest.approx(expected_weight(75.0), rel=1e-4)
    assert ch(x_near, 2, "call7") > w
    thin = env_of(glb, pressure=30000.0, rho=0.36, c=343.2)
    alpha = senses.ABSORB_DB_M * 1.204 / 0.36
    x_thin = senses.observe(near, glb, thin, radius_m=R)
    assert ch(x_thin, 2, "call7") == pytest.approx(expected_weight(75.0, p1=2.0 * 0.36 / 1.204, alpha=alpha), rel=1e-4)
    # a deaf listener hears nothing
    near.genome["hearing"][0, 0] = 0.0
    assert ch(senses.observe(near, glb, env, radius_m=R), 2, "call7") == 0.0


def test_no_sound_is_heard_below_1_kpa(glb):
    cr = population(N=2)
    put(cr, 0, P0, hearing=5.0)
    put(cr, 1, at(50.0))
    cr.loud[0, 1] = 1.0
    cr.calls[0, 1] = torch.ones(brain.VOCAL_DIMS)
    loud_air = senses.observe(cr, glb, env_of(glb, pressure=2000.0, rho=0.03), radius_m=R)
    assert ch(loud_air, 0, "call0") > 0                     # 2 kPa still carries a close call
    x, info = senses.observe(cr, glb, env_of(glb, pressure=500.0, rho=0.0075), radius_m=R, details=True)
    first = senses.sector_index(0, "call0")
    calls = torch.stack([x[..., senses.sector_index(s, "call0"):senses.sector_index(s, "call0") + brain.VOCAL_DIMS]
                         for s in range(senses.SECTORS)])
    assert float(calls.abs().max()) == 0.0 and float(info["heard"].abs().max()) == 0.0
    assert ch(x, 0, "kin") > 0 and first >= 0                # still seen


# ------------------------------------------------------------------------------------ fields, items, fires
def test_plants_and_water_ahead_show_in_the_forward_sector(glb):
    env = env_of(glb)
    cr = population(N=1)
    put(cr, 0, P0, acuity=1.0)
    ahead = torch.stack([at(1000.0 * f) for f in senses.RAY_FRACTIONS])
    cells = glb.cell_of(ahead)
    cells = cells[cells != glb.cell_of(P0)]
    env["plant_c"][0, cells] = 2.0
    env["soil_kg_m2"][0, cells] = 120.0
    x = senses.observe(cr, glb, env, radius_m=R)
    # on level ground only the samples within the horizon (about 210 m) are seen, 62.5 m (in the
    # animal's own cell, left bare) and 125 m (planted): one sample of five
    assert cells.tolist() == glb.cell_of(ahead[1:]).tolist() and float(G.angle(P0, ahead[2])) * R > 211
    cover = 1 - math.exp(-4.0)
    assert ch(x, 0, "plant") == pytest.approx(cover / 5, rel=1e-4) and ch(x, 4, "plant") == 0.0
    assert ch(x, 0, "water") == pytest.approx(1 / 5, rel=1e-4) and ch(x, 4, "water") == 0.0


def test_high_ground_sees_further(glb):
    """From a peak the plain out to the sight range is seen; from the plain only to the horizon."""
    v = int(glb.vertex_cells[:, 0].eq(glb.cell_of(P0)).nonzero()[0, 0])
    top = glb.vertices[v]
    peak = env_of(glb)
    peak["elevation_m"][0, glb.vertex_cells[v]] = 300.0       # four cells around the vertex: a pyramid
    height = senses.surface_at(glb, glb.at_vertices(C.surface_m(peak)), torch.tensor(0), top)
    assert float(height) == pytest.approx(300.0)
    cover = 1 - math.exp(-4.0)
    flat_x, high_x = [], []
    for env, out in ((env_of(glb), flat_x), (peak, high_x)):
        ahead = torch.stack([at(1000.0 * f, base=top) for f in senses.RAY_FRACTIONS[2:]])
        env["plant_c"][0, glb.cell_of(ahead)] = 2.0
        cr = population(N=1)
        put(cr, 0, top)
        out.append(ch(senses.observe(cr, glb, env, radius_m=R), 0, "plant"))
    assert flat_x[0] == 0.0                                  # 250 m and beyond lie below the horizon
    # from the 300 m peak (horizon 2.4 km) the plain at 500 m and 1 km is seen; the foot of the slope at
    # 250 m stays hidden below the slope's own line on the curved habitat
    assert high_x[0] == pytest.approx(2 / 5 * cover, rel=1e-4)


def test_food_items_fires_and_held_items(glb):
    env = env_of(glb, insolation=0.0)                        # night: fires stay visible by their own light
    cr = population(N=1)
    put(cr, 0, P0, acuity=1.0)
    pool = items.new_item_pool(1, 8)
    meat = materials.species_vector({"meat": 1.0})[None]
    items.spawn(pool, torch.tensor([0]), meat, torch.tensor([3.0]), at(30.0)[None], 290.0, -1)
    flint = materials.species_vector({"flint": 1.0})[None]
    held = items.spawn(pool, torch.tensor([0]), flint, torch.tensor([0.4]), P0[None], 290.0,
                       items.holder_code(torch.tensor([0]), 0, cr.K), sharp=torch.tensor([0.5]))
    cr.inv[0, 0, 0] = held[0]
    fires = items.new_fire_pool(1, 4)
    items.spawn_fire(fires, torch.tensor([0]), at(150.0, math.pi)[None], materials.species_vector({"wood": 2.0})[None],
                     1100.0)
    x = senses.observe(cr, glb, env, pool, fires, radius_m=R)
    e_meat = 3.0 * float(materials.FOOD_J_KG["meat"])
    assert ch(x, 0, "meat") == pytest.approx((1 - 30 / 50) * (1 - math.exp(-e_meat / senses.MEAT_SCALE_J)), rel=1e-3)
    assert ch(x, 4, "fire") == pytest.approx((1 - 150 / 1000) * min(1.0, (1100 - 290) / 1000), rel=1e-3)
    props = items.item_props(pool)
    assert me(x, "slot0_sharp") == pytest.approx(0.5)
    assert me(x, "slot0_hardness") == pytest.approx(float(props["tool_hardness"][0, held[0]]) / 10, rel=1e-5)
    assert me(x, "slot0_mass") == pytest.approx(math.tanh(0.4), rel=1e-5)
    assert me(x, "slot1_mass") == 0.0


def test_innate_wiring_reads_the_local_plant_input(glb):
    """Founders' forage logit rises with the plant cover under their feet (spec 2.10)."""
    gen = torch.Generator().manual_seed(0)
    g = brain.random_genome(gen, (1, 1), senses.IN_DIM, 16, brain.OUT_DIM, 8, "cpu",
                            innate={"plant_index": senses.PLANT_INDEX})
    cr = population(N=1, hidden=16)
    cr.genome.update(g)
    put(cr, 0, P0)
    outs = []
    for plant in (0.0, 2.0):
        x = senses.observe(cr, glb, env_of(glb, plant=plant), radius_m=R)
        out, _ = brain.think(cr.genome, brain.init_live(cr.genome), torch.zeros(1, 1, 16), x)
        outs.append(float(out[0, 0, brain.FORAGE_OUT]))
    assert outs[1] > outs[0]


# ------------------------------------------------------------------------------------ dtypes and self channels
def test_float64_fields_give_the_same_float32_input(glb):
    cr = population(N=3)
    put(cr, 0, P0)
    put(cr, 1, at(120.0, 1.0), mass=10.0)
    cr.loud[0, 1] = 1.0
    cr.calls[0, 1] = torch.linspace(-1, 1, brain.VOCAL_DIMS)
    env = env_of(glb, plant=1.0, soil=70.0)
    env64 = {k: (v.double() if v.is_floating_point() else v) for k, v in env.items()}
    x = senses.observe(cr, glb, env, radius_m=R)
    x64 = senses.observe(cr, glb, env64, radius_m=torch.tensor([R], dtype=torch.float64))
    assert x64.dtype == torch.float32 and torch.allclose(x64, x, atol=1e-5)


def test_slope_water_and_lake_channels(glb):
    # ground rising 20 % northwards (bilinear over the cells): slope ahead tanh(0.2) heading north, minus going south
    env = env_of(glb)
    env["elevation_m"][0] = 2000.0 + 0.2 * R * glb.lat
    cr = population(N=3)
    put(cr, 0, P0, heading=0.0)
    put(cr, 1, P0, heading=math.pi, mass=29.0)
    put(cr, 2, P0, heading=math.pi / 2, mass=31.0)
    x = senses.observe(cr, glb, env, radius_m=R)
    assert me(x, "slope", n=0) == pytest.approx(math.tanh(0.2), rel=0.02)
    assert me(x, "slope", n=1) == pytest.approx(-math.tanh(0.2), rel=0.02)
    assert abs(me(x, "slope", n=2)) < 0.01
    # in the sea: in_water 1, no fresh water underfoot; on land 0
    sea = env_of(glb, soil=150.0)
    here = glb.cell_of(P0)
    sea["land"][0, here] = False
    cr = population(N=2)
    put(cr, 0, P0)
    put(cr, 1, at(2000.0))
    x = senses.observe(cr, glb, sea, radius_m=R)
    assert me(x, "in_water", n=0) == 1.0 and me(x, "fresh", n=0) == 0.0
    assert me(x, "in_water", n=1) == 0.0 and me(x, "fresh", n=1) == 1.0
    # a lake: fresh water underfoot on dry soil, and lake cells ahead show in the forward water channel
    dry = env_of(glb)
    dry["lake"] = torch.zeros(1, glb.C, dtype=torch.bool)
    ahead = torch.stack([at(1000.0 * f) for f in senses.RAY_FRACTIONS])
    dry["lake"][0, glb.cell_of(ahead)] = True
    cr = population(N=1)
    put(cr, 0, P0)
    x = senses.observe(cr, glb, dry, radius_m=R)
    assert me(x, "fresh") == 1.0
    # seen within the 210 m horizon: the samples at 62.5 m (the animal's own lake cell) and 125 m; behind,
    # only its own cell
    assert ch(x, 0, "water") == pytest.approx(2 / 5, rel=1e-4) and ch(x, 4, "water") == pytest.approx(1 / 5, rel=1e-4)


# ------------------------------------------------------------------------------------ hearing at scale
def test_only_audible_pairs_get_a_line_of_sight(glb):
    """Spreading alone would let a 1 Pa call (voice 0.5) reach a hearing-2 ear 100 km away; with 5 dB/km
    of absorption it fades out before 6 km, so that pair is not tested at all."""
    env = env_of(glb)
    for d, audible in ((6000.0, False), (3000.0, True)):
        cr = population(N=2)
        put(cr, 0, P0, hearing=2.0)
        put(cr, 1, at(d, 2.0), voice=0.5)
        cr.loud[0, 1] = 1.0
        cr.calls[0, 1] = torch.ones(brain.VOCAL_DIMS) * 0.5
        x, info = senses.observe(cr, glb, env, radius_m=R, details=True)
        assert int(info["pairs"][0]) == int(audible)
        if audible:                                         # beyond the horizon: occluded, half the pressure
            w = expected_weight(d, p1=1.0 * senses.OCCLUSION, hearing=2.0)
            assert float(info["heard"][0, 0]) == pytest.approx(w, rel=1e-3) and 0 < w < 1
        else:
            assert float(info["heard"].abs().max()) == 0.0


def test_a_crowd_keeps_the_content_of_its_calls(glb):
    """64 callers 20-40 m to the east, each heard at full weight: the sector carries their weighted
    mean (not 64 x the vector clamped to +-1)."""
    env = env_of(glb)
    n = 64
    cr = population(N=n + 1)
    put(cr, 0, P0, heading=0.0)
    v = torch.linspace(-0.9, 0.9, brain.VOCAL_DIMS)
    for i in range(n):
        put(cr, i + 1, at(20.0 + 0.3 * i, math.pi / 2 + 0.3 * math.sin(i)))
    cr.loud[0, 1:] = 1.0
    cr.calls[0, 1::2] = v
    cr.calls[0, 2::2] = -0.5 * v
    x, info = senses.observe(cr, glb, env, radius_m=R, details=True)
    got = x[0, 0, senses.sector_index(2, "call0"):senses.sector_index(2, "call0") + brain.VOCAL_DIMS]
    assert float(info["heard"][0, 0]) == pytest.approx(n, rel=1e-4)
    assert torch.allclose(got, 0.25 * v, atol=1e-5)
    # a single caller below full weight is still heard as weight x vector
    one = population(N=2)
    put(one, 0, P0)
    put(one, 1, at(150.0, math.pi / 2))
    one.loud[0, 1] = 1.0
    one.calls[0, 1] = v
    x1 = senses.observe(one, glb, env, radius_m=R)
    w = expected_weight(150.0)
    assert torch.allclose(x1[0, 0, senses.sector_index(2, "call0"):senses.sector_index(2, "call0") + 8], w * v,
                          atol=1e-4)


def test_sight_lines_in_chunks_match(glb):
    gen = torch.Generator().manual_seed(1)
    P = 40
    p = torch.nn.functional.normalize(P0 + 0.02 * torch.randn(P, 3, generator=gen), dim=-1)
    q = torch.nn.functional.normalize(P0 + 0.02 * torch.randn(P, 3, generator=gen), dim=-1)
    env = env_of(glb)
    env["elevation_m"][0] = 30.0 * torch.rand(glb.C, generator=gen)
    surf_v = glb.at_vertices(C.surface_m(env))
    args = (glb, surf_v, torch.zeros(P, dtype=torch.long), p, q, torch.full((P,), 20.0), torch.full((P,), 20.0),
            torch.full((P,), R))
    whole = senses.line_of_sight(*args)
    assert torch.equal(senses.line_of_sight(*args, chunk=7), whole) and 0 < int(whole.sum()) < P


# ------------------------------------------------------------------------------------ richness
def test_richness_reads_any_species_order_relative_to_the_typical_deposit():
    W, Cc = 1, 4
    S = len(materials.SPECIES)
    stock = torch.zeros(W, Cc, S)
    stock[..., materials.IDX["basalt"]] = 2670.0
    stock[0, 1, materials.IDX["hematite"]] = 1.0
    stock[0, 2, materials.IDX["malachite"]] = 3.0
    stock[0, 3, materials.IDX["clay"]] = 50.0
    r = senses.richness_fields(stock)                          # all 31 species, inferred from the length
    assert r.shape == (W, Cc, 4)
    assert torch.allclose(r[0, :, 0], torch.full((Cc,), 1 - math.exp(-1.0)))
    assert r[0, :, 1].tolist() == pytest.approx([0.0, 1 - math.exp(-0.5), 1 - math.exp(-1.5), 0.0], rel=1e-5)
    assert r[0, 3, 2] == pytest.approx(1 - math.exp(-1.0)) and float(r[0, :3, 2].abs().max()) == 0.0
    # another order, passed explicitly, classifies the same
    order = list(reversed(materials.SPECIES))
    perm = [materials.IDX[s] for s in order]
    assert torch.allclose(senses.richness_fields(stock[..., perm], species=order), r)
    with pytest.raises(ValueError):
        senses.richness_fields(stock[..., :10])


# ------------------------------------------------------------------------------------ integration round (world review)
def test_a_floor_skips_only_lines_that_would_fail(glb):
    gen = torch.Generator().manual_seed(5)
    P = 300
    p = torch.nn.functional.normalize(P0 + 0.03 * torch.randn(P, 3, generator=gen), dim=-1)
    q = torch.nn.functional.normalize(P0 + 0.03 * torch.randn(P, 3, generator=gen), dim=-1)
    env = env_of(glb)
    env["elevation_m"][0] = 40.0 * torch.rand(glb.C, generator=gen)
    surf_v = glb.at_vertices(C.surface_m(env))
    h = 2.0 + 40.0 * torch.rand(P, generator=gen)
    args = (glb, surf_v, torch.zeros(P, dtype=torch.long), p, q, h, h.flip(0), torch.full((P,), R))
    whole = senses.line_of_sight(*args)
    fast = senses.line_of_sight(*args, floor=surf_v.amin(-1))
    assert torch.equal(fast, whole) and 0 < int(whole.sum()) < P


def test_field_rays_see_what_a_sampled_sight_line_sees(glb):
    """The running-horizon decision equals the sight-line test (heights interpolated with the sphere's sag)
    over the same earlier terrain samples."""
    gen = torch.Generator().manual_seed(6)
    env = env_of(glb)
    env["elevation_m"][0] = 60.0 * torch.rand(glb.C, generator=gen)
    surf_v = glb.at_vertices(C.surface_m(env))
    N = 6
    pos = torch.nn.functional.normalize(P0 + 0.01 * torch.randn(1, N, 3, generator=gen), dim=-1)
    head = 6.28 * torch.rand(1, N, generator=gen)
    r_v = torch.full((1, N), 2000.0)
    ground = senses.surface_at(glb, surf_v, torch.zeros(1, N, dtype=torch.long), pos)
    eye_h = ground + 0.62
    rays = senses.field_rays(glb, surf_v, pos, head, r_v, eye_h, torch.tensor([R]))
    h, d = rays["h_all"], rays["d_all"].expand_as(rays["h_all"])
    idx = [senses.RAY_TERRAIN.index(f) for f in senses.RAY_FRACTIONS]
    seen_any = 0
    for j, k in enumerate(idx):
        D = d[..., k]
        h_q = h[..., k] + senses.FIELD_HEIGHT_M
        ok = torch.ones_like(D, dtype=torch.bool)
        for i in range(k):
            t = d[..., i] / D
            line = (1 - t) * eye_h[..., None] + t * h_q - R * (D / R) ** 2 * t * (1 - t) / 2
            ok &= h[..., i] <= line + 1e-3
        assert torch.equal(rays["seen"][..., j], ok) or bool((rays["seen"][..., j] == ok).float().mean() > 0.99)
        seen_any += int(ok.sum())
    assert 0 < seen_any < N * senses.SECTORS * len(idx)
