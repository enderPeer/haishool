"""The senses against independent reference data and geometry (PLANET-V3-SPEC section 4).

Each test compares senses3 with something it was not calibrated on: ISO 9613 absorption tables, exact solid angles,
vocal-tract lengths, absolute luminances of flames and sunlit ground, the water surface over a sea of any depth, the
horizon behind a ridge, sight and hearing beyond the fine hash block, the intrinsic contrast threshold, submerged
items, realistic smell fields of patch.volatile_step, several planets' air in one batch, and the cost of many fires."""
from __future__ import annotations

import inspect
import math
import time
from dataclasses import fields, replace

import pytest
import torch

from haishool.life9.planet import constants as K
from haishool.life9.planet import materials
from haishool.life9.v3 import brain3, optics
from haishool.life9.v3 import patch as pt
from haishool.life9.v3 import senses3 as s3

R_781 = 5.674e6


def ch(name):
    return s3.CHANNELS.index(name)


def sec(x, name):
    return x[..., ch(f"s0_{name}")::s3.SECTOR_DIM][..., :8]


def flat_geom(A=1, n=16, L=256.0, z=0.0, sea=-1e4):
    return pt.geometry_from_elevation(torch.full((A, n, n), float(z)), L, sea, R_781)


def scene(geom, light=400.0, ground=0.1):
    return s3.Scene(light_vis=torch.full_like(geom.elev, light), ground_vis=torch.full_like(geom.elev, ground))


def bodies(pos, heading=0.0, mass=0.1, eye=1e-2, ear=1e-2, voice=1e-2, speed=0.0, loud=0.0, **over):
    pos = torch.as_tensor(pos, dtype=torch.float32)
    A, N = pos.shape[:2]
    full = lambda v: torch.as_tensor(v, dtype=torch.float32).expand(A, N).clone()   # noqa: E731
    M = full(mass)
    d = dict(alive=torch.ones(A, N, dtype=torch.bool), pos=pos, heading=full(heading), mass_kg=M, speed_m_s=full(speed),
             body_k=full(305.0), damage=full(0.0), reserve_j=full(5e3), reserve_cap_j=full(1e4), water_kg=0.6 * M,
             water_norm_kg=0.6 * M, gut_kg=0.01 * M, gut_cap_kg=0.05 * M, eye=full(eye), ear=full(ear),
             voice=full(voice), muscle=full(100.0), loudness=full(loud), calls=torch.ones(A, N, brain3.VOCAL_DIMS) * 0.5)
    d.update(over)
    return s3.Bodies(**d)


def obs(geom, b, sc, **kw):
    kw.setdefault("air", s3.Air.earth(geom.A))
    return s3.observe(geom, b, sc, **kw)


# ------------------------------------------------------------------------------------------------ hearing
#: ISO 9613-2:1996 table 2, atmospheric attenuation (dB/km) at 101.325 kPa, octave bands 1, 2, 4 and 8 kHz
ISO_TABLE = {(283.15, 0.7): (3.7, 9.7, 32.8, 117.0), (293.15, 0.7): (5.0, 9.0, 22.9, 76.6),
             (303.15, 0.7): (7.4, 12.7, 23.1, 59.3), (288.15, 0.2): (8.2, 28.2, 88.8, 202.0),
             (288.15, 0.5): (4.2, 10.8, 36.2, 129.0), (288.15, 0.8): (4.1, 8.3, 23.7, 82.8)}


def test_absorption_matches_the_iso_tables():
    f = torch.tensor([[1000.0, 2000.0, 4000.0, 8000.0]])
    for (T, rh), want in ISO_TABLE.items():
        got = s3.absorption_db_m(f, s3.Air.earth(1, t_k=T, rel_humidity=rh))[0] * 1000
        for g, w in zip(got.tolist(), want):
            assert g == pytest.approx(w, rel=0.03), (T, rh, g, w)
    # ultrasound at 20 C, 70 %: the relaxation terms saturate, so it grows far slower than f^2 from 1 kHz
    air = s3.Air.earth(1, t_k=293.15, rel_humidity=0.7)
    u = s3.absorption_db_m(torch.tensor([[16e3, 32e3, 64e3]]), air)[0]
    for g, w in zip(u.tolist(), (0.28, 0.92, 2.34)):
        assert g == pytest.approx(w, rel=0.03)
    a1 = float(s3.absorption_db_m(1000.0, air)[0])
    assert float(u[1]) < 0.25 * a1 * 32 ** 2                         # the old f^2 scaling gave 5.1 dB/m at 32 kHz
    # the air matters: far above the relaxation frequencies the classical term rules, about 1 / p; no O2 removes its
    # relaxation
    thin = s3.Air.earth(1, t_k=293.15, rel_humidity=0.7, pressure_pa=50e3)
    assert 1.6 < float(s3.absorption_db_m(256e3, thin)[0]) / float(s3.absorption_db_m(256e3, air)[0]) < 2.0
    no_o2 = replace(air, x_o2=torch.zeros_like(air.x_o2))             # O2's relaxation rules near 8 kHz
    assert float(s3.absorption_db_m(8000.0, no_o2)[0]) < 0.4 * float(s3.absorption_db_m(8000.0, air)[0])


def test_a_founder_call_carries_ten_metres():
    """The review's case: a 20 g founder with voice 1e-3 calls near 11 kHz; at 10 m it loses about 1 dB to the air
    (not 51 dB) and a 20 g listener with an ear gene of 1e-3 hears it."""
    f = s3.call_frequency(1e-3, 0.02, float(s3.Air.earth(1).sound_speed()[0]))
    assert 8e3 < float(f) < 14e3
    loss = 10 * float(s3.absorption_db_m(float(f), s3.Air.earth(1))[0])
    assert loss < 2.0
    geom = flat_geom()
    b = bodies([[[100.0, 100.0], [110.0, 100.0]]], mass=0.02, ear=1e-3, voice=1e-3, loud=1.0)
    x = obs(geom, b, scene(geom))
    assert float(x[0, 0, ch("s0_call0")]) > 0 and float(x[0, 0, ch("s0_noise")]) > 0


def test_call_frequency_and_loudness_share_one_calibration_body():
    """The voice organ is a tube of its own volume (aspect 4.4): the loudness calibration's 3 g/kg gives a 70 kg body a
    17 cm tract (F1 about 500 Hz) and a 30 kg one about 13 cm (children of 30 kg: about 13 cm), and that 30 kg body
    calls at about 100 dB at 1 m."""
    assert float(s3.vocal_tract_m(3e-3, 70.0)) == pytest.approx(0.17, rel=0.03)
    assert float(s3.call_frequency(3e-3, 70.0, 343.0)) == pytest.approx(343.0 / 0.68, rel=0.03)
    assert 0.12 < float(s3.vocal_tract_m(3e-3, 30.0)) < 0.14
    p = s3.call_acoustic_power(1.0, 3e-3, 200.0, 30.0)
    assert 20 * math.log10(math.sqrt(float(p) * 1.204 * 343.2 / (4 * math.pi)) / 20e-6) == pytest.approx(100, abs=1.5)
    # a tube 4.4 times longer than wide: l^3 = 4 a^2 V / pi
    v = 3e-3 * 70.0 / s3.TISSUE_DENSITY
    l = float(s3.vocal_tract_m(3e-3, 70.0))
    width = math.sqrt(4 * v / (math.pi * l))
    assert l / width == pytest.approx(4.4, rel=1e-4)


def test_arenas_with_different_air_in_one_batch():
    """Each arena hears through its own air: a batch of Earth's air and a thin cold CO2 air equals the two run alone."""
    geom2 = flat_geom(A=2)
    b2 = bodies([[[100.0, 100.0], [109.0, 100.0], [100.0, 92.0]]] * 2, mass=0.2, ear=3e-2, voice=2e-2, loud=0.8,
                speed=0.5)
    pp = torch.tensor([[78084.0, 20948.0, 28.0, 934.0], [300.0, 0.0, 12000.0, 200.0]])
    air = s3.Air.from_partial(pp, torch.tensor([288.0, 250.0]), torch.tensor([9.81, 3.7]), torch.tensor([1000.0, 0.0]))
    x = s3.observe(geom2, b2, scene(geom2), air=air)
    for a in range(2):
        g1 = flat_geom(A=1)
        b1 = s3.Bodies(**{f.name: (None if getattr(b2, f.name) is None else getattr(b2, f.name)[a:a + 1])
                          for f in fields(s3.Bodies)})
        a1 = s3.Air(**{f.name: getattr(air, f.name)[a:a + 1] for f in fields(s3.Air)})
        assert torch.allclose(x[a:a + 1], s3.observe(g1, b1, scene(g1), air=a1), atol=1e-6), a
    hear = [ch(f"s{s}_noise") for s in range(8)]
    assert not torch.allclose(x[0][..., hear], x[1][..., hear])         # dry CO2 air absorbs differently
    # and each arena's absorption comes from its own air, not arena 0's
    f = s3.call_frequency(b2.voice, b2.mass_kg, air.sound_speed()[:, None])
    a = s3.absorption_db_m(f, air)
    assert float(a[0, 0]) != pytest.approx(float(a[1, 0]), rel=0.1)


# ------------------------------------------------------------------------------------------------ vision geometry
def test_solid_angle_is_exact_for_small_targets():
    """Omega = 2 pi s^2 / (1 + sqrt(1 - s^2)) in float32 against float64 for small s; a 2 g body's solid angle falls
    monotonically out to 60 m (the old form stalled and reached 0 for 4 mm items beyond 17 m)."""
    for s in (1e-5, 1e-4, 2.4e-4, 1e-3, 1e-2, 0.3, 0.9, 1.0):
        want = 2 * math.pi * (1 - math.sqrt(1 - s * s)) if s > 1e-3 else math.pi * s * s * (1 + s * s / 4)
        got = float(s3.sphere_solid_angle(torch.tensor(s, dtype=torch.float32), torch.tensor(1.0)))
        assert got == pytest.approx(want, rel=1e-5), s
    r = float(s3.body_radius(torch.tensor(0.002)))
    d = torch.linspace(5.0, 60.0, 200)
    om = s3.sphere_solid_angle(torch.full_like(d, r), d)
    assert bool((om[1:] < om[:-1]).all())
    assert torch.allclose(om, math.pi * r * r / d.double().pow(2).float(), rtol=1e-5)
    assert float(s3.sphere_solid_angle(torch.tensor(0.002), torch.tensor(17.0))) > 0


def test_ground_channel_over_the_sea_is_the_water_surface():
    """A floating body's eye is 2 r above the water whatever the depth: over a 50 m and a 0.5 m deep sea it sees the
    same water as over flat land of the same reflectance (the old eye stood on a 50 m mast)."""
    dark = s3.night_light()
    out = []
    for z in (-50.0, -0.5, 0.0):
        geom = flat_geom(1, 32, 512.0, z=z, sea=0.0 if z < 0 else -1e4)
        b = bodies([[[256.0, 256.0]]], mass=0.1, eye=1e-2)
        out.append(sec(obs(geom, b, scene(geom, dark, 0.06))[0, 0], "ground"))
    assert float(out[0].min()) > 0
    assert torch.allclose(out[0], out[1], rtol=1e-5) and torch.allclose(out[0], out[2], rtol=1e-5)


def test_ground_rays_stop_at_a_ridge():
    """A 30 m ridge 48 m ahead: the ray sees the near ground (dark) and not the bright ground behind the ridge."""
    n, L = 32, 512.0
    xs = (torch.arange(n) + 0.5) * L / n
    ridge = (30.0 * torch.exp(-((xs - 248.0) / 10.0) ** 2))[None, :, None].expand(1, n, n).clone()
    geom = pt.geometry_from_elevation(ridge, L, -1e4, R_781)
    flat = flat_geom(1, n, L)
    ground = torch.full((1, n, n), 0.05)
    ground[0, 16:] = 0.9                                                 # x >= 256 is bright
    b = bodies([[[200.0, 300.0]]], heading=0.0)
    light = torch.full((1, n, n), 400.0)
    g_ridge = float(obs(geom, b, s3.Scene(light, ground))[0, 0, ch("s0_ground")])
    g_flat = float(obs(flat, b, s3.Scene(light, ground))[0, 0, ch("s0_ground")])
    # samples at 8, 16, 32, 64 and 128 m: x = 208, 216, 232 (dark, seen), 264 and 328 (bright, hidden by the ridge)
    assert g_flat == pytest.approx((3 * 0.05 + 2 * 0.9) / 5, rel=1e-3)
    assert g_ridge == pytest.approx(3 * 0.05 / 5, rel=1e-3)


def test_sight_and_hearing_reach_past_the_fine_block_alike_on_every_side():
    """A 1 kg body 60 m east or west of a 50 g eye (four fine cells away, outside the 3 x 3 block) is seen the same
    from both sides by day; a loud caller 100 m away is heard; info reports the far hash."""
    geom = flat_geom(1, 128, 2048.0)
    sizes = []
    for dxm, sector in ((60.0, 0), (-60.0, 4)):
        b = bodies([[[1000.0, 1000.0], [1000.0 + dxm, 1000.0]]], mass=torch.tensor([[0.5, 1.0]]), eye=0.1,
                   speed=torch.tensor([[0.0, 1.0]]))
        x, info = obs(geom, b, scene(geom, 400.0, 0.1), details=True)
        assert float(info["seen_bodies"][0, 0]) == 1 and int(info["body_count"][0, 0]) == 0
        sizes.append(float(sec(x[0, 0], "size")[sector]))
        assert float(sec(x[0, 0], "motion")[sector]) > 0
    assert sizes[0] > 0 and sizes[0] == pytest.approx(sizes[1], rel=1e-5)
    # the same 1 kg body anywhere within the fine cell of the observer: no grid artefact in range
    for ox in (1000.5, 1008.0, 1015.5):
        b = bodies([[[ox, 1000.0], [ox + 60.0, 1000.0]]], mass=torch.tensor([[0.5, 1.0]]), eye=0.1)
        _, info = obs(geom, b, scene(geom, 400.0, 0.1), details=True)
        assert float(info["seen_bodies"][0, 0]) == 1, ox
    b = bodies([[[1000.0, 1000.0], [1100.0, 1000.0]]], mass=torch.tensor([[0.5, 0.5]]), ear=3e-2, voice=3e-2,
               loud=torch.tensor([[0.0, 1.0]]))
    x, info = obs(geom, b, scene(geom), details=True)
    assert float(x[0, 0, ch("s0_call0")]) > 0 and float(info["heard"][0, 0]) == 1
    assert bool(info["beyond"][0, 1])                     # a loud call carries past the far block's sure 128 m
    assert info["far_radius_m"] == pytest.approx(128.0)


def test_reach_bounds_are_upper_bounds():
    """vision_reach and sound_reach bound the exact detection tests: nothing is detected beyond them."""
    gen = torch.Generator().manual_seed(0)
    n = 4000
    r = s3.SenseRules()
    eye = 10 ** (-6 + 4 * torch.rand(n, generator=gen))
    theta, G = s3.eye_optics(eye)
    om_res = math.pi / 4 * theta ** 2
    rad = 10 ** (-3 + 2 * torch.rand(n, generator=gen))
    E = 10 ** (-5 + 7 * torch.rand(n, generator=gen))
    lb = (0.05 + 0.5 * torch.rand(n, generator=gen)) * E / math.pi
    lt = (0.05 + 0.5 * torch.rand(n, generator=gen)) * E / math.pi
    reach = s3.vision_reach((lt - lb).abs(), lb, rad, G, om_res, r)
    for f in (0.999, 1.001, 1.5, 3.0):
        d = (reach * f).clamp_min(1e-3)
        om = s3.sphere_solid_angle(rad, d)
        dl = (lt - lb).abs()
        nb = lb * torch.maximum(om, om_res) * G
        seen = (s3.rose_snr(dl * om * G, nb) >= r.rose_k) & (dl * om >= r.c_min * lb * torch.maximum(om, om_res))
        if f > 1:
            assert not bool((seen & (reach > 0)).any()), f
        else:
            assert float(seen[reach > 0].float().mean()) > 0.2           # and the bound is not loose by far
    P = 10 ** (-9 + 7 * torch.rand(n, generator=gen))
    alpha = 10 ** (-3 + 2 * torch.rand(n, generator=gen))
    ith = 10 ** (-13 + 3 * torch.rand(n, generator=gen))
    rs = s3.sound_reach(P, alpha, ith, torch.full_like(P, 0.01))
    for f, heard_ok in ((0.999, True), (1.001, False)):
        d = (rs * f).clamp_min(0.01)
        i = P / (4 * math.pi * d * d) * torch.pow(10.0, -alpha * d / 10)
        if heard_ok:
            assert bool((i[rs > 0.02] >= ith[rs > 0.02] * 0.999).all())
        else:
            assert not bool(((i >= ith) & (rs > 0.02)).any())


def test_contrast_threshold_hides_faint_targets_by_day():
    """A 20 mg eye in daylight: photon noise alone would show a target 1 % brighter than the ground; the intrinsic
    contrast threshold (2 %) hides it, while a 50 % one is seen."""
    geom = flat_geom()
    for vis, want in ((0.303, 0), (0.45, 1)):
        b = bodies([[[100.0, 100.0], [112.0, 100.0]]], mass=0.02, eye=1e-3,
                   surface_vis=torch.tensor([[0.3, vis]]))
        _, info = obs(geom, b, scene(geom, 400.0, 0.3), details=True)
        assert float(info["seen_bodies"][0, 0]) == want, vis


def test_a_still_body_looks_and_feels_like_a_stone_of_its_size():
    """Objects share their channels: a still, silent body and an item of the same size, reflectance, mass, hardness
    and temperature give the observer the same input."""
    geom = flat_geom()
    obs_pos = [100.0, 100.0]
    M_obs, M_t = 0.5, 0.3
    r_obs = float(s3.body_radius(torch.tensor(M_obs)))
    r_t = float(s3.body_radius(torch.tensor(M_t)))
    at = [100.0 + r_obs + 0.5 * r_obs + r_t * 0.5, 100.0]                # in contact with the observer's mouth
    for where in (at, [104.0, 103.0]):
        b2 = bodies([[obs_pos, where]], mass=torch.tensor([[M_obs, M_t]]), surface_vis=torch.tensor([[0.3, 0.12]]),
                    body_k=torch.tensor([[305.0, 300.0]]))
        x_body = obs(geom, b2, scene(geom, 400.0, 0.3))[0, 0]
        b1 = bodies([[obs_pos]], mass=M_obs, surface_vis=torch.tensor([[0.3]]), body_k=torch.tensor([[305.0]]))
        it = s3.Items(pos=torch.tensor([[where]]), alive=torch.tensor([[True]]), mass=torch.tensor([[M_t]]),
                      temp_k=torch.tensor([[300.0]]), sharp=torch.zeros(1, 1), radius_m=torch.tensor([[r_t]]),
                      vis=torch.tensor([[0.12]]), emissivity=torch.tensor([[0.95]]),
                      hardness=torch.tensor([[s3.BODY_HARDNESS]]), density=torch.tensor([[s3.TISSUE_DENSITY]]))
        x_item = obs(geom, b1, scene(geom, 400.0, 0.3), items=it)[0, 0]
        assert float(x_body[:s3.INTERO0].abs().sum()) > 0
        assert torch.allclose(x_body, x_item, atol=1e-6), (x_body - x_item).abs().argmax()
    assert float(x_body[ch("s0_size")] + sec(x_body, "size").sum()) > 0


def test_submerged_items_lie_on_the_bed_and_dim_with_depth():
    """At dusk a flint under 3 m of pond water (denser than water: on the bed, light x exp(-2 K_d h)) is lost while the
    same flint on dry land is seen; a piece of wood floats and is seen on the pond."""
    geom = flat_geom()
    state = pt.init_state(geom)
    state.pond[0, 6, 6] = 3000.0                                       # 3 m of water over the cell of (100, 100)
    comp = torch.zeros(1, 2, materials.S)
    comp[0, 0, materials.IDX["flint"]] = 1.0
    comp[0, 1, materials.IDX["wood"]] = 1.0
    b = bodies([[[92.0, 100.0]]], eye=1e-2)
    dusk = scene(geom, 3e-4, 0.06)
    seen = {}
    for name, k in (("flint", 0), ("wood", 1)):
        for place, p in (("pond", [100.0, 100.0]), ("land", [100.0, 140.0])):
            it = s3.Items(pos=torch.tensor([[p]]), alive=torch.tensor([[True]]), mass=torch.tensor([[0.2]]),
                          temp_k=torch.tensor([[290.0]]), sharp=torch.zeros(1, 1), comp=comp[:, k:k + 1])
            bb = b if place == "pond" else replace(b, pos=torch.tensor([[[92.0, 140.0]]]))
            _, info = obs(geom, bb, dusk, state=state, items=it, details=True)
            seen[name, place] = float(info["seen_items"][0, 0])
    assert seen["flint", "land"] == 1 and seen["flint", "pond"] == 0
    assert seen["wood", "land"] == 1 and seen["wood", "pond"] == 1


# ------------------------------------------------------------------------------------------------ emitted light
def test_a_flame_darker_than_sunlit_ground_is_not_a_beacon_by_day():
    """Absolute luminances: a 1,100 K flame (about 0.12 W/m^2/sr) against sunlit ground (0.1 x 400 / pi = 12.7) reads
    nothing by day at 5 and 20 m; at night a resolved flame reads its luminance over the ground's,
    log10(1 + L_f / L_b) / 8."""
    l_f = float(optics.visible_exitance(torch.tensor(1100.0), optics.FLAME_EMISSIVITY)) / math.pi
    assert l_f < 0.1 * 0.1 * 400.0 / math.pi
    geom = flat_geom(1, 32, 512.0)
    fires = s3.Fires(pos=torch.tensor([[[205.0, 200.0], [300.0, 400.0]]]), alive=torch.tensor([[True, False]]),
                     temp_k=torch.tensor([[1100.0, 1100.0]]), area_m2=torch.tensor([[0.785, 0.785]]))
    b = bodies([[[200.0, 200.0], [185.0, 200.0]]], mass=0.1, eye=1e-3)
    day = sec(obs(geom, b, scene(geom, 400.0, 0.1), fires=fires)[0], "emitted")
    assert float(day.max()) < 0.1
    night = sec(obs(geom, b, scene(geom, s3.night_light(), 0.1), fires=fires)[0], "emitted")
    l_b = 0.1 * s3.night_light() / math.pi
    assert float(night[0, 0]) == pytest.approx(math.log10(1 + l_f / l_b) / 8, rel=1e-3)
    assert float(night[1, 0]) == pytest.approx(float(night[0, 0]), rel=1e-4)   # resolved at 5 and 20 m alike


def test_beacon_distance_stays_readable_at_night():
    """A small eye at night: a fire at 30 m (resolved) and at 600 m (unresolved, diluted over the blur spot) give
    different emitted codes (x / (1 + x) read 1.000 for both)."""
    geom = flat_geom(1, 128, 2048.0)
    fires = s3.Fires(pos=torch.tensor([[[1030.0, 1000.0]]]), alive=torch.tensor([[True]]),
                     temp_k=torch.tensor([[1100.0]]), area_m2=torch.tensor([[0.785]]))
    em = []
    for d in (30.0, 600.0):
        b = bodies([[[1030.0 - d, 1000.0]]], mass=0.02, eye=1e-3)
        em.append(float(obs(geom, b, scene(geom, s3.night_light(), 0.1), fires=fires)[0, 0, ch("s0_emitted")]))
    assert em[0] > em[1] + 0.05 and em[1] > 0.3


def test_many_fires_cost_a_bounded_number_of_sight_lines(monkeypatch):
    """256 fires per arena at night: each body line-tests the nearest-ranked fire_k sources per sector, and a further
    fire_k per round only in sectors with no clear one yet; the rest are culled (and still light it), so the number of
    fire sight lines is bounded by N x 8 x fire_k x fire_rounds whatever the number of fires."""
    gen = torch.Generator().manual_seed(4)
    A, N, P, L = 2, 1024, 256, 2048.0
    geom = pt.geometry_from_elevation(pt.fractal_detail(A, 128, gen, 0.8) * 5.0, L, -1e4, R_781)
    M = 0.002 * torch.exp(torch.rand(A, N, generator=gen) * math.log(100))
    b = bodies(torch.rand(A, N, 2, generator=gen) * L, mass=M, eye=1e-2)
    b = replace(b, heading=torch.rand(A, N, generator=gen) * 6.28)
    fire_lines = []
    real = s3._los_masked

    def counting(geom_, p, q, h_p, h_q, mask, prules):
        if inspect.stack()[1].function == "_fire_light":
            fire_lines.append(int(mask.sum()))
        return real(geom_, p, q, h_p, h_q, mask, prules)

    monkeypatch.setattr(s3, "_los_masked", counting)
    dark = scene(geom, s3.night_light(), 0.15)
    r = s3.SenseRules()
    counts, times, outs = {}, {}, {}
    for nf in (8, P):
        fires = s3.Fires(pos=torch.rand(A, nf, 2, generator=gen) * L, alive=torch.ones(A, nf, dtype=torch.bool),
                         temp_k=torch.full((A, nf), 1100.0), area_m2=torch.full((A, nf), 0.785))
        fire_lines.clear()
        t0 = time.perf_counter()
        outs[nf] = obs(geom, b, dark, fires=fires, details=True)
        times[nf] = time.perf_counter() - t0
        counts[nf] = (fire_lines[0], sum(fire_lines), len(fire_lines))
    for nf in (8, P):
        first, total, calls = counts[nf]
        assert 0 < first <= A * N * s3.SECTORS * r.fire_k                 # round one: fire_k per sector
        assert total <= A * N * s3.SECTORS * r.fire_k * r.fire_rounds and calls <= r.fire_rounds
    x, info = outs[P]
    assert float(info["fires_culled"].float().mean()) > 100 and float(sec(x, "emitted").max()) > 0.5
    assert times[P] < 20.0
    # the cull is exact when no sector holds more than fire_k sources, and otherwise only drops sources from the
    # emitted sum (they still light the body)
    fires2 = s3.Fires(pos=torch.rand(1, 2, 2, generator=gen) * L, alive=torch.ones(1, 2, dtype=torch.bool),
                      temp_k=torch.full((1, 2), 1100.0), area_m2=torch.full((1, 2), 0.785))
    g1 = pt.geometry_from_elevation(geom.elev[:1].clone(), L, -1e4, R_781)
    b1 = s3.Bodies(**{f.name: (None if getattr(b, f.name) is None else getattr(b, f.name)[:1, :256])
                      for f in fields(s3.Bodies)})
    sc1 = scene(g1, s3.night_light(), 0.15)
    full = s3.SenseRules(fire_k=P, fire_rounds=1)
    assert torch.equal(obs(g1, b1, sc1, fires=fires2), obs(g1, b1, sc1, fires=fires2, rules=full))
    fires64 = s3.Fires(pos=torch.rand(1, 64, 2, generator=gen) * L, alive=torch.ones(1, 64, dtype=torch.bool),
                       temp_k=torch.full((1, 64), 1100.0), area_m2=torch.full((1, 64), 0.785))
    xc, xf = obs(g1, b1, sc1, fires=fires64), obs(g1, b1, sc1, fires=fires64, rules=full)
    assert bool((sec(xc, "emitted") <= sec(xf, "emitted") + 1e-6).all())
    assert bool((xc[..., ch("intero_light")] >= xf[..., ch("intero_light")] - 1e-6).all())
    lost = (sec(xf, "emitted") > 0) & (sec(xc, "emitted") == 0)          # hidden behind 6 nearer fires (reported)
    assert float(lost.float().mean()) < 0.05
    assert float((sec(xc, "emitted") - sec(xf, "emitted")).abs().mean()) < 0.05


# ------------------------------------------------------------------------------------------------ smell
def test_realistic_smell_fields_are_in_the_code_range():
    """patch.volatile_step's fields reach the nose: a 10 kg carcass alone in a 16 m cell for 10 h codes about 0.2
    there and 0.1 three cells away (it was 150 x below its range); dense plant cover codes below 1."""
    geom = flat_geom(1, 128, 2048.0)
    vol = pt.init_volatiles(geom)
    carc = torch.zeros(1, 128, 128)
    carc[0, 64, 64] = 10.0 / geom.cell_m2
    plant = torch.full((1, 128, 128), 0.3)
    for _ in range(10):
        vol = pt.volatile_step(vol, geom, 3600.0, plant_c=plant, carcass_kg_m2=carc)
    b = bodies([[[64.5 * 16, 64.5 * 16], [67.5 * 16, 64.5 * 16]]], heading=math.pi)
    x = obs(geom, b, scene(geom), vol=vol)[0]
    assert float(x[0, ch("smell_carcass")]) > 0.15 and float(x[1, ch("smell_carcass")]) > 0.07
    assert float(x[1, ch("smell_carcass_grad")]) > 0                       # facing west, towards the carcass
    assert 0.5 < float(x[0, ch("smell_plant")]) < 1.0
