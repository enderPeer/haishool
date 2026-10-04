"""Tests of the v3 senses (PLANET-V3-SPEC section 4): physical channels only, no class labels."""
from __future__ import annotations

import math
import re
from dataclasses import fields, replace
from pathlib import Path

import pytest
import torch

from haishool.life9.planet import constants as K
from haishool.life9.planet import materials
from haishool.life9.v3 import brain3, optics
from haishool.life9.v3 import patch as pt
from haishool.life9.v3 import senses3 as s3

TAGS = {"chain", "derived", "reference", "new_rule"}
R_781 = 5.674e6
ROOT = Path(__file__).resolve().parents[3]


def _tag_ok(tag: str) -> bool:
    return bool(tag) and all(part in TAGS for part in tag.split("+"))


def obs(geom, b, scene, **kw):
    """observe with Earth's air in every arena unless the test passes its own."""
    kw.setdefault("air", s3.Air.earth(geom.A))
    return s3.observe(geom, b, scene, **kw)


def make_bodies(A, N, L, gen, mass=(0.005, 0.5), alive_p=0.9, **over):
    """Random bodies (tests): every field of s3.Bodies as a plain tensor."""
    lo, hi = mass
    M = lo * torch.exp(torch.rand(A, N, generator=gen) * math.log(hi / lo))
    d = dict(alive=torch.rand(A, N, generator=gen) < alive_p, pos=torch.rand(A, N, 2, generator=gen) * L,
             heading=torch.rand(A, N, generator=gen) * 2 * math.pi, mass_kg=M,
             speed_m_s=torch.rand(A, N, generator=gen) * 0.5, body_k=290.0 + 20 * torch.rand(A, N, generator=gen),
             damage=0.2 * torch.rand(A, N, generator=gen), reserve_j=1e4 * torch.rand(A, N, generator=gen),
             reserve_cap_j=torch.full((A, N), 1e4), water_kg=0.6 * M, water_norm_kg=0.65 * M, gut_kg=0.01 * M,
             gut_cap_kg=0.05 * M, eye=10 ** (-3 + 2 * torch.rand(A, N, generator=gen)),
             ear=10 ** (-3 + 2 * torch.rand(A, N, generator=gen)),
             voice=10 ** (-3 + 2 * torch.rand(A, N, generator=gen)), muscle=50 + 150 * torch.rand(A, N, generator=gen),
             loudness=torch.rand(A, N, generator=gen), calls=torch.rand(A, N, brain3.VOCAL_DIMS, generator=gen) * 2 - 1)
    d.update(over)
    return s3.Bodies(**d)


def flat_geom(A=1, n=16, L=256.0):
    return pt.geometry_from_elevation(torch.zeros(A, n, n), L, -1e4, R_781)


def uniform_scene(geom, light=400.0, ground=0.1):
    return s3.Scene(light_vis=torch.full_like(geom.elev, light), ground_vis=torch.full_like(geom.elev, ground))


def simple_bodies(pos, heading=0.0, mass=0.1, eye=1e-2, ear=1e-2, voice=1e-2, speed=0.3, loud=1.0, calls=None,
                  **over):
    """Bodies at given positions [A, N, 2] with equal genes (tests)."""
    pos = torch.as_tensor(pos, dtype=torch.float32)
    A, N = pos.shape[:2]
    full = lambda v: torch.as_tensor(v, dtype=torch.float32).expand(A, N).clone()   # noqa: E731
    M = full(mass)
    d = dict(alive=torch.ones(A, N, dtype=torch.bool), pos=pos, heading=full(heading), mass_kg=M, speed_m_s=full(speed),
             body_k=full(305.0), damage=full(0.0), reserve_j=full(5e3), reserve_cap_j=full(1e4), water_kg=0.6 * M,
             water_norm_kg=0.6 * M, gut_kg=0.01 * M, gut_cap_kg=0.05 * M, eye=full(eye), ear=full(ear),
             voice=full(voice), muscle=full(100.0), loudness=full(loud),
             calls=torch.ones(A, N, brain3.VOCAL_DIMS) * 0.5 if calls is None else calls)
    d.update(over)
    return s3.Bodies(**d)


def ch(name):
    return s3.CHANNELS.index(name)


def sec(x, name):
    """The 8 sector values of a vision or hearing channel: x [..., IN_DIM] -> [..., 8]."""
    return x[..., ch(f"s0_{name}")::s3.SECTOR_DIM][..., :8]


# ------------------------------------------------------------------------------------------------ provenance, layout
def test_every_constant_has_provenance():
    for f in fields(s3.SenseRules):
        tag, note = s3.PROVENANCE[f.name]
        assert note and _tag_ok(tag), f.name
    for name, (tag, note) in s3.PROVENANCE.items():
        assert note and _tag_ok(tag), name
    for name in ("SECTORS", "VOCAL", "GRIPS", "TISSUE_DENSITY", "T_COLD_DEATH_K", "T_HOT_DEATH_K", "T_LIFE_MID_K",
                 "T_LIFE_HALF_K", "MAX_EFFICACY_LM_W", "V_INTEGRAL_M", "LUMEN_PER_W", "STP_T_K", "SMELL_FORMULA",
                 "HUMAN_PANEL_FIELDS", "ISO_P_REF_PA", "ISO_T0_K", "ISO_T01_K", "ISO_X_O2", "ISO_X_N2",
                 "ISO_MOLAR_KG", "ISO_GAMMA", "BODY_HARDNESS"):
        assert name in s3.PROVENANCE, name
    # the frequency rule is not a bare reference: the tube model is ours
    assert s3.PROVENANCE["call_frequency"][0] == "reference+new_rule"


def test_layout():
    assert s3.IN_DIM == len(s3.CHANNELS) == len(set(s3.CHANNELS)) == 157
    assert s3.SECTOR_DIM == 7 + s3.VOCAL + 1
    assert s3.CHANNELS[s3.INTERO_SLICE] == tuple(f"intero_{n}" for n in brain3.INTERO_NAMES)
    assert s3.CHANNELS[s3.SMELL0] == "smell_plant" and s3.CHANNELS[s3.TOUCH0] == "touch_plant"
    assert s3.CHANNELS[s3.HELD0] == "held0_mass"
    assert ch("s3_call0") == 3 * s3.SECTOR_DIM + 7


def test_no_class_labels_and_no_forbidden_words():
    """No channel names a class of thing (not even body against item: objects share their channels); the source never
    builds a label (is_food, is_kin, ...) and holds none of the section-9 words."""
    labels = ("food", "kin", "prey", "predator", "threat", "stone", "ore", "meat", "sea", "edible", "enemy", "mate",
              "friend", "lineage", "species", "animal", "alive", "living", "fire")
    for name in s3.CHANNELS:
        for word in labels:
            assert word not in name.split("_"), name
    # what is outside the body is an object, whatever it is (interoception's 'body_temp' is the body's own state)
    for name in s3.CHANNELS[:s3.INTERO0]:
        for word in ("body", "item", "object"):
            assert word not in name.split("_"), name
    text = (ROOT / "haishool/life9/v3/senses3.py").read_text(encoding="utf-8")
    assert re.search(r"\bis_[a-z]+", text) is None
    for word in ("seed_floor", "innate", "forage_bias", "background_drink", "time_budget", "pedigree", "relatedness",
                 "LIFESPAN", "MATURITY", "reward", "geom.sea\\b", "water_at", "lineage\\[", "nan_to_num"):
        assert re.search(word, text, re.IGNORECASE if word.isalpha() else 0) is None, word


# ------------------------------------------------------------------------------------------------ shapes, determinism
def _full_case(seed=0, A=2, N=120, n=16, L=256.0, I=60, Fn=3):
    gen = torch.Generator().manual_seed(seed)
    z = pt.fractal_detail(A, n, gen, 0.8) * 3.0
    geom = pt.geometry_from_elevation(z, L, -1e4, R_781)
    b = make_bodies(A, N, L, gen, held=torch.randint(-1, I, (A, N, 2), generator=gen))
    comp = torch.zeros(A, I, materials.S)
    for k, sp in enumerate(("flint", "wood", "basalt", "meat")):
        comp[:, k::4, materials.IDX[sp]] = 1.0
    items = s3.Items(pos=torch.rand(A, I, 2, generator=gen) * L, alive=torch.rand(A, I, generator=gen) < 0.8,
                     mass=torch.rand(A, I, generator=gen) * 0.5,
                     temp_k=280 + 900 * torch.rand(A, I, generator=gen) ** 4,
                     sharp=torch.rand(A, I, generator=gen), comp=comp)
    fires = s3.Fires(pos=torch.rand(A, Fn, 2, generator=gen) * L, alive=torch.tensor([[True, False, True]] * A),
                     temp_k=torch.full((A, Fn), 1100.0), area_m2=torch.full((A, Fn), 0.5))
    state = pt.init_state(geom)
    state.plant = torch.rand(A, n, n, generator=gen) * 0.3
    state.pond = (torch.rand(A, n, n, generator=gen) < 0.1) * 50.0
    vol = pt.init_volatiles(geom)
    vol = pt.volatile_step(vol, geom, 3600.0, plant_c=state.plant,
                           carcass_kg_m2=(torch.rand(A, n, n, generator=gen) < 0.05) * 1.0)
    scene = s3.Scene(light_vis=50 + 300 * torch.rand(A, n, n, generator=gen),
                     ground_vis=pt.ground_reflectance(state, geom))
    return geom, b, scene, dict(state=state, vol=vol, items=items, fires=fires)


def test_shapes_finite_dead_rows_zero_and_order_one():
    geom, b, scene, kw = _full_case()
    x, info = obs(geom, b, scene, details=True, **kw)
    A, N = b.alive.shape
    assert x.shape == (A, N, s3.IN_DIM) and x.dtype == torch.float32
    assert bool(torch.isfinite(x).all())                    # no blanket nan_to_num: this is a real check
    assert bool((x[~b.alive] == 0).all())
    assert float(x.abs().max()) <= 1.5                     # every channel is of order 1
    bounded = [ch(f"s{s}_{c}") for s in range(8) for c in s3.SECTOR_CHANNELS] + \
        [ch(f"touch_{c}") for c in s3.TOUCH_CHANNELS] + list(range(s3.HELD0, s3.INTERO0)) + \
        [ch("intero_speed"), ch("intero_body_temp"), ch("intero_light")]
    assert float(x[..., bounded].abs().max()) <= 1.0 + 1e-6
    for key in ("theta_res", "light", "seen_bodies", "seen_items", "heard", "body_count", "item_count", "bound",
                "far_bound", "reach", "beyond", "fires_culled"):
        assert info[key].shape == (A, N), key
    # every kind of channel is exercised by this case
    xa = x[b.alive]
    for name in ("s0_cover", "s0_size", "s0_motion", "s0_flow", "s0_ground", "s0_noise", "s0_call0", "smell_plant",
                 "smell_plant_grad", "touch_plant", "held0_mass", "held1_hard", "intero_light", "intero_speed"):
        assert float(xa[:, ch(name)].abs().max()) > 0, name
    assert float(sec(x, "emitted").max()) > 0               # emitted light (fires and hot items)


def test_nan_inputs_are_not_hidden():
    """A NaN in a live body's state reaches its own row (no silent 0); dead rows stay exactly 0 whatever they hold."""
    geom, b, scene, kw = _full_case(seed=2)
    i = int(torch.nonzero(b.alive[0])[0, 0])
    bad = b.reserve_j.clone()
    bad[0, i] = float("nan")
    dead = ~b.alive
    bad[dead] = float("nan")
    x = obs(geom, replace(b, reserve_j=bad), scene, **kw)
    assert math.isnan(float(x[0, i, ch("intero_reserve")]))
    assert bool((x[dead] == 0).all())


def test_determinism_and_generator():
    geom, b, scene, kw = _full_case(seed=3)
    x1 = obs(geom, b, scene, gen=torch.Generator().manual_seed(5), **kw)
    x2 = obs(geom, b, scene, gen=torch.Generator().manual_seed(5), **kw)
    assert torch.equal(x1, x2)
    # a crowd beyond the scan budget: the stratified sample is drawn on the generator, reproducibly
    gen = torch.Generator().manual_seed(2)
    geom = flat_geom(1, 8, 64.0)
    b = make_bodies(1, 3000, 64.0, gen, alive_p=1.0)
    y1, info = obs(geom, b, uniform_scene(geom), gen=torch.Generator().manual_seed(9), details=True)
    y2 = obs(geom, b, uniform_scene(geom), gen=torch.Generator().manual_seed(9))
    assert torch.equal(y1, y2) and bool(info["bound"].all())


def test_air_is_required():
    geom = flat_geom()
    b = simple_bodies([[[100.0, 100.0]]])
    with pytest.raises(TypeError):
        s3.observe(geom, b, uniform_scene(geom))                       # no silent Earth air
    with pytest.raises(ValueError):
        s3.observe(geom, b, uniform_scene(geom), air=None)


def test_label_only_properties_do_not_change_the_input():
    """Lineage ids (and any other label carried in the body state) never reach the input."""
    geom, b, scene, kw = _full_case(seed=1)
    state = {f.name: getattr(b, f.name) for f in fields(s3.Bodies)}
    A, N = b.alive.shape
    state["lineage"] = torch.arange(A * N).reshape(A, N)
    state["genome"] = {"eye": b.eye}
    x1 = obs(geom, s3.Bodies.from_mapping(state), scene, **kw)
    state["lineage"] = torch.randperm(A * N, generator=torch.Generator().manual_seed(0)).reshape(A, N) % 3
    state["parent"] = torch.zeros(A, N)
    x2 = obs(geom, s3.Bodies.from_mapping(state), scene, **kw)
    assert torch.equal(x1, x2)
    with pytest.raises(KeyError):
        s3.Bodies.from_mapping({k: v for k, v in state.items() if k != "mass_kg"})


def test_slot_order_does_not_matter():
    """Permuting the body slots permutes the rows of x (no slot is special)."""
    geom, b, scene, kw = _full_case(seed=4, Fn=3)
    kw["items"] = replace(kw["items"], on_ground=torch.ones_like(kw["items"].alive))
    b = replace(b, held=None)
    x = obs(geom, b, scene, **kw)
    perm = torch.randperm(b.alive.shape[1], generator=torch.Generator().manual_seed(1))
    pb = s3.Bodies(**{f.name: (None if getattr(b, f.name) is None else getattr(b, f.name)[:, perm])
                      for f in fields(s3.Bodies)})
    xp = obs(geom, pb, scene, **kw)
    assert torch.allclose(xp, x[:, perm], atol=1e-5)


# ------------------------------------------------------------------------------------------------ conventions
def test_sectors_are_clockwise_from_ahead():
    h = torch.tensor(0.0)
    east, south, west, north = (torch.tensor(v) for v in ([1.0, 0.0], [0.0, -1.0], [-1.0, 0.0], [0.0, 1.0]))
    assert [int(s3.sector_of(h, v)) for v in (east, south, west, north)] == [0, 2, 4, 6]
    # heading north: north is ahead, east is to the right (clockwise)
    hn = torch.tensor(math.pi / 2)
    assert [int(s3.sector_of(hn, v)) for v in (north, east, south, west)] == [0, 2, 4, 6]
    assert torch.allclose(s3.heading_vector(hn), torch.tensor([0.0, 1.0]), atol=1e-6)
    # the ground rays run along the same sectors: a bright strip south of an east-facing body shows in sector 2
    geom = flat_geom(1, 32, 512.0)
    ground = torch.full_like(geom.elev, 0.05)
    ground[0, 16, 12:15] = 0.9                          # x in [256, 272), y in [192, 240): due south of (264, 256)
    b = simple_bodies([[[264.0, 256.0]]], heading=0.0)
    x = obs(geom, b, s3.Scene(torch.full_like(geom.elev, 400.0), ground))
    g = sec(x[0, 0], "ground")
    assert int(g.argmax()) == 2 and float(g[2]) == pytest.approx((3 * 0.05 + 2 * 0.9) / 5, rel=1e-3)
    assert torch.allclose(g[torch.arange(8) != 2], torch.full((7,), 0.05), rtol=1e-3)


# ------------------------------------------------------------------------------------------------ vision
def test_eye_optics():
    theta, G = s3.eye_optics(torch.tensor([7.6e-3, 1e-5, 1e-3, 0.0]))
    d_human = (6 * 7.6e-3 / (math.pi * s3.TISSUE_DENSITY)) ** (1 / 3)
    assert d_human == pytest.approx(0.024, rel=0.02)                     # a 7.6 g eye is a 24 mm eye
    assert float(theta[0]) == pytest.approx(math.radians(1 / 60), rel=0.05)   # one arcminute
    assert float(theta[1]) > float(theta[2]) > float(theta[0])            # bigger eyes are sharper
    assert float(G[0]) > float(G[2]) > float(G[1]) > 0 and float(G[3]) == 0
    _, Gd = s3.eye_optics(torch.tensor([1e-3]), intact=torch.tensor([0.5]))
    assert float(Gd[0]) == pytest.approx(0.5 * float(G[2]))              # damage dims the eye
    # Rose: signal over noise
    assert float(s3.rose_snr(torch.tensor(9.0), torch.tensor(0.0))) == pytest.approx(3.0)
    omega = float(s3.sphere_solid_angle(torch.tensor(1.0), torch.tensor(100.0)))
    assert omega == pytest.approx(math.pi * 1e-4, rel=1e-3)


def test_light_scale():
    assert s3.LUMEN_PER_W == pytest.approx(243.3, rel=1e-3)
    assert s3.night_light() == pytest.approx(0.002 / 243.3, rel=1e-3)
    noon = float(s3.light_level(torch.tensor(410.0)))
    assert 0.9 < noon < 1.0 and float(s3.light_level(torch.tensor(0.0))) == 0.0
    assert float(s3.light_level(torch.tensor(1e-3))) < 0.5 * noon       # full moon


def test_vision_falls_at_night_and_fire_shows():
    gen = torch.Generator().manual_seed(7)
    geom = flat_geom(1, 16, 256.0)
    b = make_bodies(1, 300, 256.0, gen, alive_p=1.0)
    fires = s3.Fires(pos=torch.tensor([[[128.0, 128.0]]]), alive=torch.tensor([[True]]),
                     temp_k=torch.tensor([[1100.0]]), area_m2=torch.tensor([[0.5]]))
    day, dinfo = obs(geom, b, uniform_scene(geom, 400.0), fires=fires, details=True)
    night, ninfo = obs(geom, b, uniform_scene(geom, s3.night_light()), fires=fires, details=True)
    vis = lambda x, name: float(sec(x, name).sum())   # noqa: E731
    for name in ("size", "cover"):
        assert vis(night, name) < 0.5 * vis(day, name), name
    # big eyes still make out the ground's large patches by starlight; small eyes lose it
    assert vis(night, "ground") < 0.8 * vis(day, "ground")
    small = replace(b, eye=torch.full_like(b.eye, 1e-4))
    sd = obs(geom, small, uniform_scene(geom, 400.0))
    sn = obs(geom, small, uniform_scene(geom, s3.night_light()))
    assert vis(sn, "ground") < 0.4 * vis(sd, "ground")
    assert vis(sn, "ground") < vis(night, "ground")
    assert float(ninfo["seen_bodies"].sum()) < 0.2 * float(dinfo["seen_bodies"].sum())
    assert float(night[..., ch("intero_light")].max()) < float(day[..., ch("intero_light")].min())
    # a 1,100 K flame is darker than the sunlit ground: nothing by day, a beacon at night
    assert vis(day, "emitted") == 0 and vis(night, "emitted") > 0
    # bigger eyes see more at night
    big = replace(b, eye=torch.full_like(b.eye, 0.3))
    _, binfo = obs(geom, big, uniform_scene(geom, s3.night_light()), details=True)
    assert float(binfo["seen_bodies"].sum()) > float(ninfo["seen_bodies"].sum())


def test_scene_from_patch_follows_the_sun_per_bout():
    geom = flat_geom(1, 16, 256.0)
    state = pt.init_state(geom)
    forcing = pt.constant_forcing(geom, sw_day_w_m2=250.0, bouts=4)
    scenes = [s3.scene_from_patch(geom, state, forcing, b) for b in range(4)]
    light = [float(s.light_vis.mean()) for s in scenes]
    assert light[2] == pytest.approx(s3.night_light(), rel=1e-3)          # 12-18 h after noon: dark
    assert light[0] > 100 and light[3] > 100                              # afternoon and morning
    gen = torch.Generator().manual_seed(0)
    b = make_bodies(1, 200, 256.0, gen, alive_p=1.0)
    x0, x2 = obs(geom, b, scenes[0], state=state), obs(geom, b, scenes[2], state=state)
    assert float(x2[..., ch("intero_light")].max()) < float(x0[..., ch("intero_light")].min())
    assert float(sec(x2, "cover").sum()) < float(sec(x0, "cover").sum())


def test_contrast_follows_the_optics_table():
    """A body on dark plant cover stands out more than on bright sand; sharper eyes see it as larger."""
    geom = flat_geom(1, 16, 256.0)
    b = simple_bodies([[[100.0, 100.0], [104.0, 100.0]]], heading=0.0, speed=1.0)
    plant = uniform_scene(geom, 400.0, optics.vis("ground:plant"))
    sand = uniform_scene(geom, 400.0, optics.vis("species:sand"))
    xp, xs = obs(geom, b, plant), obs(geom, b, sand)
    cp, cs = float(xp[0, 0, ch("s0_contrast")]), float(xs[0, 0, ch("s0_contrast")])
    skin = optics.vis("tissue:skin")
    assert cp == pytest.approx((skin - 0.05) / (skin + 0.05), rel=1e-4) and cs < 0 < cp
    sharp = replace(b, eye=torch.full_like(b.eye, 0.1))
    assert float(obs(geom, sharp, plant)[0, 0, ch("s0_size")]) > float(xp[0, 0, ch("s0_size")])
    # image motion is a value: resolution elements crossed per integration time, m / (1 + m)
    theta, _ = s3.eye_optics(torch.tensor(1e-2 * 0.1))
    m = 1.0 * 0.1 / (4.0 * float(theta))
    assert float(xp[0, 0, ch("s0_motion")]) == pytest.approx(m / (1 + m), rel=1e-4)
    assert float(xp[0, 0, ch("s0_flow")]) == pytest.approx(m / (1 + m), rel=1e-4)
    # a still body is still seen: it covers its angle, its image does not move
    still = replace(b, speed_m_s=torch.zeros_like(b.speed_m_s))
    xst = obs(geom, still, plant)
    assert float(xst[0, 0, ch("s0_motion")]) == 0 and float(xst[0, 0, ch("s0_flow")]) == 0
    assert float(xst[0, 0, ch("s0_size")]) == pytest.approx(float(xp[0, 0, ch("s0_size")]))
    assert float(xst[0, 0, ch("s0_cover")]) > 0


def test_terrain_blocks_sight_but_not_sound():
    n, L = 32, 512.0
    x = (torch.arange(n) + 0.5) * L / n
    ridge = (5.0 * torch.exp(-((x - 256.0) / 10.0) ** 2))[None, :, None].expand(1, n, n)
    geom = pt.geometry_from_elevation(ridge, L, -1e4, R_781)
    flat = flat_geom(1, n, L)
    b = simple_bodies([[[244.0, 100.0], [268.0, 100.0]]], heading=0.0, speed=1.0, loud=1.0, voice=3e-2, ear=3e-2,
                      mass=0.2)
    xr, xf = obs(geom, b, uniform_scene(geom)), obs(flat, b, uniform_scene(flat))
    assert float(xf[0, 0, ch("s0_cover")]) > 0 and float(xr[0, 0, ch("s0_cover")]) == 0
    nf, nr = float(xf[0, 0, ch("s0_noise")]), float(xr[0, 0, ch("s0_noise")])
    assert 0 < nr < nf
    # pressure halved: intensity / 4, 6 dB less, i.e. 6.02 / 60 on the sensation scale
    assert nf - nr == pytest.approx(10 * math.log10(4) / 60, abs=1e-4)


# ------------------------------------------------------------------------------------------------ hearing
def test_sound_physics():
    air = s3.Air.earth(1)
    assert float(air.density()) == pytest.approx(1.225, rel=0.01)
    assert float(air.sound_speed()) == pytest.approx(340.3, rel=0.005)
    assert s3.footfall_power(70.0, 1.4).item() == pytest.approx(4 * math.pi * 10 ** -7.5, rel=1e-6)
    p = s3.call_acoustic_power(1.0, 3e-3, 200.0, 30.0)
    p1m = math.sqrt(float(p) * 1.204 * 343.2 / (4 * math.pi))
    assert 20 * math.log10(p1m / 20e-6) == pytest.approx(100.0, abs=1.5)    # version 2's loud call
    one = torch.tensor(True)
    i10, i20 = (s3.received_intensity(torch.tensor(1.0), torch.tensor(r), torch.tensor(0.0), one) for r in (10.0, 20.0))
    assert float(i10 / i20) == pytest.approx(4.0)
    blocked = s3.received_intensity(torch.tensor(1.0), torch.tensor(10.0), torch.tensor(0.0), torch.tensor(False))
    assert float(blocked / i10) == pytest.approx(0.25)
    th = s3.ear_threshold(torch.tensor([0.02, 0.02 / 1000, 0.0]))
    assert float(th[0]) == pytest.approx(1e-12) and float(th[1]) == pytest.approx(1e-10, rel=1e-4)
    assert math.isinf(float(th[2]))


def test_no_sound_below_one_kilopascal():
    gen = torch.Generator().manual_seed(3)
    geom = flat_geom(1, 16, 256.0)
    b = make_bodies(1, 300, 256.0, gen, alive_p=1.0)
    hear = [ch(f"s{s}_{c}") for s in range(8) for c in s3.HEARING_CHANNELS]
    thick = obs(geom, b, uniform_scene(geom), air=s3.Air.earth(1))
    thin = obs(geom, b, uniform_scene(geom), air=s3.Air.earth(1, pressure_pa=500.0))
    assert float(thick[..., hear].abs().sum()) > 0
    assert float(thin[..., hear].abs().sum()) == 0
    # sight does not need air
    sight = [ch("s0_cover"), ch("s0_ground"), ch("s0_size")]
    assert torch.equal(thin[..., sight], thick[..., sight])


def test_calls_carry_content_and_small_voices_fade():
    geom = flat_geom(1, 16, 256.0)
    calls = torch.zeros(1, 2, 8)
    calls[0, 1] = torch.linspace(-1, 1, 8)
    b = simple_bodies([[[100.0, 100.0], [103.0, 100.0]]], heading=0.0, calls=calls, voice=3e-2, ear=3e-2, mass=0.5)
    x = obs(geom, b, uniform_scene(geom))
    got = x[0, 0, ch("s0_call0"):ch("s0_call0") + 8]
    w = float(got[-1])
    assert 0 < w <= 1 and torch.allclose(got, calls[0, 1] * w, atol=1e-6)
    # the same caller with a tiny voice organ calls weakly in ultrasound
    tiny = replace(b, voice=torch.full_like(b.voice, 1e-6))
    assert float(obs(geom, tiny, uniform_scene(geom))[0, 0, ch("s0_call7")]) < w
    # silence: no loudness, no call; thin ears hear less
    quiet = replace(b, loudness=torch.zeros_like(b.loudness))
    assert float(obs(geom, quiet, uniform_scene(geom))[0, 0, ch("s0_call7")]) == 0
    deaf = replace(b, ear=torch.full_like(b.ear, 1e-7))
    assert float(obs(geom, deaf, uniform_scene(geom))[0, 0, ch("s0_call7")]) < w


def test_air_from_partial_pressures():
    pp = torch.tensor([[78084.0, 20948.0, 28.0, 934.0], [0.0, 0.0, 9.0e6, 0.0]])
    air = s3.Air.from_partial(pp, torch.tensor([288.0, 737.0]), torch.tensor([9.81, 8.87]), torch.tensor([0.0, 0.0]))
    assert float(air.molar_kg[0]) == pytest.approx(0.02896, rel=1e-3)
    assert float(air.gamma[0]) == pytest.approx(1.40, abs=0.005)
    assert float(air.gamma[1]) == pytest.approx(K.CP_MOLAR_298["CO2"] / (K.CP_MOLAR_298["CO2"] - K.R_GAS), rel=1e-5)
    assert float(air.sound_speed()[1]) > 300 and float(air.density()[1]) > 50
    assert float(air.x_o2[0]) == pytest.approx(0.2095, rel=1e-3) and float(air.x_o2[1]) == 0
    # water vapour joins the mixture
    wet = s3.Air.from_partial(pp[:1], 293.15, 9.81, torch.tensor([1640.0]))
    assert float(wet.x_h2o[0]) == pytest.approx(1640.0 / (float(pp[0].sum()) + 1640.0), rel=1e-5)
    assert float(wet.molar_kg[0]) < float(air.molar_kg[0])
    with pytest.raises(TypeError):
        s3.Air.from_partial(pp, 288.0, 9.81)                                    # humidity is not assumed


# ------------------------------------------------------------------------------------------------ smell
def test_smell_concentration_and_gradient_along_the_heading():
    geom = flat_geom(1, 16, 256.0)
    vol = pt.init_volatiles(geom)
    xs = (torch.arange(16) + 0.5) * 16.0
    th = s3.smell_thresholds()
    vol.plant = (th["plant"] * (1 + xs / 16.0))[None, :, None].expand(1, 16, 16).clone()   # rises to the east
    b = simple_bodies([[[100.0, 100.0], [100.0, 100.0], [100.0, 100.0], [180.0, 100.0]]],
                      heading=torch.tensor([[0.0, math.pi, math.pi / 2, 0.0]]))
    x = obs(geom, b, uniform_scene(geom), vol=vol)
    c, g = x[0, :, ch("smell_plant")], x[0, :, ch("smell_plant_grad")]
    assert float(g[0]) > 0 > float(g[1]) and abs(float(g[2])) < 1e-6
    assert float(g[0]) == pytest.approx(-float(g[1]), rel=1e-4)
    assert float(c[3]) > float(c[0]) > 0
    conc = float(pt.bilinear(geom, vol.plant, torch.tensor([[[100.0, 100.0]]]))[0, 0])
    assert float(c[0]) == pytest.approx(math.log10(1 + conc / th["plant"]) / 3.0, rel=1e-4)
    assert abs(float(x[0, 0, ch("smell_smoke")])) == 0
    # damage blunts the nose
    hurt = replace(b, damage=torch.full_like(b.damage, 0.9))
    assert float(obs(geom, hurt, uniform_scene(geom), vol=vol)[0, 0, ch("smell_plant")]) < float(c[0])


# ------------------------------------------------------------------------------------------------ touch and held
def test_touch_under_the_mouth_and_held_items():
    geom = flat_geom(1, 16, 256.0)
    state = pt.init_state(geom)
    state.plant[0, 6, 6] = 0.5                                             # the cell of (100, 100)
    state.pond[0, 9, 6] = 80.0                                             # the cell of (150, 100)
    comp = torch.zeros(1, 3, materials.S)
    comp[0, 0, materials.IDX["flint"]] = 1.0
    comp[0, 1, materials.IDX["wood"]] = 1.0
    comp[0, 2, materials.IDX["basalt"]] = 1.0
    M = 1.0
    rad = float(s3.body_radius(torch.tensor(M)))
    items = s3.Items(pos=torch.tensor([[[100.0 + rad * 1.2, 100.0], [0.0, 0.0], [0.0, 0.0]]]),
                     alive=torch.tensor([[True, True, True]]), mass=torch.tensor([[0.2, 1.0, 3.0]]),
                     temp_k=torch.tensor([[600.0, 300.0, 305.0]]), sharp=torch.tensor([[0.0, 0.0, 0.7]]), comp=comp)
    held = torch.tensor([[[1, 2], [-1, -1], [-1, -1]]])
    b = simple_bodies([[[100.0, 100.0], [150.0, 100.0], [100.0 - 2 * rad, 100.0]]], heading=0.0, mass=M, held=held)
    x = obs(geom, b, uniform_scene(geom), state=state, items=items)
    t = x[0]
    assert float(t[0, ch("touch_plant")]) == pytest.approx(float(pt.fapar(torch.tensor(0.5))), rel=1e-5)
    assert float(t[0, ch("touch_wet")]) == 0 and float(t[1, ch("touch_wet")]) > 0
    depth = 0.08
    assert float(t[1, ch("touch_wet")]) == pytest.approx(depth / rad / (1 + depth / rad), rel=1e-4)
    assert float(t[1, ch("touch_damp")]) == 1.0
    # the flint lies in front of body 0's mouth: light, hard and hot
    flint_h = float(materials.props(comp[:, :1], torch.tensor([[0.2]]))["tool_hardness"][0, 0])
    assert float(t[0, ch("touch_hard")]) == pytest.approx(flint_h / 10, rel=1e-5)
    assert float(t[0, ch("touch_temp")]) == pytest.approx(math.tanh((600 - 305) / 10), rel=1e-5)
    assert float(t[0, ch("touch_mass")]) == pytest.approx(0.2 / 1.2, rel=1e-5)
    assert float(t[1, ch("touch_hard")]) == 0 and float(t[1, ch("touch_mass")]) == 0
    # body 2's mouth touches body 0: equal mass (1/2), soft as hide, as warm as itself
    assert float(t[2, ch("touch_mass")]) == pytest.approx(0.5)
    assert float(t[2, ch("touch_hard")]) == pytest.approx(s3.BODY_HARDNESS / 10, rel=1e-6)
    assert float(t[2, ch("touch_temp")]) == pytest.approx(0.0, abs=1e-6)
    # held: wood in the left grip, basalt in the right
    assert float(t[0, ch("held0_mass")]) == pytest.approx(0.5)
    assert float(t[0, ch("held1_mass")]) == pytest.approx(0.75)
    assert float(t[0, ch("held1_sharp")]) == pytest.approx(0.7)
    assert float(t[0, ch("held1_temp")]) == pytest.approx(0.0, abs=1e-6)
    assert float(t[0, ch("held0_temp")]) == pytest.approx(math.tanh(-0.5), rel=1e-5)
    assert float(t[0, ch("held1_hard")]) > float(t[0, ch("held0_hard")]) > 0
    assert float(t[1, ch("held0_mass")]) == 0
    # held items are not seen lying on the ground
    _, info = obs(geom, b, uniform_scene(geom), state=state, items=items, details=True)
    assert int(info["item_count"][0, 2]) == 1                              # only the flint is on the ground


def test_interoception():
    geom = flat_geom(1, 16, 256.0)
    M = torch.tensor([[0.1, 10.0]])
    b = simple_bodies([[[50.0, 50.0], [200.0, 200.0]]], mass=M, speed=torch.tensor([[0.5, 2.0]]),
                      body_k=torch.tensor([[271.0, 318.0]]), damage=torch.tensor([[0.0, 0.25]]),
                      reserve_j=torch.tensor([[2.5e3, 1e4]]), gut_kg=torch.tensor([[[0.001, 0.002], [0.0, 0.5]]]))
    x = obs(geom, b, uniform_scene(geom))
    i = x[0, :, s3.INTERO_SLICE]
    names = brain3.INTERO_NAMES
    assert i[:, names.index("reserve")].tolist() == pytest.approx([0.25, 1.0])
    assert i[:, names.index("water")].tolist() == pytest.approx([1.0, 1.0])
    assert i[:, names.index("body_temp")].tolist() == pytest.approx([-1.0, 1.0])      # signed within life's range
    assert i[:, names.index("damage")].tolist() == pytest.approx([0.0, 0.25])
    assert i[:, names.index("gut")].tolist() == pytest.approx([0.003 / 0.005, 0.5 / 0.5])
    r = s3.body_radius(M)[0]
    fr = [0.5 / math.sqrt(K.G_STANDARD * 2 * float(r[0])), 2.0 / math.sqrt(K.G_STANDARD * 2 * float(r[1]))]
    assert i[:, names.index("speed")].tolist() == pytest.approx([f / (1 + f) for f in fr], rel=1e-5)
    assert float(i[0, names.index("light")]) == pytest.approx(float(s3.light_level(torch.tensor(400.0))), rel=1e-5)
    mid = obs(geom, replace(b, body_k=torch.full_like(b.body_k, s3.T_LIFE_MID_K)), uniform_scene(geom))
    assert float(mid[0, 0, ch("intero_body_temp")]) == pytest.approx(0.0, abs=1e-6)


def test_interoception_is_order_one_for_fast_small_bodies():
    """brain3's contract: interoceptive inputs of order 1. Small bodies running fast have Froude speeds of 5 and more;
    the channel saturates as Fr / (1 + Fr)."""
    geom = flat_geom(1, 16, 256.0)
    M = torch.tensor([[0.002, 0.02, 0.2]])
    b = simple_bodies([[[30.0, 30.0], [120.0, 30.0], [30.0, 200.0]]], mass=M, speed=torch.tensor([[2.0, 3.0, 4.0]]))
    sp = obs(geom, b, uniform_scene(geom))[0, :, ch("intero_speed")]
    fr = torch.tensor([2.0, 3.0, 4.0]) / torch.sqrt(K.G_STANDARD * 2 * s3.body_radius(M)[0])
    assert float(fr.min()) > 4                                   # the raw Froude speeds are far above 1
    assert torch.allclose(sp, fr / (1 + fr), rtol=1e-5) and float(sp.max()) < 1


# ------------------------------------------------------------------------------------------------ fires and glow
def test_fire_seen_at_night_unless_terrain_hides_it():
    n, L = 32, 1024.0
    x = (torch.arange(n) + 0.5) * L / n
    ridge = (8.0 * torch.exp(-((x - 512.0) / 20.0) ** 2))[None, :, None].expand(1, n, n)
    geom = pt.geometry_from_elevation(ridge, L, -1e4, R_781)
    fires = s3.Fires(pos=torch.tensor([[[700.0, 300.0]]]), alive=torch.tensor([[True]]),
                     temp_k=torch.tensor([[1100.0]]), area_m2=torch.tensor([[0.5]]))
    # body 0 on the fire's side (100 m west of it), body 1 behind the ridge (400 m west), body 2 faces away
    b = simple_bodies([[[500.0 + 100.0, 300.0], [300.0, 300.0], [600.0, 300.0]]],
                      heading=torch.tensor([[0.0, 0.0, math.pi]]))
    dark = uniform_scene(geom, s3.night_light())
    out = obs(geom, b, dark, fires=fires)
    em = sec(out[0], "emitted")
    assert float(em[0, 0]) > 0.5 and float(em[1].sum()) == 0
    assert float(em[2, 4]) == pytest.approx(float(em[0, 0]), rel=1e-5)     # behind it: sector 4
    # it lights the night where its line is clear; behind the ridge the night stays as dark as without it
    assert float(out[0, 0, ch("intero_light")]) > float(out[0, 1, ch("intero_light")])
    none = obs(geom, b, dark)
    assert float(out[0, 1, ch("intero_light")]) == pytest.approx(float(none[0, 1, ch("intero_light")]), rel=1e-6)
    assert float(out[0, 0, ch("intero_light")]) > float(none[0, 0, ch("intero_light")])


def test_hot_items_glow():
    geom = flat_geom(1, 16, 256.0)
    comp = torch.zeros(1, 2, materials.S)
    comp[..., materials.IDX["charcoal"]] = 1.0
    b = simple_bodies([[[100.0, 100.0]]], heading=0.0)
    cold = s3.Items(pos=torch.tensor([[[110.0, 100.0], [100.0, 115.0]]]), alive=torch.tensor([[True, True]]),
                    mass=torch.tensor([[0.1, 0.1]]), temp_k=torch.tensor([[290.0, 290.0]]),
                    sharp=torch.zeros(1, 2), comp=comp)
    hot = replace(cold, temp_k=torch.tensor([[1200.0, 290.0]]))
    dark = uniform_scene(geom, s3.night_light())
    xc, xh = obs(geom, b, dark, items=cold), obs(geom, b, dark, items=hot)
    assert float(xc[0, 0, ch("s0_emitted")]) == 0 and float(xh[0, 0, ch("s0_emitted")]) > 0.5
    assert float(xh[0, 0, ch("s0_contrast")]) > 0.9 and float(xh[0, 0, ch("s0_size")]) > 0
    day = uniform_scene(geom, 400.0, 0.3)
    xd = obs(geom, b, day, items=cold)
    assert float(xd[0, 0, ch("s0_contrast")]) < 0 < float(xd[0, 0, ch("s0_size")])   # dark charcoal
    assert float(xd[0, 0, ch("s6_size")]) > 0                            # the one to the north, left of ahead


# ------------------------------------------------------------------------------------------------ adapters
def test_pools_of_manipulate_and_the_mouth_override():
    """Version 2's ItemPool and FirePool (manipulate.py's patch pools) feed the senses directly; held items are not
    on the ground; body.py's own mouth point and reach replace the sphere's."""
    from haishool.life9.planet import crafting
    from haishool.life9.planet import items as it
    geom = flat_geom(1, 16, 256.0)
    pool = it.new_item_pool(1, 3)
    pool.alive[:] = True
    pool.comp[0, :, materials.IDX["flint"]] = 1.0
    pool.mass[:] = 0.3
    pool.temp_k[:] = 300.0
    pool.pos[0, :, :2] = torch.tensor([[105.0, 100.0], [100.0, 108.0], [90.0, 100.0]])
    pool.holder[0, 2] = 1                                                  # body 0's right grip
    items = s3.Items.from_pool(pool)
    assert items.on_ground.tolist() == [[True, True, False]] and items.pos.shape == (1, 3, 2)
    fp = it.new_fire_pool(1, 2)
    fp.alive[0, 0] = True
    fp.pos[0, 0, :2] = torch.tensor([150.0, 100.0])
    fp.temp_k[0, 0] = 1100.0
    fires = s3.Fires.from_pool(fp)
    assert float(fires.area_m2[0, 0]) == pytest.approx(math.pi * float(crafting.FIRE_RADIUS_M) ** 2)
    b = simple_bodies([[[100.0, 100.0]]], heading=0.0, mass=0.5, held=torch.tensor([[[-1, 2]]]))
    x, info = obs(geom, b, uniform_scene(geom, s3.night_light()), items=items, fires=fires, details=True)
    assert int(info["item_count"][0, 0]) == 2 and float(x[0, 0, ch("s0_emitted")]) > 0.5
    assert float(x[0, 0, ch("held1_mass")]) == pytest.approx(0.3 / 0.8, rel=1e-5)
    assert float(x[0, 0, ch("touch_hard")]) == 0                          # the flint 5 m ahead is out of reach
    # a long reach to the north (body.py's mouth point and reach) touches the flint at (100, 108)
    far = replace(b, mouth_pos=torch.tensor([[[100.0, 107.0]]]), reach_m=torch.tensor([[1.5]]))
    xf = obs(geom, far, uniform_scene(geom), items=items)
    assert float(xf[0, 0, ch("touch_hard")]) > 0


def test_bodies_at_the_same_point_give_finite_inputs():
    """Review M13: two bodies at exactly the same position (d = 0) made the motion and flow channels 0 / 0 = NaN for
    every sector; the angular speed of a target closer than its own radius is taken at its radius."""
    geom = flat_geom()
    pos = torch.tensor([[[100.0, 100.0], [100.0, 100.0], [100.0, 100.0], [120.0, 100.0]]])
    for speed in (0.0, 0.3):
        b = simple_bodies(pos, speed=speed)
        x = obs(geom, b, uniform_scene(geom))
        assert bool(torch.isfinite(x).all()), speed
        if speed > 0:
            assert float(sec(x, "motion")[0, 0].max()) > 0                       # the co-located body still moves
