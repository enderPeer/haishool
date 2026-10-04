"""The v3 world (haishool/life9/v3/world3.py): construction on 781, Earth and synthetic planets, a world without land,
ten days with every ledger closing, determinism and exact resume, the viewer's data contract, the describe and run
CLI, the default capacity (4,096 slots per arena) and its reported memory, and the locomotion cost the world adds
(Kram & Taylor's force-generation cost). The experiment protocol (sampling, manifest, rules hash, outcomes, no rescue)
is tested in test_protocol.py."""
from __future__ import annotations

import contextlib
import io
import json
import math

import pytest
import torch

from haishool.life9.v3 import __main__ as cli
from haishool.life9.v3 import body as B
from haishool.life9.v3 import world3 as w3

# ledgers that close to float64 / float32 rounding of their own books
TIGHT = 1e-6
# the body <-> patch cross-books: body.py's float32 carcass and transit rounding on tiny (mg) flows reaches ~1e-6
EXCHANGE = 1e-5


def small(**kw) -> w3.V3Config:
    cfg = dict(G=8, patches=2, patch_m=256.0, cells=16, capacity=96, founders=32, items=128, fires=8, hidden=16,
               k_founder=(2, 8), bouts=4, climate_fast_chunks=1, climate_slow_chunks=1, bio_spin_days=20,
               bio_spin_segments=1, veg_spin_days=20, veg_spin_bouts=2)
    cfg.update(kw)
    return w3.V3Config(**cfg)


_CACHE: dict = {}


def earth_world(**kw) -> w3.World3:
    """A small Earth world (bodies live there), built once per config and copied through state_dict."""
    key = tuple(sorted(kw.items()))
    if key not in _CACHE:
        _CACHE[key] = w3.World3(small(source="earth", **kw), [0])
    return w3.World3.from_state(_CACHE[key].state_dict())


def check_ledgers(world: w3.World3) -> dict:
    led = world.ledgers()
    bad = {}
    for name, v in led.items():
        if not isinstance(v, dict):
            continue
        tol = EXCHANGE if name.startswith("exchange_") else TIGHT
        if not v["rel"] < tol:
            bad[name] = v["rel"]
    assert not bad, bad
    return led


# ------------------------------------------------------------------------------------------------ construction
def test_every_world_constant_has_a_provenance_entry():
    tags = {"reference", "derived", "new_rule", "chain"}
    for key, (tag, note) in w3.PROVENANCE.items():
        assert note and set(tag.split("+")) <= tags, key
    consts = [n for n in vars(w3) if n.isupper() and not n.startswith("_") and n not in ("PROVENANCE", "R_", "D_", "N_", "C_", "RN", "DN",
                                                                  "K", "B", "VIEW_SCHEMA", "STATE_VERSION", "DAY_S",
                                                                  "FOUR_PI", "M_C", "M_N", "M_O2", "M_CO2",
                                                                  "GENE_REPORT", "BEHAVIOUR_KEYS", "DAY_KEYS")]
    missing = [c for c in consts if c not in w3.PROVENANCE]
    assert not missing, missing
    for key in ("bouts", "substeps", "climate_target", "carries", "capacity", "patches", "locomotion", "barometric",
                "fire_heat", "hand_heat", "development", "brain_units", "behaviour_baseline", "capacity_bound",
                "protocol", "absent", "outcomes", "extinction", "rules_hash", "exchange_scale"):
        assert key in w3.PROVENANCE, key
    cfg = w3.V3Config()
    # the owner's decision of 4 Oct 2026: 2 km patches with 4,096 slots each, 16 patches (PROVENANCE['capacity'])
    assert (cfg.G, cfg.patches, cfg.patch_m, cfg.cells, cfg.capacity, cfg.founders, cfg.items, cfg.fires,
            cfg.hidden) == (32, 16, 2048.0, 128, 4096, 512, 1024, 64, 128)
    assert cfg.bouts == 24 and cfg.substep_len_m == 32.0 and cfg.selection and cfg.weight_dtype == "float32"
    # no rescue switch is left in the config (PROVENANCE['protocol'])
    assert not any("fallback" in name for name in cfg.to_dict())
    with pytest.raises(ValueError):
        w3.V3Config.from_dict({**cfg.to_dict(), "synthetic_fallback": True})


def test_state_memory_is_reported_and_scales_with_the_slots():
    """PROVENANCE['capacity']: the world reports its state memory; the bodies' share grows with the slots and halves
    with bfloat16 weights (the weight_dtype option is kept)."""
    sizes = {}
    for cap, dt in ((96, "float32"), (192, "float32"), (192, "bfloat16")):
        w = earth_world(capacity=cap, weight_dtype=dt)
        assert w.reports["state_mb"] > 0
        sizes[(cap, dt)] = w3._tensor_bytes(w.b.state_dict())
    per_slot = (sizes[(192, "float32")] - sizes[(96, "float32")]) / (w.A * 96)
    assert per_slot > 0
    assert sizes[(192, "bfloat16")] < 0.75 * sizes[(192, "float32")]


def test_construction_781_world7_chain():
    w = w3.World3(small(), [781])
    s = w.specs[0]
    assert w.input_notes == ["world7"] and s.seed == 781 and not w.spec_failures
    assert w.A == 2 and w.no_land == [False]
    # 781 is frozen_mean: formation's T_s is no target for the climate (no calibration of its clouds)
    assert s.frozen_mean and w.reports["climate_spin_up"]["calibrated"] == [False]
    # patches on land cells, at the planet's real radius, with fine detail of metres
    land = w.terrain0["land"][0]
    assert all(bool(land[int(c)]) for c in w.arena_cell)
    assert torch.allclose(w.geom.radius_m, torch.full((2,), s.radius_m, dtype=torch.float64))
    assert float(w.geom.detail_rms_m.min()) > 0.0
    # founders: the configured number per arena, nothing else alive, at the local air temperature
    assert w.b.alive.sum(1).tolist() == [32, 32]
    t = w3.pt.sample(w.geom, w3.pt.fine_temperature(w.geom, w._arena_forcing(w._last_cdiag(), int(w.clim.day), 1)),
                     w.b.pos)
    assert float((w.b.body_k - t).abs()[w.b.alive].max()) < 1e-3
    assert w.items.alive.sum() == 0 and w.fires.alive.sum() == 0
    check_ledgers(w)


def test_construction_earth_and_synthetic_with_a_world_without_land():
    e = earth_world()
    assert not e.specs[0].frozen_mean and e.reports["climate_spin_up"]["calibrated"] == [True]
    assert e.A == 2 and e.reports["vegetation_spin_up"]["days"] == 20
    # synthetic 6 has land, synthetic 1 is water-covered: its patches are sea (patch.choose_patch_cells' own rule, no
    # placement by habitability; review of 4 Oct 2026) and physics decides what its founders do there
    w = w3.World3(small(source="synthetic"), [6, 1])
    assert w.no_land == [False, True] and w.reports["no_land"] == [1]
    assert w.A == 4 and w.arena_world.tolist() == [0, 0, 1, 1]
    assert all(bool(w.geom.sea[a].all()) for a in (2, 3)) and not bool(w.geom.sea[0].all())
    assert w.b.alive.sum(1).tolist() == [32, 32, 32, 32]                     # founders on the sea too
    w.step_day()
    s = w.summary()
    assert s["worlds"][1]["no_land"] and s["worlds"][1]["arenas"] == [2, 3]
    rows = w.outcomes()
    assert [r["seed"] for r in s["outcomes"]] == [6, 1] and [r["no_land"] for r in rows] == [False, True]
    assert rows[1]["kind"] in ("alive", "extinct") and "sea" in rows[1]["note"]
    assert len(w.static3(1)["patches"]) == 2
    check_ledgers(w)


def test_a_planet_without_land_gets_sea_patches_and_physics_decides():
    """Review of 4 Oct 2026 (rule 4): a world without land is no longer a non-run 'no land'; its patches are drawn
    over all its cells (sea) and its founders float or sink there."""
    w = w3.World3(small(source="synthetic"), [1])
    assert w.A == 2 and w.no_land == [True] and bool(w.geom.sea.all())
    w.run(2)
    s = w.summary()
    assert s["day"] == 2 and s["worlds"][0]["no_land"]
    assert w.outcomes()[0]["kind"] in ("alive", "extinct")
    check_ledgers(w)
    assert "sea" in w.describe()


# ------------------------------------------------------------------------------------------------ the days
def test_ten_days_with_every_ledger_closing():
    w = earth_world()
    w.run(10)
    led = check_ledgers(w)
    s = w.summary()
    total = s["behaviour_total"]
    # the loop ran: bodies were born and died by physics, mouths touched the world, bodies moved
    assert sum(s["births"]) > 0 and sum(sum(v) for v in s["deaths"].values()) > 0
    assert sum(total["mouth_events"]) > 0 and sum(total["distance_m"]) > 0
    # the global layer received the arenas' flows: gas booked (bodies' CO2, patch fixation) and water to the air
    assert sum(led["air_booked"]["scale"]) > 0 and sum(led["water_booked"]["scale"]) > 0
    # local carbon and nitrogen of each arena (patch + bodies + items + transits) against the boundary flows
    assert led["local_carbon"]["rel"] < TIGHT and led["local_nitrogen"]["rel"] < TIGHT
    # gene statistics for the living and the neutral shadow side by side
    for name in ("size_kg", "fur_m", "thermo_gain", "enz_plant", "enz_meat", "eye", "ear", "voice", "k", "eta", "g",
                 "mut"):
        assert name in s["genes"] and name in s["shadow_genes"], name
    assert len(s["genes"]["g"]["mean"][0]) == len(w3.b3.INTERO_NAMES)
    for key in ("population", "lineages_alive", "capacity_full", "mouth_shares_total", "behaviour_today"):
        assert key in s, key
    assert all(lin <= pop for lin, pop in zip(s["lineages_alive"], s["population"]))
    json.dumps(s)


def test_determinism_and_exact_resume(tmp_path):
    a, b = earth_world(), earth_world()
    a.run(2)
    b.run(2)
    assert a.state_hash() == b.state_hash()
    c = earth_world()
    c.run(1)
    snap = c.state_dict()
    c.save(tmp_path / "ck.pt")
    c.run(1)
    assert c.state_hash() == a.state_hash()
    d = w3.World3.from_state(snap)
    d.run(1)
    assert d.state_hash() == a.state_hash()
    e = w3.World3.load(tmp_path / "ck.pt")
    e.run(1)
    assert e.state_hash() == a.state_hash()
    check_ledgers(e)
    assert e.summary() == a.summary()


def test_no_selection_control_and_bf16_weights_run():
    w = earth_world(selection=False, weight_dtype="bfloat16")
    assert w.b.genome["Wh"].dtype == torch.bfloat16 and w.b.wh_live.dtype == torch.bfloat16
    w.run(2)
    check_ledgers(w)


# ------------------------------------------------------------------------------------------------ viewer contract
def test_static_and_frame_format():
    w = earth_world()
    w.run(1)
    st = w.static3(0)
    assert st["schema"] == w3.VIEW_SCHEMA and st["seed"] == 0
    for key in ("radius_m", "gravity_m_s2", "surface_pressure_pa", "tidally_locked", "year_s", "star_teff_k",
                "t_surface_k", "ocean_fraction", "frozen_mean"):
        assert key in st["planet"], key
    C = w.globe.C
    g = st["globe"]
    assert g["G"] == 8 and len(g["centers"]) == C and len(g["elevation_m"]) == C and len(g["land"]) == C
    assert all(len(c) == 3 for c in g["centers"]) and set(g["land"]) <= {0, 1}
    n = w.config.cells
    assert len(st["patches"]) == 2
    for p in st["patches"]:
        assert len(p["elevation_m"]) == n * n and len(p["pond"]) == n * n and p["n"] == n and p["L"] == 256.0
        assert len(p["center"]) == 3 and isinstance(p["cell"], int)
    assert st["species"] and st["item_classes"]
    fr = w.frame3(0, fields=True, global_fields=True)
    assert set(fr) >= {"day", "bout", "sun", "patches", "global"}
    assert abs(math.hypot(*fr["sun"]) - 1) < 1e-3
    assert len(fr["global"]["T_k"]) == C and set(fr["global"]["ice"]) <= {0, 1}
    for p in fr["patches"]:
        assert all(len(r) == 12 for r in p["bodies"])
        assert len(p["bodies"]) == int(w.b.alive[p["arena"]].sum())
        assert len(p["plants"]) == n * n and len(p["water"]) == n * n
        assert max(p["plants"]) <= 255 and min(p["water"]) >= 0
        assert len(p["items"]) <= 1500 and all(len(r) == 4 for r in p["items"])
        assert all(0 <= r[2] < len(st["item_classes"]) for r in p["items"])
        assert all(len(r) == 3 for r in p["fires"])
    plain = w.frame3(0)
    assert "global" not in plain and "plants" not in plain["patches"][0]
    json.dumps(st)
    json.dumps(fr)


# ------------------------------------------------------------------------------------------------ the CLI
def test_describe_cli():
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert cli.main(["describe", "--seed", "781", "--G", "8", "--patches", "3"]) == 0
    text = out.getvalue()
    assert "formation" in text.lower() and "check_spec3" in text
    assert "patches of seed 781" in text and text.count(": cell ") == 3
    # the same cells World3 draws (the set-up's draws 1-2)
    got = w3.choose_patches(small(patches=3), [781])
    for c in got["cells"][0]:
        assert f"cell {c}," in text


def cli_sets(cfg: w3.V3Config) -> list:
    """--set arguments for every field of ``cfg`` but the source (the sample or --source gives it)."""
    out = []
    for k, v in cfg.to_dict().items():
        if k in ("source",):
            continue
        out += ["--set", f"{k}={','.join(str(x) for x in v) if isinstance(v, list) else v}"]
    return out


def test_run_cli_writes_summary_frames_and_resumes(tmp_path):
    args = ["run", "--sample", "earth:1", "--days", "2", "--out", str(tmp_path), "--view-every", "2"] + cli_sets(small())
    with contextlib.redirect_stdout(io.StringIO()):
        assert cli.main(args + ["--chunk-days", "1"]) == 0            # stops after one day, saved
        assert not (tmp_path / "outcomes.json").exists()
        assert cli.main(["run", "--resume", "--out", str(tmp_path), "--view-every", "2"]) == 0
    lines = (tmp_path / "summary.jsonl").read_text().splitlines()
    assert [json.loads(x)["day"] for x in lines] == [1, 2]
    frames = (tmp_path / "frames-0.jsonl").read_text().splitlines()
    assert len(frames) == 4 and json.loads(frames[0])["patches"]
    assert json.loads((tmp_path / "static-0.json").read_text())["schema"] == w3.VIEW_SCHEMA
    done = json.loads((tmp_path / "outcomes.json").read_text())
    assert done["complete"] and done["days_run"] == 2 and [r["seed"] for r in done["outcomes"]] == [0]
    # the CLI run is the library run: the same sample and config give the same state
    ref = w3.World3(small(source="earth"), [0])
    ref.run(2)
    assert w3.World3.load(tmp_path / "checkpoint.pt").state_hash() == ref.state_hash()


# ------------------------------------------------------------------------------------------------ locomotion
def _bodies(mass_kg, n=1, fat=0.0, air=0.01):
    g = torch.Generator().manual_seed(0)
    b = B.found(1, n, n, 2048.0, g, in_dim=4, hidden=8, t_k=310.0, brain=False)
    for name, v in (("size_kg", mass_kg), ("muscle", 100.0), ("muscle_frac", 0.4), ("fur_m", 1e-4),
                    ("air_l_kg", air), ("o2_carrier", 0.0)):
        if name in b.genome:                                           # a small lung: the lean body sinks
            b.genome[name][:] = v
    b.mass_kg[:] = mass_kg
    b.frame_kg[:] = mass_kg
    b.reserve_j[:] = fat * mass_kg * B.FAT_J_KG
    b.water_kg[:] = B.LEAN_WATER * mass_kg
    b.pos[:] = 100.0
    b.heading[:] = 0.0
    B.reset_ledgers(b)
    return b


def test_force_generation_cost_is_c_m_g_per_ground_contact():
    """Kram & Taylor's physics (PROVENANCE['KT_COST_J_PER_N']): every ground contact costs c M g, a contact carries the
    body its own length, so the cost per metre is c M g / L at any speed where air drag is small; a held load adds
    its weight. No allometric target is gated (cost_of_transport only reports the comparison with Taylor 1982)."""
    env = B.constant_env(t_air_k=300.0, dt_s=600.0)               # air to breathe; its drag is booked apart
    eff = B.MUSCLE_EFF
    for m in (0.002, 0.2, 70.0):
        length = float(B.shape(_bodies(m))["length"])
        per_contact = w3.KT_COST_J_PER_N * m * K_G
        for share in (0.05, 0.4):
            b = _bodies(m, air=0.2)
            r = w3.move_kt(b, 0.0, share, env)
            contacts = float(r["dist_m"]) / length
            force = float(r["metabolic_j"]) - float(r["drag_j"]) / eff          # less the air drag's metabolic cost
            assert contacts > 0 and force / contacts == pytest.approx(per_contact, rel=1e-3), (m, share)
        assert w3.cost_of_transport(m) == pytest.approx(w3.KT_COST_J_PER_N * K_G / length, rel=1e-6)
        # a held load of half the body's mass costs half again per metre
        b = _bodies(m, air=0.2)
        t = w3.locomotion_terms(b, 0.4, env, carry_kg=torch.full((1, 1), 0.5 * m))
        r = w3.move_kt(b, 0.0, 0.4, env, terms=t)
        force = float(r["metabolic_j"]) - float(r["drag_j"]) / eff
        assert force / (float(r["dist_m"]) / length) == pytest.approx(1.5 * per_contact, rel=1e-3)


K_G = 9.80665


def test_locomotion_takes_its_time_share_at_full_power_and_stands_the_rest():
    """Review M4: the locomotion share is a share of the bout's time (manipulate.time_split); the body moves at its full
    sustained power for that time. On land (linear cost) the distance equals spending share x P over the whole bout;
    in water (cubic drag) it is less, as it physically is."""
    env = B.constant_env(t_air_k=300.0, dt_s=3600.0)
    b = _bodies(0.02)
    t = w3.locomotion_terms(b, 0.3, env)
    p_full = float(t["p_met"])
    r = w3.move_kt(b, 0.0, 0.3, env, terms=t)
    assert float(r["moving_s"]) == pytest.approx(0.3 * 3600.0, rel=1e-6)
    assert float(r["metabolic_j"]) == pytest.approx(p_full * 0.3 * 3600.0, rel=1e-6)
    assert float(r["dist_m"]) == pytest.approx(float(t["v_land"]) * 0.3 * 3600.0, rel=1e-6)
    # the time share and the power do not depend on how the bout is cut into sub-steps
    b2 = _bodies(0.02)
    t2 = w3.locomotion_terms(b2, 0.3, env)
    parts = [w3.move_kt(b2, 0.0, 0.3, env, dt_s=900.0, terms=t2) for _ in range(4)]
    tot = w3.sum_loco(parts, 3600.0)
    assert float(tot["dist_m"]) == pytest.approx(float(r["dist_m"]), rel=1e-5)
    assert float(tot["metabolic_j"]) == pytest.approx(float(r["metabolic_j"]), rel=1e-5)


def _sea_geom(sea_cols=None, n=16, L=256.0):
    from haishool.life9.v3 import patch as pt
    z = torch.full((1, n, n), 10.0)
    if sea_cols is not None:
        z[0, sea_cols] = -50.0
    return pt.geometry_from_elevation(z, L, 0.0, 5.674e6)


def test_swimming_pays_its_paddling_efficiency_and_its_bow_wave_and_dense_bodies_tread_or_walk_the_bottom():
    """Review H1 and the review of 4 Oct 2026: a body that floats paddles at the surface at PADDLE_EFF with the wave
    drag of its bow wave; a body denser than the water treads it while it moves when its power covers the tread
    power (the rest propels it), and one too weak to tread walks the bottom with its weight in water, its airway
    under water the whole time. Swimming does not cost a small fraction of running per metre."""
    env = B.constant_env(t_air_k=300.0, dt_s=3600.0)
    fat = _bodies(0.02, fat=0.8, air=0.2)                              # floats
    lean = _bodies(0.02)                                               # sinks (a small lung, no fat)
    weak = _bodies(0.02)
    weak.genome["muscle"][:] = 1e-3                                    # too weak to tread
    assert float(B.density(fat)) < B.RHO_WATER < B.RHO_SEA < float(B.density(lean))
    tf = w3.locomotion_terms(fat, 1.0, env)
    v = float(tf["v_water"])
    sh = B.shape(fat)
    k3 = 0.5 * B.RHO_WATER * B.C_DRAG * float(sh["frontal"])
    fr = v / math.sqrt(K_G * float(sh["length"]))
    assert float(w3.wave_factor(torch.tensor(fr))) * k3 * v ** 3 == pytest.approx(
        float(tf["p_met"]) * B.MUSCLE_EFF * w3.PADDLE_EFF, rel=1e-6)
    assert fr > 0.3                                                   # near hull speed the wave drag matters
    # the dense body treads: it paddles with what the tread leaves
    tl = w3.locomotion_terms(lean, 1.0, env)
    assert bool(tl["treads"]) and 0 < float(tl["tread_w"]) < 0.2 * float(tl["p_met"])
    sl = B.shape(lean)
    vt = float(tl["v_water"])
    k3l = 0.5 * B.RHO_WATER * B.C_DRAG * float(sl["frontal"])
    frl = vt / math.sqrt(K_G * float(sl["length"]))
    assert float(w3.wave_factor(torch.tensor(frl))) * k3l * vt ** 3 == pytest.approx(
        (float(tl["p_met"]) - float(tl["tread_w"])) * B.MUSCLE_EFF * w3.PADDLE_EFF, rel=1e-6)
    # the weak one walks the bottom with its weight in water
    tw = w3.locomotion_terms(weak, 1.0, env)
    assert not bool(tw["treads"]) and float(tw["p_met"]) > 0
    sw = B.shape(weak)
    w_in_water = float(sw["m"]) * K_G * (1 - B.RHO_WATER / float(B.density(weak)))
    v_b = float(tw["v_water"])
    p = w3.KT_COST_J_PER_N * w_in_water / float(sw["length"]) * v_b + 0.5 * B.RHO_WATER * B.C_DRAG *         float(sw["frontal"]) * v_b ** 3 / B.MUSCLE_EFF
    assert p == pytest.approx(float(tw["p_met"]), rel=1e-6)
    # per metre at a founder's slow pace (a tenth of its power) swimming costs the same order as running, not 1/100
    for b in (fat, lean):
        t = w3.locomotion_terms(b, 0.1, env)
        cot_l = float(t["p_met"] / t["v_land"])
        cot_w = float(t["p_met"] / t["v_water"])
        assert cot_w > 0.1 * cot_l
    # in the sea: the treading body's airway is up while it moves and under water while it stands; the weak body is
    # under water the whole sub-step; the floater never
    geom = _sea_geom(slice(None))
    r = w3.move_kt(lean, 0.0, 0.5, env, geom=geom, dt_s=600.0)
    assert float(r["sunk_s"]) == pytest.approx(300.0, rel=1e-3) and bool(r["treading"])
    assert float(r["dive_trail_s"]) == pytest.approx(300.0, rel=1e-3)          # the stand comes last
    r = w3.move_kt(weak, 0.0, 0.5, env, geom=geom, dt_s=600.0)
    assert float(r["sunk_s"]) == pytest.approx(600.0, rel=1e-6) and float(r["dive_all"]) == 1.0
    r = w3.move_kt(fat, 0.0, 0.5, env, geom=geom, dt_s=600.0)
    assert float(r["sunk_s"]) == 0.0 and float(r["submerged"]) == 1.0                 # it swims
    assert float(B.submerged_share(fat, geom)) == pytest.approx(float(B.density(fat)) / B.RHO_SEA, rel=1e-4)


def test_the_path_is_crossed_at_each_mediums_speed():
    """Review M1: a body that starts in water and walks out crosses the water at its water speed and the land at its
    land speed (before, the share of the path in water was sampled over the land reach and kept for the whole
    sub-step, so it crossed dry ground at swimming speed)."""
    from haishool.life9.v3 import patch as pt
    env = B.constant_env(t_air_k=300.0, dt_s=3600.0)
    geom = _sea_geom(slice(0, 8))                                     # x < 128 m is sea, the rest land at 10 m
    b = _bodies(0.02, fat=0.8, air=0.2)
    b.pos[:] = torch.tensor([120.0, 100.0])
    t = w3.locomotion_terms(b, 1.0, env, geom)
    vl, vw = float(t["v_land"]), float(t["v_water"])
    r = w3.move_kt(b, 0.0, 1.0, env, geom=geom, dt_s=100.0, terms=t, samples=64)
    d = float(r["dist_m"])
    assert 8.0 < d < 120.0                                            # out of the water, not round the patch
    # 8 m of water at vw, the rest on land at vl, plus the climb of the shore from the sea level to 10 m
    t_lvl = 8.0 / vw + (d - 8.0) / vl
    climb_t = float(r["climb_j"]) / (B.MUSCLE_EFF * float(t["p_met"]))
    assert t_lvl + climb_t == pytest.approx(100.0, rel=0.03)
    assert float(r["submerged"]) == pytest.approx(8.0 / d, rel=0.15)
    assert d < 8.0 + vl * 100.0


def test_a_body_in_a_fire_heats_to_death_and_the_books_close(monkeypatch):
    """A body whose mouth is in a burning bed takes the flame's flux (PROVENANCE['fire_heat']). The arenas' air is held
    at 21 % O2 here, so the test does not depend on the planet's air (PROVENANCE['fire_possible'])."""
    from dataclasses import replace as dc_replace
    from haishool.life9.planet import items as itm
    from haishool.life9.planet import materials as mat
    w = earth_world()
    assert isinstance(w.reports["fire"][0]["fire_possible"], bool)
    real = w3.World3._air_today

    def rich_air(self):
        real(self)
        self.air = dc_replace(self.air, x_o2=torch.full_like(self.air.x_o2, 0.21))

    monkeypatch.setattr(w3.World3, "_air_today", rich_air)
    a = int(w.b.alive.sum(1).argmax())
    n = int(w.b.alive[a].nonzero()[0, 0])
    mouth = B.mouth_position(w.b)[a, n]
    bed = mat.species_vector({"wood": 2.0, "charcoal": 1.0})[None]
    slot = itm.spawn_fire(w.fires, torch.tensor([a]), torch.cat((mouth, torch.zeros(1)))[None], bed,
                          torch.tensor([1100.0]))
    assert int(slot[0]) >= 0
    kg = torch.zeros(w.A, mat.S, dtype=torch.float64)
    kg[a] = mat.species_vector({"wood": 2.0, "charcoal": 1.0}, dtype=torch.float64)
    w.iled.book_in(kg)
    em = mat.element_matrix().double()                    # the bed comes from outside the arena's books
    w.base["local"]["carbon"] += kg @ em[:, mat.ELEMENTS.index("C")]
    w.base["local"]["nitrogen"] += kg @ em[:, mat.ELEMENTS.index("N")]
    uid = int(w.b.uid[a, n])
    seen = {}

    def after(world, k):
        if k == 0:
            seen["dead"] = not bool(((world.b.uid[a] == uid) & world.b.alive[a]).any())
            # the flame's heat kills by cooking the body or by drying it out (its evaporation pays the heat)
            seen["hot"] = float(world.b.ledger["deaths_denaturation"][a] + world.b.ledger["deaths_dehydration"][a])

    w.step_day(after)
    assert seen["dead"] and seen["hot"] >= 1
    check_ledgers(w)


def test_movement_substeps_bound_each_step():
    w = earth_world()
    w.run(1)
    st = w.stats_day
    assert torch.all(st["coarse_moves"] <= st["moving"])
    assert float(st["distance_m"].sum()) > 0
    # a sub-step never carries a body below the budget farther than substep_m: K from the fastest planned travel
    assert w.config.substep_len_m == pytest.approx(2 * 256.0 / 16)


# ------------------------------------------------------------------------------------------------ the review's fixes
def test_capacity_binding_is_reported_as_a_failed_run_and_a_roomy_run_does_not_bind():
    """Review H2: births into a full arena fail, lose their development and are reported as a share of the births
    attempted; the run is marked capacity_bound from that day. A run with room to spare does not bind (the gate)."""
    roomy = earth_world()
    roomy.run(2)
    s = roomy.summary()
    assert sum(s["capacity_full"]) == 0 and not s["capacity_bound"] and s["capacity_bound_day"] is None
    w = earth_world()
    b = w.b
    b.alive[0, 32:] = True                                          # fill every slot of arena 0 ...
    src = torch.arange(64) % 32                                  # copies of the founders
    b.mass_kg[0, 32:] = b.mass_kg[0, src]
    b.frame_kg[0, 32:] = b.mass_kg[0, 32:]
    b.water_kg[0, 32:] = B.LEAN_WATER * b.mass_kg[0, 32:]
    b.reserve_j[0, 32:] = b.reserve_j[0, src]
    b.body_k[0, 32:] = b.body_k[0, src]
    for name in b.genome:
        b.genome[name][0, 32:] = b.genome[name][0, src]
    B.reset_ledgers(b)
    w._rebase()
    b.brood_j = B.child_cost_j(b) * 2                              # ... and complete every development
    w.step_day()
    s = w.summary()
    assert s["capacity_full"][0] > 0 and s["capacity_bound"] and s["capacity_bound_day"] == 1   # days completed
    assert 0 < s["capacity_full_share"][0] <= 1
    check_ledgers(w)


SEEN: dict = {}


def test_a_newborn_in_a_freed_slot_does_not_carry_the_dead_bodys_call():
    """Review L1: a child that takes the slot of a body that died in the same bout has made no call yet."""
    w = earth_world()
    b = w.b
    a = int(b.alive.sum(1).argmax())
    live = b.alive[a].nonzero()[:, 0]
    i, j = int(live[0]), int(live[1])
    b.water_kg[a, i] = 0.0                                           # dies of dehydration this bout
    b.brood_j[a, j] = 10 * float(B.child_cost_j(b)[a, j])           # its development is complete
    uid_i = int(b.uid[a, i])

    def first(world, k):
        if k == 0:
            SEEN.update(uid=int(world.b.uid[a, i]), alive=bool(world.b.alive[a, i]),
                        loud=float(world.last["loudness"][a, i]))

    w.step_day(first)
    assert SEEN["alive"] and SEEN["uid"] != uid_i                    # the slot was reused by a newborn
    assert SEEN["loud"] == 0.0


def test_set_up_books_the_sown_carbon_and_reports_the_founding(monkeypatch):
    """Review L4 and L15/H5: the sown carbon is taken from the climate's air at sowing; the founding day and the share
    of founders that died in their first bout are reported."""
    calls = []
    real = w3.World3._add_gas_arena

    def spy(self, c_fixed_kg, n_fixed_kg, *a, **k):
        calls.append(c_fixed_kg.clone())
        return real(self, c_fixed_kg, n_fixed_kg, *a, **k)

    monkeypatch.setattr(w3.World3, "_add_gas_arena", spy)
    w = w3.World3(small(source="earth"), [0])
    sown = torch.tensor(w.reports["vegetation_spin_up"]["sown_kg_c"], dtype=torch.float64)
    assert float(sown.sum()) > 0 and torch.allclose(calls[0], sown)
    found = w.reports["vegetation_spin_up"]["founding"][0]
    assert 0 <= found["day_of_year"] < found["year_days"] and abs(found["declination_deg"]) < 30
    w.step_day()
    rep = w.reports["founding"]
    assert len(rep["dead_first_bout_share"]) == w.A and 0 <= rep["extinct_first_bout_share"] <= 1


def test_the_air_thins_with_height_in_every_patch():
    """Review L3: the patches' partial pressures fall with each fine cell's height above the sea (the barometric
    factor, with the column's mean temperature), and the O2 mole fraction does not change."""
    w = earth_world()
    w.step_day()
    geom = w.geom
    h = (geom.elev.double() - geom.sea_level_m.double()[:, None, None]).clamp_min(0)
    pp = w3.cl.partial_pressures(w.clim, w.P)[w.arena_world]
    fac = w.p_o2_fine.double() / pp[:, w3.cl.I_O2][:, None, None]
    hi = h.reshape(w.A, -1).argmax(-1)
    for a in range(w.A):
        z = float(h.reshape(w.A, -1)[a, hi[a]])
        f = float(fac.reshape(w.A, -1)[a, hi[a]])
        t = float(w.t_fine.reshape(w.A, -1)[a, hi[a]])
        expect = math.exp(-float(w.gravity_w[0]) * 0.0289 * z / (8.314 * t))
        assert f == pytest.approx(expect, rel=2e-2), (a, z)
        assert f < 1.0 or z == 0.0
    x = float(w.air.x_o2[0])
    assert x == pytest.approx(float(pp[0, w3.cl.I_O2] / pp[0].sum()), rel=2e-2)


def test_deposits_deplete_the_struck_cell_and_items_leave_by_element():
    """Review L11 and L7: what strokes break off a cell is taken out of that cell's deposit only; items that leave the
    pool book their water to the air, carbon and nitrogen to the soil, the organic H and O apart, and only the other
    elements to the mineral sink."""
    from haishool.life9.planet import materials as mat
    w = earth_world()
    Sd = len(mat.CRUST_SPECIES)
    nn = w.config.cells ** 2
    assert tuple(w.deposit.shape) == (w.A, nn, Sd)
    before = w.deposit.clone()
    g = torch.zeros(w.A, nn, mat.S, dtype=torch.float64)
    sp = int(before[0, 5].argmax())
    g[0, 5, sp] = 1e-3
    w._ground_taken(g)
    d = (before - w.deposit)
    assert float(d[0, 5, sp]) == pytest.approx(1e-3 / w.geom.cell_m2, rel=1e-9)
    d[0, 5, sp] = 0.0
    assert float(d.abs().max()) == 0.0
    kg = torch.zeros(w.A, mat.S, dtype=torch.float64)
    kg[0, mat.IDX["meat"]] = 1.0
    c0 = {k: v.clone() for k, v in w.counters.items()}
    w._to_patch(None, None, kg)
    water = float(B.SPECIES_WATER[mat.IDX["meat"]])
    assert float(w.counters["items_water_air_kg"][0] - c0["items_water_air_kg"][0]) == pytest.approx(water)
    el = w.counters["mineral_sink_el_kg"][0] - c0["mineral_sink_el_kg"][0]
    for e in ("C", "N", "H", "O"):
        assert float(el[mat.ELEMENTS.index(e)]) == 0.0, e
    ho = float(w.counters["organic_ho_kg"][0] - c0["organic_ho_kg"][0])
    em = mat.element_matrix().double()[mat.IDX["meat"]]
    rest = float(em.sum()) - water - float(em[mat.ELEMENTS.index("C")] + em[mat.ELEMENTS.index("N")])
    assert ho + float(el.sum()) == pytest.approx(rest, rel=1e-9)


def test_fire_flux_rises_continuously_into_the_bed():
    """Review L2: the fires' flux on a body is the point source capped by the flame's emissive power, so it is
    continuous across the bed's rim (it jumped about 1,000-fold at 0.5 m)."""
    w = earth_world()
    fl = {"fire_pos": torch.zeros(1, 1, 3), "mean_hrr_w": torch.full((1, 1), 5700.0)}
    t0 = torch.full((1, 1), 1100.0)
    live = torch.ones(1, 1, dtype=torch.bool)
    d = torch.tensor([0.01, 0.05, 0.45, 0.49, 0.51, 0.55, 1.0, 5.0])
    pos = torch.stack((d, torch.zeros_like(d)), -1)[None]
    q = w._fire_flux(fl, t0, live, pos, 256.0)[0]
    flame = w3.K.SIGMA * 1100.0 ** 4
    assert float(q[0]) == pytest.approx(flame, rel=1e-6)                       # in the flame: its emissive power
    assert bool((q[1:] <= q[:-1]).all())                                       # falls with distance
    assert float(q[3] / q[4]) < 1.2                                            # no jump at the rim
    assert float(q[-1]) == pytest.approx(w3.cr.RADIANT_FRACTION * 5700.0 / (4 * math.pi * 25.0), rel=1e-6)


def test_the_summary_reports_bounds_baseline_and_viability():
    """Review M8, M9 and L17: mouth statistics weighted by intensity with the founders' baseline replayed on the same
    inputs, the cost bounds of the day, and the viability measures."""
    w = earth_world()
    w.step_day()
    s = w.summary()
    st = s["behaviour_today"]
    assert sum(st["sense_bodies"]) > 0 and sum(st["replay_n_on_plant"]) + sum(st["replay_n_off_plant"]) > 0
    assoc = s["mouth_association_today"]["plant"]
    for key in ("mouth_on", "mouth_off", "replay_real_on", "replay_base_on", "replay_real_off", "replay_base_off"):
        assert all(0.0 <= v <= 1.0 for v in assoc[key]), key
    assert sum(st["mouth_sum"]) >= sum(st["mouth_events"]) >= 0
    for key in ("sense_bound_share", "sense_far_bound_share", "sense_beyond_share", "no_item_slot",
                "carcass_no_slot", "coarse_moves_share", "travel_beyond_far_share", "nonfinite"):
        assert key in s["bounds_today"], key
    for key in ("births_today", "deaths_today", "turnover_today", "capacity_full_share", "shadow_lineages_alive",
                "tissue_overasked_share", "fat_runway_days_mean", "food_over_spent_today", "arenas_without_plants",
                "brood_j_mean"):
        assert key in s, key
    assert all(sl <= sp for sl, sp in zip(s["shadow_lineages_alive"], s["shadow_population"]))
    assert "fire_possible" in s["worlds"][0] and "water_to_air_kg" in s["worlds"][0]
    rep = w.bound_report()
    assert rep["n"] == int(w.b.alive.sum())
    json.dumps(s)
