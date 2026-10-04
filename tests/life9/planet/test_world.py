"""Tests of the planet world (PLANET-SPEC 2.12): construction, the day's flows and ledgers, determinism and
exact resume, a friendly population, the viewer's frame format and the command line."""
from __future__ import annotations

import io
import json
import math

import pytest
import torch

from haishool.life9.planet import chain, creatures as cr_, crafting as cf, formation as fm
from haishool.life9.planet import items as it, materials as mat
from haishool.life9.planet import world as wd
from haishool.life9.planet.__main__ import main as cli
from haishool.life9.planet.world import PlanetConfig, PlanetWorld

TAGS = {"chain", "derived", "reference", "new_rule"}
#: float32 tolerance of a ledger closed in float64: the relative error against the ledger's scale
LEDGER_REL = 1e-5
#: 781 (world7, locked) and 13 (world7 reaches only replicators: the synthetic fallback, rotating)
SEEDS = (781, 13)


def small(**kw) -> PlanetConfig:
    base = dict(G=12, capacity=64, founders=32, items=512, fires=32, hidden=32, initial_hidden=16,
                climate_fast_chunks=3, climate_slow_chunks=1, bio_spin_days=60)
    base.update(kw)
    return PlanetConfig(**base)


def tiny(**kw) -> PlanetConfig:
    base = dict(G=8, capacity=32, founders=16, items=256, fires=16, hidden=24, initial_hidden=12,
                climate_fast_chunks=2, climate_slow_chunks=1, bio_spin_days=30)
    base.update(kw)
    return PlanetConfig(**base)


def assert_closed(world, tol=LEDGER_REL):
    """Every ledger within tol of its scale; the checked ones (``world.CHECKED_LEDGERS``: the sensitive
    closures between modules) are what matters, the aggregates (INFO_LEDGERS) are reported for information."""
    led = world.ledgers()
    for name, v in led.items():
        for w, rel in enumerate(v["rel"]):
            assert math.isfinite(rel) and rel < tol, (name, w, v["error"][w], v["scale"][w])
    assert set(wd.CHECKED_LEDGERS) <= set(led)
    return led


@pytest.fixture(scope="module")
def built():
    return PlanetWorld(small(), SEEDS)


@pytest.fixture(scope="module")
def ran():
    """The same world run for 20 days, with the ledgers after every day."""
    world = PlanetWorld(small(), SEEDS)
    world.strict = True                     # a negative litter, ground-water or vapour routing raises
    worst = {}
    for _ in range(20):
        world.step()
        for name, v in world.ledgers().items():
            worst[name] = max(worst.get(name, 0.0), max(v["rel"]))
    return world, worst


# ------------------------------------------------------------------------------------------- construction
def test_construction_781_and_a_synthetic_fallback(built):
    w = built
    assert w.W == 2 and w.seeds == list(SEEDS)
    assert w.input_notes[0] == "world7" and w.specs[0].source == "world7"
    assert w.input_notes[1].startswith("synthetic fallback") and w.specs[1].source == "synthetic"
    assert w.spec_failures == {}
    assert w.specs[0].tidally_locked and not w.specs[1].tidally_locked
    C = w.globe.C
    assert C == 6 * 12 * 12
    assert w.stock.shape == (2, C, len(mat.CRUST_SPECIES)) and bool((w.stock >= 0).all())
    land = w.terrain["land"]
    assert 0 < float(land.float().mean()) < 1
    # climate spun up to the chain's surface temperature (the 3 K gate), plants grown
    target = torch.tensor([s.t_surface_target_k for s in w.specs], dtype=torch.float64)
    mean_t = (w.clim.T.double() * w.globe.area64).sum(-1) / (4 * math.pi)
    assert bool(((mean_t - target).abs() < 3.0).all()), (mean_t, target)
    assert bool((w.bio.plant_c.sum(-1) > 0).all())
    # founders on land with chain-derived bodies and brains of the configured size
    cr = w.cr
    assert cr.shape == (2, 64) and [int(x) for x in cr.alive.sum(1)] == [32, 32]
    cells = w.globe.cell_of(cr.pos)
    assert bool(land.gather(1, cells)[cr.alive].all())
    assert cr.genome["Wh"].shape[-1] == 32 and int(cr.genome["k"].max()) == 16
    assert float(cr.genome["insulation"][0, 0]) == pytest.approx((303 - w.inputs[0]["planet"]["t_surface_k"]) / 8)
    # the day-0 ledgers are at their baseline
    assert_closed(w, 1e-12)


def test_config_is_checked():
    for bad in (dict(vocal_dims=4), dict(dt_days=2), dict(founders=0), dict(source="moon"),
                dict(allowed_actions=("fly",)), dict(founder_genes={"adult_mass_kg": 5.0}),
                dict(weight_dtype="float16")):
        with pytest.raises(ValueError):
            PlanetWorld(small(**bad), (781,))


def test_planet_inputs_fallback_rules():
    x, note = wd.planet_inputs(781)
    assert x["source"] == "world7" and note == "world7"
    x, note = wd.planet_inputs(13)
    assert x["source"] == "synthetic" and "bodies" in note
    x, note = wd.planet_inputs(781, "synthetic")
    assert x["source"] == "synthetic"


# ------------------------------------------------------------------------------------------- the day and ledgers
def test_twenty_days_every_ledger_closes(ran):
    world, worst = ran
    assert world.day == 20
    for name, rel in worst.items():
        assert rel < LEDGER_REL, (name, rel)
    led = assert_closed(world)
    names = {"energy", "bodies_carbon", "bodies_water", "carbon", "carbon_bio", "carbon_mobile", "oxygen",
             "oxygen_bio", "oxygen_mobile", "oxygen_atoms", "water", "water_climate", "water_mobile", "items",
             "deposits"}
    assert names <= set(led)
    assert len(led["items"]["elements"][0]) == len(mat.ELEMENTS)
    # the ledgers were exercised: individuals breathed, ate, drank, died and rotted, items came and went
    f = world.flows
    assert bool((f["creature_o2_mol"] > 0).all()) and bool((f["creature_co2_mol"] > 0).all())
    assert bool((f["litter_c_kg"] > 0).all()) and bool((f["vapour_kg"] > 0).all())
    assert float(f["corpse_in_kg"].sum()) > 0 and float(f["decayed_kg"].sum()) > 0
    assert float(f["dep_in_kg"].sum()) > 0 and float(f["drunk_soil_kg"].sum()) > 0
    assert float(world.bio.c_harvested.sum()) > 0 and float(f["forage_water_kg"].sum()) > 0
    assert float(f["work_j"].sum()) > 0
    # metabolic water: the O of the O2 not returned as CO2 (the oxygen_atoms ledger checks it against the air)
    assert bool((world.cr.ledger["metabolic_w"] > 0).all())
    assert max(led["oxygen_atoms"]["rel"]) < 1e-6 and max(led["deposits"]["rel"]) < 1e-6


def test_summary_reports_the_world(ran):
    world, _ = ran
    s = world.summary()
    json.dumps(s)
    assert s["day"] == 20 and len(s["worlds"]) == 2
    assert s["co2_capped_share"] == pytest.approx(sum(sp.co2_capped for sp in world.specs) / 2)
    for row in s["worlds"]:
        for key in ("population", "births", "deaths", "mean_brain_units", "mean_diet", "mean_mass_kg", "mean_t_k",
                    "land_t_k", "p_o2_pa", "p_co2_pa", "npp_kg_c_m2_yr", "fires_lit", "items_by_class",
                    "transforms_kg", "co2_capped", "today"):
            assert key in row, key
        assert set(row["deaths"]) == set(cr_.CAUSES[1:])
        assert set(row["items_by_class"]) == set(mat.CLASSES)
        assert set(row["transforms_kg"]) == {t.name for t in mat.TRANSFORMS}
        assert row["p_o2_pa"] > 0 and row["p_co2_pa"] > 0 and 150 < row["mean_t_k"] < 400
        assert set(row["today"]["actions"]) == set(wd.ACTIONS)
    deaths = sum(sum(r["deaths"].values()) for r in s["worlds"])
    assert deaths + sum(r["population"] for r in s["worlds"]) - sum(r["births"] for r in s["worlds"]) == 64


def test_fires_cooking_and_transforms_are_booked():
    """An Earth world (oxygen 1 PAL, so fires burn): a fire with items in its bed, and cooks at it."""
    world = PlanetWorld(tiny(allowed_actions=("cook",)), (0,), inputs=[chain.earth_inputs()])
    cr, pool, fires = world.cr, world.pool, world.fires
    w0 = torch.tensor([0])

    def one(s):
        v = torch.zeros(mat.S)
        v[mat.IDX[s]] = 1.0
        return v[None]

    p0 = cr.pos[0, 0][None].clone()
    fuel = torch.zeros(1, mat.S)
    fuel[0, mat.IDX["charcoal"]], fuel[0, mat.IDX["wood"]] = 20.0, 10.0
    assert int(it.spawn_fire(fires, w0, p0, fuel, torch.tensor([1200.0]))[0]) >= 0
    for s, kg in (("limestone", 3.0), ("meat", 1.0), ("clay", 1.0), ("wood", 2.0)):
        it.spawn(pool, w0, one(s), torch.tensor([kg]), p0, torch.tensor([290.0]), torch.tensor([-1]))
    K = cr.inv.shape[-1]
    for n in range(4):
        cr.pos[0, n] = p0[0]
        cr.inv[0, n, 0] = it.spawn(pool, w0, one("meat"), torch.tensor([0.5]), p0, torch.tensor([300.0]),
                                   torch.tensor([n * K]))[0]
    world.base = world._stores()            # the fire and items were put in from outside: rebase
    world.step()
    f = world.flows
    assert float(f["fire_heat_j"][0]) > 1e8
    assert float(f["air_kg"][0, 0]) < 0 and float(f["air_kg"][0, 1]) > 0 and float(f["air_kg"][0, 2]) > 0
    assert float(f["to_soil_kg"][0].sum()) > 0                       # charred tissue
    names = [t.name for t in mat.TRANSFORMS]
    for name in ("clay_to_ceramic", "limestone_to_lime", "wood_to_charcoal"):
        assert float(f["transformed_kg"][0, names.index(name)]) > 0.5, name
    assert world.today["actions"][0][wd.ACT["cook"]] > 0
    assert_closed(world)
    world.step()
    assert_closed(world)


# ------------------------------------------------------------------------------------------- determinism
def test_determinism_and_exact_resume_on_cpu():
    a = PlanetWorld(tiny(), SEEDS)
    b = PlanetWorld(tiny(), SEEDS)
    assert a.state_hash() == b.state_hash()
    for _ in range(6):
        a.step()
        b.step()
    assert a.state_hash() == b.state_hash()
    buf = io.BytesIO()
    torch.save(b.state_dict(), buf)
    buf.seek(0)
    c = PlanetWorld.from_state(torch.load(buf, weights_only=True), "cpu")
    assert c.day == 6 and c.state_hash() == b.state_hash()
    for _ in range(6):
        a.step()
        c.step()
    assert a.day == c.day == 12
    assert a.state_hash() == c.state_hash()
    assert a.ledgers() == c.ledgers()
    assert json.dumps(a.summary()) == json.dumps(c.summary())
    assert a.frame(1, fields=True) == c.frame(1, fields=True)


# ------------------------------------------------------------------------------------------- a friendly world
def test_population_stays_alive_for_twenty_days_in_a_friendly_config():
    """Drinking is the only action and the founders can drink sea water (swim gene above 0.5), on the temperate
    synthetic planet (the background grazing and drinking of the time budget keep them fed and watered too)."""
    world = PlanetWorld(small(allowed_actions=("drink",), founder_genes={"swim": 0.6}), (13,))
    for _ in range(20):
        world.step()
        assert int(world.cr.alive.sum()) > 0
    assert int(world.cr.alive.sum()) >= 16
    assert world.summary()["worlds"][0]["today"]["actions"]["drink"] > 0
    assert_closed(world)


# ------------------------------------------------------------------------------------------- frames
STATIC_KEYS = {"schema", "seed", "planet", "habitat_radius_m", "G", "cells", "centers", "elevation_m", "land",
               "sea_level_m", "relief_exaggeration", "species", "actions", "item_classes"}


def test_static_and_frame_follow_the_format(ran, tmp_path):
    world, _ = ran
    for w in range(world.W):
        st = world.static(w)
        assert set(st) == STATIC_KEYS and st["schema"] == "life9-globe-v1" and st["seed"] == world.seeds[w]
        assert set(st["planet"]) == set(wd.STATIC_PLANET)
        C = world.globe.C
        assert st["cells"] == C and st["G"] == 12
        assert len(st["centers"]) == C and all(len(p) == 3 for p in st["centers"])
        assert all(isinstance(v, int) for v in st["elevation_m"]) and len(st["elevation_m"]) == C
        assert set(st["land"]) <= {0, 1} and len(st["land"]) == C
        assert st["species"] == list(mat.SPECIES) and st["actions"] == list(wd.ACTIONS)
        assert st["item_classes"] == list(mat.CLASSES)
        fr = world.frame(w, fields=True)
        assert set(fr) == {"day", "sun", "agents", "fires", "items", "fields"} and fr["day"] == 20
        assert len(fr["sun"]) == 3 and abs(sum(v * v for v in fr["sun"]) - 1) < 1e-3
        assert len(fr["agents"]) == int(world.cr.alive[w].sum())
        for row in fr["agents"]:
            assert len(row) == 13
            assert all(isinstance(row[i], int) for i in (0, 6, 8, 9, 10))
            assert 0 <= row[10] < len(wd.ACTIONS) and 0 <= row[8] < 8
        for row in fr["items"]:
            assert len(row) == 5 and isinstance(row[3], int) and 0 <= row[3] < len(mat.CLASSES) and row[4] > 0
        assert len(fr["items"]) <= wd.MAX_FRAME_ITEMS
        for row in fr["fires"]:
            assert len(row) == 4
        assert set(fr["fields"]) == {"T_k", "plant", "snow", "soil"}
        for key, values in fr["fields"].items():
            assert len(values) == C and all(isinstance(v, int) for v in values)
        assert all(0 <= v <= 255 for v in fr["fields"]["plant"] + fr["fields"]["soil"])
        assert set(fr["fields"]["snow"]) <= {0, 1}
        json.dumps(st)
        json.dumps(fr)
        assert "fields" not in world.frame(w)
    # a locked planet's star stands still over lon 0, lat 0
    assert world.sun(0) == [1.0, 0.0, 0.0]
    try:
        from haishool.life9.planet import view
    except ImportError:
        return
    report = view.build_page(world.static(0), [world.frame(0, fields=True)], tmp_path / "view.html")
    assert (tmp_path / "view.html").exists() and report["frames"] == 1


# ------------------------------------------------------------------------------------------- provenance and CLI
def test_provenance_is_tagged(built):
    prov = built.provenance()
    assert len(prov) > 500
    for key, (tag, note) in prov.items():
        assert note, key
        assert all(part in TAGS for part in str(tag).split("+")), (key, tag)
    for key in wd.WORLD_PROVENANCE:
        assert key in prov


def test_describe_cli_prints_the_spec_with_provenance(capsys):
    cli(["describe", "--seed", "781"])
    out = capsys.readouterr().out
    spec = fm.build_planet(chain.chain_inputs(781), 781)
    assert "PlanetSpec seed 781 (world7)" in out
    assert "radius_m" in out and "[derived" in out and "[chain]" in out
    assert "check_spec: passes" in out
    assert f"{spec.gravity_m_s2:.6g}" in out


def test_run_cli_writes_and_resumes(tmp_path):
    args = ["run", "--seeds", "781", "--days", "2", "--device", "cpu", "--out", str(tmp_path), "--every", "1",
            "--view-every", "1", "--G", "8", "--capacity", "16", "--founders", "8", "--items", "128",
            "--fires", "8", "--hidden", "16", "--initial-hidden", "8", "--bio-spin-days", "20",
            "--climate-fast-chunks", "1", "--climate-slow-chunks", "1"]
    cli(args)
    run = json.loads((tmp_path / "run.json").read_text(encoding="utf-8"))
    assert run["seeds"] == [781] and run["config"]["G"] == 8 and run["specs"][0]["seed"] == 781
    rows = [json.loads(line) for line in (tmp_path / "summary.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["day"] for r in rows] == [1, 2] and "ledgers" in rows[0]
    assert (tmp_path / "state.pt").exists() and (tmp_path / "view-static.json").exists()
    gates = json.loads((tmp_path / "gates.json").read_text(encoding="utf-8"))
    assert gates["day"] == 2 and all(gates["ledgers_closed"])
    assert not (tmp_path / "state.pt.tmp").exists()
    # a job killed after day 2 had logged day 3 already: the resume drops that row and logs day 3 once
    with (tmp_path / "summary.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"day": 3, "stale": True}) + "\n")
    cli(["run", "--resume", str(tmp_path / "state.pt"), "--days", "1", "--device", "cpu", "--out", str(tmp_path)])
    rows = [json.loads(line) for line in (tmp_path / "summary.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["day"] for r in rows] == [1, 2, 3] and "stale" not in rows[-1]


# ------------------------------------------------------------------------------------------- integration round
def test_founders_are_viable_for_thirty_days():
    """The founder-viability gate (WORLD_PROVENANCE['world.gates']) on a small default-like world: the founders
    start on habitable cells, graze with the forage's water for the time their day leaves, drink at fresh water,
    move about Garland's mean daily distance and stop at the shore. The temperate synthetic planet keeps at least
    half of them for 30 days; planet 781 (locked, a narrow band of wet and mild land) keeps at least 30 %."""
    world = PlanetWorld(small(), SEEDS)
    world.strict = True
    assert world.reports["founders"]["habitat"] == ["habitable", "habitable"]
    for _ in range(wd.GATE_DAYS):
        world.step()
    g = world.gates()
    assert all(g["ledgers_closed"]), g
    assert min(g["founders_alive_share"]) >= 0.3, g
    assert g["founders_viable"][1] is True, g
    assert all(r["population"] > 0 for r in world.summary()["worlds"])


def test_actions_happen_where_the_individual_looked(monkeypatch):
    """observe, think, act, then move: grazing uses the cell the senses saw (no approach for grazers), the
    resters stay put, and the day's travel is Garland-scale."""
    seen = {}
    real_observe, real_eat = wd.sn.observe, wd.cr_.eat_plants

    def observe(cr, globe, *a, **k):
        seen["cell"] = globe.cell_of(cr.pos).clone()
        return real_observe(cr, globe, *a, **k)

    def eat(cr, mask, globe, *a, **k):
        seen["eat_cell"], seen["mask"] = k["cell"].clone(), mask.clone()
        return real_eat(cr, mask, globe, *a, **k)
    monkeypatch.setattr(wd.sn, "observe", observe)
    monkeypatch.setattr(wd.cr_, "eat_plants", eat)
    world = PlanetWorld(tiny(allowed_actions=("forage", "rest", "knap")), (13,))
    p0 = world.cr.pos.clone()
    world.step()
    m = seen["mask"]
    assert bool(m.any()) and torch.equal(seen["eat_cell"][m], seen["cell"][m])
    rest = (world.last_action == wd.ACT["rest"]) & world.cr.alive
    assert bool(rest.any()) and torch.equal(world.cr.pos[rest], p0[rest])
    moved = world.cr.alive & ~rest
    km = (wd.gb.angle(world.cr.pos, p0) * world.config.habitat_radius_m / 1000)[moved]
    assert 0.2 < float(km.mean()) < 4.0                     # Garland: 2.4 km/day at 30 kg (5-10 km before)


def test_a_dried_body_routes_no_negative_water():
    """Founders dried to half their water norm die: their corpse parts hold no more water than they had, and
    nothing negative is routed (strict mode), so the ocean is not drawn on to fill corpse items."""
    world = PlanetWorld(tiny(allowed_actions=("rest",)), (13,))
    world.strict = True
    cr = world.cr
    a = cr.alive.clone()
    before = cr.water_kg.clone()
    cr.water_kg = torch.where(a, 0.5 * cr_.water_norm(cr.mass_kg), cr.water_kg)
    cr.ledger["spawned_w"] += ((cr.water_kg - before).double() * a).sum(1)
    world.clim.soil = torch.zeros_like(world.clim.soil)        # nothing to drink: they die of thirst today
    world.clim = wd.cl.reset_water_ledger(world.clim, world.globe)
    world.base = world._stores()
    world.step()
    L = cr.ledger
    assert float(L["deaths_dehydration"].sum()) == float(a.sum())
    item_w = float((cf.ledger_mass(world.pool, world.fires) @ world._water).sum())
    assert 0 < item_w <= float(L["dead_w"].sum())
    assert_closed(world)


def test_collected_lumps_deplete_the_float64_deposits():
    world = PlanetWorld(tiny(allowed_actions=("collect",)), (781,))
    assert world.stock.dtype == torch.float64
    for _ in range(3):
        world.step()
    made = float(world.flows["dep_in_kg"].sum())
    lost = float(((world.stock0 - world.stock) * world.cell_m2[None, :, None]).sum())
    assert made > 1.0 and lost == pytest.approx(made, rel=1e-6)
    led = assert_closed(world)
    assert led["deposits"]["rel"][0] < 1e-6


def test_starving_at_an_empty_reserve_is_felt_and_the_first_day_teaches_nothing():
    world = PlanetWorld(tiny(allowed_actions=("rest",)), (13,))
    cr = world.cr
    a = cr.alive.clone()
    assert bool(torch.isnan(cr.baseline[a]).all())          # the running mean starts at the first outcome
    cr.reserve_j = torch.where(a, torch.zeros_like(cr.reserve_j), cr.reserve_j)
    wh0 = cr.wh_live.clone()
    world.step()
    both = a & cr.alive
    assert float(world.today["mean_reward"][0]) < -0.5          # lean tissue burnt counts (it read 0 before)
    assert bool(torch.isfinite(cr.baseline[both]).all())
    assert torch.equal(cr.wh_live[both], wh0[both])             # modulator 0 on the first day
    world.step()
    still = both & cr.alive
    assert not torch.equal(cr.wh_live[still], wh0[still])


def test_the_pool_keeps_free_slots_by_rotting_its_oldest_carrion():
    world = PlanetWorld(tiny(allowed_actions=("rest",)), (13,))
    pool = world.pool
    I = pool.shape[1]
    n = I - 4
    comp = torch.zeros(n, mat.S)
    comp[:, mat.IDX["bone"]] = 1.0
    pos = world.cr.pos[0, :1].expand(n, 3)
    it.spawn(pool, torch.zeros(n, dtype=torch.long), comp, torch.full((n,), 0.5), pos, torch.full((n,), 290.0),
             torch.full((n,), -1))
    pool.age_d[0, :n] = torch.arange(n, dtype=torch.float32)
    world.base = world._stores()
    world.step()
    free = I - int(pool.alive[0].sum())
    assert free >= math.ceil(I * wd.POOL_FREE_SHARE)
    row = world.summary()["worlds"][0]
    assert row["items_recycled"] > 0 and row["item_pool_fill"] <= 1 - wd.POOL_FREE_SHARE + 1e-9
    assert_closed(world)


def test_strikers_walk_to_whom_they_see_and_the_encounters_are_reported():
    world = PlanetWorld(tiny(allowed_actions=("strike",)), (13,))
    cr = world.cr
    p = cr.pos[0, 0]
    jitter = torch.nn.functional.normalize(torch.randn(16, 3, generator=torch.Generator().manual_seed(0)), dim=-1)
    cr.pos[0, :16] = torch.nn.functional.normalize(p + 3e-3 * jitter, dim=-1)       # within about 30 m
    world._sync_held()
    world.base = world._stores()
    world.step()
    enc = world.summary()["worlds"][0]["encounters"]["strike"]
    assert enc["tries"] > 0 and enc["done"] > 0 and int(world.flows["strikes"][0]) == enc["done"]
    assert float(world.flows["walk_m"][0]) > 0
    assert_closed(world)


def test_state_is_copied_to_the_host_and_resumes_exactly_with_a_fire_alive():
    """Earth inputs (O2 1 PAL), a 60 kg charcoal fire with tissue and limestone in its bed: the split falls while
    the fire burns; the state is on the host, tensor_bytes counts the live tensors, and the resumed world
    (taking the state's tensors with own=True) continues exactly."""
    world = PlanetWorld(tiny(allowed_actions=("cook", "feed_fire", "rest")), (0,), inputs=[chain.earth_inputs()])
    cr, pool, fires = world.cr, world.pool, world.fires
    w0 = torch.tensor([0])
    p0 = cr.pos[0, 0][None].clone()
    fuel = torch.zeros(1, mat.S)
    fuel[0, mat.IDX["charcoal"]], fuel[0, mat.IDX["wood"]] = 60.0, 10.0
    assert int(it.spawn_fire(fires, w0, p0, fuel, torch.tensor([1200.0]))[0]) >= 0
    for s, kg in (("limestone", 3.0), ("meat", 1.0), ("bone", 0.5)):
        comp = torch.zeros(1, mat.S)
        comp[0, mat.IDX[s]] = 1.0
        it.spawn(pool, w0, comp, torch.tensor([kg]), p0, torch.tensor([290.0]), torch.tensor([-1]))
    world.base = world._stores()
    world.step()
    assert bool(fires.alive.any())
    sd = world.state_dict()
    tensors = []

    def walk(o):
        if torch.is_tensor(o):
            tensors.append(o)
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)
    for key in ("terrain", "stock", "stock0", "climate", "bio", "items", "fires", "creatures", "flows", "base",
                "carry"):
        walk(sd[key])
    assert all(t.device.type == "cpu" for t in tensors)
    assert world.tensor_bytes() == sum(t.numel() * t.element_size() for t in tensors)
    buf = io.BytesIO()
    torch.save(sd, buf)
    buf.seek(0)
    other = PlanetWorld.from_state(torch.load(buf, weights_only=True), "cpu", own=True)
    assert other.state_hash() == world.state_hash()
    for _ in range(3):
        world.step()
        other.step()
    assert other.state_hash() == world.state_hash()
    assert float(world.flows["fire_heat_j"].sum()) > 0 and float(world.flows["char_c_kg"].sum()) > 0
    assert_closed(world)


_DET = """
import sys, torch
sys.path.insert(0, {root!r})
from haishool.life9.planet.world import PlanetConfig, PlanetWorld
torch.set_num_threads({threads})
cfg = PlanetConfig(G=8, capacity=32, founders=16, items=256, fires=16, hidden=24, initial_hidden=12,
                   climate_fast_chunks=2, climate_slow_chunks=1, bio_spin_days=30)
w = PlanetWorld(cfg, (781, 13))
for _ in range(3):
    w.step()
print(w.state_hash())
"""


def test_the_hash_does_not_depend_on_the_hash_seed_or_the_threads():
    import os
    import subprocess
    import sys
    from pathlib import Path
    root = str(Path(wd.__file__).resolve().parents[3])
    out = []
    for seed, threads in (("1", 1), ("2", 4)):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        r = subprocess.run([sys.executable, "-c", _DET.format(root=root, threads=threads)], capture_output=True,
                           text=True, env=env, timeout=600)
        assert r.returncode == 0, r.stderr[-2000:]
        out.append(r.stdout.strip().splitlines()[-1])
    assert out[0] == out[1]


def test_earth_reference_control_builds_with_fire_possible():
    from haishool.life9.planet.world import PlanetConfig, PlanetWorld
    import dataclasses
    fields = {f.name for f in dataclasses.fields(PlanetConfig)}
    small = {k: v for k, v in dict(G=12, capacity=32, founders=8, items=256, fires=16, bio_spin_days=30,
                                    climate_fast_chunks=1, climate_slow_chunks=1).items() if k in fields}
    world = PlanetWorld(PlanetConfig(source="earth", **small), [1, 2], "cpu")
    assert all(s.source == "earth_reference" for s in world.specs)
    o2 = world.specs[0].partial_pressure_pa["O2"] / world.specs[0].surface_pressure_pa
    assert o2 > 0.15            # fire is physically possible on the control, unlike planet 781
    world.step()
