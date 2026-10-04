"""Regression tests of the review of 4 Oct 2026 (protocol, set-up, outputs; the body physics is in test_body_air.py):
the chain cache read first on hosts without world7, the physical planet pick, a world that depends on its seed alone,
the state held once on load and at set-up, checkpoints bound to their rules, a rules hash over the imported modules,
private constants and the Earth reference, the starting conditions pinned before the first step, errors recorded,
resumes checked, landless worlds with sea patches, the per-arena facts in the JSON outputs, the pool bounds, the
gene report, the climate carrying formation3's moles, and fire flagged outside its calibration."""
from __future__ import annotations

import contextlib
import io
import json
import math
import shutil
from pathlib import Path

import pytest
import torch

from haishool.life9.planet import chain as ch
from haishool.life9.planet import climate as cl
from haishool.life9.planet import constants as K
from haishool.life9.v3 import __main__ as cli
from haishool.life9.v3 import body as B
from haishool.life9.v3 import brain3 as b3
from haishool.life9.v3 import formation3 as F3
from haishool.life9.v3 import world3 as w3


def small(**kw) -> w3.V3Config:
    cfg = dict(G=8, patches=2, patch_m=256.0, cells=16, capacity=96, founders=32, items=128, fires=8, hidden=16,
               k_founder=(2, 8), bouts=4, climate_fast_chunks=1, climate_slow_chunks=1, bio_spin_days=20,
               bio_spin_segments=1, veg_spin_days=20, veg_spin_bouts=2)
    cfg.update(kw)
    return w3.V3Config(**cfg)


def sets(cfg: w3.V3Config) -> list:
    out = []
    for k, v in cfg.to_dict().items():
        if k != "source" and v is not None:
            out += ["--set", f"{k}={','.join(str(x) for x in v) if isinstance(v, list) else v}"]
    return out


def run_cli(argv) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert cli.main(argv) == 0
    return buf.getvalue()


@pytest.fixture
def no_world7(monkeypatch):
    """A host without world7 (no scipy): running the chain fails; only the cache can be read."""
    def fail(*a, **k):
        raise ImportError("scipy blocked")

    monkeypatch.setattr(ch, "_modules", fail)
    monkeypatch.setattr(ch, "_scan", fail)


# ------------------------------------------------------------------------------------------------ the chain inputs
def test_a_host_without_world7_reads_the_caches_first(no_world7):
    """Review H4: a no-planet seed's absent cache is read before the chain is run, so a host without scipy builds
    every cached seed."""
    w = w3.World3(small(), [0, 2, 3, 781])
    assert [r["kind"] for r in w.absent] == ["no_star", "no_habitable_planet", "no_bodies_era"] and w.seeds == [781]


def test_an_uncached_seed_on_such_a_host_is_an_error_not_an_outcome(no_world7, tmp_path, monkeypatch):
    for name in ("absent-world7-0.json", "world7-781.json"):
        shutil.copyfile(ch.CACHE_DIR / name, tmp_path / name)
    monkeypatch.setattr(ch, "CACHE_DIR", tmp_path)
    got = w3.resolve_inputs([0, 781, 5], small())
    assert got[0][0] is None and got[0][1]["kind"] == "no_star" and got[1][0]["seed"] == 781
    assert got[2][0] is None and got[2][1]["kind"] == w3.ERROR_KIND and "ImportError" in got[2][1]["cause"]
    w = w3.World3(small(), [0, 781, 5], resolved=got)
    rows = w.outcomes()
    assert [r["kind"] for r in rows][2] == w3.ERROR_KIND and rows[2]["error"] and w.reports["errors"][0]["seed"] == 5


def test_the_planet_is_the_physical_habitable_zone_pick():
    """Review H6 (rule 2): the innermost temperate rocky planet that is terran with liquid water, never the planet the
    chain climbed furthest on; the bridge's pick of chain version 2 is recorded beside it."""
    steps = [{"era": "surface_1", "surface": "terran", "water": "frozen"},
             {"era": "surface_2", "surface": "terran", "water": "liquid"},
             {"era": "surface_3", "surface": "terran", "water": "liquid"}]
    k, oceans = ch.physical_pick(steps)
    assert k == 2 and len(oceans) == 2
    assert ch.physical_pick([{"era": "surface_1", "surface": "ice", "water": "liquid"}])[0] == 0
    x = ch.chain_inputs(781)
    assert x["chain_version"] == ch.CHAIN_VERSION == "life9-chain-v3" and x["pick_rule"] == ch.PICK_RULE
    assert x["planet_index"] == 1 and x["temperate_ocean_planets"] >= 1 and x["pick_agrees_with_bridge"]
    assert w3.pick_record(x)["planet_index"] == 1


# ------------------------------------------------------------------------------------------------ the set-up
def test_a_world_depends_on_its_seed_alone():
    """Review M9: terrain, patches and founders of a seed come from its own generator, whatever else is sampled."""
    one = w3.World3(small(source="earth"), [0])
    two = w3.World3(small(source="earth"), [5, 0])
    assert two.setup_seeds[1] == one.setup_seeds[0] == w3.setup_seed(0)
    assert torch.equal(one.terrain0["elevation_m"][0], two.terrain0["elevation_m"][1])
    assert one.arena_cell.tolist() == two.arena_cell.tolist()[2:]
    for name in ("size_kg", "air_l_kg", "o2_carrier", "k", "Wh"):
        assert torch.equal(one.b.genome[name], two.b.genome[name][2:]), name
    assert torch.equal(one.b.pos, two.b.pos[2:])
    assert w3.choose_patches(small(source="earth"), [5, 0])["cells"][1] == one.arena_cell.tolist()


def test_the_set_up_holds_the_bodies_once(monkeypatch):
    """Review H8: the founders go into the state allocated once on the device; no state_dict / from_state copy."""
    def forbidden(*a, **k):
        raise AssertionError("a copy of the whole body state")

    monkeypatch.setattr(B.Bodies, "state_dict", forbidden)
    monkeypatch.setattr(B.Bodies, "from_state", classmethod(lambda cls, *a, **k: forbidden()))
    w = w3.World3(small(source="earth"), [0])
    assert w.b.alive.sum(1).tolist() == [32, 32]


def test_a_landless_world_gets_sea_patches():
    """Review M11: world7 seed 10 is water-covered; its patches are sea and its founders are placed there."""
    w = w3.World3(small(), [10])
    assert w.no_land == [True] and w.A == 2 and bool(w.geom.sea.all())
    assert w.b.alive.sum(1).tolist() == [32, 32]
    w.step_day()
    row = w.outcomes()[0]
    assert row["no_land"] and row["kind"] in ("alive", "extinct")


def test_the_spin_up_replenishment_is_reported():
    """Review L12: the carbon and gas the global biosphere's held-air spin-up made or removed are reported."""
    w = w3.World3(small(source="earth"), [0])
    rep = w.reports["spin_up_replenishment"][0]
    assert set(rep) >= {"carbon_made_kg", "sown_carbon_kg", "air_mol_before", "air_mol_after"}
    assert math.isfinite(rep["carbon_made_kg"]) and rep["air_mol_before"]["N2"] > 0
    w.step_day()
    assert w.outcomes()[0]["spin_up_replenishment"] == rep


# ------------------------------------------------------------------------------------------------ state and rules
def test_a_load_holds_the_state_once_and_a_checkpoint_carries_its_rules(tmp_path, monkeypatch):
    """Review H7 and L11: World3.load takes the loaded tensors as they are (no second or third copy); the checkpoint
    carries the rules hash and a load that expects other rules is refused; the state version is new."""
    w = w3.World3(small(source="earth"), [0])
    w.step_day()
    path = tmp_path / "ck.pt"
    w.save(path)
    seen = {}
    real = torch.load

    def load(*a, **k):
        seen["d"] = real(*a, **k)
        return seen["d"]

    monkeypatch.setattr(torch, "load", load)
    back = w3.World3.load(path)
    assert back.b.genome["Wh"].data_ptr() == seen["d"]["bodies"]["genome"]["Wh"].data_ptr()
    assert back.b.wh_live.data_ptr() == seen["d"]["bodies"]["wh_live"].data_ptr()
    assert back.state_hash() == w.state_hash()
    assert seen["d"]["rules_sha256"] == w3.rules_identity(w.config)["sha256"]
    assert w3.STATE_VERSION == "life9-v3-world3-2"
    with pytest.raises(ValueError, match="new experiment"):
        w3.World3.load(path, expect_rules="0" * 64)
    # a copy that owns its tensors still copies by default
    other = w3.World3.from_state(w.state_dict())
    assert other.b.genome["Wh"].data_ptr() != w.b.genome["Wh"].data_ptr()


def test_the_rules_hash_covers_imports_private_constants_and_the_earth_reference(monkeypatch):
    """Review M5 and M6: the formula parser, level 3's constants, life9's brain and the chain path are hashed by
    source; private constants are in the values; the Earth reference that calibrates formation3 is code, hashed."""
    files = w3.rule_sources()
    for rel in ("haishool/truth/formula.py", "haishool/cosmos/planets.py", "haishool/life9/brain.py",
                "haishool/evo/world7.py", "haishool/life8/bridge.py"):
        assert rel in files, rel
    cfg = small(source="earth")
    base = w3.rules_identity(cfg)["sha256"]
    monkeypatch.setattr(B, "_FAT_ATOMS", dict(B._FAT_ATOMS, C=52))
    assert w3.rules_identity(cfg)["sha256"] != base
    monkeypatch.undo()
    monkeypatch.setattr(F3, "_P_O2_EARTH", F3._P_O2_EARTH * 1.01)
    assert w3.rules_identity(cfg)["sha256"] != base
    monkeypatch.undo()
    real = ch.earth_inputs_computed

    def shifted():
        d = real()
        d["cloud"]["nitrogen"] *= 1.5
        return d

    monkeypatch.setattr(ch, "earth_inputs_computed", shifted)
    assert w3.rules_identity(cfg)["sha256"] != base
    monkeypatch.undo()
    assert w3.rules_identity(cfg)["sha256"] == base
    # the Earth reference is computed, not read from a per-host cache file
    assert ch.earth_inputs(cache_dir="/nonexistent") == ch.earth_inputs_computed()


# ------------------------------------------------------------------------------------------------ the command line
def test_the_starting_conditions_are_pinned_and_checked_on_resume(tmp_path, monkeypatch):
    """Review M7: DIR/inputs.json pins every seed's inputs before the first step; a resume with other inputs is
    refused."""
    out = tmp_path / "p"
    cfg = small(source="earth")
    run_cli(["run", "--sample", "earth:1", "--days", "2", "--out", str(out), "--chunk-days", "1"] + sets(cfg))
    pinned = json.loads((out / "inputs.json").read_text())
    assert pinned["seeds"][0]["inputs_sha256"] == w3.inputs_digest(ch.earth_inputs())
    real = w3.planet_inputs

    def other(seed, config):
        x, note = real(seed, config)
        x = json.loads(json.dumps(x))
        x["star"]["mass_msun"] *= 1.01
        return x, note

    monkeypatch.setattr(w3, "planet_inputs", other)
    with pytest.raises(SystemExit, match="starting conditions"):
        run_cli(["run", "--resume", "--out", str(out)])
    monkeypatch.undo()
    run_cli(["run", "--resume", "--out", str(out)])
    runs = [json.loads(x) for x in (out / "runs.jsonl").read_text().splitlines()]
    assert [r["mode"] for r in runs] == ["fresh", "resume"] and runs[-1]["day_to"] == 2
    done = json.loads((out / "outcomes.json").read_text())
    assert done["complete"] and done["errors"] == [] and done["day_base"] == "days_completed"
    row = done["outcomes"][0]
    for key in ("arena_extinct_day", "founding", "arena_capacity_bound_day", "items_bound_day",
                "arena_items_bound_day", "fires_bound_day", "nonfinite", "spin_up_replenishment", "no_land"):
        assert key in row, key
    rep = json.loads((out / "reports.json").read_text())
    assert rep["day"] == 2 and "extinct_day" in rep["reports"] and "founding" in rep["reports"]
    # a resume of the complete experiment runs, saves and rewrites nothing
    stamp = ((out / "checkpoint.pt").stat().st_mtime_ns, (out / "outcomes.json").read_text())
    text = run_cli(["run", "--resume", "--out", str(out)])
    assert "complete" in text
    assert ((out / "checkpoint.pt").stat().st_mtime_ns, (out / "outcomes.json").read_text()) == stamp


def test_errors_are_recorded_and_the_experiment_is_incomplete(tmp_path, monkeypatch):
    """Review M8: a failing day is written to DIR/errors.json and runs.jsonl, outcomes.json says incomplete."""
    out = tmp_path / "e"
    real = w3.World3.step_day

    def step(self, *a, **k):
        if self.day >= 1:
            raise RuntimeError("a lost host")
        return real(self, *a, **k)

    monkeypatch.setattr(w3.World3, "step_day", step)
    with pytest.raises(RuntimeError, match="lost host"):
        run_cli(["run", "--sample", "earth:1", "--days", "2", "--out", str(out)] + sets(small(source="earth")))
    errors = json.loads((out / "errors.json").read_text())
    assert errors[0]["stage"] == "day" and errors[0]["day"] == 1 and "lost host" in errors[0]["cause"]
    assert errors[0]["traceback"]
    runs = [json.loads(x) for x in (out / "runs.jsonl").read_text().splitlines()]
    assert "lost host" in runs[-1]["error"]
    done = json.loads((out / "outcomes.json").read_text())
    assert not done["complete"] and done["errors"]


def test_a_resume_on_another_host_must_be_declared(tmp_path, monkeypatch):
    out = tmp_path / "d"
    cfg = small(source="earth")
    run_cli(["run", "--sample", "earth:1", "--days", "2", "--out", str(out), "--chunk-days", "1"] + sets(cfg))
    real = cli.environment
    monkeypatch.setattr(cli, "environment", lambda device: dict(real(device), host="another-host"))
    with pytest.raises(SystemExit, match="allow-device-change"):
        run_cli(["run", "--resume", "--out", str(out)])
    run_cli(["run", "--resume", "--out", str(out), "--allow-device-change", "the first host was lost"])
    runs = [json.loads(x) for x in (out / "runs.jsonl").read_text().splitlines()]
    assert runs[-1]["device_change"]["changed"]["host"][1] == "another-host"


def test_resume_checks_the_source_and_refused_calls_leave_no_directory(tmp_path):
    """Review L14 and L19: a resume with a sample compares the source too (and takes it from the manifest when not
    given); a refused call creates no DIR; rng_seed redraws the worlds, so only an explicit experiment may set it;
    a synthetic sample is flagged as habitable by construction."""
    out = tmp_path / "s"
    cfg = small(source="earth")
    run_cli(["run", "--sample", "earth:1", "--days", "1", "--out", str(out)] + sets(cfg))
    with pytest.raises(SystemExit):
        run_cli(["run", "--resume", "--out", str(out), "--sample", "earth:1", "--source", "synthetic"])
    run_cli(["run", "--resume", "--out", str(out), "--sample", "earth:1"])          # the manifest's source
    nothing = tmp_path / "never"
    with pytest.raises(SystemExit, match="explicit seed list"):
        run_cli(["run", "--seeds", "781", "--source", "world7", "--days", "1", "--out", str(nothing)])
    assert not nothing.exists()
    with pytest.raises(SystemExit, match="rng_seed"):
        run_cli(["run", "--sample", "earth:1", "--days", "1", "--out", str(nothing), "--set", "rng_seed=7"]
                + sets(cfg))
    assert not nothing.exists()
    s = cli.sampling_of(cli.build_parser().parse_args(["run", "--out", "x", "--sample", "range:0:3", "--source",
                                                        "synthetic"]))
    assert s["habitability_conditioned"] and "by construction" in s["note"]


# ------------------------------------------------------------------------------------------------ what is reported
def test_the_item_pool_bound_is_reported_beside_the_slots():
    """Review M13: carcass parts without an item slot mark the arena's items_bound_day."""
    w = w3.World3(small(source="earth", items=2), [0])
    w.b.water_kg = torch.where(w.b.alive, torch.zeros_like(w.b.water_kg), w.b.water_kg)    # all die in the first bout
    B.reset_ledgers(w.b)
    w._rebase()
    w.step_day()
    assert w.reports["arena_items_bound_day"] == [1, 1] and w.reports["items_bound_day"] == 1
    row = w.outcomes()[0]
    assert row["items_bound_day"] == 1 and row["arena_items_bound_day"] == [1, 1]
    assert w.summary()["items_bound_day"] == 1
    assert "BIND" in w3.PROVENANCE["capacity"][1] and "pools_bound" in w3.PROVENANCE


def test_every_gene_is_reported_with_its_shadow():
    """Review M15: every heritable gene but the network weights is in the summary's gene statistics."""
    genes = set(B.GENE_SPECS) - set(b3.WEIGHT_GENES)
    assert genes <= set(w3.GENE_REPORT) and "k" in w3.GENE_REPORT
    w = w3.World3(small(source="earth"), [0])
    w.step_day()
    s = w.summary()
    assert {"air_l_kg", "o2_carrier"} <= set(s["genes"]) and {"air_l_kg", "o2_carrier"} <= set(s["shadow_genes"])
    # the O2 store per arena, split into the lung's and the carrier's (review H2, H5)
    st = s["o2_store"]
    assert set(st) == {"lung_mol_per_kg", "carrier_mol_per_kg", "lung_share"} and len(st["lung_share"]) == 2
    world = s["worlds"][0]
    assert "x_o2_dry" in world and "x_o2" not in world and "fire_beyond_fit" in world        # review L20, M17


def test_fire_is_uncertain_outside_its_calibrated_pressures():
    """Review M17 and L1: the 1-bar flammability limit is applied by the fires at any pressure; outside 0.5-2 bar a
    world's fire_possible is None (uncertain) with fire_beyond_fit, not True."""
    assert F3.fire_possible(0.21, 101325.0) == {"fire_possible": True, "fire_beyond_fit": False}
    assert F3.fire_possible(0.10, 101325.0) == {"fire_possible": False, "fire_beyond_fit": False}
    assert F3.fire_possible(0.157, 24400.0) == {"fire_possible": None, "fire_beyond_fit": True}
    assert F3.fire_possible(0.10, 24400.0) == {"fire_possible": False, "fire_beyond_fit": True}
    row = F3.summary_row(F3.build_planet3(ch.chain_inputs(66), 66))
    assert row["fire_beyond_fit"] and row["fire_possible"] is None and row["x_o2_dry"] >= 0.15


def test_the_climate_carries_formation3s_moles():
    """Review M3: the climate of formation3's specs is a well-mixed column: its N2 is formation3's calibrated air
    nitrogen, and its partial pressures at the start are formation3's."""
    w = w3.World3(small(source="earth", bio_spin_days=0, veg_spin_days=1, climate_fast_chunks=0,
                        climate_slow_chunks=0), [0])
    spec = w.specs[0]
    area = 4 * math.pi * spec.radius_m ** 2
    gas0 = w.P["gas0"][0]
    assert float(gas0[cl.I_N2]) * K.molar_mass("N2") * area == pytest.approx(spec.air_nitrogen_kg, rel=1e-9)
    assert float(gas0[cl.I_AR]) * K.molar_mass("Ar") * area == pytest.approx(spec.air_argon_kg, rel=1e-9)
    p = cl.gas_pressures(gas0[None], w.P)[0]
    for i, gas in enumerate(cl.GASES):
        assert float(p[i]) == pytest.approx(spec.partial_pressure_pa[gas], rel=1e-9), gas
    assert w.P["well_mixed"]


def test_the_hydrostatic_surface_pressure_is_reported():
    """Review L5: the column's weight (dry pressure plus the vapour column x g) beside the sum of the partials."""
    e = F3.model_earth()
    dry = sum(e.partial_pressure_pa[g] for g in ("N2", "O2", "CO2", "Ar"))
    assert e.surface_pressure_hydrostatic_pa == pytest.approx(dry + e.vapour_column_kg_m2 * e.gravity_m_s2, rel=1e-12)
    assert e.surface_pressure_hydrostatic_pa < e.surface_pressure_pa
    assert 1.00e5 < e.surface_pressure_hydrostatic_pa < 1.03e5


def test_the_unconsciousness_gate_uses_the_last_bouts_o2_rate():
    """Review L7: the gate's O2 use is at least the previous bout's metabolic O2 rate (thermogenesis included)."""
    w = w3.World3(small(source="earth"), [0])
    w.step_day()
    b = w.b
    tis = B.tissues(b)
    env = B.constant_env(t_air_k=290.0)
    terms = w3.locomotion_terms(b, torch.zeros_like(b.mass_kg), env)
    w.last["o2_mol_s"] = torch.full_like(b.mass_kg, 1.0)
    use = w._o2_use(b, tis, terms)
    assert torch.allclose(use[b.alive], torch.ones_like(use[b.alive]))
    w.last["o2_mol_s"] = torch.zeros_like(b.mass_kg)
    rest = B.resting_power_w(b, tis) / B.OXY_J_PER_MOL_O2
    assert torch.allclose(w._o2_use(b, tis, terms)[b.alive], rest[b.alive])


def test_a_body_without_an_o2_store_is_awake_while_it_breathes(monkeypatch):
    """The unconsciousness gate needs a debt: a body with no store at all (no air, no carrier) on land breathes, moves
    and eats; only a dive (a debt) makes it black out at once."""
    w = w3.World3(small(source="earth"), [0])
    monkeypatch.setattr(B, "o2_store_mol", lambda b, *a, **k: torch.zeros_like(b.mass_kg))
    w.step_day()
    st = w.stats_day
    assert float(st["distance_m"].sum()) > 0 and float(st["mouth_sum"].sum()) > 0
