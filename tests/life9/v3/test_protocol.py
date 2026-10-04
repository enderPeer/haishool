"""The experiment protocol of life9 v3 (world3.PROVENANCE['protocol'], the owner's five rules of 4 Oct 2026):

1. every outcome is kept: a seed without a star, a planet, a habitable planet, a bodies era or land is run and
   recorded, an extinct world is a result;
2. the sample is a rule fixed before any outcome is known (range, hash, earth replicates), an explicit seed list only
   as a declared separate experiment, and every sampled seed appears in the outcomes;
3. carrying state between stages is a documented gap (the manifest says so);
4. no automatic rescue: no fallback path exists (source text and behaviour), patches are placed without the climate,
   an extinct arena is never re-founded;
5. the rules are fixed: the manifest (written before the worlds are built) holds the rules hash, which changes with
   any rule constant, and a resume under another hash is refused."""
from __future__ import annotations

import ast
import contextlib
import datetime
import inspect
import io
import json
import shutil
from pathlib import Path

import pytest
import torch

from haishool.life9.planet import chain as ch
from haishool.life9.planet import crafting as cr
from haishool.life9.v3 import __main__ as cli
from haishool.life9.v3 import body as B
from haishool.life9.v3 import world3 as w3

PKG = Path(w3.__file__).resolve().parent


def small(**kw) -> w3.V3Config:
    cfg = dict(G=8, patches=2, patch_m=256.0, cells=16, capacity=96, founders=32, items=128, fires=8, hidden=16,
               k_founder=(2, 8), bouts=4, climate_fast_chunks=1, climate_slow_chunks=1, bio_spin_days=20,
               bio_spin_segments=1, veg_spin_days=20, veg_spin_bouts=2)
    cfg.update(kw)
    return w3.V3Config(**cfg)


def sets(cfg: w3.V3Config) -> list:
    out = []
    for k, v in cfg.to_dict().items():
        if k != "source":
            out += ["--set", f"{k}={','.join(str(x) for x in v) if isinstance(v, list) else v}"]
    return out


def run_cli(argv) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert cli.main(argv) == 0
    return buf.getvalue()


@pytest.fixture
def no_substitute(monkeypatch):
    """Any call that would put another planet in a world7 seed's place fails the test: synthetic inputs at all, Earth's
    inputs as a world's inputs (formation3 still reads them for its Earth calibration)."""
    real_earth = ch.earth_inputs

    def forbidden(*a, **k):
        raise AssertionError("a synthetic planet was substituted")

    def earth(*a, **k):
        if any(f.function in ("planet_inputs", "choose_patches", "__init__") and f.filename == w3.__file__
               for f in inspect.stack()[1:4]):
            raise AssertionError("the Earth was substituted for a world7 seed")
        return real_earth(*a, **k)

    monkeypatch.setattr(ch, "synthetic_inputs", forbidden)
    monkeypatch.setattr(ch, "earth_inputs", earth)


# ------------------------------------------------------------------------------------------------ rule 4: no rescue
RESCUE_WORDS = ("fallback", "replenish", "rescue", "refill", "refound", "reseed", "resow", "substitut", "resample",
                "redraw", "retry")


def _identifiers(tree) -> set:
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            out.add(node.id)
        elif isinstance(node, ast.Attribute):
            out.add(node.attr)
        elif isinstance(node, ast.arg):
            out.add(node.arg)
        elif isinstance(node, ast.keyword) and node.arg:
            out.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            out.add(node.name)
    return out


def test_no_fallback_path_exists_in_the_source():
    """No code name in the v3 package is a rescue mechanism, the config has no rescue switch, and the run path calls
    the synthetic and Earth inputs only under the source the experiment chose."""
    hits = []
    for path in sorted(PKG.glob("*.py")):
        names = _identifiers(ast.parse(path.read_text(encoding="utf-8")))
        hits += [f"{path.name}: {n}" for n in sorted(names) if any(w in n.lower() for w in RESCUE_WORDS)]
    assert not hits, hits
    assert not any(any(w in k for w in RESCUE_WORDS) for k in w3.V3Config().to_dict())
    # the only calls of synthetic_inputs / earth_inputs in world3 sit under `config.source == "<that source>"`
    tree = ast.parse((PKG / "world3.py").read_text(encoding="utf-8"))
    calls = {"synthetic_inputs": "synthetic", "earth_inputs": "earth"}
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            test = ast.unparse(node.test)
            for sub in ast.walk(ast.Module(body=node.body, type_ignores=[])):
                if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute) and sub.func.attr in calls:
                    assert test == f"config.source == '{calls[sub.func.attr]}'", test
                    found.append(sub.func.attr)
    every = [n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr in calls]
    assert sorted(found) == sorted(every) == ["earth_inputs", "synthetic_inputs"]


def test_a_seed_without_a_habitable_planet_is_recorded_not_replaced(no_substitute):
    """World7 seeds 0 (no star), 2 (no temperate rocky planet) and 3 (a planet whose chain made no bodies) give no
    world: each is recorded with its outcome, the days still pass, nothing takes their place."""
    w = w3.World3(small(), [0, 2, 3])
    assert w.W == 0 and w.A == 0 and w.sampled == [0, 2, 3]
    assert [r["kind"] for r in w.absent] == ["no_star", "no_habitable_planet", "no_bodies_era"]
    w.run(2)
    rows = w.outcomes()
    assert [(r["seed"], r["kind"]) for r in rows] == [(0, "no_star"), (2, "no_habitable_planet"),
                                                      (3, "no_bodies_era")]
    # 'no bodies era' is an exclusion by the chain's outcome, labelled and counted apart (an open owner decision)
    assert [r["excluded"] for r in rows] == [False, False, True] and "excluded" in rows[2]["outcome"]
    assert w.reports["excluded"] == [3]
    assert all(r["day"] == 2 and r["world"] is None and r["note"] for r in rows)
    s = w.summary()
    assert [r["seed"] for r in s["outcomes"]] == [0, 2, 3] and [r["seed"] for r in s["absent"]] == [0, 2, 3]
    assert w.ledgers()["worst"] == 0.0
    back = w3.World3.from_state(w.state_dict())
    assert back.outcomes() == rows and back.state_hash() == w.state_hash()
    assert "no star" in w.describe()


def test_a_seed_without_a_planet_next_to_one_with_a_planet(no_substitute):
    """In one experiment the starless seed is recorded and 781 is built and run beside it; the set-up generator comes
    from the whole sampled list."""
    w = w3.World3(small(), [0, 781])
    assert w.seeds == [781] and w.sampled == [0, 781] and w.rng_seed == w3.derived_seed([0, 781])
    assert w.input_notes == ["world7"] and w.inputs[0]["source"] == "world7" and w.inputs[0]["seed"] == 781
    w.step_day()
    rows = w.outcomes()
    assert [r["seed"] for r in rows] == [0, 781]
    assert rows[0]["kind"] == "no_star" and rows[1]["world"] == 0 and rows[1]["arenas"] == 2
    assert rows[1]["kind"] in ("alive", "extinct") and rows[1]["inputs_sha256"] == w3.inputs_digest(w.inputs[0])


def test_the_chain_outcome_is_cached_for_hosts_without_world7(tmp_path, monkeypatch):
    first = w3.chain_outcome(0, cache_dir=tmp_path)
    assert (tmp_path / "absent-world7-0.json").exists() and first["kind"] == "no_star"

    def no_world7(seed):
        raise ImportError("no scipy here")

    monkeypatch.setattr(ch, "_scan", no_world7)
    assert w3.chain_outcome(0, cache_dir=tmp_path) == first
    with pytest.raises(ImportError):                       # without the cache the outcome is unknown: no guess
        w3.chain_outcome(0, cache_dir=tmp_path / "empty")


def test_patches_are_placed_without_the_climate():
    """The patch cells come from the terrain's land by area alone (draw 2), before any climate or biology: World3
    gets the cells choose_patches draws with no spin-up at all, whatever the spin-ups."""
    a = w3.World3(small(source="earth"), [0, 1])
    b = w3.World3(small(source="earth", bio_spin_days=0, veg_spin_days=3, climate_fast_chunks=0,
                        climate_slow_chunks=0), [0, 1])
    got = w3.choose_patches(small(source="earth"), [0, 1])
    cells = [c for row in got["cells"] for c in row]
    assert a.arena_cell.tolist() == b.arena_cell.tolist() == cells
    land = a.terrain0["land"]
    assert all(bool(land[int(wd), int(c)]) for wd, c in zip(a.arena_world, a.arena_cell))


def test_an_extinct_world_stays_extinct_and_is_recorded():
    """Rule 1 and 4: every body dies by physics (no water), the arenas are not re-founded or refilled, and the world's
    outcome is 'extinct at day 0'; the books still close."""
    w = w3.World3(small(source="earth"), [0])
    w.b.water_kg = torch.where(w.b.alive, torch.zeros_like(w.b.water_kg), w.b.water_kg)
    B.reset_ledgers(w.b)                                   # the books start from the dry bodies
    w._rebase()
    w.step_day()
    assert int(w.b.alive.sum()) == 0 and w.reports["extinct_day"] == [1, 1]          # days completed
    w.run(2)
    assert int(w.b.alive.sum()) == 0 and float(w.b.ledger["births"].sum()) == 0
    row = w.outcomes()[0]
    assert row["kind"] == "extinct" and row["outcome"] == "extinct at day 1" and row["population"] == 0
    assert row["extinct_day"] == 1 and row["arenas_alive"] == 0 and row["day"] == 3
    assert row["arena_extinct_day"] == [1, 1] and row["day_base"] == "days_completed"
    assert row["founding"]["arenas_extinct_first_bout"] == 2 and row["founding"]["dead_first_bout_share"] == [1.0, 1.0]
    led = w.ledgers()
    bad = {k: v["rel"] for k, v in led.items() if isinstance(v, dict) and not v["rel"] < 1e-5}
    assert not bad, bad


# ------------------------------------------------------------------------------------------------ rule 2: sampling
def test_sampling_rules_are_fixed_before_any_outcome():
    r = cli.sample_seeds("range:0:63")
    assert r["seeds"] == list(range(64)) and r["pre_registered"] and r["source"] is None
    h = cli.sample_seeds("hash:round9:8:0:63")
    assert len(h["seeds"]) == len(set(h["seeds"])) == 8 and all(0 <= s <= 63 for s in h["seeds"])
    assert h["seeds"] == sorted(h["seeds"]) and cli.sample_seeds("hash:round9:8:0:63") == h
    assert cli.sample_seeds("hash:other:8:0:63")["seeds"] != h["seeds"]
    # a hash draw of N is the first N of the larger draw: adding seeds never changes the ones already drawn
    assert set(h["seeds"]) <= set(cli.sample_seeds("hash:round9:16:0:63")["seeds"])
    e = cli.sample_seeds("earth:3")
    assert e["seeds"] == [0, 1, 2] and e["source"] == "earth"
    for bad in ("range:5:2", "hash:x:70:0:63", "earth:0", "list:1,2", "range:a:b", "hash::2:0:9"):
        with pytest.raises(SystemExit):
            cli.sample_seeds(bad)


def _args(argv):
    return cli.build_parser().parse_args(["run", "--out", "unused"] + argv)


def test_explicit_seed_lists_and_sources_are_refused_unless_declared():
    with pytest.raises(SystemExit, match="explicit seed list"):
        cli.sampling_of(_args(["--seeds", "6", "7", "--source", "world7"]))
    s = cli.sampling_of(_args(["--seeds", "6", "7", "--source", "world7", "--explicit", "the two seeds of round 8"]))
    assert s["kind"] == "explicit" and not s["pre_registered"] and s["note"] and s["seeds"] == [6, 7]
    with pytest.raises(SystemExit, match="never substituted"):          # range/hash need the experiment's source
        cli.sampling_of(_args(["--sample", "range:0:63"]))
    with pytest.raises(SystemExit, match="source earth"):
        cli.sampling_of(_args(["--sample", "earth:2", "--source", "world7"]))
    with pytest.raises(SystemExit):
        cli.sampling_of(_args([]))
    p = cli.sampling_of(_args(["--sample", "range:0:63", "--source", "world7", "--part", "1:4"]))
    assert p["part"] == {"index": 1, "of": 4, "seeds": list(range(16, 32))}
    parts = [cli.sampling_of(_args(["--sample", "range:0:9", "--source", "world7", "--part", f"{k}:3"]))["part"]
             for k in range(3)]
    assert [s for q in parts for s in q["seeds"]] == list(range(10))


def test_every_sampled_seed_appears_in_the_outcomes(tmp_path, no_substitute):
    out = tmp_path / "x"
    run_cli(["run", "--sample", "range:0:3", "--source", "world7", "--days", "1", "--out", str(out)] + sets(small()))
    done = json.loads((out / "outcomes.json").read_text())
    man = json.loads((out / "manifest.json").read_text())
    assert [r["seed"] for r in done["outcomes"]] == man["sampling"]["seeds"] == [0, 1, 2, 3]
    assert all(r["outcome"] for r in done["outcomes"]) and done["complete"] and done["days_run"] == 1
    assert done["rules_sha256"] == man["rules"]["sha256"]
    last = json.loads((out / "summary.jsonl").read_text().splitlines()[-1])
    assert [r["seed"] for r in last["outcomes"]] == [0, 1, 2, 3]


# ------------------------------------------------------------------------------------------------ rule 5: fixed rules
def test_manifest_is_written_before_the_worlds_are_built_and_stepped(tmp_path, monkeypatch):
    out = tmp_path / "e"
    seen = []
    real_init, real_step = w3.World3.__init__, w3.World3.step_day

    def init(self, *a, **k):
        seen.append(("build", (out / "manifest.json").exists()))
        real_init(self, *a, **k)

    def step(self, *a, **k):
        seen.append(("step", (out / "manifest.json").exists()))
        real_step(self, *a, **k)

    monkeypatch.setattr(w3.World3, "__init__", init)
    monkeypatch.setattr(w3.World3, "step_day", step)
    cfg = small(source="earth")
    run_cli(["run", "--sample", "earth:1", "--days", "1", "--out", str(out)] + sets(cfg))
    assert seen[0] == ("build", True) and ("step", True) in seen and all(ok for _, ok in seen)
    man = json.loads((out / "manifest.json").read_text())
    assert man["protocol"] == w3.PROTOCOL_VERSION and man["sampling"]["rule"] == "earth:1"
    assert man["sampling"]["seeds"] == [0] and man["part"]["seeds"] == [0] and man["source"] == "earth"
    assert man["config"] == cfg.to_dict() and man["days"] == 1
    assert man["rules"]["sha256"] == w3.rules_identity(cfg)["sha256"]
    assert set(man["rules"]["files"]) >= {"haishool/life9/v3/world3.py", "haishool/life9/planet/climate.py"}
    assert man["environment"]["python"] and man["environment"]["torch"] == torch.__version__
    datetime.datetime.fromisoformat(man["started"])
    assert "deferred" in man["carry_state_between_stages"]
    # a directory that holds an experiment is never reused for another one
    with pytest.raises(SystemExit, match="already holds"):
        run_cli(["run", "--sample", "earth:1", "--days", "1", "--out", str(out)] + sets(cfg))


def test_rules_hash_changes_when_a_rule_changes(tmp_path, monkeypatch):
    cfg = small(source="earth")
    base = w3.rules_identity(cfg)
    assert w3.rules_identity(small(source="earth")) == base                   # deterministic
    assert w3.rules_identity(small(source="earth", capacity=97))["sha256"] != base["sha256"]   # the config
    monkeypatch.setattr(w3, "KT_COST_J_PER_N", w3.KT_COST_J_PER_N * 1.01)      # a v3 constant
    a = w3.rules_identity(cfg)
    assert a["sha256"] != base["sha256"] and a["sources_sha256"] == base["sources_sha256"]
    monkeypatch.undo()
    monkeypatch.setattr(cr, "X_O2_MIN", 0.16)                                  # a planet constant
    assert w3.rules_identity(cfg)["sha256"] != base["sha256"]
    monkeypatch.undo()
    spec = B.GENE_SPECS["size_kg"]                                             # a gene's founder range
    monkeypatch.setitem(B.GENE_SPECS, "size_kg", spec._replace(founder_hi=spec.founder_hi * 1.5))
    assert w3.rules_identity(cfg)["sha256"] != base["sha256"]
    monkeypatch.undo()
    assert w3.rules_identity(cfg) == base
    # the source: a copy of the two packages hashes the same; an edited constant (or line ending) is seen exactly
    root = tmp_path / "repo"
    for rel in w3.rule_sources():                                              # the packages and their imports
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(w3.REPO_ROOT / rel, root / rel)
    assert w3.rule_sources(root) == w3.rule_sources()
    f = root / "haishool/life9/v3/world3.py"
    text = f.read_text(encoding="utf-8")
    f.write_bytes(text.replace("\n", "\r\n").encode())                         # CRLF: the same rules
    assert w3.rule_sources(root) == w3.rule_sources()
    f.write_text(text.replace("KT_COST_J_PER_N = 0.2 ", "KT_COST_J_PER_N = 0.21"), encoding="utf-8")
    changed = {p for p, h in w3.rule_sources(root).items() if h != w3.rule_sources()[p]}
    assert changed == {"haishool/life9/v3/world3.py"}
    assert w3.rules_identity(cfg, root)["sha256"] != base["sha256"]


def test_resume_refuses_a_different_rules_hash(tmp_path, monkeypatch):
    out = tmp_path / "r"
    cfg = small(source="earth")
    argv = ["run", "--sample", "earth:1", "--days", "2", "--out", str(out)] + sets(cfg)
    run_cli(argv + ["--chunk-days", "1"])
    assert not (out / "outcomes.json").exists()
    monkeypatch.setattr(w3, "KT_COST_J_PER_N", w3.KT_COST_J_PER_N * 1.5)
    with pytest.raises(SystemExit, match="new experiment"):
        run_cli(["run", "--resume", "--out", str(out)])
    monkeypatch.undo()
    with pytest.raises(SystemExit, match="length"):                           # the length is the manifest's
        run_cli(["run", "--resume", "--out", str(out), "--days", "3"])
    with pytest.raises(SystemExit, match="config"):                           # so is the config
        run_cli(["run", "--resume", "--out", str(out), "--set", "capacity=128"])
    run_cli(["run", "--resume", "--out", str(out)])
    done = json.loads((out / "outcomes.json").read_text())
    assert done["complete"] and done["days_run"] == 2 and [r["seed"] for r in done["outcomes"]] == [0]
    days = [json.loads(x)["day"] for x in (out / "summary.jsonl").read_text().splitlines()]
    assert days == [1, 2]
    with pytest.raises(SystemExit, match="manifest"):
        run_cli(["run", "--resume", "--out", str(tmp_path / "nothing")])
