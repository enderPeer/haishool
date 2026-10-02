"""Spatial grid candidate superset and compact event logging (hash test: test_spatial_hash.py)."""
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

from haishool.life8.checkpoint import digest, load
from haishool.life8.cli import run_world
from haishool.life8.config import Config
from haishool.life8.dataset import export_transitions
from haishool.life8.entities import FoodPatch
from haishool.life8.learning import ACTIONS
from haishool.life8.spatial import GridIndex
from haishool.life8.world import RECEIVER_BANDS, SENDER_BANDS, World

ROOT = Path(__file__).resolve().parents[2]


def test_grid_candidates_include_everything_within_reach_across_the_wrap():
    rng = random.Random(5)
    width, height = 32., 24.
    records = {i: FoodPatch(i, [rng.random() * width, rng.random() * height], 1., 1.) for i in range(1, 400)}
    for item in list(records.values())[:40]:  # entities on cell boundaries and the world edge
        item.position = [float(rng.randrange(16)) * 2., rng.choice([0., 2., 23.999999999999996])]
    probe = World.create(1, Config(population=1, food_patches=0, initial_objects=0))
    index = GridIndex(width, height, 2., (records,))
    for _ in range(300):
        centre = [rng.choice([0., 31.9999, rng.random() * width]), rng.choice([0., rng.random() * height])]
        radius = rng.choice([1.5, 2., 5., 8.75, 9.5, 20.])
        near = {i for i, item in records.items() if probe.distance(centre, item.position) <= radius}
        assert near <= {item.id for item in index.candidates(records, centre, radius)}


def strip_log(state):
    state = json.loads(json.dumps(state))
    state["config"].pop("log")
    state.pop("pending_heard")
    return state


def test_compact_log_changes_records_not_dynamics():
    full = World.create(seed=6, config=Config())
    compact = World.create(seed=6, config=Config(log="compact"))
    for _ in range(120):
        full.step()
        compact.step()
    assert strip_log(full.to_dict()) == strip_log(compact.to_dict())


def test_compact_records_signals_with_sender_state_and_heard_with_next_action():
    world = World.create(seed=2, config=Config(log="compact"))
    events = []
    for _ in range(80):
        events += world.step()
    kinds = {event["type"] for event in events}
    assert kinds <= {"birth", "death", "signal", "heard", "act", "capacity_pause", "thermal_contact"}
    assert {"signal", "heard", "act", "birth"} <= kinds
    assert not any("observation" in event or "next_observation" in event for event in events)
    signals = {event["message_id"]: event for event in events if event["type"] == "signal"}
    for event in signals.values():
        assert set(event["sender"]) == set(SENDER_BANDS) and event["symbol"] in range(4)
    heard = [event for event in events if event["type"] == "heard"]
    assert len(heard) == world.counters["signals_received"] - len(world.pending_heard)
    for event in heard:
        assert event["next_action"] in ACTIONS and set(event["receiver"]) == set(RECEIVER_BANDS)
        assert event["agent"] != event["sender"] and 0 <= event["age"] <= world.config.message_ttl
        assert (event["symbol"] != event["sent_symbol"]) == event["noisy"]
        if event["message_id"] in signals:
            assert signals[event["message_id"]]["symbol"] == event["sent_symbol"]
    acts = [event for event in events if event["type"] == "act"]
    assert {event["action"] for event in acts} <= {"collect", "drop", "strike", "strike_agent", "strike_object",
                                                    "combine", "dismantle", "heat", "share", "imitate", "teach"}


def test_compact_checkpoint_continues_exactly_with_pending_heard_records():
    world = World.create(seed=3, config=Config(log="compact"))
    for _ in range(40):
        world.step()
    assert world.pending_heard
    resumed = World.from_dict(json.loads(json.dumps(world.to_dict(), sort_keys=True)))
    for _ in range(25):
        assert world.step() == resumed.step()
    assert digest(world.to_dict()) == digest(resumed.to_dict())


def test_compact_archive_is_small_resumable_and_not_exportable(tmp_path):
    def make():
        return World.create(seed=7, config=Config(population=12, log="compact"))
    direct = make()
    for _ in range(30):
        direct.step()
    first = run_world(make(), tmp_path / "a", steps=15, sample_every=5)
    resumed, _ = load(tmp_path / "a" / "checkpoint.json")
    second = run_world(resumed, tmp_path / "b", steps=15, sample_every=5)
    assert digest(load(tmp_path / "b" / "checkpoint.json")[0].to_dict()) == digest(direct.to_dict())
    assert first["bytes_written"]["events.jsonl"] == (tmp_path / "a" / "events.jsonl").stat().st_size
    assert second["config"]["log"] == "compact"
    with pytest.raises(ValueError, match="compact"):
        export_transitions(tmp_path / "a", tmp_path / "x.jsonl")


def test_module_cli_runs_a_compact_world(tmp_path):
    out = tmp_path / "cli"
    result = subprocess.run([sys.executable, "-m", "haishool.life8", "run", "--seed", "5", "--population", "6",
                             "--steps", "8", "--sample-every", "4", "--log", "compact", "--out", str(out)],
                            cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete" and manifest["config"]["log"] == "compact"
    assert manifest["schema"] == "life8-run-v2" and manifest["final_tick"] == 8
