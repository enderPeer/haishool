"""The long-run driver continues a search bundle, measures each segment with paired branches,
reseeds every random stream (and the genome seed) only for futures >= 1, splits runs exactly,
and resumes without repeating work."""
import json

from haishool.life8 import search
from haishool.life8.analysis import longrun


def _bundle(tmp_path):
    entry = {"id": "plain-5", "source": "plain", "seed": 5,
             "config": {**search.resolve_options("living", {"population_limit": 256}), "log": "compact"}}
    _, bundle = search.run_segment(entry, 40, perm_reps=5, seed=1)
    path = tmp_path / "start.json.gz"
    search._write_gz(path, bundle)
    return path, bundle


def _plain(row):
    return json.dumps({k: v for k, v in row.items() if k != "seconds"}, sort_keys=True)


def test_run_measures_every_segment_and_resumes(tmp_path):
    path, start = _bundle(tmp_path)
    out = tmp_path / "f0"
    kwargs = dict(future=0, horizon=start["tick"] + 60, every=30, reps=2, branch_ticks=15, workers=1, perm_reps=5,
                  log=lambda text: None)
    rows = longrun.run(path, out, **kwargs)
    assert [r["segment"] for r in rows] == [1, 2]
    assert rows[-1]["end_tick"] == start["tick"] + 60
    for row in rows:
        assert row["paired"] is not None and row["branch_reps"] == 2
        assert {b["arm"] for b in row["branches"]} == set(longrun.ARMS)
        assert "language_verdict" in row["criteria"] and row["criteria"]["tool_benefit"] is None
        cal = row["calibrated"]
        assert cal["verdict"] in ("all_three", "partial", "null")
        assert cal["fitness_normal_minus_scrambled"]["n"] == 2
    again = longrun.run(path, out, **kwargs)
    assert again == rows  # nothing repeated
    summary = longrun.summarise(rows)
    assert len(summary["points"]) == 2 and summary["emerged"] == []


def test_lost_row_is_resimulated_identically(tmp_path):
    path, start = _bundle(tmp_path)
    kwargs = dict(future=0, horizon=start["tick"] + 40, every=20, reps=1, branch_ticks=10, workers=1, perm_reps=5,
                  log=lambda text: None)
    whole = longrun.run(path, tmp_path / "a", **kwargs)
    longrun.run(path, tmp_path / "b", **kwargs)
    # a kill after the state but before the row of segment 2: both are re-done from segment 1
    (tmp_path / "b" / "seg-002.row.json").unlink()
    lines = (tmp_path / "b" / "rows.jsonl").read_text(encoding="utf-8").splitlines()
    (tmp_path / "b" / "rows.jsonl").write_text(lines[0] + "\n", encoding="utf-8")
    again = longrun.run(path, tmp_path / "b", **kwargs)
    assert [_plain({k: v for k, v in r.items() if k != "branches"}) for r in again] == \
        [_plain({k: v for k, v in r.items() if k != "branches"}) for r in whole]


def test_futures_reseed_everything_and_future_zero_is_exact(tmp_path):
    path, start = _bundle(tmp_path)
    assert longrun.reseed(start, 0) is start
    one, two = longrun.reseed(start, 1), longrun.reseed(start, 2)
    assert one["world"]["agents"] == start["world"]["agents"]
    for key in ("rng_state", "channel_rng_state", "seed"):
        assert one["world"][key] != start["world"][key] and one["world"][key] != two["world"][key]
    # splitting a run into segments with a gzip round trip changes nothing
    entry = longrun.entry_of(start)
    straight, end_a = search.run_segment(entry, 40, start=start, perm_reps=5, seed=3)
    _, middle = search.run_segment(entry, 20, start=start, perm_reps=5, seed=3)
    search._write_gz(tmp_path / "mid.json.gz", middle)
    _, end_b = search.run_segment(entry, 20, start=search._read_gz(tmp_path / "mid.json.gz"), perm_reps=5, seed=3)
    assert end_a["world"] == end_b["world"] and end_a["tracker"] == end_b["tracker"]
    assert straight["end_tick"] == end_b["tick"]


def test_summarise_needs_adjacent_points_and_a_failing_baseline():
    def row(segment, verdict):
        return {"future": 1, "segment": segment, "end_tick": 7500 + 5000 * segment, "population": 90,
                "measures": {"sender": {}}, "criteria": {}, "paired": {}, "calibrated": {"verdict": verdict}}
    gap = [row(1, "all_three"), row(3, "all_three")]
    assert longrun.summarise(gap)["emerged"] == []
    pair = [row(1, "all_three"), row(2, "all_three")]
    assert longrun.summarise(pair)["emerged"] == [{"future": 1, "tick": 17500}]
    assert longrun.summarise(pair, {"calibrated": {"verdict": "all_three"}})["emerged"] == []
