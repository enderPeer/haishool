"""Many-world search (haishool.life8.search): statistics, selection, files and resume."""
import gzip
import json
import random

import pytest

from haishool.life8 import discovery, lineage, search


def _rows(folder):
    return search.read_results(folder)


def test_pure_python_statistics_match_the_numpy_harness():
    rng = random.Random(3)
    xs = [rng.randrange(3) for _ in range(500)]
    ys = [(x + (rng.random() < .3)) % 4 for x in xs]
    assert search._mi(xs, ys) == pytest.approx(discovery.mutual_information(xs, ys), abs=1e-9)
    for diffs in ([1, 2, -1, 3, 0, 4], [-1] * 7, []):
        assert search.sign_test(diffs) == pytest.approx(discovery.sign_test(diffs))
    # the language cut-offs are discovery's
    for key, value in discovery.THRESHOLDS.items():
        assert search.CRITERIA[key] == value


def _signal(tick, sender, symbol, food, energy):
    return {"tick": tick, "type": "signal", "agent": sender, "symbol": symbol,
            "sender": {"food_direction": "e" if food else "none", "energy_band": energy, "food_near": False,
                       "neighbor_near": False}}


class _Table:
    def __init__(self, founder):
        self.rows, self._founder = {k: {} for k in founder}, founder

    def founder(self, identity):
        return self._founder[identity]


def test_convention_statistic_equals_the_lineage_harness():
    rng = random.Random(5)
    founder = {s: 100 + s % 3 for s in range(1, 13)}
    events = []
    for tick in range(600):
        sender = rng.randrange(1, 13)
        food, energy = rng.random() < .5, rng.choice(("low", "high"))
        lineage_symbol = (founder[sender] + food + 2 * (energy == "high")) % 4
        symbol = lineage_symbol if rng.random() < .8 else rng.randrange(4)
        events.append(_signal(tick, sender, symbol, food, energy))
    rows = [(e["tick"], e["agent"], e["sender"], e["symbol"]) for e in events]
    mine = search.convention_agreement(rows, founder, reps=50)
    theirs = lineage.convention_agreement(events, _Table(founder), reps=50)
    for key in ("within", "across", "within_blind", "across_blind", "difference", "difference_raw"):
        assert mine[key] == pytest.approx(theirs[key], abs=1e-4), key
    assert mine["difference"] > .3 and mine["p_value"] < .05


def test_inherited_symbol_habits_are_not_a_convention():
    """Critic F2: each lineage prefers its own symbol in every situation (inherited bias
    rows, no situation mapping). Raw within-lineage agreement is far above across, but the
    situation-specific difference is about 0 and the convention criterion fails."""
    rng = random.Random(6)
    founder = {s: 100 + s % 4 for s in range(1, 17)}
    events = []
    for tick in range(1600):
        sender = rng.randrange(1, 17)
        food, energy = rng.random() < .5, rng.choice(("low", "middle", "high"))
        symbol = founder[sender] % 4 if rng.random() < .7 else rng.randrange(4)
        events.append(_signal(tick, sender, symbol, food, energy))
    rows = [(e["tick"], e["agent"], e["sender"], e["symbol"]) for e in events]
    mine = search.convention_agreement(rows, founder, reps=100)
    theirs = lineage.convention_agreement(events, _Table(founder), reps=100)
    for result in (mine, theirs):
        assert result["difference_raw"] > .3  # the old statistic would call this a convention
        assert abs(result["difference"]) < .03 and result["p_value"] > .01
    row = _row(convention=mine)
    assert search.criteria(row)["convention"] is False and search.score(row)[1]["convention"] < .25


def test_within_sender_null_removes_individual_habits():
    # each sender always uses its own symbol and lives in its own energy band:
    # pooled symbol/state information is high, but a sender's symbol never depends on its state
    rng = random.Random(1)
    rows = []
    for tick in range(800):
        sender = rng.randrange(4)
        rows.append((tick, sender, {"food_direction": "none", "energy_band": ("low", "mid", "high", "full")[sender],
                                    "food_near": False, "neighbor_near": False}, sender))
    result = search.sender_information(rows, reps=50)
    assert result["excess_bits"] > 1.5 and result["p_value"] < .05
    assert abs(result["within_sender"]["excess_bits"]) < .01
    row = _row(sender=result)
    assert search.score(row)[1]["sender"] == 0  # habits earn no sender score
    assert search.criteria(row)["sender_information"] is False  # nor the sender criterion (critic B2)


def _row(sender=None, convention=None, paired=None):
    row = {"measures": {"sender": sender or {"excess_bits": None, "p_value": None, "within_sender": {}},
                        "receiver": {"excess_bits": None, "p_value": None},
                        "convention": convention or {"difference": None, "p_value": None},
                        "cooperation": {"excess": None, "p_value": None, "shares": 0},
                        "tool_damage_gain_per_agent_tick": 0.}}
    if paired is not None:
        row["paired"] = paired
    return row


def _habit_rows(rng, ticks, per_sender_mapping, senders=8, habits=None):
    """Each sender has its own state-dependent symbol habit (a private mapping from energy
    band to symbol; ``habits``, the same in every arm): within-sender information without
    any shared meaning. ``per_sender_mapping`` False gives each sender one fixed symbol."""
    bands = ("low", "middle", "high")
    habits = habits or {s: [rng.randrange(4) for _ in bands] for s in range(senders)}
    rows = []
    for tick in range(ticks):
        sender = rng.randrange(senders)
        band = rng.randrange(3) if per_sender_mapping else sender % 3
        symbol = habits[sender][band] if per_sender_mapping else sender % 4
        if rng.random() < .1:
            symbol = rng.randrange(4)
        rows.append((tick, sender, {"food_direction": "none", "energy_band": bands[band], "food_near": False,
                                    "neighbor_near": False}, symbol))
    return rows


def _paired_with(normal_rows, control_rows):
    branch = lambda arm, rows: {"arm": arm, "rep": 0, "heard": [], "signals": rows, "population": 10,
                                "survival": 1., "births": 0, "food_energy_per_agent_tick": .1}
    return search.paired_measures([branch("normal", normal_rows), branch("masked", control_rows),
                                   branch("scrambled", control_rows), branch("tools_off", normal_rows)],
                                  perm_reps=50)


def test_planted_sender_habits_fail_the_sender_criterion():
    """Critic B2 regression: per-sender habits pass the pooled test but must not pass the
    sender criterion, neither in discovery.verdict nor in search.criteria."""
    rng = random.Random(4)
    # 1. one fixed symbol per sender, each sender living in its own band: pooled excess is large
    fixed = _habit_rows(rng, 1200, per_sender_mapping=False)
    events = [{"tick": t, "type": "signal", "agent": s, "symbol": y, "sender": st} for t, s, st, y in fixed]
    sender = discovery.sender_information(events, reps=50, seed=1)
    assert sender["joint"]["excess_bits"] > .2 and sender["joint"]["p_value"] <= .02
    null_receiver = {"normal": {"pooled": {"excess_bits": 0., "p_value": .5}}, "excess_over_scrambled_bits": 0.}
    verdict = discovery.verdict(sender, null_receiver, None)
    assert verdict["criteria"]["sender_information"]["passed"] is False and verdict["verdict"] == "null"
    screen = _row(sender=search.sender_information(fixed, reps=50))
    assert search.criteria(screen)["sender_information"] is False
    # 2. private state-dependent habits, identical in the control arm (content cannot matter there):
    # within-sender excess is large in both arms, so the paired criterion and the verdict with a control fail
    habits = {s: [rng.randrange(4) for _ in range(3)] for s in range(8)}
    private = _habit_rows(rng, 1500, per_sender_mapping=True, habits=habits)
    control = _habit_rows(random.Random(5), 1500, per_sender_mapping=True, habits=habits)
    info = search.sender_information(private, reps=100)  # p can reach 0.01
    assert info["within_sender"]["excess_bits"] > .15 and info["within_sender"]["p_value"] <= .02
    paired = _paired_with(private, control)
    assert paired["sender"]["control"] == "masked" and abs(paired["sender"]["excess_over_control_bits"]) < .1
    row = _row(sender=info, paired=paired)
    assert search.criteria(row)["sender_information"] is False
    events = [{"tick": t, "type": "signal", "agent": s, "symbol": y, "sender": st} for t, s, st, y in private]
    control_events = [{"tick": t, "type": "signal", "agent": s, "symbol": y, "sender": st} for t, s, st, y in control]
    full = discovery.sender_information(events, reps=50, seed=1)
    verdict = discovery.verdict(full, null_receiver, None,
                                sender_control=discovery.sender_information(control_events, reps=50, seed=1))
    assert verdict["criteria"]["sender_information"]["passed"] is False
    # the same mapping with an uninformative control arm does pass: the criterion is not blind
    noise = [(t, s, st, random.Random(t).randrange(4)) for t, s, st, _ in control]
    passing = _row(sender=info, paired=_paired_with(private, noise))
    assert passing["paired"]["sender"]["excess_over_control_bits"] > .1
    assert search.criteria(passing)["sender_information"] is True
    assert search.score(passing)[1]["sender"] > search.score(row)[1]["sender"]


def test_reciprocity_against_the_proximity_matched_null():
    rng = random.Random(2)
    planted = [(t, 1, True, [True, False, False, False]) for t in range(60)]
    assert search.reciprocity(planted, reps=200)["excess"] == pytest.approx(.75)
    assert search.reciprocity(planted, reps=200)["p_value"] < .01
    noise = []
    for t in range(400):
        flags = [rng.random() < .3 for _ in range(4)]
        noise.append((t, 1, flags[0], flags))
    result = search.reciprocity(noise, reps=200)
    assert abs(result["excess"]) < .08 and result["p_value"] > .01
    assert search.reciprocity([(0, 1, True, [True])])["shares"] == 0  # no choice, no evidence


def test_options_pass_through_to_config_and_unknown_fields_are_refused():
    plan = search.make_plan("plain", 2, 5, {"visibility": .5})
    assert [e["id"] for e in plan] == ["plain-5", "plain-6"]
    assert plan[0]["config"]["visibility"] == .5 and plan[0]["config"]["food_patches"] == 30
    assert plan[0]["config"]["log"] == "compact"
    with pytest.raises(TypeError):
        search.make_plan("plain", 1, 0, {"no_such_field": 1})
    assert search.parse_option("food_patches=40") == ("food_patches", 40)
    assert search.parse_option("genome=evolving") == ("genome", "evolving")


def test_a_failing_world_keeps_a_row(tmp_path):
    entry = {"id": "plain-0", "source": "plain", "seed": 0, "config": {"no_such_field": 1}, "founders": None}
    row = search._segment_job((entry, str(tmp_path), 0, 10, 10, None))
    assert "error" in row and row["id"] == "plain-0"


def test_branch_arms_are_paired_and_deterministic():
    entry = search.make_plan("plain", 1, 3)[0]
    _, bundle = search.run_segment(entry, 40, perm_reps=10)
    a = search.run_branch(bundle, "normal", 0, 20)
    b = search.run_branch(bundle, "normal", 0, 20)
    off = search.run_branch(bundle, "tools_off", 0, 20)
    masked = search.run_branch(bundle, "masked", 0, 20)
    assert a == b
    assert off["arm"] == "tools_off" and off["ticks"] == 20
    assert set(search.ARMS) == {"normal", "masked", "scrambled", "tools_off"}
    measures = search.paired_measures([a, off, masked, search.run_branch(bundle, "scrambled", 0, 20)], perm_reps=10)
    assert measures["normal_minus_tools_off"]["food_energy_per_agent_tick"]["n"] == 1
    assert measures["normal_minus_masked"]["food_energy_per_agent_tick"]["n"] == 1
    assert measures["sender"]["control"] == "masked" and a["signals"] and masked["signals"]


def test_living_preset_reaches_the_search_rows_and_its_sender_state(tmp_path):
    """Critic B1: with --preset living the searched worlds have kin credit, predators and
    blooms, and the sender state and situation include the ecology bins."""
    plan = search.make_plan("plain", 2, 1, {"population_limit": 200}, preset="living")
    for entry in plan:
        assert entry["config"]["kin_credit"] > 0 and entry["config"]["predators"] == 2
        assert entry["config"]["genome"] == "evolving" and entry["config"]["population_limit"] == 200
    assert search.make_plan("plain", 1, 1)[0]["config"]["kin_credit"] == 0  # no preset: Matrix-default worlds
    out = tmp_path / "living"
    search.run_search(out, worlds=2, rounds=1, ticks=60, workers=1, perm_reps=10, preset="living", log=lambda _: None)
    rows = [r for r in _rows(out) if r["kind"] == "world"]
    assert len(rows) == 2 and all(r["kin_credit"] > 0 and r["ecology"] for r in rows)
    assert all(set(search.ECOLOGY_BANDS) <= set(r["measures"]["sender"]["state"]) for r in rows)
    meta = json.loads((out / "search.json").read_text(encoding="utf-8"))
    assert meta["args"]["preset"] == "living" and all(e["config"]["kin_credit"] > 0 for e in meta["plan"])
    assert search.sender_state(search.Config()) == search.SENDER_STATE
    with pytest.raises(ValueError):
        search.make_plan("plain", 1, 0, preset="no_such_preset")


def test_bridged_plan_alternates_world7_and_synthetic_and_splits_by_host():
    plan = search.make_plan("bridged", 4, 6, preset="living")  # world7 seeds 6 and 7 reach bodies
    assert [e["id"] for e in plan] == ["world7-6", "synthetic-7", "world7-7", "synthetic-9"]
    for entry in plan:
        assert entry["config"]["kin_credit"] > 0 and entry["founders"]["ecology_hints"]["connected"] is True
    first, second = search.split_plan(plan, [3, 1])
    assert [e["id"] for e in first + second] == [e["id"] for e in plan]
    with pytest.raises(ValueError):
        search.split_plan(plan, [2, 1])
    world = search.build_world(plan[0])
    assert world.config.kin_credit > 0 and world.eco is not None


def test_tiny_search_keeps_the_best_half_and_saves_only_kept_checkpoints(tmp_path):
    out = tmp_path / "search"
    search.run_search(out, worlds=8, rounds=2, ticks=200, keep=.5, workers=8, source="mixed", reps=2,
                      branch_ticks=50, perm_reps=20, log=lambda _: None)
    rows = _rows(out)
    worlds = [r for r in rows if r["kind"] == "world"]
    selections = [r for r in rows if r["kind"] == "selection"]
    assert len([r for r in worlds if r["round"] == 0]) == 8  # every world has a row
    assert len([r for r in worlds if r["round"] == 1]) == 4
    assert [s["round"] for s in selections] == [0, 1]
    round0 = sorted((r for r in worlds if r["round"] == 0 and not r.get("stop_reason") and "error" not in r),
                    key=lambda r: (-r["score"], r["id"]))
    assert selections[0]["kept"] == [r["id"] for r in round0[:4]]
    assert {r["id"] for r in worlds if r["round"] == 1} == set(selections[0]["kept"])
    for r in worlds:
        assert r["end_tick"] - r["start_tick"] == (200 if r["round"] == 0 else 400) or r["stop_reason"]
        assert r["paired"] is not None if r["round"] == 1 and not r["stop_reason"] else "paired" not in r
    checkpoints = sorted(p.name for p in (out / "checkpoints").iterdir())
    assert checkpoints == sorted(f"{i}.r1.json.gz" for i in selections[1]["kept"])
    assert not any((out / "pending").rglob("*.gz"))
    with gzip.open(out / "checkpoints" / checkpoints[0], "rt", encoding="utf-8") as stream:
        assert json.load(stream)["tick"] == 600
    meta = json.loads((out / "search.json").read_text(encoding="utf-8"))
    assert meta["status"] == "complete" and meta["stop_reason"] == "rounds_done"
    # running again on a finished folder adds nothing
    search.run_search(out, log=lambda _: None)
    assert _rows(out) == rows
    summary = search.report(rows)
    assert summary["rounds"][0]["worlds"] == 8 and summary["rounds"][1]["kept"] == 2


class _Stop(Exception):
    pass


def test_interrupted_search_resumes_to_the_uninterrupted_result(tmp_path):
    settings = dict(worlds=3, rounds=2, ticks=30, keep=.5, workers=1, reps=1, branch_ticks=10, perm_reps=10,
                    start_seed=11)
    search.run_search(tmp_path / "direct", log=lambda _: None, **settings)

    def interrupt(line):
        if json.loads(line).get("round") == 1 and "to_run" in json.loads(line):
            raise _Stop
    with pytest.raises(_Stop):
        search.run_search(tmp_path / "resumed", log=interrupt, **settings)
    assert [r["kind"] for r in _rows(tmp_path / "resumed")].count("selection") == 1
    search.run_search(tmp_path / "resumed", log=lambda _: None)
    strip = lambda rows: [{k: v for k, v in r.items() if k != "seconds"} for r in rows]
    assert strip(_rows(tmp_path / "resumed")) == strip(_rows(tmp_path / "direct"))
    assert sorted(p.name for p in (tmp_path / "resumed" / "checkpoints").iterdir()) == \
        sorted(p.name for p in (tmp_path / "direct" / "checkpoints").iterdir())


def test_collect_merges_host_folders(tmp_path):
    for host in ("hostA", "hostB"):
        (tmp_path / host).mkdir()
        (tmp_path / host / "results.jsonl").write_text(json.dumps(
            {"kind": "world", "round": 0, "round_ticks": 10, "id": "plain-1", "score": 1., "seconds": 1.,
             "stop_reason": None, "criteria": {"language_verdict": "screen_only"}, "components": {},
             "end_tick": 10, "population": 3, "measures": {"sender": {"excess_bits": 0},
                                                           "convention": {"difference": None},
                                                           "cooperation": {"excess": None, "shares": 0}}}) + "\n")
    summary = search.collect(tmp_path / "merged", [tmp_path / "hostA", tmp_path / "hostB"])
    assert summary["worlds"] == 2 and summary["rounds"][0]["worlds"] == 2
    assert len(search.read_results(tmp_path / "merged")) == 2


def test_output_cap_stops_before_a_round_and_a_larger_cap_resumes(tmp_path):
    out = tmp_path / "capped"
    meta = search.run_search(out, worlds=2, rounds=1, ticks=10, workers=1, perm_reps=5, max_output_mb=.5,
                             log=lambda _: None)
    assert meta["status"] == "stopped" and meta["stop_reason"] == "output_cap"
    assert _rows(out) == []
    meta = search.run_search(out, max_output_mb=50, log=lambda _: None)
    assert meta["status"] == "complete" and len([r for r in _rows(out) if r["kind"] == "world"]) == 2
