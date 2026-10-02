"""The visitor experience format (haishool/life8/experience.py)."""
import gzip
import json
import math

import pytest

from haishool import train_final
from haishool.life8 import experience as x
from haishool.life8 import learning
from haishool.life8.config import living_config
from haishool.life8.world import ACTIONS, World
from haishool.truth import is_dense, split_line


def _entry(seed=3, **overrides):
    config = living_config(population=10, population_limit=64, **overrides).to_dict()
    return {"id": f"plain-{seed}", "source": "plain", "seed": seed, "config": config, "founders": None}


def _observations(seed=3, ticks=12):
    world = World.create(seed, living_config(population=10, population_limit=64))
    seen = []
    for _ in range(ticks):
        seen += [world.observe(a) for a in world.agents.values()]
        world.step()
    return seen


def test_action_words_are_the_engine_actions_and_single_dense_tokens():
    assert x.ACTION_WORDS == tuple(ACTIONS) == tuple(learning.ACTIONS)
    for word in x.ACTION_WORDS:
        assert is_dense(f"q a. a {word}.") and " " not in word


def test_lines_from_real_observations_are_dense_short_and_round_trip():
    for observation in _observations():
        features = x.features_of(observation)
        assert features == learning.observation_features(observation, extras=True)
        for outcome in x.OUTCOMES:
            line = x.line_for(features, outcome, "eat")
            assert is_dense(line), line
            assert x.token_count(line) < x.MAX_TOKENS
            see, want, action = x.parse_line(line)
            assert (want, action) == (outcome, "eat")
            assert see == x.see_text(features)
            assert line == x.prompt_for(features, outcome) + " eat."
            assert split_line(line) == (f"life8 see {see} want {outcome}", "eat")
            names = see.split()[::2]
            assert names == [n for n in x.FEATURE_ORDER if n in names]  # fixed order


def test_every_feature_at_once_stays_under_the_token_limit():
    features = {name: "middle" for name in x.FEATURE_ORDER}
    features.update(inventory_count=4, heard_symbol=3, own_call=2, food_near=True, danger_seen=False)
    line = x.line_for(features, "thrive", "signal_3")
    assert is_dense(line) and x.token_count(line) < x.MAX_TOKENS


def test_values_absent_omitted_bools_words_integers_digits():
    see = x.see_text({"food_near": True, "danger_seen": False, "heard_symbol": None, "inventory_count": 12,
                      "food_direction": "ne"})
    assert see == "food_direction ne food_near yes inventory_count 1 2 danger_seen no"
    assert x.prompt_for({}) == "q life8 see want thrive. a"
    assert is_dense(x.line_for({}, "suffer", "rest"))


def test_unknown_features_and_values_are_refused():
    with pytest.raises(ValueError):
        x.see_text({"secret_id": "a"})
    with pytest.raises(ValueError):
        x.see_text({"food_direction": "North East"})
    with pytest.raises(ValueError):
        x.line_for({}, "great", "eat")
    with pytest.raises(ValueError):
        x.line_for({}, "thrive", "fly")
    with pytest.raises(ValueError):
        x.parse_line("q carbon protons. a 6.")


def test_outcome_cuts_and_answer_parsing():
    cuts = (-1., 2.)
    assert [x.outcome_of(v, cuts) for v in (-1.5, -1., 0., 2., 2.5)] == ["suffer", "live", "live", "live", "thrive"]
    assert x.action_of(" eat.") == "eat" and x.action_of("move_ne") == "move_ne"
    assert x.action_of("dance") is None and x.action_of("") is None


def test_collect_sums_each_agents_own_reward_over_the_horizon():
    entry = _entry()
    rows, stats = x.collect(entry, 40)
    # the same world with the same seed, full log: rebuild the reward series by hand
    world = x.build_world(entry, "full")
    series, died = {}, set()
    for _ in range(40):
        for event in world.step():
            if event["type"] == "transition":
                series.setdefault(event["agent"], {})[event["tick"]] = event["reward"]
                if event["done"]:
                    died.add(event["agent"])
    assert rows and stats["decisions"] == sum(len(v) for v in series.values())
    for tick, agent, see, action, total in rows:
        window = [series[agent][t] for t in range(tick, tick + x.HORIZON) if t in series[agent]]
        assert math.isclose(total, math.fsum(window), rel_tol=0, abs_tol=1e-12)
        assert len(window) == x.HORIZON or agent in died
        assert action in x.ACTION_WORDS
    # windows that would run past tick 40 for a living agent are dropped
    alive = set(world.agents)
    assert all(tick + x.HORIZON - 1 <= 40 for tick, agent, *_ in rows if agent in alive)
    again, _ = x.collect(entry, 40)
    assert again == rows


def test_kept_is_deterministic_and_near_the_rate():
    picks = [x.kept("p", t, a, .25) for t in range(200) for a in range(50)]
    assert picks == [x.kept("p", t, a, .25) for t in range(200) for a in range(50)]
    assert abs(sum(picks) / len(picks) - .25) < .02
    assert all(x.kept("p", t, 1, 1.) for t in range(10))


def _plan():
    plan = []
    for source, n in (("world7", 400), ("synthetic", 150)):
        plan += [{"id": f"{source}-{i}", "source": source, "seed": i} for i in range(n)]
    return plan


def test_planet_draw_is_deterministic_disjoint_and_sized():
    groups = x.select_planets(_plan())
    assert groups == x.select_planets(_plan())
    sizes = {k: len(v) for k, v in groups.items()}
    assert sizes == {"traveller": 40, "elder": 40, "train": 320, "dev": 40, "sealed": 40}
    ids = [i for v in groups.values() for i in v]
    assert len(ids) == len(set(ids))
    for key in groups:
        assert sum(i.startswith("world7") for i in groups[key]) == {"traveller": 30, "elder": 30, "train": 240,
                                                                    "dev": 30, "sealed": 30}[key]


def _raw(raw, planet, rows):
    with gzip.open(raw / f"{planet}.tsv.gz", "wt", encoding="utf-8") as handle:
        for tick, agent, action, total, see in rows:
            handle.write(f"{tick}\t{agent}\t{action}\t{total!r}\t{see}\n")


def test_assembled_data_is_what_train_final_reads_and_elder_stays_before_tick_1000(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    see = "food_direction n food_near no energy_band middle"
    groups = {"train": ["a", "b"], "dev": ["c"], "sealed": ["d"], "elder": ["e"], "traveller": ["f"]}
    for i, planet in enumerate(groups["train"] + groups["dev"] + groups["sealed"]):
        _raw(raw, planet, [(t, k, "eat" if t % 2 else "rest", float(t % 7) - 3, see) for t in range(1, 60) for k in (1, 2)])
    _raw(raw, "e", [(t, k, "move_n", float(t % 5), see) for t in (5, 990, 991, 999, 1500) for k in range(1, 30)])
    stats = {p: {"id": p, "source": "plain", "seconds": 1., "decisions": 1, "ticks": 60, "stop_reason": None,
                 "population_mean": 2., "kept_rows": 1} for p in "abcde"}
    out = tmp_path / "data"
    report = x.assemble(out, raw, groups, stats, keep=1., ticks=60, elder_ticks=1000, wall=1., workers=1,
                        plan_path="plan.json")
    low, high = report["thresholds"]["suffer_below"], report["thresholds"]["thrive_above"]
    assert low < high
    manifest, paths, _ = train_final.input_manifest(out, None, None, None, None, mix="sim")
    assert {p.name for p in paths} == {f"{x.TOPIC}-train.txt", f"{x.ELDER_TOPIC}-train.txt"}
    for topic in (x.TOPIC, x.ELDER_TOPIC):
        for split in x.SPLITS:
            lines = (out / f"{topic}-{split}.txt").read_text(encoding="utf-8").splitlines()
            assert len(lines) == report["topics"][topic]["splits"][split]["lines"]
            assert all(is_dense(line) for line in lines)
    elder = [ln for s in x.SPLITS for ln in (out / f"{x.ELDER_TOPIC}-{s}.txt").read_text(encoding="utf-8").splitlines()]
    assert len(elder) == 3 * 29  # ticks 5, 990 and 991 (window 991-1000); 999 and 1500 end after tick 1000
    assert report["planets"]["traveller"] == ["f"]
    with pytest.raises(FileExistsError):
        x.assemble(out, raw, groups, stats, keep=1., ticks=60, elder_ticks=1000, wall=1., workers=1, plan_path="p")


def test_visitor_mixture_weights_topics_by_tokens():
    sources = [train_final.Source("a", x.TOPIC, None, count=300), train_final.Source("b", x.ELDER_TOPIC, None, count=100)]
    x.proportional_sim_mixture(sources)
    assert [s.weight for s in sources] == [.75, .25]
    with pytest.raises(ValueError):
        x.proportional_sim_mixture([train_final.Source("c", "predict_gravity", None, count=1)])
