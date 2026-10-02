"""Experiment-scale claims (slow; LIFE8_RUN_SLOW=1) and a fast smoke test of the measurement code."""
import pytest

from haishool.life8.analysis import founders


def test_founder_measurement_smoke():
    row = founders.run("fixed", 1, ticks=40)
    assert row["founders"] == 24 and row["founders_alive_at_age_300"] is None and row["ticks_run"] == 40
    assert founders.run("birth_in_reward", 1, ticks=40)["condition"] == "birth_in_reward"


@pytest.mark.slow
def test_birth_fix_raises_founder_survival_against_the_same_seed_intervention():
    """Review bug 1: seeds 1-6, 600 ticks, capacity raised so every run reaches tick 300.

    Measured on 2026-10-02 (docs/life8/results/founders-birth-fix-cap256.json):
    see docs/life8/README.md for the numbers.
    """
    report = founders.measure(range(1, 7), 600, jobs=6, config={"population_limit": 256})
    fixed, old = report["totals"]["fixed"], report["totals"]["birth_in_reward"]
    assert fixed["runs_stopped_before_300"] == old["runs_stopped_before_300"] == 0
    assert fixed["founders_alive_at_age_300"] > old["founders_alive_at_age_300"] + 20
    assert report["paired_seeds_fixed_higher"] >= 5
    assert fixed["mean_eat_when_food_in_reach"] > old["mean_eat_when_food_in_reach"] + .2
