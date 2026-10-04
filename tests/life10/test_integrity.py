"""Effective inputs, paired evidence and per-planet screening gates."""
import copy
from dataclasses import asdict

import pytest

from haishool.life10 import __main__ as cli
from haishool.life10.analysis import pair_invalid_reasons, repeated_population_planets
from haishool.life10.chain import FORMAT_VERSION, SOURCE_PATHS, ChainConfig
from haishool.life10.protocol import experiment_identity


def cached():
    laws = {path: "fixed" for path in SOURCE_PATHS}
    row = {"cosmic_seed": 3, "format_version": FORMAT_VERSION,
           "config": asdict(ChainConfig()), "source_hashes": laws,
           "planets": [], "errors": [], "outcome": "no_star"}
    return row, {"files": laws}


@pytest.mark.parametrize("field,value", [
    ("config", asdict(ChainConfig(sample_atoms=1))),
    ("format_version", "different"), ("source_hashes", {"old": "law"})])
def test_invalid_cached_inputs_are_rejected(field, value):
    row, frozen = cached()
    row[field] = value
    with pytest.raises(ValueError):
        cli.validate_cached_chains([row], [3], ChainConfig(), frozen)


def test_duplicate_and_missing_cache_seeds_are_rejected():
    row, frozen = cached()
    with pytest.raises(ValueError, match="Duplicate"):
        cli.validate_cached_chains([row, row], [3], ChainConfig(), frozen)
    with pytest.raises(ValueError, match="missing"):
        cli.validate_cached_chains([row], [4], ChainConfig(), frozen)


def test_effective_chain_state_changes_experiment_identity():
    args = ({"steps": 3}, [3], {"sha256": "source"}, ["normal"])
    assert experiment_identity(*args, chain_inputs_sha256="a") != experiment_identity(*args, chain_inputs_sha256="b")


def complete_row():
    return {"status": "complete", "censored": False, "chain_errors": [],
            "requested_steps": 300, "steps": 300, "planets": 2, "stop_reason": "step_budget",
            "source_sha256": "laws", "chain_inputs_sha256": "inputs",
            "seeds": [3, 4], "repeat": 0, "physical_config": {"steps": 300, "grid": 4},
            "device": "cpu"}


@pytest.mark.parametrize("field,value", [
    ("source_sha256", "other"), ("chain_inputs_sha256", None), ("seeds", [4, 3]),
    ("repeat", 1), ("physical_config", {"steps": 300, "grid": 2}), ("device", "cuda:0"),
    ("requested_steps", 301), ("steps", 299), ("chain_errors", [{"stage": "collapse"}]),
    ("censored", True), ("status", "failed")])
def test_mismatched_or_incomplete_controls_never_validate(field, value):
    normal = complete_row()
    control = copy.deepcopy(normal)
    control[field] = value
    assert pair_invalid_reasons(normal, control)


def test_complete_identical_physical_controls_validate_and_empty_systems_are_retained():
    a = complete_row()
    assert not pair_invalid_reasons(a, copy.deepcopy(a))
    a.update(planets=0, steps=0, stop_reason="no_formed_planets")
    assert not pair_invalid_reasons(a, copy.deepcopy(a))


def sample(step, counts):
    return {"step": step, "planets": [{"seed": 3, "planet": p, "chains": n, "longest": length}
                                       for p, n, length in counts]}


def test_population_screen_requires_same_planet_both_samples():
    history = [sample(1, [(0, 12, 2), (1, 2, 4)]),
               sample(2, [(0, 12, 2), (1, 2, 4)])]
    assert not repeated_population_planets(history)
    history = [sample(1, [(0, 12, 4), (1, 2, 2)]),
               sample(2, [(0, 2, 2), (1, 12, 4)])]
    assert not repeated_population_planets(history)
    history = [sample(1, [(0, 12, 4)]), sample(2, [(0, 11, 5)])]
    assert repeated_population_planets(history) == [{"seed": 3, "planet": 0}]
    assert not repeated_population_planets([history[0], history[0]])
