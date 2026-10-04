"""Unselected-chain gates with cheap stand-ins for the legacy cosmic solvers."""
from types import SimpleNamespace

import pytest

from haishool.life10 import chain
from haishool.truth.formula import parse


def install_chain(monkeypatch, *, star_share=0.5, cloud=None, system=None, planet_error=None):
    calls = []
    composition = cloud or {"hydrogen": 0.70, "helium": 0.28, "carbon": 0.005,
                            "nitrogen": 0.002, "oxygen": 0.010, "neon": 0.001,
                            "magnesium": 0.0005, "silicon": 0.0005, "iron": 0.001,
                            "other": 0.0, "metallicity": 0.02, "time": 4.0}
    if system is None:
        system = [
            {"id": 91, "mass": 0.123456789, "solid": 0.123456789, "gas": 0,
             "orbit": 0.03, "type": "rocky", "habitable": "no"},
            {"id": 4, "mass": 12.0, "solid": 2.0, "gas": 10.0,
             "orbit": 7.0, "type": "gas", "habitable": "no"},
            {"id": 17, "mass": 1.0, "solid": 1.0, "gas": 0,
             "orbit": 30.0, "type": "rocky", "habitable": "no"},
        ]

    def rollout(name, summary, params, *, system=None):
        r = SimpleNamespace(seed=13, params=params, steps=[summary], summary=summary)
        if system is not None:
            r.system, r.bodies, r.events = system, [system], []
        calls.append(name)
        return r

    def nucleo_run(seed, **kwargs):
        return rollout("nucleo", {"hydrogen": 0.750123456, "helium": 0.2498,
                                  "deuterium": 2e-5, "helium3": 1e-5, "lithium": 4e-10}, kwargs)

    def stars_run(seed, **kwargs):
        return rollout("stars", composition, kwargs)

    def evolve(*args, **kwargs):
        calls.append("collapse")
        args[6]["radiated"] = 0.123456789
        yield None, None, None

    def measure(*args):
        return {"radius": 0.345678901, "largest_clump": star_share,
                "disc_mass": 0.4, "stage": "multiple", "clumps": 2}

    def planet_run(seed, **kwargs):
        if planet_error:
            raise planet_error
        return rollout("planets", {"planets": len(system)}, kwargs, system=system)

    w7 = SimpleNamespace(
        _params=lambda seed, given: {"particles": given["particles"], "cloud_mass": 1.0,
                                    "formation_time": 4.25, "efficiency": 0.3,
                                    "spin": 0.22, "cooling": 0.31, "t_gas": 3.0},
        enrichment_years=lambda t: int((t - 0.25) * 1e9), GYR=1e9,
        Z_CRIT=1e-5, JEANS_FACTOR=90, COOLING_PRIMORDIAL=0.1,
        EARTHS_PER_SUN=333000, DISC_RETAINED=0.08, MIN_STAR=0.08,
    )
    modules = (
        SimpleNamespace(run=nucleo_run),
        SimpleNamespace(run=stars_run, cloud=lambda r: composition),
        SimpleNamespace(evolve=evolve, measure7=measure, MASS=1.0, EPS=0.1, OUT_DT=0.05),
        SimpleNamespace(run=planet_run, luminosity7=lambda m: m ** 4,
                        flux=lambda light, orbit: light / orbit ** 2,
                        equilibrium_temperature=lambda f: 254.6 * f ** 0.25,
                        greenhouse=lambda f: 0.0),
        w7,
    )
    monkeypatch.setattr(chain, "_modules", lambda: modules)
    monkeypatch.setattr(chain, "source_hashes", lambda: {"fake-law.py": "fixed-hash"})
    return calls


def test_no_star_is_retained_without_planet_or_biology_retry(monkeypatch):
    calls = install_chain(monkeypatch, star_share=0.01)
    result = chain.build_chain(13)
    assert result["outcome"] == "no_star"
    assert result["planets"] == []
    assert calls == ["nucleo", "stars", "collapse"]
    assert result["errors"] == []
    assert result["material_ledger"]["relative_max_error"] < 1e-12


def test_zero_metals_keeps_no_solids_outcome(monkeypatch):
    calls = install_chain(monkeypatch, cloud={"hydrogen": 0.7, "helium": 0.3,
                                              "metallicity": 0.0, "time": 4.0})
    result = chain.build_chain(13)
    assert result["outcome"] == "no_disc_solids"
    assert "planets" not in calls
    assert result["star"]["disc_solids_earth"] == 0


def test_all_planets_retained_even_hot_frozen_gas_and_uninhabitable(monkeypatch):
    calls = install_chain(monkeypatch)
    result = chain.build_chain(13)
    assert result["outcome"] == "planets_formed"
    assert [p["id"] for p in result["planets"]] == [91, 4, 17]
    assert all(p["upstream_diagnostics"]["habitable"] == "no" for p in result["planets"])
    assert result["planets"][0]["mass_earth"] == 0.123456789
    assert result["stages"]["collapse"]["steps"][0]["radius"] == 0.345678901
    assert calls == ["nucleo", "stars", "collapse", "planets"]


def test_empty_system_is_valid_not_retried(monkeypatch):
    calls = install_chain(monkeypatch, system=[])
    result = chain.build_chain(13)
    assert result["outcome"] == "no_planets"
    assert result["planets"] == []
    assert calls.count("planets") == 1


def test_element_allocation_closes_and_never_injects_phosphorus(monkeypatch):
    install_chain(monkeypatch)
    result = chain.build_chain(13, config=chain.ChainConfig(sample_atoms=20000))
    led = result["material_ledger"]
    assert led["relative_max_error"] < 1e-12
    for symbol, source_mass in led["source_element_mass_kg"].items():
        allocated = sum(r[symbol] for r in led["reservoir_element_mass_kg"].values())
        assert allocated == pytest.approx(source_mass, rel=1e-12, abs=1e-12)
    for planet in result["planets"]:
        assert sum(planet["atom_inventory"].values()) == 20000
        assert planet["atom_quantum_mol"] == 1e-6
        assert planet["atom_inventory_unit"] == "integer atom-equivalent quanta"
        assert planet["atom_inventory"]["P"] == 0
        assert planet["element_mass_kg"]["P"] == 0
        assert all(isinstance(n, int) and n >= 0 for n in planet["atom_inventory"].values())
        assert all(m <= planet["element_mass_kg"][s]
                   for s, m in planet["sample_element_mass_kg"].items())
        for s, mass in planet["element_mass_kg"].items():
            assert (planet["bulk_residual_element_mass_kg"][s]
                    + planet["sample_element_mass_kg"].get(s, 0)) == pytest.approx(mass)


def test_serialization_replays_and_repeat_changes_only_microstate_seed(monkeypatch):
    install_chain(monkeypatch)
    first, replay = chain.build_chain(13, 7), chain.build_chain(13, 7)
    assert chain.canonical_json(first) == chain.canonical_json(replay)
    other = chain.build_chain(13, 8)
    assert first["stages"] == other["stages"]
    assert first["material_ledger"] == other["material_ledger"]
    for p, q in zip(first["planets"], other["planets"]):
        assert p["atom_inventory"] == q["atom_inventory"]
        assert p["chemistry_seed"] != q["chemistry_seed"]
    assert chain.stream_seed(13, 7, 91) == first["planets"][0]["chemistry_seed"]


def test_numeric_failure_is_kept_as_outcome_with_completed_stages(monkeypatch):
    install_chain(monkeypatch, planet_error=ArithmeticError("test integration failure"))
    result = chain.build_chain(13)
    assert result["outcome"] == "numeric_or_model_error"
    assert "nucleosynthesis" in result["stages"]
    assert "collapse" in result["stages"]
    assert result["errors"] == [{"stage": "planet_formation", "type": "ArithmeticError",
                                 "message": "test integration failure"}]


def test_oversubscribed_planet_resources_fail_instead_of_rescaling(monkeypatch):
    system = [{"id": 1, "mass": 1e8, "orbit": 1.0, "type": "rocky", "gas": 0}]
    install_chain(monkeypatch, system=system)
    result = chain.build_chain(13)
    assert result["outcome"] == "numeric_or_model_error"
    assert result["system"][0]["mass"] == 1e8
    assert "refusing resource creation" in result["errors"][0]["message"]


def test_nonfinite_legacy_state_is_recorded_as_error_in_valid_json(monkeypatch):
    install_chain(monkeypatch, star_share=float("nan"))
    result = chain.build_chain(13)
    assert result["outcome"] == "numeric_or_model_error"
    encoded = chain.canonical_json(result)
    assert '"nonfinite":"nan"' in encoded
    assert "NaN" not in encoded


def test_initial_molecule_convention_preserves_every_atom():
    atoms = {"H": 101, "C": 9, "N": 13, "O": 25, "He": 7, "P": 0}
    molecules = chain.initial_molecules(atoms)
    actual = {s: 0 for s in atoms}
    for formula, number in molecules.items():
        for s, stoich in parse(formula).items():
            actual[s] = actual.get(s, 0) + number * stoich
    assert actual == atoms
    assert not any(name in molecules for name in ("AMP", "GMP", "ATP"))


@pytest.mark.parametrize("config", [chain.ChainConfig(collapse_particles=2),
                                    chain.ChainConfig(sample_atoms=-1)])
def test_invalid_resolution_is_configuration_error(config):
    with pytest.raises(ValueError):
        chain.build_chain(13, config=config)
