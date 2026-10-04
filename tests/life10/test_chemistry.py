"""Conservation, controls and reproducibility of declared life10 chemistry."""
import pytest
import torch

from haishool.life10.chemistry import (
    ChemicalConfig, ChemicalEngine, ELEMENTS, REACTIONS, R_GAS, SPECIES, SPECIES_INDEX,
)


def config(**kwargs):
    return ChemicalConfig(**{"grid_size": 3, "batch_size": 2, "dt_s": 10.0,
                             "photon_input_j": 0.2, **kwargs})


def primordial(engine, seed=3):
    return engine.initialize_from_elements(seed, {"C": 180.0, "H": 2000.0,
                                                  "N": 300.0, "O": 500.0})


def test_catalog_is_balanced_and_no_phosphorus_is_invented():
    for r in REACTIONS:
        left = [sum(SPECIES[SPECIES_INDEX[s]].elements[j] * n for s, n in r.reactants)
                for j in range(len(ELEMENTS))]
        right = [sum(SPECIES[SPECIES_INDEX[s]].elements[j] * n for s, n in r.products)
                 for j in range(len(ELEMENTS))]
        assert left == right, r.name
    engine = ChemicalEngine(config())
    state = primordial(engine)
    assert torch.allclose(engine.elemental_totals(state),
                          torch.tensor([[180., 2000., 300., 500., 0.]] * 2, dtype=torch.float64),
                          rtol=0, atol=1e-10)
    assert state.amount_umol[..., SPECIES_INDEX["H3PO4"]].sum() == 0
    for name in ("HCN", "formaldehyde", "glycine", "alanine", "ribose", "adenine"):
        assert state.amount_umol[..., SPECIES_INDEX[name]].sum() == 0


def test_long_run_closes_element_and_energy_ledgers_without_negative_amounts():
    engine = ChemicalEngine(config(temperature_cycle_amplitude_k=25.0,
                                   temperature_cycle_period_s=200.0))
    state = primordial(engine)
    elements = engine.elemental_totals(state).clone()
    for _ in range(120):
        engine.step(state)
    assert torch.allclose(engine.elemental_totals(state), elements, rtol=1e-12, atol=1e-9)
    assert engine.energy_residual(state).abs().max() < 1e-8
    assert state.amount_umol.min() >= 0
    assert state.heat_j.min() >= 0
    assert state.photon_j.min() >= 0
    assert state.cumulative_photon_input_j.min() > 0
    assert state.cumulative_bath_input_j.abs().sum() > 0


def test_diffusion_conserves_each_species_and_spreads_local_inventory():
    engine = ChemicalEngine(config(max_reaction_fraction=0.0, photon_input_j=0.0))
    state = engine.initialize(0, {})
    state.amount_umol[:, 0, 0, SPECIES_INDEX["glycine"]] = 9.0
    before = state.amount_umol.sum(dim=(1, 2)).clone()
    engine.step(state)
    assert torch.allclose(state.amount_umol.sum(dim=(1, 2)), before, atol=1e-12, rtol=0)
    assert state.amount_umol[:, 0, 0, SPECIES_INDEX["glycine"]].max() < 9
    assert state.amount_umol[:, 0, 1, SPECIES_INDEX["glycine"]].min() > 0


def test_darkness_cannot_make_carbon_organics_from_declared_gases():
    engine = ChemicalEngine(config())
    dark = primordial(engine)
    light = primordial(engine)
    for _ in range(80):
        engine.step(dark, illumination=0.0)
        engine.step(light)
    assert dark.amount_umol[..., SPECIES_INDEX["formaldehyde"]].sum() == 0
    assert dark.amount_umol[..., SPECIES_INDEX["glycine"]].sum() == 0
    assert light.amount_umol[..., SPECIES_INDEX["glycine"]].sum() > 0
    assert light.amount_umol[..., SPECIES_INDEX["HCN"]].sum() > 0
    assert engine.energy_residual(light).abs().max() < 1e-8


def test_empty_world_remains_empty_under_light():
    engine = ChemicalEngine(config())
    state = engine.initialize(4, {})
    for _ in range(5):
        engine.step(state)
    assert state.amount_umol.count_nonzero() == 0
    assert engine.energy_residual(state).abs().max() < 1e-8


def test_activation_increases_with_temperature_and_reverse_returns_energy():
    cold = ChemicalEngine(config(grid_size=1, batch_size=1, temperature_k=200,
                                  diffusion_fraction=0, thermal_diffusion_fraction=0))
    hot = ChemicalEngine(config(grid_size=1, batch_size=1, temperature_k=500,
                                 diffusion_fraction=0, thermal_diffusion_fraction=0))
    r = REACTIONS[0]
    states = [e.initialize(0, {"CO2": 0.1, "H2": 0.2}) for e in (cold, hot)]
    for e, s in zip((cold, hot), states):
        s.photon_j.fill_(1.0)
        initial = e.energy_total(s).clone()
        e._reaction(s, r, 0, False, 1.0)
        assert torch.allclose(e.energy_total(s), initial, rtol=0, atol=1e-10)
    assert states[1].reaction_extent_umol[0, 0, 0] > states[0].reaction_extent_umol[0, 0, 0]
    s = hot.initialize(0, {"formaldehyde": 1.0, "H2O": 1.0})
    initial, heat = hot.energy_total(s).clone(), s.heat_j.clone()
    hot._reaction(s, r, 0, True, 1.0)
    assert s.heat_j.item() > heat.item()
    assert torch.allclose(hot.energy_total(s), initial, rtol=0, atol=1e-10)


def test_checkpoint_repeats_exact_cpu_trajectory():
    engine = ChemicalEngine(config())
    state = primordial(engine, seed=11)
    for _ in range(5):
        engine.step(state)
    restored_engine, restored = ChemicalEngine.load_checkpoint(engine.checkpoint(state))
    for _ in range(7):
        engine.step(state)
        restored_engine.step(restored)
    for name, tensor in state.checkpoint().items():
        if isinstance(tensor, torch.Tensor):
            assert torch.equal(tensor, restored.checkpoint()[name]), name
        else:
            assert tensor == restored.checkpoint()[name]


def test_thermal_rate_ratio_obeys_declared_energy_only_detailed_balance():
    engine = ChemicalEngine(config(grid_size=1, batch_size=1, dt_s=1.0))
    i = 6  # acetaldehyde formation, no photon requirement
    r = REACTIONS[i]
    forward = engine.initialize(0, {s: 0.001 for s, _ in r.reactants})
    reverse = engine.initialize(0, {s: 0.001 for s, _ in r.products})
    engine._reaction(forward, r, i, False, 1.0)
    engine._reaction(reverse, r, i, True, 1.0)
    forward_rate = forward.reaction_extent_umol[0, i, 0] / 0.001 ** sum(n for _, n in r.reactants)
    reverse_rate = reverse.reaction_extent_umol[0, i, 1] / 0.001 ** sum(n for _, n in r.products)
    expected = torch.exp(torch.tensor(-r.delta_energy_j_per_umol * 1e6 / (R_GAS * 300),
                                      dtype=torch.float64))
    assert torch.allclose(forward_rate / reverse_rate, expected, rtol=1e-12, atol=0)


def test_catalysis_does_not_create_matter_or_energy_and_accelerates_both_sides():
    engine = ChemicalEngine(config(grid_size=1, batch_size=1))
    a = engine.initialize(0, {"formaldehyde": 0.1, "glycolaldehyde": 0.1})
    b = engine.initialize(0, {"formaldehyde": 0.1, "glycolaldehyde": 0.1})
    r = REACTIONS[3]
    for reverse in (False, True):
        engine._reaction(a, r, 3, reverse, 1.0)
        engine._reaction(b, r, 3, reverse, 2.0)
    assert (b.reaction_extent_umol[:, 3] > a.reaction_extent_umol[:, 3]).all()
    assert torch.allclose(engine.elemental_totals(a), a.initial_elements_umol, atol=1e-12, rtol=0)
    assert torch.allclose(engine.elemental_totals(b), b.initial_elements_umol, atol=1e-12, rtol=0)
    assert engine.energy_residual(a).abs().max() < 1e-9
    assert engine.energy_residual(b).abs().max() < 1e-9


def test_invalid_inventory_and_configuration_are_rejected():
    with pytest.raises(ValueError):
        ChemicalConfig(diffusion_fraction=2)
    with pytest.raises(ValueError):
        ChemicalConfig(temperature_cycle_amplitude_k=400)
    engine = ChemicalEngine(config())
    with pytest.raises(ValueError):
        engine.initialize(0, {"glycine": -1})
    with pytest.raises(ValueError):
        engine.initialize_from_elements(0, {"Fe": 1})


def test_heterogeneous_permeability_is_conservative_and_blocks_walls():
    engine = ChemicalEngine(config(max_reaction_fraction=0.0, photon_input_j=0.0))
    state = engine.initialize(0, {"glycine": [90.0, 30.0]})
    before = state.amount_umol.sum(dim=(1, 2)).clone()
    wall_amounts = state.amount_umol[:, 0, 0].clone()
    permeability = torch.ones(2, 3, 3, dtype=torch.float64)
    permeability[:, 0, 0] = 0
    for _ in range(8):
        engine.step(state, permeability=permeability)
    assert torch.allclose(state.amount_umol.sum(dim=(1, 2)), before, atol=1e-12, rtol=0)
    assert torch.equal(state.amount_umol[:, 0, 0], wall_amounts)
    assert state.amount_umol.min() >= 0


def test_per_planet_budgets_and_temperatures_preserve_separate_ledgers():
    engine = ChemicalEngine(config())
    state = engine.initialize_from_elements(0, [{"H": 100, "O": 50}, {"C": 10, "O": 20}],
                                            temperature_k=torch.tensor([250., 450.]))
    assert torch.equal(engine.temperature(state)[:, 0, 0], torch.tensor([250., 450.], dtype=torch.float64))
    for _ in range(5):
        engine.step(state, temperature_k=torch.tensor([200., 700.]),
                    illumination=torch.tensor([0., 1.]))
    assert engine.energy_residual(state).abs().max() < 1e-8
    assert torch.allclose(engine.elemental_totals(state), state.initial_elements_umol, atol=1e-12, rtol=0)
    assert state.cumulative_photon_input_j[0] == 0
    assert state.cumulative_photon_input_j[1] > 0


def test_energy_exhaustion_cannot_make_negative_resources():
    engine = ChemicalEngine(config(grid_size=1, batch_size=1, dt_s=1e6,
                                   max_reaction_fraction=1.0, photon_input_j=0.0))
    state = engine.initialize(2, {"H2O": 1e10, "CO2": 1e10, "H2": 1e10})
    state.photon_j.fill_(0.17)
    for _ in range(20):
        engine.step(state)
        assert state.amount_umol.min() >= 0
        assert state.photon_j.min() >= 0
        assert state.heat_j.min() >= 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA/ROCm device in this worker")
def test_gpu_closes_ledgers():
    engine = ChemicalEngine(config(), "cuda")
    state = primordial(engine)
    for _ in range(30):
        engine.step(state)
    assert torch.allclose(engine.elemental_totals(state), state.initial_elements_umol,
                          rtol=1e-11, atol=1e-8)
    assert engine.energy_residual(state).abs().max() < 1e-7
