"""The CPU sparse bridge preserves graph laws, ledgers and CPU trajectories."""
import copy

import pytest
import torch

from haishool.life10.assemblies import Assemblies, AssemblyConfig, folded_contacts
from haishool.life10.chemistry import ChemicalConfig, ChemicalEngine, ChemicalState
from haishool.life10.compartments import Compartments, CompartmentConfig


from haishool.life10.__main__ import SparsePhase


def create(device="cpu", profile="normal", batch=2, grid=2):
    engine = ChemicalEngine(ChemicalConfig(batch_size=batch, grid_size=grid,
                                           photon_input_j=1.0, thermostat=True), device)
    # An explicitly engineered conservation/performance fixture, not an origin-of-life run.
    state = engine.initialize(42, {"glycine": 50, "alanine": 30, "hexanoic_acid": 15,
                                    "H2O": 70, "HCN": 30, "formaldehyde": 40})
    pool = Assemblies(batch, grid, [701+i for i in range(batch)],
                      AssemblyConfig(join_probability=.35, hop_probability=.2), profile)
    compartments = Compartments(batch, grid, [1101+i for i in range(batch)],
                                CompartmentConfig(nucleation_probability=.3,
                                                                  fission_probability=.2), profile)
    return engine, state, pool, compartments


def reference_step(engine, state, pool, compartments):
    mask = pool.compartment_mask(state)
    permeability = torch.where(mask, .1, 1.0).to(torch.float64) * compartments.permeability(state)
    catalyst = torch.ones_like(state.heat_j)
    if pool.profile != "no_catalysis":
        for row in pool.chains:
            b, (x, y) = row["batch"], row["cell"]
            catalyst[b, x, y] += pool.config.catalytic_gain * folded_contacts(row["sequence"])
    engine.step(state, catalyst_multiplier=catalyst, permeability=permeability)
    pool.step(state)
    compartments.step(state, engine)
    return mask


@pytest.mark.parametrize("profile", ["normal", "no_catalysis", "no_template", "no_compartments"])
def test_cpu_bridge_is_identical_to_reference(profile):
    engine, state, pool, compartments = create(profile=profile)
    other_engine, other_state, other_pool, other_compartments = create(profile=profile)
    phase = SparsePhase(other_engine, force_bridge=True)
    initial_elements = engine.elemental_totals(state).clone()
    initial_energy = engine.energy_total(state).clone()
    for _ in range(12):
        reference_mask = reference_step(engine, state, pool, compartments)
        mask, el, energy = phase.step(other_state, other_pool, other_compartments)
        assert torch.equal(reference_mask, mask)
        assert torch.equal(state.amount_umol, other_state.amount_umol)
        assert torch.equal(state.heat_j, other_state.heat_j)
        assert torch.equal(state.photon_j, other_state.photon_j)
        assert torch.equal(state.reaction_extent_umol, other_state.reaction_extent_umol)
        assert pool.summary() == other_pool.summary()
        assert compartments.summary() == other_compartments.summary()
        assert pool.state_dict() == other_pool.state_dict()
        for a, b in zip(compartments.compartments, other_compartments.compartments):
            assert torch.equal(a["cargo_umol"], b["cargo_umol"])
        assert torch.allclose(el, initial_elements, rtol=0, atol=1e-8)
        assert torch.allclose(energy, initial_energy + other_state.cumulative_input_j,
                              rtol=0, atol=1e-8)


def test_checkpoint_across_cpu_bridge_keeps_rng_and_state():
    engine, state, pool, compartments = create()
    phase = SparsePhase(engine, force_bridge=True)
    for _ in range(6):
        phase.step(state, pool, compartments)
    restored_engine, restored = ChemicalEngine.load_checkpoint(engine.checkpoint(state))
    restored_pool = Assemblies.from_dict(copy.deepcopy(pool.state_dict()))
    restored_compartments = Compartments.from_dict(compartments.state_dict())
    restored_phase = SparsePhase(restored_engine, force_bridge=True)
    for _ in range(6):
        phase.step(state, pool, compartments)
        restored_phase.step(restored, restored_pool, restored_compartments)
    assert torch.equal(state.amount_umol, restored.amount_umol)
    assert pool.summary() == restored_pool.summary()
    assert compartments.summary() == restored_compartments.summary()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA/ROCm device on this worker")
def test_gpu_bridge_closes_ledgers_and_keeps_cargo_on_cpu():
    engine, state, pool, compartments = create("cuda")
    phase = SparsePhase(engine)
    elements, energy_initial = engine.elemental_totals(state).clone(), engine.energy_total(state).clone()
    for _ in range(12):
        _, el, energy = phase.step(state, pool, compartments)
        assert torch.allclose(el, elements, rtol=1e-10, atol=1e-8)
        assert torch.allclose(energy, energy_initial + state.cumulative_input_j, rtol=1e-10, atol=1e-8)
        assert state.amount_umol.device.type == "cuda"
        assert phase.cpu_state.amount_umol.device.type == "cpu"
        assert all(row["cargo_umol"].device.type == "cpu" for row in compartments.compartments)
