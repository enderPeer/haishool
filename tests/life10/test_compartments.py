"""Chemical exchange and compartment events must preserve both ledgers."""
import torch

from haishool.life10.chemistry import ChemicalConfig, ChemicalEngine, SPECIES_INDEX
from haishool.life10.compartments import CompartmentConfig, Compartments


def setup(inventory=None, **kwargs):
    engine = ChemicalEngine(ChemicalConfig(batch_size=1, grid_size=1, photon_input_j=0,
                                           diffusion_fraction=0, thermal_diffusion_fraction=0))
    state = engine.initialize(0, inventory or {})
    state.photon_j.fill_(10.0)
    cfg = CompartmentConfig(**{"nucleation_probability": 1.0, "growth_fraction": 0.0,
                               "fission_probability": 0.0, **kwargs})
    pool = Compartments(1, 1, 3, cfg)
    return engine, state, pool


def totals(engine, state, pool):
    return (engine.elemental_totals(state) + pool.elemental_totals(),
            engine.energy_total(state) + pool.energy_total())


def assert_conserved(engine, state, pool, before):
    after = totals(engine, state, pool)
    for a, b in zip(after, before):
        assert torch.allclose(a, b, rtol=1e-12, atol=1e-9)
    assert state.amount_umol.min() >= 0
    assert state.photon_j.min() >= 0
    assert state.heat_j.min() >= 0
    assert all(r["cargo_umol"].min() >= 0 for r in pool.compartments)


def test_empty_and_no_compartment_control_never_insert_vesicles():
    engine, state, pool = setup()
    for _ in range(10):
        pool.step(state)
    assert not pool.compartments
    engine, state, pool = setup({"hexanoic_acid": 5})
    pool.profile = "no_compartments"
    before = totals(engine, state, pool)
    for _ in range(10):
        pool.step(state)
    assert not pool.compartments
    assert_conserved(engine, state, pool, before)


def test_nucleation_growth_cargo_exchange_and_fission_close_ledgers():
    engine, state, pool = setup({"hexanoic_acid": 10, "glycine": 5, "H2O": 30},
                                growth_fraction=0.5, fission_probability=1.0)
    before = totals(engine, state, pool)
    pool.step(state)
    assert pool.events["nucleated"] == 1
    assert pool.events["growth_events"] == 1
    assert pool.events["fissions"] == 1
    assert len(pool.compartments) == 2
    assert pool.summary()["cargo_umol"] > 0
    assert_conserved(engine, state, pool, before)
    for _ in range(5):
        pool.step(state)
        assert_conserved(engine, state, pool, before)


def test_dissolution_returns_all_membrane_cargo_and_surface_energy():
    engine, state, pool = setup({"hexanoic_acid": 0.351, "glycine": 10},
                                dissolution_fraction=1.0)
    pool.step(state)
    assert len(pool.compartments) == 1
    # Dilution here is a declared test intervention, not part of normal runs.
    state.amount_umol[..., SPECIES_INDEX["hexanoic_acid"]] = 0
    before = totals(engine, state, pool)
    pool.step(state)
    assert not pool.compartments
    assert pool.events["dissolutions"] == 1
    assert state.amount_umol[..., SPECIES_INDEX["hexanoic_acid"]].sum() > 0
    assert_conserved(engine, state, pool, before)


def test_capacity_censors_fission_without_deleting_material():
    engine, state, pool = setup({"hexanoic_acid": 10, "glycine": 5}, growth_fraction=0.5,
                                fission_probability=1, max_compartments=1)
    before = totals(engine, state, pool)
    for _ in range(5):
        pool.step(state)
    assert len(pool.compartments) == 1
    assert pool.summary()["censored"]
    assert pool.events["capacity_hits"] > 0
    assert_conserved(engine, state, pool, before)


def test_same_catalog_runs_inside_using_existing_heat_and_photons():
    engine, state, pool = setup({"hexanoic_acid": 3, "formaldehyde": 20, "HCN": 20,
                                "H2O": 30}, exchange_fraction=1,
                                volume_fraction_per_membrane_umol=2)
    before = totals(engine, state, pool)
    pool.step(state, engine)
    assert pool.compartments[0]["cargo_umol"][SPECIES_INDEX["glycine"]] > 0
    assert state.reaction_extent_umol.sum() > 0
    assert_conserved(engine, state, pool, before)


def test_checkpoint_continues_identical_event_and_cargo_history():
    engine, state, pool = setup({"hexanoic_acid": 5, "glycine": 10}, growth_fraction=0.3,
                                fission_probability=0.5)
    for _ in range(3):
        pool.step(state, engine)
    restored_engine, restored_state = ChemicalEngine.load_checkpoint(engine.checkpoint(state))
    restored = Compartments.from_dict(pool.state_dict())
    for _ in range(7):
        pool.step(state, engine)
        restored.step(restored_state, restored_engine)
    assert pool.summary() == restored.summary()
    assert torch.equal(state.amount_umol, restored_state.amount_umol)
    assert torch.equal(state.heat_j, restored_state.heat_j)
    assert torch.equal(state.photon_j, restored_state.photon_j)
    for a, b in zip(pool.compartments, restored.compartments):
        assert torch.equal(a["cargo_umol"], b["cargo_umol"])


def test_engineered_dark_lipid_control_forms_using_thermal_energy():
    engine, state, pool = setup({"hexanoic_acid": 10})
    state.photon_j.zero_()
    before = totals(engine, state, pool)
    before_heat = state.heat_j.clone()
    pool.step(state)
    assert len(pool.compartments) == 1
    assert state.photon_j.count_nonzero() == 0
    assert state.heat_j.item() < before_heat.item()
    assert_conserved(engine, state, pool, before)


def test_membrane_surface_energy_requires_available_thermal_energy():
    engine, state, pool = setup({"hexanoic_acid": 10})
    state.heat_j.zero_()
    before = totals(engine, state, pool)
    pool.step(state)
    assert not pool.compartments
    assert_conserved(engine, state, pool, before)


def test_raw_dark_gases_produce_no_precursors_or_membranes():
    engine = ChemicalEngine(ChemicalConfig(grid_size=1, batch_size=1))
    state = engine.initialize_from_elements(7, {"C": 180, "H": 2000, "N": 300, "O": 500})
    pool = Compartments(1, 1, 5)
    before = totals(engine, state, pool)
    for _ in range(30):
        engine.step(state, illumination=0.0)
        pool.step(state, engine)
    for name in ("HCN", "formaldehyde", "glycine", "hexanoic_acid"):
        assert state.amount_umol[..., SPECIES_INDEX[name]].count_nonzero() == 0
    assert not pool.compartments
    assert_conserved(engine, state, pool, before)
