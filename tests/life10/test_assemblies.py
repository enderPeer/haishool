import copy

import torch

from haishool.life10.assemblies import Assemblies, AssemblyConfig
from haishool.life10.chemistry import ChemicalConfig, ChemicalEngine, SPECIES_INDEX


def prepared():
    engine = ChemicalEngine(ChemicalConfig(grid_size=2, photon_input_j=0))
    state = engine.initialize(3, {"glycine": 8, "alanine": 8, "H2O": 50})
    state.photon_j.fill_(10)
    pool = Assemblies(1, 2, 8, AssemblyConfig(join_probability=.5, hydrolysis_probability=.03))
    return engine, state, pool


def test_assembly_hydrolysis_conserve_elements_and_energy():
    engine, state, pool = prepared()
    el = engine.elemental_totals(state)
    energy = engine.energy_total(state)
    for _ in range(60):
        pool.step(state)
        state.steps += 1
    assert pool.events["joined"] > 0 and pool.events["hydrolysed"] > 0
    assert torch.allclose(engine.elemental_totals(state) + pool.elemental_totals(), el, atol=1e-10, rtol=0)
    assert torch.allclose(engine.energy_total(state) + pool.energy_total(), energy, atol=1e-9, rtol=0)
    assert state.amount_umol.min() >= 0 and state.photon_j.min() >= 0


def test_empty_world_never_inserts_polymers():
    engine = ChemicalEngine(ChemicalConfig(grid_size=2))
    state = engine.initialize(0, {})
    pool = Assemblies(1, 2, 0)
    for _ in range(100):
        pool.step(state)
    assert not pool.chains


def test_dark_world_cannot_pay_for_condensation():
    _, state, pool = prepared()
    state.photon_j.zero_()
    for _ in range(20):
        pool.step(state)
    assert not pool.chains


def test_cpu_checkpoint_exact_continuation():
    engine, state, pool = prepared()
    for _ in range(8):
        pool.step(state)
    saved = copy.deepcopy(pool.state_dict())
    restored = Assemblies.from_dict(saved)
    other = copy.deepcopy(state)
    for _ in range(8):
        pool.step(state)
        restored.step(other)
    assert pool.state_dict() == restored.state_dict()
    assert torch.equal(state.amount_umol, other.amount_umol)


def test_capacity_is_reported_without_deleting_material():
    engine, state, pool = prepared()
    pool.config = AssemblyConfig(join_probability=1, max_chains=1)
    initial = engine.elemental_totals(state)
    for _ in range(20):
        pool.step(state)
    assert pool.summary()["censored"]
    assert len(pool.chains) <= 1
    assert torch.allclose(engine.elemental_totals(state) + pool.elemental_totals(), initial)


class FixedRng:
    """Force physical events, without depending on a lucky long random rollout."""

    def __init__(self, value=.75, values=()):
        self.value = value
        self.values = iter(values)
        self.choice_weights = []

    def random(self):
        return next(self.values, self.value)

    def choice(self, values):
        return values[0]

    def choices(self, values, *, weights):
        self.choice_weights.append(list(weights))
        return [values[0]]

    def randrange(self, *args):
        return args[0] if len(args) > 1 else 0


def one_cell(config):
    engine = ChemicalEngine(ChemicalConfig(grid_size=1, photon_input_j=0))
    state = engine.initialize(3, {"glycine": 8, "alanine": 8, "H2O": 50})
    state.photon_j.fill_(10)
    return state, Assemblies(1, 1, 8, config)


def test_hydrolysed_scaffold_cannot_keep_guiding_a_surviving_chain():
    state, pool = one_cell(AssemblyConfig(join_probability=0, hydrolysis_probability=.5,
                                          adsorption_probability=0, release_probability=0,
                                          hop_probability=0))
    parent = pool._new(0, (0, 0), [0, 1, 1], 0)
    child = pool._new(0, (0, 0), [1, 0], 0)
    child["scaffold"] = parent["id"]
    # Skip nucleation, cut the three-residue parent, leave the child intact.
    pool.rng = FixedRng(values=(.75, 0, .75))
    pool.step(state)
    assert parent not in pool.chains
    assert child in pool.chains
    live = {row["id"] for row in pool.chains}
    assert all(row["scaffold"] is None or row["scaffold"] in live for row in pool.chains)
    assert child["scaffold"] is None


def test_release_removes_template_affinity_in_the_same_step():
    state, pool = one_cell(AssemblyConfig(join_probability=1, hydrolysis_probability=0,
                                          adsorption_probability=0, release_probability=1,
                                          hop_probability=0, max_length=3))
    parent = pool._new(0, (0, 0), [0, 1, 0], 0)
    child = pool._new(0, (0, 0), [1, 0], 0)
    child["scaffold"] = parent["id"]
    pool.rng = FixedRng()
    pool.step(state)
    assert child["scaffold"] is None
    assert len(pool.rng.choice_weights) == 1
    # Both freely dissolved residue inventories were equal. A released
    # scaffold must not leave an invisible preference in the binding weights.
    assert pool.rng.choice_weights[0][0] == pool.rng.choice_weights[0][1]


def test_length_capacity_does_not_make_a_polymer_immobile():
    engine = ChemicalEngine(ChemicalConfig(grid_size=2, photon_input_j=0))
    state = engine.initialize(0, {})
    pool = Assemblies(1, 2, 0, AssemblyConfig(join_probability=0, hydrolysis_probability=0,
                                            adsorption_probability=0, hop_probability=1,
                                            max_length=2))
    chain = pool._new(0, (0, 0), [0, 1], 0)
    pool.rng = FixedRng()
    pool.step(state)
    assert chain["cell"] == [1, 0]


def test_scaffold_association_cannot_span_different_cells():
    engine = ChemicalEngine(ChemicalConfig(grid_size=2, photon_input_j=0))
    state = engine.initialize(0, {})
    pool = Assemblies(1, 2, 0, AssemblyConfig(join_probability=0, hydrolysis_probability=0,
                                            adsorption_probability=0, release_probability=0,
                                            hop_probability=1))
    parent = pool._new(0, (0, 0), [0, 1, 1], 0)
    child = pool._new(0, (0, 0), [1, 0], 0)
    child["scaffold"] = parent["id"]
    pool.rng = FixedRng()
    pool.step(state)
    assert child["scaffold"] is None or parent["cell"] == child["cell"]


def test_compartment_detection_does_not_create_any_material():
    engine, state, pool = prepared()
    state.amount_umol[0, 0, 0, SPECIES_INDEX["hexanoic_acid"]] = .1
    before = state.amount_umol.clone()
    mask = pool.compartment_mask(state)
    assert mask[0, 0, 0]
    assert torch.equal(before, state.amount_umol)
    control = Assemblies(1, 2, 8, profile="no_compartments")
    assert not control.compartment_mask(state).any()
