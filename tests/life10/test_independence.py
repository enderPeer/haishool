"""Independent physical planets must not share random draws or storage capacity."""
import copy
from dataclasses import fields, replace

import pytest
import torch

from haishool.life10.assemblies import Assemblies, AssemblyConfig
from haishool.life10.chemistry import ChemicalConfig, ChemicalEngine, ChemicalState
from haishool.life10.compartments import Compartments, CompartmentConfig


def initialized(keys, config=None):
    cfg = config or ChemicalConfig(batch_size=len(keys), grid_size=2, photon_input_j=0)
    engine = ChemicalEngine(cfg)
    rows = []
    for key in keys:
        single = ChemicalEngine(replace(cfg, batch_size=1))
        state = single.initialize(key, {"glycine": 8 + key % 3, "alanine": 10 + key % 2,
                                        "H2O": 80, "hexanoic_acid": 4 + key % 5})
        state.photon_j.fill_(10)
        rows.append(state)
    data = {}
    for field in fields(ChemicalState):
        values = [getattr(row, field.name) for row in rows]
        data[field.name] = torch.cat(values, 0) if isinstance(values[0], torch.Tensor) else values[0]
    return engine, ChemicalState(**data)


def normalized(pool, batch, collection):
    rows = []
    for row in getattr(pool, collection):
        if row["batch"] != batch:
            continue
        item = {**row, "batch": 0}
        for key, value in item.items():
            if isinstance(value, torch.Tensor):
                item[key] = value.tolist()
        rows.append(item)
    return sorted(rows, key=lambda row: row["id"])


def snapshot(state, pool, compartments, batch):
    return {"amounts": state.amount_umol[batch].tolist(), "heat": state.heat_j[batch].tolist(),
            "photons": state.photon_j[batch].tolist(),
            "chains": normalized(pool, batch, "chains"),
            "compartments": normalized(compartments, batch, "compartments"),
            "chain_events": dict(pool.events_by_planet[batch]),
            "compartment_events": dict(compartments.events_by_planet[batch]),
            "next_chain": pool.next_ids[batch], "next_compartment": compartments.next_ids[batch],
            "assembly_rng": pool.rngs[batch].getstate(),
            "compartment_rng": compartments.rngs[batch].getstate()}


def run(keys, steps=25):
    engine, state = initialized(keys)
    pool = Assemblies(len(keys), 2, keys, AssemblyConfig(join_probability=.45,
                                                       hydrolysis_probability=.04,
                                                       adsorption_probability=.6,
                                                       hop_probability=.5,
                                                       max_length=8, max_chains=12))
    compartments = Compartments(len(keys), 2, keys,
                                CompartmentConfig(nucleation_probability=1,
                                                  growth_fraction=.3, fission_probability=.3,
                                                  max_compartments=4))
    initial_elements = engine.elemental_totals(state).clone()
    initial_energy = engine.energy_total(state).clone()
    result = {key: [] for key in keys}
    for step in range(steps):
        engine.step(state, illumination=0)
        pool.step(state)
        compartments.step(state, engine)
        elements = (engine.elemental_totals(state) + pool.elemental_totals()
                    + compartments.elemental_totals())
        energy = engine.energy_total(state) + pool.energy_total() + compartments.energy_total()
        assert torch.allclose(elements, initial_elements, rtol=1e-12, atol=1e-9)
        assert torch.allclose(energy, initial_energy + state.cumulative_input_j, rtol=1e-12, atol=1e-9)
        assert state.amount_umol.min() >= 0 and state.heat_j.min() >= 0
        if step % 5 == 0 or step == steps - 1:
            for b, key in enumerate(keys):
                result[key].append(snapshot(state, pool, compartments, b))
    return result


def test_planet_solo_matches_batch_and_reversed_batch_at_every_sample():
    keys = [41, 72, 99]
    batch = run(keys)
    reordered = run(list(reversed(keys)))
    for key in keys:
        solo = run([key])[key]
        assert solo == batch[key] == reordered[key]
    # This is an active sparse-event test, not empty identical worlds.
    assert any(s["chain_events"]["joined"] for rows in batch.values() for s in rows)
    assert any(s["compartment_events"]["nucleated"] for rows in batch.values() for s in rows)


class FixedRng:
    def random(self):
        return .25

    def choices(self, values, *, weights):
        return [values[0]]

    def choice(self, values):
        return values[0]

    def randrange(self, *args):
        return args[0] if len(args) > 1 else 0


def test_one_planets_assembly_capacity_does_not_use_anothers_slots():
    engine, state = initialized([1, 2])
    before = engine.elemental_totals(state).clone()
    pool = Assemblies(2, 2, [1, 2], AssemblyConfig(join_probability=1,
                                                  hydrolysis_probability=0, max_chains=1))
    pool.rngs = [FixedRng(), FixedRng()]
    pool.step(state)
    assert pool.counts == [1, 1]
    assert len(pool.chains) == 2
    assert all(row["events"]["capacity_hits"] > 0 for row in pool.summary()["per_planet"])
    assert all(row["censored"] for row in pool.summary()["per_planet"])
    assert torch.allclose(engine.elemental_totals(state) + pool.elemental_totals(), before)


def test_one_planets_compartment_capacity_does_not_use_anothers_slots():
    engine, state = initialized([1, 2])
    before = engine.elemental_totals(state).clone()
    cfg = CompartmentConfig(nucleation_probability=1, growth_fraction=.5,
                            fission_probability=1, max_compartments=1)
    pool = Compartments(2, 2, [1, 2], cfg)
    pool.rngs = [FixedRng(), FixedRng()]
    pool.step(state)
    assert pool.counts == [1, 1]
    assert len(pool.compartments) == 2
    assert all(row["events"]["capacity_hits"] > 0 for row in pool.summary()["per_planet"])
    assert torch.allclose(engine.elemental_totals(state) + pool.elemental_totals(), before)


def test_local_polymer_ids_and_scaffolds_do_not_cross_planets():
    engine, state = initialized([1, 2])
    pool = Assemblies(2, 2, [1, 2], AssemblyConfig(join_probability=0,
                                                  hydrolysis_probability=0,
                                                  adsorption_probability=0,
                                                  release_probability=0, hop_probability=0))
    parent_a = pool._new(0, (0, 0), [0, 0, 0], 0)
    parent_b = pool._new(1, (1, 1), [1, 1, 1], 0)
    child_a = pool._new(0, (0, 0), [0, 1], 0)
    child_b = pool._new(1, (1, 1), [1, 0], 0)
    assert parent_a["id"] == parent_b["id"] == 1
    child_a["scaffold"] = parent_a["id"]
    child_b["scaffold"] = parent_b["id"]
    pool.step(state)
    assert child_a["scaffold"] == child_b["scaffold"] == 1


def test_batched_checkpoint_resumes_every_planets_exact_history():
    engine, state = initialized([41, 72])
    pool = Assemblies(2, 2, [41, 72], AssemblyConfig(join_probability=.5))
    compartments = Compartments(2, 2, [41, 72], CompartmentConfig(nucleation_probability=1))
    for _ in range(8):
        engine.step(state, illumination=0)
        pool.step(state)
        compartments.step(state, engine)
    restored_engine, other = ChemicalEngine.load_checkpoint(engine.checkpoint(state))
    restored_pool = Assemblies.from_dict(copy.deepcopy(pool.state_dict()))
    restored_comp = Compartments.from_dict(copy.deepcopy(compartments.state_dict()))
    for _ in range(10):
        for eng, st, ass, comp in ((engine, state, pool, compartments),
                                   (restored_engine, other, restored_pool, restored_comp)):
            eng.step(st, illumination=0)
            ass.step(st)
            comp.step(st, eng)
        for b in range(2):
            assert snapshot(state, pool, compartments, b) == snapshot(other, restored_pool, restored_comp, b)


@pytest.mark.parametrize("cls", [Assemblies, Compartments])
def test_batched_component_requires_explicit_physical_seed_list(cls):
    with pytest.raises(ValueError, match="one seed per planet"):
        cls(2, 2, 41)


def test_legacy_single_planet_component_states_keep_exact_draws():
    engine, state = initialized([41])
    pool = Assemblies(1, 2, 41, AssemblyConfig(join_probability=.5))
    compartments = Compartments(1, 2, 41, CompartmentConfig(nucleation_probability=1))
    for _ in range(4):
        engine.step(state, illumination=0)
        pool.step(state)
        compartments.step(state, engine)
    saved_pool, saved_comp = copy.deepcopy(pool.state_dict()), copy.deepcopy(compartments.state_dict())
    for saved in (saved_pool, saved_comp):
        saved["next_id"] = saved.pop("next_ids")[0]
        saved["rng"] = saved.pop("rngs")[0]
        for field in ("stream_seeds", "events_by_planet", "schema"):
            saved.pop(field)
    other_engine, other = ChemicalEngine.load_checkpoint(engine.checkpoint(state))
    other_pool = Assemblies.from_dict(saved_pool)
    other_comp = Compartments.from_dict(saved_comp)
    for _ in range(6):
        for eng, st, ass, comp in ((engine, state, pool, compartments),
                                   (other_engine, other, other_pool, other_comp)):
            eng.step(st, illumination=0)
            ass.step(st)
            comp.step(st, eng)
        assert snapshot(state, pool, compartments, 0) == snapshot(other, other_pool, other_comp, 0)


@pytest.mark.parametrize("cls", [Assemblies, Compartments])
def test_legacy_shared_rng_batch_state_is_not_silently_reinterpreted(cls):
    with pytest.raises(ValueError, match="frozen original implementation"):
        cls.from_dict({"batch": 2})
