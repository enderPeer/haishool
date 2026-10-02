import copy
import json
import math
from types import SimpleNamespace

import pytest

from haishool.life8.materials import (MATERIALS, combine_items, dismantle_item, exchange_heat, heat_item,
                                  material_properties, strike, thermal_energy, total_heat_capacity, validate_item)


def item(material="stone", mass=1.0, temperature=20.0, identifier=1, sharpness=.3):
    return {"id": identifier, "material": material, "mass": mass, "temperature": temperature,
            "sharpness": sharpness, "bond_strength": 0.0, "components": [],
            "properties": material_properties(material, mass)}


def apply_strike(source, result):
    updated = copy.deepcopy(source)
    for key in ("properties", "temperature", "sharpness"):
        updated[key] = result[key]
    return updated


def test_raw_items_are_independent_and_properties_are_required():
    first, second = material_properties("stone", 2), material_properties("stone", 2)
    first["composition"]["stone"] = 1
    assert second["composition"] == {"stone": 2}
    assert {"hardness", "friction", "heat_capacity", "durability", "max_durability"} <= set(second)
    assert MATERIALS["stone"]["hardness"] == 8
    artifact = SimpleNamespace(**item())
    assert validate_item(artifact)["mass"] == 1
    assert strike(artifact, 2, 1)["damage"] > 0


@pytest.mark.parametrize("name", MATERIALS)
def test_heat_changes_temperature_by_actual_capacity(name):
    source = item(name, mass=2.5)
    capacity = total_heat_capacity(source)
    old = thermal_energy(source)
    warmed = {**source, **heat_item(source, 7)}
    assert warmed["temperature"] == pytest.approx(20 + 7 / capacity)
    assert thermal_energy(warmed) - old == pytest.approx(7)
    cooled = {**warmed, **heat_item(warmed, -7)}
    assert cooled["temperature"] == pytest.approx(source["temperature"])
    with pytest.raises(ValueError, match="extract"):
        heat_item(source, -old - 1)


def test_heat_capacity_and_mass_both_limit_temperature_rise():
    light = item("stone", mass=1)
    heavy = item("stone", mass=3)
    high_capacity = copy.deepcopy(light)
    high_capacity["properties"]["heat_capacity"] *= 2
    assert heat_item(light, 10)["temperature"] > heat_item(high_capacity, 10)["temperature"] > heat_item(heavy, 10)["temperature"]


@pytest.mark.parametrize("conductance", [0, .05, 1, 1e6])
def test_contact_heat_is_conserved_and_cannot_overshoot(conductance):
    hot, cold = item("wood", 3, 80), item("stone", 2, 10, identifier=2)
    result = exchange_heat(hot, cold, conductance)
    before = thermal_energy(hot) + thermal_energy(cold)
    after = thermal_energy({**hot, "temperature": result["temperature_a"]}) + thermal_energy({**cold, "temperature": result["temperature_b"]})
    assert before == pytest.approx(after, abs=1e-10)
    assert result["energy_a"] == -result["energy_b"]
    assert 10 <= result["temperature_b"] <= result["temperature_a"] <= 80
    if conductance >= 1e6:
        assert result["temperature_a"] == pytest.approx(result["temperature_b"])


def test_assembly_and_dismantling_preserve_material_and_heat_with_work():
    a, b = item("stone", 2, 60), item("wood", 1, 10, identifier=2)
    untouched = copy.deepcopy((a, b))
    assembled = combine_items(a, b, effort=3)
    composite = assembled["item"]
    assert composite["mass"] == 3
    assert composite["properties"]["composition"] == {"stone": 2, "wood": 1}
    assert thermal_energy(composite) == pytest.approx(thermal_energy(a) + thermal_energy(b) + 3)
    assert assembled["work"] == assembled["heat_energy"] == 3
    assert (a, b) == untouched
    recovered = dismantle_item(composite, effort=2)
    assert recovered["success"] and len(recovered["items"]) == 2
    assert sum(part["mass"] for part in recovered["items"]) == composite["mass"]
    assert sum(thermal_energy(part) for part in recovered["items"]) == pytest.approx(thermal_energy(composite) + 2)
    assert recovered["items"][0]["properties"]["composition"] == {"stone": 2}
    assert recovered["items"][1]["properties"]["composition"] == {"wood": 1}
    json.dumps(recovered, allow_nan=False)


def test_nested_assembly_preserves_original_composition_and_components():
    first = combine_items(item("wood", 1), item("stone", 2, identifier=2), effort=4)["item"]
    first["id"] = 3
    combined = combine_items(first, item("fiber", .5, identifier=4), effort=3)["item"]
    assert combined["mass"] == 3.5
    assert combined["properties"]["composition"] == {"wood": 1, "stone": 2, "fiber": .5}
    split = dismantle_item(combined, 10)["items"]
    second_split = dismantle_item(split[0], 10)["items"]
    raw = second_split + [split[1]]
    assert sum(part["mass"] for part in raw) == 3.5
    assert sorted(part["material"] for part in raw) == ["fiber", "stone", "wood"]


def test_insufficient_dismantling_work_does_not_duplicate_material():
    assembled = combine_items(item(), item("wood", identifier=2), effort=20)["item"]
    failed = dismantle_item(assembled, .0001)
    assert not failed["success"] and failed["items"] == []
    assert failed["item"]["mass"] == assembled["mass"]
    assert thermal_energy(failed["item"]) == pytest.approx(thermal_energy(assembled) + .0001)
    assert not dismantle_item(item(), 10)["success"]
    with pytest.raises(ValueError, match="positive work"):
        combine_items(item(), item(identifier=2), 0)
    same = item()
    with pytest.raises(ValueError, match="itself"):
        combine_items(same, same, 1)


@pytest.mark.parametrize("property_name,weak,strong", [("hardness", .1, 10), ("friction", .01, 1), ("durability", 1, 100)])
def test_each_tool_property_changes_damage_at_fixed_work(property_name, weak, strong):
    a, b = item(), item()
    a["properties"][property_name], b["properties"][property_name] = weak, strong
    assert strike(a, 3, 2)["damage"] < strike(b, 3, 2)["damage"]


def test_mass_has_a_body_limited_optimum_not_an_unbounded_tool_bonus():
    light, moderate, too_heavy = item(mass=.01), item(mass=2), item(mass=1000)
    damage = [strike(source, 2, 1, manipulation=1)["damage"] for source in (light, moderate, too_heavy)]
    assert damage[1] > damage[0] and damage[1] > damage[2]
    assert strike(moderate, 2, 1, manipulation=2)["damage"] > strike(moderate, 2, 1, manipulation=.1)["damage"]


def test_strike_requires_body_work_and_closes_its_energy_partition():
    source = item(mass=2)
    no_work = strike(source, 2, 0)
    no_control = strike(source, 2, 10, manipulation=0)
    assert no_work["damage"] == no_work["wear"] == 0
    assert no_control["damage"] == no_control["wear"] == no_control["tool_heat"] == 0
    result = strike(source, 2, 10)
    assert result["work"] == pytest.approx(result["delivered_work"] + result["tool_heat"] + result["untracked_dissipation"])
    updated = apply_strike(source, result)
    assert thermal_energy(updated) - thermal_energy(source) == pytest.approx(result["tool_heat"])
    assert updated["mass"] == source["mass"] and updated["properties"]["composition"] == source["properties"]["composition"]
    assert updated["properties"]["durability"] < source["properties"]["durability"]
    assert result["damage"] > strike(None, 2, 10)["damage"]
    assert strike(source, 2, 20)["damage"] == pytest.approx(2 * result["damage"])


def test_wear_is_not_repaired_by_dismantling_and_rejoining():
    a, b = item(), item("wood", identifier=2)
    a["properties"]["durability"] = 40
    combined = combine_items(a, b, 3)["item"]
    before_dismantle = dismantle_item(combined, 2)["items"]
    assert [part["properties"]["durability"] for part in before_dismantle] == [40, 100]
    worn = apply_strike(combined, strike(combined, 10, 10))
    parts = dismantle_item(worn, 2)["items"]
    assert all(part["properties"]["durability"] < original["properties"]["durability"] for part, original in zip(parts, (a, b)))
    assert all(part["sharpness"] < original["sharpness"] for part, original in zip(parts, (a, b)))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1, True])
def test_invalid_physical_inputs_are_not_accepted(bad):
    with pytest.raises(ValueError):
        strike(item(), target_hardness=2, effort=bad)
    with pytest.raises(ValueError):
        material_properties("stone", bad)
    with pytest.raises(ValueError):
        exchange_heat(item(), item(identifier=2), conductance=bad)


def test_corrupt_mass_or_thermal_accounting_is_rejected():
    broken = item()
    broken["properties"]["composition"]["stone"] = 2
    with pytest.raises(ValueError, match="masses"):
        validate_item(broken)
    combined = combine_items(item(), item("wood", identifier=2), 2)["item"]
    combined["properties"]["parts"][0]["properties"]["heat_capacity"] *= 2
    with pytest.raises(ValueError, match="heat capacities"):
        dismantle_item(combined, 3)
