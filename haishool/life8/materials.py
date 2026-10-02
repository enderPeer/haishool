"""Small inspectable material/work/heat model, without named fabrication recipes.

All quantities use simulation units. Preset values are assumptions, not measured
SI properties. ``heat_capacity`` is *specific* heat capacity, so sensible energy
relative to the model's zero is mass * heat_capacity * temperature. Work supplied
to assembling or dismantling is dissipated as heat. Strike work is split into
delivered work, tool heat and explicitly reported untracked dissipation.

Wear reduces function/durability, not mass: debris and microscopic abrasion are
not simulated. Combining stores the original material masses and component
records; dismantling conserves both that accounting and sensible thermal energy.
Assemblies are generic mechanical joins, not chemistry, named tools or recipes.
"""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping


MATERIALS = {
    "wood": {"hardness": 2.0, "friction": 0.55, "heat_capacity": 1.7},
    "stone": {"hardness": 8.0, "friction": 0.75, "heat_capacity": 0.85},
    "fiber": {"hardness": 0.8, "friction": 0.90, "heat_capacity": 1.3},
    "clay": {"hardness": 1.0, "friction": 0.80, "heat_capacity": 1.0},
    "metal": {"hardness": 9.0, "friction": 0.40, "heat_capacity": 0.5},
}
PROPERTY_VERSION = "material-work-heat-v1"


def _number(value, name, minimum=0.0, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    value = float(value)
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} outside allowed range")
    return value


def _get(item, key, default=None):
    return item.get(key, default) if isinstance(item, Mapping) else getattr(item, key, default)


def material_properties(material: str, mass: float = 1.0) -> dict:
    """Fresh JSON-only properties for a raw material item; never shared state."""
    mass = _number(mass, "mass", minimum=1e-12)
    if material not in MATERIALS:
        raise ValueError(f"unknown raw material {material!r}")
    return {**MATERIALS[material], "version": PROPERTY_VERSION, "durability": 100.0,
            "max_durability": 100.0, "composition": {material: mass}}


def _item(item) -> dict:
    """Detached nonidentity fields of an engine artifact or a dict record."""
    if item is None:
        raise ValueError("an item is required")
    mass = _number(_get(item, "mass"), "mass", minimum=1e-12)
    material = _get(item, "material")
    if not isinstance(material, str) or not material:
        raise ValueError("material must be a nonempty label")
    properties = copy.deepcopy(_get(item, "properties") or material_properties(material, mass))
    if not isinstance(properties, dict):
        raise ValueError("properties must be a dictionary")
    for name in ("hardness", "friction"):
        properties[name] = _number(properties.get(name), name)
    properties["heat_capacity"] = _number(properties.get("heat_capacity"), "heat capacity", minimum=1e-12)
    properties["max_durability"] = _number(properties.get("max_durability", 100), "max durability", minimum=1e-12)
    properties["durability"] = _number(properties.get("durability"), "durability", maximum=properties["max_durability"])
    composition = properties.get("composition")
    if not isinstance(composition, dict) or not composition:
        raise ValueError("item requires material composition accounting")
    for name, quantity in composition.items():
        if name not in MATERIALS:
            raise ValueError(f"unknown constituent material {name!r}")
        composition[name] = _number(quantity, "constituent mass", minimum=1e-12)
    if not math.isclose(math.fsum(composition.values()), mass, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError("constituent masses do not match item mass")
    return {"material": material, "mass": mass,
            "sharpness": _number(_get(item, "sharpness", 0.0), "sharpness", maximum=1.0),
            "bond_strength": _number(_get(item, "bond_strength", 0.0), "bond strength", maximum=1.0),
            "temperature": _number(_get(item, "temperature", 20.0), "temperature"),
            "components": list(_get(item, "components", [])), "properties": properties}


def validate_item(item) -> dict:
    """Check tracked physical invariants; return detached normalized fields."""
    return _item(item)


def total_heat_capacity(item) -> float:
    state = _item(item)
    return state["mass"] * state["properties"]["heat_capacity"]


def thermal_energy(item) -> float:
    state = _item(item)
    return state["mass"] * state["properties"]["heat_capacity"] * state["temperature"]


def heat_item(item, energy: float) -> dict:
    """Add/remove actual sensible heat; removing more than available is invalid."""
    state = _item(item)
    if isinstance(energy, bool) or not isinstance(energy, (int, float)) or not math.isfinite(energy):
        raise ValueError("heat energy must be finite")
    capacity = state["mass"] * state["properties"]["heat_capacity"]
    temperature = state["temperature"] + float(energy) / capacity
    if temperature < -1e-12:
        raise ValueError("cannot extract more sensible energy than the item contains")
    return {"temperature": max(0.0, temperature), "energy": float(energy), "heat_capacity_total": capacity}


def exchange_heat(a, b, conductance: float = 0.1) -> dict:
    """One heat-contact tick, capped at equilibrium so it cannot overshoot."""
    first, second = _item(a), _item(b)
    conductance = _number(conductance, "conductance")
    ca = first["mass"] * first["properties"]["heat_capacity"]
    cb = second["mass"] * second["properties"]["heat_capacity"]
    difference = second["temperature"] - first["temperature"]
    maximum = difference / (1.0 / ca + 1.0 / cb)
    amount = math.copysign(min(abs(conductance * difference), abs(maximum)), difference)
    return {"temperature_a": first["temperature"] + amount / ca,
            "temperature_b": second["temperature"] - amount / cb,
            "energy_a": amount, "energy_b": -amount}


def strike(tool, target_hardness: float, effort: float, manipulation: float = 1.0) -> dict:
    """Convert paid body work into damage, wear and heat with no tool-name bonus.

    More mass initially helps momentum transfer but eventually exceeds the user's
    capacity. Friction affects grip, hardness affects transfer against the target,
    and wear/joint quality reduce effective transfer. Sharpness concentrates
    delivered work into a smaller damage area; it does not create energy.
    The target's structural damage is an abstract amount, not food mass.
    """
    target_hardness = _number(target_hardness, "target hardness")
    effort = _number(effort, "effort")
    manipulation = _number(manipulation, "manipulation")
    body_use = manipulation / (1.0 + manipulation)
    if tool is None:
        delivered = effort * 0.22 * body_use
        return {"damage": delivered / (1.0 + target_hardness), "work": effort, "wear": 0.0,
                "properties": None, "temperature": None, "sharpness": None,
                "delivered_work": delivered, "tool_heat": 0.0,
                "untracked_dissipation": effort - delivered}
    state = _item(tool)
    properties = state["properties"]
    mass, hardness, friction = state["mass"], properties["hardness"], properties["friction"]
    grip = friction / (0.20 + friction)
    load = mass / (mass + 0.50) * manipulation / (manipulation + 0.15 * mass) if manipulation else 0.0
    transmission = hardness / (hardness + target_hardness + 0.20)
    condition = properties["durability"] / properties["max_durability"]
    joint = 0.4 + 0.6 * state["bond_strength"] if properties.get("parts") else 1.0
    delivered = effort * grip * load * transmission * condition * joint
    damage = delivered * (1.0 + state["sharpness"]) / (1.0 + target_hardness)
    contact_work = effort * grip * load
    wear = min(properties["durability"], contact_work * 0.12 * (target_hardness + 0.20) / (hardness + 0.20))
    properties["durability"] -= wear
    sharpness = max(0.0, state["sharpness"] - wear / properties["max_durability"] * 0.3)
    heat = 0.4 * (effort - delivered) * grip * load
    temperature = heat_item(state, heat)["temperature"]
    return {"damage": damage, "work": effort, "wear": wear, "properties": properties,
            "temperature": temperature, "sharpness": sharpness, "delivered_work": delivered,
            "tool_heat": heat, "untracked_dissipation": effort - delivered - heat}


def combine_items(a, b, effort: float) -> dict:
    """Join any two materials using work; preserve constituent masses and heat.

    Return new nonidentity artifact fields. The world must remove/consume the
    two active inputs and create exactly one output. Nested component records
    are accounting/provenance, not extra active material.
    """
    first, second = _item(a), _item(b)
    effort = _number(effort, "effort")
    first_id, second_id = _get(a, "id"), _get(b, "id")
    if a is b or (first_id is not None and first_id == second_id):
        raise ValueError("cannot combine an item with itself")
    if effort == 0:
        raise ValueError("joining requires positive work")
    mass = first["mass"] + second["mass"]
    composition = {}
    for part in (first, second):
        for name, quantity in part["properties"]["composition"].items():
            composition[name] = composition.get(name, 0.0) + quantity
    properties = {name: math.fsum(part["mass"] * part["properties"][name] for part in (first, second)) / mass
                  for name in ("hardness", "friction", "heat_capacity", "durability", "max_durability")}
    properties.update(version=PROPERTY_VERSION, composition=composition, parts=[first, second],
                      assembled_durability=properties["durability"])
    bond = (-math.expm1(-effort / mass)) * min(1.0, (first["properties"]["friction"] + second["properties"]["friction"]) / 1.2)
    capacity = mass * properties["heat_capacity"]
    energy = math.fsum(thermal_energy(part) for part in (first, second)) + effort
    components = [identifier for identifier in (first_id, second_id) if identifier is not None]
    item = {"material": "aggregate", "mass": mass, "temperature": energy / capacity,
            "sharpness": max(first["sharpness"], second["sharpness"]) * (0.5 + 0.5 * bond),
            "bond_strength": bond, "components": components, "properties": properties}
    properties["assembled_sharpness"] = item["sharpness"]
    return {"item": item, "work": effort, "heat_energy": effort}


def dismantle_item(item, effort: float) -> dict:
    """Separate a stored join if work beats its bond; never mint raw resources."""
    state = _item(item)
    effort = _number(effort, "effort")
    parts = copy.deepcopy(state["properties"].get("parts", []))
    required = state["bond_strength"] * state["mass"] * 0.20
    heated = heat_item(state, effort)
    state["temperature"] = heated["temperature"]
    if not parts or effort < required or effort == 0:
        return {"success": False, "items": [], "item": state, "work": effort,
                "required_work": required, "heat_energy": effort}
    parts = [_item(part) for part in parts]
    capacity = math.fsum(total_heat_capacity(part) for part in parts)
    if not math.isclose(capacity, total_heat_capacity(state), rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError("stored component heat capacities disagree with assembly")
    if not math.isclose(math.fsum(part["mass"] for part in parts), state["mass"], rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError("stored component masses disagree with assembly")
    composed = {}
    for part in parts:
        for name, quantity in part["properties"]["composition"].items():
            composed[name] = composed.get(name, 0.0) + quantity
    if set(composed) != set(state["properties"]["composition"]) or any(
        not math.isclose(value, state["properties"]["composition"][name], rel_tol=1e-10, abs_tol=1e-10)
        for name, value in composed.items()
    ):
        raise ValueError("stored component composition disagrees with assembly")
    original = state["properties"].get("assembled_durability", state["properties"]["durability"])
    retained = min(1.0, state["properties"]["durability"] / original) if original else 0.0
    edge_loss = max(0.0, state["properties"].get("assembled_sharpness", state["sharpness"]) - state["sharpness"])
    for part in parts:
        part["temperature"] = state["temperature"]
        part["properties"]["durability"] *= retained
        part["sharpness"] = max(0.0, part["sharpness"] - edge_loss)
    return {"success": True, "items": parts, "item": None, "work": effort,
            "required_work": required, "heat_energy": effort}
