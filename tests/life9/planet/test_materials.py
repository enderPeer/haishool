"""life9 planet materials: reference tables, thermochemistry, element-balanced transforms, props, crust."""
import math

import pytest
import torch

from haishool.life9.planet import materials as m
from haishool.life9.planet.constants import molar_mass
from haishool.truth import reactions
from haishool.truth.formula import parse

# seed 781's chain cloud (stars.cloud of world7 781; PLANET-SPEC section 3)
CLOUD_781 = {"hydrogen": 0.7448380295235888, "helium": 0.25211464188612503, "carbon": 0.0004579459628254515,
             "nitrogen": 0.00013739985655418575, "oxygen": 0.0015035381403957378, "neon": 0.0003167558978065893,
             "magnesium": 0.00018173625135022164, "silicon": 0.00014674574525680933, "iron": 0.00018705420988332646,
             "other": 0.00011615252621356247, "metallicity": 0.0030473285902858833}


# ------------------------------------------------------------------ species and tables
def test_species_and_classes():
    assert len(m.SPECIES) == len(set(m.SPECIES)) == m.S
    assert 28 <= m.S <= 34
    spec_list = ("basalt granite flint sandstone limestone clay sand salt hematite magnetite malachite cassiterite "
                 "native_copper wood plant_fiber resin meat fat bone hide charcoal ash ceramic lime copper tin bronze "
                 "iron steel glass").split()
    assert set(spec_list) <= set(m.SPECIES)
    assert set(m.CRUST_SPECIES) <= set(m.SPECIES)
    assert "pyrite" in m.CRUST_SPECIES and "wood" not in m.CRUST_SPECIES
    assert set(m.CLASS) == set(m.SPECIES) and set(m.CLASS.values()) <= set(m.CLASSES)
    assert all(m.IDX[s] == i for i, s in enumerate(m.SPECIES))


def test_every_row_has_a_source_and_tag():
    for name, table in m.TABLES.items():
        assert set(table) == set(m.SPECIES), name
        for s, v in table.items():
            assert isinstance(v, m.Val) and v.tag in m.TAGS and len(v.source) > 3, (name, s)
            assert math.isfinite(float(v)) and float(v) >= 0, (name, s)
    prov = m.provenance()
    assert all(tag in m.TAGS and src for tag, src in prov.values())
    assert prov["materials.DENSITY.copper"][0] == "reference"
    assert "materials.TRANSFORMS.limestone_to_lime" in prov and "materials.crust_fractions" in prov


def test_val_behaves_as_float_and_pickles():
    import json
    import pickle
    v = m.MELT_K["copper"]
    assert v == 1357.8 and v + 1 == 1358.8 and json.dumps({"x": v}) == '{"x": 1357.8}'
    w = pickle.loads(pickle.dumps(v))
    assert w == v and w.source == v.source and w.tag == v.tag


def test_property_sanity():
    H, rho, melt = m.MOHS, m.DENSITY, m.MELT_K
    assert H["flint"] > H["limestone"] and H["flint"] > H["basalt"] > H["wood"]
    assert H["steel"] > H["iron"] > H["copper"] > H["tin"]
    assert H["basalt"] * 1.1 >= H["flint"]            # a basalt hammer can knap flint (PLANET-SPEC 4)
    assert H["wood"] * 1.1 < H["basalt"]              # a wooden hammer cannot knap basalt
    assert melt["copper"] == pytest.approx(1357.8, abs=0.05) and melt["native_copper"] == melt["copper"]
    assert melt["tin"] == pytest.approx(505.08, abs=0.05) and melt["iron"] == pytest.approx(1811.2, abs=0.05)
    assert melt["bronze"] < melt["copper"] and melt["steel"] < melt["iron"]
    assert rho["copper"] == pytest.approx(8933, rel=0.01) and rho["iron"] == pytest.approx(7870, rel=0.01)
    assert rho["hematite"] > rho["granite"] > rho["wood"] and 2500 < rho["flint"] < 2700
    assert rho["fat"] < 1000 < rho["meat"]
    assert m.BRITTLE["flint"] == 1 and m.BRITTLE["glass"] == 1 and m.BRITTLE["granite"] == 0
    assert m.TOUGHNESS["copper"] > 10 * m.TOUGHNESS["flint"]
    assert m.CP["meat"] > m.CP["wood"] > m.CP["granite"] > m.CP["copper"] > m.CP["tin"]


def test_fuels_and_food():
    E = m.COMBUSTION_J_KG
    assert E["charcoal"] == pytest.approx(393.522e3 / molar_mass("C"), rel=1e-9)   # 32.76 MJ/kg
    assert 17e6 < E["wood"] < 20e6                    # dry wood LHV about 18.5 MJ/kg
    assert 35e6 < E["fat"] < 39.5e6 and E["meat"] == 0 and E["flint"] == 0
    assert all((E[s] > 0) == (m.IGNITION_K[s] < m.NEVER_K) for s in m.SPECIES)
    assert m.FOOD_J_KG["meat"] == pytest.approx(7.0e6, rel=0.02)              # PLANET-SPEC 2.8
    assert m.FOOD_J_KG["fat"] == pytest.approx(39.5e6)
    assert m.DIGESTIBLE["plant"]["meat"] == pytest.approx(0.4) and m.DIGESTIBLE["meat"]["meat"] == pytest.approx(0.9)
    assert m.COOK_GAIN["meat"] == pytest.approx(1.3) and m.FOOD_J_KG["wood"] == 0


@pytest.mark.parametrize("fuel", ["wood", "charcoal", "fat", "resin", "plant_fiber"])
def test_burn_is_element_balanced(fuel):
    b = m.burn(fuel)
    el_in = {e: f * 1.0 for e, f in m.species_elements(fuel).items()}
    for e, f in m.formula_elements("O2").items():
        el_in[e] = el_in.get(e, 0.0) + f * b["O2"]
    el_out = {e: f * b["ash"] for e, f in m.species_elements("ash").items()}
    for g in ("CO2", "H2O"):
        for e, f in m.formula_elements(g).items():
            el_out[e] = el_out.get(e, 0.0) + f * b[g]
    for e in set(el_in) | set(el_out):
        assert el_in.get(e, 0.0) == pytest.approx(el_out.get(e, 0.0), abs=1e-12), (fuel, e)


def test_element_fractions():
    E = torch.tensor(m.ELEMENT_FRACTION, dtype=torch.float64)
    assert E.shape == (m.S, len(m.ELEMENTS))
    assert torch.allclose(E.sum(1), torch.ones(m.S, dtype=torch.float64), atol=1e-12)
    assert m.species_elements("hematite")["Fe"] == pytest.approx(2 * 55.845 / 159.687, rel=1e-6)
    assert m.species_elements("bronze") == pytest.approx({"Cu": 0.89, "Sn": 0.11})
    assert m.species_elements("meat")["N"] > 0 and m.species_elements("bone")["P"] > 0


# ------------------------------------------------------------------ thermochemistry
def test_equilibrium_temperatures():
    assert m.equilibrium_temperature("CaCO3 -> CaO + CO2") == pytest.approx(1110, abs=25)
    assert m.equilibrium_temperature("Fe2O3 + 3C -> 2Fe + 3CO") == pytest.approx(900, abs=25)
    assert m.equilibrium_temperature("SnO2 + 2C -> Sn + 2CO") == pytest.approx(924, abs=10)
    # reactions that need no heat thermodynamically or never go
    assert m.equilibrium_temperature("CaO + CO2 -> CaCO3") == pytest.approx(1118.5, abs=1)  # favoured below it
    dh, ds = m.reaction_thermo("C + O2 -> CO2")
    assert dh == pytest.approx(-393.522e3) and m.equilibrium_temperature("C + O2 -> CO2") == 0.0
    assert m.delta_g("CaCO3 -> CaO + CO2", 1300) < 0 < m.delta_g("CaCO3 -> CaO + CO2", 900)
    with pytest.raises(ValueError):
        m.reaction_thermo("CaCO3 -> CaO")
    assert m.reaction_terms("3Al2Si2O5(OH)4 -> Al6Si2O13 + 4SiO2 + 6H2O(l)")[1][2] == (6.0, "H2O(l)")


def test_thermo_rows_are_balanced_formulas():
    for f, (dh, s, src) in m.THERMO.items():
        parse(f[:-3] if f.endswith("(l)") else f)
        assert src and (s is None or s > 0)
    for el in ("C", "O2", "Fe", "Cu", "Sn"):
        assert m.THERMO[el][0] == 0.0


# ------------------------------------------------------------------ transforms
def _elements(species_kg: dict, gas_kg: dict) -> dict:
    out: dict = {}
    for s, kg in species_kg.items():
        for e, f in m.species_elements(s).items():
            out[e] = out.get(e, 0.0) + f * kg
    for g, kg in gas_kg.items():
        for e, f in m.formula_elements(g).items():
            out[e] = out.get(e, 0.0) + f * kg
    return out


@pytest.mark.parametrize("t", m.TRANSFORMS, ids=lambda t: t.name)
def test_transform_balanced(t):
    assert t.inputs[t.basis] == 1.0
    assert set(t.inputs) | set(t.outputs) <= set(m.SPECIES)
    assert set(t.gas_in) | set(t.gas_out) <= set(m.GASES)
    mass_in = sum(t.inputs.values()) + sum(t.gas_in.values())
    mass_out = sum(t.outputs.values()) + sum(t.gas_out.values())
    assert mass_in == pytest.approx(mass_out, rel=1e-12)
    a, b = _elements(t.inputs, t.gas_in), _elements(t.outputs, t.gas_out)
    for e in set(a) | set(b):
        assert a.get(e, 0.0) == pytest.approx(b.get(e, 0.0), abs=1e-12), e
    assert t.atmosphere in m.ATMOSPHERES and t.days > 0 and t.source
    assert t.min_temp_k == pytest.approx(max(t.t_eq_k or 0.0, t.onset_k))
    if t.reaction is not None:   # the integer equation itself balances atom by atom
        left, right = m.reaction_terms(t.reaction)
        sp = lambda side: [reactions.species(f, int(c)) for c, f in side]
        assert reactions.is_balanced(sp(left), sp(right))
        assert m.is_balanced(t.reaction)


def test_unique_balances_match_the_reactions():
    """Where the species set has one balance, haishool.truth.reactions finds the same coefficients."""
    for name in ("limestone_to_lime", "cassiterite_to_tin", "hematite_to_iron", "magnetite_to_iron", "clay_to_ceramic"):
        t = m.TRANSFORMS[m.TRANSFORM_INDEX[name]]
        left, right = m.reaction_terms(t.reaction)
        coefs = reactions.balance([reactions.species(f) for _, f in left], [reactions.species(f) for _, f in right])
        assert coefs == [int(c) for c, _ in left + right], name


def test_transform_conditions():
    T = {t.name: t for t in m.TRANSFORMS}
    assert T["clay_to_ceramic"].min_temp_k >= 1150
    assert T["limestone_to_lime"].min_temp_k == pytest.approx(1110, abs=25)
    assert T["limestone_to_lime"].enthalpy_j_kg > 0 and T["limestone_to_lime"].atmosphere == "any"
    assert T["wood_to_charcoal"].min_temp_k == pytest.approx(600) and T["wood_to_charcoal"].atmosphere == "smothered"
    assert T["malachite_to_copper"].atmosphere == "reducing" and T["malachite_to_copper"].min_temp_k >= 1000
    assert T["malachite_to_copper"].inputs["charcoal"] > 0
    assert T["cassiterite_to_tin"].atmosphere == "reducing" and T["cassiterite_to_tin"].min_temp_k >= 1100
    for name in ("hematite_to_iron", "magnetite_to_iron"):
        assert T[name].atmosphere == "reducing" and T[name].min_temp_k >= 1400 and T[name].outputs["iron"] > 0.69
    assert T["copper_tin_to_bronze"].min_temp_k == pytest.approx(1357.8, abs=0.1)
    assert T["iron_to_steel"].min_temp_k >= 1200 and T["iron_to_steel"].atmosphere == "reducing"
    assert T["sand_ash_to_glass"].min_temp_k >= 1500     # raised from 1400 K for the lime-rich melt
    # the Ellingham temperatures lie below the practical onsets for the smelts
    for name in ("malachite_to_copper", "cassiterite_to_tin", "hematite_to_iron"):
        assert T[name].t_eq_k < T[name].onset_k
    # every product is reachable from crust species, wood and ash
    made = {s for t in m.TRANSFORMS for s in t.outputs}
    assert {"ceramic", "lime", "charcoal", "copper", "tin", "bronze", "iron", "steel", "glass"} <= made


def test_transform_tensors_and_air():
    tt = m.transform_tensors("cpu")
    T = len(m.TRANSFORMS)
    assert tt["consume"].shape == (T, m.S) and tt["gas_out"].shape == (T, len(m.GASES))
    net = tt["produce"].sum(1) + tt["gas_out"].sum(1) - tt["consume"].sum(1)
    assert torch.allclose(net, torch.zeros(T), atol=1e-6)
    assert bool(tt["reducing"][m.TRANSFORM_INDEX["hematite_to_iron"]])
    air = m.gas_to_air(m.TRANSFORMS[m.TRANSFORM_INDEX["hematite_to_iron"]])
    t = m.TRANSFORMS[m.TRANSFORM_INDEX["hematite_to_iron"]]
    # carbon of the charcoal leaves as CO2 once the CO has burnt
    c_in = t.inputs["charcoal"]
    assert air["CO2"] * 12.011 / 44.009 == pytest.approx(c_in, rel=1e-4) and air["O2"] < 0


# ------------------------------------------------------------------ props
def _comp(**fractions):
    return m.species_vector(fractions)


def test_props_pure_and_mixed():
    comp = torch.stack([_comp(flint=1.0), _comp(wood=0.8, flint=0.2), _comp(copper=0.89, tin=0.11), _comp(bronze=1.0),
                        _comp(meat=1.0), _comp(charcoal=1.0), torch.zeros(m.S)])
    mass = torch.tensor([0.5, 1.0, 2.0, 2.0, 3.0, 1.0, 0.0])
    p = m.props(comp, mass)
    assert all(v.shape == (7,) and v.dtype == torch.float32 for v in p.values())
    assert p["hardness"][0] == pytest.approx(7.0)
    assert p["hardness"][1] == pytest.approx(float(m.MOHS["wood"]))         # dispersed flint does not harden wood
    assert p["density"][0] == pytest.approx(2600) and p["volume_m3"][0] == pytest.approx(0.5 / 2600)
    v_mix = 0.8 / 545 + 0.2 / 2600
    assert p["density"][1] == pytest.approx(1 / v_mix, rel=1e-5)
    assert p["melt_k"][2] == pytest.approx(505.08, abs=0.01)                 # unalloyed tin melts out first
    assert p["melt_k"][3] == pytest.approx(1273.0)                           # an alloy uses its own row
    assert p["food_j"][4] == pytest.approx(3.0 * float(m.FOOD_J_KG["meat"]), rel=1e-5)
    assert p["food_meat_j"][4] == pytest.approx(0.9 * p["food_j"][4], rel=1e-5) and p["cook_gain"][4] == pytest.approx(1.3)
    assert m.digestible_j(p, torch.tensor(0.5))[4] == pytest.approx(0.65 * p["food_j"][4], rel=1e-5)
    assert p["fuel_j"][5] == pytest.approx(float(m.COMBUSTION_J_KG["charcoal"]), rel=1e-6)
    assert p["ignition_k"][5] == pytest.approx(623.0) and p["ignition_k"][0] == pytest.approx(5000.0)
    assert p["brittle"][0] == 1 and p["brittle"][1] == pytest.approx(0.2)
    assert p["heat_capacity_j_k"][0] == pytest.approx(0.5 * float(m.CP["flint"]), rel=1e-6)
    assert p["hardness"][6] == 0 and p["density"][6] == 0 and p["melt_k"][6] == pytest.approx(5000.0)


def test_props_batched_and_deterministic():
    gen = torch.Generator().manual_seed(7)
    comp = torch.rand(3, 50, m.S, generator=gen) ** 4
    mass = torch.rand(3, 50, generator=gen)
    p1, p2 = m.props(comp, mass), m.props(comp.clone(), mass.clone())
    for k in p1:
        assert p1[k].shape == (3, 50) and torch.equal(p1[k], p2[k])
    flat = m.props(comp.reshape(150, m.S), mass.reshape(150))
    for k in p1:
        assert torch.equal(p1[k].reshape(150), flat[k])
    H = torch.tensor([float(m.MOHS[s]) for s in m.SPECIES])
    assert bool((p1["hardness"] <= H.max() + 1e-6).all()) and bool((p1["hardness"] >= 0).all())
    assert bool((p1["melt_k"] <= 5000).all())


# ------------------------------------------------------------------ crust
def test_crust_fractions_781():
    f = m.crust_fractions({"cloud": CLOUD_781})
    assert list(f) == list(m.CRUST_SPECIES)
    assert sum(f.values()) == pytest.approx(1.0, abs=1e-12) and min(f.values()) > 0
    k = m.crust_scales({"cloud": CLOUD_781})
    assert k["z"] == pytest.approx(0.003047 / 0.0134, rel=1e-3)
    assert k["mg_si"] > 1 > k["fe_si"]           # 781's cloud: Mg-rich, Fe-poor relative to the sun
    assert f["granite"] > f["hematite"] > f["malachite"] > f["cassiterite"] > 0
    sun = m.crust_fractions({"cloud": m.solar_cloud()})
    assert sun["limestone"] / sun["sand"] == pytest.approx(0.13 / 0.10, rel=1e-9)  # solar cloud = Earth shares
    assert f["malachite"] < sun["malachite"] and f["basalt"] > sun["basalt"]


def test_crust_responds_to_the_cloud():
    base = {"cloud": dict(CLOUD_781)}
    rich_c = {"cloud": {**CLOUD_781, "carbon": 2 * CLOUD_781["carbon"]}}
    rich_z = {"cloud": {**CLOUD_781, "metallicity": 2 * CLOUD_781["metallicity"]}}
    f0, fc, fz = (m.crust_fractions(x) for x in (base, rich_c, rich_z))
    assert fc["limestone"] > f0["limestone"]
    assert fz["malachite"] > f0["malachite"] and fz["cassiterite"] > f0["cassiterite"]
    assert fz["limestone"] == pytest.approx(f0["limestone"], rel=1e-3)   # Ca follows the chain's other/Si, not Z


# ------------------------------------------------------------------ review fixes
def test_provenance_tags_sentinels_makeup_and_transforms():
    for name, table in m.TABLES.items():
        for s, v in table.items():
            if float(v) == float(m.NEVER_K):                 # a sentinel is a rule, not a reference value
                assert v.tag == m.NEW_RULE and "sentinel" in v.source, (name, s)
    assert m.MOHS["basalt"].tag == m.NEW_RULE and "knap" in m.MOHS["basalt"].source
    prov = m.provenance()
    for s in m.SPECIES:
        assert f"materials.MAKEUP.{s}" in prov, s
    for key in ("materials.PROTEIN", "materials.ASH_MAKEUP", "materials.X_SUN", "materials.H_FG_WATER",
                "materials.SMOTHER_COVER", "materials.solar_cloud", "materials.GRANULAR.sand"):
        assert key in prov, key
    assert prov["materials.ASH_MAKEUP"][0] == m.NEW_RULE
    assert all(f"materials.GROSS_J_KG.{c}" in prov for c in m.GROSS_J_KG)
    for t in m.TRANSFORMS:
        for field in ("onset_k", "days", "enthalpy_j_kg"):
            tag, src = prov[f"materials.TRANSFORMS.{t.name}.{field}"]
            assert tag in m.TAGS and src, (t.name, field)
        assert isinstance(t.onset_k, m.Val) and isinstance(t.days, m.Val)
    # estimates of the holding time are rules, not references
    assert prov["materials.TRANSFORMS.iron_to_steel.days"][0] == m.NEW_RULE
    assert prov["materials.TRANSFORMS.sand_ash_to_glass.enthalpy_j_kg"][0] == m.NEW_RULE
    assert all(tag in m.TAGS and src for tag, src in prov.values())


def test_burn_rejects_non_fuels():
    for s in m.SPECIES:
        if s in m.FUELS:
            assert m.burn(s)["CO2"] > 0
        else:
            with pytest.raises(ValueError):
                m.burn(s)
    with pytest.raises(ValueError):
        m.burn("meat")


def test_props_sanitises_negative_and_non_finite_rows():
    rows = [
        _comp(flint=1e-6, wood=-0.99e-6),            # near-cancelling float32 residue of crafting
        _comp(flint=-1e-9),                          # only a negative residue
        _comp(wood=0.5, meat=-0.2, charcoal=0.5),
        torch.full((m.S,), float("nan")),
        _comp(flint=float("inf"), wood=1.0),
        torch.zeros(m.S),
    ]
    comp = torch.stack(rows)
    mass = torch.tensor([1.0, 1.0, 2.0, 1.0, 1.0, float("nan")])
    p = m.props(comp, mass)
    for k, v in p.items():
        assert bool(torch.isfinite(v).all()) and bool((v >= 0).all()), (k, v)
    pure = m.props(_comp(flint=1.0)[None], torch.tensor([1.0]))
    for k in p:                                       # the residue row reads as pure flint
        assert p[k][0].item() == pytest.approx(pure[k][0].item(), rel=1e-5), k
    assert p["hardness"][1] == 0 and p["volume_m3"][1] == 0 and p["melt_k"][1] == pytest.approx(5000.0)
    half = m.props(_comp(wood=0.5, charcoal=0.5)[None], torch.tensor([2.0]))
    assert p["fuel_j"][2].item() == pytest.approx(half["fuel_j"][0].item(), rel=1e-6)
    assert p["mass"][5] == 0 and p["heat_capacity_j_k"][5] == 0


def test_granular_sand_and_ash_are_no_hammer():
    comp = torch.stack([_comp(sand=1.0), _comp(basalt=1.0), _comp(ash=1.0), _comp(flint=1.0), _comp(clay=1.0),
                        _comp(sand=0.5, clay=0.5)])
    p = m.props(comp, torch.ones(6))
    sand, basalt, ash, flint, clay, mix = range(6)
    assert p["hardness"][sand] == pytest.approx(7.0)                 # the grains still scratch (abrasion)
    assert p["granular"][sand] == 1 and p["granular"][basalt] == 0 and p["granular"][mix] == pytest.approx(0.5)
    assert p["tool_hardness"][sand] == 0 and p["tool_hardness"][ash] == 0
    assert p["tool_hardness"][sand] < p["tool_hardness"][basalt] == pytest.approx(6.5)
    # the knapping rule: core hardness <= hammer x 1.1 (PLANET-SPEC 2.11)
    assert p["tool_hardness"][basalt] * 1.1 >= p["tool_hardness"][flint]
    assert not p["tool_hardness"][sand] * 1.1 >= p["tool_hardness"][flint]
    assert p["tool_hardness"][clay] == pytest.approx(float(m.MOHS["clay"]))   # a clay lump is cohesive
    assert m.GRANULAR["sand"] == 1 and m.GRANULAR["ash"] == 1 and m.GRANULAR["clay"] == 0


def test_glass_composition_and_onset():
    ox = dict(m.MAKEUP["glass"])
    assert ox["SiO2"] == pytest.approx(0.629, abs=0.001) and ox["CaO"] == pytest.approx(0.264, abs=0.001)
    assert ox["K2O"] == pytest.approx(0.107, abs=0.001)
    for text in (m.GLASS_ASH_PER_SAND.source, m.TRANSFORMS[m.TRANSFORM_INDEX["sand_ash_to_glass"]].source):
        assert "62.9" in text and "26.4" in text and "10.7" in text
    t = m.TRANSFORMS[m.TRANSFORM_INDEX["sand_ash_to_glass"]]
    assert t.min_temp_k == pytest.approx(1500.0) and t.onset_k.tag == m.NEW_RULE


def test_from_reaction_refuses_reactions_favoured_below_t_eq():
    with pytest.raises(ValueError):       # dH < 0, dS < 0: favoured only below 1118 K
        m._from_reaction("recarbonation", "lime", "CaO + CO2 -> CaCO3", "CaO", {"CaO": "lime", "CaCO3": "limestone"},
                         "any", m.Val(300.0, m.NEW_RULE, "test"), m.Val(1.0, m.NEW_RULE, "test"), "test")
    with pytest.raises(ValueError):
        m._from_reaction("x", "lime", "CaO + CO2 -> CaCO3", "CaO", {}, "vacuum", 300.0, 1.0, "test")
    for t in m.TRANSFORMS:                # every transform here has t_eq as a lower bound
        if t.reaction is not None and t.t_eq_k:
            dh, ds = m.reaction_thermo(t.reaction)
            assert dh > 0 and ds > 0 and t.min_temp_k >= t.t_eq_k


def test_transform_tensors_element_closure_float32_and_cache():
    tt = m.transform_tensors("cpu")
    T = len(m.TRANSFORMS)
    E = m.element_matrix("cpu", torch.float32)
    Eg = m.gas_element_matrix(m.GASES, "cpu", torch.float32)
    Ea = m.gas_element_matrix(m.AIR_GASES, "cpu", torch.float32)
    assert tt["air_out"].shape == (T, 3) and tt["consume"].dtype == torch.float32
    res_gas = tt["produce"] @ E + tt["gas_out"] @ Eg - tt["consume"] @ E
    res_air = tt["produce"] @ E + tt["air_out"] @ Ea - tt["consume"] @ E
    assert float(res_gas.abs().max()) < 1e-6 and float(res_air.abs().max()) < 1e-6
    for i, t in enumerate(m.TRANSFORMS):
        air = m.gas_to_air(t)
        assert tt["air_out"][i].tolist() == pytest.approx([air[g] for g in m.AIR_GASES], abs=1e-6)
        assert bool(tt["smothered"][i]) == (t.atmosphere == "smothered")
        assert bool(tt["reducing"][i]) == (t.atmosphere == "reducing")
    assert bool((tt["air_out"][:, 0] <= 0).all())        # O2 is only taken from the air
    again = m.transform_tensors("cpu")
    assert again["consume"] is tt["consume"] and again is not tt


def test_fire_atmosphere_smothered_and_reducing():
    wood, charcoal = m.IDX["wood"], m.IDX["charcoal"]
    fuel = torch.zeros(5, m.S)
    fuel[0, wood] = 5.0                     # open wood fire, big bed, no air: smothered, not reducing
    fuel[1, wood] = 5.0                     # same with bellows
    fuel[2, wood] = 0.2                     # thin bed under a 1 kg item
    fuel[3, charcoal] = 3.0                 # charcoal fire with bellows: reducing
    air = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0])
    item_mass = torch.ones(5)
    item_comp = torch.stack([_comp(wood=1.0)] * 4 + [_comp(malachite=0.8, charcoal=0.2)])
    a = m.fire_atmosphere(fuel, air, item_mass, item_comp)
    assert a["smothered"].tolist() == [True, False, False, False, False]
    assert a["reducing"].tolist() == [False, False, False, True, True]
    ok = m.atmosphere_ok(fuel, air, item_mass, item_comp)
    I = m.TRANSFORM_INDEX
    assert ok.shape == (5, len(m.TRANSFORMS))
    assert ok[:, I["wood_to_charcoal"]].tolist() == [True, False, False, False, False]
    assert ok[:, I["malachite_to_copper"]].tolist() == [False, False, False, True, True]
    assert bool(ok[:, I["clay_to_ceramic"]].all())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_props_and_tensors_on_cuda_match_cpu():
    gen = torch.Generator().manual_seed(11)
    comp = torch.rand(4, 64, m.S, generator=gen) ** 4 - 0.01
    mass = torch.rand(4, 64, generator=gen) * 3
    cpu = m.props(comp, mass)
    gpu = m.props(comp.cuda(), mass.cuda())
    for k in cpu:
        assert torch.allclose(gpu[k].cpu(), cpu[k], rtol=1e-5, atol=1e-5), k
    a, b = m.transform_tensors("cpu"), m.transform_tensors("cuda")
    for k in ("consume", "produce", "gas_out", "air_out", "min_temp_k"):
        assert torch.equal(a[k], b[k].cpu()), k


def test_props_table_product_matches_the_species_sums():
    """props sums the linear properties as one product w @ table; it must equal the per-property sums."""
    gen = torch.Generator().manual_seed(0)
    comp = torch.rand(40, m.S, generator=gen) * (torch.rand(40, m.S, generator=gen) > 0.6)
    comp[0] = 0.0
    mass = 10 * torch.rand(40, generator=gen)
    p = m.props(comp, mass)
    w = comp / comp.sum(-1, keepdim=True).clamp_min(1e-30)
    v = lambda t: torch.tensor([float(t[s]) for s in m.SPECIES])
    ref = {"toughness": (w * v(m.TOUGHNESS)).sum(-1), "brittle": (w * v(m.BRITTLE)).sum(-1),
           "cp": (w * v(m.CP)).sum(-1), "conductivity": (w * v(m.CONDUCTIVITY)).sum(-1),
           "fuel_j": (w * v(m.COMBUSTION_J_KG)).sum(-1) * mass, "food_j": (w * v(m.FOOD_J_KG)).sum(-1) * mass,
           "food_meat_j": (w * v(m.FOOD_J_KG) * v(m.DIGESTIBLE["meat"])).sum(-1) * mass,
           "volume_m3": (w / v(m.DENSITY)).sum(-1) * mass}
    for k, r in ref.items():
        assert torch.allclose(p[k], r, rtol=1e-5, atol=1e-6 * float(r.abs().max())), k
    assert float(p["mass"][0]) == pytest.approx(float(mass[0])) and float(p["food_j"][0]) == 0.0
