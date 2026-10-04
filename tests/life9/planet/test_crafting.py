"""life9 planet crafting: actions, fire physics, transforms, cooking, decay, tool effects, ledgers, determinism.

Gates (PLANET-SPEC section 4): flint knapped by basalt sharpens and basalt knapped by wood does
nothing; clay above 1150 K long enough becomes ceramic; malachite with charcoal above 1000 K becomes
copper and without charcoal does not; no fire below an O2 mole fraction of 0.15; species and
element mass balance through every transform; determinism on the CPU.
"""
import math

import pytest
import torch

from haishool.life9.planet import crafting as cr
from haishool.life9.planet import items as it
from haishool.life9.planet import materials as m

R_HAB = 1.0e4                     # habitat radius (m)
E = m.element_matrix()
EA = m.gas_element_matrix(m.AIR_GASES)
TI = m.TRANSFORM_INDEX


def at(dx_m=0.0, dy_m=0.0):
    """A unit vector dx, dy metres from the north pole of the habitat globe."""
    return torch.nn.functional.normalize(torch.tensor([dx_m / R_HAB, dy_m / R_HAB, 1.0]), dim=0)


class Scene:
    def __init__(self, W=1, N=4, K=3, I=64, F=8):
        self.W, self.N, self.K = W, N, K
        self.pool = it.new_item_pool(W, I)
        self.fires = it.new_fire_pool(W, F)
        self.inv = torch.full((W, N, K), -1, dtype=torch.long)
        self.pos = torch.stack([at(3.0 * n) for n in range(N)]).expand(W, N, 3).clone()

    def give(self, w, n, k, species, mass, temp=288.0, **kw):
        comp = m.species_vector(species if isinstance(species, dict) else {species: 1.0})[None]
        i = it.spawn(self.pool, torch.tensor([w]), comp, torch.tensor([float(mass)]), self.pos[w, n][None],
                     torch.tensor([temp]), torch.tensor([n * self.K + k]), **kw)
        assert i[0] >= 0
        self.inv[w, n, k] = i[0]
        return int(i[0])

    def ground(self, w, species, mass, p, temp=288.0):
        comp = m.species_vector(species if isinstance(species, dict) else {species: 1.0})[None]
        return int(it.spawn(self.pool, torch.tensor([w]), comp, torch.tensor([float(mass)]), p[None],
                            torch.tensor([temp]), torch.tensor([-1]))[0])

    def fire(self, w, bed: dict, p=None, air=0.0):
        p = self.pos[w, 0] if p is None else p
        return int(it.spawn_fire(self.fires, torch.tensor([w]), p[None], m.species_vector(bed)[None],
                                 torch.tensor([1000.0]), air=air)[0])

    def act(self, *who):
        a = torch.zeros(self.W, self.N, dtype=torch.bool)
        for w, n in who:
            a[w, n] = True
        return a

    def slots(self, value):
        return torch.full((self.W, self.N), value, dtype=torch.long)

    def mass(self):
        return cr.ledger_mass(self.pool, self.fires)

    def step(self, x_o2=0.21, dt=86400.0, **kw):
        return cr.fire_step(self.pool, self.fires, torch.full((self.W,), x_o2), dt, ambient_item_k=288.0,
                            ambient_fire_k=288.0, radius_m=R_HAB, inv=self.inv, **kw)

    def species(self, w, i):
        return {m.SPECIES[s]: v for s, v in enumerate(self.pool.comp[w, i].tolist()) if v > 1e-6}


def element_residual(before, after, flows):
    """max |E(after) - E(before) + E(air out) + E(to soil)|, kg."""
    res = (after - before) @ E + flows["air_kg"] @ EA + flows["to_soil_kg"] @ E
    return float(res.abs().max())


# ------------------------------------------------------------------ knapping (gate)
def test_flint_knapped_by_basalt_sharpens():
    sc = Scene()
    ham = sc.give(0, 0, 0, "basalt", 1.0)
    core = sc.give(0, 0, 1, "flint", 0.8)
    before = sc.mass()
    r = cr.knap(sc.pool, sc.inv, sc.act((0, 0)), sc.pos, body_mass_kg=60.0, hammer_slot=sc.slots(0), core_slot=sc.slots(1), effort=1.0)
    assert bool(r["ok"][0, 0])
    lim = 1 / (1 + (float(m.TOUGHNESS["flint"]) / float(cr.K_EDGE)) ** 2)
    assert 0.9 * lim < sc.pool.sharp[0, core].item() <= lim + 1e-6
    d = int(r["debris"][0, 0])
    assert sc.pool.alive[0, d] and sc.pool.holder[0, d] == -1 and sc.species(0, d) == {"flint": pytest.approx(1.0)}
    assert sc.pool.mass[0, core].item() + sc.pool.mass[0, d].item() == pytest.approx(0.8)
    assert sc.pool.sharp[0, d].item() == pytest.approx(lim)          # a fresh flake has a fresh edge
    assert sc.pool.mass[0, ham].item() == 1.0
    assert torch.allclose(sc.mass(), before)
    assert cr.check_inventory(sc.pool, sc.inv) == []
    # more knapping keeps the edge below its limit and removes more mass
    cr.knap(sc.pool, sc.inv, sc.act((0, 0)), sc.pos, body_mass_kg=60.0, hammer_slot=sc.slots(0), core_slot=sc.slots(1), effort=0.5)
    assert sc.pool.sharp[0, core].item() <= lim + 1e-6 and sc.pool.mass[0, core].item() < 0.64


def test_basalt_knapped_by_wood_does_nothing():
    sc = Scene()
    sc.give(0, 0, 0, "wood", 1.0)
    sc.give(0, 0, 1, "basalt", 1.0)
    before = it.state_hash(sc.pool)
    r = cr.knap(sc.pool, sc.inv, sc.act((0, 0)), sc.pos, body_mass_kg=60.0, hammer_slot=sc.slots(0), core_slot=sc.slots(1))
    assert not r["ok"].any() and (r["debris"] == -1).all()
    assert it.state_hash(sc.pool) == before
    # defaults: the same pair, wood cannot be the core (not brittle), basalt is too hard for wood
    r = cr.knap(sc.pool, sc.inv, sc.act((0, 0)), sc.pos, body_mass_kg=60.0)
    assert not r["ok"].any() and it.state_hash(sc.pool) == before


def test_knap_rules_defaults_and_limits():
    sc = Scene(N=3)
    sc.give(0, 0, 0, "basalt", 1.0)
    core = sc.give(0, 0, 1, "flint", 1.0)           # the default core is the one taking the best edge
    sc.give(0, 1, 0, "granite", 1.0)
    sc.give(0, 1, 1, "sand", 1.0)                   # sand is not brittle
    sc.give(0, 2, 0, "flint", 1.0)
    glass = sc.give(0, 2, 1, "glass", 1.0)
    r = cr.knap(sc.pool, sc.inv, sc.act((0, 0), (0, 1), (0, 2)), sc.pos, body_mass_kg=60.0, effort=1.0)
    assert r["ok"][0].tolist() == [True, False, True]
    assert sc.pool.sharp[0, core] > 0 and sc.pool.sharp[0, glass] > sc.pool.sharp[0, core]   # glass edges are finer
    # a hammer may be at most 10 % softer than the core: copper (3) on flint (7) fails
    sc2 = Scene()
    sc2.give(0, 0, 0, "copper", 1.0)
    sc2.give(0, 0, 1, "flint", 1.0)
    assert not cr.knap(sc2.pool, sc2.inv, sc2.act((0, 0)), sc2.pos, body_mass_kg=60.0)["ok"].any()


# ------------------------------------------------------------------ fire: oxygen, ignition, temperature
def test_no_fire_below_the_oxygen_limit():
    sc = Scene(W=3)
    for w in range(3):
        sc.give(w, 0, 0, "wood", 5.0)
    x = torch.tensor([0.14, 0.1499, 0.15])
    r = cr.make_fire(sc.pool, sc.fires, sc.inv, sc.act((0, 0), (1, 0), (2, 0)), sc.pos, x, 288.0, body_mass_kg=70.0)
    assert r["lit"][:, 0].tolist() == [False, False, True]
    assert r["no_oxygen"][:, 0].tolist() == [True, True, False]
    assert (sc.inv[:2, 0, 0] >= 0).all() and sc.inv[2, 0, 0] == -1      # a failed fire keeps its fuel in hand
    assert sc.fires.alive.sum().item() == 1
    # a burning fire goes out when the air drops below the limit; its bed becomes an item
    fl = cr.fire_step(sc.pool, sc.fires, x, 3600.0, ambient_item_k=288.0, ambient_fire_k=288.0, radius_m=R_HAB)
    assert sc.fires.alive.sum().item() == 1 and fl["heat_j"][2] > 0
    before = sc.mass()
    fl = cr.fire_step(sc.pool, sc.fires, torch.tensor([0.21, 0.21, 0.14]), 3600.0, ambient_item_k=288.0,
                      ambient_fire_k=288.0, radius_m=R_HAB)
    assert not sc.fires.alive.any() and fl["fires_out"][2] == 1 and fl["heat_j"][2] == 0
    assert element_residual(before, sc.mass(), fl) < 1e-5


def test_ignition_by_friction_and_by_sparks():
    sc = Scene(W=1, N=4)
    sc.give(0, 0, 0, "wood", 2.0)                          # strong adult at full effort: the drill lights wood
    sc.give(0, 1, 0, "wood", 2.0)                          # low effort: the tip stays below 573 K
    sc.give(0, 2, 0, "wood", 2.0)                          # a 5 kg animal cannot reach it
    sc.give(0, 3, 0, "plant_fiber", 0.2)                   # tinder + pyrite + flint at low effort: sparks
    sc.give(0, 3, 1, "pyrite", 0.3)
    sc.give(0, 3, 2, "flint", 0.5)
    mass = torch.tensor([[70.0, 70.0, 5.0, 70.0]])
    effort = torch.tensor([[1.0, 0.1, 1.0, 0.05]])
    r = cr.make_fire(sc.pool, sc.fires, sc.inv, sc.act((0, 0), (0, 1), (0, 2), (0, 3)), sc.pos, 0.21, 288.0,
                     body_mass_kg=mass, effort=effort, fuel_slot=sc.slots(0))
    assert r["lit"][0].tolist() == [True, False, False, True]
    assert r["no_ignition"][0].tolist() == [False, True, True, False]
    assert r["spark"][0].tolist() == [False, False, False, True]
    k = float(m.CONDUCTIVITY["wood"])
    p = float(cr.ARM_POWER_W)
    assert r["contact_k"][0, 0].item() == pytest.approx(288 + 0.1 * p / (8 * 0.005 * k), rel=1e-5)
    assert (r["work_j"][0] > 0).all()
    # sparks do not light wood (it needs an ember): a pyrite + flint holder with only wood fails at low effort
    sc2 = Scene(N=1)
    sc2.give(0, 0, 0, "wood", 1.0)
    sc2.give(0, 0, 1, "pyrite", 0.3)
    sc2.give(0, 0, 2, "flint", 0.5)
    r = cr.make_fire(sc2.pool, sc2.fires, sc2.inv, sc2.act((0, 0)), sc2.pos, 0.21, 288.0, body_mass_kg=70.0,
                     effort=0.05)
    assert not r["lit"].any()


def test_fire_temperature_formula():
    beds = torch.stack([m.species_vector({"wood": 5.0}), m.species_vector({"charcoal": 5.0}),
                        m.species_vector({"charcoal": 5.0}), m.species_vector({"wood": 1.0, "charcoal": 1.0})])[None]
    air = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    t = cr.fire_temperature(beds, air, torch.tensor([0.21]), 288.0)[0]
    dt0 = float(cr.FIRE_DT0_K)
    assert dt0 == 812.0                                                       # 1100 K open wood fire at 288 K
    assert t[0].item() == pytest.approx(1100.0, abs=1e-3)                     # the calibration anchor
    assert t[1].item() == pytest.approx(1384.2, abs=1e-2)                     # open charcoal: 288 + 812 x 1.35
    assert t[2].item() == pytest.approx(1713.06, abs=1e-1)                    # fully forced charcoal, a forge
    assert t[3].item() == pytest.approx(1242.1, abs=1e-1)                     # mass-weighted fuel factor 1.175
    low = cr.fire_temperature(beds, air, torch.tensor([0.15]), 288.0)[0]
    assert low[0].item() == pytest.approx(974.3, abs=1e-1)                    # x 0.15 / 0.21 under the root
    # the spec's 'forced charcoal fire about 1500 K' is a third of full effort
    third = cr.fire_temperature(beds[:, 1:2], torch.tensor([[1.0 / 3.0]]), torch.tensor([0.21]), 288.0)[0, 0]
    assert third.item() == pytest.approx(1500.0, abs=10.0)                     # 1494 K


def test_fire_burns_out_with_closed_ledgers():
    sc = Scene(W=2)
    sc.fire(0, {"wood": 7.0})
    sc.fire(1, {"charcoal": 2.0}, air=1.0)
    before = sc.mass()
    fl = sc.step(dt=2 * 86400.0, substeps=96)
    assert not sc.fires.alive.any() and fl["fires_out"].tolist() == [1, 1]
    assert element_residual(before, sc.mass(), fl) < 1e-5
    # total mass: items + beds lose exactly what went to the air
    assert (before.sum(1) - sc.mass().sum(1)).tolist() == pytest.approx(fl["air_kg"].sum(1).tolist(), abs=1e-4)
    # heat: the wood's own combustion energy minus the char left in the residue item
    left_char = sc.mass()[0, m.IDX["charcoal"]].item()
    want = 7.0 * float(m.COMBUSTION_J_KG["wood"]) - left_char * float(m.COMBUSTION_J_KG["charcoal"])
    assert fl["heat_j"][0].item() == pytest.approx(want, rel=1e-4)
    assert fl["heat_j"][1].item() == pytest.approx(2.0 * float(m.COMBUSTION_J_KG["charcoal"]), rel=1e-2)
    # moles for the gas ledger
    assert fl["o2_mol"][1].item() == pytest.approx(-fl["air_kg"][1, 0].item() / 0.031998, rel=1e-3)
    assert fl["co2_mol"][1].item() == pytest.approx(fl["o2_mol"][1].item(), rel=1e-6)   # C + O2 -> CO2
    # the ash of the wood is left as an item where the fire was
    ash = sc.mass()[0, m.IDX["ash"]].item()
    assert ash == pytest.approx(0.07, rel=1e-3)


def test_fire_lifetime_grows_with_its_size():
    def hours(kg):
        sc = Scene()
        sc.fire(0, {"wood": kg})
        for h in range(1, 60):
            sc.step(dt=3600.0, substeps=2)
            if not sc.fires.alive.any():
                return h
        return 60
    small, big = hours(2.0), hours(30.0)
    assert 3 < small < big < 60 and big > 12


def test_loose_fibre_burns_out_within_the_hour():
    """A bed is a pile (PACKING): loose fibre flares and is gone within the hour; wood lasts hours."""
    sc = Scene(W=2)
    sc.fire(0, {"plant_fiber": 1.0})
    sc.fire(1, {"wood": 1.0})
    sc.step(dt=3600.0, substeps=12)
    assert sc.fires.alive[:, 0].tolist() == [False, True]


# ------------------------------------------------------------------ transforms in fires (gates)
def test_clay_above_1150_k_long_enough_becomes_ceramic():
    sc = Scene(W=3)
    for w in range(3):
        sc.ground(w, "clay", 1.0, sc.pos[w, 0])
    sc.fire(0, {"charcoal": 10.0})                   # 1384 K for the day
    sc.fire(1, {"charcoal": 10.0})                   # the same fire, for two hours only
    sc.fire(2, {"fat": 30.0})                        # a flaming fire at 1100 K: below the onset
    ceramic = m.IDX["ceramic"]
    before = sc.mass()
    fl0 = sc.step(dt=86400.0)
    full = sc.mass()
    produce = float(m.TRANSFORMS[TI["clay_to_ceramic"]].outputs["ceramic"])
    assert full[0, ceramic].item() == pytest.approx(produce, rel=1e-5)
    assert full[0, m.IDX["clay"]].item() < 1e-6
    assert fl0["air_kg"][0, 2].item() > 0.139                                  # the water of kaolinite
    assert element_residual(before, full, fl0) < 1e-5
    # world 1 burnt for a day too; rerun it alone for two hours from scratch
    sc2 = Scene(W=2)
    sc2.ground(0, "clay", 1.0, sc2.pos[0, 0])
    sc2.fire(0, {"charcoal": 10.0})
    sc2.ground(1, "clay", 1.0, sc2.pos[1, 0])
    sc2.fire(1, {"fat": 30.0})
    fl = sc2.step(dt=7200.0, substeps=8)
    part = sc2.mass()
    assert 0 < part[0, ceramic].item() < 0.2 * produce          # under 2 h of a 12 h firing
    assert part[1, ceramic].item() == 0.0                        # 1100 K is not enough at any length
    assert sc2.fires.temp_k[1, 0].item() == pytest.approx(1100.0, abs=1.0)
    clay_i = (sc2.pool.alive[1] & (sc2.pool.comp[1, :, m.IDX["clay"]] > 0.99)).nonzero()[0, 0]
    assert sc2.pool.temp_k[1, clay_i].item() > 1050               # it did get hot
    assert fl["transformed_kg"][1, TI["clay_to_ceramic"]] == 0


def test_malachite_with_charcoal_above_1000_k_becomes_copper_and_not_without():
    sc = Scene(W=3)
    for w in range(3):
        sc.ground(w, "malachite", 1.0, sc.pos[w, 0])
    sc.fire(0, {"charcoal": 10.0})                         # reducing: charcoal in the bed
    sc.fire(1, {"fat": 30.0})                              # 1100 K but no charcoal anywhere
    sc.fire(2, {"fat": 30.0})                              # charcoal mixed into the ore lump instead
    m2 = (sc.pool.alive[2] & (sc.pool.comp[2, :, m.IDX["malachite"]] > 0.5)).nonzero()[0, 0]
    sc.pool.comp[2, m2] = m.species_vector({"malachite": 1.0, "charcoal": 0.2}) / 1.2
    sc.pool.mass[2, m2] = 1.2
    before = sc.mass()
    fl = sc.step(dt=6 * 3600.0, substeps=24)
    after = sc.mass()
    cu, mal = m.IDX["copper"], m.IDX["malachite"]
    t = m.TRANSFORMS[TI["malachite_to_copper"]]
    assert after[0, cu].item() > 0.5 * t.outputs["copper"]
    assert after[1, cu].item() == 0.0 and after[1, mal].item() == pytest.approx(1.0, rel=1e-6)
    assert after[2, cu].item() > 0.0
    assert fl["transformed_kg"][1, TI["malachite_to_copper"]] == 0
    assert element_residual(before, after, fl) < 1e-5
    # the charcoal reductant was drawn: copper made x charcoal per kg malachite
    made = fl["transformed_kg"][0, TI["malachite_to_copper"]].item()
    assert after[0, cu].item() == pytest.approx(made * t.outputs["copper"], rel=1e-4)


def _transform_case(name):
    """(item species kg, bed, air) that let transform ``name`` run in a day."""
    cases = {
        "clay_to_ceramic": ({"clay": 1.0}, {"charcoal": 20.0}, 0.0),
        "limestone_to_lime": ({"limestone": 1.0}, {"charcoal": 20.0}, 0.0),
        "wood_to_charcoal": ({"wood": 1.0}, {"charcoal": 20.0}, 0.0),          # smothered in a deep bed
        "malachite_to_copper": ({"malachite": 1.0}, {"charcoal": 20.0}, 0.0),
        "cassiterite_to_tin": ({"cassiterite": 1.0}, {"charcoal": 20.0}, 0.0),
        "hematite_to_iron": ({"hematite": 1.0}, {"charcoal": 20.0}, 1.0),
        "magnetite_to_iron": ({"magnetite": 1.0}, {"charcoal": 20.0}, 1.0),
        "copper_tin_to_bronze": ({"copper": 0.89, "tin": 0.11}, {"charcoal": 20.0}, 1.0),
        "native_copper_to_copper": ({"native_copper": 1.0}, {"charcoal": 20.0}, 1.0),
        "iron_to_steel": ({"iron": 1.0}, {"charcoal": 20.0}, 0.0),
        "sand_ash_to_glass": ({"sand": 0.5, "ash": 0.5}, {"charcoal": 20.0}, 1.0),
    }
    return cases[name]


@pytest.mark.parametrize("name", [t.name for t in m.TRANSFORMS])
def test_every_transform_runs_and_balances_species_and_elements(name):
    item, bed, air = _transform_case(name)
    sc = Scene()
    total = sum(item.values())
    i = it.spawn(sc.pool, torch.tensor([0]), m.species_vector(item)[None] / total, torch.tensor([total]),
                 sc.pos[0, 0][None], torch.tensor([288.0]), torch.tensor([-1]))[0]
    f = sc.fire(0, bed)
    before = sc.mass()
    fl = None
    for _ in range(12):                           # twelve hours, fanned every hour where the case needs it
        sc.fires.air[0, f] = air
        step = sc.step(dt=3600.0, substeps=2)
        fl = step if fl is None else {k: fl[k] + step[k] for k in ("transformed_kg", "air_kg", "to_soil_kg")}
    after = sc.mass()
    in_items = it.species_mass(sc.pool)
    t = TI[name]
    x = fl["transformed_kg"][0, t].item()
    assert x > 0.05, name
    assert element_residual(before, after, fl) < 2e-5, name
    total_before, total_after = before.sum().item(), after.sum().item()
    assert total_after + fl["air_kg"].sum().item() + fl["to_soil_kg"].sum().item() == pytest.approx(total_before, abs=2e-5)
    tr = m.TRANSFORMS[t]
    for s, kg in tr.outputs.items():
        # iron made in a reducing fire carburises on to steel; every other product is in the item,
        # charcoal from smothered wood included (it does not ignite under the bed)
        downstream = [u for u in m.TRANSFORMS if u.basis == s and fl["transformed_kg"][0, TI[u.name]] > 0]
        if not downstream:
            assert in_items[0, m.IDX[s]].item() >= (0.9 if s in m.FUELS else 0.99) * x * kg - 1e-6, (name, s)
        for u in downstream:
            assert after[0, m.IDX[next(iter(u.outputs))]].item() > 0, (name, u.name)
    assert sc.pool.sharp[0, i] == 0 or not sc.pool.alive[0, i]


def test_wood_in_a_thin_or_blown_fire_burns_instead_of_charring():
    sc = Scene(W=2)
    sc.ground(0, "wood", 2.0, sc.pos[0, 0])
    sc.fire(0, {"charcoal": 0.5})                          # the bed is thinner than the item: not smothered
    sc.ground(1, "wood", 2.0, sc.pos[1, 0])
    sc.fire(1, {"charcoal": 20.0}, air=0.5)                # forced air: not smothered
    fl = sc.step(dt=3600.0, substeps=6)
    assert fl["transformed_kg"][:, TI["wood_to_charcoal"]].tolist() == [0.0, 0.0]
    assert (sc.pool.alive & (sc.pool.comp[..., m.IDX["wood"]] > 0.5)).sum() == 0      # gone into the beds


# ------------------------------------------------------------------ actions
def _stock(W, C, values: dict):
    Sd = len(m.CRUST_SPECIES)
    st = torch.zeros(W, C, Sd)
    for (w, c, s), v in values.items():
        st[w, c, m.CRUST_SPECIES.index(s)] = v
    return st


def test_collect_from_the_ground_yield_rises_with_tool_hardness():
    sc = Scene(N=3)
    sc.give(0, 1, 0, "basalt", 1.0)
    sc.give(0, 2, 0, "copper", 1.0)
    stock = _stock(1, 3, {(0, 0, "flint"): 10.0, (0, 1, "flint"): 10.0, (0, 2, "flint"): 10.0})
    area = torch.full((3,), 100.0)
    cell = torch.tensor([[0, 1, 2]])
    before = stock.clone()
    gen = torch.Generator().manual_seed(1)
    r = cr.collect(sc.pool, sc.inv, sc.act((0, 0), (0, 1), (0, 2)), torch.zeros(1, 3, dtype=torch.long), sc.pos, cell,
                   stock, torch.zeros(1, 3), area, gen, body_mass_kg=50.0, item_temp_k=288.0, radius_m=R_HAB)
    kg = r["kg"][0]
    carry = float(cr.CARRY_FRACTION) * 50.0
    assert kg[0].item() == pytest.approx(carry * 2.5 / 7.0, rel=1e-5)          # bare hand on flint
    assert kg[1].item() == pytest.approx(carry * 6.5 / 7.0, rel=1e-5)          # basalt hammer
    assert kg[2].item() == pytest.approx(carry * 3.0 / 7.0, rel=1e-5)          # copper is softer
    assert (r["species"][0] == m.IDX["flint"]).all() and not r["picked"].any()
    assert r["ground_kg"][0, m.IDX["flint"]].item() == pytest.approx(kg.sum().item(), rel=1e-6)
    taken = (before - stock)[0, :, m.CRUST_SPECIES.index("flint")] * area
    assert taken.tolist() == pytest.approx(kg.tolist(), rel=1e-4)
    assert (sc.inv[0] >= 0).sum(-1).tolist() == [1, 2, 2] and cr.check_inventory(sc.pool, sc.inv) == []


def test_collect_shares_a_short_stock_and_takes_wood_and_loose_items():
    sc = Scene(N=4)
    stock = _stock(1, 2, {(0, 0, "clay"): 0.001})                 # 0.1 kg in the cell for two diggers
    wood = torch.tensor([[0.0, 2.0]])
    cell = torch.tensor([[0, 0, 1, 1]])
    loose = sc.ground(0, "ceramic", 0.4, at(9.5))                 # next to actor 3 (at 9 m): a 'stone' item
    gen = torch.Generator().manual_seed(0)
    choice = torch.tensor([[2, 2, 3, 0]])
    r = cr.collect(sc.pool, sc.inv, sc.act((0, 0), (0, 1), (0, 2), (0, 3)), choice, sc.pos, cell, stock, wood,
                   torch.tensor([100.0, 100.0]), gen, body_mass_kg=50.0, item_temp_k=290.0, radius_m=R_HAB)
    assert r["kg"][0, 0].item() == pytest.approx(0.05, rel=1e-4) and r["kg"][0, 1].item() == pytest.approx(0.05, rel=1e-4)
    assert stock[0, 0].sum().item() == pytest.approx(0.0, abs=1e-9)
    chop = float(cr.CARRY_FRACTION) * 50.0 * float(cr.CHOP_BLUNT)
    assert r["kg"][0, 2].item() == pytest.approx(chop, rel=1e-5) and r["species"][0, 2] == m.IDX["wood"]
    assert r["wood_kg"][0, 1].item() == pytest.approx(chop, rel=1e-5) and r["wood_kg"][0, 0] == 0
    assert bool(r["picked"][0, 3]) and r["item"][0, 3] == loose and sc.pool.holder[0, loose] == 3 * sc.K
    assert sc.pool.temp_k[0, r["item"][0, 2]].item() == 290.0
    assert cr.check_inventory(sc.pool, sc.inv) == []


def test_drop_heat_item_and_feed_fire():
    sc = Scene(N=3)
    rock = sc.give(0, 0, 0, "limestone", 1.0)
    wood = sc.give(0, 1, 0, {"wood": 0.8, "flint": 0.2}, 2.0)
    sc.give(0, 2, 1, "basalt", 1.0)
    f = sc.fire(0, {"charcoal": 3.0}, p=at(1.5))
    r = cr.drop(sc.pool, sc.inv, sc.act((0, 2)), sc.pos)
    i = int(r["item"][0, 2])
    assert sc.pool.holder[0, i] == -1 and torch.allclose(sc.pool.pos[0, i], sc.pos[0, 2])
    r = cr.heat_item(sc.pool, sc.fires, sc.inv, sc.act((0, 0)), sc.pos, radius_m=R_HAB)
    assert r["fire"][0, 0] == f and sc.pool.holder[0, rock] == -1 and torch.allclose(sc.pool.pos[0, rock], sc.fires.pos[0, f])
    before = sc.mass()
    r = cr.feed_fire(sc.pool, sc.fires, sc.inv, sc.act((0, 1)), sc.pos, radius_m=R_HAB, effort=0.7)
    assert r["fed_kg"][0, 1].item() == pytest.approx(1.6, rel=1e-5) and sc.fires.air[0, f].item() == 0.0
    r = cr.feed_fire(sc.pool, sc.fires, sc.inv, sc.act((0, 1)), sc.pos, radius_m=R_HAB, effort=0.7)  # nothing to feed
    assert r["fed_kg"][0, 1] == 0 and r["fan"][0, 1].item() == pytest.approx(0.7)
    assert sc.fires.air[0, f].item() == pytest.approx(0.7)
    assert sc.fires.fuel_kg[0, f, m.IDX["wood"]].item() == pytest.approx(1.6, rel=1e-5)
    assert sc.species(0, wood) == {"flint": pytest.approx(1.0)} and sc.pool.mass[0, wood].item() == pytest.approx(0.4)
    assert torch.allclose(sc.mass(), before) and cr.check_inventory(sc.pool, sc.inv) == []
    # out of reach: actor 2 moves to 12 m from a fire at 1.5 m
    sc.pos[0, 2] = at(12.0)
    sc.give(0, 2, 0, "wood", 1.0)
    r = cr.feed_fire(sc.pool, sc.fires, sc.inv, sc.act((0, 2)), sc.pos, radius_m=R_HAB)
    assert r["fire"][0, 2] == -1 and r["fed_kg"][0, 2] == 0
    fl = sc.step(dt=86400.0)
    assert sc.fires.air[0, f] == 0                                             # reset after the step
    assert 0.2 < fl["transformed_kg"][0, TI["limestone_to_lime"]] < 1.0                # the bed lasts part of the day


def test_fuel_part_burns_and_the_rest_stays_in_the_fire():
    sc = Scene(N=2)
    sc.give(0, 0, 0, "flint", 0.5)
    sc.give(0, 0, 1, "wood", 3.0)
    cr.combine(sc.pool, sc.inv, sc.act((0, 0)), head_slot=sc.slots(0), handle_slot=sc.slots(1))
    tool = int(sc.inv[0, 0, 0])
    before = sc.mass()
    r = cr.make_fire(sc.pool, sc.fires, sc.inv, sc.act((0, 0)), sc.pos, 0.21, 288.0, body_mass_kg=70.0)
    f = int(r["fire"][0, 0])
    assert f >= 0 and sc.fires.fuel_kg[0, f].sum().item() == pytest.approx(3.0, rel=1e-6)
    assert sc.pool.alive[0, tool] and sc.pool.holder[0, tool] == -1 and sc.species(0, tool) == {"flint": pytest.approx(1.0)}
    assert sc.pool.handle_len_m[0, tool] == 0 and sc.inv[0, 0, 0] == -1
    assert torch.allclose(sc.mass(), before) and cr.check_inventory(sc.pool, sc.inv) == []
    # clay put in the fire comes back out as ceramic with collect (ceramic is in the 'stone' group)
    clay = sc.give(0, 1, 0, "clay", 1.0)
    sc.pos[0, 1] = at(0.5)
    cr.heat_item(sc.pool, sc.fires, sc.inv, sc.act((0, 1)), sc.pos, radius_m=R_HAB)
    sc.fires.fuel_kg[0, f, m.IDX["charcoal"]] = 10.0
    sc.step(dt=86400.0)
    assert sc.species(0, clay) == {"ceramic": pytest.approx(1.0)}
    stock = _stock(1, 1, {})
    r = cr.collect(sc.pool, sc.inv, sc.act((0, 1)), torch.zeros(1, 2, dtype=torch.long), sc.pos,
                   torch.zeros(1, 2, dtype=torch.long), stock, torch.zeros(1, 1), torch.ones(1), torch.Generator(),
                   body_mass_kg=50.0, item_temp_k=288.0, radius_m=R_HAB)
    assert bool(r["picked"][0, 1]) and r["item"][0, 1] in (clay, tool)
    assert cr.check_inventory(sc.pool, sc.inv) == []


def test_combine_head_handle_binder():
    sc = Scene(N=2)
    head = sc.give(0, 0, 0, "flint", 0.5, temp=300.0, sharp=0.6)
    sc.give(0, 0, 1, "wood", 0.3, temp=280.0)
    sc.give(0, 0, 2, "plant_fiber", 0.05)
    h1 = sc.give(0, 1, 0, "flint", 0.5, sharp=0.6)
    sc.give(0, 1, 1, "wood", 0.3)
    before = sc.mass()
    heat0 = (it.item_props(sc.pool)["heat_capacity_j_k"] * sc.pool.temp_k).sum().item()
    r = cr.combine(sc.pool, sc.inv, sc.act((0, 0), (0, 1)))
    assert r["item"][0].tolist() == [head, h1]
    assert sc.pool.head_mass[0, head].item() == pytest.approx(0.5) and sc.pool.mass[0, head].item() == pytest.approx(0.85)
    L = cr.handle_length(torch.tensor(0.3), torch.tensor(float(m.DENSITY["wood"])), torch.tensor(3.0)).item()
    assert sc.pool.handle_len_m[0, head].item() == pytest.approx(L, rel=1e-5) and 0.3 < L < 0.8
    assert sc.pool.bond[0, head] > sc.pool.bond[0, h1] == pytest.approx(float(cr.BOND_FIT))
    assert sc.pool.sharp[0, head].item() == pytest.approx(0.6)
    assert torch.allclose(sc.mass(), before)
    heat1 = (it.item_props(sc.pool)["heat_capacity_j_k"] * sc.pool.temp_k).sum().item()
    assert heat1 == pytest.approx(heat0, rel=1e-6)
    assert (sc.inv[0, 0] >= 0).tolist() == [True, False, False] and cr.check_inventory(sc.pool, sc.inv) == []
    # the assembly works with its head's hardness and strikes harder than the bare head
    tool = cr.tool_of(sc.pool, sc.inv, sc.slots(0), torch.full((1, 2), 60.0))
    assert tool["hardness"][0, 0].item() == pytest.approx(7.0)
    bare_head = cr.strike_energy(torch.tensor(0.5), torch.tensor(0.0))
    assert cr.strike_energy(tool["head_mass"][0, 0], tool["handle_len_m"][0, 0]) > 1.5 * bare_head


def test_share_gives_to_the_nearest_relative_with_a_free_hand():
    sc = Scene(N=5)
    uid = torch.tensor([[10, 11, 12, 13, 14]])
    parent = torch.tensor([[-1, 10, 10, -1, 10]])          # 1, 2 and 4 are 0's children; 3 is unrelated
    sc.pos[0] = torch.stack([at(0.0), at(1.0), at(1.5), at(0.5), at(30.0)])
    alive = torch.ones(1, 5, dtype=torch.bool)
    a = sc.give(0, 0, 0, "flint", 0.3)
    b = sc.give(0, 2, 0, "meat", 1.0)
    for k in range(3):
        sc.give(0, 1, k, "sand", 0.1)                       # child 1 has no free hand
    r = cr.share(sc.pool, sc.inv, sc.act((0, 0), (0, 2)), sc.pos, uid, parent, alive, radius_m=R_HAB)
    # 0 -> 2 (1 is nearer but full; 3 is nearer but not kin); 2 -> 0 (its parent; 1 is a full sibling)
    assert r["to"][0].tolist() == [2, -1, 0, -1, -1]
    assert sc.pool.holder[0, a] // sc.K == 2 and sc.pool.holder[0, b] // sc.K == 0
    assert cr.check_inventory(sc.pool, sc.inv) == []
    # two givers (2's child and 2's parent, not kin of each other), one free hand: the lower index gives
    sc2 = Scene(N=3)
    for k in range(2):
        sc2.give(0, 2, k, "sand", 0.1)
    sc2.give(0, 0, 0, "flint", 0.1)
    sc2.give(0, 1, 0, "flint", 0.1)
    r = cr.share(sc2.pool, sc2.inv, sc2.act((0, 0), (0, 1)), sc2.pos, torch.tensor([[1, 2, 3]]),
                 torch.tensor([[3, -1, 2]]), torch.ones(1, 3, dtype=torch.bool), radius_m=R_HAB, reach_m=10.0)
    assert r["to"][0].tolist() == [2, -1, -1] and cr.check_inventory(sc2.pool, sc2.inv) == []


def test_cook_meat_gain():
    sc = Scene(N=2)
    meat = sc.give(0, 0, 0, "meat", 0.5, temp=300.0)
    raw = sc.give(0, 1, 0, "meat", 0.5, temp=300.0)
    sc.fire(0, {"charcoal": 3.0}, p=at(0.5))
    sc.pos[0, 1] = at(10.0)
    before = sc.mass()
    r = cr.cook(sc.pool, sc.fires, sc.inv, sc.act((0, 0), (0, 1)), sc.pos, 0.21, radius_m=R_HAB,
                ambient_fire_k=288.0)
    assert r["cooked"][0].tolist() == [True, False] and r["fire"][0, 1] == -1     # 1 is 9.5 m from the fire
    assert sc.pool.holder[0, meat] == 0 and sc.pool.peak_k[0, meat] > float(cr.COOK_K)
    assert sc.pool.peak_k[0, meat] < float(cr.TISSUE_CHAR_K)                     # roasted beside the fire, not charred
    assert r["charred_kg"].sum() == 0 and r["item_heat_j"][0] > 0
    gain = cr.cook_factor(sc.pool)
    assert gain[0, meat].item() == pytest.approx(1.3) and gain[0, raw].item() == 1.0
    assert torch.allclose(sc.mass(), before)
    # the cooked meat cools back toward the air
    sc.step(dt=86400.0)
    assert sc.pool.temp_k[0, meat].item() == pytest.approx(288.0, abs=0.5)
    assert cr.cooked(sc.pool)[0, meat]


# ------------------------------------------------------------------ decay
def test_decay_q10_freezing_and_litter():
    sc = Scene(W=1, N=1)
    temps = [288.0, 298.0, 250.0, 288.0, 288.0]
    kinds = ["meat", "meat", "meat", "wood", "charcoal"]
    idx = [sc.ground(0, s, 1.0, at(5.0 * j), temp=t) for j, (s, t) in enumerate(zip(kinds, temps))]
    tiny = sc.give(0, 0, 0, "meat", 1.05e-4, temp=300.0)
    before = sc.mass()
    cells = torch.arange(sc.pool.alive.shape[1])[None] % 3
    r = cr.decay(sc.pool, 1.0, inv=sc.inv, item_cell=cells, n_cells=3)
    k = float(cr.DECAY_PER_DAY["meat"])
    lost = [1 - sc.pool.mass[0, i].item() for i in idx]
    assert lost[0] == pytest.approx(1 - math.exp(-k), rel=1e-5)
    assert lost[1] == pytest.approx(1 - math.exp(-2 * k), rel=1e-5)           # Q10 = 2
    assert lost[2] == 0.0 and lost[4] == 0.0
    assert lost[3] == pytest.approx(1 - math.exp(-0.1 / 365.25), rel=1e-3)
    assert not sc.pool.alive[0, tiny] and sc.inv[0, 0, 0] == -1               # rotted away from the hand
    assert torch.allclose(before - sc.mass(), r["decayed_kg"], atol=1e-6)
    c = r["decayed_kg"][0] @ E[:, m.ELEMENTS.index("C")]
    assert r["litter_c_kg"].sum().item() == pytest.approx(c.item(), rel=1e-9)
    assert (sc.pool.age_d[0, idx] == 1.0).all()


def test_decay_litter_lands_in_each_items_cell():
    sc = Scene(W=2, N=1)
    kinds, cells = ["meat", "wood", "fat"], [2, 0, 1]
    idx = [[sc.ground(w, s, 1.0 + w, at(5.0 * j)) for j, s in enumerate(kinds)] for w in range(2)]
    item_cell = torch.zeros(2, sc.pool.alive.shape[1], dtype=torch.long)
    for w in range(2):
        item_cell[w, idx[w]] = torch.tensor(cells)
    before = it.species_mass(sc.pool)
    r = cr.decay(sc.pool, 1.0, item_cell=item_cell, n_cells=3)
    c = E[:, m.ELEMENTS.index("C")]
    for w in range(2):
        want = torch.zeros(3, dtype=torch.float64)
        for s, cell in zip(kinds, cells):
            k = float(cr.DECAY_PER_DAY[s])
            want[cell] += (1.0 + w) * (1 - math.exp(-k)) * c[m.IDX[s]]
        assert r["litter_c_kg"][w].tolist() == pytest.approx(want.tolist(), rel=1e-4)
    assert torch.allclose(before - it.species_mass(sc.pool), r["decayed_kg"], atol=1e-6)


# ------------------------------------------------------------------ fire energy, charring, fanning, handling
def test_a_small_fire_cannot_heat_or_calcine_a_boulder():
    """The items in a fire take at most LOAD_SHARE of the heat its bed releases: a 50 g charcoal fire
    barely warms a 100 kg limestone boulder, a 3 kg one heats it only as far as its heat allows; neither calcines it."""
    sc = Scene(W=2)
    rocks = [sc.ground(w, "limestone", 100.0, sc.pos[w, 0]) for w in range(2)]
    sc.fire(0, {"charcoal": 0.05})
    sc.fire(1, {"charcoal": 3.0})
    heat = torch.zeros(2, dtype=torch.float64)
    taken, calcined = heat.clone(), heat.clone()
    peak = [288.0, 288.0]
    for _ in range(30):
        fl = sc.step(dt=3600.0, substeps=2)
        heat += fl["heat_j"]
        taken += fl["item_heat_j"]
        calcined += fl["transformed_kg"][:, TI["limestone_to_lime"]]
        peak = [max(peak[w], sc.pool.temp_k[w, rocks[w]].item()) for w in range(2)]
    assert not sc.fires.alive.any()
    assert heat[0].item() == pytest.approx(0.05 * float(m.COMBUSTION_J_KG["charcoal"]), rel=0.25)
    assert (taken <= float(cr.LOAD_SHARE) * heat * (1 + 1e-9)).all()
    for w in range(2):                     # the heat the rock gained is bounded by the fire's output
        assert 100.0 * float(m.CP["limestone"]) * (peak[w] - 288.0) <= float(cr.LOAD_SHARE) * heat[w].item() * 1.001
    assert peak[0] < 300.0 and peak[1] < 1000.0 and calcined.tolist() == [0.0, 0.0]


def test_smothered_wood_leaves_a_charcoal_item_to_rake_out():
    sc = Scene(N=2)
    wood = sc.ground(0, "wood", 2.0, sc.pos[0, 0])
    sc.fire(0, {"charcoal": 60.0})                                   # a deep bed: the wood is smothered all day
    fl = sc.step(dt=86400.0)
    assert sc.fires.alive[0, 0]
    x = fl["transformed_kg"][0, TI["wood_to_charcoal"]].item()
    y = m.TRANSFORMS[TI["wood_to_charcoal"]].outputs["charcoal"]
    char = sc.pool.comp[0, wood, m.IDX["charcoal"]].item() * sc.pool.mass[0, wood].item()
    assert x > 1.5 and char >= 0.9 * x * y and sc.pool.temp_k[0, wood] > 1000
    # too hot to hold: bare hands leave it, an actor holding a stick rakes it out to cool at its feet
    stock = _stock(1, 1, {})
    args = (torch.full((1, 2), 3, dtype=torch.long), sc.pos, torch.zeros(1, 2, dtype=torch.long), stock,
            torch.zeros(1, 1), torch.ones(1), torch.Generator())
    kw = dict(body_mass_kg=50.0, item_temp_k=288.0, radius_m=R_HAB)
    r = cr.collect(sc.pool, sc.inv, sc.act((0, 0)), *args, **kw)
    assert not r["picked"][0, 0] and not r["raked"][0, 0] and torch.allclose(sc.pool.pos[0, wood], sc.fires.pos[0, 0])
    sc.give(0, 1, 0, "wood", 0.5)
    r = cr.collect(sc.pool, sc.inv, sc.act((0, 1)), *args, **kw)
    assert r["raked"][0, 1] and r["item"][0, 1] == wood and sc.pool.holder[0, wood] == -1
    assert torch.allclose(sc.pool.pos[0, wood], sc.pos[0, 1])
    sc.step(dt=86400.0)                                              # it cools beside the fire and keeps its char
    assert sc.pool.temp_k[0, wood].item() < float(cr.HANDLE_MAX_K)
    assert sc.pool.comp[0, wood, m.IDX["charcoal"]].item() * sc.pool.mass[0, wood].item() == pytest.approx(char, rel=1e-5)
    r = cr.collect(sc.pool, sc.inv, sc.act((0, 1)), *args, **kw)
    assert r["picked"][0, 1] and r["item"][0, 1] == wood and sc.pool.holder[0, wood] // sc.K == 1
    assert cr.check_inventory(sc.pool, sc.inv) == []


def test_meat_hide_and_bone_left_in_a_fire_char_away():
    sc = Scene(W=3)
    for w, s in enumerate(("meat", "hide", "bone")):
        sc.ground(w, s, 1.0, sc.pos[w, 0])
        sc.fire(w, {"charcoal": 10.0})
    food0 = (it.item_props(sc.pool)["food_j"] * sc.pool.alive).sum(1)
    before = sc.mass()
    fl = sc.step(dt=86400.0)
    food1 = (it.item_props(sc.pool)["food_j"] * sc.pool.alive).sum(1)
    assert (food1 < 0.1 * food0).all()
    for w, s in enumerate(("meat", "hide", "bone")):
        assert fl["charred_kg"][w, m.IDX[s]].item() > 0.9
    assert (fl["to_soil_kg"] >= fl["charred_kg"]).all()
    assert element_residual(before, sc.mass(), fl) < 1e-5


def test_one_fan_action_lasts_an_hour():
    sc = Scene(W=2)
    for w in range(2):
        sc.ground(w, "hematite", 1.0, sc.pos[w, 0])
        sc.fire(w, {"charcoal": 10.0})
    r = cr.feed_fire(sc.pool, sc.fires, sc.inv, sc.act((0, 0)), sc.pos, radius_m=R_HAB, fan=1.0)
    assert r["fed_kg"][0, 0] == 0 and sc.fires.air[:, 0].tolist() == [1.0, 0.0]
    fl = sc.step(dt=86400.0)
    # 1713 K for the hour of fanning, then the open charcoal fire at 1384 K, below the bloomery onset of 1400 K
    per_hour = 1.0 / (float(m.TRANSFORMS[TI["hematite_to_iron"]].days) * 24.0)
    x = fl["transformed_kg"][:, TI["hematite_to_iron"]]
    assert 0 < x[0].item() <= per_hour * float(cr.FAN_S) / 3600.0 + 1e-6 and x[1].item() == 0.0
    assert (sc.fires.air == 0).all()
    # hour by hour: fanned in the first call only
    sc2 = Scene()
    sc2.fire(0, {"charcoal": 10.0}, air=1.0)
    sc2.step(dt=3600.0, substeps=2)
    hot = sc2.fires.temp_k[0, 0].item()
    sc2.step(dt=3600.0, substeps=2)
    assert hot == pytest.approx(1713.06, abs=0.1) and sc2.fires.temp_k[0, 0].item() == pytest.approx(1384.2, abs=0.1)


def test_transform_rate_follows_the_items_charge():
    """A dilute basis converts at the rate of its own charge, not of the whole item mass."""
    sc = Scene(W=2)
    sc.ground(0, "clay", 1.0, sc.pos[0, 0])
    sc.ground(1, {"clay": 0.1, "sand": 0.9}, 1.0, sc.pos[1, 0])
    for w in range(2):
        sc.fire(w, {"charcoal": 10.0})
    fl = sc.step(dt=3 * 3600.0, substeps=6)
    share = fl["transformed_kg"][:, TI["clay_to_ceramic"]] / torch.tensor([1.0, 0.1], dtype=torch.float64)
    assert 0.1 < share[0].item() < 0.35 and 0.75 < share[1].item() / share[0].item() < 1.25


def test_make_fire_needs_ten_grams_of_fuel():
    sc = Scene(N=2)
    for n, kg in enumerate((0.005, 0.02)):
        sc.give(0, n, 0, "plant_fiber", kg)
    r = cr.make_fire(sc.pool, sc.fires, sc.inv, sc.act((0, 0), (0, 1)), sc.pos, 0.21, 288.0, body_mass_kg=70.0)
    assert r["lit"][0].tolist() == [False, True] and r["no_fuel"][0].tolist() == [True, False]
    assert sc.inv[0, 0, 0] >= 0 and sc.fires.alive.sum() == 1 and cr.check_inventory(sc.pool, sc.inv) == []


def test_knap_an_assembly_works_its_head_only():
    sc = Scene()
    sc.give(0, 0, 0, "basalt", 1.0)
    sc.give(0, 0, 1, "flint", 0.5)
    sc.give(0, 0, 2, "wood", 0.3)
    cr.combine(sc.pool, sc.inv, sc.act((0, 0)), head_slot=sc.slots(1), handle_slot=sc.slots(2))
    tool = int(sc.inv[0, 0, 1])
    L = sc.pool.handle_len_m[0, tool].item()
    before = sc.mass()
    r = cr.knap(sc.pool, sc.inv, sc.act((0, 0)), sc.pos, body_mass_kg=60.0, effort=1.0)      # defaults
    assert bool(r["ok"][0, 0]) and r["work_j"][0, 0] > 0
    d = int(r["debris"][0, 0])
    assert sc.species(0, d) == {"flint": pytest.approx(1.0)} and sc.pool.mass[0, d].item() == pytest.approx(0.1, rel=1e-5)
    kg = sc.pool.comp[0, tool] * sc.pool.mass[0, tool]
    assert kg[m.IDX["flint"]].item() == pytest.approx(0.4, rel=1e-5) and kg[m.IDX["wood"]].item() == pytest.approx(0.3, rel=1e-5)
    assert sc.pool.head_mass[0, tool].item() == pytest.approx(0.4, rel=1e-5)
    assert sc.pool.handle_len_m[0, tool].item() == L and L > 0
    lim = 1 / (1 + (float(m.TOUGHNESS["flint"]) / float(cr.K_EDGE)) ** 2)    # the flint edge, not the mix
    assert 0.9 * lim < sc.pool.sharp[0, tool].item() <= lim + 1e-6
    assert torch.allclose(sc.mass(), before, atol=1e-6) and cr.check_inventory(sc.pool, sc.inv) == []
    assert cr.tool_of(sc.pool, sc.inv, sc.slots(1), torch.full((1, 4), 60.0))["hardness"][0, 0].item() == pytest.approx(7.0)


def test_an_assembly_works_with_the_hardness_of_its_head():
    sc = Scene(N=2)
    sc.give(0, 0, 0, "copper", 1.0)
    sc.give(0, 0, 1, "flint", 0.6)                       # a stone handle heavier than half the head
    sc.give(0, 1, 0, "wood", 1.0)
    sc.give(0, 1, 1, "bone", 0.6)
    cr.combine(sc.pool, sc.inv, sc.act((0, 0), (0, 1)), head_slot=sc.slots(0), handle_slot=sc.slots(1))
    t = cr.tool_of(sc.pool, sc.inv, sc.slots(0), torch.full((1, 2), 60.0))
    hard = lambda s: float(m.MOHS[s]) * (1 - float(m.GRANULAR[s]))
    assert t["hardness"][0].tolist() == pytest.approx([hard("copper"), hard("wood")])


# ------------------------------------------------------------------ tool effects
def test_tool_effect_functions():
    e = cr.strike_energy(torch.tensor([1.0, 1.0]), torch.tensor([0.0, 0.6]), v_arm=5.0)
    assert e.tolist() == pytest.approx([12.5, 50.0])
    d = cr.strike_damage(torch.tensor(10.0), torch.tensor([0.0, 1.0]), 7.0, 0.5, 0.5)
    assert d[1] / d[0] == pytest.approx(3.0) and d[0].item() == pytest.approx(10 / 1.5)
    soft, hard = (cr.strike_damage(10.0, 0.0, torch.tensor(h), 6.5, 2.6).item() for h in (1.5, 6.5))
    assert soft < hard                                       # a wooden club does less to basalt than a stone
    assert cr.dig_yield(torch.tensor([2.5, 6.5, 7.0]), 7.0).tolist() == pytest.approx([2.5 / 7, 6.5 / 7, 1.0])
    assert cr.dig_yield(torch.tensor(1.0), 0.0).item() == 1.0                    # loose ground
    assert cr.chop_yield(torch.tensor(7.0), torch.tensor(1.0)).item() == pytest.approx(1.0)
    assert cr.chop_yield(torch.tensor(2.5), torch.tensor(0.0)).item() == pytest.approx(float(cr.CHOP_BLUNT))
    assert cr.butcher_yield(torch.tensor(0.0), torch.tensor(7.0)).item() == pytest.approx(float(cr.BUTCHER_BASE))
    assert cr.butcher_yield(torch.tensor(1.0), torch.tensor(7.0)).item() == pytest.approx(1.0)
    # bare limb
    sc = Scene(N=1)
    t = cr.tool_of(sc.pool, sc.inv, None, torch.tensor([[60.0]]))
    assert t["item"][0, 0] == -1 and t["head_mass"][0, 0].item() == pytest.approx(0.022 * 60)
    assert t["hardness"][0, 0].item() == 2.5 and t["sharp"][0, 0] == 0


def test_warmth_within_20_m():
    sc = Scene(N=3)
    sc.pos[0] = torch.stack([at(0.0), at(5.0), at(25.0)])
    sc.fire(0, {"wood": 5.0}, p=at(0.0))
    q = cr.warmth(sc.fires, sc.pos, R_HAB, torch.tensor([0.21]))
    hrr = cr.heat_release_w(sc.fires, torch.tensor([0.21]))[0, 0].item()
    assert hrr > 1e3
    assert q[0, 0].item() == pytest.approx(0.3 * hrr / (4 * math.pi), rel=1e-4)          # capped at 1 m
    assert q[0, 1].item() == pytest.approx(0.3 * hrr / (4 * math.pi * 25.0), rel=1e-3)
    assert q[0, 2].item() == 0.0
    fl = sc.step(dt=86400.0)
    day = cr.warmth_within(fl["fire_pos"], fl["mean_hrr_w"], sc.pos, R_HAB)
    assert 0 < day[0, 1].item() < q[0, 1].item()           # the day's mean is below the first hour's flux


def test_worn_insulation_needs_hide_and_fibre():
    sc = Scene(N=3)
    sc.give(0, 0, 0, {"hide": 3.0, "plant_fiber": 0.2}, 3.2)
    sc.give(0, 1, 0, "hide", 3.0)
    sc.give(0, 2, 0, {"hide": 1.5, "plant_fiber": 0.2}, 1.7)
    ins = cr.worn_insulation(sc.pool, sc.inv, torch.full((1, 3), 50.0))
    assert ins[0].tolist() == pytest.approx([1.0, 0.0, 0.5])


# ------------------------------------------------------------------ whole-tick ledgers and determinism
class World:
    """A small crafting world for whole-tick tests: two worlds of six actors and four cells."""

    def __init__(self, seed: int):
        self.gen = torch.Generator().manual_seed(seed)
        W, N, C = 2, 6, 4
        self.sc = Scene(W=W, N=N, I=96, F=8)
        self.sc.pos = torch.stack([at(2.0 * n, 0.5 * (n % 2)) for n in range(N)]).expand(W, N, 3).clone()
        self.stock = torch.rand(W, C, len(m.CRUST_SPECIES), generator=self.gen) * 5
        self.wood = torch.rand(W, C, generator=self.gen) * 10
        self.cell = torch.randint(0, C, (W, N), generator=self.gen)
        self.area = torch.full((C,), 50.0)
        self.flows = {"in": torch.zeros(W, m.S, dtype=torch.float64), "air": torch.zeros(W, 3, dtype=torch.float64),
                      "soil": torch.zeros(W, m.S, dtype=torch.float64), "rot": torch.zeros(W, m.S, dtype=torch.float64)}
        self.start = self.sc.mass()

    def day(self):
        sc, gen, fl_ = self.sc, self.gen, self.flows
        W, N = sc.W, sc.N
        everyone = torch.ones(W, N, dtype=torch.bool)
        rand = lambda: torch.rand(W, N, generator=gen)
        choice = torch.randint(0, 4, (W, N), generator=gen)
        r = cr.collect(sc.pool, sc.inv, everyone, choice, sc.pos, self.cell, self.stock, self.wood, self.area, gen,
                       body_mass_kg=40.0, item_temp_k=288.0, radius_m=R_HAB)
        fl_["in"] += r["ground_kg"]
        fl_["in"][:, m.IDX["wood"]] += r["wood_kg"].sum(1)
        self.wood -= (r["wood_kg"] / self.area).float()
        half = rand() < 0.5
        cr.knap(sc.pool, sc.inv, half, sc.pos, body_mass_kg=60.0)
        cr.combine(sc.pool, sc.inv, ~half & (rand() < 0.3))
        cr.make_fire(sc.pool, sc.fires, sc.inv, rand() < 0.3, sc.pos, 0.21, 288.0, body_mass_kg=40.0)
        cr.feed_fire(sc.pool, sc.fires, sc.inv, rand() < 0.3, sc.pos, radius_m=R_HAB)
        cr.heat_item(sc.pool, sc.fires, sc.inv, rand() < 0.3, sc.pos, radius_m=R_HAB)
        ck = cr.cook(sc.pool, sc.fires, sc.inv, rand() < 0.2, sc.pos, 0.21, radius_m=R_HAB, ambient_fire_k=288.0)
        fl_["air"] += ck["air_kg"]
        fl_["soil"] += ck["to_soil_kg"]
        cr.share(sc.pool, sc.inv, rand() < 0.2, sc.pos, torch.arange(N).expand(W, N), torch.zeros(W, N, dtype=torch.long),
                 everyone, radius_m=R_HAB)
        cr.drop(sc.pool, sc.inv, rand() < 0.2, sc.pos)
        cr.sync_held(sc.pool, sc.pos, sc.K)
        fl = sc.step(dt=86400.0, substeps=12)
        assert (fl["item_heat_j"] <= float(cr.LOAD_SHARE) * fl["heat_j"] * (1 + 1e-9) + 1e-9).all()
        fl_["air"] += fl["air_kg"]
        fl_["soil"] += fl["to_soil_kg"]
        fl_["rot"] += cr.decay(sc.pool, 1.0, inv=sc.inv)["decayed_kg"]
        assert cr.check_inventory(sc.pool, sc.inv) == []

    def state(self):
        return {"pool": it.state_dict(self.sc.pool), "fires": it.state_dict(self.sc.fires), "inv": self.sc.inv.clone(),
                "stock": self.stock.clone(), "wood": self.wood.clone(), "cell": self.cell.clone(), "gen": self.gen.get_state()}

    def load(self, st):
        self.sc.pool, self.sc.fires = it.from_state(st["pool"]), it.from_state(st["fires"])
        self.sc.inv, self.stock, self.wood = st["inv"].clone(), st["stock"].clone(), st["wood"].clone()
        self.cell = st["cell"].clone()
        self.gen.set_state(st["gen"])

    def hashes(self):
        return (it.state_hash(self.sc.pool), it.state_hash(self.sc.fires), self.sc.inv.tolist(),
                self.stock.numpy().tobytes(), self.wood.numpy().tobytes())


def _scenario(seed: int, days: int = 4):
    w = World(seed)
    for _ in range(days):
        w.day()
    return w.sc, w.start, w.flows, w.stock


def test_whole_tick_ledgers_close():
    sc, start, flows, _ = _scenario(5)
    end = sc.mass()
    want = start + flows["in"] - flows["soil"] - flows["rot"]
    # species change by transforms and burning; elements and total mass must close
    el = (end - want) @ E + flows["air"] @ EA
    scale = max(1.0, float(want.sum()))
    assert float(el.abs().max()) < 1e-5 * scale
    total = end.sum(1) - want.sum(1) + flows["air"].sum(1)
    assert float(total.abs().max()) < 1e-5 * scale
    assert float(flows["in"].sum()) > 0


def test_determinism_same_seed_same_state():
    a, _, fa, sa = _scenario(11)
    b, _, fb, sb = _scenario(11)
    c, _, _, _ = _scenario(12)
    assert it.state_hash(a.pool) == it.state_hash(b.pool) and it.state_hash(a.fires) == it.state_hash(b.fires)
    assert torch.equal(a.inv, b.inv) and torch.equal(sa, sb) and torch.equal(fa["air"], fb["air"])
    assert it.state_hash(a.pool) != it.state_hash(c.pool)


def test_exact_resume_on_cpu():
    whole = World(21)
    for _ in range(4):
        whole.day()
    first = World(21)
    for _ in range(2):
        first.day()
    saved = first.state()
    resumed = World(99)                      # a different world, then loaded from the saved state
    resumed.load(saved)
    for _ in range(2):
        resumed.day()
    assert resumed.hashes() == whole.hashes()


def test_provenance_tags():
    prov = cr.provenance()
    assert len(prov) > 60
    for key, (tag, source) in prov.items():
        assert tag in m.TAGS and source, key
    assert prov["crafting.X_O2_MIN"][0] == "reference" and "Belcher" in prov["crafting.X_O2_MIN"][1]


# ------------------------------------------------------------------ integration round (world review)
def test_a_float64_stock_loses_exactly_what_collect_makes():
    """A 3 kg lump from a 2670 kg/m^2 stock on a 9e4 m^2 cell is below float32 resolution: a float32 stock never
    depletes. A float64 stock (as the world keeps it) loses what the item holds."""
    sc = Scene(N=2)
    st32 = _stock(1, 1, {(0, 0, "granite"): 2670.0})
    st64 = st32.double()
    area = torch.full((1,), 90903.0)
    cell = torch.zeros(1, 2, dtype=torch.long)
    gi = m.CRUST_SPECIES.index("granite")
    for st in (st32, st64):
        sc2 = Scene(N=2)
        before = st.clone()
        r = cr.collect(sc2.pool, sc2.inv, sc2.act((0, 0), (0, 1)), torch.zeros(1, 2, dtype=torch.long), sc2.pos, cell,
                       st, torch.zeros(1, 1), area, torch.Generator().manual_seed(3), body_mass_kg=30.0,
                       item_temp_k=288.0, radius_m=R_HAB)
        made = float(r["ground_kg"][0, m.IDX["granite"]])
        lost = float((before - st)[0, 0, gi].double() * area[0])
        assert made > 0.5
        if st.dtype == torch.float64:
            assert lost == pytest.approx(made, abs=1e-6)            # float64 kg/m^2 x 9e4 m^2: 4e-8 kg
        else:
            assert lost == 0.0                     # the float32 stock does not resolve it


def test_tool_of_reads_the_chosen_hands_props():
    sc = Scene(N=3)
    sc.give(0, 0, 0, "flint", 0.4, sharp=0.6)
    sc.give(0, 0, 1, "basalt", 2.0)
    sc.give(0, 1, 2, "bone", 0.3)
    M = torch.full((1, 3), 40.0)
    tool = cr.tool_of(sc.pool, sc.inv, None, M)
    p = cr.item_props(sc.pool, sc.inv)
    slot = cr._best_slot(p["work_hardness"], p["present"])
    i = cr.held(sc.inv, slot)
    q = cr.item_props(sc.pool, i)                  # the reference: props of the chosen items again
    for n in (0, 1):
        assert int(tool["item"][0, n]) == int(i[0, n])
        assert float(tool["hardness"][0, n]) == pytest.approx(float(q["work_hardness"][0, n]))
        assert float(tool["head_mass"][0, n]) == pytest.approx(float(q["mass"][0, n]))
        assert float(tool["sharp"][0, n]) == pytest.approx(float(q["sharp"][0, n]))
        assert float(tool["toughness"][0, n]) == pytest.approx(float(q["toughness"][0, n]))
    assert int(tool["item"][0, 2]) == -1 and float(tool["hardness"][0, 2]) == float(cr.BARE_HARDNESS)
    assert float(tool["head_mass"][0, 2]) == pytest.approx(float(cr.LIMB_FRACTION) * 40.0)


def test_fire_step_ends_early_once_every_fire_is_out():
    """A fire that dies within the first substeps: the rest of the day is one relaxation of the items, which
    lands where the full substep loop lands (and the fire's flows are the same)."""
    out = []
    for check in (8, 10 ** 6):
        old = cr.EXIT_CHECK
        cr.EXIT_CHECK = m.Val(check, "new_rule", "test")
        try:
            sc = Scene()
            sc.fire(0, {"plant_fiber": 0.05})
            i = sc.ground(0, "basalt", 2.0, at(50.0), temp=600.0)
            fl = sc.step(substeps=48)
            out.append((fl["heat_j"].clone(), fl["air_kg"].clone(), float(sc.pool.temp_k[0, i]), sc.fires.alive.clone()))
        finally:
            cr.EXIT_CHECK = old
    (h1, a1, t1, f1), (h2, a2, t2, f2) = out
    assert not f1.any() and not f2.any()
    assert torch.allclose(h1, h2) and torch.allclose(a1, a2)
    assert t1 == pytest.approx(t2, abs=1.0) and abs(t1 - 288.0) < 5.0


def test_strike_helpers_keep_scalars_on_the_tensors_device():
    e = cr.strike_energy(torch.tensor([1.0]), torch.tensor([0.3]))
    d = cr.strike_damage(e, torch.tensor([0.5]), torch.tensor([6.0]), 2.5, torch.tensor([2.0]))
    y = cr.dig_yield(torch.tensor([3.0]), 6.0)
    assert e.device == d.device == y.device and float(y) == pytest.approx(0.5)
