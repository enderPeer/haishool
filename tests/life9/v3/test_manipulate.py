"""Tests of life9 v3's contact physics of the body's abilities on things (PLANET-V3-SPEC section 8): grip, release,
place, force (knapping and wounds), rub (friction heat, ember and tinder), press (binding), and fire, transforms,
cooking and decay on patch coordinates.

Gates: rubbing dry wood with enough work makes a fire on tinder at 21 % O2 and not at 3 %; knapping flint with
basalt sharpens it, with wood it does not; clay placed in a hot enough fire becomes ceramic; flesh placed in a fire
cooks; mass and elements balance; no named purposeful action exists.
"""
from __future__ import annotations

import ast
import math
import re
from pathlib import Path

import pytest
import torch

from haishool.life9.planet import crafting as cr
from haishool.life9.planet import items as it
from haishool.life9.planet import materials as m
from haishool.life9.v3 import manipulate as mp

L = 2048.0
DAY = 21600.0
ROOT = Path(__file__).resolve().parents[3]
TAGS = {"chain", "derived", "reference", "new_rule"}


def comp(species):
    return m.species_vector(species if isinstance(species, dict) else {species: 1.0})[None]


def stick_len(kg, species="wood"):
    return float(mp.stick_length_m(torch.tensor(kg), torch.tensor(float(m.DENSITY[species]))))


class Scene:
    """A arenas of N bodies 10 m apart on a row, mouths 0.3 m east of their centres; peak and sustained power."""

    def __init__(self, A=1, N=4, I=64, F=8, mass_kg=70.0, peak_w=1000.0, sustained_w=300.0):
        self.A, self.N = A, N
        self.items, self.fires = mp.new_pools(A, I, F)
        self.held = mp.new_held(A, N)
        self.pos = torch.stack([torch.tensor([100.0 + 10.0 * n, 100.0]) for n in range(N)]).expand(A, N, 2).clone()
        self.mouth = self.pos + torch.tensor([0.3, 0.0])
        self.alive = torch.ones(A, N, dtype=torch.bool)
        self.mass = torch.full((A, N), float(mass_kg))
        self.peak = torch.full((A, N), float(peak_w))
        self.sus = torch.full((A, N), float(sustained_w))
        self.ledger = mp.new_ledger(self.items, self.fires)

    def loose(self, a, species, kg, xy, temp=288.0, length=0.0):
        i = mp.spawn_items(self.items, [a], comp(species), torch.tensor([float(kg)]), torch.tensor([xy]),
                           torch.tensor([temp]), ledger=self.ledger, L=L, handle_len_m=length)
        assert i[0] >= 0
        return int(i[0])

    def at_mouth(self, a, n, species, kg, temp=288.0, dx=0.0, length=0.0):
        return self.loose(a, species, kg, (self.mouth[a, n, 0].item() + dx, self.mouth[a, n, 1].item()), temp, length)

    def give(self, a, n, k, species, kg, temp=288.0, length=0.0):
        """An item put straight into grip k (spawned at the mouth, booked as brought in)."""
        i = self.at_mouth(a, n, species, kg, temp, length=length)
        cr._give(self.items, self.held, torch.tensor([a]), torch.tensor([n]), k, torch.tensor([i]))
        return i

    def fire(self, a, bed: dict, xy, temp=1000.0):
        f = it.spawn_fire(self.fires, torch.tensor([a]), torch.tensor([[xy[0], xy[1], 0.0]]),
                          m.species_vector(bed)[None], torch.tensor([temp]))
        assert f[0] >= 0
        self.ledger.book_in(torch.zeros(self.A, m.S, dtype=torch.float64).index_add_(
            0, torch.tensor([a]), m.species_vector(bed, dtype=torch.float64)[None]))
        return int(f[0])

    def mask(self, *who):
        x = torch.zeros(self.A, self.N, dtype=torch.bool)
        for a, n in who:
            x[a, n] = True
        return x

    def level(self, value, *who):
        x = torch.zeros(self.A, self.N)
        for a, n in who:
            x[a, n] = value
        return x

    def rub(self, intensity=1.0, x_o2=0.21, dt=DAY, **kw):
        return mp.rub(self.items, self.fires, self.held, intensity, alive=self.alive, mouth_pos=self.mouth,
                      sustained_w=self.sus, x_o2=x_o2, air_k=288.0, L=L, dt_s=dt, body_mass_kg=self.mass,
                      ledger=self.ledger, **kw)

    def strike(self, intensity=1.0, dt=DAY, **kw):
        return mp.force(self.items, self.held, intensity, alive=self.alive, pos=self.pos, mouth_pos=self.mouth,
                        body_mass_kg=self.mass, peak_w=self.peak, sustained_w=self.sus, L=L, dt_s=dt,
                        air_k=288.0, ledger=self.ledger, **kw)

    def heat(self, dt=DAY, x_o2=0.21, **kw):
        return mp.heat_step(self.items, self.fires, self.held, x_o2=x_o2, air_k=288.0, L=L, dt_s=dt,
                            ledger=self.ledger, **kw)

    def kg(self, a, i, species):
        return float(self.items.comp[a, i, m.IDX[species]] * self.items.mass[a, i]) if self.items.alive[a, i] else 0.0

    def species_total(self, species):
        return cr.ledger_mass(self.items, self.fires)[:, m.IDX[species]]

    def balance(self):
        err = mp.ledger_error(self.ledger, self.items, self.fires)
        scale = max(1.0, float(err["total_kg"].abs().max()))
        return float(err["mass_kg"].abs().max()) / scale, float(err["elements_kg"].abs().max()) / scale


def one_stroke_dt(sc, strokes=1):
    """A bout length in which a body striking for the whole bout makes exactly ``strokes`` strokes."""
    sw = mp.swing(sc.items, sc.held, body_mass_kg=sc.mass, peak_w=sc.peak)
    rate = torch.minimum(1.0 / sw["cycle_s"], mp.LIMB_SUSTAINED_SHARE * sc.sus / sw["swing_j"])
    return (strokes + 0.5) / float(rate.min()), sw


# ------------------------------------------------------------------------------------------- rub and fire (gate)
def _fire_kit(sc, a, n, stick=0.1, board=0.5, tinder=0.03):
    """A dry stick in the left grip, a board under the mouth and a tuft of dry fibre beside it."""
    sc.give(a, n, 0, "wood", stick, length=stick_len(stick))
    b = sc.at_mouth(a, n, "wood", board)
    t = sc.at_mouth(a, n, "plant_fiber", tinder, dx=0.05) if tinder else -1
    return b, t


def test_rubbing_dry_wood_with_enough_work_makes_a_fire_on_tinder_at_21_percent_o2_not_at_3():
    sc = Scene(A=2, N=1, sustained_w=800.0)                   # the limb rubs at a quarter of it, 200 W
    boards, tinders = zip(*[_fire_kit(sc, a, 0) for a in range(2)])
    x = torch.tensor([0.21, 0.03])
    wood0 = sc.species_total("wood").clone()
    r = sc.rub(x_o2=x)
    # the same friction work in both arenas: 200 W for the bout's strokes, the spot at the pyrolysis cap
    assert torch.allclose(r["work_j"], 200.0 * r["time_s"]) and (r["time_s"] > 0.99 * DAY).all()
    assert mp.LIMB_SUSTAINED_SHARE == 0.25
    assert (r["contact_k"] > float(m.IGNITION_K["wood"])).all()
    assert r["ember"].tolist() == [[True], [False]]
    assert r["fire"][0, 0] >= 0 and r["fire"][1, 0] == -1 and r["tinder"][0, 0] == tinders[0]
    assert sc.fires.alive[0].sum() == 1 and sc.fires.alive[1].sum() == 0
    f = int(r["fire"][0, 0])
    bed = sc.fires.fuel_kg[0, f]
    assert bed[m.IDX["plant_fiber"]].item() == pytest.approx(0.03, rel=1e-5)           # the tinder caught
    assert bed[m.IDX["wood"]].item() == pytest.approx(mp.EMBER_KG, rel=1e-4)            # and the ember is in it
    assert sc.kg(0, boards[0], "wood") == pytest.approx(0.5 - mp.EMBER_KG, rel=1e-5)
    assert sc.held[0, 0, 0] >= 0 and sc.held[1, 0, 0] >= 0                              # the sticks stay in hand
    assert torch.allclose(sc.species_total("wood")[1], wood0[1])                          # no fire at 3 %
    assert sc.kg(1, tinders[1], "plant_fiber") == pytest.approx(0.03)
    assert not mp.check(sc.items, sc.held)
    # the tinder burns for the bout and warms the board in its bed (30 g of fibre cannot bring a 0.5 kg board to
    # ignition: that needs kindling), and the ledgers close
    t_board = sc.items.temp_k[0, boards[0]].item()
    fl = sc.heat(x_o2=x)
    assert fl["burnt_kg"][0, m.IDX["plant_fiber"]].item() > 0.9 * 0.03
    assert fl["burnt_kg"][1].sum() == 0 and fl["item_heat_j"][0] > 0
    assert sc.items.peak_k[0, boards[0]].item() > t_board + 20.0
    assert max(sc.balance()) < 1e-6


def test_an_ember_without_tinder_dies():
    sc = Scene(A=1, N=1, sustained_w=800.0)
    board, _ = _fire_kit(sc, 0, 0, tinder=0)
    total0 = cr.ledger_mass(sc.items, sc.fires).clone()
    r = sc.rub()
    assert r["ember"].all() and (r["fire"] < 0).all() and not sc.fires.alive.any()
    assert torch.allclose(cr.ledger_mass(sc.items, sc.fires), total0)                   # no matter moved
    assert sc.kg(0, board, "wood") == pytest.approx(0.5)


def test_too_little_rubbing_power_does_not_ignite_and_only_warms():
    sc = Scene(A=2, N=1, sustained_w=40.0)                     # a 10 W limb: 1 W of heat at the spot
    for a in range(2):
        _fire_kit(sc, a, 0)
    r = sc.rub()
    assert (r["contact_k"] < float(m.IGNITION_K["plant_fiber"])).all() and (r["contact_k"] > 300.0).all()
    assert not r["ember"].any() and (r["fire"] < 0).all() and not sc.fires.alive.any()
    warm = sc.items.temp_k[sc.items.alive]
    assert (warm >= 288.0).all() and (warm < 300.0).all() and (warm > 288.0).any()      # the bulk warms a little
    assert torch.allclose(r["heat_j"], 0.1 * 10.0 * r["time_s"], rtol=1e-5)


def test_rubbing_a_held_fibre_bundle_a_bare_limb_and_stone_on_the_ground():
    sc = Scene(A=1, N=3, sustained_w=800.0)
    sc.give(0, 0, 0, "wood", 0.1, length=stick_len(0.1))
    fibre = sc.give(0, 0, 1, "plant_fiber", 0.05)            # tinder in the other grip, rubbed by the stick
    sc.at_mouth(0, 1, "wood", 0.5)                            # body 1 rubs a board with a bare limb
    sc.give(0, 2, 0, "flint", 0.3)                            # body 2 rubs stone on the ground: no fuel, no fire
    r = sc.rub(dt=3600.0)
    assert r["fire"][0, 0] >= 0 and r["tinder"][0, 0] == fibre and sc.held[0, 0].tolist()[1] == -1
    assert sc.held[0, 0, 0] >= 0                                                         # the stick stays
    # skin chars at its cap before wood can ignite: a burn, no fire
    assert r["fire"][0, 1] == -1 and not r["ember"][0, 1]
    assert r["skin_contact_k"][0, 1].item() == pytest.approx(float(cr.TISSUE_CHAR_K))
    assert r["skin_char_j"][0, 1] > 0 and r["skin_char_kg"][0, 1] > 0 and r["skin_heat_j"][0, 1] > 0
    assert r["skin_contact_k"][0, 0] == 288.0 and r["skin_heat_j"][0, 0] == 0 and r["skin_char_j"][0, 0] == 0
    assert r["fire"][0, 2] == -1 and r["partner_b"][0, 2] == -1 and r["partner_a"][0, 2] >= 0
    assert r["energy"]["ground_j"][0, 2] > 0 and sc.held[0, 2, 0] >= 0


def test_no_fire_slot_means_no_fire_and_the_items_stay():
    sc = Scene(A=1, N=1, F=0, sustained_w=800.0)
    _fire_kit(sc, 0, 0)
    r = sc.rub(dt=3600.0)
    assert r["ember"].all() and r["fire"][0, 0] == -1 and r["no_fire_slot"][0] == 1
    assert sc.held[0, 0, 0] >= 0 and sc.items.alive.sum() == 3


def test_wet_spot_does_not_ignite():
    sc = Scene(A=2, N=1, sustained_w=800.0)
    for a in range(2):
        _fire_kit(sc, a, 0)
    r = sc.rub(dt=3600.0, wet=torch.tensor([[True], [False]]))
    assert r["fire"][0, 0] == -1 and not r["ember"][0, 0] and r["fire"][1, 0] >= 0


# ------------------------------------------------------------------------------------------- knapping (gate)
def test_knapping_flint_with_basalt_sharpens_with_wood_not():
    sc = Scene(A=2, N=1)
    sc.give(0, 0, 0, "basalt", 0.5)
    sc.give(1, 0, 0, "wood", 0.5)
    cores = [sc.at_mouth(a, 0, "flint", 1.0) for a in range(2)]
    total0 = cr.ledger_mass(sc.items, sc.fires).sum(1)
    dt, _ = one_stroke_dt(sc)
    r = sc.strike(dt=dt)
    assert r["strokes"].tolist() == [[1.0], [1.0]] and r["hits"].tolist() == [[1.0], [1.0]]
    assert r["hit_item"][0, 0] == cores[0] and r["hit_item"][1, 0] == cores[1]
    assert r["knapped"][0, 0] and not r["knapped"][1, 0]
    assert sc.items.sharp[0, cores[0]] > 0.2 and sc.items.sharp[1, cores[1]] == 0
    d = int(r["debris"][0, 0])
    lim = 1.0 / (1.0 + (float(m.TOUGHNESS["flint"]) / float(cr.K_EDGE)) ** 2)
    assert sc.items.sharp[0, d].item() == pytest.approx(lim, rel=1e-5)          # a fresh flake at the edge limit
    assert sc.items.mass[0, d].item() == pytest.approx(r["blow_j"][0, 0].item() / mp.KNAP_J_PER_KG, rel=1e-5)
    assert r["debris"][1, 0] == -1 and sc.items.mass[1, cores[1]] == 1.0
    assert torch.allclose(cr.ledger_mass(sc.items, sc.fires).sum(1), total0)     # mass only moved
    # a second stroke sharpens further toward the limit
    s1 = sc.items.sharp[0, cores[0]].item()
    sc.strike(dt=dt)
    assert s1 < sc.items.sharp[0, cores[0]].item() <= lim


def test_many_strokes_in_a_bout_take_many_flakes_into_one_heap():
    sc = Scene(A=1, N=1)
    sc.give(0, 0, 0, "basalt", 0.5)
    core = sc.at_mouth(0, 0, "flint", 1.0)
    dt, sw = one_stroke_dt(sc, strokes=10)
    r = sc.strike(dt=dt)
    flake = sw["blow_j"][0, 0].item() / mp.KNAP_J_PER_KG
    assert r["strokes"][0, 0] == 10 and r["flakes"][0, 0] == 10
    assert r["removed_kg"][0, 0].item() == pytest.approx(10 * flake, rel=1e-4)
    assert sc.items.mass[0, core].item() == pytest.approx(1.0 - 10 * flake, rel=1e-4)
    assert int((sc.items.alive[0]).sum()) == 3                                   # hammer, core and one heap
    # a whole bout of strokes works a fresh core down to its last piece, never below crafting.MIN_ITEM_KG
    sc = Scene(A=1, N=1)
    sc.give(0, 0, 0, "basalt", 0.5)
    core = sc.at_mouth(0, 0, "flint", 1.0)
    r = sc.strike()
    assert r["strokes"][0, 0] > 1000 and r["flakes"][0, 0] > 10
    assert float(cr.MIN_ITEM_KG) <= sc.items.mass[0, core] < 0.01
    assert sc.items.mass[0, int(r["debris"][0, 0])] == pytest.approx(1.0 - sc.items.mass[0, core].item(), rel=1e-5)


def test_held_core_is_struck_when_nothing_touches_the_mouth():
    sc = Scene(A=1, N=1)
    sc.give(0, 0, 0, "basalt", 0.5)
    core = sc.give(0, 0, 1, "flint", 0.8)
    dt, _ = one_stroke_dt(sc)
    r = sc.strike(dt=dt)
    assert r["hit_item"][0, 0] == core and r["knapped"][0, 0] and sc.items.sharp[0, core] > 0
    assert sc.items.holder[0, int(r["debris"][0, 0])] == -1                    # the flake falls


def test_weak_blows_and_bare_hands_do_not_knap_flint():
    sc = Scene(A=2, N=1, mass_kg=0.02, peak_w=0.5, sustained_w=0.15)   # a 20 g body with half a watt of muscle
    sc.give(0, 0, 0, "basalt", 0.005)
    for a in range(2):
        sc.at_mouth(a, 0, "flint", 0.05)
    r = sc.strike()
    assert r["blow_j"][0, 0] < mp.KNAP_J_PER_KG * float(cr.MIN_ITEM_KG)          # a flake would be below 0.1 g
    assert not r["knapped"].any() and r["strokes"][0, 0] > 100
    sc2 = Scene(A=1, N=1)
    sc2.at_mouth(0, 0, "flint", 1.0)
    assert not sc2.strike()["knapped"].any()                                       # a hand is softer than flint


# ------------------------------------------------------------------------------------------- wounds
def test_force_on_a_body_in_contact_wounds_it_and_an_edge_wounds_more():
    sc = Scene(A=1, N=4)
    sc.pos[0, 1] = sc.mouth[0, 0] + torch.tensor([0.2, 0.0])   # body 1 against body 0's mouth
    sc.pos[0, 3] = sc.mouth[0, 2] + torch.tensor([0.2, 0.0])   # body 3 against body 2's mouth
    sc.mouth[0, 1] = sc.pos[0, 1] + torch.tensor([0.0, 0.3])   # 1 and 3 look away
    sc.mouth[0, 3] = sc.pos[0, 3] + torch.tensor([0.0, 0.3])
    blade = sc.give(0, 2, 0, "flint", 0.2)
    sc.items.sharp[0, blade] = 0.7
    blunt = sc.give(0, 0, 0, "flint", 0.2)
    dt, _ = one_stroke_dt(sc, strokes=3)
    r = sc.strike(sc.level(1.0, (0, 0), (0, 2)), dt=dt)
    assert r["hit_body"][0].tolist() == [1, -1, 3, -1] and r["hits"][0, 0] == 3
    assert r["wound_j"][0, 0] == 0 and r["wound_j"][0, 2] == 0
    assert r["wound_j"][0, 1] > 0 and r["wound_j"][0, 3] > 2.0 * r["wound_j"][0, 1]
    blow = r["blow_j"][0, 0]
    skin = float(cr.K_SOFT) / (float(cr.K_SOFT) + float(m.TOUGHNESS["hide"]))
    assert r["wound_j"][0, 1].item() == pytest.approx(3 * blow.item() * skin, rel=1e-5)
    w = r["wound_j"][0, 1].item()                         # no generator: the expected share of head strokes
    assert r["wound_head_j"][0, 1].item() == pytest.approx(mp.HEAD_SHARE * w, rel=1e-5)
    assert r["damage"][0, 1].item() == pytest.approx(1 - math.exp(
        -mp.HEAD_SHARE * w / (mp.WOUND_J_PER_KG * 70.0) - (1 - mp.HEAD_SHARE) * w / (mp.WOUND_BODY_J_PER_KG * 70.0)),
        rel=1e-5)
    assert sc.items.mass[0, blunt] == pytest.approx(0.2)                        # the blow went into the body
    # out of reach: nothing is struck, the strokes still cost their work
    sc.pos[0, 1] += torch.tensor([5.0, 0.0])
    r = sc.strike(sc.level(1.0, (0, 0)), dt=dt)
    assert r["hit_body"][0, 0] == -1 and r["wound_j"].sum() == 0 and r["work_j"][0, 0] > 0


# ------------------------------------------------------------------------------------------- grip, place, release
def test_grip_lifts_only_what_the_muscles_can_and_the_strongest_wins_a_contest():
    sc = Scene(A=1, N=3, peak_w=100.0)                       # 100 N of lifting force at peak power
    sc.mouth[0, 1] = sc.mouth[0, 0] + torch.tensor([0.05, 0.0])
    sc.peak[0, 1] = 200.0
    boulder = sc.at_mouth(0, 0, "granite", 30.0, dx=0.1)   # 294 N: too heavy for either, and nearest to both
    stone = sc.at_mouth(0, 0, "flint", 2.0, dx=0.2)           # 19.6 N, in contact with both mouths
    far = sc.at_mouth(0, 2, "flint", 0.1, dx=3.0)            # 3 m away: not in contact
    act = torch.zeros(1, 3, 2, dtype=torch.bool)
    act[0, :, 0] = True
    g = mp.grip(sc.items, sc.held, act, mouth_pos=sc.mouth, peak_w=sc.peak, gravity=9.81, L=L,
                body_mass_kg=sc.mass)
    # the grips close on the nearest item, the boulder, and cannot lift it: they fail (review L12: a grip does not
    # reach past what is in front for something lighter)
    assert (g["item"] == -1).all() and sc.items.holder[0, boulder] == -1 and sc.items.holder[0, stone] == -1
    sc.items.alive[0, boulder] = False                                             # the boulder gone ...
    g = mp.grip(sc.items, sc.held, act, mouth_pos=sc.mouth, peak_w=sc.peak, gravity=9.81, L=L,
                body_mass_kg=sc.mass)
    assert g["item"][0, 1, 0] == stone and g["item"][0, 0, 0] == -1              # ... the stronger body won it
    assert g["item"][0, 2, 0] == -1 and sc.items.holder[0, far] == -1
    assert sc.items.holder[0, stone] == 1 * 2 + 0
    reach = mp.body_radius_m(torch.tensor(70.0)).item()
    assert g["work_j"][0, 1].item() == pytest.approx(2.0 * 9.81 * reach, rel=1e-5)
    assert not mp.check(sc.items, sc.held)
    # the weight already carried counts: 90 N held leaves 10 N, not enough for another 2 kg
    sc2 = Scene(A=1, N=1, peak_w=100.0)
    sc2.give(0, 0, 0, "granite", 9.0)
    sc2.at_mouth(0, 0, "flint", 2.0)
    act = torch.ones(1, 1, 2, dtype=torch.bool)
    g = mp.grip(sc2.items, sc2.held, act, mouth_pos=sc2.mouth, peak_w=sc2.peak, gravity=9.81, L=L,
                body_mass_kg=sc2.mass)
    assert (g["item"] == -1).all()


def test_a_long_stick_is_touched_at_its_end():
    sc = Scene(A=1, N=1)
    length = stick_len(0.3)
    stick = sc.at_mouth(0, 0, "wood", 0.3, dx=0.25 + 0.45 * length, length=length)   # centre beyond the reach
    c = mp.contacts(sc.items, sc.fires, alive=sc.alive, pos=sc.pos, mouth_pos=sc.mouth, body_mass_kg=sc.mass, L=L)
    assert c["item"][0, 0] == stick
    compact = Scene(A=1, N=1)
    compact.at_mouth(0, 0, "wood", 0.3, dx=0.25 + 0.45 * length)
    c = mp.contacts(compact.items, compact.fires, alive=compact.alive, pos=compact.pos, mouth_pos=compact.mouth,
                    body_mass_kg=compact.mass, L=L)
    assert c["item"][0, 0] == -1


def test_contact_across_the_periodic_boundary():
    sc = Scene(A=1, N=1)
    sc.mouth[0, 0] = torch.tensor([0.05, 1000.0])
    across = sc.loose(0, "flint", 0.5, (L - 0.05, 1000.0))
    act = torch.ones(1, 1, 2, dtype=torch.bool)
    g = mp.grip(sc.items, sc.held, act, mouth_pos=sc.mouth, peak_w=sc.peak, gravity=9.81, L=L,
                body_mass_kg=sc.mass)
    assert g["item"][0, 0, 0] == across


def test_place_into_a_fire_and_release_beside_it():
    sc = Scene(A=1, N=2)
    f = sc.fire(0, {"wood": 2.0}, (sc.mouth[0, 0, 0].item() + 0.3, sc.mouth[0, 0, 1].item()))
    a_item = sc.give(0, 0, 0, "clay", 0.5)
    b_item = sc.give(0, 0, 1, "flint", 0.5)
    c_item = sc.give(0, 1, 1, "flint", 0.5)
    pl = mp.place(sc.items, sc.fires, sc.held, sc.mask((0, 0), (0, 1)), mouth_pos=sc.mouth, L=L)
    assert pl["item"][0].tolist() == [a_item, c_item] and pl["fire"][0].tolist() == [f, -1]
    assert torch.allclose(sc.items.pos[0, a_item, :2], sc.mouth[0, 0])           # at the mouth, in the bed
    d = (sc.items.pos[0, a_item, :2] - sc.fires.pos[0, f, :2]).norm()
    assert 0 < d <= float(cr.FIRE_RADIUS_M)
    assert torch.allclose(sc.items.pos[0, c_item, :2], sc.mouth[0, 1]) and sc.items.pos[0, c_item, 2] == 0
    assert sc.held[0, 0, 0] == -1 and sc.held[0, 0, 1] == b_item
    rl = mp.release(sc.items, sc.fires, sc.held, sc.mask((0, 0)), mouth_pos=sc.mouth, L=L)
    assert rl["item"][0, 0].tolist() == [-1, b_item] and (sc.held == -1).all()
    assert not mp.check(sc.items, sc.held)


# ------------------------------------------------------------------------------------------- press
def test_press_binds_two_held_items_with_a_separate_binder_and_not_without():
    sc = Scene(A=2, N=1)
    heads, handles = [], []
    length = stick_len(0.5)
    for a in range(2):
        handles.append(sc.give(a, 0, 0, "wood", 0.5, length=length))     # near end in the left grip
        heads.append(sc.give(a, 0, 1, "flint", 0.2))                       # far end in the right grip
    fibre = sc.at_mouth(0, 0, "plant_fiber", 0.02)            # binder under arena 0's mouth only
    total0 = cr.ledger_mass(sc.items, sc.fires).sum(1)
    r = mp.press(sc.items, sc.held, torch.ones(2, 1, dtype=torch.bool), alive=sc.alive, mouth_pos=sc.mouth,
                 sustained_w=sc.sus, L=L, body_mass_kg=sc.mass)
    j = handles[0]
    assert r["item"][0, 0] == j and r["binder"][0, 0] == fibre and r["item"][1, 0] == -1
    assert r["pressed"].all()
    assert sc.items.mass[0, j].item() == pytest.approx(0.72, rel=1e-5)
    assert sc.items.head_mass[0, j].item() == pytest.approx(0.2, rel=1e-5)
    assert sc.items.handle_len_m[0, j].item() == pytest.approx(length, rel=1e-5)   # the stick's own length
    bond = float(cr.BOND_FIT) + (1 - float(cr.BOND_FIT)) * (1 - math.exp(-0.02 / (float(cr.BINDER_SHARE) * 0.2)))
    assert sc.items.bond[0, j].item() == pytest.approx(bond, rel=1e-5)
    assert not sc.items.alive[0, heads[0]] and not sc.items.alive[0, fibre]
    assert sc.held[0, 0].tolist() == [j, -1]                                       # the joint stays in the left grip
    assert sc.held[1, 0].tolist() == [handles[1], heads[1]] and sc.items.head_mass[1, handles[1]] == 0
    assert torch.allclose(cr.ledger_mass(sc.items, sc.fires).sum(1), total0)
    assert r["work_j"][0, 0].item() == pytest.approx(0.25 * 300.0 * mp.PRESS_S) and r["time_s"][1, 0] == mp.PRESS_S
    assert not mp.check(sc.items, sc.held)


# ------------------------------------------------------------------------------------------- transforms (gates)
def test_clay_placed_in_a_hot_enough_fire_becomes_ceramic():
    sc = Scene(A=2, N=1)
    spot = (sc.mouth[0, 0, 0].item(), sc.mouth[0, 0, 1].item())
    sc.fire(0, {"charcoal": 10.0}, spot)                      # a charcoal bed: about 1384 K
    sc.fire(1, {"fat": 30.0}, spot)                           # a flaming bed: 1100 K, below the 1150 K onset
    clay = [sc.give(a, 0, 0, "clay", 1.0) for a in range(2)]
    pl = mp.place(sc.items, sc.fires, sc.held, torch.ones(2, 1, dtype=torch.bool), mouth_pos=sc.mouth, L=L)
    assert (pl["fire"] >= 0).all()
    for _ in range(4):                                        # a day of four bouts
        sc.heat()
    produce = m.TRANSFORMS[m.TRANSFORM_INDEX["clay_to_ceramic"]].outputs["ceramic"]
    assert sc.species_total("ceramic")[0].item() == pytest.approx(produce, rel=1e-4)
    assert sc.species_total("clay")[0].item() < 1e-6
    assert sc.species_total("ceramic")[1].item() == 0.0 and sc.kg(1, clay[1], "clay") == pytest.approx(1.0)
    assert max(sc.balance()) < 1e-6


def test_flesh_placed_in_a_fire_cooks():
    sc = Scene(A=2, N=1)
    spot = (sc.mouth[0, 0, 0].item(), sc.mouth[0, 0, 1].item())
    sc.fire(0, {"wood": 0.3}, spot)                           # a small fire
    meat = [sc.give(a, 0, 0, "meat", 1.0, temp=300.0) for a in range(2)]
    mp.place(sc.items, sc.fires, sc.held, sc.mask((0, 0), (1, 0)), mouth_pos=sc.mouth, L=L)  # arena 1: no fire there
    fl = sc.heat()
    ck, gain = mp.denatured(sc.items), mp.digestible_gain(sc.items)
    assert ck[0, meat[0]] and sc.kg(0, meat[0], "meat") > 0.5                    # cooked, mostly not charred
    assert gain[0, meat[0]].item() == pytest.approx(float(m.COOK_GAIN["meat"]))
    assert not ck[1, meat[1]] and gain[1, meat[1]].item() == 1.0
    assert fl["charred_kg"][0].sum() < 0.5
    assert max(sc.balance()) < 1e-6
    # a mouth biting the cooked meat gets cooked meat and the ledger books it
    bite = mp.remove_mass(sc.items, sc.held, torch.tensor([0, 1]), torch.tensor(meat), torch.tensor([0.1, 0.1]),
                          ledger=sc.ledger)
    assert bite["cooked"].tolist() == [True, False]
    assert torch.allclose(bite["species_kg"].sum(-1), torch.tensor([0.1, 0.1], dtype=torch.float64), rtol=1e-6)
    assert max(sc.balance()) < 1e-6


def test_a_big_fire_chars_flesh_away_to_the_soil_of_its_cell():
    sc = Scene(A=1, N=1)
    spot = (sc.mouth[0, 0, 0].item(), sc.mouth[0, 0, 1].item())
    sc.fire(0, {"wood": 3.0}, spot)
    sc.give(0, 0, 0, "meat", 1.0, temp=300.0)
    mp.place(sc.items, sc.fires, sc.held, sc.mask((0, 0)), mouth_pos=sc.mouth, L=L)
    fl = sc.heat(cells=128)
    assert sc.species_total("meat")[0].item() < 1e-4
    assert fl["charred_kg"][0, m.IDX["meat"]].item() == pytest.approx(1.0, rel=1e-4)
    cell = int(spot[0] // 16) * 128 + int(spot[1] // 16)
    soil_c = mp.mat.element_mass(fl["to_soil_kg"])[0, m.ELEMENTS.index("C")].item()
    assert fl["soil_c_kg"][0, cell].item() == pytest.approx(soil_c, rel=1e-9)
    assert fl["soil_c_kg"].sum().item() == pytest.approx(soil_c)
    assert max(sc.balance()) < 1e-6


def test_fire_bed_across_the_periodic_boundary_heats_its_items():
    sc = Scene(A=1, N=1)
    sc.fire(0, {"charcoal": 10.0}, (L - 0.1, 500.0))
    clay = sc.loose(0, "clay", 1.0, (0.2, 500.0))           # 0.3 m from the fire's centre, across the edge
    sc.heat(dt=86400.0)
    assert sc.species_total("ceramic")[0] > 0.8 and 0 <= sc.items.pos[0, clay, 0] < L
    assert max(sc.balance()) < 1e-6


def test_decay_rots_flesh_into_the_litter_of_its_cell():
    sc = Scene(A=1, N=1)
    meat = sc.loose(0, "meat", 1.0, (40.0, 40.0))
    out = mp.decay_step(sc.items, sc.held, dt_days=1.0, L=L, cells=128, ledger=sc.ledger)
    lost = 1.0 - sc.items.mass[0, meat].item()
    assert lost == pytest.approx(1 - math.exp(-float(cr.DECAY_PER_DAY["meat"])), rel=1e-4)
    cell = 2 * 128 + 2
    c_frac = m.ELEMENT_FRACTION[m.IDX["meat"]][m.ELEMENTS.index("C")]
    n_frac = m.ELEMENT_FRACTION[m.IDX["meat"]][m.ELEMENTS.index("N")]
    assert out["litter_c_kg"][0, cell].item() == pytest.approx(lost * c_frac, rel=1e-4)
    assert out["litter_n_kg"][0, cell].item() == pytest.approx(lost * n_frac, rel=1e-4)
    assert max(sc.balance()) < 1e-6


# ------------------------------------------------------------------------------------------- no named actions (gate)
#: words of purpose: no function name holds one as a word or inflection (split at underscores: "cooked" is out)
NAMED_ACTIONS = ("light", "ignite", "cook", "roast", "knap", "collect", "forage", "hunt", "attack", "kill", "eat",
                 "drink", "feed", "flee", "share", "give", "combine", "craft", "build", "chop", "dig", "butcher",
                 "gather", "drop", "smelt", "tool", "weapon", "food", "make")
NAMED_FUNCTIONS = ("make_fire", "heat_item", "feed_fire", "cook", "knap", "collect", "combine", "share", "drop")
V2_ACTION_FUNCS = ("make_fire", "knap", "collect", "cook", "feed_fire", "heat_item", "share", "combine", "drop",
                   "arm_power_w", "action_work_j", "strike_energy", "worn_insulation", "butcher_yield")


def test_no_named_purposeful_action_function_exists():
    src = Path(mp.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    names = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    for name in names:
        assert name not in NAMED_FUNCTIONS, name
        for word in name.lower().strip("_").split("_"):
            for bad in NAMED_ACTIONS:
                inflected = word.startswith(bad) and word[len(bad):] in ("", "s", "e", "es", "ed", "en", "er", "ing")
                assert not inflected, (name, bad)
    used = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
            and n.value.id == "cr"}
    assert not used & set(V2_ACTION_FUNCS), used & set(V2_ACTION_FUNCS)
    # the whole v3 package defines no fire-lighting or cooking function
    for path in (ROOT / "haishool" / "life9" / "v3").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert re.search(r"def\s+(make_fire|cook|knap|light_fire|start_fire)\b", text) is None, path.name
    # and the primitives are the body's abilities of PLANET-V3-SPEC 0
    for ability in ("grip", "release", "force", "rub", "press", "place"):
        assert ability in names


def test_no_forbidden_mechanisms_in_the_module():
    text = Path(mp.__file__).read_text(encoding="utf-8")
    for word in ("innate", "forage", "background_drink", "time_budget", "pedigree", "relatedness", "seed_floor",
                 "lifespan", "maturity", "reward", "outcome", "kleiber", "allometr", "make_fire", "ACTIONS"):
        assert not re.search(word, text, flags=0 if word.isupper() else re.IGNORECASE), word


def test_every_constant_has_provenance():
    for key, (tag, source) in mp.provenance().items():
        assert source and all(part in TAGS for part in tag.split("+")), key
    for name in ("TISSUE_DENSITY_KG_M3", "LIFT_SPEED_M_S", "KNAP_J_PER_KG", "MAX_FLAKE_SHARE", "WOUND_J_PER_KG",
                 "PRESS_S", "GROUND_K_W_MK", "CONTACT_SCAN", "SCAN_CHUNK", "MAX_HASH_CELLS", "RUB_SPEED_M_S",
                 "MOVING_FLASH", "PECLET_BRIDGE", "PYROLYSIS_K", "EMBER_KG", "TINDER_MIN_SHARE", "TINDER_SPECIES",
                 "PYROLYSING_SPECIES", "SURFACE_LAYER_M", "LIMB_SUSTAINED_SHARE", "HEAD_SHARE",
                 "WOUND_BODY_J_PER_KG"):
        assert name in mp.PROVENANCE, name
    for name in mp.USED_CRAFTING:
        assert f"crafting.{name}" in mp.provenance()
    for name, value in vars(mp).items():
        number = isinstance(value, (int, float)) and not isinstance(value, bool)
        if name.isupper() and not name.startswith("_") and number:
            assert name in mp.PROVENANCE or name in ("S", "GRIPS", "E", "WOOD", "FIBER", "NEVER"), name
    assert "peak" in mp.PROVENANCE["LIFT_SPEED_M_S"][1]                           # it states the power it assumes


# ------------------------------------------------------------------------------------- helpers for body.py and senses
def test_dead_bodies_let_go_in_a_bout_and_events_come_from_the_generator():
    from haishool.life9.v3 import brain3
    sc = Scene(A=1, N=2)
    stone = sc.give(0, 1, 0, "flint", 0.5)
    alive = torch.tensor([[True, False]])
    out = brain3.decode(torch.full((1, 2, brain3.OUT_DIM), -30.0))           # every output off
    r = mp.bout(sc.items, sc.fires, sc.held, out, torch.Generator().manual_seed(0), alive=alive, pos=sc.pos,
                mouth_pos=sc.mouth, body_mass_kg=sc.mass, peak_w=sc.peak, sustained_w=sc.sus, gravity=9.81,
                x_o2=0.21, air_k=288.0, L=L, dt_s=DAY)
    assert r["dropped_by_dead"][0, 1, 0] == stone and sc.items.holder[0, stone] == -1
    assert torch.allclose(sc.items.pos[0, stone, :2], sc.mouth[0, 1])
    assert not r["events"]["grip"].any() and (r["work_j"] == 0).all()
    assert r["force"]["hit_body"].eq(-1).all() and r["rub"]["partner_a"].eq(-1).all()
    # with every output on, the live body grips the stone the dead one dropped... if its mouth touches it
    sc.mouth[0, 0] = sc.mouth[0, 1]
    out = brain3.decode(torch.full((1, 2, brain3.OUT_DIM), 30.0))
    out["rub"] = torch.zeros(1, 2)
    out["force"] = torch.zeros(1, 2)
    out["place"] = torch.zeros(1, 2)
    out["release"] = torch.zeros(1, 2)
    r = mp.bout(sc.items, sc.fires, sc.held, out, torch.Generator().manual_seed(0), alive=alive, pos=sc.pos,
                mouth_pos=sc.mouth, body_mass_kg=sc.mass, peak_w=sc.peak, sustained_w=sc.sus, gravity=9.81,
                x_o2=0.21, air_k=288.0, L=L, dt_s=DAY)
    assert r["grip"]["item"][0, 0, 0] == stone and sc.held[0, 0, 0] == stone


def test_heat_step_reports_each_fires_burning_rate():
    sc = Scene(A=1, N=1)
    sc.fire(0, {"wood": 2.0}, (500.0, 500.0))
    fl = sc.heat(dt=3600.0)
    rate = fl["burn_kg_s"][0, 0].item()
    burnt = fl["burnt_kg"][0].sum().item() / 3600.0
    assert rate > 0 and burnt / 3 < rate < 3 * burnt


def test_radiant_flux_is_periodic():
    sc = Scene(A=1, N=2)
    f = sc.fire(0, {"wood": 2.0}, (L - 1.0, 300.0))
    hrr = torch.zeros(1, 8)
    hrr[0, f] = 10000.0
    pos = torch.tensor([[[1.0, 300.0], [1.0, 330.0]]])
    q = mp.radiant_flux(sc.fires, hrr, pos, L)
    assert q[0, 0].item() == pytest.approx(float(cr.RADIANT_FRACTION) * 1e4 / (4 * math.pi * 4.0), rel=1e-4)
    assert q[0, 1] == 0                                                          # beyond 20 m


def test_contacts_and_touch_read_physical_properties():
    sc = Scene(A=1, N=2)
    sc.pos[0, 1] = sc.mouth[0, 0] + torch.tensor([0.25, 0.0])
    blade = sc.at_mouth(0, 0, "flint", 0.3, temp=320.0)
    sc.items.sharp[0, blade] = 0.6
    f = sc.fire(0, {"wood": 1.0}, (sc.mouth[0, 0, 0].item(), sc.mouth[0, 0, 1].item() + 0.2))
    c = mp.contacts(sc.items, sc.fires, alive=sc.alive, pos=sc.pos, mouth_pos=sc.mouth, body_mass_kg=sc.mass, L=L)
    assert c["item"][0, 0] == blade and c["body"][0, 0] == 1 and c["fire"][0, 0] == f
    assert c["item"][0, 1] == -1 and c["fire"][0, 1] == -1
    t = mp.touch(sc.items, c["item"])
    assert t["present"][0].tolist() == [True, False]
    assert t["hardness"][0, 0].item() == pytest.approx(float(m.MOHS["flint"]))
    assert t["sharp"][0, 0].item() == pytest.approx(0.6) and t["temp_k"][0, 0].item() == pytest.approx(320.0)
    assert t["mass"][0, 0].item() == pytest.approx(0.3)
