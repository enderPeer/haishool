"""Physics checks of life9 v3's contact physics (PLANET-V3-SPEC section 8) against independent derivations: the
constant-power stroke by numerical integration, rod inertia by summing mass elements, the time and power budget of a
bout, rubbing with no partner, moving-source friction and its transient, the caps of the contact temperature, the
ember-to-tinder step, held items in a fire, the wound calibration against its citation, binding by a separate item
only, pieces broken off the ground and booked out of the patch, the contact scan's second pass and the item ledger of
bites."""
from __future__ import annotations

import math

import pytest
import torch

from haishool.life9.planet import crafting as cr
from haishool.life9.planet import materials as m
from haishool.life9.v3 import brain3
from haishool.life9.v3 import manipulate as mp
from haishool.life9.v3 import patch as pt

from test_manipulate import DAY, L, Scene, one_stroke_dt, stick_len


# ------------------------------------------------------------------------------------------- the stroke
def _integrate_stroke(P, m_eff, a, steps=200000):
    """Constant power from rest: dE/dt = P, v = sqrt(2E/m), dx/dt = v, until x = a (midpoint steps in time)."""
    t_guess = 1.5 * a / (3 * P * a / m_eff) ** (1 / 3)
    h = 2 * t_guess / steps
    x, t, e = 0.0, 0.0, 0.0
    while True:
        v_mid = math.sqrt(2 * (e + 0.5 * P * h) / m_eff)
        if x + v_mid * h >= a:
            frac = (a - x) / (v_mid * h)
            t += frac * h
            e += P * frac * h
            return math.sqrt(2 * e / m_eff), t
        x += v_mid * h
        t += h
        e += P * h


def test_stroke_matches_a_numerical_integration_of_constant_power():
    sc = Scene(A=1, N=3)
    sc.give(0, 1, 0, "basalt", 0.5)
    hafted = sc.give(0, 2, 0, "basalt", 1.5)
    sc.items.head_mass[0, hafted] = 0.5                     # a 0.5 kg head on a 1 kg handle 0.6 m long
    sc.items.handle_len_m[0, hafted] = 0.6
    sw = mp.swing(sc.items, sc.held, body_mass_kg=sc.mass, peak_w=sc.peak)
    a = 2 * mp.body_radius_m(torch.tensor(70.0)).item()
    m_limb = float(cr.LIMB_FRACTION) * 70.0
    for n in range(3):
        v, t = _integrate_stroke(1000.0, sw["m_eff"][0, n].item(), a)
        assert sw["v_hand"][0, n].item() == pytest.approx(v, rel=1e-3)
        assert sw["stroke_s"][0, n].item() == pytest.approx(t, rel=1e-3)
        assert sw["swing_j"][0, n].item() == pytest.approx(1000.0 * t, rel=2e-3)        # work = P t
    # the effective mass at the hand by summing the mass elements of limb, rod and head about the shoulder
    k = 20000
    r = (torch.arange(k, dtype=torch.float64) + 0.5) / k
    limb_i = (m_limb / k * (r * a) ** 2).sum().item() / a ** 2
    assert sw["m_eff"][0, 0].item() == pytest.approx(limb_i, rel=1e-4)                  # the limb is a rod: m / 3
    rod_r = a + r * 0.6
    rod_i = (1.0 / k * rod_r ** 2).sum().item() / a ** 2
    head_i = 0.5 * (a + 0.6) ** 2 / a ** 2
    assert sw["m_eff"][0, 2].item() == pytest.approx(limb_i + rod_i + head_i, rel=1e-4)
    lam_v = (a + 0.6) / a * sw["v_hand"][0, 2].item()
    assert sw["blow_j"][0, 2].item() == pytest.approx(0.5 * (rod_i + head_i) * sw["v_hand"][0, 2].item() ** 2,
                                                      rel=1e-4)
    assert sw["blow_j"][0, 2].item() > 0.5 * 0.5 * lam_v ** 2                           # head plus the handle
    assert sw["blow_j"][0, 2] > sw["blow_j"][0, 1]                                       # the handle levers the head
    assert (sw["blow_j"] <= sw["swing_j"] * (1 + 1e-6)).all()


def test_a_plain_stick_is_a_lever_and_a_lump_of_the_same_mass_is_not():
    sc = Scene(A=1, N=2)
    sc.give(0, 0, 0, "wood", 0.4, length=stick_len(0.4))
    sc.give(0, 1, 0, "wood", 0.4)
    sw = mp.swing(sc.items, sc.held, body_mass_kg=sc.mass, peak_w=sc.peak)
    a = sw["limb_m"][0, 0].item()
    lam = 1 + stick_len(0.4) / a
    i_rod = 0.4 * (lam * lam + lam + 1) / 3
    assert sw["m_eff"][0, 0].item() == pytest.approx(float(cr.LIMB_FRACTION) * 70.0 / 3 + i_rod, rel=1e-5)
    assert sw["blow_j"][0, 0] > 1.2 * sw["blow_j"][0, 1]


# ------------------------------------------------------------------------------------------- one bout, one budget
def test_rubbing_nothing_does_no_friction_work():
    sc = Scene(A=1, N=2, sustained_w=300.0)
    sc.give(0, 1, 0, "wood", 0.1)                                   # body 1 holds a stick but rubs the ground
    r = sc.rub()
    assert r["work_j"][0, 0] == 0 and r["heat_j"][0, 0] == 0 and r["partner_a"][0, 0] == -1
    assert r["partner_b"][0, 0] == -1 and r["contact_k"][0, 0] == 288.0
    # moving the free limb costs only its own oscillation, far below the rubbing power
    assert 0 < r["limb_j"][0, 0] < 1e-3 * 300.0 * DAY
    assert r["work_j"][0, 1].item() == pytest.approx(0.25 * 300.0 * r["time_s"][0, 1].item(), rel=1e-5)
    out = brain3.decode(torch.full((1, 2, brain3.OUT_DIM), -30.0))
    out["rub"] = torch.tensor([[1.0, 0.0]])
    b = mp.bout(sc.items, sc.fires, sc.held, out, torch.Generator().manual_seed(1), alive=sc.alive, pos=sc.pos,
                mouth_pos=sc.mouth, body_mass_kg=sc.mass, peak_w=sc.peak, sustained_w=sc.sus, gravity=9.81,
                x_o2=0.21, air_k=288.0, L=L, dt_s=DAY)
    assert b["work_j"][0, 0].item() == pytest.approx(b["rub"]["limb_j"][0, 0].item(), rel=1e-6)


def test_the_bout_is_shared_and_the_power_budget_closes():
    sc = Scene(A=1, N=3, sustained_w=200.0)
    for n in range(3):
        sc.give(0, n, 0, "basalt", 0.5)
        sc.give(0, n, 1, "wood", 0.3)
    out = brain3.decode(torch.zeros(1, 3, brain3.OUT_DIM))
    out["rub"] = torch.tensor([[0.8, 0.2, 0.0]])
    out["force"] = torch.tensor([[0.6, 0.2, 0.0]])
    out["press"] = torch.tensor([[1.0, 0.0, 0.0]])
    for k in ("grip_left", "grip_right", "place", "release"):
        out[k] = torch.zeros(1, 3)
    out["grip"] = torch.zeros(1, 3, 2)
    thrust = torch.tensor([[-0.5, 0.3, 0.0]])
    b = mp.bout(sc.items, sc.fires, sc.held, out, torch.Generator().manual_seed(0), alive=sc.alive, pos=sc.pos,
                mouth_pos=sc.mouth, body_mass_kg=sc.mass, peak_w=sc.peak, sustained_w=sc.sus, gravity=9.81,
                x_o2=0.21, air_k=288.0, L=L, dt_s=DAY, thrust=thrust)
    sh = b["shares"]
    total = sh["press"] + sh["rub"] + sh["force"] + sh["locomotion"]
    assert (total <= 1 + 1e-6).all() and total[0, 0].item() == pytest.approx(1.0)
    assert sh["press"][0, 0].item() == pytest.approx(mp.PRESS_S / DAY)
    k = (1 - mp.PRESS_S / DAY) / 1.9                                     # 0.8 + 0.6 + 0.5 asked for, scaled
    assert sh["rub"][0, 0].item() == pytest.approx(0.8 * k, rel=1e-5)
    assert b["thrust_share"][0, 0].item() == pytest.approx(-0.5 * k, rel=1e-5)
    assert sh["rub"][0, 1].item() == pytest.approx(0.2) and b["thrust_share"][0, 1].item() == pytest.approx(0.3)
    assert (b["time_s"] <= DAY * (1 - sh["locomotion"]) + 1e-3).all()
    # rubbing and striking never use more than their share of the sustained energy
    limb = mp.LIMB_SUSTAINED_SHARE * sc.sus
    assert (b["rub"]["work_j"] <= limb * sh["rub"] * DAY * (1 + 1e-6)).all()
    assert (b["force"]["work_j"] <= limb * sh["force"] * DAY * (1 + 1e-6)).all()
    assert (b["force"]["strokes"][0, :2] > 0).all() and b["force"]["strokes"][0, 2] == 0


def test_lift_and_strokes_use_peak_power_rubbing_and_pressing_the_sustained():
    sc = Scene(A=1, N=2, peak_w=1000.0, sustained_w=100.0)
    sc.peak[0, 1] = 2000.0
    sc.sus[0, 1] = 50.0
    sw = mp.swing(sc.items, sc.held, body_mass_kg=sc.mass, peak_w=sc.peak)
    assert sw["v_hand"][0, 1].item() == pytest.approx(sw["v_hand"][0, 0].item() * 2 ** (1 / 3), rel=1e-5)
    r = sc.strike(dt=3600.0)
    limb = mp.LIMB_SUSTAINED_SHARE * sc.sus
    rate = torch.minimum(1 / sw["cycle_s"], limb / sw["swing_j"])
    assert r["strokes"][0].tolist() == torch.floor(3600.0 * rate[0]).tolist()
    assert (rate == limb / sw["swing_j"]).all()                            # the aerobic power spaces the strokes


# ------------------------------------------------------------------------------------------- wounds
def test_a_blow_at_the_top_of_the_skull_fracture_range_does_grave_but_not_lethal_damage():
    skin = (float(m.MOHS["hide"]), float(m.TOUGHNESS["hide"]))
    hard = float(m.MOHS["basalt"])
    for blow, lo, hi in ((69.0, 0.2, 0.45), (14.0, 0.03, 0.12)):                  # on the head
        w = cr.strike_damage(torch.tensor(blow), torch.tensor(0.0), torch.tensor(hard), *skin).item()
        assert lo < 1 - math.exp(-w / (mp.WOUND_J_PER_KG * 70.0)) < hi
    # on the chest: 400 J of blunt impact is serious (Kroell et al. 1974), 69 J slight
    for blow, lo, hi in ((400.0, 0.2, 0.45), (69.0, 0.02, 0.1)):
        w = cr.strike_damage(torch.tensor(blow), torch.tensor(0.0), torch.tensor(hard), *skin).item()
        assert lo < 1 - math.exp(-w / (mp.WOUND_BODY_J_PER_KG * 70.0)) < hi
    # one stroke with a 0.5 kg stone delivering 69 J on a 70 kg body
    sc = Scene(A=1, N=2)
    sc.give(0, 0, 0, "basalt", 0.5)
    sc.pos[0, 1] = sc.mouth[0, 0] + torch.tensor([0.2, 0.0])
    sc.mouth[0, 1] = sc.pos[0, 1] + torch.tensor([0.0, 0.3])
    a = 2 * mp.body_radius_m(torch.tensor(70.0)).item()
    m_eff = float(cr.LIMB_FRACTION) * 70.0 / 3 + 0.5
    v = math.sqrt(2 * 69.0 / 0.5)
    sc.peak[0, 0] = v ** 3 * m_eff / (3 * a)
    dt, _ = one_stroke_dt(sc)
    seen = set()
    for seed in range(60):                                  # the stroke lands on the head about one time in eleven
        r = sc.strike(sc.level(1.0, (0, 0)), dt=dt, gen=torch.Generator().manual_seed(seed))
        assert r["blow_j"][0, 0].item() == pytest.approx(69.0, rel=1e-4) and r["hits"][0, 0] == 1
        head = r["wound_head_j"][0, 1].item() > 0
        seen.add(head)
        assert (0.2 < r["damage"][0, 1].item() < 0.45) if head else (0.02 < r["damage"][0, 1].item() < 0.1)
    assert seen == {True, False}


def test_strikes_last_while_the_target_stays_in_contact():
    sc = Scene(A=1, N=4)
    for s, t in ((0, 1), (2, 3)):
        sc.pos[0, t] = sc.mouth[0, s] + torch.tensor([0.2, 0.0])
        sc.mouth[0, t] = sc.pos[0, t] + torch.tensor([0.0, 0.3])
    vel = torch.zeros(1, 4, 2)
    vel[0, 3] = torch.tensor([0.0, 0.1])                                # body 3 walks past body 2 at 0.1 m/s
    r = sc.strike(sc.level(1.0, (0, 0), (0, 2)), dt=600.0, vel=vel)
    assert r["hits"][0, 0] == r["strokes"][0, 0] and r["strokes"][0, 0] > 100
    reach = mp.body_radius_m(torch.tensor(70.0)).item()
    t_c = 2 * reach / 0.1
    rate = r["strokes"][0, 2].item() / 600.0
    assert r["hits"][0, 2].item() == pytest.approx(math.floor(t_c * rate), abs=1)
    assert r["wound_j"][0, 1] > 20 * r["wound_j"][0, 3] > 0


def test_a_vanishing_intensity_strikes_and_rubs_nothing():
    sc = Scene(A=1, N=2)
    sc.pos[0, 1] = sc.mouth[0, 0] + torch.tensor([0.2, 0.0])
    sc.give(0, 0, 0, "wood", 0.1)
    sc.at_mouth(0, 0, "wood", 0.5)
    tiny = torch.full((1, 2), 9e-14)
    r = sc.strike(tiny)
    assert (r["hit_body"] == -1).all() and (r["strokes"] == 0).all() and (r["work_j"] == 0).all()
    q = sc.rub(tiny)
    assert (q["partner_a"] == -1).all() and (q["partner_b"] == -1).all() and (q["work_j"] == 0).all()


# ------------------------------------------------------------------------------------------- friction
def _wood_pair(limb_w, share=1.0, dt=DAY, stick_length=True, ground=False):
    sc = Scene(A=1, N=1, sustained_w=limb_w / mp.LIMB_SUSTAINED_SHARE)
    sc.give(0, 0, 0, "wood", 0.1, length=stick_len(0.1) if stick_length else 0.0)
    if not ground:
        sc.at_mouth(0, 0, "wood", 0.5)
    return sc, sc.rub(share, dt=dt)


def test_the_slid_over_partner_is_a_moving_source_and_ignition_takes_real_power():
    # the old stationary-spot estimate: T_c = T + Q / (4 a (k_A + k_B)) with a = 5 mm
    k = float(m.CONDUCTIVITY["wood"])
    sc, r = _wood_pair(20.0)
    stationary = 288.0 + 0.1 * 20.0 / (4 * float(cr.DRILL_RADIUS_M) * 2 * k)
    assert r["contact_k"][0, 0] < 288.0 + 0.4 * (stationary - 288.0)
    assert not r["ember"].any()                                     # 20 W lit wood before; now it does not
    _, r = _wood_pair(30.0)
    assert not r["ember"].any()
    _, r = _wood_pair(100.0)
    assert r["ember"].all()
    # the spot heats over time: seconds of rubbing are cooler than minutes
    t = [_wood_pair(100.0, share=s)[1]["contact_k"][0, 0].item() for s in (0.001, 0.01, 0.1)]
    assert t[0] < t[1] < t[2]


def test_rubbing_wet_or_dry_ground():
    k_dry, c_dry = mp.ground_thermal(torch.tensor([[0.0]]), 150.0)
    k_wet, c_wet = mp.ground_thermal(torch.tensor([[150.0]]), 150.0)
    assert k_dry.item() == pytest.approx(mp.GROUND_K_DRY_W_MK) and k_wet.item() == pytest.approx(mp.GROUND_K_WET_W_MK)
    t = []
    for k, c in ((k_dry, c_dry), (k_wet, c_wet)):
        sc = Scene(A=1, N=1, sustained_w=400.0)
        sc.give(0, 0, 0, "flint", 0.3)
        t.append(sc.rub(ground_k=k, ground_c=c)["contact_k"][0, 0].item())
    assert t[0] > t[1] > 288.0


def test_the_contact_temperature_is_capped_and_the_excess_chars():
    sc = Scene(A=1, N=3, sustained_w=4000.0)                          # a 1 kW limb
    sc.at_mouth(0, 0, "wood", 0.5)                                    # body 0: a bare limb on wood
    sc.give(0, 1, 0, "meat", 0.5)                                     # body 1: meat on meat
    other = sc.at_mouth(0, 1, "meat", 0.5)
    sc.give(0, 2, 0, "resin", 0.2)                                    # body 2: resin on wood melts, never lights
    sc.at_mouth(0, 2, "wood", 0.5)
    r = sc.rub(cells=128)
    assert r["skin_contact_k"][0, 0].item() == pytest.approx(float(cr.TISSUE_CHAR_K))
    assert r["skin_char_j"][0, 0] > 1e5 and r["skin_char_kg"][0, 0] > 0 and r["fire"][0, 0] == -1
    assert r["contact_k"][0, 1].item() == pytest.approx(float(cr.TISSUE_CHAR_K))
    assert r["char_kg"][0, 1] > 0.1 and r["to_soil_kg"][0, m.IDX["meat"]] > 0.1
    assert not mp.denatured(sc.items)[0, other] and sc.items.alive[0, other]
    assert r["soil_c_kg"].sum().item() == pytest.approx(
        m.element_mass(r["to_soil_kg"])[0, m.ELEMENTS.index("C")].item(), rel=1e-6)
    assert r["contact_k"][0, 2].item() == pytest.approx(float(m.MELT_K["resin"])) and not r["ember"][0, 2]
    assert (r["contact_k"] <= mp.PYROLYSIS_K + 1e-3).all()
    assert max(sc.balance()) < 1e-6
    # where the spot's heat went
    e = r["energy"]
    parts = e["items_j"] + e["ground_j"] + e["skin_j"] + e["char_j"]
    assert torch.allclose(parts, e["spot_j"], rtol=1e-6)
    assert torch.allclose(e["spot_j"] + e["dissipated_j"], e["work_j"], rtol=1e-6)


def test_an_ember_lights_tinder_held_in_the_other_grip_or_touching_the_spot():
    sc = Scene(A=1, N=2, sustained_w=800.0)
    sc.give(0, 0, 0, "wood", 0.1, length=stick_len(0.1))
    fibre = sc.give(0, 0, 1, "plant_fiber", 0.05)                       # the stick is rubbed on held fibre
    sc.give(0, 1, 0, "plant_fiber", 0.05)                               # fibre rubbed on a board
    board = sc.at_mouth(0, 1, "wood", 0.5)
    r = sc.rub(dt=3600.0)
    assert r["fire"][0, 0] >= 0 and r["tinder"][0, 0] == fibre and not sc.items.alive[0, fibre]
    assert r["fire"][0, 1] >= 0 and sc.items.alive[0, board]
    assert not mp.check(sc.items, sc.held) and max(sc.balance()) < 1e-6


# ------------------------------------------------------------------------------------------- heat at a fire
def test_meat_held_in_a_fire_heats_and_meat_held_5_m_away_does_not():
    sc = Scene(A=1, N=2)
    sc.mouth[0, 1] = sc.mouth[0, 0] + torch.tensor([5.0, 0.0])
    sc.fire(0, {"wood": 0.3}, (sc.mouth[0, 0, 0].item(), sc.mouth[0, 0, 1].item()))   # a small fire
    near = sc.give(0, 0, 0, "meat", 1.0)
    far = sc.give(0, 1, 0, "meat", 1.0)
    fl = sc.heat(mouth_pos=sc.mouth, body_pos=sc.pos)
    assert mp.denatured(sc.items)[0, near] and sc.kg(0, near, "meat") > 0.5      # cooked, mostly not charred
    assert not mp.denatured(sc.items)[0, far] and sc.items.temp_k[0, far] < 300.0
    assert sc.items.temp_k[0, far] > 288.0                                # the fire's radiation reaches it
    assert sc.held[0, 0, 0] == near and sc.held[0, 1, 0] == far and not mp.check(sc.items, sc.held)
    assert torch.allclose(sc.items.pos[0, near, :2], sc.mouth[0, 0])
    assert fl["mouth_fire_k"][0, 0].item() == pytest.approx(1000.0) and fl["mouth_fire_k"][0, 1].item() == 288.0
    assert fl["radiant_w_m2"][0, 0] > fl["radiant_w_m2"][0, 1] > 0
    assert max(sc.balance()) < 1e-6
    # without the bodies' mouths heat_step leaves held items in the air, as before
    sc2 = Scene(A=1, N=1)
    sc2.fire(0, {"wood": 2.0}, (sc2.mouth[0, 0, 0].item(), sc2.mouth[0, 0, 1].item()))
    meat = sc2.give(0, 0, 0, "meat", 0.5)
    sc2.heat()
    assert not mp.denatured(sc2.items)[0, meat]


def test_a_held_stick_in_a_fire_burns_into_its_bed_and_leaves_the_grip():
    sc = Scene(A=1, N=1)
    sc.fire(0, {"wood": 2.0}, (sc.mouth[0, 0, 0].item(), sc.mouth[0, 0, 1].item()))
    sc.give(0, 0, 0, "wood", 0.02, length=stick_len(0.02))
    sc.heat(mouth_pos=sc.mouth)
    assert sc.held[0, 0, 0] == -1 and not mp.check(sc.items, sc.held) and max(sc.balance()) < 1e-6


def test_radiation_warms_loose_items_near_a_fire_and_not_far():
    sc = Scene(A=1, N=1)
    sc.fire(0, {"wood": 3.0}, (500.0, 500.0))
    near = sc.loose(0, "flint", 0.5, (502.0, 500.0))
    far = sc.loose(0, "flint", 0.5, (530.0, 500.0))
    sc.heat(dt=3600.0)
    assert sc.items.temp_k[0, near] > 288.5 and sc.items.temp_k[0, far] == pytest.approx(288.0)


# ------------------------------------------------------------------------------------------- press
def test_binder_inside_the_pressed_items_does_not_bind_and_the_grips_set_the_orientation():
    sc = Scene(A=3, N=1)
    carcass = {"meat": 0.7, "fat": 0.1, "bone": 0.15, "hide": 0.05}
    length = stick_len(0.5)
    sc.give(0, 0, 0, "basalt", 0.5)                                  # arena 0: a stone and a carcass piece, no binder
    sc.give(0, 0, 1, carcass, 2.0)
    stick = sc.give(1, 0, 0, "wood", 0.5, length=length)             # arena 1: stick left, flint right, hide strip
    flint = sc.give(1, 0, 1, "flint", 0.2)
    sc.at_mouth(1, 0, "hide", 0.05)
    lump = sc.give(2, 0, 0, "flint", 0.2)                             # arena 2: flint left, stick right
    sc.give(2, 0, 1, "wood", 0.5, length=length)
    sc.at_mouth(2, 0, "hide", 0.05)
    r = mp.press(sc.items, sc.held, torch.ones(3, 1, dtype=torch.bool), alive=sc.alive, mouth_pos=sc.mouth,
                 sustained_w=sc.sus, L=L, body_mass_kg=sc.mass)
    assert r["item"][0, 0] == -1 and r["pressed"][0, 0] and r["work_j"][0, 0] > 0
    assert r["item"][1, 0] == stick and sc.items.head_mass[1, stick].item() == pytest.approx(0.2, rel=1e-5)
    assert sc.items.handle_len_m[1, stick].item() == pytest.approx(length, rel=1e-5)
    assert r["item"][2, 0] == lump and sc.items.head_mass[2, lump].item() == pytest.approx(0.5, rel=1e-5)
    k_flint = min(1.0, float(m.TOUGHNESS["flint"]) / float(cr.K_ROD))
    assert sc.items.handle_len_m[2, lump].item() == pytest.approx(0.5 * length * k_flint, rel=1e-5)
    assert not sc.items.alive[1, flint]
    assert max(sc.balance()) < 1e-6 and not mp.check(sc.items, sc.held)


# ------------------------------------------------------------------------------------------- the ground
def _patch(A=1, n=16):
    geom = pt.geometry_from_elevation(torch.zeros(A, n, n), 256.0, -1e4, 6.4e6)
    st = pt.init_state(geom)
    st.wood = torch.full_like(st.wood, 4.0)                         # kg C/m^2 of standing wood
    st.litter = torch.full_like(st.litter, 0.3)
    st = pt.reset_ledgers(st, geom)
    return geom, st


def test_strokes_on_the_ground_break_pieces_off_and_the_patch_loses_them():
    geom, st = _patch()
    deposits = torch.zeros(1, len(m.CRUST_SPECIES))
    deposits[0, m.IDX["basalt"]] = 2670.0
    ground = {"kg_m2": mp.ground_stock(st, geom, deposits), "cells": geom.n}
    c0 = pt.carbon_ledger(st, geom).clone()
    n0 = pt.nitrogen_ledger(st, geom).clone()
    sc = Scene(A=1, N=64, I=256)
    sc.pos = torch.rand(1, 64, 2, generator=torch.Generator().manual_seed(2)) * 256.0
    sc.mouth = sc.pos + torch.tensor([0.3, 0.0])
    sc.give(0, 0, 0, "basalt", 0.5)
    r = sc.strike(dt=600.0, ground=ground, gen=torch.Generator().manual_seed(3))
    got = r["broke"] >= 0
    assert got.sum() > 40 and int(sc.items.alive[0].sum()) == 1 + int(got.sum())   # nothing appears elsewhere
    sp = r["ground_species"][got]
    assert set(sp.tolist()) <= {m.IDX["wood"], m.IDX["plant_fiber"], m.IDX["basalt"]}
    wood = got & (r["ground_species"] == m.IDX["wood"])
    assert wood.any() and (sc.items.handle_len_m[0, r["broke"][wood]] > 0).all()        # sticks are long
    assert (sc.items.handle_len_m[0, r["broke"][got & ~wood]] == 0).all()
    assert torch.allclose(sc.items.pos[0, r["broke"][got], :2], torch.remainder(sc.mouth[got], L))
    assert r["ground_kg"].sum().item() == pytest.approx(r["broke_kg"].sum().item(), rel=1e-6)
    assert max(sc.balance()) < 1e-6
    crust = mp.take_ground(st, geom, r["ground_kg"])
    assert crust[0, m.IDX["basalt"]].item() == pytest.approx(r["ground_kg"][0, :, m.IDX["basalt"]].sum().item())
    cf = {s: m.species_elements(s)["C"] for s in ("wood", "plant_fiber")}
    taken_c = sum(r["ground_kg"][0, :, m.IDX[s]].sum().item() * cf[s] for s in cf)
    assert (st.c_taken.sum() - 0).item() == pytest.approx(taken_c, rel=1e-4)
    assert pt.carbon_ledger(st, geom).item() == pytest.approx(c0.item(), abs=1e-6 * taken_c + 1e-9)
    assert pt.nitrogen_ledger(st, geom).item() == pytest.approx(n0.item(), abs=1e-9)
    # the same seed gives the same pieces
    sc2 = Scene(A=1, N=64, I=256)
    sc2.pos, sc2.mouth = sc.pos, sc.mouth
    sc2.give(0, 0, 0, "basalt", 0.5)
    r2 = sc2.strike(dt=600.0, ground=ground, gen=torch.Generator().manual_seed(3))
    assert torch.equal(r["broke_kg"], r2["broke_kg"]) and torch.equal(r["ground_species"], r2["ground_species"])


def test_bare_ground_without_stock_gives_nothing():
    geom, st = _patch()
    st.wood = torch.zeros_like(st.wood)
    st.litter = torch.zeros_like(st.litter)
    ground = {"kg_m2": mp.ground_stock(st, geom), "cells": geom.n}
    sc = Scene(A=1, N=2)
    sc.mouth = torch.tensor([[[10.0, 10.0], [100.0, 100.0]]])
    r = sc.strike(ground=ground, gen=torch.Generator().manual_seed(0))
    assert (r["broke"] == -1).all() and r["ground_kg"].sum() == 0


# ------------------------------------------------------------------------------------------- numerics and ledgers
def test_contact_second_pass_finds_the_nearest_beyond_the_scan_budget():
    A, N, M = 1, 2, 300
    q = torch.tensor([[[10.0, 10.0], [500.0, 500.0]]])
    t = torch.full((A, M, 2), 10.4)
    t[0, M - 1] = torch.tensor([10.01, 10.0])                   # the nearest has the last slot
    idx, gap, over = mp.contact(q, torch.ones(A, N, dtype=torch.bool), torch.full((A, N), 0.5), t,
                                torch.ones(A, M, dtype=torch.bool), torch.zeros(A, M), L, scan=64)
    assert over[0, 0] and not over[0, 1]
    assert idx[0, 0] == M - 1 and gap[0, 0].item() == pytest.approx(0.01, abs=1e-5) and idx[0, 1] == -1


def test_bites_book_exactly_what_the_pool_lost():
    sc = Scene(A=1, N=1)
    i = sc.loose(0, {"meat": 0.7, "fat": 0.3}, 1.2345678, (50.0, 50.0))
    j = sc.loose(0, "meat", 0.00015, (60.0, 60.0))
    for _ in range(50):
        mp.remove_mass(sc.items, sc.held, torch.tensor([0, 0, 0]), torch.tensor([i, i, j]),
                       torch.tensor([1.1e-7, 3.3e-3, 1e-5]), ledger=sc.ledger)
    err = mp.ledger_error(sc.ledger, sc.items, sc.fires)
    assert abs(err["mass_kg"].item()) < 1e-12 and err["elements_kg"].abs().max().item() < 1e-12
    assert not sc.items.alive[0, j]
