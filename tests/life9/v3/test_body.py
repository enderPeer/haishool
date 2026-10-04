"""Physics tests of v3 bodies (PLANET-V3-SPEC sections 3 and 9): heat balance, thermogenesis, locomotion, ingestion,
bites, wear and repair, the death thresholds, and the module's provenance and forbidden-mechanism checks."""
from __future__ import annotations

import ast
import math
import re
from pathlib import Path

import pytest
import torch

from haishool.life9.planet import climate as cl
from haishool.life9.planet import constants as K
from haishool.life9.planet import items as itm
from haishool.life9.planet import materials as mt
from haishool.life9.v3 import body as B
from haishool.life9.v3 import brain3 as b3
from haishool.life9.v3 import patch as pt

ROOT = Path(__file__).resolve().parents[3]
TAGS = {"chain", "derived", "reference", "new_rule"}
L = 256.0
BASE = dict(size_kg=0.02, fur_m=1e-3, skin_perm=1e-11, thermo_gain=1e-3, setpoint_k=310.0, enz_plant=0.02,
            enz_meat=0.02, repair=0.5, muscle=100.0, muscle_frac=0.3, offspring_share=0.5, eye=1e-4, ear=1e-4,
            voice=1e-4, fat_store=0.3, bladder=0.01, kidney=0.005, air_l_kg=0.2,
            o2_carrier=0.0)


def make(n=1, *, A=1, N=None, T0=300.0, seed=0, pos=None, heading=None, **genes):
    """n bodies per arena with the given genes (BASE otherwise), at their size, a founder's fat, normal water."""
    g = torch.Generator().manual_seed(seed)
    N = n if N is None else N
    b = B.found(A, N, n, L, g, in_dim=4, hidden=8, t_k=T0)
    vals = {**BASE, **genes}
    for k, v in vals.items():
        b.genome[k][:, :n] = torch.as_tensor(v, dtype=torch.float32)
    b.genome["k"][:, :n] = 8
    size = b.genome["size_kg"][:, :n]
    b.mass_kg[:, :n] = size
    b.frame_kg[:, :n] = size
    b.reserve_j[:, :n] = B.FOUNDER_FAT_PER_LEAN * size * B.FAT_J_KG
    b.water_kg[:, :n] = B.LEAN_WATER * size
    b.body_k[:, :n] = T0
    if pos is not None:
        b.pos[:, :n] = torch.as_tensor(pos, dtype=torch.float32)
    if heading is not None:
        b.heading[:, :n] = torch.as_tensor(heading, dtype=torch.float32)
    B.reset_ledgers(b)
    return b


def flat_geom(A=1, n=16, sea=None):
    z = torch.zeros(A, n, n)
    if sea is not None:
        z[:, sea[0], sea[1]] = -10.0
    return pt.geometry_from_elevation(z, L, -5.0, 5.674e6)


def _tag_ok(tag):
    return bool(tag) and all(part in TAGS for part in tag.split("+"))


# ------------------------------------------------------------------------------------------------ the contract
def test_provenance_is_complete_and_tagged():
    consts = [n for n in vars(B) if re.fullmatch(r"[A-Z][A-Z0-9_]*", n) and n not in ("PROVENANCE",)]
    skip = {"K", "R_", "D_", "N_", "C_", "RN", "DN", "G_PLANT", "G_PROTEIN", "G_LIPID", "G_COOKED", "G_INERT", "Q_KG",
            "Q_J", "Q_C", "Q_N"}
    missing = [c for c in consts if c not in skip and c not in B.PROVENANCE]
    assert not missing, missing
    for key, (tag, note) in B.PROVENANCE.items():
        assert note and _tag_ok(tag), key
    assert "GUT_CLASS_INDEX" in B.PROVENANCE and "GUT_Q_INDEX" in B.PROVENANCE


def test_no_forbidden_mechanism_in_the_source():
    src = (ROOT / "haishool/life9/v3/body.py").read_text(encoding="utf-8").lower()
    for word in ("innate", "forage", "actions", "background_drink", "time_budget", "pedigree", "relatedness",
                 "seed_floor", "lifespan", "maturity", "reward", "outcome", "kleiber", "allometr", "temperature0",
                 "litter_mass", "gestation", "cooldown"):
        assert word not in src, word


def test_body_reuses_physics_not_version_2_life_history():
    tree = ast.parse((ROOT / "haishool/life9/v3/body.py").read_text(encoding="utf-8"))
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            mods |= {f"{node.module}.{a.name}" for a in node.names}
    assert not any("creatures" in m or m.endswith("planet.brain") or "planet.brain." in m for m in mods), mods
    for name in {**b3.BODY_SPECS3, **B.EXTRA_SPECS}:
        assert name in BASE, name          # every body gene is read by the body physics
    assert set(B.GENE_SPECS) == set(b3.GENE_SPECS3) | set(B.EXTRA_SPECS)


def test_body_plan_chemistry_matches_the_carcass_species():
    e = sum(x * float(mt.FOOD_J_KG[s]) for s, x in B.BODY_PLAN.items())
    assert B.E_LEAN == pytest.approx(e, rel=1e-12)
    assert 6.5e6 < B.E_LEAN < 7.5e6                                  # the spec's 7 MJ/kg tissue cost
    c = sum(x * mt.species_elements(s)["C"] for s, x in B.BODY_PLAN.items())
    assert B.C_LEAN == pytest.approx(c, rel=1e-9)
    assert B.LEAN_WATER == pytest.approx(0.6678, abs=1e-4)
    # tripalmitin: 72.5 O2 -> 51 CO2 + 49 H2O, about 440 kJ per mol O2
    assert B.O2_MOL_PER_KG_FAT * B.M_FAT == pytest.approx(72.5)
    assert B.CO2_MOL_PER_KG_FAT * B.M_FAT == pytest.approx(51.0)
    assert B.H2O_MOL_PER_KG_FAT * B.M_FAT == pytest.approx(49.0)
    assert 4.3e5 < B.OXY_J_PER_MOL_O2 < 4.6e5
    assert B.FAT_C_PER_J * B.FAT_J_KG / B.M_C == pytest.approx(B.CO2_MOL_PER_KG_FAT, rel=1e-6)


def test_arrhenius_and_tissue_closure():
    assert float(B.arrhenius(torch.tensor(B.T_REF_K))) == pytest.approx(1.0)
    ratio = float(B.arrhenius(torch.tensor(300.0)) / B.arrhenius(torch.tensor(310.0)))
    assert ratio == pytest.approx(math.exp(-0.65 / B.K_B_EV * (1 / 300 - 1 / 310)), rel=1e-5)
    b = make(2, muscle_frac=[0.3, 0.9], enz_plant=[0.02, 0.5])
    tis = B.tissues(b)
    assert float(tis["brain"][0, 0]) == pytest.approx(8 * B.BRAIN_KG_PER_UNIT)
    # the second genome asks for more than the body: every tissue is scaled into the lean mass, nothing is left over
    total = sum(tis[k] for k in ("muscle", "enz_plant", "enz_meat", "eye", "ear", "voice", "brain", "fur", "kidney",
                                 "lung", "carrier", "rest"))
    assert torch.allclose(total, b.mass_kg, rtol=1e-5)
    assert float(tis["asked"][0, 1]) > 1 and float(tis["rest"][0, 1]) == 0.0
    assert float(tis["muscle"][0, 1]) < 0.9 * float(b.mass_kg[0, 1])


# ------------------------------------------------------------------------------------------------ heat balance
def test_a_20_g_ectotherm_cools_to_air_temperature():
    b = make(1, size_kg=0.02, thermo_gain=b3.THERMO_BOUNDS[0], T0=305.0)
    env = B.constant_env(t_air_k=288.0, rel_humidity=1.0, sw_w_m2=0.0, wind_m_s=1.0)
    e = B._env(b, env)
    hx = B.heat_exchange(b, e)
    tau_h = float(hx["C"] / hx["G"]) / 3600
    assert 0.05 < tau_h < 3                                         # a small body follows the air within hours
    B.metabolism(b, env)
    t = float(b.body_k)
    assert t < 290.0
    assert abs(t - 288.0) < 2.5                                     # at air temperature (night sky: slightly below)
    assert abs(t - float(hx["T_e"])) < 0.3                          # its own metabolic heat raises it by < 0.3 K
    # with sunshine the operative temperature, and the body, rise above the air
    sun = B.constant_env(t_air_k=288.0, rel_humidity=1.0, sw_w_m2=600.0, wind_m_s=1.0)
    B.metabolism(b, sun)
    assert float(b.body_k) > 290.0


def _hand_conductance(b, t_air, wind, p=K.P_STANDARD):
    """G (W/K) by hand: equal-volume sphere, spherical fur shell, Ranz-Marshall forced convection combined with
    Churchill's free convection (cube law) at the fur's diameter, linear radiation, scaled by the spheroid's area
    factor."""
    m = float(b.mass_kg + b.reserve_j / B.FAT_J_KG + b.gut[..., 0].sum(-1) + b.water_kg - B.LEAN_WATER * b.mass_kg)
    r1 = (3 * m / B.RHO_TISSUE / (4 * math.pi)) ** (1 / 3)
    r2 = r1 + float(b.genome["fur_m"])
    rho = p * 0.028965 / (K.R_GAS * t_air)
    nu = 1.846e-5 * (t_air / 300) ** 0.7 / rho
    k = 0.0263 * (t_air / 300) ** 0.8
    alpha = k / (rho * 1007.0)
    forced = 2 + 0.6 * math.sqrt(wind * 2 * r2 / nu) * 0.707 ** (1 / 3)
    ra = K.G_STANDARD / t_air * abs(float(b.body_k) - t_air) * (2 * r2) ** 3 / (nu * alpha)
    free = 2 + 0.589 * ra ** 0.25 / (1 + (0.469 / 0.707) ** (9 / 16)) ** (4 / 9)
    nusselt = 2 + ((forced - 2) ** 3 + (free - 2) ** 3) ** (1 / 3)
    h = nusselt * k / (2 * r2) + 4 * 0.97 * K.SIGMA * t_air ** 3
    s = B.SHAPE_FACTOR
    r_fur = (1 / r1 - 1 / r2) / (4 * math.pi * 0.04 * s)
    return 1 / (r_fur + 1 / (h * 4 * math.pi * r2 ** 2 * s))


def test_an_endotherm_holds_its_setpoint_at_the_insulation_cost():
    b = make(1, size_kg=0.2, fur_m=5e-3, skin_perm=1e-12, thermo_gain=50.0, setpoint_k=310.0, repair=1.0,
             T0=310.0)
    env = B.constant_env(t_air_k=273.0, rel_humidity=1.0, sw_w_m2=0.0, wind_m_s=1.0)
    for _ in range(3):
        e = B._env(b, env)
        hx = B.heat_exchange(b, e)
        g_hand = _hand_conductance(b, 273.0, 1.0)
        assert float(hx["G"]) == pytest.approx(g_hand, rel=1e-4)
        o2_0 = float(b.ledger["o2_mol"])
        m = B.metabolism(b, env)
        assert abs(float(b.body_k) - 310.0) < 1.0                    # the setpoint is held
        # fuel spent: the respiratory share of its heat leaves as latent heat, the rest is A dT / R_insulation
        o2 = float(b.ledger["o2_mol"]) - o2_0
        tb = float(b.body_k)
        resp_w = o2 * (float(cl.e_sat_pa(torch.tensor(tb))) - float(e["vapour_pa"])) / (
            float(e["p_o2_pa"]) * B.RESP_EXTRACTION) * B.M_H2O * float(B.latent_heat(torch.tensor(tb)))
        dry = float(m["paid_j"]) - resp_w
        loss = g_hand * (tb - float(hx["T_e"])) * env.dt_s
        assert dry == pytest.approx(loss, rel=0.03)
        assert float(m["thermo_j"]) > 10 * float(m["maintenance_j"])   # endothermy is thermogenesis, not upkeep
    # thicker fur, cheaper warmth (in wind)
    costs = []
    for fur in (2e-3, 1e-2):
        c = make(1, size_kg=0.2, fur_m=fur, skin_perm=1e-12, thermo_gain=50.0, setpoint_k=310.0, repair=1.0,
                 T0=310.0)
        costs.append(float(B.metabolism(c, env)["thermo_j"]))
    assert costs[1] < 0.7 * costs[0]


def test_thermogenesis_is_capped_by_the_aerobic_capacity():
    b = make(2, size_kg=0.2, fur_m=2e-3, skin_perm=1e-12, thermo_gain=50.0, setpoint_k=310.0, repair=0.0,
             T0=310.0, muscle=[200.0, 2.0])
    env = B.constant_env(t_air_k=273.0, rel_humidity=1.0, wind_m_s=1.0)
    cap = B.SUSTAINED_SHARE * B.aerobic(torch.tensor(21200.0)) * B.muscle_peak_w(b) / B.MUSCLE_EFF
    m = B.metabolism(b, env)
    assert float(m["thermo_j"][0, 1]) <= float(cap[0, 1]) * env.dt_s * (1 + 1e-5)
    assert abs(float(b.body_k[0, 0]) - 310.0) < 1.0                    # strong muscles hold the setpoint
    assert float(b.body_k[0, 1]) < float(b.body_k[0, 0]) - 5           # a weak body cannot stay warm
    # the cap is the muscles' aerobic power at their temperature: a cooled body can shiver less
    cold = float(B.muscle_peak_w(b)[0, 1]) / float(B.muscle_ref_w(b)[0, 1])
    assert cold == pytest.approx(float(B.arrhenius(b.body_k[0, 1])), rel=1e-5) and cold < 0.7


def test_evaporation_follows_skin_permeability_and_vapour_deficit():
    b = make(2, size_kg=0.02, skin_perm=[1e-12, 1e-6], T0=300.0)
    dry_env = B.constant_env(t_air_k=300.0, rel_humidity=0.2, wind_m_s=2.0)
    m = B.metabolism(b, dry_env)
    assert float(m["evap_kg"][0, 1]) > 100 * float(m["evap_kg"][0, 0])
    wet = make(2, size_kg=0.02, skin_perm=[1e-12, 1e-6], T0=300.0)
    m2 = B.metabolism(wet, B.constant_env(t_air_k=300.0, rel_humidity=0.95, wind_m_s=2.0))
    assert float(m2["evap_kg"][0, 1]) < 0.2 * float(m["evap_kg"][0, 1])
    # evaporation cools: the wet-skinned body ends colder
    assert float(b.body_k[0, 1]) < float(b.body_k[0, 0]) - 1.0


# ------------------------------------------------------------------------------------------------ locomotion
def test_locomotion_speed_solves_the_power_balance():
    b = make(3, size_kg=0.02, T0=300.0, heading=0.0)
    b.damage[0, 2] = 0.5
    env = B.constant_env(t_air_k=293.0)
    p_before = B.muscle_peak_w(b)
    sh = B.shape(b)
    x0 = b.pos.clone()
    r = B.move(b, torch.zeros(1, 3), torch.tensor([[0.2, 0.0, 0.2]]), env)
    p = 0.2 * B.SUSTAINED_SHARE * B.aerobic(torch.tensor(21200.0)) * p_before[0, 0]
    v = float(r["dist_m"][0, 0]) / env.dt_s
    air = B.air_props(torch.tensor(293.0), torch.tensor(K.P_STANDARD))
    m0 = float(sh["m"][0, 0])
    lhs = B.C_M * m0 * K.G_STANDARD * v + 0.5 * float(air["rho"]) * B.C_DRAG * float(sh["frontal"][0, 0]) * v ** 3         + B.FEDAK_INTERNAL[0] * m0 * v ** B.FEDAK_INTERNAL[1]                     # the limbs' internal work
    assert lhs == pytest.approx(float(p), rel=1e-4)
    assert float(r["metabolic_j"][0, 0]) == pytest.approx(float(p) * env.dt_s / B.MUSCLE_EFF, rel=1e-5)
    assert float(r["dist_m"][0, 1]) == 0.0 and float(r["metabolic_j"][0, 1]) == 0.0
    assert float(r["dist_m"][0, 2]) < float(r["dist_m"][0, 0])       # damage reduces muscle power
    want = x0[0, 0, 0] + r["dist_m"][0, 0]
    assert float(pt.wrap(b.pos[0, 0, 0] - want, L)) == pytest.approx(0.0, abs=0.05)
    assert float(b.speed[0, 0]) == pytest.approx(v, rel=1e-5)
    # less O2, less sustained power: the speed solves the same balance at the lower power, slower
    c = make(1, size_kg=0.02, heading=0.0)
    r2 = B.move(c, torch.zeros(1, 1), torch.full((1, 1), 0.2), B.constant_env(t_air_k=293.0, p_o2_pa=1000.0))
    a_ratio = float(B.aerobic(torch.tensor(1000.0)) / B.aerobic(torch.tensor(21200.0)))
    v2 = float(r2["dist_m"][0, 0]) / env.dt_s
    lhs2 = B.C_M * m0 * K.G_STANDARD * v2 + 0.5 * float(air["rho"]) * B.C_DRAG * float(sh["frontal"][0, 0]) * v2 ** 3         + B.FEDAK_INTERNAL[0] * m0 * v2 ** B.FEDAK_INTERNAL[1]
    assert lhs2 == pytest.approx(a_ratio * float(p), rel=1e-4) and v2 < v
    # the top speed uses the peak power
    assert float(B.top_speed(make(1, size_kg=0.02), env)) > 3 * v


def test_swimming_and_climbing_cost_their_physics():
    geom_sea = flat_geom(sea=(slice(None), slice(None)))
    b = make(1, size_kg=0.02, heading=0.0, pos=[[100.0, 100.0]])
    land = make(1, size_kg=0.02, heading=0.0, pos=[[100.0, 100.0]])
    env = B.constant_env(t_air_k=293.0)
    rho = float(B.density(b))
    rs = B.move(b, torch.zeros(1, 1), torch.full((1, 1), 1.0), env, geom=geom_sea)
    rl = B.move(land, torch.zeros(1, 1), torch.full((1, 1), 1.0), env, geom=flat_geom())
    assert rho < B.RHO_SEA and float(rs["submerged"]) == pytest.approx(rho / B.RHO_SEA, rel=1e-5)   # it floats
    p = 1.0 * B.SUSTAINED_SHARE * float(B.aerobic(torch.tensor(21200.0))) * float(B.muscle_peak_w(land))
    assert float(B.sustained_power_w(land, env)) == pytest.approx(p, rel=1e-6)    # its lung is no limit here
    v = float(rs["dist_m"]) / env.dt_s
    k3 = 0.5 * B.RHO_WATER * B.C_DRAG * float(B.shape(b)["frontal"])
    assert k3 * v ** 3 == pytest.approx(p, rel=1e-4)
    assert float(rs["dist_m"]) < float(rl["dist_m"])                  # at full power, water drag is slower
    assert float(rs["heat_j"]) < float(rs["metabolic_j"])            # drag work leaves into the water
    # climbing a ramp costs M g dz, paid from the same power: the body gets less far than on the flat
    n = 16
    slope = 2.0 / (L / n)                                             # 2 m per 16 m cell
    ramp = (torch.arange(n, dtype=torch.float32) * 2.0)[:, None].expand(n, n)[None].clone()
    geom = pt.geometry_from_elevation(ramp, L, -100.0, 5.674e6)
    up = make(1, size_kg=0.02, heading=0.0, pos=[[40.0, 100.0]])
    level = make(1, size_kg=0.02, heading=0.0, pos=[[40.0, 100.0]])
    power = 1e-3 * B.SUSTAINED_SHARE * float(B.aerobic(torch.tensor(21200.0))) * float(B.muscle_peak_w(up))
    z0 = float(pt.bilinear(geom, geom.elev, up.pos))
    r = B.move(up, torch.zeros(1, 1), torch.full((1, 1), 1e-3), env, geom=geom)
    rf = B.move(level, torch.zeros(1, 1), torch.full((1, 1), 1e-3), env, geom=flat_geom())
    z1 = float(pt.bilinear(geom, geom.elev, up.pos))
    m = float(B.shape(up)["m"])
    assert z1 > z0
    assert float(r["climb_j"]) == pytest.approx(m * K.G_STANDARD * (z1 - z0), rel=1e-3)
    assert float(r["work_j"]) == pytest.approx(power * env.dt_s, rel=1e-3)          # the climb is within the power
    d0 = float(rf["dist_m"])
    per_m = power * env.dt_s / d0                                                    # level work per metre
    assert float(r["dist_m"]) == pytest.approx(d0 * per_m / (per_m + m * K.G_STANDARD * slope), rel=1e-2)
    assert float(r["heat_j"]) == pytest.approx(float(r["metabolic_j"]) - float(r["climb_j"])
                                               - float(r["drag_j"]), rel=1e-4)       # potential energy leaves


# ------------------------------------------------------------------------------------------------ ingestion
def _mouth_to(b, i, target_xy):
    """Place body i so that its mouth is at target_xy (heading 0)."""
    sh = B.shape(b)
    b.heading[0, i] = 0.0
    b.pos[0, i, 0] = target_xy[0] - float(sh["a"][0, i])
    b.pos[0, i, 1] = target_xy[1]


def test_a_bite_transfers_flesh_and_wounds():
    geom = flat_geom()
    b = make(4, size_kg=[1.0, 1.0, 1.0, 0.002], enz_plant=0.05, enz_meat=0.05)
    sh = B.shape(b)
    # body 0 bites body 1 (same size, cut through its hide); body 2 swallows body 3 whole (it fits the gape)
    b.pos[0, 1] = torch.tensor([100.0, 100.0])
    _mouth_to(b, 0, (100.0 - 0.5 * float(sh["r1"][0, 1]), 100.0))
    b.pos[0, 3] = torch.tensor([180.0, 180.0])
    _mouth_to(b, 2, (180.0, 180.0))
    for i in (1, 3):
        b.heading[0, i] = math.pi / 2
    s0 = B.stocks(b)
    lean1, water1, res1 = float(b.mass_kg[0, 1]), float(b.water_kg[0, 1]), float(b.reserve_j[0, 1])
    body1 = lean1 + res1 / B.FAT_J_KG
    w0 = float(b.water_kg[0, 0])
    env = B.constant_env()
    out = B.ingest(b, torch.tensor([[1.0, 0.0, 1.0, 0.0]]), env, geom=geom)
    assert out["contact"].tolist() == [[3, 0, 3, 0]] and out["target"].tolist() == [[1, -1, 3, -1]]
    phi = float(out["bitten_kg"][0, 0]) / body1
    assert 0 < phi < 1
    assert float(b.mass_kg[0, 1]) == pytest.approx(lean1 * (1 - phi), rel=1e-5)
    assert float(b.damage[0, 1]) == pytest.approx(phi / B.WOUND_LETHAL_SHARE, rel=1e-4)
    gut = b.gut[0, 0].double()
    assert float(gut[B.G_PROTEIN, B.Q_KG]) == pytest.approx(phi * lean1 * B.LEAN_PROTEIN, rel=1e-4)
    assert float(gut[B.G_LIPID, B.Q_J]) == pytest.approx(phi * (lean1 * B.LEAN_LIPID * B.FAT_J_KG + res1),
                                                         rel=1e-4)
    assert float(b.water_kg[0, 0]) - w0 == pytest.approx(phi * water1, rel=1e-4)
    # the gut limits the bite: the biter's dry space is now full
    cap = float(B.gut_capacity_kg(b)[0, 0])
    assert float(b.gut[0, 0, :, B.Q_KG].sum()) == pytest.approx(cap, rel=1e-4)
    # the small prey is swallowed whole
    assert float(b.mass_kg[0, 3]) == 0.0 and float(out["wound"][0, 3]) >= 1 / B.WOUND_LETHAL_SHARE - 1e-6
    # energy, carbon, nitrogen and water only moved between the bodies
    s1 = B.stocks(b)
    for k in ("energy", "carbon", "nitrogen", "water"):
        assert float(s1[k]) == pytest.approx(float(s0[k]), rel=1e-6), k
    # biting cost muscular work
    assert float(out["bite_work_j"][0, 0]) > 0
    # a prey that runs leaves the gape at once: far less is cut
    c = make(2, size_kg=[1.0, 1.0], enz_plant=0.05, enz_meat=0.05)
    c.pos[0, 1] = torch.tensor([100.0, 100.0])
    _mouth_to(c, 0, (100.0 - 0.5 * float(B.shape(c)["r1"][0, 1]), 100.0))
    c.vel[0, 1] = torch.tensor([0.0, 2.0])
    o2 = B.ingest(c, torch.tensor([[1.0, 0.0]]), env, geom=geom)
    assert 0 < float(o2["bitten_kg"][0, 0]) < 0.01 * float(out["bitten_kg"][0, 0])
    # a weak jaw against a tough hide cuts less per bite
    q_strong = float(B.bite_force_n(make(1, size_kg=1.0, muscle_frac=0.5)))
    q_weak = float(B.bite_force_n(make(1, size_kg=1.0, muscle_frac=0.01)))
    assert q_weak < 0.1 * q_strong


def test_items_are_bitten_by_toughness_or_swallowed_whole():
    geom = flat_geom()
    b = make(5, size_kg=1.0, enz_plant=0.05, enz_meat=0.05)
    pool = itm.new_item_pool(1, 16)
    spots = [(30.0, 30.0), (90.0, 30.0), (150.0, 30.0), (210.0, 30.0), (30.0, 150.0)]
    species = ["meat", "flint", "flint", "meat", "fat"]
    masses = [0.5, 0.3, 0.001, 0.5, 0.2]
    for i, (xy, s, m) in enumerate(zip(spots, species, masses)):
        _mouth_to(b, i, xy)
        comp = torch.nn.functional.one_hot(torch.tensor(mt.IDX[s]), mt.S).float()
        slot = itm.spawn(pool, torch.tensor([0]), comp[None], torch.tensor([m]),
                         torch.tensor([[xy[0], xy[1], 0.0]]), torch.tensor([300.0]), torch.tensor([-1]))
        assert int(slot) == i
    pool.peak_k[0, 3] = 360.0                                         # the second meat item was cooked
    s_items = itm.species_mass(pool).clone()
    out = B.ingest(b, torch.ones(1, 5), B.constant_env(), geom=geom, pool=pool)
    assert out["contact"][0].tolist() == [2, 2, 2, 2, 2]
    eaten = out["item_kg"][0]
    dry_meat = 1 - B.SPECIES_WATER[mt.IDX["meat"]]
    cap = float(B.gut_capacity_kg(b)[0, 0])
    assert float(eaten[0]) == pytest.approx(cap / float(dry_meat), rel=1e-4)      # the gut fills with meat
    assert float(pool.mass[0, 0]) == pytest.approx(0.5 - float(eaten[0]), rel=1e-4)
    assert float(eaten[1]) == 0.0 and float(pool.mass[0, 1]) == pytest.approx(0.3)  # flint: too hard, too big
    assert float(eaten[2]) == pytest.approx(0.001) and not bool(pool.alive[0, 2])    # a pebble swallowed whole
    assert float(b.gut[0, 2, B.G_INERT, B.Q_KG]) == pytest.approx(0.001, rel=1e-5)
    assert float(b.gut[0, 3, B.G_COOKED, B.Q_KG]) > 0 and float(b.gut[0, 3, B.G_PROTEIN, B.Q_KG]) == 0.0
    assert float(b.gut[0, 4, B.G_LIPID, B.Q_J]) == pytest.approx(float(eaten[4]) * B.FAT_J_KG, rel=1e-4)
    # the items' species mass left the pool and is booked as eaten
    gone = s_items - itm.species_mass(pool)
    assert torch.allclose(gone, b.ledger["items_eaten_kg"], rtol=1e-5, atol=1e-9)
    assert float(b.ledger["i_food"]) == pytest.approx(0.001, rel=1e-5)
    # the meat's water went into body water
    assert float(b.water_kg[0, 0]) == pytest.approx(B.LEAN_WATER + float(eaten[0]) * 0.735, rel=1e-4)


def test_grazing_drinking_and_sea_water():
    geom = flat_geom(sea=(slice(12, 16), slice(None)))
    ps = pt.init_state(geom)
    ps.plant = torch.where(geom.sea, 0.0, 0.2)
    ps.pond[0, 2, 2] = 40.0
    ps = pt.reset_ledgers(ps, geom)
    b = make(4, size_kg=0.05, enz_plant=0.05)
    env = B.constant_env()
    _mouth_to(b, 0, (5 * 16 + 8.0, 5 * 16 + 8.0))                     # plants, soil water at half the bucket
    _mouth_to(b, 1, (2 * 16 + 8.0, 2 * 16 + 8.0))                     # plants and a pond
    _mouth_to(b, 2, (13 * 16 + 8.0, 5 * 16 + 8.0))                    # the sea
    _mouth_to(b, 3, (7 * 16 + 8.0, 5 * 16 + 8.0))                     # plants, after walking 20 m this bout
    b.vel[0, 3] = torch.tensor([20.0 / env.dt_s, 0.0])
    r1 = B.shape(b)["r1"][0].double()
    cap = B.gut_capacity_kg(b)[0].double()
    w0 = b.water_kg.clone()
    snap = B.exchange_snapshot(b, ps)
    out = B.ingest(b, torch.tensor([[1.0, 1.0, 0.01, 1.0]]), env, geom=geom, pstate=ps)   # body 2 sips the sea
    assert out["contact"][0].tolist() == [1, 1, 1, 1]
    cf = pt.PatchRules().bio.plant_carbon_fraction
    dens = 0.2 / cf                                                   # kg dry per m^2
    # a still mouth crops the disc it reaches; one that swept the cell crops the strip it passed, up to the gut
    disc = dens * math.pi * float(r1[0]) ** 2
    assert disc < float(cap[0])
    assert float(out["plant_kg"][0, 0]) == pytest.approx(disc, rel=1e-4)
    strip = dens * (2 * float(r1[3]) * 20.0 + math.pi * float(r1[3]) ** 2)
    assert float(out["plant_kg"][0, 3]) == pytest.approx(min(strip, float(cap[3])), rel=1e-4)
    assert float(out["plant_kg"][0, 3]) > 2 * float(out["plant_kg"][0, 0])
    assert float(b.ledger["c_food"]) == pytest.approx(cf * float(out["plant_kg"].sum()), rel=1e-9)
    assert float(out["drunk_kg"][0, 0]) == 0.0 and float(out["drunk_kg"][0, 1]) > 0
    assert float(out["sea_kg"][0, 2]) > 0 and float(out["plant_kg"][0, 2]) == 0.0
    assert float(b.salt_kg[0, 2]) == pytest.approx(B.SEA_SALINITY * float(out["sea_kg"][0, 2]), rel=1e-5)
    # plant water: 3 kg per kg dry, all of it in the body; the patch's soil (field + residual) lost exactly that
    gain0 = float(b.water_kg[0, 0].double() - w0[0, 0].double())
    assert gain0 == pytest.approx(float(out["plant_kg"][0, 0]) * B.PLANT_WATER_PER_DRY, rel=1e-5)
    errs = B.exchange_errors(b, ps, geom, snap)
    for k in ("water_in", "carbon_in", "nitrogen_in", "sea"):
        assert float(errs[k].abs()) <= 1e-9 * float(errs[f"{k}_scale"]) + 1e-15, k
    assert abs(float(pt.water_ledger(ps, geom))) < 1e-3
    assert abs(float(pt.carbon_ledger(ps, geom))) < 1e-6
    # sea water gains no water: the salt leaves in urine at the kidney's concentration (a sip: the kidney's
    # filtration can carry it; a gut-full of sea water in a bout could not be flushed)
    before = float(b.water_kg[0, 2])
    c_max = float(B.urine_salt_max(b)[0, 2])
    assert c_max == pytest.approx(B.URINE_SALT_PER_KIDNEY * 0.005, rel=1e-3)
    m = B.metabolism(b, B.constant_env(rel_humidity=1.0), geom=geom, pstate=ps)
    assert float(b.salt_kg[0, 2]) == 0.0
    assert float(m["urine_kg"][0, 2]) >= float(out["sea_kg"][0, 2]) * B.SEA_SALINITY / c_max * 0.999
    assert float(b.water_kg[0, 2]) <= before


def test_digestion_follows_enzyme_investment():
    b = make(2, size_kg=0.1, enz_plant=[0.001, 0.1], enz_meat=0.01)
    b.gut[0, :, B.G_PLANT] = torch.tensor([0.01, 0.01 * B.PLANT_J_KG, 0.0047, 0.0002])
    B.reset_ledgers(b)
    r0 = b.reserve_j.clone()
    fae = B.digest(b, B.DAY_S / 4)
    gained = b.reserve_j - r0
    share = 1 - math.exp(-B.DAY_S / 4 * float(B.arrhenius(torch.tensor(300.0))) / B.RETENTION_S)
    passed = 0.01 * share                                            # the gut passes at the rate of T_b (300 K)
    assert float(fae["kg"][0, 0]) > float(fae["kg"][0, 1])
    eff1 = 0.1 / (0.1 + B.K_ENZ_PLANT)
    assert float(fae["kg"][0, 1]) == pytest.approx(passed * (1 - eff1), rel=1e-4)
    assert float(gained[0, 1]) > 20 * float(gained[0, 0])
    assert float(b.n_kg[0, 1]) > 0                                    # absorbed nitrogen waits in the free pool
    # what was absorbed is in the store and the pool; what passed unabsorbed is in the faeces
    e_in = 0.01 * B.PLANT_J_KG * share
    assert float(fae["j"][0, 1] + fae["absorbed_j"][0, 1]) == pytest.approx(e_in, rel=1e-5)


# ------------------------------------------------------------------------------------------------ growth, wear
def test_growth_from_the_store_above_its_heritable_level_needs_nitrogen_and_burns_nothing():
    b = make(4, size_kg=0.05, repair=0.0, T0=300.0, fat_store=0.1)
    for i in range(4):
        b.mass_kg[0, i] = 0.03
        b.frame_kg[0, i] = 0.03
        b.water_kg[0, i] = 0.03 * B.LEAN_WATER
    b.reserve_j[0, :] = 0.5 * 0.03 * B.FAT_J_KG                       # 0.5 kg fat per kg lean: above the old cap
    for i in (0, 2, 3):
        b.n_kg[0, i] = 2e-5                                           # amino nitrogen for about 0.55 g of lean
    b.genome["size_kg"][0, 2] = 0.03                                  # already at its size
    b.genome["fat_store"][0, 3] = 2.0                                 # its store comes before growth
    B.reset_ledgers(b)
    r0 = b.reserve_j.double().clone()
    m = B.metabolism(b, B.constant_env(t_air_k=300.0, rel_humidity=1.0))
    dm = float(m["growth_kg"][0, 0])
    assert dm == pytest.approx(2e-5 / B.N_LEAN, rel=1e-4) and float(b.mass_kg[0, 0]) > 0.03
    assert float(m["growth_kg"][0, 1:].abs().sum()) == 0.0            # no N / no room / below its own store level
    # its heat (the conversion surplus and the synthesis overhead) is the bout's, in the heat balance
    heat = dm * (B.BUILD_FAT_J_KG + B.N_LEAN * B.UREA_J_PER_KG_N - B.E_LEAN + B.SYNTH_OVERHEAD * B.E_LEAN)
    assert float(m["growth_heat_j"][0, 0]) == pytest.approx(heat, rel=1e-4)
    # nothing is burnt to keep a store under a cap: a body that does not grow only pays the bout's costs
    for i in (1, 2, 3):
        assert float(r0[0, i] - b.reserve_j[0, i].double()) == pytest.approx(float(m["paid_j"][0, i]), abs=2.0)
        assert float(B.fat_kg(b)[0, i]) > 0.45 * float(b.mass_kg[0, i])
    errs = B.ledger_errors(b)
    for k in ("energy", "carbon", "water", "nitrogen", "inert"):
        assert float(errs[k].abs() / errs[f"{k}_scale"]) < 1e-6, k
    # growth made bone mineral from the diet's ash (booked), the store level is the gene's
    assert float(b.ledger["i_built"]) == pytest.approx(dm * B.LEAN_MINERAL, rel=1e-4)
    assert float(B.store_level_j(b)[0, 3]) == pytest.approx(2.0 * float(b.mass_kg[0, 3]) * B.FAT_J_KG, rel=1e-6)


def test_oxidative_wear_and_repair():
    b = make(3, size_kg=0.05, repair=[0.0, 1.0, 1.0], T0=305.0, thermo_gain=20.0, setpoint_k=305.0)
    b.damage[0, 2] = 0.3                                              # a wound
    env = B.constant_env(t_air_k=280.0, rel_humidity=1.0)
    m = B.metabolism(b, env)
    assert float(b.damage[0, 0]) > 0                                  # unrepaired wear grows with the O2 burnt
    o2_body = float(m["metabolic_w"][0, 0]) * env.dt_s / B.OXY_J_PER_MOL_O2
    protein = B.LEAN_PROTEIN * float(b.mass_kg[0, 0])
    assert float(m["wear"][0, 0]) == pytest.approx(B.WEAR_PROTEIN_KG_PER_MOL_O2 * o2_body
                                                   / (B.OXIDISED_LETHAL_SHARE * protein), rel=1e-4)
    assert float(b.wear[0, 0]) == pytest.approx(float(b.damage[0, 0]), rel=1e-6)   # all of it is wear
    assert float(m["wear"][0, 1]) == 0.0 and float(b.damage[0, 1]) == 0.0
    assert float(b.damage[0, 2]) < 0.3                                # repair heals the wound
    assert float(m["repair_j"][0, 1]) == pytest.approx(float(m["spent_j"][0, 1]) / 2, rel=1e-5)
    # the healing is paid at the cost of re-synthesising the harmed protein; the rest of the budget prevents wear
    healed = 0.3 - float(b.damage[0, 2]) + float(m["wear"][0, 2])
    assert float(m["heal"][0, 2]) == pytest.approx(healed, rel=1e-4)
    used = float(m["heal"][0, 2]) * float(b.mass_kg[0, 2]) * B.HEAL_J_PER_KG
    assert used <= float(m["repair_j"][0, 2]) * (1 + 1e-5)
    # damage weakens the muscles
    p0 = float(B.muscle_peak_w(b)[0, 1])
    b.damage[0, 1] = 0.5
    assert float(B.muscle_peak_w(b)[0, 1]) == pytest.approx(0.5 * p0, rel=1e-6)


# ------------------------------------------------------------------------------------------------ deaths by physics
def test_dehydration_freezing_and_no_fuel_kill_at_the_thresholds():
    b = make(10, size_kg=0.05, T0=305.0)
    nw = B.LEAN_WATER * 0.05
    b.water_kg[0, 0], b.water_kg[0, 1] = 0.59 * nw, 0.61 * nw
    b.body_k[0, 2], b.body_k[0, 3] = 270.9, 271.1
    b.body_k[0, 4], b.body_k[0, 5] = 318.1, 317.9
    b.frame_kg[0, 6], b.mass_kg[0, 6] = 0.1, 0.059
    b.frame_kg[0, 7], b.mass_kg[0, 7] = 0.1, 0.061
    b.damage[0, 8], b.damage[0, 9] = 1.0, 0.99
    cause = B.death_cause(b)[0].tolist()
    C = B.CAUSE
    assert cause == [C["dehydration"], 0, C["freezing"], 0, C["denaturation"], 0, C["no_fuel"], 0, C["damage"], 0]


def test_physics_brings_bodies_to_the_thresholds():
    env_cold = B.constant_env(t_air_k=240.0, rel_humidity=1.0, wind_m_s=3.0)
    b = make(2, size_kg=0.02, thermo_gain=b3.THERMO_BOUNDS[0], fur_m=[1e-5, 1e-5], T0=290.0)
    B.metabolism(b, env_cold)
    assert bool((B.death_cause(b)[0] == B.CAUSE["freezing"]).all())          # an ectotherm freezes
    # a free-water skin in hot dry wind dries out; a waxy one does not
    env_dry = B.constant_env(t_air_k=305.0, rel_humidity=0.1, wind_m_s=5.0)
    d = make(2, size_kg=0.01, skin_perm=[1e-5, 1e-12], fur_m=1e-5, T0=305.0)
    for _ in range(4):
        B.metabolism(d, env_dry)
    causes = B.death_cause(d)[0].tolist()
    assert causes[0] == B.CAUSE["dehydration"] and causes[1] == 0
    # running without food burns the store, then the tissue, until no fuel is left (warm air: warm muscles)
    s = make(1, size_kg=0.02, T0=303.0, repair=1.0)
    env = B.constant_env(t_air_k=303.0, rel_humidity=1.0)
    killed = False
    for _ in range(200):
        loco = B.move(s, torch.zeros(1, 1), torch.full((1, 1), 0.6), env)
        B.metabolism(s, env, loco=loco)
        if int(B.death_cause(s)[0, 0]) == B.CAUSE["no_fuel"]:
            killed = True
            break
        if int(B.death_cause(s)[0, 0]) > 0:
            break
    assert killed
    assert float(s.reserve_j[0, 0]) < 1.0
    errs = B.ledger_errors(s)
    for k in ("energy", "carbon", "water", "nitrogen", "inert"):
        assert float(errs[k].abs() / errs[f"{k}_scale"]) < 1e-6, k


# ------------------------------------------------------------------------------------------------ views
def test_views_for_the_senses_and_the_hands():
    b = make(4, size_kg=[0.002, 0.02, 0.2, 2.0], N=6)
    v = B.sense_view(b, loudness=torch.full((1, 6), 0.5))
    assert torch.allclose(v["reserve_j"][0, :4] / v["reserve_cap_j"][0, :4], torch.full((4,), 0.5), atol=1e-5)
    assert torch.allclose(v["water_kg"][0, :4] / v["water_norm_kg"][0, :4], torch.ones(4), atol=1e-5)
    assert torch.allclose(v["mass_kg"][0, :4], B.shape(b)["m"][0, :4])
    assert torch.equal(v["reach_m"], B.mouth_reach_m(b)) and bool((v["call_w"][0, :4] > 0).all())
    assert (v["mass_kg"][0, 4:] == 0).all()
    s3 = pytest.importorskip("haishool.life9.v3.senses3")
    if hasattr(s3, "Bodies") and hasattr(s3.Bodies, "from_mapping"):
        sb = s3.Bodies.from_mapping(v)
        assert torch.equal(sb.mouth_pos, v["mouth_pos"]) and torch.equal(sb.reach_m, v["reach_m"])
    env = B.constant_env()
    m0 = B.manipulation_view(b, env)
    m1 = B.manipulation_view(b, env, thrust=torch.full((1, 6), 0.75))
    assert torch.allclose(m1["power_w"], 0.25 * m0["power_w"])
    peak = B.muscle_peak_w(b)
    assert torch.allclose(m0["power_w"], B.SUSTAINED_SHARE * B.aerobic(torch.tensor(21200.0)) * peak)
