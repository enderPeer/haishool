"""Tests of breathing and buoyancy of v3 bodies: the air-volume gene ``air_l_kg`` (the owner's decision 3 of 4 Oct
2026: lungs, air sacs or a swim bladder) and the O2 carrier ``o2_carrier``, with the review of 4 Oct 2026. The air
gene spans no air to a body half air and founders draw it over its bounds; its lung tissue costs maintenance and is
the body's exchange organ, so a large lungless body suffocates in air while a small one breathes through its skin;
the air decides buoyancy against the water where the body is (sea water is denser than ponds), and a dense body
treads water while it moves and has the power for it; the store (the air from the alveolar pO2 down to a share of it,
and the carrier unloading between the same pressures) feeds a breath-hold, harmless until it is spent, after which
the O2 debt is anoxia (no fixed time to drowning); free convection cools a still body in water; and the books close
(the air itself carries no mass to them)."""
from __future__ import annotations

import math

import pytest
import torch

from haishool.life9.planet import constants as K
from haishool.life9.v3 import body as B
from haishool.life9.v3 import brain3 as b3
from haishool.life9.v3 import patch as pt

L = 256.0
BASE = dict(size_kg=0.02, fur_m=1e-3, skin_perm=1e-12, thermo_gain=1e-3, setpoint_k=310.0, enz_plant=0.02,
            enz_meat=0.02, repair=0.5, muscle=100.0, muscle_frac=0.3, offspring_share=0.5, eye=1e-4, ear=1e-4,
            voice=1e-4, fat_store=0.3, bladder=0.01, kidney=0.005, air_l_kg=0.2,
            o2_carrier=0.0)


def make(n=1, *, A=1, N=None, T0=300.0, seed=0, fat=B.FOUNDER_FAT_PER_LEAN, **genes):
    """n bodies per arena with the given genes (BASE otherwise), at their size, ``fat`` kg per kg lean, normal water."""
    g = torch.Generator().manual_seed(seed)
    N = n if N is None else N
    b = B.found(A, N, n, L, g, in_dim=4, hidden=8, t_k=T0)
    for k, v in {**BASE, **genes}.items():
        b.genome[k][:, :n] = torch.as_tensor(v, dtype=torch.float32)
    b.genome["k"][:, :n] = 8
    size = b.genome["size_kg"][:, :n]
    b.mass_kg[:, :n] = size
    b.frame_kg[:, :n] = size
    b.reserve_j[:, :n] = fat * size * B.FAT_J_KG
    b.water_kg[:, :n] = B.LEAN_WATER * size
    b.body_k[:, :n] = T0
    B.reset_ledgers(b)
    return b


def sea_geom(A=1, n=16):
    return pt.geometry_from_elevation(torch.zeros(A, n, n), L, 5.0, 5.674e6)          # every cell is sea


def pond_geom(A=1, n=16, depth_kg_m2=2000.0):
    """Dry land with a 2 m deep pond everywhere (fresh water)."""
    geom = pt.geometry_from_elevation(torch.full((A, n, n), 10.0), L, 0.0, 5.674e6)
    state = pt.init_state(geom, pt.PatchRules(patch_m=L, cells=n))
    state.pond = torch.full_like(state.pond, depth_kg_m2)
    return geom, state


def closed(b, keys=("energy", "carbon", "water", "nitrogen", "inert", "salt", "oxygen", "co2"), tol=1e-6):
    errs = B.ledger_errors(b)
    for k in keys:
        rel = (errs[k].abs() / errs[f"{k}_scale"].clamp_min(1e-30)).max()
        assert float(rel) < tol, (k, float(rel))


def volume_per_kg_lean(fat_per_lean=B.FOUNDER_FAT_PER_LEAN):
    """(mass, volume) per kg lean of a body without air at normal water and an empty gut."""
    return 1.0 + fat_per_lean, 1.0 / B.RHO_TISSUE + fat_per_lean / B.RHO_FAT


# ------------------------------------------------------------------------------------------------ the genes
def test_the_air_gene_spans_no_air_to_a_body_half_air_and_founders_draw_it_over_its_bounds():
    s = B.EXTRA_SPECS["air_l_kg"]
    assert "air_l_kg" in B.GENE_SPECS and "air_l_kg" not in b3.GENE_SPECS3
    assert s.kind == "lin" and s.lo == 0.0                          # zero itself is a value: no air store at all
    assert s.hi >= 1e3 / B.RHO_TISSUE * 0.95                        # up to about the lean tissue's own volume
    for measured in (0.0535, 0.05, 0.07, 0.2):                      # mammal lungs, swim bladders, a bird's air sacs
        assert s.lo < measured < s.hi
    # review of 4 Oct 2026: 'random among founders over a physically broad range including zero' is the gene's bounds
    assert s.founder == "uniform" and (s.founder_lo, s.founder_hi) == (s.lo, s.hi) == B.AIR_FOUNDER_L_KG
    assert s.by_mut and s.unit == pytest.approx(1e3 * (1 / B.RHO_WATER - 1 / B.RHO_TISSUE))
    c = B.EXTRA_SPECS["o2_carrier"]
    assert c.kind == "lin" and c.lo == 0.0 and (c.founder_lo, c.founder_hi) == (c.lo, c.hi)
    assert c.hi == pytest.approx(B.LEAN_PROTEIN * B.HUFNER_ML_PER_G * 1e3)   # all the lean's protein as carrier
    for measured in (15.0, 40.0, 90.0):                             # land mammals, diving mammals
        assert c.lo < measured < c.hi
    for key in ("AIR_BOUNDS_L_KG", "breath_hold", "breathing", "tread", "CARRIER_BOUNDS_ML_KG", "O2_HOLD_FLOOR_SHARE",
                "ANOXIA_TOLERANCE_S", "LUNG_O2_MOL_S_KG", "SKIN_O2_MOL_S_M2_PA", "RHO_SEA"):
        assert B.PROVENANCE[key][0], key


def test_founders_draw_the_air_and_the_carrier_broadly_and_keep_their_older_draws():
    gen = torch.Generator().manual_seed(11)
    b = B.found(4, 500, 500, L, gen, in_dim=4, hidden=8, t_k=300.0, brain=False)
    for name, (lo, hi) in (("air_l_kg", B.AIR_FOUNDER_L_KG), ("o2_carrier", B.CARRIER_FOUNDER_ML_KG)):
        a = b.genome[name][b.alive].double()
        assert float(a.min()) >= lo and float(a.max()) <= hi
        assert float(a.min()) < lo + 0.01 * (hi - lo) and float(a.max()) > hi - 0.01 * (hi - lo)
        counts = torch.histc(a, bins=4, min=lo, max=hi)                        # uniform: no value is favoured
        assert float((counts / (a.numel() / 4) - 1).abs().max()) < 0.12, name
    # few founders are denser than fresh water: buoyancy is drawn, not chosen
    sinks = float((B.density(b)[b.alive] > B.RHO_WATER).double().mean())
    assert 0.0 < sinks < 0.08
    # the draws made before the late genes existed are unchanged (FOUNDER_LATE_GENES): genome, positions, headings,
    # then the air, then the carrier
    g = torch.Generator().manual_seed(3)
    old = b3.random_genome3(g, (2, 5), 4, hidden=8, k_range=(2, 8))
    old.update(b3.random_body_genes(g, (2, 5), "cpu", {k: v for k, v in B.EXTRA_SPECS.items()
                                                       if k not in B.FOUNDER_LATE_GENES}))
    pos = torch.rand(2, 5, 2, generator=g) * L
    heading = torch.rand(2, 5, generator=g) * (2 * math.pi)
    late = b3.random_body_genes(g, (2, 5), "cpu", {k: B.EXTRA_SPECS[k] for k in B.FOUNDER_LATE_GENES})
    nb = B.found(2, 6, 5, L, torch.Generator().manual_seed(3), in_dim=4, hidden=8, k_range=(2, 8))
    for name, t in {**old, **late}.items():
        assert torch.equal(nb.genome[name][:, :5], t), name
    assert torch.equal(nb.pos[:, :5], pos) and torch.equal(nb.heading[:, :5], heading)
    # an older body state without the genes gets them absent (no air, no carrier); a protocol resume refuses it
    st = nb.state_dict()
    del st["genome"]["air_l_kg"], st["genome"]["o2_carrier"]
    back = B.Bodies.from_state(st)
    assert float(back.genome["air_l_kg"].abs().max()) == 0.0 and float(back.genome["o2_carrier"].abs().max()) == 0.0
    with pytest.raises(ValueError, match="other rules"):
        B.Bodies.from_state(st, strict=True)


def test_the_air_gene_is_inherited_with_variation_inside_its_bounds():
    rows = B.found(1, 1, 1, L, torch.Generator().manual_seed(0), in_dim=4, hidden=8)
    parent = b3.rows(rows.genome, torch.tensor([0]), torch.tensor([0]))
    n = 4000
    for start in (0.0, 0.1, 1.0):
        par = {k: v.expand(n, *v.shape[1:]).clone() for k, v in parent.items()}
        par["air_l_kg"][:] = start
        par["mut"][:] = 0.05
        child = b3.mutate3(torch.Generator().manual_seed(7), par, specs=B.GENE_SPECS)
        c = child["air_l_kg"].double()
        assert float(c.min()) >= 0.0 and float(c.max()) <= 1.0
        assert float(c.std()) > 0                                              # children vary
        if start == 0.1:                                                       # a step of mut x AIR_UNIT_L_KG
            sd = float((c - start).std() / child["mut"].double().mean())
            assert sd == pytest.approx(B.AIR_UNIT_L_KG, rel=0.05)
        if start == 0.0:
            assert float((c > 0).double().mean()) > 0.99                       # reflected off zero, never below


# ------------------------------------------------------------------------------------------------ costs
def test_the_air_and_the_carrier_cost_their_tissue_and_its_maintenance():
    b = make(3, air_l_kg=[0.0, 0.2, 0.0], o2_carrier=[0.0, 0.0, 30.0])
    tis = B.tissues(b)
    lean = float(b.mass_kg[0, 1])
    lung = float(tis["lung"][0, 1])
    assert float(tis["lung"][0, 0]) == 0.0
    assert lung == pytest.approx(0.2 * B.AIR_TISSUE_KG_PER_L * lean, rel=1e-6)
    carrier = float(tis["carrier"][0, 2])
    assert carrier == pytest.approx(30.0 / B.HUFNER_ML_PER_G * 1e-3 * lean, rel=1e-6)     # 22 g protein per kg
    assert float(tis["rest"][0, 0] - tis["rest"][0, 1]) == pytest.approx(lung, rel=1e-3)     # they displace tissue
    assert float(tis["rest"][0, 0] - tis["rest"][0, 2]) == pytest.approx(carrier, rel=1e-3)
    total = sum(tis[k] for k in ("muscle", "enz_plant", "enz_meat", "eye", "ear", "voice", "brain", "fur", "kidney",
                                 "lung", "carrier", "rest"))
    assert torch.allclose(total, b.mass_kg, rtol=1e-5)
    maint = B.maintenance_ref_w(b, tis)
    assert float(maint[0, 1] - maint[0, 0]) == pytest.approx(lung * (B.C_LUNG_W_KG - B.C_REST_W_KG), rel=2e-3)
    assert float(maint[0, 2] - maint[0, 0]) == pytest.approx(carrier * (B.C_CARRIER_W_KG - B.C_REST_W_KG), rel=2e-3)
    assert B.C_LUNG_W_KG > B.C_CARRIER_W_KG > B.C_REST_W_KG                    # real costs, not relabellings
    # the lung is charged per wet kg at the human lung's 1.64 W over its 1.2 kg (review: not a blood-free rate)
    assert B.C_LUNG_W_KG == pytest.approx(1.64 / 1.2, rel=0.02)
    assert float(B.air_volume_m3(b, tis)[0, 1]) == pytest.approx(0.2e-3 * lean, rel=1e-6)
    # a genome asking for more tissue than the body has gets a smaller lung, and so less air
    over = make(1, air_l_kg=1.0, muscle_frac=0.95)
    t2 = B.tissues(over)
    assert float(t2["asked"]) > 1
    scale = float(t2["lung"]) / (B.AIR_TISSUE_KG_PER_L * float(over.mass_kg))
    assert scale < 1.0
    assert float(B.air_volume_m3(over, t2)) == pytest.approx(1e-3 * scale * float(over.mass_kg), rel=1e-5)


# ------------------------------------------------------------------------------------------------ breathing
def test_the_lung_is_the_exchange_organ_and_small_bodies_breathe_through_their_skin():
    """Review H3: O2 uptake needs an exchange organ. A 1 kg body without a lung takes up only what its skin passes,
    less than its resting costs: it has no sustained power and runs an O2 debt in air (hypoxia); with a mammal's lung
    it breathes freely. The skin supplies a small body's needs (supply / costs ~ m^(-1/3))."""
    env = B.constant_env(t_air_k=300.0, dt_s=3600.0)
    big = make(2, size_kg=1.0, air_l_kg=[0.0, 0.0535])
    supply = B.o2_supply_mol_s(big, env) * B.OXY_J_PER_MOL_O2
    rest = B.resting_power_w(big)
    skin = B.SKIN_O2_MOL_S_M2_PA * B.shape(big)["area"] * 21200.0 * B.OXY_J_PER_MOL_O2
    assert float(supply[0, 0]) == pytest.approx(float(skin[0, 0]), rel=1e-5)          # no lung: the skin alone
    assert float(supply[0, 0]) < float(rest[0, 0]) < float(supply[0, 1])
    lung = B.LUNG_O2_MOL_S_KG * float(B.tissues(big)["lung"][0, 1]) * B.OXY_J_PER_MOL_O2 * float(
        B.inspired_po2(B._env(big, env), big.body_k)[0, 1]) / B.P_O2_REF_PA
    assert float(supply[0, 1] - supply[0, 0]) == pytest.approx(lung, rel=1e-4)
    sus = B.sustained_power_w(big, env)
    assert float(sus[0, 0]) == 0.0 and float(sus[0, 1]) > 0
    out = B.metabolism(big, env)
    assert float(out["anoxic_mol"][0, 0]) > 0 and float(out["anoxia"][0, 0]) > 0       # it suffocates in air
    assert float(out["anoxic_mol"][0, 1]) == 0.0
    # thermogenesis is capped by the supply too
    assert float(out["thermo_j"][0, 0]) == 0.0
    # supply over resting costs falls with size: small lungless bodies live on their skin
    sizes = [1e-3, 1e-2, 1e-1, 1.0]
    small = make(4, size_kg=sizes, air_l_kg=0.0)
    ratio = (B.o2_supply_mol_s(small, env) * B.OXY_J_PER_MOL_O2 / B.resting_power_w(small))[0].tolist()
    assert ratio == sorted(ratio, reverse=True) and ratio[0] > 1.0 > ratio[-1]
    # thin O2 means less supply (Fick), in proportion to the inspired pO2
    thin = B.constant_env(t_air_k=300.0, p_o2_pa=2500.0)
    lung_thin = B.o2_supply_mol_s(big, thin) - B.SKIN_O2_MOL_S_M2_PA * B.shape(big)["area"] * 2500.0
    lung_full = B.o2_supply_mol_s(big, env) - skin / B.OXY_J_PER_MOL_O2
    p_thin = float(B.inspired_po2(B._env(big, thin), big.body_k)[0, 1])
    p_full = float(B.inspired_po2(B._env(big, env), big.body_k)[0, 1])
    assert float(lung_thin[0, 1] / lung_full[0, 1]) == pytest.approx(p_thin / p_full, rel=1e-4)


# ------------------------------------------------------------------------------------------------ buoyancy
def test_a_body_floats_or_sinks_against_the_water_where_it_is():
    """Review M1: the sea is about 1,027 kg/m^3; ponds are fresh. A body between the two floats at sea and sinks in a
    pond; one without air sinks in both; one with enough air floats in both."""
    assert B.RHO_SEA == pytest.approx(1000.0 + 0.77 * 35, rel=1e-6)
    m, v = volume_per_kg_lean()
    fresh = (m / B.RHO_WATER - v) * 1e3                         # L/kg that make the founder as dense as fresh water
    salt = (m / B.RHO_SEA - v) * 1e3
    assert 0.0 < salt < fresh < 0.05
    mid = 0.5 * (salt + fresh)
    b = make(3, air_l_kg=[0.0, mid, 1.2 * fresh])
    rho = B.density(b)[0]
    for i, air in enumerate((0.0, mid, 1.2 * fresh)):
        assert float(rho[i]) == pytest.approx(m / (v + air * 1e-3), rel=1e-5)
    assert float(rho[0]) > B.RHO_SEA > float(rho[1]) > B.RHO_WATER > float(rho[2])
    sea = sea_geom()
    assert B.sunk(b, sea)[0].tolist() == [True, False, False]                  # at sea the middle one floats
    s = B.submerged_share(b, sea)[0]
    assert float(s[0]) == 1.0 and float(s[1]) == pytest.approx(float(rho[1]) / B.RHO_SEA, rel=1e-5)
    geom, pond = pond_geom()
    assert B.sunk(b, geom, pond)[0].tolist() == [True, True, False]           # in a fresh pond it sinks
    assert float(B.submerged_share(b, geom, pond)[0, 2]) == pytest.approx(float(rho[2]) / B.RHO_WATER, rel=1e-5)
    # a fat body floats without air; air and fat add their buoyancy
    fat = make(2, air_l_kg=[0.0, 0.1], fat=0.6)
    r = B.density(fat)[0]
    assert float(r[0]) < B.RHO_WATER and float(r[1]) < float(r[0])


def test_a_dense_body_treads_water_while_it_moves_with_the_power_for_it():
    """Review M2: holding the airway up needs a vertical force of the weight in water; its actuator-disc power is a
    small share of a body's sustained power, so a moving body treads and a still one sinks; a body too weak to tread
    sinks while it moves."""
    sea = sea_geom()
    env = B.constant_env(t_air_k=300.0, dt_s=600.0)
    strong = make(2, air_l_kg=0.01, fat=0.0)
    rho = float(B.density(strong)[0, 0])
    assert rho > B.RHO_SEA
    sh = B.shape(strong)
    force = (rho - B.RHO_SEA) / rho * float(sh["m"][0, 0]) * K.G_STANDARD
    ideal = force ** 1.5 / math.sqrt(2 * B.RHO_SEA * float(sh["frontal"][0, 0]))
    tread = B.tread_power_w(strong, sh, rho_w=B.RHO_SEA)
    assert float(tread[0, 0]) == pytest.approx(ideal / (B.PADDLE_EFF * B.MUSCLE_EFF), rel=1e-4)
    assert float(tread[0, 0]) < 0.1 * float(B.sustained_power_w(strong, env)[0, 0]) / B.MUSCLE_EFF
    r = B.move(strong, torch.zeros(1, 2), torch.tensor([[1.0, 0.0]]), env, geom=sea)
    assert r["sunk_s"][0].tolist() == [0.0, 600.0]                              # moving it treads; still it sinks
    assert float(r["dist_m"][0, 0]) > 0
    # floating bodies need no treading
    assert float(B.tread_power_w(make(1, air_l_kg=0.5))) == 0.0
    # a body without the sustained power to tread sinks while it moves
    weak = make(1, air_l_kg=0.01, fat=0.0, muscle=1e-3)
    assert float(B.sustained_power_w(weak, env)) * 1.0 < float(B.tread_power_w(weak, rho_w=B.RHO_SEA)) * B.MUSCLE_EFF
    r = B.move(weak, torch.zeros(1, 1), torch.ones(1, 1), env, geom=sea)
    assert float(r["sunk_s"]) == pytest.approx(600.0)


# ------------------------------------------------------------------------------------------------ the O2 store
def test_the_store_is_the_air_from_the_alveolar_level_down_to_a_share_of_it_and_the_carrier():
    """Review H2 and L2: the lung's air is used from the alveolar pO2 (inspired x (1 - extraction)) down to
    O2_HOLD_FLOOR_SHARE of it, so the air gene stores O2 on thin-air planets too; the carrier unloads between the same
    two pressures by the aerobic law (one O2-transport rule for breathing and breath-holding)."""
    b = make(3, T0=310.0, air_l_kg=[0.0, 0.1, 0.2], o2_carrier=[15.0, 15.0, 0.0])
    for p_o2 in (21200.0, 2504.6):                                   # Earth and world7 781's sea-level pO2
        env = B.constant_env(t_air_k=300.0, p_o2_pa=p_o2, p_air_pa=101325.0 if p_o2 > 1e4 else 53707.0)
        parts = B.o2_store_parts(b, env)
        lung, car = parts["lung"][0].double(), parts["carrier"][0].double()
        assert float(lung[0]) == 0.0 and float(lung[1]) > 0                    # air 0 stores nothing in the lung
        assert float(lung[2]) == pytest.approx(2 * float(lung[1]), rel=1e-5)   # the store grows with the air
        e = B._env(b, env)
        p_alv = float(B.inspired_po2(e, b.body_k)[0, 1]) * (1 - B.RESP_EXTRACTION)
        want = 0.1e-3 * float(b.mass_kg[0, 1]) * p_alv * (1 - B.O2_HOLD_FLOOR_SHARE) / (K.R_GAS * 310.0)
        assert float(lung[1]) == pytest.approx(want, rel=1e-4)                 # ideal gas, n = V dp / (R T)
        bound = 15.0 * float(b.mass_kg[0, 0]) / B.ML_PER_MOL_STP
        a = lambda p: p / (p + B.O2_HALF_PA)                                   # noqa: E731
        assert float(car[0]) == pytest.approx(bound * (a(p_alv) - a(B.O2_HOLD_FLOOR_SHARE * p_alv)), rel=1e-4)
        assert float(car[2]) == 0.0
    # no O2 in the air, no store at all
    none = B.o2_store_mol(b, B.constant_env(t_air_k=300.0, p_o2_pa=0.0))
    assert float(none.abs().max()) == 0.0
    # a resting mammal-like 70 kg body (0.054 L/kg, 15 mL/kg) loses consciousness within about 1-3 min
    m = make(1, T0=310.0, size_kg=70.0, air_l_kg=0.0535, o2_carrier=15.0, muscle_frac=0.4)
    hold = float(B.breath_hold_s(m, B.constant_env(t_air_k=310.0)))
    assert 40.0 < hold < 200.0, hold


def test_a_breath_hold_within_the_store_is_harmless_and_the_debt_beyond_it_is_anoxia():
    """Review H1: damage only for the O2 debt beyond the store, at ANOXIA_TOLERANCE_S of the resting use; a short dive
    each bout does nothing; the debt of a dive that runs on is carried into the next bout; drowning is booked as
    anoxia."""
    sea = sea_geom()
    env = B.constant_env(t_air_k=300.0, dt_s=3600.0)
    probe = make(1, air_l_kg=0.01, fat=0.0)
    o2_use = float(B.resting_power_w(probe)) / B.OXY_J_PER_MOL_O2
    hold = float(B.breath_hold_s(probe, env))
    assert 10.0 < hold < 1000.0
    # 5 s, 30 s and 60 s under water in each of many hourly bouts: within the store, no damage at all
    for dive in (5.0, min(30.0, 0.5 * hold), 0.8 * hold):
        b = make(1, air_l_kg=0.01, fat=0.0)
        for _ in range(24):
            out = B.metabolism(b, env, loco={"metabolic_j": torch.zeros(1, 1), "heat_j": torch.zeros(1, 1),
                                             "sunk_s": torch.full((1, 1), dive)})
            b.reserve_j[:] = 0.02 * b.mass_kg * B.FAT_J_KG                      # fed
            b.water_kg[:] = B.LEAN_WATER * b.mass_kg                           # watered
        assert float(out["anoxic_mol"]) == 0.0 and float(b.anoxia) == 0.0, dive
        assert float(b.damage) < 1e-3, dive                                     # only wear
    # one dive past the store: the debt beyond it is anoxia
    b = make(1, air_l_kg=0.01, fat=0.0)
    out = B.metabolism(b, env, loco={"metabolic_j": torch.zeros(1, 1), "heat_j": torch.zeros(1, 1),
                                     "sunk_s": torch.full((1, 1), 2 * hold)})
    use = float(out["metabolic_w"]) / B.OXY_J_PER_MOL_O2
    store = float(out["o2_store_mol"])
    assert float(out["anoxic_mol"]) == pytest.approx(2 * hold * use - store, rel=1e-3)
    q_ref = float(B.maintenance_ref_w(b)) / B.OXY_J_PER_MOL_O2
    assert float(out["anoxia"]) == pytest.approx(float(out["anoxic_mol"]) / (B.ANOXIA_TOLERANCE_S * q_ref), rel=2e-3)
    assert float(b.o2_debt_mol) == 0.0                                          # it breathed again: repaid
    # a dive that runs on: the debt is carried, and only its new part beyond the store is booked
    b = make(1, air_l_kg=0.01, fat=0.0)
    short = B.constant_env(t_air_k=300.0, dt_s=0.6 * hold)
    loco = B.move(b, torch.zeros(1, 1), torch.zeros(1, 1), short, geom=sea)
    assert float(loco["dive_all"]) == 1.0
    o1 = B.metabolism(b, short, geom=sea, loco=loco)
    assert float(o1["anoxic_mol"]) == 0.0 and float(b.o2_debt_mol) > 0.5 * float(o1["o2_store_mol"])
    loco = B.move(b, torch.zeros(1, 1), torch.zeros(1, 1), short, geom=sea)
    o2 = B.metabolism(b, short, geom=sea, loco=loco)
    total = float(b.o2_debt_mol)
    assert float(o2["anoxic_mol"]) == pytest.approx(total - float(o2["o2_store_mol"]), rel=1e-3)
    # drowned: a death that anoxia dominates is booked as anoxia
    b = make(1, air_l_kg=0.01, fat=0.0)
    B.metabolism(b, env, geom=sea, loco=B.move(b, torch.zeros(1, 1), torch.zeros(1, 1), env, geom=sea))
    assert float(b.damage) >= 1.0 and float(b.anoxia) >= 0.5 * float(b.damage)
    assert B.death_cause(b)[0].tolist() == [B.CAUSE["anoxia"]]
    assert "anoxia" in B.CAUSES and "deaths_anoxia" in B.LEDGER_KEYS
    # with use proportional to cost, a dive that a cold, slow body survives harms a warm one (no fixed time)
    warm, cold = make(1, T0=310.0, air_l_kg=0.01, fat=0.0), make(1, T0=285.0, air_l_kg=0.01, fat=0.0)
    assert float(B.breath_hold_s(cold, env)) > float(B.breath_hold_s(warm, env))
    assert o2_use > 0


def test_dive_pieces_merge_in_order():
    """The dive structure of a bout from its parts in order: a dive continues across parts until a breath."""
    t = lambda *v: torch.tensor([list(v)])                                      # noqa: E731
    air = B.dive_piece(t(0.0), t(100.0))
    under = B.dive_piece(t(100.0), t(100.0))
    d = B.dive_merge(B.dive_merge(under, under), air)
    assert d["dive_lead_s"].tolist() == [[200.0]] and float(d["dive_all"]) == 0.0
    d = B.dive_merge(B.dive_merge(air, under), B.dive_merge(under, air))
    assert d["dive_inner_s"].tolist() == [[200.0]] and d["dive_lead_s"].tolist() == [[0.0]]
    d = B.dive_merge(air, under)
    assert d["dive_trail_s"].tolist() == [[100.0]]
    nothing = B.dive_piece(t(0.0), t(0.0))                                      # a stretch of no time is neutral
    assert B.dive_merge(under, nothing)["dive_all"].tolist() == [[1.0]]


# ------------------------------------------------------------------------------------------------ heat in water
def test_a_still_body_in_cold_water_loses_heat_by_free_convection():
    """Review M4: free convection (Churchill) and the water's drift cool a still body in water far faster than
    conduction alone (Nu = 2 before), and faster than in still air."""
    b = make(1, size_kg=1.0, T0=310.0, fur_m=1e-3)
    still = B.constant_env(t_air_k=280.0, wind_m_s=0.0)
    e = B._env(b, still)
    sh = B.shape(b)
    in_water = B.heat_exchange(b, e, submerged=torch.ones(1, 1), sh=sh)
    in_air = B.heat_exchange(b, e, submerged=torch.zeros(1, 1), sh=sh)
    assert float(in_water["G"]) > 2 * float(in_air["G"])
    d = 2 * float(sh["r2"])
    ra = K.G_STANDARD * float(B.water_expansion(torch.tensor(280.0))) * 30.0 * d ** 3 / (B.NU_WATER * B.ALPHA_WATER)
    nu_free = float(B.nusselt_free(torch.tensor(ra), B.PR_WATER))
    assert nu_free > 50                                               # about 100 at Ra ~1e9 (review)
    assert float(B.water_expansion(torch.tensor(293.15))) == pytest.approx(2.07e-4, rel=0.03)
    assert float(B.water_expansion(torch.tensor(277.13))) < 1e-5     # water is densest near 4 C
    # the water's drift with the wind adds forced convection
    windy = B.heat_exchange(b, B._env(b, B.constant_env(t_air_k=280.0, wind_m_s=10.0)), submerged=torch.ones(1, 1))
    assert float(windy["G"]) > float(in_water["G"])


# ------------------------------------------------------------------------------------------------ the books
def test_the_air_carries_no_mass_to_the_books_and_the_ledgers_close():
    # two bodies alike but for their air hold the same stocks
    a = make(2, air_l_kg=[0.0, 0.0])
    c = make(2, air_l_kg=[0.0, 0.5])
    for k, v in B.stocks(a).items():
        assert float(B.stocks(c)[k]) == pytest.approx(float(v), rel=1e-7), k
    # bodies of every air store moving, eating and dividing on a coast with the sea on half of it, for 40 bouts
    n = 24
    z = torch.full((1, 16, 16), 10.0)
    z[0, :, :8] = -10.0
    geom = pt.geometry_from_elevation(z, L, 0.0, 5.674e6)
    gen = torch.Generator().manual_seed(9)
    b = make(n, N=64, air_l_kg=torch.linspace(0.0, 0.3, n).tolist(), size_kg=0.05, o2_carrier=15.0)
    b.pos[:, :n, 0] = torch.linspace(4.0, 252.0, n)
    b.pos[:, :n, 1] = 128.0
    B.reset_ledgers(b)
    env = B.constant_env(t_air_k=300.0)
    deaths = 0.0
    for _ in range(40):
        motor = {"turn": torch.rand(1, 64, generator=gen) * 2 - 1, "thrust": torch.rand(1, 64, generator=gen) * 2 - 1,
                 "mouth": torch.rand(1, 64, generator=gen), "divide": torch.rand(1, 64, generator=gen)}
        out = B.bout(b, motor, env, gen, geom=geom)
        deaths += float(out["deaths"]["dead"].sum())
        closed(b)
    assert float(b.ledger["births"]) > 0 and deaths > 0
