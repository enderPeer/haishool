"""Tests of haishool.life9.planet.creatures (PLANET-SPEC 2.8 and the creature gates of section 4)."""
import math
import re

import pytest

torch = pytest.importorskip("torch")

from haishool.life9.planet import biosphere as bs  # noqa: E402
from haishool.life9.planet import brain, chain, items, materials, senses  # noqa: E402
from haishool.life9.planet import creatures as C  # noqa: E402
from haishool.life9.planet import globe as G  # noqa: E402

TAGS = ("chain", "derived", "reference", "new_rule")
R = 10000.0
EARTH_O2 = 21200.0


@pytest.fixture(scope="module")
def glb():
    return G.Globe(6)


def flat_terrain(glb, W, elevation=0.0):
    Cc = glb.C
    return {"elevation_m": torch.full((W, Cc), float(elevation)), "sea_level_m": torch.full((W,), -1.0),
            "depth_m": torch.zeros(W, Cc), "land": torch.ones(W, Cc, dtype=torch.bool),
            "coast": torch.zeros(W, Cc, dtype=torch.bool)}


def one_animal(glb, mass=1.0, insulation=1.0, seed=0, capacity=4, founders=1, W=1):
    gen = torch.Generator().manual_seed(seed)
    ter = flat_terrain(glb, W)
    cr = C.spawn_founders(chain.synthetic_inputs(3), glb, ter, gen, capacity=capacity, founders=founders, hidden=8,
                          initial_hidden=4, adult_mass_kg=mass)
    cr.genome["insulation"][:] = insulation
    cr.genome["acuity"][:] = 1.0
    cr.genome["hearing"][:] = 1.0
    cr.genome["diet"][:] = 0.0
    return cr, ter, gen


def rebase(cr):
    """Restart the ledgers from the current stocks after a test edits the state by hand."""
    s = C.stocks(cr)
    for v in cr.ledger.values():
        v.zero_()
    cr.ledger["spawned_j"] += s["energy"]
    cr.ledger["spawned_c"] += s["carbon"]
    cr.ledger["spawned_w"] += s["water"]


def env_of(glb, W, t_k, p_o2=EARTH_O2):
    return {"t_air_k": torch.full((W, glb.C), float(t_k)), "p_o2_pa": torch.full((W,), float(p_o2))}


def live_day(cr, glb, ter, env, *, plants, soil, pool):
    C.eat_plants(cr, cr.alive, glb, C.field_harvester(plants))
    C.drink(cr, cr.alive, glb, soil, ter["land"], radius_m=R, t_air_k=env["t_air_k"])
    C.physiology(cr, glb, env)
    return C.deaths(cr, pool)


def days_to_death(glb, *, food, water, t_k, mass=1.0, age=None, p_o2=EARTH_O2, limit=400):
    cr, ter, _ = one_animal(glb, mass)
    if age is not None:
        cr.age_d[cr.alive] = age
    W = 1
    env = env_of(glb, W, t_k, p_o2)
    plants = torch.full((W, glb.C), 1e9 if food else 0.0)
    soil = torch.full((W, glb.C), 100.0 if water else 0.0)
    pool = items.new_item_pool(W, 32)
    for day in range(1, limit + 1):
        out = live_day(cr, glb, ter, env, plants=plants, soil=soil, pool=pool)
        if bool(out["dead"].any()):
            return day, C.CAUSES[int(out["cause"][out["dead"]][0])], cr
    return None, "alive", cr


# ------------------------------------------------------------------------------------ constants
def test_body_constants_are_derived_from_the_materials_makeup():
    assert C.TISSUE_J_KG == pytest.approx(7.216e6, rel=1e-3)
    assert C.TISSUE_C_KG == pytest.approx(0.1557, rel=1e-3)
    assert 0.6 < C.WATER_FRACTION < 0.7                      # mammals: total body water 0.6-0.7
    assert C.FAT_C_PER_J * C.FAT_J_KG == pytest.approx(materials.ELEMENT_FRACTION[materials.IDX["fat"]][1])
    assert 1.05 < C.GROWTH_J_KG / C.TISSUE_J_KG < 1.2         # a cost of growth, not a gain
    # every food carries at least fat's carbon per joule, so the conversion CO2 is never negative
    for s in ("meat", "bone", "hide", "fat", "plant_fiber"):
        cj = materials.ELEMENT_FRACTION[materials.IDX[s]][1] / float(materials.FOOD_J_KG[s])
        assert cj >= C.FAT_C_PER_J * (1 - 1e-9)
    assert C.PLANT_C_FRACTION / C.PLANT_J_KG > C.FAT_C_PER_J
    # fat respired alone has the RQ of fat oxidation, about 0.71
    rq = C.FAT_C_PER_J / C.M_C * C.OXY_J_MOL
    assert 0.69 < rq < 0.74


def test_allometry_values():
    one = torch.tensor(1.0)
    assert float(C.basal_w(one)) == pytest.approx(3.4)
    assert float(C.lifespan_d(one)) == pytest.approx(11.6 * 365.25)
    assert float(C.maturity_d(one)) == pytest.approx(0.6 * 365.25)
    assert float(C.gestation_d(torch.tensor(16.0))) == pytest.approx(130.0)
    assert float(C.litter_mass_kg(one)) == pytest.approx(0.1)
    assert float(C.aerobic(torch.tensor(3000.0))) == pytest.approx(0.5)
    # the von Bertalanffy rate reaches 90 % of adult at maturity from the neonate mass
    adult, m0 = torch.tensor(30.0), torch.tensor(1.0)
    k = C.vb_k(adult, m0)
    t = C.maturity_d(adult)
    m = (adult ** (1 / 3) - (adult ** (1 / 3) - m0 ** (1 / 3)) * torch.exp(-k * t)) ** 3
    assert float(m / adult) == pytest.approx(0.9, rel=1e-4)


def _numeric_constants(mod):
    """Upper-case numbers defined in the module (not imported from constants, which is all reference)."""
    from haishool.life9.planet import constants
    return [k for k, v in vars(mod).items() if k.isupper() and not k.startswith("_") and not hasattr(constants, k)
            and isinstance(v, (int, float, tuple, dict)) and not isinstance(v, bool)]


def test_every_numeric_constant_is_named_in_the_provenance():
    for mod in (C, senses):
        text = " ".join(f"{k} {note}" for k, (_, note) in mod.PROVENANCE.items())
        skip = {"PROVENANCE", "LEDGER", "CAUSES", "CAUSE", "BODY_GENES", "BODY_SPECS", "GENE_SPECS", "TWO_PI", "M_C",
                "SECTOR_CHANNELS", "SELF_CHANNELS", "SLOT_CHANNELS", "RICH_CLASSES", "SECTOR_DIM", "SELF0", "IN_DIM",
                "PLANT_INDEX", "SECTORS", "VOCAL", "K_SLOTS", "SENSE_SHARE_781", "SENSE_UNIT", "COT_EXP",
                "V_TROT_EXP", "WATER_EXP", "LIFESPAN_EXP", "MATURITY_EXP", "GESTATION_EXP", "LITTER_EXP",
                "CALL_REF_KG", "CALL_MASS_EXP", "PLANT_DIG_DIET", "T_LC_PER_INS", "ETA_LIFT", "C_SOUND_EARTH",
                "FAT_MIN_KG", "FOUNDER_SPEED", "FOUNDER_SWIM", "FOUNDER_LITTER", "C_EARTH", "FIELD_HEIGHT_M",
                "FIRE_HEIGHT_M", "SOIL_FULL", "TEMP_REF_K", "HARD_SCALE", "ITEM_MASS_KG", "ITEM_J", "ITEM_TEMP_K"}
        for name in _numeric_constants(mod):
            if name in skip:
                continue
            assert name in text, f"{mod.__name__}.{name} has no provenance entry"
    # the grouped ones are named in their group's note or key
    groups = {"COT_EXP": "COT_J_KG_M", "V_TROT_EXP": "V_TROT", "WATER_EXP": "WATER_TURNOVER", "ETA_LIFT": "ETA_LIFT",
              "LITTER_EXP": "LITTER_MASS", "T_LC_PER_INS": "T_LC0"}
    for name, key in groups.items():
        assert key in C.PROVENANCE, name


def test_provenance_tags_cover_the_set_up_numbers():
    for mod in (C, senses):
        for key, (tag, note) in mod.PROVENANCE.items():
            assert tag in TAGS, key
            assert note, key
    for name in ("KLEIBER_W", "COT_J_KG_M", "WATER_TURNOVER", "LIFESPAN_YR", "FOUNDER_MASS_KG", "SENSE_W",
                 "BODY_MIX", "GROWTH_J_KG", "CORPSE", "SWIM_CD", "AEROBIC_SCOPE"):
        assert name in C.PROVENANCE
    for name, spec in C.BODY_SPECS.items():
        assert spec.source, name
        assert name in C.BODY_GENES
    # numbers the spec chose without a published source are modelling choices, not references
    for name in ("T_LC0", "AEROBIC_HALF_PA", "HYPOXIA_PA", "DEHYDRATION_LETHAL", "DRINK_SOIL_MIN", "SWIM_SALT",
                 "PLANT_DIG0", "RESERVE_MAX_FRACTION", "CORPSE", "PLANT_INTAKE", "founder_diet"):
        assert C.PROVENANCE[name][0] == "new_rule", name
    assert senses.PROVENANCE["FRESH_SOIL"][0] == "new_rule"
    for mod in (C, senses):
        for key, (tag, note) in mod.PROVENANCE.items():
            if tag == "reference":                             # a cited source: an author and year, or a standard
                assert re.search(r"(1[89]\d\d|20[0-2]\d)|CRC|ISO|CODATA|NIST|USDA", note), key


# ------------------------------------------------------------------------------------ founders
def test_founders_carry_the_chain_genes_of_781(glb):
    inputs = chain.chain_inputs(781)
    W = 2
    gen = torch.Generator().manual_seed(4)
    ter = G.make_terrain(glb, W, gen, 600.0, 3e10, R)
    cr = C.spawn_founders(inputs, glb, ter, gen, capacity=32, founders=12, hidden=16, initial_hidden=8)
    g = cr.genome
    a = cr.alive
    assert a.sum(1).tolist() == [12, 12] and bool(a[:, :12].all())
    assert torch.allclose(g["acuity"][a], torch.tensor(2.04))
    assert torch.allclose(g["hearing"][a], torch.tensor(0.685))
    assert torch.allclose(g["voice"][a], torch.tensor(0.08))
    assert torch.allclose(g["diet"][a], torch.tensor(0.034))
    assert torch.allclose(g["insulation"][a], torch.tensor((303.0 - 288.0) / 8))
    assert bool((g["adult_mass_kg"][a] == C.FOUNDER_MASS_KG).all()) and bool((g["litter"][a] == 2).all())
    # the only innate wiring: forage bias and the plant input through relay unit 0
    assert torch.allclose(g["bo"][..., brain.FORAGE_OUT][a], torch.tensor(1.0))
    assert torch.allclose(g["Wx"][..., senses.PLANT_INDEX, 0][a], torch.tensor(2.0))
    assert torch.equal(cr.wh_live, g["Wh"])
    # on land, young adults, uid per world
    assert bool(ter["land"].gather(1, glb.cell_of(cr.pos))[a].all())
    assert torch.allclose(cr.age_d[a], C.maturity_d(g["adult_mass_kg"][a]))
    assert cr.uid[0, :12].tolist() == list(range(12)) and cr.next_uid.tolist() == [12, 12]
    assert torch.allclose(cr.pos.norm(dim=-1), torch.ones(W, 32))
    table = C.founder_genes(inputs)
    assert table["acuity"][1] == "chain" and table["insulation"][1] == "derived"
    assert table["adult_mass_kg"][1] == "new_rule"
    # the bodies era's predator_share is the predators' share of all biomass, a proxy for the meat share
    assert table["diet"][1] == "new_rule" and "ALL biomass" in table["diet"][2]


def test_founders_lie_on_their_drawn_land_cells_at_G48():
    glb48 = G.Globe(48)
    gen = torch.Generator().manual_seed(9)
    ter = G.make_terrain(glb48, 2, gen, 600.0, 3e10, R)
    cr = C.spawn_founders(chain.synthetic_inputs(3), glb48, ter, gen, capacity=2048, founders=2048, hidden=4,
                          initial_hidden=2)
    cell = glb48.cell_of(cr.pos)
    assert bool(ter["land"].gather(1, cell).all())
    # spread over their cells, not stacked on the centres
    off = G.angle(cr.pos, glb48.centers[cell])
    assert float((off < 1e-6).float().mean()) < 0.01 and float(off.max()) < glb48.delta


def test_founders_are_seeded(glb):
    a, _, _ = one_animal(glb, seed=5, founders=3, capacity=6)
    b, _, _ = one_animal(glb, seed=5, founders=3, capacity=6)
    assert C.state_hash(a) == C.state_hash(b)
    c, _, _ = one_animal(glb, seed=6, founders=3, capacity=6)
    assert C.state_hash(a) != C.state_hash(c)


# ------------------------------------------------------------------------------------ the creature gates
def test_fed_and_watered_individual_keeps_its_reserve_in_mild_weather(glb):
    for mass in (1.0, 30.0):
        cr, ter, _ = one_animal(glb, mass)
        start = cr.reserve_j[cr.alive].clone()
        env = env_of(glb, 1, 296.0)                       # just above T_lc = 295 K at insulation 1
        plants = torch.full((1, glb.C), 1e9)
        soil = torch.full((1, glb.C), 100.0)
        pool = items.new_item_pool(1, 8)
        for _ in range(60):
            assert not bool(live_day(cr, glb, ter, env, plants=plants, soil=soil, pool=pool)["dead"].any())
        assert bool(cr.alive.any())
        assert float(cr.reserve_j[cr.alive]) >= float(start)
        assert float(cr.health[cr.alive]) == 1.0
        assert float(cr.water_kg[cr.alive]) >= C.DEHYDRATION_LETHAL * C.WATER_FRACTION * mass


def test_dehydration_cold_starvation_and_age_kill_in_that_order(glb):
    t_dry, why_dry, _ = days_to_death(glb, food=True, water=False, t_k=296.0)
    # 200 K: 0.174 W/K x 95 K = 16.5 W of thermogenesis plus basal outruns the 12.8 W of full grazing
    t_cold, why_cold, _ = days_to_death(glb, food=True, water=True, t_k=200.0)
    t_hunger, why_hunger, _ = days_to_death(glb, food=False, water=True, t_k=296.0)
    life = float(C.lifespan_d(torch.tensor(1.0)))
    t_age, why_age, _ = days_to_death(glb, food=True, water=True, t_k=296.0, age=math.ceil(life) - 3)
    assert why_dry == "dehydration" and why_hunger == "starvation" and why_age == "age"
    assert why_cold in ("starvation", "cold")               # cold drains the reserve faster than food refills it
    assert 1 <= t_dry <= 4
    assert t_dry < t_cold < t_hunger < 120
    assert t_age == 3 and t_hunger < life
    # beyond the aerobic ceiling (thin air: 1.5 kPa O2, a = 1/3) cold kills by hypothermia within days, even fed
    t_freeze, why_freeze, _ = days_to_death(glb, food=True, water=True, t_k=220.0, p_o2=1500.0)
    assert why_freeze == "cold" and t_freeze <= 5
    # hypoxia below 1 kPa O2
    t_hyp, why_hyp, _ = days_to_death(glb, food=True, water=True, t_k=296.0, p_o2=400.0)
    assert why_hyp == "hypoxia" and t_hyp <= 2


def test_cold_costs_the_published_conductance(glb):
    """Herreid & Kessel 1967: minimal conductance 1.02 W_g^-0.505 ml O2 g^-1 h^-1 C^-1, about 0.17 W/K
    for 1 kg. A fed 1 kg animal (insulation 1, T_lc 295 K) at 270 K pays 0.17 x 25 = 4.4 W and keeps
    alive (the spec's 1.0 W/K would cost 25 W, beyond its 12.8 W of grazing)."""
    one_kg = 1.02 * 1000.0 ** -0.505 * 1000.0 * 20.1 / 3600.0         # W/K of a 1 kg mammal
    assert C.C_TH == pytest.approx(one_kg, rel=0.02) and 0.16 < C.C_TH < 0.19
    assert C.PROVENANCE["C_TH"][0] == "reference" and "Herreid" in C.PROVENANCE["C_TH"][1]
    cr, ter, _ = one_animal(glb, 1.0)
    out = C.physiology(cr, glb, env_of(glb, 1, 270.0))
    a = cr.alive
    assert float(out["thermo_j"][a]) == pytest.approx(C.C_TH * 25.0 * 86400.0, rel=1e-5)
    t, why, _ = days_to_death(glb, food=True, water=True, t_k=270.0, limit=60)
    assert why == "alive"


def test_heat_evaporates_the_metabolic_heat_and_drinkers_keep_up(glb):
    cr, ter, _ = one_animal(glb, 30.0)
    a = cr.alive
    out = C.physiology(cr, glb, env_of(glb, 1, 320.0))
    resp = float(out["o2_mol"][a]) * C.OXY_J_MOL
    gain = C.C_TH * 30 ** 0.5 / 1.0 * 10.0 * 86400.0
    assert float(out["evap_kg"][a]) == pytest.approx((resp + gain) / C.LATENT_J_KG, rel=1e-5)
    turnover = C.WATER_TURNOVER * 30 ** C.WATER_EXP
    assert float(out["water_loss_kg"][a]) == pytest.approx(turnover + float(out["evap_kg"][a]), rel=1e-5)
    # at 309 K nothing is evaporated
    cr2, _, _ = one_animal(glb, 30.0)
    assert float(C.physiology(cr2, glb, env_of(glb, 1, 309.0))["evap_kg"].abs().max()) == 0.0
    # with water a 1 kg animal lives through a 320 K month (drinking for the heat too); without, it dries out
    t, why, _ = days_to_death(glb, food=True, water=True, t_k=320.0, limit=30)
    assert why == "alive"
    t, why, _ = days_to_death(glb, food=True, water=False, t_k=320.0)
    assert why == "dehydration" and t <= 2


def test_starvation_burns_lean_tissue_and_books_its_carbon(glb):
    cr, ter, _ = one_animal(glb, 1.0)
    cr.reserve_j[cr.alive] = 1000.0
    rebase(cr)
    out = C.physiology(cr, glb, env_of(glb, 1, 296.0))
    a = cr.alive
    assert float(out["burnt_kg"][a]) > 0
    assert float(cr.mass_kg[a]) < 1.0 and float(cr.frame_kg[a]) == 1.0
    assert abs(float(cr.reserve_j[a])) < 1e-2
    err = C.ledger_errors(cr)
    for k in ("energy", "carbon", "water"):
        assert abs(float(err[k][0])) <= 1e-6 * float(err[f"{k}_scale"][0])


# ------------------------------------------------------------------------------------ health
def fed_day(cr, glb, ter, env, pool):
    W = cr.shape[0]
    C.eat_plants(cr, cr.alive, glb, C.field_harvester(torch.full((W, glb.C), 1e9)))
    C.drink(cr, cr.alive, glb, torch.full((W, glb.C), 100.0), ter["land"], radius_m=R)
    C.physiology(cr, glb, env)
    return C.deaths(cr, pool)


def test_a_lethal_strike_is_not_healed_and_a_strike_blocks_that_days_healing(glb):
    cr, ter, _ = one_animal(glb, 5.0, founders=4, capacity=4)
    env = env_of(glb, 1, 296.0)
    pool = items.new_item_pool(1, 32)
    w, n = torch.zeros(4, dtype=torch.long), torch.arange(4)
    C.injure(cr, w, n, torch.tensor([1.0, 1.03, 0.5, 0.0]))
    out = fed_day(cr, glb, ter, env, pool)
    assert out["dead"][0].tolist() == [True, True, False, False]
    assert [C.CAUSES[int(c)] for c in out["cause"][0, :2]] == ["injury", "injury"]
    assert float(cr.health[0, 2]) == pytest.approx(0.5) and float(cr.health[0, 3]) == 1.0   # no healing that day
    assert float(cr.struck.abs().max()) == 0.0                                                # cleared by physiology
    fed_day(cr, glb, ter, env, pool)
    assert float(cr.health[0, 2]) == pytest.approx(0.5 + C.HEAL_PER_DAY)                      # heals the next day


def test_the_cause_is_the_largest_harm_of_the_day(glb):
    """A strike to -0.5 then a day of hypothermia (thin air, 220 K: 0.4 health lost) dies of injury;
    a light strike (0.1) on the same day as the cold is recorded as cold."""
    cr, ter, _ = one_animal(glb, 1.0, founders=2, capacity=2)
    pool = items.new_item_pool(1, 32)
    C.injure(cr, torch.tensor([0, 0]), torch.tensor([0, 1]), torch.tensor([1.5, 0.1]))
    env = env_of(glb, 1, 220.0, p_o2=1500.0)
    out = C.physiology(cr, glb, env)
    gap = float(out["cold_gap"][0, 1])
    assert 0.2 < gap < 1.0
    assert float(cr.health[0, 1]) == pytest.approx(0.9 - gap, rel=1e-5)
    dead = C.deaths(cr, pool)
    assert dead["dead"][0].tolist() == [True, False]
    assert C.CAUSES[int(dead["cause"][0, 0])] == "injury" and C.CAUSES[int(cr.harm[0, 1])] == "cold"


# ------------------------------------------------------------------------------------ dtypes
def test_float64_fields_keep_the_state_float32(glb):
    """climate.pressures gives float64 [W] Pa; nothing of the state may turn float64."""
    W = 2
    gen = torch.Generator().manual_seed(7)
    ter = G.make_terrain(glb, W, gen, 600.0, 2e10, R)
    cr = C.spawn_founders(chain.synthetic_inputs(8), glb, ter, gen, capacity=24, founders=8, hidden=8,
                          initial_hidden=4, adult_mass_kg=3.0)
    cr.cooldown_d[:] = 0.0
    before = {k: v.dtype for k, v in cr.state_dict().items() if k not in ("genome", "ledger")}
    genes = {k: v.dtype for k, v in cr.genome.items()}
    ter64 = {k: (v.double() if v.is_floating_point() else v) for k, v in ter.items()}
    env = {"t_air_k": torch.full((W, glb.C), 315.0, dtype=torch.float64),
           "p_o2_pa": torch.full((W,), EARTH_O2, dtype=torch.float64),
           "air_density_kg_m3": torch.full((W,), 1.2, dtype=torch.float64)}
    pool = items.new_item_pool(W, 64)
    for _ in range(3):
        travel = C.move(cr, glb, ter64, torch.rand(W, 24, dtype=torch.float64, generator=gen),
                        torch.rand(W, 24, dtype=torch.float64, generator=gen),
                        gravity_m_s2=torch.tensor([9.8, 3.7], dtype=torch.float64),
                        p_o2_pa=env["p_o2_pa"], radius_m=torch.tensor([R, R]).double())
        assert travel["cost_j"].dtype == torch.float32
        C.eat_plants(cr, cr.alive, glb, C.field_harvester(torch.full((W, glb.C), 50.0, dtype=torch.float64)))
        C.drink(cr, cr.alive, glb, torch.full((W, glb.C), 90.0, dtype=torch.float64), ter["land"], radius_m=R,
                t_air_k=env["t_air_k"])
        out = C.physiology(cr, glb, env, travel, heat_k=torch.zeros(W, 24, dtype=torch.float64),
                           insulation_add=torch.zeros(W, 24, dtype=torch.float64))
        assert out["o2_mol"].dtype == torch.float32
        C.injure(cr, torch.tensor([0]), torch.tensor([1]), torch.tensor([0.2], dtype=torch.float64))
        C.deaths(cr, pool)
        C.reproduce(cr, gen)
    after = {k: v.dtype for k, v in cr.state_dict().items() if k not in ("genome", "ledger")}
    assert after == before and before["reserve_j"] == torch.float32 and before["health"] == torch.float32
    assert {k: v.dtype for k, v in cr.genome.items()} == genes
    assert all(v.dtype == torch.float64 for v in cr.ledger.values())


def busy_days(cr, glb, ter, gen, pool, plants, days):
    W, N = cr.shape
    env = {"t_air_k": torch.full((W, glb.C), 290.0), "p_o2_pa": torch.full((W,), EARTH_O2)}
    soil = torch.full((W, glb.C), 80.0)
    harvest = C.field_harvester(plants)
    for day in range(days):
        travel = C.move(cr, glb, ter, torch.rand(W, N, generator=gen) * 2 - 1, torch.rand(W, N, generator=gen) * 0.05,
                        gravity_m_s2=9.0, p_o2_pa=EARTH_O2, radius_m=R)
        C.eat_meat(cr, cr.alive & (cr.genome["diet"] > 0.5), pool, radius_m=R, reach_m=4000.0)
        C.eat_plants(cr, cr.alive & (cr.genome["diet"] < 0.5), glb, harvest)
        C.drink(cr, cr.alive, glb, soil, ter["land"], radius_m=R)
        C.physiology(cr, glb, env, travel)
        if day % 4 == 1:
            w, n = cr.alive.nonzero(as_tuple=True)
            C.injure(cr, w[::5], n[::5], 1.5)
        C.deaths(cr, pool)
        C.reproduce(cr, gen)


def test_cpu_runs_are_exact_and_resume_from_state(glb):
    def start():
        gen = torch.Generator().manual_seed(21)
        ter = G.make_terrain(glb, 2, gen, 600.0, 2e10, R)
        cr = C.spawn_founders(chain.synthetic_inputs(8), glb, ter, gen, capacity=40, founders=12, hidden=8,
                              initial_hidden=4, adult_mass_kg=1.0)
        cr.genome["diet"][:, ::3] = 0.9
        cr.cooldown_d[:] = 0.0
        return cr, ter, gen, items.new_item_pool(2, 128), torch.full((2, glb.C), 30.0)
    a, ter, gen_a, pool_a, plants_a = start()
    busy_days(a, glb, ter, gen_a, pool_a, plants_a, 24)
    b, ter, gen_b, pool_b, plants_b = start()
    busy_days(b, glb, ter, gen_b, pool_b, plants_b, 12)
    saved, gen_state = b.state_dict(), gen_b.get_state()
    pool_state, plant_state = pool_b.state_dict(), plants_b.clone()
    c = C.Creatures.from_state(saved)
    gen_c = torch.Generator()
    gen_c.set_state(gen_state)
    pool_c = items.from_state(pool_state)
    busy_days(c, glb, ter, gen_c, pool_c, plant_state, 12)
    assert float(a.ledger["births"].sum()) > 0 and float(a.ledger["deaths_injury"].sum()) > 0
    assert C.state_hash(a) == C.state_hash(c)
    assert items.state_hash(pool_a) == items.state_hash(pool_c)
    gas = C.gas_mol(a)
    assert bool((gas["o2_mol"] > 0).all()) and bool((gas["co2_mol"] > 0).all())


# ------------------------------------------------------------------------------------ movement
def test_move_follows_the_sphere_and_costs_by_physics(glb):
    cr, ter, _ = one_animal(glb, 30.0, founders=2, capacity=2)
    W = 1
    cr.genome["swim"][:] = 0.0
    p0 = cr.pos.clone()
    speed = torch.tensor([[1.0, 0.0]])
    out = C.move(cr, glb, ter, torch.zeros(W, 2), speed, gravity_m_s2=9.81, p_o2_pa=EARTH_O2, radius_m=R)
    v = C.V_TROT * 30 ** C.V_TROT_EXP * float(C.aerobic(torch.tensor(EARTH_O2)))
    assert float(out["dist_m"][0, 0]) == pytest.approx(v * C.MOVE_S, rel=1e-5)
    moved = float(G.angle(p0[0, 0], cr.pos[0, 0])) * R
    assert moved == pytest.approx(v * C.MOVE_S, rel=1e-3)
    assert torch.equal(cr.pos[0, 1], p0[0, 1]) and float(out["cost_j"][0, 1]) == 0.0
    cot = C.COT_J_KG_M * 30 ** C.COT_EXP * 30 * v * C.MOVE_S
    assert float(out["cost_j"][0, 0]) == pytest.approx(cot, rel=1e-4)            # flat land: walking only
    # climbing a uniform slope: ground rising 2 % northwards, walked due north from the equator
    ter2 = flat_terrain(glb, 1)
    ter2["elevation_m"][0] = 400.0 + 0.02 * R * glb.lat                       # above the sea level everywhere
    start = torch.tensor([1.0, 0.0, 0.0])
    cr.pos = start.expand(1, 2, 3).clone()
    cr.heading = torch.zeros(1, 2)
    slow = torch.tensor([[0.3, 0.0]])                                            # 6.5 km: up to 37 degrees north
    out2 = C.move(cr, glb, ter2, torch.zeros(W, 2), slow, gravity_m_s2=9.81, p_o2_pa=EARTH_O2, radius_m=R)
    d2 = float(out2["dist_m"][0, 0])
    assert float(out2["climb_m"][0, 0]) == pytest.approx(0.02 * d2, rel=0.03)
    walk = C.COT_J_KG_M * 30 ** C.COT_EXP * 30 * d2
    assert float(out2["cost_j"][0, 0]) == pytest.approx(walk + 30 * 9.81 * float(out2["climb_m"][0, 0]) / 0.25,
                                                        rel=1e-4)
    # walking back down costs no climbing
    cr.heading = torch.full((1, 2), math.pi)
    back = C.move(cr, glb, ter2, torch.zeros(W, 2), slow, gravity_m_s2=9.81, p_o2_pa=EARTH_O2, radius_m=R)
    assert float(back["climb_m"][0, 0]) < 0.02 * 0.02 * d2


def test_climb_converges_with_the_path_samples():
    """One sample per smallest cell spacing of the longest path: the climb over real terrain is within
    3 % of a 4 x finer sampling (8 fixed samples missed about a third)."""
    glb48 = G.Globe(48)
    gen = torch.Generator().manual_seed(2)
    ter = G.make_terrain(glb48, 1, gen, 600.0, 3e10, R)
    cr = C.spawn_founders(chain.synthetic_inputs(3), glb48, ter, gen, capacity=256, founders=256, hidden=8,
                          initial_hidden=4, adult_mass_kg=30.0)
    cr.genome["speed"][:] = 1.2
    p0, h0 = cr.pos.clone(), cr.heading.clone()
    climbs = {}
    for smp in (8, None, "fine"):
        cr.pos, cr.heading = p0.clone(), h0.clone()
        n = 4 * auto if smp == "fine" else smp
        out = C.move(cr, glb48, ter, torch.zeros(1, 256), torch.ones(1, 256), gravity_m_s2=9.8, p_o2_pa=EARTH_O2,
                     radius_m=R, samples=n)
        if smp is None:
            auto = out["samples"]
            longest = float(out["speed_m_s"].max()) * C.MOVE_S / R
            assert auto == math.ceil(longest / glb48.min_spacing) or auto == C.PATH_SAMPLES_MAX
        climbs[smp] = float(out["climb_m"].mean())
    assert climbs[None] == pytest.approx(climbs["fine"], rel=0.03)
    assert climbs[8] < 0.85 * climbs["fine"]


def test_swimming_costs_drag_and_sea_water_rehydrates_only_swimmers(glb):
    cr, ter, _ = one_animal(glb, 30.0, founders=2, capacity=2)
    sea = dict(ter)
    sea["land"] = torch.zeros_like(ter["land"])
    sea["elevation_m"] = torch.full_like(ter["elevation_m"], -100.0)
    sea["sea_level_m"] = torch.zeros(1)
    cr.genome["swim"][0, :] = torch.tensor([0.0, 1.0])
    out = C.move(cr, glb, sea, torch.zeros(1, 2), torch.full((1, 2), 0.5), gravity_m_s2=9.81, p_o2_pa=EARTH_O2,
                 radius_m=R)
    assert bool(out["in_water"].all()) and float(out["land_m"].abs().max()) == 0.0
    assert float(out["climb_m"].abs().max()) == 0.0
    drag_k = 0.5 * 1000.0 * C.SWIM_CD * C.MEEH_K * 30 ** (2 / 3)
    for i, eta in enumerate((C.ETA_PADDLE, C.ETA_LIFT)):
        swim = out["swim_m"][0, i]
        v_w = swim / C.MOVE_S                                   # the speed actually swum (capped by aerobic power)
        assert float(v_w) <= float(out["speed_m_s"][0, i]) * (1 + 1e-6)
        assert float(out["cost_j"][0, i]) == pytest.approx(float(drag_k * v_w ** 2 * swim / (0.25 * eta)), rel=1e-4)
    assert float(out["cost_j"][0, 1] / out["swim_m"][0, 1]) < float(out["cost_j"][0, 0] / out["swim_m"][0, 0])
    # a swimmer held back by its aerobic ceiling works at exactly that ceiling: (10 a - 1) x basal
    cr.pos = torch.tensor([1.0, 0.0, 0.0]).expand(1, 2, 3).clone()
    full = C.move(cr, glb, sea, torch.zeros(1, 2), torch.ones(1, 2), gravity_m_s2=9.81, p_o2_pa=EARTH_O2, radius_m=R)
    v_w = full["swim_m"][0, 0] / C.MOVE_S
    assert float(v_w) < float(full["speed_m_s"][0, 0])                         # capped
    seconds = float(full["swim_m"][0, 0] / v_w)                               # time actually swum
    ceiling = (C.AEROBIC_SCOPE * float(C.aerobic(torch.tensor(EARTH_O2))) - 1) * float(C.basal_w(torch.tensor(30.0)))
    assert float(full["cost_j"][0, 0]) / seconds == pytest.approx(ceiling, rel=1e-4)
    cr.water_kg[:] = 15.0
    got = C.drink(cr, cr.alive, glb, torch.zeros(1, glb.C), sea["land"], radius_m=R)
    assert float(got["from_sea"][0, 0]) == 0.0 and bool(got["salt"][0, 0])
    assert float(got["from_sea"][0, 1]) > 0.0 and float(cr.water_kg[0, 1]) > 15.0


# ------------------------------------------------------------------------------------ ledgers
def test_energy_carbon_and_water_ledgers_close_through_a_busy_run(glb):
    W, N = 2, 48
    gen = torch.Generator().manual_seed(11)
    ter = G.make_terrain(glb, W, gen, 600.0, 2e10, R)
    cr = C.spawn_founders(chain.synthetic_inputs(8), glb, ter, gen, capacity=N, founders=16, hidden=8,
                          initial_hidden=4, adult_mass_kg=1.0)
    cr.genome["diet"][:, ::2] = 0.9
    cr.cooldown_d[:] = 0.0
    env = {"t_air_k": torch.full((W, glb.C), 290.0), "p_o2_pa": torch.full((W,), EARTH_O2)}
    plants = torch.full((W, glb.C), 50.0)
    soil = torch.full((W, glb.C), 80.0)
    pool = items.new_item_pool(W, 256)
    harvest = C.field_harvester(plants)
    meat_j = 0.0
    for day in range(120):
        turn = torch.rand(W, N, generator=gen) * 2 - 1
        speed = torch.rand(W, N, generator=gen) * 0.05
        travel = C.move(cr, glb, ter, turn, speed, gravity_m_s2=9.0, p_o2_pa=EARTH_O2, radius_m=R)
        meat = C.eat_meat(cr, cr.alive & (cr.genome["diet"] > 0.5), pool, radius_m=R, reach_m=4000.0)
        meat_j += float(meat["energy_j"].sum())
        C.eat_plants(cr, cr.alive & (cr.genome["diet"] < 0.5), glb, harvest)
        C.drink(cr, cr.alive, glb, soil, ter["land"], radius_m=R)
        cr.loud = torch.rand(W, N, generator=gen) * cr.alive
        C.physiology(cr, glb, env, travel)
        if day % 10 == 5:                                   # predators of a kind: strike a few
            w, n = cr.alive.nonzero(as_tuple=True)
            pick = torch.arange(0, w.numel(), 7)
            C.injure(cr, w[pick], n[pick], 1.5)
        out = C.deaths(cr, pool)
        assert bool((out["litter_c"][out["dead"]] > 0).all())
        C.reproduce(cr, gen)
    L = cr.ledger
    assert float(L["births"].sum()) > 0 and float(L["deaths_injury"].sum()) > 0 and meat_j > 0
    err = C.ledger_errors(cr)
    for k in ("energy", "carbon", "water"):
        rel = (err[k].abs() / err[f"{k}_scale"]).max()
        assert float(rel) < 2e-6, (k, err[k].tolist(), err[f"{k}_scale"].tolist())


# ------------------------------------------------------------------------------------ reproduction
def test_reproduction_makes_mutated_offspring_whose_live_weights_are_their_genome(glb):
    W, N = 1, 12
    gen = torch.Generator().manual_seed(3)
    ter = flat_terrain(glb, W)
    cr = C.spawn_founders(chain.synthetic_inputs(2), glb, ter, gen, capacity=N, founders=2, hidden=16,
                          initial_hidden=6, adult_mass_kg=10.0)
    cr.genome["litter"][0, :2] = torch.tensor([3, 1])
    cr.reserve_j[0, :2] = C.reserve_max(cr.mass_kg[0, :2])
    cr.cooldown_d[:] = 0.0
    cr.wh_live[0, :2] += 0.5                                   # learned weights must not be inherited
    rebase(cr)
    before = cr.reserve_j[0, :2].clone()
    out = C.reproduce(cr, gen)
    assert out["born"].tolist() == [4]
    kids = out["child_n"]
    assert sorted(kids.tolist()) == [2, 3, 4, 5]
    g = cr.genome
    assert torch.equal(cr.wh_live[0, kids], g["Wh"][0, kids])
    assert float(cr.brain_state[0, kids].abs().max()) == 0.0
    parents = (cr.parent[0, kids, None] == cr.uid[0, None, :2]).long().argmax(-1)      # the parents' slots
    assert sorted(parents.tolist()) == [0, 0, 0, 1]
    assert cr.parent[0, kids].tolist() == cr.uid[0, parents].tolist()
    assert sorted(cr.uid[0, kids].tolist()) == [2, 3, 4, 5] and cr.next_uid.tolist() == [6]
    assert cr.generation[0, kids].tolist() == [1, 1, 1, 1]
    assert cr.founder[0, kids].tolist() == cr.founder[0, parents].tolist()
    for name in ("Wx", "Wh", "Wo"):
        assert not torch.equal(g[name][0, kids], g[name][0, parents])
        assert float((g[name][0, kids] - g[name][0, parents]).abs().max()) < 1.0      # small steps
    for name, spec in C.BODY_SPECS.items():
        v = g[name][0, kids].float()
        assert bool(((v >= spec.lo) & (v <= spec.hi)).all()), name
    m0 = C.litter_mass_kg(torch.tensor(10.0)) / torch.tensor([3.0, 1.0])[parents]
    assert torch.allclose(cr.mass_kg[0, kids], m0)
    cost = m0 * C.GROWTH_J_KG + C.CHILD_RESERVE * C.reserve_max(m0)
    paid = before - cr.reserve_j[0, :2]
    assert torch.allclose(paid, torch.zeros(2).index_add(0, parents, cost), rtol=1e-5)
    assert torch.allclose(cr.cooldown_d[0, :2], C.gestation_d(torch.tensor(10.0)))
    assert bool(cr.alive[0, kids].all()) and torch.allclose(cr.water_kg[0, kids], C.water_norm(m0))
    # nobody is ready again until the cooldown has passed
    assert C.reproduce(cr, gen)["born"].tolist() == [0]
    err = C.ledger_errors(cr)
    assert abs(float(err["energy"][0])) <= 1e-6 * float(err["energy_scale"][0])
    assert abs(float(err["carbon"][0])) <= 1e-6 * float(err["carbon_scale"][0])


def test_reproduction_is_seeded_and_stops_at_capacity(glb):
    def run(seed):
        gen = torch.Generator().manual_seed(seed)
        ter = flat_terrain(glb, 1)
        cr = C.spawn_founders(chain.synthetic_inputs(2), glb, ter, gen, capacity=5, founders=2, hidden=8,
                              initial_hidden=4, adult_mass_kg=5.0)
        cr.genome["litter"][:] = 6
        cr.reserve_j[:] = C.reserve_max(cr.mass_kg)
        cr.cooldown_d[:] = 0.0
        out = C.reproduce(cr, gen)
        return cr, out
    a, oa = run(1)
    b, _ = run(1)
    assert C.state_hash(a) == C.state_hash(b)
    assert oa["born"].tolist() == [3] and bool(a.alive.all())


def test_at_capacity_the_parents_are_served_in_a_random_order(glb):
    """One free slot, two ready parents: over seeds both get it (not always the lower slot)."""
    winners = set()
    for seed in range(12):
        gen = torch.Generator().manual_seed(seed)
        cr = C.spawn_founders(chain.synthetic_inputs(2), glb, flat_terrain(glb, 1), gen, capacity=3, founders=2,
                              hidden=8, initial_hidden=4, adult_mass_kg=5.0)
        cr.genome["litter"][:] = 1
        cr.reserve_j[:] = C.reserve_max(cr.mass_kg)
        cr.cooldown_d[:] = 0.0
        out = C.reproduce(cr, gen)
        assert out["born"].tolist() == [1]
        winners.add(int(cr.parent[0, 2]))
        bred = cr.cooldown_d[0, :2] > 0
        assert int(bred.sum()) == 1 and int(cr.uid[0, :2][bred]) == int(cr.parent[0, 2])   # only the winner paid
    assert winners == {0, 1}


def test_a_parent_breeds_only_if_it_can_spare_the_litters_water(glb):
    gen = torch.Generator().manual_seed(5)
    cr = C.spawn_founders(chain.synthetic_inputs(2), glb, flat_terrain(glb, 1), gen, capacity=8, founders=2,
                          hidden=8, initial_hidden=4, adult_mass_kg=5.0)
    cr.genome["litter"][:] = 2
    cr.reserve_j[:] = C.reserve_max(cr.mass_kg)
    cr.cooldown_d[:] = 0.0
    norm = C.water_norm(torch.tensor(5.0))
    litter_w = float(C.water_norm(C.litter_mass_kg(torch.tensor(5.0))))
    cr.water_kg[0, :2] = torch.tensor([float(norm), C.DEHYDRATION_LETHAL * float(norm) + 0.5 * litter_w])
    out = C.reproduce(cr, gen)
    assert out["born"].tolist() == [2] and cr.parent[0, out["child_n"]].tolist() == [int(cr.uid[0, 0])] * 2
    assert float(cr.water_kg[0, 0]) >= C.DEHYDRATION_LETHAL * float(norm)
    assert float(cr.cooldown_d[0, 1]) == 0.0


# ------------------------------------------------------------------------------------ corpses and meat
def test_a_body_becomes_meat_bone_hide_and_fat_items(glb):
    cr, ter, _ = one_animal(glb, 2.0, founders=1, capacity=2)
    pool = items.new_item_pool(1, 16)
    held = items.spawn(pool, torch.tensor([0]), materials.species_vector({"flint": 1.0})[None], torch.tensor([0.3]),
                       torch.tensor([[1.0, 0.0, 0.0]]), 290.0, items.holder_code(torch.tensor([0]), 0, cr.K))
    cr.inv[0, 0, 0] = held[0]
    cr.reserve_j[0, 0] = 1e7
    cr.water_kg[0, 0] = 1.3
    body_c = float(C.body_carbon_kg(cr.mass_kg[0, 0], cr.reserve_j[0, 0]))
    body_j = float(C.body_energy_j(cr.mass_kg[0, 0], cr.reserve_j[0, 0]))
    C.injure(cr, torch.tensor([0]), torch.tensor([0]), 2.0)
    pos = cr.pos[0, 0].clone()
    out = C.deaths(cr, pool)
    assert bool(out["dead"][0, 0]) and C.CAUSES[int(out["cause"][0, 0])] == "injury"
    assert not bool(cr.alive[0, 0])
    sp = items.species_mass(pool)[0]
    for s, share in C.CORPSE.items():
        assert float(sp[materials.IDX[s]]) == pytest.approx(2.0 * share, rel=1e-6)
    assert float(sp[materials.IDX["fat"]]) == pytest.approx(1e7 / C.FAT_J_KG, rel=1e-6)
    corpse = pool.alive & (pool.comp[..., materials.IDX["flint"]] == 0)
    assert int(corpse.sum()) == 4
    assert torch.allclose(pool.pos[corpse], pos.expand(4, 3))
    assert bool((pool.holder[corpse] == -1).all()) and torch.allclose(pool.temp_k[corpse], torch.tensor(C.T_BODY))
    # the held flint falls where the body lies
    assert int(pool.holder[0, held[0]]) == -1 and torch.allclose(pool.pos[0, held[0]], pos)
    # carbon, energy and water: items plus litter (and released water) are the body
    item_c = float(materials.element_mass(out["items_species_kg"])[0, materials.ELEMENTS.index("C")])
    assert item_c + float(out["litter_c"][0, 0]) == pytest.approx(body_c, rel=1e-6)
    assert float(out["item_j"][0, 0]) + float(out["litter_j"][0, 0]) == pytest.approx(body_j, rel=1e-6)
    item_w = sum(float(sp[materials.IDX[s]]) * dict(materials.MAKEUP[s]).get("H2O", 0.0) for s in C.CORPSE)
    assert item_w + float(out["water_kg"][0, 0]) == pytest.approx(1.3, rel=1e-6)
    assert float(out["litter_c"][0, 0]) > 0
    assert torch.allclose(out["items_species_kg"][0], items.species_mass(pool)[0]
                          - materials.species_vector({"flint": 0.3}, dtype=torch.float64), atol=1e-6)


def test_meat_eaters_digest_corpses_and_cooking_helps_up_to_the_gross_energy(glb):
    cr, ter, _ = one_animal(glb, 5.0, founders=3, capacity=3)
    pool = items.new_item_pool(1, 16)
    cr.pos[0, 1:] = cr.pos[0, 0]
    cr.genome["diet"][0] = torch.tensor([0.0, 1.0, 1.0])
    cr.reserve_j[0] = torch.tensor([0.0, 0.0, 0.0])             # a lean prey (no fat item), hungry eaters
    cr.health[0, 0] = 0.0
    rebase(cr)
    C.deaths(cr, pool)
    meat = int((pool.alive & (pool.comp[..., materials.IDX["meat"]] == 1)).nonzero()[0, 1])
    food = float(materials.FOOD_J_KG["meat"])
    left = float(pool.mass[0, meat])
    out = C.eat_meat(cr, cr.alive & (torch.arange(3) == 1), pool, radius_m=R)
    assert int(out["item"][0, 1]) == meat
    eaten = left - float(pool.mass[0, meat])
    dry_cap = C.PLANT_INTAKE * 5.0 ** 0.75 / (1 - 0.735)            # the dry-matter cap in wet meat
    assert eaten == pytest.approx(dry_cap, rel=1e-4)
    assert float(out["energy_j"][0, 1]) == pytest.approx(eaten * food * 0.9, rel=1e-4)
    assert float(out["water_kg"][0, 1]) == pytest.approx(eaten * 0.735, rel=1e-4)
    # cooked meat: digestible share 0.9 x 1.3 is capped at the gross energy
    pool.peak_k[0, meat] = 350.0
    left = float(pool.mass[0, meat])
    out2 = C.eat_meat(cr, cr.alive & (torch.arange(3) == 2), pool, radius_m=R)
    assert int(out2["item"][0, 2]) == meat
    eaten2 = left - float(pool.mass[0, meat])
    assert float(out2["energy_j"][0, 2]) == pytest.approx(eaten2 * food, rel=1e-4)
    eaten_c = float(materials.element_mass(out["eaten_species_kg"] + out2["eaten_species_kg"])[0, 1])
    assert eaten_c == pytest.approx(float(out["food_c"].sum() + out2["food_c"].sum()), rel=1e-5)
    err = C.ledger_errors(cr)
    for k in ("energy", "carbon", "water"):
        assert abs(float(err[k][0])) <= 1e-6 * float(err[f"{k}_scale"][0])
    # eat_meat rebinds the state like eat_plants: a snapshot taken by reference does not change
    r0, w0 = cr.reserve_j, cr.water_kg
    r0_values = r0.clone()
    pool.peak_k[0, meat] = 290.0
    assert float(C.eat_meat(cr, cr.alive & (torch.arange(3) == 1), pool, radius_m=R)["energy_j"][0, 1]) > 0
    assert torch.equal(r0, r0_values) and cr.reserve_j is not r0 and cr.water_kg is not w0
    # an eater too far from any food item finds nothing
    cr.pos[0, 1] = -cr.pos[0, 1]
    assert int(C.eat_meat(cr, cr.alive & (torch.arange(3) == 1), pool, radius_m=R)["item"][0, 1]) == -1


def test_the_best_item_is_chosen_among_many_and_far_ones_are_ignored(glb):
    """The candidate search (reach mask, then digestible energy of the candidates only) picks, per
    eater, the item with the most digestible energy within reach, the held one included."""
    cr, ter, _ = one_animal(glb, 5.0, founders=3, capacity=3)
    pool = items.new_item_pool(1, 64)
    base = cr.pos[0, 0].clone()
    cr.pos[0, 1:] = base
    cr.genome["diet"][0] = torch.tensor([0.0, 1.0, 1.0])
    near, _ = G.move(base, torch.tensor(1.0), torch.tensor(2.0), R)
    far, _ = G.move(base, torch.tensor(1.0), torch.tensor(50.0), R)
    meat = materials.species_vector({"meat": 1.0})[None]

    def mk(kg, p, holder=-1):
        return int(items.spawn(pool, torch.tensor([0]), meat, torch.tensor([kg]), p[None], 290.0, holder)[0])
    small, big_far = mk(0.5, near), mk(30.0, far)
    middling = mk(1.5, near)
    held = mk(1.0, base, items.holder_code(torch.tensor([2]), 0, cr.K))
    cr.inv[0, 2, 0] = held
    pick = C.eat_meat(cr, cr.alive & (torch.arange(3) > 0), pool, radius_m=R)["item"][0].tolist()
    assert pick[1] == middling and pick[2] == middling and big_far not in pick
    # with the middling one gone, the held item (1 kg) beats the 0.5 kg one for its holder only
    items.remove(pool, torch.tensor([0]), torch.tensor([middling]))
    pick = C.eat_meat(cr, cr.alive & (torch.arange(3) > 0), pool, radius_m=R)["item"][0].tolist()
    assert pick[2] == held and pick[1] == small


def test_plant_harvest_books_what_the_biosphere_gave(glb):
    """biosphere.harvest hands each request its share of what left the float32 pool, sometimes more
    than asked; the eaters book exactly that, so the biosphere's c_harvested equals their food_c."""
    W, N = 2, 256
    gen = torch.Generator().manual_seed(13)
    ter = flat_terrain(glb, W)
    cr = C.spawn_founders(chain.synthetic_inputs(3), glb, ter, gen, capacity=N, founders=N, hidden=4,
                          initial_hidden=2, adult_mass_kg=10.0)
    cr.reserve_j[:] = 0.0
    rebase(cr)
    z = torch.zeros(W, glb.C)
    z64 = torch.zeros(W, dtype=torch.float64)
    bio = bs.BioState(plant_c=0.3 + torch.rand(W, glb.C, generator=gen), wood_c=z.clone(), litter_c=z.clone(),
                      nutrient=z.clone(), phyto_c=z.clone(), cell_m2=glb.area64 * R ** 2, area64=glb.area64,
                      carbon0=z64.clone(), oxygen0=z64.clone(), c_harvested=z64.clone(), c_wood_collected=z64.clone(),
                      c_buried=z64.clone(), c_litter_added=z64.clone())
    asked = []

    def harvest(w, cell, kg):
        asked.append(kg.clone())
        return bs.harvest(bio, w, cell, kg)
    out = C.eat_plants(cr, cr.alive, glb, harvest)
    assert C.PLANT_C_FRACTION == bs.BioRules.plant_carbon_fraction
    taken_c = bio.c_harvested * float(bio.cell_m2.sum())        # global-mean kg C/m^2 x the habitat's area
    food_c = out["food_c"].double().sum(1)
    assert torch.allclose(food_c, taken_c, rtol=1e-5)
    assert torch.allclose(cr.ledger["food_c"], taken_c, rtol=1e-5)
    eaten = out["kg_dry"][cr.alive]
    assert bool((eaten > asked[0]).any())                      # the float32 pool rounds; booked as given
    err = C.ledger_errors(cr)
    for k in ("energy", "carbon"):
        assert float((err[k].abs() / err[f"{k}_scale"]).max()) < 1e-6


def test_drinkers_share_the_soil_water_above_the_limit(glb):
    """Thirsty drinkers on one cell share the water above 50 kg/m^2 in proportion to their need; the
    soil never goes below the limit and soil_taken x area is what was drunk."""
    W = 1
    small_r = 100.0                                            # cells of about 580 m^2: water runs short
    cr, ter, _ = one_animal(glb, 30.0, founders=64, capacity=64)
    cr.pos[:] = cr.pos[0, 0]
    cr.water_kg[0] = torch.linspace(0.0, 15.0, 64)
    soil = torch.full((W, glb.C), 51.0)
    cell = glb.cell_of(cr.pos[0, 0])
    need = C.water_norm(cr.mass_kg[0]) + C.WATER_TURNOVER * 30 ** C.WATER_EXP - cr.water_kg[0]
    area = float(glb.area[cell]) * small_r ** 2
    assert float(need.sum()) > 1.0 * area                     # more need than water above the limit
    out = C.drink(cr, cr.alive, glb, soil, ter["land"], radius_m=small_r)
    drunk = out["from_soil"][0]
    assert float(drunk.sum()) == pytest.approx(1.0 * area, rel=1e-5)
    assert float(out["soil_taken_kg_m2"][0, cell]) * area == pytest.approx(float(drunk.sum()), rel=1e-5)
    assert float(soil[0, cell] - out["soil_taken_kg_m2"][0, cell]) >= C.DRINK_SOIL_MIN - 1e-4
    assert torch.allclose(drunk / need, (drunk / need)[0].expand(64), rtol=1e-4)        # pro rata
    other = torch.ones(glb.C, dtype=torch.bool)
    other[cell] = False
    assert float(out["soil_taken_kg_m2"][0, other].abs().max()) == 0.0


def test_placing_into_a_used_slot_forgets_what_its_occupant_learned(glb):
    cr, ter, _ = one_animal(glb, 2.0, founders=1, capacity=2)
    cr.wh_live[0, 0] += 0.7
    cr.brain_state[0, 0] = 0.3
    cr.baseline[0, 0] = 0.2
    C.injure(cr, torch.tensor([0]), torch.tensor([0]), 2.0)
    C.deaths(cr, items.new_item_pool(1, 16))
    m = torch.tensor(2.0)
    C.place(cr, torch.tensor([0]), torch.tensor([0]), pos=cr.pos[0, 0][None].clone(), heading=0.0, mass=m,
            reserve=C.reserve_max(m) / 2, water=C.water_norm(m), age=1.0, cooldown=0.0)
    assert torch.equal(cr.wh_live[0, 0], cr.genome["Wh"][0, 0])
    assert float(cr.brain_state[0, 0].abs().max()) == 0.0 and float(cr.baseline[0, 0]) == 0.0
    assert float(cr.struck[0, 0]) == 0.0 and int(cr.harm[0, 0]) == 0 and float(cr.health[0, 0]) == 1.0


def test_plant_harvest_is_shared_and_never_negative(glb):
    stock = torch.tensor([[3.0, 0.0, 5.0]])
    w = torch.tensor([0, 0, 0, 0])
    cell = torch.tensor([0, 0, 1, 2])
    req = torch.tensor([2.0, 4.0, 1.0, 1.0])
    got = C.share_scatter(stock, w, cell, req)
    assert torch.allclose(got, torch.tensor([1.0, 2.0, 0.0, 1.0]))
    assert torch.allclose(stock, torch.tensor([[0.0, 0.0, 4.0]]))


# ------------------------------------------------------------------------------------ integration round (world review)
def test_a_dried_body_leaves_parts_no_wetter_than_itself(glb):
    """A body below the parts' makeup water (0.57 of the norm) used to hand out negative water; now its meat,
    bone and hide are scaled to what its water fills, the rest of their dry matter goes to litter."""
    cr, ter, _ = one_animal(glb, 30.0, capacity=4, founders=2)
    pool = items.new_item_pool(1, 32)
    a = cr.alive
    cr.water_kg[a] = torch.tensor([0.5, 0.79]) * C.water_norm(cr.mass_kg[a])
    rebase(cr)
    out = C.deaths(cr, pool)
    assert bool(out["dead"][a].all())
    assert float(out["water_kg"].min()) >= 0.0
    water = torch.tensor([dict(materials.MAKEUP[s]).get("H2O", 0.0) for s in materials.SPECIES])
    item_w = float((items.species_mass(pool)[0].float() * water).sum())
    dead_w = float(cr.ledger["dead_w"][0])
    assert item_w <= dead_w + 1e-4
    assert float(out["water_kg"][0].sum()) + item_w == pytest.approx(dead_w, rel=1e-5)
    e = C.ledger_errors(cr)
    for k in ("energy", "carbon", "water"):
        assert abs(float(e[k][0])) <= 1e-5 * float(e[f"{k}_scale"][0]), k


def test_respiration_makes_metabolic_water_and_synthesis_takes_no_oxygen(glb):
    """O2 is the reserve energy oxidised / 4.5e5 J/mol; the O2 not returned as CO2 becomes body water; growing
    tissue (a juvenile) adds heat to the energy ledger but no O2."""
    cr, ter, _ = one_animal(glb, 30.0, capacity=4, founders=2)
    a = cr.alive
    cr.mass_kg[0, 1] = 10.0                       # a juvenile of a 30 kg adult grows today
    cr.frame_kg[0, 1] = 10.0
    cr.reserve_j[0, 1] = 0.9 * C.reserve_max(torch.tensor(10.0))
    cr.water_kg[a] = C.water_norm(cr.mass_kg[a])
    rebase(cr)
    out = C.physiology(cr, glb, env_of(glb, 1, 295.0))
    spent = sum(out[k] for k in ("basal_j", "brain_j", "sense_j", "thermo_j", "call_j", "travel_j"))
    assert torch.allclose(out["o2_mol"][a], (spent / C.OXY_J_MOL)[a], rtol=1e-6)
    assert float(out["growth_overhead_j"][0, 1]) > 0
    ox_c = spent * C.FAT_C_PER_J
    met = 2 * (spent / C.OXY_J_MOL - ox_c / C.M_C) * C.M_H2O
    assert torch.allclose(out["metabolic_w"][a], met[a], rtol=1e-5) and float(met[0, 0]) > 0.05
    # O atoms: 2 O2 = 2 CO2 of the fat oxidised + metabolic water, booked in the ledger
    L = cr.ledger
    o_in = 2 * float(L["o2_mol"][0])
    o_out = 2 * float(L["oxidised_c"][0]) / C.M_C + float(L["metabolic_w"][0]) / C.M_H2O
    assert o_in == pytest.approx(o_out, rel=1e-5)
    e = C.ledger_errors(cr)
    assert abs(float(e["water"][0])) <= 1e-6 * float(e["water_scale"][0])
    # reproduction's synthesis heat takes no O2
    gen = torch.Generator().manual_seed(1)
    cr.age_d[0, 0] = 1e4
    cr.cooldown_d[0, 0] = 0.0
    cr.reserve_j[0, 0] = C.reserve_max(cr.mass_kg[0, 0])
    rp = C.reproduce(cr, gen)
    assert int(rp["born"].sum()) > 0 and float(rp["o2_mol"].abs().max()) == 0.0


def test_water_above_the_norm_is_excreted(glb):
    cr, ter, _ = one_animal(glb, 30.0)
    a = cr.alive
    norm = C.water_norm(cr.mass_kg[a])
    cr.water_kg[a] = 2.0 * norm
    rebase(cr)
    out = C.physiology(cr, glb, env_of(glb, 1, 295.0))
    assert float(cr.water_kg[a]) == pytest.approx(float(norm), rel=1e-6)
    assert float(out["excreted_kg"][a]) > 0.9 * float(norm)
    e = C.ledger_errors(cr)
    assert abs(float(e["water"][0])) <= 1e-6 * float(e["water_scale"][0])


def test_grazing_takes_a_share_of_the_day_and_the_forages_water(glb):
    cr, ter, _ = one_animal(glb, 30.0, capacity=4, founders=2)
    a = cr.alive
    cr.reserve_j[a] = 0.0
    rebase(cr)
    plants = torch.full((1, glb.C), 1e9)
    asked = []

    def water(w, c, kg):
        asked.append(kg.clone())
        return kg * 0.5                          # the soil gives half
    share = torch.tensor([[1.0, 0.25, 0.0, 0.0]])
    out = C.eat_plants(cr, a, glb, C.field_harvester(plants), share=share, water=water)
    full = C.PLANT_INTAKE * 30.0 ** 0.75
    assert float(out["kg_dry"][0, 0]) == pytest.approx(full, rel=1e-5)
    assert float(out["kg_dry"][0, 1]) == pytest.approx(0.25 * full, rel=1e-5)
    assert float(asked[0][0]) == pytest.approx(C.FORAGE_WATER_KG * full, rel=1e-5)
    assert torch.allclose(out["water_kg"][a], C.FORAGE_WATER_KG * 0.5 * out["kg_dry"][a], rtol=1e-6)
    assert float(cr.ledger["food_w"][0]) == pytest.approx(float(out["water_kg"].sum()), rel=1e-6)
    e = C.ledger_errors(cr)
    assert abs(float(e["water"][0])) <= 1e-6 * float(e["water_scale"][0])


def test_founders_follow_a_habitat_weight(glb):
    gen = torch.Generator().manual_seed(4)
    ter = flat_terrain(glb, 2)
    weight = torch.zeros(2, glb.C)
    weight[0, 5] = 1.0
    weight[0, 9] = 1.0                            # world 1 has none: the land rule
    cr = C.spawn_founders(chain.synthetic_inputs(3), glb, ter, gen, capacity=40, founders=40, hidden=8,
                          initial_hidden=4, cell_weight=weight)
    cells = glb.cell_of(cr.pos)
    assert set(cells[0].tolist()) <= {5, 9}
    assert len(set(cells[1].tolist())) > 10


def test_travel_time_and_the_shore(glb):
    """``seconds`` sets the day's travel time; with ``shore_stop`` a walker stops at the last land sample."""
    gen = torch.Generator().manual_seed(2)
    ter = flat_terrain(glb, 1)
    cr = C.spawn_founders(chain.synthetic_inputs(3), glb, ter, gen, capacity=2, founders=2, hidden=8,
                          initial_hidden=4, adult_mass_kg=30.0)
    cr.genome["swim"][:] = 0.0
    cell0 = glb.cell_of(cr.pos)
    p0 = cr.pos.clone()
    out = C.move(cr, glb, ter, torch.zeros(1, 2), torch.ones(1, 2), gravity_m_s2=9.81, p_o2_pa=21200.0, radius_m=R,
                 seconds=600.0)
    v = C.V_TROT * 30.0 ** C.V_TROT_EXP * C.aerobic(torch.tensor(21200.0))
    assert torch.allclose(out["dist_m"], (v * 600.0).expand(1, 2), rtol=1e-5)
    # the sea everywhere but the start cells: a walker stops where it is, a swimmer swims on
    sea = dict(ter)
    sea["land"] = torch.zeros(1, glb.C, dtype=torch.bool)
    sea["land"][0, cell0[0]] = True
    cr.pos = p0.clone()
    cr.genome["swim"][0, 1] = 0.9
    out = C.move(cr, glb, sea, torch.zeros(1, 2), torch.ones(1, 2), gravity_m_s2=9.81, p_o2_pa=21200.0,
                 radius_m=R, shore_stop=True, samples=64)
    assert float(out["swim_m"][0, 0]) == 0.0 and float(out["dist_m"][0, 0]) < float(out["dist_m"][0, 1])
    assert bool(sea["land"][0].gather(0, glb.cell_of(cr.pos[0, :1])).all())
    assert float(out["swim_m"][0, 1]) > 0
