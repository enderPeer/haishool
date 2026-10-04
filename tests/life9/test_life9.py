import pytest

torch = pytest.importorskip("torch")

from haishool.life9 import brain, probe, terrain  # noqa: E402
from haishool.life9.config import Config9  # noqa: E402
from haishool.life9.world import World9  # noqa: E402

SMALL = Config9(worlds=3, capacity=40, founders=20, grid=32, size=32.0)


def run(cfg, seed, ticks):
    world = World9(cfg, seed)
    for _ in range(ticks):
        world.step()
    return world


def same(a, b):
    for name, value in a.state_dict()["tensors"].items():
        assert torch.equal(value, b.state_dict()["tensors"][name]), name
    for name in a.genome:
        assert torch.equal(a.genome[name], b.genome[name]), name


def test_terrain_is_periodic_and_in_range():
    gen = torch.Generator().manual_seed(1)
    ground = terrain.make_terrain(gen, 2, 32, 6.0, 4, "cpu")
    assert float(ground.min()) >= 0 and float(ground.max()) <= 6.0 + 1e-5
    points = torch.tensor([[[3.3, 7.1], [3.3 + 32, 7.1 - 64]]]).repeat(2, 1, 1)
    h = terrain.height_at(ground, points, 32.0)
    assert torch.allclose(h[:, 0], h[:, 1], atol=1e-5)


def test_a_ridge_blocks_sight_and_is_symmetric():
    ground = torch.zeros(1, 32, 32)
    ground[0, 15:17, :] = 5.0                      # a wall across x = 15..16 cells
    a = torch.tensor([[[10.0, 10.0]]])
    delta = torch.tensor([[[[12.0, 0.0], [0.0, 6.0]]]])   # across the wall / along it
    seen = terrain.line_of_sight(ground, 32.0, a, torch.tensor([[1.0]]), delta, torch.tensor([[[1.0, 1.0]]]), 8)
    assert seen.tolist() == [[[False, True]]]
    back = terrain.line_of_sight(ground, 32.0, a + delta[:, :, 0], torch.tensor([[1.0]]), -delta[:, :, :1],
                                 torch.tensor([[[1.0]]]), 8)
    assert back.tolist() == [[[False]]]


def test_same_seed_same_world_and_exact_resume():
    straight = run(SMALL, 5, 40)
    same(straight, run(SMALL, 5, 40))
    half = run(SMALL, 5, 20)
    resumed = World9.from_state(half.state_dict())
    for _ in range(20):
        resumed.step()
    same(straight, resumed)


def test_energy_ledger_closes():
    world = run(SMALL, 2, 120)
    assert world.stats["births"].sum() > 0
    assert float(world.ledger_error().abs().max()) < 0.05


def test_unused_brain_units_stay_silent_and_size_is_paid():
    world = run(SMALL, 3, 30)
    mask = brain.unit_mask(world.genome["k"], SMALL.hidden)
    assert float((world.state * (1 - mask)).abs().max()) == 0.0
    small = World9(SMALL.but(initial_hidden=4), 9)
    large = World9(SMALL.but(initial_hidden=24), 9)
    small.step(), large.step()
    assert float(large.ledger["metabolism"].sum()) > float(small.ledger["metabolism"].sum())


def test_offspring_start_from_the_genome_not_from_learned_weights():
    world = World9(SMALL.but(reproduction_threshold=31.0, maturity=0, birth_cooldown=200), 4)
    world.step()
    born = world.generation >= 1
    assert born.any()
    assert torch.equal(world.wh_live[born], world.genome["Wh"][born])
    for _ in range(5):
        world.step()
    founders = world.alive & (world.generation == 0)
    assert not torch.equal(world.wh_live[founders], world.genome["Wh"][founders])   # plastic within a life


def test_deaf_channel_hears_nothing_and_oracle_calls_only_on_sight():
    deaf = World9(SMALL.but(channel="deaf"), 6)
    deaf.calls.fill_(1.0)
    assert float(deaf.sense()["heard"].max()) == 0.0
    oracle = World9(SMALL.but(oracle=True), 6)
    seen = oracle.sense()["sees_predator"]
    oracle.step()
    calling = oracle.calls.norm(dim=-1) > 0
    assert not (calling & ~seen).any()


def test_arms_share_terrain_and_founders():
    a, b = World9(SMALL, 7), World9(SMALL.but(channel="scrambled", oracle=True), 7)
    assert torch.equal(a.terrain, b.terrain) and torch.equal(a.pos, b.pos)
    assert torch.equal(a.genome["Wx"], b.genome["Wx"])


def test_sign_test():
    assert probe.sign_test(0, 0) == 1.0
    assert probe.sign_test(8, 0) == pytest.approx(2 / 256)
    assert probe.sign_test(4, 4) == 1.0


def test_frame_for_the_viewer():
    world = run(SMALL, 8, 3)
    frame = world.frame(0)
    assert len(frame["agents"]) == int(world.alive[0].sum())
    assert len(frame["agents"][0]) == 10 and len(frame["predators"]) == SMALL.predators
