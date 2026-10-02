"""The spatial grid gives the same state hash as the full scan over 300 ticks for 3 seeds."""
import pytest

from haishool.life8.checkpoint import digest
from haishool.life8.config import Config
from haishool.life8.world import World


def run_hash(seed, ticks, grid, log="compact"):
    world = World.create(seed=seed, config=Config(log=log))
    world.spatial_index = grid
    for _ in range(ticks):
        world.step()
    return digest(world.to_dict()), world


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_grid_and_full_scan_give_the_same_state_hash_over_300_ticks(seed):
    with_grid, world = run_hash(seed, 300, True)
    scanned, _ = run_hash(seed, 300, False)
    assert with_grid == scanned
    assert world.tick == 300 and len(world.agents) > 60 and world.counters["signals_received"] > 1000


def test_grid_matches_scan_in_full_log_mode():
    assert run_hash(4, 80, True, "full")[0] == run_hash(4, 80, False, "full")[0]
