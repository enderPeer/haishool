"""brain3 at patch scale: 8,192 bodies (one patch's capacity) think, learn and mutate on the CPU."""
import time

import pytest

torch = pytest.importorskip("torch")

from haishool.life9.v3 import brain3 as b3  # noqa: E402

N = 8192
IN_DIM = 160            # about what senses3 gives: 8 sectors x (vision + hearing D + noise) + smell + touch + intero


def timed(fn, reps=3):
    fn()
    t = time.perf_counter()
    for _ in range(reps):
        r = fn()
    return (time.perf_counter() - t) / reps, r


def test_think_8192_bodies_on_cpu():
    g = torch.Generator().manual_seed(0)
    genome = b3.random_genome3(g, (N,), IN_DIM)                       # hidden 128, founders k in [2, 32]
    wh = b3.init_live(genome)
    state = b3.init_state((N,), b3.HIDDEN, "cpu")
    x = torch.rand(N, IN_DIM, generator=g)
    intero = torch.rand(N, b3.N_INTERO, generator=g)
    alive = torch.ones(N, dtype=torch.bool)
    t_think, (out, new) = timed(lambda: b3.think(genome, wh, state, x))
    t_learn, _ = timed(lambda: b3.hebbian(wh, genome, state, new, intero, alive))
    t_mut, _ = timed(lambda: b3.mutate3(g, genome), reps=1)
    genome["k"].fill_(b3.HIDDEN)                                      # the largest brains
    t_full, (out_full, _) = timed(lambda: b3.think(genome, wh, state, x))
    print(f"\nbrain3, {N} bodies, in {IN_DIM}, hidden {b3.HIDDEN}, {torch.get_num_threads()} threads: "
          f"think (founders, k <= 32) {t_think * 1e3:.1f} ms, hebbian {t_learn * 1e3:.1f} ms, "
          f"mutate3 (all {N}) {t_mut * 1e3:.0f} ms, think at k = 128 {t_full * 1e3:.1f} ms")
    assert out.shape == (N, b3.OUT_DIM) and bool(torch.isfinite(out).all())
    assert bool(torch.isfinite(out_full).all())
    assert torch.equal(wh, genome["Wh"])                               # founders did not learn
    assert t_think < 1.0 and t_full < 4.0
