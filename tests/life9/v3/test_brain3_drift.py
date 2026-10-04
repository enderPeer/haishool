"""brain3 without selection: every gene drifts only toward its stated neutral law, and nothing grows with time.

Mutation is the variation half of the one feedback loop. If its steps depended on a gene's own value,
the operator alone would push genes somewhere (a hidden prior). These tests run many independent
lineages through mutate3 with no selection at all and check the stated laws (brain3.NEUTRAL_LAW).
"""
import math

import pytest

torch = pytest.importorskip("torch")

from haishool.life9.v3 import brain3 as b3  # noqa: E402

MUT_ONLY = {"mut": b3.GENE_SPECS3["mut"]}


def gen(seed=0):
    return torch.Generator().manual_seed(seed)


def tiny_population(n, seed, specs=None, hidden=2, k=2):
    """n lineages with the smallest brain (the genes are under test) and every gene of ``specs``."""
    g = gen(seed)
    pop = {"Wx": torch.zeros(n, 1, hidden), "Wh": torch.zeros(n, hidden, hidden), "b": torch.zeros(n, hidden),
           "Wo": torch.zeros(n, hidden, 1), "bo": torch.zeros(n, 1), "k": torch.full((n,), k, dtype=torch.long)}
    pop.update(b3.random_body_genes(g, (n,), specs=specs))
    return g, pop


def to01(name, v):
    """A gene's value as its position in [0, 1] along its mutation coordinate."""
    s = b3.GENE_SPECS3[name]
    lo, hi = b3.coord_bounds(s)
    return (b3.to_coord(s.kind, v.double().clamp(s.lo, s.hi)) - lo) / (hi - lo)


def from01(name, u):
    s = b3.GENE_SPECS3[name]
    lo, hi = b3.coord_bounds(s)
    return b3.from_coord(s.kind, lo + (hi - lo) * u.double()).float()


def ks_uniform(u):
    """Kolmogorov-Smirnov distance of the sample ``u`` from U(0, 1)."""
    u = u.flatten().double().sort().values
    n = u.numel()
    i = torch.arange(1, n + 1, dtype=torch.float64)
    return float(torch.maximum(i / n - u, u - (i - 1) / n).max())


@pytest.mark.parametrize("held_mut", [None, .3])
def test_every_gene_keeps_its_stated_neutral_law(held_mut):
    """Start every gene at its stated neutral law; generations of mutation alone must leave that law intact.

    A stationary law is invariant under the kernel. With a value-dependent step (e.g. sd mut x |v|),
    mass flows toward small values and the uniform-in-coordinate start is visibly deformed.
    """
    n, G = 20000, 150
    g, pop = tiny_population(n, 1)
    for name in b3.GENE_SPECS3:
        pop[name] = from01(name, torch.rand(pop[name].shape, generator=g, dtype=torch.float64))
    if held_mut is not None:
        pop["mut"].fill_(held_mut)
    for _ in range(G):
        pop = b3.mutate3(g, pop, k_rate=0.)
        if held_mut is not None:
            pop["mut"].fill_(held_mut)
    for name in b3.GENE_SPECS3:
        if held_mut is not None and name == "mut":
            continue
        u = to01(name, pop[name])
        d = ks_uniform(u)
        assert d < 2.2 / math.sqrt(u.numel()), (name, d)            # KS at alpha 1e-4 per gene


def test_genes_started_mid_range_keep_their_median():
    """The reviewer's case: start mid-range, a few hundred neutral generations, no collapse toward a bound."""
    n, G = 20000, 300
    g, pop = tiny_population(n, 2)
    start = {}
    for name in b3.GENE_SPECS3:
        pop[name] = from01(name, torch.full(pop[name].shape, .5, dtype=torch.float64))
        start[name] = pop[name].clone()
    pop["mut"].fill_(.1)
    for _ in range(G):
        pop = b3.mutate3(g, pop, k_rate=0.)
        pop["mut"].fill_(.1)
    for name in b3.GENE_SPECS3:
        if name == "mut":
            continue
        u = to01(name, pop[name]).flatten()
        assert float(u.std()) > .01, name                                       # it did move
        se = float(u.std()) / u.numel() ** .5
        assert abs(float(u.mean()) - .5) < 4.5 * se, (name, float(u.mean()))   # no drift in its coordinate
        assert abs(float(u.median()) - .5) < max(6 * se, 1e-3), (name, float(u.median()))
    # in physical units: the median of each gene the reviewer measured is still its starting value
    for name in ("fur_m", "thermo_gain", "eye", "enz_plant", "eta"):
        med, s0 = float(pop[name].median()), float(start[name][0])
        assert math.log(med / s0) == pytest.approx(0., abs=.06), (name, med, s0)
    assert float(pop["g"].abs().median()) > .8                                  # |g| did not fall toward 0


def test_mut_has_no_trend_without_selection():
    """mut steps by a fixed MUT_TAU in log space: its median does not slide toward low mutation rates."""
    n = 10000
    g, pop = tiny_population(n, 3, specs=MUT_ONLY)
    lo, hi = math.log(b3.MUT_BOUNDS[0]), math.log(b3.MUT_BOUNDS[1])
    mid = (lo + hi) / 2
    pop["mut"].fill_(math.exp(mid))
    medians = []
    for gen_i in range(1, 1001):
        pop = b3.mutate3(g, pop, specs=MUT_ONLY, k_rate=0.)
        if gen_i % 250 == 0:
            medians.append(float(pop["mut"].log().median()))
    # a log step equal to mut itself slid the median from 0.031 to 0.023 (ln -0.3) by generation 500
    assert all(abs(m - mid) < .1 for m in medians), (mid, medians)
    assert not all(b < a for a, b in zip(medians, medians[1:])), medians        # no monotone slide
    spread = pop["mut"].log().std()
    assert float(spread) > 1.5                                                  # it did wander, both ways


def test_k_has_a_uniform_neutral_law():
    n, top = 10000, 16
    g, pop = tiny_population(n, 4, specs=MUT_ONLY, hidden=top)
    pop["k"] = torch.randint(2, top + 1, (n,), generator=g)
    for _ in range(60):
        pop = b3.mutate3(g, pop, specs=MUT_ONLY, k_rate=.5)
    counts = torch.bincount(pop["k"], minlength=top + 1)[2:].double()
    expected = n / (top - 1)
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    assert chi2 < 45., chi2                          # 14 degrees of freedom: p ~ 5e-5 at 45


def _growth_effect(pop, seed, H, I):
    """sd of the change in raw outputs when one unit is switched on by k + 1 (newborn state: zero at that unit)."""
    g = gen(seed)
    grown = b3.mutate3(g, pop, specs=MUT_ONLY, k_rate=1.)
    up = grown["k"] > pop["k"]
    child = {name: t[up] for name, t in grown.items()}
    m = int(up.sum())
    k_old = pop["k"][up]
    x = torch.rand(m, I, generator=g)
    state = (torch.rand(m, H, generator=g) * 2 - 1) * (torch.arange(H) < k_old[:, None])
    out_on, _ = b3.think(child, child["Wh"], state, x)
    off = dict(child, k=k_old)
    out_off, _ = b3.think(off, child["Wh"], state, x)
    u = k_old[0].item()
    return float((out_on - out_off).std()), float(child["Wo"][:, u].std())


def test_a_new_unit_has_the_same_effect_at_every_generation():
    """Inactive units' weights never drift, so switching one on is no more disruptive late than early."""
    n, H, I, G = 2048, 16, 4, 400
    g = gen(5)
    pop = b3.random_genome3(g, (n,), I, hidden=H, layout=b3.Layout3(D=1), k_range=(8, 8), specs=MUT_ONLY)
    founders = {name: t.clone() for name, t in pop.items()}
    early, wo_early = _growth_effect(pop, 50, H, I)
    pop["mut"].fill_(.1)
    for _ in range(G):
        pop = b3.mutate3(g, pop, specs=MUT_ONLY, k_rate=0.)
        pop["mut"].fill_(.1)
    late, wo_late = _growth_effect(pop, 50, H, I)
    inactive = torch.arange(H) >= 8
    assert torch.equal(pop["Wo"][:, inactive], founders["Wo"][:, inactive])          # never mutated
    assert torch.equal(pop["Wx"][:, :, inactive], founders["Wx"][:, :, inactive])
    assert torch.equal(pop["Wh"][:, inactive], founders["Wh"][:, inactive])
    assert torch.equal(pop["Wh"][:, :, inactive], founders["Wh"][:, :, inactive])
    active_wo = pop["Wo"][:, ~inactive]
    assert float(active_wo.std()) > 1.5 * float(founders["Wo"][:, ~inactive].std())  # the expressed ones did drift
    assert late / early == pytest.approx(1., abs=.12), (early, late)
    assert wo_early == pytest.approx(1 / 3, rel=.05) and wo_late == pytest.approx(1 / 3, rel=.05)


def test_a_unit_grown_far_from_the_founders_size_is_drawn_at_its_own_fan_in():
    """A brain that grew from k = 2 to k = 8 switches unit 8 on at the k = 9 scale, not the k = 2 founder draw."""
    n, H, I = 8192, 16, 4
    g = gen(6)
    pop = b3.random_genome3(g, (n,), I, hidden=H, layout=b3.Layout3(D=1), k_range=(2, 2), specs=MUT_ONLY)
    assert float(pop["Wo"][:, 8].std()) == pytest.approx(2 ** -.5, rel=.05)         # stored at the k = 2 scale
    pop["k"].fill_(8)
    grown = b3.mutate3(g, pop, specs=MUT_ONLY, k_rate=1.)
    up = grown["k"] == 9
    assert float(grown["Wo"][up, 8].std()) == pytest.approx(1 / 3, rel=.05)
    assert float(grown["Wh"][up][:, 8, :].std()) == pytest.approx(.5 / 3, rel=.05)
