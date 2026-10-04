import time
from dataclasses import fields
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from haishool.life9.planet import brain  # noqa: E402

I, H, K0 = 40, 32, 12
O = brain.OUT_DIM
TAGS = ("chain", "derived", "reference", "new_rule")


def gen(seed=0):
    return torch.Generator().manual_seed(seed)


def population(seed=0, shape=(2, 64), dtype=torch.float32, innate=None, hidden=H, k0=K0, in_dim=I, params=None):
    g = gen(seed)
    genome = brain.random_genome(g, shape, in_dim, hidden, O, k0, "cpu", innate=innate, dtype=dtype, params=params)
    return g, genome


def flat_rows(genome):
    n = genome["k"].numel()
    return {name: t.reshape(n, *t.shape[2:]).clone() for name, t in genome.items()}


def reference_think(genome, wh, state, x):
    """Independent einsum reference: new = tanh(x Wx + h Wh + b) on units < k, out = new Wo + bo."""
    H_ = wh.shape[-1]
    mask = (torch.arange(H_) < genome["k"][..., None]).float()
    h = state * mask
    pre = torch.einsum("...i,...ij->...j", x, genome["Wx"].float()) \
        + torch.einsum("...i,...ij->...j", h, wh.float()) + genome["b"].float()
    new = torch.tanh(pre) * mask
    return torch.einsum("...j,...jo->...o", new, genome["Wo"].float()) + genome["bo"].float(), new


def test_layout_defaults_and_parametrised():
    assert (brain.OUT_TURN, brain.OUT_SPEED, brain.OUT_LOUD) == (0, 1, 2)
    assert len(brain.ACTIONS) == 14 and len(brain.COLLECT_CLASSES) == 4
    assert brain.VOCAL == slice(3, 11) and brain.ACTION == slice(11, 25) and brain.COLLECT == slice(25, 29)
    assert brain.OUT_DIM == 29 and brain.FORAGE_OUT == 12
    for D, A in ((8, 14), (4, 6), (1, 1)):
        lay = brain.Layout(D=D, A=A)
        parts = [lay.turn, lay.speed, lay.loud, *range(lay.out_dim)[lay.vocal],
                 *range(lay.out_dim)[lay.action], *range(lay.out_dim)[lay.collect]]
        assert parts == list(range(lay.out_dim)) == list(range(brain.out_dim(D, A)))
        assert lay.action == brain.action_slice(D, A) and lay.collect == brain.collect_slice(D, A)
    with pytest.raises(ValueError):
        brain.Layout(A=6).action_out("forage")


def test_random_genome_is_seeded_and_well_formed():
    _, a = population(3)
    _, b = population(3)
    _, c = population(4)
    for name in a:
        assert torch.equal(a[name], b[name]), name
    assert not torch.equal(a["Wx"], c["Wx"])
    assert a["Wx"].shape == (2, 64, I, H) and a["Wh"].shape == (2, 64, H, H) and a["Wo"].shape == (2, 64, H, O)
    assert a["k"].dtype == torch.long and bool((a["k"] == K0).all())
    p = brain.BrainParams()
    assert bool((a["eta"] == torch.tensor(p.eta0)).all()) and bool((a["temperature"] == 1.).all())
    with pytest.raises(ValueError):
        brain.random_genome(gen(), (1, 2), I, H, O, H + 1, "cpu")


def test_random_genome_takes_eta_and_temperature_from_params():
    _, a = population(3, params=SimpleNamespace(eta0=.003, temperature0=2.5))
    assert bool((a["eta"] == torch.tensor(.003)).all()) and bool((a["temperature"] == 2.5).all())
    _, b = population(3)
    assert torch.equal(a["Wx"], b["Wx"])                           # same weight draws


def test_think_matches_an_einsum_reference():
    g, genome = population(12)
    genome["k"] = torch.randint(2, H + 1, genome["k"].shape, generator=g)
    genome["b"] = torch.randn(genome["b"].shape, generator=g)        # non-zero biases on purpose
    genome["bo"] = torch.randn(genome["bo"].shape, generator=g)
    wh = brain.init_live(genome) + .3 * torch.randn(2, 64, H, H, generator=g)   # asymmetric live weights
    state = torch.randn(2, 64, H, generator=g)
    x = torch.rand(2, 64, I, generator=g)
    out, new = brain.think(genome, wh, state, x)
    ref_out, ref_new = reference_think(genome, wh, state, x)
    assert torch.allclose(out, ref_out, atol=1e-5) and torch.allclose(new, ref_new, atol=1e-5)
    # a transposed recurrent matrix would give something else
    _, tr_new = reference_think(genome, wh.transpose(-1, -2), state, x)
    assert not torch.allclose(new, tr_new, atol=1e-3)


def test_masked_units_stay_zero_and_never_matter():
    g, genome = population(1)
    genome["k"] = torch.randint(2, H + 1, genome["k"].shape, generator=g)
    k = genome["k"]
    beyond = torch.arange(H) >= k[..., None]                      # [W, N, H]
    wh = brain.init_live(genome)
    state = torch.randn(2, 64, H, generator=g)                     # junk beyond k on purpose
    x = torch.rand(2, 64, I, generator=g)
    out, new = brain.think(genome, wh, state, x, H)
    assert bool((new[beyond] == 0).all())
    # change every weight that touches an inactive unit: nothing observable may change
    other = {name: t.clone() for name, t in genome.items()}
    other["Wx"] = torch.where(beyond[..., None, :], 7.0, other["Wx"])
    other["b"] = torch.where(beyond, -5.0, other["b"])
    other["Wo"] = torch.where(beyond[..., :, None], 9.0, other["Wo"])
    wh2 = torch.where(beyond[..., :, None] | beyond[..., None, :], 4.0, wh)
    out2, new2 = brain.think(other, wh2, state, x, H)
    assert torch.equal(out, out2) and torch.equal(new, new2)
    # several steps of thinking and learning keep them zero and their weights untouched
    for _ in range(5):
        out, nxt = brain.think(genome, wh, new, x, H)
        before = wh.clone()
        brain.hebbian(wh, genome["eta"] * 100, torch.ones(2, 64), new, nxt, torch.ones(2, 64, dtype=torch.bool),
                      3.0, k=k)
        touched = beyond[..., :, None] | beyond[..., None, :]
        assert torch.equal(wh[touched], before[touched])
        assert bool((nxt[beyond] == 0).all())
        new = nxt
    assert not torch.equal(wh, brain.init_live(genome))           # the active block did learn


def test_cpu_think_is_exact_and_chunking_agrees():
    g, genome = population(2)
    wh = brain.init_live(genome)
    x = torch.rand(2, 64, I, generator=g)
    state = torch.zeros(2, 64, H)
    a = brain.think(genome, wh, state, x)
    b = brain.think(genome, wh, state, x)
    c = brain.think(genome, wh, state, x, chunk=17)
    d = brain.think(genome, wh, state, x, width=H)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    for other in (c, d):
        assert torch.allclose(a[0], other[0], atol=1e-6) and torch.allclose(a[1], other[1], atol=1e-6)


def test_width_below_max_k_raises_on_cpu():
    _, genome = population(2)
    genome["k"][0, 0] = 20
    wh, x = brain.init_live(genome), torch.rand(2, 64, I, generator=gen(1))
    with pytest.raises(ValueError):
        brain.think(genome, wh, torch.zeros(2, 64, H), x, width=12)
    with pytest.raises(ValueError):
        brain.hebbian(wh, genome["eta"], torch.ones(2, 64), torch.zeros(2, 64, H), torch.zeros(2, 64, H),
                      torch.ones(2, 64, dtype=torch.bool), 3.0, k=genome["k"], width=12)
    out, _ = brain.think(genome, wh, torch.zeros(2, 64, H), x, width=20)
    assert torch.allclose(out, brain.think(genome, wh, torch.zeros(2, 64, H), x)[0], atol=1e-6)


def test_innate_wiring_makes_founders_forage_on_plants():
    plant = 5
    _, genome = population(0, innate={"plant_index": plant})
    wh = brain.init_live(genome)
    x = torch.zeros(2, 64, I)
    lo, _ = brain.think(genome, wh, torch.zeros(2, 64, H), x)
    x[..., plant] = 1.
    hi, _ = brain.think(genome, wh, torch.zeros(2, 64, H), x)
    p_lo = brain.action_probs(lo[..., brain.ACTION], genome["temperature"])[..., brain.ACT["forage"]]
    p_hi = brain.action_probs(hi[..., brain.ACTION], genome["temperature"])[..., brain.ACT["forage"]]
    assert bool((p_hi > p_lo).all())
    assert float(p_lo.mean()) > 1.5 / len(brain.ACTIONS)        # the forage bias alone
    # the relay unit is clean: only the plant input drives it, only the forage logit reads it
    assert bool((genome["Wx"][..., :, 0].count_nonzero(-1) == 1).all())
    assert bool((genome["Wo"][..., 0, :].count_nonzero(-1) == 1).all())
    assert bool((genome["Wh"][..., 0, :] == 0).all()) and bool((genome["Wh"][..., :, 0] == 0).all())
    with pytest.raises(ValueError):
        population(0, innate={"plant_index": I})


def test_offspring_start_from_genome_not_learned_weights():
    g, genome = population(5)
    wh = brain.init_live(genome)
    state = torch.zeros(2, 64, H)
    alive = torch.ones(2, 64, dtype=torch.bool)
    for _ in range(20):                                            # a life of learning
        x = torch.rand(2, 64, I, generator=g)
        _, new = brain.think(genome, wh, state, x)
        brain.hebbian(wh, genome["eta"] * 100, torch.ones(2, 64), state, new, alive, 3.0, k=genome["k"])
        state = new
    assert float((wh - genome["Wh"]).abs().max()) > 1e-3
    w, p, c = torch.tensor([0, 1, 1]), torch.tensor([3, 7, 7]), torch.tensor([10, 20, 21])
    exact = brain.BrainParams(hidden=H, weight_mutation_rel=0., bias_mutation_sd=0., hidden_mutation_rate=0.)
    child = brain.mutate(g, brain.rows(genome, w, p), exact, {**brain.BRAIN_SPECS,
                                                               "eta": brain.GeneSpec(0, 1, 0., "abs"),
                                                               "temperature": brain.GeneSpec(.1, 10, 0., "log")})
    for name in genome:
        assert torch.equal(child[name], genome[name][w, p]), name   # the genome, not wh_live
    assert not torch.equal(child["Wh"], wh[w, p])
    baseline = torch.ones(2, 64)
    brain.install(genome, wh, state, w, c, child, baseline=baseline)
    assert torch.equal(wh[w, c], genome["Wh"][w, c]) and bool((state[w, c] == 0).all())
    assert bool((baseline[w, c] == 0).all()) and float(baseline.sum()) == 2 * 64 - 3
    for name in genome:
        assert torch.equal(genome[name][w, c], genome[name][w, p]), name


def test_install_refuses_a_child_with_other_genes():
    _, genome = population(5)
    wh, state = brain.init_live(genome), torch.zeros(2, 64, H)
    w, c = torch.tensor([0]), torch.tensor([3])
    child = brain.rows(genome, w, torch.tensor([4]))
    del child["temperature"]                                       # would inherit slot 3's old temperature
    with pytest.raises(KeyError):
        brain.install(genome, wh, state, w, c, child)
    child = brain.rows(genome, w, torch.tensor([4]))
    child["voice"] = torch.ones(1)
    with pytest.raises(KeyError):
        brain.install(genome, wh, state, w, c, child)


def test_mutate_bounds_k_steps_body_genes_and_determinism():
    _, genome = population(6, shape=(1, 4000), hidden=8, k0=4, in_dim=6)
    n = 4000
    parent = flat_rows(genome)
    parent["k"][:5] = 2                                            # at the lower bound
    parent["k"][5:10] = 8                                          # at the upper bound
    hi = brain.BRAIN_SPECS["eta"].hi
    parent["eta"][:] = hi * .995
    parent["temperature"][:] = 9.9
    parent["speed"] = torch.full((n,), 1.19)
    parent["adult_mass_kg"] = torch.full((n,), 30.)
    parent["litter"] = torch.full((n,), 6, dtype=torch.long)
    body = {"speed": brain.GeneSpec(.3, 1.2, .03, "range"), "adult_mass_kg": brain.GeneSpec(.01, 1000., .03, "log"),
            "litter": brain.GeneSpec(1, 6, .2, "step")}
    specs = {**brain.BRAIN_SPECS, **body}
    params = brain.BrainParams(hidden=8, hidden_mutation_rate=.3, weight_clip=3.0)
    a = brain.mutate(gen(9), parent, params, specs)
    b = brain.mutate(gen(9), parent, params, specs)
    for name in a:
        assert torch.equal(a[name], b[name]), name
    dk = a["k"] - parent["k"]
    assert set(dk.unique().tolist()) <= {-1, 0, 1}
    assert abs(float((dk != 0).float()[10:].mean()) - .3) < .04
    assert int(a["k"].min()) >= 2 and int(a["k"].max()) <= 8
    f32 = torch.tensor                                            # bounds compared in float32
    assert bool((a["eta"] <= f32(hi)).all()) and bool((a["eta"] >= 0).all()) and bool((a["eta"] == f32(hi)).any())
    assert bool((a["temperature"] <= f32(10.)).all()) and bool((a["temperature"] >= f32(.1)).all())
    assert bool((a["speed"] <= f32(1.2)).all()) and bool((a["speed"] >= f32(.3)).all())
    ratio = (a["adult_mass_kg"] / 30.).log()
    assert abs(float(ratio.std()) - .03) < .005
    assert a["litter"].dtype == torch.long and int(a["litter"].max()) <= 6
    assert set((a["litter"] - 6).unique().tolist()) <= {-1, 0}
    # weight noise is relative to the founder sd at the parent's fan-in; biases are absolute
    m, bias = params.weight_mutation_rel, params.bias_mutation_sd
    kf = parent["k"].float()[:, None, None]
    expect = {"Wx": torch.full((n, 1, 1), m / 6 ** .5), "Wh": .5 * m / kf.sqrt(), "Wo": m / kf.sqrt()}
    for name, sd in expect.items():
        z = (a[name] - parent[name]) / sd
        assert abs(float(z.std()) - 1) < .02, name
    for name in ("b", "bo"):
        assert abs(float((a[name] - parent[name]).std()) - bias) < .003, name
    parent["mystery"] = torch.zeros(n)
    with pytest.raises(KeyError):
        brain.mutate(gen(9), parent, params, specs)


def test_mutate_clamps_weight_genes_at_the_clip():
    _, genome = population(6, shape=(1, 500), hidden=8, k0=4, in_dim=6)
    parent = flat_rows(genome)
    sign = torch.where(torch.arange(500) % 2 == 0, 1., -1.)
    for name in brain.WEIGHTS:
        parent[name] = 2.99 * sign.reshape(500, *[1] * (parent[name].dim() - 1)).expand_as(parent[name]).clone()
    params = brain.BrainParams(hidden=8, weight_mutation_rel=.3, bias_mutation_sd=.05)
    child = brain.mutate(gen(1), parent, params)
    for name in brain.WEIGHTS:
        assert float(child[name].abs().max()) == 3.0, name
        assert bool((child[name] == 3.0).any()) and bool((child[name] == -3.0).any()), name


def test_mutate_never_grows_k_beyond_the_genome():
    _, genome = population(6, shape=(1, 4000))                     # H = 32 < BrainParams().hidden = 128
    parent = flat_rows(genome)
    parent["k"][:] = H
    child = brain.mutate(gen(2), parent)                           # default params
    assert int(child["k"].max()) <= H and bool((child["k"] == H - 1).any())
    small = brain.mutate(gen(2), parent, SimpleNamespace(hidden=20))   # a config cap below the genome
    assert int(small["k"].max()) <= 20


def logit_change(k, n=256, seed=0, in_dim=96, hidden=128):
    """(founder action-logit spread, sd of the logit change one mutation makes) at brain size k."""
    g = gen(seed)
    founders = brain.random_genome(g, (1, n), in_dim, hidden, O, k, "cpu")
    xs = torch.rand(5, 1, n, in_dim, generator=g)

    def run(genome):
        wh, s = brain.init_live(genome), torch.zeros(1, n, hidden)
        for x in xs:
            out, s = brain.think(genome, wh, s, x)
        return out[..., brain.ACTION]

    before = run(founders)
    child = brain.mutate(g, flat_rows(founders), brain.BrainParams(hidden_mutation_rate=0.))
    after = run({name: t[None] for name, t in child.items()})
    return float(before.std()), float((after - before).std())


def test_one_mutation_moves_logits_a_small_k_independent_fraction():
    ratios = []
    for k in (16, 48, 128):
        spread, change = logit_change(k)
        ratios.append(change / spread)
    assert max(ratios) < .3                                        # the flat absolute 0.05 gave 0.7-1.05
    assert max(ratios) / min(ratios) < 1.25                        # mutational load does not grow with k


def test_spectral_radius_drift_without_selection_does_not_grow_with_k():
    growth = {}
    for k in (16, 128):
        g = gen(3)
        founders = brain.random_genome(g, (1, 48), 8, 128, O, k, "cpu")
        rows = flat_rows(founders)
        r0 = torch.linalg.eigvals(rows["Wh"][:, :k, :k]).abs().max(-1).values.mean()
        for _ in range(20):
            rows = brain.mutate(g, rows, brain.BrainParams(hidden_mutation_rate=0.))
        r20 = torch.linalg.eigvals(rows["Wh"][:, :k, :k]).abs().max(-1).values.mean()
        growth[k] = float(r20 / r0)
    assert all(1. < v < 1.25 for v in growth.values()), growth   # old code: x2 at k=16, x5 at k=128
    assert abs(growth[128] - growth[16]) < .1, growth


def test_sampling_is_seeded_follows_softmax_and_masks():
    g = gen(11)
    logits = torch.randn(1, 6, 14, generator=g)
    temp = torch.tensor([[.5, 1., 2., 1., 1., .1]])
    a = brain.sample_actions(logits, temp, gen(1))
    b = brain.sample_actions(logits, temp, gen(1))
    assert a.dtype == torch.long and a.shape == (1, 6) and torch.equal(a, b)
    many = logits.expand(20000, 6, 14)
    draws = brain.sample_actions(many, temp.expand(20000, 6), gen(2))
    assert not torch.equal(draws, brain.sample_actions(many, temp.expand(20000, 6), gen(3)))
    freq = torch.nn.functional.one_hot(draws, 14).float().mean(0)
    probs = brain.action_probs(logits, temp)[0]
    assert float((freq - probs).abs().max()) < .015
    allowed = torch.zeros(14, dtype=torch.bool)
    allowed[[2, 5]] = True
    masked = brain.sample_actions(many, temp.expand(20000, 6), gen(4), allowed.expand(20000, 6, 14))
    assert set(masked.unique().tolist()) <= {2, 5}
    out = torch.randn(3, 5, brain.OUT_DIM, generator=g)
    d1 = brain.act(out, {"temperature": torch.ones(3, 5)}, gen(7))
    d2 = brain.act(out, {"temperature": torch.ones(3, 5)}, gen(7))
    assert torch.equal(d1["action"], d2["action"]) and torch.equal(d1["collect_class"], d2["collect_class"])
    assert int(d1["collect_class"].max()) < 4 and d1["vocal"].shape == (3, 5, 8)
    assert float(d1["speed"].min()) >= 0 and float(d1["turn"].abs().max()) <= 1


def test_a_row_with_nothing_allowed_is_rest_in_both_sampling_and_probabilities():
    logits = torch.randn(4, 14, generator=gen(1))
    temp = torch.ones(4)
    allowed = torch.ones(4, 14, dtype=torch.bool)
    allowed[1] = False                                             # nothing allowed
    allowed[2, :3] = False
    probs = brain.action_probs(logits, temp, allowed)
    assert bool(torch.isfinite(probs).all()) and torch.allclose(probs.sum(-1), torch.ones(4))
    assert float(probs[1, 0]) == 1.0 and float(probs[2, :3].sum()) == 0.
    assert torch.equal(probs[0], brain.action_probs(logits, temp)[0])
    for seed in range(5):
        assert int(brain.sample_actions(logits, temp, gen(seed), allowed)[1]) == 0


def test_bfloat16_mode_runs_and_matches_float32():
    _, full = population(8)
    _, light = population(8, dtype=torch.bfloat16)
    for name in brain.WEIGHTS:
        assert light[name].dtype == torch.bfloat16
        assert torch.equal(light[name], full[name].to(torch.bfloat16))
    assert brain.genome_bytes(light) < .55 * brain.genome_bytes(full)
    ref = {name: (t.float() if t.dtype == torch.bfloat16 else t) for name, t in light.items()}
    x = torch.rand(2, 64, I, generator=gen(1))
    state = torch.zeros(2, 64, H)
    wh_light = brain.init_live(light, torch.bfloat16)
    out_l, new_l = brain.think(light, wh_light, state, x, chunk=50)
    out_r, new_r = brain.think(ref, brain.init_live(ref), state, x)
    assert out_l.dtype == torch.float32 and new_l.dtype == torch.float32
    assert torch.allclose(out_l, out_r, atol=1e-5) and torch.allclose(new_l, new_r, atol=1e-5)
    # float32-live weights with bfloat16 genes also work
    out_m, _ = brain.think(light, brain.init_live(light), state, x)
    assert torch.allclose(out_m, out_r, atol=1e-5)
    child = brain.mutate(gen(2), brain.rows(light, torch.tensor([0]), torch.tensor([1])), brain.BrainParams(hidden=H))
    assert child["Wx"].dtype == torch.bfloat16
    brain.hebbian(wh_light, light["eta"], torch.ones(2, 64), new_l, new_l, torch.ones(2, 64, dtype=torch.bool),
                  3.0, k=light["k"], gen=gen(3), chunk=40)
    assert wh_light.dtype == torch.bfloat16


def test_hebbian_matches_the_outer_product_exactly_and_skips_dead_slots():
    g = gen(21)
    shape, Hh = (2, 16), 10
    wh = torch.randn(*shape, Hh, Hh, generator=g)
    start = wh.clone()
    before = torch.randn(*shape, Hh, generator=g)                  # asymmetric: before != after
    after = torch.rand(*shape, Hh, generator=g) - .2
    eta = torch.rand(shape, generator=g) * .1
    mod = torch.rand(shape, generator=g) * 2 - 1
    alive = torch.rand(shape, generator=g) < .6
    brain.hebbian(wh, eta, mod, before, after, alive, 2.0)
    dW = (eta * mod * alive)[..., None, None] * before[..., :, None] * after[..., None, :]
    expect = (start + dW).clamp(-2.0, 2.0)
    assert torch.allclose(wh, expect, atol=1e-6, rtol=0)
    i, j = 3, 7                                                    # dW[i, j] = eta mod before_i after_j
    w0, c0 = torch.nonzero(alive)[0].tolist()
    want = (start[w0, c0, i, j] + eta[w0, c0] * mod[w0, c0] * before[w0, c0, i] * after[w0, c0, j]).clamp(-2, 2)
    assert abs(float(wh[w0, c0, i, j] - want)) < 1e-6
    assert torch.equal(wh[~alive], start[~alive].clamp(-2.0, 2.0))  # dead slots: no learning
    # the same in bfloat16 with stochastic rounding: dead slots bit-identical, chunk 0 still updates
    wb = start.clamp(-2, 2).to(torch.bfloat16)
    wb0 = wb.clone()
    brain.hebbian(wb, eta, mod, before, after, alive, 2.0, gen=gen(4), chunk=0)
    assert torch.equal(wb[~alive], wb0[~alive])
    assert not torch.equal(wb[alive], wb0[alive])
    expect_b = (wb0.float() + dW).clamp(-2.0, 2.0)
    assert torch.allclose(wb.float(), expect_b, atol=2 ** -7, rtol=0)   # within one bf16 ulp at |w| < 2


def test_hebbian_width_matches_k_and_rejects_unsafe_rounding():
    g, genome = population(9)
    wh_a, wh_b = brain.init_live(genome), brain.init_live(genome)
    x = torch.rand(2, 64, I, generator=g)
    s0 = torch.zeros(2, 64, H)
    _, s1 = brain.think(genome, wh_a, s0, x)
    _, s2 = brain.think(genome, wh_a, s1, x)
    args = (genome["eta"] * 100, torch.ones(2, 64), s1, s2, torch.ones(2, 64, dtype=torch.bool), 3.0)
    brain.hebbian(wh_a, *args, k=genome["k"])
    brain.hebbian(wh_b, *args, width=K0)
    assert torch.equal(wh_a, wh_b)
    with pytest.raises(ValueError):
        brain.hebbian(brain.init_live(genome, torch.bfloat16), *args)            # bf16 without gen
    with pytest.raises(ValueError):
        brain.hebbian(brain.init_live(genome, torch.float16), *args, gen=gen())  # unsupported dtype


def test_bfloat16_hebbian_rounds_stochastically_without_bias():
    shape = (1, 64)
    wh = torch.ones(*shape, 8, 8, dtype=torch.bfloat16)
    eta, mod = torch.full(shape, 1.), torch.ones(shape)
    alive = torch.ones(shape, dtype=torch.bool)
    pre, post = torch.full((*shape, 8), .001), torch.ones(*shape, 8)
    g = gen(5)
    for _ in range(500):                                           # 500 x 0.001, each below half a bf16 ulp
        brain.hebbian(wh, eta, mod, pre, post, alive, 3.0, gen=g)
    near = (torch.ones(8, 8) + .001).to(torch.bfloat16)
    assert bool((near == 1).all())                                 # nearest rounding would lose every step
    assert abs(float(wh.float().mean()) - 1.5) < .01               # stochastic rounding keeps them


def test_stochastic_rounding_is_unbiased_for_negative_and_mixed_values():
    x = torch.full((200_000,), -1.0023)
    r = brain.to_bf16_stochastic(x, gen(1)).float()
    assert set(r.unique().tolist()) == {-1.0, -1.0078125}         # the two neighbours only
    assert abs(float(r.double().mean()) + 1.0023) < 5e-5           # nearest rounding is off by 2.3e-3
    assert float(x.to(torch.bfloat16).float()[0]) == -1.0
    v = torch.tensor([-2.7183, -0.001234, 0.3337, 1.0023])
    many = brain.to_bf16_stochastic(v.repeat(100_000, 1), gen(2)).double().mean(0)
    ulp = torch.tensor([2 ** -6, 2 ** -17, 2 ** -9, 2 ** -7], dtype=torch.float64)
    assert bool(((many - v.double()).abs() < .02 * ulp).all())


def test_outcome_is_per_basal_day_and_ignores_investment():
    for M in (1., 30., 300.):
        mass = torch.tensor([M])
        basal = 3.4 * M ** .75 * 86400
        assert abs(float(brain.basal_j_day(mass)) / basal - 1) < 1e-6
        z = torch.zeros(1)
        fast = brain.outcome(torch.tensor([-basal]), z, mass, z)
        assert abs(float(fast) + 1) < 1e-6                         # a fasting day is -1 at every mass
        birth = brain.outcome(torch.tensor([-0.4 * basal * 30]), z, mass, z, invested_j=torch.tensor([.4 * basal * 30]))
        assert float(birth) == 0.                                  # paying for offspring is not a loss
    r = brain.outcome(torch.tensor([2e6]), torch.tensor([.5]), torch.tensor([10.]), torch.tensor([-.1]),
                      basal_j=torch.tensor([1e6]))
    assert torch.allclose(r, torch.tensor([2 + .5 - .8]))         # spec water and health terms unchanged
    alive = torch.tensor([True, False])
    mod, base = brain.modulate(torch.tensor([1., 1.]), torch.tensor([.5, .5]), alive)
    assert torch.allclose(mod, torch.tensor([float(torch.tanh(torch.tensor(.5))), 0.]))
    assert torch.allclose(base, torch.tensor([.525, 0.]))


def test_learning_rates_are_restated_per_day():
    p = brain.BrainParams()
    life = 11.6 * 30 ** .2 * 365.25
    assert abs(brain.REF_LIFE_D - life) < 1e-6
    assert abs(p.eta0 * life - .01 * 600) < 1e-9                   # the flat Hebbian total over a life
    eta = brain.BRAIN_SPECS["eta"]
    assert abs(eta.hi / p.eta0 - 20) < 1e-9 and abs(eta.scale / p.eta0 - .5) < 1e-9   # flat ratios kept


def test_params_from_a_config_like_object_and_provenance_is_complete():
    p = brain.brain_params(SimpleNamespace(hidden=64, initial_hidden=8, weight_mutation_rel=.2, mutation_scale=9.))
    assert (p.hidden, p.initial_hidden, p.weight_mutation_rel, p.weight_clip) == (64, 8, .2, 3.0)
    assert brain.brain_params().hidden == 128 and brain.brain_params().initial_hidden == 48
    keys = [f.name for f in fields(brain.BrainParams)] + list(brain.BRAIN_SPECS) + list(brain.INNATE_DEFAULT)
    keys += ["ACTIONS", "COLLECT_CLASSES", "VOCAL_DIMS", "KLEIBER_W", "LIFESPAN_YR", "FLAT_LIFE_TICKS",
             "REF_MASS_KG", "TICK_TO_DAY", "OUTCOME_WATER", "OUTCOME_HEALTH", "outcome", "modulator",
             "init_scale", "innate"]
    missing = [key for key in keys if key not in brain.PROVENANCE]
    assert not missing, missing
    for key, (tag, note) in brain.PROVENANCE.items():
        assert tag in TAGS and note, key


def test_think_2048_individuals_is_fast_on_cpu():
    g = gen(0)
    in_dim = 180                                                   # about the size senses.observe gives
    genome = brain.random_genome(g, (2, 1024), in_dim, 128, brain.OUT_DIM, 48, "cpu",
                                 innate={"plant_index": 3})
    wh = brain.init_live(genome)
    state = torch.zeros(2, 1024, 128)
    x = torch.rand(2, 1024, in_dim, generator=g)
    brain.think(genome, wh, state, x)
    t = time.perf_counter()
    for _ in range(5):
        out, state = brain.think(genome, wh, state, x)
    per_step = (time.perf_counter() - t) / 5
    print(f"\nthink(): 2048 individuals, in 180, hidden 128 (48 active): {per_step * 1e3:.2f} ms per step")
    assert out.shape == (2, 1024, brain.OUT_DIM) and bool(torch.isfinite(out).all())
    assert per_step < 0.1


def test_outcome_weighs_water_by_the_lethal_margin():
    """With a lethal water margin the water term is 8 x d_water / margin: losing the whole margin weighs as
    much as losing all health; the default keeps the spec's 10 d_water / M."""
    mass = torch.tensor([30.0])
    margin = 0.2 * 0.668 * mass
    z = torch.zeros(1)
    lose_all = brain.outcome(z, -margin, mass, z, basal_j=torch.tensor([1.0]), water_margin_kg=margin)
    assert torch.allclose(lose_all, torch.tensor([-brain.OUTCOME_HEALTH]))
    drink = brain.outcome(z, torch.tensor([2.0]), mass, z, water_margin_kg=margin)
    assert float(drink) == pytest.approx(8 * 2.0 / float(margin))
    assert float(brain.outcome(z, torch.tensor([2.0]), mass, z)) == pytest.approx(10 * 2.0 / 30.0)
    assert "outcome_water_margin" in brain.PROVENANCE
