"""brain3: founders with nothing wired in, learning only through evolved genes, mutation by a heritable scale."""
import ast
import inspect
import math
import re
import types
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from haishool.life9.v3 import brain3 as b3  # noqa: E402

I, H = 24, 32
O = b3.OUT_DIM
TAGS = ("chain", "derived", "reference", "new_rule")
BODY_GENES = ("size_kg", "fur_m", "skin_perm", "thermo_gain", "setpoint_k", "enz_plant", "enz_meat",
              "repair", "muscle", "muscle_frac", "offspring_share", "eye", "ear", "voice")


def gen(seed=0):
    return torch.Generator().manual_seed(seed)


def f32(x):
    """A Python bound as float32 stores it (tensors are float32, bounds are Python floats)."""
    return float(torch.tensor(x, dtype=torch.float32))


def founders(seed=0, shape=(2, 64), hidden=H, in_dim=I, k_range=(2, 16), dtype=torch.float32):
    g = gen(seed)
    return g, b3.random_genome3(g, shape, in_dim, hidden=hidden, k_range=k_range, dtype=dtype)


def flat(genome):
    n = genome["k"].numel()
    lead = genome["k"].dim()
    return {name: t.reshape(n, *t.shape[lead:]).clone() for name, t in genome.items()}


def reference_think(genome, wh, state, x):
    mask = (torch.arange(wh.shape[-1]) < genome["k"][..., None]).float()
    pre = torch.einsum("...i,...ij->...j", x, genome["Wx"].float()) \
        + torch.einsum("...i,...ij->...j", state * mask, wh.float()) + genome["b"].float()
    new = torch.tanh(pre) * mask
    return torch.einsum("...j,...jo->...o", new, genome["Wo"].float()) + genome["bo"].float(), new


def active_mask(genome, name):
    """Bool mask of the expressed entries of a weight gene (units < k)."""
    k = genome["k"]
    Hh = genome["Wh"].shape[-1]
    act = torch.arange(Hh) < k[..., None]
    t = genome[name]
    if name == "Wx":
        return act[..., None, :].expand(t.shape)
    if name == "Wh":
        return (act[..., :, None] & act[..., None, :]).expand(t.shape)
    if name == "b":
        return act
    if name == "Wo":
        return act[..., :, None].expand(t.shape)
    return torch.ones(t.shape, dtype=torch.bool)


# ---------------------------------------------------------------------------- layout and decode
def test_layout_is_the_physical_abilities_in_spec_order():
    assert b3.SCALAR_OUTPUTS == ("turn", "thrust", "mouth", "grip_left", "grip_right", "release", "force", "rub",
                                 "press", "place", "loudness")
    consts = (b3.OUT_TURN, b3.OUT_THRUST, b3.OUT_MOUTH, b3.OUT_GRIP_LEFT, b3.OUT_GRIP_RIGHT, b3.OUT_RELEASE,
              b3.OUT_FORCE, b3.OUT_RUB, b3.OUT_PRESS, b3.OUT_PLACE, b3.OUT_LOUDNESS)
    assert consts == tuple(range(11))
    assert b3.OUT_VOCAL == slice(11, 19) and b3.OUT_DIVIDE == 19 and b3.OUT_DIM == 20
    for D in (1, 3, 8, 16):
        lay = b3.Layout3(D=D)
        assert lay.out_dim == 12 + D and len(lay.names) == lay.out_dim
        idx = [*range(11), *range(lay.out_dim)[lay.vocal], lay.divide]
        assert idx == list(range(lay.out_dim))
        assert lay.index("divide") == lay.divide and lay.index("rub") == 7


def test_decode_maps_to_physical_ranges_with_raw_zero_at_the_middle():
    g = gen(1)
    raw = torch.randn(3, 50, O, generator=g) * 20
    d = b3.decode(raw)
    for name in b3.SCALAR_OUTPUTS + ("divide",):
        lo = -1. if name in b3.SIGNED_OUTPUTS else 0.
        assert d[name].shape == (3, 50) and float(d[name].min()) >= lo and float(d[name].max()) <= 1.
    assert d["vocal"].shape == (3, 50, 8) and float(d["vocal"].abs().max()) <= 1.
    assert torch.equal(d["grip"], torch.stack((d["grip_left"], d["grip_right"]), -1))
    mid = b3.decode(torch.zeros(O))
    for name in b3.SCALAR_OUTPUTS + ("divide",):
        assert float(mid[name]) == (0. if name in b3.SIGNED_OUTPUTS else .5), name
    assert torch.equal(mid["vocal"], torch.zeros(8))
    # each output reads its own raw unit and nothing else
    for j, name in enumerate(b3.LAYOUT3.names):
        z = torch.zeros(O)
        z[j] = 2.
        dz = b3.decode(z)
        changed = [n for n in b3.SCALAR_OUTPUTS + ("divide",) if float(dz[n]) != float(mid[n])]
        if name.startswith("vocal"):
            assert changed == [] and float(dz["vocal"][int(name[5:])]) == pytest.approx(math.tanh(2.))
        else:
            assert changed == [name]


def test_fire_is_only_ever_bernoulli_on_the_intent_and_seeded():
    p = torch.linspace(0, 1, 11).repeat(4000, 1)
    a, b = b3.fire(p, gen(5)), b3.fire(p, gen(5))
    assert torch.equal(a, b) and a.dtype == torch.bool
    freq = a.float().mean(0)
    assert torch.allclose(freq, p[0], atol=.03)
    assert not bool(a[:, 0].any()) and bool(a[:, -1].all())
    # a raw 0 (intent 1/2) is an even chance for every body, never a deterministic 'always' or 'never'
    half = b3.fire(torch.full((20000,), .5), gen(6))
    assert float(half.float().mean()) == pytest.approx(.5, abs=.015)
    with pytest.raises(TypeError):
        b3.fire(torch.tensor([.2, .5, .51]))
    with pytest.raises(TypeError):
        b3.fire(torch.tensor([.2, .5, .51]), None)
    assert "fire" in b3.PROVENANCE


# ---------------------------------------------------------------------------- founders
def test_random_genome3_takes_no_wiring_argument():
    params = inspect.signature(b3.random_genome3).parameters
    assert set(params) == {"gen", "shape", "in_dim", "hidden", "layout", "n_intero", "k_range", "device",
                           "dtype", "specs"}


def test_founders_are_seeded_well_formed_and_learn_nothing():
    _, a = founders(3)
    _, b = founders(3)
    _, c = founders(4)
    assert set(a) == {*b3.WEIGHT_GENES, "k", *b3.GENE_SPECS3}
    for name in a:
        assert torch.equal(a[name], b[name]), name
    assert not torch.equal(a["Wx"], c["Wx"])
    assert a["Wx"].shape == (2, 64, I, H) and a["Wh"].shape == (2, 64, H, H) and a["Wo"].shape == (2, 64, H, O)
    assert a["g"].shape == (2, 64, b3.N_INTERO) and a["k"].dtype == torch.long
    assert int(a["k"].min()) >= 2 and int(a["k"].max()) <= 16 and len(a["k"].unique()) > 8
    assert not bool(a["b"].any()) and not bool(a["bo"].any())          # no bias toward any output
    assert not bool(a["g"].any())                                      # no notion of good: m = 0 exactly
    assert bool((a["eta"] == f32(b3.ETA_MIN)).all())                    # learning at its negligible floor
    for name, s in b3.GENE_SPECS3.items():
        v = a[name]
        assert v.dtype == torch.float32
        assert float(v.min()) >= f32(s.founder_lo) and float(v.max()) <= f32(s.founder_hi), name
    assert float(a["size_kg"].min()) >= .002 and float(a["size_kg"].max()) <= f32(.2)   # spec 6: 2-200 g
    assert bool((a["thermo_gain"] == f32(b3.THERMO_BOUNDS[0])).all())                  # all ectotherms
    assert float(a["thermo_gain"].max()) * 47. < .1      # under 0.1 W/kg even 47 K below any setpoint
    for organ in ("eye", "ear", "voice"):
        assert float(a[organ].max()) <= f32(b3.ORGAN_FOUNDER[1])                         # organs small
    share = a["offspring_share"]
    assert float((share > .5).float().mean()) == pytest.approx(.5, abs=.12)    # drawn on both sides of 1/2
    with pytest.raises(ValueError):
        b3.random_genome3(gen(), (4,), I, hidden=H, k_range=(1, 8))
    with pytest.raises(ValueError):
        b3.random_genome3(gen(), (4,), I, hidden=H, k_range=(2, H + 1))


def test_founder_weights_lie_within_the_clip_so_no_first_mutation_shrinks_them():
    _, a = founders(22, shape=(4096,), hidden=H, k_range=(2, 2))     # k = 2: Wo sd 0.71, the widest founders
    for name in ("Wx", "Wh", "Wo"):
        assert float(a[name].abs().max()) <= b3.WEIGHT_CLIP, name
    assert float(a["Wo"].abs().max()) > 2.5                          # the tail is there, folded, not cut
    child = b3.mutate3(gen(1), a, k_rate=0.)
    keep = ~active_mask(a, "Wo")
    assert torch.equal(child["Wo"][keep], a["Wo"][keep])             # nothing moved but expressed weights


def test_founder_weight_spread_follows_each_founders_own_fan_in():
    _, a = founders(7, shape=(4096,), hidden=H, k_range=(2, H))
    k = a["k"].float()
    for name, fan in (("Wh", k), ("Wo", k)):
        z = a[name] * (fan.sqrt() / b3.INIT_GAIN[name])[:, None, None]
        assert float(z.std()) == pytest.approx(1., abs=.01), name
    assert float((a["Wx"] * I ** .5).std()) == pytest.approx(1., abs=.01)
    for name in ("Wx", "Wh", "Wo"):
        assert abs(float(a[name].mean())) < 2e-3


def test_no_output_is_favoured_over_random_genomes():
    """Over random founders the raw outputs are centred on 0 and equally spread: no motor output leans anywhere."""
    n = 8192
    g, a = founders(11, shape=(n,), hidden=H, k_range=(2, H))
    wh, state = b3.init_live(a), b3.init_state((n,), H, "cpu")
    for _ in range(3):
        x = torch.rand(n, I, generator=g)                       # non-negative senses, as physical channels are
        out, state = b3.think(a, wh, state, x)
    sd = out.std(0)
    se = sd / n ** .5
    assert bool((out.mean(0).abs() < 4.5 * se).all()), (out.mean(0) / se)
    assert float(sd.max() / sd.min()) < 1.12
    d = b3.decode(out)
    for name in b3.SCALAR_OUTPUTS + ("divide",):
        mid = 0. if name in b3.SIGNED_OUTPUTS else .5
        assert float(d[name].mean()) == pytest.approx(mid, abs=.02), name
        assert float((d[name] > mid).float().mean()) == pytest.approx(.5, abs=.03), name
    assert float(d["vocal"].mean()) == pytest.approx(0., abs=.01)


# ---------------------------------------------------------------------------- network
def test_think_matches_an_einsum_reference_and_chunks_agree():
    g, a = founders(2)
    wh = b3.init_live(a)
    state = torch.randn(2, 64, H, generator=g)
    x = torch.rand(2, 64, I, generator=g)
    out, new = b3.think(a, wh, state, x)
    ref_out, ref_new = reference_think(a, wh, state, x)
    assert torch.allclose(out, ref_out, atol=1e-5) and torch.allclose(new, ref_new, atol=1e-6)
    out2, new2 = b3.think(a, wh, state, x, chunk=17)
    assert torch.equal(out, out2) and torch.equal(new, new2)
    out3, _ = b3.think(a, wh, state, x)
    assert torch.equal(out, out3)                                  # CPU exact


def test_masked_units_stay_zero_never_matter_and_never_learn_whatever_the_caller_passes():
    g, a = founders(9)
    k = a["k"]
    masked = torch.arange(H) >= k[..., None]                        # [2, 64, H]
    wh = b3.init_live(a)
    state = torch.randn(2, 64, H, generator=g)
    x = torch.rand(2, 64, I, generator=g)
    out, new = b3.think(a, wh, state, x)
    assert not bool(new[masked].any())
    # scrambling every weight of the masked units changes nothing
    a2 = {n: t.clone() for n, t in a.items()}
    wh2 = wh.clone()
    a2["Wx"] = torch.where(masked[..., None, :], torch.randn_like(a["Wx"]) * 9, a["Wx"])
    a2["Wo"] = torch.where(masked[..., :, None], torch.randn_like(a["Wo"]) * 9, a["Wo"])
    a2["b"] = torch.where(masked, torch.randn_like(a["b"]) * 9, a["b"])
    wh2 = torch.where(masked[..., :, None] | masked[..., None, :], torch.randn_like(wh) * 9, wh2)
    out2, new2 = b3.think(a2, wh2, state, x)
    assert torch.allclose(out2, out, atol=1e-6) and torch.allclose(new2, new, atol=1e-6)
    # with learning on, only the active block changes, although before and after carry activity on
    # the masked units (hebbian masks them itself)
    a["eta"].fill_(.05)
    a["g"].fill_(1.)
    before = wh.clone()
    garbage_after = new + masked.float() * torch.rand(2, 64, H, generator=g)
    b3.hebbian(wh, a, state, garbage_after, torch.rand(2, 64, b3.N_INTERO, generator=g),
               torch.ones(2, 64, dtype=torch.bool), width=H)
    pair = masked[..., :, None] | masked[..., None, :]
    assert torch.equal(wh[pair], before[pair])
    assert bool((wh[~pair] != before[~pair]).any())


# ---------------------------------------------------------------------------- learning
def test_founders_never_change_their_live_weights():
    g, a = founders(5)
    wh = b3.init_live(a)
    state = b3.init_state((2, 64), H, "cpu")
    alive = torch.ones(2, 64, dtype=torch.bool)
    for _ in range(25):
        x = torch.rand(2, 64, I, generator=g)
        out, new = b3.think(a, wh, state, x)
        intero = torch.randn(2, 64, b3.N_INTERO, generator=g) * 3
        m = b3.hebbian(wh, a, state, new, intero, alive)
        assert not bool(m.any())
        state = new
    assert torch.equal(wh, a["Wh"])


def test_modulated_hebbian_matches_the_formula_and_skips_dead_slots():
    g, a = founders(6, shape=(3, 16))
    a["eta"] = torch.rand(3, 16, generator=g) * .05
    a["g"] = torch.randn(3, 16, b3.N_INTERO, generator=g)
    wh = b3.init_live(a)
    mask = (torch.arange(H) < a["k"][..., None]).float()
    before = (torch.rand(3, 16, H, generator=g) * 2 - 1) * mask
    after = (torch.rand(3, 16, H, generator=g) * 2 - 1) * mask
    intero = torch.rand(3, 16, b3.N_INTERO, generator=g)
    alive = torch.rand(3, 16, generator=g) < .7
    w0 = wh.clone()
    m = b3.hebbian(wh, a, before, after, intero, alive)
    m_ref = torch.tanh((a["g"] * intero).sum(-1))
    assert torch.allclose(m, m_ref, atol=1e-6)
    dw = (a["eta"] * m_ref * alive)[..., None, None] * before[..., :, None] * after[..., None, :]
    assert torch.allclose(wh, (w0 + dw).clamp(-b3.WEIGHT_CLIP, b3.WEIGHT_CLIP), atol=1e-7)
    assert torch.equal(wh[~alive], w0[~alive])
    assert torch.equal(b3.modulator(torch.zeros(5, 7), torch.randn(5, 7)), torch.zeros(5))


def test_bfloat16_storage_thinks_and_learns_with_stochastic_rounding():
    g, a = founders(8, dtype=torch.bfloat16)
    assert a["Wx"].dtype == torch.bfloat16 and a["mut"].dtype == torch.float32
    wh = b3.init_live(a, torch.bfloat16)
    state = b3.init_state((2, 64), H, "cpu")
    x = torch.rand(2, 64, I, generator=g)
    out, new = b3.think(a, wh, state, x)
    assert out.dtype == torch.float32 and bool(torch.isfinite(out).all())
    a["eta"].fill_(.01)
    a["g"].fill_(1.)
    alive = torch.ones(2, 64, dtype=torch.bool)
    with pytest.raises(ValueError):
        b3.hebbian(wh, a, new, new, torch.ones(2, 64, 7), alive)
    b3.hebbian(wh, a, new, new, torch.ones(2, 64, 7), alive, gen=g)
    child = b3.mutate3(g, flat(a))
    assert child["Wx"].dtype == torch.bfloat16 and child["eta"].dtype == torch.float32


def _bf16_grid(x):
    """(truncated toward zero, one bf16 ulp further from zero, fraction of the ulp) for float32 ``x``."""
    bits = x.contiguous().view(torch.int32)
    lo_bits = bits & -65536
    lo = lo_bits.view(torch.float32)
    hi = (lo_bits + 65536).view(torch.float32)
    frac = (bits & 65535).double() / 65536.
    return lo, hi, frac


def test_bfloat16_mutation_is_stochastically_rounded_and_unbiased_at_small_mut():
    _, a = founders(21, shape=(2048,), hidden=H, k_range=(H, H))      # every unit active
    a["mut"].fill_(b3.MUT_BOUNDS[0])                                   # mut = 1e-3: steps below one ulp
    a16 = {n: (t.to(torch.bfloat16) if n in b3.WEIGHT_GENES else t.clone()) for n, t in a.items()}
    p32 = {n: (t.float() if n in b3.WEIGHT_GENES else t.clone()) for n, t in a16.items()}
    c16 = b3.mutate3(gen(5), a16, k_rate=0.)
    c32 = b3.mutate3(gen(5), p32, k_rate=0.)       # the same draws up to the rounding: the exact float32 child
    for name in ("Wx", "Wh", "Wo"):
        x, y, p = c32[name], c16[name].float(), p32[name]
        lo, hi, frac = _bf16_grid(x)
        assert bool(((y == lo) | (y == hi)).all()), name                    # one of the two neighbours
        up = (y == hi) & (hi != lo)
        assert float(up.double().mean()) == pytest.approx(float(frac.mean()), abs=2e-3), name
        err = (y - x).double()
        assert abs(float(err.mean())) < 4.5 * float(err.std()) / err.numel() ** .5, name   # unbiased
        # fraction changed: the stochastic-rounding prediction (nearest rounding changes far fewer)
        p_change = torch.where(lo == p, frac, torch.where(hi == p, 1. - frac, torch.ones_like(frac)))
        changed = (y != p).double().mean()
        assert float(changed) == pytest.approx(float(p_change.mean()), abs=3e-3), name
        nearest = (x.to(torch.bfloat16).float() != p).double().mean()
        assert float(changed) > float(nearest) + .05, name
        # realised step: second moment = float32 step's + rounding's (x - lo)(hi - x); never below intended
        m2 = ((y - p).double() ** 2).mean()
        pred = ((x - p).double() ** 2).mean() + ((x - lo).double() * (hi - x).double()).mean()
        assert float(m2) == pytest.approx(float(pred), rel=.02), name
        assert float(m2.sqrt()) >= .98 * float(((x - p).double() ** 2).mean().sqrt()), name


# ---------------------------------------------------------------------------- mutation
def test_mutation_steps_do_not_depend_on_the_gene_value():
    """Two parents at different (interior) values take the same step in the gene's coordinate."""
    n = 512
    _, a = founders(12, shape=(n,), hidden=8, in_dim=4, k_range=(4, 4))
    a["mut"].fill_(.05)
    lo_par, hi_par = {k: t.clone() for k, t in a.items()}, {k: t.clone() for k, t in a.items()}
    for name, s in b3.GENE_SPECS3.items():
        c_lo, c_hi = b3.coord_bounds(s)
        for par, q in ((lo_par, .3), (hi_par, .7)):
            if name != "mut":
                par[name] = b3.from_coord(s.kind, torch.full_like(a[name], c_lo + q * (c_hi - c_lo)))
    c_a, c_b = b3.mutate3(gen(3), lo_par), b3.mutate3(gen(3), hi_par)
    for name, s in b3.GENE_SPECS3.items():
        if name == "mut":
            continue
        d_a = b3.to_coord(s.kind, c_a[name].double()) - b3.to_coord(s.kind, lo_par[name].double())
        d_b = b3.to_coord(s.kind, c_b[name].double()) - b3.to_coord(s.kind, hi_par[name].double())
        span = b3.coord_bounds(s)[1] - b3.coord_bounds(s)[0]
        assert float((d_a - d_b).abs().max()) < 1e-4 * span, name
        assert float(d_a.std()) > 0, name


def test_mutate3_moves_eta_and_g_off_their_founder_values_by_small_steps():
    g, a = founders(12, shape=(4096,))
    child = b3.mutate3(g, a)
    mut = child["mut"]                                                  # the child's own scale steps its genes
    assert float((child["g"] != 0).float().mean()) > .99                # the readout leaves 0 at once ...
    assert bool((child["g"].abs() <= 6 * mut[:, None] * b3.G_UNIT).all())   # ... by steps of mut x G_UNIT
    ratio = child["eta"] / a["eta"]
    assert bool((child["eta"] >= f32(b3.ETA_MIN)).all())               # reflected at the floor: only up
    assert float((ratio > 1).float().mean()) > .99
    assert bool((ratio.log() <= 6 * mut).all())                        # a log step of mut e-folds
    assert float(child["eta"].max()) <= b3.ETA_MAX


def test_drift_without_selection_has_no_sign_for_the_readout_gains():
    g, a = founders(13, shape=(4096,), hidden=8, in_dim=4, k_range=(2, 8))
    a["mut"].fill_(.2)
    for _ in range(30):
        a = b3.mutate3(g, a)
        a["mut"].fill_(.2)
    gm = a["g"].mean(0)
    se = a["g"].std(0) / 4096 ** .5
    assert bool((gm.abs() < 4.5 * se).all()), gm / se
    assert float(a["g"].abs().max()) <= b3.G_MAX


def test_mut_gene_scales_every_mutation_and_takes_a_fixed_log_step_itself():
    g, a = founders(14, shape=(2048,), hidden=H, k_range=(H // 2, H // 2))
    lo, hi = {n: t.clone() for n, t in a.items()}, {n: t.clone() for n, t in a.items()}
    lo["mut"].fill_(.01)
    hi["mut"].fill_(.1)
    for name in ("setpoint_k", "repair"):                              # lin genes away from their bounds
        lo[name].fill_(sum(b3.GENE_SPECS3[name][:2]) / 2)
        hi[name].fill_(sum(b3.GENE_SPECS3[name][:2]) / 2)
    c_lo, c_hi = b3.mutate3(gen(1), lo), b3.mutate3(gen(1), hi)
    for name in ("Wx", "Wh", "Wo", "b", "bo", "setpoint_k", "repair", "size_kg", "muscle"):
        if name in ("size_kg", "muscle"):
            d_lo = (c_lo[name] / lo[name]).log().std()
            d_hi = (c_hi[name] / hi[name]).log().std()
        else:
            on = active_mask(a, name) if name in b3.WEIGHT_GENES else torch.ones_like(a[name], dtype=torch.bool)
            d_lo, d_hi = (c_lo[name] - lo[name])[on].std(), (c_hi[name] - hi[name])[on].std()
        assert float(d_hi / d_lo) == pytest.approx(10., rel=.05), name
    # sd of a weight step is the child's mut x the founder sd at the fan-in
    # (mut' = mut e^(0.1 z), so E mut'^2 = 1.02 mut^2)
    k = a["k"][0].item()
    on = active_mask(a, "Wh")
    assert float((c_hi["Wh"] - hi["Wh"])[on].std()) == pytest.approx(.1 * .5 / k ** .5, rel=.03)
    assert float((c_hi["b"] - hi["b"])[active_mask(a, "b")].std()) == pytest.approx(.1 * b3.BIAS_UNIT, rel=.03)
    # mut's own log step is MUT_TAU whatever mut is (a step of its own size would drag mut down)
    for par, ch in ((lo, c_lo), (hi, c_hi)):
        assert float((ch["mut"] / par["mut"]).log().std()) == pytest.approx(b3.MUT_TAU, rel=.05)


def test_mutate3_steps_k_by_one_within_bounds():
    g, a = founders(15, shape=(20000,), hidden=H, in_dim=4, k_range=(2, H))
    child = b3.mutate3(g, a)
    dk = child["k"] - a["k"]
    assert set(dk.unique().tolist()) <= {-1, 0, 1}
    inner = (a["k"] > 2) & (a["k"] < H)
    assert float((dk[inner] != 0).float().mean()) == pytest.approx(b3.K_MUT_RATE, abs=.006)
    assert float((dk[inner] > 0).float().mean()) == pytest.approx(b3.K_MUT_RATE / 2, abs=.004)
    assert int(child["k"].min()) >= 2 and int(child["k"].max()) <= H
    capped = b3.mutate3(gen(2), a, hidden=10)
    assert int(capped["k"].max()) <= 10
    still = b3.mutate3(gen(2), a, k_rate=0.)
    assert torch.equal(still["k"], a["k"])


def test_inactive_units_never_mutate_and_a_new_unit_is_drawn_at_the_founder_scale():
    g, a = founders(23, shape=(8192,), hidden=H, in_dim=I, k_range=(8, 8))
    still = b3.mutate3(gen(1), a, k_rate=0.)
    for name in ("Wx", "Wh", "b", "Wo"):
        off = ~active_mask(a, name)
        assert torch.equal(still[name][off], a[name][off]), name
        assert bool((still[name][~off] != a[name][~off]).any()), name
    grown = b3.mutate3(gen(2), a, k_rate=1.)
    up = grown["k"] == 9
    assert .4 < float(up.float().mean()) < .6
    u = 8
    assert float(grown["Wo"][up, u].std()) == pytest.approx(1 / 9 ** .5, rel=.03)
    assert float(grown["Wh"][up][:, :, u].std()) == pytest.approx(.5 / 9 ** .5, rel=.03)
    assert float(grown["Wh"][up][:, u, :].std()) == pytest.approx(.5 / 9 ** .5, rel=.03)
    assert float(grown["Wx"][up][:, :, u].std()) == pytest.approx(1 / I ** .5, rel=.03)
    assert not bool(grown["b"][up, u].any())
    assert not torch.equal(grown["Wo"][up, u], a["Wo"][up, u])        # fresh, not the stored founder draw
    down = grown["k"] == 7                                              # a unit switched off keeps its weights
    assert torch.equal(grown["Wo"][down][:, 8:], a["Wo"][down][:, 8:])


def test_genes_stay_in_bounds_under_heavy_mutation():
    g, a = founders(16, shape=(2048,), hidden=8, in_dim=4, k_range=(2, 8))
    for name, s in b3.GENE_SPECS3.items():                      # start every gene at a bound
        a[name] = torch.full_like(a[name], s.hi if name in ("eta", "fur_m", "muscle", "eye") else s.lo)
    for _ in range(40):
        a = b3.mutate3(g, a)
        a["mut"] = torch.full_like(a["mut"], b3.MUT_BOUNDS[1])
    for name, s in b3.GENE_SPECS3.items():
        assert float(a[name].min()) >= f32(s.lo) and float(a[name].max()) <= f32(s.hi), name
        assert bool(torch.isfinite(a[name]).all()), name
    for name in b3.WEIGHT_GENES:
        assert float(a[name].abs().max()) <= b3.WEIGHT_CLIP, name


def test_reflect_is_identity_inside_and_mirrors_outside():
    v = torch.tensor([.3, 1.2, -.3, 2.4, -1.7, 0., 1., 3.])
    r = b3.reflect(v, 0., 1.)
    assert torch.allclose(r, torch.tensor([.3, .8, .3, .4, .3, 0., 1., 1.]), atol=1e-6)
    inside = torch.rand(1000, generator=gen()) * 47 + 271
    assert torch.equal(b3.reflect(inside, 271., 318.), inside)
    with pytest.raises(ValueError):
        b3.reflect(v, 1., 1.)
    big = torch.randn(3, 1000, generator=gen(1)) * 4
    assert torch.equal(b3._reflect_(big.clone(), -3., 3., chunk=7), b3.reflect(big, -3., 3.))


def test_coordinates_round_trip():
    for kind, v in (("lin", torch.tensor([-2., 0., 5.])), ("log", torch.tensor([1e-6, 1., 1e5])),
                    ("logit", torch.tensor([1e-6, .5, 1 - 1e-6]))):
        assert torch.allclose(b3.from_coord(kind, b3.to_coord(kind, v.double())), v.double(), rtol=1e-9)
    with pytest.raises(ValueError):
        b3.to_coord("rel", torch.ones(1))
    s = b3.GENE_SPECS3["offspring_share"]
    assert b3.coord_bounds(s)[0] == pytest.approx(-b3.coord_bounds(s)[1])       # symmetric about 1/2


def test_one_mutation_moves_outputs_a_k_independent_fraction():
    ratios = []
    for k in (8, 32, 128):
        g, a = founders(17, shape=(1024,), hidden=128, in_dim=I, k_range=(k, k))
        a["mut"].fill_(.05)
        x = torch.rand(1024, I, generator=g)
        state = torch.rand(1024, 128, generator=g) * 2 - 1
        out, _ = b3.think(a, b3.init_live(a), state, x)
        child = b3.mutate3(g, a, k_rate=0.)
        out_c, _ = b3.think(child, b3.init_live(child), state, x)
        ratios.append(float((out_c - out).std() / out.std()))
    assert max(ratios) < .15
    assert max(ratios) / min(ratios) < 1.3, ratios


def test_mutate3_is_deterministic_and_refuses_unknown_genes():
    _, a = founders(18)
    rows = flat(a)
    c1, c2 = b3.mutate3(gen(3), rows), b3.mutate3(gen(3), rows)
    for name in c1:
        assert torch.equal(c1[name], c2[name]), name
    c3 = b3.mutate3(gen(4), rows)
    assert not torch.equal(c1["Wx"], c3["Wx"])
    with pytest.raises(KeyError):
        b3.mutate3(gen(), {**rows, "mystery": rows["eta"]})
    no_mut = dict(rows)
    del no_mut["mut"]
    with pytest.raises(KeyError):
        b3.mutate3(gen(), no_mut)
    # any leading batch: (2, 64) rows give the same children as the flattened rows
    c_lead = b3.mutate3(gen(3), a)
    for name in c1:
        assert torch.equal(c_lead[name].reshape(c1[name].shape), c1[name]), name


def test_children_install_into_slots_without_learned_weights():
    g, a = founders(19)
    wh = b3.init_live(a)
    wh.add_(1.)                                                    # stands for a lifetime of learning
    state = torch.ones(2, 64, H)
    w, c = torch.tensor([0, 1]), torch.tensor([3, 5])
    child = b3.mutate3(g, b3.rows(a, w, c))
    b3.install(a, wh, state, w, c, child)
    for name in child:
        assert torch.equal(a[name][w, c], child[name]), name
    assert torch.equal(wh[w, c], child["Wh"]) and not bool(state[w, c].any())


# ---------------------------------------------------------------------------- tissue budget and bounds
def test_tissue_budget_closes_and_scales_an_overfull_genome_proportionally():
    g = gen(24)
    n = 1000
    genes = {name: torch.rand(n, generator=g) * .4 for name in b3.TISSUE_GENES}       # asks 0 to 2.4
    brain = torch.rand(n, generator=g) * .05
    t = b3.tissue_shares(genes, brain)
    parts = [*b3.TISSUE_GENES, "brain"]
    total = sum(t[name] for name in parts) + t["rest"]
    assert torch.allclose(total, torch.ones(n), atol=1e-6)
    assert all(bool((t[name] >= 0).all()) for name in parts + ["rest"])
    asked = sum(genes.values()) + brain
    assert torch.allclose(t["asked"], asked)
    fits = asked <= 1
    assert 0 < int(fits.sum()) < n
    for name in b3.TISSUE_GENES:
        assert torch.equal(t[name][fits], genes[name][fits])                       # a genome that fits is as asked
        assert torch.allclose(t[name][~fits] / t["eye"][~fits], genes[name][~fits] / genes["eye"][~fits], rtol=1e-5)
    assert not bool(t["rest"][~fits].any())
    assert torch.allclose(t["rest"][fits], 1 - asked[fits], atol=1e-6)
    _, a = founders(25, shape=(4096,))
    assert float(b3.tissue_shares(a)["asked"].max()) < 1.               # founders fit in their bodies
    assert set(b3.TISSUE_GENES) <= set(b3.BODY_SPECS3)
    for name in b3.TISSUE_GENES:
        assert b3.BODY_SPECS3[name].hi == 1.                          # bounded by the whole body, not a record


def test_bound_report_says_how_much_of_the_population_sits_on_each_bound():
    _, a = founders(26, shape=(100,), hidden=H, k_range=(8, 8))
    a["k"][:10] = H
    a["k"][10:15] = 2
    a["size_kg"][:30] = b3.SIZE_BOUNDS_KG[1]
    a["setpoint_k"][:] = sum(b3.SETPOINT_BOUNDS_K) / 2
    a["setpoint_k"][:40] = b3.SETPOINT_BOUNDS_K[0] + 1e-3        # within one step of the floor
    a["eye"][:20] = 1.
    a["Wo"][20:40, 0, :4] = b3.WEIGHT_CLIP                       # unit 0 is active in every body
    alive = torch.ones(100, dtype=torch.bool)
    rep = b3.bound_report(a, alive)
    assert rep["n"] == 100
    assert rep["k@hi"] == pytest.approx(.10) and rep["k@lo"] == pytest.approx(.05)
    assert rep["eta@lo"] == 1. and rep["eta@hi"] == 0.             # founders sit on the learning floor
    assert rep["thermo_gain@lo"] == 1.
    assert rep["g@lo"] == 0. and rep["g@hi"] == 0.
    assert rep["size_kg@hi"] == pytest.approx(.30) and rep["size_kg@lo"] == 0.
    assert rep["setpoint_k@lo"] == pytest.approx(.40)
    assert rep["tissue@full"] == pytest.approx(.20)
    n_on = int(active_mask(a, "Wo").sum())
    assert rep["Wo@clip"] == pytest.approx(20 * 4 / n_on)
    assert rep["Wx@clip"] == 0.
    for name in b3.GENE_SPECS3:
        assert f"{name}@lo" in rep and f"{name}@hi" in rep
    half = torch.arange(100) < 50
    rep2 = b3.bound_report(a, half, wh_live=b3.init_live(a))
    assert rep2["n"] == 50 and rep2["size_kg@hi"] == pytest.approx(30 / 50)
    assert rep2["Wh_live@clip"] == rep2["Wh@clip"]
    assert b3.bound_report(a, torch.zeros(100, dtype=torch.bool)) == {"n": 0}


# ---------------------------------------------------------------------------- tables and provenance
def test_body_gene_table_covers_spec_section_3_with_physical_bounds():
    assert set(b3.BODY_SPECS3) == set(BODY_GENES)
    assert set(b3.LEARN_SPECS3) == {"eta", "g", "mut"}
    b3.check_specs(b3.GENE_SPECS3)
    for name, s in b3.GENE_SPECS3.items():
        assert s.units and s.source and name in b3.PROVENANCE, name
        assert s.kind in b3.NEUTRAL_LAW
        assert ("neutral law" in b3.PROVENANCE[name][1]) or name in ("enz_meat", "ear", "voice"), name
    s = b3.BODY_SPECS3
    assert (s["setpoint_k"].lo, s["setpoint_k"].hi) == (271., 318.)
    assert s["thermo_gain"].founder == "fixed" and s["thermo_gain"].founder_lo == s["thermo_gain"].lo
    assert s["thermo_gain"].lo * 10 < .02                    # the floor is an ectotherm: 0.01 W/kg at 10 K
    # the skin gene is the skin alone: from below the waxiest cuticle measured (1e-11 SI) to free water
    assert s["skin_perm"].lo <= 1e-11 / 5
    assert s["skin_perm"].hi >= b3.WATER_KG_PER_J / 1.0      # a skin resistance of 1 s/m or less
    assert s["skin_perm"].founder_lo == s["skin_perm"].lo and s["skin_perm"].founder_hi == s["skin_perm"].hi
    # offspring share: both sides of 1/2 open, symmetric
    sh = s["offspring_share"]
    assert sh.kind == "logit" and sh.hi > .99 and sh.lo == pytest.approx(1 - sh.hi)
    assert sh.founder_lo < .5 < sh.founder_hi
    assert b3.GENE_SPECS3["mut"].by_mut is False and b3.GENE_SPECS3["mut"].unit == b3.MUT_TAU
    assert all(spec.by_mut for name, spec in b3.GENE_SPECS3.items() if name != "mut")
    for bad in (s["eye"]._replace(lo=1.), s["repair"]._replace(kind="log"), s["eye"]._replace(founder_hi=2.),
                s["eye"]._replace(kind="wild"), s["eye"]._replace(unit=0.), sh._replace(hi=1.),
                s["eye"]._replace(founder="logituniform", founder_hi=1.), s["eye"]._replace(by_mut=1)):
        with pytest.raises(ValueError):
            b3.check_specs({"eye": bad})


def test_body_genes_can_be_drawn_alone_for_body_py():
    g = gen(20)
    genes = b3.random_body_genes(g, (3, 5), specs=b3.BODY_SPECS3)
    assert set(genes) == set(BODY_GENES)
    assert all(t.shape == (3, 5) and t.dtype == torch.float32 for t in genes.values())
    assert b3.random_body_genes(gen(20), (3, 5), specs=b3.BODY_SPECS3)["size_kg"].equal(genes["size_kg"])


def test_provenance_is_complete_and_tagged():
    skip = {"PROVENANCE", "R_GAS", "GS3", "R_", "D_", "N_", "C_"}
    consts = [n for n in vars(b3) if re.fullmatch(r"[A-Z][A-Z0-9_]*", n) and n not in skip]
    keys = consts + list(b3.GENE_SPECS3) + ["k", "weight_mutation", "reflect", "modulator", "hebbian", "fire",
                                             "inactive_units", "unit_growth", "bf16_mutation", "mut_first",
                                             "tissue_shares", "bound_report"]
    missing = [key for key in keys if key not in b3.PROVENANCE]
    assert not missing, missing
    for key, (tag, note) in b3.PROVENANCE.items():
        assert tag in TAGS and note, key
    assert "1/mut^2" in b3.PROVENANCE["MUT_TAU"][1]                   # the stated reason for a fixed step
    for kind, law in b3.NEUTRAL_LAW.items():
        assert law and kind in b3.KINDS


def test_no_forbidden_mechanism_in_the_source():
    src = Path(b3.__file__).read_text(encoding="utf-8")
    for word in ("innate", "forage", "ACTIONS", "background_drink", "time_budget", "pedigree", "relatedness",
                 "seed_floor", "lifespan", "maturity", "reward", "outcome", "kleiber", "allometr", "temperature0"):
        assert word.lower() not in src.lower(), word
    assert "sample_actions" not in src and "softmax" not in src


def test_brain3_takes_only_the_network_from_version_2():
    """Only version 2's network, live-weight and storage helpers come in; its named actions, wired-in
    forage path, fixed scoring and allometric constants are not reachable through brain3."""
    allowed = {"think", "init_live", "init_state", "rows", "install", "genome_bytes", "hebbian",
               "to_bf16_stochastic", "INIT_GAIN", "WEIGHTS"}
    tree = ast.parse(Path(b3.__file__).read_text(encoding="utf-8"))
    taken = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[-1] == "brain":
            taken |= {alias.name for alias in node.names}
        if isinstance(node, ast.Import):
            assert not any(alias.name.endswith("planet.brain") for alias in node.names)
    assert taken and taken <= allowed, taken - allowed
    for bad in ("pb", "ACTIONS", "ACT", "INNATE_DEFAULT", "outcome", "KLEIBER_W", "LIFESPAN_YR", "basal_j_day",
                "modulate", "sample_actions", "action_probs", "act", "random_genome", "mutate", "BRAIN_SPECS",
                "FORAGE_OUT", "LAYOUT"):
        assert not hasattr(b3, bad), bad
    for value in vars(b3).values():
        if isinstance(value, types.ModuleType):
            assert "haishool" not in value.__name__, value.__name__
