"""Batched recurrent brains: one small network per individual, all evaluated in one call.

* The genome holds the network's starting weights, its active size ``k`` (hidden units
  0..k-1 are used, the rest are masked), its plasticity rate ``eta`` and two body genes
  (voice, speed). Brain size is a gene with a metabolic price, so capacity is evolved
  rather than fixed by the code (the cap is ``Config9.hidden``, reported when reached).
* The hidden state carries the past: a call heard three ticks ago can still shape what
  the individual does now (life8's linear learner saw only the freshest call).
* Within a life the recurrent weights change by reward-modulated Hebbian learning; the
  modulator is the individual's own energy and health change minus its running mean
  (life8's honest reward). Offspring start from the genome, never from learned weights.
"""
from __future__ import annotations

import torch

GENES = ("Wx", "Wh", "b", "Wo", "bo")


def random_genome(gen, shape, cfg, device):
    worlds, n = shape
    i, h, o = cfg.in_dim, cfg.hidden, cfg.out_dim

    def normal(*dims, scale):
        return torch.randn(worlds, n, *dims, generator=gen, device=device) * scale

    return {
        "Wx": normal(i, h, scale=1.0 / i ** .5),
        "Wh": normal(h, h, scale=.5 / h ** .5),
        "b": torch.zeros(worlds, n, h, device=device),
        "Wo": normal(h, o, scale=1.0 / h ** .5),
        "bo": torch.zeros(worlds, n, o, device=device),
        "k": torch.full((worlds, n), cfg.initial_hidden, dtype=torch.long, device=device),
        "eta": torch.full((worlds, n), .01, device=device),
        "voice": torch.rand(worlds, n, generator=gen, device=device),
        "speed": .6 + .3 * torch.rand(worlds, n, generator=gen, device=device),
    }


def unit_mask(k, hidden):
    return (torch.arange(hidden, device=k.device) < k[..., None]).float()


def think(genome, wh_live, state, x, hidden):
    """One step for every slot: returns (outputs [W,N,O], new hidden state [W,N,H])."""
    worlds, n, i = x.shape
    b = worlds * n
    mask = unit_mask(genome["k"], hidden)
    h = state * mask
    pre = (torch.bmm(x.reshape(b, 1, i), genome["Wx"].reshape(b, i, hidden))
           + torch.bmm(h.reshape(b, 1, hidden), wh_live.reshape(b, hidden, hidden))).reshape(worlds, n, hidden)
    new = torch.tanh(pre + genome["b"]) * mask
    out = torch.bmm(new.reshape(b, 1, hidden), genome["Wo"].reshape(b, hidden, -1)).reshape(worlds, n, -1) + genome["bo"]
    return out, new


def hebbian(wh_live, eta, modulator, before, after, alive, clip):
    """Reward-modulated Hebbian change of the live recurrent weights (in place)."""
    gain = (eta * modulator * alive.float())[..., None, None]
    wh_live.add_(gain * before[..., :, None] * after[..., None, :]).clamp_(-clip, clip)


def mutate(gen, parent, cfg):
    """Child genome rows from parent rows (each tensor has a leading batch of children)."""
    def noise(t, scale):
        return torch.randn(t.shape, generator=gen, device=t.device) * scale

    child = {name: parent[name] + noise(parent[name], cfg.mutation_scale) for name in GENES}
    step = torch.randint(-1, 2, parent["k"].shape, generator=gen, device=parent["k"].device)
    change = torch.rand(parent["k"].shape, generator=gen, device=parent["k"].device) < cfg.hidden_mutation_rate
    child["k"] = (parent["k"] + step * change).clamp(2, cfg.hidden)
    child["eta"] = (parent["eta"] + noise(parent["eta"], .005)).clamp(0, .2)
    child["voice"] = (parent["voice"] + noise(parent["voice"], .05)).clamp(0, 1.5)
    child["speed"] = (parent["speed"] + noise(parent["speed"], .03)).clamp(.3, 1.2)
    return child
