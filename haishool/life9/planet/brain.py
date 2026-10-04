"""Bigger batched recurrent brains for the planet engine (PLANET-SPEC section 2.10).

This generalises ``haishool/life9/brain.py`` (the flat prototype, imported, not edited):

* Dimensions are arguments: input size from ``senses``, up to ``hidden`` units (default 128,
  four times the flat prototype), outputs laid out by :class:`Layout` (turn, speed, loudness,
  the vocal vector of ``D`` channels, ``A`` action logits and 4 collect-class logits).
* The active size ``k`` (units ``0..k-1``) is a gene; inactive units are exactly zero and are
  not computed (the batch is evaluated only up to ``max(k)``). Each active unit costs
  metabolism (paid in ``creatures``).
* Actions are drawn from softmax(logits / temperature gene) with the world generator
  (Gumbel-max, so it is vectorised and exact on the CPU).
* Within a life the live recurrent weights change by Hebbian learning modulated by the
  individual's own outcome (reserve per basal day, water and health change, minus its running
  mean). Offspring start from the genome; nothing learned is inherited. The flat prototype's
  per-tick rates are restated per day (one tick is one day here; see ``TICK_TO_DAY``).
* Weight-gene mutation is relative to each matrix's founder scale (fan-in), so a mutation
  moves the outputs by the same fraction of their spread at any brain size ``k``: the evolved
  size is priced by the neuron cost, not by mutational load.
* Memory-light mode: weight genes (and optionally the live weights) may be stored in
  bfloat16 while every product is computed in float32. Hebbian updates to bfloat16 live
  weights use stochastic rounding drawn from the world generator, so small changes are not
  lost to rounding.

Every tensor has the leading batch axes of the population (``[W, N]``); there are no Python
loops over individuals. The genome is a dict of named tensors: the weight genes ``Wx Wh b Wo
bo``, ``k``, ``eta``, ``temperature``, plus any body genes another module adds; ``mutate``
handles all of them through :class:`GeneSpec` bounds.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import NamedTuple

import torch

from ..brain import GENES as WEIGHTS, unit_mask
from .constants import DAY_S, YEAR_S

# ---------------------------------------------------------------------------- output layout
ACTIONS = ("rest", "forage", "eat_meat", "drink", "strike", "collect", "drop", "knap", "combine",
           "make_fire", "feed_fire", "heat_item", "cook", "share")
ACT = {name: i for i, name in enumerate(ACTIONS)}
COLLECT_CLASSES = ("stone", "ore", "clay", "wood")
VOCAL_DIMS = 8
OUT_TURN, OUT_SPEED, OUT_LOUD = 0, 1, 2


def vocal_slice(D=VOCAL_DIMS):
    return slice(3, 3 + D)


def action_slice(D=VOCAL_DIMS, A=len(ACTIONS)):
    return slice(3 + D, 3 + D + A)


def collect_slice(D=VOCAL_DIMS, A=len(ACTIONS), C=len(COLLECT_CLASSES)):
    return slice(3 + D + A, 3 + D + A + C)


def out_dim(D=VOCAL_DIMS, A=len(ACTIONS), C=len(COLLECT_CLASSES)):
    return 3 + D + A + C


@dataclass(frozen=True)
class Layout:
    """Where each output lives: turn, speed, loudness, vocal [D], action logits [A], collect class [C]."""
    D: int = VOCAL_DIMS
    A: int = len(ACTIONS)
    C: int = len(COLLECT_CLASSES)
    turn: int = OUT_TURN
    speed: int = OUT_SPEED
    loud: int = OUT_LOUD

    @property
    def vocal(self):
        return vocal_slice(self.D)

    @property
    def action(self):
        return action_slice(self.D, self.A)

    @property
    def collect(self):
        return collect_slice(self.D, self.A, self.C)

    @property
    def out_dim(self):
        return out_dim(self.D, self.A, self.C)

    def action_out(self, name):
        """Output index of a named action logit (the names are ACTIONS; needs A == len(ACTIONS))."""
        if self.A != len(ACTIONS):
            raise ValueError("named actions need A == len(ACTIONS)")
        return self.action.start + ACT[name]


LAYOUT = Layout()
VOCAL, ACTION, COLLECT, OUT_DIM = LAYOUT.vocal, LAYOUT.action, LAYOUT.collect, LAYOUT.out_dim
FORAGE_OUT = LAYOUT.action_out("forage")

# ---------------------------------------------------------------------------- time scale and outcome
KLEIBER_W = 3.4            # W kg^-0.75, basal rate (Kleiber 1947: 70 kcal/day x M^0.75 = 3.39 W; PLANET-SPEC 2.8)
LIFESPAN_YR = 11.6         # maximum lifespan 11.6 yr x M^0.20 (Calder 1984, mammals; PLANET-SPEC 2.8)
LIFESPAN_EXP = 0.20
FLAT_LIFE_TICKS = 600      # the flat prototype's life in abstract ticks (haishool/life9/config.py:36)
REF_MASS_KG = 30.0         # new_rule: the body whose life the flat per-tick learning rate is spread over
REF_LIFE_D = LIFESPAN_YR * REF_MASS_KG ** LIFESPAN_EXP * YEAR_S / DAY_S    # 8,365 days at 30 kg
TICK_TO_DAY = FLAT_LIFE_TICKS / REF_LIFE_D                                  # 0.0717
OUTCOME_WATER, OUTCOME_HEALTH = 10.0, 8.0   # PLANET-SPEC 2.10 weights of 10 d_water / M and 8 d_health

INIT_GAIN = {"Wx": 1.0, "Wh": 0.5, "Wo": 1.0}   # founder sd = gain / sqrt(fan-in), as the flat prototype


# ---------------------------------------------------------------------------- parameters and genes
@dataclass(frozen=True)
class BrainParams:
    """Brain settings. Any config object with attributes of the same names can stand in.

    Rates of learning within a life are per day (one tick); mutation is per generation.
    """
    hidden: int = 128
    initial_hidden: int = 48
    weight_mutation_rel: float = 0.1
    bias_mutation_sd: float = 0.05
    hidden_mutation_rate: float = 0.05
    weight_clip: float = 3.0
    baseline_rate: float = 0.05
    eta0: float = 0.01 * TICK_TO_DAY
    temperature0: float = 1.0


def brain_params(cfg=None):
    """BrainParams from a config-like object (missing attributes keep their defaults)."""
    if cfg is None:
        return BrainParams()
    if isinstance(cfg, BrainParams):
        return cfg
    base = BrainParams()
    return BrainParams(**{f.name: getattr(cfg, f.name, getattr(base, f.name)) for f in fields(BrainParams)})


class GeneSpec(NamedTuple):
    """Bounds and mutation of one float or integer gene.

    kind: ``abs`` adds N(0, scale); ``range`` adds N(0, scale x (hi - lo)); ``log`` multiplies by
    exp(N(0, scale)); ``step`` adds -1 or +1 with probability ``scale``. The result is clamped
    to [lo, hi] and keeps the parent's dtype.
    """
    lo: float
    hi: float
    scale: float
    kind: str = "range"
    source: str = ""


BRAIN_SPECS = {
    "eta": GeneSpec(0.0, 0.2 * TICK_TO_DAY, 0.005 * TICK_TO_DAY, "abs",
                    "flat prototype haishool/life9/brain.py:72 (bounds 0-0.2, step 0.005 per tick) x TICK_TO_DAY"),
    "temperature": GeneSpec(0.1, 10.0, 0.03, "log",
                            "new_rule: bounds 0.1-10; 3 % log-normal step (gene scale of PLANET-SPEC 2.8)"),
}

INNATE_DEFAULT = {"forage_bias": 1.0, "plant_gain": 2.0, "relay_gain": 2.0, "relay_unit": 0}

PROVENANCE = {
    # BrainParams fields
    "hidden": ("new_rule", "PLANET-SPEC 2.10: maximum 128 units, 4 x the flat prototype's 32"),
    "initial_hidden": ("new_rule", "PLANET-SPEC 2.10: 48 units at the start; world7 senses era for 781 "
                                   "evolved 60.8 of a 64-neuron budget (evo/senses.py:436), abstract neurons"),
    "weight_mutation_rel": ("new_rule", "per generation, Wx, Wh and Wo get N(0, (0.1 x founder sd)^2) with the "
                                        "founder sd taken at the parent's own fan-in: Wx 1/sqrt(in_dim), "
                                        "Wh 0.5/sqrt(k), Wo 1/sqrt(k); one mutation moves the logits by about "
                                        "0.2 of the founder spread at every k (the flat absolute 0.05 was "
                                        "0.35-0.7 of the founder sd here and grew with k)"),
    "bias_mutation_sd": ("new_rule", "absolute N(0, 0.05) on b and bo, the flat prototype's value "
                                     "(config.py:69); about 0.1 of the founder pre-activation and logit spreads"),
    "hidden_mutation_rate": ("new_rule", "flat prototype config.py:70, per generation; k moves by exactly +-1 "
                                         "within [2, min(hidden, the genome's units)]"),
    "weight_clip": ("new_rule", "flat prototype config.py:62; bounds live weights and weight genes"),
    "baseline_rate": ("new_rule", "0.05 per day (time constant 20 days): the flat world's 0.05 per tick "
                                  "(world.py:233) restated per day and deliberately not rescaled by life length, "
                                  "so the running mean follows weather and seasons instead of crediting them "
                                  "to behaviour"),
    "eta0": ("derived", "0.01 per flat tick (brain.py:34) x TICK_TO_DAY = 7.2e-4 per day: the same Hebbian "
                        "total over a life (0.01 x 600 ticks = 7.17e-4 x 8,365 days)"),
    "temperature0": ("new_rule", "plain softmax at the start"),
    # genes (GeneSpec names)
    "eta": ("derived", "bounds 0 to 0.2 x TICK_TO_DAY, step 0.005 x TICK_TO_DAY per generation: the flat "
                       "prototype's per-tick values (brain.py:72) restated per day"),
    "temperature": ("new_rule", "softmax temperature gene, start 1, bounds 0.1-10, 3 % log-normal step"),
    # layout (PLANET-SPEC 2.10 design)
    "ACTIONS": ("new_rule", "PLANET-SPEC 2.10: the 14 actions, in order"),
    "COLLECT_CLASSES": ("new_rule", "PLANET-SPEC 2.10: collect takes stone, ore, clay or wood"),
    "VOCAL_DIMS": ("new_rule", "PLANET-SPEC 2.10: D = 8 vocal channels"),
    # time scale, outcome and initial weights
    "KLEIBER_W": ("reference", "Kleiber 1947, Physiol. Rev. 27:511: 70 kcal/day x M^0.75 = 3.39 W; PLANET-SPEC 2.8"),
    "LIFESPAN_YR": ("reference", "Calder 1984, Size, Function and Life History: 11.6 yr x M^0.20; PLANET-SPEC 2.8"),
    "FLAT_LIFE_TICKS": ("new_rule", "the flat prototype's lifespan, haishool/life9/config.py:36"),
    "REF_MASS_KG": ("new_rule", "30 kg reference body for restating per-tick learning per day (the spec does "
                                "not fix the founders' mass)"),
    "TICK_TO_DAY": ("derived", "FLAT_LIFE_TICKS / (11.6 yr x 30^0.20 x 365.25 d) = 600 / 8,365 = 0.0717"),
    "outcome": ("new_rule", "PLANET-SPEC 2.10 with the reserve term per basal day instead of per 1e6 J: "
                            "(d_reserve + invested) / (3.4 M^0.75 x 86,400 s) + 10 d_water/M + 8 d_health; "
                            "energy moved into offspring or growth is not a loss"),
    "OUTCOME_WATER": ("new_rule", "PLANET-SPEC 2.10: 10 d_water / M"),
    "outcome_water_margin": ("new_rule", "with a lethal water margin given, the water term is OUTCOME_HEALTH x "
                                         "d_water / margin: losing the whole margin (dehydration death) weighs as much "
                                         "as losing all health (8); 10 d_water / M valued a 2-day water margin at -1.3, "
                                         "a fifth of one grazing day"),
    "OUTCOME_HEALTH": ("new_rule", "PLANET-SPEC 2.10: 8 d_health"),
    "modulator": ("new_rule", "tanh(outcome - running mean) (flat world.py:230-233)"),
    "init_scale": ("new_rule", "Wx ~ N(0, 1/in_dim), Wh ~ N(0, 0.25/k0), Wo ~ N(0, 1/k0) with k0 = initial_hidden: "
                               "the active block starts with the flat prototype's gains 0.5 and 1"),
    "innate": ("chain", "world7 bodies era gives grazer lineages (evo/bodies.py GUILDS; 781: grazers 26.1 "
                        "biomass units), so founders lean to forage and to the local plant input"),
    "forage_bias": ("new_rule", "innate forage logit bias 1"),
    "plant_gain": ("new_rule", "innate plant input -> relay unit gain 2"),
    "relay_gain": ("new_rule", "innate relay unit -> forage logit gain 2"),
    "relay_unit": ("new_rule", "the relay is unit 0, active at every k >= 2"),
}


def _f32(t):
    return t if t.dtype == torch.float32 else t.float()


def _numel(shape):
    return math.prod(shape)


def _width(k, hidden, width=None):
    """Number of units evaluated: max active k in the batch (one host sync), or a given width.

    A given width below max(k) would silently drop units, so it is checked when ``k`` is on the
    CPU (free there); on an accelerator the caller must keep its cached width >= max(k).
    """
    if width is None:
        width = hidden if k is None else (int(k.max()) if k.numel() else 1)
    elif k is not None and k.device.type == "cpu" and k.numel() and int(k.max()) > width:
        raise ValueError(f"width {int(width)} is below max(k) = {int(k.max())}: active units would be dropped")
    return max(1, min(hidden, int(width)))


# ---------------------------------------------------------------------------- genome
def random_genome(gen, shape, in_dim, hidden, out_dim, initial_hidden, device, innate=None,
                  dtype=torch.float32, params=None):
    """Founder genomes for ``shape`` (e.g. (W, N)). Weight genes are stored in ``dtype``.

    ``eta`` and ``temperature`` start at ``params.eta0`` and ``params.temperature0``.
    ``innate`` (founders' only innate wiring) is a dict with ``plant_index`` (the local plant
    input) and optional ``forage_out``, ``forage_bias``, ``plant_gain``, ``relay_gain``,
    ``relay_unit``: the forage logit gets a bias and a positive path from the plant input
    through one clean relay unit (its other input, recurrent and output weights are zero).
    """
    p = brain_params(params)
    shape = tuple(shape)
    if not 2 <= initial_hidden <= hidden:
        raise ValueError("initial_hidden must lie within 2 and hidden")
    i, h, o, k0 = in_dim, hidden, out_dim, initial_hidden

    def normal(*dims, scale):
        return torch.randn(*shape, *dims, generator=gen, device=device) * scale

    g = {
        "Wx": normal(i, h, scale=INIT_GAIN["Wx"] / i ** .5),
        "Wh": normal(h, h, scale=INIT_GAIN["Wh"] / k0 ** .5),
        "b": torch.zeros(*shape, h, device=device),
        "Wo": normal(h, o, scale=INIT_GAIN["Wo"] / k0 ** .5),
        "bo": torch.zeros(*shape, o, device=device),
    }
    if innate is not None:
        spec = {**INNATE_DEFAULT, "forage_out": FORAGE_OUT, **innate}
        u, plant, f = int(spec["relay_unit"]), int(spec["plant_index"]), int(spec["forage_out"])
        if not (0 <= u < 2 and 0 <= plant < i and 0 <= f < o):
            raise ValueError("innate indices out of range (relay unit must be 0 or 1, always active)")
        g["Wx"][..., :, u] = 0.
        g["Wx"][..., plant, u] = spec["plant_gain"]
        g["Wh"][..., :, u] = 0.
        g["Wh"][..., u, :] = 0.
        g["Wo"][..., u, :] = 0.
        g["Wo"][..., u, f] = spec["relay_gain"]
        g["bo"][..., f] = spec["forage_bias"]
    g = {name: t.to(dtype) for name, t in g.items()}
    g["k"] = torch.full(shape, k0, dtype=torch.long, device=device)
    g["eta"] = torch.full(shape, p.eta0, device=device)
    g["temperature"] = torch.full(shape, p.temperature0, device=device)
    return g


def init_live(genome, dtype=torch.float32):
    """Live recurrent weights at birth: a copy of the genome's Wh (never inherited learning)."""
    return genome["Wh"].to(dtype, copy=True)


def init_state(shape, hidden, device):
    return torch.zeros(*shape, hidden, device=device)


def rows(genome, *index):
    """Genome rows at an index (e.g. world and slot index tensors) -> dict of [n, ...] tensors."""
    return {name: t[index] for name, t in genome.items()}


def install(genome, wh_live, state, w, c, child, baseline=None):
    """Write child genomes into slots (w, c): every gene, live weights from the child's Wh, zero state.

    ``child`` must hold exactly the genome's genes (KeyError otherwise), so no slot keeps a gene
    of its previous occupant. ``baseline`` (the modulator's running mean), if given, is zeroed.
    """
    if set(child) != set(genome):
        raise KeyError(f"child genes differ from the genome: missing {sorted(set(genome) - set(child))}, "
                       f"extra {sorted(set(child) - set(genome))}")
    for name in genome:
        genome[name][w, c] = child[name].to(genome[name].dtype)
    wh_live[w, c] = child["Wh"].to(wh_live.dtype)
    state[w, c] = 0.
    if baseline is not None:
        baseline[w, c] = 0.


def genome_bytes(genome):
    return sum(t.numel() * t.element_size() for t in genome.values())


# ---------------------------------------------------------------------------- one step
def think(genome, wh_live, state, x, hidden=None, *, width=None, chunk=None):
    """One step for every slot: (outputs [..., O], new hidden state [..., H]) in float32.

    new = tanh(x Wx + h Wh_live + b) on units < k (exactly zero beyond), out = new Wo + bo.
    Only the first ``max(k)`` units are evaluated; a caller that knows that bound may pass it as
    ``width`` to skip the one host sync (it must be at least ``max(k)``: checked when ``k`` is on
    the CPU, the caller's duty on an accelerator, so update a cached width after every install).
    Weights stored in a narrower dtype are cast to float32 per chunk of ``chunk`` individuals
    (default 4,096), so the float32 copies never exceed one chunk.
    """
    H = genome["Wh"].shape[-1]
    if hidden is not None and hidden != H:
        raise ValueError(f"hidden {hidden} does not match the genome ({H})")
    lead, I = x.shape[:-1], x.shape[-1]
    B = _numel(lead)
    O = genome["Wo"].shape[-1]
    k = genome["k"].reshape(B)
    He = _width(k, H, width)
    Wx = genome["Wx"].reshape(B, I, H)
    Wh = wh_live.reshape(B, H, H)
    Wo = genome["Wo"].reshape(B, H, O)
    b, bo = genome["b"].reshape(B, H), genome["bo"].reshape(B, O)
    narrow = any(t.dtype != torch.float32 for t in (Wx, Wh, Wo, b, bo))
    step = max(1, B) if (chunk is None and not narrow) else max(1, chunk or 4096)
    mask = unit_mask(k, He)
    xs = _f32(x).reshape(B, 1, I)
    h = _f32(state).reshape(B, H)[:, :He] * mask
    new = torch.zeros(B, H, device=x.device)
    out = torch.empty(B, O, device=x.device)
    for s in range(0, B, step):
        e = min(B, s + step)
        pre = torch.bmm(xs[s:e], _f32(Wx[s:e, :, :He])) \
            + torch.bmm(h[s:e, None], _f32(Wh[s:e, :He, :He]))
        hn = torch.tanh(pre[:, 0] + _f32(b[s:e, :He])) * mask[s:e]
        new[s:e, :He] = hn
        out[s:e] = torch.bmm(hn[:, None], _f32(Wo[s:e, :He]))[:, 0] + _f32(bo[s:e])
    return out.reshape(*lead, O), new.reshape(*lead, H)


def _scaled(logits, temperature_gene, allowed):
    """logits / temperature with disallowed entries at -inf; a row with nothing allowed allows only 0 (rest)."""
    z = _f32(logits) / _f32(temperature_gene)[..., None]
    if allowed is not None:
        first = torch.arange(z.shape[-1], device=z.device) == 0
        allowed = allowed | (first & ~allowed.any(-1, keepdim=True))
        z = z.masked_fill(~allowed, float("-inf"))
    return z


def sample_actions(logits, temperature_gene, gen, allowed=None):
    """Draw one index per row from softmax(logits / temperature) -> long [...].

    Gumbel-max with the given generator: exact softmax sampling, vectorised, reproducible on
    the CPU. ``allowed`` (bool [..., A]) removes actions; a row with none allowed returns 0.
    """
    z = _scaled(logits, temperature_gene, allowed)
    u = torch.rand(z.shape, generator=gen, device=z.device).clamp_min(torch.finfo(torch.float32).tiny)
    return (z - torch.log(-torch.log(u))).argmax(-1)


def action_probs(logits, temperature_gene, allowed=None):
    """softmax(logits / temperature) over the allowed entries; a row with none allowed is all on 0."""
    return torch.softmax(_scaled(logits, temperature_gene, allowed), -1)


def decode(out, layout=LAYOUT):
    """Squash raw outputs: turn in [-1, 1], speed and loudness in [0, 1], vocal in [-1, 1]^D, logits."""
    return {"turn": torch.tanh(out[..., layout.turn]),
            "speed": torch.sigmoid(out[..., layout.speed]),
            "loud": torch.sigmoid(out[..., layout.loud]),
            "vocal": torch.tanh(out[..., layout.vocal]),
            "action_logits": out[..., layout.action],
            "collect_logits": out[..., layout.collect]}


def act(out, genome, gen, layout=LAYOUT, allowed=None, allowed_class=None):
    """decode() plus the sampled ``action`` and ``collect_class`` (two draws, in that order)."""
    d = decode(out, layout)
    d["action"] = sample_actions(d["action_logits"], genome["temperature"], gen, allowed)
    d["collect_class"] = sample_actions(d["collect_logits"], genome["temperature"], gen, allowed_class)
    return d


# ---------------------------------------------------------------------------- learning within a life
def basal_j_day(mass_kg):
    """Basal energy over one day: 3.4 W x M^0.75 x 86,400 s (Kleiber 1947; PLANET-SPEC 2.8)."""
    return KLEIBER_W * mass_kg.clamp_min(1e-6) ** 0.75 * DAY_S


def outcome(d_reserve_j, d_water_kg, mass_kg, d_health, *, invested_j=None, basal_j=None, water_margin_kg=None):
    """The individual's own outcome over a day (PLANET-SPEC 2.10, reserve term per basal day).

    (d_reserve + invested) / basal + 10 d_water / M + 8 d_health. ``invested_j`` is energy the
    reserve paid into offspring tissue (or own growth) that day; it is added back so a birth is
    not felt as a loss. ``basal_j`` defaults to :func:`basal_j_day` of ``mass_kg``, so a fasting
    day is about -1 and a full grazing day about +2.8 at any body mass. ``d_reserve_j`` is the
    change of body energy the caller counts (the world passes the reserve change minus the lean
    tissue burnt, so starving at an empty reserve is felt).

    With ``water_margin_kg`` (the body's lethal water margin, e.g. 0.2 x the norm) the water term
    is OUTCOME_HEALTH x d_water / margin instead (PROVENANCE['outcome_water_margin']): losing the
    whole margin counts like losing all health, both being death.
    """
    if invested_j is not None:
        d_reserve_j = d_reserve_j + invested_j
    basal = basal_j_day(mass_kg) if basal_j is None else basal_j
    if water_margin_kg is None:
        water = OUTCOME_WATER * d_water_kg / mass_kg.clamp_min(1e-6)
    else:
        water = OUTCOME_HEALTH * d_water_kg / torch.as_tensor(water_margin_kg).clamp_min(1e-9)
    return d_reserve_j / basal + water + OUTCOME_HEALTH * d_health


def modulate(reward, baseline, alive, rate=BrainParams.baseline_rate):
    """(modulator, new baseline): tanh(outcome - running mean); dead slots get 0 and a reset mean."""
    mod = torch.tanh(reward - baseline) * alive
    base = torch.where(alive, baseline + rate * (reward - baseline), torch.zeros_like(baseline))
    return mod, base


def to_bf16_stochastic(x, gen):
    """float32 -> bfloat16 with stochastic rounding (unbiased: E[result] = x), drawn from ``gen``."""
    bits = _f32(x).contiguous().view(torch.int32)
    noise = torch.randint(0, 1 << 16, bits.shape, generator=gen, device=bits.device, dtype=torch.int32)
    return ((bits + noise) & -65536).view(torch.float32).to(torch.bfloat16)


def hebbian(wh_live, eta, modulator, before, after, alive, clip, *, k=None, width=None, gen=None, chunk=4096):
    """Reward-modulated Hebbian change of the live recurrent weights (in place), as the flat brain.

    dW[i, j] = eta x modulator x before[i] x after[j] for living slots, then clipped. Masked
    units are zero in ``before`` and ``after`` so their weights never change; ``k`` or ``width``
    (as in :func:`think`; a known width skips the host sync) limits the work to the first max(k)
    units. Live weights are float32 or bfloat16; bfloat16 is updated in float32 per chunk and
    rounded back stochastically with ``gen``, which is then required.
    """
    if wh_live.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(f"live weights must be float32 or bfloat16, not {wh_live.dtype}")
    if wh_live.dtype == torch.bfloat16 and gen is None:
        raise ValueError("bfloat16 live weights need gen: nearest rounding would lose small updates")
    H = wh_live.shape[-1]
    B = wh_live.numel() // (H * H)
    He = _width(None if k is None else k.reshape(-1), H, width)
    wh = wh_live.view(B, H, H)
    gain = (_f32(eta) * _f32(modulator) * alive.float()).reshape(B, 1)
    pre = _f32(before).reshape(B, H)[:, :He, None]
    post = (_f32(after).reshape(B, H)[:, :He] * gain)[:, None, :]
    if wh.dtype == torch.float32:
        wh[:, :He, :He].addcmul_(pre, post).clamp_(-clip, clip)   # in place on the active block
        return wh_live
    chunk = max(1, int(chunk))
    for s in range(0, B, chunk):
        e = min(B, s + chunk)
        w32 = wh[s:e, :He, :He].float().addcmul_(pre[s:e], post[s:e]).clamp_(-clip, clip)
        wh[s:e, :He, :He] = to_bf16_stochastic(w32, gen)
    return wh_live


# ---------------------------------------------------------------------------- inheritance
def mutate(gen, parent, params=None, specs=None):
    """Child genome rows from parent genome rows (leading batch of children).

    The parent's genome is the only input, so nothing learned is inherited. Weight genes get
    N(0, sd) with sd = weight_mutation_rel x the founder sd at the parent's fan-in (Wx
    1/sqrt(in_dim), Wh 0.5/sqrt(k), Wo 1/sqrt(k)), biases N(0, bias_mutation_sd); all are bounded
    by +-weight_clip. ``k`` moves by exactly +-1 with probability ``hidden_mutation_rate`` within
    [2, min(hidden, the genome's units)]; every other gene follows its :class:`GeneSpec` in
    ``specs`` (default BRAIN_SPECS; pass ``{**BRAIN_SPECS, **body}``). An unknown gene raises, so
    no gene silently stops evolving. Draw order is fixed.
    """
    p = brain_params(params)
    specs = BRAIN_SPECS if specs is None else specs
    unknown = sorted(n for n in parent if n not in WEIGHTS and n != "k" and n not in specs)
    if unknown:
        raise KeyError(f"genes without a GeneSpec: {unknown}")

    def noise(t, scale):
        return torch.randn(t.shape, generator=gen, device=t.device) * scale

    k = parent["k"]
    I, H = parent["Wx"].shape[-2], parent["Wh"].shape[-1]
    root_k = k.clamp(2, H).to(device=k.device, dtype=torch.float32).sqrt()[..., None, None]
    m = p.weight_mutation_rel
    sd = {"Wx": m * INIT_GAIN["Wx"] / I ** .5, "Wh": m * INIT_GAIN["Wh"] / root_k,
          "Wo": m * INIT_GAIN["Wo"] / root_k, "b": p.bias_mutation_sd, "bo": p.bias_mutation_sd}
    child = {}
    for name in WEIGHTS:
        t = parent[name]
        child[name] = (_f32(t) + noise(t, sd[name])).clamp_(-p.weight_clip, p.weight_clip).to(t.dtype)
    step = torch.where(torch.rand(k.shape, generator=gen, device=k.device) < .5, -1, 1)
    change = torch.rand(k.shape, generator=gen, device=k.device) < p.hidden_mutation_rate
    child["k"] = (k + step * change).clamp(2, min(p.hidden, H)).to(k.dtype)
    for name in sorted(n for n in parent if n in specs):
        t, s = parent[name], specs[name]
        if s.kind == "abs":
            v = _f32(t) + noise(t, s.scale)
        elif s.kind == "range":
            v = _f32(t) + noise(t, s.scale * (s.hi - s.lo))
        elif s.kind == "log":
            v = _f32(t) * torch.exp(noise(t, s.scale))
        elif s.kind == "step":
            up = torch.where(torch.rand(t.shape, generator=gen, device=t.device) < .5, -1, 1)
            hit = torch.rand(t.shape, generator=gen, device=t.device) < s.scale
            v = t + up * hit
        else:
            raise ValueError(f"unknown GeneSpec kind {s.kind!r} for {name}")
        child[name] = v.clamp(s.lo, s.hi).to(t.dtype)
    return child
