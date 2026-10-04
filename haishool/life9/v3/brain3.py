"""Brains and genes of v3 bodies: nothing wired in, learning only if evolution switches it on (PLANET-V3-SPEC 5, 6).

The network is version 2's batched recurrent brain (``planet.brain``: only its network, live-weight and
storage helpers are imported, never edited):

    h' = tanh(x Wx + h Wh_live + b) on the active units 0..k-1 (exactly zero beyond), out = h' Wo + bo

What v3 changes:

* Founders get small random weights and zero biases. No input is wired to any output, so no motor
  output is favoured: over random genomes every raw output is symmetric about 0, i.e. at the middle
  of its physical range after the squash (:func:`decode`).
* The outputs are the body's physical abilities, continuous and physically scaled (:class:`Layout3`):
  turn, thrust, mouth, grip_left, grip_right, release, force, rub, press, place, loudness, the vocal
  vector [D] and divide. What a body does with them is not supplied; ``body.py`` and
  ``manipulate.py`` turn them into physics, each with its physical cost. Discrete events (divide,
  release, place) fire as Bernoulli draws on the intensity (:func:`fire`).
* Learning within a life is Hebbian on the live recurrent weights, gated by an evolved readout of
  the body's own interoception, m = tanh(sum_i g_i x intero_i). The gains ``g`` and the plasticity
  rate ``eta`` are genes. Founders have g = 0 exactly (and eta at its negligible floor), so m = 0:
  no learning and no notion of good. If learning helps, evolution turns it on and decides what it
  values.
* Variation at division: every float gene takes one Gaussian step in its own coordinate (``lin``:
  the value; ``log``: its logarithm; ``logit``: its log-odds), of size ``mut`` x the gene's unit,
  mirror-reflected into its bounds. The step never depends on the gene's own value, so without
  selection a gene's only drift is toward its stated neutral law, uniform in its coordinate
  (:data:`NEUTRAL_LAW`). ``mut`` itself takes log steps of the fixed size :data:`MUT_TAU`, and the
  child's new ``mut`` scales every other step (self-adaptation). Weights of inactive units never
  mutate; a unit switched on by k + 1 is drawn fresh at the founder scale of the new fan-in.
* :data:`BODY_SPECS3` is the table of section-3 body genes (units, physical bounds, mutation kind,
  founder draw) that ``body.py`` imports; :func:`tissue_shares` is the closure of the body's
  tissue budget; :func:`bound_report` says how much of a population sits on each bound, so no bound
  acts silently.

Shapes: every genome tensor has the population's leading axes (e.g. ``[A, N]`` arena slots, or
``[n]`` rows); there are no Python loops over bodies. Randomness comes only from the generator
passed in, so the CPU is exactly reproducible. State is float32; weight genes may be stored in
bfloat16 (products are float32, as in version 2; mutation rounds them stochastically).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import torch

from ..planet.brain import INIT_GAIN as _V2_INIT_GAIN, WEIGHTS as _V2_WEIGHTS
from ..planet.brain import genome_bytes, init_live, init_state, install, rows  # noqa: F401 (re-exported)
from ..planet.brain import hebbian as _v2_hebbian, think as _v2_think, to_bf16_stochastic as _bf16_stochastic
from ..planet.constants import R_GAS

# ---------------------------------------------------------------------------- motor outputs
VOCAL_DIMS = 8
SCALAR_OUTPUTS = ("turn", "thrust", "mouth", "grip_left", "grip_right", "release", "force", "rub", "press",
                  "place", "loudness")
SIGNED_OUTPUTS = ("turn", "thrust", "vocal")      # tanh to [-1, 1]; every other output is sigmoid to [0, 1]
OUT_TURN, OUT_THRUST, OUT_MOUTH, OUT_GRIP_LEFT, OUT_GRIP_RIGHT, OUT_RELEASE = 0, 1, 2, 3, 4, 5
OUT_FORCE, OUT_RUB, OUT_PRESS, OUT_PLACE, OUT_LOUDNESS = 6, 7, 8, 9, 10


@dataclass(frozen=True)
class Layout3:
    """Where each raw output lives: the 11 scalar abilities (SCALAR_OUTPUTS order), vocal [D], divide."""
    D: int = VOCAL_DIMS

    def index(self, name):
        """Raw output index of a scalar output (or of ``divide``)."""
        if name == "divide":
            return self.divide
        return SCALAR_OUTPUTS.index(name)

    @property
    def vocal(self):
        return slice(len(SCALAR_OUTPUTS), len(SCALAR_OUTPUTS) + self.D)

    @property
    def divide(self):
        return len(SCALAR_OUTPUTS) + self.D

    @property
    def out_dim(self):
        return len(SCALAR_OUTPUTS) + self.D + 1

    @property
    def names(self):
        """Every raw output's name in order (vocal channels as ``vocal0``..)."""
        return (*SCALAR_OUTPUTS, *(f"vocal{i}" for i in range(self.D)), "divide")


LAYOUT3 = Layout3()
OUT_VOCAL, OUT_DIVIDE, OUT_DIM = LAYOUT3.vocal, LAYOUT3.divide, LAYOUT3.out_dim

# ---------------------------------------------------------------------------- interoception read by the modulator
INTERO_NAMES = ("reserve", "water", "body_temp", "damage", "gut", "light", "speed")
N_INTERO = len(INTERO_NAMES)

# ---------------------------------------------------------------------------- network and mutation constants
HIDDEN = 128
K_FOUNDER = (2, 32)
K_MUT_RATE = 0.05
WEIGHT_CLIP = 3.0
BIAS_UNIT = 0.5
WEIGHT_GENES = tuple(_V2_WEIGHTS)         # ("Wx", "Wh", "b", "Wo", "bo"), version 2's weight genes
INIT_GAIN = dict(_V2_INIT_GAIN)           # founder sd = gain / sqrt(fan-in): Wx 1, Wh 0.5, Wo 1

# ---------------------------------------------------------------------------- learning genes
ETA_MIN = 1e-6         # per bout: physically no learning (the founders' value)
ETA_MAX = 0.1          # per bout
G_MAX = 8.0
G_UNIT = 1.0
MUT_BOUNDS = (1e-3, 0.5)
MUT_FOUNDER = (0.01, 0.1)
MUT_TAU = 0.1          # fixed log step of the mutation-scale gene
LOG_UNIT = 1.0         # one e-fold per unit of mut, for every log and logit gene

# ---------------------------------------------------------------------------- body-gene physics used for bounds
M_WATER_KG_MOL = 0.018015                 # kg/mol, H2O
T_SKIN_REF_K = 303.15                     # K, skin temperature at which a vapour conductance is stated
WATER_KG_PER_J = M_WATER_KG_MOL / (R_GAS * T_SKIN_REF_K)   # s^2/m^2: vapour flux per Pa per (s/m) of resistance
SKIN_R_MIN_S_M = 0.1                      # s/m, the skin's own resistance when it is free water
SKIN_PERM_MIN = 1e-12                     # kg m^-2 s^-1 Pa^-1, below the most waterproof cuticles measured
SKIN_PERM_MAX = WATER_KG_PER_J / SKIN_R_MIN_S_M
SETPOINT_BOUNDS_K = (271.0, 318.0)        # K, freezing and protein-denaturation death thresholds (spec 3)
SIZE_BOUNDS_KG = (1e-6, 1e5)
SIZE_FOUNDER_KG = (0.002, 0.2)
THERMO_BOUNDS = (1e-3, 200.0)             # W kg^-1 K^-1; the floor is an ectotherm
FUR_BOUNDS_M = (1e-6, 0.1)
FUR_FOUNDER_M = (1e-5, 2e-3)
MUSCLE_BOUNDS_W_KG = (1.0, 500.0)         # W per kg of muscle
MUSCLE_FOUNDER_W_KG = (10.0, 200.0)
TISSUE_MIN = 1e-6                         # kg per kg body: a tissue that is physically absent
TISSUE_GENES = ("muscle_frac", "enz_plant", "enz_meat", "eye", "ear", "voice")
MUSCLE_FRAC_FOUNDER = (0.05, 0.5)
ENZ_FOUNDER = (1e-3, 0.05)
ORGAN_FOUNDER = (1e-5, 2e-3)
SHARE_BOUNDS = (1e-6, 1.0 - 1e-6)
SHARE_FOUNDER = (0.01, 0.99)


class GeneSpec3(NamedTuple):
    """Units, physical bounds, mutation and founder draw of one float gene.

    Mutation: one Gaussian step in the gene's coordinate c(v), of sd ``scale x unit`` where scale is
    the child's ``mut`` gene (or 1 when ``by_mut`` is False, as for ``mut`` itself), then mirror
    reflection into [c(lo), c(hi)]:

    * ``lin``:   c(v) = v                  neutral law uniform on [lo, hi]
    * ``log``:   c(v) = ln v               neutral law log-uniform on [lo, hi] (needs lo > 0)
    * ``logit``: c(v) = ln(v / (1 - v))    neutral law uniform in log-odds (needs 0 < lo < hi < 1)

    The step does not depend on v, so the reflected walk's stationary law without selection is
    uniform in c (:data:`NEUTRAL_LAW`). Founders: ``fixed`` (founder_lo), ``uniform``,
    ``loguniform`` or ``logituniform`` on [founder_lo, founder_hi]. ``per_intero`` genes carry one
    value per interoceptive channel (the readout gains g).
    """
    lo: float
    hi: float
    kind: str
    unit: float
    founder: str
    founder_lo: float
    founder_hi: float
    units: str
    source: str
    per_intero: bool = False
    by_mut: bool = True


GS3 = GeneSpec3
LEARN_SPECS3 = {
    "eta": GS3(ETA_MIN, ETA_MAX, "log", LOG_UNIT, "fixed", ETA_MIN, ETA_MIN, "per bout",
               "Hebbian rate; founders at the floor ETA_MIN, and their g = 0 makes m = 0 exactly, so they "
               "do not learn (spec 5)"),
    "g": GS3(-G_MAX, G_MAX, "lin", G_UNIT, "fixed", 0.0, 0.0, "per unit of each interoceptive channel",
             "readout gains of m = tanh(sum g_i x intero_i); founders 0 (spec 5)", True),
    "mut": GS3(MUT_BOUNDS[0], MUT_BOUNDS[1], "log", MUT_TAU, "loguniform", MUT_FOUNDER[0], MUT_FOUNDER[1],
               "relative (weights: x founder sd; lin genes: x unit; log and logit genes: e-folds)",
               "heritable mutation scale (spec 6); log step of the fixed size MUT_TAU, never scaled by itself",
               by_mut=False),
}

BODY_SPECS3 = {
    "size_kg": GS3(SIZE_BOUNDS_KG[0], SIZE_BOUNDS_KG[1], "log", LOG_UNIT, "loguniform", *SIZE_FOUNDER_KG, "kg",
                   "lean mass at which growth stops"),
    "fur_m": GS3(FUR_BOUNDS_M[0], FUR_BOUNDS_M[1], "log", LOG_UNIT, "loguniform", *FUR_FOUNDER_M, "m",
                 "insulation thickness; body.py takes the spherical shell resistance (1/r1 - 1/r2)/(4 pi k_fur) "
                 "with r2 = r1 + fur and convection at r2, so thick fur on a small body can lose heat faster "
                 "(critical radius), and charges the fur's keratin mass"),
    "skin_perm": GS3(SKIN_PERM_MIN, SKIN_PERM_MAX, "log", LOG_UNIT, "loguniform", SKIN_PERM_MIN, SKIN_PERM_MAX,
                     "kg m^-2 s^-1 Pa^-1",
                     "the skin's own water-vapour conductance only; body.py puts it in series with the boundary "
                     "layer, g_bl = WATER_KG_PER_J / r_bl with r_bl from the same Nu/Sh correlation as the "
                     "convective h (Lewis analogy): evaporation = A (e_sat(T_b) - e_air) / (1/skin_perm + 1/g_bl)"),
    "thermo_gain": GS3(THERMO_BOUNDS[0], THERMO_BOUNDS[1], "log", LOG_UNIT, "fixed", THERMO_BOUNDS[0],
                       THERMO_BOUNDS[0], "W kg^-1 K^-1",
                       "thermogenesis per kg of body mass per K below the setpoint; the floor is an ectotherm; a "
                       "gain, not a power: body.py caps total heat production by the aerobic capacity (the pO2 "
                       "Michaelis term) and by the fuel available"),
    "setpoint_k": GS3(SETPOINT_BOUNDS_K[0], SETPOINT_BOUNDS_K[1], "lin", SETPOINT_BOUNDS_K[1] - SETPOINT_BOUNDS_K[0],
                      "uniform", *SETPOINT_BOUNDS_K, "K",
                      "body temperature that thermogenesis defends (inert while thermo_gain is at its floor)"),
    "enz_plant": GS3(TISSUE_MIN, 1.0, "log", LOG_UNIT, "loguniform", *ENZ_FOUNDER, "kg per kg body",
                     "gut tissue investing in plant digestion; efficiency enz / (enz + K) in body.py; part of the "
                     "tissue budget (tissue_shares)"),
    "enz_meat": GS3(TISSUE_MIN, 1.0, "log", LOG_UNIT, "loguniform", *ENZ_FOUNDER, "kg per kg body",
                    "gut tissue investing in flesh digestion; efficiency enz / (enz + K) in body.py; part of the "
                    "tissue budget (tissue_shares)"),
    "repair": GS3(0.0, 1.0, "lin", 1.0, "uniform", 0.0, 1.0, "share of fuel",
                  "share of fuel spent repairing damage"),
    "muscle": GS3(MUSCLE_BOUNDS_W_KG[0], MUSCLE_BOUNDS_W_KG[1], "log", LOG_UNIT, "loguniform",
                  *MUSCLE_FOUNDER_W_KG, "W per kg muscle",
                  "peak mechanical power of muscle tissue; P_muscle = muscle x muscle_frac x lean mass"),
    "muscle_frac": GS3(TISSUE_MIN, 1.0, "log", LOG_UNIT, "loguniform", *MUSCLE_FRAC_FOUNDER, "kg per kg body",
                       "muscle mass share of lean mass; maintenance paid on it; part of the tissue budget "
                       "(tissue_shares)"),
    "offspring_share": GS3(SHARE_BOUNDS[0], SHARE_BOUNDS[1], "logit", LOG_UNIT, "logituniform", *SHARE_FOUNDER,
                           "share",
                           "share of the parent's mass and reserve given to the child at division; the child is "
                           "the body with the mutated genome, a zero hidden state and live weights from its "
                           "genome (install), so a share above 1/2 means the fresh body takes most of the mass"),
    "eye": GS3(TISSUE_MIN, 1.0, "log", LOG_UNIT, "loguniform", *ORGAN_FOUNDER, "kg per kg body",
               "eye mass; vision range and sharpness scale with it (saturating optics in senses3)"),
    "ear": GS3(TISSUE_MIN, 1.0, "log", LOG_UNIT, "loguniform", *ORGAN_FOUNDER, "kg per kg body",
               "ear mass; hearing sensitivity scales with it (saturating acoustics in senses3)"),
    "voice": GS3(TISSUE_MIN, 1.0, "log", LOG_UNIT, "loguniform", *ORGAN_FOUNDER, "kg per kg body",
                 "sound-organ mass; the loudest call scales with it"),
}
GENE_SPECS3 = {**LEARN_SPECS3, **BODY_SPECS3}
KINDS = ("lin", "log", "logit")
NEUTRAL_LAW = {"lin": "uniform on [lo, hi]",
               "log": "log-uniform on [lo, hi] (uniform in ln v)",
               "logit": "uniform in ln(v / (1 - v)) on [lo, hi] (symmetric about 1/2)"}
FOUNDER_DRAWS = ("fixed", "uniform", "loguniform", "logituniform")

R_, D_, N_, C_ = "reference", "derived", "new_rule", "chain"
PROVENANCE = {
    # layout
    "VOCAL_DIMS": (N_, "PLANET-V3-SPEC 4: calls are D = 8 dimensional vectors (version 2's D)"),
    "SCALAR_OUTPUTS": (N_, "PLANET-V3-SPEC 0 and 5: the body's physical abilities, in the spec's order; each "
                           "is a continuous intensity, none names a purpose"),
    "SIGNED_OUTPUTS": (N_, "turn (left/right), thrust (along or against the heading: legged bodies can step "
                           "back) and the vocal vector are signed, tanh to [-1, 1]; intensities (mouth opening, "
                           "grip, release, force, rub, press, place, loudness, divide) are sigmoid to [0, 1]. "
                           "Raw 0 is the middle of each range"),
    "OUT_TURN": (N_, "raw output layout: SCALAR_OUTPUTS order 0..10, then vocal [D], then divide"),
    "OUT_THRUST": (N_, "see OUT_TURN"), "OUT_MOUTH": (N_, "see OUT_TURN"),
    "OUT_GRIP_LEFT": (N_, "see OUT_TURN"), "OUT_GRIP_RIGHT": (N_, "see OUT_TURN"),
    "OUT_RELEASE": (N_, "see OUT_TURN"), "OUT_FORCE": (N_, "see OUT_TURN"), "OUT_RUB": (N_, "see OUT_TURN"),
    "OUT_PRESS": (N_, "see OUT_TURN"), "OUT_PLACE": (N_, "see OUT_TURN"), "OUT_LOUDNESS": (N_, "see OUT_TURN"),
    "OUT_VOCAL": (N_, "slice(11, 11 + D)"), "OUT_DIVIDE": (N_, "11 + D"), "OUT_DIM": (N_, "12 + D = 20"),
    "LAYOUT3": (N_, "Layout3(D = VOCAL_DIMS)"),
    "INTERO_NAMES": (N_, "PLANET-V3-SPEC 4 interoception, in the spec's order: reserve / capacity, water / "
                         "normal, body temperature, damage, gut fill, light level, own speed. The contract with "
                         "senses3: dimensionless and of order 1 (open until senses3 guarantees it); the readout "
                         "reads whatever vector it is given"),
    "N_INTERO": (N_, "len(INTERO_NAMES) = 7"),
    "fire": (N_, "a discrete physical event (division, release, placing an item) happens with probability "
                 "equal to its intensity in [0, 1], drawn from the world generator; there is no deterministic "
                 "threshold, so a raw 0 (intensity 1/2) is an even chance, never 'always' or 'never'"),
    # network
    "HIDDEN": (N_, "PLANET-V3-SPEC 5: hidden size 128 (version 2's maximum); k <= HIDDEN is the gene; how "
                   "often k sits at HIDDEN is reported by bound_report"),
    "K_FOUNDER": (N_, "founders' active units drawn uniformly in [2, 32]: small brains for tiny founders "
                      "(spec 6: organ sizes small), varied so selection has something to act on; 2 is version "
                      "2's minimum"),
    "K_MUT_RATE": (N_, "k moves by exactly +-1 with probability 0.05 per division within [2, hidden] (a step "
                       "that would leave the range leaves k where it is): the flat prototype's "
                       "hidden_mutation_rate (haishool/life9/config.py:70), version 2's value. The walk is "
                       "symmetric, so its neutral law is uniform on [2, hidden]"),
    "k": (N_, "active hidden units: founders uniform in K_FOUNDER, +-1 with K_MUT_RATE within [2, hidden], "
              "neutral law uniform; maintenance per unit is paid in body.py"),
    "WEIGHT_CLIP": (N_, "weight genes bounded to +-3 (the flat prototype, config.py:62, version 2's "
                        "weight_clip); weight genes are mirror-reflected into it like every gene (neutral law "
                        "uniform on [-3, 3]); live weights keep version 2's clip in hebbian; founders' draws are "
                        "reflected into it too, so no first mutation shrinks them"),
    "BIAS_UNIT": (D_, "bias genes mutate with sd mut x 0.5: version 2's ratio of its absolute bias step "
                      "(0.05) to its relative weight step (0.1)"),
    "WEIGHT_GENES": (N_, "version 2's weight genes Wx, Wh, b, Wo, bo (planet.brain.WEIGHTS)"),
    "INIT_GAIN": (N_, "founder weights N(0, gain^2 / fan-in) with gains Wx 1, Wh 0.5, Wo 1 (version 2, "
                      "planet.brain.INIT_GAIN; the flat prototype's gains); fan-in of Wh and Wo is the "
                      "founder's own k; drawn values mirror-reflected into +-WEIGHT_CLIP (at k = 2 about 2e-5 "
                      "of Wo entries fold). Biases b and bo start at exactly 0, so no output is favoured"),
    "weight_mutation": (N_, "weights of active units get N(0, (mut x founder sd at the parent's fan-in)^2): Wx "
                            "1/sqrt(in), Wh 0.5/sqrt(k), Wo 1/sqrt(k), so one mutation moves outputs by a "
                            "k-independent fraction (version 2's rule with its fixed 0.1 replaced by the child's "
                            "mut); biases N(0, (mut x BIAS_UNIT)^2); then reflected into +-WEIGHT_CLIP"),
    "inactive_units": (N_, "weights of units >= the parent's k (Wx columns, Wh rows and columns, b, Wo rows) get "
                           "no noise: they are not expressed, so selection cannot hold them, and drifting they "
                           "would switch on later with ever larger weights (a hidden cost on brain growth "
                           "growing with time)"),
    "unit_growth": (N_, "when k grows by one, the new unit (index = the parent's k) is drawn fresh at the founder "
                        "scale for the child's fan-in k': Wx column N(0, 1/in), Wh row and column "
                        "N(0, 0.25/k'), Wo row N(0, 1/k'), bias 0, reflected into +-WEIGHT_CLIP; so a k + 1 step "
                        "has the same effect on the outputs at every generation"),
    "bf16_mutation": (D_, "bfloat16 weight genes are rounded stochastically after the step (version 2's "
                          "to_bf16_stochastic, drawn from the same generator after all weight noise): the stored "
                          "step is unbiased (its mean is the float32 step); a step below one unit in the last "
                          "place (2^-8 of the weight's magnitude) is stored as a 0 or 1 ulp jump with matching "
                          "probability, so its variance is ulp x E|step|, at least the intended variance"),
    "mut_first": (N_, "mut mutates first and the child's own mut scales every other step of that child: "
                      "evolution-strategy self-adaptation (Schwefel 1977, Numerische Optimierung von "
                      "Computer-Modellen; Beyer & Schwefel 2002, Natural Computing 1:3), so a mutation scale is "
                      "tested by the steps it makes"),
    "reflect": (N_, "every float gene and weight is mirror-reflected into its bounds in its mutation coordinate "
                    "(values inside unchanged). Because the step never depends on the gene's own value, the "
                    "reflected walk has, without selection, the uniform law in that coordinate (NEUTRAL_LAW: "
                    "lin uniform, log log-uniform, logit uniform in log-odds) and no other drift; a clamp "
                    "would pile genomes on the bound"),
    "modulator": (N_, "PLANET-V3-SPEC 5: m = tanh(sum_i g_i x intero_i), g genes, founders g = 0"),
    "hebbian": (N_, "version 2's plasticity dW = eta x m x h_before (outer) h_after on the active block, "
                    "clipped to +-WEIGHT_CLIP; m from the evolved readout instead of a supplied signal; before "
                    "and after are masked to each body's own k here, so inactive units never learn whatever "
                    "the caller passes"),
    "tissue_shares": (N_, "closure of the tissue budget, lean = muscle + gut enzyme tissue + eye + ear + voice "
                             "+ brain + rest: each tissue's share is its gene while the asked sum is <= 1 (rest "
                             "= 1 - sum); a genome asking more than the body gets every tissue scaled by 1/sum "
                             "and rest = 0 (mass conservation, no tissue favoured); body.py enforces it and "
                             "bound_report counts how often it binds"),
    "bound_report": (N_, "PLANET-V3-SPEC 6, caps are never silent: the fraction of a population at or within one "
                         "mutation step of each gene's bounds, of k at 2 and at hidden, of active weights on "
                         "the clip, and of tissue budgets that ask for more than the body"),
    # learning genes
    "ETA_MIN": (N_, "1e-6 per bout, physically no learning: over 1e4 bouts (about 7 years at 4 bouts a day) "
                    "full co-activity moves a weight by at most 0.01, a tenth of the founder spread of Wh at "
                    "k = 25 (0.5/5); a log gene needs a positive floor. Founders sit here, and their g = 0 "
                    "makes the modulator exactly 0, so founders do not learn at all"),
    "ETA_MAX": (N_, "upper bound 0.1 per bout: one bout at full modulation then moves a recurrent weight by "
                    "0.1, the founder spread of Wh at k = 25 (0.5/sqrt(k)), so the most plastic brain rewrites "
                    "its recurrent weights within a few bouts; a bound, not a target"),
    "G_MAX": (D_, "|g| <= 8: tanh(8) = 1 - 2.3e-7, so for interoceptive channels of order 1 (the senses3 "
                  "contract) a larger gain changes nothing; whether it binds is reported by bound_report"),
    "G_UNIT": (D_, "1 per unit channel: the gain at which one order-1 channel spans tanh's linear range; a "
                   "lin step of mut x 1 from the founders' exact 0"),
    "MUT_BOUNDS": (N_, "mut in [1e-3, 0.5]: at 1e-3 a weight moves by a thousandth of the founder spread per "
                       "division (bfloat16 storage realises it in expectation through stochastic rounding, see "
                       "bf16_mutation); above 0.5 a child's weights are mostly new draws and heredity is lost"),
    "MUT_FOUNDER": (N_, "founders' mut log-uniform in [0.01, 0.1]: spans version 2's fixed steps (weights 0.1 of "
                        "the founder sd, body genes 0.03); evolution may move it"),
    "MUT_TAU": (N_, "mut' = mut x exp(0.1 N(0, 1)), reflected in log space into MUT_BOUNDS: a fixed log step, "
                    "the order evolution strategies use for one global mutation strength (Beyer & Schwefel "
                    "2002, Natural Computing 1:3: tau = 1/sqrt(2 n) is 0.1 for n = 50 genes). The step does not "
                    "depend on mut, so the neutral law of mut is log-uniform on MUT_BOUNDS (a step equal to mut "
                    "itself would pull mut down: stationary density 1/mut^2 in log space)"),
    "LOG_UNIT": (N_, "log and logit genes step by mut e-folds (mut = 0.03 is a 3 % change per division), the "
                     "same relative meaning mut has for weights (fraction of the founder spread)"),
    # body-gene physics
    "M_WATER_KG_MOL": (R_, "molar mass of water 18.015 g/mol (IUPAC standard atomic weights H 1.008, O 15.999)"),
    "T_SKIN_REF_K": (N_, "30 C, the skin temperature at which a vapour resistance is converted to a conductance "
                         "per Pa"),
    "WATER_KG_PER_J": (D_, "M_w / (R T) at 303.15 K = 7.15e-6 s^2/m^2: vapour flux per Pa of vapour-pressure "
                           "difference for a resistance of 1 s/m (ideal gas)"),
    "SKIN_R_MIN_S_M": (D_, "0.1 s/m: a thirtieth of the smallest boundary-layer resistance inside the size "
                           "bounds (a 1 mg body, d = 1.2 mm, in a 10 m/s wind: g_v = 0.147 sqrt(u/d) "
                           "mol m^-2 s^-1 = 13 mol m^-2 s^-1 = 0.33 m/s, r_bl = 3 s/m; Campbell & Norman 1998, "
                           "An Introduction to Environmental Biophysics, ch. 7), so the most permeable skin is "
                           "free water at every size and wind; the boundary layer itself is body.py's (series)"),
    "SKIN_PERM_MIN": (R_, "1e-12 kg m^-2 s^-1 Pa^-1: a tenth of the most waterproof cuticles measured, desert "
                          "arthropods at 0.5-2 ug cm^-2 h^-1 mmHg^-1 = 1-4e-11 SI (Hadley 1994, Water Relations "
                          "of Terrestrial Arthropods; Edney 1977, Water Balance in Land Arthropods), and below "
                          "the waxiest tetrapod skins (Lillywhite 2006, J. Exp. Biol. 209:202), so the bound "
                          "does not bind before physics does"),
    "SKIN_PERM_MAX": (D_, "WATER_KG_PER_J / SKIN_R_MIN_S_M = 7.15e-5 kg m^-2 s^-1 Pa^-1"),
    "SETPOINT_BOUNDS_K": (R_, "PLANET-V3-SPEC 3 death thresholds: 271 K (body fluids freeze; plasma freezes "
                              "near -0.6 C, with some supercooling) and 318 K (protein denaturation, critical "
                              "thermal maxima of most animals near 45 C)"),
    "SIZE_BOUNDS_KG": (N_, "1 mg to 100 t: below about a milligram the body's physics (legged locomotion, fur, "
                           "convective heat balance) no longer applies; 1e5 kg exceeds the largest land animals "
                           "(sauropods, of order 5e4-7e4 kg)"),
    "SIZE_FOUNDER_KG": (N_, "PLANET-V3-SPEC 6: founders 2-200 g, the chain's bodies era (about 256 cells) "
                            "expressed physically as tiny animals; log-uniform, a starting range, not a target"),
    "THERMO_BOUNDS": (R_, "1e-3 to 200 W/(kg K). Floor: at a 10 K deficit 0.01 W/kg, under 2 % of resting "
                          "skeletal muscle (13 kcal/(kg day) = 0.63 W/kg; Elia 1992, in Energy Metabolism: "
                          "Tissue Determinants and Cellular Corollaries), i.e. an ectotherm. Top: at a 1 K "
                          "deficit it equals the highest sustained whole-body rate measured, hovering "
                          "hummingbirds ~ 40 ml O2/(g h) = 220 W/kg (Suarez 1992, Experientia 48:565); the heat "
                          "itself is capped by body.py's aerobic capacity and fuel, not by this bound"),
    "FUR_BOUNDS_M": (R_, "1 um to 0.1 m. Floor: 1 um of fur (k ~ 0.04 W/(m K)) adds 2.5e-5 m^2 K/W, under 1 % of "
                         "the convective 1/h of the smallest body in the strongest wind (1 mg, d = 1.2 mm, "
                         "10 m/s: Nu = 2 + 0.6 Re^0.5 Pr^(1/3) ~ 17, h ~ 360 W/(m^2 K), 1/h ~ 2.8e-3), i.e. no "
                         "fur. Top: arctic mammal coats insulate up to 5-7 cm (Scholander, Walters, Hock & "
                         "Irving 1950, Biol. Bull. 99:225), a measured record; bound_report shows whether it binds"),
    "FUR_FOUNDER_M": (N_, "founders log-uniform 10 um - 2 mm: from no fur to a small mammal's coat"),
    "MUSCLE_BOUNDS_W_KG": (R_, "1-500 W per kg muscle: peak cyclic power of vertebrate muscle reaches about "
                               "400 W/kg (quail pectoralis at take-off; Askew, Marsh & Ellington 2001, J. Exp. "
                               "Biol. 204:3601), a measured record; 1 W/kg is a muscle that barely works; "
                               "bound_report shows whether the top binds"),
    "MUSCLE_FOUNDER_W_KG": (N_, "founders log-uniform 10-200 W/kg muscle"),
    "TISSUE_MIN": (N_, "1e-6 of body mass: a tissue that is physically absent (an eye of 2 ug on a 2 g body; a "
                       "gut tissue whose efficiency enz / (enz + K) is ~1e-4 for K of order 0.01); a log gene "
                       "needs a positive floor"),
    "TISSUE_GENES": (N_, "the genes that are shares of lean mass and enter the tissue budget (tissue_shares): "
                         "muscle_frac, enz_plant, enz_meat, eye, ear, voice. Each is bounded above by 1, the "
                         "whole body (mass conservation), not by a measured record"),
    "MUSCLE_FRAC_FOUNDER": (N_, "founders log-uniform 5-50 % muscle (insects to mammals carry about 10-50 %)"),
    "ENZ_FOUNDER": (N_, "founders log-uniform 0.1-5 % digestive tissue (ICRP 23 Reference Man: GI tract "
                        "tissue 1.2 kg of 70 kg, 1.7 %)"),
    "ORGAN_FOUNDER": (N_, "PLANET-V3-SPEC 6 organ sizes small: founders log-uniform 1e-5 to 0.2 % of body mass "
                          "(a mouse's eyes are about 0.1 %)"),
    "SHARE_BOUNDS": (D_, "[1e-6, 1 - 1e-6]: symmetric, so parent and child sides are open alike; 1 - 1e-6 is "
                         "about the closest share to 1 whose complement float32 still resolves to 6 % (spacing "
                         "6e-8 near 1). The floor lets a 1 t body make a 1 g child; bound_report shows whether "
                         "it binds"),
    "SHARE_FOUNDER": (N_, "founders logit-uniform on [0.01, 0.99]: drawn broadly, symmetric between small and "
                          "large children"),
    # body genes (GeneSpec3 names): bounds, kind, founder, neutral law
    "size_kg": (N_, "bounds SIZE_BOUNDS_KG, log step mut, neutral law log-uniform; founders SIZE_FOUNDER_KG"),
    "fur_m": (N_, "bounds FUR_BOUNDS_M, log step mut, neutral law log-uniform; founders FUR_FOUNDER_M"),
    "skin_perm": (D_, "bounds SKIN_PERM_MIN .. SKIN_PERM_MAX = 1e-12 .. 7.15e-5 kg m^-2 s^-1 Pa^-1 (the skin "
                      "alone; the boundary layer in series is body.py's); log step, neutral law log-uniform; "
                      "founders log-uniform over the whole range (drawn broadly)"),
    "thermo_gain": (N_, "bounds THERMO_BOUNDS, log step mut, neutral law log-uniform; founders at the floor: "
                        "every founder is an ectotherm, endothermy must evolve (spec 3)"),
    "setpoint_k": (N_, "bounds SETPOINT_BOUNDS_K, lin step mut x 47 K (the bounds' span), neutral law "
                       "uniform; founders uniform over the bounds (inert while thermo_gain is at its floor)"),
    "enz_plant": (N_, "bounds [TISSUE_MIN, 1], log step mut, neutral law log-uniform; founders ENZ_FOUNDER"),
    "enz_meat": (N_, "as enz_plant"),
    "repair": (N_, "a share in [0, 1]; lin step mut x 1, neutral law uniform; founders uniform over [0, 1]"),
    "muscle": (N_, "bounds MUSCLE_BOUNDS_W_KG, log step mut, neutral law log-uniform; founders "
                   "MUSCLE_FOUNDER_W_KG; power = muscle x muscle_frac x lean mass"),
    "muscle_frac": (N_, "bounds [TISSUE_MIN, 1], log step mut, neutral law log-uniform; founders "
                        "MUSCLE_FRAC_FOUNDER; with muscle it gives P_muscle, so no fixed muscle share is supplied"),
    "offspring_share": (N_, "bounds SHARE_BOUNDS, logit step mut, neutral law uniform in log-odds (symmetric "
                            "about 1/2); founders SHARE_FOUNDER. Not capped at 1/2: the child is not the "
                            "parent relabelled (it alone is mutated and starts unlearned), so budding, "
                            "rejuvenation and putting almost everything into one offspring stay open"),
    "eye": (N_, "bounds [TISSUE_MIN, 1] (mass conservation), log step mut, neutral law log-uniform; founders "
                "ORGAN_FOUNDER"),
    "ear": (N_, "as eye: its own bound is mass conservation, not another organ's record"),
    "voice": (N_, "as eye: its own bound is mass conservation, not another organ's record"),
    "eta": (N_, "bounds [ETA_MIN, ETA_MAX], log step mut, neutral law log-uniform; founders at ETA_MIN "
                "(spec 5: no learning)"),
    "g": (N_, "bounds [-G_MAX, G_MAX], lin step mut x G_UNIT, neutral law uniform, one gain per interoceptive "
              "channel; founders exactly 0 (spec 5)"),
    "mut": (N_, "bounds MUT_BOUNDS, log step MUT_TAU (fixed), neutral law log-uniform; founders MUT_FOUNDER"),
    # tables
    "LEARN_SPECS3": (N_, "the learning genes eta, g and mut (spec 5, 6); each entry has its own provenance"),
    "BODY_SPECS3": (N_, "the section-3 body genes with units, bounds, mutation kind and founder draw, plus "
                        "muscle_frac (muscle mass, so P_muscle needs no fixed share); each entry has its own "
                        "provenance"),
    "GENE_SPECS3": (N_, "LEARN_SPECS3 and BODY_SPECS3: every float gene a v3 genome carries"),
    "KINDS": (N_, "GeneSpec3 mutation coordinates: lin (v), log (ln v), logit (ln(v / (1 - v)))"),
    "NEUTRAL_LAW": (D_, "a symmetric step of value-independent size, reflected at both bounds, has the uniform "
                        "law in its coordinate as its stationary distribution (the transition kernel is "
                        "symmetric, hence doubly stochastic)"),
    "FOUNDER_DRAWS": (N_, "GeneSpec3 founder draws: fixed, uniform, loguniform, logituniform"),
}


# ---------------------------------------------------------------------------- helpers
def _f32(t):
    return t if t.dtype == torch.float32 else t.float()


def _trail(t, like):
    """``t`` (leading batch shape) viewed with trailing singleton axes to broadcast against ``like``."""
    return t.reshape(*t.shape, *([1] * (like.dim() - t.dim())))


def reflect(v, lo, hi):
    """Mirror-reflect ``v`` into [lo, hi]; values already inside are returned unchanged (exactly)."""
    if hi <= lo:
        raise ValueError(f"empty bounds [{lo}, {hi}]")
    w = hi - lo
    y = torch.remainder(v - lo, 2.0 * w)
    folded = lo + torch.where(y > w, 2.0 * w - y, y)
    return torch.where((v >= lo) & (v <= hi), v, folded.clamp(lo, hi))


def _reflect_(v, lo, hi, chunk=1 << 22):
    """:func:`reflect` in place on a contiguous tensor, in chunks of elements (bounded temporaries)."""
    flat = v.view(-1)
    for s in range(0, flat.numel(), chunk):
        part = flat[s:s + chunk]
        if part.device.type != "cpu" or bool(((part < lo) | (part > hi)).any()):   # inside values are kept
            flat[s:s + chunk] = reflect(part, lo, hi)
    return v


def to_coord(kind, v):
    """A gene's mutation coordinate: v (lin), ln v (log) or ln(v / (1 - v)) (logit)."""
    if kind == "lin":
        return v
    if kind == "log":
        return torch.log(v)
    if kind == "logit":
        return torch.log(v) - torch.log1p(-v)
    raise ValueError(f"unknown GeneSpec3 kind {kind!r}")


def from_coord(kind, c):
    """Inverse of :func:`to_coord`."""
    if kind == "lin":
        return c
    if kind == "log":
        return torch.exp(c)
    if kind == "logit":
        return torch.sigmoid(c)
    raise ValueError(f"unknown GeneSpec3 kind {kind!r}")


def coord_bounds(spec):
    """(c(lo), c(hi)) of a GeneSpec3 in its mutation coordinate, as Python floats."""
    def c(x):
        if spec.kind == "lin":
            return float(x)
        if spec.kind == "log":
            return math.log(x)
        if spec.kind == "logit":
            return math.log(x) - math.log1p(-x)
        raise ValueError(f"unknown GeneSpec3 kind {spec.kind!r}")
    return c(spec.lo), c(spec.hi)


def _draw_founder(gen, spec, shape, device):
    if spec.founder == "fixed":
        return torch.full(shape, float(spec.founder_lo), device=device)
    u = torch.rand(shape, generator=gen, device=device)
    if spec.founder == "uniform":
        return spec.founder_lo + (spec.founder_hi - spec.founder_lo) * u
    if spec.founder == "loguniform":
        a, b = math.log(spec.founder_lo), math.log(spec.founder_hi)
        return torch.exp(a + (b - a) * u)
    if spec.founder == "logituniform":
        a = math.log(spec.founder_lo) - math.log1p(-spec.founder_lo)
        b = math.log(spec.founder_hi) - math.log1p(-spec.founder_hi)
        return torch.sigmoid(a + (b - a) * u)
    raise ValueError(f"unknown founder draw {spec.founder!r}")


def check_specs(specs):
    """Raise if a GeneSpec3 table is inconsistent (bounds, kinds, founder ranges inside the bounds)."""
    for name, s in specs.items():
        if not s.lo < s.hi:
            raise ValueError(f"{name}: lo must be below hi")
        if s.kind not in KINDS or s.founder not in FOUNDER_DRAWS:
            raise ValueError(f"{name}: unknown kind {s.kind!r} or founder draw {s.founder!r}")
        if s.kind == "log" and s.lo <= 0:
            raise ValueError(f"{name}: a log gene needs lo > 0")
        if s.kind == "logit" and not (0 < s.lo and s.hi < 1):
            raise ValueError(f"{name}: a logit gene needs 0 < lo < hi < 1")
        if not s.unit > 0:
            raise ValueError(f"{name}: unit must be positive")
        if not s.lo <= s.founder_lo <= s.founder_hi <= s.hi:
            raise ValueError(f"{name}: founder range must lie within the bounds")
        if s.founder == "loguniform" and s.founder_lo <= 0:
            raise ValueError(f"{name}: a log-uniform founder draw needs founder_lo > 0")
        if s.founder == "logituniform" and not (0 < s.founder_lo and s.founder_hi < 1):
            raise ValueError(f"{name}: a logit-uniform founder draw needs 0 < founder_lo, founder_hi < 1")
        if not isinstance(s.by_mut, bool) or not isinstance(s.per_intero, bool):
            raise ValueError(f"{name}: by_mut and per_intero must be bool")


check_specs(GENE_SPECS3)


# ---------------------------------------------------------------------------- genome
def random_body_genes(gen, shape, device="cpu", specs=None, n_intero=N_INTERO):
    """Founder values of every gene in ``specs`` (default GENE_SPECS3), float32, in sorted name order."""
    specs = GENE_SPECS3 if specs is None else specs
    check_specs(specs)
    shape = tuple(shape)
    out = {}
    for name in sorted(specs):
        s = specs[name]
        out[name] = _draw_founder(gen, s, shape + ((n_intero,) if s.per_intero else ()), device)
    return out


def random_genome3(gen, shape, in_dim, *, hidden=HIDDEN, layout=LAYOUT3, n_intero=N_INTERO, k_range=K_FOUNDER,
                   device="cpu", dtype=torch.float32, specs=None):
    """Founder genomes for ``shape`` (e.g. (A, N)): brain weights, ``k`` and every gene of ``specs``.

    Draw order (fixed): k uniform in ``k_range``; Wx, Wh, Wo ~ N(0, gain^2 / fan-in) with the
    founder's own k as the fan-in of Wh and Wo, mirror-reflected into +-WEIGHT_CLIP; b and bo
    exactly 0; then the genes of ``specs`` (default GENE_SPECS3: eta at its floor, g = 0, mut, and
    the section-3 body genes) in sorted order. Weight genes are stored in ``dtype`` (nearest
    rounding: fresh draws carry no step to bias); other genes are float32 and ``k`` is long.
    """
    shape = tuple(shape)
    k_lo, k_hi = int(k_range[0]), int(k_range[1])
    if not 2 <= k_lo <= k_hi <= hidden:
        raise ValueError(f"k_range {k_range} must lie within [2, hidden = {hidden}]")
    i, h, o = int(in_dim), int(hidden), layout.out_dim
    k = torch.randint(k_lo, k_hi + 1, shape, generator=gen, device=device)
    root_k = k.to(torch.float32).sqrt()[..., None, None]

    def normal(*dims, sd):
        t = torch.randn(*shape, *dims, generator=gen, device=device).mul_(sd)
        return _reflect_(t, -WEIGHT_CLIP, WEIGHT_CLIP)

    g = {
        "Wx": normal(i, h, sd=INIT_GAIN["Wx"] / i ** .5),
        "Wh": normal(h, h, sd=INIT_GAIN["Wh"] / root_k),
        "b": torch.zeros(*shape, h, device=device),
        "Wo": normal(h, o, sd=INIT_GAIN["Wo"] / root_k),
        "bo": torch.zeros(*shape, o, device=device),
    }
    g = {name: t.to(dtype) for name, t in g.items()}
    g["k"] = k
    g.update(random_body_genes(gen, shape, device, specs, n_intero))
    return g


# ---------------------------------------------------------------------------- one bout
def _active(k, H):
    """float mask [..., H]: 1 on each body's active units 0..k-1."""
    return (torch.arange(H, device=k.device) < k[..., None]).to(torch.float32)


def think(genome, wh_live, state, x, *, width=None, chunk=None):
    """(raw outputs [..., OUT_DIM], new hidden state [..., H]) in float32, version 2's network.

    Units >= k are exactly zero and never matter. ``width`` (>= max k) skips the host sync.
    """
    return _v2_think(genome, wh_live, state, x, width=width, chunk=chunk)


def decode(out, layout=LAYOUT3):
    """Raw outputs -> physically scaled motor commands (a dict of float32 tensors).

    turn, thrust in [-1, 1] (tanh); mouth, grip_left, grip_right, release, force, rub, press,
    place, loudness and divide in [0, 1] (sigmoid); vocal in [-1, 1]^D (tanh); ``grip`` stacks
    [left, right] on a last axis of 2. A raw 0 is the middle of every range.
    """
    out = _f32(out)
    sig, tan = torch.sigmoid(out), torch.tanh(out)
    d = {name: (tan if name in SIGNED_OUTPUTS else sig)[..., i] for i, name in enumerate(SCALAR_OUTPUTS)}
    d["vocal"] = tan[..., layout.vocal]
    d["divide"] = sig[..., layout.divide]
    d["grip"] = torch.stack((d["grip_left"], d["grip_right"]), -1)
    return d


def fire(intent, gen):
    """A discrete physical event from an intensity in [0, 1]: Bernoulli(intent) drawn from ``gen``.

    The only mode (a draw, never a threshold). The world's bodies use rates instead: body.py's division is a
    development rate, and manipulate.bout's grips, press, place and release are continuous-time events
    (manipulate.EVENT_RULE); manipulate.events falls back to this draw when it is given no bout length.
    """
    if not isinstance(gen, torch.Generator):
        raise TypeError("fire needs the world's torch.Generator: events are Bernoulli draws, never a threshold")
    return torch.rand(intent.shape, generator=gen, device=intent.device) < intent


# ---------------------------------------------------------------------------- learning within a life
def modulator(g, intero):
    """m = tanh(sum_i g_i x intero_i) over the last axis: the evolved readout of the body's own state."""
    return torch.tanh((_f32(g) * _f32(intero)).sum(-1))


def hebbian(wh_live, genome, before, after, intero, alive, *, clip=WEIGHT_CLIP, width=None, gen=None, chunk=4096):
    """Modulated Hebbian change of the live recurrent weights (in place); returns the modulator m.

    dW[i, j] = eta x m x before[i] x after[j] on each body's active block (before and after are
    masked to units < k here), for living slots, clipped to +-clip, with m = :func:`modulator`
    (genome['g'], intero). Founders (g = 0) never change. bfloat16 live weights need ``gen``
    (stochastic rounding, version 2).
    """
    m = modulator(genome["g"], intero)
    act = _active(genome["k"], wh_live.shape[-1])
    _v2_hebbian(wh_live, genome["eta"], m, _f32(before) * act, _f32(after) * act, alive, clip, k=genome["k"],
                width=width, gen=gen, chunk=chunk)
    return m


# ---------------------------------------------------------------------------- inheritance
def _gene_step(spec, v, z, scale):
    """One mutation of a float gene: a step of sd scale x unit in its coordinate, reflected into its bounds."""
    c_lo, c_hi = coord_bounds(spec)
    c = to_coord(spec.kind, _f32(v).clamp(spec.lo, spec.hi))
    c = reflect(c + z * scale * spec.unit, c_lo, c_hi)
    return from_coord(spec.kind, c).clamp(spec.lo, spec.hi)


def mutate3(gen, parent, *, hidden=None, specs=None, k_rate=K_MUT_RATE, clip=WEIGHT_CLIP):
    """Child genome rows from parent genome rows (any leading batch of children; nothing learned is inherited).

    Draw order (fixed): (1) ``mut`` takes its log step of fixed size MUT_TAU; the child's new mut
    scales every later step. (2) ``k`` moves by +-1 with probability ``k_rate`` within
    [2, min(hidden, H)]. (3) Weights of the parent's active units get N(0, (mut x founder sd at the
    parent's fan-in)^2) (Wx 1/sqrt(in), Wh 0.5/sqrt(k), Wo 1/sqrt(k)), biases N(0, (mut x
    BIAS_UNIT)^2), reflected into +-clip; inactive units' weights are left exactly as they are.
    (4) A unit switched on by k + 1 is drawn fresh at the founder scale of the child's k (bias 0).
    (5) bfloat16 weight genes are rounded stochastically. (6) Every other gene by its
    :class:`GeneSpec3`, in sorted order. An unknown gene raises.
    """
    specs = GENE_SPECS3 if specs is None else specs
    unknown = sorted(n for n in parent if n not in WEIGHT_GENES and n != "k" and n not in specs)
    if unknown:
        raise KeyError(f"genes without a GeneSpec3: {unknown}")
    if "mut" not in parent or "mut" not in specs:
        raise KeyError("the parent (and the spec table) needs a 'mut' gene")
    k = parent["k"]
    lead = tuple(k.shape)
    mut_parent = _f32(parent["mut"])
    if tuple(mut_parent.shape) != lead:
        raise ValueError(f"mut {tuple(mut_parent.shape)} must have the batch shape of k {lead}")
    dev = k.device

    def noise(shape):
        return torch.randn(tuple(shape), generator=gen, device=dev)

    child = {}
    # (1) the mutation scale itself
    s_mut = specs["mut"]
    child["mut"] = _gene_step(s_mut, mut_parent, noise(lead), mut_parent if s_mut.by_mut else 1.0) \
        .to(parent["mut"].dtype)
    mut = _f32(child["mut"])

    # (2) active units
    I, H = parent["Wx"].shape[-2], parent["Wh"].shape[-1]
    O = parent["Wo"].shape[-1]
    top = H if hidden is None else min(int(hidden), H)
    step = torch.where(torch.rand(lead, generator=gen, device=dev) < .5, -1, 1)
    change = torch.rand(lead, generator=gen, device=dev) < k_rate
    k_child = (k + step * change).clamp(2, top).to(k.dtype)

    # (3) weights of the parent's active units
    B = k.numel()
    kf, kc = k.reshape(B), k_child.reshape(B)
    act = _active(kf, H)                                                     # [B, H]
    mutf = mut.reshape(B)
    root_k = kf.clamp(2, H).to(torch.float32).sqrt()
    sd = {"Wx": mutf * (INIT_GAIN["Wx"] / I ** .5), "Wh": mutf * INIT_GAIN["Wh"] / root_k,
          "Wo": mutf * INIT_GAIN["Wo"] / root_k, "b": mutf * BIAS_UNIT, "bo": mutf * BIAS_UNIT}
    expressed = {"Wx": (act[:, None, :],), "Wh": (act[:, :, None], act[:, None, :]), "b": (act,),
                 "Wo": (act[:, :, None],), "bo": ()}
    w = {}
    for name in WEIGHT_GENES:
        t = parent[name]
        tail = tuple(t.shape[len(lead):])
        z = noise((B, *tail))
        for mask in expressed[name]:
            z.mul_(mask)
        z.mul_(sd[name].reshape(B, *([1] * len(tail))))
        z.add_(_f32(t).reshape(B, *tail))
        w[name] = _reflect_(z, -clip, clip)

    # (4) a unit switched on is drawn fresh at the founder scale of the child's fan-in
    grew = kc > kf
    u = kf.clamp(max=H - 1)
    r = torch.arange(B, device=dev)
    fan = kc.clamp(2, H).to(torch.float32).sqrt()[:, None]
    new_x = _reflect_(noise((B, I)).mul_(INIT_GAIN["Wx"] / I ** .5), -clip, clip)
    new_in = _reflect_(noise((B, H)).mul_(INIT_GAIN["Wh"] / fan), -clip, clip)
    new_out = _reflect_(noise((B, H)).mul_(INIT_GAIN["Wh"] / fan), -clip, clip)
    new_o = _reflect_(noise((B, O)).mul_(INIT_GAIN["Wo"] / fan), -clip, clip)
    g1 = grew[:, None]
    w["Wx"][r, :, u] = torch.where(g1, new_x, w["Wx"][r, :, u])
    w["Wh"][r, :, u] = torch.where(g1, new_in, w["Wh"][r, :, u])
    w["Wh"][r, u, :] = torch.where(g1, new_out, w["Wh"][r, u, :])
    w["b"][r, u] = torch.where(grew, torch.zeros_like(w["b"][r, u]), w["b"][r, u])
    w["Wo"][r, u, :] = torch.where(g1, new_o, w["Wo"][r, u, :])

    # (5) storage
    for name in WEIGHT_GENES:
        t = parent[name]
        v = w[name].reshape(t.shape)
        if t.dtype == torch.bfloat16:
            child[name] = _bf16_stochastic(v, gen)
        else:
            child[name] = v.to(t.dtype)
    child["k"] = k_child

    # (6) every other gene
    for name in sorted(n for n in parent if n in specs and n != "mut"):
        t, s = parent[name], specs[name]
        scale = _trail(mut, t) if s.by_mut else 1.0
        child[name] = _gene_step(s, t, noise(t.shape), scale).to(t.dtype)
    return child


# ---------------------------------------------------------------------------- tissue budget
def tissue_shares(genes, brain_frac=None):
    """Shares of lean mass per tissue under the closure lean = muscle + gut + organs + brain + rest.

    ``genes`` holds the :data:`TISSUE_GENES`; ``brain_frac`` (optional, broadcastable) is the brain's
    share that body.py computes from k. While the asked sum is <= 1 each tissue gets its gene and
    ``rest`` = 1 - sum; above 1 every tissue is scaled by 1/sum and rest = 0. Returns a dict of
    float32 tensors: every tissue, ``brain`` (if given), ``rest`` and ``asked`` (the unscaled sum).
    """
    parts = {name: _f32(genes[name]) for name in TISSUE_GENES}
    if brain_frac is not None:
        parts["brain"] = _f32(torch.as_tensor(brain_frac, device=parts["eye"].device))
    asked = sum(parts.values())
    scale = 1.0 / asked.clamp_min(1.0)
    out = {name: p * scale for name, p in parts.items()}
    out["rest"] = (1.0 - asked).clamp_min(0.0)
    out["asked"] = asked
    return out


# ---------------------------------------------------------------------------- bounds that bind
def bound_report(genome, alive=None, *, specs=None, hidden=None, clip=WEIGHT_CLIP, wh_live=None):
    """How much of a population sits on each bound (shares in [0, 1], for analysis; one host sync per entry).

    Keys: ``n`` (bodies counted); ``k@lo`` (k == 2) and ``k@hi`` (k == min(hidden, H)); ``<W>@clip``
    for each weight gene (and ``Wh_live@clip`` if ``wh_live`` is given): the share of the active
    entries with |w| >= clip; ``<gene>@lo`` and ``<gene>@hi``: the share of bodies (of entries, for
    g) within one of their own mutation steps of the bound, in the gene's coordinate;
    ``tissue@full``: the share whose tissue genes ask for more than the body.
    """
    specs = GENE_SPECS3 if specs is None else specs
    k = genome["k"]
    sel = torch.ones(k.shape, dtype=torch.bool, device=k.device) if alive is None else alive.to(torch.bool)
    n = int(sel.sum())
    rep = {"n": n}
    if n == 0:
        return rep

    def share(x):
        return float(x.float().mean()) if x.numel() else 0.0

    kk = k[sel]
    H = genome["Wh"].shape[-1] if "Wh" in genome else int(kk.max())
    top = H if hidden is None else min(int(hidden), H)
    rep["k@lo"], rep["k@hi"] = share(kk <= 2), share(kk >= top)
    if "Wh" in genome:
        act = torch.arange(H, device=k.device) < kk[:, None]
        masks = {"Wx": act[:, None, :], "Wh": act[:, :, None] & act[:, None, :], "b": act,
                 "Wo": act[:, :, None], "bo": None}
        tensors = {name: genome[name] for name in WEIGHT_GENES if name in genome}
        if wh_live is not None:
            tensors["Wh_live"], masks["Wh_live"] = wh_live, masks["Wh"]
        for name, t in tensors.items():
            on = _f32(t[sel]).abs() >= clip
            m = masks[name]
            if m is None:
                rep[f"{name}@clip"] = share(on)
            else:
                m = m.expand_as(on)
                rep[f"{name}@clip"] = float((on & m).sum()) / max(1, int(m.sum()))
    mut = _f32(genome["mut"][sel]) if "mut" in genome else None
    for name in sorted(specs):
        if name not in genome:
            continue
        s = specs[name]
        c = to_coord(s.kind, _f32(genome[name][sel]).clamp(s.lo, s.hi))
        c_lo, c_hi = coord_bounds(s)
        step = s.unit * (_trail(mut, c) if (s.by_mut and mut is not None) else 1.0)
        rep[f"{name}@lo"] = share(c - c_lo <= step)
        rep[f"{name}@hi"] = share(c_hi - c <= step)
    if all(name in genome for name in TISSUE_GENES):
        asked = tissue_shares({name: genome[name][sel] for name in TISSUE_GENES})["asked"]
        rep["tissue@full"] = share(asked > 1.0)
    return rep
