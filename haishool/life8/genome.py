"""Heritable behaviour and life history for life8 organisms (review gap 2 and gap 18).

A genome sits next to the anatomy. It holds:

* ``bias``: innate weights {feature: {action: weight}} over a fixed set of local
  observation features (INNATE_FEATURES: the bias feature, food in reach, food
  direction, energy/health band, neighbour in reach, held tool, and the heard symbol).
  They become the starting weights of a newborn's controller, so they include the
  signal rows and the heard-symbol features. Founders have none.
* ``alpha`` and ``epsilon``: the newborn controller's learning rate and base
  exploration rate (None: the learner profile's default).
* ``lifespan`` and ``maturity`` (ticks; None: Config.max_age / Config.maturity_age).
  They only change, and only matter, with ``Config.age_effects``.

Rules kept from Matrix: learned weights are never inherited (inherit() never reads a
controller, only the parent's genome), there is no bonus for any action, and nothing
here names a meaning for any symbol. Genome mutation draws from its own RNG, seeded
from (world seed, child id), so ``Config.genome="frozen"`` leaves the world RNG stream
and every trajectory exactly as before.

Age curve (``Config.age_effects``): ability(age) multiplies movement, eating
efficiency and strike force. A juvenile starts at 0.5 of its adult ability and grows
linearly until its maturity age; a later maturity gives a stronger adult
(adult factor (maturity / Config.maturity_age) ** 0.25, within [0.8, 1.2]). After
70 % of its lifespan an organism declines linearly to 0.5 at the end of its life.
A longer heritable lifespan costs maintenance: metabolism x (lifespan / max_age) ** 0.5.
"""
from __future__ import annotations

import json
import math
import random

from . import learning

GENOME_VERSION = "life8-genome-v1"
BIAS_BOUND = 3.0
ALPHA_BOUNDS = (.02, 1.)
EPSILON_BOUNDS = (.01, 1.)
LIFESPAN_BOUNDS = (.5, 2.)    # x Config.max_age
MATURITY_BOUNDS = (.4, 2.5)   # x Config.maturity_age
JUVENILE_START = .5
SENESCENCE_ONSET = .7
SENESCENT_END = .5
ADULT_BOUNDS = (.8, 1.2)
GENOME_SALT = "life8-genome"

_DIRECTIONS = ("n", "ne", "e", "se", "s", "sw", "w", "nw", "none")
INNATE_FEATURES = (
    (learning.BIAS_FEATURE, None),
    *(("food_near", value) for value in (True, False)),
    *(("food_direction", value) for value in _DIRECTIONS),
    *(("energy_band", value) for value in ("low", "middle", "high")),
    *(("health_band", value) for value in ("hurt", "well")),
    *(("neighbor_near", value) for value in (True, False)),
    *(("tool_available", value) for value in (True, False)),
    *(("heard_symbol", value) for value in (0, 1, 2, 3)),
)


def feature_key(name, value):
    """The controller's own feature string for (name, value); the bias feature as is."""
    if name == learning.BIAS_FEATURE:
        return learning.BIAS_FEATURE
    if hasattr(learning, "feature_name"):
        return learning.feature_name(name, value)
    return json.dumps([name, value], separators=(",", ":"))


INNATE_KEYS = tuple(feature_key(name, value) for name, value in INNATE_FEATURES)


def rng_for(seed, child_id):
    """Genome mutation stream for one birth; independent of the world RNG."""
    return random.Random(f"{GENOME_SALT}:{seed}:{child_id}")


def _clip(value, low, high):
    return min(high, max(low, value))


def expressed(genome, config):
    """Every genome value with its default filled in."""
    profile = learning.active_profile()
    genome = genome or {}
    return {"bias": genome.get("bias", {}),
            "alpha": profile["alpha"] if genome.get("alpha") is None else genome["alpha"],
            "epsilon": profile["epsilon"] if genome.get("epsilon") is None else genome["epsilon"],
            "lifespan": float(config.max_age) if genome.get("lifespan") is None else genome["lifespan"],
            "maturity": float(config.maturity_age) if genome.get("maturity") is None else genome["maturity"]}


def maturity_age(agent, config):
    value = (agent.genome or {}).get("maturity")
    return config.maturity_age if value is None else value


def _mutate_scalar(value, low, high, rate, scale, rng):
    if rng.random() < rate:
        value = _clip(value * math.exp(rng.uniform(-scale, scale)), low, high)
    return value


def inherit(genome, config, rng, partner=None, actions=learning.ACTIONS):
    """A child's genome from its parent's genome (and a partner's, for sexual births).

    Never reads a controller: what a parent learned is not passed on. With a
    partner, each innate feature row and each scalar gene comes from one parent
    chosen with equal probability (recombination), then mutation applies.
    """
    mine = expressed(genome, config)
    if partner is not None:
        other = expressed(partner, config)
        rows = {}
        for key in INNATE_KEYS:
            row = (mine if rng.random() < .5 else other)["bias"].get(key)
            if row:
                rows[key] = dict(row)
        mine = {name: (mine if rng.random() < .5 else other)[name] for name in ("alpha", "epsilon", "lifespan", "maturity")}
        mine["bias"] = rows
    bias = {key: dict(row) for key, row in mine["bias"].items()}
    rate, scale = config.bias_mutation_rate, config.bias_mutation_scale
    for key in INNATE_KEYS:
        row = bias.get(key, {})
        for action in actions:
            if rng.random() < rate:
                value = _clip(row.get(action, 0.) + rng.gauss(0., scale), -BIAS_BOUND, BIAS_BOUND)
                if value:
                    row[action] = value
                else:
                    row.pop(action, None)
        if row:
            bias[key] = row
        else:
            bias.pop(key, None)
    child = {"version": GENOME_VERSION, "bias": {key: bias[key] for key in sorted(bias)}}
    child["alpha"] = _mutate_scalar(mine["alpha"], *ALPHA_BOUNDS, config.mutation_rate, config.mutation_scale, rng)
    child["epsilon"] = _mutate_scalar(mine["epsilon"], *EPSILON_BOUNDS, config.mutation_rate, config.mutation_scale, rng)
    if config.age_effects:
        low, high = (config.max_age * bound for bound in LIFESPAN_BOUNDS)
        child["lifespan"] = _mutate_scalar(mine["lifespan"], max(1., low), high, config.mutation_rate, config.mutation_scale, rng)
        low, high = (config.maturity_age * bound for bound in MATURITY_BOUNDS)
        child["maturity"] = _mutate_scalar(mine["maturity"], low, high, config.mutation_rate, config.mutation_scale, rng)
    return child


def recombine_anatomy(left, right, rng):
    """Each anatomy gene from one parent, chosen with equal probability."""
    return {name: (left if rng.random() < .5 else right)[name] for name in sorted(left)}


def seed_controller(genome, *, max_memories=32):
    """A newborn controller whose starting weights are the innate biases.

    Uses the learner's own constructor, learning.controller_from_biases, when it
    exists; otherwise adapter_controller() below (same result, tested). Learned state
    (q, memory, trace, visits, counters) is empty either way.
    """
    bias = (genome or {}).get("bias", {})
    alpha, epsilon = (genome or {}).get("alpha"), (genome or {}).get("epsilon")
    constructor = getattr(learning, "controller_from_biases", None)
    if constructor is not None:
        return constructor(bias or None, alpha=alpha, epsilon=epsilon, max_memories=max_memories)
    return adapter_controller(genome, max_memories=max_memories)


def adapter_controller(genome, *, max_memories=32):
    """Fallback seeding: a blank controller with the biases written as starting weights
    (the bias feature first, then sorted features, capped at the feature capacity)."""
    bias = (genome or {}).get("bias", {})
    alpha, epsilon = (genome or {}).get("alpha"), (genome or {}).get("epsilon")
    controller = learning.new_controller(alpha=alpha, epsilon=epsilon, max_memories=max_memories)
    rows = {key: {action: float(value) for action, value in row.items() if value}
            for key, row in bias.items() if any(row.values())}
    if rows:
        keys = [learning.BIAS_FEATURE] + sorted(key for key in rows if key != learning.BIAS_FEATURE)
        keys = keys[:controller["max_features"]]
        controller["weights"] = {key: dict(sorted(rows.get(key, {}).items())) for key in keys}
        controller["feature_order"] = list(keys)
    return controller


def ability(agent, config):
    """Age curve on ability; exactly 1.0 when Config.age_effects is off."""
    if not config.age_effects:
        return 1.
    genome = expressed(agent.genome, config)
    maturity = genome["maturity"]
    adult = _clip((maturity / config.maturity_age) ** .25, *ADULT_BOUNDS) if config.maturity_age > 0 else 1.
    factor = adult
    if maturity > 0 and agent.age < maturity:
        factor *= JUVENILE_START + (1 - JUVENILE_START) * agent.age / maturity
    onset = SENESCENCE_ONSET * agent.lifespan
    if agent.age > onset:
        span = max(1e-9, agent.lifespan - onset)
        factor *= 1 - (1 - SENESCENT_END) * min(1., (agent.age - onset) / span)
    return factor


def upkeep(agent, config):
    """Metabolic multiplier for a heritable lifespan; exactly 1.0 when age effects are off."""
    if not config.age_effects:
        return 1.
    return (expressed(agent.genome, config)["lifespan"] / config.max_age) ** .5


def innate_values(genome, observation_bins):
    """Innate Q of every action for a dict of local bins (name -> value)."""
    bias = (genome or {}).get("bias", {})
    active = [learning.BIAS_FEATURE] + [feature_key(name, value) for name, value in sorted(observation_bins.items())
                                        if value is not None]
    return {action: math.fsum(bias.get(key, {}).get(action, 0.) for key in active) for action in learning.ACTIONS}


# Contexts for innate P(eat | food in reach): food in reach in each direction, each energy band.
NEAR_CONTEXTS = tuple({"food_near": True, "food_direction": direction, "energy_band": band, "health_band": "well",
                       "neighbor_near": False, "tool_available": False}
                      for direction in _DIRECTIONS[:-1] for band in ("low", "middle", "high"))


def innate_choice_probability(genome, config, action="eat", contexts=NEAR_CONTEXTS, actions=learning.ACTIONS):
    """P(action) of a newborn's first decision, from its genome alone, averaged over contexts.

    Epsilon-greedy over the innate values: (1 - epsilon) shared among the tied best
    actions plus epsilon / n for a random choice (epsilon = the genome's base rate).
    """
    epsilon = expressed(genome, config)["epsilon"]
    total = 0.
    for context in contexts:
        values = innate_values(genome, context)
        best = max(values[a] for a in actions)
        tied = [a for a in actions if values[a] == best]
        total += (1 - epsilon) * (action in tied) / len(tied) + epsilon / len(actions)
    return total / len(contexts)


def validate_genome(genome, config):
    if not isinstance(genome, dict):
        raise ValueError("genome must be an object")
    if set(genome) - {"version", "bias", "alpha", "epsilon", "lifespan", "maturity"}:
        raise ValueError("unknown genome field")
    bias = genome.get("bias", {})
    if not isinstance(bias, dict) or set(bias) - set(INNATE_KEYS):
        raise ValueError("innate bias outside the innate feature set")
    for row in bias.values():
        if not isinstance(row, dict) or set(row) - set(learning.ACTIONS):
            raise ValueError("innate bias names an unknown action")
        for value in row.values():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
                    or abs(value) > BIAS_BOUND:
                raise ValueError("innate bias outside its bound")
    for name, (low, high), scale in (("alpha", ALPHA_BOUNDS, 1.), ("epsilon", EPSILON_BOUNDS, 1.),
                                      ("lifespan", LIFESPAN_BOUNDS, config.max_age),
                                      ("maturity", MATURITY_BOUNDS, config.maturity_age)):
        value = genome.get(name)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) \
                or not low * scale - 1e-9 <= value <= max(high * scale, 1.) + 1e-9:
            raise ValueError(f"genome {name} outside its bounds")
    return True
