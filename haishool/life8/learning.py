"""Finite, JSON-serializable factored Q learning with no innate meanings.

Linear action values start at zero, ties/exploration use only the supplied RNG, and the
reward is supplied by the engine's actual energy/health change. There are no
bonuses for tools, named technologies, communication, cooperation or symbols.
The controller sees a finite projection of *its own local observations*;
individual IDs and continuous world coordinates cannot become hidden answers.
One-hot local feature values share learned weights across joint contexts. The
exact-state q/visits tables remain bounded diagnostics, not the decision policy.
TD updates are normalized by the number of active features. Weights and action
values have explicit bounds, with deterministic bounded feature-vocabulary LRU.

Demonstrations are individual local observation/action/outcome samples, never
copies of another organism's Q table. The world is responsible for proximity,
visibility and teaching/observation costs. Newborns call new_controller().

life8 (from Matrix 2cf045a): a demonstration may carry the demonstrator's next
observable context, and is then scored with the same bootstrapped target as the
learner's own experience (reward + gamma * max_a Q(next)), so an approach move that
leads to food is no longer taught as harmful. Feature strings are memoized and the
engine's scalar observations skip a JSON round trip; both give identical values.

life8 multi-step credit (controller v4, review gap 4). Three additions, each a
controller parameter stored in the controller itself (so a checkpoint carries it):

* Eligibility traces, naive Q(lambda) with replacing traces. The controller keeps
  the feature lists and actions of its last ``trace_steps`` own decisions. Each TD
  error also moves the weights of those earlier (feature, action) pairs, scaled by
  (gamma * lam) ** k / n_k for the decision k steps back with n_k active features.
  A (feature, action) pair that occurs in several remembered steps is credited once,
  with its largest factor. The trace is emptied at a terminal transition and is never
  touched by demonstrations (another organism's steps are not this one's history).
  ``trace_steps`` is the smallest k with (gamma * lam) ** k < TRACE_CUTOFF, at most 32.
* Conjunction features, computed from the controller's own local bins only:
  food_direction x food distance band (band from the nearest visible food's distance,
  edges FOOD_DISTANCE_BANDS), food_near x energy_band, and, as a documented hook,
  heard_symbol x heard_bearing. The world does not supply a bearing yet; when a heard
  message carries a ``bearing`` value it becomes the ``heard_bearing`` bin and the
  conjunction appears with no other change.
* Exploration that decays with the controller's own decision count (its age in
  decisions): epsilon_t = epsilon * (floor + (1 - floor) * h / (h + decisions)), with
  ``epsilon_floor`` and ``epsilon_halflife`` = h. Setting ``epsilon`` to 0 still
  gives a fully greedy controller.

* Replay from the existing transition buffer (``memory``, review gap 17): after each own
  outcome the controller re-learns ``replay`` remembered transitions, chosen by a fixed
  stride from its update counter (no random numbers), scored with the current weights.
* An egocentric frame (``egocentric``): weights see direction-valued bins and move
  actions relative to the visible food's bearing, so what is learned about stepping
  toward food in one compass direction holds in the other seven. It assumes only that
  the world has no preferred direction. Actions, memories and q/visits stay in the
  world frame. ``conjunction_gain`` (default 1, i.e. off) gives conjunction features a
  larger share of each TD error; it measured no clear benefit and stays off.

* Delayed credit for one decision (``credit_decision``, round-8 fix F1): an outcome that
  the world attributes to an earlier decision (Config.kin_credit: a relative's outcome
  after hearing a call) moves only that decision's (feature, action) weights, as if it had
  been part of that decision's reward, and the remembered transition, if still held.

None of these names an action: the reward stays the engine's energy/health change,
traces, replay and features treat every action alike (relabelling two actions in the
experience relabels the learned weights and nothing else), and newborns start blank
(zero weights, empty trace and memory). The ``one_step`` profile (lam 0, no
conjunctions, no replay, no frame, constant epsilon .15) is controller v3's rule; it
exists for intervention runs. ``use_profile`` selects the profile that new_controller()
fills unspecified parameters from (the engine calls new_controller() for founders and
newborns). controller_from_biases() makes a controller with given innate starting
weights, learning rate and exploration, for a heritable genome.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from contextlib import contextmanager


SCHEMA_VERSION = "life8-factored-q-controller-v4"
POLICY_VERSION = "zero-initialized-onehot-linear-q-lambda-replay-egocentric-conjunctions-decaying-epsilon-v4"
BIAS_FEATURE = "__bias__"
ACTIONS = (
    "rest", "move_n", "move_ne", "move_e", "move_se", "move_s", "move_sw", "move_w", "move_nw",
    "eat", "collect", "drop", "strike", "strike_agent", "strike_object", "combine", "dismantle", "heat", "share",
    "signal_0", "signal_1", "signal_2", "signal_3", "imitate", "teach", "reproduce",
)
STATE_FEATURES = ("food_direction", "food_near", "object_near", "neighbor_near", "energy_band",
                  "tool_available", "health_band", "inventory_count",
                  "target_hardness_band", "target_open_band", "held_mass_band", "held_hardness_band",
                  "held_friction_band", "held_durability_band", "held_sharpness_band",
                  "held_temperature_band", "held_bond_band", "neighbor_familiar", "neighbor_last_exchange")
# Living-world bins (world.py, Config ecology fields). The world puts them in the
# observation only when their mechanism is on, so default worlds see exactly the
# features above. Added by the round-8 ecology stage (additive).
ECOLOGY_FEATURES = ("danger_seen", "danger_direction", "bloom_seen", "heard_age_band", "heard_kin_band",
                    "own_call", "neighbor_balance")
# Bins derived from the controller's own local observation (only with conjunctions on).
EXTRA_FEATURES = ("food_distance_band", "heard_bearing")
# Edges of the nearest visible food's distance (world units). 1.5 is the default
# interaction range; a default organism moves about 0.87 per step and sees 5.0.
FOOD_DISTANCE_BANDS = ((1.5, "reach"), (3.0, "close"), (5.0, "mid"))
FOOD_DISTANCE_BEYOND = "far"
# Pairs of bins whose joint value becomes one extra one-hot feature.
# The last one is a coarse "situation" tile: one weight per (food bearing, distance,
# hardness, energy) cell and action, a small tabular part next to the linear one.
CONJUNCTIONS = (("food_direction", "food_distance_band"), ("food_near", "energy_band"),
                ("food_direction", "food_distance_band", "target_hardness_band", "energy_band"),
                ("heard_symbol", "heard_bearing"))
TRACE_CUTOFF = .05
MAX_TRACE_STEPS = 32
MAX_REPLAY = 16

PROFILES = {
    # Measured in docs/life8/README.md, "Multi-step credit" (lam .5-.9 indistinguishable there).
    "multistep": {"alpha": .25, "gamma": .9, "epsilon": .3, "lam": .8, "conjunctions": True,
                  "epsilon_floor": .1, "epsilon_halflife": 200, "egocentric": True, "replay": 4,
                  "conjunction_gain": 1.},
    # Controller v3 (life8 foundation, and Matrix 2cf045a's rule): one-step TD, constant epsilon.
    "one_step": {"alpha": .25, "gamma": .9, "epsilon": .15, "lam": 0., "conjunctions": False,
                 "epsilon_floor": 1., "epsilon_halflife": 0, "egocentric": False, "replay": 0,
                 "conjunction_gain": 1.},
}
DEFAULT_PROFILE = "multistep"
_ACTIVE = [PROFILES[DEFAULT_PROFILE]]


@contextmanager
def use_profile(name, **overrides):
    """Within the block, new_controller() fills unspecified parameters from ``name``
    (a key of PROFILES), with ``overrides`` replacing single parameters.

    For intervention runs in one process (e.g. analysis.credit). Controllers already
    created keep their own stored parameters, so a checkpoint is unaffected; a resumed
    run must use the same profile for the organisms born after the resume.
    """
    if name not in PROFILES:
        raise ValueError(f"unknown learner profile {name!r}; known: {', '.join(sorted(PROFILES))}")
    unknown = set(overrides) - set(PROFILES[name])
    if unknown:
        raise ValueError(f"unknown learner parameters {sorted(unknown)}")
    _ACTIVE.append({**PROFILES[name], **overrides})
    try:
        yield _ACTIVE[-1]
    finally:
        _ACTIVE.pop()


def active_profile():
    """The parameter set new_controller() currently fills defaults from."""
    return dict(_ACTIVE[-1])


def _finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _int(value, label, low):
    if isinstance(value, bool) or not isinstance(value, int) or value < low:
        raise ValueError(f"{label} must be an integer of at least {low}")
    return value


def trace_length(gamma, lam):
    """Smallest k with (gamma * lam) ** k < TRACE_CUTOFF, capped at MAX_TRACE_STEPS."""
    decay = gamma * lam
    if decay <= 0:
        return 0
    steps, factor = 0, 1.
    while factor >= TRACE_CUTOFF and steps < MAX_TRACE_STEPS:
        factor *= decay
        steps += 1
    return steps


def new_controller(*, alpha=None, gamma=None, epsilon=None, max_states=128, max_memories=32, max_features=256,
                   reward_clip=1000.0, lam=None, conjunctions=None, epsilon_floor=None, epsilon_halflife=None,
                   trace_steps=None, egocentric=None, replay=None, conjunction_gain=None, profile=None) -> dict:
    """A genuinely untrained controller: no copied policy, memories, trace or word map.

    Unspecified learning parameters come from ``profile`` (default: the active one,
    see use_profile). ``trace_steps`` defaults to trace_length(gamma, lam).
    """
    if profile is not None and profile not in PROFILES:
        raise ValueError(f"unknown learner profile {profile!r}")
    base = PROFILES[profile] if profile is not None else _ACTIVE[-1]
    pick = lambda value, name: base[name] if value is None else value
    alpha, gamma, epsilon, lam, floor = (_finite(pick(value, name), name) for value, name in
                                         ((alpha, "alpha"), (gamma, "gamma"), (epsilon, "epsilon"),
                                          (lam, "lam"), (epsilon_floor, "epsilon_floor")))
    conjunctions = pick(conjunctions, "conjunctions")
    egocentric = pick(egocentric, "egocentric")
    replay = _int(pick(replay, "replay"), "replay", 0)
    conjunction_gain = _finite(pick(conjunction_gain, "conjunction_gain"), "conjunction_gain")
    if not 1 <= conjunction_gain <= 100:
        raise ValueError("conjunction_gain must lie within 1 and 100")
    if replay > MAX_REPLAY:
        raise ValueError(f"replay must be at most {MAX_REPLAY}")
    if type(egocentric) is not bool:
        raise ValueError("egocentric must be boolean")
    halflife = _int(pick(epsilon_halflife, "epsilon_halflife"), "epsilon_halflife", 0)
    if not 0 < alpha <= 1 or not 0 <= gamma <= .99 or not 0 <= epsilon <= 1:
        raise ValueError("alpha must be (0,1], gamma [0,.99], epsilon [0,1]")
    if not 0 <= lam <= 1 or not 0 <= floor <= 1:
        raise ValueError("lam and epsilon_floor must lie within zero and one")
    if type(conjunctions) is not bool:
        raise ValueError("conjunctions must be boolean")
    trace_steps = trace_length(gamma, lam) if trace_steps is None else _int(trace_steps, "trace_steps", 0)
    if trace_steps > MAX_TRACE_STEPS or (trace_steps and gamma * lam <= 0):
        raise ValueError("trace_steps must be at most 32, and zero when gamma * lam is zero")
    if isinstance(max_states, bool) or not isinstance(max_states, int) or max_states < 2:
        raise ValueError("max_states must be an integer of at least two")
    if isinstance(max_memories, bool) or not isinstance(max_memories, int) or max_memories < 0:
        raise ValueError("max_memories must be a nonnegative integer")
    if isinstance(max_features, bool) or not isinstance(max_features, int) or max_features < 2:
        raise ValueError("max_features must be an integer of at least two")
    reward_clip = _finite(reward_clip, "reward clip")
    if not 0 < reward_clip <= 1e6:
        raise ValueError("reward_clip must be positive and at most 1e6")
    return {"schema_version": SCHEMA_VERSION, "policy_version": POLICY_VERSION,
            "alpha": alpha, "gamma": gamma, "epsilon": epsilon,
            "lam": lam, "trace_steps": trace_steps, "conjunctions": conjunctions, "egocentric": egocentric, "replay": replay, "conjunction_gain": conjunction_gain,
            "epsilon_floor": floor, "epsilon_halflife": halflife,
            "max_states": max_states, "max_memories": max_memories, "reward_clip": reward_clip,
            "max_features": max_features, "weights": {}, "feature_order": [],
            "q": {}, "visits": {}, "state_order": [], "memory": [], "trace": [],
            "decisions": 0, "updates": 0, "demonstrations": 0}


PARAMETERS = ("alpha", "gamma", "epsilon", "max_states", "max_memories", "max_features", "reward_clip",
              "lam", "conjunctions", "epsilon_floor", "epsilon_halflife", "trace_steps", "egocentric", "replay", "conjunction_gain")


def feature_name(name, value):
    """The weight-table key of the one-hot feature ``name == value`` (e.g. ("food_near", True))."""
    return _feature_string(name, value)


def controller_from_biases(bias_weights=None, *, alpha=None, epsilon=None, **parameters) -> dict:
    """A newborn controller whose weights start at given innate values (for a heritable genome).

    ``bias_weights`` is either {action: weight}, placed on the always-active bias
    feature, or {feature: {action: weight}}, where a feature is BIAS_FEATURE or a key
    made by feature_name(name, value) (e.g. feature_name("food_near", True)). Weights
    must be finite and within the controller's value bound; unknown actions are
    refused. ``alpha`` (learning rate), ``epsilon`` (initial exploration) and any other
    new_controller() parameter may be given; the rest come from the active profile.
    The result has no memories, trace, visits or decisions: only its starting weights
    differ from new_controller(), and they are ordinary weights that learning then
    changes. Zero weights are dropped; empty or None biases give exactly
    new_controller(...). Features beyond the capacity are cut (bias feature first,
    then sorted). The engine does not call this; the genome code that seeds newborns does.

    With an egocentric controller (the default profile) the weights live in the
    food-relative frame: whenever food is visible, food_direction reads "n" and
    move_n means "step toward the food" (move_ne: 45 degrees clockwise of it, and so
    on). A bias on ("food_direction", "e") is therefore never active there; use
    ("food_direction", "n") with move_n for an innate approach.
    """
    controller = new_controller(alpha=alpha, epsilon=epsilon, **parameters)
    if not bias_weights:
        return controller
    if not isinstance(bias_weights, Mapping):
        raise ValueError("bias_weights must be a mapping")
    if all(not isinstance(value, Mapping) for value in bias_weights.values()):
        bias_weights = {BIAS_FEATURE: bias_weights}
    bound = _value_bound(controller)
    rows = {}
    for feature, row in bias_weights.items():
        if not isinstance(feature, str) or not isinstance(row, Mapping):
            raise ValueError("bias_weights must map features to {action: weight}")
        clean = {}
        for action, value in row.items():
            if action not in ACTIONS:
                raise ValueError(f"unknown action {action!r} in bias_weights")
            value = _finite(value, "bias weight")
            if abs(value) > bound:
                raise ValueError("bias weight exceeds the controller's value bound")
            if value:
                clean[action] = value
        if clean:
            rows[feature] = clean
    if not rows:
        return controller
    # The bias feature first, then sorted features, capped at the feature capacity.
    features = ([BIAS_FEATURE] + sorted(feature for feature in rows if feature != BIAS_FEATURE))[:controller["max_features"]]
    controller["weights"] = {feature: dict(sorted(rows.get(feature, {}).items())) for feature in features}
    controller["feature_order"] = features
    return controller


def current_epsilon(controller):
    """Exploration rate for the next decision; decays with the controller's decision count."""
    epsilon, halflife = controller["epsilon"], controller["epsilon_halflife"]
    floor = controller["epsilon_floor"]
    if not halflife or floor == 1:
        return epsilon
    return epsilon * (floor + (1 - floor) * halflife / (halflife + controller["decisions"]))


def _json_copy(value):
    """Accept only finite JSON data; all memories are independent of caller mutation."""
    try:
        return json.loads(json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":")))
    except (TypeError, ValueError) as exc:
        raise ValueError("observations must be finite JSON data") from exc


def _food_distance_band(observation):
    food = observation.get("food")
    if not isinstance(food, list) or not food or not isinstance(food[0], Mapping):
        return None
    distance = food[0].get("distance")
    if isinstance(distance, bool) or not isinstance(distance, (int, float)) or not math.isfinite(distance):
        return None
    for edge, name in FOOD_DISTANCE_BANDS:
        if distance <= edge:
            return name
    return FOOD_DISTANCE_BEYOND


def observation_features(observation, *, extras=True) -> dict:
    """Local bins plus an unlabelled heard token, rather than unique organism IDs.

    The engine supplies these already-local features. For small standalone
    scientific trials, a JSON observation without the engine envelope is also
    accepted directly. This does not confer meanings on any symbol. With
    ``extras`` (controllers with conjunctions), the nearest visible food's distance
    band and, when the chosen heard message carries one, its bearing are added.
    """
    if not isinstance(observation, Mapping):
        return {"observation": _json_copy(observation)}
    engine_observation = any(name in observation for name in (*STATE_FEATURES, "self", "food", "objects", "neighbors", "messages"))
    if not engine_observation:
        return _json_copy(dict(observation))
    features = {name: observation[name] for name in STATE_FEATURES if name in observation}
    for name in ECOLOGY_FEATURES:
        if name in observation:
            features[name] = observation[name]
    own = observation.get("self")
    if "inventory_count" not in features and isinstance(own, Mapping) and "inventory_count" in own:
        features["inventory_count"] = own["inventory_count"]
    messages = observation.get("messages") or []
    heard = [message for message in messages if isinstance(message, Mapping) and "symbol" in message]
    chosen = None
    if heard:
        def recency(item):
            age = item.get("age", 0)
            return _finite(age, "message age")
        # Input order breaks equal-age ties; sender identity is deliberately absent.
        chosen = min(heard, key=recency)
        features["heard_symbol"] = chosen["symbol"]
    else:
        features["heard_symbol"] = None
    if extras:
        band = _food_distance_band(observation)
        if band is not None:
            features["food_distance_band"] = band
        # Hook (review gap 6): a bearing on the heard message becomes a bin, and with it
        # the heard_symbol x heard_bearing conjunction. The world does not supply one yet.
        if chosen is not None and chosen.get("bearing") is not None:
            features["heard_bearing"] = chosen["bearing"]
    if all(type(value) in _SCALARS for value in features.values()):
        # Exactly what the JSON round trip returns for these immutable scalars.
        return {name: features[name] for name in sorted(features)}
    return _json_copy(features)


_SCALARS = (str, bool, int, type(None))
_FEATURE_STRINGS = {}


def _feature_string(name, value):
    if type(value) in (str, bool, int):
        key = (name, type(value), value)
        text = _FEATURE_STRINGS.get(key)
        if text is None:
            if len(_FEATURE_STRINGS) > 100000:
                _FEATURE_STRINGS.clear()
            text = _FEATURE_STRINGS[key] = json.dumps([name, value], sort_keys=True, allow_nan=False, separators=(",", ":"))
        return text
    return json.dumps([name, value], sort_keys=True, allow_nan=False, separators=(",", ":"))


def _project(controller, observation):
    return _frame(controller, observation)[0]


# Compass directions in clockwise order (the world's move_<direction> actions).
COMPASS = ("n", "ne", "e", "se", "s", "sw", "w", "nw")
_COMPASS_INDEX = {name: index for index, name in enumerate(COMPASS)}
DIRECTIONAL_FEATURES = ("food_direction", "heard_bearing", "danger_direction")


def _frame(controller, observation):
    """Projected bins (world frame) and the rotation k (45-degree steps) of the egocentric frame.

    With ``egocentric``, the weights see every direction-valued bin and every move
    action relative to the visible food's bearing: the bins are rotated by -k so the
    food bearing reads "n" (see _framed_features), and move_<d> is weighted as
    move_<d - k>. This only assumes the world has no preferred compass direction; it
    says nothing about which move is good. Without visible food (or without
    egocentric) k is 0 and nothing changes. Memories, the diagnostic q/visits tables
    and every returned action stay in the world frame.
    """
    projected = observation_features(observation, extras=controller.get("conjunctions", False))
    if not controller.get("egocentric"):
        return projected, 0
    bearing = projected.get("food_direction")
    return projected, (_COMPASS_INDEX.get(bearing, 0) if isinstance(bearing, str) else 0)


def _framed_features(controller, projected, k):
    if k:
        projected = dict(projected)
        for name in DIRECTIONAL_FEATURES:
            value = projected.get(name)
            if isinstance(value, str) and value in _COMPASS_INDEX:
                projected[name] = COMPASS[(_COMPASS_INDEX[value] - k) % 8]
    return _project_features(controller, projected)


def _to_frame(action, k):
    if k and action.startswith("move_"):
        return "move_" + COMPASS[(_COMPASS_INDEX[action[5:]] - k) % 8]
    return action


def _frame_actions(available, k):
    """The available actions in the frame (a partial move set is rotated, never completed)."""
    return tuple(_to_frame(action, k) for action in available) if k else available


def state_key(observation) -> str:
    return json.dumps(observation_features(observation), sort_keys=True, allow_nan=False, separators=(",", ":"))


def _check_schema(controller):
    if not isinstance(controller, dict) or controller.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("incompatible controller schema")


def validate_controller(controller) -> dict:
    """Validate a loaded controller without changing its continuation state."""
    _check_schema(controller)
    if controller.get("policy_version") != POLICY_VERSION:
        raise ValueError("incompatible factored policy version")
    if any(name not in controller for name in PARAMETERS):
        raise ValueError("controller is missing a learning parameter")
    new_controller(**{name: controller[name] for name in PARAMETERS})
    if controller["epsilon_halflife"] < 0:
        raise ValueError("epsilon_halflife must be nonnegative")
    q, visits, order = controller.get("q"), controller.get("visits"), controller.get("state_order")
    if not isinstance(q, dict) or not isinstance(visits, dict) or not isinstance(order, list):
        raise ValueError("invalid controller state tables")
    if len(q) > controller["max_states"] or len(order) != len(set(order)) or set(order) != set(q) or set(visits) != set(q):
        raise ValueError("state capacity/order/visit tables disagree")
    for key, values in q.items():
        if not isinstance(key, str) or not isinstance(values, dict) or not set(values) <= set(ACTIONS):
            raise ValueError("invalid Q state/action")
        for value in values.values():
            _finite(value, "Q value")
            if abs(value) > controller["reward_clip"] / (1.0 - controller["gamma"]) + 1e-7:
                raise ValueError("Q value exceeds the bounded outcome return")
        if not isinstance(visits[key], dict) or not set(visits[key]) <= set(ACTIONS):
            raise ValueError("invalid action visits")
        if any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in visits[key].values()):
            raise ValueError("visits must be nonnegative integers")
    memory = controller.get("memory")
    if not isinstance(memory, list) or len(memory) > controller["max_memories"]:
        raise ValueError("memory capacity exceeded")
    for name in ("decisions", "updates", "demonstrations"):
        count = controller.get(name)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("controller counters must be nonnegative integers")
    weights, feature_order = controller.get("weights"), controller.get("feature_order")
    if not isinstance(weights, dict) or not isinstance(feature_order, list):
        raise ValueError("invalid factored weight tables")
    if len(weights) > controller["max_features"] or len(feature_order) != len(set(feature_order)) or set(weights) != set(feature_order):
        raise ValueError("feature capacity/order tables disagree")
    if weights and BIAS_FEATURE not in weights:
        raise ValueError("factored policy is missing its bias feature")
    bound = _value_bound(controller)
    for feature, values in weights.items():
        if not isinstance(feature, str) or not isinstance(values, dict) or not set(values) <= set(ACTIONS):
            raise ValueError("invalid feature/action weights")
        for value in values.values():
            if abs(_finite(value, "feature weight")) > bound + 1e-7:
                raise ValueError("feature weight exceeds its bounded return")
    trace = controller.get("trace")
    if not isinstance(trace, list) or len(trace) > controller["trace_steps"]:
        raise ValueError("eligibility trace exceeds its length")
    for step in trace:
        if not isinstance(step, dict) or set(step) != {"features", "action"} or step["action"] not in ACTIONS:
            raise ValueError("invalid eligibility trace step")
        features = step["features"]
        if not isinstance(features, list) or not features or len(features) > controller["max_features"] \
                or not all(isinstance(feature, str) for feature in features) or len(set(features)) != len(features):
            raise ValueError("invalid eligibility trace features")
    _json_copy(controller)
    return {"states": len(q), "memories": len(memory), "updates": controller["updates"],
            "demonstrations": controller["demonstrations"], "features": len(weights), "trace": len(trace),
            "policy_version": POLICY_VERSION}


def _actions(actions):
    values = tuple(ACTIONS if actions is None else actions)
    if not values or len(values) != len(set(values)) or any(action not in ACTIONS for action in values):
        raise ValueError("actions must be a nonempty distinct subset of the canonical actions")
    return values


def _touch(controller, key):
    table, order = controller["q"], controller["state_order"]
    if key in table:
        order.remove(key)
    elif len(order) >= controller["max_states"]:
        removed = order.pop(0)
        del table[removed]
        del controller["visits"][removed]
    order.append(key)
    table.setdefault(key, {})
    controller["visits"].setdefault(key, {})
    return table[key]


def _value_bound(controller):
    return controller["reward_clip"] / (1.0 - controller["gamma"])


def _project_features(controller, projected):
    features = [BIAS_FEATURE]
    for name, value in sorted(projected.items()):
        if value is not None:
            features.append(_feature_string(name, value))
    if controller.get("conjunctions"):
        for names in CONJUNCTIONS:
            parts = [projected.get(name) for name in names]
            if all(part is not None and type(part) in (str, bool, int) for part in parts):
                features.append(_feature_string("*".join(names), "|".join(json.dumps(part) for part in parts)))
    return features[:controller["max_features"]]


def _features(controller, observation):
    """Generic categorical features; no action-specific or food-specific encoding.

    Missing values have no active indicator. False/zero remain genuine observed
    categories. Sorting and capacity truncation are explicit and reproducible.
    """
    return _framed_features(controller, *_frame(controller, observation))


def _touch_features(controller, features):
    weights, order = controller["weights"], controller["feature_order"]
    for feature in features:
        if feature in weights:
            order.remove(feature)
        elif len(order) >= controller["max_features"]:
            # The neutral bias is always retained; all observed categories use LRU.
            old = next(key for key in order if key != BIAS_FEATURE)
            order.remove(old)
            del weights[old]
        order.append(feature)
        weights.setdefault(feature, {})


def _values(controller, features, available):
    """Sum each action's weights over the active features.

    math.fsum is correctly rounded, so the result does not depend on the order in
    which terms are collected; absent features contribute an exact 0.0.
    """
    bound = _value_bound(controller)
    weights = controller["weights"]
    rows = [row for row in map(weights.get, features) if row]
    count = len(features)
    fsum = math.fsum
    result = {}
    for action in available:
        terms = [row[action] for row in rows if action in row]
        if len(terms) < count:
            terms.append(0.0)
        result[action] = max(-bound, min(bound, fsum(terms)))
    return result


def action_values(controller, observation, actions=None) -> dict:
    """Read-only factored predictions; unknown features/actions contribute zero.

    Keys are world actions (an egocentric controller's frame is undone)."""
    _check_schema(controller)
    projected, k = _frame(controller, observation)
    available = _actions(actions)
    values = _values(controller, _framed_features(controller, projected, k), _frame_actions(available, k))
    return {action: values[_to_frame(action, k)] for action in available}


def _update_weights(controller, features, action, target, trace=()):
    """Shared own-outcome/demonstration update; no copying of another policy.

    ``trace`` holds earlier own decisions (most recent first). Each shares this TD
    error with factor (gamma * lam) ** k / n_k; a (feature, action) pair is credited
    once, with its largest factor. Features evicted since are skipped.
    """
    _touch_features(controller, features)
    prediction = _values(controller, features, (action,))[action]
    bound = _value_bound(controller)
    delta = max(-2 * bound, min(2 * bound, target - prediction))
    weights = controller["weights"]
    gain = controller.get("conjunction_gain", 1.0)
    if not trace and gain == 1.0:
        increment = controller["alpha"] * delta / len(features)
        for feature in features:
            row = weights[feature]
            row[action] = max(-bound, min(bound, row.get(action, 0.0) + increment))
        return _values(controller, features, (action,))[action]
    step = controller["alpha"] * delta
    # Each step's share of the error: c_f / sum(c) with c = gain for conjunctions, 1 otherwise
    # (gain 1 is the plain 1/n normalisation).
    shares = _shares(features, gain)
    credit = {(feature, action): share for feature, share in shares.items()}
    decay, factor = controller["gamma"] * controller["lam"], 1.0
    for past in trace:
        factor *= decay
        past_action = past["action"]
        for feature, share in _shares(past["features"], gain).items():
            share *= factor
            key = (feature, past_action)
            held = credit.get(key)
            if held is None or share > held:
                credit[key] = share
    for (feature, past_action), share in credit.items():
        row = weights.get(feature)
        if row is not None:
            row[past_action] = max(-bound, min(bound, row.get(past_action, 0.0) + step * share))
    return _values(controller, features, (action,))[action]


_CONJUNCTION_FEATURE = {}


def _is_conjunction(feature):
    flag = _CONJUNCTION_FEATURE.get(feature)
    if flag is None:
        if len(_CONJUNCTION_FEATURE) > 100000:
            _CONJUNCTION_FEATURE.clear()
        flag = _CONJUNCTION_FEATURE[feature] = feature != BIAS_FEATURE and "*" in json.loads(feature)[0]
    return flag


def _shares(features, gain):
    if gain == 1.0:
        n = len(features)
        return {feature: 1.0 / n for feature in features}
    weights = [gain if _is_conjunction(feature) else 1.0 for feature in features]
    total = math.fsum(weights)
    return {feature: weight / total for feature, weight in zip(features, weights)}


def _key(projected):
    return json.dumps(projected, sort_keys=True, allow_nan=False, separators=(",", ":"))


def choose_action(controller, observation, rng, enabled=True, actions=None) -> str:
    """Epsilon-greedy action; disabled learning is an uninformed random baseline."""
    _check_schema(controller)
    available = _actions(actions)
    if not enabled:
        return rng.choice(available)
    projected, k = _frame(controller, observation)
    _touch(controller, _key(projected))
    features = _framed_features(controller, projected, k)
    _touch_features(controller, features)
    framed = _frame_actions(available, k)
    values = _values(controller, features, framed)
    epsilon = current_epsilon(controller)
    controller["decisions"] += 1
    if rng.random() < epsilon:
        return rng.choice(available)
    best = max(values.get(action, 0.0) for action in framed)
    # Ties are listed in the world's action order, so with k = 0 this is the previous rule exactly.
    return rng.choice([action for action, inner in zip(available, framed) if values.get(inner, 0.0) == best])


def _bounded_reward(controller, reward):
    reward = _finite(reward, "outcome reward")
    return max(-controller["reward_clip"], min(controller["reward_clip"], reward))


def _remember(controller, projected, action, reward, next_projected, source, terminal=False, frame=0, next_frame=0):
    """A transition in the world frame; ``frame``/``next_frame`` are the egocentric rotations."""
    if not controller["max_memories"]:
        return
    controller["memory"].append({"observation": dict(projected), "action": action,
                                  "reward": reward, "next_observation": dict(next_projected),
                                  "source": source, "terminal": bool(terminal),
                                  "frame": frame, "next_frame": next_frame})
    del controller["memory"][:-controller["max_memories"]]


def _replay(controller, available):
    """Re-learn ``replay`` remembered transitions (own outcomes and demonstrations).

    The memory holds projected bins and their egocentric rotations, so a replayed
    transition is scored like a fresh one, with the current weights and no trace. The
    entries are picked by a fixed stride from the update counter, so replay uses no
    random numbers. The newest entry (just learned) is skipped. The future value uses
    the actions available now (the world's set does not change during a run).
    """
    memory, count = controller["memory"], controller["replay"]
    size = len(memory) - 1
    if not count or size < 1:
        return
    gamma = controller["gamma"]
    start = controller["updates"] * 7
    for index in sorted({(start + 5 * j) % size for j in range(count)}):
        entry = memory[index]
        frame, next_frame = entry.get("frame", 0), entry.get("next_frame", 0)
        future = 0.0
        if not entry.get("terminal"):
            future = max(_values(controller, _framed_features(controller, entry["next_observation"], next_frame),
                                 _frame_actions(available, next_frame)).values())
        _update_weights(controller, _framed_features(controller, entry["observation"], frame),
                        _to_frame(entry["action"], frame), entry["reward"] + gamma * future)


def learn(controller, observation, action, reward, next_observation, enabled=True, *, terminal=False, actions=None):
    """Learn from one actual outcome; the engine supplies energy/health reward.

    The TD error also updates the controller's recent own decisions through its
    eligibility trace (none when trace_steps is 0); a terminal outcome empties it.
    """
    _check_schema(controller)
    if not enabled:
        return
    available = _actions(actions)
    if action not in ACTIONS:
        raise ValueError("unknown action")
    reward = _bounded_reward(controller, reward)
    (projected, k), (next_projected, next_k) = _frame(controller, observation), _frame(controller, next_observation)
    inner = _to_frame(action, k)
    key = _key(projected)
    values = _touch(controller, key)
    _touch(controller, _key(next_projected))
    future = 0.0 if terminal else max(_values(controller, _framed_features(controller, next_projected, next_k),
                                              _frame_actions(available, next_k)).values())
    features = _framed_features(controller, projected, k)
    trace = controller["trace"]
    values[action] = _update_weights(controller, features, inner, reward + controller["gamma"] * future, trace)
    if controller["trace_steps"]:
        # Trace steps hold weight keys, so their features and action are in the frame.
        trace.insert(0, {"features": list(features), "action": inner})
        del trace[controller["trace_steps"]:]
    if terminal:
        trace.clear()
    visits = controller["visits"][key]
    visits[action] = visits.get(action, 0) + 1
    controller["updates"] += 1
    _remember(controller, projected, action, reward, next_projected, "own_outcome", terminal, k, next_k)
    if not terminal:
        _replay(controller, available)


def decision_features(controller, observation):
    """The weight keys (features, in the controller's frame) that learn() updates for a
    decision taken on ``observation``. With credit_decision they let an outcome that
    arrives later reach exactly that decision (world.py: kin credit for a call)."""
    _check_schema(controller)
    return list(_features(controller, observation))


def remembered(controller):
    """Transitions remembered so far (own outcomes plus demonstrations; each adds one memory entry)."""
    return controller["updates"] + controller["demonstrations"]


def credit_decision(controller, features, action, amount, *, remembered_at=None, enabled=True) -> bool:
    """A delayed reward for one earlier own decision (``features`` from decision_features,
    ``action`` its weight key; actions other than move_* are never rotated by the
    egocentric frame).

    Each of the decision's (feature, action) weights moves by alpha * amount * share, the
    share of _update_weights (1/n for n features): exactly what ``amount`` added to that
    decision's reward would have added to its TD step. No other action, decision or trace
    step changes, and features evicted since are skipped. ``remembered_at`` is
    remembered(controller) right after that decision was learned; if its transition is
    still in memory, the remembered reward includes ``amount``, so replay keeps the credit.
    """
    _check_schema(controller)
    if not enabled:
        return False
    amount = _bounded_reward(controller, amount)
    if not amount:
        return False
    bound, step, weights = _value_bound(controller), controller["alpha"] * amount, controller["weights"]
    for feature, share in _shares(features, controller.get("conjunction_gain", 1.0)).items():
        row = weights.get(feature)
        if row is not None:
            row[action] = max(-bound, min(bound, row.get(action, 0.0) + step * share))
    if remembered_at is not None:
        memory = controller["memory"]
        position = len(memory) - 1 - (remembered(controller) - remembered_at)
        if 0 <= position < len(memory) and memory[position]["source"] == "own_outcome" \
                and memory[position]["action"] == action:
            memory[position]["reward"] = _bounded_reward(controller, memory[position]["reward"] + amount)
    return True


def observe_demo(controller, observation, action, outcome, enabled=True, *, next_observation=None,
                 terminal=False, actions=None) -> bool:
    """One local demonstrated outcome, with no peer weights or semantic labels.

    ``outcome`` is the demonstrator's measured reward, or a record carrying its
    energy_delta/health_delta. A mapping with an already-computed ``reward`` is
    accepted for engine integration. Proximity and costs belong to the world.
    With ``next_observation`` (the demonstrator's observable next context) the
    target is bootstrapped exactly like ``learn``; without it (or when
    ``terminal``) the demonstration is scored as a one-step outcome. A demonstration
    neither uses nor changes the learner's own eligibility trace.
    """
    _check_schema(controller)
    if not enabled:
        return False
    if action not in ACTIONS:
        raise ValueError("unknown demonstrated action")
    if isinstance(outcome, Mapping):
        if "energy_delta" in outcome or "health_delta" in outcome:
            reward = _finite(outcome.get("energy_delta", 0), "energy delta") + 8 * _finite(outcome.get("health_delta", 0), "health delta")
        elif "reward" in outcome:
            reward = outcome["reward"]
        else:
            raise ValueError("demonstration requires a measured outcome")
    else:
        reward = outcome
    reward = _bounded_reward(controller, reward)
    projected, k = _frame(controller, observation)
    key = _key(projected)
    values = _touch(controller, key)
    future = 0.0
    next_projected, next_k = projected, k
    if next_observation is not None:
        next_projected, next_k = _frame(controller, next_observation)
        if not terminal:
            _touch(controller, _key(next_projected))
            future = max(_values(controller, _framed_features(controller, next_projected, next_k),
                                 _frame_actions(_actions(actions), next_k)).values())
    values[action] = _update_weights(controller, _framed_features(controller, projected, k), _to_frame(action, k),
                                     reward + controller["gamma"] * future)
    visits = controller["visits"][key]
    visits[action] = visits.get(action, 0) + 1
    controller["demonstrations"] += 1
    _remember(controller, projected, action, reward, next_projected, "local_demonstration",
              terminal or next_observation is None, k, next_k)
    return True
