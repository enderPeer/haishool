"""Visitors: inhabitants of a life8 world controlled by a model trained only on life8 experience.

A visitor is an ordinary individual of the world (a founder, or for the elder arm a living
individual of a later tick) whose choice of action comes from a :class:`Policy` instead of its
own learning controller. The visitor:

* perceives only its own local observation: the prompt is
  ``experience.prompt_for(experience.features_of(world.observe(agent)))`` with ``want thrive``,
  the same features its own learner would see, nothing else (no ids, no coordinates, no other
  individual's state);
* acts only through its own body: the policy returns one action word, which goes to
  ``World.step`` as that individual's per-agent override, so the engine executes it with its
  normal cost, its default target choice and every normal consequence (it can starve, be
  wounded, be eaten by a predator and die);
* has ordinary offspring: births are the engine's; a child is never a visitor, starts from its
  genome like any newborn (nothing the model or the body's controller learned is inherited)
  and chooses with its own controller.

RNG alignment (no engine change). ``World.step`` draws each organism's decision from the world
RNG in id order, but skips the draw for an overridden organism, so a plain override would shift
every later random number and the twin would diverge even if the visitor acted exactly like the
native. :meth:`Visitors.decide` therefore makes every organism's decision itself, in the order
and with the calls ``World.step`` uses (``learning.choose_action`` on each organism's own
controller with ``world.rng``), and passes all of them as overrides: natives get their own
controller's choice, a visitor's body still draws its native choice (so the RNG stream and the
body's controller bookkeeping stay those of the untouched twin) but that choice is discarded and
the policy's action is executed. With a policy that returns the native choice the visitor world
equals the untouched twin bit for bit (tested), so any difference between twins comes from the
visitors' actions alone.

Valid actions (``allowed_actions``): ``"body"`` (default) is the world's enabled actions minus
what this body cannot do at this tick from its own state: no call without a voice organ, no
drop/dismantle/heat with empty hands, no combine with fewer than two items, no collect with full
hands. ``"engine"`` is exactly the enabled list a native learner chooses from (the mirror check
uses it). An answer outside the valid set becomes ``rest`` and is counted as a fallback.

Policies: :class:`TorchPolicy` (a ``haishool.student`` checkpoint, highest-scoring valid action
token after ``... want thrive. a``, batched, CPU or CUDA) and :class:`FakePolicy` (tests and
null arms: ``rest``, ``random``, ``first``, ``mirror``, a table or a function). Nothing here
imports torch unless a TorchPolicy is made.

Snapshots: :meth:`Visitors.snapshot` is ``world.snapshot()`` with ``visitor: true``,
``visitor_label`` and ``visitor_since`` on visitor agents (view3d highlights them);
:func:`write_archive` writes a run archive (manifest, snapshots, events, summary) that
``python -m haishool.life8.view3d`` can read.
"""
from __future__ import annotations

import json
import math
import random
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from . import experience, learning

SCHEMA = "life8-visitor-v1"
VALID_MODES = ("body", "engine")
FALLBACK = "rest"
WANT = "thrive"  # a visitor asks for the best outcome class (experience.OUTCOMES)


# ---------------------------------------------------------------- policies

@runtime_checkable
class Policy(Protocol):
    """Maps prompts ``q life8 see ... want thrive. a`` to action words.

    ``allowed`` (optional) gives, per prompt, the action words the engine allows that individual
    at that tick; a policy should answer inside it (the caller replaces anything else by rest)."""

    def act(self, prompts: list[str], allowed: list[Sequence[str]] | None = None) -> list[str]:
        ...


def pick(scores: Mapping[str, float], allowed: Sequence[str] | None) -> str:
    """The highest-scoring allowed action word (ties: ACTION_WORDS order); rest when none is scored."""
    best, best_score = None, -math.inf
    choices = experience.ACTION_WORDS if allowed is None else [a for a in experience.ACTION_WORDS if a in set(allowed)]
    for action in choices:
        score = scores.get(action)
        if score is not None and score > best_score:
            best, best_score = action, score
    return best if best is not None else FALLBACK


class FakePolicy:
    """A policy without a model, for tests and null arms.

    mode: ``rest`` (always rest), ``first`` (first allowed word), ``random`` (uniform over the
    allowed words, own seeded RNG), ``mirror`` (the body's own native choice, which
    :class:`Visitors` passes in; the twin-identity check), a dict prompt -> action, or a
    function ``(prompt, allowed) -> action``."""

    def __init__(self, mode: str | Mapping | Callable = "rest", seed: int = 0):
        self.mode, self.rng = mode, random.Random(f"life8-fake-policy:{seed}")
        self.wants_native = mode == "mirror"
        self.calls: list[list[str]] = []

    def act(self, prompts, allowed=None, native=None):
        self.calls.append(list(prompts))
        out = []
        for index, prompt in enumerate(prompts):
            ok = list(allowed[index]) if allowed is not None else list(experience.ACTION_WORDS)
            if self.mode == "rest":
                out.append("rest")
            elif self.mode == "first":
                out.append(ok[0] if ok else FALLBACK)
            elif self.mode == "random":
                out.append(self.rng.choice(ok) if ok else FALLBACK)
            elif self.mode == "mirror":
                if native is None:
                    raise ValueError("the mirror policy needs the native choices")
                out.append(native[index])
            elif isinstance(self.mode, Mapping):
                out.append(self.mode.get(prompt, FALLBACK))
            elif callable(self.mode):
                out.append(self.mode(prompt, ok))
            else:
                raise ValueError(f"unknown fake policy mode {self.mode!r}")
        return out


class TorchPolicy:
    """A ``haishool.student`` checkpoint as a policy.

    Each prompt is encoded like a training sample (``<eos>`` then the words; the last ``ctx``
    tokens are kept), the batch is right-padded (causal attention: padding after the last real
    token changes nothing before it), and the next-token log-probabilities at the last real
    position are read for the action words. The answer is the highest-scoring allowed word that
    is in the vocabulary, rest when none is. ``device=None`` picks CUDA when available."""

    def __init__(self, path, device: str | None = None, batch_size: int = 256):
        from haishool import student
        if student.torch is None:
            raise RuntimeError("torch is not installed: a TorchPolicy needs it (pip install torch)")
        self.torch, self.student = student.torch, student
        self.device = device or ("cuda" if self.torch.cuda.is_available() else "cpu")
        self.path = str(path)
        self.model, self.vocab = student.load(Path(path), self.device)
        self.ctx = self.model.config.ctx
        self.batch_size = batch_size
        self.action_ids = {w: self.vocab.stoi[w] for w in experience.ACTION_WORDS if w in self.vocab.stoi}
        self.missing_actions = [w for w in experience.ACTION_WORDS if w not in self.vocab.stoi]
        self.unknown_words: Counter = Counter()
        self.prompts_seen = 0

    def encode(self, prompt: str) -> list[int]:
        words = [self.student.EOS, *self.student.tokens(prompt)]
        for word in words:
            if word not in self.vocab.stoi:
                self.unknown_words[word] += 1
        return self.vocab.encode(words)[-self.ctx:]

    def scores(self, prompts: list[str]) -> list[dict[str, float]]:
        """Log-probability of each in-vocabulary action word as the next token, per prompt."""
        torch = self.torch
        out: list[dict[str, float]] = []
        names = list(self.action_ids)
        index = torch.tensor([self.action_ids[n] for n in names], device=self.device)
        with torch.no_grad():
            for start in range(0, len(prompts), self.batch_size):
                chunk = [self.encode(p) for p in prompts[start:start + self.batch_size]]
                width = max(len(ids) for ids in chunk)
                x = torch.zeros((len(chunk), width), dtype=torch.long, device=self.device)
                for row, ids in enumerate(chunk):
                    x[row, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=self.device)
                lengths = torch.tensor([len(ids) - 1 for ids in chunk], device=self.device)
                logits, _ = self.model(x)
                last = logits[torch.arange(len(chunk), device=self.device), lengths].float()
                logp = torch.log_softmax(last, dim=-1)[:, index].cpu().tolist()
                out += [dict(zip(names, row)) for row in logp]
        self.prompts_seen += len(prompts)
        return out

    def act(self, prompts, allowed=None):
        if not prompts:
            return []
        scored = self.scores(list(prompts))
        return [pick(s, allowed[i] if allowed is not None else None) for i, s in enumerate(scored)]

    def describe(self) -> dict:
        return {"kind": "torch", "path": self.path, "device": self.device, "ctx": self.ctx,
                "missing_actions": self.missing_actions, "prompts": self.prompts_seen,
                "unknown_words": dict(self.unknown_words.most_common(20))}


_POLICIES: dict = {}


def load_policy(spec: str, *, device: str | None = None, seed: int = 0):
    """``fake:<mode>`` (rest, first, random, mirror) or a student checkpoint path (cached per process)."""
    if spec.startswith("fake:"):
        return FakePolicy(spec[5:], seed=seed)
    key = (spec, device)
    if key not in _POLICIES:
        _POLICIES[key] = TorchPolicy(spec, device=device)
    return _POLICIES[key]


def describe_policy(policy) -> dict:
    if hasattr(policy, "describe"):
        return policy.describe()
    if isinstance(policy, FakePolicy):
        return {"kind": "fake", "mode": policy.mode if isinstance(policy.mode, str) else "custom"}
    return {"kind": type(policy).__name__}


# ---------------------------------------------------------------- the controller

def allowed_actions(world, agent, mode: str = "body") -> list[str]:
    """Action words the engine allows this individual at this tick (see module doc)."""
    if mode not in VALID_MODES:
        raise ValueError(f"valid mode must be one of {VALID_MODES}")
    enabled = world._enabled_actions()
    if mode == "engine":
        return enabled
    held, limit = len(agent.inventory), world.config.inventory_limit
    out = []
    for action in enabled:
        if action.startswith("signal_") and not agent.anatomy.voice:
            continue
        if action in ("drop", "dismantle", "heat") and held < 1:
            continue
        if action == "combine" and held < 2:
            continue
        if action == "collect" and held >= limit:
            continue
        out.append(action)
    return out


def visitor_prompt(world, agent, want: str = WANT) -> str:
    """The visitor's question from its own observation only."""
    return experience.prompt_for(experience.features_of(world.observe(agent)), want)


class Visitors:
    """The visitors of one world and the per-tick decision for World.step (see module doc)."""

    def __init__(self, policy, ids=(), *, label: str = "visitor", valid: str = "body",
                 want: str = WANT, placed_tick: int | None = None):
        if valid not in VALID_MODES:
            raise ValueError(f"valid mode must be one of {VALID_MODES}")
        self.policy, self.label, self.valid, self.want = policy, label, valid, want
        self.ids: set[int] = set()
        self.placed: dict[int, int] = {}
        self.native_choices: dict[int, str] = {}
        self.last_overrides: dict[int, str] = {}
        self.last_prompts: dict[int, str] = {}
        self.decisions = 0
        self.fallbacks = 0
        self.actions: Counter = Counter()
        self.matches_native = 0
        if ids:
            self.ids.update(int(i) for i in ids)
            for identity in self.ids:
                self.placed[identity] = placed_tick if placed_tick is not None else 0

    def place(self, world, ids) -> list[int]:
        """Make living individuals ``ids`` visitors from the current tick on."""
        placed = []
        for identity in ids:
            if identity not in world.agents:
                raise ValueError(f"individual {identity} is not alive at tick {world.tick}")
            self.ids.add(identity)
            self.placed.setdefault(identity, world.tick)
            placed.append(identity)
        return placed

    def active(self, world) -> list[int]:
        return sorted(i for i in self.ids if i in world.agents)

    def decide(self, world) -> dict[int, str]:
        """Every organism's action for the next World.step, drawn as World.step would draw it,
        with the visitors' native choices replaced by the policy's answers."""
        if world.stop_reason:
            return {}
        cfg = world.config
        enabled = world._enabled_actions()
        overrides, observations = {}, {}
        world._index = world._grid((world.foods, world.objects, world.agents))
        try:
            for identity in sorted(world.agents):
                agent = world.agents[identity]
                observation = world.observe(agent)
                native = learning.choose_action(agent.controller, observation, world.rng,
                                                enabled=cfg.learning, actions=enabled)
                overrides[identity] = native
                if identity in self.ids:
                    observations[identity] = observation
        finally:
            world._index = None
        self.native_choices = dict(overrides)
        visitors = sorted(observations)
        self.last_prompts = {}
        if visitors:
            prompts = [experience.prompt_for(experience.features_of(observations[i]), self.want) for i in visitors]
            allowed = [allowed_actions(world, world.agents[i], self.valid) for i in visitors]
            if getattr(self.policy, "wants_native", False):
                answers = self.policy.act(prompts, allowed, native=[overrides[i] for i in visitors])
            else:
                answers = self.policy.act(prompts, allowed)
            if len(answers) != len(prompts):
                raise ValueError("the policy must answer every prompt")
            for identity, prompt, answer, ok in zip(visitors, prompts, answers, allowed):
                if answer not in ok:
                    answer = FALLBACK
                    self.fallbacks += 1
                self.matches_native += answer == overrides[identity]
                overrides[identity] = answer
                self.last_prompts[identity] = prompt
                self.actions[answer] += 1
                self.decisions += 1
        self.last_overrides = {i: overrides[i] for i in visitors}
        return overrides

    def step(self, world) -> list[dict]:
        """One tick of ``world`` with the visitors acting (a stopped world returns [])."""
        if world.stop_reason:
            return []
        return world.step(self.decide(world))

    def snapshot(self, world) -> dict:
        snap = world.snapshot()
        for agent in snap["agents"]:
            if agent["id"] in self.ids:
                agent["visitor"] = True
                agent["visitor_label"] = self.label
                agent["visitor_since"] = self.placed.get(agent["id"])
        snap["visitors"] = {"label": self.label, "ids": sorted(self.ids), "alive": self.active(world)}
        return snap

    def summary(self) -> dict:
        return {"schema": SCHEMA, "label": self.label, "valid": self.valid, "want": self.want,
                "ids": sorted(self.ids), "placed": {str(k): v for k, v in sorted(self.placed.items())},
                "decisions": self.decisions, "fallbacks": self.fallbacks,
                "same_as_native": self.matches_native, "actions": dict(sorted(self.actions.items())),
                "policy": describe_policy(self.policy)}


def run(world, visitors: Visitors | None, ticks: int, *, on_step=None) -> dict:
    """Advance ``world`` to tick ``ticks`` (absolute), with visitors if given; ``on_step(world,
    events)`` sees every step's events. Returns the world's summary."""
    while world.tick < ticks and not world.stop_reason:
        events = visitors.step(world) if visitors is not None else world.step()
        if on_step is not None:
            on_step(world, events)
    return world.summary()


def write_archive(world, visitors: Visitors | None, out, ticks: int, *, sample_every: int = 10) -> Path:
    """Run ``world`` to tick ``ticks`` and write a view3d-readable archive with visitor flags.

    Not a verified ``cli.run_world`` archive (no checkpoint receipt); manifest
    ``"visitor_world": true`` marks it as a visitor arm, never a search result."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    snap = (visitors.snapshot if visitors is not None else lambda w: w.snapshot())
    manifest = {"schema": "life8-visitor-archive-v1", "visitor_world": visitors is not None, "seed": world.seed,
                "config": world.config.to_dict(), "initial_tick": world.tick, "requested_ticks": ticks,
                "sample_every": sample_every, "visitors": visitors.summary() if visitors is not None else None,
                "limits": "Visitor arm: some individuals were controlled by a model; never mixed into search results."}
    with (out / "events.jsonl").open("w", encoding="utf-8") as events, \
            (out / "snapshots.jsonl").open("w", encoding="utf-8") as snapshots:
        def sample():
            state = snap(world)
            state["metrics"] = world.summary()
            snapshots.write(json.dumps(state, allow_nan=False, separators=(",", ":")) + "\n")
        sample()
        count = 0
        while world.tick < ticks and not world.stop_reason:
            records = visitors.step(world) if visitors is not None else world.step()
            for record in records:
                events.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
            count += 1
            if count % sample_every == 0:
                sample()
        if count % sample_every:
            sample()
    manifest.update(final_tick=world.tick, stop_reason=world.stop_reason,
                    visitors=visitors.summary() if visitors is not None else None)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(world.summary(), indent=1), encoding="utf-8")
    return out
