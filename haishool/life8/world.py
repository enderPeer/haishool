"""Persistent, local, two-dimensional ecology with costly embodied actions.

This is a coarse research prototype, not atomistic physics or a claim of language
or civilization. Toroidal geometry, energy conversion and interaction rules are
explicit. Artifacts and learned memories persist; no technology milestones exist.

Vendored into Haishool round 8 (life8) from Matrix 2cf045a. Changes are listed in
docs/life8/README.md; each bug fix has a regression test in tests/life8.
"""
from __future__ import annotations

import copy
import json
import math
import random
from collections import Counter
from dataclasses import asdict

from .config import Config
from .entities import Agent, Anatomy, Artifact, FoodPatch
from . import genome, learning, materials
from .spatial import GridIndex

SCHEMA_VERSION = "life8-state-v2"
MODEL_VERSION = "life8-persistent-ecology-v2"
GRID_CELL = 2.0
MAX_WORK_EFFORT = 5.0  # bound on one combine/dismantle/heat action's requested work
CHANNEL_SALT = 0x9E3779B97F4A7C15  # Config.channel="scrambled" draws from Random(seed ^ CHANNEL_SALT)
# Living-world ecology (blooms, predators, call power, kin credit) draws only from
# Random(seed ^ ECOLOGY_SALT), so the world RNG stream is the same with it on or off.
ECOLOGY_SALT = 0x2545F4914F6CDD1D
ECOLOGY_SENDER_BANDS = ("danger_seen", "danger_direction", "bloom_seen")
ECOLOGY_RECEIVER_BANDS = ("danger_seen", "bloom_seen", "heard_kin_band")
KIN_DEPTH = 8  # generations searched for a shared ancestor
# Kin credit pays the hearer's reward minus its running mean (an advantage), so the
# steady metabolic drift of every hearer is not credited; the mean is an exponential
# average of the hearer's own reward with this rate (about a 16-tick memory).
KIN_BASELINE_RATE = 1 / 16
# Teacher features a nearby learner could see: its situation and what it holds,
# never its internal energy/health bands, private social memory or heard symbol.
DEMO_FEATURES = tuple(name for name in learning.STATE_FEATURES
                      if name not in ("energy_band", "health_band", "neighbor_familiar", "neighbor_last_exchange"))
SOCIAL_KINDS = ("energy_shared", "strikes", "demonstrations", "imitations")
SENDER_BANDS = ("energy_band", "health_band", "food_direction", "food_near", "object_near", "neighbor_near",
                "target_hardness_band", "tool_available")
RECEIVER_BANDS = ("energy_band", "health_band", "food_direction", "food_near", "neighbor_near")
COMPACT_ACTIONS = ("collect", "drop", "strike", "strike_agent", "strike_object", "combine", "dismantle", "heat",
                   "share", "imitate", "teach")
DIRECTIONS = {"n": (0, -1), "ne": (1, -1), "e": (1, 0), "se": (1, 1),
              "s": (0, 1), "sw": (-1, 1), "w": (-1, 0), "nw": (-1, -1)}
ACTIONS = ("rest", *("move_" + key for key in DIRECTIONS), "eat", "collect", "drop", "strike", "strike_agent", "strike_object", "combine",
           "dismantle", "heat", "share", "signal_0", "signal_1", "signal_2", "signal_3", "imitate", "teach", "reproduce")


def _compass(dx, dy, here=0.):
    """Nearest of the 8 compass bins to (dx, dy); "here" closer than ``here``."""
    norm = math.hypot(dx, dy)
    if norm < max(here, 1e-12):
        return "here"
    return min(DIRECTIONS, key=lambda name: math.dist([dx / norm, dy / norm],
               [v / math.hypot(*DIRECTIONS[name]) for v in DIRECTIONS[name]]))


def _tuples(value):
    return tuple(_tuples(v) for v in value) if isinstance(value, list) else value


class World:
    # Not part of the state: the grid only narrows candidates and never changes results.
    spatial_index = True
    _index = None
    _birth_transfer = {}
    eco = None  # the living-world ecology record (Config.ecology_on()); None in Matrix-default worlds

    @classmethod
    def create(cls, seed=85, config=None):
        if type(seed) is not int or not 0 <= seed < 2 ** 64:
            raise ValueError("seed must be an unsigned 64-bit integer")
        world = cls()
        world.seed, world.config = seed, config or Config()
        if not isinstance(world.config, Config):
            raise ValueError("config must be Config")
        world.rng = random.Random(seed)
        world.channel_rng = random.Random(seed ^ CHANNEL_SALT)  # used only by the scrambled channel
        world.tick = 0
        world.stop_reason = None
        world.agents, world.objects, world.foods = {}, {}, {}
        world.messages = []
        world.pending_heard = []  # compact log only: heard messages awaiting the receiver's next action
        world.next_agent_id = world.next_object_id = world.next_message_id = 1
        world.counters, world.social_edges, world.action_counts, world.artifact_stats = {}, {}, {}, {}
        world.ledger = {key: 0. for key in ("initial_agent_energy", "initial_food", "food_regrown", "food_eaten",
            "energy_from_food", "action_energy", "metabolic_energy", "birth_cost", "energy_wasted", "dead_energy",
            "initial_material_mass", "initial_material_heat", "heat_to_objects", "heat_to_environment", "heat_to_body")}
        cfg = world.config
        for index in range(cfg.food_patches):
            world.foods[index + 1] = FoodPatch(index + 1, world._position(), cfg.food_capacity, cfg.food_capacity,
                                             0. if index % 2 == 0 else .6 + world.rng.random() * .8, cfg.food_regrowth)
        for index in range(cfg.initial_objects):
            material = ("wood", "stone", "fiber")[index % 3]
            world._new_object({"material": material, "mass": 1., "sharpness": .2 if material == "stone" else .03,
                               "properties": materials.material_properties(material, 1.)}, world._position())
        for _ in range(cfg.population):
            anatomy = Anatomy(**{key: world.rng.uniform(.75, 1.25) for key in
                                ("body_mass", "speed", "sensing", "manipulation", "metabolism")})
            world._new_agent(world._position(), cfg.initial_energy, anatomy)
        world.ledger["initial_agent_energy"] = sum(a.energy for a in world.agents.values())
        world.ledger["initial_food"] = sum(f.amount for f in world.foods.values())
        world.ledger["initial_material_mass"] = sum(o.mass for o in world.objects.values())
        world.ledger["initial_material_heat"] = sum(materials.thermal_energy(o) for o in world.objects.values())
        world.eco_rng = random.Random(seed ^ ECOLOGY_SALT)
        world.eco = world._ecology_init() if cfg.ecology_on() else None
        if not world.agents:
            world.stop_reason = "extinction"
        world.validate()
        return world

    # ------------------------------------------------------------ living-world ecology
    def _ecology_init(self):
        """The ecology record: pedigree (for kinship), call-power genes, permanent
        injuries, predators, blooms, and for kin credit the pending hearings, the calls
        they credit and the hearers' reward baselines. Drawn from eco_rng only."""
        cfg, rng = self.config, self.eco_rng
        eco = {"pedigree": {str(i): [] for i in sorted(self.agents)},
               "call_power": {str(i): rng.uniform(.75, 1.25) for i in sorted(self.agents)},
               "scar": {}, "kin_credit": {}, "calls": {}, "reward_mean": {},
               "blooms": {}, "next_food_id": max(self.foods, default=0) + 1,
               "predators": [{"id": n + 1, "position": [rng.random() * cfg.width, rng.random() * cfg.height],
                              "cooldown": 0, "strikes": 0} for n in range(cfg.predators)]}
        if cfg.bloom_interval:
            self.ledger.setdefault("food_bloomed", 0.)
            self.ledger.setdefault("food_spoiled", 0.)
        return eco

    def _health_factor(self, agent):
        return .5 + .5 * agent.health if self.config.health_ability else 1.

    def _ceiling(self, agent):
        return 1. - self.eco["scar"].get(str(agent.id), 0.) if self.eco is not None else 1.

    def _wound(self, agent, amount):
        """Health loss; with permanent_injury a share of it lowers the health ceiling for good."""
        agent.health = max(0., agent.health - amount)
        if self.eco is not None and self.config.permanent_injury and amount > 0:
            key = str(agent.id)
            self.eco["scar"][key] = min(1., self.eco["scar"].get(key, 0.) + self.config.permanent_injury * amount)
            self._bump("permanent_injury", self.config.permanent_injury * amount)

    def _ancestors(self, identity):
        cache = self.__dict__.setdefault("_ancestor_cache", {})
        if identity in cache:
            return cache[identity]
        pedigree, depth, frontier = self.eco["pedigree"], {identity: 0}, [identity]
        for level in range(1, KIN_DEPTH + 1):
            frontier = [p for i in frontier for p in pedigree.get(str(i), []) if p is not None and p not in depth]
            for p in frontier:
                depth.setdefault(p, level)
            if not frontier:
                break
        cache[identity] = depth
        return depth

    def relatedness(self, a, b):
        """Pedigree kinship 2^-(d_a + d_b) through the closest shared ancestor within
        KIN_DEPTH generations (parent-child 0.5, siblings 0.25; for sexual pedigrees the
        max over shared ancestors, a lower bound). 0 between founder lineages."""
        if a == b:
            return 1.
        if self.eco is None:
            return 0.
        left, right = self._ancestors(a), self._ancestors(b)
        best = 0.
        for ancestor, da in left.items():
            db = right.get(ancestor)
            if db is not None:
                best = max(best, 2. ** -(da + db))
        return best

    @staticmethod
    def kin_band(r):
        return "close" if r >= .5 else "kin" if r >= .125 else "distant" if r > 0 else "stranger"

    def _predator_phase(self, events):
        """Predators chase the nearest individual within predator_sight and strike it in
        interaction range: wound = predator_damage x 2 m_p / (m_p + body_mass). After a
        strike a predator is sated for predator_cooldown ticks and wanders. They are
        hazards without an energy budget; their wounds are real health loss."""
        cfg = self.config
        for predator in self.eco["predators"]:
            predator["cooldown"] = max(0, predator["cooldown"] - 1)
            best = None
            for agent in (self.agents.values() if predator["cooldown"] == 0 else ()):  # a sated predator wanders
                d = self.distance(predator["position"], agent.position)
                if d <= cfg.predator_sight and (best is None or (d, agent.id) < best[:2]):
                    best = (d, agent.id, agent)
            if best is None:
                angle = self.eco_rng.random() * 2 * math.pi
                step = cfg.predator_speed * .5
                predator["position"] = self._wrap([predator["position"][0] + math.cos(angle) * step,
                                                   predator["position"][1] + math.sin(angle) * step])
                continue
            distance, _, prey = best
            if distance <= cfg.interaction_range:
                wound = cfg.predator_damage * 2 * cfg.predator_mass / (cfg.predator_mass + prey.anatomy.body_mass)
                self._wound(prey, wound)
                predator["cooldown"] = cfg.predator_cooldown
                predator["strikes"] += 1
                self._bump("predator_strikes")
                events.append({"tick": self.tick, "type": "predation", "agent": prey.id, "predator": predator["id"],
                               "wound": wound, "health": prey.health})
                if prey.health <= 0:
                    self._death(prey, "predation", events)
                continue
            dx, dy = self.delta(predator["position"], prey.position)
            step = min(cfg.predator_speed, distance)
            predator["position"] = self._wrap([predator["position"][0] + dx / distance * step,
                                               predator["position"][1] + dy / distance * step])

    def _bloom_phase(self, events):
        """Short-lived rich patches: one appears every bloom_interval ticks at a random
        place and spoils bloom_ttl ticks later. Food in and out is in the ledger."""
        cfg, blooms = self.config, self.eco["blooms"]
        for key in sorted(blooms, key=int):
            food = self.foods.get(int(key))
            if food is None or food.amount <= 1e-12 or self.tick >= blooms[key]:
                if food is not None:
                    self.ledger["food_spoiled"] += food.amount
                    del self.foods[int(key)]
                del blooms[key]
        if self.tick % cfg.bloom_interval == 0:
            identity = self.eco["next_food_id"]
            self.eco["next_food_id"] += 1
            position = [self.eco_rng.random() * cfg.width, self.eco_rng.random() * cfg.height]
            self.foods[identity] = FoodPatch(identity, position, cfg.bloom_amount, cfg.bloom_amount, 0., 0.)
            blooms[str(identity)] = self.tick + cfg.bloom_ttl
            self.ledger["food_bloomed"] += cfg.bloom_amount
            self._bump("blooms")
            if cfg.log == "compact":
                events.append({"tick": self.tick, "type": "bloom", "food": identity, "position": position})

    def _kin_credit(self, rewards):
        """Sender credit route (Config.kin_credit), part 1: what each call earns this tick.

        For kin_window ticks after a relative hears a call, the call earns kin_credit x
        relatedness x the hearer's advantage (its reward minus its running mean,
        KIN_BASELINE_RATE). Returns {message id: credit}; _credit_calls books it on the
        call decision that caused the hearing, never on what the sender does later. It
        never depends on which symbol was sent. Updates the hearers' reward baselines."""
        eco, cfg = self.eco, self.config
        pending, calls, means = eco["kin_credit"], eco["calls"], eco["reward_mean"]
        advantage = {i: reward - means.get(str(i), reward) for i, reward in rewards.items()}
        credit = {}
        for key in sorted(pending):
            message, sender, receiver, r, start, until = pending[key]
            if start <= self.tick <= until and receiver in rewards and sender in self.agents \
                    and str(message) in calls:
                credit[message] = credit.get(message, 0.) + cfg.kin_credit * r * advantage[receiver]
            if self.tick >= until or sender not in self.agents or receiver not in self.agents:
                del pending[key]
        for identity in sorted(rewards):
            if identity in self.agents:
                old = means.get(str(identity))
                means[str(identity)] = rewards[identity] if old is None else \
                    old + KIN_BASELINE_RATE * (rewards[identity] - old)
        return credit

    def _credit_calls(self, credit, events):
        """Kin credit, part 2: each call's credit goes to the sender's decision to make that
        call (learning.credit_decision on the features and signal action stored when the
        call was learned), discounted by gamma ** (ticks since the call). Then forgets call
        records that can earn nothing more (expired, sender dead, or no kin can still hear)."""
        calls, full = self.eco["calls"], self.config.log == "full"
        for message in sorted(credit):
            record = calls[str(message)]
            sender = self.agents[record["sender"]]
            amount = credit[message] * sender.controller["gamma"] ** (self.tick - record["tick"])
            learning.credit_decision(sender.controller, record["features"], record["action"], amount,
                                     remembered_at=record["remembered"], enabled=self.config.learning)
            if full:
                events.append({"tick": self.tick, "type": "kin_credit", "agent": sender.id, "message_id": message,
                               "action": record["action"], "call_tick": record["tick"], "credit": amount})
        heard = {row[0] for row in self.eco["kin_credit"].values()}
        for key in sorted(calls, key=int):
            record = calls[key]
            if self.tick >= record["until"] or record["sender"] not in self.agents or \
                    (self.tick >= record["tick"] + self.config.message_ttl and int(key) not in heard):
                del calls[key]

    def _remember_call(self, agent, observation, kind, message):
        """Kin credit: the decision behind a call, kept until no relative can still credit it."""
        cfg = self.config
        self.eco["calls"][str(message)] = {
            "sender": agent.id, "tick": self.tick, "action": kind,
            "features": learning.decision_features(agent.controller, observation),
            "remembered": learning.remembered(agent.controller),
            "until": self.tick + cfg.message_ttl + cfg.kin_window}

    def _position(self):
        return [self.rng.random() * self.config.width, self.rng.random() * self.config.height]

    def _wrap(self, position):
        return [position[0] % self.config.width, position[1] % self.config.height]

    def delta(self, a, b):
        return [(b[0] - a[0] + self.config.width / 2) % self.config.width - self.config.width / 2,
                (b[1] - a[1] + self.config.height / 2) % self.config.height - self.config.height / 2]

    def distance(self, a, b):
        return math.hypot(*self.delta(a, b))

    def _new_agent(self, position, energy, anatomy, parent_id=None):
        identity = self.next_agent_id
        self.next_agent_id += 1
        agent = Agent(identity, parent_id, self._wrap(position), 0, energy, 1., anatomy,
                      controller=learning.new_controller(max_memories=self.config.memory_limit), born_tick=self.tick,
                      lifespan=max(1, round(self.config.max_age * self.rng.uniform(.8, 1.2))))
        self.agents[identity] = agent
        return agent

    def _new_object(self, item, position, holder=None, maker=None):
        identity = self.next_object_id
        self.next_object_id += 1
        allowed = {key: copy.deepcopy(item[key]) for key in ("material", "mass", "sharpness", "bond_strength",
                   "temperature", "components", "properties") if key in item}
        obj = Artifact(identity, position=self._wrap(position), holder=holder, maker_id=maker, **allowed)
        self.objects[identity] = obj
        return obj

    def _bump(self, name, amount=1):
        self.counters[name] = self.counters.get(name, 0) + amount

    def _spend(self, agent, cost, ledger="action_energy"):
        paid = min(agent.energy, max(0., float(cost)))
        agent.energy -= paid
        self.ledger[ledger] += paid
        return paid

    def _gain(self, agent, amount):
        kept = min(amount, max(0., self.config.max_energy - agent.energy))
        agent.energy += kept
        self.ledger["energy_wasted"] += amount - kept

    def _local(self, agent, records, radius, predicate=lambda _: True):
        index = self._index
        pool = index.candidates(records, agent.position, radius) if index is not None and index.covers(records) \
            else records.values()
        rows = []
        # Same arithmetic as distance()/delta(), inlined for speed.
        width, height = self.config.width, self.config.height
        x, y = agent.position
        hypot = math.hypot
        for item in pool:
            if predicate(item):
                other = item.position
                distance = hypot((other[0] - x + width / 2) % width - width / 2,
                                 (other[1] - y + height / 2) % height - height / 2)
                if distance <= radius:
                    rows.append((distance, item.id, item))
        rows.sort()
        return [(distance, item) for distance, _, item in rows]

    def _grid(self, tables):
        return GridIndex(self.config.width, self.config.height, GRID_CELL, tables) if self.spatial_index else None

    def _enabled_actions(self):
        return [a for a in ACTIONS if not (not self.config.tools and a in ("collect", "combine", "dismantle", "heat"))
                and not (not self.config.communication and a.startswith("signal_"))
                and not (not self.config.culture and a in ("teach", "imitate"))]

    def observe(self, agent):
        agent = self.agents[agent] if isinstance(agent, int) else agent
        radius = agent.anatomy.sensor_range * self.config.visibility * self._health_factor(agent)  # 1.0 factors are exact
        def locate(distance, obj):
            dx, dy = self.delta(agent.position, obj.position)
            return {"id": obj.id, "distance": distance, "dx": dx, "dy": dy}
        food = [{**locate(d, f), "amount": f.amount, "hardness": f.hardness, "opening": f.opening}
                for d, f in self._local(agent, self.foods, radius, lambda f: f.amount > 0)]
        full = self.config.log == "full"
        objects = [{**locate(d, o), "material": o.material, "properties": copy.deepcopy(o.properties) if full else o.properties,
                    "mass": o.mass, "sharpness": o.sharpness, "temperature": o.temperature} for d, o in self._local(agent, self.objects, radius, lambda o: o.holder is None)]
        neighbors = [{**locate(d, a), "last_action": a.last_action, "last_outcome": a.last_outcome}
                     for d, a in self._local(agent, self.agents, radius, lambda a: a.id != agent.id)]
        heard = [{**message, "age": self.tick - message["tick"]} for message in agent.memory.get("messages", [])
                 if self.tick - message["tick"] <= self.config.message_ttl] if self.config.communication and agent.anatomy.hearing else []
        if self.config.channel == "masked":  # the call is heard, its symbol is not shown to the controller
            heard = [{key: value for key, value in message.items() if key != "symbol"} for message in heard]
        direction = "none"
        if food:
            dx, dy = food[0]["dx"], food[0]["dy"]
            direction = min(DIRECTIONS, key=lambda name: math.dist(
                [dx / max(math.hypot(dx, dy), 1e-12), dy / max(math.hypot(dx, dy), 1e-12)],
                [v / math.hypot(*DIRECTIONS[name]) for v in DIRECTIONS[name]]))
        reach = self.config.interaction_range
        familiar = agent.memory.get("social", {}).get(str(neighbors[0]["id"]), {}) if neighbors else {}
        tool = self.objects[agent.inventory[0]] if agent.inventory else None
        def band(value, thresholds):
            return "low" if value < thresholds[0] else "middle" if value < thresholds[1] else "high"
        properties = tool.properties if tool else {}
        material_bins = {"held_mass_band": band(tool.mass, (1.5, 3.)) if tool else None,
                         "held_hardness_band": band(properties.get("hardness", 0), (2., 6.)) if tool else None,
                         "held_friction_band": band(properties.get("friction", 0), (.4, .75)) if tool else None,
                         "held_durability_band": band(properties.get("durability", 0) / max(1., properties.get("max_durability", 100)), (.25, .66)) if tool else None,
                         "held_sharpness_band": band(tool.sharpness, (.2, .6)) if tool else None,
                         "held_temperature_band": band(tool.temperature, (25., 70.)) if tool else None,
                         "held_bond_band": band(tool.bond_strength, (.2, .6)) if tool else None,
                         "target_hardness_band": band(food[0]["hardness"], (.1, .8)) if food else None,
                         "target_open_band": band(food[0]["opening"] / max(.001, food[0]["hardness"]), (.25, .8)) if food else None}
        observed = {"self": {"id": agent.id, "energy": agent.energy, "health": agent.health, "age": agent.age,
                          "inventory_count": len(agent.inventory), "anatomy": agent.anatomy.to_dict()},
                "food": food, "objects": objects, "neighbors": neighbors, "messages": heard,
                "food_direction": direction, "food_near": bool(food and food[0]["distance"] <= reach),
                "object_near": bool(objects and objects[0]["distance"] <= reach),
                "neighbor_near": bool(neighbors and neighbors[0]["distance"] <= reach),
                "neighbor_familiar": bool(familiar.get("interactions", 0)),
                "neighbor_last_exchange": familiar.get("last_kind", "none"),
                "energy_band": "low" if agent.energy < 8 else "high" if agent.energy > 28 else "middle",
                "health_band": "hurt" if agent.health < .7 else "well", "inventory_count": len(agent.inventory),
                "tool_available": self.config.tools and bool(agent.inventory), **material_bins}
        if self.eco is not None:
            self._observe_ecology(agent, observed, radius, food, neighbors, heard)
        return observed

    def _observe_ecology(self, agent, observed, radius, food, neighbors, heard):
        """Living-world bins, each present only when its mechanism is on."""
        cfg = self.config
        if cfg.predators:
            seen = None
            for predator in self.eco["predators"]:
                d = self.distance(agent.position, predator["position"])
                if d <= radius * cfg.predator_visibility and (seen is None or (d, predator["id"]) < seen[:2]):
                    seen = (d, predator["id"], predator)
            observed["danger_seen"] = seen is not None
            observed["danger_direction"] = _compass(*self.delta(agent.position, seen[2]["position"])) if seen else "none"
        if cfg.bloom_interval:
            observed["bloom_seen"] = any(str(f["id"]) in self.eco["blooms"] for f in food)
        if cfg.receiver_features:
            for message in heard:
                if "x" in message:
                    message["bearing"] = _compass(*self.delta(agent.position, [message["x"], message["y"]]), here=1.)
            chosen = min(heard, key=lambda m: m["age"]) if heard else None
            observed["heard_age_band"] = None if chosen is None else "fresh" if chosen["age"] <= 1 else \
                "recent" if chosen["age"] <= 4 else "old"
            observed["heard_kin_band"] = None if chosen is None else self.kin_band(chosen.get("kin", 0.))
            own = [m for m in self.messages if m["sender"] == agent.id and self.tick <= m["expires_tick"]]
            observed["own_call"] = own[-1]["symbol"] if own else None
        if cfg.reciprocity_features:
            balance = "none"
            if neighbors:
                record = agent.memory.get("social", {}).get(str(neighbors[0]["id"]), {})
                net = record.get("received", 0.) - record.get("given", 0.)
                balance = "i_owe" if net > .5 else "they_owe" if net < -.5 else "even"
            observed["neighbor_balance"] = balance

    def _target(self, agent, records, action, predicate=lambda _: True):
        if "target" in action:
            target = records.get(action["target"])
            if target is not None and predicate(target) and self.distance(agent.position, target.position) <= self.config.interaction_range:
                return target
            return None
        local = self._local(agent, records, self.config.interaction_range, predicate)
        return local[0][1] if local else None

    def _social(self, source, target, kind, amount=1., mutual=True):
        """Record an interaction. ``mutual=False`` when the target did nothing and
        need not notice (imitation of a passive demonstrator)."""
        key = f"{source.id}:{target.id}"
        edge = self.social_edges.setdefault(key, {"source": source.id, "target": target.id, "counts": {}, "last_tick": self.tick})
        edge["counts"][kind] = edge["counts"].get(kind, 0) + amount
        edge["last_tick"] = self.tick
        for a, b in ((source, target), (target, source)) if mutual else ((source, target),):
            memory = a.memory.setdefault("social", {})
            record = memory.setdefault(str(b.id), {"last_tick": self.tick, "interactions": 0})
            record["last_tick"] = self.tick
            record["interactions"] += 1
            record["last_kind"] = kind
            record["last_direction"] = "outgoing" if a.id == source.id else "incoming"
            while len(memory) > self.config.memory_limit:
                oldest = min(memory, key=lambda k: (memory[k]["last_tick"], int(k)))
                del memory[oldest]

    def _deliver_messages(self, events):
        self.messages = [m for m in self.messages if self.tick <= m["expires_tick"]]
        if not self.config.communication:
            return
        full = self.config.log == "full"
        index = self._grid((self.agents,)) if self.messages else None
        for message in self.messages:
            if index is not None:
                receivers = sorted(a.id for a in index.candidates(self.agents, message["position"], message["range"]))
            else:
                receivers = sorted(self.agents)
            for identity in receivers:
                receiver = self.agents[identity]
                distance = self.distance(receiver.position, message["position"])
                if identity == message["sender"] or identity in message["delivered_to"] or not receiver.anatomy.hearing or distance > message["range"]:
                    continue
                symbol = message["symbol"]
                noisy = self.rng.random() < self.config.signal_noise
                if noisy:
                    symbol = (symbol + self.rng.randint(1, 3)) % 4
                if self.config.channel == "scrambled":  # content destroyed; world RNG stream untouched
                    symbol = self.channel_rng.randrange(4)
                # "tick" is when the call was made, so a late receiver sees its true age.
                record = {"sender": message["sender"], "symbol": symbol,
                    "tick": message["created_tick"], "heard_tick": self.tick, "distance": distance,
                    "message_id": message["id"]}
                if self.eco is not None and (self.config.receiver_features or self.config.kin_credit):
                    r = self.relatedness(identity, message["sender"])
                    if self.config.receiver_features:  # where the call came from and who made it, as perceived
                        record.update(x=message["position"][0], y=message["position"][1], kin=r)
                    if self.config.kin_credit and r > 0 and message["sender"] in self.agents:
                        # one pending hearing per (call, relative); it credits that call only
                        self.eco["kin_credit"][f"{message['id']}:{identity}"] = [
                            message["id"], message["sender"], identity, r, self.tick + 1,
                            self.tick + self.config.kin_window]
                receiver.memory.setdefault("messages", []).append(record)
                receiver.memory["messages"] = receiver.memory["messages"][-self.config.memory_limit:]
                message["delivered_to"].append(identity)
                self._bump("signals_received")
                if full:
                    events.append({"tick": self.tick, "type": "message_received", "agent": identity,
                                   "sender": message["sender"], "symbol": symbol, "noisy": noisy,
                                   "age": self.tick - message["created_tick"]})
                else:
                    self.pending_heard.append({"tick": self.tick, "agent": identity, "sender": message["sender"],
                        "message_id": message["id"], "symbol": symbol, "sent_symbol": message["symbol"],
                        "noisy": noisy, "age": self.tick - message["created_tick"], "distance": distance})

    def _tool_update(self, tool, result):
        if tool:
            for key in ("properties", "temperature", "sharpness", "bond_strength"):
                if key in result:
                    setattr(tool, key, copy.deepcopy(result[key]))

    def _fertile(self, agent):
        cfg = self.config
        return not (agent.age < genome.maturity_age(agent, cfg) or agent.cooldown
                    or agent.energy < cfg.reproduction_threshold or agent.health < .65)

    def _birth(self, agent, events):
        """One birth. Asexual (default): the parent alone pays and its anatomy mutates
        with the world RNG exactly as in Matrix. Sexual: the nearest fertile, willing
        partner in interaction range (willing = automatic reproduction, or its last
        action was reproduce) is required; each parent pays the full birth cost and half
        of the offspring energy, and every gene comes from one parent chosen at random.
        With Config.genome="evolving" the child's genome is inherited from the parents'
        genomes (never from their learned weights) and seeds its controller. Genome
        draws use genome.rng_for(seed, child id), so the frozen default is unchanged."""
        cfg = self.config
        if not self._fertile(agent):
            return False
        if len(self.agents) >= cfg.population_limit:
            self.stop_reason = "population_capacity"
            events.append({"tick": self.tick, "type": "capacity_pause", "capacity": cfg.population_limit})
            return False
        partner = None
        if cfg.reproduction == "sexual":
            mates = self._local(agent, self.agents, cfg.interaction_range, lambda a: a.id != agent.id and self._fertile(a)
                                and (cfg.automatic_reproduction or a.last_action == "reproduce"))
            if not mates:
                return False
            partner = mates[0][1]
        grng = genome.rng_for(self.seed, self.next_agent_id) if partner is not None or cfg.genome == "evolving" else None
        genes = agent.anatomy.to_dict() if partner is None else genome.recombine_anatomy(
            agent.anatomy.to_dict(), partner.anatomy.to_dict(), grng)
        draw = self.rng if partner is None else grng
        changed = []
        for name in sorted(genes):
            if draw.random() < (cfg.organ_flip_rate if name in ("voice", "hearing") else cfg.mutation_rate):
                if name in ("voice", "hearing"):
                    genes[name] = 1 - genes[name]
                else:
                    genes[name] = min(2.5, max(.3, genes[name] * math.exp(draw.uniform(-cfg.mutation_scale, cfg.mutation_scale))))
                changed.append(name)
        if partner is None:
            paid = self._spend(agent, cfg.reproduction_cost, "birth_cost")
            agent.energy -= cfg.offspring_energy
            # The ledger keeps the transfer; the learning reward does not (see step()).
            self._birth_transfer[agent.id] = self._birth_transfer.get(agent.id, 0.) + paid + cfg.offspring_energy
        else:
            for parent in (agent, partner):
                paid = self._spend(parent, cfg.reproduction_cost, "birth_cost")
                parent.energy -= cfg.offspring_energy / 2
                self._birth_transfer[parent.id] = self._birth_transfer.get(parent.id, 0.) + paid + cfg.offspring_energy / 2
                parent.cooldown = cfg.birth_cooldown
        child = self._new_agent([agent.position[0] + self.rng.uniform(-.5, .5), agent.position[1] + self.rng.uniform(-.5, .5)],
                                cfg.offspring_energy, Anatomy(**genes), agent.id)
        if cfg.genome == "evolving":
            child.genome = genome.inherit(agent.genome, cfg, grng, partner.genome if partner is not None else None)
            child.controller = genome.seed_controller(child.genome, max_memories=cfg.memory_limit)
            if cfg.age_effects:
                child.lifespan = max(1, round(child.lifespan * genome.expressed(child.genome, cfg)["lifespan"] / cfg.max_age))
        if self.eco is not None:
            self.eco["pedigree"][str(child.id)] = [agent.id] + ([partner.id] if partner is not None else [])
            source = agent if partner is None or self.eco_rng.random() < .5 else partner
            power = self.eco["call_power"].get(str(source.id), 1.)
            if self.eco_rng.random() < cfg.mutation_rate:
                power = min(2.5, max(.3, power * math.exp(self.eco_rng.uniform(-cfg.mutation_scale, cfg.mutation_scale))))
            self.eco["call_power"][str(child.id)] = power
        agent.cooldown = cfg.birth_cooldown
        self._bump("births")
        event = {"tick": self.tick, "type": "birth", "parent": agent.id, "agent": child.id,
                 "mutations": changed, "energy_transferred": cfg.offspring_energy}
        if partner is not None:
            event["partner"] = partner.id
        events.append(event)
        return True

    def _execute(self, agent, action, observation, events):
        kind = action["type"]
        detail = {"success": False}
        self._spend(agent, self.config.action_cost * (.25 if kind == "rest" else 1.))
        if kind == "rest":
            ceiling = self._ceiling(agent)
            if agent.energy > 3 and agent.health < ceiling:
                paid = self._spend(agent, .02)
                agent.health = min(ceiling, agent.health + .004 * paid / .02)
            detail["success"] = True
        elif kind.startswith("move_"):
            dx, dy = DIRECTIONS[kind[5:]]
            distance = agent.anatomy.move_distance * max(.2, agent.health) * genome.ability(agent, self.config)
            load = sum(self.objects[i].mass for i in agent.inventory)
            required = .08 * (agent.anatomy.body_mass + .3 * load) * distance ** 2
            distance *= math.sqrt(self._spend(agent, required) / max(required, 1e-12))
            norm = math.hypot(dx, dy)
            agent.position = self._wrap([agent.position[0] + dx / norm * distance, agent.position[1] + dy / norm * distance])
            detail.update(success=distance > 0, distance=distance)
        elif kind == "eat":
            food = self._target(agent, self.foods, action, lambda f: f.amount > 0)
            bloom = food is not None and self.eco is not None and str(food.id) in self.eco["blooms"]
            if bloom and self.config.bloom_partners > 1 and sum(
                    1 for other in self.agents.values() if other.id != agent.id
                    and self.distance(other.position, food.position) <= self.config.interaction_range) < self.config.bloom_partners - 1:
                detail.update(target=food.id, reason="bloom needs partners")
                food = None
            if food:
                remaining_hardness = max(0., food.hardness - food.opening)
                efficiency = max(.05, 1. - remaining_hardness / max(.2, agent.anatomy.manipulation * genome.ability(agent, self.config)
                                                                    * self._health_factor(agent)))
                quantity = min(food.amount, efficiency)
                food.amount -= quantity
                energy = quantity * self.config.food_energy * (self.config.bloom_richness if bloom else 1.)
                if bloom:
                    self._bump("bloom_meals")
                self._gain(agent, energy)
                self.ledger["food_eaten"] += quantity
                self.ledger["energy_from_food"] += energy
                self._bump("meals")
                detail.update(success=True, target=food.id, quantity=quantity, energy=energy)
        elif kind == "collect" and self.config.tools:
            obj = self._target(agent, self.objects, action, lambda o: o.holder is None)
            if obj and len(agent.inventory) < self.config.inventory_limit and obj.mass <= 4 * agent.anatomy.manipulation:
                obj.holder = agent.id
                agent.inventory.append(obj.id)
                detail.update(success=True, target=obj.id)
        elif kind == "drop":
            identity = action.get("object", action.get("target", agent.inventory[0] if agent.inventory else None))
            if identity in agent.inventory:
                agent.inventory.remove(identity)
                self.objects[identity].holder = None
                self.objects[identity].position = list(agent.position)
                detail.update(success=True, target=identity)
        elif kind in ("strike", "strike_agent", "strike_object"):
            target_kind = action.get("target_kind", {"strike_agent": "agent", "strike_object": "object"}.get(kind, "food"))
            records = self.agents if target_kind == "agent" else self.objects if target_kind == "object" else self.foods
            tool = self.objects[agent.inventory[0]] if self.config.tools and agent.inventory else None
            target = self._target(agent, records, action, lambda x: x is not agent and x is not tool
                                  and (target_kind != "object" or x.holder in (None, agent.id)))
            if target:
                effort = self._spend(agent, min(1.5, float(action.get("effort", 1.))))
                hardness = target.hardness if target_kind == "food" else target.properties.get("hardness", 1.) if target_kind == "object" else .5
                skill = agent.anatomy.manipulation * genome.ability(agent, self.config) * self._health_factor(agent)
                result = materials.strike(tool, hardness, effort, manipulation=skill)
                self._tool_update(tool, result)
                self.ledger["heat_to_objects"] += result.get("tool_heat", 0.)
                damage = max(0., result["damage"])
                # Damage the target can still absorb: soft or already-open food absorbs none.
                # Organism strikes: strike_injury per unit damage (0.08 = Matrix), optionally
                # scaled by attacker / target body mass (Config.injury_mass_scaling).
                per_damage = self.config.strike_injury * (agent.anatomy.body_mass / target.anatomy.body_mass
                                                          if target_kind == "agent" and self.config.injury_mass_scaling else 1.)
                room = max(0., target.hardness - target.opening) if target_kind == "food" else \
                    target.health / per_damage if target_kind == "agent" else max(0., target.properties.get("durability", 1.))
                if target_kind == "food":
                    target.opening = min(target.hardness, target.opening + damage)
                elif target_kind == "agent":
                    self._wound(target, per_damage * damage)
                    self._social(agent, target, "strikes")
                    if target.health <= 0:
                        self._death(target, "injury", events)
                else:
                    target.properties["durability"] = max(0., target.properties.get("durability", 1.) - damage)
                    delivered = result.get("delivered_work", 0.)
                    self._tool_update(target, materials.heat_item(target, delivered))
                    self.ledger["heat_to_objects"] += delivered
                if tool:
                    baseline = materials.strike(None, hardness, effort, manipulation=skill)
                    self._bump("tool_strikes")
                    self._bump("tool_damage_gain", min(damage, room) - min(max(0., baseline["damage"]), room))
                    use = self.artifact_stats.setdefault(str(tool.id), {"uses": 0, "delivered_work": 0.})
                    use["uses"] += 1
                    use["delivered_work"] += result.get("delivered_work", 0.)
                detail.update(success=True, target=target.id, target_kind=target_kind, damage=damage,
                              effective_damage=min(damage, room), tool=tool.id if tool else None)
        elif kind == "combine" and self.config.tools and len(agent.inventory) >= 2:
            left, right = (self.objects[i] for i in agent.inventory[:2])
            work = self._spend(agent, min(MAX_WORK_EFFORT, float(action.get("effort", .8))))
            if work <= 0:
                return detail
            result = materials.combine_items(left, right, work)
            self.ledger["heat_to_objects"] += result["heat_energy"]
            item = result["item"]
            item["components"] = [left.id, right.id]
            new = self._new_object(item, agent.position, agent.id, agent.id)
            for old in (left, right):
                agent.inventory.remove(old.id)
                del self.objects[old.id]
            agent.inventory.insert(0, new.id)  # the new composite is the held tool
            self._bump("combinations")
            detail.update(success=True, artifact=new.id, consumed=[left.id, right.id])
        elif kind == "dismantle" and self.config.tools and agent.inventory:
            obj = self.objects[agent.inventory[0]]
            work = self._spend(agent, min(MAX_WORK_EFFORT, float(action.get("effort", .8))))
            result = materials.dismantle_item(obj, work)
            self.ledger["heat_to_objects"] += result["heat_energy"]
            if result["success"]:
                if len(self.objects) - 1 + len(result["items"]) > self.config.object_limit:
                    self.stop_reason = "object_capacity"
                    detail["reason"] = "object capacity pause"
                    self._tool_update(obj, materials.heat_item(obj, result["heat_energy"]))
                else:
                    agent.inventory.remove(obj.id)
                    del self.objects[obj.id]
                    created = []
                    for item in result["items"]:
                        held = len(agent.inventory) < self.config.inventory_limit
                        new = self._new_object(item, agent.position, agent.id if held else None, agent.id)
                        if held:
                            agent.inventory.append(new.id)
                        created.append(new.id)
                    detail.update(success=True, artifacts=created, consumed=obj.id)
                    self._bump("dismantlings")
            elif "item" in result:
                self._tool_update(obj, result["item"])
        elif kind == "heat" and self.config.tools and agent.inventory:
            obj = self.objects[agent.inventory[0]]
            energy = self._spend(agent, min(MAX_WORK_EFFORT, float(action.get("effort", 1.))))
            result = materials.heat_item(obj, energy)
            self._tool_update(obj, result)
            self.ledger["heat_to_objects"] += result["energy"]
            detail.update(success=True, artifact=obj.id, heat=result["energy"])
        elif kind == "share":
            target = self._target(agent, self.agents, action, lambda a: a.id != agent.id)
            if target:
                quantity = min(2., max(0., agent.energy - 2.), self.config.max_energy - target.energy)
                if quantity > 0:  # a share that moves nothing is not a social exchange
                    agent.energy -= quantity
                    target.energy += quantity
                    self._social(agent, target, "energy_shared", quantity)
                    if self.config.reciprocity_features:  # directed per-peer tallies
                        given = agent.memory["social"].get(str(target.id))
                        got = target.memory["social"].get(str(agent.id))
                        if given is not None:
                            given["given"] = given.get("given", 0.) + quantity
                        if got is not None:
                            got["received"] = got.get("received", 0.) + quantity
                    self._bump("shared_energy", quantity)
                detail.update(success=quantity > 0, target=target.id, energy=quantity)
        elif kind.startswith("signal_") and self.config.communication and agent.anatomy.voice:
            power = self.eco["call_power"].get(str(agent.id), 1.) if self.config.call_power and self.eco is not None else None
            # Matrix: reach and cost come from the sensing gene; with call_power, from the voice-power gene.
            required = self.config.signal_cost * ((1 + .15 * agent.anatomy.sensing) if power is None else (1 + .3 * power))
            paid = self._spend(agent, required)
            if paid + 1e-12 >= required:
                reach = 2. + 3. * agent.anatomy.sensing if power is None else self.config.call_reach * power
                message = {"id": self.next_message_id, "sender": agent.id, "symbol": int(kind[-1]),
                           "position": list(agent.position), "created_tick": self.tick,
                           "expires_tick": self.tick + self.config.message_ttl, "range": reach, "delivered_to": []}
                self.next_message_id += 1
                self.messages.append(message)
                self._bump("signals_sent")
                detail.update(success=True, message_id=message["id"], symbol=message["symbol"], range=reach)
                if self.config.log == "compact":
                    events.append({"tick": self.tick, "type": "signal", "agent": agent.id, "symbol": message["symbol"],
                                   "message_id": message["id"], "range": reach,
                                   "sender": {band: observation.get(band) for band in SENDER_BANDS
                                              + tuple(b for b in ECOLOGY_SENDER_BANDS if b in observation)}})
        elif kind in ("imitate", "teach") and self.config.culture:
            target = self._target(agent, self.agents, action, lambda a: a.id != agent.id)
            if target:
                paid = self._spend(agent, self.config.culture_cost)
                teacher, learner = (target, agent) if kind == "imitate" else (agent, target)
                demo = teacher.memory.get("last_demo")
                if demo and paid + 1e-12 >= self.config.culture_cost:
                    adopted = learning.observe_demo(learner.controller, demo["observation"], demo["action"], demo["outcome"],
                                                    enabled=self.config.learning, next_observation=demo.get("next_observation"),
                                                    actions=self._enabled_actions())
                    if kind == "teach":
                        self._social(teacher, learner, "demonstrations")
                    else:  # the demonstrator was passive: only the imitator records the interaction
                        self._social(learner, teacher, "imitations", mutual=False)
                    self._bump("demonstrations")
                    detail.update(success=bool(adopted), teacher=teacher.id, learner=learner.id, demonstrated_action=demo["action"])
        elif kind == "reproduce":
            detail["success"] = self._birth(agent, events)
        return detail

    def _death(self, agent, reason, events):
        for identity in agent.inventory:
            self.objects[identity].holder = None
            self.objects[identity].position = list(agent.position)
        if self.eco is not None:
            self.eco["scar"].pop(str(agent.id), None)
            self.eco["call_power"].pop(str(agent.id), None)
            self.eco["reward_mean"].pop(str(agent.id), None)
        self.ledger["dead_energy"] += agent.energy
        agent.energy = 0.
        agent.health = 0.
        del self.agents[agent.id]
        self._bump("deaths")
        events.append({"tick": self.tick, "type": "death", "agent": agent.id, "reason": reason,
                       "released_objects": list(agent.inventory)})

    def step(self, actions=None):
        if self.stop_reason:
            return []
        if actions is not None and not isinstance(actions, dict):
            raise ValueError("actions must be keyed by integer agent ID")
        overrides = actions or {}
        normalized = {}
        for identity, value in overrides.items():
            if type(identity) is not int or identity not in self.agents:
                raise ValueError("action override references an unknown agent")
            action = {"type": value} if isinstance(value, str) else copy.deepcopy(value)
            if not isinstance(action, dict) or action.get("type") not in ACTIONS:
                raise ValueError("unsupported action override")
            if set(action) - {"type", "target", "target_kind", "object", "effort"}:
                raise ValueError("unknown action parameter")
            if any(type(action[key]) is not int or action[key] < 1 for key in ("target", "object") if key in action):
                raise ValueError("target/object IDs must be positive integers")
            if "target_kind" in action and action["target_kind"] not in ("food", "agent", "object"):
                raise ValueError("unknown target kind")
            if "effort" in action and (isinstance(action["effort"], bool) or not isinstance(action["effort"], (int, float)) or not math.isfinite(action["effort"]) or action["effort"] < 0):
                raise ValueError("effort must be finite and nonnegative")
            normalized[identity] = action
        events = []
        full = self.config.log == "full"
        self._birth_transfer = {}
        # Decide from exactly the previous completed boundary. Advance the clock
        # only after observing/queuing, so message expiry and ages do not change
        # invisibly between a saved next_observation and the following decision.
        starting = {identity: self.agents[identity] for identity in sorted(self.agents)}
        self._index = self._grid((self.foods, self.objects, self.agents))
        try:
            before = {identity: {"energy": agent.energy, "health": agent.health, "observation": self.observe(agent)}
                      for identity, agent in starting.items()}
        finally:
            self._index = None
        enabled = self._enabled_actions()
        queued = {}
        for identity, agent in starting.items():
            chosen = learning.choose_action(agent.controller, before[identity]["observation"], self.rng,
                enabled=self.config.learning, actions=enabled) if identity not in normalized else normalized[identity]["type"]
            queued[identity] = normalized.get(identity, {"type": chosen})
        if self.pending_heard:
            # Compact log: each heard message with the receiver's state and the action it chose next.
            for row in self.pending_heard:
                receiver = before.get(row["agent"])
                events.append({**row, "type": "heard", "next_action": queued[row["agent"]]["type"] if receiver else None,
                               "receiver": {band: receiver["observation"].get(band) for band in RECEIVER_BANDS
                                            + tuple(b for b in ECOLOGY_RECEIVER_BANDS if b in receiver["observation"])}
                               if receiver else None})
            self.pending_heard = []
        self.tick += 1
        order = list(starting)
        self.rng.shuffle(order)
        outcomes = {}
        for identity in order:
            if identity not in self.agents:
                outcomes[identity] = {"success": False, "executed": False, "reason": "died before queued action"}
                continue
            agent = self.agents[identity]
            action = queued[identity]
            kind = action["type"]
            detail = self._execute(agent, action, before[identity]["observation"], events)
            detail["executed"] = True
            outcomes[identity] = detail
            if not full and kind in COMPACT_ACTIONS:
                events.append({"tick": self.tick, "type": "act", "agent": identity, "action": kind, **detail})
            for obj_id in agent.inventory:
                obj = self.objects[obj_id]
                excess = max(0., obj.temperature - self.config.safe_touch_temperature)
                capacity = materials.total_heat_capacity(obj)
                transferred = min(excess * capacity, excess * self.config.thermal_touch_conductance)
                if transferred:
                    self._tool_update(obj, materials.heat_item(obj, -transferred))
                    injury = transferred * self.config.thermal_injury_per_energy / agent.anatomy.body_mass
                    agent.health = max(0., agent.health - injury)
                    self.ledger["heat_to_body"] += transferred
                    self._bump("thermal_contacts")
                    events.append({"tick": self.tick, "type": "thermal_contact", "agent": agent.id,
                                   "artifact": obj.id, "transferred_energy": transferred, "injury": injury})
            self._spend(agent, self.config.base_metabolism * agent.anatomy.metabolic_factor * genome.upkeep(agent, self.config), "metabolic_energy")
            agent.age += 1
            agent.cooldown = max(0, agent.cooldown - 1)
            for obj_id in agent.inventory:
                self.objects[obj_id].position = list(agent.position)
            if agent.energy <= 0 or agent.health <= 0 or agent.age >= agent.lifespan:
                self._death(agent, "starvation" if agent.energy <= 0 else "injury" if agent.health <= 0 else "age", events)
        if self.eco is not None and self.config.predators:
            self._predator_phase(events)
        if self.config.automatic_reproduction:
            for identity in sorted(starting):
                if identity in self.agents:
                    self._birth(self.agents[identity], events)
        regrown = 0.
        for food in self.foods.values():
            added = min(food.regrowth, food.capacity - food.amount)
            food.amount += added
            regrown += added
            food.opening *= .995
        self.ledger["food_regrown"] += regrown
        if self.eco is not None and self.config.bloom_interval:
            self._bloom_phase(events)
        for obj in self.objects.values():
            cooling = max(0., obj.temperature - self.config.ambient_temperature) * .02
            obj.temperature -= cooling
            self.ledger["heat_to_environment"] += cooling * obj.mass * obj.properties.get("heat_capacity", 1.)
        if not self.agents:
            self.stop_reason = "extinction"
        self._deliver_messages(events)
        # Finalize every starting organism after ALL ecological effects. Incoming
        # harm/help and reproductive transfers cannot disappear between rewards.
        # The energy a parent hands to its offspring (and the birth cost) stays in the
        # ledger but is not booked against whichever action the parent happened to take.
        transitions = {}
        for identity, agent in starting.items():
            energy_delta = agent.energy - before[identity]["energy"]
            health_delta = agent.health - before[identity]["health"]
            birth = self._birth_transfer.get(identity, 0.)
            reward = energy_delta + 8 * health_delta + (0. if self.config.birth_transfer_in_reward else birth)
            done = identity not in self.agents
            agent.last_action, agent.last_outcome = queued[identity]["type"], reward
            transitions[identity] = (reward, energy_delta, health_delta, done, birth)
        credit = self._kin_credit({i: t[0] for i, t in transitions.items()}) \
            if self.eco is not None and self.config.kin_credit else {}
        self._index = self._grid((self.foods, self.objects, self.agents))
        try:
            for identity, agent in starting.items():
                reward, energy_delta, health_delta, done, birth = transitions[identity]
                if done:
                    following = {"terminal": True, "self": {"id": identity, "energy": 0., "health": 0., "age": agent.age},
                                 "food": [], "objects": [], "neighbors": [], "messages": []}
                else:
                    following = self.observe(agent)
                transitions[identity] = (reward, energy_delta, health_delta, done, birth, following)
        finally:
            self._index = None
        for identity, agent in starting.items():
            observation = before[identity]["observation"]
            action = queued[identity]
            kind = action["type"]
            reward, energy_delta, health_delta, terminal, birth, following = transitions[identity]
            # Every transition learns the organism's own outcome only. Kin credit reaches a
            # call's own decision later (_credit_calls), never the sender's later actions.
            learning.learn(agent.controller, observation, kind, reward, following,
                           enabled=self.config.learning, terminal=terminal, actions=enabled)
            if self.eco is not None and self.config.kin_credit and not terminal and kind.startswith("signal_") \
                    and "message_id" in outcomes[identity]:
                self._remember_call(agent, observation, kind, outcomes[identity]["message_id"])
            agent.last_action, agent.last_outcome = kind, reward
            agent.memory.setdefault("observations", []).append({"tick": self.tick, "action": kind, "reward": reward,
                                                                 "food_direction": observation["food_direction"]})
            agent.memory["observations"] = agent.memory["observations"][-self.config.memory_limit:]
            if outcomes[identity]["executed"] and not terminal and kind not in ("teach", "imitate", "reproduce") \
                    and not kind.startswith("signal_"):
                # What a nearby learner could see: the demonstrator's situation before and
                # after, its action and its outcome; not its internal bands or heard symbol.
                context = {key: observation[key] for key in DEMO_FEATURES if key in observation}
                after = {key: following[key] for key in DEMO_FEATURES if key in following}
                agent.memory["last_demo"] = {"observation": context, "action": kind, "outcome": reward,
                                             "next_observation": after, "tick": self.tick}
            if outcomes[identity]["executed"]:
                counts = self.action_counts.setdefault(str(identity), {})
                counts[kind] = counts.get(kind, 0) + 1
            if full:
                events.append({"tick": self.tick, "type": "action", "agent": identity, "action": kind,
                               "reward": reward, "birth_transfer": birth, "energy_before": before[identity]["energy"],
                               "energy_after": agent.energy, "health_before": before[identity]["health"],
                               "health_after": agent.health, **outcomes[identity]})
                events.append({"tick": self.tick, "type": "transition", "agent": identity,
                               "observation": observation, "action": copy.deepcopy(action), "reward": reward,
                               "reward_components": {"energy_delta": energy_delta, "health_delta": health_delta,
                                                     "health_weight": 8., "birth_transfer": birth,
                                                     "birth_transfer_in_reward": self.config.birth_transfer_in_reward,
                                                     "includes_death": True},
                               "next_observation": following, "done": terminal, "executed": outcomes[identity]["executed"],
                               "timing": "full ecological tick including other organisms, physiology and resources"})
        if self.eco is not None and self.config.kin_credit:
            self._credit_calls(credit, events)
        if full:
            events.append({"tick": self.tick, "type": "environment", "food_regrown": regrown, "stop_reason": self.stop_reason})
        return events

    def summary(self):
        living = sorted(self.agents.values(), key=lambda a: a.id)
        specialties, dominant = [], Counter()
        for agent in living:
            counts = self.action_counts.get(str(agent.id), {})
            if counts:
                specialties.append(max(counts.values()) / sum(counts.values()))
                dominant[min(counts, key=lambda action: (-counts[action], action))] += 1
        return {"tick": self.tick, "population": len(living), "objects": len(self.objects), "food": sum(f.amount for f in self.foods.values()),
                "mean_energy": sum(a.energy for a in living) / len(living) if living else 0.,
                "mean_age": sum(a.age for a in living) / len(living) if living else 0.,
                "mean_health": sum(a.health for a in living) / len(living) if living else 0.,
                "counters": dict(self.counters), "social_edge_count": len(self.social_edges),
                # None when no living individual has acted (e.g. after extinction), not 0.
                # The share of each individual's most frequent action; dominant_actions
                # shows which action that is (usually eat), so it is not a specialization index.
                "mean_action_concentration": sum(specialties) / len(specialties) if specialties else None,
                "dominant_actions": dict(sorted(dominant.items())),
                "composite_objects": sum(bool(o.components) for o in self.objects.values()),
                "stop_reason": self.stop_reason,
                "interpretation": "Measured prototype behavior; no claim of autonomous technology, language, or society emergence."}

    def snapshot(self):
        def located(obj):
            return {**obj.to_dict(), "x": obj.position[0], "y": obj.position[1]}
        return {"tick": self.tick, "config": self.config.to_dict(), "summary": self.summary(),
                "agents": [{k: v for k, v in located(a).items() if k not in ("controller", "memory")}
                           for a in sorted(self.agents.values(), key=lambda a: a.id)],
                "objects": [located(o) for o in sorted(self.objects.values(), key=lambda o: o.id)],
                "foods": [located(f) for f in sorted(self.foods.values(), key=lambda f: f.id)],
                "messages": copy.deepcopy(self.messages), "social_edges": copy.deepcopy(sorted(
                    self.social_edges.values(), key=lambda edge: (edge["source"], edge["target"])))}

    def to_dict(self):
        return copy.deepcopy({"schema_version": SCHEMA_VERSION, "model_version": MODEL_VERSION,
            "seed": self.seed, "config": self.config.to_dict(), "tick": self.tick, "stop_reason": self.stop_reason,
            "rng_state": self.rng.getstate(), "channel_rng_state": self.channel_rng.getstate(),
            "next_agent_id": self.next_agent_id,
            "next_object_id": self.next_object_id, "next_message_id": self.next_message_id,
            "agents": [a.to_dict() for a in sorted(self.agents.values(), key=lambda a: a.id)],
            "objects": [o.to_dict() for o in sorted(self.objects.values(), key=lambda o: o.id)],
            "foods": [f.to_dict() for f in sorted(self.foods.values(), key=lambda f: f.id)], "messages": self.messages,
            "pending_heard": self.pending_heard,
            "counters": self.counters, "ledger": self.ledger, "social_edges": self.social_edges,
            "action_counts": self.action_counts, "artifact_stats": self.artifact_stats,
            **({"ecology": self.eco, "eco_rng_state": self.eco_rng.getstate()} if self.eco is not None else {})})

    @classmethod
    def from_dict(cls, data):
        data = copy.deepcopy(data)
        if data.get("schema_version") != SCHEMA_VERSION or data.get("model_version") != MODEL_VERSION:
            raise ValueError("incompatible life8 state schema/model version (Matrix and older life8 states are refused)")
        world = cls()
        world.config = Config.from_dict(data["config"])
        for key in ("seed", "tick", "stop_reason", "next_agent_id", "next_object_id", "next_message_id", "messages",
                    "pending_heard", "counters", "ledger", "social_edges", "action_counts", "artifact_stats"):
            setattr(world, key, data[key])
        world.agents = {row["id"]: Agent.from_dict(row) for row in sorted(data["agents"], key=lambda row: row["id"])}
        world.objects = {row["id"]: Artifact.from_dict(row) for row in sorted(data["objects"], key=lambda row: row["id"])}
        world.foods = {row["id"]: FoodPatch.from_dict(row) for row in sorted(data["foods"], key=lambda row: row["id"])}
        if any(len(getattr(world, name)) != len(data[name]) for name in ("agents", "objects", "foods")):
            raise ValueError("duplicate entity IDs in checkpoint")
        world.rng = random.Random()
        world.rng.setstate(_tuples(data["rng_state"]))
        # States written before the channel switch never drew from it: a fresh seeded RNG is exact.
        world.channel_rng = random.Random(world.seed ^ CHANNEL_SALT)
        if "channel_rng_state" in data:
            world.channel_rng.setstate(_tuples(data["channel_rng_state"]))
        world.eco_rng = random.Random(world.seed ^ ECOLOGY_SALT)
        world.eco = None
        if world.config.ecology_on():
            if "ecology" not in data or "eco_rng_state" not in data:
                raise ValueError("a living-world config needs its ecology record in the state")
            world.eco = data["ecology"]
            if isinstance(world.eco, dict) and "calls" not in world.eco:
                # Written before kin credit was attributed to the call (fix F1): pending
                # sender-level credits cannot be attributed to a call, so such states are refused.
                if world.eco.get("kin_credit"):
                    raise ValueError("ecology record has kin credit pending in the pre-F1 format; "
                                     "restart the run from a state without pending kin credit")
                world.eco.update(calls={}, reward_mean={})
            world.eco_rng.setstate(_tuples(data["eco_rng_state"]))
        world.validate()
        return world

    def _validate_ecology(self):
        eco, cfg = self.eco, self.config
        if not isinstance(eco, dict) or set(eco) != {"pedigree", "call_power", "scar", "kin_credit", "calls",
                                                     "reward_mean", "blooms", "next_food_id", "predators"}:
            raise ValueError("invalid ecology record")
        for key, row in eco["kin_credit"].items():
            if not isinstance(row, list) or len(row) != 6 or key != f"{row[0]}:{row[2]}":
                raise ValueError("invalid pending kin credit")
        for key, record in eco["calls"].items():
            if not isinstance(record, dict) or not str(record.get("action", "")).startswith("signal_") \
                    or not isinstance(record.get("features"), list):
                raise ValueError("invalid kin-credit call record")
        if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v)
               for v in eco["reward_mean"].values()):
            raise ValueError("invalid kin-credit reward baseline")
        if len(eco["predators"]) != cfg.predators:
            raise ValueError("ecology record disagrees with the configuration")
        for identity, agent in self.agents.items():
            if str(identity) not in eco["pedigree"] or str(identity) not in eco["call_power"]:
                raise ValueError("living organism missing from the ecology record")
            if not .3 <= eco["call_power"][str(identity)] <= 2.5:
                raise ValueError("call power outside prototype bounds")
            if agent.health > self._ceiling(agent) + 1e-9:
                raise ValueError("health above its permanent-injury ceiling")
        if any(not 0 <= v <= 1 for v in eco["scar"].values()):
            raise ValueError("invalid permanent injury")
        for key, expires in eco["blooms"].items():
            if int(key) not in self.foods or type(expires) is not int:
                raise ValueError("invalid bloom record")
        if eco["next_food_id"] <= max(self.foods, default=0):
            raise ValueError("bloom identity counter is not monotonic")
        for predator in eco["predators"]:
            if not 0 <= predator["position"][0] < cfg.width or not 0 <= predator["position"][1] < cfg.height:
                raise ValueError("predator outside the world")

    def validate(self):
        json.dumps(self.to_dict(), allow_nan=False)
        def integer(value, name, low=0, high=None):
            if type(value) is not int or value < low or high is not None and value > high:
                raise ValueError(f"invalid integer {name}")
        def number(value, name, low=None):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or low is not None and value < low:
                raise ValueError(f"invalid number {name}")
        integer(self.tick, "tick")
        integer(self.seed, "seed", 0, 2 ** 64 - 1)
        for name in ("next_agent_id", "next_object_id", "next_message_id"):
            integer(getattr(self, name), name, 1)
        if self.stop_reason not in (None, "extinction", "population_capacity", "object_capacity"):
            raise ValueError("invalid stop reason")
        if (not self.agents) != (self.stop_reason == "extinction"):
            raise ValueError("extinction status disagrees with population")
        if len(self.agents) > self.config.population_limit or len(self.objects) > self.config.object_limit:
            raise ValueError("compute safety capacity exceeded")
        owned = set()
        for identity, agent in self.agents.items():
            learning.validate_controller(agent.controller)
            genome.validate_genome(agent.genome, self.config)
            integer(identity, "agent id", 1, self.next_agent_id - 1)
            integer(agent.age, "agent age")
            integer(agent.born_tick, "birth tick", 0, self.tick)
            integer(agent.lifespan, "lifespan", 1)
            integer(agent.cooldown, "birth cooldown", 0, self.config.birth_cooldown)
            if agent.parent_id is not None:
                integer(agent.parent_id, "parent id", 1, agent.id - 1)
            if agent.age != self.tick - agent.born_tick or agent.age >= agent.lifespan:
                raise ValueError("organism age does not match its lifetime clock")
            if identity != agent.id:
                raise ValueError("invalid organism identity/ancestry")
            number(agent.energy, "agent energy", 0)
            number(agent.health, "agent health", 0)
            number(agent.last_outcome, "last outcome")
            if agent.last_action not in ACTIONS:
                raise ValueError("invalid previous action")
            if not 0 <= agent.energy <= self.config.max_energy + 1e-9 or not 0 <= agent.health <= 1 or agent.age < 0:
                raise ValueError("invalid organism state")
            if len(agent.inventory) > self.config.inventory_limit or len(set(agent.inventory)) != len(agent.inventory):
                raise ValueError("invalid inventory")
            for key, value in agent.anatomy.to_dict().items():
                if key in ("voice", "hearing"):
                    if value not in (0, 1): raise ValueError("invalid sensory gene")
                elif not .3 <= value <= 2.5:
                    raise ValueError("anatomy gene outside prototype bounds")
            for obj_id in agent.inventory:
                if obj_id in owned or obj_id not in self.objects or self.objects[obj_id].holder != identity:
                    raise ValueError("inconsistent held-object ownership")
                owned.add(obj_id)
            memory = agent.memory
            if not isinstance(memory, dict) or not isinstance(memory.get("messages"), list) or not isinstance(memory.get("observations"), list) or not isinstance(memory.get("social"), dict):
                raise ValueError("invalid local memory containers")
            if any(len(memory[key]) > self.config.memory_limit for key in ("messages", "observations", "social")):
                raise ValueError("local memory exceeds configured capacity")
            for message in memory["messages"]:
                if not isinstance(message, dict): raise ValueError("invalid received message")
                integer(message.get("tick"), "received message tick", 0, self.tick)
                integer(message.get("message_id"), "received message id", 1, self.next_message_id - 1)
                integer(message.get("sender"), "received message sender", 1, self.next_agent_id - 1)
                integer(message.get("symbol"), "heard symbol", 0, 3)
                number(message.get("distance"), "heard distance", 0)
            for row in memory["observations"]:
                if not isinstance(row, dict) or row.get("action") not in ACTIONS:
                    raise ValueError("invalid remembered observation")
                integer(row.get("tick"), "remembered tick", 0, self.tick)
                number(row.get("reward"), "remembered reward")
            for peer, row in memory["social"].items():
                if not isinstance(peer, str) or not peer.isdecimal() or str(int(peer)) != peer or not isinstance(row, dict):
                    raise ValueError("invalid social memory")
                integer(int(peer), "remembered peer", 1, self.next_agent_id - 1)
                integer(row.get("last_tick"), "social memory tick", 0, self.tick)
                integer(row.get("interactions"), "social interaction count")
                if row.get("last_kind") not in SOCIAL_KINDS or row.get("last_direction") not in ("incoming", "outgoing"):
                    raise ValueError("invalid social interaction memory")
            if "last_demo" in memory:
                demo = memory["last_demo"]
                if not isinstance(demo, dict) or demo.get("action") not in ACTIONS or not isinstance(demo.get("observation"), dict) \
                        or not isinstance(demo.get("next_observation"), dict):
                    raise ValueError("invalid demonstration memory")
                if set(demo["observation"]) - set(DEMO_FEATURES) or set(demo["next_observation"]) - set(DEMO_FEATURES):
                    raise ValueError("demonstration carries private demonstrator state")
                integer(demo.get("tick"), "demonstration tick", 0, self.tick)
                number(demo.get("outcome"), "demonstration outcome")
        for identity, obj in self.objects.items():
            integer(identity, "object id", 1, self.next_object_id - 1)
            if identity != obj.id:
                raise ValueError("invalid object identity")
            if obj.maker_id is not None:
                integer(obj.maker_id, "object maker", 1, self.next_agent_id - 1)
            for component in obj.components:
                integer(component, "historical material component", 1, obj.id - 1)
            if obj.mass <= 0 or obj.holder is not None and obj.id not in owned or obj.temperature < 0:
                raise ValueError("invalid material object")
            materials.validate_item(obj)
        for identity, food in self.foods.items():
            integer(identity, "food id", 1)
            if identity != food.id:
                raise ValueError("invalid resource identity")
            if not 0 <= food.amount <= food.capacity + 1e-9 or food.hardness < 0 or food.opening < 0:
                raise ValueError("invalid resource patch")
        for records in (self.agents, self.objects, self.foods):
            for obj in records.values():
                if len(obj.position) != 2 or not 0 <= obj.position[0] < self.config.width or not 0 <= obj.position[1] < self.config.height:
                    raise ValueError("entity outside toroidal world bounds")
        if self.next_agent_id <= max(self.agents, default=0) or self.next_object_id <= max(self.objects, default=0):
            raise ValueError("future identity counter is not monotonic")
        for name in ("births", "deaths", "signals_sent", "combinations", "dismantlings"):
            integer(self.counters.get(name, 0), name + " counter")
        if self.next_agent_id != self.config.population + self.counters.get("births", 0) + 1:
            raise ValueError("organism identity counter differs from birth history")
        if len(self.agents) != self.config.population + self.counters.get("births", 0) - self.counters.get("deaths", 0):
            raise ValueError("population differs from birth/death history")
        if self.next_object_id != self.config.initial_objects + self.counters.get("combinations", 0) + 2 * self.counters.get("dismantlings", 0) + 1:
            raise ValueError("material identity counter differs from fabrication history")
        if self.next_message_id != self.counters.get("signals_sent", 0) + 1:
            raise ValueError("message identity counter differs from signal history")
        if not isinstance(self.messages, list):
            raise ValueError("invalid signal queue")
        if not isinstance(self.pending_heard, list) or (self.pending_heard and self.config.log != "compact"):
            raise ValueError("invalid pending heard-message log")
        for row in self.pending_heard:
            if not isinstance(row, dict): raise ValueError("invalid pending heard-message record")
            integer(row.get("agent"), "pending receiver", 1, self.next_agent_id - 1)
            integer(row.get("tick"), "pending heard tick", 0, self.tick)
        signal_ids = set()
        for message in self.messages:
            if not isinstance(message, dict): raise ValueError("invalid signal record")
            integer(message.get("id"), "signal id", 1, self.next_message_id - 1)
            if message["id"] in signal_ids: raise ValueError("duplicate signal id")
            signal_ids.add(message["id"])
            integer(message.get("sender"), "signal sender", 1, self.next_agent_id - 1)
            integer(message.get("symbol"), "signal symbol", 0, 3)
            integer(message.get("created_tick"), "signal creation", 0, self.tick)
            integer(message.get("expires_tick"), "signal expiry", self.tick)
            if message["expires_tick"] != message["created_tick"] + self.config.message_ttl:
                raise ValueError("signal expiry differs from configured lifetime")
            number(message.get("range"), "signal range", 0)
            if not isinstance(message.get("delivered_to"), list) or len(set(message["delivered_to"])) != len(message["delivered_to"]):
                raise ValueError("invalid delivered signal recipient list")
            for recipient in message["delivered_to"]:
                integer(recipient, "signal recipient", 1, self.next_agent_id - 1)
            if not isinstance(message.get("position"), list) or len(message["position"]) != 2 \
                    or not 0 <= message["position"][0] < self.config.width or not 0 <= message["position"][1] < self.config.height:
                raise ValueError("invalid signal source position")
        energy_in = self.ledger["initial_agent_energy"] + self.ledger["energy_from_food"]
        energy_out = sum(a.energy for a in self.agents.values()) + sum(self.ledger[k] for k in
                       ("action_energy", "metabolic_energy", "birth_cost", "energy_wasted", "dead_energy"))
        food_in = self.ledger["initial_food"] + self.ledger["food_regrown"] + self.ledger.get("food_bloomed", 0.)
        food_out = sum(f.amount for f in self.foods.values()) + self.ledger["food_eaten"] + self.ledger.get("food_spoiled", 0.)
        if (self.eco is not None) != self.config.ecology_on():
            raise ValueError("ecology record disagrees with the configuration")
        if self.eco is not None:
            self._validate_ecology()
        mass = sum(o.mass for o in self.objects.values())
        heat_in = self.ledger["initial_material_heat"] + self.ledger["heat_to_objects"]
        heat_out = sum(materials.thermal_energy(o) for o in self.objects.values()) + self.ledger["heat_to_environment"] + self.ledger["heat_to_body"]
        if not math.isclose(energy_in, energy_out, abs_tol=1e-7, rel_tol=1e-10):
            raise ValueError("agent energy ledger does not close")
        if not math.isclose(food_in, food_out, abs_tol=1e-7, rel_tol=1e-10):
            raise ValueError("food ledger does not close")
        if not math.isclose(mass, self.ledger["initial_material_mass"], abs_tol=1e-8, rel_tol=1e-10):
            raise ValueError("active material mass ledger does not close")
        if not math.isclose(heat_in, heat_out, abs_tol=1e-7, rel_tol=1e-10):
            raise ValueError("material thermal energy ledger does not close")
        return {"ok": True, "tick": self.tick, "population": len(self.agents),
                "energy_residual": energy_in - energy_out, "food_residual": food_in - food_out,
                "material_mass_residual": mass - self.ledger["initial_material_mass"],
                "material_heat_residual": heat_in - heat_out}
