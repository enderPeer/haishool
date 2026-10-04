"""life9: many 3D living worlds simulated together as tensors (CPU, CUDA or ROCm).

One ``World9`` holds ``cfg.worlds`` independent worlds. Every world has its own terrain,
food, predators and up to ``cfg.capacity`` individuals; all of them advance in one batched
step. The worlds are the paired replicates of an experiment: the same seed gives the same
terrains and founders in every arm, and a statistic is computed per world.

What the engine does not do (as in life8): no dictionary, no meaning for any call vector,
no reward for calling, listening, cooperating or anything named. Selection comes from
survival and reproduction; lifetime learning comes from the individual's own energy and
health change. ``oracle`` is a diagnostic switch for the listener gate, never a mechanism
of a real run.

Step order: sense (from the tick's start state, including last tick's calls), think, move
and pay, eat, call, predators, physiology, deaths, lifetime learning, births.
"""
from __future__ import annotations

import math

import torch

from . import brain, terrain
from .config import Config9

TWO_PI = 2 * math.pi
STATS = ("births", "deaths_starvation", "deaths_predation", "deaths_age", "strikes", "calls", "eaten",
         "alive_ticks", "capacity_full", "heard_moves", "heard_flee", "quiet_moves", "quiet_flee")
LEDGER = ("eaten", "move", "climb", "calls", "metabolism", "birth_cost", "dead", "overflow")


class World9:
    def __init__(self, cfg: Config9, seed: int, device="cpu"):
        self.cfg, self.seed, self.device = cfg, seed, torch.device(device)
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(seed)
        self.tick = 0
        W, N, dev = cfg.worlds, cfg.capacity, self.device
        self.terrain = terrain.make_terrain(self.gen, W, cfg.grid, cfg.relief, cfg.octaves, dev)
        # food prefers valleys: draw 4x candidates, keep the lowest
        candidates = self._rand(W, cfg.food_patches * 4, 2) * cfg.size
        low = self._ground(candidates).argsort(dim=1)[:, :cfg.food_patches]
        self.food_pos = candidates.gather(1, low[..., None].expand(-1, -1, 2))
        fertility = (1 - self._ground(self.food_pos) / cfg.relief).clamp(0, 1) ** 2
        self.food_rate = cfg.food_regrowth * (.25 + .75 * fertility)
        self.food = torch.full((W, cfg.food_patches), cfg.food_capacity / 2, device=dev)
        self.pred_pos = self._rand(W, cfg.predators, 2) * cfg.size
        self.pred_rest = torch.zeros(W, cfg.predators, dtype=torch.long, device=dev)
        self.pred_chase = torch.zeros(W, cfg.predators, dtype=torch.long, device=dev)
        self.alive = torch.zeros(W, N, dtype=torch.bool, device=dev)
        self.alive[:, :cfg.founders] = True
        self.pos = self._rand(W, N, 2) * cfg.size
        self.heading = self._rand(W, N) * TWO_PI
        self.energy = torch.where(self.alive, cfg.initial_energy, 0.).to(dev)
        self.health = self.alive.float()
        self.age = torch.zeros(W, N, dtype=torch.long, device=dev)
        self.cooldown = torch.zeros(W, N, dtype=torch.long, device=dev)
        self.uid = torch.arange(N, device=dev).repeat(W, 1).where(self.alive, -1)
        self.next_uid = torch.full((W,), cfg.founders, dtype=torch.long, device=dev)
        self.parent = torch.full((W, N), -1, dtype=torch.long, device=dev)
        self.founder = self.uid.clone()
        self.generation = torch.zeros(W, N, dtype=torch.long, device=dev)
        self.genome = brain.random_genome(self.gen, (W, N), cfg, dev)
        self.wh_live = self.genome["Wh"].clone()
        self.state = torch.zeros(W, N, cfg.hidden, device=dev)
        self.calls = torch.zeros(W, N, cfg.vocal_dims, device=dev)   # loudness x vector, heard next tick
        self.baseline = torch.zeros(W, N, device=dev)
        self.stats = {name: torch.zeros(W, dtype=torch.float64, device=dev) for name in STATS}
        self.ledger = {name: torch.zeros(W, dtype=torch.float64, device=dev) for name in LEDGER}
        self.energy_start = self.energy.double().sum(1)

    # ------------------------------------------------------------------ helpers
    def _rand(self, *shape):
        return torch.rand(*shape, generator=self.gen, device=self.device)

    def _ground(self, points):
        return terrain.height_at(self.terrain, points, self.cfg.size)

    def _wrap(self, points):
        return torch.remainder(points, self.cfg.size)

    def _pairs(self, a, b):
        """Wrapped offsets from every a [W,N,2] to every b [W,M,2] -> ([W,N,M,2], [W,N,M])."""
        half = self.cfg.size / 2
        delta = torch.remainder(b[:, None, :, :] - a[:, :, None, :] + half, self.cfg.size) - half
        return delta, delta.norm(dim=-1)

    def _sectors(self, delta):
        angle = torch.atan2(delta[..., 1], delta[..., 0]) - self.heading[:, :, None]
        index = (torch.remainder(angle, TWO_PI) / TWO_PI * self.cfg.sectors).long().clamp(max=self.cfg.sectors - 1)
        return torch.nn.functional.one_hot(index, self.cfg.sectors).float()

    def _los(self, a_xy, a_z, delta, b_z):
        return terrain.line_of_sight(self.terrain, self.cfg.size, a_xy, a_z, delta, b_z, self.cfg.los_samples)

    # ------------------------------------------------------------------ senses
    def sense(self):
        cfg, W, N = self.cfg, self.cfg.worlds, self.cfg.capacity
        ground = self._ground(self.pos)
        z = ground + cfg.eye_height
        sight = cfg.sight * (1 + cfg.height_sight_gain * ground / cfg.relief)
        eye = torch.eye(N, dtype=torch.bool, device=self.device)
        # others: sight and sound share one line-of-sight test
        d_a, dist_a = self._pairs(self.pos, self.pos)
        other = self.alive[:, None, :] & self.alive[:, :, None] & ~eye
        clear_a = self._los(self.pos, z, d_a, z[:, None, :])
        sec_a = self._sectors(d_a)
        seen_a = other & clear_a & (dist_a < sight[..., None])
        agents_in = (sec_a * ((1 - dist_a / sight[..., None]).clamp(min=0) * seen_a)[..., None]).amax(dim=2)
        loudness = (1 - dist_a / cfg.call_range).clamp(min=0) * torch.where(clear_a, 1., cfg.sound_occlusion) * other
        # what reached the ears, whether or not the channel lets it be heard (the deaf arm's matched control)
        audible = torch.einsum("wnm,wm->wn", loudness, self.calls.norm(dim=-1))
        if cfg.channel == "deaf":
            loudness = torch.zeros_like(loudness)
        sound = torch.einsum("wnms,wnm,wmd->wnsd", sec_a, loudness, self.calls)
        # food
        d_f, dist_f = self._pairs(self.pos, self.food_pos)
        food_z = self._ground(self.food_pos) + .2
        clear_f = self._los(self.pos, z, d_f, food_z[:, None, :])
        seen_f = clear_f & (dist_f < sight[..., None]) & (self.food > .5)[:, None, :]
        food_in = (self._sectors(d_f) * ((1 - dist_f / sight[..., None]).clamp(min=0) * seen_f)[..., None]).amax(dim=2)
        # predators (stealthy: seen only within predator_detect x sight)
        d_p, dist_p = self._pairs(self.pos, self.pred_pos)
        pred_z = self._ground(self.pred_pos) + .8
        clear_p = self._los(self.pos, z, d_p, pred_z[:, None, :])
        detect = sight * cfg.predator_detect
        seen_p = clear_p & (dist_p < detect[..., None]) & self.alive[..., None]
        if cfg.predators:
            preds_in = (self._sectors(d_p) * ((1 - dist_p / detect[..., None]).clamp(min=0) * seen_p)[..., None]).amax(dim=2)
        else:
            preds_in = torch.zeros_like(food_in)
        ahead = self.pos + torch.stack((self.heading.cos(), self.heading.sin()), -1)
        scalars = torch.stack((self.energy / cfg.max_energy, self.health, (self._ground(ahead) - ground) / cfg.relief,
                               ground / cfg.relief, torch.ones_like(ground), self.age.float() / cfg.lifespan), -1)
        x = torch.cat((agents_in, food_in, preds_in, sound.reshape(W, N, -1), scalars), -1) * self.alive[..., None]
        return {"x": x, "dist_f": dist_f, "dist_p": dist_p, "d_p": d_p, "clear_p": clear_p, "ground": ground,
                "sees_predator": seen_p.any(-1), "heard": sound.norm(dim=(-1, -2)), "audible": audible}

    # ------------------------------------------------------------------ one tick
    def step(self):
        cfg, W, N = self.cfg, self.cfg.worlds, self.cfg.capacity
        alive0 = self.alive.clone()
        energy0, health0 = self.energy.clone(), self.health.clone()
        seen = self.sense()
        out, new_state = brain.think(self.genome, self.wh_live, self.state, seen["x"], cfg.hidden)
        before_state, self.state = self.state, new_state * alive0[..., None]
        turn, speed_out, eat_out, loud_out = out[..., 0], out[..., 1], out[..., 2], out[..., 3]
        vocal = torch.tanh(out[..., 4:])
        live = alive0.float()

        # move and pay (slow when hurt; climbing costs; descending is free)
        self.heading = torch.remainder(self.heading + torch.tanh(turn) * cfg.max_turn, TWO_PI)
        speed = torch.sigmoid(speed_out) * self.genome["speed"] * cfg.max_speed * (.5 + .5 * self.health) * live
        step_xy = torch.stack((self.heading.cos(), self.heading.sin()), -1) * speed[..., None]
        near_pred_before = seen["dist_p"].amin(-1) if cfg.predators else None
        self.pos = self._wrap(self.pos + step_xy)
        climb = (self._ground(self.pos) - seen["ground"]).clamp(min=0) * live
        self._pay(cfg.move_cost * speed ** 2, "move")
        self._pay(cfg.climb_cost * climb, "climb")

        # eat from the nearest food patch in reach (shared fairly when patches run short)
        reach = torch.where((seen["dist_f"] <= cfg.reach) & (self.food > 0)[:, None, :], seen["dist_f"], math.inf)
        nearest = reach.argmin(-1)
        wants = (eat_out > 0) & alive0 & torch.isfinite(reach.amin(-1))
        demand = torch.zeros_like(self.food).scatter_add_(1, nearest, cfg.bite * wants.float())
        share = torch.where(demand > 0, (self.food / demand.clamp(min=1e-9)).clamp(max=1), 0.)
        got = cfg.bite * wants.float() * share.gather(1, nearest)
        self.food = (self.food - torch.zeros_like(self.food).scatter_add_(1, nearest, got)).clamp(min=0)
        self.energy = self.energy + got
        overflow = (self.energy - cfg.max_energy).clamp(min=0)
        self.energy = self.energy - overflow
        self._book("eaten", got)
        self._book("overflow", overflow)
        self.stats["eaten"] += got.double().sum(1)

        # call: continuous vector x loudness, heard next tick; silence is free
        loud = torch.sigmoid(loud_out) * self.genome["voice"] * live
        if cfg.oracle:   # diagnostic: the caller's own choice is replaced by a fixed meaning
            loud = torch.where(seen["sees_predator"], self.genome["voice"].clamp(max=1), 0.) * live
            vocal = torch.zeros_like(vocal)
            vocal[..., 0] = 1.
        loud = torch.where(loud >= cfg.silence, loud, 0.)
        self._pay(cfg.call_cost * loud, "calls")
        if cfg.channel == "masked":     # presence and loudness kept, content removed
            vocal = torch.full_like(vocal, cfg.vocal_dims ** -.5)
        elif cfg.channel == "scrambled":
            noise = torch.randn(vocal.shape, generator=self.gen, device=self.device)
            vocal = noise / noise.norm(dim=-1, keepdim=True).clamp(min=1e-9) * vocal.norm(dim=-1, keepdim=True)
        self.calls = vocal * loud[..., None]
        self.stats["calls"] += (loud > 0).double().sum(1)

        # predators move and strike; a listener measure: do hearers who cannot see a predator move away from it?
        if cfg.predators:
            self._predators()
            near_pred_after = self._pairs(self.pos, self.pred_pos)[1].amin(-1)
            blind = alive0 & ~seen["sees_predator"] & (near_pred_before < cfg.call_range)
            fled = (near_pred_after > near_pred_before + 1e-6).double()
            heard = blind & (seen["audible"] > .05)    # a call reached them (heard unless the arm is deaf)
            quiet = blind & (seen["audible"] <= .05)
            self.stats["heard_moves"] += heard.double().sum(1)
            self.stats["heard_flee"] += (fled * heard).sum(1)
            self.stats["quiet_moves"] += quiet.double().sum(1)
            self.stats["quiet_flee"] += (fled * quiet).sum(1)

        # physiology
        brain_units = self.genome["k"].float()
        self._pay((cfg.base_metabolism + cfg.neuron_cost * brain_units) * live, "metabolism")
        heal = ((self.energy > cfg.max_energy / 2) & alive0).float() * .002
        self.health = torch.minimum(self.health + heal, torch.ones_like(self.health)) * live
        self.food = torch.minimum(self.food + self.food_rate, torch.full_like(self.food, cfg.food_capacity))
        self.age = self.age + alive0.long()
        self.cooldown = (self.cooldown - 1).clamp(min=0)
        self.stats["alive_ticks"] += live.double().sum(1)

        # deaths
        starved = alive0 & (self.energy <= 0)
        killed = alive0 & ~starved & (self.health <= 0)
        old = alive0 & ~starved & ~killed & (self.age >= cfg.lifespan)
        dead = starved | killed | old
        self.stats["deaths_starvation"] += starved.double().sum(1)
        self.stats["deaths_predation"] += killed.double().sum(1)
        self.stats["deaths_age"] += old.double().sum(1)
        self._book("dead", self.energy * dead)
        self.energy = torch.where(dead, 0., self.energy)
        self.alive = alive0 & ~dead

        # lifetime learning from the individual's own outcome (energy + 8 x health change)
        reward = (self.energy - energy0) + 8 * (self.health - health0)
        if cfg.plasticity:
            modulator = torch.tanh(reward - self.baseline)
            brain.hebbian(self.wh_live, self.genome["eta"], modulator, before_state, self.state,
                          self.alive, cfg.weight_clip)
        self.baseline = torch.where(self.alive, self.baseline + .05 * (reward - self.baseline), 0.)

        self._births()
        self.tick += 1

    def _pay(self, cost, ledger):
        self.energy = self.energy - cost
        self._book(ledger, cost)

    def _book(self, ledger, amount):
        self.ledger[ledger] += amount.double().sum(1)

    def _predators(self):
        cfg = self.cfg
        d_pn, dist = self._pairs(self.pred_pos, self.pos)                       # [W,P,N]
        pred_z = self._ground(self.pred_pos) + .8
        prey_z = self._ground(self.pos) + cfg.eye_height
        clear = self._los(self.pred_pos, pred_z, d_pn, prey_z[:, None, :])
        visible = clear & (dist < cfg.predator_sight) & self.alive[:, None, :]
        masked = torch.where(visible, dist, math.inf)
        best, target = masked.min(-1)
        ready = self.pred_rest == 0
        hunting = ready & torch.isfinite(best)
        vector = d_pn.gather(2, target[..., None, None].expand(-1, -1, 1, 2)).squeeze(2)
        step = torch.minimum(best.nan_to_num(posinf=0.), torch.full_like(best, cfg.predator_speed))
        direction = vector / vector.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        wander = (self._rand(*self.pred_pos.shape) - .5) * .6
        move = torch.where(hunting[..., None], direction * step[..., None], wander)
        self.pred_pos = self._wrap(self.pred_pos + move)
        strike = hunting & (best - step <= cfg.predator_strike)
        damage = torch.zeros_like(self.health).scatter_add_(1, target, strike.float() * cfg.predator_damage)
        self.health = (self.health - damage).clamp(min=0)
        self.stats["strikes"] += strike.double().sum(1)
        self.pred_chase = torch.where(hunting, self.pred_chase + 1, 0)
        tired = self.pred_chase >= cfg.predator_stamina
        self.pred_rest = torch.where(strike | tired, cfg.predator_rest, (self.pred_rest - 1).clamp(min=0))
        self.pred_chase = torch.where(strike | tired, 0, self.pred_chase)

    def _births(self):
        cfg, W, N = self.cfg, self.cfg.worlds, self.cfg.capacity
        ready = self.alive & (self.energy > cfg.reproduction_threshold) & (self.age >= cfg.maturity) & (self.cooldown == 0)
        free = ~self.alive
        count = torch.minimum(ready.sum(1), free.sum(1))
        self.stats["capacity_full"] += (ready.sum(1) > free.sum(1)).double()
        if int(count.sum()) == 0:
            return
        parents = torch.argsort((~ready).to(torch.int8), dim=1, stable=True)
        slots = torch.argsort((~free).to(torch.int8), dim=1, stable=True)
        rank = torch.arange(N, device=self.device).expand(W, N)
        valid = rank < count[:, None]
        w = torch.arange(W, device=self.device)[:, None].expand(W, N)[valid]
        p, c, j = parents[valid], slots[valid], rank[valid]
        self._pay_at(w, p, cfg.offspring_energy + cfg.birth_cost)
        self.ledger["birth_cost"].index_add_(0, w, torch.full(w.shape, cfg.birth_cost, dtype=torch.float64, device=self.device))
        self.cooldown[w, p] = cfg.birth_cooldown
        self.alive[w, c] = True
        self.energy[w, c] = cfg.offspring_energy
        self.health[w, c] = 1.
        self.age[w, c] = 0
        self.cooldown[w, c] = 0
        self.pos[w, c] = self._wrap(self.pos[w, p] + torch.randn(len(w), 2, generator=self.gen, device=self.device) * .5)
        self.heading[w, c] = torch.rand(len(w), generator=self.gen, device=self.device) * TWO_PI
        child = brain.mutate(self.gen, {name: value[w, p] for name, value in self.genome.items()}, cfg)
        for name, value in child.items():
            self.genome[name][w, c] = value
        self.wh_live[w, c] = child["Wh"]
        self.state[w, c] = 0.
        self.calls[w, c] = 0.
        self.baseline[w, c] = 0.
        self.uid[w, c] = self.next_uid[w] + j
        self.next_uid += count
        self.parent[w, c] = self.uid[w, p]
        self.founder[w, c] = self.founder[w, p]
        self.generation[w, c] = self.generation[w, p] + 1
        self.stats["births"].index_add_(0, w, torch.ones(len(w), dtype=torch.float64, device=self.device))

    def _pay_at(self, w, p, amount):
        self.energy[w, p] -= amount   # the offspring energy moves to the child; the birth cost leaves

    # ------------------------------------------------------------------ records
    def ledger_error(self):
        """Per world: energy now minus (start + eaten - every cost - energy of the dead - overflow)."""
        L = self.ledger
        expected = self.energy_start + L["eaten"] - L["move"] - L["climb"] - L["calls"] - L["metabolism"] \
            - L["birth_cost"] - L["dead"] - L["overflow"]
        return self.energy.double().sum(1) - expected

    def summary(self):
        alive = self.alive.sum(1)
        mean = lambda t: ((t.float() * self.alive).sum(1) / alive.clamp(min=1)).tolist()
        stats = {name: value.tolist() for name, value in self.stats.items()}
        return {"tick": self.tick, "alive": alive.tolist(), "mean_hidden": mean(self.genome["k"]),
                "mean_voice": mean(self.genome["voice"]), "mean_eta": mean(self.genome["eta"]),
                "max_generation": self.generation.where(self.alive, 0).amax(1).tolist(), "stats": stats}

    def state_dict(self):
        names = ("terrain", "food_pos", "food_rate", "food", "pred_pos", "pred_rest", "pred_chase", "alive", "pos",
                 "heading", "energy", "health", "age", "cooldown", "uid", "next_uid", "parent", "founder",
                 "generation", "wh_live", "state", "calls", "baseline", "energy_start")
        return {"schema": "life9-state-v1", "config": self.cfg.to_dict(), "seed": self.seed, "tick": self.tick,
                "rng": self.gen.get_state(), "tensors": {n: getattr(self, n) for n in names},
                "genome": self.genome, "stats": self.stats, "ledger": self.ledger}

    @classmethod
    def from_state(cls, data, device="cpu"):
        if data.get("schema") != "life9-state-v1":
            raise ValueError("not a life9-state-v1 checkpoint")
        world = cls(Config9(**data["config"]), data["seed"], device)
        world.tick = data["tick"]
        world.gen.set_state(data["rng"])
        for name, value in data["tensors"].items():
            setattr(world, name, value.to(world.device))
        world.genome = {k: v.to(world.device) for k, v in data["genome"].items()}
        world.stats = {k: v.to(world.device) for k, v in data["stats"].items()}
        world.ledger = {k: v.to(world.device) for k, v in data["ledger"].items()}
        return world

    def frame(self, w=0):
        """A compact, rounded record of world ``w`` for the 3D viewer."""
        live = self.alive[w].nonzero().squeeze(-1)
        pos, ground = self.pos[w, live], self._ground(self.pos[w:w + 1, live]).squeeze(0)
        calls = self.calls[w, live]
        rows = torch.stack((self.uid[w, live].float(), pos[:, 0], pos[:, 1], ground, self.heading[w, live],
                            self.energy[w, live], self.founder[w, live].float(), calls.norm(dim=-1),
                            calls.argmax(-1).float(), self.genome["k"][w, live].float()), -1)
        preds = torch.cat((self.pred_pos[w], self._ground(self.pred_pos[w:w + 1]).squeeze(0)[:, None]), -1)
        food = torch.cat((self.food_pos[w], self._ground(self.food_pos[w:w + 1]).squeeze(0)[:, None],
                          self.food[w][:, None]), -1)
        r = lambda t: [[round(v, 2) for v in row] for row in t.tolist()]
        return {"tick": self.tick, "agents": r(rows), "predators": r(preds), "food": r(food)}
