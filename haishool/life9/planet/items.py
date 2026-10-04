"""The item pool and the fire pool (PLANET-SPEC section 2.11): data structures only.

Items and fires live in fixed-capacity slots per world, ``[W, I]`` and ``[W, F]``. A slot is
used when ``alive`` is true. :func:`spawn` and :func:`spawn_fire` fill the lowest free slots of
each world in request order and return ``-1`` for a request whose world is full or whose data
is invalid (see each function): they never overwrite a live slot, and an invalid request takes
no slot, so the caller books exactly the requests with a slot >= 0. :func:`remove` and
:func:`remove_fire` free slots and return the species mass they held, so the caller can book it
in the item ledger. The crafting actions that create,
change and consume items live in ``crafting.py``.

An item's ``holder`` is the world-local inventory slot ``n * K + k`` of individual ``n``'s
``k``-th hand (K slots each), or ``-1`` when it lies on the ground at ``pos``.

State tensors are float32 (``holder`` int64, ``alive`` bool); ledger sums are float64. No RNG is
used here, so every operation is exactly reproducible on the CPU.
"""
from __future__ import annotations

import ctypes
import hashlib
from dataclasses import dataclass, fields

import torch

from haishool.life9.planet import materials

ITEM_FIELDS = ("alive", "comp", "mass", "temp_k", "sharp", "pos", "holder", "head_mass", "handle_len_m", "bond",
               "peak_k", "hot_days", "age_d")
FIRE_FIELDS = ("alive", "pos", "fuel_kg", "temp_k", "air")


@dataclass
class ItemPool:
    """Items of W worlds in I slots each. ``comp`` [W, I, S] holds mass fractions, ``mass`` kg,
    ``temp_k`` K, ``sharp`` in [0, 1], ``pos`` [W, I, 3] (unit-sphere position), ``holder`` the
    inventory code or -1; assembly fields ``head_mass`` (kg), ``handle_len_m``, ``bond`` in [0, 1];
    process fields ``peak_k`` (highest temperature reached), ``hot_days`` (days spent at or above
    a transform's temperature) and ``age_d``."""
    alive: torch.Tensor
    comp: torch.Tensor
    mass: torch.Tensor
    temp_k: torch.Tensor
    sharp: torch.Tensor
    pos: torch.Tensor
    holder: torch.Tensor
    head_mass: torch.Tensor
    handle_len_m: torch.Tensor
    bond: torch.Tensor
    peak_k: torch.Tensor
    hot_days: torch.Tensor
    age_d: torch.Tensor

    @property
    def shape(self) -> tuple[int, int, int]:
        """(W, I, S)."""
        return tuple(self.comp.shape)

    @property
    def device(self) -> torch.device:
        return self.alive.device

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: getattr(self, name).clone() for name in ITEM_FIELDS}

    @classmethod
    def from_state(cls, state: dict, device=None) -> "ItemPool":
        return cls(**{name: state[name].to(device).clone() if device is not None else state[name].clone()
                      for name in ITEM_FIELDS})


@dataclass
class FirePool:
    """Fires of W worlds in F slots each: ``pos`` [W, F, 3], ``fuel_kg`` [W, F, S] (kg of each
    species in the fire), ``temp_k`` K and ``air`` (the forced-air effort, 0 for an open fire)."""
    alive: torch.Tensor
    pos: torch.Tensor
    fuel_kg: torch.Tensor
    temp_k: torch.Tensor
    air: torch.Tensor

    @property
    def shape(self) -> tuple[int, int, int]:
        """(W, F, S)."""
        return tuple(self.fuel_kg.shape)

    @property
    def device(self) -> torch.device:
        return self.alive.device

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: getattr(self, name).clone() for name in FIRE_FIELDS}

    @classmethod
    def from_state(cls, state: dict, device=None) -> "FirePool":
        return cls(**{name: state[name].to(device).clone() if device is not None else state[name].clone()
                      for name in FIRE_FIELDS})


def new_item_pool(W: int, I: int, S: int | None = None, device="cpu") -> ItemPool:
    """An empty item pool."""
    S = materials.S if S is None else S
    f = lambda *shape: torch.zeros(*shape, dtype=torch.float32, device=device)
    return ItemPool(alive=torch.zeros(W, I, dtype=torch.bool, device=device), comp=f(W, I, S), mass=f(W, I),
                    temp_k=f(W, I), sharp=f(W, I), pos=f(W, I, 3),
                    holder=torch.full((W, I), -1, dtype=torch.long, device=device), head_mass=f(W, I),
                    handle_len_m=f(W, I), bond=f(W, I), peak_k=f(W, I), hot_days=f(W, I), age_d=f(W, I))


def new_fire_pool(W: int, F: int, S: int | None = None, device="cpu") -> FirePool:
    """An empty fire pool."""
    S = materials.S if S is None else S
    f = lambda *shape: torch.zeros(*shape, dtype=torch.float32, device=device)
    return FirePool(alive=torch.zeros(W, F, dtype=torch.bool, device=device), pos=f(W, F, 3), fuel_kg=f(W, F, S),
                    temp_k=f(W, F), air=f(W, F))


# ------------------------------------------------------------------------------ slot allocation
def allocate(alive: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Free slots for requests in worlds ``w`` [M]: the r-th request of a world (in request order)
    gets that world's r-th lowest free slot, or -1 when the world has fewer free slots. [M] long."""
    W, P = alive.shape
    w = w.long()
    if w.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=alive.device)
    free = ~alive
    ar = torch.arange(P, device=alive.device)
    order = torch.sort(torch.where(free, ar, ar + P), dim=1).indices      # free slots first, ascending
    onehot = torch.nn.functional.one_hot(w, W)                             # [M, W]
    rank = (onehot.cumsum(0) * onehot).sum(1) - 1                          # requests before this one in its world
    ok = rank < free.sum(1)[w]
    return torch.where(ok, order[w, rank.clamp(max=P - 1)], torch.full_like(rank, -1))


def _first_of(w: torch.Tensor, idx: torch.Tensor, P: int) -> torch.Tensor:
    """True for the first occurrence of each (world, slot) pair, so a slot is freed (and booked) once."""
    key = w * P + idx
    sk, perm = torch.sort(key, stable=True)
    first_sorted = torch.ones_like(sk, dtype=torch.bool)
    first_sorted[1:] = sk[1:] != sk[:-1]
    first = torch.empty_like(first_sorted)
    first[perm] = first_sorted
    return first


def _per_request(value, M: int, dtype, device, trailing: tuple = ()) -> torch.Tensor:
    """``value`` (a scalar, one row or M rows; None = 0) broadcast to [M, *trailing] on ``device``."""
    if value is None:
        return torch.zeros(M, *trailing, dtype=dtype, device=device)
    t = torch.as_tensor(value, device=device).to(dtype)
    if t.dim() == len(trailing):
        t = t.unsqueeze(0)
    return t.expand(M, *trailing)


def _finite_rows(t: torch.Tensor) -> torch.Tensor:
    """[M] True where every entry of row m of ``t`` [M, ...] is finite."""
    f = torch.isfinite(t)
    return f.all(-1) if f.dim() > 1 else f


def _index_pair(w, idx, device) -> tuple[torch.Tensor, torch.Tensor]:
    """(world, slot) requests as two broadcast [M] long tensors on ``device``."""
    w = torch.as_tensor(w, device=device).long().reshape(-1)
    idx = torch.as_tensor(idx, device=device).long().reshape(-1)
    M = max(int(w.shape[0]), int(idx.shape[0]))
    return w.expand(M), idx.expand(M)


# ------------------------------------------------------------------------------ items
def spawn(pool: ItemPool, w: torch.Tensor, comp: torch.Tensor, mass: torch.Tensor, pos: torch.Tensor,
          temp_k: torch.Tensor, holder: torch.Tensor, *, sharp=None, head_mass=None, handle_len_m=None,
          bond=None) -> torch.Tensor:
    """Create M items (vectorised); M = len(w). ``comp`` [M, S] is clamped to >= 0 and renormalised
    to mass fractions; ``holder`` is the inventory code or -1. Every per-request argument may be a
    scalar (or one row) for all requests or M values (rows). Returns the slot index [M]: -1 where
    the world is full (nothing is overwritten) or the request is invalid: ``comp`` not finite or
    without a positive entry, ``mass`` not finite or <= 0, ``temp_k`` not finite or < 0, ``pos`` or
    an assembly field not finite, or ``holder`` < -1. Invalid requests take no slot. ``peak_k``
    starts at ``temp_k``; ``hot_days`` and ``age_d`` at 0; the optional assembly fields default to 0."""
    W, I, S = pool.shape
    dev = pool.device
    w = torch.as_tensor(w, device=dev).long().reshape(-1)
    M = int(w.shape[0])
    comp = _per_request(comp, M, torch.float32, dev, (S,))
    mass = _per_request(mass, M, torch.float32, dev)
    pos = _per_request(pos, M, torch.float32, dev, (3,))
    temp = _per_request(temp_k, M, torch.float32, dev)
    hold = _per_request(holder, M, torch.long, dev)
    extra = {name: _per_request(value, M, torch.float32, dev)
             for name, value in (("sharp", sharp), ("head_mass", head_mass), ("handle_len_m", handle_len_m),
                                 ("bond", bond))}
    c = comp.clamp_min(0.0)
    total = c.sum(-1)
    valid = (_finite_rows(comp) & torch.isfinite(total) & (total > 0) & torch.isfinite(mass) & (mass > 0)
             & torch.isfinite(temp) & (temp >= 0) & _finite_rows(pos) & (hold >= -1))
    for value in extra.values():
        valid &= torch.isfinite(value)
    idx = torch.full((M,), -1, dtype=torch.long, device=dev)
    idx[valid] = allocate(pool.alive, w[valid])
    ok = idx >= 0
    wo, io = w[ok], idx[ok]
    pool.alive[wo, io] = True
    pool.comp[wo, io] = c[ok] / total[ok, None]
    pool.mass[wo, io] = mass[ok]
    pool.temp_k[wo, io] = temp[ok]
    pool.peak_k[wo, io] = temp[ok]
    pool.pos[wo, io] = pos[ok]
    pool.holder[wo, io] = hold[ok]
    for name, value in extra.items():
        getattr(pool, name)[wo, io] = value[ok]
    pool.hot_days[wo, io] = 0.0
    pool.age_d[wo, io] = 0.0
    return idx


def remove(pool: ItemPool, w: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Free item slots (``idx`` -1 or a dead slot is ignored; a repeated pair counts once). Returns
    the species mass each request removed, [M, S] float64, for the item ledger."""
    W, I, S = pool.shape
    w, idx = _index_pair(w, idx, pool.device)
    out = torch.zeros(w.shape[0], S, dtype=torch.float64, device=pool.device)
    ok = idx >= 0
    wo, io = w[ok], idx[ok]
    live = pool.alive[wo, io] & _first_of(wo, io, I)
    out[ok] = pool.comp[wo, io].double() * (pool.mass[wo, io].double() * live)[:, None]
    pool.alive[wo, io] = False
    pool.holder[wo, io] = -1
    for name in ("comp", "mass", "temp_k", "sharp", "head_mass", "handle_len_m", "bond", "peak_k", "hot_days", "age_d"):
        getattr(pool, name)[wo, io] = 0.0
    return out


def species_mass(pool: ItemPool) -> torch.Tensor:
    """[W, S] float64: the kg of each species in the live items of each world (item ledger)."""
    m = pool.mass.double() * pool.alive
    return (pool.comp.double() * m[..., None]).sum(1)


def element_mass(pool: ItemPool) -> torch.Tensor:
    """[W, E] float64: kg of each element (materials.ELEMENTS) in the live items of each world."""
    return materials.element_mass(species_mass(pool))


def count(pool) -> torch.Tensor:
    """[W] number of live slots (items or fires)."""
    return pool.alive.sum(1)


def item_props(pool: ItemPool) -> dict[str, torch.Tensor]:
    """materials.props of every slot, [W, I] each (dead slots have zero mass)."""
    return materials.props(pool.comp, pool.mass * pool.alive)


def holder_code(n: torch.Tensor, k: torch.Tensor | int, K: int) -> torch.Tensor:
    """Inventory code of individual ``n``'s slot ``k`` (K slots each)."""
    return n.long() * K + k


def holder_split(code: torch.Tensor, K: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(individual, slot) of an inventory code; both -1 where the code is -1."""
    code = code.long()
    held = code >= 0
    n = torch.where(held, torch.div(code, K, rounding_mode="floor"), torch.full_like(code, -1))
    k = torch.where(held, code % K, torch.full_like(code, -1))
    return n, k


# ------------------------------------------------------------------------------ fires
def spawn_fire(fires: FirePool, w: torch.Tensor, pos: torch.Tensor, fuel_kg: torch.Tensor, temp_k: torch.Tensor,
               air=None) -> torch.Tensor:
    """Create M fires (vectorised); ``fuel_kg`` [M, S] kg per species. Per-request arguments may be
    a scalar (or one row) or M values. Returns the slot [M], -1 where the world's fire pool is full
    (never overwrites) or the request is invalid (``fuel_kg`` not finite or negative anywhere,
    ``temp_k`` or ``air`` not finite or negative, ``pos`` not finite); invalid requests take no slot."""
    W, F, S = fires.shape
    dev = fires.device
    w = torch.as_tensor(w, device=dev).long().reshape(-1)
    M = int(w.shape[0])
    pos = _per_request(pos, M, torch.float32, dev, (3,))
    fuel = _per_request(fuel_kg, M, torch.float32, dev, (S,))
    temp = _per_request(temp_k, M, torch.float32, dev)
    air = _per_request(air, M, torch.float32, dev)
    valid = (_finite_rows(fuel) & (fuel >= 0).all(-1) & torch.isfinite(temp) & (temp >= 0) & torch.isfinite(air)
             & (air >= 0) & _finite_rows(pos))
    idx = torch.full((M,), -1, dtype=torch.long, device=dev)
    idx[valid] = allocate(fires.alive, w[valid])
    ok = idx >= 0
    wo, io = w[ok], idx[ok]
    fires.alive[wo, io] = True
    fires.pos[wo, io] = pos[ok]
    fires.fuel_kg[wo, io] = fuel[ok]
    fires.temp_k[wo, io] = temp[ok]
    fires.air[wo, io] = air[ok]
    return idx


def remove_fire(fires: FirePool, w: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Put out fires (``idx`` -1 or a dead slot is ignored; a repeated pair counts once). Returns
    the fuel each request held, [M, S] float64 kg, for the ledger."""
    W, F, S = fires.shape
    w, idx = _index_pair(w, idx, fires.device)
    out = torch.zeros(w.shape[0], S, dtype=torch.float64, device=fires.device)
    ok = idx >= 0
    wo, io = w[ok], idx[ok]
    live = fires.alive[wo, io] & _first_of(wo, io, F)
    out[ok] = fires.fuel_kg[wo, io].double() * live[:, None]
    fires.alive[wo, io] = False
    for name in ("pos", "fuel_kg", "temp_k", "air"):
        getattr(fires, name)[wo, io] = 0.0
    return out


def fire_species_mass(fires: FirePool) -> torch.Tensor:
    """[W, S] float64: kg of each species held as fuel in the live fires of each world."""
    return (fires.fuel_kg.double() * fires.alive[..., None]).sum(1)


# ------------------------------------------------------------------------------ state
def state_dict(pool) -> dict[str, torch.Tensor]:
    """The pool's tensors (cloned), for checkpoints."""
    return pool.state_dict()


def from_state(state: dict, device=None):
    """An ItemPool or FirePool from :func:`state_dict` output (the field names tell which)."""
    cls = ItemPool if "comp" in state else FirePool
    return cls.from_state(state, device)


def state_hash(pool) -> str:
    """SHA-256 over the pool's tensors for determinism checks: the pool type, then per field its
    name, dtype, shape and row-major bytes (``.contiguous()`` on the CPU, so the memory layout of
    the live tensors does not matter). A [16, 4096] item pool hashes in milliseconds."""
    h = hashlib.sha256(type(pool).__name__.encode())
    for f in fields(pool):
        t = getattr(pool, f.name).detach().cpu().contiguous()
        h.update(f"|{f.name}|{t.dtype}|{tuple(t.shape)}|".encode())
        if t.numel():
            h.update(ctypes.string_at(t.data_ptr(), t.numel() * t.element_size()))
    return h.hexdigest()
