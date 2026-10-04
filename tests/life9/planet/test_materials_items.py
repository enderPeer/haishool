"""life9 planet item and fire pools: vectorised spawn/remove, full pools, ledgers, state and determinism."""
import pytest
import torch

from haishool.life9.planet import items as it
from haishool.life9.planet import materials as m

S = m.S


def _req(w, species, mass, holder=None):
    w = torch.tensor(w, dtype=torch.long)
    comp = torch.stack([m.species_vector({s: 1.0}) for s in species])
    mass = torch.tensor(mass, dtype=torch.float32)
    pos = torch.nn.functional.normalize(torch.arange(len(w) * 3, dtype=torch.float32).reshape(-1, 3) + 1, dim=-1)
    holder = torch.full_like(w, -1) if holder is None else torch.tensor(holder, dtype=torch.long)
    return w, comp, mass, pos, torch.full((len(w),), 288.0), holder


def test_new_pools():
    p = it.new_item_pool(2, 5, device="cpu")
    assert p.shape == (2, 5, S) and p.pos.shape == (2, 5, 3) and p.holder.dtype == torch.long
    assert not p.alive.any() and bool((p.holder == -1).all()) and p.comp.dtype == torch.float32
    f = it.new_fire_pool(2, 3)
    assert f.shape == (2, 3, S) and not f.alive.any()


def test_spawn_fills_lowest_free_slots_in_request_order():
    p = it.new_item_pool(3, 4)
    idx = it.spawn(p, *_req([1, 0, 1, 1, 2], ["flint", "wood", "clay", "meat", "copper"], [0.5, 2.0, 1.0, 3.0, 0.2],
                            holder=[-1, 7, -1, -1, 3]))
    assert idx.tolist() == [0, 0, 1, 2, 0]
    assert p.alive.sum().item() == 5 and it.count(p).tolist() == [1, 3, 1]
    assert p.comp[1, 1, m.IDX["clay"]] == 1 and p.mass[1, 2] == 3.0 and p.holder[0, 0] == 7
    assert p.peak_k[1, 0] == 288.0 and p.age_d[1, 0] == 0 and p.sharp[1, 0] == 0
    # comp is renormalised to mass fractions
    w, comp, mass, pos, temp, holder = _req([2], ["sand"], [1.0])
    i2 = it.spawn(p, w, comp * 3.0, mass, pos, temp, holder, sharp=torch.tensor([0.4]), bond=0.5)
    assert i2.tolist() == [1] and p.comp[2, 1].sum().item() == pytest.approx(1.0)
    assert p.sharp[2, 1].item() == pytest.approx(0.4) and p.bond[2, 1].item() == pytest.approx(0.5)


def test_full_pool_never_overwrites():
    p = it.new_item_pool(2, 3)
    first = it.spawn(p, *_req([0, 0, 0], ["flint"] * 3, [1.0, 2.0, 3.0]))
    assert first.tolist() == [0, 1, 2]
    before = it.state_hash(p)
    full = it.spawn(p, *_req([0, 0], ["wood", "wood"], [9.0, 9.0]))
    assert full.tolist() == [-1, -1] and it.state_hash(p) == before
    mixed = it.spawn(p, *_req([0, 1, 1, 1, 1], ["wood"] * 5, [9.0, 1.0, 1.0, 1.0, 1.0]))
    assert mixed.tolist() == [-1, 0, 1, 2, -1]
    assert p.mass[0].tolist() == [1.0, 2.0, 3.0]


def test_remove_and_reuse_and_ledger():
    p = it.new_item_pool(2, 4)
    idx = it.spawn(p, *_req([0, 0, 0, 1], ["flint", "wood", "malachite", "bronze"], [1.0, 2.0, 4.0, 8.0]))
    total = it.species_mass(p)
    assert total.dtype == torch.float64 and total.shape == (2, S)
    assert total[0, m.IDX["wood"]].item() == pytest.approx(2.0) and total[1, m.IDX["bronze"]].item() == pytest.approx(8.0)
    el = it.element_mass(p)
    assert el.shape == (2, len(m.ELEMENTS)) and el.sum().item() == pytest.approx(15.0)
    assert el[1, m.ELEMENTS.index("Sn")].item() == pytest.approx(0.88)
    # remove the wood twice in one call, an empty slot and a -1: the wood is booked once
    gone = it.remove(p, torch.tensor([0, 0, 0, 1]), torch.tensor([1, 1, 3, -1]))
    assert gone.dtype == torch.float64 and gone[:, m.IDX["wood"]].tolist() == [2.0, 0.0, 0.0, 0.0]
    assert gone.sum().item() == pytest.approx(2.0)
    after = it.species_mass(p)
    assert torch.allclose(after + gone.sum(0, keepdim=True) * torch.tensor([[1.0], [0.0]], dtype=torch.float64), total)
    assert not p.alive[0, 1] and p.holder[0, 1] == -1 and p.mass[0, 1] == 0
    again = it.spawn(p, *_req([0, 0], ["clay", "clay"], [1.0, 1.0]))
    assert again.tolist() == [1, 3]               # the freed slot first, then the next free one


def test_item_props_and_holder_codes():
    p = it.new_item_pool(1, 3)
    it.spawn(p, *_req([0, 0], ["flint", "charcoal"], [0.5, 2.0], holder=[it.holder_code(torch.tensor(4), 1, 3).item(), -1]))
    pr = it.item_props(p)
    assert pr["hardness"].shape == (1, 3)
    assert pr["hardness"][0, 0].item() == pytest.approx(7.0) and pr["fuel_j"][0, 2].item() == 0
    assert pr["fuel_j"][0, 1].item() == pytest.approx(2.0 * float(m.COMBUSTION_J_KG["charcoal"]), rel=1e-6)
    n, k = it.holder_split(p.holder, 3)
    assert n[0].tolist() == [4, -1, -1] and k[0].tolist() == [1, -1, -1]


def test_fire_pool():
    f = it.new_fire_pool(2, 2)
    fuel = torch.stack([m.species_vector({"wood": 3.0}), m.species_vector({"charcoal": 1.0, "wood": 1.0}),
                        m.species_vector({"fat": 0.2})])
    pos = torch.tensor([[0.0, 0.0, 1.0]] * 3)
    idx = it.spawn_fire(f, torch.tensor([1, 1, 1]), pos, fuel, torch.tensor([900.0, 1100.0, 800.0]),
                        air=torch.tensor([0.0, 1.0, 0.0]))
    assert idx.tolist() == [0, 1, -1] and f.air[1, 1] == 1.0 and f.temp_k[1, 0] == 900.0
    ledger = it.fire_species_mass(f)
    assert ledger.dtype == torch.float64 and ledger[1, m.IDX["wood"]].item() == pytest.approx(4.0)
    out = it.remove_fire(f, torch.tensor([1, 1]), torch.tensor([0, 0]))
    assert out[:, m.IDX["wood"]].tolist() == [3.0, 0.0] and not f.alive[1, 0]
    assert it.spawn_fire(f, torch.tensor([1]), pos[:1], fuel[2:], 700.0).tolist() == [0]


def test_state_roundtrip():
    p = it.new_item_pool(2, 4)
    it.spawn(p, *_req([0, 1], ["flint", "hide"], [1.0, 2.0]))
    q = it.from_state(it.state_dict(p))
    assert isinstance(q, it.ItemPool) and it.state_hash(q) == it.state_hash(p)
    q.mass[0, 0] = 5.0
    assert p.mass[0, 0] == 1.0                     # the state is a copy
    f = it.new_fire_pool(1, 2)
    it.spawn_fire(f, torch.tensor([0]), torch.zeros(1, 3), m.species_vector({"wood": 1.0})[None], torch.tensor([900.0]))
    g = it.from_state(f.state_dict(), device="cpu")
    assert isinstance(g, it.FirePool) and it.state_hash(g) == it.state_hash(f)


def _random_run(seed: int, steps: int = 30) -> tuple[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    W, I = 3, 64
    p = it.new_item_pool(W, I)
    booked = torch.zeros(W, S, dtype=torch.float64)
    for _ in range(steps):
        M = int(torch.randint(0, 12, (1,), generator=gen))
        w = torch.randint(0, W, (M,), generator=gen)
        comp = torch.rand(M, S, generator=gen) ** 3
        mass = torch.rand(M, generator=gen) * 5
        pos = torch.nn.functional.normalize(torch.randn(M, 3, generator=gen), dim=-1)
        idx = it.spawn(p, w, comp, mass, pos, torch.full((M,), 288.0), torch.full((M,), -1))
        ok = idx >= 0
        booked.index_add_(0, w[ok], comp[ok].double() / comp[ok].double().sum(1, keepdim=True) * mass[ok].double()[:, None])
        R = int(torch.randint(0, 8, (1,), generator=gen))
        rw, ri = torch.randint(0, W, (R,), generator=gen), torch.randint(-1, I, (R,), generator=gen)
        gone = it.remove(p, rw, ri)
        booked.index_add_(0, rw, -gone)
    assert torch.allclose(it.species_mass(p), booked, atol=1e-4)      # the item ledger closes
    return it.state_hash(p), it.species_mass(p)


def test_determinism_and_ledger():
    h1, m1 = _random_run(3)
    h2, m2 = _random_run(3)
    h3, _ = _random_run(4)
    assert h1 == h2 and torch.equal(m1, m2) and h1 != h3


# ------------------------------------------------------------------ review fixes
def test_spawn_rejects_invalid_requests_and_keeps_the_ledger():
    p = it.new_item_pool(1, 8)
    nan = float("nan")
    comp = torch.stack([
        torch.zeros(S),                                   # sums to 0
        torch.full((S,), nan),                            # not finite
        -m.species_vector({"flint": 1.0}),                # only negative
        m.species_vector({"flint": 1.0}),                 # valid comp, negative mass
        m.species_vector({"flint": 1.0}),                 # NaN mass
        m.species_vector({"flint": 1.0}),                 # inf mass
        m.species_vector({"flint": 1.0}),                 # zero mass
        m.species_vector({"flint": 1.0}),                 # NaN temperature
        m.species_vector({"wood": 1.0}),                  # valid
    ])
    mass = torch.tensor([2.0, 2.0, 2.0, -1.0, nan, float("inf"), 0.0, 1.0, 3.0])
    temp = torch.tensor([288.0] * 7 + [nan, 288.0])
    w = torch.zeros(9, dtype=torch.long)
    idx = it.spawn(p, w, comp, mass, torch.tensor([0.0, 0.0, 1.0]), temp, -1)
    assert idx.tolist() == [-1] * 8 + [0]             # the invalid requests take no slot
    assert it.count(p).tolist() == [1]
    assert it.species_mass(p).sum().item() == pytest.approx(float(p.mass[p.alive].sum()))
    assert it.species_mass(p)[0, m.IDX["wood"]].item() == pytest.approx(3.0)
    assert bool(torch.isfinite(p.comp).all()) and bool((p.comp >= 0).all())


def test_spawn_clamps_negative_residues():
    p = it.new_item_pool(1, 2)
    comp = m.species_vector({"flint": 0.7, "wood": 0.3}).clone()
    comp[m.IDX["meat"]] = -1e-7                         # float32 residue left by crafting
    idx = it.spawn(p, torch.tensor([0]), comp[None], torch.tensor([2.0]), torch.tensor([[0.0, 0.0, 1.0]]), 288.0, -1)
    assert idx.tolist() == [0] and bool((p.comp >= 0).all())
    assert p.comp[0, 0].sum().item() == pytest.approx(1.0) and p.comp[0, 0, m.IDX["meat"]] == 0
    assert it.species_mass(p).sum().item() == pytest.approx(2.0, rel=1e-6)


def test_spawn_broadcasts_scalar_and_single_row_arguments():
    p = it.new_item_pool(2, 4)
    w = torch.tensor([0, 1, 1])
    flint = m.species_vector({"flint": 1.0})
    idx = it.spawn(p, w, flint, 0.5, torch.tensor([0.0, 0.0, 1.0]), 300.0, -1,
                   sharp=torch.tensor([0.4]), head_mass=torch.tensor(0.2), bond=[0.1, 0.2, 0.3])
    assert idx.tolist() == [0, 0, 1]
    assert p.holder[1, :2].tolist() == [-1, -1] and p.mass[1, 1].item() == pytest.approx(0.5)
    assert p.sharp[1, 1].item() == pytest.approx(0.4) and p.head_mass[0, 0].item() == pytest.approx(0.2)
    assert p.bond[1, 1].item() == pytest.approx(0.3) and p.temp_k[0, 0] == 300.0
    idx2 = it.spawn(p, torch.tensor([0, 0]), flint[None], torch.tensor([1.0]), torch.tensor([[1.0, 0.0, 0.0]]),
                    torch.tensor(250.0), torch.tensor(5))       # 0-dim holder and [1]-shaped mass
    assert idx2.tolist() == [1, 2] and p.holder[0, 1:3].tolist() == [5, 5] and p.peak_k[0, 2] == 250.0
    assert it.remove(p, 0, torch.tensor([1, 2]))[:, m.IDX["flint"]].tolist() == [1.0, 1.0]   # scalar world


def test_spawn_fire_rejects_invalid_fuel():
    f = it.new_fire_pool(1, 4)
    fuel = torch.stack([m.species_vector({"wood": 1.0}), -m.species_vector({"wood": 1.0}),
                        torch.full((S,), float("nan")), m.species_vector({"charcoal": 2.0})])
    idx = it.spawn_fire(f, torch.zeros(4, dtype=torch.long), torch.tensor([0.0, 0.0, 1.0]), fuel, 900.0)
    assert idx.tolist() == [0, -1, -1, 1]
    assert it.fire_species_mass(f).sum().item() == pytest.approx(3.0)
    assert it.spawn_fire(f, torch.tensor([0]), torch.zeros(3), fuel[:1], 900.0, air=-1.0).tolist() == [-1]


def test_state_hash_fast_layout_independent_and_typed():
    import time
    W, I = 16, 4096
    p = it.new_item_pool(W, I)
    gen = torch.Generator().manual_seed(5)
    p.comp.copy_(torch.rand(W, I, S, generator=gen))
    p.mass.copy_(torch.rand(W, I, generator=gen))
    p.alive.copy_(torch.rand(W, I, generator=gen) > 0.5)
    t0 = time.perf_counter()
    h = it.state_hash(p)
    assert time.perf_counter() - t0 < 1.0                  # was about 25 s with bytes(untyped_storage())
    assert it.state_hash(p) == h
    # the same values in a permuted (dense, non-row-major) layout hash the same
    q = it.from_state(it.state_dict(p))
    q.comp = q.comp.permute(2, 0, 1).contiguous().permute(1, 2, 0)
    assert not q.comp.is_contiguous() and torch.equal(q.comp, p.comp) and it.state_hash(q) == h
    # dtype and shape are part of the hash
    r = it.from_state(it.state_dict(p))
    r.mass = r.mass.double()
    assert it.state_hash(r) != h
    a, b = it.new_item_pool(1, 6), it.new_item_pool(2, 3)
    assert it.state_hash(a) != it.state_hash(b)
    q.mass[3, 7] += 1.0
    assert it.state_hash(q) != h


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_spawn_remove_props_on_cuda_match_cpu():
    def run(device):
        gen = torch.Generator().manual_seed(9)
        p = it.new_item_pool(3, 32, device=device)
        for _ in range(5):
            w = torch.randint(0, 3, (12,), generator=gen)
            comp = torch.rand(12, S, generator=gen) ** 3
            mass = torch.rand(12, generator=gen) * 4
            pos = torch.nn.functional.normalize(torch.randn(12, 3, generator=gen), dim=-1)
            it.spawn(p, w.to(device), comp.to(device), mass.to(device), pos.to(device), 288.0, -1)
            it.remove(p, torch.randint(0, 3, (4,), generator=gen).to(device),
                      torch.randint(-1, 32, (4,), generator=gen).to(device))
        return p
    cpu, gpu = run("cpu"), run("cuda")
    assert torch.equal(cpu.alive, gpu.alive.cpu()) and torch.equal(cpu.holder, gpu.holder.cpu())
    assert torch.allclose(cpu.comp, gpu.comp.cpu(), atol=1e-6) and torch.allclose(cpu.mass, gpu.mass.cpu())
    assert torch.allclose(it.species_mass(cpu), it.species_mass(gpu).cpu(), atol=1e-6)
    pc, pg = it.item_props(cpu), it.item_props(gpu)
    for k in pc:
        assert torch.allclose(pc[k], pg[k].cpu(), rtol=1e-5, atol=1e-5), k
