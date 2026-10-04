"""The neighbour-based senses equal a brute-force reference on a small case (PLANET-V3-SPEC 4).

The reference loops in Python over every observer and every target in its 3 x 3 fine cells (all count) and in its
3 x 3 coarse cells beyond them (they count only when detected on their own), and over every fire, recomputes each
pair's physics in float64 and aggregates per sector by hand; ``senses3.observe`` does the same through the two spatial
hashes, the reach bounds, the streamed fires, packed line-of-sight tests and per-sector reductions. The reference does
not apply the reach bounds: matching it shows they never drop a detectable target. The organ helpers (eye optics, ear
threshold, call power and frequency, footfall power, ISO absorption) are tested against reference data in
test_senses3_physics.py and reused here."""
from __future__ import annotations

import math

import torch

from haishool.life9.planet import materials
from haishool.life9.v3 import brain3, optics
from haishool.life9.v3 import patch as pt
from haishool.life9.v3 import senses3 as s3

R_781 = 5.674e6
RULES = s3.SenseRules(K=64, max_scan=512, far_cells=2, far_k=64, fire_k=8, fire_rounds=1)


def make_bodies(A, N, L, gen, mass=(0.005, 0.5), alive_p=0.9, **over):
    """Random bodies: every field of s3.Bodies as a plain tensor."""
    lo, hi = mass
    M = lo * torch.exp(torch.rand(A, N, generator=gen) * math.log(hi / lo))
    d = dict(alive=torch.rand(A, N, generator=gen) < alive_p, pos=torch.rand(A, N, 2, generator=gen) * L,
             heading=torch.rand(A, N, generator=gen) * 2 * math.pi, mass_kg=M,
             speed_m_s=torch.rand(A, N, generator=gen) * 0.5, body_k=290.0 + 20 * torch.rand(A, N, generator=gen),
             damage=0.2 * torch.rand(A, N, generator=gen), reserve_j=1e4 * torch.rand(A, N, generator=gen),
             reserve_cap_j=torch.full((A, N), 1e4), water_kg=0.6 * M, water_norm_kg=0.65 * M, gut_kg=0.01 * M,
             gut_cap_kg=0.05 * M, eye=10 ** (-3 + 2 * torch.rand(A, N, generator=gen)),
             ear=10 ** (-3 + 2 * torch.rand(A, N, generator=gen)),
             voice=10 ** (-3 + 2 * torch.rand(A, N, generator=gen)), muscle=50 + 150 * torch.rand(A, N, generator=gen),
             loudness=torch.rand(A, N, generator=gen), calls=torch.rand(A, N, brain3.VOCAL_DIMS, generator=gen) * 2 - 1)
    d.update(over)
    return s3.Bodies(**d)


def _case(seed=21, A=2, N=40, I=30, n=8, L=64.0):
    gen = torch.Generator().manual_seed(seed)
    z = pt.fractal_detail(A, n, gen, 0.8) * 0.6
    geom = pt.geometry_from_elevation(z, L, -1e4, R_781)
    b = make_bodies(A, N, L, gen, mass=(0.01, 2.0), alive_p=0.85, held=None)
    b.speed_m_s = torch.rand(A, N, generator=gen) * 0.05
    comp = torch.zeros(A, I, materials.S)
    for k, sp in enumerate(("flint", "wood", "charcoal", "bone", "sand")):
        comp[:, k::5, materials.IDX[sp]] = 1.0
    temp = torch.full((A, I), 290.0)
    temp[:, ::7] = 1100.0                                                      # a few glowing items
    items = s3.Items(pos=torch.rand(A, I, 2, generator=gen) * L, alive=torch.rand(A, I, generator=gen) < 0.85,
                     mass=0.01 + torch.rand(A, I, generator=gen), temp_k=temp, sharp=torch.rand(A, I, generator=gen),
                     comp=comp, on_ground=torch.rand(A, I, generator=gen) < 0.9)
    # contacts: in each arena body 1 stands at body 0's mouth and item 0 lies at body 2's mouth
    rad = s3.body_radius(b.mass_kg)
    for a in range(A):
        b.alive[a, :3] = True
        h0, h2 = s3.heading_vector(b.heading[a, 0]), s3.heading_vector(b.heading[a, 2])
        b.pos[a, 1] = b.pos[a, 0] + h0 * (rad[a, 0] + 0.9 * rad[a, 1])
        items.alive[a, 0] = items.on_ground[a, 0] = True
        items.pos[a, 0] = b.pos[a, 2] + h2 * (1.3 * rad[a, 2])
    fires = s3.Fires(pos=torch.rand(A, 2, 2, generator=gen) * L, alive=torch.tensor([[True, False], [True, True]]),
                     temp_k=torch.tensor([[1000.0, 900.0], [1050.0, 950.0]]), area_m2=torch.full((A, 2), 0.05))
    light = 10 ** (-3 + 2 * torch.rand(A, n, n, generator=gen))               # dusk: some targets near threshold
    ground = 0.05 + 0.5 * torch.rand(A, n, n, generator=gen)
    return geom, z, b, items, fires, s3.Scene(light_vis=light, ground_vis=ground)


def _wrap(d, L):
    return (d + 0.5 * L) % L - 0.5 * L


def _cell(p, L, n):
    dx = L / n
    return min(int((p[0] % L) // dx), n - 1), min(int((p[1] % L) // dx), n - 1)


def _in_block(ci, cj, n):
    if n < 3:
        return True
    return ((cj[0] - ci[0]) % n in (0, 1, n - 1)) and ((cj[1] - ci[1]) % n in (0, 1, n - 1))


def _sector(h, dvec):
    w = 2 * math.pi / 8
    rel = h - math.atan2(dvec[1], dvec[0])
    return min(int(((rel + w / 2) % (2 * math.pi)) // w), 7)


def _los(geom1, p, qs, hp, hqs):
    if not qs:
        return []
    P = torch.tensor([[list(p)] * len(qs)], dtype=torch.float32)
    Q = torch.tensor([list(q) for q in qs], dtype=torch.float32)[None]
    hp = hp if isinstance(hp, list) else [hp] * len(qs)
    return pt.line_of_sight(geom1, P, Q, torch.tensor([hp]), torch.tensor([hqs]))[0].tolist()


def _omega(r, d):
    s = min(1.0, r / d)
    return 2 * math.pi * (1 - math.sqrt(max(0.0, 1 - s * s))) if s > 1e-3 else math.pi * s * s * (1 + s * s / 4)


def _rose(dn, nb):
    return dn / math.sqrt(max(nb + dn, 1e-30))


def _reference(geom, z, b, items, fires, scene, r=RULES):
    A, N = b.alive.shape
    n, L = geom.n, geom.L
    nc = max(1, n // r.far_cells)
    air = s3.Air.earth(A)
    k, cmin = r.rose_k, r.c_min
    out = torch.zeros(A, N, 8, s3.SECTOR_DIM, dtype=torch.float64)
    touch = torch.zeros(A, N, 3, dtype=torch.float64)
    light_lvl = torch.zeros(A, N, dtype=torch.float64)
    far_used = 0
    props = materials.props(items.comp, items.mass)
    irad = ((3 * props["volume_m3"] / (4 * math.pi)) ** (1 / 3)).double()
    ivis = optics.mix(items.comp).double()
    iem = optics.mix(items.comp, table=optics.EMISSIVITY)
    glow = optics.visible_exitance(items.temp_k, iem).double()
    ihard = props["tool_hardness"].double()
    M = b.mass_kg.double()
    rad = (3 * M / (4 * math.pi * s3.TISSUE_DENSITY)) ** (1 / 3)
    theta_res, G = s3.eye_optics(b.eye * b.mass_kg, 1 - b.damage)
    theta_res, G = theta_res.double(), G.double()
    i_th = s3.ear_threshold(b.ear * b.mass_kg, 1 - b.damage).double()
    p_call = s3.call_acoustic_power(b.loudness, b.voice, b.muscle, b.mass_kg).double()
    p_noise = s3.footfall_power(b.mass_kg, b.speed_m_s).double()
    a_call = s3.absorption_db_m(s3.call_frequency(b.voice, b.mass_kg, air.sound_speed()[:, None]), air).double()
    a_noise = s3.absorption_db_m(1000.0, air).double()
    skin = optics.vis("tissue:skin")
    for a in range(A):
        geom1 = pt.geometry_from_elevation(z[a:a + 1], L, -1e4, R_781)
        light, ground = scene.light_vis[a].double(), scene.ground_vis[a].double()

        def at(field, p):
            cx, cy = _cell(p, L, n)
            return float(field[cx, cy])

        fl = []
        for f in range(fires.alive.shape[1]):
            if bool(fires.alive[a, f]):
                fp = tuple(fires.pos[a, f].tolist())
                pw = float(optics.visible_exitance(fires.temp_k[a, f], optics.FLAME_EMISSIVITY)) * float(
                    fires.area_m2[a, f])
                fr = math.sqrt(float(fires.area_m2[a, f]) / (4 * math.pi))
                fl.append((fp, pw, fr, at(ground, fp) * at(light, fp) / math.pi))

        def fire_view(p, h, eye=None):
            """[(fire, irradiance, sector vector, matters, eye sees, x, line clear)] of every fire at point p."""
            rows = []
            for fp, pw, fr, lbf in fl:
                v = _wrap(torch.tensor(fp, dtype=torch.float64) - torch.tensor(p, dtype=torch.float64), L).tolist()
                d = math.hypot(*v)
                irr = pw / (4 * math.pi * max(d, fr) ** 2)
                sees, x = False, 0.0
                if eye is not None:
                    th, Gi = eye
                    ns = irr * Gi
                    nb = lbf * max(_omega(fr, d), math.pi / 4 * th * th) * Gi
                    sees = _rose(ns, nb) >= k and ns >= cmin * nb and irr > 0
                    x = min(ns / max(nb, 1e-30), 1e30)
                matters = (irr >= cmin * at(light, p) and irr > 0) or sees
                rows.append([irr, v, matters, sees, x, (p[0] + v[0], p[1] + v[1]), 2 * fr])
            clear = _los(geom1, p, [row[5] for row in rows], h, [row[6] for row in rows])
            for row, c in zip(rows, clear):
                row.append(c)
            return rows

        def lit(p, h, eye=None):
            return at(light, p) + sum(irr * (1.0 if (not m or c) else 0.0)
                                      for irr, _, m, _, _, _, _, c in fire_view(p, h, eye))

        P = b.pos[a].double()
        E_body = {j: lit(P[j].tolist(), 2 * float(rad[a, j]), (float(theta_res[a, j]), float(G[a, j])))
                  for j in range(N) if bool(b.alive[a, j])}
        E_item = {t: lit(items.pos[a, t].tolist(), 2 * float(irad[a, t]))
                  for t in range(items.alive.shape[1]) if bool(items.alive[a, t]) and bool(items.on_ground[a, t])}
        for i in range(N):
            if not bool(b.alive[a, i]):
                continue
            pi_, hi = P[i].tolist(), float(b.heading[a, i])
            ci, cci = _cell(pi_, L, n), _cell(pi_, L, nc)
            th, Gi, ri, Ith = float(theta_res[a, i]), float(G[a, i]), float(rad[a, i]), float(i_th[a, i])
            om_res = math.pi / 4 * th * th
            hvec = (math.cos(hi), math.sin(hi))
            light_lvl[a, i] = math.log10(1 + E_body[i] / s3.night_light()) / 8
            objs = []                # (d, sector, seen, theta, size, contrast, motion, x_glow)
            emit = [0.0] * 8
            callsum = [[0.0] * 8 for _ in range(8)]
            wsum, isum = [0.0] * 8, [0.0] * 8
            contact = None
            targets = [("b", j, P[j].tolist()) for j in range(N) if j != i and bool(b.alive[a, j])] + \
                [("i", t, items.pos[a, t].tolist()) for t in E_item]
            for kind, j, pj in targets:
                fine = _in_block(ci, _cell(pj, L, n), n)
                if not fine and not _in_block(cci, _cell(pj, L, nc), nc):
                    continue
                v = _wrap(torch.tensor(pj, dtype=torch.float64) - P[i], L).tolist()
                d = math.hypot(*v)
                q = (pi_[0] + v[0], pi_[1] + v[1])
                s = _sector(hi, v)
                if kind == "b":
                    rj, E = float(rad[a, j]), E_body[j]
                    lt, lb, mg = skin * E / math.pi, at(ground, pj) * E / math.pi, 0.0
                    speed, mass, hard, temp = float(b.speed_m_s[a, j]), float(M[a, j]), s3.BODY_HARDNESS, \
                        float(b.body_k[a, j])
                else:
                    rj, E = float(irad[a, j]), E_item[j]
                    mg = float(glow[a, j])
                    lt, lb = (float(ivis[a, j]) * E + mg) / math.pi, at(ground, pj) * E / math.pi
                    speed, mass, hard, temp = 0.0, float(items.mass[a, j]), float(ihard[a, j]), \
                        float(items.temp_k[a, j])
                clear = _los(geom1, pi_, [q], 2 * ri, [2 * rj])[0]
                om = _omega(rj, d)
                nb = lb * max(om, om_res) * Gi
                dl = abs(lt - lb)
                seen = clear and _rose(dl * om * Gi, nb) >= k and dl * om >= cmin * lb * max(om, om_res)
                ns = mg * min(1.0, rj / d) ** 2 * Gi
                gseen = clear and mg > 0 and _rose(ns, nb) >= k and ns >= cmin * nb
                theta = 2 * math.asin(min(1.0, rj / d))
                m = speed * r.integration_s / (d * th)
                heard = False
                if kind == "b":
                    rs = max(d, rj)
                    occ = 1.0 if clear else r.occlusion ** 2
                    ic = float(p_call[a, j]) / (4 * math.pi * rs * rs) * 10 ** (-float(a_call[a, j]) * rs / 10) * occ
                    inz = float(p_noise[a, j]) / (4 * math.pi * rs * rs) * 10 ** (-float(a_noise[a]) * rs / 10) * occ
                    heard = fine or ic >= Ith or inz >= Ith
                    if heard:
                        w = min(1.0, max(0.0, 10 * math.log10(max(ic, 1e-38) / Ith) / 60))
                        for dd in range(8):
                            callsum[s][dd] += w * float(b.calls[a, j, dd])
                        wsum[s] += w
                        isum[s] += ic + inz
                if not fine and (seen or gseen or (heard and kind == "b")):
                    far_used += 1
                objs.append((d, s, seen, theta, theta / th / (1 + theta / th),
                             (lt - lb) / (lt + lb) if lt + lb > 0 else 0.0, m, ns / max(nb, 1e-30) if gseen else 0.0))
                gap = math.hypot(v[0] - ri * hvec[0], v[1] - ri * hvec[1]) - rj
                if gap <= r.mouth_reach_radii * ri and (contact is None or gap < contact[0]):
                    contact = (gap, mass / float(M[a, i]), hard, temp)
            objs.sort(key=lambda o: o[0])
            first = {}
            cover, flow = [0.0] * 8, [0.0] * 8
            for d, s, seen, theta, size, con, m, xg in objs:
                emit[s] += xg
                if seen:
                    cover[s] += theta
                    flow[s] += m
                    if s not in first:
                        first[s] = (size, con, m / (1 + m))
            for row in fire_view(pi_, 2 * ri, (th, Gi)):
                irr, v, matters, sees, x, _, _, c = row
                if sees and c:
                    emit[_sector(hi, v)] += x
            for s in range(8):
                if s in first:
                    out[a, i, s, 0], out[a, i, s, 1], out[a, i, s, 2] = first[s]
                out[a, i, s, 3] = min(1.0, cover[s] / (2 * math.pi / 8))
                out[a, i, s, 4] = flow[s] / (1 + flow[s])
                out[a, i, s, 6] = min(1.0, math.log10(1 + emit[s]) / 8)
                for dd in range(8):
                    out[a, i, s, 7 + dd] = max(-1.0, min(1.0, callsum[s][dd] / max(1.0, wsum[s])))
                lvl = 10 * math.log10(max(isum[s], 1e-38) / Ith)
                out[a, i, s, 15] = min(1.0, max(0.0, lvl / 60))
            if contact is not None:
                touch[a, i] = torch.tensor([contact[1] / (1 + contact[1]), contact[2] / 10,
                                            math.tanh((contact[3] - float(b.body_k[a, i])) / 10)], dtype=torch.float64)
    return out, touch, light_lvl, far_used


def test_neighbour_senses_equal_a_brute_force_reference():
    geom, z, b, items, fires, scene = _case()
    x = s3.observe(geom, b, scene, air=s3.Air.earth(geom.A), items=items, fires=fires, rules=RULES)
    ref, touch, lvl, far_used = _reference(geom, z, b, items, fires, scene)
    A, N = b.alive.shape
    alive = b.alive
    sec = x[..., :8 * s3.SECTOR_DIM].reshape(A, N, 8, s3.SECTOR_DIM).double()
    for c, name in enumerate(s3.SECTOR_CHANNELS):
        if name == "ground":
            continue                                    # the ground rays have their own tests (no neighbours)
        err = (sec[..., c] - ref[..., c]).abs()[alive]
        tol = 1e-4 * (1 + ref[..., c].abs()[alive])
        assert bool((err <= tol).all()), (name, float(err.max()), (err > tol).nonzero()[:5].tolist())
    for c, name in enumerate(("touch_mass", "touch_hard", "touch_temp")):
        assert torch.allclose(x[..., s3.CHANNELS.index(name)].double(), touch[..., c], atol=1e-5), name
    assert torch.allclose(x[..., s3.CHANNELS.index("intero_light")].double(), lvl, atol=1e-5)
    # the case is not trivial: seen, unseen, occluded, heard, far, glowing and touching pairs
    assert far_used > 10                                                    # targets past the fine block count
    assert float((sec[..., 0] > 0)[alive].float().mean()) > 0.02          # nearest objects seen
    assert float((sec[..., 2] > 0)[alive].float().mean()) > 0.02          # moving ones among them
    assert float((sec[..., 6] > 0)[alive].float().mean()) > 0.02          # light received
    assert float((sec[..., 15] > 0)[alive].float().mean()) > 0.05         # sound heard
    assert float((touch[..., 0] > 0).float().sum()) > 0                   # some contact


def test_cross_neighbours_equal_brute_force():
    gen = torch.Generator().manual_seed(5)
    A, N, M, n, L = 2, 120, 200, 16, 128.0
    geom = pt.geometry_from_elevation(torch.zeros(A, n, n), L, -1e4, R_781)
    q = torch.rand(A, N, 2, generator=gen) * L
    t = torch.rand(A, M, 2, generator=gen) * L
    t[0, :15, 0] = torch.rand(15, generator=gen) * 1.0                      # crowd the seam
    qa = torch.rand(A, N, generator=gen) < 0.8
    ta = torch.rand(A, M, generator=gen) < 0.8
    idx, off, dist, count = s3.cross_neighbours(geom, q, qa, t, ta, K=M, max_scan=M)
    for a in range(A):
        for i in range(N):
            ci = _cell(q[a, i].tolist(), L, n)
            want = {j for j in range(M)
                    if bool(qa[a, i]) and bool(ta[a, j]) and _in_block(ci, _cell(t[a, j].tolist(), L, n), n)}
            got = {int(j) for j in idx[a, i] if int(j) >= 0}
            assert got == want, (a, i)
            assert int(count[a, i]) == len(want)
    valid = idx >= 0
    other = torch.gather(t, 1, idx.clamp_min(0).reshape(A, -1, 1).expand(-1, -1, 2)).reshape(A, N, M, 2)
    assert torch.allclose(off[valid], pt.wrap(other - q[:, :, None], L)[valid], atol=1e-4)
    assert bool((dist[..., 1:] >= dist[..., :-1]).all())
    # a small K keeps the K nearest; no targets give padding
    idx4, _, d4, _ = s3.cross_neighbours(geom, q, qa, t, ta, K=4, max_scan=M)
    assert torch.allclose(d4, dist[..., :4])
    e_idx, _, e_d, e_c = s3.cross_neighbours(geom, q, qa, t[:, :0], ta[:, :0], K=4)
    assert bool((e_idx == -1).all()) and bool(torch.isinf(e_d).all()) and bool((e_c == 0).all())


def test_the_far_hash_equals_brute_force():
    """The coarse hash (cells of 4 fine cells; 3 x 3 of them, the fine block excluded, each target within its reach)
    keeps exactly the brute-force set, ranked by distance over reach."""
    gen = torch.Generator().manual_seed(8)
    A, N, M, n, L, cell = 2, 60, 300, 32, 512.0, 4
    geom = pt.geometry_from_elevation(torch.zeros(A, n, n), L, -1e4, R_781)
    q = torch.rand(A, N, 2, generator=gen) * L
    t = torch.rand(A, M, 2, generator=gen) * L
    qa = torch.rand(A, N, generator=gen) < 0.9
    ta = torch.rand(A, M, generator=gen) < 0.9
    reach = 200 * torch.rand(A, M, generator=gen)
    idx, off, dist, count = s3.cross_neighbours(geom, q, qa, t, ta, K=M, max_scan=M, cell=cell, reach=reach,
                                                exclude_block=True)
    nc = n // cell
    for a in range(A):
        for i in range(N):
            pq = q[a, i].tolist()
            want = set()
            for j in range(M):
                pj = t[a, j].tolist()
                d = math.hypot(*_wrap(torch.tensor(pj) - torch.tensor(pq), L).tolist())
                if (bool(qa[a, i]) and bool(ta[a, j]) and _in_block(_cell(pq, L, nc), _cell(pj, L, nc), nc)
                        and not _in_block(_cell(pq, L, n), _cell(pj, L, n), n) and d <= float(reach[a, j])):
                    want.add(j)
            got = [int(j) for j in idx[a, i] if int(j) >= 0]
            assert set(got) == want, (a, i)
            assert int(count[a, i]) == len(want)
            ratio = [float(dist[a, i, kk]) / float(reach[a, j]) for kk, j in enumerate(got)]
            assert ratio == sorted(ratio)
