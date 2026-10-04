"""The habitat globe: cube-sphere grid, great-circle motion, terrain, sea level and ground deposits.

The grid is an equiangular gnomonic cube-sphere (Ronchi, Iacono & Paolucci 1996, J. Comput. Phys.
124, 93): six faces of G x G cells. Cell (f, i, j) covers the gnomonic angles alpha in
[a_i, a_i+1] and beta in [a_j, a_j+1] of face f, with a_k = -pi/4 + k pi/(2G). Its sides are
great-circle arcs and its solid angle is exact (the rectangle formula of the gnomonic
projection), so the six faces tile the sphere and the areas sum to 4 pi. Cell index
c = f G^2 + i G + j. The geometry is built once in float64 on the CPU and stored in float32 on
the device, so every device sees the same grid.

Conventions. Positions are unit vectors (x, y, z) with z through the north pole and
lon = atan2(y, x). A heading is a compass angle from local north, clockwise seen from outside
(0 north, pi/2 east): the direction is cos(h) north + sin(h) east. At a pole, where north is
undefined, the basis follows lon = atan2(y, x). Angles on the unit sphere are in radians; a
distance in metres is the angle times the habitat radius. Every rule that sets up a world is
written in angles (rad), not in cells, so the pattern does not depend on G.

The Laplacian is the unit-sphere (angular) Laplacian, so a diffusion coefficient in the
Budyko-Sellers form (W m^-2 K^-1) means the same thing at every resolution and radius. It is a
conservative finite-volume operator (antisymmetric side fluxes, so sum(area * laplacian) is zero)
of the diamond type, which stays consistent on the skewed cells of the cube-sphere; its low
eigenvalues converge to -l(l+1) at second order and it is stable (no growing mode). It is
supraconvergent: diffusion with it converges at second order, but its pointwise truncation error
does not shrink on the cells along the face rims (about 0.6 for Y_2^0 at every G; the RMS falls
only as G^-0.5). Use it inside ``diffuse`` or in divergence form, not as a pointwise curvature.

Every number that sets up a world carries a provenance tag in ``PROVENANCE``.
"""
from __future__ import annotations

import math
import warnings
from collections.abc import Sequence

import torch

from .constants import EARTH_OCEAN_MASS_FRACTION, G, M_EARTH, R_EARTH

TWO_PI = 2 * math.pi

# Face frames (normal, u, v) with u x v = normal: going +u then +v is anticlockwise seen from outside.
_FACES = (((1, 0, 0), (0, 1, 0), (0, 0, 1)),
          ((0, 1, 0), (-1, 0, 0), (0, 0, 1)),
          ((-1, 0, 0), (0, -1, 0), (0, 0, 1)),
          ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
          ((0, 0, 1), (0, 1, 0), (-1, 0, 0)),
          ((0, 0, -1), (0, 1, 0), (1, 0, 0)))
# Neighbour slots as face-local (di, dj): the 4 side neighbours first, then the 4 diagonals.
OFFSETS = ((1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1))

# ------------------------------------------------------------------------------- set-up numbers
# terrain
FAULTS = 128            # new_rule: random great-circle faults per world
FAULT_WIDTH_RAD = 0.05  # new_rule: tanh width of a fault scarp (500 m on the 10 km habitat)
WAVES = 64              # new_rule: random plane waves per world
WAVE_K = (1.0, 24.0)    # new_rule: wavenumbers in rad^-1, log-uniform (largest feature a hemisphere)
WAVE_SLOPE = 0.5        # reference: amplitude k^-0.5 per log band = degree variance ~ l^-2 (Turcotte 1987)
FAULT_SHARE = 0.5       # new_rule: faults and waves mixed half and half (each at unit std)
# reference: ocean 1.332e9 km3 at 3,682 m mean depth = 3.618e8 km2 of Earth's 5.101e8 km2 (Charette & Smith 2010,
# Oceanography 23(2), 112)
EARTH_OCEAN_FRACTION = 0.709
# new_rule (calibrated): the share of the habitat's relief shell that Earth's ocean fills. It is fitted on this
# terrain generator so that the mean ocean fraction over many worlds is EARTH_OCEAN_FRACTION (test_globe checks it).
EARTH_FILL = 0.143
SEA_LEVEL_ITERS = 80    # numerics: float64 bisection steps for the sea level (the bracket shrinks by 2^-80)
INLAND_CAP_RAD = 0.3    # numerics: inland distances are worked out up to this angle (beyond every deposit rule)
# deposits
ROCK_DENSITY = 2670.0   # reference: standard upper-crust (Bouguer reduction) density, Hinze 2003, Geophysics 68, 1559
ACCESSIBLE_DEPTH_M = 1.0  # new_rule: the ground an individual can reach (surface stone, outcrop, shallow digging)
CLAY_START = 0.1        # new_rule: clay starts at a tenth of its crust share; weathering adds it later
ORE_PATCHES = 12        # new_rule: ore patches per ore kind per world
ORE_RADIUS_RAD = (0.02, 0.06)  # new_rule: patch radius (200-600 m on the 10 km habitat)
DEPOSIT_NOISE = 0.5     # new_rule: log-normal patchiness (std of ln) of every deposit's share
NOISE_WAVES = 32        # new_rule: plane waves in each deposit's patchiness field
NOISE_K = (2.0, 24.0)   # new_rule: their wavenumbers in rad^-1 (features 0.13-1.6 rad across)
SHELF_SCALE = 0.05      # new_rule: shelf weight exp(-depth / (SHELF_SCALE x relief)) on the sea floor
SAND_SCALE_RAD = 0.05   # new_rule: coastal sand falls off as exp(-distance inland / SAND_SCALE_RAD)
SAND_FLOOR = 0.05       # new_rule: inland sand (dunes, river sand) relative to the coast's
SALT_MIN_DIST_RAD = 0.08  # new_rule: salt basins lie at least this far inland (800 m on the 10 km habitat)
SUPPORT_MARGIN = 2.0    # new_rule: a deposit's area must cover SUPPORT_MARGIN x its share of the land, else it widens
FILL_ITERS = 400        # numerics: at most this many matrix-scaling sweeps when the ground is filled
FILL_TOL = 1e-9         # numerics: relative error of every species' inventory at which the sweeps stop

# Where each crust species is exposed (new_rule after textbook geology); unknown species: "rock".
DEPOSIT_KIND = {"granite": "felsic", "basalt": "mafic", "sandstone": "sediment", "limestone": "carbonate",
                "flint": "chert", "clay": "clay", "sand": "sand", "salt": "evaporite", "hematite": "iron_ore",
                "magnetite": "magnetite", "malachite": "copper_ore", "native_copper": "copper_ore",
                "cassiterite": "tin_ore", "pyrite": "sulfide_ore"}
# new_rule: share weight of a kind on land, (base + slope x up) ** power x factor, with up = height above the sea
# over the highest point's (0 at the shore, 1 at the top) and factor "patch" (its ore patches), "coast" (the
# coastal sand band) or "inland" (the share of the cell at least SALT_MIN_DIST_RAD inland) or None.
LAND_WEIGHTS = {"felsic": (0.2, 1.6, 1, None),       # granite: uplands and shields
                "mafic": (1.6, -1.2, 1, None),       # basalt: lowland flows and rifts
                "sediment": (1.4, -0.8, 1, None),    # sandstone: basins
                "carbonate": (1.2, -0.8, 1, None),   # limestone: lowland platforms
                "chert": (1.2, -0.8, 1, "patch"),    # flint: nodules in carbonate, in patches
                "clay": (1.0, -1.0, 1, None),        # clay: low, wet ground
                "sand": (1.0, 0.0, 1, "coast"),      # sand: beaches and coastal dunes
                "evaporite": (1.0, -1.0, 3, "inland"),  # salt: low inland basins (until climate gives aridity)
                "iron_ore": (1.0, 0.0, 1, "patch"),
                "magnetite": (2.1, -1.2, 1, "patch"),   # magnetite: patches, more in mafic low ground
                "copper_ore": (0.5, 1.0, 1, "patch"),   # copper: patches, more in uplands
                "tin_ore": (0.2, 1.6, 1, "patch"),      # tin: patches that follow granite
                "sulfide_ore": (1.0, 0.0, 1, "patch"),
                "rock": (1.0, 0.0, 1, None)}
# new_rule: share weight of a marine kind on the sea floor, deep + (shelf - deep) x shelfness, where shelfness =
# exp(-depth / (SHELF_SCALE x relief)); other kinds are absent from the sea floor.
MARINE_WEIGHTS = {"mafic": (1.0, 0.2),       # oceanic crust bare in the deep
                  "carbonate": (0.6, 1.5),   # carbonate platforms and ooze
                  "clay": (1.0, 0.5),        # pelagic clay
                  "sand": (0.05, 1.0)}       # shelf sand


def _note(*names):
    return "; ".join(f"{name} = {globals()[name]}" for name in names)


PROVENANCE = {
    "globe.grid": ("new_rule", "equiangular gnomonic cube-sphere, 6 x G x G cells (Ronchi et al. 1996)"),
    "globe.area": ("derived", "exact solid angle of each gnomonic rectangle; sums to 4 pi"),
    "terrain.faults": ("new_rule", f"{_note('FAULTS', 'FAULT_WIDTH_RAD')}: random great-circle faults, tanh scarp "
                       "width in rad"),
    "terrain.waves": ("new_rule", f"{_note('WAVES', 'WAVE_K')}: plane waves sin(k p.d + phase), k log-uniform in "
                      "rad^-1"),
    "terrain.wave_slope": ("reference", f"{_note('WAVE_SLOPE')}: amplitude k^-slope, topography degree variance "
                           "~ l^-2 (Turcotte 1987, JGR 92, E597)"),
    "terrain.mix": ("new_rule", f"{_note('FAULT_SHARE')}: faults and waves at unit std, rescaled to [0, relief_m]"),
    "terrain.sea_level": ("derived", f"{_note('SEA_LEVEL_ITERS')}: float64 bisection, the exact spherical-shell "
                          "ocean volume equals the water volume"),
    "terrain.earth_ocean_fraction": ("reference", f"{_note('EARTH_OCEAN_FRACTION')} of Earth's surface is ocean "
                                     "(Charette & Smith 2010, Oceanography 23(2), 112)"),
    "terrain.water_volume": ("new_rule", f"{_note('EARTH_FILL')} (calibrated): habitat water keeps the real "
                             "planet's ratio of equivalent ocean depth to relief (relief ~ 1/g); an Earth ocean "
                             "fills EARTH_FILL of the relief shell, fitted so that it covers EARTH_OCEAN_FRACTION "
                             "on average on this terrain generator"),
    "terrain.inland": ("new_rule", f"{_note('INLAND_CAP_RAD')}: inland distance = angle to the nearest ocean cell "
                       "centre (vector propagation, Danielsson 1980) less half a cell, capped (numerics)"),
    "deposits.rock_density": ("reference", f"{_note('ROCK_DENSITY')} kg/m3 (Hinze 2003, Geophysics 68, 1559)"),
    "deposits.accessible_depth_m": ("new_rule", f"{_note('ACCESSIBLE_DEPTH_M')} m of reachable ground: every "
                                    "land and sea-floor cell holds ROCK_DENSITY x ACCESSIBLE_DEPTH_M kg/m2 of "
                                    "ground (less the clay still to form)"),
    "deposits.land_mix": ("new_rule", "the land holds the crust mix (formation, a continental lithology mix): the "
                          "land-area mean of every species is its crust share; each land cell's column is split "
                          "among the species in proportion to LAND_WEIGHTS x patchiness x per-species factors "
                          "found by matrix scaling (Sinkhorn & Knopp 1967), "
                          f"{_note('FILL_ITERS', 'FILL_TOL')} (numerics)"),
    "deposits.sea_floor": ("new_rule", "the sea floor holds only the marine species (basalt, limestone, clay, "
                           "sand) in their crust proportions renormalised among themselves, kept out of the land "
                           f"inventory; {_note('MARINE_WEIGHTS')}"),
    "deposits.patterns": ("new_rule", f"species -> kind by DEPOSIT_KIND (textbook geology); {_note('LAND_WEIGHTS')}; "
                          f"{_note('SAND_SCALE_RAD', 'SAND_FLOOR')}; "
                          f"{_note('SALT_MIN_DIST_RAD')} (salt: a placeholder for dry basins until climate gives "
                          "aridity); unknown species: rock"),
    "deposits.shelf": ("new_rule", f"{_note('SHELF_SCALE')} of the relief; the shelf then covers about 9-12 % of "
                       "the ocean on this terrain, near Earth's continental shelf, about 9 % of the ocean "
                       "(Harris et al. 2014, Mar. Geol. 352, 4)"),
    "deposits.support": ("new_rule", f"{_note('SUPPORT_MARGIN')}: a species whose own area is too small widens "
                         "to low land ((1 - up)^3), then to all land; without land it is absent (reported)"),
    "deposits.clay_start": ("new_rule", f"{_note('CLAY_START')} of the crust's clay share at the start"),
    "deposits.ore_patches": ("new_rule", f"{_note('ORE_PATCHES', 'ORE_RADIUS_RAD')}: Gaussian caps exp(-chord^2 / "
                             "r^2), each carrying pi r^2 whatever the grid"),
    "deposits.noise": ("new_rule", f"{_note('DEPOSIT_NOISE', 'NOISE_WAVES', 'NOISE_K')}: log-normal patchiness "
                       "exp(DEPOSIT_NOISE x unit-std plane waves)"),
}


# ------------------------------------------------------------------------------- sphere geometry
def _unit(v):
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-30)


def _dot(a, b):
    return (a * b).sum(-1)


def _cross(a, b):
    a, b = torch.broadcast_tensors(a, b)
    return torch.linalg.cross(a, b, dim=-1)


def _wrap(x, top):
    """torch.remainder can round up to exactly ``top``; fold that back into [0, top)."""
    return torch.where(x >= top, x - top, x)


def tangent_basis(p):
    """(east, north) unit vectors at points p [..., 3]."""
    lon = torch.atan2(p[..., 1], p[..., 0])
    east = torch.stack((-lon.sin(), lon.cos(), torch.zeros_like(lon)), -1)
    return east, _cross(_unit(p), east)


def angle(p, q):
    """Geodesic angle (rad) between unit vectors p and q [..., 3] (broadcast)."""
    return torch.atan2(_cross(p, q).norm(dim=-1), _dot(p, q))


def move(p, heading, dist_m, radius_m):
    """Move p [..., 3] by dist_m along the great circle of heading [...] on a sphere of radius_m.

    Returns (p2, heading2): the end point and the heading there, still from local north, in
    [0, 2 pi). dist_m and radius_m broadcast against heading.
    """
    east, north = tangent_basis(p)
    d = heading.cos()[..., None] * north + heading.sin()[..., None] * east
    theta = (torch.as_tensor(dist_m, dtype=p.dtype, device=p.device) / radius_m).to(p.dtype)
    c, s = theta.cos()[..., None], theta.sin()[..., None]
    p2 = _unit(c * p + s * d)
    d2 = c * d - s * p
    e2, n2 = tangent_basis(p2)
    return p2, _wrap(torch.remainder(torch.atan2(_dot(d2, e2), _dot(d2, n2)), TWO_PI), TWO_PI)


def bearing(p, q, heading):
    """Direction of q seen from p, relative to p's heading, in [-pi, pi) (positive = clockwise)."""
    east, north = tangent_basis(p)
    t = q - _dot(p, q)[..., None] * p
    azimuth = torch.atan2(_dot(t, east), _dot(t, north))
    return _wrap(torch.remainder(azimuth - heading + math.pi, TWO_PI), TWO_PI) - math.pi


def slerp(p, q, t):
    """Points at fraction t [...] of the great-circle arc from p to q (for sight lines).

    The arc leaves p along the tangent (p x q) x p, which is normal to p however close q comes to
    p or -p; for antipodal q (any great circle is shortest) it leaves eastwards.
    """
    omega = angle(p, q)[..., None]
    t = torch.as_tensor(t, dtype=p.dtype, device=p.device)[..., None]
    tang = _cross(_cross(p, q), p)
    size = tang.norm(dim=-1, keepdim=True)
    east, _ = tangent_basis(p)
    u = torch.where(size > 1e-6, tang / size.clamp_min(1e-30), east)
    return _unit((t * omega).cos() * p + (t * omega).sin() * u)


def _cell_of(p, frame, G):
    """Exact face projection: the cell index of points p [..., 3] (frame [6, 3, 3] of p's dtype)."""
    face = (p @ frame[:, 0].T).argmax(-1)
    n, u, v = frame[face, 0], frame[face, 1], frame[face, 2]
    pn = _dot(p, n)
    step = math.pi / (2 * G)
    i = torch.floor((torch.atan(_dot(p, u) / pn) + math.pi / 4) / step).long().clamp(0, G - 1)
    j = torch.floor((torch.atan(_dot(p, v) / pn) + math.pi / 4) / step).long().clamp(0, G - 1)
    return face * G * G + i * G + j


def _at_vertices(field, vertex_cells, vertex_w):
    # relative to the vertex's first cell, so a constant field is reproduced exactly despite rounding
    first = field[..., vertex_cells[:, 0]]
    return first + ((field[..., vertex_cells] - first[..., None]) * vertex_w).sum(-1)


def _laplacian(field, lap_a, lap_b, side_v, vertex_cells, vertex_w, nbr4, area):
    fv = _at_vertices(field, vertex_cells, vertex_w)
    flux = (lap_a * (field[..., nbr4] - field[..., None])
            + lap_b * (fv[..., side_v[..., 1]] - fv[..., side_v[..., 0]]))
    return flux.sum(-1) / area


# Corner order of a cell: v00, v10, v11, v01 (anticlockwise from outside). Side k of OFFSETS[k] runs
# anticlockwise from corner SIDES[k][0] to SIDES[k][1], so a shared side runs the other way for the
# neighbour.
SIDES = ((1, 2), (2, 3), (3, 0), (0, 1))


class Globe:
    """Equiangular gnomonic cube-sphere with G x G cells per face, C = 6 G^2 cells.

    Attributes (on ``device``): centers [C,3] unit vectors, area [C] steradians (float32; area64 in
    float64 for ledger sums), spacing [C] = sqrt(area) (the cell's size in rad), lat [C], lon [C],
    east [C,3], north [C,3], corners [C,4,3] (anticlockwise from outside), nbr [C,8] long (slots in
    ``OFFSETS`` order: 4 side neighbours, then 4 diagonals; across face edges and corners),
    nbr_valid [C,8] bool (False only for the missing diagonal of the 24 cells at the 8 cube corners,
    where the slot holds the cell itself), nbr4 [C,4] the side neighbours. Vertices: V = C + 2 grid
    vertices at ``vertices`` [V,3], corner_id [C,4], vertex_cells [V,4] and vertex_w [V,4] (linear
    interpolation from the 3 or 4 cells around a vertex). delta = pi / (2G) is the nominal cell
    angle and min_spacing the smallest angle between side neighbours' centres.

    The Laplacian is a diamond-type finite-volume scheme (Coudiere, Vila & Villedieu 1999, M2AN 33,
    493): the flux through a side uses the centre difference across it and the vertex difference
    along it, so it stays consistent on the skewed cells near the cube corners and across the face
    edges, where the plain two-point flux is not. Fluxes are antisymmetric, so the scheme conserves.
    """

    def __init__(self, G: int = 48, device="cpu"):
        if G < 2:
            raise ValueError("G must be at least 2")
        self.G, self.C, self.device = int(G), 6 * int(G) ** 2, torch.device(device)
        self.delta = math.pi / (2 * self.G)
        g, f64 = self.G, torch.float64
        frame = torch.tensor(_FACES, dtype=f64)
        edges = -math.pi / 4 + self.delta * torch.arange(g + 1, dtype=f64)
        mids = edges[:-1] + self.delta / 2
        f, i, j = (t.reshape(-1) for t in torch.meshgrid(torch.arange(6), torch.arange(g), torch.arange(g),
                                                          indexing="ij"))
        own = torch.arange(self.C)

        def point(face, a, b):
            return _unit(frame[face, 0] + a.tan()[..., None] * frame[face, 1] + b.tan()[..., None] * frame[face, 2])

        centers = point(f, mids[i], mids[j])
        x = edges.tan()

        def rect(a, b):  # solid angle of the gnomonic rectangle [0, x_a] x [0, x_b]
            return torch.atan(x[a] * x[b] / torch.sqrt(1 + x[a] ** 2 + x[b] ** 2))

        area = rect(i + 1, j + 1) - rect(i, j + 1) - rect(i + 1, j) + rect(i, j)
        # neighbours: a probe point at the target's centre, or just across the face edge
        off = torch.tensor(OFFSETS)
        ti, tj = i[:, None] + off[:, 0], j[:, None] + off[:, 1]
        in_i, in_j = (ti >= 0) & (ti < g), (tj >= 0) & (tj < g)
        out = math.pi / 4 + self.delta / 4
        a = torch.where(in_i, mids[ti.clamp(0, g - 1)], torch.where(ti < 0, -out, out))
        b = torch.where(in_j, mids[tj.clamp(0, g - 1)], torch.where(tj < 0, -out, out))
        valid = in_i | in_j
        nbr = torch.where(valid, _cell_of(point(f[:, None].expand(-1, 8), a, b), frame, g), own[:, None])
        nbr4 = nbr[:, :4]
        back = (nbr4[nbr4] == own[:, None, None]).float().argmax(-1)  # slot of c in its neighbour's list
        if not bool((nbr4[nbr4, back] == own[:, None]).all()):
            raise RuntimeError("cube-sphere side neighbours are not symmetric")

        # vertices: merge the corners that coincide (V = C + 2 by Euler's formula)
        corners = torch.stack([point(f, edges[i + di], edges[j + dj])
                               for di, dj in ((0, 0), (1, 0), (1, 1), (0, 1))], 1)          # [C,4,3]
        key = torch.round(corners.reshape(-1, 3) * 1e7).long()
        _, corner_id = torch.unique(key, dim=0, return_inverse=True)
        corner_id = corner_id.reshape(self.C, 4)
        V = int(corner_id.max()) + 1
        count = torch.bincount(corner_id.reshape(-1), minlength=V)
        if V != self.C + 2 or int(count.min()) < 3 or int(count.max()) > 4:
            raise RuntimeError("cube-sphere vertices did not merge")
        order = torch.argsort(corner_id.reshape(-1), stable=True)
        start = torch.cumsum(count, 0) - count
        vid = corner_id.reshape(-1)[order]
        rank = torch.arange(4 * self.C) - start[vid]
        vertex_cells = torch.empty(V, 4, dtype=torch.long)
        vertex_cells[:, 3] = (order // 4)[start]                 # the 4th slot of 3-cell vertices repeats one
        vertex_cells[vid, rank] = order // 4
        vertex_pos = torch.zeros(V, 3, dtype=f64).index_put_((corner_id.reshape(-1),), corners.reshape(-1, 3))
        # linear (least-squares) interpolation weights in the vertex's tangent plane
        e_v, n_v = tangent_basis(vertex_pos)
        rel = centers[vertex_cells] - vertex_pos[:, None]
        design = torch.stack((torch.ones(V, 4, dtype=f64), _dot(rel, e_v[:, None]), _dot(rel, n_v[:, None])), -1)
        use = torch.ones(V, 4, dtype=f64)
        use[:, 3] = (count == 4).to(f64)
        vertex_w = torch.linalg.pinv(design * use[..., None])[:, 0] * use                    # [V,4]
        # diamond fluxes: flux = a (f_n - f_c) + b (f_v2 - f_v1) through each side, worked out in the
        # gnomonic plane tangent at the side's midpoint (great circles are straight lines there)
        side = torch.tensor(SIDES)
        v1, v2 = corners[:, side[:, 0]], corners[:, side[:, 1]]                             # [C,4,3]
        mid = _unit(v1 + v2)

        def plane(p):
            return p / _dot(p, mid)[..., None]

        length = angle(v1, v2)
        e = plane(centers[nbr4]) - plane(centers[:, None])
        tv = plane(v2) - plane(v1)
        ell = tv.norm(dim=-1)
        t_hat = tv / ell[..., None]
        n_hat = _cross(t_hat, mid)
        s, h = _dot(e, t_hat), _dot(e, n_hat)
        if float(h.min()) <= 0:
            raise RuntimeError("cube-sphere side normal points the wrong way")
        lap_a, lap_b = length / h, -length * s / (h * ell)
        lap_a = 0.5 * (lap_a + lap_a[nbr4, back])
        lap_b = 0.5 * (lap_b + lap_b[nbr4, back])
        side_v = corner_id[:, side]                                                         # [C,4,2]
        if not bool((side_v[nbr4, back].flip(-1) == side_v).all()):
            raise RuntimeError("shared sides do not run opposite ways")

        dev, f32 = self.device, torch.float32
        self._frame = frame.to(dev, f32)
        self.face, self.i, self.j = f.to(dev), i.to(dev), j.to(dev)
        self.centers = centers.to(dev, f32)
        self.area64 = area.to(dev)
        self.area = area.to(dev, f32)
        self.spacing = area.sqrt().to(dev, f32)
        self.min_spacing = float(angle(centers[:, None], centers[nbr4]).min())
        self.lat = torch.atan2(centers[:, 2], centers[:, :2].norm(dim=-1)).to(dev, f32)
        self.lon = torch.atan2(centers[:, 1], centers[:, 0]).to(dev, f32)
        east, north = tangent_basis(centers)
        self.east, self.north = east.to(dev, f32), north.to(dev, f32)
        self.corners = corners.to(dev, f32)
        self.nbr, self.nbr_valid = nbr.to(dev), valid.to(dev)
        self.nbr4 = self.nbr[:, :4].contiguous()
        self.V = V
        self.vertices = vertex_pos.to(dev, f32)
        self.corner_id, self.vertex_cells = corner_id.to(dev), vertex_cells.to(dev)
        self.vertex_w = vertex_w.to(dev, f32)
        self._side_v = side_v.to(dev)
        self.lap_a, self.lap_b = lap_a.to(dev, f32), lap_b.to(dev, f32)
        self._lap64 = tuple(t.to(dev) for t in (lap_a, lap_b, side_v, vertex_cells, vertex_w, nbr4, area))
        # spectral radius of the Laplacian (power iteration on the CPU, +10 %): sets the sub-steps
        probe = torch.randn(self.C, generator=torch.Generator().manual_seed(0), dtype=f64)
        lap64 = (lap_a, lap_b, side_v, vertex_cells, vertex_w, nbr4, area)
        for _ in range(300):
            probe = _laplacian(probe, *lap64)
            radius = float(probe.norm())
            probe = probe / radius
        self.lap_radius = 1.1 * radius

    # ---------------------------------------------------------------------------- lookups
    def cell_of(self, p):
        """Cell index (long [...]) containing each point p [..., 3]: exact face projection."""
        return _cell_of(p, self._frame.to(p.dtype), self.G)

    def sample(self, field, p):
        """field [W, C] read at points p [W, ..., 3] -> [W, ...] (the containing cell's value)."""
        cells = self.cell_of(p)
        return field.gather(1, cells.reshape(cells.shape[0], -1)).reshape(cells.shape)

    def angle(self, p, q):
        return angle(p, q)

    def move(self, p, heading, dist_m, radius_m):
        return move(p, heading, dist_m, radius_m)

    def bearing(self, p, q, heading):
        return bearing(p, q, heading)

    # ---------------------------------------------------------------------------- operators
    def at_vertices(self, field):
        """field [..., C] interpolated to the grid vertices [..., V]."""
        return _at_vertices(field, self.vertex_cells, self.vertex_w)

    def laplacian(self, field):
        """Unit-sphere Laplacian of field [..., C] (rad^-2); sum(area * laplacian) is zero.

        Supraconvergent: its pointwise error does not shrink with G on the face-rim cells (O(1) there),
        while diffusion with it converges at second order. Use it in ``diffuse`` or in divergence
        form, not as a pointwise curvature.
        """
        return _laplacian(field, self.lap_a, self.lap_b, self._side_v, self.vertex_cells, self.vertex_w,
                          self.nbr4, self.area)

    def laplacian_csr(self):
        """The float64 Laplacian of :meth:`laplacian` as a sparse CSR matrix [C, C] (about 13 entries per
        row: the four side neighbours and the cells of the side ends' vertex interpolation), built once
        and cached; ``diffuse`` applies it with one sparse product per sub-step."""
        if getattr(self, "_csr", None) is None:
            lap_a, lap_b, side_v, vertex_cells, vertex_w, nbr4, area = (t.cpu() for t in self._lap64)
            C = self.C
            cell = torch.arange(C)[:, None].expand(C, 4)
            inv_area = (1.0 / area)[:, None].expand(C, 4)
            a = lap_a * inv_area
            rows = [cell.reshape(-1), cell.reshape(-1)]
            cols = [nbr4.reshape(-1), cell.reshape(-1)]
            vals = [a.reshape(-1), -a.reshape(-1)]
            # a vertex value: sum_j w_j f[cell_j] + (1 - sum_j w_j) f[cell_0]
            vw = torch.cat((vertex_w, (1.0 - vertex_w.sum(-1, keepdim=True))), -1)            # [V, 5]
            vc = torch.cat((vertex_cells, vertex_cells[:, :1]), -1)                            # [V, 5]
            b = lap_b * inv_area                                                                # [C, 4]
            for end, sign in ((1, 1.0), (0, -1.0)):
                v = side_v[..., end]                                                            # [C, 4]
                rows.append(cell[..., None].expand(C, 4, 5).reshape(-1))
                cols.append(vc[v].reshape(-1))
                vals.append((sign * b[..., None] * vw[v]).reshape(-1))
            m = torch.sparse_coo_tensor(torch.stack((torch.cat(rows), torch.cat(cols))), torch.cat(vals),
                                        (C, C), dtype=torch.float64, check_invariants=False).coalesce()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")              # "sparse CSR support is in beta"
                self._csr = m.to_sparse_csr().to(self.device)
        return self._csr

    def diffuse(self, field, kappa, steps: int | None = None, *, check: bool = True):
        """Diffuse field [..., C] by d f / d t = kappa lap f over a unit 'time' (kappa in rad^2).

        kappa is a number, [W, 1] or [W, C] (for example D dt / C_heat per cell, so a 1-day climate
        step is one call); it must not be negative. Explicit sub-steps, as many as the Laplacian's
        spectral bound needs to be stable (an explicit ``steps`` below that raises ValueError);
        deterministic. The sub-steps run in float64 and the result is cast back to the field's
        dtype once, so the content sum(area * f / kappa) is conserved (antisymmetric fluxes) up to
        that one rounding per call, with no bias accumulating over the sub-steps. A cell with kappa 0
        holds its value and acts as a fixed boundary (then the content is not defined). Each sub-step is
        one sparse product with :meth:`laplacian_csr`. ``check=False`` with a given ``steps`` skips the
        kappa range check (two host syncs) for a caller that worked the sub-steps out once.
        """
        kappa = torch.as_tensor(kappa, dtype=torch.float64, device=field.device)
        if check or steps is None:
            top, low = float(kappa.max()), float(kappa.min())
            if low < 0:
                raise ValueError("kappa must not be negative")
            if top == 0:
                return field.clone()
            need = max(1, math.ceil(top * self.lap_radius))
            if steps is None:
                steps = need
            elif steps < need:
                raise ValueError(f"{steps} sub-steps are unstable for kappa {top:g}: need at least {need}")
        C = field.shape[-1]
        lead = field.shape[:-1]
        rate = (kappa / steps).expand(*lead, C).reshape(-1, C).T                    # [C, B]
        L = self.laplacian_csr()
        out = field.double().reshape(-1, C).T.contiguous()                           # [C, B]
        for _ in range(steps):
            out = out + rate * (L @ out)
        return out.T.reshape(*lead, C).to(field.dtype)

    def smooth(self, field, length_rad: float):
        """Smooth field [..., C] by diffusion with kernel std length_rad (the same at every G)."""
        return self.diffuse(field, 0.5 * length_rad ** 2)

    def hops(self, source, max_hops: int):
        """Neighbour hops (long [W, C]) from each cell to the nearest source cell, capped at max_hops + 1.

        A grid count: its length in rad is about hops x delta, so rules should use ``distance``.
        """
        dist = torch.where(source, 0, max_hops + 1).long()
        for _ in range(max_hops):
            dist = torch.minimum(dist, dist[..., self.nbr].amin(-1) + 1)
        return dist

    def distance(self, source, max_rad: float):
        """Angle (rad, float32 [..., C]) from each cell centre to the nearest source cell centre, at most max_rad.

        source is bool [..., C]; without a source within max_rad the result is max_rad. Vector
        propagation over the 8 neighbours (Danielsson 1980, Comput. Graph. Image Process. 14, 227):
        each sweep a cell keeps the nearest source point that it or a neighbour knows of, and there
        are enough sweeps to cross max_rad, so the result does not depend on G beyond the cell size.
        """
        src = torch.where(source[..., None], self.centers, -self.centers)     # the antipode: none known yet
        for _ in range(math.ceil(max_rad / self.min_spacing) + 1):
            cand = torch.cat((src[..., None, :], src[..., self.nbr, :]), -2)     # [..., C, 9, 3]
            best = _dot(cand, self.centers[:, None]).argmax(-1)
            src = cand.gather(-2, best[..., None, None].expand(*best.shape, 1, 3)).squeeze(-2)
        return angle(self.centers, src).clamp_max(max_rad)


# ------------------------------------------------------------------------------- terrain
def _standard(globe, field):
    """Area-weighted zero mean, unit std over the sphere."""
    w = globe.area / globe.area.sum()
    mean = (field * w).sum(-1, keepdim=True)
    std = (((field - mean) ** 2) * w).sum(-1, keepdim=True).sqrt().clamp_min(1e-12)
    return (field - mean) / std


def plane_waves(globe, W, gen, count=WAVES, k_range=WAVE_K, slope=WAVE_SLOPE, chunk=16):
    """Smooth random field [W, C]: sum of sin(k p.d + phase) with amplitude k^-slope, unit std."""
    dev = globe.device
    d = _unit(torch.randn(W, count, 3, generator=gen, device=dev))
    k = k_range[0] * (k_range[1] / k_range[0]) ** torch.rand(W, count, generator=gen, device=dev)
    phase = torch.rand(W, count, generator=gen, device=dev) * TWO_PI
    field = torch.zeros(W, globe.C, device=dev)
    for s in range(0, count, chunk):
        arg = torch.einsum("cx,wkx->wck", globe.centers, d[:, s:s + chunk]) * k[:, None, s:s + chunk]
        field = field + (torch.sin(arg + phase[:, None, s:s + chunk]) * k[:, None, s:s + chunk] ** -slope).sum(-1)
    return _standard(globe, field)


def great_circle_faults(globe, W, gen, count=FAULTS, width=FAULT_WIDTH_RAD, chunk=32):
    """Random great-circle faults: one side up, the other down (a soft tanh scarp), unit std."""
    dev = globe.device
    normal = _unit(torch.randn(W, count, 3, generator=gen, device=dev))
    field = torch.zeros(W, globe.C, device=dev)
    for s in range(0, count, chunk):
        field = field + torch.tanh(torch.einsum("cx,wkx->wck", globe.centers, normal[:, s:s + chunk]) / width).sum(-1)
    return _standard(globe, field)


def _per_world(x, W, device):
    return torch.as_tensor(x, dtype=torch.float64, device=device).reshape(-1).expand(W)


def ocean_volume(globe, elevation_m, sea_level_m, radius_m):
    """Exact ocean volume (float64 [W], m^3): spherical shells between the ground and the sea surface.

    sum over cells of area x ((R + sea)^3 - (R + ground)^3) / 3 where the ground is below the sea.
    """
    radius = torch.as_tensor(radius_m, dtype=torch.float64, device=elevation_m.device).reshape(-1, 1)
    ground = radius + elevation_m.double()
    sea = radius + torch.as_tensor(sea_level_m, dtype=torch.float64, device=elevation_m.device).reshape(-1, 1)
    return (globe.area64 * (sea ** 3 - ground ** 3).clamp_min(0)).sum(-1) / 3


def sea_level(globe, elevation_m, water_volume_m3, radius_m, iters=SEA_LEVEL_ITERS):
    """Sea level (float32 [W], m above the radius_m sphere) at which ``ocean_volume`` equals water_volume_m3 [W].

    Float64 bisection on the exact shell volume. A water volume larger than the relief can hold
    floods everything (the sea stands above the highest point); no water gives the lowest point.
    For a water ledger, keep the ocean store as a volume (or mass) in float64 and re-solve the sea
    level with this function after every exchange. Convert a column exchanged with the ocean
    (vapour, soil, snow in kg/m^2) with one fixed cell measure everywhere (for example
    area64 x R^2), never with depth_m x area x R^2: that flat measure differs from the shell
    volume by several per cent on a 10 km habitat.
    """
    W = elevation_m.shape[0]
    water = _per_world(water_volume_m3, W, elevation_m.device)
    radius = _per_world(radius_m, W, elevation_m.device)
    e64 = elevation_m.double()
    lo, top = e64.amin(-1), e64.amax(-1)
    hi = torch.pow((radius + top) ** 3 + 3 * water / (4 * math.pi), 1 / 3) - radius
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        short = ocean_volume(globe, elevation_m, mid, radius) < water
        lo, hi = torch.where(short, mid, lo), torch.where(short, hi, mid)
    return torch.where(water > 0, 0.5 * (lo + hi), e64.amin(-1)).float()


def inland_distance(globe, land, cap=INLAND_CAP_RAD):
    """Distance (rad, float32 [W, C]) from each land cell centre to the coast, 0 on the ocean, at most cap.

    The coast lies half a cell beyond the nearest ocean cell centre (globe.distance less half the
    cell's spacing). A world without ocean is cap everywhere on land.
    """
    d = globe.distance(~land, cap + globe.delta)
    return torch.where(land, (d - 0.5 * globe.spacing).clamp(0, cap), torch.zeros_like(d))


def make_terrain(globe, W, gen, relief_m, water_volume_m3, radius_m, iters=SEA_LEVEL_ITERS):
    """Elevation, sea level, ocean depth and land for W worlds.

    relief_m [W] (or a number) is the height of the highest point above the lowest; elevation is
    measured from the sphere of radius_m (0 = the lowest ground). The sea level comes from
    ``sea_level`` (the exact ocean volume equals water_volume_m3 [W]). Returns elevation_m,
    depth_m [W, C] float32, sea_level_m [W] float32, land and coast [W, C] bool, inland_rad [W, C]
    (``inland_distance``), ocean_fraction [W] and ocean_volume_m3 [W] (float64, the volume at the
    stored float32 sea level). coast is the land beside the ocean (8 neighbours): a band one cell
    wide, so its area depends on G; rules that should not depend on G use inland_rad.
    """
    dev = globe.device
    relief = _per_world(relief_m, W, dev)
    radius = _per_world(radius_m, W, dev)
    height = FAULT_SHARE * great_circle_faults(globe, W, gen) + (1 - FAULT_SHARE) * plane_waves(globe, W, gen)
    low, high = height.amin(-1, keepdim=True), height.amax(-1, keepdim=True)
    elevation = ((height - low) / (high - low).clamp_min(1e-12) * relief[:, None].float()).float()
    sea = sea_level(globe, elevation, water_volume_m3, radius, iters)
    depth = (sea[:, None] - elevation).clamp_min(0)
    land = elevation >= sea[:, None]
    ocean = ~land
    coast = land & ocean[:, globe.nbr].any(-1)
    fraction = (ocean.double() * globe.area64).sum(-1) / (4 * math.pi)
    return {"elevation_m": elevation, "sea_level_m": sea, "depth_m": depth, "land": land, "coast": coast,
            "inland_rad": inland_distance(globe, land), "ocean_fraction": fraction.float(),
            "ocean_volume_m3": ocean_volume(globe, elevation, sea, radius)}


def habitat_water_volume(ocean_mass_kg, planet_radius_m, gravity_m_s2, relief_m, radius_m):
    """Ocean water volume (m^3, float64) for the habitat globe (new_rule, see PROVENANCE).

    The real planet's equivalent ocean depth (ocean volume over its surface) relative to its
    relief (relief ~ 1/g) is kept on the habitat. The scale is calibrated on Earth: an Earth ocean
    (EARTH_OCEAN_MASS_FRACTION x M_EARTH on R_EARTH with g = G M/R^2) fills EARTH_FILL of the
    habitat's relief shell, which covers Earth's ocean fraction on average on this terrain.
    Arguments are numbers or tensors [W].
    """
    f64 = torch.float64
    mass, rp, g, relief, radius = (torch.as_tensor(v, dtype=f64) for v in
                                   (ocean_mass_kg, planet_radius_m, gravity_m_s2, relief_m, radius_m))
    earth_g = G * M_EARTH / R_EARTH ** 2
    ratio = mass / (EARTH_OCEAN_MASS_FRACTION * M_EARTH) * (R_EARTH / rp) ** 2 * (g / earth_g)
    return ratio * EARTH_FILL * 4 * math.pi / 3 * ((radius + relief) ** 3 - radius ** 3)


# ------------------------------------------------------------------------------- deposits
def ore_patches(globe, W, gen, where, count=ORE_PATCHES, radius_rad=ORE_RADIUS_RAD):
    """Sum of Gaussian caps exp(-chord^2 / r^2) [W, C] centred on random cells drawn by area within where [W, C].

    Each cap is scaled so that its sampled integral sum(area x cap) is pi r^2 (its continuum value),
    so a patch carries the same mass at every G even when r is below the cell size; where the cap
    is resolved its peak is about 1.
    """
    dev = globe.device
    chance = where.double() * globe.area64 + 1e-12 * globe.area64     # a world without land still draws
    cell = torch.multinomial(chance, count, replacement=True, generator=gen)
    r = radius_rad[0] + (radius_rad[1] - radius_rad[0]) * torch.rand(W, count, generator=gen, device=dev)
    chord2 = 2 * (1 - torch.einsum("cx,wkx->wck", globe.centers, globe.centers[cell]))
    cap = torch.exp(-chord2 / r[:, None] ** 2)                                        # [W, C, K]
    mass = (cap.double() * globe.area64[:, None]).sum(1)                               # [W, K]
    return (cap * (math.pi * r.double() ** 2 / mass).float()[:, None]).sum(-1)


def _fill(weight, inside, share, area64, iters=FILL_ITERS, tol=FILL_TOL):
    """Split a column of sum(share) in every inside cell among the species (float64).

    weight [W, C, S] >= 0, inside [W, C] bool, share [W, S] >= 0. The fill of cell c is
    sum(share) x weight[c, s] k_s / sum_s' weight[c, s'] k_s', with factors k_s found by matrix
    scaling (Sinkhorn & Knopp 1967, Pacific J. Math. 21, 343) so that the area mean over the
    inside cells of species s is share[s]. Returns (fill [W, C, S], error [W]): the largest
    relative error of a species' mean.
    """
    tiny = 1e-300
    w = weight.double() * inside[..., None]
    a = (area64 * inside)[..., None]                                    # [W, C, 1]
    total = a.sum(1).clamp_min(tiny)                                    # [W, 1]
    share = share.double()
    column = share.sum(-1, keepdim=True)                                # [W, 1]
    live = (share > 0) & (a.sum(1) > 0)
    mean_w = (w * a).sum(1) / total
    k = torch.where(live & (mean_w > 0), share / mean_w.clamp_min(tiny), 0.0)
    for it in range(iters + 1):
        d = (w * k[:, None]).sum(-1, keepdim=True)                      # [W, C, 1]
        inv = torch.where(d > 0, 1 / d.clamp_min(tiny), 0.0)            # a cell with no weight stays empty
        level = column * (w * inv * a).sum(1) / total                   # mean of species s per unit k_s
        err = torch.where(live, (k * level / share.clamp_min(tiny) - 1).abs(), 0.0).amax(-1)
        if it == iters or (it % 8 == 0 and float(err.max()) < tol):
            break
        k = torch.where(live & (level > 0), share / level.clamp_min(tiny), 0.0)
    return column[..., None] * w * k[:, None] * inv, err


def make_deposits(globe, W, gen, terrain, crust, species: Sequence[str],
                  accessible_kg_m2=ROCK_DENSITY * ACCESSIBLE_DEPTH_M, clay_start=CLAY_START, report=None):
    """Accessible ground stock (kg/m^2, float32 [W, C, S]) of each crust species.

    crust [W, S] holds mass fractions in the order of ``species`` (a continental mix). Every
    cell holds a column of accessible_kg_m2 x sum(crust) of ground (1 m of rock), split among the
    species (``_fill``):
    - Land holds the crust mix: the mean over the land area of species s is
      accessible_kg_m2 x crust[s]. Its share in a cell follows LAND_WEIGHTS of its kind
      (DEPOSIT_KIND): felsic rock on high ground, mafic and sediments low, ores and flint in
      patches on land (copper minerals share theirs, tin follows granite), sand along the coast
      (SAND_SCALE_RAD) and salt in low ground at least SALT_MIN_DIST_RAD inland (a placeholder
      for dry basins until the climate gives aridity); times log-normal patchiness.
    - The sea floor holds only the marine species (basalt, limestone, clay, sand) in their crust
      proportions renormalised among themselves (MARINE_WEIGHTS: sand and carbonate on the
      shelf, basalt and clay in the deep); it is not part of the land inventory.
    Clay starts at clay_start of its share everywhere (the column lacks the clay still to form).
    Rules are written in rad, so the pattern does not depend on G. A species whose own area
    covers less than SUPPORT_MARGIN x its share of the land widens to low land, then to all land;
    on a world without land the land species are absent and a RuntimeWarning is issued.
    Whether a cell is reachable is the caller's rule.

    report, if a dict, receives support_level [W, S] (0 own area, 1 low land, 2 all land,
    3 absent: no land; -1 for a species with no crust share), land_error and ocean_error [W]
    (the largest relative error of a species' inventory, about FILL_TOL).
    """
    dev = globe.device
    crust = torch.as_tensor(crust, dtype=torch.float64, device=dev).reshape(W, -1)
    if crust.shape[1] != len(species):
        raise ValueError("crust has one column per species")
    if bool((crust < 0).any()):
        raise ValueError("crust fractions must not be negative")
    land_b = terrain["land"]
    land = land_b.float()
    ocean = 1 - land
    elev, sea = terrain["elevation_m"], terrain["sea_level_m"][:, None]
    top = elev.amax(-1, keepdim=True)
    up = ((elev - sea) / (top - sea).clamp_min(1e-6)).clamp(0, 1) * land
    relief = (top - elev.amin(-1, keepdim=True)).clamp_min(1e-6)
    shelf = ocean * torch.exp(-terrain["depth_m"] / (SHELF_SCALE * relief))
    inland = terrain.get("inland_rad")
    if inland is None:
        inland = inland_distance(globe, land_b)
    # the cell spans [lo, lo + h] of distance inland: the coastal band and the inland share are cell averages
    h = globe.spacing
    lo = (inland - 0.5 * h).clamp_min(0)
    factors = {"coast": SAND_SCALE_RAD / h * (torch.exp(-lo / SAND_SCALE_RAD) - torch.exp(-(lo + h) / SAND_SCALE_RAD))
               + SAND_FLOOR,
               "inland": ((lo + h - SALT_MIN_DIST_RAD) / h).clamp(0, 1)}
    patches = {}
    own, noises, sea_w, kinds = [], [], [], []
    for name in species:
        kind = DEPOSIT_KIND.get(name, "rock")
        base, slope, power, factor = LAND_WEIGHTS[kind]
        w = (base + slope * up).clamp_min(0) ** power
        if factor == "patch":
            if kind not in patches:
                patches[kind] = ore_patches(globe, W, gen, land_b)
            w = w * patches[kind]
        elif factor is not None:
            w = w * factors[factor]
        noise = torch.exp(DEPOSIT_NOISE * plane_waves(globe, W, gen, count=NOISE_WAVES, k_range=NOISE_K))
        own.append(w * noise * land)
        noises.append(noise * land)
        deep, on_shelf = MARINE_WEIGHTS.get(kind, (0.0, 0.0))
        sea_w.append((deep + (on_shelf - deep) * shelf) * noise * ocean)
        kinds.append(kind)
    own, anywhere, sea_w = (torch.stack(x, -1) for x in (own, noises, sea_w))   # [W, C, S]
    low = ((1 - up) ** 3)[..., None] * anywhere

    # land: own area, else low land, else all land
    a = (globe.area64 * land_b)[..., None]
    land_area = a.sum(1)                                                       # [W, 1]
    need = (SUPPORT_MARGIN * crust / crust.sum(-1, keepdim=True).clamp_min(1e-300)).clamp_max(1.0) - 1e-12

    def covers(w):
        return ((w > 0) * a).sum(1) >= need * land_area

    ok_own, ok_low = covers(own), covers(low)
    level = torch.where(ok_own, 0, torch.where(ok_low, 1, 2))
    level = torch.where(land_area > 0, level, 3)
    weight = torch.where((level == 0)[:, None], own, torch.where((level == 1)[:, None], low, anywhere))
    fill_land, land_err = _fill(weight, land_b, crust, globe.area64)

    # sea floor: the marine species in their crust proportions
    marine = torch.tensor([k in MARINE_WEIGHTS for k in kinds], device=dev)
    sea_share = crust * marine
    msum = sea_share.sum(-1, keepdim=True)
    sea_share = torch.where(msum > 0, sea_share * crust.sum(-1, keepdim=True) / msum.clamp_min(1e-300), 0.0)
    fill_sea, sea_err = _fill(sea_w, ~land_b, sea_share, globe.area64)

    stock = accessible_kg_m2 * (fill_land + fill_sea)
    clay = torch.tensor([k == "clay" for k in kinds], device=dev)
    stock = torch.where(clay, stock * clay_start, stock).float()

    level = torch.where(crust > 0, level, -1)
    absent = (level == 3).any(-1)
    if bool(absent.any()):
        warnings.warn(f"worlds {absent.nonzero().flatten().tolist()} have no land: their land deposits are absent",
                      RuntimeWarning, stacklevel=2)
    worst = float(torch.cat((land_err, sea_err)).max())
    if worst > 1e3 * FILL_TOL:
        warnings.warn(f"deposit inventories match the crust mix only to {worst:.2e}", RuntimeWarning, stacklevel=2)
    if report is not None:
        report.update(support_level=level, land_error=land_err, ocean_error=sea_err)
    return stock
