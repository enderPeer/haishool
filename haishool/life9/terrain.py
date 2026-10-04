"""Periodic height-field terrain, height lookup and line of sight (batched over worlds).

The ground is a height field on a torus: x and y wrap, z is the height. Every body stands
on the ground, so z is physics here, not decoration: climbing costs energy, high ground
sees further, and hills block sight (fully) and sound (partly).
"""
from __future__ import annotations

import math

import torch


def make_terrain(gen, worlds, grid, relief, octaves, device):
    """Heights in [0, relief], shape [W, G, G], periodic (a sum of whole-wave sinusoids)."""
    axis = torch.arange(grid, device=device, dtype=torch.float32) / grid * 2 * math.pi
    x, y = torch.meshgrid(axis, axis, indexing="ij")
    heights = torch.zeros(worlds, grid, grid, device=device)
    for octave in range(1, octaves + 1):
        for _ in range(2):
            kx = torch.randint(-octave, octave + 1, (worlds, 1, 1), generator=gen, device=device).float()
            ky = torch.randint(-octave, octave + 1, (worlds, 1, 1), generator=gen, device=device).float()
            ky = torch.where((kx == 0) & (ky == 0), torch.ones_like(ky), ky)
            phase = torch.rand(worlds, 1, 1, generator=gen, device=device) * 2 * math.pi
            amplitude = torch.rand(worlds, 1, 1, generator=gen, device=device) / octave ** 1.2
            heights = heights + amplitude * torch.sin(kx * x + ky * y + phase)
    low = heights.amin(dim=(1, 2), keepdim=True)
    high = heights.amax(dim=(1, 2), keepdim=True)
    return (heights - low) / (high - low).clamp(min=1e-6) * relief


def height_at(terrain, points, size):
    """Bilinear ground height at points [W, M, 2] (world units, wrapped) -> [W, M]."""
    worlds, grid, _ = terrain.shape
    flat = terrain.reshape(worlds, grid * grid)
    u = torch.remainder(points / size * grid, grid)
    i0 = torch.floor(u).long()
    f = u - i0.float()
    i1 = (i0 + 1) % grid
    i0 = i0 % grid

    def at(ix, iy):
        return flat.gather(1, ix * grid + iy)

    fx, fy = f[..., 0], f[..., 1]
    return (at(i0[..., 0], i0[..., 1]) * (1 - fx) * (1 - fy) + at(i1[..., 0], i0[..., 1]) * fx * (1 - fy)
            + at(i0[..., 0], i1[..., 1]) * (1 - fx) * fy + at(i1[..., 0], i1[..., 1]) * fx * fy)


def line_of_sight(terrain, size, a_xy, a_z, delta, b_z, samples):
    """True where the straight segment from a to a + delta stays above the ground.

    a_xy [W, N, 2], a_z [W, N], delta [W, N, M, 2] (already wrapped), b_z broadcastable
    to [W, N, M]. The sample points (k+1)/(samples+1) are symmetric, so the test from a to b
    equals the test from b to a.
    """
    worlds, n, m, _ = delta.shape
    blocked = torch.zeros(worlds, n, m, dtype=torch.bool, device=delta.device)
    for k in range(samples):
        t = (k + 1) / (samples + 1)
        points = a_xy[:, :, None, :] + t * delta
        ground = height_at(terrain, points.reshape(worlds, n * m, 2), size).reshape(worlds, n, m)
        blocked |= ground > a_z[:, :, None] * (1 - t) + b_z * t
    return ~blocked
