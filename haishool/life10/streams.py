"""Independent deterministic sparse-event streams for physical planets."""
from __future__ import annotations

import hashlib
import random


def planet_streams(batch, seeds, kind):
    """Accept each planet's supplied chemistry seed, independent of batch order.

    A scalar preserves the single-planet component API and its historical draws.
    Multi-planet callers must supply one physical seed per planet. List inputs
    use a stream-specific salt so assemblies and compartments are independent.
    """
    if isinstance(seeds, int) and not isinstance(seeds, bool):
        if batch != 1:
            raise ValueError("Multi-planet sparse dynamics require one seed per planet")
        values = [seeds]
    else:
        values = list(seeds)
        if len(values) != batch:
            raise ValueError("Supply exactly one chemistry seed per planet")
        if any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in values):
            raise ValueError("Planet seeds must be nonnegative integers")
        values = [int.from_bytes(hashlib.sha256(f"life10:{kind}:{s}".encode()).digest()[:8], "little")
                  for s in values]
    return values, [random.Random(s) for s in values]


def tuples(value):
    return tuple(tuples(x) for x in value) if isinstance(value, (tuple, list)) else value
