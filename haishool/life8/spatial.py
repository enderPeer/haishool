"""Uniform-grid candidate index for local queries on the toroidal world.

The index only *narrows the candidate set*. Callers still compute the exact same
toroidal distance with the same function, filter by the same radius and sort by
(distance, id), so a query returns exactly what a full scan returns as long as no
entity moves while the index is in use. The world builds an index immediately
before a read-only pass (all decisions, all next observations, message delivery)
and drops it afterwards; it is never serialized and never changes results.
"""
from __future__ import annotations

import math

# Positions and toroidal distances carry rounding errors of order 1e-14 world
# units. Widening every query by MARGIN (far above that, far below a cell) means
# an entity whose computed distance is within the radius always lies in a
# visited cell: a cell boundary just outside the widened window is more than
# MARGIN beyond the radius, so anything bucketed beyond it is out of range anyway.
MARGIN = 1e-6


class GridIndex:
    """Bucket entities by cell; return a superset of those within a radius."""

    def __init__(self, width, height, cell, tables):
        self.width, self.height = float(width), float(height)
        self.nx = max(1, int(self.width // cell))
        self.ny = max(1, int(self.height // cell))
        self.cw, self.ch = self.width / self.nx, self.height / self.ny
        self._tables = {}
        for records in tables:
            cells = {}
            for item in records.values():
                cells.setdefault(self._cell(item.position), []).append(item)
            self._tables[id(records)] = (records, cells)

    def _cell(self, position):
        return int(position[0] / self.cw) % self.nx, int(position[1] / self.ch) % self.ny

    def covers(self, records):
        entry = self._tables.get(id(records))
        return entry is not None and entry[0] is records

    def candidates(self, records, position, radius):
        """Entities whose cell lies within the radius plus a one-cell safety margin."""
        cells = self._tables[id(records)][1]
        reach = radius + MARGIN
        x0, x1 = math.floor((position[0] - reach) / self.cw), math.floor((position[0] + reach) / self.cw)
        y0, y1 = math.floor((position[1] - reach) / self.ch), math.floor((position[1] + reach) / self.ch)
        columns = range(self.nx) if x1 - x0 + 1 >= self.nx else sorted({x % self.nx for x in range(x0, x1 + 1)})
        rows = range(self.ny) if y1 - y0 + 1 >= self.ny else sorted({y % self.ny for y in range(y0, y1 + 1)})
        found = []
        for x in columns:
            for y in rows:
                bucket = cells.get((x, y))
                if bucket:
                    found.extend(bucket)
        return found
