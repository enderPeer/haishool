"""Viewer of life9 v3 runs: one self-contained HTML page with two linked views, the planet at real size and a
habitat patch at true scale.

    build_page(static3, frames, out_path, max_mb=15) -> report
    python -m haishool.life9.v3.view3 STATIC.json FRAMES.jsonl --out PAGE.html [--max-mb 15] [--every 1]

Input format "life9-v3-view-1" (the world module produces it; this module only reads it and imports no world
code):

    static3 = {"schema": "life9-v3-view-1", "seed": int,
               "planet": {radius_m, gravity_m_s2, surface_pressure_pa, tidally_locked, year_s, star_teff_k,
                          t_surface_k, ocean_fraction, frozen_mean, ...},   # formation3.summary_row subset
               "globe": {"G": int, "centers": [[x, y, z]] * C (4 dp), "elevation_m": [C ints],
                         "land": [C 0/1], "sea_level_m": float},
               "patches": [{"index": p, "cell": int, "center": [x, y, z], "L": m, "n": cells,
                            "elevation_m": [n*n], "pond": [n*n 0/1]}],
               "species": [names],
               optional "item_classes": [names]   # what an item's class_index indexes (default materials.CLASSES)
               optional "gene_columns": [names]   # names of extra body columns after the twelfth
               optional "bouts": int              # bouts per day (shown only)
               optional "scales": {"plants": [value_at_255, unit], "water": [value_at_255, unit]}}
    frame3  = {"day": int, "bout": int, "sun": [x, y, z] (unit vector toward the star, planet coordinates),
               optional "global": {"T_k": [C ints], "ice": [C 0/1]},
               "patches": [{"index": p,
                            "bodies": [[uid, x, y, heading, mass_g, lineage, k_active, mouth, thrust, loud,
                                        vocal_argmax, damage(, gene values ...)]],
                            optional "plants": [n*n 0-255], optional "water": [n*n 0-255],
                            "items": [[x, y, class_index, mass_kg]] (at most 1,500),
                            "fires": [[x, y, temp_k]]}]}

Conventions (patch.py and body.py): positions are metres (x east, y north) on the patch's periodic square
[0, L); fine fields are flat with cell = ix n + iy; ``heading`` is radians from +x (east) toward +y (north);
``mouth``, ``loud`` and ``damage`` lie in [0, 1], ``thrust`` in [-1, 1] (brain3.decode). The patch's local
frame on the globe is (east, north, up) of its centre (globe.tangent_basis: east = (-sin lon, cos lon, 0));
a patch may carry its own "east" and "north" vectors to override it.

Encoding. Numbers are little-endian typed arrays in base64, compressed with zlib (deflate) when that is
smaller: {"t": type, "n": count, "b": base64(, "z": 1 deflated, "d": 1 delta coded, "s": scale, "o": offset,
"lg": 1 log10)}; value = stored (summed when delta coded) / s + o, then 10 ** value when "lg". Positions are
stored in steps of 1 / floor(65535 / L) m (3.2 cm on a 2,048 m patch), headings in 1/40 rad, masses as log10 in
steps of 1/4000 (0.06 %), mouth, loudness and damage in 1/250, thrust in 1/125. Bodies are sorted by uid inside a
patch (uids delta coded); lineages are a sorted id list plus an index per body. The page carries its own small
inflater (no library), so it still loads only the two pinned scripts in ``ALLOWED_SCRIPTS``.

Size bound: while the page would exceed ``max_mb`` the builder thins the larger part, fields or frames. Fields
(the global T_k and ice, the patches' plants and water): one record in FK is kept (the first and the last) down to
``MIN_FIELD_RECORDS``, then the patch fields are averaged over s x s fine cells (s = 2, 4), then thinned in time
again. Frames: one in K is kept (the first and the last always) down to ``MIN_FRAMES``, then fewer items are drawn
per patch (an even spread, down to ``ITEMS_FLOOR``; only while items are at least ``ITEMS_SHARE`` of the frames'
bytes), then frames thin to two, then items are dropped. Bodies are never dropped. The report and the page state what was thinned; if nothing more can be thinned the page is still
written and the report says ``fits: False``.

The report's ``nonfinite_values`` (non-finite numbers stored as 0, and non-positive masses), ``wrapped_positions``
(positions outside [0, L) wrapped onto the patch), ``duplicate_uids`` and ``item_class_out_of_range`` count what the
written page holds (the static part, the kept frames and field records).

The viewer shows recorded data and adds no behaviour. Its display choices (enlarged body markers, ring size from
loudness, colours of lineages, call channels and item classes, vertical exaggeration, interpolation between
frames) are display only. Standard library only, apart from the cube-sphere grid check (``planet.view``) and the
default item class names (``planet.materials``).
"""
from __future__ import annotations

import argparse
import array
import base64
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import zlib

TEMPLATE = Path(__file__).resolve().parent / "viewer3.html"
DATA_TOKEN = "__LIFE9_V3_VIEW_DATA__"
SCHEMA = "life9-v3-view-1"
PAGE_SCHEMA = "life9-v3-view-page-1"
ALLOWED_SCRIPTS = ("https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js",
                   "https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js")
MAX_MB = 15.0
ITEMS_CAP = 1500            # items per patch per frame (the frame format's own cap)
ITEMS_FLOOR = 200           # items per patch kept until the frames are thinned to two
MIN_FRAMES = 48             # frames kept before fewer items are drawn
ITEMS_SHARE = 0.2           # items are thinned before frames only while they are at least this share of the frames
MIN_FIELD_RECORDS = 8       # field records per list kept before the patch fields are averaged in space
MAX_FIELD_FACTOR = 4        # coarsest block of the patch fields (s x s fine cells)
MIN_FIELD_SIDE = 8          # a coarsened patch field keeps at least this many cells per side
UNIT_SCALE = 1e4            # unit-vector components in 1e-4 steps
HEADING_SCALE = 40.0        # headings in 1/40 rad (1.4 degrees)
FRACTION_SCALE = 250.0      # mouth, loudness, damage in 1/250
SIGNED_SCALE = 125.0        # thrust in 1/125
LOG_SCALE = 4000.0          # log10 of masses in 1/4000 (0.058 % per step)
MASS_G_LOG_OFFSET = -6.0    # body masses from 1 ug up (u2 reaches 2.4e10 g)
MASS_KG_LOG_OFFSET = -9.0   # item masses from 1 ug up (u2 reaches 2.4e7 kg)
ELEVATION_SCALE = 100.0     # fine elevations in cm when they are not whole metres
TIME_BOUTS = 4096           # frame order key = day x TIME_BOUTS + bout (bout must be below it)
TOP_LINEAGES = 8            # lineages that get their own colour on the page
BODY_COLUMNS = ("uid", "x", "y", "heading", "mass_g", "lineage", "k_active", "mouth", "thrust", "loud",
                "vocal_argmax", "damage")
ITEM_COLUMNS = ("x", "y", "class_index", "mass_kg")
FIRE_COLUMNS = ("x", "y", "temp_k")
PLANET_KEYS = ("radius_m", "gravity_m_s2", "surface_pressure_pa", "tidally_locked", "year_s", "star_teff_k",
               "t_surface_k", "ocean_fraction", "frozen_mean")
GLOBAL_FIELDS = ("T_k", "ice")
PATCH_FIELDS = ("plants", "water")
DEFAULT_ITEM_CLASSES = ("stone", "ore", "clay", "wood", "organic", "metal", "product")  # materials.CLASSES

# typecode in the page, array module code, smallest and largest value
_INT_TYPES = (("i1", "b", -2 ** 7, 2 ** 7 - 1), ("u1", "B", 0, 2 ** 8 - 1), ("i2", "h", -2 ** 15, 2 ** 15 - 1),
              ("u2", "H", 0, 2 ** 16 - 1), ("i4", "i", -2 ** 31, 2 ** 31 - 1))
_ARRAY_CODE = {name: code for name, code, _, _ in _INT_TYPES} | {"f4": "f", "f8": "d"}
assert array.array("i").itemsize == 4 and array.array("f").itemsize == 4 and array.array("h").itemsize == 2

NOTES = {
    "supplied": ("Nothing is pre-installed (PLANET-V3-SPEC section 0). The bodies begin with random brains and random "
                 "body genes. Nothing tells them what to eat or drink, where to go, what a sound means or what is "
                 "good; no score is handed out, and they have no built-in wiring and no named purposeful moves. "
                 "Supplied are only physics and chemistry, the planet's chain, replication with heritable random "
                 "variation, death by physics, and a body's physical abilities."),
    "display": ("Display only: the colours of lineages, call channels and item classes are chosen by the viewer and "
                "are not known to the bodies. Bodies and items are drawn enlarged by the stated factor unless "
                "'True size' is on; ring size follows loudness; vertical exaggeration is labelled; motion between "
                "recorded bouts is drawn straight when 'Smooth' is on."),
    "meaning": ("No meaning is supplied. A call is an 8-number vector the brain emits; the ring colour is only the "
                "index of its largest component. Mouth, thrust and loudness are motor output intensities; nothing "
                "names what they are for."),
    "scale": ("The globe is drawn at the planet's real radius with true relief unless an exaggeration is chosen. A "
              "patch is a true-scale square of side L at one land cell, with a periodic boundary: leaving one side "
              "enters the opposite side (PLANET-V3-SPEC section 1)."),
}
PROVENANCE = {
    "body_enlargement": ("display", "bodies and items drawn at their true-size radius times a stated factor"),
    "item_density": ("display", "items drawn as spheres of 2,000 kg/m^3 of their mass (composition not recorded)"),
    "body_density": ("display", "bodies drawn as ellipsoids of 1,050 kg/m^3 tissue (PLANET-V3-SPEC section 3)"),
    "vertical_exaggeration": ("display", "patch heights drawn x1 (true) unless another factor is chosen"),
    "field_factor": ("display", "patch fields averaged over s x s fine cells to fit the page"),
}


class FrameError(ValueError):
    """The static or frame data do not follow the life9-v3-view-1 format."""


# ------------------------------------------------------------------------------------------ codec
def _finite(value, counter):
    try:
        value = float(value)
    except (TypeError, ValueError):
        counter[0] += 1
        return 0.0
    if math.isfinite(value):
        return value
    counter[0] += 1
    return 0.0


def _bytes(arr: array.array) -> bytes:
    if sys.byteorder != "little":
        arr = array.array(arr.typecode, arr)
        arr.byteswap()
    return arr.tobytes()


def _pack(raw: bytes, out: dict, compress: bool) -> dict:
    if compress and len(raw) >= 48:
        packed = zlib.compress(raw, 9)
        if len(packed) + 8 < len(raw):
            out["z"] = 1
            raw = packed
    out["b"] = base64.b64encode(raw).decode("ascii")
    return out


def encode_column(values, *, scale: float | None = None, offset: float = 0.0, kind: str | None = None,
                  delta: bool = False, log: bool = False, bad=None, compress: bool = True) -> dict:
    """Pack numbers as the smallest typed array that holds them: {"t", "n", "b"(, "z", "d", "s", "o", "lg")}.

    kind "bit" packs 0/1 values 8 per byte; kind "f4"/"f8" stores floats; otherwise values (log10 of them when
    ``log``, minus offset, times scale when given) are rounded to integers, then first differences when ``delta``.
    ``bad`` is a one-element list that counts non-finite inputs (stored as 0) and, with ``log``, non-positive
    ones (stored at the offset). The bytes are deflated when that is smaller."""
    bad = [0] if bad is None else bad
    values = list(values)
    n = len(values)
    if kind == "bit":
        packed = bytearray((n + 7) // 8)
        for i, v in enumerate(values):
            if _finite(v, bad) >= 0.5:
                packed[i >> 3] |= 1 << (i & 7)
        return _pack(bytes(packed), {"t": "bit", "n": n}, compress)
    if kind in ("f4", "f8"):
        return _pack(_bytes(array.array(_ARRAY_CODE[kind], [_finite(v, bad) for v in values])),
                     {"t": kind, "n": n}, compress)
    xs = []
    for v in values:
        before = bad[0]
        x = _finite(v, bad)
        if log:
            if x > 0:
                x = math.log10(x)
            else:
                bad[0] += bad[0] == before  # a non-positive value (a non-finite one is counted already)
                x = offset
        xs.append(x)
    if scale is None:
        ints = [int(round(v - offset)) for v in xs]
    else:
        ints = [int(round((v - offset) * scale)) for v in xs]
    if delta and ints:
        ints = [ints[0]] + [b - a for a, b in zip(ints, ints[1:])]
    lo, hi = (min(ints), max(ints)) if ints else (0, 0)
    for name, code, low, high in _INT_TYPES:
        if low <= lo and hi <= high:
            out, raw = {"t": name, "n": n}, _bytes(array.array(code, ints))
            break
    else:  # beyond int32: exact up to 2^53 in float64
        out, raw = {"t": "f8", "n": n}, _bytes(array.array("d", [float(v) for v in ints]))
    if scale is not None:
        out["s"] = scale
    if offset:
        out["o"] = offset
    if delta:
        out["d"] = 1
    if log:
        out["lg"] = 1
    return _pack(raw, out, compress)


def decode_column(col: dict | None) -> list | None:
    """Inverse of :func:`encode_column` (for tests and analysis)."""
    if col is None:
        return None
    raw = base64.b64decode(col["b"])
    if col.get("z"):
        raw = zlib.decompress(raw)
    if col["t"] == "bit":
        return [(raw[i >> 3] >> (i & 7)) & 1 for i in range(col["n"])]
    arr = array.array(_ARRAY_CODE[col["t"]])
    arr.frombytes(raw)
    if sys.byteorder != "little":
        arr.byteswap()
    values = list(arr)
    if col.get("d"):
        acc, summed = 0, []
        for v in values:
            acc += v
            summed.append(acc)
        values = summed
    if "s" in col or "o" in col or col.get("lg"):
        scale, offset = col.get("s", 1.0), col.get("o", 0.0)
        values = [v / scale + offset for v in values]
        if col.get("lg"):
            values = [10.0 ** v for v in values]
    return values


def _json(obj) -> str:
    return json.dumps(obj, allow_nan=False, separators=(",", ":"), ensure_ascii=True)


def _escape(text: str) -> str:
    return text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


# ------------------------------------------------------------------------------------- the grid
def grid_kind(G: int, centers) -> str:
    """'cube-sphere' if the centres are the life9 grid in its cell order (``planet.view.grid_kind``), else 'points'."""
    try:
        from ..planet.view import grid_kind as v2_grid_kind
    except Exception:  # pragma: no cover - the viewer must not depend on the engine
        return "points"
    return v2_grid_kind(G, centers)


def tangent_basis(center) -> tuple[list[float], list[float], list[float]]:
    """(east, north, up) unit vectors at a point: east = (-sin lon, cos lon, 0) as globe.tangent_basis."""
    x, y, z = (float(v) for v in center)
    norm = math.sqrt(x * x + y * y + z * z) or 1.0
    up = [x / norm, y / norm, z / norm]
    h = math.hypot(up[0], up[1])
    east = [-up[1] / h, up[0] / h, 0.0] if h > 1e-12 else [0.0, 1.0, 0.0]
    north = [up[1] * east[2] - up[2] * east[1], up[2] * east[0] - up[0] * east[2], up[0] * east[1] - up[1] * east[0]]
    return east, north, up


def position_scale(L: float) -> float:
    """Steps per metre of stored positions: floor(65535 / L), so [0, L] fits two bytes."""
    return float(math.floor(65535.0 / L)) if L <= 65535.0 else 65535.0 / L


def coarsen_field(values, n: int, factor: int) -> list[int]:
    """Block means (rounded) of a flat n x n field (cell = ix n + iy) over factor x factor cells."""
    if factor == 1:
        return [int(round(v)) for v in values]
    m = n // factor
    sums = [0.0] * (m * m)
    for ix in range(n):
        row, base = (ix // factor) * m, ix * n
        for iy in range(n):
            sums[row + iy // factor] += values[base + iy]
    k = float(factor * factor)
    return [int(round(s / k)) for s in sums]


# ------------------------------------------------------------------------------------ validation
def _need(obj, key, where):
    if not isinstance(obj, dict) or key not in obj:
        raise FrameError(f"{where} is missing {key!r}")
    return obj[key]


def _check_static(static: dict) -> dict:
    if not isinstance(static, dict) or static.get("schema") != SCHEMA:
        raise FrameError(f"static['schema'] must be {SCHEMA!r}")
    globe = _need(static, "globe", "static")
    for key in ("G", "centers", "elevation_m", "land", "sea_level_m"):
        _need(globe, key, "static['globe']")
    C = len(globe["centers"])
    if C == 0:
        raise FrameError("static['globe']['centers'] is empty")
    for key in ("elevation_m", "land"):
        if len(globe[key]) != C:
            raise FrameError(f"static['globe'][{key!r}] has {len(globe[key])} entries, centers has {C}")
    if any(len(p) != 3 for p in globe["centers"]):
        raise FrameError("every globe centre must be [x, y, z]")
    planet = static.get("planet") or {}
    radius = planet.get("radius_m")
    if radius is None or not float(radius) > 0:
        raise FrameError("static['planet']['radius_m'] must be a positive number")
    patches = {}
    for k, ps in enumerate(_need(static, "patches", "static")):
        where = f"static['patches'][{k}]"
        for key in ("index", "center", "L", "n", "elevation_m"):
            _need(ps, key, where)
        n, L = int(ps["n"]), float(ps["L"])
        if n < 2 or not L > 0:
            raise FrameError(f"{where}: n must be at least 2 and L positive")
        if len(ps["elevation_m"]) != n * n:
            raise FrameError(f"{where}: elevation_m has {len(ps['elevation_m'])} cells, expected n*n = {n * n}")
        if ps.get("pond") is not None and len(ps["pond"]) != n * n:
            raise FrameError(f"{where}: pond has {len(ps['pond'])} cells, expected n*n = {n * n}")
        if len(ps["center"]) != 3:
            raise FrameError(f"{where}: center must be [x, y, z]")
        for key in ("east", "north"):
            if ps.get(key) is not None and len(ps[key]) != 3:
                raise FrameError(f"{where}: {key} must be [x, y, z]")
        index = int(ps["index"])
        if index in patches:
            raise FrameError(f"{where}: patch index {index} appears twice")
        patches[index] = {"slot": k, "n": n, "L": L}
    return {"C": C, "G": int(globe["G"]), "patches": patches}


def _check_frame(frame: dict, i: int, shape: dict, n_genes: int):
    if not isinstance(frame, dict) or "day" not in frame:
        raise FrameError(f"frame {i} has no 'day'")
    bout = int(frame.get("bout", 0))
    if not 0 <= bout < TIME_BOUTS:
        raise FrameError(f"frame {i}: bout must lie in [0, {TIME_BOUTS})")
    sun = frame.get("sun")
    if sun is not None and len(sun) != 3:
        raise FrameError(f"frame {i}: sun must be [x, y, z]")
    for key, values in (frame.get("global") or {}).items():
        if key in GLOBAL_FIELDS and values is not None and len(values) != shape["C"]:
            raise FrameError(f"frame {i}: global {key!r} has {len(values)} cells, expected {shape['C']}")
    seen = set()
    for pf in frame.get("patches") or ():
        index = int(_need(pf, "index", f"frame {i} patch entry"))
        if index not in shape["patches"]:
            raise FrameError(f"frame {i}: patch {index} is not in static['patches']")
        if index in seen:
            raise FrameError(f"frame {i}: patch {index} appears twice")
        seen.add(index)
        n = shape["patches"][index]["n"]
        for key, width in (("bodies", len(BODY_COLUMNS)), ("items", len(ITEM_COLUMNS)), ("fires", len(FIRE_COLUMNS))):
            for row in pf.get(key) or ():
                if len(row) < width:
                    raise FrameError(f"frame {i} (day {frame['day']}): a {key} row of patch {index} has {len(row)} "
                                     f"values, needs {width}")
        for key in PATCH_FIELDS:
            if pf.get(key) is not None and len(pf[key]) != n * n:
                raise FrameError(f"frame {i}: patch {index} {key!r} has {len(pf[key])} cells, expected {n * n}")


# --------------------------------------------------------------------------------------- encoding
def _unit_rows(rows, bad):
    out = []
    for row in rows:
        x, y, z = (_finite(v, bad) for v in row[:3])
        norm = math.sqrt(x * x + y * y + z * z)
        if norm > 0:
            x, y, z = x / norm, y / norm, z / norm
        out += (x, y, z)
    return out


def _wrap_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _wrap_pos(values, L, counts):
    out = []
    for v in values:
        if v < 0 or v >= L:
            counts["wrapped_positions"] += 1
            v = v % L
        out.append(v)
    return out


def _item_classes(static):
    if static.get("item_classes"):
        return [str(c) for c in static["item_classes"]]
    try:
        from ..planet.materials import CLASSES
        return list(CLASSES)
    except Exception:  # pragma: no cover - the viewer must not depend on the engine tables
        return list(DEFAULT_ITEM_CLASSES)


def _scalar(value):
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return value if math.isfinite(value) else None
    if isinstance(value, dict) and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value.values()):
        return {str(k): (v if math.isfinite(v) else None) for k, v in value.items()}
    return None


def _encode_static(static, grid, item_classes, bad):
    planet = {}
    for key, value in (static.get("planet") or {}).items():
        v = _scalar(value)
        if v is not None or value is None:
            planet[key] = v
    globe = static["globe"]
    patches = []
    for ps in static["patches"]:
        n, L = int(ps["n"]), float(ps["L"])
        elev = [_finite(v, bad) for v in ps["elevation_m"]]
        whole = all(float(v).is_integer() for v in elev)
        east, north, up = tangent_basis(ps["center"])
        if ps.get("east") is not None:
            east = _unit_rows([ps["east"]], bad)
        if ps.get("north") is not None:
            north = _unit_rows([ps["north"]], bad)
        patches.append({
            "index": int(ps["index"]), "cell": ps.get("cell"), "L": L, "n": n, "ps": position_scale(L),
            "center": [round(v, 6) for v in up], "east": [round(v, 6) for v in east],
            "north": [round(v, 6) for v in north],
            "elevation_m": encode_column(elev, scale=None if whole else ELEVATION_SCALE, delta=True, bad=bad),
            "z_min": min(elev), "z_max": max(elev), "z_mean": sum(elev) / len(elev),
            "pond": encode_column(ps.get("pond") or [0] * (n * n), kind="bit", bad=bad)})
    scales = {}
    for key, value in (static.get("scales") or {}).items():
        if key in PATCH_FIELDS and isinstance(value, (list, tuple)) and len(value) == 2:
            scales[key] = [_scalar(value[0]), str(value[1])]
    return {
        "source_schema": static["schema"], "seed": static.get("seed"), "planet": planet,
        "species": [str(s) for s in static.get("species") or []], "item_classes": list(item_classes),
        "gene_columns": [str(g) for g in static.get("gene_columns") or []], "bouts": static.get("bouts"),
        "scales": scales,
        "globe": {"G": int(globe["G"]), "C": len(globe["centers"]), "grid": grid,
                  "centers": encode_column(_unit_rows(globe["centers"], bad), scale=UNIT_SCALE),
                  "elevation_m": encode_column(globe["elevation_m"], bad=bad),
                  "land": encode_column(globe["land"], kind="bit", bad=bad),
                  "sea_level_m": _finite(globe["sea_level_m"], bad)},
        "patches": patches}


def _encode_bodies(rows, L, ps, n_genes, counts, bad):
    rows = sorted(rows, key=lambda r: _finite(r[0], [0]))
    uids = [int(round(_finite(r[0], bad))) for r in rows]
    counts["duplicate_uids"] += sum(1 for a, b in zip(uids, uids[1:]) if a == b)
    col = lambda k: [r[k] for r in rows]  # noqa: E731
    lineages = [int(round(_finite(v, bad))) for v in col(5)]
    ids = sorted(set(lineages))
    slot = {v: i for i, v in enumerate(ids)}
    out = {
        "uid": encode_column(uids, delta=True, bad=bad),
        "x": encode_column(_wrap_pos([_finite(v, bad) for v in col(1)], L, counts), scale=ps, bad=bad),
        "y": encode_column(_wrap_pos([_finite(v, bad) for v in col(2)], L, counts), scale=ps, bad=bad),
        "h": encode_column([_wrap_angle(_finite(v, bad)) for v in col(3)], scale=HEADING_SCALE, bad=bad),
        "m": encode_column(col(4), scale=LOG_SCALE, offset=MASS_G_LOG_OFFSET, log=True, bad=bad),
        "lin": {"ids": encode_column(ids, delta=True, bad=bad), "idx": encode_column([slot[v] for v in lineages])},
        "k": encode_column(col(6), bad=bad),
        "mo": encode_column(col(7), scale=FRACTION_SCALE, bad=bad),
        "th": encode_column(col(8), scale=SIGNED_SCALE, bad=bad),
        "ld": encode_column(col(9), scale=FRACTION_SCALE, bad=bad),
        "v": encode_column(col(10), bad=bad),
        "dm": encode_column(col(11), scale=FRACTION_SCALE, bad=bad)}
    if n_genes and rows and all(len(r) >= len(BODY_COLUMNS) + n_genes for r in rows):
        out["g"] = [encode_column(col(len(BODY_COLUMNS) + j), kind="f4", bad=bad) for j in range(n_genes)]
    return out


def _encode_patch_frame(pf, info, cap, n_genes, n_classes, counts, bad):
    L, ps = info["L"], info["ps"]
    bodies = pf.get("bodies") or []
    out = {"n": len(bodies)}
    if bodies:
        out["a"] = _encode_bodies(bodies, L, ps, n_genes, counts, bad)
    items = pf.get("items") or []
    total = len(items)
    if total > cap:
        step = total / cap if cap else 0
        items = [items[int(k * step)] for k in range(cap)] if cap else []
    if total:
        out["it"] = {"n": len(items), "total": total}
        if items:
            classes = [int(round(_finite(r[2], bad))) for r in items]
            counts["item_class_out_of_range"] += sum(1 for c in classes if not 0 <= c < n_classes)
            out["it"].update(
                x=encode_column(_wrap_pos([_finite(r[0], bad) for r in items], L, counts), scale=ps, bad=bad),
                y=encode_column(_wrap_pos([_finite(r[1], bad) for r in items], L, counts), scale=ps, bad=bad),
                c=encode_column(classes),
                m=encode_column([r[3] for r in items], scale=LOG_SCALE, offset=MASS_KG_LOG_OFFSET, log=True, bad=bad))
    fires = pf.get("fires") or []
    if fires:
        out["f"] = {"n": len(fires),
                    "x": encode_column(_wrap_pos([_finite(r[0], bad) for r in fires], L, counts), scale=ps, bad=bad),
                    "y": encode_column(_wrap_pos([_finite(r[1], bad) for r in fires], L, counts), scale=ps, bad=bad),
                    "t": encode_column([r[2] for r in fires], bad=bad)}
    return out


def _stride(n: int, k: int) -> list[int]:
    if n == 0:
        return []
    picks = list(range(0, n, k))
    if picks[-1] != n - 1:
        picks.append(n - 1)
    return picks


def _keep_fields(n: int, k: int) -> list[int]:
    return _stride(n, k) if k < n else [0][:n]


def _time_key(frame) -> int:
    return int(frame["day"]) * TIME_BOUTS + int(frame.get("bout", 0))


# -------------------------------------------------------------------------------------- the page
def build_page(static3: dict, frames: list[dict], out_path, max_mb: float = MAX_MB, *, every: int = 1,
               title: str | None = None, template=TEMPLATE, stamp: bool = True, log=None) -> dict:
    """Write the two-view page for one world; returns a report (path, bytes, frames, what was thinned).

    ``every`` keeps one frame in K from the start (it doubles while the page is over ``max_mb``). Frames are
    ordered by (day, bout) (a stable sort; the report says if it reordered). ``stamp`` writes the build time
    into the page (False gives byte-identical pages for identical input)."""
    if every < 1:
        raise ValueError("every must be at least 1")
    if not max_mb > 0:
        raise ValueError("max_mb must be positive")
    shape = _check_static(static3)
    n_genes = len(static3.get("gene_columns") or ())
    frames = list(frames)
    for i, frame in enumerate(frames):
        _check_frame(frame, i, shape, n_genes)
    keys = [_time_key(f) for f in frames]
    reordered = any(b < a for a, b in zip(keys, keys[1:]))
    if reordered:
        order = sorted(range(len(frames)), key=lambda i: keys[i])
        frames, keys = [frames[i] for i in order], [keys[i] for i in order]
    budget = int(max_mb * 1e6)
    text = Path(template).read_text(encoding="utf-8")
    if text.count(DATA_TOKEN) != 1:
        raise ValueError("the viewer template must contain the data token exactly once")
    shell_bytes = len(text.encode("utf-8")) - len(DATA_TOKEN)

    bad = [0]  # the static part's bad values; frames and field records keep their own counts with their encoding
    item_classes = _item_classes(static3)
    grid = grid_kind(shape["G"], static3["globe"]["centers"])
    static_block = _encode_static(static3, grid, item_classes, bad)
    static_json = _escape(_json(static_block))
    slots = [ps["index"] for ps in static_block["patches"]]
    info = {ps["index"]: ps for ps in static_block["patches"]}

    # field records: global T_k and ice, each patch's plants and water (in time order)
    field_lists = {("global", key): [] for key in GLOBAL_FIELDS}
    for p in slots:
        for key in PATCH_FIELDS:
            field_lists[(p, key)] = []
    for i, frame in enumerate(frames):
        for key in GLOBAL_FIELDS:
            if (frame.get("global") or {}).get(key) is not None:
                field_lists[("global", key)].append(i)
        for pf in frame.get("patches") or ():
            for key in PATCH_FIELDS:
                if pf.get(key) is not None:
                    field_lists[(int(pf["index"]), key)].append(i)
    ignored = sorted({k for f in frames for k in (f.get("global") or {}) if k not in GLOBAL_FIELDS})
    field_records_in = sum(len(v) for v in field_lists.values())
    sides = [info[p]["n"] for p in slots]
    factors = [1]
    while (factors[-1] * 2 <= MAX_FIELD_FACTOR and sides
           and all(s % (factors[-1] * 2) == 0 and s // (factors[-1] * 2) >= MIN_FIELD_SIDE for s in sides)):
        factors.append(factors[-1] * 2)

    frame_cache, field_cache = {}, {}

    def frame_json(i, cap):
        key = (i, cap)
        if key not in frame_cache:
            frame, fbad = frames[i], [0]
            fcounts = {"wrapped_positions": 0, "duplicate_uids": 0, "item_class_out_of_range": 0}
            by_index = {int(pf["index"]): pf for pf in frame.get("patches") or ()}
            sun = frame.get("sun") or (1.0, 0.0, 0.0)
            enc = {"d": int(frame["day"]), "b": int(frame.get("bout", 0)),
                   "s": [round(v, 6) for v in _unit_rows([sun], fbad)],
                   "p": [_encode_patch_frame(by_index[p], info[p], cap, n_genes, len(item_classes), fcounts, fbad)
                         if p in by_index else None for p in slots]}
            frame_cache[key] = (_escape(_json(enc)), fcounts, fbad[0])
        return frame_cache[key][0]

    def field_json(owner, key, i, factor):
        ck = (owner, key, i, factor if owner != "global" else 1)
        if ck not in field_cache:
            frame, fbad = frames[i], [0]
            rec = {"d": int(frame["day"]), "b": int(frame.get("bout", 0))}
            if owner == "global":
                values = frame["global"][key]
                rec["v"] = (encode_column(values, kind="bit", bad=fbad) if key == "ice"
                            else encode_column(values, bad=fbad))
            else:
                pf = next(pf for pf in frame["patches"] if int(pf["index"]) == owner)
                n = info[owner]["n"]
                rec["f"] = factor
                rec["v"] = encode_column(coarsen_field([_finite(v, fbad) for v in pf[key]], n, factor), bad=fbad)
            field_cache[ck] = (_escape(_json(rec)), fbad[0])
        return field_cache[ck][0]

    first = [int(frames[0]["day"]), int(frames[0].get("bout", 0))] if frames else None
    last = [int(frames[-1]["day"]), int(frames[-1].get("bout", 0))] if frames else None
    built = datetime.now(timezone.utc).isoformat(timespec="seconds") if stamp else None

    def items_share(kept, cap):
        """Share of the frames' bytes taken by items (sampled on up to five kept frames)."""
        sample = sorted({kept[int(j * (len(kept) - 1) / 4)] for j in range(5)}) if kept else []
        full = sum(len(frame_json(i, cap)) for i in sample)
        return 1.0 - sum(len(frame_json(i, 0)) for i in sample) / full if full else 0.0

    def kept_field_map(FK):
        return {k: [v[j] for j in _keep_fields(len(v), FK)] for k, v in field_lists.items()}

    def summary(kept, cap):
        lineage_rows, masses, max_bodies, max_items, max_fires, k_max = {}, [], 0, 0, 0, 0
        for i in kept:
            for pf in frames[i].get("patches") or ():
                bodies = pf.get("bodies") or []
                max_bodies = max(max_bodies, len(bodies))
                max_items = max(max_items, min(cap, len(pf.get("items") or ())))
                max_fires = max(max_fires, len(pf.get("fires") or ()))
                for r in bodies:
                    lin = int(round(_finite(r[5], [0])))
                    lineage_rows[lin] = lineage_rows.get(lin, 0) + 1
                    m = _finite(r[4], [0])
                    if m > 0:
                        masses.append(m)
                    k_max = max(k_max, _finite(r[6], [0]))
        top = sorted(lineage_rows.items(), key=lambda e: (-e[1], e[0]))[:TOP_LINEAGES]
        if masses:
            geo = math.exp(sum(math.log(m) for m in masses) / len(masses))
            mass = [min(masses), geo, max(masses)]
        else:
            mass = None
        return {"top_lineages": [list(e) for e in top], "lineages_seen": len(lineage_rows), "mass_g": mass,
                "max_bodies": max_bodies, "max_items": max_items, "max_fires": max_fires, "k_max": k_max}

    def make_meta(kept, kept_fields, K, FK, factor, cap, fits, summ):
        return {
            "schema": PAGE_SCHEMA, "title": title or f"Planet {static3.get('seed')}: life9 v3", "built_utc": built,
            "frames_in": len(frames), "frames": len(kept), "every": K, "first": first, "last": last,
            "field_records_in": field_records_in, "field_records": sum(len(v) for v in kept_fields.values()),
            "field_every": FK, "field_factor": factor,
            "items_cap": cap,
            "items_dropped": sum(max(0, len(pf.get("items") or ()) - cap) for i in kept
                                 for pf in frames[i].get("patches") or ()),
            "fits": fits, "max_mb": max_mb, "time_bouts": TIME_BOUTS, **summ,
            "body_columns": list(BODY_COLUMNS), "item_columns": list(ITEM_COLUMNS), "fire_columns": list(FIRE_COLUMNS),
            "conventions": {"position": "metres, x east, y north, periodic on [0, L)",
                            "field_cell": "flat index ix n + iy", "heading": "radians from +x (east) toward +y (north)",
                            "sun": "unit vector toward the star in planet coordinates"},
            "notes": NOTES, "provenance": {k: list(v) for k, v in PROVENANCE.items()}}

    summ_all = summary(range(len(frames)), ITEMS_CAP)  # for the size estimate (the meta's length barely varies)
    K, FK, factor_at, cap = int(every), 1, 0, ITEMS_CAP
    while True:
        kept = _stride(len(frames), K)
        kept_fields = kept_field_map(FK)
        factor = factors[factor_at]
        frames_bytes = sum(len(frame_json(i, cap)) + 1 for i in kept)
        fields_bytes = sum(len(field_json(owner, key, i, factor)) + 1
                           for (owner, key), picks in kept_fields.items() for i in picks)
        meta_bytes = len(_escape(_json(make_meta(kept, kept_fields, K, FK, factor, cap, False, summ_all)))) + 256
        size = shell_bytes + len(static_json) + frames_bytes + fields_bytes + meta_bytes + 64 * len(field_lists)
        if size <= budget:
            fits = True
            break
        can_frames = len(kept) > 2
        longest = max((len(v) for v in kept_fields.values()), default=0)
        has_patch_fields = any(v for (owner, _), v in kept_fields.items() if owner != "global")
        can_time = longest > 1
        can_space = factor_at + 1 < len(factors) and has_patch_fields
        if (can_time or can_space) and (fields_bytes >= frames_bytes or not can_frames):
            if can_time and (longest > MIN_FIELD_RECORDS or not can_space):
                FK *= 2
            else:
                factor_at += 1
        elif can_frames and (len(kept) > MIN_FRAMES or cap <= ITEMS_FLOOR or items_share(kept, cap) < ITEMS_SHARE):
            K *= 2
        elif cap > ITEMS_FLOOR:
            cap = max(ITEMS_FLOOR, cap // 2)
        elif can_frames:
            K *= 2
        elif cap > 0:
            cap = cap // 4 if cap >= 64 else 0
        else:
            fits = False
            break
        if log:
            log(f"view3 page {size / 1e6:.2f} MB > {max_mb} MB: every={K}, field_every={FK}, "
                f"field_factor={factors[factor_at]}, items_cap={cap}")

    summ = summary(kept, cap)
    meta = make_meta(kept, kept_fields, K, FK, factor, cap, fits, summ)
    fields_out = {"global": {key: [] for key in GLOBAL_FIELDS}, "patch": [{key: [] for key in PATCH_FIELDS}
                                                                          for _ in slots]}
    for (owner, key), picks in kept_fields.items():
        parts = [field_json(owner, key, i, factor) for i in picks]
        if owner == "global":
            fields_out["global"][key] = parts
        else:
            fields_out["patch"][slots.index(owner)][key] = parts
    fields_text = ('{"global":{' + ",".join(f'"{k}":[' + ",".join(v) + "]" for k, v in fields_out["global"].items())
                   + '},"patch":[' + ",".join("{" + ",".join(f'"{k}":[' + ",".join(v) + "]" for k, v in rec.items())
                                              + "}" for rec in fields_out["patch"]) + "]}")
    # bad values and data problems in what the page holds (the static part, the kept frames and field records)
    counts = {"wrapped_positions": 0, "duplicate_uids": 0, "item_class_out_of_range": 0}
    nonfinite = bad[0]
    for i in kept:
        _, fcounts, fbad = frame_cache[(i, cap)]
        nonfinite += fbad
        for k, v in fcounts.items():
            counts[k] += v
    for (owner, key), picks in kept_fields.items():
        nonfinite += sum(field_cache[(owner, key, i, factor if owner != "global" else 1)][1] for i in picks)
    payload = ('{"meta":' + _escape(_json(meta)) + ',"static":' + static_json
               + ',"frames":[' + ",".join(frame_json(i, cap) for i in kept) + ']'
               + ',"fields":' + fields_text + '}')
    html = text.replace(DATA_TOKEN, payload)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = html.encode("utf-8")
    out.write_bytes(data)
    size = len(data)
    return {"out": str(out.resolve()), "bytes": size, "megabytes": round(size / 1e6, 3), "max_mb": max_mb,
            "fits": size <= budget, "frames_in": len(frames), "frames": len(kept), "every": K,
            "field_records_in": field_records_in, "field_records": meta["field_records"], "field_every": FK,
            "field_factor": factor, "grid": grid, "items_cap": cap, "items_dropped": meta["items_dropped"],
            "max_bodies": summ["max_bodies"], "lineages_seen": summ["lineages_seen"], "first": first, "last": last,
            "reordered": reordered, "nonfinite_values": nonfinite, **counts, "ignored_global_keys": ignored,
            "scripts": list(ALLOWED_SCRIPTS)}


# ------------------------------------------------------------------------------------- reading back
def read_payload(path) -> dict:
    """The JSON embedded in a built page (columns still encoded)."""
    html = Path(path).read_text(encoding="utf-8")
    start = html.index('<script id="view3-data" type="application/json">')
    start = html.index(">", start) + 1
    end = html.index("</script>", start)
    return json.loads(html[start:end])


def _triples(values):
    return [values[k:k + 3] for k in range(0, len(values), 3)]


def decode_patch_frame(pf: dict | None) -> dict | None:
    """One encoded patch entry back to rows: bodies as BODY_COLUMNS (+ genes), items, fires."""
    if pf is None:
        return None
    out = {"bodies": [], "items": [], "fires": [], "items_total": (pf.get("it") or {}).get("total", 0)}
    a = pf.get("a")
    if a:
        cols = {k: decode_column(v) for k, v in a.items() if k not in ("lin", "g")}
        ids = decode_column(a["lin"]["ids"])
        lineage = [ids[j] for j in decode_column(a["lin"]["idx"])]
        genes = [decode_column(g) for g in a.get("g") or ()]
        for n in range(pf["n"]):
            row = [cols["uid"][n], cols["x"][n], cols["y"][n], cols["h"][n], cols["m"][n], lineage[n], cols["k"][n],
                   cols["mo"][n], cols["th"][n], cols["ld"][n], cols["v"][n], cols["dm"][n]]
            out["bodies"].append(row + [g[n] for g in genes])
    it = pf.get("it")
    if it and it["n"]:
        out["items"] = [list(r) for r in zip(decode_column(it["x"]), decode_column(it["y"]), decode_column(it["c"]),
                                             decode_column(it["m"]))]
    f = pf.get("f")
    if f:
        out["fires"] = [list(r) for r in zip(decode_column(f["x"]), decode_column(f["y"]), decode_column(f["t"]))]
    return out


def load_page(path) -> dict:
    """A built page decoded to plain lists: meta, static (globe and patches), frames, fields."""
    data = read_payload(path)
    st = json.loads(json.dumps(data["static"]))
    g = st["globe"]
    g["centers"] = _triples(decode_column(g["centers"]))
    g["elevation_m"] = decode_column(g["elevation_m"])
    g["land"] = decode_column(g["land"])
    for ps in st["patches"]:
        ps["elevation_m"] = decode_column(ps["elevation_m"])
        ps["pond"] = decode_column(ps["pond"])
    frames = [{"day": fr["d"], "bout": fr["b"], "sun": fr["s"], "patches": [decode_patch_frame(p) for p in fr["p"]]}
              for fr in data["frames"]]
    fields = {"global": {k: [{"day": r["d"], "bout": r["b"], "values": decode_column(r["v"])} for r in recs]
                         for k, recs in data["fields"]["global"].items()},
              "patch": [{k: [{"day": r["d"], "bout": r["b"], "factor": r["f"], "values": decode_column(r["v"])}
                             for r in recs] for k, recs in rec.items()} for rec in data["fields"]["patch"]]}
    return {"meta": data["meta"], "static": st, "frames": frames, "fields": fields}


# ------------------------------------------------------------------------------------------- CLI
def _read_frames(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m haishool.life9.v3.view3",
                                     description="Write the self-contained two-view page (globe and patch) for one "
                                                 "life9 v3 world.")
    parser.add_argument("static", type=Path, help="static.json (schema life9-v3-view-1)")
    parser.add_argument("frames", type=Path, help="frames.jsonl, one frame per line")
    parser.add_argument("--out", type=Path, required=True, help="HTML file to write")
    parser.add_argument("--max-mb", type=float, default=MAX_MB, help=f"size bound in MB (default {MAX_MB:g})")
    parser.add_argument("--every", type=int, default=1, help="keep one frame in K (doubles to fit the bound)")
    args = parser.parse_args(argv)
    try:
        static = json.loads(args.static.read_text(encoding="utf-8"))
        report = build_page(static, list(_read_frames(args.frames)), args.out, max_mb=args.max_mb, every=args.every,
                            log=lambda text: print(text, file=sys.stderr, flush=True))
    except (FrameError, ValueError, OSError) as exc:
        parser.exit(2, f"view3: {exc}\n")
    print(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    main()
