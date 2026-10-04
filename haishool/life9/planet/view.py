"""Globe viewer for life9 planet runs: one self-contained HTML page from globe frames.

    build_page(static, frames, out_path, *, max_mb=12.0, every=1) -> report
    python -m haishool.life9.planet.view STATIC.json FRAMES.jsonl --out PAGE.html [--max-mb 12] [--every 1]

Input format "life9-globe-v1" (``world.frame(w)`` produces it; this module only reads it):

    static = {"schema": "life9-globe-v1", "seed": int,
              "planet": {radius_m, gravity_m_s2, surface_pressure_pa, tidally_locked, year_s, star_teff_k,
                         insolation_w_m2, t_surface_target_k, ...}   # a PlanetSpec.to_dict() subset
              "habitat_radius_m": float, "G": int, "cells": C, "centers": [[x, y, z]] * C (unit vectors),
              "elevation_m": [C], "land": [C 0/1], "sea_level_m": float, "relief_exaggeration": float,
              "species": [names], "actions": [names], optional "item_classes": [names]}
    frame  = {"day": int, "sun": [x, y, z] (unit vector toward the star, planet coordinates),
              "agents": [[uid, x, y, z, heading_rad, mass_kg, founder, loud, vocal_argmax, k_active,
                          action_index, health(, diet)]],
              "fires": [[x, y, z, temp_k]], "items": [[x, y, z, class_index, mass_kg]] (ground items),
              optional "fields": {"T_k": [C], "plant": [C 0-255], "snow": [C 0/1], "soil": [C 0-255]}}

An agent row may carry a 13th value, ``diet`` (0 plants .. 1 meat); the viewer then offers a diet
colouring. ``class_index`` indexes ``static["item_classes"]`` when given, else ``materials.CLASSES``.

The page is ``viewer_globe.html`` with the data embedded as JSON in place of ``DATA_TOKEN``. Numeric
arrays are stored as little-endian typed arrays in base64 (``{"t": type, "b": base64, "s": scale,
"o": offset}``, value = stored / s + o): unit vectors in 1e-4 steps (1 m on the 10 km habitat), headings
in 1e-4 rad, loudness, health and diet in 1/250 steps, masses as float32, snow as bits. The page loads
only the two pinned scripts in ``ALLOWED_SCRIPTS``.

Size bound: while the page would exceed ``max_mb`` the builder thins the larger of the two parts,
fields or frames. Fields: it keeps one field frame in FK (thinning in time down to
``MIN_FIELD_FRAMES``), then averages the fields over s x s blocks of cube-sphere cells (s = 2, 4;
only when the centres are the life9 cube-sphere grid, checked here), then thins in time again.
Frames: it keeps one frame in K (the first and the last always) down to ``MIN_FRAMES``, then draws
fewer ground items per frame (an even spread, down to ``ITEMS_FLOOR``), then thins frames to two,
then drops items. The report and the page say what was thinned. If nothing more can be thinned
the page is still written and the report says ``fits: False``.

The viewer shows recorded data and adds no behaviour. Its display choices (marker sizes, colours,
ring radius proportional to loudness, interpolation between frames) are display only, not world
numbers. Standard library only (it reads the cube-sphere face frames from ``globe`` and the item
class names from ``materials``; it imports no world code).
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

TEMPLATE = Path(__file__).resolve().parent / "viewer_globe.html"
DATA_TOKEN = "__LIFE9_GLOBE_DATA__"
SCHEMA = "life9-globe-v1"
PAGE_SCHEMA = "life9-globe-page-v1"
ALLOWED_SCRIPTS = ("https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js",
                   "https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js")
MAX_MB = 12.0
MIN_FIELD_FRAMES = 8        # field frames kept before the fields are coarsened in space
MAX_FIELD_FACTOR = 4        # coarsest spatial block of the fields (s x s cells)
ITEMS_CAP = 2000            # ground items per frame (the frame format's own cap)
MIN_FRAMES = 64             # frames kept before fewer ground items are drawn per frame
ITEMS_FLOOR = 250           # items per frame kept until the frames are thinned to 2
UNIT_SCALE = 1e4            # unit-vector components in 1e-4 steps
ANGLE_SCALE = 1e4           # headings in 1e-4 rad
FRACTION_SCALE = 250        # loudness, health, diet in 1/250 steps
CUBE_TOLERANCE = 2e-4       # a centre within this of the cube-sphere formula (4-dp rounding is 5e-5)
AGENT_COLUMNS = ("uid", "x", "y", "z", "heading_rad", "mass_kg", "founder", "loud", "vocal_argmax", "k_active",
                 "action_index", "health")
OPTIONAL_AGENT_COLUMNS = ("diet",)
FIELD_KEYS = ("T_k", "plant", "snow", "soil")
PLANET_KEYS = ("radius_m", "gravity_m_s2", "surface_pressure_pa", "tidally_locked", "year_s", "star_teff_k",
               "insolation_w_m2", "t_surface_target_k")
DEFAULT_ITEM_CLASSES = ("stone", "ore", "clay", "wood", "organic", "metal", "product")  # materials.CLASSES

# typecode in the page, array module code, smallest and largest value
_INT_TYPES = (("i1", "b", -2 ** 7, 2 ** 7 - 1), ("u1", "B", 0, 2 ** 8 - 1), ("i2", "h", -2 ** 15, 2 ** 15 - 1),
              ("u2", "H", 0, 2 ** 16 - 1), ("i4", "i", -2 ** 31, 2 ** 31 - 1))
_ARRAY_CODE = {name: code for name, code, _, _ in _INT_TYPES} | {"f4": "f", "f8": "d"}
assert array.array("i").itemsize == 4 and array.array("f").itemsize == 4 and array.array("h").itemsize == 2

NOTES = {
    "habitat": ("Labelled new_rule (PLANET-SPEC section 1): a real planet cannot be inhabited at the scale of "
                "individuals, so the world is a small globe. Its surface, gravity, air, light, temperature and "
                "chemistry are the planet's own derived values; its geography keeps the planet's ocean coverage, "
                "latitude and substellar climate and crust composition; curvature, the horizon and walking "
                "distances are those of the small globe."),
    "meaning": ("No meaning is supplied (PLANET-SPEC 0.4). A call is an 8-number vector the brain emits; the "
                "channel shown is only the index of its largest component, and nothing in the world gives a "
                "call, a channel or an action a meaning or a reward."),
    "display": ("Display only: relief is exaggerated by the stated factor, markers are not to scale, ring size "
                "follows loudness, and motion between recorded frames is interpolated."),
}
PROVENANCE = {
    "habitat_radius_m": ("new_rule", "habitat globe compression, PLANET-SPEC 1"),
    "relief_exaggeration": ("new_rule", "display only: vertical scale of the drawn relief"),
    "field_factor": ("new_rule", "display only: fields averaged over s x s cube-sphere cells to fit the page"),
}


class FrameError(ValueError):
    """The static or frame data do not follow the life9-globe-v1 format."""


# ------------------------------------------------------------------------------------------ codec
def _finite(value, counter):
    value = float(value)
    if math.isfinite(value):
        return value
    counter[0] += 1
    return 0.0


def _b64(arr: array.array) -> str:
    if sys.byteorder != "little":
        arr = array.array(arr.typecode, arr)
        arr.byteswap()
    return base64.b64encode(arr.tobytes()).decode("ascii")


def encode_column(values, *, scale: float | None = None, offset: float = 0.0, kind: str | None = None,
                  bad=None) -> dict:
    """Pack numbers as the smallest typed array that holds them: {"t", "b"(, "s", "o")}.

    kind "f4" stores float32; kind "bit" packs 0/1 values 8 per byte ({"t": "bit", "n": count});
    otherwise values (minus offset, times scale when given) are rounded to integers. ``bad`` is a
    one-element list that counts non-finite inputs (stored as 0)."""
    bad = [0] if bad is None else bad
    if kind == "bit":
        flags = [1 if _finite(v, bad) >= 0.5 else 0 for v in values]
        packed = bytearray((len(flags) + 7) // 8)
        for i, flag in enumerate(flags):
            if flag:
                packed[i >> 3] |= 1 << (i & 7)
        return {"t": "bit", "n": len(flags), "b": base64.b64encode(bytes(packed)).decode("ascii")}
    if kind in ("f4", "f8"):
        return {"t": kind, "b": _b64(array.array(_ARRAY_CODE[kind], [_finite(v, bad) for v in values]))}
    if scale is None:
        ints = [int(round(_finite(v, bad))) for v in values]
    else:
        ints = [int(round((_finite(v, bad) - offset) * scale)) for v in values]
    lo, hi = (min(ints), max(ints)) if ints else (0, 0)
    for name, code, low, high in _INT_TYPES:
        if low <= lo and hi <= high:
            out = {"t": name, "b": _b64(array.array(code, ints))}
            break
    else:  # beyond int32: exact up to 2^53 in float64
        out = {"t": "f8", "b": _b64(array.array("d", [float(v) for v in ints]))}
    if scale is not None:
        out["s"] = scale
        if offset:
            out["o"] = offset
    return out


def decode_column(col: dict | None) -> list | None:
    """Inverse of :func:`encode_column` (for tests and analysis)."""
    if col is None:
        return None
    raw = base64.b64decode(col["b"])
    if col["t"] == "bit":
        return [(raw[i >> 3] >> (i & 7)) & 1 for i in range(col["n"])]
    arr = array.array(_ARRAY_CODE[col["t"]])
    arr.frombytes(raw)
    if sys.byteorder != "little":
        arr.byteswap()
    values = list(arr)
    if "s" in col:
        scale, offset = col["s"], col.get("o", 0.0)
        return [v / scale + offset for v in values]
    return values


def _json(obj) -> str:
    return json.dumps(obj, allow_nan=False, separators=(",", ":"), ensure_ascii=True)


# ------------------------------------------------------------------------------------- the grid
def cube_sphere_centers(G: int) -> list[tuple[float, float, float]]:
    """Cell centres of the life9 equiangular cube-sphere (``globe.Globe``): c = f G^2 + i G + j."""
    from .globe import _FACES
    delta = math.pi / (2 * G)
    tans = [math.tan(-math.pi / 4 + delta * (k + 0.5)) for k in range(G)]
    out = []
    for n, u, v in _FACES:
        for i in range(G):
            for j in range(G):
                p = [n[k] + tans[i] * u[k] + tans[j] * v[k] for k in range(3)]
                norm = math.sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2)
                out.append((p[0] / norm, p[1] / norm, p[2] / norm))
    return out


def grid_kind(G: int, centers) -> str:
    """'cube-sphere' if the centres are the life9 grid in its cell order, else 'points'."""
    if len(centers) != 6 * G * G:
        return "points"
    for (x, y, z), (a, b, c) in zip(centers, cube_sphere_centers(G)):
        if abs(x - a) > CUBE_TOLERANCE or abs(y - b) > CUBE_TOLERANCE or abs(z - c) > CUBE_TOLERANCE:
            return "points"
    return "cube-sphere"


def coarse_index(G: int, factor: int) -> list[int]:
    """For each fine cube-sphere cell its s x s block: f (G/s)^2 + (i // s) (G/s) + j // s."""
    gc = G // factor
    return [f * gc * gc + (i // factor) * gc + j // factor for f in range(6) for i in range(G) for j in range(G)]


def coarsen(values, G: int, factor: int, *, majority=False) -> list[float]:
    """Mean (or majority for 0/1 flags) of a cell field over s x s cube-sphere blocks."""
    if factor == 1:
        return list(values)
    gc = G // factor
    sums, counts = [0.0] * (6 * gc * gc), [0] * (6 * gc * gc)
    for cell, block in enumerate(coarse_index(G, factor)):
        sums[block] += float(values[cell])
        counts[block] += 1
    means = [s / n for s, n in zip(sums, counts)]
    return [1.0 if m >= 0.5 else 0.0 for m in means] if majority else means


# ------------------------------------------------------------------------------------ validation
def _check_static(static: dict) -> dict:
    if not isinstance(static, dict) or static.get("schema") != SCHEMA:
        raise FrameError(f"static['schema'] must be {SCHEMA!r}")
    for key in ("G", "cells", "centers", "elevation_m", "land", "sea_level_m", "habitat_radius_m"):
        if key not in static:
            raise FrameError(f"static is missing {key!r}")
    G, C = int(static["G"]), int(static["cells"])
    for key in ("centers", "elevation_m", "land"):
        if len(static[key]) != C:
            raise FrameError(f"static[{key!r}] has {len(static[key])} entries, cells is {C}")
    if any(len(p) != 3 for p in static["centers"]):
        raise FrameError("every centre must be [x, y, z]")
    if not float(static["habitat_radius_m"]) > 0:
        raise FrameError("habitat_radius_m must be positive")
    return {"G": G, "C": C}


def _check_frame(frame: dict, index: int, C: int):
    if not isinstance(frame, dict) or "day" not in frame:
        raise FrameError(f"frame {index} has no 'day'")
    for key, width in (("agents", len(AGENT_COLUMNS)), ("fires", 4), ("items", 5)):
        for row in frame.get(key) or ():
            if len(row) < width:
                raise FrameError(f"frame {index} (day {frame['day']}): a {key} row has {len(row)} values, "
                                 f"needs {width}")
    sun = frame.get("sun")
    if sun is not None and len(sun) != 3:
        raise FrameError(f"frame {index}: sun must be [x, y, z]")
    for key, values in (frame.get("fields") or {}).items():
        if key in FIELD_KEYS and len(values) != C:
            raise FrameError(f"frame {index}: field {key!r} has {len(values)} cells, expected {C}")


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


def _wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _encode_static(static, grid, item_classes):
    planet = {k: v for k, v in (static.get("planet") or {}).items()
              if isinstance(v, (int, float, str, bool)) or v is None or
              (isinstance(v, dict) and all(isinstance(x, (int, float)) for x in v.values()))}
    for key, value in planet.items():
        if isinstance(value, float) and not math.isfinite(value):
            planet[key] = None
    bad = [0]
    elevation = encode_column(static["elevation_m"], bad=bad)
    return {
        "source_schema": static["schema"], "seed": static.get("seed"), "planet": planet,
        "habitat_radius_m": float(static["habitat_radius_m"]), "G": int(static["G"]),
        "cells": int(static["cells"]), "grid": grid,
        "centers": encode_column(_unit_rows(static["centers"], bad), scale=UNIT_SCALE),
        "elevation_m": elevation, "land": encode_column(static["land"], kind="bit"),
        "sea_level_m": _finite(static["sea_level_m"], bad),
        "relief_exaggeration": _finite(static.get("relief_exaggeration", 1.0), bad) or 1.0,
        "species": list(static.get("species") or []), "actions": list(static.get("actions") or []),
        "item_classes": list(item_classes)}, bad[0]


def _encode_frame(frame, items_cap, bad):
    agents = frame.get("agents") or []
    has_diet = bool(agents) and all(len(row) > len(AGENT_COLUMNS) for row in agents)
    col = lambda k: [row[k] for row in agents]  # noqa: E731
    sun = frame.get("sun") or (1.0, 0.0, 0.0)
    out = {"d": int(frame["day"]), "s": [round(v, 5) for v in _unit_rows([sun], bad)], "n": len(agents)}
    if agents:
        out["a"] = {
            "uid": encode_column(col(0), bad=bad),
            "pos": encode_column(_unit_rows([row[1:4] for row in agents], bad), scale=UNIT_SCALE),
            "heading": encode_column([_wrap(_finite(v, bad)) for v in col(4)], scale=ANGLE_SCALE),
            "mass": encode_column(col(5), kind="f4", bad=bad),
            "founder": encode_column(col(6), bad=bad),
            "loud": encode_column(col(7), scale=FRACTION_SCALE, bad=bad),
            "vocal": encode_column(col(8), bad=bad),
            "k": encode_column(col(9), bad=bad),
            "action": encode_column(col(10), bad=bad),
            "health": encode_column(col(11), scale=FRACTION_SCALE, bad=bad)}
        if has_diet:
            out["a"]["diet"] = encode_column(col(12), scale=FRACTION_SCALE, bad=bad)
    fires = frame.get("fires") or []
    if fires:
        out["f"] = {"n": len(fires), "pos": encode_column(_unit_rows(fires, bad), scale=UNIT_SCALE),
                    "temp": encode_column([row[3] for row in fires], bad=bad)}
    items = frame.get("items") or []
    total = len(items)
    if total > items_cap:
        step = total / items_cap if items_cap else 0
        items = [items[int(k * step)] for k in range(items_cap)] if items_cap else []
    if total:
        out["i"] = {"n": len(items), "total": total}
        if items:
            out["i"].update(pos=encode_column(_unit_rows(items, bad), scale=UNIT_SCALE),
                            cls=encode_column([row[3] for row in items], bad=bad),
                            mass=encode_column([row[4] for row in items], kind="f4", bad=bad))
    return out


def _encode_fields(day, fields, G, factor, bad):
    out = {"d": int(day)}
    for key in FIELD_KEYS:
        if key not in fields:
            continue
        values = [_finite(v, bad) for v in fields[key]]
        if key == "snow":
            out[key] = encode_column(coarsen(values, G, factor, majority=True), kind="bit")
        else:
            out[key] = encode_column(coarsen(values, G, factor))
    return out


def _stride(n: int, k: int) -> list[int]:
    if n == 0:
        return []
    picks = list(range(0, n, k))
    if picks[-1] != n - 1:
        picks.append(n - 1)
    return picks


def _escape(text: str) -> str:
    return text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def _item_classes(static):
    if static.get("item_classes"):
        return list(static["item_classes"])
    try:
        from .materials import CLASSES
        return list(CLASSES)
    except Exception:  # pragma: no cover - the viewer must not depend on the engine tables
        return list(DEFAULT_ITEM_CLASSES)


# -------------------------------------------------------------------------------------- the page
def build_page(static: dict, frames: list[dict], out_path, *, max_mb: float = MAX_MB, every: int = 1,
               title: str | None = None, template=TEMPLATE, log=None) -> dict:
    """Write the globe page for one world; returns a report (path, bytes, frames, what was thinned).

    ``every`` keeps one frame in K from the start (it doubles while the page is over ``max_mb``).
    Frames are ordered by day (a stable sort; the report says if it reordered)."""
    if every < 1:
        raise ValueError("every must be at least 1")
    if not max_mb > 0:
        raise ValueError("max_mb must be positive")
    shape = _check_static(static)
    G, C = shape["G"], shape["C"]
    frames = list(frames)
    for i, frame in enumerate(frames):
        _check_frame(frame, i, C)
    days = [int(f["day"]) for f in frames]
    reordered = any(b < a for a, b in zip(days, days[1:]))
    if reordered:
        frames = sorted(frames, key=lambda f: int(f["day"]))
    grid = grid_kind(G, static["centers"])
    budget = int(max_mb * 1e6)
    text = Path(template).read_text(encoding="utf-8")
    if text.count(DATA_TOKEN) != 1:
        raise ValueError("the globe template must contain the data token exactly once")
    shell_bytes = len(text.encode("utf-8")) - len(DATA_TOKEN)
    static_block, bad_static = _encode_static(static, grid, _item_classes(static))
    static_json = _escape(_json(static_block))

    field_src = [i for i, f in enumerate(frames) if f.get("fields")]
    ignored = sorted({k for i in field_src for k in frames[i]["fields"] if k not in FIELD_KEYS})
    factors = [1]
    while (grid == "cube-sphere" and factors[-1] * 2 <= MAX_FIELD_FACTOR and G % (factors[-1] * 2) == 0
           and G // (factors[-1] * 2) >= 2):
        factors.append(factors[-1] * 2)

    frame_cache, field_cache, bad = {}, {}, [0]

    def frame_json(i, cap):
        key = (i, cap)
        if key not in frame_cache:
            frame_cache[key] = _escape(_json(_encode_frame(frames[i], cap, bad)))
        return frame_cache[key]

    def field_json(i, factor):
        key = (i, factor)
        if key not in field_cache:
            field_cache[key] = _escape(_json(_encode_fields(frames[i]["day"], frames[i]["fields"], G, factor, bad)))
        return field_cache[key]

    first_day = int(frames[0]["day"]) if frames else None
    last_day = int(frames[-1]["day"]) if frames else None
    built = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def make_meta(kept, kept_fields, K, FK, factor, cap, fits):
        return {
            "schema": PAGE_SCHEMA, "title": title or f"Planet {static.get('seed')} habitat globe", "built_utc": built,
            "frames_in": len(frames), "frames": len(kept), "every": K,
            "field_frames_in": len(field_src), "field_frames": len(kept_fields), "field_every": FK,
            "field_factor": factor, "field_G": G // factor if grid == "cube-sphere" else None,
            "items_cap": cap, "items_dropped": sum(max(0, len(frames[i].get("items") or ()) - cap) for i in kept),
            "first_day": first_day, "last_day": last_day,
            "max_agents": max((len(frames[i].get("agents") or ()) for i in kept), default=0),
            "fits": fits, "max_mb": max_mb, "grid": grid,
            "agent_columns": list(AGENT_COLUMNS) + list(OPTIONAL_AGENT_COLUMNS),
            "notes": NOTES, "provenance": {k: list(v) for k, v in PROVENANCE.items()}}

    K, FK, factor_at, cap = int(every), 1, 0, ITEMS_CAP
    while True:
        kept = _stride(len(frames), K)
        # fields thin down to the first field frame alone (frames keep their first and last)
        picks = _stride(len(field_src), FK) if FK < len(field_src) else [0][:len(field_src)]
        kept_fields = [field_src[j] for j in picks]
        factor = factors[factor_at]
        frames_bytes = sum(len(frame_json(i, cap)) + 1 for i in kept)
        fields_bytes = sum(len(field_json(i, factor)) + 1 for i in kept_fields)
        meta_bytes = len(_escape(_json(make_meta(kept, kept_fields, K, FK, factor, cap, False)))) + 40
        size = shell_bytes + len(static_json) + frames_bytes + fields_bytes + meta_bytes
        if size <= budget:
            fits = True
            break
        can_frames = len(kept) > 2
        can_time = len(kept_fields) > 1
        can_space = factor_at + 1 < len(factors)
        if (can_time or can_space) and (fields_bytes >= frames_bytes or not can_frames):
            if can_time and (len(kept_fields) > MIN_FIELD_FRAMES or not can_space):
                FK *= 2
            else:
                factor_at += 1
        elif can_frames and (len(kept) > MIN_FRAMES or cap <= ITEMS_FLOOR):
            K *= 2
        elif cap > ITEMS_FLOOR:
            cap //= 2
        elif can_frames:
            K *= 2
        elif cap > 0:
            cap = cap // 4 if cap >= 64 else 0
        else:
            fits = False
            break
        if log:
            log(f"globe page {size / 1e6:.2f} MB > {max_mb} MB: every={K}, field_every={FK}, "
                f"field_factor={factors[factor_at]}, items_cap={cap}")

    meta = make_meta(kept, kept_fields, K, FK, factor, cap, fits)
    payload = ('{"meta":' + _escape(_json(meta)) + ',"static":' + static_json
               + ',"frames":[' + ",".join(frame_json(i, cap) for i in kept) + ']'
               + ',"fields":[' + ",".join(field_json(i, factor) for i in kept_fields) + ']}')
    html = text.replace(DATA_TOKEN, payload)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = html.encode("utf-8")
    out.write_bytes(data)
    size = len(data)
    return {"out": str(out.resolve()), "bytes": size, "megabytes": round(size / 1e6, 3), "max_mb": max_mb,
            "fits": size <= budget, "frames_in": len(frames), "frames": len(kept), "every": K,
            "field_frames_in": len(field_src), "field_frames": len(kept_fields), "field_every": FK,
            "field_factor": factor, "grid": grid, "items_cap": cap, "items_dropped": meta["items_dropped"],
            "max_agents": meta["max_agents"], "first_day": first_day, "last_day": last_day,
            "reordered": reordered, "nonfinite_values": bad[0] + bad_static, "ignored_field_keys": ignored}


# ------------------------------------------------------------------------------------- reading back
def read_payload(path) -> dict:
    """The JSON embedded in a built page (columns still encoded)."""
    html = Path(path).read_text(encoding="utf-8")
    start = html.index('<script id="globe-data" type="application/json">')
    start = html.index(">", start) + 1
    end = html.index("</script>", start)
    return json.loads(html[start:end])


def _triples(values):
    return [values[k:k + 3] for k in range(0, len(values), 3)]


def decode_frame(fr: dict) -> dict:
    """One encoded frame back to rows: agents as AGENT_COLUMNS (+ diet), fires, items."""
    out = {"day": fr["d"], "sun": fr["s"], "agents": [], "fires": [], "items": [],
           "items_total": fr.get("i", {}).get("total", 0)}
    a = fr.get("a")
    if a:
        cols = {k: decode_column(v) for k, v in a.items()}
        pos = _triples(cols["pos"])
        for n in range(fr["n"]):
            row = [cols["uid"][n], *pos[n], cols["heading"][n], cols["mass"][n], cols["founder"][n],
                   cols["loud"][n], cols["vocal"][n], cols["k"][n], cols["action"][n], cols["health"][n]]
            if "diet" in cols:
                row.append(cols["diet"][n])
            out["agents"].append(row)
    f = fr.get("f")
    if f:
        temps = decode_column(f["temp"])
        out["fires"] = [[*p, t] for p, t in zip(_triples(decode_column(f["pos"])), temps)]
    i = fr.get("i")
    if i and i["n"]:
        out["items"] = [[*p, c, m] for p, c, m in zip(_triples(decode_column(i["pos"])), decode_column(i["cls"]),
                                                      decode_column(i["mass"]))]
    return out


def load_page(path) -> dict:
    """A built page decoded to plain lists: meta, static (centres, elevation, land), frames, fields."""
    data = read_payload(path)
    st = dict(data["static"])
    st["centers"] = _triples(decode_column(st["centers"]))
    st["elevation_m"] = decode_column(st["elevation_m"])
    st["land"] = decode_column(st["land"])
    fields = [{"day": f["d"], **{k: decode_column(f[k]) for k in FIELD_KEYS if k in f}} for f in data["fields"]]
    return {"meta": data["meta"], "static": st, "frames": [decode_frame(f) for f in data["frames"]],
            "fields": fields}


# ------------------------------------------------------------------------------------------- CLI
def _read_frames(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m haishool.life9.planet.view",
                                     description="Write the self-contained globe page for one life9 world.")
    parser.add_argument("static", type=Path, help="static.json (schema life9-globe-v1)")
    parser.add_argument("frames", type=Path, help="frames.jsonl, one frame per line")
    parser.add_argument("--out", type=Path, required=True, help="HTML file to write")
    parser.add_argument("--max-mb", type=float, default=MAX_MB, help=f"size bound in MB (default {MAX_MB:g})")
    parser.add_argument("--every", type=int, default=1, help="keep one frame in K (doubles to fit the bound)")
    args = parser.parse_args(argv)
    try:
        static = json.loads(args.static.read_text(encoding="utf-8"))
        report = build_page(static, list(_read_frames(args.frames)), args.out, max_mb=args.max_mb,
                            every=args.every, log=lambda text: print(text, file=sys.stderr, flush=True))
    except (FrameError, ValueError, OSError) as exc:
        parser.exit(2, f"view: {exc}\n")
    print(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    main()
