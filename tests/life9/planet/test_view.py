"""The globe viewer page (haishool/life9/planet/view.py + viewer_globe.html) on synthetic globe frames.

The fixture is generated here in the life9-globe-v1 format (static + frames); world.py is not used.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import random
import re
import shutil
import subprocess

import pytest
import torch

from haishool.life9.planet import view
from haishool.life9.planet.globe import Globe

ACTIONS = ("rest", "forage", "eat_meat", "drink", "strike", "collect", "drop", "knap", "combine", "make_fire",
           "feed_fire", "heat_item", "cook", "share")
NODE = shutil.which("node")


# ------------------------------------------------------------------------------------- fixtures
def _unit(rnd):
    x, y, z = (rnd.gauss(0, 1) for _ in range(3))
    n = math.sqrt(x * x + y * y + z * z)
    return [x / n, y / n, z / n]


def make_static(G=8, seed=5, centers=None):
    rnd = random.Random(seed)
    if centers is None:
        centers = [[round(v, 4) for v in p] for p in Globe(G).centers.tolist()]
    C = len(centers)
    elevation = [int(300 + 250 * math.sin(3 * p[0]) * math.cos(2 * p[2]) + rnd.uniform(-40, 40)) for p in centers]
    sea = 320.0
    return {"schema": "life9-globe-v1", "seed": seed,
            "planet": {"radius_m": 5674430.0, "gravity_m_s2": 7.5441, "surface_pressure_pa": 88971.0,
                       "tidally_locked": True, "year_s": 1.035e7, "star_teff_k": 4300.0, "insolation_w_m2": 1203.0,
                       "t_surface_target_k": 287.67, "day_length_s": None,
                       "partial_pressure_pa": {"O2": 2500.0, "N2": 70000.0}},
            "habitat_radius_m": 10_000.0, "G": G, "cells": C, "centers": centers, "elevation_m": elevation,
            "land": [1 if e > sea else 0 for e in elevation], "sea_level_m": sea, "relief_exaggeration": 3.0,
            "species": ["granite", "basalt", "clay"], "actions": list(ACTIONS)}


def make_frames(C, n_frames=24, n_agents=40, seed=11, field_every=5, diet=True, items=30, fires=3):
    rnd = random.Random(seed)
    frames = []
    for day in range(n_frames):
        agents = []
        for k in range(n_agents + day % 3):  # the population changes a little
            row = [1000 + k, *[round(v, 4) for v in _unit(rnd)], round(rnd.uniform(0, 2 * math.pi), 4),
                   round(rnd.uniform(0.5, 90.0), 3), k % 7, round(rnd.random(), 3), rnd.randrange(8),
                   rnd.randrange(8, 128), rnd.randrange(len(ACTIONS)), round(rnd.random(), 3)]
            if diet:
                row.append(round(rnd.random(), 3))
            agents.append(row)
        frame = {"day": 3 * day, "sun": [1.0, 0.0, 0.0], "agents": agents,
                 "fires": [[*[round(v, 4) for v in _unit(rnd)], rnd.randrange(700, 1700)] for _ in range(fires)],
                 "items": [[*[round(v, 4) for v in _unit(rnd)], rnd.randrange(7), round(rnd.uniform(0.01, 5), 3)]
                           for _ in range(items)]}
        if day % field_every == 0:
            frame["fields"] = {"T_k": [rnd.randrange(200, 330) for _ in range(C)],
                               "plant": [rnd.randrange(256) for _ in range(C)],
                               "snow": [rnd.randrange(2) for _ in range(C)],
                               "soil": [rnd.randrange(256) for _ in range(C)]}
        frames.append(frame)
    return frames


@pytest.fixture(scope="module")
def world():
    static = make_static()
    return static, make_frames(static["cells"])


@pytest.fixture(scope="module")
def page(world, tmp_path_factory):
    static, frames = world
    out = tmp_path_factory.mktemp("globe") / "globe.html"
    return out, view.build_page(static, frames, out)


def _norm(p):
    n = math.sqrt(sum(v * v for v in p))
    return [v / n for v in p]


def _close_angle(a, b, tol):
    d = (a - b) % (2 * math.pi)
    return min(d, 2 * math.pi - d) <= tol


# --------------------------------------------------------------------------------- round trip
def test_report_matches_the_written_page(world, page):
    out, report = page
    static, frames = world
    assert report["bytes"] == out.stat().st_size and report["fits"]
    assert report["frames"] == report["frames_in"] == len(frames) and report["every"] == 1
    assert report["field_frames"] == report["field_frames_in"] == 5 and report["field_factor"] == 1
    assert report["grid"] == "cube-sphere" and report["nonfinite_values"] == 0 and not report["reordered"]
    assert report["first_day"] == 0 and report["last_day"] == frames[-1]["day"]
    meta = view.read_payload(out)["meta"]
    for key in ("frames", "every", "field_frames", "field_every", "field_factor", "items_cap", "max_agents"):
        assert meta[key] == report[key], key


def test_every_frame_decodes_back_to_its_rows(world, page):
    static, frames = world
    got = view.load_page(page[0])
    assert len(got["frames"]) == len(frames)
    for src, dec in zip(frames, got["frames"]):
        assert dec["day"] == src["day"] and len(dec["agents"]) == len(src["agents"])
        for a, b in zip(src["agents"], dec["agents"]):
            assert b[0] == a[0] and b[6] == a[6] and b[8:11] == a[8:11]
            assert all(abs(x - y) <= 6e-5 for x, y in zip(_norm(a[1:4]), b[1:4]))
            assert _close_angle(a[4], b[4], 1e-4)
            assert b[5] == pytest.approx(a[5], rel=1e-6)
            for col in (7, 11, 12):
                assert abs(b[col] - a[col]) <= 0.5 / view.FRACTION_SCALE + 1e-9
        assert [f[3] for f in dec["fires"]] == [f[3] for f in src["fires"]]
        assert [i[3] for i in dec["items"]] == [i[3] for i in src["items"]]
        assert all(i[4] == pytest.approx(j[4], rel=1e-6) for i, j in zip(src["items"], dec["items"]))
        assert all(abs(x - y) <= 6e-5 for i, j in zip(src["items"], dec["items"]) for x, y in zip(_norm(i[:3]), j[:3]))


def test_static_and_fields_decode_exactly(world, page):
    static, frames = world
    got = view.load_page(page[0])
    st = got["static"]
    assert st["elevation_m"] == static["elevation_m"] and st["land"] == static["land"]
    assert all(abs(x - y) <= 6e-5 for p, q in zip(static["centers"], st["centers"]) for x, y in zip(_norm(p), q))
    assert st["planet"]["tidally_locked"] is True and st["planet"]["partial_pressure_pa"]["O2"] == 2500.0
    assert st["actions"] == list(ACTIONS) and st["item_classes"][:4] == ["stone", "ore", "clay", "wood"]
    src_fields = [f for f in frames if "fields" in f]
    assert [f["day"] for f in got["fields"]] == [f["day"] for f in src_fields]
    for src, dec in zip(src_fields, got["fields"]):
        for key in view.FIELD_KEYS:
            assert dec[key] == src["fields"][key], key


def test_page_names_the_compression_and_the_absence_of_meaning(page):
    html = page[0].read_text(encoding="utf-8")
    assert "pattern in miniature" in html and "new_rule" in html and "No meaning is supplied" in html
    meta = view.read_payload(page[0])["meta"]
    assert meta["provenance"]["habitat_radius_m"][0] == "new_rule"
    assert "meaning" in meta["notes"]["meaning"] and "section 1" in meta["notes"]["habitat"]
    for needle in ("prefers-color-scheme", 'data-theme="dark"', "prefers-reduced-motion", "max-width:720px",
                   "aria-label", ":focus-visible", "Relief is exaggerated"):
        assert needle in html, needle


# ------------------------------------------------------------------------------- page contract
def test_template_loads_only_the_two_pinned_scripts():
    html = view.TEMPLATE.read_text(encoding="utf-8")
    assert html.count(view.DATA_TOKEN) == 1
    assert re.findall(r'<script[^>]*\ssrc="([^"]+)"', html) == list(view.ALLOWED_SCRIPTS)
    urls = set(re.findall(r'(?:https?:)?//[A-Za-z0-9.-]+\.[A-Za-z]{2,}[^\s"\'<>)]*', html))
    assert urls == set(view.ALLOWED_SCRIPTS)
    assert not re.search(r'<link\b|@import|url\(|<iframe|<img|fetch\(|XMLHttpRequest|import\(|<object|<embed', html)


def test_built_page_has_no_other_external_resources(page):
    html = page[0].read_text(encoding="utf-8")
    outside = re.sub(r'<script id="globe-data" type="application/json">.*?</script>', "", html, flags=re.S)
    assert re.findall(r'<script[^>]*\ssrc="([^"]+)"', outside) == list(view.ALLOWED_SCRIPTS)
    assert set(re.findall(r'https?://[^\s"\'<>)]+', outside)) == set(view.ALLOWED_SCRIPTS)
    data = re.search(r'<script id="globe-data" type="application/json">(.*?)</script>', html, re.S).group(1)
    assert "<" not in data  # escaped, so the data cannot close its script tag


def _inline_script(html):
    scripts = re.findall(r"<script>(.*?)</script>", html, re.S)
    assert len(scripts) == 1
    return scripts[0]


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_inline_script_is_valid_javascript(page, tmp_path):
    source = tmp_path / "inline.js"
    source.write_text(_inline_script(page[0].read_text(encoding="utf-8")), encoding="utf-8")
    result = subprocess.run([NODE, "--check", str(source)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def _run_codec(tmp_path, data, body):
    """Run the page's own codec section in node on the embedded data; body prints JSON."""
    html = view.TEMPLATE.read_text(encoding="utf-8")
    codec = re.search(r"/\* codec:begin \*/(.*?)/\* codec:end \*/", html, re.S).group(1)
    (tmp_path / "codec.js").write_text(codec, encoding="utf-8")
    (tmp_path / "data.json").write_text(json.dumps(data), encoding="utf-8")
    script = tmp_path / "run.js"
    script.write_text(
        "const fs = require('fs');\n"
        "const CODEC = new Function(fs.readFileSync(process.argv[2], 'utf8') + '\\nreturn CODEC;')();\n"
        "const DATA = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));\n"
        "const A = x => x ? Array.from(x) : null;\n" + body, encoding="utf-8")
    result = subprocess.run([NODE, str(script), str(tmp_path / "codec.js"), str(tmp_path / "data.json")],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_page_codec_decodes_like_python(page, tmp_path):
    data = view.read_payload(page[0])
    js = _run_codec(tmp_path, data, """
const F = CODEC.frame(DATA.frames[2]);
const fd = CODEC.fields(DATA.fields[1], null);
console.log(JSON.stringify({n: F.n, uid: A(F.uid), pos: A(F.pos), heading: A(F.heading), mass: A(F.mass),
  founder: A(F.founder), loud: A(F.loud), vocal: A(F.vocal), k: A(F.k), action: A(F.action), health: A(F.health),
  diet: A(F.diet), fires: A(F.fires.temp), items: A(F.items.cls), T: A(fd.T_k), snow: A(fd.snow),
  elev: A(CODEC.col(DATA.static.elevation_m)), land: A(CODEC.col(DATA.static.land))}));
""")
    py = view.load_page(page[0])
    rows = py["frames"][2]["agents"]
    assert js["n"] == len(rows) and js["uid"] == [r[0] for r in rows]
    assert js["founder"] == [r[6] for r in rows] and js["vocal"] == [r[8] for r in rows]
    assert js["k"] == [r[9] for r in rows] and js["action"] == [r[10] for r in rows]
    for name, col in (("heading", 4), ("mass", 5), ("loud", 7), ("health", 11), ("diet", 12)):
        assert js[name] == pytest.approx([r[col] for r in rows], rel=1e-5, abs=1e-6), name
    assert js["pos"] == pytest.approx([v for r in rows for v in _norm(r[1:4])], abs=2e-6)  # the page normalises
    assert js["fires"] == [f[3] for f in py["frames"][2]["fires"]]
    assert js["items"] == [i[3] for i in py["frames"][2]["items"]]
    assert js["T"] == py["fields"][1]["T_k"] and js["snow"] == py["fields"][1]["snow"]
    assert js["elev"] == py["static"]["elevation_m"] and js["land"] == py["static"]["land"]


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_page_cell_lookup_and_block_map_match_the_globe(tmp_path):
    G = 8
    globe = Globe(G)
    gen = torch.Generator().manual_seed(3)
    pts = torch.nn.functional.normalize(torch.randn(500, 3, generator=gen, dtype=torch.float64), dim=-1)
    pts = torch.cat([pts, globe.centers.double()])
    js = _run_codec(tmp_path, {"pts": pts.tolist()}, """
const cell = CODEC.cubeCell(8);
console.log(JSON.stringify({cells: DATA.pts.map(p => cell(p[0], p[1], p[2])), map: A(CODEC.coarseMap(8, 2))}));
""")
    assert js["cells"] == globe.cell_of(pts).tolist()
    assert js["map"] == view.coarse_index(G, 2)


# --------------------------------------------------------------------------------- size bound
def _fields_bytes(path):
    return sum(len(json.dumps(f, separators=(",", ":"))) for f in view.read_payload(path)["fields"])


def test_fields_coarsen_in_space_on_the_cube_sphere(tmp_path):
    G = 16
    static = make_static(G)
    frames = make_frames(static["cells"], n_frames=8, n_agents=5, field_every=1, items=0, fires=0)
    full = view.build_page(static, frames, tmp_path / "full.html", max_mb=50)
    assert full["field_factor"] == 1
    # 8 field frames is MIN_FIELD_FRAMES: the next step is to average 2 x 2 blocks, which saves about 3/4
    bound = (full["bytes"] - 0.5 * _fields_bytes(tmp_path / "full.html")) / 1e6
    report = view.build_page(static, frames, tmp_path / "small.html", max_mb=bound)
    assert report["fits"] and report["bytes"] <= bound * 1e6
    assert report["field_factor"] == 2 and report["field_frames"] == 8 and report["frames"] == 8
    got = view.load_page(tmp_path / "small.html")
    assert view.read_payload(tmp_path / "small.html")["meta"]["field_G"] == G // 2
    for src, dec in zip(frames, got["fields"]):
        for key in ("T_k", "plant", "soil"):
            means = view.coarsen(src["fields"][key], G, 2)
            assert dec[key] == [int(round(m)) for m in means], key
        assert dec["snow"] == [int(v) for v in view.coarsen(src["fields"]["snow"], G, 2, majority=True)]
        assert len(dec["T_k"]) == 6 * (G // 2) ** 2


def test_coarsen_is_the_block_mean():
    G = 4
    values = list(range(6 * G * G))
    out = view.coarsen(values, G, 2)
    assert len(out) == 6 * 4
    # block (f=1, I=1, J=0) holds cells f G^2 + i G + j for i in {2,3}, j in {0,1}
    cells = [16 + i * 4 + j for i in (2, 3) for j in (0, 1)]
    assert out[1 * 4 + 1 * 2 + 0] == sum(cells) / 4
    # snow: a block is snow when at least half its cells are (ties count as snow)
    flags = [0] * (6 * G * G)
    for c in (0, 1, 4):          # block 0 holds cells 0, 1, 4, 5: three of four
        flags[c] = 1
    for c in (2, 3):             # block 1 holds cells 2, 3, 6, 7: two of four
        flags[c] = 1
    flags[8] = 1                 # block 2 holds cells 8, 9, 12, 13: one of four
    assert view.coarsen(flags, G, 2, majority=True)[:4] == [1.0, 1.0, 0.0, 0.0]


def test_frames_thin_to_fit_and_keep_the_first_and_last(world, tmp_path):
    static, _ = world
    frames = make_frames(static["cells"], n_frames=80, n_agents=200, field_every=40, items=10, fires=1)
    full = view.build_page(static, frames, tmp_path / "full.html", max_mb=50)
    bound = full["bytes"] * 0.4 / 1e6
    report = view.build_page(static, frames, tmp_path / "thin.html", max_mb=bound)
    assert report["fits"] and report["bytes"] <= bound * 1e6
    assert report["bytes"] == (tmp_path / "thin.html").stat().st_size
    assert report["every"] >= 2 and report["frames"] < 80
    days = [f["day"] for f in view.load_page(tmp_path / "thin.html")["frames"]]
    assert days[0] == frames[0]["day"] and days[-1] == frames[-1]["day"]
    assert days[:-1] == [frames[i]["day"] for i in range(0, 80, report["every"])]


def test_points_grid_is_never_coarsened_in_space(tmp_path):
    rnd = random.Random(2)
    centers = [[round(v, 4) for v in _unit(rnd)] for _ in range(6 * 8 * 8)]
    static = make_static(8, centers=centers)
    frames = make_frames(static["cells"], n_frames=20, n_agents=3, field_every=1, items=0, fires=0)
    full = view.build_page(static, frames, tmp_path / "full.html", max_mb=50)
    assert full["grid"] == "points"
    bound = (full["bytes"] - 0.5 * _fields_bytes(tmp_path / "full.html")) / 1e6
    report = view.build_page(static, frames, tmp_path / "small.html", max_mb=bound)
    assert report["fits"] and report["field_factor"] == 1 and report["field_every"] > 1
    assert view.load_page(tmp_path / "small.html")["fields"][0]["T_k"] == frames[0]["fields"]["T_k"]


def test_a_bound_that_cannot_be_met_still_writes_the_thinnest_page(world, tmp_path):
    static, frames = world
    report = view.build_page(static, frames, tmp_path / "tiny.html", max_mb=0.01)
    assert not report["fits"] and (tmp_path / "tiny.html").is_file()
    assert report["frames"] == 2 and report["field_frames"] == 1 and report["items_cap"] == 0
    assert report["items_dropped"] == sum(len(frames[i]["items"]) for i in (0, len(frames) - 1))
    assert view.read_payload(tmp_path / "tiny.html")["meta"]["fits"] is False


def test_items_are_thinned_evenly_and_counted():
    frame = {"day": 0, "items": [[1.0, 0.0, 0.0, k % 7, 1.0] for k in range(100)]}
    enc = view._encode_frame(frame, 10, [0])
    assert enc["i"]["n"] == 10 and enc["i"]["total"] == 100
    assert view.decode_column(enc["i"]["cls"]) == [(10 * k) % 7 for k in range(10)]


# -------------------------------------------------------------------------- input handling
def test_frames_out_of_order_are_sorted_and_reported(world, tmp_path):
    static, frames = world
    shuffled = list(reversed(frames[:6]))
    report = view.build_page(static, shuffled, tmp_path / "s.html")
    assert report["reordered"]
    assert [f["day"] for f in view.load_page(tmp_path / "s.html")["frames"]] == [f["day"] for f in frames[:6]]


def test_non_finite_values_are_counted_and_zeroed(world, tmp_path):
    static, frames = world
    frame = json.loads(json.dumps(frames[0]))
    frame["agents"][0][4] = float("nan")
    frame["agents"][1][5] = float("inf")
    report = view.build_page(static, [frame], tmp_path / "n.html")
    assert report["nonfinite_values"] == 2
    row0, row1 = view.load_page(tmp_path / "n.html")["frames"][0]["agents"][:2]
    assert row0[4] == 0.0 and row1[5] == 0.0


def test_rows_without_diet_and_frames_without_agents(world, tmp_path):
    static, _ = world
    frames = make_frames(static["cells"], n_frames=3, n_agents=4, diet=False)
    frames.append({"day": 99, "sun": [0.0, 1.0, 0.0], "agents": [], "fires": [], "items": []})
    report = view.build_page(static, frames, tmp_path / "d.html")
    got = view.load_page(tmp_path / "d.html")["frames"]
    assert report["frames"] == 4 and all(len(r) == 12 for f in got[:3] for r in f["agents"])
    assert got[-1]["agents"] == [] and got[-1]["sun"] == [0.0, 1.0, 0.0]


def test_no_frames_still_makes_a_page(world, tmp_path):
    static, _ = world
    report = view.build_page(static, [], tmp_path / "e.html")
    assert report["frames"] == 0 and report["fits"] and report["first_day"] is None
    assert view.load_page(tmp_path / "e.html")["frames"] == []


@pytest.mark.parametrize("change, message", [
    (lambda s, f: s.update(schema="life9-globe-v0"), "schema"),
    (lambda s, f: s.update(elevation_m=s["elevation_m"][:-1]), "elevation_m"),
    (lambda s, f: s.pop("sea_level_m"), "sea_level_m"),
    (lambda s, f: f[0]["agents"][0].__delitem__(slice(5, None)), "agents row"),
    (lambda s, f: f[5]["fields"].update(T_k=[280]), "T_k"),
    (lambda s, f: f[0].pop("day"), "day"),
    (lambda s, f: f[0].update(sun=[1.0, 0.0]), "sun"),
])
def test_malformed_input_is_refused(world, tmp_path, change, message):
    static, frames = json.loads(json.dumps(world[0])), json.loads(json.dumps(world[1]))
    change(static, frames)
    with pytest.raises(view.FrameError, match=message):
        view.build_page(static, frames, tmp_path / "bad.html")


def test_columns_take_the_smallest_type():
    assert view.encode_column([0, 1, 255])["t"] == "u1"
    assert view.encode_column([-1, 100])["t"] == "i1"
    assert view.encode_column([0, 40000])["t"] == "u2"
    assert view.encode_column([-5, 40000])["t"] == "i4"
    assert view.encode_column([2 ** 40])["t"] == "f8" and view.decode_column(view.encode_column([2 ** 40])) == [2 ** 40]
    assert view.encode_column([0.5, -0.25], scale=1e4)["t"] == "i2"
    bits = [1, 0, 1, 1, 0, 0, 0, 0, 1]
    assert view.decode_column(view.encode_column(bits, kind="bit")) == bits
    assert view.decode_column(view.encode_column([1.5, -2.25], kind="f4")) == [1.5, -2.25]


def test_cube_sphere_centres_match_the_globe():
    G = 6
    ours = torch.tensor(view.cube_sphere_centers(G), dtype=torch.float64)
    assert torch.allclose(ours, Globe(G).centers.double(), atol=1e-6)
    assert view.grid_kind(G, [[round(v, 4) for v in p] for p in ours.tolist()]) == "cube-sphere"
    assert view.grid_kind(G, list(reversed(ours.tolist()))) == "points"


def test_cli_builds_from_json_files(world, tmp_path, capsys):
    static, frames = world
    (tmp_path / "static.json").write_text(json.dumps(static), encoding="utf-8")
    (tmp_path / "frames.jsonl").write_text("\n".join(json.dumps(f) for f in frames) + "\n", encoding="utf-8")
    out = tmp_path / "cli.html"
    view.main([str(tmp_path / "static.json"), str(tmp_path / "frames.jsonl"), "--out", str(out), "--every", "4"])
    report = json.loads(capsys.readouterr().out)
    assert out.is_file() and report["bytes"] == out.stat().st_size
    assert report["frames"] == len(range(0, len(frames), 4)) + (0 if (len(frames) - 1) % 4 == 0 else 1)


def test_view_module_imports_no_world_code():
    source = Path(view.__file__).read_text(encoding="utf-8")
    assert not re.search(r"^\s*(from|import)\s+[.\w]*world\b", source, re.M)
    assert "torch" not in re.findall(r"^\s*import (\w+)", source, re.M)
