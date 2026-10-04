"""The v3 viewer page (haishool/life9/v3/view3.py + viewer3.html) on synthetic life9-v3-view-1 data.

The fixture is generated here (static3 + frames); world3.py is not used.
"""
from __future__ import annotations

import ast
import base64
import json
import math
from pathlib import Path
import random
import re
import shutil
import subprocess
import zlib

import pytest
import torch

from haishool.life9.planet.globe import Globe, tangent_basis
from haishool.life9.v3 import view3

ROOT = Path(__file__).resolve().parents[3]
NODE = shutil.which("node")
CLASSES = ("stone", "ore", "clay", "wood", "organic", "metal", "product")


# ------------------------------------------------------------------------------------- fixtures
def make_static(G=8, patches=2, n=16, L=2048.0, seed=5, gene_columns=None, sea_patch=True):
    """A cube-sphere globe with `patches` square patches of n x n cells (one partly under the sea)."""
    rnd = random.Random(seed)
    centers = [[round(v, 4) for v in p] for p in Globe(G).centers.tolist()]
    elevation = [int(300 + 900 * math.sin(3 * p[0]) * math.cos(2 * p[2]) + rnd.uniform(-60, 60)) for p in centers]
    sea = 320.0
    land = [1 if e > sea else 0 for e in elevation]
    cells = [c for c in range(len(centers)) if land[c]][:patches]
    pats = []
    for p, cell in enumerate(cells):
        base = sea - 4.0 if (sea_patch and p == 1) else float(elevation[cell])
        elev = []
        for ix in range(n):
            for iy in range(n):
                z = base + 12 * math.sin(2 * math.pi * ix / n) * math.cos(2 * math.pi * iy / n) + rnd.uniform(-1, 1)
                elev.append(int(round(z)))
        pond = [1 if rnd.random() < 0.04 else 0 for _ in range(n * n)]
        pats.append({"index": p, "cell": cell, "center": centers[cell], "L": L, "n": n, "elevation_m": elev,
                     "pond": pond})
    static = {"schema": view3.SCHEMA, "seed": seed,
              "planet": {"radius_m": 5674430.0, "gravity_m_s2": 7.5441, "surface_pressure_pa": 88971.0,
                         "tidally_locked": False, "year_s": 1.035e7, "star_teff_k": 4300.0, "t_surface_k": 270.8,
                         "ocean_fraction": 0.513, "frozen_mean": True, "p_co2_pa": 12.0},
              "globe": {"G": G, "centers": centers, "elevation_m": elevation, "land": land, "sea_level_m": sea},
              "patches": pats, "species": ["basalt", "granite", "flint", "wood", "meat"],
              "item_classes": list(CLASSES), "bouts": 4}
    if gene_columns:
        static["gene_columns"] = list(gene_columns)
    return static


def make_frames(static, days=3, bouts=4, bodies=40, seed=11, field_every=4, items=30, fires=2, genes=0):
    """Bodies that persist (same uids) and move a little between bouts, plus births and deaths."""
    rnd = random.Random(seed)
    frames, uid = [], 1000
    pop = {}
    for ps in static["patches"]:
        L = ps["L"]
        rows = []
        for _ in range(bodies):
            rows.append([uid, rnd.uniform(0, L), rnd.uniform(0, L), rnd.uniform(0, 2 * math.pi),
                         10 ** rnd.uniform(math.log10(2), math.log10(200)), rnd.randrange(6) + 10 * ps["index"]])
            uid += 1
        pop[ps["index"]] = rows
    C = len(static["globe"]["centers"])
    for k in range(days * bouts):
        day, bout = divmod(k, bouts)
        ang = 2 * math.pi * k / bouts
        frame = {"day": day, "bout": bout, "sun": [math.cos(ang), math.sin(ang), 0.1], "patches": []}
        if k % field_every == 0:
            frame["global"] = {"T_k": [rnd.randrange(220, 310) for _ in range(C)],
                               "ice": [rnd.randrange(2) for _ in range(C)]}
        for ps in static["patches"]:
            L, n, rows = ps["L"], ps["n"], pop[ps["index"]]
            for r in rows:  # walk a little (wrapping), turn a little
                r[1] = (r[1] + rnd.uniform(-20, 20)) % L
                r[2] = (r[2] + rnd.uniform(-20, 20)) % L
                r[3] = (r[3] + rnd.uniform(-0.3, 0.3)) % (2 * math.pi)
            if k:  # one death, one birth
                rows.pop(rnd.randrange(len(rows)))
                parent = rows[rnd.randrange(len(rows))]
                rows.append([uid, parent[1], parent[2], rnd.uniform(0, 2 * math.pi), parent[4] * 0.4, parent[5]])
                uid += 1
            body_rows = []
            for r in rows:
                row = [r[0], round(r[1], 3), round(r[2], 3), round(r[3], 4), round(r[4], 4), r[5],
                       rnd.randrange(2, 33), round(rnd.random(), 3), round(rnd.uniform(-1, 1), 3),
                       round(rnd.random(), 3), rnd.randrange(8), round(rnd.random() * 0.5, 3)]
                row += [round(rnd.uniform(0, 1), 4) for _ in range(genes)]
                body_rows.append(row)
            entry = {"index": ps["index"], "bodies": body_rows,
                     "items": [[round(rnd.uniform(0, L), 2), round(rnd.uniform(0, L), 2), rnd.randrange(len(CLASSES)),
                                round(10 ** rnd.uniform(-3, 1), 5)] for _ in range(items)],
                     "fires": [[round(rnd.uniform(0, L), 2), round(rnd.uniform(0, L), 2), rnd.randrange(700, 1500)]
                               for _ in range(fires)]}
            if k % field_every == 0:
                entry["plants"] = [rnd.randrange(256) for _ in range(n * n)]
                entry["water"] = [rnd.randrange(256) if rnd.random() < 0.05 else 0 for _ in range(n * n)]
            frame["patches"].append(entry)
        frames.append(frame)
    return frames


@pytest.fixture(scope="module")
def world():
    static = make_static(gene_columns=("size_kg", "fur_m"))
    return static, make_frames(static, genes=2)


@pytest.fixture(scope="module")
def page(world, tmp_path_factory):
    static, frames = world
    out = tmp_path_factory.mktemp("view3") / "view3.html"
    return out, view3.build_page(static, frames, out)


def _sorted_bodies(entry):
    return sorted(entry["bodies"], key=lambda r: r[0])


def _close_angle(a, b, tol):
    d = (a - b) % (2 * math.pi)
    return min(d, 2 * math.pi - d) <= tol


# --------------------------------------------------------------------------------- round trip
def test_report_matches_the_written_page(world, page):
    out, report = page
    static, frames = world
    assert report["bytes"] == out.stat().st_size and report["fits"] and report["bytes"] <= 15e6
    assert report["frames"] == report["frames_in"] == len(frames) and report["every"] == 1
    assert report["field_records"] == report["field_records_in"] == 3 * (2 + 2 * 2) and report["field_factor"] == 1
    assert report["grid"] == "cube-sphere" and report["nonfinite_values"] == 0 and not report["reordered"]
    assert report["wrapped_positions"] == 0 and report["duplicate_uids"] == 0 and report["item_class_out_of_range"] == 0
    assert report["first"] == [0, 0] and report["last"] == [2, 3]
    assert report["scripts"] == list(view3.ALLOWED_SCRIPTS)
    meta = view3.read_payload(out)["meta"]
    for key in ("frames", "every", "field_records", "field_every", "field_factor", "items_cap", "max_bodies"):
        assert meta[key] == report[key], key
    assert meta["max_bodies"] == 40 and meta["k_max"] <= 32 and len(meta["top_lineages"]) == 8


def test_bodies_items_and_fires_decode_back(world, page):
    static, frames = world
    got = view3.load_page(page[0])
    assert len(got["frames"]) == len(frames)
    for src, dec in zip(frames, got["frames"]):
        assert (dec["day"], dec["bout"]) == (src["day"], src["bout"])
        for entry, pdec in zip(src["patches"], dec["patches"]):
            ps = static["patches"][entry["index"]]
            step = 1 / view3.position_scale(ps["L"])
            rows = _sorted_bodies(entry)
            assert len(pdec["bodies"]) == len(rows)
            for a, b in zip(rows, pdec["bodies"]):
                assert b[0] == a[0] and b[5] == a[5] and b[6] == a[6] and b[10] == a[10]
                assert abs(b[1] - a[1]) <= step / 2 + 1e-9 and abs(b[2] - a[2]) <= step / 2 + 1e-9
                assert _close_angle(a[3], b[3], 0.5 / view3.HEADING_SCALE + 1e-9)
                assert b[4] == pytest.approx(a[4], rel=3e-4)  # log steps of 1/4000
                for col in (7, 9, 11):
                    assert abs(b[col] - a[col]) <= 0.5 / view3.FRACTION_SCALE + 1e-9
                assert abs(b[8] - a[8]) <= 0.5 / view3.SIGNED_SCALE + 1e-9
                assert b[12:] == pytest.approx(a[12:], rel=1e-6)  # gene columns as float32
            assert [i[2] for i in pdec["items"]] == [i[2] for i in entry["items"]]
            assert all(j[3] == pytest.approx(i[3], rel=3e-4) for i, j in zip(entry["items"], pdec["items"]))
            assert all(abs(j[0] - i[0]) <= step / 2 + 1e-9 for i, j in zip(entry["items"], pdec["items"]))
            assert [f[2] for f in pdec["fires"]] == [f[2] for f in entry["fires"]]


def test_static_and_fields_decode_exactly(world, page):
    static, frames = world
    got = view3.load_page(page[0])
    st = got["static"]
    assert st["globe"]["elevation_m"] == static["globe"]["elevation_m"] and st["globe"]["land"] == static["globe"]["land"]
    assert all(abs(x - y) <= 6e-5 for p, q in zip(static["globe"]["centers"], st["globe"]["centers"])
               for x, y in zip(p, q))
    for src, dec in zip(static["patches"], st["patches"]):
        assert dec["elevation_m"] == src["elevation_m"] and dec["pond"] == src["pond"]
        assert dec["index"] == src["index"] and dec["n"] == src["n"] and dec["L"] == src["L"]
    assert st["planet"]["frozen_mean"] is True and st["planet"]["p_co2_pa"] == 12.0
    assert st["item_classes"] == list(CLASSES) and st["gene_columns"] == ["size_kg", "fur_m"]
    src_global = [f for f in frames if "global" in f]
    for key in view3.GLOBAL_FIELDS:
        recs = got["fields"]["global"][key]
        assert [(r["day"], r["bout"]) for r in recs] == [(f["day"], f["bout"]) for f in src_global]
        assert [r["values"] for r in recs] == [f["global"][key] for f in src_global]
    for slot, ps in enumerate(static["patches"]):
        for key in view3.PATCH_FIELDS:
            src = [next(e for e in f["patches"] if e["index"] == ps["index"])[key] for f in frames
                   if key in next(e for e in f["patches"] if e["index"] == ps["index"])]
            assert [r["values"] for r in got["fields"]["patch"][slot][key]] == src


def test_patch_basis_matches_the_globe(world):
    static, _ = world
    centers = torch.tensor([ps["center"] for ps in static["patches"]], dtype=torch.float64)
    east, north = tangent_basis(centers)
    for k, ps in enumerate(static["patches"]):
        e, n, up = view3.tangent_basis(ps["center"])
        assert e == pytest.approx(east[k].tolist(), abs=1e-9) and n == pytest.approx(north[k].tolist(), abs=1e-9)


def test_page_states_that_nothing_is_preinstalled(page):
    html = page[0].read_text(encoding="utf-8")
    assert "Nothing is pre-installed" in html and "Colours are display only" in html
    assert "do not know them" in html and "No meaning is supplied" in html
    assert "The colours of lineages and item classes" in html
    meta = view3.read_payload(page[0])["meta"]
    assert "not known to the bodies" in meta["notes"]["display"] and "section 0" in meta["notes"]["supplied"]
    assert all(tag == "display" for tag, _ in meta["provenance"].values())
    for needle in ("prefers-color-scheme", 'data-theme="dark"', "prefers-reduced-motion", "max-width:760px",
                   "aria-label", ":focus-visible", "aria-live", "keydown", "Heights are drawn at true scale",
                   "True size", "real size", "true scale", "--lin1", "--cls0", "--ch0"):
        assert needle in html, needle


# ------------------------------------------------------------------------------- page contract
def test_template_loads_only_the_two_pinned_scripts():
    html = view3.TEMPLATE.read_text(encoding="utf-8")
    assert html.count(view3.DATA_TOKEN) == 1
    assert re.findall(r'<script[^>]*\ssrc="([^"]+)"', html) == list(view3.ALLOWED_SCRIPTS)
    urls = set(re.findall(r'(?:https?:)?//[A-Za-z0-9.-]+\.[A-Za-z]{2,}[^\s"\'<>)]*', html))
    assert urls == set(view3.ALLOWED_SCRIPTS)
    assert not re.search(r'<link\b|@import|url\(|<iframe|<img|fetch\(|XMLHttpRequest|import\(|<object|<embed|'
                         r'new Worker|importScripts|WebSocket|EventSource|sendBeacon', html)


def test_built_page_has_no_other_external_resources(page):
    html = page[0].read_text(encoding="utf-8")
    outside = re.sub(r'<script id="view3-data" type="application/json">.*?</script>', "", html, flags=re.S)
    assert re.findall(r'<script[^>]*\ssrc="([^"]+)"', outside) == list(view3.ALLOWED_SCRIPTS)
    assert set(re.findall(r'https?://[^\s"\'<>)]+', outside)) == set(view3.ALLOWED_SCRIPTS)
    data = re.search(r'<script id="view3-data" type="application/json">(.*?)</script>', html, re.S).group(1)
    assert "<" not in data  # escaped, so the data cannot close its script tag
    assert len(re.findall(r"<script\b", html)) == 4  # data, the two pinned scripts, the inline viewer


def test_no_section_9_words_in_the_viewer_files():
    """The package-wide forbidden-mechanism grep (PLANET-V3-SPEC section 9) must stay clean with the viewer in it."""
    words = ("innate", "forage", "actions", "background_drink", "time_budget", "pedigree", "relatedness", "seed_floor",
             "lifespan", "maturity", "reward", "outcome", "kleiber", "allometr", "temperature0", "litter_mass",
             "gestation", "cooldown", "softmax")
    for path in (ROOT / "haishool/life9/v3/view3.py", view3.TEMPLATE):
        text = path.read_text(encoding="utf-8").lower()
        for word in words:
            assert word not in text, (path.name, word)


def test_the_viewer_imports_no_world_code():
    tree = ast.parse((ROOT / "haishool/life9/v3/view3.py").read_text(encoding="utf-8"))
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mods.add(("." * node.level) + (node.module or ""))
        elif isinstance(node, ast.Import):
            mods |= {a.name for a in node.names}
    assert mods <= {"__future__", "argparse", "array", "base64", "datetime", "json", "math", "pathlib", "sys", "zlib",
                    "..planet.view", "..planet.materials"}, mods


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
    """Run the page's own codec section in node on data; body prints JSON."""
    html = view3.TEMPLATE.read_text(encoding="utf-8")
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
def test_page_inflater_matches_zlib(tmp_path):
    """The page's own inflater (no library) on stored, fixed-code and dynamic-code streams and long matches."""
    rnd = random.Random(4)
    cases = {
        "stored": zlib.compress(bytes(rnd.randrange(256) for _ in range(70_000)), 0),   # several stored blocks
        "random": zlib.compress(bytes(rnd.randrange(256) for _ in range(5_000)), 9),
        "fixed": zlib.compress(b"abcabcabcd" * 3, 9),
        "dynamic": zlib.compress(bytes(rnd.choice(b"aaaabbbcdx") for _ in range(50_000)), 9),
        "runs": zlib.compress(bytes(300_000) + bytes(range(256)) * 300, 9),           # 258-byte matches, far distances
        "levels": zlib.compress(bytes(int(127 + 120 * math.sin(i / 37.0)) for i in range(40_000)), 6),
        "empty": zlib.compress(b"", 9),
    }
    raw = {k: zlib.decompress(v) for k, v in cases.items()}
    data = {k: base64.b64encode(v).decode("ascii") for k, v in cases.items()}
    js = _run_codec(tmp_path, data, """
const out = {};
for (const k of Object.keys(DATA)) {
  const bin = Buffer.from(DATA[k], 'base64');
  const got = CODEC.inflate(new Uint8Array(bin));
  out[k] = Buffer.from(got).toString('base64');
}
let bad = 0;
try { CODEC.inflate(new Uint8Array([1, 2, 3])); } catch (e) { bad = 1; }
out.refused = bad;
console.log(JSON.stringify(out));
""")
    assert js.pop("refused") == 1
    for k, v in raw.items():
        assert base64.b64decode(js[k]) == v, k


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_page_codec_decodes_like_python(page, tmp_path):
    data = view3.read_payload(page[0])
    js = _run_codec(tmp_path, data, """
const out = {};
for (const slot of [0, 1]) {
  const F = CODEC.patchFrame(DATA.frames[5].p[slot]);
  out[slot] = {n: F.n, uid: A(F.uid), x: A(F.x), y: A(F.y), h: A(F.h), m: A(F.m), lin: A(F.lin), k: A(F.k),
               mo: A(F.mo), th: A(F.th), ld: A(F.ld), v: A(F.v), dm: A(F.dm), g0: A(F.genes[0]), linCount: F.linCount,
               ic: A(F.items.c), im: A(F.items.m), ft: A(F.fires.t), total: F.items.total};
}
const plants = A(CODEC.col(DATA.fields.patch[1].plants[1].v));
const ice = A(CODEC.col(DATA.fields.global.ice[2].v));
const elev = A(CODEC.col(DATA.static.patches[1].elevation_m));
const pond = A(CODEC.col(DATA.static.patches[0].pond));
const land = A(CODEC.col(DATA.static.globe.land));
console.log(JSON.stringify({p: out, plants, ice, elev, pond, land, find: CODEC.find(out[0].uid ? Float64Array.from(out[0].uid) : null, out[0].uid[3])}));
""")
    py = view3.load_page(page[0])
    for slot in (0, 1):
        rows = py["frames"][5]["patches"][slot]["bodies"]
        got = js["p"][str(slot)]
        assert got["n"] == len(rows) and got["uid"] == [r[0] for r in rows]
        assert got["lin"] == [r[5] for r in rows] and got["k"] == [r[6] for r in rows] and got["v"] == [r[10] for r in rows]
        assert got["linCount"] == len({r[5] for r in rows})
        for name, col in (("x", 1), ("y", 2), ("h", 3), ("m", 4), ("mo", 7), ("th", 8), ("ld", 9), ("dm", 11)):
            assert got[name] == pytest.approx([r[col] for r in rows], rel=1e-12, abs=1e-12), name
        assert got["g0"] == pytest.approx([r[12] for r in rows], rel=1e-7)
        items = py["frames"][5]["patches"][slot]["items"]
        assert got["ic"] == [i[2] for i in items] and got["im"] == pytest.approx([i[3] for i in items], rel=1e-12)
        assert got["ft"] == [f[2] for f in py["frames"][5]["patches"][slot]["fires"]]
    assert js["plants"] == py["fields"]["patch"][1]["plants"][1]["values"]
    assert js["ice"] == py["fields"]["global"]["ice"][2]["values"]
    assert js["elev"] == py["static"]["patches"][1]["elevation_m"] and js["pond"] == py["static"]["patches"][0]["pond"]
    assert js["land"] == py["static"]["globe"]["land"] and js["find"] == 3


# --------------------------------------------------------------------------------- size bound
def test_full_scale_page_fits_the_bound(tmp_path):
    """Four 128 x 128 patches, 2,048 bodies and 1,500 items each, plants and water every 4 bouts, the globe at G = 32:
    the page fits 15 MB, keeps every body of the frames it keeps, and says how it thinned."""
    static = make_static(G=32, patches=4, n=128, seed=8)
    frames = make_frames(static, days=12, bouts=4, bodies=2048, items=1500, fires=4, seed=3)
    report = view3.build_page(static, frames, tmp_path / "full.html", max_mb=15)
    assert report["fits"] and report["bytes"] <= 15e6 and report["bytes"] == (tmp_path / "full.html").stat().st_size
    assert report["max_bodies"] == 2048
    got = view3.load_page(tmp_path / "full.html")
    assert [f["day"] * 4 + f["bout"] for f in got["frames"]][0] == 0
    assert (got["frames"][-1]["day"], got["frames"][-1]["bout"]) == (11, 3)
    by_key = {(f["day"], f["bout"]): f for f in frames}
    for dec in got["frames"]:
        src = by_key[(dec["day"], dec["bout"])]
        assert [len(p["bodies"]) for p in dec["patches"]] == [len(e["bodies"]) for e in src["patches"]]
    meta = got["meta"]
    assert meta["frames"] == report["frames"] and meta["items_cap"] == report["items_cap"]


def test_frames_thin_to_fit_and_keep_the_first_and_last(tmp_path):
    static = make_static(patches=2, n=16)
    frames = make_frames(static, days=20, bouts=4, bodies=300, items=20, fires=1, field_every=40)
    full = view3.build_page(static, frames, tmp_path / "full.html", max_mb=50)
    bound = full["bytes"] * 0.4 / 1e6
    report = view3.build_page(static, frames, tmp_path / "thin.html", max_mb=bound)
    assert report["fits"] and report["bytes"] <= bound * 1e6 and report["bytes"] == (tmp_path / "thin.html").stat().st_size
    assert report["every"] >= 2 and report["frames"] < 80
    keys = [(f["day"], f["bout"]) for f in view3.load_page(tmp_path / "thin.html")["frames"]]
    want = [(frames[i]["day"], frames[i]["bout"]) for i in range(0, 80, report["every"])]
    assert keys[0] == (0, 0) and keys[-1] == (19, 3) and keys[:len(want)] == want


def test_patch_fields_coarsen_in_space(tmp_path):
    static = make_static(patches=2, n=32)
    frames = make_frames(static, days=2, bouts=4, bodies=3, items=0, fires=0, field_every=1)
    full = view3.build_page(static, frames, tmp_path / "full.html", max_mb=50)
    assert full["field_factor"] == 1
    payload = view3.read_payload(tmp_path / "full.html")
    patch_bytes = sum(len(json.dumps(r, separators=(",", ":"))) for rec in payload["fields"]["patch"]
                      for recs in rec.values() for r in recs)
    bound = (full["bytes"] - 0.5 * patch_bytes) / 1e6
    report = view3.build_page(static, frames, tmp_path / "small.html", max_mb=bound)
    assert report["fits"] and report["field_factor"] >= 2 and report["field_records"] == report["field_records_in"]
    got = view3.load_page(tmp_path / "small.html")
    f = report["field_factor"]
    src = next(e for e in frames[0]["patches"] if e["index"] == 0)["plants"]
    rec = got["fields"]["patch"][0]["plants"][0]
    assert rec["factor"] == f and rec["values"] == view3.coarsen_field(src, 32, f)
    assert len(rec["values"]) == (32 // f) ** 2


def test_coarsen_field_is_the_block_mean():
    n = 4
    values = list(range(n * n))  # cell = ix n + iy
    out = view3.coarsen_field(values, n, 2)
    # block (1, 0) holds cells ix in {2, 3}, iy in {0, 1}: 8, 9, 12, 13
    assert out == [round((0 + 1 + 4 + 5) / 4), round((2 + 3 + 6 + 7) / 4), round((8 + 9 + 12 + 13) / 4),
                   round((10 + 11 + 14 + 15) / 4)]


def test_a_bound_that_cannot_be_met_still_writes_the_thinnest_page(world, tmp_path):
    static, frames = world
    report = view3.build_page(static, frames, tmp_path / "tiny.html", max_mb=0.01)
    assert not report["fits"] and (tmp_path / "tiny.html").is_file()
    assert report["frames"] == 2 and report["items_cap"] == 0
    got = view3.load_page(tmp_path / "tiny.html")
    assert all(len(recs) <= 1 for recs in got["fields"]["global"].values())
    assert [len(p["bodies"]) for p in got["frames"][-1]["patches"]] == [len(e["bodies"]) for e in frames[-1]["patches"]]
    assert report["items_dropped"] == sum(len(e["items"]) for i in (0, len(frames) - 1) for e in frames[i]["patches"])
    assert view3.read_payload(tmp_path / "tiny.html")["meta"]["fits"] is False


def test_items_are_thinned_evenly_and_counted():
    entry = {"index": 0, "bodies": [], "items": [[1.0, 2.0, k % 7, 1.0] for k in range(100)], "fires": []}
    info = {"L": 2048.0, "ps": view3.position_scale(2048.0)}
    counts = {"wrapped_positions": 0, "duplicate_uids": 0, "item_class_out_of_range": 0}
    enc = view3._encode_patch_frame(entry, info, 10, 0, 7, counts, [0])
    assert enc["it"]["n"] == 10 and enc["it"]["total"] == 100
    assert view3.decode_column(enc["it"]["c"]) == [(10 * k) % 7 for k in range(10)]


# -------------------------------------------------------------------------- input handling
def test_frames_out_of_order_are_sorted_and_reported(world, tmp_path):
    static, frames = world
    shuffled = list(reversed(frames[:6]))
    report = view3.build_page(static, shuffled, tmp_path / "s.html")
    assert report["reordered"]
    got = view3.load_page(tmp_path / "s.html")["frames"]
    assert [(f["day"], f["bout"]) for f in got] == [(f["day"], f["bout"]) for f in frames[:6]]


def test_bad_values_are_counted_not_hidden(world, tmp_path):
    static, frames = world
    frame = json.loads(json.dumps(frames[0]))
    rows = frame["patches"][0]["bodies"]
    rows[0][3] = float("nan")                 # heading
    rows[1][4] = -1.0                         # mass (log of a non-positive number)
    rows[2][1] = 2048.0 + 5.0                 # x beyond L: wrapped
    rows[3][2] = -3.0                         # y below 0: wrapped
    frame["patches"][0]["items"][0][2] = 42   # a class the page has no name for
    frame["patches"][0]["bodies"].append(list(rows[4]))  # a duplicate uid
    report = view3.build_page(static, [frame], tmp_path / "n.html")
    assert report["nonfinite_values"] == 2 and report["wrapped_positions"] == 2
    assert report["item_class_out_of_range"] == 1 and report["duplicate_uids"] == 1
    dec = {r[0]: r for r in view3.load_page(tmp_path / "n.html")["frames"][0]["patches"][0]["bodies"]}
    assert dec[rows[0][0]][3] == 0.0
    assert dec[rows[2][0]][1] == pytest.approx(5.0, abs=0.02) and dec[rows[3][0]][2] == pytest.approx(2045.0, abs=0.02)


def test_bad_values_are_counted_once_when_the_page_is_thinned(tmp_path):
    static = make_static(patches=2, n=16)
    frames = make_frames(static, days=20, bouts=4, bodies=300, items=400, fires=1, field_every=40)
    frames[0]["patches"][0]["bodies"][0][1] = -1.0                 # wrapped (the first frame is always kept)
    frames[0]["patches"][0]["bodies"][1][4] = float("nan")          # a non-finite mass counts once
    frames[-1]["patches"][1]["items"][0][2] = 99                    # an unnamed class, in the last frame
    full = view3.build_page(static, frames, tmp_path / "full.html", max_mb=50)
    assert (full["wrapped_positions"], full["nonfinite_values"], full["item_class_out_of_range"]) == (1, 1, 1)
    thin = view3.build_page(static, frames, tmp_path / "thin.html", max_mb=full["bytes"] * 0.3 / 1e6)
    assert thin["every"] > 1 and thin["frames"] < full["frames"]
    assert (thin["wrapped_positions"], thin["nonfinite_values"]) == (1, 1)
    assert thin["item_class_out_of_range"] == (1 if thin["items_cap"] > 0 else 0)  # the even spread keeps item 0


def test_frames_without_some_patches_and_without_bodies(world, tmp_path):
    static, frames = world
    f0 = json.loads(json.dumps(frames[0]))
    f0["patches"] = f0["patches"][:1]
    f1 = {"day": 9, "bout": 0, "sun": [0.0, 1.0, 0.0], "patches": [{"index": 1, "bodies": [], "items": [], "fires": []}]}
    report = view3.build_page(static, [f0, f1], tmp_path / "m.html")
    got = view3.load_page(tmp_path / "m.html")["frames"]
    assert report["frames"] == 2 and got[0]["patches"][1] is None and got[1]["patches"][0] is None
    assert got[1]["patches"][1]["bodies"] == [] and got[1]["sun"] == [0.0, 1.0, 0.0]


def test_no_frames_still_makes_a_page(world, tmp_path):
    static, _ = world
    report = view3.build_page(static, [], tmp_path / "e.html")
    assert report["frames"] == 0 and report["fits"] and report["first"] is None
    assert view3.load_page(tmp_path / "e.html")["frames"] == []


def test_the_page_is_deterministic_without_the_stamp(world, tmp_path):
    static, frames = world
    view3.build_page(static, frames, tmp_path / "a.html", stamp=False)
    view3.build_page(static, frames, tmp_path / "b.html", stamp=False)
    assert (tmp_path / "a.html").read_bytes() == (tmp_path / "b.html").read_bytes()
    assert view3.read_payload(tmp_path / "a.html")["meta"]["built_utc"] is None


@pytest.mark.parametrize("change, message", [
    (lambda s, f: s.update(schema="life9-v3-view-0"), "schema"),
    (lambda s, f: s["globe"].update(elevation_m=s["globe"]["elevation_m"][:-1]), "elevation_m"),
    (lambda s, f: s["globe"].pop("sea_level_m"), "sea_level_m"),
    (lambda s, f: s["planet"].pop("radius_m"), "radius_m"),
    (lambda s, f: s["patches"][0].update(elevation_m=[0] * 5), "elevation_m"),
    (lambda s, f: s["patches"][1].update(index=0), "twice"),
    (lambda s, f: f[0]["patches"][0]["bodies"][0].__delitem__(slice(8, None)), "bodies row"),
    (lambda s, f: f[0]["patches"][0].update(plants=[1, 2]), "plants"),
    (lambda s, f: f[0]["patches"][0].update(index=7), "not in static"),
    (lambda s, f: f[0]["global"].update(T_k=[280]), "T_k"),
    (lambda s, f: f[0].pop("day"), "day"),
    (lambda s, f: f[0].update(sun=[1.0, 0.0]), "sun"),
    (lambda s, f: f[0].update(bout=-1), "bout"),
])
def test_malformed_input_is_refused(world, tmp_path, change, message):
    static, frames = json.loads(json.dumps(world[0])), json.loads(json.dumps(world[1][:2]))
    change(static, frames)
    with pytest.raises(view3.FrameError, match=message):
        view3.build_page(static, frames, tmp_path / "bad.html")


def test_columns_take_the_smallest_type_and_round_trip():
    assert view3.encode_column([0, 1, 255])["t"] == "u1"
    assert view3.encode_column([-1, 100])["t"] == "i1"
    assert view3.encode_column([0, 40000])["t"] == "u2"
    assert view3.encode_column([-5, 40000])["t"] == "i4"
    big = view3.encode_column([2 ** 40])
    assert big["t"] == "f8" and view3.decode_column(big) == [2 ** 40]
    uids = list(range(10_000, 10_500, 3))
    enc = view3.encode_column(uids, delta=True)
    assert enc["t"] == "i2" and enc["d"] == 1 and view3.decode_column(enc) == uids  # first value, then steps of 3
    masses = [0.002, 1.5, 200.0, 7.3e4]
    dec = view3.decode_column(view3.encode_column(masses, scale=4000.0, offset=-6.0, log=True))
    assert dec == pytest.approx(masses, rel=3e-4)
    bits = [1, 0, 1, 1, 0, 0, 0, 0, 1] * 20
    enc = view3.encode_column(bits, kind="bit")
    assert view3.decode_column(enc) == bits
    zeros = view3.encode_column([0] * 5000)
    assert zeros.get("z") == 1 and len(zeros["b"]) < 200 and view3.decode_column(zeros) == [0] * 5000


def test_cli_writes_the_page(world, tmp_path, capsys):
    static, frames = world
    (tmp_path / "static.json").write_text(json.dumps(static), encoding="utf-8")
    (tmp_path / "frames.jsonl").write_text("\n".join(json.dumps(f) for f in frames[:3]), encoding="utf-8")
    report = view3.main([str(tmp_path / "static.json"), str(tmp_path / "frames.jsonl"), "--out", str(tmp_path / "c.html")])
    assert report["frames"] == 3 and (tmp_path / "c.html").is_file()
    assert json.loads(capsys.readouterr().out)["bytes"] == report["bytes"]
