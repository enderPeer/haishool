"""The 3D viewer page (haishool/life8/view3d.py): built from a tiny living-world run archive."""
import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest

from haishool.life8 import view3d
from haishool.life8.cli import run_world
from haishool.life8.config import LIVING, Config
from haishool.life8.world import World

TICKS = 100
SAMPLE = 2


@pytest.fixture(scope="module")
def archive(tmp_path_factory):
    """A 100-tick living-preset run (small population), snapshots every 2 ticks."""
    values = {**LIVING, "population": 10, "food_patches": 12, "initial_objects": 9, "log": "compact"}
    world = World.create(seed=3, config=Config(**values))
    out = tmp_path_factory.mktemp("view3d") / "run"
    run_world(world, out, steps=TICKS, sample_every=SAMPLE, checkpoint_every=TICKS)
    return out


@pytest.fixture(scope="module")
def page(archive, tmp_path_factory):
    out = tmp_path_factory.mktemp("page") / "world.html"
    report = view3d.build(archive, out, log=lambda _: None)
    return out, report


def embedded(path):
    html = Path(path).read_text(encoding="utf-8")
    match = re.search(r'<script id="life8-data" type="application/json">(.*?)</script>', html, re.S)
    assert match, "data script tag missing"
    return html, json.loads(match.group(1))


def archive_ids(archive):
    ids = set()
    with (archive / "snapshots.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            ids.update(agent["id"] for agent in json.loads(line)["agents"])
    with (archive / "events.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            if event.get("type") == "birth":
                ids.update((event["agent"], event["parent"]))
    return ids


def test_embedded_json_parses_and_has_every_snapshot_as_a_frame(archive, page):
    path, report = page
    _, data = embedded(path)
    ticks = [json.loads(line)["tick"] for line in (archive / "snapshots.jsonl").read_text(encoding="utf-8").splitlines()]
    assert ticks == list(range(0, TICKS + 1, SAMPLE))
    frames = data["frames"]
    assert frames["t"] == ticks
    assert data["meta"]["frames"] == report["frames"] == len(ticks) == TICKS // SAMPLE + 1
    for key in ("a", "f", "o"):
        assert len(frames[key]) == len(ticks)
    assert all(len(row) % 6 == 0 for row in frames["a"])
    assert data["meta"]["title"] == "Haishool Living Worlds"
    # the living preset has predators; the replay must match the archive exactly to be used
    assert data["meta"]["predator_source"] == "replay", data["meta"]["predator_note"]
    assert len(frames["p"]) == len(ticks) and all(len(row) == 4 * LIVING["predators"] for row in frames["p"])


def test_every_individual_in_the_frames_exists_in_the_archive(archive, page):
    _, data = embedded(page[0])
    known = archive_ids(archive)
    framed = {row[i] for row in data["frames"]["a"] for i in range(0, len(row), 6)}
    assert framed and framed <= known
    table = {row[0] for row in data["ind"]}
    assert framed <= table <= known
    columns = data["ind_columns"]
    for row in data["ind"]:
        record = dict(zip(columns, row))
        assert record["founder"] in table
        assert record["parent"] is None or record["parent"] in table
    for event in data["events"]:
        if event[1] in (data["kinds"].index("birth"), data["kinds"].index("death")):
            assert event[2] in known
    assert {call[2] for call in data["calls"]} <= known


def test_page_stays_under_the_size_bound_and_resamples_to_fit(archive, page, tmp_path):
    path, report = page
    assert report["bytes"] == path.stat().st_size < view3d.MAX_BYTES
    full = view3d.build(archive, tmp_path / "full.html", replay="off", log=lambda _: None)
    sparse = view3d.build(archive, tmp_path / "sparse.html", every=8, replay="off", log=lambda _: None)
    assert full["every"] == 1 and sparse["bytes"] < full["bytes"]
    small = tmp_path / "small.html"
    bound = (full["bytes"] + sparse["bytes"]) // 2  # reachable only by sampling more sparsely
    shrunk = view3d.build(archive, small, max_bytes=bound, replay="off", log=lambda _: None)
    assert shrunk["bytes"] <= bound and small.stat().st_size <= bound
    assert 1 < shrunk["every"] <= 8 and shrunk["frames"] < full["frames"] and shrunk["heard_detail"]
    _, data = embedded(small)
    assert data["frames"]["t"][0] == 0 and data["frames"]["t"][-1] == TICKS


def test_every_and_max_ticks_select_frames(archive, tmp_path):
    out = tmp_path / "sparse.html"
    report = view3d.build(archive, out, every=8, max_ticks=50, replay="off", log=lambda _: None)
    _, data = embedded(out)
    assert data["frames"]["t"] == [0, 8, 16, 24, 32, 40, 48, 50]
    assert report["frames"] == 8 and report["last_tick"] == 50
    assert all(event[0] <= 50 for event in data["events"])
    assert data["meta"]["predator_source"] == "events"


def test_template_loads_only_the_two_pinned_scripts():
    html = view3d.TEMPLATE.read_text(encoding="utf-8")
    assert html.count(view3d.DATA_TOKEN) == 1
    sources = re.findall(r'<script[^>]*\ssrc="([^"]+)"', html)
    assert sources == list(view3d.ALLOWED_SCRIPTS)
    urls = set(re.findall(r'(?:https?:)?//[A-Za-z0-9.-]+\.[A-Za-z]{2,}[^\s"\'<>)]*', html))
    assert urls == set(view3d.ALLOWED_SCRIPTS)
    assert not re.search(r'<link\b|@import|url\(|<iframe|<img|fetch\(|XMLHttpRequest|import\(', html)


def test_built_page_has_no_other_external_resources(page):
    html, _ = embedded(page[0])
    outside = re.sub(r'<script id="life8-data" type="application/json">.*?</script>', "", html, flags=re.S)
    assert re.findall(r'<script[^>]*\ssrc="([^"]+)"', outside) == list(view3d.ALLOWED_SCRIPTS)
    assert set(re.findall(r'https?://[^\s"\'<>)]+', outside)) == set(view3d.ALLOWED_SCRIPTS)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_inline_script_is_valid_javascript(page, tmp_path):
    html, _ = embedded(page[0])
    scripts = re.findall(r'<script>(.*?)</script>', html, re.S)
    assert len(scripts) == 1
    source = tmp_path / "inline.js"
    source.write_text(scripts[0], encoding="utf-8")
    result = subprocess.run(["node", "--check", str(source)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_cli_build_writes_the_page(archive, tmp_path, capsys):
    out = tmp_path / "cli.html"
    view3d.main(["build", str(archive), "--out", str(out), "--every", "4", "--replay", "off"])
    report = json.loads(capsys.readouterr().out)
    assert out.is_file() and report["frames"] == TICKS // 4 + 1 and report["bytes"] == out.stat().st_size
