"""Build new stored cell anatomy from an archived phenotype, then render it.

This adds a developmental model; it does not recover anatomy that world7 never
simulated. The source world archive remains immutable and independently usable.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import struct
import subprocess
import sys

from haishool.inspect_world import ROOT, REQUIRED_SOURCE, canonical, freeze_source, sha, source_files, utc, verify_archive, write_json


def _verify_source(directory, profile, renderer_sha):
    code = (directory / "code").resolve()
    if not code.is_relative_to(directory.resolve()):
        raise ValueError("anatomy code resolves outside the archive")
    required = {*REQUIRED_SOURCE, "haishool/anatomy_run.py", "haishool/evo/anatomy.py"}
    if not isinstance(profile, dict) or not isinstance(profile.get("files"), dict) or not required <= profile["files"].keys():
        raise ValueError("anatomy source profile lacks required files")
    if hashlib.sha256(canonical(profile["files"])).hexdigest() != profile["sha256"]:
        raise ValueError("anatomy source identity mismatch")
    inventory = {p.relative_to(code).as_posix() for p in source_files(code)}
    if inventory != set(profile["files"]):
        raise ValueError("anatomy source inventory mismatch")
    profile_path = (code / "source-profile.json").resolve()
    if not profile_path.is_relative_to(code) or json.loads(profile_path.read_text(encoding="utf-8")) != profile:
        raise ValueError("anatomy frozen profile mismatch")
    for name, expected in profile["files"].items():
        path = (code / name).resolve()
        if not path.is_relative_to(code) or sha(path) != expected:
            raise ValueError(f"anatomy source changed: {name}")
    renderer = (code / "scripts/render_anatomy_blender.py").resolve()
    if not renderer.is_relative_to(code) or sha(renderer) != renderer_sha:
        raise ValueError("renderer source changed")


def select_phenotype(data, planet=1, snapshot=-1, phenotype_id=None):
    if phenotype_id is not None:
        if phenotype_id not in data["phenotypes"]:
            raise ValueError("phenotype is not in this saved world")
        return data["phenotypes"][phenotype_id], {"selection": "explicit phenotype", "phenotype_id": phenotype_id}
    population = next((p for p in data["planets"] if p["id"] == planet), None)
    if population is None or not population.get("snapshots"):
        raise ValueError("this planet has no saved biological specimens")
    try:
        state = population["snapshots"][snapshot]
    except IndexError as exc:
        raise ValueError("snapshot index is outside this saved population") from exc
    if not state.get("groups"):
        raise ValueError("this snapshot has no live specimen groups")
    group = max(state["groups"], key=lambda g: g.get("count", 0))
    source = data["phenotypes"][group["phenotype_id"]]
    return source, {"selection": "most numerous saved phenotype", "planet": planet, "snapshot": state["step"],
                    "generation": state.get("generation"), "represented_count": group.get("count"),
                    "count_unit": group.get("count_unit"), "phenotype_id": source["id"]}


def build(archive: Path, out: Path, *, planet=1, snapshot=-1, phenotype_id=None, seed=85, config=None, timeout=180):
    archive, out = archive.resolve(), out.resolve()
    source_manifest = verify_archive(archive)
    data = json.loads((archive / "inhabitants.json").read_text(encoding="utf-8"))
    phenotype, selection = select_phenotype(data, planet, snapshot, phenotype_id)
    out.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": "anatomy-archive-v1", "status": "preparing", "started_utc": utc(),
                "world_seed": source_manifest["seed"], "source_archive": str(archive),
                "source_world_manifest_sha256": sha(archive / "manifest.json"),
                "source_inhabitants_sha256": sha(archive / "inhabitants.json"), "selection": selection,
                "scope": "New cell-based developmental model conditioned on recorded traits; not an atom-by-atom reconstruction."}
    write_json(out / "manifest.json", manifest)
    try:
        profile = freeze_source(ROOT, out / "code")
        if "haishool/evo/anatomy.py" not in profile["files"]:
            raise FileNotFoundError("anatomy model is missing from the source snapshot")
        renderer = ROOT / "scripts" / "render_anatomy_blender.py"
        renderer_bytes = renderer.read_bytes()
        renderer_sha = hashlib.sha256(renderer_bytes).hexdigest()
        if sha(renderer) != renderer_sha:
            raise RuntimeError("renderer source changed during capture")
        target = out / "code" / "scripts" / renderer.name
        target.parent.mkdir(parents=True)
        target.write_bytes(renderer_bytes)
        manifest.update(status="building", source_profile=profile, renderer_sha256=renderer_sha)
        write_json(out / "manifest.json", manifest)
        write_json(out / "request.json", {"phenotype": phenotype, "seed": seed, "config": config,
                   "source_world": {k: manifest[k] for k in ("world_seed", "source_world_manifest_sha256", "source_inhabitants_sha256", "selection")}})
        env = dict(os.environ, PYTHONPATH=str(out / "code"), PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        with (out / "build.log").open("w", encoding="utf-8") as log:
            subprocess.run([sys.executable, "-m", "haishool.anatomy_run", "_worker", "--directory", str(out)],
                           cwd=out / "code", env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=timeout)
        receipt = json.loads((out / "anatomy-validation.json").read_text(encoding="utf-8"))
        if not receipt["validation"]["ok"]:
            raise ValueError("anatomical validation failed")
        manifest.update(status="complete", completed_utc=utc(), validation=receipt["validation"], dependencies=receipt["dependencies"],
                        files={name: sha(out / name) for name in ("request.json", "anatomy.json", "anatomy-validation.json")})
        write_json(out / "manifest.json", manifest)
        verify(out)
        return manifest
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}", completed_utc=utc())
        write_json(out / "manifest.json", manifest)
        raise


def worker(directory):
    directory = directory.resolve()
    if ROOT.resolve() != (directory / "code").resolve():
        raise RuntimeError("anatomy worker must run from its frozen source directory")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    _verify_source(directory, manifest["source_profile"], manifest["renderer_sha256"])
    from haishool.evo.anatomy import build_anatomy, validate_anatomy
    request = json.loads((directory / "request.json").read_text(encoding="utf-8"))
    if any(request["source_world"].get(k) != manifest[k] for k in ("world_seed", "source_world_manifest_sha256", "source_inhabitants_sha256", "selection")):
        raise ValueError("anatomy world provenance differs from request")
    data = build_anatomy(request["phenotype"], seed=request["seed"], config=request["config"])
    verdict = validate_anatomy(data)
    if not verdict["ok"]:
        raise ValueError(verdict["errors"])
    _verify_source(directory, manifest["source_profile"], manifest["renderer_sha256"])
    write_json(directory / "anatomy.json", data)
    write_json(directory / "anatomy-validation.json", {"validation": verdict,
                "dependencies": {"python": platform.python_version(),
                                 **{k: importlib.metadata.version(k) for k in ("numpy", "scipy")}},
                "content_sha256": hashlib.sha256(canonical(data)).hexdigest()})


def verify(directory):
    directory = directory.resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "anatomy-archive-v1" or manifest.get("status") != "complete":
        raise ValueError("anatomy archive is not complete")
    if not {"request.json", "anatomy.json", "anatomy-validation.json"} <= manifest.get("files", {}).keys():
        raise ValueError("anatomy archive lacks required hashes")
    for name, expected in manifest["files"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or sha(path) != expected:
            raise ValueError(f"anatomy archive mismatch: {name}")
    _verify_source(directory, manifest["source_profile"], manifest["renderer_sha256"])
    receipt = json.loads((directory / "anatomy-validation.json").read_text(encoding="utf-8"))
    data = json.loads((directory / "anatomy.json").read_text(encoding="utf-8"))
    if not receipt["validation"]["ok"] or hashlib.sha256(canonical(data)).hexdigest() != receipt["content_sha256"]:
        raise ValueError("anatomy data/validation mismatch")
    request = json.loads((directory / "request.json").read_text(encoding="utf-8"))
    if any(request["source_world"].get(k) != manifest[k] for k in ("world_seed", "source_world_manifest_sha256", "source_inhabitants_sha256", "selection")):
        raise ValueError("anatomy world provenance differs from request")
    if data.get("schema_version") != "cell-anatomy-v1" or type(data.get("seed")) is not int or data["seed"] != request["seed"]:
        raise ValueError("anatomy schema or development seed mismatch")
    source, phenotype = data["source"], request["phenotype"]
    if source["phenotype_id"] != phenotype.get("id") or source["kind"] != phenotype["kind"] or source["traits"] != phenotype["traits"]:
        raise ValueError("anatomy does not match the selected source phenotype")
    if hashlib.sha256(canonical({k: v for k, v in source.items() if k != "sha256"})).hexdigest() != source["sha256"]:
        raise ValueError("anatomy phenotype identity mismatch")
    if any(data["config"].get(k) != v for k, v in (request["config"] or {}).items()):
        raise ValueError("anatomy configuration differs from request")
    if data["validation"] != receipt["validation"] or manifest["validation"] != receipt["validation"]:
        raise ValueError("anatomy validation reports disagree")
    if hashlib.sha256(canonical({k: v for k, v in data.items() if k != "content_sha256"})).hexdigest() != data["content_sha256"]:
        raise ValueError("anatomy internal content identity mismatch")
    return manifest


def render(directory: Path, blender: Path, *, preview=False, samples=64, threads=8, timeout=1800):
    directory = directory.resolve()
    manifest = verify(directory)
    anatomy_sha = sha(directory / "anatomy.json")
    destination = directory / ("preview" if preview else "render")
    if destination.exists():
        raise FileExistsError(f"render destination already exists: {destination}")
    destination.mkdir()
    arguments = [str(blender), "--background", "--factory-startup", "--threads", str(threads), "--python",
                 str(directory / "code/scripts/render_anatomy_blender.py"), "--", "--anatomy", str(directory / "anatomy.json"),
                 "--out", str(destination), "--samples", str(samples), "--engine", "cycles", "--device", "cpu",
                 "--label", f"World {manifest['world_seed']}"]
    if preview:
        arguments.append("--preview")
    with (destination / "blender.log").open("w", encoding="utf-8") as log:
        subprocess.run(arguments, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=timeout)
    verify(directory)
    receipt = json.loads((destination / ("preview-receipt.json" if preview else "render-receipt.json")).read_text(encoding="utf-8"))
    if receipt["source"]["sha256"] != anatomy_sha:
        raise ValueError("render used a different anatomy input")
    anatomy = json.loads((directory / "anatomy.json").read_text(encoding="utf-8"))
    if receipt["source"]["cell_count"] != len(anatomy["cells"]) or receipt["geometry"]["source_coordinate_transform"] != "identity" or receipt["geometry"]["added_anatomical_features"]:
        raise ValueError("render changed the stored anatomical scope")
    for name in ("blend", "mesh", "png"):
        output = receipt["outputs"][name]
        path = Path(output["path"]).resolve()
        if not path.is_relative_to(destination) or sha(path) != output["sha256"]:
            raise ValueError(f"render output does not match receipt: {name}")
    image = destination / ("anatomy-preview.png" if preview else "anatomy-4k.png")
    if Path(receipt["outputs"]["png"]["path"]).resolve() != image.resolve():
        raise ValueError("render receipt names a different delivered PNG")
    header = image.read_bytes()[:24]
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError("renderer did not produce a PNG")
    dimensions = struct.unpack(">II", header[16:24])
    if dimensions != ((960, 540) if preview else (3840, 2160)):
        raise ValueError(f"unexpected render dimensions: {dimensions}")
    write_json(destination / "verified-output.json", {"source_anatomy_sha256": sha(directory / "anatomy.json"),
               "width": dimensions[0], "height": dimensions[1], "image_sha256": sha(image),
               "files": {p.name: sha(p) for p in destination.iterdir() if p.is_file() and p.name != "verified-output.json"},
               "scope": manifest["scope"]})
    return {"image": str(image), "width": dimensions[0], "height": dimensions[1], "verified": True}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build")
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--planet", type=int, default=1)
    p.add_argument("--snapshot", type=int, default=-1)
    p.add_argument("--phenotype-id")
    p.add_argument("--seed", type=int, default=85)
    p.add_argument("--config", type=Path)
    p = sub.add_parser("render")
    p.add_argument("directory", type=Path)
    p.add_argument("--blender", type=Path, default=Path("C:/Program Files/Blender Foundation/Blender 5.2/blender.exe"))
    p.add_argument("--preview", action="store_true")
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--threads", type=int, default=8)
    p = sub.add_parser("verify")
    p.add_argument("directory", type=Path)
    p = sub.add_parser("_worker")
    p.add_argument("--directory", type=Path, required=True)
    args = vars(ap.parse_args())
    command = args.pop("command")
    if command == "build":
        args["config"] = json.loads(args["config"].read_text(encoding="utf-8")) if args["config"] else None
        result = build(**args)
        print(json.dumps({"status": result["status"], "validation": result["validation"]}))
    elif command == "render":
        print(json.dumps(render(**args)))
    elif command == "verify":
        print(json.dumps({"verified": True, "world": verify(**args)["world_seed"]}))
    else:
        worker(**args)


if __name__ == "__main__":
    main()
