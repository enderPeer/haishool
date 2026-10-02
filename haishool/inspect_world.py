"""Run a world with a frozen engine and export inspectable biological specimens.

    python -m haishool.inspect_world run --seed 85 --out runs/inhabitants/world-85
    python -m haishool.inspect_world serve --root runs/inhabitants --port 8653

This optional observer never changes simulation dynamics or the existing training
lines. Each run saves its source, data, dependency versions, validation, and a
standalone viewer. A server starts each simulation in a fresh frozen subprocess.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_VERSION = 1
ARCHIVE_FILES = ("request.json", "receipt.json", "world.json", "inhabitants.json", "viewer.html")
REQUIRED_SOURCE = ("haishool/inspect_world.py", "haishool/evo/world7.py",
                   "haishool/evo/inhabitants.py", "docs/inhabitants.html")


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def plain(value):
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"unsupported archive value: {type(value).__name__}")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=plain).encode("utf-8")


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=plain) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_files(root):
    paths = list((root / "haishool").rglob("*.py"))
    paths += list((root / "data" / "truth-v5").glob("*.json"))
    paths += list((root / "data" / "truth-v5").glob("*.jsonl"))
    paths += [root / "docs" / "inhabitants.html"]
    return sorted(paths)


def freeze_source(root: Path, destination: Path):
    """Capture stable bytes first, then write them; concurrent edits never enter mid-run."""
    root, destination = root.resolve(), destination.resolve()
    for _ in range(3):
        paths = source_files(root)
        if any(not path.resolve().is_relative_to(root) for path in paths):
            raise ValueError("source file resolves outside the source root")
        blobs = {str(p.relative_to(root)).replace("\\", "/"): p.read_bytes() for p in paths}
        hashes = {p: hashlib.sha256(b).hexdigest() for p, b in blobs.items()}
        if paths == source_files(root) and all(sha(root / p) == h for p, h in hashes.items()):
            break
    else:
        raise RuntimeError("Simulation source is changing during capture. Retry when this edit finishes.")
    for name in REQUIRED_SOURCE:
        if name not in blobs:
            raise FileNotFoundError(f"render-enabled source is missing {name}")
    destination.mkdir(parents=True, exist_ok=False)
    for name, data in blobs.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    profile = {"sha256": hashlib.sha256(canonical(hashes)).hexdigest(), "files": hashes}
    write_json(destination / "source-profile.json", profile)
    return profile


def bake_viewer(template: str, data: dict, *, live=False):
    """JSON text cannot terminate its script element, even in user-provided metadata."""
    values = {"inhabitant-data": data, "inhabitant-config": {"live": live}}
    for identifier, value in values.items():
        encoded = json.dumps(value, ensure_ascii=True, allow_nan=False, default=plain).replace("<", "\\u003c")
        pattern = r'(<script\b[^>]*\bid=[\"\']' + identifier + r'[\"\'][^>]*>).*?(</script>)'
        template, count = re.subn(pattern, lambda m: m[1] + encoded + m[2], template, count=1, flags=re.S)
        if count != 1:
            raise ValueError(f"viewer is missing data element {identifier}")
    return template


def verify_archive(directory: Path):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("archive manifest must be an object")
    if manifest.get("archive_version") != ARCHIVE_VERSION:
        raise ValueError("unsupported archive version")
    if manifest.get("status") != "complete":
        raise ValueError("archive is not complete")
    if not isinstance(manifest.get("files"), dict) or not set(ARCHIVE_FILES) <= manifest["files"].keys():
        raise ValueError("archive is missing required file hashes")
    for name, expected in manifest["files"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory.resolve()) or sha(path) != expected:
            raise ValueError(f"archive file mismatch: {name}")
    profile = manifest["source_profile"]
    if not isinstance(profile, dict):
        raise ValueError("source profile must be an object")
    code = (directory / "code").resolve()
    if not code.is_relative_to(directory):
        raise ValueError("archived code resolves outside the archive")
    if not isinstance(profile.get("files"), dict) or not set(REQUIRED_SOURCE) <= profile["files"].keys():
        raise ValueError("source profile is missing required source files")
    captured_names = {str(path.relative_to(code)).replace("\\", "/") for path in source_files(code)}
    if captured_names != set(profile["files"]):
        raise ValueError("archived source inventory differs from profile")
    if hashlib.sha256(canonical(profile["files"])).hexdigest() != profile["sha256"]:
        raise ValueError("source profile identity mismatch")
    profile_path = (code / "source-profile.json").resolve()
    if not profile_path.is_relative_to(code):
        raise ValueError("archived source profile resolves outside the code directory")
    if json.loads(profile_path.read_text(encoding="utf-8")) != profile:
        raise ValueError("archived source profile differs from manifest")
    for name, expected in profile["files"].items():
        path = (directory / "code" / name).resolve()
        if not path.is_relative_to((directory / "code").resolve()) or sha(path) != expected:
            raise ValueError(f"archived source mismatch: {name}")
    data = json.loads((directory / "inhabitants.json").read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != "inhabitants-v1":
        raise ValueError("unsupported inhabitant schema")
    request = json.loads((directory / "request.json").read_text(encoding="utf-8"))
    world = json.loads((directory / "world.json").read_text(encoding="utf-8"))
    seed = manifest.get("seed")
    if not all(isinstance(item, dict) for item in (request, world, data.get("source"))):
        raise ValueError("archive seed provenance records must be objects")
    if type(seed) is not int or not 0 <= seed <= 10 ** 9 or not all(
            type(item) is int and item == seed for item in (request.get("seed"), world.get("seed"), data["source"].get("world_seed"))):
        raise ValueError("archive seed provenance differs between records")
    if hashlib.sha256(canonical(data)).hexdigest() != manifest["inhabitants_content_sha256"]:
        raise ValueError("inhabitant content mismatch")
    return manifest


def _worker(request: Path, directory: Path):
    """Only the frozen child imports the evolving simulator."""
    from haishool.evo import inhabitants, world7
    req = json.loads(request.read_text(encoding="utf-8"))
    profile = json.loads((ROOT / "source-profile.json").read_text(encoding="utf-8"))
    began = time.monotonic()
    world = world7.run(req["seed"], **req["params"])
    verdict = world7.conserved(world)
    if not verdict.ok:
        raise ValueError(f"world conservation/consistency check failed: {verdict.reason}")
    data = inhabitants.export_inhabitants(world, steps=req["steps"])
    write_json(directory / "inhabitants.json", data)
    levels = {name: {"sim": r.sim, "seed": r.seed, "params": r.params, "summary": r.summary}
              for name, r in world.levels.items()}
    attempts = {name: [{"seed": r.seed, "params": r.params, "summary": r.summary} for r in runs]
                for name, runs in world.attempts.items()}
    write_json(directory / "world.json", {"seed": world.seed, "params": world.params, "eras": world.steps,
                                          "summary": world.summary, "levels": levels, "attempts": attempts})
    dependencies = {name: importlib.metadata.version(name) for name in ("numpy", "scipy")}
    # Imported Python modules are from this immutable snapshot, not the editable checkout.
    if any(sha(ROOT / name) != value for name, value in profile["files"].items()):
        raise RuntimeError("frozen source changed during the run")
    receipt = {"validation": {"world_conserved": True, "reason": verdict.reason},
               "dependencies": {"python": platform.python_version(), **dependencies},
               "platform": platform.platform(), "seconds": round(time.monotonic() - began, 3),
               "inhabitants_content_sha256": hashlib.sha256(canonical(data)).hexdigest()}
    write_json(directory / "receipt.json", receipt)


def run_world(seed: int, out: Path, *, params=None, steps=None, source_root: Path = ROOT, timeout=180):
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 10 ** 9:
        raise ValueError("seed must be an integer from 0 to 1,000,000,000")
    if params is not None and not isinstance(params, dict):
        raise ValueError("world parameters must be a JSON object")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    manifest = {"archive_version": ARCHIVE_VERSION, "status": "preparing", "seed": seed, "started_utc": utc()}
    write_json(out / "manifest.json", manifest)
    try:
        profile = freeze_source(source_root, out / "code")
        manifest.update(status="running", source_profile=profile)
        write_json(out / "manifest.json", manifest)
        write_json(out / "request.json", {"seed": seed, "params": params or {}, "steps": steps})
        env = dict(os.environ, PYTHONPATH=str(out / "code"), PYTHONDONTWRITEBYTECODE="1",
                   OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        with (out / "run.log").open("w", encoding="utf-8") as log:
            subprocess.run([sys.executable, "-m", "haishool.inspect_world", "_worker",
                            "--request", str(out / "request.json"), "--directory", str(out)],
                           cwd=out / "code", env=env, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=timeout)
        receipt = json.loads((out / "receipt.json").read_text(encoding="utf-8"))
        data = json.loads((out / "inhabitants.json").read_text(encoding="utf-8"))
        template = (out / "code" / "docs" / "inhabitants.html").read_text(encoding="utf-8")
        (out / "viewer.html").write_text(bake_viewer(template, data), encoding="utf-8")
        files = {name: sha(out / name) for name in ARCHIVE_FILES}
        manifest.update(status="complete", completed_utc=utc(), files=files, **receipt)
        write_json(out / "manifest.json", manifest)
        verify_archive(out)
        return {"run_id": out.name, "directory": str(out), "seed": seed, "manifest": manifest, "data": data}
    except Exception as exc:
        manifest.update(status="failed", completed_utc=utc(), error=f"{type(exc).__name__}: {exc}")
        write_json(out / "manifest.json", manifest)
        raise


def serve(root: Path, host="127.0.0.1", port=8653):
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    state = {"busy": False, "latest": None}
    candidates = sorted(root.glob("*/manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in candidates:
        try:
            verify_archive(p.parent)
            state["latest"] = p.parent
            break
        except (ValueError, OSError, KeyError):
            continue

    class Handler(BaseHTTPRequestHandler):
        def send(self, code, content, content_type="application/json; charset=utf-8"):
            raw = content if isinstance(content, bytes) else json.dumps(content, allow_nan=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/api/info":
                self.send(200, {"service": "Haishool inhabitants", "live": True, "busy": state["busy"],
                                "archive_version": ARCHIVE_VERSION, "latest_run": state["latest"].name if state["latest"] else None})
            elif path == "/":
                data = json.loads((state["latest"] / "inhabitants.json").read_text(encoding="utf-8")) if state["latest"] else {
                    "schema_version": "inhabitants-v1", "mapping_version": "phenotype-display-v1", "source": {},
                    "limitations": [], "genotypes": {}, "phenotypes": {}, "planets": [],
                    "empty_reason": "Run a world to inspect its saved inhabitants."}
                template = (ROOT / "docs" / "inhabitants.html").read_text(encoding="utf-8")
                self.send(200, bake_viewer(template, data, live=True).encode("utf-8"), "text/html; charset=utf-8")
            elif match := re.fullmatch(r"/runs/([a-zA-Z0-9_-]+)/(viewer\.html|inhabitants\.json|world\.json|manifest\.json)", path):
                target = root / match[1] / match[2]
                if target.is_file() and target.resolve().is_relative_to(root):
                    ctype = "text/html; charset=utf-8" if target.suffix == ".html" else "application/json; charset=utf-8"
                    self.send(200, target.read_bytes(), ctype)
                else:
                    self.send(404, {"error": "Archive not found"})
            else:
                self.send(404, {"error": "Not found"})

        def do_POST(self):
            if urlsplit(self.path).path != "/api/run":
                self.send(404, {"error": "Not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4096:
                    raise ValueError("Request must be 1 to 4,096 bytes")
                request = json.loads(self.rfile.read(length))
                if not isinstance(request, dict) or set(request) != {"seed"}:
                    raise ValueError("Supply one world seed")
                seed = request["seed"]
                if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 10 ** 9:
                    raise ValueError("Seed must be an integer from 0 to 1,000,000,000")
            except (ValueError, TypeError) as exc:
                self.send(400, {"error": str(exc)})
                return
            if not lock.acquire(blocking=False):
                self.send(409, {"error": "A world is already running. Please wait for it to finish."})
                return
            try:
                state["busy"] = True
                run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + f"-seed-{seed}"
                result = run_world(seed, root / run_id)
                state["latest"] = Path(result["directory"])
                self.send(200, result["data"])
            except Exception as exc:
                self.send(500, {"error": f"World run failed: {type(exc).__name__}. The failed archive and log were retained."})
            finally:
                state["busy"] = False
                lock.release()

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"http://{host}:{server.server_port}/", flush=True)
    server.serve_forever()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", type=Path)
    p.add_argument("--params", type=Path, help="Optional world parameters as a JSON object")
    p.add_argument("--step", type=int, action="append", help="Save only these snapshot indices; repeatable")
    p.add_argument("--timeout", type=float, default=180)
    p = sub.add_parser("serve")
    p.add_argument("--root", type=Path, default=Path("runs/inhabitants"))
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8653)
    p = sub.add_parser("verify")
    p.add_argument("directory", type=Path)
    p = sub.add_parser("_worker")
    p.add_argument("--request", type=Path, required=True)
    p.add_argument("--directory", type=Path, required=True)
    args = ap.parse_args()
    if args.command == "run":
        out = args.out or Path("runs/inhabitants") / (dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + f"-seed-{args.seed}")
        result = run_world(args.seed, out, params=json.loads(args.params.read_text(encoding="utf-8")) if args.params else None,
                           steps=args.step, timeout=args.timeout)
        print(json.dumps({"seed": result["seed"], "archive": result["directory"],
                          "viewer": str(Path(result["directory"]) / "viewer.html"), "verified": True}))
    elif args.command == "serve":
        serve(args.root, args.host, args.port)
    elif args.command == "verify":
        manifest = verify_archive(args.directory.resolve())
        print(json.dumps({"verified": True, "seed": manifest["seed"], "source_sha256": manifest["source_profile"]["sha256"]}))
    else:
        _worker(args.request, args.directory)


if __name__ == "__main__":
    main()
