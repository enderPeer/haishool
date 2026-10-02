"""Read-only loopback viewer. No simulation launch or arbitrary filesystem API."""
from __future__ import annotations

from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
from urllib.parse import parse_qs, urlsplit

VIEWER = Path(__file__).resolve().parent / "viewer.html"


def offline_viewer(path, data):
    encoded = json.dumps(data, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c")
    html = VIEWER.read_text(encoding="utf-8").replace("/* EMBED_DATA */", "window.MATRIX_EMBED=" + encoded + ";")
    Path(path).write_text(html, encoding="utf-8")


def read_run(root, name):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in (".", ".."):
        raise ValueError("invalid run name")
    folder = (root / name).resolve()
    if not folder.is_relative_to(root) or not folder.is_dir():
        raise ValueError("run is outside the archive")
    def resource(filename):
        path = (folder / filename).resolve()
        if not path.is_relative_to(folder):
            raise ValueError("archive resource escapes its run")
        return path
    manifest = json.loads(resource("manifest.json").read_text(encoding="utf-8"))
    summary = json.loads(resource("summary.json").read_text(encoding="utf-8"))
    frames = deque(maxlen=500)
    with resource("snapshots.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            try:
                frames.append(json.loads(line))
            except json.JSONDecodeError:
                if line.endswith("\n"):
                    raise
                break  # final line may currently be being written
    return {"manifest": manifest, "summary": summary, "frames": list(frames)}


def serve(root, port=8654):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlsplit(self.path)
            try:
                if parsed.path == "/":
                    body, mime = VIEWER.read_bytes(), "text/html; charset=utf-8"
                elif parsed.path == "/api/runs":
                    names = [p.name for p in sorted(root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
                             if p.is_dir() and p.resolve().is_relative_to(root) and (p / "manifest.json").exists()]
                    body, mime = json.dumps(names).encode(), "application/json"
                elif parsed.path == "/api/run":
                    name = parse_qs(parsed.query).get("name", [""])[0]
                    body, mime = json.dumps(read_run(root, name), allow_nan=False).encode(), "application/json"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(body)
            except (ValueError, OSError, KeyError) as exc:
                self.send_error(400, "Run unavailable or invalid")
        def log_message(self, fmt, *args):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"life8 inspector: http://127.0.0.1:{port}/", flush=True)
    server.serve_forever()
