"""A chat window for the life8 visitor model (docs/life8/visitors.md).

The visitor model knows no English: it reads one inhabitant's situation in the experience format
(``q life8 see <feature> <value> ... want <outcome>. a``) and answers with one action word. This
server lets a person build such a situation, or draw a real one from the held-out dev planets,
and shows the model's probability for every action next to what the inhabitant really did.

    python -m haishool.visitor_chat --checkpoint runs/visitor/export/visitor-6x384-fp16.pt \
        --data data --port 8660

It lives outside ``haishool/life8`` on purpose: a new module there would change
``life8.checkpoint.source_identity()`` and stop running life8 checkpoints from resuming.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import threading
from collections import Counter, defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from haishool import student
from haishool.life8 import experience as E

PAGE = Path(__file__).with_name("visitor_chat.html")
#: features that every line carries; the others are left out when absent (nothing heard, no held tool, ...)
ALWAYS = {"food_direction", "food_near", "object_near", "neighbor_near", "energy_band", "tool_available",
          "health_band", "inventory_count", "neighbor_familiar", "neighbor_last_exchange", "bloom_seen",
          "neighbor_balance"}


def split_see(see: str) -> dict:
    """``{feature: value}`` of the ``see`` part of a line (a value may be several digit tokens)."""
    names, out, current, buf = set(E.FEATURE_ORDER), {}, None, []
    for token in see.split():
        if token in names:
            if current:
                out[current] = " ".join(buf)
            current, buf = token, []
        else:
            buf.append(token)
    if current:
        out[current] = " ".join(buf)
    return out


def prompt_of(features: dict, want: str) -> str:
    see = " ".join(f"{name} {features[name]}" for name in E.FEATURE_ORDER if features.get(name))
    return f"q life8 see {see} want {want}. a" if see else f"q life8 see want {want}. a"


def read_text_prompt(text: str) -> tuple[dict, str]:
    """A typed line: a full prompt, a full experience line, or just ``<feature> <value> ...``."""
    t = " ".join(text.lower().replace(".", " ").split())
    for head in ("q life8 see ", "life8 see ", "see "):
        if t.startswith(head):
            t = t[len(head):]
    want = "thrive"
    if " want " in f" {t} ":
        t, _, rest = f" {t} ".partition(" want ")
        words = rest.split()
        if words and words[0] in E.OUTCOMES:
            want = words[0]
    return split_see(t.strip()), want


class Model:
    def __init__(self, checkpoint: Path, device: str = "cpu", threads: int = 4):
        torch = student.torch
        if torch is None:
            raise RuntimeError("torch is not installed")
        torch.set_num_threads(threads)
        self.torch = torch
        self.device = device
        self.model, self.vocab = student.load(checkpoint, device)
        self.ctx = self.model.config.ctx
        self.actions = [a for a in E.ACTION_WORDS if a in self.vocab.stoi]
        self.index = torch.tensor([self.vocab.stoi[a] for a in self.actions], device=device)
        self.lock = threading.Lock()

    def ask(self, prompt: str) -> dict:
        words = [student.EOS, *student.tokens(prompt)]
        unknown = [w for w in words if w not in self.vocab.stoi]
        ids = self.vocab.encode(words)[-self.ctx:]
        torch = self.torch
        with self.lock, torch.no_grad():
            logits, _ = self.model(torch.tensor([ids], device=self.device))
            probs = torch.softmax(logits[0, -1].float(), dim=-1)
        on_actions = probs[self.index]
        mass = float(on_actions.sum())
        ranked = sorted(zip(self.actions, (on_actions / max(mass, 1e-12)).tolist()), key=lambda x: -x[1])
        top_any = torch.topk(probs, 3)
        return {"prompt": prompt, "unknown_words": unknown,
                "actions": [{"action": a, "p": round(p, 4)} for a, p in ranked],
                "answer": ranked[0][0],
                "mass_on_actions": round(mass, 4),
                "free_top": [{"token": self.vocab.itos[i], "p": round(float(p), 4)}
                             for p, i in zip(top_any.values.tolist(), top_any.indices.tolist())]}


def load_examples(data: Path, lines: int, keep: int, seed: int = 20261002):
    """Feature values seen in the dev split (for the menus) and a sample of real dev situations."""
    values, present, n = defaultdict(Counter), Counter(), 0
    rng, sample = random.Random(seed), []
    path = data / f"{E.TOPIC}-dev.txt"
    with path.open(encoding="utf-8") as stream:
        for line in itertools.islice(stream, lines):
            try:
                see, want, action = E.parse_line(line.strip())
            except ValueError:
                continue
            n += 1
            feats = split_see(see)
            for name, value in feats.items():
                values[name][value] += 1
                present[name] += 1
            if len(sample) < keep:
                sample.append((feats, want, action))
            elif rng.random() < keep / n:
                sample[rng.randrange(keep)] = (feats, want, action)
    schema = [{"name": name, "always": name in ALWAYS, "share": round(present[name] / max(n, 1), 3),
               "values": [v for v, _ in values[name].most_common()]} for name in E.FEATURE_ORDER]
    return schema, sample, n


def make_handler(model: Model, schema, sample, info):
    page = PAGE.read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet
            pass

        def _send(self, code, body: bytes, kind="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", kind + ("; charset=utf-8" if "json" in kind or "html" in kind else ""))
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, data, code=200):
            self._send(code, json.dumps(data).encode("utf-8"))

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                return self._send(200, page, "text/html")
            if self.path == "/api/schema":
                return self._json({"features": schema, "outcomes": list(E.OUTCOMES), "actions": model.actions,
                                   "info": info})
            if self.path.startswith("/api/example"):
                feats, want, action = random.choice(sample)
                prompt = prompt_of(feats, want)
                return self._json({"features": feats, "want": want, "native": action, **model.ask(prompt)})
            return self._json({"error": "not found"}, 404)

        def do_POST(self):
            if self.path != "/api/ask":
                return self._json({"error": "not found"}, 404)
            try:
                size = min(int(self.headers.get("Content-Length", 0)), 20000)
                body = json.loads(self.rfile.read(size) or b"{}")
                if body.get("text"):
                    feats, want = read_text_prompt(str(body["text"]))
                else:
                    feats = {k: str(v) for k, v in (body.get("features") or {}).items()
                             if k in E.FEATURE_ORDER and v not in (None, "")}
                    want = body.get("want", "thrive")
                if want not in E.OUTCOMES:
                    raise ValueError(f"want must be one of {E.OUTCOMES}")
                bad = sorted(set(feats) - set(E.FEATURE_ORDER))
                if bad:
                    raise ValueError(f"unknown features: {bad}")
                result = model.ask(prompt_of(feats, want))
                return self._json({"features": feats, "want": want, **result})
            except (ValueError, json.JSONDecodeError) as error:
                return self._json({"error": str(error)}, 400)

    return Handler


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True, help="folder with predict_life8_experience-dev.txt")
    ap.add_argument("--eval", type=Path, help="eval.json of the model, shown on the page")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8660)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--scan-lines", type=int, default=300000)
    ap.add_argument("--examples", type=int, default=5000)
    args = ap.parse_args(argv)
    model = Model(args.checkpoint, args.device, args.threads)
    schema, sample, n = load_examples(args.data, args.scan_lines, args.examples)
    info = {"checkpoint": args.checkpoint.name, "layers": model.model.config.n_layer if hasattr(model.model.config, "n_layer") else None,
            "scanned_dev_lines": n, "examples": len(sample)}
    if args.eval and args.eval.exists():
        ev = json.loads(args.eval.read_text(encoding="utf-8"))
        row = ev["splits"].get(f"{E.TOPIC}/sealed", {})
        info["sealed"] = {"model": row.get("model", {}).get("all", {}).get("accuracy"),
                          "baseline": row.get("baseline_majority_per_observation", {}).get("all", {}).get("accuracy"),
                          "unseen_model": row.get("model", {}).get("observation_unseen", {}).get("accuracy"),
                          "unseen_baseline": row.get("baseline_majority_per_observation", {}).get("observation_unseen", {}).get("accuracy")}
    server = ThreadingHTTPServer((args.host, args.port), make_handler(model, schema, sample, info))
    print(json.dumps({"serving": f"http://{args.host}:{args.port}/", **info}), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
