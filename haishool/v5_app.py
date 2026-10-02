"""Separate v5 preview: route maths/science before the legacy fact/link parser.

The neural model supplies the answer. A rule gate checks that answer afterward;
the reference result is returned separately and never substituted for the model.
The released app and its public services are not modified by this wrapper.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import threading
from pathlib import Path
from typing import Callable

from haishool.student import tokens
from haishool.truth import is_finite, parse_num
from haishool.truth.loop import GATE_ERROR, gate_verdict

NUMBER = re.compile(r"(?<!\S)(?:minus )?[0-9](?: [0-9])*(?: point(?: [0-9])+)?(?: e (?:minus )?[0-9](?: [0-9])*)?(?!\S)")


def display_dense(text: str, prompt: str = "") -> str:
    """Render digit tokens without inventing facts or rounding the model output."""
    # Balance answers are coefficient lists, with one coefficient per digit;
    # interpreting their adjacent tokens as one number changes the answer.
    if prompt.startswith("balance ") and text.split() and all(w in "0123456789" and len(w) == 1 for w in text.split()):
        return ", ".join(text.split())
    def number(match):
        value = match.group()
        if not is_finite(parse_num(value)):
            return value
        return value.replace("minus ", "-").replace("point", ".").replace(" ", "")
    return NUMBER.sub(number, text).replace("_", " ")


def unsupported(message: str, prompt: str = "") -> dict:
    return {"route": "unsupported", "topic": "science", "answer": message,
            "query": f"q {prompt}. a" if prompt else "", "dense": "",
            "verification": {"status": "unavailable", "ok": None,
                             "expected": None, "expected_display": None, "reason": message}}


def make_handler(ask_model: Callable[[str, int], str], legacy_ask: Callable[[str], dict],
                 vocab_words: set[str], context: int = 256, router=None):
    if router is None:
        from haishool.science_routes import route
        router = route
    lock = threading.Lock()

    def ask(question: str) -> dict:
        question = question.strip()
        if not question or len(question) > 1200:
            return unsupported("Enter one question of at most 1,200 characters.")
        routed = router(question)
        if routed is None:
            if len(question) > 300:
                return unsupported("Please shorten this fact or relationship question to 300 characters.")
            with lock:
                result = dict(legacy_ask(question))
            result.update(route="legacy", topic="facts_and_links",
                          verification={"status": "unavailable", "ok": None, "expected": None,
                                        "expected_display": None, "reason": "No maths/science gate applies to this fact or link question."})
            return result
        if routed.error:
            return unsupported(routed.error, routed.prompt)
        prompt = f"q {routed.prompt}. a"
        words = ["<eos>", *tokens(prompt)]
        missing = sorted(set(words) - vocab_words)
        if missing:
            return unsupported("This model has not learned these input words: " + ", ".join(missing[:10]) + ".", routed.prompt)
        room = context - len(words)
        if room < 8:
            return unsupported("This question is too long for the model's context. Please use fewer inputs.", routed.prompt)
        with lock:
            dense = ask_model(prompt, room)
            verdict = gate_verdict(routed.gate, routed.prompt, dense)
        status = "unavailable" if GATE_ERROR in verdict.reason else "accepted" if verdict.ok else "rejected"
        return {"route": "science", "topic": routed.topic, "label": routed.label,
                "answer": display_dense(dense, routed.prompt) if dense else "The model returned no answer.",
                "query": prompt, "dense": dense,
                "verification": {"status": status, "ok": verdict.ok if status != "unavailable" else None,
                                 "expected": verdict.expected,
                                 "expected_display": display_dense(verdict.expected, routed.prompt) if verdict.expected is not None else None,
                                 "reason": verdict.reason},
                "output_limit_reached": len(tokens(dense)) >= room}
    return ask


def make_app(handler, model_info: dict, page: str):
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse
    from pydantic import BaseModel, Field
    from haishool.science_routes import EXAMPLES
    # Publish the body type in module globals for FastAPI's postponed-annotation resolver.
    class V5Ask(BaseModel):
        question: str = Field(min_length=1, max_length=1200)
    globals()["V5Ask"] = V5Ask
    app = FastAPI(title="Haishool v5 preview")

    @app.get("/", response_class=HTMLResponse)
    def home():
        return HTMLResponse(page, headers={"Cache-Control": "no-store"})

    @app.get("/api/info")
    def info():
        return model_info

    @app.get("/api/examples")
    def examples():
        return {"examples": EXAMPLES}

    @app.post("/api/ask")
    def ask(body: V5Ask):
        return handler(body.question)
    return app


def build(model_path: Path, records_path: Path, device: str = "cpu", hops: Path | None = None,
          page_path: Path | None = None):
    import torch
    from haishool import app as legacy
    from haishool.student import generate, load
    from haishool.science_routes import registry
    torch.set_num_threads(4)
    # Reuse the released routes as an unchanged fallback. A second small model
    # instance avoids reaching into the legacy function's private closure/state.
    old = legacy.build(model_path, records_path, device, hops)
    endpoints = {r.path: r.endpoint for r in old.routes if hasattr(r, "endpoint")}
    model, vocab = load(model_path, device)
    def ask_model(prompt, room):
        return generate(model, vocab, prompt, max_new=room, device=device)
    handler = make_handler(ask_model, lambda q: endpoints["/api/ask"](legacy.Ask(question=q)),
                           set(vocab.stoi), model.config.ctx)
    sha = hashlib.sha256()
    with model_path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(block)
    info = {**endpoints["/api/info"](), "version": "v5", "science_ready": True,
            "science_topics": [g.topic for g in registry()], "checkpoint_sha256": sha.hexdigest()}
    page_path = page_path or Path(__file__).resolve().parents[1] / "docs" / "v5-chat.html"
    return make_app(handler, info, page_path.read_text(encoding="utf-8"))


def main():
    import uvicorn
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--hops", type=Path)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8652)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--page", type=Path)
    args = ap.parse_args()
    uvicorn.run(build(args.model, args.records, args.device, args.hops, args.page), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
