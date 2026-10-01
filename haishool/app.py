"""Test chat: English question -> dense query -> student model -> rule translator -> English.

    python -m haishool.app --model runs/full/student.pt --records data/records.jsonl --port 8650

The page shows the English answer and, below it, the dense query and the model's raw dense
answer, so it is always visible what the model itself produced.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from haishool.student import answer, describe, generate, load
from haishool.links import Graph, load_graph, node_in, path_text, relation_for, relation_sentence, two_nodes, view_for
from haishool.search import Index, resolve
from haishool.translate import SELF, alias_index, record_text, sentence, subject

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Haishool</title>
<style>
:root{--bg:#f5f4ef;--fg:#202924;--muted:#66706a;--card:#fffdf8;--line:#d9d6cc;--accent:#2f6b4f}
@media (prefers-color-scheme:dark){:root{--bg:#161a18;--fg:#e8ebe6;--muted:#9aa49d;--card:#1f2421;--line:#333a36;--accent:#7cc4a0}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif}
main{max-width:760px;margin:0 auto;padding:24px 16px 120px}h1{font-size:22px;margin:0 0 4px}
.sub{color:var(--muted);font-size:14px;margin-bottom:20px}.msg{background:var(--card);border:1px solid var(--line);
border-radius:12px;padding:12px 14px;margin:10px 0}.you{border-color:var(--accent)}.who{font-size:12px;
letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}.dense{font:13px/1.4 ui-monospace,monospace;
color:var(--muted);margin-top:8px;white-space:pre-wrap}form{position:fixed;left:0;right:0;bottom:0;background:var(--bg);
border-top:1px solid var(--line);padding:12px 16px}form div{max-width:760px;margin:0 auto;display:flex;gap:8px}
input{flex:1;padding:10px 12px;border-radius:10px;border:1px solid var(--line);background:var(--card);color:var(--fg);font-size:16px}
button{padding:10px 16px;border-radius:10px;border:0;background:var(--accent);color:#fff;font-size:16px}
.hint{color:var(--muted);font-size:14px}code{font-size:13px}</style></head><body><main>
<h1>Haishool</h1><div class="sub">A small model trained only on dense facts about __N__ objects, people, countries, ideas and works.
Its answers are turned into English by fixed rules; the grey line shows what the model itself wrote.</div>
<div class="hint">Try: <code>who are you</code> · <code>who is the kanzler</code> ·
<code>how many states does germany have</code> · <code>who was albert einstein</code> · <code>what is game theory</code> ·
<code>which shoe am i wearing right now</code> · <code>what color is a banana</code></div><div id="log"></div></main>
<form id="f"><div><input id="q" autocomplete="off" placeholder="Ask about objects, people, countries, history, science" autofocus>
<button>Ask</button></div></form><script>
const log=document.getElementById('log'),f=document.getElementById('f'),q=document.getElementById('q');
function add(cls,who,text,dense){const d=document.createElement('div');d.className='msg '+cls;
const w=document.createElement('div');w.className='who';w.textContent=who;d.append(w);
const t=document.createElement('div');t.textContent=text;d.append(t);
if(dense){const x=document.createElement('div');x.className='dense';x.textContent=dense;d.append(x)}
log.append(d);d.scrollIntoView({behavior:'smooth'})}
f.addEventListener('submit',async e=>{e.preventDefault();const text=q.value.trim();if(!text)return;q.value='';
add('you','You',text);try{const r=await fetch('api/ask',{method:'POST',headers:{'content-type':'application/json'},
body:JSON.stringify({question:text})});const j=await r.json();
add('bot','Homunculi',j.answer,j.dense?('query: '+j.query+'\\nmodel: '+j.dense):'')}catch(err){add('bot','Homunculi','Error: '+err)}});
</script></body></html>"""


class Ask(BaseModel):
    question: str = Field(min_length=1, max_length=300)


def build(model_path: Path, records_path: Path, device: str, hops: Path | None = None) -> FastAPI:
    model, vocab = load(model_path, device)
    rows = [json.loads(line) for line in records_path.open(encoding="utf-8")]
    known = {r["obj"]: r["values"].get("type") for r in rows} | {SELF: "self"}
    aliases = alias_index(rows)
    values = {r["obj"]: r["values"] for r in rows}
    # round 4: the links between things (training lines + edge list from haishool.relations)
    graph: Graph | None = None
    if hops is not None:
        graph = load_graph(hops / "hops-train-r4.txt", hops / "hops-edges-r4.jsonl")
        for node in graph.nodes:
            known.setdefault(node, "node")
            rows.append({"obj": node, "values": {"type": "node"}}) if node not in values else None
    index = Index(rows)
    app = FastAPI()
    app.add_middleware(CORSMiddleware, allow_origins=["https://enderpeer.github.io"],
                       allow_methods=["GET", "POST"], allow_headers=["content-type"])
    # the chat window of the website (docs/chat.html) when it is there; it falls back to this server's API
    chat = Path(__file__).resolve().parent.parent / "docs" / "chat.html"
    page = chat.read_text(encoding="utf-8") if chat.exists() else PAGE.replace("__N__", str(len(known) - 1))

    @app.get("/", response_class=HTMLResponse)
    def home() -> str:
        return page

    @app.get("/api/info")
    def info() -> dict:
        return {"objects": len(known) - 1, "params": model.num_params(), "vocab": len(vocab.itos),
                "model": model_path.name, "linked": len(graph.nodes) if graph else 0}

    def ask_links(question: str) -> dict | None:
        """Round-4 routes: a path between two linked things, or one relation (optionally in one view)."""
        if graph is None:
            return None
        pair = two_nodes(question, graph, aliases)
        if pair:
            a, b = pair
            path = generate(model, vocab, f"q {a} hop {b}. a", max_new=90, device=device)
            n = generate(model, vocab, f"q {a} hops {b}. a", device=device)
            return {"answer": path_text(path, n, graph), "query": f"q {a} hop {b}. a q {a} hops {b}. a",
                    "dense": f"{path} | {n}"}
        obj = node_in(question, graph, aliases)
        if obj is None:
            return None
        rel = relation_for(question, obj, graph)
        if rel is None:
            return None
        view = view_for(question, graph)
        prompt = f"q {obj} {rel}" + (f" according_to {view}" if view else "") + ". a"
        dense = generate(model, vocab, prompt, device=device)
        text = relation_sentence(obj, rel, dense, graph)
        if view:
            text = f"According to {view.replace('_', ' ').title()}: {text}"
        return {"answer": text, "query": prompt, "dense": dense}

    def describe_node(obj: str, etype: str | None) -> dict:
        """A linked thing: its record facts key by key, then its links (the free description of a
        linked thing gives only the links)."""
        parts, dense_parts, query = [], [], []
        own = [k for k in values.get(obj, {}) if k not in ("type", "aliases")]
        if own:
            got = [(k, answer(model, vocab, obj, k, device)) for k in own]
            line = f"{obj}. " + " ".join(f"{k} {d}." for k, d in got)
            parts += record_text(line, etype if etype not in ("self", "node") else None)
            dense_parts.append(line)
        for rel in graph.rels.get(obj, [])[:6]:
            d = generate(model, vocab, f"q {obj} {rel}. a", device=device)
            parts.append(relation_sentence(obj, rel, d, graph))
            dense_parts.append(f"{rel} {d}.")
            query.append(rel)
        return {"answer": " ".join(parts), "query": f"{obj}. + links: " + " ".join(query), "dense": " ".join(dense_parts)}

    @app.post("/api/ask")
    def ask(body: Ask) -> dict:
        with torch.no_grad():
            linked = ask_links(body.question)
            if linked:
                return linked
        r = resolve(body.question, known, aliases, index)
        obj, attrs, note, prefix = r["obj"], r["attrs"], r["note"], r["prefix"]
        etype = known.get(obj)
        with torch.no_grad():
            if attrs is None and graph is not None and obj in graph.nodes and not prefix:
                return describe_node(obj, etype)
            if attrs is None:
                dense = describe(model, vocab, obj, device)
                text = " ".join(record_text(dense, etype if etype != "self" else None)) or "I know its name but no facts."
                query = f"{obj}."
            else:
                got = [(a, answer(model, vocab, obj, a, device)) for a in attrs]
                text = " ".join(sentence(obj, a, d, etype) for a, d in got)
                name = subject(obj, etype if etype != "self" else None)
                if note.startswith(("found by", "spelling")) and name.lower() not in text.lower():
                    text = f"{name}: {text[:1].lower()}{text[1:]}"  # "Halo: its genre is ..."
                query = " ".join(f"q {obj} {a}. a" for a, _ in got)
                dense = " | ".join(d for _, d in got)
        return {"answer": prefix + text, "query": (note + "\n" if note else "") + query, "dense": dense}

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8650)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--hops", type=Path, default=None, help="folder with the round-4 hops files (links model)")
    args = ap.parse_args()
    uvicorn.run(build(args.model, args.records, args.device, args.hops), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
