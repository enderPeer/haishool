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

from haishool.student import answer, describe, load
from haishool.translate import SELF, parse, record_text, sentence

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
<h1>Haishool · objects</h1><div class="sub">A small model trained only on dense facts about __N__ everyday objects.
Its answers are turned into English by fixed rules; the grey line shows what the model itself wrote.</div>
<div class="hint">Try: <code>what color is a banana</code> · <code>what is a hammer used for</code> ·
<code>where do you find a cow</code> · <code>tell me about a guitar</code> · <code>who are you</code> ·
<code>where do you run</code></div><div id="log"></div></main>
<form id="f"><div><input id="q" autocomplete="off" placeholder="Ask about an everyday object" autofocus>
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


def build(model_path: Path, records_path: Path, device: str) -> FastAPI:
    model, vocab = load(model_path, device)
    known = {json.loads(line)["obj"] for line in records_path.open(encoding="utf-8")} | {SELF}
    app = FastAPI()
    app.add_middleware(CORSMiddleware, allow_origins=["https://enderpeer.github.io"],
                       allow_methods=["GET", "POST"], allow_headers=["content-type"])
    page = PAGE.replace("__N__", str(len(known) - 1))

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return page

    @app.get("/api/info")
    def info() -> dict:
        return {"objects": len(known) - 1, "params": model.num_params(), "vocab": len(vocab.itos),
                "model": str(model_path)}

    @app.post("/api/ask")
    def ask(body: Ask) -> dict:
        p = parse(body.question, known)
        obj, attr = p["obj"], p["attr"]
        if obj is None:
            return {"answer": "I don't know that object yet. I only know everyday objects like a "
                              "spoon, a dog, a car or a banana.", "query": None, "dense": None}
        with torch.no_grad():
            if attr is None:
                dense = describe(model, vocab, obj, device)
                text = " ".join(record_text(dense)) or "I don't know much about it."
                return {"answer": text, "query": f"{obj}.", "dense": dense}
            dense = answer(model, vocab, obj, attr, device)
        return {"answer": sentence(obj, attr, dense), "query": f"q {obj} {attr}. a", "dense": dense}

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8650)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    uvicorn.run(build(args.model, args.records, args.device), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
