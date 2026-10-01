"""Test chat for round 4 (hops): the round-3 chat plus questions about how things are linked.

    python -m haishool.hops_app --model model/haishool-v4b.pt --records data/records-r3-all.jsonl \
        --seed data/hops-seed-r4.jsonl --port 8651

``/`` is the chat. Questions that name two things ("how is turkey connected to led zeppelin",
"does turkey border israel"), ask for a side ("according to palestine, what is jerusalem the
capital of") or for the links of one thing ("what is hagia sophia linked to") are asked of the
model in the round-4 dense form. Every step the model writes is checked against the real links
(green: the link exists, red: the model made it up). Everything else goes to the round-3 chat
(:mod:`haishool.app`) unchanged. ``/explore`` picks two things directly.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from haishool import app as chat_app
from haishool.relations import INVERSE, build_graph, canonical, inverse, paths_from, step_text
from haishool.student import generate, load
from haishool.translate import alias_index, candidates

#: how a relation reads in English, as "X <phrase> Y" (the rest: underscores become spaces)
PHRASE = {
    "allied_with": "was allied with", "relations_with": "has diplomatic relations with", "related_to": "is related to",
    "shares_food": "shares a dish with", "bandmate": "played in a band with", "coast_on": "lies on the",
    "coast_of": "washes the coast of", "part_of": "is part of", "has_part": "includes", "city_in": "is in",
    "has_city": "is home to", "?": "(jumps to)", "capital_of": "is the capital of", "has_capital": "has the capital",
    "holy_to": "is holy to", "holds_holy": "holds holy", "father_figure_of": "is a father figure of",
    "has_father_figure": "has the father figure", "prophet_of": "is a prophet of", "has_prophet": "has the prophet",
    "son_of_god_in": "is the Son of God in", "holds_son_of_god": "holds as Son of God", "home_of": "was home to",
    "had_capital": "had its capital in", "was_capital_of": "was the capital of", "old_name_from": "has an old name from",
    "gave_old_name_to": "gave an old name to", "spoke_language_of": "spoke the language of",
    "language_spoken_by": "has its language spoken by", "state_religion": "had the state religion",
    "state_religion_of": "was the state religion of", "built_by": "was built by", "used_by": "was used by",
    "ended_by": "was ended by", "led_by": "was led by", "scene_of": "was the scene of", "ruled_by": "was ruled by",
    "followed_by": "was followed by", "had_side": "had on one side", "recognised_by": "was recognised by",
    "signed_by": "was signed by", "origin_in": "comes from", "origin_of": "is the origin of",
    "popularised_in": "was popularised in", "brought_by": "was brought by", "word_from": "is a word from",
    "gave_word": "gave the word", "culture_in": "has a culture in", "has_culture_of": "has a culture of",
    "national_style_of": "is a national style of", "has_national_style": "has as its national style",
    "drunk_in": "is drunk in", "origin_claimed_by": "is claimed as its own by", "claims_origin_of": "claims as its own",
    "refined_in": "was refined in", "eaten_in": "is eaten in", "grown_in": "grows in", "born_in": "was born in",
    "birthplace_of": "is the birthplace of", "founded_by": "was founded by", "based_in": "is based in",
    "base_of": "is the base of", "honoured_by": "was honoured by", "headlined_by": "was headlined by",
    "formed_in": "was formed in", "formed_here": "saw the founding of", "member_of": "was a member of",
    "has_member": "had the member", "played_by": "is played by", "influenced_by": "was influenced by",
    "made_by": "was made by", "inspired_by": "was inspired by", "travelled_in": "travelled in",
    "visited_by": "was visited by", "recorded_in": "was recorded in", "recording_place_of": "saw the recording of",
    "played_with": "was played with", "played_on": "played on", "played_in": "played in",
    "known_for": "is known for", "makes_known": "made famous", "country": "is from", "country_of": "is the country of",
    "involved_in": "was involved in", "caused_by": "was caused by", "resulted_in": "resulted in",
    "result_of": "was a result of", "holder": "is held by", "held": "holds the office", "made_of": "is made of",
    "material_of": "is a material of", "found_in": "is found in", "place_of": "is a place for",
    "has_component": "has the part", "component_of": "is a part of", "located_in": "is located in",
    "location_of": "is the location of", "is_a": "is a", "has_instance": "includes", "of_type": "is a",
    "type_of": "is the type of", "continent": "is in", "continent_of": "is the continent of", "currency": "uses the",
    "currency_of": "is the currency of", "language": "speaks", "language_of": "is spoken in", "era": "is from the",
    "era_of": "is the era of", "genre": "has the genre", "genre_of": "is the genre of", "field": "belongs to",
    "field_of": "is the field of", "role": "has the role", "role_of": "is the role of",
}
#: question words -> relation they ask about (first match wins)
ASKS = [
    (r"\bborder", "borders"), (r"\bcapital", "capital_of"), (r"\bmember|\bband", "member_of"),
    (r"\bshare\w* (a )?(food|dish)", "shares_food"), (r"\bholy|\bsacred", "holy_to"), (r"\bprophet", "prophet_of"),
    (r"\bborn", "born_in"), (r"\bfound(ed|er)", "founded"), (r"\bmade of|\bmaterial", "made_of"),
    (r"\beat(en|s)?\b", "eaten_in"), (r"\bsign", "signed"), (r"\binfluenc", "influenced_by"), (r"\bcoast", "coast_on"),
    (r"\bpart of", "part_of"), (r"\bmade by|\bcreat|\bwr[io]te", "made"), (r"\ballied|\bally", "allied_with"),
    (r"\brecogni[sz]", "recognised"), (r"\bfought|\bfight", "fought_in"), (r"\blocated|\bwhere\b", "located_in"),
    (r"\brul(ed|e)", "ruled"), (r"\bclaim", "origin_claimed_by"), (r"\bkind of|\bis an? ", "is_a"),
]
LINK_WORDS = re.compile(r"connect|link|relat|between|path|hops?\b|how far|way from|get from|to get to|in common|share")
VIEW_WORDS = re.compile(r"according to|who (sees|believes|claims|considers|says|recogni[sz]es|thinks)|disputed|contested|whose")
YES_NO = re.compile(r"^(is|are|was|were|does|do|did|has|have|can)\b")

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Haishool Hops Chat</title>
<style>
:root{--bg:#f5f4ef;--fg:#202924;--muted:#66706a;--card:#fffdf8;--line:#d9d6cc;--accent:#2f6b4f;
--ok:#2f6b4f;--okbg:#e3f0e8;--bad:#a33a2c;--badbg:#f6e1dd;--view:#7a5a12;--viewbg:#f4ecd6}
@media (prefers-color-scheme:dark){:root{--bg:#161a18;--fg:#e8ebe6;--muted:#9aa49d;--card:#1f2421;--line:#333a36;
--accent:#7cc4a0;--ok:#8fd3ae;--okbg:#1e3329;--bad:#f0a093;--badbg:#3a221e;--view:#e3c27a;--viewbg:#352c17}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif}
main{max-width:780px;margin:0 auto;padding:24px 16px 130px}h1{font-size:22px;margin:0 0 4px}
.sub{color:var(--muted);font-size:14px;margin-bottom:14px}a{color:var(--accent)}
.hint{color:var(--muted);font-size:14px;display:flex;flex-wrap:wrap;gap:6px;margin-bottom:6px}
.hint button{font-size:13px;padding:4px 10px;border-radius:999px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
.msg{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px;margin:10px 0}
.you{border-color:var(--accent)}.who{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted)}
.dense{font:13px/1.4 ui-monospace,monospace;color:var(--muted);margin-top:8px;white-space:pre-wrap;word-break:break-word}
.steps{margin:8px 0 0;padding:0;list-style:none}.steps li{margin:4px 0;display:flex;gap:8px;align-items:baseline}
.mark{font-weight:700;width:1.2em;flex:none}.ok{color:var(--ok)}.bad{color:var(--bad)}.view{color:var(--view)}
.tag{font-size:12px;padding:1px 6px;border-radius:6px;background:var(--viewbg);color:var(--view);margin-left:4px}
.verdict{margin-top:8px;font-size:14px;font-weight:600}
form{position:fixed;left:0;right:0;bottom:0;background:var(--bg);border-top:1px solid var(--line);padding:12px 16px}
form div{max-width:780px;margin:0 auto;display:flex;gap:8px}
input{flex:1;min-width:0;padding:10px 12px;border-radius:10px;border:1px solid var(--line);background:var(--card);color:var(--fg);font-size:16px}
button.send{padding:10px 16px;border-radius:10px;border:0;background:var(--accent);color:var(--bg);font-size:16px}
</style></head><body><main>
<h1>Haishool · Hops</h1>
<div class="sub">Round 4 test model (__PARAMS__ parameters), trained on __RECORDS__ records and __EDGES__ links between
__NODES__ things. For link questions the model writes the path itself; each step is checked against the real
links (<span class="ok">✓ exists</span>, <span class="bad">✗ made up</span>, <span class="view">◐ held by one side</span>).
Other questions go to the normal chat. <a href="explore">Explorer →</a></div>
<div class="hint" id="hints"></div><div id="log"></div></main>
<form id="f"><div><input id="q" autocomplete="off" placeholder="How is Turkey connected to Led Zeppelin?" autofocus>
<button class="send">Ask</button></div></form><script>
const HINTS=['How is Turkey connected to Led Zeppelin?','What links Israel and Turkey?','Does Turkey border Israel?',
'Does Turkey border Greece?','According to Palestine, what is Jerusalem the capital of?','Who claims baklava?',
'What is Hagia Sophia linked to?','How is a spoon connected to the Beatles?','What is the capital of Japan?'];
const log=document.getElementById('log'),f=document.getElementById('f'),q=document.getElementById('q');
for(const h of HINTS){const b=document.createElement('button');b.type='button';b.textContent=h;b.onclick=()=>{q.value=h;f.requestSubmit()};
document.getElementById('hints').append(b)}
function el(tag,cls,text){const e=document.createElement(tag);if(cls)e.className=cls;if(text!=null)e.textContent=text;return e}
function add(cls,who,j){const d=el('div','msg '+cls);d.append(el('div','who',who));d.append(el('div',null,j.answer));
if(j.steps&&j.steps.length){const ul=el('ul','steps');for(const s of j.steps){const li=el('li');
const k=s.ok===false?'bad':(s.view&&s.view!=='fact'?'view':'ok');li.append(el('span','mark '+k,k==='bad'?'✗':k==='view'?'◐':'✓'));
const t=el('span',null,s.text);if(s.view&&s.view!=='fact')t.append(el('span','tag','according to '+s.view.replaceAll('_',' ')));
li.append(t);ul.append(li)}d.append(ul)}
if(j.verdict)d.append(el('div','verdict '+(j.good?'ok':'bad'),j.verdict));
if(j.dense)d.append(el('div','dense',(j.query?'query: '+j.query+'\\n':'')+'model: '+j.dense));
log.append(d);d.scrollIntoView({behavior:'smooth'})}
f.addEventListener('submit',async e=>{e.preventDefault();const text=q.value.trim();if(!text)return;q.value='';
add('you','You',{answer:text});try{const r=await fetch('api/ask',{method:'POST',headers:{'content-type':'application/json'},
body:JSON.stringify({question:text})});const j=await r.json();add('bot','Homunculi',j)}catch(err){add('bot','Homunculi',{answer:'Error: '+err})}});
</script></body></html>"""

EXPLORE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Haishool Hops Explorer</title>
<style>
:root{--bg:#f5f4ef;--fg:#202924;--muted:#66706a;--card:#fffdf8;--line:#d9d6cc;--accent:#2f6b4f;--ok:#2f6b4f;--bad:#a33a2c;--view:#7a5a12}
@media (prefers-color-scheme:dark){:root{--bg:#161a18;--fg:#e8ebe6;--muted:#9aa49d;--card:#1f2421;--line:#333a36;--accent:#7cc4a0;--ok:#8fd3ae;--bad:#f0a093;--view:#e3c27a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif}
main{max-width:780px;margin:0 auto;padding:24px 16px}h1{font-size:22px;margin:0 0 4px}a{color:var(--accent)}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px;margin:12px 0}
.row{display:flex;flex-wrap:wrap;gap:8px}input{flex:1;min-width:140px;padding:9px 10px;border-radius:10px;border:1px solid var(--line);background:var(--bg);color:var(--fg);font-size:15px}
button{padding:9px 14px;border-radius:10px;border:0;background:var(--accent);color:var(--bg);font-size:15px;cursor:pointer}
.mono{font:13px/1.45 ui-monospace,monospace;white-space:pre-wrap;word-break:break-word}.muted{color:var(--muted);font-size:14px}
.ok{color:var(--ok)}.bad{color:var(--bad)}.view{color:var(--view)}
</style></head><body><main><h1>Hops explorer</h1><div class="muted"><a href="./">← chat</a> · model path vs. cheapest path in the data</div>
<div class="card"><div class="row"><input id="a" list="names" value="turkey"><input id="b" list="names" value="led_zeppelin">
<button id="go">Find hop</button><button id="rnd">Random</button></div><datalist id="names"></datalist><div id="out" class="mono" style="margin-top:10px"></div></div>
</main><script>
const $=id=>document.getElementById(id);let names=[];
fetch('api/nodes').then(r=>r.json()).then(j=>{names=j.nodes;const d=$('names');for(const n of names.slice(0,4000)){const o=document.createElement('option');o.value=n;d.append(o)}});
async function go(){const out=$('out');out.textContent='…';const r=await fetch('api/hop',{method:'POST',headers:{'content-type':'application/json'},
body:JSON.stringify({a:$('a').value.trim(),b:$('b').value.trim()})});if(!r.ok){out.textContent=await r.text();return}const j=await r.json();out.textContent='';
const line=(t,c)=>{const s=document.createElement('div');if(c)s.className=c;s.textContent=t;out.append(s)};
line('model ('+j.model.hops+' hops):');for(const s of j.model.steps)line((s.ok===false?'✗ ':s.view!=='fact'?'◐ ':'✓ ')+s.text,s.ok===false?'bad':s.view!=='fact'?'view':'ok');
line(j.verdict,j.good?'ok':'bad');line('');line('data ('+j.gold.hops+' hops):');for(const s of j.gold.steps)line('  '+s.text);
line('');line('raw: '+j.model.raw,'muted')}
$('go').onclick=go;$('rnd').onclick=()=>{$('a').value=names[Math.floor(Math.random()*names.length)];$('b').value=names[Math.floor(Math.random()*names.length)];go()};
</script></body></html>"""


class Ask(BaseModel):
    question: str = Field(min_length=1, max_length=300)


class HopQ(BaseModel):
    a: str = Field(max_length=120)
    b: str = Field(max_length=120)


RELATIONS = set(INVERSE) | set(INVERSE.values())


def parse_path(start: str, text: str) -> list[dict]:
    """``turkey recognised israel borders palestine according_to x ...`` -> steps (model output).

    A thing where a relation should be is kept as a step with relation ``?`` (a jump the model
    made up), so a garbled answer shows as wrong instead of being misread.
    """
    words = text.split()
    if words[:1] == [start]:
        words = words[1:]
    steps, i = [], 0
    while i < len(words):
        if words[i] not in RELATIONS:
            steps.append({"rel": "?", "to": words[i], "view": "fact"})
            i += 1
            continue
        if i + 1 >= len(words):
            break
        step = {"rel": words[i], "to": words[i + 1], "view": "fact"}
        i += 2
        if i + 1 < len(words) and words[i] == "according_to":
            step["view"] = words[i + 1]
            i += 2
        steps.append(step)
    return steps


def name(node: str, types: dict[str, str]) -> str:
    if node.startswith("type_"):
        return "a " + node[5:]
    plain = node.replace("_", " ")
    return plain.title() if types.get(node) not in ("object", "value", "kind", "concept") else plain


def clause(a: str, rel: str, b: str, types: dict[str, str]) -> str:
    return f"{name(a, types)} {PHRASE.get(rel, rel.replace('_', ' '))} {name(b, types)}"


def build(model_path: Path, records_path: Path, seed_path: Path, device: str) -> FastAPI:
    model, vocab = load(model_path, device)
    g = build_graph(seed_path, records_path)
    types = g.types
    real: dict[tuple[str, str, str], str] = {}
    for e in g.edges:
        real[(e.a, e.rel, e.b)] = e.view
        real[(e.b, inverse(e.rel), e.a)] = e.view
    views_of: dict[tuple[str, str], set[str]] = {}
    for (a, rel, _), v in real.items():
        if v != "fact":
            views_of.setdefault((a, rel), set()).add(v)
    rows = [json.loads(ln) for ln in records_path.open(encoding="utf-8")]
    known = {n: t for n, t in types.items() if not n.startswith("type_")}
    aliases = alias_index(rows)
    aliases.update({"usa": "united_states", "us": "united_states", "america": "united_states", "uk": "united_kingdom",
                    "britain": "united_kingdom", "england": "united_kingdom", "zeppelin": "led_zeppelin",
                    "beatles": "the_beatles", "constantinople": "istanbul", "jews": "judaism", "muslims": "islam",
                    "christians": "christianity", "palestinians": "palestine", "israelis": "israel"})
    chat = chat_app.build(model_path, records_path, device)
    round3_ask = next(r.endpoint for r in chat.routes if getattr(r, "path", "") == "/api/ask")

    def run(prompt: str, **kw) -> str:
        with torch.no_grad():
            return generate(model, vocab, prompt, device=device, **kw)

    def things(question: str) -> list[str]:
        """Things named in the question, in the order they appear, longer names first on overlap."""
        text = " " + re.sub(r"[^a-z0-9]+", " ", question.lower()) + " "
        found = sorted(candidates(question, known, aliases), key=lambda c: (-c[0], -c[1]))
        picked: list[tuple[int, str]] = []
        taken: list[tuple[int, int]] = []
        for _, _, node in found:
            words = [w for w, n in aliases.items() if n == node and " " + w.replace("_", " ") + " " in text]
            for form in [node.replace("_", " "), *[w.replace("_", " ") for w in words], node.replace("_", " ").rstrip("s")]:
                pos = text.find(" " + form)
                if pos >= 0:
                    span = (pos, pos + len(form) + 1)
                    if not any(s < span[1] and span[0] < t for s, t in taken):
                        taken.append(span)
                        picked.append((pos, node))
                    break
        return [n for _, n in sorted(picked)]

    def hop(a: str, b: str) -> dict:
        raw = run(f"q {a} hop {b}. a", max_new=60)
        steps = parse_path(a, raw)
        cur, valid = a, True
        for s in steps:
            known_view = real.get((cur, s["rel"], s["to"]))
            s["ok"] = known_view is not None
            valid &= s["ok"]
            s["text"] = clause(cur, s["rel"], s["to"], types)
            cur = s["to"]
        gold = paths_from(g, a, {b}).get(b, [])
        gold_text = step_text(gold)
        reaches = cur == b
        verdict = ("Exactly the cheapest path in the data." if raw == f"{a} {gold_text}" else
                   "A different path, but every step is a real link." if valid and reaches else
                   f"The path does not end at {name(b, types)}." if not reaches else
                   "The path uses links that do not exist (✗).")
        g_steps, prev = [], a
        for r, m, v in gold:
            g_steps.append({"rel": r, "to": m, "view": v, "text": clause(prev, r, m, types)})
            prev = m
        return {"raw": raw, "steps": steps, "valid": valid, "reaches": reaches, "verdict": verdict,
                "good": valid and reaches, "hops": run(f"q {a} hops {b}. a"), "gold": gold, "gold_steps": g_steps}

    app = FastAPI()
    # the landing page on GitHub Pages (docs/hops.html) asks this server directly
    app.add_middleware(CORSMiddleware, allow_origins=["https://enderpeer.github.io"],
                       allow_methods=["GET", "POST"], allow_headers=["content-type"])

    @app.get("/api/info")
    def info() -> dict:
        return {"model": model_path.name, "params": model.num_params(), "vocab": len(vocab.itos),
                "records": len(rows), "nodes": len(types), "edges": len(g.edges)}

    @app.get("/", response_class=HTMLResponse)
    def home() -> str:
        return (PAGE.replace("__PARAMS__", f"{model.num_params() / 1e6:.1f} million")
                .replace("__RECORDS__", f"{len(rows):,}").replace("__EDGES__", f"{len(g.edges):,}")
                .replace("__NODES__", f"{len(types):,}"))

    @app.get("/explore", response_class=HTMLResponse)
    def explore() -> str:
        return EXPLORE

    @app.get("/api/nodes")
    def api_nodes() -> dict:
        return {"nodes": sorted(n for n in types if not n.startswith("type_"))}

    @app.post("/api/hop")
    def api_hop(q: HopQ) -> dict:
        a, b = q.a.strip().lower().replace(" ", "_"), q.b.strip().lower().replace(" ", "_")
        a, b = aliases.get(a, a), aliases.get(b, b)
        if a not in types or b not in types or a == b:
            raise HTTPException(404, "pick two different known things")
        h = hop(a, b)
        return {"model": {"raw": h["raw"], "steps": h["steps"], "hops": h["hops"]}, "verdict": h["verdict"],
                "good": h["good"], "gold": {"hops": len(h["gold"]), "steps": h["gold_steps"]}}

    @app.post("/api/ask")
    def ask(body: Ask) -> dict:
        question = body.question.strip()
        low = question.lower()
        named = things(question)
        rel = next((r for pat, r in ASKS if re.search(pat, low)), None)

        if VIEW_WORDS.search(low) and named:
            # "according to palestine, what is jerusalem the capital of" / "who claims baklava"
            holders = [n for n in named if any(n in vs for vs in views_of.values())]
            about = [n for n in named if n not in holders] or named[:1]
            n = about[0]
            rels = [rel] if rel else []
            rels += [r for (x, r) in views_of if x == n and r not in rels]
            for r in rels:
                if (n, r) not in views_of and inverse(r) in {rr for (x, rr) in views_of if x == n}:
                    r = inverse(r)
                if (n, r) not in views_of:
                    continue
                whole = run(f"q {n} {r}. a")
                dense = [whole]
                wanted = [h for h in holders if h in views_of[(n, r)]] or sorted(views_of[(n, r)])
                steps = []
                for v in wanted:
                    got = run(f"q {n} {r} according_to {v}. a")
                    dense.append(f"according_to {v}: {got}")
                    targets = got.split()
                    ok = bool(targets) and all(real.get((n, r, t)) == v for t in targets)
                    steps.append({"text": f"{name(n, types)} {PHRASE.get(r, r.replace('_', ' '))} "
                                          f"{', '.join(name(t, types) for t in targets) or '(nothing)'}",
                                  "view": v, "ok": ok})
                agreed = "contested" in whole.split()
                answer = (f"This is not agreed on: different sides hold different answers about "
                          f"{name(n, types)}." if agreed else f"Only one side holds this about {name(n, types)}.")
                return {"answer": answer, "steps": steps, "query": f"q {n} {r}. a  (+ according_to …)",
                        "dense": "\n".join(dense), "kind": "views"}

        if len(named) >= 2 and rel and YES_NO.match(low):
            a, b = named[0], named[1]
            a_, r_, b_ = canonical(a, rel, b)
            got = run(f"q {a_} {r_} {b_}. a")
            if got == "no" and inverse(rel) == rel:
                got2 = run(f"q {b_} {r_} {a_}. a")
                got = got2 if got2 != "no" else got
            truth = real.get((a_, r_, b_))
            data = "yes" if truth == "fact" else ("no" if truth is None else f"according_to {truth}")
            text = clause(a_, r_, b_, types)
            answer = (f"Yes: {text}." if got == "yes" else f"No, as far as my data goes, it is not true that {text[0].lower() + text[1:]}."
                      if got == "no" else f"Only according to {got.replace('according_to ', '').replace('_', ' ').title()}: {text}."
                      if got.startswith("according_to") else f"The model answered: {got}")
            agrees = got == data or (got == "no" and truth is None)
            return {"answer": answer, "verdict": "The data agrees." if agrees else f"The data says: {data}.",
                    "good": agrees, "query": f"q {a_} {r_} {b_}. a", "dense": got, "kind": "yesno"}

        if len(named) >= 2 and (LINK_WORDS.search(low) or not rel):
            a, b = named[0], named[1]
            h = hop(a, b)
            n_hops = len(h["steps"])
            answer = (f"{name(a, types)} reaches {name(b, types)} in {n_hops} hop{'s' if n_hops != 1 else ''}:"
                      if h["reaches"] else f"The model did not find its way from {name(a, types)} to {name(b, types)}:")
            return {"answer": answer, "steps": h["steps"], "verdict": h["verdict"], "good": h["good"],
                    "query": f"q {a} hop {b}. a", "dense": h["raw"], "kind": "hop"}

        if len(named) == 1 and LINK_WORDS.search(low) and named[0] in types:
            n = named[0]
            raw = run(f"{n} links.", stop="", max_new=70)
            steps = []
            for chunk in [c.strip() for c in raw.split(".") if c.strip()]:
                r, *targets = chunk.split()
                for t in targets:
                    v = real.get((n, r, t))
                    steps.append({"text": clause(n, r, t, types), "ok": v is not None, "view": v or "fact"})
            return {"answer": f"What {name(n, types)} is linked to:", "steps": steps,
                    "query": f"{n} links.", "dense": raw, "kind": "links"}

        out = round3_ask(chat_app.Ask(question=question))
        return out | {"kind": "round3"}

    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--seed", type=Path, default=Path("data/hops-seed-r4.jsonl"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8651)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    uvicorn.run(build(args.model, args.records, args.seed, args.device), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
