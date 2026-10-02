"""Experiment: can the 6x384 model learn the maths gate as a rule when it sees 1.5 M distinct lines
(each a few times) instead of 9,000 lines (each ~200 times)? Held-out = fresh seed, prompts not in training."""
import json, math, random, sys, time
from collections import defaultdict
from pathlib import Path
import torch
from haishool.model import GPT, GPTConfig
from haishool.student import EOS, Vocab, generate, tokens
from haishool.truth import maths, split_line

name, layers, width, heads, n_lines, steps = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
g = maths.gate()
train, seen, seed = [], set(), 1000
while len(train) < n_lines:
    for ln in g.generate(random.Random(seed), 20000):
        if ln.prompt not in seen:
            seen.add(ln.prompt); train.append(ln.text)
    seed += 1
test = [ln for ln in g.generate(random.Random(999999), 6000) if ln.prompt not in seen][:3000]
print(json.dumps({"train_lines": len(train), "test_lines": len(test), "seeds": seed - 1000}), flush=True)
words = sorted({w for t in train for w in tokens(t)} | {w for ln in test for w in tokens(ln.text)})
vocab = Vocab(words)
random.Random(1).shuffle(train)
ids = []
for t in train:
    ids.extend(vocab.encode(tokens(t))); ids.append(vocab.stoi[EOS])
stream = torch.tensor(ids, dtype=torch.int32)
cfg = GPTConfig(vocab_size=max(32, len(vocab.itos)), n_layer=layers, n_head=heads, d_model=width, ctx=96, dropout=0.1)
torch.manual_seed(1)
model = GPT(cfg).to("cuda")
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
print(json.dumps({"tokens": len(ids), "params": model.num_params(), "epochs": round(steps * 128 * 96 / len(ids), 1)}), flush=True)
t0 = time.monotonic()
for step in range(1, steps + 1):
    lr = 1e-3 * min(1.0, step / 200) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps)))
    for gr in opt.param_groups: gr["lr"] = lr
    idx = torch.randint(0, len(stream) - 97, (128,))
    x = torch.stack([stream[i:i + 96] for i in idx]).long().cuda(); y = torch.stack([stream[i + 1:i + 97] for i in idx]).long().cuda()
    _, loss = model(x, y)
    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    if step % 1000 == 0: print(json.dumps({"step": step, "loss": round(loss.item(), 4), "s": round(time.monotonic() - t0)}), flush=True)
model.eval()
OPS = ["plus", "minus", "times", "over", "power", "root", "percent", "open", "point", "steps", "x"]
def bucket(prompt):
    w = prompt.split()
    return w[0] + ":" + "+".join(o for o in OPS if o in w[1:])
res = defaultdict(lambda: [0, 0, 0])
def run(lines, tag):
    for ln in lines:
        prompt, gold = (ln.prompt, ln.answer) if hasattr(ln, "prompt") else split_line(ln)
        got = generate(model, vocab, f"q {prompt}. a", max_new=90, device="cuda")
        b = res[tag + " " + bucket(prompt)]
        b[0] += 1; b[1] += got == gold; b[2] += g.check(prompt, got).ok
run(test, "heldout"); run(random.Random(2).sample(train, 1000), "trained")
out = {k: {"n": v[0], "exact": round(v[1] / v[0], 3), "gate_ok": round(v[2] / v[0], 3)} for k, v in sorted(res.items())}
tot = {t: [sum(v[i] for k, v in res.items() if k.startswith(t)) for i in range(3)] for t in ("heldout", "trained")}
out["TOTAL"] = {t: {"n": v[0], "exact": round(v[1] / v[0], 3), "gate_ok": round(v[2] / v[0], 3)} for t, v in tot.items()}
Path(f"exp/{name}.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out["TOTAL"]), flush=True)
