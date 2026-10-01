"""The student: a small GPT trained from scratch only on dense object records.

    python -m haishool.student train --records data/records.jsonl --out runs/full --holdout 0
    python -m haishool.student train --records data/records.jsonl --out runs/eval --holdout 0.1

Vocabulary: every word of the dense corpus plus ``.``, ``q``, ``a`` and ``<eos>`` (word level,
so the model spends no capacity on spelling). Training text: every record as a line and every
query/answer pair, shuffled anew for each pass, joined by ``<eos>`` into one stream; random
windows of ``ctx`` tokens, next-token loss everywhere.

With ``--holdout f`` a seeded fraction f of (object, attribute) pairs is removed from training
entirely (from the queries AND from the records), and the run reports exact-match accuracy on
the trained pairs (recall) and on the held-out pairs (inference from similar objects).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

from haishool.model import GPT, GPTConfig
from haishool.schema import ORDER, Record

EOS = "<eos>"
SPECIAL = ["<pad>", EOS, ".", "q", "a"]


def tokens(text: str) -> list[str]:
    return text.replace(".", " . ").split()


def load_records(path: Path) -> list[Record]:
    out = []
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        out.append(Record(row["obj"], row["values"]))
    return out


def split_pairs(records: list[Record], holdout: float, seed: int) -> tuple[list[Record], list[tuple[str, str, str]]]:
    rng = random.Random(seed)
    train, held = [], []
    for rec in records:
        values = dict(rec.values)
        if holdout > 0 and rec.obj != "homunculi":
            for key in ORDER:
                if key in values and key != "kind" and rng.random() < holdout:
                    held.append((rec.obj, key, values.pop(key)))
        train.append(Record(rec.obj, values))
    return train, held


class Vocab:
    def __init__(self, words: list[str]) -> None:
        self.itos = SPECIAL + sorted(set(words) - set(SPECIAL))
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def encode(self, ws: list[str]) -> list[int]:
        return [self.stoi[w] for w in ws if w in self.stoi]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.itos[i] for i in ids]


def corpus(records: list[Record], passes: int, seed: int) -> list[str]:
    samples = [r.line() for r in records] + [q for r in records for q in r.queries()]
    rng = random.Random(seed)
    stream: list[str] = []
    for _ in range(passes):
        rng.shuffle(samples)
        for s in samples:
            stream.extend(tokens(s))
            stream.append(EOS)
    return stream


@torch.no_grad()
def generate(model: GPT, vocab: Vocab, prompt: str, *, stop: str = ".", max_new: int = 40, device: str = "cpu") -> str:
    ids = vocab.encode(tokens(prompt))
    if not ids:
        return ""
    x = torch.tensor([ids], device=device)
    out: list[str] = []
    for _ in range(max_new):
        logits, _ = model(x[:, -model.config.ctx:])
        nxt = int(logits[0, -1].argmax())
        word = vocab.itos[nxt]
        if word == EOS or (stop and word == stop):
            break
        out.append(word)
        x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
    return " ".join(out)


def answer(model: GPT, vocab: Vocab, obj: str, key: str, device: str = "cpu") -> str:
    return generate(model, vocab, f"q {obj} {key}. a", device=device)


def describe(model: GPT, vocab: Vocab, obj: str, device: str = "cpu") -> str:
    return obj + ". " + generate(model, vocab, f"{obj}.", stop="", max_new=60, device=device)


def train(records_path: Path, out: Path, holdout: float, seed: int, steps: int, device: str) -> dict:
    records = load_records(records_path)
    train_recs, held = split_pairs(records, holdout, seed)
    words = [w for r in records for w in tokens(r.line())] + [w for r in records for q in r.queries() for w in tokens(q)]
    vocab = Vocab(words)
    stream = torch.tensor(vocab.encode(corpus(train_recs, passes=60, seed=seed)), dtype=torch.long)
    cfg = GPTConfig(vocab_size=max(32, len(vocab.itos)), n_layer=6, n_head=8, d_model=384, ctx=96, dropout=0.1)
    torch.manual_seed(seed)
    model = GPT(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    batch = 128
    t0 = time.monotonic()
    log = []
    for step in range(1, steps + 1):
        lr = 1e-3 * min(1.0, step / 200) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps)))
        for g in opt.param_groups:
            g["lr"] = lr
        idx = torch.randint(0, len(stream) - cfg.ctx - 1, (batch,))
        x = torch.stack([stream[i:i + cfg.ctx] for i in idx]).to(device)
        y = torch.stack([stream[i + 1:i + cfg.ctx + 1] for i in idx]).to(device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 250 == 0 or step == steps:
            log.append({"step": step, "loss": round(float(loss), 4), "seconds": round(time.monotonic() - t0, 1)})
            print(json.dumps(log[-1]), flush=True)
    model.eval()

    def acc(pairs: list[tuple[str, str, str]]) -> dict:
        if not pairs:
            return {"n": 0}
        exact = overlap = 0
        for obj, key, gold in pairs:
            got = answer(model, vocab, obj, key, device)
            exact += got == gold
            g, p = set(gold.split()), set(got.split())
            overlap += len(g & p) / len(g) if g else 0
        return {"n": len(pairs), "exact": round(exact / len(pairs), 4), "word_recall": round(overlap / len(pairs), 4)}

    trained_pairs = [(r.obj, k, r.values[k]) for r in train_recs for k in ORDER if k in r.values]
    rng = random.Random(seed + 1)
    sample = rng.sample(trained_pairs, min(1000, len(trained_pairs)))
    report = {"records": len(records), "vocab": len(vocab.itos), "params": model.num_params(),
              "train_tokens_per_pass": len(stream) // 60, "steps": steps, "holdout": holdout,
              "recall_trained_pairs": acc(sample), "inference_heldout_pairs": acc(held), "log": log}
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "gpt_config": cfg.to_json(), "itos": vocab.itos}, out / "student.pt")
    (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "log"}), flush=True)
    return report


def load(path: Path, device: str = "cpu") -> tuple[GPT, Vocab]:
    ck = torch.load(path, map_location=device, weights_only=True)
    model = GPT(GPTConfig.from_json(ck["gpt_config"])).to(device)
    model.load_state_dict({k: v.float() for k, v in ck["model_state"].items()})
    model.eval()
    vocab = Vocab([])
    vocab.itos = ck["itos"]
    vocab.stoi = {w: i for i, w in enumerate(vocab.itos)}
    return model, vocab


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--records", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--holdout", type=float, default=0.0)
    t.add_argument("--seed", type=int, default=20261001)
    t.add_argument("--steps", type=int, default=4000)
    t.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    train(args.records, args.out, args.holdout, args.seed, args.steps, args.device)


if __name__ == "__main__":
    main()
