"""The student: a small GPT trained from scratch only on dense fact lines.

    python -m haishool.student train --records data/records.jsonl --out runs/full --holdout 0
    python -m haishool.student train --records data/records.jsonl --out runs/eval --holdout 0.1
    python -m haishool.student train --records data/records-r3-all.jsonl --out runs/r5 \\
        --extra-lines data/hops-v4b/hops-train-r4.txt --extra-lines data/truth-v5/truth-train-r5.txt \\
        --init model/haishool-v4b.pt --steps 8000

Vocabulary: every word of the dense corpus plus ``.``, ``q``, ``a`` and ``<eos>`` (word level,
so the model spends no capacity on spelling). Training text: every record as a line and every
query/answer pair, shuffled anew for each pass, joined by ``<eos>`` into one stream; random
windows of ``ctx`` tokens, next-token loss everywhere.

With ``--holdout f`` a seeded fraction f of (object, attribute) pairs is removed from training
entirely (from the queries AND from the records), and the run reports exact-match accuracy on
the trained pairs (recall) and on the held-out pairs (inference from similar objects).

``--extra-lines file`` (may be repeated) adds ready-made dense lines to every pass, each after an
``<eos>`` like any other sample: round 4 ``data/hops-v4b/hops-train-r4.txt`` (the relations
between things), round 5 the truth gates' lines, round 6 the toy-universe rollouts. With
``--holdout f`` the same fraction of hop pairs (both ``q x hop y`` and ``q x hops y`` go
together) and of the prompts of the round-5/6 questions is left out of training entirely, so
the report shows whether the model can compose a path, or compute an answer, it never saw.
Whether a round-5/6 prompt is held out depends only on the seed and the prompt, not on the
other lines, so a later run with more files (the feedback loop's) holds out the same questions;
a worked ``steps`` answer, a ``check`` and a ``judge`` of that question go with its
underlying question, since they can give the answer away. Record lines (no ``q``) and the other round-4 questions (relation, view, yesno)
are always trained, as before; a held-out round-5 fact is therefore still in its record line,
and the held-out number says whether the model can turn the record into the answer. The report
has one ``extra_files`` entry per file with the exact-match accuracy per kind of line
(:func:`extra_kind`):

    q turkey borders. a greece syria.                    relation
    q turkey borders greece. a yes.                      yesno
    q islam has_prophet according_to islam. a ...        view
    q turkey hop led_zeppelin. a turkey has_city ...     hop
    q turkey hops led_zeppelin. a 4.                     hops
    q calc 1 2 plus 7. a 1 9.                            calc     (any verb of KIND_WORDS)
    q friction mu 0 point 3 n 5 0. a 1 5.                friction
    q gravity seed 7 step 2 0 clumps. a 4.               gravity  (a toy-universe simulation)
    q carbon protons. a 6.                               fact
    q friction unit. a newton.                           fact     (a thing and one of its keys)

A verb that is not in ``KIND_WORDS`` still gets its own bucket in a file where it starts 30 or
more different questions (:func:`verbs_of`), so a gate may gain a verb without a change here.

``--init checkpoint`` continues training from a saved model instead of from scratch: the old
vocabulary keeps every token id and the new words are appended after it (so old prompts encode
identically), the token embedding (tied to the output layer) grows by copying the old rows and
initialising the new ones like a fresh model, the optimiser starts afresh, and the peak
learning rate defaults to 3e-4 instead of 1e-3 (same warm-up and cosine schedule). With
``--holdout`` the held-out numbers only mean something when the checkpoint was trained with the
same seed and holdout: a checkpoint trained on everything has seen every held-out line.

An extra line longer than the context, or outside the dense alphabet
(:func:`haishool.truth.is_dense`), is not trained: it is dropped, named on stderr and counted
in the report (``dropped_long_lines``, ``dropped_not_dense_lines``, in total and per file).

``--layers --width --heads --ctx`` size a model trained from scratch (defaults 6, 384, 8, 96:
16 million parameters); with ``--init`` they come from the checkpoint (``--ctx`` may grow).
Without any of the new flags training is unchanged: same corpus, same vocabulary order, same
model, same checkpoint (``{"model_state", "gpt_config", "itos"}``).
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import random
import sys
import time
from array import array
from pathlib import Path

try:
    import torch

    from haishool.model import GPT, GPTConfig
except ImportError:  # the pure-python parts (tokens, Vocab, split_extra, extra_kind, corpus) work without torch
    torch = None  # type: ignore[assignment]

from haishool.schema import QUERY_KEYS, Record

EOS = "<eos>"
SPECIAL = ["<pad>", EOS, ".", "q", "a"]
#: the model trained in rounds 1-4: 6 layers, 384 wide, 8 heads, 96 tokens of context
LAYERS, WIDTH, HEADS, CTX = 6, 384, 8, 96
#: steps of linear warm-up before the cosine schedule
WARMUP = 200
#: passes over the samples in one training stream (``train_tokens_per_pass`` divides by it)
PASSES = 60
#: first words of the round-5 questions that ask for more than one key of one thing: a
#: calculation (``q calc 1 2 plus 7``), a comparison, a search through the table (``q elements
#: period 1``); each is its own report bucket, named after the word. The first rows are the
#: verbs of the five gates of :mod:`haishool.truth` and of its loop, the last are spare.
KIND_WORDS = frozenset({
    "calc", "solve", "compare", "check", "judge",
    "element", "elements", "count", "lighter", "heavier", "higher", "lower", "neutrons",
    "atoms", "molar_mass", "mass_percent", "element_count", "heavier_molecule", "substance", "substances",
    "balance", "count_atoms", "reactions",
    "forces", "stronger", "longer_range", "weight", "gravity_force", "coulomb_force", "spring_force",
    "friction", "pressure", "acceleration", "net_force", "buoyancy", "centripetal", "orbital_speed",
    "escape_speed", "drag", "magnetic_force",
    "steps", "convert", "round", "simplify", "factor", "order", "sort", "predict", "derive", "expand",
    "estimate", "classify", "verify", "prove", "evaluate", "reduce", "react", "products", "sum", "mean",
    "max", "min", "gcd", "lcm", "mod", "prime", "divides", "between", "nearest", "unit", "force", "law",
})
#: the toy-universe simulations of round 6 (:mod:`haishool.cosmos`), each its own report bucket
SIMS = frozenset({"nucleo", "gravity", "planets", "chem", "life", "world"})
#: a word that starts this many different long questions of one file is a verb (:func:`verbs_of`)
VERB_PROMPTS = 30
#: the report buckets of round 4, always listed so that old and new reports compare
CLASSIC_KINDS = ("relation", "view", "yesno", "hop", "hops")


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
            for key in QUERY_KEYS:
                if key in values and key not in ("kind", "type") and rng.random() < holdout:
                    held.append((rec.obj, key, values.pop(key)))
        train.append(Record(rec.obj, values))
    return train, held


class Vocab:
    def __init__(self, words: list[str]) -> None:
        self.itos = SPECIAL + sorted(set(words) - set(SPECIAL))
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    @classmethod
    def grown(cls, itos: list[str], words: list[str]) -> Vocab:
        """An old vocabulary with the new words appended after it, so every old token keeps its id."""
        v = cls([])
        v.itos = list(itos) + [s for s in SPECIAL if s not in itos] + sorted(set(words) - set(itos) - set(SPECIAL))
        v.stoi = {w: i for i, w in enumerate(v.itos)}
        return v

    def encode(self, ws: list[str]) -> list[int]:
        return [self.stoi[w] for w in ws if w in self.stoi]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.itos[i] for i in ids]


def load_extra(path: Path | None) -> list[str]:
    if not path:
        return []
    # utf-8-sig: a byte-order mark (some Windows tools write one) must not become part of a word
    return [ln.strip() for ln in path.open(encoding="utf-8-sig") if ln.strip()]


def drop_not_dense(lines: list[str]) -> tuple[list[str], list[str]]:
    """(kept, dropped): a line outside the dense alphabet would put stray words into the vocabulary."""
    from haishool.truth import is_dense  # late, like _relations(): the chat needs none of it
    kept, dropped = [], []
    for ln in lines:
        (kept if is_dense(ln) else dropped).append(ln)
    return kept, dropped


def drop_long(lines: list[str], ctx: int) -> tuple[list[str], list[str]]:
    """(kept, dropped): a line of more than ``ctx`` tokens never fits one window, so it is not trained."""
    kept, dropped = [], []
    for ln in lines:
        (dropped if len(tokens(ln)) > ctx else kept).append(ln)
    return kept, dropped


def split_extra(lines: list[str], holdout: float, seed: int) -> tuple[list[str], list[str]]:
    """Hold out whole hop pairs (``q x hop y`` and ``q x hops y`` go together) and a seeded
    fraction of the prompts of the round-5/6 questions; a held-out prompt leaves training
    entirely, with every line that asks it and every line that judges an answer to it.

    Record lines are always trained, and so are the other round-4 questions (relation, view,
    yesno): a held-out path is then made of links the model has seen, as in rounds 4 and 4b, and
    a run on a hops file alone splits exactly as it did then. A round-5/6 prompt is held out by
    :func:`_draw`, whatever the other lines are: adding a file never moves a question of an
    earlier run from the held-out side into training.
    """
    if holdout <= 0:
        return lines, []
    rng = random.Random(seed + 7)
    pairs = sorted({_hop_pair(ln) for ln in lines if _hop_kind(ln)})
    held_pairs = {p for p in pairs if rng.random() < holdout}
    train, held = [], []
    for ln in lines:
        if _hop_kind(ln):
            out = held if _hop_pair(ln) in held_pairs else train
        else:
            out = held if _is_new_question(ln) and _draw(seed, _held_key(ln)) < holdout else train
        out.append(ln)
    return train, held


def _is_new_question(line: str) -> bool:
    """A question outside the round-4 buckets: a round-5 fact or calculation, a round-6 rollout."""
    return line.startswith("q ") and extra_kind(line) not in CLASSIC_KINDS


def _draw(seed: int, key: str) -> float:
    """A number in [0, 1) fixed by the seed and the key alone (the same on every machine and run)."""
    return int.from_bytes(hashlib.sha256(f"{seed} {key}".encode()).digest()[:8], "big") / 2 ** 64


def _held_key(line: str) -> str:
    """Stable key shared by a question and its worked, check and judge variants."""
    from haishool.truth.splits import canonical_prompt
    return "q " + canonical_prompt(line)


def _prompt(line: str) -> str:
    """``q turkey borders. a greece syria.`` -> ``q turkey borders``."""
    return line.split(". a ")[0]


def _hop_kind(line: str) -> str:
    """``hop`` / ``hops`` for path queries (``q x hop y. a ...``), else ``""``."""
    w = line.split(".")[0].split()
    return w[2] if len(w) == 4 and w[0] == "q" and w[2] in ("hop", "hops") else ""


def _hop_pair(line: str) -> tuple[str, ...]:
    return tuple(line.split(".")[0].split()[1::2])


@functools.cache
def _relations() -> frozenset[str]:
    """The relation names of the round-4 hops files, whose question lines keep their old buckets.

    Imported late: the chat loads this module for ``load`` and ``generate`` and must not need
    what the graph code needs.
    """
    from haishool.relations import INVERSE
    return frozenset(INVERSE) | frozenset(INVERSE.values())


def verbs_of(lines: list[str], least: int = VERB_PROMPTS) -> frozenset[str]:
    """:data:`KIND_WORDS` plus the verbs a file shows by itself: every first word that starts
    ``least`` or more different questions of three or more words. A thing starts a handful
    (``q air part oxygen``, ``q iron plus oxygen gives``), a verb hundreds."""
    starts: dict[str, set[str]] = {}
    for ln in lines:
        words = _prompt(ln).split()
        if len(words) > 3 and words[0] == "q" and words[1] not in KIND_WORDS and extra_kind(ln) == "fact":
            found = starts.setdefault(words[1], set())
            if len(found) < least:
                found.add(_prompt(ln))
    return KIND_WORDS | {word for word, prompts in starts.items() if len(prompts) >= least}


def extra_kind(line: str, verbs: frozenset[str] = KIND_WORDS) -> str:
    """Report bucket of an extra query line (``""`` for a record line).

    Round 4 (the relation is a name from :mod:`haishool.relations`): ``hop`` / ``hops`` for path
    queries, ``yesno`` for ``q turkey borders greece``, ``view`` with ``according_to``, else
    ``relation``; a round-5 key that is also a relation name (``q water found_in``) lands here
    too. Round 5: a question of three or more words that starts with one of ``verbs``
    (:data:`KIND_WORDS`, or a file's :func:`verbs_of`; ``q calc 1 2 plus 7``) is its own bucket,
    named after the verb. Round 6: ``q gravity seed 7 ...`` and ``q chem valence c`` are the
    bucket of their simulation (:data:`SIMS`, or any name followed by ``seed`` and a number).
    Everything else is a ``fact``: a thing and one of its keys (``q carbon protons``, also
    ``q friction unit``), or a thing with a longer key (``q air part oxygen``).
    """
    if not line.startswith("q "):
        return ""
    if _hop_kind(line):
        return _hop_kind(line)
    words = _prompt(line).split()
    if len(words) > 2 and words[2] in _relations():  # before the verbs: ``force`` is also a record
        if len(words) == 4:
            return "yesno"  # q turkey borders greece. a yes.
        return "view" if "according_to" in words else "relation"
    if len(words) > 3 and (words[1] in verbs or words[1] in SIMS or (words[2] == "seed" and words[3].isdigit())):
        return words[1]
    return "fact"


def corpus(records: list[Record], passes: int, seed: int, extra: list[str] | None = None) -> list[str]:
    samples = [r.line() for r in records] + [q for r in records for q in r.queries()] + list(extra or [])
    rng = random.Random(seed)
    stream: list[str] = []
    for _ in range(passes):
        rng.shuffle(samples)
        for s in samples:
            stream.extend(tokens(s))
            stream.append(EOS)
    return stream


def _need_torch() -> None:
    if torch is None:
        raise RuntimeError("torch is not installed: training and loading a model need it (pip install torch)")


def corpus_ids(records: list[Record], passes: int, seed: int, vocab: Vocab, extra: list[str] | None = None) -> array:
    """:func:`corpus` encoded with ``vocab``: token for token the same stream, but every sample is
    split and encoded once and a token takes four bytes instead of a word in a Python list (the
    rollouts of round 6 make streams of hundreds of millions of tokens)."""
    samples = [r.line() for r in records] + [q for r in records for q in r.queries()] + list(extra or [])
    coded = [array("i", vocab.encode([*tokens(s), EOS])) for s in samples]
    rng = random.Random(seed)
    stream = array("i")
    for _ in range(passes):
        rng.shuffle(coded)  # the same permutation as corpus(): a shuffle depends on the length alone
        for ids in coded:
            stream.extend(ids)
    return stream


def generate(model: GPT, vocab: Vocab, prompt: str, *, stop: str = ".", max_new: int = 40, device: str = "cpu") -> str:
    # every training sample follows an <eos>; without it a describe prompt often ends at once
    ids = vocab.encode([EOS, *tokens(prompt)])
    if len(ids) < 2:
        return ""
    with torch.no_grad():
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
    return obj + ". " + generate(model, vocab, f"{obj}.", stop="", max_new=90, device=device)


def grown_state(old: dict, fresh: dict) -> dict:
    """``old`` weights laid over a ``fresh``, possibly larger, state of the same architecture.

    A tensor of the old shape is taken over as it is; one that grew in its first dimension
    (``wte.weight`` for new words, with the tied ``lm_head.weight``; ``wpe.weight`` for a longer
    context) keeps the old rows and the fresh initialisation of the new ones, written into
    ``fresh`` in place. Any other difference is an error, since the architecture must match.
    """
    out = {}
    if unknown := sorted(set(old) - set(fresh)):
        raise KeyError(f"checkpoint has {unknown[0]!r}, which this model lacks: not the same architecture")
    for key, new in fresh.items():
        if key not in old:
            raise KeyError(f"checkpoint has no {key!r}: not the same architecture")
        prev = old[key]
        if tuple(prev.shape) == tuple(new.shape):
            out[key] = prev
        elif tuple(prev.shape[1:]) == tuple(new.shape[1:]) and prev.shape[0] <= new.shape[0]:
            new[: prev.shape[0]] = prev
            out[key] = new
        else:
            raise ValueError(f"{key}: checkpoint shape {tuple(prev.shape)} does not fit {tuple(new.shape)}")
    return out


def train(records_path: Path, out: Path, holdout: float, seed: int, steps: int, device: str,
          extra_paths: list[Path] | Path | None = None, *, init: Path | None = None, lr: float | None = None,
          layers: int | None = None, width: int | None = None, heads: int | None = None, ctx: int | None = None,
          eval_sample: int = 1000, extra_eval: int = 500) -> dict:
    _need_torch()
    records = load_records(records_path)
    train_recs, held = split_pairs(records, holdout, seed)
    paths = [Path(p) for p in ([extra_paths] if isinstance(extra_paths, (str, Path)) else extra_paths or [])]
    ck = torch.load(init, map_location=device, weights_only=True) if init else None
    old_cfg = GPTConfig.from_json(ck["gpt_config"]) if ck else None
    if old_cfg:
        for name, want, have in (("layers", layers, old_cfg.n_layer), ("width", width, old_cfg.d_model), ("heads", heads, old_cfg.n_head)):
            if want is not None and want != have:
                raise ValueError(f"--{name} {want} does not match the checkpoint ({have}); a model keeps its shape")
        if ctx and ctx < old_cfg.ctx:
            raise ValueError(f"--ctx {ctx} is shorter than the checkpoint's ({old_cfg.ctx}); a context may grow, not shrink")
    ctx = ctx or (old_cfg.ctx if old_cfg else CTX)
    # every file's lines, minus those outside the dense alphabet and those too long for one window
    files = []
    for path in paths:
        dense, not_dense = drop_not_dense(load_extra(path))
        kept, long = drop_long(dense, ctx)
        files.append((kept, long, not_dense))
        if long or not_dense:
            print(f"{path}: dropped {len(long)} lines longer than {ctx} tokens and {len(not_dense)} not dense, "
                  f"e.g. {(long + not_dense)[0][:120]!r}", file=sys.stderr, flush=True)
    extra = [ln for kept, _, _ in files for ln in kept]
    extra_train, extra_held = split_extra(extra, holdout, seed)
    words = {w for r in records for w in tokens(r.line())} | {w for r in records for q in r.queries() for w in tokens(q)}
    for ln in extra:
        words.update(tokens(ln))
    vocab = Vocab.grown(ck["itos"], sorted(words)) if ck else Vocab(sorted(words))
    ids = corpus_ids(train_recs, PASSES, seed, vocab, extra_train)
    stream = torch.frombuffer(ids, dtype=torch.int32) if ids.itemsize == 4 else torch.tensor(ids.tolist(), dtype=torch.int32)
    if old_cfg:
        cfg = GPTConfig.from_json({**ck["gpt_config"], "vocab_size": max(old_cfg.vocab_size, len(vocab.itos)), "ctx": ctx})
    else:
        cfg = GPTConfig(vocab_size=max(32, len(vocab.itos)), n_layer=layers or LAYERS, n_head=heads or HEADS,
                        d_model=width or WIDTH, ctx=ctx, dropout=0.1)
    torch.manual_seed(seed)
    model = GPT(cfg).to(device)
    if ck:
        old_state = {k: v.float() for k, v in ck["model_state"].items()}
        model.load_state_dict(grown_state(old_state, {k: v.clone() for k, v in model.state_dict().items()}))
    peak_lr = lr if lr is not None else (3e-4 if ck else 1e-3)
    opt = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.1)
    batch = 128
    t0 = time.monotonic()
    log = []
    for step in range(1, steps + 1):
        lr_now = peak_lr * min(1.0, step / WARMUP) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps)))
        for g in opt.param_groups:
            g["lr"] = lr_now
        idx = torch.randint(0, len(stream) - cfg.ctx - 1, (batch,))
        x = torch.stack([stream[i:i + cfg.ctx] for i in idx]).long().to(device)
        y = torch.stack([stream[i + 1:i + cfg.ctx + 1] for i in idx]).long().to(device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 250 == 0 or step == steps:
            log.append({"step": step, "loss": round(loss.item(), 4), "seconds": round(time.monotonic() - t0, 1)})
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

    trained_pairs = [(r.obj, k, r.values[k]) for r in train_recs for k in QUERY_KEYS if k in r.values]
    rng = random.Random(seed + 1)
    sample = rng.sample(trained_pairs, min(eval_sample, len(trained_pairs)))
    report = {"records": len(records), "vocab": len(vocab.itos), "params": model.num_params(),
              "train_tokens_per_pass": len(stream) // PASSES, "steps": steps, "holdout": holdout,
              "recall_trained_pairs": acc(sample), "inference_heldout_pairs": acc(held), "log": log}
    if ck:
        report["init"] = {"checkpoint": str(init), "vocab_old": len(ck["itos"]), "new_words": len(vocab.itos) - len(ck["itos"])}
    report["lr"] = peak_lr
    report["config"] = cfg.to_json()
    report["dropped_long_lines"] = sum(len(long) for _, long, _ in files)
    report["dropped_not_dense_lines"] = sum(len(not_dense) for _, _, not_dense in files)
    if paths:
        def acc_lines(lines: list[str], limit: int, verbs: frozenset[str]) -> dict:
            by_kind: dict[str, list[str]] = {kind: [] for kind in CLASSIC_KINDS}
            for ln in lines:
                if kind := extra_kind(ln, verbs):
                    by_kind.setdefault(kind, []).append(ln)
            out_: dict[str, dict] = {}
            for kind in CLASSIC_KINDS + tuple(sorted(set(by_kind) - set(CLASSIC_KINDS))):
                some = random.Random(seed + 2).sample(by_kind[kind], min(limit, len(by_kind[kind])))
                exact = 0
                for ln in some:
                    prompt, _, gold = ln.partition(". a ")
                    gold = gold.rstrip(".")
                    # room for the whole answer and its final ".": some round-5 answers pass 40 tokens
                    room = max(40, len(gold.split()) + 1)
                    exact += generate(model, vocab, prompt + ". a", max_new=room, device=device) == gold
                out_[kind] = {"n": len(some), "exact": round(exact / len(some), 4) if some else None}
            return out_

        held_set = set(extra_held)
        report["extra_files"] = []
        for path, (kept, long, not_dense) in zip(paths, files):
            file_train = [ln for ln in kept if ln not in held_set]
            file_held = [ln for ln in kept if ln in held_set]
            verbs = verbs_of(kept)
            report["extra_files"].append({
                "file": str(path), "lines": len(kept) + len(long) + len(not_dense), "trained": len(file_train),
                "held_out": len(file_held), "dropped_long_lines": len(long), "dropped_not_dense_lines": len(not_dense),
                "recall_trained": acc_lines(file_train, extra_eval, verbs),
                "inference_heldout": acc_lines(file_held, extra_eval, verbs)})
        report["extra_lines"] = report["extra_files"][0]
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "gpt_config": cfg.to_json(), "itos": vocab.itos}, out / "student.pt")
    (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "log"}), flush=True)
    return report


def load(path: Path, device: str = "cpu") -> tuple[GPT, Vocab]:
    _need_torch()
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
    t.add_argument("--device", default="cuda" if torch is not None and torch.cuda.is_available() else "cpu")
    t.add_argument("--extra-lines", type=Path, action="append",
                   help="ready-made dense lines, e.g. data/hops-v4b/hops-train-r4.txt; may be repeated")
    t.add_argument("--init", type=Path, help="continue from this checkpoint (student.pt) instead of from scratch")
    t.add_argument("--lr", type=float, help="peak learning rate (default 1e-3, or 3e-4 with --init)")
    t.add_argument("--layers", type=int, help=f"transformer blocks (default {LAYERS}; from the checkpoint with --init)")
    t.add_argument("--width", type=int, help=f"model width (default {WIDTH}; from the checkpoint with --init)")
    t.add_argument("--heads", type=int, help=f"attention heads (default {HEADS}; from the checkpoint with --init)")
    t.add_argument("--ctx", type=int, help=f"context in tokens (default {CTX}; from the checkpoint with --init, may grow)")
    t.add_argument("--eval-sample", type=int, default=1000, help="trained (object, key) pairs sampled for recall")
    t.add_argument("--extra-eval", type=int, default=500, help="extra lines sampled per kind for the report")
    args = ap.parse_args()
    train(args.records, args.out, args.holdout, args.seed, args.steps, args.device, args.extra_lines,
          init=args.init, lr=args.lr, layers=args.layers, width=args.width, heads=args.heads, ctx=args.ctx,
          eval_sample=args.eval_sample, extra_eval=args.extra_eval)


if __name__ == "__main__":
    main()
