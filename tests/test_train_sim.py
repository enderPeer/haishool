"""Training on simulated worlds only: train_final's mix "sim" and the round-7 run of train_sim."""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest

from haishool import curriculum7, final_run, student, train_final, train_sim
from haishool.evo import LEVELS
from haishool.truth import split_line

ROUND7 = [f"predict_{n}" for n in LEVELS] + [f"predict_{n}_rules7" for n in curriculum7.OLD] + \
         [f"legacy_{n}" for n in LEVELS] + [f"legacy_{n}_rules7" for n in curriculum7.OLD]


# ---------------------------------------------------------------------------------------------
# fixtures


def final_fixture(tmp_path):
    """The final-format fixture of tests/test_train_final.py: old facts, hops, truth and cosmos."""
    data = tmp_path / "curriculum"
    data.mkdir()
    records = tmp_path / "records.jsonl"
    records.write_text(json.dumps({"obj": "apple", "values": {"kind": "fruit", "color": "red"}}) + "\n",
                       encoding="utf-8")
    hops = tmp_path / "hops.txt"
    hops.write_text("q apple borders pear. a yes.\nq apple hop pear. a apple borders pear.\n"
                    "q apple hops pear. a 1.\n", encoding="utf-8")
    report = {"seed": 20261001, "topics": {}}
    for topic in [*train_final.TRUTH_SHARES, "predict_life", "legacy_life"]:
        path = data / f"{topic}-train.txt"
        path.write_text(f"q {topic} input 1. a 2.\n", encoding="utf-8")
        report["topics"][topic] = {"splits": {"train": {"sha256": train_final.digest(path)}}}
        (data / f"{topic}-dev.txt").write_text("q secret_dev. a secret_dev.\n", encoding="utf-8")
        (data / f"{topic}-sealed.txt").write_text("q secret_sealed. a secret_sealed.\n", encoding="utf-8")
    (data / "report.json").write_text(json.dumps(report), encoding="utf-8")
    return data, records, hops


def sim_fixture(tmp_path, topics=None):
    """A simulation-only data directory: three lesson topics and two story topics of round 7."""
    data = tmp_path / "sim"
    data.mkdir()
    topics = topics or {"predict_stars": 3, "predict_cells": 1, "predict_world7": 2,
                        "legacy_stars": 2, "legacy_nucleo_rules7": 1}
    report = {"seed": 20261001, "topics": {}}
    for topic, n in topics.items():
        path = data / f"{topic}-train.txt"
        path.write_text("".join(f"q {topic} item {i}. a {i}.\n" for i in range(n)), encoding="utf-8")
        report["topics"][topic] = {"splits": {"train": {"sha256": train_final.digest(path)}}}
        (data / f"{topic}-dev.txt").write_text("q secret_dev. a secret_dev.\n", encoding="utf-8")
        (data / f"{topic}-sealed.txt").write_text("q secret_sealed. a secret_sealed.\n", encoding="utf-8")
    (data / "report.json").write_text(json.dumps(report), encoding="utf-8")
    return data


def prepare_sim(tmp_path, **topics):
    data = sim_fixture(tmp_path, topics or None)
    report, paths, hashes = train_final.input_manifest(data, None, None, None, None, "sim")
    out = tmp_path / "run"
    out.mkdir()
    meta = train_final.prepare(data, out, None, None, holdout=0., seed=20261001, ctx=32,
                               paths=paths, report=report, mix="sim")
    return data, out, meta, hashes


# ---------------------------------------------------------------------------------------------
# train_final: mix "sim"


def test_sim_mixture_gives_lessons_80_and_stories_20_split_equally_by_topic(tmp_path):
    _, out, meta, hashes = prepare_sim(tmp_path)
    weights = {s["topic"]: s["weight"] for s in meta["sources"]}
    assert weights == pytest.approx({"predict_stars": .8 / 3, "predict_cells": .8 / 3, "predict_world7": .8 / 3,
                                     "legacy_stars": .1, "legacy_nucleo_rules7": .1})
    totals = defaultdict(float)
    for source in meta["sources"]:
        totals[train_final.category(source["topic"])] += source["weight"]
        assert (out / "cache" / source["bin"]).stat().st_size == 4 * source["tokens"]
    assert dict(totals) == pytest.approx({"predict": .8, "legacy": .2})
    # no old facts or hops: no source, no file, nothing hashed
    assert not any(s["topic"].startswith("old_") for s in meta["sources"])
    assert not (out / "old-recall.txt").exists() and not (out / "old-heldout.txt").exists()
    assert not list((out / "cache").glob("old_*.txt"))
    assert all(Path(p).parent.name == "sim" for p in hashes)
    assert "secret_dev" not in meta["itos"] and "secret_sealed" not in meta["itos"]


def test_sim_mixture_skips_empty_topics_and_splits_by_tokens_within_a_topic():
    sources = [train_final.Source("a", "predict_stars", Path("a"), count=30),
               train_final.Source("b", "predict_cells", Path("b"), count=10),
               train_final.Source("c", "predict_bodies", Path("c"), count=0),
               train_final.Source("d", "legacy_world7", Path("d"), count=3),
               train_final.Source("e", "legacy_world7", Path("e"), count=1)]
    train_final.mixture(sources, "sim")
    assert [s.weight for s in sources] == pytest.approx([.4, .4, 0., .15, .05])


def test_sim_mixture_requires_both_groups():
    with pytest.raises(ValueError, match="predict and legacy"):
        train_final.mixture([train_final.Source("a", "predict_stars", Path("a"), count=5)], "sim")


@pytest.mark.parametrize("topic", ["maths", "old_facts", "old_hops", "forces"])
def test_sim_rejects_a_topic_that_is_not_a_lesson_or_story(tmp_path, topic):
    data = sim_fixture(tmp_path, {"predict_stars": 1, "legacy_stars": 1, topic: 1})
    with pytest.raises(ValueError, match="only predict_\\* and legacy_\\*"):
        train_final.input_manifest(data, None, None, None, None, "sim")
    sources = [train_final.Source("a", "predict_stars", Path("a"), count=5),
               train_final.Source("b", "legacy_stars", Path("b"), count=5),
               train_final.Source("c", topic, Path("c"), count=5)]
    with pytest.raises(ValueError, match="only predict_\\* and legacy_\\*"):
        train_final.mixture(sources, "sim")


def test_sim_rejects_unknown_topics_feedback_and_mixes(tmp_path):
    data = sim_fixture(tmp_path, {"predict_stars": 1, "legacy_stars": 1, "gossip": 1})
    with pytest.raises(ValueError, match="unrecognized training topic"):
        train_final.input_manifest(data, None, None, None, None, "sim")
    feedback = tmp_path / "feedback.txt"
    feedback.write_text("q stars predict x. a 1.\n", encoding="utf-8")
    (tmp_path / "b").mkdir()
    with pytest.raises(ValueError, match="feedback"):
        train_final.input_manifest(sim_fixture(tmp_path / "b"), None, None, None, feedback, "sim")
    with pytest.raises(ValueError, match="unknown mix"):
        train_final.mixture([], "both")


#: prepare() of final_fixture with the code before the mix option (haishool/train_final.py of the
#: round-5 final run, ~/haishool-final-20261001 on adler): every file it writes, by sha256
FINAL_FILES = {
    "cache/old_facts.txt": "48aa7cf1a4db99ea12435b93e5aeb6f7308e12fcab73c25107d828c6beb4a245",
    "cache/old_hops.txt": "d2d344f3ab2f6e56d72248eef38bb4fadc0ff4b6e1d419d514681c529a7f5160",
    "cache/prepared.json": "4b60483431cec02d9beceb01a6463f43413362c9b8056a8f326f96c9beb296ae",
    "cache/source-00.u32": "333679c0f4c88e322585713964cb42cc9aa995016dce8bf064f8e8d04357959d",
    "cache/source-01.u32": "f6a2c93ed3c2949196a64af9af59000bdd245ac46089f15634b160488b4346f3",
    "cache/source-02.u32": "6bc3481f3f38534e40d29e62a422e4be72acac8bc96cf433cd1c347dc1a57a7d",
    "cache/source-03.u32": "626b2e1cc08b41585e358703c9dd2c47876cf49ef157c234198d06304d8739ef",
    "cache/source-04.u32": "9acb957d3faef3cadb9d46153dee9ea031c4c44c77aeab9d3d572f7e41091a73",
    "cache/source-05.u32": "c4ae90c6590572c619cde83bf76e85c5ea0c376869856e2abdba70cb93946c0f",
    "cache/source-06.u32": "3a4422c4ddb211dd7aa89e0e8670d77a4de807c997dcdce9599895310fc69f3d",
    "cache/source-07.u32": "97007db89b891c6a83fcc736c57d8ec03751832501c4d120bafc78ccd7b986d1",
    "cache/source-08.u32": "d0d8b2e7d404d7ca7e2e3aff7f53a260c4e28f645a2934bf53e1a3400fa63377",
    "old-heldout.txt": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "old-recall.txt": "17bfee40be8d6310c514a510e3503d49d75356d4ef8f4d8216f2be34b45d41df",
}
FINAL_WEIGHTS = {"old_facts": 0.18518518518518517, "old_hops": 0.21481481481481485, "elements": .04,
                 "forces": .08, "legacy_life": .05, "maths": .2, "predict_life": .15, "reactions": .04,
                 "substances": .04}


@pytest.mark.parametrize("explicit", [False, True])
def test_final_mix_prepares_exactly_what_the_code_before_the_option_did(tmp_path, explicit):
    data, records, hops = final_fixture(tmp_path)
    mix = {"mix": "final"} if explicit else {}
    report, paths, hashes = train_final.input_manifest(data, records, hops, None, None, **mix)
    assert {Path(p).name for p in hashes} == {"report.json", "records.jsonl", "hops.txt"} | \
        {p.name for p in data.glob("*-train.txt")}
    out = tmp_path / "run"
    out.mkdir()
    meta = train_final.prepare(data, out, records, hops, holdout=.1, seed=20261001, ctx=32,
                               paths=paths, report=report, **mix)
    files = {p.relative_to(out).as_posix(): train_final.digest(p) for p in sorted(out.rglob("*")) if p.is_file()}
    assert files == FINAL_FILES
    assert {s["topic"]: s["weight"] for s in meta["sources"]} == pytest.approx(FINAL_WEIGHTS, rel=1e-12)


def test_final_mix_still_requires_old_facts_and_truth(tmp_path):
    sources = [train_final.Source("a", "predict_stars", Path("a"), count=5),
               train_final.Source("b", "legacy_stars", Path("b"), count=5)]
    with pytest.raises(ValueError, match="old, truth"):
        train_final.mixture(sources)


# ---------------------------------------------------------------------------------------------
# train_sim: gates


@pytest.mark.parametrize("topic", ROUND7)
def test_gate_lookup_for_every_round7_topic(topic):
    gate = train_sim.gate_for(topic)
    name = topic.split("_", 1)[1]
    if topic.startswith("predict_"):
        assert gate is curriculum7.lesson_gates()[topic]
        lines = gate.generate(random.Random(7), 20)
    else:
        level = name.removesuffix("_rules7")
        package = "cosmos" if name.endswith("_rules7") else "evo"
        assert type(gate).__module__ == f"haishool.{package}.{level}"
        # the evaluation's gate accepts what the builder's check accepted, rules 7 included
        rollout, lines_of, check, _, _ = curriculum7._level(topic)
        lines = [ln for ln in lines_of(rollout(1)) if ln.kind != "record"][:40]
        assert lines and all(check(ln.prompt, ln.answer).ok for ln in lines)
        if name.endswith("_rules7"):
            assert all(" rules 7 " in ln.prompt for ln in lines)
    assert all(gate.check(ln.prompt, ln.answer).ok for ln in lines)
    assert not all(gate.check(ln.prompt, "9 9 9 9 9 9").ok for ln in lines)


def test_round7_topics_are_exactly_what_the_builder_writes():
    assert sorted(ROUND7) == sorted(list(curriculum7.lesson_gates()) + curriculum7.story_topics())


@pytest.mark.parametrize("topic", ["maths", "predict_world", "legacy_world", "legacy_world7_rules7",
                                   "legacy_cells_rules7", "predict_gossip", "old_facts"])
def test_gate_lookup_rejects_topics_outside_round7(topic):
    with pytest.raises(ValueError):
        train_sim.gate_for(topic)


def test_recorder_pairs_score_lines_gold_and_answer(monkeypatch, tmp_path):
    gate = curriculum7.lesson_gates()["predict_stars"]
    lines = gate.generate(random.Random(3), 6)
    path = tmp_path / "predict_stars-dev.txt"
    path.write_text("".join(ln.text + "\n" for ln in lines), encoding="utf-8")
    answers = [ln.answer if i % 2 else "0" for i, ln in enumerate(lines)]
    monkeypatch.setattr(final_run, "ask_many", lambda model, vocab, prompts, device: list(answers))
    recorder = train_sim.Recorder(gate)
    result = final_run.score_lines(None, None, path, recorder, "cpu")
    judged = recorder.judged()
    assert [(p, g, a) for p, g, a, _ in judged] == [(ln.prompt, ln.answer, a) for ln, a in zip(lines, answers)]
    assert sum(ok for *_, ok in judged) == pytest.approx(result["scores"]["all"]["gate_ok"] * len(lines))


# ---------------------------------------------------------------------------------------------
# train_sim: evaluation without a model


def eval_fixture(tmp_path):
    """A data directory of one lesson topic (real gate) and legacy_world7 (judged by exact answers)."""
    data = tmp_path / "data"
    data.mkdir()
    gate = curriculum7.lesson_gates()["predict_stars"]
    lessons = gate.generate(random.Random(5), 9)
    world = {
        "train": ["q world7 seed 1 1 say predator near. a ka mi.", "q world7 seed 1 1 meaning ka mi. a predator near.",
                  "q world7 seed 1 1 era star age. a 3.", "q world7 seed 1 1 era replicators_1 age. a 2."],
        "dev": ["q world7 seed 8 5 say predator small. a me pa.", "q world7 seed 8 5 meaning me pa. a predator small.",
                "q world7 seed 8 5 word fire. a ka.", "q world7 seed 8 5 era star age. a 1.",
                "q world7 span cells. a 3.", "q world7 seed 8 5 era replicators_9 age. a 1."],
    }
    world["sealed"] = world["dev"][:4]
    report = {"seed": 20261001, "round": 7, "topics": {}}
    files = {"predict_stars": {"train": lessons[:3], "dev": lessons[3:6], "sealed": lessons[6:]},
             "legacy_world7": world}
    for topic, splits in files.items():
        report["topics"][topic] = {"splits": {}}
        for split, rows in splits.items():
            path = data / f"{topic}-{split}.txt"
            path.write_text("".join((r if isinstance(r, str) else r.text) + "\n" for r in rows), encoding="utf-8")
            report["topics"][topic]["splits"][split] = {"sha256": train_final.digest(path)}
    (data / "report.json").write_text(json.dumps(report), encoding="utf-8")
    words = {w for p in data.glob("*-train.txt") for t in p.read_text().splitlines() for w in student.tokens(t)}
    words |= {w for p in data.glob("*.txt") for t in p.read_text().splitlines() for w in student.tokens(t)}
    vocab = student.Vocab(sorted(words - {"replicators_9"}))
    golds = dict(pa for f in data.glob("*.txt") for t in f.read_text().splitlines() if (pa := split_line(t)))
    return data, vocab, golds


def test_evaluate_scores_lessons_stories_language_and_unscorable_lines(monkeypatch, tmp_path):
    data, vocab, golds = eval_fixture(tmp_path)
    wrong = {"world7 seed 8 5 meaning me pa", "world7 seed 8 5 era star age"}
    def ask(model, v, prompts, device):
        assert all("replicators_9" not in p for p in prompts)  # never asked: ask_many would refuse it
        return ["0" if p in wrong else golds[p] for p in prompts]
    real = train_sim.gate_for
    monkeypatch.setattr(train_sim, "gate_for", lambda t: train_sim.Exact(golds) if t.startswith("legacy_") else real(t))
    monkeypatch.setattr(final_run, "ask_many", ask)
    monkeypatch.setattr(student, "load", lambda checkpoint, device: (None, vocab))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(set_num_threads=lambda n: None))
    checkpoint = tmp_path / "student.pt"
    checkpoint.write_bytes(b"a checkpoint")
    root = tmp_path / "runs" / "sim7"
    dev = train_sim.evaluate(checkpoint, data, "dev", root / "dev.json", device="cpu")
    lessons, world = dev["topics"]["predict_stars"], dev["topics"]["legacy_world7"]
    assert lessons["all_lines"] == {"n": 3, "right": 3, "exact": 3, "unscorable": 0, "gate_ok": 1.}
    assert world["scores"]["all"]["n"] == 5  # final_run.score_lines over the lines that can be asked
    assert world["all_lines"]["n"] == 6 and world["all_lines"]["right"] == 3 and world["all_lines"]["unscorable"] == 1
    assert world["unscorable"]["words"] == {"replicators_9": 1}
    assert world["seeded"] == {"n": 5, "right": 2} and world["seedless"] == {"n": 1, "right": 1}
    assert world["language"] == {"meaning": {"n": 1, "right": 0}, "say": {"n": 1, "right": 1},
                                 "word": {"n": 1, "right": 1}}
    assert dev["selection_score"] == 1.
    # one evaluation per split and checkpoint: same inputs return the report, another checkpoint is refused
    assert train_sim.evaluate(checkpoint, data, "dev", root / "dev.json", device="cpu") == dev
    other = tmp_path / "other.pt"
    other.write_bytes(b"another checkpoint")
    with pytest.raises(ValueError, match="different inputs"):
        train_sim.evaluate(other, data, "dev", root / "dev.json", device="cpu")
    with pytest.raises(ValueError, match="dev or sealed"):
        train_sim.evaluate(checkpoint, data, "train", root / "train.json", device="cpu")
    train_sim.evaluate(checkpoint, data, "sealed", root / "sealed.json", device="cpu")
    monkeypatch.setattr(train_sim, "story_gate", lambda t: train_sim.Exact(golds))
    recalled = train_sim.recall(checkpoint, data, root / "train-recall.json", device="cpu", limit=2)
    assert set(recalled["topics"]) == {"legacy_world7"}
    assert recalled["topics"]["legacy_world7"]["all_lines"]["n"] == 2
    assert recalled["world7_language"]["language"] == {"meaning": {"n": 1, "right": 1}, "say": {"n": 1, "right": 1}}
    settings = {"name": "sim7", "layers": 8, "width": 512, "heads": 8, "ctx": 256, "steps": 10}
    summary = train_sim.summarize(root, checkpoint, settings)
    assert summary["dev"]["lessons"]["micro_gate_ok"] == 1.
    assert summary["dev"]["stories"]["n"] == 6 and summary["dev"]["stories"]["unscorable"] == 1
    assert summary["dev"]["stories"]["seeded"] == {"n": 5, "right": 2, "share": .4}
    assert summary["dev"]["world7_language"] == {"say": {"n": 1, "right": 1}, "meaning": {"n": 1, "right": 0},
                                                 "word": {"n": 1, "right": 1}}
    assert summary["sealed"]["world7_language"]["meaning"] == {"n": 1, "right": 0}
    assert summary["train_recall"]["world7_language"]["say"] == {"n": 1, "right": 1}
    assert json.loads((root / "summary.json").read_text()) == summary


def test_check_golds_accepts_round7_gold_and_stops_on_a_rejected_one(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    gate = curriculum7.lesson_gates()["predict_cells"]
    lines = gate.generate(random.Random(11), 4)
    report = {"seed": 20261001, "topics": {"predict_cells": {"splits": {}}}}
    for split, rows in (("dev", lines[:2]), ("sealed", lines[2:])):
        path = data / f"predict_cells-{split}.txt"
        path.write_text("".join(ln.text + "\n" for ln in rows), encoding="utf-8")
        report["topics"]["predict_cells"]["splits"][split] = {"sha256": train_final.digest(path)}
    (data / "report.json").write_text(json.dumps(report), encoding="utf-8")
    checked = train_sim.check_golds(data, tmp_path / "gates.json", workers=1)
    assert checked["ok"] and checked["files"]["predict_cells-dev.txt"]["lines"] == 2
    bad = data / "predict_cells-sealed.txt"
    bad.write_text(f"q {lines[2].prompt}. a 9 9 9 9 9 9.\n", encoding="utf-8")
    report["topics"]["predict_cells"]["splits"]["sealed"]["sha256"] = train_final.digest(bad)
    (data / "report.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="reject dataset gold"):
        train_sim.check_golds(data, tmp_path / "gates.json", workers=1)


# ---------------------------------------------------------------------------------------------
# with torch: a tiny training run of each mix, and the float16 export


def test_tiny_cpu_sim_training_resume_and_fp16_export(tmp_path):
    torch = pytest.importorskip("torch")
    data = sim_fixture(tmp_path)
    out = tmp_path / "run"
    args = dict(data=data, out=out, layers=1, width=16, heads=2, ctx=32, steps=2, batch=2, device="cpu",
                checkpoint_every=1, holdout=0., mix="sim")
    # records and hops are not required, and ignored when given
    result = train_final.train(**args, records=tmp_path / "missing.jsonl", hops=tmp_path / "missing.txt")
    assert result["mix"] == "sim" and result["run_config"]["mix"] == "sim"
    assert not any("missing" in p for p in result["run_config"]["sources"])
    assert {s["topic"] for s in result["source_mixture"]} == {"predict_stars", "predict_cells", "predict_world7",
                                                               "legacy_stars", "legacy_nucleo_rules7"}
    assert not (out / "old-recall.txt").exists()
    first, _ = student.load(out / "student.pt")
    train_final.train(**args, resume=out / "initialized.pt")
    again, _ = student.load(out / "student.pt")
    for key, tensor in first.state_dict().items():
        assert torch.equal(tensor, again.state_dict()[key])
    (tmp_path / "r.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "h.txt").write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        train_final.train(**{**args, "mix": "final"}, records=tmp_path / "r.jsonl", hops=tmp_path / "h.txt",
                          resume=out / "latest.pt")
    probe = sorted(data.glob("predict_*-train.txt"))[0]  # export probes the first lesson file's lines
    target = tmp_path / "export" / "sim-fp16.pt"
    manifest = train_sim.export(out / "student.pt", target, data, device="cpu")
    saved = torch.load(target, weights_only=True)
    assert set(saved) == {"model_state", "gpt_config", "itos"}
    assert all(v.dtype == torch.float16 for v in saved["model_state"].values() if v.is_floating_point())
    model, vocab = student.load(target)
    assert vocab.itos == torch.load(out / "student.pt", weights_only=True)["itos"]
    assert manifest["dtype"] == "float16" and len(manifest["probe"]) == len(probe.read_text().splitlines())
    assert manifest["max_abs_error_vs_float32"] < 1e-2


def test_final_mix_training_config_has_no_mix_key(tmp_path):
    pytest.importorskip("torch")
    data, records, hops = final_fixture(tmp_path)
    result = train_final.train(data=data, out=tmp_path / "run", records=records, hops=hops, layers=1, width=16,
                               heads=2, ctx=32, steps=1, batch=1, device="cpu", checkpoint_every=1)
    assert "mix" not in result and "mix" not in result["run_config"]
    assert {Path(p).name for p in result["run_config"]["sources"]} >= {"records.jsonl", "hops.txt"}
    assert (tmp_path / "run" / "old-recall.txt").exists()
